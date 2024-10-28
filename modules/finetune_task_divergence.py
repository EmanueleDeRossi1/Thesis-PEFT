import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
from divergences.mkmmd import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from torchmetrics import Accuracy, F1Score
import os


class FineTuneTaskDivergence(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.save_hyperparameters(hparams)
        self.base_model_name = self.hparams['pretrained_model_name']
        self.num_classes = self.hparams['num_classes']
        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, token=hf_token)
        # Instead of using LoRA, finetune the full model
        self.model = base_model
        self.criterion = CrossEntropyLoss()
        # Initialize F1 and Accuracy measures
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)  # accuracy
        self.f1 = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')  # F1
        # Initialize MK-MMD with Gaussian Kernels
        self.kernels = [GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.0), GaussianKernel(alpha=2.0)]
        self.mk_mmd_loss = MultipleKernelMaximumMeanDiscrepancy(self.kernels, linear=False)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Forward pass
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        cls_token = outputs.hidden_states[-1][:, 0, :]
        return cls_token, outputs.logits

    def compute_loss_and_metrics(self, stage, cls_token, logits, source_labels, target_labels=None, is_target_batch=False, batch_idx=None):
        batch_size = source_labels.shape[0]
        # this needs to be changed: 
        # in training i dont have target_labels, but i still need to calculate the divergence loss
        # but i cannot calculate the task loss of the target
        if stage=="train":
            if is_target_batch:
                cls_token_source, cls_token_target = cls_token[:batch_size], cls_token[batch_size:]
                logits_source, logits_target = logits[:batch_size], logits[batch_size:]


                # Calculate divergence and task loss
                divergence = self.mk_mmd_loss(cls_token_source, cls_token_target)
                task_loss = self.criterion(logits_source, source_labels)
                
                # calculate alpha
                start_steps = self.current_epoch * batch_size
                total_steps = self.hparams["n_epochs"] * batch_size
                p = float(batch_idx + start_steps) / total_steps
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
                # calculate loss as trade off between task and divergence loss
                loss = alpha * task_loss + (1 - alpha) * divergence
            else:
                cls_token_source, logits_source = cls_token[batch_size], logits[batch_size]
                task_loss = self.criterion(logits_source, source_labels)
                loss = task_loss
                divergence=None
        else:
            cls_token_source, cls_token_target = cls_token[:batch_size], cls_token[batch_size:]
            logits_source, logits_target = logits[:batch_size], logits[batch_size:]

            # Calculate divergence and task loss
            divergence = self.mk_mmd_loss(cls_token_source, cls_token_target)
            task_loss = self.criterion(logits_source, source_labels)
            loss = 0.5 * task_loss + 0.5 * divergence


        # Calculate metrics for source
        preds_source = torch.argmax(logits_source, dim=1)
        source_accuracy = self.accuracy(source_labels, preds_source)
        source_f1 = self.f1(source_labels, preds_source)

        # Log metrics
        log_dict = {f"source_{stage}/accuracy": source_accuracy, f"source_{stage}/f1": source_f1}
        if divergence is not None:
            log_dict[f"{stage}/divergence"] = divergence  # Only log divergence if it exists

        # Calculate metrics for target
        if target_labels is not None:
            preds_target = torch.argmax(logits_target, dim=1)
            target_accuracy = self.accuracy(target_labels, preds_target)
            target_f1 = self.f1(target_labels, preds_target)
            log_dict.update({f"target_{stage}/accuracy": target_accuracy, f"target_{stage}/f1": target_f1,})

        self.log_dict(log_dict, on_step=False, on_epoch=True)


        return loss

    def training_step(self, batch, batch_idx):
        source_batch = batch[0]["source"]
        target_batch = batch[1]["source"] if len(batch) > 1 else None

        # Prepare input tensors for source and target
        source_input_ids, source_attention_mask = source_batch["input_ids"], source_batch["attention_mask"]
        input_ids, attention_mask = source_input_ids, source_attention_mask
        token_type_ids = source_batch.get("token_type_ids", None)
        source_labels = source_batch["label"]

        if target_batch:
            # Concatenate source and target data if target_batch is present
            target_input_ids, target_attention_mask = target_batch["input_ids"], target_batch["attention_mask"]
            input_ids = torch.cat((input_ids, target_input_ids), dim=0)
            attention_mask = torch.cat((attention_mask, target_attention_mask), dim=0)
            if token_type_ids is not None:
                token_type_ids = torch.cat((token_type_ids, target_batch["token_type_ids"]), dim=0)
            target_labels = target_batch["label"]

        # Single forward pass for combined source and target inputs
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.compute_loss_and_metrics("train", cls_token, logits, source_labels, target_labels=target_labels if target_labels is not None else None, is_target_batch=bool(target_batch), batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, stage="validation")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, stage="test")

    def _eval_step(self, batch, stage):
        source_input_ids = batch["source"]["input_ids"]
        source_attention_mask = batch["source"]["attention_mask"]
        source_token_type_ids = batch["source"].get("token_type_ids", None)
        source_labels = batch["source"]["label"]

        target_input_ids = batch["target"]["input_ids"]
        target_attention_mask = batch["target"]["attention_mask"]
        target_token_type_ids = batch["target"].get("token_type_ids", None)
        target_labels = batch["target"]["label"]

        input_ids = torch.cat((source_input_ids, target_input_ids), dim=0)
        attention_mask = torch.cat((source_attention_mask, target_attention_mask), dim=0)
        token_type_ids = None  # Initialize to None
        if source_token_type_ids is not None and target_token_type_ids is not None:
            token_type_ids = torch.cat((source_token_type_ids, target_token_type_ids), dim=0)

        # Single forward pass for combined source and target data
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Calculate loss and metrics
        loss = self.compute_loss_and_metrics(stage, cls_token, logits, source_labels, target_labels=target_labels)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(self.model.parameters(), lr=float(self.hparams['learning_rate']))
        # , weight_decay=float(self.hparams['weight_decay']))
        total_steps= self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
        return [optimizer], [scheduler]