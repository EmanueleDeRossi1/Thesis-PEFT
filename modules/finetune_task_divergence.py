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

    def compute_loss_and_metrics(self, cls_token, logits, source_labels, target_labels=None):
        batch_size = source_labels.shape[0]
        if target_labels is not None:
            cls_token_source, cls_token_target = cls_token[:batch_size], cls_token[batch_size:]
            logits_source, logits_target = logits[:batch_size], logits[batch_size:]
            print("LOGITS TARGET if: ", logits_target)
            print("TARGET LABELS INSIDE COMPUTE if: ", target_labels)


            # Calculate divergence and task loss
            divergence = self.mk_mmd_loss(cls_token_source, cls_token_target)
            task_loss = self.criterion(logits_source, source_labels) + self.criterion(logits_target, target_labels)
            loss = 0.5 * task_loss + 0.5 * divergence
        else:
            # sarà che si devo fare [:batch_size] cosi?? 
            cls_token_source, logits_source = cls_token[:batch_size], logits[:batch_size]
            print("LOGITS SOURCE else: ", logits_source)
            print("LABELS INSIDE COMPUTE else: ", source_labels)
            print("LEN LOGITS SOURCE: ", logits_source.shape)
            print("LEN LABELS SOURCE: ", source_labels.shape)
            task_loss = self.criterion(logits_source, source_labels)
            loss = task_loss

        # Calculate metrics for source
        preds_source = torch.argmax(logits_source, dim=1)
        accuracy = self.accuracy(source_labels, preds_source)
        f1 = self.f1(source_labels, preds_source)

        # Log metrics and losses
        self.log_dict({
            "loss": loss, "task_loss": task_loss, "divergence": divergence,
            "accuracy": accuracy, "f1": f1
        }, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        source_batch = batch[0]["source"]
        target_batch = batch[1]["source"] if len(batch) > 1 else None
        print("JUST BATCH: ", batch, type(batch))
        print("SOURCE BATCH: ", source_batch, type(source_batch))
        print("IS BATCH THE SAME THAT SOURCE BATCH: ", source_batch==batch)

        # Prepare input tensors for source and target
        source_input_ids, source_attention_mask = source_batch["input_ids"], source_batch["attention_mask"]
        input_ids, attention_mask = source_input_ids, source_attention_mask
        token_type_ids = source_batch.get("token_type_ids", None)
        source_labels = source_batch["label"]
        print("LABELS: ", source_labels, type(source_labels))

        if target_batch:
            # Concatenate source and target data if target_batch is present
            target_input_ids, target_attention_mask = target_batch["input_ids"], target_batch["attention_mask"]
            input_ids = torch.cat((input_ids, target_input_ids), dim=0)
            attention_mask = torch.cat((attention_mask, target_attention_mask), dim=0)
            if token_type_ids is not None:
                token_type_ids = torch.cat((token_type_ids, target_batch["token_type_ids"]), dim=0)

        # Single forward pass for combined source and target inputs
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.compute_loss_and_metrics(cls_token, logits, source_labels)

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
        loss = self.compute_loss_and_metrics(cls_token, logits, source_labels, target_labels)
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