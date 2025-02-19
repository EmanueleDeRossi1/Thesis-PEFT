import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
from src.divergences.mkmmd import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from torchmetrics import Accuracy, F1Score
import os
import wandb


class FineTune(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        # add hparams for hparams tuning
        if wandb.run:
            self.learning_rate = wandb.config.learning_rate
            self.weight_decay = wandb.config.weight_decay
            self.batch_size = wandb.config.batch_size

        else:
            self.learning_rate = self.hparams['learning_rate']
            self.weight_decay = self.hparams['weight_decay']
            self.batch_size = self.hparams['batch_size']
            self.n_epochs = self.hparams['n_epochs']
            
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.base_model_name = self.hparams['pretrained_model_name']
        self.num_classes = self.hparams['num_classes']
        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, token=hf_token)
        self.num_classes = self.hparams['num_classes']

        # Instead of using LoRA, finetune the full model
        self.model = base_model
        self.criterion = CrossEntropyLoss()
        # Initialize F1 and Accuracy measures
        self.accuracy = Accuracy(task='binary', num_classes=self.num_classes)  # accuracy
        self.f1 = F1Score(task='binary', num_classes=self.num_classes)  # F1
        # Initialize MK-MMD with Gaussian Kernels
        self.kernels = [GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.0), GaussianKernel(alpha=2.0)]
        self.mk_mmd_loss = MultipleKernelMaximumMeanDiscrepancy(self.kernels, linear=False)

        # for debugging
        self.model.print_trainable_parameters()

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

    def compute_train_loss_and_metrics(self, cls_token, logits, source_labels, target_labels, batch_idx=None):
        batch_size = source_labels.shape[0]
        # Task loss for source data
        if target_labels is not None:
            # Separate source and target portions
            source_cls_token, target_cls_token = cls_token[:len(source_labels)], cls_token[len(source_labels):]
            source_logits, target_logits = logits[:len(source_labels)], logits[len(source_labels):]
                        
            # Calculate divergence
            divergence_loss = self.mk_mmd_loss(source_cls_token, target_cls_token)

            source_task_loss = self.criterion(source_logits, source_labels)

            # calculate alpha dinamically
            start_steps = self.current_epoch * batch_size
            total_steps = self.n_epochs * batch_size
            p = float(batch_idx + start_steps) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            # calculate loss as trade off between task and divergence loss
            total_loss = alpha * source_task_loss + (1 - alpha) * divergence_loss
   
            # Log individual losses
            self.log("train/source_task_loss", source_task_loss, on_step=True, on_epoch=True) 
            self.log("train/divergence_loss", divergence_loss, on_step=True, on_epoch=True)

            preds_target = torch.argmax(target_logits, dim=1)
            target_accuracy = self.accuracy(target_labels, preds_target)
            target_f1 = self.f1(target_labels, preds_target)
            # Log the accuracy and f1 at the end of the epoch
            self.log("target_train/accuracy",  target_accuracy, on_step=False, on_epoch=True) 
            self.log("target_train/f1", target_f1, on_step=False, on_epoch=True)
        else:
            source_cls_token = cls_token
            source_logits = logits

        source_task_loss = self.criterion(source_logits, source_labels)
        total_loss = source_task_loss

        # Log the task loss
        self.log("train/source_task_loss", source_task_loss, on_step=True, on_epoch=True)
        
        preds_source = torch.argmax(source_logits, dim=1)
        source_accuracy = self.accuracy(source_labels, preds_source)
        source_f1 = self.f1(source_labels, preds_source)
        self.log("source_train/accuracy",  source_accuracy, on_step=False, on_epoch=True) 
        self.log("source_train/f1", source_f1, on_step=False, on_epoch=True)

    
        return total_loss
    
    def compute_eval_loss_and_metrics(self, stage, cls_token, logits, source_labels, target_labels):
        batch_size = target_labels.shape[0]
        if source_labels is not None:
            # Separate source and target portions if source batch is present
            source_cls_token, target_cls_token = cls_token[:batch_size], cls_token[batch_size:]
            source_logits, target_logits = logits[:batch_size], logits[batch_size:]
            
            # Calculate divergence
            divergence_loss = self.mk_mmd_loss(source_cls_token, target_cls_token)

            target_task_loss = self.criterion(target_logits, target_labels)
            
            # Combine task and divergence losses
            total_loss =  0.5 * target_task_loss + 0.5 * divergence_loss

            # Log divergence loss
            self.log(f"{stage}/divergence_loss", divergence_loss, on_step=True, on_epoch=True)

            # Calculate accuracy and f1 for source dataset
            preds_source = torch.argmax(source_logits, dim=1)
            source_accuracy = self.accuracy(source_labels, preds_source)
            source_f1 = self.f1(source_labels, preds_source)
            self.log(f"source_{stage}/accuracy",  source_accuracy, on_step=False, on_epoch=True)
            self.log(f"source_{stage}/f1", source_f1, on_step=False, on_epoch=True)
        
        else:
            target_cls_token = cls_token
            target_logits = logits

            target_task_loss = self.criterion(target_logits, target_labels)
            total_loss = target_task_loss

        preds_target = torch.argmax(target_logits, dim=1)
        target_accuracy = self.accuracy(target_labels, preds_target)
        target_f1 = self.f1(target_labels, preds_target)
        self.log(f"target_{stage}/task_loss", target_task_loss, on_step=True, on_epoch=True)

        # Calculate f1 and accuracy for target dataset
        self.log(f"target_{stage}/accuracy",  target_accuracy, on_step=False, on_epoch=True)
        self.log(f"target_{stage}/f1", target_f1, on_step=False, on_epoch=True)

        return total_loss


    def training_step(self, batch, batch_idx):
        source_batch = batch["source"]
        target_batch = batch["target"] if "target" in batch else None

        # Prepare input tensors for source and target
        source_input_ids, source_attention_mask = source_batch["input_ids"], source_batch["attention_mask"]
        input_ids, attention_mask = source_input_ids, source_attention_mask
        token_type_ids = source_batch.get("token_type_ids", None)
        source_labels = source_batch["label"]
        target_labels = None

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
        return self.compute_train_loss_and_metrics(cls_token, logits, source_labels, target_labels, batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, stage="validation")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, stage="test")

    def _eval_step(self, batch, stage):
        source_batch = batch["source"] if "source" in batch else None
        target_batch = batch["target"]

        target_input_ids, target_attention_mask = target_batch["input_ids"], target_batch["attention_mask"]
        input_ids, attention_mask = target_input_ids, target_attention_mask
        token_type_ids = target_batch.get("token_type_ids", None)
        target_labels = target_batch["label"]
        source_labels = None

        if source_batch:
            # Concatenate source and target data if source_batch is present
            source_input_ids, source_attention_mask = source_batch["input_ids"], source_batch["attention_mask"]
            input_ids = torch.cat((source_input_ids, input_ids), dim=0)
            attention_mask = torch.cat((source_attention_mask, attention_mask), dim=0)
            if token_type_ids is not None:
                token_type_ids = torch.cat((source_batch["token_type_ids"], token_type_ids), dim=0)
            source_labels = source_batch["label"]
        
        # Single forward pass for combined source and target data
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Calculate loss and metrics
        loss = self.compute_eval_loss_and_metrics(stage, cls_token, logits, source_labels, target_labels)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True)



    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(self.model.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))
        total_steps= self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
        return [optimizer], [scheduler]