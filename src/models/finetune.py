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

        # Load hyperparameters
        self._load_hyperparameters()

        self.base_model_name = self.hparams["pretrained_model_name"]
        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name)

        # Finetune the full model instead of using LoRA
        self.model = base_model

        # Loss function
        self.criterion = CrossEntropyLoss()

        # Evaluation metrics
        self.accuracy = Accuracy(task="binary", num_classes=self.num_classes)
        self.f1 = F1Score(task="binary", num_classes=self.num_classes)

        # MK-MMD for domain adaptation
        self.kernels = [GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.0), GaussianKernel(alpha=2.0)]
        self.mk_mmd_loss = MultipleKernelMaximumMeanDiscrepancy(self.kernels, linear=False)

        # Debugging: Print trainable parameters
        self.model.print_trainable_parameters()

    def _load_hyperparameters(self):
        """Loads hyperparameters from wandb if tuning is enabled, else from hparams."""
        hparams_source = wandb.config if wandb.run else self.hparams
        self.learning_rate = hparams_source.get("learning_rate", 5e-5)
        self.weight_decay = hparams_source.get("weight_decay", 0.01)
        self.batch_size = hparams_source.get("batch_size", 32)
        self.shuffle = hparams_source.get("shuffle", True)
        self.num_classes = hparams_source.get("num_classes", 2)
        self.n_epochs = hparams_source.get("n_epochs", 3)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        cls_token = outputs.hidden_states[-1][:, 0, :]
        return cls_token, outputs.logits

    def training_step(self, batch, batch_idx):
        """Trains the model only on the source dataset but applies domain adaptation if the target dataset is available."""
        source_batch = batch["source"]
        target_batch = batch.get("target", None)

        # Prepare inputs
        input_ids, attention_mask, token_type_ids, source_labels, target_labels = self._prepare_inputs(source_batch, target_batch)

        # Forward pass
        cls_token, logits = self(input_ids, attention_mask, token_type_ids)

        # Compute task loss for the source domain
        source_task_loss = self.criterion(logits[:len(source_labels)], source_labels)

        # Compute divergence loss if a target dataset is available
        divergence_loss = 0.0
        if target_labels is not None:
            source_cls_token, target_cls_token = cls_token[:len(source_labels)], cls_token[len(source_labels):]
            divergence_loss = self.mk_mmd_loss(source_cls_token, target_cls_token)

        # Compute dynamic alpha
        alpha = self._compute_alpha(batch_idx)

        # Combine task and divergence loss
        total_loss = alpha * source_task_loss + (1 - alpha) * divergence_loss

        # Log losses and metrics
        self.log("train/source_task_loss", source_task_loss, on_epoch=True)
        self.log("train/divergence_loss", divergence_loss, on_epoch=True)

        preds = torch.argmax(logits[:len(source_labels)], dim=1)
        acc = self.accuracy(source_labels, preds)
        f1 = self.f1(source_labels, preds)

        self.log("train/accuracy", acc, on_epoch=True)
        self.log("train/f1", f1, on_epoch=True)

        return total_loss

    def _compute_alpha(self, batch_idx):
        """Dynamically computes alpha based on training progress."""
        total_steps = self.n_epochs * self.trainer.estimated_stepping_batches
        current_step = self.current_epoch * self.trainer.estimated_stepping_batches + batch_idx
        p = float(current_step) / total_steps
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1  # Alpha scales from 0 → 1 smoothly

    def _prepare_inputs(self, source_batch, target_batch):
        """Prepares input tensors for the model."""
        source_input_ids, source_attention_mask = source_batch["input_ids"], source_batch["attention_mask"]
        input_ids, attention_mask = source_input_ids, source_attention_mask
        token_type_ids = source_batch.get("token_type_ids", None)
        source_labels = source_batch["label"]
        target_labels = None

        if target_batch:
            target_input_ids, target_attention_mask = target_batch["input_ids"], target_batch["attention_mask"]
            input_ids = torch.cat((input_ids, target_input_ids), dim=0)
            attention_mask = torch.cat((attention_mask, target_attention_mask), dim=0)
            if token_type_ids is not None:
                token_type_ids = torch.cat((token_type_ids, target_batch.get("token_type_ids", None)), dim=0)
            target_labels = target_batch["label"]

        return input_ids, attention_mask, token_type_ids, source_labels, target_labels

    def validation_step(self, batch, batch_idx):
        """Evaluates the model only on the target dataset."""
        return self._eval_step(batch, "validation")

    def test_step(self, batch, batch_idx):
        """Tests the model only on the target dataset."""
        return self._eval_step(batch, "test")

    def _eval_step(self, batch, stage):
        """Evaluates the model using only the target dataset."""
        target_batch = batch["target"]
        target_labels = target_batch["label"]
        input_ids = target_batch["input_ids"]
        attention_mask = target_batch["attention_mask"]
        token_type_ids = target_batch.get("token_type_ids", None)

        # Forward pass
        cls_token, logits = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, target_labels)

        # Compute and log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(target_labels, preds)
        f1 = self.f1(target_labels, preds)

        self.log(f"{stage}/loss", loss, on_epoch=True)
        self.log(f"{stage}/accuracy", acc, on_epoch=True)
        self.log(f"{stage}/f1", f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configures optimizer and scheduler."""
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }