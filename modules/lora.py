import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
from divergences.mkmmd import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
import torchmetrics
import os


class LoRA_module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.test_step_outputs = []  
        hf_token = os.getenv("HUGGINGFACE_TOKEN")


        self.save_hyperparameters(hparams)

        base_model_name = "bert-base-uncased"

        self.num_classes = self.hparams.get('num_classes', 2)


        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, token=hf_token)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence Classification task
            bias='lora_only', # Update only lora's parameters
            inference_mode=False,  # Enable training mode
            r=self.hparams.get('lora_r', 8),  # Low-rank matrix rank
            lora_alpha=self.hparams.get('lora_alpha', 32),  # Scaling factor
            lora_dropout=self.hparams.get('lora_dropout', 0.1)  # Dropout probability
        )
        
        self.model = get_peft_model(base_model, peft_config)
        self.criterion = CrossEntropyLoss()
        
        # Initialize F1 and Accuracy measures
        self.accuracy = torchmetrics.Accuracy()  # accuracy
        self.f1 = torchmetrics.F1(num_classes=self.num_classes, average='macro')  # F1


        # Initialize MK-MMD with Gaussian Kernels
        self.kernels = [GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.0), GaussianKernel(alpha=2.0)]
        self.mk_mmd_loss = MultipleKernelMaximumMeanDiscrepancy(self.kernels, linear=False)

        # for debugging
        self.model.print_trainable_parameters()

    
    def forward(self, input_ids, attention_mask):
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Return pooled_output to calculate the divergence between source and target later
        pooled_output = outputs.pooler_output
        
        return pooled_output, outputs.logits

    def training_step(self, batch, batch_idx):
        # Concatenate source and target data for domain adaptation
        input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        attention_mask = torch.cat([batch["source_attention_mask"], batch["target_attention_mask"]], dim=0)
        labels = batch["label_source"]

        # Compute steps for dynamic alpha calculation
        start_steps = self.current_epoch * batch["source_input_ids"].shape[0]
        total_steps = self.hparams["epochs"] * batch["source_input_ids"].shape[0]
        p = float(batch_idx + start_steps) / total_steps
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1  # Used to weigh task loss vs divergence

        # Forward pass using pooler_output
        pooled_output, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        # Split source and target features for divergence calculation
        src_feature, trg_feature = torch.split(pooled_output, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Compute MK-MMD divergence using pooler_output
        divergence = self.mk_mmd_loss(src_feature, trg_feature)

        # Split logits back into source and target (only using source labels for task loss)
        logits, _ = torch.split(logits, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Total loss: weighted combination of task loss and divergence
        loss = alpha * task_loss + (1 - alpha) * divergence

        # Calculate metrics: accuracy and F1
        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        self.log("train/accuracy", accuracy)
        self.log("train/f1", f1)
        self.log("train/task_loss", task_loss)
        self.log("train/loss", loss)
        self.log("train/divergence", divergence)

        return loss

    def validation_step(self, batch, batch_idx):
        # Concatenate source and target data for domain adaptation
        input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        attention_mask = torch.cat([batch["source_attention_mask"], batch["target_attention_mask"]], dim=0)

        # Forward pass through the model
        pooled_output, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        # Split source and target features for divergence calculation
        src_feature, trg_feature = torch.split(pooled_output, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Compute MK-MMD divergence using pooler_output
        divergence = self.mk_mmd_loss(src_feature, trg_feature)

        # Split logits back into source and target (source and target labels)
        logits_source, logits_target = torch.split(logits, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Compute task-specific losses for both source and target
        source_taskclf_loss = self.criterion(logits_source, batch["label_source"])
        target_taskclf_loss = self.criterion(logits_target, batch["label_target"])

        # Combine losses with divergence
        source_loss = source_taskclf_loss + divergence
        target_loss = target_taskclf_loss + divergence
        total_loss = source_taskclf_loss + target_taskclf_loss + divergence

        # Compute accuracy and F1 scores for both source and target
        source_preds = torch.argmax(logits_source, dim=1)
        target_preds = torch.argmax(logits_target, dim=1)

        source_accuracy = self.accuracy(batch["label_source"], source_preds)
        source_f1 = self.f1(batch["label_source"], source_preds)

        target_accuracy = self.accuracy(batch["label_target"], target_preds.cpu())
        target_f1 = self.f1(batch["label_target"], target_preds)

        # Log metrics for both source and target test performance
        self.log("source_test/loss", source_loss)
        self.log("source_test/taskclf_loss", source_taskclf_loss)
        self.log("test/domain_loss", divergence)
        self.log("test/loss", total_loss)
        self.log("source_test/accuracy", source_accuracy)
        self.log("source_test/f1", source_f1)
        self.log("target_test/loss", target_loss)
        self.log("target_test/taskclf_loss", target_taskclf_loss)
        self.log("target_test/accuracy", target_accuracy)
        self.log("target_test/f1", target_f1)

        # Collect metrics for epoch end logging
        metrics = {
            "source_test/loss": source_loss,
            "source_test/accuracy": source_accuracy,
            "source_test/f1": source_f1,
            "target_test/loss": target_loss,
            "target_test/accuracy": target_accuracy,
            "target_test/f1": target_f1,
        }

        return metrics    
    
    def test_step(self, batch, batch_idx):
        # Concatenate source and target data for domain adaptation
        input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        attention_mask = torch.cat([batch["source_attention_mask"], batch["target_attention_mask"]], dim=0)

        # Forward pass through the model
        pooled_output, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        # Split source and target features for divergence calculation
        src_feature, trg_feature = torch.split(pooled_output, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Compute MK-MMD divergence using pooler_output
        divergence = self.mk_mmd_loss(src_feature, trg_feature)

        # Split logits back into source and target (source and target labels)
        logits_source, logits_target = torch.split(logits, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Compute task-specific losses for both source and target
        source_taskclf_loss = self.criterion(logits_source, batch["label_source"])
        target_taskclf_loss = self.criterion(logits_target, batch["label_target"])

        # Combine losses with divergence
        source_loss = source_taskclf_loss + divergence
        target_loss = target_taskclf_loss + divergence
        total_loss = source_taskclf_loss + target_taskclf_loss + divergence

        # Compute accuracy and F1 scores for both source and target
        source_preds = torch.argmax(logits_source, dim=1)
        target_preds = torch.argmax(logits_target, dim=1)

        source_accuracy = self.accuracy(batch["label_source"], source_preds)
        source_f1 = self.f1(batch["label_source"], source_preds)

        target_accuracy = self.accuracy(batch["label_target"], target_preds)
        target_f1 = self.f1(batch["label_target"], target_preds)

        # Log metrics for both source and target test performance
        self.log("source_test/loss", source_loss)
        self.log("source_test/taskclf_loss", source_taskclf_loss)
        self.log("test/domain_loss", divergence)
        self.log("test/loss", total_loss)
        self.log("source_test/accuracy", source_accuracy)
        self.log("source_test/f1", source_f1)
        self.log("target_test/loss", target_loss)
        self.log("target_test/taskclf_loss", target_taskclf_loss)
        self.log("target_test/accuracy", target_accuracy)
        self.log("target_test/f1", target_f1)

        # Collect metrics for epoch end logging
        metrics = {
            "source_test/loss": source_loss,
            "source_test/accuracy": source_accuracy,
            "source_test/f1": source_f1,
            "target_test/loss": target_loss,
            "target_test/accuracy": target_accuracy,
            "target_test/f1": target_f1,
        }

        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        """Logs overall test metrics after all batches."""
        avg_source_loss = torch.tensor([x['source_test/loss'] for x in self.test_step_outputs]).mean()
        avg_source_acc = torch.tensor([x['source_test/accuracy'] for x in self.test_step_outputs]).mean()
        avg_source_f1 = torch.tensor([x['source_test/f1'] for x in self.test_step_outputs]).mean()

        avg_target_loss = torch.tensor([x['target_test/loss'] for x in self.test_step_outputs]).mean()
        avg_target_acc = torch.tensor([x['target_test/accuracy'] for x in self.test_step_outputs]).mean()
        avg_target_f1 = torch.tensor([x['target_test/f1'] for x in self.test_step_outputs]).mean()

        # Log final test metrics for source and target datasets
        self.log("avg_source_test_loss", avg_source_loss.item())
        self.log("avg_source_test_acc", avg_source_acc.item())
        self.log("avg_source_test_f1", avg_source_f1.item())
        self.log("avg_target_test_loss", avg_target_loss.item())
        self.log("avg_target_test_acc", avg_target_acc.item())
        self.log("avg_target_test_f1", avg_target_f1.item())

        # Clear the test step outputs for the next epoch
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(self.model.parameters(), lr=float(self.hparams.learning_rate))
        return optimizer
