import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
from divergences.mkmmd import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from torchmetrics import Accuracy, F1Score
import os


class LoRA_module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.test_step_outputs = [] 
        self.validation_step_outputs = [] 
        hf_token = os.getenv("HUGGINGFACE_TOKEN")


        self.save_hyperparameters(hparams)

        base_model_name = self.hparams['pretrained_model_name']

        self.num_classes = self.hparams['num_classes']

        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, token=hf_token)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence Classification task
            bias='lora_only', # Update only lora's bias term
            inference_mode=False,  # Enable training mode
            r=self.hparams['lora_r'],  # Low-rank matrix rank
            lora_alpha=self.hparams['lora_alpha'],  # Scaling factor
            lora_dropout=self.hparams['lora_dropout']  # Dropout probability
        )
        
        self.model = get_peft_model(base_model, peft_config)

        self.criterion = CrossEntropyLoss()
        
        # Initialize F1 and Accuracy measures
        self.accuracy = Accuracy(task='binary')  # accuracy
        self.f1 = F1Score(task='binary', num_classes=self.num_classes, average='macro')  # F1


        # Initialize MK-MMD with Gaussian Kernels
        self.kernels = [GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.0), GaussianKernel(alpha=2.0)]
        self.mk_mmd_loss = MultipleKernelMaximumMeanDiscrepancy(self.kernels, linear=False)

        # for debugging
        self.model.print_trainable_parameters()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  
        cls_token = last_hidden_state[:, 0, :]
        return cls_token, outputs.logits

    def training_step(self, batch, batch_idx):
        # Concatenate source and target data for domain adaptation
        input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        attention_mask = torch.cat([batch["source_attention_mask"], batch["target_attention_mask"]], dim=0)
        token_type_ids = torch.cat([batch["source_token_type_ids"], batch["target_token_type_ids"]], dim=0)
        labels = batch["label_source"]

        # Compute steps for dynamic alpha calculation
        start_steps = self.current_epoch * batch["source_input_ids"].shape[0]
        total_steps = self.hparams["n_epochs"] * batch["source_input_ids"].shape[0]
        p = float(batch_idx + start_steps) / total_steps
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1  # Used to weigh task loss vs divergence

        # Forward pass using cls token
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Split source and target features for divergence calculation
        src_feature, trg_feature = torch.split(cls_token, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Compute MK-MMD divergence using cls token
        divergence = self.mk_mmd_loss(src_feature, trg_feature)

        # Split logits back into source and target (only using source labels for task loss)
        logits, _ = torch.split(logits, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Total loss: weighted combination of task loss and divergence
        loss = alpha * task_loss + (1 - alpha) * divergence

        # Calculate metrics: accuracy and F1
        preds = torch.argmax(logits, dim=1)


        # print(f"Labels from batch {batch_idx}: {labels}")
        # print(f"Predictions from batch {batch_idx}: {preds}")
        # print(f"Logits from batch {batch_idx}:{logits}")

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
        token_type_ids = torch.cat([batch["source_token_type_ids"], batch["target_token_type_ids"]], dim=0)


        # Forward pass using cls token
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Split source and target features for divergence calculation
        src_feature, trg_feature = torch.split(cls_token, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Compute MK-MMD divergence using cls token
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

        print(f"Validation Target labels from batch {batch_idx}: {batch['label_target']}")
        print(f"Validation Target predictions from batch {batch_idx}: {target_preds}")
        # print(f"Validation logits source from batch {batch_idx}:{logits_source}")
        # print(f"Validation logits target from batch {batch_idx}:{logits_target}")
        print(f"Validation Source labels from batch {batch_idx}: {batch['label_source']}")
        print(f"Validation Source predictions from batch {batch_idx}: {source_preds}")


        source_accuracy = self.accuracy(batch['label_source'], source_preds)
        source_f1 = self.f1(batch['label_source'], source_preds)

        target_accuracy = self.accuracy(batch['label_target'], target_preds)
        target_f1 = self.f1(batch['label_target'], target_preds)

        # Collect metrics for epoch end logging
        metrics = {
            "source_validation/loss": source_loss,
            "source_validation/accuracy": source_accuracy,
            "source_validation/f1": source_f1,
            "target_validation/loss": target_loss,
            "target_validation/accuracy": target_accuracy,
            "target_validation/f1": target_f1,
        }

        self.validation_step_outputs.append(metrics)

        return metrics    
    
    def on_validation_epoch_end(self):
        # Calculate the mean of the collected metrics across all validation steps
        avg_source_loss = torch.tensor([x['source_validation/loss'] for x in self.validation_step_outputs]).mean()
        avg_source_accuracy = torch.tensor([x['source_validation/accuracy'] for x in self.validation_step_outputs]).mean()
        avg_source_f1 = torch.tensor([x['source_validation/f1'] for x in self.validation_step_outputs]).mean()

        avg_target_loss = torch.tensor([x['target_validation/loss'] for x in self.validation_step_outputs]).mean()
        avg_target_accuracy = torch.tensor([x['target_validation/accuracy'] for x in self.validation_step_outputs]).mean()
        avg_target_f1 = torch.tensor([x['target_validation/f1'] for x in self.validation_step_outputs]).mean()

        # Log the average metrics
        self.log("avg_source_validation_loss", avg_source_loss)
        self.log("avg_source_validation_accuracy", avg_source_accuracy)
        self.log("avg_source_validation_f1", avg_source_f1)
        
        self.log("avg_target_validation_loss", avg_target_loss)
        self.log("avg_target_validation_accuracy", avg_target_accuracy)
        self.log("avg_target_validation_f1", avg_target_f1)

        # Clear the step outputs for memory efficiency
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # Concatenate source and target data for domain adaptation
        input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        attention_mask = torch.cat([batch["source_attention_mask"], batch["target_attention_mask"]], dim=0)
        token_type_ids = torch.cat([batch["source_token_type_ids"], batch["target_token_type_ids"]], dim=0)

        # Forward pass using cls token
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Split source and target features for divergence calculation
        src_feature, trg_feature = torch.split(cls_token, split_size_or_sections=input_ids.shape[0] // 2, dim=0)

        # Compute MK-MMD divergence using cls token
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

        print(f"Target labels from batch {batch_idx}: {batch['label_target']}")
        print(f"Target predictions from batch {batch_idx}: {target_preds}")
        # print(f"Logits source from batch {batch_idx}:{logits_source}")
        # print(f"Logits target from batch {batch_idx}:{logits_target}")
        print(f"Source labels from batch {batch_idx}: {batch['label_source']}")
        print(f"Source predictions from batch {batch_idx}: {source_preds}")


        source_accuracy = self.accuracy(batch['label_source'], source_preds)
        source_f1 = self.f1(batch['label_source'], source_preds)

        target_accuracy = self.accuracy(batch['label_target'], target_preds)
        target_f1 = self.f1(batch['label_target'], target_preds)

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
        self.log("avg_source_test_loss", avg_source_loss)
        self.log("avg_source_test_acc", avg_source_acc)
        self.log("avg_source_test_f1", avg_source_f1)
        self.log("avg_target_test_loss", avg_target_loss)
        self.log("avg_target_test_acc", avg_target_acc)
        self.log("avg_target_test_f1", avg_target_f1)

        # Clear the test step outputs for the next epoch
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(self.model.parameters(), lr=float(self.hparams.learning_rate))
        return optimizer
