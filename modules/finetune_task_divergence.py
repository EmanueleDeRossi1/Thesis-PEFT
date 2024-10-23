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

    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # Forward pass
        if self.base_model_name == 'bert-base-uncased':
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  
        cls_token = last_hidden_state[:, 0, :]
        return cls_token, outputs.logits

    def training_step(self, batch, batch_idx):

        # # Concatenate source and target data for domain adaptation
        # input_ids = torch.cat([batch["source_input_ids"], batch["target_input_ids"]], dim=0)
        # attention_mask = torch.cat([batch["source_attention_mask"], batch["target_attention_mask"]], dim=0)
        # token_type_ids = torch.cat([batch["source_token_type_ids"], batch["target_token_type_ids"]], dim=0)
        # labels = batch["label_source"]

        source_batch, target_batch = batch  # Unpack source and target batches

        # Get source data
        source_input_ids = source_batch["source_input_ids"]
        source_attention_mask = source_batch["source_attention_mask"]
        source_token_type_ids = source_batch.get("source_token_type_ids")
        labels = source_batch["label_source"]
        
        # Forward pass using cls token
        # cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            # Forward pass for source
        cls_token_source, logits_source = self(input_ids=source_input_ids, attention_mask=source_attention_mask, token_type_ids=source_token_type_ids)
        

        # logits_source = logits[:batch["source_input_ids"].shape[0]]  # Only source logits for task loss

        # Calculate task loss (classification loss)
        task_loss = self.criterion(logits_source, labels)

        # If the target data is available, calculate total loss
        if "target_input_ids" in batch:
            # Get target data
            target_input_ids = target_batch["target_input_ids"]
            target_attention_mask = target_batch["target_attention_mask"]
            target_token_type_ids = target_batch.get("target_token_type_ids")

            # Forward pass for target
            cls_token_target, _ = self(input_ids=target_input_ids, attention_mask=target_attention_mask, token_type_ids=target_token_type_ids)

            # Compute MK-MMD divergence using cls token
            divergence = self.mk_mmd_loss(cls_token_source, cls_token_target)

            # Compute steps for dynamic alpha calculation
            start_steps = self.current_epoch * batch["target_input_ids"].shape[0]
            total_steps = self.hparams["n_epochs"] * batch["target_input_ids"].shape[0]
            p = float(batch_idx + start_steps) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1  # Used to weigh task loss vs divergence


            # Total loss: weighted combination of task loss and divergence
            loss = alpha * task_loss + (1 - alpha) * divergence

            # Log divergence only when target data is available
            self.log("train/divergence", divergence)
        else:
            # No target data, only use task loss
            loss = task_loss

        # Calculate metrics: accuracy and F1
        preds = torch.argmax(logits_source, dim=1)

        # print(f"Labels from batch {batch_idx}: {labels}")
        # print(f"Predictions from batch {batch_idx}: {preds}")
        # print(f"Logits from batch {batch_idx}:{logits}")

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        self.log("train/accuracy", accuracy)
        self.log("train/f1", f1)
        self.log("train/task_loss", task_loss)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        # Get source data
        source_input_ids = batch["source_input_ids"]
        source_attention_mask = batch["source_attention_mask"]
        source_token_type_ids = batch.get("source_token_type_ids")

        # Get target data
        target_input_ids = batch["target_input_ids"]
        target_attention_mask = batch["target_attention_mask"]
        target_token_type_ids = batch.get("target_token_type_ids")

        source_labels = batch["label_target"]
        target_labels = batch["label_source"]

        # Forward pass for source
        cls_token_source, logits_source = self(input_ids=source_input_ids, attention_mask=source_attention_mask, token_type_ids=source_token_type_ids)
        # Forward pass for target
        cls_token_target, logits_target = self(input_ids=target_input_ids, attention_mask=target_attention_mask, token_type_ids=target_token_type_ids)

        # Compute MK-MMD divergence using cls token
        divergence = self.mk_mmd_loss(cls_token_source, cls_token_target)

        # Compute task-specific losses for both source and target
        source_taskclf_loss = self.criterion(logits_source, source_labels)
        target_taskclf_loss = self.criterion(logits_target, target_labels)
        total_loss = source_taskclf_loss + target_taskclf_loss + divergence

        # Combine losses with divergence
        source_loss = source_taskclf_loss + divergence
        target_loss = target_taskclf_loss + divergence

        # Compute accuracy and F1 scores for both source and target
        source_preds = torch.argmax(logits_source, dim=1)
        target_preds = torch.argmax(logits_target, dim=1)

        print(f"Validation Target labels from batch {batch_idx}: {target_labels}")
        print(f"Validation Target predictions from batch {batch_idx}: {target_preds}")
        # print(f"Validation logits source from batch {batch_idx}:{logits_source}")
        # print(f"Validation logits target from batch {batch_idx}:{logits_target}")
        print(f"Validation Source labels from batch {batch_idx}: {source_labels}")
        print(f"Validation Source predictions from batch {batch_idx}: {source_preds}")


        source_accuracy = self.accuracy(source_labels, source_preds)
        source_f1 = self.f1(source_labels, source_preds)

        target_accuracy = self.accuracy(target_labels, target_preds)
        target_f1 = self.f1(target_labels, target_preds)

        self.log("source_validation/loss", source_loss, on_step=False, on_epoch=True)
        self.log("source_validation/accuracy", source_accuracy, on_step=False, on_epoch=True)
        self.log("source_validation/f1", source_f1, on_step=False, on_epoch=True)

        self.log("target_validation/loss", target_loss, on_step=False, on_epoch=True)
        self.log("target_validation/accuracy", target_accuracy, on_step=False, on_epoch=True)
        self.log("target_validation/f1", target_f1, on_step=False, on_epoch=True)

        self.log("validation/divergence", divergence, on_step=False, on_epoch=True)
        self.log("validation/total_loss", total_loss, on_step=False, on_epoch=True)

        return {"total_loss": total_loss, "source_f1": source_f1, "source_accuracy" : source_accuracy}    
    

    def test_step(self, batch, batch_idx):
        # Get source data
        source_input_ids = batch["source_input_ids"]
        source_attention_mask = batch["source_attention_mask"]
        source_token_type_ids = batch.get("source_token_type_ids")

        # Get target data
        target_input_ids = batch["target_input_ids"]
        target_attention_mask = batch["target_attention_mask"]
        target_token_type_ids = batch.get("target_token_type_ids")

        source_labels = batch["label_target"]
        target_labels = batch["label_source"]

        # Forward pass for source
        cls_token_source, logits_source = self(input_ids=source_input_ids, attention_mask=source_attention_mask, token_type_ids=source_token_type_ids)
        # Forward pass for target
        cls_token_target, logits_target = self(input_ids=target_input_ids, attention_mask=target_attention_mask, token_type_ids=target_token_type_ids)

        # Compute MK-MMD divergence using cls token
        divergence = self.mk_mmd_loss(cls_token_source, cls_token_target)
        
        # Compute task-specific losses for both source and target
        source_taskclf_loss = self.criterion(logits_source, source_labels)
        target_taskclf_loss = self.criterion(logits_target, target_labels)

        # Combine losses with divergence
        source_loss = source_taskclf_loss + divergence
        target_loss = target_taskclf_loss + divergence
        total_loss = source_taskclf_loss + target_taskclf_loss + divergence

        # Compute accuracy and F1 scores for both source and target
        source_preds = torch.argmax(logits_source, dim=1)
        target_preds = torch.argmax(logits_target, dim=1)

        print(f"Target labels from batch {batch_idx}: {target_labels}")
        print(f"Target predictions from batch {batch_idx}: {target_preds}")
        # print(f"Logits source from batch {batch_idx}:{logits_source}")
        # print(f"Logits target from batch {batch_idx}:{logits_target}")
        print(f"Source labels from batch {batch_idx}: {source_labels}")
        print(f"Source predictions from batch {batch_idx}: {source_preds}")


        source_accuracy = self.accuracy(source_labels, source_preds)
        source_f1 = self.f1(source_labels, source_preds)

        target_accuracy = self.accuracy(target_labels, target_preds)
        target_f1 = self.f1(target_labels, target_preds)

        self.log("source_test/loss", source_loss, on_step=False, on_epoch=True)
        self.log("source_test/accuracy", source_accuracy, on_step=False, on_epoch=True)
        self.log("source_test/f1", source_f1, on_step=False, on_epoch=True)

        self.log("target_test/loss", target_loss, on_step=False, on_epoch=True)
        self.log("target_test/accuracy", target_accuracy, on_step=False, on_epoch=True)
        self.log("target_test/f1", target_f1, on_step=False, on_epoch=True)

        self.log("test/divergence", divergence, on_step=False, on_epoch=True)
        self.log("test/total_loss", total_loss, on_step=False, on_epoch=True)

        return {"total_loss": total_loss, "source_f1": source_f1, "source_accuracy" : source_accuracy}    



    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(self.model.parameters(), lr=float(self.hparams['learning_rate']))
        # , weight_decay=float(self.hparams['weight_decay']))
        total_steps= self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
        return [optimizer], [scheduler]