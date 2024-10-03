import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
import os


class FineTuneTask(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.test_step_outputs = [] 
        self.validation_step_outputs = [] 
        hf_token = os.getenv("HUGGINGFACE_TOKEN")


        self.save_hyperparameters(hparams)

        base_model_name = self.hparams['pretrained_model_name']

        self.num_classes = self.hparams['num_classes']

        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, token=hf_token)
        
        # No LoRA for simplicity here, using base model directly
        self.model = base_model

        self.criterion = CrossEntropyLoss()
        
        # Initialize F1 and Accuracy measures
        self.accuracy = Accuracy(task='binary')  # accuracy
        self.f1 = F1Score(task='binary', num_classes=self.num_classes, average='macro')  # F1

        # You can remove MK-MMD and related domain adaptation components
        # as we're now focusing only on training the target dataset
    
    def forward(self, input_ids, attention_mask):
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  
        cls_token = last_hidden_state[:, 0, :]
        return cls_token, outputs.logits

    def training_step(self, batch, batch_idx):
        # Use only target data for training
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        labels = batch["label_target"]

        # Forward pass
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Calculate metrics: accuracy and F1
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        self.log("train/accuracy", accuracy)
        self.log("train/f1", f1)
        self.log("train/task_loss", task_loss)

        return task_loss

    def validation_step(self, batch):
        # Use only target data for validation
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        labels = batch["label_target"]

        # Forward pass
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Compute accuracy and F1 scores
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        # Collect metrics for epoch-end logging
        metrics = {
            "validation/loss": task_loss,
            "validation/accuracy": accuracy,
            "validation/f1": f1,
        }

        self.validation_step_outputs.append(metrics)

        return metrics    
    
    def on_validation_epoch_end(self):
        # Calculate the mean of the collected metrics across all validation steps
        avg_loss = torch.tensor([x['validation/loss'] for x in self.validation_step_outputs]).mean()
        avg_accuracy = torch.tensor([x['validation/accuracy'] for x in self.validation_step_outputs]).mean()
        avg_f1 = torch.tensor([x['validation/f1'] for x in self.validation_step_outputs]).mean()

        # Log the average metrics
        self.log("avg_validation_loss", avg_loss)
        self.log("avg_validation_accuracy", avg_accuracy)
        self.log("avg_validation_f1", avg_f1)

        # Clear the step outputs for memory efficiency
        self.validation_step_outputs.clear()

    def test_step(self, batch):
        # Use only target data for testing
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        labels = batch["label_target"]

        # Forward pass
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Compute accuracy and F1 scores
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        # Collect metrics for epoch-end logging
        metrics = {
            "test/loss": task_loss,
            "test/accuracy": accuracy,
            "test/f1": f1,
        }

        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        """Logs overall test metrics after all batches."""
        avg_loss = torch.tensor([x['test/loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.tensor([x['test/accuracy'] for x in self.test_step_outputs]).mean()
        avg_f1 = torch.tensor([x['test/f1'] for x in self.test_step_outputs]).mean()

        # Log final test metrics
        self.log("avg_test_loss", avg_loss.item())
        self.log("avg_test_accuracy", avg_acc.item())
        self.log("avg_test_f1", avg_f1.item())

        # Clear the test step outputs for the next epoch
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(self.model.parameters(), lr=float(self.hparams.learning_rate))
        return optimizer
