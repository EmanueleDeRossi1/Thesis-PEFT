import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from torch.nn import CrossEntropyLoss, Dropout
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
import os


class FineTuneTask(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        hf_token = os.getenv("HUGGINGFACE_TOKEN")


        self.save_hyperparameters(hparams)

        base_model_name = self.hparams['pretrained_model_name']

        self.num_classes = self.hparams['num_classes']

        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, token=hf_token)
        
        # No LoRA for simplicity here, using base model directly
        self.model = base_model

        self.criterion = CrossEntropyLoss()

        # Add dropout
        self.dropout = Dropout(p=self.hparams['dropout'])
        
        # Initialize F1 and Accuracy measures
        self.accuracy = Accuracy(task='binary')  # accuracy
        self.f1 = F1Score(task='binary', num_classes=self.num_classes, average='macro')  # F1

        # You can remove MK-MMD and related domain adaptation components
        # as we're now focusing only on training the target dataset
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  
        cls_token = last_hidden_state[:, 0, :]
        # Apply dropout before passing it to the classifier
        # cls_token = self.dropout(cls_token)

        # Get logits from the model's classification head
        # logits = self.model.classifier(cls_token)

        return cls_token, outputs.logits
    

    def training_step(self, batch, batch_idx):
        # Use only target data for training
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        token_type_ids = batch["target_token_type_ids"]
        labels = batch["label_target"]
        print("Labels: ",labels)

        # Forward pass
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Calculate metrics: accuracy and F1
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        # Log metrics at epoch level
        self.log("train/loss", task_loss, on_step=False, on_epoch=True)
        self.log("train/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True)

        return task_loss

    def validation_step(self, batch, batch_idx):
        # Use only target data for validation
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        token_type_ids = batch["target_token_type_ids"]
        labels = batch["label_target"]

        # Forward pass
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Compute accuracy and F1 scores
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        # Log metrics at epoch level
        self.log("validation/loss", task_loss, on_step=False, on_epoch=True)
        self.log("validation/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("validation/f1", f1, on_step=False, on_epoch=True)

        return {"loss": task_loss, "accuracy": accuracy, "f1": f1}
    

    def test_step(self, batch, batch_idx):
        # Use only target data for testing
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        token_type_ids = batch["target_token_type_ids"]
        labels = batch["label_target"]

        # Forward pass
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Compute accuracy and F1 scores
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        # Log metrics at epoch level
        self.log("test/loss", task_loss, on_step=False, on_epoch=True)
        self.log("test/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True)

        return {"loss": task_loss, "accuracy": accuracy, "f1": f1}



    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(self.model.parameters(), lr=float(self.hparams.learning_rate))
        return optimizer
