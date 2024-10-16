import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from torch.nn import CrossEntropyLoss, Dropout
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score, Recall, Precision
import os


class FineTuneTask(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        hf_token = os.getenv("HUGGINGFACE_TOKEN")


        self.save_hyperparameters(hparams)

        self.base_model_name = self.hparams['pretrained_model_name']

        self.num_classes = self.hparams['num_classes']

        base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, token=hf_token)
        
        # No LoRA for simplicity here, using base model directly
        self.model = base_model

        self.criterion = CrossEntropyLoss()

        # Add dropout
        self.dropout = Dropout(p=self.hparams['dropout'])
        
        # Initialize F1 and Accuracy measures
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)  # accuracy

        self.f1 = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')  # F1

        # Add also recall to see why f1 so low in validation & test
        self.recall = Recall(task='multiclass', num_classes=self.num_classes, average='none')  
        self.precision = Precision(task='multiclass', num_classes=self.num_classes, average='none')  # Per-class precision


        # You can remove MK-MMD and related domain adaptation components
        # as we're now focusing only on training the target dataset
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):

        # Forward pass
        if self.base_model_name == 'bert-base-uncased':
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  
        cls_token = last_hidden_state[:, 0, :]
        # Apply dropout before passing it to the classifier
        #cls_token = self.dropout(cls_token)

        # Get logits from the model's classification head
        #logits = self.model.classifier(cls_token)

        return cls_token, outputs.logits
    

    def training_step(self, batch, batch_idx):
        # Use only target data for training
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        token_type_ids = batch.get("target_token_type_ids", None)  # Use token_type_ids if available
        labels = batch["label_target"]

        # Forward pass
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Calculate metrics: accuracy and F1
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)
        precision = self.precision(labels, preds)

        # add recall
        recall = self.recall(labels, preds)

        # Log metrics at epoch level
        self.log("train/loss", task_loss, on_step=False, on_epoch=True)
        self.log("train/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True)

        self.log(f"train/recall_class_0 (negative)", recall[0].item(), on_step=False, on_epoch=True)
        self.log(f"train/recall_class_1 (positive)", recall[1].item(), on_step=False, on_epoch=True)

        self.log(f"train/precision_class_0 (negative)", precision[0].item(), on_step=False, on_epoch=True)
        self.log(f"train/precision_class_1 (positive)", precision[1].item(), on_step=False, on_epoch=True)

        return task_loss

    def validation_step(self, batch, batch_idx):
        # Use only target data for validation
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        token_type_ids = batch.get("target_token_type_ids", None)  # Use token_type_ids if available
        labels = batch["label_target"]

        # Forward pass
        # since I'm using roberta, delete token_type_ids
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Compute accuracy and F1 scores
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)
        recall = self.recall(labels, preds)
        precision = self.precision(labels, preds)

        # print(f"Labels: {labels}")
        # print(f"Preds: {preds}")

        # Log metrics at epoch level
        self.log("validation/loss", task_loss, on_step=False, on_epoch=True)
        self.log("validation/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("validation/f1", f1, on_step=False, on_epoch=True)

        self.log(f"validation/recall_class_0 (negative)", recall[0].item(), on_step=False, on_epoch=True)
        self.log(f"validation/recall_class_1 (positive)", recall[1].item(), on_step=False, on_epoch=True)

        self.log(f"validation/precision_class_0 (negative)", precision[0].item(), on_step=False, on_epoch=True)
        self.log(f"validation/precision_class_1 (positive)", precision[1].item(), on_step=False, on_epoch=True)

        return {"loss": task_loss, "accuracy": accuracy, "f1": f1}
    

    def test_step(self, batch, batch_idx):
        # Use only target data for testing
        input_ids = batch["target_input_ids"]
        attention_mask = batch["target_attention_mask"]
        token_type_ids = batch.get("target_token_type_ids", None)  # Use token_type_ids if available
        labels = batch["label_target"]

        # Forward pass
        # since I'm using roberta, delete token_type_ids
        cls_token, logits = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Task loss (classification loss)
        task_loss = self.criterion(logits, labels)

        # Compute accuracy and F1 scores
        preds = torch.argmax(logits, dim=1)

        accuracy = self.accuracy(labels, preds)
        f1 = self.f1(labels, preds)

        recall = self.recall(labels, preds)
        precision = self.precision(labels, preds)

        # Log metrics at epoch level
        self.log("test/loss", task_loss, on_step=False, on_epoch=True)
        self.log("test/accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True)

        self.log(f"test/recall_class_0 (negative)", recall[0].item(), on_step=False, on_epoch=True)
        self.log(f"test/recall_class_1 (positive)", recall[1].item(), on_step=False, on_epoch=True)

        self.log(f"test/precision_class_0 (negative)", precision[0].item(), on_step=False, on_epoch=True)
        self.log(f"test/precision_class_1 (positive)", precision[1].item(), on_step=False, on_epoch=True)


        return {"loss": task_loss, "accuracy": accuracy, "f1": f1}


    def configure_optimizers(self):
        # AdamW optimizer
        # prova a aggiungere anche un weight decay 
        optimizer = AdamW(self.model.parameters(), lr=float(self.hparams['learning_rate']))
        # , weight_decay=float(self.hparams['weight_decay']))
        total_steps= self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
        return [optimizer], [scheduler]