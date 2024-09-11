import torch
import pytorch_lightning as pl
from transformers import BertModel
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn import CrossEntropyLoss
import numpy as np
import wandb
from divergences.mkmmd import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW
import os


class LoRA_module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        # wandb.init(project="LoRA_model_training", config=hparams)
        hf_token = os.getenv("HUGGINGFACE_TOKEN")


        self.save_hyperparameters(hparams)

        base_model_name = "bert-base-uncased"

        base_model = BertModel.from_pretrained(base_model_name, token=hf_token)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence Classification task
            inference_mode=False,  # Enable training mode
            r=self.hparams.get('lora_r', 8),  # Low-rank matrix rank
            lora_alpha=self.hparams.get('lora_alpha', 32),  # Scaling factor
            lora_dropout=self.hparams.get('lora_dropout', 0.1)  # Dropout probability
        )
        
        self.model = get_peft_model(base_model, peft_config)
        self.criterion = CrossEntropyLoss()

        # Initialize MK-MMD with Gaussian Kernels
        self.kernels = [GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.0), GaussianKernel(alpha=2.0)]
        self.mk_mmd_loss = MultipleKernelMaximumMeanDiscrepancy(self.kernels, linear=False)

        # for debugging
        self.model.print_trainable_parameters()

    
    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if labels is not None:
            # Calculate loss if labels are provided
            loss = self.criterion(logits, labels)
            return loss, logits
        return logits

    def training_step(self, batch, batch_idx):
        # Get source and target data
        source_input_ids = batch['source_input_ids']
        source_attention_mask = batch['source_attention_mask']
        target_input_ids = batch['target_input_ids']
        target_attention_mask = batch['target_attention_mask']
        label_source = batch['label_source']

        # Perform forward pass for source domain
        loss_source, source_logits = self.forward(source_input_ids, source_attention_mask, label_source)

        # Perform forward pass for target domain (unlabeled)
        target_logits = self.forward(target_input_ids, target_attention_mask)

        # Extract features from the embeddings layer for MK-MMD
        source_features = self.model.get_input_embeddings()(source_input_ids)
        target_features = self.model.get_input_embeddings()(target_input_ids)

        # Compute MK-MMD loss
        mmd_loss = self.mk_mmd_loss(source_features, target_features)

        # Combine source classification loss with MK-MMD loss
        total_loss = loss_source + self.hparams['mmd_lambda'] * mmd_loss

        # Compute accuracy and F1 for source domain
        preds_source = torch.argmax(source_logits, dim=-1)
        acc_source = accuracy_score(label_source.cpu(), preds_source.cpu())
        f1_source = f1_score(label_source.cpu(), preds_source.cpu(), average='weighted')

        # Log metrics to wandb
        #wandb.log({
        #    "train_loss": total_loss.item(),
        #    "train_acc": acc_source,
        #    "train_f1": f1_source,
        #    "mmd_loss": mmd_loss.item(),
        #})

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['source_input_ids'], batch['source_attention_mask'], batch['label_source']
        loss, logits = self.forward(input_ids, attention_mask, labels)

        # Compute predicted labels and accuracy/F1
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')

        # Log validation loss, accuracy, and F1-score
        #wandb.log({
        #    "val_loss": loss.item(),
        #    "val_acc": acc,
        #    "val_f1": f1,
        #})

        return {'val_loss': loss.item(), 'val_acc': acc, 'val_f1': f1}    
    
    def test_step(self, batch, batch_idx, dataset_type='source'):
        """Evaluate on both source and target datasets during testing."""
        input_ids = batch[f'{dataset_type}_input_ids']
        attention_mask = batch[f'{dataset_type}_attention_mask']
        labels = batch[f'label_{dataset_type}']

        # Forward pass
        loss, logits = self.forward(input_ids, attention_mask, labels)

        # Calculate predictions and metrics
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')

        # Log metrics
        #wandb.log({
        #    f"{dataset_type}_test_loss": loss.item(),
        #    f"{dataset_type}_test_acc": acc,
        #    f"{dataset_type}_test_f1": f1,
        #})

        return {
            f'{dataset_type}_test_loss': loss.item(),
            f'{dataset_type}_test_acc': acc,
            f'{dataset_type}_test_f1': f1
        }

    def test_epoch_end(self, outputs):
        """Logs overall test metrics after all batches."""
        avg_source_loss = torch.tensor([x['source_test_loss'] for x in outputs]).mean()
        avg_source_acc = torch.tensor([x['source_test_acc'] for x in outputs]).mean()
        avg_source_f1 = torch.tensor([x['source_test_f1'] for x in outputs]).mean()

        avg_target_loss = torch.tensor([x['target_test_loss'] for x in outputs]).mean()
        avg_target_acc = torch.tensor([x['target_test_acc'] for x in outputs]).mean()
        avg_target_f1 = torch.tensor([x['target_test_f1'] for x in outputs]).mean()

        # Log final test metrics for source and target datasets
        #wandb.log({
        #    "avg_source_test_loss": avg_source_loss.item(),
        #    "avg_source_test_acc": avg_source_acc.item(),
        #    "avg_source_test_f1": avg_source_f1.item(),
        #    "avg_target_test_loss": avg_target_loss.item(),
        #    "avg_target_test_acc": avg_target_acc.item(),
        #    "avg_target_test_f1": avg_target_f1.item(),
        #})


    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer
