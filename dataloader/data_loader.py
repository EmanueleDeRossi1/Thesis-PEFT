from typing import Optional, Dict, Any
import os
import torch
import pytorch_lightning as pl
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class SourceTargetDataset(Dataset):
    def __init__(
        self, 
        source_filepath: str, 
        target_filepath: str, 
        tokenizer: AutoTokenizer, 
        padding: bool, 
        max_seq_length: int
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding

        self.source_df = pd.read_csv(source_filepath)
        self.target_df = pd.read_csv(target_filepath)

    def __len__(self):
        return min(len(self.source_df), len(self.target_df))

    def __getitem__(self, index):
        source_text = self.source_df.iloc[index]["pairs"]
        label_source = self.source_df.iloc[index]["labels"]

        encoded_source = self.tokenizer(
            source_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding=self.padding
        )
        source_input_ids = encoded_source["input_ids"]
        source_attention_mask = encoded_source["attention_mask"]

        target_text = self.target_df.iloc[index]["pairs"]
        label_target = self.target_df.iloc[index]["labels"]

        encoded_target = self.tokenizer(
            target_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding=self.padding
        )
        target_input_ids = encoded_target["input_ids"]
        target_attention_mask = encoded_target["attention_mask"]


        return {
            "source_input_ids": torch.tensor(source_input_ids),
            "source_attention_mask": torch.tensor(source_attention_mask),
            "target_input_ids": torch.tensor(target_input_ids),
            "target_attention_mask": torch.tensor(target_attention_mask),
            "label_source": torch.tensor(label_source, dtype=torch.long),
            "label_target": torch.tensor(label_target, dtype=torch.long),
            }


class DataModuleSourceTarget(pl.LightningDataModule):
    def __init__(self, hparams: Dict[str, Any]):
        super(DataModuleSourceTarget, self).__init__()

        self.dataset_dir = hparams["dataset_dir"]
        self.source_folder = hparams["source_folder"]
        self.target_folder = hparams["target_folder"]
        self.pretrained_model_name = hparams["pretrained_model_name"]
        self.padding = hparams["padding"]
        self.max_seq_length = hparams["max_seq_length"]
        self.batch_size = hparams["batch_size"]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name, use_fast=True
        )

    def prepare_data(self):
        self.setup_datasets()

    def setup_datasets(self):
        source_train_path = os.path.join(self.dataset_dir, self.source_folder, "train.csv")
        source_val_path = os.path.join(self.dataset_dir, self.source_folder, "valid.csv")
        source_test_path = os.path.join(self.dataset_dir, self.source_folder, "test.csv")

        target_train_path = os.path.join(self.dataset_dir, self.target_folder, "train.csv")
        target_val_path = os.path.join(self.dataset_dir, self.target_folder, "valid.csv")
        target_test_path = os.path.join(self.dataset_dir, self.target_folder, "test.csv")

        self.train_dataset = SourceTargetDataset(
            source_filepath=source_train_path,
            target_filepath=target_train_path,
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length
        )
        self.val_dataset = SourceTargetDataset(
            source_filepath=source_val_path,
            target_filepath=target_val_path,
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length
        )
        self.test_dataset = SourceTargetDataset(
            source_filepath=source_test_path,
            target_filepath=target_test_path,
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length
        )

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is None:
            self.setup_datasets()

        if stage == "fit":
            self.train_dataset = self.train_dataset
            self.val_dataset = self.val_dataset
        elif stage == "test":
            self.test_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)


# Example usage
if __name__ == "__main__":
    
    dataset_dir = os.path.join(os.getcwd(), "data")
    
    
    hparams = {
        "dataset_dir": dataset_dir,         
        "source_folder": "ab",               
        "target_folder": "wa1",            
        "pretrained_model_name": "bert-base-uncased",
        "padding": "max_length",
        "max_seq_length": 256,
        "batch_size": 32,
    }

    ab_wa1_dm = DataModuleSourceTarget(hparams)
    ab_wa1_dm.prepare_data()
    
    
    train_loader = ab_wa1_dm.train_dataloader()
    for batch in train_loader:
        print("Batch:")
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
        break


