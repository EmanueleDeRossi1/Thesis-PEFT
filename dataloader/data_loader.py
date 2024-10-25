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
        max_seq_length: int,
        phase: str,

    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.phase = phase
        self.uses_token_type_ids = "token_type_ids" in self.tokenizer.model_input_names

        self.source_df = pd.read_csv(source_filepath)

        # Ensure the source and target datasets are the same length during validation or testing
        if self.phase in ["validation", "test"]:
            # Shuffle both datasets before truncating
            self.source_df = self.source_df.sample(frac=1).reset_index(drop=True)
            self.target_df = pd.read_csv(target_filepath).sample(frac=1).reset_index(drop=True)

            # Truncate both datasets to the same minimum length
            min_length = min(len(self.source_df), len(self.target_df))
            self.source_df = self.source_df.iloc[:min_length].reset_index(drop=True)
            self.target_df = self.target_df.iloc[:min_length].reset_index(drop=True)
        else:
            self.target_df = pd.read_csv(target_filepath) if target_filepath else None


    def __len__(self):
        return len(self.source_df)
    
    def process_text(self, df, index):
        text = df.iloc[index]["pairs"]
        label = df.iloc[index]["labels"]
        sequences = text.split("[SEP]")
        left, right = sequences[0].strip(), sequences[1].strip()
        encoded = self.tokenizer(
            text=left, text_pair=right, max_length=self.max_seq_length,
            truncation=True, padding=self.padding
        )
        data = {
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"]),
            "label": torch.tensor(label, dtype=torch.long)
        }
        if self.uses_token_type_ids:
            data["token_type_ids"] = torch.tensor(encoded["token_type_ids"])
        return data

    

    def __getitem__(self, index):
        data = {"source": self.process_text(self.source_df, index)}
        
        # Process target data if available
        if self.target_df is not None:
            data["target"] = self.process_text(self.target_df, index)
        
        return data

class DataModuleSourceTarget(pl.LightningDataModule):
    def __init__(self, source_folder: str, target_folder: str, hparams: Dict[str, Any]):
        super(DataModuleSourceTarget, self).__init__()
        self.dataset_dir = hparams["dataset_dir"]
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.pretrained_model_name = hparams["pretrained_model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=True)
        self.padding = hparams["padding"]
        self.max_seq_length = hparams["max_seq_length"]
        self.batch_size = hparams["batch_size"]

        # self.train_dataset = None
        # self.val_dataset = None
        # self.test_dataset = None


    def train_dataloader(self):
        # Directly load datasets here for training
        source_train = os.path.join(self.dataset_dir, self.source_folder, "train.csv")
        target_train = os.path.join(self.dataset_dir, self.target_folder, "train.csv")
        source_dataset = SourceTargetDataset(source_filepath=source_train, target_filepath=None, 
                                             tokenizer=self.tokenizer, padding=self.padding, 
                                             max_seq_length=self.max_seq_length, phase="train")
        target_dataset = SourceTargetDataset(source_filepath=target_train, target_filepath=target_train,
                                             tokenizer=self.tokenizer, padding=self.padding, 
                                             max_seq_length=self.max_seq_length, phase="train")
        return DataLoader(source_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True),\
              DataLoader(target_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)

    def val_dataloader(self):
        # Directly load datasets here for validation
        source_val = os.path.join(self.dataset_dir, self.source_folder, "valid.csv")
        target_val = os.path.join(self.dataset_dir, self.target_folder, "valid.csv")
        val_dataset = SourceTargetDataset(
            source_filepath=source_val, target_filepath=target_val, tokenizer=self.tokenizer,
            padding=self.padding, max_seq_length=self.max_seq_length, phase="validation"
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        # Directly load datasets here for testing
        source_test = os.path.join(self.dataset_dir, self.source_folder, "test.csv")
        target_test = os.path.join(self.dataset_dir, self.target_folder, "test.csv")
        test_dataset = SourceTargetDataset(
            source_filepath=source_test, target_filepath=target_test, tokenizer=self.tokenizer,
            padding=self.padding, max_seq_length=self.max_seq_length, phase="test"
        )
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=1)
