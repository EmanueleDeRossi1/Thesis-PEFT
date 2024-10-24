from typing import Optional, Dict, Any
import os
import torch
import pytorch_lightning as pl
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# class SourceTargetDataset(Dataset):
#     def __init__(
#         self, 
#         source_filepath: str, 
#         target_filepath: str, 
#         tokenizer: AutoTokenizer, 
#         padding: bool, 
#         max_seq_length: int
#     ):
#         self.tokenizer = tokenizer
#         self.max_seq_length = max_seq_length
#         self.padding = padding

#         self.source_df = pd.read_csv(source_filepath)
#         self.target_df = pd.read_csv(target_filepath)
#         # Check if the tokenizer requires token_type_ids (e.g., BERT does, RoBERTa doesn't)
#         self.uses_token_type_ids = "token_type_ids" in self.tokenizer.model_input_names


    # def __len__(self):
    #     return min(len(self.source_df), len(self.target_df))

    # def __getitem__(self, index):
    #     source_text = self.source_df.iloc[index]["pairs"]
    #     label_source = self.source_df.iloc[index]["labels"]

    #     # Split the source text into two sequences based on the [SEP] token
    #     sequences_source = source_text.split("[SEP]")

    #     # Take the left and right instances of the pairs
    #     left_source = sequences_source[0].strip()
    #     right_source = sequences_source[1].strip()

    #     encoded_source = self.tokenizer(
    #         text=left_source,
    #         text_pair=right_source,
    #         max_length=self.max_seq_length,
    #         truncation=True,
    #         padding=self.padding
    #     )
    #     source_input_ids = encoded_source["input_ids"]
    #     source_attention_mask = encoded_source["attention_mask"]
    #     # Conditionally include token_type_ids if required
    #     if self.uses_token_type_ids:
    #         source_token_type_ids = encoded_source["token_type_ids"]

        

    #     target_text = self.target_df.iloc[index]["pairs"]
    #     label_target = self.target_df.iloc[index]["labels"]

    #     # Split the source text into two sequences based on the [SEP] token
    #     sequences_target = target_text.split("[SEP]")

    #     # Take the left and right instances of the pairs
    #     left_target = sequences_target[0].strip()
    #     right_target = sequences_target[1].strip()

    #     encoded_target = self.tokenizer(
    #         text=left_target,
    #         text_pair=right_target,
    #         max_length=self.max_seq_length,
    #         truncation=True,
    #         padding=self.padding
    #     )
    #     target_input_ids = encoded_target["input_ids"]
    #     target_attention_mask = encoded_target["attention_mask"]
    #     # Conditionally include token_type_ids if required
    #     if self.uses_token_type_ids:
    #         target_token_type_ids = encoded_target["token_type_ids"]

    #     data= {
    #         "source_input_ids": torch.tensor(source_input_ids),
    #         "source_attention_mask": torch.tensor(source_attention_mask),
    #         "target_input_ids": torch.tensor(target_input_ids),
    #         "target_attention_mask": torch.tensor(target_attention_mask),
    #         "label_source": torch.tensor(label_source, dtype=torch.long),
    #         "label_target": torch.tensor(label_target, dtype=torch.long)
    #         }
        
    #             # Conditionally add token_type_ids if the model requires them
    #     if self.uses_token_type_ids:
    #         data["source_token_type_ids"] = torch.tensor(source_token_type_ids)
    #         data["target_token_type_ids"] = torch.tensor(target_token_type_ids)
    #     return data


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

        self.source_df = pd.read_csv(source_filepath)
        self.target_df = pd.read_csv(target_filepath)
        # Check if the tokenizer requires token_type_ids (e.g., BERT does, RoBERTa doesn't)
        self.uses_token_type_ids = "token_type_ids" in self.tokenizer.model_input_names

        # Ensure the source and target datasets are the same length during validation or testing
        if self.phase in ["validation", "test"]:
            # Shuffle both datasets before truncating
            self.source_df = self.source_df.sample(frac=1).reset_index(drop=True)
            self.target_df = self.target_df.sample(frac=1).reset_index(drop=True)

            # Truncate both datasets to the same minimum length
            min_length = min(len(self.source_df), len(self.target_df))
            self.source_df = self.source_df.iloc[:min_length].reset_index(drop=True)
            self.target_df = self.target_df.iloc[:min_length].reset_index(drop=True)



    def __len__(self):
        return len(self.source_df)
    

    def __getitem__(self, index):
        source_text = self.source_df.iloc[index]["pairs"]
        label_source = self.source_df.iloc[index]["labels"]

        # Split the source text into two sequences based on the [SEP] token
        sequences_source = source_text.split("[SEP]")

        # Take the left and right instances of the pairs
        left_source = sequences_source[0].strip()
        right_source = sequences_source[1].strip()

        encoded_source = self.tokenizer(
            text=left_source,
            text_pair=right_source,
            max_length=self.max_seq_length,
            truncation=True,
            padding=self.padding
        )
        source_input_ids = encoded_source["input_ids"]
        source_attention_mask = encoded_source["attention_mask"]
        # Conditionally include token_type_ids if required
        if self.uses_token_type_ids:
            source_token_type_ids = encoded_source["token_type_ids"]

        # Store source data into data dictionary
        data = {
        "source_input_ids": torch.tensor(source_input_ids),
        "source_attention_mask": torch.tensor(source_attention_mask),
        "label_source": torch.tensor(label_source, dtype=torch.long)
        }

        # If the model requires token_type_ids, add them
        if self.uses_token_type_ids:
            data["source_token_type_ids"] = torch.tensor(source_token_type_ids)

        # This condition will always be true during validation and testing, since I made sure that 
        # during validation and testing the source and target dataset have the same length
        if index < len(self.target_df):
            target_text = self.target_df.iloc[index]["pairs"]
            label_target = self.target_df.iloc[index]["labels"]

            # Split the source text into two sequences based on the [SEP] token
            sequences_target = target_text.split("[SEP]")

            # Take the left and right instances of the pairs
            left_target = sequences_target[0].strip()
            right_target = sequences_target[1].strip()

            encoded_target = self.tokenizer(
                text=left_target,
                text_pair=right_target,
                max_length=self.max_seq_length,
                truncation=True,
                padding=self.padding
            )
            target_input_ids = encoded_target["input_ids"]
            target_attention_mask = encoded_target["attention_mask"]
            # If the model requires token_type_ids, add them
            if self.uses_token_type_ids:
                target_token_type_ids = encoded_target["token_type_ids"]

            # Add target data to the dictionary
            data.update({
                "target_input_ids": torch.tensor(target_input_ids),
                "target_attention_mask": torch.tensor(target_attention_mask),
                "label_target": torch.tensor(label_target, dtype=torch.long)
            })

            # Conditionally add token_type_ids for the target if the model requires them
            if self.uses_token_type_ids:
                data["target_token_type_ids"] = torch.tensor(target_token_type_ids)
        
        return data

class DataModuleSourceTarget(pl.LightningDataModule):
    def __init__(self, source_folder: str, target_folder: str, hparams: Dict[str, Any]):
        super(DataModuleSourceTarget, self).__init__()

        self.dataset_dir = hparams["dataset_dir"]
        self.source_folder = source_folder
        self.target_folder = target_folder
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

        # self.train_dataset = SourceTargetDataset(
        #     source_filepath=source_train_path,
        #     target_filepath=target_train_path,
        #     tokenizer=self.tokenizer,
        #     padding=self.padding,
        #     max_seq_length=self.max_seq_length,
        #     phase="train"
        # )

        # Separate source and target datasets for training
        self.train_source_dataset = SourceTargetDataset(
            source_filepath=source_train_path,
            target_filepath=target_train_path,  # This can be ignored in this case
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
            phase="train"
        )

        self.train_target_dataset = SourceTargetDataset(
            source_filepath=target_train_path,
            target_filepath=target_train_path,  # Use the target data here
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
            phase="train"
        )

        self.val_dataset = SourceTargetDataset(
            source_filepath=source_val_path,
            target_filepath=target_val_path,
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
            phase="validation"
        )
        self.test_dataset = SourceTargetDataset(
            source_filepath=source_test_path,
            target_filepath=target_test_path,
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_length=self.max_seq_length,
            phase="test"
        )

    def setup(self, stage: Optional[str] = None):
        if self.train_source_dataset is None or self.train_target_dataset is None:
            self.setup_datasets()

    def train_dataloader(self):
        # Create separate dataloaders for source and target during training
        source_loader = DataLoader(self.train_source_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
        target_loader = DataLoader(self.train_target_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
        return source_loader, target_loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)