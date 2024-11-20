from typing import Optional, Dict, Any
import os
import torch
import pytorch_lightning as pl
import pandas as pd
import random
from itertools import zip_longest

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.dataloader import default_collate

class SourceTargetDataset(Dataset):
    def __init__(
        self, 
        source_filepath, 
        target_filepath, 
        tokenizer: AutoTokenizer, 
        padding: bool, 
        max_seq_length: int,
        phase: str,
        num_instances: Optional[int] = None
        ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.phase = phase
        self.uses_token_type_ids = "token_type_ids" in self.tokenizer.model_input_names
        self.seed_counter = 0  # Initialize the seed counter for each epoch

        self.source_df = pd.read_csv(source_filepath)
        self.target_df = pd.read_csv(target_filepath)

        # Limit the number of instances if num_instances is specified and phase is "train"
        print("The phase is: ", phase)
        print("The number of instances is: ", num_instances, type(num_instances))
        if phase == "train" and num_instances is not None:
            print("The phase inside if is: ", phase)
            print("The number of instances inside if is: ", num_instances, type(num_instances))
            self.source_df = self.source_df.head(num_instances)

        # Define pairing based on the phase
        if self.phase == "train":
            # Use all source data, allowing target data to be None if shorter
            self.data_pairs = list(
                 zip_longest(self.source_df.iterrows(), self.target_df[:len(self.source_df)].iterrows(), fillvalue=None)
            )
        else:
            # Use all target data, allowing source data to be None if shorter
            self.data_pairs = list(
                zip_longest(self.source_df[:len(self.target_df)].iterrows(), self.target_df.iterrows(), fillvalue=None)
            )
        

    def shuffle_source_data(self):
        self.source_df = self.source_df.sample(frac=1, random_state=self.seed_counter).reset_index(drop=True)
        self.data_pairs = list(
            zip_longest(self.source_df.iterrows(), self.target_df[:len(self.source_df)].iterrows(), fillvalue=None)
        )
        self.seed_counter += 1  # Increment the seed counter for the next epoch



    def __len__(self):
            return len(self.data_pairs)

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

        # Get the pair from data_pairs
        source_pair = self.data_pairs[index][0]
        target_pair = self.data_pairs[index][1]

        item = {}

        # source_pair is a tuple (index, row)
        if source_pair is not None:
            _, source_row = source_pair
            source_data = self.process_text(self.source_df, source_row.name)
            item["source"] = source_data
        else:
            item["source"] = None

        # target_pair is a tuple (index, row)
        if target_pair is not None:
            _, target_row = target_pair
            target_data = self.process_text(self.target_df, target_row.name)
            item["target"] = target_data
        else:
            item["target"] = None
            
        return item




class DataModuleSourceTarget(pl.LightningDataModule):
    def __init__(self, source_folder: str, target_folder: str, hparams: Dict[str, Any]):
        super(DataModuleSourceTarget, self).__init__()
        self.dataset_dir = hparams["dataset_dir"]
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.num_instances = hparams["num_instances"]
        self.pretrained_model_name = hparams["pretrained_model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=True)
        self.padding = hparams["padding"]
        self.max_seq_length = hparams["max_seq_length"]
        self.batch_size = hparams["batch_size"]

    def setup(self, stage: Optional[str] = None):
        source_train = os.path.join(self.dataset_dir, self.source_folder, "train.csv")
        target_train = os.path.join(self.dataset_dir, self.target_folder, "train.csv")
        source_val = os.path.join(self.dataset_dir, self.source_folder, "valid.csv")
        target_val = os.path.join(self.dataset_dir, self.target_folder, "valid.csv")
        source_test = os.path.join(self.dataset_dir, self.source_folder, "test.csv")
        target_test = os.path.join(self.dataset_dir, self.target_folder, "test.csv")

        # Load datasets for different phases
        if stage == "fit" or stage is None:
            self.train_dataset = SourceTargetDataset(
            source_filepath=source_train, target_filepath=target_train, tokenizer=self.tokenizer, 
            padding=self.padding, max_seq_length=self.max_seq_length, phase="train", num_instances=self.num_instances
            )
            self.val_dataset = SourceTargetDataset(
            source_filepath=source_val, target_filepath=target_val, tokenizer=self.tokenizer, 
            padding=self.padding, max_seq_length=self.max_seq_length, phase="validation"
            )
        if stage == "test" or stage is None:
            self.test_dataset = SourceTargetDataset(
            source_filepath=source_test, target_filepath=target_test, tokenizer=self.tokenizer, 
            padding=self.padding, max_seq_length=self.max_seq_length, phase="test"
            )


    def train_dataloader(self):
        self.train_dataset.shuffle_source_data()
        print("The lenght of the train dataset is: ", len(self.train_dataset))
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False, collate_fn=custom_collate_fn)


def custom_collate_fn(batch):
    # Separate source and target data
    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]
        
    collated_batch = {}

    if any(source is None for source in sources):
        collated_batch["source"] = None
    else:
        collated_batch["source"] = default_collate(sources)
        
    if any(target is None for target in targets):
        collated_batch["target"] = None
    else:
        collated_batch["target"] = default_collate(targets)
    
    return collated_batch


