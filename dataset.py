from peft import get_peft_model
from sklearn.model_selection import train_test_split
from pathlib import Path


import pandas as pd
import os
import csv

from transformers import BertTokenizer

# add comments and description functions to this file

class DatasetProcessor:
    def __init__(self, dataset_name, tokenizer=None, max_seq_length=128):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = max_seq_length
        self.data = None


    def csv_to_list(self, file_path):
        pairs = []
        labels = []
        with open(file_path, 'r', encoding='latin') as file:
            reader = csv.reader(file)
            # skip header row
            next(reader)
            for row in reader:
                pairs.append(row[0])
                labels.append(row[1])
        return pairs, labels


    def load_and_split_dataset(self):
        # get current directory
        current_dir = os.path.dirname(__file__)
        
        # get path to dataset in .csv format
        dataset_path = os.path.abspath(os.path.join(current_dir, "data", self.dataset_name, self.dataset_name + ".csv"))
        
        x, y = self.csv_to_list(dataset_path)
        
        # Split the data into training and testing (80/20)
        # and then training data into training and validation (80% of the 80% the training data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
        
    
        self.data = {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test': (x_test, y_test)
        }


    def convert_to_features(self, dataset, max_seq_length, tokenizer, pad_token=0, cls_token='[CLS]', sep_token='[SEP]'):
        
        features = {'train': [], 'val': [], 'test': []}

        for key in dataset.keys():
            texts, labels = dataset[key]
            for text, label in zip(texts, labels):
                left, right = text.split(sep_token)[0], text.split(sep_token)[1]
                ltokens = tokenizer.tokenize(left)
                rtokens = tokenizer.tokenize(right)
            
                # Adjust lengths if too long
                more = len(ltokens) + len(rtokens) - max_seq_length + 3
                if more > 0:
                    if more < len(rtokens):  # Truncate right tokens
                        rtokens = rtokens[:len(rtokens) - more]
                    elif more < len(ltokens):  # Truncate left tokens
                        ltokens = ltokens[:len(ltokens) - more]
                    else:
                        # If the combined length is still too long, skip this example      
                        print("skipping example: ", texts)
                        continue
                    
                tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]
                segment_ids = [0] * (len(ltokens) + 2) + [1] * (len(rtokens) + 1)


                
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                
                # Padding
                padding_length = max_seq_length - len(input_ids)
                input_ids += [pad_token] * padding_length
                input_mask += [0] * padding_length
                segment_ids += [0] * padding_length
                
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                features[key].append({
                    'input_ids': input_ids,
                    'attention_mask': input_mask,
                    'segment_ids': segment_ids,
                    'labels': label
                })
        
        return features



if __name__ == "__main__":
    dp = DatasetProcessor(dataset_name="wa1")
    
    dp.load_and_split_dataset()
    
    features = dp.convert_to_features(dp.data, max_seq_length=128, tokenizer=dp.tokenizer)