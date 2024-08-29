import pytorch_lightning as pl
#from peft import get_peft_model
import os
import csv
from transformers import AutoTokenizer


def csv_to_list(file_path):
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

class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = {}

    def load_dataset(self):
        current_dir = os.path.dirname(__file__)
        base_path = os.path.abspath(os.path.join(current_dir, "data", self.dataset_name))
        
        # Define file paths for train, validation, and test datasets
        train_path = os.path.join(base_path, "train.csv")
        val_path = os.path.join(base_path, "valid.csv")
        test_path = os.path.join(base_path, "test.csv")
        
        # Load each dataset
        x_train, y_train = csv_to_list(train_path)
        x_val, y_val = csv_to_list(val_path)
        x_test, y_test = csv_to_list(test_path)
        
        # Store the data in the dictionary
        self.data = {
            'train': (x_train, y_train),
            'valid': (x_val, y_val),
            'test': (x_test, y_test)
        }
        
        
        
        
        
        
        
def check_if_pairs_too_long(loader, max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]'):

    pairs = loader[0]
    labels = loader[1]
    
    num_long_pairs = 0

    for pair, label in zip(pairs, labels):
        # add ER situation
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                num_long_pairs += 1
                if more <len(rtokens) : # remove excessively long string
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("too long!")
        if sep_token not in pair:
            print("no sep token found")
    if num_long_pairs > 0:
        print(f"    There are {num_long_pairs} pairs that exceed max. seq. length in this set")


if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data")
    for dataset in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, dataset)):
            # Instantiate DatasetLoader with a dataset name
            dataset_name = dataset # Replace with the actual dataset name
            loader = DatasetLoader(dataset_name)
            
            # Load the dataset
            loader.load_dataset()
            
            max_seq_length = 128
            tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased')
            print("Analyzing training set of ... ", dataset_name)
            check_if_pairs_too_long(loader.data['train'], max_seq_length=max_seq_length, tokenizer=tokenizer)
            print("Analyzing validation set of ... ", dataset_name)
            check_if_pairs_too_long(loader.data['valid'], max_seq_length=max_seq_length, tokenizer=tokenizer)
            print("Analyzing test set of ... ", dataset_name)
            check_if_pairs_too_long(loader.data['test'], max_seq_length=max_seq_length, tokenizer=tokenizer)
            
            # Access the loaded data
            #print(loader.data['train'])  # Example: print training data
