import pandas as pd
import os



def preprocess_and_split_wdc(input_folder):
    """
    Preprocesses and splits the WDC dataset into training, validation, and gold datasets by extracting and processing pairs with IDs from the validation set.
    
    :param input_folder: Path to the folder containing the dataset files, which includes subdirectories for training, validation, and gold data.
    :return: A dictionary where each key is a dataset name and each value is another dictionary containing the processed train, valid, and gold splits.
    """

    dataset_splits = {}

    for dataset in os.listdir(input_folder):
        dataset_splits[dataset] = {}
        # There are 3 files: train, valid, and gold
        # train and valid  are .json files containing a dataset with product pairs
        # valid is a .csv file that contains only the ids of the product pairs
        # I need to extract pairs with the same id from valid.csv from the training file and put them into a validation dataset
        # and eliminate the pairs with the same id from valid.csv from the training file
        
        # Load datasets
        train_df = pd.read_json(os.path.join(input_folder,  f'{dataset}/{dataset}_train/{dataset}_train_small.json.gz'), compression='gzip', lines=True)
        valid_df = pd.read_csv(os.path.join(input_folder, f'{dataset}/{dataset}_valid/{dataset}_valid_small.csv'))
        id_valid_set = valid_df["pair_id"].unique()
        gold_df = pd.read_json(os.path.join(input_folder, f'{dataset}/{dataset}_gs.json.gz'), compression='gzip', lines=True)
        
        # Filter pairs from the training DataFrame
        valid_pairs_df = train_df[train_df["pair_id"].isin(id_valid_set)].copy()
        
        # Remove the valid pairs from the training DataFrame
        train_pairs_df = train_df[~train_df["pair_id"].isin(id_valid_set)].copy()
        
        # Prepare the datasets
        dataset_splits[dataset]["train"] = train_pairs_df[["title_left", "title_right", "label"]].apply(lambda x: pd.Series([f'{x["title_left"]} [SEP] {x["title_right"]}', x["label"]]), axis=1)
        dataset_splits[dataset]["valid"] = valid_pairs_df[["title_left", "title_right", "label"]].apply(lambda x: pd.Series([f'{x["title_left"]} [SEP] {x["title_right"]}', x["label"]]), axis=1)
        dataset_splits[dataset]["gold"] = gold_df[["title_left", "title_right", "label"]].apply(lambda x: pd.Series([f'{x["title_left"]} [SEP] {x["title_right"]}', x["label"]]), axis=1)
        
        # Rename columns to "pairs" and "labels"
        for split in dataset_splits[dataset]:
            dataset_splits[dataset][split].columns = ["pairs", "labels"]
        
    return dataset_splits

def save_datasets(dataset_splits, output_folder):
    """
    Saves the processed datasets to CSV files in the specified output folder.
    
    :param dataset_splits: A dictionary containing the processed train, valid, and gold splits for each dataset.
    :param output_folder: Path to the folder where the processed datasets will be saved.
    """
    for dataset, splits in dataset_splits.items():
        # Create a directory for the dataset
        dataset_dir = os.path.join(output_folder, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save train, valid, and gold datasets to CSV files
        for split_name, split_data in splits.items():
            file_path = os.path.join(dataset_dir, f'{split_name}.csv')
            split_data.to_csv(file_path, index=False, header=["pairs", "labels"])
            print(f"Saved {split_name} dataset to {file_path}")


current_dir = os.getcwd()
main_dir = os.path.join(current_dir, "wdc")
output_dir = os.path.join(current_dir, "processed_datasets")


dataset_splits = preprocess_and_split_wdc(main_dir)
save_datasets(dataset_splits, output_dir)
