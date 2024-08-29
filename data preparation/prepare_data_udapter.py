import pandas as pd
import os
from sklearn.model_selection import train_test_split
import itertools

# Function to reorder and save data for domain adaptation tasks
def prepare_data_updater(list_of_tuples, main_folder, output_folder):
    """
    Reorders the data from different domains and saves them in a specific format for use in the UDAPTER model.
    
    For each domain pair, it loads the existing train, dev, and test sets, removes labels from the target domain's training set,
    and then saves the reordered datasets into separate CSV files.

    Parameters:
        list_of_tuples (list): List of tuples representing domain pairs.
        output_dataset_folder (str): Folder name for the output dataset.
        input_dataset_path (str): Path to the input dataset directory.
        output_dataset_path (str): Path to the output dataset directory.

    Returns:
        None
    """
    
    for pair in list_of_tuples:
        source = pair[0]
        target = pair[1]
        print(source, pair)

        source_folder = os.path.join(main_folder, source)
        target_folder = os.path.join(main_folder, target)

        
        source_train = pd.read_csv(os.path.join(source_folder, "train.csv"))
        source_dev = pd.read_csv(os.path.join(source_folder, "valid.csv"))
        source_test = pd.read_csv(os.path.join(source_folder, "test.csv"))
        
        target_train = pd.read_csv(os.path.join(target_folder, "train.csv"))
        target_dev = pd.read_csv(os.path.join(target_folder, "valid.csv"))
        target_test = pd.read_csv(os.path.join(target_folder, "test.csv"))

        
        
        # Drop the "labels" column in target_train
        target_train = target_train.drop(columns=["labels"])
        
        # Create output folder
        output_folder_dataset = os.path.join(output_folder, f"{source}_{target}")
        os.makedirs(output_folder_dataset, exist_ok=True)

        # Save data to CSV files
        source_train.to_csv(os.path.join(output_folder_dataset, "train_source.csv"), index=False)
        source_dev.to_csv(os.path.join(output_folder_dataset, "dev_source.csv"), index=False)
        source_test.to_csv(os.path.join(output_folder_dataset, "test_source.csv"), index=False)
        target_train.to_csv(os.path.join(output_folder_dataset, "target_unlabelled.csv"), index=False)
        target_dev.to_csv(os.path.join(output_folder_dataset, "dev_target.csv"), index=False)
        target_test.to_csv(os.path.join(output_folder_dataset, "test_target.csv"), index=False)


benchmark_datasets = [("wa1", "ab"),
                  ("ab", "wa1"),
                  ("ds", "da"),
                  ("da", "ds"),
                  ("fz", "dzy"),
                  ("dzy", "fz"),
                  ("ri", "ab"),
                  ("ri", "wa1"),   
                  ("ia", "da"), 
                  ("ia", "ds"), 
                  ("b2", "fz"), 
                  ("b2", "dzy")]

wdc_datasets = all_tuples = list(itertools.permutations({"computers", "cameras", "watches", "shoes"}, 2))

inverted_datasets = [
                  ("ab", "ri"),
                  ("wa1", "ri"),   
                  ("da", "ia"), 
                  ("ds", "ia"), 
                  ("fz", "b2"), 
                  ("dzy", "b2")]


main_folder = os.getcwd()

# Call functions to create datasets
prepare_data_updater(benchmark_datasets, r"C:\Users\emanu\Documents\GitHub\Thesis-Adapters\data", r"C:\Users\emanu\Documents\GitHub\Thesis-Adapters\data for udapter\benchmark")
prepare_data_updater(wdc_datasets, r"C:\Users\emanu\Documents\GitHub\Thesis-Adapters\data", r"C:\Users\emanu\Documents\GitHub\Thesis-Adapters\data for udapter\wdc")
prepare_data_updater(inverted_datasets, r"C:\Users\emanu\Documents\GitHub\Thesis-Adapters\data", r"C:\Users\emanu\Documents\GitHub\Thesis-Adapters\data for udapter\inverted_datasets")
