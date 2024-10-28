import pandas as pd
import os

def check_order_and_shuffle(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the directory path of the CSV file
    directory_path = os.path.dirname(csv_path)
    
    # Check if the 'labels' column exists
    if 'labels' not in df.columns:
        print(f"The 'labels' column does not exist in the CSV file at {directory_path}.")
        return
    
    # Get the labels column as a list
    labels = df['labels'].tolist()
    
    # Check if the labels are in order (all 1s followed by all 0s)
    first_zero_index = next((i for i, label in enumerate(labels) if label == 0), None)
    if first_zero_index is not None and all(label == 0 for label in labels[first_zero_index:]):
        print(f"The 'labels' column is ordered (all 1s followed by all 0s) in {directory_path}. Shuffling the dataset...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(csv_path, index=False)  # Save the shuffled dataset
        print(f"Dataset at {directory_path} shuffled and saved.")
    else:
        print(f"The 'labels' column in {directory_path} is not ordered. Dataset not shuffled.")


data_dir = os.path.join(os.getcwd(), 'data')
# CHANGE THE DATA IN DADER
data_dir = '/home/derossi/DADER/data_copy'

for dataset_dir in os.listdir(data_dir):
    dataset_dir = os.path.join(data_dir, dataset_dir)  # Corrected to use dataset_dir
    if os.path.isdir(dataset_dir):  # Check if it's a directory
        for csv_file in os.listdir(dataset_dir):  # List files in the subdirectory
            csv_path = os.path.join(dataset_dir, csv_file)  # Construct full path to the file
            if os.path.isfile(csv_path) and csv_path.endswith('.csv'):  # Check if it's a .csv file
                check_order_and_shuffle(csv_path)  # Process the .csv file
