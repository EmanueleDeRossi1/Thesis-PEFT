import csv
import os
from sklearn.model_selection import train_test_split
import pandas as pd


def convert_in_dict(reader):
    """
    Converts a CSV reader object into a dictionary where the first column is used as the key and the remaining columns are concatenated into a single string value.
    
    :param reader: CSV reader object for the table file.
    :return: A dictionary mapping the first column's values to concatenated string values of the remaining columns.
    """
    table = {}
    header = next(reader)
    
    for row in reader:
        id_value = row[0]
        # convert comma-separated values in a single string
        instance = " ".join(row[1:])
        table[id_value] = instance
    return table


def check_duplicate_id_zomato_yelp(folder):

    """
    Checks for duplicate IDs between two specific CSV files in a given folder and returns the set of duplicate IDs.
    
    :param folder: Path to the folder containing the CSV files "zomato yelp 1.csv" and "zomato yelp 2.csv".
    :return: A set of IDs that are present in both CSV files.
    """
    first_file_id = set()
    second_file_id = set()

    for csv_file in folder:
        first_csv = os.path.join(folder, "zomato yelp 1.csv")
        second_csv = os.path.join(folder, "zomato yelp 2.csv")
    
    with open(first_csv, "r") as file1:
        reader1 = csv.reader(file1)
        for _ in range(6):
            next(reader1, None)
        for row in reader1:
            first_file_id.add(row[0])   
            
    with open(second_csv, "r") as file2:
        reader2 = csv.reader(file2)
        for _ in range(6):
            next(reader2, None)

        for row in reader2:
            second_file_id.add(row[0])

    duplicates = first_file_id.intersection(second_file_id)
    return duplicates

# 3 id duplicates found across the 2 files
# changed by hand id duplicates in second file:
    # 2 -> 10628
    # 5720 -> 10629
    # 6670 -> 10630
    
# I then manually concatenated the two files

def preprocess_and_split_magellan(input_folder):
    """
    Preprocesses and splits Magellan dataset files by reading input CSV files, combining columns, and saving the processed data to new CSV files.
    
    :param input_folder: Path to the folder containing the input dataset files.
    """
    for file in os.listdir(input_folder):
        if "processed" not in file:
            csv_file_path = os.path.join(input_folder, file)
            
            with open(csv_file_path, mode='r', newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                
                # Extract column names dynamically, excluding .ID columns
                ltable_columns = [col for col in reader.fieldnames if col.startswith('ltable.') and not col.endswith('.ID')]
                rtable_columns = [col for col in reader.fieldnames if col.startswith('rtable.') and not col.endswith('.ID')]
                
                # Prepare to write to the output CSV file
                output_file = csv_file_path[:-4] + " processed.csv"
                
                with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
                    fieldnames = ['pairs', 'labels']
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for row in reader:
                        # Extract ltable and rtable values
                        ltable_values = [row[col] for col in ltable_columns]
                        rtable_values = [row[col] for col in rtable_columns]
                        
                        # Combine values into strings, ignoring empty values
                        ltable_string = ' '.join(value for value in ltable_values if value)
                        rtable_string = ' '.join(value for value in rtable_values if value)
                        
                        # Create combined string with separator
                        pairs_string = f"{ltable_string} [SEP] {rtable_string}"
                        
                        # Get the label from the last column
                        label = row['match_label']
                        
                        # Write to the output file
                        writer.writerow({'pairs': pairs_string, 'labels': label})
        
            print(f"Processed file saved to {output_file}")

def split_data(input_folder):
    """
    Splits the processed dataset files into training, validation, and test sets, and saves them into separate CSV files.
    
    :param input_folder: Path to the folder containing the processed dataset files.
    """

    for file in os.listdir(input_folder):
        if "processed" in file and file.endswith('.csv'):
            processed_file_path = os.path.join(input_folder, file)
            
            # Load the processed data
            data = pd.read_csv(processed_file_path)
            
            # Split the data
            train_ratio = 0.6
            validation_ratio = 0.2
            test_ratio = 0.2
            seed = 42
            
            train, test = train_test_split(data, test_size=test_ratio, random_state=seed)
            train, valid = train_test_split(train, test_size=validation_ratio / (1 - test_ratio), random_state=seed)
            
            # Define output folder name based on the processed file name
            dataset_name = os.path.splitext(file)[0]
            output_folder = os.path.join(input_folder, dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            
            # Save the splits
            train.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
            valid.to_csv(os.path.join(output_folder, 'valid.csv'), index=False)
            test.to_csv(os.path.join(output_folder, 'test.csv'), index=False)
            
            print(f"Data split and saved in {output_folder}")



current_folder = os.getcwd()


zomato_yelp_folder = os.path.join(current_folder, "zomato yelp")

duplicates = check_duplicate_id_zomato_yelp(zomato_yelp_folder)

input_folder = os.path.join(current_folder, "magellan datasets")

# preprocess_and_split_magellan(input_folder)

split_data(input_folder)

