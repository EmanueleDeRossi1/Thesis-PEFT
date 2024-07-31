import csv
import os

#input_file = r"C:\Users\emanu\Downloads\tableA (2).csv"

def extract_by_id(tableA, tableB, split_reader, output_file):
    """
    Extracts pairs from tableA and tableB based on IDs provided in split_reader and writes them to an output CSV file.
    
    :param tableA: Dictionary mapping IDs to strings from tableA.
    :param tableB: Dictionary mapping IDs to strings from tableB.
    :param split_reader: CSV reader object for the split file (test, train, or valid).
    :param output_file: Path to the output CSV file where results will be written.
    :return: The path to the output file.
    """
    with open(output_file, "w", encoding="utf-8") as csvfile:
        
        csvfile.write("pairs,labels\n")

        header = next(split_reader)

        for row in split_reader:
            instance_a = row[0]
            instance_b = row[1]
            label = row[2]
            
            # Concatenate the instances from tableA and tableB with [SEP] separator
            instance = tableA[instance_a] + " [SEP] " + tableB[instance_b]
            new_row = instance + "," + label
            # Write the new row to the output file
            csvfile.write(new_row + "\n")
            
    return output_file
        
        
    
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


def preprocess_and_split_deepmatcher(input_folder):
    """
    Processes and splits data for DeepMatcher by reading the input files and writing processed data to output files.
    
    :param input_folder: Path to the folder containing the input data files and where output files will be saved.
    """
    for folder in os.listdir(input_folder):
        data_folder = os.path.join(input_folder, folder, "exp_data")
        
        tableA_filename = os.path.join(data_folder, "tableA.csv")
        tableB_filename = os.path.join(data_folder, "tableB.csv")
        test_filename = os.path.join(data_folder, "test.csv")
        train_filename = os.path.join(data_folder, "train.csv")
        valid_filename = os.path.join(data_folder, "valid.csv")
        
        
        with open(tableA_filename, 'r', newline='', encoding='utf-8') as tableA_file, \
             open(tableB_filename, 'r', newline='', encoding='utf-8') as tableB_file, \
             open(test_filename, 'r', newline='', encoding='utf-8') as test_file, \
             open(train_filename, 'r', newline='', encoding='utf-8') as train_file, \
             open(valid_filename, 'r', newline='', encoding='utf-8') as valid_file:
            
            tableA_reader = csv.reader(tableA_file)
            tableB_reader = csv.reader(tableB_file)
            
            # convert reader in dict for easy access
            tableA = convert_in_dict(tableA_reader)
            tableB = convert_in_dict(tableB_reader)

            test_reader = csv.reader(test_file)
            train_reader = csv.reader(train_file)
            valid_reader = csv.reader(valid_file)
            
            output_folder = os.path.join(input_folder, "data deepmatcher processed", folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            extract_by_id(tableA, tableB, test_reader, os.path.join(output_folder, "test.csv"))
            extract_by_id(tableA, tableB, train_reader, os.path.join(output_folder, "train.csv"))
            extract_by_id(tableA, tableB, valid_reader, os.path.join(output_folder, "valid.csv"))




current_folder = os.getcwd()

input_folder = os.path.join(current_folder, "data deepmatcher")


preprocess_and_split_deepmatcher(input_folder)