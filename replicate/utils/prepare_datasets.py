# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:30:02 2024
@author: emanu
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import itertools

# Function to split data and save to CSV files
def data_splitting(list_of_tuples, output_dataset_folder, input_dataset_path, output_dataset_path):
    """
    Splits the data into train, dev, and test sets for each domain pair and saves them to CSV files.

    Parameters:
        list_of_tuples (list): List of tuples representing domain pairs.
        output_dataset_folder (str): Folder name for the output dataset.
        input_dataset_path (str): Path to the input dataset directory.
        output_dataset_path (str): Path to the output dataset directory.

    Returns:
        None
    """
    for pair in list_of_tuples:
        # Load data for source and target domains
        source_filename = os.path.join(input_dataset_path, pair[0], pair[0] + ".csv")
        target_filename = os.path.join(input_dataset_path, pair[1], pair[1] + ".csv")
        source = pd.read_csv(source_filename)
        source = source.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data
        target = pd.read_csv(target_filename)
        target = target.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data
        
        # Split data for source domain
        source_train, source_temp = train_test_split(source, test_size=0.4, random_state=42)
        source_dev, source_test = train_test_split(source_temp, test_size=0.5, random_state=42)
        
        # Split data for target domain
        target_train, target_temp = train_test_split(target, test_size=0.4, random_state=42)
        target_dev, target_test = train_test_split(target_temp, test_size=0.5, random_state=42)
        
        # Drop the "labels" column in target_train
        target_train = target_train.drop(columns=["labels"])
        
        # Create output folder
        output_folder = os.path.join(output_dataset_path, output_dataset_folder, f"{pair[0]}_{pair[1]}")
        os.makedirs(output_folder, exist_ok=True)

        # Save data to CSV files
        source_train.to_csv(os.path.join(output_folder, "train_source.csv"), index=False)
        source_dev.to_csv(os.path.join(output_folder, "dev_source.csv"), index=False)
        source_test.to_csv(os.path.join(output_folder, "test_source.csv"), index=False)
        target_train.to_csv(os.path.join(output_folder, "target_unlabelled.csv"), index=False)
        target_dev.to_csv(os.path.join(output_folder, "dev_target.csv"), index=False)
        target_test.to_csv(os.path.join(output_folder, "test_target.csv"), index=False)

# Function to create benchmark datasets
def create_benchmark_datasets(input_dataset_path, output_dataset_path):
    """
    Creates benchmark datasets based on specified domain pairs.

    Parameters:
        input_dataset_path (str): Path to the input benchmark dataset directory.
        output_dataset_path (str): Path to the output benchmark dataset directory.

    Returns:
        None
    """
    similar_doms = [
        ("wa1", "ab"),
        ("ab", "wa1"),
        ("ds", "da"),
        ("da", "ds"),
        ("fz", "dzy"),
        ("dzy", "fz")
    ]

    diff_doms = [
        ("ri", "ab"),
        ("ri", "wa1"),   
        ("ia", "da"), 
        ("ia", "ds"), 
        ("b2", "fz"), 
        ("b2", "dzy")
    ]
    
    # Split data for similar domains
    data_splitting(similar_doms, "similar_domains", input_dataset_path, output_dataset_path)
    
    # Split data for different domains
    data_splitting(diff_doms, "different_domains", input_dataset_path, output_dataset_path)
    
# Function to create WDS datasets
def create_wds_datasets(input_dataset_path, output_dataset_path):
    """
    Creates WDS datasets.

    Parameters:
        input_dataset_path (str): Path to the input WDS dataset directory.
        output_dataset_path (str): Path to the output WDS dataset directory.

    Returns:
        None
    """
    wds_datasets = {"computers", "cameras", "watches", "shoes"}
    
    # Generate all possible tuples with distinct elements and preserve order
    all_tuples = list(itertools.permutations(wds_datasets, 2))
    
    # Split data for WDS datasets
    data_splitting(all_tuples, "wds", input_dataset_path, output_dataset_path)

# Call functions to create datasets
create_wds_datasets(r"C:\Users\emanu\Desktop\Thesis\original_data\wds", r"C:\Users\emanu\Desktop\Thesis\data")
create_benchmark_datasets(r"C:\Users\emanu\Desktop\Thesis\original_data\benchmark", r"C:\Users\emanu\Desktop\Thesis\data")
