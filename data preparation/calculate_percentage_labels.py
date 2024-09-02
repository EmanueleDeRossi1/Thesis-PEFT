# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:01:13 2024

@author: emanu
"""

print("something")

import pandas as pd
import os

import matplotlib.pyplot as plt

print("something")


def calculate_label_percentages(df):
    label_counts = df['labels'].value_counts(normalize=True) * 100
    return label_counts.sort_index()



def see_splits(input_folder):
    datasets = os.listdir(input_folder)
    labels = set()
    
    # Initialize a dictionary to store the DataFrame for each dataset
    dataset_splits = {}

    # Iterate over each dataset in the input folder
    for dataset in datasets:
        dataset_path = os.path.join(input_folder, dataset)
        
        if os.path.isdir(dataset_path):  # Ensure it's a directory
            # Read train, test, and valid CSV files
            train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
            test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
            valid_df = pd.read_csv(os.path.join(dataset_path, 'valid.csv'))

            # Calculate label percentages for each split
            train_percentages = calculate_label_percentages(train_df)
            test_percentages = calculate_label_percentages(test_df)
            valid_percentages = calculate_label_percentages(valid_df)
            
            # Ensure all labels are included
            all_labels = sorted(set(train_percentages.index) | set(test_percentages.index) | set(valid_percentages.index))
            labels.update(all_labels)

            # Store the percentages in the dataset_splits dictionary
            dataset_splits[dataset] = pd.DataFrame({
                'train': train_percentages.reindex(all_labels, fill_value=0),
                'test': test_percentages.reindex(all_labels, fill_value=0),
                'valid': valid_percentages.reindex(all_labels, fill_value=0),
            })
    # Prepare DataFrames with multi-level columns
    df_list = []
    for dataset, df in dataset_splits.items():
        # Convert column names to MultiIndex
        df.columns = pd.MultiIndex.from_product([[dataset], df.columns])
        df_list.append(df)
    
    # Concatenate all DataFrames along the columns
    df_combined = pd.concat(df_list, axis=1)

    return df_combined


current_dir = os.getcwd()
main_dir = os.path.join(current_dir, "data")
result_df = see_splits(main_dir)
