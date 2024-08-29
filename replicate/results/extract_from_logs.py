# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:29:09 2024

@author: emanu
"""

import re
import os
import numpy as np
import pandas as pd


def extract_log_data(log_file_path):
    data = {}

    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        log_content = file.read()

        # Extract all instances of source-target dataset combinations and their corresponding seed
        source_target_seed_matches = list(re.finditer(r'The source-target datasets are: (\w+_\w+) with seed (\d+)', log_content))
        
        for idx, match in enumerate(source_target_seed_matches):
            source_target = match.group(1)
            seed = int(match.group(2))

            # Determine the end of the current match to the start of the next match or end of content
            start_pos = match.end()
            end_pos = source_target_seed_matches[idx + 1].start() if idx + 1 < len(source_target_seed_matches) else len(log_content)

            # Initialize the dictionary for this source-target combination
            data[(source_target, seed)] = {
                'f1_score': None,
                'training_time': None,
                'inference_time': None
            }

            # Extract the section of the log content relevant to the current source-target pair
            section = log_content[start_pos:end_pos]

            # Extract F1 score
            if 'DADER' in log_file_path:
                f1_score_match = re.search(r'The F1 score is: ([\d.]+)', section)
                if f1_score_match:
                    data[(source_target, seed)]['f1_score'] = float(f1_score_match.group(1))
            if 'UDAPTER' in log_file_path:
                # Extract F1 score, ensuring it's only saved if it's a number
                f1_score_match = re.search(r'wandb:          The F1 score is:\s+([\d.]+)', section)
                if f1_score_match:
                    f1_score = f1_score_match.group(1)
                    if f1_score:  # Ensure it's a valid number
                        data[(source_target, seed)]['f1_score'] = float(f1_score)


            # Extract training time
            training_time_match = re.search(r'The training time is: ([\d.]+)', section)
            if training_time_match:
                data[(source_target, seed)]['training_time'] = float(training_time_match.group(1))

            # Extract inference time
            inference_time_match = re.search(r'The inference time is: ([\d.]+)', section)
            if inference_time_match:
                data[(source_target, seed)]['inference_time'] = float(inference_time_match.group(1))

    return data

def calculate_stats_by_dataset(data):
    # Dictionary to hold stats for each source-target dataset
    dataset_stats = {}
    
    # Group data by source-target dataset
    for (source_target, seed), values in data.items():
        if source_target not in dataset_stats:
            dataset_stats[source_target] = {
                "inference_time": [],
                "training_time": [],
                "f1_score": []
            }
        
        # Append the values for the current seed
        dataset_stats[source_target]["inference_time"].append(values["inference_time"])
        dataset_stats[source_target]["training_time"].append(values["training_time"])
        dataset_stats[source_target]["f1_score"].append(values["f1_score"])

    # Calculate mean and standard deviation for each metric in each dataset
    for source_target, metrics in dataset_stats.items():

        metrics["mean_inference_time"] = np.mean(metrics["inference_time"])
        metrics["mean_training_time"] = np.mean(metrics["training_time"])
        metrics["mean_f1_score"] = np.mean(metrics["f1_score"])
        metrics["std_f1_score"] = np.std(metrics["f1_score"], ddof=1)  # Sample standard deviation
        metrics["std_inference_time"] = np.std(metrics["inference_time"], ddof=1) 
        metrics["std_training_time"] = np.std(metrics["training_time"], ddof=1) 

        # Clean up the lists to avoid redundancy in the final output
        del metrics["inference_time"]
        del metrics["training_time"]
        del metrics["f1_score"]

    return dataset_stats


if __name__ == '__main__':
    
    main_directory = r'C:\Users\emanu\Documents\GitHub\Thesis-Adapters\replicate2\log files'
    
    wdc_udapter_filename = os.path.join(main_directory, 'UDAPTER WDC.out')
    wdc_udapter_extracted = extract_log_data(wdc_udapter_filename)
    wdc_udapter_stats = calculate_stats_by_dataset(wdc_udapter_extracted)
    wdc_udapter_df = pd.DataFrame(wdc_udapter_stats).T
    
    benchmark_udapter_filename = os.path.join(main_directory, 'UDAPTER benchmark.out')
    benchmark_udapter_extracted = extract_log_data(benchmark_udapter_filename)
    benchmark_udapter_stats = calculate_stats_by_dataset(benchmark_udapter_extracted)
    benchmark_udapter_df = pd.DataFrame(benchmark_udapter_stats).T

    
    wdc_dader_filename = os.path.join(main_directory, 'DADER WDC.out')
    wdc_dader_extracted = extract_log_data(wdc_dader_filename)
    wdc_dader_stats = calculate_stats_by_dataset(wdc_dader_extracted)
    wdc_dader_df = pd.DataFrame(wdc_dader_stats).T


    benchmark_dader_filename = os.path.join(main_directory, 'DADER benchmark.out')
    benchmark_dader_extracted = extract_log_data(benchmark_dader_filename)
    benchmark_dader_stats = calculate_stats_by_dataset(benchmark_dader_extracted)
    benchmark_dader_df = pd.DataFrame(benchmark_dader_stats).T

    # now inverted
    inverted_udapter_filename = os.path.join(main_directory, 'UDAPTER inverted.out')
    inverted_udapter_extracted = extract_log_data(inverted_udapter_filename)
    inverted_udapter_stats = calculate_stats_by_dataset(inverted_udapter_extracted)
    inverted_udapter_df = pd.DataFrame(inverted_udapter_stats).T

    inverted_dader_filename = os.path.join(main_directory, 'DADER inverted.out')
    inverted_dader_extracted = extract_log_data(inverted_dader_filename)
    inverted_dader_stats = calculate_stats_by_dataset(inverted_dader_extracted)
    inverted_dader_df = pd.DataFrame(inverted_dader_stats).T
    
    
    
    data_order = [
    "wa1_ab", "ab_wa1", "ds_da", "da_ds", "dzy_fz", "fz_dzy", 
    "ri_ab", "ab_ri", "ri_wa1", "wa1_ri", "ia_da", "da_ia", 
    "ia_ds", "ds_ia", "b2_fz", "fz_b2", "b2_dzy", "dzy_b2", 
    "computers_watches", "watches_computers", "cameras_watches", 
    "watches_cameras", "shoes_watches", "watches_shoes", 
    "computers_shoes", "shoes_computers", "cameras_shoes", 
    "shoes_cameras", "computers_cameras", "cameras_computers"
]


    
    merged_udapter = {**benchmark_udapter_extracted, **inverted_udapter_extracted, **wdc_udapter_extracted}
    merged_dader = {**benchmark_dader_extracted, **inverted_dader_extracted, **wdc_dader_extracted}
    
    
    # Step 3: Convert the data to a pandas DataFrame
    rows = []
    for (pair_seed, metrics) in merged_dader.items():
        pair, seed = pair_seed  # Unpack the pair and seed correctly
        rows.append({
            "Pair": pair,
            "Seed": seed,
            "F1 Score": metrics['f1_score'],
            "Training Time": metrics['training_time'],
            "Inference Time": metrics['inference_time']
        })
    
    df = pd.DataFrame(rows)
    
    # Define the order list
    data_order = [
        "wa1_ab", "ab_wa1", "ds_da", "da_ds", "dzy_fz", "fz_dzy", 
        "ri_ab", "ab_ri", "ri_wa1", "wa1_ri", "ia_da", "da_ia", 
        "ia_ds", "ds_ia", "b2_fz", "fz_b2", "b2_dzy", "dzy_b2", 
        "computers_watches", "watches_computers", "cameras_watches", 
        "watches_cameras", "shoes_watches", "watches_shoes", 
        "computers_shoes", "shoes_computers", "cameras_shoes", 
        "shoes_cameras", "computers_cameras", "cameras_computers"
    ]
    
    # Step 4: Create a new column in df for sorting based on the data_order list
    df['Order'] = df['Pair'].map(lambda x: data_order.index(x) if x in data_order else len(data_order))
    
    # Step 5: Sort the DataFrame by this new 'Order' column
    df_sorted = df.sort_values(by='Order').drop(columns='Order')
    
    # Step 6: Save the sorted DataFrame to a CSV file
    df_sorted.to_csv('dader_results.csv', index=False)
    
    print(df_sorted)

    
    
    # Step 3: Convert the data to a pandas DataFrame
    rows = []
    for (pair_seed, metrics) in merged_udapter.items():
        pair, seed = pair_seed  # Unpack the pair and seed correctly
        rows.append({
            "Pair": pair,
            "Seed": seed,
            "F1 Score": metrics['f1_score'],
            "Training Time": metrics['training_time'],
            "Inference Time": metrics['inference_time']
        })
    
    df = pd.DataFrame(rows)
    
    # Define the order list
    data_order = [
        "wa1_ab", "ab_wa1", "ds_da", "da_ds", "dzy_fz", "fz_dzy", 
        "ri_ab", "ab_ri", "ri_wa1", "wa1_ri", "ia_da", "da_ia", 
        "ia_ds", "ds_ia", "b2_fz", "fz_b2", "b2_dzy", "dzy_b2", 
        "computers_watches", "watches_computers", "cameras_watches", 
        "watches_cameras", "shoes_watches", "watches_shoes", 
        "computers_shoes", "shoes_computers", "cameras_shoes", 
        "shoes_cameras", "computers_cameras", "cameras_computers"
    ]
    
    # Step 4: Create a new column in df for sorting based on the data_order list
    df['Order'] = df['Pair'].map(lambda x: data_order.index(x) if x in data_order else len(data_order))
    
    # Step 5: Sort the DataFrame by this new 'Order' column
    df_sorted = df.sort_values(by='Order').drop(columns='Order')
    
    # Step 6: Save the sorted DataFrame to a CSV file
    df_sorted.to_csv('udapter_results.csv', index=False)
    
    print(df_sorted)
