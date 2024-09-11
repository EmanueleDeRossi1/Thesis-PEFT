#!/bin/bash

export HUGGINGFACE_TOKEN="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"

# Define the source and target domains
SRC_DOMAINS=("cameras")
TRG_DOMAINS=("computers")

# Loop through source and target domains
for src in "${SRC_DOMAINS[@]}"; do
  for trg in "${TRG_DOMAINS[@]}"; do
    # Skip if source and target domains are the same
    if [ "$src" != "$trg" ]; then
      echo "Training model with source domain: $src and target domain: $trg"

      # Update config.yaml with the correct source and target domains
      sed -i "s|source_folder: .*|source_folder: \"$src\"|" config.yaml
      sed -i "s|target_folder: .*|target_folder: \"$trg\"|" config.yaml

      # Run the training script for the current source-target domain pair
      python train.py

    fi
  done
done

