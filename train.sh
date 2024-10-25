#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --gpus=1
#SBATCH -p gpu
#SBATCH --time=10:00

# RICORDATI DI AGGIUNGERE DOPO SBATCH GPU A100 40GB

source .venv/bin/activate
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

export HUGGINGFACE_TOKEN="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"

# Define the source and target domains
SRC_DOMAINS=("computers")
TGT_DOMAINS=("cameras")

# Loop through source and target domains
for src in "${SRC_DOMAINS[@]}"; do
  for tgt in "${TGT_DOMAINS[@]}"; do
    # Skip if source and target domains are the same
    if [ "$src" != "$tgt" ]; then
      echo "Training model with source domain: $src and target domain: $tgt"

      # Pass source and target folders as arguments
      srun python train.py --src "$src" --tgt "$tgt" 

    fi
  done
done

