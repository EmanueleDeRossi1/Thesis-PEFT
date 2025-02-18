#!/bin/bash
#SBATCH --job-name=lora
#SBATCH -p gpu
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:a100_40gb:1


echo change to random 

source .venv/bin/activate
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

# Check if Slurm (srun) is available
if command -v srun &> /dev/null; then
    RUN_CMD="srun python"
else
    RUN_CMD="python"
fi

# Similar and Different Domains

# Define the source and target domains
SRC_DOMAINS=("wa1" "ab" "ds" "da" "dzy" "fz" "ri" "ab" "ri" "wa1" "ia" "da" "ia" "ds" "b2"  "fz" "b2" "dzy")
TRG_DOMAINS=("ab" "wa1" "da" "ds" "fz" "dzy" "ab" "ri" "wa1" "ri" "da" "ia" "ds" "ia" "fz"  "b2" "dzy" "b2")


MODEL="lora" # or "finetune" to finetune the full model or lora

# Loop through source and target domains
for seed in 3000 1000 42; do
  for i in $(seq 0 $((${#SRC_DOMAINS[@]} - 1))); do
    src=${SRC_DOMAINS[$i]}
    tgt=${TRG_DOMAINS[$i]}
    # Skip if source and target domains are the same
    if [ "$src" != "$tgt" ]; then
      echo "Training model: $MODEL with source domain: $src and target domain: $tgt"
      # Pass source and target folders as arguments, if --haparam_tuning is passed, hyperparameter tuning will be performed
      $RUN_CMD  train.py --src "$src" --tgt "$tgt" --seed $seed --model "$MODEL" #--hparam_tuning 
      if grep -q -- '--hparam_tuning' <<< "$@"; then
        echo "Hyperparameter tuning enabled"
      else
        echo "Hyperparameter tuning disabled"
      fi
    fi
  done
done


##############
# WDC Datasets
##############

# Define the source and target domains
SRC_DOMAINS=(computers cameras shoes computers cameras computers)
TRG_DOMAINS=(watches watches watches shoes shoes cameras)


MODEL="lora" # or "finetune" to finetune the full model or lora

# Loop through source and target domains
for seed in 3000 1000 42; do
  for i in $(seq 0 $((${#SRC_DOMAINS[@]} - 1))); do
    src=${SRC_DOMAINS[$i]}
    tgt=${TRG_DOMAINS[$i]}
    # Skip if source and target domains are the same
    if [ "$src" != "$tgt" ]; then
      echo "Training model: $MODEL with source domain: $src and target domain: $tgt"
      # Pass source and target folders as arguments, if --haparam_tuning is passed, hyperparameter tuning will be performed
      $RUN_CMD  train.py --src "$src" --tgt "$tgt" --seed $seed --model "$MODEL" #--hparam_tuning 
      # Now invert source and target domains to train also this combination
      src=${TRG_DOMAINS[$i]}
      tgt=${SRC_DOMAINS[$i]}
      $RUN_CMD  train.py --src "$src" --tgt "$tgt" --seed $seed --model "$MODEL" --hparam_tuning 
      if grep -q -- '--hparam_tuning' <<< "$@"; then
        echo "Hyperparameter tuning enabled"
      else
        echo "Hyperparameter tuning disabled"
      fi
    fi
  done
done




# FOR DOING HYPERPARAMETER TUNING

# SRC_DOMAINS=("computers")
# TGT_DOMAINS=("watches")

# MODEL="lora" # or "finetune" to finetune the full model or lora
# seed=42
# for src in "${SRC_DOMAINS[@]}"; do
#   for tgt in "${TGT_DOMAINS[@]}"; do
#     # Skip if source and target domains are the same
#     if [ "$src" != "$tgt" ]; then
#       echo "Training model: $MODEL with source domain: $src and target domain: $tgt"
#       # Pass source and target folders as arguments, if --haparam_tuning is passed, hyperparameter tuning will be performed
#       $RUN_CMD  train.py --src "$src" --tgt "$tgt" --seed $seed --model "$MODEL" --hparam_tuning 
#       if grep -q -- '--hparam_tuning' <<< "$@"; then
#         echo "Hyperparameter tuning enabled"
#       fi
#     fi
#   done
# done

