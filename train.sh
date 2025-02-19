#!/bin/bash
#SBATCH --job-name=lora
#SBATCH -p gpu
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:a100_40gb:1


# Check if Slurm (srun) is available
if command -v srun &> /dev/null; then
    RUN_CMD="srun python3"
else
    RUN_CMD="python3"
fi

# Default values
ENABLE_HPARAM_TUNING="false"
MODEL="lora"  # Default model

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hparam-tuning) ENABLE_HPARAM_TUNING="true";;
        --model) MODEL="$2"; shift;;  # Assign the model name from argument
        *) echo "Unknown parameter: $1"; exit 1;;
    esac
    shift
done


declare -A DATASETS

# Similar and Different Domains
DATASETS["similar_different"]="wa1:ab ab:wa1 ds:da da:ds dzy:fz fz:dzy ri:ab ab:ri ri:wa1 wa1:ri ia:da da:ia ia:ds ds:ia b2:fz fz:b2 b2:dzy dzy:b2"

# WDC Datasets (bidirectional)
DATASETS["wdc"]="computers:watches watches:computers cameras:shoes shoes:cameras computers:cameras cameras:computers"


# Function to run training
run_training() {
    local src=$1
    local tgt=$2
    local seed=$3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training model: $MODEL | Source: $src | Target: $tgt | Seed: $seed"

    CMD="$RUN_CMD src/training/trainer.py --src \"$src\" --tgt \"$tgt\" --seed $seed --model \"$MODEL\""
    
    if [ "$ENABLE_HPARAM_TUNING" = "true" ]; then
        CMD="$CMD --hparam_tuning"
    fi

    echo "Executing: $CMD"
    $CMD &> logs/${src}_${tgt}_${seed}.log 2>&1
}

# Loop through datasets
for dataset in "${!DATASETS[@]}"; do
    echo "Processing dataset group: $dataset"
    for pair in ${DATASETS[$dataset]}; do
        src=${pair%%:*}
        tgt=${pair##*:}

        for seed in 3000 1000 42; do
            if [ "$src" != "$tgt" ]; then
                run_training "$src" "$tgt" "$seed"
            fi
        done
    done
done

echo "All training jobs completed!"