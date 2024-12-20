# Training the Model

The `train.sh` script is used to train models for domain adaptation tasks using various source and target domain pairs. It supports both LoRA-based fine-tuning and full-model fine-tuning, allowing you to experiment with parameter-efficient techniques or traditional methods.

## How It Works

The script:
1. Loops through source and target dataset experimented in the paper, using 3 random seeds 
4. Use the `--hparam_tuning` flag to enable hyperparameter tuning.

## Steps to Train the Model

1. **Set Up Your Environment**:

2. *Run the train.sh script using the SLURM scheduler*

```
sbatch train.sh
```