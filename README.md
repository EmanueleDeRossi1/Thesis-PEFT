# Training the Model

The `train.sh` script is used to train models for domain adaptation tasks using various source and target domain pairs. It supports both LoRA-based fine-tuning and full-model fine-tuning, allowing you to experiment with parameter-efficient techniques or traditional methods.

## How It Works

The script:
1. Loops through source and target dataset experimented in the paper, using 3 random seeds 
4. Use the `--hparam_tuning` flag to enable hyperparameter tuning.

## Steps to Train the Model

1. **Set Up Your Environment**:

...to-do...

2. **Run the train.sh script using the SLURM scheduler**

```
sbatch train.sh
```


## Repository Structure

### Folders
- **data preparation**: Scripts used for preparing and cleaning the datasets
- **data**: Contains preprocessed datasets used for experiments.
- **dataloader**: dataloader for training.
- **divergences**: Contains the code from https://github.com/declare-lab/domadapter/tree/main/domadapter to measure MK-MMD loss.
- **modules**: Lora and Fine-Tune on the whole model.
- **paper summaries**: Summaries of related research papers, for personal use.
- **results**: Stores results from LoRA experiments (f1, training and inference time).

### Files
- **config.yaml**: Configuration file for training when hparam tuning flag is **not** activated. 
- **ft_sweep_config.yaml**: Configuration for sweeps (hparam tuning) for full fine-tune model.
- **lora_sweep_config.yaml**: Configuration for sweeps (hparam tuning) for Lora.
- **train.py**: Main training script for running the experiments.
- **train.sh**: Bash script for submitting jobs to a SLURM cluster for training across various domain pairs.
