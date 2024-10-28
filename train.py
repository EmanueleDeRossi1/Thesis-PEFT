import argparse
import os
import yaml
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from modules.lora import LoRA_module 

from modules.finetune_task import FineTuneTask
from modules.finetune_task_divergence import FineTuneTaskDivergence

from dataloader.data_loader import DataModuleSourceTarget

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def load_hparams(yaml_file):
    """Load hyperparameters from a YAML file."""
    with open(yaml_file, 'r') as stream:
        hparams = yaml.safe_load(stream)
    return hparams

if __name__ == "__main__":
    # Argument parsing for source and target folders
    parser = argparse.ArgumentParser(description="Training script for domain adaptation")
    parser.add_argument('--src', type=str, required=True, help="Source dataset folder")
    parser.add_argument('--tgt', type=str, required=True, help="Target dataset folder")
    args = parser.parse_args()

    # Load hyperparameters from config.yaml
    hparams = load_hparams('config.yaml')

    # set random seed
    set_seed(hparams['random_seed'])

    # Set the device
    device = torch.device('cuda')


    # Initialize the data module
    data_module = DataModuleSourceTarget(source_folder=args.src,
                                          target_folder=args.tgt,
                                          hparams=hparams)

    # Initialize the model
    # model = LoRA_module(hparams).to(device)
    #model = FineTuneTask(hparams)
    model = FineTuneTaskDivergence(hparams).to(device)


    # Initialize WandB logger
    wandb_logger = WandbLogger(project="LoRA_model_training", config=hparams)

    # Add ModelCheckpoint callback to save the best model based on validation F1
    checkpoint_callback = ModelCheckpoint(
        monitor='source_validation/f1',  # Monitor validation F1 score of the source data (since in theory we don't have target data available)
        mode='max',  # Save the best model based on the highest F1 score
        save_top_k=1,  # Only save the best model
        filename='best_model',  # Name of the saved model file
        verbose=True
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=hparams['accelerator'],  # Which accelarator to use (on hparams, gpu)
        devices=hparams['devices'],  # Number of gpus to use (on hparams, 1)
        max_epochs=hparams['n_epochs'],
        log_every_n_steps=10, 
        callbacks=[checkpoint_callback]
    )

    # Start training
    print(f"Training with source dataset: {args.src} and target dataset: {args.tgt}")
    trainer.fit(model, datamodule=data_module)

    # After training, load the best model (highest f1) based on validation performance
    best_model_path = checkpoint_callback.best_model_path  # Get the path of the best model
    print(f"Loading best model from: {best_model_path}")
    
    best_model = FineTuneTask.load_from_checkpoint(best_model_path, hparams=hparams)

    # Run testing with the best model
    trainer.test(best_model, datamodule=data_module)