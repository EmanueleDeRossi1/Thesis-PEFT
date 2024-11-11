import argparse
import os
import yaml
import torch
import random
import wandb
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
        config = yaml.safe_load(stream)
    common_hparams = config['common_parameters']
    model_name = config['model_name']
    model_hparams = config['models'][model_name]
    return common_hparams, model_name, model_hparams

def get_model_class(model_name):
    """Return the appropriate model based on the model name."""
    if model_name == 'finetune_task_divergence':
        return FineTuneTaskDivergence
    elif model_name == 'lora':
        return LoRA_module
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train(args):

    common_hparams, model_name, model_hparams = load_hparams('config.yaml')

    # Set random seed
    set_seed(common_hparams['random_seed'])


    # Merge parameters with wandb.config if running a sweep
    if args.hparam_tuning:
        print("Loading hparams from wandb.config")
        wandb.init()
        hparams = {**common_hparams, **model_hparams['default_parameters'], **wandb.config}
    else:
        print("NOT loading hparams from wandb.config")
        hparams = {**common_hparams, **model_hparams['default_parameters']}

    # Override dataset directories with command-line arguments
    hparams['source_folder'] = args.src
    hparams['target_folder'] = args.tgt


    # Set the device
    device = torch.device('cuda')

    # Initialize the data module
    data_module = DataModuleSourceTarget(source_folder=args.src,
                                          target_folder=args.tgt,
                                          hparams=hparams)

    # Initialize the model
    ModelClass = get_model_class(model_name)
    model = ModelClass(hparams=hparams).to(device)


    # Initialize WandB logger
    wandb_logger = WandbLogger(project=model_name, config=hparams)

    # Add ModelCheckpoint callback to save the best model based on validation F1
    checkpoint_callback = ModelCheckpoint(
        monitor='target_validation/f1',  # Monitor validation F1 score of the source data (since in theory we don't have target data available)
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
        callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs=1, # so that I can shuffle source data at the beginning of each epoch
    )

    # Start training
    print(f"Training with source dataset: {args.src} and target dataset: {args.tgt}")
    trainer.fit(model, datamodule=data_module)

    # After training, load the best model (highest f1) based on validation performance
    best_model_path = checkpoint_callback.best_model_path  # Get the path of the best model
    print(f"Loading best model from: {best_model_path}")
    
    best_model = ModelClass.load_from_checkpoint(best_model_path, hparams=hparams)

    # Run testing with the best model
    trainer.test(best_model, datamodule=data_module)

def main():

    parser = argparse.ArgumentParser(description="Training script for domain adaptation")
    parser.add_argument('--src', type=str, required=True, help="Source dataset folder")
    parser.add_argument('--tgt', type=str, required=True, help="Target dataset folder")
    parser.add_argument('--hparam_tuning', action='store_true', help="If set, hyperparameter tuning is being performed")
    args = parser.parse_args()

    if args.hparam_tuning:
        print("Hyperparameter tuning is enabled")
        # Load the sweep configuration from the YAML file
        with open('sweep_config.yaml', 'r') as file:
            sweep_config = yaml.safe_load(file)

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project='LoRA_model_training')

        # Run the sweep agent
        wandb.agent(sweep_id, function=lambda: train(args))
    else:
        print("Hyperparameter tuning is disabled")
        train(args)


if __name__ == "__main__":
    main()