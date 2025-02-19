import argparse
import os
import yaml
import torch
import time
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.lora import LoRA_module 

from src.models.finetune import FineTune

from dataset.dataloader import DataModuleSourceTarget

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def load_hparams(yaml_file):
    """Load hyperparameters from a YAML file."""
    with open(yaml_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def get_model_class(model_name):
    """Return the appropriate model based on the model name."""
    if model_name == 'finetune':
        return FineTune
    elif model_name == 'lora':
        return LoRA_module
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    

def measure_inference_time(model, data_loader, device='cuda'):
    # Measure the inference time for a single batch
    model.to(device)
    print("\n using device: ", device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # get the first (and only) batch from dataloader
        batch = next(iter(data_loader))
        input_ids = batch['target']['input_ids'].to(device)
        attention_mask = batch['target']['attention_mask'].to(device)
        if 'token_type_ids' in batch['target']:
            token_type_ids = batch['target']['token_type_ids'].to(device)
            inputs = (input_ids, attention_mask, token_type_ids)
        else:
            inputs = (input_ids, attention_mask)

        start_time = time.time()

        outputs = model(*inputs)

        end_time = time.time()

        total_time = end_time - start_time

    # number of instances in a batch
    return total_time


def train(args):

    hparams = load_hparams('config/config.yaml')
    model_name = args.model

    # Set random seed
    

    # Override dataset directories with command-line arguments
    hparams['source_folder'] = args.src
    hparams['target_folder'] = args.tgt
    hparams['model_name'] = args.model
    hparams['random_seed'] = args.seed
    
    set_seed(hparams['random_seed'])

    # Merge parameters with wandb.config if running a sweep
    if args.hparam_tuning:
        print("Loading hparams from wandb.config")
        wandb.init()
        hparams.update(wandb.config)
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the data module
    datamodule = DataModuleSourceTarget(source_folder=args.src,
                                          target_folder=args.tgt,
                                          hparams=hparams)

    # Initialize the model
    ModelClass = get_model_class(model_name)
    model = ModelClass(hparams=hparams).to(device)


    # Initialize WandB logger
    wandb_logger = WandbLogger(project=model_name, config=hparams)

    # Add ModelCheckpoint callback to save the best model based on validation F1
    if model_name == 'lora_task':
        monitor_metric = 'validation/f1'
    else:
        # try with different metrics to monitor
        monitor_metric = "target_validation/f1"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,  # Monitor validation F1 score
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
    start_time = time.time()
    print(f"Training with source dataset: {args.src} and target dataset: {args.tgt}")
    trainer.fit(model, datamodule=datamodule)

    training_time = time.time() - start_time

    # After training, load the best model (highest f1) based on validation performance
    best_model_path = checkpoint_callback.best_model_path  # Get the path of the best model
    print(f"Loading best model from: {best_model_path}")
    
    best_model = ModelClass.load_from_checkpoint(best_model_path, hparams=hparams)

    # Run testing with the best model
    trainer.test(best_model, datamodule=datamodule)

    test_loader = datamodule.test_dataloader()

    inference_time = measure_inference_time(model, test_loader, device=device)

    target_test_f1 = trainer.logged_metrics['target_test/real_f1']

    print("########## RESULTS ##########")
    print(f"{args.src}_{args.tgt},{args.seed},{target_test_f1},{training_time},{inference_time}\n")

    csv_file = "/work/derossi/Thesis-Adapters/results_qv.csv"
    # Save the results on a .csv file
    with open(csv_file, "a") as f:
        f.write(f"{args.src}_{args.tgt},{args.seed},{hparams['lora_r']},{target_test_f1},{training_time},{inference_time}\n")
        print("Write successful")


def main():

    parser = argparse.ArgumentParser(description="Training script for domain adaptation")
    parser.add_argument('--src', type=str, required=True, help="Source dataset folder")
    parser.add_argument('--tgt', type=str, required=True, help="Target dataset folder")
    parser.add_argument('--seed', type=int, required=True, help="Random seed")
    parser.add_argument('--model', type=str, required=True, help="PEFT model")
    parser.add_argument('--hparam_tuning', action='store_true', help="If set, hyperparameter tuning is being performed")
    args = parser.parse_args()

    if args.hparam_tuning:
        print("Hyperparameter tuning is enabled")
        if args.model == 'lora':
            # Load the sweep configuration from LoRA YAML file
            with open('config/lora_sweep_config.yaml', 'r') as file:
                sweep_config = yaml.safe_load(file)
        elif args.model == 'finetune':
            # Load the sweep configuration from finetuning YAML file
            with open('config/finetune_sweep_config.yaml', 'r') as file:
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