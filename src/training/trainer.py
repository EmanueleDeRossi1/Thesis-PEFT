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


### Utility Functions ###

def set_seed(seed):
    """Ensures reproducibility by setting random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def load_hparams(yaml_file):
    """Loads hyperparameters from a YAML file."""
    with open(yaml_file, "r") as stream:
        return yaml.safe_load(stream)


def get_model_class(model_name):
    """Returns the model class based on the model name."""
    if model_name == "finetune":
        return FineTune
    elif model_name == "lora":
        return LoRA_module
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def measure_inference_time(model, data_loader, device="cuda"):
    """Measures inference time for a single batch."""
    model.to(device)
    model.eval()
    print("\nUsing device:", device)

    with torch.no_grad():
        batch = next(iter(data_loader))
        input_ids = batch["target"]["input_ids"].to(device)
        attention_mask = batch["target"]["attention_mask"].to(device)
        inputs = (input_ids, attention_mask)

        start_time = time.time()
        model(*inputs)
        total_time = time.time() - start_time

    return total_time


### 🔹 Training Function ###

def train_model(args, hparams):
    """Handles training of the model using PyTorch Lightning."""
    set_seed(hparams["random_seed"])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Data Module
    datamodule = DataModuleSourceTarget(
        source_folder=args.src, target_folder=args.tgt, hparams=hparams
    )

    # Initialize Model
    ModelClass = get_model_class(args.model)
    model = ModelClass(hparams=hparams).to(device)

    # Initialize WandB logger
    wandb_logger = WandbLogger(project=args.model, config=hparams)

    # Define the validation metric to monitor
    monitor_metric = "validation/f1" 
    # ModelCheckpoint callback (saves the best model)
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        filename="best_model",
        verbose=True,
    )

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=hparams["accelerator"],  
        devices=hparams["devices"],  
        max_epochs=hparams["n_epochs"],
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs=1,
    )

    # Start training
    print(f"Training on: {args.src} → {args.tgt} | Seed: {args.seed}")
    start_time = time.time()
    trainer.fit(model, datamodule=datamodule)
    training_time = time.time() - start_time

    return model, trainer, checkpoint_callback, training_time


### 🔹 Evaluation Function ###

def evaluate_model(model, trainer, checkpoint_callback, datamodule, args, hparams):
    """Handles evaluation and result saving after training."""
    best_model_path = checkpoint_callback.best_model_path

    if not best_model_path:
        print("No best model found. Skipping evaluation.")
        return

    print(f"Loading best model from: {best_model_path}")
    best_model = model.load_from_checkpoint(best_model_path, hparams=hparams)

    # Run testing
    trainer.test(best_model, datamodule=datamodule)

    # Measure inference time
    test_loader = datamodule.test_dataloader()
    inference_time = measure_inference_time(best_model, test_loader)

    # Extract final test metric
    target_test_f1 = trainer.logged_metrics.get("target_test/real_f1", "N/A")

    # Save results
    results_file = "results_qv.csv"
    with open(results_file, "a") as f:
        f.write(
            f"{args.src}_{args.tgt},{args.seed},{hparams['lora_r']},{target_test_f1},{inference_time}\n"
        )

    print("Results saved successfully:", results_file)
    print(f"Final F1 Score: {target_test_f1}, Training Time: {inference_time}s")


### 🔹 Main Function ###

def main():
    """Handles argument parsing and launches training/evaluation."""
    parser = argparse.ArgumentParser(description="Training script for domain adaptation")
    parser.add_argument("--src", type=str, required=True, help="Source dataset folder")
    parser.add_argument("--tgt", type=str, required=True, help="Target dataset folder")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--model", type=str, required=True, help="PEFT model")
    parser.add_argument("--hparam_tuning", action="store_true", help="Enable hyperparameter tuning")
    args = parser.parse_args()

    # Load Hyperparameters
    hparams = load_hparams("config/config.yaml")
    hparams["source_folder"] = args.src
    hparams["target_folder"] = args.tgt
    hparams["model_name"] = args.model
    hparams["random_seed"] = args.seed

    # Handle Hyperparameter Tuning
    if args.hparam_tuning:
        print("Hyperparameter tuning is enabled")
        sweep_config_file = f"config/{args.model}_sweep_config.yaml"

        if not os.path.exists(sweep_config_file):
            print(f"Error: Sweep config file {sweep_config_file} not found!")
            return

        with open(sweep_config_file, "r") as file:
            sweep_config = yaml.safe_load(file)

        sweep_id = wandb.sweep(sweep_config, project="LoRA_model_training")
        wandb.agent(sweep_id, function=lambda: train_model(args, hparams))

    else:
        print("Hyperparameter tuning is disabled")
        model, trainer, checkpoint_callback, training_time = train_model(args, hparams)
        evaluate_model(model, trainer, checkpoint_callback, DataModuleSourceTarget(args.src, args.tgt, hparams), args, hparams)


if __name__ == "__main__":
    main()