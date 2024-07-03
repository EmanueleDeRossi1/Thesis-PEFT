import click
import pathlib
import gc
import os
from domadapter.datamodules.mnli_dm import DataModuleSourceTarget
from domadapter.datamodules.sa_dm import SADataModuleSourceTarget
from domadapter.datamodules.benchmark_dm import BenchmarkDataModuleSourceTarget
from domadapter.datamodules.wdc_dm import WDCDataModuleSourceTarget
from domadapter.models.adapters.joint_domain_task_adapter import JointDomainTaskAdapter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import json
from domadapter.console import console
from rich.prompt import Confirm
import shutil
import wandb

import time
import torch


@click.command()
@click.option("--dataset-cache-dir", type=str, help="Cache directory for dataset.")
@click.option(
    "--source-target", type=str, help="Source and target domain in source_target format"
)
@click.option("--pretrained-model-name", type=str, help="PLM to be used from HF")
@click.option(
    "--padding", type=str, help="Add padding while tokenizing upto max length"
)
@click.option("--max-seq-length", type=str, help="seq length for tokenizer")
@click.option(
    "--num-classes",
    type=int,
    help="Number of classes for task adapter classification head",
)
@click.option("--bsz", type=int, help="batch size")
@click.option(
    "--data-module",
    type=str,
    help="data module on which trained model is to be trained (MNLI/SA)",
)
@click.option("--reduction-factor", help="Factor by which the hidden size is reduced")
@click.option(
    "--divergence",
    type=str,
    help="divergence on which trained domain adapter is to be loaded",
)
@click.option("--train-proportion", type=float, help="Train on small proportion")
@click.option("--dev-proportion", type=float, help="Validate on small proportion")
@click.option("--test-proportion", type=float, help="Test on small proportion")
@click.option("--exp-dir", type=str, help="Experiment directory to store artefacts")
@click.option("--seed", type=str, help="Seed for reproducibility")
@click.option("--lr", type=float, help="Learning rate for the entire model")
@click.option("--epochs", type=int, help="Number of epochs to run the training")
@click.option("--gpu", type=int, default=None, help="GPU to run the program on")
@click.option("--log-freq", type=int, help="Log wandb after how many steps")
@click.option(
    "--gradient_clip_norm",
    type=float,
    help="Clips the gradient if the norm is grater than this value",
    required=False,
    default=5.0,
)



def train_domain_task_adapter(
    bsz,
    dataset_cache_dir,
    pretrained_model_name,
    divergence,
    data_module,
    reduction_factor,
    train_proportion,
    dev_proportion,
    test_proportion,
    num_classes,
    max_seq_length,
    padding,
    source_target,
    exp_dir,
    seed,
    log_freq,
    gradient_clip_norm,
    lr,
    epochs,
    gpu,
):
    dataset_cache_dir = pathlib.Path(dataset_cache_dir)
    exp_dir = pathlib.Path(exp_dir)

    exp_dir = exp_dir.joinpath(source_target, f"joint_domain_task_adapter")

    if not exp_dir.is_dir():
        exp_dir.mkdir(parents=True)

    seed_everything(seed)

    hyperparams = {
        "bsz": bsz,
        "train_proportion": train_proportion,
        "dev_proportion": dev_proportion,
        "test_proportion": test_proportion,
        "source_target": source_target,
        "reduction_factor": int(reduction_factor),
        "num_classes": int(num_classes),
        "dataset_cache_dir": str(dataset_cache_dir),
        "exp_dir": str(exp_dir),
        "divergence": str(divergence),
        "seed": seed,
        "learning_rate": lr,
        "epochs": int(epochs),
        "gpu": gpu,
        "gradient_clip_norm": gradient_clip_norm,
        "pretrained_model_name": str(pretrained_model_name),
        "max_seq_length": int(max_seq_length),
        "padding": str(padding),
    }

    ###########################################################################
    # Setup the dataset
    ###########################################################################
    if data_module == "mnli":
        dm = DataModuleSourceTarget(hyperparams)
        project_name = f"MNLI_{pretrained_model_name}"
    elif data_module == "sa":
        dm = SADataModuleSourceTarget(hyperparams)
        project_name = f"SA_{pretrained_model_name}"
    elif data_module == "benchmark":
        dm = BenchmarkDataModuleSourceTarget(hyperparams)
        project_name = f"Benchmark_{pretrained_model_name}"
    elif data_module == "wdc":
        dm = WDCDataModuleSourceTarget(hyperparams)
        project_name = f"WDC_{pretrained_model_name}"

    dm.prepare_data()

    model = JointDomainTaskAdapter(hyperparams)

    ###########################################################################
    # SETUP THE LOGGERS and Checkpointers
    ###########################################################################
    run_id = wandb.util.generate_id()
    exp_dir = exp_dir.joinpath(run_id)

    logger = WandbLogger(
        save_dir=exp_dir,
        id=run_id,
        project=project_name,
        job_type=f"Joint domain task adapter {reduction_factor}",
        group=source_target,
    )

    checkpoints_dir = exp_dir.joinpath("checkpoints")
    checkpoints_dir.mkdir(parents=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        save_top_k=1,
        mode="min",
        monitor="source_val/loss",
    )

    callbacks = [checkpoint_callback]

    trainer = Trainer(
        limit_train_batches=train_proportion,
        limit_val_batches=dev_proportion,
        limit_test_batches=test_proportion,
        callbacks=callbacks,
        terminate_on_nan=True,
        log_every_n_steps=log_freq,
        gradient_clip_val=gradient_clip_norm,
        gpus=str(gpu),
        max_epochs=epochs,
        logger=logger,
    )

    # Start measuring training time
    start_time = time.time()

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    trainer.fit(model, train_loader, val_loader)

    # Calculate training time
    training_time = time.time() - start_time


    dm.setup("test")
    test_loader = dm.test_dataloader()
    trainer.test(model, test_loader)
    device = "cuda"

    best_ckpt_path = checkpoint_callback.best_model_path

    model = JointDomainTaskAdapter.load_from_checkpoint(best_ckpt_path)

    model.save_adapter(
        str(checkpoints_dir), f"adapter_{source_target}"
    )  # save adapter after loading model
    os.remove(best_ckpt_path)  # remove saved model

    print(f"Starting inference time measurement with model: {model}")

    inference_time = measure_inference_time(model, test_loader, device=device)
    print(f"Inference time per prediction: {inference_time} seconds")


    hparams_file = exp_dir.joinpath("hparams.json")

    test_results = trainer.test(model, test_loader)

    results_file = exp_dir.joinpath("results.json")

    with open(results_file, "w") as fp:
        json.dump(test_results, fp)


    with open(hparams_file, "w") as fp:
        json.dump(hyperparams, fp)

    # Measure model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Read F1 score and other metrics from logs
    metrics = test_results[0]

    # Save the results
    results = {
        "training_time": training_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "metrics": metrics,
        "inference_time": inference_time}

    results_file = exp_dir.joinpath("results.json")

    with open(results_file, "w") as fp:
        json.dump(results, fp)
    print(f"The source-target datasets are: {source_target} with seed {seed}")
    print(f"Training time: {training_time} seconds")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    print(f"Metrics: {metrics}")
    print(f"Inference time: {inference_time}")
    del model
    gc.collect()

def measure_inference_time(model, data_loader, device='cuda'):
    # Measure the inference time for a single prediction
    model.to(device)
    model.eval()
    total_time = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['source_input_ids'].to(device)
            attention_mask = batch['source_attention_mask'].to(device)

            # Measure time for each prediction in the batch
            for i in range(input_ids.size(0)):
                inputs = (input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0))

                start_time = time.time()
                outputs = model(*inputs)
                end_time = time.time()

                total_time += (end_time - start_time)
                total_predictions += 1

            break  # Measure only for one batch

    avg_inference_time = total_time / total_predictions

    return avg_inference_time


if __name__ == "__main__":
    train_domain_task_adapter()
