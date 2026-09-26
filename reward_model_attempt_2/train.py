import os

import torch
from torch.optim import AdamW

from src.data.dataset import (
    PreferenceDataset,
    create_dataloader,
    load_preference_data,
)
from src.models.reward_model import GPT2RewardModel
from src.training.trainer import RewardModelTrainer
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config


CONFIG_PATH = r"./configs/default.yaml"


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = load_config(CONFIG_PATH)

    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    train_data, val_data = load_preference_data(
        dataset_name=data_config["dataset_name"],
        train_split=data_config["train_split"],
        val_split=data_config["val_split"],
        train_sample_size=data_config["train_sample_size"],
        val_sample_size=data_config["val_sample_size"],
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"]
    )

    # GPT-2 does not have a pad token
    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataset = PreferenceDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=data_config["max_length"],
    )

    val_dataset = PreferenceDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=data_config["max_length"],
    )

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    train_dataloader = create_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=training_config["batch_size"],
        shuffle=True,
    )

    val_dataloader = create_dataloader(
        dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = GPT2RewardModel(
        model_name=model_config["name"]
    )

    model.to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = RewardModelTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    os.makedirs(
        checkpoint_config["directory"],
        exist_ok=True,
    )

    best_metric = float("-inf")

    for epoch in range(training_config["num_epochs"]):

        train_metrics = trainer.train_epoch(
            train_dataloader
        )

        val_metrics = trainer.validate(
            val_dataloader
        )

        print(
            f"Epoch {epoch + 1}/{training_config['num_epochs']} "
            f"| Train Loss: {train_metrics['loss']:.4f} "
            f"| Train Accuracy: {train_metrics['accuracy']:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} "
            f"| Val Accuracy: {val_metrics['accuracy']:.4f}"
        )

        # --------------------------------------------------
        # Best checkpoint
        # --------------------------------------------------

        current_metric = val_metrics[
            checkpoint_config["metric"]
        ]

        if current_metric > best_metric:

            best_metric = current_metric

            checkpoint_path = os.path.join(
                checkpoint_config["directory"],
                checkpoint_config["best_model_name"],
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                path=checkpoint_path,
            )

            print(
                f"  → New best checkpoint saved "
                f"({checkpoint_config['metric']}: "
                f"{current_metric:.4f})"
            )


if __name__ == "__main__":
    main()