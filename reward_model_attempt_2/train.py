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
from src.utils.seed import set_seed
from transformers import get_linear_schedule_with_warmup


CONFIG_PATH = r"./configs/default.yaml"


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = load_config(CONFIG_PATH)
    set_seed(config["seed"])

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
    
    train_data, val_data, test_data = load_preference_data(
        dataset_name=data_config["dataset_name"],
        train_split=data_config["train_split"],
        test_split=data_config["test_split"],
        validation_ratio=data_config["validation_ratio"],
        seed=config["seed"],
        train_sample_size=data_config["train_sample_size"],
        validation_sample_size=data_config["validation_sample_size"],
        test_sample_size=data_config["test_sample_size"],
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
        num_workers=data_config["num_workers"]
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
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if (
            "bias" in name
            or "ln_" in name
            or "layernorm" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    print(
    f"Parameters with weight decay: "
    f"{len(decay_parameters)}"
    )

    print(
        f"Parameters without weight decay: "
        f"{len(no_decay_parameters)}"
    )

    optimizer = AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": training_config["weight_decay"],
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=training_config["learning_rate"],
)
    # print("weight decay applied...")

    # --------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------

    num_training_steps = (
        len(train_dataloader)
        * training_config["num_epochs"]
    )

    num_warmup_steps = int(
        num_training_steps
        * training_config["scheduler"]["warmup_ratio"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = RewardModelTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler = scheduler,
        gradient_clip_norm= training_config["gradient_clip_norm"]
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
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Current learning rate: {current_lr:.8f}"
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
    config = load_config(CONFIG_PATH)
    set_seed(config["seed"])
    main()