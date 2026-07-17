#!/usr/bin/env python
"""
Training script for the reward model.

Usage
-----
    # Train with the default config
    python scripts/train.py

    # Override specific config values via Hydra-style CLI (OmegaConf)
    python scripts/train.py training.learning_rate=5e-5 training.batch_size=16

    # Use a different config file
    python scripts/train.py --config configs/my_experiment.yaml

Design: this script only does orchestration — loading config, wiring components
together, and calling trainer.train(). All logic lives in the library modules.
This file should be under 80 lines of actual code.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Ensure the package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reward_model.dataset import PreferenceDataset, create_dataloaders, setup_tokenizer
from reward_model.evaluator import RewardModelEvaluator
from reward_model.loss import BradleyTerryLoss
from reward_model.model import RewardModel
from reward_model.trainer import RewardModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reward model on preference data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML config file (default: configs/default.yaml)",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    cfg = OmegaConf.load(args.config)
    logger.info(f"Config loaded from: {args.config}")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.training.seed)

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {cfg.model.backbone}")
    tokenizer = setup_tokenizer(cfg.model.backbone)

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    logger.info("Loading training dataset...")
    train_ds = PreferenceDataset.from_config(tokenizer, cfg, split=cfg.data.train_split)
    logger.info(f"Training examples: {len(train_ds):,}")

    logger.info("Loading validation dataset...")
    val_ds = PreferenceDataset.from_config(tokenizer, cfg, split=cfg.data.val_split)
    logger.info(f"Validation examples: {len(val_ds):,}")

    train_loader, val_loader = create_dataloaders(train_ds, val_ds, tokenizer, cfg)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    logger.info(f"Building model: {cfg.model.backbone}")
    model = RewardModel(cfg)
    stats = model.parameter_stats()
    logger.info(f"Model ready. {stats}")

    # -----------------------------------------------------------------------
    # Loss, trainer
    # -----------------------------------------------------------------------
    loss_fn = BradleyTerryLoss(use_stable=cfg.loss.use_stable)
    trainer = RewardModelTrainer(model, loss_fn, train_loader, val_loader, cfg)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    logger.info("Starting training...")
    trainer.train()

    # -----------------------------------------------------------------------
    # Final evaluation on the full val set
    # -----------------------------------------------------------------------
    logger.info("Running final evaluation...")
    device = next(model.parameters()).device
    evaluator = RewardModelEvaluator(model, loss_fn, device=device)
    report = evaluator.evaluate(val_loader)
    logger.info("\n" + report.summary())

    # Save final evaluation results alongside the last checkpoint
    import json
    results_path = Path(cfg.training.output_dir) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Evaluation results saved to: {results_path}")


if __name__ == "__main__":
    main()