#!/usr/bin/env python
"""
Score (prompt, response) pairs with a trained reward model checkpoint.

Usage
-----
    # Compare two responses to the same prompt
    python scripts/inference.py \\
        --checkpoint checkpoints/checkpoint_best.pt \\
        --prompt "Human: What is the capital of France?" \\
        --response-a "Paris is the capital of France." \\
        --response-b "I don't know."

    # Score a single response (no comparison)
    python scripts/inference.py \\
        --checkpoint checkpoints/checkpoint_best.pt \\
        --prompt "Human: Explain gravity." \\
        --response-a "Gravity is a fundamental force..."

Output
------
    Response A score:  0.8312
    Response B score: -0.4521
    Difference (A-B):  1.2833
    Preferred:         A

Design notes
------------
The checkpoint stores the full config (as a plain dict) alongside the weights.
This means inference.py is self-contained: given a .pt file, it can reconstruct
the exact model architecture without needing the original config file.
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reward_model.dataset import setup_tokenizer
from reward_model.model import RewardModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score responses using a trained reward model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint file saved by RewardModelTrainer.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='The prompt text. Example: "Human: What is 2+2?"',
    )
    parser.add_argument(
        "--response-a",
        type=str,
        required=True,
        dest="response_a",
        help="First response to score.",
    )
    parser.add_argument(
        "--response-b",
        type=str,
        default=None,
        dest="response_b",
        help="Second response to compare against (optional).",
    )
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[RewardModel, OmegaConf]:
    """
    Load a RewardModel from a checkpoint saved by RewardModelTrainer.

    The checkpoint contains the full config (saved by _save_checkpoint),
    so we can reconstruct the exact model architecture without needing
    the original YAML config file.

    Args:
        checkpoint_path: Path to a .pt file.
        device:          Device to load the model onto.

    Returns:
        (model in eval() mode, config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config from the saved dict
    cfg = OmegaConf.create(ckpt["cfg"])

    # Build model (this will try to download pretrained weights for the backbone)
    model = RewardModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    return model, cfg


def score_text(
    model: RewardModel,
    tokenizer,
    text: str,
    max_length: int,
    device: torch.device,
) -> float:
    """
    Compute the reward score for a single text string.

    Args:
        text: The full (prompt + response) string to score.
              Format should match your training data format.

    Returns:
        Scalar reward score. Higher = the model believes this is a better response.
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        score = model(input_ids, attention_mask)

    return score.item()


def format_text(prompt: str, response: str) -> str:
    """
    Combine prompt and response into the format used during training.

    hh-rlhf format: "\n\nHuman: <prompt>\n\nAssistant: <response>"
    We follow this convention so the tokenized text matches the
    distribution the model was trained on.
    """
    # Strip existing prefixes to avoid double-adding them
    prompt = prompt.strip()
    if not prompt.startswith("Human:"):
        prompt = f"Human: {prompt}"

    response = response.strip()
    if not response.startswith("Assistant:"):
        response = f"Assistant: {response}"

    return f"\n\n{prompt}\n\n{response}"


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = setup_tokenizer(cfg.model.backbone)
    max_length = cfg.data.max_length

    # Score response A
    text_a = format_text(args.prompt, args.response_a)
    score_a = score_text(model, tokenizer, text_a, max_length, device)

    print()
    print(f"Response A: {args.response_a[:80]}{'...' if len(args.response_a) > 80 else ''}")
    print(f"  Score: {score_a:.4f}")

    # Score response B (if provided)
    if args.response_b is not None:
        text_b = format_text(args.prompt, args.response_b)
        score_b = score_text(model, tokenizer, text_b, max_length, device)

        print()
        print(f"Response B: {args.response_b[:80]}{'...' if len(args.response_b) > 80 else ''}")
        print(f"  Score: {score_b:.4f}")

        print()
        diff = score_a - score_b
        preferred = "A" if diff > 0 else "B" if diff < 0 else "tie"
        print(f"Difference (A - B): {diff:+.4f}")
        print(f"Preferred response: {preferred}")
    else:
        print()
        print("(Pass --response-b to compare two responses)")


if __name__ == "__main__":
    main()