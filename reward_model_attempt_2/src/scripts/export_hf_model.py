import os

import torch

from transformers import AutoTokenizer

from src.utils.config import load_config

from exported_model.configuration_reward import (
    GPT2RewardConfig,
)

from exported_model.modeling_reward import (
    GPT2RewardModel,
)


CONFIG_PATH = r"./configs/default.yaml"

CHECKPOINT_PATH = (
    "./checkpoints/best_model.pt"
)

EXPORT_DIRECTORY = "./exported_model_hf"


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = load_config(CONFIG_PATH)

    model_config = config["model"]
    data_config = config["data"]

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Load original checkpoint
    # --------------------------------------------------

    print(
        f"Loading checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    # --------------------------------------------------
    # Create custom configuration
    # --------------------------------------------------

    reward_config = GPT2RewardConfig(
        model_name=model_config["name"],
        max_length=data_config["max_length"],
    )

    # --------------------------------------------------
    # Register AutoClass support
    # --------------------------------------------------

    GPT2RewardConfig.register_for_auto_class()

    GPT2RewardModel.register_for_auto_class(
        "AutoModel"
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = GPT2RewardModel(
        reward_config
    )

    # --------------------------------------------------
    # Load trained weights
    # --------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Export
    # --------------------------------------------------

    os.makedirs(
        EXPORT_DIRECTORY,
        exist_ok=True,
    )

    print(
        f"Saving model to: "
        f"{EXPORT_DIRECTORY}"
    )

    model.save_pretrained(
        EXPORT_DIRECTORY
    )

    # --------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"]
    )

    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(
        EXPORT_DIRECTORY
    )

    print()
    print("Export complete.")


if __name__ == "__main__":
    main()