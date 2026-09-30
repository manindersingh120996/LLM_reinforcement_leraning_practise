import os

import torch

from src.models.reward_model import GPT2RewardModel
from src.utils.config import load_config
from src.utils.seed import set_seed


CONFIG_PATH = r"./configs/default.yaml"

EXPORT_DIRECTORY = "./exported_model"


def main():

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    config = load_config(CONFIG_PATH)

    set_seed(config["seed"])

    model_config = config["model"]
    checkpoint_config = config["checkpoint"]

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------

    checkpoint_path = os.path.join(
        checkpoint_config["directory"],
        checkpoint_config["best_model_name"],
    )

    print(
        f"Loading checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = GPT2RewardModel(
        model_name=model_config["name"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_metric']:.4f}"
    )

    # --------------------------------------------------
    # Export model
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

    print("Model export complete.")


if __name__ == "__main__":
    main()