import torch

from transformers import AutoTokenizer

from src.data.dataset import (
    PreferenceDataset,
    create_dataloader,
    load_preference_data,
)

from src.models.reward_model import GPT2RewardModel

from src.evaluation.evaluator import evaluate

from src.utils.config import load_config
from src.utils.seed import set_seed


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

    _, _, test_data = load_preference_data(
        dataset_name=data_config["dataset_name"],
        train_split=data_config["train_split"],
        test_split=data_config["test_split"],
        validation_ratio=data_config["validation_ratio"],
        seed=config["seed"],
        train_sample_size=data_config["train_sample_size"],
        validation_sample_size=data_config["validation_sample_size"],
        test_sample_size=data_config["test_sample_size"],
    )

    print(
        f"Test examples: {len(test_data)}"
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"]
    )

    # GPT-2 does not have a pad token
    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    test_dataset = PreferenceDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=data_config["max_length"],
    )

    # --------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------

    test_dataloader = create_dataloader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = GPT2RewardModel(
        model_name=model_config["name"]
    )

    model.to(device)

    # --------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------

    best_checkpoint_path = (
        f"{checkpoint_config['directory']}/"
        f"{checkpoint_config['best_model_name']}"
    )

    print(
        f"Loading best checkpoint: "
        f"{best_checkpoint_path}"
    )

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_metric']:.4f}"
    )

    # --------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------

    test_metrics = evaluate(
        model=model,
        dataloader=test_dataloader,
        device=device,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("Final Test Results")
    print("------------------")
    print(
        f"Test Loss:     {test_metrics['loss']:.4f}"
    )
    print(
        f"Test Accuracy: {test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()