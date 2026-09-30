import os

import torch
from transformers import AutoTokenizer

from src.data.dataset import (
    PreferenceDataset,
    create_dataloader,
    load_preference_data,
)
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
    data_config = config["data"]
    checkpoint_config = config["checkpoint"]
    training_config = config["training"]

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Load test data
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

    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    test_dataset = PreferenceDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=data_config["max_length"],
    )

    test_dataloader = create_dataloader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    # --------------------------------------------------
    # Original checkpoint model
    # --------------------------------------------------

    checkpoint_path = os.path.join(
        checkpoint_config["directory"],
        checkpoint_config["best_model_name"],
    )

    print()
    print("Loading original checkpoint...")

    original_model = GPT2RewardModel(
        model_name=model_config["name"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    original_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    original_model.to(device)
    original_model.eval()

    # --------------------------------------------------
    # Exported model
    # --------------------------------------------------

    print("Loading exported model...")

    exported_model = GPT2RewardModel.from_pretrained(
        EXPORT_DIRECTORY
    )

    exported_model.to(device)
    exported_model.eval()

    # --------------------------------------------------
    # Compare model outputs
    # --------------------------------------------------

    max_difference = 0.0
    total_examples = 0

    with torch.no_grad():

        for batch in test_dataloader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # ------------------------------
            # Chosen rewards
            # ------------------------------

            original_chosen = original_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )

            exported_chosen = exported_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )

            # ------------------------------
            # Rejected rewards
            # ------------------------------

            original_rejected = original_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )

            exported_rejected = exported_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )

            # ------------------------------
            # Differences
            # ------------------------------

            chosen_difference = torch.max(
                torch.abs(
                    original_chosen
                    - exported_chosen
                )
            ).item()

            rejected_difference = torch.max(
                torch.abs(
                    original_rejected
                    - exported_rejected
                )
            ).item()

            batch_difference = max(
                chosen_difference,
                rejected_difference,
            )

            max_difference = max(
                max_difference,
                batch_difference,
            )

            total_examples += (
                original_chosen.size(0)
            )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("Export Round-Trip Test")
    print("----------------------")
    print(
        f"Examples tested: {total_examples}"
    )
    print(
        f"Maximum reward difference: "
        f"{max_difference:.10f}"
    )

    # --------------------------------------------------
    # Pass / Fail
    # --------------------------------------------------

    tolerance = 1e-6

    if max_difference < tolerance:

        print()
        print(
            "PASS: Exported model matches "
            "the original checkpoint."
        )

    else:

        print()
        print(
            "FAIL: Exported model differs "
            "from the original checkpoint."
        )


if __name__ == "__main__":
    main()