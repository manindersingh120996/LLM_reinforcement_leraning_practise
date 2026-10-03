import torch

from src.losses.bradley_terry import bradley_terry_loss


def evaluate(
    model,
    dataloader,
    device,
    mixed_precision=None,
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    use_bf16 = (
        mixed_precision == "bf16"
        and device.type == "cuda"
    )

    with torch.no_grad():

        for batch in dataloader:

            # -----------------------------------------------------
            # Move batch to device
            # -----------------------------------------------------

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # =====================================================
            # BF16 VALIDATION
            # =====================================================

            if use_bf16:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    chosen_rewards = model(
                        input_ids=batch[
                            "chosen_input_ids"
                        ],
                        attention_mask=batch[
                            "chosen_attention_mask"
                        ],
                    )

                    rejected_rewards = model(
                        input_ids=batch[
                            "rejected_input_ids"
                        ],
                        attention_mask=batch[
                            "rejected_attention_mask"
                        ],
                    )

                # Calculate loss in FP32
                loss = bradley_terry_loss(
                    chosen_rewards=chosen_rewards.float(),
                    rejected_rewards=rejected_rewards.float(),
                )

            # =====================================================
            # FP32 VALIDATION
            # =====================================================

            else:

                chosen_rewards = model(
                    input_ids=batch[
                        "chosen_input_ids"
                    ],
                    attention_mask=batch[
                        "chosen_attention_mask"
                    ],
                )

                rejected_rewards = model(
                    input_ids=batch[
                        "rejected_input_ids"
                    ],
                    attention_mask=batch[
                        "rejected_attention_mask"
                    ],
                )

                loss = bradley_terry_loss(
                    chosen_rewards=chosen_rewards,
                    rejected_rewards=rejected_rewards,
                )

            # =====================================================
            # METRICS
            # =====================================================

            batch_size = chosen_rewards.size(0)

            correct = (
                chosen_rewards > rejected_rewards
            ).sum().item()

            total_correct += correct
            total_examples += batch_size

            total_loss += (
                loss.item() * batch_size
            )

    # =============================================================
    # FINAL METRICS
    # =============================================================

    average_loss = (
        total_loss / total_examples
    )

    accuracy = (
        total_correct / total_examples
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }