import torch

from src.losses.bradley_terry import bradley_terry_loss


def evaluate(model, dataloader, device):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():

        for batch in dataloader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            chosen_rewards = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )

            rejected_rewards = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )

            loss = bradley_terry_loss(
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
            )

            batch_size = chosen_rewards.size(0)

            correct = (
                chosen_rewards > rejected_rewards
            ).sum().item()

            total_correct += correct
            total_examples += batch_size

            total_loss += loss.item() * batch_size

    average_loss = total_loss / total_examples

    accuracy = total_correct / total_examples

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }