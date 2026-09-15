import torch

from src.losses.bradley_terry import bradley_terry_loss


class RewardModelTrainer:

    def __init__(
        self,
        model,
        optimizer,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch in dataloader:

            batch = {
                key: value.to(self.device)
                for key, value in batch.items()
            }

            chosen_rewards = self.model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )

            rejected_rewards = self.model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )

            loss = bradley_terry_loss(
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

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