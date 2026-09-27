import torch

from src.losses.bradley_terry import bradley_terry_loss
from src.evaluation.evaluator import evaluate


class RewardModelTrainer:

    def __init__(
        self,
        model,
        optimizer,
        device,
        scheduler = None,
        gradient_clip_norm = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm

    def train_epoch(self, dataloader, global_step):

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

            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm,
                )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            global_step += 1
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
            "global_step" : global_step
        }

    def validate(self, dataloader):

        return evaluate(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
        )