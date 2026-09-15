import torch


def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:

    reward_difference = (
        chosen_rewards - rejected_rewards
    )

    loss = -torch.nn.functional.logsigmoid(
        reward_difference
    ).mean()

    return loss