import torch
import torch.nn as nn

from transformers import GPT2Model
from huggingface_hub import PyTorchModelHubMixin


class GPT2RewardModel(
    nn.Module,
    PyTorchModelHubMixin,
):

    def __init__(
        self,
        model_name: str = "gpt2",
    ):
        super().__init__()

        self.model_name = model_name

        self.backbone = GPT2Model.from_pretrained(
            model_name
        )

        hidden_size = self.backbone.config.n_embd

        self.reward_head = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        sequence_lengths = (
            attention_mask.sum(dim=1) - 1
        )

        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device,
        )

        final_hidden_states = hidden_states[
            batch_indices,
            sequence_lengths,
        ]

        rewards = self.reward_head(
            final_hidden_states
        ).squeeze(-1)

        return rewards