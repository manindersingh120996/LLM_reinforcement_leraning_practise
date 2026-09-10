import torch
import torch.nn as nn

from transformers import GPT2Model


class GPT2RewardModel(nn.Module):

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()

        # --------------------------------------------------------
        # GPT-2 backbone
        # --------------------------------------------------------

        self.backbone = GPT2Model.from_pretrained(
            model_name
        )

        # Hidden dimension of GPT-2
        hidden_size = self.backbone.config.n_embd

        # --------------------------------------------------------
        # Reward head
        # --------------------------------------------------------

        self.reward_head = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------------
        # GPT-2 forward pass
        # --------------------------------------------------------

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # hidden_states:
        # [batch_size, sequence_length, hidden_size]

        # --------------------------------------------------------
        # Find the final real token for every sequence
        # --------------------------------------------------------

        sequence_lengths = attention_mask.sum(dim=1) - 1

        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device,
        )

        final_hidden_states = hidden_states[
            batch_indices,
            sequence_lengths,
        ]

        # final_hidden_states:
        # [batch_size, hidden_size]

        # --------------------------------------------------------
        # Convert representation into scalar reward
        # --------------------------------------------------------

        rewards = self.reward_head(
            final_hidden_states
        ).squeeze(-1)

        # rewards:
        # [batch_size]

        return rewards