import torch
import torch.nn as nn

from transformers import GPT2Config
from transformers import GPT2Model
from transformers import PreTrainedModel

from .configuration_reward import GPT2RewardConfig


class GPT2RewardModel(PreTrainedModel):

    config_class = GPT2RewardConfig

    def __init__(
        self,
        config: GPT2RewardConfig,
    ):
        super().__init__(config)

        # ----------------------------------------------
        # Reconstruct GPT-2 architecture from config.
        #
        # IMPORTANT:
        # GPT2Model(config) does NOT download the
        # original GPT-2 weights.
        # ----------------------------------------------

        backbone_config = GPT2Config(
            **config.backbone_config
        )

        self.backbone = GPT2Model(
            backbone_config
        )

        # ----------------------------------------------
        # Reward head
        # ----------------------------------------------

        hidden_size = (
            self.backbone.config.n_embd
        )

        self.reward_head = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):

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