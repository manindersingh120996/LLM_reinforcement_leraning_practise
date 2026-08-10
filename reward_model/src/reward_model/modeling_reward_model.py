"""
Standalone reward model — uploaded to HuggingFace Hub alongside weights.

This file is executed by HuggingFace when a user does:
    AutoModel.from_pretrained("username/repo", trust_remote_code=True)

It must be completely self-contained: no imports from the local reward_model
package. Only standard library and transformers/torch are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, PretrainedConfig, PreTrainedModel


class RewardModelConfig(PretrainedConfig):
    """
    Configuration for the reward model.

    Stores both the GPT-2 architecture parameters and reward-model-specific
    parameters. Saved as config.json; from_pretrained reads it to reconstruct
    the exact architecture without any extra setup.

    Usage
    -----
    Loaded automatically by from_pretrained — you don't construct this manually.
    """

    model_type = "reward_model"

    def __init__(
        self,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_positions: int = 1024,
        n_inner: Optional[int] = None,
        vocab_size: int = 50257,
        dropout: float = 0.1,
        num_unfrozen_layers: int = 4,
        backbone_name: str = "gpt2",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_positions = n_positions
        self.n_inner = n_inner
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_unfrozen_layers = num_unfrozen_layers
        self.backbone_name = backbone_name


@dataclass
class ParameterStats:
    total: int
    trainable: int
    frozen: int
    trainable_pct: float

    def __str__(self) -> str:
        return (
            f"Parameters | total: {self.total:,} | "
            f"trainable: {self.trainable:,} ({self.trainable_pct:.1f}%) | "
            f"frozen: {self.frozen:,}"
        )


class RewardModel(PreTrainedModel):
    """
    Scalar reward model built on GPT-2.

    Takes a (prompt + response) token sequence and outputs a single scalar score.
    Higher score = the model believes this is a better response.

    Loading
    -------
    From HuggingFace Hub (no local code needed):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "manindersingh120996/reward-model-hh-rlhf",
            trust_remote_code=True,
        )
        model.eval()
        scores = model(input_ids, attention_mask)  # shape: (batch_size,)

    With local package installed:
        from reward_model.model import RewardModel
        model = RewardModel.from_pretrained("manindersingh120996/reward-model-hh-rlhf")

    Architecture
    ------------
        Input  →  GPT-2 backbone  →  last real token hidden state
               →  Dropout  →  Linear(hidden_dim, 1, bias=False)  →  scalar
    """

    config_class = RewardModelConfig

    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__(config)

        gpt2_cfg = GPT2Config(
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_positions=config.n_positions,
            n_inner=config.n_inner,
            vocab_size=config.vocab_size,
        )
        self.backbone = GPT2Model(gpt2_cfg)
        self.hidden_dim: int = config.n_embd

        self.scalar_head = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        nn.init.normal_(
            self.scalar_head[1].weight,
            mean=0.0,
            std=1.0 / (self.hidden_dim ** 0.5),
        )

    def forward(
        self,
        input_ids: torch.Tensor,       # (batch_size, seq_len)
        attention_mask: torch.Tensor,  # (batch_size, seq_len)
        **kwargs,
    ) -> torch.Tensor:                 # (batch_size,)
        """
        Score a batch of (prompt + response) sequences.

        Args:
            input_ids:      Token IDs. Right-padded with pad_token for batching.
            attention_mask: 1 for real tokens, 0 for padding.

        Returns:
            Scalar reward scores, shape (batch_size,).
            Higher = the model believes this response is better.
        """
        batch_size = input_ids.shape[0]

        backbone_output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_output.last_hidden_state
        # (batch_size, seq_len, hidden_dim)

        # Index last REAL token — cannot use [:, -1, :] as that may be padding
        last_real_token_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(batch_size, device=input_ids.device)
        last_hidden = hidden_states[batch_idx, last_real_token_idx]
        # (batch_size, hidden_dim)

        return self.scalar_head(last_hidden).squeeze(-1)
        # (batch_size,)

    def _apply_layer_freezing(self, num_unfrozen_layers: int) -> None:
        if num_unfrozen_layers == -1:
            return
        for param in self.backbone.parameters():
            param.requires_grad = False
        if num_unfrozen_layers > 0:
            total = len(self.backbone.h)
            for layer in self.backbone.h[total - num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.backbone.ln_f.parameters():
                param.requires_grad = True
        for param in self.scalar_head.parameters():
            param.requires_grad = True

    def parameter_stats(self) -> ParameterStats:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return ParameterStats(
            total=total,
            trainable=trainable,
            frozen=frozen,
            trainable_pct=round(100.0 * trainable / total, 2),
        )
