"""
Reward Model — built on PreTrainedModel for native HuggingFace compatibility.

Why PreTrainedModel instead of nn.Module
-----------------------------------------
Subclassing PreTrainedModel gives us save_pretrained() and from_pretrained()
for free. This means:

    # Save after training
    model.save_pretrained("checkpoints/hub_export")
    model.push_to_hub("manindersingh120996/reward-model-hh-rlhf")

    # Load anywhere, any project
    from reward_model.model import RewardModel
    model = RewardModel.from_pretrained("manindersingh120996/reward-model-hh-rlhf")

No custom serialisation logic. No separate scalar_head.pt. No from_hub method.
All weights — backbone and scalar head — are saved together in one standard
model.safetensors file. The architecture is fully described by RewardModelConfig
in config.json.

Two construction paths
-----------------------
Training (loads pretrained GPT-2 backbone):
    model = RewardModel.from_pretrained_gpt2("gpt2", cfg)

Testing / offline (random weights, no download):
    model = RewardModel.from_config_test(n_embd=64, n_layer=4, n_head=2)

Loading a trained checkpoint:
    model = RewardModel.from_pretrained("username/repo-id")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, PretrainedConfig, PreTrainedModel


class RewardModelConfig(PretrainedConfig):
    """
    Configuration for RewardModel.

    Stores GPT-2 architecture parameters + reward model specific parameters.
    Saved as config.json by save_pretrained; loaded by from_pretrained to
    reconstruct the exact architecture without any extra setup.
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

    Subclasses PreTrainedModel — save_pretrained, push_to_hub, and from_pretrained
    all work natively. No custom serialisation code needed.

    Architecture:
        Input  → GPT-2 backbone → last real token hidden state
               → Dropout → Linear(hidden_dim, 1, bias=False) → scalar score

    Loading a trained model:
        from reward_model.model import RewardModel
        model = RewardModel.from_pretrained("manindersingh120996/reward-model-hh-rlhf")
        model.eval()
        score = model(input_ids, attention_mask)   # shape: (batch_size,)
    """

    config_class = RewardModelConfig

    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__(config)

        gpt2_config = GPT2Config(
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_positions=config.n_positions,
            n_inner=config.n_inner,
            vocab_size=config.vocab_size,
        )
        self.backbone = GPT2Model(gpt2_config)
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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Score a batch of (prompt + response) sequences.

        Args:
            input_ids:      Shape (batch_size, seq_len). Right-padded token IDs.
            attention_mask: Shape (batch_size, seq_len). 1=real token, 0=padding.

        Returns:
            Shape (batch_size,). Scalar reward score per sequence.
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

    # ------------------------------------------------------------------
    # Construction classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_gpt2(
        cls,
        backbone_name: str,
        cfg,
    ) -> "RewardModel":
        """
        Create RewardModel for training, initialised with pretrained GPT-2 weights.

        Flow:
            1. Download GPT-2 config (architecture dims only, not weights).
            2. Build RewardModelConfig combining GPT-2 dims + reward model params.
            3. Construct model with random weights.
            4. Load pretrained GPT-2 weights into backbone.
            5. Apply layer freezing.

        For loading a trained reward model checkpoint use from_pretrained() instead.
        """
        from transformers import AutoConfig

        backbone_cfg = AutoConfig.from_pretrained(backbone_name)

        reward_config = RewardModelConfig(
            n_embd=backbone_cfg.n_embd,
            n_layer=backbone_cfg.n_layer,
            n_head=backbone_cfg.n_head,
            n_positions=backbone_cfg.n_positions,
            n_inner=getattr(backbone_cfg, "n_inner", None),
            vocab_size=backbone_cfg.vocab_size,
            dropout=cfg.model.dropout,
            num_unfrozen_layers=cfg.model.num_unfrozen_layers,
            backbone_name=backbone_name,
        )

        model = cls(reward_config)

        pretrained_backbone = GPT2Model.from_pretrained(backbone_name)
        model.backbone.load_state_dict(pretrained_backbone.state_dict())
        del pretrained_backbone

        model._apply_layer_freezing(cfg.model.num_unfrozen_layers)
        return model

    @classmethod
    def from_config_test(
        cls,
        n_embd: int = 64,
        n_layer: int = 4,
        n_head: int = 2,
        vocab_size: int = 128,
        dropout: float = 0.0,
        num_unfrozen_layers: int = 2,
    ) -> "RewardModel":
        """
        Tiny RewardModel for tests — no network access, loads in under 1 second.

        Applies layer freezing identically to production so test coverage is real.
        """
        config = RewardModelConfig(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=128,
            vocab_size=vocab_size,
            dropout=dropout,
            num_unfrozen_layers=num_unfrozen_layers,
        )
        model = cls(config)
        model._apply_layer_freezing(num_unfrozen_layers)
        return model

    # ------------------------------------------------------------------
    # Saving: use inherited save_pretrained() and push_to_hub()
    # ------------------------------------------------------------------
    #
    # model.save_pretrained("path")          saves config.json + model.safetensors
    # model.push_to_hub("username/repo")     pushes both to HuggingFace Hub
    #
    # See train.py for the full save + push flow with local backup.

    # ------------------------------------------------------------------
    # Layer freezing
    # ------------------------------------------------------------------

    def _apply_layer_freezing(self, num_unfrozen_layers: int) -> None:
        """
        Freeze backbone, leaving the top N transformer blocks trainable.

        Lower layers → syntax/low-level patterns → frozen.
        Upper layers → task-specific representations → trainable.
        Final layer norm (ln_f) and scalar head → always trainable.
        """
        if num_unfrozen_layers == -1:
            return

        for param in self.backbone.parameters():
            param.requires_grad = False

        if num_unfrozen_layers > 0:
            total = len(self.backbone.h)
            if num_unfrozen_layers > total:
                raise ValueError(
                    f"num_unfrozen_layers={num_unfrozen_layers} exceeds "
                    f"total transformer blocks={total}."
                )
            for layer in self.backbone.h[total - num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

            for param in self.backbone.ln_f.parameters():
                param.requires_grad = True

        for param in self.scalar_head.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def parameter_stats(self) -> ParameterStats:
        """Return trainable vs frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return ParameterStats(
            total=total,
            trainable=trainable,
            frozen=frozen,
            trainable_pct=round(100.0 * trainable / total, 2),
        )