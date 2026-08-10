"""
Reward model — local package entry point.

Re-exports everything from modeling_reward_model.py (the self-contained file
that also gets uploaded to HuggingFace Hub), then adds:

  - from_pretrained_gpt2  training-time construction from pretrained GPT-2
  - from_config_test      offline construction for tests (no downloads)
  - save_pretrained       override that adds auto_map + copies model code
                          so trust_remote_code=True works on the hub

Loading options
---------------
With local package installed (this repo):
    from reward_model.model import RewardModel
    model = RewardModel.from_pretrained("manindersingh120996/reward-model-hh-rlhf")

Without local package (any machine, any project):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "manindersingh120996/reward-model-hh-rlhf",
        trust_remote_code=True,
    )
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import GPT2Model

# Re-export everything from the standalone hub file
from reward_model.modeling_reward_model import (  # noqa: F401
    ParameterStats,
    RewardModel as _RewardModelBase,
    RewardModelConfig,
)


class RewardModel(_RewardModelBase):
    """
    RewardModel with added training-time construction methods and a
    save_pretrained override that makes trust_remote_code loading work.

    The class itself is identical to the hub version — the only additions
    are classmethods that depend on the training environment (OmegaConf,
    HuggingFace Hub downloads) or are only needed locally (tests).
    """

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

        Use this at the start of a training run. For loading a trained checkpoint
        use from_pretrained("username/repo") instead.

        Args:
            backbone_name: HuggingFace model ID — "gpt2", "gpt2-medium", etc.
            cfg:           Config with cfg.model.dropout and
                           cfg.model.num_unfrozen_layers.
                           Accepts OmegaConf DictConfig or any object with
                           those attributes (e.g. types.SimpleNamespace).
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

        model = cls(reward_config)                              # random weights

        pretrained_backbone = GPT2Model.from_pretrained(backbone_name)
        model.backbone.load_state_dict(pretrained_backbone.state_dict())
        del pretrained_backbone                                 # free memory

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
    # save_pretrained override
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """
        Save weights + config + model code so trust_remote_code loading works.

        What this adds on top of the standard PreTrainedModel.save_pretrained:

        1. Writes "auto_map" into config.json before saving:
               "auto_map": {
                   "AutoConfig": "modeling_reward_model.RewardModelConfig",
                   "AutoModel":  "modeling_reward_model.RewardModel"
               }
           This tells HuggingFace which file and class to load when a user
           calls AutoModel.from_pretrained(..., trust_remote_code=True).

        2. Copies modeling_reward_model.py into the save directory.
           That file is what HuggingFace downloads and executes on the user's
           machine when trust_remote_code=True is passed.

        After save_pretrained + push_to_hub, your repo contains:
            config.json                 (has auto_map + architecture params)
            model.safetensors           (all weights: backbone + scalar head)
            modeling_reward_model.py    (model class code — executed remotely)
        """
        # Tell HuggingFace which file+class corresponds to AutoConfig/AutoModel
        self.config.auto_map = {
            "AutoConfig": "modeling_reward_model.RewardModelConfig",
            "AutoModel":  "modeling_reward_model.RewardModel",
        }

        # Standard save: writes config.json + model.safetensors
        super().save_pretrained(save_directory, **kwargs)

        # Copy the standalone model code alongside the weights
        # HuggingFace downloads this file when trust_remote_code=True is used
        src = Path(__file__).parent / "modeling_reward_model.py"
        dst = Path(save_directory) / "modeling_reward_model.py"
        shutil.copy(src, dst)