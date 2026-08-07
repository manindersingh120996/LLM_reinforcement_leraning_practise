"""
Reward Model Architecture.

Overview
--------
A reward model takes a (prompt + response) token sequence and outputs a single
scalar score representing the quality of that response. Higher score = better
response, as judged by human preferences.

Architecture
------------
    Input:    [prompt tokens] + [SEP] + [response tokens] + padding
    Backbone: GPT-2 transformer (pretrained weights, partially frozen)
    Pool:     Hidden state of the last REAL token (ignores padding)
    Head:     Dropout → Linear(hidden_dim → 1) → scalar score

Training Signal
---------------
The model is called separately for chosen and rejected responses:
    r_chosen   = model(prompt + chosen_response)
    r_rejected = model(prompt + rejected_response)

These two scalars are passed to BradleyTerryLoss, which computes:
    loss = softplus(-(r_chosen - r_rejected))

The model learns to push r_chosen higher and r_rejected lower.

Why GPT2Model, not GPT2LMHeadModel
------------------------------------
GPT2LMHeadModel includes a Linear(hidden_dim, vocab_size) head for next-token
prediction. We don't need token probabilities — we need a scalar quality score.
GPT2Model gives us the raw hidden states, to which we attach our own head.
Using GPT2LMHeadModel would load ~38M extra parameters we immediately discard.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import GPT2Config, GPT2Model


@dataclass
class ParameterStats:
    """Summary of trainable vs frozen parameter counts after freezing is applied."""
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


class RewardModel(nn.Module):
    """
    Scalar reward model built on a pretrained GPT-2 backbone.

    The model scores any (prompt, response) sequence with a single scalar.
    It does not know about "chosen" or "rejected" — that distinction lives
    in the training loop. The model simply answers: "how good is this response?"

    Parameters
    ----------
    cfg : DictConfig
        OmegaConf config with at minimum:
            cfg.model.backbone          — HuggingFace model ID (e.g. "gpt2")
            cfg.model.num_unfrozen_layers — int, how many top layers to fine-tune
            cfg.model.dropout           — float, dropout rate on the scalar head

    Example
    -------
    >>> cfg = OmegaConf.load("configs/default.yaml")
    >>> model = RewardModel(cfg)
    >>> print(model.parameter_stats())
    >>>
    >>> input_ids      = torch.randint(0, 50257, (4, 128))   # batch of 4 sequences
    >>> attention_mask = torch.ones(4, 128, dtype=torch.long)
    >>> scores = model(input_ids, attention_mask)             # shape: (4,)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # --- Backbone ---
        # GPT2Model: transformer layers only, no LM head
        # from_pretrained loads weights from HuggingFace Hub on first call,
        # then caches locally (~500MB for gpt2-small)
        self.backbone = GPT2Model.from_pretrained(cfg.model.backbone)
        self.hidden_dim: int = self.backbone.config.n_embd
        # gpt2:        hidden_dim = 768,  12 layers, 117M params
        # gpt2-medium: hidden_dim = 1024, 24 layers, 345M params
        # gpt2-large:  hidden_dim = 1280, 36 layers, 774M params

        # --- Scalar Head ---
        # Maps the last real token's hidden state → single reward score.
        #
        # Why bias=False?
        # The absolute value of reward scores is meaningless — only the
        # DIFFERENCE between two scores matters (that's what BradleyTerryLoss
        # uses). A bias term shifts all scores by a constant, which cancels out
        # in the difference. It adds a parameter that learns nothing useful.
        #
        # Why Dropout before the Linear?
        # The scalar head sees the same hidden state for every forward pass
        # of the same sequence. Without dropout, it can easily overfit to
        # surface features of the training preference pairs.
        self.scalar_head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )

        # --- Head Initialisation ---
        # Initialise weights with small values so initial reward scores are
        # near zero. This ensures our initial loss is close to log(2) ≈ 0.693
        # (the theoretical baseline when the model has learned nothing).
        # If initial scores are large, the loss starts at an arbitrary value
        # and it's harder to diagnose whether training is progressing normally.
        nn.init.normal_(
            self.scalar_head[1].weight,
            mean=0.0,
            std=1.0 / (self.hidden_dim ** 0.5),
        )

        # --- Layer Freezing ---
        self._apply_layer_freezing(cfg.model.num_unfrozen_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a batch of (prompt + response) sequences.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape (batch_size, seq_len). Token IDs for the concatenated
            prompt + response sequences. Shorter sequences are right-padded
            with [PAD] to match the longest sequence in the batch.
        attention_mask : torch.Tensor
            Shape (batch_size, seq_len). 1 for real tokens, 0 for [PAD] tokens.
            Used to locate the last real token position in each sequence.

        Returns
        -------
        torch.Tensor
            Shape (batch_size,). One scalar reward score per sequence.
            Higher score = model believes this response is better.
        """
        batch_size = input_ids.shape[0]

        # --- Backbone Forward Pass ---
        # last_hidden_state: (batch_size, seq_len, hidden_dim)
        # GPT-2 uses causal attention: each token attends only to prior tokens.
        # The last real token's hidden state has attended to the entire sequence.
        backbone_output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_output.last_hidden_state
        # shape: (batch_size, seq_len, hidden_dim)

        # --- Last Real Token Pooling ---
        # We cannot use hidden_states[:, -1, :] because right-padded sequences
        # have [PAD] tokens at the end. The last position is padding, not content.
        #
        # Correct approach: use attention_mask to count real tokens, then index.
        #
        # Example: attention_mask = [[1, 1, 1, 1, 0, 0]]
        #          mask.sum(dim=1) = [4]       → 4 real tokens
        #          last_real_idx   = [4-1] = [3]  → position 3 is last real token ✓
        last_real_token_idx = attention_mask.sum(dim=1) - 1
        # shape: (batch_size,)

        batch_idx = torch.arange(batch_size, device=input_ids.device)

        # Advanced indexing: for each item in the batch, pick the hidden state
        # at its specific last-real-token position.
        last_hidden = hidden_states[batch_idx, last_real_token_idx]
        # shape: (batch_size, hidden_dim)

        # --- Scalar Head ---
        # (batch_size, hidden_dim) → (batch_size, 1) → (batch_size,)
        scores = self.scalar_head(last_hidden).squeeze(-1)
        # squeeze(-1): remove the trailing dimension added by Linear(hidden_dim, 1)
        # Result shape: (batch_size,) — exactly what BradleyTerryLoss expects

        return scores

    def _apply_layer_freezing(self, num_unfrozen_layers: int) -> None:
        """
        Freeze the bottom layers of the backbone, leaving the top
        `num_unfrozen_layers` transformer blocks trainable.

        Freezing strategy rationale
        ---------------------------
        GPT-2 transformer blocks learn hierarchical representations:
          - Lower blocks (0-3):  syntax, tokenization, basic grammar
          - Middle blocks (4-7): semantics, factual associations
          - Upper blocks (8-11): task-specific, context-sensitive reasoning

        Reward modelling requires high-level quality judgements, which
        correspond to upper-layer representations. We freeze the lower layers
        to preserve the general language understanding built during pretraining,
        and fine-tune only the upper layers to adapt to the quality ranking task.

        Parameters
        ----------
        num_unfrozen_layers : int
            Number of top transformer blocks to leave trainable.
            -1 trains all layers (use for debugging or very large datasets).
            0 freezes all backbone layers (trains scalar head only — fast).
        """
        if num_unfrozen_layers == -1:
            # Train the full backbone — suitable when the dataset is large
            # or when you want maximum adaptation at the cost of speed
            return

        # Step 1: freeze everything in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        if num_unfrozen_layers == 0:
            # Scalar head is always trainable — fall through to end
            pass
        else:
            total_layers = len(self.backbone.h)  # 12 for gpt2-small

            if num_unfrozen_layers > total_layers:
                raise ValueError(
                    f"num_unfrozen_layers={num_unfrozen_layers} exceeds "
                    f"total transformer blocks={total_layers} in {self.backbone.config._name_or_path}"
                )

            # Step 2: unfreeze the last N transformer blocks
            # self.backbone.h is a ModuleList of transformer blocks
            layers_to_unfreeze = self.backbone.h[total_layers - num_unfrozen_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

            # Step 3: always unfreeze the final layer norm
            # ln_f is applied after all transformer blocks, before our head.
            # It normalises the hidden states the scalar head will receive.
            # Freezing it while training upper blocks creates a mismatch.
            for param in self.backbone.ln_f.parameters():
                param.requires_grad = True

        # Step 4: scalar head is always trainable (new weights, no pretrained values)
        for param in self.scalar_head.parameters():
            param.requires_grad = True

    @classmethod
    def from_config(cls, gpt2_config: GPT2Config, num_unfrozen_layers: int = 2, dropout: float = 0.0) -> "RewardModel":
        """
        Create a RewardModel from a GPT2Config with randomly initialized weights.

        This factory method exists for two purposes:
            1. Testing: create a model without downloading pretrained weights.
               Network access in tests is slow, fragile, and couples tests to
               external infrastructure. Tests should be hermetic.
            2. Training from scratch: if you want to train the backbone from
               scratch rather than fine-tuning a pretrained model.

        Args:
            gpt2_config: A GPT2Config defining the model architecture.
            num_unfrozen_layers: Layers to leave trainable (same as __init__).
            dropout: Dropout on the scalar head.

        Example
        -------
        >>> from transformers import GPT2Config
        >>> config = GPT2Config(n_layer=2, n_head=2, n_embd=64)  # tiny model
        >>> model = RewardModel.from_config(config)
        """
        from omegaconf import OmegaConf

        # We need to build the same structure __init__ expects,
        # but we'll bypass it by building the object directly.
        instance = cls.__new__(cls)
        torch.nn.Module.__init__(instance)

        # Build backbone from config (no download)
        instance.backbone = GPT2Model(gpt2_config)
        instance.hidden_dim = gpt2_config.n_embd

        # Scalar head
        instance.scalar_head = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(instance.hidden_dim, 1, bias=False),
        )
        torch.nn.init.normal_(
            instance.scalar_head[1].weight,
            mean=0.0,
            std=1.0 / (instance.hidden_dim ** 0.5),
        )

        # Apply freezing
        instance._apply_layer_freezing(num_unfrozen_layers)
        return instance

    def save_to_hub(
        self,
        repo_id: str,
        cfg: "DictConfig",
        evaluation_report=None,
        private: bool = False,
        commit_message: str = "Add reward model",
        token: Optional[str] = None,
    ) -> str:
        """
        Push the trained reward model to HuggingFace Hub.

        What gets saved
        ---------------
        The repo contains two distinct things:

        1. The fine-tuned backbone in HuggingFace's standard format (config.json +
           model.safetensors). This means anyone can do:
               GPT2Model.from_pretrained("your-repo")
           to get just the backbone, independent of our codebase.

        2. The scalar head weights as a separate scalar_head.pt (tiny — one Linear
           layer, a few KB). Kept separate because it is not a standard HF model
           type; it needs to be loaded alongside the backbone.

        Additionally:
            reward_model_config.json — hyperparams and eval metrics (for record-keeping)
            README.md               — auto-generated model card with metrics

        Loading later (in PPO or anywhere else)
        ----------------------------------------
            from reward_model.model import RewardModel
            model = RewardModel.from_hub("your-username/your-repo")
            model.eval()

        Args:
            repo_id:           HuggingFace repo ID, e.g. "username/reward-model-hh-rlhf".
            cfg:               OmegaConf config used during training.
            evaluation_report: Optional EvaluationReport from the final eval run.
                               Its metrics are embedded in the model card and config.
            private:           Whether to create a private repo.
            commit_message:    Commit message shown on the hub.
            token:             HuggingFace token. Defaults to HF_TOKEN env variable.

        Returns:
            URL of the HuggingFace repo.
        """
        import json
        import tempfile
        from pathlib import Path as _Path
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)

        # Create the repo if it doesn't already exist
        repo_url = create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            token=token,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = _Path(tmp)

            # ------------------------------------------------------------------
            # 1. Save backbone (standard HuggingFace format)
            # ------------------------------------------------------------------
            # save_pretrained writes: config.json + model.safetensors (or pytorch_model.bin)
            self.backbone.save_pretrained(tmp)

            # ------------------------------------------------------------------
            # 2. Save scalar head state dict
            # ------------------------------------------------------------------
            # The scalar head is nn.Sequential(Dropout, Linear(hidden_dim, 1, bias=False)).
            # Its state_dict is tiny — just one weight matrix of shape (1, hidden_dim).
            torch.save(
                self.scalar_head.state_dict(),
                tmp / "scalar_head.pt",
            )

            # ------------------------------------------------------------------
            # 3. Save reward model config (human-readable record of the run)
            # ------------------------------------------------------------------
            rm_cfg: dict = {
                "backbone":            cfg.model.backbone,
                "num_unfrozen_layers": cfg.model.num_unfrozen_layers,
                "dropout":             cfg.model.dropout,
                "max_length":          cfg.data.max_length,
                "training_dataset":    cfg.data.dataset_name,
                "loss":                "bradley_terry_stable",
                "learning_rate":       cfg.training.learning_rate,
                "batch_size":          cfg.training.batch_size,
            }
            if evaluation_report is not None:
                rm_cfg.update({
                    "preference_accuracy": round(evaluation_report.preference_accuracy, 4),
                    "mean_margin":         round(evaluation_report.mean_margin, 4),
                    "score_separation":    round(evaluation_report.score_separation, 4),
                    "n_eval_examples":     evaluation_report.n_examples,
                })

            with open(tmp / "reward_model_config.json", "w") as f:
                json.dump(rm_cfg, f, indent=2)

            # ------------------------------------------------------------------
            # 4. Generate model card (README.md)
            # ------------------------------------------------------------------
            model_card = _build_model_card(repo_id, rm_cfg, evaluation_report)
            with open(tmp / "README.md", "w") as f:
                f.write(model_card)

            # ------------------------------------------------------------------
            # 5. Upload everything to the Hub
            # ------------------------------------------------------------------
            api.upload_folder(
                folder_path=str(tmp),
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
            )

        return repo_url

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> "RewardModel":
        """
        Load a trained RewardModel directly from HuggingFace Hub.

        This is the primary loading interface for PPO and inference. It
        downloads everything it needs from the hub — no local config file or
        codebase assumptions required.

        What it does:
            1. Downloads reward_model_config.json to get architecture params.
            2. Loads the fine-tuned backbone via GPT2Model.from_pretrained
               (uses HF's standard caching — no re-download if already cached).
            3. Downloads scalar_head.pt and loads the weights.
            4. Freezes all parameters (inference mode by default).

        Args:
            repo_id: HuggingFace repo ID, e.g. "manindersingh120996/reward-model-hh-rlhf".
            token:   HuggingFace token for private repos. Defaults to HF_TOKEN env var.
            device:  Target device. Defaults to CUDA if available, else CPU.

        Returns:
            RewardModel in eval() mode with all parameters frozen.
            Call model.train() and unfreeze specific layers if you want to fine-tune further.

        Example (PPO setup)
        -------------------
        >>> from reward_model.model import RewardModel
        >>> reward_model = RewardModel.from_hub("manindersingh120996/reward-model-hh-rlhf")
        >>> reward_model.eval()
        >>>
        >>> # Score a response
        >>> score = reward_model(input_ids, attention_mask)   # shape: (batch_size,)
        """
        import json
        from huggingface_hub import hf_hub_download

        # ------------------------------------------------------------------ #
        # 1. Download and read the reward model config                        #
        # ------------------------------------------------------------------ #
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="reward_model_config.json",
            token=token,
        )
        with open(config_path) as f:
            rm_cfg = json.load(f)

        dropout = rm_cfg.get("dropout", 0.0)

        # ------------------------------------------------------------------ #
        # 2. Load fine-tuned backbone from HF standard format                #
        # ------------------------------------------------------------------ #
        # GPT2Model.from_pretrained looks for config.json + model.safetensors
        # at the repo root — exactly where save_to_hub put them.
        instance = cls.__new__(cls)
        torch.nn.Module.__init__(instance)
        instance.backbone = GPT2Model.from_pretrained(repo_id, token=token)
        instance.hidden_dim = instance.backbone.config.n_embd

        # ------------------------------------------------------------------ #
        # 3. Reconstruct scalar head and load weights                         #
        # ------------------------------------------------------------------ #
        instance.scalar_head = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(instance.hidden_dim, 1, bias=False),
        )

        scalar_path = hf_hub_download(
            repo_id=repo_id,
            filename="scalar_head.pt",
            token=token,
        )
        # weights_only=True: security best practice — don't execute arbitrary
        # Python code embedded in the checkpoint (safe for our own files, but
        # good habit for any file you didn't create yourself).
        instance.scalar_head.load_state_dict(
            torch.load(scalar_path, map_location="cpu", weights_only=True)
        )

        # ------------------------------------------------------------------ #
        # 4. Freeze everything and move to device                             #
        # ------------------------------------------------------------------ #
        # All parameters frozen by default: this model is used for inference
        # (scoring responses in PPO), not for further training.
        # If you want to fine-tune further, call model.train() and selectively
        # set requires_grad=True on the layers you want to update.
        for param in instance.parameters():
            param.requires_grad = False

        instance.eval()

        if device is not None:
            instance.to(device)
        elif torch.cuda.is_available():
            instance.to(torch.device("cuda"))

        return instance

    def parameter_stats(self) -> ParameterStats:
        """
        Return a summary of trainable vs frozen parameter counts.

        Call this immediately after constructing the model to verify your
        freezing strategy is working as intended. Print the result before
        training starts and include it in your experiment logs.

        Example
        -------
        >>> model = RewardModel(cfg)
        >>> stats = model.parameter_stats()
        >>> print(stats)
        Parameters | total: 124,439,808 | trainable: 28,328,192 (22.8%) | frozen: 96,111,616
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return ParameterStats(
            total=total,
            trainable=trainable,
            frozen=frozen,
            trainable_pct=round(100.0 * trainable / total, 2),
        )


# ---------------------------------------------------------------------------
# Model card generator (module-level helper, not a class method)
# ---------------------------------------------------------------------------

def _build_model_card(
    repo_id: str,
    rm_cfg: dict,
    evaluation_report=None,
) -> str:
    """
    Generate a HuggingFace model card (README.md) for the reward model.

    The card is generated programmatically so it always reflects the actual
    training run metrics rather than being manually written and going stale.
    """
    acc_line  = ""
    eval_section = ""

    if evaluation_report is not None:
        acc = rm_cfg.get("preference_accuracy", "N/A")
        margin = rm_cfg.get("mean_margin", "N/A")
        d = rm_cfg.get("score_separation", "N/A")

        acc_line = f"\n- **Preference accuracy**: {acc:.4f}" if isinstance(acc, float) else ""
        eval_section = f"""
## Evaluation Results

Evaluated on the hh-rlhf test split.

| Metric | Value |
|--------|-------|
| Preference Accuracy | {acc:.4f} |
| Mean Margin | {margin:.4f} |
| Cohen's d (score separation) | {d:.4f} |

**Interpretation**: A random model scores 0.50 preference accuracy. A well-trained
reward model on hh-rlhf typically reaches 0.70–0.78. Cohen's d > 0.8 indicates
the chosen and rejected score distributions are well separated.
""" if isinstance(acc, float) else ""

    return f"""---
language: en
license: apache-2.0
tags:
  - reward-model
  - rlhf
  - gpt2
  - preference-learning
  - bradley-terry
datasets:
  - Anthropic/hh-rlhf
---

# Reward Model — GPT-2 fine-tuned on hh-rlhf

A scalar reward model trained from scratch using the Bradley-Terry pairwise
ranking loss on Anthropic's hh-rlhf human preference dataset. Built as Phase 1
of a full RLHF implementation (Reward Model → PPO → DPO → GRPO).{acc_line}

## Model Description

- **Backbone**: {rm_cfg.get("backbone", "gpt2")} (fine-tuned, top {rm_cfg.get("num_unfrozen_layers", "N/A")} layers unfrozen)
- **Head**: Linear({rm_cfg.get("backbone", "gpt2")} hidden_dim → 1, no bias)
- **Training data**: [{rm_cfg.get("training_dataset", "Anthropic/hh-rlhf")}](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- **Loss**: Bradley-Terry NLL (numerically stable softplus form)
- **Max sequence length**: {rm_cfg.get("max_length", 512)} tokens
{eval_section}

## Usage

```python
from reward_model.model import RewardModel
from transformers import AutoTokenizer
import torch

# Load the reward model (all parameters frozen for inference)
reward_model = RewardModel.from_hub("{repo_id}")
reward_model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("{rm_cfg.get("backbone", "gpt2")}")
tokenizer.pad_token = tokenizer.eos_token

# Score a (prompt + response) pair
text = "\\n\\nHuman: What is 2+2?\\n\\nAssistant: The answer is 4."
encoding = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    score = reward_model(encoding["input_ids"], encoding["attention_mask"])

print(f"Reward score: {{score.item():.4f}}")
```

## Architecture

```
Input: [prompt tokens] + [response tokens]
         ↓
GPT-2 transformer backbone (layers {rm_cfg.get("num_unfrozen_layers", "N/A")} from top fine-tuned)
         ↓
Last real token hidden state (indexed via attention_mask)
         ↓
Dropout({rm_cfg.get("dropout", 0.1)}) → Linear(hidden_dim, 1, bias=False)
         ↓
scalar reward score
```

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | {rm_cfg.get("learning_rate", "N/A")} |
| Batch size | {rm_cfg.get("batch_size", "N/A")} |
| Unfrozen layers | {rm_cfg.get("num_unfrozen_layers", "N/A")} |
| Loss | Bradley-Terry (softplus stable form) |

## Files in This Repo

| File | Description |
|------|-------------|
| `config.json` | GPT-2 architecture config |
| `model.safetensors` | Fine-tuned backbone weights |
| `scalar_head.pt` | Scalar regression head weights |
| `reward_model_config.json` | Training hyperparameters and eval metrics |

## Citation

If you use this model, please cite:
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* (InstructGPT)
- Anthropic. *hh-rlhf dataset.* HuggingFace Datasets.
"""