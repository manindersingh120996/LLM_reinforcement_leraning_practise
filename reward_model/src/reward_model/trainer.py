"""
RewardModelTrainer: wires together model, loss, and data loaders.

Responsibilities
----------------
- Run forward passes for chosen and rejected responses (batching trick)
- Compute Bradley-Terry loss
- Backpropagate and clip gradients
- Step AdamW optimizer and learning rate scheduler
- Periodically evaluate on the validation set
- Save checkpoints (model + optimizer + scheduler state)
- Log training metrics at every step

The batching trick (explained in _train_step)
---------------------------------------------
Instead of calling self.model twice per batch (once for chosen, once for
rejected), we concatenate them in the batch dimension and call the model once.
This halves the number of transformer forward passes and improves GPU utilisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from reward_model.dataset import PreferenceBatch
from reward_model.loss import BradleyTerryLoss
from reward_model.model import RewardModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainStepOutput:
    """
    Metrics from a single training step.

    All values are Python floats (detached from the computation graph).
    Returning a structured dataclass instead of a plain dict means:
    - Field names are explicit and type-checkable
    - Adding a new field later doesn't break callers that don't use it
    - Tests can access specific fields by name without magic strings
    """

    loss: float
    preference_accuracy: float      # fraction of pairs where chosen > rejected
    mean_margin: float              # mean(chosen_score - rejected_score)
    grad_norm: float                # global grad norm BEFORE clipping
    learning_rate: float            # LR after this step's scheduler update


@dataclass
class EvalOutput:
    """Aggregated metrics from a full evaluation pass over the validation set."""

    mean_loss: float
    preference_accuracy: float
    mean_margin: float
    n_examples: int                 # total val examples scored this eval


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RewardModelTrainer:
    """
    Training loop for the reward model.

    Parameters
    ----------
    model : RewardModel
        The model to train. Moved to `device` in __init__.
    loss_fn : BradleyTerryLoss
        The pairwise ranking loss.
    train_loader : DataLoader
        DataLoader over the training PreferenceDataset.
    val_loader : DataLoader
        DataLoader over the validation PreferenceDataset.
    cfg : DictConfig
        OmegaConf config. Reads cfg.training.* and cfg.logging.*
    device : torch.device, optional
        Target device. Defaults to CUDA if available, else CPU.

    Usage
    -----
    >>> trainer = RewardModelTrainer(model, loss_fn, train_loader, val_loader, cfg)
    >>> trainer.train()               # full training run
    >>> output = trainer._train_step(batch)  # single step (useful for testing)
    >>> metrics = trainer._evaluate()        # full val evaluation
    """

    def __init__(
        self,
        model: RewardModel,
        loss_fn: BradleyTerryLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        # AdamW: decoupled weight decay, correct for LLM fine-tuning.
        # Only pass parameters that require gradients — frozen params would
        # waste memory in the optimizer's momentum buffers.
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Linear warmup → linear decay.
        # Warmup: LR goes from ~0 to learning_rate over warmup_steps.
        # Decay: LR decreases linearly from learning_rate to 0 by the end.
        total_steps = len(train_loader) * cfg.training.max_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=total_steps,
        )

        # Training state — tracked so checkpoints can resume exactly
        self.global_step = 0
        self.best_val_accuracy = 0.0

        Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop for cfg.training.max_epochs epochs."""
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        for epoch in range(self.cfg.training.max_epochs):
            logger.info(f"=== Epoch {epoch + 1}/{self.cfg.training.max_epochs} ===")

            for batch in self.train_loader:
                step_out = self._train_step(batch)
                self.global_step += 1

                if self.global_step % self.cfg.logging.log_every_n_steps == 0:
                    self._log_train(step_out)

                if self.global_step % self.cfg.training.eval_every_n_steps == 0:
                    eval_out = self._evaluate()
                    self._log_eval(eval_out)
                    if eval_out.preference_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = eval_out.preference_accuracy
                        self._save_checkpoint(step=self.global_step, tag="best")

                if self.global_step % self.cfg.training.save_every_n_steps == 0:
                    self._save_checkpoint(
                        step=self.global_step, tag=f"step_{self.global_step}"
                    )

        logger.info(f"Training complete. Best val accuracy: {self.best_val_accuracy:.4f}")

    # -----------------------------------------------------------------------
    # Core training step
    # -----------------------------------------------------------------------

    def _train_step(self, batch: PreferenceBatch) -> TrainStepOutput:
        """
        One training step: forward → loss → backward → clip → optimizer.

        The batching trick
        ------------------
        The model processes every sequence independently (no cross-sequence
        attention). So instead of two separate forward passes:

            chosen_scores   = model(chosen_ids,   chosen_mask)   # pass 1
            rejected_scores = model(rejected_ids, rejected_mask)  # pass 2

        We concatenate in the batch dimension and do one pass:

            all_ids    = cat([chosen_ids,   rejected_ids],   dim=0)  # (2B, L)
            all_masks  = cat([chosen_mask,  rejected_mask],  dim=0)  # (2B, L)
            all_scores = model(all_ids, all_masks)                   # (2B,)
            chosen_scores, rejected_scores = all_scores.chunk(2)     # (B,), (B,)

        This halves the number of backbone forward passes per step.

        The alignment step
        ------------------
        The collator pads chosen and rejected to their own max lengths within
        each group. A batch might have chosen.shape=(B, 45), rejected=(B, 62).
        You cannot concatenate (B, 45) and (B, 62) on dim=0.
        _align_seq_lengths pads the shorter side to match the longer, with
        zeros in both input_ids and attention_mask positions.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        c_ids  = batch.chosen_input_ids.to(self.device)
        c_mask = batch.chosen_attention_mask.to(self.device)
        r_ids  = batch.rejected_input_ids.to(self.device)
        r_mask = batch.rejected_attention_mask.to(self.device)

        # Align sequence lengths so cat works
        c_ids, r_ids, c_mask, r_mask = self._align_seq_lengths(
            c_ids, r_ids, c_mask, r_mask
        )

        # Single forward pass
        all_scores = self.model(
            torch.cat([c_ids, r_ids], dim=0),
            torch.cat([c_mask, r_mask], dim=0),
        )
        B = c_ids.shape[0]
        loss_out = self.loss_fn(all_scores[:B], all_scores[B:])
        loss_out.loss.backward()

        # Log pre-clip grad norm (diagnostic — shows whether clipping engaged)
        grad_norm = self._grad_norm()

        if self.cfg.training.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.gradient_clip_norm,
            )

        self.optimizer.step()
        self.scheduler.step()

        return TrainStepOutput(
            loss=loss_out.loss.item(),
            preference_accuracy=loss_out.preference_accuracy.item(),
            mean_margin=loss_out.mean_margin.item(),
            grad_norm=grad_norm,
            learning_rate=self.scheduler.get_last_lr()[0],
        )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> EvalOutput:
        """
        Run the full validation set and return aggregated metrics.

        Two separate mechanisms work together here:

        @torch.no_grad()
            Disables gradient tracking for the entire method. No computation
            graph is built. Memory usage drops significantly — you don't need
            to store intermediate activations for a backward pass that will
            never happen.

        model.eval()
            Disables dropout. Without this, each forward pass produces
            slightly different scores due to dropout randomness, making
            evaluation metrics noisy and non-reproducible.

        We restore model.train() at the end so the next _train_step works
        correctly. Not restoring this is a common subtle bug — the model
        stays in eval() mode, dropout is always off during training, and
        the model overfits faster than expected.
        """
        self.model.eval()

        total_loss, total_acc, total_margin, n_total = 0.0, 0.0, 0.0, 0

        for batch in self.val_loader:
            c_ids  = batch.chosen_input_ids.to(self.device)
            c_mask = batch.chosen_attention_mask.to(self.device)
            r_ids  = batch.rejected_input_ids.to(self.device)
            r_mask = batch.rejected_attention_mask.to(self.device)

            c_ids, r_ids, c_mask, r_mask = self._align_seq_lengths(
                c_ids, r_ids, c_mask, r_mask
            )

            all_scores = self.model(
                torch.cat([c_ids, r_ids], dim=0),
                torch.cat([c_mask, r_mask], dim=0),
            )
            B = c_ids.shape[0]
            loss_out = self.loss_fn(all_scores[:B], all_scores[B:])

            # Weighted accumulation so final metrics are correct averages
            # regardless of whether the last batch is smaller than the rest.
            total_loss   += loss_out.loss.item() * B
            total_acc    += loss_out.preference_accuracy.item() * B
            total_margin += loss_out.mean_margin.item() * B
            n_total      += B

        self.model.train()  # always restore — leaving eval() mode on is a bug

        return EvalOutput(
            mean_loss=total_loss / n_total,
            preference_accuracy=total_acc / n_total,
            mean_margin=total_margin / n_total,
            n_examples=n_total,
        )

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, step: int, tag: str) -> None:
        """
        Save model, optimizer, and scheduler state.

        Why save all three (not just model weights)?

        Model weights alone let you do inference. But if you resume training
        from weights-only, the optimizer starts cold: its momentum buffers are
        zeroed, and its adaptive LR estimates are reset. This causes a
        temporary loss spike — the "cold restart" effect — until the optimizer
        re-warms. Saving the optimizer state makes resuming seamless.

        The scheduler state saves your position in the warmup/decay curve.
        Without it, LR resets to the warmup start value when you resume.

        The config is saved for reproducibility. Six months later, when you
        can't remember what hyperparameters produced a checkpoint, it's there.
        """
        checkpoint = {
            "step":                   step,
            "model_state_dict":       self.model.state_dict(),
            "optimizer_state_dict":   self.optimizer.state_dict(),
            "scheduler_state_dict":   self.scheduler.state_dict(),
            "best_val_accuracy":      self.best_val_accuracy,
            "cfg":                    OmegaConf.to_container(self.cfg, resolve=True),
        }
        path = Path(self.cfg.training.output_dir) / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint to resume training.

        Restores model weights, optimizer state, scheduler state,
        global step counter, and best validation accuracy.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step      = ckpt["step"]
        self.best_val_accuracy = ckpt.get("best_val_accuracy", 0.0)
        logger.info(f"Resumed from step {self.global_step}")

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _align_seq_lengths(
        self,
        c_ids:  torch.Tensor,
        r_ids:  torch.Tensor,
        c_mask: torch.Tensor,
        r_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Right-pad chosen or rejected so both have the same sequence length.

        The collator pads chosen and rejected to their own group's max length
        independently. This means chosen might be (B, 45) and rejected (B, 62).
        The batching trick requires we concatenate on dim=0, which requires
        dim=1 to match. We pad the shorter side to match the longer.

        Padding is with zeros for both input_ids and attention_mask.
        Zero in attention_mask means "ignore this position" — the model will
        not attend to the added padding positions.
        The specific value used in input_ids for these positions doesn't matter
        since they're masked out anyway.
        """
        cL, rL = c_ids.shape[1], r_ids.shape[1]
        if cL == rL:
            return c_ids, r_ids, c_mask, r_mask

        target = max(cL, rL)

        def pad(t: torch.Tensor) -> torch.Tensor:
            n = target - t.shape[1]
            if n == 0:
                return t
            return torch.cat(
                [t, torch.zeros(t.shape[0], n, dtype=t.dtype, device=t.device)],
                dim=1,
            )

        return pad(c_ids), pad(r_ids), pad(c_mask), pad(r_mask)

    def _grad_norm(self) -> float:
        """
        Global gradient L2 norm across all parameters.

        Computed BEFORE clipping so the raw norm is visible in logs.
        Useful for diagnosing:
            - Exploding gradients: norm spikes to >10 or Inf
            - Dead gradients: norm near 0 (learning stopped)
            - Stable training: norm consistently below gradient_clip_norm
        """
        total = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        )
        return total ** 0.5

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_train(self, out: TrainStepOutput) -> None:
        logger.info(
            f"step={self.global_step:5d} | "
            f"loss={out.loss:.4f} | "
            f"acc={out.preference_accuracy:.3f} | "
            f"margin={out.mean_margin:+.3f} | "
            f"grad={out.grad_norm:.3f} | "
            f"lr={out.learning_rate:.2e}"
        )

    def _log_eval(self, out: EvalOutput) -> None:
        logger.info(
            f"[EVAL] step={self.global_step:5d} | "
            f"loss={out.mean_loss:.4f} | "
            f"acc={out.preference_accuracy:.3f} | "
            f"margin={out.mean_margin:+.3f} | "
            f"n={out.n_examples}"
        )