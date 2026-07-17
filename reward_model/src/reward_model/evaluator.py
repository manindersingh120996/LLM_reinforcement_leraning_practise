"""
Standalone reward model evaluator.

Separate from the trainer so you can evaluate any checkpoint at any time
without needing to reconstruct a training loop. The evaluator's job is to
answer the question: "how good is this reward model?" across multiple
dimensions, not just preference accuracy.

Metrics produced
----------------
Primary:
    preference_accuracy  — fraction of pairs where chosen_score > rejected_score.
                           This is the most important single number. A random model
                           scores 0.50. A good reward model reaches 0.72–0.78 on
                           hh-rlhf. Above 0.80 often indicates overfitting.

    mean_margin          — mean(chosen_score - rejected_score).
                           High accuracy + low margin = fragile model.
                           Both should be high for a robust reward model.

    mean_loss            — mean Bradley-Terry NLL on the evaluation set.

Distribution (for diagnosing what the model has learned):
    chosen_mean/std      — statistics of chosen response scores.
    rejected_mean/std    — statistics of rejected response scores.

    score_separation     — Cohen's d: (chosen_mean - rejected_mean) / pooled_std.
                           Interpretation:
                               d < 0.2  → negligible separation (bad)
                               d ~ 0.5  → moderate separation (okay)
                               d ~ 0.8  → large separation (good)
                               d > 1.0  → very strong separation (great)
                           This is more informative than accuracy alone because it
                           measures HOW MUCH the model separates the distributions,
                           not just whether it ranks them correctly on average.

Usage
-----
>>> evaluator = RewardModelEvaluator(model, loss_fn)
>>> report = evaluator.evaluate(val_loader)
>>> print(report.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from reward_model.dataset import PreferenceBatch
from reward_model.loss import BradleyTerryLoss
from reward_model.model import RewardModel


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """
    Complete evaluation report for a reward model checkpoint.

    Structured as a dataclass so fields are accessible by name, the report
    can be serialised to a dict for logging, and adding new metrics later
    doesn't break existing code that uses specific fields.
    """

    # Primary metrics
    preference_accuracy: float    # fraction where chosen_score > rejected_score
    mean_loss: float              # mean Bradley-Terry NLL
    mean_margin: float            # mean(chosen_score - rejected_score)

    # Score distribution stats
    chosen_mean: float            # mean reward score for chosen responses
    chosen_std: float             # std of chosen scores
    rejected_mean: float          # mean reward score for rejected responses
    rejected_std: float           # std of rejected scores

    # Separation quality
    score_separation: float       # Cohen's d between chosen and rejected distributions

    # Bookkeeping
    n_examples: int               # total pairs evaluated
    n_correct: int                # pairs where chosen_score > rejected_score

    def summary(self) -> str:
        """
        Human-readable summary for printing to console or logs.

        Format mirrors what you'd want to see in a training log:
        the most important metric first, supporting metrics after.
        """
        lines = [
            "=" * 55,
            "Reward Model Evaluation Report",
            "=" * 55,
            f"  Examples evaluated : {self.n_examples:,}",
            f"  Correct rankings   : {self.n_correct:,}",
            "",
            "  Primary Metrics",
            f"    Preference accuracy : {self.preference_accuracy:.4f}",
            f"    Mean loss           : {self.mean_loss:.4f}",
            f"    Mean margin         : {self.mean_margin:+.4f}",
            "",
            "  Score Distributions",
            f"    Chosen   : mean={self.chosen_mean:+.4f}  std={self.chosen_std:.4f}",
            f"    Rejected : mean={self.rejected_mean:+.4f}  std={self.rejected_std:.4f}",
            "",
            "  Separation",
            f"    Cohen's d : {self.score_separation:.4f}  "
            f"({'negligible' if self.score_separation < 0.2 else 'moderate' if self.score_separation < 0.5 else 'large' if self.score_separation < 0.8 else 'very large'})",
            "=" * 55,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to a flat dict for logging to wandb, mlflow, etc."""
        return {
            "eval/preference_accuracy": self.preference_accuracy,
            "eval/mean_loss":           self.mean_loss,
            "eval/mean_margin":         self.mean_margin,
            "eval/chosen_mean":         self.chosen_mean,
            "eval/chosen_std":          self.chosen_std,
            "eval/rejected_mean":       self.rejected_mean,
            "eval/rejected_std":        self.rejected_std,
            "eval/score_separation":    self.score_separation,
            "eval/n_examples":          self.n_examples,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RewardModelEvaluator:
    """
    Computes evaluation metrics for a trained reward model.

    Designed to be used independently of the training loop. You can point it
    at any checkpoint and any DataLoader.

    Args:
        model   : Trained RewardModel. Must already be on the correct device.
        loss_fn : BradleyTerryLoss instance (use_stable=True recommended).
        device  : Device to run evaluation on. Defaults to wherever the model is.

    Example
    -------
    >>> evaluator = RewardModelEvaluator(model, loss_fn, device)
    >>> report = evaluator.evaluate(val_loader)
    >>> print(report.summary())
    >>> metrics = report.to_dict()   # log to wandb, mlflow, etc.
    """

    def __init__(
        self,
        model: RewardModel,
        loss_fn: BradleyTerryLoss,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def evaluate(self, val_loader) -> EvaluationReport:
        """
        Run evaluation on the full val set and return an EvaluationReport.

        The entire method is wrapped in @torch.no_grad() — no gradient tracking,
        no computation graph, significantly lower memory usage during eval.

        Args:
            val_loader: Any DataLoader that yields PreferenceBatch objects.

        Returns:
            EvaluationReport with all metrics populated.
        """
        self.model.eval()

        # Accumulators — collect raw scores across all batches
        # We store all scores so we can compute std correctly at the end.
        # Incremental std computation exists but is numerically less stable.
        all_chosen_scores:   list[torch.Tensor] = []
        all_rejected_scores: list[torch.Tensor] = []
        all_losses:          list[float] = []
        n_total = 0
        n_correct = 0

        for batch in val_loader:
            c_ids  = batch.chosen_input_ids.to(self.device)
            c_mask = batch.chosen_attention_mask.to(self.device)
            r_ids  = batch.rejected_input_ids.to(self.device)
            r_mask = batch.rejected_attention_mask.to(self.device)

            # Align lengths for the batching trick
            c_ids, r_ids, c_mask, r_mask = self._align(c_ids, r_ids, c_mask, r_mask)

            all_scores = self.model(
                torch.cat([c_ids, r_ids], dim=0),
                torch.cat([c_mask, r_mask], dim=0),
            )
            B = c_ids.shape[0]
            chosen_scores   = all_scores[:B]
            rejected_scores = all_scores[B:]

            loss_out = self.loss_fn(chosen_scores, rejected_scores)

            all_chosen_scores.append(chosen_scores.cpu())
            all_rejected_scores.append(rejected_scores.cpu())
            all_losses.append(loss_out.loss.item() * B)   # weight by batch size

            n_correct += (chosen_scores > rejected_scores).sum().item()
            n_total   += B

        self.model.train()

        # Concatenate all collected scores into single tensors
        chosen_all   = torch.cat(all_chosen_scores)    # (n_total,)
        rejected_all = torch.cat(all_rejected_scores)  # (n_total,)

        chosen_mean  = chosen_all.mean().item()
        chosen_std   = chosen_all.std().item()
        rejected_mean = rejected_all.mean().item()
        rejected_std  = rejected_all.std().item()

        return EvaluationReport(
            preference_accuracy = n_correct / n_total,
            mean_loss           = sum(all_losses) / n_total,
            mean_margin         = (chosen_all - rejected_all).mean().item(),
            chosen_mean         = chosen_mean,
            chosen_std          = chosen_std,
            rejected_mean       = rejected_mean,
            rejected_std        = rejected_std,
            score_separation    = _cohens_d(
                chosen_mean, chosen_std,
                rejected_mean, rejected_std,
                n_total, n_total,
            ),
            n_examples = n_total,
            n_correct  = n_correct,
        )

    def _align(
        self,
        c_ids:  torch.Tensor,
        r_ids:  torch.Tensor,
        c_mask: torch.Tensor,
        r_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-pad chosen or rejected to the same seq_len for concatenation."""
        cL, rL = c_ids.shape[1], r_ids.shape[1]
        if cL == rL:
            return c_ids, r_ids, c_mask, r_mask

        target = max(cL, rL)

        def pad(t: torch.Tensor) -> torch.Tensor:
            n = target - t.shape[1]
            return t if n == 0 else torch.cat(
                [t, torch.zeros(t.shape[0], n, dtype=t.dtype, device=t.device)],
                dim=1,
            )

        return pad(c_ids), pad(r_ids), pad(c_mask), pad(r_mask)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _cohens_d(
    mean_a: float,
    std_a:  float,
    mean_b: float,
    std_b:  float,
    n_a:    int,
    n_b:    int,
) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d = (mean_A - mean_B) / pooled_std

    The pooled standard deviation weights each group's variance by its
    sample size, giving a more accurate estimate when group sizes differ.

    Interpretation (Cohen's original conventions):
        |d| < 0.2  → negligible
        |d| ~ 0.5  → medium
        |d| ~ 0.8  → large

    For reward models, d > 0.8 indicates the chosen and rejected score
    distributions are well-separated — the model has learned a strong,
    robust quality signal.

    Returns 0.0 if either std is 0 (degenerate case — all scores identical).
    """
    # Pooled variance: weighted average of individual variances
    pooled_var = ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
    pooled_std = math.sqrt(pooled_var)

    if pooled_std < 1e-10:
        return 0.0

    return (mean_a - mean_b) / pooled_std