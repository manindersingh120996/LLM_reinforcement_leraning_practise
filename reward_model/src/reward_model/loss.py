"""
Bradley-Terry Pairwise Ranking Loss.
 
Mathematical Background
-----------------------
Under the Bradley-Terry model, the probability that a human prefers a
"chosen" response over a "rejected" response is modelled as:
 
    P(chosen ≻ rejected | prompt) = σ(r_chosen - r_rejected)
 
where r(·) is the reward model's scalar output and σ is the sigmoid function.
 
We have a dataset of N human preference pairs {(prompt_i, chosen_i, rejected_i)}.
We want to find the reward function r that maximises the likelihood of all
observed preferences:
 
    L(r) = ∏ σ(r(prompt_i, chosen_i) - r(prompt_i, rejected_i))
 
Taking the negative log-likelihood (standard ML objective: minimise NLL):
 
    NLL = -∑ log σ(r_chosen_i - r_rejected_i)
 
Using the identity  -log(σ(x)) = log(1 + e^{-x}) = softplus(-x):
 
    NLL = ∑ softplus(-(r_chosen_i - r_rejected_i))
 
The softplus form is numerically stable: it never produces Inf or NaN even
when score_diff is a large negative number (which happens early in training).
The naive form breaks here because log(σ(-100)) = log(≈0) → -inf in float32.
 
Reference
---------
Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block
designs: I. The method of paired comparisons. Biometrika, 39(3/4), 324–345.
 
Ouyang, L. et al. (2022). Training language models to follow instructions
with human feedback. NeurIPS. (InstructGPT)
"""


from dataclasses import dataclass
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
@dataclass
class BradleyTerryLossOutput:
    """
    Structured output from BradleyTerryLoss.forward().
 
    Using a dataclass instead of returning a plain tuple means:
    - Callers access fields by name (output.loss), not by index (output[0])
    - Adding a new diagnostic field later won't break existing callers
    - The return type is self-documenting
 
    Attributes
    ----------
    loss : torch.Tensor
        Scalar mean NLL loss. This is the value you call .backward() on.
    chosen_scores : torch.Tensor
        Shape (batch_size,). Raw reward scores for chosen responses.
    rejected_scores : torch.Tensor
        Shape (batch_size,). Raw reward scores for rejected responses.
    preference_accuracy : torch.Tensor
        Scalar in [0, 1]. Fraction of pairs where chosen_score > rejected_score.
        This is your primary eval metric — a well-trained reward model should
        approach 0.70–0.80 on held-out data (higher is overfitting).
    mean_margin : torch.Tensor
        Scalar. Mean of (chosen_score - rejected_score) across the batch.
        Tracks whether the model is learning to separate scores, not just rank
        them correctly. A model can have 100% accuracy but near-zero margin
        (dangerous: vulnerable to any perturbation). You want both high.
    """
 
    loss: torch.Tensor
    chosen_scores: torch.Tensor
    rejected_scores: torch.Tensor
    preference_accuracy: torch.Tensor
    mean_margin: torch.Tensor
 
class BradleyTerryLoss(nn.Module):
    """
    Pairwise ranking loss implementing the Bradley-Terry model.
 
    Two implementations are provided to illustrate the numerical stability issue:
 
    Naive form:
        loss = -log(σ(r_chosen - r_rejected))
        ⚠️  Produces NaN/Inf when score differences are very negative (early training).
 
    Stable form (default):
        loss = softplus(-(r_chosen - r_rejected))
        ✅  Mathematically equivalent, numerically safe for all inputs.
 
    In production, always use use_stable=True. The naive form is included
    purely to make the stability problem visible in tests.
 
    Parameters
    ----------
    use_stable : bool
        If True (default), use the softplus numerically stable form.
        If False, use the naive -log(sigmoid) form.
 
    Example
    -------
    >>> loss_fn = BradleyTerryLoss(use_stable=True)
    >>> chosen_scores = torch.tensor([2.0, 1.5, 3.0])   # shape: (batch_size,)
    >>> rejected_scores = torch.tensor([0.5, 0.8, 1.0]) # shape: (batch_size,)
    >>> output = loss_fn(chosen_scores, rejected_scores)
    >>> output.loss.backward()  # gradients flow back to the reward model
    """
    def __init__(self, use_stable:bool = True)->None:
        super().__init__()
        self.use_stable = self.use_stable

    def forward(
            self,
            chosen_scores: torch.Tensor,
            rejected_scores : torch.Tensor,

    ) -> BradleyTerryLossOutput:
        """
        Compute the Bradley-Terry pairwise ranking loss.
 
        Parameters
        ----------
        chosen_scores : torch.Tensor
            Shape (batch_size,). Scalar reward output for chosen responses.
            These come from reward_model(prompt, chosen_response).
        rejected_scores : torch.Tensor
            Shape (batch_size,). Scalar reward output for rejected responses.
            These come from reward_model(prompt, rejected_response).
 
        Returns
        -------
        BradleyTerryLossOutput
            Structured output containing loss and diagnostic metrics.
 
        Raises
        ------
        AssertionError
            If scores are not 1D or have mismatched shapes.
        """

        self._validate_inputs(chosen_scores,rejected_scores)

        # Core quantity: how much more does the model prefer chosen over rejected?
        # Shape: (batch_size,)
        score_diff = chosen_scores - rejected_scores

        if self.use_stable:
            # Stable : -log(sigma(x)) = log(1 + e^{-x}) = softplus(-x)
            # F.softplus handles the log-sum-exp trick internally to avoid overflow
            # Gradient of softplus(-x) w.r.t x = -sigma(-x) = -(1-sigma(x))
            # This is well-behaved everywhere and doesn't face the vanishing gradient problem
            per_pair_loss = F.softplus(-score_diff) # shape (batch_size,)
        else:
            # Naive: directly compute -log(sigma(score_diff))
            # ⚠️  When score_diff = -100: σ(-100) ≈ 3.7e-44 → log(3.7e-44) → -inf
            # float32 underflows to 0 before the log, giving log(0) = -inf.
            # This kills training silently in early epochs.
            per_pair_loss = -torch.log(torch.sigmoid(score_diff))

        # Average over the batch — makes loss scale-invariant to batch size
        loss = per_pair_loss.mean()

        # Diagnostic  metrics : computed without gradients since they're for monitorying only
        # torch.no_grad() , here is not just and optimisation , it signals that therser values
        # do not participate in backprop.
        with torch.no_grad():
            # Fraction of pairs where the model correctly ranks chosen over rejected.
            # This is the single most important metric for reward model quality.
            preference_accuracy = (score_diff > 0).float().mean()
            # Average score gap between chosen and rejected.
            # High accuracy + low margin = fragile model. High both = robust model.
            mean_margin = score_diff.mean()

        return BradleyTerryLossOutput(
            loss = loss,
            chosen_scores=chosen_scores,
            rejected_scores=rejected_scores,
            preference_accuracy=preference_accuracy,
            mean_margin=mean_margin
        )
    
    def _validate_inputs(
            self,
            chosen_scores: torch.Tensor,
            rejected_scores: torch.Tensor,
    ) -> None:
        """
        Validate tensor shapes before computation.
 
        Fail fast with a clear error message rather than letting PyTorch
        produce a cryptic broadcast error mid-computation.
        """
        assert chosen_scores.dim() == 1, (
            f"chosen_scores must be 1D (batch_size,). "
            f"Got shape {chosen_scores.shape}. "
            f"Did you forget to squeeze() the model output?"
        )
        assert rejected_scores.dim() == 1, (
            f"rejected_scores must be 1D (batch_size,). "
            f"Got shape {rejected_scores.shape}. "
            f"Did you forget to squeeze() the model output?"
        )
        assert chosen_scores.shape == rejected_scores.shape, (
            f"chosen_scores and rejected_scores must have the same shape. "
            f"Got chosen={chosen_scores.shape}, rejected={rejected_scores.shape}."
        )



 