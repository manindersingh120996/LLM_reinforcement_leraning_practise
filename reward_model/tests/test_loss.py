"""
Unit tests for BradleyTerryLoss.


Tests are grouped by what they verify:
    1. Output contract      — shapes, types, dataclass structure
    2. Mathematical         — correctness of the loss computation
    3. Gradient flow        — training will actually work
    4. Numerical stability  — stable vs naive implementations
    5. Input validation     — defensive error handling
"""

import math

import pytest
import torch
import torch.nn.functional as F

from reward_model.loss import BradleyTerryLoss, BradleyTerryLossOutput


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def naive_loss() -> BradleyTerryLoss:
    """Bradley-Terry loss using the naive -log(sigmoid) form."""
    return BradleyTerryLoss(use_stable=False)


@pytest.fixture
def stable_loss() -> BradleyTerryLoss:
    """Bradley-Terry loss using the numerically stable softplus form."""
    return BradleyTerryLoss(use_stable=True)


@pytest.fixture
def batch_scores(batch_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Random chosen and rejected scores for a generic batch."""
    torch.manual_seed(42)
    return torch.randn(batch_size), torch.randn(batch_size)


# =============================================================================
# 1. Output Contract Tests
# =============================================================================

class TestOutputContract:
    """Verify the structure and shapes of what BradleyTerryLoss returns."""

    def test_returns_dataclass(self, stable_loss: BradleyTerryLoss) -> None:
        """Output must be a BradleyTerryLossOutput, not a plain tensor or tuple."""
        chosen = torch.tensor([1.0, 2.0, 3.0])
        rejected = torch.tensor([0.5, 1.0, 1.5])
        output = stable_loss(chosen, rejected)
        assert isinstance(output, BradleyTerryLossOutput), (
            f"Expected BradleyTerryLossOutput, got {type(output)}"
        )

    def test_loss_is_scalar(self, stable_loss: BradleyTerryLoss) -> None:
        """
        Loss must be a scalar (0-dimensional) tensor.
        optimizer.step() requires loss.backward(), which requires a scalar.
        """
        chosen = torch.randn(8)
        rejected = torch.randn(8)
        output = stable_loss(chosen, rejected)
        assert output.loss.shape == torch.Size([]), (
            f"loss must be scalar. Got shape {output.loss.shape}"
        )

    def test_score_fields_preserved(self, stable_loss: BradleyTerryLoss) -> None:
        """chosen_scores and rejected_scores in output must be the inputs passed in."""
        chosen = torch.tensor([1.0, 2.0, 3.0])
        rejected = torch.tensor([0.5, 1.0, 1.5])
        output = stable_loss(chosen, rejected)
        torch.testing.assert_close(output.chosen_scores, chosen)
        torch.testing.assert_close(output.rejected_scores, rejected)

    def test_diagnostic_metrics_are_scalar(self, stable_loss: BradleyTerryLoss) -> None:
        """preference_accuracy and mean_margin must be scalars (aggregates over batch)."""
        chosen = torch.randn(16)
        rejected = torch.randn(16)
        output = stable_loss(chosen, rejected)
        assert output.preference_accuracy.shape == torch.Size([])
        assert output.mean_margin.shape == torch.Size([])


# =============================================================================
# 2. Mathematical Correctness Tests
# =============================================================================

class TestMathematicalCorrectness:
    """
    Verify that the loss behaves exactly as derived from the Bradley-Terry model.
    These tests don't just check "it runs" — they check "it is correct."
    """

    def test_chance_level_equals_log_two(self, stable_loss: BradleyTerryLoss) -> None:
        """
        When chosen_score == rejected_score (score_diff = 0):
            σ(0) = 0.5
            -log(0.5) = log(2) ≈ 0.6931

        This is the entropy of a fair coin — the theoretical maximum uncertainty.
        Every training run should start near this value and decrease from there.
        If your initial loss is wildly different, something is wrong.
        """
        # Use a large batch so the mean is very precise
        zero_diff = torch.zeros(10_000)
        output = stable_loss(zero_diff, zero_diff)

        expected_loss = math.log(2)  # ≈ 0.6931
        assert abs(output.loss.item() - expected_loss) < 1e-4, (
            f"At chance, loss should equal log(2)={expected_loss:.4f}. "
            f"Got {output.loss.item():.4f}"
        )

    def test_perfect_predictions_approach_zero_loss(
        self, stable_loss: BradleyTerryLoss
    ) -> None:
        """
        When chosen_score >> rejected_score, σ(score_diff) → 1 and loss → 0.
        A perfectly calibrated reward model would achieve loss = 0.
        """
        chosen = torch.full((32,), 10.0)
        rejected = torch.full((32,), -10.0)
        output = stable_loss(chosen, rejected)

        assert output.loss.item() < 1e-3, (
            f"Perfect predictions should give near-zero loss. "
            f"Got {output.loss.item():.6f}"
        )

    def test_inverted_predictions_give_high_loss(
        self, stable_loss: BradleyTerryLoss
    ) -> None:
        """
        When rejected_score >> chosen_score (model is maximally wrong), loss is large.
        This is the "early training failure" regime the KL penalty later guards against.
        """
        chosen = torch.full((32,), -10.0)
        rejected = torch.full((32,), 10.0)
        output = stable_loss(chosen, rejected)

        # Loss should be approximately 20 here (softplus(20) ≈ 20 for large x)
        assert output.loss.item() > 10.0, (
            f"Maximally wrong predictions should give very high loss. "
            f"Got {output.loss.item():.4f}"
        )

    def test_loss_decreases_as_margin_increases(
        self, stable_loss: BradleyTerryLoss
    ) -> None:
        """
        As the gap between chosen and rejected scores grows, loss must decrease
        monotonically. This is the fundamental learning signal: the model is
        rewarded for increasing the margin, not just getting the sign right.
        """
        margins = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        losses = []

        for margin in margins:
            chosen = torch.full((64,), margin / 2)
            rejected = torch.full((64,), -margin / 2)
            output = stable_loss(chosen, rejected)
            losses.append(output.loss.item())

        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1], (
                f"Loss should decrease as margin increases. "
                f"At margin={margins[i]}: {losses[i]:.4f}, "
                f"at margin={margins[i+1]}: {losses[i+1]:.4f}"
            )

    def test_loss_is_always_positive(self, stable_loss: BradleyTerryLoss) -> None:
        """
        NLL is always non-negative. loss = 0 is the theoretical lower bound.
        If we ever get negative loss, something is deeply wrong.
        """
        torch.manual_seed(0)
        for trial in range(20):
            chosen = torch.randn(16)
            rejected = torch.randn(16)
            output = stable_loss(chosen, rejected)
            assert output.loss.item() >= 0.0, (
                f"Loss must be non-negative. Got {output.loss.item():.6f} on trial {trial}"
            )

    def test_preference_accuracy_all_correct(
        self, stable_loss: BradleyTerryLoss
    ) -> None:
        """When chosen > rejected for every pair, accuracy must be 1.0."""
        chosen = torch.tensor([2.0, 3.0, 4.0, 5.0])
        rejected = torch.tensor([1.0, 1.0, 1.0, 1.0])
        output = stable_loss(chosen, rejected)
        assert output.preference_accuracy.item() == 1.0

    def test_preference_accuracy_all_wrong(self, stable_loss: BradleyTerryLoss) -> None:
        """When chosen < rejected for every pair, accuracy must be 0.0."""
        chosen = torch.tensor([1.0, 1.0, 1.0, 1.0])
        rejected = torch.tensor([2.0, 3.0, 4.0, 5.0])
        output = stable_loss(chosen, rejected)
        assert output.preference_accuracy.item() == 0.0

    def test_mean_margin_is_correct(self, stable_loss: BradleyTerryLoss) -> None:
        """mean_margin should equal mean(chosen_scores - rejected_scores)."""
        chosen = torch.tensor([3.0, 2.0, 1.0])
        rejected = torch.tensor([1.0, 1.0, 1.0])
        output = stable_loss(chosen, rejected)

        expected_margin = (chosen - rejected).mean()
        torch.testing.assert_close(output.mean_margin, expected_margin)


# =============================================================================
# 3. Gradient Flow Tests
# =============================================================================

class TestGradientFlow:
    """
    Verify that gradients flow correctly through the loss.
    Without this, optimizer.step() does nothing, and training silently fails.
    """

    def test_gradients_flow_through_stable_loss(
        self, stable_loss: BradleyTerryLoss
    ) -> None:
        """Backward pass must produce non-None, non-NaN gradients."""
        chosen = torch.randn(8, requires_grad=True)
        rejected = torch.randn(8, requires_grad=True)

        output = stable_loss(chosen, rejected)
        output.loss.backward()

        assert chosen.grad is not None, "Gradient did not reach chosen_scores"
        assert rejected.grad is not None, "Gradient did not reach rejected_scores"
        assert not torch.isnan(chosen.grad).any(), "NaN gradient in chosen_scores"
        assert not torch.isnan(rejected.grad).any(), "NaN gradient in rejected_scores"

    def test_gradients_flow_through_naive_loss(
        self, naive_loss: BradleyTerryLoss
    ) -> None:
        """Naive form should also produce valid gradients for moderate inputs."""
        chosen = torch.randn(8, requires_grad=True)
        rejected = torch.randn(8, requires_grad=True)

        output = naive_loss(chosen, rejected)
        output.loss.backward()

        assert chosen.grad is not None
        assert rejected.grad is not None

    def test_gradient_direction_is_correct(self, stable_loss: BradleyTerryLoss) -> None:
        """
        The gradient of the loss w.r.t. chosen_scores must be negative.
        This means: to minimise the loss, the optimiser increases chosen_scores.
        That is exactly the learning signal we want from a reward model.

        Gradient derivation:
            loss = softplus(-(r_c - r_r)) = softplus(r_r - r_c)
            d(loss)/d(r_c) = -σ(r_r - r_c) < 0  ← always negative
            d(loss)/d(r_r) = +σ(r_r - r_c) > 0  ← always positive
        """
        chosen = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        rejected = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

        output = stable_loss(chosen, rejected)
        output.loss.backward()

        # Gradient w.r.t. chosen should be negative (loss decreases when chosen goes up)
        assert (chosen.grad < 0).all(), (
            f"Gradient w.r.t. chosen_scores should be negative. Got {chosen.grad}"
        )
        # Gradient w.r.t. rejected should be positive (loss increases when rejected goes up)
        assert (rejected.grad > 0).all(), (
            f"Gradient w.r.t. rejected_scores should be positive. Got {rejected.grad}"
        )

    def test_diagnostic_metrics_have_no_gradient(
        self, stable_loss: BradleyTerryLoss
    ) -> None:
        """
        preference_accuracy and mean_margin are computed inside torch.no_grad().
        Calling .backward() through them should raise. This confirms they are
        monitoring-only values and not accidentally participating in training.
        """
        chosen = torch.randn(8, requires_grad=True)
        rejected = torch.randn(8, requires_grad=True)
        output = stable_loss(chosen, rejected)

        # These tensors should not require grad — they were computed inside no_grad
        assert not output.preference_accuracy.requires_grad
        assert not output.mean_margin.requires_grad


# =============================================================================
# 4. Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """
    Compare naive vs stable implementations. The key property:
    stable is always well-behaved; naive fails for large negative score diffs.
    """

    def test_stable_and_naive_agree_for_moderate_inputs(
        self,
        stable_loss: BradleyTerryLoss,
        naive_loss: BradleyTerryLoss,
    ) -> None:
        """
        For moderate inputs (score_diff in [-5, 5]), both implementations
        are numerically equivalent to 4 decimal places.
        """
        torch.manual_seed(42)
        # Clamp to moderate range to avoid the known naive failure zone
        chosen = torch.randn(64).clamp(-2.5, 2.5)
        rejected = torch.randn(64).clamp(-2.5, 2.5)

        stable_output = stable_loss(chosen, rejected)
        naive_output = naive_loss(chosen, rejected)

        torch.testing.assert_close(
            stable_output.loss,
            naive_output.loss,
            rtol=1e-4,
            atol=1e-4,
            msg="Stable and naive losses must agree for moderate inputs",
        )

    def test_stable_handles_very_negative_score_diff(
        self, stable_loss: BradleyTerryLoss
    ) -> None:
        """
        score_diff = -100 → σ(-100) ≈ 3.7e-44 (float32 underflows to 0).
        Naive form: log(0) = -inf. Stable form: softplus(100) ≈ 100. Finite.

        This simulates early training where the model strongly prefers the
        wrong response. The stable form must survive this gracefully.
        """
        chosen = torch.full((16,), -50.0)   # very low score for chosen
        rejected = torch.full((16,), 50.0)  # very high score for rejected
        # score_diff = chosen - rejected = -100

        output = stable_loss(chosen, rejected)

        assert not torch.isnan(output.loss), (
            "Stable loss must not produce NaN for large negative score differences"
        )
        assert not torch.isinf(output.loss), (
            "Stable loss must not produce Inf for large negative score differences"
        )
        assert output.loss.item() > 0.0  # Still a valid positive loss

    def test_naive_fails_for_very_negative_score_diff(
        self, naive_loss: BradleyTerryLoss
    ) -> None:
        """
        Document the known failure mode of the naive implementation.
        This test is deliberately checking that the naive form breaks here.
        If this test fails (naive unexpectedly passes), it means the hardware
        or PyTorch version is handling the underflow differently — worth investigating.
        """
        chosen = torch.full((16,), -50.0)
        rejected = torch.full((16,), 50.0)

        output = naive_loss(chosen, rejected)

        # We EXPECT numerical failure here — this is the known limitation
        has_numerical_issue = torch.isnan(output.loss) or torch.isinf(output.loss)
        assert has_numerical_issue, (
            "Naive loss should fail (NaN or Inf) for very negative score differences. "
            "If it passes, the float32 underflow may not be triggering — check inputs."
        )

    def test_stable_handles_very_positive_score_diff(
        self, stable_loss: BradleyTerryLoss
    ) -> None:
        """
        score_diff = +100 → loss should be near zero, not NaN.
        Both implementations handle this case, but let's be explicit.
        """
        chosen = torch.full((16,), 50.0)
        rejected = torch.full((16,), -50.0)

        output = stable_loss(chosen, rejected)

        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)
        assert output.loss.item() < 1e-3


# =============================================================================
# 5. Input Validation Tests
# =============================================================================

class TestInputValidation:
    """
    Verify defensive error handling. These tests make the error messages
    we wrote worth writing — they confirm that bad inputs fail loudly and clearly.
    """

    def test_mismatched_batch_sizes_raise(self, stable_loss: BradleyTerryLoss) -> None:
        """Different batch sizes must raise with a clear error."""
        chosen = torch.randn(8)
        rejected = torch.randn(16)  # wrong size

        with pytest.raises(AssertionError, match="same shape"):
            stable_loss(chosen, rejected)

    def test_2d_chosen_raises(self, stable_loss: BradleyTerryLoss) -> None:
        """
        2D input (batch_size, 1) is a very common mistake when the model
        outputs shape (batch_size, 1) and you forget to squeeze().
        Must raise with a helpful error message.
        """
        chosen = torch.randn(8, 1)   # forgot squeeze()
        rejected = torch.randn(8, 1)

        with pytest.raises(AssertionError, match="1D"):
            stable_loss(chosen, rejected)

    def test_2d_rejected_raises(self, stable_loss: BradleyTerryLoss) -> None:
        """Same check for rejected scores."""
        chosen = torch.randn(8)
        rejected = torch.randn(8, 1)  # forgot squeeze()

        with pytest.raises(AssertionError, match="1D"):
            stable_loss(chosen, rejected)

    def test_single_pair_works(self, stable_loss: BradleyTerryLoss) -> None:
        """Batch size of 1 must work (useful for debugging)."""
        chosen = torch.tensor([2.0])
        rejected = torch.tensor([1.0])
        output = stable_loss(chosen, rejected)
        assert output.loss.item() > 0.0

    def test_large_batch_works(self, stable_loss: BradleyTerryLoss) -> None:
        """Loss should work for large batches (e.g., during eval over full dataset)."""
        chosen = torch.randn(512)
        rejected = torch.randn(512)
        output = stable_loss(chosen, rejected)
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)