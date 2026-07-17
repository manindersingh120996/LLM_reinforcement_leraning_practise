"""
Tests for RewardModelEvaluator and EvaluationReport.

Strategy: same pattern as test_trainer — tiny model, MinimalTokenizer,
from_list dataset. No network access, runs in under a second.

Key invariants tested:
    - Report fields have correct types and valid ranges
    - Weighted averaging is correct (n_examples matches loader total)
    - Cohen's d is positive when chosen scores > rejected scores on average
    - summary() and to_dict() produce correct output shapes
    - Model is restored to train() mode after evaluate()
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader

from conftest import TINY_GPT2_CONFIG, MinimalTokenizer
from reward_model.dataset import PreferenceCollator, PreferenceDataset
from reward_model.evaluator import EvaluationReport, RewardModelEvaluator, _cohens_d
from reward_model.loss import BradleyTerryLoss
from reward_model.model import RewardModel


# =============================================================================
# Fixtures
# =============================================================================

EXAMPLES = [
    {"chosen":   "Human: What is 2+2?\nAssistant: 4.",
     "rejected": "Human: What is 2+2?\nAssistant: 5."},
    {"chosen":   "Human: Hi\nAssistant: Hello, how can I help?",
     "rejected": "Human: Hi\nAssistant: ..."},
    {"chosen":   "Human: Color of the sky?\nAssistant: Blue.",
     "rejected": "Human: Color of the sky?\nAssistant: Purple."},
    {"chosen":   "Human: Capital of France?\nAssistant: Paris.",
     "rejected": "Human: Capital of France?\nAssistant: Berlin."},
]


@pytest.fixture(scope="module")
def tokenizer():
    return MinimalTokenizer()


@pytest.fixture(scope="module")
def val_loader(tokenizer):
    ds = PreferenceDataset.from_list(tokenizer, EXAMPLES, max_length=64)
    collator = PreferenceCollator(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(ds, batch_size=2, collate_fn=collator, shuffle=False)


@pytest.fixture(scope="module")
def evaluator():
    model   = RewardModel.from_config(TINY_GPT2_CONFIG, num_unfrozen_layers=2, dropout=0.0)
    loss_fn = BradleyTerryLoss(use_stable=True)
    return RewardModelEvaluator(model, loss_fn, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def report(evaluator, val_loader):
    """Run evaluation once, share the result across all tests in this module."""
    return evaluator.evaluate(val_loader)


# =============================================================================
# 1. Report field types and validity
# =============================================================================

class TestReportFields:

    def test_returns_evaluation_report(self, report) -> None:
        assert isinstance(report, EvaluationReport)

    def test_preference_accuracy_in_range(self, report) -> None:
        assert 0.0 <= report.preference_accuracy <= 1.0

    def test_mean_loss_is_positive_and_finite(self, report) -> None:
        assert report.mean_loss > 0.0
        assert math.isfinite(report.mean_loss)

    def test_score_stds_are_non_negative(self, report) -> None:
        """Standard deviation can't be negative."""
        assert report.chosen_std >= 0.0
        assert report.rejected_std >= 0.0

    def test_n_examples_is_positive(self, report) -> None:
        assert report.n_examples > 0

    def test_n_correct_does_not_exceed_n_examples(self, report) -> None:
        assert 0 <= report.n_correct <= report.n_examples

    def test_n_correct_consistent_with_accuracy(self, report) -> None:
        """n_correct / n_examples must equal preference_accuracy."""
        expected = report.n_correct / report.n_examples
        assert abs(report.preference_accuracy - expected) < 1e-6


# =============================================================================
# 2. Coverage — all examples are evaluated
# =============================================================================

class TestCoverage:

    def test_n_examples_equals_loader_total(self, report, val_loader) -> None:
        """
        The evaluator must score every example in the val_loader.
        If this fails, some batches are being skipped or double-counted.
        """
        total = sum(batch.chosen_input_ids.shape[0] for batch in val_loader)
        assert report.n_examples == total, (
            f"Evaluated {report.n_examples} examples but loader has {total}."
        )


# =============================================================================
# 3. Model mode
# =============================================================================

class TestModelMode:

    def test_model_in_eval_mode_during_evaluate(self, evaluator, val_loader) -> None:
        """Model must be in eval() mode (dropout off) during evaluation."""
        training_flags = []
        orig = evaluator.loss_fn.forward

        def patched(chosen, rejected):
            training_flags.append(evaluator.model.training)
            return orig(chosen, rejected)

        evaluator.loss_fn.forward = patched
        try:
            evaluator.evaluate(val_loader)
        finally:
            evaluator.loss_fn.forward = orig

        assert len(training_flags) > 0
        assert not any(training_flags), (
            "model.training was True during evaluate(). "
            "model.eval() must be called before the eval loop."
        )

    def test_model_restored_to_train_mode_after_evaluate(
        self, evaluator, val_loader
    ) -> None:
        """After evaluate() returns, model must be back in train() mode."""
        evaluator.evaluate(val_loader)
        assert evaluator.model.training, (
            "model.training is False after evaluate() returned."
        )


# =============================================================================
# 4. Score separation — Cohen's d
# =============================================================================

class TestScoreSeparation:

    def test_score_separation_is_finite(self, report) -> None:
        assert math.isfinite(report.score_separation)

    def test_cohens_d_positive_when_chosen_higher(self) -> None:
        """
        When chosen distribution has a higher mean than rejected,
        Cohen's d should be positive.
        """
        d = _cohens_d(
            mean_a=2.0, std_a=1.0,  # chosen
            mean_b=0.0, std_b=1.0,  # rejected
            n_a=100, n_b=100,
        )
        assert d > 0.0

    def test_cohens_d_zero_when_distributions_identical(self) -> None:
        """If chosen and rejected have identical distributions, d = 0."""
        d = _cohens_d(
            mean_a=1.0, std_a=0.5,
            mean_b=1.0, std_b=0.5,
            n_a=50, n_b=50,
        )
        assert abs(d) < 1e-9

    def test_cohens_d_handles_zero_std(self) -> None:
        """
        If all scores are identical (std=0), pooled_std=0.
        Should return 0.0 gracefully, not raise ZeroDivisionError.
        """
        d = _cohens_d(
            mean_a=1.0, std_a=0.0,
            mean_b=1.0, std_b=0.0,
            n_a=10, n_b=10,
        )
        assert d == 0.0

    def test_cohens_d_known_value(self) -> None:
        """
        Verify against a manually computed value.
        mean_a=2, mean_b=0, std_a=std_b=1, equal n.
        Pooled std = 1. d = (2-0)/1 = 2.0.
        """
        d = _cohens_d(
            mean_a=2.0, std_a=1.0,
            mean_b=0.0, std_b=1.0,
            n_a=1000, n_b=1000,
        )
        # With large n, the (n-1) correction is negligible: d ≈ 2.0
        assert abs(d - 2.0) < 0.01


# =============================================================================
# 5. Report output methods
# =============================================================================

class TestReportOutput:

    def test_summary_returns_string(self, report) -> None:
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_accuracy(self, report) -> None:
        """The summary must show the preference accuracy."""
        s = report.summary()
        # Check the value is present (formatted to at least 2 decimal places)
        assert f"{report.preference_accuracy:.4f}" in s

    def test_to_dict_returns_dict(self, report) -> None:
        d = report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_expected_keys(self, report) -> None:
        d = report.to_dict()
        required_keys = {
            "eval/preference_accuracy",
            "eval/mean_loss",
            "eval/mean_margin",
            "eval/chosen_mean",
            "eval/chosen_std",
            "eval/rejected_mean",
            "eval/rejected_std",
            "eval/score_separation",
            "eval/n_examples",
        }
        assert required_keys.issubset(set(d.keys())), (
            f"Missing keys: {required_keys - set(d.keys())}"
        )

    def test_to_dict_values_are_numeric(self, report) -> None:
        d = report.to_dict()
        for key, val in d.items():
            assert isinstance(val, (int, float)), (
                f"Expected numeric value for {key}, got {type(val)}"
            )