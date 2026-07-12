"""
Unit tests for RewardModel.

Tests are grouped by what they verify:
    1. Output contract     — shapes and types
    2. Determinism         — same input, same output
    3. Padding correctness — last real token pooling works
    4. Gradient flow       — backward pass reaches trainable parameters
    5. Freezing strategy   — frozen layers have no gradients, unfrozen do
    6. Parameter stats     — utility method is accurate

Note on model loading speed
---------------------------
GPT-2 weights are downloaded from HuggingFace on first run (~500MB).
Subsequent runs use the local cache. Tests marked with the `model` fixture
share a single loaded model instance to avoid repeated loading overhead.
"""

import pytest
import torch
from omegaconf import OmegaConf
from transformers import GPT2Config

from reward_model.model import RewardModel, ParameterStats

# A tiny GPT-2 config used across all tests.
# Using a real (small) architecture rather than mocking so tests exercise
# actual model behaviour, but without downloading 500MB of weights.
# n_layer=4, n_head=2, n_embd=64 → model with ~100K params, loads in <1s.
TINY_GPT2_CONFIG = GPT2Config(
    n_layer=4,
    n_head=2,
    n_embd=64,
    n_positions=128,
    vocab_size=1000,
)
NUM_UNFROZEN = 2  # freeze bottom 2 layers, train top 2


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def cfg():
    """Config used by tests that construct RewardModel via from_config."""
    return OmegaConf.create({
        "model": {
            "backbone": "gpt2",
            "num_unfrozen_layers": NUM_UNFROZEN,
            "dropout": 0.0,
        }
    })


@pytest.fixture(scope="module")
def model():
    """
    Single RewardModel instance reused across all tests.

    Uses from_config() with a tiny architecture so tests run in <1s
    without any network access. This is the right pattern for ML tests:
    verify behaviour with a small equivalent model, not the production model.
    """
    m = RewardModel.from_config(TINY_GPT2_CONFIG, num_unfrozen_layers=NUM_UNFROZEN, dropout=0.0)
    m.eval()
    return m


def make_batch(
    batch_size: int = 4,
    seq_len: int = 32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a random batch of (input_ids, attention_mask).
    Vocab size matches TINY_GPT2_CONFIG (1000) to avoid embedding index errors.
    """
    torch.manual_seed(seed)
    input_ids = torch.randint(0, TINY_GPT2_CONFIG.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, attention_mask


# =============================================================================
# 1. Output Contract Tests
# =============================================================================

class TestOutputContract:
    """Verify that the model returns the right shapes and types."""

    def test_output_shape_matches_batch_size(self, model: RewardModel) -> None:
        """
        For a batch of N sequences, model must return N scalar scores.
        BradleyTerryLoss expects shape (batch_size,) — not (batch_size, 1).
        """
        input_ids, attention_mask = make_batch(batch_size=4, seq_len=32)
        with torch.no_grad():
            scores = model(input_ids, attention_mask)
        assert scores.shape == (4,), f"Expected (4,), got {scores.shape}"

    def test_output_is_1d(self, model: RewardModel) -> None:
        """Output must be 1-dimensional. A common bug: forgetting to squeeze()."""
        input_ids, attention_mask = make_batch(batch_size=8, seq_len=16)
        with torch.no_grad():
            scores = model(input_ids, attention_mask)
        assert scores.dim() == 1, (
            f"Scores must be 1D (batch_size,). Got {scores.dim()}D. "
            f"Did you forget .squeeze(-1) in the scalar head?"
        )

    def test_output_dtype_is_float32(self, model: RewardModel) -> None:
        """Scores must be float32 for compatibility with BradleyTerryLoss."""
        input_ids, attention_mask = make_batch(batch_size=4, seq_len=32)
        with torch.no_grad():
            scores = model(input_ids, attention_mask)
        assert scores.dtype == torch.float32

    def test_output_is_finite(self, model: RewardModel) -> None:
        """No NaN or Inf in output, even on random input."""
        input_ids, attention_mask = make_batch(batch_size=16, seq_len=64)
        with torch.no_grad():
            scores = model(input_ids, attention_mask)
        assert not torch.isnan(scores).any(), "Output contains NaN"
        assert not torch.isinf(scores).any(), "Output contains Inf"

    def test_batch_size_one_works(self, model: RewardModel) -> None:
        """Single-example batch must work (common in inference/debugging)."""
        input_ids, attention_mask = make_batch(batch_size=1, seq_len=32)
        with torch.no_grad():
            scores = model(input_ids, attention_mask)
        assert scores.shape == (1,)

    def test_different_batch_sizes_work(self, model: RewardModel) -> None:
        """Model must handle any batch size without recompilation."""
        for batch_size in [1, 4, 8, 16]:
            input_ids, attention_mask = make_batch(batch_size=batch_size, seq_len=32)
            with torch.no_grad():
                scores = model(input_ids, attention_mask)
            assert scores.shape == (batch_size,)


# =============================================================================
# 2. Determinism Tests
# =============================================================================

class TestDeterminism:
    """
    Verify reproducibility. A model in eval() mode with dropout=0.0
    must produce identical output for identical input.
    """

    def test_same_input_gives_same_output(self, model: RewardModel) -> None:
        """Identical inputs in eval mode must produce identical scores."""
        input_ids, attention_mask = make_batch(batch_size=4, seq_len=32)

        with torch.no_grad():
            scores_first = model(input_ids, attention_mask)
            scores_second = model(input_ids, attention_mask)

        torch.testing.assert_close(
            scores_first, scores_second,
            msg="Same input in eval mode must give same output"
        )

    def test_different_inputs_give_different_outputs(self, model: RewardModel) -> None:
        """Different token sequences should (almost certainly) produce different scores."""
        input_ids_a, mask_a = make_batch(batch_size=4, seq_len=32, seed=0)
        input_ids_b, mask_b = make_batch(batch_size=4, seq_len=32, seed=1)

        with torch.no_grad():
            scores_a = model(input_ids_a, mask_a)
            scores_b = model(input_ids_b, mask_b)

        assert not torch.allclose(scores_a, scores_b), (
            "Different token sequences must produce different scores. "
            "If they're identical, the model weights may not be loaded correctly."
        )


# =============================================================================
# 3. Padding Correctness Tests
# =============================================================================

class TestPaddingCorrectness:
    """
    Verify that we're correctly using attention_mask to find the last REAL token.
    This is the most important correctness test for the model architecture.
    A model that uses hidden_states[:, -1, :] (last position, ignoring padding)
    will score padding tokens instead of response content.
    """

    def test_right_padding_does_not_change_score(self, model: RewardModel) -> None:
        """
        Adding right-padding to a sequence must not change the reward score.

        Setup:
            Sequence A: [tok_1, tok_2, ..., tok_10]              — no padding
            Sequence B: [tok_1, tok_2, ..., tok_10, PAD, PAD, PAD] — padded

        Both must produce the same score, because the real content is identical.
        """
        torch.manual_seed(99)
        seq_len = 10
        real_tokens = torch.randint(1, TINY_GPT2_CONFIG.vocab_size, (1, seq_len))  # avoid token 0 (PAD)
        mask_no_pad = torch.ones(1, seq_len, dtype=torch.long)

        # Create padded version: 5 PAD tokens (token id 0) at the end
        n_pad = 5
        pad_tokens = torch.zeros(1, n_pad, dtype=torch.long)
        padded_input_ids = torch.cat([real_tokens, pad_tokens], dim=1)  # (1, 15)
        padded_mask = torch.cat([
            mask_no_pad,
            torch.zeros(1, n_pad, dtype=torch.long)
        ], dim=1)  # (1, 15)

        with torch.no_grad():
            score_no_pad = model(real_tokens, mask_no_pad)
            score_padded = model(padded_input_ids, padded_mask)

        torch.testing.assert_close(
            score_no_pad,
            score_padded,
            rtol=1e-4,
            atol=1e-4,
            msg=(
                "Padding tokens must not affect the reward score. "
                "If this fails, the model is using hidden_states[:, -1, :] "
                "instead of indexing with the attention_mask."
            )
        )

    def test_last_token_index_computation(self, model: RewardModel) -> None:
        """
        Directly verify that last_real_token_idx = attention_mask.sum(dim=1) - 1
        gives the correct index for sequences with and without padding.
        """
        # Sequence of 10 real tokens + 5 padding
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        expected_idx = torch.tensor([9])  # last 1 is at position 9

        computed_idx = attention_mask.sum(dim=1) - 1
        torch.testing.assert_close(computed_idx, expected_idx)

    def test_different_padding_lengths_in_same_batch(self, model: RewardModel) -> None:
        """
        A batch where different examples have different amounts of padding
        must still produce correct (finite, non-NaN) scores.
        This is the realistic scenario during training.
        """
        # Example 1: 8 real tokens, 4 padding
        # Example 2: 10 real tokens, 2 padding
        # Example 3: 12 real tokens, 0 padding
        seq_len = 12
        input_ids = torch.randint(1, TINY_GPT2_CONFIG.vocab_size, (3, seq_len))
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # 8 real tokens
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 10 real tokens
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 12 real tokens
        ], dtype=torch.long)

        with torch.no_grad():
            scores = model(input_ids, attention_mask)

        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()


# =============================================================================
# 4. Gradient Flow Tests
# =============================================================================

class TestGradientFlow:
    """
    Verify that backpropagation reaches trainable parameters.
    Without correct gradient flow, the optimizer updates nothing and
    training silently fails (loss may appear to move due to LR scheduler
    warmup, but weights don't change).
    """

    def test_loss_backward_works(self, model: RewardModel) -> None:
        """End-to-end: loss.backward() must not crash."""
        # Temporarily switch to train mode (dropout was 0.0 anyway, so same result)
        model.train()
        input_ids, attention_mask = make_batch(batch_size=4, seq_len=32)

        scores = model(input_ids, attention_mask)
        loss = scores.mean()   # dummy loss — just checks backward works
        loss.backward()        # must not raise

        model.eval()  # restore eval mode for other tests

    def test_gradient_reaches_scalar_head(self, model: RewardModel) -> None:
        """Gradient must flow into the scalar head weights (always trainable)."""
        model.train()
        model.zero_grad()

        input_ids, attention_mask = make_batch(batch_size=4, seq_len=32)
        scores = model(input_ids, attention_mask)
        scores.mean().backward()

        # Check the Linear layer inside the Sequential scalar head (index 1)
        head_weight_grad = model.scalar_head[1].weight.grad
        assert head_weight_grad is not None, "Gradient did not reach scalar_head"
        assert not torch.isnan(head_weight_grad).any(), "NaN gradient in scalar_head"

        model.eval()

    def test_gradient_reaches_unfrozen_backbone_layers(
        self, model: RewardModel
    ) -> None:
        """Gradient must reach the last (unfrozen) transformer block."""
        model.train()
        model.zero_grad()

        input_ids, attention_mask = make_batch(batch_size=4, seq_len=32)
        scores = model(input_ids, attention_mask)
        scores.mean().backward()

        # The very last transformer block (h[-1]) must have gradients
        last_block = model.backbone.h[-1]
        for name, param in last_block.named_parameters():
            assert param.grad is not None, (
                f"Last transformer block must have gradient. Missing on: {name}"
            )

        model.eval()


# =============================================================================
# 5. Freezing Strategy Tests
# =============================================================================

class TestFreezingStrategy:
    """
    Verify that the layer freezing works exactly as intended.
    Frozen layers must have param.requires_grad=False and no .grad after backward.
    Unfrozen layers must have param.requires_grad=True and a .grad after backward.
    """

    def test_embedding_layers_are_frozen(self, model: RewardModel) -> None:
        """Token embeddings (wte) and position embeddings (wpe) must be frozen."""
        for name, param in model.backbone.named_parameters():
            if name.startswith("wte") or name.startswith("wpe"):
                assert not param.requires_grad, (
                    f"Embedding layer {name} should be frozen but requires_grad=True"
                )

    def test_lower_transformer_blocks_are_frozen(self, model: RewardModel) -> None:
        """
        The first (total_layers - num_unfrozen_layers) blocks must be frozen.
        With num_unfrozen_layers=2 and 12 total blocks, blocks 0-9 must be frozen.
        """
        total = len(model.backbone.h)
        frozen_count = total - 2  # num_unfrozen_layers=2 from cfg fixture

        for i in range(frozen_count):
            block = model.backbone.h[i]
            for name, param in block.named_parameters():
                assert not param.requires_grad, (
                    f"Block {i} (should be frozen) has requires_grad=True on {name}"
                )

    def test_upper_transformer_blocks_are_unfrozen(self, model: RewardModel) -> None:
        """The last num_unfrozen_layers blocks must have requires_grad=True."""
        total = len(model.backbone.h)

        for i in range(total - 2, total):  # last 2 blocks
            block = model.backbone.h[i]
            for name, param in block.named_parameters():
                assert param.requires_grad, (
                    f"Block {i} (should be unfrozen) has requires_grad=False on {name}"
                )

    def test_final_layer_norm_is_unfrozen(self, model: RewardModel) -> None:
        """ln_f (final layer norm) must be trainable."""
        for name, param in model.backbone.ln_f.named_parameters():
            assert param.requires_grad, (
                f"Final layer norm {name} should be unfrozen but requires_grad=False"
            )

    def test_scalar_head_is_always_unfrozen(self, model: RewardModel) -> None:
        """Scalar head is new weights — always trainable regardless of config."""
        for name, param in model.scalar_head.named_parameters():
            assert param.requires_grad, (
                f"Scalar head {name} should always be trainable but requires_grad=False"
            )

    def test_frozen_layers_receive_no_gradients(self, model: RewardModel) -> None:
        """
        After a backward pass, frozen parameters must have .grad == None.
        This confirms that PyTorch is not computing or storing gradients for them,
        which saves memory and computation during training.
        """
        model.train()
        model.zero_grad()

        input_ids, attention_mask = make_batch(batch_size=4, seq_len=32)
        scores = model(input_ids, attention_mask)
        scores.mean().backward()

        # Embedding layers are frozen — must have no gradient
        assert model.backbone.wte.weight.grad is None, (
            "Frozen embedding layer should not accumulate gradients"
        )

        # First transformer block is frozen — must have no gradient
        for param in model.backbone.h[0].parameters():
            assert param.grad is None, (
                "First (frozen) transformer block must not have gradients"
            )

        model.eval()


# =============================================================================
# 6. Parameter Stats Tests
# =============================================================================

class TestParameterStats:

    def test_parameter_stats_returns_correct_type(self, model: RewardModel) -> None:
        stats = model.parameter_stats()
        assert isinstance(stats, ParameterStats)

    def test_trainable_plus_frozen_equals_total(self, model: RewardModel) -> None:
        """Basic accounting: trainable + frozen must sum to total."""
        stats = model.parameter_stats()
        assert stats.trainable + stats.frozen == stats.total

    def test_trainable_pct_is_reasonable(self, model: RewardModel) -> None:
        """
        With 2 out of 12 layers unfrozen + scalar head, expect roughly 15–30%
        of parameters to be trainable. If this is 0% or 100%, freezing is broken.
        """
        stats = model.parameter_stats()
        assert 5.0 < stats.trainable_pct < 60.0, (
            f"Expected 5-60% trainable params with 2 unfrozen layers. "
            f"Got {stats.trainable_pct}%. Check _apply_layer_freezing()."
        )

    def test_all_layers_unfrozen_gives_100_pct(self, cfg) -> None:
        """When num_unfrozen_layers=-1, all parameters must be trainable."""
        full_model = RewardModel.from_config(TINY_GPT2_CONFIG, num_unfrozen_layers=-1)
        stats = full_model.parameter_stats()
        assert stats.trainable == stats.total, (
            f"With num_unfrozen_layers=-1, all params should be trainable. "
            f"Got {stats.trainable_pct}% trainable."
        )

    def test_zero_unfrozen_layers_trains_only_head(self, cfg) -> None:
        """When num_unfrozen_layers=0, only the scalar head is trainable."""
        head_model = RewardModel.from_config(TINY_GPT2_CONFIG, num_unfrozen_layers=0)

        trainable_names = [
            name for name, p in head_model.named_parameters() if p.requires_grad
        ]
        assert all("scalar_head" in n for n in trainable_names), (
            f"With num_unfrozen_layers=0, only scalar_head should be trainable. "
            f"Got trainable: {trainable_names}"
        )