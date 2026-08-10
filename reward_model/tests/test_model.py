"""
Unit tests for RewardModel.
All tests use RewardModel.from_config_test() — no network, no downloads.
"""
import pytest
import torch

from reward_model.model import RewardModel, RewardModelConfig, ParameterStats
from conftest import TINY_REWARD_CONFIG

NUM_UNFROZEN = 2


@pytest.fixture(scope="module")
def model():
    m = RewardModel.from_config_test(num_unfrozen_layers=NUM_UNFROZEN)
    m.eval()
    return m


def make_batch(batch_size=4, seq_len=32, seed=42):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, TINY_REWARD_CONFIG.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, attention_mask


class TestOutputContract:
    def test_output_shape(self, model):
        ids, mask = make_batch(batch_size=4, seq_len=32)
        with torch.no_grad():
            scores = model(ids, mask)
        assert scores.shape == (4,)

    def test_output_is_1d(self, model):
        ids, mask = make_batch()
        with torch.no_grad():
            scores = model(ids, mask)
        assert scores.dim() == 1

    def test_output_dtype_float32(self, model):
        ids, mask = make_batch()
        with torch.no_grad():
            scores = model(ids, mask)
        assert scores.dtype == torch.float32

    def test_output_is_finite(self, model):
        ids, mask = make_batch(batch_size=16)
        with torch.no_grad():
            scores = model(ids, mask)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_batch_size_one(self, model):
        ids, mask = make_batch(batch_size=1)
        with torch.no_grad():
            scores = model(ids, mask)
        assert scores.shape == (1,)

    def test_various_batch_sizes(self, model):
        for bs in [1, 4, 8, 16]:
            ids, mask = make_batch(batch_size=bs)
            with torch.no_grad():
                scores = model(ids, mask)
            assert scores.shape == (bs,)


class TestDeterminism:
    def test_same_input_same_output(self, model):
        ids, mask = make_batch()
        with torch.no_grad():
            s1 = model(ids, mask)
            s2 = model(ids, mask)
        torch.testing.assert_close(s1, s2)

    def test_different_inputs_different_outputs(self, model):
        ids_a, mask_a = make_batch(seed=0)
        ids_b, mask_b = make_batch(seed=1)
        with torch.no_grad():
            sa = model(ids_a, mask_a)
            sb = model(ids_b, mask_b)
        assert not torch.allclose(sa, sb)


class TestPaddingCorrectness:
    def test_right_padding_does_not_change_score(self, model):
        torch.manual_seed(99)
        seq_len = 10
        real_tokens = torch.randint(1, TINY_REWARD_CONFIG.vocab_size, (1, seq_len))
        mask_no_pad = torch.ones(1, seq_len, dtype=torch.long)

        n_pad = 5
        padded_ids = torch.cat([real_tokens, torch.zeros(1, n_pad, dtype=torch.long)], dim=1)
        padded_mask = torch.cat([mask_no_pad, torch.zeros(1, n_pad, dtype=torch.long)], dim=1)

        with torch.no_grad():
            s_no_pad = model(real_tokens, mask_no_pad)
            s_padded = model(padded_ids, padded_mask)

        torch.testing.assert_close(s_no_pad, s_padded, rtol=1e-4, atol=1e-4)

    def test_last_token_index_computation(self, model):
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        expected = torch.tensor([9])
        assert torch.equal(mask.sum(dim=1) - 1, expected)

    def test_mixed_padding_lengths_no_nan(self, model):
        seq_len = 12
        ids = torch.randint(1, TINY_REWARD_CONFIG.vocab_size, (3, seq_len))
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long)
        with torch.no_grad():
            scores = model(ids, mask)
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()


class TestGradientFlow:
    def test_loss_backward_works(self, model):
        model.train()
        ids, mask = make_batch()
        scores = model(ids, mask)
        scores.mean().backward()
        model.eval()

    def test_gradient_reaches_scalar_head(self, model):
        model.train()
        model.zero_grad()
        ids, mask = make_batch()
        model(ids, mask).mean().backward()
        head_grad = model.scalar_head[1].weight.grad
        assert head_grad is not None
        assert not torch.isnan(head_grad).any()
        model.eval()

    def test_gradient_reaches_unfrozen_block(self, model):
        model.train()
        model.zero_grad()
        ids, mask = make_batch()
        model(ids, mask).mean().backward()
        for name, param in model.backbone.h[-1].named_parameters():
            assert param.grad is not None, f"Missing grad: {name}"
        model.eval()


class TestFreezingStrategy:
    def test_embedding_frozen(self, model):
        for name, p in model.backbone.named_parameters():
            if "wte" in name or "wpe" in name:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_lower_blocks_frozen(self, model):
        total = len(model.backbone.h)
        for i in range(total - NUM_UNFROZEN):
            for name, p in model.backbone.h[i].named_parameters():
                assert not p.requires_grad, f"Block {i}.{name} should be frozen"

    def test_upper_blocks_unfrozen(self, model):
        total = len(model.backbone.h)
        for i in range(total - NUM_UNFROZEN, total):
            for name, p in model.backbone.h[i].named_parameters():
                assert p.requires_grad, f"Block {i}.{name} should be trainable"

    def test_scalar_head_always_unfrozen(self, model):
        for name, p in model.scalar_head.named_parameters():
            assert p.requires_grad, f"scalar_head.{name} should be trainable"

    def test_frozen_get_no_grad_after_backward(self, model):
        model.train()
        model.zero_grad()
        ids, mask = make_batch()
        model(ids, mask).mean().backward()
        assert model.backbone.wte.weight.grad is None
        assert all(p.grad is None for p in model.backbone.h[0].parameters())
        model.eval()


class TestParameterStats:
    def test_returns_parameter_stats(self, model):
        assert isinstance(model.parameter_stats(), ParameterStats)

    def test_trainable_plus_frozen_equals_total(self, model):
        s = model.parameter_stats()
        assert s.trainable + s.frozen == s.total

    def test_trainable_pct_reasonable(self, model):
        s = model.parameter_stats()
        assert 5.0 < s.trainable_pct < 60.0

    def test_all_unfrozen_when_minus_one(self):
        m = RewardModel.from_config_test(num_unfrozen_layers=-1)
        s = m.parameter_stats()
        assert s.trainable == s.total

    def test_head_only_when_zero_unfrozen(self):
        m = RewardModel.from_config_test(num_unfrozen_layers=0)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        assert all("scalar_head" in n for n in trainable)


class TestConfig:
    def test_config_stored_correctly(self):
        config = RewardModelConfig(n_embd=64, n_layer=4, n_head=2, dropout=0.05)
        assert config.n_embd == 64
        assert config.dropout == 0.05
        assert config.model_type == "reward_model"

    def test_model_built_from_config(self):
        config = RewardModelConfig(n_embd=64, n_layer=4, n_head=2, vocab_size=128)
        model = RewardModel(config)
        assert model.hidden_dim == 64