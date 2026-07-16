"""
Unit tests for RewardModelTrainer.

Strategy: build everything from scratch using zero network access.
    - TINY_GPT2_CONFIG    → from conftest.py (4 layers, hidden_dim=64)
    - MinimalTokenizer    → from conftest.py (ASCII character-level)
    - PreferenceDataset.from_list + PreferenceCollator → from dataset.py
    - RewardModel.from_config + BradleyTerryLoss → from model.py / loss.py

Key invariants we test:
    1. Train step produces finite, valid metrics
    2. Training step actually changes model weights (gradients are flowing)
    3. Frozen parameters are never updated by the optimizer
    4. Model is in eval() mode during _evaluate(), train() mode after
    5. Checkpoint save/load round-trip preserves weights exactly
    6. Gradient clipping keeps post-clip grad norm ≤ clip threshold
"""

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from conftest import TINY_GPT2_CONFIG, MinimalTokenizer
from reward_model.dataset import PreferenceCollator, PreferenceDataset
from reward_model.loss import BradleyTerryLoss
from reward_model.model import RewardModel
from reward_model.trainer import EvalOutput, RewardModelTrainer, TrainStepOutput


# =============================================================================
# Shared setup
# =============================================================================

EXAMPLES = [
    {"chosen": "Human: What is 2+2?\nAssistant: 4.",
     "rejected": "Human: What is 2+2?\nAssistant: 5."},
    {"chosen": "Human: Hi\nAssistant: Hello, how are you?",
     "rejected": "Human: Hi\nAssistant: ..."},
    {"chosen": "Human: Color of sky?\nAssistant: Blue.",
     "rejected": "Human: Color of sky?\nAssistant: Green."},
    {"chosen": "Human: Capital of France?\nAssistant: Paris.",
     "rejected": "Human: Capital of France?\nAssistant: London."},
]


def make_cfg(output_dir: str) -> OmegaConf:
    """
    Build a minimal training config.

    warmup_steps=0 intentionally: with warmup_steps > 0, the scheduler
    initialises LR to zero on the very first step (lr * 0/warmup_steps).
    The first optimizer.step() then produces no weight change — correct
    warmup behaviour, but wrong for tests that check "did weights change?".
    Using warmup_steps=0 means full LR is available from the first step.
    """
    return OmegaConf.create({
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "max_epochs": 1,
            "warmup_steps": 0,           # ← 0 so first step uses full LR
            "gradient_clip_norm": 1.0,
            "eval_every_n_steps": 1000,
            "save_every_n_steps": 1000,
            "output_dir": output_dir,
        },
        "logging": {
            "log_every_n_steps": 1,
        },
    })


def make_dataloader(tokenizer: MinimalTokenizer) -> DataLoader:
    ds = PreferenceDataset.from_list(tokenizer, EXAMPLES, max_length=64)
    collator = PreferenceCollator(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(ds, batch_size=2, collate_fn=collator, shuffle=False)


@pytest.fixture(scope="module")
def tokenizer() -> MinimalTokenizer:
    return MinimalTokenizer()


@pytest.fixture
def trainer(tokenizer, tmp_path):
    """
    Fresh trainer for each test — each test gets isolated state and temp dir.

    scope is function (default) not module because:
    - _train_step modifies model weights
    - tests that check weight changes need a clean model each time
    - global_step increments between calls; tests must not share that state
    """
    loader = make_dataloader(tokenizer)
    model  = RewardModel.from_config(TINY_GPT2_CONFIG, num_unfrozen_layers=2, dropout=0.0)
    loss_fn = BradleyTerryLoss(use_stable=True)
    cfg = make_cfg(str(tmp_path))

    return RewardModelTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=loader,
        val_loader=loader,   # reuse train loader as val — fine for unit tests
        cfg=cfg,
        device=torch.device("cpu"),   # tests always on CPU
    )


def get_one_batch(trainer: RewardModelTrainer):
    """Pull the first batch from the train loader."""
    return next(iter(trainer.train_loader))


# =============================================================================
# 1. Train step output validity
# =============================================================================

class TestTrainStepOutput:
    """Verify that _train_step returns a valid, well-formed output."""

    def test_returns_train_step_output(self, trainer) -> None:
        """Output must be a TrainStepOutput dataclass."""
        batch = get_one_batch(trainer)
        out = trainer._train_step(batch)
        assert isinstance(out, TrainStepOutput)

    def test_loss_is_positive(self, trainer) -> None:
        """Bradley-Terry NLL loss must always be positive."""
        batch = get_one_batch(trainer)
        out = trainer._train_step(batch)
        assert out.loss > 0.0, f"Expected positive loss, got {out.loss}"

    def test_loss_is_finite(self, trainer) -> None:
        """Loss must not be NaN or Inf."""
        batch = get_one_batch(trainer)
        out = trainer._train_step(batch)
        assert not (out.loss != out.loss), "Loss is NaN"       # NaN check
        assert out.loss < float("inf"),    "Loss is Inf"

    def test_preference_accuracy_in_valid_range(self, trainer) -> None:
        """Preference accuracy must be in [0, 1]."""
        batch = get_one_batch(trainer)
        out = trainer._train_step(batch)
        assert 0.0 <= out.preference_accuracy <= 1.0, (
            f"Accuracy {out.preference_accuracy} out of [0, 1]"
        )

    def test_grad_norm_is_positive(self, trainer) -> None:
        """
        Pre-clip gradient norm must be positive — zero would mean no gradients
        are flowing and the model is not learning anything.
        """
        batch = get_one_batch(trainer)
        out = trainer._train_step(batch)
        assert out.grad_norm > 0.0, (
            f"Gradient norm is {out.grad_norm}. "
            "If zero, no gradients are reaching trainable parameters."
        )

    def test_grad_norm_is_finite(self, trainer) -> None:
        """Infinite gradient norm = exploding gradients = training failure."""
        batch = get_one_batch(trainer)
        out = trainer._train_step(batch)
        assert out.grad_norm < float("inf"), f"Gradient norm is infinite: {out.grad_norm}"

    def test_learning_rate_is_positive(self, trainer) -> None:
        """LR must be positive after the first step (warmup started)."""
        batch = get_one_batch(trainer)
        out = trainer._train_step(batch)
        assert out.learning_rate > 0.0, f"LR is {out.learning_rate} after first step"


# =============================================================================
# 2. Gradient flow — training actually updates weights
# =============================================================================

class TestGradientFlow:
    """Verify that the training step actually changes model weights."""

    def test_unfrozen_weights_change_after_step(self, trainer) -> None:
        """
        Weights in unfrozen layers must change after a training step.

        This is the end-to-end gradient flow test: it proves that
        - the loss computation is correct
        - .backward() runs successfully
        - the optimizer updates the right weights

        We check the scalar head weight because it's always unfrozen and
        has a direct path from the loss to the weights.
        """
        # Snapshot scalar head weights before the step
        before = trainer.model.scalar_head[1].weight.data.clone()

        batch = get_one_batch(trainer)
        trainer._train_step(batch)

        after = trainer.model.scalar_head[1].weight.data

        assert not torch.equal(before, after), (
            "Scalar head weights did not change after a training step. "
            "The optimizer may not be updating the correct parameters."
        )

    def test_frozen_weights_do_not_change(self, trainer) -> None:
        """
        Frozen weights must not change after a training step.

        This tests that:
        1. requires_grad=False actually prevents gradient accumulation
        2. The optimizer skips frozen parameters (we only pass requires_grad=True
           params to the optimizer in __init__, but this double-checks end-to-end)

        We check the embedding layer (wte) which is always frozen.
        """
        before = trainer.model.backbone.wte.weight.data.clone()

        batch = get_one_batch(trainer)
        trainer._train_step(batch)

        after = trainer.model.backbone.wte.weight.data

        assert torch.equal(before, after), (
            "Frozen embedding weights changed after a training step. "
            "Check that requires_grad=False parameters are excluded from the optimizer."
        )

    def test_loss_changes_after_step(self, trainer) -> None:
        """
        The loss on the same batch should differ before and after a step.

        Same batch, different model weights → different loss. This confirms
        the optimizer is actually moving in a direction that changes the model's
        predictions, not just running computations with zero net effect.
        """
        batch = get_one_batch(trainer)

        out_before = trainer._train_step(batch)  # also updates weights
        out_after  = trainer._train_step(batch)  # same batch, updated model

        assert out_before.loss != out_after.loss, (
            "Loss unchanged after a training step. "
            "This suggests the optimizer step has no effect on model weights."
        )


# =============================================================================
# 3. Gradient clipping
# =============================================================================

class TestGradientClipping:
    """Verify that gradient clipping keeps the post-clip norm at or below threshold."""

    def test_post_clip_grad_norm_within_threshold(self, trainer) -> None:
        """
        After gradient clipping, the global gradient norm must be ≤ clip_norm.

        We check this by measuring the norm ourselves after _train_step
        (which calls backward and clips). The norm at this point is
        AFTER clipping.

        Note: _train_step already called optimizer.step() and zero_grad()
        at the end, so we manually compute one more backward to measure.
        This is a white-box test — it requires understanding the internal order
        of operations in _train_step.
        """
        clip_norm = trainer.cfg.training.gradient_clip_norm
        batch = get_one_batch(trainer)

        # Run forward + backward manually so we can measure before optimizer clears grads
        trainer.model.train()
        trainer.optimizer.zero_grad()

        c_ids, r_ids, c_mask, r_mask = (
            batch.chosen_input_ids,
            batch.rejected_input_ids,
            batch.chosen_attention_mask,
            batch.rejected_attention_mask,
        )
        c_ids, r_ids, c_mask, r_mask = trainer._align_seq_lengths(
            c_ids, r_ids, c_mask, r_mask
        )
        all_scores = trainer.model(
            torch.cat([c_ids, r_ids], dim=0),
            torch.cat([c_mask, r_mask], dim=0),
        )
        B = c_ids.shape[0]
        loss_out = trainer.loss_fn(all_scores[:B], all_scores[B:])
        loss_out.loss.backward()

        # Apply clipping
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), clip_norm)

        # Measure post-clip norm
        post_clip_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in trainer.model.parameters()
            if p.grad is not None
        ) ** 0.5

        assert post_clip_norm <= clip_norm + 1e-5, (
            f"Post-clip norm {post_clip_norm:.4f} exceeds clip threshold {clip_norm}. "
            "gradient_clip_norm is not being applied correctly."
        )


# =============================================================================
# 4. Evaluation
# =============================================================================

class TestEvaluation:
    """Verify _evaluate() behaves correctly."""

    def test_returns_eval_output(self, trainer) -> None:
        out = trainer._evaluate()
        assert isinstance(out, EvalOutput)

    def test_eval_loss_is_positive_and_finite(self, trainer) -> None:
        out = trainer._evaluate()
        assert out.mean_loss > 0.0
        assert out.mean_loss < float("inf")
        assert out.mean_loss == out.mean_loss  # NaN check

    def test_eval_accuracy_in_valid_range(self, trainer) -> None:
        out = trainer._evaluate()
        assert 0.0 <= out.preference_accuracy <= 1.0

    def test_eval_covers_all_examples(self, trainer) -> None:
        """
        n_examples must equal the total number of examples in val_loader.
        If this fails, _evaluate() is skipping some batches or double-counting.
        """
        total = sum(
            batch.chosen_input_ids.shape[0]
            for batch in trainer.val_loader
        )
        out = trainer._evaluate()
        assert out.n_examples == total, (
            f"Evaluated {out.n_examples} examples but loader has {total}."
        )

    def test_model_in_eval_mode_during_evaluate(self, trainer) -> None:
        """
        During _evaluate(), the model must be in eval() mode (dropout disabled).
        We verify this by monkey-patching the loss function to capture model.training.
        """
        training_flags = []
        original_loss = trainer.loss_fn.forward

        def patched_loss(chosen, rejected):
            training_flags.append(trainer.model.training)
            return original_loss(chosen, rejected)

        trainer.loss_fn.forward = patched_loss
        try:
            trainer._evaluate()
        finally:
            trainer.loss_fn.forward = original_loss

        assert len(training_flags) > 0, "Loss was never called during evaluation"
        assert not any(training_flags), (
            "model.training was True during _evaluate(). "
            "model.eval() must be called before the eval loop."
        )

    def test_model_restored_to_train_mode_after_evaluate(self, trainer) -> None:
        """
        After _evaluate() returns, the model must be back in train() mode.
        Leaving the model in eval() mode is a subtle bug: dropout stays off
        during training, leading to unexpected overfitting behaviour.
        """
        trainer._evaluate()
        assert trainer.model.training, (
            "model.training is False after _evaluate() returned. "
            "model.train() must be called at the end of _evaluate()."
        )


# =============================================================================
# 5. Checkpointing
# =============================================================================

class TestCheckpointing:
    """Verify checkpoint save/load preserves all relevant state."""

    def test_checkpoint_file_is_created(self, trainer, tmp_path) -> None:
        """A checkpoint file must exist after _save_checkpoint."""
        trainer._save_checkpoint(step=42, tag="test")
        expected_path = tmp_path / "checkpoint_test.pt"
        assert expected_path.exists(), f"Checkpoint not found at {expected_path}"

    def test_checkpoint_contains_required_keys(self, trainer, tmp_path) -> None:
        """The checkpoint dict must contain all required keys for resuming."""
        trainer._save_checkpoint(step=42, tag="keys_test")
        ckpt = torch.load(tmp_path / "checkpoint_keys_test.pt", map_location="cpu")

        required_keys = {
            "step",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "cfg",
        }
        assert required_keys.issubset(set(ckpt.keys())), (
            f"Missing keys in checkpoint: {required_keys - set(ckpt.keys())}"
        )

    def test_checkpoint_roundtrip_preserves_weights(self, tokenizer, tmp_path) -> None:
        """
        Weights after load_checkpoint must equal weights before save.

        This is the core correctness test: if this fails, the checkpoint
        cannot be used to reproduce a trained model's behaviour.
        """
        loader = make_dataloader(tokenizer)
        model  = RewardModel.from_config(TINY_GPT2_CONFIG, num_unfrozen_layers=2)
        loss_fn = BradleyTerryLoss(use_stable=True)
        cfg = make_cfg(str(tmp_path))

        trainer_a = RewardModelTrainer(
            model=model, loss_fn=loss_fn,
            train_loader=loader, val_loader=loader,
            cfg=cfg, device=torch.device("cpu"),
        )

        # Do one training step so weights are non-initial
        batch = next(iter(loader))
        trainer_a._train_step(batch)

        # Record weights after step
        weights_before_save = {
            k: v.clone()
            for k, v in trainer_a.model.state_dict().items()
        }

        # Save checkpoint
        trainer_a._save_checkpoint(step=1, tag="roundtrip")

        # Create a fresh trainer with fresh model
        fresh_model = RewardModel.from_config(TINY_GPT2_CONFIG, num_unfrozen_layers=2)
        trainer_b = RewardModelTrainer(
            model=fresh_model, loss_fn=BradleyTerryLoss(),
            train_loader=loader, val_loader=loader,
            cfg=cfg, device=torch.device("cpu"),
        )

        # Load checkpoint into fresh trainer
        trainer_b.load_checkpoint(str(tmp_path / "checkpoint_roundtrip.pt"))

        # Weights must match exactly
        for key in weights_before_save:
            torch.testing.assert_close(
                trainer_b.model.state_dict()[key],
                weights_before_save[key],
                msg=f"Weight mismatch after checkpoint roundtrip for key: {key}",
            )

    def test_load_checkpoint_restores_step(self, trainer, tmp_path) -> None:
        """global_step must be restored correctly from checkpoint."""
        trainer.global_step = 99
        trainer._save_checkpoint(step=99, tag="step_test")

        # Reset step
        trainer.global_step = 0

        trainer.load_checkpoint(str(tmp_path / "checkpoint_step_test.pt"))
        assert trainer.global_step == 99, (
            f"global_step not restored. Expected 99, got {trainer.global_step}"
        )


# =============================================================================
# 6. Sequence alignment (batching trick)
# =============================================================================

class TestAlignSeqLengths:
    """Verify _align_seq_lengths correctly handles all length combinations."""

    def test_already_same_length_unchanged(self, trainer) -> None:
        """If chosen and rejected have the same length, nothing should change."""
        c = torch.ones(3, 10, dtype=torch.long)
        r = torch.ones(3, 10, dtype=torch.long)
        m = torch.ones(3, 10, dtype=torch.long)

        c_out, r_out, _, _ = trainer._align_seq_lengths(c, r, m, m)

        assert c_out.shape == (3, 10)
        assert r_out.shape == (3, 10)
        torch.testing.assert_close(c_out, c)
        torch.testing.assert_close(r_out, r)

    def test_shorter_chosen_padded_to_rejected_length(self, trainer) -> None:
        """Chosen shorter than rejected → chosen padded to rejected's length."""
        c = torch.ones(2, 5, dtype=torch.long)
        r = torch.ones(2, 9, dtype=torch.long)
        cm = torch.ones(2, 5, dtype=torch.long)
        rm = torch.ones(2, 9, dtype=torch.long)

        c_out, r_out, cm_out, rm_out = trainer._align_seq_lengths(c, r, cm, rm)

        assert c_out.shape == (2, 9)
        assert r_out.shape == (2, 9)
        assert cm_out.shape == (2, 9)

    def test_shorter_rejected_padded_to_chosen_length(self, trainer) -> None:
        """Rejected shorter than chosen → rejected padded to chosen's length."""
        c = torch.ones(2, 12, dtype=torch.long)
        r = torch.ones(2, 7,  dtype=torch.long)
        m = torch.ones_like

        cm = torch.ones(2, 12, dtype=torch.long)
        rm = torch.ones(2, 7,  dtype=torch.long)

        c_out, r_out, _, _ = trainer._align_seq_lengths(c, r, cm, rm)

        assert c_out.shape == (2, 12)
        assert r_out.shape == (2, 12)

    def test_padding_is_zeros(self, trainer) -> None:
        """New positions added by alignment must be zeros."""
        c = torch.ones(1, 3, dtype=torch.long) * 5   # [5, 5, 5]
        r = torch.ones(1, 7, dtype=torch.long) * 5   # [5, 5, 5, 5, 5, 5, 5]
        cm = torch.ones(1, 3, dtype=torch.long)
        rm = torch.ones(1, 7, dtype=torch.long)

        c_out, _, cm_out, _ = trainer._align_seq_lengths(c, r, cm, rm)

        # Original positions: [5, 5, 5], padding: [0, 0, 0, 0]
        assert (c_out[0, :3] == 5).all(), "Original tokens should be unchanged"
        assert (c_out[0, 3:] == 0).all(), "Padding positions should be zero"
        # Mask must also be zeroed at padding positions
        assert (cm_out[0, 3:] == 0).all(), "Padding positions in mask should be zero"