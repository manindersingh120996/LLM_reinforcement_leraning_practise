# Reward Model — From Scratch

A production-quality reward model built from first principles, implementing the Bradley-Terry pairwise ranking loss on top of a pretrained GPT-2 backbone. Trained on Anthropic's [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) preference dataset.

This is Phase 1 of a full RLHF implementation roadmap (PPO → DPO → GRPO).

---

## What This Is

A reward model takes a (prompt, response) pair and outputs a single scalar score representing how good the response is. It learns this by training on human preference data: pairs of (chosen, rejected) responses where humans labeled which response they preferred.

The loss function is derived from the [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model):

```
loss = -log(σ(r_chosen - r_rejected))
     = softplus(-(r_chosen - r_rejected))   # numerically stable form
```

The model learns to assign higher scores to chosen responses and lower scores to rejected ones.

---

## Repository Structure

```
reward_model/
├── configs/
│   └── default.yaml          # All hyperparameters — edit here, not in code
│
├── src/reward_model/
│   ├── loss.py               # BradleyTerryLoss (naive + numerically stable)
│   ├── model.py              # RewardModel: GPT-2 backbone + scalar head
│   ├── dataset.py            # PreferenceDataset, PreferenceCollator, DataLoaders
│   ├── trainer.py            # RewardModelTrainer: training loop + checkpointing
│   └── evaluator.py          # RewardModelEvaluator: metrics + EvaluationReport
│
├── scripts/
│   ├── train.py              # Entry point: python scripts/train.py
│   └── inference.py          # Score a (prompt, response) pair
│
└── tests/
    ├── conftest.py            # MinimalTokenizer, TINY_GPT2_CONFIG (shared)
    ├── test_loss.py           # 25 tests — Bradley-Terry loss properties
    ├── test_model.py          # 25 tests — architecture, padding, freezing
    ├── test_dataset.py        # 26 tests — tokenisation, collation, masking
    ├── test_trainer.py        # 25 tests — gradient flow, checkpointing
    └── test_evaluator.py      # 20 tests — metrics, Cohen's d, report format
```

---

## Installation

```bash
# Clone and enter the repo
git clone <repo-url>
cd reward_model

# Install in editable mode (changes to src/ are reflected immediately)
pip install -e ".[dev]"
```

---

## Training

```bash
# Train with the default config (downloads hh-rlhf on first run, ~2GB)
python scripts/train.py

# Override any config value from the command line
python scripts/train.py --config configs/default.yaml
```

The default config trains GPT-2 small with the top 4 transformer blocks unfrozen. Expected training time on a single A100: ~45 minutes for 1 epoch. Expected validation preference accuracy: 0.70–0.75.

Checkpoints are saved to `checkpoints/`. The best checkpoint (by validation preference accuracy) is saved as `checkpoint_best.pt`.

---

## Inference

```bash
# Compare two responses — the model says which is better
python scripts/inference.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --prompt "Human: What is the capital of France?" \
    --response-a "Paris is the capital of France." \
    --response-b "I don't know."

# Output:
# Response A score:  0.8312
# Response B score: -0.4521
# Difference (A-B):  1.2833
# Preferred response: A
```

---

## Testing

The full test suite runs in under 3 seconds with no network access:

```bash
# Run all 121 tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_loss.py -v

# Run with coverage
python -m pytest tests/ --cov=reward_model --cov-report=term-missing
```

**Test design principle**: every test encodes a mathematical invariant derived from theory, not just "does the code run." For example, `test_chance_level_equals_log_two` verifies that at initialization, the loss equals `log(2) ≈ 0.693` — the theoretical maximum uncertainty under the Bradley-Terry model. This is derived analytically, not empirically.

---

## Architecture

```
Input: [prompt tokens] + [response tokens]  →  shape (seq_len,)
         ↓
GPT-2 transformer backbone (12 layers, hidden_dim=768)
  - Layers 0–7: frozen (preserve pretrained language understanding)
  - Layers 8–11: trainable (adapt to quality ranking task)
  - Final layer norm: trainable
         ↓
Last real token hidden state  →  shape (hidden_dim,)
  (indexed via attention_mask.sum(dim=1) - 1 to ignore padding)
         ↓
Dropout(p=0.1) → Linear(768, 1, bias=False)  →  shape (1,)
         ↓
squeeze(-1)  →  scalar reward score
```

**Why the last token?** GPT-2 uses causal (left-to-right) attention. The last real token has attended to every preceding token in the sequence — it's the only position that has seen the complete context. Its hidden state is therefore the richest summary of the full (prompt, response) pair.

**Why no bias in the scalar head?** Only the *difference* between chosen and rejected scores matters — the absolute values cancel out in the loss. A bias term shifts all scores by a constant, which contributes nothing but adds a parameter to learn.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `GPT2Model` not `GPT2LMHeadModel` | We don't need vocabulary logits. `GPT2Model` gives raw hidden states; no need to load and discard the 50k-dim LM head. |
| Numerically stable loss (`softplus`) | Naive `-log(sigmoid(x))` underflows to NaN when `x = -100` (early training). `softplus(-x)` is mathematically equivalent but numerically safe everywhere. |
| Dynamic padding (in collator, not dataset) | Pad to the longest sequence in each batch, not to the global max. Reduces average computation per step by ~40% on hh-rlhf. |
| Save optimizer + scheduler in checkpoint | Resuming with weights-only causes a "cold restart" — the optimizer's momentum buffers are zeroed and the LR resets. Full checkpoint restore is seamless. |
| `from_config` factory for testing | Tests should never hit the network. A factory method that builds from a `GPT2Config` (random weights, no download) makes the entire test suite hermetic. |

---

## Evaluation Metrics

After training, `RewardModelEvaluator` computes:

- **Preference accuracy**: fraction of held-out pairs where chosen_score > rejected_score. Random baseline: 0.50. Target: 0.70–0.78. Above 0.82 often indicates overfitting.
- **Mean margin**: mean(chosen_score − rejected_score). A model with 75% accuracy and a margin of 2.5 is much more robust than one with 75% accuracy and margin 0.1.
- **Cohen's d**: effect size measuring how well the chosen and rejected score distributions are separated. d > 0.8 is "large" by Cohen's conventions — a well-trained reward model should reach this.

---

## Next Steps (Phase 2 — PPO)

This reward model is Phase 1 of the RLHF roadmap. In Phase 2:

1. **Initialize** a policy from the same GPT-2 backbone (or the SFT model).
2. **Generate** responses with the policy for a set of prompts.
3. **Score** each response with this reward model.
4. **Apply KL penalty**: `total_reward = r_model_score − β * KL(π_current ∥ π_SFT)`.
5. **Run PPO** to update the policy to maximize total reward.

The KL penalty prevents the policy from drifting too far from the original SFT model — without it, the policy reward-hacks by finding responses that score highly on the reward model but are degenerate in practice.