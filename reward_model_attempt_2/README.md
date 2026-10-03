# GPT-2 Reward Model from Scratch

A from-scratch implementation of a GPT-2-based reward model trained on the Anthropic HH-RLHF preference dataset.

The project focuses on understanding reward modeling from first principles: preference data, the Bradley–Terry objective, reward-model architecture, optimization, checkpointing, evaluation, reproducibility, and deployment as a Hugging Face Transformers model.

---

## Project Overview

A reward model learns to assign higher scores to responses that humans prefer.

Instead of training directly on a scalar quality label, this project uses pairwise preferences:

```text
Prompt
  │
  ├── Chosen response
  │
  └── Rejected response
```

The reward model produces:

```text
chosen response   → r_w
rejected response → r_l
```

and is trained so that:

$$
r_w > r_l
$$

The project implements this using the Bradley–Terry preference model.

---

## Final Results

The final model was trained on the full `Anthropic/hh-rlhf` training split with a deterministic validation split.

| Split      | Examples |       Loss |   Accuracy |
| ---------- | -------: | ---------: | ---------: |
| Train      |  144,720 |    0.5704* |    68.78%* |
| Validation |   16,080 |     0.6311 | **62.48%** |
| Test       |    8,552 | **0.6362** | **62.61%** |

* Epoch-5 training metrics are shown for reference. The selected model is the Epoch-4 checkpoint because it achieved the best validation accuracy.

### Selected checkpoint

```text
Epoch: 4
Validation Loss: 0.6311
Validation Accuracy: 62.48%
```

### Final untouched test set

```text
Test Loss:     0.6362
Test Accuracy: 62.61%
```

The test set was not used for checkpoint selection.

---

# Architecture

```text
                Response
                    │
                    ▼
             GPT-2 Tokenizer
                    │
                    ▼
              Token IDs
                    │
                    ▼
            GPT-2 Transformer
                    │
                    ▼
       Final non-padding hidden state
                    │
                    ▼
             Linear Reward Head
                    │
                    ▼
             Scalar Reward
```

For a preference pair:

```text
                ┌─────────────────────┐
Chosen ────────►│                     │
                │    Reward Model     │──► r_w
Rejected ──────►│                     │──► r_l
                └─────────────────────┘
                         │
                         ▼
                    r_w - r_l
                         │
                         ▼
                 Bradley–Terry loss
```

---

# Why a Reward Model?

A language model can generate many plausible responses, but supervised next-token prediction does not directly encode which of two complete responses a human would prefer.

Preference learning changes the training signal.

Instead of:

```text
response → quality label
```

we use:

```text
response A vs response B
          ↓
human preference
```

The reward model learns a scalar function whose ordering approximates those preferences.

---

# Bradley–Terry Objective

For chosen response \(y_w\) and rejected response \(y_l\):

$$
P(y_w \succ y_l)
=
\sigma(r_w-r_l)
$$

where:

$$
\sigma(z)
=
\frac{1}{1+e^{-z}}
$$

The negative log-likelihood becomes:

$$
\mathcal{L}
=
-\log\sigma(r_w-r_l)
$$

The implementation uses:

```python
torch.nn.functional.logsigmoid()
```

rather than explicitly calculating the sigmoid and then taking its logarithm.

This provides better numerical stability.

---

# Model Architecture Details

## GPT-2 Backbone

The project uses GPT-2 as the pretrained representation backbone.

The Transformer produces contextual token representations:

$$
H =
(h_1,h_2,\ldots,h_T)
$$

## Final Token Representation

The reward model needs a single representation for the complete response.

Because padding may be introduced dynamically during batching, the implementation finds the final non-padding token using the attention mask.

```python
sequence_lengths = attention_mask.sum(dim=1) - 1
```

The corresponding hidden state is selected and passed to the reward head.

## Reward Head

The reward head is a single linear projection:

$$
r = Wh+b
$$

producing one scalar reward.

There is deliberately no sigmoid on this output.

The Bradley–Terry objective only needs the relative difference:

$$
r_w-r_l
$$

---

# Dataset

Dataset:

```text
Anthropic/hh-rlhf
```

The original training split is deterministically divided into:

```text
Training:   144,720
Validation:  16,080
```

The original test split remains untouched:

```text
Test: 8,552
```

The test set is loaded separately and is only used after model selection.

---

# Data Pipeline

The data pipeline performs:

```text
HH-RLHF example
       ↓
chosen / rejected responses
       ↓
tokenization
       ↓
truncation
       ↓
dynamic padding
       ↓
DataLoader
       ↓
reward model
```

Chosen and rejected responses are tokenized independently.

Padding is performed dynamically by the batch collator rather than padding every dataset example to the maximum sequence length.

---

# Training

The training loop implements:

1. Forward pass for chosen response
2. Forward pass for rejected response
3. Bradley–Terry loss
4. Backpropagation
5. Gradient clipping
6. AdamW update
7. Learning-rate scheduler update
8. Training metrics
9. Validation
10. Best/latest checkpoint saving

---

# Optimization

## AdamW

AdamW is used as the optimizer.

Weight decay is separated into parameter groups.

Bias and normalization parameters are excluded from weight decay.

This gives:

```text
Parameters with weight decay:    51 tensors
Parameters without weight decay: 99 tensors
```

---

# Learning-Rate Scheduling

The training configuration uses:

```yaml
learning_rate: 1.0e-5

scheduler:
  type: "linear"
  warmup_ratio: 0.1
```

The intended schedule is:

```text
warmup
   ↓
maximum learning rate
   ↓
linear decay
   ↓
zero at the final planned optimizer step
```

The scheduler is defined in terms of **optimizer updates**, not examples.

For the full run:

```text
4,523 batches/epoch
× 5 epochs
= 22,615 optimizer steps
```

Warmup:

```text
22,615 × 0.1
≈ 2,261 steps
```

---

# Important Scheduler Debugging Lesson

During the final training experiment, the scheduler exposed an implementation inconsistency.

The scheduler was configured for:

```text
22,615 steps
```

but the recorded global step reached:

```text
27,138
```

This caused the learning rate to reach:

```text
0
```

during Epoch 5.

The best checkpoint was already selected at Epoch 4, so the final published model is the Epoch-4 checkpoint.

The repository subsequently treats epoch/global-step consistency as an explicit invariant.

For a checkpoint completed at epoch \(E\):

$$
global\_step
=
E\times steps\_per\_epoch
$$

A checkpoint that violates this invariant should not be silently resumed.

---

# Mixed Precision

Full training was performed using BF16 mixed precision.

```yaml
mixed_precision: "bf16"
```

This allowed the A40 GPU to train with a larger batch size while reducing memory pressure.

The training run used:

```text
Batch size: 32
GPU: NVIDIA A40
```

BF16 was applied to the model forward pass while the reward values were converted to FP32 for the Bradley–Terry loss.

---

# Gradient Clipping

Gradient clipping is applied before the optimizer update:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

This limits the global gradient norm and provides protection against unusually large updates.

---

# Reproducibility

The project explicitly seeds:

* Python
* NumPy
* PyTorch
* CUDA

and configures cuDNN for deterministic behavior.

The seed used for the final experiment was:

```yaml
seed: 42
```

---

# Checkpointing

Two checkpoints are maintained:

```text
checkpoints/
├── best_model.pt
└── latest_model.pt
```

### `best_model.pt`

Stores the checkpoint with the best validation metric.

### `latest_model.pt`

Stores the most recently completed epoch and is intended for resuming training.

Each checkpoint stores:

* model state
* optimizer state
* scheduler state
* epoch
* global step
* best metric
* validation loss
* validation accuracy
* configuration
* seed

---

# Development Before Full Training

The project was first tested using tiny datasets before committing GPU resources to the complete HH-RLHF dataset.

The development pipeline used small subsets to verify:

* dataset loading
* tokenization
* collating
* Bradley–Terry loss
* training
* evaluation
* checkpoint creation
* checkpoint resume
* best-checkpoint selection
* export
* Hugging Face loading
* AutoModel loading

This allowed the complete pipeline to be debugged before full training.

---

# Full Training Configuration

```yaml
seed: 42

model:
  name: "gpt2"

data:
  dataset_name: "Anthropic/hh-rlhf"
  train_split: "train"
  test_split: "test"
  max_length: 512
  validation_ratio: 0.1
  train_sample_size: null
  validation_sample_size: null
  test_sample_size: null
  num_workers: 0

training:
  batch_size: 32
  learning_rate: 1.0e-5
  weight_decay: 0.01
  num_epochs: 5
  gradient_clip_norm: 1.0
  mixed_precision: "bf16"

  scheduler:
    type: "linear"
    warmup_ratio: 0.1

checkpoint:
  directory: "checkpoints"
  best_model_name: "best_model.pt"
  latest_model_name: "latest_model.pt"
  metric: "accuracy"
  resume: false
```

---

# Project Structure

```text
reward_model/
│
├── configs/
│   └── default.yaml
│
├── src/
│   ├── data/
│   │   └── dataset.py
│   │
│   ├── models/
│   │   └── reward_model.py
│   │
│   ├── losses/
│   │   └── bradley_terry.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   └── utils/
│       ├── checkpoint.py
│       ├── config.py
│       └── seed.py
│
├── Scripts / Other usefull scripts
│
├── train.py
├── evaluate.py
├── inference.py
├── export.py
├── requirements.txt
└── README.md
```

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Linux / RunPod

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the PyTorch build appropriate for the target machine. PyTorch publishes CUDA-specific wheels for supported versions.

---

# Running the Project

## 1. Test Dataset

```bash
python Cell In the testing.ipynb 
```

## 2. Test Bradley–Terry Loss

```bash
python Cell In the testing.ipynb
```

## 3. Test Training Step

```bash
python Cell In the testing.ipynb
```

## 4. Test Evaluation

```bash
python Cell In the testing.ipynb
```

## 5. Test Checkpointing

```bash
python Cell In the testing.ipynb
```

## 6. Test Best Checkpoint Selection

```bash
python Cell In the testing.ipynb
```

## 7. Test Configuration

```bash
python Cell In the testing.ipynb
```

## 8. Test Data Loading

```bash
python Cell In the testing.ipynb
```

---

# Full Training

Set:

```yaml
train_sample_size: null
validation_sample_size: null
test_sample_size: null
```

Then:

```bash
python train.py
```

For long-running remote training, using `tmux` is recommended:

```bash
tmux new -s reward_training
python train.py
```

Detach with:

```text
Ctrl+B
D
```

Reconnect later with:

```bash
tmux attach -t reward_training
```

---

# Evaluation

After training:

```bash
python evaluate.py
```

The evaluation loads:

```text
checkpoints/best_model.pt
```

rather than the latest checkpoint.

This ensures that test performance is measured using the checkpoint selected by validation performance.

---

# Export

Export the best checkpoint into a Hugging Face Transformers-compatible directory:

```bash
python -m src.scripts.export_hf_model
```

The exported directory contains:

```text
exported_model_hf/
├── config.json
├── configuration_reward.py
├── merges.txt
├── model.safetensors
├── modeling_reward.py
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

---

# Export Verification

The exported model was compared against the original PyTorch checkpoint.

Final verification:

```text
Examples tested: 20
Maximum reward difference: 0.0000000000

PASS: Exported model matches the original checkpoint.
```

This confirms numerical equivalence for the tested examples.

---

# Hugging Face AutoModel

The custom model is implemented using:

```python
PreTrainedModel
```

and the configuration uses:

```python
PretrainedConfig
```

The model registers itself for:

```text
AutoConfig
AutoModel
```

using the repository's custom configuration and modeling files.

The resulting Hub configuration contains an `auto_map` mapping the Auto classes to the custom implementation.

This allows users to load the model through:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "philomath-1209/gpt2-reward_model_hh-rlhf",
    trust_remote_code=True,
)
```

---

# Why `trust_remote_code=True`?

This project does not correspond to a built-in Transformers model class.

The Hub repository therefore contains:

```text
configuration_reward.py
modeling_reward.py
```

These files define the custom reward-model architecture.

`trust_remote_code=True` allows Transformers to load that custom implementation.

Only use this option with repositories whose code you trust.

---

# Hugging Face Repository

The final model is published at:

`philomath-1209/gpt2-reward_model_hh-rlhf`

The repository contains the model weights, tokenizer, configuration, custom Transformers implementation, and model card.

---

# What This Project Demonstrates

This project intentionally covers the complete lifecycle of a small reward-model system:

```text
Preference Dataset
       ↓
Data Processing
       ↓
Model Architecture
       ↓
Preference Objective
       ↓
Optimization
       ↓
Validation
       ↓
Checkpoint Selection
       ↓
Test Evaluation
       ↓
Model Export
       ↓
Numerical Verification
       ↓
Hugging Face Hub
       ↓
Remote Inference
```

The focus is not only on obtaining a trained model, but on understanding the engineering and mathematical decisions that make the training pipeline reliable and reproducible.

---

# Future Work

The trained reward model provides the foundation for future RLHF experiments.

Possible next stages include:

```text
Policy Model
     ↓
Generate Responses
     ↓
Reward Model
     ↓
Scalar Rewards
     ↓
RL Optimization
     ↓
Updated Policy
```

The reward model can also be used independently to rank multiple candidate responses.

---

# Limitations

The reported test accuracy is a pairwise preference accuracy on the HH-RLHF test set.

It should not be interpreted as a universal measure of response quality or alignment.

The model's behavior is dependent on:

* the training dataset
* the preference annotations
* the GPT-2 backbone
* the training configuration
* the distribution of the evaluation data

A higher reward should be interpreted as a relative preference under the learned reward function rather than an absolute quality score.

---

# Reproducibility Notes

The project uses:

```text
Seed: 42
Maximum sequence length: 512
Batch size: 32
BF16 mixed precision
Learning rate: 1e-5
Weight decay: 0.01
Gradient clipping: 1.0
```

The final model was selected using validation accuracy and evaluated once on the untouched test split.

The final export was independently verified against the original checkpoint.

---

# Final Takeaway

The main objective of this project was to understand reward modeling by implementing the major components rather than treating reward modeling as a black-box training command.

The resulting model achieved:

```text
Validation Accuracy: 62.48%
Test Accuracy:       62.61%
```

on the HH-RLHF preference task used in this project.

More importantly, the project provides a complete path from:

```text
Human preference pairs
        ↓
Mathematical preference objective
        ↓
GPT-2 reward model
        ↓
Validated checkpoint
        ↓
Hugging Face model
```

and forms the foundation for the next stage of experimentation with RLHF.
