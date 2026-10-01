# GPT-2 Reward Model for Human Preference Modeling

A GPT-2-based reward model trained to assign scalar rewards to responses according to human preference data.

> **Development checkpoint:** This repository currently contains a small development/test model trained on only 20 training examples from `Anthropic/hh-rlhf`. It was created to validate the complete reward-model training, evaluation, checkpointing, serialization, and inference pipeline. It is **not a production-quality reward model** and its development metrics should not be interpreted as representative of the model's eventual performance after full training.

---

## Model Description

This model is a preference-based reward model built on top of GPT-2.

Given a response, the model produces a single scalar reward:

$$
r(x) \in \mathbb{R}
$$

where higher values indicate that the model considers the response more preferable according to the preference data used during training.

For a preference pair consisting of a preferred response \(y_w\) and a rejected response \(y_l\), the model is trained so that:

$$
r(y_w) > r(y_l)
$$

The model uses a Bradley-Terry preference objective:

$$
P(y_w \succ y_l)
=
\sigma(r(y_w)-r(y_l))
$$

with the loss:

$$
L
=
-\log
\sigma(r(y_w)-r(y_l))
$$

---

## Architecture

The model consists of:

```text
Input response
      │
      ▼
   GPT-2
      │
      ▼
Final non-padding hidden state
      │
      ▼
Linear reward head
      │
      ▼
Scalar reward
```

### Backbone

* GPT-2
* Hugging Face `GPT2Model`
* Hidden representation taken from the final non-padding token

### Reward head

A single linear layer maps the final hidden representation to one scalar:

$$
r = W h + b
$$

No sigmoid is applied to the final reward.

The reward remains an unrestricted real-valued scalar because the Bradley-Terry objective operates on the difference between two rewards.

---

## Training Dataset

The model was trained using:

**Dataset:** `Anthropic/hh-rlhf`

The dataset provides preference pairs containing:

* a preferred (`chosen`) response
* a rejected (`rejected`) response

The development pipeline separates the data into:

```text
Original training split
        │
        ├── Training examples
        └── Validation examples

Original test split
        │
        └── Test examples
```

The test split is kept separate from training and validation.

---

## Development Training Configuration

The currently exported checkpoint was intentionally trained using a very small dataset to validate the implementation.

| Parameter               | Value               |
| ----------------------- | ------------------- |
| Backbone                | GPT-2               |
| Dataset                 | `Anthropic/hh-rlhf` |
| Training examples       | 20                  |
| Validation examples     | 20                  |
| Test examples           | 20                  |
| Maximum sequence length | 512                 |
| Batch size              | 4                   |
| Learning rate           | `1e-5`              |
| Weight decay            | `0.01`              |
| Gradient clipping       | `1.0`               |
| Scheduler               | Linear              |
| Warmup ratio            | `0.1`               |
| Epochs                  | 5                   |
| Preference objective    | Bradley-Terry       |
| Random seed             | 42                  |

The development configuration was intentionally small so that the complete pipeline could be repeatedly tested locally before running full-scale training.

---

## Development Results

The best checkpoint was selected using validation accuracy.

### Best validation result

```text
Validation accuracy: 65%
```

The final evaluation was then performed on the separate development test subset.

### Development test result

```text
Test examples: 20
Test loss:     1.1870
Test accuracy: 55%
```

With only 20 test examples, each example corresponds to 5 percentage points of accuracy.

Therefore, these numbers should **not** be interpreted as a reliable estimate of generalization performance.

The purpose of this experiment was primarily to verify that:

* preference data can be loaded correctly
* tokenization works
* dynamic padding works
* the reward model produces scalar scores
* Bradley-Terry training works
* validation evaluation works
* the best checkpoint can be selected
* training can resume from checkpoints
* optimizer and scheduler state can be restored
* the exported model reproduces the trained model exactly
* the exported model can be loaded independently for inference

---

## Model Export Verification

The trained model was exported from the best training checkpoint into Hugging Face-compatible files.

The exported model was then loaded independently and compared against the original training checkpoint.

The maximum reward difference across the tested examples was:

```text
0.0000000000
```

This verifies that the serialized model reproduces the original checkpoint outputs for the tested inputs.

---

## Intended Use

The reward model is intended to be used as a learned preference function.

For example, given two candidate responses:

```text
Prompt
  │
  ├── Response A
  │
  └── Response B
```

the model can produce:

```text
r(A)
r(B)
```

and the responses can be compared:

```python
if reward_a > reward_b:
    preferred = response_a
else:
    preferred = response_b
```

The model can also serve as the reward component in an RLHF-style training pipeline where candidate responses are generated by a policy and the reward model provides a scalar training signal.

---

## Loading the Model

The model can be loaded using the Hugging Face Hub-compatible interface:

```python
from src.models.reward_model import GPT2RewardModel

model = GPT2RewardModel.from_pretrained(
    "philomath-1209/gpt2-reward_model_hh-rlhf"
)
```

For local use:

```python
from src.models.reward_model import GPT2RewardModel

model = GPT2RewardModel.from_pretrained(
    "./exported_model"
)
```

The model should be placed in evaluation mode for inference:

```python
model.eval()
```

---

## Scoring a Response

```python
import torch
from transformers import AutoTokenizer

from src.models.reward_model import GPT2RewardModel


model = GPT2RewardModel.from_pretrained(
    "philomath-1209/gpt2-reward_model_hh-rlhf"
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model.eval()

response = (
    "The Earth revolves around the Sun."
)

inputs = tokenizer(
    response,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

with torch.no_grad():

    reward = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

print(reward.item())
```

The output is a scalar reward value.

The absolute numerical value of a reward should not be interpreted independently. For preference modeling, the important quantity is generally the difference between rewards:

$$
\Delta r =
r(y_a)-r(y_b)
$$

---

## Comparing Two Responses

```python
import torch
from transformers import AutoTokenizer

from src.models.reward_model import GPT2RewardModel


model = GPT2RewardModel.from_pretrained(
    "philomath-1209/gpt2-reward_model_hh-rlhf"
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model.eval()


response_a = "Response A goes here."

response_b = "Response B goes here."


inputs_a = tokenizer(
    response_a,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

inputs_b = tokenizer(
    response_b,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)


with torch.no_grad():

    reward_a = model(
        input_ids=inputs_a["input_ids"],
        attention_mask=inputs_a["attention_mask"],
    )

    reward_b = model(
        input_ids=inputs_b["input_ids"],
        attention_mask=inputs_b["attention_mask"],
    )


reward_a = reward_a.item()
reward_b = reward_b.item()

print("Response A:", reward_a)
print("Response B:", reward_b)

if reward_a > reward_b:
    print("Model preference: Response A")
else:
    print("Model preference: Response B")
```

---

## How the Reward Model Is Trained

For each preference pair:

```text
chosen response
      │
      ▼
GPT-2 + reward head
      │
      ▼
r_chosen


rejected response
      │
      ▼
GPT-2 + reward head
      │
      ▼
r_rejected
```

The reward difference is:

$$
\Delta r =
r_{\text{chosen}}
-
r_{\text{rejected}}
$$

The Bradley-Terry probability is:

$$
P(\text{chosen} \succ \text{rejected})
=
\sigma(\Delta r)
$$

and the training loss is:

$$
L =
-\log\sigma(\Delta r)
$$

Thus, training encourages the reward model to assign higher rewards to preferred responses.

---

## Training Pipeline

The complete development pipeline used for this model was:

```text
Anthropic/hh-rlhf
        │
        ▼
Preference Dataset
        │
        ▼
Tokenization + Dynamic Padding
        │
        ▼
GPT-2 Backbone
        │
        ▼
Scalar Reward Head
        │
        ▼
Bradley-Terry Loss
        │
        ▼
AdamW
        │
        ├── Weight decay
        ├── Gradient clipping
        └── Linear LR schedule + warmup
        │
        ▼
Validation
        │
        ▼
Best Checkpoint
        │
        ▼
Final Test Evaluation
        │
        ▼
Hugging Face Export
```

---

## Checkpointing and Reproducibility

The training pipeline supports:

* best-model checkpointing
* latest-checkpoint checkpointing
* optimizer-state restoration
* scheduler-state restoration
* epoch restoration
* global-step restoration
* validation metric restoration
* deterministic random seed configuration

The development training used:

```text
Seed: 42
```

---

## Limitations

This development checkpoint has significant limitations.

### 1. Extremely small training dataset

Only 20 training examples were used.

This is intentionally insufficient for training a useful general-purpose reward model.

### 2. Extremely small evaluation sets

Only 20 validation examples and 20 test examples were used.

The reported 55% test accuracy therefore has very high statistical uncertainty.

### 3. Development model, not final model

This checkpoint exists to validate the engineering and training pipeline.

A full training run on the complete dataset is required before drawing conclusions about the model's generalization ability.

### 4. Reward values are not absolute quality scores

A reward such as:

```text
3.5
```

does not inherently mean that a response has some universal quality level.

The model learns relative preferences. Reward differences are more meaningful than isolated reward values.

### 5. Preference-model bias

The model inherits biases and limitations from the preference data used during training.

A high reward does not guarantee factual correctness, safety, usefulness, or truthfulness.

---

## Future Full Training

The next version of this model is intended to be trained using the full available training data rather than the development subset.

The planned full-training configuration will:

1. Load the complete training split.
2. Create a reproducible validation split from the training data.
3. Keep the original test split untouched.
4. Train the reward model on the full training portion.
5. Select the best checkpoint using validation performance.
6. Evaluate the selected checkpoint once on the held-out test set.
7. Export the final model.
8. Publish the resulting model to the Hugging Face Hub.

The metrics from that full training run should replace the development metrics in the final model card.

---

## Repository Status

**Current status:** Development / pipeline validation

The current model should be considered an experimental checkpoint rather than a production-ready reward model.

The primary purpose of this release is to demonstrate a complete implementation of a GPT-2 reward model trained with preference learning and the Bradley-Terry objective, including training, evaluation, checkpointing, resumption, serialization, and inference.
