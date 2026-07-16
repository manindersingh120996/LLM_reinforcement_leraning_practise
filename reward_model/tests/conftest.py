"""
Shared test utilities.

pytest automatically discovers conftest.py and makes everything in it
available to all test files in the same directory and below — no import
needed. Fixtures defined here can be used by any test module.

We put two things here that are needed across multiple test files:

    MinimalTokenizer — a zero-dependency, zero-download character-level
                       tokenizer with the same interface as HuggingFace
                       tokenizers. Used in test_dataset.py and test_trainer.py.

    TINY_GPT2_CONFIG — a GPT-2Config with 4 layers, 2 heads, hidden_dim=64.
                       Creates a ~100K parameter model in under a second.
                       Used in test_model.py and test_trainer.py.
"""

import torch
from transformers import GPT2Config


# ---------------------------------------------------------------------------
# Tiny GPT-2 config — fast model creation, no downloads
# ---------------------------------------------------------------------------

TINY_GPT2_CONFIG = GPT2Config(
    n_layer=4,
    n_head=2,
    n_embd=64,
    n_positions=128,
    vocab_size=128,          # matches MinimalTokenizer (ASCII range)
)


# ---------------------------------------------------------------------------
# Minimal tokenizer — same interface as HuggingFace, no downloads
# ---------------------------------------------------------------------------

class MinimalTokenizer:
    """
    Character-level tokenizer for testing.

    Converts each character to its ASCII value (0-127).
    Returns tensors in exactly the format HuggingFace tokenizers return,
    so it can replace a real tokenizer anywhere in the codebase for tests.

    We use ASCII values because our TINY_GPT2_CONFIG has vocab_size=128,
    so any ASCII character is a valid token ID (no out-of-bounds embeddings).

    pad_token_id = 0   (null character)
    eos_token_id = 127 (DEL character — never appears in normal text)
    """

    pad_token_id: int = 0
    eos_token_id: int = 127
    pad_token: str = "\x00"
    eos_token: str = "\x7f"

    def __call__(
        self,
        text: str,
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = False,
        return_tensors: str = "pt",
    ) -> dict:
        token_ids = [ord(c) % 128 for c in text][:max_length]
        attention_mask = [1] * len(token_ids)

        if return_tensors == "pt":
            return {
                "input_ids":      torch.tensor([token_ids],      dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }

        raise ValueError(f"Unsupported return_tensors: {return_tensors}")