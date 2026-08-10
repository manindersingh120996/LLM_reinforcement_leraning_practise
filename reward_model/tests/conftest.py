"""
Shared test utilities — auto-discovered by pytest, no import needed.
"""

import torch
from reward_model.model import RewardModelConfig


# Tiny config: 4 layers, hidden_dim=64, vocab=128
# Creates a ~100K param model in under 1 second. No downloads.
TINY_REWARD_CONFIG = RewardModelConfig(
    n_embd=64, n_layer=4, n_head=2,
    n_positions=128, vocab_size=128,
    dropout=0.0, num_unfrozen_layers=2,
)


class MinimalTokenizer:
    """
    Character-level tokenizer for testing. No pretrained weights, no network.
    ASCII values (0-127) as token IDs — matches TINY_REWARD_CONFIG.vocab_size=128.
    """
    pad_token_id: int = 0
    eos_token_id: int = 127
    pad_token: str = "\x00"
    eos_token: str = "\x7f"

    def __call__(self, text, max_length=512, truncation=True,
                 padding=False, return_tensors="pt"):
        token_ids = [ord(c) % 128 for c in text][:max_length]
        attention_mask = [1] * len(token_ids)
        if return_tensors == "pt":
            return {
                "input_ids":      torch.tensor([token_ids],      dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        raise ValueError(f"Unsupported return_tensors: {return_tensors}")