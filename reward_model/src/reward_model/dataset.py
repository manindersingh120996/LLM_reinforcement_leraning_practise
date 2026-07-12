"""
Preference Dataset for Reward Model Training.

Loads (prompt, chosen_response, rejected_response) preference pairs and
tokenizes them for input to RewardModel.

Data source: Anthropic/hh-rlhf on HuggingFace.

Each example in hh-rlhf looks like:
    {
        "chosen":   "\\n\\nHuman: <prompt>\\n\\nAssistant: <chosen response>",
        "rejected": "\\n\\nHuman: <prompt>\\n\\nAssistant: <rejected response>"
    }

Both fields are the complete conversation — prompt AND response together.
This is why we tokenize the full string rather than prompt and response
separately.

Key design decisions:
    - No padding in __getitem__: padding is done dynamically in the collator
      to minimise wasted computation across variable-length sequences.
    - Two factory classmethods: from_config (production) and from_list (tests).
    - squeeze(0): HuggingFace tokenizers return (1, seq_len); we strip the
      leading batch dimension here and let the collator add it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Data structure returned by the collator
# ---------------------------------------------------------------------------

@dataclass
class PreferenceBatch:
    """
    A padded batch of preference pairs, ready to feed into RewardModel.

    Attributes
    ----------
    chosen_input_ids : torch.Tensor
        Shape (batch_size, chosen_seq_len). Token IDs for chosen responses.
        chosen_seq_len = max sequence length in this batch after dynamic padding.
    chosen_attention_mask : torch.Tensor
        Shape (batch_size, chosen_seq_len). 1 for real tokens, 0 for padding.
    rejected_input_ids : torch.Tensor
        Shape (batch_size, rejected_seq_len). Token IDs for rejected responses.
    rejected_attention_mask : torch.Tensor
        Shape (batch_size, rejected_seq_len). 1 for real tokens, 0 for padding.

    Note: chosen_seq_len and rejected_seq_len can differ within the same batch.
    The model processes chosen and rejected separately, so they don't need to
    be the same length.
    """

    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """
    Dataset of (chosen, rejected) human preference pairs.

    Tokenizes sequences on-the-fly in __getitem__. In production with very
    large datasets (>1M examples) you would pre-tokenize and cache to disk
    using datasets.map(). For this project, on-the-fly is simpler and fast
    enough.

    Construction — two classmethods, one for each use case:
        production:  PreferenceDataset.from_config(tokenizer, cfg, split)
        testing:     PreferenceDataset.from_list(tokenizer, examples)

    Never call __init__ directly — use one of the classmethods above.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data,                         # HuggingFace Dataset or plain list
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        cfg: DictConfig,
        split: str,
    ) -> PreferenceDataset:
        """
        Production constructor: load hh-rlhf from HuggingFace.

        Requires internet access on first call (caches locally afterwards).

        Args:
            tokenizer: Must have pad_token set (typically to eos_token for GPT-2).
            cfg:       OmegaConf config with cfg.data.* fields.
            split:     "train" or "test".
        """
        from datasets import load_dataset  # lazy import: only needed in production

        raw = load_dataset(cfg.data.dataset_name, split=split)

        # Subsample validation set to keep eval fast
        if split == cfg.data.val_split and cfg.data.val_size < len(raw):
            raw = raw.select(range(cfg.data.val_size))

        return cls(tokenizer=tokenizer, data=raw, max_length=cfg.data.max_length)

    @classmethod
    def from_list(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        examples: list[dict],
        max_length: int = 512,
    ) -> PreferenceDataset:
        """
        Test constructor: build a dataset from a plain Python list.

        No network access, no HuggingFace dependency. Each element of
        `examples` must be a dict with "chosen" and "rejected" string keys.

        Why a classmethod instead of just passing a list to __init__?
        Because classmethods are discoverable, self-documenting, and
        inherited automatically by subclasses. If you later subclass
        PreferenceDataset, from_list works on the subclass too without
        any changes.

        Example
        -------
        >>> examples = [
        ...     {"chosen":   "Human: Hi\\nAssistant: Hello!",
        ...      "rejected": "Human: Hi\\nAssistant: ..."},
        ... ]
        >>> ds = PreferenceDataset.from_list(tokenizer, examples, max_length=64)
        >>> len(ds)
        1
        """
        return cls(tokenizer=tokenizer, data=examples, max_length=max_length)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Tokenize one preference pair and return 1D tensors.

        The output has no batch dimension. The collator adds it when
        it stacks multiple __getitem__ outputs into a batch.

        Why squeeze(0)?
        HuggingFace tokenizers always return tensors with a leading batch
        dimension, even when tokenizing a single string:
            tokenizer("hello", return_tensors="pt")["input_ids"]  → shape (1, N)
        We strip that dimension here so each item is shape (N,).
        The collator receives a list of (N,) tensors and pads them to the same
        length before stacking into (batch_size, max_N).
        If we did NOT squeeze, the collator would produce (batch_size, 1, N) —
        a 3D tensor that would immediately crash the model's forward pass.

        Why padding=False?
        We want dynamic padding (pad only to the longest sequence in each
        batch), not static padding (pad everything to max_length). Dynamic
        padding is handled in the collator. If we padded here, we'd waste
        memory on padding tokens that carry no information.
        """
        item = self.data[idx]

        chosen_enc = self.tokenizer(
            item["chosen"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        rejected_enc = self.tokenizer(
            item["rejected"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids":        chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask":   chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids":      rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class PreferenceCollator:
    """
    Collates a list of preference pair dicts into a padded PreferenceBatch.

    The DataLoader calls this on every batch after gathering individual
    __getitem__ outputs. Its job: pad variable-length sequences to the same
    length so they can be stacked into a 2D tensor.

    Why a callable class instead of a plain function?
    The collator needs to store state: the pad_token_id. A class lets you
    store that in __init__ and use it in __call__. A plain function would
    need it passed every call or captured in a closure — both more awkward.

    Dynamic padding explanation:
        Static:  pad every sequence to max_length (512). Fast to implement,
                 wastes computation. If average sequence = 150 tokens, you
                 process 362 padding tokens per sequence doing nothing.
        Dynamic: pad each batch only to its longest sequence. If the longest
                 sequence in a batch is 180 tokens, pad to 180. Average waste
                 near zero. Training is significantly faster.

    Args:
        pad_token_id: Token ID used for padding input_ids.
                      Attention mask padding is always 0 (hardcoded).
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> PreferenceBatch:
        """
        Pad and stack a list of variable-length preference pairs.

        Args:
            batch: List of dicts from PreferenceDataset.__getitem__.
                   Each dict has four 1D tensors of variable lengths.

        Returns:
            PreferenceBatch with padded 2D tensors.
        """
        return PreferenceBatch(
            chosen_input_ids=self._pad(
                [item["chosen_input_ids"] for item in batch],
                pad_value=self.pad_token_id,
            ),
            chosen_attention_mask=self._pad(
                [item["chosen_attention_mask"] for item in batch],
                pad_value=0,
            ),
            rejected_input_ids=self._pad(
                [item["rejected_input_ids"] for item in batch],
                pad_value=self.pad_token_id,
            ),
            rejected_attention_mask=self._pad(
                [item["rejected_attention_mask"] for item in batch],
                pad_value=0,
            ),
        )

    def _pad(
        self,
        sequences: list[torch.Tensor],
        pad_value: int,
    ) -> torch.Tensor:
        """
        Right-pad a list of 1D tensors to the same length, then stack.

        Why right-padding specifically?
        Our model uses attention_mask.sum(dim=1) - 1 to find the last real
        token. This works correctly with both left and right padding.
        Right padding is the HuggingFace convention for decoder-only models
        and is what most implementations expect.

        Args:
            sequences: List of 1D tensors, each shape (seq_len_i,).
            pad_value: Value to fill padding positions with.

        Returns:
            Tensor of shape (len(sequences), max_seq_len).
        """
        max_len = max(seq.shape[0] for seq in sequences)

        padded = []
        for seq in sequences:
            n_pad = max_len - seq.shape[0]
            if n_pad > 0:
                pad_tensor = torch.full((n_pad,), fill_value=pad_value, dtype=seq.dtype)
                seq = torch.cat([seq, pad_tensor])
            padded.append(seq)

        return torch.stack(padded)  # (batch_size, max_len)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    train_dataset: PreferenceDataset,
    val_dataset: PreferenceDataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders with the preference collator.

    Design decisions:
        shuffle=True  for training:   ensures the model doesn't memorise
                                      the order of preference pairs.
        shuffle=False for validation: reproducible eval metrics.
        drop_last=True for training:  all training batches have identical size,
                                      which matters for stable gradient estimates.
                                      The last batch is often smaller and noisier.
        drop_last=False for val:      evaluate on every single example — we don't
                                      want to miss any val examples just because
                                      the set doesn't divide evenly into batches.
        pin_memory=True:              copies tensors to pinned (page-locked) CPU
                                      memory, making GPU transfer ~2x faster.
                                      Only useful when training on GPU.

    Args:
        train_dataset: Training split of PreferenceDataset.
        val_dataset:   Validation split of PreferenceDataset.
        tokenizer:     Must be the same tokenizer used to build both datasets.
        cfg:           Config (uses cfg.training.batch_size, cfg.data.num_workers).

    Returns:
        (train_loader, val_loader)
    """
    collator = PreferenceCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Tokenizer setup helper
# ---------------------------------------------------------------------------

def setup_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Load and configure a tokenizer for reward model training.

    GPT-2 does not define a pad token — it was designed for generation only.
    For reward model training we need batching, which requires padding.
    The standard fix: use the EOS token as the pad token.

    This works because the attention mask tells the model to ignore padding
    positions, so the specific token ID used for padding doesn't matter as
    long as it's a valid token.

    Args:
        model_name: HuggingFace model ID (e.g. "gpt2").

    Returns:
        Tokenizer with pad_token set.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        # GPT-2 specific: no pad token → use EOS token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer