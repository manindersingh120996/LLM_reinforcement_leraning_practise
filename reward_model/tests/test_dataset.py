"""
Unit tests for PreferenceDataset and PreferenceCollator.

Design: zero network access, zero file system dependencies.

The two tools that make this possible:
    1. PreferenceDataset.from_list  — build dataset from a plain Python list
    2. MinimalTokenizer             — character-level tokenizer, no downloads

MinimalTokenizer converts each character to its ASCII value and returns
PyTorch tensors in exactly the format HuggingFace tokenizers return.
Its tokenization is meaningless for NLP — but that doesn't matter here.
We are testing the PIPELINE (squeeze, padding, masking, batching), not
whether GPT-2 BPE tokenization is correct. Those are different concerns.

This is a general principle: test one thing at a time. Use the simplest
possible implementation of every dependency that isn't what you're testing.
"""

import pytest
import torch

from reward_model.dataset import (
    PreferenceDataset,
    PreferenceCollator,
    PreferenceBatch,
)


# =============================================================================
# Minimal tokenizer — no downloads, same interface as HuggingFace tokenizers
# =============================================================================

class MinimalTokenizer:
    """
    Character-level tokenizer for testing. No pretrained weights, no network.

    Interface matches HuggingFace PreTrainedTokenizer exactly (the parts
    we use in dataset.py), so it can be swapped in transparently.

    Tokenization: each character → its ASCII value (0-127).
    Special tokens: pad=0, eos=127.

    Why build this instead of mocking?
    A mock would return hardcoded tensors. This actually runs the same code
    path (calling tokenizer, getting tensors, squeezing, padding) with a
    real — if simple — tokenizer. It's closer to a real integration test.
    """

    pad_token_id: int = 0
    eos_token_id: int = 127
    pad_token: str = "\x00"

    def __call__(
        self,
        text: str,
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = False,
        return_tensors: str = "pt",
    ) -> dict:
        """
        Tokenize a string. Returns the same dict structure as HuggingFace.
        """
        # ASCII values, clipped to [0, 127], truncated to max_length
        token_ids = [ord(c) % 128 for c in text][:max_length]
        attention_mask = [1] * len(token_ids)

        if return_tensors == "pt":
            # HuggingFace always returns (1, seq_len) — the leading 1 is the
            # batch dimension. __getitem__ squeezes it away. This is exactly
            # the behaviour we test in TestDatasetItems below.
            return {
                "input_ids":      torch.tensor([token_ids],      dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }

        raise ValueError(f"Unsupported return_tensors: {return_tensors}")


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def tokenizer() -> MinimalTokenizer:
    return MinimalTokenizer()


@pytest.fixture
def examples() -> list[dict]:
    """Four synthetic preference pairs of varying length."""
    return [
        {
            "chosen":   "Human: What is 2+2?\nAssistant: It is 4.",       # ~40 chars
            "rejected": "Human: What is 2+2?\nAssistant: I don't know.",  # ~43 chars
        },
        {
            "chosen":   "Human: Hi\nAssistant: Hello, how are you?",      # ~39 chars
            "rejected": "Human: Hi\nAssistant: ...",                      # ~22 chars
        },
        {
            "chosen":   "Human: Tell me about Paris.\nAssistant: Paris is the capital of France, known for the Eiffel Tower.",
            "rejected": "Human: Tell me about Paris.\nAssistant: It is a city.",
        },
        {
            "chosen":   "Human: Short.\nAssistant: Yes.",
            "rejected": "Human: Short.\nAssistant: No.",
        },
    ]


@pytest.fixture
def dataset(tokenizer, examples) -> PreferenceDataset:
    return PreferenceDataset.from_list(tokenizer, examples, max_length=128)


@pytest.fixture
def collator(tokenizer) -> PreferenceCollator:
    return PreferenceCollator(pad_token_id=tokenizer.pad_token_id)


# =============================================================================
# 1. from_list classmethod
# =============================================================================

class TestFromList:
    """Verify the from_list alternative constructor works correctly."""

    def test_returns_preference_dataset_instance(self, tokenizer, examples) -> None:
        """from_list must return a PreferenceDataset, not some other type."""
        ds = PreferenceDataset.from_list(tokenizer, examples)
        assert isinstance(ds, PreferenceDataset)

    def test_length_matches_input_list(self, tokenizer, examples) -> None:
        """Dataset length must equal number of examples passed in."""
        ds = PreferenceDataset.from_list(tokenizer, examples)
        assert len(ds) == len(examples)

    def test_max_length_is_stored(self, tokenizer, examples) -> None:
        """max_length config must be respected and stored on the dataset."""
        ds = PreferenceDataset.from_list(tokenizer, examples, max_length=64)
        assert ds.max_length == 64

    def test_single_example_works(self, tokenizer) -> None:
        """Dataset of one example must work (boundary case)."""
        single = [{"chosen": "a", "rejected": "b"}]
        ds = PreferenceDataset.from_list(tokenizer, single)
        assert len(ds) == 1

    def test_empty_dataset_works(self, tokenizer) -> None:
        """Empty dataset must return length 0 without crashing."""
        ds = PreferenceDataset.from_list(tokenizer, [])
        assert len(ds) == 0


# =============================================================================
# 2. Dataset items — __getitem__
# =============================================================================

class TestDatasetItems:
    """Verify that __getitem__ returns correctly shaped and typed tensors."""

    def test_returns_four_keys(self, dataset) -> None:
        """Each item must have exactly the four expected keys."""
        item = dataset[0]
        expected_keys = {
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        }
        assert set(item.keys()) == expected_keys

    def test_all_tensors_are_1d(self, dataset) -> None:
        """
        Tensors must be 1D after squeeze(0).

        This is the key test for the squeeze(0) step. HuggingFace tokenizers
        return (1, seq_len). If we forgot to squeeze, these would be 2D and
        the collator would produce 3D tensors, crashing the model.
        """
        item = dataset[0]
        for key, tensor in item.items():
            assert tensor.dim() == 1, (
                f"Expected 1D tensor for '{key}', got {tensor.dim()}D. "
                f"Did you forget .squeeze(0) in __getitem__?"
            )

    def test_all_tensors_are_long_dtype(self, dataset) -> None:
        """Token IDs and attention masks must be int64 for embedding lookup."""
        item = dataset[0]
        for key, tensor in item.items():
            assert tensor.dtype == torch.long, (
                f"Expected torch.long for '{key}', got {tensor.dtype}. "
                f"Embedding layers require integer inputs."
            )

    def test_chosen_and_rejected_have_same_length(self, tokenizer) -> None:
        """
        When chosen and rejected texts are the same length, their token
        sequences should also have the same length.
        """
        # Craft a pair where chosen and rejected are same character count
        examples = [{"chosen": "AAAAAA", "rejected": "BBBBBB"}]
        ds = PreferenceDataset.from_list(tokenizer, examples, max_length=128)
        item = ds[0]

        assert item["chosen_input_ids"].shape == item["rejected_input_ids"].shape

    def test_truncation_respects_max_length(self, tokenizer) -> None:
        """
        Sequences longer than max_length must be truncated to max_length.
        This verifies that max_length is actually applied.
        """
        long_text = "A" * 500   # 500 characters
        examples = [{"chosen": long_text, "rejected": long_text}]
        ds = PreferenceDataset.from_list(tokenizer, examples, max_length=50)
        item = ds[0]

        assert item["chosen_input_ids"].shape[0] <= 50, (
            f"Sequence not truncated. Length: {item['chosen_input_ids'].shape[0]}"
        )

    def test_short_text_not_padded(self, tokenizer) -> None:
        """
        Short sequences must NOT be padded here — padding is the collator's job.
        """
        examples = [{"chosen": "Hi", "rejected": "Ok"}]  # 2-char texts
        ds = PreferenceDataset.from_list(tokenizer, examples, max_length=512)
        item = ds[0]

        # "Hi" tokenizes to 2 tokens — should NOT be padded to 512
        assert item["chosen_input_ids"].shape[0] == 2, (
            f"Short sequence was padded in __getitem__. "
            f"Got length {item['chosen_input_ids'].shape[0]}, expected 2. "
            f"Padding should only happen in the collator."
        )

    def test_attention_mask_all_ones_no_padding(self, tokenizer) -> None:
        """
        Without padding, attention mask should be all 1s (every token is real).
        """
        examples = [{"chosen": "Hello", "rejected": "World"}]
        ds = PreferenceDataset.from_list(tokenizer, examples, max_length=512)
        item = ds[0]

        assert item["chosen_attention_mask"].all(), (
            "Attention mask should be all 1s when no padding is applied"
        )

    def test_input_ids_and_mask_same_length(self, dataset) -> None:
        """input_ids and attention_mask must always be the same length."""
        for i in range(len(dataset)):
            item = dataset[i]
            assert item["chosen_input_ids"].shape == item["chosen_attention_mask"].shape
            assert item["rejected_input_ids"].shape == item["rejected_attention_mask"].shape

    def test_dataset_is_indexable(self, dataset) -> None:
        """All indices 0..len(dataset)-1 must be accessible without error."""
        for i in range(len(dataset)):
            item = dataset[i]
            assert "chosen_input_ids" in item


# =============================================================================
# 3. Collator — padding and batching
# =============================================================================

class TestCollator:
    """
    Verify that the collator correctly pads variable-length sequences and
    produces properly structured PreferenceBatch objects.
    """

    def _make_item(self, chosen_len: int, rejected_len: int) -> dict:
        """Helper: create a fake dataset item with specified sequence lengths."""
        return {
            "chosen_input_ids":        torch.ones(chosen_len,   dtype=torch.long),
            "chosen_attention_mask":   torch.ones(chosen_len,   dtype=torch.long),
            "rejected_input_ids":      torch.ones(rejected_len, dtype=torch.long),
            "rejected_attention_mask": torch.ones(rejected_len, dtype=torch.long),
        }

    def test_returns_preference_batch(self, collator) -> None:
        """Collator must return a PreferenceBatch dataclass."""
        items = [self._make_item(10, 8), self._make_item(6, 12)]
        batch = collator(items)
        assert isinstance(batch, PreferenceBatch)

    def test_output_is_2d(self, collator) -> None:
        """All output tensors must be 2D (batch_size, seq_len)."""
        items = [self._make_item(10, 8), self._make_item(6, 12)]
        batch = collator(items)

        assert batch.chosen_input_ids.dim() == 2
        assert batch.chosen_attention_mask.dim() == 2
        assert batch.rejected_input_ids.dim() == 2
        assert batch.rejected_attention_mask.dim() == 2

    def test_batch_size_correct(self, collator) -> None:
        """First dimension of all tensors must equal the number of items."""
        n_items = 5
        items = [self._make_item(10, 8) for _ in range(n_items)]
        batch = collator(items)

        assert batch.chosen_input_ids.shape[0] == n_items
        assert batch.rejected_input_ids.shape[0] == n_items

    def test_pads_to_longest_sequence(self, collator) -> None:
        """
        Sequences must be padded to the length of the LONGEST sequence in
        the batch — not to some fixed global max_length.

        This is the core test for dynamic padding. If this fails, the collator
        is doing static padding, which wastes computation.
        """
        # Chosen lengths: 10, 6, 8 → should pad to 10
        # Rejected lengths: 5, 12, 7 → should pad to 12
        items = [
            self._make_item(chosen_len=10, rejected_len=5),
            self._make_item(chosen_len=6,  rejected_len=12),
            self._make_item(chosen_len=8,  rejected_len=7),
        ]
        batch = collator(items)

        assert batch.chosen_input_ids.shape == (3, 10), (
            f"Expected chosen shape (3, 10), got {batch.chosen_input_ids.shape}"
        )
        assert batch.rejected_input_ids.shape == (3, 12), (
            f"Expected rejected shape (3, 12), got {batch.rejected_input_ids.shape}"
        )

    def test_padding_uses_correct_pad_token_id(self, collator) -> None:
        """
        Padded positions in input_ids must contain the pad_token_id (0 in
        MinimalTokenizer), not random values.
        """
        # Item 0: length 4. Item 1: length 8.
        # After padding to 8, item 0 should have 4 zeros appended.
        pad_id = 0   # MinimalTokenizer.pad_token_id
        items = [
            {"chosen_input_ids":        torch.ones(4, dtype=torch.long) * 99,
             "chosen_attention_mask":   torch.ones(4, dtype=torch.long),
             "rejected_input_ids":      torch.ones(4, dtype=torch.long) * 99,
             "rejected_attention_mask": torch.ones(4, dtype=torch.long)},
            {"chosen_input_ids":        torch.ones(8, dtype=torch.long) * 99,
             "chosen_attention_mask":   torch.ones(8, dtype=torch.long),
             "rejected_input_ids":      torch.ones(8, dtype=torch.long) * 99,
             "rejected_attention_mask": torch.ones(8, dtype=torch.long)},
        ]
        batch = PreferenceCollator(pad_token_id=pad_id)(items)

        # First 4 positions of row 0 should be real tokens (99)
        assert (batch.chosen_input_ids[0, :4] == 99).all(), "Real tokens overwritten"
        # Last 4 positions of row 0 should be pad_id (0)
        assert (batch.chosen_input_ids[0, 4:] == pad_id).all(), "Padding wrong value"

    def test_attention_mask_zeros_at_padding_positions(self, collator) -> None:
        """
        The attention mask must be 0 at padding positions and 1 at real positions.
        This is the critical invariant the model relies on to find the last real token.
        """
        items = [
            {"chosen_input_ids":        torch.ones(3, dtype=torch.long),
             "chosen_attention_mask":   torch.ones(3, dtype=torch.long),
             "rejected_input_ids":      torch.ones(3, dtype=torch.long),
             "rejected_attention_mask": torch.ones(3, dtype=torch.long)},
            {"chosen_input_ids":        torch.ones(7, dtype=torch.long),
             "chosen_attention_mask":   torch.ones(7, dtype=torch.long),
             "rejected_input_ids":      torch.ones(7, dtype=torch.long),
             "rejected_attention_mask": torch.ones(7, dtype=torch.long)},
        ]
        batch = collator(items)

        # Row 0 chosen: 3 real tokens → positions 0,1,2 should be 1; positions 3..6 should be 0
        assert (batch.chosen_attention_mask[0, :3] == 1).all(), "Real positions should be 1"
        assert (batch.chosen_attention_mask[0, 3:] == 0).all(), "Padding positions should be 0"

        # Row 1 chosen: 7 real tokens, no padding → all 1s
        assert (batch.chosen_attention_mask[1, :] == 1).all(), "No padding: all should be 1"

    def test_single_item_batch_works(self, collator) -> None:
        """Batch size of 1 must work (used in inference)."""
        items = [self._make_item(10, 8)]
        batch = collator(items)
        assert batch.chosen_input_ids.shape[0] == 1

    def test_all_same_length_no_padding_needed(self, collator) -> None:
        """
        When all sequences have the same length, no padding is added.
        The attention mask should be all 1s.
        """
        items = [self._make_item(10, 10) for _ in range(4)]
        batch = collator(items)

        assert batch.chosen_input_ids.shape == (4, 10)
        assert (batch.chosen_attention_mask == 1).all(), (
            "When no padding needed, attention mask should be all 1s"
        )

    def test_real_tokens_unchanged_by_collator(self, collator) -> None:
        """
        The collator must not modify the actual token values — only append
        padding. Original token IDs must survive collation unchanged.
        """
        token_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)
        items = [
            {"chosen_input_ids": token_ids,
             "chosen_attention_mask": torch.ones(4, dtype=torch.long),
             "rejected_input_ids": token_ids,
             "rejected_attention_mask": torch.ones(4, dtype=torch.long)},
            {"chosen_input_ids": torch.ones(8, dtype=torch.long),
             "chosen_attention_mask": torch.ones(8, dtype=torch.long),
             "rejected_input_ids": torch.ones(8, dtype=torch.long),
             "rejected_attention_mask": torch.ones(8, dtype=torch.long)},
        ]
        batch = collator(items)

        torch.testing.assert_close(
            batch.chosen_input_ids[0, :4],
            token_ids,
            msg="Collator must not modify original token IDs",
        )


# =============================================================================
# 4. End-to-end: dataset → collator → batch
# =============================================================================

class TestEndToEnd:
    """
    Verify the full pipeline: dataset items → collator → batch ready for model.
    """

    def test_batch_from_real_dataset_items(self, dataset, collator) -> None:
        """
        Collect items from a real dataset and collate them.
        All output tensors must be finite (no NaN, no extreme values).
        """
        items = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = collator(items)

        assert not torch.isnan(batch.chosen_input_ids.float()).any()
        assert not torch.isnan(batch.rejected_input_ids.float()).any()

    def test_batch_shapes_are_consistent(self, dataset, collator) -> None:
        """
        input_ids and attention_mask must have the same shape for both
        chosen and rejected. The model reads them together.
        """
        items = [dataset[i] for i in range(len(dataset))]
        batch = collator(items)

        assert batch.chosen_input_ids.shape == batch.chosen_attention_mask.shape, (
            "chosen input_ids and attention_mask must have identical shape"
        )
        assert batch.rejected_input_ids.shape == batch.rejected_attention_mask.shape, (
            "rejected input_ids and attention_mask must have identical shape"
        )

    def test_mask_sum_equals_original_lengths(self, tokenizer, collator) -> None:
        """
        After padding, attention_mask.sum(dim=1) for each sequence must equal
        the original sequence length before padding.

        This is the invariant the model uses to find the last real token:
            last_real_idx = attention_mask.sum(dim=1) - 1

        If this breaks, the model indexes into padding positions instead of
        real content — the most common subtle bug in reward model implementations.
        """
        examples = [
            {"chosen": "Hi",      "rejected": "Ok"},        # short
            {"chosen": "Hello!",  "rejected": "Goodbye!"},  # medium
            {"chosen": "A" * 20,  "rejected": "B" * 15},   # longer
        ]
        ds = PreferenceDataset.from_list(tokenizer, examples, max_length=128)
        items = [ds[i] for i in range(len(ds))]

        # Record original lengths before collation
        original_chosen_lens  = [item["chosen_input_ids"].shape[0]  for item in items]
        original_rejected_lens = [item["rejected_input_ids"].shape[0] for item in items]

        batch = collator(items)

        # After padding, the mask sum must recover original lengths
        recovered_chosen_lens  = batch.chosen_attention_mask.sum(dim=1).tolist()
        recovered_rejected_lens = batch.rejected_attention_mask.sum(dim=1).tolist()

        assert recovered_chosen_lens == original_chosen_lens, (
            f"Mask sums don't match original lengths.\n"
            f"Original: {original_chosen_lens}\n"
            f"Recovered: {recovered_chosen_lens}"
        )
        assert recovered_rejected_lens == original_rejected_lens, (
            f"Rejected mask sums don't match original lengths.\n"
            f"Original: {original_rejected_lens}\n"
            f"Recovered: {recovered_rejected_lens}"
        )