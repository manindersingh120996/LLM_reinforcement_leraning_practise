from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


class PreferenceDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        chosen = example["chosen"]
        rejected = example["rejected"]

        chosen_encoding = self.tokenizer(
            chosen,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        rejected_encoding = self.tokenizer(
            rejected,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
        }


import torch


class PreferenceCollator:

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        chosen_input_ids = self._pad(
            [item["chosen_input_ids"] for item in batch],
            pad_value=self.pad_token_id,
        )

        chosen_attention_mask = self._pad(
            [item["chosen_attention_mask"] for item in batch],
            pad_value=0,
        )

        rejected_input_ids = self._pad(
            [item["rejected_input_ids"] for item in batch],
            pad_value=self.pad_token_id,
        )

        rejected_attention_mask = self._pad(
            [item["rejected_attention_mask"] for item in batch],
            pad_value=0,
        )

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

    def _pad(
        self,
        sequences: list[torch.Tensor],
        pad_value: int,
    ) -> torch.Tensor:

        max_length = max(
            sequence.size(0)
            for sequence in sequences
        )

        padded_sequences = []

        for sequence in sequences:

            padding_length = max_length - sequence.size(0)

            if padding_length > 0:

                padding = torch.full(
                    (padding_length,),
                    fill_value=pad_value,
                    dtype=sequence.dtype,
                )

                sequence = torch.cat(
                    [sequence, padding]
                )

            padded_sequences.append(sequence)

        return torch.stack(padded_sequences)

def create_dataloader(
    dataset: PreferenceDataset,
    tokenizer,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
):
    collator = PreferenceCollator(
        pad_token_id=tokenizer.pad_token_id
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return dataloader