from torch.utils.data import Dataset
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