import torch

from transformers import AutoModel, AutoTokenizer


MODEL_ID = "YOUR_USERNAME/gpt2-reward-model"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model.eval()


chosen = "The Earth revolves around the Sun."
rejected = "The Sun revolves around the Earth."


chosen_inputs = tokenizer(
    chosen,
    return_tensors="pt",
)

rejected_inputs = tokenizer(
    rejected,
    return_tensors="pt",
)


with torch.no_grad():
    chosen_reward = model(
        **chosen_inputs
    )

    rejected_reward = model(
        **rejected_inputs
    )


print("Remote Hugging Face Model")
print("-------------------------")
print(
    f"Chosen reward:   "
    f"{chosen_reward.item():.6f}"
)
print(
    f"Rejected reward: "
    f"{rejected_reward.item():.6f}"
)