"""Simple reward model placeholder."""

import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)
