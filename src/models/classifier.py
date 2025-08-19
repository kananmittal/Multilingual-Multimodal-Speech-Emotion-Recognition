import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden: int = 128, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
