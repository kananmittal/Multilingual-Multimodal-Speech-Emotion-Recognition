import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, proj_dim: int):
        super().__init__()
        self.proj_a = nn.Sequential(
            nn.Linear(audio_dim, proj_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(proj_dim, proj_dim)
        )
        self.proj_t = nn.Sequential(
            nn.Linear(text_dim, proj_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(proj_dim, proj_dim)
        )
        gate_hidden = max(32, proj_dim // 2)
        self.gate_a = nn.Sequential(nn.Linear(proj_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        self.gate_t = nn.Sequential(nn.Linear(proj_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))

    def forward(self, audio_vec: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        a = self.proj_a(audio_vec)
        t = self.proj_t(text_vec)
        wa = torch.sigmoid(self.gate_a(a))  # [B,1]
        wt = torch.sigmoid(self.gate_t(t))  # [B,1]
        wsum = wa + wt + 1e-8
        wa, wt = wa / wsum, wt / wsum
        return wa * a + wt * t
