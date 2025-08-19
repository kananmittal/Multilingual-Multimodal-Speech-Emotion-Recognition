import torch
import torch.nn as nn


class AttentiveStatsPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [batch, seq_len, input_dim]
        mask: [batch, seq_len] with 1 for valid, 0 for pad (optional)
        returns: [batch, 2 * input_dim] concatenated (mean, std)
        """
        attn_logits = self.attention(x).squeeze(-1)  # [B, S]
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, S, 1]
        mean = torch.sum(attn * x, dim=1)  # [B, D]
        var = torch.sum(attn * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mean, std], dim=-1)


