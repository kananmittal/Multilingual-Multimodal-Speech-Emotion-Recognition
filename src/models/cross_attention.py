import torch
import torch.nn as nn
from typing import Optional


class CrossModalAttention(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, shared_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Ensure shared_dim is divisible by num_heads
        self.shared_dim = shared_dim
        self.num_heads = num_heads
        assert shared_dim % num_heads == 0, f"shared_dim {shared_dim} must be divisible by num_heads {num_heads}"
        
        # Audio attending to Text
        self.q_a = nn.Linear(audio_dim, shared_dim)
        self.k_t = nn.Linear(text_dim, shared_dim)
        self.v_t = nn.Linear(text_dim, shared_dim)
        self.attn_a = nn.MultiheadAttention(shared_dim, num_heads, dropout=dropout, batch_first=True)
        self.out_a = nn.Linear(shared_dim, audio_dim)

        # Text attending to Audio
        self.q_t = nn.Linear(text_dim, shared_dim)
        self.k_a = nn.Linear(audio_dim, shared_dim)
        self.v_a = nn.Linear(audio_dim, shared_dim)
        self.attn_t = nn.MultiheadAttention(shared_dim, num_heads, dropout=dropout, batch_first=True)
        self.out_t = nn.Linear(shared_dim, text_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(audio_dim)
        self.norm_t = nn.LayerNorm(text_dim)

    def forward(self, audio_seq: torch.Tensor, text_seq: torch.Tensor, audio_mask: Optional[torch.Tensor] = None, text_mask: Optional[torch.Tensor] = None):
        # Prepare masks for MultiheadAttention: need key padding mask (True for pad)
        t_kpm = (text_mask == 0) if text_mask is not None else None
        a_kpm = (audio_mask == 0) if audio_mask is not None else None

        # A <- T
        qa = self.q_a(audio_seq)  # [B, S_a, shared_dim]
        kt = self.k_t(text_seq)   # [B, S_t, shared_dim]
        vt = self.v_t(text_seq)   # [B, S_t, shared_dim]
        a_ctx, _ = self.attn_a(qa, kt, vt, key_padding_mask=t_kpm)
        a_out = self.out_a(a_ctx)
        audio_enh = self.norm_a(audio_seq + self.dropout(a_out))

        # T <- A
        qt = self.q_t(text_seq)   # [B, S_t, shared_dim]
        ka = self.k_a(audio_seq)  # [B, S_a, shared_dim]
        va = self.v_a(audio_seq)  # [B, S_a, shared_dim]
        t_ctx, _ = self.attn_t(qt, ka, va, key_padding_mask=a_kpm)
        t_out = self.out_t(t_ctx)
        text_enh = self.norm_t(text_seq + self.dropout(t_out))

        return audio_enh, text_enh


