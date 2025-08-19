import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .pooling import AttentiveStatsPooling

class TextEncoder(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", adapter_dim: int = 256, freeze_base: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hid = self.encoder.config.hidden_size
        self.adapter = nn.Sequential(
            nn.Linear(hid, adapter_dim), nn.ReLU(), nn.Linear(adapter_dim, hid)
        )
        self.pool = AttentiveStatsPooling(hid)

    def forward(self, text_list):
        # text_list: List[str] of length batch
        encoded = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded)
        seq = outputs.last_hidden_state  # [batch, seq, hid]
        seq = seq + self.adapter(seq)
        mask = encoded.get("attention_mask", None)
        if mask is not None:
            mask = mask.to(seq.dtype)
        return seq, mask
