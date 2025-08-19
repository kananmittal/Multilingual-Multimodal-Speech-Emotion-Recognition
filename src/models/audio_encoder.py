import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from .pooling import AttentiveStatsPooling

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", adapter_dim: int = 256, freeze_base: bool = True):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hid = self.encoder.config.hidden_size
        self.adapter = nn.Sequential(
            nn.Linear(hid, adapter_dim), nn.ReLU(), nn.Linear(adapter_dim, hid)
        )
        self.pool = AttentiveStatsPooling(hid)

    def forward(self, audio_waveforms):
        """
        audio_waveforms: List[1D Tensor] of variable lengths
        """
        # Process each audio file individually to avoid padding issues
        batch_seqs = []
        batch_masks = []
        
        for audio in audio_waveforms:
            # Process single audio file
            inputs = self.feature_extractor(
                [audio],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Ensure input_values is 2D [batch, time] for Wav2Vec2
            if inputs['input_values'].dim() == 4:
                inputs['input_values'] = inputs['input_values'].squeeze(1).squeeze(1)
            elif inputs['input_values'].dim() == 3:
                if inputs['input_values'].size(1) > 1:
                    inputs['input_values'] = inputs['input_values'].mean(dim=1)
                else:
                    inputs['input_values'] = inputs['input_values'].squeeze(1)
            
            outputs = self.encoder(**inputs)
            seq = outputs.last_hidden_state  # [1, seq, hidden]
            seq = seq + self.adapter(seq)
            mask = inputs.get("attention_mask", None)
            if mask is not None:
                mask = mask.to(seq.dtype)
            
            batch_seqs.append(seq.squeeze(0))  # Remove batch dimension
            batch_masks.append(mask.squeeze(0) if mask is not None else None)
        
        # Pad sequences to same length
        max_len = max(seq.size(0) for seq in batch_seqs)
        padded_seqs = []
        padded_masks = []
        
        for seq, mask in zip(batch_seqs, batch_masks):
            # Pad sequence
            if seq.size(0) < max_len:
                pad_len = max_len - seq.size(0)
                seq = torch.nn.functional.pad(seq, (0, 0, 0, pad_len))
            
            # Pad mask
            if mask is not None and mask.size(0) < max_len:
                pad_len = max_len - mask.size(0)
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
            elif mask is None:
                mask = torch.ones(max_len, dtype=seq.dtype, device=seq.device)
            
            padded_seqs.append(seq)
            padded_masks.append(mask)
        
        # Stack into batch
        batch_seq = torch.stack(padded_seqs)
        batch_mask = torch.stack(padded_masks)
        
        return batch_seq, batch_mask
