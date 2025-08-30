import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from .pooling import AttentiveStatsPooling
from .quality_gates import FrontEndQualityGates, create_quality_gates
from .audio_conditioning import AudioConditioningModule, create_audio_conditioning

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", adapter_dim: int = 256, freeze_base: bool = True, 
                 use_quality_gates: bool = True, vad_method: str = "webrtc",
                 use_audio_conditioning: bool = True):
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
        
        # Initialize quality gates
        self.use_quality_gates = use_quality_gates
        if use_quality_gates:
            self.quality_gates = create_quality_gates(vad_method=vad_method)
            # Quality feature fusion layer
            self.quality_fusion = nn.Sequential(
                nn.Linear(hid + 8, hid),  # 8 quality features
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # Initialize audio conditioning
        self.use_audio_conditioning = use_audio_conditioning
        if use_audio_conditioning:
            self.audio_conditioning = create_audio_conditioning()
            # Conditioning feature fusion layer
            self.conditioning_fusion = nn.Sequential(
                nn.Linear(hid + 12, hid),  # 12 conditioning features
                nn.ReLU(),
                nn.Dropout(0.1)
            )

    def forward(self, audio_waveforms, texts=None):
        """
        audio_waveforms: List[1D Tensor] of variable lengths
        texts: Optional list of text strings for language detection
        """
        # Process each audio file individually to avoid padding issues
        batch_seqs = []
        batch_masks = []
        quality_features_list = []
        conditioning_features_list = []
        
        for i, audio in enumerate(audio_waveforms):
            # Apply quality gates if enabled
            if self.use_quality_gates:
                text = texts[i] if texts and i < len(texts) else None
                processed_audio, quality_metrics, should_process = self.quality_gates(audio, text)
                
                # Store quality features for fusion
                quality_features_list.append(quality_metrics.quality_features)
                
                # If quality gates recommend rejection, use zero audio
                if not should_process:
                    # Create zero audio of same length
                    processed_audio = torch.zeros_like(audio)
            else:
                processed_audio = audio
                quality_features_list.append(torch.zeros(8, device=audio.device))
            
            # Apply audio conditioning if enabled
            if self.use_audio_conditioning:
                conditioned_audio, conditioning_features = self.audio_conditioning(processed_audio)
                processed_audio = conditioned_audio
                conditioning_features_list.append(conditioning_features.conditioning_features)
            else:
                conditioning_features_list.append(torch.zeros(12, device=audio.device))
            
            # Process single audio file
            inputs = self.feature_extractor(
                [processed_audio],
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
            
            # Fuse quality and conditioning features if enabled
            if self.use_quality_gates or self.use_audio_conditioning:
                # Expand features to match sequence length
                if self.use_quality_gates:
                    quality_features = quality_features_list[i].unsqueeze(0).expand(seq.size(1), -1)
                else:
                    quality_features = torch.zeros(seq.size(1), 8, device=seq.device)
                
                if self.use_audio_conditioning:
                    conditioning_features = conditioning_features_list[i].unsqueeze(0).expand(seq.size(1), -1)
                else:
                    conditioning_features = torch.zeros(seq.size(1), 12, device=seq.device)
                
                # Concatenate all features
                all_features = torch.cat([quality_features, conditioning_features], dim=-1)
                seq = torch.cat([seq.squeeze(0), all_features], dim=-1)
                
                # Apply appropriate fusion
                if self.use_quality_gates and self.use_audio_conditioning:
                    # Combined fusion for both quality and conditioning features
                    seq = self.quality_fusion(seq).unsqueeze(0)
                elif self.use_quality_gates:
                    seq = self.quality_fusion(seq).unsqueeze(0)
                elif self.use_audio_conditioning:
                    seq = self.conditioning_fusion(seq).unsqueeze(0)
            
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
