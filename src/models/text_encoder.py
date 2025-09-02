import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .pooling import AttentiveStatsPooling
from .asr_integration import EnhancedASRIntegration, create_enhanced_asr

class TextEncoder(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", adapter_dim: int = 256, freeze_base: bool = True,
                 use_asr_integration: bool = True, asr_model_name: str = "openai/whisper-large-v3"):
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
        
        # Initialize ASR integration
        self.use_asr_integration = use_asr_integration
        if use_asr_integration:
            self.asr_integration = create_enhanced_asr(asr_model_name)
            # ASR feature fusion layer
            self.asr_fusion = nn.Sequential(
                nn.Linear(hid + 8, hid),  # 8 ASR features
                nn.ReLU(),
                nn.Dropout(0.1)
            )

    def forward(self, text_list, audio_waveforms=None):
        """
        text_list: List[str] of length batch (can be None if using ASR)
        audio_waveforms: Optional audio for ASR transcription
        """
        # Handle ASR integration when no text is provided
        if (text_list is None or all(text == "" for text in text_list)) and audio_waveforms is not None:
            # Use ASR to generate transcripts
            asr_results = []
            for audio in audio_waveforms:
                asr_result = self.asr_integration(audio)
                asr_results.append(asr_result)
                text_list = [result.text for result in asr_results]
        
        # Process text through encoder
        encoded = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded)
        seq = outputs.last_hidden_state  # [batch, seq, hid]
        seq = seq + self.adapter(seq)
        
        # Fuse ASR features if available
        if self.use_asr_integration and audio_waveforms is not None:
            asr_features_list = []
            for audio in audio_waveforms:
                asr_result = self.asr_integration(audio)
                asr_features_list.append(asr_result.asr_features)
            
            # Expand ASR features to match sequence length
            for i, asr_features in enumerate(asr_features_list):
                asr_features_expanded = asr_features.unsqueeze(0).expand(seq.size(1), -1)
                # Concatenate and fuse
                seq_i = torch.cat([seq[i], asr_features_expanded], dim=-1)
                seq[i] = self.asr_fusion(seq_i)
        
        mask = encoded.get("attention_mask", None)
        if mask is not None:
            mask = mask.to(seq.dtype)
        return seq, mask
