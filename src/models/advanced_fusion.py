import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention mechanism for multimodal fusion."""
    
    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Projections for audio and text
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention layers
        self.audio_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_to_audio_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor, 
                audio_mask: Optional[torch.Tensor] = None, text_mask: Optional[torch.Tensor] = None):
        """
        Args:
            audio_features: [batch, audio_seq_len, audio_dim]
            text_features: [batch, text_seq_len, text_dim]
            audio_mask: [batch, audio_seq_len] or None
            text_mask: [batch, text_seq_len] or None
        """
        # Project to common hidden dimension
        audio_proj = self.audio_proj(audio_features)  # [batch, audio_seq_len, hidden_dim]
        text_proj = self.text_proj(text_features)     # [batch, text_seq_len, hidden_dim]
        
        # Cross-attention: audio attends to text
        audio_attended, _ = self.audio_to_text_attn(
            audio_proj, text_proj, text_proj,
            key_padding_mask=text_mask
        )
        audio_enhanced = self.norm1(audio_attended + audio_proj)
        audio_enhanced = self.norm1(audio_enhanced + self.ffn1(audio_enhanced))
        
        # Cross-attention: text attends to audio
        text_attended, _ = self.text_to_audio_attn(
            text_proj, audio_proj, audio_proj,
            key_padding_mask=audio_mask
        )
        text_enhanced = self.norm2(text_attended + text_proj)
        text_enhanced = self.norm2(text_enhanced + self.ffn2(text_enhanced))
        
        return audio_enhanced, text_enhanced

class AdvancedFusionLayer(nn.Module):
    """Advanced fusion layer with transformer-based cross-attention and ensemble methods."""
    
    def __init__(self, audio_dim: int, text_dim: int, fusion_dim: int = 2048, num_heads: int = 8):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Multi-head cross-attention
        self.cross_attention = MultiHeadCrossAttention(audio_dim, text_dim, fusion_dim, num_heads)
        
        # Ensemble fusion methods
        self.gated_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.concat_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.additive_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Fusion weight predictor
        self.fusion_weights = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 3),  # 3 fusion methods
            nn.Softmax(dim=-1)
        )
        
        # Self-attention for final refinement
        self.self_attention = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True)
        self.final_norm = nn.LayerNorm(fusion_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, audio_vec: torch.Tensor, text_vec: torch.Tensor, 
                audio_mask: Optional[torch.Tensor] = None, text_mask: Optional[torch.Tensor] = None):
        """
        Args:
            audio_vec: [batch, audio_dim] - pooled audio features
            text_vec: [batch, text_dim] - pooled text features
        """
        batch_size = audio_vec.size(0)
        
        # Project to fusion dimension first
        audio_proj = self.cross_attention.audio_proj(audio_vec)  # [batch, fusion_dim]
        text_proj = self.cross_attention.text_proj(text_vec)     # [batch, fusion_dim]
        
        # For pooled features, we'll use a simpler fusion approach
        # since we don't have sequence-level information
        concat_features = torch.cat([audio_proj, text_proj], dim=-1)
        
        # Ensemble fusion methods
        gated_out = self.gated_fusion(concat_features)
        concat_out = self.concat_fusion(concat_features)
        additive_out = self.additive_fusion(audio_proj + text_proj)
        
        # Learn fusion weights
        fusion_weights = self.fusion_weights(concat_features)  # [batch, 3]
        
        # Weighted ensemble
        fused = (fusion_weights[:, 0:1] * gated_out + 
                fusion_weights[:, 1:2] * concat_out + 
                fusion_weights[:, 2:3] * additive_out)
        
        # Self-attention refinement (simplified for pooled features)
        fused_seq = fused.unsqueeze(1)  # [batch, 1, fusion_dim]
        refined, _ = self.self_attention(fused_seq, fused_seq, fused_seq)
        refined = self.final_norm(refined.squeeze(1) + fused)
        
        # Final output projection
        output = self.output_proj(refined)
        
        return output

class MultiScaleFusionLayer(nn.Module):
    """Multi-scale fusion using different layers of encoders."""
    
    def __init__(self, audio_dims: list, text_dims: list, fusion_dim: int = 2048):
        super().__init__()
        self.num_scales = len(audio_dims)
        
        # Scale-specific fusion layers
        self.scale_fusions = nn.ModuleList([
            AdvancedFusionLayer(audio_dims[i], text_dims[i], fusion_dim // self.num_scales)
            for i in range(self.num_scales)
        ])
        
        # Scale combination
        self.scale_combiner = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, audio_features: list, text_features: list):
        """
        Args:
            audio_features: List of [batch, audio_dim_i] for different scales
            text_features: List of [batch, text_dim_i] for different scales
        """
        scale_outputs = []
        
        for i, (audio_feat, text_feat) in enumerate(zip(audio_features, text_features)):
            scale_out = self.scale_fusions[i](audio_feat, text_feat)
            scale_outputs.append(scale_out)
        
        # Combine scales
        combined = torch.cat(scale_outputs, dim=-1)
        output = self.scale_combiner(combined)
        
        return output
