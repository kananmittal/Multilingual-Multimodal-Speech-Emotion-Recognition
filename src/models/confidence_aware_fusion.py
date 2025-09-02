import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging


@dataclass
class ConfidenceFeatures:
    """Container for all confidence and quality features used in fusion."""
    # Audio quality features
    snr_db: float
    speech_prob: float
    clipping_percent: float
    denoise_gain_db: float
    
    # Text quality features
    asr_conf_segment: float
    lid_entropy: float
    text_reliability_score: float
    
    # Processing flags
    conditioning_applied: bool
    quality_gates_passed: bool
    
    # Cross-modal agreement
    audio_text_similarity: float
    
    # Segmentation quality
    boundary_confidence: float
    segment_emotion_consistency: float
    
    # Temporal context
    previous_segment_confidence: float
    emotion_transition_probability: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert confidence features to tensor."""
        return torch.tensor([
            self.snr_db / 50.0,  # Normalize to [0, 1]
            self.speech_prob,
            self.clipping_percent / 100.0,
            self.denoise_gain_db / 20.0,  # Normalize to [0, 1]
            self.asr_conf_segment,
            self.lid_entropy / 2.0,  # Normalize to [0, 1]
            self.text_reliability_score,
            float(self.conditioning_applied),
            float(self.quality_gates_passed),
            self.audio_text_similarity,
            self.boundary_confidence,
            self.segment_emotion_consistency,
            self.previous_segment_confidence,
            self.emotion_transition_probability
        ], dtype=torch.float32)


class DynamicGatingMLP(nn.Module):
    """
    Dynamic Gating MLP that learns modality weights from confidence features.
    
    Architecture: 14 confidence features → 32 → 16 → 2 (audio_weight, text_weight)
    """
    
    def __init__(self, confidence_dim: int = 14, hidden_dim: int = 32):
        super().__init__()
        
        self.confidence_dim = confidence_dim
        self.hidden_dim = hidden_dim
        
        # Gating MLP architecture
        self.gating_mlp = nn.Sequential(
            nn.Linear(confidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),  # audio_weight, text_weight
            nn.Softmax(dim=-1)  # Ensure weights sum to 1.0
        )
        
        # Initialize with equal weighting
        with torch.no_grad():
            self.gating_mlp[-2].weight.fill_(0.0)
            self.gating_mlp[-2].bias.fill_(0.0)  # This will give equal weights after softmax
        
        logging.info(f"Dynamic Gating MLP initialized with {confidence_dim} confidence features")
    
    def forward(self, confidence_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic modality weights from confidence features.
        
        Args:
            confidence_features: [batch_size, confidence_dim]
            
        Returns:
            audio_weight: [batch_size, 1] - weight for audio modality
            text_weight: [batch_size, 1] - weight for text modality
        """
        # Ensure input is 2D
        if confidence_features.dim() == 1:
            confidence_features = confidence_features.unsqueeze(0)
        
        # Compute weights through gating MLP
        weights = self.gating_mlp(confidence_features)  # [batch_size, 2]
        
        # Split into audio and text weights
        audio_weight = weights[:, 0:1]  # [batch_size, 1]
        text_weight = weights[:, 1:2]   # [batch_size, 1]
        
        return audio_weight, text_weight


class PolicyBasedClamps(nn.Module):
    """
    Policy-based clamps that enforce confidence-based rules for modality weighting.
    """
    
    def __init__(self):
        super().__init__()
        
        # Confidence thresholds
        self.snr_threshold_low = 10.0  # dB
        self.asr_conf_threshold_low = 0.5
        self.lid_entropy_threshold_high = 1.5
        self.speech_prob_threshold_low = 0.7
        
        # Clamp values
        self.audio_weight_min = 0.1
        self.audio_weight_max = 0.9
        self.text_weight_min = 0.1
        self.text_weight_max = 0.9
        
        logging.info("Policy-based clamps initialized")
    
    def forward(self, audio_weight: torch.Tensor, text_weight: torch.Tensor, 
                confidence_features: ConfidenceFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply policy-based clamps to modality weights.
        
        Args:
            audio_weight: [batch_size, 1] - computed audio weight
            text_weight: [batch_size, 1] - computed text weight
            confidence_features: Confidence features for policy decisions
            
        Returns:
            clamped_audio_weight: [batch_size, 1] - clamped audio weight
            clamped_text_weight: [batch_size, 1] - clamped text weight
        """
        # Convert to tensor if needed
        if isinstance(confidence_features, ConfidenceFeatures):
            conf_tensor = confidence_features.to_tensor()
        else:
            conf_tensor = confidence_features
        
        # Apply policy-based clamps
        clamped_audio_weight = audio_weight.clone()
        clamped_text_weight = text_weight.clone()
        
        # Policy 1: If SNR < 10dB, cap audio_weight at 0.3, boost text_weight
        if confidence_features.snr_db < self.snr_threshold_low:
            clamped_audio_weight = torch.clamp(clamped_audio_weight, max=0.3)
            # Boost text weight proportionally
            clamped_text_weight = 1.0 - clamped_audio_weight
        
        # Policy 2: If ASR_conf < 0.5, cap text_weight at 0.4, boost audio_weight
        if confidence_features.asr_conf_segment < self.asr_conf_threshold_low:
            clamped_text_weight = torch.clamp(clamped_text_weight, max=0.4)
            # Boost audio weight proportionally
            clamped_audio_weight = 1.0 - clamped_text_weight
        
        # Policy 3: If LID_entropy > 1.5, reduce both weights, increase uncertainty
        if confidence_features.lid_entropy > self.lid_entropy_threshold_high:
            # Reduce both weights to indicate uncertainty
            clamped_audio_weight = clamped_audio_weight * 0.7
            clamped_text_weight = clamped_text_weight * 0.7
            # Normalize to sum to 1.0
            total_weight = clamped_audio_weight + clamped_text_weight
            clamped_audio_weight = clamped_audio_weight / total_weight
            clamped_text_weight = clamped_text_weight / total_weight
        
        # Policy 4: If speech_prob < 0.7, heavily penalize audio_weight
        if confidence_features.speech_prob < self.speech_prob_threshold_low:
            clamped_audio_weight = torch.clamp(clamped_audio_weight, max=0.2)
            # Boost text weight proportionally
            clamped_text_weight = 1.0 - clamped_audio_weight
        
        # Policy 5: Emergency fallback - if both modalities unreliable
        audio_unreliable = (confidence_features.snr_db < 5.0 and 
                           confidence_features.speech_prob < 0.5)
        text_unreliable = (confidence_features.asr_conf_segment < 0.3 and 
                          confidence_features.lid_entropy > 2.0)
        
        if audio_unreliable and text_unreliable:
            # Use equal weighting as fallback
            clamped_audio_weight = torch.ones_like(clamped_audio_weight) * 0.5
            clamped_text_weight = torch.ones_like(clamped_text_weight) * 0.5
        
        # Ensure weights sum to 1.0
        total_weight = clamped_audio_weight + clamped_text_weight
        clamped_audio_weight = clamped_audio_weight / total_weight
        clamped_text_weight = clamped_text_weight / total_weight
        
        return clamped_audio_weight, clamped_text_weight


class AdaptiveFusionLayer(nn.Module):
    """
    Adaptive Fusion Layer with confidence-aware dynamic weighting.
    
    Combines audio and text features with weights determined by input quality
    and confidence measures.
    """
    
    def __init__(self, audio_dim: int, text_dim: int, proj_dim: int = 256):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.proj_dim = proj_dim
        
        # Feature projections
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Dynamic gating MLP
        self.dynamic_gating = DynamicGatingMLP(confidence_dim=14)
        
        # Policy-based clamps
        self.policy_clamps = PolicyBasedClamps()
        
        # Confidence feature projection for fusion
        self.confidence_projection = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, proj_dim // 4)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(proj_dim + proj_dim // 4, proj_dim),  # features + confidence
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Fusion confidence estimation
        self.fusion_confidence_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        logging.info(f"Adaptive Fusion Layer initialized: {audio_dim} + {text_dim} → {proj_dim}")
    
    def forward(self, audio_vec: torch.Tensor, text_vec: torch.Tensor, 
                confidence_features: ConfidenceFeatures) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Adaptive fusion with confidence-aware weighting.
        
        Args:
            audio_vec: [batch_size, audio_dim] - audio features
            text_vec: [batch_size, text_dim] - text features
            confidence_features: Confidence features for dynamic weighting
            
        Returns:
            fused_features: [batch_size, proj_dim] - fused features
            fusion_confidence: [batch_size, 1] - overall fusion confidence
            fusion_info: Dict with fusion details
        """
        # Project features to common dimension
        audio_proj = self.audio_projection(audio_vec)  # [batch_size, proj_dim]
        text_proj = self.text_projection(text_vec)     # [batch_size, proj_dim]
        
        # Compute dynamic weights from confidence features
        conf_tensor = confidence_features.to_tensor()
        if conf_tensor.dim() == 1:
            conf_tensor = conf_tensor.unsqueeze(0)
        
        # Get initial weights from gating MLP
        audio_weight, text_weight = self.dynamic_gating(conf_tensor)
        
        # Apply policy-based clamps
        clamped_audio_weight, clamped_text_weight = self.policy_clamps(
            audio_weight, text_weight, confidence_features
        )
        
        # Weighted combination
        weighted_audio = clamped_audio_weight * audio_proj
        weighted_text = clamped_text_weight * text_proj
        
        # Add residual connection
        fused_features = weighted_audio + weighted_text
        
        # Add confidence features as auxiliary input
        confidence_proj = self.confidence_projection(conf_tensor)
        fused_with_conf = torch.cat([fused_features, confidence_proj], dim=-1)
        
        # Final fusion
        fused_features = self.fusion_layer(fused_with_conf)
        
        # Estimate fusion confidence
        fusion_confidence = self.fusion_confidence_head(fused_features)
        
        # Prepare fusion information
        fusion_info = {
            'audio_weight': clamped_audio_weight.detach(),
            'text_weight': clamped_text_weight.detach(),
            'raw_audio_weight': audio_weight.detach(),
            'raw_text_weight': text_weight.detach(),
            'confidence_features': conf_tensor.detach(),
            'fusion_confidence': fusion_confidence.detach()
        }
        
        return fused_features, fusion_confidence, fusion_info
    
    def get_fusion_report(self, fusion_info: Dict) -> str:
        """Generate a human-readable fusion report."""
        # Handle tensor indexing properly
        conf_features = fusion_info['confidence_features']
        if conf_features.dim() == 2:
            conf_features = conf_features.squeeze(0)  # Remove batch dimension
        
        report = f"""
Adaptive Fusion Report:
======================
Modality Weights:
  Audio Weight: {fusion_info['audio_weight'].item():.3f}
  Text Weight: {fusion_info['text_weight'].item():.3f}
  Raw Audio Weight: {fusion_info['raw_audio_weight'].item():.3f}
  Raw Text Weight: {fusion_info['raw_text_weight'].item():.3f}

Fusion Confidence: {fusion_info['fusion_confidence'].item():.3f}

Confidence Features:
  SNR: {conf_features[0].item() * 50:.1f} dB
  Speech Probability: {conf_features[1].item():.3f}
  ASR Confidence: {conf_features[4].item():.3f}
  LID Entropy: {conf_features[5].item() * 2:.3f}
  Text Reliability: {conf_features[6].item():.3f}
"""
        return report


class ConfidenceAwareFusion(nn.Module):
    """
    Complete confidence-aware fusion system.
    
    Integrates all components: dynamic gating, policy clamps, and adaptive fusion.
    """
    
    def __init__(self, audio_dim: int, text_dim: int, proj_dim: int = 256):
        super().__init__()
        
        self.adaptive_fusion = AdaptiveFusionLayer(audio_dim, text_dim, proj_dim)
        
        # Feature dimension tracking
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.proj_dim = proj_dim
        
        logging.info(f"Confidence-Aware Fusion initialized: {audio_dim} + {text_dim} → {proj_dim}")
    
    def forward(self, audio_vec: torch.Tensor, text_vec: torch.Tensor, 
                confidence_features: ConfidenceFeatures) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Complete confidence-aware fusion pipeline.
        
        Args:
            audio_vec: [batch_size, audio_dim] - audio features
            text_vec: [batch_size, text_dim] - text features
            confidence_features: Confidence features for dynamic weighting
            
        Returns:
            fused_features: [batch_size, proj_dim] - fused features
            fusion_confidence: [batch_size, 1] - overall fusion confidence
            fusion_info: Dict with complete fusion details
        """
        return self.adaptive_fusion(audio_vec, text_vec, confidence_features)
    
    def get_fusion_report(self, fusion_info: Dict) -> str:
        """Get detailed fusion report."""
        return self.adaptive_fusion.get_fusion_report(fusion_info)


# Utility function for easy integration
def create_confidence_aware_fusion(audio_dim: int, text_dim: int, 
                                 proj_dim: int = 256) -> ConfidenceAwareFusion:
    """Factory function to create confidence-aware fusion with default settings."""
    return ConfidenceAwareFusion(audio_dim, text_dim, proj_dim)
