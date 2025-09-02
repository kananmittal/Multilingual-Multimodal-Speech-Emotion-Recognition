import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Deque
from dataclasses import dataclass
from collections import deque
import logging


@dataclass
class TemporalSegment:
    """Container for temporal segment information."""
    segment_id: int
    start_time: float
    end_time: float
    features: torch.Tensor  # [256] - emotion features
    confidence: float
    emotion_prediction: torch.Tensor  # [num_emotions] - emotion logits
    speaker_embedding: Optional[torch.Tensor] = None  # [speaker_dim] - speaker features


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal Positional Encoding for segment order.
    
    Adds position information to temporal features.
    """
    
    def __init__(self, feature_dim: int = 256, max_segments: int = 10):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.max_segments = max_segments
        
        # Create positional encoding matrix
        pe = torch.zeros(max_segments, feature_dim)
        position = torch.arange(0, max_segments, dtype=torch.float).unsqueeze(1)
        
        # Use different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * 
                           (-np.log(10000.0) / feature_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
        logging.info(f"Temporal Positional Encoding initialized: {max_segments} segments, {feature_dim} features")
    
    def forward(self, x: torch.Tensor, segment_positions: torch.Tensor) -> torch.Tensor:
        """
        Add temporal positional encoding to features.
        
        Args:
            x: [batch_size, num_segments, feature_dim] - temporal features
            segment_positions: [batch_size, num_segments] - segment positions (0, 1, 2, ...)
            
        Returns:
            [batch_size, num_segments, feature_dim] - features with positional encoding
        """
        batch_size, num_segments, feature_dim = x.shape
        
        # Get positional encodings for each segment position
        pos_encodings = self.pe[segment_positions]  # [batch_size, num_segments, feature_dim]
        
        # Add positional encoding to features
        return x + pos_encodings


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution for temporal modeling.
    
    Ensures no information leakage from future segments.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Causal padding: (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        # Normalization and activation
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize convolution weights."""
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal convolution.
        
        Args:
            x: [batch_size, in_channels, seq_len] - input features
            
        Returns:
            [batch_size, out_channels, seq_len] - convolved features
        """
        # Apply causal convolution
        conv_out = self.conv(x)
        
        # Remove padding from the end to maintain causal property
        if self.padding > 0:
            conv_out = conv_out[:, :, :-self.padding]
        
        # Transpose for layer normalization
        conv_out = conv_out.transpose(1, 2)  # [batch_size, seq_len, out_channels]
        
        # Apply normalization and dropout
        conv_out = self.norm(conv_out)
        conv_out = self.dropout(conv_out)
        
        # Transpose back
        conv_out = conv_out.transpose(1, 2)  # [batch_size, out_channels, seq_len]
        
        return conv_out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for emotion dynamics modeling.
    
    Architecture: 2-layer TCN with causal convolutions and residual connections.
    """
    
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        # First TCN layer: feature_dim → hidden_dim
        self.tcn_layer1 = CausalConv1d(
            in_channels=feature_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            dropout=dropout
        )
        
        # Second TCN layer: hidden_dim → feature_dim
        self.tcn_layer2 = CausalConv1d(
            in_channels=hidden_dim,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            dilation=2,  # Larger receptive field
            dropout=dropout
        )
        
        # Residual connection projection (if dimensions don't match)
        self.residual_proj = nn.Linear(feature_dim, feature_dim) if feature_dim != hidden_dim else nn.Identity()
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(feature_dim)
        
        logging.info(f"Temporal ConvNet initialized: {feature_dim} → {hidden_dim} → {feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: [batch_size, num_segments, feature_dim] - temporal features
            
        Returns:
            [batch_size, num_segments, feature_dim] - temporally processed features
        """
        batch_size, num_segments, feature_dim = x.shape
        
        # Transpose for convolution: [batch_size, feature_dim, num_segments]
        x_conv = x.transpose(1, 2)
        
        # First TCN layer
        h1 = self.tcn_layer1(x_conv)
        
        # Second TCN layer
        h2 = self.tcn_layer2(h1)
        
        # Transpose back: [batch_size, num_segments, feature_dim]
        h2 = h2.transpose(1, 2)
        
        # Residual connection
        residual = self.residual_proj(x)
        output = h2 + residual
        
        # Final normalization
        output = self.final_norm(output)
        
        return output


class ConfidenceAwareSmoothing(nn.Module):
    """
    Confidence-Aware Temporal Smoothing for emotion predictions.
    
    Smooths predictions based on current and historical confidence scores.
    """
    
    def __init__(self, smoothing_threshold: float = 0.9, min_confidence: float = 0.3):
        super().__init__()
        
        self.smoothing_threshold = smoothing_threshold
        self.min_confidence = min_confidence
        
        logging.info(f"Confidence-Aware Smoothing initialized: threshold={smoothing_threshold}")
    
    def forward(self, current_pred: torch.Tensor, current_conf: torch.Tensor,
                temporal_pred: torch.Tensor, temporal_conf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply confidence-aware temporal smoothing.
        
        Args:
            current_pred: [batch_size, num_emotions] - current segment prediction
            current_conf: [batch_size, 1] - current segment confidence
            temporal_pred: [batch_size, num_emotions] - temporally smoothed prediction
            temporal_conf: [batch_size, 1] - temporal average confidence
            
        Returns:
            smoothed_pred: [batch_size, num_emotions] - smoothed prediction
            final_confidence: [batch_size, 1] - final confidence score
        """
        batch_size = current_pred.shape[0]
        
        # Ensure confidence is in [0, 1]
        current_conf = torch.clamp(current_conf, 0.0, 1.0)
        temporal_conf = torch.clamp(temporal_conf, 0.0, 1.0)
        
        # Compute smoothing factor
        # α = current_confidence / (current_confidence + temporal_avg_confidence)
        smoothing_factor = current_conf / (current_conf + temporal_conf + 1e-8)
        
        # Prevent sudden emotion jumps unless confidence is very high
        high_confidence_mask = current_conf > self.smoothing_threshold
        
        # Apply smoothing
        # If high confidence: use current prediction
        # Otherwise: use weighted combination
        smoothed_pred = torch.where(
            high_confidence_mask,
            current_pred,
            smoothing_factor * current_pred + (1 - smoothing_factor) * temporal_pred
        )
        
        # Compute final confidence
        final_confidence = torch.max(current_conf, temporal_conf)
        
        # Apply minimum confidence threshold
        low_confidence_mask = final_confidence < self.min_confidence
        final_confidence = torch.where(low_confidence_mask, 
                                     torch.ones_like(final_confidence) * self.min_confidence,
                                     final_confidence)
        
        return smoothed_pred, final_confidence
    
    def get_smoothing_info(self, current_conf: torch.Tensor, temporal_conf: torch.Tensor) -> Dict:
        """Get smoothing information for analysis."""
        smoothing_factor = current_conf / (current_conf + temporal_conf + 1e-8)
        high_confidence_ratio = (current_conf > self.smoothing_threshold).float().mean()
        
        return {
            'smoothing_factor': smoothing_factor.mean().item(),
            'high_confidence_ratio': high_confidence_ratio.item(),
            'current_confidence': current_conf.mean().item(),
            'temporal_confidence': temporal_conf.mean().item()
        }


class SpeakerChangeDetector(nn.Module):
    """
    Speaker Change Detection for temporal modeling.
    
    Monitors speaker embedding consistency across segments.
    """
    
    def __init__(self, speaker_dim: int = 128, similarity_threshold: float = 0.7):
        super().__init__()
        
        self.speaker_dim = speaker_dim
        self.similarity_threshold = similarity_threshold
        
        # Speaker embedding projection (if needed)
        self.speaker_projection = nn.Linear(speaker_dim, speaker_dim)
        
        logging.info(f"Speaker Change Detector initialized: dim={speaker_dim}, threshold={similarity_threshold}")
    
    def forward(self, speaker_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect speaker changes based on embedding similarity.
        
        Args:
            speaker_embeddings: [batch_size, num_segments, speaker_dim] - speaker embeddings
            
        Returns:
            speaker_changes: [batch_size, num_segments] - boolean mask for speaker changes
            similarities: [batch_size, num_segments-1] - cosine similarities
        """
        batch_size, num_segments, speaker_dim = speaker_embeddings.shape
        
        if num_segments < 2:
            # No speaker changes possible with single segment
            return torch.zeros(batch_size, num_segments, dtype=torch.bool, device=speaker_embeddings.device), \
                   torch.zeros(batch_size, 0, device=speaker_embeddings.device)
        
        # Project speaker embeddings
        projected_embeddings = self.speaker_projection(speaker_embeddings)
        
        # Normalize embeddings
        normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        
        # Compute cosine similarities between consecutive segments
        similarities = torch.sum(
            normalized_embeddings[:, :-1] * normalized_embeddings[:, 1:],
            dim=-1
        )  # [batch_size, num_segments-1]
        
        # Detect speaker changes (similarity below threshold)
        speaker_changes = similarities < self.similarity_threshold  # [batch_size, num_segments-1]
        
        # Pad to match segment count (first segment has no change)
        padded_changes = torch.zeros(batch_size, num_segments, dtype=torch.bool, device=speaker_embeddings.device)
        padded_changes[:, 1:] = speaker_changes
        
        return padded_changes, similarities
    
    def get_speaker_info(self, speaker_changes: torch.Tensor, similarities: torch.Tensor) -> Dict:
        """Get speaker change information for analysis."""
        if similarities.numel() == 0:
            return {'num_changes': 0, 'avg_similarity': 1.0, 'change_ratio': 0.0}
        
        num_changes = speaker_changes.sum().item()
        avg_similarity = similarities.mean().item()
        change_ratio = num_changes / similarities.numel()
        
        return {
            'num_changes': num_changes,
            'avg_similarity': avg_similarity,
            'change_ratio': change_ratio
        }


class TemporalBuffer:
    """
    Temporal Buffer for maintaining context across segments.
    
    Manages sliding window of recent segments for temporal modeling.
    """
    
    def __init__(self, max_segments: int = 3, feature_dim: int = 256):
        super().__init__()
        
        self.max_segments = max_segments
        self.feature_dim = feature_dim
        
        # Circular buffer for segments
        self.segments: Deque[TemporalSegment] = deque(maxlen=max_segments)
        
        logging.info(f"Temporal Buffer initialized: max_segments={max_segments}, feature_dim={feature_dim}")
    
    def add_segment(self, segment: TemporalSegment):
        """Add a new segment to the buffer."""
        self.segments.append(segment)
    
    def get_temporal_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get temporal features from buffer.
        
        Returns:
            features: [num_segments, feature_dim] - segment features
            confidences: [num_segments] - confidence scores
            predictions: [num_segments, num_emotions] - emotion predictions
        """
        if len(self.segments) == 0:
            return torch.empty(0, self.feature_dim), torch.empty(0), torch.empty(0, 0)
        
        # Extract features, confidences, and predictions
        features = torch.stack([seg.features for seg in self.segments])
        confidences = torch.tensor([seg.confidence for seg in self.segments])
        predictions = torch.stack([seg.emotion_prediction for seg in self.segments])
        
        return features, confidences, predictions
    
    def get_speaker_embeddings(self) -> Optional[torch.Tensor]:
        """Get speaker embeddings from buffer."""
        speaker_embeddings = [seg.speaker_embedding for seg in self.segments if seg.speaker_embedding is not None]
        
        if not speaker_embeddings:
            return None
        
        return torch.stack(speaker_embeddings)
    
    def clear(self):
        """Clear the temporal buffer."""
        self.segments.clear()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.segments) == self.max_segments
    
    def get_buffer_info(self) -> Dict:
        """Get buffer information."""
        return {
            'num_segments': len(self.segments),
            'is_full': self.is_full(),
            'segment_ids': [seg.segment_id for seg in self.segments],
            'time_range': (self.segments[0].start_time, self.segments[-1].end_time) if self.segments else (0, 0)
        }


class TemporalModelingModule(nn.Module):
    """
    Complete Temporal Modeling Module.
    
    Integrates TCN, confidence-aware smoothing, and speaker change detection.
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 hidden_dim: int = 128,
                 max_segments: int = 3,
                 speaker_dim: int = 128,
                 num_emotions: int = 4,
                 smoothing_threshold: float = 0.9,
                 speaker_threshold: float = 0.7):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_segments = max_segments
        self.speaker_dim = speaker_dim
        self.num_emotions = num_emotions
        
        # Temporal components
        self.temporal_encoding = TemporalPositionalEncoding(feature_dim, max_segments)
        self.temporal_tcn = TemporalConvNet(feature_dim, hidden_dim)
        self.confidence_smoothing = ConfidenceAwareSmoothing(smoothing_threshold)
        self.speaker_detector = SpeakerChangeDetector(speaker_dim, speaker_threshold)
        
        # Temporal buffer
        self.temporal_buffer = TemporalBuffer(max_segments, feature_dim)
        
        # Emotion prediction head
        self.emotion_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_emotions)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logging.info(f"Temporal Modeling Module initialized: {feature_dim} → {hidden_dim} → {num_emotions}")
    
    def forward(self, 
                current_features: torch.Tensor,
                current_speaker_embedding: Optional[torch.Tensor] = None,
                current_confidence: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass through temporal modeling module.
        
        Args:
            current_features: [batch_size, feature_dim] - current segment features
            current_speaker_embedding: [batch_size, speaker_dim] - current speaker embedding
            current_confidence: [batch_size, 1] - current segment confidence
            
        Returns:
            Dict containing temporal predictions and information
        """
        batch_size = current_features.shape[0]
        
        # Get temporal context from buffer
        temporal_features, temporal_confidences, temporal_predictions = self.temporal_buffer.get_temporal_features()
        
        if len(temporal_features) == 0:
            # No temporal context available
            current_pred = self.emotion_head(current_features)
            current_conf = self.confidence_head(current_features)
            
            return {
                'emotion_prediction': current_pred,
                'confidence': current_conf,
                'temporal_context_available': False,
                'speaker_change_detected': torch.zeros(batch_size, dtype=torch.bool, device=current_features.device),
                'smoothing_applied': False,
                'temporal_features': torch.empty(0),
                'speaker_similarities': torch.empty(0),
                'buffer_info': self.temporal_buffer.get_buffer_info()
            }
        
        # Prepare temporal input
        # Concatenate temporal features with current features
        # temporal_features: [num_segments, feature_dim]
        # current_features: [batch_size, feature_dim]
        # We need to expand temporal_features to match batch_size
        temporal_features_expanded = temporal_features.unsqueeze(0).expand(batch_size, -1, -1)
        current_features_expanded = current_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        all_features = torch.cat([temporal_features_expanded, current_features_expanded], dim=1)
        num_segments = all_features.shape[1]
        
        # Add positional encoding
        # Ensure positions are within bounds
        segment_positions = torch.arange(num_segments, device=current_features.device)
        segment_positions = torch.clamp(segment_positions, 0, self.max_segments - 1)
        segment_positions = segment_positions.unsqueeze(0).expand(batch_size, -1)
        
        temporal_input = self.temporal_encoding(all_features, segment_positions)
        
        # Apply temporal TCN
        temporal_output = self.temporal_tcn(temporal_input)
        
        # Get current segment output (last segment)
        current_temporal_features = temporal_output[:, -1, :]  # [batch_size, feature_dim]
        
        # Current segment predictions
        current_pred = self.emotion_head(current_temporal_features)
        current_conf = self.confidence_head(current_temporal_features)
        
        # Speaker change detection
        speaker_changes = torch.zeros(batch_size, 1, dtype=torch.bool, device=current_features.device)
        speaker_similarities = torch.empty(batch_size, 0, device=current_features.device)
        
        if current_speaker_embedding is not None:
            speaker_embeddings = self.temporal_buffer.get_speaker_embeddings()
            if speaker_embeddings is not None:
                # Add current speaker embedding
                # speaker_embeddings: [num_segments, speaker_dim]
                # current_speaker_embedding: [batch_size, speaker_dim]
                speaker_embeddings_expanded = speaker_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                current_speaker_expanded = current_speaker_embedding.unsqueeze(1)  # [batch_size, 1, speaker_dim]
                
                all_speaker_embeddings = torch.cat([speaker_embeddings_expanded, current_speaker_expanded], dim=1)
                
                speaker_changes, speaker_similarities = self.speaker_detector(all_speaker_embeddings)
                speaker_changes = speaker_changes[:, -1:]  # Only current segment
        
        # Confidence-aware smoothing
        if len(temporal_predictions) > 0:
            # Use temporal average for smoothing
            temporal_pred = temporal_predictions.mean(dim=0, keepdim=True).expand(batch_size, -1)
            temporal_conf = temporal_confidences.mean().unsqueeze(0).expand(batch_size, 1)
            
            smoothed_pred, final_confidence = self.confidence_smoothing(
                current_pred, current_conf, temporal_pred, temporal_conf
            )
            smoothing_applied = True
        else:
            smoothed_pred = current_pred
            final_confidence = current_conf
            smoothing_applied = False
        
        return {
            'emotion_prediction': smoothed_pred,
            'confidence': final_confidence,
            'temporal_context_available': True,
            'speaker_change_detected': speaker_changes.squeeze(-1),
            'smoothing_applied': smoothing_applied,
            'temporal_features': temporal_output,
            'speaker_similarities': speaker_similarities,
            'buffer_info': self.temporal_buffer.get_buffer_info()
        }
    
    def update_buffer(self, segment: TemporalSegment):
        """Update temporal buffer with new segment."""
        self.temporal_buffer.add_segment(segment)
    
    def reset_buffer(self):
        """Reset temporal buffer."""
        self.temporal_buffer.clear()
    
    def get_temporal_report(self, outputs: Dict) -> str:
        """Generate temporal modeling report."""
        report = f"""
Temporal Modeling Report:
========================
Temporal Context: {'Available' if outputs['temporal_context_available'] else 'Not Available'}
Speaker Change: {'Detected' if outputs['speaker_change_detected'].any() else 'Not Detected'}
Smoothing Applied: {'Yes' if outputs['smoothing_applied'] else 'No'}

Buffer Information:
  {outputs['buffer_info']}

Confidence: {outputs['confidence'].mean().item():.3f}
"""
        return report


# Utility functions
def create_temporal_modeling_module(feature_dim: int = 256,
                                  hidden_dim: int = 128,
                                  max_segments: int = 3,
                                  num_emotions: int = 4) -> TemporalModelingModule:
    """Factory function to create temporal modeling module."""
    return TemporalModelingModule(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        max_segments=max_segments,
        num_emotions=num_emotions
    )


def create_temporal_segment(segment_id: int, start_time: float, end_time: float,
                          features: torch.Tensor, confidence: float,
                          emotion_prediction: torch.Tensor,
                          speaker_embedding: Optional[torch.Tensor] = None) -> TemporalSegment:
    """Factory function to create temporal segment."""
    return TemporalSegment(
        segment_id=segment_id,
        start_time=start_time,
        end_time=end_time,
        features=features,
        confidence=confidence,
        emotion_prediction=emotion_prediction,
        speaker_embedding=speaker_embedding
    )
