import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging


@dataclass
class LanguageInfo:
    """Container for language-related information."""
    language_id: int
    language_name: str
    confidence: float
    is_primary: bool = True


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    
    During forward pass: identity function
    During backward pass: negates gradients
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for adversarial training.
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class LanguageAdversarialHead(nn.Module):
    """
    Language Adversarial Head for learning language-invariant features.
    
    Architecture: 256 → 128 → num_languages
    Objective: Predict language (but gradients are reversed to learn invariant features)
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, num_languages: int = 7):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_languages = num_languages
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer(alpha=1.0)
        
        # Language classifier
        self.language_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_languages)
        )
        
        # Language mapping
        self.language_names = [
            "English", "Spanish", "French", "German", "Italian", "Portuguese", "Other"
        ]
        
        logging.info(f"Language Adversarial Head initialized: {input_dim} → {num_languages} languages")
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradient reversal.
        
        Args:
            features: [batch_size, input_dim] - fused features
            
        Returns:
            language_logits: [batch_size, num_languages] - language predictions
            language_probs: [batch_size, num_languages] - language probabilities
        """
        # Apply gradient reversal
        reversed_features = self.gradient_reversal(features)
        
        # Language classification
        language_logits = self.language_classifier(reversed_features)
        language_probs = F.softmax(language_logits, dim=-1)
        
        return language_logits, language_probs
    
    def get_language_prediction(self, language_probs: torch.Tensor) -> List[LanguageInfo]:
        """
        Convert language probabilities to language predictions.
        
        Args:
            language_probs: [batch_size, num_languages] - language probabilities
            
        Returns:
            List of LanguageInfo objects
        """
        predictions = []
        
        for i in range(language_probs.shape[0]):
            probs = language_probs[i]
            language_id = torch.argmax(probs).item()
            confidence = probs[language_id].item()
            language_name = self.language_names[language_id]
            
            predictions.append(LanguageInfo(
                language_id=language_id,
                language_name=language_name,
                confidence=confidence
            ))
        
        return predictions


class AdapterLayer(nn.Module):
    """
    Adapter Layer for fine-tuning pretrained encoders.
    
    Architecture: layer_norm → down_project → ReLU → up_project → residual
    """
    
    def __init__(self, hidden_size: int, adapter_size: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        # Adapter components
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize adapter weights
        self._init_adapter_weights()
        
    def _init_adapter_weights(self):
        """Initialize adapter weights to be close to identity."""
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        
        nn.init.xavier_uniform_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter layer.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - input features
            
        Returns:
            [batch_size, seq_len, hidden_size] - adapted features
        """
        # Layer normalization
        normalized = self.layer_norm(hidden_states)
        
        # Adapter transformation
        down_projected = self.down_project(normalized)
        activated = F.relu(down_projected)
        up_projected = self.up_project(activated)
        
        # Dropout and residual connection
        adapted = self.dropout(up_projected)
        output = hidden_states + adapted
        
        return output


class AdapterTunedAudioEncoder(nn.Module):
    """
    Adapter-tuned version of the audio encoder.
    
    Inserts adapter layers every 3 transformer layers.
    """
    
    def __init__(self, base_encoder, adapter_size: int = 64, adapter_frequency: int = 3):
        super().__init__()
        
        self.base_encoder = base_encoder
        self.adapter_size = adapter_size
        self.adapter_frequency = adapter_frequency
        
        # Freeze base encoder parameters
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        
        # Add adapter layers
        self.adapters = nn.ModuleList()
        self._add_adapters()
        
        logging.info(f"Adapter-tuned Audio Encoder initialized with {len(self.adapters)} adapters")
    
    def _add_adapters(self):
        """Add adapter layers to the base encoder."""
        # This is a simplified implementation
        # In practice, you would need to modify the transformer layers directly
        # For now, we'll add adapters as separate layers
        
        # Get the number of layers from the base encoder
        if hasattr(self.base_encoder, 'encoder') and hasattr(self.base_encoder.encoder, 'layers'):
            num_layers = len(self.base_encoder.encoder.layers)
        else:
            # Fallback: assume 12 layers (typical for base models)
            num_layers = 12
        
        # Add adapters every adapter_frequency layers
        for i in range(0, num_layers, self.adapter_frequency):
            adapter = AdapterLayer(
                hidden_size=768,  # Typical hidden size for base models
                adapter_size=self.adapter_size
            )
            self.adapters.append(adapter)
    
    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through adapter-tuned encoder.
        
        Args:
            input_values: [batch_size, seq_len] - input audio features
            attention_mask: [batch_size, seq_len] - attention mask
            
        Returns:
            [batch_size, seq_len, hidden_size] - encoded features
        """
        # Get base encoder output
        outputs = self.base_encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply adapters
        adapter_idx = 0
        for i in range(0, hidden_states.shape[1], self.adapter_frequency):
            if adapter_idx < len(self.adapters):
                # Apply adapter to this layer's output
                # Note: This is a simplified approach
                # In practice, adapters would be integrated into the transformer layers
                hidden_states = self.adapters[adapter_idx](hidden_states)
                adapter_idx += 1
        
        return hidden_states


class AdapterTunedTextEncoder(nn.Module):
    """
    Adapter-tuned version of the text encoder.
    
    Similar to audio encoder but for text features.
    """
    
    def __init__(self, base_encoder, adapter_size: int = 64, adapter_frequency: int = 3):
        super().__init__()
        
        self.base_encoder = base_encoder
        self.adapter_size = adapter_size
        self.adapter_frequency = adapter_frequency
        
        # Freeze base encoder parameters
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        
        # Add adapter layers
        self.adapters = nn.ModuleList()
        self._add_adapters()
        
        logging.info(f"Adapter-tuned Text Encoder initialized with {len(self.adapters)} adapters")
    
    def _add_adapters(self):
        """Add adapter layers to the base encoder."""
        # Similar to audio encoder
        if hasattr(self.base_encoder, 'encoder') and hasattr(self.base_encoder.encoder, 'layers'):
            num_layers = len(self.base_encoder.encoder.layers)
        else:
            num_layers = 12
        
        for i in range(0, num_layers, self.adapter_frequency):
            adapter = AdapterLayer(
                hidden_size=768,
                adapter_size=self.adapter_size
            )
            self.adapters.append(adapter)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through adapter-tuned encoder.
        
        Args:
            input_ids: [batch_size, seq_len] - input token ids
            attention_mask: [batch_size, seq_len] - attention mask
            
        Returns:
            [batch_size, seq_len, hidden_size] - encoded features
        """
        # Get base encoder output
        outputs = self.base_encoder(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply adapters
        adapter_idx = 0
        for i in range(0, hidden_states.shape[1], self.adapter_frequency):
            if adapter_idx < len(self.adapters):
                hidden_states = self.adapters[adapter_idx](hidden_states)
                adapter_idx += 1
        
        return hidden_states


class CrossLingualConsistencyLoss(nn.Module):
    """
    Cross-Lingual Consistency Loss for enforcing similar embeddings
    across languages for the same emotion.
    """
    
    def __init__(self, temperature: float = 0.1, weight: float = 0.05):
        super().__init__()
        
        self.temperature = temperature
        self.weight = weight
        
        logging.info(f"Cross-Lingual Consistency Loss initialized with weight={weight}")
    
    def forward(self, embeddings: torch.Tensor, emotion_labels: torch.Tensor, 
                language_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-lingual consistency loss.
        
        Args:
            embeddings: [batch_size, embedding_dim] - emotion embeddings
            emotion_labels: [batch_size] - emotion labels
            language_ids: [batch_size] - language identifiers
            
        Returns:
            consistency_loss: scalar tensor
        """
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / self.temperature
        
        # Create emotion consistency mask
        emotion_mask = (emotion_labels.unsqueeze(1) == emotion_labels.unsqueeze(0)).float()
        
        # Create language diversity mask (different languages)
        language_mask = (language_ids.unsqueeze(1) != language_ids.unsqueeze(0)).float()
        
        # Combined mask: same emotion, different languages
        consistency_mask = emotion_mask * language_mask
        
        # Remove self-similarity
        consistency_mask.fill_diagonal_(0)
        
        # Compute consistency loss
        if consistency_mask.sum() > 0:
            # Get similarities for consistent pairs
            consistent_similarities = similarity_matrix[consistency_mask.bool()]
            
            # Loss: maximize similarity for same emotion across languages
            # Use MSE loss to ensure non-negative values
            target_similarity = torch.ones_like(consistent_similarities)
            consistency_loss = F.mse_loss(consistent_similarities, target_similarity)
        else:
            consistency_loss = torch.tensor(0.0, device=embeddings.device)
        
        return self.weight * consistency_loss


class CrossLingualVarianceHandler(nn.Module):
    """
    Complete Cross-Lingual Variance Handler.
    
    Integrates language-adversarial training, adapter-based tuning,
    and cross-lingual consistency loss.
    """
    
    def __init__(self, 
                 audio_encoder, 
                 text_encoder, 
                 fusion_layer,
                 num_languages: int = 7,
                 adapter_size: int = 64,
                 consistency_weight: float = 0.05):
        super().__init__()
        
        # Adapter-tuned encoders
        self.audio_encoder = AdapterTunedAudioEncoder(audio_encoder, adapter_size)
        self.text_encoder = AdapterTunedTextEncoder(text_encoder, adapter_size)
        
        # Fusion layer
        self.fusion_layer = fusion_layer
        
        # Language adversarial head
        self.language_adversarial = LanguageAdversarialHead(
            input_dim=256,  # Assuming fusion output dimension
            num_languages=num_languages
        )
        
        # Cross-lingual consistency loss
        self.consistency_loss = CrossLingualConsistencyLoss(weight=consistency_weight)
        
        # Language mapping
        self.language_names = self.language_adversarial.language_names
        
        logging.info(f"Cross-Lingual Variance Handler initialized with {num_languages} languages")
    
    def forward(self, 
                audio_input: torch.Tensor,
                text_input: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                language_ids: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass through cross-lingual variance handler.
        
        Args:
            audio_input: [batch_size, seq_len] - audio features
            text_input: [batch_size, seq_len] - text tokens
            attention_mask: [batch_size, seq_len] - attention mask
            language_ids: [batch_size] - language identifiers
            
        Returns:
            Dict containing all outputs and intermediate results
        """
        # Encode with adapter-tuned encoders
        audio_features = self.audio_encoder(audio_input, attention_mask)
        text_features = self.text_encoder(text_input, attention_mask)
        
        # Fusion (assuming fusion_layer expects specific input format)
        # This would need to be adapted based on your actual fusion layer
        fused_features = self.fusion_layer(audio_features, text_features)
        
        # Language adversarial prediction
        language_logits, language_probs = self.language_adversarial(fused_features)
        
        # Language predictions
        language_predictions = self.language_adversarial.get_language_prediction(language_probs)
        
        # Compute consistency loss if language_ids provided
        consistency_loss = torch.tensor(0.0, device=fused_features.device)
        if language_ids is not None:
            # This would need emotion_labels from your dataset
            # For now, we'll skip the consistency loss computation
            pass
        
        return {
            'audio_features': audio_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'language_logits': language_logits,
            'language_probs': language_probs,
            'language_predictions': language_predictions,
            'consistency_loss': consistency_loss
        }
    
    def compute_losses(self, 
                      emotion_logits: torch.Tensor,
                      emotion_labels: torch.Tensor,
                      language_logits: torch.Tensor,
                      language_labels: torch.Tensor,
                      consistency_loss: torch.Tensor,
                      lambda_adversarial: float = 0.1) -> Dict:
        """
        Compute all losses for cross-lingual training.
        
        Args:
            emotion_logits: [batch_size, num_emotions] - emotion predictions
            emotion_labels: [batch_size] - emotion ground truth
            language_logits: [batch_size, num_languages] - language predictions
            language_labels: [batch_size] - language ground truth
            consistency_loss: scalar - cross-lingual consistency loss
            lambda_adversarial: float - weight for adversarial loss
            
        Returns:
            Dict containing all losses
        """
        # Emotion classification loss
        emotion_loss = F.cross_entropy(emotion_logits, emotion_labels)
        
        # Language adversarial loss (with gradient reversal)
        language_loss = F.cross_entropy(language_logits, language_labels)
        
        # Total loss: emotion_loss - λ * language_loss + consistency_loss
        total_loss = emotion_loss - lambda_adversarial * language_loss + consistency_loss
        
        return {
            'emotion_loss': emotion_loss,
            'language_loss': language_loss,
            'consistency_loss': consistency_loss,
            'total_loss': total_loss
        }
    
    def get_language_statistics(self, language_predictions: List[LanguageInfo]) -> Dict:
        """Get statistics about language predictions."""
        if not language_predictions:
            return {}
        
        language_counts = {}
        total_confidence = 0.0
        
        for pred in language_predictions:
            lang_name = pred.language_name
            if lang_name not in language_counts:
                language_counts[lang_name] = {'count': 0, 'confidence': 0.0}
            
            language_counts[lang_name]['count'] += 1
            language_counts[lang_name]['confidence'] += pred.confidence
            total_confidence += pred.confidence
        
        # Normalize confidences
        for lang_name in language_counts:
            language_counts[lang_name]['avg_confidence'] = (
                language_counts[lang_name]['confidence'] / language_counts[lang_name]['count']
            )
        
        return {
            'language_distribution': language_counts,
            'total_samples': len(language_predictions),
            'avg_confidence': total_confidence / len(language_predictions)
        }


# Utility functions
def create_cross_lingual_variance_handler(audio_encoder, text_encoder, fusion_layer,
                                        num_languages: int = 7) -> CrossLingualVarianceHandler:
    """Factory function to create cross-lingual variance handler."""
    return CrossLingualVarianceHandler(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        fusion_layer=fusion_layer,
        num_languages=num_languages
    )


def get_adapter_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only adapter parameters for training."""
    adapter_params = []
    for name, param in model.named_parameters():
        if 'adapter' in name or 'adapters' in name:
            adapter_params.append(param)
    return adapter_params


def freeze_base_parameters(model: nn.Module):
    """Freeze base encoder parameters, keep only adapters trainable."""
    for name, param in model.named_parameters():
        if 'adapter' not in name and 'adapters' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
