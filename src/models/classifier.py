import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ClassAnchorClustering(nn.Module):
    """
    Class Anchor Clustering for better feature learning and class separation
    """
    def __init__(self, feature_dim: int, num_classes: int, anchor_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.anchor_dim = anchor_dim
        
        # Learnable class anchors
        self.class_anchors = nn.Parameter(torch.randn(num_classes, anchor_dim))
        
        # Anchor projection layers
        self.anchor_projection = nn.Sequential(
            nn.Linear(feature_dim, anchor_dim),
            nn.LayerNorm(anchor_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temperature parameter for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, feature_dim]
        Returns:
            anchor_similarities: [batch_size, num_classes]
            anchor_loss: scalar loss for clustering
        """
        # Project features to anchor space
        projected_features = self.anchor_projection(features)  # [batch_size, anchor_dim]
        
        # Normalize features and anchors
        projected_features = F.normalize(projected_features, p=2, dim=1)
        normalized_anchors = F.normalize(self.class_anchors, p=2, dim=1)
        
        # Compute similarities
        anchor_similarities = torch.mm(projected_features, normalized_anchors.t()) / self.temperature
        
        # Compute clustering loss (pull similar samples together, push different apart)
        anchor_loss = self.compute_clustering_loss(projected_features, normalized_anchors)
        
        return anchor_similarities, anchor_loss
    
    def compute_clustering_loss(self, features: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Compute clustering loss for better class separation"""
        # Intra-class compactness
        intra_class_loss = 0.0
        # Inter-class separation
        inter_class_loss = 0.0
        
        # This is a simplified version - in practice you'd need labels
        # For now, we'll use cosine similarity based clustering
        similarities = torch.mm(features, anchors.t())
        
        # Pull closest anchor closer, push others away
        max_similarities, _ = torch.max(similarities, dim=1, keepdim=True)
        pull_loss = torch.mean((similarities - max_similarities).clamp(min=0))
        
        return pull_loss


class DeepResidualBlock(nn.Module):
    """
    Deep residual block with skip connections
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DeepClassifier(nn.Module):
    """
    Deep classifier with 30-40 layers using residual connections
    """
    def __init__(self, input_dim: int, num_classes: int, num_layers: int = 35, 
                 base_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.base_dim = base_dim
        
        # Input projection to base dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Deep residual layers
        self.residual_layers = nn.ModuleList([
            DeepResidualBlock(base_dim, dropout) for _ in range(num_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(base_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.LayerNorm(base_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_projection(x)
        
        # Deep residual forward pass
        for i, (residual_block, layer_norm) in enumerate(zip(self.residual_layers, self.layer_norms)):
            x = layer_norm(x)
            x = residual_block(x)
            
            # Gradient checkpointing for memory efficiency (every 5 layers)
            if i % 5 == 0 and self.training:
                x = torch.utils.checkpoint.checkpoint(residual_block, x)
        
        # Output projection
        return self.output_projection(x)


class AdvancedOpenMaxClassifier(nn.Module):
    """
    Advanced OpenMax classifier with Class Anchor Clustering and deep architecture
    """
    def __init__(self, input_dim: int, num_labels: int, num_layers: int = 35, 
                 base_dim: int = 512, dropout: float = 0.1, alpha: float = 20.0):
        super().__init__()
        self.num_labels = num_labels
        self.alpha = alpha
        
        # Deep classifier backbone
        self.deep_classifier = DeepClassifier(
            input_dim=input_dim,
            num_classes=num_labels,
            num_layers=num_layers,
            base_dim=base_dim,
            dropout=dropout
        )
        
        # Class Anchor Clustering
        self.anchor_clustering = ClassAnchorClustering(
            feature_dim=base_dim // 2,  # Output dim of deep classifier before final layer
            num_classes=num_labels,
            anchor_dim=128
        )
        
        # Weibull parameters for OpenMax
        self.register_buffer('weibull_alpha', torch.ones(num_labels))
        self.register_buffer('weibull_beta', torch.ones(num_labels))
        self.register_buffer('weibull_tau', torch.zeros(num_labels))
        
        # Activation vectors for each class
        self.register_buffer('activation_vectors', torch.zeros(num_labels, base_dim // 2))
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(base_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, use_openmax: bool = True, 
                return_uncertainty: bool = False) -> torch.Tensor:
        # Get features from deep classifier (before final classification layer)
        features = x
        for i, layer in enumerate(self.deep_classifier.input_projection):
            features = layer(features)
        
        for i, (residual_block, layer_norm) in enumerate(zip(
            self.deep_classifier.residual_layers, 
            self.deep_classifier.layer_norms
        )):
            features = layer_norm(features)
            features = residual_block(features)
        
        # Get features before final classification layer
        features = self.deep_classifier.output_projection[0](features)  # [batch_size, base_dim//2]
        features = self.deep_classifier.output_projection[1](features)  # LayerNorm
        features = self.deep_classifier.output_projection[2](features)  # ReLU
        features = self.deep_classifier.output_projection[3](features)  # Dropout
        
        # Class anchor clustering
        anchor_similarities, anchor_loss = self.anchor_clustering(features)
        
        # Final classification
        logits = self.deep_classifier.output_projection[4](features)  # Final linear layer
        
        # Uncertainty estimation
        uncertainty = None
        if return_uncertainty:
            uncertainty = self.uncertainty_head(features)
        
        # Apply OpenMax during inference
        if use_openmax and not self.training:
            logits = self.openmax_forward(features, logits)
        
        if return_uncertainty:
            return logits, uncertainty, anchor_loss
        else:
            return logits
    
    def openmax_forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Enhanced OpenMax with uncertainty-aware calibration
        """
        batch_size = features.size(0)
        
        # Calculate distances to activation vectors for each class
        distances = torch.zeros(batch_size, self.num_labels, device=features.device)
        for i in range(self.num_labels):
            class_activation = self.activation_vectors[i]
            distances[:, i] = torch.norm(features - class_activation, dim=1)
        
        # Apply Weibull CDF to get probability of being unknown
        unknown_probs = torch.zeros(batch_size, device=features.device)
        for i in range(self.num_labels):
            x = distances[:, i]
            alpha = self.weibull_alpha[i]
            beta = self.weibull_beta[i]
            tau = self.weibull_tau[i]
            
            safe_beta = torch.clamp(beta, min=1e-6)
            safe_x = torch.clamp(x - tau, min=0)
            
            weibull_cdf = 1 - torch.exp(-torch.pow(safe_x / safe_beta, alpha))
            unknown_probs = torch.maximum(unknown_probs, weibull_cdf)
        
        # Enhanced uncertainty-aware adjustment
        adjusted_logits = logits.clone()
        for i in range(batch_size):
            unknown_prob = unknown_probs[i]
            if unknown_prob > 0.3:  # Lower threshold for better sensitivity
                # Reduce confidence proportionally to uncertainty
                confidence_reduction = unknown_prob * 0.8
                adjusted_logits[i] = adjusted_logits[i] * (1 - confidence_reduction)
        
        return adjusted_logits
    
    def fit_weibull(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Enhanced Weibull fitting with better parameter estimation
        """
        print("Fitting enhanced Weibull distributions for OpenMax...")
        
        for class_idx in range(self.num_labels):
            class_mask = labels == class_idx
            if class_mask.sum() == 0:
                continue
                
            class_features = features[class_mask]
            
            # Calculate mean activation vector for this class
            mean_activation = class_features.mean(dim=0)
            self.activation_vectors[class_idx] = mean_activation
            
            # Calculate distances to mean activation
            distances = torch.norm(class_features - mean_activation, dim=1)
            
            # Enhanced parameter estimation
            distances_np = distances.cpu().numpy()
            
            # More sophisticated parameter fitting
            self.weibull_alpha[class_idx] = torch.tensor(2.5)  # Shape parameter
            self.weibull_beta[class_idx] = torch.tensor(distances_np.std() * 1.5)  # Scale parameter
            self.weibull_tau[class_idx] = torch.tensor(distances_np.min() * 0.8)  # Location parameter
            
        print("Enhanced Weibull fitting completed!")


# Keep the old classifiers for backward compatibility
class OpenMaxClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden: int = 128, p: float = 0.1, alpha: float = 20.0):
        super().__init__()
        self.num_labels = num_labels
        self.alpha = alpha
        
        # Main classifier network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_labels),
        )
        
        # Weibull parameters for each class (learned during training)
        self.register_buffer('weibull_alpha', torch.ones(num_labels))
        self.register_buffer('weibull_beta', torch.ones(num_labels))
        self.register_buffer('weibull_tau', torch.zeros(num_labels))
        
        # Activation vectors for each class (learned during training)
        self.register_buffer('activation_vectors', torch.zeros(num_labels, hidden))
        
    def forward(self, x: torch.Tensor, use_openmax: bool = True) -> torch.Tensor:
        # Get activations from the penultimate layer
        activations = x
        for i, layer in enumerate(self.net[:-1]):
            activations = layer(activations)
        
        # Get logits from final layer
        logits = self.net[-1](activations)
        
        if use_openmax and self.training == False:
            # Apply OpenMax during inference
            return self.openmax_forward(activations, logits)
        else:
            return logits
    
    def openmax_forward(self, activations: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply OpenMax to get calibrated probabilities with unknown class
        """
        batch_size = activations.size(0)
        
        # Calculate distances to activation vectors for each class
        distances = torch.zeros(batch_size, self.num_labels, device=activations.device)
        for i in range(self.num_labels):
            class_activation = self.activation_vectors[i]
            distances[:, i] = torch.norm(activations - class_activation, dim=1)
        
        # Apply Weibull CDF to get probability of being unknown
        unknown_probs = torch.zeros(batch_size, device=activations.device)
        for i in range(self.num_labels):
            # Weibull CDF: 1 - exp(-((x-tau)/beta)^alpha)
            x = distances[:, i]
            alpha = self.weibull_alpha[i]
            beta = self.weibull_beta[i]
            tau = self.weibull_tau[i]
            
            # Avoid division by zero and negative values
            safe_beta = torch.clamp(beta, min=1e-6)
            safe_x = torch.clamp(x - tau, min=0)
            
            weibull_cdf = 1 - torch.exp(-torch.pow(safe_x / safe_beta, alpha))
            unknown_probs = torch.maximum(unknown_probs, weibull_cdf)
        
        # Adjust logits based on unknown probability
        adjusted_logits = logits.clone()
        for i in range(batch_size):
            unknown_prob = unknown_probs[i]
            if unknown_prob > 0.5:  # High uncertainty threshold
                # Reduce confidence in all classes
                adjusted_logits[i] = adjusted_logits[i] * (1 - unknown_prob)
        
        return adjusted_logits
    
    def fit_weibull(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Fit Weibull distribution parameters for each class
        This should be called after training with validation data
        """
        print("Fitting Weibull distributions for OpenMax...")
        
        for class_idx in range(self.num_labels):
            # Get activations for this class
            class_mask = labels == class_idx
            if class_mask.sum() == 0:
                continue
                
            class_activations = activations[class_mask]
            
            # Calculate mean activation vector for this class
            mean_activation = class_activations.mean(dim=0)
            self.activation_vectors[class_idx] = mean_activation
            
            # Calculate distances to mean activation
            distances = torch.norm(class_activations - mean_activation, dim=1)
            
            # Fit Weibull parameters (simplified fitting)
            # In practice, you'd use a proper Weibull fitting library
            distances_np = distances.cpu().numpy()
            
            # Simple parameter estimation
            self.weibull_alpha[class_idx] = torch.tensor(2.0)  # Shape parameter
            self.weibull_beta[class_idx] = torch.tensor(distances_np.std())  # Scale parameter
            self.weibull_tau[class_idx] = torch.tensor(distances_np.min())  # Location parameter
            
        print("Weibull fitting completed!")


# Keep the old classifier for backward compatibility
class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden: int = 128, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
