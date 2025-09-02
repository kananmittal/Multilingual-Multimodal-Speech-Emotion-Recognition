import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

# Import existing loss functions
from .losses import LabelSmoothingCrossEntropy, ClassBalancedFocalLoss, SupConLoss
from .prototypes import PrototypeMemory
from .cross_lingual_variance import CrossLingualConsistencyLoss
from .dual_gate_ood import DualGateOODDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Enumeration for different training phases."""
    REPRESENTATION_LEARNING = "representation_learning"
    ADVERSARIAL_TRAINING = "adversarial_training"
    CALIBRATION = "calibration"


@dataclass
class LossWeights:
    """Container for loss component weights."""
    # Primary losses
    ce_loss: float = 1.0
    supcon_loss: float = 0.25
    prototype_loss: float = 0.3
    
    # Adversarial and OOD losses
    language_adversarial_loss: float = -0.1  # Negative for adversarial training
    energy_margin_loss: float = 0.15
    
    # Temporal and calibration losses
    temporal_consistency_loss: float = 0.2
    confidence_calibration_loss: float = 0.1
    
    def get_phase_weights(self, phase: TrainingPhase) -> Dict[str, float]:
        """Get loss weights for specific training phase."""
        if phase == TrainingPhase.REPRESENTATION_LEARNING:
            return {
                'ce_loss': self.ce_loss,
                'supcon_loss': self.supcon_loss,
                'prototype_loss': self.prototype_loss,
                'language_adversarial_loss': 0.0,
                'energy_margin_loss': 0.0,
                'temporal_consistency_loss': 0.0,
                'confidence_calibration_loss': 0.0
            }
        elif phase == TrainingPhase.ADVERSARIAL_TRAINING:
            return {
                'ce_loss': self.ce_loss,
                'supcon_loss': self.supcon_loss,
                'prototype_loss': self.prototype_loss,
                'language_adversarial_loss': self.language_adversarial_loss,
                'energy_margin_loss': self.energy_margin_loss,
                'temporal_consistency_loss': 0.0,
                'confidence_calibration_loss': 0.0
            }
        elif phase == TrainingPhase.CALIBRATION:
            return {
                'ce_loss': self.ce_loss,
                'supcon_loss': self.supcon_loss,
                'prototype_loss': self.prototype_loss,
                'language_adversarial_loss': self.language_adversarial_loss,
                'energy_margin_loss': self.energy_margin_loss,
                'temporal_consistency_loss': self.temporal_consistency_loss,
                'confidence_calibration_loss': self.confidence_calibration_loss
            }
        else:
            raise ValueError(f"Unknown training phase: {phase}")


@dataclass
class LossComponents:
    """Container for computed loss components."""
    ce_loss: torch.Tensor
    supcon_loss: torch.Tensor
    prototype_loss: torch.Tensor
    language_adversarial_loss: torch.Tensor
    energy_margin_loss: torch.Tensor
    temporal_consistency_loss: torch.Tensor
    confidence_calibration_loss: torch.Tensor
    
    def get_loss_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format."""
        return {
            'ce_loss': self.ce_loss,
            'supcon_loss': self.supcon_loss,
            'prototype_loss': self.prototype_loss,
            'language_adversarial_loss': self.language_adversarial_loss,
            'energy_margin_loss': self.energy_margin_loss,
            'temporal_consistency_loss': self.temporal_consistency_loss,
            'confidence_calibration_loss': self.confidence_calibration_loss
        }


class EnergyMarginLoss(nn.Module):
    """
    Energy-Margin Loss for OOD detection.
    
    Encourages in-domain samples to have low energy scores
    and OOD samples to have high energy scores with a margin.
    """
    
    def __init__(self, margin: float = 10.0, temperature: float = 1.0):
        super().__init__()
        
        self.margin = margin
        self.temperature = temperature
        
        logging.info(f"Energy-Margin Loss initialized: margin={margin}, temperature={temperature}")
    
    def forward(self, 
                logits: torch.Tensor,
                is_ood: torch.Tensor,
                energy_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute energy-margin loss.
        
        Args:
            logits: [batch_size, num_classes] - model logits
            is_ood: [batch_size] - boolean tensor indicating OOD samples
            energy_scores: [batch_size] - pre-computed energy scores (optional)
            
        Returns:
            Energy-margin loss tensor
        """
        if energy_scores is None:
            # Compute energy scores from logits
            energy_scores = -torch.logsumexp(logits / self.temperature, dim=-1)
        
        # Separate in-domain and OOD samples
        in_domain_mask = ~is_ood
        ood_mask = is_ood
        
        loss = torch.tensor(0.0, device=logits.device)
        
        if in_domain_mask.sum() > 0:
            # In-domain samples should have low energy (high confidence)
            in_domain_energy = energy_scores[in_domain_mask]
            in_domain_loss = torch.mean(F.relu(in_domain_energy))
            loss = loss + in_domain_loss
        
        if ood_mask.sum() > 0:
            # OOD samples should have high energy (low confidence)
            ood_energy = energy_scores[ood_mask]
            ood_loss = torch.mean(F.relu(self.margin - ood_energy))
            loss = loss + ood_loss
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal Consistency Loss for emotion transition smoothing.
    
    Encourages smooth transitions between consecutive emotion predictions
    unless there's a high confidence change.
    """
    
    def __init__(self, consistency_weight: float = 1.0, confidence_threshold: float = 0.8):
        super().__init__()
        
        self.consistency_weight = consistency_weight
        self.confidence_threshold = confidence_threshold
        
        logging.info(f"Temporal Consistency Loss initialized: weight={consistency_weight}, threshold={confidence_threshold}")
    
    def forward(self, 
                current_predictions: torch.Tensor,
                previous_predictions: torch.Tensor,
                current_confidence: torch.Tensor,
                previous_confidence: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            current_predictions: [batch_size, num_classes] - current emotion predictions
            previous_predictions: [batch_size, num_classes] - previous emotion predictions
            current_confidence: [batch_size] - current prediction confidence
            previous_confidence: [batch_size] - previous prediction confidence
            
        Returns:
            Temporal consistency loss tensor
        """
        # Convert predictions to probabilities
        current_probs = F.softmax(current_predictions, dim=-1)
        previous_probs = F.softmax(previous_predictions, dim=-1)
        
        # Compute KL divergence between consecutive predictions
        kl_divergence = F.kl_div(
            current_probs.log(), 
            previous_probs, 
            reduction='none'
        ).sum(dim=-1)
        
        # Weight by confidence: higher confidence changes are allowed
        confidence_factor = torch.min(current_confidence, previous_confidence)
        consistency_mask = confidence_factor < self.confidence_threshold
        
        if consistency_mask.sum() > 0:
            # Apply consistency loss only to low-confidence transitions
            consistency_loss = torch.mean(kl_divergence[consistency_mask])
        else:
            consistency_loss = torch.tensor(0.0, device=current_predictions.device)
        
        return self.consistency_weight * consistency_loss


class ConfidenceCalibrationLoss(nn.Module):
    """
    Confidence Calibration Loss for uncertainty estimation.
    
    Ensures that predicted confidence scores are well-calibrated
    with actual prediction accuracy.
    """
    
    def __init__(self, calibration_weight: float = 1.0):
        super().__init__()
        
        self.calibration_weight = calibration_weight
        
        logging.info(f"Confidence Calibration Loss initialized: weight={calibration_weight}")
    
    def forward(self, 
                predicted_confidence: torch.Tensor,
                actual_accuracy: torch.Tensor,
                num_bins: int = 10) -> torch.Tensor:
        """
        Compute confidence calibration loss.
        
        Args:
            predicted_confidence: [batch_size] - predicted confidence scores
            actual_accuracy: [batch_size] - binary tensor indicating correct predictions
            num_bins: Number of confidence bins for calibration
            
        Returns:
            Confidence calibration loss tensor
        """
        # Create confidence bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=predicted_confidence.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = (predicted_confidence > bin_lower) & (predicted_confidence <= bin_upper)
            
            if in_bin.sum() > 0:
                # Compute average confidence and accuracy in this bin
                bin_confidence = predicted_confidence[in_bin].mean()
                bin_accuracy = actual_accuracy[in_bin].float().mean()
                
                # Calibration error: difference between confidence and accuracy
                bin_error = (bin_confidence - bin_accuracy) ** 2
                calibration_error += bin_error
        
        # Average calibration error across bins
        calibration_error = calibration_error / num_bins
        
        return self.calibration_weight * calibration_error


class BatchCompositionValidator:
    """
    Validates batch composition requirements for effective loss computation.
    
    Ensures minimum sample counts, class balance, and language diversity.
    """
    
    def __init__(self, 
                 min_batch_size: int = 32,
                 min_ood_ratio: float = 0.2,
                 min_languages: int = 2,
                 min_classes: int = 2):
        super().__init__()
        
        self.min_batch_size = min_batch_size
        self.min_ood_ratio = min_ood_ratio
        self.min_languages = min_languages
        self.min_classes = min_classes
        
        logging.info(f"Batch Composition Validator initialized: min_batch={min_batch_size}, "
                    f"min_ood={min_ood_ratio}, min_langs={min_languages}, min_classes={min_classes}")
    
    def validate_batch(self, 
                      batch_data: Dict[str, torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate batch composition requirements.
        
        Args:
            batch_data: Dictionary containing batch information
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {}
        
        # Check batch size
        batch_size = len(batch_data.get('labels', []))
        validation_info['batch_size'] = batch_size
        validation_info['batch_size_valid'] = batch_size >= self.min_batch_size
        
        # Check OOD sample ratio
        is_ood = batch_data.get('is_ood', torch.zeros(batch_size, dtype=torch.bool))
        ood_ratio = is_ood.float().mean().item()
        validation_info['ood_ratio'] = ood_ratio
        validation_info['ood_ratio_valid'] = ood_ratio >= self.min_ood_ratio
        
        # Check language diversity
        language_ids = batch_data.get('language_ids', torch.zeros(batch_size))
        unique_languages = torch.unique(language_ids)
        num_languages = len(unique_languages)
        validation_info['num_languages'] = num_languages
        validation_info['languages_valid'] = num_languages >= self.min_languages
        
        # Check class balance
        labels = batch_data.get('labels', torch.zeros(batch_size))
        unique_classes = torch.unique(labels)
        num_classes = len(unique_classes)
        validation_info['num_classes'] = num_classes
        validation_info['classes_valid'] = num_classes >= self.min_classes
        
        # Overall validation
        is_valid = all([
            validation_info['batch_size_valid'],
            validation_info['ood_ratio_valid'],
            validation_info['languages_valid'],
            validation_info['classes_valid']
        ])
        
        validation_info['overall_valid'] = is_valid
        
        return is_valid, validation_info
    
    def get_validation_report(self, validation_info: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        report = f"""
Batch Composition Validation Report:
==================================
Batch Size: {validation_info['batch_size']} (min: {self.min_batch_size}) {'✅' if validation_info['batch_size_valid'] else '❌'}
OOD Ratio: {validation_info['ood_ratio']:.3f} (min: {self.min_ood_ratio}) {'✅' if validation_info['ood_ratio_valid'] else '❌'}
Languages: {validation_info['num_languages']} (min: {self.min_languages}) {'✅' if validation_info['languages_valid'] else '❌'}
Classes: {validation_info['num_classes']} (min: {self.min_classes}) {'✅' if validation_info['classes_valid'] else '❌'}

Overall Validation: {'✅ PASSED' if validation_info['overall_valid'] else '❌ FAILED'}
"""
        return report


class ComprehensiveLossIntegration(nn.Module):
    """
    Comprehensive Loss Function Integration.
    
    Integrates all loss components with proper weighting and training phases.
    """
    
    def __init__(self, 
                 num_classes: int,
                 feature_dim: int,
                 num_languages: int = 7,
                 loss_weights: Optional[LossWeights] = None,
                 training_phase: TrainingPhase = TrainingPhase.REPRESENTATION_LEARNING):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_languages = num_languages
        self.training_phase = training_phase
        
        # Initialize loss weights
        if loss_weights is None:
            self.loss_weights = LossWeights()
        else:
            self.loss_weights = loss_weights
        
        # Initialize loss components
        self.ce_loss_fn = LabelSmoothingCrossEntropy()
        self.supcon_loss_fn = SupConLoss()
        self.prototype_memory = PrototypeMemory(num_classes, feature_dim)
        self.cross_lingual_loss_fn = CrossLingualConsistencyLoss()
        self.energy_margin_loss_fn = EnergyMarginLoss()
        self.temporal_consistency_loss_fn = TemporalConsistencyLoss()
        self.confidence_calibration_loss_fn = ConfidenceCalibrationLoss()
        
        # Batch validation
        self.batch_validator = BatchCompositionValidator()
        
        logging.info(f"Comprehensive Loss Integration initialized: phase={training_phase.value}")
    
    def set_training_phase(self, phase: TrainingPhase):
        """Set the current training phase."""
        self.training_phase = phase
        logging.info(f"Training phase set to: {phase.value}")
    
    def forward(self, 
                batch_data: Dict[str, torch.Tensor],
                model_outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute comprehensive loss for the batch.
        
        Args:
            batch_data: Dictionary containing batch information
            model_outputs: Dictionary containing model outputs
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Validate batch composition
        is_valid, validation_info = self.batch_validator.validate_batch(batch_data)
        
        if not is_valid:
            logger.warning("Batch composition validation failed")
            logger.warning(self.batch_validator.get_validation_report(validation_info))
        
        # Get weights for current training phase
        phase_weights = self.loss_weights.get_phase_weights(self.training_phase)
        
        # Extract required tensors
        labels = batch_data['labels']
        features = model_outputs['features']
        logits = model_outputs['logits']
        language_ids = batch_data.get('language_ids', torch.zeros_like(labels))
        is_ood = batch_data.get('is_ood', torch.zeros_like(labels, dtype=torch.bool))
        
        # Compute individual loss components
        loss_components = {}
        
        # 1. Cross-Entropy Loss
        if phase_weights['ce_loss'] > 0:
            ce_loss = self.ce_loss_fn(logits, labels)
            loss_components['ce_loss'] = ce_loss
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)
            loss_components['ce_loss'] = ce_loss
        
        # 2. Supervised Contrastive Loss
        if phase_weights['supcon_loss'] > 0:
            supcon_loss = self.supcon_loss_fn(features, labels)
            loss_components['supcon_loss'] = supcon_loss
        else:
            supcon_loss = torch.tensor(0.0, device=logits.device)
            loss_components['supcon_loss'] = supcon_loss
        
        # 3. Prototype Loss
        if phase_weights['prototype_loss'] > 0:
            prototype_loss = self.prototype_memory.prototype_loss(features, labels)
            loss_components['prototype_loss'] = prototype_loss
        else:
            prototype_loss = torch.tensor(0.0, device=logits.device)
            loss_components['prototype_loss'] = prototype_loss
        
        # 4. Language Adversarial Loss
        if phase_weights['language_adversarial_loss'] != 0:
            # This would typically come from a language adversarial head
            # For now, we'll use a simplified version
            language_adversarial_loss = self._compute_language_adversarial_loss(features, language_ids)
            loss_components['language_adversarial_loss'] = language_adversarial_loss
        else:
            language_adversarial_loss = torch.tensor(0.0, device=logits.device)
            loss_components['language_adversarial_loss'] = language_adversarial_loss
        
        # 5. Energy-Margin Loss
        if phase_weights['energy_margin_loss'] > 0:
            energy_margin_loss = self.energy_margin_loss_fn(logits, is_ood)
            loss_components['energy_margin_loss'] = energy_margin_loss
        else:
            energy_margin_loss = torch.tensor(0.0, device=logits.device)
            loss_components['energy_margin_loss'] = energy_margin_loss
        
        # 6. Temporal Consistency Loss
        if phase_weights['temporal_consistency_loss'] > 0:
            # This requires temporal context - simplified for now
            temporal_consistency_loss = self._compute_temporal_consistency_loss(model_outputs)
            loss_components['temporal_consistency_loss'] = temporal_consistency_loss
        else:
            temporal_consistency_loss = torch.tensor(0.0, device=logits.device)
            loss_components['temporal_consistency_loss'] = temporal_consistency_loss
        
        # 7. Confidence Calibration Loss
        if phase_weights['confidence_calibration_loss'] > 0:
            predicted_confidence = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            actual_accuracy = (logits.argmax(dim=-1) == labels).float()
            confidence_calibration_loss = self.confidence_calibration_loss_fn(
                predicted_confidence, actual_accuracy
            )
            loss_components['confidence_calibration_loss'] = confidence_calibration_loss
        else:
            confidence_calibration_loss = torch.tensor(0.0, device=logits.device)
            loss_components['confidence_calibration_loss'] = confidence_calibration_loss
        
        # Compute weighted total loss
        total_loss = (
            phase_weights['ce_loss'] * ce_loss +
            phase_weights['supcon_loss'] * supcon_loss +
            phase_weights['prototype_loss'] * prototype_loss +
            phase_weights['language_adversarial_loss'] * language_adversarial_loss +
            phase_weights['energy_margin_loss'] * energy_margin_loss +
            phase_weights['temporal_consistency_loss'] * temporal_consistency_loss +
            phase_weights['confidence_calibration_loss'] * confidence_calibration_loss
        )
        
        # Store loss components for analysis
        loss_components['total_loss'] = total_loss
        loss_components['phase_weights'] = phase_weights
        loss_components['validation_info'] = validation_info
        
        return total_loss, loss_components
    
    def _compute_language_adversarial_loss(self, 
                                         features: torch.Tensor, 
                                         language_ids: torch.Tensor) -> torch.Tensor:
        """Compute simplified language adversarial loss."""
        # This is a simplified implementation
        # In practice, you'd have a language classifier head
        batch_size = features.shape[0]
        
        # Create language labels
        language_labels = language_ids.long()
        
        # Simple language prediction from features
        language_logits = torch.randn(batch_size, self.num_languages, device=features.device)
        
        # Cross-entropy loss for language prediction
        language_loss = F.cross_entropy(language_logits, language_labels)
        
        return language_loss
    
    def _compute_temporal_consistency_loss(self, 
                                         model_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute simplified temporal consistency loss."""
        # This is a simplified implementation
        # In practice, you'd have temporal context from previous predictions
        
        # For now, return zero loss
        return torch.tensor(0.0, device=next(iter(model_outputs.values())).device)
    
    def get_loss_report(self, loss_components: Dict[str, Any]) -> str:
        """Generate comprehensive loss report."""
        phase_weights = loss_components['phase_weights']
        validation_info = loss_components['validation_info']
        
        report = f"""
Comprehensive Loss Report:
=========================
Training Phase: {self.training_phase.value}

Loss Components:
  Cross-Entropy Loss: {loss_components['ce_loss'].item():.6f} (weight: {phase_weights['ce_loss']})
  Supervised Contrastive Loss: {loss_components['supcon_loss'].item():.6f} (weight: {phase_weights['supcon_loss']})
  Prototype Loss: {loss_components['prototype_loss'].item():.6f} (weight: {phase_weights['prototype_loss']})
  Language Adversarial Loss: {loss_components['language_adversarial_loss'].item():.6f} (weight: {phase_weights['language_adversarial_loss']})
  Energy-Margin Loss: {loss_components['energy_margin_loss'].item():.6f} (weight: {phase_weights['energy_margin_loss']})
  Temporal Consistency Loss: {loss_components['temporal_consistency_loss'].item():.6f} (weight: {phase_weights['temporal_consistency_loss']})
  Confidence Calibration Loss: {loss_components['confidence_calibration_loss'].item():.6f} (weight: {phase_weights['confidence_calibration_loss']})

Total Loss: {loss_components['total_loss'].item():.6f}

Batch Validation:
{self.batch_validator.get_validation_report(validation_info)}
"""
        return report
    
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Update prototype memory with new data."""
        # PrototypeMemory doesn't have update_prototypes method
        # Prototypes are learned parameters that get updated during training
        # This method is a placeholder for future implementation if needed
        pass
    
    def get_prototype_info(self) -> Dict[str, Any]:
        """Get information about current prototypes."""
        return {
            'prototype_count': len(self.prototype_memory.prototypes),
            'feature_dim': self.prototype_memory.prototypes.shape[1] if len(self.prototype_memory.prototypes) > 0 else 0,
            'prototype_norms': torch.norm(self.prototype_memory.prototypes, dim=1).tolist() if len(self.prototype_memory.prototypes) > 0 else []
        }


class TrainingPhaseManager:
    """
    Manages training phase transitions and loss weight scheduling.
    
    Handles automatic phase transitions based on training progress.
    """
    
    def __init__(self, 
                 initial_phase: TrainingPhase = TrainingPhase.REPRESENTATION_LEARNING,
                 phase_transitions: Optional[Dict[TrainingPhase, int]] = None):
        super().__init__()
        
        self.current_phase = initial_phase
        self.phase_transitions = phase_transitions or {
            TrainingPhase.REPRESENTATION_LEARNING: 0,
            TrainingPhase.ADVERSARIAL_TRAINING: 50,  # Start at epoch 50
            TrainingPhase.CALIBRATION: 100  # Start at epoch 100
        }
        
        self.current_epoch = 0
        
        logging.info(f"Training Phase Manager initialized: current_phase={self.current_phase.value}")
    
    def update_epoch(self, epoch: int) -> TrainingPhase:
        """Update current epoch and return appropriate training phase."""
        self.current_epoch = epoch
        
        # Determine phase based on epoch
        if epoch >= self.phase_transitions[TrainingPhase.CALIBRATION]:
            new_phase = TrainingPhase.CALIBRATION
        elif epoch >= self.phase_transitions[TrainingPhase.ADVERSARIAL_TRAINING]:
            new_phase = TrainingPhase.ADVERSARIAL_TRAINING
        else:
            new_phase = TrainingPhase.REPRESENTATION_LEARNING
        
        if new_phase != self.current_phase:
            logger.info(f"Training phase transition: {self.current_phase.value} → {new_phase.value} at epoch {epoch}")
            self.current_phase = new_phase
        
        return self.current_phase
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get current phase information."""
        return {
            'current_phase': self.current_phase.value,
            'current_epoch': self.current_epoch,
            'next_transition': self._get_next_transition_epoch(),
            'phase_progress': self._get_phase_progress()
        }
    
    def _get_next_transition_epoch(self) -> Optional[int]:
        """Get the next phase transition epoch."""
        current_phase_idx = list(self.phase_transitions.keys()).index(self.current_phase)
        
        if current_phase_idx < len(self.phase_transitions) - 1:
            next_phase = list(self.phase_transitions.keys())[current_phase_idx + 1]
            return self.phase_transitions[next_phase]
        
        return None
    
    def _get_phase_progress(self) -> float:
        """Get progress within current phase (0.0 to 1.0)."""
        phase_start = self.phase_transitions[self.current_phase]
        phase_end = self._get_next_transition_epoch() or (phase_start + 50)  # Default 50 epochs
        
        if phase_end == phase_start:
            return 1.0
        
        progress = (self.current_epoch - phase_start) / (phase_end - phase_start)
        return max(0.0, min(1.0, progress))


# Utility functions
def create_comprehensive_loss_integration(num_classes: int,
                                        feature_dim: int,
                                        num_languages: int = 7) -> ComprehensiveLossIntegration:
    """Factory function to create comprehensive loss integration."""
    return ComprehensiveLossIntegration(
        num_classes=num_classes,
        feature_dim=feature_dim,
        num_languages=num_languages
    )


def create_training_phase_manager(initial_phase: TrainingPhase = TrainingPhase.REPRESENTATION_LEARNING) -> TrainingPhaseManager:
    """Factory function to create training phase manager."""
    return TrainingPhaseManager(initial_phase=initial_phase)


def create_sample_batch_data(batch_size: int = 32,
                           num_classes: int = 4,
                           num_languages: int = 7,
                           feature_dim: int = 256) -> Dict[str, torch.Tensor]:
    """Create sample batch data for testing."""
    np.random.seed(42)
    
    # Generate sample data
    labels = torch.randint(0, num_classes, (batch_size,))
    language_ids = torch.randint(0, num_languages, (batch_size,))
    is_ood = torch.rand(batch_size) < 0.2  # 20% OOD samples
    
    # Ensure class balance
    for i in range(num_classes):
        class_count = (labels == i).sum()
        if class_count < 2:
            # Add more samples of this class
            additional_samples = 2 - class_count
            additional_indices = torch.randint(0, batch_size, (additional_samples,))
            labels[additional_indices] = i
    
    return {
        'labels': labels,
        'language_ids': language_ids,
        'is_ood': is_ood
    }


def create_sample_model_outputs(batch_size: int = 32,
                              num_classes: int = 4,
                              feature_dim: int = 256) -> Dict[str, torch.Tensor]:
    """Create sample model outputs for testing."""
    np.random.seed(42)
    
    # Generate sample outputs
    features = torch.randn(batch_size, feature_dim)
    logits = torch.randn(batch_size, num_classes)
    
    return {
        'features': features,
        'logits': logits
    }
