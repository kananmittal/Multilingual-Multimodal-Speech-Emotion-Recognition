import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import logging


class OODStage(Enum):
    """Enumeration for OOD detection stages."""
    EARLY = "early"
    LATE = "late"
    COMBINED = "combined"


class OODReason(Enum):
    """Enumeration for OOD detection reasons."""
    LOW_SNR = "low_snr"
    HIGH_CLIPPING = "high_clipping"
    LOW_SPEECH_PROB = "low_speech_prob"
    HIGH_LID_ENTROPY = "high_lid_entropy"
    LOW_LANGUAGE_CONF = "low_language_conf"
    HIGH_MUSIC_PROB = "high_music_prob"
    HIGH_LAUGHTER_PROB = "high_laughter_prob"
    EXCESSIVE_CONDITIONING = "excessive_conditioning"
    HIGH_ENERGY = "high_energy"
    HIGH_PROTOTYPE_DISTANCE = "high_prototype_distance"
    COMBINED_THRESHOLD = "combined_threshold"


@dataclass
class EarlyOODResult:
    """Result of early-stage OOD detection."""
    is_ood: bool
    confidence_score: float
    reason: Optional[OODReason]
    quality_metrics: Dict[str, float]
    early_abstain: bool


@dataclass
class LateOODResult:
    """Result of late-stage OOD detection."""
    is_ood: bool
    energy_score: float
    prototype_distance: float
    combined_score: float
    confidence_score: float
    reason: Optional[OODReason]


@dataclass
class CombinedOODResult:
    """Combined result of dual-gate OOD detection."""
    is_ood: bool
    stage: OODStage
    confidence_score: float
    reason: Optional[OODReason]
    early_result: Optional[EarlyOODResult]
    late_result: Optional[LateOODResult]
    computational_savings: bool


class EarlyOODDetector(nn.Module):
    """
    Early-Stage OOD Detection for front-end abstention.
    
    Detects OOD samples based on quality, language, and content criteria
    to enable early computational savings.
    """
    
    def __init__(self, 
                 snr_threshold: float = 5.0,
                 clipping_threshold: float = 30.0,
                 speech_prob_threshold: float = 0.4,
                 lid_entropy_threshold: float = 2.0,
                 language_conf_threshold: float = 0.3,
                 music_prob_threshold: float = 0.5,
                 laughter_prob_threshold: float = 0.6,
                 conditioning_threshold: float = 15.0):
        super().__init__()
        
        # Quality thresholds
        self.snr_threshold = snr_threshold
        self.clipping_threshold = clipping_threshold
        self.speech_prob_threshold = speech_prob_threshold
        
        # Language thresholds
        self.lid_entropy_threshold = lid_entropy_threshold
        self.language_conf_threshold = language_conf_threshold
        
        # Content thresholds
        self.music_prob_threshold = music_prob_threshold
        self.laughter_prob_threshold = laughter_prob_threshold
        
        # Processing thresholds
        self.conditioning_threshold = conditioning_threshold
        
        logging.info(f"Early OOD Detector initialized with quality-based thresholds")
    
    def forward(self, quality_metrics: Dict[str, float]) -> EarlyOODResult:
        """
        Perform early-stage OOD detection.
        
        Args:
            quality_metrics: Dictionary containing quality metrics
            
        Returns:
            EarlyOODResult with detection outcome
        """
        # Extract metrics with defaults
        snr_db = quality_metrics.get('snr_db', float('inf'))
        clipping_percent = quality_metrics.get('clipping_percent', 0.0)
        speech_prob = quality_metrics.get('speech_prob', 1.0)
        lid_entropy = quality_metrics.get('lid_entropy', 0.0)
        language_conf = quality_metrics.get('language_conf', 1.0)
        music_prob = quality_metrics.get('music_prob', 0.0)
        laughter_prob = quality_metrics.get('laughter_prob', 0.0)
        denoise_gain_db = quality_metrics.get('denoise_gain_db', 0.0)
        
        # Quality-based detection
        quality_ood = (snr_db < self.snr_threshold or 
                      clipping_percent > self.clipping_threshold or 
                      speech_prob < self.speech_prob_threshold)
        
        # Language-based detection
        language_ood = (lid_entropy > self.lid_entropy_threshold or 
                       language_conf < self.language_conf_threshold)
        
        # Content-based detection
        content_ood = (music_prob > self.music_prob_threshold or 
                      laughter_prob > self.laughter_prob_threshold)
        
        # Processing-based detection
        processing_ood = denoise_gain_db > self.conditioning_threshold
        
        # Combined early OOD detection
        is_ood = quality_ood or language_ood or content_ood or processing_ood
        
        # Determine reason
        reason = None
        if quality_ood:
            if snr_db < self.snr_threshold:
                reason = OODReason.LOW_SNR
            elif clipping_percent > self.clipping_threshold:
                reason = OODReason.HIGH_CLIPPING
            elif speech_prob < self.speech_prob_threshold:
                reason = OODReason.LOW_SPEECH_PROB
        elif language_ood:
            if lid_entropy > self.lid_entropy_threshold:
                reason = OODReason.HIGH_LID_ENTROPY
            elif language_conf < self.language_conf_threshold:
                reason = OODReason.LOW_LANGUAGE_CONF
        elif content_ood:
            if music_prob > self.music_prob_threshold:
                reason = OODReason.HIGH_MUSIC_PROB
            elif laughter_prob > self.laughter_prob_threshold:
                reason = OODReason.HIGH_LAUGHTER_PROB
        elif processing_ood:
            reason = OODReason.EXCESSIVE_CONDITIONING
        
        # Compute confidence score (lower = more confident in OOD)
        confidence_factors = []
        if snr_db < self.snr_threshold:
            confidence_factors.append(1.0 - (snr_db / self.snr_threshold))
        if clipping_percent > self.clipping_threshold:
            confidence_factors.append((clipping_percent - self.clipping_threshold) / (100 - self.clipping_threshold))
        if speech_prob < self.speech_prob_threshold:
            confidence_factors.append(1.0 - (speech_prob / self.speech_prob_threshold))
        if lid_entropy > self.lid_entropy_threshold:
            confidence_factors.append((lid_entropy - self.lid_entropy_threshold) / (3.0 - self.lid_entropy_threshold))
        
        confidence_score = np.mean(confidence_factors) if confidence_factors else 0.0
        confidence_score = np.clip(confidence_score, 0.0, 1.0)
        
        return EarlyOODResult(
            is_ood=is_ood,
            confidence_score=confidence_score,
            reason=reason,
            quality_metrics=quality_metrics,
            early_abstain=is_ood
        )


class EnergyBasedOODDetector(nn.Module):
    """
    Energy-Based OOD Detection using free energy scores.
    
    Computes energy scores from model logits and calibrates them
    using temperature scaling.
    """
    
    def __init__(self, temperature: float = 1.0, energy_threshold: float = 0.5):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.energy_threshold = energy_threshold
        
        logging.info(f"Energy-Based OOD Detector initialized with temperature={temperature}")
    
    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy scores for OOD detection.
        
        Args:
            logits: [batch_size, num_classes] - model logits
            
        Returns:
            energy_scores: [batch_size] - energy scores (lower = more confident)
            calibrated_logits: [batch_size, num_classes] - temperature-scaled logits
        """
        # Apply temperature scaling
        calibrated_logits = logits / self.temperature
        
        # Compute free energy: E(x) = -log(Σ exp(logit_i))
        energy_scores = -torch.logsumexp(calibrated_logits, dim=-1)
        
        return energy_scores, calibrated_logits
    
    def calibrate_temperature(self, val_logits: torch.Tensor, val_labels: torch.Tensor):
        """Calibrate temperature using validation data."""
        # This would typically use a validation set to optimize temperature
        # For now, we'll use a simple heuristic
        with torch.no_grad():
            # Compute optimal temperature based on validation performance
            temperatures = torch.linspace(0.1, 10.0, 100)
            best_temp = 1.0
            best_score = float('inf')
            
            for temp in temperatures:
                self.temperature.data.fill_(temp)
                energy_scores, _ = self.forward(val_logits)
                
                # Simple calibration metric (could be more sophisticated)
                score = energy_scores.std().item()
                if score < best_score:
                    best_score = score
                    best_temp = temp
            
            self.temperature.data.fill_(best_temp)
            logging.info(f"Temperature calibrated to {best_temp:.3f}")


class PrototypeDistanceOODDetector(nn.Module):
    """
    Prototype-Distance OOD Detection using Mahalanobis distance.
    
    Computes distance to nearest emotion prototype using learned
    covariance matrices.
    """
    
    def __init__(self, num_classes: int, feature_dim: int, distance_threshold: float = 2.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.distance_threshold = distance_threshold
        
        # Learnable prototypes for each emotion class
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        
        # Learnable covariance matrices (diagonal for efficiency)
        self.covariances = nn.Parameter(torch.ones(num_classes, feature_dim))
        
        # Initialize prototypes and covariances
        self._init_parameters()
        
        logging.info(f"Prototype Distance OOD Detector initialized: {num_classes} classes, {feature_dim} features")
    
    def _init_parameters(self):
        """Initialize prototypes and covariances."""
        # Initialize prototypes with small random values
        nn.init.xavier_uniform_(self.prototypes)
        
        # Initialize covariances to identity-like values
        nn.init.ones_(self.covariances)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prototype distances for OOD detection.
        
        Args:
            features: [batch_size, feature_dim] - input features
            
        Returns:
            distances: [batch_size, num_classes] - distances to each prototype
            min_distances: [batch_size] - minimum distance to any prototype
        """
        batch_size = features.shape[0]
        
        # Compute Mahalanobis distance to each prototype
        distances = torch.zeros(batch_size, self.num_classes, device=features.device)
        
        for i in range(self.num_classes):
            # Get prototype and covariance for this class
            prototype = self.prototypes[i]  # [feature_dim]
            covariance = self.covariances[i]  # [feature_dim]
            
            # Compute squared Mahalanobis distance
            diff = features - prototype.unsqueeze(0)  # [batch_size, feature_dim]
            inv_cov = 1.0 / (covariance + 1e-8)  # [feature_dim]
            
            # Mahalanobis distance: sqrt(diff^T * inv_cov * diff)
            mahal_dist = torch.sqrt(torch.sum(diff * diff * inv_cov.unsqueeze(0), dim=-1))
            distances[:, i] = mahal_dist
        
        # Get minimum distance to any prototype
        min_distances, _ = torch.min(distances, dim=-1)
        
        return distances, min_distances
    
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Update prototypes using training data."""
        with torch.no_grad():
            for i in range(self.num_classes):
                class_mask = (labels == i)
                if class_mask.sum() > 0:
                    class_features = features[class_mask]
                    
                    # Update prototype as mean of class features
                    new_prototype = class_features.mean(dim=0)
                    self.prototypes[i].data.copy_(new_prototype)
                    
                    # Update covariance as variance of class features
                    new_covariance = class_features.var(dim=0) + 1e-8
                    self.covariances[i].data.copy_(new_covariance)


class LateStageOODDetector(nn.Module):
    """
    Late-Stage OOD Detection combining energy and prototype distance.
    
    Combines energy-based and prototype-distance scores for robust
    OOD detection.
    """
    
    def __init__(self, 
                 num_classes: int,
                 feature_dim: int,
                 energy_weight: float = 0.6,
                 prototype_weight: float = 0.4,
                 combined_threshold: float = 0.5):
        super().__init__()
        
        self.energy_weight = energy_weight
        self.prototype_weight = prototype_weight
        self.combined_threshold = combined_threshold
        
        # OOD detection components
        self.energy_detector = EnergyBasedOODDetector()
        self.prototype_detector = PrototypeDistanceOODDetector(num_classes, feature_dim)
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.tensor([energy_weight, prototype_weight]))
        
        logging.info(f"Late-Stage OOD Detector initialized with energy_weight={energy_weight}, prototype_weight={prototype_weight}")
    
    def forward(self, 
                logits: torch.Tensor, 
                features: torch.Tensor) -> LateOODResult:
        """
        Perform late-stage OOD detection.
        
        Args:
            logits: [batch_size, num_classes] - model logits
            features: [batch_size, feature_dim] - input features
            
        Returns:
            LateOODResult with detection outcome
        """
        batch_size = logits.shape[0]
        
        # Energy-based detection
        energy_scores, _ = self.energy_detector(logits)
        
        # Prototype-distance detection
        prototype_distances, min_distances = self.prototype_detector(features)
        
        # Normalize scores to [0, 1] range
        energy_scores_norm = torch.sigmoid(-energy_scores)  # Higher = more confident
        distance_scores_norm = torch.exp(-min_distances)    # Higher = more confident
        
        # Combine scores using learnable weights
        weights = F.softmax(self.combination_weights, dim=0)
        combined_score = (weights[0] * energy_scores_norm + 
                         weights[1] * distance_scores_norm)
        
        # Determine OOD status
        is_ood = combined_score < self.combined_threshold
        
        # Determine reason
        reason = None
        if energy_scores_norm.mean() < 0.3:
            reason = OODReason.HIGH_ENERGY
        elif distance_scores_norm.mean() < 0.3:
            reason = OODReason.HIGH_PROTOTYPE_DISTANCE
        else:
            reason = OODReason.COMBINED_THRESHOLD
        
        # Compute confidence score
        confidence_score = combined_score.mean().item()
        
        return LateOODResult(
            is_ood=is_ood.any().item(),
            energy_score=energy_scores.mean().item(),
            prototype_distance=min_distances.mean().item(),
            combined_score=combined_score.mean().item(),
            confidence_score=confidence_score,
            reason=reason
        )


class AdaptiveThresholdManager(nn.Module):
    """
    Adaptive Threshold Manager for per-language and per-SNR thresholds.
    
    Maintains separate OOD thresholds for different languages and SNR ranges.
    """
    
    def __init__(self, 
                 num_languages: int = 7,
                 snr_ranges: List[Tuple[float, float]] = None):
        super().__init__()
        
        self.num_languages = num_languages
        
        if snr_ranges is None:
            self.snr_ranges = [(-float('inf'), 10.0), (10.0, 20.0), (20.0, float('inf'))]
        else:
            self.snr_ranges = snr_ranges
        
        # Initialize thresholds for each language-SNR combination
        self.thresholds = nn.Parameter(torch.ones(num_languages, len(self.snr_ranges)) * 0.5)
        
        # Global fallback threshold
        self.global_threshold = nn.Parameter(torch.tensor(0.5))
        
        logging.info(f"Adaptive Threshold Manager initialized: {num_languages} languages, {len(self.snr_ranges)} SNR ranges")
    
    def get_threshold(self, 
                     language_id: int, 
                     snr_db: float) -> float:
        """
        Get OOD threshold for specific language and SNR.
        
        Args:
            language_id: Language identifier
            snr_db: SNR in dB
            
        Returns:
            OOD threshold for the given conditions
        """
        # Ensure language_id is within bounds
        language_id = max(0, min(language_id, self.num_languages - 1))
        
        # Find SNR range
        snr_range_idx = 0
        for i, (low, high) in enumerate(self.snr_ranges):
            if low <= snr_db < high:
                snr_range_idx = i
                break
        
        # Get specific threshold
        specific_threshold = self.thresholds[language_id, snr_range_idx].item()
        
        # Use global threshold as fallback if specific threshold is too extreme
        if specific_threshold < 0.1 or specific_threshold > 0.9:
            return self.global_threshold.item()
        
        return specific_threshold
    
    def update_thresholds(self, 
                         language_id: int, 
                         snr_range_idx: int, 
                         new_threshold: float):
        """Update threshold for specific language-SNR combination."""
        with torch.no_grad():
            self.thresholds[language_id, snr_range_idx].data.fill_(new_threshold)
    
    def get_threshold_info(self) -> Dict:
        """Get information about current thresholds."""
        return {
            'language_snr_thresholds': self.thresholds.detach().numpy().tolist(),
            'global_threshold': self.global_threshold.item(),
            'snr_ranges': self.snr_ranges,
            'num_languages': self.num_languages
        }


class DualGateOODDetector(nn.Module):
    """
    Complete Dual-Gate OOD Detection System.
    
    Combines early-stage abstention with late-stage detection for
    robust OOD detection and computational savings.
    """
    
    def __init__(self, 
                 num_classes: int,
                 feature_dim: int,
                 num_languages: int = 7,
                 early_abstain: bool = True,
                 late_detection: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_languages = num_languages
        self.early_abstain = early_abstain
        self.late_detection = late_detection
        
        # Early-stage OOD detection
        if early_abstain:
            self.early_detector = EarlyOODDetector()
        
        # Late-stage OOD detection
        if late_detection:
            self.late_detector = LateStageOODDetector(num_classes, feature_dim)
        
        # Adaptive threshold management
        self.threshold_manager = AdaptiveThresholdManager(num_languages)
        
        # Outlier exposure training support
        self.outlier_exposure_loss = nn.CrossEntropyLoss()
        
        logging.info(f"Dual-Gate OOD Detector initialized: {num_classes} classes, {feature_dim} features")
    
    def forward(self, 
                quality_metrics: Dict[str, float],
                logits: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                language_id: Optional[int] = None) -> CombinedOODResult:
        """
        Perform dual-gate OOD detection.
        
        Args:
            quality_metrics: Quality metrics for early detection
            logits: Model logits for late detection
            features: Input features for late detection
            language_id: Language identifier for threshold selection
            
        Returns:
            CombinedOODResult with complete detection outcome
        """
        early_result = None
        late_result = None
        is_ood = False
        stage = OODStage.EARLY
        reason = None
        confidence_score = 0.0
        computational_savings = False
        
        # Early-stage detection
        if self.early_abstain:
            early_result = self.early_detector(quality_metrics)
            
            if early_result.early_abstain:
                is_ood = True
                stage = OODStage.EARLY
                reason = early_result.reason
                confidence_score = early_result.confidence_score
                computational_savings = True
                
                return CombinedOODResult(
                    is_ood=is_ood,
                    stage=stage,
                    confidence_score=confidence_score,
                    reason=reason,
                    early_result=early_result,
                    late_result=late_result,
                    computational_savings=computational_savings
                )
        
        # Late-stage detection
        if self.late_detection and logits is not None and features is not None:
            late_result = self.late_detector(logits, features)
            
            # Get adaptive threshold
            snr_db = quality_metrics.get('snr_db', 20.0)
            if language_id is None:
                language_id = 0  # Default to first language
            
            threshold = self.threshold_manager.get_threshold(language_id, snr_db)
            
            # Apply threshold
            is_ood = late_result.combined_score < threshold
            stage = OODStage.LATE
            reason = late_result.reason
            confidence_score = late_result.confidence_score
        
        # Combined result
        return CombinedOODResult(
            is_ood=is_ood,
            stage=stage,
            confidence_score=confidence_score,
            reason=reason,
            early_result=early_result,
            late_result=late_result,
            computational_savings=computational_savings
        )
    
    def train_with_outlier_exposure(self, 
                                   in_domain_logits: torch.Tensor,
                                   in_domain_labels: torch.Tensor,
                                   outlier_logits: torch.Tensor,
                                   outlier_labels: torch.Tensor):
        """
        Train OOD detector with outlier exposure.
        
        Args:
            in_domain_logits: Logits for in-domain samples
            in_domain_labels: Labels for in-domain samples
            outlier_logits: Logits for outlier samples
            outlier_labels: Labels for outlier samples (can be arbitrary)
        """
        # In-domain classification loss
        in_domain_loss = self.outlier_exposure_loss(in_domain_logits, in_domain_labels)
        
        # Outlier exposure loss (encourage high uncertainty for outliers)
        outlier_loss = self.outlier_exposure_loss(outlier_logits, outlier_labels)
        
        # Combined loss
        total_loss = in_domain_loss + 0.5 * outlier_loss
        
        return total_loss
    
    def get_detection_report(self, result: CombinedOODResult) -> str:
        """Generate comprehensive OOD detection report."""
        report = f"""
Dual-Gate OOD Detection Report:
==============================
OOD Status: {'OOD Detected' if result.is_ood else 'In-Domain'}
Detection Stage: {result.stage.value}
Confidence Score: {result.confidence_score:.3f}
Computational Savings: {'Yes' if result.computational_savings else 'No'}

Early-Stage Detection:
  {'Early Abstention: Yes' if result.early_result and result.early_result.early_abstain else 'Early Abstention: No'}
  {'Reason: ' + result.early_result.reason.value if result.early_result and result.early_result.reason else 'No early detection'}

Late-Stage Detection:
  {'Energy Score: ' + f'{result.late_result.energy_score:.3f}' if result.late_result else 'Not performed'}
  {'Prototype Distance: ' + f'{result.late_result.prototype_distance:.3f}' if result.late_result else 'Not performed'}
  {'Combined Score: ' + f'{result.late_result.combined_score:.3f}' if result.late_result else 'Not performed'}
"""
        return report


# Utility functions
def create_dual_gate_ood_detector(num_classes: int,
                                 feature_dim: int,
                                 num_languages: int = 7) -> DualGateOODDetector:
    """Factory function to create dual-gate OOD detector."""
    return DualGateOODDetector(
        num_classes=num_classes,
        feature_dim=feature_dim,
        num_languages=num_languages
    )


def create_quality_metrics(snr_db: float = 20.0,
                          clipping_percent: float = 5.0,
                          speech_prob: float = 0.9,
                          lid_entropy: float = 0.8,
                          language_conf: float = 0.9,
                          music_prob: float = 0.1,
                          laughter_prob: float = 0.1,
                          denoise_gain_db: float = 2.0) -> Dict[str, float]:
    """Factory function to create quality metrics dictionary."""
    return {
        'snr_db': snr_db,
        'clipping_percent': clipping_percent,
        'speech_prob': speech_prob,
        'lid_entropy': lid_entropy,
        'language_conf': language_conf,
        'music_prob': music_prob,
        'laughter_prob': laughter_prob,
        'denoise_gain_db': denoise_gain_db
    }
