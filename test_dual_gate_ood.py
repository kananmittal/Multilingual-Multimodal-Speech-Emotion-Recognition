#!/usr/bin/env python3
"""
Test for Dual-Gate OOD Detection
Verifies early abstention, late-stage detection, and adaptive thresholds
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from src.models.dual_gate_ood import (
    DualGateOODDetector,
    create_dual_gate_ood_detector,
    create_quality_metrics,
    EarlyOODDetector,
    EnergyBasedOODDetector,
    PrototypeDistanceOODDetector,
    LateStageOODDetector,
    AdaptiveThresholdManager,
    OODStage,
    OODReason
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_early_ood_detector():
    """Test the Early-Stage OOD Detection component."""
    
    print("=" * 60)
    print("EARLY-STAGE OOD DETECTION TEST")
    print("=" * 60)
    
    # Initialize early OOD detector
    early_detector = EarlyOODDetector()
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Normal Quality',
            'metrics': create_quality_metrics(
                snr_db=25.0,
                clipping_percent=5.0,
                speech_prob=0.9,
                lid_entropy=0.8,
                language_conf=0.9,
                music_prob=0.1,
                laughter_prob=0.1,
                denoise_gain_db=2.0
            ),
            'expected_ood': False
        },
        {
            'name': 'Low SNR',
            'metrics': create_quality_metrics(
                snr_db=3.0,  # Below 5dB threshold
                clipping_percent=5.0,
                speech_prob=0.9,
                lid_entropy=0.8,
                language_conf=0.9,
                music_prob=0.1,
                laughter_prob=0.1,
                denoise_gain_db=2.0
            ),
            'expected_ood': True,
            'expected_reason': OODReason.LOW_SNR
        },
        {
            'name': 'High Clipping',
            'metrics': create_quality_metrics(
                snr_db=25.0,
                clipping_percent=35.0,  # Above 30% threshold
                speech_prob=0.9,
                lid_entropy=0.8,
                language_conf=0.9,
                music_prob=0.1,
                laughter_prob=0.1,
                denoise_gain_db=2.0
            ),
            'expected_ood': True,
            'expected_reason': OODReason.HIGH_CLIPPING
        },
        {
            'name': 'Low Speech Probability',
            'metrics': create_quality_metrics(
                snr_db=25.0,
                clipping_percent=5.0,
                speech_prob=0.3,  # Below 0.4 threshold
                lid_entropy=0.8,
                language_conf=0.9,
                music_prob=0.1,
                laughter_prob=0.1,
                denoise_gain_db=2.0
            ),
            'expected_ood': True,
            'expected_reason': OODReason.LOW_SPEECH_PROB
        },
        {
            'name': 'High LID Entropy',
            'metrics': create_quality_metrics(
                snr_db=25.0,
                clipping_percent=5.0,
                speech_prob=0.9,
                lid_entropy=2.5,  # Above 2.0 threshold
                language_conf=0.9,
                music_prob=0.1,
                laughter_prob=0.1,
                denoise_gain_db=2.0
            ),
            'expected_ood': True,
            'expected_reason': OODReason.HIGH_LID_ENTROPY
        },
        {
            'name': 'High Music Probability',
            'metrics': create_quality_metrics(
                snr_db=25.0,
                clipping_percent=5.0,
                speech_prob=0.9,
                lid_entropy=0.8,
                language_conf=0.9,
                music_prob=0.7,  # Above 0.5 threshold
                laughter_prob=0.1,
                denoise_gain_db=2.0
            ),
            'expected_ood': True,
            'expected_reason': OODReason.HIGH_MUSIC_PROB
        }
    ]
    
    print(f"Testing {len(test_cases)} scenarios...")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['name']}")
        
        # Run early OOD detection
        result = early_detector(test_case['metrics'])
        
        print(f"  Expected OOD: {test_case['expected_ood']}")
        print(f"  Detected OOD: {result.is_ood}")
        print(f"  Early Abstain: {result.early_abstain}")
        print(f"  Confidence Score: {result.confidence_score:.3f}")
        print(f"  Reason: {result.reason.value if result.reason else 'None'}")
        
        # Verify results
        assert result.is_ood == test_case['expected_ood'], \
            f"OOD detection mismatch for {test_case['name']}"
        
        if test_case['expected_ood']:
            assert result.early_abstain, f"Should abstain early for {test_case['name']}"
            if 'expected_reason' in test_case:
                assert result.reason == test_case['expected_reason'], \
                    f"Reason mismatch for {test_case['name']}"
        
        print(f"  ✅ Test case passed")
    
    print(f"\n✅ Early-stage OOD detection working correctly")

def test_energy_based_ood_detector():
    """Test the Energy-Based OOD Detection component."""
    
    print("\n" + "=" * 60)
    print("ENERGY-BASED OOD DETECTION TEST")
    print("=" * 60)
    
    # Initialize energy-based detector
    energy_detector = EnergyBasedOODDetector(temperature=1.0)
    
    # Test data
    batch_size = 4
    num_classes = 4
    
    print(f"Testing with batch_size={batch_size}, num_classes={num_classes}")
    print("-" * 40)
    
    # Test case 1: High confidence logits (in-domain)
    high_conf_logits = torch.tensor([
        [5.0, 1.0, 1.0, 1.0],  # High confidence for class 0
        [1.0, 5.0, 1.0, 1.0],  # High confidence for class 1
        [1.0, 1.0, 5.0, 1.0],  # High confidence for class 2
        [1.0, 1.0, 1.0, 5.0]   # High confidence for class 3
    ])
    
    energy_scores1, calibrated_logits1 = energy_detector(high_conf_logits)
    
    print(f"High Confidence Logits:")
    print(f"  Energy scores: {energy_scores1}")
    print(f"  Mean energy: {energy_scores1.mean().item():.3f}")
    
    # Test case 2: Low confidence logits (potential OOD)
    low_conf_logits = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],  # Uniform distribution
        [0.5, 0.5, 0.5, 0.5],  # Very low confidence
        [0.1, 0.1, 0.1, 0.1],  # Extremely low confidence
        [0.0, 0.0, 0.0, 0.0]   # Zero confidence
    ])
    
    energy_scores2, calibrated_logits2 = energy_detector(low_conf_logits)
    
    print(f"\nLow Confidence Logits:")
    print(f"  Energy scores: {energy_scores2}")
    print(f"  Mean energy: {energy_scores2.mean().item():.3f}")
    
    # Verify behavior
    # High confidence should have lower energy scores
    # Low confidence should have higher energy scores
    high_conf_mean = energy_scores1.mean().item()
    low_conf_mean = energy_scores2.mean().item()
    
    print(f"\nEnergy Score Analysis:")
    print(f"  High confidence mean: {high_conf_mean:.3f}")
    print(f"  Low confidence mean: {low_conf_mean:.3f}")
    print(f"  Difference: {low_conf_mean - high_conf_mean:.3f}")
    
    assert low_conf_mean > high_conf_mean, "Low confidence should have higher energy scores"
    
    # Test temperature calibration
    print(f"\nTesting temperature calibration...")
    val_logits = torch.randn(10, num_classes)
    val_labels = torch.randint(0, num_classes, (10,))
    
    energy_detector.calibrate_temperature(val_logits, val_labels)
    print(f"  Temperature after calibration: {energy_detector.temperature.item():.3f}")
    
    print(f"✅ Energy-based OOD detection working correctly")

def test_prototype_distance_ood_detector():
    """Test the Prototype-Distance OOD Detection component."""
    
    print("\n" + "=" * 60)
    print("PROTOTYPE-DISTANCE OOD DETECTION TEST")
    print("=" * 60)
    
    # Initialize prototype detector
    num_classes = 4
    feature_dim = 256
    prototype_detector = PrototypeDistanceOODDetector(num_classes, feature_dim)
    
    # Test data
    batch_size = 3
    
    print(f"Testing with batch_size={batch_size}, num_classes={num_classes}, feature_dim={feature_dim}")
    print("-" * 40)
    
    # Create test features
    features = torch.randn(batch_size, feature_dim)
    
    # Test forward pass
    distances, min_distances = prototype_detector(features)
    
    print(f"Input features shape: {features.shape}")
    print(f"Distances shape: {distances.shape}")
    print(f"Min distances shape: {min_distances.shape}")
    
    # Verify shapes
    assert distances.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes})"
    assert min_distances.shape == (batch_size,), f"Expected shape ({batch_size},)"
    
    print(f"  Distances to prototypes:")
    for i in range(batch_size):
        print(f"    Sample {i+1}: {distances[i]}")
    
    print(f"  Minimum distances: {min_distances}")
    
    # Test prototype update
    print(f"\nTesting prototype update...")
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Store original prototypes
    original_prototypes = prototype_detector.prototypes.clone()
    
    # Update prototypes
    prototype_detector.update_prototypes(features, labels)
    
    # Check if prototypes changed
    prototypes_changed = not torch.allclose(original_prototypes, prototype_detector.prototypes)
    print(f"  Prototypes updated: {prototypes_changed}")
    
    # Verify prototypes are reasonable
    prototype_norms = torch.norm(prototype_detector.prototypes, dim=1)
    print(f"  Prototype norms: {prototype_norms}")
    
    assert torch.all(prototype_norms > 0), "Prototypes should have non-zero norms"
    
    print(f"✅ Prototype-distance OOD detection working correctly")

def test_late_stage_ood_detector():
    """Test the Late-Stage OOD Detection component."""
    
    print("\n" + "=" * 60)
    print("LATE-STAGE OOD DETECTION TEST")
    print("=" * 60)
    
    # Initialize late-stage detector
    num_classes = 4
    feature_dim = 256
    late_detector = LateStageOODDetector(num_classes, feature_dim)
    
    # Test data
    batch_size = 3
    
    print(f"Testing with batch_size={batch_size}, num_classes={num_classes}, feature_dim={feature_dim}")
    print("-" * 40)
    
    # Create test inputs
    logits = torch.randn(batch_size, num_classes)
    features = torch.randn(batch_size, feature_dim)
    
    # Test forward pass
    result = late_detector(logits, features)
    
    print(f"Input logits shape: {logits.shape}")
    print(f"Input features shape: {features.shape}")
    
    print(f"Detection Results:")
    print(f"  Is OOD: {result.is_ood}")
    print(f"  Energy Score: {result.energy_score:.3f}")
    print(f"  Prototype Distance: {result.prototype_distance:.3f}")
    print(f"  Combined Score: {result.combined_score:.3f}")
    print(f"  Confidence Score: {result.confidence_score:.3f}")
    print(f"  Reason: {result.reason.value if result.reason else 'None'}")
    
    # Verify result structure
    assert isinstance(result.is_ood, bool), "is_ood should be boolean"
    assert isinstance(result.energy_score, float), "energy_score should be float"
    assert isinstance(result.prototype_distance, float), "prototype_distance should be float"
    assert isinstance(result.combined_score, float), "combined_score should be float"
    assert isinstance(result.confidence_score, float), "confidence_score should be float"
    assert result.reason is None or isinstance(result.reason, OODReason), "reason should be OODReason or None"
    
    # Test combination weights
    weights = late_detector.combination_weights
    print(f"\nCombination Weights:")
    print(f"  Raw weights: {weights}")
    print(f"  Softmax weights: {F.softmax(weights, dim=0)}")
    
    print(f"✅ Late-stage OOD detection working correctly")

def test_adaptive_threshold_manager():
    """Test the Adaptive Threshold Manager component."""
    
    print("\n" + "=" * 60)
    print("ADAPTIVE THRESHOLD MANAGER TEST")
    print("=" * 60)
    
    # Initialize threshold manager
    num_languages = 7
    threshold_manager = AdaptiveThresholdManager(num_languages)
    
    print(f"Testing with {num_languages} languages")
    print("-" * 40)
    
    # Test threshold retrieval for different language-SNR combinations
    test_combinations = [
        (0, 5.0),   # Language 0, low SNR
        (1, 15.0),  # Language 1, medium SNR
        (2, 25.0),  # Language 2, high SNR
        (6, 8.0),   # Language 6, low SNR
    ]
    
    print(f"Testing threshold retrieval:")
    for language_id, snr_db in test_combinations:
        threshold = threshold_manager.get_threshold(language_id, snr_db)
        print(f"  Language {language_id}, SNR {snr_db}dB: threshold = {threshold:.3f}")
        
        # Verify threshold is reasonable
        assert 0.0 <= threshold <= 1.0, f"Threshold should be in [0, 1], got {threshold}"
    
    # Test threshold update
    print(f"\nTesting threshold update...")
    language_id = 0
    snr_range_idx = 0
    new_threshold = 0.7
    
    threshold_manager.update_thresholds(language_id, snr_range_idx, new_threshold)
    
    # Verify update
    retrieved_threshold = threshold_manager.get_threshold(language_id, 5.0)
    print(f"  Updated threshold: {retrieved_threshold:.3f}")
    print(f"  Expected threshold: {new_threshold:.3f}")
    
    assert abs(retrieved_threshold - new_threshold) < 1e-6, "Threshold update failed"
    
    # Test threshold info
    threshold_info = threshold_manager.get_threshold_info()
    print(f"\nThreshold Information:")
    print(f"  Number of languages: {threshold_info['num_languages']}")
    print(f"  SNR ranges: {threshold_info['snr_ranges']}")
    print(f"  Global threshold: {threshold_info['global_threshold']:.3f}")
    
    print(f"✅ Adaptive threshold manager working correctly")

def test_dual_gate_ood_detector():
    """Test the complete Dual-Gate OOD Detection system."""
    
    print("\n" + "=" * 60)
    print("DUAL-GATE OOD DETECTION TEST")
    print("=" * 60)
    
    # Initialize dual-gate detector
    num_classes = 4
    feature_dim = 256
    num_languages = 7
    dual_gate_detector = create_dual_gate_ood_detector(
        num_classes=num_classes,
        feature_dim=feature_dim,
        num_languages=num_languages
    )
    
    print(f"Testing with {num_classes} classes, {feature_dim} features, {num_languages} languages")
    print("-" * 40)
    
    # Test case 1: Early abstention (low SNR)
    print(f"\nTest Case 1: Early Abstention (Low SNR)")
    quality_metrics = create_quality_metrics(snr_db=3.0)  # Below 5dB threshold
    
    result1 = dual_gate_detector(
        quality_metrics=quality_metrics,
        language_id=0
    )
    
    print(f"  OOD Status: {'OOD Detected' if result1.is_ood else 'In-Domain'}")
    print(f"  Detection Stage: {result1.stage.value}")
    print(f"  Computational Savings: {'Yes' if result1.computational_savings else 'No'}")
    print(f"  Early Result: {result1.early_result.reason.value if result1.early_result and result1.early_result.reason else 'None'}")
    
    # Verify early abstention
    assert result1.is_ood, "Should detect OOD for low SNR"
    assert result1.stage == OODStage.EARLY, "Should use early stage detection"
    assert result1.computational_savings, "Should provide computational savings"
    
    # Test case 2: Late-stage detection (normal quality)
    print(f"\nTest Case 2: Late-Stage Detection (Normal Quality)")
    quality_metrics = create_quality_metrics(snr_db=25.0)  # Normal SNR
    logits = torch.randn(2, num_classes)
    features = torch.randn(2, feature_dim)
    
    result2 = dual_gate_detector(
        quality_metrics=quality_metrics,
        logits=logits,
        features=features,
        language_id=1
    )
    
    print(f"  OOD Status: {'OOD Detected' if result2.is_ood else 'In-Domain'}")
    print(f"  Detection Stage: {result2.stage.value}")
    print(f"  Computational Savings: {'Yes' if result2.computational_savings else 'No'}")
    print(f"  Late Result: {result2.late_result.reason.value if result2.late_result and result2.late_result.reason else 'None'}")
    
    # Verify late-stage detection
    assert result2.stage == OODStage.LATE, "Should use late stage detection"
    assert not result2.computational_savings, "Should not provide computational savings"
    
    # Test case 3: Outlier exposure training
    print(f"\nTest Case 3: Outlier Exposure Training")
    in_domain_logits = torch.randn(4, num_classes)
    in_domain_labels = torch.randint(0, num_classes, (4,))
    outlier_logits = torch.randn(2, num_classes)
    outlier_labels = torch.randint(0, num_classes, (2,))
    
    training_loss = dual_gate_detector.train_with_outlier_exposure(
        in_domain_logits, in_domain_labels, outlier_logits, outlier_labels
    )
    
    print(f"  Training Loss: {training_loss.item():.6f}")
    
    # Verify training loss
    assert training_loss.item() > 0, "Training loss should be positive"
    
    # Generate detection report
    print(f"\nDetection Report for Test Case 1:")
    report = dual_gate_detector.get_detection_report(result1)
    print(report)
    
    print(f"✅ Dual-gate OOD detection working correctly")

def test_computational_savings():
    """Test computational savings through early abstention."""
    
    print("\n" + "=" * 60)
    print("COMPUTATIONAL SAVINGS TEST")
    print("=" * 60)
    
    # Initialize detector
    dual_gate_detector = create_dual_gate_ood_detector(4, 256, 7)
    
    # Test scenarios with different quality levels
    test_scenarios = [
        {
            'name': 'High Quality (No Early Abstention)',
            'metrics': create_quality_metrics(snr_db=25.0, speech_prob=0.9),
            'expected_savings': False
        },
        {
            'name': 'Low SNR (Early Abstention)',
            'metrics': create_quality_metrics(snr_db=3.0, speech_prob=0.9),
            'expected_savings': True
        },
        {
            'name': 'High Clipping (Early Abstention)',
            'metrics': create_quality_metrics(snr_db=25.0, clipping_percent=35.0),
            'expected_savings': True
        },
        {
            'name': 'Low Speech Probability (Early Abstention)',
            'metrics': create_quality_metrics(snr_db=25.0, speech_prob=0.3),
            'expected_savings': True
        }
    ]
    
    print(f"Testing {len(test_scenarios)} scenarios...")
    print("-" * 40)
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}: {scenario['name']}")
        
        # Measure processing time
        start_time = time.time()
        result = dual_gate_detector(
            quality_metrics=scenario['metrics'],
            language_id=0
        )
        processing_time = time.time() - start_time
        
        print(f"  Processing time: {processing_time:.6f}s")
        print(f"  OOD detected: {result.is_ood}")
        print(f"  Stage: {result.stage.value}")
        print(f"  Computational savings: {result.computational_savings}")
        
        # Verify computational savings
        assert result.computational_savings == scenario['expected_savings'], \
            f"Computational savings mismatch for {scenario['name']}"
        
        if scenario['expected_savings']:
            assert result.stage == OODStage.EARLY, "Should use early stage for savings"
            assert result.is_ood, "Should detect OOD for early abstention"
        
        print(f"  ✅ Scenario passed")
    
    print(f"\n✅ Computational savings working correctly")

if __name__ == "__main__":
    # Test individual components
    test_early_ood_detector()
    test_energy_based_ood_detector()
    test_prototype_distance_ood_detector()
    test_late_stage_ood_detector()
    test_adaptive_threshold_manager()
    
    # Test complete system
    test_dual_gate_ood_detector()
    test_computational_savings()
    
    print(f"\n" + "=" * 80)
    print("DUAL-GATE OOD DETECTION TESTING COMPLETE")
    print("=" * 80)
    print("✅ Early-stage OOD detection with abstention")
    print("✅ Energy-based OOD detection with temperature scaling")
    print("✅ Prototype-distance OOD detection with Mahalanobis distance")
    print("✅ Late-stage OOD detection with score combination")
    print("✅ Adaptive thresholds for per-language and per-SNR")
    print("✅ Outlier exposure training support")
    print("✅ Computational savings through early abstention")
    print("✅ Ready for integration with main pipeline")
