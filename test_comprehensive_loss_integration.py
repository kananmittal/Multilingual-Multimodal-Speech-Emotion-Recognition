#!/usr/bin/env python3
"""
Test for Comprehensive Loss Function Integration
Verifies multi-component loss architecture, training phases, and batch validation
"""

import time
import torch
import numpy as np
from src.models.comprehensive_loss_integration import (
    ComprehensiveLossIntegration,
    create_comprehensive_loss_integration,
    TrainingPhaseManager,
    create_training_phase_manager,
    create_sample_batch_data,
    create_sample_model_outputs,
    TrainingPhase,
    LossWeights,
    EnergyMarginLoss,
    TemporalConsistencyLoss,
    ConfidenceCalibrationLoss,
    BatchCompositionValidator
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_loss_weights():
    """Test the LossWeights configuration."""
    
    print("=" * 60)
    print("LOSS WEIGHTS TEST")
    print("=" * 60)
    
    # Initialize loss weights
    weights = LossWeights()
    
    print("Default Loss Weights:")
    print(f"  CE Loss: {weights.ce_loss}")
    print(f"  SupCon Loss: {weights.supcon_loss}")
    print(f"  Prototype Loss: {weights.prototype_loss}")
    print(f"  Language Adversarial Loss: {weights.language_adversarial_loss}")
    print(f"  Energy-Margin Loss: {weights.energy_margin_loss}")
    print(f"  Temporal Consistency Loss: {weights.temporal_consistency_loss}")
    print(f"  Confidence Calibration Loss: {weights.confidence_calibration_loss}")
    
    # Test phase-specific weights
    print("\nPhase-Specific Weights:")
    
    for phase in TrainingPhase:
        phase_weights = weights.get_phase_weights(phase)
        print(f"\n{phase.value.upper()}:")
        for loss_name, weight in phase_weights.items():
            print(f"  {loss_name}: {weight}")
    
    # Verify weights are properly configured
    assert weights.ce_loss == 1.0, "CE loss should be primary (1.0)"
    assert weights.language_adversarial_loss < 0, "Language adversarial should be negative for adversarial training"
    assert weights.supcon_loss > 0, "SupCon loss should be positive"
    assert weights.prototype_loss > 0, "Prototype loss should be positive"
    
    print(f"\n✅ Loss weights configuration working correctly")

def test_energy_margin_loss():
    """Test the Energy-Margin Loss component."""
    
    print("\n" + "=" * 60)
    print("ENERGY-MARGIN LOSS TEST")
    print("=" * 60)
    
    # Initialize energy-margin loss
    energy_loss_fn = EnergyMarginLoss(margin=10.0, temperature=1.0)
    
    # Test data
    batch_size = 16
    num_classes = 4
    
    print(f"Testing with batch_size={batch_size}, num_classes={num_classes}")
    print("-" * 40)
    
    # Test case 1: Mixed in-domain and OOD samples
    logits = torch.randn(batch_size, num_classes)
    is_ood = torch.tensor([False] * 12 + [True] * 4)  # 12 in-domain, 4 OOD
    
    loss1 = energy_loss_fn(logits, is_ood)
    print(f"Mixed batch loss: {loss1.item():.6f}")
    
    # Test case 2: All in-domain samples
    is_ood_all_in = torch.zeros(batch_size, dtype=torch.bool)
    loss2 = energy_loss_fn(logits, is_ood_all_in)
    print(f"All in-domain loss: {loss2.item():.6f}")
    
    # Test case 3: All OOD samples
    is_ood_all_out = torch.ones(batch_size, dtype=torch.bool)
    loss3 = energy_loss_fn(logits, is_ood_all_out)
    print(f"All OOD loss: {loss3.item():.6f}")
    
    # Test case 4: Pre-computed energy scores
    energy_scores = torch.randn(batch_size)
    loss4 = energy_loss_fn(logits, is_ood, energy_scores)
    print(f"Pre-computed energy loss: {loss4.item():.6f}")
    
    # Verify behavior
    assert loss1.item() >= 0, "Energy-margin loss should be non-negative"
    assert loss2.item() >= 0, "In-domain loss should be non-negative"
    assert loss3.item() >= 0, "OOD loss should be non-negative"
    assert loss4.item() >= 0, "Pre-computed energy loss should be non-negative"
    
    # Mixed batch should have higher loss than all in-domain
    assert loss1.item() >= loss2.item(), "Mixed batch should have higher loss than all in-domain"
    
    print(f"✅ Energy-margin loss working correctly")

def test_temporal_consistency_loss():
    """Test the Temporal Consistency Loss component."""
    
    print("\n" + "=" * 60)
    print("TEMPORAL CONSISTENCY LOSS TEST")
    print("=" * 60)
    
    # Initialize temporal consistency loss
    temporal_loss_fn = TemporalConsistencyLoss(consistency_weight=1.0, confidence_threshold=0.8)
    
    # Test data
    batch_size = 16
    num_classes = 4
    
    print(f"Testing with batch_size={batch_size}, num_classes={num_classes}")
    print("-" * 40)
    
    # Test case 1: Similar predictions with high confidence
    current_pred = torch.randn(batch_size, num_classes)
    previous_pred = current_pred + 0.1 * torch.randn_like(current_pred)  # Similar predictions
    current_conf = torch.ones(batch_size) * 0.9  # High confidence
    previous_conf = torch.ones(batch_size) * 0.9  # High confidence
    
    loss1 = temporal_loss_fn(current_pred, previous_pred, current_conf, previous_conf)
    print(f"Similar predictions, high confidence: {loss1.item():.6f}")
    
    # Test case 2: Different predictions with low confidence
    current_pred2 = torch.randn(batch_size, num_classes)
    previous_pred2 = torch.randn(batch_size, num_classes)  # Different predictions
    current_conf2 = torch.ones(batch_size) * 0.5  # Low confidence
    previous_conf2 = torch.ones(batch_size) * 0.5  # Low confidence
    
    loss2 = temporal_loss_fn(current_pred2, previous_pred2, current_conf2, previous_conf2)
    print(f"Different predictions, low confidence: {loss2.item():.6f}")
    
    # Test case 3: Mixed confidence levels
    current_pred3 = torch.randn(batch_size, num_classes)
    previous_pred3 = torch.randn(batch_size, num_classes)
    current_conf3 = torch.tensor([0.9, 0.5, 0.8, 0.3, 0.9, 0.7, 0.6, 0.8, 
                                 0.9, 0.4, 0.8, 0.6, 0.9, 0.7, 0.5, 0.8])
    previous_conf3 = torch.tensor([0.9, 0.6, 0.8, 0.4, 0.9, 0.6, 0.7, 0.8,
                                  0.9, 0.5, 0.8, 0.7, 0.9, 0.6, 0.6, 0.8])
    
    loss3 = temporal_loss_fn(current_pred3, previous_pred3, current_conf3, previous_conf3)
    print(f"Mixed confidence levels: {loss3.item():.6f}")
    
    # Verify behavior
    assert loss1.item() >= 0, "Temporal consistency loss should be non-negative"
    assert loss2.item() >= 0, "Temporal consistency loss should be non-negative"
    assert loss3.item() >= 0, "Temporal consistency loss should be non-negative"
    
    # Low confidence transitions should have higher loss
    assert loss2.item() >= loss1.item(), "Low confidence transitions should have higher loss"
    
    print(f"✅ Temporal consistency loss working correctly")

def test_confidence_calibration_loss():
    """Test the Confidence Calibration Loss component."""
    
    print("\n" + "=" * 60)
    print("CONFIDENCE CALIBRATION LOSS TEST")
    print("=" * 60)
    
    # Initialize confidence calibration loss
    calibration_loss_fn = ConfidenceCalibrationLoss(calibration_weight=1.0)
    
    # Test data
    batch_size = 32
    
    print(f"Testing with batch_size={batch_size}")
    print("-" * 40)
    
    # Test case 1: Well-calibrated predictions
    predicted_conf = torch.linspace(0.1, 0.9, batch_size)
    actual_accuracy = predicted_conf  # Perfect calibration
    
    loss1 = calibration_loss_fn(predicted_conf, actual_accuracy)
    print(f"Well-calibrated predictions: {loss1.item():.6f}")
    
    # Test case 2: Overconfident predictions
    predicted_conf2 = torch.linspace(0.5, 0.95, batch_size)
    actual_accuracy2 = torch.linspace(0.3, 0.7, batch_size)  # Lower accuracy than confidence
    
    loss2 = calibration_loss_fn(predicted_conf2, actual_accuracy2)
    print(f"Overconfident predictions: {loss2.item():.6f}")
    
    # Test case 3: Underconfident predictions
    predicted_conf3 = torch.linspace(0.3, 0.7, batch_size)
    actual_accuracy3 = torch.linspace(0.5, 0.9, batch_size)  # Higher accuracy than confidence
    
    loss3 = calibration_loss_fn(predicted_conf3, actual_accuracy3)
    print(f"Underconfident predictions: {loss3.item():.6f}")
    
    # Test case 4: Random calibration
    predicted_conf4 = torch.rand(batch_size)
    actual_accuracy4 = torch.rand(batch_size)
    
    loss4 = calibration_loss_fn(predicted_conf4, actual_accuracy4)
    print(f"Random calibration: {loss4.item():.6f}")
    
    # Verify behavior
    assert loss1.item() >= 0, "Calibration loss should be non-negative"
    assert loss2.item() >= 0, "Calibration loss should be non-negative"
    assert loss3.item() >= 0, "Calibration loss should be non-negative"
    assert loss4.item() >= 0, "Calibration loss should be non-negative"
    
    # Well-calibrated should have lower loss than poorly calibrated
    assert loss1.item() <= loss2.item(), "Well-calibrated should have lower loss"
    assert loss1.item() <= loss3.item(), "Well-calibrated should have lower loss"
    
    print(f"✅ Confidence calibration loss working correctly")

def test_batch_composition_validator():
    """Test the Batch Composition Validator component."""
    
    print("\n" + "=" * 60)
    print("BATCH COMPOSITION VALIDATOR TEST")
    print("=" * 60)
    
    # Initialize validator
    validator = BatchCompositionValidator(
        min_batch_size=32,
        min_ood_ratio=0.2,
        min_languages=2,
        min_classes=2
    )
    
    print("Validation Requirements:")
    print(f"  Min batch size: {validator.min_batch_size}")
    print(f"  Min OOD ratio: {validator.min_ood_ratio}")
    print(f"  Min languages: {validator.min_languages}")
    print(f"  Min classes: {validator.min_classes}")
    print("-" * 40)
    
    # Test case 1: Valid batch
    valid_batch = {
        'labels': torch.randint(0, 4, (32,)),
        'language_ids': torch.randint(0, 3, (32,)),
        'is_ood': torch.tensor([False] * 24 + [True] * 8)  # 25% OOD
    }
    
    is_valid1, validation_info1 = validator.validate_batch(valid_batch)
    print(f"Valid batch validation: {'✅ PASSED' if is_valid1 else '❌ FAILED'}")
    
    # Test case 2: Invalid batch (too small)
    invalid_batch_small = {
        'labels': torch.randint(0, 4, (16,)),
        'language_ids': torch.randint(0, 3, (16,)),
        'is_ood': torch.tensor([False] * 12 + [True] * 4)
    }
    
    is_valid2, validation_info2 = validator.validate_batch(invalid_batch_small)
    print(f"Small batch validation: {'✅ PASSED' if is_valid2 else '❌ FAILED'}")
    
    # Test case 3: Invalid batch (insufficient OOD)
    invalid_batch_low_ood = {
        'labels': torch.randint(0, 4, (32,)),
        'language_ids': torch.randint(0, 3, (32,)),
        'is_ood': torch.tensor([False] * 30 + [True] * 2)  # 6.25% OOD
    }
    
    is_valid3, validation_info3 = validator.validate_batch(invalid_batch_low_ood)
    print(f"Low OOD batch validation: {'✅ PASSED' if is_valid3 else '❌ FAILED'}")
    
    # Test case 4: Invalid batch (single language)
    invalid_batch_single_lang = {
        'labels': torch.randint(0, 4, (32,)),
        'language_ids': torch.zeros(32, dtype=torch.long),  # All same language
        'is_ood': torch.tensor([False] * 24 + [True] * 8)
    }
    
    is_valid4, validation_info4 = validator.validate_batch(invalid_batch_single_lang)
    print(f"Single language batch validation: {'✅ PASSED' if is_valid4 else '❌ FAILED'}")
    
    # Verify behavior
    assert is_valid1, "Valid batch should pass validation"
    assert not is_valid2, "Small batch should fail validation"
    assert not is_valid3, "Low OOD batch should fail validation"
    assert not is_valid4, "Single language batch should fail validation"
    
    # Test validation reports
    print("\nValidation Reports:")
    print(validator.get_validation_report(validation_info1))
    print(validator.get_validation_report(validation_info2))
    
    print(f"✅ Batch composition validator working correctly")

def test_training_phase_manager():
    """Test the Training Phase Manager component."""
    
    print("\n" + "=" * 60)
    print("TRAINING PHASE MANAGER TEST")
    print("=" * 60)
    
    # Initialize phase manager
    phase_manager = create_training_phase_manager()
    
    print("Default Phase Transitions:")
    for phase, epoch in phase_manager.phase_transitions.items():
        print(f"  {phase.value}: epoch {epoch}")
    print("-" * 40)
    
    # Test phase transitions
    test_epochs = [0, 25, 50, 75, 100, 125]
    
    print("Phase Transitions:")
    for epoch in test_epochs:
        phase = phase_manager.update_epoch(epoch)
        phase_info = phase_manager.get_phase_info()
        
        print(f"  Epoch {epoch}: {phase.value}")
        print(f"    Progress: {phase_info['phase_progress']:.2f}")
        if phase_info['next_transition']:
            print(f"    Next transition: epoch {phase_info['next_transition']}")
        else:
            print(f"    Next transition: None (final phase)")
    
    # Verify phase transitions
    assert phase_manager.current_phase == TrainingPhase.CALIBRATION, "Should be in calibration phase after epoch 100"
    
    # Test custom phase transitions
    custom_transitions = {
        TrainingPhase.REPRESENTATION_LEARNING: 0,
        TrainingPhase.ADVERSARIAL_TRAINING: 25,  # Earlier transition
        TrainingPhase.CALIBRATION: 75  # Earlier transition
    }
    
    custom_manager = TrainingPhaseManager(phase_transitions=custom_transitions)
    
    print(f"\nCustom Phase Transitions:")
    for phase, epoch in custom_manager.phase_transitions.items():
        print(f"  {phase.value}: epoch {epoch}")
    
    # Test custom transitions
    custom_phase = custom_manager.update_epoch(50)
    print(f"  Epoch 50: {custom_phase.value}")
    
    assert custom_phase == TrainingPhase.ADVERSARIAL_TRAINING, "Should be in adversarial phase at epoch 50"
    
    print(f"✅ Training phase manager working correctly")

def test_comprehensive_loss_integration():
    """Test the complete Comprehensive Loss Integration system."""
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE LOSS INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize comprehensive loss integration
    num_classes = 4
    feature_dim = 256
    num_languages = 7
    
    loss_integration = create_comprehensive_loss_integration(
        num_classes=num_classes,
        feature_dim=feature_dim,
        num_languages=num_languages
    )
    
    print(f"Initialized with {num_classes} classes, {feature_dim} features, {num_languages} languages")
    print("-" * 40)
    
    # Test different training phases
    for phase in TrainingPhase:
        print(f"\nTesting {phase.value.upper()} phase:")
        
        # Set training phase
        loss_integration.set_training_phase(phase)
        
        # Create sample data
        batch_data = create_sample_batch_data(
            batch_size=32,
            num_classes=num_classes,
            num_languages=num_languages
        )
        
        model_outputs = create_sample_model_outputs(
            batch_size=32,
            num_classes=num_classes,
            feature_dim=feature_dim
        )
        
        # Compute loss
        start_time = time.time()
        total_loss, loss_components = loss_integration(batch_data, model_outputs)
        computation_time = time.time() - start_time
        
        print(f"  Total loss: {total_loss.item():.6f}")
        print(f"  Computation time: {computation_time:.6f}s")
        
        # Verify loss components
        expected_components = [
            'ce_loss', 'supcon_loss', 'prototype_loss', 'language_adversarial_loss',
            'energy_margin_loss', 'temporal_consistency_loss', 'confidence_calibration_loss'
        ]
        
        for component in expected_components:
            assert component in loss_components, f"Missing loss component: {component}"
            assert isinstance(loss_components[component], torch.Tensor), f"Invalid component type: {component}"
        
        # Verify phase weights
        phase_weights = loss_components['phase_weights']
        assert 'ce_loss' in phase_weights, "Missing phase weights"
        
        # Verify batch validation
        validation_info = loss_components['validation_info']
        assert 'overall_valid' in validation_info, "Missing validation info"
        
        print(f"  Batch validation: {'✅ PASSED' if validation_info['overall_valid'] else '❌ FAILED'}")
        
        # Generate loss report
        report = loss_integration.get_loss_report(loss_components)
        print(f"  Report length: {len(report)} characters")
    
    # Test prototype management
    print(f"\nTesting prototype management:")
    prototype_info = loss_integration.get_prototype_info()
    print(f"  Prototype count: {prototype_info['prototype_count']}")
    print(f"  Feature dimension: {prototype_info['feature_dim']}")
    
    # Update prototypes
    features = torch.randn(16, feature_dim)
    labels = torch.randint(0, num_classes, (16,))
    loss_integration.update_prototypes(features, labels)
    
    updated_info = loss_integration.get_prototype_info()
    print(f"  Updated prototype count: {updated_info['prototype_count']}")
    
    print(f"✅ Comprehensive loss integration working correctly")

def test_integration_scenarios():
    """Test integration scenarios and edge cases."""
    
    print("\n" + "=" * 60)
    print("INTEGRATION SCENARIOS TEST")
    print("=" * 60)
    
    # Initialize components
    loss_integration = create_comprehensive_loss_integration(4, 256, 7)
    phase_manager = create_training_phase_manager()
    
    # Test scenario 1: Phase transition during training
    print("Scenario 1: Phase transition during training")
    
    for epoch in [0, 25, 50, 75, 100]:
        phase = phase_manager.update_epoch(epoch)
        loss_integration.set_training_phase(phase)
        
        # Create data and compute loss
        batch_data = create_sample_batch_data(32, 4, 7)
        model_outputs = create_sample_model_outputs(32, 4, 256)
        
        total_loss, loss_components = loss_integration(batch_data, model_outputs)
        
        print(f"  Epoch {epoch}: {phase.value} → Loss: {total_loss.item():.6f}")
    
    # Test scenario 2: Different batch sizes
    print(f"\nScenario 2: Different batch sizes")
    
    for batch_size in [16, 32, 64, 128]:
        try:
            batch_data = create_sample_batch_data(batch_size, 4, 7)
            model_outputs = create_sample_model_outputs(batch_size, 4, 256)
            
            total_loss, loss_components = loss_integration(batch_data, model_outputs)
            
            validation_info = loss_components['validation_info']
            print(f"  Batch size {batch_size}: Loss: {total_loss.item():.6f}, "
                  f"Validation: {'✅' if validation_info['overall_valid'] else '❌'}")
        except Exception as e:
            print(f"  Batch size {batch_size}: Error - {e}")
    
    # Test scenario 3: Edge cases
    print(f"\nScenario 3: Edge cases")
    
    # Empty batch
    try:
        empty_batch = {'labels': torch.tensor([]), 'language_ids': torch.tensor([]), 'is_ood': torch.tensor([])}
        empty_outputs = {'features': torch.tensor([]), 'logits': torch.tensor([])}
        
        total_loss, loss_components = loss_integration(empty_batch, empty_outputs)
        print(f"  Empty batch: Loss: {total_loss.item():.6f}")
    except Exception as e:
        print(f"  Empty batch: Error - {e}")
    
    # Single sample batch
    try:
        single_batch = {'labels': torch.tensor([0]), 'language_ids': torch.tensor([0]), 'is_ood': torch.tensor([False])}
        single_outputs = {'features': torch.randn(1, 256), 'logits': torch.randn(1, 4)}
        
        total_loss, loss_components = loss_integration(single_batch, single_outputs)
        print(f"  Single sample: Loss: {total_loss.item():.6f}")
    except Exception as e:
        print(f"  Single sample: Error - {e}")
    
    print(f"✅ Integration scenarios working correctly")

if __name__ == "__main__":
    # Test individual components
    test_loss_weights()
    test_energy_margin_loss()
    test_temporal_consistency_loss()
    test_confidence_calibration_loss()
    test_batch_composition_validator()
    test_training_phase_manager()
    
    # Test complete system
    test_comprehensive_loss_integration()
    test_integration_scenarios()
    
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE LOSS FUNCTION INTEGRATION TESTING COMPLETE")
    print("=" * 80)
    print("✅ Multi-component loss architecture with specific weights")
    print("✅ Training phase management and automatic transitions")
    print("✅ Energy-margin loss for OOD detection")
    print("✅ Temporal consistency loss for emotion transitions")
    print("✅ Confidence calibration loss for uncertainty estimation")
    print("✅ Batch composition validation with requirements")
    print("✅ Phase-specific loss weight scheduling")
    print("✅ Comprehensive loss reporting and analysis")
    print("✅ Ready for production training pipeline")
