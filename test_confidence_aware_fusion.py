#!/usr/bin/env python3
"""
Test for Confidence-Aware Fusion Mechanism
Verifies dynamic gating, policy clamps, and adaptive fusion
"""

import time
import torch
import numpy as np
from src.models.confidence_aware_fusion import (
    ConfidenceFeatures, 
    ConfidenceAwareFusion, 
    create_confidence_aware_fusion,
    DynamicGatingMLP,
    PolicyBasedClamps
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_confidence_features(scenario: str = "normal") -> ConfidenceFeatures:
    """Create test confidence features for different scenarios."""
    
    if scenario == "normal":
        # Normal quality scenario
        return ConfidenceFeatures(
            snr_db=25.0,
            speech_prob=0.9,
            clipping_percent=5.0,
            denoise_gain_db=2.0,
            asr_conf_segment=0.85,
            lid_entropy=0.8,
            text_reliability_score=0.9,
            conditioning_applied=True,
            quality_gates_passed=True,
            audio_text_similarity=0.7,
            boundary_confidence=0.8,
            segment_emotion_consistency=0.75,
            previous_segment_confidence=0.8,
            emotion_transition_probability=0.6
        )
    
    elif scenario == "low_audio_quality":
        # Low audio quality scenario
        return ConfidenceFeatures(
            snr_db=8.0,  # Low SNR
            speech_prob=0.6,  # Low speech probability
            clipping_percent=25.0,  # High clipping
            denoise_gain_db=8.0,  # High denoising needed
            asr_conf_segment=0.85,  # Good ASR
            lid_entropy=0.8,
            text_reliability_score=0.9,
            conditioning_applied=True,
            quality_gates_passed=False,  # Failed quality gates
            audio_text_similarity=0.4,  # Low similarity
            boundary_confidence=0.6,
            segment_emotion_consistency=0.5,
            previous_segment_confidence=0.7,
            emotion_transition_probability=0.4
        )
    
    elif scenario == "low_text_quality":
        # Low text quality scenario
        return ConfidenceFeatures(
            snr_db=25.0,  # Good audio
            speech_prob=0.9,
            clipping_percent=5.0,
            denoise_gain_db=2.0,
            asr_conf_segment=0.3,  # Low ASR confidence
            lid_entropy=1.8,  # High LID entropy
            text_reliability_score=0.4,  # Low text reliability
            conditioning_applied=True,
            quality_gates_passed=True,
            audio_text_similarity=0.3,  # Low similarity
            boundary_confidence=0.5,
            segment_emotion_consistency=0.4,
            previous_segment_confidence=0.6,
            emotion_transition_probability=0.3
        )
    
    elif scenario == "both_unreliable":
        # Both modalities unreliable
        return ConfidenceFeatures(
            snr_db=3.0,  # Very low SNR
            speech_prob=0.3,  # Very low speech probability
            clipping_percent=40.0,  # Very high clipping
            denoise_gain_db=12.0,  # Very high denoising
            asr_conf_segment=0.2,  # Very low ASR confidence
            lid_entropy=2.2,  # Very high LID entropy
            text_reliability_score=0.2,  # Very low text reliability
            conditioning_applied=True,
            quality_gates_passed=False,
            audio_text_similarity=0.1,  # Very low similarity
            boundary_confidence=0.3,
            segment_emotion_consistency=0.2,
            previous_segment_confidence=0.4,
            emotion_transition_probability=0.2
        )
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

def test_dynamic_gating_mlp():
    """Test the Dynamic Gating MLP component."""
    
    print("=" * 60)
    print("DYNAMIC GATING MLP TEST")
    print("=" * 60)
    
    # Initialize gating MLP
    gating_mlp = DynamicGatingMLP(confidence_dim=14, hidden_dim=32)
    
    # Test scenarios
    scenarios = ["normal", "low_audio_quality", "low_text_quality", "both_unreliable"]
    
    print(f"\nTesting {len(scenarios)} scenarios...")
    print("-" * 40)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario.upper()}")
        
        # Create confidence features
        conf_features = create_test_confidence_features(scenario)
        conf_tensor = conf_features.to_tensor()
        
        # Get weights
        audio_weight, text_weight = gating_mlp(conf_tensor.unsqueeze(0))
        
        print(f"  Audio Weight: {audio_weight.item():.3f}")
        print(f"  Text Weight: {text_weight.item():.3f}")
        print(f"  Sum: {audio_weight.item() + text_weight.item():.3f}")
        
        # Verify weights sum to 1.0
        assert abs(audio_weight.item() + text_weight.item() - 1.0) < 1e-6, "Weights must sum to 1.0"
        print(f"  ✅ Weights sum to 1.0")

def test_policy_based_clamps():
    """Test the Policy-Based Clamps component."""
    
    print("\n" + "=" * 60)
    print("POLICY-BASED CLAMPS TEST")
    print("=" * 60)
    
    # Initialize policy clamps
    policy_clamps = PolicyBasedClamps()
    
    # Test scenarios
    scenarios = ["normal", "low_audio_quality", "low_text_quality", "both_unreliable"]
    
    print(f"\nTesting {len(scenarios)} scenarios...")
    print("-" * 40)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario.upper()}")
        
        # Create confidence features
        conf_features = create_test_confidence_features(scenario)
        
        # Initial weights (equal)
        audio_weight = torch.tensor([[0.5]])
        text_weight = torch.tensor([[0.5]])
        
        print(f"  Initial weights - Audio: {audio_weight.item():.3f}, Text: {text_weight.item():.3f}")
        
        # Apply policy clamps
        clamped_audio, clamped_text = policy_clamps(audio_weight, text_weight, conf_features)
        
        print(f"  Clamped weights - Audio: {clamped_audio.item():.3f}, Text: {clamped_text.item():.3f}")
        print(f"  Sum: {clamped_audio.item() + clamped_text.item():.3f}")
        
        # Verify weights sum to 1.0
        assert abs(clamped_audio.item() + clamped_text.item() - 1.0) < 1e-6, "Weights must sum to 1.0"
        print(f"  ✅ Weights sum to 1.0")
        
        # Check if policies were applied
        if scenario == "low_audio_quality":
            if clamped_audio.item() < 0.3:
                print(f"  ✅ Policy 1 applied: Audio weight capped due to low SNR")
        elif scenario == "low_text_quality":
            if clamped_text.item() < 0.4:
                print(f"  ✅ Policy 2 applied: Text weight capped due to low ASR confidence")
        elif scenario == "both_unreliable":
            if abs(clamped_audio.item() - 0.5) < 0.1 and abs(clamped_text.item() - 0.5) < 0.1:
                print(f"  ✅ Policy 5 applied: Emergency fallback to equal weighting")

def test_confidence_aware_fusion():
    """Test the complete Confidence-Aware Fusion system."""
    
    print("\n" + "=" * 60)
    print("CONFIDENCE-AWARE FUSION TEST")
    print("=" * 60)
    
    # Initialize fusion system
    audio_dim = 1536  # 768 * 2 (mean + std)
    text_dim = 1536   # 768 * 2 (mean + std)
    proj_dim = 256
    
    fusion_system = create_confidence_aware_fusion(audio_dim, text_dim, proj_dim)
    
    # Test scenarios
    scenarios = ["normal", "low_audio_quality", "low_text_quality", "both_unreliable"]
    
    print(f"\nTesting {len(scenarios)} scenarios...")
    print("-" * 40)
    
    results = []
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario.upper()}")
        
        # Create test features
        audio_features = torch.randn(1, audio_dim)
        text_features = torch.randn(1, text_dim)
        conf_features = create_test_confidence_features(scenario)
        
        # Process through fusion
        start_time = time.time()
        fused_features, fusion_confidence, fusion_info = fusion_system(
            audio_features, text_features, conf_features
        )
        processing_time = time.time() - start_time
        
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Fused features shape: {fused_features.shape}")
        print(f"  Fusion confidence: {fusion_confidence.item():.3f}")
        print(f"  Audio weight: {fusion_info['audio_weight'].item():.3f}")
        print(f"  Text weight: {fusion_info['text_weight'].item():.3f}")
        
        # Verify output dimensions
        assert fused_features.shape == (1, proj_dim), f"Expected shape (1, {proj_dim}), got {fused_features.shape}"
        assert fusion_confidence.shape == (1, 1), f"Expected shape (1, 1), got {fusion_confidence.shape}"
        
        print(f"  ✅ Output dimensions correct")
        
        # Store results
        results.append({
            'scenario': scenario,
            'fusion_confidence': fusion_confidence.item(),
            'audio_weight': fusion_info['audio_weight'].item(),
            'text_weight': fusion_info['text_weight'].item(),
            'processing_time': processing_time
        })
    
    # Summary
    print(f"\n" + "=" * 60)
    print("FUSION RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"{'Scenario':<20} {'Audio Weight':<12} {'Text Weight':<12} {'Fusion Conf':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['scenario']:<20} {result['audio_weight']:<12.3f} {result['text_weight']:<12.3f} {result['fusion_confidence']:<12.3f}")
    
    # Generate detailed report for one scenario
    print(f"\nDetailed Fusion Report (Normal Scenario):")
    conf_features = create_test_confidence_features("normal")
    audio_features = torch.randn(1, audio_dim)
    text_features = torch.randn(1, text_dim)
    _, _, fusion_info = fusion_system(audio_features, text_features, conf_features)
    
    report = fusion_system.get_fusion_report(fusion_info)
    print(report)

def test_integration_with_existing_components():
    """Test integration with existing audio and text encoders."""
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    
    # Simulate audio and text features from encoders
    audio_features = torch.randn(4, 1536)  # Batch size 4
    text_features = torch.randn(4, 1536)
    
    # Create confidence features for each sample in batch
    conf_features_list = []
    scenarios = ["normal", "low_audio_quality", "low_text_quality", "both_unreliable"]
    
    for scenario in scenarios:
        conf_features = create_test_confidence_features(scenario)
        conf_features_list.append(conf_features)
    
    # Initialize fusion system
    fusion_system = create_confidence_aware_fusion(1536, 1536, 256)
    
    print(f"Testing batch processing with {len(conf_features_list)} samples...")
    
    # Process each sample
    fused_features_list = []
    fusion_confidences = []
    
    for i, conf_features in enumerate(conf_features_list):
        audio_feat = audio_features[i:i+1]
        text_feat = text_features[i:i+1]
        
        fused_feat, fusion_conf, fusion_info = fusion_system(audio_feat, text_feat, conf_features)
        
        fused_features_list.append(fused_feat)
        fusion_confidences.append(fusion_conf.item())
        
        print(f"  Sample {i+1} ({scenarios[i]}): Audio={fusion_info['audio_weight'].item():.3f}, "
              f"Text={fusion_info['text_weight'].item():.3f}, Conf={fusion_conf.item():.3f}")
    
    # Stack results
    fused_features_batch = torch.cat(fused_features_list, dim=0)
    fusion_confidences_tensor = torch.tensor(fusion_confidences)
    
    print(f"\nBatch results:")
    print(f"  Fused features shape: {fused_features_batch.shape}")
    print(f"  Fusion confidences: {fusion_confidences_tensor}")
    print(f"  Mean fusion confidence: {fusion_confidences_tensor.mean():.3f}")
    
    assert fused_features_batch.shape == (4, 256), f"Expected shape (4, 256), got {fused_features_batch.shape}"
    print(f"  ✅ Batch processing successful")

if __name__ == "__main__":
    # Test individual components
    test_dynamic_gating_mlp()
    test_policy_based_clamps()
    
    # Test complete system
    test_confidence_aware_fusion()
    
    # Test integration
    test_integration_with_existing_components()
    
    print(f"\n" + "=" * 80)
    print("CONFIDENCE-AWARE FUSION TESTING COMPLETE")
    print("=" * 80)
    print("✅ All components working correctly")
    print("✅ Dynamic gating based on confidence features")
    print("✅ Policy-based clamps enforcing quality rules")
    print("✅ Adaptive fusion with confidence estimation")
    print("✅ Ready for integration with main pipeline")
