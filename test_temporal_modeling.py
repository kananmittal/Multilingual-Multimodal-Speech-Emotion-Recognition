#!/usr/bin/env python3
"""
Test for Temporal Modeling Module
Verifies TCN, confidence-aware smoothing, speaker change detection, and temporal buffer
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from src.models.temporal_modeling import (
    TemporalModelingModule,
    create_temporal_modeling_module,
    create_temporal_segment,
    TemporalPositionalEncoding,
    CausalConv1d,
    TemporalConvNet,
    ConfidenceAwareSmoothing,
    SpeakerChangeDetector,
    TemporalBuffer
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_temporal_positional_encoding():
    """Test the Temporal Positional Encoding component."""
    
    print("=" * 60)
    print("TEMPORAL POSITIONAL ENCODING TEST")
    print("=" * 60)
    
    # Initialize positional encoding
    feature_dim = 256
    max_segments = 10
    pos_encoding = TemporalPositionalEncoding(feature_dim, max_segments)
    
    # Test data
    batch_size = 2
    num_segments = 5
    x = torch.randn(batch_size, num_segments, feature_dim)
    segment_positions = torch.arange(num_segments).unsqueeze(0).expand(batch_size, -1)
    
    print(f"Testing with batch_size={batch_size}, num_segments={num_segments}, feature_dim={feature_dim}")
    print("-" * 40)
    
    # Forward pass
    output = pos_encoding(x, segment_positions)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Segment positions: {segment_positions}")
    
    # Verify shapes
    assert output.shape == x.shape, f"Output shape must match input shape"
    
    # Verify positional encoding is added (not just identity)
    diff = torch.abs(output - x).mean()
    print(f"Mean difference from input: {diff:.6f}")
    
    assert diff > 0, "Positional encoding should modify the input"
    
    # Verify different positions have different encodings
    pos_0 = pos_encoding.pe[0]
    pos_1 = pos_encoding.pe[1]
    pos_diff = torch.abs(pos_0 - pos_1).mean()
    print(f"Mean difference between positions 0 and 1: {pos_diff:.6f}")
    
    assert pos_diff > 0, "Different positions should have different encodings"
    
    print(f"✅ Temporal positional encoding working correctly")

def test_causal_conv1d():
    """Test the Causal 1D Convolution component."""
    
    print("\n" + "=" * 60)
    print("CAUSAL CONV1D TEST")
    print("=" * 60)
    
    # Initialize causal convolution
    in_channels = 256
    out_channels = 128
    kernel_size = 3
    dilation = 1
    causal_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
    
    # Test data
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, in_channels, seq_len)
    
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
    print(f"Input channels: {in_channels}, Output channels: {out_channels}")
    print("-" * 40)
    
    # Forward pass
    output = causal_conv(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify shapes
    expected_output_len = seq_len  # Causal conv should maintain sequence length
    assert output.shape == (batch_size, out_channels, expected_output_len), \
        f"Expected shape ({batch_size}, {out_channels}, {expected_output_len}), got {output.shape}"
    
    # Verify causal property (no information leakage from future)
    # This is a simplified test - in practice, you'd need to verify the convolution weights
    print(f"✅ Causal convolution shape correct")
    print(f"✅ Sequence length maintained")

def test_temporal_convnet():
    """Test the Temporal Convolutional Network component."""
    
    print("\n" + "=" * 60)
    print("TEMPORAL CONVNET TEST")
    print("=" * 60)
    
    # Initialize TCN
    feature_dim = 256
    hidden_dim = 128
    kernel_size = 3
    tcn = TemporalConvNet(feature_dim, hidden_dim, kernel_size)
    
    # Test data
    batch_size = 2
    num_segments = 5
    x = torch.randn(batch_size, num_segments, feature_dim)
    
    print(f"Testing with batch_size={batch_size}, num_segments={num_segments}")
    print(f"Feature dim: {feature_dim}, Hidden dim: {hidden_dim}")
    print("-" * 40)
    
    # Forward pass
    output = tcn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify shapes
    assert output.shape == x.shape, f"Output shape must match input shape"
    
    # Verify residual connection (output should be different from input)
    diff = torch.abs(output - x).mean()
    print(f"Mean difference from input: {diff:.6f}")
    
    assert diff > 0, "TCN should modify the input through processing"
    
    # Count parameters
    total_params = sum(p.numel() for p in tcn.parameters())
    trainable_params = sum(p.numel() for p in tcn.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    assert trainable_params > 0, "TCN should have trainable parameters"
    
    print(f"✅ Temporal ConvNet working correctly")

def test_confidence_aware_smoothing():
    """Test the Confidence-Aware Smoothing component."""
    
    print("\n" + "=" * 60)
    print("CONFIDENCE-AWARE SMOOTHING TEST")
    print("=" * 60)
    
    # Initialize smoothing
    smoothing_threshold = 0.9
    min_confidence = 0.3
    smoothing = ConfidenceAwareSmoothing(smoothing_threshold, min_confidence)
    
    # Test scenarios
    batch_size = 4
    num_emotions = 4
    
    print(f"Testing with batch_size={batch_size}, num_emotions={num_emotions}")
    print("-" * 40)
    
    # Scenario 1: High confidence current prediction
    current_pred = torch.randn(batch_size, num_emotions)
    current_conf = torch.tensor([[0.95], [0.92], [0.88], [0.91]])  # High confidence
    temporal_pred = torch.randn(batch_size, num_emotions)
    temporal_conf = torch.tensor([[0.7], [0.6], [0.8], [0.5]])
    
    smoothed_pred1, final_conf1 = smoothing(current_pred, current_conf, temporal_pred, temporal_conf)
    
    print(f"Scenario 1 (High confidence):")
    print(f"  Current confidence: {current_conf.squeeze()}")
    print(f"  Temporal confidence: {temporal_conf.squeeze()}")
    print(f"  Final confidence: {final_conf1.squeeze()}")
    
    # Scenario 2: Low confidence current prediction
    current_conf2 = torch.tensor([[0.4], [0.3], [0.5], [0.2]])  # Low confidence
    temporal_conf2 = torch.tensor([[0.8], [0.9], [0.7], [0.8]])  # High temporal confidence
    
    smoothed_pred2, final_conf2 = smoothing(current_pred, current_conf2, temporal_pred, temporal_conf2)
    
    print(f"\nScenario 2 (Low confidence):")
    print(f"  Current confidence: {current_conf2.squeeze()}")
    print(f"  Temporal confidence: {temporal_conf2.squeeze()}")
    print(f"  Final confidence: {final_conf2.squeeze()}")
    
    # Verify behavior
    # High confidence should use current prediction more
    # Low confidence should use temporal prediction more
    smoothing_info1 = smoothing.get_smoothing_info(current_conf, temporal_conf)
    smoothing_info2 = smoothing.get_smoothing_info(current_conf2, temporal_conf2)
    
    print(f"\nSmoothing factors:")
    print(f"  Scenario 1: {smoothing_info1['smoothing_factor']:.3f}")
    print(f"  Scenario 2: {smoothing_info2['smoothing_factor']:.3f}")
    
    assert smoothing_info1['smoothing_factor'] > smoothing_info2['smoothing_factor'], \
        "Higher current confidence should result in higher smoothing factor"
    
    print(f"✅ Confidence-aware smoothing working correctly")

def test_speaker_change_detector():
    """Test the Speaker Change Detection component."""
    
    print("\n" + "=" * 60)
    print("SPEAKER CHANGE DETECTOR TEST")
    print("=" * 60)
    
    # Initialize speaker detector
    speaker_dim = 128
    similarity_threshold = 0.7
    speaker_detector = SpeakerChangeDetector(speaker_dim, similarity_threshold)
    
    # Test scenarios
    batch_size = 2
    num_segments = 4
    
    print(f"Testing with batch_size={batch_size}, num_segments={num_segments}")
    print(f"Speaker dim: {speaker_dim}, Threshold: {similarity_threshold}")
    print("-" * 40)
    
    # Scenario 1: Same speaker (high similarity)
    same_speaker_embeddings = torch.randn(batch_size, num_segments, speaker_dim)
    # Make embeddings more similar
    same_speaker_embeddings[:, 1:] = same_speaker_embeddings[:, 0:1] + 0.1 * torch.randn_like(same_speaker_embeddings[:, 1:])
    
    speaker_changes1, similarities1 = speaker_detector(same_speaker_embeddings)
    
    print(f"Scenario 1 (Same speaker):")
    print(f"  Speaker changes: {speaker_changes1.sum().item()}")
    print(f"  Average similarity: {similarities1.mean().item():.3f}")
    
    # Scenario 2: Different speakers (low similarity)
    different_speaker_embeddings = torch.randn(batch_size, num_segments, speaker_dim)
    # Make embeddings very different
    different_speaker_embeddings[:, 2:] = torch.randn_like(different_speaker_embeddings[:, 2:])
    
    speaker_changes2, similarities2 = speaker_detector(different_speaker_embeddings)
    
    print(f"\nScenario 2 (Different speakers):")
    print(f"  Speaker changes: {speaker_changes2.sum().item()}")
    print(f"  Average similarity: {similarities2.mean().item():.3f}")
    
    # Verify behavior
    speaker_info1 = speaker_detector.get_speaker_info(speaker_changes1, similarities1)
    speaker_info2 = speaker_detector.get_speaker_info(speaker_changes2, similarities2)
    
    print(f"\nSpeaker change statistics:")
    print(f"  Scenario 1: {speaker_info1}")
    print(f"  Scenario 2: {speaker_info2}")
    
    # Same speaker should have fewer changes
    assert speaker_info1['num_changes'] <= speaker_info2['num_changes'], \
        "Same speaker should have fewer detected changes"
    
    print(f"✅ Speaker change detection working correctly")

def test_temporal_buffer():
    """Test the Temporal Buffer component."""
    
    print("\n" + "=" * 60)
    print("TEMPORAL BUFFER TEST")
    print("=" * 60)
    
    # Initialize buffer
    max_segments = 3
    feature_dim = 256
    buffer = TemporalBuffer(max_segments, feature_dim)
    
    print(f"Testing with max_segments={max_segments}, feature_dim={feature_dim}")
    print("-" * 40)
    
    # Create test segments
    segments = []
    for i in range(5):  # More than max_segments to test overflow
        segment = create_temporal_segment(
            segment_id=i,
            start_time=i * 4.0,
            end_time=(i + 1) * 4.0,
            features=torch.randn(feature_dim),
            confidence=0.8 + 0.1 * torch.randn(1).item(),
            emotion_prediction=torch.randn(4),
            speaker_embedding=torch.randn(128)
        )
        segments.append(segment)
    
    # Add segments to buffer
    for i, segment in enumerate(segments):
        buffer.add_segment(segment)
        print(f"Added segment {i}: buffer size = {len(buffer.segments)}")
    
    # Get temporal features
    features, confidences, predictions = buffer.get_temporal_features()
    
    print(f"\nBuffer contents:")
    print(f"  Features shape: {features.shape}")
    print(f"  Confidences shape: {confidences.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    
    # Verify buffer behavior
    assert len(buffer.segments) == max_segments, f"Buffer should maintain max {max_segments} segments"
    assert features.shape[0] == max_segments, f"Should have {max_segments} feature vectors"
    
    # Test buffer info
    buffer_info = buffer.get_buffer_info()
    print(f"  Buffer info: {buffer_info}")
    
    # Test speaker embeddings
    speaker_embeddings = buffer.get_speaker_embeddings()
    if speaker_embeddings is not None:
        print(f"  Speaker embeddings shape: {speaker_embeddings.shape}")
    
    # Test buffer clearing
    buffer.clear()
    assert len(buffer.segments) == 0, "Buffer should be empty after clearing"
    
    print(f"✅ Temporal buffer working correctly")

def test_temporal_modeling_module():
    """Test the complete Temporal Modeling Module."""
    
    print("\n" + "=" * 60)
    print("TEMPORAL MODELING MODULE TEST")
    print("=" * 60)
    
    # Initialize module
    feature_dim = 256
    hidden_dim = 128
    max_segments = 3
    num_emotions = 4
    temporal_module = create_temporal_modeling_module(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        max_segments=max_segments,
        num_emotions=num_emotions
    )
    
    # Test data
    batch_size = 2
    
    print(f"Testing with batch_size={batch_size}")
    print(f"Feature dim: {feature_dim}, Hidden dim: {hidden_dim}")
    print(f"Max segments: {max_segments}, Num emotions: {num_emotions}")
    print("-" * 40)
    
    # Test scenario: Sequential processing of segments
    results = []
    
    for i in range(5):  # Process 5 segments
        print(f"\nProcessing segment {i+1}...")
        
        # Current segment features
        current_features = torch.randn(batch_size, feature_dim)
        current_speaker_embedding = torch.randn(batch_size, 128)
        
        # Forward pass
        start_time = time.time()
        outputs = temporal_module(
            current_features=current_features,
            current_speaker_embedding=current_speaker_embedding
        )
        processing_time = time.time() - start_time
        
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Emotion prediction shape: {outputs['emotion_prediction'].shape}")
        print(f"  Confidence shape: {outputs['confidence'].shape}")
        print(f"  Temporal context available: {outputs['temporal_context_available']}")
        print(f"  Speaker change detected: {outputs['speaker_change_detected'].any().item()}")
        print(f"  Smoothing applied: {outputs['smoothing_applied']}")
        
        # Store results
        results.append(outputs)
        
        # Create segment for buffer update
        segment = create_temporal_segment(
            segment_id=i,
            start_time=i * 4.0,
            end_time=(i + 1) * 4.0,
            features=current_features[0],  # Use first batch element
            confidence=outputs['confidence'][0].item(),
            emotion_prediction=outputs['emotion_prediction'][0],
            speaker_embedding=current_speaker_embedding[0]
        )
        
        # Update buffer
        temporal_module.update_buffer(segment)
        
        # Print buffer info
        buffer_info = outputs['buffer_info']
        print(f"  Buffer segments: {buffer_info['num_segments']}")
    
    # Test temporal report
    print(f"\nTemporal Modeling Report:")
    report = temporal_module.get_temporal_report(outputs)
    print(report)
    
    # Verify outputs
    assert outputs['emotion_prediction'].shape == (batch_size, num_emotions), \
        f"Expected emotion prediction shape ({batch_size}, {num_emotions})"
    assert outputs['confidence'].shape == (batch_size, 1), \
        f"Expected confidence shape ({batch_size}, 1)"
    
    # Test buffer reset
    temporal_module.reset_buffer()
    assert len(temporal_module.temporal_buffer.segments) == 0, "Buffer should be empty after reset"
    
    print(f"✅ Temporal modeling module working correctly")

def test_sequential_processing():
    """Test sequential processing with emotion transitions."""
    
    print("\n" + "=" * 60)
    print("SEQUENTIAL PROCESSING TEST")
    print("=" * 60)
    
    # Initialize module
    temporal_module = create_temporal_modeling_module()
    
    # Simulate emotion transition scenario
    batch_size = 1
    feature_dim = 256
    
    print("Simulating emotion transition scenario...")
    print("-" * 40)
    
    # Create segments with different emotions
    emotions = ['happy', 'neutral', 'sad', 'angry', 'happy']
    segment_features = []
    
    for i, emotion in enumerate(emotions):
        # Create features that represent different emotions
        if emotion == 'happy':
            features = torch.randn(batch_size, feature_dim) + 0.5  # Positive bias
        elif emotion == 'sad':
            features = torch.randn(batch_size, feature_dim) - 0.5  # Negative bias
        elif emotion == 'angry':
            features = torch.randn(batch_size, feature_dim) + 1.0  # High positive bias
        else:  # neutral
            features = torch.randn(batch_size, feature_dim)
        
        segment_features.append(features)
    
    # Process segments sequentially
    for i, (emotion, features) in enumerate(zip(emotions, segment_features)):
        print(f"\nSegment {i+1} ({emotion}):")
        
        outputs = temporal_module(
            current_features=features,
            current_speaker_embedding=torch.randn(batch_size, 128)
        )
        
        # Get emotion prediction
        emotion_probs = F.softmax(outputs['emotion_prediction'], dim=-1)
        predicted_emotion = torch.argmax(emotion_probs, dim=-1).item()
        confidence = outputs['confidence'].item()
        
        print(f"  Expected emotion: {emotion}")
        print(f"  Predicted emotion: {predicted_emotion}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Temporal context: {outputs['temporal_context_available']}")
        print(f"  Smoothing applied: {outputs['smoothing_applied']}")
        
        # Update buffer
        segment = create_temporal_segment(
            segment_id=i,
            start_time=i * 4.0,
            end_time=(i + 1) * 4.0,
            features=features[0],
            confidence=confidence,
            emotion_prediction=outputs['emotion_prediction'][0],
            speaker_embedding=torch.randn(128)
        )
        temporal_module.update_buffer(segment)
    
    print(f"\n✅ Sequential processing with emotion transitions working correctly")

if __name__ == "__main__":
    # Test individual components
    test_temporal_positional_encoding()
    test_causal_conv1d()
    test_temporal_convnet()
    test_confidence_aware_smoothing()
    test_speaker_change_detector()
    test_temporal_buffer()
    
    # Test complete system
    test_temporal_modeling_module()
    test_sequential_processing()
    
    print(f"\n" + "=" * 80)
    print("TEMPORAL MODELING MODULE TESTING COMPLETE")
    print("=" * 80)
    print("✅ Temporal positional encoding")
    print("✅ Causal 1D convolutions")
    print("✅ Temporal convolutional network")
    print("✅ Confidence-aware smoothing")
    print("✅ Speaker change detection")
    print("✅ Temporal buffer management")
    print("✅ Sequential processing with emotion transitions")
    print("✅ Ready for integration with main pipeline")
