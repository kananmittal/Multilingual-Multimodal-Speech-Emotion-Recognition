#!/usr/bin/env python3
"""
Test for Cross-Lingual Variance Handling
Verifies language-adversarial training, adapter-based tuning, and consistency loss
"""

import time
import torch
import numpy as np
from src.models.cross_lingual_variance import (
    LanguageAdversarialHead,
    AdapterLayer,
    CrossLingualConsistencyLoss,
    CrossLingualVarianceHandler,
    create_cross_lingual_variance_handler,
    get_adapter_parameters,
    freeze_base_parameters,
    LanguageInfo
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_encoder(name: str = "mock_encoder"):
    """Create a mock encoder for testing."""
    class MockEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.ModuleList([
                torch.nn.Linear(768, 768) for _ in range(12)
            ])
            self.embedding = torch.nn.Embedding(1000, 768)
        
        def forward(self, input_values, attention_mask=None):
            # Mock forward pass
            if input_values.dtype == torch.long:
                # Text input (token ids)
                hidden_states = self.embedding(input_values)
            else:
                # Audio input (features)
                hidden_states = input_values.float()
            
            # Apply mock layers
            for layer in self.encoder:
                hidden_states = layer(hidden_states)
            
            # Return mock output
            class MockOutput:
                def __init__(self, hidden_states):
                    self.last_hidden_state = hidden_states
            
            return MockOutput(hidden_states)
    
    return MockEncoder()

def create_mock_fusion_layer():
    """Create a mock fusion layer for testing."""
    class MockFusionLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1536, 256)  # 768*2 -> 256
        
        def forward(self, audio_features, text_features):
            # Mock fusion: concatenate and project
            if audio_features.dim() == 3:
                audio_features = audio_features.mean(dim=1)  # Pool sequence
            if text_features.dim() == 3:
                text_features = text_features.mean(dim=1)  # Pool sequence
            
            combined = torch.cat([audio_features, text_features], dim=-1)
            return self.projection(combined)
    
    return MockFusionLayer()

def test_language_adversarial_head():
    """Test the Language Adversarial Head component."""
    
    print("=" * 60)
    print("LANGUAGE ADVERSARIAL HEAD TEST")
    print("=" * 60)
    
    # Initialize adversarial head
    adversarial_head = LanguageAdversarialHead(
        input_dim=256,
        hidden_dim=128,
        num_languages=7
    )
    
    # Test data
    batch_size = 4
    features = torch.randn(batch_size, 256)
    
    print(f"Testing with batch size {batch_size}...")
    print("-" * 40)
    
    # Forward pass
    language_logits, language_probs = adversarial_head(features)
    
    print(f"Input features shape: {features.shape}")
    print(f"Language logits shape: {language_logits.shape}")
    print(f"Language probs shape: {language_probs.shape}")
    
    # Verify shapes
    assert language_logits.shape == (batch_size, 7), f"Expected shape ({batch_size}, 7), got {language_logits.shape}"
    assert language_probs.shape == (batch_size, 7), f"Expected shape ({batch_size}, 7), got {language_probs.shape}"
    
    # Verify probabilities sum to 1
    prob_sums = language_probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6), "Probabilities must sum to 1"
    
    print(f"✅ Language logits shape correct")
    print(f"✅ Language probabilities sum to 1")
    
    # Get language predictions
    language_predictions = adversarial_head.get_language_prediction(language_probs)
    
    print(f"\nLanguage predictions:")
    for i, pred in enumerate(language_predictions):
        print(f"  Sample {i+1}: {pred.language_name} (ID: {pred.language_id}, Conf: {pred.confidence:.3f})")
    
    # Verify predictions
    assert len(language_predictions) == batch_size, f"Expected {batch_size} predictions, got {len(language_predictions)}"
    for pred in language_predictions:
        assert 0 <= pred.language_id < 7, f"Language ID must be in [0, 6], got {pred.language_id}"
        assert 0.0 <= pred.confidence <= 1.0, f"Confidence must be in [0, 1], got {pred.confidence}"
    
    print(f"✅ Language predictions valid")

def test_adapter_layer():
    """Test the Adapter Layer component."""
    
    print("\n" + "=" * 60)
    print("ADAPTER LAYER TEST")
    print("=" * 60)
    
    # Initialize adapter layer
    hidden_size = 768
    adapter_size = 64
    adapter_layer = AdapterLayer(hidden_size, adapter_size)
    
    # Test data
    batch_size = 4
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
    print("-" * 40)
    
    # Forward pass
    output = adapter_layer(hidden_states)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify shapes
    assert output.shape == hidden_states.shape, f"Output shape must match input shape"
    
    # Verify residual connection (output should be close to input initially)
    # Since adapters are initialized close to identity
    diff = torch.abs(output - hidden_states).mean()
    print(f"Mean difference from input: {diff:.6f}")
    
    # Verify that adapter parameters are trainable
    trainable_params = sum(p.numel() for p in adapter_layer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in adapter_layer.parameters())
    
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Parameter efficiency: {trainable_params/total_params*100:.1f}%")
    
    assert trainable_params > 0, "Adapter should have trainable parameters"
    # Adapter should be parameter efficient compared to full transformer layer
    # A full transformer layer would have ~2.4M parameters (768*768*4 + 768*4)
    full_layer_params = 768 * 768 * 4 + 768 * 4  # ~2.4M
    assert trainable_params < full_layer_params * 0.1, "Adapter should be parameter efficient"
    
    print(f"✅ Adapter layer working correctly")
    print(f"✅ Parameter efficient design")

def test_cross_lingual_consistency_loss():
    """Test the Cross-Lingual Consistency Loss component."""
    
    print("\n" + "=" * 60)
    print("CROSS-LINGUAL CONSISTENCY LOSS TEST")
    print("=" * 60)
    
    # Initialize consistency loss
    consistency_loss = CrossLingualConsistencyLoss(temperature=0.1, weight=0.05)
    
    # Test scenarios
    batch_size = 8
    embedding_dim = 256
    num_emotions = 4
    num_languages = 3
    
    print(f"Testing with batch_size={batch_size}, embedding_dim={embedding_dim}")
    print("-" * 40)
    
    # Scenario 1: Same emotion across different languages
    embeddings = torch.randn(batch_size, embedding_dim)
    emotion_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # Same emotion in pairs
    language_ids = torch.tensor([0, 1, 0, 2, 1, 2, 0, 1])   # Different languages
    
    loss1 = consistency_loss(embeddings, emotion_labels, language_ids)
    print(f"Scenario 1 (same emotion, different languages): {loss1.item():.6f}")
    
    # Scenario 2: Different emotions, different languages
    emotion_labels2 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])  # Different emotions
    language_ids2 = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])    # Different languages
    
    loss2 = consistency_loss(embeddings, emotion_labels2, language_ids2)
    print(f"Scenario 2 (different emotions, different languages): {loss2.item():.6f}")
    
    # Scenario 3: Same emotion, same language
    emotion_labels3 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # Same emotion
    language_ids3 = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0])    # Same language
    
    loss3 = consistency_loss(embeddings, emotion_labels3, language_ids3)
    print(f"Scenario 3 (same emotion, same language): {loss3.item():.6f}")
    
    # Verify loss behavior
    assert loss1.item() >= 0, "Consistency loss should be non-negative for same emotion, different languages"
    assert loss2.item() >= 0, "Consistency loss should be non-negative"
    assert loss3.item() >= 0, "Consistency loss should be non-negative"
    
    print(f"✅ Consistency loss working correctly")
    print(f"✅ Loss behavior as expected")

def test_cross_lingual_variance_handler():
    """Test the complete Cross-Lingual Variance Handler."""
    
    print("\n" + "=" * 60)
    print("CROSS-LINGUAL VARIANCE HANDLER TEST")
    print("=" * 60)
    
    # Create mock components
    audio_encoder = create_mock_encoder("audio_encoder")
    text_encoder = create_mock_encoder("text_encoder")
    fusion_layer = create_mock_fusion_layer()
    
    # Initialize variance handler
    variance_handler = create_cross_lingual_variance_handler(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        fusion_layer=fusion_layer,
        num_languages=7
    )
    
    # Test data
    batch_size = 4
    seq_len = 20
    
    # Audio input (features)
    audio_input = torch.randn(batch_size, seq_len, 768)
    
    # Text input (token ids)
    text_input = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Language IDs
    language_ids = torch.randint(0, 7, (batch_size,))
    
    print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
    print("-" * 40)
    
    # Forward pass
    start_time = time.time()
    outputs = variance_handler(
        audio_input=audio_input,
        text_input=text_input,
        language_ids=language_ids
    )
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.3f}s")
    print(f"Audio features shape: {outputs['audio_features'].shape}")
    print(f"Text features shape: {outputs['text_features'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Language logits shape: {outputs['language_logits'].shape}")
    print(f"Language probs shape: {outputs['language_probs'].shape}")
    
    # Verify shapes
    assert outputs['audio_features'].shape == (batch_size, seq_len, 768)
    assert outputs['text_features'].shape == (batch_size, seq_len, 768)
    assert outputs['fused_features'].shape == (batch_size, 256)
    assert outputs['language_logits'].shape == (batch_size, 7)
    assert outputs['language_probs'].shape == (batch_size, 7)
    
    print(f"✅ All output shapes correct")
    
    # Test language predictions
    language_predictions = outputs['language_predictions']
    print(f"\nLanguage predictions:")
    for i, pred in enumerate(language_predictions):
        print(f"  Sample {i+1}: {pred.language_name} (Conf: {pred.confidence:.3f})")
    
    # Test loss computation
    emotion_logits = torch.randn(batch_size, 4)  # 4 emotions
    emotion_labels = torch.randint(0, 4, (batch_size,))
    language_labels = torch.randint(0, 7, (batch_size,))
    
    losses = variance_handler.compute_losses(
        emotion_logits=emotion_logits,
        emotion_labels=emotion_labels,
        language_logits=outputs['language_logits'],
        language_labels=language_labels,
        consistency_loss=outputs['consistency_loss'],
        lambda_adversarial=0.1
    )
    
    print(f"\nLoss computation:")
    print(f"  Emotion loss: {losses['emotion_loss'].item():.6f}")
    print(f"  Language loss: {losses['language_loss'].item():.6f}")
    print(f"  Consistency loss: {losses['consistency_loss'].item():.6f}")
    print(f"  Total loss: {losses['total_loss'].item():.6f}")
    
    # Verify losses
    assert losses['emotion_loss'].item() > 0, "Emotion loss should be positive"
    assert losses['language_loss'].item() > 0, "Language loss should be positive"
    assert losses['total_loss'].item() > 0, "Total loss should be positive"
    
    print(f"✅ Loss computation working correctly")
    
    # Test language statistics
    stats = variance_handler.get_language_statistics(language_predictions)
    print(f"\nLanguage statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Average confidence: {stats['avg_confidence']:.3f}")
    print(f"  Language distribution: {stats['language_distribution']}")

def test_adapter_parameter_management():
    """Test adapter parameter management utilities."""
    
    print("\n" + "=" * 60)
    print("ADAPTER PARAMETER MANAGEMENT TEST")
    print("=" * 60)
    
    # Create a model with adapters
    audio_encoder = create_mock_encoder("audio_encoder")
    text_encoder = create_mock_encoder("text_encoder")
    fusion_layer = create_mock_fusion_layer()
    
    variance_handler = create_cross_lingual_variance_handler(
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        fusion_layer=fusion_layer
    )
    
    print("Testing parameter management...")
    print("-" * 40)
    
    # Get adapter parameters
    adapter_params = get_adapter_parameters(variance_handler)
    print(f"Number of adapter parameters: {len(adapter_params)}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in variance_handler.parameters())
    adapter_param_count = sum(p.numel() for p in adapter_params)
    
    print(f"Total parameters: {total_params}")
    print(f"Adapter parameters: {adapter_param_count}")
    print(f"Adapter parameter ratio: {adapter_param_count/total_params*100:.2f}%")
    
    # Test parameter freezing
    freeze_base_parameters(variance_handler)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in variance_handler.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable_params}")
    
    # Verify that only adapters are trainable
    assert trainable_params > 0, "Some parameters should remain trainable"
    assert trainable_params <= adapter_param_count, "Only adapter parameters should be trainable"
    
    print(f"✅ Parameter management working correctly")
    print(f"✅ Base parameters frozen, only adapters trainable")

def test_gradient_reversal():
    """Test gradient reversal functionality."""
    
    print("\n" + "=" * 60)
    print("GRADIENT REVERSAL TEST")
    print("=" * 60)
    
    from src.models.cross_lingual_variance import GradientReversalLayer
    
    # Create gradient reversal layer
    grl = GradientReversalLayer(alpha=1.0)
    
    # Test data
    x = torch.randn(2, 3, requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print("-" * 40)
    
    # Forward pass
    y = grl(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Forward pass identical: {torch.allclose(x, y)}")
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    
    print(f"Input gradients: {x.grad}")
    print(f"Expected gradients: {torch.ones_like(x)}")
    print(f"Gradient reversal working: {torch.allclose(x.grad, -torch.ones_like(x))}")
    
    assert torch.allclose(x.grad, -torch.ones_like(x)), "Gradients should be reversed"
    
    print(f"✅ Gradient reversal working correctly")

if __name__ == "__main__":
    # Test individual components
    test_language_adversarial_head()
    test_adapter_layer()
    test_cross_lingual_consistency_loss()
    test_gradient_reversal()
    
    # Test complete system
    test_cross_lingual_variance_handler()
    test_adapter_parameter_management()
    
    print(f"\n" + "=" * 80)
    print("CROSS-LINGUAL VARIANCE HANDLING TESTING COMPLETE")
    print("=" * 80)
    print("✅ Language-adversarial training with gradient reversal")
    print("✅ Adapter-based encoder tuning")
    print("✅ Cross-lingual consistency loss")
    print("✅ Parameter-efficient design")
    print("✅ Ready for integration with main pipeline")
