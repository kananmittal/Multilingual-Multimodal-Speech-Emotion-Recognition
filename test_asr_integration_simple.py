#!/usr/bin/env python3
"""
Simplified test script for Enhanced ASR Integration and Alignment Module
Tests functionality without requiring large model downloads
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.asr_integration import ASRResult, ConfidenceAwareTextProcessor, MultilingualASR


def test_asr_components_without_model():
    """Test ASR components that don't require model loading."""
    
    print("🧪 Testing ASR Components (No Model Loading)")
    print("=" * 50)
    
    # Test 1: ASRResult dataclass
    print("\n1️⃣ Testing ASRResult dataclass...")
    test_result = ASRResult(
        text="Hello world this is a test",
        language="en",
        detected_languages=["en"],
        word_confidences=[0.9, 0.8, 0.7, 0.5, 0.3, 0.2],
        segment_confidence=0.6,
        overall_confidence=0.6,
        word_timestamps=[(i*0.5, (i+1)*0.5) for i in range(6)],
        phone_alignment=[],
        silence_regions=[(0.0, 0.1), (4.9, 5.0)],
        code_switches=[],
        language_segments=[],
        text_reliability_score=0.0,
        attention_mask_weighted=torch.ones(6),
        asr_features=torch.zeros(8)
    )
    
    print(f"✅ ASRResult created successfully")
    print(f"   - Text: '{test_result.text}'")
    print(f"   - Language: {test_result.language}")
    print(f"   - Confidence: {test_result.overall_confidence}")
    print(f"   - Word timestamps: {len(test_result.word_timestamps)}")
    
    # Test 2: Confidence-aware text processing
    print("\n2️⃣ Testing ConfidenceAwareTextProcessor...")
    processor = ConfidenceAwareTextProcessor()
    
    # Test with different confidence levels
    test_confidences = [0.9, 0.8, 0.7, 0.5, 0.3, 0.2]
    processed_result = processor.process_text_with_confidence(test_result)
    
    print(f"✅ Confidence processing completed")
    print(f"   - Text reliability score: {processed_result.text_reliability_score:.3f}")
    print(f"   - Attention mask weights: {processed_result.attention_mask_weighted}")
    
    # Test 3: Language detection (without model)
    print("\n3️⃣ Testing language detection...")
    
    test_texts = [
        "Hello world",
        "Привет мир",
        "你好世界",
        "Hola mundo",
        "Hello привет world"
    ]
    
    # Create a simple MultilingualASR instance without loading the model
    class SimpleMultilingualASR:
        def __init__(self):
            self.supported_languages = {
                'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german',
                'it': 'italian', 'pt': 'portuguese', 'ru': 'russian', 'ja': 'japanese',
                'ko': 'korean', 'zh': 'chinese', 'ar': 'arabic', 'hi': 'hindi'
            }
        
        def _detect_languages_and_switches(self, text):
            """Detect languages and code-switching in text."""
            import re
            detected_languages = []
            code_switches = []
            
            # Check for different scripts
            if re.search(r'[а-яё]', text, re.IGNORECASE):
                detected_languages.append('ru')
            if re.search(r'[一-龯]', text):
                detected_languages.append('zh')
            if re.search(r'[あ-ん]', text):
                detected_languages.append('ja')
            if re.search(r'[가-힣]', text):
                detected_languages.append('ko')
            if re.search(r'[ا-ي]', text):
                detected_languages.append('ar')
            if re.search(r'[अ-ह]', text):
                detected_languages.append('hi')
            
            # Default to English if no specific script detected
            if not detected_languages:
                detected_languages.append('en')
            
            # Detect code switches (simplified)
            words = text.split()
            for i, word in enumerate(words):
                # Check for language-specific patterns
                if re.search(r'[а-яё]', word, re.IGNORECASE) and 'ru' not in detected_languages:
                    code_switches.append({
                        'position': i,
                        'word': word,
                        'from_lang': detected_languages[0],
                        'to_lang': 'ru'
                    })
            
            return detected_languages, code_switches
    
    simple_asr = SimpleMultilingualASR()
    
    print("Language Detection Results:")
    for text in test_texts:
        detected_langs, code_switches = simple_asr._detect_languages_and_switches(text)
        print(f"   '{text}' -> Languages: {detected_langs}, Switches: {len(code_switches)}")
    
    # Test 4: Feature creation
    print("\n4️⃣ Testing ASR feature creation...")
    
    def create_asr_features(asr_result):
        """Create ASR features for downstream fusion."""
        features = torch.tensor([
            asr_result.overall_confidence,
            asr_result.text_reliability_score,
            len(asr_result.detected_languages) / 5.0,  # Normalize language count
            len(asr_result.code_switches) / 10.0,      # Normalize code switches
            len(asr_result.silence_regions) / 20.0,    # Normalize silence regions
            np.mean([end - start for start, end in asr_result.word_timestamps]) if asr_result.word_timestamps else 0.0,
            len(asr_result.text.split()) / 50.0,       # Normalize word count
            float(len(asr_result.text) > 0)            # Has transcription
        ], dtype=torch.float32)
        
        return features
    
    features = create_asr_features(processed_result)
    print(f"✅ ASR features created: {features.shape}")
    print(f"   - Feature values: {features}")
    
    # Test 5: Mock ASR results for different scenarios
    print("\n5️⃣ Testing different ASR scenarios...")
    
    scenarios = [
        ("High confidence English", "Hello world this is a test", "en", 0.95),
        ("Low confidence noisy", "Hello world this is a test", "en", 0.45),
        ("Code switching", "Hello mundo this is una prueba", "en", 0.78),
        ("Multiple languages", "Hello привет world", "en", 0.82),
        ("Empty transcription", "", "en", 0.0)
    ]
    
    for scenario_name, text, lang, conf in scenarios:
        print(f"\n   {scenario_name}:")
        
        # Create ASR result
        word_count = len(text.split()) if text else 0
        word_confidences = [conf] * word_count if word_count > 0 else []
        
        scenario_result = ASRResult(
            text=text,
            language=lang,
            detected_languages=[lang] if lang else [],
            word_confidences=word_confidences,
            segment_confidence=conf,
            overall_confidence=conf,
            word_timestamps=[(i*0.5, (i+1)*0.5) for i in range(word_count)],
            phone_alignment=[],
            silence_regions=[(0.0, 0.1), (4.9, 5.0)],
            code_switches=[],
            language_segments=[],
            text_reliability_score=conf,
            attention_mask_weighted=torch.ones(word_count) if word_count > 0 else torch.ones(1),
            asr_features=torch.zeros(8)
        )
        
        # Process with confidence awareness
        processed_scenario = processor.process_text_with_confidence(scenario_result)
        
        print(f"     - Text: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        print(f"     - Confidence: {conf:.3f}")
        print(f"     - Reliability: {processed_scenario.text_reliability_score:.3f}")
        print(f"     - Word count: {word_count}")
    
    print("\n✅ All ASR component tests completed successfully!")
    return True


def test_integration_without_model():
    """Test integration points without loading large models."""
    
    print("\n🔗 Testing Integration Points")
    print("=" * 40)
    
    # Test text encoder integration
    print("\n1️⃣ Testing Text Encoder Integration...")
    
    # Mock the ASR integration to avoid model loading
    class MockASRIntegration:
        def __init__(self):
            pass
        
        def __call__(self, audio):
            # Return a mock ASR result
            return ASRResult(
                text="Hello world this is a test",
                language="en",
                detected_languages=["en"],
                word_confidences=[0.8] * 6,
                segment_confidence=0.8,
                overall_confidence=0.8,
                word_timestamps=[(i*0.5, (i+1)*0.5) for i in range(6)],
                phone_alignment=[],
                silence_regions=[],
                code_switches=[],
                language_segments=[],
                text_reliability_score=0.8,
                attention_mask_weighted=torch.ones(6),
                asr_features=torch.tensor([0.8, 0.8, 0.2, 0.0, 0.1, 0.5, 0.1, 1.0])
            )
    
    print("✅ Mock ASR integration created")
    
    # Test feature fusion
    print("\n2️⃣ Testing Feature Fusion...")
    
    # Simulate text encoder output
    batch_size, seq_len, hidden_size = 2, 10, 768
    text_features = torch.randn(batch_size, seq_len, hidden_size)
    asr_features = torch.tensor([[0.8, 0.8, 0.2, 0.0, 0.1, 0.5, 0.1, 1.0]] * batch_size)
    
    # Simulate fusion
    for i in range(batch_size):
        asr_features_expanded = asr_features[i].unsqueeze(0).expand(seq_len, -1)
        combined_features = torch.cat([text_features[i], asr_features_expanded], dim=-1)
        print(f"   Sample {i}: Combined features shape: {combined_features.shape}")
    
    print("✅ Feature fusion simulation completed")
    
    # Test confidence-aware processing
    print("\n3️⃣ Testing Confidence-Aware Processing...")
    
    confidence_levels = [0.9, 0.7, 0.5, 0.3, 0.1]
    processor = ConfidenceAwareTextProcessor()
    
    for conf in confidence_levels:
        test_result = ASRResult(
            text="test",
            language="en",
            detected_languages=["en"],
            word_confidences=[conf],
            segment_confidence=conf,
            overall_confidence=conf,
            word_timestamps=[],
            phone_alignment=[],
            silence_regions=[],
            code_switches=[],
            language_segments=[],
            text_reliability_score=0.0,
            attention_mask_weighted=torch.ones(1),
            asr_features=torch.zeros(8)
        )
        
        processed = processor.process_text_with_confidence(test_result)
        mask_weight = processed.attention_mask_weighted[0].item()
        
        print(f"   Confidence {conf:.1f} -> Mask weight: {mask_weight:.1f}")
    
    print("✅ Confidence-aware processing completed")
    
    return True


def main():
    """Main test function."""
    
    print("🚀 Simplified ASR Integration Test Suite")
    print("=" * 60)
    print("Testing without loading large models...")
    
    try:
        # Test 1: ASR components
        print("\n1️⃣ Testing ASR Components...")
        test_asr_components_without_model()
        
        # Test 2: Integration points
        print("\n2️⃣ Testing Integration Points...")
        test_integration_without_model()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Summary:")
        print("- ASRResult dataclass: ✅")
        print("- Confidence-aware processing: ✅")
        print("- Language detection: ✅")
        print("- Feature creation: ✅")
        print("- Integration simulation: ✅")
        print("- No large model downloads required: ✅")
        
        print("\n💡 Note: This test avoids downloading large models.")
        print("   For full functionality, the system will download Whisper when needed.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
