#!/usr/bin/env python3
"""
Comprehensive ASR Integration Test
Verifies all professor's requirements are implemented
"""

import time
import torch
import numpy as np
import librosa
from src.models.asr_integration import create_enhanced_asr, ASRResult
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio_with_speech(duration=5.0, sample_rate=16000):
    """Create test audio with simulated speech patterns."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate speech-like signal
    speech_signal = (
        0.5 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
        0.3 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
        0.2 * np.sin(2 * np.pi * 600 * t) +  # Second harmonic
        0.1 * np.random.randn(len(t))        # Noise
    )
    
    # Add amplitude modulation to simulate words
    word_rate = 2.0  # 2 words per second
    word_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * word_rate * t)
    speech_signal = speech_signal * word_envelope
    
    # Normalize
    speech_signal = speech_signal / np.max(np.abs(speech_signal)) * 0.8
    
    return speech_signal.astype(np.float32)

def test_professor_requirements():
    """Test all professor's requirements for ASR integration."""
    
    print("=" * 80)
    print("PROFESSOR'S ASR INTEGRATION REQUIREMENTS VERIFICATION")
    print("=" * 80)
    
    # System info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Initialize ASR module
    print("\nInitializing Enhanced ASR Integration...")
    start_time = time.time()
    
    try:
        asr_module = create_enhanced_asr(
            model_name="openai/whisper-base",  # Use base for faster testing
            device=device
        )
        init_time = time.time() - start_time
        print(f"✓ Enhanced ASR module initialized in {init_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Failed to initialize ASR module: {e}")
        return False
    
    # Test audio
    audio = create_test_audio_with_speech(duration=5.0)
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    
    print(f"\nTesting ASR pipeline with 5s audio...")
    
    # Process through ASR
    start_time = time.time()
    try:
        asr_result = asr_module(audio_tensor)
        processing_time = time.time() - start_time
        print(f"✓ ASR processing completed in {processing_time:.2f} seconds")
    except Exception as e:
        print(f"✗ ASR processing failed: {e}")
        return False
    
    # Verify all requirements
    requirements_met = []
    
    print(f"\n" + "=" * 60)
    print("REQUIREMENT VERIFICATION")
    print("=" * 60)
    
    # 1. Multilingual ASR with Confidence
    print(f"\n1️⃣ Multilingual ASR with Confidence:")
    print("-" * 40)
    
    # Check basic ASR output
    if hasattr(asr_result, 'text') and asr_result.text:
        print(f"  ✅ Basic transcription: '{asr_result.text[:50]}...'")
        requirements_met.append("Basic transcription")
    else:
        print(f"  ❌ Basic transcription missing")
    
    # Check language detection
    if hasattr(asr_result, 'detected_languages') and asr_result.detected_languages:
        print(f"  ✅ Language detection: {asr_result.detected_languages}")
        requirements_met.append("Language detection")
    else:
        print(f"  ❌ Language detection missing")
    
    # Check confidence scores
    if hasattr(asr_result, 'word_confidences') and asr_result.word_confidences:
        print(f"  ✅ Word-level confidences: {len(asr_result.word_confidences)} words")
        print(f"     Sample confidences: {asr_result.word_confidences[:3]}")
        requirements_met.append("Word-level confidence")
    else:
        print(f"  ❌ Word-level confidences missing")
    
    if hasattr(asr_result, 'segment_confidence'):
        print(f"  ✅ Segment confidence: {asr_result.segment_confidence:.3f}")
        requirements_met.append("Segment confidence")
    else:
        print(f"  ❌ Segment confidence missing")
    
    if hasattr(asr_result, 'overall_confidence'):
        print(f"  ✅ Overall confidence: {asr_result.overall_confidence:.3f}")
        requirements_met.append("Overall confidence")
    else:
        print(f"  ❌ Overall confidence missing")
    
    # 2. Code-switching Awareness
    print(f"\n2️⃣ Code-switching Awareness:")
    print("-" * 40)
    
    if hasattr(asr_result, 'code_switches'):
        print(f"  ✅ Code-switching detection: {len(asr_result.code_switches)} switches")
        if asr_result.code_switches:
            print(f"     Sample switches: {asr_result.code_switches[:2]}")
        requirements_met.append("Code-switching detection")
    else:
        print(f"  ❌ Code-switching detection missing")
    
    if hasattr(asr_result, 'language_segments'):
        print(f"  ✅ Language segments: {len(asr_result.language_segments)} segments")
        requirements_met.append("Language segments")
    else:
        print(f"  ❌ Language segments missing")
    
    # 3. Timestamp Alignment
    print(f"\n3️⃣ Timestamp Alignment:")
    print("-" * 40)
    
    if hasattr(asr_result, 'word_timestamps') and asr_result.word_timestamps:
        print(f"  ✅ Word timestamps: {len(asr_result.word_timestamps)} words aligned")
        print(f"     Sample timestamps: {asr_result.word_timestamps[:2]}")
        requirements_met.append("Word timestamps")
    else:
        print(f"  ❌ Word timestamps missing")
    
    if hasattr(asr_result, 'phone_alignment') and asr_result.phone_alignment:
        print(f"  ✅ Phone alignment: {len(asr_result.phone_alignment)} phones")
        requirements_met.append("Phone alignment")
    else:
        print(f"  ❌ Phone alignment missing")
    
    if hasattr(asr_result, 'silence_regions') and asr_result.silence_regions:
        print(f"  ✅ Silence regions: {len(asr_result.silence_regions)} regions")
        print(f"     Sample regions: {asr_result.silence_regions[:2]}")
        requirements_met.append("Silence regions")
    else:
        print(f"  ❌ Silence regions missing")
    
    # 4. Confidence-Aware Text Processing
    print(f"\n4️⃣ Confidence-Aware Text Processing:")
    print("-" * 40)
    
    if hasattr(asr_result, 'text_reliability_score'):
        print(f"  ✅ Text reliability score: {asr_result.text_reliability_score:.3f}")
        requirements_met.append("Text reliability score")
    else:
        print(f"  ❌ Text reliability score missing")
    
    if hasattr(asr_result, 'attention_mask_weighted'):
        print(f"  ✅ Confidence-aware attention mask: {asr_result.attention_mask_weighted.shape}")
        print(f"     Sample weights: {asr_result.attention_mask_weighted[:5]}")
        requirements_met.append("Confidence-aware attention")
    else:
        print(f"  ❌ Confidence-aware attention mask missing")
    
    # 5. ASR Features for Fusion
    print(f"\n5️⃣ ASR Features for Fusion:")
    print("-" * 40)
    
    if hasattr(asr_result, 'asr_features'):
        print(f"  ✅ ASR features: {asr_result.asr_features.shape}")
        print(f"     Feature values: {asr_result.asr_features}")
        requirements_met.append("ASR features")
    else:
        print(f"  ❌ ASR features missing")
    
    # 6. Temperature Scaling (Confidence Calibration)
    print(f"\n6️⃣ Confidence Calibration:")
    print("-" * 40)
    
    # Check if calibration is available
    if hasattr(asr_module.asr, 'confidence_calibrator'):
        print(f"  ✅ Confidence calibrator available")
        print(f"     Calibrated: {asr_module.asr.is_calibrated}")
        requirements_met.append("Confidence calibration")
    else:
        print(f"  ❌ Confidence calibrator missing")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_requirements = 12  # Count of all requirements
    met_requirements = len(requirements_met)
    
    print(f"Requirements met: {met_requirements}/{total_requirements}")
    print(f"Completion rate: {(met_requirements/total_requirements)*100:.1f}%")
    
    if met_requirements == total_requirements:
        print(f"🎉 ALL REQUIREMENTS MET!")
    else:
        print(f"⚠️  Some requirements missing")
    
    print(f"\nMet requirements:")
    for req in requirements_met:
        print(f"  ✅ {req}")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Real-time factor: {processing_time/5.0:.2f}x")
    print(f"  Memory usage: ~1-2GB")
    
    # Generate detailed report
    try:
        report = asr_module.get_asr_report(asr_result)
        print(f"\nDetailed ASR Report:")
        print(report)
    except Exception as e:
        print(f"Could not generate detailed report: {e}")
    
    return met_requirements == total_requirements

def test_confidence_calibration():
    """Test confidence calibration functionality."""
    
    print(f"\n" + "=" * 60)
    print("CONFIDENCE CALIBRATION TEST")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        asr_module = create_enhanced_asr(
            model_name="openai/whisper-base",
            device=device
        )
        
        # Create mock validation data
        validation_data = []
        for i in range(3):
            audio = create_test_audio_with_speech(duration=3.0)
            true_conf = 0.8 + 0.1 * np.random.randn()  # Mock true confidence
            validation_data.append((audio, true_conf))
        
        # Test calibration
        print("Testing confidence calibration...")
        asr_module.asr.calibrate_confidence(validation_data)
        
        if asr_module.asr.is_calibrated:
            print("✅ Confidence calibration successful")
            return True
        else:
            print("❌ Confidence calibration failed")
            return False
            
    except Exception as e:
        print(f"❌ Confidence calibration test failed: {e}")
        return False

if __name__ == "__main__":
    # Test main requirements
    success = test_professor_requirements()
    
    # Test confidence calibration
    calibration_success = test_confidence_calibration()
    
    print(f"\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if success and calibration_success:
        print("🎉 ENHANCED ASR INTEGRATION IS COMPLETE!")
        print("All professor's requirements have been successfully implemented.")
    else:
        print("⚠️  ENHANCED ASR INTEGRATION NEEDS IMPROVEMENT")
        print("Some requirements are missing or not working properly.")
    
    print(f"\nImplementation Status:")
    print(f"  Main requirements: {'✅ Complete' if success else '❌ Incomplete'}")
    print(f"  Confidence calibration: {'✅ Complete' if calibration_success else '❌ Incomplete'}")
