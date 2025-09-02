#!/usr/bin/env python3
"""
Test script for ASR Integration Module
Measures performance and timing for different audio inputs
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

def create_test_audio(duration=5.0, sample_rate=16000, noise_level=0.1):
    """Create synthetic test audio with speech-like characteristics."""
    # Generate a simple speech-like signal (sine wave with modulation)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Base frequency (speech-like)
    base_freq = 200 + 100 * np.sin(2 * np.pi * 0.5 * t)  # Varying frequency
    
    # Generate signal
    signal = np.sin(2 * np.pi * base_freq * t)
    
    # Add some harmonics
    signal += 0.3 * np.sin(2 * np.pi * 2 * base_freq * t)
    signal += 0.1 * np.sin(2 * np.pi * 3 * base_freq * t)
    
    # Add noise
    noise = noise_level * np.random.randn(len(signal))
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal.astype(np.float32)

def test_asr_integration():
    """Test the ASR integration module with timing measurements."""
    
    print("=" * 60)
    print("ASR Integration Module Performance Test")
    print("=" * 60)
    
    # System info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Initialize ASR module
    print("\nInitializing ASR Integration Module...")
    start_time = time.time()
    
    try:
        asr_module = create_enhanced_asr(
            model_name="openai/whisper-large-v3",
            device=device
        )
        init_time = time.time() - start_time
        print(f"✓ ASR module initialized in {init_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Failed to initialize ASR module: {e}")
        return
    
    # Test with different audio durations
    test_durations = [3.0, 5.0, 10.0, 15.0]
    results = []
    
    print(f"\nTesting with different audio durations...")
    print("-" * 40)
    
    for duration in test_durations:
        print(f"\nTesting {duration}s audio...")
        
        # Create test audio
        audio = create_test_audio(duration=duration)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # Measure processing time
        start_time = time.time()
        
        try:
            # Process through ASR
            asr_result = asr_module(audio_tensor)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            real_time_factor = processing_time / duration
            words_per_second = len(asr_result.text.split()) / processing_time if processing_time > 0 else 0
            
            results.append({
                'duration': duration,
                'processing_time': processing_time,
                'real_time_factor': real_time_factor,
                'words_per_second': words_per_second,
                'confidence': asr_result.overall_confidence,
                'text_length': len(asr_result.text),
                'detected_languages': len(asr_result.detected_languages)
            })
            
            print(f"  ✓ Processing time: {processing_time:.2f}s")
            print(f"  ✓ Real-time factor: {real_time_factor:.2f}x")
            print(f"  ✓ Words per second: {words_per_second:.1f}")
            print(f"  ✓ Confidence: {asr_result.overall_confidence:.3f}")
            print(f"  ✓ Text: \"{asr_result.text[:50]}{'...' if len(asr_result.text) > 50 else ''}\"")
            
        except Exception as e:
            print(f"  ✗ Error processing {duration}s audio: {e}")
            results.append({
                'duration': duration,
                'processing_time': float('inf'),
                'real_time_factor': float('inf'),
                'words_per_second': 0,
                'confidence': 0,
                'text_length': 0,
                'detected_languages': 0
            })
    
    # Summary statistics
    print(f"\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r['processing_time'] != float('inf')]
    
    if successful_results:
        avg_processing_time = np.mean([r['processing_time'] for r in successful_results])
        avg_real_time_factor = np.mean([r['real_time_factor'] for r in successful_results])
        avg_confidence = np.mean([r['confidence'] for r in successful_results])
        
        print(f"Average processing time: {avg_processing_time:.2f}s")
        print(f"Average real-time factor: {avg_real_time_factor:.2f}x")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Estimate for 1 hour of audio
        hours_1_audio = 3600  # seconds
        estimated_time_1_hour = hours_1_audio * avg_real_time_factor / 3600  # hours
        
        print(f"\nEstimated time for 1 hour of audio: {estimated_time_1_hour:.1f} hours")
        print(f"Estimated time for 10 minutes of audio: {estimated_time_1_hour * 10 / 60:.1f} minutes")
        
        # Memory usage estimation
        print(f"\nMemory usage: ~2-4GB (Whisper large-v3 model)")
        print(f"Recommended: 8GB+ RAM for optimal performance")
        
    else:
        print("No successful processing results to analyze")
    
    # Detailed results table
    print(f"\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)
    print(f"{'Duration':<8} {'Time':<8} {'RT Factor':<10} {'WPS':<8} {'Confidence':<10}")
    print("-" * 60)
    
    for result in results:
        if result['processing_time'] != float('inf'):
            print(f"{result['duration']:<8.1f}s {result['processing_time']:<8.2f}s "
                  f"{result['real_time_factor']:<10.2f}x {result['words_per_second']:<8.1f} "
                  f"{result['confidence']:<10.3f}")
    else:
            print(f"{result['duration']:<8.1f}s {'FAILED':<8} {'N/A':<10} {'N/A':<8} {'N/A':<10}")

if __name__ == "__main__":
    test_asr_integration()
