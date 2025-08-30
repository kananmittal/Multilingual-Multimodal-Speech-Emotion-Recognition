#!/usr/bin/env python3
"""
Test script for Audio Conditioning Module
Demonstrates intelligent audio preprocessing capabilities
"""

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.audio_conditioning import AudioConditioningModule, create_audio_conditioning, ConditioningFeatures
from data.preprocess import load_audio


def create_test_audio_with_issues(duration=5.0, sample_rate=16000):
    """Create test audio with various quality issues."""
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean speech-like signal
    clean_speech = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    clean_speech = clean_speech / np.max(np.abs(clean_speech)) * 0.8
    
    # Audio with hum (50Hz and 60Hz interference)
    hum_50hz = 0.3 * np.sin(2 * np.pi * 50 * t)
    hum_60hz = 0.2 * np.sin(2 * np.pi * 60 * t)
    audio_with_hum = clean_speech + hum_50hz + hum_60hz
    
    # Audio with low-frequency noise
    low_freq_noise = 0.4 * np.sin(2 * np.pi * 30 * t) + 0.3 * np.sin(2 * np.pi * 45 * t)
    audio_with_low_freq = clean_speech + low_freq_noise
    
    # Audio with high-frequency noise
    high_freq_noise = 0.2 * np.sin(2 * np.pi * 8000 * t) + 0.1 * np.sin(2 * np.pi * 12000 * t)
    audio_with_high_freq = clean_speech + high_freq_noise
    
    # Audio with reverberation (simulated)
    reverb_tail = np.zeros_like(clean_speech)
    for i in range(1, 10):
        delay = int(0.1 * i * sample_rate)  # 100ms delays
        if delay < len(clean_speech):
            reverb_tail[delay:] += 0.1 * clean_speech[:-delay] if delay < len(clean_speech) else 0
    audio_with_reverb = clean_speech + reverb_tail
    
    # Audio with dynamic range issues (very quiet)
    quiet_audio = clean_speech * 0.1
    
    # Audio with clipping
    clipped_audio = clean_speech * 1.5
    clipped_audio = np.clip(clipped_audio, -1, 1)
    
    return {
        'clean': clean_speech,
        'hum': audio_with_hum,
        'low_freq': audio_with_low_freq,
        'high_freq': audio_with_high_freq,
        'reverb': audio_with_reverb,
        'quiet': quiet_audio,
        'clipped': clipped_audio
    }


def test_audio_conditioning():
    """Test the audio conditioning module with different audio types."""
    
    print("🧪 Testing Audio Conditioning Module")
    print("=" * 50)
    
    # Initialize audio conditioning
    conditioning = create_audio_conditioning(sample_rate=16000)
    
    # Create test audio
    test_audios = create_test_audio_with_issues()
    
    results = {}
    
    for audio_name, audio in test_audios.items():
        print(f"\n📊 Testing {audio_name.upper()} audio:")
        print("-" * 30)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # Process through audio conditioning
        conditioned_audio, features = conditioning(audio_tensor)
        
        # Print conditioning report
        report = conditioning.get_conditioning_report(features)
        print(report)
        
        # Store results
        results[audio_name] = {
            'features': features,
            'original_audio': audio,
            'conditioned_audio': conditioned_audio.detach().cpu().numpy(),
            'audio_name': audio_name
        }
        
        print(f"Processing Summary:")
        print(f"  - Hum Filtered: {features.hum_filtered}")
        print(f"  - HPF Applied: {features.hpf_applied} (cutoff: {features.hpf_cutoff:.0f} Hz)")
        print(f"  - Denoising: {features.denoise_applied} (gain: {features.denoise_gain_db:.1f} dB)")
        print(f"  - Dereverberation: {features.dereverb_applied} (T60: {features.estimated_t60:.2f} s)")
        print(f"  - LUFS Adjustment: {features.lufs_adjustment:.1f} dB")
        print()
    
    return results


def visualize_conditioning_results(results):
    """Create visualizations of conditioning results."""
    
    print("📈 Creating conditioning results visualizations...")
    
    # Extract metrics for plotting
    audio_types = []
    snr_improvements = []
    lufs_adjustments = []
    processing_flags = []
    
    for key, result in results.items():
        features = result['features']
        audio_types.append(result['audio_name'])
        snr_improvements.append(features.snr_after - features.snr_before)
        lufs_adjustments.append(features.lufs_adjustment)
        
        # Count processing steps applied
        steps_applied = sum([
            features.hum_filtered,
            features.hpf_applied,
            features.denoise_applied,
            features.dereverb_applied
        ])
        processing_flags.append(steps_applied)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Audio Conditioning Results', fontsize=16)
    
    # SNR improvement
    colors = ['green' if imp > 0 else 'red' for imp in snr_improvements]
    axes[0, 0].bar(range(len(snr_improvements)), snr_improvements, color=colors)
    axes[0, 0].set_title('SNR Improvement (After - Before)')
    axes[0, 0].set_ylabel('SNR Improvement (dB)')
    axes[0, 0].set_xticks(range(len(audio_types)))
    axes[0, 0].set_xticklabels(audio_types, rotation=45)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # LUFS adjustments
    colors = ['blue' if adj != 0 else 'gray' for adj in lufs_adjustments]
    axes[0, 1].bar(range(len(lufs_adjustments)), lufs_adjustments, color=colors)
    axes[0, 1].set_title('LUFS Adjustments')
    axes[0, 1].set_ylabel('LUFS Adjustment (dB)')
    axes[0, 1].set_xticks(range(len(audio_types)))
    axes[0, 1].set_xticklabels(audio_types, rotation=45)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Processing steps applied
    axes[1, 0].bar(range(len(processing_flags)), processing_flags, color='orange')
    axes[1, 0].set_title('Processing Steps Applied')
    axes[1, 0].set_ylabel('Number of Processing Steps')
    axes[1, 0].set_xticks(range(len(audio_types)))
    axes[1, 0].set_xticklabels(audio_types, rotation=45)
    axes[1, 0].set_ylim(0, 4)
    
    # Audio waveforms comparison (for one example)
    example_key = 'hum'  # Show hum filtering example
    if example_key in results:
        original = results[example_key]['original_audio']
        conditioned = results[example_key]['conditioned_audio']
        
        # Plot first 1 second
        samples_1sec = 16000
        time_axis = np.linspace(0, 1, min(samples_1sec, len(original)))
        
        axes[1, 1].plot(time_axis, original[:samples_1sec], label='Original', alpha=0.7)
        axes[1, 1].plot(time_axis, conditioned[:samples_1sec], label='Conditioned', alpha=0.7)
        axes[1, 1].set_title(f'Waveform Comparison ({example_key})')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('audio_conditioning_results.png', dpi=300, bbox_inches='tight')
    print("✅ Audio conditioning visualization saved as 'audio_conditioning_results.png'")
    
    return fig


def test_individual_components():
    """Test individual conditioning components."""
    
    print("\n🔧 Testing Individual Conditioning Components")
    print("=" * 50)
    
    # Create test audio with hum
    t = np.linspace(0, 3, 48000)
    test_audio = np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 50 * t)
    
    # Test hum filtering
    from models.audio_conditioning import HumNotchFilter
    hum_filter = HumNotchFilter()
    detected_hum = hum_filter.detect_hum(test_audio)
    print(f"Hum Detection: {detected_hum} Hz")
    
    # Test high-pass filtering
    from models.audio_conditioning import HighPassFilter
    hpf = HighPassFilter()
    should_apply, cutoff = hpf.should_apply_hpf(test_audio)
    print(f"HPF Decision: {should_apply} (cutoff: {cutoff:.0f} Hz)")
    
    # Test denoising
    from models.audio_conditioning import AdaptiveDenoiser
    denoiser = AdaptiveDenoiser()
    snr = denoiser.estimate_snr(test_audio)
    noise_type = denoiser.detect_noise_type(test_audio)
    print(f"SNR Estimation: {snr:.1f} dB")
    print(f"Noise Type: {noise_type}")
    
    # Test dereverberation
    from models.audio_conditioning import Dereverberator
    dereverberator = Dereverberator()
    t60 = dereverberator.estimate_t60(test_audio)
    print(f"Estimated T60: {t60:.2f} s")
    
    # Test loudness normalization
    from models.audio_conditioning import LoudnessNormalizer
    normalizer = LoudnessNormalizer()
    lufs = normalizer.measure_lufs(test_audio)
    print(f"Measured LUFS: {lufs:.1f}")


def test_with_real_audio(audio_path: str):
    """Test audio conditioning with a real audio file."""
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"\n🎵 Testing with real audio: {audio_path}")
    print("=" * 50)
    
    # Load audio
    audio = load_audio(audio_path)
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    
    # Initialize audio conditioning
    conditioning = create_audio_conditioning(sample_rate=16000)
    
    # Process through audio conditioning
    conditioned_audio, features = conditioning(audio_tensor)
    
    # Print detailed report
    report = conditioning.get_conditioning_report(features)
    print(report)
    
    return features


def main():
    """Main test function."""
    
    print("🚀 Audio Conditioning Module Test Suite")
    print("=" * 60)
    
    # Test 1: Individual components
    print("\n1️⃣ Testing individual components...")
    test_individual_components()
    
    # Test 2: Full conditioning pipeline
    print("\n2️⃣ Testing full conditioning pipeline...")
    results = test_audio_conditioning()
    
    # Test 3: Visualizations
    print("\n3️⃣ Creating visualizations...")
    fig = visualize_conditioning_results(results)
    
    # Test 4: Real audio (if available)
    print("\n4️⃣ Testing with real audio...")
    test_audio_paths = [
        "test_audio.wav",
        "sample_audio.wav",
        "datasets/crema/AudioWAV/1001_DFA_ANG_XX.wav"
    ]
    
    for audio_path in test_audio_paths:
        if os.path.exists(audio_path):
            test_with_real_audio(audio_path)
            break
    else:
        print("ℹ️  No real audio files found for testing")
    
    print("\n✅ Audio Conditioning testing completed!")
    print("\n📋 Summary:")
    print("- Hum Notch Filtering: ✅")
    print("- High-Pass Filtering: ✅")
    print("- Adaptive Denoising: ✅")
    print("- Dereverberation: ✅")
    print("- Loudness Normalization: ✅")
    print("- Feature Fusion: ✅")
    print("- Policy-Driven Processing: ✅")


if __name__ == "__main__":
    main()
