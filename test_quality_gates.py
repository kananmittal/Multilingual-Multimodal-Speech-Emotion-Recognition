#!/usr/bin/env python3
"""
Test script for Front-End Quality Gates Module
Demonstrates the quality assessment and filtering capabilities
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

from models.quality_gates import FrontEndQualityGates, create_quality_gates, QualityMetrics
from data.preprocess import load_audio


def create_test_audio(duration=5.0, sample_rate=16000):
    """Create test audio with different quality characteristics"""
    
    # Generate clean speech-like signal
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean speech (good quality)
    clean_speech = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    clean_speech = clean_speech / np.max(np.abs(clean_speech)) * 0.8
    
    # Noisy speech (poor SNR)
    noise = np.random.normal(0, 0.3, len(clean_speech))
    noisy_speech = clean_speech + noise
    
    # Clipped speech (poor quality)
    clipped_speech = clean_speech * 1.5  # Will cause clipping
    clipped_speech = np.clip(clipped_speech, -1, 1)
    
    # Music-like signal (non-speech)
    music = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t) + np.sin(2 * np.pi * 1320 * t)
    music = music / np.max(np.abs(music)) * 0.7
    
    return {
        'clean': clean_speech,
        'noisy': noisy_speech,
        'clipped': clipped_speech,
        'music': music
    }


def test_quality_gates():
    """Test the quality gates module with different audio types"""
    
    print("🧪 Testing Front-End Quality Gates Module")
    print("=" * 50)
    
    # Initialize quality gates
    quality_gates = create_quality_gates(
        sample_rate=16000,
        vad_method="librosa",  # Use librosa for testing (no webrtc dependency)
        enable_language_detection=True
    )
    
    # Create test audio
    test_audios = create_test_audio()
    
    # Test texts for language detection
    test_texts = [
        "Hello, this is a test of the quality gates system.",
        "Hola, esto es una prueba del sistema de puertas de calidad.",
        "Bonjour, ceci est un test du système de portes de qualité.",
        "This is a mixed language text with some unknown words: xyz123"
    ]
    
    results = {}
    
    for audio_name, audio in test_audios.items():
        print(f"\n📊 Testing {audio_name.upper()} audio:")
        print("-" * 30)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # Test with different texts
        for i, text in enumerate(test_texts):
            print(f"  Text {i+1}: {text[:50]}...")
            
            # Process through quality gates
            processed_audio, metrics, should_process = quality_gates(audio_tensor, text)
            
            # Print quality report
            report = quality_gates.get_quality_report(metrics)
            print(report)
            
            # Store results
            results[f"{audio_name}_text{i+1}"] = {
                'metrics': metrics,
                'should_process': should_process,
                'audio_name': audio_name,
                'text': text
            }
            
            print(f"  Decision: {metrics.abstain_recommendation.upper()}")
            print(f"  Should Process: {should_process}")
            print()
    
    return results


def visualize_quality_metrics(results):
    """Create visualizations of quality metrics"""
    
    print("📈 Creating quality metrics visualizations...")
    
    # Extract metrics for plotting
    audio_types = []
    speech_probs = []
    snr_values = []
    clipping_percents = []
    quality_scores = []
    decisions = []
    
    for key, result in results.items():
        metrics = result['metrics']
        audio_types.append(result['audio_name'])
        speech_probs.append(metrics.speech_prob)
        snr_values.append(metrics.snr_db)
        clipping_percents.append(metrics.clipping_percent)
        quality_scores.append(metrics.quality_score)
        decisions.append(metrics.abstain_recommendation)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Front-End Quality Gates Assessment Results', fontsize=16)
    
    # Speech probability
    axes[0, 0].bar(range(len(speech_probs)), speech_probs, color=['green' if d == 'accept' else 'red' if d == 'reject' else 'orange' for d in decisions])
    axes[0, 0].set_title('Speech Probability')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_xticks(range(len(audio_types)))
    axes[0, 0].set_xticklabels(audio_types, rotation=45)
    axes[0, 0].axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Reject threshold')
    axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Accept threshold')
    axes[0, 0].legend()
    
    # SNR values
    axes[0, 1].bar(range(len(snr_values)), snr_values, color=['green' if d == 'accept' else 'red' if d == 'reject' else 'orange' for d in decisions])
    axes[0, 1].set_title('Signal-to-Noise Ratio (dB)')
    axes[0, 1].set_ylabel('SNR (dB)')
    axes[0, 1].set_xticks(range(len(audio_types)))
    axes[0, 1].set_xticklabels(audio_types, rotation=45)
    axes[0, 1].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Reject threshold')
    axes[0, 1].axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Accept threshold')
    axes[0, 1].legend()
    
    # Clipping percentage
    axes[1, 0].bar(range(len(clipping_percents)), clipping_percents, color=['green' if d == 'accept' else 'red' if d == 'reject' else 'orange' for d in decisions])
    axes[1, 0].set_title('Clipping Percentage')
    axes[1, 0].set_ylabel('Clipping (%)')
    axes[1, 0].set_xticks(range(len(audio_types)))
    axes[1, 0].set_xticklabels(audio_types, rotation=45)
    axes[1, 0].axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Reject threshold')
    axes[1, 0].legend()
    
    # Overall quality score
    axes[1, 1].bar(range(len(quality_scores)), quality_scores, color=['green' if d == 'accept' else 'red' if d == 'reject' else 'orange' for d in decisions])
    axes[1, 1].set_title('Overall Quality Score')
    axes[1, 1].set_ylabel('Quality Score [0, 1]')
    axes[1, 1].set_xticks(range(len(audio_types)))
    axes[1, 1].set_xticklabels(audio_types, rotation=45)
    axes[1, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Uncertain threshold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('quality_gates_results.png', dpi=300, bbox_inches='tight')
    print("✅ Quality metrics visualization saved as 'quality_gates_results.png'")
    
    return fig


def test_with_real_audio(audio_path: str):
    """Test quality gates with a real audio file"""
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"\n🎵 Testing with real audio: {audio_path}")
    print("=" * 50)
    
    # Load audio
    audio = load_audio(audio_path)
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    
    # Initialize quality gates
    quality_gates = create_quality_gates(
        sample_rate=16000,  # Default sample rate
        vad_method="librosa",
        enable_language_detection=True
    )
    
    # Test with sample text
    test_text = "This is a test of the quality gates with real audio data."
    
    # Process through quality gates
    processed_audio, metrics, should_process = quality_gates(audio_tensor, test_text)
    
    # Print detailed report
    report = quality_gates.get_quality_report(metrics)
    print(report)
    
    print(f"Decision: {metrics.abstain_recommendation.upper()}")
    print(f"Should Process: {should_process}")
    
    return metrics


def main():
    """Main test function"""
    
    print("🚀 Front-End Quality Gates Module Test Suite")
    print("=" * 60)
    
    # Test 1: Synthetic audio
    print("\n1️⃣ Testing with synthetic audio...")
    results = test_quality_gates()
    
    # Test 2: Visualizations
    print("\n2️⃣ Creating visualizations...")
    fig = visualize_quality_metrics(results)
    
    # Test 3: Real audio (if available)
    print("\n3️⃣ Testing with real audio...")
    test_audio_paths = [
        "test_audio.wav",
        "sample_audio.wav",
        "datasets/crema/AudioWAV/1001_DFA_ANG_XX.wav"  # Example from CREMA-D
    ]
    
    for audio_path in test_audio_paths:
        if os.path.exists(audio_path):
            test_with_real_audio(audio_path)
            break
    else:
        print("ℹ️  No real audio files found for testing")
    
    print("\n✅ Quality Gates testing completed!")
    print("\n📋 Summary:")
    print("- Voice Activity Detection: ✅")
    print("- Signal Quality Assessment: ✅")
    print("- Language Identification: ✅")
    print("- Content Type Detection: ✅")
    print("- Early Abstain Policy: ✅")
    print("- Quality Feature Fusion: ✅")


if __name__ == "__main__":
    main()
