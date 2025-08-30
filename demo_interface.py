#!/usr/bin/env python3
"""
Demo script for Emotion Recognition Interface
Shows how to use the interface for different scenarios
"""

import os
import sys
from src.interface import EmotionRecognitionInterface

def demo_text_only():
    """Demo for text-only emotion recognition"""
    print("="*60)
    print("📝 TEXT-ONLY EMOTION RECOGNITION DEMO")
    print("="*60)
    
    # Initialize interface (you'll need to provide checkpoint path)
    checkpoint_path = "checkpoints/best_model.pt"  # Update this path
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path in the script")
        return
    
    interface = EmotionRecognitionInterface(checkpoint_path)
    
    # Test texts with different emotions
    test_texts = [
        "I am so happy today! Everything is going great!",
        "I feel really sad and lonely right now.",
        "I'm angry about what happened yesterday.",
        "The weather is nice today.",
        "I don't know how I feel about this situation."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Sample {i}: '{text}'")
        print("-" * 40)
        
        try:
            results = interface.predict_emotion(text=text, use_tta=False)
            
            print(f"🎭 Primary Emotion: {results['emotion_labels'][0]}")
            print(f"🎯 Confidence: {results['confidence'][0]:.3f}")
            print(f"❓ Uncertainty: {results['uncertainty'][0]:.3f}")
            
            print("📊 Probabilities:")
            for emotion, prob in zip(['Neutral', 'Happy', 'Sad', 'Angry'], results['probabilities'][0]):
                print(f"  {emotion}: {prob:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_audio_only():
    """Demo for audio-only emotion recognition"""
    print("="*60)
    print("🎵 AUDIO-ONLY EMOTION RECOGNITION DEMO")
    print("="*60)
    
    checkpoint_path = "checkpoints/best_model.pt"  # Update this path
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    interface = EmotionRecognitionInterface(checkpoint_path)
    
    # Test audio files (you'll need to provide actual audio files)
    audio_files = [
        "samples/happy_audio.wav",
        "samples/sad_audio.wav", 
        "samples/angry_audio.wav",
        "samples/neutral_audio.wav"
    ]
    
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"⚠️  Audio file not found: {audio_file}")
            continue
            
        print(f"\n🎵 Processing: {audio_file}")
        print("-" * 40)
        
        try:
            results = interface.predict_emotion(audio_path=audio_file, use_tta=True)
            
            print(f"🎭 Primary Emotion: {results['emotion_labels'][0]}")
            print(f"🎯 Confidence: {results['confidence'][0]:.3f}")
            print(f"❓ Uncertainty: {results['uncertainty'][0]:.3f}")
            
            print("📊 Probabilities:")
            for emotion, prob in zip(['Neutral', 'Happy', 'Sad', 'Angry'], results['probabilities'][0]):
                print(f"  {emotion}: {prob:.3f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_multimodal():
    """Demo for multimodal emotion recognition"""
    print("="*60)
    print("🎭 MULTIMODAL EMOTION RECOGNITION DEMO")
    print("="*60)
    
    checkpoint_path = "checkpoints/best_model.pt"  # Update this path
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    interface = EmotionRecognitionInterface(checkpoint_path)
    
    # Test multimodal samples
    multimodal_samples = [
        {
            "audio": "samples/happy_audio.wav",
            "text": "I'm so excited about this!",
            "description": "Happy audio + happy text"
        },
        {
            "audio": "samples/sad_audio.wav", 
            "text": "I'm feeling great today!",
            "description": "Sad audio + happy text (mismatch)"
        },
        {
            "audio": "samples/angry_audio.wav",
            "text": "I'm really frustrated with this situation.",
            "description": "Angry audio + angry text"
        }
    ]
    
    for sample in multimodal_samples:
        audio_file = sample["audio"]
        text = sample["text"]
        description = sample["description"]
        
        if not os.path.exists(audio_file):
            print(f"⚠️  Audio file not found: {audio_file}")
            continue
            
        print(f"\n🎭 {description}")
        print(f"🎵 Audio: {audio_file}")
        print(f"📝 Text: '{text}'")
        print("-" * 50)
        
        try:
            results = interface.predict_emotion(
                audio_path=audio_file, 
                text=text, 
                use_tta=True
            )
            
            print(f"🎭 Primary Emotion: {results['emotion_labels'][0]}")
            print(f"🎯 Confidence: {results['confidence'][0]:.3f}")
            print(f"❓ Uncertainty: {results['uncertainty'][0]:.3f}")
            print(f"🔧 Calibration Error: {results['calibration_error']:.3f}")
            print(f"📌 Anchor Loss: {results['anchor_loss']:.3f}")
            
            print("📊 Probabilities:")
            for emotion, prob in zip(['Neutral', 'Happy', 'Sad', 'Angry'], results['probabilities'][0]):
                print(f"  {emotion}: {prob:.3f}")
            
            print("🔍 Top-K Predictions:")
            for i, (label, prob) in enumerate(zip(results['top_k_predictions']['labels'][0], 
                                                 results['top_k_predictions']['probabilities'][0])):
                print(f"  {i+1}. {label}: {prob:.3f}")
            
            print("⚠️  Analysis Flags:")
            analysis = results['analysis']
            flags = ['High Confidence', 'Low Confidence', 'High Entropy', 'Low Margin']
            values = [analysis['high_confidence'][0], analysis['low_confidence'][0], 
                     analysis['high_entropy'][0], analysis['low_margin'][0]]
            
            for flag, value in zip(flags, values):
                status = "✓" if value else "✗"
                print(f"  {flag}: {status}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_batch_processing():
    """Demo for batch processing"""
    print("="*60)
    print("📦 BATCH PROCESSING DEMO")
    print("="*60)
    
    checkpoint_path = "checkpoints/best_model.pt"  # Update this path
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    interface = EmotionRecognitionInterface(checkpoint_path)
    
    # Batch of texts
    texts = [
        "I'm feeling wonderful today!",
        "This is really disappointing.",
        "I'm furious about what happened!",
        "The weather is quite pleasant.",
        "I'm not sure how to feel about this."
    ]
    
    print("📝 Processing batch of texts...")
    
    try:
        results = interface.batch_predict(texts=texts, use_tta=False)
        
        print(f"\n✅ Processed {len(results)} samples")
        print("\n📊 Batch Results Summary:")
        print("-" * 40)
        
        for i, result in enumerate(results, 1):
            print(f"Sample {i}: {result['emotion_labels'][0]} "
                  f"(Confidence: {result['confidence'][0]:.3f}, "
                  f"Uncertainty: {result['uncertainty'][0]:.3f})")
        
        # Save batch results
        interface.save_results(results, "batch_results.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_visualization():
    """Demo for result visualization"""
    print("="*60)
    print("📊 VISUALIZATION DEMO")
    print("="*60)
    
    checkpoint_path = "checkpoints/best_model.pt"  # Update this path
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    interface = EmotionRecognitionInterface(checkpoint_path)
    
    # Test sample
    text = "I'm feeling really happy and excited about the future!"
    
    print(f"📝 Analyzing: '{text}'")
    
    try:
        results = interface.predict_emotion(text=text, use_tta=False)
        
        print("🎭 Generating visualization...")
        interface.visualize_results(results, save_path="emotion_analysis.png")
        
        print("✅ Visualization completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main demo function"""
    print("🎭 EMOTION RECOGNITION INTERFACE DEMO")
    print("="*60)
    print("This demo shows how to use the emotion recognition interface")
    print("for different scenarios: text-only, audio-only, multimodal, etc.")
    print("="*60)
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or update the checkpoint path")
        print("You can still see the interface structure and usage examples")
        print()
    
    # Run demos
    try:
        demo_text_only()
        print("\n" + "="*60)
        
        demo_audio_only()
        print("\n" + "="*60)
        
        demo_multimodal()
        print("\n" + "="*60)
        
        demo_batch_processing()
        print("\n" + "="*60)
        
        demo_visualization()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("This is expected if the model checkpoint is not available")
    
    print("\n" + "="*60)
    print("🎉 Demo completed!")
    print("Check the generated files:")
    print("- batch_results.json: Batch processing results")
    print("- emotion_analysis.png: Visualization of results")
    print("="*60)

if __name__ == "__main__":
    main()


