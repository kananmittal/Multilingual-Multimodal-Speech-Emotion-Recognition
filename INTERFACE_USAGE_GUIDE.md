# Emotion Recognition Interface - Usage Guide

## 🚀 Quick Start

### 1. Command Line Interface

#### Basic Usage
```bash
# Text-only emotion recognition
python src/interface.py --checkpoint checkpoints/best_model.pt --text "I'm feeling happy today!"

# Audio-only emotion recognition
python src/interface.py --checkpoint checkpoints/best_model.pt --audio samples/audio.wav

# Multimodal emotion recognition
python src/interface.py --checkpoint checkpoints/best_model.pt --audio samples/audio.wav --text "I'm excited about this!"

# Save results to JSON
python src/interface.py --checkpoint checkpoints/best_model.pt --text "I'm sad" --output results.json

# Generate visualization
python src/interface.py --checkpoint checkpoints/best_model.pt --text "I'm angry" --visualize --save_viz analysis.png
```

#### Advanced Options
```bash
# Disable test-time augmentation
python src/interface.py --checkpoint checkpoints/best_model.pt --text "Hello" --no_tta

# Use specific device
python src/interface.py --checkpoint checkpoints/best_model.pt --text "Hello" --device cuda

# Full example with all options
python src/interface.py \
    --checkpoint checkpoints/best_model.pt \
    --audio samples/audio.wav \
    --text "I'm feeling great!" \
    --output results.json \
    --visualize \
    --save_viz analysis.png \
    --device cuda
```

### 2. Python API

#### Basic Usage
```python
from src.interface import EmotionRecognitionInterface

# Initialize interface
interface = EmotionRecognitionInterface("checkpoints/best_model.pt")

# Text-only prediction
results = interface.predict_emotion(text="I'm feeling happy today!")
print(f"Emotion: {results['emotion_labels'][0]}")
print(f"Confidence: {results['confidence'][0]:.3f}")

# Audio-only prediction
results = interface.predict_emotion(audio_path="samples/audio.wav")
print(f"Emotion: {results['emotion_labels'][0]}")

# Multimodal prediction
results = interface.predict_emotion(
    audio_path="samples/audio.wav",
    text="I'm excited about this!"
)
print(f"Emotion: {results['emotion_labels'][0]}")
```

#### Advanced Usage
```python
# Batch processing
texts = ["I'm happy", "I'm sad", "I'm angry", "I'm neutral"]
results = interface.batch_predict(texts=texts)

# Save results
interface.save_results(results, "batch_results.json")

# Generate visualization
interface.visualize_results(results, save_path="analysis.png")

# Custom options
results = interface.predict_emotion(
    text="I'm not sure how I feel",
    use_tta=False,  # Disable test-time augmentation
    return_detailed=True  # Get detailed analysis
)
```

---

## 📊 Output Format

### Basic Results
```python
{
    'predictions': [1],  # Emotion indices [0: Neutral, 1: Happy, 2: Sad, 3: Angry]
    'probabilities': [[0.1, 0.8, 0.05, 0.05]],  # Probability distribution
    'confidence': [0.85],  # Confidence scores
    'uncertainty': [0.15],  # Uncertainty scores
    'logits': [[-2.3, 1.5, -3.1, -2.8]],  # Raw logits
    'anchor_loss': 0.023,  # Clustering loss
    'emotion_labels': ['Happy'],  # Human-readable labels
    'modalities': {
        'audio': True,
        'text': True
    }
}
```

### Detailed Results (with `return_detailed=True`)
```python
{
    # ... basic results ...
    'top_k_predictions': {
        'indices': [[1, 0]],  # Top-2 emotion indices
        'probabilities': [[0.8, 0.1]],  # Top-2 probabilities
        'labels': [['Happy', 'Neutral']]  # Top-2 labels
    },
    'entropy': [0.45],  # Prediction entropy
    'margin': [0.7],  # Margin between top-1 and top-2
    'calibration_error': 0.023,  # Calibration error
    'analysis': {
        'high_confidence': [True],  # Confidence > 0.8
        'low_confidence': [False],  # Confidence < 0.5
        'high_entropy': [False],    # Entropy > 1.0
        'low_margin': [False]       # Margin < 0.3
    }
}
```

---

## 🎯 Use Cases

### 1. Text-Only Emotion Recognition
```python
# Analyze text sentiment/emotion
text = "I'm feeling really excited about the new project!"
results = interface.predict_emotion(text=text)

print(f"Detected emotion: {results['emotion_labels'][0]}")
print(f"Confidence: {results['confidence'][0]:.3f}")
```

### 2. Audio-Only Emotion Recognition
```python
# Analyze speech emotion from audio file
audio_path = "speech_sample.wav"
results = interface.predict_emotion(audio_path=audio_path)

print(f"Speech emotion: {results['emotion_labels'][0]}")
print(f"Uncertainty: {results['uncertainty'][0]:.3f}")
```

### 3. Multimodal Emotion Recognition
```python
# Combine audio and text for robust emotion detection
results = interface.predict_emotion(
    audio_path="speech.wav",
    text="I'm feeling great today!"
)

# Check for modality mismatch
if results['analysis']['high_entropy'][0]:
    print("⚠️  High uncertainty - possible audio-text mismatch")
```

### 4. Batch Processing
```python
# Process multiple samples efficiently
texts = [
    "I'm so happy!",
    "This is disappointing.",
    "I'm furious!",
    "The weather is nice."
]

results = interface.batch_predict(texts=texts)

for i, result in enumerate(results):
    print(f"Sample {i+1}: {result['emotion_labels'][0]} "
          f"(Confidence: {result['confidence'][0]:.3f})")
```

### 5. Uncertainty Analysis
```python
# Analyze prediction reliability
results = interface.predict_emotion(text="I'm not sure how I feel")

if results['uncertainty'][0] > 0.5:
    print("⚠️  High uncertainty - prediction may be unreliable")

if results['analysis']['low_margin'][0]:
    print("⚠️  Low margin - multiple emotions possible")
```

---

## 📈 Visualization

### Generate Comprehensive Plots
```python
# Create detailed visualization
results = interface.predict_emotion(text="I'm feeling wonderful!")
interface.visualize_results(results, save_path="emotion_analysis.png")
```

### Visualization Components
1. **Emotion Probabilities**: Bar chart of emotion probabilities
2. **Confidence Analysis**: Confidence vs uncertainty comparison
3. **Raw Logits**: Raw classification scores
4. **Top-K Predictions**: Top emotion predictions
5. **Analysis Flags**: Quality indicators
6. **Summary Statistics**: Complete prediction summary

---

## ⚙️ Configuration Options

### Model Parameters
```python
# Initialize with custom settings
interface = EmotionRecognitionInterface(
    checkpoint_path="checkpoints/best_model.pt",
    device="cuda"  # or "cpu"
)
```

### Prediction Options
```python
results = interface.predict_emotion(
    audio_path="audio.wav",
    text="Hello",
    use_tta=True,        # Test-time augmentation
    return_detailed=True  # Detailed analysis
)
```

### Batch Processing Options
```python
results = interface.batch_predict(
    audio_paths=["audio1.wav", "audio2.wav"],
    texts=["Hello", "Goodbye"],
    use_tta=True
)
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Checkpoint Not Found
```bash
# Error: Checkpoint not found
# Solution: Update checkpoint path
python src/interface.py --checkpoint /correct/path/to/checkpoint.pt --text "Hello"
```

#### 2. Audio File Issues
```bash
# Error: Audio file not found or corrupted
# Solution: Check file path and format
# Supported formats: WAV, MP3, FLAC
```

#### 3. Memory Issues
```bash
# Error: CUDA out of memory
# Solution: Use CPU or reduce batch size
python src/interface.py --checkpoint model.pt --text "Hello" --device cpu
```

#### 4. Import Errors
```bash
# Error: Module not found
# Solution: Install dependencies
pip install torch torchaudio transformers matplotlib seaborn
```

### Performance Tips

1. **Use GPU**: Set `device="cuda"` for faster inference
2. **Batch Processing**: Use `batch_predict()` for multiple samples
3. **Disable TTA**: Use `use_tta=False` for faster single predictions
4. **Memory Management**: Use CPU for large batches if GPU memory is limited

---

## 📝 Examples

### Example 1: Simple Text Analysis
```python
from src.interface import EmotionRecognitionInterface

interface = EmotionRecognitionInterface("checkpoints/best_model.pt")

text = "I'm feeling really happy and excited about the future!"
results = interface.predict_emotion(text=text)

print(f"🎭 Emotion: {results['emotion_labels'][0]}")
print(f"🎯 Confidence: {results['confidence'][0]:.3f}")
print(f"📊 Probabilities:")
for emotion, prob in zip(['Neutral', 'Happy', 'Sad', 'Angry'], results['probabilities'][0]):
    print(f"  {emotion}: {prob:.3f}")
```

### Example 2: Audio Analysis with Visualization
```python
results = interface.predict_emotion(
    audio_path="speech.wav",
    use_tta=True,
    return_detailed=True
)

# Generate visualization
interface.visualize_results(results, save_path="speech_analysis.png")

# Save results
interface.save_results(results, "speech_results.json")
```

### Example 3: Batch Processing with Analysis
```python
texts = [
    "I'm so excited!",
    "This is terrible.",
    "I'm really angry!",
    "The weather is nice."
]

results = interface.batch_predict(texts=texts)

# Analyze batch results
high_confidence_count = sum(1 for r in results if r['confidence'][0] > 0.8)
print(f"High confidence predictions: {high_confidence_count}/{len(results)}")

# Save batch results
interface.save_results(results, "batch_analysis.json")
```

---

This interface provides a comprehensive and easy-to-use API for your advanced emotion recognition system! 🚀


