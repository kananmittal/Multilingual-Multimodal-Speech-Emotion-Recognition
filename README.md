# Multilingual Multimodal Speech Emotion Recognition

A state-of-the-art multimodal emotion recognition system that combines audio and text features for robust emotion classification across multiple languages.

## 🚀 Features

- **Multimodal Fusion**: Combines audio (Wav2Vec2) and text (XLM-RoBERTa) features
- **Cross-Modal Attention**: Bidirectional attention between audio and text modalities
- **Advanced Loss Functions**: Label smoothing, class-balanced focal loss, supervised contrastive learning
- **Prototype Memory**: Learnable emotion prototypes for better clustering
- **OOD Detection**: Advanced OpenMax with Class Anchor Clustering and deep architecture (35 layers)
- **Test-Time Augmentation**: Speed perturbation and noise addition for robust inference
- **Temperature Scaling**: Calibrated probability outputs

## 🏗️ Architecture

### 1. Audio Encoder
- **Model**: Wav2Vec2-Base (multilingual)
- **Features**: 768-dimensional embeddings
- **Adapter**: Lightweight bottleneck layers (768→256→768)
- **Pooling**: Attentive Statistics Pooling (mean + std)

### 2. Text Encoder  
- **Model**: XLM-RoBERTa-Base (multilingual)
- **Features**: 768-dimensional embeddings
- **Adapter**: Lightweight bottleneck layers (768→256→768)
- **Pooling**: Attentive Statistics Pooling (mean + std)

### 3. Cross-Modal Attention
- **Mechanism**: Multi-head attention (8 heads, 256 dim)
- **Direction**: Bidirectional (audio↔text)
- **Output**: Enhanced modality-specific features

### 4. Fusion & Classification
- **Projection**: MLP layers to common dimension (512)
- **Gated Fusion**: Learnable weights for modality combination
- **Classifier**: Advanced OpenMax with 35-layer deep architecture + Class Anchor Clustering
- **Features**: Residual connections, LayerNorm, Dropout (0.15), Uncertainty estimation
- **Output**: 4 emotion classes + uncertainty estimation + anchor clustering

## 📊 Supported Datasets

- **RAVDESS**: 1,440 samples (speech modality)
- **CREMA-D**: 7,442 samples (when available)
- **IEMOCAP**: Coming soon
- **Zero-shot**: Hindi, Bengali, Telugu SER corpora

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/kananmittal/Multilingual-Multimodal-Speech-Emotion-Recognition.git
cd Multilingual-Multimodal-Speech-Emotion-Recognition

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
├── src/
│   ├── models/
│   │   ├── audio_encoder.py      # Wav2Vec2 with adapters
│   │   ├── text_encoder.py       # XLM-RoBERTa with adapters
│   │   ├── cross_attention.py    # Cross-modal attention
│   │   ├── pooling.py           # Attentive statistics pooling
│   │   ├── fusion.py            # Gated fusion layer
│   │   ├── classifier.py        # Advanced OpenMax with 35-layer deep architecture + Class Anchor Clustering
│   │   ├── prototypes.py        # Prototype memory
│   │   └── losses.py            # Advanced loss functions
│   ├── data/
│   │   ├── dataset.py           # Dataset loader
│   │   ├── dataset_loader.py    # Multi-dataset loader
│   │   └── preprocess.py        # Audio preprocessing & augmentation
│   ├── train.py                 # Training script
│   ├── eval.py                  # Evaluation script
│   └── utils.py                 # Utility functions
├── datasets/                    # Dataset directories
├── checkpoints/                 # Model checkpoints
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Prepare Data
```bash
# Create manifests from your datasets
python src/data/dataset_loader.py
```

### 2. Train Model
```bash
# Basic training
python src/train.py --epochs 30 --batch_size 8 --use_amp --augment

# Advanced training with custom parameters
python src/train.py \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_amp \
    --augment \
    --proto_weight 0.01
```

### 3. Evaluate Model
```bash
# Basic evaluation
python src/eval.py --manifest val_manifest.jsonl --checkpoint checkpoints/best_model.pt

# Advanced evaluation with TTA and calibration
python src/eval.py \
    --manifest test_manifest.jsonl \
    --checkpoint checkpoints/best_model.pt \
    --use_tta \
    --calibrate \
    --val_manifest val_manifest.jsonl
```

## 📈 Performance

Current results on RAVDESS validation set:
- **Weighted F1**: 0.23 (needs improvement)
- **Per-class Accuracy**: 
  - Neutral: 0.00
  - Happy: 1.00  
  - Sad: 0.00
  - Angry: 0.00

**Note**: Model shows signs of class collapse - currently investigating and improving.

## 🔧 Configuration

### Training Parameters
- **Learning Rate**: 1e-4 (reduced from 3e-4)
- **Batch Size**: 8-16 (depending on GPU memory)
- **Epochs**: 30-50
- **Warmup**: 10% of total steps
- **Scheduler**: Cosine decay

### Loss Weights
- **CE + Label Smoothing**: 1.0
- **Class-Balanced Focal**: 0.3
- **Supervised Contrastive**: 0.1 (disabled for now)
- **Prototype Loss**: 0.01

### Augmentations
- **Speed Perturb**: ±10% (50% chance)
- **Noise Addition**: SNR 10-20dB (50% chance)
- **Test-Time**: 5 augmentations averaged

## 🎯 Future Improvements

1. **Fix Class Collapse**: Investigate loss function balance
2. **Add More Datasets**: IEMOCAP, CREMA-D (full version)
3. **Stronger Backbones**: WavLM-Large, HuBERT-Large
4. **Better Augmentations**: SpecAugment, Mixup
5. **Ensemble Methods**: Model averaging, snapshot ensemble
6. **Zero-shot Evaluation**: Test on Hindi/Bengali/Telugu

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{multilingual_ser_2025,
  title={Multilingual Multimodal Speech Emotion Recognition},
  author={Kanan Mittal},
  year={2025},
  url={https://github.com/kananmittal/Multilingual-Multimodal-Speech-Emotion-Recognition}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

