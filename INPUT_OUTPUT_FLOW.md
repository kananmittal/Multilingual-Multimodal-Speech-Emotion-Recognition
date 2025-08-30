# Input/Output Flow in Advanced OpenMax Classifier

## 🔄 Complete Data Flow

```
INPUT → PROCESSING → OUTPUT
```

---

## 📥 INPUT

### Where to Place Input:
```python
# Input: Fused multimodal features from fusion layer
x = torch.Tensor([batch_size, 512])  # 512-dimensional fused features

# Example usage:
logits, uncertainty, anchor_loss = classifier(x, use_openmax=True, return_uncertainty=True)
```

### Input Requirements:
- **Shape**: `[batch_size, 512]`
- **Type**: `torch.Tensor`
- **Source**: Output from `FusionLayer` (audio + text features)
- **Data Type**: Float32

---

## 🔄 PROCESSING PIPELINE

### 1. Input Projection
```python
# 512 → 512 with LayerNorm + ReLU + Dropout
x = self.input_projection(x)  # [batch_size, 512]
```

### 2. Deep Residual Processing (35 Layers)
```python
# 35 residual blocks with skip connections
for i in range(35):
    x = layer_norm(x)
    x = residual_block(x)  # x + residual_connection
    # Gradient checkpointing every 5 layers
```

### 3. Feature Extraction (Before Final Classification)
```python
# Extract features before final classification layer
features = self.output_projection[0:4](x)  # [batch_size, 256]
# This is where Class Anchor Clustering and Uncertainty Estimation happen
```

### 4. Class Anchor Clustering
```python
# Compute similarities to learnable class anchors
anchor_similarities, anchor_loss = self.anchor_clustering(features)
```

### 5. Final Classification
```python
# Final linear layer: 256 → 4 (emotion classes)
logits = self.output_projection[4](features)  # [batch_size, 4]
```

### 6. Uncertainty Estimation
```python
# Estimate prediction uncertainty
uncertainty = self.uncertainty_head(features)  # [batch_size, 1]
```

### 7. OpenMax Calibration (During Inference)
```python
# Apply Weibull-based uncertainty calibration
if use_openmax and not training:
    logits = self.openmax_forward(features, logits)
```

---

## 📤 OUTPUT

### Output Format:
```python
# Option 1: Basic output (logits only)
logits = classifier(x, use_openmax=True, return_uncertainty=False)
# Returns: torch.Tensor([batch_size, 4])

# Option 2: Full output (logits + uncertainty + anchor_loss)
logits, uncertainty, anchor_loss = classifier(x, use_openmax=True, return_uncertainty=True)
# Returns: 
# - logits: torch.Tensor([batch_size, 4])
# - uncertainty: torch.Tensor([batch_size, 1]) 
# - anchor_loss: torch.Tensor(scalar)
```

### Output Details:

#### 1. Logits (Classification Scores)
```python
logits = torch.Tensor([batch_size, 4])
# Shape: [batch_size, num_emotions]
# Values: Raw classification scores (not probabilities)
# Index mapping:
# 0: Neutral
# 1: Happy  
# 2: Sad
# 3: Angry
```

#### 2. Uncertainty (Confidence Scores)
```python
uncertainty = torch.Tensor([batch_size, 1])
# Shape: [batch_size, 1]
# Values: 0.0 (certain) to 1.0 (uncertain)
# Usage: confidence = 1 - uncertainty
```

#### 3. Anchor Loss (Clustering Loss)
```python
anchor_loss = torch.Tensor(scalar)
# Shape: scalar value
# Purpose: Loss for class anchor clustering
# Used during training for better class separation
```

---

## 🎯 USAGE EXAMPLES

### Training Mode:
```python
# During training
logits, uncertainty, anchor_loss = classifier(
    fused_features, 
    use_openmax=False,  # Don't use OpenMax during training
    return_uncertainty=True
)

# Get predictions
predictions = torch.argmax(logits, dim=1)  # [batch_size]
```

### Inference Mode:
```python
# During inference
logits, uncertainty, anchor_loss = classifier(
    fused_features,
    use_openmax=True,  # Use OpenMax for uncertainty calibration
    return_uncertainty=True
)

# Get predictions and confidence
predictions = torch.argmax(logits, dim=1)  # [batch_size]
confidence = 1 - uncertainty.squeeze()     # [batch_size]
probabilities = torch.softmax(logits, dim=1)  # [batch_size, 4]
```

### Simple Inference (Logits Only):
```python
# Just get classification logits
logits = classifier(fused_features, use_openmax=True, return_uncertainty=False)
predictions = torch.argmax(logits, dim=1)
```

---

## 🔧 INTEGRATION WITH FULL PIPELINE

### Complete Pipeline Flow:
```python
# 1. Audio Processing
audio_embeddings = audio_encoder(audio_list)  # [batch_size, seq_len, 768]

# 2. Text Processing  
text_embeddings = text_encoder(text_list)     # [batch_size, seq_len, 768]

# 3. Cross-Modal Attention
enhanced_audio, enhanced_text = cross_attention(audio_embeddings, text_embeddings)

# 4. Pooling
audio_vectors = pool_a(enhanced_audio)  # [batch_size, 1536]
text_vectors = pool_t(enhanced_text)    # [batch_size, 1536]

# 5. Fusion
fused_features = fusion(audio_vectors, text_vectors)  # [batch_size, 512]

# 6. Classification (YOUR INPUT GOES HERE)
logits, uncertainty, anchor_loss = classifier(fused_features)  # [batch_size, 4]

# 7. Get Results
predictions = torch.argmax(logits, dim=1)
confidence = 1 - uncertainty.squeeze()
```

---

## 📊 DIMENSION SUMMARY

| Component | Input Shape | Output Shape | Purpose |
|-----------|-------------|--------------|---------|
| **Input** | `[batch_size, 512]` | - | Fused audio+text features |
| **Deep Classifier** | `[batch_size, 512]` | `[batch_size, 256]` | Feature extraction |
| **Class Anchors** | `[batch_size, 256]` | `[batch_size, 4]` + scalar | Clustering |
| **Uncertainty** | `[batch_size, 256]` | `[batch_size, 1]` | Confidence estimation |
| **Final Output** | `[batch_size, 256]` | `[batch_size, 4]` | Emotion classification |

---

## 🎯 KEY POINTS

1. **Input**: 512-dim fused features from fusion layer
2. **Processing**: 35-layer deep network with residual connections
3. **Output**: Logits (4 emotions) + uncertainty + anchor loss
4. **OpenMax**: Applied during inference for uncertainty calibration
5. **Training**: Use `return_uncertainty=True` for full loss computation
6. **Inference**: Use `use_openmax=True` for calibrated predictions
