#!/usr/bin/env python3
"""
Create CREMA dataset manifest with 70-20-10 split
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

def extract_emotion_from_filename(filename):
    """Extract emotion from CREMA filename format: ActorID_Sentence_Emotion_Intensity.wav"""
    parts = filename.split('_')
    if len(parts) >= 3:
        emotion = parts[2].upper()
        # Map CREMA emotions to 6-class system (Anger, Disgust, Fear, Happy, Neutral, Sad)
        # Keep existing indices for the original four, and add distinct labels for DIS and FEA.
        # 0: Angry, 1: Happy, 2: Sad, 3: Neutral, 4: Disgust, 5: Fear
        emotion_map = {
            'ANG': 0,  # Angry
            'HAP': 1,  # Happy  
            'SAD': 2,  # Sad
            'NEU': 3,  # Neutral
            'DIS': 4,  # Disgust
            'FEA': 5,  # Fear
        }
        return emotion_map.get(emotion, 3)  # Default to neutral
    return 3  # Default to neutral

def create_crema_manifest():
    """Create CREMA manifest with balanced 70-20-10 split"""
    
    crema_dir = Path("datasets/crema")
    if not crema_dir.exists():
        print(f"❌ CREMA directory not found: {crema_dir}")
        return
    
    # Collect all audio files
    audio_files = list(crema_dir.glob("*.wav"))
    print(f"📁 Found {len(audio_files)} CREMA audio files")
    
    # Group by emotion for balanced splitting
    emotion_groups = defaultdict(list)
    
    for audio_file in audio_files:
        emotion = extract_emotion_from_filename(audio_file.name)
        relative_path = str(audio_file.relative_to(Path("datasets")))
        emotion_groups[emotion].append({
            'audio': relative_path,
            'text': f"Audio sample from crema dataset with emotion {emotion}",
            'label': emotion,
            'dataset': 'crema'
        })
    
    # Print emotion distribution
    print("📊 Emotion distribution:")
    for emotion, files in emotion_groups.items():
        print(f"  Emotion {emotion}: {len(files)} files")
    
    # Create balanced splits
    train_data = []
    val_data = []
    test_data = []
    
    for emotion, files in emotion_groups.items():
        random.shuffle(files)
        n = len(files)
        
        # 70-20-10 split
        train_end = int(0.7 * n)
        val_end = int(0.9 * n)
        
        train_data.extend(files[:train_end])
        val_data.extend(files[train_end:val_end])
        test_data.extend(files[val_end:])
    
    # Shuffle the final datasets
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"📊 Final split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Save manifests
    with open('crema_train_70.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open('crema_val_20.jsonl', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    with open('crema_test_10.jsonl', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print("✅ CREMA manifests created:")
    print("  - crema_train_70.jsonl")
    print("  - crema_val_20.jsonl") 
    print("  - crema_test_10.jsonl")

if __name__ == "__main__":
    random.seed(42)  # For reproducible splits
    create_crema_manifest()
