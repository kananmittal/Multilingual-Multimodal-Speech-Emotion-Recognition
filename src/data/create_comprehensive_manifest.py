#!/usr/bin/env python3
"""
Create comprehensive manifest from TESS, CREMA, and RAVDESS datasets.
This script will scan all audio files and create a unified manifest for training.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Emotion mapping for each dataset
EMOTION_MAPPING = {
    'tess': {
        'angry': 0,
        'disgust': 0,  # Map to angry
        'fear': 0,     # Map to angry
        'happy': 1,
        'pleasant_surprised': 1,  # Map to happy
        'sad': 2,
        'neutral': 3
    },
    'crema': {
        'ANG': 0,  # Angry
        'DIS': 0,  # Disgust (map to angry)
        'FEA': 0,  # Fear (map to angry)
        'HAP': 1,  # Happy
        'SAD': 2,  # Sad
        'NEU': 3   # Neutral
    },
    'ravdess': {
        'ang': 0,  # Angry
        'dis': 0,  # Disgust (map to angry)
        'fea': 0,  # Fear (map to angry)
        'hap': 1,  # Happy
        'sur': 1,  # Surprise (map to happy)
        'sad': 2,  # Sad
        'neu': 3   # Neutral
    }
}

def get_emotion_from_filename(filepath: str, dataset: str) -> int:
    """Extract emotion label from filename based on dataset."""
    filename = os.path.basename(filepath).lower()
    
    if dataset == 'tess':
        # TESS format: OAF_back_angry.wav
        for emotion, label in EMOTION_MAPPING['tess'].items():
            if emotion in filename:
                return label
        return 3  # Default to neutral
    
    elif dataset == 'crema':
        # CREMA format: 1001_DFA_ANG_HI.wav
        for emotion, label in EMOTION_MAPPING['crema'].items():
            if emotion in filename:
                return label
        return 3  # Default to neutral
    
    elif dataset == 'ravdess':
        # RAVDESS format: 03-01-01-01-01-01-01.wav
        # Parse the emotion code from the filename
        parts = filename.replace('.wav', '').split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion_map = {
                '01': 3,  # neutral
                '02': 1,  # calm (map to happy)
                '03': 1,  # happy
                '04': 2,  # sad
                '05': 0,  # angry
                '06': 0,  # fearful (map to angry)
                '07': 0,  # disgust (map to angry)
                '08': 1   # surprised (map to happy)
            }
            return emotion_map.get(emotion_code, 3)
        return 3  # Default to neutral
    
    return 3  # Default to neutral

def create_manifest_entry(filepath: str, dataset: str) -> Dict[str, Any]:
    """Create a manifest entry for a single audio file."""
    # Get relative path from datasets directory
    rel_path = os.path.relpath(filepath, 'datasets')
    
    # Extract emotion label
    emotion_label = get_emotion_from_filename(filepath, dataset)
    
    # Create dummy text (in real scenario, this would come from ASR or transcriptions)
    # For now, we'll use placeholder text that can be processed by the text encoder
    dummy_text = f"Audio sample from {dataset} dataset with emotion {emotion_label}"
    
    return {
        "audio": rel_path,
        "text": dummy_text,
        "label": emotion_label,
        "dataset": dataset
    }

def scan_dataset(dataset_path: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Scan a dataset directory and create manifest entries."""
    manifest_entries = []
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return manifest_entries
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                filepath = os.path.join(root, file)
                entry = create_manifest_entry(filepath, dataset_name)
                manifest_entries.append(entry)
    
    print(f"Found {len(manifest_entries)} audio files in {dataset_name}")
    return manifest_entries

def create_comprehensive_manifest(output_path: str = "comprehensive_manifest.jsonl"):
    """Create comprehensive manifest from all three datasets."""
    print("Creating comprehensive manifest from TESS, CREMA, and RAVDESS datasets...")
    
    all_entries = []
    
    # Scan each dataset
    datasets = [
        ('datasets/tess', 'tess'),
        ('datasets/crema', 'crema'),
        ('datasets/ravdess', 'ravdess')
    ]
    
    for dataset_path, dataset_name in datasets:
        entries = scan_dataset(dataset_path, dataset_name)
        all_entries.extend(entries)
    
    # Shuffle the entries for random distribution
    random.shuffle(all_entries)
    
    # Save comprehensive manifest
    with open(output_path, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created comprehensive manifest with {len(all_entries)} total samples")
    print(f"Saved to: {output_path}")
    
    # Print dataset distribution
    dataset_counts = {}
    emotion_counts = {}
    
    for entry in all_entries:
        dataset = entry['dataset']
        emotion = entry['label']
        
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("\nDataset distribution:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} samples")
    
    print("\nEmotion distribution:")
    emotion_names = ['angry', 'happy', 'sad', 'neutral']
    for emotion_id, count in emotion_counts.items():
        print(f"  {emotion_names[emotion_id]}: {count} samples")
    
    return all_entries

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create comprehensive manifest
    entries = create_comprehensive_manifest()
    
    print(f"\nComprehensive manifest created successfully!")
    print(f"Total samples: {len(entries)}")
    print(f"Ready for 70-20-10 split!")
