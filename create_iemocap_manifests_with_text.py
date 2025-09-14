#!/usr/bin/env python3
"""
Create IEMOCAP manifests with text transcriptions and emotion labels included.
This script parses the dialog/transcriptions and EmoEvaluation files to extract text and emotions.
"""

import os
import json
import glob
from pathlib import Path

def parse_transcription_file(transcription_path):
    """Parse a transcription file and return a dict of utterance_id -> text."""
    transcriptions = {}
    
    try:
        with open(transcription_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
                # Extract utterance ID and text
                if ']: ' in line:
                    parts = line.split(']: ', 1)
                    if len(parts) == 2:
                        utterance_id = parts[0].split(' [')[0]  # Remove timestamp
                        text = parts[1]
                        transcriptions[utterance_id] = text
    except Exception as e:
        print(f"Error parsing {transcription_path}: {e}")
    
    return transcriptions

def parse_emotion_file(emotion_path):
    """Parse an emotion annotation file and return a dict of utterance_id -> emotion."""
    emotions = {}
    
    try:
        with open(emotion_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                # Look for lines that start with [ (timestamp lines)
                if line.startswith('['):
                    # Format: [6.2901 - 8.2357]       Ses01F_impro01_F000     neu     [2.5000, 2.5000, 2.5000]
                    # Split by multiple spaces to handle the formatting
                    parts = line.split()
                    if len(parts) >= 4:
                        # parts[0] = "[6.2901", parts[1] = "-", parts[2] = "8.2357]", parts[3] = "Ses01F_impro01_F000", parts[4] = "neu"
                        utterance_id = parts[3]
                        emotion = parts[4]
                        emotions[utterance_id] = emotion
    except Exception as e:
        print(f"Error parsing {emotion_path}: {e}")
    
    return emotions

def create_manifest_with_text():
    """Create manifest files with text transcriptions and emotion labels included."""
    
    # IEMOCAP label mapping
    label_map = {
        'ang': 0, 'hap': 1, 'exc': 1, 'neu': 2, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'dis': 7, 'oth': 8, 'xxx': 9
    }
    
    # Base paths
    iemocap_base = "datasets/IEMOCAP"
    sessions = ["Session1", "Session2", "Session3", "Session4", "Session5"]
    
    train_items = []
    val_items = []
    
    for session in sessions:
        print(f"Processing {session}...")
        
        # Get all wav files in this session
        wav_pattern = os.path.join(iemocap_base, session, "sentences", "wav", "**", "*.wav")
        wav_files = glob.glob(wav_pattern, recursive=True)
        
        # Parse all transcription files for this session
        transcription_dir = os.path.join(iemocap_base, session, "dialog", "transcriptions")
        transcriptions = {}
        
        if os.path.exists(transcription_dir):
            for transcription_file in os.listdir(transcription_dir):
                if transcription_file.endswith('.txt'):
                    transcription_path = os.path.join(transcription_dir, transcription_file)
                    file_transcriptions = parse_transcription_file(transcription_path)
                    transcriptions.update(file_transcriptions)
        
        # Parse all emotion annotation files for this session
        emotion_dir = os.path.join(iemocap_base, session, "dialog", "EmoEvaluation")
        emotions = {}
        
        if os.path.exists(emotion_dir):
            for emotion_file in os.listdir(emotion_dir):
                if emotion_file.endswith('.txt'):
                    emotion_path = os.path.join(emotion_dir, emotion_file)
                    file_emotions = parse_emotion_file(emotion_path)
                    emotions.update(file_emotions)
        
        print(f"  Found {len(wav_files)} wav files")
        print(f"  Found {len(transcriptions)} transcriptions")
        print(f"  Found {len(emotions)} emotion annotations")
        
        for wav_path in wav_files:
            # Extract utterance ID from filename
            filename = os.path.basename(wav_path)
            utterance_id = os.path.splitext(filename)[0]
            
            # Get text transcription
            text = transcriptions.get(utterance_id, "")
            
            # Get emotion label from emotion annotations
            emotion = emotions.get(utterance_id, "neu")  # Default to neutral
            label = label_map.get(emotion, 2)  # Default to neutral (class 2)
            
            # Create manifest entry
            item = {
                "audio_path": wav_path,
                "text": text,
                "label": label
            }
            
            # Split into train/val (use Session5 for validation)
            if session == "Session5":
                val_items.append(item)
            else:
                train_items.append(item)
    
    # Save manifests
    print(f"\nSaving manifests...")
    print(f"Train items: {len(train_items)}")
    print(f"Val items: {len(val_items)}")
    
    # Save training manifest
    with open("train_manifest.jsonl", "w") as f:
        for item in train_items:
            f.write(json.dumps(item) + "\n")
    
    # Save validation manifest
    with open("val_manifest.jsonl", "w") as f:
        for item in val_items:
            f.write(json.dumps(item) + "\n")
    
    print("Manifests created successfully!")
    
    # Print some statistics
    train_text_count = sum(1 for item in train_items if item["text"].strip())
    val_text_count = sum(1 for item in val_items if item["text"].strip())
    
    print(f"\nText statistics:")
    print(f"Train items with text: {train_text_count}/{len(train_items)} ({100*train_text_count/len(train_items):.1f}%)")
    print(f"Val items with text: {val_text_count}/{len(val_items)} ({100*val_text_count/len(val_items):.1f}%)")
    
    # Print label distribution
    from collections import Counter
    train_labels = [item["label"] for item in train_items]
    val_labels = [item["label"] for item in val_items]
    
    print(f"\nLabel distribution:")
    print(f"Train: {dict(Counter(train_labels))}")
    print(f"Val: {dict(Counter(val_labels))}")

if __name__ == "__main__":
    create_manifest_with_text()