#!/usr/bin/env python3
"""
Split a comprehensive manifest file into train, validation, and test sets.
Supports 70-20-10 split with balanced sampling across datasets and emotions.
"""

import json
import random
import os
import argparse
from typing import List, Dict, Any
from collections import defaultdict

def create_balanced_split(
    manifest_entries: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1
):
    """
    Create balanced split ensuring each dataset and emotion class is represented proportionally.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Group entries by dataset and emotion
    grouped_entries = defaultdict(lambda: defaultdict(list))
    for entry in manifest_entries:
        dataset = entry['dataset']
        emotion = entry['label']
        grouped_entries[dataset][emotion].append(entry)
    
    train_entries = []
    val_entries = []
    test_entries = []
    
    # Split each group proportionally
    for dataset in grouped_entries:
        for emotion in grouped_entries[dataset]:
            entries = grouped_entries[dataset][emotion]
            random.shuffle(entries)  # Shuffle within each group
            
            total_count = len(entries)
            train_count = int(total_count * train_ratio)
            val_count = int(total_count * val_ratio)
            test_count = total_count - train_count - val_count
            
            # Split the entries
            train_entries.extend(entries[:train_count])
            val_entries.extend(entries[train_count:train_count + val_count])
            test_entries.extend(entries[train_count + val_count:])
    
    # Shuffle the final splits
    random.shuffle(train_entries)
    random.shuffle(val_entries)
    random.shuffle(test_entries)
    
    return train_entries, val_entries, test_entries

def save_manifest(entries: List[Dict[str, Any]], output_path: str):
    """Save manifest entries to a JSONL file."""
    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

def print_split_statistics(train_entries, val_entries, test_entries):
    """Print detailed statistics about the split."""
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    # Overall counts
    print(f"Total samples: {len(train_entries) + len(val_entries) + len(test_entries)}")
    print(f"Train: {len(train_entries)} ({len(train_entries)/(len(train_entries) + len(val_entries) + len(test_entries))*100:.1f}%)")
    print(f"Validation: {len(val_entries)} ({len(val_entries)/(len(train_entries) + len(val_entries) + len(test_entries))*100:.1f}%)")
    print(f"Test: {len(test_entries)} ({len(test_entries)/(len(train_entries) + len(val_entries) + len(test_entries))*100:.1f}%)")
    
    # Dataset distribution
    for split_name, split_entries in [("Train", train_entries), ("Validation", val_entries), ("Test", test_entries)]:
        print(f"\n{split_name} Split - Dataset Distribution:")
        dataset_counts = defaultdict(int)
        for entry in split_entries:
            dataset_counts[entry['dataset']] += 1
        
        for dataset, count in sorted(dataset_counts.items()):
            percentage = count / len(split_entries) * 100
            print(f"  {dataset}: {count} samples ({percentage:.1f}%)")
    
    # Emotion distribution
    for split_name, split_entries in [("Train", train_entries), ("Validation", val_entries), ("Test", test_entries)]:
        print(f"\n{split_name} Split - Emotion Distribution:")
        emotion_counts = defaultdict(int)
        for entry in split_entries:
            emotion_counts[entry['label']] += 1
        
        emotion_names = ['angry', 'happy', 'sad', 'neutral']
        for emotion_id in range(4):
            count = emotion_counts[emotion_id]
            percentage = count / len(split_entries) * 100
            print(f"  {emotion_names[emotion_id]}: {count} samples ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Split manifest into train/val/test sets")
    parser.add_argument('--input', type=str, default='comprehensive_manifest.jsonl',
                       help='Input comprehensive manifest file')
    parser.add_argument('--train_out', type=str, default='train_70.jsonl',
                       help='Output training manifest file')
    parser.add_argument('--val_out', type=str, default='val_20.jsonl',
                       help='Output validation manifest file')
    parser.add_argument('--test_out', type=str, default='test_10.jsonl',
                       help='Output test manifest file')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        print("Please run the comprehensive manifest creation script first:")
        print("python src/data/create_comprehensive_manifest.py")
        return
    
    # Load manifest entries
    print(f"Loading manifest from {args.input}...")
    with open(args.input, 'r') as f:
        manifest_entries = [json.loads(line.strip()) for line in f]
    
    print(f"Loaded {len(manifest_entries)} manifest entries")
    
    # Create balanced split
    print("Creating balanced split...")
    train_entries, val_entries, test_entries = create_balanced_split(
        manifest_entries, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    # Save split manifests
    print("Saving split manifests...")
    save_manifest(train_entries, args.train_out)
    save_manifest(val_entries, args.val_out)
    save_manifest(test_entries, args.test_out)
    
    # Print statistics
    print_split_statistics(train_entries, val_entries, test_entries)
    
    print(f"\nSplit manifests saved successfully!")
    print(f"Train: {args.train_out}")
    print(f"Validation: {args.val_out}")
    print(f"Test: {args.test_out}")

if __name__ == "__main__":
    main()


