#!/usr/bin/env python3
"""
Test the trained model with frozen weights for consistent evaluation.
This script loads the best checkpoint, freezes all weights, and evaluates on test data.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.classifier import AdvancedOpenMaxClassifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.prototypes import PrototypeMemory
from data.dataset import SERDataset
from utils import weighted_f1, energy_score
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from tqdm import tqdm

def collate_fn(batch):
    audios, texts, labels = zip(*batch)
    return list(audios), list(texts), torch.tensor(labels, dtype=torch.long)

def freeze_model_weights(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def load_and_freeze_model(checkpoint_path, device):
    """Load model from checkpoint and freeze all weights."""
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize models
    audio_encoder = AudioEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    audio_hid = audio_encoder.encoder.config.hidden_size
    text_hid = text_encoder.encoder.config.hidden_size
    cross = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8).to(device)
    pool_a = AttentiveStatsPooling(audio_hid).to(device)
    pool_t = AttentiveStatsPooling(text_hid).to(device)
    fusion = FusionLayer(audio_hid * 2, text_hid * 2, 512).to(device)
    classifier = AdvancedOpenMaxClassifier(
        input_dim=512, 
        num_labels=4, 
        num_layers=35, 
        base_dim=512, 
        dropout=0.15
    ).to(device)
    prototypes = PrototypeMemory(4, 512).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dicts
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    cross.load_state_dict(checkpoint['cross'])
    pool_a.load_state_dict(checkpoint['pool_a'])
    pool_t.load_state_dict(checkpoint['pool_t'])
    fusion.load_state_dict(checkpoint['fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    prototypes.load_state_dict(checkpoint['prototypes'])
    
    # Freeze all models
    print("Freezing all model weights...")
    audio_encoder = freeze_model_weights(audio_encoder)
    text_encoder = freeze_model_weights(text_encoder)
    cross = freeze_model_weights(cross)
    pool_a = freeze_model_weights(pool_a)
    pool_t = freeze_model_weights(pool_t)
    fusion = freeze_model_weights(fusion)
    classifier = freeze_model_weights(classifier)
    prototypes = freeze_model_weights(prototypes)
    
    print("✅ All model weights frozen successfully!")
    
    return audio_encoder, text_encoder, cross, pool_a, pool_t, fusion, classifier, prototypes

def evaluate_model(audio_encoder, text_encoder, cross, pool_a, pool_t, fusion, classifier, 
                  test_loader, device):
    """Evaluate the frozen model on test data."""
    print("Starting evaluation with frozen weights...")
    
    all_preds, all_labels, all_energies, all_probs = [], [], [], []
    
    with torch.no_grad():
        for audio_list, text_list, labels in tqdm(test_loader, desc="Evaluating"):
            labels = labels.to(device)
            
            # Forward pass through frozen models
            a_seq, a_mask = audio_encoder(audio_list, text_list)
            t_seq, t_mask = text_encoder(text_list)
            a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
            a_vec = pool_a(a_enh, a_mask)
            t_vec = pool_t(t_enh, t_mask)
            fused = fusion(a_vec, t_vec)
            
            # Get predictions
            logits, uncertainty, anchor_loss = classifier(fused, use_openmax=True, return_uncertainty=True)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            energies = energy_score(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_energies.extend(energies.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_energies), np.array(all_probs)

def print_evaluation_results(preds, labels, energies, probs):
    """Print comprehensive evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS (FROZEN MODEL)")
    print("="*60)
    
    # Overall metrics
    f1_weighted = weighted_f1(torch.tensor(preds), torch.tensor(labels))
    accuracy = (preds == labels).mean()
    
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Energy analysis
    print(f"\nEnergy Score Analysis:")
    print(f"  Mean: {energies.mean():.3f}")
    print(f"  Std: {energies.std():.3f}")
    print(f"  Min: {energies.min():.3f}")
    print(f"  Max: {energies.max():.3f}")
    
    # Confidence analysis
    max_probs = probs.max(axis=1)
    print(f"\nConfidence Analysis:")
    print(f"  Mean confidence: {max_probs.mean():.3f}")
    print(f"  Std confidence: {max_probs.std():.3f}")
    print(f"  High confidence (>0.8): {(max_probs > 0.8).mean():.3f}")
    print(f"  Low confidence (<0.5): {(max_probs < 0.5).mean():.3f}")
    
    # Per-class metrics
    emotion_names = ['angry', 'happy', 'sad', 'neutral']
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=emotion_names))
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i, emotion in enumerate(emotion_names):
        class_mask = labels == i
        if class_mask.sum() > 0:
            class_acc = (preds[class_mask] == labels[class_mask]).mean()
            print(f"  {emotion}: {class_acc:.3f} ({class_mask.sum()} samples)")
    
    return f1_weighted, accuracy

def main():
    parser = argparse.ArgumentParser(description="Test frozen model on test set")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/epoch_1_f1_0.4884.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--test_manifest', type=str, default='test_10.jsonl',
                       help='Path to test manifest file')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load and freeze model
    models = load_and_freeze_model(args.checkpoint, device)
    audio_encoder, text_encoder, cross, pool_a, pool_t, fusion, classifier, prototypes = models
    
    # Load test data
    print(f"Loading test data from {args.test_manifest}")
    test_ds = SERDataset(args.test_manifest)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Test set: {len(test_ds)} samples")
    
    # Evaluate
    preds, labels, energies, probs = evaluate_model(
        audio_encoder, text_encoder, cross, pool_a, pool_t, fusion, classifier, test_loader, device
    )
    
    # Print results
    f1_score, accuracy = print_evaluation_results(preds, labels, energies, probs)
    
    print(f"\n🎯 Final Results:")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Model: {args.checkpoint}")
    print(f"  Weights: FROZEN ✅")

if __name__ == "__main__":
    main()
