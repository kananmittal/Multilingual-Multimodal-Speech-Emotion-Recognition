import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.classifier import OpenMaxClassifier, AdvancedOpenMaxClassifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.prototypes import PrototypeMemory
from data.dataset import SERDataset
from utils import weighted_f1, energy_score
from data.preprocess import speed_perturb, add_noise_snr
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from tqdm import tqdm


def collate_fn(batch):
    audios, texts, labels = zip(*batch)
    return list(audios), list(texts), torch.tensor(labels, dtype=torch.long)


def test_time_augmentation(audio_list, num_augs=5):
    """Apply test-time augmentation to audio"""
    augmented_audios = []
    
    for audio in audio_list:
        augs = [audio]  # Original audio
        
        # Speed perturbations
        for factor in [0.95, 1.05]:
            augs.append(speed_perturb(audio, factor))
        
        # Light noise additions
        for snr in [15, 20]:
            augs.append(add_noise_snr(audio, snr))
        
        # Take first num_augs
        augmented_audios.append(augs[:num_augs])
    
    return augmented_audios


def temperature_scaling(logits, temperature=1.0):
    """Apply temperature scaling to logits"""
    return logits / temperature


def find_optimal_temperature(val_logits, val_labels, device):
    """Find optimal temperature for calibration"""
    temperatures = torch.logspace(-1, 2, 100, device=device)
    best_temp = 1.0
    best_ece = float('inf')
    
    for temp in temperatures:
        scaled_logits = temperature_scaling(val_logits, temp)
        probs = torch.softmax(scaled_logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        # Simple ECE approximation
        ece = torch.mean(torch.abs(max_probs - (preds == val_labels).float()))
        
        if ece < best_ece:
            best_ece = ece
            best_temp = temp.item()
    
    return best_temp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--num_tta', type=int, default=5, help='Number of TTA augmentations')
    parser.add_argument('--calibrate', action='store_true', help='Use temperature scaling')
    parser.add_argument('--val_manifest', type=str, help='Validation manifest for temperature calibration')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
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
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
    except:
        # Try with weights_only=False for older checkpoints
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    audio_encoder.load_state_dict(ckpt['audio_encoder'])
    text_encoder.load_state_dict(ckpt['text_encoder'])
    cross.load_state_dict(ckpt['cross'])
    pool_a.load_state_dict(ckpt['pool_a'])
    pool_t.load_state_dict(ckpt['pool_t'])
    fusion.load_state_dict(ckpt['fusion'])
    classifier.load_state_dict(ckpt['classifier'])
    prototypes.load_state_dict(ckpt['prototypes'])

    # Set to eval mode
    audio_encoder.eval()
    text_encoder.eval()
    cross.eval()
    pool_a.eval()
    pool_t.eval()
    fusion.eval()
    classifier.eval()

    # Temperature calibration
    optimal_temp = 1.0
    if args.calibrate and args.val_manifest:
        print("Calibrating temperature...")
        val_ds = SERDataset(args.val_manifest)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        val_logits = []
        val_labels = []
        
        with torch.no_grad():
            for audio_list, text_list, labels in tqdm(val_loader, desc="Calibrating"):
                labels = labels.to(device)
                a_seq, a_mask = audio_encoder(audio_list)
                t_seq, t_mask = text_encoder(text_list)
                a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                a_vec = pool_a(a_enh, a_mask)
                t_vec = pool_t(t_enh, t_mask)
                fused = fusion(a_vec, t_vec)
                logits, uncertainty, anchor_loss = classifier(fused, use_openmax=False, return_uncertainty=True)
                
                val_logits.append(logits)
                val_labels.append(labels)
        
        val_logits = torch.cat(val_logits, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        optimal_temp = find_optimal_temperature(val_logits, val_labels, device)
        print(f"Optimal temperature: {optimal_temp:.3f}")

    # Evaluation
    print("Evaluating...")
    ds = SERDataset(args.manifest)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    all_preds, all_labels, all_energies, all_probs = [], [], [], []
    
    with torch.no_grad():
        for audio_list, text_list, labels in tqdm(loader, desc="Evaluating"):
            labels = labels.to(device)
            
            if args.use_tta:
                # Apply test-time augmentation
                aug_audio_lists = test_time_augmentation(audio_list, args.num_tta)
                all_logits = []
                
                for aug_audios in aug_audio_lists:
                    a_seq, a_mask = audio_encoder(aug_audios)
                    t_seq, t_mask = text_encoder(text_list)
                    a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                    a_vec = pool_a(a_enh, a_mask)
                    t_vec = pool_t(t_enh, t_mask)
                    fused = fusion(a_vec, t_vec)
                    logits = classifier(fused)
                    all_logits.append(logits)
                
                # Average logits across augmentations
                logits = torch.stack(all_logits).mean(dim=0)
            else:
                a_seq, a_mask = audio_encoder(audio_list)
                t_seq, t_mask = text_encoder(text_list)
                a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                a_vec = pool_a(a_enh, a_mask)
                t_vec = pool_t(t_enh, t_mask)
                fused = fusion(a_vec, t_vec)
                logits, uncertainty, anchor_loss = classifier(fused, use_openmax=True, return_uncertainty=True)
            
            # Apply temperature scaling (optional, OpenMax already provides calibration)
            if args.calibrate:
                logits = temperature_scaling(logits, optimal_temp)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            energies = energy_score(logits)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_energies.extend(energies.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_energies = np.array(all_energies)
    all_probs = np.array(all_probs)

    # Calculate metrics
    f1_weighted = weighted_f1(torch.tensor(all_preds), torch.tensor(all_labels))
    
    # Classification report
    emotion_names = ['neutral', 'happy', 'sad', 'angry']
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Energy Score - Mean: {all_energies.mean():.3f}, Std: {all_energies.std():.3f}")
    print(f"Temperature: {optimal_temp:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, emotion in enumerate(emotion_names):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            print(f"  {emotion}: {class_acc:.3f} ({class_mask.sum()} samples)")
    
    # Confidence analysis
    max_probs = all_probs.max(axis=1)
    print(f"\nConfidence Analysis:")
    print(f"  Mean confidence: {max_probs.mean():.3f}")
    print(f"  Std confidence: {max_probs.std():.3f}")
    print(f"  High confidence (>0.8): {(max_probs > 0.8).mean():.3f}")
    print(f"  Low confidence (<0.5): {(max_probs < 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
