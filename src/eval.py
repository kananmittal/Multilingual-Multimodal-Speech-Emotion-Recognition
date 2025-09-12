import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
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
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','mps','cuda'])
    parser.add_argument('--fusion_mode', type=str, default='gate', choices=['gate','concat'])
    parser.add_argument('--save_preds', type=str, default='', help='Path to save per-sample predictions as JSONL')
    parser.add_argument('--save_probs', type=str, default='', help='Path to save probability matrix as .npy')
    args = parser.parse_args()

    if args.device != 'auto':
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    audio_encoder = AudioEncoder(freeze_base=True).to(device)
    text_encoder = TextEncoder(freeze_base=True).to(device)
    audio_hid = audio_encoder.encoder.config.hidden_size
    text_hid = text_encoder.encoder.config.hidden_size
    cross = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8).to(device)
    pool_a = AttentiveStatsPooling(audio_hid).to(device)
    pool_t = AttentiveStatsPooling(text_hid).to(device)
    fusion = FusionLayer(audio_hid * 2, text_hid * 2, 1024).to(device)
    # Infer num_labels from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    # Try to infer classifier in/out dims
    cls_sd = ckpt.get('classifier', {})
    # Find last linear weight
    num_labels = None
    for k, v in cls_sd.items():
        if k.endswith('weight') and v.dim() == 2:
            num_labels = v.size(0)
    if num_labels is None:
        num_labels = 6
    if args.fusion_mode == 'concat':
        classifier_in = audio_hid * 2 + text_hid * 2
    else:
        classifier_in = 1024
    classifier = Classifier(classifier_in, num_labels=num_labels).to(device)

    audio_encoder.load_state_dict(ckpt['audio_encoder'])
    text_encoder.load_state_dict(ckpt['text_encoder'])
    cross.load_state_dict(ckpt['cross'])
    pool_a.load_state_dict(ckpt['pool_a'])
    pool_t.load_state_dict(ckpt['pool_t'])
    fusion.load_state_dict(ckpt['fusion'])
    classifier.load_state_dict(ckpt['classifier'])

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
                fused = torch.cat([a_vec, t_vec], dim=-1) if args.fusion_mode == 'concat' else fusion(a_vec, t_vec)
                logits = classifier(fused)
                
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
    saved_samples = []
    
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
                fused = torch.cat([a_vec, t_vec], dim=-1) if args.fusion_mode == 'concat' else fusion(a_vec, t_vec)
                logits = classifier(fused)
            
            # Apply temperature scaling
            if args.calibrate:
                logits = temperature_scaling(logits, optimal_temp)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            energies = energy_score(logits)
            
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()
            energies_np = energies.cpu().numpy()
            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            all_energies.extend(energies_np)
            all_probs.extend(probs_np)
            if args.save_preds:
                # Save lightweight per-sample info
                for i in range(len(preds_np)):
                    saved_samples.append({
                        'pred': int(preds_np[i]),
                        'label': int(labels_np[i]),
                        'energy': float(energies_np[i])
                    })

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_energies = np.array(all_energies)
    all_probs = np.array(all_probs)

    # Calculate metrics
    f1_weighted = weighted_f1(torch.tensor(all_preds), torch.tensor(all_labels))
    
    # Classification report (names optional if not 4 classes)
    emotion_names = None
    if len(set(all_labels)) in (4, 6, 8, 11):
        # Optionally map common sizes to placeholder names
        if len(set(all_labels)) == 4:
            emotion_names = ['neutral', 'happy', 'sad', 'angry']
        elif len(set(all_labels)) == 6:
            emotion_names = [f'class_{i}' for i in range(6)]
        elif len(set(all_labels)) == 8:
            emotion_names = [f'class_{i}' for i in range(8)]
        elif len(set(all_labels)) == 11:
            emotion_names = [f'class_{i}' for i in range(11)]
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Energy Score - Mean: {all_energies.mean():.3f}, Std: {all_energies.std():.3f}")
    print(f"Temperature: {optimal_temp:.3f}")
    
    print("\nClassification Report:")
    if emotion_names is not None:
        print(classification_report(all_labels, all_preds, target_names=emotion_names))
    else:
        print(classification_report(all_labels, all_preds))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(num_labels):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            name = emotion_names[i] if emotion_names and i < len(emotion_names) else f'class_{i}'
            print(f"  {name}: {class_acc:.3f} ({class_mask.sum()} samples)")
    
    # Confidence analysis
    max_probs = all_probs.max(axis=1)
    print(f"\nConfidence Analysis:")
    print(f"  Mean confidence: {max_probs.mean():.3f}")
    print(f"  Std confidence: {max_probs.std():.3f}")
    print(f"  High confidence (>0.8): {(max_probs > 0.8).mean():.3f}")
    print(f"  Low confidence (<0.5): {(max_probs < 0.5).mean():.3f}")

    # Optional saves
    if args.save_probs:
        np.save(args.save_probs, all_probs)
        print(f"Saved probabilities to {args.save_probs}")
    if args.save_preds:
        import json
        with open(args.save_preds, 'w') as f:
            for row in saved_samples:
                f.write(json.dumps(row) + "\n")
        print(f"Saved predictions to {args.save_preds}")


if __name__ == "__main__":
    main()
