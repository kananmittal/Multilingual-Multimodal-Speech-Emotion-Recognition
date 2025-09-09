#!/usr/bin/env python3
"""
M3 Mac Optimized CREMA Training Script — STABILIZED
- Fixes NaN loss on MPS by lowering LR, disabling GradScaler, normalizing embeddings for prototype loss,
  softening focal loss, and adding runtime NaN guards.
- Also fixes a small import indentation issue and makes augmentation opt‑in by default.
"""

import os
import math
import time
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# Ensure local packages are importable
import sys
sys.path.append('src')

from models import AudioEncoder, TextEncoder, FusionLayer
from models.classifier import AdvancedOpenMaxClassifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.losses import LabelSmoothingCrossEntropy, ClassBalancedFocalLoss, SupConLoss
from models.prototypes import PrototypeMemory
from data.dataset import SERDataset
from data.preprocess import speed_perturb  # keep noise out for now
from utils import weighted_f1

NUM_LABELS = 6  # Angry, Happy, Sad, Neutral, Disgust, Fear


def collate_fn(batch):
    audios, texts, labels = zip(*batch)
    return list(audios), list(texts), torch.tensor(labels, dtype=torch.long)


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warmup and restarts - optimized for M3 Mac"""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, warmup_epochs=0):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Guard divide-by-zero when warmup_epochs == 0
            factor = 0.0 if self.warmup_epochs == 0 else self.last_epoch / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]

        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult

        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]


def get_optimal_device():
    """Get the best available device - optimized for M3 Mac"""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.mps.manual_seed(seed)  # available on recent PyTorch builds
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="M3 Mac Optimized CREMA Training — Stabilized")
    parser.add_argument('--train_manifest', type=str, default='crema_train_70.jsonl')
    parser.add_argument('--val_manifest', type=str, default='crema_val_20.jsonl')
    parser.add_argument('--test_manifest', type=str, default='crema_test_10.jsonl')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=20)
    # Higher base LR to avoid numerical instability
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    # AMP disabled on MPS; also disable GradScaler (stability)
    parser.add_argument('--use_amp', action='store_true', default=False)
    # Make augmentation opt-in for stability during bring-up
    parser.add_argument('--augment', action='store_true', default=False)
    # Softer prototype regularization by default
    parser.add_argument('--proto_weight', type=float, default=0.01)
    parser.add_argument('--save_dir', type=str, default='checkpoints_crema_m3_stable')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    # Slightly looser clip (norm) to avoid aggressive truncation
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    # Reduce focal gamma to minimize loss spikiness
    parser.add_argument('--focal_gamma', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=2, help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Get optimal device
    device = get_optimal_device()
    print(f"🍎 Using device: {device}")
    print(f"📊 Training on CREMA dataset with M3 Mac stabilization")

    if device == 'mps':
        print("🚀 Using Apple M3 GPU acceleration (MPS)")
        print("⚡ MPS optimizations enabled")
    elif device == 'cuda':
        print("🚀 Using NVIDIA GPU acceleration (CUDA)")
    else:
        print("⚠️  Using CPU - consider GPU acceleration for faster training")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Save training configuration
    config = vars(args)
    config['device'] = device
    config['start_time'] = datetime.now().isoformat()
    with open(os.path.join(args.save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Load datasets
    print("📁 Loading datasets...")
    train_ds = SERDataset(args.train_manifest)
    val_ds = SERDataset(args.val_manifest)

    print(f"📁 Train samples: {len(train_ds)}")
    print(f"📁 Val samples: {len(val_ds)}")

    # Optimize DataLoader for M3 Mac MPS
    num_workers = 0  # MPS doesn't support multiprocessing
    pin_memory = False  # MPS doesn't benefit from pinned memory

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )

    # Initialize models
    print("🔧 Initializing models...")
    audio_encoder = AudioEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    audio_hid = audio_encoder.encoder.config.hidden_size
    text_hid = text_encoder.encoder.config.hidden_size
    cross = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8, dropout=0.15).to(device)
    pool_a = AttentiveStatsPooling(audio_hid).to(device)
    pool_t = AttentiveStatsPooling(text_hid).to(device)
    fusion = FusionLayer(audio_hid * 2, text_hid * 2, 512).to(device)
    classifier = AdvancedOpenMaxClassifier(
        input_dim=512,
        num_labels=NUM_LABELS,
        num_layers=35,
        base_dim=512,
        dropout=0.2
    ).to(device)
    prototypes = PrototypeMemory(NUM_LABELS, 512).to(device)

    # Optimizer with more reasonable learning rates
    print("⚙️  Setting up optimizer...")
    base_lr = args.lr
    optimizer = optim.AdamW([
        # Encoders: moderate LR
        {'params': audio_encoder.parameters(), 'lr': base_lr * 0.1, 'weight_decay': 0.01},
        {'params': text_encoder.parameters(), 'lr': base_lr * 0.1, 'weight_decay': 0.01},

        # Attention and pooling: moderate LR
        {'params': cross.parameters(), 'lr': base_lr * 0.5, 'weight_decay': 0.01},
        {'params': pool_a.parameters(), 'lr': base_lr * 0.5, 'weight_decay': 0.01},
        {'params': pool_t.parameters(), 'lr': base_lr * 0.5, 'weight_decay': 0.01},

        # Fusion: higher LR
        {'params': fusion.parameters(), 'lr': base_lr * 0.8, 'weight_decay': 0.01},

        # Classifier: standard LR
        {'params': classifier.deep_classifier.parameters(), 'lr': base_lr * 1.0, 'weight_decay': 0.01},
        {'params': classifier.anchor_clustering.parameters(), 'lr': base_lr * 1.0, 'weight_decay': 0.01},
        {'params': classifier.uncertainty_head.parameters(), 'lr': base_lr * 1.0, 'weight_decay': 0.01},

        # Prototypes: moderate LR
        {'params': prototypes.parameters(), 'lr': base_lr * 0.5, 'weight_decay': 0.01},
    ], eps=1e-8, betas=(0.9, 0.999))

    # Loss functions with better stability
    ce_smooth = LabelSmoothingCrossEntropy(args.label_smoothing)
    cb_focal = ClassBalancedFocalLoss(beta=0.99, gamma=args.focal_gamma, num_classes=NUM_LABELS)
    supcon = SupConLoss(temperature=0.1)

    # AMP off on MPS — disable scaler too for stability
    use_amp = False
    scaler = GradScaler(enabled=False)
    print("⚠️  Mixed Precision (AMP) disabled for MPS compatibility; GradScaler disabled")

    # Load checkpoint if resuming
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0
    training_history = []
    checkpoint = None

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"📂 Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)

        audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        cross.load_state_dict(checkpoint['cross'])
        pool_a.load_state_dict(checkpoint['pool_a'])
        pool_t.load_state_dict(checkpoint['pool_t'])
        fusion.load_state_dict(checkpoint['fusion'])
        classifier.load_state_dict(checkpoint['classifier'])
        prototypes.load_state_dict(checkpoint['prototypes'])

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'best_f1' in checkpoint:
            best_f1 = checkpoint['best_f1']
        if 'training_history' in checkpoint:
            training_history = checkpoint['training_history']

        start_epoch = checkpoint['epoch'] + 1
        print(f"🔄 Resuming from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Scheduler with higher minimum LR
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        T_0=len(train_loader) * 4,  # Restart every 4 epochs
        T_mult=1,
        eta_min=base_lr * 0.01,  # Higher minimum LR
        warmup_epochs=args.warmup_epochs
    )

    if checkpoint and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    print("🎯 M3 Mac Training Configuration (Stabilized):")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Focal gamma: {args.focal_gamma}")
    print(f"  Gradient clipping: {args.gradient_clip}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Save every: {args.save_every} epochs")

    # Training loop
    training_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\n🚀 Epoch {epoch+1}/{args.epochs}")

        # Training phase
        audio_encoder.train(); text_encoder.train(); cross.train()
        pool_a.train(); pool_t.train(); fusion.train(); classifier.train(); prototypes.train()

        train_loss = 0.0
        train_f1 = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (audio_list, text_list, labels) in enumerate(pbar):
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device if device != 'cpu' else 'cpu', enabled=use_amp):
                # Forward pass
                a_seq, a_mask = audio_encoder(audio_list, text_list)
                t_seq, t_mask = text_encoder(text_list)
                a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                a_vec = pool_a(a_enh, a_mask)
                t_vec = pool_t(t_enh, t_mask)
                fused_raw = fusion(a_vec, t_vec)

                # Classifier on raw fused (keep distribution)
                logits = classifier(fused_raw, use_openmax=True)

                # Losses
                ce_loss = ce_smooth(logits, labels)
                focal_loss = cb_focal(logits, labels)

                # Prototype loss on normalized embeddings for stability
                fused_norm = nn.functional.normalize(fused_raw, p=2, dim=-1)
                proto_loss = prototypes.prototype_loss(fused_norm, labels)

                # Combined loss with better stability
                total_loss = ce_loss + 0.1 * focal_loss + args.proto_weight * proto_loss
                
                # Additional NaN checks for individual losses
                if not torch.isfinite(ce_loss):
                    print(f"⚠️ CE loss is NaN: {ce_loss}")
                    ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
                if not torch.isfinite(focal_loss):
                    print(f"⚠️ Focal loss is NaN: {focal_loss}")
                    focal_loss = torch.tensor(0.0, device=device, requires_grad=True)
                if not torch.isfinite(proto_loss):
                    print(f"⚠️ Proto loss is NaN: {proto_loss}")
                    proto_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                total_loss = ce_loss + 0.1 * focal_loss + args.proto_weight * proto_loss

                # Optional augmentation (disabled by default)
                if args.augment and torch.rand(1).item() < 0.2:
                    aug_audio = [speed_perturb(audio, 0.95 + torch.rand(1).item() * 0.1) for audio in audio_list]
                    a_seq_aug, a_mask_aug = audio_encoder(aug_audio, text_list)
                    a_enh_aug, _ = cross(a_seq_aug, t_seq, a_mask_aug, t_mask)
                    a_vec_aug = pool_a(a_enh_aug, a_mask_aug)
                    fused_aug_raw = fusion(a_vec_aug, t_vec)
                    logits_aug = classifier(fused_aug_raw, use_openmax=True)
                    aug_loss = ce_smooth(logits_aug, labels)
                    total_loss = total_loss + 0.2 * aug_loss

            # NaN/Inf guard
            if not torch.isfinite(total_loss):
                print("⚠️ Non-finite loss detected. Skipping batch.")
                print({
                    'ce': float(ce_loss.detach().cpu()) if torch.isfinite(ce_loss) else 'nan',
                    'focal': float(focal_loss.detach().cpu()) if torch.isfinite(focal_loss) else 'nan',
                    'proto': float(proto_loss.detach().cpu()) if torch.isfinite(proto_loss) else 'nan',
                })
                continue

            # Backward + step
            total_loss.backward()
            nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params'] if p.requires_grad],
                                     args.gradient_clip)
            optimizer.step()
            scheduler.step()

            # Metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                f1 = weighted_f1(preds, labels)  # already a float
                train_loss += total_loss.item()
                train_f1 += f1
                num_batches += 1


            pbar.set_postfix({
                'Loss': f'{(train_loss/num_batches):.4f}',
                'F1': f'{(train_f1/num_batches):.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_train_loss = train_loss / max(1, num_batches)
        avg_train_f1 = train_f1 / max(1, num_batches)

        # Validation phase
        audio_encoder.eval(); text_encoder.eval(); cross.eval()
        pool_a.eval(); pool_t.eval(); fusion.eval(); classifier.eval(); prototypes.eval()

        val_loss = 0.0
        val_f1 = 0.0
        val_batches = 0

        with torch.no_grad():
            for audio_list, text_list, labels in tqdm(val_loader, desc="Validation"):
                labels = labels.to(device)

                with autocast(device_type=device if device != 'cpu' else 'cpu', enabled=use_amp):
                    a_seq, a_mask = audio_encoder(audio_list, text_list)
                    t_seq, t_mask = text_encoder(text_list)
                    a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                    a_vec = pool_a(a_enh, a_mask)
                    t_vec = pool_t(t_enh, t_mask)
                    fused_raw = fusion(a_vec, t_vec)

                    logits = classifier(fused_raw, use_openmax=True)
                    ce_loss = ce_smooth(logits, labels)
                    focal_loss = cb_focal(logits, labels)

                    fused_norm = nn.functional.normalize(fused_raw, p=2, dim=-1)
                    proto_loss = prototypes.prototype_loss(fused_norm, labels)

                    total = ce_loss + 0.2 * focal_loss + args.proto_weight * proto_loss

                preds = torch.argmax(logits, dim=1)
                f1 = weighted_f1(preds, labels)

                if torch.isfinite(total):
                    val_loss += float(total.detach().cpu())
                else:
                    print("⚠️ Non-finite val loss encountered; counting as zero for averaging")
                
                val_f1 += f1
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        avg_val_f1 = val_f1 / max(1, val_batches)

        epoch_time = time.time() - epoch_start_time

        # Store training history
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_f1': avg_train_f1,
            'val_loss': avg_val_loss,
            'val_f1': avg_val_f1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        training_history.append(epoch_stats)

        print(f"📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f}, Val   F1: {avg_val_f1:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Epoch Time: {epoch_time/60:.1f} minutes")

        # Save best model
        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'audio_encoder': audio_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'cross': cross.state_dict(),
                'pool_a': pool_a.state_dict(),
                'pool_t': pool_t.state_dict(),
                'fusion': fusion.state_dict(),
                'classifier': classifier.state_dict(),
                'prototypes': prototypes.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_f1': best_f1,
                'val_f1': avg_val_f1,
                'val_loss': avg_val_loss,
                'train_f1': avg_train_f1,
                'train_loss': avg_train_loss,
                'training_time': time.time() - training_start_time,
                'device': device,
                'config': vars(args),
                'training_history': training_history
            }

            path = os.path.join(args.save_dir, f'best_crema_m3_stable_f1_{avg_val_f1:.4f}.pt')
            torch.save(checkpoint, path)
            print(f"💾 Saved best model with F1: {avg_val_f1:.4f} -> {path}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{args.patience})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            if checkpoint is None:
                checkpoint = {}
            checkpoint['training_history'] = training_history
            path = os.path.join(args.save_dir, f'epoch_{epoch+1}_f1_{avg_val_f1:.4f}.pt')
            torch.save(checkpoint, path)
            print(f"💾 Saved periodic checkpoint -> {path}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - training_start_time
    print(f"\n🎉 Training completed!")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Final epoch: {epoch+1}")

    # Save final training history
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"📊 Training history saved to {args.save_dir}/training_history.json")


if __name__ == "__main__":
    main()
