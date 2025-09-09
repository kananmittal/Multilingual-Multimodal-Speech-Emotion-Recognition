#!/usr/bin/env python3
"""
M3 Mac Optimized CREMA Training Script - ULTRA STABLE VERSION
Handles NaN gradients and numerical instability issues
"""

import os
import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append('src')

from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.classifier import AdvancedOpenMaxClassifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.losses import LabelSmoothingCrossEntropy, ClassBalancedFocalLoss, SupConLoss
from models.prototypes import PrototypeMemory
from data.dataset import SERDataset
from utils import weighted_f1, energy_score
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse
from data.preprocess import speed_perturb, add_noise_snr
import math
import time
from datetime import datetime
import json
import numpy as np

NUM_LABELS = 4  # Angry, Happy, Sad, Neutral

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
            return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

def get_optimal_device():
    """Get the best available device - optimized for M3 Mac"""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def check_for_nan(tensor, name="tensor"):
    """Check for NaN values and handle them"""
    if torch.isnan(tensor).any():
        print(f"⚠️  NaN detected in {name}")
        return True
    return False

def safe_forward_pass(audio_encoder, text_encoder, cross, pool_a, pool_t, fusion, classifier, audio_list, text_list, device):
    """Safe forward pass with NaN checking at each step"""
    try:
        # Audio encoding
        a_seq, a_mask = audio_encoder(audio_list, text_list)
        if check_for_nan(a_seq, "audio_sequence"):
            return None, None, None, None
        
        # Text encoding
        t_seq, t_mask = text_encoder(text_list)
        if check_for_nan(t_seq, "text_sequence"):
            return None, None, None, None
        
        # Cross attention
        a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
        if check_for_nan(a_enh, "audio_enhanced") or check_for_nan(t_enh, "text_enhanced"):
            return None, None, None, None
        
        # Pooling
        a_vec = pool_a(a_enh, a_mask)
        t_vec = pool_t(t_enh, t_mask)
        if check_for_nan(a_vec, "audio_vector") or check_for_nan(t_vec, "text_vector"):
            return None, None, None, None
        
        # Fusion
        fused = fusion(a_vec, t_vec)
        if check_for_nan(fused, "fused_features"):
            return None, None, None, None
        
        # Classification
        logits = classifier(fused, use_openmax=True)
        if check_for_nan(logits, "logits"):
            return None, None, None, None
        
        return logits, fused, a_vec, t_vec
        
    except Exception as e:
        print(f"⚠️  Error in forward pass: {e}")
        return None, None, None, None

def safe_loss_computation(logits, labels, ce_loss_fn, focal_loss_fn, proto_loss_fn, fused_features, device):
    """Safely compute losses with extensive NaN checking"""
    try:
        # Check inputs
        if check_for_nan(logits, "logits"):
            return None, None, None
        
        # Clamp logits to prevent overflow
        logits_clipped = torch.clamp(logits, min=-5.0, max=5.0)
        
        # Compute CE loss
        ce_loss = ce_loss_fn(logits_clipped, labels)
        if check_for_nan(ce_loss, "ce_loss"):
            return None, None, None
        
        # Clamp CE loss
        ce_loss = torch.clamp(ce_loss, min=0.0, max=10.0)
        
        # Compute focal loss
        focal_loss = focal_loss_fn(logits_clipped, labels)
        if check_for_nan(focal_loss, "focal_loss"):
            return None, None, None
        
        # Clamp focal loss
        focal_loss = torch.clamp(focal_loss, min=0.0, max=10.0)
        
        # Compute prototype loss
        proto_loss = proto_loss_fn(fused_features, labels)
        if check_for_nan(proto_loss, "proto_loss"):
            return None, None, None
        
        # Clamp prototype loss
        proto_loss = torch.clamp(proto_loss, min=0.0, max=10.0)
        
        return ce_loss, focal_loss, proto_loss
        
    except Exception as e:
        print(f"⚠️  Error in loss computation: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="M3 Mac Optimized CREMA Training - ULTRA STABLE")
    parser.add_argument('--train_manifest', type=str, default='crema_train_70.jsonl')
    parser.add_argument('--val_manifest', type=str, default='crema_val_20.jsonl')
    parser.add_argument('--test_manifest', type=str, default='crema_test_10.jsonl')
    parser.add_argument('--epochs', type=int, default=15)  # Reduced epochs
    parser.add_argument('--batch_size', type=int, default=8)  # Very small batch size for stability
    parser.add_argument('--lr', type=float, default=1e-5)  # Very low LR for stability
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--use_amp', action='store_true', default=False)  # Disabled for MPS
    parser.add_argument('--augment', action='store_true', default=False)  # Disabled for stability
    parser.add_argument('--proto_weight', type=float, default=0.001)  # Very small weight
    parser.add_argument('--save_dir', type=str, default='checkpoints_crema_m3_stable')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_clip', type=float, default=0.01)  # Very low clipping
    parser.add_argument('--label_smoothing', type=float, default=0.01)  # Very small smoothing
    parser.add_argument('--focal_gamma', type=float, default=0.5)  # Very low gamma
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=2, help='Save checkpoint every N epochs')
    args = parser.parse_args()
    
    # Get optimal device
    device = get_optimal_device()
    print(f"🍎 Using device: {device}")
    print(f"📊 Training on CREMA dataset with M3 Mac optimization (ULTRA STABLE VERSION)")
    
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
        persistent_workers=False  # MPS optimization
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=False  # MPS optimization
    )

    # Initialize models with ultra-conservative settings
    print("🔧 Initializing models...")
    audio_encoder = AudioEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    audio_hid = audio_encoder.encoder.config.hidden_size
    text_hid = text_encoder.encoder.config.hidden_size
    cross = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8, dropout=0.05).to(device)
    pool_a = AttentiveStatsPooling(audio_hid).to(device)
    pool_t = AttentiveStatsPooling(text_hid).to(device)
    fusion = FusionLayer(audio_hid * 2, text_hid * 2, 512).to(device)
    classifier = AdvancedOpenMaxClassifier(
        input_dim=512, 
        num_labels=NUM_LABELS, 
        num_layers=15,  # Much fewer layers for stability
        base_dim=512, 
        dropout=0.05  # Very low dropout
    ).to(device)
    prototypes = PrototypeMemory(NUM_LABELS, 512).to(device)

    # Ultra-conservative weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # Apply initialization
    cross.apply(init_weights)
    pool_a.apply(init_weights)
    pool_t.apply(init_weights)
    fusion.apply(init_weights)
    classifier.apply(init_weights)
    prototypes.apply(init_weights)

    # Ultra-conservative optimizer
    print("⚙️  Setting up optimizer...")
    optimizer = optim.AdamW([
        # Encoders: Extremely low LR
        {'params': audio_encoder.parameters(), 'lr': args.lr * 0.001, 'weight_decay': 0.2},
        {'params': text_encoder.parameters(), 'lr': args.lr * 0.001, 'weight_decay': 0.2},
        
        # Attention and pooling: Very low LR
        {'params': cross.parameters(), 'lr': args.lr * 0.1, 'weight_decay': 0.3},
        {'params': pool_a.parameters(), 'lr': args.lr * 0.1, 'weight_decay': 0.3},
        {'params': pool_t.parameters(), 'lr': args.lr * 0.1, 'weight_decay': 0.3},
        
        # Fusion: Low LR
        {'params': fusion.parameters(), 'lr': args.lr * 0.3, 'weight_decay': 0.25},
        
        # Classifier: Medium LR
        {'params': classifier.deep_classifier.parameters(), 'lr': args.lr * 0.5, 'weight_decay': 0.3},
        {'params': classifier.anchor_clustering.parameters(), 'lr': args.lr * 0.6, 'weight_decay': 0.25},
        {'params': classifier.uncertainty_head.parameters(), 'lr': args.lr * 0.4, 'weight_decay': 0.25},
        
        # Prototypes: Low LR
        {'params': prototypes.parameters(), 'lr': args.lr * 0.2, 'weight_decay': 0.25},
    ], eps=1e-8, betas=(0.9, 0.999))

    # Ultra-conservative loss functions
    ce_smooth = LabelSmoothingCrossEntropy(args.label_smoothing)
    cb_focal = ClassBalancedFocalLoss(beta=0.9999, gamma=args.focal_gamma, num_classes=NUM_LABELS)
    supcon = SupConLoss(temperature=0.2)  # Higher temperature for stability

    # MPS doesn't support AMP, so disable it
    use_amp = False  # Always disabled for MPS
    scaler = GradScaler(device, enabled=use_amp)
    
    print("⚠️  Mixed Precision (AMP) disabled for MPS compatibility")

    # Load checkpoint if resuming
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0
    training_history = []
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"📂 Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        # Load model states
        audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        cross.load_state_dict(checkpoint['cross'])
        pool_a.load_state_dict(checkpoint['pool_a'])
        pool_t.load_state_dict(checkpoint['pool_t'])
        fusion.load_state_dict(checkpoint['fusion'])
        classifier.load_state_dict(checkpoint['classifier'])
        prototypes.load_state_dict(checkpoint['prototypes'])
        
        # Load optimizer and scheduler states
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'best_f1' in checkpoint:
            best_f1 = checkpoint['best_f1']
        if 'training_history' in checkpoint:
            training_history = checkpoint['training_history']
        
        # Set starting epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔄 Resuming from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Ultra-conservative scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        T_0=len(train_loader) * 2,  # Restart every 2 epochs
        T_mult=1,
        eta_min=args.lr * 0.0001,
        warmup_epochs=args.warmup_epochs
    )
    
    # Load scheduler state after creating scheduler
    if args.resume_from and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    print(f"🎯 M3 Mac Training Configuration (ULTRA STABLE):")
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
        audio_encoder.train()
        text_encoder.train()
        cross.train()
        pool_a.train()
        pool_t.train()
        fusion.train()
        classifier.train()
        prototypes.train()
        
        train_loss = 0.0
        train_f1 = 0.0
        num_batches = 0
        nan_count = 0
        skip_count = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (audio_list, text_list, labels) in enumerate(pbar):
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device, enabled=use_amp):
                # Safe forward pass
                logits, fused, a_vec, t_vec = safe_forward_pass(
                    audio_encoder, text_encoder, cross, pool_a, pool_t, fusion, classifier, 
                    audio_list, text_list, device
                )
                
                if logits is None:  # NaN detected in forward pass
                    skip_count += 1
                    print(f"⚠️  Skipping batch {batch_idx} due to NaN in forward pass")
                    continue
                
                # Safe loss computation
                ce_loss, focal_loss, proto_loss = safe_loss_computation(
                    logits, labels, ce_smooth, cb_focal, prototypes.prototype_loss, fused, device
                )
                
                if ce_loss is None:  # NaN detected in loss computation
                    nan_count += 1
                    print(f"⚠️  Skipping batch {batch_idx} due to NaN in loss computation")
                    continue
                
                # Combined loss with ultra-conservative weights
                total_loss = ce_loss + 0.1 * focal_loss + args.proto_weight * proto_loss
                
                # Check final loss
                if check_for_nan(total_loss, "total_loss"):
                    nan_count += 1
                    print(f"⚠️  Skipping batch {batch_idx} due to NaN in total loss")
                    continue
            
            # Backward pass with ultra-conservative gradient clipping
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            
            # Check gradients for NaN
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 
                args.gradient_clip
            )
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"⚠️  NaN/Inf gradient detected, skipping update")
                optimizer.zero_grad()
                skip_count += 1
                continue
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Calculate metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                f1 = weighted_f1(preds, labels)
                train_loss += total_loss.item()
                train_f1 += f1.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'F1': f'{f1.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'NaN': f'{nan_count}',
                    'Skip': f'{skip_count}'
                })
        
        if num_batches == 0:
            print("⚠️  No valid batches in this epoch, skipping validation")
            continue
            
        avg_train_loss = train_loss / num_batches
        avg_train_f1 = train_f1 / num_batches
        
        # Validation phase
        audio_encoder.eval()
        text_encoder.eval()
        cross.eval()
        pool_a.eval()
        pool_t.eval()
        fusion.eval()
        classifier.eval()
        prototypes.eval()
        
        val_loss = 0.0
        val_f1 = 0.0
        val_batches = 0
        val_nan_count = 0
        val_skip_count = 0
        
        with torch.no_grad():
            for audio_list, text_list, labels in tqdm(val_loader, desc="Validation"):
                labels = labels.to(device)
                
                with autocast(device, enabled=use_amp):
                    # Safe forward pass
                    logits, fused, a_vec, t_vec = safe_forward_pass(
                        audio_encoder, text_encoder, cross, pool_a, pool_t, fusion, classifier, 
                        audio_list, text_list, device
                    )
                    
                    if logits is None:  # NaN detected in forward pass
                        val_skip_count += 1
                        continue
                    
                    # Safe loss computation
                    ce_loss, focal_loss, proto_loss = safe_loss_computation(
                        logits, labels, ce_smooth, cb_focal, prototypes.prototype_loss, fused, device
                    )
                    
                    if ce_loss is None:  # NaN detected in loss computation
                        val_nan_count += 1
                        continue
                    
                    total_loss = ce_loss + 0.1 * focal_loss + args.proto_weight * proto_loss
                    
                    if check_for_nan(total_loss, "val_total_loss"):
                        val_nan_count += 1
                        continue
                    
                    preds = torch.argmax(logits, dim=1)
                    f1 = weighted_f1(preds, labels)
                    
                    val_loss += total_loss.item()
                    val_f1 += f1.item()
                    val_batches += 1
        
        if val_batches == 0:
            print("⚠️  No valid validation batches, skipping epoch")
            continue
            
        avg_val_loss = val_loss / val_batches
        avg_val_f1 = val_f1 / val_batches
        
        epoch_time = time.time() - epoch_start_time
        
        # Store training history
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_f1': avg_train_f1,
            'val_loss': avg_val_loss,
            'val_f1': avg_val_f1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
            'nan_count': nan_count,
            'val_nan_count': val_nan_count,
            'skip_count': skip_count,
            'val_skip_count': val_skip_count
        }
        training_history.append(epoch_stats)
        
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val F1: {avg_val_f1:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Epoch Time: {epoch_time/60:.1f} minutes")
        print(f"  NaN batches: {nan_count} train, {val_nan_count} val")
        print(f"  Skip batches: {skip_count} train, {val_skip_count} val")
        
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
            
            torch.save(checkpoint, os.path.join(args.save_dir, f'best_crema_m3_stable_f1_{avg_val_f1:.4f}.pt'))
            print(f"💾 Saved best model with F1: {avg_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{args.patience})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint['training_history'] = training_history
            torch.save(checkpoint, os.path.join(args.save_dir, f'epoch_{epoch+1}_f1_{avg_val_f1:.4f}.pt'))
            print(f"💾 Saved periodic checkpoint")
        
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
