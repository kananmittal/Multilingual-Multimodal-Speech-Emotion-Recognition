#!/usr/bin/env python3
"""
Final optimized CREMA training script with full MPS support and best practices
"""

import os
import torch
from torch.utils.data import DataLoader
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

NUM_LABELS = 4  # Angry, Happy, Sad, Neutral

def collate_fn(batch):
    audios, texts, labels = zip(*batch)
    return list(audios), list(texts), torch.tensor(labels, dtype=torch.long)

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warmup and restarts"""
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
    """Get the best available device with proper MPS support"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def main():
    parser = argparse.ArgumentParser(description="Final Optimized CREMA Training")
    parser.add_argument('--train_manifest', type=str, default='crema_train_70.jsonl')
    parser.add_argument('--val_manifest', type=str, default='crema_val_20.jsonl')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--proto_weight', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='checkpoints_crema_final')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.15)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    args = parser.parse_args()
    
    # Get optimal device
    device = get_optimal_device()
    print(f"🚀 Using device: {device}")
    print(f"📊 Training on CREMA dataset with optimized settings")
    
    if device == 'mps':
        print("🍎 Using Apple M3 GPU acceleration (MPS)")
    elif device == 'cuda':
        print("🚀 Using NVIDIA GPU acceleration (CUDA)")
    else:
        print("⚠️  Using CPU - consider GPU acceleration for faster training")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load datasets
    train_ds = SERDataset(args.train_manifest)
    val_ds = SERDataset(args.val_manifest)
    
    print(f"📁 Train samples: {len(train_ds)}")
    print(f"📁 Val samples: {len(val_ds)}")

    # Optimize DataLoader for device
    num_workers = 0 if device == 'mps' else 2  # MPS doesn't support multiprocessing
    pin_memory = (device == 'cuda')  # Only pin memory for CUDA
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Initialize models with enhanced dropout
    print("🔧 Initializing models...")
    audio_encoder = AudioEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    audio_hid = audio_encoder.encoder.config.hidden_size
    text_hid = text_encoder.encoder.config.hidden_size
    cross = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8, dropout=0.2).to(device)
    pool_a = AttentiveStatsPooling(audio_hid).to(device)
    pool_t = AttentiveStatsPooling(text_hid).to(device)
    fusion = FusionLayer(audio_hid * 2, text_hid * 2, 512).to(device)
    classifier = AdvancedOpenMaxClassifier(
        input_dim=512, 
        num_labels=NUM_LABELS, 
        num_layers=35, 
        base_dim=512, 
        dropout=0.25  # Higher dropout for better regularization
    ).to(device)
    prototypes = PrototypeMemory(NUM_LABELS, 512).to(device)

    # Enhanced optimizer with different learning rates and weight decay
    print("⚙️  Setting up optimizer...")
    optimizer = optim.AdamW([
        # Encoders: Lower LR, higher weight decay
        {'params': audio_encoder.parameters(), 'lr': args.lr * 0.05, 'weight_decay': 0.05},
        {'params': text_encoder.parameters(), 'lr': args.lr * 0.05, 'weight_decay': 0.05},
        
        # Attention and pooling: Medium LR
        {'params': cross.parameters(), 'lr': args.lr * 0.5, 'weight_decay': 0.1},
        {'params': pool_a.parameters(), 'lr': args.lr * 0.5, 'weight_decay': 0.1},
        {'params': pool_t.parameters(), 'lr': args.lr * 0.5, 'weight_decay': 0.1},
        
        # Fusion: Higher LR
        {'params': fusion.parameters(), 'lr': args.lr, 'weight_decay': 0.1},
        
        # Classifier: Highest LR for fast adaptation
        {'params': classifier.deep_classifier.parameters(), 'lr': args.lr * 2.0, 'weight_decay': 0.15},
        {'params': classifier.anchor_clustering.parameters(), 'lr': args.lr * 3.0, 'weight_decay': 0.1},
        {'params': classifier.uncertainty_head.parameters(), 'lr': args.lr * 1.5, 'weight_decay': 0.1},
        
        # Prototypes: Medium LR
        {'params': prototypes.parameters(), 'lr': args.lr * 0.8, 'weight_decay': 0.1},
    ], eps=1e-8, betas=(0.9, 0.999))

    # Enhanced loss functions
    ce_smooth = LabelSmoothingCrossEntropy(args.label_smoothing)
    cb_focal = ClassBalancedFocalLoss(beta=0.9999, gamma=args.focal_gamma, num_classes=NUM_LABELS)
    supcon = SupConLoss(temperature=0.05)  # Lower temperature for tighter clustering

    # MPS doesn't support AMP, so disable it for MPS
    use_amp = args.use_amp and device != 'mps'
    scaler = GradScaler(device, enabled=use_amp)
    
    if device == 'mps' and args.use_amp:
        print("⚠️  Mixed Precision (AMP) disabled for MPS - using full precision")

    # Load checkpoint if resuming
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0
    
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
        
        # Set starting epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔄 Resuming from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Enhanced scheduler with warmup and restarts
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        T_0=len(train_loader) * 3,  # Restart every 3 epochs
        T_mult=1,
        eta_min=args.lr * 0.01,
        warmup_epochs=args.warmup_epochs
    )
    
    # Load scheduler state after creating scheduler
    if args.resume_from and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    print(f"🎯 Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Focal gamma: {args.focal_gamma}")
    print(f"  Gradient clipping: {args.gradient_clip}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Early stopping patience: {args.patience}")

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
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (audio_list, text_list, labels) in enumerate(pbar):
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device, enabled=use_amp):
                # Forward pass
                a_seq, a_mask = audio_encoder(audio_list, text_list)
                t_seq, t_mask = text_encoder(text_list)
                a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                a_vec = pool_a(a_enh, a_mask)
                t_vec = pool_t(t_enh, t_mask)
                fused = fusion(a_vec, t_vec)
                
                # Get predictions and losses
                logits = classifier(fused, use_openmax=True)
                ce_loss = ce_smooth(logits, labels)
                focal_loss = cb_focal(logits, labels)
                
                # Prototype loss
                proto_loss = prototypes.prototype_loss(fused, labels)
                
                # Combined loss
                total_loss = ce_loss + 0.5 * focal_loss + args.proto_weight * proto_loss
                
                # Data augmentation
                if args.augment and torch.rand(1) < 0.3:
                    # Speed perturbation
                    aug_audio = [speed_perturb(audio, 0.9 + torch.rand(1) * 0.2) for audio in audio_list]
                    a_seq_aug, a_mask_aug = audio_encoder(aug_audio, text_list)
                    a_enh_aug, _ = cross(a_seq_aug, t_seq, a_mask_aug, t_mask)
                    a_vec_aug = pool_a(a_enh_aug, a_mask_aug)
                    fused_aug = fusion(a_vec_aug, t_vec)
                    logits_aug = classifier(fused_aug, use_openmax=True)
                    aug_loss = ce_smooth(logits_aug, labels)
                    total_loss += 0.3 * aug_loss
            
            # Backward pass with gradient clipping
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 
                args.gradient_clip
            )
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
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
        
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
        
        with torch.no_grad():
            for audio_list, text_list, labels in tqdm(val_loader, desc="Validation"):
                labels = labels.to(device)
                
                with autocast(device, enabled=use_amp):
                    a_seq, a_mask = audio_encoder(audio_list, text_list)
                    t_seq, t_mask = text_encoder(text_list)
                    a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                    a_vec = pool_a(a_enh, a_mask)
                    t_vec = pool_t(t_enh, t_mask)
                    fused = fusion(a_vec, t_vec)
                    
                    logits = classifier(fused, use_openmax=True)
                    ce_loss = ce_smooth(logits, labels)
                    focal_loss = cb_focal(logits, labels)
                    proto_loss = prototypes.prototype_loss(fused, labels)
                    total_loss = ce_loss + 0.5 * focal_loss + args.proto_weight * proto_loss
                    
                    preds = torch.argmax(logits, dim=1)
                    f1 = weighted_f1(preds, labels)
                    
                    val_loss += total_loss.item()
                    val_f1 += f1.item()
                    val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_f1 = val_f1 / val_batches
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val F1: {avg_val_f1:.4f}")
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
                'config': vars(args)
            }
            
            torch.save(checkpoint, os.path.join(args.save_dir, f'best_crema_f1_{avg_val_f1:.4f}.pt'))
            print(f"💾 Saved best model with F1: {avg_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{args.patience})")
        
        # Save epoch checkpoint
        torch.save(checkpoint, os.path.join(args.save_dir, f'epoch_{epoch}_f1_{avg_val_f1:.4f}.pt'))
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - training_start_time
    print(f"\n🎉 Training completed!")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Final epoch: {epoch+1}")

if __name__ == "__main__":
    main()
