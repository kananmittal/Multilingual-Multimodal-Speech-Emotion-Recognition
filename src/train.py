import os
import torch
from torch.utils.data import DataLoader
from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.classifier import OpenMaxClassifier, AdvancedOpenMaxClassifier
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

NUM_LABELS = 4  # Angry, Happy, Sad, Neutral (example)

def collate_fn(batch):
    audios, texts, labels = zip(*batch)
    # Keep variable-length waveforms as list; feature extractor handles padding later
    return list(audios), list(texts), torch.tensor(labels, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_manifest', type=str, default='train_70.jsonl')
    parser.add_argument('--val_manifest', type=str, default='val_20.jsonl')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--proto_weight', type=float, default=0.05)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    # Device selection with MPS support for Apple Silicon
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    train_ds = SERDataset(args.train_manifest)
    val_ds = SERDataset(args.val_manifest)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

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
        num_labels=NUM_LABELS, 
        num_layers=35,  # Deep architecture
        base_dim=512, 
        dropout=0.15  # Higher dropout for regularization
    ).to(device)
    prototypes = PrototypeMemory(NUM_LABELS, 512).to(device)

    # Advanced optimizer with different learning rates for different components
    optimizer = optim.AdamW([
        {'params': audio_encoder.parameters(), 'lr': args.lr * 0.1, 'weight_decay': 0.025},
        {'params': text_encoder.parameters(), 'lr': args.lr * 0.1, 'weight_decay': 0.025},
        {'params': cross.parameters(), 'lr': args.lr, 'weight_decay': 0.05},
        {'params': pool_a.parameters(), 'lr': args.lr, 'weight_decay': 0.05},
        {'params': pool_t.parameters(), 'lr': args.lr, 'weight_decay': 0.05},
        {'params': fusion.parameters(), 'lr': args.lr, 'weight_decay': 0.05},
        {'params': classifier.deep_classifier.parameters(), 'lr': args.lr * 1.5, 'weight_decay': 0.06},
        {'params': classifier.anchor_clustering.parameters(), 'lr': args.lr * 2.0, 'weight_decay': 0.04},
        {'params': classifier.uncertainty_head.parameters(), 'lr': args.lr * 1.0, 'weight_decay': 0.05},
        {'params': prototypes.parameters(), 'lr': args.lr, 'weight_decay': 0.05},
    ], weight_decay=0.05)
    ce_smooth = LabelSmoothingCrossEntropy(0.1)
    cb_focal = ClassBalancedFocalLoss(beta=0.9999, gamma=2.0, num_classes=NUM_LABELS)
    supcon = SupConLoss(temperature=0.07)

    scaler = GradScaler(enabled=args.use_amp)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Loading checkpoint from {args.resume_from}")
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
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Set starting epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(start_epoch, args.epochs):
        audio_encoder.train(); text_encoder.train(); fusion.train(); classifier.train()
        step = 0
        for (audio_list, text_list, labels) in tqdm(train_loader):
            labels = labels.to(device)

            # Optional augmentations
            if args.augment:
                aug_list = []
                for wav in audio_list:
                    w = wav
                    # speed perturb 50% chance
                    if torch.rand(1).item() < 0.5:
                        factor = 0.9 + 0.2 * torch.rand(1).item()  # [0.9, 1.1]
                        w = speed_perturb(w, factor)
                    # add noise 50% chance
                    if torch.rand(1).item() < 0.5:
                        snr = 10 + 10 * torch.rand(1).item()  # [10,20] dB
                        w = add_noise_snr(w, snr)
                    aug_list.append(w)
                audio_list = aug_list

            a_seq, a_mask = audio_encoder(audio_list, text_list)
            t_seq, t_mask = text_encoder(text_list)
            a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
            a_vec = pool_a(a_enh, a_mask)
            t_vec = pool_t(t_enh, t_mask)
            fused = fusion(a_vec, t_vec)
            with autocast(enabled=args.use_amp):
                logits, uncertainty, anchor_loss = classifier(fused, use_openmax=False, return_uncertainty=True)
                # Start with simpler loss combination
                ce_loss = ce_smooth(logits, labels)
                focal_loss = cb_focal(logits, labels)
                loss = ce_loss + 0.3 * focal_loss  # Reduced focal weight
                
                # Add anchor clustering loss for better class separation
                loss = loss + 0.1 * anchor_loss
                
                # Add uncertainty regularization (encourage low uncertainty for correct predictions)
                uncertainty_loss = torch.mean(uncertainty * (labels == logits.argmax(dim=1)).float())
                loss = loss + 0.05 * uncertainty_loss
                
                # Add prototype loss with smaller weight
                if args.proto_weight > 0:
                    proto_loss = prototypes.prototype_loss(fused, labels)
                    loss = loss + 0.01 * proto_loss  # Very small prototype weight
            optimizer.zero_grad(set_to_none=True)
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            step += 1

        # Eval
        audio_encoder.eval(); text_encoder.eval(); fusion.eval(); classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for (audio_list, text_list, labels) in val_loader:
                labels = labels.to(device)

                a_seq, a_mask = audio_encoder(audio_list, text_list)
                t_seq, t_mask = text_encoder(text_list)
                a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                a_vec = pool_a(a_enh, a_mask)
                t_vec = pool_t(t_enh, t_mask)
                fused = fusion(a_vec, t_vec)
                with torch.no_grad():
                    logits = classifier(fused)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())

        f1 = weighted_f1(torch.stack(all_preds), torch.stack(all_labels))
        print(f"Epoch {epoch} F1: {f1}")

        # Fit Weibull distributions for OpenMax after each epoch
        if epoch == args.epochs - 1:  # After last epoch
            print("Fitting Weibull distributions for OpenMax...")
            classifier.eval()
            all_features = []
            all_val_labels = []
            
            with torch.no_grad():
                for audio_list, text_list, labels in val_loader:
                    labels = labels.to(device)
                    a_seq, a_mask = audio_encoder(audio_list, text_list)
                    t_seq, t_mask = text_encoder(text_list)
                    a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                    a_vec = pool_a(a_enh, a_mask)
                    t_vec = pool_t(t_enh, t_mask)
                    fused = fusion(a_vec, t_vec)
                    
                    # Get features from deep classifier (before final classification layer)
                    features = fused
                    for i, layer in enumerate(classifier.deep_classifier.input_projection):
                        features = layer(features)
                    
                    for i, (residual_block, layer_norm) in enumerate(zip(
                        classifier.deep_classifier.residual_layers, 
                        classifier.deep_classifier.layer_norms
                    )):
                        features = layer_norm(features)
                        features = residual_block(features)
                    
                    # Get features before final classification layer
                    features = classifier.deep_classifier.output_projection[0](features)
                    features = classifier.deep_classifier.output_projection[1](features)
                    features = classifier.deep_classifier.output_projection[2](features)
                    features = classifier.deep_classifier.output_projection[3](features)
                    
                    all_features.append(features)
                    all_val_labels.append(labels)
            
            all_features = torch.cat(all_features, dim=0)
            all_val_labels = torch.cat(all_val_labels, dim=0)
            
            # Fit Weibull parameters
            classifier.fit_weibull(all_features, all_val_labels)

        # Save checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt = {
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
            'epoch': epoch,
            'f1': f1,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'epoch_{epoch}_f1_{f1:.4f}.pt'))

if __name__ == "__main__":
    main()
