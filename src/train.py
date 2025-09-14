import os
import torch
from torch.utils.data import DataLoader
from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.advanced_fusion import AdvancedFusionLayer, MultiScaleFusionLayer
from data.dataset import SERDataset
from utils import weighted_f1, energy_score
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import argparse
from data.preprocess import speed_perturb, add_noise_snr
from collections import Counter

NUM_LABELS = 6  # default; will be overridden dynamically below

def collate_fn(batch):
    audios, texts, labels = zip(*batch)
    # Keep variable-length waveforms as list; feature extractor handles padding later
    return list(audios), list(texts), torch.tensor(labels, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_manifest', type=str, default='train_manifest.jsonl')
    parser.add_argument('--val_manifest', type=str, default='val_manifest.jsonl')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--proto_weight', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','mps','cuda'])
    parser.add_argument('--fusion_mode', type=str, default='advanced', choices=['gate','concat','advanced','multiscale'])
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--fusion_dim', type=int, default=2048, help='Fusion layer dimension')
    parser.add_argument('--unfreeze_epochs', type=int, default=2, help='Epochs to wait before unfreezing encoders')
    args = parser.parse_args()
    if args.device != 'auto':
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    train_ds = SERDataset(args.train_manifest)
    val_ds = SERDataset(args.val_manifest)
    # Dynamically determine number of labels from dataset
    num_labels = max([it['label'] for it in train_ds.items] + [0]) + 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device=='cuda'))

    # Start with frozen encoders (freeze_base=True)
    audio_encoder = AudioEncoder(freeze_base=True).to(device)
    text_encoder = TextEncoder(freeze_base=True).to(device)
    audio_hid = audio_encoder.encoder.config.hidden_size
    text_hid = text_encoder.encoder.config.hidden_size
    cross = CrossModalAttention(audio_hid, text_hid, shared_dim=512, num_heads=8).to(device)
    pool_a = AttentiveStatsPooling(audio_hid).to(device)
    pool_t = AttentiveStatsPooling(text_hid).to(device)
    
    # Advanced fusion layer
    if args.fusion_mode == 'advanced':
        fusion = AdvancedFusionLayer(audio_hid * 2, text_hid * 2, args.fusion_dim).to(device)
        classifier_in = args.fusion_dim
    elif args.fusion_mode == 'multiscale':
        # Multi-scale fusion with different encoder layers
        fusion = MultiScaleFusionLayer([audio_hid, audio_hid//2], [text_hid, text_hid//2], args.fusion_dim).to(device)
        classifier_in = args.fusion_dim
    else:
        # Original fusion
        fusion = FusionLayer(audio_hid * 2, text_hid * 2, args.fusion_dim).to(device)
        if args.fusion_mode == 'concat':
            classifier_in = audio_hid * 2 + text_hid * 2
        else:
            classifier_in = args.fusion_dim
    
    classifier = Classifier(classifier_in, num_labels).to(device)

    # Build param groups: encoders (low lr), heads (higher lr)
    encoder_params = list(audio_encoder.parameters()) + list(text_encoder.parameters())
    head_params = list(cross.parameters()) + list(pool_a.parameters()) + list(pool_t.parameters()) + \
                  list(fusion.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1, 'weight_decay': 0.01},
        {'params': head_params, 'lr': args.lr, 'weight_decay': 0.01},
    ])
    
    # Compute class weights from training labels to handle imbalance
    label_counts = Counter([it['label'] for it in train_ds.items])
    max_count = max(label_counts.values()) if label_counts else 1
    class_weights = torch.tensor([max_count / max(1, label_counts.get(c, 1)) for c in range(num_labels)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = GradScaler(enabled=args.use_amp and device=='cuda')

    steps_per_epoch = len(train_loader)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        cross.load_state_dict(checkpoint['cross'])
        pool_a.load_state_dict(checkpoint['pool_a'])
        pool_t.load_state_dict(checkpoint['pool_t'])
        fusion.load_state_dict(checkpoint['fusion'])
        classifier.load_state_dict(checkpoint['classifier'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")
        
        # Create new scheduler for remaining epochs
        remaining_epochs = args.epochs - start_epoch
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[args.lr * 0.1, args.lr],
            steps_per_epoch=max(1, steps_per_epoch),
            epochs=remaining_epochs,
            pct_start=args.warmup_ratio,
            anneal_strategy='cos',
            div_factor=1.0,
            final_div_factor=10.0,
        )
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[args.lr * 0.1, args.lr],
            steps_per_epoch=max(1, steps_per_epoch),
            epochs=args.epochs,
            pct_start=args.warmup_ratio,
            anneal_strategy='cos',
            div_factor=1.0,
            final_div_factor=10.0,
        )

    def unfreeze_encoders():
        # Unfreeze more layers progressively
        try:
            # Wav2Vec2 - unfreeze last 8 layers (more aggressive)
            wav_layers = getattr(audio_encoder.encoder, 'encoder').layers
            for p in wav_layers[-8:].parameters():
                p.requires_grad = True
            print(f"Unfroze last 8 Wav2Vec2 layers ({len(wav_layers)} total)")
        except Exception as e:
            print(f"Could not unfreeze Wav2Vec2 layers: {e}")
        
        try:
            # XLM-RoBERTa - unfreeze last 8 layers
            roberta_layers = getattr(text_encoder.encoder, 'encoder').layer
            for p in roberta_layers[-8:].parameters():
                p.requires_grad = True
            print(f"Unfroze last 8 XLM-RoBERTa layers ({len(roberta_layers)} total)")
        except Exception as e:
            print(f"Could not unfreeze XLM-RoBERTa layers: {e}")
    
    def unfreeze_all_encoders():
        # Unfreeze all encoder parameters
        for p in audio_encoder.parameters():
            p.requires_grad = True
        for p in text_encoder.parameters():
            p.requires_grad = True
        print("Unfroze all encoder parameters")

    for epoch in range(start_epoch, args.epochs):
        # Progressive unfreezing strategy
        if epoch == args.unfreeze_epochs:
            unfreeze_encoders()
        elif epoch == args.unfreeze_epochs + 2:
            unfreeze_all_encoders()
        audio_encoder.train(); text_encoder.train(); fusion.train(); classifier.train()
        step = 0
        for (audio_list, text_list, labels) in tqdm(train_loader):
            labels = labels.to(device)

            # Aggressive augmentations
            if args.augment:
                aug_list = []
                aug_text_list = []
                for wav, text in zip(audio_list, text_list):
                    w = wav
                    t = text
                    
                    # Audio augmentations (80% chance each)
                    if torch.rand(1).item() < 0.8:
                        factor = 0.8 + 0.4 * torch.rand(1).item()  # [0.8, 1.2] - more aggressive
                        w = speed_perturb(w, factor)
                    if torch.rand(1).item() < 0.8:
                        snr = 5 + 15 * torch.rand(1).item()  # [5,20] dB - more noise
                        w = add_noise_snr(w, snr)
                    if torch.rand(1).item() < 0.5:
                        # Volume scaling
                        scale = 0.5 + 1.0 * torch.rand(1).item()  # [0.5, 1.5]
                        w = w * scale
                    
                    # Text augmentations (50% chance)
                    if torch.rand(1).item() < 0.5 and len(t.strip()) > 0:
                        # Random word dropout
                        words = t.split()
                        if len(words) > 2:
                            keep_prob = 0.7 + 0.2 * torch.rand(1).item()  # [0.7, 0.9]
                            keep_words = [w for w in words if torch.rand(1).item() < keep_prob]
                            if len(keep_words) > 0:
                                t = ' '.join(keep_words)
                    
                    aug_list.append(w)
                    aug_text_list.append(t)
                audio_list = aug_list
                text_list = aug_text_list

            a_seq, a_mask = audio_encoder(audio_list)
            t_seq, t_mask = text_encoder(text_list)
            a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
            a_vec = pool_a(a_enh, a_mask)
            t_vec = pool_t(t_enh, t_mask)
            
            if args.fusion_mode == 'concat':
                fused = torch.cat([a_vec, t_vec], dim=-1)
            elif args.fusion_mode in ['advanced', 'multiscale']:
                fused = fusion(a_vec, t_vec, a_mask, t_mask)
            else:
                fused = fusion(a_vec, t_vec)
            with autocast(enabled=(args.use_amp and device=='cuda')):
                logits = classifier(fused)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            if args.use_amp and device=='cuda':
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

                a_seq, a_mask = audio_encoder(audio_list)
                t_seq, t_mask = text_encoder(text_list)
                a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
                a_vec = pool_a(a_enh, a_mask)
                t_vec = pool_t(t_enh, t_mask)
                if args.fusion_mode == 'concat':
                    fused = torch.cat([a_vec, t_vec], dim=-1)
                else:
                    fused = fusion(a_vec, t_vec)
                logits = classifier(fused)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())

        f1 = weighted_f1(torch.stack(all_preds), torch.stack(all_labels))
        print(f"Epoch {epoch} F1: {f1}")

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
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'f1': f1,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f'epoch_{epoch}_f1_{float(f1):.4f}.pt'))

if __name__ == "__main__":
    main()
