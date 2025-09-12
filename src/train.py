import os
import torch
from torch.utils.data import DataLoader
from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from data.dataset import SERDataset
from utils import weighted_f1, energy_score
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import argparse
from data.preprocess import speed_perturb, add_noise_snr
from collections import Counter

NUM_LABELS = 6  # default; will be overridden if label map exists elsewhere

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
    parser.add_argument('--fusion_mode', type=str, default='gate', choices=['gate','concat'])
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    if args.device != 'auto':
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    train_ds = SERDataset(args.train_manifest)
    val_ds = SERDataset(args.val_manifest)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device=='cuda'))

    # Start with frozen encoders (freeze_base=True)
    audio_encoder = AudioEncoder(freeze_base=True).to(device)
    text_encoder = TextEncoder(freeze_base=True).to(device)
    audio_hid = audio_encoder.encoder.config.hidden_size
    text_hid = text_encoder.encoder.config.hidden_size
    cross = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8).to(device)
    pool_a = AttentiveStatsPooling(audio_hid).to(device)
    pool_t = AttentiveStatsPooling(text_hid).to(device)
    fusion = FusionLayer(audio_hid * 2, text_hid * 2, 1024).to(device)
    if args.fusion_mode == 'concat':
        classifier_in = audio_hid * 2 + text_hid * 2
    else:
        classifier_in = 1024
    classifier = Classifier(classifier_in, NUM_LABELS).to(device)

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
    class_weights = torch.tensor([max_count / max(1, label_counts.get(c, 1)) for c in range(NUM_LABELS)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = GradScaler(enabled=args.use_amp and device=='cuda')

    steps_per_epoch = len(train_loader)
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

    def unfreeze_last_layers():
        # Unfreeze last 4 transformer blocks in each encoder if available
        try:
            # Wav2Vec2
            wav_layers = getattr(audio_encoder.encoder, 'encoder').layers
            for p in wav_layers[-4:].parameters():
                p.requires_grad = True
        except Exception:
            pass
        try:
            # XLM-RoBERTa
            roberta_layers = getattr(text_encoder.encoder, 'encoder').layer
            for p in roberta_layers[-4:].parameters():
                p.requires_grad = True
        except Exception:
            pass

    for epoch in range(args.epochs):
        # After first epoch, unfreeze last layers to finetune
        if epoch == 1:
            unfreeze_last_layers()
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

            a_seq, a_mask = audio_encoder(audio_list)
            t_seq, t_mask = text_encoder(text_list)
            a_enh, t_enh = cross(a_seq, t_seq, a_mask, t_mask)
            a_vec = pool_a(a_enh, a_mask)
            t_vec = pool_t(t_enh, t_mask)
            if args.fusion_mode == 'concat':
                fused = torch.cat([a_vec, t_vec], dim=-1)
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
