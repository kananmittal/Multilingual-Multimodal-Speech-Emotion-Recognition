#!/usr/bin/env python3
"""
Two-phase training with differential learning rates as per academic requirements.
Phase 1: Representation learning with frozen encoders
Phase 2: Fine-tuning with differential learning rates
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.classifier import AdvancedOpenMaxClassifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.prototypes import PrototypeMemory
from models.cross_lingual_variance import CrossLingualVarianceHandler
from models.confidence_aware_fusion import ConfidenceAwareFusion
from models.temporal_modeling import TemporalModelingModule
from models.dual_gate_ood import DualGateOODDetector
from models.comprehensive_loss_integration import ComprehensiveLossIntegration, TrainingPhase
from data.dataset import SERDataset
from utils import weighted_f1, energy_score
from evaluation.cross_lingual_metrics import create_cross_lingual_evaluator
from evaluation.calibration_metrics import create_calibration_evaluator

class TwoPhaseTrainer:
    """Two-phase trainer with differential learning rates."""
    
    def __init__(self, args):
        self.args = args
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.current_phase = TrainingPhase.REPRESENTATION_LEARNING
        
        # Initialize models
        self._initialize_models()
        self._initialize_optimizers()
        self._initialize_loss_integration()
        
        # Metrics tracking
        self.training_history = {
            'phase1': {'losses': [], 'f1_scores': [], 'learning_rates': []},
            'phase2': {'losses': [], 'f1_scores': [], 'learning_rates': []}
        }
        
    def _initialize_models(self):
        """Initialize all model components."""
        print("Initializing models...")
        
        # Core encoders
        self.audio_encoder = AudioEncoder().to(self.device)
        self.text_encoder = TextEncoder().to(self.device)
        
        # Get hidden dimensions
        audio_hid = self.audio_encoder.encoder.config.hidden_size
        text_hid = self.text_encoder.encoder.config.hidden_size
        
        # Cross-modal components
        self.cross = CrossModalAttention(audio_hid, text_hid, shared_dim=256, num_heads=8).to(self.device)
        self.pool_a = AttentiveStatsPooling(audio_hid).to(self.device)
        self.pool_t = AttentiveStatsPooling(text_hid).to(self.device)
        self.fusion = ConfidenceAwareFusion(proj_dim=512).to(self.device)
        
        # Classifier and prototypes
        self.classifier = AdvancedOpenMaxClassifier(
            input_dim=512, num_labels=4, num_layers=35, 
            base_dim=512, dropout=0.15
        ).to(self.device)
        self.prototypes = PrototypeMemory(4, 512).to(self.device)
        
        # Advanced components
        self.cross_lingual_handler = CrossLingualVarianceHandler(
            audio_encoder=self.audio_encoder,
            text_encoder=self.text_encoder,
            fusion_layer=self.fusion,
            num_languages=5,  # en, fr, de, es, other
            adapter_size=64,
            consistency_weight=0.1
        ).to(self.device)
        
        self.temporal_module = TemporalModelingModule(
            hidden_dim=256, speaker_dim=128, num_emotions=4
        ).to(self.device)
        
        self.ood_detector = DualGateOODDetector(
            num_classes=4, feature_dim=512, num_languages=5,
            early_abstain=True, late_detection=True
        ).to(self.device)
        
        print("✅ All models initialized successfully!")
        
    def _initialize_optimizers(self):
        """Initialize optimizers with differential learning rates."""
        print("Initializing optimizers with differential learning rates...")
        
        # Phase 1: Representation learning (frozen encoders)
        self.phase1_optimizer = optim.AdamW([
            {'params': self.cross.parameters(), 'lr': 5e-4},
            {'params': self.pool_a.parameters(), 'lr': 5e-4},
            {'params': self.pool_t.parameters(), 'lr': 5e-4},
            {'params': self.fusion.parameters(), 'lr': 5e-4},
            {'params': self.classifier.parameters(), 'lr': 5e-4},
            {'params': self.prototypes.parameters(), 'lr': 5e-4},
            {'params': self.cross_lingual_handler.parameters(), 'lr': 5e-4},
            {'params': self.temporal_module.parameters(), 'lr': 5e-4},
            {'params': self.ood_detector.parameters(), 'lr': 5e-4}
        ], weight_decay=0.01)
        
        # Phase 2: Fine-tuning (unfrozen encoders with lower LR)
        self.phase2_optimizer = optim.AdamW([
            # Encoders with lower learning rate
            {'params': self.audio_encoder.parameters(), 'lr': 1e-5},
            {'params': self.text_encoder.parameters(), 'lr': 1e-5},
            # Other components with higher learning rate
            {'params': self.cross.parameters(), 'lr': 5e-4},
            {'params': self.pool_a.parameters(), 'lr': 5e-4},
            {'params': self.pool_t.parameters(), 'lr': 5e-4},
            {'params': self.fusion.parameters(), 'lr': 5e-4},
            {'params': self.classifier.parameters(), 'lr': 5e-4},
            {'params': self.prototypes.parameters(), 'lr': 5e-4},
            {'params': self.cross_lingual_handler.parameters(), 'lr': 5e-4},
            {'params': self.temporal_module.parameters(), 'lr': 5e-4},
            {'params': self.ood_detector.parameters(), 'lr': 5e-4}
        ], weight_decay=0.01)
        
        # Learning rate schedulers
        self.phase1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.phase1_optimizer, T_max=self.args.phase1_epochs, eta_min=1e-6
        )
        self.phase2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.phase2_optimizer, T_max=self.args.phase2_epochs, eta_min=1e-6
        )
        
        print("✅ Optimizers initialized with differential learning rates!")
        
    def _initialize_loss_integration(self):
        """Initialize comprehensive loss integration."""
        self.loss_integration = ComprehensiveLossIntegration(
            num_classes=4,
            feature_dim=512,
            prototype_memory=self.prototypes,
            temperature=0.07,
            margin=0.3
        )
        
    def _freeze_encoders(self):
        """Freeze encoder weights for phase 1."""
        print("🔒 Freezing encoder weights for representation learning...")
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        print("✅ Encoders frozen!")
        
    def _unfreeze_encoders(self):
        """Unfreeze encoder weights for phase 2."""
        print("🔓 Unfreezing encoder weights for fine-tuning...")
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        print("✅ Encoders unfrozen!")
        
    def _get_current_optimizer(self):
        """Get current optimizer based on training phase."""
        if self.current_phase == TrainingPhase.REPRESENTATION_LEARNING:
            return self.phase1_optimizer
        else:
            return self.phase2_optimizer
            
    def _get_current_scheduler(self):
        """Get current scheduler based on training phase."""
        if self.current_phase == TrainingPhase.REPRESENTATION_LEARNING:
            return self.phase1_scheduler
        else:
            return self.phase2_scheduler
            
    def _get_current_lr(self):
        """Get current learning rate."""
        optimizer = self._get_current_optimizer()
        return optimizer.param_groups[0]['lr']
        
    def train_phase(self, train_loader, val_loader, phase_name, num_epochs):
        """Train for a specific phase."""
        print(f"\n🚀 Starting {phase_name} training...")
        print(f"Learning rates: Encoders={self._get_current_lr():.2e}, Others=5e-4")
        
        best_f1 = 0.0
        scaler = GradScaler(enabled=self.args.use_amp)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_f1 = self._train_epoch(train_loader, scaler)
            
            # Validation
            val_loss, val_f1 = self._validate_epoch(val_loader)
            
            # Update scheduler
            scheduler = self._get_current_scheduler()
            scheduler.step()
            
            # Track metrics
            phase_key = 'phase1' if phase_name == 'Representation Learning' else 'phase2'
            self.training_history[phase_key]['losses'].append(train_loss)
            self.training_history[phase_key]['f1_scores'].append(val_f1)
            self.training_history[phase_key]['learning_rates'].append(self._get_current_lr())
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                self._save_checkpoint(phase_name, epoch, val_f1)
                
            print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            print(f"LR: {self._get_current_lr():.2e}")
            
        return best_f1
        
    def _train_epoch(self, train_loader, scaler):
        """Train for one epoch."""
        self._set_training_mode(True)
        
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        for batch_idx, (audio_list, text_list, labels) in enumerate(tqdm(train_loader, desc="Training")):
            labels = labels.to(self.device)
            
            optimizer = self._get_current_optimizer()
            optimizer.zero_grad()
            
            with autocast(enabled=self.args.use_amp):
                # Forward pass
                loss, preds = self._forward_pass(audio_list, text_list, labels)
                
            # Backward pass
            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        f1_score = weighted_f1(torch.tensor(all_preds), torch.tensor(all_labels))
        
        return avg_loss, f1_score
        
    def _validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self._set_training_mode(False)
        
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for audio_list, text_list, labels in tqdm(val_loader, desc="Validation"):
                labels = labels.to(self.device)
                
                # Forward pass
                loss, preds = self._forward_pass(audio_list, text_list, labels)
                
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        f1_score = weighted_f1(torch.tensor(all_preds), torch.tensor(all_labels))
        
        return avg_loss, f1_score
        
    def _forward_pass(self, audio_list, text_list, labels):
        """Perform forward pass through the model."""
        # Encode audio and text
        a_seq, a_mask = self.audio_encoder(audio_list, text_list)
        t_seq, t_mask = self.text_encoder(text_list)
        
        # Cross-modal attention
        a_enh, t_enh = self.cross(a_seq, t_seq, a_mask, t_mask)
        
        # Pooling
        a_vec = self.pool_a(a_enh, a_mask)
        t_vec = self.pool_t(t_enh, t_mask)
        
        # Fusion
        fused = self.fusion(a_vec, t_vec)
        
        # Temporal modeling
        temporal_output = self.temporal_module(fused, None, None)  # Simplified for now
        
        # Cross-lingual processing
        cross_lingual_output = self.cross_lingual_handler(a_seq, t_seq, a_mask, t_mask)
        
        # OOD detection
        ood_result = self.ood_detector(fused, None)  # Simplified for now
        
        # Classification
        logits, uncertainty, anchor_loss = self.classifier(fused, use_openmax=True, return_uncertainty=True)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Calculate comprehensive loss
        loss = self.loss_integration.compute_loss(
            logits, labels, fused, self.current_phase
        )
        
        return loss, preds
        
    def _set_training_mode(self, training):
        """Set training mode for all models."""
        self.audio_encoder.train(training)
        self.text_encoder.train(training)
        self.cross.train(training)
        self.pool_a.train(training)
        self.pool_t.train(training)
        self.fusion.train(training)
        self.classifier.train(training)
        self.prototypes.train(training)
        self.cross_lingual_handler.train(training)
        self.temporal_module.train(training)
        self.ood_detector.train(training)
        
    def _save_checkpoint(self, phase_name, epoch, f1_score):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'phase': phase_name,
            'f1_score': f1_score,
            'audio_encoder': self.audio_encoder.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'cross': self.cross.state_dict(),
            'pool_a': self.pool_a.state_dict(),
            'pool_t': self.pool_t.state_dict(),
            'fusion': self.fusion.state_dict(),
            'classifier': self.classifier.state_dict(),
            'prototypes': self.prototypes.state_dict(),
            'cross_lingual_handler': self.cross_lingual_handler.state_dict(),
            'temporal_module': self.temporal_module.state_dict(),
            'ood_detector': self.ood_detector.state_dict(),
            'training_history': self.training_history
        }
        
        filename = f"checkpoints/{phase_name.lower().replace(' ', '_')}_epoch_{epoch}_f1_{f1_score:.4f}.pt"
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")
        
    def run_training(self, train_loader, val_loader):
        """Run complete two-phase training."""
        print("🎯 Starting Two-Phase Training Protocol")
        print("=" * 60)
        
        # Phase 1: Representation Learning (frozen encoders)
        print("\n📚 PHASE 1: Representation Learning")
        print("Encoders: FROZEN (lr=0)")
        print("Other components: lr=5e-4")
        self._freeze_encoders()
        self.current_phase = TrainingPhase.REPRESENTATION_LEARNING
        
        phase1_f1 = self.train_phase(
            train_loader, val_loader, 
            "Representation Learning", 
            self.args.phase1_epochs
        )
        
        # Phase 2: Fine-tuning (unfrozen encoders)
        print("\n🔧 PHASE 2: Fine-tuning")
        print("Encoders: UNFROZEN (lr=1e-5)")
        print("Other components: lr=5e-4")
        self._unfreeze_encoders()
        self.current_phase = TrainingPhase.ADVERSARIAL_TRAINING
        
        phase2_f1 = self.train_phase(
            train_loader, val_loader, 
            "Fine-tuning", 
            self.args.phase2_epochs
        )
        
        # Final results
        print("\n" + "=" * 60)
        print("🎉 TWO-PHASE TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Phase 1 (Representation Learning): F1 = {phase1_f1:.4f}")
        print(f"Phase 2 (Fine-tuning): F1 = {phase2_f1:.4f}")
        print(f"Improvement: {((phase2_f1 - phase1_f1) / phase1_f1 * 100):+.2f}%")
        
        # Save training history
        history_file = f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"📊 Training history saved to: {history_file}")

def main():
    parser = argparse.ArgumentParser(description="Two-phase training with differential learning rates")
    parser.add_argument('--train_manifest', type=str, default='train_70.jsonl')
    parser.add_argument('--val_manifest', type=str, default='val_20.jsonl')
    parser.add_argument('--phase1_epochs', type=int, default=3, help='Epochs for representation learning')
    parser.add_argument('--phase2_epochs', type=int, default=2, help='Epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create data loaders
    train_ds = SERDataset(args.train_manifest)
    val_ds = SERDataset(args.val_manifest)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Initialize trainer
    trainer = TwoPhaseTrainer(args)
    
    # Run training
    trainer.run_training(train_loader, val_loader)

if __name__ == "__main__":
    main()
