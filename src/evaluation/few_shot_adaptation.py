#!/usr/bin/env python3
"""
Few-shot adaptation pipeline for testing model performance with limited labeled data.
Implements adaptation experiments and performance gap analysis.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.classifier import AdvancedOpenMaxClassifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.prototypes import PrototypeMemory
from data.dataset import SERDataset
from utils import weighted_f1, energy_score

@dataclass
class FewShotResult:
    """Results from few-shot adaptation experiment."""
    num_shots: int
    target_language: str
    f1_score: float
    accuracy: float
    adaptation_gap: float  # Performance gap from zero-shot to full fine-tuning
    recovery_rate: float   # Percentage of gap recovered

class FewShotAdapter:
    """Handles few-shot adaptation experiments."""
    
    def __init__(self, base_model, device):
        self.base_model = base_model
        self.device = device
        self.results = []
        
    def create_few_shot_dataset(self, full_dataset: SERDataset, num_shots: int, 
                               target_language: str = None) -> Tuple[SERDataset, SERDataset]:
        """Create few-shot training set and evaluation set."""
        
        # Get indices for target language if specified
        if target_language:
            # This is a simplified approach - in practice you'd need language labels
            target_indices = list(range(len(full_dataset)))
        else:
            target_indices = list(range(len(full_dataset)))
        
        # Ensure we have enough samples
        if len(target_indices) < num_shots:
            print(f"Warning: Only {len(target_indices)} samples available, using all")
            num_shots = len(target_indices)
        
        # Randomly sample few-shot examples
        random.seed(42)  # For reproducibility
        few_shot_indices = random.sample(target_indices, num_shots)
        remaining_indices = [i for i in target_indices if i not in few_shot_indices]
        
        # Create few-shot dataset
        few_shot_dataset = Subset(full_dataset, few_shot_indices)
        eval_dataset = Subset(full_dataset, remaining_indices)
        
        return few_shot_dataset, eval_dataset
    
    def adapt_model(self, few_shot_dataset: SERDataset, num_epochs: int = 5) -> Dict:
        """Adapt the model using few-shot data."""
        
        # Create data loader
        few_shot_loader = DataLoader(few_shot_dataset, batch_size=4, shuffle=True)
        
        # Clone base model for adaptation
        adapted_model = self._clone_model(self.base_model)
        
        # Fine-tune only the classifier and fusion layers (freeze encoders)
        self._freeze_encoders(adapted_model)
        
        # Setup optimizer for adaptation
        optimizer = torch.optim.AdamW([
            {'params': adapted_model['fusion'].parameters(), 'lr': 1e-4},
            {'params': adapted_model['classifier'].parameters(), 'lr': 1e-4},
            {'params': adapted_model['prototypes'].parameters(), 'lr': 1e-4}
        ], weight_decay=0.01)
        
        # Training loop
        adapted_model['fusion'].train()
        adapted_model['classifier'].train()
        adapted_model['prototypes'].train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            all_preds, all_labels = [], []
            
            for audio_list, text_list, labels in few_shot_loader:
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    a_seq, a_mask = adapted_model['audio_encoder'](audio_list, text_list)
                    t_seq, t_mask = adapted_model['text_encoder'](text_list)
                
                a_enh, t_enh = adapted_model['cross'](a_seq, t_seq, a_mask, t_mask)
                a_vec = adapted_model['pool_a'](a_enh, a_mask)
                t_vec = adapted_model['pool_t'](t_enh, t_mask)
                fused = adapted_model['fusion'](a_vec, t_vec)
                
                # Classification
                logits = adapted_model['classifier'](fused, use_openmax=True)
                
                # Loss
                loss = torch.nn.functional.cross_entropy(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            f1 = weighted_f1(torch.tensor(all_preds), torch.tensor(all_labels))
            accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={total_loss/len(few_shot_loader):.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")
        
        return adapted_model
    
    def evaluate_adaptation(self, adapted_model: Dict, eval_dataset: SERDataset) -> Tuple[float, float]:
        """Evaluate adapted model performance."""
        
        eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
        
        adapted_model['fusion'].eval()
        adapted_model['classifier'].eval()
        adapted_model['prototypes'].eval()
        
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for audio_list, text_list, labels in eval_loader:
                labels = labels.to(self.device)
                
                # Forward pass
                a_seq, a_mask = adapted_model['audio_encoder'](audio_list, text_list)
                t_seq, t_mask = adapted_model['text_encoder'](text_list)
                
                a_enh, t_enh = adapted_model['cross'](a_seq, t_seq, a_mask, t_mask)
                a_vec = adapted_model['pool_a'](a_enh, a_mask)
                t_vec = adapted_model['pool_t'](t_enh, t_mask)
                fused = adapted_model['fusion'](a_vec, t_vec)
                
                # Classification
                logits = adapted_model['classifier'](fused, use_openmax=True)
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        f1 = weighted_f1(torch.tensor(all_preds), torch.tensor(all_labels))
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        return f1, accuracy
    
    def run_few_shot_experiment(self, full_dataset: SERDataset, 
                               shot_counts: List[int] = [10, 25, 50, 100, 250, 500],
                               target_language: str = None,
                               zero_shot_performance: float = None,
                               full_fine_tune_performance: float = None) -> List[FewShotResult]:
        """Run complete few-shot adaptation experiment."""
        
        print(f"🚀 Starting Few-Shot Adaptation Experiment")
        print(f"Target Language: {target_language or 'All'}")
        print(f"Shot Counts: {shot_counts}")
        
        results = []
        
        for num_shots in shot_counts:
            print(f"\n📊 Testing with {num_shots} shots...")
            
            # Create few-shot dataset
            few_shot_dataset, eval_dataset = self.create_few_shot_dataset(
                full_dataset, num_shots, target_language
            )
            
            print(f"  Few-shot samples: {len(few_shot_dataset)}")
            print(f"  Evaluation samples: {len(eval_dataset)}")
            
            # Adapt model
            adapted_model = self.adapt_model(few_shot_dataset)
            
            # Evaluate
            f1_score, accuracy = self.evaluate_adaptation(adapted_model, eval_dataset)
            
            # Calculate adaptation gap and recovery rate
            adaptation_gap = 0.0
            recovery_rate = 0.0
            
            if zero_shot_performance is not None and full_fine_tune_performance is not None:
                adaptation_gap = full_fine_tune_performance - zero_shot_performance
                performance_gap = full_fine_tune_performance - f1_score
                recovery_rate = max(0, (adaptation_gap - performance_gap) / adaptation_gap) * 100
            
            # Store results
            result = FewShotResult(
                num_shots=num_shots,
                target_language=target_language or 'all',
                f1_score=f1_score,
                accuracy=accuracy,
                adaptation_gap=adaptation_gap,
                recovery_rate=recovery_rate
            )
            
            results.append(result)
            
            print(f"  ✅ F1: {f1_score:.4f}, Accuracy: {accuracy:.4f}")
            if recovery_rate > 0:
                print(f"  📈 Recovery Rate: {recovery_rate:.1f}%")
        
        self.results = results
        return results
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate few-shot adaptation report."""
        
        if not self.results:
            return "No results available. Run experiment first."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FEW-SHOT ADAPTATION EXPERIMENT REPORT")
        report_lines.append("=" * 80)
        
        # Summary table
        report_lines.append(f"\nRESULTS SUMMARY:")
        report_lines.append(f"{'Shots':<8} {'F1':<8} {'Accuracy':<10} {'Recovery':<10}")
        report_lines.append("-" * 40)
        
        for result in self.results:
            recovery_str = f"{result.recovery_rate:.1f}%" if result.recovery_rate > 0 else "N/A"
            report_lines.append(f"{result.num_shots:<8} {result.f1_score:<8.4f} {result.accuracy:<10.4f} {recovery_str:<10}")
        
        # Analysis
        report_lines.append(f"\nANALYSIS:")
        
        # Performance scaling
        f1_scores = [r.f1_score for r in self.results]
        shot_counts = [r.num_shots for r in self.results]
        
        if len(f1_scores) > 1:
            # Calculate improvement rate
            improvement_rate = (f1_scores[-1] - f1_scores[0]) / (shot_counts[-1] - shot_counts[0])
            report_lines.append(f"  Performance improvement per shot: {improvement_rate:.6f}")
            
            # Find optimal shot count
            best_result = max(self.results, key=lambda x: x.f1_score)
            report_lines.append(f"  Best performance: {best_result.f1_score:.4f} with {best_result.num_shots} shots")
            
            # Diminishing returns analysis
            if len(f1_scores) >= 3:
                early_improvement = (f1_scores[1] - f1_scores[0]) / (shot_counts[1] - shot_counts[0])
                late_improvement = (f1_scores[-1] - f1_scores[-2]) / (shot_counts[-1] - shot_counts[-2])
                
                if late_improvement < early_improvement * 0.5:
                    report_lines.append("  ⚠️  Diminishing returns observed with higher shot counts")
                else:
                    report_lines.append("  ✅ Consistent improvement across shot counts")
        
        # Recovery rate analysis
        recovery_rates = [r.recovery_rate for r in self.results if r.recovery_rate > 0]
        if recovery_rates:
            avg_recovery = np.mean(recovery_rates)
            report_lines.append(f"  Average recovery rate: {avg_recovery:.1f}%")
            
            if avg_recovery > 70:
                report_lines.append("  🎯 Excellent adaptation capability")
            elif avg_recovery > 50:
                report_lines.append("  ✅ Good adaptation capability")
            else:
                report_lines.append("  ⚠️  Limited adaptation capability")
        
        # Recommendations
        report_lines.append(f"\nRECOMMENDATIONS:")
        
        # Find sweet spot
        if len(f1_scores) >= 3:
            # Find point of diminishing returns
            for i in range(1, len(f1_scores)):
                improvement = f1_scores[i] - f1_scores[i-1]
                if improvement < 0.01:  # Less than 1% improvement
                    sweet_spot = shot_counts[i-1]
                    report_lines.append(f"  🎯 Sweet spot: {sweet_spot} shots (diminishing returns after)")
                    break
            else:
                report_lines.append(f"  📈 Continue increasing shots for better performance")
        
        # Cost-benefit analysis
        if len(f1_scores) >= 2:
            cost_benefit = (f1_scores[-1] - f1_scores[0]) / (shot_counts[-1] - shot_counts[0])
            report_lines.append(f"  Cost-benefit ratio: {cost_benefit:.6f} F1 improvement per shot")
        
        full_report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(full_report)
            print(f"Few-shot adaptation report saved to: {output_path}")
        
        return full_report
    
    def _clone_model(self, model_dict: Dict) -> Dict:
        """Create a copy of the model for adaptation."""
        import copy
        # Use deep copy to avoid modifying the original model
        cloned_model = copy.deepcopy(model_dict)
        
        # Move all models to device
        for key, model in cloned_model.items():
            cloned_model[key] = model.to(self.device)
        
        return cloned_model
    
    def _freeze_encoders(self, model_dict: Dict):
        """Freeze encoder parameters during adaptation."""
        for key in ['audio_encoder', 'text_encoder', 'cross', 'pool_a', 'pool_t']:
            if key in model_dict:
                for param in model_dict[key].parameters():
                    param.requires_grad = False

def create_few_shot_adapter(base_model: Dict, device: str) -> FewShotAdapter:
    """Create a few-shot adapter instance."""
    return FewShotAdapter(base_model, device)
