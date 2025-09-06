#!/usr/bin/env python3
"""
Robustness evaluation for noise, SNR degradation, and code-mixing scenarios.
Implements comprehensive robustness testing as per academic requirements.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import json
from datetime import datetime

from models import AudioEncoder, TextEncoder, FusionLayer, Classifier
from models.classifier import AdvancedOpenMaxClassifier
from models.cross_attention import CrossModalAttention
from models.pooling import AttentiveStatsPooling
from models.prototypes import PrototypeMemory
from data.dataset import SERDataset
from utils import weighted_f1, energy_score
from data.preprocess import add_noise_snr

@dataclass
class RobustnessResult:
    """Results from robustness evaluation."""
    condition: str
    snr_level: Optional[float]
    noise_type: Optional[str]
    code_mixing_ratio: Optional[float]
    f1_score: float
    accuracy: float
    degradation: float  # Performance drop from baseline
    ood_detection_rate: float  # How often OOD detection triggers

class RobustnessEvaluator:
    """Evaluates model robustness under various conditions."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.baseline_performance = None
        self.results = []
        
    def set_baseline_performance(self, baseline_f1: float, baseline_accuracy: float):
        """Set baseline performance for degradation calculation."""
        self.baseline_performance = {
            'f1': baseline_f1,
            'accuracy': baseline_accuracy
        }
    
    def add_noise_to_audio(self, audio: torch.Tensor, snr_db: float, noise_type: str = 'gaussian') -> torch.Tensor:
        """Add noise to audio at specified SNR level."""
        
        if noise_type == 'gaussian':
            # Add Gaussian white noise
            signal_power = torch.mean(audio ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(audio) * torch.sqrt(noise_power)
            noisy_audio = audio + noise
            
        elif noise_type == 'babble':
            # Add babble noise (simulated with multiple sine waves)
            t = torch.arange(audio.shape[-1]) / 16000  # Assuming 16kHz
            babble_noise = torch.zeros_like(audio)
            for freq in [100, 200, 300, 400, 500]:  # Multiple frequencies
                babble_noise += 0.1 * torch.sin(2 * np.pi * freq * t)
            
            # Scale to achieve target SNR
            signal_power = torch.mean(audio ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            babble_noise = babble_noise * torch.sqrt(noise_power / torch.mean(babble_noise ** 2))
            noisy_audio = audio + babble_noise
            
        elif noise_type == 'music':
            # Add music-like noise (chord progression)
            t = torch.arange(audio.shape[-1]) / 16000
            music_noise = torch.zeros_like(audio)
            # Add chord frequencies
            chord_freqs = [261.63, 329.63, 392.00]  # C major chord
            for freq in chord_freqs:
                music_noise += 0.05 * torch.sin(2 * np.pi * freq * t)
            
            # Scale to achieve target SNR
            signal_power = torch.mean(audio ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            music_noise = music_noise * torch.sqrt(noise_power / torch.mean(music_noise ** 2))
            noisy_audio = audio + music_noise
            
        else:
            # Default to Gaussian noise
            noisy_audio = self.add_noise_to_audio(audio, snr_db, 'gaussian')
        
        return noisy_audio
    
    def create_code_mixed_text(self, text: str, mixing_ratio: float, target_language: str = 'hi') -> str:
        """Create code-mixed text by replacing words with target language equivalents."""
        
        # Simple code-mixing simulation
        # In practice, you'd use a proper code-mixing dataset or translation model
        
        if mixing_ratio == 0.0:
            return text
        
        # Common Hindi-English code-mixing patterns
        hindi_equivalents = {
            'the': 'yeh',
            'is': 'hai',
            'and': 'aur',
            'in': 'mein',
            'to': 'ko',
            'of': 'ka',
            'a': 'ek',
            'that': 'woh',
            'it': 'yeh',
            'with': 'ke saath',
            'for': 'ke liye',
            'on': 'par',
            'at': 'par',
            'by': 'se',
            'from': 'se',
            'up': 'upar',
            'down': 'neeche',
            'good': 'accha',
            'bad': 'bura',
            'big': 'bada',
            'small': 'chota'
        }
        
        words = text.split()
        num_words_to_replace = int(len(words) * mixing_ratio)
        
        if num_words_to_replace == 0:
            return text
        
        # Randomly select words to replace
        indices_to_replace = random.sample(range(len(words)), num_words_to_replace)
        
        mixed_text = words.copy()
        for idx in indices_to_replace:
            word = words[idx].lower()
            if word in hindi_equivalents:
                mixed_text[idx] = hindi_equivalents[word]
        
        return ' '.join(mixed_text)
    
    def evaluate_noise_robustness(self, test_loader: DataLoader, 
                                snr_levels: List[float] = [20, 15, 10, 5, 0, -5],
                                noise_types: List[str] = ['gaussian', 'babble', 'music']) -> List[RobustnessResult]:
        """Evaluate model robustness under different noise conditions."""
        
        print(f"🔊 Testing Noise Robustness")
        print(f"SNR Levels: {snr_levels} dB")
        print(f"Noise Types: {noise_types}")
        
        results = []
        
        for noise_type in noise_types:
            print(f"\n📊 Testing {noise_type} noise...")
            
            for snr_db in snr_levels:
                print(f"  Testing SNR: {snr_db} dB")
                
                # Create noisy test loader
                noisy_loader = self._create_noisy_loader(test_loader, snr_db, noise_type)
                
                # Evaluate performance
                f1_score, accuracy, ood_rate = self._evaluate_robustness(noisy_loader)
                
                # Calculate degradation
                degradation = 0.0
                if self.baseline_performance:
                    degradation = self.baseline_performance['f1'] - f1_score
                
                # Store results
                result = RobustnessResult(
                    condition=f"{noise_type}_noise",
                    snr_level=snr_db,
                    noise_type=noise_type,
                    code_mixing_ratio=None,
                    f1_score=f1_score,
                    accuracy=accuracy,
                    degradation=degradation,
                    ood_detection_rate=ood_rate
                )
                
                results.append(result)
                
                print(f"    ✅ F1: {f1_score:.4f}, Degradation: {degradation:.4f}")
                print(f"    🎯 OOD Detection Rate: {ood_rate:.2f}")
        
        return results
    
    def evaluate_code_mixing_robustness(self, test_loader: DataLoader,
                                      mixing_ratios: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                      target_languages: List[str] = ['hi', 'bn']) -> List[RobustnessResult]:
        """Evaluate model robustness under code-mixing scenarios."""
        
        print(f"🌍 Testing Code-Mixing Robustness")
        print(f"Mixing Ratios: {mixing_ratios}")
        print(f"Target Languages: {target_languages}")
        
        results = []
        
        for target_lang in target_languages:
            print(f"\n📊 Testing {target_lang.upper()} code-mixing...")
            
            for mixing_ratio in mixing_ratios:
                print(f"  Testing mixing ratio: {mixing_ratio:.1f}")
                
                # Create code-mixed test loader
                mixed_loader = self._create_code_mixed_loader(test_loader, mixing_ratio, target_lang)
                
                # Evaluate performance
                f1_score, accuracy, ood_rate = self._evaluate_robustness(mixed_loader)
                
                # Calculate degradation
                degradation = 0.0
                if self.baseline_performance:
                    degradation = self.baseline_performance['f1'] - f1_score
                
                # Store results
                result = RobustnessResult(
                    condition=f"{target_lang}_code_mixing",
                    snr_level=None,
                    noise_type=None,
                    code_mixing_ratio=mixing_ratio,
                    f1_score=f1_score,
                    accuracy=accuracy,
                    degradation=degradation,
                    ood_detection_rate=ood_rate
                )
                
                results.append(result)
                
                print(f"    ✅ F1: {f1_score:.4f}, Degradation: {degradation:.4f}")
                print(f"    🎯 OOD Detection Rate: {ood_rate:.2f}")
        
        return results
    
    def _create_noisy_loader(self, test_loader: DataLoader, snr_db: float, noise_type: str) -> DataLoader:
        """Create a data loader with noisy audio."""
        
        class NoisyDataset:
            def __init__(self, original_loader, snr_db, noise_type):
                self.original_loader = original_loader
                self.snr_db = snr_db
                self.noise_type = noise_type
                self.evaluator = None  # Will be set later
            
            def __iter__(self):
                for batch in self.original_loader:
                    audio_list, text_list, labels = batch
                    
                    # Add noise to audio
                    noisy_audio_list = []
                    for audio in audio_list:
                        if self.evaluator:
                            noisy_audio = self.evaluator.add_noise_to_audio(audio, self.snr_db, self.noise_type)
                        else:
                            # Fallback: simple noise addition
                            signal_power = torch.mean(audio ** 2)
                            noise_power = signal_power / (10 ** (self.snr_db / 10))
                            noise = torch.randn_like(audio) * torch.sqrt(noise_power)
                            noisy_audio = audio + noise
                        
                        noisy_audio_list.append(noisy_audio)
                    
                    yield noisy_audio_list, text_list, labels
        
        noisy_dataset = NoisyDataset(test_loader, snr_db, noise_type)
        noisy_dataset.evaluator = self
        
        return noisy_dataset
    
    def _create_code_mixed_loader(self, test_loader: DataLoader, mixing_ratio: float, target_language: str) -> DataLoader:
        """Create a data loader with code-mixed text."""
        
        class CodeMixedDataset:
            def __init__(self, original_loader, mixing_ratio, target_language):
                self.original_loader = original_loader
                self.mixing_ratio = mixing_ratio
                self.target_language = target_language
                self.evaluator = None  # Will be set later
            
            def __iter__(self):
                for batch in self.original_loader:
                    audio_list, text_list, labels = batch
                    
                    # Create code-mixed text
                    mixed_text_list = []
                    for text in text_list:
                        if self.evaluator:
                            mixed_text = self.evaluator.create_code_mixed_text(text, self.mixing_ratio, self.target_language)
                        else:
                            mixed_text = text  # Fallback
                        mixed_text_list.append(mixed_text)
                    
                    yield audio_list, mixed_text_list, labels
        
        mixed_dataset = CodeMixedDataset(test_loader, mixing_ratio, target_language)
        mixed_dataset.evaluator = self
        
        return mixed_dataset
    
    def _evaluate_robustness(self, test_loader) -> Tuple[float, float, float]:
        """Evaluate model performance on modified test data."""
        
        self.model['fusion'].eval()
        self.model['classifier'].eval()
        
        all_preds, all_labels = [], []
        ood_detections = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                audio_list, text_list, labels = batch
                labels = labels.to(self.device)
                
                # Forward pass
                a_seq, a_mask = self.model['audio_encoder'](audio_list, text_list)
                t_seq, t_mask = self.model['text_encoder'](text_list)
                a_enh, t_enh = self.model['cross'](a_seq, t_seq, a_mask, t_mask)
                a_vec = self.model['pool_a'](a_enh, a_mask)
                t_vec = self.model['pool_t'](t_enh, t_mask)
                fused = self.model['fusion'](a_vec, t_vec)
                
                # Get predictions
                logits = self.model['classifier'](fused, use_openmax=True)
                # Simulate uncertainty for OOD detection
                probs = torch.softmax(logits, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                uncertainty = 1.0 - max_probs  # Use 1 - max_prob as uncertainty
                
                # Check for OOD detection (high uncertainty)
                ood_mask = max_probs < 0.5  # Low confidence indicates OOD
                ood_detections += ood_mask.sum().item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_samples += len(labels)
        
        # Calculate metrics
        f1_score = weighted_f1(torch.tensor(all_preds), torch.tensor(all_labels))
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        ood_rate = ood_detections / max(total_samples, 1)
        
        return f1_score, accuracy, ood_rate
    
    def generate_robustness_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive robustness evaluation report."""
        
        if not self.results:
            return "No results available. Run evaluation first."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL ROBUSTNESS EVALUATION REPORT")
        report_lines.append("=" * 80)
        
        # Noise robustness results
        noise_results = [r for r in self.results if r.condition.endswith('_noise')]
        if noise_results:
            report_lines.append(f"\nNOISE ROBUSTNESS ANALYSIS:")
            report_lines.append("-" * 50)
            
            # Group by noise type
            noise_types = list(set(r.noise_type for r in noise_results))
            for noise_type in noise_types:
                type_results = [r for r in noise_results if r.noise_type == noise_type]
                type_results.sort(key=lambda x: x.snr_level or 0)
                
                report_lines.append(f"\n{noise_type.upper()} Noise:")
                report_lines.append(f"{'SNR (dB)':<10} {'F1':<8} {'Degradation':<12} {'OOD Rate':<10}")
                report_lines.append("-" * 45)
                
                for result in type_results:
                    report_lines.append(f"{result.snr_level:<10} {result.f1_score:<8.4f} {result.degradation:<12.4f} {result.ood_detection_rate:<10.2f}")
                
                # Analyze degradation pattern
                if len(type_results) >= 2:
                    snr_levels = [r.snr_level for r in type_results if r.snr_level is not None]
                    degradations = [r.degradation for r in type_results]
                    
                    # Calculate degradation rate
                    if len(snr_levels) > 1:
                        degradation_rate = (degradations[-1] - degradations[0]) / (snr_levels[-1] - snr_levels[0])
                        report_lines.append(f"  Degradation rate: {degradation_rate:.4f} F1 per dB")
                        
                        if degradation_rate > 0.05:
                            report_lines.append("  ⚠️  High sensitivity to noise")
                        elif degradation_rate < 0.02:
                            report_lines.append("  ✅ Good noise robustness")
                        else:
                            report_lines.append("  ⚠️  Moderate noise sensitivity")
        
        # Code-mixing robustness results
        code_mixing_results = [r for r in self.results if 'code_mixing' in r.condition]
        if code_mixing_results:
            report_lines.append(f"\nCODE-MIXING ROBUSTNESS ANALYSIS:")
            report_lines.append("-" * 50)
            
            # Group by target language
            target_langs = list(set(r.condition.split('_')[0] for r in code_mixing_results))
            for target_lang in target_langs:
                lang_results = [r for r in code_mixing_results if r.condition.startswith(target_lang)]
                lang_results.sort(key=lambda x: x.code_mixing_ratio or 0)
                
                report_lines.append(f"\n{target_lang.upper()} Code-Mixing:")
                report_lines.append(f"{'Mixing Ratio':<12} {'F1':<8} {'Degradation':<12} {'OOD Rate':<10}")
                report_lines.append("-" * 45)
                
                for result in lang_results:
                    report_lines.append(f"{result.code_mixing_ratio:<12.1f} {result.f1_score:<8.4f} {result.degradation:<12.4f} {result.ood_detection_rate:<10.2f}")
                
                # Analyze code-mixing impact
                if len(lang_results) >= 2:
                    mixing_ratios = [r.code_mixing_ratio for r in lang_results if r.code_mixing_ratio is not None]
                    degradations = [r.degradation for r in lang_results]
                    
                    # Calculate degradation rate
                    if len(mixing_ratios) > 1:
                        degradation_rate = (degradations[-1] - degradations[0]) / (mixing_ratios[-1] - mixing_ratios[0])
                        report_lines.append(f"  Degradation rate: {degradation_rate:.4f} F1 per mixing ratio")
                        
                        if degradation_rate > 0.1:
                            report_lines.append("  ⚠️  High sensitivity to code-mixing")
                        elif degradation_rate < 0.05:
                            report_lines.append("  ✅ Good code-mixing robustness")
                        else:
                            report_lines.append("  ⚠️  Moderate code-mixing sensitivity")
        
        # Overall robustness assessment
        report_lines.append(f"\nOVERALL ROBUSTNESS ASSESSMENT:")
        report_lines.append("-" * 50)
        
        if self.baseline_performance:
            report_lines.append(f"Baseline F1: {self.baseline_performance['f1']:.4f}")
            report_lines.append(f"Baseline Accuracy: {self.baseline_performance['accuracy']:.4f}")
        
        # Calculate average degradation
        avg_degradation = np.mean([r.degradation for r in self.results])
        report_lines.append(f"Average Performance Degradation: {avg_degradation:.4f}")
        
        # Robustness score (inverse of degradation)
        if avg_degradation > 0:
            robustness_score = max(0, 1 - avg_degradation)
            report_lines.append(f"Robustness Score: {robustness_score:.4f}")
            
            if robustness_score > 0.8:
                report_lines.append("  🎯 Excellent robustness")
            elif robustness_score > 0.6:
                report_lines.append("  ✅ Good robustness")
            elif robustness_score > 0.4:
                report_lines.append("  ⚠️  Moderate robustness")
            else:
                report_lines.append("  ❌ Poor robustness")
        
        # OOD detection effectiveness
        avg_ood_rate = np.mean([r.ood_detection_rate for r in self.results])
        report_lines.append(f"Average OOD Detection Rate: {avg_ood_rate:.4f}")
        
        if avg_ood_rate > 0.3:
            report_lines.append("  🎯 Effective OOD detection")
        elif avg_ood_rate > 0.1:
            report_lines.append("  ✅ Good OOD detection")
        else:
            report_lines.append("  ⚠️  Limited OOD detection")
        
        # Recommendations
        report_lines.append(f"\nRECOMMENDATIONS:")
        
        if avg_degradation > 0.1:
            report_lines.append("  🔧 Consider noise augmentation during training")
            report_lines.append("  🔧 Implement more robust audio preprocessing")
        
        if avg_ood_rate < 0.2:
            report_lines.append("  🎯 Improve OOD detection thresholds")
            report_lines.append("  🎯 Add more diverse training data")
        
        # Find most challenging conditions
        worst_conditions = sorted(self.results, key=lambda x: x.degradation, reverse=True)[:3]
        report_lines.append(f"\nMost Challenging Conditions:")
        for i, result in enumerate(worst_conditions, 1):
            condition_desc = f"{result.condition}"
            if result.snr_level is not None:
                condition_desc += f" at {result.snr_level}dB SNR"
            if result.code_mixing_ratio is not None:
                condition_desc += f" with {result.code_mixing_ratio:.1f} mixing ratio"
            
            report_lines.append(f"  {i}. {condition_desc}: {result.degradation:.4f} degradation")
        
        full_report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(full_report)
            print(f"Robustness evaluation report saved to: {output_path}")
        
        return full_report

def create_robustness_evaluator(model: Dict, device: str) -> RobustnessEvaluator:
    """Create a robustness evaluator instance."""
    return RobustnessEvaluator(model, device)
