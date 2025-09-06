#!/usr/bin/env python3
"""
Cross-lingual transfer ratio and per-language performance metrics.
Implements F1target/F1source ratio and comprehensive language-specific evaluation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score, classification_report
import json
import os

@dataclass
class LanguagePerformance:
    """Performance metrics for a specific language."""
    language: str
    f1_score: float
    accuracy: float
    precision: float
    recall: float
    support: int
    confusion_matrix: np.ndarray
    per_class_f1: Dict[int, float]

@dataclass
class CrossLingualTransferMetrics:
    """Cross-lingual transfer performance metrics."""
    source_language: str
    target_languages: List[str]
    transfer_ratios: Dict[str, float]  # F1target/F1source
    source_performance: LanguagePerformance
    target_performances: Dict[str, LanguagePerformance]
    overall_transfer_ratio: float  # Average across all targets

class CrossLingualEvaluator:
    """Evaluates cross-lingual transfer performance."""
    
    def __init__(self, language_detector=None):
        self.language_detector = language_detector
        self.language_performances = {}
        
    def detect_language(self, text: str) -> str:
        """Detect language of text input."""
        if self.language_detector:
            try:
                return self.language_detector.detect(text)
            except:
                pass
        
        # Fallback: simple heuristic based on common words
        text_lower = text.lower()
        if any(word in text_lower for word in ['the', 'and', 'is', 'in', 'to']):
            return 'en'
        elif any(word in text_lower for word in ['le', 'la', 'de', 'et', 'est']):
            return 'fr'
        elif any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist']):
            return 'de'
        elif any(word in text_lower for word in ['el', 'la', 'de', 'y', 'es']):
            return 'es'
        else:
            return 'en'  # Default to English
    
    def evaluate_per_language(self, 
                            predictions: np.ndarray, 
                            labels: np.ndarray, 
                            texts: List[str],
                            language_mapping: Optional[Dict[str, str]] = None) -> Dict[str, LanguagePerformance]:
        """Evaluate performance for each detected language."""
        
        # Detect languages for each text
        detected_languages = []
        for text in texts:
            lang = self.detect_language(text)
            if language_mapping:
                lang = language_mapping.get(lang, lang)
            detected_languages.append(lang)
        
        # Group by language
        language_groups = {}
        for i, lang in enumerate(detected_languages):
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(i)
        
        # Convert to numpy arrays for proper indexing
        predictions_array = np.array(predictions)
        labels_array = np.array(labels)
        
        # Calculate performance for each language
        for lang, indices in language_groups.items():
            if len(indices) < 5:  # Skip languages with too few samples
                continue
                
            lang_preds = predictions_array[indices]
            lang_labels = labels_array[indices]
            
            # Calculate metrics
            f1 = f1_score(lang_labels, lang_preds, average='weighted', zero_division=0)
            accuracy = (lang_preds == lang_labels).mean()
            
            # Per-class metrics
            report = classification_report(lang_labels, lang_preds, 
                                        output_dict=True, zero_division=0)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(lang_labels, lang_preds)
            
            # Per-class F1
            per_class_f1 = {}
            for class_id in range(len(cm)):
                if class_id in report:
                    per_class_f1[class_id] = report[str(class_id)]['f1-score']
            
            self.language_performances[lang] = LanguagePerformance(
                language=lang,
                f1_score=f1,
                accuracy=accuracy,
                precision=report['weighted avg']['precision'],
                recall=report['weighted avg']['recall'],
                support=len(indices),
                confusion_matrix=cm,
                per_class_f1=per_class_f1
            )
        
        return self.language_performances
    
    def calculate_transfer_ratios(self, 
                                source_language: str = 'en',
                                target_languages: Optional[List[str]] = None) -> CrossLingualTransferMetrics:
        """Calculate cross-lingual transfer ratios."""
        
        if source_language not in self.language_performances:
            raise ValueError(f"Source language '{source_language}' not found in evaluations")
        
        source_perf = self.language_performances[source_language]
        
        if target_languages is None:
            target_languages = [lang for lang in self.language_performances.keys() 
                              if lang != source_language]
        
        transfer_ratios = {}
        target_performances = {}
        
        for target_lang in target_languages:
            if target_lang in self.language_performances:
                target_perf = self.language_performances[target_lang]
                target_performances[target_lang] = target_perf
                
                # Calculate transfer ratio: F1target/F1source
                if source_perf.f1_score > 0:
                    ratio = target_perf.f1_score / source_perf.f1_score
                else:
                    ratio = 0.0
                transfer_ratios[target_lang] = ratio
        
        # Overall transfer ratio (average across targets)
        if transfer_ratios:
            overall_ratio = np.mean(list(transfer_ratios.values()))
        else:
            overall_ratio = 0.0
        
        return CrossLingualTransferMetrics(
            source_language=source_language,
            target_languages=target_languages,
            transfer_ratios=transfer_ratios,
            source_performance=source_perf,
            target_performances=target_performances,
            overall_transfer_ratio=overall_ratio
        )
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive cross-lingual performance report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CROSS-LINGUAL PERFORMANCE EVALUATION REPORT")
        report_lines.append("=" * 80)
        
        # Language-specific performance
        report_lines.append("\nPER-LANGUAGE PERFORMANCE:")
        report_lines.append("-" * 50)
        
        for lang, perf in sorted(self.language_performances.items()):
            report_lines.append(f"\n{lang.upper()}:")
            report_lines.append(f"  F1 Score: {perf.f1_score:.4f}")
            report_lines.append(f"  Accuracy: {perf.accuracy:.4f}")
            report_lines.append(f"  Precision: {perf.precision:.4f}")
            report_lines.append(f"  Recall: {perf.recall:.4f}")
            report_lines.append(f"  Support: {perf.support}")
            
            # Per-class F1
            if perf.per_class_f1:
                report_lines.append("  Per-class F1:")
                emotion_names = ['angry', 'happy', 'sad', 'neutral']
                for class_id, f1 in perf.per_class_f1.items():
                    if class_id < len(emotion_names):
                        report_lines.append(f"    {emotion_names[class_id]}: {f1:.4f}")
        
        # Transfer ratios
        try:
            transfer_metrics = self.calculate_transfer_ratios()
            report_lines.append(f"\nCROSS-LINGUAL TRANSFER RATIOS:")
            report_lines.append("-" * 50)
            report_lines.append(f"Source Language: {transfer_metrics.source_language.upper()}")
            report_lines.append(f"Source F1: {transfer_metrics.source_performance.f1_score:.4f}")
            report_lines.append(f"\nTransfer Ratios (F1target/F1source):")
            
            for target_lang, ratio in transfer_metrics.transfer_ratios.items():
                report_lines.append(f"  {target_lang.upper()}: {ratio:.4f}")
            
            report_lines.append(f"\nOverall Transfer Ratio: {transfer_metrics.overall_transfer_ratio:.4f}")
            
        except Exception as e:
            report_lines.append(f"\nTransfer ratio calculation failed: {e}")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Cross-lingual report saved to: {output_path}")
        
        return report

def create_cross_lingual_evaluator():
    """Create a cross-lingual evaluator instance."""
    try:
        from langdetect import detect
        language_detector = detect
    except ImportError:
        print("Warning: langdetect not available, using fallback language detection")
        language_detector = None
    
    return CrossLingualEvaluator(language_detector)
