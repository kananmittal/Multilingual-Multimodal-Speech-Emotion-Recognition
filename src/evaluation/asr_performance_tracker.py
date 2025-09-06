#!/usr/bin/env python3
"""
ASR performance tracking per language.
Implements WER calculation, confidence tracking, and language-specific metrics.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import os
from datetime import datetime

@dataclass
class ASRMetrics:
    """ASR performance metrics for a specific language."""
    language: str
    total_samples: int
    total_words: int
    total_errors: int
    wer: float
    confidence_mean: float
    confidence_std: float
    word_level_confidence: List[float]
    segment_level_confidence: List[float]
    error_types: Dict[str, int]  # substitution, deletion, insertion
    processing_time_mean: float
    processing_time_std: float

@dataclass
class ASRPerformanceReport:
    """Comprehensive ASR performance report."""
    overall_wer: float
    language_specific_wer: Dict[str, float]
    confidence_correlation: float
    processing_efficiency: Dict[str, float]
    error_analysis: Dict[str, Dict[str, int]]
    recommendations: List[str]

class ASRPerformanceTracker:
    """Tracks ASR performance metrics per language."""
    
    def __init__(self):
        self.language_metrics = defaultdict(lambda: {
            'total_samples': 0,
            'total_words': 0,
            'total_errors': 0,
            'confidences': [],
            'processing_times': [],
            'error_types': defaultdict(int)
        })
        self.detected_languages = set()
        
    def detect_language(self, text: str) -> str:
        """Detect language of text input."""
        # Simple heuristic-based language detection
        text_lower = text.lower()
        
        # English indicators
        if any(word in text_lower for word in ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']):
            return 'en'
        
        # French indicators
        elif any(word in text_lower for word in ['le', 'la', 'de', 'et', 'est', 'un', 'une', 'dans', 'sur', 'avec']):
            return 'fr'
        
        # German indicators
        elif any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'von', 'mit', 'auf']):
            return 'de'
        
        # Spanish indicators
        elif any(word in text_lower for word in ['el', 'la', 'de', 'y', 'es', 'en', 'un', 'una', 'con', 'por']):
            return 'es'
        
        # Italian indicators
        elif any(word in text_lower for word in ['il', 'la', 'di', 'e', 'è', 'in', 'un', 'una', 'con', 'per']):
            return 'it'
        
        else:
            return 'en'  # Default to English
    
    def calculate_wer(self, reference: str, hypothesis: str) -> Tuple[float, Dict[str, int]]:
        """Calculate Word Error Rate using Levenshtein distance."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Dynamic programming for edit distance
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        
        # Backtrack to find error types
        errors = {'substitution': 0, 'deletion': 0, 'insertion': 0}
        i, j = m, n
        
        while i > 0 and j > 0:
            if ref_words[i - 1] == hyp_words[j - 1]:
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                errors['deletion'] += 1
                i -= 1
            elif dp[i][j] == dp[i][j - 1] + 1:
                errors['insertion'] += 1
                j -= 1
            else:
                errors['substitution'] += 1
                i -= 1
                j -= 1
        
        # Handle remaining insertions/deletions
        while i > 0:
            errors['deletion'] += 1
            i -= 1
        while j > 0:
            errors['insertion'] += 1
            j -= 1
        
        total_errors = dp[m][n]
        wer = total_errors / max(len(ref_words), 1)
        
        return wer, errors
    
    def update_metrics(self, 
                      reference_text: str, 
                      hypothesis_text: str, 
                      confidence_scores: List[float],
                      processing_time: float,
                      detected_language: Optional[str] = None) -> None:
        """Update ASR performance metrics."""
        
        # Detect language if not provided
        if detected_language is None:
            detected_language = self.detect_language(reference_text)
        
        self.detected_languages.add(detected_language)
        
        # Calculate WER and error types
        wer, error_types = self.calculate_wer(reference_text, hypothesis_text)
        
        # Update language-specific metrics
        metrics = self.language_metrics[detected_language]
        metrics['total_samples'] += 1
        metrics['total_words'] += len(reference_text.split())
        metrics['total_errors'] += int(wer * len(reference_text.split()))
        
        # Update confidence scores
        if confidence_scores:
            metrics['confidences'].extend(confidence_scores)
        
        # Update processing time
        metrics['processing_times'].append(processing_time)
        
        # Update error types
        for error_type, count in error_types.items():
            metrics['error_types'][error_type] += count
    
    def get_language_metrics(self, language: str) -> Optional[ASRMetrics]:
        """Get ASR metrics for a specific language."""
        if language not in self.language_metrics:
            return None
        
        metrics = self.language_metrics[language]
        
        if metrics['total_samples'] == 0:
            return None
        
        # Calculate WER
        wer = metrics['total_errors'] / max(metrics['total_words'], 1)
        
        # Calculate confidence statistics
        confidences = metrics['confidences']
        confidence_mean = np.mean(confidences) if confidences else 0.0
        confidence_std = np.std(confidences) if confidences else 0.0
        
        # Calculate processing time statistics
        processing_times = metrics['processing_times']
        processing_time_mean = np.mean(processing_times) if processing_times else 0.0
        processing_time_std = np.std(processing_times) if processing_times else 0.0
        
        return ASRMetrics(
            language=language,
            total_samples=metrics['total_samples'],
            total_words=metrics['total_words'],
            total_errors=metrics['total_errors'],
            wer=wer,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            word_level_confidence=metrics['confidences'],
            segment_level_confidence=[],  # Placeholder for segment-level confidence
            error_types=dict(metrics['error_types']),
            processing_time_mean=processing_time_mean,
            processing_time_std=processing_time_std
        )
    
    def generate_performance_report(self) -> ASRPerformanceReport:
        """Generate comprehensive ASR performance report."""
        
        # Calculate overall WER
        total_words = sum(metrics['total_words'] for metrics in self.language_metrics.values())
        total_errors = sum(metrics['total_errors'] for metrics in self.language_metrics.values())
        overall_wer = total_errors / max(total_words, 1)
        
        # Language-specific WER
        language_specific_wer = {}
        for language in self.detected_languages:
            metrics = self.get_language_metrics(language)
            if metrics:
                language_specific_wer[language] = metrics.wer
        
        # Confidence correlation analysis
        all_confidences = []
        all_wers = []
        for language in self.detected_languages:
            metrics = self.get_language_metrics(language)
            if metrics and metrics.word_level_confidence:
                all_confidences.extend(metrics.word_level_confidence)
                all_wers.extend([metrics.wer] * len(metrics.word_level_confidence))
        
        confidence_correlation = 0.0
        if len(all_confidences) > 1:
            confidence_correlation = np.corrcoef(all_confidences, all_wers)[0, 1]
            if np.isnan(confidence_correlation):
                confidence_correlation = 0.0
        
        # Processing efficiency
        processing_efficiency = {}
        for language in self.detected_languages:
            metrics = self.get_language_metrics(language)
            if metrics:
                processing_efficiency[language] = {
                    'avg_time_per_word': metrics.processing_time_mean / max(metrics.total_words, 1),
                    'words_per_second': metrics.total_words / max(metrics.processing_time_mean, 1)
                }
        
        # Error analysis
        error_analysis = {}
        for language in self.detected_languages:
            metrics = self.get_language_metrics(language)
            if metrics:
                error_analysis[language] = dict(metrics.error_types)
        
        # Generate recommendations
        recommendations = []
        
        if overall_wer > 0.15:
            recommendations.append("Overall WER is high (>15%). Consider improving audio quality or model fine-tuning.")
        
        # Language-specific recommendations
        for language, wer in language_specific_wer.items():
            if wer > 0.2:
                recommendations.append(f"WER for {language.upper()} is very high ({wer:.1%}). Language-specific training may be needed.")
            elif wer < 0.05:
                recommendations.append(f"Excellent performance for {language.upper()} ({wer:.1%}). Consider using as reference for other languages.")
        
        if confidence_correlation < -0.3:
            recommendations.append("Low confidence-WER correlation. Model may be overconfident in predictions.")
        
        return ASRPerformanceReport(
            overall_wer=overall_wer,
            language_specific_wer=language_specific_wer,
            confidence_correlation=confidence_correlation,
            processing_efficiency=processing_efficiency,
            error_analysis=error_analysis,
            recommendations=recommendations
        )
    
    def print_report(self, output_path: Optional[str] = None) -> str:
        """Print comprehensive ASR performance report."""
        
        report = self.generate_performance_report()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ASR PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        
        # Overall performance
        report_lines.append(f"\nOVERALL PERFORMANCE:")
        report_lines.append(f"  Total Words Processed: {total_words:,}")
        report_lines.append(f"  Total Errors: {total_errors:,}")
        report_lines.append(f"  Overall WER: {report.overall_wer:.2%}")
        
        # Language-specific performance
        report_lines.append(f"\nLANGUAGE-SPECIFIC PERFORMANCE:")
        for language, wer in sorted(report.language_specific_wer.items()):
            metrics = self.get_language_metrics(language)
            if metrics:
                report_lines.append(f"\n  {language.upper()}:")
                report_lines.append(f"    WER: {wer:.2%}")
                report_lines.append(f"    Samples: {metrics.total_samples}")
                report_lines.append(f"    Words: {metrics.total_words}")
                report_lines.append(f"    Confidence: {metrics.confidence_mean:.3f} ± {metrics.confidence_std:.3f}")
                report_lines.append(f"    Processing Time: {metrics.processing_time_mean:.3f}s ± {metrics.processing_time_std:.3f}s")
                
                if metrics.error_types:
                    report_lines.append(f"    Error Types:")
                    for error_type, count in metrics.error_types.items():
                        report_lines.append(f"      {error_type}: {count}")
        
        # Processing efficiency
        report_lines.append(f"\nPROCESSING EFFICIENCY:")
        for language, efficiency in report.processing_efficiency.items():
            report_lines.append(f"  {language.upper()}:")
            report_lines.append(f"    Avg Time per Word: {efficiency['avg_time_per_word']:.4f}s")
            report_lines.append(f"    Words per Second: {efficiency['words_per_second']:.2f}")
        
        # Confidence analysis
        report_lines.append(f"\nCONFIDENCE ANALYSIS:")
        report_lines.append(f"  Confidence-WER Correlation: {report.confidence_correlation:.3f}")
        if report.confidence_correlation < -0.3:
            report_lines.append("  ⚠️  Low correlation suggests overconfidence issues")
        elif report.confidence_correlation > -0.1:
            report_lines.append("  ✅ Good correlation between confidence and performance")
        
        # Recommendations
        if report.recommendations:
            report_lines.append(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                report_lines.append(f"  {i}. {rec}")
        
        # Save report
        full_report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(full_report)
            print(f"ASR performance report saved to: {output_path}")
        
        return full_report
    
    def save_metrics(self, output_path: str) -> None:
        """Save all metrics to JSON file."""
        metrics_data = {}
        
        for language, metrics in self.language_metrics.items():
            metrics_data[language] = {
                'total_samples': metrics['total_samples'],
                'total_words': metrics['total_words'],
                'total_errors': metrics['total_errors'],
                'confidence_stats': {
                    'mean': np.mean(metrics['confidences']) if metrics['confidences'] else 0.0,
                    'std': np.std(metrics['confidences']) if metrics['confidences'] else 0.0,
                    'min': np.min(metrics['confidences']) if metrics['confidences'] else 0.0,
                    'max': np.max(metrics['confidences']) if metrics['confidences'] else 0.0
                },
                'processing_time_stats': {
                    'mean': np.mean(metrics['processing_times']) if metrics['processing_times'] else 0.0,
                    'std': np.std(metrics['processing_times']) if metrics['processing_times'] else 0.0
                },
                'error_types': dict(metrics['error_types'])
            }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"ASR metrics saved to: {output_path}")

def create_asr_performance_tracker() -> ASRPerformanceTracker:
    """Create an ASR performance tracker instance."""
    return ASRPerformanceTracker()
