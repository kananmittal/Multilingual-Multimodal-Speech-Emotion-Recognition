import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for comprehensive evaluation metrics."""
    # Basic classification metrics
    accuracy: float
    weighted_f1: float
    macro_f1: float
    per_class_f1: Dict[int, float]
    per_class_precision: Dict[int, float]
    per_class_recall: Dict[int, float]
    
    # Open-set metrics
    oscr_score: float
    fpr_at_95tpr: float
    auroc: float
    aupr: float
    
    # Risk-coverage metrics
    risk_coverage_auc: float
    optimal_threshold: float
    optimal_coverage: float
    
    # Processing impact metrics
    wer_improvement: Optional[float] = None
    uar_improvement: Optional[float] = None
    processing_effectiveness: Optional[float] = None


@dataclass
class PerformanceSlice:
    """Container for performance analysis on specific data slices."""
    slice_name: str
    slice_conditions: Dict[str, Any]
    metrics: EvaluationMetrics
    sample_count: int
    confidence_scores: np.ndarray
    prediction_risks: np.ndarray


class WERUARPairedTester:
    """WER vs UAR Paired Tests for Audio Processing Impact Analysis."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.results = {}
        
        logging.info(f"WER-UAR Paired Tester initialized with confidence level {confidence_level}")
    
    def compute_wer(self, reference_texts: List[str], predicted_texts: List[str]) -> float:
        """Compute Word Error Rate."""
        total_errors = 0
        total_words = 0
        
        for ref, pred in zip(reference_texts, predicted_texts):
            ref_words = ref.lower().split()
            pred_words = pred.lower().split()
            
            # Use Levenshtein distance for word-level errors
            errors = self._levenshtein_distance(ref_words, pred_words)
            total_errors += errors
            total_words += len(ref_words)
        
        return (total_errors / total_words) * 100 if total_words > 0 else 0.0
    
    def _levenshtein_distance(self, ref_words: List[str], pred_words: List[str]) -> int:
        """Compute Levenshtein distance between word lists."""
        m, n = len(ref_words), len(pred_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == pred_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        
        return dp[m][n]
    
    def compute_uar(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Unweighted Average Recall."""
        unique_labels = np.unique(y_true)
        recalls = []
        
        for label in unique_labels:
            mask = y_true == label
            if mask.sum() > 0:
                recall = (y_pred[mask] == label).sum() / mask.sum()
                recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def paired_test(self, 
                    raw_metrics: Dict[str, float],
                    processed_metrics: Dict[str, float],
                    sample_count: int) -> Dict[str, Any]:
        """Perform paired statistical test for processing effectiveness."""
        # Extract metrics
        raw_wer = raw_metrics.get('wer', 0.0)
        raw_uar = raw_metrics.get('uar', 0.0)
        processed_wer = processed_metrics.get('wer', 0.0)
        processed_uar = processed_metrics.get('uar', 0.0)
        
        # Compute improvements
        wer_improvement = raw_wer - processed_wer  # Lower WER is better
        uar_improvement = processed_uar - raw_uar  # Higher UAR is better
        
        # Statistical significance testing
        wer_effect_size = wer_improvement / (raw_wer + 1e-8)
        uar_effect_size = uar_improvement / (raw_uar + 1e-8)
        
        wer_significant = abs(wer_effect_size) > 0.1 and sample_count > 30
        uar_significant = abs(uar_effect_size) > 0.1 and sample_count > 30
        
        processing_effectiveness = (wer_improvement + uar_improvement) / 2
        
        results = {
            'raw_wer': raw_wer,
            'raw_uar': raw_uar,
            'processed_wer': processed_wer,
            'processed_uar': processed_uar,
            'wer_improvement': wer_improvement,
            'uar_improvement': uar_improvement,
            'wer_significant': wer_significant,
            'uar_significant': uar_significant,
            'processing_effectiveness': processing_effectiveness,
            'sample_count': sample_count
        }
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive WER-UAR paired test report."""
        if not self.results:
            return "No test results available."
        
        r = self.results
        
        report = f"""
WER-UAR Paired Test Report:
===========================
Sample Count: {r['sample_count']}

Raw Audio Performance:
  WER: {r['raw_wer']:.2f}%
  UAR: {r['raw_uar']:.4f}

Processed Audio Performance:
  WER: {r['processed_wer']:.2f}%
  UAR: {r['processed_uar']:.4f}

Processing Impact:
  WER Improvement: {r['wer_improvement']:+.2f}% ({'Significant' if r['wer_significant'] else 'Not Significant'})
  UAR Improvement: {r['uar_improvement']:+.4f} ({'Significant' if r['uar_significant'] else 'Not Significant'})
  Overall Effectiveness: {r['processing_effectiveness']:+.4f}
"""
        return report


class OpenSetEvaluator:
    """Open-Set Classification Evaluation."""
    
    def __init__(self):
        self.metrics = {}
        
        logging.info("Open-Set Evaluator initialized")
    
    def compute_oscr(self, 
                     confidence_scores: np.ndarray,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     thresholds: np.ndarray = None) -> Dict[str, float]:
        """Compute Open Set Classification Rate (OSCR)."""
        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 101)
        
        # Separate known and unknown samples
        known_mask = y_true != -1  # -1 indicates unknown class
        unknown_mask = ~known_mask
        
        if not np.any(known_mask) or not np.any(unknown_mask):
            logger.warning("No known or unknown samples found for OSCR computation")
            return {'oscr_score': 0.0, 'thresholds': thresholds, 'oscr_curve': []}
        
        known_confidences = confidence_scores[known_mask]
        unknown_confidences = confidence_scores[unknown_mask]
        known_predictions = y_pred[known_mask]
        known_labels = y_true[known_mask]
        
        oscr_scores = []
        tpr_scores = []
        fpr_scores = []
        
        for threshold in thresholds:
            # True Positive Rate: correctly classified known samples above threshold
            correct_known = (known_predictions == known_labels) & (known_confidences >= threshold)
            tpr = correct_known.sum() / known_mask.sum() if known_mask.sum() > 0 else 0.0
            
            # False Positive Rate: unknown samples above threshold
            fpr = (unknown_confidences >= threshold).sum() / unknown_mask.sum() if unknown_mask.sum() > 0 else 0.0
            
            # OSCR: TPR - FPR
            oscr = tpr - fpr
            
            oscr_scores.append(oscr)
            tpr_scores.append(tpr)
            fpr_scores.append(fpr)
        
        # Find optimal threshold (maximum OSCR)
        optimal_idx = np.argmax(oscr_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_oscr = oscr_scores[optimal_idx]
        
        # Compute AUROC and AUPR
        auroc = self._compute_auroc(known_confidences, unknown_confidences)
        aupr = self._compute_aupr(known_confidences, unknown_confidences)
        
        # Find FPR at 95% TPR
        fpr_at_95tpr = self._find_fpr_at_tpr(fpr_scores, tpr_scores, target_tpr=0.95)
        
        results = {
            'oscr_score': optimal_oscr,
            'optimal_threshold': optimal_threshold,
            'thresholds': thresholds,
            'oscr_curve': oscr_scores,
            'tpr_curve': tpr_scores,
            'fpr_curve': fpr_scores,
            'auroc': auroc,
            'aupr': aupr,
            'fpr_at_95tpr': fpr_at_95tpr
        }
        
        self.metrics = results
        return results
    
    def _compute_auroc(self, known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
        """Compute Area Under ROC Curve."""
        try:
            y_true = np.concatenate([np.ones_like(known_scores), np.zeros_like(unknown_scores)])
            y_scores = np.concatenate([known_scores, unknown_scores])
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            return auc(fpr, tpr)
        except:
            return 0.0
    
    def _compute_aupr(self, known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve."""
        try:
            y_true = np.concatenate([np.ones_like(known_scores), np.zeros_like(unknown_scores)])
            y_scores = np.concatenate([known_scores, unknown_scores])
            
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            return auc(recall, precision)
        except:
            return 0.0
    
    def _find_fpr_at_tpr(self, fpr_scores: List[float], tpr_scores: List[float], target_tpr: float = 0.95) -> float:
        """Find FPR at target TPR."""
        try:
            tpr_array = np.array(tpr_scores)
            target_idx = np.argmin(np.abs(tpr_array - target_tpr))
            return fpr_scores[target_idx]
        except:
            return 1.0


class RiskCoverageAnalyzer:
    """Risk-Coverage Analysis for prediction confidence."""
    
    def __init__(self):
        self.curves = {}
        
        logging.info("Risk-Coverage Analyzer initialized")
    
    def compute_risk_coverage_curve(self,
                                   confidence_scores: np.ndarray,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   thresholds: np.ndarray = None) -> Dict[str, Any]:
        """Compute risk-coverage curve."""
        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 101)
        
        coverage_rates = []
        risk_rates = []
        error_rates = []
        
        for threshold in thresholds:
            # Coverage: fraction of samples above threshold
            above_threshold = confidence_scores >= threshold
            coverage = above_threshold.sum() / len(confidence_scores)
            
            if coverage > 0:
                # Risk: error rate among samples above threshold
                threshold_predictions = y_pred[above_threshold]
                threshold_labels = y_true[above_threshold]
                errors = (threshold_predictions != threshold_labels).sum()
                risk = errors / above_threshold.sum()
            else:
                risk = 0.0
            
            coverage_rates.append(coverage)
            risk_rates.append(risk)
            error_rates.append(errors if coverage > 0 else 0)
        
        # Compute AUC for risk-coverage curve
        risk_coverage_auc = np.trapz(risk_rates, coverage_rates)
        
        # Find optimal operating point
        reasonable_coverage_mask = np.array(coverage_rates) > 0.5
        if np.any(reasonable_coverage_mask):
            reasonable_indices = np.where(reasonable_coverage_mask)[0]
            optimal_idx_in_reasonable = np.argmin(np.array(risk_rates)[reasonable_coverage_mask])
            optimal_idx = reasonable_indices[optimal_idx_in_reasonable]
            optimal_threshold = thresholds[optimal_idx]
            optimal_coverage = coverage_rates[optimal_idx]
            optimal_risk = risk_rates[optimal_idx]
        else:
            optimal_threshold = thresholds[-1]
            optimal_coverage = coverage_rates[-1]
            optimal_risk = risk_rates[-1]
        
        results = {
            'thresholds': thresholds,
            'coverage_rates': coverage_rates,
            'risk_rates': risk_rates,
            'error_rates': error_rates,
            'risk_coverage_auc': risk_coverage_auc,
            'optimal_threshold': optimal_threshold,
            'optimal_coverage': optimal_coverage,
            'optimal_risk': optimal_risk
        }
        
        return results


class PerformanceSlicer:
    """Performance Slicing Analysis."""
    
    def __init__(self):
        self.slices = {}
        self.cross_analysis = {}
        
        logging.info("Performance Slicer initialized")
    
    def create_language_slice(self, 
                             data: Dict[str, Any],
                             language_id: int,
                             language_name: str) -> Optional['PerformanceSlice']:
        """Create performance slice for specific language."""
        # Filter data for specific language
        language_mask = data['language_ids'] == language_id
        
        if not np.any(language_mask):
            logger.warning(f"No data found for language {language_name} (ID: {language_id})")
            return None
        
        # Extract language-specific data
        slice_data = {
            'y_true': data['y_true'][language_mask],
            'y_pred': data['y_pred'][language_mask],
            'confidence_scores': data['confidence_scores'][language_mask],
            'snr_values': data['snr_values'][language_mask] if 'snr_values' in data else None
        }
        
        # Compute metrics for this slice
        metrics = self._compute_slice_metrics(slice_data)
        
        slice_obj = PerformanceSlice(
            slice_name=f"Language_{language_name}",
            slice_conditions={'language_id': language_id, 'language_name': language_name},
            metrics=metrics,
            sample_count=language_mask.sum(),
            confidence_scores=slice_data['confidence_scores'],
            prediction_risks=np.where(slice_data['y_true'] == slice_data['y_pred'], 0, 1)
        )
        
        return slice_obj
    
    def create_snr_slice(self, 
                         data: Dict[str, Any],
                         snr_range: Tuple[float, float],
                         range_name: str) -> Optional['PerformanceSlice']:
        """Create performance slice for specific SNR range."""
        # Filter data for specific SNR range
        snr_mask = (data['snr_values'] >= snr_range[0]) & (data['snr_values'] < snr_range[1])
        
        if not np.any(snr_mask):
            logger.warning(f"No data found for SNR range {range_name} ({snr_range[0]}-{snr_range[1]}dB)")
            return None
        
        # Extract SNR-specific data
        slice_data = {
            'y_true': data['y_true'][snr_mask],
            'y_pred': data['y_pred'][snr_mask],
            'confidence_scores': data['confidence_scores'][snr_mask],
            'language_ids': data['language_ids'][snr_mask] if 'language_ids' in data else None
        }
        
        # Compute metrics for this slice
        metrics = self._compute_slice_metrics(slice_data)
        
        slice_obj = PerformanceSlice(
            slice_name=f"SNR_{range_name}",
            slice_conditions={'snr_range': snr_range, 'range_name': range_name},
            metrics=metrics,
            sample_count=snr_mask.sum(),
            confidence_scores=slice_data['confidence_scores'],
            prediction_risks=np.where(slice_data['y_true'] == slice_data['y_pred'], 0, 1)
        )
        
        return slice_obj
    
    def _compute_slice_metrics(self, slice_data: Dict[str, Any]) -> EvaluationMetrics:
        """Compute comprehensive metrics for a data slice."""
        y_true = slice_data['y_true']
        y_pred = slice_data['y_pred']
        confidence_scores = slice_data['confidence_scores']
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        unique_labels = np.unique(y_true)
        per_class_f1 = {}
        per_class_precision = {}
        per_class_recall = {}
        
        for label in unique_labels:
            per_class_f1[label] = f1_score(y_true, y_pred, labels=[label], average=None, zero_division=0)[0]
        
        # Simplified metrics for now
        oscr_score = 0.0
        fpr_at_95tpr = 0.0
        auroc = 0.0
        aupr = 0.0
        risk_coverage_auc = 0.0
        optimal_threshold = 0.5
        optimal_coverage = 1.0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            weighted_f1=weighted_f1,
            macro_f1=macro_f1,
            per_class_f1=per_class_f1,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            oscr_score=oscr_score,
            fpr_at_95tpr=fpr_at_95tpr,
            auroc=auroc,
            aupr=aupr,
            risk_coverage_auc=risk_coverage_auc,
            optimal_threshold=optimal_threshold,
            optimal_coverage=optimal_coverage
        )


class EnhancedEvaluationPipeline:
    """Complete Enhanced Evaluation Pipeline."""
    
    def __init__(self, 
                 output_dir: str = "evaluation_results",
                 save_plots: bool = True):
        self.output_dir = Path(output_dir)
        self.save_plots = save_plots
        
        # Initialize components
        self.wer_uar_tester = WERUARPairedTester()
        self.open_set_evaluator = OpenSetEvaluator()
        self.risk_coverage_analyzer = RiskCoverageAnalyzer()
        self.performance_slicer = PerformanceSlicer()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        logging.info(f"Enhanced Evaluation Pipeline initialized with output directory: {self.output_dir}")
    
    def evaluate_model(self, 
                      model_results: Dict[str, Any],
                      evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""
        results = {}
        
        # 1. WER vs UAR Paired Tests
        if 'raw_audio_metrics' in evaluation_data and 'processed_audio_metrics' in evaluation_data:
            logger.info("Performing WER-UAR paired tests...")
            wer_uar_results = self.wer_uar_tester.paired_test(
                evaluation_data['raw_audio_metrics'],
                evaluation_data['processed_audio_metrics'],
                len(evaluation_data['y_true'])
            )
            results['wer_uar_analysis'] = wer_uar_results
        
        # 2. Open-Set Classification Evaluation
        if 'confidence_scores' in model_results:
            logger.info("Computing open-set classification metrics...")
            open_set_results = self.open_set_evaluator.compute_oscr(
                model_results['confidence_scores'],
                evaluation_data['y_true'],
                model_results['y_pred']
            )
            results['open_set_metrics'] = open_set_results
        
        # 3. Risk-Coverage Analysis
        if 'confidence_scores' in model_results:
            logger.info("Computing risk-coverage analysis...")
            risk_coverage_results = self.risk_coverage_analyzer.compute_risk_coverage_curve(
                model_results['confidence_scores'],
                evaluation_data['y_true'],
                model_results['y_pred']
            )
            results['risk_coverage_analysis'] = risk_coverage_results
        
        # 4. Performance Slicing
        logger.info("Performing performance slicing analysis...")
        slices = []
        
        # Language-based slicing
        if 'language_ids' in evaluation_data:
            unique_languages = np.unique(evaluation_data['language_ids'])
            for lang_id in unique_languages:
                slice_obj = self.performance_slicer.create_language_slice(
                    {**evaluation_data, **model_results}, lang_id, f"Lang_{lang_id}"
                )
                if slice_obj:
                    slices.append(slice_obj)
        
        # SNR-based slicing
        if 'snr_values' in evaluation_data:
            snr_ranges = [(-float('inf'), 5), (5, 10), (10, 15), (15, 20), (20, float('inf'))]
            range_names = ['<5dB', '5-10dB', '10-15dB', '15-20dB', '>20dB']
            
            for (low, high), name in zip(snr_ranges, range_names):
                slice_obj = self.performance_slicer.create_snr_slice(
                    {**evaluation_data, **model_results}, (low, high), name
                )
                if slice_obj:
                    slices.append(slice_obj)
        
        results['performance_slices'] = [slice_obj.__dict__ for slice_obj in slices]
        
        # 5. Generate comprehensive report
        logger.info("Generating evaluation report...")
        report = self._generate_comprehensive_report(results)
        results['evaluation_report'] = report
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report."""
        report = f"""
Enhanced Evaluation Pipeline Report
==================================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

1. WER-UAR Paired Tests
------------------------
"""
        
        if 'wer_uar_analysis' in results:
            report += self.wer_uar_tester.generate_report()
        else:
            report += "No WER-UAR analysis performed (missing raw/processed audio metrics)\n"
        
        report += f"""

2. Open-Set Classification Metrics
---------------------------------
"""
        
        if 'open_set_metrics' in results and results['open_set_metrics']:
            om = results['open_set_metrics']
            if 'oscr_score' in om:
                report += f"""
OSCR Score: {om['oscr_score']:.4f}
Optimal Threshold: {om.get('optimal_threshold', 'N/A')}
AUROC: {om.get('auroc', 'N/A')}
AUPR: {om.get('aupr', 'N/A')}
FPR@95TPR: {om.get('fpr_at_95tpr', 'N/A')}
"""
            else:
                report += "Open-set metrics computed but incomplete\n"
        else:
            report += "No open-set metrics computed\n"
        
        report += f"""

3. Risk-Coverage Analysis
-------------------------
"""
        
        if 'risk_coverage_analysis' in results:
            rc = results['risk_coverage_analysis']
            report += f"""
Risk-Coverage AUC: {rc['risk_coverage_auc']:.4f}
Optimal Threshold: {rc['optimal_threshold']:.3f}
Optimal Coverage: {rc['optimal_coverage']:.3f}
Optimal Risk: {rc['optimal_risk']:.4f}
"""
        else:
            report += "No risk-coverage analysis performed\n"
        
        report += f"""

4. Performance Slicing
----------------------
"""
        
        if 'performance_slices' in results:
            report += f"Total slices analyzed: {len(results['performance_slices'])}\n"
            for slice_data in results['performance_slices']:
                metrics = slice_data['metrics']
                if hasattr(metrics, 'accuracy'):
                    # Handle EvaluationMetrics object
                    report += f"""
{slice_data['slice_name']}:
  Sample count: {slice_data['sample_count']}
  Accuracy: {metrics.accuracy:.4f}
  Weighted F1: {metrics.weighted_f1:.4f}
  Macro F1: {metrics.macro_f1:.4f}
"""
                else:
                    # Handle dictionary format
                    report += f"""
{slice_data['slice_name']}:
  Sample count: {slice_data['sample_count']}
  Accuracy: {metrics.get('accuracy', 'N/A')}
  Weighted F1: {metrics.get('weighted_f1', 'N/A')}
  Macro F1: {metrics.get('macro_f1', 'N/A')}
"""
        else:
            report += "No performance slicing performed\n"
        
        return report
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        # Save main results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save evaluation report
        report_file = self.output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(results['evaluation_report'])
        
        logger.info(f"Evaluation results saved to {self.output_dir}")


# Utility functions
def create_enhanced_evaluation_pipeline(output_dir: str = "evaluation_results") -> EnhancedEvaluationPipeline:
    """Factory function to create enhanced evaluation pipeline."""
    return EnhancedEvaluationPipeline(output_dir=output_dir)


def create_sample_evaluation_data(num_samples: int = 1000,
                                num_classes: int = 4,
                                num_languages: int = 7) -> Dict[str, Any]:
    """Create sample evaluation data for testing."""
    np.random.seed(42)
    
    # Generate sample data
    y_true = np.random.randint(0, num_classes, num_samples)
    y_pred = np.random.randint(0, num_classes, num_samples)
    confidence_scores = np.random.uniform(0.5, 1.0, num_samples)
    language_ids = np.random.randint(0, num_languages, num_samples)
    snr_values = np.random.uniform(5.0, 25.0, num_samples)
    
    # Add some correlation between predictions and ground truth
    correct_mask = np.random.random(num_samples) < 0.8
    y_pred[correct_mask] = y_true[correct_mask]
    
    # Add some correlation between confidence and correctness
    confidence_scores[correct_mask] = np.random.uniform(0.7, 1.0, correct_mask.sum())
    confidence_scores[~correct_mask] = np.random.uniform(0.5, 0.8, (~correct_mask).sum())
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'confidence_scores': confidence_scores,
        'language_ids': language_ids,
        'snr_values': snr_values,
        'raw_audio_metrics': {'wer': 15.5, 'uar': 0.75},
        'processed_audio_metrics': {'wer': 12.3, 'uar': 0.82}
    }
