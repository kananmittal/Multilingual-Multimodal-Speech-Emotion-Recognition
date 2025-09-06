#!/usr/bin/env python3
"""
Calibration metrics including Expected Calibration Error (ECE).
Implements comprehensive model calibration evaluation.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

@dataclass
class CalibrationMetrics:
    """Model calibration metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    reliability_diagram: Tuple[np.ndarray, np.ndarray]  # (confidence, accuracy)
    calibration_curve: Tuple[np.ndarray, np.ndarray]  # (confidence, fraction_positives)
    n_bins: int

class CalibrationEvaluator:
    """Evaluates model calibration quality."""
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        
    def expected_calibration_error(self, 
                                 confidences: np.ndarray, 
                                 accuracies: np.ndarray, 
                                 bin_counts: np.ndarray) -> float:
        """Calculate Expected Calibration Error."""
        ece = 0.0
        total_samples = np.sum(bin_counts)
        
        for i in range(len(confidences)):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / total_samples) * np.abs(confidences[i] - accuracies[i])
        
        return ece
    
    def maximum_calibration_error(self, 
                                confidences: np.ndarray, 
                                accuracies: np.ndarray) -> float:
        """Calculate Maximum Calibration Error."""
        return np.max(np.abs(confidences - accuracies))
    
    def compute_calibration_metrics(self, 
                                  predictions: np.ndarray, 
                                  labels: np.ndarray, 
                                  probabilities: np.ndarray) -> CalibrationMetrics:
        """Compute comprehensive calibration metrics."""
        
        # Get confidence scores (max probability for each prediction)
        confidences = np.max(probabilities, axis=1)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Initialize arrays
        bin_confidences = np.zeros(self.n_bins)
        bin_accuracies = np.zeros(self.n_bins)
        bin_counts = np.zeros(self.n_bins)
        
        # Compute metrics for each bin
        for bin_idx in range(self.n_bins):
            # Find samples in this bin
            in_bin = np.logical_and(
                confidences > bin_lowers[bin_idx],
                confidences <= bin_uppers[bin_idx]
            )
            
            bin_counts[bin_idx] = np.sum(in_bin)
            
            if bin_counts[bin_idx] > 0:
                # Compute accuracy for this bin
                bin_accuracies[bin_idx] = np.mean(predictions[in_bin] == labels[in_bin])
                
                # Compute average confidence for this bin
                bin_confidences[bin_idx] = np.mean(confidences[in_bin])
        
        # Compute ECE and MCE
        ece = self.expected_calibration_error(bin_confidences, bin_accuracies, bin_counts)
        mce = self.maximum_calibration_error(bin_confidences, bin_accuracies)
        
        # Compute reliability diagram
        reliability_diagram = (bin_confidences, bin_accuracies)
        
        # Compute calibration curve using sklearn
        try:
            # For binary case, we need to convert to binary labels
            if len(np.unique(labels)) == 2:
                binary_labels = (labels == 1).astype(int)
                binary_probs = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
                fraction_positives, mean_predicted_value = calibration_curve(
                    binary_labels, binary_probs, n_bins=self.n_bins
                )
                calibration_curve_data = (mean_predicted_value, fraction_positives)
            else:
                # For multi-class, use the predicted class probabilities
                predicted_probs = probabilities[np.arange(len(predictions)), predictions]
                fraction_positives, mean_predicted_value = calibration_curve(
                    (predictions == labels).astype(int), predicted_probs, n_bins=self.n_bins
                )
                calibration_curve_data = (mean_predicted_value, fraction_positives)
        except Exception:
            # Fallback: use our computed values
            calibration_curve_data = (bin_confidences, bin_accuracies)
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            reliability_diagram=reliability_diagram,
            calibration_curve=calibration_curve_data,
            n_bins=self.n_bins
        )
    
    def plot_calibration_diagram(self, 
                                metrics: CalibrationMetrics, 
                                save_path: Optional[str] = None) -> None:
        """Plot reliability diagram and calibration curve."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Reliability diagram
        confidences, accuracies = metrics.reliability_diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax1.scatter(confidences, accuracies, c='blue', alpha=0.7, s=50)
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Reliability Diagram (ECE: {metrics.ece:.4f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calibration curve
        pred_conf, true_fraction = metrics.calibration_curve
        ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax2.plot(pred_conf, true_fraction, 'o-', c='red', alpha=0.7, label='Model')
        ax2.set_xlabel('Mean Predicted Confidence')
        ax2.set_ylabel('Fraction of Positives')
        ax2.set_title(f'Calibration Curve (MCE: {metrics.mce:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration diagram saved to: {save_path}")
        
        plt.show()
    
    def print_calibration_report(self, metrics: CalibrationMetrics) -> str:
        """Print comprehensive calibration report."""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MODEL CALIBRATION EVALUATION")
        report_lines.append("=" * 60)
        
        report_lines.append(f"\nCalibration Metrics:")
        report_lines.append(f"  Expected Calibration Error (ECE): {metrics.ece:.4f}")
        report_lines.append(f"  Maximum Calibration Error (MCE): {metrics.mce:.4f}")
        report_lines.append(f"  Number of bins: {metrics.n_bins}")
        
        # ECE interpretation
        if metrics.ece < 0.05:
            ece_quality = "Excellent"
        elif metrics.ece < 0.1:
            ece_quality = "Good"
        elif metrics.ece < 0.15:
            ece_quality = "Fair"
        else:
            ece_quality = "Poor"
        
        report_lines.append(f"\nECE Quality: {ece_quality}")
        report_lines.append(f"  ECE < 0.05: Excellent calibration")
        report_lines.append(f"  ECE < 0.10: Good calibration")
        report_lines.append(f"  ECE < 0.15: Fair calibration")
        report_lines.append(f"  ECE >= 0.15: Poor calibration")
        
        # Reliability diagram summary
        confidences, accuracies = metrics.reliability_diagram
        report_lines.append(f"\nReliability Diagram Summary:")
        report_lines.append(f"  Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
        report_lines.append(f"  Accuracy range: {accuracies.min():.3f} - {accuracies.max():.3f}")
        
        # Overconfidence/underconfidence analysis
        overconfident = np.sum(confidences > accuracies)
        underconfident = np.sum(confidences < accuracies)
        well_calibrated = np.sum(np.abs(confidences - accuracies) < 0.05)
        
        report_lines.append(f"\nCalibration Analysis:")
        report_lines.append(f"  Overconfident bins: {overconfident}")
        report_lines.append(f"  Underconfident bins: {underconfident}")
        report_lines.append(f"  Well-calibrated bins: {well_calibrated}")
        
        report = "\n".join(report_lines)
        print(report)
        return report

def create_calibration_evaluator(n_bins: int = 15) -> CalibrationEvaluator:
    """Create a calibration evaluator instance."""
    return CalibrationEvaluator(n_bins=n_bins)
