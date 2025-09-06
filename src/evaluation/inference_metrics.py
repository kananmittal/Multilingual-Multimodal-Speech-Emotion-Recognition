#!/usr/bin/env python3
"""
Inference throughput and efficiency metrics.
Implements comprehensive performance benchmarking for model inference.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import psutil
import os
from pathlib import Path
import json

@dataclass
class ThroughputMetrics:
    """Inference throughput metrics."""
    samples_per_second: float
    words_per_second: float
    audio_seconds_per_second: float
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]
    cpu_usage_percent: float

@dataclass
class EfficiencyMetrics:
    """Model efficiency metrics."""
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    parameters_per_second: float
    flops_per_sample: Optional[float]
    energy_efficiency: Optional[float]  # samples per joule (if power monitoring available)

class InferenceBenchmarker:
    """Benchmarks model inference performance and efficiency."""
    
    def __init__(self, device: str = 'auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.benchmark_results = []
        
    def get_memory_usage(self) -> Tuple[float, Optional[float]]:
        """Get current memory usage in MB."""
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # GPU memory (if available)
        gpu_memory_mb = None
        if torch.cuda.is_available() and self.device == 'cuda':
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
        return cpu_memory_mb, gpu_memory_mb
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    def count_parameters(self, model: torch.nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def estimate_model_size(self, model: torch.nn.Module) -> float:
        """Estimate model size in MB."""
        total_params, _ = self.count_parameters(model)
        # Assume float32 parameters (4 bytes each)
        size_bytes = total_params * 4
        return size_bytes / 1024 / 1024
    
    def benchmark_inference(self, 
                           model: torch.nn.Module,
                           sample_inputs: List[Tuple],
                           num_warmup: int = 10,
                           num_runs: int = 100,
                           batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[int, ThroughputMetrics]:
        """Benchmark inference performance across different batch sizes."""
        
        print(f"🚀 Starting inference benchmark on {self.device}")
        print(f"Warmup runs: {num_warmup}, Benchmark runs: {num_runs}")
        
        model.eval()
        model.to(self.device)
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n📊 Benchmarking batch size: {batch_size}")
            
            # Prepare batch input
            batch_inputs = sample_inputs[:batch_size]
            if len(batch_inputs) < batch_size:
                # Pad with last sample if needed
                last_sample = batch_inputs[-1] if batch_inputs else sample_inputs[0]
                batch_inputs.extend([last_sample] * (batch_size - len(batch_inputs)))
            
            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    try:
                        _ = model(*batch_inputs)
                    except Exception as e:
                        print(f"Warning: Warmup failed: {e}")
                        break
            
            # Benchmark
            latencies = []
            memory_usage = []
            cpu_usage = []
            
            with torch.no_grad():
                for run in range(num_runs):
                    # Record start time and memory
                    start_time = time.time()
                    start_memory, start_gpu_memory = self.get_memory_usage()
                    start_cpu = self.get_cpu_usage()
                    
                    try:
                        # Run inference
                        outputs = model(*batch_inputs)
                        
                        # Record end time and memory
                        end_time = time.time()
                        end_memory, end_gpu_memory = self.get_memory_usage()
                        end_cpu = self.get_cpu_usage()
                        
                        # Calculate metrics
                        latency = (end_time - start_time) * 1000  # Convert to ms
                        memory_delta = end_memory - start_memory
                        cpu_delta = end_cpu - start_cpu
                        
                        latencies.append(latency)
                        memory_usage.append(memory_delta)
                        cpu_usage.append(cpu_delta)
                        
                    except Exception as e:
                        print(f"Warning: Benchmark run {run} failed: {e}")
                        continue
            
            if not latencies:
                print(f"❌ All benchmark runs failed for batch size {batch_size}")
                continue
            
            # Calculate statistics
            latencies = np.array(latencies)
            memory_usage = np.array(memory_usage)
            cpu_usage = np.array(cpu_usage)
            
            # Throughput metrics
            latency_mean = np.mean(latencies)
            latency_std = np.std(latencies)
            latency_p50 = np.percentile(latencies, 50)
            latency_p95 = np.percentile(latencies, 95)
            latency_p99 = np.percentile(latencies, 99)
            
            # Convert to samples per second
            samples_per_second = batch_size / (latency_mean / 1000)
            
            # Estimate words per second (assuming average 15 words per sample)
            avg_words_per_sample = 15
            words_per_second = samples_per_second * avg_words_per_sample
            
            # Estimate audio processing rate (assuming average 10 seconds per sample)
            avg_audio_seconds = 10
            audio_seconds_per_second = samples_per_second * avg_audio_seconds
            
            # Memory and CPU usage
            memory_usage_mb = np.mean(memory_usage)
            cpu_usage_percent = np.mean(cpu_usage)
            
            # GPU memory (if available)
            gpu_memory_mb = None
            if torch.cuda.is_available() and self.device == 'cuda':
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Create metrics object
            throughput_metrics = ThroughputMetrics(
                samples_per_second=samples_per_second,
                words_per_second=words_per_second,
                audio_seconds_per_second=audio_seconds_per_second,
                latency_mean=latency_mean,
                latency_std=latency_std,
                latency_p50=latency_p50,
                latency_p95=latency_p95,
                latency_p99=latency_p99,
                memory_usage_mb=memory_usage_mb,
                gpu_memory_mb=gpu_memory_mb,
                cpu_usage_percent=cpu_usage_percent
            )
            
            batch_results[batch_size] = throughput_metrics
            
            # Print results
            print(f"  ✅ Batch size {batch_size}:")
            print(f"    Throughput: {samples_per_second:.2f} samples/sec")
            print(f"    Latency: {latency_mean:.2f}ms ± {latency_std:.2f}ms")
            print(f"    P95 Latency: {latency_p95:.2f}ms")
            print(f"    Memory: {memory_usage_mb:.2f} MB")
            if gpu_memory_mb:
                print(f"    GPU Memory: {gpu_memory_mb:.2f} MB")
        
        return batch_results
    
    def calculate_efficiency_metrics(self, 
                                   model: torch.nn.Module,
                                   throughput_metrics: Dict[int, ThroughputMetrics]) -> EfficiencyMetrics:
        """Calculate model efficiency metrics."""
        
        # Parameter counts
        total_params, trainable_params = self.count_parameters(model)
        
        # Model size
        model_size_mb = self.estimate_model_size(model)
        
        # Parameters per second (using batch size 1 as baseline)
        baseline_throughput = throughput_metrics.get(1, throughput_metrics[min(throughput_metrics.keys())])
        params_per_second = total_params * baseline_throughput.samples_per_second
        
        # FLOPs estimation (simplified)
        flops_per_sample = None
        try:
            # This is a rough estimate - in practice, you'd use torch.profiler
            # For transformer models, approximate FLOPs per token
            if hasattr(model, 'config'):
                hidden_size = getattr(model.config, 'hidden_size', 768)
                num_layers = getattr(model.config, 'num_hidden_layers', 12)
                flops_per_sample = hidden_size * hidden_size * num_layers * 2  # Rough estimate
        except:
            pass
        
        # Energy efficiency (placeholder - would need power monitoring)
        energy_efficiency = None
        
        return EfficiencyMetrics(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size_mb,
            parameters_per_second=params_per_second,
            flops_per_sample=flops_per_sample,
            energy_efficiency=energy_efficiency
        )
    
    def generate_benchmark_report(self, 
                                 throughput_metrics: Dict[int, ThroughputMetrics],
                                 efficiency_metrics: EfficiencyMetrics,
                                 output_path: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("INFERENCE BENCHMARK REPORT")
        report_lines.append("=" * 80)
        
        # Model efficiency
        report_lines.append(f"\nMODEL EFFICIENCY:")
        report_lines.append(f"  Total Parameters: {efficiency_metrics.total_parameters:,}")
        report_lines.append(f"  Trainable Parameters: {efficiency_metrics.trainable_parameters:,}")
        report_lines.append(f"  Model Size: {efficiency_metrics.model_size_mb:.2f} MB")
        report_lines.append(f"  Parameters per Second: {efficiency_metrics.parameters_per_second:,.0f}")
        
        if efficiency_metrics.flops_per_sample:
            report_lines.append(f"  Estimated FLOPs per Sample: {efficiency_metrics.flops_per_sample:,.0f}")
        
        # Throughput analysis
        report_lines.append(f"\nTHROUGHPUT ANALYSIS:")
        for batch_size in sorted(throughput_metrics.keys()):
            metrics = throughput_metrics[batch_size]
            report_lines.append(f"\n  Batch Size {batch_size}:")
            report_lines.append(f"    Throughput: {metrics.samples_per_second:.2f} samples/sec")
            report_lines.append(f"    Words per Second: {metrics.words_per_second:.2f}")
            report_lines.append(f"    Audio Processing: {metrics.audio_seconds_per_second:.2f} sec/sec")
            report_lines.append(f"    Latency: {metrics.latency_mean:.2f}ms ± {metrics.latency_std:.2f}ms")
            report_lines.append(f"    P95 Latency: {metrics.latency_p95:.2f}ms")
            report_lines.append(f"    P99 Latency: {metrics.latency_p99:.2f}ms")
            report_lines.append(f"    Memory Usage: {metrics.memory_usage_mb:.2f} MB")
            if metrics.gpu_memory_mb:
                report_lines.append(f"    GPU Memory: {metrics.gpu_memory_mb:.2f} MB")
            report_lines.append(f"    CPU Usage: {metrics.cpu_usage_percent:.2f}%")
        
        # Performance recommendations
        report_lines.append(f"\nPERFORMANCE RECOMMENDATIONS:")
        
        # Find optimal batch size
        optimal_batch_size = max(throughput_metrics.keys(), 
                               key=lambda x: throughput_metrics[x].samples_per_second)
        optimal_throughput = throughput_metrics[optimal_batch_size].samples_per_second
        
        report_lines.append(f"  🎯 Optimal batch size: {optimal_batch_size} (throughput: {optimal_throughput:.2f} samples/sec)")
        
        # Latency analysis
        baseline_latency = throughput_metrics[1].latency_mean
        report_lines.append(f"  ⏱️  Baseline latency (batch=1): {baseline_latency:.2f}ms")
        
        # Memory analysis
        max_memory = max(metrics.memory_usage_mb for metrics in throughput_metrics.values())
        report_lines.append(f"  💾 Peak memory usage: {max_memory:.2f} MB")
        
        # Scaling efficiency
        if len(throughput_metrics) > 1:
            batch_sizes = sorted(throughput_metrics.keys())
            scaling_efficiency = []
            for i in range(1, len(batch_sizes)):
                prev_batch = batch_sizes[i-1]
                curr_batch = batch_sizes[i]
                prev_throughput = throughput_metrics[prev_batch].samples_per_second
                curr_throughput = throughput_metrics[curr_batch].samples_per_second
                
                expected_throughput = prev_throughput * (curr_batch / prev_batch)
                actual_efficiency = curr_throughput / expected_throughput
                scaling_efficiency.append(actual_efficiency)
            
            avg_scaling = np.mean(scaling_efficiency)
            report_lines.append(f"  📈 Average scaling efficiency: {avg_scaling:.2f}")
            
            if avg_scaling < 0.8:
                report_lines.append("    ⚠️  Poor scaling - consider optimizing memory access or reducing overhead")
            elif avg_scaling > 0.95:
                report_lines.append("    ✅ Excellent scaling - model is well-optimized")
        
        # Save report
        full_report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(full_report)
            print(f"Benchmark report saved to: {output_path}")
        
        return full_report
    
    def save_benchmark_data(self, 
                           throughput_metrics: Dict[int, ThroughputMetrics],
                           efficiency_metrics: EfficiencyMetrics,
                           output_path: str) -> None:
        """Save benchmark data to JSON file."""
        
        # Convert to serializable format
        benchmark_data = {
            'efficiency_metrics': {
                'total_parameters': efficiency_metrics.total_parameters,
                'trainable_parameters': efficiency_metrics.trainable_parameters,
                'model_size_mb': efficiency_metrics.model_size_mb,
                'parameters_per_second': efficiency_metrics.parameters_per_second,
                'flops_per_sample': efficiency_metrics.flops_per_sample
            },
            'throughput_metrics': {}
        }
        
        for batch_size, metrics in throughput_metrics.items():
            benchmark_data['throughput_metrics'][str(batch_size)] = {
                'samples_per_second': metrics.samples_per_second,
                'words_per_second': metrics.words_per_second,
                'audio_seconds_per_second': metrics.audio_seconds_per_second,
                'latency_mean': metrics.latency_mean,
                'latency_std': metrics.latency_std,
                'latency_p50': metrics.latency_p50,
                'latency_p95': metrics.latency_p95,
                'latency_p99': metrics.latency_p99,
                'memory_usage_mb': metrics.memory_usage_mb,
                'gpu_memory_mb': metrics.gpu_memory_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent
            }
        
        with open(output_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"Benchmark data saved to: {output_path}")

def create_inference_benchmarker(device: str = 'auto') -> InferenceBenchmarker:
    """Create an inference benchmarker instance."""
    return InferenceBenchmarker(device=device)
