#!/usr/bin/env python3
"""
Test MPS performance vs CPU for model operations
"""

import torch
import time
import numpy as np

def test_device_performance():
    """Test performance difference between CPU and MPS"""
    
    print("🔍 Testing Device Performance...")
    
    # Test data
    batch_size = 8
    seq_len = 1000
    hidden_dim = 512
    
    # Create test tensors
    x_cpu = torch.randn(batch_size, seq_len, hidden_dim)
    
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    results = {}
    
    for device in devices:
        print(f"\n📊 Testing {device.upper()}...")
        
        # Move tensor to device
        x = x_cpu.to(device)
        
        # Test operations
        operations = {
            'Matrix Multiplication': lambda: torch.matmul(x, x.transpose(-1, -2)),
            'LayerNorm': lambda: torch.nn.functional.layer_norm(x, x.shape[-1:]),
            'Softmax': lambda: torch.softmax(x, dim=-1),
            'Dropout': lambda: torch.nn.functional.dropout(x, p=0.1, training=True),
            'ReLU': lambda: torch.relu(x),
        }
        
        device_results = {}
        
        for op_name, op_func in operations.items():
            # Warmup
            for _ in range(5):
                _ = op_func()
            
            if device == 'mps':
                torch.mps.synchronize()  # Wait for MPS operations to complete
            
            # Time the operation
            start_time = time.time()
            for _ in range(20):
                result = op_func()
            
            if device == 'mps':
                torch.mps.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 20
            device_results[op_name] = avg_time
            
            print(f"  {op_name}: {avg_time*1000:.2f}ms")
        
        results[device] = device_results
    
    # Compare results
    if len(results) > 1:
        print(f"\n📈 Performance Comparison:")
        cpu_results = results['cpu']
        mps_results = results['mps']
        
        for op_name in cpu_results:
            cpu_time = cpu_results[op_name]
            mps_time = mps_results[op_name]
            speedup = cpu_time / mps_time
            print(f"  {op_name}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} on MPS")
    
    return results

def test_model_forward_pass():
    """Test a simple model forward pass"""
    
    print(f"\n🧪 Testing Model Forward Pass...")
    
    # Simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
            self.norm = torch.nn.LayerNorm(hidden_dim)
            self.dropout = torch.nn.Dropout(0.1)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.norm(x)
            x = self.dropout(x)
            x = torch.relu(self.linear2(x))
            x = self.norm(x)
            x = self.dropout(x)
            x = self.linear3(x)
            return x
    
    # Test data
    batch_size = 8
    seq_len = 1000
    input_dim = 512
    hidden_dim = 512
    output_dim = 4
    
    x_cpu = torch.randn(batch_size, seq_len, input_dim)
    
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    for device in devices:
        print(f"\n📊 Testing {device.upper()} Model...")
        
        model = SimpleModel(input_dim, hidden_dim, output_dim).to(device)
        x = x_cpu.to(device)
        
        # Warmup
        for _ in range(5):
            _ = model(x)
        
        if device == 'mps':
            torch.mps.synchronize()
        
        # Time forward pass
        start_time = time.time()
        for _ in range(20):
            output = model(x)
        
        if device == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 20
        
        print(f"  Forward pass: {avg_time*1000:.2f}ms")
        print(f"  Output shape: {output.shape}")

if __name__ == "__main__":
    print("🍎 MPS Performance Test for Mac M3 Air")
    print("=" * 50)
    
    # Test basic operations
    test_device_performance()
    
    # Test model forward pass
    test_model_forward_pass()
    
    print(f"\n✅ Performance test completed!")
