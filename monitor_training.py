#!/usr/bin/env python3
"""
Monitor training progress
"""

import os
import time
import subprocess
import psutil

def get_training_process():
    """Get the training process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if 'train_crema_optimized.py' in ' '.join(proc.info['cmdline']):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def monitor_training():
    """Monitor training progress"""
    print("🔍 Monitoring CREMA Training Progress...")
    print("=" * 60)
    
    while True:
        proc = get_training_process()
        
        if proc is None:
            print("❌ Training process not found")
            break
        
        # Get process info
        cpu_percent = proc.cpu_percent()
        memory_mb = proc.memory_info().rss / 1024 / 1024
        
        # Check for checkpoints
        checkpoint_dir = "checkpoints_crema"
        checkpoints = []
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        
        # Print status
        print(f"\r🔄 Training Status:")
        print(f"   CPU Usage: {cpu_percent:.1f}%")
        print(f"   Memory: {memory_mb:.0f}MB")
        print(f"   Checkpoints: {len(checkpoints)}")
        
        if checkpoints:
            print(f"   Latest: {sorted(checkpoints)[-1]}")
        
        # Check if training completed
        if cpu_percent < 5 and len(checkpoints) > 0:
            print("\n✅ Training appears to be completed!")
            break
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
