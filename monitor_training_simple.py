#!/usr/bin/env python3
"""
Simple training monitor without external dependencies
"""

import os
import time
import subprocess

def monitor_training():
    """Monitor training progress"""
    print("🔍 Monitoring CREMA Training Progress...")
    print("=" * 60)
    
    while True:
        try:
            # Check if training process is running
            result = subprocess.run(['pgrep', '-f', 'train_crema_final'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print("❌ Training process not found")
                break
            
            # Check for checkpoints
            checkpoint_dir = "checkpoints_crema_final"
            checkpoints = []
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            
            # Print status
            print(f"\r🔄 Training Status:")
            print(f"   Process: Running")
            print(f"   Checkpoints: {len(checkpoints)}")
            
            if checkpoints:
                print(f"   Latest: {sorted(checkpoints)[-1]}")
                print("✅ Training completed!")
                break
            else:
                print("   Status: First epoch in progress...")
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_training()
