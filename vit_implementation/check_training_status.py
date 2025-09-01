#!/usr/bin/env python3
import os
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def check_training_status():
    # Check for checkpoints
    checkpoint_dir = './checkpoints'
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            path = os.path.join(checkpoint_dir, ckpt)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {ckpt}: {size_mb:.2f} MB")
            
            # Load checkpoint and show info
            checkpoint = torch.load(path, map_location='cpu')
            if 'epoch' in checkpoint:
                print(f"    Epoch: {checkpoint['epoch']}")
            if 'train_accuracies' in checkpoint and checkpoint['train_accuracies']:
                print(f"    Last Train Acc: {checkpoint['train_accuracies'][-1]:.2f}%")
            if 'val_accuracies' in checkpoint and checkpoint['val_accuracies']:
                print(f"    Last Val Acc: {checkpoint['val_accuracies'][-1]:.2f}%")
    else:
        print("No checkpoints found yet. Training may still be in progress.")
    
    # Check TensorBoard logs
    log_dir = './logs'
    event_files = [f for f in os.listdir(log_dir) if 'events.out.tfevents' in f]
    
    if event_files:
        print(f"\nFound TensorBoard log file: {event_files[0]}")
        print("Training is running. Monitor progress at: http://127.0.0.1:6006")
    else:
        print("\nNo TensorBoard logs found.")
    
    # Check if training process is running
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'train.py' in result.stdout:
            print("\n✅ Training process is currently running")
        else:
            print("\n⚠️  No active training process found")
    except:
        pass

if __name__ == "__main__":
    check_training_status()