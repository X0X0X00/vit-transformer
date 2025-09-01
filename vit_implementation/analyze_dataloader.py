#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from data.dataset import get_dataloaders
from data.balanced_dataset import get_balanced_dataloaders, CIFAR10_CLASSES

def analyze_class_distribution(dataloader, name="Dataset"):
    """分析数据集中各类别的分布"""
    class_counts = Counter()
    total_samples = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        for target in targets:
            class_counts[target.item()] += 1
            total_samples += 1
        
        # 只分析前10个batch来快速估算
        if batch_idx >= 9:
            break
    
    print(f"\n{name} - Class Distribution (first 10 batches):")
    print("-" * 50)
    for class_idx in range(10):
        count = class_counts.get(class_idx, 0)
        percentage = (count / total_samples) * 100
        print(f"{CIFAR10_CLASSES[class_idx]:>8}: {count:>4} samples ({percentage:>5.1f}%)")
    
    print(f"Total samples analyzed: {total_samples}")
    
    # 计算不平衡度
    counts = [class_counts.get(i, 0) for i in range(10)]
    if min(counts) > 0:
        imbalance_ratio = max(counts) / min(counts)
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    return class_counts

def compare_batch_variations(dataloader, name="Dataset", num_batches=5):
    """分析连续几个batch的类别分布变化"""
    print(f"\n{name} - Batch-to-Batch Variation:")
    print("-" * 50)
    
    batch_distributions = []
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        batch_counter = Counter()
        for target in targets:
            batch_counter[target.item()] += 1
        
        # 计算每个类别在此batch中的比例
        batch_size = len(targets)
        batch_dist = [batch_counter.get(i, 0) / batch_size for i in range(10)]
        batch_distributions.append(batch_dist)
        
        print(f"Batch {batch_idx + 1}:")
        for class_idx in range(10):
            count = batch_counter.get(class_idx, 0)
            percentage = (count / batch_size) * 100
            print(f"  {CIFAR10_CLASSES[class_idx]:>8}: {count:>2}/{batch_size} ({percentage:>5.1f}%)")
        print()
    
    # 计算变异系数 (CV = std/mean)
    batch_distributions = np.array(batch_distributions)
    cv_per_class = []
    
    print("Class Distribution Stability (CV = std/mean):")
    for class_idx in range(10):
        class_percentages = batch_distributions[:, class_idx] * 100
        if class_percentages.mean() > 0:
            cv = class_percentages.std() / class_percentages.mean()
            cv_per_class.append(cv)
            print(f"  {CIFAR10_CLASSES[class_idx]:>8}: CV = {cv:.3f}")
        else:
            cv_per_class.append(0)
            print(f"  {CIFAR10_CLASSES[class_idx]:>8}: CV = 0.000 (no samples)")
    
    avg_cv = np.mean(cv_per_class)
    print(f"Average CV across classes: {avg_cv:.3f}")
    print(f"{'Higher CV = more uneven batches' if avg_cv > 0.5 else 'Lower CV = more even batches'}")

def main():
    print("🔍 Analyzing CIFAR-10 DataLoader Distribution")
    print("=" * 60)
    
    # 原始数据加载器
    print("\n📊 Original DataLoaders:")
    train_loader_orig, val_loader_orig = get_dataloaders(batch_size=128, image_size=32)
    
    train_dist_orig = analyze_class_distribution(train_loader_orig, "Original Training")
    val_dist_orig = analyze_class_distribution(val_loader_orig, "Original Validation")
    
    # 分析batch变化
    compare_batch_variations(train_loader_orig, "Original Training")
    
    print("\n" + "="*60)
    print("📊 Balanced DataLoaders (No Augmentation):")
    
    # 平衡数据加载器（无增强）
    train_loader_bal, val_loader_bal = get_balanced_dataloaders(
        batch_size=128, 
        image_size=32,
        use_augmentation=False,
        use_balanced_sampling=True
    )
    
    train_dist_bal = analyze_class_distribution(train_loader_bal, "Balanced Training")
    compare_batch_variations(train_loader_bal, "Balanced Training")
    
    print("\n" + "="*60)
    print("💡 Recommendations:")
    print("-" * 60)
    
    # 检查原始数据是否平衡
    orig_counts = [train_dist_orig.get(i, 0) for i in range(10)]
    if len(set(orig_counts)) > 1:
        print("1. 🔴 Training batches show class imbalance")
        print("   → Use balanced sampling or stratified batches")
    else:
        print("1. ✅ Training batches are reasonably balanced")
    
    print("2. 🔄 If train_acc < val_acc, check:")
    print("   → Dropout rate (reduce if too high)")
    print("   → Data augmentation intensity")
    print("   → Model regularization settings")
    
    print("3. 🎯 For more stable training:")
    print("   → Use balanced_dataset.py with use_balanced_sampling=True")
    print("   → Reduce augmentation intensity")
    print("   → Use smaller dropout rates")

if __name__ == "__main__":
    main()