#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt
from data.dataset import get_dataloaders
from model.vit import vit_small
import numpy as np

def diagnose_model_training():
    print("🔍 ViT Training Diagnosis")
    print("=" * 50)
    
    # 1. 检查模型大小和复杂度
    model = vit_small()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # 2. 检查数据
    train_loader, val_loader = get_dataloaders(batch_size=32, image_size=224)
    
    # 获取一个batch检查数据
    data_batch, target_batch = next(iter(train_loader))
    print(f"\n📈 Data Analysis:")
    print(f"Input shape: {data_batch.shape}")
    print(f"Input range: [{data_batch.min():.3f}, {data_batch.max():.3f}]")
    print(f"Target range: [{target_batch.min()}, {target_batch.max()}]")
    print(f"Batch size: {len(target_batch)}")
    print(f"Number of classes: {len(torch.unique(target_batch))}")
    
    # 3. 模型前向传播测试
    model.eval()
    with torch.no_grad():
        output, attention = model(data_batch[:4])  # 测试4个样本
        print(f"\n🔧 Model Forward Pass:")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Softmax output range: [{torch.softmax(output, dim=1).min():.3f}, {torch.softmax(output, dim=1).max():.3f}]")
    
    # 4. 检查attention权重
    if attention:
        attn = attention[0][0]  # 第一层，第一个head
        print(f"Attention shape: {attn.shape}")
        print(f"Attention range: [{attn.min():.3f}, {attn.max():.3f}]")
    
    # 5. 建议的改进参数
    print(f"\n💡 Recommended Improvements:")
    print("="*50)
    
    if total_params > 20_000_000:  # > 20M parameters
        print("🔴 Model might be too large for CIFAR-10:")
        print("   → Try vit_tiny or reduce embed_dim")
        print("   → Use image_size=32 instead of 224")
        print("   → Increase batch_size to 128+")
    
    print("🎯 Suggested training parameters:")
    print("   → Learning rate: 1e-3 to 3e-4")
    print("   → Batch size: 128-256") 
    print("   → Epochs: 50-100")
    print("   → Weight decay: 0.01-0.05")
    print("   → Dropout: 0.0-0.1")
    print("   → Image size: 32 (native CIFAR-10)")
    print("   → Patch size: 4 (32/8=4 patches per side)")
    
    return {
        'total_params': total_params,
        'input_shape': data_batch.shape,
        'output_shape': output.shape
    }

def create_optimized_training_config():
    """创建针对CIFAR-10优化的配置"""
    config_small = {
        'model': 'custom',
        'image_size': 32,
        'patch_size': 4,
        'embed_dim': 192,
        'num_layers': 6,
        'num_heads': 3,
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'dropout': 0.0,
        'num_epochs': 30
    }
    
    config_medium = {
        'model': 'custom', 
        'image_size': 32,
        'patch_size': 4,
        'embed_dim': 256,
        'num_layers': 8,
        'num_heads': 4,
        'batch_size': 96,
        'lr': 5e-4,
        'weight_decay': 0.02,
        'dropout': 0.05,
        'num_epochs': 50
    }
    
    print("\n🚀 Optimized Training Commands:")
    print("="*50)
    
    print("💨 Fast training (small model, ~5M params):")
    cmd1 = f"python train.py --model custom --image_size {config_small['image_size']} --patch_size {config_small['patch_size']} --embed_dim {config_small['embed_dim']} --num_layers {config_small['num_layers']} --num_heads {config_small['num_heads']} --batch_size {config_small['batch_size']} --lr {config_small['lr']} --weight_decay {config_small['weight_decay']} --dropout {config_small['dropout']} --num_epochs {config_small['num_epochs']}"
    print(f"   {cmd1}")
    
    print(f"\n⚡ Balanced training (medium model, ~10M params):")
    cmd2 = f"python train.py --model custom --image_size {config_medium['image_size']} --patch_size {config_medium['patch_size']} --embed_dim {config_medium['embed_dim']} --num_layers {config_medium['num_layers']} --num_heads {config_medium['num_heads']} --batch_size {config_medium['batch_size']} --lr {config_medium['lr']} --weight_decay {config_medium['weight_decay']} --dropout {config_medium['dropout']} --num_epochs {config_medium['num_epochs']}"
    print(f"   {cmd2}")
    
    print(f"\n🎯 Expected accuracy: 70-80% (should reach >75%)")

if __name__ == "__main__":
    diagnose_model_training()
    create_optimized_training_config()