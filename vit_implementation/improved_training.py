#!/usr/bin/env python3
"""
改进的训练脚本，包含早停和正则化
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.balanced_dataset import get_balanced_dataloaders
from model.vit import VisionTransformer
from utils.training import Trainer

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

def train_with_early_stopping():
    print("🎯 训练优化的ViT模型 - 防止过拟合版本")
    print("="*60)
    
    # 优化的超参数
    config = {
        'image_size': 32,
        'patch_size': 4, 
        'embed_dim': 192,  # 减小模型容量
        'num_layers': 8,   # 适中的深度
        'num_heads': 6,
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 0.05,  # 增加正则化
        'dropout': 0.1,        # 适量dropout
        'num_epochs': 150,
        'patience': 10         # 早停耐心值
    }
    
    # 数据加载器
    print("📊 加载数据...")
    train_loader, val_loader = get_balanced_dataloaders(
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        use_augmentation=True,      # 使用数据增强
        use_balanced_sampling=True  # 平衡采样
    )
    
    # 模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  设备: {device}")
    
    model = VisionTransformer(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_classes=10
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🔧 模型参数: {num_params:,}")
    
    # 优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'],
        eta_min=config['lr'] * 0.01
    )
    
    # 早停
    early_stopping = EarlyStopping(
        patience=config['patience'], 
        min_delta=0.001,
        restore_best_weights=True
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir='./logs_improved',
        checkpoint_dir='./checkpoints_improved'
    )
    
    print("🚀 开始训练...")
    
    # 训练循环（带早停）
    best_val_acc = 0
    for epoch in range(config['num_epochs']):
        # 训练
        train_loss, train_acc = trainer.train_epoch(epoch)
        
        # 验证
        val_loss, val_acc = trainer.validate(epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        trainer.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
        trainer.writer.add_scalar('Train/Accuracy', train_acc, epoch)
        trainer.writer.add_scalar('Val/Loss_Epoch', val_loss, epoch)
        trainer.writer.add_scalar('Val/Accuracy', val_acc, epoch)
        trainer.writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # 打印结果
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.2e}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trainer.save_checkpoint(epoch, is_best=True)
            print(f'  💾 保存最佳模型 (Val Acc: {val_acc:.2f}%)')
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f"\n⏹️  早停触发! 在 Epoch {epoch+1}")
            print(f"🎯 最佳验证准确率: {best_val_acc:.2f}%")
            break
    
    trainer.writer.close()
    return best_val_acc

if __name__ == "__main__":
    final_acc = train_with_early_stopping()
    print(f"\n🎉 训练完成! 最终准确率: {final_acc:.2f}%")