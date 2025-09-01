#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import argparse
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import get_dataloaders
from data.balanced_dataset import get_balanced_dataloaders
from model.vit import vit_small, vit_base, VisionTransformer
from utils.training import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR-10')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='vit_small', 
                       choices=['vit_small', 'vit_base', 'custom'],
                       help='Model architecture')
    parser.add_argument('--use_transformer_plusplus', action='store_true',
                       help='Use Transformer++ architecture')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=384,
                       help='Embedding dimension (only for custom model)')
    parser.add_argument('--num_layers', type=int, default=12,
                       help='Number of transformer layers (only for custom model)')
    parser.add_argument('--num_heads', type=int, default=6,
                       help='Number of attention heads (only for custom model)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Warmup epochs for cosine scheduler')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='',
                       help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--use_balanced_sampling', action='store_true',
                       help='Use balanced sampling for training')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    
    return parser.parse_args()

def create_model(args):
    if args.model == 'vit_small':
        model = vit_small(
            num_classes=10, 
            use_transformer_plusplus=args.use_transformer_plusplus
        )
    elif args.model == 'vit_base':
        model = vit_base(
            num_classes=10, 
            use_transformer_plusplus=args.use_transformer_plusplus
        )
    elif args.model == 'custom':
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_transformer_plusplus=args.use_transformer_plusplus
        )
    
    return model

def create_optimizer(model, args):
    return optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

def create_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        return CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs - args.warmup_epochs,
            eta_min=args.lr * 0.1
        )
    elif args.scheduler == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        return None

def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f'Using device: {device}')
    print(f'Arguments: {args}')
    
    # Create data loaders
    print('Loading data...')
    if args.use_balanced_sampling or args.no_augmentation:
        train_loader, val_loader = get_balanced_dataloaders(
            batch_size=args.batch_size,
            image_size=args.image_size,
            root_dir=args.data_dir,
            use_augmentation=not args.no_augmentation,
            use_balanced_sampling=args.use_balanced_sampling
        )
        if args.use_balanced_sampling:
            print("Using balanced sampling for training")
        if args.no_augmentation:
            print("Data augmentation disabled")
    else:
        train_loader, val_loader = get_dataloaders(
            batch_size=args.batch_size,
            image_size=args.image_size,
            root_dir=args.data_dir
        )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # Create model
    print('Creating model...')
    model = create_model(args)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,}')
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f'Resumed from epoch {start_epoch}')
    
    # Train model
    print('Starting training...')
    best_acc = trainer.train(args.num_epochs, args.save_every)
    
    print(f'Training completed! Best validation accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()