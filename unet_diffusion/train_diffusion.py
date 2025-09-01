#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.unet import UNet
from model.diffusion import GaussianDiffusion
from utils.metrics import FIDScore, InceptionScore, prepare_images_for_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net Diffusion Model on CIFAR-10')
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=32, help='Image size (CIFAR-10 is 32x32)')
    parser.add_argument('--channels', nargs='+', type=int, default=[64, 128, 256, 512], 
                       help='Channel dimensions for U-Net')
    parser.add_argument('--time_dim', type=int, default=256, help='Time embedding dimension')
    
    # Diffusion parameters
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start value')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end value')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='./logs', help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--sample_dir', type=str, default='./generated_samples', help='Generated samples directory')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=20, help='Generate samples every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    
    # Sampling parameters
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM sampling steps')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    
    return parser.parse_args()

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return dataloader

def save_samples(samples, epoch, sample_dir, nrow=4):
    os.makedirs(sample_dir, exist_ok=True)
    
    # Convert to tensor if numpy
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)
    
    # Normalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Save grid
    grid = torchvision.utils.make_grid(samples, nrow=nrow, normalize=True)
    save_path = os.path.join(sample_dir, f'samples_epoch_{epoch}.png')
    torchvision.utils.save_image(grid, save_path)
    
    return save_path

def train_epoch(model, diffusion, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        batch_size = data.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
        
        # Calculate loss
        loss = diffusion.get_loss(data, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_model(diffusion, real_dataloader, device, num_samples=1000, ddim_steps=50):
    """Evaluate model using FID and IS scores"""
    diffusion.model.eval()
    
    # Generate fake samples
    print("Generating samples for evaluation...")
    fake_samples = []
    num_batches = (num_samples + 16 - 1) // 16  # Generate in batches of 16
    
    for _ in tqdm(range(num_batches), desc="Generating samples"):
        with torch.no_grad():
            batch_size = min(16, num_samples - len(fake_samples) * 16)
            if batch_size <= 0:
                break
                
            samples = diffusion.ddim_sample(
                shape=(batch_size, 3, 32, 32),
                ddim_timesteps=ddim_steps
            )
            fake_samples.append(samples)
    
    fake_samples = torch.cat(fake_samples, dim=0)[:num_samples]
    
    # Get real samples
    print("Collecting real samples...")
    real_samples = []
    for batch_idx, (data, _) in enumerate(real_dataloader):
        real_samples.append(data)
        if len(real_samples) * data.shape[0] >= num_samples:
            break
    
    real_samples = torch.cat(real_samples, dim=0)[:num_samples]
    
    # Prepare for metrics (normalize to [0, 1])
    fake_samples = (fake_samples + 1) / 2
    real_samples = (real_samples + 1) / 2
    
    fake_samples = prepare_images_for_metrics(fake_samples)
    real_samples = prepare_images_for_metrics(real_samples)
    
    # Calculate FID
    fid_calculator = FIDScore(device=device)
    fid_score = fid_calculator.calculate_fid(real_samples, fake_samples)
    
    # Calculate IS
    is_calculator = InceptionScore(device=device)
    is_mean, is_std = is_calculator.calculate_is(fake_samples)
    
    return fid_score, is_mean, is_std

def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f'Using device: {device}')
    print(f'Arguments: {args}')
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Create dataloader
    print('Loading data...')
    dataloader = get_dataloader(args)
    print(f'Dataset size: {len(dataloader.dataset)}')
    
    # Create model
    print('Creating model...')
    model = UNet(
        in_channels=3,
        out_channels=3,
        time_dim=args.time_dim,
        channels=args.channels,
        use_attention=[False, True, True, False]  # Use attention in middle layers
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,}')
    
    # Create diffusion process
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    print('Starting training...')
    best_fid = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        # Training
        avg_loss = train_epoch(model, diffusion, dataloader, optimizer, device, epoch)
        
        # Logging
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        print(f'Epoch {epoch+1}/{args.num_epochs}: Loss = {avg_loss:.4f}')
        
        # Generate samples
        if (epoch + 1) % args.sample_every == 0:
            print('Generating samples...')
            model.eval()
            with torch.no_grad():
                samples = diffusion.ddim_sample(
                    shape=(args.num_samples, 3, args.image_size, args.image_size),
                    ddim_timesteps=args.ddim_steps
                )
            
            # Save samples
            save_path = save_samples(samples, epoch + 1, args.sample_dir)
            print(f'Samples saved to {save_path}')
        
        # Evaluate model
        if (epoch + 1) % (args.sample_every * 2) == 0:  # Less frequent evaluation
            fid_score, is_mean, is_std = evaluate_model(
                diffusion, dataloader, device, 
                num_samples=1000, ddim_steps=args.ddim_steps
            )
            
            writer.add_scalar('Eval/FID', fid_score, epoch)
            writer.add_scalar('Eval/IS_mean', is_mean, epoch)
            writer.add_scalar('Eval/IS_std', is_std, epoch)
            
            print(f'FID: {fid_score:.2f}, IS: {is_mean:.2f} ± {is_std:.2f}')
            
            # Save best model
            if fid_score < best_fid:
                best_fid = fid_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fid_score': fid_score,
                    'is_mean': is_mean,
                    'is_std': is_std,
                }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
    
    print(f'Training completed! Best FID: {best_fid:.2f}')
    writer.close()

if __name__ == '__main__':
    main()