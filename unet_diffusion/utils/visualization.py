import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
from PIL import Image
import seaborn as sns
from tqdm import tqdm

def denormalize_tensor(tensor):
    """Denormalize tensor from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2

def save_image_grid(images, save_path, nrow=8, normalize=True):
    """Save a grid of images"""
    if normalize:
        images = denormalize_tensor(images)
    
    grid = make_grid(images, nrow=nrow, normalize=False, padding=2)
    torchvision.utils.save_image(grid, save_path)
    return grid

def plot_diffusion_process(diffusion, x0, timesteps_to_show=None, save_path=None):
    """Visualize the forward diffusion process"""
    if timesteps_to_show is None:
        timesteps_to_show = [0, 100, 200, 300, 500, 700, 900, 999]
    
    device = x0.device
    batch_size = min(4, x0.shape[0])  # Show first 4 images
    
    fig, axes = plt.subplots(batch_size, len(timesteps_to_show), 
                           figsize=(2*len(timesteps_to_show), 2*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        image = x0[i:i+1]
        
        for j, t in enumerate(timesteps_to_show):
            t_tensor = torch.tensor([t], device=device)
            noisy_image, _ = diffusion.forward_diffusion_sample(image, t_tensor)
            
            # Convert to numpy and denormalize
            img_np = denormalize_tensor(noisy_image[0]).cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            
            axes[i, j].imshow(img_np)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f't={t}')
    
    plt.suptitle('Forward Diffusion Process')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_denoising_process(diffusion, num_images=4, save_path=None):
    """Visualize the reverse denoising process"""
    device = next(diffusion.model.parameters()).device
    
    # Start with pure noise
    x = torch.randn(num_images, 3, 32, 32, device=device)
    
    # Store intermediate results
    timesteps_to_show = [999, 800, 600, 400, 200, 100, 50, 0]
    stored_images = []
    
    diffusion.model.eval()
    with torch.no_grad():
        for i in tqdm(reversed(range(diffusion.timesteps)), desc='Denoising'):
            t = torch.full((num_images,), i, device=device, dtype=torch.long)
            x = diffusion.p_sample(x, t, i)
            
            if i in timesteps_to_show:
                stored_images.append(x.clone())
    
    # Plot results
    fig, axes = plt.subplots(num_images, len(timesteps_to_show), 
                           figsize=(2*len(timesteps_to_show), 2*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        for j, stored_img in enumerate(stored_images):
            img_np = denormalize_tensor(stored_img[i]).cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            
            axes[i, j].imshow(img_np)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f't={timesteps_to_show[j]}')
    
    plt.suptitle('Reverse Denoising Process')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, stored_images

def plot_loss_curves(losses, save_path=None):
    """Plot training loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compare_sampling_methods(diffusion, num_samples=16, save_path=None):
    """Compare different sampling methods (DDPM vs DDIM)"""
    device = next(diffusion.model.parameters()).device
    
    diffusion.model.eval()
    
    # DDPM sampling (full 1000 steps)
    print("Generating samples with DDPM (1000 steps)...")
    with torch.no_grad():
        ddpm_samples = torch.tensor(diffusion.sample(32, num_samples, 3)[-1])
    
    # DDIM sampling (50 steps)
    print("Generating samples with DDIM (50 steps)...")
    with torch.no_grad():
        ddim_samples = diffusion.ddim_sample(
            (num_samples, 3, 32, 32), 
            ddim_timesteps=50
        )
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # DDPM samples
    ddpm_grid = make_grid(denormalize_tensor(ddpm_samples), nrow=8, normalize=False, padding=2)
    axes[0].imshow(ddmp_grid.permute(1, 2, 0))
    axes[0].set_title('DDPM Sampling (1000 steps)')
    axes[0].axis('off')
    
    # DDIM samples
    ddim_grid = make_grid(denormalize_tensor(ddim_samples), nrow=8, normalize=False, padding=2)
    axes[1].imshow(ddim_grid.permute(1, 2, 0))
    axes[1].set_title('DDIM Sampling (50 steps)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ddpm_samples, ddim_samples

def plot_fid_is_curves(fid_scores, is_scores, epochs, save_path=None):
    """Plot FID and IS score curves over training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # FID scores (lower is better)
    ax1.plot(epochs, fid_scores, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('FID Score')
    ax1.set_title('FID Score Over Training (Lower is Better)')
    ax1.grid(True, alpha=0.3)
    
    # IS scores (higher is better)
    is_means = [score[0] for score in is_scores]
    is_stds = [score[1] for score in is_scores]
    
    ax2.errorbar(epochs, is_means, yerr=is_stds, fmt='r-', linewidth=2, marker='o', capsize=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Inception Score')
    ax2.set_title('Inception Score Over Training (Higher is Better)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def analyze_denoising_steps(diffusion, num_steps_list=[10, 20, 50, 100, 200], 
                           num_samples=8, save_path=None):
    """Analyze the effect of different numbers of denoising steps"""
    device = next(diffusion.model.parameters()).device
    
    results = {}
    diffusion.model.eval()
    
    for num_steps in num_steps_list:
        print(f"Generating samples with {num_steps} denoising steps...")
        with torch.no_grad():
            samples = diffusion.ddim_sample(
                (num_samples, 3, 32, 32),
                ddim_timesteps=num_steps
            )
        results[num_steps] = samples
    
    # Create comparison plot
    fig, axes = plt.subplots(len(num_steps_list), 1, 
                           figsize=(12, 3*len(num_steps_list)))
    
    for i, num_steps in enumerate(num_steps_list):
        samples = results[num_steps]
        grid = make_grid(denormalize_tensor(samples), nrow=8, normalize=False, padding=2)
        
        axes[i].imshow(grid.permute(1, 2, 0))
        axes[i].set_title(f'{num_steps} Denoising Steps')
        axes[i].axis('off')
    
    plt.suptitle('Effect of Number of Denoising Steps on Sample Quality')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, results

def create_training_summary_diffusion(trainer_logs, save_path=None):
    """Create comprehensive training summary for diffusion model"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    if 'train_losses' in trainer_logs:
        epochs = range(1, len(trainer_logs['train_losses']) + 1)
        axes[0, 0].plot(epochs, trainer_logs['train_losses'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # FID scores
    if 'fid_scores' in trainer_logs and 'fid_epochs' in trainer_logs:
        axes[0, 1].plot(trainer_logs['fid_epochs'], trainer_logs['fid_scores'], 
                       'r-', linewidth=2, marker='o')
        axes[0, 1].set_title('FID Score (Lower is Better)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('FID')
        axes[0, 1].grid(True, alpha=0.3)
    
    # IS scores
    if 'is_scores' in trainer_logs and 'is_epochs' in trainer_logs:
        is_means = [score[0] for score in trainer_logs['is_scores']]
        is_stds = [score[1] for score in trainer_logs['is_scores']]
        axes[1, 0].errorbar(trainer_logs['is_epochs'], is_means, yerr=is_stds,
                           fmt='g-', linewidth=2, marker='o', capsize=5)
        axes[1, 0].set_title('Inception Score (Higher is Better)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IS')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Combined FID and IS on same plot (normalized)
    if all(key in trainer_logs for key in ['fid_scores', 'is_scores', 'fid_epochs']):
        # Normalize scores for comparison
        fid_norm = np.array(trainer_logs['fid_scores'])
        fid_norm = (fid_norm - fid_norm.min()) / (fid_norm.max() - fid_norm.min())
        
        is_means = np.array([score[0] for score in trainer_logs['is_scores']])
        is_norm = (is_means - is_means.min()) / (is_means.max() - is_means.min())
        is_norm = 1 - is_norm  # Invert so lower is better (like FID)
        
        axes[1, 1].plot(trainer_logs['fid_epochs'], fid_norm, 'r-', 
                       linewidth=2, marker='o', label='FID (normalized)')
        axes[1, 1].plot(trainer_logs['is_epochs'], is_norm, 'g-', 
                       linewidth=2, marker='s', label='IS (inverted, normalized)')
        axes[1, 1].set_title('Normalized Metrics Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Normalized Score (Lower is Better)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Diffusion Model Training Summary', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig