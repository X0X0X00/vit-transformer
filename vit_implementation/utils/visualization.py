import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from PIL import Image
import torchvision.transforms as transforms

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def visualize_attention_maps(model, image, layer_idx=-1, head_idx=0, patch_size=16, save_path=None):
    """Visualize attention maps from Vision Transformer"""
    model.eval()
    
    with torch.no_grad():
        # Get model output and attention weights
        logits, attention_weights = model(image.unsqueeze(0))
        
        # Get attention from specified layer and head
        attention = attention_weights[layer_idx][0, head_idx]  # [num_patches + 1, num_patches + 1]
        
        # Remove class token attention
        attention = attention[1:, 1:]  # [num_patches, num_patches]
        
        # Compute attention to each patch
        attention_mean = attention.mean(dim=0)  # Average attention received by each patch
        
        # Reshape attention to spatial dimensions
        num_patches = int(np.sqrt(attention_mean.shape[0]))
        attention_map = attention_mean.reshape(num_patches, num_patches)
        
        # Upsample attention map to image size
        attention_map = attention_map.cpu().numpy()
        image_size = image.shape[-1]
        attention_resized = cv2.resize(attention_map, (image_size, image_size))
        
        # Normalize image for visualization
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        im1 = axes[1].imshow(attention_resized, cmap='viridis')
        axes[1].set_title(f'Attention Map (Layer {layer_idx}, Head {head_idx})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        axes[2].imshow(image_np)
        axes[2].imshow(attention_resized, alpha=0.6, cmap='jet')
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    return fig, attention_map

def plot_attention_heatmap(attention_weights, layer_idx=-1, save_path=None):
    """Plot attention heatmap for all heads in a layer"""
    attention = attention_weights[layer_idx][0]  # [num_heads, num_patches + 1, num_patches + 1]
    num_heads = attention.shape[0]
    
    # Remove class token for cleaner visualization
    attention = attention[:, 1:, 1:]  # [num_heads, num_patches, num_patches]
    
    # Create subplot for each head
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for head in range(num_heads):
        row = head // cols
        col = head % cols
        
        # Average attention for this head
        head_attention = attention[head].mean(dim=0).cpu().numpy()
        num_patches = int(np.sqrt(head_attention.shape[0]))
        head_attention = head_attention.reshape(num_patches, num_patches)
        
        im = axes[row, col].imshow(head_attention, cmap='viridis')
        axes[row, col].set_title(f'Head {head}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col])
    
    # Hide empty subplots
    for head in range(num_heads, rows * cols):
        row = head // cols
        col = head % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Attention Heatmaps - Layer {layer_idx}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def visualize_feature_space(model, dataloader, num_samples=1000, save_path=None):
    """Visualize feature space using t-SNE"""
    model.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if len(features) >= num_samples:
                break
                
            data = data.cuda() if torch.cuda.is_available() else data
            
            # Get features before classification head
            x = model.patch_embed(data)
            x = model.pos_embed(x)
            x = model.dropout(x)
            x, _ = model.transformer(x)
            
            # Use class token as feature representation
            cls_features = x[:, 0].cpu().numpy()
            
            features.append(cls_features)
            labels.append(target.numpy())
    
    # Concatenate all features and labels
    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Feature Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Add class names if available
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i/10), 
                         markersize=8) for i in range(10)]
    plt.legend(handles, class_names, title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return features_2d

def compare_models_performance(results_dict, save_path=None):
    """Compare performance of different models"""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    accuracies = [results_dict[model]['val_acc'] for model in models]
    axes[0].bar(models, accuracies, color=['blue', 'red', 'green', 'orange'][:len(models)])
    axes[0].set_ylabel('Validation Accuracy (%)')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    # Loss comparison
    losses = [results_dict[model]['val_loss'] for model in models]
    axes[1].bar(models, losses, color=['blue', 'red', 'green', 'orange'][:len(models)])
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Model Loss Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(losses):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_patch_embeddings(model, save_path=None):
    """Visualize patch embeddings using PCA"""
    # Get patch embedding weights
    patch_embed_weight = model.patch_embed.projection.weight.data.cpu().numpy()
    
    # Reshape to image patches
    patch_size = model.patch_embed.patch_size
    in_channels = model.patch_embed.in_channels
    embed_dim = model.patch_embed.embed_dim
    
    # Reshape weights to (embed_dim, channels, patch_size, patch_size)
    patches = patch_embed_weight.reshape(embed_dim, in_channels, patch_size, patch_size)
    
    # Apply PCA to reduce dimensionality
    patches_flat = patches.reshape(embed_dim, -1)
    pca = PCA(n_components=2)
    patches_2d = pca.fit_transform(patches_flat)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(patches_2d[:, 0], patches_2d[:, 1], 
                         c=range(embed_dim), cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Embedding Index')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA of Patch Embeddings')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return patches_2d

def create_training_summary(trainer, save_path=None):
    """Create a comprehensive training summary plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # Training loss
    axes[0, 0].plot(epochs, trainer.train_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Validation loss
    axes[0, 1].plot(epochs, trainer.val_losses, 'r-', linewidth=2)
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Training accuracy
    axes[1, 0].plot(epochs, trainer.train_accuracies, 'b-', linewidth=2)
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(True)
    
    # Validation accuracy
    axes[1, 1].plot(epochs, trainer.val_accuracies, 'r-', linewidth=2)
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].grid(True)
    
    plt.suptitle('Training Summary', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig