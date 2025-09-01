import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding using convolution
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.shape[0]
        
        # Apply patch embedding
        x = self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        
        # Flatten patches
        x = rearrange(x, 'b e h w -> b (h w) e')
        
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # Learnable positional embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Expand class token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate class token with patches
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.position_embedding
        
        return x