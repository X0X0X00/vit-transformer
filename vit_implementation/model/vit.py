import torch
import torch.nn as nn
from .patch_embed import PatchEmbedding, PositionalEmbedding
from .transformer import TransformerEncoder
from .transformer_plusplus import TransformerEncoderPlusPlus

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
        use_transformer_plusplus=False
    ):
        super().__init__()
        
        # Adjust patch size if image is too small
        if image_size < patch_size:
            patch_size = image_size // 2
        
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        
        self.pos_embed = PositionalEmbedding(
            self.patch_embed.num_patches, embed_dim
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Choose between standard transformer or transformer++
        if use_transformer_plusplus:
            self.transformer = TransformerEncoderPlusPlus(
                embed_dim, num_layers, num_heads, mlp_ratio, dropout
            )
        else:
            self.transformer = TransformerEncoder(
                embed_dim, num_layers, num_heads, mlp_ratio, dropout
            )
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embedding
        x = self.pos_embed(x)
        x = self.dropout(x)
        
        # Transformer blocks
        x, attention_weights = self.transformer(x)
        
        # Classification token (first token)
        cls_token = x[:, 0]
        
        # Classification head
        logits = self.head(cls_token)
        
        return logits, attention_weights

def vit_small(num_classes=10, use_transformer_plusplus=False):
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        embed_dim=384,
        num_layers=12,
        num_heads=6,
        num_classes=num_classes,
        use_transformer_plusplus=use_transformer_plusplus
    )

def vit_base(num_classes=10, use_transformer_plusplus=False):
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_classes=num_classes,
        use_transformer_plusplus=use_transformer_plusplus
    )

def vit_large(num_classes=10, use_transformer_plusplus=False):
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        embed_dim=1024,
        num_layers=24,
        num_heads=16,
        num_classes=num_classes,
        use_transformer_plusplus=use_transformer_plusplus
    )