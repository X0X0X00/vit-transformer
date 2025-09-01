import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .mlp import MLPPlusPlus

class TransformerBlockPlusPlus(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Enhanced MLP with gating
        self.mlp = MLPPlusPlus(embed_dim, mlp_ratio, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Enhanced MLP with residual connection
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)
        
        return x, attn_weights

class TransformerEncoderPlusPlus(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlockPlusPlus(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
            
        x = self.norm(x)
        
        return x, attention_weights