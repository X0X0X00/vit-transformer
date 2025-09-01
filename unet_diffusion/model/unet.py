import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).chunk(3, dim=1)
        q, k, v = qkv
        
        # Reshape for multi-head attention
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)
        
        # Attention
        attn = torch.softmax(torch.matmul(q.transpose(-1, -2), k) / math.sqrt(c // self.num_heads), dim=-1)
        out = torch.matmul(v, attn.transpose(-1, -2))
        
        # Reshape back
        out = out.view(b, c, h, w)
        out = self.out(out)
        
        return x + out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_attention=False):
        super().__init__()
        
        self.resnet1 = ResNetBlock(in_channels, out_channels, time_dim)
        self.resnet2 = ResNetBlock(out_channels, out_channels, time_dim)
        
        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
            
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x, time_emb):
        h = self.resnet1(x, time_emb)
        h = self.resnet2(h, time_emb)
        h = self.attention(h)
        
        return self.downsample(h), h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_attention=False):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
        self.resnet1 = ResNetBlock(in_channels + out_channels, out_channels, time_dim)
        self.resnet2 = ResNetBlock(out_channels, out_channels, time_dim)
        
        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        
        x = self.resnet1(x, time_emb)
        x = self.resnet2(x, time_emb)
        x = self.attention(x)
        
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_dim=256,
        channels=[64, 128, 256, 512],
        use_attention=[False, True, True, False]
    ):
        super().__init__()
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        
        # Encoder (downsampling)
        self.downs = nn.ModuleList()
        in_ch = channels[0]
        for i, (out_ch, use_attn) in enumerate(zip(channels, use_attention)):
            if i == 0:
                # First block doesn't change channels
                self.downs.append(DownBlock(in_ch, in_ch, time_dim, use_attn))
            else:
                self.downs.append(DownBlock(in_ch, out_ch, time_dim, use_attn))
                in_ch = out_ch
        
        # Middle block
        self.mid_block1 = ResNetBlock(channels[-1], channels[-1], time_dim)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResNetBlock(channels[-1], channels[-1], time_dim)
        
        # Decoder (upsampling)
        self.ups = nn.ModuleList()
        channels_reversed = list(reversed(channels))
        use_attention_reversed = list(reversed(use_attention))
        
        for i, (out_ch, use_attn) in enumerate(zip(channels_reversed, use_attention_reversed)):
            if i == 0:
                # First up block
                self.ups.append(UpBlock(channels[-1], out_ch, time_dim, use_attn))
            else:
                in_ch = channels_reversed[i-1]
                self.ups.append(UpBlock(in_ch, out_ch, time_dim, use_attn))
        
        # Final convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, time):
        # Time embedding
        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)
        
        # Initial conv
        x = self.conv_in(x)
        
        # Store skip connections
        skips = [x]
        
        # Encoder
        for down in self.downs:
            x, skip = down(x, time_emb)
            skips.append(skip)
        
        # Middle
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)
        
        # Decoder
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, time_emb)
        
        # Final conv
        x = self.conv_out(x)
        
        return x