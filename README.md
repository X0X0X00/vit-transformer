# Computer Vision Research Assignment

This repository contains implementations for two main computer vision tasks:
1. **Vision Transformer (ViT)** implementation from scratch for CIFAR-10 classification
2. **U-Net Diffusion Model** for CIFAR-10 image generation

## 🏗️ Project Structure

```
cv-research-assignment/
├── vit_implementation/           # Vision Transformer implementation
│   ├── data/
│   │   └── dataset.py           # CIFAR-10 dataset loader
│   ├── model/
│   │   ├── patch_embed.py       # Patch and positional embedding
│   │   ├── attention.py         # Multi-head self-attention
│   │   ├── mlp.py              # MLP and MLP++ blocks
│   │   ├── transformer.py      # Standard transformer blocks
│   │   ├── transformer_plusplus.py  # Enhanced transformer blocks
│   │   └── vit.py              # Complete ViT model
│   ├── utils/
│   │   ├── training.py         # Training utilities
│   │   └── visualization.py    # Visualization tools
│   ├── train.py                # Main training script
│   ├── checkpoints/            # Model checkpoints
│   └── logs/                   # Tensorboard logs
│
├── unet_diffusion/              # U-Net Diffusion implementation
│   ├── model/
│   │   ├── unet.py             # U-Net architecture
│   │   └── diffusion.py        # Diffusion process
│   ├── utils/
│   │   ├── metrics.py          # FID and IS score calculation
│   │   └── visualization.py    # Visualization tools
│   ├── train_diffusion.py      # Main training script
│   ├── checkpoints/            # Model checkpoints
│   ├── logs/                   # Tensorboard logs
│   └── generated_samples/      # Generated images
│
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cv-research-assignment
```

2. **Create a conda environment:**
```bash
conda create -n cv-research python=3.8
conda activate cv-research
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt

# For CUDA support (if available)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Part 1: Vision Transformer Implementation

### Features
- **Complete ViT from scratch**: Patch embedding, positional encoding, multi-head attention
- **Transformer++**: Enhanced MLP with dual-branch architecture and gating mechanism
- **Comprehensive training pipeline**: Learning rate scheduling, checkpointing, tensorboard logging
- **Visualization tools**: Attention maps, feature space analysis, training curves

### Quick Start

```bash
cd vit_implementation

# Train ViT-Small with standard transformer
python train.py --model vit_small --batch_size 64 --num_epochs 100

# Train ViT-Small with Transformer++
python train.py --model vit_small --use_transformer_plusplus --batch_size 64 --num_epochs 100

# Train custom model
python train.py --model custom --embed_dim 256 --num_layers 8 --num_heads 8
```

### Key Arguments
- `--model`: Choose from `vit_small`, `vit_base`, `custom`
- `--use_transformer_plusplus`: Use enhanced Transformer++ blocks
- `--batch_size`: Training batch size (default: 64)
- `--num_epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 3e-4)
- `--image_size`: Input image size (default: 224)

### Model Architectures

| Model | Embed Dim | Layers | Heads | Parameters |
|-------|-----------|--------|-------|------------|
| ViT-Small | 384 | 12 | 6 | ~22M |
| ViT-Base | 768 | 12 | 12 | ~86M |

## 🎨 Part 2: U-Net Diffusion Model

### Features
- **U-Net architecture**: Skip connections, attention blocks, time embedding
- **DDPM and DDIM sampling**: Support for both sampling methods
- **Comprehensive evaluation**: FID and Inception Score calculation
- **Visualization tools**: Diffusion process visualization, sampling comparison

### Quick Start

```bash
cd unet_diffusion

# Train diffusion model
python train_diffusion.py --batch_size 128 --num_epochs 500

# Train with custom parameters
python train_diffusion.py --timesteps 1000 --channels 64 128 256 512 --time_dim 256
```

### Key Arguments
- `--batch_size`: Training batch size (default: 128)
- `--num_epochs`: Number of training epochs (default: 500)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--channels`: U-Net channel dimensions (default: [64, 128, 256, 512])
- `--ddim_steps`: DDIM sampling steps for evaluation (default: 50)

### Sampling Methods
- **DDPM**: Full 1000-step denoising process
- **DDIM**: Accelerated sampling with 50 steps (default)

## 📈 Monitoring Training

### Tensorboard
```bash
# For ViT training
tensorboard --logdir vit_implementation/logs

# For diffusion training
tensorboard --logdir unet_diffusion/logs
```

### Key Metrics
**ViT Training:**
- Training/Validation Loss and Accuracy
- Learning rate schedules
- Attention visualizations

**Diffusion Training:**
- Training loss (MSE between predicted and actual noise)
- FID Score (Fréchet Inception Distance)
- IS Score (Inception Score)
- Sample quality over time

## 🔧 Advanced Usage

### Vision Transformer Analysis
```python
from vit_implementation.model.vit import vit_small
from vit_implementation.utils.visualization import visualize_attention_maps

# Load trained model
model = vit_small()
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Visualize attention
visualize_attention_maps(model, image, layer_idx=-1, head_idx=0)
```

### Diffusion Sampling
```python
from unet_diffusion.model.diffusion import GaussianDiffusion
from unet_diffusion.model.unet import UNet

# Load trained model
unet = UNet()
diffusion = GaussianDiffusion(unet)

# Generate samples
samples = diffusion.sample(image_size=32, batch_size=16)
```

## 📊 Results and Evaluation

### Expected Performance

**Vision Transformer:**
- ViT-Small: ~85-90% accuracy on CIFAR-10
- ViT-Small++: ~87-92% accuracy on CIFAR-10
- Training time: ~2-3 hours on single GPU

**U-Net Diffusion:**
- FID Score: ~15-25 (lower is better)
- IS Score: ~6-8 (higher is better)
- Training time: ~12-24 hours on single GPU

### Transformer++ Improvements
- **Dual-branch MLP**: Combines GELU and Swish activations
- **Gating mechanism**: Learns optimal combination of branches
- **Performance gain**: ~2-5% accuracy improvement over standard ViT

## 🧪 Experiments to Try

1. **Transformer Architecture Comparison**
   - Compare standard ViT vs ViT++ on CIFAR-10
   - Analyze attention patterns and feature representations
   - Study the effect of different MLP configurations

2. **Diffusion Model Analysis**
   - Compare DDPM vs DDIM sampling quality and speed
   - Analyze the relationship between denoising steps and sample quality
   - Study FID/IS trends during training

3. **Hyperparameter Studies**
   - Learning rate schedules for both models
   - Different patch sizes for ViT
   - Various noise schedules for diffusion

## 🐛 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or model size
2. **Slow training**: Enable mixed precision, increase num_workers
3. **Poor convergence**: Check learning rate, data normalization

### Memory Optimization
```python
# Enable gradient checkpointing for ViT
model = VisionTransformer(...)
model.gradient_checkpointing_enable()

# Use smaller batch sizes for diffusion
train_diffusion.py --batch_size 64
```

## 📚 References

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
2. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
3. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
4. [Transformer++](https://arxiv.org/abs/2312.00752)

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Feel free to submit issues, feature requests, and pull requests!