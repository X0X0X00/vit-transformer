import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from PIL import Image

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = inception_v3(pretrained=True)
        else:
            inception = inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps"""
        outp = []
        x = inp

        if self.resize_input:
            x = nn.functional.interpolate(x,
                                        size=(299, 299),
                                        mode='bilinear',
                                        align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

class FIDScore:
    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = InceptionV3([3], normalize_input=False).to(device)
        self.inception_model.eval()

    def get_activations(self, images, batch_size=50):
        """Calculate activations for a batch of images"""
        self.inception_model.eval()
        
        n_images = len(images)
        n_batches = (n_images + batch_size - 1) // batch_size
        
        pred_arr = np.empty((n_images, 2048))
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_images)
            
            batch = images[start:end]
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch).to(self.device)
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
                
            with torch.no_grad():
                pred = self.inception_model(batch)[0]
                
            # If model output is not scalar, apply global average pooling
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start:end] = pred
            
        return pred_arr

    def calculate_statistics(self, images):
        """Calculate mean and covariance statistics for images"""
        activations = self.get_activations(images)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def calculate_fid(self, real_images, fake_images):
        """Calculate FID score between real and fake images"""
        mu1, sigma1 = self.calculate_statistics(real_images)
        mu2, sigma2 = self.calculate_statistics(fake_images)
        
        fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value

class InceptionScore:
    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()

    def calculate_is(self, images, batch_size=50, splits=10):
        """Calculate Inception Score"""
        N = len(images)
        
        # Get predictions
        preds = np.zeros((N, 1000))
        
        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size]
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch).to(self.device)
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            # Resize to 299x299 for Inception
            if batch.size(-1) != 299:
                batch = nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Normalize to [-1, 1] then to [0, 1]
            if batch.min() < 0:
                batch = (batch + 1) / 2
            
            with torch.no_grad():
                pred = self.inception_model(batch)
                pred = nn.functional.softmax(pred, dim=1)
                preds[i:i+batch_size] = pred.cpu().numpy()

        # Calculate IS
        split_scores = []
        
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
            
        return np.mean(split_scores), np.std(split_scores)

def entropy(p, q):
    return np.sum(p * np.log(p / q + 1e-8))

def prepare_images_for_metrics(images):
    """Prepare images for FID/IS calculation"""
    if isinstance(images, np.ndarray):
        if images.dtype != np.float32:
            images = images.astype(np.float32)
        # Normalize to [0, 1] if needed
        if images.max() > 1.0:
            images = images / 255.0
    elif isinstance(images, torch.Tensor):
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
    
    return images