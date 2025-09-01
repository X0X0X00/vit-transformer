import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class GaussianDiffusion:
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device='cuda'
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Create beta schedule
        self.betas = self.create_beta_schedule(beta_start, beta_end, timesteps).to(device)
        
        # Pre-compute values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def create_beta_schedule(self, beta_start, beta_end, timesteps):
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_diffusion_sample(self, x_0, t):
        """
        Takes an image and a timestep as input and returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(self.device) * x_0 + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise, noise
    
    def get_loss(self, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise, noise_pred)
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))
    
    @torch.no_grad()
    def ddim_sample(self, shape, eta=0.0, ddim_timesteps=50):
        """
        DDIM sampling for faster generation
        """
        device = next(self.model.parameters()).device
        
        # Create DDIM timesteps
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i, j in tqdm(zip(reversed(ddim_timestep_seq), reversed(ddim_timestep_prev_seq)), 
                        desc='DDIM sampling', total=len(ddim_timestep_seq)):
            
            t = torch.full((b,), i, device=device, dtype=torch.long)
            prev_t = torch.full((b,), j, device=device, dtype=torch.long)
            
            alpha_cumprod_t = self.get_index_from_list(self.alphas_cumprod, t, img.shape)
            alpha_cumprod_t_prev = self.get_index_from_list(self.alphas_cumprod, prev_t, img.shape)
            
            predicted_noise = self.model(img, t)
            
            # Direction pointing to x_t
            pred_x0 = (img - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Direction pointing towards predicted x_0
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta**2 * (1 - alpha_cumprod_t_prev / alpha_cumprod_t * (1 - alpha_cumprod_t))) * predicted_noise
            
            # Random noise
            noise = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)) * torch.randn_like(img)
            
            img = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + noise
        
        return img