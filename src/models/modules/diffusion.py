#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler

class Diffusion(nn.Module):
  def __init__(self, image_size: int, in_channels: int, out_channels: int, timesteps: int):
    super().__init__()
    self.model = UNet2DModel(
      sample_size = image_size,
      in_channels = in_channels,
      out_channels = out_channels,
      layers_per_block = 2,
      block_out_channels = (128, 256, 512),
      down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
      ),
      up_block_types=(
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
      ),
    )
    self.noise_scheduler = DDPMScheduler(num_train_timesteps = timesteps)
    self.timesteps = timesteps
    self.image_size = image_size
  def forward(self, noisy_image, timesteps):
    noise_pred = self.model(noisy_image, timesteps).sample # epsilon
    return noise_pred
  def sample(self):
    noise = torch.randn((1,3,self.image_size,self.image_size)).to(next(self.parameters()).device)
    input = noise
    for t in self.noise_scheduler.timesteps:
      with torch.no_grad():
        noisy_residual = self.model(input, t).sample # epsilon(x_t, t)
      previous_noisy_sample = self.noise_scheduler.step(noisy_residual, t, input).prev_sample # x_{t-1}
      input = previous_noisy_sample
    image = (input / 2 + 0.5).clamp(0,1).squeeze()
    image = torch.round(torch.permute(image, (1,2,0)) * 255).to(torch.uint8).cpu().numpy()[:,:,::-1]
    return image
