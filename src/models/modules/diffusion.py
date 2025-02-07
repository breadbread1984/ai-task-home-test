#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet2DModel, DDIMScheduler

class Diffusion(nn.Module):
  def __init__(self, image_size: int, in_channels: int, out_channels: int, timesteps: int):
    super().__init__()
    self.model = UNet2DModel(
      sample_size = image_size,
      in_channels = in_channels,
      out_channels = out_channels,
      layers_per_block = 2,
      block_out_channels = (128, 256, 256, 256),
      down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        "DownBlock2D"
      ),
      up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D"
      ),
    )
    self.noise_scheduler = DDIMScheduler(num_train_timesteps = timesteps)
    self.noise_scheduler.set_timesteps(timesteps)
    self.image_size = image_size
  def forward(self, noisy_image, timesteps):
    noise_pred = self.model(noisy_image, timesteps).sample # epsilon
    return noise_pred
  def sample(self):
    noise = torch.randn((1,3,self.image_size,self.image_size)).to(next(self.parameters()).device)
    with torch.no_grad():
      for t in reversed(range(self.noise_scheduler.num_train_timesteps)):
        model_output = self.model(noise, t).sample # epsilon(x_t, t)
        noise = self.noise_scheduler.step(model_output, t, noise).prev_sample # x_{t-1}
    image = ((noise + 1.) * 127.5).squeeze(dim = 0).to(torch.uint8)
    image = torch.permute(image, (1,2,0)).cpu().numpy()
    return image
