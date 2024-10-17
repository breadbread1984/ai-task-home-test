#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler

class Diffusion(nn.Module):
  def __init__(self, image_size: int, in_channels: int, out_channels: int):
    super().__init__()
    self.model = UNet2DModel(
      sample_size = image_size,
      in_channels = in_channels,
      out_channels = out_channels,
      layers_per_block = 2,
      block_out_channels = (64, 128, 256),
    )
    self.noise_scheduler = DDPMScheduler(num_train_timesteps = 1000)
  def forward(self, noisy_image, timesteps):
    noise_pred = self.model(noisy_image, timesteps).sample # epsilon
    return noise_pred
