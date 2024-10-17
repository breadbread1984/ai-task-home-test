#!/usr/bin/python3

import wandb
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class CIFAR10DiffusionModule(LightningModule):
  def __init__(self,
               model: nn.Module,
               **kwargs
  ):
    super().__init__()
    self.save_hyperparameters()

    self.model = model
    self.criterion = torch.nn.MSELoss()

  def forward(self, x, t):
    self.model(x, t)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr = self.hparams.lr)

  def training_step(self, batch, batch_idx):
    log_dict, loss = self.step(batch, batch_idx)
    self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
    return loss

  def validation_step(self, batch, batch_idx):
    log_dict, loss = self.step(batch, batch_idx)
    self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
    return None

  def test_step(self, batch, batch_idx):
    log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
    self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
    return None

  def step(self, batch, batch_idx):
    log_dict = {}
    x, _ = batch # train non conditional diffusion, use no class
    t = torch.randint(0, self.model.timesteps, (x.size(0),), device = next(self.parameters()).device).long()
    noise = torch.randn_like(x)
    x_noisy = x + noise * t[:, None, None, None] # forward process
    output = self.model(x_noisy, t)
    loss = self.criterion(output, noise)
    log_dict['loss'] = loss
    return log_dict, loss

  def on_epoch_end(self):
    self.model.eval()
    with torch.no_grad():
      image = self.model.sample()
      np.save('sample.npy', image)
    for logger in self.trainer.logger:
      if type(logger).__name__ == "WandbLogger":
        logger.experiment.log({"gen_imgs": [wandb.Image(image, caption = f"Epoch {self.current_epoch}")]})
