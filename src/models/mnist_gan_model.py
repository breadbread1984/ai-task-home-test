from typing import Union, Dict, Any, Tuple, Optional

import wandb
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = torch.nn.BCELoss()

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        gen_dict, loss = self.step(batch, batch_idx, 0)
        dist_dict, loss = self.step(batch, batch_idx, 1)
        log_dict = gen_dict
        log_dict.update(dist_dict)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        real, y = batch
        fake = self.generator(noise, y)
        pred_fake = self.discriminator(fake, y)
        pred_real = self.discriminator(real, y)
        gen_loss = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake, device = pred_fake.device))
        dis_loss = self.adversarial_loss(pred_real, torch.ones_like(pred_real, device = pred_real.device)) + \
                   self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake, device = pred_fake.device))
        log_dict = {'gen_loss': gen_loss, 'dis_loss': dis_loss}
        self.log_dict({"/".join(("test, k")): v for k, v in log_dict.items()})
        return log_dict

    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        # TODO: implement the step method of the GAN model.
        #     : This function should return both a dictionary of losses
        #     : and current loss of the network being optimised.
        #     :
        #     : When training with pytorch lightning, because we defined 2 optimizers in
        #     : the `configure_optimizers` function above, we use the `optimizer_idx` parameter
        #     : to keep a track of which network is being optimised.

        imgs, labels = batch
        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # TODO: Create adversarial ground truths
        real, y = batch

        # TODO: Create noise and labels for generator input
        noise = torch.normal(mean = 0, std = 1, size = (real.shape[0], 64)).to(next(self.parameters()).device)

        if optimizer_idx == 0 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the generator
            # HINT: when optimizer_idx == 0 the model is optimizing the generator

            # TODO: Generate a batch of images
            fake = self.generator(noise, y)

            # TODO: Calculate loss to measure generator's ability to fool the discriminator
            pred = self.discriminator(fake, y)
            loss = self.adversarial_loss(pred, torch.ones_like(pred, device = pred.device))

        if optimizer_idx == 1 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the discriminator
            # HINT: when optimizer_idx == 1 the model is optimizing the discriminator

            # TODO: Generate a batch of images
            fake = self.generator(noise, y)

            # TODO: Calculate loss for real images
            pred_real = self.discriminator(real, y)

            # TODO: Calculate loss for fake images
            pred_fake = self.discriminator(fake.detach(), y)

            # TODO: Calculate total discriminator loss
            loss = self.adversarial_loss(
              torch.cat([pred_real, pred_fake], dim = 0),
              torch.cat([torch.ones_like(pred_real, device = pred_real.device),
                         torch.zeros_like(pred_fake, device = pred_fake.device)], dim = 0))
        log_dict['gen_loss' if optimizer_idx == 0 else 'dis_loss'] = loss

        return log_dict, loss

    def on_epoch_end(self):
        # TODO: implement functionality to log predicted images to wandb
        #     : at the end of each epoch

        # TODO: Create fake images
        noise = torch.normal(mean = 0, std = 1, size = (10, 64)).to(next(self.parameters()).device)
        y = torch.arange(10, dtype = torch.int32).to(next(self.parameters()).device)
        self.generator.eval()
        with torch.no_grad():
          fake = self.generator(noise, y).cpu().permute(0,2,3,1).numpy() # fake.shape = (10,32,32,1)
          fake = np.reshape(np.transpose(np.reshape(fake, (2,5,32,32,1)), (0,2,1,3,4)), (2*32,5*32,1))
          img = ((fake + 1) / 2 * 255).astype(np.uint8)

        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                # TODO: log fake images to wandb (https://docs.wandb.ai/guides/track/log/media)
                #     : replace `None` with your wandb Image object
                logger.experiment.log({"gen_imgs": [wandb.Image(img, caption = f"Epoch {self.current_epoch}")]})
