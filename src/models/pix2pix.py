"""
Traditional Unconditional GANs, with different loss modes including:
1. Binary Cross Entroy (vanilla_gan)
2. Least Square Error(lsgan)
"""
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from src.utils import utils
from .base import BaseModel
import torchmetrics


class Model(BaseModel):
    def __init__(
        self,
        channels,
        width,
        height,
        netG,
        netD,
        latent_dim=100,
        loss_mode="vanilla",
        lrG: float = 0.0002,
        lrD: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        input_normalize=True,
        optim="adam",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = hydra.utils.instantiate(
            netG, input_nc=channels, output_nc=channels
        )
        self.discriminator = hydra.utils.instantiate(
            netD, input_nc=2*channels
        )


    def forward(self, z):
        output = self.generator(z)
        output = output.reshape(
            z.shape[0], self.hparams.channels, self.hparams.height, self.hparams.width
        )
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch["A"], batch["B"]  # (N, C, H, W)

        if optimizer_idx == 0:
            # generate images
            fake_B = self.generator(real_A)

            fake_logit = self.discriminator(torch.cat((real_A, fake_B), dim=1)) # (N, 2C, H, W)
            valid = torch.ones_like(fake_logit)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(fake_logit, valid)
            self.log("train_loss/g_loss", g_loss, prog_bar=True)
        
            if self.global_step % 200 == 0:
                # log sampled images
                self.log_images(fake_B, "train_images/fake_B")
                self.log_images(real_B, "train_images/real_B")
                self.log_images(real_A, "train_images/real_A")

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # real loss
            real_pair = torch.cat((real_A, real_B), dim=1)
            real_logit = self.discriminator(real_pair)
            valid = torch.ones_like(real_logit)
            real_loss = F.mse_loss(real_logit, valid)

            # fake loss
            fake_B = self.generator(real_A)
            fake_pair = torch.cat((real_A, fake_B), dim=1)
            fake_logit = self.discriminator(fake_pair)
            fake = torch.zeros_like(fake_logit)

            fake_loss = self.adversarial_loss(fake_logit, fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("train_loss/d_loss", d_loss)
            self.log("train_log/real_logit", real_logit.mean())
            self.log("train_log/fake_logit", fake_logit.mean())

            return d_loss

    def adversarial_loss(self, y_hat, y):
        if self.hparams.loss_mode == "vanilla":
            return F.binary_cross_entropy_with_logits(y_hat, y)
        elif self.hparams.loss_mode == "lsgan":
            return F.mse_loss(y_hat, y)

    def configure_optimizers(self):
        lrG = self.hparams.lrG
        lrD = self.hparams.lrD
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        if self.hparams.optim == "adam":
            opt_g = torch.optim.Adam(
                self.generator.parameters(), lr=lrG, betas=(b1, b2)
            )
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(), lr=lrD, betas=(b1, b2)
            )
        elif self.hparams.optim == "sgd":
            opt_g = torch.optim.SGD(self.generator.parameters(), lr=lrG)
            opt_d = torch.optim.SGD(self.discriminator.parameters(), lr=lrD)
        return [opt_g, opt_d]
