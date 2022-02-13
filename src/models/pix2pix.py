"""
Image-to-Image Translation with Conditional Adversarial Networks.
https://arxiv.org/abs/1611.07004
"""
import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .base import BaseModel
import torchmetrics


class Model(BaseModel):
    def __init__(
        self,
        in_channels, # channels of input imagee
        out_channels, # channels of input imagee
        width, # width of input image
        height, # height of input image
        netG, # config dict of generator
        netD, # config dict of discriminator
        loss_mode="vanilla",
        lrG: float = 0.0002,
        lrD: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        input_normalize=True,
        lambda_L1=100, 
        optim="adam",
        scheduler=None,
        init_type="normal",
        eval_fid=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = hydra.utils.instantiate(netG)
        self.init_weights(self.generator, init_type=init_type)

        self.discriminator = hydra.utils.instantiate(netD)
        self.init_weights(self.discriminator, init_type=init_type)


    def forward(self, img_A):
        output = self.generator(img_A)
        output = output.reshape(
            img_A.shape[0], self.hparams.out_channels, self.hparams.height, self.hparams.width
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
            self.log("train_loss/g_adv_loss", g_loss)

            if self.hparams.lambda_L1 > 0:
                recon_loss =  F.l1_loss(fake_B, real_B)
                g_loss = g_loss + self.hparams.lambda_L1 * recon_loss
                self.log("train_loss/g_recon_loss", recon_loss)
        
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
    
    def on_validation_epoch_start(self) -> None:
        if self.hparams.eval_fid:
            self.fid = torchmetrics.FID().to(self.device)

    def validation_step(self, batch, batch_idx):
        real_A, real_B = batch["A"], batch["B"]
        fake_B = self.forward(real_A)
        if self.hparams.eval_fid:
            self.fid.update(self.image_float2int(real_B), real=True)
            self.fid.update(self.image_float2int(fake_B), real=False)
        if batch_idx == 0:
            self.log_images(fake_B, "val_images/fake_B")
            self.log_images(real_B, "val_images/real_B")
            self.log_images(real_A, "val_images/real_A")
 

    def on_validation_epoch_end(self):
        if self.hparams.eval_fid:
            self.log("metrics/fid", self.fid.compute())

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
        
        schedule_g = self.get_scheduler(opt_g)
        schedule_d = self.get_scheduler(opt_d)

        return [opt_g, opt_d], [schedule_g, schedule_d]
