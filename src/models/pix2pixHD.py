"""
High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
https://arxiv.org/abs/1711.11585

Keypoint:
1. Two-scale generator
    1. First train a global generator with 1024x512
    2. Then train a local enhancer network for 2048 x 1024
2. Multiscale discriminator.
    Downsample images and send to different discrminators.

Objective:
1. adversarial loss with multiscale discriminators
2. feature matching loss
3. perceptual loss
4. introduce instance information by instance boundary
5. Add instance feature encoder for controllable generation.
"""
import torch
import torch.nn.functional as F

from src.networks.conv import MultiscaleDiscriminator, ResnetGenerator
from src.utils.losses import PerceptualLoss
from .base import BaseModel
import torchmetrics


class Model(BaseModel):
    def __init__(
        self,
        in_channels, # channels of input imagee
        out_channels, # channels of output imagee
        width, # width of input image
        height, # height of input image
        loss_mode="vanilla",
        lrG: float = 0.0002,
        lrD: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        input_normalize=True,
        lambda_P=10,  # weight for perceptual loss
        lambda_FM=10,  # weight for feature matching loss
        optim="adam",
        scheduler=None,
        init_type="normal",
        eval_fid=False,
        **kwargs,
    ):
        # TODO: implement two stage traning with local enhancer
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = ResnetGenerator(in_channels, out_channels, norm_layer="instance")
        self.init_weights(self.generator, init_type=init_type)

        self.discriminator = MultiscaleDiscriminator(in_channels+out_channels, norm_layer="instance", getIntermFeat=True)
        self.init_weights(self.discriminator, init_type=init_type)

        self.perceptual_loss = PerceptualLoss()


    def forward(self, img_A):
        output = self.generator(img_A)
        output = output.reshape(
            img_A.shape[0], self.hparams.out_channels, self.hparams.height, self.hparams.width
        )
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        # TODO: disable automatic backward to reduce overhead brought by feature matching loss
        real_A, real_B = batch["A"], batch["B"]  # (N, C, H, W)

        # train generator
        if optimizer_idx == 0:
            # generate images
            fake_B = self.generator(real_A)

            all_fake_result = self.discriminator(torch.cat((real_A, fake_B), dim=1))
            all_real_result = self.discriminator(torch.cat((real_A, real_B), dim=1))


            fake_logits = [result[-1] for result in all_fake_result] # (N, 2C, H, W)
            adv_loss = []
            for fake_logit in fake_logits:
                valid = torch.ones_like(fake_logit)
                adv_loss.append(self.adversarial_loss(fake_logit, valid))
            # NOTE: use max instead of mean as in the original paper
            adv_loss = torch.stack(adv_loss).max()
            self.log("train_loss/g_adv_loss", adv_loss)
            g_loss = adv_loss

            if self.hparams.lambda_P > 0:
                p_loss = self.perceptual_loss(fake_B, real_B)
                g_loss = g_loss + self.hparams.lambda_P * p_loss

                self.log("train_loss/g_perceptual_loss", p_loss)

            if self.hparams.lambda_FM > 0:
                fm_loss = 0
                real_features = [feature for result in all_real_result for feature in result[:-1]]
                fake_features = [feature for result in all_fake_result for feature in result[:-1]]
                for real_feature, fake_feature in zip(real_features, fake_features):
                    fm_loss += F.l1_loss(real_feature, fake_feature)
                g_loss = g_loss + self.hparams.lambda_FM * fm_loss

            if self.global_step % 200 == 0:
                # log sampled images
                self.log_images(fake_B, "train_images/fake_B")
                self.log_images(real_B, "train_images/real_B")
                # self.log_images(real_A, "train_images/real_A")

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # real loss
            real_pair = torch.cat((real_A, real_B), dim=1)
            all_real_results = self.discriminator(real_pair)
            real_logits = [result[-1] for result in all_real_results]
            real_adv_loss = 0
            for real_logit in real_logits:
                valid = torch.ones_like(real_logit)
                real_adv_loss = real_adv_loss + self.adversarial_loss(real_logit, valid) 

            # fake loss
            fake_B = self.generator(real_A)
            fake_pair = torch.cat((real_A, fake_B), dim=1)
            all_fake_results = self.discriminator(fake_pair)
            fake_logits = [result[-1] for result in all_fake_results]

            fake_adv_loss = 0
            for fake_logit in fake_logits:
                fake = torch.zeros_like(fake_logit)
                fake_adv_loss = fake_adv_loss + self.adversarial_loss(fake_logit, fake)


            # discriminator loss is the average of these
            d_loss = (real_adv_loss + fake_adv_loss) / 2
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
            # self.log_images(real_A, "val_images/real_A")
 

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
