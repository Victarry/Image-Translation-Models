"""
Semantic Image Synthesis with Spatially-Adaptive Normalization.
https://arxiv.org/abs/1903.07291
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.networks.conv import MultiscaleDiscriminator, ResnetGenerator
from src.utils.losses import PerceptualLoss
from .base import BaseModel
import torchmetrics

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    
    def forward(self, x):
        return self.conv(x)

class SPADE(nn.Module):
    def __init__(self, label_channels, feature_channels) -> None:
        super().__init__()
        self.share_conv = Conv2d(label_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_scale = Conv2d(128, feature_channels, kernel_size=3, padding=1)
        self.conv_bias = Conv2d(128, feature_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(feature_channels)
    
    def forward(self, feature, label):
        feature = self.norm(feature)
        x = self.share_conv(label) 
        x = self.relu(x)

        scale = self.conv_scale(x) # (N, C, H, W)
        bias = self.conv_bias(x)
        feature = feature * scale + bias
        return feature

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, label_channels) -> None:
        super().__init__()
        self.conv = Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.spade = SPADE(label_channels, input_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, label):
        x = self.spade(x, label)
        x = self.relu(x)
        x = self.conv(x)
        return x

class SPADEBlock(nn.Module):
    def __init__(self, input_channels, middle_channels, output_channels, label_channels) -> None:
        super().__init__()
        self.conv1 = ConvBlock(input_channels, middle_channels, label_channels)
        self.conv2 = ConvBlock(middle_channels, output_channels, label_channels)
    
    def forward(self, x, label):
        """Receive upsampled feature and downsampled semantic map as input, output new features.
        """
        x = self.conv1(x, label)
        x = self.conv2(x, label)
        return x

class SPADEResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, label):
        pass

class Generator(nn.Module):
    def __init__(self, label_classes, height=256, width=512, out_channels=3) -> None:
        super().__init__()
        self.resolution = height
        self.width = width
        self.fc = nn.Linear(256, width*4//height*4*1024)

        self.middle_layer_res = [int(2**i) for i in range(2, int(np.log2(self.resolution)))]
        self.middel_layers = nn.ModuleList([SPADEBlock(self.get_channels(res), 
            self.get_channels(res), self.get_channels(res*2), label_classes) for res in self.middle_layer_res])
        self.last_layer = SPADEBlock(self.get_channels(self.resolution), self.get_channels(self.resolution), out_channels, label_classes)
    
    def get_channels(self, res):
        if res in [4, 8, 16]:
            return 1024
        elif res in [32]:
            return 512
        elif res in [64]:
            return 256
        elif res in [128]:
            return 128
        elif res in [256]:
            return 64
        else:
            print(res)
            raise NotImplementedError()

    def forward(self, label):
        N = label.shape[0]
        x = torch.randn(N, 256).type_as(label)
        x = self.fc(x).reshape(N, 1024, 4, -1)

        for res, layer in zip(self.middle_layer_res, self.middel_layers) :
            down_label = F.interpolate(label, scale_factor=res / self.resolution, mode='nearest')
            x = layer(x, down_label)
            x = F.upsample_bilinear(x, scale_factor=2)
        img = self.last_layer(x, label)
        return img


class Model(BaseModel):
    def __init__(
        self,
        in_channels, # channels of input imagee
        out_channels, # channels of output imagee
        width, # width of input image
        height, # height of input image
        loss_mode="hinge",
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
        self.generator = Generator(in_channels, height, width, out_channels)
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
            fake_B = self.forward(real_A)

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
            fake_B = self.forward(real_A)
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
