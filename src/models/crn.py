"""
Photographic Image Synthesis with Cascaded Reﬁnement Networks
https://arxiv.org/abs/1707.09405

Keypoint:
1. refinement from low-resolution to high-resolution
    1. Replace traditional encoder-decoder architecture with all upsampled layers.
    2. Everty module takes label as input
2. Use perceputual loss instead of MSE.
3. Diverse image generation by emit k images in one shot.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.losses import PerceptualLoss
from .base import BaseModel
import torchmetrics
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, last_layer=False) -> None:
        super().__init__()
        self.last_layer = last_layer
        if last_layer:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(output_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x):
        x = self.conv(x)
        if not self.last_layer:
            x = self.norm(x)
            x = self.lrelu(x)
        return x

class CRNBlock(nn.Module):
    def __init__(self, input_channels, middle_channels, output_channels, last_layer=False) -> None:
        super().__init__()
        self.conv1 = ConvBlock(input_channels, middle_channels)
        self.conv2 = ConvBlock(middle_channels, output_channels, last_layer=last_layer)
    
    def forward(self, x):
        """Receive upsampled feature and downsampled semantic map as input, output new features.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class CRN(nn.Module):
    def __init__(self, label_classes=36, resolution=1024, out_channels=3) -> None:
        super().__init__()
        self.resolution = resolution

        self.first_layer = CRNBlock(label_classes, self.get_channels(4), self.get_channels(4))
        self.middle_layer_res = [int(2**i) for i in range(3, int(np.log2(resolution)))]
        self.middel_layers = nn.ModuleList([CRNBlock(label_classes+self.get_channels(res/2), 
            self.get_channels(res), self.get_channels(res)) for res in self.middle_layer_res])
        self.last_layer = CRNBlock(label_classes+self.get_channels(resolution/2), self.get_channels(resolution), out_channels, last_layer=True)
    
    def get_channels(self, res):
        if res in [4, 8, 16, 32, 64]:
            return 1024
        if res in [128, 256]:
            return 512
        if res in [512]:
            return 128
        if res in [1024]:
            return 32

    def forward(self, label):
        down_label = F.interpolate(label, scale_factor= 4 / self.resolution, mode='nearest') # start from 4x8
        x = self.first_layer(down_label)

        for res, layer in zip(self.middle_layer_res, self.middel_layers) :
            x = F.upsample_bilinear(x, scale_factor=2)
            down_label = F.interpolate(label, scale_factor=res / self.resolution, mode='nearest')
            x = torch.cat([x, down_label], dim=1)
            x = layer(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = torch.cat([x, label], dim=1)
        img = self.last_layer(x)
        return img


class Model(BaseModel):
    def __init__(
        self,
        in_channels, # channels of input imagee
        out_channels,
        width, # width of input image
        height, # height of input image
        lrG: float = 0.001,
        b1: float = 0.5,
        b2: float = 0.999,
        input_normalize=True,
        optim="adam",
        scheduler=None,
        init_type="normal",
        eval_fid=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = CRN(resolution=height, label_classes=in_channels, out_channels=out_channels)
        self.init_weights(self.generator, init_type=init_type)

        self.perceptual_loss = PerceptualLoss()


    def forward(self, img_A):
        output = self.generator(img_A)
        output = output.reshape(
            img_A.shape[0], self.hparams.out_channels, self.hparams.height, self.hparams.width
        )
        return output

    def training_step(self, batch, batch_idx):
        real_A, real_B = batch["A"], batch["B"]  # (N, C, H, W)

        # generate images
        fake_B = self.generator(real_A)
        g_loss = self.perceptual_loss(fake_B, real_B)
        self.log("train_loss/g_loss", g_loss)
        
        if self.global_step % 200 == 0:
            # log sampled images
            self.log_images(fake_B, "train_images/fake_B")
            self.log_images(real_B, "train_images/real_B")
            # self.log_images(real_A, "train_images/real_A")

        return g_loss

    
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
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        if self.hparams.optim == "adam":
            opt_g = torch.optim.Adam(
                self.generator.parameters(), lr=lrG, betas=(b1, b2)
            )
        elif self.hparams.optim == "sgd":
            opt_g = torch.optim.SGD(self.generator.parameters(), lr=lrG)
        
        schedule_g = self.get_scheduler(opt_g)

        return [opt_g], [schedule_g]
