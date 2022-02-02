import io

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from pytorch_lightning import LightningModule
from src.utils.utils import get_logger
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.transforms import ToTensor


class BaseModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.console = get_logger()

    def get_grid_images(self, imgs, nimgs=64, nrow=8):
        imgs = imgs.reshape(
            -1, self.hparams.channels, self.hparams.height, self.hparams.width
        )
        if self.hparams.input_normalize:
            grid = torchvision.utils.make_grid(
                imgs[:nimgs], nrow=nrow, normalize=True, value_range=(-1, 1)
            )
        else:
            grid = torchvision.utils.make_grid(imgs[:nimgs], normalize=False, nrow=nrow)
        return grid
    
    def log_hist(self, tensor, name):
        assert tensor.dim() == 1
        array = np.array(tensor.detach().cpu().numpy())
        self.logger.experiment.add_histogram(name, array, self.global_step)

    def log_images(self, imgs, name, nimgs=64, nrow=8):
        grid = self.get_grid_images(imgs, nimgs=nimgs, nrow=nrow)
        self.logger.experiment.add_image(name, grid, self.global_step)

    def image_float2int(self, imgs):
        if self.hparams.input_normalize:
            imgs = (imgs + 1) / 2
        imgs = (imgs * 255).to(torch.uint8)
        return imgs
    
    def tensor_to_array(self, *tensors):
        output = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                output.append(np.array(tensor.detach().cpu().numpy()))
            else:
                output.append(tensor)
        return output

    def plot_scatter(self, name, x, y, c=None, s=None, xlim=None, ylim=None):
        x, y, c, s = self.tensor_to_array(x, y, c, s)

        plt.figure()
        plt.scatter(x=x, y=y, s=s, c=c, cmap="tab10", alpha=1)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.title("Latent distribution")
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()
        buf.seek(0)
        visual_image = ToTensor()(PIL.Image.open(buf))
        self.logger.experiment.add_image(name, visual_image, self.global_step)

    def get_scheduler(self, optimizer):
        """Return a learning rate scheduler

        Parameters:
            optimizer          -- the optimizer of the network
            opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                                opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

        For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
        and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
        """
        opt = self.hparams.scheduler
        if opt.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

    def init_weights(self, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>