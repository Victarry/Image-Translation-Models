from telnetlib import IP
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def get_interpolation_method(method):
    if method == 'nearest':
        return transforms.InterpolationMode.NEAREST
    elif method == 'bicubic':
        return transforms.InterpolationMode.BICUBIC
    elif method == 'bilinear':
        return transforms.InterpolationMode.BILINEAR

def get_transform(config, convert=True):
    transform_list = []
    if 'grayscale' in config:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in config:
        osize = [config.resize.height, config.resize.width]
        if 'method' not in config.resize:
            method = transforms.InterpolationMode.BICUBIC
        else:
            method = get_interpolation_method(config.resize.method)
        transform_list.append(transforms.Resize(osize, method))

    if 'crop' in config:
        osize = [config.crop.height, config.crop.width]
        transform_list.append(transforms.RandomCrop(osize))

    if 'flip' in config:
        transform_list.append(transforms.RandomHorizontalFlip())

    if 'convert' in config:
        transform_list += [transforms.ToTensor()]
        if 'grayscale' in config:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
    if 'onehot' in config:
        transform_list += [transforms.PILToTensor()]
        def f(x):
            return F.one_hot(x.long().squeeze(), num_classes=config.onehot.num_classes).float().permute(2, 0, 1)
        transform_list += [transforms.Lambda(lambda x : f(x))]
    return transforms.Compose(transform_list)
