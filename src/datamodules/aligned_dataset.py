from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from src.datamodules.base_dataset import make_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.transforms import get_transform


class ConcatAlignedDataset(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, reverse_AB=False, A_transforms=None, B_transforms=None):
        super().__init__()
        self.reverse_AB = reverse_AB
        self.dir_AB = Path(data_root)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths
        self.A_transforms = A_transforms
        self.B_transforms = B_transforms

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        A = self.A_transforms(A)
        B = self.B_transforms(B)

        if self.reverse_AB:
            return {'A': B, 'B': A, 'A_paths': str(AB_path), 'B_paths': str(AB_path)}
        else:
            return {'A': A, 'B': B, 'A_paths': str(AB_path), 'B_paths': str(AB_path)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


class AlignedDataset(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains two folders A and B having identity image names.
    """
    def __init__(self, data_root, reverse_AB=False, A_transforms=None, B_transforms=None) -> None:
        super().__init__()
        self.reverse_AB = reverse_AB
        self.dir_A = Path(data_root) / 'A'  # get the image directory
        self.dir_B = Path(data_root) / 'B'
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_transforms = A_transforms
        self.B_transforms = B_transforms
    
    def __getitem__(self, index):
        B_path = self.B_paths[index]
        A_path = (self.dir_A / B_path.relative_to(B_path.parent.parent)).with_name(B_path.stem + '_gtFine_labelIds.png')

        A_img = Image.open(A_path)
        B_img = Image.open(B_path)

        A = self.A_transforms(A_img)
        B = self.B_transforms(B_img)

        if self.reverse_AB:
            return {'A': B, 'B': A, 'A_paths': str(B_path), 'B_paths': str(A_path)}
        else:
            return {'A': A, 'B': B, 'A_paths': str(A_path), 'B_paths': str(B_path)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.B_paths)

class AlignedDatamodule(pl.LightningDataModule):
    def __init__(self, data_root, format="concat", batch_size=32, num_workers=8, reverse_AB=False, A_transforms: dict = None, B_transforms: dict = None, **kargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = Path(data_root)
        self.reverse_AB = reverse_AB
        self.A_transforms = get_transform(A_transforms)
        self.B_transforms = get_transform(B_transforms)
        self.format = format

    def prepare_data(self):
        # download data etc.
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if self.format == "concat": # paried images are concated into a single image
            self.train_data = ConcatAlignedDataset(self.data_root / 'train', reverse_AB=self.reverse_AB, A_transforms=self.A_transforms, B_transforms=self.B_transforms)
            self.val_data = ConcatAlignedDataset(self.data_root / 'val', reverse_AB=self.reverse_AB, A_transforms=self.A_transforms, B_transforms=self.B_transforms)
        elif self.format == "normal": # paried images are the same name but different folder
            self.train_data = AlignedDataset(self.data_root / 'train', reverse_AB=self.reverse_AB, A_transforms=self.A_transforms, B_transforms=self.B_transforms)
            self.val_data = AlignedDataset(self.data_root / 'val', reverse_AB=self.reverse_AB, A_transforms=self.A_transforms, B_transforms=self.B_transforms)


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
