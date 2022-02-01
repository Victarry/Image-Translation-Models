from pathlib import Path
from torch.utils.data import Dataset
from src.datamodules.base_dataset import make_dataset
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms


class AlignedDataset(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, max_dataset_size=100000):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.dir_AB = Path(data_root)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, max_dataset_size))  # get image paths

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

        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        A_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        B_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': str(AB_path), 'B_paths': str(AB_path)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

class AlignedDatamodule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=32, num_workers=8, **kargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = Path(data_root)

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_data = AlignedDataset(self.data_root / 'train')
        self.test_data = AlignedDataset(self.data_root / 'val')


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
