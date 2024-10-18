#!/usr/bin/python3

from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

class CIFAR10DataModule(LightningDataModule):
  def __init__(self,
               data_dir: str = "data/",
               train_val_split: Tuple[int, int] = (45000, 5000),
               batch_size: int = 64,
               num_workers: int = 0,
               pin_memory: bool = False,
               img_size: int = 32,
  ):
    super().__init__()
    self.data_dir = data_dir
    self.train_val_split = train_val_split
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.img_size = img_size

    self.transforms = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    self.dims = (3, self.img_size, self.img_size)
    self.data_train: Optional[Dataset] = None
    self.data_val: Optional[Dataset] = None
    self.data_test: Optional[Dataset] = None

  def prepare_data(self):
    CIFAR10(root='./data', train=True, download=True)
    CIFAR10(root='./data', train=False, download=True)

  def setup(self, stage: Optional[str] = None):
    trainset = CIFAR10(root='./data', train=True, download=True, transform=self.transforms)
    self.data_train, self.data_val = random_split(trainset, self.train_val_split)
    self.data_test = CIFAR10(root='./data', train=False, download=True, transform=self.transforms)

  def train_dataloader(self):
    return DataLoader(
      dataset = self.data_train,
      batch_size = self.batch_size,
      num_workers = self.num_workers,
      pin_memory = self.pin_memory,
      shuffle = True)

  def val_dataloader(self):
    return DataLoader(
      dataset = self.data_val,
      batch_size = self.batch_size,
      num_workers = self.num_workers,
      pin_memory = self.pin_memory,
      shuffle = True)

  def test_dataloader(self):
    return DataLoader(
      dataset = self.data_test,
      batch_size = self.batch_size,
      num_workers = self.num_workers,
      pin_memory = self.pin_memory,
      shuffle = True)

