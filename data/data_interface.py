from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data.sampler import WeightedRandomSampler
from omegaconf import DictConfig
from torch.utils.data.dataset import Dataset
from typing import Optional
import torch


class DataInterface(pl.LightningDataModule):
    def __init__(
        self,
        train_set: Dataset,
        test_set=Optional[Dataset],
        batch_size: int = 64,
        split_propotion:float = 0.2,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.train_set = train_set
        if test_set is None:
            self.train_set, self.test_set = torch.utils.data.random_split(
                train_set,
                [
                    int(len(train_set) * (1 - split_propotion)),
                    int(len(train_set) * split_propotion),
                ],
            )
        else:
            self.test_set = test_set
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Split train set into train and validation set
            self.train_set, self.val_set = torch.utils.data.random_split(
                self.train_set,
                [
                    int(len(self.train_set) * 0.8),
                    int(len(self.train_set) * 0.2),
                ],
            )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        
        
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
