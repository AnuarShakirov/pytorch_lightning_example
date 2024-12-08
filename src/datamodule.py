"""Module to create lightning datamodule."""

from typing import TYPE_CHECKING

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.dataset import DatasetTaskAgnostic
from src.settings import BATCH_SIZE, NUM_WORKERS, PATH_TO_DATA, SHUFFLE_DATA, TRAIN_SIZE


if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import Dataset


class LogsDataModule(LightningDataModule):
    """Class to prepare data for training."""

    def __init__(self) -> None:
        """Initialize class."""
        super().__init__()
        self.batch_size: int = BATCH_SIZE
        self.path_to_datastat: Path = PATH_TO_DATA / "stat_table.csv"

        self.dataset_train: Dataset
        self.dataset_val: Dataset

    def prepare_data(self) -> None:
        """Method to prepare data."""
        data_stat: pd.DataFrame = pd.read_csv(self.path_to_datastat)
        # split data
        train_samples: pd.DataFrame = data_stat.sample(frac=TRAIN_SIZE)
        val_samples: pd.DataFrame = data_stat.drop(train_samples.index)
        # create datasets
        self.dataset_train = DatasetTaskAgnostic(train_samples)
        self.dataset_val = DatasetTaskAgnostic(val_samples)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(self.dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE_DATA)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(self.dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE_DATA)
