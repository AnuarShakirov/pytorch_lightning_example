"""Module to create class for dataset."""

from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.settings import PATH_TO_DATA, PATH_TO_SAVE_DATASET


if TYPE_CHECKING:
    from pathlib import Path


class DatasetTaskAgnostic(Dataset):
    """Class to load prepared data."""

    def __init__(self, data_stat: pd.DataFrame) -> None:
        """Initialize class."""
        super().__init__()
        self.data_stat: pd.DataFrame = data_stat
        self.path_to_data: Path = PATH_TO_SAVE_DATASET

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data_stat)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return item by index."""
        # берем индекс из статистики
        sample_index = self.data_stat.iloc[index]["sample_index"]
        # загружаем данные из торч файлов
        input_data = torch.load(self.path_to_data / f"input_{sample_index}.pt", weights_only=True)
        output_data = torch.load(self.path_to_data / f"output_{sample_index}.pt", weights_only=True)
        return input_data, output_data


if __name__ == "__main__":
    path_to_datastat = PATH_TO_DATA / "stat_table.csv"
    data_stat = pd.read_csv(path_to_datastat)
    dataset = DatasetTaskAgnostic(data_stat)
    # get item
    print(next(iter(dataset))[0])
