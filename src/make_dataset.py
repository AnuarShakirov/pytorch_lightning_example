"""Module to make dataset from raw data."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.settings import DEPTH_COLUMN, LOGS, MASKING_VALUE, PATH_TO_DATA, PATH_TO_NORWAY_DATA, PATH_TO_SAVE_DATASET, RESISTIVITY_LOGS, SLICE_LEN


class DataPrepare:
    """Class to prepare data."""

    def __init__(self) -> None:
        """Initialize class."""
        self.path_to_data: Path = PATH_TO_NORWAY_DATA  # путь к файлу
        self.data: pd.DataFrame
        self.dict_processed_data: dict = {"input": [], "output": [], "sample_index": []}  # словарь для хранения обработанных данных
        self.stat_table: dict = {"WELL": [], "sample_index": [], "DEPTH_TOP": [], "DEPTH_BOTTOM": []}  # словарь для хранения статистики по данным

    def load_csv(self) -> None:
        """Load csv file."""
        self.data = pd.read_csv(self.path_to_data, sep=";")

    def fake_pocessing(self) -> None:
        """Fake processing.
        Тут на самом деле включаем все методы по контролю качества данных, и их подготовке.

        """
        self.data = self.data.dropna(subset=LOGS, how="any")  # дропнем Nan значения
        self.data[RESISTIVITY_LOGS] = self.data[RESISTIVITY_LOGS].apply(
            lambda x: np.log1p(x),
        )  # логарифмируем сопротивление
        # нормализуем данные
        scaler = StandardScaler()
        self.data[LOGS] = scaler.fit_transform(self.data[LOGS])

    @staticmethod
    def mask_data(df_cur: pd.DataFrame, logs: list[str] = LOGS, mask_percentage: float = 0.3) -> pd.DataFrame:
        """Method to randomly mask data."""
        for log in logs:
            mask = np.random.choice([True, False], size=df_cur.shape[0], p=[mask_percentage, 1 - mask_percentage])  # noqa: NPY002
            df_cur.loc[mask, log] = MASKING_VALUE
        return df_cur

    def save_processed_data(self) -> None:
        """Method to save processed data."""
        PATH_TO_SAVE_DATASET.mkdir(parents=True, exist_ok=True)
        for idx in range(len(self.dict_processed_data["input"])):
            sample_index: int = self.dict_processed_data["sample_index"][idx]
            torch.save(
                self.dict_processed_data["input"][idx],
                PATH_TO_SAVE_DATASET / f"input_{sample_index}.pt",
            )
            torch.save(
                self.dict_processed_data["output"][idx],
                PATH_TO_SAVE_DATASET / f"output_{sample_index}.pt",
            )
        pd.DataFrame(self.stat_table).to_csv(
            PATH_TO_DATA / "stat_table.csv",
            index=False,
        )
        logger.info(
            f"Data saved successfully. Total N of samples is {len(self.dict_processed_data['input'])}",
        )

    def split_by_slices(self) -> None:
        """Method to split data by slices."""
        # орагнизуем подготовку к обработке данных
        cur_slice_index: int = 0
        pbar = tqdm(self.data.groupby("WELL"))
        pbar.set_description("Splitting data by slices")
        for well, df_well in pbar:  # группируем данные по скважинам
            for idx in range(0, len(df_well), SLICE_LEN):  # нарезаем данные на сегменты без перекрытия
                slice_data = df_well.iloc[idx : idx + SLICE_LEN]  # делаем срез
                if len(slice_data) < SLICE_LEN:  # если длина среза меньше SLICE_LEN, то пропускаем
                    continue
                # маскируем данные
                slice_data_masked: pd.DataFrame = self.mask_data(slice_data)
                # конвертируем данные в торч тензоры
                slice_data_torch: torch.Tensor = torch.as_tensor(slice_data[LOGS].values, dtype=torch.float32)
                slice_data_masked_torch: torch.Tensor = torch.as_tensor(slice_data_masked[LOGS].values, dtype=torch.float32)
                # добавляем данные в словарь
                self.dict_processed_data["input"].append(slice_data_masked_torch)
                self.dict_processed_data["output"].append(slice_data_torch)
                self.dict_processed_data["sample_index"].append(cur_slice_index)
                cur_slice_index += 1
                # заполняем статистику
                self.stat_table["WELL"].append(well)
                self.stat_table["sample_index"].append(cur_slice_index)
                self.stat_table["DEPTH_TOP"].append(slice_data[DEPTH_COLUMN].min())
                self.stat_table["DEPTH_BOTTOM"].append(slice_data[DEPTH_COLUMN].max())

    def __call__(self) -> None:
        """Run class."""
        self.load_csv()
        self.fake_pocessing()
        self.split_by_slices()
        self.save_processed_data()


if __name__ == "__main__":
    data_prepare = DataPrepare()
    data_prepare()
