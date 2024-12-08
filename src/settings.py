"""Module for storing global variables and settings."""

from pathlib import Path


# определяем пути до файлов
PATH_TO_DATA: Path = Path("data")
PATH_TO_NORWAY_DATA: Path = Path(PATH_TO_DATA, "Norweydata.csv").resolve()
PATH_TO_SAVE_DATASET: Path = Path(PATH_TO_DATA, "processed_dataset").resolve()
PATH_TO_SAVE_MODELS: Path = Path(PATH_TO_DATA, "models").resolve()

# определяем конфиги по обработке данных
LOGS: list[str] = ["NPHI", "RDEP", "DTC", "RHOB", "GR"]
RESISTIVITY_LOGS: list[str] = ["RDEP"]
SLICE_LEN: int = 100
MASKING_VALUE: int = -999
DEPTH_COLUMN: str = "DEPTH_MD"

# определяем конфиги по обучению модели
TRAIN_SIZE: float = 0.8
BATCH_SIZE: int = 32
NUM_WORKERS: int = 4
SHUFFLE_DATA: bool = True
LEARNING_RATE: float = 1e-3