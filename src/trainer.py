"""Module to run model training pipeline."""

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.datamodule import LogsDataModule
from src.lightning_module import TaskAgnosticModule
from src.model import TransformerAutoencoder
from src.settings import PATH_TO_SAVE_MODELS


def train_model() -> None:
    """Method to train model."""
    pl.seed_everything(0)
    # Load data
    datamodule = LogsDataModule()
    # initialize model
    model = TransformerAutoencoder()
    lightning_module = TaskAgnosticModule(model=model)
    # callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=PATH_TO_SAVE_MODELS,
            filename="best_model",
            monitor="valid_loss",
            mode="min",
        ),
    ]
    # initialize trainer
    trainer = Trainer(
        max_epochs=1,
        callbacks=callbacks,
        fast_dev_run=True,
    )
    trainer.fit(model=lightning_module, datamodule=datamodule)

if __name__ == "__main__":
    train_model()
