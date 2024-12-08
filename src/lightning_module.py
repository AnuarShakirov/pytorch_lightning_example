"""Module to train model."""

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MeanMetric

from src.metrics import get_metrics
from src.settings import LEARNING_RATE, MASKING_VALUE


class TaskAgnosticModule(pl.LightningModule):
    """Class to train model."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize class."""
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()

        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        metrics = get_metrics()
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

        self.save_hyperparameters()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(input_tensor)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Training step."""
        input_tensor, target_tensor = batch # получаем батч
        mask = input_tensor == MASKING_VALUE # получаем маску по которой будем считать лосс
        output = self.model(input_tensor) # прогнояем батч через модель
        masked_true = target_tensor.clone()[mask] # получаем тру значения по маске
        masked_pred = output[mask]
        loss = self.loss(masked_pred, masked_true)
        self._train_loss(loss)
        self.train_metrics(masked_pred, masked_true)
        return loss

    def on_train_epoch_end(self) -> None:
        """Log metrics at the end of the epoch."""
        self.log(
            "mean_train_loss",
            self._train_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )
        self._train_loss.reset()

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Validation step."""
        input_tensor, target_tensor = batch
        mask = input_tensor == MASKING_VALUE
        output = self.model(input_tensor)
        masked_true = target_tensor.clone()[mask]
        masked_pred = output[mask]
        loss = self.loss(masked_pred, masked_true)
        self._valid_loss(loss)
        self.valid_metrics(masked_pred, masked_true)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log metrics at the end of the epoch."""
        self.log(
            "mean_valid_loss",
            self._valid_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )
        self._valid_loss.reset()

        self.log_dict(self.valid_metrics.compute(), prog_bar=True, on_epoch=True)
        self.valid_metrics.reset()

    def configure_optimizers(self) -> torch.optim:
        """Configure the Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
