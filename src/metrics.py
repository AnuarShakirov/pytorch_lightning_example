"""Module to store metrics for model evaluation."""

from typing import Any

from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MetricCollection


def get_metrics(**kwargs: Any) -> MetricCollection:
    """Create a MetricCollection for regression tasks."""
    return MetricCollection(
        {
            "rmse": MeanSquaredError(squared=False, **kwargs),  # RMSE
            "mape": MeanAbsolutePercentageError(**kwargs),      # MAPE
            "mae": MeanAbsoluteError(**kwargs),                 # MAE
        }
    )

