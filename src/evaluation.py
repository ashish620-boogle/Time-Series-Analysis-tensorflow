from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_predictions(y_true, y_pred) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "r2": float(r2)}


def as_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([metrics])


def save_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: np.ndarray,
    title: str,
    path: Path,
    max_points: int = 500,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
):
    """Save a simple actual vs predicted line plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    n = min(len(y_true), len(y_pred), max_points)
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    index = index[:n]
    lower = np.array(lower).reshape(-1)[:n] if lower is not None else None
    upper = np.array(upper).reshape(-1)[:n] if upper is not None else None

    plt.figure(figsize=(10, 4))
    plt.plot(index, y_true, label="Actual", linewidth=2)
    plt.plot(index, y_pred, label="Predicted", linewidth=2)
    if lower is not None and upper is not None:
        plt.fill_between(index, lower, upper, color="gray", alpha=0.2, label="Prediction interval")
    plt.title(title)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
