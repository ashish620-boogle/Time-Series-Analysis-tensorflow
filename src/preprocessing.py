from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config

DEFAULT_FEATURES: List[str] = [
    "Close",
    "Volume",
    "returns",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
]


@dataclass
class Scalers:
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment raw price data with cyclical time features."""
    enriched = df.copy()
    idx = enriched.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for time feature engineering.")

    enriched["hour"] = idx.hour
    enriched["dayofweek"] = idx.dayofweek

    enriched["hour_sin"] = np.sin(2 * np.pi * enriched["hour"] / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * enriched["hour"] / 24)
    enriched["dayofweek_sin"] = np.sin(2 * np.pi * enriched["dayofweek"] / 7)
    enriched["dayofweek_cos"] = np.cos(2 * np.pi * enriched["dayofweek"] / 7)
    return enriched


def train_test_split_df(df: pd.DataFrame, test_size: float = config.TEST_SPLIT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def make_windows(features: np.ndarray, targets: np.ndarray, window_size: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    total = len(features)
    limit = total - window_size - horizon + 1
    if limit <= 0:
        raise ValueError(
            f"Not enough samples ({total}) for window_size={window_size} and horizon={horizon}. "
            "Increase history length or decrease window/horizon."
        )
    for start in range(limit):
        end = start + window_size
        X.append(features[start:end])
        y.append(targets[end + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def scale_and_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    window_size: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Scalers]:
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaler.fit(train_df[feature_cols])
    target_scaler.fit(train_df[[target_col]])

    train_features = feature_scaler.transform(train_df[feature_cols])
    test_features = feature_scaler.transform(test_df[feature_cols])

    train_target = target_scaler.transform(train_df[[target_col]]).flatten()
    test_target = target_scaler.transform(test_df[[target_col]]).flatten()

    X_train, y_train = make_windows(train_features, train_target, window_size, horizon)
    X_test, y_test = make_windows(test_features, test_target, window_size, horizon)
    return X_train, y_train, X_test, y_test, Scalers(feature_scaler, target_scaler)


def prepare_inference_window(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_scaler: MinMaxScaler,
    window_size: int,
) -> np.ndarray:
    """Return the most recent window of scaled features for inference."""
    if len(df) < window_size:
        raise ValueError(f"Need at least {window_size} rows to form an inference window.")
    window_df = df.tail(window_size)
    scaled = feature_scaler.transform(window_df[feature_cols])
    return np.expand_dims(scaled, axis=0)
