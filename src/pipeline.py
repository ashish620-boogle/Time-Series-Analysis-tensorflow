from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from . import config
from .data_fetcher import fetch_stock_history
from .evaluation import evaluate_predictions, save_prediction_plot
from .modeling import build_lstm_model, r2_loss, r2_metric, train_model
from .nbeats import NBeatsBlock, build_nbeats_model
from .preprocessing import (
    DEFAULT_FEATURES,
    Scalers,
    add_time_features,
    prepare_inference_window,
    scale_and_window,
    train_test_split_df,
)


@dataclass
class TrainingArtifacts:
    model: keras.Model
    scalers: Scalers
    feature_cols: Sequence[str]
    history: Dict
    metrics: Dict
    data: pd.DataFrame


class EnsembleModel:
    """Lightweight ensemble wrapper averaging predictions from member models."""

    def __init__(self, models: List[keras.Model]):
        self.models = models

    def predict(self, X, verbose=0):
        preds = [m.predict(X, verbose=verbose) for m in self.models]
        return np.mean(preds, axis=0)


def prepare_dataset(
    ticker: str = config.TICKER,
    period: str = config.PERIOD,
    interval: str = config.INTERVAL,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    target_col: str = "Close",
    window_size: int = config.WINDOW_SIZE,
    horizon: int = config.HORIZON,
):
    raw = fetch_stock_history(ticker=ticker, period=period, interval=interval)
    enriched = add_time_features(raw).dropna()
    selected_features = [col for col in feature_cols if col in enriched.columns]
    if target_col not in enriched.columns:
        raise ValueError(f"target column {target_col} missing from dataset")

    train_df, test_df = train_test_split_df(enriched, test_size=config.TEST_SPLIT)
    X_train, y_train, X_test, y_test, scalers = scale_and_window(
        train_df=train_df,
        test_df=test_df,
        feature_cols=selected_features,
        target_col=target_col,
        window_size=window_size,
        horizon=horizon,
    )
    test_index = test_df.index[window_size + horizon - 1 :]
    return X_train, y_train, X_test, y_test, scalers, selected_features, enriched, test_index


def run_training(
    ticker: str = config.TICKER,
    period: str = config.PERIOD,
    interval: str = config.INTERVAL,
    window_size: int = config.WINDOW_SIZE,
    horizon: int = config.HORIZON,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    model_type: Literal["lstm", "nbeats", "both"] = "both",
) -> TrainingArtifacts:
    (
        X_train,
        y_train,
        X_test,
        y_test,
        scalers,
        feature_cols,
        data,
        test_index,
    ) = prepare_dataset(
        ticker=ticker,
        period=period,
        interval=interval,
        window_size=window_size,
        horizon=horizon,
    )

    config.ensure_artifact_dir()
    metrics_by_model: Dict[str, Dict] = {}
    histories: Dict[str, Dict] = {}
    models: Dict[str, keras.Model] = {}

    def evaluate_and_save(
        name: str,
        preds_scaled: np.ndarray,
        member_preds_scaled: List[np.ndarray] | None = None,
    ):
        y_test_inv = scalers.target_scaler.inverse_transform(y_test.reshape(-1, 1)).squeeze()
        preds_inv = scalers.target_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).squeeze()

        lower_inv = upper_inv = None
        if member_preds_scaled is not None and len(member_preds_scaled) > 1:
            member_inv = [
                scalers.target_scaler.inverse_transform(np.array(p).reshape(-1, 1)).squeeze()
                for p in member_preds_scaled
            ]
            lower_inv = np.percentile(member_inv, 2.5, axis=0)
            upper_inv = np.percentile(member_inv, 97.5, axis=0)

        metrics = evaluate_predictions(y_test_inv, preds_inv)
        metrics_by_model[name] = metrics
        plot_path = config.PLOTS_DIR / f"{name}_pred_vs_actual.png"
        save_prediction_plot(
            y_true=y_test_inv,
            y_pred=preds_inv,
            index=np.array(test_index),
            title=f"{name.upper()} predictions",
            path=plot_path,
            lower=lower_inv,
            upper=upper_inv,
        )

    # Train LSTM
    if model_type in ("lstm", "both"):
        lstm_model = build_lstm_model(window_size=window_size, num_features=len(feature_cols), horizon=horizon)
        lstm_history = train_model(
            lstm_model,
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=epochs,
            batch_size=batch_size,
        )
        preds_scaled = lstm_model.predict(X_test, verbose=0)
        evaluate_and_save("lstm", preds_scaled)
        histories["lstm"] = lstm_history.history
        models["lstm"] = lstm_model
        lstm_model.save(config.LSTM_MODEL_PATH)

    # Train N-BEATS
    if model_type in ("nbeats", "both"):
        ensemble_size = config.NBEATS_ENSEMBLE
        nbeats_members = []
        member_preds = []
        for idx in range(ensemble_size):
            nbeats_model = build_nbeats_model(
                window_size=window_size,
                horizon=horizon,
                num_features=len(feature_cols),
                units=config.NBEATS_UNITS,
                stacks=config.NBEATS_STACKS,
                learning_rate=config.NBEATS_LEARNING_RATE,
            )
            nbeats_history = train_model(
                nbeats_model,
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=max(epochs * config.NBEATS_EPOCH_BOOST, epochs + 10),
                batch_size=batch_size,
            )
            preds_scaled = nbeats_model.predict(X_test, verbose=0)
            member_preds.append(preds_scaled)
            nbeats_members.append(nbeats_model)
            histories[f"nbeats_member_{idx}"] = nbeats_history.history

        ensemble_preds = np.mean(member_preds, axis=0)
        nbeats_ensemble = EnsembleModel(nbeats_members)
        evaluate_and_save("nbeats", ensemble_preds, member_preds_scaled=member_preds)
        models["nbeats"] = nbeats_ensemble

    if not models:
        raise ValueError(f"No models trained for model_type={model_type}")

    # Choose best model by MAE
    best_name = min(metrics_by_model.items(), key=lambda kv: kv[1]["mae"])[0]
    best_model = models[best_name]

    joblib.dump(scalers.feature_scaler, config.FEATURE_SCALER_PATH)
    joblib.dump(scalers.target_scaler, config.TARGET_SCALER_PATH)
    meta = {
        "feature_cols": list(feature_cols),
        "ticker": ticker,
        "interval": interval,
        "period": period,
        "best_model": best_name,
        "metrics": metrics_by_model,
        "nbeats_member_paths": [],
    }
    Path(config.METRICS_PATH).write_text(json.dumps(metrics_by_model, indent=2))

    # Save models and ensemble members
    if best_name == "nbeats" and isinstance(best_model, EnsembleModel):
        member_paths = []
        for idx, member in enumerate(best_model.models):
            path = config.ARTIFACT_DIR / f"nbeats_member_{idx}.keras"
            member.save(path)
            member_paths.append(str(path))
        meta["nbeats_member_paths"] = member_paths
        # Save the first member for compatibility
        best_model.models[0].save(config.BEST_MODEL_PATH)
    else:
        best_model.save(config.BEST_MODEL_PATH)

    meta_path = Path(config.BEST_MODEL_PATH).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    return TrainingArtifacts(
        model=best_model,
        scalers=scalers,
        feature_cols=feature_cols,
        history=histories.get(best_name, {}),
        metrics=metrics_by_model.get(best_name, {}),
        data=data,
    )


def load_artifacts() -> TrainingArtifacts:
    model_path = config.BEST_MODEL_PATH if config.BEST_MODEL_PATH.exists() else config.LSTM_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run main.py to train first.")
    model = keras.models.load_model(
        model_path,
        custom_objects={"NBeatsBlock": NBeatsBlock, "r2_loss": r2_loss, "r2_metric": r2_metric},
        safe_mode=False,
    )
    feature_scaler = joblib.load(config.FEATURE_SCALER_PATH)
    target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    meta_path = Path(model_path).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    feature_cols = meta.get("feature_cols", DEFAULT_FEATURES)
    # Load ensemble if defined
    ensemble_paths = meta.get("nbeats_member_paths", [])
    if ensemble_paths:
        members = [
            keras.models.load_model(
                p,
                custom_objects={"NBeatsBlock": NBeatsBlock, "r2_loss": r2_loss, "r2_metric": r2_metric},
                safe_mode=False,
            )
            for p in ensemble_paths
        ]
        model = EnsembleModel(members)
    return TrainingArtifacts(
        model=model,
        scalers=Scalers(feature_scaler, target_scaler),
        feature_cols=feature_cols,
        history={},
        metrics={},
        data=pd.DataFrame(),
    )


def predict_next_minute(
    latest_df: pd.DataFrame,
    artifacts: TrainingArtifacts,
    window_size: int = config.WINDOW_SIZE,
) -> float:
    window = prepare_inference_window(
        df=latest_df,
        feature_cols=artifacts.feature_cols,
        feature_scaler=artifacts.scalers.feature_scaler,
        window_size=window_size,
    )
    scaled_pred = artifacts.model.predict(window, verbose=0).squeeze()
    pred_price = artifacts.scalers.target_scaler.inverse_transform(np.array(scaled_pred).reshape(-1, 1)).squeeze()
    return float(pred_price)

# Backward compatibility alias
predict_next_step = predict_next_minute
