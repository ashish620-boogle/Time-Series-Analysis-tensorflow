import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from src import config
from src.data_fetcher import fetch_stock_history
from src.nbeats import NBeatsBlock
from src.pipeline import EnsembleModel, TrainingArtifacts, load_artifacts, predict_next_minute, run_training
from src.preprocessing import DEFAULT_FEATURES, Scalers, add_time_features
from src.modeling import r2_loss, r2_metric
from src.trading import Portfolio, execute_signal, trading_signal
import joblib


st.set_page_config(page_title="Live Stock Forecaster", layout="wide")
st.title("Live Minute Stock Forecasting & Strategy")


def init_state():
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = None
    if "last_retrained" not in st.session_state:
        st.session_state.last_retrained = None
    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None
    if "prediction_target_minute" not in st.session_state:
        st.session_state.prediction_target_minute = None
    if "data" not in st.session_state:
        st.session_state.data = None
    if "last_data_timestamp" not in st.session_state:
        st.session_state.last_data_timestamp = None
    if "data_params" not in st.session_state:
        st.session_state.data_params = None
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = Portfolio(cash=10000.0)
    if "initial_cash" not in st.session_state:
        st.session_state.initial_cash = 10000.0
    if "last_signal_timestamp" not in st.session_state:
        st.session_state.last_signal_timestamp = None
    if "portfolio_snapshots" not in st.session_state:
        st.session_state.portfolio_snapshots = []
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "best"


@st.cache_data(ttl=2 * 60, show_spinner=False)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    raw = fetch_stock_history(ticker=ticker, period=period, interval=interval)
    return add_time_features(raw).dropna()


def load_model_choice(model_choice: str) -> TrainingArtifacts | None:
    path_map = {
        "best": config.BEST_MODEL_PATH,
        "lstm": config.LSTM_MODEL_PATH,
        "nbeats": config.NBEATS_MODEL_PATH,
    }
    model_path = path_map.get(model_choice, config.BEST_MODEL_PATH)
    if not model_path.exists():
        return None

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


def ensure_artifacts_loaded():
    if st.session_state.artifacts is None:
        st.session_state.artifacts = load_model_choice(st.session_state.model_choice)


def predict_with_guard(data: pd.DataFrame, window_size: int) -> float | None:
    if st.session_state.artifacts is None:
        return None
    feature_cols = st.session_state.artifacts.feature_cols
    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        st.warning(f"Missing feature columns for prediction: {', '.join(missing)}")
        return None
    latest_ts = data.index[-1].to_pydatetime()
    target_minute = latest_ts + dt.timedelta(minutes=1)
    if st.session_state.latest_prediction is None or st.session_state.prediction_target_minute != target_minute:
        pred = predict_next_minute(data, st.session_state.artifacts, window_size=window_size)
        st.session_state.latest_prediction = pred
        st.session_state.prediction_target_minute = target_minute
    return st.session_state.latest_prediction


def build_prediction_series(
    data: pd.DataFrame, window_size: int, max_points: int = 300
) -> pd.DataFrame | None:
    if st.session_state.artifacts is None:
        return None
    if len(data) < window_size + 1:
        return None
    feature_cols = [col for col in st.session_state.artifacts.feature_cols if col in data.columns]
    if not feature_cols:
        return None
    features = st.session_state.artifacts.scalers.feature_scaler.transform(data[feature_cols])

    start_idx = max(window_size, len(features) - max_points - 1)
    windows = []
    for idx in range(start_idx, len(features) - window_size):
        windows.append(features[idx : idx + window_size])
    if not windows:
        return None
    X = np.array(windows, dtype=np.float32)
    preds_scaled = st.session_state.artifacts.model.predict(X, verbose=0)
    preds = st.session_state.artifacts.scalers.target_scaler.inverse_transform(preds_scaled)

    preds_flat = np.asarray(preds).reshape(-1)
    actual = data["Close"].iloc[start_idx + window_size : start_idx + window_size + len(preds_flat)].values
    actual_flat = np.asarray(actual).reshape(-1)

    # Align lengths defensively
    n = min(len(preds_flat), len(actual_flat))
    pred_index = data.index[start_idx + window_size : start_idx + window_size + n]
    return pd.DataFrame(
        {
            "Actual": actual_flat[:n],
            "Predicted": preds_flat[:n],
        },
        index=pred_index,
    )


def update_profit_series(timestamp: pd.Timestamp, current_price: float) -> float:
    portfolio = st.session_state.portfolio
    total_value = portfolio.value(current_price)
    total_profit = total_value - st.session_state.initial_cash
    snapshots = st.session_state.portfolio_snapshots
    if not snapshots or snapshots[-1]["timestamp"] != timestamp:
        snapshots.append(
            {
                "timestamp": timestamp,
                "total_value": total_value,
                "total_profit": total_profit,
                "realized_pnl": portfolio.realized_pnl,
                "unrealized_pnl": portfolio.unrealized_pnl(current_price),
            }
        )
    return total_profit


def app():
    init_state()
    ensure_artifacts_loaded()

    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Ticker", value=config.TICKER)
    interval = st.sidebar.selectbox("Interval", options=["1m", "2m", "5m"], index=0)
    period = st.sidebar.selectbox("Period", options=["7d", "5d", "2d", "1d"], index=0)
    window_size = st.sidebar.slider("Window (minutes)", min_value=30, max_value=360, value=config.WINDOW_SIZE, step=30)
    horizon = st.sidebar.number_input("Horizon (minutes ahead)", min_value=1, max_value=1, value=1)
    epochs = st.sidebar.slider("Epochs", min_value=5, max_value=50, value=config.EPOCHS, step=5)
    batch_size = st.sidebar.selectbox("Batch size", options=[32, 64, 128, 256], index=2)
    threshold = st.sidebar.slider("Trade threshold (%)", min_value=0.02, max_value=1.0, value=0.1, step=0.02) / 100
    refresh_data = st.sidebar.button("Load latest data")
    retrain_now = st.sidebar.button("Retrain on new data")
    model_choice = st.sidebar.selectbox("Model for predictions", options=["best", "lstm", "nbeats"], index=0)

    data_params = (ticker, period, interval)
    params_changed = st.session_state.data_params != data_params
    choice_changed = st.session_state.model_choice != model_choice
    if refresh_data:
        load_data.clear()
    if refresh_data or params_changed:
        st.session_state.data = None
        st.session_state.latest_prediction = None
        st.session_state.prediction_target_minute = None
        st.session_state.last_signal_timestamp = None
    if choice_changed:
        st.session_state.model_choice = model_choice
        st.session_state.artifacts = None
        st.session_state.latest_prediction = None
        st.session_state.prediction_target_minute = None

    if st.session_state.data is None or params_changed:
        st.session_state.data = load_data(ticker=ticker, period=period, interval=interval)
        st.session_state.data_params = data_params
    data = st.session_state.data
    st.session_state.last_data_timestamp = data.index[-1]
    st.sidebar.caption(
        f"Loaded {len(data)} rows ({interval}, period {period}; 1m limited to 7 days). Last bar {data.index[-1]}"
    )

    if retrain_now or st.session_state.artifacts is None:
        with st.spinner("Retraining model on latest data..."):
            run_training(
                ticker=ticker,
                period=period,
                interval=interval,
                window_size=window_size,
                horizon=horizon,
                epochs=epochs,
                batch_size=batch_size,
            )
        st.session_state.last_retrained = dt.datetime.utcnow()
        st.session_state.latest_prediction = None
        st.session_state.prediction_target_minute = None
        st.session_state.artifacts = load_model_choice(model_choice)
        st.success(f"Model retrained at {st.session_state.last_retrained:%Y-%m-%d %H:%M UTC}")

    ensure_artifacts_loaded()
    if st.session_state.artifacts is None:
        st.error("No trained model found. Train a model first.")
        st.stop()

    latest_price_value = float(data["Close"].iloc[-1])
    prediction = predict_with_guard(data, window_size=window_size)
    pred_series = build_prediction_series(data, window_size=window_size, max_points=300)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest price", f"${latest_price_value:,.2f}")
    if prediction is not None:
        col2.metric("Predicted next minute", f"${prediction:,.2f}")
        delta_pct = (prediction - latest_price_value) / latest_price_value * 100
        col3.metric("Expected move", f"{delta_pct:+.2f} %")
    if st.session_state.last_retrained:
        col4.metric("Last retrain (UTC)", st.session_state.last_retrained.strftime("%H:%M"))

    st.subheader("Price vs Prediction")
    if pred_series is not None:
        st.line_chart(pred_series, height=280)
    else:
        st.line_chart(data["Close"], height=280)

    latest_timestamp = data.index[-1]
    signal = "hold"
    if prediction is not None and st.session_state.last_signal_timestamp != latest_timestamp:
        signal = trading_signal(prediction, latest_price_value, threshold=threshold)
        st.session_state.last_signal_timestamp = latest_timestamp
        if signal != "hold":
            execute_signal(st.session_state.portfolio, signal, latest_price_value, fraction=0.001)
    if signal != "hold":
        st.info(f"Strategy signal: {signal.upper()} (threshold {threshold*100:.2f}%)")

    portfolio = st.session_state.portfolio
    stats = portfolio.to_dict(latest_price_value)
    total_profit = update_profit_series(latest_timestamp, latest_price_value)

    st.subheader("Profit Over Time")
    if st.session_state.portfolio_snapshots:
        profit_df = pd.DataFrame(st.session_state.portfolio_snapshots).set_index("timestamp")
        st.line_chart(profit_df[["total_profit"]], height=200)

    st.subheader("Portfolio")
    p1, p2, p3 = st.columns(3)
    p1.metric("Cash", f"${stats['cash']:,.2f}")
    p2.metric("Position (shares)", f"{stats['position_shares']:.4f}")
    p3.metric("Total value", f"${stats['total_value']:,.2f}")

    p4, p5, p6 = st.columns(3)
    p4.metric("Avg price", f"${stats['avg_price']:,.2f}")
    p5.metric("Realized PnL", f"${stats['realized_pnl']:,.2f}")
    p6.metric("Unrealized PnL", f"${stats['unrealized_pnl']:,.2f}")

    st.metric("Total profit", f"${total_profit:,.2f}")

    with st.expander("Trade History", expanded=False):
        if portfolio.history:
            hist_df = pd.DataFrame([t.__dict__ for t in portfolio.history])
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.caption("No trades executed yet.")

    st.caption("Use 'Load latest data' to fetch new bars and 'Retrain on new data' to update the model.")


if __name__ == "__main__":
    app()
