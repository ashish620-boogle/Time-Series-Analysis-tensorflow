from pathlib import Path

# Default configuration for the forecasting pipeline
TICKER = "SPY"  # S&P 500 ETF, high liquidity
INTERVAL = "1m"  # minute-level data for next-minute forecasts
PERIOD = "7d"  # yfinance limit for 1m data; 7 days max per request

WINDOW_SIZE = 120  # two hours of minute-level context
HORIZON = 1  # predict one step (next minute) ahead
TEST_SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# N-BEATS specific tuning for better accuracy on limited 1m data
NBEATS_UNITS = 512
NBEATS_STACKS = 6
NBEATS_ENSEMBLE = 5
NBEATS_LEARNING_RATE = 1e-4
NBEATS_EPOCH_BOOST = 2  # multiplier on base epochs

ARTIFACT_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACT_DIR / "plots"

# Model artifact paths
LSTM_MODEL_PATH = ARTIFACT_DIR / "stock_forecaster_lstm.keras"
NBEATS_MODEL_PATH = ARTIFACT_DIR / "stock_forecaster_nbeats.keras"
BEST_MODEL_PATH = ARTIFACT_DIR / "stock_forecaster_best.keras"
FEATURE_SCALER_PATH = ARTIFACT_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = ARTIFACT_DIR / "target_scaler.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def ensure_artifact_dir() -> Path:
    """Create artifact directory if missing."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR
