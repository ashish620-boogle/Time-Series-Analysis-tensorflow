import argparse
from pathlib import Path

from src import config
from src.data_fetcher import fetch_stock_history
from src.pipeline import load_artifacts, predict_next_minute, run_training
from src.preprocessing import add_time_features
import warnings

warnings.filterwarnings("ignore")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and forecast next-minute stock prices with TensorFlow.")
    parser.add_argument("--ticker", default=config.TICKER, help="Ticker symbol to fetch (default: SPY)")
    parser.add_argument("--period", default=config.PERIOD, help="Historical period to request from yfinance.")
    parser.add_argument("--interval", default=config.INTERVAL, help="Sampling interval (default: 1m).")
    parser.add_argument("--window-size", type=int, default=config.WINDOW_SIZE, help="Sliding window size.")
    parser.add_argument("--horizon", type=int, default=config.HORIZON, help="Forecast horizon in steps.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Training batch size.")
    parser.add_argument(
        "--model-type",
        choices=["lstm", "nbeats", "both"],
        default="both",
        help="Which model(s) to train; 'both' will compare and deploy the best.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and use the latest saved model to forecast the next minute.",
    )
    return parser.parse_args()


def train_and_report(args: argparse.Namespace):
    artifacts = run_training(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        window_size=args.window_size,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
    )
    print(f"Model trained on {len(artifacts.data)} rows of {args.ticker} ({args.interval}) data.")
    print(f"Best model metrics (inverse scaled): {artifacts.metrics}")
    return artifacts


def forecast_next_minute(args: argparse.Namespace):
    latest_df = add_time_features(
        fetch_stock_history(ticker=args.ticker, period=args.period, interval=args.interval)
    ).dropna()
    artifacts = load_artifacts()
    prediction = predict_next_minute(latest_df, artifacts, window_size=args.window_size)
    latest_close = float(latest_df["Close"].iloc[-1])
    print(f"Latest close: {latest_close:.2f}")
    print(f"Predicted next minute close: {prediction:.2f}")


def main():
    args = parse_args()
    config.ensure_artifact_dir()
    artifacts = None
    if not args.predict_only:
        artifacts = train_and_report(args)
    else:
        if not Path(config.BEST_MODEL_PATH).exists():
            raise SystemExit("No trained model found. Run without --predict-only first.")
    forecast_next_minute(args)


if __name__ == "__main__":
    main()
