# Time Series Forecasting with TensorFlow (Stocks, LSTM + N-BEATS)

## Conceptual Overview

- **Problem**: Forecast the next stock price step from recent market history and basic time features, then simulate trading to observe PnL.
- **Data**: Live market data via `yfinance`. Default: SPY, 1m bars, ~7 days (provider limit). Features: OHLCV, returns, and cyclical time encodings (hour/day-of-week sine/cosine).
- **Preprocessing**:
  - Chronological train/test split (no shuffling).
  - MinMax scaling of features and target (Close) with persisted scalers.
  - Sliding windows (`window_size`) with horizon=1 to keep temporal order.
  - Inference uses the latest `window_size` rows scaled with saved scalers.
- **Models**:
  - **LSTM**: Two stacked LSTM layers with dropout and a dense head. Optimizer: Adam; loss: R²-based (`r2_loss`); metrics include MAE, MSE, and R².
  - **N-BEATS Ensemble**: Simplified fully-connected N-BEATS blocks (backcast/forecast) stacked and ensembled (default 5 members). Predictions are averaged; 95% prediction intervals come from member dispersion. Also trained with R² loss.
  - **Selection**: Optionally train both; the best (lowest MAE on inverse-scaled test set) is saved as the deployed artifact. Individual LSTM/N-BEATS artifacts are also saved for manual selection.
- **Evaluation**:
  - Metrics: MAE, RMSE, MAPE, R² (all inverse-scaled to price units).
  - Plots: Actual vs. predicted with optional prediction intervals, saved under `artifacts/plots/`.
  - Metrics JSON persisted to `artifacts/metrics.json`.
- **Artifacts & Metadata**:
  - Models: `artifacts/stock_forecaster_best.keras`, plus LSTM/N-BEATS-specific files.
  - Scalers: `artifacts/feature_scaler.pkl`, `artifacts/target_scaler.pkl`.
  - Meta JSON per model includes feature columns and ensemble member paths (for N-BEATS).
- **Streamlit App** (`app.py`):
  - Manual data refresh, manual retrain.
  - Model selection dropdown (`best`, `lstm`, `nbeats`) for live predictions.
  - Price vs. prediction chart, profit-over-time chart.
  - Threshold-based trading rule (buy/sell vs. hold), portfolio metrics, trade history.
  - Starts with $10,000 cash and trades 0.1% of cash per signal by default.
- **Training Logic**:
  - R²-based loss encourages variance explained; EarlyStopping prevents overfit.
  - N-BEATS ensemble trains multiple seeds; intervals derived from member spread.
  - Temporal integrity: no shuffling; validation is the chronological test split.
- **Live Constraints**:
  - 1m data limited to ~7 days; use coarser intervals (e.g., 5m) to access longer history.
  - Predictions target next step only (horizon=1); the app remains minute-aligned by default.

## How to Run

1) **Install dependencies**
```bash
pip install -e .
```

2) **Train (CLI)**
```bash
# Train both models and save the best
python main.py --model-type both

# Train a single model
python main.py --model-type lstm
python main.py --model-type nbeats
```

3) **Run the Streamlit app**
```bash
streamlit run app.py
```
- In the sidebar, load data, retrain, choose model (`best`/`lstm`/`nbeats`), and adjust thresholds/hyperparameters.

4) **Artifacts & results**
- Metrics: `artifacts/metrics.json`
- Plots: `artifacts/plots/*`
- Models: `artifacts/stock_forecaster_best.keras`, plus LSTM/N-BEATS files.

5) **Notes**
- To improve R² or stability, try coarser intervals (5m/15m), adjust `window_size`, or retune the N-BEATS ensemble size.
- The trading rule is intentionally simple; adjust `fraction` or logic in `src/trading.py` for different behavior.

## Repository Publishing
Remote push is not performed from this environment. To publish, run `git add/commit` and `git push` to your GitHub repo (`https://github.com/ashish620-boogle/Time-Series-Analysis-tensorflow`) from your local machine.
