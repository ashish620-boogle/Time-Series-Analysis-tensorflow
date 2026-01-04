from __future__ import annotations

from typing import Optional, Tuple

import requests
import pandas as pd
import yfinance as yf

from . import config


def _yf_session() -> requests.Session:
    """
    Build a requests session with a stable user agent.

    Streamlit Cloud occasionally blocks yfinance's rotating/impersonated
    agents; a consistent UA avoids the "Impersonating chrome136 is not
    supported" error.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; StockForecaster/1.0)"})
    return session


def _strip_timezone(index: pd.Index) -> pd.Index:
    """Return tz-naive timestamps to keep downstream numpy/tf happy."""
    if getattr(index, "tz", None):
        return index.tz_convert(None)
    return index


def fetch_stock_history(
    ticker: str = config.TICKER,
    period: str = config.PERIOD,
    interval: str = config.INTERVAL,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical stock/ETF data with the longest possible window.

    By default pulls minute-level SPY data (`period="7d"`, `interval="1m"`), which
    is the largest history yfinance exposes for 1m bars in a single request.
    """
    period, start, end = _sanitize_request(period, interval, start, end)
    session = _yf_session()
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
            session=session,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download data for {ticker} (period={period}, interval={interval}): {exc}. "
            "Try a coarser interval like 5m/2m or a shorter period."
        ) from exc
    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker} with period={period}, interval={interval}. "
            "Try a coarser interval like 5m/2m or a shorter period."
        )

    df.index = _strip_timezone(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df["returns"] = df["Close"].pct_change()
    df = df.dropna()
    df.sort_index(inplace=True)
    return df


def _sanitize_request(
    period: Optional[str],
    interval: str,
    start: Optional[str],
    end: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Clamp requests to stay within yfinance's 1m data limits."""
    if interval != "1m":
        return period, start, end

    # If caller provided explicit start/end, keep only the last 7 days.
    if start or end:
        end_ts = pd.to_datetime(end, errors="coerce") if end else pd.Timestamp.utcnow()
        if pd.isna(end_ts):
            end_ts = pd.Timestamp.utcnow()
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert(None)
        start_ts = pd.to_datetime(start, errors="coerce") if start else end_ts - pd.Timedelta(days=7)
        if pd.isna(start_ts):
            start_ts = end_ts - pd.Timedelta(days=7)
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert(None)
        if (end_ts - start_ts).days > 7:
            start_ts = end_ts - pd.Timedelta(days=7)
        return None, start_ts.isoformat(), end_ts.isoformat()

    if period is None:
        return "7d", start, end

    period_lower = str(period).lower()
    if period_lower == "max":
        return "7d", start, end
    if period_lower.endswith("d"):
        try:
            days = int(period_lower[:-1])
        except ValueError:
            return "7d", start, end
        return ("7d", start, end) if days > 7 else (period_lower, start, end)
    # Any non-day period (mo, y, etc.) gets clamped to 7d for 1m bars.
    return "7d", start, end


def latest_price(ticker: str = config.TICKER) -> float:
    """Get the most recent close price using a lightweight request."""
    snapshot = yf.download(
        ticker, period="1d", interval="1m", progress=False, session=_yf_session(), threads=True
    )
    snapshot.index = _strip_timezone(snapshot.index)
    if snapshot.empty:
        raise RuntimeError(f"Unable to retrieve latest price for {ticker}")
    return float(snapshot["Close"].dropna().iloc[-1])
