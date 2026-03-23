"""
data_pipeline.py
----------------
Centralised data-fetching and quality-checking layer for QuantAgent.

Responsibilities:
  - Resolve ticker symbols for NSE (.NS), US stocks, indices, crypto
  - Fetch OHLCV data via yfinance with proper interval/period limits
  - Run data quality checks (gaps, zero-volume candles, bad OHLC values)
  - Return a clean, validated DataFrame ready for QuantAgent ingestion
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from qa_logger import get_logger
_log = get_logger("DataPipeline")

# ---------------------------------------------------------------------------
# Asset catalogue
# ---------------------------------------------------------------------------

# US indices and ETFs  (internal_code -> yfinance_symbol, display_name)
US_ASSETS: dict[str, tuple[str, str]] = {
    "SPX":  ("^GSPC",     "S&P 500"),
    "DJI":  ("^DJI",      "Dow Jones"),
    "NQ":   ("NQ=F",      "Nasdaq Futures"),
    "QQQ":  ("QQQ",       "Invesco QQQ Trust"),
    "VIX":  ("^VIX",      "Volatility Index"),
    "DXY":  ("DX-Y.NYB",  "US Dollar Index"),
    "GC":   ("GC=F",      "Gold Futures"),
    "CL":   ("CL=F",      "Crude Oil"),
    "ES":   ("ES=F",      "E-mini S&P 500"),
    "BTC":  ("BTC-USD",   "Bitcoin"),
    # US large-caps
    "AAPL": ("AAPL",      "Apple Inc."),
    "MSFT": ("MSFT",      "Microsoft"),
    "GOOGL":("GOOGL",     "Alphabet"),
    "AMZN": ("AMZN",      "Amazon"),
    "NVDA": ("NVDA",      "NVIDIA"),
    "META": ("META",      "Meta Platforms"),
    "TSLA": ("TSLA",      "Tesla Inc."),
    "JPM":  ("JPM",       "JPMorgan Chase"),
}

# NSE (India) stocks  (internal_code -> yfinance_symbol, display_name)
NSE_ASSETS: dict[str, tuple[str, str]] = {
    "RELIANCE":    ("RELIANCE.NS",    "Reliance Industries"),
    "TCS":         ("TCS.NS",         "Tata Consultancy Services"),
    "INFY":        ("INFY.NS",        "Infosys"),
    "HDFCBANK":    ("HDFCBANK.NS",    "HDFC Bank"),
    "ICICIBANK":   ("ICICIBANK.NS",   "ICICI Bank"),
    "SBIN":        ("SBIN.NS",        "State Bank of India"),
    "BAJFINANCE":  ("BAJFINANCE.NS",  "Bajaj Finance"),
    "HINDUNILVR":  ("HINDUNILVR.NS",  "Hindustan Unilever"),
    "WIPRO":       ("WIPRO.NS",       "Wipro"),
    "HCLTECH":     ("HCLTECH.NS",     "HCL Technologies"),
    "TATAMOTORS":  ("TATAMOTORS.NS",  "Tata Motors"),
    "ADANIENT":    ("ADANIENT.NS",    "Adani Enterprises"),
    "KOTAKBANK":   ("KOTAKBANK.NS",   "Kotak Mahindra Bank"),
    "LT":          ("LT.NS",          "Larsen & Toubro"),
    "AXISBANK":    ("AXISBANK.NS",    "Axis Bank"),
    "SUNPHARMA":   ("SUNPHARMA.NS",   "Sun Pharma"),
    "MARUTI":      ("MARUTI.NS",      "Maruti Suzuki"),
    "TITAN":       ("TITAN.NS",       "Titan Company"),
    "NIFTY50":     ("^NSEI",          "Nifty 50"),
    "BANKNIFTY":   ("^NSEBANK",       "Bank Nifty"),
    "SENSEX":      ("^BSESN",         "BSE Sensex"),
}

# Combined lookup: internal_code -> (yf_symbol, display_name)
ALL_ASSETS: dict[str, tuple[str, str]] = {**US_ASSETS, **NSE_ASSETS}

# yfinance hard limits per interval (max days of history available)
INTERVAL_MAX_DAYS: dict[str, int] = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1h":  730,
    "4h":  730,
    "1d":  1825,
    "1wk": 1825,
    "1mo": 1825,
}

# ---------------------------------------------------------------------------
# Symbol resolver
# ---------------------------------------------------------------------------

def resolve_symbol(ticker: str) -> tuple[str, str]:
    """
    Given any ticker string (user input or internal code), return
    (yfinance_symbol, display_name).

    Resolution order:
      1. Exact match in ALL_ASSETS catalogue
      2. If ends with .NS or .BO, treat as direct yfinance symbol
      3. Try appending .NS and see if it looks like a valid NSE ticker
      4. Fall through — use as-is (valid for most US tickers)
    """
    ticker = ticker.strip().upper()

    # 1. Catalogue lookup
    if ticker in ALL_ASSETS:
        yf_sym, name = ALL_ASSETS[ticker]
        return yf_sym, name

    # 2. Already has exchange suffix
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker, ticker.replace(".NS", "").replace(".BO", "")

    # 3. Heuristic: looks like an NSE symbol (all alpha, 2-20 chars, no digits)
    if re.match(r"^[A-Z]{2,20}$", ticker) and ticker not in US_ASSETS:
        # Probe with .NS suffix — we don't actually call yfinance here,
        # we just return the .NS candidate; fetch will fail gracefully if wrong
        return f"{ticker}.NS", ticker

    # 4. Treat as US ticker / direct yfinance symbol
    return ticker, ticker


def get_market(yf_symbol: str) -> str:
    """Return 'NSE', 'BSE', or 'US' based on the yfinance symbol."""
    if yf_symbol.endswith(".NS"):
        return "NSE"
    if yf_symbol.endswith(".BO"):
        return "BSE"
    return "US"


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def _quality_check(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Run a battery of quality checks and repairs on a raw OHLCV DataFrame.
    Returns a cleaned DataFrame (may have fewer rows than input).
    """
    original_len = len(df)
    issues: list[str] = []

    # 1. Drop rows where any OHLC value is NaN or zero
    ohlc = ["Open", "High", "Low", "Close"]
    mask_nan  = df[ohlc].isnull().any(axis=1)
    mask_zero = (df[ohlc] == 0).any(axis=1)
    bad = mask_nan | mask_zero
    if bad.sum():
        issues.append(f"dropped {bad.sum()} rows with NaN/zero OHLC")
        df = df[~bad].copy()

    # 2. Drop rows where High < Low (data corruption)
    inverted = df["High"] < df["Low"]
    if inverted.sum():
        issues.append(f"dropped {inverted.sum()} rows where High < Low")
        df = df[~inverted].copy()

    # 3. Drop rows where Close is outside [Low, High]
    bad_close = (df["Close"] < df["Low"]) | (df["Close"] > df["High"])
    if bad_close.sum():
        issues.append(f"dropped {bad_close.sum()} rows where Close outside [Low, High]")
        df = df[~bad_close].copy()

    # 4. Zero-volume candles: flag but keep (some indices don't report volume)
    if "Volume" in df.columns:
        zero_vol = (df["Volume"] == 0).sum()
        if zero_vol > len(df) * 0.3:
            issues.append(f"warning: {zero_vol}/{len(df)} candles have zero volume")

    # 5. Duplicate timestamps
    dupes = df["Datetime"].duplicated().sum()
    if dupes:
        issues.append(f"dropped {dupes} duplicate timestamps")
        df = df.drop_duplicates(subset="Datetime").copy()

    # 6. Sort by time
    df = df.sort_values("Datetime").reset_index(drop=True)

    if issues:
        _log.warning(f"{symbol}: {'; '.join(issues)} ({original_len}→{len(df)} rows)")
    else:
        _log.ok(f"{symbol} — {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> tuple[pd.DataFrame, str, str]:
    """
    Fetch OHLCV data for any ticker (NSE or US).

    Parameters
    ----------
    ticker   : internal code (e.g. "RELIANCE", "AAPL") or raw yf symbol
    interval : yfinance interval string ("1m", "5m", "15m", "1h", "4h", "1d" …)
    start    : datetime start (timezone-naive, local time)
    end      : datetime end

    Returns
    -------
    (df, yf_symbol, display_name)
      df is a clean DataFrame with columns: Datetime, Open, High, Low, Close, Volume
      df is empty if fetch failed or quality checks left < 10 rows
    """
    yf_symbol, display_name = resolve_symbol(ticker)
    market = get_market(yf_symbol)

    # Clamp date range to yfinance limits
    max_days = INTERVAL_MAX_DAYS.get(interval, 730)
    earliest_allowed = datetime.now() - timedelta(days=max_days)
    if start < earliest_allowed:
        _log.warning(f"{yf_symbol}: date clamped to {earliest_allowed:%Y-%m-%d} (max {max_days}d for {interval})")
        start = earliest_allowed

    try:
        _log.info(f"Fetching {yf_symbol} ({market}) {interval}  {start:%d-%b %H:%M} → {end:%d-%b %H:%M}")

        raw = yf.download(
            tickers=yf_symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            prepost=False,
            progress=False,
        )

        if raw is None or raw.empty:
            _log.warning(f"No data returned for {yf_symbol}")
            return pd.DataFrame(), yf_symbol, display_name

        # Flatten MultiIndex columns if present
        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Normalise datetime column name
        if "Datetime" not in raw.columns and "Date" in raw.columns:
            raw = raw.rename(columns={"Date": "Datetime"})

        # Keep only standard columns
        keep = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"]
                if c in raw.columns]
        df = raw[keep].copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Quality check
        df = _quality_check(df, yf_symbol)

        if len(df) < 10:
            _log.warning(f"{yf_symbol}: only {len(df)} rows after quality check — aborting")
            return pd.DataFrame(), yf_symbol, display_name

        return df, yf_symbol, display_name

    except Exception as exc:
        _log.error(f"Fetch error for {yf_symbol}: {exc}")
        return pd.DataFrame(), yf_symbol, display_name


# ---------------------------------------------------------------------------
# Catalogue helpers (used by the web API)
# ---------------------------------------------------------------------------

def list_assets() -> list[dict]:
    """Return the full asset catalogue as a list of dicts for the frontend."""
    result = []
    for code, (yf_sym, name) in ALL_ASSETS.items():
        market = get_market(yf_sym)
        result.append({"code": code, "yf_symbol": yf_sym, "name": name, "market": market})
    return result


def get_display_name(ticker: str) -> str:
    """Return a human-readable display name for any ticker."""
    ticker = ticker.strip().upper()
    if ticker in ALL_ASSETS:
        return ALL_ASSETS[ticker][1]
    _, name = resolve_symbol(ticker)
    return name


def get_interval_limits() -> dict[str, int]:
    """Return the max-days dict for all supported intervals."""
    return INTERVAL_MAX_DAYS.copy()