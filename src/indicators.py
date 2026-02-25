"""
====================================================
🛡️ PROJECT AEGIS - Technical Indicators Wrapper
====================================================
Drop-in replacement for pandas_ta using the 'ta' library.
pandas_ta is unmaintained and broken on PyPI.
This module provides the SAME function signatures so
existing code only needs to change one import line.

Usage (in any file):
    import indicators as ta
    df["RSI"] = ta.rsi(df["Close"], length=14)
    # ... exactly the same API as before
====================================================
"""

import pandas as pd
import numpy as np
import ta as _ta


def _ensure_series(data) -> pd.Series:
    """Convert DataFrame column to Series if needed (yfinance multi-index fix)."""
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level columns from yfinance downloads."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ──────────────────────────────────────────────────
#   MOMENTUM
# ──────────────────────────────────────────────────
def rsi(close, length=14):
    """Relative Strength Index."""
    close = _ensure_series(close)
    return _ta.momentum.RSIIndicator(close, window=length).rsi()


# ──────────────────────────────────────────────────
#   TREND
# ──────────────────────────────────────────────────
def sma(close, length=20):
    """Simple Moving Average."""
    close = _ensure_series(close)
    return _ta.trend.SMAIndicator(close, window=length).sma_indicator()


def ema(close, length=20):
    """Exponential Moving Average."""
    close = _ensure_series(close)
    return _ta.trend.EMAIndicator(close, window=length).ema_indicator()


def macd(close, fast=12, slow=26, signal=9):
    """
    MACD indicator. Returns DataFrame with 3 columns:
      col 0: MACD line
      col 1: MACD histogram
      col 2: MACD signal line
    (Same column order as pandas_ta)
    """
    close = _ensure_series(close)
    indicator = _ta.trend.MACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
    result = pd.DataFrame(index=close.index)
    result["MACD_12_26_9"]  = indicator.macd()          # col 0
    result["MACDh_12_26_9"] = indicator.macd_diff()     # col 1
    result["MACDs_12_26_9"] = indicator.macd_signal()   # col 2
    return result


def adx(high, low, close, length=14):
    """
    Average Directional Index. Returns DataFrame with ADX_<length> column.
    """
    high = _ensure_series(high)
    low = _ensure_series(low)
    close = _ensure_series(close)
    indicator = _ta.trend.ADXIndicator(high, low, close, window=length)
    result = pd.DataFrame(index=close.index)
    result[f"ADX_{length}"] = indicator.adx()
    result[f"DMP_{length}"] = indicator.adx_pos()
    result[f"DMN_{length}"] = indicator.adx_neg()
    return result


# ──────────────────────────────────────────────────
#   VOLATILITY
# ──────────────────────────────────────────────────
def atr(high, low, close, length=14):
    """Average True Range."""
    high = _ensure_series(high)
    low = _ensure_series(low)
    close = _ensure_series(close)
    return _ta.volatility.AverageTrueRange(high, low, close, window=length).average_true_range()


def bbands(close, length=20, std=2):
    """
    Bollinger Bands. Returns DataFrame with 3 columns:
      col 0: Lower Band
      col 1: Middle Band
      col 2: Upper Band
    (Same column order as pandas_ta)
    """
    close = _ensure_series(close)
    indicator = _ta.volatility.BollingerBands(close, window=length, window_dev=std)
    result = pd.DataFrame(index=close.index)
    result["BBL"] = indicator.bollinger_lband()   # col 0
    result["BBM"] = indicator.bollinger_mavg()    # col 1
    result["BBU"] = indicator.bollinger_hband()   # col 2
    return result


# ──────────────────────────────────────────────────
#   VOLUME
# ──────────────────────────────────────────────────
def obv(close, volume):
    """On-Balance Volume."""
    close = _ensure_series(close)
    volume = _ensure_series(volume)
    return _ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
