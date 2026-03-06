"""
====================================================
🌐 PROJECT AEGIS — Intermarket Correlation Engine (Phase 10)
====================================================
Monitors global macro instruments as leading indicators
for Nifty and auto-adjusts portfolio exposure:

  Gold   (GC=F / GLD)     — risk-haven demand
  USD/INR (USDINR=X)      — FII flow proxy
  Crude Oil (CL=F / USO)  — inflation / input cost
  US 10Y Yield (^TNX)     — global rate direction

Signal logic:
  • Rising gold + falling Nifty     → risk-off, reduce
  • Rising USD/INR + falling Nifty  → FII outflow, reduce
  • Crude spike > 2σ                → inflation fear, reduce
  • US 10Y rising sharply           → taper tantrum risk

Returns an exposure multiplier (0.5 – 1.2) to scale
position sizes, plus per-instrument detail.
====================================================
"""

import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import STOCK_WATCHLIST, TOP_N_STOCKS

# ──────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_INTERMARKET = os.path.join(DATA, "intermarket.json")

# Instruments to track
INSTRUMENTS = {
    "gold":     {"ticker": "GC=F",      "name": "Gold Futures",    "period": "3mo", "interval": "1d"},
    "usdinr":   {"ticker": "USDINR=X",  "name": "USD/INR",        "period": "3mo", "interval": "1d"},
    "crude":    {"ticker": "CL=F",      "name": "Crude Oil WTI",  "period": "3mo", "interval": "1d"},
    "us10y":    {"ticker": "^TNX",      "name": "US 10Y Yield",   "period": "3mo", "interval": "1d"},
    "nifty":    {"ticker": "^NSEI",     "name": "Nifty 50",       "period": "3mo", "interval": "1d"},
}

# Cache to avoid repeated downloads within same session
_cache: Dict[str, dict] = {}
_cache_ts: float = 0
CACHE_TTL = 1800  # 30 minutes


# ──────────────────────────────────────────────────
#  DATA FETCHER
# ──────────────────────────────────────────────────
def _fetch(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Download price series with retry."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if hasattr(df.columns, 'droplevel'):
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1 * (attempt + 1))
    return pd.DataFrame()


def _returns(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Simple log returns."""
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return np.log(df["Close"] / df["Close"].shift(window)).dropna()


def _zscore(series: pd.Series, lookback: int = 20) -> float:
    """Z-score of the last value vs recent window."""
    if len(series) < lookback + 1:
        return 0.0
    window = series.iloc[-(lookback + 1):-1]
    mu = float(window.mean())
    sigma = float(window.std())
    if sigma < 1e-10:
        return 0.0
    return (float(series.iloc[-1]) - mu) / sigma


# ──────────────────────────────────────────────────
#  SIGNAL ANALYSIS
# ──────────────────────────────────────────────────
def _analyse_instrument(key: str, df: pd.DataFrame) -> Dict:
    """Compute directional signal for one instrument."""
    if df.empty:
        return {"signal": "NEUTRAL", "z": 0.0, "change_5d": 0.0, "change_20d": 0.0, "price": 0.0}

    price = float(df["Close"].iloc[-1])
    rets = _returns(df)
    z = _zscore(rets)

    # 5-day & 20-day changes
    change_5d = 0.0
    change_20d = 0.0
    if len(df) > 5:
        change_5d = (price / float(df["Close"].iloc[-6]) - 1) * 100
    if len(df) > 20:
        change_20d = (price / float(df["Close"].iloc[-21]) - 1) * 100

    # Directional label
    if z > 1.5:
        signal = "STRONG_UP"
    elif z > 0.5:
        signal = "UP"
    elif z < -1.5:
        signal = "STRONG_DOWN"
    elif z < -0.5:
        signal = "DOWN"
    else:
        signal = "NEUTRAL"

    return {
        "signal": signal,
        "z": round(z, 3),
        "change_5d": round(change_5d, 2),
        "change_20d": round(change_20d, 2),
        "price": round(price, 2),
    }


def _correlation_matrix(dataframes: Dict[str, pd.DataFrame], lookback: int = 60) -> Dict:
    """Cross-instrument correlation over rolling window."""
    rets = {}
    for key, df in dataframes.items():
        r = _returns(df).tail(lookback)
        if len(r) > 10:
            rets[key] = r
    if len(rets) < 2:
        return {}
    combined = pd.DataFrame(rets).dropna()
    if combined.empty:
        return {}
    corr = combined.corr()
    result = {}
    for i in corr.index:
        for j in corr.columns:
            if i < j:
                result[f"{i}_vs_{j}"] = round(float(corr.loc[i, j]), 3)
    return result


# ──────────────────────────────────────────────────
#  EXPOSURE MULTIPLIER LOGIC
# ──────────────────────────────────────────────────
def _compute_exposure_multiplier(signals: Dict[str, Dict]) -> tuple:
    """
    Compute portfolio exposure multiplier (0.5 – 1.2).
    Returns (multiplier, reasons).
    """
    multiplier = 1.0
    reasons = []

    gold = signals.get("gold", {})
    usdinr = signals.get("usdinr", {})
    crude = signals.get("crude", {})
    us10y = signals.get("us10y", {})
    nifty = signals.get("nifty", {})

    nifty_dir = nifty.get("signal", "NEUTRAL")
    nifty_z = nifty.get("z", 0)

    # ── Rule 1: Gold surging + Nifty weak → risk-off ──
    gold_z = gold.get("z", 0)
    if gold_z > 1.0 and nifty_z < -0.5:
        multiplier -= 0.15
        reasons.append(f"Gold surging (z={gold_z:+.1f}) + Nifty weak → risk-off")

    # ── Rule 2: USD/INR rising + Nifty weak → FII outflow ──
    usdinr_z = usdinr.get("z", 0)
    if usdinr_z > 1.0 and nifty_z < 0:
        multiplier -= 0.15
        reasons.append(f"USD/INR rising (z={usdinr_z:+.1f}) → FII outflow pressure")

    # ── Rule 3: Crude oil spike → inflation risk ──
    crude_z = crude.get("z", 0)
    if crude_z > 1.5:
        multiplier -= 0.10
        reasons.append(f"Crude spike (z={crude_z:+.1f}) → inflation fear")

    # ── Rule 4: US 10Y yield surging → taper tantrum ──
    us10y_z = us10y.get("z", 0)
    if us10y_z > 1.5:
        multiplier -= 0.10
        reasons.append(f"US 10Y yield surging (z={us10y_z:+.1f}) → rate risk")

    # ── Rule 5: Favourable macro → boost ──
    if (gold_z < 0 and usdinr_z < -0.5 and nifty_z > 0.5):
        multiplier += 0.10
        reasons.append("Favourable macro: Gold down, INR strengthening, Nifty up")

    if nifty_z > 1.0 and crude_z < 0:
        multiplier += 0.05
        reasons.append("Nifty strong + crude cooling")

    # Clamp
    multiplier = max(0.50, min(1.20, multiplier))

    if not reasons:
        reasons.append("Macro conditions neutral — no adjustment")

    return round(multiplier, 3), reasons


# ──────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────
def analyse_intermarket(use_cache: bool = True) -> Dict:
    """
    Full intermarket analysis.

    Returns:
      {
        "exposure_multiplier": float,
        "reasons": list[str],
        "macro_bias": str,  # RISK_ON / RISK_OFF / NEUTRAL
        "instruments": {key: {signal, z, change_5d/20d, price}},
        "correlations": {pair: corr},
        "timestamp": str,
      }
    """
    global _cache, _cache_ts

    if use_cache and _cache and (time.time() - _cache_ts < CACHE_TTL):
        return _cache

    dataframes = {}
    signals = {}

    for key, meta in INSTRUMENTS.items():
        df = _fetch(meta["ticker"], meta["period"], meta["interval"])
        dataframes[key] = df
        signals[key] = _analyse_instrument(key, df)

    multiplier, reasons = _compute_exposure_multiplier(signals)
    corrs = _correlation_matrix(dataframes)

    # Macro bias label
    if multiplier >= 1.05:
        bias = "RISK_ON"
    elif multiplier <= 0.85:
        bias = "RISK_OFF"
    else:
        bias = "NEUTRAL"

    result = {
        "exposure_multiplier": multiplier,
        "reasons": reasons,
        "macro_bias": bias,
        "instruments": signals,
        "correlations": corrs,
        "timestamp": datetime.now(IST).isoformat(),
    }

    _cache = result
    _cache_ts = time.time()
    return result


def get_intermarket_gate(data: dict = None) -> tuple:
    """
    Gate check: block if extreme risk-off.
    Returns (ok, reason, data).
    """
    if data is None:
        data = analyse_intermarket()
    mult = data.get("exposure_multiplier", 1.0)
    bias = data.get("macro_bias", "NEUTRAL")

    if mult <= 0.60:
        return False, f"Intermarket RISK-OFF (mult={mult:.0%}): {'; '.join(data.get('reasons', []))}", data
    if mult <= 0.80:
        return True, f"Intermarket caution (mult={mult:.0%})", data
    return True, f"Macro {bias} (mult={mult:.0%})", data


def get_exposure_multiplier(data: dict = None) -> float:
    """Quick accessor for the exposure multiplier."""
    if data is None:
        data = analyse_intermarket()
    return data.get("exposure_multiplier", 1.0)


def save_intermarket_state(data: dict = None):
    """Save to JSON for dashboard."""
    os.makedirs(DATA, exist_ok=True)
    if data is None:
        data = analyse_intermarket()
    with open(FILE_INTERMARKET, "w") as f:
        json.dump(data, f, indent=2, default=str)
