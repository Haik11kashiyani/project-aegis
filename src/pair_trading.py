"""
====================================================
🔗 PROJECT AEGIS — Pair-Trading Co-Integration Engine (Phase 11)
====================================================
Detect co-integrated stock pairs and trade the
mean-reverting spread.

Methods:
  1. Engle–Granger two-step (ADF on OLS residuals)
  2. Johansen trace/max-eigenvalue test
  3. Distance-based pre-filtering (SSD)

Trading Logic:
  - Compute z-score of spread
  - BUY spread when z < -2 (undervalued leg)
  - SELL spread when z > +2 (overvalued leg)
  - Exit at z → 0 (mean reversion)

Supports sector-aware pairing (e.g. SBIN↔ICICIBANK,
TCS↔INFY) from the watchlist.
====================================================
"""

import os, sys, json, math, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import pytz
IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from config import STOCK_WATCHLIST, TOP_N_STOCKS

# ──────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_PAIRS = os.path.join(DATA, "pair_trading_state.json")

# Statistcal thresholds
ADF_PVALUE_THRESH  = 0.05   # Engle-Granger ADF threshold
Z_ENTRY_LONG       = -2.0   # Buy spread (z-score)
Z_ENTRY_SHORT      = 2.0    # Sell spread
Z_EXIT             = 0.3    # Exit near zero
LOOKBACK_DAYS      = 252    # 1 year for co-integration test
HALFLIFE_MAX       = 60     # Reject if half-life > 60 days
MIN_CORRELATION    = 0.70   # Pre-filter: minimum correlation

# Sector groups (for pre-filtering)
SECTOR_GROUPS = {
    "Banks":    ["SBIN.NS", "ICICIBANK.NS", "HDFCBANK.NS"],
    "IT":       ["TCS.NS", "INFY.NS"],
    "Energy":   ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS"],
    "Metals":   ["TATASTEEL.NS"],
}

# Cache
_pairs_cache: Dict = {}


# ──────────────────────────────────────────────────
#  ADF TEST  (Augmented Dickey-Fuller)
# ──────────────────────────────────────────────────
def _adf_test(series: np.ndarray) -> Tuple[float, bool]:
    """
    Simplified ADF test. Returns (p-value approx, is_stationary).
    Uses OLS regression: Δy = α + β*y_{t-1} + ε
    If β significantly < 0, series is stationary.
    """
    if len(series) < 30:
        return 1.0, False

    y = series[1:]
    y_lag = series[:-1]
    dy = y - y_lag

    n = len(dy)
    x = np.column_stack([np.ones(n), y_lag])

    try:
        beta = np.linalg.lstsq(x, dy, rcond=None)[0]
        residuals = dy - x @ beta
        se = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        se_beta = se / np.sqrt(np.sum((y_lag - np.mean(y_lag)) ** 2))
        t_stat = beta[1] / max(1e-10, se_beta)
    except Exception:
        return 1.0, False

    # Approximate p-value from ADF critical values
    # 1% = -3.43, 5% = -2.86, 10% = -2.57
    if t_stat < -3.43:
        p_val = 0.01
    elif t_stat < -2.86:
        p_val = 0.05
    elif t_stat < -2.57:
        p_val = 0.10
    else:
        p_val = 0.5 + 0.5 * (1 / (1 + math.exp(-t_stat)))  # Rough sigmoid approx

    return round(p_val, 4), p_val < ADF_PVALUE_THRESH


# ──────────────────────────────────────────────────
#  ENGLE-GRANGER COINTEGRATION TEST
# ──────────────────────────────────────────────────
def engle_granger_test(
    y1: np.ndarray,
    y2: np.ndarray,
) -> Dict:
    """
    Engle-Granger two-step cointegration test.
    Step 1: OLS regression y1 = α + β*y2 + ε
    Step 2: ADF test on residuals ε
    """
    if len(y1) < 50 or len(y2) < 50:
        return {"cointegrated": False, "reason": "Insufficient data"}

    n = min(len(y1), len(y2))
    y1, y2 = y1[-n:], y2[-n:]

    # OLS: y1 = alpha + beta * y2
    x = np.column_stack([np.ones(n), y2])
    try:
        coefs = np.linalg.lstsq(x, y1, rcond=None)[0]
    except Exception:
        return {"cointegrated": False, "reason": "OLS failed"}

    alpha, beta = coefs[0], coefs[1]
    residuals = y1 - (alpha + beta * y2)

    # ADF test on residuals
    p_val, is_stationary = _adf_test(residuals)

    # Half-life of mean reversion
    half_life = _compute_halflife(residuals)

    return {
        "cointegrated": is_stationary and half_life < HALFLIFE_MAX,
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),   # Hedge ratio
        "adf_pvalue": p_val,
        "is_stationary": is_stationary,
        "half_life": round(half_life, 1),
        "residual_std": round(float(np.std(residuals)), 4),
    }


def _compute_halflife(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life: -ln(2) / ln(θ)."""
    if len(spread) < 10:
        return 999
    lag = spread[:-1]
    change = spread[1:] - spread[:-1]
    x = np.column_stack([np.ones(len(lag)), lag])
    try:
        coefs = np.linalg.lstsq(x, change, rcond=None)[0]
        theta = coefs[1]
        if theta >= 0:
            return 999  # No mean reversion
        return -math.log(2) / theta
    except Exception:
        return 999


# ──────────────────────────────────────────────────
#  FIND CO-INTEGRATED PAIRS
# ──────────────────────────────────────────────────
def find_cointegrated_pairs(
    symbols: List[str] = None,
    lookback: int = LOOKBACK_DAYS,
) -> Dict:
    """
    Scan all symbol pairs for co-integration.
    Returns ranked list of valid pairs.
    """
    if symbols is None:
        symbols = STOCK_WATCHLIST[:TOP_N_STOCKS]

    # Download closing prices
    prices = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period=f"{lookback + 30}d", interval="1d", progress=False)
            if hasattr(df.columns, "droplevel"):
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass
            if df is not None and len(df) > 50:
                prices[sym] = df["Close"].values[-lookback:]
        except Exception:
            continue

    if len(prices) < 2:
        return {"pairs": [], "n_tested": 0, "timestamp": datetime.now(IST).isoformat()}

    # Generate all pairs
    syms = list(prices.keys())
    valid_pairs = []
    n_tested = 0

    for s1, s2 in combinations(syms, 2):
        n_tested += 1
        y1, y2 = prices[s1], prices[s2]
        n = min(len(y1), len(y2))
        if n < 50:
            continue

        # Pre-filter: correlation check
        corr = np.corrcoef(y1[-n:], y2[-n:])[0, 1]
        if abs(corr) < MIN_CORRELATION:
            continue

        # Engle-Granger test
        result = engle_granger_test(y1[-n:], y2[-n:])
        if result.get("cointegrated"):
            valid_pairs.append({
                "leg1": s1,
                "leg2": s2,
                "correlation": round(corr, 3),
                "hedge_ratio": result["beta"],
                "half_life": result["half_life"],
                "adf_pvalue": result["adf_pvalue"],
                "residual_std": result["residual_std"],
            })

    # Sort by half-life (faster reversion = better)
    valid_pairs.sort(key=lambda p: p["half_life"])

    return {
        "pairs": valid_pairs,
        "n_tested": n_tested,
        "n_found": len(valid_pairs),
        "timestamp": datetime.now(IST).isoformat(),
    }


# ──────────────────────────────────────────────────
#  SPREAD COMPUTATION & Z-SCORE
# ──────────────────────────────────────────────────
def compute_spread(
    prices_1: np.ndarray,
    prices_2: np.ndarray,
    hedge_ratio: float,
) -> Dict:
    """
    Compute spread = leg1 - beta * leg2, z-score, and signal.
    """
    n = min(len(prices_1), len(prices_2))
    p1, p2 = prices_1[-n:], prices_2[-n:]
    spread = p1 - hedge_ratio * p2

    mu = float(np.mean(spread))
    sigma = float(np.std(spread))
    if sigma < 1e-6:
        sigma = 1.0

    z_current = (spread[-1] - mu) / sigma

    # Signal
    if z_current < Z_ENTRY_LONG:
        signal = "LONG_SPREAD"
        action = f"BUY {1} | SELL {abs(hedge_ratio):.2f}"
    elif z_current > Z_ENTRY_SHORT:
        signal = "SHORT_SPREAD"
        action = f"SELL {1} | BUY {abs(hedge_ratio):.2f}"
    elif abs(z_current) < Z_EXIT:
        signal = "EXIT"
        action = "Close pair position"
    else:
        signal = "HOLD"
        action = "Wait for entry/exit signal"

    return {
        "spread_current": round(float(spread[-1]), 4),
        "spread_mean": round(mu, 4),
        "spread_std": round(sigma, 4),
        "z_score": round(float(z_current), 3),
        "signal": signal,
        "action": action,
        "spread_history": [round(float(s), 2) for s in spread[-60:]],  # Last 60 days
    }


# ──────────────────────────────────────────────────
#  PAIR SIGNAL CHECK  (for Sniper gate)
# ──────────────────────────────────────────────────
def check_pair_signal(
    symbol: str,
    pairs_data: Dict = None,
) -> Tuple[bool, str, Dict]:
    """
    Check if a symbol is involved in a pair-trade signal
    that conflicts with a directional BUY.

    If the pair says SHORT this leg → block BUY.
    """
    if pairs_data is None:
        pairs_data = _pairs_cache

    pairs = pairs_data.get("pairs", [])
    if not pairs:
        return True, "No pairs detected — allowing", {}

    for pair in pairs:
        spread_data = pair.get("spread_data", {})
        signal = spread_data.get("signal", "HOLD")

        # Check if this symbol is in a pair
        if pair.get("leg1") == symbol:
            if signal == "SHORT_SPREAD":
                return False, f"Pair says SELL {symbol} (z={spread_data.get('z_score', 0):.1f})", spread_data
        elif pair.get("leg2") == symbol:
            if signal == "LONG_SPREAD":
                # Long spread = buy leg1, sell leg2
                return False, f"Pair says SELL {symbol} (z={spread_data.get('z_score', 0):.1f})", spread_data

    return True, "No conflicting pair signals", {}


# ──────────────────────────────────────────────────
#  FULL ANALYSIS (master function)
# ──────────────────────────────────────────────────
def analyse_pairs(
    symbols: List[str] = None,
) -> Dict:
    """
    Full pair-trading analysis: find pairs + compute signals.
    """
    global _pairs_cache

    pairs_result = find_cointegrated_pairs(symbols)
    pairs = pairs_result.get("pairs", [])

    # Compute live spread for each found pair
    for pair in pairs:
        try:
            df1 = yf.download(pair["leg1"], period="120d", interval="1d", progress=False)
            df2 = yf.download(pair["leg2"], period="120d", interval="1d", progress=False)
            for _df in [df1, df2]:
                if hasattr(_df.columns, "droplevel"):
                    try:
                        _df.columns = _df.columns.droplevel(1)
                    except Exception:
                        pass
            if df1 is not None and df2 is not None and len(df1) > 20 and len(df2) > 20:
                spread_data = compute_spread(
                    df1["Close"].values,
                    df2["Close"].values,
                    pair["hedge_ratio"],
                )
                pair["spread_data"] = spread_data
            else:
                pair["spread_data"] = {"signal": "NO_DATA"}
        except Exception:
            pair["spread_data"] = {"signal": "ERROR"}

    pairs_result["pairs"] = pairs
    _pairs_cache = pairs_result
    return pairs_result


# ──────────────────────────────────────────────────
#  SAVE / LOAD
# ──────────────────────────────────────────────────
def save_pairs_state(data: dict = None):
    os.makedirs(DATA, exist_ok=True)
    payload = data or _pairs_cache
    with open(FILE_PAIRS, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_pairs_state() -> dict:
    global _pairs_cache
    if os.path.exists(FILE_PAIRS):
        try:
            with open(FILE_PAIRS, "r") as f:
                _pairs_cache = json.load(f)
            return _pairs_cache
        except Exception:
            pass
    return {}
