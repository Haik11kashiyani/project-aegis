"""
====================================================
💧 PROJECT AEGIS — Liquidity Score Filter (Phase 10)
====================================================
Estimates bid-ask spread and average daily volume
percentile for each stock, then blocks entries on
illiquid names which carry higher slippage risk.

Liquidity Score (0-100) combines:
  - Volume percentile vs 60-day average  (40%)
  - Estimated spread quality             (30%)
  - Dollar-volume (₹ turnover)           (30%)

Gate logic:
  score < 20  → BLOCK (very illiquid)
  score < 40  → REDUCE sizing by 50%
  score >= 40 → PASS

Spread estimation (no L2 data):
  spread_bps ≈ 2 × ATR(1) / Close × 10000
  (tight approximation; real spread rarely > intraday range)
====================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import pytz

IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import STOCK_WATCHLIST, TOP_N_STOCKS

# ──────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_LIQUIDITY = os.path.join(DATA, "liquidity_state.json")

# Cache
_liq_cache: Dict = {}
_liq_cache_ts: float = 0
CACHE_TTL = 900  # 15 min


# ──────────────────────────────────────────────────
#  SPREAD ESTIMATOR
# ──────────────────────────────────────────────────
def estimate_spread_bps(df: pd.DataFrame) -> float:
    """
    Estimate bid-ask spread in basis points from OHLC data.
    Uses Corwin-Schultz (2012) high-low estimator as approximation.
    Falls back to 2 × ATR(1) / Close if not enough data.
    """
    if df is None or len(df) < 5:
        return 50.0  # default 50 bps

    try:
        high = df["High"].values[-20:]
        low = df["Low"].values[-20:]
        close = df["Close"].values[-20:]

        # Corwin-Schultz beta
        betas = []
        for i in range(1, len(high)):
            h2 = max(high[i], high[i - 1])
            l2 = min(low[i], low[i - 1])
            if h2 > 0 and l2 > 0:
                beta = (np.log(h2 / l2)) ** 2
                betas.append(beta)

        gammas = []
        for i in range(len(high)):
            if high[i] > 0 and low[i] > 0:
                gamma = (np.log(high[i] / low[i])) ** 2
                gammas.append(gamma)

        if betas and gammas:
            avg_beta = np.mean(betas)
            avg_gamma = np.mean(gammas)
            k = avg_beta / avg_gamma if avg_gamma > 0 else 1.0
            # Approximate spread
            alpha = (np.sqrt(2 * avg_beta) - np.sqrt(avg_gamma)) / (
                3 - 2 * np.sqrt(2)
            )
            spread_pct = max(0, 2 * (np.exp(alpha) - 1))
            spread_bps = spread_pct * 10000
            return round(max(2.0, min(spread_bps, 200.0)), 1)
    except Exception:
        pass

    # Fallback: 2 * ATR(1) / Close
    try:
        atr1 = abs(float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1]))
        price = float(df["Close"].iloc[-1])
        if price > 0:
            return round(max(2.0, (atr1 / price) * 10000), 1)
    except Exception:
        pass

    return 50.0


# ──────────────────────────────────────────────────
#  VOLUME PERCENTILE
# ──────────────────────────────────────────────────
def volume_percentile(df: pd.DataFrame, lookback: int = 60) -> float:
    """
    Where does today's volume sit vs the last N days?
    Returns 0-100 percentile.
    """
    if df is None or len(df) < 10:
        return 50.0
    try:
        vols = df["Volume"].tail(lookback + 1).values
        today = vols[-1]
        hist = vols[:-1]
        pct = (np.sum(hist <= today) / len(hist)) * 100 if len(hist) > 0 else 50.0
        return round(float(pct), 1)
    except Exception:
        return 50.0


# ──────────────────────────────────────────────────
#  DOLLAR VOLUME (₹ TURNOVER)
# ──────────────────────────────────────────────────
def rupee_turnover(df: pd.DataFrame, days: int = 5) -> float:
    """Average daily ₹ turnover over last N days."""
    if df is None or len(df) < days:
        return 0.0
    try:
        last = df.tail(days)
        turnover = (last["Close"] * last["Volume"]).mean()
        return round(float(turnover), 0)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────
#  COMPOSITE LIQUIDITY SCORE
# ──────────────────────────────────────────────────
def compute_liquidity_score(
    vol_pct: float,
    spread_bps: float,
    turnover: float,
    turnover_benchmark: float = 5e8,  # ₹50 Cr
) -> float:
    """
    Composite score 0-100.
    Higher = more liquid = safer to trade.
    """
    # Volume component (0-40)
    vol_score = min(40.0, vol_pct * 0.4)

    # Spread component (0-30): lower spread = better
    # 5 bps → 30, 50 bps → 15, 100+ bps → 0
    if spread_bps <= 5:
        spread_score = 30.0
    elif spread_bps <= 100:
        spread_score = 30.0 * (1 - (spread_bps - 5) / 95.0)
    else:
        spread_score = 0.0

    # Turnover component (0-30)
    if turnover_benchmark > 0:
        turnover_ratio = min(1.0, turnover / turnover_benchmark)
    else:
        turnover_ratio = 0.5
    turnover_score = turnover_ratio * 30.0

    total = vol_score + spread_score + turnover_score
    return round(max(0, min(100, total)), 1)


# ──────────────────────────────────────────────────
#  PER-STOCK ANALYSIS
# ──────────────────────────────────────────────────
def analyse_stock_liquidity(symbol: str, df: pd.DataFrame = None) -> Dict:
    """
    Full liquidity analysis for one stock.
    """
    if df is None:
        try:
            df = yf.download(symbol, period="6mo", interval="1d", progress=False)
            if hasattr(df.columns, 'droplevel'):
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return {"symbol": symbol, "score": 0, "gate": "BLOCK", "error": "no_data"}

    spread = estimate_spread_bps(df)
    vol_pct = volume_percentile(df)
    turnover = rupee_turnover(df)
    score = compute_liquidity_score(vol_pct, spread, turnover)

    # Gate decision
    if score < 20:
        gate = "BLOCK"
    elif score < 40:
        gate = "REDUCE"
    else:
        gate = "PASS"

    return {
        "symbol": symbol,
        "score": score,
        "spread_bps": spread,
        "volume_percentile": vol_pct,
        "rupee_turnover": turnover,
        "gate": gate,
    }


# ──────────────────────────────────────────────────
#  BATCH & GATE
# ──────────────────────────────────────────────────
def analyse_all_liquidity(
    symbols: List[str] = None,
) -> Dict:
    """Analyse liquidity for all stocks."""
    global _liq_cache, _liq_cache_ts

    if _liq_cache and (time.time() - _liq_cache_ts < CACHE_TTL):
        return _liq_cache

    if symbols is None:
        symbols = STOCK_WATCHLIST[:TOP_N_STOCKS]

    results = {}
    blocked = 0
    reduced = 0

    for sym in symbols:
        try:
            res = analyse_stock_liquidity(sym)
            results[sym] = res
            if res.get("gate") == "BLOCK":
                blocked += 1
            elif res.get("gate") == "REDUCE":
                reduced += 1
        except Exception as e:
            results[sym] = {"symbol": sym, "score": 0, "gate": "BLOCK", "error": str(e)}
            blocked += 1

    # Average score
    scores = [r["score"] for r in results.values() if "score" in r]
    avg_score = round(np.mean(scores), 1) if scores else 0

    data = {
        "stocks": results,
        "avg_score": avg_score,
        "blocked_count": blocked,
        "reduced_count": reduced,
        "total": len(symbols),
        "timestamp": datetime.now(IST).isoformat(),
    }

    _liq_cache = data
    _liq_cache_ts = time.time()
    return data


def check_liquidity_gate(symbol: str, df: pd.DataFrame = None) -> Tuple[bool, str, Dict]:
    """
    Gate check for one stock.
    Returns (ok, reason, data).
    """
    data = analyse_stock_liquidity(symbol, df)
    gate = data.get("gate", "PASS")
    score = data.get("score", 0)

    if gate == "BLOCK":
        return False, f"Illiquid (score={score:.0f}/100, spread={data.get('spread_bps', 0):.0f}bps)", data
    if gate == "REDUCE":
        return True, f"Low liquidity (score={score:.0f}/100) — sizing halved", data
    return True, f"Liquidity OK (score={score:.0f}/100)", data


def get_liquidity_sizing_factor(symbol: str, df: pd.DataFrame = None) -> float:
    """Return sizing factor: 0.0 (block), 0.5 (reduce), 1.0 (pass)."""
    data = analyse_stock_liquidity(symbol, df)
    gate = data.get("gate", "PASS")
    if gate == "BLOCK":
        return 0.0
    if gate == "REDUCE":
        return 0.5
    return 1.0


def save_liquidity_state(data: dict = None):
    """Save for dashboard."""
    os.makedirs(DATA, exist_ok=True)
    if data is None:
        data = analyse_all_liquidity()
    with open(FILE_LIQUIDITY, "w") as f:
        json.dump(data, f, indent=2, default=str)
