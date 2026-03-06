"""
====================================================
⚡ PROJECT AEGIS — Intraday Momentum Scalper (Phase 10)
====================================================
Sub-15-min VWAP-cross strategy for stocks already in
the portfolio.  Captures intraday mean-reversion /
momentum continuation using:

  1. 5-min candles → VWAP + EMA_9
  2. VWAP-cross signal:
       Price crosses ABOVE VWAP + price > EMA_9  → SCALE IN
       Price crosses BELOW VWAP + RSI_5 > 70     → PARTIAL OUT
  3. Volume spike confirmation (2× avg 5-min bar)
  4. Risk: tight 0.3% hard stop on scalp portion
  5. Auto-close all scalp portions by 14:45 IST

This operates as an overlay: it only acts on stocks
the core Sniper already holds.  It does NOT open new
positions — just scales in/out for intraday alpha.
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
from typing import Dict, List, Optional, Tuple

import pytz

IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import CAPITAL, STOCK_WATCHLIST, TOP_N_STOCKS

# ──────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_SCALPER = os.path.join(DATA, "scalper_state.json")

SCALP_HARD_STOP_PCT = 0.003     # 0.3% hard stop
SCALP_TARGET_PCT = 0.005        # 0.5% target
SCALP_MAX_CAPITAL_PCT = 0.10    # Max 10% of capital per scalp
VOLUME_SPIKE_THRESHOLD = 2.0    # 2× average bar volume
RSI_OVERBOUGHT = 70
SCALP_CLOSE_HOUR = 14
SCALP_CLOSE_MINUTE = 45
MIN_BARS_FOR_SIGNAL = 20        # Need 20+ bars of 5-min data

# Cache
_scalp_cache: Dict = {}
_cache_ts: float = 0
CACHE_TTL = 120  # 2 minutes


# ──────────────────────────────────────────────────
#  5-MIN DATA FETCHER
# ──────────────────────────────────────────────────
def _fetch_5min(symbol: str) -> pd.DataFrame:
    """Fetch intraday 5-min candles."""
    for attempt in range(2):
        try:
            df = yf.download(symbol, period="5d", interval="5m", progress=False)
            if hasattr(df.columns, 'droplevel'):
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


# ──────────────────────────────────────────────────
#  INTRADAY INDICATORS (5-min resolution)
# ──────────────────────────────────────────────────
def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute cumulative intraday VWAP."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (tp * df["Volume"]).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def _compute_rsi(series: pd.Series, period: int = 5) -> pd.Series:
    """Fast RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_ema(series: pd.Series, span: int = 9) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ──────────────────────────────────────────────────
#  SIGNAL DETECTION
# ──────────────────────────────────────────────────
def detect_scalp_signal(symbol: str, df: pd.DataFrame = None) -> Dict:
    """
    Detect VWAP-cross scalp signals on 5-min data.

    Returns:
      {
        "signal": "SCALE_IN" | "PARTIAL_OUT" | "HOLD" | "NO_DATA",
        "price": float,
        "vwap": float,
        "ema9": float,
        "rsi5": float,
        "volume_spike": bool,
        "confidence": float (0-1),
        "reason": str,
      }
    """
    if df is None:
        df = _fetch_5min(symbol)

    if df is None or len(df) < MIN_BARS_FOR_SIGNAL:
        return {"signal": "NO_DATA", "reason": "Insufficient data"}

    # Filter to today only (for VWAP reset)
    today = datetime.now(IST).date()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_today = df[df.index.tz_convert(IST).date == today].copy()
    if len(df_today) < 10:
        # Not enough today data; use last available day
        df_today = df.tail(78).copy()  # ~6.5 hours of 5-min

    vwap = _compute_vwap(df_today)
    ema9 = _compute_ema(df_today["Close"], 9)
    rsi5 = _compute_rsi(df_today["Close"], 5)
    avg_vol = df_today["Volume"].rolling(20).mean()

    price = float(df_today["Close"].iloc[-1])
    cur_vwap = float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else price
    cur_ema = float(ema9.iloc[-1]) if not pd.isna(ema9.iloc[-1]) else price
    cur_rsi = float(rsi5.iloc[-1]) if not pd.isna(rsi5.iloc[-1]) else 50
    cur_vol = float(df_today["Volume"].iloc[-1])
    cur_avg_vol = float(avg_vol.iloc[-1]) if not pd.isna(avg_vol.iloc[-1]) else cur_vol

    vol_spike = cur_vol > (VOLUME_SPIKE_THRESHOLD * cur_avg_vol) if cur_avg_vol > 0 else False

    # Previous bar for crossover detection
    prev_price = float(df_today["Close"].iloc[-2])
    prev_vwap = float(vwap.iloc[-2]) if len(vwap) > 1 and not pd.isna(vwap.iloc[-2]) else cur_vwap

    signal = "HOLD"
    reason = "No clear signal"
    confidence = 0.0

    # ── SCALE IN: price crosses above VWAP + above EMA9 ──
    crossed_above_vwap = (prev_price <= prev_vwap) and (price > cur_vwap)
    above_ema = price > cur_ema
    if crossed_above_vwap and above_ema:
        confidence = 0.6
        if vol_spike:
            confidence = 0.8
        if cur_rsi < 60:  # Not overbought
            confidence += 0.1
        signal = "SCALE_IN"
        reason = f"VWAP crossover ↑ (VWAP={cur_vwap:.2f}, EMA9={cur_ema:.2f})"

    # ── PARTIAL OUT: price drops below VWAP + RSI overbought ──
    crossed_below_vwap = (prev_price >= prev_vwap) and (price < cur_vwap)
    if crossed_below_vwap and cur_rsi > RSI_OVERBOUGHT:
        confidence = 0.7
        signal = "PARTIAL_OUT"
        reason = f"VWAP breakdown ↓ + RSI({cur_rsi:.0f}) overbought"

    # ── MEAN REVERSION: price far below VWAP + strong volume ──
    distance_pct = (price - cur_vwap) / cur_vwap * 100 if cur_vwap > 0 else 0
    if distance_pct < -0.5 and vol_spike and cur_rsi < 35:
        confidence = 0.65
        signal = "SCALE_IN"
        reason = f"Mean reversion: price {distance_pct:+.2f}% below VWAP + RSI({cur_rsi:.0f})"

    return {
        "signal": signal,
        "price": round(price, 2),
        "vwap": round(cur_vwap, 2),
        "ema9": round(cur_ema, 2),
        "rsi5": round(cur_rsi, 1),
        "volume_spike": vol_spike,
        "confidence": round(min(1.0, confidence), 3),
        "reason": reason,
        "vwap_distance_pct": round(distance_pct, 3),
    }


# ──────────────────────────────────────────────────
#  SCALP POSITION SIZING
# ──────────────────────────────────────────────────
def calculate_scalp_size(
    price: float, capital: float, confidence: float = 0.5,
) -> Dict:
    """Compute scalp size and risk parameters."""
    max_amount = capital * SCALP_MAX_CAPITAL_PCT * confidence
    qty = max(1, int(max_amount / price)) if price > 0 else 0
    stop = round(price * (1 - SCALP_HARD_STOP_PCT), 2)
    target = round(price * (1 + SCALP_TARGET_PCT), 2)

    return {
        "qty": qty,
        "amount": round(qty * price, 2),
        "stop": stop,
        "target": target,
        "risk": round(qty * (price - stop), 2),
        "reward": round(qty * (target - price), 2),
    }


# ──────────────────────────────────────────────────
#  SCAN ALL HELD POSITIONS
# ──────────────────────────────────────────────────
def scan_scalp_opportunities(
    held_symbols: List[str],
    capital: float = None,
) -> Dict:
    """
    Scan all currently held stocks for scalp signals.
    Returns combined result with per-stock signals.
    """
    if capital is None:
        capital = CAPITAL

    results = {}
    scale_in_count = 0
    partial_out_count = 0

    now = datetime.now(IST)
    # Don't scalp after 14:45 IST
    if now.hour > SCALP_CLOSE_HOUR or (now.hour == SCALP_CLOSE_HOUR and now.minute >= SCALP_CLOSE_MINUTE):
        return {
            "stocks": {},
            "scale_in": 0,
            "partial_out": 0,
            "status": "CLOSED",
            "reason": "Past scalp window (14:45 IST)",
            "timestamp": now.isoformat(),
        }

    for sym in held_symbols:
        try:
            sig = detect_scalp_signal(sym)
            if sig["signal"] == "SCALE_IN":
                sizing = calculate_scalp_size(sig["price"], capital, sig["confidence"])
                sig["sizing"] = sizing
                scale_in_count += 1
            elif sig["signal"] == "PARTIAL_OUT":
                sig["sizing"] = {}
                partial_out_count += 1
            results[sym] = sig
        except Exception as e:
            results[sym] = {"signal": "ERROR", "reason": str(e)[:60]}

    return {
        "stocks": results,
        "scale_in": scale_in_count,
        "partial_out": partial_out_count,
        "status": "ACTIVE",
        "timestamp": now.isoformat(),
    }


# ──────────────────────────────────────────────────
#  SCALP TRACKER (per-day state)
# ──────────────────────────────────────────────────
_scalp_trades: List[dict] = []


def record_scalp(symbol: str, action: str, price: float, qty: int, pnl: float = 0):
    """Record a scalp trade for journaling."""
    _scalp_trades.append({
        "symbol": symbol,
        "action": action,
        "price": price,
        "qty": qty,
        "pnl": round(pnl, 2),
        "time": datetime.now(IST).strftime("%H:%M:%S"),
    })


def get_scalp_summary() -> Dict:
    """Summary of today's scalp trades."""
    total_pnl = sum(t["pnl"] for t in _scalp_trades)
    wins = sum(1 for t in _scalp_trades if t["pnl"] > 0 and t["action"] != "SCALE_IN")
    losses = sum(1 for t in _scalp_trades if t["pnl"] < 0 and t["action"] != "SCALE_IN")
    return {
        "trades": len(_scalp_trades),
        "total_pnl": round(total_pnl, 2),
        "wins": wins,
        "losses": losses,
        "details": _scalp_trades[-20:],
        "timestamp": datetime.now(IST).isoformat(),
    }


def save_scalper_state(data: dict = None):
    """Persist for dashboard."""
    os.makedirs(DATA, exist_ok=True)
    if data is None:
        data = get_scalp_summary()
    with open(FILE_SCALPER, "w") as f:
        json.dump(data, f, indent=2, default=str)
