"""
====================================================
🎯 PROJECT AEGIS — Dynamic Stop-Loss Engine (Phase 11)
====================================================
Replaces static ATR-based stops with intelligent
trailing stops using:

1. **Chandelier Exit**: Highest-High - ATR × multiplier
2. **Trailing ATR**: Ratchets up as price advances
3. **Volatility-Adaptive**: Wider in high-vol regimes
4. **Time Decay**: Tightens stops for stale positions
5. **Partial Exit**: Step-down at profit milestones

Each open position gets its own stop state that
is updated on every scan cycle.
====================================================
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import CAPITAL

# ──────────────────────────────────────────────────
#  PATHS & DEFAULTS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_DYN_SL = os.path.join(DATA, "dynamic_stoploss_state.json")

# Chandelier defaults
ATR_PERIOD     = 14
ATR_MULTIPLIER = 3.0    # Chandelier: HH - ATR × mult
TRAIL_STEP     = 0.002  # Min ratchet step (0.2%)
TIGHTEN_DAYS   = 5      # After N days, start tightening
TIGHTEN_RATE   = 0.05   # Reduce multiplier by 5% / day
MIN_MULTIPLIER = 1.5    # Floor after tightening

# Partial exit milestones
PARTIAL_MILESTONES = [
    {"pct_gain": 3.0,  "exit_frac": 0.25, "label": "Lock 25% @ +3%"},
    {"pct_gain": 6.0,  "exit_frac": 0.25, "label": "Lock 50% @ +6%"},
    {"pct_gain": 10.0, "exit_frac": 0.25, "label": "Lock 75% @ +10%"},
]

# In-memory state
_stop_states: Dict[str, Dict] = {}


# ──────────────────────────────────────────────────
#  ATR CALCULATOR
# ──────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Standard ATR from OHLC dataframe."""
    if df is None or len(df) < period + 1:
        return 0.0
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0
    # Wilder's smoothed ATR
    atr = np.mean(tr[:period])
    for t in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[t]) / period
    return float(atr)


# ──────────────────────────────────────────────────
#  CHANDELIER EXIT
# ──────────────────────────────────────────────────
def compute_chandelier_exit(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    multiplier: float = ATR_MULTIPLIER,
    lookback: int = 22,
) -> float:
    """
    Chandelier Exit (long): Highest High(lookback) - ATR × mult.
    """
    if df is None or len(df) < period + 1:
        return 0.0
    atr = compute_atr(df, period)
    hh = float(df["High"].tail(lookback).max())
    chandelier = hh - atr * multiplier
    return round(chandelier, 2)


# ──────────────────────────────────────────────────
#  INIT STOP STATE FOR NEW TRADE
# ──────────────────────────────────────────────────
def init_stop(
    symbol: str,
    entry_price: float,
    entry_date: str,
    initial_stop: float,
    qty: int,
    df: pd.DataFrame = None,
) -> Dict:
    """
    Initialize dynamic stop state for a new position.
    """
    atr = compute_atr(df) if df is not None else entry_price * 0.02
    chandelier = compute_chandelier_exit(df) if df is not None else initial_stop

    state = {
        "symbol": symbol,
        "entry_price": entry_price,
        "entry_date": entry_date,
        "initial_stop": initial_stop,
        "current_stop": max(initial_stop, chandelier),
        "highest_price": entry_price,
        "current_multiplier": ATR_MULTIPLIER,
        "current_atr": atr,
        "original_qty": qty,
        "remaining_qty": qty,
        "partial_exits": [],
        "trailing_active": False,
        "days_held": 0,
        "last_update": datetime.now(IST).isoformat(),
        "history": [{
            "date": entry_date,
            "stop": max(initial_stop, chandelier),
            "price": entry_price,
            "event": "INIT",
        }],
    }
    _stop_states[symbol] = state
    return state


# ──────────────────────────────────────────────────
#  UPDATE TRAILING STOP (per scan cycle)
# ──────────────────────────────────────────────────
def update_trailing_stop(
    symbol: str,
    current_price: float,
    df: pd.DataFrame = None,
) -> Dict:
    """
    Update the trailing stop for an open position.
    Called on each scan cycle (~every 5 min).

    Logic:
    1. Compute new Chandelier from latest data
    2. Ratchet up: new stop = max(old stop, chandelier)
    3. Apply time-decay tightening after TIGHTEN_DAYS
    4. Check partial exit milestones
    5. Never lower the stop
    """
    state = _stop_states.get(symbol)
    if state is None:
        return {"error": f"No stop state for {symbol}"}

    entry = state["entry_price"]
    old_stop = state["current_stop"]

    # --- Update highest price ---
    if current_price > state["highest_price"]:
        state["highest_price"] = current_price
        state["trailing_active"] = True

    # --- ATR & Chandelier ---
    if df is not None and len(df) > ATR_PERIOD + 1:
        atr = compute_atr(df)
        state["current_atr"] = atr
    else:
        atr = state.get("current_atr", entry * 0.02)

    # --- Time decay: tighten after N days ---
    try:
        entry_dt = datetime.fromisoformat(state["entry_date"])
        days_held = (datetime.now(IST) - entry_dt).days
    except Exception:
        days_held = state.get("days_held", 0)
    state["days_held"] = days_held

    mult = state["current_multiplier"]
    if days_held > TIGHTEN_DAYS:
        extra_days = days_held - TIGHTEN_DAYS
        mult = max(MIN_MULTIPLIER, ATR_MULTIPLIER * (1 - TIGHTEN_RATE * extra_days))
        state["current_multiplier"] = round(mult, 2)

    # --- Chandelier stop from high ---
    chandelier_stop = state["highest_price"] - atr * mult

    # --- Trailing percentage stop (backup) ---
    trail_pct_stop = state["highest_price"] * (1 - 0.02 - atr / max(0.01, state["highest_price"]))

    # --- New stop = max of all methods, never lower than old ---
    new_stop = max(old_stop, chandelier_stop, trail_pct_stop)

    # Floor: never above current price - 1 tick
    new_stop = min(new_stop, current_price * 0.995)

    # Floor: never below initial stop unless explicitly activated
    if not state["trailing_active"]:
        new_stop = max(new_stop, state["initial_stop"])

    state["current_stop"] = round(new_stop, 2)

    # --- Partial exit check ---
    pct_gain = (current_price - entry) / entry * 100
    exits_fired = [p["pct_gain"] for p in state["partial_exits"]]

    partial_action = None
    for milestone in PARTIAL_MILESTONES:
        if pct_gain >= milestone["pct_gain"] and milestone["pct_gain"] not in exits_fired:
            exit_qty = int(state["original_qty"] * milestone["exit_frac"])
            if exit_qty > 0 and state["remaining_qty"] > exit_qty:
                state["remaining_qty"] -= exit_qty
                state["partial_exits"].append({
                    "pct_gain": milestone["pct_gain"],
                    "qty": exit_qty,
                    "price": current_price,
                    "date": datetime.now(IST).isoformat(),
                    "label": milestone["label"],
                })
                partial_action = milestone["label"]

    # --- Record history ---
    event = "TRAIL" if new_stop > old_stop else "HOLD"
    if partial_action:
        event = f"PARTIAL: {partial_action}"

    state["history"].append({
        "date": datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
        "stop": state["current_stop"],
        "price": current_price,
        "event": event,
    })

    # Keep history manageable (last 200 entries)
    if len(state["history"]) > 200:
        state["history"] = state["history"][-200:]

    state["last_update"] = datetime.now(IST).isoformat()
    return state


# ──────────────────────────────────────────────────
#  CHECK DYNAMIC EXIT  (replaces static check)
# ──────────────────────────────────────────────────
def check_dynamic_exit(
    symbol: str,
    current_price: float,
    df: pd.DataFrame = None,
) -> Tuple[bool, str, Dict]:
    """
    Main exit check (called by Sniper).
    Returns (should_exit, reason, data).
    """
    state = _stop_states.get(symbol)
    if state is None:
        # No dynamic state — fall back to static
        return False, "No dynamic stop state", {}

    # Update the trailing stop first
    updated = update_trailing_stop(symbol, current_price, df)
    if "error" in updated:
        return False, updated["error"], {}

    stop = state["current_stop"]
    entry = state["entry_price"]

    result = {
        "current_stop": stop,
        "entry_price": entry,
        "highest_price": state["highest_price"],
        "pct_from_high": round((state["highest_price"] - current_price) / max(0.01, state["highest_price"]) * 100, 2),
        "pct_gain": round((current_price - entry) / entry * 100, 2),
        "days_held": state["days_held"],
        "multiplier": state["current_multiplier"],
        "remaining_qty": state["remaining_qty"],
        "partial_exits": len(state["partial_exits"]),
    }

    if current_price <= stop:
        reason = (
            f"DYNAMIC STOP HIT: ₹{current_price:.2f} <= ₹{stop:.2f} "
            f"(from high ₹{state['highest_price']:.2f}, mult={state['current_multiplier']:.1f}×ATR)"
        )
        return True, reason, result

    return False, f"Stop at ₹{stop:.2f} ({result['pct_from_high']:.1f}% from high)", result


# ──────────────────────────────────────────────────
#  REMOVE STOP STATE  (after exit)
# ──────────────────────────────────────────────────
def remove_stop(symbol: str):
    """Remove stop state after position is closed."""
    _stop_states.pop(symbol, None)


def get_all_stop_states() -> Dict:
    """Get all active stop states (for dashboard)."""
    return {
        sym: {
            "entry_price": s["entry_price"],
            "current_stop": s["current_stop"],
            "highest_price": s["highest_price"],
            "multiplier": s["current_multiplier"],
            "days_held": s["days_held"],
            "remaining_qty": s["remaining_qty"],
            "partial_exits": len(s["partial_exits"]),
            "trailing_active": s["trailing_active"],
            "history_len": len(s["history"]),
        }
        for sym, s in _stop_states.items()
    }


# ──────────────────────────────────────────────────
#  SAVE / LOAD
# ──────────────────────────────────────────────────
def save_stoploss_state(data: dict = None):
    os.makedirs(DATA, exist_ok=True)
    payload = data or {
        "stops": {sym: _serialize_stop(s) for sym, s in _stop_states.items()},
        "timestamp": datetime.now(IST).isoformat(),
    }
    with open(FILE_DYN_SL, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_stoploss_state() -> Dict:
    global _stop_states
    if os.path.exists(FILE_DYN_SL):
        try:
            with open(FILE_DYN_SL, "r") as f:
                raw = json.load(f)
            stops = raw.get("stops", {})
            for sym, s in stops.items():
                _stop_states[sym] = s
            return raw
        except Exception:
            pass
    return {}


def _serialize_stop(state: Dict) -> Dict:
    """Serialize stop state for JSON (handle numpy etc.)."""
    s = {}
    for k, v in state.items():
        if isinstance(v, (np.integer,)):
            s[k] = int(v)
        elif isinstance(v, (np.floating,)):
            s[k] = float(v)
        else:
            s[k] = v
    return s
