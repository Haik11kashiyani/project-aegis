"""
====================================================
💰 PROJECT AEGIS - Adaptive Position Sizing (Kelly Criterion)
====================================================
Replaces fixed bullet_size = CAPITAL / MAX_BULLETS with
mathematically optimal position sizing using:

  Kelly % = W - (1 - W) / R
    W = Win rate (per stock or overall)
    R = Payoff ratio (avg_win / avg_loss)

Features:
  - Per-stock Kelly from trade history
  - Half-Kelly (conservative default)
  - Floor/ceiling limits to prevent ruin
  - Regime-aware scaling (reduce in bear)
  - Decay toward fixed sizing with low sample count
====================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CAPITAL, MAX_BULLETS, TRADE_LOG_FILE,
)

# ──────────────────────────────────────────────────
#  KELLY CONFIG
# ──────────────────────────────────────────────────
KELLY_FRACTION   = 0.5    # Half-Kelly (more conservative)
MIN_TRADES_KELLY = 10     # Need this many trades before trusting Kelly
MIN_POSITION_PCT = 0.02   # Floor: min 2% of capital per trade
MAX_POSITION_PCT = 0.25   # Ceiling: max 25% of capital per trade
DEFAULT_PCT      = None   # Computed as 1/MAX_BULLETS if None
REGIME_BEAR_SCALE = 0.5   # Scale down Kelly by 50% in bear regime
REGIME_BULL_SCALE = 1.2   # Scale up Kelly by 20% in bull regime (still capped)

KELLY_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "kelly_state.json",
)


# ──────────────────────────────────────────────────
#  COMPUTE PER-STOCK KELLY
# ──────────────────────────────────────────────────
def compute_kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """
    Compute raw Kelly Criterion fraction.
    kelly = W - (1 - W) / R

    Args:
        win_rate: Win probability (0-1)
        payoff_ratio: avg_win / avg_loss (absolute)
    Returns:
        Raw Kelly fraction (can be negative → don't trade)
    """
    if payoff_ratio <= 0:
        return 0.0
    kelly = win_rate - (1 - win_rate) / payoff_ratio
    return kelly


def load_trade_history() -> pd.DataFrame:
    """Load trade history from CSV."""
    try:
        if os.path.exists(TRADE_LOG_FILE) and os.path.getsize(TRADE_LOG_FILE) > 0:
            df = pd.read_csv(TRADE_LOG_FILE)
            if "pnl" in df.columns or "PnL" in df.columns:
                if "pnl" not in df.columns:
                    df.rename(columns={"PnL": "pnl"}, inplace=True)
                return df
    except Exception:
        pass
    return pd.DataFrame()


def compute_stock_kelly_map(
    trade_df: pd.DataFrame = None,
    lookback_days: int = 90,
) -> dict:
    """
    Compute Kelly fraction for each stock from trade history.

    Returns dict: {symbol: {kelly_raw, kelly_adj, win_rate, payoff, trades, recommended_pct}}
    """
    if trade_df is None:
        trade_df = load_trade_history()

    if trade_df.empty:
        return {}

    # Filter to recent trades
    if "date" in trade_df.columns or "exit_date" in trade_df.columns:
        date_col = "exit_date" if "exit_date" in trade_df.columns else "date"
        try:
            trade_df[date_col] = pd.to_datetime(trade_df[date_col])
            cutoff = datetime.now() - timedelta(days=lookback_days)
            trade_df = trade_df[trade_df[date_col] >= cutoff]
        except Exception:
            pass

    sym_col = "symbol" if "symbol" in trade_df.columns else "Symbol"
    if sym_col not in trade_df.columns:
        return {}

    kelly_map = {}
    for sym, grp in trade_df.groupby(sym_col):
        trades = len(grp)
        if trades < 3:
            continue

        wins = grp[grp["pnl"] > 0]
        losses = grp[grp["pnl"] <= 0]

        win_rate = len(wins) / trades
        avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0
        avg_loss = abs(float(losses["pnl"].mean())) if len(losses) > 0 else 1

        payoff = avg_win / avg_loss if avg_loss > 0 else 0
        kelly_raw = compute_kelly_fraction(win_rate, payoff)

        # Apply half-Kelly and limits
        kelly_adj = kelly_raw * KELLY_FRACTION

        # Blend toward default with low sample count
        if trades < MIN_TRADES_KELLY:
            default_pct = 1.0 / MAX_BULLETS
            blend = trades / MIN_TRADES_KELLY
            kelly_adj = kelly_adj * blend + default_pct * (1 - blend)

        # Clamp
        kelly_adj = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, kelly_adj))

        kelly_map[sym] = {
            "kelly_raw": round(kelly_raw, 4),
            "kelly_adj": round(kelly_adj, 4),
            "win_rate": round(win_rate, 4),
            "payoff_ratio": round(payoff, 2),
            "trades": trades,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "recommended_pct": round(kelly_adj * 100, 2),
        }

    return kelly_map


# ──────────────────────────────────────────────────
#  OVERALL PORTFOLIO KELLY
# ──────────────────────────────────────────────────
def compute_overall_kelly(trade_df: pd.DataFrame = None) -> dict:
    """Compute portfolio-wide Kelly stats."""
    if trade_df is None:
        trade_df = load_trade_history()

    if trade_df.empty or "pnl" not in trade_df.columns:
        return {
            "kelly_raw": 0,
            "kelly_adj": round(1.0 / MAX_BULLETS, 4),
            "win_rate": 0.5,
            "payoff_ratio": 1.0,
            "trades": 0,
        }

    trades = len(trade_df)
    wins = trade_df[trade_df["pnl"] > 0]
    losses = trade_df[trade_df["pnl"] <= 0]

    win_rate = len(wins) / trades if trades > 0 else 0.5
    avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0
    avg_loss = abs(float(losses["pnl"].mean())) if len(losses) > 0 else 1
    payoff = avg_win / avg_loss if avg_loss > 0 else 0

    kelly_raw = compute_kelly_fraction(win_rate, payoff)
    kelly_adj = kelly_raw * KELLY_FRACTION
    kelly_adj = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, kelly_adj))

    return {
        "kelly_raw": round(kelly_raw, 4),
        "kelly_adj": round(kelly_adj, 4),
        "win_rate": round(win_rate, 4),
        "payoff_ratio": round(payoff, 2),
        "trades": trades,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
    }


# ──────────────────────────────────────────────────
#  PRIMARY API — Get position size for a trade
# ──────────────────────────────────────────────────
def get_kelly_position_size(
    symbol: str,
    current_price: float,
    capital: float = CAPITAL,
    regime: str = "neutral",
    kelly_map: dict = None,
) -> dict:
    """
    Get optimal position size for a specific trade.

    Args:
        symbol: Stock symbol
        current_price: Current price per share
        capital: Available capital
        regime: 'bull', 'bear', or 'neutral'
        kelly_map: Pre-computed kelly map (avoids re-loading)

    Returns:
        {qty, amount, pct, kelly_pct, regime_adjusted, method}
    """
    if kelly_map is None:
        kelly_map = compute_stock_kelly_map()

    default_pct = 1.0 / MAX_BULLETS
    method = "DEFAULT"

    if symbol in kelly_map:
        pct = kelly_map[symbol]["kelly_adj"]
        method = "KELLY"
    else:
        # No stock-specific data — use overall Kelly
        overall = compute_overall_kelly()
        if overall["trades"] >= MIN_TRADES_KELLY:
            pct = overall["kelly_adj"]
            method = "KELLY_OVERALL"
        else:
            pct = default_pct
            method = "DEFAULT"

    # Regime scaling
    regime_factor = 1.0
    if regime == "bear":
        regime_factor = REGIME_BEAR_SCALE
    elif regime == "bull":
        regime_factor = REGIME_BULL_SCALE

    adjusted_pct = pct * regime_factor
    adjusted_pct = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, adjusted_pct))

    amount = capital * adjusted_pct
    qty = max(1, int(amount / current_price)) if current_price > 0 else 0

    return {
        "qty": qty,
        "amount": round(amount, 2),
        "pct": round(adjusted_pct * 100, 2),
        "kelly_pct": round(pct * 100, 2),
        "regime_adjusted": regime != "neutral",
        "regime_factor": regime_factor,
        "method": method,
    }


# ──────────────────────────────────────────────────
#  SAVE / REPORT
# ──────────────────────────────────────────────────
def save_kelly_state(kelly_map: dict = None, overall: dict = None):
    """Save Kelly state for dashboard consumption."""
    if kelly_map is None:
        kelly_map = compute_stock_kelly_map()
    if overall is None:
        overall = compute_overall_kelly()

    state = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "overall": overall,
        "per_stock": kelly_map,
    }

    try:
        os.makedirs(os.path.dirname(KELLY_STATE_FILE), exist_ok=True)
        with open(KELLY_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception:
        pass

    return state


# ──────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n💰 Kelly Criterion Position Sizing Report")
    print("=" * 55)

    overall = compute_overall_kelly()
    print(f"\n  Portfolio Kelly:")
    print(f"    Win Rate      : {overall['win_rate'] * 100:.1f}%")
    print(f"    Payoff Ratio  : {overall['payoff_ratio']:.2f}")
    print(f"    Raw Kelly     : {overall['kelly_raw'] * 100:.2f}%")
    print(f"    Adjusted Kelly: {overall['kelly_adj'] * 100:.2f}%")
    print(f"    Trades Used   : {overall['trades']}")

    kelly_map = compute_stock_kelly_map()
    if kelly_map:
        print(f"\n  Per-Stock Kelly:")
        for sym, ks in sorted(kelly_map.items(), key=lambda x: x[1]["kelly_adj"], reverse=True):
            print(f"    {sym:15} | WR: {ks['win_rate']*100:5.1f}% | "
                  f"PR: {ks['payoff_ratio']:5.2f} | "
                  f"Kelly: {ks['kelly_adj']*100:5.2f}% | "
                  f"Trades: {ks['trades']}")
    else:
        print(f"\n  No per-stock data yet (need trade history)")

    save_kelly_state(kelly_map, overall)
    print(f"\n  📁 State saved to {KELLY_STATE_FILE}")
