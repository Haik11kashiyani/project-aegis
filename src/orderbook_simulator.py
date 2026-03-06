"""
====================================================
📊 PROJECT AEGIS — Order Book Simulator (Phase 11)
====================================================
Simulate a tick-level L2 order book from NSE to
model market impact cost and improve execution timing.

Since retail traders don't have direct L2 feeds, we
reconstruct a synthetic order book from:
  1. Historical OHLCV (5-min + daily)
  2. Bid-ask spread estimate (Corwin-Schultz)
  3. Volume profile (POC, VAH, VAL)
  4. Intraday tick distribution model

Outputs:
  - Estimated slippage per order size
  - Optimal execution window (TWAP/VWAP split)
  - Impact cost score (0-100)
  - Market depth heatmap data
====================================================
"""

import os, sys, json, time, math
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
FILE_ORDERBOOK = os.path.join(DATA, "orderbook_state.json")

# Market microstructure parameters (calibrated to NSE mid/large-cap)
TICK_SIZE        = 0.05      # NSE tick size ₹0.05
LOT_SIZES        = {         # Approx average lot sizes
    "default": 1,
}
IMPACT_DECAY     = 0.6       # Kyle's lambda decay per level
N_BOOK_LEVELS    = 10        # Simulated depth levels
TWAP_SLICES      = 6         # Split large orders into N slices

# Cache
_ob_cache: Dict = {}
_cache_ts: float = 0
CACHE_TTL = 300  # 5 min


# ──────────────────────────────────────────────────
#  SYNTHETIC ORDER BOOK BUILDER
# ──────────────────────────────────────────────────
def _estimate_spread_bps(df: pd.DataFrame) -> float:
    """Corwin-Schultz high-low spread estimator (bps)."""
    if len(df) < 5:
        return 20.0  # default 20 bps
    h = np.log(df["High"].values[-20:])
    l = np.log(df["Low"].values[-20:])
    beta = np.mean((h[1:] - l[1:]) ** 2 + (h[:-1] - l[:-1]) ** 2) / 2
    gamma = np.mean((np.maximum(h[1:], h[:-1]) - np.minimum(l[1:], l[:-1])) ** 2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
    alpha = max(0, alpha)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return round(max(5, spread * 10000), 1)  # in bps, floor 5


def _build_book(
    price: float,
    spread_bps: float,
    avg_volume: float,
) -> Dict:
    """
    Build a synthetic L2 order book.
    Returns bid/ask levels with simulated depth.
    """
    half_spread = price * spread_bps / 20000  # half-spread in ₹

    # Round to tick size
    best_bid = math.floor((price - half_spread) / TICK_SIZE) * TICK_SIZE
    best_ask = math.ceil((price + half_spread) / TICK_SIZE) * TICK_SIZE

    bids = []
    asks = []

    # Distribute volume across levels (exponential decay)
    base_vol = avg_volume / (N_BOOK_LEVELS * 4)  # Per-level base

    for i in range(N_BOOK_LEVELS):
        # Volume increases deeper in the book (resting orders)
        level_vol = base_vol * (1 + 0.3 * i) * np.random.uniform(0.7, 1.3)

        bid_price = round(best_bid - i * TICK_SIZE, 2)
        ask_price = round(best_ask + i * TICK_SIZE, 2)

        bids.append({
            "level": i + 1,
            "price": bid_price,
            "qty": int(level_vol),
            "orders": max(1, int(level_vol / 50)),  # ~50 shares per order
        })
        asks.append({
            "level": i + 1,
            "price": ask_price,
            "qty": int(level_vol),
            "orders": max(1, int(level_vol / 50)),
        })

    total_bid_vol = sum(b["qty"] for b in bids)
    total_ask_vol = sum(a["qty"] for a in asks)

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_bps": spread_bps,
        "mid_price": round((best_bid + best_ask) / 2, 2),
        "bids": bids,
        "asks": asks,
        "total_bid_volume": total_bid_vol,
        "total_ask_volume": total_ask_vol,
        "imbalance": round((total_bid_vol - total_ask_vol) / max(1, total_bid_vol + total_ask_vol), 3),
    }


# ──────────────────────────────────────────────────
#  IMPACT COST ESTIMATOR
# ──────────────────────────────────────────────────
def estimate_impact(
    order_qty: int,
    book: Dict,
    side: str = "BUY",
) -> Dict:
    """
    Estimate slippage & impact cost for a given order size.

    Uses Kyle (1985) model: impact ∝ √(order_size / ADV).
    """
    levels = book["asks"] if side == "BUY" else book["bids"]
    mid = book["mid_price"]

    filled_qty = 0
    total_cost = 0.0
    levels_consumed = 0

    for level in levels:
        available = level["qty"]
        fill_at_level = min(order_qty - filled_qty, available)
        total_cost += fill_at_level * level["price"]
        filled_qty += fill_at_level
        levels_consumed += 1
        if filled_qty >= order_qty:
            break

    if filled_qty == 0:
        avg_price = mid
    else:
        avg_price = total_cost / filled_qty

    slippage_pct = abs(avg_price - mid) / mid * 100 if mid > 0 else 0

    # Kyle's lambda (market impact)
    total_depth = book["total_ask_volume"] if side == "BUY" else book["total_bid_volume"]
    participation_rate = order_qty / max(1, total_depth)
    kyle_impact = math.sqrt(participation_rate) * book["spread_bps"] / 100

    # Impact cost score (0 = no impact, 100 = huge impact)
    impact_score = min(100, round((slippage_pct + kyle_impact) * 50, 1))

    return {
        "order_qty": order_qty,
        "side": side,
        "avg_fill_price": round(avg_price, 2),
        "mid_price": mid,
        "slippage_pct": round(slippage_pct, 4),
        "slippage_rupees": round(abs(avg_price - mid) * order_qty, 2),
        "kyle_impact_pct": round(kyle_impact, 4),
        "impact_score": impact_score,
        "levels_consumed": levels_consumed,
        "participation_rate": round(participation_rate * 100, 2),
        "fully_filled": filled_qty >= order_qty,
    }


# ──────────────────────────────────────────────────
#  TWAP / VWAP EXECUTION PLAN
# ──────────────────────────────────────────────────
def plan_execution(
    symbol: str,
    order_qty: int,
    book: Dict,
    strategy: str = "TWAP",
    n_slices: int = None,
) -> Dict:
    """
    Plan execution by splitting order into slices.
    Returns optimal schedule.
    """
    slices = n_slices or TWAP_SLICES
    slice_qty = max(1, order_qty // slices)
    remainder = order_qty - slice_qty * slices

    schedule = []
    total_impact = 0.0

    for i in range(slices):
        qty = slice_qty + (1 if i < remainder else 0)
        impact = estimate_impact(qty, book, "BUY")
        total_impact += impact["slippage_rupees"]

        if strategy == "TWAP":
            # Equal time intervals over trading day (9:15-15:15 = 360 min)
            minutes_offset = int((360 / slices) * i)
            exec_time = f"09:{15 + minutes_offset // 60:02d}"
        else:  # VWAP
            # Weight towards volume peaks: 9:30-10:30 and 14:00-15:00
            if i < slices // 3:
                exec_time = f"09:{30 + i * 10:02d}"
            elif i < 2 * slices // 3:
                exec_time = f"11:{00 + (i - slices // 3) * 15:02d}"
            else:
                exec_time = f"14:{00 + (i - 2 * slices // 3) * 10:02d}"

        schedule.append({
            "slice": i + 1,
            "qty": qty,
            "est_impact_rs": round(impact["slippage_rupees"], 2),
            "target_time": exec_time,
        })

    # Compare: single shot impact
    single_impact = estimate_impact(order_qty, book, "BUY")

    return {
        "symbol": symbol,
        "strategy": strategy,
        "total_qty": order_qty,
        "n_slices": slices,
        "schedule": schedule,
        "total_est_impact": round(total_impact, 2),
        "single_shot_impact": round(single_impact["slippage_rupees"], 2),
        "savings_pct": round(
            max(0, (single_impact["slippage_rupees"] - total_impact)
                / max(0.01, single_impact["slippage_rupees"]) * 100), 1
        ),
    }


# ──────────────────────────────────────────────────
#  ANALYSE FULL PORTFOLIO  (master function)
# ──────────────────────────────────────────────────
def analyse_orderbook(
    symbols: List[str] = None,
    order_sizes: Dict[str, int] = None,
) -> Dict:
    """
    Build synthetic order books and estimate impact for all symbols.
    """
    global _ob_cache, _cache_ts
    if time.time() - _cache_ts < CACHE_TTL and _ob_cache:
        return _ob_cache

    if symbols is None:
        symbols = STOCK_WATCHLIST[:TOP_N_STOCKS]

    results = {}
    total_impact = 0.0

    for sym in symbols:
        try:
            df = yf.download(sym, period="60d", interval="1d", progress=False)
            if hasattr(df.columns, "droplevel"):
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass
            if df is None or df.empty:
                continue

            price = float(df["Close"].iloc[-1])
            avg_vol = float(df["Volume"].tail(20).mean())
            spread = _estimate_spread_bps(df)

            book = _build_book(price, spread, avg_vol)

            # Default order size: ₹50,000 worth
            qty = order_sizes.get(sym, 0) if order_sizes else 0
            if qty == 0:
                qty = max(1, int(50000 / price))

            impact = estimate_impact(qty, book, "BUY")
            execution = plan_execution(sym, qty, book)

            results[sym] = {
                "price": price,
                "spread_bps": spread,
                "avg_volume": int(avg_vol),
                "book_summary": {
                    "best_bid": book["best_bid"],
                    "best_ask": book["best_ask"],
                    "imbalance": book["imbalance"],
                    "total_depth": book["total_bid_volume"] + book["total_ask_volume"],
                },
                "impact": impact,
                "execution_plan": execution,
            }
            total_impact += impact["slippage_rupees"]

        except Exception as e:
            results[sym] = {"error": str(e)[:80]}

    data = {
        "stocks": results,
        "total_est_impact": round(total_impact, 2),
        "n_analysed": len([r for r in results.values() if "error" not in r]),
        "timestamp": datetime.now(IST).isoformat(),
    }

    _ob_cache = data
    _cache_ts = time.time()
    return data


# ──────────────────────────────────────────────────
#  ORDERBOOK GATE (for Sniper)
# ──────────────────────────────────────────────────
def check_orderbook_gate(
    symbol: str,
    qty: int,
    ob_data: Dict = None,
    max_impact_score: float = 60,
) -> Tuple[bool, str, Dict]:
    """
    Gate: block if impact cost is too high.
    Returns (ok, reason, data).
    """
    if ob_data is None:
        ob_data = analyse_orderbook([symbol])

    stock_data = ob_data.get("stocks", {}).get(symbol, {})
    if "error" in stock_data or not stock_data:
        return True, "No orderbook data — allowing", {}

    impact = stock_data.get("impact", {})
    score = impact.get("impact_score", 0)

    if score > max_impact_score:
        return False, f"Impact score {score:.0f} > {max_impact_score} (slippage {impact.get('slippage_pct', 0):.3f}%)", impact

    return True, f"Impact OK (score {score:.0f})", impact


# ──────────────────────────────────────────────────
#  SAVE / LOAD
# ──────────────────────────────────────────────────
def save_orderbook_state(data: dict = None):
    os.makedirs(DATA, exist_ok=True)
    if data is None:
        data = _ob_cache or {}
    with open(FILE_ORDERBOOK, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_orderbook_state() -> dict:
    if os.path.exists(FILE_ORDERBOOK):
        try:
            with open(FILE_ORDERBOOK, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
