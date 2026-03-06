"""
====================================================
PROJECT AEGIS — Portfolio Rebalancing Engine (Phase 8)
====================================================
Weekly auto-rebalance across sectors and stocks based on:
 ● Kelly-optimal allocation (from kelly_sizing.py)
 ● Sector momentum scores (from sector_rotation.py)
 ● Risk parity weights (from risk_parity.py)
 ● Correlation-adjusted diversification

The engine computes target weights and generates rebalance
orders (paper-only), logging everything for the dashboard.

Run:
    python src/rebalancer.py
====================================================
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("aegis.rebalancer")

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_REBALANCE = os.path.join(DATA, "rebalance_state.json")
FILE_REBALANCE_LOG = os.path.join(DATA, "rebalance_log.json")

# ── Sector mapping (NSE stocks) ──
SECTOR_MAP = {
    "RELIANCE.NS":   "Energy",
    "HDFCBANK.NS":   "Banking",
    "ICICIBANK.NS":  "Banking",
    "SBIN.NS":       "Banking",
    "INFY.NS":       "IT",
    "TCS.NS":        "IT",
    "TATASTEEL.NS":  "Metals",
    "NTPC.NS":       "Power",
    "POWERGRID.NS":  "Power",
    "COALINDIA.NS":  "Mining",
}


# ══════════════════════════════════════════════════
#  CURRENT PORTFOLIO SNAPSHOT
# ══════════════════════════════════════════════════
def get_current_allocations(
    active_trades: List[dict],
    capital: float,
    current_prices: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Calculate current portfolio allocation weights.
    Returns dict with per-stock and per-sector weights.
    """
    positions = {}
    for trade in active_trades:
        if trade.get("status") != "OPEN":
            continue
        sym = trade.get("stock", "")
        price = current_prices.get(sym, trade.get("price", 0)) if current_prices else trade.get("price", 0)
        qty = trade.get("qty", 0)
        value = price * qty
        positions[sym] = positions.get(sym, 0) + value

    total = sum(positions.values())
    cash = max(0, capital - total)

    stock_weights = {sym: (val / capital * 100) if capital > 0 else 0
                     for sym, val in positions.items()}

    sector_weights = {}
    for sym, val in positions.items():
        sector = SECTOR_MAP.get(sym, "Other")
        sector_weights[sector] = sector_weights.get(sector, 0) + val
    sector_pcts = {s: (v / capital * 100) if capital > 0 else 0
                   for s, v in sector_weights.items()}

    return {
        "positions": {k: round(v, 2) for k, v in positions.items()},
        "stock_weights": {k: round(v, 2) for k, v in stock_weights.items()},
        "sector_weights": {k: round(v, 2) for k, v in sector_pcts.items()},
        "total_invested": round(total, 2),
        "cash": round(cash, 2),
        "cash_pct": round((cash / capital * 100) if capital > 0 else 100, 2),
    }


# ══════════════════════════════════════════════════
#  TARGET WEIGHT COMPUTATION
# ══════════════════════════════════════════════════
def compute_target_weights(
    stocks: List[str],
    capital: float,
    kelly_map: Optional[Dict] = None,
    sector_momentum: Optional[Dict] = None,
    risk_parity_weights: Optional[Dict] = None,
    max_single_stock: float = 25.0,
    max_single_sector: float = 40.0,
    min_cash_pct: float = 20.0,
) -> Dict:
    """
    Compute target allocation weights by blending:
     - Kelly fractions (40% weight)
     - Sector momentum (30% weight)
     - Risk parity (30% weight)

    Returns {symbol: target_pct, "_cash": cash_pct}.
    """
    n = len(stocks)
    if n == 0:
        return {"_cash": 100.0}

    # Base: equal weight
    equal_w = (100.0 - min_cash_pct) / n

    final_weights = {}
    for sym in stocks:
        w = equal_w

        # Kelly influence (40%)
        if kelly_map and sym in kelly_map:
            kelly_pct = kelly_map[sym].get("fraction", 0.10) * 100
            w = w * 0.60 + kelly_pct * 0.40

        # Sector momentum influence (30%)
        if sector_momentum:
            sector = SECTOR_MAP.get(sym, "Other")
            sectors_data = sector_momentum.get("sectors", {})
            if sector in sectors_data:
                state = sectors_data[sector].get("state", "HOLD")
                if state in ("STRONG", "ROTATING_IN"):
                    w *= 1.30  # Boost 30%
                elif state in ("ROTATING_OUT", "WEAKENING"):
                    w *= 0.70  # Reduce 30%

        # Risk parity influence (30%)
        if risk_parity_weights and sym in risk_parity_weights:
            rp_w = risk_parity_weights[sym] * (100 - min_cash_pct)
            w = w * 0.70 + rp_w * 0.30

        final_weights[sym] = w

    # Normalize so total ≤ (100 - min_cash)
    total_w = sum(final_weights.values())
    max_invest = 100.0 - min_cash_pct
    if total_w > max_invest:
        scale = max_invest / total_w
        final_weights = {k: v * scale for k, v in final_weights.items()}

    # Apply per-stock cap
    for sym in final_weights:
        final_weights[sym] = min(final_weights[sym], max_single_stock)

    # Apply per-sector cap
    sector_totals = {}
    for sym, w in final_weights.items():
        sector = SECTOR_MAP.get(sym, "Other")
        sector_totals[sector] = sector_totals.get(sector, 0) + w

    for sector, total in sector_totals.items():
        if total > max_single_sector:
            scale = max_single_sector / total
            for sym in final_weights:
                if SECTOR_MAP.get(sym, "Other") == sector:
                    final_weights[sym] *= scale

    # Round
    final_weights = {k: round(v, 2) for k, v in final_weights.items()}
    cash_pct = round(100.0 - sum(final_weights.values()), 2)
    final_weights["_cash"] = max(cash_pct, min_cash_pct)

    return final_weights


# ══════════════════════════════════════════════════
#  REBALANCE ORDER GENERATION
# ══════════════════════════════════════════════════
def generate_rebalance_orders(
    current_alloc: Dict,
    target_weights: Dict,
    capital: float,
    current_prices: Dict[str, float],
    threshold_pct: float = 3.0,
) -> List[Dict]:
    """
    Compare current vs target allocations and generate orders.
    Only rebalance if drift exceeds threshold_pct.
    """
    orders = []
    current_weights = current_alloc.get("stock_weights", {})

    all_symbols = set(list(current_weights.keys()) + [
        k for k in target_weights.keys() if k != "_cash"
    ])

    for sym in all_symbols:
        cur_w = current_weights.get(sym, 0)
        tgt_w = target_weights.get(sym, 0)
        drift = tgt_w - cur_w

        if abs(drift) < threshold_pct:
            continue

        price = current_prices.get(sym, 0)
        if price <= 0:
            continue

        target_value = (tgt_w / 100.0) * capital
        current_value = current_alloc["positions"].get(sym, 0)
        delta_value = target_value - current_value
        delta_qty = int(delta_value / price) if price > 0 else 0

        if delta_qty == 0:
            continue

        orders.append({
            "symbol": sym,
            "action": "BUY" if delta_qty > 0 else "SELL",
            "qty": abs(delta_qty),
            "price": round(price, 2),
            "value": round(abs(delta_value), 2),
            "current_weight": round(cur_w, 2),
            "target_weight": round(tgt_w, 2),
            "drift": round(drift, 2),
        })

    # Sort: sells first (free up capital), then buys
    orders.sort(key=lambda o: (0 if o["action"] == "SELL" else 1, -o["value"]))
    return orders


# ══════════════════════════════════════════════════
#  FULL REBALANCE RUN
# ══════════════════════════════════════════════════
def run_rebalance(
    active_trades: List[dict],
    stocks: List[str],
    capital: float,
    current_prices: Dict[str, float],
    kelly_map: Optional[Dict] = None,
    sector_momentum: Optional[Dict] = None,
    risk_parity_weights: Optional[Dict] = None,
) -> Dict:
    """
    Full rebalance cycle: analyse → compute targets → generate orders.
    Returns comprehensive result dict.
    """
    result = {
        "timestamp": datetime.now(IST).isoformat(),
        "capital": capital,
        "stocks": stocks,
        "current_allocation": {},
        "target_weights": {},
        "orders": [],
        "turnover": 0.0,
        "turnover_pct": 0.0,
        "num_orders": 0,
        "summary": "",
    }

    # Current state
    current = get_current_allocations(active_trades, capital, current_prices)
    result["current_allocation"] = current

    # Targets
    targets = compute_target_weights(
        stocks, capital, kelly_map, sector_momentum, risk_parity_weights
    )
    result["target_weights"] = targets

    # Orders
    orders = generate_rebalance_orders(current, targets, capital, current_prices)
    result["orders"] = orders
    result["num_orders"] = len(orders)

    total_turnover = sum(o["value"] for o in orders)
    result["turnover"] = round(total_turnover, 2)
    result["turnover_pct"] = round((total_turnover / capital * 100) if capital > 0 else 0, 2)

    buys = [o for o in orders if o["action"] == "BUY"]
    sells = [o for o in orders if o["action"] == "SELL"]
    result["summary"] = (
        f"Rebalance: {len(orders)} orders ({len(buys)} buys, {len(sells)} sells). "
        f"Turnover: ₹{total_turnover:,.2f} ({result['turnover_pct']:.1f}% of capital)."
    )

    return result


# ══════════════════════════════════════════════════
#  SHOULD-REBALANCE CHECK (weekly trigger)
# ══════════════════════════════════════════════════
def should_rebalance(last_rebalance_date: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if it's time to rebalance.
    Default: once per week (Monday after market open).
    """
    now = datetime.now(IST)

    # Only on weekdays
    if now.weekday() >= 5:
        return False, "Weekend — no rebalance"

    # Only on Monday (or if never rebalanced)
    if last_rebalance_date:
        try:
            last = datetime.fromisoformat(last_rebalance_date)
            days_since = (now - last).days
            if days_since < 5:
                return False, f"Last rebalance {days_since}d ago — next due in {5 - days_since}d"
        except Exception:
            pass
    else:
        return True, "First rebalance — proceeding"

    if now.weekday() == 0 and now.hour >= 10:
        return True, "Monday rebalance window open"

    return False, f"Not Monday (weekday={now.weekday()}) — waiting"


# ══════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════
def save_rebalance_state(data: Dict):
    """Save rebalance result to JSON."""
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(FILE_REBALANCE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save rebalance state: {e}")


def load_rebalance_state() -> Dict:
    """Load last rebalance state."""
    try:
        if os.path.exists(FILE_REBALANCE):
            with open(FILE_REBALANCE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def log_rebalance(data: Dict):
    """Append rebalance event to history."""
    try:
        os.makedirs(DATA, exist_ok=True)
        history = []
        if os.path.exists(FILE_REBALANCE_LOG):
            with open(FILE_REBALANCE_LOG) as f:
                history = json.load(f)
        history.append({
            "timestamp": data.get("timestamp", datetime.now(IST).isoformat()),
            "num_orders": data.get("num_orders", 0),
            "turnover": data.get("turnover", 0),
            "summary": data.get("summary", ""),
        })
        # Keep last 52 weeks
        history = history[-52:]
        with open(FILE_REBALANCE_LOG, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to log rebalance: {e}")


# ══════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    test_trades = [
        {"stock": "RELIANCE.NS", "price": 2450, "qty": 1, "status": "OPEN"},
        {"stock": "HDFCBANK.NS", "price": 1650, "qty": 1, "status": "OPEN"},
    ]
    test_stocks = ["RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS", "INFY.NS", "TCS.NS"]
    test_prices = {
        "RELIANCE.NS": 2480, "HDFCBANK.NS": 1670,
        "SBIN.NS": 620, "INFY.NS": 1550, "TCS.NS": 3850,
    }
    test_capital = 1000.0

    print("═" * 60)
    print("  PORTFOLIO REBALANCING ENGINE — Test")
    print("═" * 60)

    result = run_rebalance(test_trades, test_stocks, test_capital, test_prices)
    print(f"\n  Current: {result['current_allocation']['stock_weights']}")
    print(f"  Targets: {result['target_weights']}")
    print(f"  Orders:  {result['num_orders']}")
    for o in result["orders"]:
        print(f"    {o['action']} {o['symbol']}: {o['qty']} shares @ ₹{o['price']:,.2f} (drift {o['drift']:+.1f}%)")
    print(f"\n  {result['summary']}")

    should, reason = should_rebalance()
    print(f"\n  Should rebalance now? {should} — {reason}")
