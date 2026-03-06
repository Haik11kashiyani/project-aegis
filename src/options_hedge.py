"""
====================================================
PROJECT AEGIS — Options Hedging Layer (Phase 8)
====================================================
Auto-calculates protective put requirements when
portfolio exposure exceeds a configurable threshold.
For paper-trading, this logs theoretical hedge costs
so the user can see how hedging would affect P&L.

Features:
 ● Monitors total portfolio exposure (sum of open positions)
 ● Calculates hedge ratio using delta-neutral approach
 ● Estimates put option premiums via Black-Scholes
 ● Logs hedge recommendations to data/hedge_state.json
 ● Integrates with broker_bridge for live hedging (future)
====================================================
"""

import os
import json
import math
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("aegis.options_hedge")

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_HEDGE = os.path.join(DATA, "hedge_state.json")

# ── Constants ──
RISK_FREE_RATE = 0.065          # India 10-year ~6.5%
DEFAULT_VOLATILITY = 0.25       # Annualized vol estimate
DAYS_TO_EXPIRY = 30             # Assume monthly puts
LOT_SIZE_MAP = {                # NSE F&O lot sizes (approximate)
    "RELIANCE.NS": 250,  "HDFCBANK.NS": 550, "ICICIBANK.NS": 1375,
    "SBIN.NS": 1500,     "INFY.NS": 300,     "TCS.NS": 150,
    "TATASTEEL.NS": 1700,"NTPC.NS": 2700,    "POWERGRID.NS": 3600,
    "COALINDIA.NS": 1800,
}
DEFAULT_LOT = 500


# ══════════════════════════════════════════════════
#  BLACK-SCHOLES PUT PRICING
# ══════════════════════════════════════════════════
def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European put price.
    S = spot, K = strike, T = time to expiry (years), r = risk-free rate, sigma = vol
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put_price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return max(put_price, 0.0)


def _estimate_iv(symbol: str, df=None) -> float:
    """Estimate implied vol from recent price history (annualized)."""
    try:
        if df is not None and len(df) > 20:
            returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            daily_vol = float(returns.tail(20).std())
            ann_vol = daily_vol * math.sqrt(252)
            return max(0.10, min(ann_vol, 0.80))  # clamp 10%–80%
    except Exception:
        pass
    return DEFAULT_VOLATILITY


# ══════════════════════════════════════════════════
#  PORTFOLIO EXPOSURE ANALYSIS
# ══════════════════════════════════════════════════
def calculate_portfolio_exposure(active_trades: List[dict], capital: float) -> Dict:
    """
    Calculate total portfolio exposure and per-stock breakdown.
    Returns dict with exposure_pct, per-stock values, hedge_needed flag.
    """
    total_exposure = 0.0
    per_stock = {}
    for trade in active_trades:
        if trade.get("status") != "OPEN":
            continue
        sym = trade.get("stock", "UNKNOWN")
        value = trade.get("price", 0) * trade.get("qty", 0)
        total_exposure += value
        per_stock[sym] = per_stock.get(sym, 0) + value

    exposure_pct = (total_exposure / capital * 100) if capital > 0 else 0
    return {
        "total_exposure": round(total_exposure, 2),
        "exposure_pct": round(exposure_pct, 2),
        "per_stock": {k: round(v, 2) for k, v in per_stock.items()},
        "num_positions": len(per_stock),
    }


# ══════════════════════════════════════════════════
#  HEDGE RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════
def compute_hedge_requirements(
    active_trades: List[dict],
    capital: float,
    exposure_threshold: float = 60.0,
    hedge_ratio: float = 0.50,
    df_map: Optional[Dict] = None,
) -> Dict:
    """
    Compute protective put recommendations for portfolio.

    Args:
        active_trades:     List of open trade dicts from state
        capital:           Total capital
        exposure_threshold: % above which hedging kicks in (default 60%)
        hedge_ratio:       Fraction of exposure to hedge (default 50%)
        df_map:            Optional {symbol: DataFrame} for IV estimation

    Returns:
        Dict with hedge recommendations per stock + total cost.
    """
    exposure = calculate_portfolio_exposure(active_trades, capital)

    result = {
        "timestamp": datetime.now(IST).isoformat(),
        "exposure": exposure,
        "hedge_needed": exposure["exposure_pct"] > exposure_threshold,
        "threshold_pct": exposure_threshold,
        "hedge_ratio": hedge_ratio,
        "recommendations": [],
        "total_hedge_cost": 0.0,
        "cost_as_pct_capital": 0.0,
    }

    if not result["hedge_needed"]:
        result["summary"] = (
            f"Exposure {exposure['exposure_pct']:.1f}% < threshold {exposure_threshold:.0f}%. "
            f"No hedge needed."
        )
        return result

    T = DAYS_TO_EXPIRY / 365.0  # Time to expiry in years

    for sym, value in exposure["per_stock"].items():
        hedge_value = value * hedge_ratio

        # Get current price from trades
        prices = [t["price"] for t in active_trades
                  if t.get("stock") == sym and t.get("status") == "OPEN"]
        spot = prices[-1] if prices else 0
        if spot <= 0:
            continue

        # Estimate IV
        df = df_map.get(sym) if df_map else None
        iv = _estimate_iv(sym, df)

        # Strike = 3% OTM put
        strike = round(spot * 0.97, 2)

        # Price the put
        put_premium = black_scholes_put(spot, strike, T, RISK_FREE_RATE, iv)

        # Number of lots (or shares) to hedge
        lot_size = LOT_SIZE_MAP.get(sym, DEFAULT_LOT)
        shares_to_hedge = max(1, int(hedge_value / spot))
        lots_needed = max(1, math.ceil(shares_to_hedge / lot_size))
        total_premium = put_premium * lots_needed * lot_size

        # Greeks (simplified)
        d1 = 0
        if iv > 0 and T > 0:
            d1 = (math.log(spot / strike) + (RISK_FREE_RATE + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        delta = _norm_cdf(d1) - 1  # Put delta is negative

        rec = {
            "symbol": sym,
            "spot_price": round(spot, 2),
            "strike": strike,
            "expiry_days": DAYS_TO_EXPIRY,
            "iv": round(iv, 4),
            "put_premium": round(put_premium, 2),
            "lots_needed": lots_needed,
            "lot_size": lot_size,
            "total_cost": round(total_premium, 2),
            "delta": round(delta, 4),
            "hedge_value": round(hedge_value, 2),
            "protection_pct": round(((spot - strike) / spot) * 100, 2),
        }
        result["recommendations"].append(rec)
        result["total_hedge_cost"] += total_premium

    result["total_hedge_cost"] = round(result["total_hedge_cost"], 2)
    result["cost_as_pct_capital"] = round(
        (result["total_hedge_cost"] / capital * 100) if capital > 0 else 0, 2
    )
    result["summary"] = (
        f"Exposure {exposure['exposure_pct']:.1f}% > {exposure_threshold:.0f}%. "
        f"Hedge {len(result['recommendations'])} positions. "
        f"Estimated cost: ₹{result['total_hedge_cost']:,.2f} ({result['cost_as_pct_capital']:.2f}% of capital)."
    )
    return result


# ══════════════════════════════════════════════════
#  HEDGE GATE — Called from Sniper buy loop
# ══════════════════════════════════════════════════
def check_hedge_gate(
    active_trades: List[dict],
    capital: float,
    max_unhedged_exposure: float = 80.0,
) -> Tuple[bool, str, Dict]:
    """
    Gate check: should we allow a new trade given current exposure?
    Returns (allow, reason, hedge_info).
    If exposure > max_unhedged_exposure → block new buys until hedged.
    """
    exposure = calculate_portfolio_exposure(active_trades, capital)
    exp_pct = exposure["exposure_pct"]

    if exp_pct >= max_unhedged_exposure:
        reason = (
            f"Exposure {exp_pct:.1f}% ≥ max unhedged {max_unhedged_exposure:.0f}%. "
            f"Block new positions until hedge applied."
        )
        return False, reason, exposure

    if exp_pct >= max_unhedged_exposure * 0.8:
        reason = (
            f"Exposure {exp_pct:.1f}% approaching limit. "
            f"Proceed with caution."
        )
        return True, reason, exposure

    return True, f"Exposure {exp_pct:.1f}% — safe", exposure


# ══════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════
def save_hedge_state(hedge_info: Dict):
    """Save latest hedge analysis to JSON for dashboard."""
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(FILE_HEDGE, "w") as f:
            json.dump(hedge_info, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save hedge state: {e}")


def load_hedge_state() -> Dict:
    """Load last hedge state."""
    try:
        if os.path.exists(FILE_HEDGE):
            with open(FILE_HEDGE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    # Simulate some open trades
    test_trades = [
        {"stock": "RELIANCE.NS", "price": 2450.0, "qty": 2, "status": "OPEN"},
        {"stock": "HDFCBANK.NS", "price": 1650.0, "qty": 3, "status": "OPEN"},
        {"stock": "TCS.NS",      "price": 3800.0, "qty": 1, "status": "OPEN"},
    ]
    test_capital = 1000.0

    print("═" * 60)
    print("  OPTIONS HEDGING LAYER — Test")
    print("═" * 60)

    exposure = calculate_portfolio_exposure(test_trades, test_capital)
    print(f"\n  Exposure: ₹{exposure['total_exposure']:,.2f} ({exposure['exposure_pct']:.1f}%)")

    hedge = compute_hedge_requirements(test_trades, test_capital, exposure_threshold=30)
    print(f"  Hedge needed: {hedge['hedge_needed']}")
    print(f"  Summary: {hedge['summary']}")

    for rec in hedge["recommendations"]:
        print(f"\n  {rec['symbol']}:")
        print(f"    Spot: ₹{rec['spot_price']:,.2f} | Strike: ₹{rec['strike']:,.2f}")
        print(f"    Put premium: ₹{rec['put_premium']:,.2f}")
        print(f"    Lots: {rec['lots_needed']} × {rec['lot_size']} = ₹{rec['total_cost']:,.2f}")
        print(f"    Delta: {rec['delta']:.4f} | OTM: {rec['protection_pct']:.2f}%")

    allow, reason, _ = check_hedge_gate(test_trades, test_capital, max_unhedged_exposure=50)
    print(f"\n  Gate check: {'ALLOW' if allow else 'BLOCK'} — {reason}")
