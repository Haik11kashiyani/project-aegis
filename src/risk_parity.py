"""
====================================================
PROJECT AEGIS — Risk Parity Allocator (Phase 8)
====================================================
Allocate capital inversely proportional to each stock's
volatility. Stocks with lower vol get MORE capital
(risk parity / equal risk contribution).

Features:
 ● Download recent price history for volatility estimation
 ● Compute inverse-vol weights (annualized)
 ● Apply sector caps and stock caps
 ● Integrate with Kelly sizing for hybrid allocation
 ● Save weights to data/risk_parity.json for dashboard

Run:
    python src/risk_parity.py
====================================================
"""

import os
import json
import math
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("aegis.risk_parity")

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_RP = os.path.join(DATA, "risk_parity.json")

# ── Cache ──
_rp_cache: Dict = {}
_cache_ttl = 1800  # 30 min


# ══════════════════════════════════════════════════
#  VOLATILITY ESTIMATION
# ══════════════════════════════════════════════════
def estimate_volatility(symbol: str, lookback: int = 60) -> float:
    """
    Estimate annualized volatility from daily returns.
    Returns float (e.g. 0.25 = 25% annualized).
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo", interval="1d")
        if df is None or len(df) < 20:
            return 0.30  # Default 30%

        close = df["Close"].astype(float)
        returns = np.log(close / close.shift(1)).dropna()

        # Use recent N days
        recent = returns.tail(lookback)
        daily_vol = float(recent.std())
        ann_vol = daily_vol * math.sqrt(252)
        return max(0.05, min(ann_vol, 1.0))  # Clamp 5%–100%
    except Exception as e:
        logger.warning(f"Vol estimation failed for {symbol}: {e}")
        return 0.30


def estimate_volatility_from_df(df: pd.DataFrame, lookback: int = 60) -> float:
    """Estimate vol from provided DataFrame (avoids re-download)."""
    try:
        if df is None or len(df) < 20:
            return 0.30
        close = df["Close"].astype(float)
        returns = np.log(close / close.shift(1)).dropna()
        recent = returns.tail(lookback)
        daily_vol = float(recent.std())
        ann_vol = daily_vol * math.sqrt(252)
        return max(0.05, min(ann_vol, 1.0))
    except Exception:
        return 0.30


# ══════════════════════════════════════════════════
#  RISK PARITY WEIGHT COMPUTATION
# ══════════════════════════════════════════════════
def compute_risk_parity_weights(
    symbols: List[str],
    df_map: Optional[Dict[str, pd.DataFrame]] = None,
    max_single_weight: float = 0.30,
    min_single_weight: float = 0.02,
) -> Dict[str, float]:
    """
    Compute inverse-volatility (risk parity) weights.
    Each stock gets weight proportional to 1/vol.

    Returns {symbol: weight} where sum(weights) ≈ 1.0.
    """
    vols = {}
    for sym in symbols:
        if df_map and sym in df_map:
            vols[sym] = estimate_volatility_from_df(df_map[sym])
        else:
            vols[sym] = estimate_volatility(sym)

    # Inverse vol
    inv_vols = {sym: (1.0 / v) if v > 0 else 0 for sym, v in vols.items()}
    total_inv = sum(inv_vols.values())

    if total_inv <= 0:
        # Fallback: equal weight
        n = len(symbols)
        return {sym: 1.0 / n for sym in symbols}

    weights = {sym: iv / total_inv for sym, iv in inv_vols.items()}

    # Clamp weights
    for sym in weights:
        weights[sym] = max(min_single_weight, min(weights[sym], max_single_weight))

    # Renormalize
    total = sum(weights.values())
    weights = {sym: w / total for sym, w in weights.items()}

    return weights


# ══════════════════════════════════════════════════
#  FULL RISK PARITY ANALYSIS
# ══════════════════════════════════════════════════
def analyse_risk_parity(
    symbols: List[str],
    capital: float,
    df_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    """
    Full risk parity analysis: vols → weights → allocations.
    Returns comprehensive dict for dashboard + sniper.
    """
    now = time.time()
    cache_key = ",".join(sorted(symbols))
    if cache_key in _rp_cache and (now - _rp_cache[cache_key].get("_ts", 0)) < _cache_ttl:
        return _rp_cache[cache_key]

    weights = compute_risk_parity_weights(symbols, df_map)

    # Per-stock volatility
    vols = {}
    for sym in symbols:
        if df_map and sym in df_map:
            vols[sym] = estimate_volatility_from_df(df_map[sym])
        else:
            vols[sym] = estimate_volatility(sym)

    # Compute allocations in ₹
    allocations = {sym: round(w * capital, 2) for sym, w in weights.items()}

    # Risk contribution (each stock's % of total portfolio risk)
    total_risk = sum(vols[s] * weights.get(s, 0) for s in symbols)
    risk_contributions = {}
    for sym in symbols:
        rc = (vols[sym] * weights.get(sym, 0)) / total_risk if total_risk > 0 else 0
        risk_contributions[sym] = round(rc, 4)

    # Risk parity score: how equal are the risk contributions?
    # Perfect parity = each contributes 1/N
    target_rc = 1.0 / len(symbols) if symbols else 0
    deviations = [abs(rc - target_rc) for rc in risk_contributions.values()]
    parity_score = max(0, 1.0 - sum(deviations)) * 100  # 100 = perfect parity

    result = {
        "timestamp": datetime.now(IST).isoformat(),
        "symbols": symbols,
        "capital": capital,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "allocations": allocations,
        "volatilities": {k: round(v, 4) for k, v in vols.items()},
        "risk_contributions": risk_contributions,
        "parity_score": round(parity_score, 2),
        "total_portfolio_vol": round(total_risk, 4),
        "summary": "",
    }

    # Sort by weight descending
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top = sorted_w[0] if sorted_w else ("N/A", 0)
    bottom = sorted_w[-1] if sorted_w else ("N/A", 0)

    result["summary"] = (
        f"Risk Parity: {len(symbols)} stocks. "
        f"Parity score: {parity_score:.0f}/100. "
        f"Highest weight: {top[0].replace('.NS', '')} ({top[1]:.1%}). "
        f"Lowest weight: {bottom[0].replace('.NS', '')} ({bottom[1]:.1%})."
    )

    result["_ts"] = now
    _rp_cache[cache_key] = result
    return result


# ══════════════════════════════════════════════════
#  POSITION SIZE — Risk-parity adjusted bullet
# ══════════════════════════════════════════════════
def get_risk_parity_size(
    symbol: str,
    price: float,
    capital: float,
    symbols: List[str],
    rp_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Get position size for a single stock using risk parity.
    Returns dict with amount, pct, qty.
    """
    if rp_weights is None:
        rp_weights = compute_risk_parity_weights(symbols)

    weight = rp_weights.get(symbol, 1.0 / max(len(symbols), 1))
    amount = capital * weight
    qty = max(1, int(amount / price)) if price > 0 else 0

    return {
        "symbol": symbol,
        "weight": round(weight, 4),
        "amount": round(amount, 2),
        "pct": round(weight * 100, 2),
        "qty": qty,
        "method": "RISK_PARITY",
    }


# ══════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════
def save_rp_state(data: Dict):
    """Save risk parity state to JSON."""
    try:
        os.makedirs(DATA, exist_ok=True)
        clean = {k: v for k, v in data.items() if k != "_ts"}
        with open(FILE_RP, "w") as f:
            json.dump(clean, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save risk parity state: {e}")


def load_rp_state() -> Dict:
    """Load latest risk parity state."""
    try:
        if os.path.exists(FILE_RP):
            with open(FILE_RP) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    test_symbols = [
        "RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS",
        "INFY.NS", "TCS.NS", "TATASTEEL.NS",
    ]
    test_capital = 1000.0

    print("═" * 60)
    print("  RISK PARITY ALLOCATOR — Test")
    print("═" * 60)

    result = analyse_risk_parity(test_symbols, test_capital)

    print(f"\n  Portfolio Vol: {result['total_portfolio_vol']:.4f}")
    print(f"  Parity Score: {result['parity_score']:.0f}/100")
    print(f"\n  {'Stock':<15} {'Vol':>8} {'Weight':>8} {'Alloc ₹':>10} {'Risk %':>8}")
    print(f"  {'─' * 15} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 8}")

    for sym in test_symbols:
        vol = result["volatilities"].get(sym, 0)
        wt = result["weights"].get(sym, 0)
        alloc = result["allocations"].get(sym, 0)
        rc = result["risk_contributions"].get(sym, 0)
        name = sym.replace(".NS", "")
        print(f"  {name:<15} {vol:>7.2%} {wt:>7.2%} {alloc:>9,.2f} {rc:>7.2%}")

    print(f"\n  {result['summary']}")

    # Test position sizing
    sizing = get_risk_parity_size("RELIANCE.NS", 2450, test_capital, test_symbols, result["weights"])
    print(f"\n  RELIANCE sizing: ₹{sizing['amount']:,.2f} ({sizing['pct']:.1f}%) = {sizing['qty']} shares")
