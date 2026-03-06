"""
====================================================
📊 PROJECT AEGIS — Portfolio Greeks Heat Map (Phase 10)
====================================================
Aggregates portfolio-level option Greeks (delta, gamma,
theta, vega) from live / estimated option hedges and
produces a heat-map snapshot for the dashboard.

Greeks sourced from:
  1. option_chain.py  (live NSE chain if available)
  2. Black-Scholes analytical Greeks (fallback)

Portfolio-level values:
  Net Delta  — directional risk
  Net Gamma  — curvature / acceleration
  Net Theta  — daily time-decay cost
  Net Vega   — volatility sensitivity

A traffic-light system flags when any Greek
exceeds safe thresholds.
====================================================
"""

import os
import sys
import json
import math
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import STOCK_WATCHLIST, TOP_N_STOCKS, CAPITAL

# ──────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_GREEKS = os.path.join(DATA, "greeks_heatmap.json")

RISK_FREE_RATE = 0.065
DEFAULT_IV = 0.25
DAYS_TO_EXPIRY = 30


# ──────────────────────────────────────────────────
#  NORMAL CDF
# ──────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ──────────────────────────────────────────────────
#  BLACK-SCHOLES GREEKS
# ──────────────────────────────────────────────────
def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "put",
) -> Dict[str, float]:
    """
    Full Greeks for a European option.
    Returns: {delta, gamma, theta, vega, premium}
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "premium": 0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)
    vega = S * _norm_pdf(d1) * sqrt_T / 100  # per 1% vol change

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
                 - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365
        premium = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        delta = _norm_cdf(d1) - 1
        theta = (-(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
                 + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365
        premium = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "premium": round(max(0, premium), 2),
    }


# ──────────────────────────────────────────────────
#  PORTFOLIO GREEK AGGREGATION
# ──────────────────────────────────────────────────
def aggregate_greeks(
    active_trades: List[dict],
    capital: float = None,
    vol_estimates: Dict[str, float] = None,
) -> Dict:
    """
    Compute portfolio-level Greeks from open positions.

    Each position contributes equity delta = +qty (long stock ≈ +1 delta per share).
    If hedged with puts, the put delta offsets partially.

    Returns per-stock + aggregate Greeks + heat map levels.
    """
    if capital is None:
        capital = CAPITAL
    if vol_estimates is None:
        vol_estimates = {}

    per_stock = {}
    net_delta = 0.0
    net_gamma = 0.0
    net_theta = 0.0
    net_vega = 0.0
    total_premium = 0.0
    total_notional = 0.0

    for trade in active_trades:
        if trade.get("status") != "OPEN":
            continue
        sym = trade.get("stock", "UNKNOWN")
        price = float(trade.get("price", 0))
        qty = int(trade.get("qty", 0))
        if price <= 0 or qty <= 0:
            continue

        notional = price * qty
        total_notional += notional

        # Equity Greeks: long stock = delta +1 per share
        stock_delta = qty  # in share units
        stock_delta_notional = notional  # ₹ delta

        # Protective put estimate
        iv = vol_estimates.get(sym, DEFAULT_IV)
        T = DAYS_TO_EXPIRY / 365
        strike = price * 0.95  # 5% OTM put
        put_g = bs_greeks(price, strike, T, RISK_FREE_RATE, iv, "put")

        # Put offsets (per-share × qty)
        put_delta = put_g["delta"] * qty
        put_gamma = put_g["gamma"] * qty
        put_theta = put_g["theta"] * qty
        put_vega = put_g["vega"] * qty
        put_prem = put_g["premium"] * qty

        # Net for this stock
        net_stock_delta = stock_delta + put_delta
        net_stock_gamma = put_gamma  # Stock gamma = 0
        net_stock_theta = put_theta  # Stock theta = 0
        net_stock_vega = put_vega

        per_stock[sym] = {
            "price": price,
            "qty": qty,
            "notional": round(notional, 2),
            "iv": round(iv, 3),
            "equity_delta": round(stock_delta, 2),
            "put_delta": round(put_delta, 2),
            "net_delta": round(net_stock_delta, 2),
            "gamma": round(net_stock_gamma, 6),
            "theta": round(net_stock_theta, 2),
            "vega": round(net_stock_vega, 4),
            "put_premium": round(put_prem, 2),
            "hedge_strike": round(strike, 2),
        }

        net_delta += net_stock_delta
        net_gamma += net_stock_gamma
        net_theta += net_stock_theta
        net_vega += net_stock_vega
        total_premium += put_prem

    # Delta as % of capital
    delta_pct = (net_delta * 100 / capital) if capital > 0 else 0
    # Theta as % of capital per day
    theta_pct = (net_theta / capital * 100) if capital > 0 else 0

    # Traffic-light assessment
    alerts = []
    delta_level = "GREEN"
    if abs(delta_pct) > 5:
        delta_level = "RED"
        alerts.append(f"High delta exposure: {delta_pct:+.1f}% of capital")
    elif abs(delta_pct) > 3:
        delta_level = "YELLOW"

    theta_level = "GREEN"
    if abs(theta_pct) > 0.1:
        theta_level = "RED"
        alerts.append(f"High daily theta bleed: {theta_pct:.3f}%")
    elif abs(theta_pct) > 0.05:
        theta_level = "YELLOW"

    gamma_level = "GREEN"
    if abs(net_gamma) > 0.01:
        gamma_level = "YELLOW"

    vega_level = "GREEN"
    if abs(net_vega) > capital * 0.001:
        vega_level = "YELLOW"

    return {
        "per_stock": per_stock,
        "net_delta": round(net_delta, 2),
        "net_gamma": round(net_gamma, 6),
        "net_theta": round(net_theta, 2),
        "net_vega": round(net_vega, 4),
        "total_premium": round(total_premium, 2),
        "total_notional": round(total_notional, 2),
        "delta_pct": round(delta_pct, 2),
        "theta_pct_daily": round(theta_pct, 4),
        "levels": {
            "delta": delta_level,
            "gamma": gamma_level,
            "theta": theta_level,
            "vega": vega_level,
        },
        "alerts": alerts,
        "open_positions": len(per_stock),
        "timestamp": datetime.now(IST).isoformat(),
    }


# ──────────────────────────────────────────────────
#  HEAT MAP MATRIX (for dashboard visualization)
# ──────────────────────────────────────────────────
def generate_heatmap_matrix(greeks_data: Dict) -> Dict:
    """
    Build a 2D matrix for a plotly heatmap.
    Rows = stocks, Columns = [Delta, Gamma, Theta, Vega]
    Values normalized to [-1, +1] for colour scaling.
    """
    per_stock = greeks_data.get("per_stock", {})
    if not per_stock:
        return {"stocks": [], "metrics": [], "matrix": []}

    stocks = list(per_stock.keys())
    metrics = ["Delta", "Gamma", "Theta", "Vega"]
    matrix = []

    # Collect raw values
    raw = {"Delta": [], "Gamma": [], "Theta": [], "Vega": []}
    for sym in stocks:
        d = per_stock[sym]
        raw["Delta"].append(d.get("net_delta", 0))
        raw["Gamma"].append(d.get("gamma", 0))
        raw["Theta"].append(d.get("theta", 0))
        raw["Vega"].append(d.get("vega", 0))

    # Normalise each metric to [-1, +1]
    for metric in metrics:
        vals = raw[metric]
        if not vals:
            matrix.append([0] * len(stocks))
            continue
        max_abs = max(abs(v) for v in vals) or 1.0
        matrix.append([round(v / max_abs, 3) for v in vals])

    # Transpose: dashboard expects [stocks × metrics]
    matrix_t = list(map(list, zip(*matrix))) if matrix else []

    return {
        "stocks": [s.replace(".NS", "") for s in stocks],
        "metrics": metrics,
        "matrix": matrix_t,
    }


# ──────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────
def analyse_portfolio_greeks(
    active_trades: List[dict] = None,
    capital: float = None,
) -> Dict:
    """
    Full analysis + heat map generation.
    """
    if active_trades is None:
        active_trades = []
    greeks = aggregate_greeks(active_trades, capital)
    heatmap = generate_heatmap_matrix(greeks)
    greeks["heatmap"] = heatmap
    return greeks


def save_greeks_state(data: dict = None):
    """Persist for dashboard."""
    os.makedirs(DATA, exist_ok=True)
    if data is None:
        data = {"timestamp": datetime.now(IST).isoformat()}
    with open(FILE_GREEKS, "w") as f:
        json.dump(data, f, indent=2, default=str)
