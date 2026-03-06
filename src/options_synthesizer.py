"""
====================================================
Phase 12 — Options Strategy Synthesizer
====================================================
Automatically generates protective options overlays
for the equity portfolio.  Reads portfolio Greeks
from ``greeks_heatmap.py`` and VaR from
``var_stress.py`` to decide which strategy fits best.

Supported strategies
--------------------
* **Collar**        — Buy protective PUT, sell covered CALL.
* **Iron Condor**   — Sell OTM PUT + CALL spreads around
                      the current index/stock price.
* **Protective Put** — Buy ATM PUT for tail-risk hedging.
* **Straddle**       — Buy ATM PUT + CALL when vol is cheap
                       and regime is uncertain.
* **Bear Put Spread** — Buy higher PUT, sell lower PUT for
                        directional downside protection.

The synthesizer scores each strategy on cost-efficiency,
downside protection, and regime fit, then returns the
top recommendation.

Key functions used by sniper.py
-------------------------------
* ``synthesize_strategy()``   — top recommendation
* ``generate_collar()``       — single-stock collar
* ``generate_iron_condor()``  — index-level condor
* ``get_protection_cost()``   — estimated premium
* ``get_synth_status()``      — dashboard summary
* ``save_synth_state() / load_synth_state()``
====================================================
"""

import os, sys, json, math
from datetime import datetime
import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "options_synth_state.json")

# ── Black-Scholes helpers (for premium estimation) ──
_RISK_FREE = 0.065          # India 10Y ~6.5 %
_DEFAULT_IV = 0.22          # 22 % annualised vol
_DTE_DEFAULT = 30           # days to expiry


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes PUT price.  T in years."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    from math import log, sqrt, exp
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        # Approx CDF via Abramowitz & Stegun
        def _ncdf(x):
            a = 0.2316419; b1 = 0.319381530; b2 = -0.356563782
            b3 = 1.781477937; b4 = -1.821255978; b5 = 1.330274429
            t = 1.0 / (1.0 + a * abs(x))
            poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
            pdf = (1.0 / sqrt(2 * 3.141592653589793)) * exp(-0.5 * x * x)
            cdf = 1.0 - pdf * poly
            return cdf if x >= 0 else 1.0 - cdf
        return K * exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)
    except Exception:
        return 0.0


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes CALL via put-call parity."""
    from math import exp
    put = _bs_put(S, K, T, r, sigma)
    return put + S - K * exp(-r * T)


# --------------------------------------------------
#  Strategy generators
# --------------------------------------------------
def generate_collar(
    symbol: str,
    spot: float,
    qty: int,
    iv: float = _DEFAULT_IV,
    dte: int = _DTE_DEFAULT,
    put_otm_pct: float = 0.05,
    call_otm_pct: float = 0.05,
) -> dict:
    """
    Collar: buy PUT at spot*(1 - put_otm_pct), sell CALL
    at spot*(1 + call_otm_pct).
    """
    T = dte / 365.0
    put_K = round(spot * (1 - put_otm_pct), 2)
    call_K = round(spot * (1 + call_otm_pct), 2)
    put_premium = _bs_put(spot, put_K, T, _RISK_FREE, iv) * qty
    call_premium = _bs_call(spot, call_K, T, _RISK_FREE, iv) * qty
    net_cost = put_premium - call_premium
    max_loss = (spot - put_K) * qty + max(net_cost, 0)
    max_gain = (call_K - spot) * qty - max(net_cost, 0)

    return {
        "strategy": "COLLAR",
        "symbol": symbol,
        "spot": spot,
        "qty": qty,
        "put_strike": put_K,
        "call_strike": call_K,
        "put_premium": round(put_premium, 2),
        "call_premium": round(call_premium, 2),
        "net_cost": round(net_cost, 2),
        "max_loss": round(max_loss, 2),
        "max_gain": round(max_gain, 2),
        "dte": dte,
        "iv": iv,
    }


def generate_iron_condor(
    symbol: str,
    spot: float,
    qty: int,
    iv: float = _DEFAULT_IV,
    dte: int = _DTE_DEFAULT,
    width_pct: float = 0.05,
    wing_pct: float = 0.03,
) -> dict:
    """Short Iron Condor (sell OTM put + call spreads)."""
    T = dte / 365.0
    # Sell strikes
    sell_put_K = round(spot * (1 - width_pct), 2)
    sell_call_K = round(spot * (1 + width_pct), 2)
    # Buy wings
    buy_put_K = round(sell_put_K * (1 - wing_pct), 2)
    buy_call_K = round(sell_call_K * (1 + wing_pct), 2)

    credit = (
        _bs_put(spot, sell_put_K, T, _RISK_FREE, iv)
        + _bs_call(spot, sell_call_K, T, _RISK_FREE, iv)
        - _bs_put(spot, buy_put_K, T, _RISK_FREE, iv)
        - _bs_call(spot, buy_call_K, T, _RISK_FREE, iv)
    ) * qty
    max_loss = ((sell_put_K - buy_put_K) * qty) - credit

    return {
        "strategy": "IRON_CONDOR",
        "symbol": symbol,
        "spot": spot,
        "qty": qty,
        "sell_put": sell_put_K,
        "buy_put": buy_put_K,
        "sell_call": sell_call_K,
        "buy_call": buy_call_K,
        "net_credit": round(credit, 2),
        "max_loss": round(max(max_loss, 0), 2),
        "dte": dte,
        "iv": iv,
    }


def generate_protective_put(
    symbol: str,
    spot: float,
    qty: int,
    iv: float = _DEFAULT_IV,
    dte: int = _DTE_DEFAULT,
    otm_pct: float = 0.03,
) -> dict:
    """Simple protective PUT."""
    T = dte / 365.0
    put_K = round(spot * (1 - otm_pct), 2)
    premium = _bs_put(spot, put_K, T, _RISK_FREE, iv) * qty
    max_loss = (spot - put_K) * qty + premium

    return {
        "strategy": "PROTECTIVE_PUT",
        "symbol": symbol,
        "spot": spot,
        "qty": qty,
        "put_strike": put_K,
        "premium": round(premium, 2),
        "max_loss": round(max_loss, 2),
        "dte": dte,
        "iv": iv,
    }


def generate_straddle(
    symbol: str,
    spot: float,
    qty: int,
    iv: float = _DEFAULT_IV,
    dte: int = _DTE_DEFAULT,
) -> dict:
    """Long straddle (ATM put + call)."""
    T = dte / 365.0
    K = round(spot, 2)
    put_p = _bs_put(spot, K, T, _RISK_FREE, iv) * qty
    call_p = _bs_call(spot, K, T, _RISK_FREE, iv) * qty
    total = put_p + call_p
    breakeven_up = K + total / max(qty, 1)
    breakeven_dn = K - total / max(qty, 1)

    return {
        "strategy": "STRADDLE",
        "symbol": symbol,
        "spot": spot,
        "qty": qty,
        "strike": K,
        "put_premium": round(put_p, 2),
        "call_premium": round(call_p, 2),
        "total_cost": round(total, 2),
        "breakeven_up": round(breakeven_up, 2),
        "breakeven_dn": round(breakeven_dn, 2),
        "dte": dte,
        "iv": iv,
    }


def generate_bear_put_spread(
    symbol: str,
    spot: float,
    qty: int,
    iv: float = _DEFAULT_IV,
    dte: int = _DTE_DEFAULT,
    width_pct: float = 0.05,
) -> dict:
    """Bear put spread — buy higher put, sell lower put."""
    T = dte / 365.0
    buy_K = round(spot, 2)
    sell_K = round(spot * (1 - width_pct), 2)
    debit = (_bs_put(spot, buy_K, T, _RISK_FREE, iv)
             - _bs_put(spot, sell_K, T, _RISK_FREE, iv)) * qty
    max_gain = (buy_K - sell_K) * qty - debit
    max_loss = debit

    return {
        "strategy": "BEAR_PUT_SPREAD",
        "symbol": symbol,
        "spot": spot,
        "qty": qty,
        "buy_strike": buy_K,
        "sell_strike": sell_K,
        "net_debit": round(debit, 2),
        "max_gain": round(max(max_gain, 0), 2),
        "max_loss": round(max(max_loss, 0), 2),
        "dte": dte,
        "iv": iv,
    }


# --------------------------------------------------
#  Strategy scoring & synthesis
# --------------------------------------------------
def _score_strategy(strat: dict, regime: str, var_pct: float, portfolio_delta: float) -> float:
    """
    Score a strategy 0-100 based on cost-efficiency, protection and
    regime fit.
    """
    score = 50.0
    name = strat.get("strategy", "")

    # Cost efficiency — lower cost is better
    cost = abs(strat.get("net_cost", strat.get("premium", strat.get("total_cost", strat.get("net_debit", 0)))))
    spot_val = strat.get("spot", 1) * strat.get("qty", 1)
    cost_pct = cost / max(spot_val, 1) * 100
    if cost_pct < 1:
        score += 15
    elif cost_pct < 2:
        score += 8
    elif cost_pct > 5:
        score -= 10

    # Regime fit
    reg_up = regime.upper() if regime else "SIDEWAYS"
    if reg_up == "BEAR":
        if name in ("PROTECTIVE_PUT", "BEAR_PUT_SPREAD", "COLLAR"):
            score += 20
        elif name == "STRADDLE":
            score += 10
    elif reg_up == "BULL":
        if name == "COLLAR":
            score += 15
        elif name == "IRON_CONDOR":
            score += 10
    else:  # SIDEWAYS
        if name == "IRON_CONDOR":
            score += 15
        elif name == "STRADDLE":
            score += 12

    # High VaR → protection strategies
    if var_pct > 3:
        if name in ("PROTECTIVE_PUT", "COLLAR"):
            score += 15
    if var_pct > 5:
        score += 5  # any hedge is good

    # Portfolio delta consideration
    if portfolio_delta > 5 and name in ("COLLAR", "PROTECTIVE_PUT", "BEAR_PUT_SPREAD"):
        score += 10

    return max(0, min(100, score))


def synthesize_strategy(
    active_trades: list,
    capital: float,
    regime: str = "SIDEWAYS",
    var_pct: float = 2.0,
    portfolio_delta: float = 0.0,
    iv_map: dict = None,
) -> dict:
    """
    Score all strategy types for each open position and return the
    best overall recommendation.
    """
    if not active_trades:
        return {"recommendation": "NO_POSITIONS", "strategies": []}

    all_strats = []
    for trade in active_trades:
        if trade.get("status") != "OPEN":
            continue
        sym = trade.get("stock", "")
        spot = float(trade.get("price", 0))
        qty = int(trade.get("qty", 0))
        iv = (iv_map or {}).get(sym, _DEFAULT_IV)
        if spot <= 0 or qty <= 0:
            continue

        candidates = [
            generate_collar(sym, spot, qty, iv),
            generate_iron_condor(sym, spot, qty, iv),
            generate_protective_put(sym, spot, qty, iv),
            generate_straddle(sym, spot, qty, iv),
            generate_bear_put_spread(sym, spot, qty, iv),
        ]
        for c in candidates:
            c["score"] = round(_score_strategy(c, regime, var_pct, portfolio_delta), 1)
        all_strats.extend(candidates)

    if not all_strats:
        return {"recommendation": "NO_CANDIDATES", "strategies": []}

    all_strats.sort(key=lambda x: x.get("score", 0), reverse=True)
    best = all_strats[0]

    total_cost = sum(
        abs(s.get("net_cost", s.get("premium", s.get("total_cost", s.get("net_debit", 0)))))
        for s in all_strats[:3]
    )

    return {
        "recommendation": best["strategy"],
        "top_strategy": best,
        "all_strategies": all_strats[:10],
        "total_hedge_cost": round(total_cost, 2),
        "cost_pct_of_capital": round(total_cost / max(capital, 1) * 100, 2),
        "regime": regime,
        "var_pct": var_pct,
        "n_positions_hedged": len([t for t in active_trades if t.get("status") == "OPEN"]),
        "timestamp": datetime.now(IST).isoformat(),
    }


def get_protection_cost(active_trades: list, iv_map: dict = None) -> dict:
    """Estimate total premium to fully hedge the portfolio with puts."""
    total = 0.0
    per_stock = {}
    for trade in active_trades:
        if trade.get("status") != "OPEN":
            continue
        sym = trade.get("stock", "")
        spot = float(trade.get("price", 0))
        qty = int(trade.get("qty", 0))
        iv = (iv_map or {}).get(sym, _DEFAULT_IV)
        pp = generate_protective_put(sym, spot, qty, iv)
        cost = pp["premium"]
        total += cost
        per_stock[sym] = round(cost, 2)
    return {"total_cost": round(total, 2), "per_stock": per_stock}


# --------------------------------------------------
#  Status / persistence
# --------------------------------------------------
def get_synth_status() -> dict:
    """Dashboard summary."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"status": "NO_DATA"}


def save_synth_state(data: dict = None):
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = data if data is not None else get_synth_status()
    with open(STATE_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_synth_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
