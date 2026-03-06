"""
====================================================
PROJECT AEGIS — VaR & Stress Testing Engine
====================================================
Portfolio risk analytics:
  1. Parametric VaR (variance-covariance)
  2. Historical VaR (sorted P&L)
  3. Monte Carlo VaR (10,000 simulations)
  4. Scenario / Stress Tests (2008, COVID, Rate-hike, Flash-crash)

Outputs
-------
data/var_stress.json
====================================================
"""

import os, json, time, math, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
VAR_FILE = os.path.join(DATA_DIR, "var_stress.json")
_cache = {"result": None, "ts": 0}
CACHE_TTL = 1800


# ══════════════════════════════════════════════════
#  SCENARIOS — Historical stress events
# ══════════════════════════════════════════════════
STRESS_SCENARIOS = {
    "2008_GFC": {
        "name": "2008 Global Financial Crisis",
        "equity_shock": -0.55,
        "vol_multiplier": 3.0,
        "correlation_spike": 0.90,
        "description": "Lehman-era shock: ~55% equity drawdown, correlations spike to 0.9",
    },
    "COVID_2020": {
        "name": "COVID-19 Crash (Mar 2020)",
        "equity_shock": -0.38,
        "vol_multiplier": 4.0,
        "correlation_spike": 0.85,
        "description": "Pandemic panic: ~38% drawdown in 5 weeks, extreme vol spike",
    },
    "RATE_HIKE": {
        "name": "Aggressive Rate Hike Cycle",
        "equity_shock": -0.15,
        "vol_multiplier": 1.8,
        "correlation_spike": 0.60,
        "description": "Central bank tightening: 15% correction, moderate vol rise",
    },
    "FLASH_CRASH": {
        "name": "Flash Crash / Circuit Breaker",
        "equity_shock": -0.20,
        "vol_multiplier": 5.0,
        "correlation_spike": 0.95,
        "description": "Intraday crash: 20% drop, massive vol, all correlations spike",
    },
    "SECTOR_ROTATION": {
        "name": "Sector Rotation (IT to Banks)",
        "equity_shock": -0.08,
        "vol_multiplier": 1.3,
        "correlation_spike": 0.40,
        "description": "Sector-specific drawdown: 8%, moderate disruption",
    },
}


# ══════════════════════════════════════════════════
#  HELPER — Fetch portfolio return series
# ══════════════════════════════════════════════════
def _fetch_returns(symbols: list[str], lookback: int = 252) -> pd.DataFrame:
    """Fetch daily returns for a list of symbols."""
    try:
        import yfinance as yf
        data = yf.download(symbols, period=f"{lookback}d", interval="1d",
                           progress=False, auto_adjust=True)
        if data is None or data.empty:
            return pd.DataFrame()
        close = data["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(symbols[0])
        returns = np.log(close / close.shift(1)).dropna()
        return returns
    except Exception:
        return pd.DataFrame()


def _portfolio_returns(returns_df: pd.DataFrame, weights: dict) -> np.ndarray:
    """Compute weighted portfolio returns."""
    cols = [c for c in returns_df.columns if c in weights]
    if not cols:
        # Equal weight fallback
        return returns_df.mean(axis=1).values
    w = np.array([weights.get(c, 1.0 / len(cols)) for c in cols])
    w = w / w.sum()
    return (returns_df[cols].values @ w)


# ══════════════════════════════════════════════════
#  VaR CALCULATIONS
# ══════════════════════════════════════════════════
def parametric_var(returns: np.ndarray, confidence: float = 0.95, capital: float = 1000) -> dict:
    """Variance-covariance (parametric) VaR."""
    mu = returns.mean()
    sigma = returns.std()
    # Normal quantile
    from math import erfc, sqrt
    # Approximation of inverse normal CDF
    z = _norm_ppf(1 - confidence)
    var_pct = -(mu + z * sigma)
    var_rupees = var_pct * capital
    return {
        "method": "Parametric",
        "confidence": confidence,
        "var_pct": round(float(var_pct) * 100, 4),
        "var_rupees": round(float(var_rupees), 2),
        "mean_daily_return": round(float(mu) * 100, 4),
        "daily_volatility": round(float(sigma) * 100, 4),
    }


def historical_var(returns: np.ndarray, confidence: float = 0.95, capital: float = 1000) -> dict:
    """Historical simulation VaR."""
    sorted_r = np.sort(returns)
    idx = int((1 - confidence) * len(sorted_r))
    var_pct = -float(sorted_r[max(idx, 0)])
    var_rupees = var_pct * capital
    # CVaR (Expected Shortfall)
    tail = sorted_r[:max(idx, 1)]
    cvar_pct = -float(tail.mean())
    return {
        "method": "Historical",
        "confidence": confidence,
        "var_pct": round(var_pct * 100, 4),
        "var_rupees": round(var_rupees, 2),
        "cvar_pct": round(cvar_pct * 100, 4),
        "cvar_rupees": round(cvar_pct * capital, 2),
        "worst_day": round(float(sorted_r[0]) * 100, 4),
        "best_day": round(float(sorted_r[-1]) * 100, 4),
    }


def monte_carlo_var(returns: np.ndarray, confidence: float = 0.95,
                    capital: float = 1000, n_sims: int = 10000,
                    horizon: int = 1) -> dict:
    """Monte Carlo simulated VaR."""
    mu = returns.mean()
    sigma = returns.std()
    np.random.seed(42)
    simulated = np.random.normal(mu * horizon, sigma * math.sqrt(horizon), n_sims)
    sorted_sim = np.sort(simulated)
    idx = int((1 - confidence) * n_sims)
    var_pct = -float(sorted_sim[max(idx, 0)])
    var_rupees = var_pct * capital
    # CVaR
    tail = sorted_sim[:max(idx, 1)]
    cvar_pct = -float(tail.mean())
    return {
        "method": "Monte Carlo",
        "confidence": confidence,
        "simulations": n_sims,
        "horizon_days": horizon,
        "var_pct": round(var_pct * 100, 4),
        "var_rupees": round(var_rupees, 2),
        "cvar_pct": round(cvar_pct * 100, 4),
        "cvar_rupees": round(cvar_pct * capital, 2),
        "sim_mean": round(float(simulated.mean()) * 100, 4),
        "sim_std": round(float(simulated.std()) * 100, 4),
    }


# ══════════════════════════════════════════════════
#  STRESS TESTS
# ══════════════════════════════════════════════════
def run_stress_tests(capital: float, current_positions: list | None = None) -> list[dict]:
    """Apply each scenario to portfolio and estimate impact."""
    results = []
    portfolio_value = capital
    if current_positions:
        for pos in current_positions:
            if pos.get("status") == "OPEN":
                portfolio_value += pos.get("qty", 0) * pos.get("price", 0)

    for key, scenario in STRESS_SCENARIOS.items():
        shock = scenario["equity_shock"]
        loss = portfolio_value * abs(shock)
        remaining = portfolio_value + (portfolio_value * shock)
        results.append({
            "scenario": key,
            "name": scenario["name"],
            "description": scenario["description"],
            "equity_shock_pct": round(shock * 100, 1),
            "estimated_loss": round(loss, 2),
            "portfolio_after": round(remaining, 2),
            "vol_multiplier": scenario["vol_multiplier"],
            "correlation_spike": scenario["correlation_spike"],
            "severity": "EXTREME" if abs(shock) > 0.30 else ("HIGH" if abs(shock) > 0.15 else "MODERATE"),
        })
    return results


# ══════════════════════════════════════════════════
#  DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════
def analyse_drawdowns(returns: np.ndarray) -> dict:
    """Compute max drawdown, drawdown duration stats."""
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max

    max_dd = float(np.min(drawdowns))
    max_dd_idx = int(np.argmin(drawdowns))

    # Current drawdown
    current_dd = float(drawdowns[-1]) if len(drawdowns) > 0 else 0.0

    # Drawdown periods
    in_dd = drawdowns < 0
    dd_periods = 0
    longest_dd = 0
    current_length = 0
    for v in in_dd:
        if v:
            current_length += 1
            longest_dd = max(longest_dd, current_length)
        else:
            if current_length > 0:
                dd_periods += 1
            current_length = 0

    return {
        "max_drawdown_pct": round(max_dd * 100, 4),
        "max_drawdown_day": max_dd_idx,
        "current_drawdown_pct": round(current_dd * 100, 4),
        "drawdown_periods": dd_periods,
        "longest_drawdown_days": longest_dd,
        "avg_drawdown_pct": round(float(drawdowns[drawdowns < 0].mean()) * 100, 4) if (drawdowns < 0).any() else 0.0,
    }


# ══════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════
def analyse_var_stress(symbols: list[str] | None = None,
                       weights: dict | None = None,
                       capital: float = 1000,
                       positions: list | None = None,
                       force: bool = False) -> dict:
    """
    Full VaR + stress test analysis.
    Returns comprehensive risk report.
    """
    global _cache
    if not force and _cache["result"] and (time.time() - _cache["ts"] < CACHE_TTL):
        return _cache["result"]

    if symbols is None:
        symbols = [
            "TATASTEEL.NS", "SBIN.NS", "RELIANCE.NS", "HDFCBANK.NS",
            "ICICIBANK.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
            "INFY.NS", "TCS.NS",
        ]

    if weights is None:
        weights = {s: 1.0 / len(symbols) for s in symbols}

    returns_df = _fetch_returns(symbols, lookback=252)
    if returns_df.empty or len(returns_df) < 30:
        result = {
            "error": "insufficient data",
            "parametric_var": {},
            "historical_var": {},
            "monte_carlo_var": {},
            "stress_tests": run_stress_tests(capital, positions),
            "drawdowns": {},
            "timestamp": _now_ist(),
        }
        _cache = {"result": result, "ts": time.time()}
        return result

    port_ret = _portfolio_returns(returns_df, weights)

    p_var = parametric_var(port_ret, 0.95, capital)
    h_var = historical_var(port_ret, 0.95, capital)
    mc_var = monte_carlo_var(port_ret, 0.95, capital)

    # Also compute 99% VaR
    p_var_99 = parametric_var(port_ret, 0.99, capital)
    h_var_99 = historical_var(port_ret, 0.99, capital)
    mc_var_99 = monte_carlo_var(port_ret, 0.99, capital)

    stress = run_stress_tests(capital, positions)
    dd = analyse_drawdowns(port_ret)

    # Overall risk score 0-100
    max_var = max(p_var["var_pct"], h_var["var_pct"], mc_var["var_pct"])
    max_dd_abs = abs(dd.get("max_drawdown_pct", 0))
    risk_score = min(100, int(max_var * 10 + max_dd_abs * 5))

    result = {
        "parametric_var_95": p_var,
        "historical_var_95": h_var,
        "monte_carlo_var_95": mc_var,
        "parametric_var_99": p_var_99,
        "historical_var_99": h_var_99,
        "monte_carlo_var_99": mc_var_99,
        "stress_tests": stress,
        "drawdowns": dd,
        "risk_score": risk_score,
        "risk_level": "LOW" if risk_score < 30 else ("MODERATE" if risk_score < 60 else "HIGH"),
        "capital": capital,
        "symbols": symbols,
        "observations": len(port_ret),
        "timestamp": _now_ist(),
    }

    _cache = {"result": result, "ts": time.time()}
    return result


def save_var_state(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(VAR_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ══════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════
def _norm_ppf(p: float) -> float:
    """Rational approximation of inverse normal CDF (Abramowitz & Stegun)."""
    if p <= 0:
        return -10
    if p >= 1:
        return 10
    if p > 0.5:
        return -_norm_ppf(1 - p)
    t = math.sqrt(-2 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))


def _now_ist() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
