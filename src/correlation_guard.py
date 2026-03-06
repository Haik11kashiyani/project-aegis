"""
====================================================
PROJECT AEGIS — Correlation Guard v1.0
====================================================
Monitors inter-stock correlations to prevent buying
highly correlated positions (diversification guard).

If RELIANCE and TATASTEEL are 0.85 correlated,
buying both is like doubling down on the same bet.

Key features:
  - Rolling correlation matrix (30-day)
  - High-correlation warnings before buying
  - Portfolio concentration risk score
  - Suggested diversification adjustments

Output: data/correlation_matrix.json
====================================================
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "correlation_matrix.json")

# High correlation threshold
HIGH_CORR_THRESHOLD = 0.75  # Above this, 2 stocks are "too similar"
CRITICAL_CORR_THRESHOLD = 0.90


def _download_closes(symbols: list, period: str = "60d") -> pd.DataFrame:
    """Download close prices for multiple symbols."""
    closes = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period=period, interval="1d", progress=False)
            if hasattr(df.columns, 'droplevel') and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df is not None and not df.empty and "Close" in df.columns:
                closes[sym] = df["Close"].values.flatten()
        except Exception:
            continue

    if not closes:
        return pd.DataFrame()

    # Align to same length
    min_len = min(len(v) for v in closes.values())
    aligned = {k: v[-min_len:] for k, v in closes.items()}
    return pd.DataFrame(aligned)


def compute_correlation_matrix(symbols: list, period: str = "60d") -> dict:
    """
    Compute rolling correlation matrix for the given symbols.
    
    Returns:
    {
        "matrix": {sym1: {sym2: corr_value, ...}, ...},
        "high_pairs": [("SYM1", "SYM2", 0.89), ...],
        "avg_correlation": 0.45,
    }
    """
    df = _download_closes(symbols, period)
    if df.empty or len(df) < 20:
        return {"matrix": {}, "high_pairs": [], "avg_correlation": 0}

    # Compute returns (correlation of returns is more meaningful than prices)
    returns = df.pct_change().dropna()
    if returns.empty:
        return {"matrix": {}, "high_pairs": [], "avg_correlation": 0}

    corr = returns.corr()

    # Convert to serializable format
    matrix = {}
    for sym1 in corr.index:
        matrix[sym1] = {}
        for sym2 in corr.columns:
            matrix[sym1][sym2] = round(float(corr.loc[sym1, sym2]), 3)

    # Find high-correlation pairs
    high_pairs = []
    seen = set()
    for sym1 in corr.index:
        for sym2 in corr.columns:
            if sym1 >= sym2:
                continue
            pair_key = f"{sym1}-{sym2}"
            if pair_key in seen:
                continue
            seen.add(pair_key)
            c = float(corr.loc[sym1, sym2])
            if abs(c) >= HIGH_CORR_THRESHOLD:
                high_pairs.append((sym1, sym2, round(c, 3)))

    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Average pairwise correlation (excluding diagonal)
    vals = []
    for sym1 in corr.index:
        for sym2 in corr.columns:
            if sym1 != sym2:
                vals.append(abs(float(corr.loc[sym1, sym2])))
    avg_corr = np.mean(vals) if vals else 0

    return {
        "matrix": matrix,
        "high_pairs": [{"stock1": p[0], "stock2": p[1], "correlation": p[2]} for p in high_pairs],
        "avg_correlation": round(avg_corr, 3),
    }


def check_before_buy(candidate: str, current_positions: list, symbols: list) -> dict:
    """
    Before buying a stock, check if it's highly correlated with existing positions.
    
    Args:
        candidate: Stock symbol to buy
        current_positions: List of stock symbols currently held
        symbols: Full watchlist
    
    Returns:
        {
            "approved": True/False,
            "risk_level": "LOW" / "MEDIUM" / "HIGH",
            "conflicts": [{"stock": "SBIN.NS", "correlation": 0.87}],
            "recommendation": "text"
        }
    """
    if not current_positions:
        return {
            "approved": True, "risk_level": "LOW",
            "conflicts": [], "recommendation": "No existing positions — safe to buy"
        }

    # Download and compute correlation between candidate and held stocks
    check_list = [candidate] + current_positions
    df = _download_closes(check_list, "30d")
    if df.empty or candidate not in df.columns:
        return {
            "approved": True, "risk_level": "UNKNOWN",
            "conflicts": [], "recommendation": "Insufficient data — allowing trade"
        }

    returns = df.pct_change().dropna()
    if returns.empty:
        return {"approved": True, "risk_level": "UNKNOWN", "conflicts": [], "recommendation": "No return data"}

    conflicts = []
    for held in current_positions:
        if held in returns.columns:
            c = float(returns[candidate].corr(returns[held]))
            if abs(c) >= HIGH_CORR_THRESHOLD:
                conflicts.append({
                    "stock": held,
                    "correlation": round(c, 3),
                    "severity": "CRITICAL" if abs(c) >= CRITICAL_CORR_THRESHOLD else "HIGH",
                })

    if not conflicts:
        return {
            "approved": True, "risk_level": "LOW",
            "conflicts": [], "recommendation": "Low correlation with current positions"
        }

    max_corr = max(c["correlation"] for c in conflicts)
    has_critical = any(c["severity"] == "CRITICAL" for c in conflicts)

    if has_critical:
        return {
            "approved": False, "risk_level": "HIGH",
            "conflicts": conflicts,
            "recommendation": f"BLOCKED: {candidate} is {max_corr:.0%} correlated with held position. Too risky."
        }
    else:
        return {
            "approved": True, "risk_level": "MEDIUM",
            "conflicts": conflicts,
            "recommendation": f"WARNING: {candidate} has {len(conflicts)} high-corr positions (max {max_corr:.0%}). Reduce size."
        }


def portfolio_concentration_score(symbols: list) -> dict:
    """
    Compute how concentrated/diversified a portfolio is.
    Score: 0 = perfectly diversified, 100 = all stocks move together.
    """
    data = compute_correlation_matrix(symbols)
    avg = data.get("avg_correlation", 0)
    n_high = len(data.get("high_pairs", []))

    score = min(100, avg * 100 + n_high * 5)

    if score < 30:
        rating = "WELL_DIVERSIFIED"
    elif score < 55:
        rating = "MODERATE"
    elif score < 75:
        rating = "CONCENTRATED"
    else:
        rating = "HIGHLY_CONCENTRATED"

    return {
        "concentration_score": round(score, 1),
        "rating": rating,
        "avg_pairwise_correlation": avg,
        "high_correlation_pairs": n_high,
    }


def analyse_correlations(symbols: list = None) -> dict:
    """Full correlation analysis. Saves to JSON."""
    if symbols is None:
        from config import STOCK_WATCHLIST
        symbols = STOCK_WATCHLIST

    print(f"[CORR] Computing correlation matrix for {len(symbols)} stocks...")
    matrix_data = compute_correlation_matrix(symbols)
    concentration = portfolio_concentration_score(symbols)

    result = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": symbols,
        "correlation": matrix_data,
        "concentration": concentration,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[CORR] Correlation Analysis:")
    print(f"  Avg Pairwise : {matrix_data['avg_correlation']:.3f}")
    print(f"  High Pairs   : {len(matrix_data['high_pairs'])}")
    print(f"  Concentration: {concentration['concentration_score']:.1f}/100 ({concentration['rating']})")
    for p in matrix_data["high_pairs"][:5]:
        s1 = p["stock1"].replace(".NS", "")
        s2 = p["stock2"].replace(".NS", "")
        print(f"  ⚠️  {s1} ↔ {s2}: {p['correlation']:.3f}")

    return result


if __name__ == "__main__":
    analyse_correlations()
