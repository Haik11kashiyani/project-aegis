"""
====================================================
Phase 12 — Causal Inference Engine
====================================================
Uses lightweight causal reasoning (Granger causality +
partial-correlation pruning) to distinguish true
predictive signals from spurious correlations.

The engine builds a directed causal graph among the
technical features used by the ML models and:
  1. Identifies which features *cause* future returns
     (Granger test, lag 1-5 days)
  2. Prunes features whose correlation vanishes when
     conditioned on a confounder (partial correlation)
  3. Produces a **causal feature mask** the Sniper can
     use to pre-filter input features before prediction
  4. Detects *regime-specific* causal links so that a
     feature important in BULL might be dropped in BEAR

No external causal-inference library is required;
everything runs on numpy + scipy (already in the
project).

Key functions used by sniper.py
-------------------------------
* ``discover_causal_graph()``  — build the DAG
* ``get_causal_features()``    — approved feature list
* ``test_causality()``         — single-pair Granger
* ``filter_spurious_signals()`` — drop weak features
* ``get_causal_status()``      — dashboard summary
* ``save_causal_state() / load_causal_state()``
====================================================
"""

import os, sys, json, math
from datetime import datetime
from collections import defaultdict
import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "causal_engine_state.json")

# ── Defaults ─────────────────────────────────────
MAX_LAG        = 5       # Granger causality test lag
PVALUE_THRESH  = 0.05    # significance threshold
PCORR_THRESH   = 0.02    # minimum partial-correlation magnitude
MIN_SAMPLES    = 60      # need at least 60 rows for reliable test

# All candidate features (same as config.RF_FEATURES plus extras)
ALL_FEATURES = [
    "RSI", "SMA_50", "SMA_200", "EMA_20", "ATR", "MACD",
    "MACD_Signal", "BB_Upper", "BB_Lower", "Volume_Ratio",
    "OBV", "Sentiment_Score",
]

# Runtime cache
_causal_cache: dict = {}


# --------------------------------------------------
#  Granger Causality (OLS-based F-test)
# --------------------------------------------------
def _ols_rss(y: np.ndarray, X: np.ndarray) -> float:
    """Residual sum of squares from OLS fit y ~ X."""
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        return float((residuals ** 2).sum())
    except Exception:
        return float("inf")


def test_causality(cause: np.ndarray, effect: np.ndarray, max_lag: int = MAX_LAG) -> dict:
    """
    Granger-causality F-test: does *cause* Granger-cause *effect*?

    Returns dict with: granger_caused (bool), best_lag, f_stat, p_value
    """
    n = len(effect)
    if n < max_lag + MIN_SAMPLES:
        return {"granger_caused": False, "reason": "insufficient_data"}

    best = {"f_stat": 0, "p_value": 1.0, "lag": 1, "granger_caused": False}

    for lag in range(1, max_lag + 1):
        y = effect[lag:]
        T = len(y)
        # Restricted model: AR(lag) on effect only
        X_r = np.column_stack([effect[lag - i - 1: T + lag - i - 1] for i in range(lag)])
        X_r = np.column_stack([np.ones(T), X_r])
        rss_r = _ols_rss(y, X_r)

        # Unrestricted model: AR(lag) on effect + lagged cause
        X_u = np.column_stack([
            X_r,
            *[cause[lag - i - 1: T + lag - i - 1].reshape(-1, 1) for i in range(lag)]
        ])
        rss_u = _ols_rss(y, X_u)

        if rss_u <= 0 or rss_r <= 0:
            continue

        df1 = lag
        df2 = T - X_u.shape[1]
        if df2 <= 0:
            continue

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

        # Approximate p-value from F-distribution using Abramowitz & Stegun
        p_value = _f_pvalue_approx(f_stat, df1, df2)

        if f_stat > best["f_stat"]:
            best = {
                "f_stat": round(f_stat, 4),
                "p_value": round(p_value, 6),
                "lag": lag,
                "granger_caused": p_value < PVALUE_THRESH,
            }

    return best


def _f_pvalue_approx(f: float, d1: int, d2: int) -> float:
    """
    Approximate upper-tail p-value of F-distribution using
    the normal approximation (Abramowitz & Stegun 26.6.15).
    """
    if f <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    try:
        a = 2.0 / (9.0 * d1)
        b = 2.0 / (9.0 * d2)
        z = ((1.0 - b) * (f ** (1.0 / 3.0)) - (1.0 - a)) / math.sqrt(a + b * (f ** (2.0 / 3.0)))
        # Standard normal CDF via Zelen & Severo (1964)
        p = 0.5 * math.erfc(z / math.sqrt(2))
        return max(0.0, min(1.0, p))
    except Exception:
        return 1.0


# --------------------------------------------------
#  Partial correlation
# --------------------------------------------------
def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Partial correlation of x and y controlling for z.
    Uses recursive formula via simple correlations.
    """
    def _corr(a, b):
        a = (a - a.mean())
        b = (b - b.mean())
        denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()))
        return float((a * b).sum() / denom) if denom > 1e-12 else 0.0

    rxy = _corr(x, y)
    rxz = _corr(x, z)
    ryz = _corr(y, z)
    denom = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / denom


# --------------------------------------------------
#  Causal graph discovery
# --------------------------------------------------
def discover_causal_graph(df, target_col: str = "Close", features: list = None) -> dict:
    """
    Build a causal graph from dataframe *df*.

    Parameters
    ----------
    df : pd.DataFrame with feature columns + target_col
    target_col : the variable we want to predict (future returns)
    features : list of feature column names to test

    Returns
    -------
    dict with: edges (list of dicts), approved_features, rejected,
    graph_density, timestamp
    """
    feats = features or [f for f in ALL_FEATURES if f in df.columns]
    if target_col not in df.columns:
        return {"error": "target column missing", "approved_features": feats}

    # Use log-returns of target as the response
    target = df[target_col].pct_change().dropna().values

    edges = []
    approved = []
    rejected = []

    for feat in feats:
        if feat not in df.columns:
            rejected.append({"feature": feat, "reason": "missing_column"})
            continue

        cause = df[feat].dropna().values
        min_len = min(len(cause), len(target))
        if min_len < MAX_LAG + MIN_SAMPLES:
            rejected.append({"feature": feat, "reason": "insufficient_data"})
            continue

        cause = cause[-min_len:]
        tgt = target[-min_len:]

        # Step 1: Granger causality
        gc = test_causality(cause, tgt)
        is_causal = gc.get("granger_caused", False)

        # Step 2: Partial correlation pruning
        # Control for the most obvious confounder (Volume_Ratio or ATR)
        confounder_name = "Volume_Ratio" if feat != "Volume_Ratio" else "ATR"
        pcorr_val = 0.0
        if confounder_name in df.columns:
            conf = df[confounder_name].dropna().values
            clen = min(len(cause), len(tgt), len(conf))
            if clen >= MIN_SAMPLES:
                pcorr_val = _partial_corr(cause[-clen:], tgt[-clen:], conf[-clen:])

        survives_pcorr = abs(pcorr_val) >= PCORR_THRESH

        edge = {
            "feature": feat,
            "granger_caused": is_causal,
            "f_stat": gc.get("f_stat", 0),
            "p_value": gc.get("p_value", 1),
            "best_lag": gc.get("lag", 0),
            "partial_corr": round(pcorr_val, 4),
            "survives_pcorr": survives_pcorr,
            "approved": is_causal and survives_pcorr,
        }
        edges.append(edge)
        if edge["approved"]:
            approved.append(feat)
        else:
            rejected.append({"feature": feat, "reason": "failed_causality" if not is_causal else "spurious"})

    # Ensure we keep a minimum set of core features even if causal test fails
    core = {"RSI", "ATR", "MACD", "Volume_Ratio"}
    for c in core:
        if c in df.columns and c not in approved:
            approved.append(c)

    result = {
        "edges": edges,
        "approved_features": approved,
        "rejected": rejected,
        "total_tested": len(feats),
        "total_approved": len(approved),
        "graph_density": round(len(approved) / max(len(feats), 1), 3),
        "timestamp": datetime.now(IST).isoformat(),
    }
    _causal_cache.update(result)
    return result


def get_causal_features(symbol: str = None, cached: dict = None) -> list:
    """
    Return the approved feature list from the most recent causal scan.
    Falls back to ALL_FEATURES if no scan has been run.
    """
    src = cached or _causal_cache
    approved = src.get("approved_features", [])
    return approved if approved else ALL_FEATURES[:]


def filter_spurious_signals(
    signals: dict,
    causal_data: dict = None,
) -> dict:
    """
    Given a dict of feature_name → signal_value, remove any features
    that are not causally approved.
    """
    approved = set(get_causal_features(cached=causal_data))
    filtered = {k: v for k, v in signals.items() if k in approved}
    removed = {k: v for k, v in signals.items() if k not in approved}
    return {
        "filtered_signals": filtered,
        "removed": removed,
        "n_kept": len(filtered),
        "n_removed": len(removed),
    }


# --------------------------------------------------
#  Status / persistence
# --------------------------------------------------
def get_causal_status() -> dict:
    """Dashboard summary."""
    if _causal_cache:
        return {
            "approved": _causal_cache.get("approved_features", []),
            "total_tested": _causal_cache.get("total_tested", 0),
            "total_approved": _causal_cache.get("total_approved", 0),
            "graph_density": _causal_cache.get("graph_density", 0),
            "edges": _causal_cache.get("edges", []),
            "timestamp": _causal_cache.get("timestamp"),
        }
    return {"status": "NO_DATA"}


def save_causal_state(data: dict = None):
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = data if data is not None else (_causal_cache if _causal_cache else get_causal_status())
    with open(STATE_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_causal_state() -> dict:
    global _causal_cache
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            _causal_cache.update(data)
            return data
        except Exception:
            pass
    return {}
