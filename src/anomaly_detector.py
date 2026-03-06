"""
====================================================
🔍 PROJECT AEGIS — Anomaly Detector (Phase 11)
====================================================
Isolation Forest-based anomaly detection on
price / volume / sentiment patterns.

Flags unusual market conditions BEFORE they trigger
false buy signals in the main pipeline.

Features used (per stock):
  1. Price returns (1d, 5d, 20d)
  2. Volume z-score
  3. RSI deviation from 50
  4. ATR percentile rank
  5. Price-volume divergence
  6. Spread (high-low) percentile
  7. Gap (open vs prev close)
  8. Sentiment score delta (if available)

Anomaly score: -1 (anomaly) to +1 (normal).
Gate blocks when score < threshold.
====================================================
"""

import os, sys, json, math, warnings, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from config import STOCK_WATCHLIST, TOP_N_STOCKS

# ──────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
FILE_ANOMALY = os.path.join(DATA, "anomaly_state.json")
FILE_MODEL   = os.path.join(MODELS_DIR, "isolation_forest.pkl")

N_ESTIMATORS     = 100
CONTAMINATION    = 0.05   # Expected anomaly fraction
ANOMALY_THRESH   = -0.3   # Block if score < this
LOOKBACK         = 120    # Days of training data
N_FEATURES       = 8

# Try sklearn
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Cache
_anomaly_cache: Dict = {}
_models: Dict = {}  # Per-stock models


# ──────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ──────────────────────────────────────────────────
def _extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract anomaly-detection features from OHLCV dataframe.
    Returns (n_samples, N_FEATURES) array.
    """
    if df is None or len(df) < 25:
        return np.array([])

    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    o = df["Open"].values.astype(float)

    n = len(c)
    features = []

    for i in range(20, n):
        # 1. Returns: 1d, 5d, 20d
        ret_1d  = (c[i] - c[i - 1]) / max(1e-6, c[i - 1])
        ret_5d  = (c[i] - c[i - 5]) / max(1e-6, c[i - 5])
        ret_20d = (c[i] - c[i - 20]) / max(1e-6, c[i - 20])

        # 2. Volume z-score (20d)
        vol_20 = v[i - 20:i]
        vol_mean = np.mean(vol_20)
        vol_std = np.std(vol_20)
        vol_z = (v[i] - vol_mean) / max(1e-6, vol_std)

        # 3. RSI deviation from 50
        gains = np.maximum(0, np.diff(c[i - 14:i + 1]))
        losses = np.abs(np.minimum(0, np.diff(c[i - 14:i + 1])))
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-6
        rs = avg_gain / max(1e-6, avg_loss)
        rsi = 100 - 100 / (1 + rs)
        rsi_dev = (rsi - 50) / 50  # Normalized -1 to +1

        # 4. ATR percentile rank (14d)
        tr = np.maximum(
            h[i - 14:i] - l[i - 14:i],
            np.maximum(
                np.abs(h[i - 14:i] - c[i - 15:i - 1]),
                np.abs(l[i - 14:i] - c[i - 15:i - 1])
            )
        )
        atr = np.mean(tr) if len(tr) > 0 else 0
        atr_pct = atr / max(1e-6, c[i])  # As % of price

        # 5. Price-volume divergence
        # Price up + volume down = divergence > 0
        pv_div = ret_1d * (-vol_z)

        # 6. Daily range percentile (high-low spread)
        spread = (h[i] - l[i]) / max(1e-6, c[i])

        # 7. Gap (open vs prev close)
        gap = (o[i] - c[i - 1]) / max(1e-6, c[i - 1])

        features.append([
            ret_1d, ret_5d, ret_20d,
            vol_z, rsi_dev, atr_pct,
            pv_div, spread + gap,  # Combine spread & gap
        ])

    return np.array(features)


# ──────────────────────────────────────────────────
#  TRAIN ISOLATION FOREST
# ──────────────────────────────────────────────────
def train_isolation_forest(
    stock_dfs: Dict[str, pd.DataFrame] = None,
    symbols: List[str] = None,
) -> Dict:
    """
    Train one global Isolation Forest on pooled features.
    Also trains per-stock models for finer granularity.
    """
    global _models

    if not HAS_SKLEARN:
        return {"status": "sklearn not available", "fallback": True}

    if symbols is None:
        symbols = STOCK_WATCHLIST[:TOP_N_STOCKS]

    all_features = []
    per_stock_features = {}

    for sym in symbols:
        df = stock_dfs.get(sym) if stock_dfs else None
        if df is None:
            continue
        feats = _extract_features(df)
        if len(feats) > 10:
            all_features.append(feats)
            per_stock_features[sym] = feats

    if not all_features:
        return {"status": "No data for training", "models_trained": 0}

    # Pool all features → Global model
    pooled = np.vstack(all_features)
    scaler = StandardScaler()
    pooled_scaled = scaler.fit_transform(pooled)

    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(pooled_scaled)

    _models["__global__"] = {"model": clf, "scaler": scaler}

    # Per-stock models
    for sym, feats in per_stock_features.items():
        if len(feats) > 30:
            sc = StandardScaler()
            scaled = sc.fit_transform(feats)
            m = IsolationForest(
                n_estimators=N_ESTIMATORS // 2,
                contamination=CONTAMINATION,
                random_state=42,
            )
            m.fit(scaled)
            _models[sym] = {"model": m, "scaler": sc}

    # Save models
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(FILE_MODEL, "wb") as f:
            pickle.dump(_models, f)
    except Exception:
        pass

    return {
        "status": "trained",
        "n_samples": len(pooled),
        "models_trained": 1 + len(per_stock_features),
        "symbols": list(per_stock_features.keys()),
    }


# ──────────────────────────────────────────────────
#  DETECT ANOMALIES (predict)
# ──────────────────────────────────────────────────
def detect_anomalies(
    stock_dfs: Dict[str, pd.DataFrame],
    symbols: List[str] = None,
) -> Dict:
    """
    Detect anomalies in current market data.
    Returns anomaly scores per stock.
    """
    global _anomaly_cache

    if symbols is None:
        symbols = list(stock_dfs.keys())

    results = {}

    for sym in symbols:
        df = stock_dfs.get(sym)
        if df is None:
            continue

        feats = _extract_features(df)
        if len(feats) == 0:
            results[sym] = {"score": 0.0, "is_anomaly": False, "method": "no_data"}
            continue

        latest = feats[-1:].reshape(1, -1)

        if HAS_SKLEARN and (sym in _models or "__global__" in _models):
            # Use per-stock model if available, else global
            model_data = _models.get(sym, _models.get("__global__"))
            if model_data:
                scaled = model_data["scaler"].transform(latest)
                score = float(model_data["model"].decision_function(scaled)[0])
                pred = int(model_data["model"].predict(scaled)[0])
                results[sym] = {
                    "score": round(score, 4),
                    "is_anomaly": pred == -1,
                    "method": "isolation_forest",
                    "features": {
                        "ret_1d": round(float(feats[-1, 0]) * 100, 2),
                        "ret_5d": round(float(feats[-1, 1]) * 100, 2),
                        "vol_z": round(float(feats[-1, 3]), 2),
                        "rsi_dev": round(float(feats[-1, 4]) * 50 + 50, 1),
                    },
                }
                continue

        # Fallback: simple statistical anomaly detection
        score = _fallback_anomaly(feats[-1])
        results[sym] = {
            "score": round(score, 4),
            "is_anomaly": score < ANOMALY_THRESH,
            "method": "statistical_fallback",
        }

    _anomaly_cache = {
        "stocks": results,
        "n_anomalies": sum(1 for r in results.values() if r.get("is_anomaly")),
        "n_total": len(results),
        "timestamp": datetime.now(IST).isoformat(),
    }
    return _anomaly_cache


def _fallback_anomaly(features: np.ndarray) -> float:
    """
    Simple z-score-based anomaly check (no sklearn).
    Returns score: negative = more anomalous.
    """
    # Each feature's absolute value; extreme = anomalous
    abs_vals = np.abs(features)
    max_z = float(np.max(abs_vals))
    mean_z = float(np.mean(abs_vals))

    # Score: 0 = normal, more negative = more anomalous
    if max_z > 3.0:
        return -0.8
    elif max_z > 2.5:
        return -0.4
    elif mean_z > 1.5:
        return -0.2
    else:
        return 0.1 + (1.0 - mean_z) * 0.3


# ──────────────────────────────────────────────────
#  ANOMALY GATE (for Sniper buy-loop)
# ──────────────────────────────────────────────────
def check_anomaly_gate(
    symbol: str,
    anomaly_data: Dict = None,
    threshold: float = ANOMALY_THRESH,
) -> Tuple[bool, str, Dict]:
    """
    Gate: block BUY if anomaly detected.
    Returns (ok, reason, data).
    """
    if anomaly_data is None:
        anomaly_data = _anomaly_cache

    stocks = anomaly_data.get("stocks", {})
    info = stocks.get(symbol, {})

    if not info:
        return True, "No anomaly data — allowing", {}

    score = info.get("score", 0.0)
    is_anom = info.get("is_anomaly", False)

    if is_anom or score < threshold:
        return False, f"ANOMALY detected (score={score:.3f}, method={info.get('method', '?')})", info

    return True, f"Normal (score={score:.3f})", info


# ──────────────────────────────────────────────────
#  LOAD MODELS (on startup)
# ──────────────────────────────────────────────────
def load_anomaly_models():
    """Load saved Isolation Forest models."""
    global _models
    if HAS_SKLEARN and os.path.exists(FILE_MODEL):
        try:
            with open(FILE_MODEL, "rb") as f:
                _models = pickle.load(f)
            return True
        except Exception:
            pass
    return False


# ──────────────────────────────────────────────────
#  SAVE / LOAD STATE
# ──────────────────────────────────────────────────
def save_anomaly_state(data: dict = None):
    os.makedirs(DATA, exist_ok=True)
    payload = data or _anomaly_cache
    with open(FILE_ANOMALY, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_anomaly_state() -> dict:
    global _anomaly_cache
    if os.path.exists(FILE_ANOMALY):
        try:
            with open(FILE_ANOMALY, "r") as f:
                _anomaly_cache = json.load(f)
            return _anomaly_cache
        except Exception:
            pass
    return {}
