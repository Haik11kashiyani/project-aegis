"""
====================================================
🔬 PROJECT AEGIS - Model Drift Detector
====================================================
Monitors prediction distributions over time and flags
when models have drifted from their training baseline.

Drift Detection Methods:
  1. KL Divergence     — Information-theoretic distance
  2. PSI (Pop Stability Index) — Industry standard
  3. Confidence Decay  — Average confidence trending down
  4. Accuracy Decay    — Rolling accuracy on recent trades

When drift is detected:
  - Flag models for retraining
  - Reduce confidence weight of drifted models
  - Alert via Telegram/dashboard
  - Log drift events for analysis
====================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from config import (
    STOCK_WATCHLIST, TOP_N_STOCKS, TRADE_LOG_FILE,
    RF_FEATURES, CONFIDENCE_THRESHOLD,
)

# ──────────────────────────────────────────────────
#  DRIFT CONFIG
# ──────────────────────────────────────────────────
KL_DIVERGENCE_THRESHOLD  = 0.15    # KL div above this → drift
PSI_THRESHOLD            = 0.20    # PSI above this → significant drift
CONFIDENCE_DECAY_WINDOW  = 20      # Rolling window for confidence trend
ACCURACY_DECAY_WINDOW    = 30      # Rolling window for accuracy
ACCURACY_MIN_THRESHOLD   = 0.40    # Below this → severe drift
DRIFT_CHECK_MIN_SAMPLES  = 15      # Need this many predictions to check
BASELINE_BINS            = 10      # Histogram bins for distribution compare
RETRAIN_FLAG_COOLDOWN_H  = 24      # Don't re-flag within this period

DRIFT_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "drift_state.json",
)
DRIFT_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "drift_log.json",
)
PREDICTION_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prediction_log.json",
)


# ──────────────────────────────────────────────────
#  PREDICTION LOGGER (called by sniper during trades)
# ──────────────────────────────────────────────────
def log_prediction(
    symbol: str,
    model_name: str,
    confidence: float,
    prediction: int,        # 1=buy, 0=hold
    actual: int = None,     # 1=profit, 0=loss (set later)
):
    """
    Log a model prediction for drift analysis.
    Called by sniper.py after each model inference.
    """
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "model": model_name,
        "confidence": round(confidence, 4),
        "prediction": prediction,
        "actual": actual,
    }

    try:
        log = []
        if os.path.exists(PREDICTION_LOG_FILE):
            with open(PREDICTION_LOG_FILE, "r") as f:
                log = json.load(f)

        log.append(entry)

        # Keep last 2000 entries
        if len(log) > 2000:
            log = log[-2000:]

        os.makedirs(os.path.dirname(PREDICTION_LOG_FILE), exist_ok=True)
        with open(PREDICTION_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass


def update_prediction_actuals(symbol: str, profitable: bool):
    """
    Retroactively update actual outcomes for recent predictions.
    Called when a trade closes.
    """
    try:
        if not os.path.exists(PREDICTION_LOG_FILE):
            return

        with open(PREDICTION_LOG_FILE, "r") as f:
            log = json.load(f)

        actual_val = 1 if profitable else 0
        updated = 0
        for entry in reversed(log):
            if entry["symbol"] == symbol and entry["actual"] is None:
                entry["actual"] = actual_val
                updated += 1
                if updated >= 4:  # Update last 4 model predictions for this stock
                    break

        with open(PREDICTION_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass


# ──────────────────────────────────────────────────
#  STATISTICAL DRIFT MEASURES
# ──────────────────────────────────────────────────
def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence D_KL(p || q).
    Both p and q should be probability distributions (sum to 1).
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = BASELINE_BINS) -> float:
    """
    Population Stability Index.
    PSI < 0.1  → No drift
    PSI 0.1-0.2 → Moderate drift
    PSI > 0.2  → Significant drift
    """
    # Create histogram bins from expected
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions
    eps = 1e-10
    expected_pct = (expected_hist / expected_hist.sum()) + eps
    actual_pct = (actual_hist / actual_hist.sum()) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_confidence_trend(confidences: list, window: int = CONFIDENCE_DECAY_WINDOW) -> dict:
    """
    Detect confidence decay trend.
    Returns slope, mean, and decay flag.
    """
    if len(confidences) < window:
        return {"slope": 0, "mean": 0, "decaying": False}

    recent = np.array(confidences[-window:])
    x = np.arange(len(recent))

    # Linear regression slope
    slope = np.polyfit(x, recent, 1)[0]
    mean = float(recent.mean())

    return {
        "slope": round(float(slope), 6),
        "mean": round(mean, 4),
        "decaying": slope < -0.005,  # Declining by >0.5% per prediction
    }


def compute_rolling_accuracy(predictions: list, window: int = ACCURACY_DECAY_WINDOW) -> dict:
    """
    Compute rolling accuracy from predictions with known actuals.
    """
    with_actuals = [p for p in predictions if p.get("actual") is not None]

    if len(with_actuals) < 5:
        return {"accuracy": None, "sample_size": len(with_actuals), "decayed": False}

    recent = with_actuals[-window:]
    correct = sum(1 for p in recent if p["prediction"] == p["actual"])
    accuracy = correct / len(recent)

    return {
        "accuracy": round(accuracy, 4),
        "sample_size": len(recent),
        "decayed": accuracy < ACCURACY_MIN_THRESHOLD,
    }


# ──────────────────────────────────────────────────
#  DRIFT ANALYSIS ENGINE
# ──────────────────────────────────────────────────
def analyse_model_drift(prediction_log: list = None) -> dict:
    """
    Comprehensive drift analysis across all models.

    Returns:
        {
            models: {model_name: {status, kl_div, psi, conf_trend, accuracy, ...}},
            alerts: [str],
            retrain_needed: [model_names],
            timestamp: str,
        }
    """
    if prediction_log is None:
        try:
            if os.path.exists(PREDICTION_LOG_FILE):
                with open(PREDICTION_LOG_FILE, "r") as f:
                    prediction_log = json.load(f)
            else:
                prediction_log = []
        except Exception:
            prediction_log = []

    if len(prediction_log) < DRIFT_CHECK_MIN_SAMPLES:
        return {
            "models": {},
            "alerts": ["Not enough predictions for drift analysis"],
            "retrain_needed": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_predictions": len(prediction_log),
        }

    # Group by model
    by_model = defaultdict(list)
    for p in prediction_log:
        by_model[p.get("model", "unknown")].append(p)

    models_report = {}
    alerts = []
    retrain_needed = []

    for model_name, preds in by_model.items():
        if len(preds) < DRIFT_CHECK_MIN_SAMPLES:
            models_report[model_name] = {
                "status": "insufficient_data",
                "predictions": len(preds),
            }
            continue

        confidences = [p["confidence"] for p in preds if isinstance(p.get("confidence"), (int, float))]

        # Split into baseline (first half) and recent (second half)
        mid = len(confidences) // 2
        if mid < 5:
            models_report[model_name] = {
                "status": "insufficient_data",
                "predictions": len(preds),
            }
            continue

        baseline = np.array(confidences[:mid])
        recent = np.array(confidences[mid:])

        # 1. KL Divergence
        kl_div = 0.0
        try:
            # Create histograms and compute KL
            min_c, max_c = 0.0, 1.0
            bins = np.linspace(min_c, max_c, BASELINE_BINS + 1)
            base_hist, _ = np.histogram(baseline, bins=bins, density=True)
            recent_hist, _ = np.histogram(recent, bins=bins, density=True)
            if base_hist.sum() > 0 and recent_hist.sum() > 0:
                kl_div = compute_kl_divergence(recent_hist, base_hist)
        except Exception:
            kl_div = 0.0

        # 2. PSI
        psi = 0.0
        try:
            psi = compute_psi(baseline, recent)
        except Exception:
            psi = 0.0

        # 3. Confidence trend
        conf_trend = compute_confidence_trend(confidences)

        # 4. Rolling accuracy
        accuracy_info = compute_rolling_accuracy(preds)

        # Determine overall status
        drift_flags = []
        if kl_div > KL_DIVERGENCE_THRESHOLD:
            drift_flags.append(f"KL={kl_div:.3f}")
        if psi > PSI_THRESHOLD:
            drift_flags.append(f"PSI={psi:.3f}")
        if conf_trend["decaying"]:
            drift_flags.append(f"conf_decay={conf_trend['slope']:.4f}")
        if accuracy_info.get("decayed"):
            drift_flags.append(f"acc={accuracy_info['accuracy']:.2f}")

        if len(drift_flags) >= 2:
            status = "SEVERE_DRIFT"
            retrain_needed.append(model_name)
            alerts.append(f"🚨 {model_name}: SEVERE drift detected ({', '.join(drift_flags)})")
        elif len(drift_flags) == 1:
            status = "MILD_DRIFT"
            alerts.append(f"⚠️ {model_name}: Mild drift ({drift_flags[0]})")
        else:
            status = "STABLE"

        models_report[model_name] = {
            "status": status,
            "predictions": len(preds),
            "kl_divergence": round(kl_div, 4),
            "psi": round(psi, 4),
            "confidence_trend": conf_trend,
            "accuracy": accuracy_info,
            "baseline_mean_conf": round(float(baseline.mean()), 4),
            "recent_mean_conf": round(float(recent.mean()), 4),
            "drift_flags": drift_flags,
        }

    result = {
        "models": models_report,
        "alerts": alerts,
        "retrain_needed": retrain_needed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_predictions": len(prediction_log),
    }

    # Save state
    try:
        os.makedirs(os.path.dirname(DRIFT_STATE_FILE), exist_ok=True)
        with open(DRIFT_STATE_FILE, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    # Append to drift log
    try:
        log = []
        if os.path.exists(DRIFT_LOG_FILE):
            with open(DRIFT_LOG_FILE, "r") as f:
                log = json.load(f)
        log.append({
            "timestamp": result["timestamp"],
            "alerts_count": len(alerts),
            "retrain_needed": retrain_needed,
            "model_summary": {k: v["status"] for k, v in models_report.items()},
        })
        if len(log) > 500:
            log = log[-500:]
        with open(DRIFT_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass

    return result


# ──────────────────────────────────────────────────
#  DRIFT PENALTY (used by sniper for confidence scaling)
# ──────────────────────────────────────────────────
def get_drift_penalty(model_name: str) -> float:
    """
    Returns a confidence multiplier (0.0 to 1.0) based on drift severity.
    1.0 = no drift (full confidence)
    0.5 = severe drift (halve confidence)
    """
    try:
        if os.path.exists(DRIFT_STATE_FILE):
            with open(DRIFT_STATE_FILE, "r") as f:
                state = json.load(f)
            model_info = state.get("models", {}).get(model_name, {})
            status = model_info.get("status", "STABLE")

            if status == "SEVERE_DRIFT":
                return 0.5
            elif status == "MILD_DRIFT":
                return 0.8
            else:
                return 1.0
    except Exception:
        pass
    return 1.0


def get_all_drift_penalties() -> dict:
    """Get drift penalties for all models."""
    try:
        if os.path.exists(DRIFT_STATE_FILE):
            with open(DRIFT_STATE_FILE, "r") as f:
                state = json.load(f)
            penalties = {}
            for model_name, info in state.get("models", {}).items():
                status = info.get("status", "STABLE")
                if status == "SEVERE_DRIFT":
                    penalties[model_name] = 0.5
                elif status == "MILD_DRIFT":
                    penalties[model_name] = 0.8
                else:
                    penalties[model_name] = 1.0
            return penalties
    except Exception:
        pass
    return {}


def should_retrain(model_name: str) -> bool:
    """Check if a specific model needs retraining."""
    try:
        if os.path.exists(DRIFT_STATE_FILE):
            with open(DRIFT_STATE_FILE, "r") as f:
                state = json.load(f)
            return model_name in state.get("retrain_needed", [])
    except Exception:
        pass
    return False


# ──────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 Model Drift Detector")
    print("=" * 55)

    result = analyse_model_drift()

    print(f"\n  Total predictions logged: {result['total_predictions']}")
    print(f"  Models analysed: {len(result['models'])}")

    if result["models"]:
        print(f"\n  Model Status:")
        for model, info in result["models"].items():
            status = info.get("status", "?")
            if status == "SEVERE_DRIFT":
                icon = "🚨"
            elif status == "MILD_DRIFT":
                icon = "⚠️"
            elif status == "STABLE":
                icon = "✅"
            else:
                icon = "❓"

            print(f"    {icon} {model:15} | {status:15} | "
                  f"KL: {info.get('kl_divergence', 0):.4f} | "
                  f"PSI: {info.get('psi', 0):.4f} | "
                  f"Preds: {info.get('predictions', 0)}")

    if result["alerts"]:
        print(f"\n  Alerts:")
        for a in result["alerts"]:
            print(f"    {a}")

    if result["retrain_needed"]:
        print(f"\n  🔄 Retrain needed: {', '.join(result['retrain_needed'])}")
    else:
        print(f"\n  ✅ No retraining needed")

    print(f"\n  📁 State saved to {DRIFT_STATE_FILE}")
