"""
====================================================
🔄 PROJECT AEGIS — Model Versioning & A/B Deployment (Phase 11)
====================================================
Federated model versioning with shadow deployment:

1. **Version Registry**: Track all model versions with
   metadata (date, params, training metrics).
2. **Shadow Deploy**: Run new model version alongside
   production without affecting live trades.
3. **A/B Compare**: Compare shadow vs production on
   Sharpe, accuracy, drawdown over evaluation window.
4. **Auto-Promote**: When shadow beats production by
   threshold over N days, auto-promote to production.
5. **Rollback**: Instant rollback to any prior version.

Works with: RF, XGBoost, LSTM, RL-Sizer, Transformer
====================================================
"""

import os, sys, json, shutil, hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import STOCK_WATCHLIST, TOP_N_STOCKS

# ──────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
FILE_VERSIONING = os.path.join(DATA, "model_versioning_state.json")

# A/B test parameters
EVAL_WINDOW_DAYS   = 7       # Days to evaluate shadow
MIN_TRADES_EVAL    = 5       # Min trades for comparison
SHARPE_MARGIN      = 0.15    # Shadow must beat prod by this
ACCURACY_MARGIN    = 0.02    # 2% accuracy improvement
AUTO_PROMOTE       = True    # Auto-promote winning shadow
MAX_VERSIONS       = 10      # Keep last N versions

# Model types tracked
MODEL_TYPES = [
    "random_forest",
    "xgboost",
    "lstm_daily",
    "lstm_intraday",
    "rl_sizer",
    "transformer",
]

# Registry
_registry: Dict = {
    "versions": [],
    "production": {},
    "shadow": {},
    "comparison_history": [],
}


# ──────────────────────────────────────────────────
#  VERSION REGISTRY
# ──────────────────────────────────────────────────
def register_model_version(
    model_type: str,
    model_path: str,
    metrics: Dict = None,
    params: Dict = None,
    tag: str = None,
) -> Dict:
    """
    Register a new model version in the registry.
    Archives the model file and records metadata.
    """
    os.makedirs(VERSIONS_DIR, exist_ok=True)

    # Generate version ID
    timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    version_id = f"{model_type}_{timestamp}"

    # Hash the model file for dedup
    file_hash = ""
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        # Archive the model
        archive_path = os.path.join(VERSIONS_DIR, f"{version_id}_{file_hash}")
        ext = os.path.splitext(model_path)[1]
        archive_path += ext
        try:
            shutil.copy2(model_path, archive_path)
        except Exception:
            archive_path = model_path
    else:
        archive_path = model_path

    version = {
        "version_id": version_id,
        "model_type": model_type,
        "original_path": model_path,
        "archive_path": archive_path,
        "file_hash": file_hash,
        "tag": tag or "auto",
        "created_at": datetime.now(IST).isoformat(),
        "metrics": metrics or {},
        "params": params or {},
        "status": "registered",  # registered → shadow → production → retired
    }

    _registry["versions"].append(version)

    # Keep only last N versions per type
    type_versions = [v for v in _registry["versions"] if v["model_type"] == model_type]
    if len(type_versions) > MAX_VERSIONS:
        old = type_versions[:-MAX_VERSIONS]
        for v in old:
            v["status"] = "retired"
            # Optionally delete archived file
            try:
                if os.path.exists(v.get("archive_path", "")):
                    if v["archive_path"].startswith(VERSIONS_DIR):
                        os.remove(v["archive_path"])
            except Exception:
                pass

    return version


# ──────────────────────────────────────────────────
#  SHADOW DEPLOYMENT
# ──────────────────────────────────────────────────
def deploy_shadow(
    model_type: str,
    version_id: str = None,
) -> Dict:
    """
    Deploy a model version as shadow (alongside production).
    If version_id is None, use latest registered version.
    """
    type_versions = [
        v for v in _registry["versions"]
        if v["model_type"] == model_type and v["status"] in ("registered", "retired")
    ]

    if not type_versions:
        return {"error": f"No versions available for {model_type}"}

    if version_id:
        version = next((v for v in type_versions if v["version_id"] == version_id), None)
    else:
        version = type_versions[-1]  # Latest

    if version is None:
        return {"error": f"Version {version_id} not found"}

    version["status"] = "shadow"
    _registry["shadow"][model_type] = {
        "version_id": version["version_id"],
        "deployed_at": datetime.now(IST).isoformat(),
        "predictions": [],
        "trades": [],
        "pnl": [],
    }

    return {
        "status": "shadow_deployed",
        "model_type": model_type,
        "version_id": version["version_id"],
    }


# ──────────────────────────────────────────────────
#  SHADOW PREDICT (record shadow predictions)
# ──────────────────────────────────────────────────
def shadow_predict(
    model_type: str,
    symbol: str,
    prediction: float,
    actual: float = None,
    metadata: Dict = None,
) -> bool:
    """
    Record a prediction from the shadow model.
    Used for later comparison with production.
    """
    shadow = _registry["shadow"].get(model_type)
    if shadow is None:
        return False

    shadow["predictions"].append({
        "symbol": symbol,
        "prediction": prediction,
        "actual": actual,
        "timestamp": datetime.now(IST).isoformat(),
        "metadata": metadata or {},
    })

    return True


def record_shadow_trade(
    model_type: str,
    symbol: str,
    action: str,
    price: float,
    pnl: float = 0,
):
    """Record a shadow trade for comparison."""
    shadow = _registry["shadow"].get(model_type)
    if shadow is None:
        return

    shadow["trades"].append({
        "symbol": symbol,
        "action": action,
        "price": price,
        "pnl": pnl,
        "timestamp": datetime.now(IST).isoformat(),
    })
    if pnl != 0:
        shadow["pnl"].append(pnl)


# ──────────────────────────────────────────────────
#  COMPARE SHADOW vs PRODUCTION
# ──────────────────────────────────────────────────
def compare_versions(
    model_type: str,
    production_trades: List[Dict] = None,
) -> Dict:
    """
    Compare shadow model vs production model.
    Returns comparison metrics and promotion recommendation.
    """
    shadow = _registry["shadow"].get(model_type)
    if shadow is None:
        return {"status": "no_shadow", "recommendation": "HOLD"}

    # Shadow metrics
    shadow_preds = shadow.get("predictions", [])
    shadow_pnl = shadow.get("pnl", [])
    shadow_trades = shadow.get("trades", [])

    if len(shadow_preds) < MIN_TRADES_EVAL:
        return {
            "status": "insufficient_data",
            "shadow_predictions": len(shadow_preds),
            "min_required": MIN_TRADES_EVAL,
            "recommendation": "WAIT",
        }

    # Shadow accuracy (where actual is known)
    correct = 0
    total_with_actual = 0
    for p in shadow_preds:
        if p.get("actual") is not None:
            total_with_actual += 1
            # Prediction > 0.5 → BUY correct if actual > 0
            pred_up = p["prediction"] > 0.5
            actual_up = p["actual"] > 0
            if pred_up == actual_up:
                correct += 1

    shadow_accuracy = correct / max(1, total_with_actual)

    # Shadow Sharpe
    shadow_sharpe = _compute_sharpe(shadow_pnl)

    # Production metrics (from passed data or registry)
    prod_pnl = []
    prod_accuracy = 0.5  # Default baseline
    if production_trades:
        prod_pnl = [t.get("pnl", 0) for t in production_trades if t.get("pnl")]
        prod_correct = sum(1 for t in production_trades if t.get("pnl", 0) > 0)
        prod_accuracy = prod_correct / max(1, len(production_trades))

    prod_sharpe = _compute_sharpe(prod_pnl)

    # Decision
    sharpe_diff = shadow_sharpe - prod_sharpe
    acc_diff = shadow_accuracy - prod_accuracy

    if sharpe_diff > SHARPE_MARGIN and acc_diff > -ACCURACY_MARGIN:
        recommendation = "PROMOTE"
    elif sharpe_diff < -SHARPE_MARGIN:
        recommendation = "REJECT"
    else:
        recommendation = "CONTINUE"

    comparison = {
        "model_type": model_type,
        "shadow_version": shadow.get("version_id", "?"),
        "eval_window": {
            "predictions": len(shadow_preds),
            "trades": len(shadow_trades),
        },
        "shadow_metrics": {
            "sharpe": round(shadow_sharpe, 3),
            "accuracy": round(shadow_accuracy, 4),
            "total_pnl": round(sum(shadow_pnl), 2),
            "n_predictions": total_with_actual,
        },
        "production_metrics": {
            "sharpe": round(prod_sharpe, 3),
            "accuracy": round(prod_accuracy, 4),
            "total_pnl": round(sum(prod_pnl), 2),
        },
        "comparison": {
            "sharpe_diff": round(sharpe_diff, 3),
            "accuracy_diff": round(acc_diff, 4),
        },
        "recommendation": recommendation,
        "timestamp": datetime.now(IST).isoformat(),
    }

    _registry["comparison_history"].append(comparison)
    return comparison


def _compute_sharpe(pnl_list: List[float], risk_free: float = 0.0) -> float:
    """Compute Sharpe ratio from PnL list."""
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list, dtype=float)
    mean_ret = np.mean(arr) - risk_free
    std_ret = np.std(arr)
    if std_ret < 1e-6:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(252))


# ──────────────────────────────────────────────────
#  AUTO-PROMOTE / ROLLBACK
# ──────────────────────────────────────────────────
def auto_promote_check(model_type: str) -> Dict:
    """
    Check if shadow should be auto-promoted.
    If yes, swap production model.
    """
    comparison = compare_versions(model_type)

    if comparison.get("recommendation") != "PROMOTE" or not AUTO_PROMOTE:
        return comparison

    shadow = _registry["shadow"].get(model_type, {})
    version_id = shadow.get("version_id")
    if not version_id:
        return comparison

    # Find version
    version = next(
        (v for v in _registry["versions"] if v["version_id"] == version_id),
        None,
    )
    if version is None:
        return comparison

    # Promote: move shadow to production
    old_prod = _registry["production"].get(model_type, {})
    if old_prod:
        # Retire old production
        old_v = next(
            (v for v in _registry["versions"] if v["version_id"] == old_prod.get("version_id")),
            None,
        )
        if old_v:
            old_v["status"] = "retired"

    version["status"] = "production"
    _registry["production"][model_type] = {
        "version_id": version_id,
        "promoted_at": datetime.now(IST).isoformat(),
        "archive_path": version.get("archive_path", ""),
    }

    # Copy archived model to production path
    prod_path = version.get("original_path", "")
    archive = version.get("archive_path", "")
    if archive and prod_path and os.path.exists(archive):
        try:
            shutil.copy2(archive, prod_path)
        except Exception:
            pass

    # Clear shadow slot
    _registry["shadow"].pop(model_type, None)

    comparison["auto_promoted"] = True
    comparison["new_production"] = version_id
    return comparison


def rollback(model_type: str, version_id: str) -> Dict:
    """
    Rollback production to a specific version.
    """
    version = next(
        (v for v in _registry["versions"]
         if v["version_id"] == version_id and v["model_type"] == model_type),
        None,
    )
    if version is None:
        return {"error": f"Version {version_id} not found"}

    # Restore model file
    archive = version.get("archive_path", "")
    prod_path = version.get("original_path", "")
    if archive and prod_path and os.path.exists(archive):
        try:
            shutil.copy2(archive, prod_path)
        except Exception:
            pass

    version["status"] = "production"
    _registry["production"][model_type] = {
        "version_id": version_id,
        "promoted_at": datetime.now(IST).isoformat(),
        "rollback": True,
    }

    return {
        "status": "rolled_back",
        "model_type": model_type,
        "version_id": version_id,
    }


# ──────────────────────────────────────────────────
#  STATUS & DASHBOARD DATA
# ──────────────────────────────────────────────────
def get_versioning_status() -> Dict:
    """Get full versioning status for dashboard."""
    return {
        "production": _registry["production"],
        "shadow": {
            mt: {
                "version_id": s.get("version_id"),
                "deployed_at": s.get("deployed_at"),
                "n_predictions": len(s.get("predictions", [])),
                "n_trades": len(s.get("trades", [])),
                "total_pnl": round(sum(s.get("pnl", [])), 2),
            }
            for mt, s in _registry["shadow"].items()
        },
        "versions_count": len(_registry["versions"]),
        "versions_by_type": {
            mt: len([v for v in _registry["versions"] if v["model_type"] == mt])
            for mt in MODEL_TYPES
        },
        "recent_comparisons": _registry["comparison_history"][-5:],
        "timestamp": datetime.now(IST).isoformat(),
    }


# ──────────────────────────────────────────────────
#  SAVE / LOAD
# ──────────────────────────────────────────────────
def save_versioning_state(data: dict = None):
    os.makedirs(DATA, exist_ok=True)
    payload = data or {
        "registry": {
            "versions": _registry["versions"],
            "production": _registry["production"],
            "shadow": {
                k: {
                    "version_id": v.get("version_id"),
                    "deployed_at": v.get("deployed_at"),
                    "n_predictions": len(v.get("predictions", [])),
                    "total_pnl": round(sum(v.get("pnl", [])), 2),
                }
                for k, v in _registry["shadow"].items()
            },
            "comparison_history": _registry["comparison_history"][-20:],
        },
        "timestamp": datetime.now(IST).isoformat(),
    }
    with open(FILE_VERSIONING, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_versioning_state() -> Dict:
    global _registry
    if os.path.exists(FILE_VERSIONING):
        try:
            with open(FILE_VERSIONING, "r") as f:
                raw = json.load(f)
            reg = raw.get("registry", {})
            _registry["versions"] = reg.get("versions", [])
            _registry["production"] = reg.get("production", {})
            _registry["comparison_history"] = reg.get("comparison_history", [])
            return raw
        except Exception:
            pass
    return {}
