"""
====================================================
🔧 PROJECT AEGIS — Auto Hyperparameter Tuner (Phase 10)
====================================================
Optuna-based periodic sweep of gate thresholds,
model hyper-params, and sizing parameters.

Objective:  Maximize paper-trading Sharpe from
            historical trade log via simulated replay.

Sweeps:
  ─ NeuroVoter consensus threshold (3-6)
  ─ Kelly fraction (0.10 – 0.50)
  ─ Confidence floor (0.50 – 0.80)
  ─ Risk per trade % (0.5 – 3.0)
  ─ RL epsilon (0.01 – 0.30)
  ─ Liquidity gate threshold (10 – 50)
  ─ Intermarket exposure bounds
  ─ Model retrain window (30 – 120 days)

Persistence: best params → data/tuner_state.json
Schedule:    weekends only (or manual)
====================================================
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))

from config import CAPITAL

# Try optuna; graceful fallback if not installed
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

import pytz
IST = pytz.timezone("Asia/Kolkata")

# ──────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_TUNER = os.path.join(DATA, "tuner_state.json")
TRADE_LOG = os.path.join(DATA, "trade_history.csv")
BEST_PARAMS = os.path.join(DATA, "best_params.json")

N_TRIALS = 60          # Optuna trials per sweep
N_STARTUP_TRIALS = 10  # Random search before Bayesian TPE
MIN_TRADES = 30        # Need this many logged trades


# ──────────────────────────────────────────────────
#  LOAD TRADE HISTORY
# ──────────────────────────────────────────────────
def _load_trades() -> pd.DataFrame:
    """Load trade_history.csv for replay."""
    if not os.path.exists(TRADE_LOG):
        return pd.DataFrame()
    try:
        df = pd.read_csv(TRADE_LOG)
        if "pnl" not in df.columns and "PnL" in df.columns:
            df.rename(columns={"PnL": "pnl"}, inplace=True)
        if "pnl" not in df.columns:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────
#  SIMULATED REPLAY ENGINE
# ──────────────────────────────────────────────────
def _simulate_sharpe(trades: pd.DataFrame, params: Dict) -> float:
    """
    Replay historical trades with modified parameters
    and return the resulting Sharpe ratio.

    We scale PnL by how parameters would have changed
    position size, confidence filtering, etc.
    """
    if len(trades) < MIN_TRADES:
        return 0.0

    conf_floor = params.get("confidence_floor", 0.60)
    kelly_frac = params.get("kelly_fraction", 0.25)
    risk_pct = params.get("risk_per_trade_pct", 1.5) / 100
    consensus_thr = params.get("consensus_threshold", 4)
    liq_gate = params.get("liquidity_gate", 30)

    filtered_pnl = []

    for _, row in trades.iterrows():
        # Apply confidence filter
        conf = float(row.get("confidence", row.get("Confidence", 0.5)))
        if conf < conf_floor:
            continue

        # Apply consensus filter
        votes = int(row.get("votes", row.get("Votes", consensus_thr)))
        if votes < consensus_thr:
            continue

        # Scale PnL by Kelly fraction adjustment
        base_pnl = float(row["pnl"])
        scale = kelly_frac / 0.25  # relative to default 0.25
        scale *= risk_pct / 0.015  # relative to default 1.5%
        filtered_pnl.append(base_pnl * scale)

    if len(filtered_pnl) < 10:
        return -1.0

    returns = np.array(filtered_pnl)
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)

    if std_r < 1e-9:
        return 0.0

    sharpe = (mean_r / std_r) * np.sqrt(252)  # Annualised
    # Penalize if too few trades (we want activity)
    activity_penalty = max(0, 1 - len(filtered_pnl) / MIN_TRADES)
    sharpe -= activity_penalty

    return float(sharpe)


# ──────────────────────────────────────────────────
#  OPTUNA OBJECTIVE
# ──────────────────────────────────────────────────
def _create_objective(trades: pd.DataFrame):
    """Create objective closure for Optuna."""

    def objective(trial):
        params = {
            "consensus_threshold": trial.suggest_int("consensus_threshold", 3, 6),
            "kelly_fraction": trial.suggest_float("kelly_fraction", 0.10, 0.50, step=0.05),
            "confidence_floor": trial.suggest_float("confidence_floor", 0.45, 0.85, step=0.05),
            "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.5, 3.0, step=0.25),
            "rl_epsilon": trial.suggest_float("rl_epsilon", 0.01, 0.30, step=0.02),
            "liquidity_gate": trial.suggest_int("liquidity_gate", 10, 50, step=5),
            "intermarket_min_mult": trial.suggest_float("intermarket_min_mult", 0.3, 0.7, step=0.1),
            "retrain_window_days": trial.suggest_int("retrain_window_days", 30, 120, step=15),
        }
        return _simulate_sharpe(trades, params)

    return objective


# ──────────────────────────────────────────────────
#  RUN TUNING SWEEP
# ──────────────────────────────────────────────────
def run_tuning_sweep(n_trials: int = None) -> Dict:
    """
    Run Optuna sweep.  Returns best params + study stats.
    Saves results to data/tuner_state.json.
    """
    if not HAS_OPTUNA:
        return {
            "status": "SKIPPED",
            "reason": "optuna not installed — pip install optuna",
            "timestamp": datetime.now(IST).isoformat(),
        }

    trades = _load_trades()
    if len(trades) < MIN_TRADES:
        return {
            "status": "SKIPPED",
            "reason": f"Only {len(trades)} trades logged, need {MIN_TRADES}",
            "timestamp": datetime.now(IST).isoformat(),
        }

    trials = n_trials or N_TRIALS

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS),
        study_name="aegis_tuner",
    )
    study.optimize(_create_objective(trades), n_trials=trials)

    best = study.best_params
    best_val = study.best_value

    # Top-5 trials
    top5 = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)[:5]
    top5_list = [
        {"params": t.params, "sharpe": round(t.value, 4)}
        for t in top5 if t.value is not None
    ]

    result = {
        "status": "COMPLETED",
        "best_params": best,
        "best_sharpe": round(best_val, 4),
        "n_trials": trials,
        "n_trades_used": len(trades),
        "top_5": top5_list,
        "timestamp": datetime.now(IST).isoformat(),
    }

    save_tuner_state(result)
    _apply_best_params(best)
    return result


# ──────────────────────────────────────────────────
#  APPLY BEST PARAMS  (write to best_params.json)
# ──────────────────────────────────────────────────
def _apply_best_params(params: Dict):
    """
    Write tuned params so Sniper / modules can read them
    on next run.  Non-destructive: only overwrites keys
    present in the tuned params dict.
    """
    os.makedirs(DATA, exist_ok=True)
    existing = {}
    if os.path.exists(BEST_PARAMS):
        try:
            with open(BEST_PARAMS, "r") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update(params)
    existing["_tuned_at"] = datetime.now(IST).isoformat()
    with open(BEST_PARAMS, "w") as f:
        json.dump(existing, f, indent=2, default=str)


# ──────────────────────────────────────────────────
#  QUICK STATUS (no sweep)
# ──────────────────────────────────────────────────
def get_tuner_status() -> Dict:
    """Read last tuner state from disk."""
    if os.path.exists(FILE_TUNER):
        try:
            with open(FILE_TUNER, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"status": "NO_DATA", "reason": "Tuner not yet run"}


def get_best_params() -> Dict:
    """Read best_params.json if available."""
    if os.path.exists(BEST_PARAMS):
        try:
            with open(BEST_PARAMS, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ──────────────────────────────────────────────────
#  PARAMETER SENSITIVITY ANALYSIS
# ──────────────────────────────────────────────────
def sensitivity_report(param_name: str = "confidence_floor") -> Dict:
    """
    Run a 1D sweep of a single parameter while keeping
    others at their tuned values.  Returns list of
    (value, sharpe) pairs for plotting.
    """
    trades = _load_trades()
    if len(trades) < MIN_TRADES:
        return {"status": "NO_DATA"}

    base_params = get_best_params()
    if not base_params:
        base_params = {
            "consensus_threshold": 4,
            "kelly_fraction": 0.25,
            "confidence_floor": 0.60,
            "risk_per_trade_pct": 1.5,
            "liquidity_gate": 30,
        }

    ranges = {
        "confidence_floor": np.arange(0.40, 0.90, 0.05),
        "kelly_fraction": np.arange(0.05, 0.55, 0.05),
        "risk_per_trade_pct": np.arange(0.5, 3.5, 0.25),
        "consensus_threshold": range(2, 7),
        "liquidity_gate": range(10, 55, 5),
    }

    vals = ranges.get(param_name, np.arange(0, 1, 0.1))
    curve = []
    for v in vals:
        test_params = dict(base_params)
        test_params[param_name] = float(v)
        s = _simulate_sharpe(trades, test_params)
        curve.append({"value": round(float(v), 4), "sharpe": round(s, 4)})

    return {
        "parameter": param_name,
        "curve": curve,
        "timestamp": datetime.now(IST).isoformat(),
    }


# ──────────────────────────────────────────────────
#  SAVE STATE
# ──────────────────────────────────────────────────
def save_tuner_state(data: dict = None):
    """Persist tuner state for dashboard."""
    os.makedirs(DATA, exist_ok=True)
    if data is None:
        data = get_tuner_status()
    with open(FILE_TUNER, "w") as f:
        json.dump(data, f, indent=2, default=str)
