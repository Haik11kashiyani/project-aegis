"""
====================================================
🧮 PROJECT AEGIS — Bayesian Signal Fusion (Phase 10)
====================================================
Replaces simple majority-vote NeuroVoter aggregation
with a full Bayesian posterior-update that weighs each
model / voter by its RECENT accuracy (not all-time).

How it works:
  1. Each model (RF, XGB, dailyLSTM, intraLSTM) and
     each NeuroVoter has a Beta(α, β) prior tracking
     its accuracy over a rolling window.
  2. On each new signal, compute the posterior
     predictive P(correct | history) for every source.
  3. Weight each BUY/SELL signal by that posterior.
  4. The combined posterior probability is used as the
     final signal strength.

Benefits:
  • A model that went cold last week automatically
    gets less weight — no manual tuning required.
  • New models start with a diffuse prior and earn
    weight as they prove themselves.
  • The prior itself is an anti-overfit mechanism:
    extraordinary confidence needs extraordinary
    evidence.
====================================================
"""

import os
import sys
import json
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import TRADE_LOG_FILE

# ──────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_BAYESIAN = os.path.join(DATA, "bayesian_fusion.json")
FILE_BAYES_PRIORS = os.path.join(DATA, "bayesian_priors.json")

# Rolling window for accuracy tracking (trades)
ROLLING_WINDOW = 50
# Prior strength — higher = more conservative start
PRIOR_ALPHA = 2.0   # "virtual correct" observations
PRIOR_BETA = 2.0    # "virtual incorrect" observations
# Minimum weight any source can have (prevents zero-out)
MIN_WEIGHT = 0.05
# Decay factor per day of no data (slow drift to prior)
DAILY_DECAY = 0.98


# ──────────────────────────────────────────────────
#  BETA DISTRIBUTION HELPERS
# ──────────────────────────────────────────────────
def beta_mean(alpha: float, beta_v: float) -> float:
    """Mean of Beta(alpha, beta) = alpha / (alpha + beta)."""
    return alpha / (alpha + beta_v) if (alpha + beta_v) > 0 else 0.5


def beta_variance(alpha: float, beta_v: float) -> float:
    """Variance of Beta distribution."""
    s = alpha + beta_v
    if s <= 0:
        return 0.25
    return (alpha * beta_v) / (s ** 2 * (s + 1))


def beta_entropy(alpha: float, beta_v: float) -> float:
    """Approximate entropy of Beta distribution (higher = more uncertain)."""
    mu = beta_mean(alpha, beta_v)
    if mu <= 0 or mu >= 1:
        return 0.0
    return -(mu * math.log(mu + 1e-10) + (1 - mu) * math.log(1 - mu + 1e-10))


# ──────────────────────────────────────────────────
#  SOURCE TRACKER
# ──────────────────────────────────────────────────
class SourceTracker:
    """Tracks Beta(α,β) prior for one signal source."""

    def __init__(self, name: str, alpha: float = PRIOR_ALPHA,
                 beta_v: float = PRIOR_BETA):
        self.name = name
        self.alpha = alpha
        self.beta_v = beta_v
        self.observations: List[dict] = []  # [{correct: bool, ts: str}]

    def record(self, correct: bool):
        """Update posterior with new observation."""
        if correct:
            self.alpha += 1.0
        else:
            self.beta_v += 1.0
        self.observations.append({
            "correct": correct,
            "ts": datetime.now(IST).isoformat(),
        })
        # Keep rolling window
        if len(self.observations) > ROLLING_WINDOW * 2:
            self._trim()

    def _trim(self):
        """Keep only ROLLING_WINDOW most recent observations and rebuild αβ."""
        self.observations = self.observations[-ROLLING_WINDOW:]
        correct = sum(1 for o in self.observations if o["correct"])
        incorrect = len(self.observations) - correct
        self.alpha = PRIOR_ALPHA + correct
        self.beta_v = PRIOR_BETA + incorrect

    @property
    def accuracy(self) -> float:
        return beta_mean(self.alpha, self.beta_v)

    @property
    def uncertainty(self) -> float:
        return math.sqrt(beta_variance(self.alpha, self.beta_v))

    @property
    def confidence(self) -> float:
        """Inversely proportional to uncertainty."""
        return max(0, 1 - 2 * self.uncertainty)

    @property
    def weight(self) -> float:
        """Weight for fusion — accuracy × confidence, floored."""
        return max(MIN_WEIGHT, self.accuracy * self.confidence)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "alpha": round(self.alpha, 2),
            "beta": round(self.beta_v, 2),
            "accuracy": round(self.accuracy, 4),
            "uncertainty": round(self.uncertainty, 4),
            "confidence": round(self.confidence, 4),
            "weight": round(self.weight, 4),
            "observations": len(self.observations),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SourceTracker':
        st = cls(d["name"], d.get("alpha", PRIOR_ALPHA), d.get("beta", PRIOR_BETA))
        st.observations = d.get("raw_observations", [])
        return st


# ──────────────────────────────────────────────────
#  BAYESIAN FUSION ENGINE
# ──────────────────────────────────────────────────
class BayesianFusion:
    """
    Maintains per-source accuracy priors and fuses signals.
    """

    # Standard model sources
    MODEL_SOURCES = ["RF", "XGB", "DailyLSTM", "IntraLSTM"]
    # NeuroVoter sources added dynamically
    VOTER_SOURCES = [
        "MomentumHunter", "ValueSeeker", "RiskManager",
        "SentimentReader", "PatternDetective", "QuantAnalyst",
    ]
    ALL_SOURCES = MODEL_SOURCES + VOTER_SOURCES

    def __init__(self):
        self.trackers: Dict[str, SourceTracker] = {}
        self._load()
        # Ensure all sources exist
        for name in self.ALL_SOURCES:
            if name not in self.trackers:
                self.trackers[name] = SourceTracker(name)

    def _load(self):
        try:
            if os.path.exists(FILE_BAYES_PRIORS):
                with open(FILE_BAYES_PRIORS, "r") as f:
                    data = json.load(f)
                for d in data.get("sources", []):
                    st = SourceTracker.from_dict(d)
                    self.trackers[st.name] = st
        except Exception:
            pass

    def save(self):
        os.makedirs(DATA, exist_ok=True)
        data = {
            "sources": [],
            "timestamp": datetime.now(IST).isoformat(),
        }
        for st in self.trackers.values():
            d = st.to_dict()
            d["raw_observations"] = st.observations[-ROLLING_WINDOW:]
            data["sources"].append(d)
        with open(FILE_BAYES_PRIORS, "w") as f:
            json.dump(data, f, indent=2)

    # ──────────────────────────────────────────────
    #  SIGNAL FUSION
    # ──────────────────────────────────────────────
    def fuse_signals(
        self,
        signals: Dict[str, float],
    ) -> Dict:
        """
        Fuse multiple source signals using Bayesian weights.

        Args:
            signals: {source_name: signal_value}
                     signal_value ∈ [-1, +1] (negative=sell, positive=buy)

        Returns:
            {
                "fused_signal": float,       # weighted average [-1, +1]
                "fused_probability": float,  # mapped to [0, 1] (buy probability)
                "should_buy": bool,
                "conviction": float,
                "weights": {source: weight},
                "contributions": {source: signal × weight},
            }
        """
        weights = {}
        contributions = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for name, signal in signals.items():
            tracker = self.trackers.get(name)
            if tracker is None:
                tracker = SourceTracker(name)
                self.trackers[name] = tracker
            w = tracker.weight
            weights[name] = round(w, 4)
            contribution = signal * w
            contributions[name] = round(contribution, 4)
            weighted_sum += contribution
            total_weight += w

        fused = weighted_sum / total_weight if total_weight > 0 else 0.0
        fused = max(-1.0, min(1.0, fused))

        # Map to probability [0, 1]
        buy_prob = (fused + 1) / 2

        # Conviction = how far from 0.5 (scaled 0-1)
        conviction = abs(fused)

        return {
            "fused_signal": round(fused, 4),
            "fused_probability": round(buy_prob, 4),
            "should_buy": fused > 0.15,  # Slightly positive threshold
            "conviction": round(conviction, 4),
            "weights": weights,
            "contributions": contributions,
            "total_weight": round(total_weight, 4),
        }

    # ──────────────────────────────────────────────
    #  OUTCOME RECORDING
    # ──────────────────────────────────────────────
    def record_outcome(
        self,
        predictions: Dict[str, float],
        was_profitable: bool,
    ):
        """
        After trade closes, update each source's accuracy.
        If the source predicted BUY (>0) and it was profitable → correct.
        If the source predicted SELL (<0) and it was unprofitable → correct.
        """
        for name, signal in predictions.items():
            tracker = self.trackers.get(name)
            if tracker is None:
                continue
            predicted_buy = signal > 0
            correct = (predicted_buy and was_profitable) or (
                not predicted_buy and not was_profitable
            )
            tracker.record(correct)
        self.save()

    # ──────────────────────────────────────────────
    #  DASHBOARD DATA
    # ──────────────────────────────────────────────
    def get_summary(self) -> Dict:
        """Return data for dashboard."""
        sources = []
        for name in self.ALL_SOURCES:
            tracker = self.trackers.get(name)
            if tracker:
                sources.append(tracker.to_dict())

        weights = {s["name"]: s["weight"] for s in sources}
        best = max(sources, key=lambda x: x["weight"]) if sources else {}
        worst = min(sources, key=lambda x: x["weight"]) if sources else {}

        return {
            "sources": sources,
            "weights": weights,
            "best_source": best.get("name", "N/A"),
            "worst_source": worst.get("name", "N/A"),
            "avg_accuracy": round(np.mean([s["accuracy"] for s in sources]), 4) if sources else 0,
            "total_observations": sum(s["observations"] for s in sources),
            "timestamp": datetime.now(IST).isoformat(),
        }


# ──────────────────────────────────────────────────
#  SINGLETON
# ──────────────────────────────────────────────────
_fusion: Optional[BayesianFusion] = None


def get_fusion() -> BayesianFusion:
    global _fusion
    if _fusion is None:
        _fusion = BayesianFusion()
    return _fusion


# ──────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────
def bayesian_fuse(
    rf_conf: float, xgb_conf: float, lstm_conf: float, intra_conf: float,
    voter_signals: Dict[str, float] = None,
    threshold: float = 0.60,
) -> Dict:
    """
    Convenience function: combine 4 model confidences + NeuroVoter
    signals through Bayesian fusion.

    Model confidences [0,1] are mapped to [-1,+1]: signal = 2*(conf - 0.5).
    """
    fusion = get_fusion()

    signals = {
        "RF":         2 * (rf_conf - 0.5),
        "XGB":        2 * (xgb_conf - 0.5),
        "DailyLSTM":  2 * (lstm_conf - 0.5),
        "IntraLSTM":  2 * (intra_conf - 0.5),
    }
    if voter_signals:
        signals.update(voter_signals)

    result = fusion.fuse_signals(signals)
    result["threshold"] = threshold
    result["method"] = "BAYESIAN_FUSION"
    return result


def record_bayesian_outcome(
    rf_conf: float, xgb_conf: float, lstm_conf: float, intra_conf: float,
    voter_signals: Dict[str, float] = None,
    was_profitable: bool = False,
):
    """Record trade outcome to update priors."""
    fusion = get_fusion()
    signals = {
        "RF":         2 * (rf_conf - 0.5),
        "XGB":        2 * (xgb_conf - 0.5),
        "DailyLSTM":  2 * (lstm_conf - 0.5),
        "IntraLSTM":  2 * (intra_conf - 0.5),
    }
    if voter_signals:
        signals.update(voter_signals)
    fusion.record_outcome(signals, was_profitable)


def save_bayesian_state(data: dict = None):
    """Save fusion summary for dashboard."""
    os.makedirs(DATA, exist_ok=True)
    if data is None:
        data = get_fusion().get_summary()
    with open(FILE_BAYESIAN, "w") as f:
        json.dump(data, f, indent=2, default=str)
