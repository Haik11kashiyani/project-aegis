"""
====================================================
PROJECT AEGIS — Regime Detector (Hidden Markov Model)
====================================================
Detects market regime (BULL / BEAR / SIDEWAYS) using a
lightweight Gaussian HMM fitted on returns + volatility.

No external HMM library needed — uses a simplified
Baum-Welch / Viterbi implementation with NumPy only.

Outputs
-------
data/regime_state.json
  {
    "regime": "BULL" | "BEAR" | "SIDEWAYS",
    "confidence": 0.0 - 1.0,
    "regime_history": [...],
    "personality_override": "AGGRESSIVE" | "CONSERVATIVE" | null,
    "sizing_multiplier": 0.5 - 1.5,
    "timestamp": "..."
  }
====================================================
"""

import os, json, time, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REGIME_FILE = os.path.join(DATA_DIR, "regime_state.json")
_cache = {"result": None, "ts": 0}
CACHE_TTL = 1800  # 30 min


# ══════════════════════════════════════════════════
#  GAUSSIAN HMM (3-state, NumPy-only)
# ══════════════════════════════════════════════════
class SimpleGaussianHMM:
    """Minimal 3-state Gaussian HMM (BULL / SIDEWAYS / BEAR)."""

    def __init__(self, n_states: int = 3, n_iter: int = 30, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        # Initialise parameters (will be overwritten by fit)
        self.pi = np.ones(n_states) / n_states
        self.A = np.full((n_states, n_states), 1.0 / n_states)
        self.means = np.zeros(n_states)
        self.variances = np.ones(n_states)
        self._fitted = False

    @staticmethod
    def _gauss_pdf(x: float, mu: float, var: float) -> float:
        if var <= 1e-12:
            var = 1e-12
        return (1.0 / math.sqrt(2 * math.pi * var)) * math.exp(-0.5 * ((x - mu) ** 2) / var)

    def _emission(self, obs: np.ndarray) -> np.ndarray:
        """T × K emission probability matrix."""
        T = len(obs)
        B = np.zeros((T, self.n_states))
        for t in range(T):
            for k in range(self.n_states):
                B[t, k] = self._gauss_pdf(obs[t], self.means[k], self.variances[k])
        return np.clip(B, 1e-300, None)

    def _forward(self, B: np.ndarray):
        T, K = B.shape
        alpha = np.zeros((T, K))
        alpha[0] = self.pi * B[0]
        alpha[0] /= alpha[0].sum() + 1e-300
        for t in range(1, T):
            for j in range(K):
                alpha[t, j] = B[t, j] * np.sum(alpha[t - 1] * self.A[:, j])
            s = alpha[t].sum()
            if s > 0:
                alpha[t] /= s
        return alpha

    def _backward(self, B: np.ndarray):
        T, K = B.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(K):
                beta[t, i] = np.sum(self.A[i] * B[t + 1] * beta[t + 1])
            s = beta[t].sum()
            if s > 0:
                beta[t] /= s
        return beta

    def fit(self, obs: np.ndarray):
        """Baum-Welch EM."""
        T = len(obs)
        if T < 10:
            return self
        # Smart init: sort returns → assign thirds to BEAR / SIDEWAYS / BULL
        sorted_obs = np.sort(obs)
        third = max(T // 3, 1)
        self.means = np.array([
            sorted_obs[:third].mean(),        # BEAR (low returns)
            sorted_obs[third:2*third].mean(),  # SIDEWAYS
            sorted_obs[2*third:].mean(),       # BULL (high returns)
        ])
        self.variances = np.array([
            sorted_obs[:third].var() + 1e-6,
            sorted_obs[third:2*third].var() + 1e-6,
            sorted_obs[2*third:].var() + 1e-6,
        ])
        # Sticky transitions (0.8 self, 0.1+0.1 others)
        self.A = np.full((self.n_states, self.n_states), 0.1)
        np.fill_diagonal(self.A, 0.8)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(self.n_states) / self.n_states

        prev_ll = -1e18
        for iteration in range(self.n_iter):
            B = self._emission(obs)
            alpha = self._forward(B)
            beta = self._backward(B)

            # Gamma: P(state_t | obs)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Xi: P(state_t, state_{t+1} | obs)
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.A[i, j] * B[t + 1, j] * beta[t + 1, j]
                s = xi[t].sum()
                if s > 0:
                    xi[t] /= s

            # Re-estimate
            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)
            for k in range(self.n_states):
                g_sum = gamma[:, k].sum() + 1e-300
                g_sum_no_last = gamma[:-1, k].sum() + 1e-300
                self.means[k] = (gamma[:, k] * obs).sum() / g_sum
                diff = obs - self.means[k]
                self.variances[k] = (gamma[:, k] * diff ** 2).sum() / g_sum + 1e-6
                for j in range(self.n_states):
                    self.A[k, j] = xi[:, k, j].sum() / g_sum_no_last
            self.A /= self.A.sum(axis=1, keepdims=True) + 1e-300

            # Log-likelihood (approx)
            ll = np.log(alpha[-1].sum() + 1e-300)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self._fitted = True
        return self

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi decoding → state sequence."""
        T = len(obs)
        B = self._emission(obs)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = np.log(self.pi + 1e-300) + np.log(B[0] + 1e-300)
        for t in range(1, T):
            for j in range(self.n_states):
                candidates = delta[t - 1] + np.log(self.A[:, j] + 1e-300)
                psi[t, j] = int(np.argmax(candidates))
                delta[t, j] = candidates[psi[t, j]] + np.log(B[t, j] + 1e-300)

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def state_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return current state probabilities (last timestep)."""
        B = self._emission(obs)
        alpha = self._forward(B)
        probs = alpha[-1]
        return probs / (probs.sum() + 1e-300)


# ══════════════════════════════════════════════════
#  REGIME MAPPING
# ══════════════════════════════════════════════════
REGIME_NAMES = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}
PERSONALITY_MAP = {
    "BULL": "AGGRESSIVE",
    "BEAR": "CONSERVATIVE",
    "SIDEWAYS": None,  # keep user default
}
SIZING_MAP = {
    "BULL": 1.3,
    "BEAR": 0.5,
    "SIDEWAYS": 0.85,
}


def _get_market_returns(lookback_days: int = 252) -> np.ndarray:
    """Fetch Nifty 50 daily returns for regime fitting."""
    try:
        import yfinance as yf
        nifty = yf.download("^NSEI", period=f"{lookback_days}d", interval="1d",
                            progress=False, auto_adjust=True)
        if nifty is None or nifty.empty:
            raise ValueError("empty data")
        close = nifty["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        returns = np.log(close / close.shift(1)).dropna().values
        return returns
    except Exception:
        return np.array([])


# ══════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════
def detect_regime(lookback: int = 252, force: bool = False) -> dict:
    """
    Fit HMM on Nifty 50 returns → classify current market regime.
    Returns dict with regime, confidence, personality override, sizing.
    """
    global _cache
    if not force and _cache["result"] and (time.time() - _cache["ts"] < CACHE_TTL):
        return _cache["result"]

    returns = _get_market_returns(lookback)
    if len(returns) < 60:
        result = {
            "regime": "SIDEWAYS",
            "confidence": 0.0,
            "regime_history": [],
            "personality_override": None,
            "sizing_multiplier": 1.0,
            "state_probs": {"BEAR": 0.33, "SIDEWAYS": 0.34, "BULL": 0.33},
            "timestamp": _now_ist(),
            "error": "insufficient data",
        }
        _cache = {"result": result, "ts": time.time()}
        return result

    hmm = SimpleGaussianHMM(n_states=3, n_iter=40)
    hmm.fit(returns)

    # Ensure states ordered by mean: state0=BEAR, state1=SIDEWAYS, state2=BULL
    order = np.argsort(hmm.means)
    ordered_means = hmm.means[order]
    # Remap if needed
    remap = {int(order[i]): i for i in range(3)}

    path = hmm.predict(returns)
    remapped_path = np.array([remap[s] for s in path])

    probs = hmm.state_probs(returns)
    remapped_probs = np.array([probs[order[i]] for i in range(3)])

    current_state = int(remapped_path[-1])
    regime = REGIME_NAMES.get(current_state, "SIDEWAYS")
    confidence = float(remapped_probs[current_state])

    # Build history (last 60 days)
    history = []
    for s in remapped_path[-60:]:
        history.append(REGIME_NAMES.get(int(s), "SIDEWAYS"))

    # Regime transition detection
    if len(remapped_path) >= 5:
        recent = remapped_path[-5:]
        if len(set(recent)) > 1:
            # Transition in progress
            transition = True
        else:
            transition = False
    else:
        transition = False

    result = {
        "regime": regime,
        "confidence": round(confidence, 4),
        "regime_history": history,
        "personality_override": PERSONALITY_MAP.get(regime),
        "sizing_multiplier": SIZING_MAP.get(regime, 1.0),
        "state_probs": {
            "BEAR": round(float(remapped_probs[0]), 4),
            "SIDEWAYS": round(float(remapped_probs[1]), 4),
            "BULL": round(float(remapped_probs[2]), 4),
        },
        "state_means": {
            "BEAR": round(float(ordered_means[0]) * 100, 4),
            "SIDEWAYS": round(float(ordered_means[1]) * 100, 4),
            "BULL": round(float(ordered_means[2]) * 100, 4),
        },
        "transition_active": transition,
        "lookback_days": lookback,
        "observations": len(returns),
        "timestamp": _now_ist(),
    }

    _cache = {"result": result, "ts": time.time()}
    return result


def get_regime_gate(current_regime: dict | None = None) -> tuple[bool, str, dict]:
    """
    Sniper buy-loop gate.
    - BEAR with high confidence → BLOCK
    - SIDEWAYS → ALLOW with warning
    - BULL → ALLOW
    Returns (allow, reason, regime_data).
    """
    if current_regime is None:
        current_regime = detect_regime()

    regime = current_regime.get("regime", "SIDEWAYS")
    conf = current_regime.get("confidence", 0)

    if regime == "BEAR" and conf > 0.65:
        return False, f"BEAR regime (conf={conf:.0%}) — new buys blocked", current_regime
    if regime == "BEAR":
        return True, f"Weak BEAR signal (conf={conf:.0%}) — proceed with caution", current_regime
    if regime == "SIDEWAYS":
        return True, f"SIDEWAYS regime — normal sizing", current_regime
    return True, f"BULL regime (conf={conf:.0%}) — full speed", current_regime


def get_regime_sizing(base_amount: float, regime_data: dict | None = None) -> float:
    """Apply regime multiplier to position size."""
    if regime_data is None:
        regime_data = detect_regime()
    mult = regime_data.get("sizing_multiplier", 1.0)
    return round(base_amount * mult, 2)


def save_regime_state(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REGIME_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_ist() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        pass
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


# Import here to avoid circular
from datetime import datetime
