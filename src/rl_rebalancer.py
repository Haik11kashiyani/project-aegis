"""
====================================================
Phase 12 — RL Portfolio Rebalancer (PPO / A2C Style)
====================================================
A simple tabular Reinforcement Learning agent that
learns when and how to rebalance portfolio weights.

The agent observes a *discretised state* composed of:
  • Market regime  (bull / bear / sideways)
  • Drawdown bucket (low / med / high)
  • Concentration   (ok / over-concentrated)
  • Days since last rebalance (short / med / long)
  • Breadth signal  (positive / neutral / negative)

It chooses one of four *actions*:
  0  — HOLD  (do nothing)
  1  — LIGHT re-weight (nudge toward equal-weight)
  2  — FULL  re-weight (snap to target weights)
  3  — DEFENSIVE (rotate into lower-vol assets)

Rewards are based on next-day portfolio Sharpe delta.

Key functions used by sniper.py
-------------------------------
* ``get_rl_rebalance_action()``  — recommended action
* ``compute_rl_weights()``       — target weight vector
* ``record_rebalance_outcome()`` — reward feedback
* ``train_rl_rebalancer()``      — batch Q-update
* ``save_rl_rebal_state() / load_rl_rebal_state()``
====================================================
"""

import os, sys, json, math, random, time
from datetime import datetime
from collections import defaultdict
import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "rl_rebalancer_state.json")
QTABLE_FILE = os.path.join(DATA_DIR, "rl_rebal_qtable.json")

# ── Hyper-parameters ─────────────────────────────
ALPHA          = 0.10      # learning rate
GAMMA          = 0.95      # discount factor
EPSILON        = 0.15      # exploration rate
EPSILON_MIN    = 0.03
EPSILON_DECAY  = 0.995
N_ACTIONS      = 4         # HOLD, LIGHT, FULL, DEFENSIVE
REBAL_CD_DAYS  = 3         # min days between rebalances

# ── State discretisers ───────────────────────────
_REGIME_MAP  = {"BULL": 0, "BEAR": 1, "SIDEWAYS": 2}
_DD_BREAKS   = [0.02, 0.05]        # <2 %, 2-5 %, >5 %
_CONC_BREAK  = 0.35                 # Herfindahl > 0.35 ⇒ over
_DAYS_BREAKS = [3, 10]              # short / med / long
_BREADTH_MAP = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}

# Runtime
_qtable: dict = {}
_epsilon: float = EPSILON
_history: list = []


# --------------------------------------------------
#  State encoding
# --------------------------------------------------
def _discretise_dd(dd_pct: float) -> int:
    if dd_pct < _DD_BREAKS[0]:
        return 0
    elif dd_pct < _DD_BREAKS[1]:
        return 1
    return 2


def _discretise_days(d: int) -> int:
    if d < _DAYS_BREAKS[0]:
        return 0
    elif d < _DAYS_BREAKS[1]:
        return 1
    return 2


def _encode_state(
    regime: str = "SIDEWAYS",
    drawdown_pct: float = 0.0,
    herfindahl: float = 0.0,
    days_since_rebal: int = 5,
    breadth_signal: str = "NEUTRAL",
) -> str:
    r = _REGIME_MAP.get(regime.upper(), 2)
    d = _discretise_dd(abs(drawdown_pct))
    c = 1 if herfindahl > _CONC_BREAK else 0
    t = _discretise_days(days_since_rebal)
    b = _BREADTH_MAP.get(breadth_signal.upper(), 1)
    return f"{r}_{d}_{c}_{t}_{b}"


# --------------------------------------------------
#  Q-table helpers
# --------------------------------------------------
def _get_q(state: str) -> list:
    """Return Q-values for all actions in *state*."""
    if state not in _qtable:
        _qtable[state] = [0.0] * N_ACTIONS
    return _qtable[state]


def _pick_action(state: str) -> int:
    """ε-greedy action selection."""
    global _epsilon
    if random.random() < _epsilon:
        return random.randint(0, N_ACTIONS - 1)
    q = _get_q(state)
    return int(np.argmax(q))


def _update_q(state: str, action: int, reward: float, next_state: str):
    """Standard Q-learning update."""
    q = _get_q(state)
    nq = _get_q(next_state)
    q[action] += ALPHA * (reward + GAMMA * max(nq) - q[action])


# --------------------------------------------------
#  Weight computation
# --------------------------------------------------
def _compute_herfindahl(weights: dict) -> float:
    """Herfindahl-Hirschman Index for concentration."""
    if not weights:
        return 0.0
    vals = np.array(list(weights.values()), dtype=float)
    s = vals.sum()
    if s <= 0:
        return 0.0
    normed = vals / s
    return float((normed ** 2).sum())


def compute_rl_weights(
    symbols: list,
    current_weights: dict = None,
    action: int = 0,
    vol_map: dict = None,
) -> dict:
    """
    Produce target portfolio weights given the RL *action*.

    Parameters
    ----------
    symbols : list of ticker strings
    current_weights : dict symbol→float (current allocation fraction)
    action : int (0=HOLD, 1=LIGHT, 2=FULL, 3=DEFENSIVE)
    vol_map : dict symbol→float (realised vol; for DEFENSIVE)

    Returns
    -------
    dict symbol → float (target weight, sums to 1.0)
    """
    n = len(symbols)
    if n == 0:
        return {}

    equal = {s: 1.0 / n for s in symbols}
    cw = current_weights or equal

    if action == 0:  # HOLD
        return cw

    if action == 1:  # LIGHT — half-step toward equal
        out = {}
        for s in symbols:
            out[s] = cw.get(s, 0) * 0.5 + equal[s] * 0.5
        total = sum(out.values()) or 1.0
        return {s: v / total for s, v in out.items()}

    if action == 2:  # FULL — snap to equal weight
        return equal

    if action == 3:  # DEFENSIVE — inverse vol
        if vol_map and any(vol_map.values()):
            inv = {}
            for s in symbols:
                v = vol_map.get(s, 0.01) or 0.01
                inv[s] = 1.0 / v
            total = sum(inv.values()) or 1.0
            return {s: inv[s] / total for s in symbols}
        return equal

    return equal


# --------------------------------------------------
#  Public API for sniper.py
# --------------------------------------------------
def get_rl_rebalance_action(
    symbols: list,
    current_weights: dict = None,
    regime: str = "SIDEWAYS",
    drawdown_pct: float = 0.0,
    days_since_rebal: int = 5,
    breadth_signal: str = "NEUTRAL",
    vol_map: dict = None,
) -> dict:
    """
    Ask the RL agent whether (and how) to rebalance.

    Returns
    -------
    dict with keys: action (int), action_name (str),
    target_weights (dict), state_key, should_rebalance (bool)
    """
    hh = _compute_herfindahl(current_weights or {})
    state_key = _encode_state(regime, drawdown_pct, hh, days_since_rebal, breadth_signal)
    action = _pick_action(state_key)

    names = {0: "HOLD", 1: "LIGHT", 2: "FULL", 3: "DEFENSIVE"}
    target = compute_rl_weights(symbols, current_weights, action, vol_map)

    return {
        "action": action,
        "action_name": names.get(action, "HOLD"),
        "target_weights": target,
        "state_key": state_key,
        "should_rebalance": action != 0,
        "herfindahl": round(hh, 4),
    }


def record_rebalance_outcome(
    state_key: str,
    action: int,
    reward: float,
    next_state_key: str = None,
):
    """Record a reward signal after a rebalance decision."""
    global _epsilon
    ns = next_state_key or state_key
    _update_q(state_key, action, reward, ns)
    _epsilon = max(EPSILON_MIN, _epsilon * EPSILON_DECAY)
    _history.append({
        "state": state_key,
        "action": action,
        "reward": round(reward, 4),
        "next_state": ns,
        "timestamp": datetime.now(IST).isoformat(),
    })
    if len(_history) > 2000:
        _history[:] = _history[-2000:]


def train_rl_rebalancer(episodes: int = 50) -> dict:
    """
    Batch replay training from history.
    """
    if len(_history) < 5:
        return {"trained": False, "reason": "insufficient_history"}

    trained = 0
    for _ in range(episodes):
        sample = random.choice(_history)
        _update_q(
            sample["state"],
            sample["action"],
            sample["reward"],
            sample.get("next_state", sample["state"]),
        )
        trained += 1

    return {
        "trained": True,
        "episodes": trained,
        "states_explored": len(_qtable),
        "epsilon": round(_epsilon, 4),
    }


# --------------------------------------------------
#  Persistence
# --------------------------------------------------
def save_rl_rebal_state(data: dict = None):
    """Save Q-table + history."""
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = data if data is not None else {
        "qtable_size": len(_qtable),
        "epsilon": _epsilon,
        "history_len": len(_history),
        "timestamp": datetime.now(IST).isoformat(),
        "recent_actions": _history[-20:] if _history else [],
    }
    with open(STATE_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    # Q-table separately
    with open(QTABLE_FILE, "w") as f:
        json.dump(_qtable, f, indent=2, default=str)


def load_rl_rebal_state() -> dict:
    """Restore Q-table + state."""
    global _qtable, _epsilon
    if os.path.exists(QTABLE_FILE):
        try:
            with open(QTABLE_FILE, "r") as f:
                _qtable.update(json.load(f))
        except Exception:
            pass
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            _epsilon = data.get("epsilon", EPSILON)
            return data
        except Exception:
            pass
    return {}


def get_rebalancer_status() -> dict:
    """Dashboard-friendly summary."""
    return {
        "qtable_states": len(_qtable),
        "epsilon": round(_epsilon, 4),
        "history_len": len(_history),
        "last_action": _history[-1] if _history else None,
        "timestamp": datetime.now(IST).isoformat(),
    }
