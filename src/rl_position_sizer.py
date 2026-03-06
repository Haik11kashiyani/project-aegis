"""
====================================================
🤖 PROJECT AEGIS — RL Position Sizer (Phase 10)
====================================================
Replaces fixed Kelly/risk-parity with a Q-learning
agent that learns optimal position sizing from
historical trade reward signals.

State:  (regime, volatility_bucket, win_streak_bucket,
         drawdown_bucket, breadth_bucket)
Action: sizing_pct ∈ {2%, 5%, 8%, 12%, 18%, 25%}
Reward: Sharpe-adjusted P&L of the trade

The Q-table is persisted to JSON and updated after
every closed trade.  During live trading the agent
picks the action with highest Q-value for the current
state (ε-greedy with decaying exploration).
====================================================
"""

import os
import sys
import json
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import CAPITAL, MAX_BULLETS, TRADE_LOG_FILE

# ──────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RL_STATE_FILE = os.path.join(DATA, "rl_sizer_state.json")
RL_QTABLE_FILE = os.path.join(DATA, "rl_qtable.json")

# Action space — sizing percentages of capital
ACTIONS = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]
N_ACTIONS = len(ACTIONS)

# Q-learning hyper-parameters
ALPHA = 0.15          # Learning rate
GAMMA = 0.90          # Discount factor
EPSILON_START = 0.30  # Initial exploration
EPSILON_MIN = 0.05    # Minimum exploration
EPSILON_DECAY = 0.998 # Per-episode decay
DEFAULT_Q = 0.0       # Optimistic initialisation


# ──────────────────────────────────────────────────
#  STATE DISCRETISATION
# ──────────────────────────────────────────────────
def _bucket_vol(atr_pct: float) -> str:
    """Bucket annualised vol into LOW / MED / HIGH."""
    if atr_pct < 0.015:
        return "LOW"
    elif atr_pct < 0.03:
        return "MED"
    return "HIGH"


def _bucket_streak(win_streak: int) -> str:
    if win_streak >= 3:
        return "HOT"
    elif win_streak <= -3:
        return "COLD"
    return "NEUTRAL"


def _bucket_drawdown(dd_pct: float) -> str:
    if dd_pct < -0.05:
        return "DEEP"
    elif dd_pct < -0.02:
        return "MILD"
    return "NONE"


def _bucket_breadth(breadth_score: float) -> str:
    if breadth_score > 65:
        return "STRONG"
    elif breadth_score < 35:
        return "WEAK"
    return "NEUTRAL"


def build_state(
    regime: str = "SIDEWAYS",
    atr_pct: float = 0.02,
    win_streak: int = 0,
    drawdown_pct: float = 0.0,
    breadth_score: float = 50.0,
) -> str:
    """
    Encode the 5D state as a hashable string key.
    Example: "BULL|MED|NEUTRAL|NONE|STRONG"
    """
    return "|".join([
        regime.upper(),
        _bucket_vol(atr_pct),
        _bucket_streak(win_streak),
        _bucket_drawdown(drawdown_pct),
        _bucket_breadth(breadth_score),
    ])


# ──────────────────────────────────────────────────
#  Q-TABLE MANAGEMENT
# ──────────────────────────────────────────────────
class QTable:
    """Sparse Q-table backed by JSON file."""

    def __init__(self):
        self.table: Dict[str, List[float]] = {}
        self.epsilon = EPSILON_START
        self.episode_count = 0
        self._load()

    # ── persistence ──
    def _load(self):
        try:
            if os.path.exists(RL_QTABLE_FILE):
                with open(RL_QTABLE_FILE, "r") as f:
                    data = json.load(f)
                self.table = data.get("q", {})
                self.epsilon = data.get("epsilon", EPSILON_START)
                self.episode_count = data.get("episodes", 0)
        except Exception:
            self.table = {}

    def save(self):
        os.makedirs(DATA, exist_ok=True)
        with open(RL_QTABLE_FILE, "w") as f:
            json.dump({
                "q": self.table,
                "epsilon": round(self.epsilon, 6),
                "episodes": self.episode_count,
                "timestamp": datetime.now(IST).isoformat(),
            }, f, indent=2)

    # ── value access ──
    def _ensure(self, state: str):
        if state not in self.table:
            self.table[state] = [DEFAULT_Q] * N_ACTIONS

    def get_values(self, state: str) -> List[float]:
        self._ensure(state)
        return self.table[state]

    # ── action selection ──
    def select_action(self, state: str, training: bool = True) -> int:
        """ε-greedy action selection. Returns action index."""
        self._ensure(state)
        if training and random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        q_vals = self.table[state]
        max_q = max(q_vals)
        # Break ties randomly
        best = [i for i, v in enumerate(q_vals) if abs(v - max_q) < 1e-9]
        return random.choice(best)

    # ── learning ──
    def update(self, state: str, action_idx: int, reward: float,
               next_state: str):
        """Standard Q-learning update."""
        self._ensure(state)
        self._ensure(next_state)
        old_q = self.table[state][action_idx]
        best_next = max(self.table[next_state])
        new_q = old_q + ALPHA * (reward + GAMMA * best_next - old_q)
        self.table[state][action_idx] = round(new_q, 6)
        self.episode_count += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# ──────────────────────────────────────────────────
#  REWARD SHAPING
# ──────────────────────────────────────────────────
def compute_reward(pnl: float, capital: float, sizing_pct: float) -> float:
    """
    Sharpe-adjusted reward so the agent prefers risk-efficient sizing.
    reward = pnl_pct / sizing_pct  (penalise large bets that don't pay)
    Clip to [-3, +3] to prevent Q-value blow-up.
    """
    if sizing_pct <= 0:
        return 0.0
    pnl_pct = pnl / capital if capital > 0 else 0.0
    raw = pnl_pct / sizing_pct
    return max(-3.0, min(3.0, round(raw, 4)))


# ──────────────────────────────────────────────────
#  WIN-STREAK TRACKER
# ──────────────────────────────────────────────────
def _compute_streak(trade_df: pd.DataFrame) -> int:
    """Count consecutive same-direction results from most recent trade."""
    if trade_df.empty or "Actual_Profit" not in trade_df.columns:
        return 0
    closed = trade_df[trade_df["Status"].isin(
        ["TARGET_HIT", "STOP_LOSS", "FORCE_CLOSED", "MOMENTUM_EXIT",
         "RSI_EXIT", "TIME_DECAY", "PARTIAL_EXIT"]
    )].tail(20)
    if closed.empty:
        return 0
    streak = 0
    direction = None
    for pnl in reversed(closed["Actual_Profit"].values):
        pnl_f = float(pnl) if not pd.isna(pnl) else 0.0
        cur_dir = 1 if pnl_f > 0 else -1
        if direction is None:
            direction = cur_dir
        if cur_dir != direction:
            break
        streak += cur_dir
    return streak


# ──────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────
_qtable: Optional[QTable] = None


def _get_qtable() -> QTable:
    global _qtable
    if _qtable is None:
        _qtable = QTable()
    return _qtable


def get_rl_position_size(
    symbol: str,
    price: float,
    capital: float,
    regime: str = "SIDEWAYS",
    atr_pct: float = 0.02,
    drawdown_pct: float = 0.0,
    breadth_score: float = 50.0,
    trade_df: pd.DataFrame = None,
    training: bool = True,
) -> Dict:
    """
    Ask the RL agent for recommended position sizing.

    Returns:
      {
        "amount": float,        # ₹ to allocate
        "pct": float,           # % of capital
        "action_idx": int,
        "state": str,
        "q_values": list,
        "method": "RL_QLEARN",
      }
    """
    qt = _get_qtable()

    # Build state
    streak = 0
    if trade_df is None:
        try:
            if os.path.exists(TRADE_LOG_FILE):
                trade_df = pd.read_csv(TRADE_LOG_FILE)
        except Exception:
            trade_df = pd.DataFrame()
    if trade_df is not None and not trade_df.empty:
        streak = _compute_streak(trade_df)

    state = build_state(regime, atr_pct, streak, drawdown_pct, breadth_score)
    action_idx = qt.select_action(state, training=training)
    sizing_pct = ACTIONS[action_idx]
    amount = capital * sizing_pct

    # Floor & ceiling
    floor = capital * 0.02
    ceil = capital * 0.25
    amount = max(floor, min(ceil, amount))
    sizing_pct = amount / capital

    return {
        "amount": round(amount, 2),
        "pct": round(sizing_pct * 100, 2),
        "action_idx": action_idx,
        "state": state,
        "q_values": [round(v, 4) for v in qt.get_values(state)],
        "method": "RL_QLEARN",
    }


def record_trade_result(
    state: str,
    action_idx: int,
    pnl: float,
    capital: float,
    next_regime: str = "SIDEWAYS",
    next_atr_pct: float = 0.02,
    next_drawdown_pct: float = 0.0,
    next_breadth: float = 50.0,
):
    """
    Call after a trade closes to update the Q-table.
    """
    qt = _get_qtable()
    sizing_pct = ACTIONS[action_idx] if 0 <= action_idx < N_ACTIONS else 0.10
    reward = compute_reward(pnl, capital, sizing_pct)

    # Build next-state
    try:
        trade_df = pd.read_csv(TRADE_LOG_FILE) if os.path.exists(TRADE_LOG_FILE) else pd.DataFrame()
    except Exception:
        trade_df = pd.DataFrame()
    streak = _compute_streak(trade_df)
    next_state = build_state(next_regime, next_atr_pct, streak, next_drawdown_pct, next_breadth)

    qt.update(state, action_idx, reward, next_state)
    qt.save()


def get_rl_stats() -> Dict:
    """Return current RL agent statistics for dashboard."""
    qt = _get_qtable()
    return {
        "states_explored": len(qt.table),
        "total_episodes": qt.episode_count,
        "epsilon": round(qt.epsilon, 4),
        "actions": [f"{a*100:.0f}%" for a in ACTIONS],
        "timestamp": datetime.now(IST).isoformat(),
    }


def save_rl_state(data: dict = None):
    """Save RL state summary for dashboard."""
    os.makedirs(DATA, exist_ok=True)
    if data is None:
        data = get_rl_stats()
    data["timestamp"] = datetime.now(IST).isoformat()
    with open(RL_STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ──────────────────────────────────────────────────
#  OFFLINE BATCH TRAINING FROM HISTORY
# ──────────────────────────────────────────────────
def batch_train(
    regime_default: str = "SIDEWAYS",
    atr_pct_default: float = 0.02,
    breadth_default: float = 50.0,
):
    """
    Replay all closed trades from history and update Q-table
    so the agent starts with some knowledge on first run.
    """
    try:
        if not os.path.exists(TRADE_LOG_FILE):
            return {"status": "no_history"}
        df = pd.read_csv(TRADE_LOG_FILE)
        closed = df[df["Status"].isin(
            ["TARGET_HIT", "STOP_LOSS", "FORCE_CLOSED",
             "MOMENTUM_EXIT", "RSI_EXIT", "TIME_DECAY"]
        )]
        if closed.empty:
            return {"status": "no_closed_trades"}

        qt = _get_qtable()
        trained = 0
        dd_pct = 0.0
        peak_capital = CAPITAL

        for _, row in closed.iterrows():
            pnl = float(row.get("Actual_Profit", 0))
            price = float(row.get("Entry_Price", 0))
            if price <= 0:
                continue

            # Approximate state from row data
            state = build_state(
                regime=regime_default,
                atr_pct=atr_pct_default,
                win_streak=0,
                drawdown_pct=dd_pct,
                breadth_score=breadth_default,
            )

            # Pick random action (exploration replay)
            action_idx = random.randint(0, N_ACTIONS - 1)
            sizing_pct = ACTIONS[action_idx]
            reward = compute_reward(pnl, CAPITAL, sizing_pct)

            # Update running drawdown
            peak_capital = max(peak_capital, CAPITAL + pnl)
            dd_pct = (CAPITAL + pnl - peak_capital) / peak_capital if peak_capital > 0 else 0

            next_state = build_state(
                regime=regime_default,
                atr_pct=atr_pct_default,
                win_streak=0,
                drawdown_pct=dd_pct,
                breadth_score=breadth_default,
            )

            qt.update(state, action_idx, reward, next_state)
            trained += 1

        qt.save()
        return {"status": "ok", "trades_replayed": trained, "states": len(qt.table)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
