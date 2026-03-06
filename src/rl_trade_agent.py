"""
====================================================
🤖 PROJECT AEGIS — Reinforcement Learning Trade Agent
====================================================
Phase 14 · Fully Dynamic

Deep Q-Network (DQN) that learns optimal entry/exit timing from
REAL P&L reward signals. Uses experience replay, target network,
epsilon-greedy exploration, and trains on actual market data.

State features (all computed from real data):
  - RSI (14), MACD histogram, EMA ratio (12/50)
  - ATR % of price, Bollinger %B, Volume ratio
  - Drawdown from recent peak, Returns (5d, 20d)
  - Position flag (in/out), Unrealised PnL %
  - Regime code, Hour of day (normalised)

Actions: HOLD(0), BUY(1), SELL(2)

Reward: Realised P&L % on SELL, small penalty for HOLD with
open position (encourages timely exits), small penalty for
invalid actions (buying when already in, selling when out).

====================================================
"""

import os, json, time, math, random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None

# ── Paths ────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "rl_trade_agent_state.json")
WEIGHTS_FILE = os.path.join(DATA_DIR, "rl_trade_agent_weights.npz")

# ── DQN Hyperparameters ─────────────────────────
STATE_DIM       = 13          # Number of state features
N_ACTIONS       = 3           # HOLD, BUY, SELL
HIDDEN_1        = 128         # First hidden layer size
HIDDEN_2        = 64          # Second hidden layer size
GAMMA           = 0.97        # Discount factor
LR              = 0.0005      # Learning rate
EPSILON_START   = 1.0         # Initial exploration rate
EPSILON_END     = 0.05        # Minimum exploration rate
EPSILON_DECAY   = 0.998       # Decay per training step
BATCH_SIZE      = 64          # Mini-batch size for replay
MEMORY_SIZE     = 10000       # Experience replay buffer size
TARGET_UPDATE   = 50          # Steps between target network syncs
HOLD_PENALTY    = -0.0005     # Small penalty for holding open position
INVALID_PENALTY = -0.005      # Penalty for invalid actions

ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ═══════════════════════════════════════════════════
#  SIMPLE NEURAL NETWORK (NumPy — no extra deps)
# ═══════════════════════════════════════════════════

def _relu(x):
    return np.maximum(0, x)

def _relu_deriv(x):
    return (x > 0).astype(float)

class DQNetwork:
    """Simple 2-hidden-layer Q-Network implemented in pure NumPy."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS,
                 h1: int = HIDDEN_1, h2: int = HIDDEN_2, lr: float = LR):
        self.lr = lr
        # Xavier initialisation
        self.W1 = np.random.randn(state_dim, h1) * math.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * math.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, n_actions) * math.sqrt(2.0 / h2)
        self.b3 = np.zeros(n_actions)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x shape: (batch, state_dim) or (state_dim,)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._z1 = x @ self.W1 + self.b1
        self._a1 = _relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = _relu(self._z2)
        self._q = self._a2 @ self.W3 + self.b3
        self._input = x
        return self._q

    def train_batch(self, states: np.ndarray, targets: np.ndarray):
        """One gradient descent step on MSE loss (Q-learning targets)."""
        q_pred = self.forward(states)
        batch = states.shape[0]
        # dL/dQ
        dq = (2.0 / batch) * (q_pred - targets)

        # Layer 3
        dW3 = self._a2.T @ dq
        db3 = dq.sum(axis=0)

        # Layer 2
        da2 = dq @ self.W3.T
        dz2 = da2 * _relu_deriv(self._z2)
        dW2 = self._a1.T @ dz2
        db2 = dz2.sum(axis=0)

        # Layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * _relu_deriv(self._z1)
        dW1 = self._input.T @ dz1
        db1 = dz1.sum(axis=0)

        # Gradient clipping
        for g in [dW1, db1, dW2, db2, dW3, db3]:
            np.clip(g, -1.0, 1.0, out=g)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def copy_from(self, other: "DQNetwork"):
        """Copy weights from another network (target network sync)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    def get_weights(self) -> Dict:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
        }

    def set_weights(self, w: Dict):
        self.W1 = w["W1"]
        self.b1 = w["b1"]
        self.W2 = w["W2"]
        self.b2 = w["b2"]
        self.W3 = w["W3"]
        self.b3 = w["b3"]


# ═══════════════════════════════════════════════════
#  STATE EXTRACTION — All from REAL data
# ═══════════════════════════════════════════════════

def extract_state(df: pd.DataFrame, current_price: float,
                  in_position: bool = False, unrealised_pnl: float = 0.0,
                  regime_code: int = 1, hour: int = 12) -> np.ndarray:
    """Extract 13-dimensional state vector from REAL price data."""
    if df is None or len(df) < 30:
        return np.zeros(STATE_DIM)

    close = df["Close"].values.astype(float)

    # RSI (14-period, real)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(pd.Series(gain).rolling(14).mean().iloc[-1])
    avg_loss = float(pd.Series(loss).rolling(14).mean().iloc[-1])
    rsi = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-8)))

    # MACD histogram (real)
    fast = pd.Series(close).ewm(span=12, adjust=False).mean()
    slow = pd.Series(close).ewm(span=26, adjust=False).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = float(macd_line.iloc[-1] - signal.iloc[-1])

    # EMA ratio (real)
    ema12 = float(pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1])
    ema50 = float(pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1])
    ema_ratio = ema12 / max(ema50, 1e-8) - 1.0  # Normalised around 0

    # ATR % (real)
    high = df["High"].values.astype(float) if "High" in df.columns else close
    low = df["Low"].values.astype(float) if "Low" in df.columns else close
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
    atr_pct = atr / max(current_price, 1)

    # Bollinger %B (real)
    bb_mid = float(pd.Series(close).rolling(20).mean().iloc[-1])
    bb_std = float(pd.Series(close).rolling(20).std().iloc[-1])
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct = (current_price - bb_lower) / max(bb_upper - bb_lower, 1e-8)

    # Volume ratio (real)
    if "Volume" in df.columns:
        vol = df["Volume"].values.astype(float)
        vol_ma = float(np.mean(vol[-20:])) if len(vol) >= 20 else float(np.mean(vol))
        vol_ratio = float(vol[-1]) / max(vol_ma, 1)
    else:
        vol_ratio = 1.0

    # Drawdown from recent peak (real)
    peak = float(np.max(close[-60:]))
    dd = (current_price - peak) / max(peak, 1)

    # Returns (real)
    ret_5d = (close[-1] - close[-6]) / max(close[-6], 1) if len(close) > 6 else 0
    ret_20d = (close[-1] - close[-21]) / max(close[-21], 1) if len(close) > 21 else 0

    # Position flag and unrealised PnL
    pos_flag = 1.0 if in_position else 0.0

    # Hour normalised to [0, 1]
    hour_norm = hour / 24.0

    # Regime code normalised
    regime_norm = regime_code / 3.0  # 0=BEAR, 1=SIDEWAYS, 2=BULL, 3=VOLATILE

    state = np.array([
        rsi / 100.0,           # [0, 1]
        np.tanh(macd_hist * 100),  # [-1, 1]
        np.tanh(ema_ratio * 10),   # [-1, 1]
        min(atr_pct * 20, 1.0),    # [0, ~1]
        np.clip(bb_pct, -0.5, 1.5),   # [~-0.5, 1.5]
        min(vol_ratio / 3.0, 1.0),    # [0, 1]
        np.clip(dd * 5, -1, 0),        # [-1, 0]
        np.tanh(ret_5d * 20),     # [-1, 1]
        np.tanh(ret_20d * 10),    # [-1, 1]
        pos_flag,                  # 0 or 1
        np.tanh(unrealised_pnl * 10),  # [-1, 1]
        regime_norm,               # [0, 1]
        hour_norm,                 # [0, 1]
    ], dtype=np.float32)

    return state


# ═══════════════════════════════════════════════════
#  RL TRADE AGENT — DQN with Experience Replay
# ═══════════════════════════════════════════════════

class RLTradeAgent:
    """DQN agent for entry/exit timing. Fully dynamic — learns from real trades."""

    def __init__(self):
        self.policy_net = DQNetwork()
        self.target_net = DQNetwork()
        self.target_net.copy_from(self.policy_net)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.train_steps = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.recent_actions = []   # last 100 actions for stats
        self.trade_history = []    # last 200 trades for dashboard

    def select_action(self, state: np.ndarray, in_position: bool) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        q_values = self.policy_net.forward(state)[0]
        return int(np.argmax(q_values))

    def get_q_values(self, state: np.ndarray) -> Dict:
        """Return Q-values for all actions (for dashboard display)."""
        q = self.policy_net.forward(state)[0]
        return {ACTION_NAMES[i]: round(float(q[i]), 4) for i in range(N_ACTIONS)}

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))

    def train_step(self):
        """Sample mini-batch and do one gradient step."""
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        batch = random.sample(list(self.memory), BATCH_SIZE)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=float)

        # Current Q-values
        q_current = self.policy_net.forward(states)

        # Target Q-values (Double DQN: use policy net to select, target net to evaluate)
        q_next_policy = self.policy_net.forward(next_states)
        q_next_target = self.target_net.forward(next_states)
        best_actions = np.argmax(q_next_policy, axis=1)
        q_next_best = q_next_target[np.arange(BATCH_SIZE), best_actions]

        targets = q_current.copy()
        for i in range(BATCH_SIZE):
            targets[i, actions[i]] = rewards[i] + GAMMA * q_next_best[i] * (1 - dones[i])

        # Train
        self.policy_net.train_batch(states, targets)

        self.train_steps += 1
        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        # Sync target network
        if self.train_steps % TARGET_UPDATE == 0:
            self.target_net.copy_from(self.policy_net)

        # Return loss for logging
        loss = float(np.mean((q_current - targets) ** 2))
        return loss

    def record_trade(self, sym: str, action: str, pnl_pct: float, entry: float, exit_price: float):
        """Record completed trade result."""
        self.total_trades += 1
        self.total_pnl += pnl_pct
        if pnl_pct > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.trade_history.append({
            "symbol": sym,
            "action": action,
            "pnl_pct": round(pnl_pct, 4),
            "entry": entry,
            "exit": exit_price,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.trade_history) > 200:
            self.trade_history = self.trade_history[-200:]

    def log_action(self, sym: str, action: int, q_values: Dict):
        """Log a recent action for dashboard stats."""
        self.recent_actions.append({
            "symbol": sym,
            "action": ACTION_NAMES.get(action, "?"),
            "q_values": q_values,
            "epsilon": round(self.epsilon, 4),
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.recent_actions) > 100:
            self.recent_actions = self.recent_actions[-100:]


# ═══════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════
_agent: Optional[RLTradeAgent] = None


def get_agent() -> RLTradeAgent:
    global _agent
    if _agent is None:
        _agent = RLTradeAgent()
        _try_load_weights()
    return _agent


def _try_load_weights():
    """Load persisted network weights if available."""
    global _agent
    if _agent is None:
        return
    if os.path.exists(WEIGHTS_FILE):
        try:
            data = np.load(WEIGHTS_FILE, allow_pickle=True)
            w = {k: data[k] for k in data.files}
            _agent.policy_net.set_weights(w)
            _agent.target_net.copy_from(_agent.policy_net)
        except Exception:
            pass
    # Load stats from state file
    state = load_agent_state()
    if state:
        _agent.epsilon = state.get("epsilon", EPSILON_START)
        _agent.train_steps = state.get("train_steps", 0)
        _agent.total_trades = state.get("total_trades", 0)
        _agent.wins = state.get("wins", 0)
        _agent.losses = state.get("losses", 0)
        _agent.total_pnl = state.get("total_pnl", 0.0)
        _agent.recent_actions = state.get("recent_actions", [])
        _agent.trade_history = state.get("trade_history", [])


# ═══════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════

def get_rl_action(sym: str, df: pd.DataFrame, current_price: float,
                  in_position: bool = False, unrealised_pnl: float = 0.0,
                  regime_code: int = 1, hour: int = 12) -> Dict:
    """Get the DQN agent's recommended action.
    Returns dict with action, action_name, q_values, confidence."""
    agent = get_agent()
    state = extract_state(df, current_price, in_position, unrealised_pnl, regime_code, hour)
    action = agent.select_action(state, in_position)
    q_vals = agent.get_q_values(state)
    agent.log_action(sym, action, q_vals)

    # Confidence = softmax of Q-values
    q_arr = np.array(list(q_vals.values()))
    q_exp = np.exp(q_arr - np.max(q_arr))
    probs = q_exp / q_exp.sum()
    confidence = float(probs[action])

    return {
        "action": action,
        "action_name": ACTION_NAMES[action],
        "q_values": q_vals,
        "confidence": round(confidence, 4),
        "epsilon": round(agent.epsilon, 4),
        "state_vector": state.tolist(),
    }


def record_rl_experience(sym: str, df: pd.DataFrame, current_price: float,
                         action: int, reward: float, next_df: pd.DataFrame,
                         next_price: float, done: bool,
                         in_position: bool = False, unrealised_pnl: float = 0.0,
                         regime_code: int = 1, hour: int = 12):
    """Record a state transition and train."""
    agent = get_agent()
    state = extract_state(df, current_price, in_position, unrealised_pnl, regime_code, hour)
    next_state = extract_state(next_df, next_price, in_position, unrealised_pnl, regime_code, hour)
    agent.store_experience(state, action, reward, next_state, done)
    agent.train_step()


def batch_train_agent(n_steps: int = 20):
    """Run multiple training steps on existing replay buffer."""
    agent = get_agent()
    total_loss = 0.0
    for _ in range(n_steps):
        loss = agent.train_step()
        total_loss += loss
    return round(total_loss / max(n_steps, 1), 6)


def record_rl_trade(sym: str, action: str, pnl_pct: float, entry: float, exit_price: float):
    """Record a completed trade for tracking."""
    agent = get_agent()
    agent.record_trade(sym, action, pnl_pct, entry, exit_price)


def get_agent_status() -> Dict:
    """Return agent status for dashboard."""
    agent = get_agent()
    win_rate = agent.wins / max(agent.total_trades, 1) * 100
    return {
        "status": "ACTIVE" if agent.train_steps > 0 else "WARMING_UP",
        "epsilon": round(agent.epsilon, 4),
        "train_steps": agent.train_steps,
        "memory_size": len(agent.memory),
        "total_trades": agent.total_trades,
        "wins": agent.wins,
        "losses": agent.losses,
        "win_rate": round(win_rate, 1),
        "total_pnl_pct": round(agent.total_pnl * 100, 2),
        "recent_actions": agent.recent_actions[-20:],
        "trade_history": agent.trade_history[-30:],
        "timestamp": datetime.now().isoformat(),
    }


# ── Persistence ──────────────────────────────────

def save_agent_state():
    os.makedirs(DATA_DIR, exist_ok=True)
    agent = get_agent()

    # Save network weights
    w = agent.policy_net.get_weights()
    np.savez(WEIGHTS_FILE, **w)

    # Save stats
    state = {
        "epsilon": agent.epsilon,
        "train_steps": agent.train_steps,
        "total_trades": agent.total_trades,
        "wins": agent.wins,
        "losses": agent.losses,
        "total_pnl": agent.total_pnl,
        "recent_actions": agent.recent_actions[-100:],
        "trade_history": agent.trade_history[-200:],
        "timestamp": datetime.now().isoformat(),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_agent_state() -> Dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
