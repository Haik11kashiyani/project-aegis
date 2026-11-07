import json
import os
from datetime import datetime, timezone

RL_MODEL_PATH = "rl_qtable.json"
TRADE_LOG_PATH = "trade_history.json"
DASHBOARD_STATE_PATH = "dashboard_state.json"

class AegisRLAgent:
    """
    Contextual Q-Learning Agent with Asymmetric Drawdown Penalties
    and L2 Weight Decay to prevent state-space overfitting.
    """
    def __init__(self, alpha=0.08, gamma=0.90, epsilon=0.10, weight_decay=0.005):
        self.alpha = alpha               # Conservative learning rate
        self.gamma = gamma               # Discount factor
        self.epsilon = epsilon           # Exploration rate
        self.weight_decay = weight_decay # L2 regularization shrinks extreme Q-values
        self.actions = ["HOLD", "LONG", "SHORT"]
        self.q_table = self._load_q_table()

    def _load_q_table(self):
        if os.path.exists(RL_MODEL_PATH):
            try:
                with open(RL_MODEL_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_state_key(self, regime: str, trend: str, rsi_bucket: str) -> str:
        return f"{regime}|{trend}|{rsi_bucket}"

    def ensure_state(self, state_key: str):
        if state_key not in self.q_table:
            self.q_table[state_key] = {"HOLD": 0.25, "LONG": 0.0, "SHORT": 0.0, "visits": 0}

    def compute_shaped_reward(self, action: str, net_pnl_pct: float, max_adverse_excursion_pct: float, is_chop: bool) -> float:
        """
        Asymmetric reward function that rewards skipping choppy markets (+0.25R)
        and penalizes drawdowns 1.5x harder than lucky wins.
        """
        fee_pct = 0.15
        if action == "HOLD":
            return 0.25 if is_chop else -0.05

        adjusted_pnl = net_pnl_pct - fee_pct
        drawdown_penalty = 1.5 * abs(max_adverse_excursion_pct)

        if adjusted_pnl > 0:
            raw_reward = min(adjusted_pnl * 1.2, 4.0) - (0.5 * drawdown_penalty)
        else:
            raw_reward = (adjusted_pnl * 1.5) - drawdown_penalty

        return round(max(min(raw_reward, 5.0), -5.0), 4)

    def update_from_experience(self, state_key: str, action: str, reward: float, next_state_key: str):
        self.ensure_state(state_key)
        self.ensure_state(next_state_key)

        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key][a] for a in self.actions)

        # Bellman update with L2 weight decay to prevent Q-value overfitting
        regularized_q = current_q * (1.0 - self.weight_decay)
        updated_q = regularized_q + self.alpha * (reward + self.gamma * max_next_q - regularized_q)

        self.q_table[state_key][action] = round(updated_q, 4)
        self.q_table[state_key]["visits"] += 1

    def choose_action(self, state_key: str, is_chop: bool) -> dict:
        self.ensure_state(state_key)
        if is_chop:
            return {"action": "HOLD", "q_value": self.q_table[state_key]["HOLD"], "reason": "ADX_CHOP_GUARD"}
        q_vals = self.q_table[state_key]
        best_act = max(self.actions, key=lambda a: q_vals.get(a, 0.0))
        return {"action": best_act, "q_value": q_vals.get(best_act, 0.0), "reason": "RL_OPTIMAL_POLICY"}

    def save(self):
        with open(RL_MODEL_PATH, "w") as f:
            json.dump(self.q_table, f, indent=2)


def run_off_market_learner():
    agent = AegisRLAgent()
    experiences = []
    if os.path.exists(TRADE_LOG_PATH):
        try:
            with open(TRADE_LOG_PATH, "r") as f:
                experiences = json.load(f)
        except Exception:
            experiences = []

    processed = 0
    total_reward = 0.0
    for exp in experiences:
        if exp.get("rl_processed"):
            continue
        state_key = agent.get_state_key(
            exp.get("regime", "NORMAL_VOL"),
            exp.get("trend", "SIDEWAYS"),
            exp.get("rsi_bucket", "RSI_NEUTRAL")
        )
        reward = agent.compute_shaped_reward(
            action=exp.get("action", "HOLD"),
            net_pnl_pct=float(exp.get("pnl_pct", 0.0)),
            max_adverse_excursion_pct=float(exp.get("mae_pct", 0.0)),
            is_chop=bool(exp.get("is_chop", False))
        )
        agent.update_from_experience(state_key, exp.get("action", "HOLD"), reward, state_key)
        exp["rl_processed"] = True
        exp["assigned_reward"] = reward
        processed += 1
        total_reward += reward

    agent.save()
    if experiences:
        with open(TRADE_LOG_PATH, "w") as f:
            json.dump(experiences[-500:], f, indent=2)

    print(f"[Aegis RL Learner] Processed {processed} trades | Shaped Reward: {total_reward:.2f}")

if __name__ == "__main__":
    run_off_market_learner()
