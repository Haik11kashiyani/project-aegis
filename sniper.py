import json
import os
from datetime import datetime, timezone
from learner import AegisRLAgent

REGIME_PATH = "market_regime.json"
PARAMS_PATH = "best_params.json"
DASHBOARD_STATE_PATH = "dashboard_state.json"

def run_rl_sniper():
    agent = AegisRLAgent()

    regime_data = {"vol_regime": "NORMAL_VOL", "trend": "BULL_TREND", "adx_score": 26.5, "is_chop": False}
    if os.path.exists(REGIME_PATH):
        with open(REGIME_PATH, "r") as f:
            regime_data = json.load(f)

    params = {"fast_ema": 14, "slow_ema": 50, "rsi_period": 14, "rsi_buy": 60}
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            params = json.load(f)

    state_key = agent.get_state_key(
        regime_data.get("vol_regime", "NORMAL_VOL"),
        regime_data.get("trend", "SIDEWAYS"),
        "RSI_NEUTRAL"
    )
    decision = agent.choose_action(state_key, is_chop=regime_data.get("is_chop", False))

    telemetry = {
        "state_key": state_key,
        "decision": decision,
        "regime": regime_data,
        "active_params": params,
        "execution_mode": "LIVE_BROKER" if os.getenv("BROKER_API_KEY") else "PAPER_SAFE_SIMULATION",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    with open(DASHBOARD_STATE_PATH, "w") as f:
        json.dump(telemetry, f, indent=2)

    print(f"[Aegis RL Sniper] Decision: {decision['action']} (Q={decision['q_value']}) | Mode: {telemetry['execution_mode']}")

if __name__ == "__main__":
    run_rl_sniper()
