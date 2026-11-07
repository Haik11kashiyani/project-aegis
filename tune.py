import json
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

CONFIG_PATH = "best_params.json"

def fetch_clean_data(symbol="BTC-USD", period="180d", interval="1h") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required.")
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"Empty DataFrame returned for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

def evaluate_slice(df: pd.DataFrame, fast_ema: int, slow_ema: int, rsi_period: int, rsi_buy: int, fee_pct: float = 0.0015):
    data = df.copy()
    data["ema_fast"] = data["Close"].ewm(span=fast_ema, adjust=False).mean()
    data["ema_slow"] = data["Close"].ewm(span=slow_ema, adjust=False).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    data["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # Shift position by +1 bar to eliminate lookahead overfitting
    long_cond = (data["ema_fast"] > data["ema_slow"]) & (data["rsi"] < rsi_buy)
    data["position"] = pd.Series(np.where(long_cond, 1.0, 0.0), index=data.index).shift(1).fillna(0.0)

    trades = data["position"].diff().abs().fillna(0.0)
    trade_count = int((trades > 0).sum() // 2)
    bar_returns = data["Close"].pct_change().fillna(0.0)
    strat_returns = (data["position"] * bar_returns) - (trades * fee_pct)

    if strat_returns.std() == 0 or trade_count < 8:
        return {"sharpe": -99.0, "max_dd": 1.0, "trades": trade_count}

    sharpe = (strat_returns.mean() / (strat_returns.std() + 1e-9)) * math.sqrt(252 * 24)
    cum = (1 + strat_returns).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return {"sharpe": float(sharpe), "max_dd": abs(max_dd), "trades": trade_count}

def walk_forward_anti_overfit_tuner():
    df = fetch_clean_data()
    n = len(df)
    purge_bars = 24
    split_idx = int(n * 0.70)

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx + purge_bars:]

    best_candidate = None
    best_objective = -999.0
    rejected_overfit = 0

    for fast in [9, 14, 21]:
        for slow in [35, 50, 80]:
            if fast >= slow:
                continue
            for rsi_p in [10, 14]:
                for rsi_b in [55, 60, 65]:
                    ins = evaluate_slice(train_df, fast, slow, rsi_p, rsi_b)
                    oos = evaluate_slice(val_df, fast, slow, rsi_p, rsi_b)

                    if ins["trades"] < 12 or oos["trades"] < 6:
                        continue
                    # Hard Anti-Overfit Rule: Reject if Out-of-Sample Sharpe drops > 35%
                    if ins["sharpe"] <= 0 or oos["sharpe"] < (ins["sharpe"] * 0.65):
                        rejected_overfit += 1
                        continue

                    gap_penalty = abs(ins["sharpe"] - oos["sharpe"])
                    objective = oos["sharpe"] - (1.8 * oos["max_dd"]) - (0.4 * gap_penalty)

                    if objective > best_objective:
                        best_objective = objective
                        best_candidate = {
                            "fast_ema": fast,
                            "slow_ema": slow,
                            "rsi_period": rsi_p,
                            "rsi_buy": rsi_b,
                            "in_sample_sharpe": round(ins["sharpe"], 2),
                            "out_of_sample_sharpe": round(oos["sharpe"], 2),
                            "rejected_overfit_combinations": rejected_overfit,
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }

    if best_candidate is None:
        best_candidate = {
            "fast_ema": 14, "slow_ema": 50, "rsi_period": 14, "rsi_buy": 60,
            "status": "FALLBACK_CONSERVATIVE_DEFENSE",
            "rejected_overfit_combinations": rejected_overfit,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    with open(CONFIG_PATH, "w") as f:
        json.dump(best_candidate, f, indent=2)
    print(f"[Aegis Tuner] Saved non-overfitted parameters: {best_candidate}")

if __name__ == "__main__":
    walk_forward_anti_overfit_tuner()
