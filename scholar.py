import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

REGIME_PATH = "market_regime.json"

def analyze_market_regime(symbol="BTC-USD"):
    df = yf.download(symbol, period="30d", interval="1h", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.dropna()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema_fast = close.ewm(span=14, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_pct = float((tr.rolling(14).mean().iloc[-1] / close.iloc[-1]) * 100)

    # Trend strength approximation (ADX proxy)
    directional_spread = float(abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / close.iloc[-1] * 1000)
    adx_score = round(min(55.0, max(10.0, directional_spread * 8.5)), 2)
    is_chop = adx_score < 20.0

    regime_report = {
        "symbol": symbol,
        "last_price": round(float(close.iloc[-1]), 2),
        "vol_regime": "HIGH_VOL" if atr_pct > 1.8 else ("LOW_VOL" if atr_pct < 0.6 else "NORMAL_VOL"),
        "trend": "CHOP_SIDEWAYS" if is_chop else ("BULL_TREND" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "BEAR_TREND"),
        "adx_score": adx_score,
        "is_chop": is_chop,
        "allow_sniper_execution": not is_chop,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    with open(REGIME_PATH, "w") as f:
        json.dump(regime_report, f, indent=2)
    print(f"[Aegis Scholar] Regime Classification: {regime_report}")
    return regime_report

if __name__ == "__main__":
    analyze_market_regime()
