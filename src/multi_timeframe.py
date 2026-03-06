"""
====================================================
PROJECT AEGIS — Multi-Timeframe Consensus (Phase 8)
====================================================
Adds weekly & monthly direction signals alongside the
existing daily LSTM / RF / XGB models. All 3 timeframes
must agree for high-conviction BUY signals.

Approach:
 ● Download weekly & monthly OHLCV via yfinance
 ● Compute trend direction using SMA crossovers + MACD
 ● Each timeframe votes: BULLISH / NEUTRAL / BEARISH
 ● Consensus requires ≥2 out of 3 timeframes BULLISH
 ● Save state to data/multi_timeframe.json for dashboard

This is a lightweight indicator-based approach rather than
training separate weekly/monthly LSTMs (those can be added
later as Phase 9 enhancement).
====================================================
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("aegis.multi_timeframe")

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_MTF = os.path.join(DATA, "multi_timeframe.json")

# ── Cache to avoid hammering yfinance ──
_tf_cache: Dict[str, Dict] = {}
_cache_ttl = 900  # 15 minutes


# ══════════════════════════════════════════════════
#  TIMEFRAME ANALYSIS
# ══════════════════════════════════════════════════
def _compute_trend(df: pd.DataFrame) -> Dict:
    """
    Compute trend direction from OHLCV DataFrame.
    Returns dict with direction, strength, indicators.
    """
    if df is None or len(df) < 30:
        return {"direction": "NEUTRAL", "strength": 0.0, "reason": "Insufficient data"}

    close = df["Close"].astype(float)

    # SMA crossovers
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(20).mean()

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Latest values
    cur_close = float(close.iloc[-1])
    cur_sma20 = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else cur_close
    cur_sma50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else cur_close
    cur_macd = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0
    cur_signal = float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0
    cur_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

    # Score each indicator: +1 bullish, -1 bearish, 0 neutral
    scores = []
    reasons = []

    # SMA crossover
    if cur_close > cur_sma20 > cur_sma50:
        scores.append(1)
        reasons.append("Price > SMA20 > SMA50")
    elif cur_close < cur_sma20 < cur_sma50:
        scores.append(-1)
        reasons.append("Price < SMA20 < SMA50")
    else:
        scores.append(0)
        reasons.append("SMA mixed")

    # MACD
    if cur_macd > cur_signal and cur_macd > 0:
        scores.append(1)
        reasons.append("MACD bullish crossover")
    elif cur_macd < cur_signal and cur_macd < 0:
        scores.append(-1)
        reasons.append("MACD bearish crossover")
    else:
        scores.append(0)
        reasons.append("MACD neutral")

    # RSI
    if cur_rsi > 55:
        scores.append(1)
        reasons.append(f"RSI {cur_rsi:.0f} bullish")
    elif cur_rsi < 45:
        scores.append(-1)
        reasons.append(f"RSI {cur_rsi:.0f} bearish")
    else:
        scores.append(0)
        reasons.append(f"RSI {cur_rsi:.0f} neutral")

    # Momentum (10-period return)
    if len(close) >= 10:
        momentum = (cur_close - float(close.iloc[-10])) / float(close.iloc[-10]) * 100
        if momentum > 2:
            scores.append(1)
            reasons.append(f"Momentum +{momentum:.1f}%")
        elif momentum < -2:
            scores.append(-1)
            reasons.append(f"Momentum {momentum:.1f}%")
        else:
            scores.append(0)
            reasons.append(f"Momentum flat {momentum:.1f}%")

    total = sum(scores)
    max_score = len(scores)
    strength = abs(total) / max_score if max_score > 0 else 0

    if total >= 2:
        direction = "BULLISH"
    elif total <= -2:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "strength": round(strength, 3),
        "score": total,
        "max_score": max_score,
        "indicators": {
            "close": round(cur_close, 2),
            "sma_20": round(cur_sma20, 2),
            "sma_50": round(cur_sma50, 2),
            "macd": round(cur_macd, 4),
            "macd_signal": round(cur_signal, 4),
            "rsi": round(cur_rsi, 2),
        },
        "reasons": reasons,
    }


def _download_tf_data(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Download data for a specific timeframe."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.warning(f"Failed to download {interval} data for {symbol}: {e}")
    return None


# ══════════════════════════════════════════════════
#  MULTI-TIMEFRAME ANALYSIS (per stock)
# ══════════════════════════════════════════════════
def analyse_multi_timeframe(symbol: str, daily_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Analyse 3 timeframes (daily, weekly, monthly) for a stock.
    Returns consensus dict.
    """
    now = time.time()
    cache_key = symbol
    if cache_key in _tf_cache and (now - _tf_cache[cache_key].get("_ts", 0)) < _cache_ttl:
        return _tf_cache[cache_key]

    result = {
        "symbol": symbol,
        "timestamp": datetime.now(IST).isoformat(),
        "timeframes": {},
        "consensus": "NEUTRAL",
        "conviction": 0.0,
        "should_trade": True,
        "reason": "",
    }

    # ── Daily (use provided df if available) ──
    if daily_df is not None and len(daily_df) >= 30:
        result["timeframes"]["daily"] = _compute_trend(daily_df)
    else:
        df_d = _download_tf_data(symbol, "6mo", "1d")
        result["timeframes"]["daily"] = _compute_trend(df_d) if df_d is not None else {
            "direction": "NEUTRAL", "strength": 0, "reason": "No data"
        }

    # ── Weekly ──
    df_w = _download_tf_data(symbol, "2y", "1wk")
    result["timeframes"]["weekly"] = _compute_trend(df_w) if df_w is not None else {
        "direction": "NEUTRAL", "strength": 0, "reason": "No data"
    }

    # ── Monthly ──
    df_m = _download_tf_data(symbol, "5y", "1mo")
    result["timeframes"]["monthly"] = _compute_trend(df_m) if df_m is not None else {
        "direction": "NEUTRAL", "strength": 0, "reason": "No data"
    }

    # ── Consensus ──
    directions = {tf: info.get("direction", "NEUTRAL")
                  for tf, info in result["timeframes"].items()}

    bullish_count = sum(1 for d in directions.values() if d == "BULLISH")
    bearish_count = sum(1 for d in directions.values() if d == "BEARISH")

    if bullish_count >= 2:
        result["consensus"] = "BULLISH"
        result["conviction"] = bullish_count / 3.0
        result["should_trade"] = True
        result["reason"] = f"{bullish_count}/3 timeframes BULLISH"
    elif bearish_count >= 2:
        result["consensus"] = "BEARISH"
        result["conviction"] = bearish_count / 3.0
        result["should_trade"] = False
        result["reason"] = f"{bearish_count}/3 timeframes BEARISH — avoid longs"
    else:
        result["consensus"] = "MIXED"
        result["conviction"] = 0.33
        result["should_trade"] = True  # Allow but with lower conviction
        result["reason"] = "Timeframes disagree — trade with caution"

    # Cache
    result["_ts"] = now
    _tf_cache[cache_key] = result
    return result


# ══════════════════════════════════════════════════
#  SNIPER GATE — Buy-loop check
# ══════════════════════════════════════════════════
def check_multi_timeframe_gate(
    symbol: str,
    daily_df: Optional[pd.DataFrame] = None,
    require_consensus: bool = True,
) -> Tuple[bool, str, Dict]:
    """
    Gate check for sniper buy loop.
    Returns (allow, reason, analysis_dict).
    """
    try:
        analysis = analyse_multi_timeframe(symbol, daily_df)
    except Exception as e:
        logger.warning(f"[MTF] Analysis failed for {symbol}: {e}")
        return True, f"MTF analysis failed ({e}), allowing trade", {}

    consensus = analysis.get("consensus", "NEUTRAL")

    if consensus == "BEARISH" and require_consensus:
        return False, analysis["reason"], analysis

    if consensus == "MIXED":
        return True, f"MTF mixed — proceed with caution ({analysis['reason']})", analysis

    return True, f"MTF {consensus} — {analysis['reason']}", analysis


# ══════════════════════════════════════════════════
#  BATCH ANALYSIS (all stocks)
# ══════════════════════════════════════════════════
def analyse_all_stocks(symbols: list, df_map: Optional[Dict] = None) -> Dict:
    """Run multi-timeframe analysis on a list of stocks."""
    results = {
        "timestamp": datetime.now(IST).isoformat(),
        "stocks": {},
        "summary": {"bullish": 0, "bearish": 0, "mixed": 0, "neutral": 0},
    }

    for sym in symbols:
        df = df_map.get(sym) if df_map else None
        analysis = analyse_multi_timeframe(sym, df)
        results["stocks"][sym] = analysis
        consensus = analysis.get("consensus", "NEUTRAL").lower()
        if consensus in results["summary"]:
            results["summary"][consensus] += 1

    return results


# ══════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════
def save_mtf_state(data: Dict):
    """Save multi-timeframe analysis to JSON."""
    try:
        os.makedirs(DATA, exist_ok=True)
        # Remove internal cache timestamps
        clean = {}
        for k, v in data.items():
            if isinstance(v, dict):
                clean[k] = {kk: vv for kk, vv in v.items() if kk != "_ts"}
            else:
                clean[k] = v
        with open(FILE_MTF, "w") as f:
            json.dump(clean, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save MTF state: {e}")


def load_mtf_state() -> Dict:
    """Load latest multi-timeframe state."""
    try:
        if os.path.exists(FILE_MTF):
            with open(FILE_MTF) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    import yfinance as yf

    test_symbols = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS"]
    print("═" * 60)
    print("  MULTI-TIMEFRAME CONSENSUS — Test")
    print("═" * 60)

    for sym in test_symbols:
        result = analyse_multi_timeframe(sym)
        print(f"\n  {sym}:")
        for tf, info in result["timeframes"].items():
            print(f"    {tf:8s}: {info['direction']:8s} (strength {info.get('strength', 0):.2f})")
            for r in info.get("reasons", []):
                print(f"              → {r}")
        print(f"    CONSENSUS: {result['consensus']} | Conviction: {result['conviction']:.2f}")
        print(f"    Trade OK: {result['should_trade']} — {result['reason']}")
