"""
====================================================
PROJECT AEGIS — Intraday Time Pattern Analyser v1.0
====================================================
ML model for time-of-day effects in Indian markets:
  - 9:15-9:30 Opening gap fills
  - 12:00-13:00 Lunch dip
  - 14:30-15:10 Power hour trend continuation
  - Pre/post event volatility windows

Learns from historical intraday data to optimise
entry timing — when to fire bullets for maximum edge.

Output: data/intraday_patterns.json
====================================================
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dtime
import pytz

IST = pytz.timezone("Asia/Kolkata")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "intraday_patterns.json")

# ── Time Windows (IST) ──
TIME_WINDOWS = {
    "OPENING_RUSH":   {"start": dtime(9, 15),  "end": dtime(9, 45),  "desc": "Opening gap & volatility"},
    "MORNING_TREND":  {"start": dtime(9, 45),  "end": dtime(11, 30), "desc": "Institutional flow sets trend"},
    "LUNCH_DRIFT":    {"start": dtime(11, 30), "end": dtime(13, 0),  "desc": "Low volume, range-bound"},
    "AFTERNOON":      {"start": dtime(13, 0),  "end": dtime(14, 30), "desc": "Fresh positioning begins"},
    "POWER_HOUR":     {"start": dtime(14, 30), "end": dtime(15, 15), "desc": "High volume close rush"},
}


def _download_intraday(symbol: str, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    """Download intraday data."""
    for attempt in range(2):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if hasattr(df.columns, 'droplevel') and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df is not None and not df.empty:
                return df
        except Exception:
            import time
            time.sleep(1)
    return pd.DataFrame()


def _classify_window(t: dtime) -> str:
    """Classify a time into one of the trading windows."""
    for name, window in TIME_WINDOWS.items():
        if window["start"] <= t < window["end"]:
            return name
    return "OTHER"


def analyse_time_patterns(symbols: list = None) -> dict:
    """
    Analyse intraday time patterns for the given stocks.
    
    For each time window, compute:
      - Average return
      - Win rate (candle close > open)
      - Volume profile
      - Volatility (std of returns)
    """
    if symbols is None:
        from config import STOCK_WATCHLIST
        symbols = STOCK_WATCHLIST

    # Aggregate across all stocks
    window_stats = {name: {
        "returns": [], "volumes": [], "wins": 0, "total": 0,
        "high_vol_entries": 0,
    } for name in TIME_WINDOWS}

    per_stock = {}

    for sym in symbols:
        print(f"  [PATTERN] Analysing {sym}...")
        df = _download_intraday(sym, "60d", "15m")
        if df.empty:
            continue

        df = df.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)

        df["Return"] = df["Close"].pct_change() * 100
        df["Time"] = df.index.time
        df["Window"] = df["Time"].apply(_classify_window)

        stock_windows = {}
        for name in TIME_WINDOWS:
            subset = df[df["Window"] == name]
            if subset.empty:
                continue

            rets = subset["Return"].dropna().values
            vols = subset["Volume"].values if "Volume" in subset else []
            wins = (subset["Close"] > subset["Open"]).sum()
            total = len(subset)

            window_stats[name]["returns"].extend(rets.tolist())
            window_stats[name]["volumes"].extend(vols.tolist() if len(vols) > 0 else [])
            window_stats[name]["wins"] += int(wins)
            window_stats[name]["total"] += total

            avg_ret = float(np.mean(rets)) if len(rets) > 0 else 0
            win_rate = (wins / total * 100) if total > 0 else 0
            volatility = float(np.std(rets)) if len(rets) > 1 else 0

            stock_windows[name] = {
                "avg_return": round(avg_ret, 4),
                "win_rate": round(win_rate, 1),
                "volatility": round(volatility, 4),
                "candles": total,
            }

        per_stock[sym] = stock_windows

    # Aggregate summary
    summary = {}
    for name, stats in window_stats.items():
        rets = stats["returns"]
        avg_ret = float(np.mean(rets)) if rets else 0
        volatility = float(np.std(rets)) if len(rets) > 1 else 0
        win_rate = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
        avg_vol = float(np.mean(stats["volumes"])) if stats["volumes"] else 0

        # Scoring: high win_rate + positive return + reasonable vol = good entry window
        score = (win_rate - 50) * 0.5 + avg_ret * 20 - volatility * 5
        score = max(-100, min(100, score))

        if score > 15:
            recommendation = "STRONG_ENTRY"
        elif score > 5:
            recommendation = "GOOD_ENTRY"
        elif score > -5:
            recommendation = "NEUTRAL"
        elif score > -15:
            recommendation = "WEAK_ENTRY"
        else:
            recommendation = "AVOID"

        summary[name] = {
            "description": TIME_WINDOWS[name]["desc"],
            "time_range": f"{TIME_WINDOWS[name]['start'].strftime('%H:%M')}-{TIME_WINDOWS[name]['end'].strftime('%H:%M')}",
            "avg_return": round(avg_ret, 4),
            "win_rate": round(win_rate, 1),
            "volatility": round(volatility, 4),
            "avg_volume": round(avg_vol, 0),
            "candles_analysed": stats["total"],
            "entry_score": round(score, 1),
            "recommendation": recommendation,
        }

    # Best time to trade?
    best_window = max(summary.items(), key=lambda x: x[1]["entry_score"]) if summary else ("NONE", {})

    result = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "per_stock": per_stock,
        "best_window": best_window[0],
        "best_window_detail": best_window[1] if isinstance(best_window[1], dict) else {},
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Print
    print(f"\n[PATTERNS] Intraday Time Pattern Analysis:")
    for name, data in sorted(summary.items(), key=lambda x: x[1]["entry_score"], reverse=True):
        icon = "🟢" if data["recommendation"] in ("STRONG_ENTRY", "GOOD_ENTRY") else ("🔴" if data["recommendation"] == "AVOID" else "🟡")
        print(f"  {icon} {name:16} | {data['time_range']} | WR: {data['win_rate']:.0f}% | "
              f"Ret: {data['avg_return']:+.4f}% | Score: {data['entry_score']:+.1f} | {data['recommendation']}")
    print(f"  ⏰ Best Entry Window: {best_window[0]}")

    return result


def get_current_window() -> dict:
    """Get the current time window and its characteristics."""
    now = datetime.now(IST).time()
    window = _classify_window(now)

    try:
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            if window in summary:
                return {
                    "current_window": window,
                    **summary[window],
                }
    except Exception:
        pass

    return {
        "current_window": window,
        "recommendation": "NEUTRAL",
        "entry_score": 0,
    }


def should_delay_entry() -> tuple:
    """
    Check if current time window suggests delaying entry.
    Returns (should_delay: bool, reason: str, optimal_window: str)
    """
    current = get_current_window()
    rec = current.get("recommendation", "NEUTRAL")

    if rec in ("AVOID", "WEAK_ENTRY"):
        try:
            with open(OUTPUT_FILE, "r") as f:
                data = json.load(f)
            best = data.get("best_window", "POWER_HOUR")
            return True, f"Current window ({current['current_window']}) is {rec}. Best: {best}", best
        except Exception:
            pass
        return True, f"Current window ({current['current_window']}) is {rec}", "POWER_HOUR"

    return False, f"Current window ({current['current_window']}) is {rec} — OK to trade", current["current_window"]


if __name__ == "__main__":
    analyse_time_patterns()
