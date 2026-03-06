"""
====================================================
PROJECT AEGIS — Sector Rotation Detector v1.0
====================================================
Tracks which NSE sectors are rotating in/out of favor.
Uses sector indices (Nifty Bank, Nifty IT, Nifty Metal, etc.)
to detect momentum shifts and auto-adjust stock preferences.

Key features:
  - Sector momentum scoring (5, 10, 20 day returns)
  - Rotation detection (which sectors gaining/losing momentum)
  - Stock-to-sector mapping for Aegis watchlist
  - Recommendations to overweight/underweight sectors

Output: data/sector_rotation.json
====================================================
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "sector_rotation.json")

# ── Sector Index Tickers (Yahoo Finance) ──
SECTOR_INDICES = {
    "Banking":   "^NSEBANK",
    "IT":        "^CNXIT",
    "Metal":     "^CNXMETAL",
    "Energy":    "^CNXENERGY",
    "Infra":     "^CNXINFRA",
    "Pharma":    "^CNXPHARMA",
    "FMCG":      "^CNXFMCG",
    "Auto":      "^CNXAUTO",
    "Realty":     "^CNXREALTY",
    "PSE":       "^CNXPSE",
    "Nifty50":   "^NSEI",       # Benchmark
}

# ── Stock → Sector Mapping for Aegis Watchlist ──
STOCK_SECTOR_MAP = {
    "HDFCBANK.NS":   "Banking",
    "ICICIBANK.NS":  "Banking",
    "SBIN.NS":       "Banking",
    "TCS.NS":        "IT",
    "INFY.NS":       "IT",
    "RELIANCE.NS":   "Energy",
    "TATASTEEL.NS":  "Metal",
    "NTPC.NS":       "Energy",
    "POWERGRID.NS":  "Infra",
    "COALINDIA.NS":  "Metal",
}


def _safe_download(ticker: str, period: str = "60d") -> pd.DataFrame:
    """Download with retry."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if hasattr(df.columns, 'droplevel') and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df is not None and not df.empty:
                return df
        except Exception:
            import time
            time.sleep(1)
    return pd.DataFrame()


def compute_sector_momentum() -> dict:
    """
    Compute momentum scores for each sector.
    Returns dict with per-sector analysis.
    """
    results = {}

    for sector_name, ticker in SECTOR_INDICES.items():
        try:
            df = _safe_download(ticker, "60d")
            if df.empty or len(df) < 20:
                results[sector_name] = {"status": "NO_DATA", "ticker": ticker}
                continue

            close = df["Close"].values.flatten()

            # Returns over different windows
            ret_5d = ((close[-1] / close[-5]) - 1) * 100 if len(close) >= 5 else 0
            ret_10d = ((close[-1] / close[-10]) - 1) * 100 if len(close) >= 10 else 0
            ret_20d = ((close[-1] / close[-20]) - 1) * 100 if len(close) >= 20 else 0

            # Momentum acceleration: is recent momentum stronger than older?
            if len(close) >= 20:
                old_10d = ((close[-10] / close[-20]) - 1) * 100
                new_10d = ret_10d
                acceleration = new_10d - old_10d
            else:
                acceleration = 0

            # Volume trend (is participation increasing?)
            if "Volume" in df.columns and len(df) >= 20:
                vol = df["Volume"].values.flatten()
                vol_recent = np.mean(vol[-5:]) if len(vol) >= 5 else 0
                vol_old = np.mean(vol[-20:-10]) if len(vol) >= 20 else vol_recent
                vol_trend = ((vol_recent / vol_old) - 1) * 100 if vol_old > 0 else 0
            else:
                vol_trend = 0

            # Composite momentum score (weighted)
            score = (ret_5d * 0.4) + (ret_10d * 0.3) + (ret_20d * 0.2) + (acceleration * 0.1)

            # Detect rotation state
            if score > 2.0 and acceleration > 0:
                state = "ROTATING_IN"
            elif score > 0.5:
                state = "STRONG"
            elif score > -0.5:
                state = "NEUTRAL"
            elif score > -2.0:
                state = "WEAKENING"
            else:
                state = "ROTATING_OUT"

            results[sector_name] = {
                "ticker": ticker,
                "price": round(float(close[-1]), 2),
                "ret_5d": round(ret_5d, 2),
                "ret_10d": round(ret_10d, 2),
                "ret_20d": round(ret_20d, 2),
                "acceleration": round(acceleration, 2),
                "vol_trend": round(vol_trend, 1),
                "momentum_score": round(score, 2),
                "state": state,
            }
        except Exception as e:
            results[sector_name] = {"status": "ERROR", "error": str(e)[:60]}

    return results


def get_sector_recommendations(sector_data: dict, watchlist: list = None) -> dict:
    """
    Based on sector momentum, recommend overweight/underweight for each stock.
    
    Returns:
      {
        "overweight": ["TATASTEEL.NS", ...],
        "neutral": [...],
        "underweight": [...],
        "sector_ranking": [("Metal", 3.5), ("Banking", 1.2), ...],
        "stock_adjustments": {"TATASTEEL.NS": +0.10, "SBIN.NS": -0.05, ...}
      }
    """
    if watchlist is None:
        watchlist = list(STOCK_SECTOR_MAP.keys())

    # Rank sectors by momentum
    sector_scores = {}
    for sec, data in sector_data.items():
        if isinstance(data, dict) and "momentum_score" in data:
            sector_scores[sec] = data["momentum_score"]

    sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

    # Determine threshold
    scores = [s for _, s in sorted_sectors]
    avg_score = np.mean(scores) if scores else 0

    overweight = []
    neutral = []
    underweight = []
    stock_adjustments = {}

    for sym in watchlist:
        sector = STOCK_SECTOR_MAP.get(sym, "Unknown")
        sec_score = sector_scores.get(sector, 0)

        if sec_score > avg_score + 1.0:
            overweight.append(sym)
            stock_adjustments[sym] = round(min(0.15, sec_score * 0.03), 3)
        elif sec_score < avg_score - 1.0:
            underweight.append(sym)
            stock_adjustments[sym] = round(max(-0.15, sec_score * 0.03), 3)
        else:
            neutral.append(sym)
            stock_adjustments[sym] = 0.0

    return {
        "overweight": overweight,
        "neutral": neutral,
        "underweight": underweight,
        "sector_ranking": sorted_sectors,
        "stock_adjustments": stock_adjustments,
        "avg_momentum": round(avg_score, 2),
    }


def analyse_sector_rotation(watchlist: list = None) -> dict:
    """
    Full sector rotation analysis. Saves to JSON and returns results.
    """
    print("[SECTOR] Computing sector momentum...")
    sector_data = compute_sector_momentum()

    print("[SECTOR] Generating stock recommendations...")
    recommendations = get_sector_recommendations(sector_data, watchlist)

    result = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "sectors": sector_data,
        "recommendations": recommendations,
        "stock_sector_map": STOCK_SECTOR_MAP,
    }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary
    print(f"\n[SECTOR] Rotation Analysis Complete:")
    for sec, data in sorted(sector_data.items(),
                            key=lambda x: x[1].get("momentum_score", 0) if isinstance(x[1], dict) else -999,
                            reverse=True):
        if isinstance(data, dict) and "momentum_score" in data:
            icon = "🟢" if data["state"] in ("ROTATING_IN", "STRONG") else ("🔴" if data["state"] == "ROTATING_OUT" else "🟡")
            print(f"  {icon} {sec:12} | Score: {data['momentum_score']:+.2f} | 5d: {data['ret_5d']:+.2f}% | State: {data['state']}")

    ow = recommendations["overweight"]
    uw = recommendations["underweight"]
    if ow:
        print(f"  ⬆️  Overweight:  {', '.join(s.replace('.NS','') for s in ow)}")
    if uw:
        print(f"  ⬇️  Underweight: {', '.join(s.replace('.NS','') for s in uw)}")

    return result


if __name__ == "__main__":
    analyse_sector_rotation()
