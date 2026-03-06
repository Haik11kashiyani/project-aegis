"""
====================================================
PROJECT AEGIS — Market Breadth Indicator v1.0
====================================================
Tracks the health of the broader market using:
  - Nifty 50 Advance/Decline ratio
  - Nifty 50 breadth (% stocks above SMA50/SMA200)
  - Market-wide momentum & VIX correlation
  - Breadth divergence detection (index up but breadth weak)

This acts as a MACRO FILTER before taking any trade.
If market breadth is unhealthy, reduced position sizing or skip.

Output: data/market_breadth.json
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
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "market_breadth.json")

# ── Nifty 50 constituents (top 30 by weight for speed) ──
NIFTY50_SAMPLE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS", "COALINDIA.NS", "ONGC.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "HCLTECH.NS", "TECHM.NS", "BAJAJFINSV.NS",
]


def _download_stock(symbol: str, period: str = "60d") -> pd.DataFrame:
    """Download with retry."""
    for attempt in range(2):
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            if hasattr(df.columns, 'droplevel') and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df is not None and not df.empty:
                return df
        except Exception:
            import time
            time.sleep(1)
    return pd.DataFrame()


def compute_breadth(stocks: list = None) -> dict:
    """
    Compute market breadth metrics for the given stock universe.
    
    Returns:
    {
        "advance_decline_ratio": 1.5,  # >1 = more advancers
        "pct_above_sma50": 65.0,       # % of stocks above 50 SMA
        "pct_above_sma200": 55.0,      # % of stocks above 200 SMA
        "breadth_score": 72.0,         # 0-100 composite score
        "signal": "HEALTHY",           # HEALTHY | CAUTION | WEAK
        "detail": {...per-stock data...}
    }
    """
    if stocks is None:
        stocks = NIFTY50_SAMPLE

    advancing = 0
    declining = 0
    above_sma50 = 0
    above_sma200 = 0
    total_analysed = 0
    stock_details = {}

    print(f"[BREADTH] Analysing {len(stocks)} stocks...")

    for sym in stocks:
        try:
            df = _download_stock(sym, "250d")
            if df.empty or len(df) < 50:
                continue

            close = df["Close"].values.flatten()
            total_analysed += 1

            # Today's change
            if len(close) >= 2:
                change = ((close[-1] / close[-2]) - 1) * 100
                if change > 0:
                    advancing += 1
                else:
                    declining += 1
            else:
                change = 0

            # SMA checks
            sma50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
            sma200 = np.mean(close[-200:]) if len(close) >= 200 else None

            is_above_50 = close[-1] > sma50
            is_above_200 = (close[-1] > sma200) if sma200 is not None else None

            if is_above_50:
                above_sma50 += 1
            if is_above_200:
                above_sma200 += 1

            stock_details[sym] = {
                "price": round(float(close[-1]), 2),
                "change": round(change, 2),
                "above_sma50": is_above_50,
                "above_sma200": is_above_200,
            }
        except Exception:
            continue

    if total_analysed == 0:
        return {
            "advance_decline_ratio": 1.0,
            "pct_above_sma50": 50.0,
            "pct_above_sma200": 50.0,
            "breadth_score": 50.0,
            "signal": "NO_DATA",
            "stocks_analysed": 0,
        }

    # Compute ratios
    ad_ratio = advancing / max(1, declining)
    pct_sma50 = (above_sma50 / total_analysed) * 100
    # For SMA200, only count stocks that had enough data
    stocks_with_200 = sum(1 for s in stock_details.values() if s.get("above_sma200") is not None)
    pct_sma200 = (above_sma200 / max(1, stocks_with_200)) * 100

    # Composite breadth score (0-100)
    ad_component = min(100, ad_ratio * 40)   # AD ratio contributes up to 40
    sma50_component = pct_sma50 * 0.35       # SMA50 breadth contributes 35
    sma200_component = pct_sma200 * 0.25     # SMA200 breadth contributes 25
    breadth_score = ad_component + sma50_component + sma200_component

    # Signal classification
    if breadth_score >= 65:
        signal = "HEALTHY"
    elif breadth_score >= 45:
        signal = "CAUTION"
    else:
        signal = "WEAK"

    return {
        "advance_decline_ratio": round(ad_ratio, 2),
        "advancing": advancing,
        "declining": declining,
        "pct_above_sma50": round(pct_sma50, 1),
        "pct_above_sma200": round(pct_sma200, 1),
        "breadth_score": round(breadth_score, 1),
        "signal": signal,
        "stocks_analysed": total_analysed,
        "detail": stock_details,
    }


def get_vix_level() -> dict:
    """Get India VIX for fear gauge."""
    try:
        df = _download_stock("^INDIAVIX", "5d")
        if not df.empty:
            vix = float(df["Close"].values.flatten()[-1])
            if vix > 25:
                vix_signal = "HIGH_FEAR"
            elif vix > 18:
                vix_signal = "ELEVATED"
            elif vix > 12:
                vix_signal = "NORMAL"
            else:
                vix_signal = "COMPLACENT"
            return {"vix": round(vix, 2), "signal": vix_signal}
    except Exception:
        pass
    return {"vix": 0, "signal": "UNKNOWN"}


def analyse_market_breadth() -> dict:
    """
    Full market breadth analysis. Saves to JSON.
    
    Returns the breadth data + recommendation for trading:
      position_size_factor: 1.0 (normal), 0.5 (half), 0.0 (no trade)
    """
    breadth = compute_breadth()
    vix = get_vix_level()

    # Position size recommendation
    if breadth["signal"] == "HEALTHY" and vix["signal"] in ("NORMAL", "COMPLACENT"):
        size_factor = 1.0
        recommendation = "Full size — breadth healthy, VIX calm"
    elif breadth["signal"] == "HEALTHY" and vix["signal"] == "ELEVATED":
        size_factor = 0.75
        recommendation = "75% size — breadth ok but VIX elevated"
    elif breadth["signal"] == "CAUTION":
        size_factor = 0.50
        recommendation = "50% size — breadth cautious, be selective"
    elif breadth["signal"] == "WEAK" or vix["signal"] == "HIGH_FEAR":
        size_factor = 0.25
        recommendation = "25% size — weak breadth or high fear"
    else:
        size_factor = 0.5
        recommendation = "50% default — insufficient data"

    result = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "breadth": breadth,
        "vix": vix,
        "position_size_factor": size_factor,
        "recommendation": recommendation,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[BREADTH] Market Breadth Analysis:")
    print(f"  A/D Ratio  : {breadth['advance_decline_ratio']:.2f} ({breadth.get('advancing',0)}A / {breadth.get('declining',0)}D)")
    print(f"  % > SMA50  : {breadth['pct_above_sma50']:.1f}%")
    print(f"  % > SMA200 : {breadth['pct_above_sma200']:.1f}%")
    print(f"  Score      : {breadth['breadth_score']:.1f}/100 → {breadth['signal']}")
    print(f"  VIX        : {vix['vix']} ({vix['signal']})")
    print(f"  Sizing     : {size_factor:.0%} — {recommendation}")

    return result


if __name__ == "__main__":
    analyse_market_breadth()
