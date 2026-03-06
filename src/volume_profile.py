"""
====================================================
PROJECT AEGIS — Volume Profile & Order Flow
====================================================
Volume-at-price analysis:
  - Volume profile (histogram by price level)
  - VWAP bands (1σ, 2σ)
  - Buy/sell pressure imbalance (volume delta proxy)
  - Point of Control (POC), Value Area High/Low

Outputs
-------
data/volume_profile.json
====================================================
"""

import os, json, time, warnings, math
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
VP_FILE = os.path.join(DATA_DIR, "volume_profile.json")
_cache = {}
CACHE_TTL = 600  # 10 min


# ══════════════════════════════════════════════════
#  VWAP CALCULATION
# ══════════════════════════════════════════════════
def compute_vwap(df: pd.DataFrame) -> dict:
    """
    Compute VWAP and bands from OHLCV data.
    Returns VWAP, upper/lower 1σ & 2σ bands.
    """
    if df is None or df.empty:
        return {}

    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    if hasattr(typical, "columns"):
        typical = typical.iloc[:, 0]

    vol = df["Volume"]
    if hasattr(vol, "columns"):
        vol = vol.iloc[:, 0]

    cum_vol = vol.cumsum()
    cum_tp_vol = (typical * vol).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    # Standard deviation bands
    cum_tp2_vol = ((typical ** 2) * vol).cumsum()
    variance = (cum_tp2_vol / cum_vol) - (vwap ** 2)
    variance = variance.clip(lower=0)
    std = np.sqrt(variance)

    last_vwap = float(vwap.iloc[-1]) if not vwap.empty else 0
    last_std = float(std.iloc[-1]) if not std.empty else 0

    return {
        "vwap": round(last_vwap, 2),
        "upper_1sd": round(last_vwap + last_std, 2),
        "lower_1sd": round(last_vwap - last_std, 2),
        "upper_2sd": round(last_vwap + 2 * last_std, 2),
        "lower_2sd": round(last_vwap - 2 * last_std, 2),
        "std": round(last_std, 2),
    }


# ══════════════════════════════════════════════════
#  VOLUME PROFILE
# ══════════════════════════════════════════════════
def compute_volume_profile(df: pd.DataFrame, bins: int = 30) -> dict:
    """
    Volume-at-price histogram.
    Returns POC, Value Area High/Low, histogram data.
    """
    if df is None or df.empty or len(df) < 5:
        return {}

    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    vol = df["Volume"]
    if hasattr(vol, "columns"):
        vol = vol.iloc[:, 0]

    price_min = float(close.min())
    price_max = float(close.max())
    if price_max == price_min:
        price_max = price_min + 1

    # Create price bins
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)

    for i in range(len(close)):
        p = float(close.iloc[i])
        v = float(vol.iloc[i])
        idx = int((p - price_min) / (price_max - price_min) * (bins - 1))
        idx = max(0, min(bins - 1, idx))
        bin_volumes[idx] += v

    # Point of Control (highest volume price)
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    # Value Area (70% of volume)
    total_vol = bin_volumes.sum()
    target_vol = total_vol * 0.70
    sorted_indices = np.argsort(bin_volumes)[::-1]
    cumulative = 0
    va_indices = []
    for idx in sorted_indices:
        cumulative += bin_volumes[idx]
        va_indices.append(idx)
        if cumulative >= target_vol:
            break

    va_low = (bin_edges[min(va_indices)] + bin_edges[min(va_indices) + 1]) / 2
    va_high = (bin_edges[max(va_indices)] + bin_edges[max(va_indices) + 1]) / 2

    # Histogram for dashboard
    histogram = []
    for i in range(bins):
        histogram.append({
            "price": round((bin_edges[i] + bin_edges[i + 1]) / 2, 2),
            "volume": round(float(bin_volumes[i]), 0),
            "pct": round(float(bin_volumes[i]) / total_vol * 100, 2) if total_vol > 0 else 0,
            "is_poc": i == poc_idx,
            "in_value_area": i in va_indices,
        })

    return {
        "poc": round(poc_price, 2),
        "value_area_high": round(va_high, 2),
        "value_area_low": round(va_low, 2),
        "total_volume": round(total_vol, 0),
        "histogram": histogram,
        "bins": bins,
    }


# ══════════════════════════════════════════════════
#  VOLUME DELTA (Buy/Sell Pressure)
# ══════════════════════════════════════════════════
def compute_volume_delta(df: pd.DataFrame) -> dict:
    """
    Estimate buy vs sell pressure using close vs open position in candle.
    (True tick data would need L2 data; this is a proxy.)
    """
    if df is None or df.empty:
        return {}

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    open_ = df["Open"]
    vol = df["Volume"]

    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
        open_ = open_.iloc[:, 0]
        high = high.iloc[:, 0]
        low = low.iloc[:, 0]
        vol = vol.iloc[:, 0]

    # Buy/sell split based on close position in candle range
    ranges = high - low
    ranges = ranges.replace(0, np.nan).fillna(1)
    buy_pct = (close - low) / ranges
    sell_pct = (high - close) / ranges

    buy_vol = (buy_pct * vol).sum()
    sell_vol = (sell_pct * vol).sum()
    delta = buy_vol - sell_vol
    total = buy_vol + sell_vol

    # Recent delta (last 5 bars)
    recent_buy = (buy_pct.tail(5) * vol.tail(5)).sum()
    recent_sell = (sell_pct.tail(5) * vol.tail(5)).sum()
    recent_delta = recent_buy - recent_sell

    pressure = "BUY" if delta > 0 else "SELL"
    strength = abs(delta) / total * 100 if total > 0 else 0

    return {
        "total_buy_volume": round(float(buy_vol), 0),
        "total_sell_volume": round(float(sell_vol), 0),
        "delta": round(float(delta), 0),
        "pressure": pressure,
        "strength_pct": round(strength, 2),
        "recent_delta_5bar": round(float(recent_delta), 0),
        "recent_pressure": "BUY" if recent_delta > 0 else "SELL",
    }


# ══════════════════════════════════════════════════
#  FULL STOCK ANALYSIS
# ══════════════════════════════════════════════════
def analyse_volume_profile(symbol: str, df: pd.DataFrame = None) -> dict:
    """Full volume analysis for a single stock."""
    cache_key = f"vp_{symbol}"
    if cache_key in _cache and (time.time() - _cache[cache_key]["ts"] < CACHE_TTL):
        return _cache[cache_key]["data"]

    if df is None:
        try:
            import yfinance as yf
            df = yf.download(symbol, period="60d", interval="1d",
                             progress=False, auto_adjust=True)
        except Exception:
            return {"symbol": symbol, "error": "no data"}

    if df is None or df.empty:
        return {"symbol": symbol, "error": "empty data"}

    vwap = compute_vwap(df)
    profile = compute_volume_profile(df)
    delta = compute_volume_delta(df)

    close = df["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    current_price = float(close.iloc[-1]) if not close.empty else 0

    # Position relative to VWAP & value area
    if vwap and current_price > 0:
        vwap_dist = ((current_price - vwap["vwap"]) / vwap["vwap"]) * 100 if vwap["vwap"] > 0 else 0
        above_vwap = current_price > vwap["vwap"]
    else:
        vwap_dist = 0
        above_vwap = False

    in_value_area = False
    if profile:
        in_value_area = profile.get("value_area_low", 0) <= current_price <= profile.get("value_area_high", float("inf"))

    result = {
        "symbol": symbol,
        "current_price": current_price,
        "vwap": vwap,
        "volume_profile": profile,
        "volume_delta": delta,
        "above_vwap": above_vwap,
        "vwap_distance_pct": round(vwap_dist, 2),
        "in_value_area": in_value_area,
        "timestamp": _now_ist(),
    }

    _cache[cache_key] = {"data": result, "ts": time.time()}
    return result


def analyse_all_volume(symbols: list[str]) -> dict:
    """Batch volume profile for all stocks."""
    results = {}
    for sym in symbols:
        try:
            results[sym] = analyse_volume_profile(sym)
        except Exception as e:
            results[sym] = {"symbol": sym, "error": str(e)}

    # Aggregate
    buy_pressure = sum(1 for r in results.values() if r.get("volume_delta", {}).get("pressure") == "BUY")
    sell_pressure = sum(1 for r in results.values() if r.get("volume_delta", {}).get("pressure") == "SELL")

    return {
        "stocks": results,
        "market_pressure": "BUY" if buy_pressure > sell_pressure else ("SELL" if sell_pressure > buy_pressure else "NEUTRAL"),
        "buy_pressure_count": buy_pressure,
        "sell_pressure_count": sell_pressure,
        "timestamp": _now_ist(),
    }


def check_volume_gate(symbol: str, df: pd.DataFrame = None) -> tuple[bool, str, dict]:
    """
    Sniper gate: block if strong sell pressure + below VWAP + outside value area.
    """
    data = analyse_volume_profile(symbol, df)
    delta = data.get("volume_delta", {})
    above_vwap = data.get("above_vwap", True)
    in_va = data.get("in_value_area", True)

    pressure = delta.get("pressure", "BUY")
    strength = delta.get("strength_pct", 0)

    # Block only if: SELL pressure > 60% AND below VWAP AND outside value area
    if pressure == "SELL" and strength > 60 and not above_vwap and not in_va:
        return False, f"Strong SELL pressure ({strength:.0f}%), below VWAP, outside VA — blocking", data
    if pressure == "SELL" and strength > 40 and not above_vwap:
        return True, f"SELL pressure caution ({strength:.0f}%), below VWAP — proceed carefully", data
    if pressure == "BUY" and above_vwap:
        return True, f"BUY pressure ({strength:.0f}%), above VWAP — strong", data
    return True, f"Volume neutral — {pressure} ({strength:.0f}%)", data


def save_vp_state(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(VP_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_ist() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
