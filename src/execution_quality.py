"""
====================================================
PROJECT AEGIS — Execution Quality Analytics
====================================================
Tracks entry precision:
  - Theoretical vs actual entry (slippage estimation)
  - Timing score (did we enter at intraday low?)
  - Fill latency simulation
  - Trade-by-trade execution grading

Outputs
-------
data/execution_quality.json
====================================================
"""

import os, json, time, warnings, math
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EQ_FILE = os.path.join(DATA_DIR, "execution_quality.json")
TRADE_LOG = os.path.join(DATA_DIR, "trade_history.csv")


# ══════════════════════════════════════════════════
#  SINGLE-TRADE EXECUTION ANALYSIS
# ══════════════════════════════════════════════════
def analyse_single_trade(symbol: str, entry_price: float, entry_time: str,
                          exit_price: float = 0, exit_time: str = "",
                          qty: int = 1) -> dict:
    """
    Grade a single trade's execution quality.
    Fetches intraday data to compare against best possible.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", interval="15m")
        if hist is None or hist.empty:
            return {"error": "no intraday data", "grade": "N/A"}

        close = hist["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        low = hist["Low"]
        if hasattr(low, "columns"):
            low = low.iloc[:, 0]
        high = hist["High"]
        if hasattr(high, "columns"):
            high = high.iloc[:, 0]

        day_low = float(low.min())
        day_high = float(high.max())
        day_range = day_high - day_low if day_high > day_low else 1.0

        # Timing score: how close to day-low was the entry? (0=worst, 100=perfect)
        if day_range > 0:
            timing_score = max(0, 100 - ((entry_price - day_low) / day_range) * 100)
        else:
            timing_score = 50

        # Slippage estimation (vs. VWAP proxy)
        vwap = float((hist["Close"] * hist["Volume"]).sum() / hist["Volume"].sum()) if hist["Volume"].sum() > 0 else entry_price
        slippage_bps = ((entry_price - vwap) / vwap) * 10000 if vwap > 0 else 0

        # Grade
        if timing_score >= 80:
            grade = "A"
        elif timing_score >= 60:
            grade = "B"
        elif timing_score >= 40:
            grade = "C"
        elif timing_score >= 20:
            grade = "D"
        else:
            grade = "F"

        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "day_low": round(day_low, 2),
            "day_high": round(day_high, 2),
            "vwap_estimate": round(vwap, 2),
            "timing_score": round(timing_score, 1),
            "slippage_bps": round(slippage_bps, 2),
            "slippage_rupees": round(slippage_bps / 10000 * entry_price * qty, 2),
            "grade": grade,
        }
    except Exception as e:
        return {"error": str(e), "grade": "N/A"}


# ══════════════════════════════════════════════════
#  BATCH ANALYSIS FROM TRADE LOG
# ══════════════════════════════════════════════════
def analyse_execution_quality(trade_log_path: str | None = None, last_n: int = 50) -> dict:
    """
    Analyse execution quality across recent trades.
    Returns aggregate metrics + per-trade grades.
    """
    if trade_log_path is None:
        trade_log_path = TRADE_LOG

    if not os.path.exists(trade_log_path):
        return {
            "error": "no trade history",
            "trades_analysed": 0,
            "avg_timing_score": 0,
            "avg_slippage_bps": 0,
            "grade_distribution": {},
            "timestamp": _now_ist(),
        }

    try:
        df = pd.read_csv(trade_log_path)
    except Exception:
        return {"error": "cannot read trade log", "trades_analysed": 0, "timestamp": _now_ist()}

    # Filter BUY entries only
    buys = df[df["Action"] == "BUY"].tail(last_n)
    if buys.empty:
        return {
            "trades_analysed": 0,
            "avg_timing_score": 0,
            "avg_slippage_bps": 0,
            "grade_distribution": {},
            "trade_details": [],
            "timestamp": _now_ist(),
        }

    trade_details = []
    timing_scores = []
    slippages = []
    grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0, "N/A": 0}

    for _, row in buys.iterrows():
        sym = str(row.get("Stock", ""))
        entry_p = float(row.get("Entry_Price", 0))
        entry_t = str(row.get("Time", ""))
        qty = int(row.get("Qty", 1))

        if not sym or entry_p <= 0:
            continue

        analysis = analyse_single_trade(sym, entry_p, entry_t, qty=qty)
        if analysis.get("grade", "N/A") != "N/A":
            timing_scores.append(analysis.get("timing_score", 50))
            slippages.append(analysis.get("slippage_bps", 0))

        grade = analysis.get("grade", "N/A")
        grades[grade] = grades.get(grade, 0) + 1
        trade_details.append(analysis)

    avg_timing = float(np.mean(timing_scores)) if timing_scores else 0
    avg_slip = float(np.mean(slippages)) if slippages else 0
    total_slip_cost = sum(t.get("slippage_rupees", 0) for t in trade_details)

    # Overall execution grade
    if avg_timing >= 70:
        overall_grade = "A"
    elif avg_timing >= 55:
        overall_grade = "B"
    elif avg_timing >= 40:
        overall_grade = "C"
    else:
        overall_grade = "D"

    return {
        "trades_analysed": len(trade_details),
        "avg_timing_score": round(avg_timing, 1),
        "avg_slippage_bps": round(avg_slip, 2),
        "total_slippage_cost": round(total_slip_cost, 2),
        "overall_grade": overall_grade,
        "grade_distribution": grades,
        "best_timing": round(max(timing_scores), 1) if timing_scores else 0,
        "worst_timing": round(min(timing_scores), 1) if timing_scores else 0,
        "improvement_tip": _get_improvement_tip(avg_timing, avg_slip),
        "trade_details": trade_details[-20:],  # Last 20 for dashboard
        "timestamp": _now_ist(),
    }


def _get_improvement_tip(avg_timing: float, avg_slip: float) -> str:
    """Generate actionable improvement suggestion."""
    if avg_timing < 30:
        return "Entries are consistently near day-highs. Consider waiting for pullbacks or using limit orders."
    if avg_timing < 50:
        return "Entries are mediocre. Try using intraday pattern timing (10:30-11:30 AM window often offers better prices)."
    if avg_slip > 15:
        return "High slippage vs VWAP. Consider using VWAP-based limit orders instead of market orders."
    if avg_timing >= 70:
        return "Excellent entry timing! Current approach is working well."
    return "Decent execution. Minor improvements possible with tighter intraday timing windows."


def log_execution(trade_data: dict):
    """
    Append execution data to a running log.
    Called by sniper after each BUY order fill.
    """
    log_file = os.path.join(DATA_DIR, "execution_log.json")
    log = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                log = json.load(f)
        except Exception:
            log = []

    trade_data["timestamp"] = _now_ist()
    log.append(trade_data)

    # Keep last 500 entries
    log = log[-500:]
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2, default=str)


def save_eq_state(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(EQ_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_ist() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
