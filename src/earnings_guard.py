"""
====================================================
📅 PROJECT AEGIS - Earnings Calendar Guard
====================================================
Blocks or reduces position size near quarterly results
to avoid earnings surprise risk.

Data Sources (fallback chain):
  1. yfinance .calendar property
  2. Manual overrides from data/earnings_overrides.json
  3. Conservative: flag all positions in known season months

Features:
  - Block new buys within N days of results
  - Reduce existing position targets (tighten stops)
  - Dashboard-visible earnings calendar
  - Auto-refresh daily
====================================================
"""

import os
import sys
import json
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────
EARNINGS_BLOCK_DAYS    = 3      # Block new buys this many days before results
EARNINGS_REDUCE_DAYS   = 5      # Tighten stops this many days before
EARNINGS_REDUCE_FACTOR = 0.5    # Reduce target by 50% near earnings
EARNINGS_CACHE_HOURS   = 24     # Re-fetch after this many hours

EARNINGS_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "earnings_calendar.json",
)
EARNINGS_OVERRIDE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "earnings_overrides.json",
)

# Known Indian earnings season months (approximate)
EARNINGS_SEASON_MONTHS = {
    "Q1": [7, 8],        # Apr-Jun results announced Jul-Aug
    "Q2": [10, 11],      # Jul-Sep results announced Oct-Nov
    "Q3": [1, 2],        # Oct-Dec results announced Jan-Feb
    "Q4": [4, 5],        # Jan-Mar results announced Apr-May
}
ALL_EARNINGS_MONTHS = [1, 2, 4, 5, 7, 8, 10, 11]


# ──────────────────────────────────────────────────
#  FETCH EARNINGS DATES
# ──────────────────────────────────────────────────
def fetch_earnings_date_yf(symbol: str) -> str | None:
    """
    Try to get next earnings date from yfinance.
    Returns date string 'YYYY-MM-DD' or None.
    """
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is not None:
            if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.columns:
                date = pd.to_datetime(cal["Earnings Date"].iloc[0])
                return date.strftime("%Y-%m-%d")
            elif isinstance(cal, dict) and "Earnings Date" in cal:
                dates = cal["Earnings Date"]
                if isinstance(dates, list) and len(dates) > 0:
                    return pd.to_datetime(dates[0]).strftime("%Y-%m-%d")
                elif hasattr(dates, "strftime"):
                    return dates.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def load_overrides() -> dict:
    """Load manual earnings date overrides."""
    try:
        if os.path.exists(EARNINGS_OVERRIDE_FILE):
            with open(EARNINGS_OVERRIDE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def load_cached_calendar() -> dict:
    """Load cached earnings calendar."""
    try:
        if os.path.exists(EARNINGS_CACHE_FILE):
            with open(EARNINGS_CACHE_FILE, "r") as f:
                data = json.load(f)
            # Check freshness
            ts = data.get("timestamp", "")
            if ts:
                cached_time = datetime.strptime(ts, "%Y-%m-%d %H:%M")
                if (datetime.now() - cached_time).total_seconds() < EARNINGS_CACHE_HOURS * 3600:
                    return data.get("calendar", {})
    except Exception:
        pass
    return {}


def refresh_earnings_calendar(symbols: list) -> dict:
    """
    Build fresh earnings calendar for all symbols.
    Uses: cache → overrides → yfinance → conservative fallback.
    """
    cached = load_cached_calendar()
    overrides = load_overrides()

    calendar = {}
    today = datetime.now()

    for sym in symbols:
        # Priority 1: Manual override
        if sym in overrides:
            calendar[sym] = {
                "next_earnings": overrides[sym],
                "source": "override",
                "status": "confirmed",
            }
            continue

        # Priority 2: Cached (if still valid)
        if sym in cached:
            cached_date = cached[sym].get("next_earnings")
            if cached_date:
                try:
                    ed = datetime.strptime(cached_date, "%Y-%m-%d")
                    if ed >= today - timedelta(days=1):
                        calendar[sym] = cached[sym]
                        continue
                except Exception:
                    pass

        # Priority 3: yfinance
        yf_date = fetch_earnings_date_yf(sym)
        if yf_date:
            calendar[sym] = {
                "next_earnings": yf_date,
                "source": "yfinance",
                "status": "estimated",
            }
            continue

        # Priority 4: Conservative — assume next earnings in current season
        est_date = estimate_next_earnings(today)
        calendar[sym] = {
            "next_earnings": est_date,
            "source": "estimated_season",
            "status": "estimated",
        }

    # Save cache
    try:
        os.makedirs(os.path.dirname(EARNINGS_CACHE_FILE), exist_ok=True)
        with open(EARNINGS_CACHE_FILE, "w") as f:
            json.dump({
                "timestamp": today.strftime("%Y-%m-%d %H:%M"),
                "calendar": calendar,
            }, f, indent=2)
    except Exception:
        pass

    return calendar


def estimate_next_earnings(today: datetime = None) -> str:
    """Estimate next earnings date based on Indian quarterly season."""
    if today is None:
        today = datetime.now()

    current_month = today.month

    # Find next earnings month
    future_months = [m for m in ALL_EARNINGS_MONTHS if m >= current_month]
    if future_months:
        next_month = future_months[0]
        year = today.year
    else:
        next_month = ALL_EARNINGS_MONTHS[0]
        year = today.year + 1

    # Assume mid-month
    return f"{year}-{next_month:02d}-15"


# ──────────────────────────────────────────────────
#  PRIMARY API - GUARD CHECK
# ──────────────────────────────────────────────────
def check_earnings_guard(
    symbol: str,
    calendar: dict = None,
    today: datetime = None,
) -> dict:
    """
    Check if a stock is near earnings and should be blocked/reduced.

    Args:
        symbol: Stock symbol
        calendar: Pre-fetched earnings calendar
        today: Override for simulation/testing

    Returns:
        {
            blocked: bool,        # True = don't open new position
            reduce_target: bool,  # True = tighten stop/target
            reduce_factor: float, # Factor to multiply target distance
            days_to_earnings: int,
            earnings_date: str,
            reason: str,
        }
    """
    if today is None:
        today = datetime.now()

    result = {
        "blocked": False,
        "reduce_target": False,
        "reduce_factor": 1.0,
        "days_to_earnings": 999,
        "earnings_date": None,
        "reason": "clear",
    }

    if calendar is None:
        calendar = load_cached_calendar()

    if symbol not in calendar:
        return result

    entry = calendar[symbol]
    earnings_date_str = entry.get("next_earnings")
    if not earnings_date_str:
        return result

    try:
        earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
    except Exception:
        return result

    days_to = (earnings_date - today).days
    result["days_to_earnings"] = days_to
    result["earnings_date"] = earnings_date_str

    # Already passed
    if days_to < -1:
        result["reason"] = "earnings_passed"
        return result

    # Block zone
    if 0 <= days_to <= EARNINGS_BLOCK_DAYS:
        result["blocked"] = True
        result["reason"] = f"earnings_in_{days_to}_days"
        return result

    # Reduce zone
    if EARNINGS_BLOCK_DAYS < days_to <= EARNINGS_REDUCE_DAYS:
        result["reduce_target"] = True
        result["reduce_factor"] = EARNINGS_REDUCE_FACTOR
        result["reason"] = f"earnings_near_{days_to}_days"
        return result

    result["reason"] = "clear"
    return result


def is_earnings_season(today: datetime = None) -> bool:
    """Check if we're currently in a general earnings season."""
    if today is None:
        today = datetime.now()
    return today.month in ALL_EARNINGS_MONTHS


def get_upcoming_earnings(
    symbols: list,
    calendar: dict = None,
    days_ahead: int = 14,
) -> list:
    """
    Get list of stocks with upcoming earnings for dashboard display.
    Returns sorted list of {symbol, date, days_to, status, source}.
    """
    if calendar is None:
        calendar = load_cached_calendar()

    today = datetime.now()
    upcoming = []

    for sym in symbols:
        if sym not in calendar:
            continue
        entry = calendar[sym]
        ed = entry.get("next_earnings")
        if not ed:
            continue

        try:
            earnings_date = datetime.strptime(ed, "%Y-%m-%d")
            days_to = (earnings_date - today).days
            if -1 <= days_to <= days_ahead:
                upcoming.append({
                    "symbol": sym,
                    "date": ed,
                    "days_to": days_to,
                    "status": entry.get("status", "unknown"),
                    "source": entry.get("source", "unknown"),
                    "blocked": days_to <= EARNINGS_BLOCK_DAYS,
                })
        except Exception:
            continue

    return sorted(upcoming, key=lambda x: x["days_to"])


# ──────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    from config import STOCK_WATCHLIST, TOP_N_STOCKS

    symbols = STOCK_WATCHLIST[:TOP_N_STOCKS]
    print(f"\n📅 Earnings Calendar Guard")
    print(f"{'=' * 55}")
    print(f"  Block zone  : {EARNINGS_BLOCK_DAYS} days before")
    print(f"  Reduce zone : {EARNINGS_REDUCE_DAYS} days before")
    print(f"  Earnings season? {'YES' if is_earnings_season() else 'NO'}")
    print(f"\n  Refreshing calendar for {len(symbols)} stocks...")

    cal = refresh_earnings_calendar(symbols)

    print(f"\n  Earnings Calendar:")
    for sym in symbols:
        guard = check_earnings_guard(sym, cal)
        if guard["blocked"]:
            icon = "🚫"
        elif guard["reduce_target"]:
            icon = "⚠️"
        else:
            icon = "✅"

        ed = guard.get("earnings_date", "unknown")
        days = guard["days_to_earnings"]
        print(f"    {icon} {sym:15} | Earnings: {ed} | "
              f"Days: {days:3d} | {guard['reason']}")

    upcoming = get_upcoming_earnings(symbols, cal)
    if upcoming:
        print(f"\n  ⏰ Upcoming ({len(upcoming)} stocks within 14 days):")
        for u in upcoming:
            print(f"    {'🚫' if u['blocked'] else '📅'} {u['symbol']} — "
                  f"{u['date']} ({u['days_to']}d) [{u['source']}]")

    print(f"\n  📁 Calendar cached at {EARNINGS_CACHE_FILE}")
