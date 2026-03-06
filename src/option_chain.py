"""
====================================================
PROJECT AEGIS — Live NSE Option Chain Fetcher
====================================================
Fetches real option chain data from NSE India APIs,
replacing Black-Scholes estimates with actual market prices.

Falls back to BS calculation if NSE API is unavailable.

Outputs
-------
data/option_chain.json
====================================================
"""

import os, json, time, math, warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OC_FILE = os.path.join(DATA_DIR, "option_chain.json")
_cache = {}
CACHE_TTL = 300  # 5 min

# NSE F&O lot sizes (as of 2025-2026 cycle — update as needed)
LOT_SIZES = {
    "TATASTEEL": 850, "SBIN": 750, "RELIANCE": 250,
    "HDFCBANK": 550, "ICICIBANK": 700, "NTPC": 2250,
    "POWERGRID": 2700, "COALINDIA": 1700, "INFY": 600,
    "TCS": 175, "NIFTY": 25, "BANKNIFTY": 15,
}


# ══════════════════════════════════════════════════
#  NSE API FETCHER
# ══════════════════════════════════════════════════
_nse_session = None

def _get_nse_session():
    """Create a session with NSE cookies."""
    global _nse_session
    if _nse_session is not None:
        return _nse_session
    try:
        import urllib.request
        # First hit NSE homepage to get cookies
        url = "https://www.nseindia.com"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        urllib.request.urlopen(req, timeout=10)
        _nse_session = True
        return _nse_session
    except Exception:
        return None


def fetch_option_chain_nse(symbol: str) -> dict | None:
    """
    Fetch live option chain from NSE India.
    Returns parsed data or None on failure.
    """
    cache_key = f"oc_{symbol}"
    if cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry["ts"] < CACHE_TTL:
            return entry["data"]

    clean = symbol.replace(".NS", "").replace(".BO", "")
    try:
        import urllib.request
        _get_nse_session()

        if clean in ("NIFTY", "BANKNIFTY"):
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={clean}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={clean}"

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        if "records" not in data:
            return None

        records = data["records"]
        parsed = _parse_nse_chain(records, clean)
        _cache[cache_key] = {"data": parsed, "ts": time.time()}
        return parsed

    except Exception:
        return None


def _parse_nse_chain(records: dict, symbol: str) -> dict:
    """Parse NSE API response into our standard format."""
    spot = records.get("underlyingValue", 0)
    expiry_dates = records.get("expiryDates", [])
    near_expiry = expiry_dates[0] if expiry_dates else ""

    all_data = records.get("data", [])
    calls, puts = [], []
    total_ce_oi, total_pe_oi = 0, 0
    max_ce_oi_strike, max_pe_oi_strike = 0, 0
    max_ce_oi, max_pe_oi = 0, 0

    for row in all_data:
        strike = row.get("strikePrice", 0)
        expiry = row.get("expiryDate", "")
        if expiry != near_expiry:
            continue

        # Call
        ce = row.get("CE", {})
        if ce:
            ce_oi = ce.get("openInterest", 0)
            total_ce_oi += ce_oi
            if ce_oi > max_ce_oi:
                max_ce_oi = ce_oi
                max_ce_oi_strike = strike
            calls.append({
                "strike": strike,
                "ltp": ce.get("lastPrice", 0),
                "bid": ce.get("bidprice", 0),
                "ask": ce.get("askPrice", 0),
                "oi": ce_oi,
                "volume": ce.get("totalTradedVolume", 0),
                "iv": ce.get("impliedVolatility", 0),
                "change_oi": ce.get("changeinOpenInterest", 0),
            })

        # Put
        pe = row.get("PE", {})
        if pe:
            pe_oi = pe.get("openInterest", 0)
            total_pe_oi += pe_oi
            if pe_oi > max_pe_oi:
                max_pe_oi = pe_oi
                max_pe_oi_strike = strike
            puts.append({
                "strike": strike,
                "ltp": pe.get("lastPrice", 0),
                "bid": pe.get("bidprice", 0),
                "ask": pe.get("askPrice", 0),
                "oi": pe_oi,
                "volume": pe.get("totalTradedVolume", 0),
                "iv": pe.get("impliedVolatility", 0),
                "change_oi": pe.get("changeinOpenInterest", 0),
            })

    # PCR
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

    # Support / Resistance from max OI
    return {
        "symbol": symbol,
        "spot": spot,
        "expiry": near_expiry,
        "calls": calls,
        "puts": puts,
        "pcr": round(pcr, 4),
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "max_ce_oi_strike": max_ce_oi_strike,
        "max_pe_oi_strike": max_pe_oi_strike,
        "resistance": max_ce_oi_strike,
        "support": max_pe_oi_strike,
        "sentiment": "BULLISH" if pcr > 1.2 else ("BEARISH" if pcr < 0.7 else "NEUTRAL"),
        "source": "NSE_LIVE",
        "timestamp": _now_ist(),
    }


# ══════════════════════════════════════════════════
#  BLACK-SCHOLES FALLBACK
# ══════════════════════════════════════════════════
def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def bs_option_price(S: float, K: float, T: float, r: float = 0.065,
                    sigma: float = 0.25, option_type: str = "put") -> dict:
    """Black-Scholes pricing with Greeks."""
    if T <= 0:
        T = 1 / 365
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1

    gamma = math.exp(-d1 ** 2 / 2) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))
    theta = (-(S * sigma * math.exp(-d1 ** 2 / 2)) / (2 * math.sqrt(2 * math.pi * T))
             - r * K * math.exp(-r * T) * _norm_cdf(-d2 if option_type == "put" else d2))
    vega = S * math.sqrt(T) * math.exp(-d1 ** 2 / 2) / math.sqrt(2 * math.pi) / 100

    return {
        "price": round(price, 2),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta / 365, 4),
        "vega": round(vega, 4),
        "iv_input": sigma,
    }


def get_hedge_price(symbol: str, spot: float, strike: float = 0,
                    vol: float = 0.25) -> dict:
    """
    Get protective put price — live NSE first, BS fallback.
    """
    clean = symbol.replace(".NS", "").replace(".BO", "")

    # Try live chain first
    chain = fetch_option_chain_nse(symbol)
    if chain and chain.get("puts"):
        # Find nearest ATM put
        if strike == 0:
            strike = round(spot / 50) * 50  # Round to nearest 50
        best = None
        best_diff = float("inf")
        for p in chain["puts"]:
            diff = abs(p["strike"] - strike)
            if diff < best_diff:
                best_diff = diff
                best = p
        if best and best.get("ltp", 0) > 0:
            lot_size = LOT_SIZES.get(clean, 1)
            return {
                "source": "NSE_LIVE",
                "strike": best["strike"],
                "premium": best["ltp"],
                "iv": best.get("iv", 0),
                "bid": best.get("bid", 0),
                "ask": best.get("ask", 0),
                "oi": best.get("oi", 0),
                "lot_size": lot_size,
                "total_cost": best["ltp"] * lot_size,
            }

    # BS fallback
    if strike == 0:
        strike = round(spot * 0.97 / 50) * 50  # 3% OTM put
    bs = bs_option_price(spot, strike, 30 / 365, sigma=vol)
    lot_size = LOT_SIZES.get(clean, 1)
    return {
        "source": "BLACK_SCHOLES",
        "strike": strike,
        "premium": bs["price"],
        "iv": vol,
        "delta": bs["delta"],
        "gamma": bs["gamma"],
        "theta": bs["theta"],
        "vega": bs["vega"],
        "lot_size": lot_size,
        "total_cost": bs["price"] * lot_size,
    }


# ══════════════════════════════════════════════════
#  FULL CHAIN ANALYSIS
# ══════════════════════════════════════════════════
def analyse_option_chain(symbols: list[str]) -> dict:
    """Analyse option chains for all stocks in watchlist."""
    results = {}
    nse_hits, bs_fallbacks = 0, 0

    for sym in symbols:
        clean = sym.replace(".NS", "").replace(".BO", "")
        chain = fetch_option_chain_nse(sym)
        if chain:
            results[sym] = chain
            nse_hits += 1
        else:
            # Minimal BS-based chain
            try:
                import yfinance as yf
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="5d")
                if hist is not None and not hist.empty:
                    close = hist["Close"]
                    if hasattr(close, "columns"):
                        close = close.iloc[:, 0]
                    spot = float(close.iloc[-1])
                    hedge = get_hedge_price(sym, spot)
                    results[sym] = {
                        "symbol": clean,
                        "spot": spot,
                        "source": "BLACK_SCHOLES",
                        "atm_put": hedge,
                        "pcr": 1.0,
                        "sentiment": "NEUTRAL",
                        "timestamp": _now_ist(),
                    }
                    bs_fallbacks += 1
            except Exception:
                results[sym] = {"symbol": clean, "error": "no data", "source": "NONE"}

    return {
        "chains": results,
        "nse_live": nse_hits,
        "bs_fallback": bs_fallbacks,
        "total": len(symbols),
        "timestamp": _now_ist(),
    }


def save_oc_state(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OC_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_ist() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
