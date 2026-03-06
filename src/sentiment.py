"""
====================================================
PROJECT AEGIS - Smart Sentiment Engine v2
====================================================
REAL news sentiment that actually works:

1. STOCK-SPECIFIC NEWS   — Headlines about the company
2. SECTOR NEWS           — What's happening in the sector
3. GLOBAL MACRO EVENTS   — Wars, rate hikes, oil prices, trade deals
4. INDIA MARKET MOOD     — India VIX, FII flows, RBI decisions
5. KEYWORD IMPACT SYSTEM — Detects high-impact words and phrases

Sources:
  - GoogleNews (stock + sector + macro headlines)
  - RSS feeds from major financial outlets (fallback)
  - Global market data from yfinance (VIX, crude, DXY)

The engine returns a COMPOSITE score:
  -1.0 (extremely bearish) → 0.0 (neutral) → +1.0 (extremely bullish)

Unlike v1 which used raw TextBlob polarity, v2 uses a
FINANCIAL-DOMAIN keyword scoring system that understands
terms like "rate hike" = bearish, "buyback" = bullish.
====================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional

# ──────────────────────────────────────────────────
#  Ticker → Company / Sector mapping
# ──────────────────────────────────────────────────
TICKER_TO_NAME = {
    "TATASTEEL.NS":  "Tata Steel",
    "SBIN.NS":       "State Bank of India SBI",
    "RELIANCE.NS":   "Reliance Industries",
    "HDFCBANK.NS":   "HDFC Bank",
    "ICICIBANK.NS":  "ICICI Bank",
    "NTPC.NS":       "NTPC Limited",
    "POWERGRID.NS":  "Power Grid Corporation",
    "COALINDIA.NS":  "Coal India",
    "INFY.NS":       "Infosys",
    "TCS.NS":        "TCS Tata Consultancy",
    "ITC.NS":        "ITC Limited",
    "WIPRO.NS":      "Wipro",
    "BHARTIARTL.NS": "Bharti Airtel",
    "LT.NS":         "Larsen Toubro",
    "AXISBANK.NS":   "Axis Bank",
}

TICKER_TO_SECTOR = {
    "TATASTEEL.NS":  "metals",
    "SBIN.NS":       "banking",
    "RELIANCE.NS":   "energy",
    "HDFCBANK.NS":   "banking",
    "ICICIBANK.NS":  "banking",
    "NTPC.NS":       "power",
    "POWERGRID.NS":  "power",
    "COALINDIA.NS":  "mining",
    "INFY.NS":       "it",
    "TCS.NS":        "it",
    "ITC.NS":        "fmcg",
    "WIPRO.NS":      "it",
    "BHARTIARTL.NS": "telecom",
    "LT.NS":         "infrastructure",
    "AXISBANK.NS":   "banking",
}

# Sector → search queries for sector-level news
SECTOR_QUERIES = {
    "metals":         ["steel industry India", "metal prices commodity"],
    "banking":        ["Indian banking sector", "RBI interest rate policy"],
    "energy":         ["crude oil price India", "energy sector India"],
    "power":          ["India power sector electricity", "coal price energy"],
    "mining":         ["coal mining India", "commodity prices mining"],
    "it":             ["Indian IT sector", "US tech layoffs outsourcing"],
    "fmcg":           ["FMCG India consumer", "rural demand India"],
    "telecom":        ["India telecom 5G", "telecom tariff India"],
    "infrastructure": ["India infrastructure spending", "government capex India"],
}

# Global macro queries — affects ALL stocks
GLOBAL_MACRO_QUERIES = [
    "India stock market today",
    "global recession risk",
    "US Federal Reserve interest rate",
    "crude oil price today",
    "India rupee dollar exchange",
    "war conflict geopolitical risk",
    "FII DII investment India",
    "India VIX volatility",
]

# ──────────────────────────────────────────────────
#  FINANCIAL KEYWORD SCORING (domain-specific)
#  Unlike TextBlob which treats "rate hike" as neutral,
#  our system knows it's BEARISH for stocks.
# ──────────────────────────────────────────────────

BULLISH_KEYWORDS = [
    # Strong bullish
    ("buyback", 0.8), ("dividend", 0.6), ("upgrade", 0.7),
    ("outperform", 0.7), ("beat estimate", 0.8), ("record profit", 0.9),
    ("record revenue", 0.8), ("strong result", 0.7), ("raised guidance", 0.8),
    ("order win", 0.7), ("new contract", 0.6), ("expansion plan", 0.5),
    ("profit surge", 0.8), ("revenue growth", 0.6), ("market share gain", 0.6),
    ("breakout", 0.5), ("rally", 0.5), ("bullish", 0.6), ("surge", 0.6),
    ("boom", 0.5), ("recovery", 0.5), ("rate cut", 0.7),
    ("stimulus", 0.6), ("reform", 0.5), ("FII buying", 0.7),
    ("ceasefire", 0.6), ("peace deal", 0.7), ("trade deal", 0.6),
    ("GDP growth", 0.5), ("manufacturing growth", 0.5),
    ("strong earnings", 0.7), ("better than expected", 0.7),
    ("positive outlook", 0.6), ("buy rating", 0.6),
    # Moderate bullish
    ("stable", 0.3), ("growth", 0.3), ("demand", 0.3),
    ("investment", 0.3), ("partnership", 0.4), ("innovation", 0.3),
    ("acquisition", 0.3), ("merger", 0.3),
]

BEARISH_KEYWORDS = [
    # Strong bearish
    ("war", -0.8), ("invasion", -0.9), ("military strike", -0.9),
    ("nuclear", -0.9), ("sanctions", -0.7), ("embargo", -0.7),
    ("recession", -0.8), ("crash", -0.9), ("collapse", -0.8),
    ("default", -0.9), ("bankruptcy", -0.9), ("fraud", -0.9),
    ("scam", -0.8), ("downgrade", -0.7), ("miss estimate", -0.7),
    ("rate hike", -0.6), ("rate increase", -0.6), ("inflation surge", -0.6),
    ("profit warning", -0.8), ("loss widens", -0.7), ("guidance cut", -0.8),
    ("layoff", -0.5), ("job cut", -0.5), ("plant shutdown", -0.6),
    ("sell-off", -0.7), ("correction", -0.5), ("bearish", -0.6),
    ("FII selling", -0.7), ("capital outflow", -0.6),
    ("crude oil surge", -0.5), ("rupee fall", -0.5), ("rupee depreciation", -0.5),
    ("geopolitical tension", -0.6), ("conflict escalation", -0.7),
    ("trade war", -0.7), ("tariff hike", -0.6),
    ("worse than expected", -0.7), ("negative outlook", -0.6),
    ("sell rating", -0.6), ("probe", -0.5), ("investigation", -0.5),
    ("debt crisis", -0.7), ("liquidity crisis", -0.8),
    # Moderate bearish
    ("volatility", -0.3), ("uncertainty", -0.3), ("tension", -0.3),
    ("concern", -0.3), ("decline", -0.4), ("falling", -0.3),
    ("weak", -0.4), ("slowdown", -0.4), ("slump", -0.5),
]


def _keyword_score(text: str) -> float:
    """
    Score a headline using financial-domain keywords.
    Returns a value roughly between -1.0 and +1.0.
    """
    text_lower = text.lower()
    total_score = 0.0
    matches = 0

    for keyword, score in BULLISH_KEYWORDS:
        if keyword.lower() in text_lower:
            total_score += score
            matches += 1

    for keyword, score in BEARISH_KEYWORDS:
        if keyword.lower() in text_lower:
            total_score += score
            matches += 1

    if matches == 0:
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            return blob.sentiment.polarity * 0.5  # Dampen TextBlob (less reliable)
        except ImportError:
            return 0.0

    avg = total_score / max(matches, 1)
    return max(-1.0, min(1.0, avg))


def _ticker_to_query(symbol: str) -> str:
    """Convert Yahoo Finance ticker to a news search query."""
    name = TICKER_TO_NAME.get(symbol)
    if name:
        return f"{name} stock India"
    base = symbol.replace(".NS", "").replace(".BO", "")
    return f"{base} stock India"


# ──────────────────────────────────────────────────
#  NEWS FETCHERS
# ──────────────────────────────────────────────────

def _fetch_googlenews(query: str, lookback_days: int = 3, max_articles: int = 10) -> list:
    """Fetch headlines from GoogleNews."""
    try:
        from GoogleNews import GoogleNews
        gn = GoogleNews(lang="en", region="IN")
        gn.set_period(f"{lookback_days}d")
        gn.get_news(query)
        results = gn.results(sort=True)
        headlines = [r.get("title", "") for r in results[:max_articles] if r.get("title")]
        gn.clear()
        return headlines
    except ImportError:
        return []
    except Exception:
        return []


def _fetch_rss_headlines(query: str, max_articles: int = 5) -> list:
    """Fallback: fetch from free RSS feeds (Economic Times)."""
    import urllib.request
    import xml.etree.ElementTree as ET

    rss_feeds = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    ]

    headlines = []
    query_words = set(query.lower().split())

    for feed_url in rss_feeds:
        try:
            req = urllib.request.Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.iter("item"):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    title = title_el.text.strip()
                    title_lower = title.lower()
                    if any(w in title_lower for w in query_words) or "market" in title_lower or "stock" in title_lower:
                        headlines.append(title)
                        if len(headlines) >= max_articles:
                            break
        except Exception:
            continue
    return headlines


# ──────────────────────────────────────────────────
#  GLOBAL MARKET MOOD (VIX, crude, USD/INR)
# ──────────────────────────────────────────────────

_global_mood_cache = {"score": 0.0, "details": {}, "timestamp": 0}
GLOBAL_MOOD_CACHE_SECONDS = 600


def get_global_market_mood() -> dict:
    """
    Check global market indicators to gauge overall mood.
    Returns {"score": float, "details": dict}

    Factors:
      - India VIX (>20 = fearful, <15 = calm)
      - US market (S&P 500 direction)
      - Crude oil (high = bad for India)
      - USD/INR (weakening rupee = bad)
    """
    global _global_mood_cache
    now = time.time()
    if now - _global_mood_cache["timestamp"] < GLOBAL_MOOD_CACHE_SECONDS:
        return _global_mood_cache

    mood_score = 0.0
    details = {}

    try:
        import yfinance as yf
        from indicators import flatten_yf_columns

        # India VIX
        try:
            vix = yf.download("^INDIAVIX", period="5d", interval="1d", progress=False)
            vix = flatten_yf_columns(vix)
            if vix is not None and not vix.empty:
                india_vix = float(vix["Close"].iloc[-1])
                details["india_vix"] = round(india_vix, 2)
                if india_vix > 25:
                    mood_score -= 0.4
                    details["vix_signal"] = "HIGH_FEAR"
                elif india_vix > 20:
                    mood_score -= 0.2
                    details["vix_signal"] = "MODERATE_FEAR"
                elif india_vix < 13:
                    mood_score += 0.2
                    details["vix_signal"] = "CALM"
                else:
                    details["vix_signal"] = "NORMAL"
        except Exception:
            details["india_vix"] = "N/A"

        # S&P 500
        try:
            sp500 = yf.download("^GSPC", period="5d", interval="1d", progress=False)
            sp500 = flatten_yf_columns(sp500)
            if sp500 is not None and len(sp500) >= 2:
                sp_change = (float(sp500["Close"].iloc[-1]) - float(sp500["Close"].iloc[-2])) / float(sp500["Close"].iloc[-2])
                details["sp500_change"] = round(sp_change * 100, 2)
                if sp_change < -0.02:
                    mood_score -= 0.3
                    details["sp500_signal"] = "BEARISH"
                elif sp_change < -0.01:
                    mood_score -= 0.15
                    details["sp500_signal"] = "WEAK"
                elif sp_change > 0.01:
                    mood_score += 0.15
                    details["sp500_signal"] = "BULLISH"
                else:
                    details["sp500_signal"] = "NEUTRAL"
        except Exception:
            details["sp500_change"] = "N/A"

        # Crude Oil (Brent)
        try:
            crude = yf.download("BZ=F", period="5d", interval="1d", progress=False)
            crude = flatten_yf_columns(crude)
            if crude is not None and len(crude) >= 2:
                crude_price = float(crude["Close"].iloc[-1])
                crude_change = (float(crude["Close"].iloc[-1]) - float(crude["Close"].iloc[-2])) / float(crude["Close"].iloc[-2])
                details["crude_price"] = round(crude_price, 2)
                details["crude_change"] = round(crude_change * 100, 2)
                if crude_price > 90:
                    mood_score -= 0.3
                    details["crude_signal"] = "BEARISH"
                elif crude_price > 80:
                    mood_score -= 0.1
                    details["crude_signal"] = "CAUTIOUS"
                elif crude_price < 65:
                    mood_score += 0.2
                    details["crude_signal"] = "BULLISH"
                else:
                    details["crude_signal"] = "NEUTRAL"
        except Exception:
            details["crude_price"] = "N/A"

        # USD/INR
        try:
            usdinr = yf.download("USDINR=X", period="5d", interval="1d", progress=False)
            usdinr = flatten_yf_columns(usdinr)
            if usdinr is not None and len(usdinr) >= 2:
                inr_change = (float(usdinr["Close"].iloc[-1]) - float(usdinr["Close"].iloc[-2])) / float(usdinr["Close"].iloc[-2])
                details["usdinr"] = round(float(usdinr["Close"].iloc[-1]), 2)
                details["usdinr_change"] = round(inr_change * 100, 2)
                if inr_change > 0.005:
                    mood_score -= 0.15
                    details["inr_signal"] = "WEAKENING"
                elif inr_change < -0.005:
                    mood_score += 0.1
                    details["inr_signal"] = "STRENGTHENING"
                else:
                    details["inr_signal"] = "STABLE"
        except Exception:
            details["usdinr"] = "N/A"

    except ImportError:
        pass

    mood_score = max(-1.0, min(1.0, mood_score))
    details["composite_mood"] = round(mood_score, 3)
    _global_mood_cache = {"score": mood_score, "details": details, "timestamp": now}
    return _global_mood_cache


# ──────────────────────────────────────────────────
#  MACRO NEWS SENTIMENT (wars, geopolitics, etc.)
# ──────────────────────────────────────────────────

_macro_cache = {"score": 0.0, "headlines": [], "timestamp": 0}
MACRO_CACHE_SECONDS = 600


def get_macro_sentiment(lookback_days: int = 3) -> dict:
    """
    Fetch global macro news and return aggregate sentiment.
    Catches wars, sanctions, global crises that affect ALL stocks.
    """
    global _macro_cache
    now = time.time()
    if now - _macro_cache["timestamp"] < MACRO_CACHE_SECONDS:
        return _macro_cache

    all_headlines = []
    for query in GLOBAL_MACRO_QUERIES:
        headlines = _fetch_googlenews(query, lookback_days=lookback_days, max_articles=5)
        if not headlines:
            headlines = _fetch_rss_headlines(query, max_articles=3)
        all_headlines.extend(headlines)

    if not all_headlines:
        _macro_cache = {"score": 0.0, "headlines": [], "timestamp": now}
        return _macro_cache

    # Deduplicate
    seen = set()
    unique = []
    for h in all_headlines:
        h_lower = h.lower().strip()
        if h_lower not in seen:
            seen.add(h_lower)
            unique.append(h)

    scores = [_keyword_score(h) for h in unique]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_score = max(-1.0, min(1.0, avg_score))

    n_bearish = sum(1 for s in scores if s < -0.2)
    n_bullish = sum(1 for s in scores if s > 0.2)

    print(f"   📰  MACRO: {len(unique)} headlines | "
          f"Score: {avg_score:+.3f} | "
          f"Bullish: {n_bullish} | Bearish: {n_bearish}")

    _macro_cache = {
        "score": round(avg_score, 4),
        "headlines": unique[:20],
        "n_bullish": n_bullish,
        "n_bearish": n_bearish,
        "timestamp": now,
    }
    return _macro_cache


# ──────────────────────────────────────────────────
#  STOCK SENTIMENT (enhanced v2)
# ──────────────────────────────────────────────────

def get_sentiment_score(
    symbol: str,
    max_articles: int = 10,
    lookback_days: int = 3,
) -> float:
    """
    COMPOSITE sentiment score for a stock.

    Components (weighted):
      40% — Stock-specific news
      25% — Sector news
      20% — Global macro news
      15% — Global market mood (VIX, crude, FX)

    Returns float between -1.0 and +1.0
    """
    # ── 1. Stock-specific news ──
    stock_score = 0.0
    stock_query = _ticker_to_query(symbol)
    stock_headlines = _fetch_googlenews(stock_query, lookback_days, max_articles)
    if not stock_headlines:
        stock_headlines = _fetch_rss_headlines(stock_query, max_articles=5)

    if stock_headlines:
        stock_scores = [_keyword_score(h) for h in stock_headlines]
        stock_score = sum(stock_scores) / len(stock_scores)
        print(f"   📰  {symbol}: {len(stock_headlines)} stock headlines → {stock_score:+.3f}")
    else:
        print(f"   📰  No stock news for {symbol}")

    # ── 2. Sector news ──
    sector_score = 0.0
    sector = TICKER_TO_SECTOR.get(symbol, "")
    if sector and sector in SECTOR_QUERIES:
        sector_headlines = []
        for sq in SECTOR_QUERIES[sector]:
            sh = _fetch_googlenews(sq, lookback_days, max_articles=5)
            sector_headlines.extend(sh)
        if sector_headlines:
            sec_scores = [_keyword_score(h) for h in sector_headlines]
            sector_score = sum(sec_scores) / len(sec_scores)
            print(f"   📰  {symbol} sector ({sector}): {len(sector_headlines)} headlines → {sector_score:+.3f}")

    # ── 3. Global macro news ──
    macro = get_macro_sentiment(lookback_days)
    macro_score = macro.get("score", 0.0)

    # ── 4. Global market mood ──
    mood = get_global_market_mood()
    mood_score = mood.get("score", 0.0)

    # ── COMPOSITE SCORE (weighted) ──
    composite = (
        0.40 * stock_score +
        0.25 * sector_score +
        0.20 * macro_score +
        0.15 * mood_score
    )
    composite = max(-1.0, min(1.0, composite))

    print(f"   📰  {symbol} COMPOSITE: {composite:+.3f} "
          f"(stock={stock_score:+.3f}, sector={sector_score:+.3f}, "
          f"macro={macro_score:+.3f}, mood={mood_score:+.3f})")

    return round(composite, 4)


def get_sentiment_details(symbol: str, max_articles: int = 10, lookback_days: int = 3) -> dict:
    """
    Like get_sentiment_score but returns full breakdown for the dashboard.
    """
    result = {
        "symbol": symbol,
        "stock_score": 0.0, "stock_headlines": [],
        "sector": TICKER_TO_SECTOR.get(symbol, "unknown"),
        "sector_score": 0.0, "sector_headlines": [],
        "macro_score": 0.0, "macro_headlines": [],
        "mood_score": 0.0, "mood_details": {},
        "composite": 0.0,
    }

    # Stock
    stock_query = _ticker_to_query(symbol)
    stock_hl = _fetch_googlenews(stock_query, lookback_days, max_articles)
    if not stock_hl:
        stock_hl = _fetch_rss_headlines(stock_query, max_articles=5)
    result["stock_headlines"] = stock_hl[:10]
    if stock_hl:
        result["stock_score"] = round(sum(_keyword_score(h) for h in stock_hl) / len(stock_hl), 4)

    # Sector
    sector = result["sector"]
    if sector in SECTOR_QUERIES:
        sec_hl = []
        for sq in SECTOR_QUERIES[sector]:
            sec_hl.extend(_fetch_googlenews(sq, lookback_days, 5))
        result["sector_headlines"] = sec_hl[:10]
        if sec_hl:
            result["sector_score"] = round(sum(_keyword_score(h) for h in sec_hl) / len(sec_hl), 4)

    # Macro
    macro = get_macro_sentiment(lookback_days)
    result["macro_score"] = macro.get("score", 0.0)
    result["macro_headlines"] = macro.get("headlines", [])[:10]

    # Mood
    mood = get_global_market_mood()
    result["mood_score"] = mood.get("score", 0.0)
    result["mood_details"] = mood.get("details", {})

    # Composite
    result["composite"] = round(
        0.40 * result["stock_score"] +
        0.25 * result["sector_score"] +
        0.20 * result["macro_score"] +
        0.15 * result["mood_score"],
        4
    )
    return result


def get_bulk_sentiment(
    symbols: list,
    max_articles: int = 10,
    lookback_days: int = 3,
) -> dict:
    """Fetch sentiment for multiple tickers. Returns {symbol: score}."""
    print(f"\n📰  Fetching ENHANCED sentiment for {len(symbols)} stocks ...")
    results = {}
    for sym in symbols:
        results[sym] = get_sentiment_score(sym, max_articles, lookback_days)
    return results


# ──────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SENTIMENT ENGINE v2 — Self Test")
    print("=" * 60)

    print("\n--- Global Market Mood ---")
    mood = get_global_market_mood()
    for k, v in mood.get("details", {}).items():
        print(f"   {k}: {v}")

    print("\n--- Macro Sentiment ---")
    macro = get_macro_sentiment()
    print(f"   Score: {macro.get('score', 0)}")
    for h in macro.get("headlines", [])[:5]:
        print(f"   • {h}")

    print("\n--- Stock Sentiment ---")
    test_symbols = ["TATASTEEL.NS", "SBIN.NS", "RELIANCE.NS"]
    scores = get_bulk_sentiment(test_symbols)
    for sym, score in scores.items():
        bar = "+" * int(abs(score) * 20) if score >= 0 else "-" * int(abs(score) * 20)
        print(f"   {sym:<18} → {score:+.3f}  {bar}")
