"""
====================================================
🛡️ PROJECT AEGIS - Sentiment Analyser
====================================================
Scrapes recent news headlines for a given stock and
returns a sentiment score between -1.0 (very negative)
and +1.0 (very positive) using TextBlob NLP.

Sources tried in order:
  1. GoogleNews (free, no API key)
  2. Fallback → neutral 0.0 if scraping fails

The score is used as the 12th feature ("Sentiment_Score")
in the Random Forest model.
====================================================
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

# ──────────────────────────────────────────────────
#  Ticker → Human-readable company name mapping
#  (used as search query for news)
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


def _ticker_to_query(symbol: str) -> str:
    """Convert Yahoo Finance ticker to a news search query."""
    name = TICKER_TO_NAME.get(symbol)
    if name:
        return f"{name} stock India"
    # Fallback: strip .NS/.BO suffix
    base = symbol.replace(".NS", "").replace(".BO", "")
    return f"{base} stock India"


def get_sentiment_score(
    symbol: str,
    max_articles: int = 10,
    lookback_days: int = 3,
) -> float:
    """
    Fetch recent news for `symbol` and return average sentiment.

    Returns:
        float between -1.0 and +1.0  (0.0 = neutral / no data)
    """
    query = _ticker_to_query(symbol)

    # ── Try GoogleNews ──
    try:
        from GoogleNews import GoogleNews
        from textblob import TextBlob

        gn = GoogleNews(lang="en", region="IN")
        gn.set_period(f"{lookback_days}d")
        gn.get_news(query)
        results = gn.results(sort=True)

        if not results:
            print(f"   📰  No news found for {symbol}. Sentiment = 0.0")
            return 0.0

        headlines = [r.get("title", "") for r in results[:max_articles] if r.get("title")]

        if not headlines:
            return 0.0

        scores = []
        for headline in headlines:
            blob = TextBlob(headline)
            scores.append(blob.sentiment.polarity)    # -1 to +1

        avg = sum(scores) / len(scores)
        avg = max(-1.0, min(1.0, avg))               # clamp

        print(f"   📰  {symbol}: {len(headlines)} headlines → Sentiment = {avg:.3f}")
        return round(avg, 4)

    except ImportError:
        print(f"   📰  GoogleNews/TextBlob not installed. Sentiment = 0.0")
        return 0.0
    except Exception as e:
        print(f"   📰  Sentiment fetch failed for {symbol}: {e}. Returning 0.0")
        return 0.0


def get_bulk_sentiment(
    symbols: list[str],
    max_articles: int = 10,
    lookback_days: int = 3,
) -> dict[str, float]:
    """
    Fetch sentiment for multiple tickers.
    Returns {symbol: score} dict.
    """
    print(f"\n📰  Fetching sentiment for {len(symbols)} stocks ...")
    results = {}
    for sym in symbols:
        results[sym] = get_sentiment_score(sym, max_articles, lookback_days)
    return results


# ──────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    test_symbols = ["TATASTEEL.NS", "SBIN.NS", "RELIANCE.NS"]
    scores = get_bulk_sentiment(test_symbols)
    for sym, score in scores.items():
        bar = "+" * int(abs(score) * 20) if score >= 0 else "-" * int(abs(score) * 20)
        print(f"   {sym:<18} → {score:+.3f}  {bar}")
