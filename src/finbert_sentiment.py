"""
====================================================
PROJECT AEGIS — FinBERT Sentiment Analyser
====================================================
Upgrades the keyword-based news_detector with a distilled
FinBERT transformer for financial sentiment classification.

Falls back to keyword matching if transformers/torch not installed.

Usage: pip install transformers torch  (or transformers[torch])

Outputs
-------
data/finbert_sentiment.json
====================================================
"""

import os, json, time, re, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FINBERT_FILE = os.path.join(DATA_DIR, "finbert_sentiment.json")

# Try to load FinBERT once
_pipeline = None
_finbert_available = False

def _load_finbert():
    """Lazy-load the FinBERT pipeline."""
    global _pipeline, _finbert_available
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        _finbert_available = True
        return _pipeline
    except Exception:
        _finbert_available = False
        return None


# ══════════════════════════════════════════════════
#  KEYWORD FALLBACK (same as news_detector patterns)
# ══════════════════════════════════════════════════
_POS_PATTERNS = [
    r"(?:beat|top|exceed|surpass).*(?:estimate|expectation|forecast)",
    r"(?:revenue|profit|earnings).*(?:surge|jump|soar|rise|grow)",
    r"(?:upgrade|overweight|outperform|buy)\s+(?:rating|target)",
    r"(?:expansion|launch|partnership|deal|contract|acquire)",
    r"(?:dividend|buyback|bonus|split)",
    r"(?:strong|robust|solid|record).*(?:quarter|results|growth)",
]
_NEG_PATTERNS = [
    r"(?:miss|below|disappoint|fail).*(?:estimate|expectation)",
    r"(?:revenue|profit|earnings).*(?:decline|drop|fall|plunge|slump)",
    r"(?:downgrade|underweight|underperform|sell)\s+(?:rating|target)",
    r"(?:lawsuit|fraud|scandal|investigation|penalty|fine|ban)",
    r"(?:debt|default|bankruptcy|restructur|insolvency)",
    r"(?:weak|poor|dismal).*(?:quarter|results|outlook)",
]


def _keyword_sentiment(text: str) -> dict:
    """Fallback keyword-based sentiment."""
    text_lower = text.lower()
    pos_hits = sum(1 for p in _POS_PATTERNS if re.search(p, text_lower))
    neg_hits = sum(1 for p in _NEG_PATTERNS if re.search(p, text_lower))
    total = pos_hits + neg_hits
    if total == 0:
        return {"label": "neutral", "score": 0.5, "method": "keyword"}
    if pos_hits > neg_hits:
        return {"label": "positive", "score": 0.5 + 0.5 * (pos_hits / total), "method": "keyword"}
    elif neg_hits > pos_hits:
        return {"label": "negative", "score": 0.5 + 0.5 * (neg_hits / total), "method": "keyword"}
    return {"label": "neutral", "score": 0.5, "method": "keyword"}


# ══════════════════════════════════════════════════
#  CORE SENTIMENT FUNCTION
# ══════════════════════════════════════════════════
def analyse_text(text: str) -> dict:
    """
    Classify a single piece of text → sentiment.
    Returns: {"label": "positive"|"negative"|"neutral", "score": 0-1, "method": ...}
    """
    pipe = _load_finbert()
    if pipe is not None:
        try:
            results = pipe(text[:512])
            if results and isinstance(results[0], list):
                results = results[0]
            # Results: list of {"label": "positive"/"negative"/"neutral", "score": float}
            best = max(results, key=lambda x: x["score"])
            return {
                "label": best["label"].lower(),
                "score": round(best["score"], 4),
                "all_scores": {r["label"].lower(): round(r["score"], 4) for r in results},
                "method": "finbert",
            }
        except Exception:
            pass
    return _keyword_sentiment(text)


def analyse_headlines(headlines: list[str]) -> dict:
    """
    Analyse a batch of headlines → aggregate sentiment.
    Returns: {"overall": ..., "positive": N, "negative": N, "neutral": N, "avg_score": ...}
    """
    if not headlines:
        return {
            "overall": "neutral",
            "positive": 0, "negative": 0, "neutral": 0,
            "avg_score": 0.0,
            "details": [],
            "method": "finbert" if _finbert_available else "keyword",
        }

    details = []
    pos, neg, neu = 0, 0, 0
    score_sum = 0.0

    for h in headlines[:50]:  # Cap at 50 for speed
        result = analyse_text(h)
        label = result["label"]
        if label == "positive":
            pos += 1
            score_sum += result["score"]
        elif label == "negative":
            neg += 1
            score_sum -= result["score"]
        else:
            neu += 1
        details.append({
            "text": h[:200],
            "label": label,
            "score": result["score"],
        })

    total = pos + neg + neu
    avg = score_sum / total if total > 0 else 0

    if pos > neg * 1.5:
        overall = "positive"
    elif neg > pos * 1.5:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "total": total,
        "avg_score": round(avg, 4),
        "details": details,
        "method": "finbert" if _finbert_available else "keyword",
    }


def analyse_stock_sentiment(symbol: str) -> dict:
    """Fetch headlines for a stock from yfinance and analyse."""
    headlines = []
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        for item in news[:20]:
            title = item.get("title", "")
            if title:
                headlines.append(title)
    except Exception:
        pass

    # Google News RSS fallback
    if len(headlines) < 5:
        try:
            import urllib.request
            import xml.etree.ElementTree as ET
            clean = symbol.replace(".NS", "").replace(".BO", "")
            url = f"https://news.google.com/rss/search?q={clean}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.findall(".//item")[:15]:
                title = item.findtext("title", "")
                if title:
                    headlines.append(title)
        except Exception:
            pass

    result = analyse_headlines(headlines)
    result["symbol"] = symbol
    result["headline_count"] = len(headlines)
    return result


def analyse_all_stocks(symbols: list[str]) -> dict:
    """Batch analyse all watchlist stocks."""
    stock_results = {}
    overall_pos, overall_neg, overall_neu = 0, 0, 0

    for sym in symbols:
        try:
            r = analyse_stock_sentiment(sym)
            stock_results[sym] = r
            overall_pos += r.get("positive", 0)
            overall_neg += r.get("negative", 0)
            overall_neu += r.get("neutral", 0)
        except Exception as e:
            stock_results[sym] = {"error": str(e), "overall": "neutral"}

    total = overall_pos + overall_neg + overall_neu
    if total > 0:
        market_sentiment = (overall_pos - overall_neg) / total
    else:
        market_sentiment = 0.0

    return {
        "stocks": stock_results,
        "market_sentiment": round(market_sentiment, 4),
        "market_label": "BULLISH" if market_sentiment > 0.2 else ("BEARISH" if market_sentiment < -0.2 else "NEUTRAL"),
        "total_headlines": total,
        "positive": overall_pos,
        "negative": overall_neg,
        "neutral": overall_neu,
        "finbert_available": _finbert_available,
        "method": "finbert" if _finbert_available else "keyword",
        "timestamp": _now_ist(),
    }


def get_finbert_gate(symbol: str, threshold: float = -0.4) -> tuple[bool, str, dict]:
    """
    Sniper gate: block if FinBERT sentiment is very negative.
    Returns (allow, reason, data).
    """
    data = analyse_stock_sentiment(symbol)
    overall = data.get("overall", "neutral")
    avg_score = data.get("avg_score", 0)

    if overall == "negative" and avg_score < threshold:
        return False, f"FinBERT NEGATIVE (score={avg_score:.2f}) — blocking buy", data
    if overall == "negative":
        return True, f"FinBERT cautious (score={avg_score:.2f}) — proceed carefully", data
    if overall == "positive":
        return True, f"FinBERT POSITIVE (score={avg_score:.2f}) — sentiment boost", data
    return True, f"FinBERT neutral (score={avg_score:.2f})", data


def save_finbert_state(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FINBERT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_ist() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
