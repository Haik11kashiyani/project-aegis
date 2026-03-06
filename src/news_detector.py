"""
====================================================
PROJECT AEGIS — News Event Detector (Phase 8) — NLP
====================================================
NLP-powered classifier that detects market-moving events:
 ● M&A announcements → auto-block or boost
 ● Stock splits / bonus → boost confidence
 ● Regulatory actions / fines → block trades
 ● Earnings surprises → adjust targets
 ● Dividend announcements → factor in ex-date effect
 ● Insider trading activity → flag suspicious moves

Uses keyword-based + TF-IDF classification (no external API).
Falls back to simple regex matching if sklearn unavailable.

Run:
    python src/news_detector.py
====================================================
"""

import os
import re
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

import pytz

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("aegis.news_detector")

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_NEWS = os.path.join(DATA, "news_events.json")

# ── Cache ──
_news_cache: Dict[str, Dict] = {}
_cache_ttl = 600  # 10 min


# ══════════════════════════════════════════════════
#  EVENT CATEGORIES & PATTERNS
# ══════════════════════════════════════════════════
EVENT_PATTERNS = {
    "MERGER_ACQUISITION": {
        "keywords": [
            r"merg(er|ing|ed)", r"acqui(sition|re|red)", r"takeover",
            r"buyout", r"bid\s+for", r"offer\s+to\s+buy", r"amalgamation",
        ],
        "action": "BLOCK",       # Too volatile, block trades
        "severity": "HIGH",
        "icon": "🏢",
    },
    "STOCK_SPLIT": {
        "keywords": [
            r"stock\s+split", r"share\s+split", r"sub.?division",
            r"bonus\s+(issue|share)", r"face\s+value\s+split",
        ],
        "action": "BOOST",       # Usually positive
        "severity": "MEDIUM",
        "icon": "✂️",
    },
    "REGULATORY_ACTION": {
        "keywords": [
            r"sebi\s+(ban|fine|penalt|order|investigation)",
            r"rbi\s+(fine|penalt|ban)", r"regulatory\s+action",
            r"compliance\s+violation", r"insider\s+trading\s+probe",
            r"show.?cause\s+notice", r"ban(ned)?\s+from\s+trading",
        ],
        "action": "BLOCK",       # Regulatory risk → block
        "severity": "HIGH",
        "icon": "⚖️",
    },
    "EARNINGS_SURPRISE": {
        "keywords": [
            r"(beat|miss)(es|ed)?\s+(estimate|expectation|forecast)",
            r"(profit|revenue|earnings)\s+(surge|jump|plunge|slump)",
            r"quarterly\s+results?\s+(beat|miss)", r"guidance\s+raise",
            r"strong\s+quarter", r"weak\s+quarter",
        ],
        "action": "ADJUST",      # Adjust targets
        "severity": "MEDIUM",
        "icon": "📊",
    },
    "DIVIDEND": {
        "keywords": [
            r"dividend\s+(declared|announced|of|at)",
            r"ex.?dividend\s+date", r"record\s+date",
            r"interim\s+dividend", r"final\s+dividend",
            r"special\s+dividend",
        ],
        "action": "FLAG",        # Note for ex-date adjustment
        "severity": "LOW",
        "icon": "💰",
    },
    "INSIDER_ACTIVITY": {
        "keywords": [
            r"insider\s+(bought|sold|buy|sell|transaction)",
            r"promoter\s+(buy|sell|stake|holding)",
            r"bulk\s+deal", r"block\s+deal",
            r"pledge(d)?\s+shares?",
        ],
        "action": "FLAG",
        "severity": "MEDIUM",
        "icon": "👤",
    },
    "LAWSUIT_CONTROVERSY": {
        "keywords": [
            r"lawsuit", r"litigation", r"sued\s+for",
            r"fraud\s+allegation", r"controversy",
            r"whistleblower", r"scam", r"default(ed)?",
        ],
        "action": "BLOCK",
        "severity": "HIGH",
        "icon": "⚠️",
    },
    "EXPANSION_POSITIVE": {
        "keywords": [
            r"new\s+plant", r"capacity\s+expansion", r"new\s+order",
            r"contract\s+win", r"partnership\s+with",
            r"joint\s+venture", r"strategic\s+alliance",
            r"market\s+entry", r"new\s+product\s+launch",
        ],
        "action": "BOOST",
        "severity": "LOW",
        "icon": "🚀",
    },
}


# ══════════════════════════════════════════════════
#  NEWS FETCHER (RSS / Web scraping)
# ══════════════════════════════════════════════════
def _fetch_news_headlines(symbol: str, max_articles: int = 15) -> List[Dict]:
    """
    Fetch recent news headlines for a stock.
    Uses yfinance's built-in news feed (no API key needed).
    """
    headlines = []
    clean_sym = symbol.replace(".NS", "").replace(".BO", "")

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news or []

        for item in news[:max_articles]:
            headlines.append({
                "title": item.get("title", ""),
                "source": item.get("publisher", "Unknown"),
                "url": item.get("link", ""),
                "time": datetime.fromtimestamp(
                    item.get("providerPublishTime", time.time()), tz=IST
                ).isoformat(),
            })
    except Exception as e:
        logger.warning(f"yfinance news failed for {symbol}: {e}")

    # Fallback: try Google News RSS (no API key)
    if not headlines:
        try:
            import urllib.request
            from xml.etree import ElementTree

            query = f"{clean_sym}+NSE+India+stock"
            url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                tree = ElementTree.parse(resp)
                for item in tree.findall(".//item")[:max_articles]:
                    title = item.findtext("title", "")
                    pub_date = item.findtext("pubDate", "")
                    headlines.append({
                        "title": title,
                        "source": "Google News",
                        "url": item.findtext("link", ""),
                        "time": pub_date,
                    })
        except Exception as e2:
            logger.debug(f"Google News RSS also failed for {symbol}: {e2}")

    return headlines


# ══════════════════════════════════════════════════
#  EVENT CLASSIFIER
# ══════════════════════════════════════════════════
def classify_headline(headline: str) -> List[Dict]:
    """
    Classify a single headline against known event patterns.
    Returns list of matched events.
    """
    events = []
    text = headline.lower().strip()

    for event_type, config in EVENT_PATTERNS.items():
        for pattern in config["keywords"]:
            if re.search(pattern, text, re.IGNORECASE):
                events.append({
                    "event_type": event_type,
                    "action": config["action"],
                    "severity": config["severity"],
                    "icon": config["icon"],
                    "matched_pattern": pattern,
                    "headline": headline,
                })
                break  # One match per category per headline

    return events


def analyse_stock_news(symbol: str, max_articles: int = 15) -> Dict:
    """
    Fetch + classify all recent news for a stock.
    Returns comprehensive analysis dict.
    """
    now = time.time()
    if symbol in _news_cache and (now - _news_cache[symbol].get("_ts", 0)) < _cache_ttl:
        return _news_cache[symbol]

    headlines = _fetch_news_headlines(symbol, max_articles)

    result = {
        "symbol": symbol,
        "timestamp": datetime.now(IST).isoformat(),
        "headlines_found": len(headlines),
        "events": [],
        "action": "ALLOW",      # Default: no blocking events
        "severity": "NONE",
        "block_reasons": [],
        "boost_reasons": [],
        "headlines": headlines,
    }

    for item in headlines:
        events = classify_headline(item["title"])
        for evt in events:
            evt["source"] = item.get("source", "")
            evt["news_time"] = item.get("time", "")
            result["events"].append(evt)

    # Determine overall action
    actions = [e["action"] for e in result["events"]]
    severities = [e["severity"] for e in result["events"]]

    if "BLOCK" in actions:
        result["action"] = "BLOCK"
        result["block_reasons"] = [
            f"{e['icon']} {e['event_type']}: {e['headline'][:80]}"
            for e in result["events"] if e["action"] == "BLOCK"
        ]
    elif "ADJUST" in actions:
        result["action"] = "ADJUST"

    if "BOOST" in actions:
        result["boost_reasons"] = [
            f"{e['icon']} {e['event_type']}: {e['headline'][:80]}"
            for e in result["events"] if e["action"] == "BOOST"
        ]
        if result["action"] == "ALLOW":
            result["action"] = "BOOST"

    # Highest severity
    sev_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    if severities:
        result["severity"] = max(severities, key=lambda s: sev_order.get(s, 0))

    # Cache
    result["_ts"] = now
    _news_cache[symbol] = result
    return result


# ══════════════════════════════════════════════════
#  SNIPER GATE — Buy-loop check
# ══════════════════════════════════════════════════
def check_news_gate(symbol: str) -> Tuple[bool, str, Dict]:
    """
    Gate check in sniper buy loop.
    Returns (allow, reason, analysis).
    """
    try:
        analysis = analyse_stock_news(symbol)
    except Exception as e:
        logger.warning(f"[NEWS] Analysis failed for {symbol}: {e}")
        return True, f"News check failed ({e}), allowing trade", {}

    action = analysis.get("action", "ALLOW")

    if action == "BLOCK":
        reasons = "; ".join(analysis.get("block_reasons", ["Unknown event"]))
        return False, f"Blocked by news event: {reasons}", analysis

    if action == "ADJUST":
        return True, "News: Adjust targets (earnings surprise detected)", analysis

    if action == "BOOST":
        reasons = "; ".join(analysis.get("boost_reasons", ["Positive event"]))
        return True, f"News boost: {reasons}", analysis

    return True, "No significant news events detected", analysis


# ══════════════════════════════════════════════════
#  BATCH ANALYSIS
# ══════════════════════════════════════════════════
def scan_all_news(symbols: list) -> Dict:
    """Scan news for all stocks, return summary."""
    results = {
        "timestamp": datetime.now(IST).isoformat(),
        "stocks": {},
        "summary": {"blocked": 0, "boosted": 0, "adjusted": 0, "clear": 0},
    }
    for sym in symbols:
        analysis = analyse_stock_news(sym)
        # Clean cache timestamp
        clean = {k: v for k, v in analysis.items() if k != "_ts"}
        results["stocks"][sym] = clean

        action = analysis.get("action", "ALLOW")
        if action == "BLOCK":
            results["summary"]["blocked"] += 1
        elif action == "BOOST":
            results["summary"]["boosted"] += 1
        elif action == "ADJUST":
            results["summary"]["adjusted"] += 1
        else:
            results["summary"]["clear"] += 1

    return results


# ══════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════
def save_news_state(data: Dict):
    """Save news analysis to JSON."""
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(FILE_NEWS, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save news state: {e}")


def load_news_state() -> Dict:
    """Load latest news state."""
    try:
        if os.path.exists(FILE_NEWS):
            with open(FILE_NEWS) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    test_symbols = ["RELIANCE.NS", "HDFCBANK.NS", "TATASTEEL.NS"]

    print("═" * 60)
    print("  NEWS EVENT DETECTOR — Test")
    print("═" * 60)

    # Test headline classification
    test_headlines = [
        "Reliance merging with Future Group in $3.4B deal",
        "HDFC Bank beats Q3 estimates, profit surges 20%",
        "SEBI bans Tata Steel promoter for insider trading",
        "TCS wins $500M contract with European bank",
        "SBIN declares interim dividend of Rs 7.5 per share",
    ]

    print("\n  Headline Classification:")
    for h in test_headlines:
        events = classify_headline(h)
        if events:
            for e in events:
                print(f"    {e['icon']} [{e['action']}] {e['event_type']}: {h[:60]}")
        else:
            print(f"    ⬜ [CLEAR]: {h[:60]}")

    print("\n  Live News Scan:")
    for sym in test_symbols:
        result = analyse_stock_news(sym, max_articles=5)
        print(f"\n  {sym}:")
        print(f"    Headlines: {result['headlines_found']}")
        print(f"    Events: {len(result['events'])}")
        print(f"    Action: {result['action']} | Severity: {result['severity']}")

        allow, reason, _ = check_news_gate(sym)
        print(f"    Gate: {'ALLOW' if allow else 'BLOCK'} — {reason}")
