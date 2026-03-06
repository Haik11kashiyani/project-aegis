"""
====================================================
📊 PROJECT AEGIS — Sentiment Momentum Index (SMI)
====================================================
Phase 14 · Fully Dynamic

Composite real-time index that combines REAL signals from:
  - FinBERT sentiment scores (per-stock + aggregate)
  - News Detector article sentiment
  - Market mood / Fear & Greed proxy
  - Social buzz proxy (volume anomaly + search trends if avail)

Uses exponential decay weighting so recent sentiment carries more
weight than stale signals. Computes per-stock and market-wide SMI.

SMI range: [-1.0, +1.0]
  > +0.5  = Strong bullish sentiment momentum
  > +0.2  = Moderate bullish
  > -0.2  = Neutral
  > -0.5  = Moderate bearish
  < -0.5  = Strong bearish sentiment momentum

====================================================
"""

import os, json, math, time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None

# ── Paths ────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "sentiment_momentum_state.json")

# ── Decay / Weighting Parameters ─────────────────
DECAY_HALF_LIFE_HOURS = 6.0        # Signal half-life (hours)
DECAY_LAMBDA = math.log(2) / (DECAY_HALF_LIFE_HOURS * 3600)  # per-second decay

# Component weights (sum to 1.0)
W_FINBERT  = 0.35    # FinBERT NLP weight
W_NEWS     = 0.25    # News article sentiment weight
W_MOOD     = 0.20    # Market mood / fear-greed weight
W_SOCIAL   = 0.20    # Social buzz proxy weight


# ═══════════════════════════════════════════════════
#  SIGNAL SOURCES — All from REAL module outputs
# ═══════════════════════════════════════════════════

def _score_finbert(sym: str, finbert_state: Dict) -> Tuple[float, float, str]:
    """Extract FinBERT sentiment score for a stock from REAL module state.
    Returns (score, age_seconds, detail_str)."""
    stock_scores = finbert_state.get("stock_scores", {})
    sym_data = stock_scores.get(sym, {})
    if not sym_data:
        return 0.0, 999999, "no FinBERT data"

    score = sym_data.get("score", 0)
    label = sym_data.get("label", "NEUTRAL")
    ts = finbert_state.get("timestamp", "")
    age = _compute_age_seconds(ts)

    detail = f"FinBERT: {label} ({score:+.3f})"
    return float(score), age, detail


def _score_news(sym: str, news_state: Dict) -> Tuple[float, float, str]:
    """Extract news sentiment from REAL News Detector module state."""
    stocks = news_state.get("stocks", {})
    sym_news = stocks.get(sym, {})
    if not sym_news:
        return 0.0, 999999, "no news data"

    sentiment = sym_news.get("sentiment", 0)
    count = sym_news.get("article_count", 0)
    ts = news_state.get("timestamp", "")
    age = _compute_age_seconds(ts)

    # Scale: more articles = more confidence in the signal
    confidence_mult = min(count / 5.0, 1.5) if count > 0 else 0.5
    adjusted = sentiment * confidence_mult
    adjusted = max(-1.0, min(1.0, adjusted))

    detail = f"News: {count} articles, sent={sentiment:+.3f}, adj={adjusted:+.3f}"
    return adjusted, age, detail


def _score_mood(market_mood: Dict) -> Tuple[float, float, str]:
    """Extract market mood / fear-greed proxy from REAL market mood data."""
    if not market_mood:
        return 0.0, 999999, "no mood data"

    # Fear & Greed: 0=extreme fear, 100=extreme greed
    fg = market_mood.get("fear_greed", 50)
    # Normalise to [-1, 1]: 50 → 0, 0 → -1, 100 → +1
    normalised = (fg - 50) / 50.0

    ts = market_mood.get("timestamp", "")
    age = _compute_age_seconds(ts)

    detail = f"Mood: F&G={fg}, normalised={normalised:+.3f}"
    return normalised, age, detail


def _score_social(sym: str, df: Optional[pd.DataFrame]) -> Tuple[float, float, str]:
    """Social buzz proxy from REAL volume anomaly.
    High unusual volume = buzz indicator. Uses real OHLCV data."""
    if df is None or len(df) < 25 or "Volume" not in df.columns:
        return 0.0, 999999, "no volume data"

    vol = df["Volume"].values.astype(float)
    vol_ma = float(np.mean(vol[-20:])) if len(vol) >= 20 else float(np.mean(vol))
    vol_current = float(vol[-1])
    vol_ratio = vol_current / max(vol_ma, 1)

    # Map volume ratio to sentiment signal:
    # < 0.5x → negative buzz (people leaving), > 2x → positive buzz
    if vol_ratio > 2.0:
        score = min((vol_ratio - 1.0) * 0.3, 1.0)
        detail = f"Social: BUZZ ({vol_ratio:.1f}x vol)"
    elif vol_ratio > 1.3:
        score = (vol_ratio - 1.0) * 0.2
        detail = f"Social: mild interest ({vol_ratio:.1f}x vol)"
    elif vol_ratio < 0.5:
        score = -(1.0 - vol_ratio) * 0.3
        detail = f"Social: quiet ({vol_ratio:.1f}x vol)"
    else:
        score = 0.0
        detail = f"Social: normal ({vol_ratio:.1f}x vol)"

    # Also check price momentum as social proxy
    close = df["Close"].values.astype(float)
    ret_1d = (close[-1] - close[-2]) / max(close[-2], 1) if len(close) > 1 else 0

    # Strong moves attract social attention
    if abs(ret_1d) > 0.03:  # >3% daily move
        social_boost = np.sign(ret_1d) * 0.15
        score = max(-1.0, min(1.0, score + social_boost))
        detail += f" | 1d-move={ret_1d*100:+.1f}%"

    return score, 0, detail


def _compute_age_seconds(timestamp_str: str) -> float:
    """Compute age in seconds from ISO timestamp string."""
    if not timestamp_str:
        return 999999
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if ts.tzinfo is None and IST:
            ts = IST.localize(ts)
        now = datetime.now(IST) if IST else datetime.now()
        if now.tzinfo is None and IST:
            now = IST.localize(now)
        age = (now - ts).total_seconds()
        return max(age, 0)
    except Exception:
        return 999999


def _apply_decay(score: float, age_seconds: float) -> float:
    """Apply exponential decay to a signal based on its age."""
    decay_factor = math.exp(-DECAY_LAMBDA * age_seconds)
    return score * decay_factor


# ═══════════════════════════════════════════════════
#  SMI COMPUTATION
# ═══════════════════════════════════════════════════

def compute_smi(sym: str,
                df: Optional[pd.DataFrame] = None,
                finbert_state: Optional[Dict] = None,
                news_state: Optional[Dict] = None,
                market_mood: Optional[Dict] = None) -> Dict:
    """Compute Sentiment Momentum Index for a single stock.
    All inputs are REAL module outputs — nothing hardcoded."""

    finbert_state = finbert_state or {}
    news_state = news_state or {}
    market_mood = market_mood or {}

    components = []

    # FinBERT component
    fb_score, fb_age, fb_detail = _score_finbert(sym, finbert_state)
    fb_decayed = _apply_decay(fb_score, fb_age)
    components.append({
        "source": "FinBERT", "raw": round(fb_score, 4),
        "decayed": round(fb_decayed, 4), "weight": W_FINBERT,
        "age_hours": round(fb_age / 3600, 1), "detail": fb_detail,
    })

    # News component
    news_score, news_age, news_detail = _score_news(sym, news_state)
    news_decayed = _apply_decay(news_score, news_age)
    components.append({
        "source": "News", "raw": round(news_score, 4),
        "decayed": round(news_decayed, 4), "weight": W_NEWS,
        "age_hours": round(news_age / 3600, 1), "detail": news_detail,
    })

    # Market Mood component
    mood_score, mood_age, mood_detail = _score_mood(market_mood)
    mood_decayed = _apply_decay(mood_score, mood_age)
    components.append({
        "source": "Mood", "raw": round(mood_score, 4),
        "decayed": round(mood_decayed, 4), "weight": W_MOOD,
        "age_hours": round(mood_age / 3600, 1), "detail": mood_detail,
    })

    # Social Buzz component
    social_score, social_age, social_detail = _score_social(sym, df)
    social_decayed = _apply_decay(social_score, social_age)
    components.append({
        "source": "Social", "raw": round(social_score, 4),
        "decayed": round(social_decayed, 4), "weight": W_SOCIAL,
        "age_hours": round(social_age / 3600, 1), "detail": social_detail,
    })

    # Weighted combination
    smi = (
        fb_decayed * W_FINBERT +
        news_decayed * W_NEWS +
        mood_decayed * W_MOOD +
        social_decayed * W_SOCIAL
    )
    smi = max(-1.0, min(1.0, smi))

    # Classify
    if smi > 0.5:
        label = "STRONG_BULLISH"
    elif smi > 0.2:
        label = "BULLISH"
    elif smi > -0.2:
        label = "NEUTRAL"
    elif smi > -0.5:
        label = "BEARISH"
    else:
        label = "STRONG_BEARISH"

    return {
        "symbol": sym,
        "smi": round(smi, 4),
        "label": label,
        "components": components,
        "timestamp": datetime.now().isoformat(),
    }


def compute_market_smi(symbols: List[str],
                       stock_dfs: Optional[Dict[str, pd.DataFrame]] = None,
                       finbert_state: Optional[Dict] = None,
                       news_state: Optional[Dict] = None,
                       market_mood: Optional[Dict] = None) -> Dict:
    """Compute market-wide SMI across all tracked stocks.
    Uses REAL data from each stock to produce aggregate index."""
    stock_dfs = stock_dfs or {}
    per_stock = {}
    smi_values = []

    for sym in symbols:
        df = stock_dfs.get(sym)
        result = compute_smi(sym, df, finbert_state, news_state, market_mood)
        per_stock[sym] = result
        smi_values.append(result["smi"])

    if smi_values:
        market_avg = float(np.mean(smi_values))
        market_std = float(np.std(smi_values))
        best_sym = symbols[int(np.argmax(smi_values))]
        worst_sym = symbols[int(np.argmin(smi_values))]
    else:
        market_avg = 0.0
        market_std = 0.0
        best_sym = "—"
        worst_sym = "—"

    # Classify market
    if market_avg > 0.3:
        market_label = "BULLISH"
    elif market_avg > -0.3:
        market_label = "NEUTRAL"
    else:
        market_label = "BEARISH"

    return {
        "market_smi": round(market_avg, 4),
        "market_label": market_label,
        "market_std": round(market_std, 4),
        "best_sentiment": best_sym,
        "worst_sentiment": worst_sym,
        "per_stock": per_stock,
        "n_stocks": len(symbols),
        "timestamp": datetime.now().isoformat(),
    }


def check_smi_gate(sym: str, smi_result: Optional[Dict] = None,
                   df: Optional[pd.DataFrame] = None,
                   finbert_state: Optional[Dict] = None,
                   news_state: Optional[Dict] = None,
                   market_mood: Optional[Dict] = None,
                   threshold: float = -0.35) -> Tuple[bool, str, float]:
    """Gate check: reject trades with very negative sentiment momentum.
    Returns (passed, reason, smi_value)."""
    if smi_result is None:
        smi_result = compute_smi(sym, df, finbert_state, news_state, market_mood)

    smi = smi_result.get("smi", 0)
    label = smi_result.get("label", "NEUTRAL")

    if smi < threshold:
        return False, f"SMI too low: {smi:.3f} ({label})", smi
    return True, f"SMI OK: {smi:.3f} ({label})", smi


def get_smi_sizing_factor(smi: float) -> float:
    """Adjust position sizing based on SMI. Bullish = larger, bearish = smaller."""
    if smi > 0.5:
        return 1.15   # 15% boost
    elif smi > 0.2:
        return 1.05   # 5% boost
    elif smi > -0.2:
        return 1.0    # Neutral
    elif smi > -0.5:
        return 0.90   # 10% reduction
    else:
        return 0.75   # 25% reduction


def get_smi_status(state: Optional[Dict] = None) -> Dict:
    """Return status for dashboard."""
    if state is None:
        state = load_smi_state()
    if not state:
        return {"status": "NO_DATA"}
    return {
        "status": "ACTIVE",
        "market_smi": state.get("market_smi", 0),
        "market_label": state.get("market_label", "NEUTRAL"),
        "best_sentiment": state.get("best_sentiment", "—"),
        "worst_sentiment": state.get("worst_sentiment", "—"),
        "per_stock": state.get("per_stock", {}),
        "n_stocks": state.get("n_stocks", 0),
        "history": state.get("history", []),
        "timestamp": state.get("timestamp", "—"),
    }


def record_smi_snapshot(state: Dict, market_result: Dict) -> Dict:
    """Append a snapshot to SMI history for time-series display."""
    history = state.get("history", [])
    history.append({
        "smi": market_result.get("market_smi", 0),
        "label": market_result.get("market_label", "NEUTRAL"),
        "timestamp": market_result.get("timestamp", datetime.now().isoformat()),
    })
    state["history"] = history[-500:]  # Keep last 500 snapshots

    # Merge latest market result
    for k, v in market_result.items():
        state[k] = v

    return state


# ── Persistence ──────────────────────────────────

def save_smi_state(state: Dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_smi_state() -> Dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
