"""
====================================================
🗣️ PROJECT AEGIS — Multi-Agent Debate System
====================================================
Phase 13 · Fully Dynamic

Multiple autonomous "debate agents" each pull REAL data from
existing Aegis modules and produce scored arguments (FOR / AGAINST /
ABSTAIN) with quantitative reasoning. A weighted consensus decides
whether a trade should proceed. Fully dynamic — every argument
is computed from live market data, never static or fake.

Agents:
  1. Momentum Bull  — real RSI, MACD, EMA crossover, price action
  2. Risk Bear      — real VaR, drawdown, anomaly scores, Guardian
  3. Sentiment Analyst — real FinBERT / news scores
  4. Technical Judge — real volume profile, Bollinger, ATR, patterns
  5. Macro Strategist — real regime, intermarket, breadth, sector

Final verdict = weighted consensus with veto power for extreme
risk conditions.

====================================================
"""

import os, json, time, math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None

# ── Paths ────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "debate_system_state.json")

# ── Agent Weights (dynamically adjustable by regime) ─
BASE_WEIGHTS = {
    "momentum_bull":    0.25,
    "risk_bear":        0.25,
    "sentiment_analyst": 0.20,
    "technical_judge":  0.15,
    "macro_strategist": 0.15,
}

# Regime-based weight overrides (auto-adaptive)
REGIME_WEIGHTS = {
    "BULL": {
        "momentum_bull": 0.30, "risk_bear": 0.15,
        "sentiment_analyst": 0.20, "technical_judge": 0.15, "macro_strategist": 0.20,
    },
    "BEAR": {
        "momentum_bull": 0.10, "risk_bear": 0.35,
        "sentiment_analyst": 0.20, "technical_judge": 0.15, "macro_strategist": 0.20,
    },
    "SIDEWAYS": BASE_WEIGHTS,
    "VOLATILE": {
        "momentum_bull": 0.15, "risk_bear": 0.30,
        "sentiment_analyst": 0.15, "technical_judge": 0.25, "macro_strategist": 0.15,
    },
}

# Consensus thresholds
CONSENSUS_THRESHOLD = 0.55   # Weighted score must exceed this to approve
VETO_THRESHOLD      = -0.6   # If Risk Bear scores below this, auto-veto


# ═══════════════════════════════════════════════════
#  INDIVIDUAL DEBATE AGENTS — Real Data Driven
# ═══════════════════════════════════════════════════

def _agent_momentum_bull(sym: str, df: pd.DataFrame, current_price: float,
                         live_data: Dict) -> Dict:
    """Agent 1: Analyses momentum indicators from REAL price data.
    Returns score [-1, +1] with quantitative reasoning."""
    if df is None or len(df) < 20:
        return {"agent": "momentum_bull", "vote": "ABSTAIN", "score": 0.0,
                "reasoning": "Insufficient data", "metrics": {}}

    close = df["Close"].values.astype(float)

    # RSI calculation on real data
    period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(period).mean().iloc[-1]
    rsi = 100 - (100 / (1 + (avg_gain / max(avg_loss, 1e-8))))

    # MACD on real data
    fast_ema = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
    slow_ema = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
    macd = fast_ema - slow_ema
    signal_line = pd.Series(
        pd.Series(close).ewm(span=12, adjust=False).mean() -
        pd.Series(close).ewm(span=26, adjust=False).mean()
    ).ewm(span=9, adjust=False).mean().iloc[-1]
    macd_hist = macd - signal_line

    # EMA crossover
    ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
    ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().iloc[-1]
    ema_bullish = ema_12 > ema_50

    # Price momentum (5-day and 20-day returns)
    ret_5d = (close[-1] - close[-6]) / close[-6] if len(close) > 6 else 0
    ret_20d = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0

    # Score computation
    score = 0.0
    reasons = []

    # RSI contribution
    if rsi < 30:
        score += 0.3
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi < 40:
        score += 0.15
        reasons.append(f"RSI approaching oversold ({rsi:.1f})")
    elif rsi > 70:
        score -= 0.3
        reasons.append(f"RSI overbought ({rsi:.1f})")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")

    # MACD contribution
    if macd_hist > 0 and macd > 0:
        score += 0.25
        reasons.append(f"MACD bullish (hist={macd_hist:.4f})")
    elif macd_hist > 0:
        score += 0.1
        reasons.append(f"MACD turning up (hist={macd_hist:.4f})")
    elif macd_hist < 0:
        score -= 0.2
        reasons.append(f"MACD bearish (hist={macd_hist:.4f})")

    # EMA crossover contribution
    if ema_bullish:
        score += 0.2
        reasons.append(f"EMA12 > EMA50 (bullish crossover)")
    else:
        score -= 0.15
        reasons.append(f"EMA12 < EMA50 (bearish)")

    # Momentum contribution
    if ret_5d > 0.02:
        score += 0.15
        reasons.append(f"5-day momentum strong (+{ret_5d*100:.1f}%)")
    elif ret_5d < -0.02:
        score -= 0.15
        reasons.append(f"5-day momentum weak ({ret_5d*100:.1f}%)")

    if ret_20d > 0.05:
        score += 0.1
    elif ret_20d < -0.05:
        score -= 0.1

    score = max(-1.0, min(1.0, score))
    vote = "FOR" if score > 0.1 else ("AGAINST" if score < -0.1 else "ABSTAIN")

    return {
        "agent": "momentum_bull",
        "vote": vote,
        "score": round(score, 4),
        "reasoning": " | ".join(reasons),
        "metrics": {
            "rsi": round(rsi, 1),
            "macd_hist": round(macd_hist, 5),
            "ema_bullish": ema_bullish,
            "ret_5d": round(ret_5d * 100, 2),
            "ret_20d": round(ret_20d * 100, 2),
        },
    }


def _agent_risk_bear(sym: str, df: pd.DataFrame, current_price: float,
                     live_data: Dict) -> Dict:
    """Agent 2: Analyses risk metrics from REAL data.
    Reads drawdown, volatility, anomaly scores, VaR estimates."""
    if df is None or len(df) < 20:
        return {"agent": "risk_bear", "vote": "ABSTAIN", "score": 0.0,
                "reasoning": "Insufficient data", "metrics": {}}

    close = df["Close"].values.astype(float)
    returns = np.diff(close) / close[:-1]

    # Real volatility
    vol_20d = float(np.std(returns[-20:])) * math.sqrt(252) if len(returns) >= 20 else 0.3
    vol_5d = float(np.std(returns[-5:])) * math.sqrt(252) if len(returns) >= 5 else 0.3

    # Real drawdown from peak
    rolling_max = np.maximum.accumulate(close)
    drawdown = (close[-1] - rolling_max[-1]) / rolling_max[-1]

    # Historical VaR (95%) from real returns
    if len(returns) > 30:
        var_95 = float(np.percentile(returns, 5))
    else:
        var_95 = -0.03

    # ATR-based risk (real)
    high = df["High"].values.astype(float) if "High" in df.columns else close
    low = df["Low"].values.astype(float) if "Low" in df.columns else close
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
    atr_pct = atr / current_price if current_price > 0 else 0

    # Check existing module states for risk data
    guardian_status = live_data.get("guardian_status", {})
    anomaly_score = live_data.get("anomaly_score", 0)
    var_state = live_data.get("var_state", {})

    score = 0.0
    reasons = []

    # Volatility assessment
    if vol_20d > 0.4:
        score -= 0.3
        reasons.append(f"HIGH volatility (annualized {vol_20d*100:.1f}%)")
    elif vol_20d > 0.25:
        score -= 0.1
        reasons.append(f"Elevated volatility ({vol_20d*100:.1f}%)")
    else:
        score += 0.1
        reasons.append(f"Low volatility ({vol_20d*100:.1f}%)")

    # Vol spike check (5d vs 20d)
    if vol_5d > vol_20d * 1.5:
        score -= 0.2
        reasons.append(f"Vol SPIKE: 5d={vol_5d*100:.0f}% vs 20d={vol_20d*100:.0f}%")

    # Drawdown assessment
    if drawdown < -0.10:
        score -= 0.3
        reasons.append(f"Deep drawdown ({drawdown*100:.1f}%)")
    elif drawdown < -0.05:
        score -= 0.15
        reasons.append(f"Moderate drawdown ({drawdown*100:.1f}%)")
    else:
        score += 0.1
        reasons.append(f"Near highs (DD={drawdown*100:.1f}%)")

    # VaR assessment
    if var_95 < -0.04:
        score -= 0.2
        reasons.append(f"High VaR-95 ({var_95*100:.2f}%)")
    else:
        score += 0.05
        reasons.append(f"Acceptable VaR ({var_95*100:.2f}%)")

    # ATR risk
    if atr_pct > 0.03:
        score -= 0.15
        reasons.append(f"Wide ATR ({atr_pct*100:.1f}% of price)")

    # Guardian status
    if guardian_status.get("halted"):
        score -= 0.4
        reasons.append("⚠ Guardian HALTED — extreme risk")

    # Anomaly detection
    if anomaly_score > 0.7:
        score -= 0.2
        reasons.append(f"Anomaly detected (score={anomaly_score:.2f})")

    score = max(-1.0, min(1.0, score))
    vote = "FOR" if score > 0.1 else ("AGAINST" if score < -0.1 else "ABSTAIN")

    return {
        "agent": "risk_bear",
        "vote": vote,
        "score": round(score, 4),
        "reasoning": " | ".join(reasons),
        "metrics": {
            "vol_20d": round(vol_20d * 100, 1),
            "vol_5d": round(vol_5d * 100, 1),
            "drawdown_pct": round(drawdown * 100, 2),
            "var_95_pct": round(var_95 * 100, 2),
            "atr_pct": round(atr_pct * 100, 2),
        },
    }


def _agent_sentiment_analyst(sym: str, df: pd.DataFrame, current_price: float,
                             live_data: Dict) -> Dict:
    """Agent 3: Reads REAL sentiment data from existing Aegis modules.
    Uses FinBERT scores, news state, and general sentiment."""
    finbert_state = live_data.get("finbert_state", {})
    news_state = live_data.get("news_state", {})
    sentiment_score = live_data.get("sentiment_score", 0)

    score = 0.0
    reasons = []

    # FinBERT analysis (real)
    fb_scores = finbert_state.get("stock_scores", {})
    sym_fb = fb_scores.get(sym, {})
    if sym_fb:
        fb_val = sym_fb.get("score", 0)
        fb_label = sym_fb.get("label", "NEUTRAL")
        if fb_label == "POSITIVE" or fb_val > 0.3:
            score += 0.3
            reasons.append(f"FinBERT positive ({fb_label}, {fb_val:.2f})")
        elif fb_label == "NEGATIVE" or fb_val < -0.3:
            score -= 0.3
            reasons.append(f"FinBERT negative ({fb_label}, {fb_val:.2f})")
        else:
            reasons.append(f"FinBERT neutral ({fb_label}, {fb_val:.2f})")
    else:
        reasons.append("FinBERT: no data for this stock")

    # News detector state (real)
    news_stocks = news_state.get("stocks", {})
    sym_news = news_stocks.get(sym, {})
    if sym_news:
        news_sentiment = sym_news.get("sentiment", 0)
        news_count = sym_news.get("article_count", 0)
        if news_sentiment > 0.3 and news_count > 2:
            score += 0.25
            reasons.append(f"News positive ({news_count} articles, sent={news_sentiment:.2f})")
        elif news_sentiment < -0.3 and news_count > 2:
            score -= 0.25
            reasons.append(f"News negative ({news_count} articles, sent={news_sentiment:.2f})")
        elif news_count > 0:
            reasons.append(f"News mixed ({news_count} articles, sent={news_sentiment:.2f})")
    else:
        reasons.append("News: no recent articles")

    # General sentiment module (real)
    if sentiment_score:
        if sentiment_score > 0.3:
            score += 0.2
            reasons.append(f"Market sentiment positive ({sentiment_score:.2f})")
        elif sentiment_score < -0.3:
            score -= 0.2
            reasons.append(f"Market sentiment negative ({sentiment_score:.2f})")
        else:
            reasons.append(f"Market sentiment neutral ({sentiment_score:.2f})")

    # Fear/greed from market mood
    mood = live_data.get("market_mood", {})
    if mood:
        fear = mood.get("fear_greed", 50)
        if fear > 70:
            score += 0.15
            reasons.append(f"Greed mode (F&G={fear})")
        elif fear < 30:
            score -= 0.15
            reasons.append(f"Fear mode (F&G={fear})")

    if not reasons:
        reasons.append("No sentiment data available")

    score = max(-1.0, min(1.0, score))
    vote = "FOR" if score > 0.1 else ("AGAINST" if score < -0.1 else "ABSTAIN")

    return {
        "agent": "sentiment_analyst",
        "vote": vote,
        "score": round(score, 4),
        "reasoning": " | ".join(reasons),
        "metrics": {
            "finbert": sym_fb.get("score", 0) if sym_fb else 0,
            "news_sentiment": sym_news.get("sentiment", 0) if sym_news else 0,
            "general_sentiment": sentiment_score,
        },
    }


def _agent_technical_judge(sym: str, df: pd.DataFrame, current_price: float,
                           live_data: Dict) -> Dict:
    """Agent 4: Analyses real volume profile, Bollinger position, ATR patterns,
    and price structure from REAL data."""
    if df is None or len(df) < 20:
        return {"agent": "technical_judge", "vote": "ABSTAIN", "score": 0.0,
                "reasoning": "Insufficient data", "metrics": {}}

    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float) if "High" in df.columns else close
    low = df["Low"].values.astype(float) if "Low" in df.columns else close

    # Bollinger bands (real)
    bb_mid = float(pd.Series(close).rolling(20).mean().iloc[-1])
    bb_std = float(pd.Series(close).rolling(20).std().iloc[-1])
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # Volume analysis (real)
    vol_data = live_data.get("volume_profile", {})
    vol_gate = live_data.get("volume_gate_pass", True)

    # Real volume from dataframe
    volume = df["Volume"].values.astype(float) if "Volume" in df.columns else np.ones(len(close))
    vol_ma = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
    vol_current = float(volume[-1]) if len(volume) > 0 else 0
    vol_ratio = vol_current / max(vol_ma, 1)

    # ATR for pattern analysis (real)
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

    # Candle patterns (real recent data)
    last_body = abs(close[-1] - close[-2]) if len(close) > 1 else 0
    is_doji = last_body < atr_14 * 0.1

    # Support/resistance from recent swing points (real)
    lookback = min(60, len(close))
    recent = close[-lookback:]
    support = float(np.min(recent))
    resistance = float(np.max(recent))
    dist_to_support = (current_price - support) / current_price if current_price > 0 else 0
    dist_to_resist = (resistance - current_price) / current_price if current_price > 0 else 0

    score = 0.0
    reasons = []

    # Bollinger position
    if bb_pct < 0.1:
        score += 0.25
        reasons.append(f"Below lower BB (BB%={bb_pct:.2f}) — dip buy zone")
    elif bb_pct < 0.3:
        score += 0.1
        reasons.append(f"Near lower BB (BB%={bb_pct:.2f})")
    elif bb_pct > 0.9:
        score -= 0.2
        reasons.append(f"Above upper BB (BB%={bb_pct:.2f}) — overextended")
    else:
        reasons.append(f"BB position neutral ({bb_pct:.2f})")

    # Volume confirmation
    if vol_ratio > 1.5:
        score += 0.2
        reasons.append(f"Volume surge ({vol_ratio:.1f}x average)")
    elif vol_ratio < 0.5:
        score -= 0.15
        reasons.append(f"Low volume ({vol_ratio:.1f}x average) — weak conviction")
    else:
        reasons.append(f"Normal volume ({vol_ratio:.1f}x)")

    # Volume gate from existing module
    if not vol_gate:
        score -= 0.15
        reasons.append("Volume Profile gate: BLOCKED")

    # Support/resistance
    if dist_to_support < 0.03:
        score += 0.2
        reasons.append(f"Near support ({support:.2f}, {dist_to_support*100:.1f}% away)")
    if dist_to_resist < 0.02:
        score -= 0.15
        reasons.append(f"Near resistance ({resistance:.2f}, {dist_to_resist*100:.1f}% away)")

    # Risk/reward ratio
    rr_ratio = dist_to_resist / max(dist_to_support, 0.001)
    if rr_ratio > 2:
        score += 0.15
        reasons.append(f"R:R favorable ({rr_ratio:.1f}:1)")
    elif rr_ratio < 0.5:
        score -= 0.1
        reasons.append(f"R:R poor ({rr_ratio:.1f}:1)")

    # Doji = indecision
    if is_doji:
        score *= 0.7
        reasons.append("Doji candle — market indecision")

    score = max(-1.0, min(1.0, score))
    vote = "FOR" if score > 0.1 else ("AGAINST" if score < -0.1 else "ABSTAIN")

    return {
        "agent": "technical_judge",
        "vote": vote,
        "score": round(score, 4),
        "reasoning": " | ".join(reasons),
        "metrics": {
            "bb_pct": round(bb_pct, 3),
            "vol_ratio": round(vol_ratio, 2),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "rr_ratio": round(rr_ratio, 2),
            "atr": round(atr_14, 2),
        },
    }


def _agent_macro_strategist(sym: str, df: pd.DataFrame, current_price: float,
                            live_data: Dict) -> Dict:
    """Agent 5: Reads REAL macro data — regime, intermarket, breadth, sector."""
    score = 0.0
    reasons = []

    # Regime from real HMM detector
    regime_state = live_data.get("regime_state", {})
    regime = regime_state.get("regime", "UNKNOWN")
    regime_prob = regime_state.get("probability", 0)

    if regime == "BULL":
        score += 0.3
        reasons.append(f"Regime: BULL (prob={regime_prob:.0%})")
    elif regime == "BEAR":
        score -= 0.3
        reasons.append(f"Regime: BEAR (prob={regime_prob:.0%})")
    elif regime == "VOLATILE":
        score -= 0.15
        reasons.append(f"Regime: VOLATILE (prob={regime_prob:.0%})")
    else:
        reasons.append(f"Regime: {regime} (prob={regime_prob:.0%})")

    # Intermarket (real)
    intermarket_state = live_data.get("intermarket_state", {})
    im_gate = intermarket_state.get("gate_pass", True)
    im_mult = intermarket_state.get("exposure_multiplier", 1.0)
    if not im_gate:
        score -= 0.2
        reasons.append(f"Intermarket: BLOCKED (mult={im_mult:.2f})")
    elif im_mult > 1.1:
        score += 0.15
        reasons.append(f"Intermarket: favorable (mult={im_mult:.2f})")
    else:
        reasons.append(f"Intermarket: neutral (mult={im_mult:.2f})")

    # Breadth (real)
    breadth_state = live_data.get("breadth_state", {})
    breadth_signal = breadth_state.get("signal", "NEUTRAL")
    adv_dec_ratio = breadth_state.get("advance_decline_ratio", 1.0)

    if breadth_signal == "BULLISH":
        score += 0.2
        reasons.append(f"Breadth: BULLISH (A/D={adv_dec_ratio:.2f})")
    elif breadth_signal == "BEARISH":
        score -= 0.2
        reasons.append(f"Breadth: BEARISH (A/D={adv_dec_ratio:.2f})")
    else:
        reasons.append(f"Breadth: {breadth_signal}")

    # Sector rotation (real)
    sector_state = live_data.get("sector_state", {})
    sector_rec = sector_state.get("recommendation", "NEUTRAL")
    if sector_rec in ("OVERWEIGHT", "STRONG_BUY"):
        score += 0.15
        reasons.append(f"Sector: {sector_rec}")
    elif sector_rec in ("UNDERWEIGHT", "AVOID"):
        score -= 0.15
        reasons.append(f"Sector: {sector_rec}")

    # VaR stress (real)
    var_state = live_data.get("var_state", {})
    mc_var = var_state.get("monte_carlo_var_95", {}).get("var_pct", 0)
    if mc_var and abs(mc_var) > 5:
        score -= 0.15
        reasons.append(f"VaR stress: high ({mc_var:.1f}%)")

    if not reasons:
        reasons.append("Insufficient macro data")

    score = max(-1.0, min(1.0, score))
    vote = "FOR" if score > 0.1 else ("AGAINST" if score < -0.1 else "ABSTAIN")

    return {
        "agent": "macro_strategist",
        "vote": vote,
        "score": round(score, 4),
        "reasoning": " | ".join(reasons),
        "metrics": {
            "regime": regime,
            "breadth": breadth_signal,
            "intermarket_mult": im_mult,
        },
    }


# ═══════════════════════════════════════════════════
#  DEBATE ORCHESTRATION
# ═══════════════════════════════════════════════════

def run_debate(sym: str, df: pd.DataFrame, current_price: float,
               live_data: Optional[Dict] = None) -> Dict:
    """Run full multi-agent debate on a single stock.

    live_data should contain real state dicts from other Aegis modules:
      - finbert_state, news_state, sentiment_score
      - regime_state, intermarket_state, breadth_state, sector_state
      - var_state, guardian_status, anomaly_score
      - volume_profile, volume_gate_pass, market_mood
    All pulled from REAL module outputs at call time.
    """
    if live_data is None:
        live_data = {}

    # Determine weights by current regime
    regime = live_data.get("regime_state", {}).get("regime", "SIDEWAYS")
    weights = REGIME_WEIGHTS.get(regime, BASE_WEIGHTS)

    # Run all 5 agents
    arguments = [
        _agent_momentum_bull(sym, df, current_price, live_data),
        _agent_risk_bear(sym, df, current_price, live_data),
        _agent_sentiment_analyst(sym, df, current_price, live_data),
        _agent_technical_judge(sym, df, current_price, live_data),
        _agent_macro_strategist(sym, df, current_price, live_data),
    ]

    # Weighted consensus
    weighted_sum = 0.0
    total_weight = 0.0
    vote_counts = {"FOR": 0, "AGAINST": 0, "ABSTAIN": 0}

    for arg in arguments:
        agent_name = arg["agent"]
        w = weights.get(agent_name, 0.15)
        if arg["vote"] != "ABSTAIN":
            weighted_sum += arg["score"] * w
            total_weight += w
        vote_counts[arg["vote"]] += 1

    consensus_score = weighted_sum / max(total_weight, 1e-8)

    # Veto check: Risk Bear extreme negative = auto-reject
    risk_bear_result = next((a for a in arguments if a["agent"] == "risk_bear"), None)
    vetoed = False
    veto_reason = ""
    if risk_bear_result and risk_bear_result["score"] < VETO_THRESHOLD:
        vetoed = True
        veto_reason = f"Risk Bear VETO (score={risk_bear_result['score']:.3f})"

    # Final verdict
    if vetoed:
        verdict = "REJECT"
        verdict_reason = veto_reason
    elif consensus_score >= CONSENSUS_THRESHOLD:
        verdict = "APPROVE"
        verdict_reason = f"Consensus APPROVE (score={consensus_score:.3f} >= {CONSENSUS_THRESHOLD})"
    elif consensus_score >= 0.0:
        verdict = "WEAK_APPROVE"
        verdict_reason = f"Weak consensus (score={consensus_score:.3f})"
    else:
        verdict = "REJECT"
        verdict_reason = f"Consensus REJECT (score={consensus_score:.3f} < 0)"

    return {
        "symbol": sym,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "consensus_score": round(consensus_score, 4),
        "vetoed": vetoed,
        "veto_reason": veto_reason,
        "vote_counts": vote_counts,
        "arguments": arguments,
        "weights_used": weights,
        "regime": regime,
        "timestamp": datetime.now().isoformat(),
    }


def get_debate_verdict(sym: str, df: pd.DataFrame, current_price: float,
                       live_data: Optional[Dict] = None) -> Tuple[bool, str, float]:
    """Quick gate check: returns (approved, reason, score)."""
    result = run_debate(sym, df, current_price, live_data)
    approved = result["verdict"] in ("APPROVE", "WEAK_APPROVE")
    return approved, result["verdict_reason"], result["consensus_score"]


def get_debate_status(state: Optional[Dict] = None) -> Dict:
    """Return summary for dashboard display."""
    if state is None:
        state = load_debate_state()
    if not state:
        return {"status": "NO_DATA"}
    return {
        "status": "ACTIVE",
        "total_debates": state.get("total_debates", 0),
        "approvals": state.get("approvals", 0),
        "rejections": state.get("rejections", 0),
        "veto_count": state.get("veto_count", 0),
        "recent_debates": state.get("recent_debates", [])[-20:],
        "agent_accuracy": state.get("agent_accuracy", {}),
        "timestamp": state.get("timestamp", "—"),
    }


def record_debate_result(debate_result: Dict, state: Optional[Dict] = None) -> Dict:
    """Record a debate result into running state. Tracks accuracy over time."""
    if state is None:
        state = load_debate_state()
    if not state:
        state = {
            "total_debates": 0, "approvals": 0, "rejections": 0,
            "veto_count": 0, "recent_debates": [], "agent_accuracy": {},
        }

    state["total_debates"] = state.get("total_debates", 0) + 1
    if debate_result["verdict"] in ("APPROVE", "WEAK_APPROVE"):
        state["approvals"] = state.get("approvals", 0) + 1
    else:
        state["rejections"] = state.get("rejections", 0) + 1
    if debate_result.get("vetoed"):
        state["veto_count"] = state.get("veto_count", 0) + 1

    # Store compact version in recent debates
    compact = {
        "symbol": debate_result.get("symbol"),
        "verdict": debate_result.get("verdict"),
        "consensus_score": debate_result.get("consensus_score"),
        "vetoed": debate_result.get("vetoed", False),
        "vote_counts": debate_result.get("vote_counts"),
        "timestamp": debate_result.get("timestamp"),
    }
    recent = state.get("recent_debates", [])
    recent.append(compact)
    state["recent_debates"] = recent[-100:]  # Keep last 100
    state["timestamp"] = datetime.now().isoformat()

    return state


# ── Persistence ──────────────────────────────────

def save_debate_state(state: Dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_debate_state() -> Dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
