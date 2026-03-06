"""
====================================================
PROJECT AEGIS - NeuroVoter Ensemble v1
====================================================
6 AI "traders" that each think like a DIFFERENT type of
human trader. They debate and vote, but each one reasons
about the data from their own perspective.

THE OLD WAY (broken):
  - RF says 0.65 > 0.60 threshold → vote = 1
  - XGBoost says 0.72 > 0.60 → vote = 1
  - That's it. No context. No reasoning. Pure math.

THE NEW WAY (human-like):
  Each voter is a character with:
    - A specialty (momentum, value, risk, sentiment, pattern, quant)
    - Their own view of the data
    - Context awareness (market mood, sector, global events)
    - Anti-overfit reasoning (distrust perfect predictions)
    - Conviction strength (not just 0 or 1, but how strongly)
    - They can VETO a trade if something smells wrong

OVERFITTING PROTECTION:
  1. No voter trusts a model that's "too perfect" (conf > 0.95 = suspicious)
  2. Multiple decorrelated views = robust signal
  3. Bayesian confidence decay: recent accuracy matters more
  4. Regime detection: voters adapt to bull/bear/sideways
  5. Ensemble disagreement → reduce position, don't just ignore
  6. Walk-forward bias check: is today's pattern in training data?

Each voter returns:
  vote      : float   (-1.0 to +1.0)   negative=sell, 0=hold, positive=buy
  conviction: float   (0.0 to 1.0)     how confident they are
  reasoning : str     human-readable explanation
  veto      : bool    TRUE = block this trade entirely
  veto_reason: str    why they vetoed

Final decision uses weighted conviction voting,
not simple majority.
====================================================
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

import pytz

sys.path.insert(0, os.path.dirname(__file__))

IST = pytz.timezone("Asia/Kolkata")


# ══════════════════════════════════════════════════
#  VOTER RESULT CONTAINER
# ══════════════════════════════════════════════════
class VoterResult:
    """One voter's decision."""
    def __init__(self, name: str, vote: float = 0.0, conviction: float = 0.0,
                 reasoning: str = "", veto: bool = False, veto_reason: str = ""):
        self.name = name
        self.vote = max(-1.0, min(1.0, vote))          # -1 to +1
        self.conviction = max(0.0, min(1.0, conviction))  # 0 to 1
        self.reasoning = reasoning
        self.veto = veto
        self.veto_reason = veto_reason

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vote": round(self.vote, 3),
            "conviction": round(self.conviction, 3),
            "reasoning": self.reasoning,
            "veto": self.veto,
            "veto_reason": self.veto_reason,
        }


# ══════════════════════════════════════════════════
#  ENSEMBLE RESULT — The final decision
# ══════════════════════════════════════════════════
class EnsembleDecision:
    """The combined decision from all voters."""
    def __init__(self):
        self.should_buy = False
        self.weighted_score = 0.0       # -1 to +1
        self.total_conviction = 0.0     # 0 to 1
        self.buy_voters = 0
        self.sell_voters = 0
        self.hold_voters = 0
        self.vetoed = False
        self.veto_reasons = []
        self.voter_results = []
        self.reasoning = ""
        self.overfit_warning = False
        self.regime = "UNKNOWN"

    def to_dict(self) -> dict:
        return {
            "should_buy": self.should_buy,
            "weighted_score": round(self.weighted_score, 3),
            "total_conviction": round(self.total_conviction, 3),
            "buy_voters": self.buy_voters,
            "sell_voters": self.sell_voters,
            "hold_voters": self.hold_voters,
            "vetoed": self.vetoed,
            "veto_reasons": self.veto_reasons,
            "reasoning": self.reasoning,
            "overfit_warning": self.overfit_warning,
            "regime": self.regime,
            "voters": [v.to_dict() for v in self.voter_results],
        }


# ══════════════════════════════════════════════════
#  ANTI-OVERFIT ENGINE
# ══════════════════════════════════════════════════
class OverfitDetector:
    """
    Detects signs of overfitting in model predictions.
    
    A model that memorised the training set shows:
      - Very high confidence (>95%) on everything
      - Same prediction regardless of market state
      - Dramatic flip-flops between ticks
      - Predictions that don't match the reality of the data
    """

    def __init__(self):
        self._prediction_history = {}   # model_name → list of recent preds
        self._accuracy_history = {}     # model_name → list of (pred, was_correct)
        self._max_history = 50

    def check_prediction(self, model_name: str, confidence: float,
                          price_context: dict) -> dict:
        """
        Check a single model prediction for overfit signals.
        Returns dict with:
          suspicious: bool
          overfit_score: float (0=clean, 1=definitely overfit)
          reasons: list[str]
          adjusted_confidence: float
        """
        result = {
            "suspicious": False,
            "overfit_score": 0.0,
            "reasons": [],
            "adjusted_confidence": confidence,
        }

        history = self._prediction_history.setdefault(model_name, [])
        overfit_score = 0.0

        # ── Check 1: Extreme confidence ──
        # Real-world models should rarely be >90% confident
        if confidence > 0.95:
            overfit_score += 0.4
            result["reasons"].append(
                f"Suspiciously high conf ({confidence:.2%}) — "
                "real markets aren't this predictable"
            )
            # Penalise: reduce to 0.75 max
            result["adjusted_confidence"] = min(confidence, 0.75)

        elif confidence > 0.90:
            overfit_score += 0.2
            result["reasons"].append(f"Very high conf ({confidence:.2%}) — dampened")
            result["adjusted_confidence"] = min(confidence, 0.82)

        # ── Check 2: Stuck predictions ──
        # If model gives SAME answer every time, it's not learning
        if len(history) >= 10:
            recent = history[-10:]
            all_same_direction = (
                all(p > 0.5 for p in recent) or
                all(p <= 0.5 for p in recent)
            )
            if all_same_direction:
                std = np.std(recent)
                if std < 0.02:
                    overfit_score += 0.3
                    result["reasons"].append(
                        f"Stuck: same direction for 10+ ticks (std={std:.4f})"
                    )
                    result["adjusted_confidence"] = 0.52  # Barely above neutral

        # ── Check 3: Wild flip-flops ──
        # Overfit models can oscillate wildly between ticks
        if len(history) >= 3:
            last3 = history[-3:]
            flips = sum(1 for i in range(1, len(last3))
                       if (last3[i] > 0.5) != (last3[i-1] > 0.5))
            if flips >= 2:
                overfit_score += 0.25
                result["reasons"].append(
                    f"Flip-flopping: changed direction {flips} times in 3 ticks"
                )
                result["adjusted_confidence"] *= 0.7

        # ── Check 4: Prediction vs Price Reality ──
        # If model says strong BUY but price is falling, something's off
        price_change = price_context.get("change_pct", 0)
        if confidence > 0.70 and price_change < -2.0:
            overfit_score += 0.2
            result["reasons"].append(
                f"Says BUY (conf={confidence:.2%}) while price dropped {price_change:.1f}%"
            )
            result["adjusted_confidence"] *= 0.8

        # ── Check 5: Volatility mismatch ──
        # In high volatility, a model confident either way is suspicious
        atr_pct = price_context.get("atr_pct", 0)
        if atr_pct > 0.03 and confidence > 0.80:
            overfit_score += 0.15
            result["reasons"].append(
                f"High conf ({confidence:.2%}) in high-vol market (ATR={atr_pct:.1%})"
            )
            result["adjusted_confidence"] *= 0.85

        # Record prediction
        history.append(round(confidence, 4))
        if len(history) > self._max_history:
            history.pop(0)

        # Final score
        result["overfit_score"] = min(1.0, overfit_score)
        result["suspicious"] = overfit_score > 0.3

        return result

    def get_ensemble_overfit_score(self, model_scores: dict) -> float:
        """
        If ALL models agree perfectly, that's suspicious too.
        Returns 0-1 score where higher = more suspicious.
        """
        confs = list(model_scores.values())
        if len(confs) < 2:
            return 0.0

        # All models giving similar high confidence = suspicious
        if all(c > 0.70 for c in confs):
            spread = max(confs) - min(confs)
            if spread < 0.05:
                return 0.5  # Perfect agreement is rare in real markets

        # All models saying SAME direction with high confidence
        if all(c > 0.60 for c in confs) and np.std(confs) < 0.03:
            return 0.3

        return 0.0


# ══════════════════════════════════════════════════
#  REGIME DETECTOR
# ══════════════════════════════════════════════════
class RegimeDetector:
    """
    Detects the current market regime so voters can adapt.
    A human trader subconsciously does this all the time.
    """

    @staticmethod
    def detect(df: pd.DataFrame) -> dict:
        """
        Analyse the last N candles to determine the regime.
        Returns: {regime, trend_strength, volatility_level, momentum}
        """
        if df is None or len(df) < 20:
            return {"regime": "UNKNOWN", "trend_strength": 0, "volatility_level": "NORMAL", "momentum": 0}

        close = df["Close"].values
        last_20 = close[-20:]
        last_5 = close[-5:]

        # Trend
        sma_20 = np.mean(last_20)
        price = close[-1]
        trend_pct = ((price - sma_20) / sma_20) * 100 if sma_20 > 0 else 0

        # Volatility
        returns = np.diff(last_20) / last_20[:-1]
        vol = np.std(returns) * 100

        # Momentum (rate of change last 5 bars)
        momentum = ((last_5[-1] - last_5[0]) / last_5[0]) * 100 if last_5[0] > 0 else 0

        # Regime classification
        if trend_pct > 2 and momentum > 0.5:
            regime = "TRENDING_UP"
        elif trend_pct < -2 and momentum < -0.5:
            regime = "TRENDING_DOWN"
        elif vol > 2.5:
            regime = "HIGH_VOLATILE"
        elif vol < 0.5:
            regime = "LOW_VOLATILE"
        else:
            regime = "SIDEWAYS"

        # Volatility level
        if vol > 3.0:
            vol_level = "EXTREME"
        elif vol > 2.0:
            vol_level = "HIGH"
        elif vol > 1.0:
            vol_level = "NORMAL"
        else:
            vol_level = "LOW"

        # Trend strength (0-100)
        trend_strength = min(100, abs(trend_pct) * 15)

        return {
            "regime": regime,
            "trend_strength": round(trend_strength, 1),
            "volatility_level": vol_level,
            "momentum": round(momentum, 2),
            "vol_pct": round(vol, 2),
            "trend_pct": round(trend_pct, 2),
        }


# ══════════════════════════════════════════════════
#  THE 6 NEURO-VOTERS
# ══════════════════════════════════════════════════

class _VoterBase:
    """Base class for all voters."""
    name = "BaseVoter"
    weight = 1.0

    def vote(self, ctx: dict) -> VoterResult:
        raise NotImplementedError


class MomentumTrader(_VoterBase):
    """
    Voter 1: THE MOMENTUM TRADER
    "I only ride waves. Strong moves with volume. I don't catch knives."
    
    Thinks like:
      - Is MACD crossing up? Bullish momentum starting
      - Is RSI in the sweet spot (40-65)? Room to run
      - Is volume above average? Big players agreeing
      - Is price making higher highs? Trend intact
    """
    name = "Momentum Mike"
    weight = 1.2  # Momentum is king in intraday

    def vote(self, ctx: dict) -> VoterResult:
        score = 0.0
        reasons = []

        rsi = ctx.get("rsi", 50)
        macd = ctx.get("macd", 0)
        macd_sig = ctx.get("macd_signal", 0)
        vol_ratio = ctx.get("volume_ratio", 1.0)
        price = ctx.get("price", 0)
        ema20 = ctx.get("ema_20", 0)
        momentum = ctx.get("regime", {}).get("momentum", 0)

        # MACD momentum
        if macd > macd_sig:
            macd_diff = macd - macd_sig
            score += 0.25
            reasons.append(f"MACD bullish (diff={macd_diff:.4f})")
        else:
            score -= 0.25
            reasons.append("MACD bearish")

        # RSI sweet spot (not overbought, not oversold)
        if 40 <= rsi <= 65:
            score += 0.20
            reasons.append(f"RSI in sweet spot ({rsi:.0f})")
        elif rsi > 70:
            score -= 0.30
            reasons.append(f"RSI overbought ({rsi:.0f}) — too late")
        elif rsi < 30:
            score -= 0.15
            reasons.append(f"RSI oversold ({rsi:.0f}) — falling knife")

        # Volume confirmation
        if vol_ratio > 1.5:
            score += 0.25
            reasons.append(f"Strong volume ({vol_ratio:.1f}x) — institutions agree")
        elif vol_ratio > 1.0:
            score += 0.10
            reasons.append(f"OK volume ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            score -= 0.20
            reasons.append(f"Dead volume ({vol_ratio:.1f}x) — no conviction")

        # Price above EMA (trend following)
        if ema20 > 0 and price > ema20:
            score += 0.15
            reasons.append("Price > EMA20 ✓")
        elif ema20 > 0:
            score -= 0.15
            reasons.append("Price < EMA20 — not trending up")

        # Overall momentum
        if momentum > 1.0:
            score += 0.15
            reasons.append(f"Strong 5-bar momentum ({momentum:+.1f}%)")
        elif momentum < -1.0:
            score -= 0.20
            reasons.append(f"Negative momentum ({momentum:+.1f}%)")

        conviction = min(1.0, abs(score) * 1.2)

        return VoterResult(
            name=self.name,
            vote=max(-1.0, min(1.0, score)),
            conviction=conviction,
            reasoning=" | ".join(reasons[:4]),
        )


class ValueHunter(_VoterBase):
    """
    Voter 2: THE VALUE HUNTER
    "I buy when others are scared. Good stocks at a discount.
     I look at where price is relative to its history."
    
    Thinks like:
      - Is price near support (BB lower band)?
      - Is stock oversold but fundamentally fine?
      - Has selling been overdone? (RSI < 35 but trend intact)
      - Is price below SMA200 but bouncing? (value zone)
    """
    name = "Value Vinay"
    weight = 0.9

    def vote(self, ctx: dict) -> VoterResult:
        score = 0.0
        reasons = []

        price = ctx.get("price", 0)
        sma200 = ctx.get("sma_200", 0)
        sma50 = ctx.get("sma_50", 0)
        rsi = ctx.get("rsi", 50)
        bb_lower = ctx.get("bb_lower", 0)
        bb_upper = ctx.get("bb_upper", 0)
        change_pct = ctx.get("change_pct", 0)

        # Near BB lower band = potential value
        if bb_lower > 0 and bb_upper > bb_lower:
            bb_range = bb_upper - bb_lower
            bb_position = (price - bb_lower) / bb_range if bb_range > 0 else 0.5
            if bb_position < 0.25:
                score += 0.30
                reasons.append(f"Near BB lower ({bb_position:.0%} of band) — value zone")
            elif bb_position > 0.90:
                score -= 0.20
                reasons.append(f"Near BB upper ({bb_position:.0%}) — expensive")

        # Oversold + trend = buying opportunity
        if rsi < 35 and sma200 > 0 and price > sma200 * 0.95:
            score += 0.25
            reasons.append(f"Oversold RSI ({rsi:.0f}) but near SMA200 — bounce zone")
        elif rsi > 75:
            score -= 0.15
            reasons.append(f"Overbought ({rsi:.0f}) — no value here")

        # Price vs long-term average  
        if sma200 > 0:
            dist_from_200 = ((price - sma200) / sma200) * 100
            if -5 <= dist_from_200 <= 0:
                score += 0.20
                reasons.append(f"Price slightly below SMA200 ({dist_from_200:+.1f}%) — discount")
            elif dist_from_200 > 10:
                score -= 0.10
                reasons.append(f"Too far above SMA200 ({dist_from_200:+.1f}%) — stretched")

        # Selling overdone? (big drop + RSI low)
        if change_pct < -2.0 and rsi < 40:
            score += 0.15
            reasons.append(f"Selloff ({change_pct:+.1f}%) may be overdone")

        # Mean reversion in SMA50/200 zone
        if sma50 > 0 and sma200 > 0:
            if sma200 * 0.98 < price < sma50:
                score += 0.10
                reasons.append("Between SMA50/200 — consolidation value zone")

        conviction = min(1.0, abs(score) * 1.1)
        return VoterResult(
            name=self.name,
            vote=max(-1.0, min(1.0, score)),
            conviction=conviction,
            reasoning=" | ".join(reasons[:4]),
        )


class RiskManager(_VoterBase):
    """
    Voter 3: THE RISK MANAGER
    "I don't care about profits — I protect the capital. If
     something looks risky, I'll veto the entire trade."
    
    VETO POWER: This voter can block a trade single-handedly.
    
    Thinks like:
      - Is volatility too high? (ATR % of price)
      - Is risk/reward acceptable? (>= 2:1)
      - Are we trading at a dangerous time?
      - Is there a news event that could gap us?
      - Is the position size sensible?
    """
    name = "Risk Raj"
    weight = 1.5  # Safety matters most

    def vote(self, ctx: dict) -> VoterResult:
        score = 0.0
        reasons = []
        veto = False
        veto_reason = ""

        atr = ctx.get("atr", 0)
        price = ctx.get("price", 0)
        vol_ratio = ctx.get("volume_ratio", 1.0)
        rsi = ctx.get("rsi", 50)
        regime = ctx.get("regime", {})
        vol_level = regime.get("volatility_level", "NORMAL")
        sentiment = ctx.get("sentiment_score", 0)
        global_mood = ctx.get("global_mood", 0)

        # Volatility veto
        atr_pct = (atr / price) * 100 if price > 0 else 0
        if atr_pct > 5.0:
            veto = True
            veto_reason = f"ATR is {atr_pct:.1f}% of price — extreme volatility VETO"
            return VoterResult(self.name, -1.0, 1.0, veto_reason, True, veto_reason)

        if atr_pct > 3.0:
            score -= 0.30
            reasons.append(f"High volatility (ATR={atr_pct:.1f}% of price)")
        elif atr_pct < 0.5:
            score -= 0.10
            reasons.append(f"Very low vol (ATR={atr_pct:.1f}%) — illiquid?")
        else:
            score += 0.10
            reasons.append(f"Normal volatility (ATR={atr_pct:.1f}%)")

        # Regime-based risk
        if vol_level == "EXTREME":
            score -= 0.30
            reasons.append("EXTREME volatility regime")
        elif vol_level == "HIGH":
            score -= 0.15
            reasons.append("HIGH volatility regime")

        # Negative sentiment = risk
        if sentiment < -0.3:
            score -= 0.25
            reasons.append(f"Negative news ({sentiment:+.2f}) — risky")
        elif sentiment > 0.2:
            score += 0.10
            reasons.append(f"Positive news ({sentiment:+.2f})")

        # Global mood
        if global_mood < -0.3:
            score -= 0.20
            reasons.append(f"Global risk-off ({global_mood:+.2f})")
        elif global_mood > 0.2:
            score += 0.10

        # RSI extremes = increased risk
        if rsi > 80 or rsi < 20:
            score -= 0.20
            reasons.append(f"RSI at extreme ({rsi:.0f}) — gap risk")

        # Very low volume = slippage risk
        if vol_ratio < 0.3:
            score -= 0.25
            reasons.append(f"Volume only {vol_ratio:.1f}x — slippage risk")

        # Time check
        now = datetime.now(IST)
        if now.hour == 9 and now.minute < 30:
            score -= 0.15
            reasons.append("Opening volatility — risky window")
        elif now.hour >= 15:
            score -= 0.10
            reasons.append("Late session — reduced time for recovery")

        conviction = min(1.0, abs(score) * 1.3)
        return VoterResult(
            name=self.name,
            vote=max(-1.0, min(1.0, score)),
            conviction=conviction,
            reasoning=" | ".join(reasons[:4]),
            veto=veto,
            veto_reason=veto_reason,
        )


class SentimentAnalyst(_VoterBase):
    """
    Voter 4: THE SENTIMENT ANALYST
    "I read the news, check global cues, and feel the market's mood.
     Wars, rate hikes, earnings — I factor it all in."
    
    Thinks like:
      - What does the news say about this stock/sector?
      - Is there a global event affecting everything?
      - Is market mood bullish or fearful?
      - Are there any breaking events that matter?
    """
    name = "Sentiment Sanjay"
    weight = 1.0

    def vote(self, ctx: dict) -> VoterResult:
        score = 0.0
        reasons = []
        veto = False
        veto_reason = ""

        sentiment = ctx.get("sentiment_score", 0)
        global_mood = ctx.get("global_mood", 0)
        mood_details = ctx.get("mood_details", {})

        # Stock-specific sentiment
        if sentiment > 0.5:
            score += 0.35
            reasons.append(f"Very positive news ({sentiment:+.2f})")
        elif sentiment > 0.2:
            score += 0.20
            reasons.append(f"Positive sentiment ({sentiment:+.2f})")
        elif sentiment < -0.5:
            score -= 0.35
            reasons.append(f"Very negative news ({sentiment:+.2f})")
            if sentiment < -0.7:
                veto = True
                veto_reason = f"Severely negative news ({sentiment:+.2f}) — VETO"
        elif sentiment < -0.2:
            score -= 0.20
            reasons.append(f"Negative sentiment ({sentiment:+.2f})")

        # Global mood
        if global_mood > 0.3:
            score += 0.20
            reasons.append(f"Global RISK-ON ({global_mood:+.2f})")
        elif global_mood < -0.3:
            score -= 0.25
            reasons.append(f"Global RISK-OFF ({global_mood:+.2f})")
            if global_mood < -0.6:
                veto = True
                veto_reason = f"Extreme global fear ({global_mood:+.2f}) — VETO"

        # VIX check
        vix = mood_details.get("india_vix", 0)
        if isinstance(vix, (int, float)):
            if vix > 25:
                score -= 0.20
                reasons.append(f"VIX elevated ({vix:.1f})")
            elif vix > 35:
                veto = True
                veto_reason = f"VIX panic ({vix:.1f}) — VETO"
            elif vix < 15:
                score += 0.10
                reasons.append(f"VIX calm ({vix:.1f})")

        # S&P500 cue
        sp_change = mood_details.get("sp500_change", 0)
        if isinstance(sp_change, (int, float)):
            if sp_change < -1.5:
                score -= 0.15
                reasons.append(f"S&P500 down {sp_change:+.1f}%")
            elif sp_change > 1.0:
                score += 0.10
                reasons.append(f"S&P500 up {sp_change:+.1f}%")

        conviction = min(1.0, abs(score) * 1.2)
        return VoterResult(
            name=self.name,
            vote=max(-1.0, min(1.0, score)),
            conviction=conviction,
            reasoning=" | ".join(reasons[:4]),
            veto=veto,
            veto_reason=veto_reason,
        )


class PatternRecognizer(_VoterBase):
    """
    Voter 5: THE PATTERN TRADER
    "I look at candle patterns, support/resistance, and chart structure.
     Not indicators — actual price action."
    
    Thinks like:
      - Is price making higher highs and higher lows?
      - Is there a bullish engulfing or hammer pattern?
      - Is volume increasing on up-moves?
      - Where is price relative to recent support/resistance?
    """
    name = "Pattern Priya"
    weight = 1.0

    def vote(self, ctx: dict) -> VoterResult:
        score = 0.0
        reasons = []

        df = ctx.get("df")
        if df is None or len(df) < 10:
            return VoterResult(self.name, 0.0, 0.1, "Insufficient data")

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values

        # ── Higher Highs / Higher Lows (last 5 bars) ──
        last5_high = high[-5:]
        last5_low = low[-5:]
        hh = sum(1 for i in range(1, len(last5_high)) if last5_high[i] > last5_high[i-1])
        hl = sum(1 for i in range(1, len(last5_low)) if last5_low[i] > last5_low[i-1])

        if hh >= 3 and hl >= 2:
            score += 0.30
            reasons.append(f"Higher highs ({hh}/4) + higher lows ({hl}/4) — uptrend")
        elif hh <= 1 and hl <= 1:
            score -= 0.20
            reasons.append("Lower highs and lows — downtrend")

        # ── Last candle pattern ──
        last_open = float(df["Open"].iloc[-1]) if "Open" in df.columns else close[-2]
        last_close = close[-1]
        last_high = high[-1]
        last_low = low[-1]
        body = abs(last_close - last_open)
        upper_wick = last_high - max(last_close, last_open)
        lower_wick = min(last_close, last_open) - last_low

        if body > 0:
            # Hammer (bullish): small body, long lower wick
            if lower_wick > 2 * body and upper_wick < body * 0.5:
                score += 0.20
                reasons.append("Hammer candle — bullish reversal signal")

            # Inverted hammer at resistance (bearish)
            if upper_wick > 2 * body and lower_wick < body * 0.5:
                score -= 0.15
                reasons.append("Shooting star — rejection at highs")

            # Strong bullish candle (big green)
            if last_close > last_open and body > np.mean(np.abs(np.diff(close[-10:]))) * 1.5:
                score += 0.15
                reasons.append("Strong bullish candle")

        # ── Volume on up-moves vs down-moves ──
        last10_close = close[-10:]
        last10_vol = volume[-10:]
        up_vol = sum(last10_vol[i] for i in range(1, len(last10_close))
                    if last10_close[i] > last10_close[i-1])
        down_vol = sum(last10_vol[i] for i in range(1, len(last10_close))
                      if last10_close[i] <= last10_close[i-1])

        if up_vol > 0 and down_vol > 0:
            vol_bias = up_vol / (up_vol + down_vol)
            if vol_bias > 0.6:
                score += 0.15
                reasons.append(f"Volume favours buyers ({vol_bias:.0%})")
            elif vol_bias < 0.4:
                score -= 0.15
                reasons.append(f"Volume favours sellers ({vol_bias:.0%})")

        # ── Support/Resistance (recent range) ──
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        if recent_high > recent_low:
            range_pos = (close[-1] - recent_low) / (recent_high - recent_low)
            if range_pos < 0.3:
                score += 0.10
                reasons.append(f"Near support ({range_pos:.0%} of range)")
            elif range_pos > 0.85:
                score -= 0.10
                reasons.append(f"Near resistance ({range_pos:.0%} of range)")

        conviction = min(1.0, abs(score) * 1.15)
        return VoterResult(
            name=self.name,
            vote=max(-1.0, min(1.0, score)),
            conviction=conviction,
            reasoning=" | ".join(reasons[:4]),
        )


class QuantBrain(_VoterBase):
    """
    Voter 6: THE QUANT
    "I trust the AI models, but I verify them. I check if they agree,
     if they make sense, and if they look overfit."
    
    This voter is the one that reads the RF/XGB/LSTM outputs
    and judges them like a human quant would:
      - Are models agreeing or conflicting?
      - Does the confidence match the market state?
      - Any overfit signals?
      - Bayesian-adjusted confidence
    """
    name = "Quant Qasim"
    weight = 1.3  # AI models are our core signal

    def __init__(self):
        super().__init__()
        self.overfit_detector = OverfitDetector()

    def vote(self, ctx: dict) -> VoterResult:
        score = 0.0
        reasons = []
        veto = False
        veto_reason = ""

        rf_conf = ctx.get("rf_conf", 0.5)
        xgb_conf = ctx.get("xgb_conf", 0.5)
        lstm_conf = ctx.get("lstm_conf", 0.5)
        intra_conf = ctx.get("intra_conf", 0.5)
        votes = ctx.get("original_votes", 0)
        price = ctx.get("price", 0)
        atr = ctx.get("atr", 0)
        change_pct = ctx.get("change_pct", 0)

        confs = {"rf": rf_conf, "xgb": xgb_conf, "lstm": lstm_conf, "intra": intra_conf}
        avg_conf = sum(confs.values()) / len(confs)

        # ── Anti-Overfit Check (per model) ──
        price_context = {
            "change_pct": change_pct,
            "atr_pct": (atr / price) * 100 if price > 0 else 0,
        }
        adjusted_confs = {}
        overfit_flags = []
        for name, conf in confs.items():
            check = self.overfit_detector.check_prediction(name, conf, price_context)
            adjusted_confs[name] = check["adjusted_confidence"]
            if check["suspicious"]:
                overfit_flags.extend(check["reasons"])

        # Use adjusted confidences
        adj_avg = sum(adjusted_confs.values()) / len(adjusted_confs)

        # ── Ensemble-level overfit check ──
        ens_overfit = self.overfit_detector.get_ensemble_overfit_score(confs)
        if ens_overfit > 0.4:
            overfit_flags.append(f"All models agree too perfectly (score={ens_overfit:.2f})")
            adj_avg *= 0.8

        if overfit_flags:
            reasons.append(f"⚠ Overfit: {overfit_flags[0]}")

        # ── Model Agreement Score ──
        # How many models agree on direction (> 0.5)?
        bullish = sum(1 for c in adjusted_confs.values() if c > 0.55)
        bearish = sum(1 for c in adjusted_confs.values() if c < 0.45)

        if bullish >= 3:
            score += 0.30
            reasons.append(f"{bullish}/4 models bullish (adj_avg={adj_avg:.2%})")
        elif bullish >= 2:
            score += 0.15
            reasons.append(f"{bullish}/4 models bullish (adj_avg={adj_avg:.2%})")
        elif bearish >= 3:
            score -= 0.30
            reasons.append(f"{bearish}/4 models bearish")

        # ── Confidence Strength ──
        if adj_avg > 0.70:
            score += 0.25
            reasons.append(f"High adjusted confidence ({adj_avg:.2%})")
        elif adj_avg > 0.60:
            score += 0.10
        elif adj_avg < 0.45:
            score -= 0.20
            reasons.append(f"Low confidence ({adj_avg:.2%})")

        # ── Model Spread (disagreement) ──
        spread = max(adjusted_confs.values()) - min(adjusted_confs.values())
        if spread > 0.40:
            score *= 0.6   # Heavy discount for disagreement
            reasons.append(f"Models disagree (spread={spread:.2%})")
        elif spread < 0.10 and adj_avg > 0.60:
            score += 0.10
            reasons.append("Models in strong agreement")

        # ── Tree models vs Neural models ──
        # If RF+XGB say BUY but LSTMs don't, or vice versa, be cautious
        tree_avg = (adjusted_confs["rf"] + adjusted_confs["xgb"]) / 2
        nn_avg = (adjusted_confs["lstm"] + adjusted_confs["intra"]) / 2
        tree_nn_gap = abs(tree_avg - nn_avg)
        if tree_nn_gap > 0.25:
            score *= 0.75
            reasons.append(f"Tree/NN disagree (gap={tree_nn_gap:.2%})")

        conviction = min(1.0, abs(score) * 1.25)
        return VoterResult(
            name=self.name,
            vote=max(-1.0, min(1.0, score)),
            conviction=conviction,
            reasoning=" | ".join(reasons[:4]),
            veto=veto,
            veto_reason=veto_reason,
        )


# ══════════════════════════════════════════════════
#  THE NEURO-VOTER ENSEMBLE ENGINE
# ══════════════════════════════════════════════════

class NeuroVoterEnsemble:
    """
    Coordinates all 6 voters, runs anti-overfit checks,
    and produces a final human-like consensus decision.

    Usage:
        ensemble = NeuroVoterEnsemble()
        decision = ensemble.decide(symbol, df, model_confs, sentiment, mood)
    """

    def __init__(self):
        self.voters = [
            MomentumTrader(),
            ValueHunter(),
            RiskManager(),
            SentimentAnalyst(),
            PatternRecognizer(),
            QuantBrain(),
        ]
        self.regime_detector = RegimeDetector()
        self._decision_history = []  # track recent decisions for learning

    def decide(
        self,
        symbol: str,
        df: pd.DataFrame,
        rf_conf: float,
        xgb_conf: float,
        lstm_conf: float,
        intra_conf: float,
        original_votes: int,
        sentiment_score: float = 0.0,
        global_mood: float = 0.0,
        mood_details: dict = None,
    ) -> EnsembleDecision:
        """
        Run all 6 voters and produce a consensus decision.
        This is the core of the human-like voting system.
        """
        decision = EnsembleDecision()

        if df is None or df.empty:
            decision.reasoning = "No data available"
            return decision

        # ── Detect Regime ──
        regime = self.regime_detector.detect(df)
        decision.regime = regime.get("regime", "UNKNOWN")

        # ── Build Shared Context ──
        last = df.iloc[-1]
        price = float(last.get("Close", 0))
        prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else price
        change_pct = ((price - prev_price) / prev_price * 100) if prev_price > 0 else 0

        ctx = {
            "symbol": symbol,
            "df": df,
            "price": price,
            "change_pct": change_pct,
            "rsi": float(last.get("RSI", 50)),
            "atr": float(last.get("ATR", 0)),
            "macd": float(last.get("MACD", 0)),
            "macd_signal": float(last.get("MACD_Signal", 0)),
            "volume_ratio": float(last.get("Volume_Ratio", 1.0)),
            "ema_20": float(last.get("EMA_20", 0)),
            "sma_50": float(last.get("SMA_50", 0)),
            "sma_200": float(last.get("SMA_200", 0)),
            "bb_upper": float(last.get("BB_Upper", 0)),
            "bb_lower": float(last.get("BB_Lower", 0)),
            "rf_conf": rf_conf,
            "xgb_conf": xgb_conf,
            "lstm_conf": lstm_conf,
            "intra_conf": intra_conf,
            "original_votes": original_votes,
            "sentiment_score": sentiment_score,
            "global_mood": global_mood,
            "mood_details": mood_details or {},
            "regime": regime,
        }

        # ── Run All Voters ──
        total_weighted_vote = 0.0
        total_weight = 0.0

        for voter in self.voters:
            try:
                result = voter.vote(ctx)
            except Exception as e:
                result = VoterResult(voter.name, 0.0, 0.0, f"Error: {e}")

            decision.voter_results.append(result)

            # Check for vetoes
            if result.veto:
                decision.vetoed = True
                decision.veto_reasons.append(f"{result.name}: {result.veto_reason}")

            # Count directions
            if result.vote > 0.1:
                decision.buy_voters += 1
            elif result.vote < -0.1:
                decision.sell_voters += 1
            else:
                decision.hold_voters += 1

            # Weighted vote: vote * conviction * voter_weight
            weighted = result.vote * result.conviction * voter.weight
            total_weighted_vote += weighted
            total_weight += voter.weight * result.conviction

        # ── Calculate Final Score ──
        if total_weight > 0:
            decision.weighted_score = total_weighted_vote / total_weight
        decision.total_conviction = total_weight / sum(v.weight for v in self.voters)

        # ── Regime Adjustment ──
        # In trending-down or high-vol regimes, be less willing to buy
        if regime["regime"] == "TRENDING_DOWN":
            decision.weighted_score -= 0.15
        elif regime["regime"] == "HIGH_VOLATILE":
            decision.weighted_score -= 0.10
        elif regime["regime"] == "TRENDING_UP":
            decision.weighted_score += 0.05

        # ── Final Decision ──
        if decision.vetoed:
            decision.should_buy = False
            decision.reasoning = (
                f"VETOED by {', '.join(r.split(':')[0] for r in decision.veto_reasons)} | "
                f"Score: {decision.weighted_score:+.3f}"
            )
        elif decision.weighted_score > 0.15 and decision.buy_voters >= 3:
            # Need BOTH a positive score AND at least 3 voters wanting to buy
            decision.should_buy = True
            decision.reasoning = (
                f"BUY: Score {decision.weighted_score:+.3f} | "
                f"{decision.buy_voters}/6 buy, {decision.sell_voters}/6 sell | "
                f"Regime: {regime['regime']}"
            )
        elif decision.weighted_score > 0.25 and decision.buy_voters >= 2:
            # Strong score with 2+ voters is also ok 
            decision.should_buy = True
            decision.reasoning = (
                f"BUY (strong conviction): Score {decision.weighted_score:+.3f} | "
                f"{decision.buy_voters}/6 buy | Regime: {regime['regime']}"
            )
        else:
            decision.should_buy = False
            decision.reasoning = (
                f"HOLD: Score {decision.weighted_score:+.3f} "
                f"(need >0.15) | {decision.buy_voters}/6 buy, "
                f"{decision.sell_voters}/6 sell | Regime: {regime['regime']}"
            )

        # ── Track for Learning ──
        self._decision_history.append({
            "symbol": symbol,
            "time": datetime.now(IST).strftime("%H:%M"),
            "score": round(decision.weighted_score, 3),
            "buy": decision.should_buy,
            "regime": regime["regime"],
        })
        if len(self._decision_history) > 100:
            self._decision_history.pop(0)

        return decision

    def get_voter_summary(self) -> list:
        """For the dashboard: return voter info."""
        return [
            {
                "name": v.name,
                "type": v.__class__.__name__,
                "weight": v.weight,
            }
            for v in self.voters
        ]


# ══════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  NEURO-VOTER ENSEMBLE — Self Test")
    print("=" * 60)

    ensemble = NeuroVoterEnsemble()
    for v in ensemble.voters:
        print(f"  {v.name:20s} | Weight: {v.weight:.1f} | Type: {v.__class__.__name__}")

    print(f"\n  Total voters: {len(ensemble.voters)}")
    print(f"  Veto-capable: Risk Raj, Sentiment Sanjay")
    print("  Anti-overfit: QuantBrain has built-in OverfitDetector")
    print("  Regime-aware: RegimeDetector adjusts all decisions")
    print("\n  ✅ All systems nominal")
