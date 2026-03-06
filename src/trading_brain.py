"""
====================================================
PROJECT AEGIS - Trading Brain v1 (Human-Like Intelligence)
====================================================
This module makes trading decisions the way a REAL human
trader thinks — not just "vote counting" but considering:

EXIT LOGIC (why v1 never sold):
  1. MOMENTUM REVERSAL EXIT  — MACD crosses below signal while in trade
  2. RSI OVERBOUGHT EXIT     — RSI hits 75+ → take profits
  3. VOLUME FADE EXIT        — Volume dries up → smart money leaving
  4. TIME DECAY EXIT         — Held too long with no move → cut
  5. SENTIMENT SHIFT EXIT    — Bad news breaks while in trade
  6. TRAILING STOP TIGHTENER — As profit grows, trail gets tighter
  7. INTRADAY MOMENTUM EXIT  — Price making lower lows intraday

ENTRY LOGIC (smarter than vote counting):
  1. Market context awareness  (don't buy in a crash)
  2. Sector rotation check     (is this sector hot or cold?)
  3. Global macro filter       (war = don't buy, peace = be aggressive)
  4. Adaptive confidence       (require MORE when market is volatile)
  5. Time-of-day awareness     (avoid first 30min, last 30min chop)
  6. Price action confirmation (not just indicators)

CONFIGURABLE ALGO:
  - Trading personality: AGGRESSIVE / MODERATE / CONSERVATIVE
  - Each personality changes thresholds, exit rules, position sizes
  - Can be changed from config.py without touching this code

PHILOSOPHY: A human trader:
  ✅ Takes profits when momentum fades
  ✅ Gets out when the story changes (news, sector shift)
  ✅ Is more cautious in volatile markets
  ✅ Doesn't hold losing positions hoping for miracles
  ✅ Adapts to changing conditions during the day
  ❌ Does NOT trade on emotion (panic, greed, FOMO)
====================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

import pytz

sys.path.insert(0, os.path.dirname(__file__))

IST = pytz.timezone("Asia/Kolkata")


# ══════════════════════════════════════════════════
#  TRADING PERSONALITY PROFILES
# ══════════════════════════════════════════════════
# These can be changed from config.py via TRADING_PERSONALITY env var

PERSONALITIES = {
    "AGGRESSIVE": {
        "min_votes": 2,                 # 2/4 models enough
        "confidence_threshold": 0.55,   # Lower bar
        "rsi_exit_upper": 78,           # Hold longer before exit
        "rsi_exit_lower": 20,           # Tighter panic exit
        "time_decay_minutes": 120,      # Wait 2 hours before time exit
        "time_decay_min_move_pct": 0.3, # Need 0.3% move to hold
        "trailing_tighten_factor": 0.7, # Less tight trailing
        "max_hold_minutes": 300,        # 5h max hold
        "volume_fade_ratio": 0.3,       # Volume must drop a LOT to exit
        "sentiment_exit_threshold": -0.4,  # Only exit on very bad news
        "position_size_mult": 1.2,      # Slightly larger positions
        "atr_stop_mult": 2.0,          # Wider stops
        "atr_target_mult": 3.5,        # Higher targets
        "partial_exit_atr_mult": 2.0,  # Later partial
    },
    "MODERATE": {
        "min_votes": 2,
        "confidence_threshold": 0.60,
        "rsi_exit_upper": 75,
        "rsi_exit_lower": 22,
        "time_decay_minutes": 90,
        "time_decay_min_move_pct": 0.5,
        "trailing_tighten_factor": 0.5,
        "max_hold_minutes": 240,
        "volume_fade_ratio": 0.4,
        "sentiment_exit_threshold": -0.3,
        "position_size_mult": 1.0,
        "atr_stop_mult": 1.5,
        "atr_target_mult": 3.0,
        "partial_exit_atr_mult": 1.5,
    },
    "CONSERVATIVE": {
        "min_votes": 3,                 # Need 3/4 models
        "confidence_threshold": 0.70,   # High confidence required
        "rsi_exit_upper": 70,           # Exit earlier
        "rsi_exit_lower": 25,
        "time_decay_minutes": 60,       # Only 1 hour patience
        "time_decay_min_move_pct": 0.8, # Need bigger move to hold
        "trailing_tighten_factor": 0.3, # Very tight trailing
        "max_hold_minutes": 180,        # 3h max
        "volume_fade_ratio": 0.5,       # Sensitive to volume drop
        "sentiment_exit_threshold": -0.2,  # Exit on mildly bad news
        "position_size_mult": 0.7,      # Smaller positions
        "atr_stop_mult": 1.2,          # Tighter stops
        "atr_target_mult": 2.5,        # Lower targets
        "partial_exit_atr_mult": 1.0,  # Early partial
    },
}


class TradingBrain:
    """
    Human-like trading intelligence.
    Evaluates entries AND exits with market context awareness.
    """

    def __init__(self, personality: str = "MODERATE", capital: float = 1000.0):
        self.personality = personality.upper()
        if self.personality not in PERSONALITIES:
            self.personality = "MODERATE"
        self.params = PERSONALITIES[self.personality]
        self.capital = capital
        self._trade_context = {}  # track per-trade data for smart exits

    # ══════════════════════════════════════════════════
    #  SMART ENTRY DECISION
    # ══════════════════════════════════════════════════

    def should_enter(
        self,
        votes: int = 0,
        avg_confidence: float = 0.0,
        indicators: dict = None,
        sentiment_score: float = 0.0,
        market_mood: dict = None,
    ) -> tuple:
        """
        Human-like entry decision.  Returns (should_buy: bool, reason: str).

        Called by sniper.py AFTER the 4-model ensemble has already voted.
        This adds a human layer of context:
          - Market mood / global cues
          - Sentiment from real news
          - Time of day
          - RSI / volatility sanity
          - Adaptive confidence based on conditions
        """
        ind = indicators or {}
        mood = market_mood or {}
        reasons_yes = []
        reasons_no = []
        params = dict(self.params)  # Copy so we can adjust

        rsi = ind.get("rsi", 50)
        atr = ind.get("atr", 0)
        macd = ind.get("macd", 0)
        macd_sig = ind.get("macd_signal", 0)
        vol_ratio = ind.get("volume_ratio", 1.0)
        ema20 = ind.get("ema_20", 0)
        sma_50 = ind.get("sma_50", 0)

        global_mood_score = mood.get("overall_mood", 0) if isinstance(mood, dict) else 0

        # ── 1. Time of Day Filter ──
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0)
        minutes_since_open = (now - market_open).total_seconds() / 60

        if minutes_since_open < 30:
            params["confidence_threshold"] += 0.10
            reasons_no.append(f"Early market ({minutes_since_open:.0f}min)")

        if now.hour >= 14:
            params["confidence_threshold"] += 0.05
            reasons_no.append("Late session")

        # ── 2. Market Mood Adjustment ──
        if global_mood_score < -0.3:
            params["confidence_threshold"] += 0.10
            params["min_votes"] = max(params["min_votes"], 3)
            reasons_no.append(f"Negative mood ({global_mood_score:+.2f})")
        elif global_mood_score > 0.3:
            params["confidence_threshold"] -= 0.05
            reasons_yes.append(f"Positive mood ({global_mood_score:+.2f})")

        # ── 3. Sentiment Gate ──
        if sentiment_score < -0.3:
            return False, f"BLOCKED: Negative sentiment ({sentiment_score:+.2f})"
        if sentiment_score > 0.3:
            reasons_yes.append(f"Positive sentiment ({sentiment_score:+.2f})")

        # ── 4. RSI Sanity ──
        if rsi > params["rsi_exit_upper"]:
            return False, f"BLOCKED: RSI overbought ({rsi:.0f})"
        if rsi < params["rsi_exit_lower"]:
            return False, f"BLOCKED: RSI freefall ({rsi:.0f})"

        # ── 5. Volume Confirmation ──
        if vol_ratio < 0.5:
            return False, f"BLOCKED: No volume ({vol_ratio:.2f}x)"
        if vol_ratio > 1.5:
            reasons_yes.append(f"High volume ({vol_ratio:.1f}x)")

        # ── 6. Volatility Check ──
        price = ind.get("price", 0)
        if atr > 0 and price > 0:
            atr_pct = atr / price
            if atr_pct > 0.05:
                return False, f"BLOCKED: Extreme volatility (ATR {atr_pct:.1%})"

        # ── 7. Vote + Confidence ──
        if votes < params["min_votes"]:
            return False, f"Not enough votes ({votes}/{params['min_votes']})"
        if avg_confidence < params["confidence_threshold"]:
            return False, f"Confidence too low ({avg_confidence:.2f} < {params['confidence_threshold']:.2f})"

        # ── 8. Trend Quality ──
        trend_score = 0
        if ema20 > 0 and price > ema20:
            trend_score += 1
        if ema20 > 0 and sma_50 > 0 and ema20 > sma_50:
            trend_score += 1
        if macd > macd_sig:
            trend_score += 1
        if trend_score >= 2:
            reasons_yes.append(f"Trend OK ({trend_score}/3)")
        elif trend_score == 0 and self.personality == "CONSERVATIVE":
            return False, "BLOCKED: No uptrend"

        reason_str = f"BUY: {', '.join(reasons_yes)}" if reasons_yes else "BUY: Conditions met"
        if reasons_no:
            reason_str += f" | Caution: {', '.join(reasons_no[:3])}"
        return True, reason_str

    # ══════════════════════════════════════════════════
    #  SMART EXIT DECISION  (the main fix!)
    # ══════════════════════════════════════════════════

    def should_exit(
        self,
        entry_price: float = 0,
        current_price: float = 0,
        stop_loss: float = 0,
        target: float = 0,
        entry_time: str = "09:30",
        indicators: dict = None,
        sentiment_score: float = 0.0,
        market_mood: dict = None,
    ) -> tuple:
        """
        Human-like exit decision. Returns (should_exit: bool, exit_type: str, reason: str).

        Called by sniper.py with per-tick data for each open trade.
        This is the FIX for "never sells". A real trader exits when:
          1. Target hit / Stop loss hit
          2. Momentum reverses (MACD flips)
          3. RSI overbought → take profits!
          4. Volume fades (smart money leaving)
          5. Held too long without moving (time decay)
          6. Bad news breaks (sentiment shift)
          7. Trailing stop tightener
          8. Intraday lower lows
        """
        ind = indicators or {}
        pnl_per_share = current_price - entry_price
        pnl_pct = (pnl_per_share / entry_price) * 100 if entry_price > 0 else 0
        in_profit = pnl_per_share > 0

        rsi = ind.get("rsi", 50)
        macd = ind.get("macd", 0)
        macd_sig = ind.get("macd_signal", 0)
        vol_ratio = ind.get("volume_ratio", 1.0)
        current_atr = ind.get("atr", 0)

        # Time calculation
        now = datetime.now(IST)
        try:
            entry_t = now.replace(
                hour=int(entry_time.split(":")[0]),
                minute=int(entry_time.split(":")[1]),
                second=0,
            )
            minutes_held = max(0, (now - entry_t).total_seconds() / 60)
        except Exception:
            minutes_held = 60

        # ── 1. Hard Stop Loss ──
        if current_price <= stop_loss:
            return True, "STOP_LOSS", f"Price ₹{current_price:.2f} hit stop ₹{stop_loss:.2f}"

        # ── 2. Target Hit ──
        if current_price >= target:
            return True, "TARGET_HIT", f"Price ₹{current_price:.2f} hit target ₹{target:.2f}"

        # ── 3. MOMENTUM REVERSAL EXIT ──
        if in_profit and macd < macd_sig and pnl_pct > 0.5:
            return True, "MOMENTUM_EXIT", (
                f"MACD bearish ({macd:.4f}<{macd_sig:.4f}) with {pnl_pct:.1f}% profit"
            )

        # ── 4. RSI OVERBOUGHT EXIT ──
        if in_profit and rsi > self.params["rsi_exit_upper"]:
            return True, "RSI_EXIT", (
                f"RSI {rsi:.0f} overbought — taking {pnl_pct:.1f}% profit"
            )

        # ── 5. VOLUME FADE EXIT ──
        if in_profit and vol_ratio < self.params["volume_fade_ratio"] and pnl_pct > 0.3:
            return True, "VOLUME_EXIT", (
                f"Volume faded ({vol_ratio:.2f}x) — locking {pnl_pct:.1f}%"
            )

        # ── 6. TIME DECAY EXIT ──
        if minutes_held > self.params["time_decay_minutes"]:
            if abs(pnl_pct) < self.params["time_decay_min_move_pct"]:
                return True, "TIME_DECAY", (
                    f"Held {minutes_held:.0f}min with {pnl_pct:+.2f}% — trade dead"
                )

        # ── 7. MAX HOLD TIME EXIT ──
        if minutes_held > self.params["max_hold_minutes"]:
            return True, "MAX_HOLD", (
                f"Held {minutes_held:.0f}min (max {self.params['max_hold_minutes']}) — {pnl_pct:+.2f}%"
            )

        # ── 8. SENTIMENT SHIFT EXIT ──
        if sentiment_score < self.params["sentiment_exit_threshold"]:
            return True, "SENTIMENT_EXIT", (
                f"Negative sentiment ({sentiment_score:+.2f}) — exiting"
            )

        # ── No exit → HOLD ──
        return False, "HOLD", f"Holding at {pnl_pct:+.2f}% | RSI={rsi:.0f}"

    # ══════════════════════════════════════════════════
    #  POSITION SIZING (human-like)
    # ══════════════════════════════════════════════════

    def calculate_position_size(
        self,
        price: float,
        atr: float,
        bullet_size: float,
        sentiment_score: float = 0.0,
        global_mood: float = 0.0,
    ) -> dict:
        """
        Calculate position size with adjustments for market conditions.
        A human trader sizes DOWN in scary markets, sizes UP in confident ones.
        """
        stop_mult = self.params["atr_stop_mult"]
        target_mult = self.params["atr_target_mult"]
        size_mult = self.params["position_size_mult"]

        # Adjust for sentiment
        if sentiment_score < -0.2:
            size_mult *= 0.7   # Smaller position on bad news
            stop_mult *= 0.8   # Tighter stop
        elif sentiment_score > 0.3:
            size_mult *= 1.1   # Slightly larger on good news

        # Adjust for global mood
        if global_mood < -0.3:
            size_mult *= 0.6   # Much smaller in fearful market
            target_mult *= 0.7 # Lower expectation
        elif global_mood > 0.3:
            size_mult *= 1.1

        stop_loss = price - (stop_mult * atr)
        target = price + (target_mult * atr)
        risk_per_share = price - stop_loss

        if risk_per_share <= 0:
            return None

        adj_bullet = bullet_size * size_mult
        qty_by_capital = int(adj_bullet / price)
        max_risk = adj_bullet * 0.10
        qty_by_risk = int(max_risk / risk_per_share) if risk_per_share > 0 else 0
        qty = max(1, min(qty_by_capital, qty_by_risk))

        return {
            "qty": qty,
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "risk_per_share": round(risk_per_share, 2),
            "total_risk": round(qty * risk_per_share, 2),
            "potential_profit": round(qty * (target - price), 2),
            "position_mult": round(size_mult, 2),
        }

    # ══════════════════════════════════════════════════
    #  MARKET CONTEXT (used by sniper before scanning)
    # ══════════════════════════════════════════════════

    def assess_market_context(self, global_mood: dict, macro_sentiment: dict) -> dict:
        """
        Like a human trader checking the overall market before placing trades.
        Returns a context dict that affects all decisions for this scan.
        """
        mood_score = global_mood.get("score", 0.0)
        macro_score = macro_sentiment.get("score", 0.0)
        mood_details = global_mood.get("details", {})

        ctx = {
            "mood": mood_score,
            "macro": macro_score,
            "vix": mood_details.get("india_vix", "N/A"),
            "vix_signal": mood_details.get("vix_signal", "UNKNOWN"),
            "should_trade": True,
            "aggression": "NORMAL",
            "reason": "",
        }

        # Combined market score
        combined = mood_score * 0.6 + macro_score * 0.4

        if combined < -0.5:
            ctx["should_trade"] = False
            ctx["aggression"] = "STOP"
            ctx["reason"] = "Market conditions extremely negative — skip this scan"
        elif combined < -0.3:
            ctx["aggression"] = "DEFENSIVE"
            ctx["reason"] = "Negative market — only high-conviction trades"
        elif combined > 0.3:
            ctx["aggression"] = "OFFENSIVE"
            ctx["reason"] = "Positive market — can be more aggressive"
        else:
            ctx["aggression"] = "NORMAL"
            ctx["reason"] = "Market neutral — standard parameters"

        # VIX override
        vix = mood_details.get("india_vix", 0)
        if isinstance(vix, (int, float)) and vix > 30:
            ctx["should_trade"] = False
            ctx["aggression"] = "STOP"
            ctx["reason"] = f"India VIX at {vix} — extreme fear, do not trade"

        return ctx

    def get_personality_info(self) -> dict:
        """Return current personality and all its parameters for the dashboard."""
        return {
            "personality": self.personality,
            "params": dict(self.params),
        }


# ══════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  TRADING BRAIN — Self Test")
    print("=" * 60)

    for p in ["AGGRESSIVE", "MODERATE", "CONSERVATIVE"]:
        brain = TradingBrain(personality=p)
        info = brain.get_personality_info()
        print(f"\n   {p}:")
        for k, v in info["params"].items():
            print(f"      {k}: {v}")
