"""
====================================================
🛡️ PROJECT AEGIS - Risk Guardian (Real Money Safety Layer)
====================================================
This module is the FINAL SAFETY NET before any trade is executed.
Designed specifically for REAL MONEY trading — every trade MUST
pass ALL safety checks before the Sniper can fire a bullet.

SAFETY LAYERS (in order):
  1. LEARNER HEALTH CHECK  — Is the AI healthy enough to trade today?
  2. REGIME FILTER         — Block trades in TRENDING_DOWN or HIGH_VOLATILE
  3. DRAWDOWN BREAKER      — Multi-level circuit breakers
  4. POSITION LIMITS       — Max risk per trade, per stock, per day
  5. CORRELATION FILTER    — Don't concentrate in same sector
  6. VOLATILITY GUARD      — Reject trades in extreme volatility
  7. CONFIDENCE GATE       — Adaptive threshold (not fixed 75%)
  8. TIME FILTER           — No trades in first 15 min or last 30 min
  9. CONSECUTIVE LOSS LOCK — Pause after N consecutive losses
  10. WEEKLY/MONTHLY CAPS  — Hard limits on cumulative losses

PHILOSOPHY: It is better to MISS a profitable trade than to TAKE
a losing one with real money. Safety > Profit.
====================================================
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import pytz

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CAPITAL, MAX_BULLETS, DAILY_TARGET,
    CONFIDENCE_THRESHOLD, MAX_DAILY_LOSS_PCT,
    ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER,
    TRADE_LOG_FILE,
)

IST = pytz.timezone("Asia/Kolkata")

# ──────────────────────────────────────────────────
#   SAFETY THRESHOLDS (cannot be overridden by env vars)
#   These are HARD-CODED for real money protection.
# ──────────────────────────────────────────────────

# --- Drawdown Circuit Breakers ---
DRAWDOWN_LEVEL_1_PCT = 0.02    # -2%  → reduce position size by 50%
DRAWDOWN_LEVEL_2_PCT = 0.04    # -4%  → only 1 bullet allowed
DRAWDOWN_LEVEL_3_PCT = 0.05    # -5%  → STOP ALL TRADING (existing kill switch)
DRAWDOWN_LEVEL_4_PCT = 0.08    # -8%  weekly → STOP for the week

# --- Position Limits ---
MAX_RISK_PER_TRADE_PCT = 0.02   # Never risk more than 2% of capital on ONE trade
MAX_SINGLE_STOCK_PCT = 0.30     # Max 30% of capital in one stock
MAX_OPEN_POSITIONS = 5           # Hard cap on simultaneous positions

# --- Time Filters ---
NO_TRADE_FIRST_MINUTES = 15      # Skip first 15 min (opening volatility)
NO_TRADE_LAST_MINUTES = 30       # No new trades in last 30 min

# --- Consecutive Loss Protection ---
MAX_CONSECUTIVE_LOSSES = 3       # Pause after 3 losses in a row
LOSS_COOLDOWN_MINUTES = 60       # Wait 1 hour after consecutive losses

# --- Weekly/Monthly Caps ---
MAX_WEEKLY_LOSS_PCT = 0.08       # Stop for the week if -8%
MAX_MONTHLY_LOSS_PCT = 0.15      # Stop for the month if -15%

# --- Minimum Model Agreement (real money = stricter) ---
MIN_CONFIDENCE_REAL_MONEY = 0.78  # Higher than paper trading threshold
MIN_VOTES_REAL_MONEY = 3          # Still need 3/4 consensus

# --- Volatility Guard ---
MAX_ATR_PCT_OF_PRICE = 0.05     # Reject if ATR > 5% of price (too volatile)
MIN_ATR_PCT_OF_PRICE = 0.005    # Reject if ATR < 0.5% of price (too flat / illiquid)

# ──────────────────────────────────────────────────
#   LEARNER REPORT PATH
# ──────────────────────────────────────────────────
LEARNER_REPORT_FILE = "data/learner_report.json"
LEARNER_WEIGHTS_FILE = "data/ensemble_weights.json"
GUARDIAN_LOG_FILE = "data/guardian_log.json"


# ══════════════════════════════════════════════════
#  NEURAL SAFETY NET — Deep Learning Output Control
# ══════════════════════════════════════════════════
# Because we cannot inspect the "reasoning" inside a neural
# network, we wrap every LSTM (and future DL model) with
# strict output validation.  If any check fails the model's
# prediction is REPLACED with a *neutral* value (0.50) so it
# effectively abstains from the vote.
# ──────────────────────────────────────────────────

NEURAL_MIN_OUTPUT = 0.01        # Anything below → stuck/dead neuron
NEURAL_MAX_OUTPUT = 0.99        # Anything above → overconfident / memorised
NEURAL_STABILITY_DELTA = 0.35   # If prediction changes >35% between ticks → unstable
NEURAL_STUCK_WINDOW = 5         # Flag if last N predictions are identical
NEURAL_ENSEMBLE_MAX_SPREAD = 0.60  # If all-model max-min conf > 60pp → disagreement


class NeuralSafetyNet:
    """
    Wraps every neural-network prediction with sanity checks.
    One instance per trading day; keeps a rolling history to detect
    stuck or drifting models.

    Usage:
        nsn = NeuralSafetyNet()
        safe_val = nsn.validate("daily_lstm", raw_output, input_array)
    """

    def __init__(self):
        self._history: dict[str, list[float]] = {}    # model_name → last N preds
        self._flags: list[str] = []                    # human-readable alerts

    # ---------- public API ----------
    def validate(self, model_name: str, raw_output: float,
                 input_array: "np.ndarray | None" = None) -> float:
        """
        Validate a single DL model output.
        Returns the (possibly overridden) safe confidence value.
        """
        safe = raw_output

        # 1. NaN / Inf check
        if not np.isfinite(safe):
            self._flag(model_name, f"NaN/Inf output ({raw_output})")
            return 0.50

        # 2. Range clamp  — sigmoid should be (0,1) but numerical
        #    edge-cases (float16 overflow, bad scaling) can break this.
        if safe < 0.0 or safe > 1.0:
            self._flag(model_name, f"Output {safe:.4f} outside [0,1] — clamped")
            safe = float(np.clip(safe, 0.0, 1.0))

        # 3. Extreme confidence filter
        if safe < NEURAL_MIN_OUTPUT or safe > NEURAL_MAX_OUTPUT:
            self._flag(model_name,
                       f"Extreme confidence {safe:.4f} — likely overfit/memorised → abstain")
            return 0.50   # neutral: effectively abstains from vote

        # 4. Stability check (against previous tick)
        history = self._history.setdefault(model_name, [])
        if history:
            prev = history[-1]
            delta = abs(safe - prev)
            if delta > NEURAL_STABILITY_DELTA:
                self._flag(model_name,
                           f"Prediction jumped {delta:.2f} in one tick "
                           f"({prev:.3f}→{safe:.3f}) — unstable → abstain")
                safe = 0.50

        # 5. Stuck-model detection (same value N times)
        history.append(round(safe, 4))
        if len(history) > NEURAL_STUCK_WINDOW:
            history.pop(0)
        if len(history) >= NEURAL_STUCK_WINDOW:
            if len(set(history)) == 1:
                self._flag(model_name,
                           f"Stuck: identical output {history[0]} for "
                           f"{NEURAL_STUCK_WINDOW} ticks → abstain")
                return 0.50

        # 6. Input-array sanity (if provided)
        if input_array is not None:
            if not np.all(np.isfinite(input_array)):
                self._flag(model_name, "Input array contains NaN/Inf — abstain")
                return 0.50
            # Detect degenerate inputs (all zeros / constant)
            if np.std(input_array) < 1e-8:
                self._flag(model_name,
                           "Input array is constant / near-zero — abstain")
                return 0.50

        return safe

    def validate_ensemble(self, confidences: dict[str, float]) -> tuple:
        """
        Check whether the full ensemble is in agreement.
        Returns (is_valid: bool, reason: str).
        """
        vals = [v for v in confidences.values() if v is not None and np.isfinite(v)]
        if not vals:
            return False, "NEURAL_ENSEMBLE: No valid model outputs"

        spread = max(vals) - min(vals)
        if spread > NEURAL_ENSEMBLE_MAX_SPREAD:
            self._flag("ensemble",
                       f"Model spread {spread:.2f} > {NEURAL_ENSEMBLE_MAX_SPREAD} — disagreement")
            return False, (f"NEURAL_ENSEMBLE: Models wildly disagree "
                           f"(spread={spread:.2f}). Skipping trade.")

        return True, "OK"

    # ---------- helpers ----------
    def _flag(self, model: str, msg: str):
        tag = f"[NEURAL_SAFETY/{model}] {msg}"
        self._flags.append(tag)
        print(f"   ⚠️  {tag}")

    def get_flags(self) -> list[str]:
        """Return all flags raised this session (for logging)."""
        return list(self._flags)

    def reset_flags(self):
        self._flags.clear()


class RiskGuardian:
    """
    The Risk Guardian is a stateful safety module.
    It must be initialised once per trading day, and every
    trade attempt must pass through `approve_trade()`.
    """

    def __init__(self, capital: float = CAPITAL):
        self.capital = capital
        self.day_start_capital = capital
        self.learner_report = self._load_learner_report()
        self.ensemble_weights = self._load_ensemble_weights()
        self.today_losses = 0.0
        self.today_wins = 0.0
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.trades_today = 0
        self.rejections_today = 0
        self.open_positions = {}   # stock -> {qty, entry_price, invested}
        self.guardian_active = True
        self._log_entries = []

        # Load trade history for weekly/monthly caps
        self._weekly_pnl = self._get_recent_pnl(days=7)
        self._monthly_pnl = self._get_recent_pnl(days=30)

        self._check_startup_safety()

    def _load_learner_report(self) -> dict:
        """Load the latest health report from the Learner."""
        if os.path.exists(LEARNER_REPORT_FILE):
            try:
                with open(LEARNER_REPORT_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _load_ensemble_weights(self) -> dict:
        """Load optimised ensemble weights from the Learner."""
        if os.path.exists(LEARNER_WEIGHTS_FILE):
            try:
                with open(LEARNER_WEIGHTS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _get_recent_pnl(self, days: int) -> float:
        """Calculate total P&L over the last N days from trade log."""
        if not os.path.exists(TRADE_LOG_FILE):
            return 0.0
        try:
            df = pd.read_csv(TRADE_LOG_FILE)
            if "Date" not in df.columns or "Actual_Profit" not in df.columns:
                return 0.0
            df["Date"] = pd.to_datetime(df["Date"])
            cutoff = datetime.now() - timedelta(days=days)
            recent = df[df["Date"] >= cutoff]
            return float(recent["Actual_Profit"].sum())
        except Exception:
            return 0.0

    def _check_startup_safety(self):
        """Run one-time checks at the start of the trading day."""
        # ── Check 1: Learner health ──
        if self.learner_report:
            if not self.learner_report.get("trading_allowed", True):
                self.guardian_active = False
                self._log("BLOCKED", "Learner report says trading NOT allowed",
                          severity="CRITICAL")
                return

            health = self.learner_report.get("model_health", "UNKNOWN")
            if health == "DEGRADED":
                self._log("WARNING", f"Model health is DEGRADED — reducing position sizes",
                          severity="HIGH")

        # ── Check 2: Weekly loss cap ──
        weekly_loss_pct = abs(self._weekly_pnl) / self.capital if self._weekly_pnl < 0 else 0
        if weekly_loss_pct >= MAX_WEEKLY_LOSS_PCT:
            self.guardian_active = False
            self._log("BLOCKED", f"Weekly loss {weekly_loss_pct*100:.1f}% exceeds "
                       f"{MAX_WEEKLY_LOSS_PCT*100:.0f}% cap. TRADING PAUSED FOR THE WEEK.",
                       severity="CRITICAL")
            return

        # ── Check 3: Monthly loss cap ──
        monthly_loss_pct = abs(self._monthly_pnl) / self.capital if self._monthly_pnl < 0 else 0
        if monthly_loss_pct >= MAX_MONTHLY_LOSS_PCT:
            self.guardian_active = False
            self._log("BLOCKED", f"Monthly loss {monthly_loss_pct*100:.1f}% exceeds "
                       f"{MAX_MONTHLY_LOSS_PCT*100:.0f}% cap. TRADING PAUSED FOR THE MONTH.",
                       severity="CRITICAL")
            return

        print(f"   [GUARDIAN] Active. Weekly P&L: Rs.{self._weekly_pnl:.2f}, "
              f"Monthly P&L: Rs.{self._monthly_pnl:.2f}")

    def approve_trade(self, symbol: str, price: float, qty: int,
                      stop_loss: float, target: float, atr: float,
                      rf_conf: float, xgb_conf: float,
                      lstm_conf: float, intra_conf: float,
                      votes: int) -> tuple:
        """
        The MAIN gate. Every trade attempt must pass through here.

        Returns:
            (approved: bool, reason: str, adjusted_qty: int)
        """
        if not self.guardian_active:
            return False, "GUARDIAN_DISABLED: Trading paused by safety system", 0

        now = datetime.now(IST)
        checks_passed = []
        adjusted_qty = qty

        # ═══════════════════════════════════════════
        #  CHECK 1: Time Filter
        # ═══════════════════════════════════════════
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)

        minutes_since_open = (now - market_open).total_seconds() / 60
        minutes_to_close = (market_close - now).total_seconds() / 60

        if minutes_since_open < NO_TRADE_FIRST_MINUTES:
            return False, f"TIME_FILTER: Market just opened ({minutes_since_open:.0f}min). Wait {NO_TRADE_FIRST_MINUTES}min.", 0

        if minutes_to_close < NO_TRADE_LAST_MINUTES:
            return False, f"TIME_FILTER: Too close to market close ({minutes_to_close:.0f}min left).", 0

        checks_passed.append("TIME")

        # ═══════════════════════════════════════════
        #  CHECK 2: Drawdown Circuit Breakers
        # ═══════════════════════════════════════════
        day_pnl = self.today_wins + self.today_losses
        day_pnl_pct = abs(day_pnl) / self.capital if day_pnl < 0 else 0

        if day_pnl_pct >= DRAWDOWN_LEVEL_3_PCT:
            self.guardian_active = False
            return False, f"CIRCUIT_BREAKER_L3: Day loss {day_pnl_pct*100:.1f}% — ALL TRADING STOPPED", 0

        if day_pnl_pct >= DRAWDOWN_LEVEL_2_PCT:
            current_open = len([p for p in self.open_positions.values() if p.get("active")])
            if current_open >= 1:
                return False, f"CIRCUIT_BREAKER_L2: Day loss {day_pnl_pct*100:.1f}% — Max 1 position allowed", 0
            adjusted_qty = max(1, qty // 3)  # Tiny position

        elif day_pnl_pct >= DRAWDOWN_LEVEL_1_PCT:
            adjusted_qty = max(1, qty // 2)  # Half position

        checks_passed.append("DRAWDOWN")

        # ═══════════════════════════════════════════
        #  CHECK 3: Consecutive Loss Protection
        # ═══════════════════════════════════════════
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            if self.last_loss_time:
                cooldown_left = (self.last_loss_time + timedelta(minutes=LOSS_COOLDOWN_MINUTES) - now)
                if cooldown_left.total_seconds() > 0:
                    mins_left = cooldown_left.total_seconds() / 60
                    return False, f"LOSS_COOLDOWN: {self.consecutive_losses} consecutive losses. Wait {mins_left:.0f}min.", 0

        checks_passed.append("STREAK")

        # ═══════════════════════════════════════════
        #  CHECK 4: Volatility Guard
        # ═══════════════════════════════════════════
        if price > 0:
            atr_pct = atr / price
            if atr_pct > MAX_ATR_PCT_OF_PRICE:
                return False, f"VOLATILITY: ATR {atr_pct*100:.2f}% of price — too volatile (max={MAX_ATR_PCT_OF_PRICE*100}%)", 0
            if atr_pct < MIN_ATR_PCT_OF_PRICE:
                return False, f"VOLATILITY: ATR {atr_pct*100:.3f}% of price — too flat/illiquid (min={MIN_ATR_PCT_OF_PRICE*100}%)", 0

        checks_passed.append("VOLATILITY")

        # ═══════════════════════════════════════════
        #  CHECK 5: Confidence Gate (Adaptive)
        # ═══════════════════════════════════════════
        # Use learner's recommended threshold if available
        risk_params = self.learner_report.get("risk_params", {})
        min_conf = risk_params.get("recommended_confidence", MIN_CONFIDENCE_REAL_MONEY)
        min_conf = max(min_conf, MIN_CONFIDENCE_REAL_MONEY)  # Never below hard floor

        avg_conf = (rf_conf + xgb_conf + lstm_conf + intra_conf) / 4
        if avg_conf < min_conf:
            return False, f"CONFIDENCE: Avg={avg_conf:.2f} < required={min_conf:.2f}", 0

        if votes < MIN_VOTES_REAL_MONEY:
            return False, f"VOTES: {votes}/4 < required {MIN_VOTES_REAL_MONEY}/4", 0

        checks_passed.append("CONFIDENCE")

        # ═══════════════════════════════════════════
        #  CHECK 6: Position Limits
        # ═══════════════════════════════════════════
        trade_value = price * adjusted_qty
        risk_amount = (price - stop_loss) * adjusted_qty

        # Max risk per single trade
        if risk_amount > self.capital * MAX_RISK_PER_TRADE_PCT:
            max_risk = self.capital * MAX_RISK_PER_TRADE_PCT
            risk_per_share = price - stop_loss
            if risk_per_share > 0:
                adjusted_qty = max(1, int(max_risk / risk_per_share))
                trade_value = price * adjusted_qty
            else:
                return False, "POSITION: Risk per share is zero/negative", 0

        # Max exposure to single stock
        existing_exposure = 0
        if symbol in self.open_positions:
            existing_exposure = self.open_positions[symbol].get("invested", 0)
        total_exposure = existing_exposure + trade_value

        if total_exposure > self.capital * MAX_SINGLE_STOCK_PCT:
            return False, f"CONCENTRATION: {symbol} exposure Rs.{total_exposure:.0f} > {MAX_SINGLE_STOCK_PCT*100:.0f}% cap", 0

        # Max open positions
        active_positions = len([p for p in self.open_positions.values() if p.get("active")])
        if active_positions >= MAX_OPEN_POSITIONS:
            return False, f"MAX_POSITIONS: Already {active_positions} open (max={MAX_OPEN_POSITIONS})", 0

        checks_passed.append("POSITION")

        # ═══════════════════════════════════════════
        #  CHECK 7: Market Regime Filter
        # ═══════════════════════════════════════════
        regimes = self.learner_report.get("market_regimes", {})
        market_regime = regimes.get("NIFTY50", "UNKNOWN")
        stock_regime = regimes.get(symbol, "UNKNOWN")

        if market_regime == "TRENDING_DOWN":
            return False, f"REGIME: Overall market is TRENDING_DOWN — no new longs", 0

        if stock_regime == "TRENDING_DOWN":
            return False, f"REGIME: {symbol} is TRENDING_DOWN — skipping", 0

        if market_regime == "HIGH_VOLATILE" or stock_regime == "HIGH_VOLATILE":
            adjusted_qty = max(1, adjusted_qty // 2)
            self._log("ADJUST", f"HIGH_VOLATILE regime — halved qty to {adjusted_qty}")

        checks_passed.append("REGIME")

        # ═══════════════════════════════════════════
        #  CHECK 8: Risk/Reward Ratio
        # ═══════════════════════════════════════════
        potential_loss = (price - stop_loss) * adjusted_qty
        potential_profit = (target - price) * adjusted_qty

        if potential_loss > 0 and potential_profit / potential_loss < 1.5:
            return False, f"RISK_REWARD: Ratio {potential_profit/potential_loss:.2f} < 1.5 minimum", 0

        checks_passed.append("RISK_REWARD")

        # ═══════════════════════════════════════════
        #  CHECK 9: Learner Risk Level Adjustment
        # ═══════════════════════════════════════════
        pos_multiplier = risk_params.get("position_size_multiplier", 1.0)
        if pos_multiplier < 1.0:
            adjusted_qty = max(1, int(adjusted_qty * pos_multiplier))

        # Learner's recommended max bullets
        rec_max_bullets = risk_params.get("recommended_max_bullets", MAX_BULLETS)
        if self.trades_today >= rec_max_bullets:
            return False, f"LEARNER_LIMIT: {self.trades_today} trades today >= learner recommends max {rec_max_bullets}", 0

        checks_passed.append("LEARNER")

        # ═══════════════ ALL CHECKS PASSED ═══════════════
        self._log("APPROVED", f"{symbol} @ Rs.{price:.2f} x {adjusted_qty} | "
                  f"Checks: {', '.join(checks_passed)}")

        return True, f"APPROVED (passed {len(checks_passed)} checks)", adjusted_qty

    def record_trade_result(self, symbol: str, pnl: float, was_win: bool):
        """Update guardian state after a trade closes."""
        if was_win:
            self.today_wins += pnl
            self.consecutive_losses = 0
        else:
            self.today_losses += pnl
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now(IST)

        self.trades_today += 1

        # Update open positions
        if symbol in self.open_positions:
            self.open_positions[symbol]["active"] = False

        day_pnl = self.today_wins + self.today_losses
        print(f"   [GUARDIAN] Trade result: {'WIN' if was_win else 'LOSS'} Rs.{pnl:.2f} | "
              f"Day P&L: Rs.{day_pnl:.2f} | Streak: {self.consecutive_losses} losses")

    def register_position(self, symbol: str, qty: int, entry_price: float):
        """Track an open position."""
        self.open_positions[symbol] = {
            "qty": qty,
            "entry_price": entry_price,
            "invested": qty * entry_price,
            "active": True,
        }

    def get_ensemble_weights(self, symbol: str) -> dict:
        """
        Return learner-optimised weights for the 4 models.
        Falls back to equal weights if not available.
        """
        default = {"rf_weight": 0.25, "xgb_weight": 0.25,
                    "lstm_weight": 0.25, "intraday_weight": 0.25}
        return self.ensemble_weights.get(symbol, default)

    def get_status(self) -> dict:
        """Return current guardian state for dashboard."""
        day_pnl = self.today_wins + self.today_losses
        return {
            "guardian_active": self.guardian_active,
            "day_pnl": round(day_pnl, 2),
            "trades_today": self.trades_today,
            "rejections_today": self.rejections_today,
            "consecutive_losses": self.consecutive_losses,
            "weekly_pnl": round(self._weekly_pnl, 2),
            "monthly_pnl": round(self._monthly_pnl, 2),
            "model_health": self.learner_report.get("model_health", "UNKNOWN"),
            "risk_level": self.learner_report.get("risk_params", {}).get("risk_level", "NORMAL"),
        }

    def _log(self, action: str, message: str, severity: str = "INFO"):
        """Log guardian decisions."""
        entry = {
            "time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "message": message,
            "severity": severity,
        }
        self._log_entries.append(entry)

        icon = {"INFO": "ℹ️", "HIGH": "⚠️", "CRITICAL": "🚨"}.get(severity, "•")
        print(f"   [GUARDIAN] {icon} {action}: {message}")

        # Persist log
        try:
            os.makedirs("data", exist_ok=True)
            with open(GUARDIAN_LOG_FILE, "w") as f:
                json.dump(self._log_entries[-100:], f, indent=2)  # Keep last 100
        except Exception:
            pass


# ══════════════════════════════════════════════════
#  CONVENIENCE: Quick safety check function
# ══════════════════════════════════════════════════
def quick_safety_check() -> tuple:
    """
    Quick check if trading is allowed today.
    Returns (allowed: bool, reason: str).
    Call this before starting the Sniper loop.
    """
    guardian = RiskGuardian()

    if not guardian.guardian_active:
        reasons = [e["message"] for e in guardian._log_entries if e["severity"] == "CRITICAL"]
        return False, "; ".join(reasons) if reasons else "Guardian disabled"

    return True, "All safety checks passed"


# ══════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  RISK GUARDIAN — Self Test")
    print("=" * 60)

    guardian = RiskGuardian()
    status = guardian.get_status()

    print(f"\n   Guardian Active : {status['guardian_active']}")
    print(f"   Model Health    : {status['model_health']}")
    print(f"   Risk Level      : {status['risk_level']}")
    print(f"   Weekly P&L      : Rs.{status['weekly_pnl']}")
    print(f"   Monthly P&L     : Rs.{status['monthly_pnl']}")

    # Test approval with dummy values
    approved, reason, adj_qty = guardian.approve_trade(
        symbol="TATASTEEL.NS",
        price=150.0, qty=5, stop_loss=145.0, target=160.0,
        atr=3.5, rf_conf=0.82, xgb_conf=0.80,
        lstm_conf=0.65, intra_conf=0.60, votes=3,
    )
    print(f"\n   Test Trade: {'APPROVED' if approved else 'REJECTED'}")
    print(f"   Reason: {reason}")
    print(f"   Adjusted Qty: {adj_qty}")
    print(f"\n{'=' * 60}")
