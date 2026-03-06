"""
====================================================
PROJECT AEGIS — Smart Graduated Alerts
====================================================
Tiered Telegram alerts with context:
  - INFO:     Normal events (scans, small P&L)
  - WARNING:  Attention needed (drawdown, drift, near earnings)
  - CRITICAL: Immediate action (guardian stop, max loss, regime shift)
  - DIGEST:   Daily P&L summary, anomaly detection

Throttles alerts to avoid spam. Batches similar alerts.

Outputs
-------
data/smart_alerts.json
====================================================
"""

import os, json, time, warnings
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ALERTS_FILE = os.path.join(DATA_DIR, "smart_alerts.json")
ALERT_LOG_FILE = os.path.join(DATA_DIR, "smart_alert_log.json")


# ══════════════════════════════════════════════════
#  SEVERITY LEVELS
# ══════════════════════════════════════════════════
class Severity:
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    DIGEST = "DIGEST"


SEVERITY_EMOJI = {
    Severity.INFO: "ℹ️",
    Severity.WARNING: "⚠️",
    Severity.CRITICAL: "🚨",
    Severity.DIGEST: "📊",
}

SEVERITY_PRIORITY = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
    Severity.DIGEST: 1,
}


# ══════════════════════════════════════════════════
#  THROTTLE — Prevent spam
# ══════════════════════════════════════════════════
_throttle_cache = {}
THROTTLE_WINDOWS = {
    Severity.INFO: 300,       # 5 min between same INFO alerts
    Severity.WARNING: 120,    # 2 min between same WARNING alerts
    Severity.CRITICAL: 0,     # No throttle on CRITICAL
    Severity.DIGEST: 3600,    # 1 hour between digests
}


def _should_throttle(alert_key: str, severity: str) -> bool:
    """Return True if this alert should be suppressed."""
    window = THROTTLE_WINDOWS.get(severity, 300)
    if window == 0:
        return False
    last = _throttle_cache.get(alert_key, 0)
    if time.time() - last < window:
        return True
    _throttle_cache[alert_key] = time.time()
    return False


# ══════════════════════════════════════════════════
#  ALERT QUEUE
# ══════════════════════════════════════════════════
_alert_queue: list[dict] = []


def enqueue_alert(severity: str, category: str, title: str,
                  message: str, data: dict | None = None,
                  symbol: str = "") -> dict:
    """
    Add an alert to the queue.
    Combines throttling + severity classification.
    Returns the alert dict (or empty if throttled).
    """
    alert_key = f"{severity}:{category}:{symbol}"
    if _should_throttle(alert_key, severity):
        return {}

    alert = {
        "severity": severity,
        "category": category,
        "title": title,
        "message": message,
        "symbol": symbol,
        "data": data or {},
        "timestamp": _now_ist(),
        "priority": SEVERITY_PRIORITY.get(severity, 0),
    }
    _alert_queue.append(alert)
    _log_alert(alert)
    return alert


# ══════════════════════════════════════════════════
#  PRE-BUILT ALERT GENERATORS
# ══════════════════════════════════════════════════
def alert_trade_opened(symbol: str, price: float, qty: int,
                       confidence: float, neuro_score: float):
    """INFO: New position opened."""
    enqueue_alert(
        Severity.INFO, "TRADE",
        f"BUY {symbol.replace('.NS', '')}",
        f"Bought {qty} × ₹{price:,.2f} | Conf: {confidence:.0%} | Neuro: {neuro_score:+.2f}",
        {"price": price, "qty": qty, "confidence": confidence},
        symbol=symbol,
    )


def alert_trade_closed(symbol: str, entry: float, exit_price: float,
                       pnl: float, exit_type: str):
    """INFO/WARNING depending on result."""
    sev = Severity.INFO if pnl >= 0 else Severity.WARNING
    icon = "🟢" if pnl >= 0 else "🔴"
    enqueue_alert(
        sev, "TRADE",
        f"{icon} CLOSE {symbol.replace('.NS', '')} ({exit_type})",
        f"Entry: ₹{entry:,.2f} → Exit: ₹{exit_price:,.2f} | P&L: ₹{pnl:,.2f}",
        {"entry": entry, "exit": exit_price, "pnl": pnl, "exit_type": exit_type},
        symbol=symbol,
    )


def alert_drawdown(current_dd_pct: float, max_dd_pct: float):
    """WARNING: Drawdown approaching limit."""
    sev = Severity.CRITICAL if current_dd_pct > max_dd_pct * 0.8 else Severity.WARNING
    enqueue_alert(
        sev, "RISK",
        f"Drawdown Alert: {current_dd_pct:.1f}%",
        f"Current DD: {current_dd_pct:.1f}% | Max allowed: {max_dd_pct:.1f}%",
        {"current_dd": current_dd_pct, "max_dd": max_dd_pct},
    )


def alert_guardian_halt(reason: str):
    """CRITICAL: Guardian stopped trading."""
    enqueue_alert(
        Severity.CRITICAL, "GUARDIAN",
        "🚨 TRADING HALTED",
        f"Risk Guardian halted all trading: {reason}",
        {"reason": reason},
    )


def alert_regime_shift(old_regime: str, new_regime: str, confidence: float):
    """WARNING: Market regime changed."""
    enqueue_alert(
        Severity.WARNING if new_regime != "BEAR" else Severity.CRITICAL,
        "REGIME",
        f"Regime: {old_regime} → {new_regime}",
        f"HMM detected regime shift to {new_regime} (conf={confidence:.0%})",
        {"old": old_regime, "new": new_regime, "confidence": confidence},
    )


def alert_model_drift(model_name: str, drift_score: float):
    """WARNING: Model drift detected."""
    enqueue_alert(
        Severity.WARNING, "DRIFT",
        f"Model Drift: {model_name}",
        f"{model_name} drift score: {drift_score:.2f} — may need retraining",
        {"model": model_name, "drift": drift_score},
    )


def alert_anomaly(metric: str, value: float, threshold: float):
    """WARNING: Statistical anomaly detected."""
    enqueue_alert(
        Severity.WARNING, "ANOMALY",
        f"Anomaly: {metric}",
        f"{metric} = {value:.2f} (threshold: ±{threshold:.2f})",
        {"metric": metric, "value": value, "threshold": threshold},
    )


def alert_daily_digest(total_pnl: float, trades: int, wins: int,
                        losses: int, capital: float,
                        regime: str = "UNKNOWN", risk_score: int = 0):
    """DIGEST: End-of-day summary."""
    pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0
    wr = (wins / trades * 100) if trades > 0 else 0
    icon = "🟢" if total_pnl >= 0 else "🔴"

    msg = (
        f"{icon} Day P&L: ₹{total_pnl:,.2f} ({pnl_pct:+.1f}%)\n"
        f"Trades: {trades} | W/L: {wins}/{losses} | WR: {wr:.0f}%\n"
        f"Regime: {regime} | Risk Score: {risk_score}/100"
    )
    enqueue_alert(
        Severity.DIGEST, "DAILY",
        f"Daily Report ({icon} ₹{total_pnl:+,.0f})",
        msg,
        {
            "pnl": total_pnl, "pnl_pct": pnl_pct,
            "trades": trades, "wins": wins, "losses": losses,
            "regime": regime, "risk_score": risk_score,
        },
    )


# ══════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════
def detect_anomalies(state: dict, trade_history: list | None = None) -> list[dict]:
    """
    Check for statistical anomalies in today's trading.
    Returns list of anomaly alerts.
    """
    anomalies = []
    pnl = state.get("total_profit", 0)
    capital = state.get("capital", 1000)
    trades = state.get("trades_taken", 0)

    # Anomaly 1: Unusual loss
    if pnl < -capital * 0.03:
        anomalies.append({
            "type": "UNUSUAL_LOSS",
            "message": f"Day loss exceeds 3% (₹{pnl:,.2f})",
            "severity": Severity.CRITICAL,
        })

    # Anomaly 2: Too many trades
    if trades > 15:
        anomalies.append({
            "type": "OVERTRADE",
            "message": f"High trade count: {trades} (normal: 3-8)",
            "severity": Severity.WARNING,
        })

    # Anomaly 3: All losses
    if trades > 3 and state.get("trades_won", 0) == 0:
        anomalies.append({
            "type": "LOSING_STREAK",
            "message": f"All {trades} trades were losses",
            "severity": Severity.WARNING,
        })

    # Anomaly 4: Unusual win streak
    if state.get("trades_won", 0) > 8:
        anomalies.append({
            "type": "UNUSUAL_WINS",
            "message": f"Win streak: {state['trades_won']} — verify data integrity",
            "severity": Severity.INFO,
        })

    for a in anomalies:
        alert_anomaly(a["type"], 0, 0)

    return anomalies


# ══════════════════════════════════════════════════
#  TELEGRAM DISPATCH
# ══════════════════════════════════════════════════
def dispatch_alerts(bot=None) -> int:
    """
    Send queued alerts via Telegram bot.
    Returns number of alerts sent.
    """
    global _alert_queue
    if not _alert_queue:
        return 0

    sent = 0
    # Sort by priority (CRITICAL first)
    sorted_alerts = sorted(_alert_queue, key=lambda a: -a.get("priority", 0))

    for alert in sorted_alerts:
        emoji = SEVERITY_EMOJI.get(alert["severity"], "")
        text = f"{emoji} *{alert['title']}*\n{alert['message']}"

        if bot:
            try:
                bot.send_message(text)
                sent += 1
            except Exception:
                pass
        else:
            # Fallback: try to use the existing notifier
            try:
                from notifier import send_telegram_message
                send_telegram_message(text)
                sent += 1
            except Exception:
                pass

    _alert_queue = []
    return sent


# ══════════════════════════════════════════════════
#  STATE & LOGGING
# ══════════════════════════════════════════════════
def _log_alert(alert: dict):
    """Append alert to persistent log."""
    log = []
    if os.path.exists(ALERT_LOG_FILE):
        try:
            with open(ALERT_LOG_FILE, "r") as f:
                log = json.load(f)
        except Exception:
            log = []

    log.append(alert)
    # Keep last 1000
    log = log[-1000:]
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2, default=str)


def get_alert_stats() -> dict:
    """Get alert statistics for dashboard."""
    if not os.path.exists(ALERT_LOG_FILE):
        return {"total": 0, "info": 0, "warning": 0, "critical": 0, "digest": 0}

    try:
        with open(ALERT_LOG_FILE, "r") as f:
            log = json.load(f)
    except Exception:
        return {"total": 0, "info": 0, "warning": 0, "critical": 0, "digest": 0}

    stats = {"total": len(log), "info": 0, "warning": 0, "critical": 0, "digest": 0}
    for a in log:
        sev = a.get("severity", "INFO").lower()
        stats[sev] = stats.get(sev, 0) + 1

    # Recent alerts (last 20)
    stats["recent"] = log[-20:][::-1]
    return stats


def save_alert_state(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_ist() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
