"""
====================================================
PROJECT AEGIS — Notification System v1
====================================================
Sends alerts via Telegram and/or Discord when key events
occur: trades, vetoes, regime changes, daily P&L summaries.

Setup:
  1. Create a Telegram Bot via @BotFather → get BOT_TOKEN
  2. Get your Chat ID via @userinfobot
  3. Set env vars:
       AEGIS_TELEGRAM_TOKEN=your_bot_token
       AEGIS_TELEGRAM_CHAT_ID=your_chat_id
  4. (Optional) AEGIS_DISCORD_WEBHOOK=your_webhook_url

The system works without any setup — alerts just silently
skip if no credentials are configured.
====================================================
"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Optional

import pytz

IST = pytz.timezone("Asia/Kolkata")

# ──────────────────────────────────────────────────
#  CREDENTIALS (from environment variables)
# ──────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("AEGIS_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("AEGIS_TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK = os.getenv("AEGIS_DISCORD_WEBHOOK", "")

# Alert toggle
ALERTS_ENABLED = os.getenv("AEGIS_ALERTS_ENABLED", "true").lower() == "true"

# ──────────────────────────────────────────────────
#  LOCAL LOG (always works, no setup needed)
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERT_LOG_FILE = os.path.join(BASE, "data", "alert_history.json")


def _log_alert(alert_type: str, message: str, details: dict = None):
    """Save every alert to local JSON log regardless of Telegram/Discord."""
    try:
        os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)
        history = []
        if os.path.exists(ALERT_LOG_FILE):
            with open(ALERT_LOG_FILE, "r") as f:
                history = json.load(f)
        
        entry = {
            "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
            "type": alert_type,
            "message": message,
        }
        if details:
            entry["details"] = details

        history.append(entry)

        # Keep last 500 alerts
        if len(history) > 500:
            history = history[-500:]

        with open(ALERT_LOG_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception:
        pass


# ──────────────────────────────────────────────────
#  TELEGRAM SENDER
# ──────────────────────────────────────────────────
def _send_telegram(message: str) -> bool:
    """Send Telegram message. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────
#  DISCORD SENDER
# ──────────────────────────────────────────────────
def _send_discord(message: str) -> bool:
    """Send Discord webhook message. Returns True on success."""
    if not DISCORD_WEBHOOK:
        return False
    try:
        # Strip HTML tags for Discord (it uses markdown)
        clean = message.replace("<b>", "**").replace("</b>", "**")
        clean = clean.replace("<i>", "_").replace("</i>", "_")
        clean = clean.replace("<code>", "`").replace("</code>", "`")

        data = json.dumps({"content": clean}).encode("utf-8")
        req = urllib.request.Request(
            DISCORD_WEBHOOK, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception:
        return False


def _send(message: str):
    """Send to all configured channels."""
    _send_telegram(message)
    _send_discord(message)


# ══════════════════════════════════════════════════
#  PUBLIC ALERT FUNCTIONS
# ══════════════════════════════════════════════════

def alert_trade_buy(symbol: str, price: float, qty: int, confidence: float,
                    neuro_score: float = 0, regime: str = ""):
    """Alert: New position opened."""
    if not ALERTS_ENABLED:
        return
    name = symbol.replace(".NS", "")
    msg = (
        f"🟢 <b>BUY — {name}</b>\n"
        f"Price: ₹{price:,.2f} | Qty: {qty}\n"
        f"AI Confidence: {confidence:.2f}\n"
        f"NeuroScore: {neuro_score:+.3f} | Regime: {regime}"
    )
    _log_alert("BUY", msg, {
        "symbol": symbol, "price": price, "qty": qty,
        "confidence": confidence, "neuro_score": neuro_score, "regime": regime,
    })
    _send(msg)


def alert_trade_exit(symbol: str, entry_price: float, exit_price: float,
                     pnl: float, exit_type: str):
    """Alert: Position closed."""
    if not ALERTS_ENABLED:
        return
    name = symbol.replace(".NS", "")
    icon = "✅" if pnl > 0 else ("❌" if pnl < 0 else "⚪")
    result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "EVEN")
    pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

    msg = (
        f"{icon} <b>EXIT — {name} ({result})</b>\n"
        f"₹{entry_price:,.2f} → ₹{exit_price:,.2f} ({pct:+.2f}%)\n"
        f"P&L: ₹{pnl:,.2f} | Type: {exit_type}"
    )
    _log_alert("EXIT", msg, {
        "symbol": symbol, "entry": entry_price, "exit": exit_price,
        "pnl": pnl, "exit_type": exit_type,
    })
    _send(msg)


def alert_veto(symbol: str, reasons: list):
    """Alert: Trade was vetoed by a NeuroVoter."""
    if not ALERTS_ENABLED:
        return
    name = symbol.replace(".NS", "")
    msg = (
        f"🚫 <b>VETO — {name}</b>\n"
        f"Reasons: {', '.join(reasons)}"
    )
    _log_alert("VETO", msg, {"symbol": symbol, "reasons": reasons})
    _send(msg)


def alert_regime_change(old_regime: str, new_regime: str, symbol: str = "MARKET"):
    """Alert: Market regime changed."""
    if not ALERTS_ENABLED:
        return
    regime_icons = {
        "TRENDING_UP": "📈", "TRENDING_DOWN": "📉",
        "HIGH_VOLATILE": "🌊", "LOW_VOLATILE": "🧊",
        "SIDEWAYS": "➡️",
    }
    icon = regime_icons.get(new_regime, "🔄")
    msg = (
        f"{icon} <b>REGIME CHANGE — {symbol}</b>\n"
        f"{old_regime} → {new_regime}"
    )
    _log_alert("REGIME_CHANGE", msg, {
        "symbol": symbol, "old": old_regime, "new": new_regime,
    })
    _send(msg)


def alert_guardian_stop(reason: str):
    """Alert: Risk Guardian halted trading."""
    if not ALERTS_ENABLED:
        return
    msg = f"🚨 <b>GUARDIAN HALT</b>\n{reason}"
    _log_alert("GUARDIAN_HALT", msg, {"reason": reason})
    _send(msg)


def alert_daily_summary(total_pnl: float, total_trades: int, wins: int,
                        losses: int, capital: float):
    """Alert: End-of-day summary."""
    if not ALERTS_ENABLED:
        return
    pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0
    wr = (wins / total_trades * 100) if total_trades > 0 else 0
    icon = "📈" if total_pnl >= 0 else "📉"

    msg = (
        f"{icon} <b>DAILY SUMMARY</b>\n"
        f"P&L: ₹{total_pnl:,.2f} ({pnl_pct:+.2f}%)\n"
        f"Trades: {total_trades} | W: {wins} L: {losses} | WR: {wr:.0f}%\n"
        f"Capital: ₹{capital + total_pnl:,.2f}"
    )
    _log_alert("DAILY_SUMMARY", msg, {
        "pnl": total_pnl, "trades": total_trades,
        "wins": wins, "losses": losses, "capital": capital,
    })
    _send(msg)


def alert_custom(title: str, message: str, alert_type: str = "INFO"):
    """Alert: Custom message."""
    if not ALERTS_ENABLED:
        return
    msg = f"ℹ️ <b>{title}</b>\n{message}"
    _log_alert(alert_type, msg)
    _send(msg)


# ──────────────────────────────────────────────────
#  HELPER: Check if alerts are configured
# ──────────────────────────────────────────────────
def is_configured() -> dict:
    """Return configuration status."""
    return {
        "telegram": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "discord": bool(DISCORD_WEBHOOK),
        "alerts_enabled": ALERTS_ENABLED,
        "local_log": True,  # Always works
    }
