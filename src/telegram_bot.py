"""
====================================================
🤖 PROJECT AEGIS - Telegram Bot (Two-Way Control)
====================================================
Upgrades from one-way notifier to full interactive bot.

Commands:
  /status     — Portfolio overview, open positions, P&L
  /pnl        — Detailed P&L breakdown per stock
  /holdings   — Current open positions with unrealised P&L
  /report     — Generate & send daily performance report
  /equity     — Mini equity curve (last 30 days)
  /kelly      — Current Kelly sizing per stock
  /earnings   — Upcoming earnings calendar
  /watchlist  — Show current watchlist & rankings
  /health     — System health check
  /help       — List all commands

Environment Variables Required:
  TELEGRAM_BOT_TOKEN  — Bot token from @BotFather
  TELEGRAM_CHAT_ID    — Your chat ID (from @userinfobot)
====================================================
"""

import os
import sys
import json
import threading
import time
import warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
POLL_INTERVAL      = 2   # seconds between polling for updates

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# Attempt to import python-telegram-bot
try:
    import requests as _req
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ──────────────────────────────────────────────────
#  TELEGRAM API (raw HTTP – no extra dependency)
# ──────────────────────────────────────────────────
class TelegramAPI:
    """Minimal Telegram Bot API wrapper using requests."""

    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_update_id = 0

    def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to a chat."""
        if not HAS_REQUESTS or not self.token:
            return False
        try:
            # Telegram limit is 4096 chars
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks:
                _req.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": parse_mode,
                    },
                    timeout=10,
                )
            return True
        except Exception as e:
            print(f"  [TG] Send failed: {e}")
            return False

    def get_updates(self) -> list:
        """Get new messages (long polling)."""
        if not HAS_REQUESTS or not self.token:
            return []
        try:
            resp = _req.get(
                f"{self.base_url}/getUpdates",
                params={
                    "offset": self.last_update_id + 1,
                    "timeout": POLL_INTERVAL,
                },
                timeout=POLL_INTERVAL + 5,
            )
            data = resp.json()
            if data.get("ok") and data.get("result"):
                updates = data["result"]
                if updates:
                    self.last_update_id = updates[-1]["update_id"]
                return updates
        except Exception:
            pass
        return []


# ──────────────────────────────────────────────────
#  DATA LOADERS
# ──────────────────────────────────────────────────
def load_json_safe(filename: str) -> dict:
    """Load a JSON file from data dir."""
    path = os.path.join(DATA_DIR, filename)
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def load_csv_safe(filename: str):
    """Load a CSV from data dir."""
    import pandas as pd
    path = os.path.join(DATA_DIR, filename)
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────
#  COMMAND HANDLERS
# ──────────────────────────────────────────────────
def cmd_status() -> str:
    """Portfolio overview."""
    state = load_json_safe("dashboard_state.json")
    daily = load_json_safe("daily_state.json")

    if not state and not daily:
        return "📊 <b>Status</b>\n\nNo data available yet. Run the Sniper first."

    pnl = state.get("total_pnl", daily.get("total_pnl", 0))
    positions = state.get("open_positions", daily.get("open_positions", []))
    capital = state.get("capital", daily.get("capital", 0))
    bullets = state.get("bullets_used", daily.get("bullets_used", 0))
    max_b = state.get("max_bullets", daily.get("max_bullets", 5))

    msg = "📊 <b>Portfolio Status</b>\n\n"
    msg += f"💰 Capital: ₹{capital:,.2f}\n"
    msg += f"📈 Total P&L: ₹{pnl:,.2f}\n"
    msg += f"🔫 Bullets: {bullets}/{max_b}\n"
    msg += f"📦 Open Positions: {len(positions) if isinstance(positions, list) else positions}\n"

    if isinstance(positions, list) and positions:
        msg += "\n<b>Positions:</b>\n"
        for p in positions[:10]:
            sym = p.get("symbol", "?")
            entry = p.get("entry_price", 0)
            curr = p.get("current_price", entry)
            upnl = (curr - entry) * p.get("qty", 0) if curr and entry else 0
            icon = "🟢" if upnl >= 0 else "🔴"
            msg += f"  {icon} {sym}: ₹{entry:.2f}→₹{curr:.2f} (₹{upnl:+,.2f})\n"

    return msg


def cmd_pnl() -> str:
    """P&L breakdown."""
    import pandas as pd
    trades = load_csv_safe("trade_history.csv")

    if trades is None or trades.empty:
        return "💹 <b>P&L Report</b>\n\nNo completed trades yet."

    pnl_col = "pnl" if "pnl" in trades.columns else "PnL"
    sym_col = "symbol" if "symbol" in trades.columns else "Symbol"

    if pnl_col not in trades.columns:
        return "💹 <b>P&L Report</b>\n\nTrade history format not recognized."

    total_pnl = trades[pnl_col].sum()
    total_trades = len(trades)
    wins = len(trades[trades[pnl_col] > 0])
    wr = (wins / total_trades * 100) if total_trades > 0 else 0

    msg = "💹 <b>P&L Report</b>\n\n"
    msg += f"Total P&L: ₹{total_pnl:,.2f}\n"
    msg += f"Trades: {total_trades} ({wins}W / {total_trades - wins}L)\n"
    msg += f"Win Rate: {wr:.1f}%\n"

    if sym_col in trades.columns:
        by_stock = trades.groupby(sym_col)[pnl_col].agg(["sum", "count"])
        by_stock = by_stock.sort_values("sum", ascending=False)
        msg += "\n<b>Per Stock:</b>\n"
        for sym, row in by_stock.iterrows():
            icon = "🟢" if row["sum"] >= 0 else "🔴"
            msg += f"  {icon} {sym}: ₹{row['sum']:,.2f} ({int(row['count'])} trades)\n"

    return msg


def cmd_holdings() -> str:
    """Current open positions."""
    state = load_json_safe("dashboard_state.json")
    positions = state.get("open_positions", [])

    if not positions:
        return "📦 <b>Holdings</b>\n\nNo open positions."

    msg = "📦 <b>Open Holdings</b>\n\n"
    total_val = 0
    total_upnl = 0

    for p in positions:
        sym = p.get("symbol", "?")
        qty = p.get("qty", 0)
        entry = p.get("entry_price", 0)
        curr = p.get("current_price", entry)
        sl = p.get("stop_loss", 0)
        tgt = p.get("target", 0)
        upnl = (curr - entry) * qty if curr and entry else 0
        val = curr * qty if curr else 0
        total_val += val
        total_upnl += upnl

        icon = "🟢" if upnl >= 0 else "🔴"
        msg += f"{icon} <b>{sym}</b>\n"
        msg += f"   {qty}x @ ₹{entry:.2f} → ₹{curr:.2f}\n"
        msg += f"   SL: ₹{sl:.2f} | Tgt: ₹{tgt:.2f}\n"
        msg += f"   P&L: ₹{upnl:+,.2f}\n\n"

    msg += f"<b>Total Value: ₹{total_val:,.2f}</b>\n"
    msg += f"<b>Unrealised P&L: ₹{total_upnl:+,.2f}</b>"
    return msg


def cmd_equity() -> str:
    """Mini equity curve text table."""
    state = load_json_safe("dashboard_state.json")
    curve = state.get("equity_curve", [])

    if not curve:
        return "📉 <b>Equity Curve</b>\n\nNo data yet."

    recent = curve[-15:]  # Last 15 data points
    msg = "📈 <b>Equity Curve (Last 15)</b>\n\n<pre>"
    msg += f"{'Date':>12} {'Equity':>10}\n"
    msg += "-" * 24 + "\n"
    for pt in recent:
        d = pt.get("date", "?")[:10]
        eq = pt.get("equity", 0)
        msg += f"{d:>12} ₹{eq:>9,.2f}\n"
    msg += "</pre>"
    return msg


def cmd_kelly() -> str:
    """Kelly sizing info."""
    kelly = load_json_safe("kelly_state.json")

    if not kelly:
        return "💰 <b>Kelly Sizing</b>\n\nNo Kelly data. Run kelly_sizing.py first."

    overall = kelly.get("overall", {})
    per_stock = kelly.get("per_stock", {})

    msg = "💰 <b>Kelly Criterion Sizing</b>\n\n"
    msg += f"Overall Win Rate: {overall.get('win_rate', 0)*100:.1f}%\n"
    msg += f"Payoff Ratio: {overall.get('payoff_ratio', 0):.2f}\n"
    msg += f"Kelly %: {overall.get('kelly_adj', 0)*100:.2f}%\n"
    msg += f"Trades Used: {overall.get('trades', 0)}\n"

    if per_stock:
        msg += "\n<b>Per Stock:</b>\n"
        for sym, info in sorted(per_stock.items(),
                                key=lambda x: x[1].get("kelly_adj", 0), reverse=True):
            msg += (f"  {sym}: {info.get('kelly_adj', 0)*100:.1f}% "
                    f"(WR: {info.get('win_rate', 0)*100:.0f}%)\n")

    return msg


def cmd_earnings() -> str:
    """Upcoming earnings."""
    cal = load_json_safe("earnings_calendar.json")
    calendar = cal.get("calendar", {})

    if not calendar:
        return "📅 <b>Earnings Calendar</b>\n\nNo data. Run earnings_guard.py first."

    today = datetime.now()
    msg = "📅 <b>Earnings Calendar</b>\n\n"

    entries = []
    for sym, info in calendar.items():
        ed = info.get("next_earnings", "")
        try:
            dt = datetime.strptime(ed, "%Y-%m-%d")
            days_to = (dt - today).days
            entries.append((sym, ed, days_to, info.get("source", "?")))
        except Exception:
            continue

    entries.sort(key=lambda x: x[2])
    for sym, ed, days, src in entries:
        if days <= 3:
            icon = "🚫"
        elif days <= 7:
            icon = "⚠️"
        else:
            icon = "📅"
        msg += f"  {icon} {sym}: {ed} ({days}d) [{src}]\n"

    return msg


def cmd_watchlist() -> str:
    """Current watchlist and rankings."""
    ranking = load_csv_safe("daily_ranking.csv")
    if ranking is None or ranking.empty:
        from config import STOCK_WATCHLIST, TOP_N_STOCKS
        syms = STOCK_WATCHLIST[:TOP_N_STOCKS]
        msg = "📋 <b>Watchlist</b>\n\n"
        for i, s in enumerate(syms, 1):
            msg += f"  {i}. {s}\n"
        return msg

    msg = "📋 <b>Daily Ranking</b>\n\n"
    for i, row in ranking.head(15).iterrows():
        sym = row.get("symbol", row.get("Symbol", "?"))
        score = row.get("score", row.get("Score", 0))
        msg += f"  {i+1}. {sym} — Score: {score:.2f}\n"

    return msg


def cmd_health() -> str:
    """System health check."""
    msg = "🏥 <b>System Health</b>\n\n"

    # Check key files
    checks = {
        "Dashboard State": "dashboard_state.json",
        "Daily State": "daily_state.json",
        "Trade History": "trade_history.csv",
        "Best Params": "best_params.json",
        "Ensemble Weights": "ensemble_weights.json",
        "Guardian Log": "guardian_log.json",
        "Learner Report": "learner_report.json",
    }

    for name, filename in checks.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            size = os.path.getsize(path)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            age_h = (datetime.now() - mtime).total_seconds() / 3600
            icon = "✅" if age_h < 24 else "⚠️"
            msg += f"  {icon} {name}: {size:,}B (updated {age_h:.1f}h ago)\n"
        else:
            msg += f"  ❌ {name}: MISSING\n"

    # Model files
    model_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    if os.path.exists(model_dir):
        model_count = len([f for f in os.listdir(model_dir) if f.endswith((".h5", ".pkl", ".joblib"))])
        msg += f"\n  🧠 Models: {model_count} files in models/\n"

    return msg


def cmd_help() -> str:
    """Help message."""
    return (
        "🤖 <b>Project Aegis Bot</b>\n\n"
        "<b>Commands:</b>\n"
        "  /status — Portfolio overview\n"
        "  /pnl — Detailed P&L breakdown\n"
        "  /holdings — Open positions\n"
        "  /equity — Equity curve\n"
        "  /kelly — Kelly sizing info\n"
        "  /earnings — Earnings calendar\n"
        "  /watchlist — Stock rankings\n"
        "  /health — System health check\n"
        "  /help — This message\n"
    )


# Command dispatch
COMMANDS = {
    "/status":    cmd_status,
    "/pnl":       cmd_pnl,
    "/holdings":  cmd_holdings,
    "/equity":    cmd_equity,
    "/kelly":     cmd_kelly,
    "/earnings":  cmd_earnings,
    "/watchlist": cmd_watchlist,
    "/health":    cmd_health,
    "/help":      cmd_help,
    "/start":     cmd_help,
}


# ──────────────────────────────────────────────────
#  BOT LOOP
# ──────────────────────────────────────────────────
class AegisTelegramBot:
    """Interactive Telegram bot for Project Aegis."""

    def __init__(self):
        self.api = TelegramAPI(TELEGRAM_BOT_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
        self.running = False

    def send_alert(self, message: str) -> bool:
        """Send a one-way alert (compatible with existing notifier)."""
        if not self.chat_id:
            return False
        return self.api.send_message(self.chat_id, message)

    def process_message(self, text: str, chat_id: str) -> str:
        """Process an incoming message and return response."""
        text = text.strip().lower().split("@")[0]  # Remove @botname

        if text in COMMANDS:
            try:
                return COMMANDS[text]()
            except Exception as e:
                return f"⚠️ Error: {e}"

        return (
            "❓ Unknown command.\n"
            "Type /help for available commands."
        )

    def poll_loop(self):
        """Main polling loop — runs forever."""
        print(f"  🤖 Telegram bot started (polling every {POLL_INTERVAL}s)")

        while self.running:
            try:
                updates = self.api.get_updates()
                for update in updates:
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    if text and chat_id:
                        # Optional: restrict to configured chat_id
                        if self.chat_id and chat_id != self.chat_id:
                            continue

                        response = self.process_message(text, chat_id)
                        self.api.send_message(chat_id, response)

            except Exception as e:
                print(f"  [TG] Poll error: {e}")
                time.sleep(5)

            time.sleep(POLL_INTERVAL)

    def start(self, background: bool = True):
        """Start the bot."""
        if not TELEGRAM_BOT_TOKEN:
            print("  [TG] No TELEGRAM_BOT_TOKEN set. Bot disabled.")
            return False
        if not HAS_REQUESTS:
            print("  [TG] 'requests' module not available. Bot disabled.")
            return False

        self.running = True
        if background:
            thread = threading.Thread(target=self.poll_loop, daemon=True)
            thread.start()
            return True
        else:
            self.poll_loop()
            return True

    def stop(self):
        """Stop the bot."""
        self.running = False


# ──────────────────────────────────────────────────
#  SINGLETON INSTANCE
# ──────────────────────────────────────────────────
_bot_instance = None


def get_bot() -> AegisTelegramBot:
    """Get or create bot singleton."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = AegisTelegramBot()
    return _bot_instance


def send_telegram_alert(message: str) -> bool:
    """Convenience function for sending alerts."""
    bot = get_bot()
    return bot.send_alert(message)


# ──────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤖 Project Aegis Telegram Bot")
    print("=" * 45)

    if not TELEGRAM_BOT_TOKEN:
        print("  ❌ TELEGRAM_BOT_TOKEN not set!")
        print("  Set environment variable to enable bot.")
        print("  Get token from @BotFather on Telegram.")
        sys.exit(1)

    if not TELEGRAM_CHAT_ID:
        print("  ⚠️  TELEGRAM_CHAT_ID not set (will respond to any chat)")

    bot = get_bot()
    print(f"  Starting bot in foreground mode...")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        bot.start(background=False)
    except KeyboardInterrupt:
        bot.stop()
        print("\n  Bot stopped.")
