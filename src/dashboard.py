"""
====================================================
PROJECT AEGIS - Streamlit Dashboard (Live + Local)
====================================================
Real-time dashboard — runs 100% on YOUR machine.
No data leaves your PC. Auto-refreshes every 30s.

Run:
    streamlit run src/dashboard.py
====================================================
"""

import os
import sys
import json
import time
import warnings
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import pytz

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Aegis - Live Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────
#  Paths + Config
# ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_FILE = os.path.join(DATA_DIR, "daily_state.json")
TRADE_LOG = os.path.join(DATA_DIR, "trade_history.csv")
RANKING_FILE = os.path.join(DATA_DIR, "daily_ranking.csv")
DASHBOARD_FILE = os.path.join(DATA_DIR, "dashboard_state.json")

IST = pytz.timezone("Asia/Kolkata")

# Add src/ to path so we can import config
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
try:
    from config import STOCK_WATCHLIST, TOP_N_STOCKS, CAPITAL
except ImportError:
    STOCK_WATCHLIST = ["TATASTEEL.NS", "SBIN.NS", "RELIANCE.NS"]
    TOP_N_STOCKS = 3
    CAPITAL = 1000


# ──────────────────────────────────────────────────
#  Auto-refresh (every 30 seconds)
# ──────────────────────────────────────────────────
def auto_refresh():
    """Inject JS for auto-refresh countdown."""
    refresh_sec = 30
    st.markdown(
        f"""
        <script>
            var timer = {refresh_sec};
            var el = document.getElementById('refresh-timer');
            setInterval(function() {{
                timer--;
                if (el) el.innerText = timer;
                if (timer <= 0) {{
                    window.location.reload();
                }}
            }}, 1000);
        </script>
        <div style='text-align:right; color:grey; font-size:12px;'>
            Auto-refresh in <span id='refresh-timer'>{refresh_sec}</span>s
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────
#  Live Price Fetcher (cached 30s)
# ──────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_live_prices(symbols: list) -> dict:
    """Fetch current prices from Yahoo Finance. Runs locally — no data shared."""
    import yfinance as yf
    prices = {}
    changes = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.fast_info
            price = getattr(info, "last_price", None)
            prev = getattr(info, "previous_close", None)
            if price:
                prices[sym] = round(float(price), 2)
                if prev and prev > 0:
                    changes[sym] = round(((price - prev) / prev) * 100, 2)
                else:
                    changes[sym] = 0.0
            else:
                # Fallback: use history
                hist = yf.download(sym, period="2d", interval="1d", progress=False)
                if not hist.empty:
                    prices[sym] = round(float(hist["Close"].iloc[-1]), 2)
                    if len(hist) >= 2:
                        prev_c = float(hist["Close"].iloc[-2])
                        changes[sym] = round(((prices[sym] - prev_c) / prev_c) * 100, 2)
                    else:
                        changes[sym] = 0.0
        except Exception:
            prices[sym] = 0.0
            changes[sym] = 0.0
    return prices, changes


def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# ──────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────
st.title("🛡️ Project Aegis — Live Dashboard")
st.caption("4-Model Ensemble · 100% Local · Auto-refreshes every 30s")
auto_refresh()

now_ist = datetime.now(IST)
st.markdown(f"**🕐 {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}** · Market: "
            f"{'🟢 OPEN' if 9 <= now_ist.hour < 16 else '🔴 CLOSED'}")

st.divider()

# ──────────────────────────────────────────────────
#  LIVE STOCK PRICES
# ──────────────────────────────────────────────────
st.subheader("📡 Live Stock Prices")

# Determine which stocks to show
ranking_df = load_csv(RANKING_FILE)
if not ranking_df.empty and "symbol" in ranking_df.columns:
    show_symbols = ranking_df["symbol"].tolist()
else:
    show_symbols = STOCK_WATCHLIST[:TOP_N_STOCKS]

with st.spinner("Fetching live prices..."):
    prices, changes = fetch_live_prices(show_symbols)

# Display price tiles
cols = st.columns(min(len(show_symbols), 5))
for i, sym in enumerate(show_symbols[:5]):
    with cols[i]:
        price = prices.get(sym, 0)
        change = changes.get(sym, 0)
        label = sym.replace(".NS", "").replace(".BO", "")
        st.metric(label, f"₹{price:.2f}", f"{change:+.2f}%")

# Show remaining as table if more than 5
if len(show_symbols) > 5:
    extra_data = []
    for sym in show_symbols[5:]:
        extra_data.append({
            "Stock": sym.replace(".NS", ""),
            "Price (₹)": prices.get(sym, 0),
            "Change (%)": changes.get(sym, 0),
        })
    if extra_data:
        st.dataframe(pd.DataFrame(extra_data), use_container_width=True, hide_index=True)

st.divider()

# ──────────────────────────────────────────────────
#  SNIPER STATUS (from state file)
# ──────────────────────────────────────────────────
st.subheader("🔫 Sniper Status")
dash = load_json(DASHBOARD_FILE)
state = load_json(STATE_FILE)

if dash:
    st.markdown(f"**Last Sniper Update:** {dash.get('last_updated', 'N/A')}")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        pnl = dash.get("total_profit", 0)
        pnl_pct = dash.get("total_profit_pct", 0)
        st.metric("Day P&L", f"₹{pnl:.2f}", f"{pnl_pct:.2f}%")
    with col2:
        status = dash.get("status", "N/A")
        st.metric("Status", status)
    with col3:
        st.metric("Trades", dash.get("trades_taken", 0))
    with col4:
        won = dash.get("trades_won", 0)
        taken = dash.get("trades_taken", 0)
        wr = f"{(won / taken * 100):.0f}%" if taken > 0 else "N/A"
        st.metric("Win Rate", wr)
    with col5:
        st.metric("Open Positions", dash.get("active_count", 0))
else:
    st.info("⏳ No sniper data yet. Run `python src/scholar.py` then `python src/sniper.py`")

# Active Positions with live P&L
if state and "active_trades" in state:
    open_trades = [t for t in state["active_trades"] if t.get("status") == "OPEN"]
    if open_trades:
        st.markdown("**Open Positions (Live P&L):**")
        rows = []
        for t in open_trades:
            sym = t.get("stock", "")
            entry = t.get("price", 0)
            qty = t.get("qty", 0)
            live = prices.get(sym, entry)
            unrealised = (live - entry) * qty
            rows.append({
                "Stock": sym.replace(".NS", ""),
                "Entry ₹": entry,
                "Live ₹": live,
                "Qty": qty,
                "SL ₹": t.get("stop_loss", 0),
                "Target ₹": t.get("target", 0),
                "Unrealised P&L ₹": round(unrealised, 2),
                "Votes": t.get("votes", 0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ──────────────────────────────────────────────────
#  STOCK RANKING (from Scholar)
# ──────────────────────────────────────────────────
st.subheader("📊 Stock Ranking (from Scholar)")

if not ranking_df.empty:
    display_cols = [c for c in ["rank", "symbol", "rf_prob", "xgb_prob",
                                 "lstm_prob", "intraday_prob", "avg_prob",
                                 "votes", "sentiment", "date"]
                    if c in ranking_df.columns]

    # Add live price column
    if "symbol" in ranking_df.columns:
        ranking_df["Live ₹"] = ranking_df["symbol"].map(prices)
        ranking_df["Change %"] = ranking_df["symbol"].map(changes)
        display_cols = ["rank", "symbol", "Live ₹", "Change %"] + [
            c for c in display_cols if c not in ("rank", "symbol")
        ]
        display_cols = [c for c in display_cols if c in ranking_df.columns]

    st.dataframe(
        ranking_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart
    if "symbol" in ranking_df.columns and "avg_prob" in ranking_df.columns:
        chart_data = ranking_df.set_index("symbol")[
            [c for c in ["rf_prob", "xgb_prob", "lstm_prob", "intraday_prob"]
             if c in ranking_df.columns]
        ]
        st.bar_chart(chart_data)
else:
    st.info("No ranking data. Run `python src/scholar.py` first.")

st.divider()

# ──────────────────────────────────────────────────
#  TRADE HISTORY
# ──────────────────────────────────────────────────
st.subheader("📋 Trade History")
trades_df = load_csv(TRADE_LOG)

if not trades_df.empty:
    if "Actual_Profit" in trades_df.columns:
        total_pnl = trades_df["Actual_Profit"].sum()
        closed = trades_df[trades_df["Action"] != "BUY"]
        total_trades = len(closed)
        wins = len(trades_df[trades_df["Actual_Profit"] > 0])
        losses = len(trades_df[trades_df["Actual_Profit"] < 0])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total P&L", f"₹{total_pnl:.2f}")
        with col2:
            st.metric("Total Trades", total_trades)
        with col3:
            st.metric("Wins", wins)
        with col4:
            st.metric("Losses", losses)

    # Cumulative P&L chart
    if "Date" in trades_df.columns and "Actual_Profit" in trades_df.columns:
        pnl_by_date = trades_df.groupby("Date")["Actual_Profit"].sum().cumsum()
        if len(pnl_by_date) > 1:
            st.markdown("**Cumulative P&L Over Time:**")
            st.line_chart(pnl_by_date, use_container_width=True)

    # Per-stock breakdown
    if "Stock" in trades_df.columns and "Actual_Profit" in trades_df.columns:
        st.markdown("**P&L by Stock:**")
        stock_pnl = trades_df.groupby("Stock")["Actual_Profit"].agg(["sum", "count"])
        stock_pnl.columns = ["Total P&L ₹", "Trades"]
        st.dataframe(stock_pnl, use_container_width=True)

    # Filter
    if "Stock" in trades_df.columns:
        stocks = ["All"] + sorted(trades_df["Stock"].unique().tolist())
        selected = st.selectbox("Filter by Stock", stocks)
        if selected != "All":
            trades_df = trades_df[trades_df["Stock"] == selected]

    st.dataframe(trades_df.tail(50), use_container_width=True, hide_index=True)
else:
    st.info("No trade history yet. Run the Sniper first.")

# ──────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f"""
    - **Watchlist**: {len(STOCK_WATCHLIST)} stocks
    - **Top-N**: {TOP_N_STOCKS}
    - **Capital**: ₹{CAPITAL}
    - **Models**: RF + XGB + LSTM + Intraday
    - **Voting**: 3-out-of-4 consensus
    - **Stop Loss**: 1.5 × ATR
    - **Target**: 3.0 × ATR
    - **Daily Target**: 2%
    - **Kill Switch**: -5%
    """)

    st.divider()
    st.markdown("🔒 **100% Local** — No data leaves your PC")
    st.markdown("**Project Aegis** v2.1")
    st.markdown(f"Last refresh: {now_ist.strftime('%H:%M:%S IST')}")

    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
