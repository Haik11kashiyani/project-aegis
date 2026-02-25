"""
====================================================
PROJECT AEGIS - Streamlit Dashboard
====================================================
A live dashboard to monitor the AI trading bot's
performance, trade history, and model voting results.

Run locally:
    streamlit run src/dashboard.py

Deploy free on Streamlit Community Cloud:
    1. Push repo to GitHub
    2. Go to share.streamlit.io
    3. Connect your repo → pick src/dashboard.py
====================================================
"""

import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime

# ──────────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Aegis - AI Trading Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────
#  Data Paths
# ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
STATE_FILE = os.path.join(DATA_DIR, "daily_state.json")
TRADE_LOG = os.path.join(DATA_DIR, "trade_history.csv")
RANKING_FILE = os.path.join(DATA_DIR, "daily_ranking.csv")
DASHBOARD_FILE = os.path.join(DATA_DIR, "dashboard_state.json")


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# ──────────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────────
st.title("🛡️ Project Aegis — AI Trading Dashboard")
st.caption("4-Model Ensemble: Random Forest + XGBoost + Daily LSTM + Intraday LSTM")

# ──────────────────────────────────────────────────
#  Dashboard State
# ──────────────────────────────────────────────────
dash = load_json(DASHBOARD_FILE)
state = load_json(STATE_FILE)

if dash:
    st.markdown(f"**Last Updated:** {dash.get('last_updated', 'N/A')}")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        pnl = dash.get("total_profit", 0)
        st.metric("Day P&L", f"₹{pnl:.2f}", f"{dash.get('total_profit_pct', 0):.2f}%")
    with col2:
        st.metric("Status", dash.get("status", "N/A"))
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
    st.info("No dashboard data yet. Run the Scholar + Sniper first.")

st.divider()

# ──────────────────────────────────────────────────
#  Stock Ranking
# ──────────────────────────────────────────────────
st.subheader("📊 Stock Ranking (from Scholar)")
ranking_df = load_csv(RANKING_FILE)

if not ranking_df.empty:
    # Show ranking table
    display_cols = [c for c in ["rank", "symbol", "rf_prob", "xgb_prob",
                                 "lstm_prob", "intraday_prob", "avg_prob",
                                 "votes", "sentiment", "date"]
                    if c in ranking_df.columns]
    st.dataframe(
        ranking_df[display_cols].style.format({
            "rf_prob": "{:.3f}", "xgb_prob": "{:.3f}",
            "lstm_prob": "{:.3f}", "intraday_prob": "{:.3f}",
            "avg_prob": "{:.3f}", "sentiment": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart of average probabilities
    if "symbol" in ranking_df.columns and "avg_prob" in ranking_df.columns:
        chart_data = ranking_df.set_index("symbol")[["rf_prob", "xgb_prob", "lstm_prob", "intraday_prob"]]
        st.bar_chart(chart_data)
else:
    st.info("No ranking data. Run the Scholar first.")

st.divider()

# ──────────────────────────────────────────────────
#  Trade History
# ──────────────────────────────────────────────────
st.subheader("📋 Trade History")
trades_df = load_csv(TRADE_LOG)

if not trades_df.empty:
    # Summary metrics
    if "Actual_Profit" in trades_df.columns:
        total_pnl = trades_df["Actual_Profit"].sum()
        total_trades = len(trades_df[trades_df["Action"] != "BUY"])
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
            st.line_chart(pnl_by_date, use_container_width=True)

    # Filter by stock
    if "Stock" in trades_df.columns:
        stocks = ["All"] + sorted(trades_df["Stock"].unique().tolist())
        selected = st.selectbox("Filter by Stock", stocks)
        if selected != "All":
            trades_df = trades_df[trades_df["Stock"] == selected]

    # Show last 50 trades
    st.dataframe(trades_df.tail(50), use_container_width=True, hide_index=True)
else:
    st.info("No trade history yet. Run the Sniper first.")

st.divider()

# ──────────────────────────────────────────────────
#  Active Trades
# ──────────────────────────────────────────────────
st.subheader("🔫 Active Positions")

if state and "active_trades" in state:
    open_trades = [t for t in state["active_trades"] if t.get("status") == "OPEN"]
    if open_trades:
        st.dataframe(pd.DataFrame(open_trades), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions right now.")
else:
    st.info("No state data available.")

# ──────────────────────────────────────────────────
#  Sidebar: Config Info
# ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("""
    - **Models**: RF + XGBoost + Daily LSTM + Intraday LSTM
    - **Voting**: 3-out-of-4 consensus
    - **Confidence**: 75% (RF/XGB), 55% (LSTMs)
    - **Stop Loss**: 1.5 × ATR
    - **Target**: 3.0 × ATR
    - **Daily Target**: 2%
    - **Kill Switch**: -5%
    - **Bullets**: 5
    """)

    st.divider()
    st.markdown("**Project Aegis** v2.0")
    st.markdown(f"Dashboard loaded: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if st.button("🔄 Refresh Data"):
        st.rerun()
