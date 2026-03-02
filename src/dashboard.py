"""
====================================================
PROJECT AEGIS — Live Dashboard v4 (Real-Time Charts)
====================================================
Features:
  - Live clock updating every 2 seconds (no page reload)
  - Live stock prices updating every 10 seconds
  - Interactive Plotly candlestick / line charts per stock
  - Full backtest P&L analysis with equity curve
  - Stock ranking with model confidence
  - Model details and health status
  - Trade history with filtering
  - Learner report and Risk Guardian status

Uses Streamlit Fragments (st.fragment) for partial
re-renders — only the live sections refresh, the rest
of the page stays untouched. Zero page flicker.

Run:
    streamlit run src/dashboard.py
====================================================
"""

import os
import sys
import json
import warnings
import subprocess
import signal
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
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
#  Paths
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MDLS = os.path.join(BASE, "models")

FILE_STATE     = os.path.join(DATA, "daily_state.json")
FILE_TRADES    = os.path.join(DATA, "trade_history.csv")
FILE_RANKING   = os.path.join(DATA, "daily_ranking.csv")
FILE_DASHBOARD = os.path.join(DATA, "dashboard_state.json")
FILE_BACKTEST  = os.path.join(DATA, "backtest_results.csv")
FILE_LEARNER   = os.path.join(DATA, "learner_report.json")
FILE_GUARDIAN  = os.path.join(DATA, "guardian_log.json")
FILE_SNIPER_LOG = os.path.join(DATA, "sniper_output.log")
FILE_ANALYSIS  = os.path.join(DATA, "live_analysis.json")
FILE_WEIGHTS   = os.path.join(DATA, "ensemble_weights.json")
FILE_PARAMS    = os.path.join(DATA, "best_params.json")

IST = pytz.timezone("Asia/Kolkata")

# ──────────────────────────────────────────────────
#  Config import
# ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(BASE, "src"))
try:
    from config import (
        STOCK_WATCHLIST, TOP_N_STOCKS, CAPITAL,
        ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER,
        CONFIDENCE_THRESHOLD, MAX_DAILY_LOSS_PCT,
        DAILY_TARGET, MAX_BULLETS, MIN_VOTES_TO_BUY,
        RF_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SPLIT,
        XGB_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
        LSTM_LOOKBACK, LSTM_EPOCHS, INTRADAY_LOOKBACK, INTRADAY_EPOCHS,
        RF_FEATURES,
    )
except ImportError:
    STOCK_WATCHLIST = [
        "TATASTEEL.NS", "SBIN.NS", "RELIANCE.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
        "INFY.NS", "TCS.NS",
    ]
    TOP_N_STOCKS = 3
    CAPITAL = 1000
    ATR_STOP_MULTIPLIER = 1.5
    ATR_TARGET_MULTIPLIER = 3.0
    CONFIDENCE_THRESHOLD = 0.75
    MAX_DAILY_LOSS_PCT = 0.05
    DAILY_TARGET = 0.02
    MAX_BULLETS = 5
    MIN_VOTES_TO_BUY = 3
    RF_ESTIMATORS = 200
    RF_MAX_DEPTH = 12
    RF_MIN_SPLIT = 50
    XGB_ESTIMATORS = 200
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.05
    LSTM_LOOKBACK = 60
    LSTM_EPOCHS = 10
    INTRADAY_LOOKBACK = 48
    INTRADAY_EPOCHS = 8
    RF_FEATURES = [
        "RSI", "SMA_50", "SMA_200", "EMA_20", "ATR", "MACD",
        "MACD_Signal", "BB_Upper", "BB_Lower", "Volume_Ratio",
        "OBV", "Sentiment_Score",
    ]

# ──────────────────────────────────────────────────
#  Premium CSS Design System
# ──────────────────────────────────────────────────
st.markdown("""<div style="display: none;">
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<style>
/* ── HYPER-MINIMALIST TRUE BLACK THEME ── */

/* ── Hide Streamlit Default Cruft Safely ── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stApp { background-color: #000000 !important; }

/* ── Base Typography ── */
html, body, [class*="css"], p, span, div { font-family: 'Geist', 'Inter', sans-serif !important; color: #f0f0f3 !important; }
h1, h2, h3, h4, h5, h6 { letter-spacing: -0.03em !important; font-weight: 600 !important; color: #ffffff !important; }

/* ── Improved Text Visibility ── */
.stMarkdown, .stMarkdown p, .stMarkdown span { color: #f0f0f3 !important; }
.stCaption, .stCaption p { color: #b0b0b8 !important; font-size: 0.82rem !important; }
li, ul, ol { color: #e8e8ec !important; }
.stAlert p, .stAlert span { color: #e8e8ec !important; }

/* ── Material Icon Helper ── */
.mi { font-family: 'Material Icons Round'; font-size: 16px; vertical-align: middle; margin-right: 6px; opacity: 0.6; }

/* ── Minimalist Metric Cards ── */
div[data-testid="stMetric"] {
    background: transparent;
    border: 1px solid #27272a;
    border-radius: 6px; padding: 12px 16px;
    margin-bottom: 8px;
}
div[data-testid="stMetric"] label { color: #c0c0c8 !important; font-size: 0.78rem !important; letter-spacing: 0.05em; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 500 !important; font-size: 1.6rem !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* ── Flat Hero Banner ── */
.hero-banner { border-bottom: 1px solid #27272a; padding: 16px 0 24px 0; margin-bottom: 24px; }
.hero-title { font-size: 1.5rem; font-weight: 600; color: #ffffff; letter-spacing: -0.04em; margin: 0; display: flex; align-items: center; }
.hero-sub { font-size: 0.85rem; color: #a1a1aa; margin-top: 8px; font-weight: 400; }

/* ── Sharp Status Badges ── */
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.badge-green { background: #052e16; color: #34d399; border: 1px solid #064e3b; }
.badge-red { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }

/* ── Section Headers ── */
.section-header { display: flex; align-items: center; margin: 32px 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid #27272a; }
.section-header h3 { margin: 0; font-weight: 600; font-size: 1.1rem; color: #ffffff !important; }

/* ── True Black Sidebar ── */
section[data-testid="stSidebar"] { background: #09090b !important; border-right: 1px solid #27272a; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] span { color: #d8d8e0 !important; }
section[data-testid="stSidebar"] strong, section[data-testid="stSidebar"] b { color: #ffffff !important; }

/* ── Dataframes & Tables ── */
[data-testid="stDataFrame"] { border: 1px solid #27272a; border-radius: 6px; overflow: hidden; background: #0a0a0c; }
th { color: #d0d0d8 !important; font-size: 0.78rem !important; text-transform: uppercase; font-weight: 600 !important; }
td { color: #f0f0f3 !important; font-size: 0.85rem !important; }

/* ── Expander Background Override ── */
[data-testid="stExpander"] { border: 1px solid #27272a !important; background: transparent !important; }

/* ── Code Blocks (Console output) ── */
pre, code { background: #09090b !important; border: 1px solid #27272a !important; border-radius: 4px; font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important; color: #d4d4d8 !important; padding: 12px !important; }

/* ── Subtle Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 6px; }

/* ── Live Dot ── */
.live-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10b981; margin-right: 8px; animation: gentle-pulse 2s infinite; }
@keyframes gentle-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* ── Footer ── */
.footer { text-align: center; font-size: 0.8rem; color: #b0b0b8; padding: 12px 0; }

/* ── Tab text visibility ── */
button[data-baseweb="tab"] { color: #d0d0d8 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #ffffff !important; font-weight: 600 !important; }

/* ── Info / Warning / Success / Error box text ── */
div[data-testid="stAlert"] p { color: #f0f0f3 !important; }

/* ── Safety: Let Streamlit handle buttons & tabs to prevent overlap ── */
</style>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  SNIPER SUBPROCESS MANAGEMENT
# ══════════════════════════════════════════════════
def _is_sniper_alive() -> bool:
    """Check if the sniper subprocess is still running."""
    proc = st.session_state.get("sniper_proc")
    if proc is None:
        return False
    # poll() returns None if process is still running
    return proc.poll() is None


def _start_sniper():
    """Launch sniper.py as a background subprocess, capturing output to log."""
    if _is_sniper_alive():
        return  # Already running
    os.makedirs(DATA, exist_ok=True)
    log_fh = open(FILE_SNIPER_LOG, "a", encoding="utf-8")
    log_fh.write(f"\n{'='*60}\n")
    log_fh.write(f"  Sniper started at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}\n")
    log_fh.write(f"{'='*60}\n")
    log_fh.flush()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"       # Prevent cp1252 emoji crash on Windows
    env["PYTHONUNBUFFERED"] = "1"            # Disable output buffering for live log
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(BASE, "src", "sniper.py")],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=BASE,
        env=env,
        # On Windows use CREATE_NEW_PROCESS_GROUP; on Unix use preexec_fn
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    st.session_state["sniper_proc"] = proc
    st.session_state["sniper_log_fh"] = log_fh
    st.session_state["sniper_stopped_by_user"] = False


def _stop_sniper():
    """Terminate the sniper subprocess gracefully."""
    proc = st.session_state.get("sniper_proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    # Close log file handle
    log_fh = st.session_state.get("sniper_log_fh")
    if log_fh:
        try:
            log_fh.close()
        except Exception:
            pass
    st.session_state["sniper_proc"] = None
    st.session_state["sniper_log_fh"] = None
    st.session_state["sniper_stopped_by_user"] = True


def _get_sniper_log_tail(n_lines=25) -> str:
    """Read the last N lines from the sniper log file."""
    if not os.path.exists(FILE_SNIPER_LOG):
        return "(no log output yet)"
    try:
        with open(FILE_SNIPER_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        return "".join(tail)
    except Exception:
        return "(could not read log)"


# ══════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def fetch_prices_live(symbols):
    """Fetch live prices (NO cache — used inside auto-refresh fragment)."""
    import yfinance as yf
    prices, changes, highs, lows, vols = {}, {}, {}, {}, {}
    for s in symbols:
        try:
            info = yf.Ticker(s).fast_info
            p = float(getattr(info, "last_price", 0) or 0)
            prev = float(getattr(info, "previous_close", 0) or 0)
            prices[s] = round(p, 2)
            changes[s] = round(((p - prev) / prev) * 100, 2) if prev > 0 else 0.0
            highs[s] = round(float(getattr(info, "day_high", 0) or 0), 2)
            lows[s] = round(float(getattr(info, "day_low", 0) or 0), 2)
            vols[s] = int(getattr(info, "last_volume", 0) or 0)
        except Exception:
            prices[s] = 0.0
            changes[s] = 0.0
            highs[s] = 0.0
            lows[s] = 0.0
            vols[s] = 0
    return prices, changes, highs, lows, vols


@st.cache_data(ttl=45)
def fetch_chart_data(symbol, period, interval):
    """Cached intraday/daily OHLCV for charts. Refreshes every 45s."""
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def make_chart(df, symbol, style="candlestick"):
    """Build a Plotly candlestick or line chart with volume."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    name = symbol.replace(".NS", "")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.75, 0.25],
    )

    has_ohlc = {"Open", "High", "Low", "Close"}.issubset(df.columns)

    if style == "candlestick" and has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
            name=name,
        ), row=1, col=1)
    elif "Close" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], mode="lines",
            line=dict(color="#42a5f5", width=1.5),
            fill="tozeroy", fillcolor="rgba(66,165,245,0.08)",
            name=name,
        ), row=1, col=1)

    if "Volume" in df.columns and has_ohlc:
        clrs = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["Close"].values, df["Open"].values)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], marker_color=clrs,
            opacity=0.35, showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=370,
        margin=dict(l=0, r=5, t=35, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"  {name}", x=0, font=dict(size=14, color="#e0e0e0")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis2=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def get_model_files_info():
    """Scan models/ and return inventory."""
    info = []
    if os.path.isdir(MDLS):
        for f in sorted(os.listdir(MDLS)):
            fp = os.path.join(MDLS, f)
            if not os.path.isfile(fp):
                continue
            sz = os.path.getsize(fp) / 1024
            mt = datetime.fromtimestamp(os.path.getmtime(fp))
            t = "Unknown"
            if "_rf.pkl" in f:
                t = "Random Forest"
            elif "_xgb.pkl" in f:
                t = "XGBoost"
            elif "_intraday_lstm" in f:
                t = "Intraday LSTM"
            elif "_lstm" in f and "scaler" not in f:
                t = "Daily LSTM"
            elif "_intraday_scaler" in f:
                t = "Intraday Scaler"
            elif "_scaler.pkl" in f:
                t = "Scaler"
            stk = (
                f.split("_NS_")[0].replace("_", ".") + ".NS"
                if "_NS_" in f
                else f.split("_")[0]
            )
            info.append({
                "File": f,
                "Stock": stk,
                "Type": t,
                "Size (KB)": round(sz, 1),
                "Trained": mt.strftime("%Y-%m-%d %H:%M"),
            })
    return info


# ══════════════════════════════════════════════════
#  SYMBOLS (from ranking or watchlist)
# ══════════════════════════════════════════════════
_rank = load_csv(FILE_RANKING)
ALL_SYMS = (
    _rank["symbol"].tolist()
    if (not _rank.empty and "symbol" in _rank.columns)
    else STOCK_WATCHLIST
)


# ══════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-header">⚙️ <h3>Configuration</h3></div>', unsafe_allow_html=True)
    st.markdown(f"""
- **Watchlist:** {len(STOCK_WATCHLIST)} stocks
- **Top-N:** {TOP_N_STOCKS}
- **Capital:** ₹{CAPITAL:,}
- **Bullet Size:** ₹{CAPITAL // MAX_BULLETS}
- **Models:** RF + XGB + LSTM + Intraday
- **Voting:** {MIN_VOTES_TO_BUY}-out-of-4 consensus
- **Confidence:** {CONFIDENCE_THRESHOLD}
- **Stop Loss:** {ATR_STOP_MULTIPLIER}× ATR
- **Target:** {ATR_TARGET_MULTIPLIER}× ATR
- **Daily Target:** {DAILY_TARGET * 100:.0f}%
- **Kill Switch:** -{MAX_DAILY_LOSS_PCT * 100:.0f}%
    """)

    st.divider()
    st.markdown('<div class="section-header">🧠 <h3>Model Config</h3></div>', unsafe_allow_html=True)
    st.markdown(f"""
| Model | Config |
|-------|--------|
| RF Trees | {RF_ESTIMATORS} |
| RF Depth | {RF_MAX_DEPTH} |
| XGB Rounds | {XGB_ESTIMATORS} |
| XGB Depth | {XGB_MAX_DEPTH} |
| XGB LR | {XGB_LEARNING_RATE} |
| LSTM Days | {LSTM_LOOKBACK} |
| LSTM Epochs | {LSTM_EPOCHS} |
| Intra Candles | {INTRADAY_LOOKBACK} |
| Intra Epochs | {INTRADAY_EPOCHS} |
    """)

    st.divider()
    st.markdown('🔒 **100% Local** — No data leaves your PC', unsafe_allow_html=True)

    if st.button("Refresh All Data", use_container_width=True, icon="🔄"):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════
#  HEADER — Hero Banner + Live Clock
# ══════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ Project Aegis</div>
    <div class="hero-sub">4-Model AI Ensemble · Real-Time Charts · Plotly · 100% Local</div>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=2)
def _live_clock():
    """Header clock — refreshes every 2s with zero page reload."""
    n = datetime.now(IST)
    wd = n.weekday() < 5
    m = n.hour * 60 + n.minute
    is_open = wd and (9 * 60 + 15 <= m <= 15 * 60 + 30)

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.markdown(
            f'<div style="display: flex; align-items: center;"><span class="live-dot"></span>'
            f' <strong>{n.strftime("%Y-%m-%d %H:%M:%S")} IST</strong></div>',
            unsafe_allow_html=True,
        )
    with c2:
        if is_open:
            st.markdown('<span class="badge badge-green">🟢 MARKET OPEN</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-red">🔴 MARKET CLOSED</span>', unsafe_allow_html=True)
    with c3:
        mc = len(os.listdir(MDLS)) if os.path.isdir(MDLS) else 0
        st.markdown(f'<span class="badge badge-blue">💾 {mc} models</span>', unsafe_allow_html=True)


_live_clock()
st.divider()


# ══════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Trading",
    "Backtest P&L",
    "Ranking & Signals",
    "Model Details",
    "Trade History",
    "Learner & Guardian",
])


# ══════════════════════════════════════════════════
#  TAB 1 — LIVE TRADING & CHARTS
# ══════════════════════════════════════════════════
with tab1:

    # ── LIVE PRICES (auto-refresh every 10s, no page reload) ──
    @st.fragment(run_every=10)
    def _live_prices():
        st.markdown('<div class="section-header">📈 <h3>Live Stock Prices</h3></div>', unsafe_allow_html=True)
        st.caption("Auto-refreshing every 10 seconds — no page reload")

        prices, changes, highs, lows, vols = fetch_prices_live(ALL_SYMS)

        # Row 1: first 5 stocks
        row1_n = min(len(ALL_SYMS), 5)
        cols = st.columns(row1_n)
        for i in range(row1_n):
            s = ALL_SYMS[i]
            with cols[i]:
                st.metric(
                    s.replace(".NS", ""),
                    f"₹{prices[s]:,.2f}",
                    f"{changes[s]:+.2f}%",
                )

        # Row 2: next 5 stocks
        if len(ALL_SYMS) > 5:
            row2_n = min(len(ALL_SYMS) - 5, 5)
            cols2 = st.columns(row2_n)
            for i in range(row2_n):
                s = ALL_SYMS[5 + i]
                with cols2[i]:
                    st.metric(
                        s.replace(".NS", ""),
                        f"₹{prices[s]:,.2f}",
                        f"{changes[s]:+.2f}%",
                    )

        # Detail table
        rows = []
        for s in ALL_SYMS:
            ch = changes[s]
            rows.append({
                "Stock": s.replace(".NS", ""),
                "Price ₹": prices[s],
                "Change %": ch,
                "Day High ₹": highs[s],
                "Day Low ₹": lows[s],
                "Volume": f"{vols[s]:,}",
                "Signal": "🟢 UP" if ch > 0.5 else ("🔴 DOWN" if ch < -0.5 else "⚪ FLAT"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    _live_prices()

    st.divider()

    # ── STOCK CHARTS (interactive Plotly) ──
    st.markdown('<div class="section-header">📈 <h3>Live Stock Market Charts</h3></div>', unsafe_allow_html=True)
    st.caption("Interactive candlestick & line charts — zoom, pan, hover for details")

    cc1, cc2, cc3 = st.columns([1, 1, 2])
    with cc1:
        chart_style = st.selectbox(
            "Chart Style", ["Candlestick", "Line"], key="cstyle"
        )
    with cc2:
        TF_MAP = {
            "Today (5m candles)": ("1d", "5m"),
            "5 Days (15m)": ("5d", "15m"),
            "1 Month (1h)": ("1mo", "1h"),
            "3 Months (daily)": ("3mo", "1d"),
            "6 Months (daily)": ("6mo", "1d"),
            "1 Year (daily)": ("1y", "1d"),
        }
        tf_choice = st.selectbox("Timeframe", list(TF_MAP.keys()), key="ctf")
    with cc3:
        chart_stocks = st.multiselect(
            "Select Stocks to Chart",
            ALL_SYMS,
            default=ALL_SYMS[: min(TOP_N_STOCKS, len(ALL_SYMS))],
            format_func=lambda x: x.replace(".NS", ""),
            key="cstocks",
        )

    if chart_stocks:
        _per, _intv = TF_MAP[tf_choice]
        _cs = chart_style.lower()

        progress = st.progress(0, text="Loading charts...")
        chart_cols = st.columns(2)
        for idx, sym in enumerate(chart_stocks):
            with chart_cols[idx % 2]:
                try:
                    cdf = fetch_chart_data(sym, _per, _intv)
                    if not cdf.empty:
                        fig = make_chart(cdf, sym, _cs)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No data for {sym.replace('.NS', '')}")
                except Exception as e:
                    st.warning(f"{sym.replace('.NS','')}: {e}")
            progress.progress(
                (idx + 1) / len(chart_stocks),
                text=f"Loaded {sym.replace('.NS', '')}",
            )
        progress.empty()
    else:
        st.info("Select stocks above to view charts.")

    st.divider()

    # ── SNIPER STATUS (with auto-start) ──
    st.markdown('<div class="section-header">🎯 <h3>Sniper Status</h3></div>', unsafe_allow_html=True)
    _dash = load_json(FILE_DASHBOARD)
    _state = load_json(FILE_STATE)

    # Determine if market is open right now
    _n = datetime.now(IST)
    _mkt_open = (_n.weekday() < 5) and (
        9 * 60 + 15 <= _n.hour * 60 + _n.minute <= 15 * 60 + 30
    )
    _sniper_running = _is_sniper_alive()

    # ── Auto-start: launch sniper if market open & not running & user hasn't stopped it ──
    if _mkt_open and not _sniper_running and not st.session_state.get("sniper_stopped_by_user", False):
        _start_sniper()
        _sniper_running = True
        st.toast("🚀 Sniper auto-started! Trading has begun.", icon="🔫")

    # ── Show dashboard state metrics if available ──
    if _dash:
        st.markdown(f"**Last Update:** {_dash.get('last_updated', 'N/A')}")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            _pnl = _dash.get("total_profit", 0)
            st.metric("Day P&L", f"₹{_pnl:,.2f}", f"{_dash.get('total_profit_pct', 0):+.2f}%")
        with c2:
            st.metric("Status", _dash.get("status", "N/A"))
        with c3:
            st.metric("Trades", _dash.get("trades_taken", 0))
        with c4:
            w = _dash.get("trades_won", 0)
            tk = _dash.get("trades_taken", 0)
            st.metric("Win Rate", f"{w / tk * 100:.0f}%" if tk else "N/A")
        with c5:
            st.metric("Open Pos", _dash.get("active_count", 0))

    # ── Sniper process status + controls ──
    if _sniper_running:
        st.success("🟢 Sniper is **RUNNING** — AI ensemble is actively scanning & trading.")
        bc1, bc2 = st.columns([1, 5])
        with bc1:
            if st.button("🛑 Stop Sniper", type="primary", use_container_width=True, key="stop_sniper"):
                _stop_sniper()
                st.rerun()

    st.divider()

    # ══════════════════════════════════════════════
    #  LIVE AI ANALYSIS — Auto-refreshing section
    # ══════════════════════════════════════════════
    st.markdown('<div class="section-header">🧠 <h3>Live AI Analysis</h3></div>', unsafe_allow_html=True)

    _analysis = load_json(FILE_ANALYSIS)
    if not _analysis or "stocks" not in _analysis:
        st.info("⏳ Waiting for first scan cycle... Start the Sniper to see live AI analysis.")
    else:
        # Header row: last scan time, scan number, day P&L
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            st.metric("Last Scan", _analysis.get("timestamp", "N/A").split(" ")[-2] if " " in _analysis.get("timestamp", "") else _analysis.get("timestamp", "N/A"))
        with ac2:
            st.metric("Scan #", _analysis.get("scan_number", 0))
        with ac3:
            _apnl = _analysis.get("day_pnl", 0)
            st.metric("Day P&L", f"₹{_apnl:,.2f}", f"{_analysis.get('day_pnl_pct', 0):+.2f}%")
        with ac4:
            st.metric("Open / Bullets", f"{_analysis.get('open_positions', 0)} / {_analysis.get('bullets_left', 0)}")

        # Per-stock analysis cards
        stocks_data = _analysis.get("stocks", [])
        if stocks_data:
            for s in stocks_data:
                signal = s.get("signal", "WAIT")
                signal_color = "#10b981" if signal == "BUY" else "#f59e0b"
                signal_icon = "🟢" if signal == "BUY" else "🟡"
                name = s.get("name", "???")
                price = s.get("price", 0)
                change = s.get("change_pct", 0)
                change_color = "#10b981" if change >= 0 else "#ef4444"

                # Card header
                st.markdown(f"""
                <div style="border: 1px solid #27272a; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: #09090b;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <span style="font-size: 1.15rem; font-weight: 600; color: #ffffff;">{name}</span>
                            <span style="font-size: 0.85rem; color: #a1a1aa; margin-left: 10px;">₹{price:,.2f}</span>
                            <span style="font-size: 0.8rem; color: {change_color}; margin-left: 6px;">({change:+.2f}%)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 0.75rem; background: {'#052e16' if signal=='BUY' else '#451a03'}; color: {signal_color}; padding: 4px 10px; border-radius: 4px; font-weight: 600; border: 1px solid {'#064e3b' if signal=='BUY' else '#78350f'};">{signal_icon} {signal}</span>
                            <span style="font-size: 0.75rem; color: #a1a1aa; background: #18181b; padding: 4px 8px; border-radius: 4px; border: 1px solid #27272a;">{s.get('votes', 0)}/4 Votes</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Model confidence bars
                models = [
                    ("Random Forest", s.get("rf_conf", 0), "#3b82f6"),
                    ("XGBoost", s.get("xgb_conf", 0), "#8b5cf6"),
                    ("LSTM (Daily)", s.get("lstm_conf", 0), "#06b6d4"),
                    ("LSTM (Intraday)", s.get("intra_conf", 0), "#f59e0b"),
                ]
                bars_html = ""
                for mname, mconf, mcolor in models:
                    pct = min(max(mconf * 100, 0), 100)
                    vote_icon = "✅" if mconf >= 0.5 else "❌"
                    bars_html += f"""
                    <div style="display: flex; align-items: center; margin-bottom: 6px;">
                        <div style="width: 110px; font-size: 0.72rem; color: #a1a1aa;">{mname}</div>
                        <div style="flex: 1; background: #18181b; border-radius: 3px; height: 14px; margin: 0 8px; border: 1px solid #27272a; overflow: hidden;">
                            <div style="width: {pct:.0f}%; height: 100%; background: {mcolor}; border-radius: 3px; transition: width 0.5s;"></div>
                        </div>
                        <div style="width: 55px; font-size: 0.72rem; color: #e4e4e7; text-align: right;">{mconf:.2%}</div>
                        <div style="width: 20px; text-align: center; font-size: 0.7rem;">{vote_icon}</div>
                    </div>
                    """

                st.markdown(bars_html, unsafe_allow_html=True)

                # Technical indicators row
                rsi = s.get("rsi", 0)
                rsi_color = "#ef4444" if rsi > 70 else ("#10b981" if rsi < 30 else "#a1a1aa")
                sent = s.get("sentiment", 0)
                sent_label = "Bullish" if sent > 0.1 else ("Bearish" if sent < -0.1 else "Neutral")
                sent_color = "#10b981" if sent > 0.1 else ("#ef4444" if sent < -0.1 else "#a1a1aa")

                tech_html = f"""
                    <div style="display: flex; gap: 16px; margin-top: 8px; flex-wrap: wrap;">
                        <span style="font-size: 0.7rem; color: #a1a1aa;">RSI: <span style="color: {rsi_color}; font-weight: 500;">{rsi:.1f}</span></span>
                        <span style="font-size: 0.7rem; color: #a1a1aa;">ATR: <span style="color: #e4e4e7;">{s.get('atr', 0):.2f}</span></span>
                        <span style="font-size: 0.7rem; color: #a1a1aa;">MACD: <span style="color: #e4e4e7;">{s.get('macd', 0):.4f}</span></span>
                        <span style="font-size: 0.7rem; color: #a1a1aa;">Sentiment: <span style="color: {sent_color};">{sent_label} ({sent:.3f})</span></span>
                        <span style="font-size: 0.7rem; color: #a1a1aa;">Vol Ratio: <span style="color: #e4e4e7;">{s.get('volume_ratio', 0):.2f}x</span></span>
                    </div>
                """
                if signal == "BUY" and s.get("stop_loss", 0) > 0:
                    tech_html += f"""
                    <div style="display: flex; gap: 16px; margin-top: 6px;">
                        <span style="font-size: 0.7rem; color: #a1a1aa;">SL: <span style="color: #ef4444; font-weight: 500;">₹{s.get('stop_loss', 0):,.2f}</span></span>
                        <span style="font-size: 0.7rem; color: #a1a1aa;">Target: <span style="color: #10b981; font-weight: 500;">₹{s.get('target', 0):,.2f}</span></span>
                    </div>
                    """

                # Guardian gate status
                g_status = s.get("guardian", "")
                g_reason = s.get("guardian_reason", "")
                if g_status == "APPROVED":
                    tech_html += f"""
                    <div style="margin-top: 8px; padding: 6px 10px; background: #052e16; border: 1px solid #064e3b; border-radius: 4px;">
                        <span style="font-size: 0.72rem; color: #34d399; font-weight: 500;">✅ Guardian: APPROVED — {g_reason}</span>
                    </div>
                    """
                elif g_status == "BLOCKED":
                    tech_html += f"""
                    <div style="margin-top: 8px; padding: 6px 10px; background: #450a0a; border: 1px solid #7f1d1d; border-radius: 4px;">
                        <span style="font-size: 0.72rem; color: #f87171; font-weight: 500;">🛡️ Guardian: BLOCKED — {g_reason}</span>
                    </div>
                    """

                st.markdown(tech_html + "</div>", unsafe_allow_html=True)

                # Reason for this stock
                reason_text = s.get("reason", "")
                if reason_text:
                    st.caption(f"   ↳ {reason_text}")

        # Summary bar
        buy_count = sum(1 for s in stocks_data if s.get("signal") == "BUY")
        wait_count = len(stocks_data) - buy_count
        avg_conf = sum(s.get("avg_conf", 0) for s in stocks_data) / max(len(stocks_data), 1)
        st.markdown(f"""
        <div style="border: 1px solid #27272a; border-radius: 6px; padding: 10px 16px; margin-top: 10px; background: #09090b; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.8rem; color: #a1a1aa;">Scanning <b style="color: #fff;">{len(stocks_data)}</b> stocks</span>
            <span style="font-size: 0.8rem; color: #10b981;">🟢 {buy_count} BUY</span>
            <span style="font-size: 0.8rem; color: #f59e0b;">🟡 {wait_count} WAIT</span>
            <span style="font-size: 0.8rem; color: #a1a1aa;">Avg Confidence: <b style="color: #fff;">{avg_conf:.2%}</b></span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Live Trades Ledger (Enhanced with Outcomes) ──
    st.markdown('<div class="section-header">🧾 <h3>Today\'s Live Trades</h3></div>', unsafe_allow_html=True)
    
    trades_df = load_csv(FILE_TRADES)
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    
    if trades_df.empty:
        st.info("No trades executed yet today.")
    else:
        # Ensure numeric columns
        for _nc in ["Entry_Price", "Exit_Price", "Qty", "AI_Confidence", "Votes", "Actual_Profit", "Stop_Loss", "Target"]:
            if _nc in trades_df.columns:
                trades_df[_nc] = pd.to_numeric(trades_df[_nc], errors="coerce").fillna(0)

        # Filter for today's trades only
        today_trades = trades_df[trades_df["Date"].str.startswith(today_str, na=False)].copy()
        
        if today_trades.empty:
            st.info("No trades executed yet today.")
        else:
            # ── Summary Stats Bar ──
            total_today = len(today_trades)
            buys_today = len(today_trades[today_trades["Action"] == "BUY"])
            sells_today = total_today - buys_today
            open_trades = len(today_trades[today_trades["Status"] == "OPEN"]) if "Status" in today_trades.columns else 0
            closed_trades = total_today - open_trades
            day_realized_pnl = today_trades["Actual_Profit"].sum() if "Actual_Profit" in today_trades.columns else 0
            wins_today = len(today_trades[today_trades["Actual_Profit"] > 0]) if "Actual_Profit" in today_trades.columns else 0
            losses_today = len(today_trades[today_trades["Actual_Profit"] < 0]) if "Actual_Profit" in today_trades.columns else 0

            # Compute unrealized P&L for open positions from live state
            _live_state = load_json(FILE_STATE)
            unrealized_pnl = 0
            active_trades_info = []
            if _live_state and "active_trades" in _live_state:
                for _at in _live_state["active_trades"]:
                    if _at.get("status") == "OPEN":
                        active_trades_info.append(_at)

            pnl_color = "#10b981" if day_realized_pnl >= 0 else "#ef4444"
            st.markdown(f"""
            <div style="border: 1px solid #27272a; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; background: #09090b;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <span style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.05em;">Total Trades</span>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #fff;">{total_today}</div>
                    </div>
                    <div>
                        <span style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase;">Buys</span>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #10b981;">🟢 {buys_today}</div>
                    </div>
                    <div>
                        <span style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase;">Sells / Exits</span>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #ef4444;">🔴 {sells_today}</div>
                    </div>
                    <div>
                        <span style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase;">Open</span>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #f59e0b;">{open_trades}</div>
                    </div>
                    <div>
                        <span style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase;">Wins / Losses</span>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #fff;">
                            <span style="color: #10b981;">{wins_today}W</span> / <span style="color: #ef4444;">{losses_today}L</span>
                        </div>
                    </div>
                    <div>
                        <span style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase;">Realized P&L</span>
                        <div style="font-size: 1.3rem; font-weight: 600; color: {pnl_color};">₹{day_realized_pnl:,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Open Positions Monitor ──
            open_pos_df = today_trades[today_trades.get("Status", pd.Series(dtype=str)).eq("OPEN")] if "Status" in today_trades.columns else pd.DataFrame()
            if not open_pos_df.empty:
                st.markdown("**🔓 Open Positions:**")
                for _, orow in open_pos_df.iterrows():
                    _sym_name = str(orow.get('Stock', '')).replace('.NS', '')
                    _entry = orow.get('Entry_Price', 0)
                    _sl = orow.get('Stop_Loss', 0)
                    _tgt = orow.get('Target', 0)
                    _qty = int(orow.get('Qty', 0))
                    _conf = orow.get('AI_Confidence', 0)
                    _risk = (_entry - _sl) * _qty if _sl > 0 else 0
                    _reward = (_tgt - _entry) * _qty if _tgt > 0 else 0
                    _rr = _reward / _risk if _risk > 0 else 0

                    st.markdown(f"""
                    <div style="border: 1px solid #27272a; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; background: #0c0c10; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <span style="font-size: 1rem; font-weight: 600; color: #fff;">{_sym_name}</span>
                            <span style="font-size: 0.7rem; background: #052e16; color: #34d399; padding: 2px 6px; border-radius: 3px; margin-left: 8px; border: 1px solid #064e3b;">OPEN</span>
                        </div>
                        <span style="font-size: 0.8rem; color: #e4e4e7;">Entry: <b style="color: #fff;">₹{_entry:,.2f}</b></span>
                        <span style="font-size: 0.8rem; color: #ef4444;">SL: ₹{_sl:,.2f}</span>
                        <span style="font-size: 0.8rem; color: #10b981;">Target: ₹{_tgt:,.2f}</span>
                        <span style="font-size: 0.8rem; color: #a1a1aa;">Qty: {_qty} | Conf: {_conf:.2f} | R:R {_rr:.1f}x</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Closed Trades with P&L ──
            closed_df = today_trades[today_trades.get("Status", pd.Series(dtype=str)).ne("OPEN")] if "Status" in today_trades.columns else pd.DataFrame()
            if not closed_df.empty:
                st.markdown("**🔒 Closed Trades:**")
                for _, crow in closed_df.iterrows():
                    _sym_name = str(crow.get('Stock', '')).replace('.NS', '')
                    _entry = crow.get('Entry_Price', 0)
                    _exit = crow.get('Exit_Price', 0)
                    _pnl = crow.get('Actual_Profit', 0)
                    _qty = int(crow.get('Qty', 0))
                    _action = str(crow.get('Action', ''))
                    _is_win = _pnl > 0
                    _pnl_color = '#10b981' if _is_win else '#ef4444'
                    _badge = '✅ WIN' if _is_win else ('❌ LOSS' if _pnl < 0 else '⚪ EVEN')
                    _badge_bg = '#052e16' if _is_win else ('#450a0a' if _pnl < 0 else '#27272a')
                    _badge_border = '#064e3b' if _is_win else ('#7f1d1d' if _pnl < 0 else '#3f3f46')
                    _badge_txt = '#34d399' if _is_win else ('#f87171' if _pnl < 0 else '#a1a1aa')

                    st.markdown(f"""
                    <div style="border: 1px solid #27272a; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; background: #0c0c10; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <span style="font-size: 1rem; font-weight: 600; color: #fff;">{_sym_name}</span>
                        <span style="font-size: 0.8rem; color: #a1a1aa;">{_action}</span>
                        <span style="font-size: 0.8rem; color: #e4e4e7;">₹{_entry:,.2f} → ₹{_exit:,.2f}</span>
                        <span style="font-size: 0.8rem; color: {_pnl_color}; font-weight: 600;">P&L: ₹{_pnl:,.2f}</span>
                        <span style="font-size: 0.7rem; background: {_badge_bg}; color: {_badge_txt}; padding: 3px 8px; border-radius: 4px; border: 1px solid {_badge_border}; font-weight: 600;">{_badge}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Full Trades Table ──
            st.markdown("**📋 All Today's Trades (Table):**")
            _display_cols = [c for c in ["Time", "Stock", "Action", "Entry_Price", "Exit_Price", "Qty", "Stop_Loss", "Target", "AI_Confidence", "Votes", "Actual_Profit", "Status"] if c in today_trades.columns]
            ledger = today_trades[_display_cols].copy()

            # Format for display
            if "Time" in ledger.columns:
                pass  # Time column already has just time
            ledger["Stock"] = ledger["Stock"].str.replace(".NS", "", regex=False)
            if "Entry_Price" in ledger.columns:
                ledger["Entry_Price"] = ledger["Entry_Price"].apply(lambda x: f"₹{x:,.2f}")
            if "Exit_Price" in ledger.columns:
                ledger["Exit_Price"] = ledger["Exit_Price"].apply(lambda x: f"₹{x:,.2f}" if x > 0 else "—")
            if "Stop_Loss" in ledger.columns:
                ledger["Stop_Loss"] = ledger["Stop_Loss"].apply(lambda x: f"₹{x:,.2f}")
            if "Target" in ledger.columns:
                ledger["Target"] = ledger["Target"].apply(lambda x: f"₹{x:,.2f}")
            if "AI_Confidence" in ledger.columns:
                ledger.rename(columns={"AI_Confidence": "Confidence"}, inplace=True)
                ledger["Confidence"] = ledger["Confidence"].apply(lambda x: f"{x:.2f}")
            if "Votes" in ledger.columns:
                ledger["Votes"] = ledger["Votes"].apply(lambda x: f"{int(x)}/4")
            if "Actual_Profit" in ledger.columns:
                ledger.rename(columns={"Actual_Profit": "P&L"}, inplace=True)
                ledger["P&L"] = ledger["P&L"].apply(lambda x: f"₹{x:,.2f}" if x != 0 else "—")

            # Action formatting
            def format_action(val):
                if val == "BUY": return "🟢 BUY"
                if val == "SELL": return "🔴 SELL"
                if "EXIT" in str(val): return "⚪ EXIT"
                if "FORCE" in str(val): return "🟠 FORCE_CLOSE"
                return val
            if "Action" in ledger.columns:
                ledger["Action"] = ledger["Action"].apply(format_action)

            # Status formatting
            def format_status(val):
                if val == "OPEN": return "🟢 OPEN"
                if "HIT_TARGET" in str(val): return "🎯 TARGET"
                if "HIT_SL" in str(val): return "🛑 STOP LOSS"
                if "FORCE" in str(val): return "🟠 FORCED"
                return str(val)
            if "Status" in ledger.columns:
                ledger["Status"] = ledger["Status"].apply(format_status)

            st.dataframe(ledger, use_container_width=True, hide_index=True)

    # ── Technical Logs Expander ──
    if os.path.exists(FILE_SNIPER_LOG):
        with st.expander("🛠️ Raw Sniper Console (For debugging)", expanded=False):
            st.code(_get_sniper_log_tail(15), language="text")

    st.divider()    


    # ── SYSTEM HEALTH ──
    st.markdown('<div class="section-header">❤️‍🩹 <h3>System Health</h3></div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    _mf = get_model_files_info()
    with h1:
        st.metric("Model Files", len(_mf))
    with h2:
        st.metric(
            "Ranking",
            "✅ Ready" if os.path.exists(FILE_RANKING) and os.path.getsize(FILE_RANKING) > 0 else "❌ Missing",
        )
    with h3:
        st.metric(
            "Backtest",
            "✅ Ready" if os.path.exists(FILE_BACKTEST) and os.path.getsize(FILE_BACKTEST) > 0 else "❌ Missing",
        )
    with h4:
        st.metric(
            "Learner",
            "✅ Ready" if os.path.exists(FILE_LEARNER) and os.path.getsize(FILE_LEARNER) > 0 else "❌ Not Run",
        )


# ══════════════════════════════════════════════════
#  TAB 2 — BACKTEST P&L ANALYSIS
# ══════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 <h3>Backtest P&L Analysis — Full Report</h3></div>', unsafe_allow_html=True)

    bt = load_csv(FILE_BACKTEST)

    if bt.empty:
        st.warning("⚠️ No backtest data. Run: `python src/backtest.py`")
    else:
        # Separate trades from summary rows
        smask = bt["Date"].astype(str).isin(["---", "", "nan", "None"])
        trades = bt[~smask].copy()
        summrows = bt[smask]

        # Parse summary
        sd = {}
        for _, r in summrows.iterrows():
            k = str(r.get("Stock", "")).strip()
            if k and k not in ("=== SUMMARY ===", "nan", "None", ""):
                for c in ["PnL", "Invested", "Qty", "PnL_%"]:
                    if c in r and pd.notna(r[c]):
                        try:
                            sd[k] = float(r[c])
                            break
                        except (ValueError, TypeError):
                            pass

        cap0 = sd.get("Starting Capital", CAPITAL)
        tot_inv = sd.get("Total Invested", 0)
        n_trades = int(sd.get("Total Trades", len(trades)))
        n_wins = int(sd.get("Winning Trades", 0))
        n_loss = int(sd.get("Losing Trades", 0))
        wr = sd.get("Win Rate %", 0)
        tot_pnl = sd.get("Total Profit/Loss", 0)
        roc = sd.get("Return on Capital %", 0)
        avg_t = sd.get("Avg PnL per Trade", 0)
        best_t = sd.get("Best Trade", 0)
        worst_t = sd.get("Worst Trade", 0)
        mdd = sd.get("Max Drawdown", 0)
        cap_f = sd.get("Final Capital", cap0 + tot_pnl)

        # Fallback calc from trades
        if not trades.empty and tot_pnl == 0 and "PnL" in trades.columns:
            trades["PnL"] = pd.to_numeric(trades["PnL"], errors="coerce")
            tot_pnl = trades["PnL"].sum()
            n_wins = int((trades["PnL"] > 0).sum())
            n_loss = int((trades["PnL"] <= 0).sum())
            n_trades = len(trades)
            wr = n_wins / n_trades * 100 if n_trades else 0
            roc = tot_pnl / cap0 * 100
            cap_f = cap0 + tot_pnl
            if "Invested" in trades.columns:
                trades["Invested"] = pd.to_numeric(trades["Invested"], errors="coerce")
                tot_inv = trades["Invested"].sum()
            if n_trades:
                avg_t = tot_pnl / n_trades
                best_t = trades["PnL"].max()
                worst_t = trades["PnL"].min()
            if "Cumulative_PnL" in trades.columns:
                trades["Cumulative_PnL"] = pd.to_numeric(trades["Cumulative_PnL"], errors="coerce")
                mdd = (trades["Cumulative_PnL"].cummax() - trades["Cumulative_PnL"]).max()

        profit = (tot_pnl or 0) >= 0

        # Headline
        if profit:
            st.success(f"### ✅ NET PROFIT: ₹{tot_pnl:,.2f}  ({roc:+.2f}% return)")
        else:
            st.error(f"### ❌ NET LOSS: ₹{tot_pnl:,.2f}  ({roc:+.2f}% return)")

        st.divider()

        # Portfolio Summary
        st.markdown('<div class="section-header">💼 <h3>Portfolio Summary</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Starting Capital", f"₹{cap0:,.2f}")
        with c2:
            st.metric("Total Invested", f"₹{tot_inv:,.2f}")
        with c3:
            st.metric("Final Capital", f"₹{cap_f:,.2f}", f"{roc:+.2f}%")
        with c4:
            st.metric("Net P&L", f"₹{tot_pnl:,.2f}", "PROFIT ✅" if profit else "LOSS ❌")

        st.divider()

        # Trade Stats
        st.markdown('<div class="section-header">📉 <h3>Trade Statistics</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Trades", n_trades)
        with c2:
            st.metric("Wins ✅", n_wins)
        with c3:
            st.metric("Losses ❌", n_loss)
        with c4:
            st.metric("Win Rate", f"{wr:.1f}%")
        with c5:
            st.metric("Avg P&L/Trade", f"₹{avg_t:,.2f}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Best Trade 🏆", f"₹{best_t:,.2f}")
        with c2:
            st.metric("Worst Trade 💀", f"₹{worst_t:,.2f}")
        with c3:
            st.metric("Max Drawdown 📉", f"₹{mdd:,.2f}")

        st.divider()

        # Equity Curve (Plotly)
        if not trades.empty and "Cumulative_PnL" in trades.columns:
            st.markdown('<div class="section-header">📉 <h3>Equity Curve</h3></div>', unsafe_allow_html=True)
            trades["Cumulative_PnL"] = pd.to_numeric(trades["Cumulative_PnL"], errors="coerce")
            edf = trades[["Date", "Cumulative_PnL"]].dropna()
            if not edf.empty:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=edf["Date"], y=edf["Cumulative_PnL"],
                    mode="lines", fill="tozeroy",
                    line=dict(color="#42a5f5", width=2),
                    fillcolor="rgba(66,165,245,0.1)",
                ))
                fig.update_layout(
                    template="plotly_dark", height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14,17,23,0.9)",
                    yaxis_title="Cumulative P&L (₹)",
                    xaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
                    yaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.divider()

        # P&L by Stock
        if not trades.empty and "Stock" in trades.columns:
            st.markdown('<div class="section-header">🏢 <h3>P&L Breakdown by Stock</h3></div>', unsafe_allow_html=True)
            trades["PnL"] = pd.to_numeric(trades["PnL"], errors="coerce")

            ss = trades.groupby("Stock").agg(
                Trades=("PnL", "count"),
                Wins=("PnL", lambda x: (x > 0).sum()),
                Total_PnL=("PnL", "sum"),
                Avg_PnL=("PnL", "mean"),
                Best=("PnL", "max"),
                Worst=("PnL", "min"),
            ).round(2)
            ss["Win_%"] = (ss["Wins"] / ss["Trades"] * 100).round(1)

            if "Invested" in trades.columns:
                trades["Invested"] = pd.to_numeric(trades["Invested"], errors="coerce")
                ss["Invested"] = trades.groupby("Stock")["Invested"].sum().round(2)
                ss["ROI_%"] = (ss["Total_PnL"] / ss["Invested"] * 100).round(2)

            st.dataframe(ss, use_container_width=True)

            # Bar chart
            import plotly.express as px

            fig = px.bar(
                ss.reset_index(), x="Stock", y="Total_PnL",
                color="Total_PnL",
                color_continuous_scale=["#ef5350", "#ffeb3b", "#26a69a"],
                title="P&L by Stock",
            )
            fig.update_layout(
                template="plotly_dark", height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.9)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Outcome distribution
        if not trades.empty and "Result" in trades.columns:
            st.markdown("### 🎯 Trade Outcome Distribution")
            c1, c2 = st.columns(2)
            with c1:
                rc = trades["Result"].value_counts()
                st.dataframe(rc.rename("Count"), use_container_width=True)
            with c2:
                if "PnL_%" in trades.columns:
                    trades["PnL_%"] = pd.to_numeric(trades["PnL_%"], errors="coerce")
                    ar = trades.groupby("Result")["PnL_%"].mean().round(2)
                    st.dataframe(ar.rename("Avg Return %"), use_container_width=True)

        st.divider()

        # Model Confidence at entry
        if not trades.empty:
            confs = [c for c in ["RF_Conf", "XGB_Conf"] if c in trades.columns]
            if confs:
                st.markdown("### 🔬 Model Confidence at Trade Entry")
                for c in confs:
                    trades[c] = pd.to_numeric(trades[c], errors="coerce")
                cdf = trades[["Date", "Stock"] + confs + ["PnL"]].dropna()
                cdf["Outcome"] = cdf["PnL"].apply(lambda x: "✅ WIN" if x > 0 else "❌ LOSS")
                st.dataframe(cdf, use_container_width=True, hide_index=True)

        st.divider()

        # All trades table
        if not trades.empty:
            st.markdown("### 📋 All Backtest Trades")
            dcols = [
                c
                for c in [
                    "Date", "Stock", "Entry", "Exit", "Qty", "Invested",
                    "PnL", "PnL_%", "Cumulative_PnL", "Result",
                    "Stop_Loss", "Target", "RF_Conf", "XGB_Conf",
                ]
                if c in trades.columns
            ]
            if "Stock" in trades.columns:
                sl = ["All"] + sorted(trades["Stock"].dropna().unique().tolist())
                sel = st.selectbox("Filter by Stock", sl, key="btf")
                if sel != "All":
                    trades = trades[trades["Stock"] == sel]

            st.dataframe(
                trades[dcols] if dcols else trades,
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "⬇️ Download Backtest CSV",
                trades.to_csv(index=False), "aegis_backtest.csv", "text/csv",
            )


# ══════════════════════════════════════════════════
#  TAB 3 — RANKING & SIGNALS
# ══════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🏆 <h3>Stock Ranking & AI Signals</h3></div>', unsafe_allow_html=True)

    ranking_df = load_csv(FILE_RANKING)

    if ranking_df.empty:
        st.warning("⚠️ No ranking data. Run `python src/scholar.py` first.")
    else:
        st.markdown('<div class="section-header">🥇 <h3>Today\'s Stock Ranking</h3></div>', unsafe_allow_html=True)
        st.caption("Scholar scans all stocks, trains 4 models each, ranks by average confidence.")

        # Top picks
        if "rank" in ranking_df.columns:
            top = ranking_df[ranking_df["rank"] <= TOP_N_STOCKS]
            if not top.empty:
                st.markdown(f'<div class="section-header">🎯 <h3>Top {TOP_N_STOCKS} Picks (Will Be Traded)</h3></div>', unsafe_allow_html=True)
                pcols = st.columns(min(len(top), 3))
                for i, (_, row) in enumerate(top.iterrows()):
                    with pcols[i % 3]:
                        sym_clean = str(row.get("symbol", "")).replace(".NS", "")
                        avg_p = row.get("avg_prob", 0)
                        votes = int(row.get("votes", 0))
                        st.metric(
                            f"#{int(row['rank'])} {sym_clean}",
                            f"{avg_p:.1%} confidence",
                            f"{votes}/4 votes",
                        )
                st.divider()

        # Full ranking table
        st.markdown('<div class="section-header">📊 <h3>Full Ranking Table</h3></div>', unsafe_allow_html=True)
        dcols = [
            c
            for c in [
                "rank", "symbol", "rf_prob", "xgb_prob",
                "lstm_prob", "intraday_prob", "avg_prob",
                "votes", "sentiment", "date",
            ]
            if c in ranking_df.columns
        ]
        st.dataframe(
            ranking_df[dcols] if dcols else ranking_df,
            use_container_width=True, hide_index=True,
        )

        # Confidence comparison chart (Plotly grouped bar)
        if "symbol" in ranking_df.columns:
            prob_cols = [
                c
                for c in ["rf_prob", "xgb_prob", "lstm_prob", "intraday_prob"]
                if c in ranking_df.columns
            ]
            if prob_cols:
                st.markdown('<div class="section-header">⚖️ <h3>Model Confidence Comparison</h3></div>', unsafe_allow_html=True)
                import plotly.graph_objects as go

                fig = go.Figure()
                names = ranking_df["symbol"].str.replace(".NS", "")
                colors = ["#42a5f5", "#66bb6a", "#ffa726", "#ab47bc"]
                for i, pc in enumerate(prob_cols):
                    fig.add_trace(go.Bar(
                        name=pc.replace("_prob", "").upper(),
                        x=names,
                        y=ranking_df[pc],
                        marker_color=colors[i % len(colors)],
                    ))
                fig.update_layout(
                    barmode="group", template="plotly_dark", height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14,17,23,0.9)",
                    yaxis_title="Probability",
                    xaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
                    yaxis=dict(gridcolor="rgba(128,128,128,0.08)"),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Votes chart
        if "votes" in ranking_df.columns and "symbol" in ranking_df.columns:
            st.markdown('<div class="section-header">🗳️ <h3>Model Agreement (Votes out of 4)</h3></div>', unsafe_allow_html=True)
            import plotly.express as px

            vdf = ranking_df.copy()
            vdf["name"] = vdf["symbol"].str.replace(".NS", "")
            fig = px.bar(
                vdf, x="name", y="votes",
                color="votes", color_continuous_scale=["#ef5350", "#ffeb3b", "#26a69a"],
                title="Votes per Stock",
            )
            fig.update_layout(
                template="plotly_dark", height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.9)",
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════
#  TAB 4 — MODEL DETAILS & HEALTH
# ══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🤖 <h3>Model Details & Health</h3></div>', unsafe_allow_html=True)

    model_files = get_model_files_info()

    if not model_files:
        st.warning("⚠️ No models trained yet. Run `python src/scholar.py`")
    else:
        st.markdown('<div class="section-header">📁 <h3>Trained Model Files</h3></div>', unsafe_allow_html=True)
        mf_df = pd.DataFrame(model_files)
        st.dataframe(mf_df, use_container_width=True, hide_index=True)

        # Per-stock summary
        st.markdown('<div class="section-header">🏢 <h3>Models Per Stock</h3></div>', unsafe_allow_html=True)
        if "Stock" in mf_df.columns:
            sm = mf_df.groupby("Stock").agg(
                Count=("Type", "count"),
                Types=("Type", lambda x: ", ".join(sorted(set(x)))),
                Total_KB=("Size (KB)", "sum"),
            ).round(1)
            st.dataframe(sm, use_container_width=True)

        total_mb = sum(m["Size (KB)"] for m in model_files) / 1024
        st.metric("Total Model Size", f"{total_mb:.1f} MB")

    st.divider()

    # Features
    st.markdown('<div class="section-header">🔍 <h3>Features Used by AI Models</h3></div>', unsafe_allow_html=True)
    if RF_FEATURES:
        descriptions = [
            "Relative Strength Index (momentum oscillator)",
            "50-day Simple Moving Average (medium trend)",
            "200-day Simple Moving Average (long-term trend)",
            "20-day Exponential Moving Average (short trend)",
            "Average True Range (volatility measure)",
            "MACD Line (momentum direction)",
            "MACD Signal Line (crossover trigger)",
            "Bollinger Band Upper (resistance level)",
            "Bollinger Band Lower (support level)",
            "Volume Ratio vs 20-day avg (volume pressure)",
            "On-Balance Volume (cumulative volume trend)",
            "News Sentiment Score (-1 to +1, NLP-based)",
        ]
        feat_df = pd.DataFrame({
            "Feature": RF_FEATURES,
            "Description": descriptions[: len(RF_FEATURES)],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.divider()

    # Ensemble Weights
    st.markdown('<div class="section-header">⚖️ <h3>Ensemble Model Weights</h3></div>', unsafe_allow_html=True)
    weights = load_json(FILE_WEIGHTS)
    if weights:
        st.caption("Learner optimises trust level for each model per stock:")
        wrows = []
        for stock, w in weights.items():
            if isinstance(w, dict):
                wrows.append({
                    "Stock": stock.replace(".NS", ""),
                    "RF": w.get("rf", 0.25),
                    "XGB": w.get("xgb", 0.25),
                    "LSTM": w.get("lstm", 0.25),
                    "Intraday": w.get("intraday", 0.25),
                })
        if wrows:
            st.dataframe(pd.DataFrame(wrows), use_container_width=True, hide_index=True)
    else:
        st.info("Weights not optimised yet. Run `python src/learner.py`.")

    st.divider()

    # Best Params
    st.markdown('<div class="section-header">⚙️ <h3>Auto-Tuned Hyperparameters</h3></div>', unsafe_allow_html=True)
    best_params = load_json(FILE_PARAMS)
    if best_params:
        if isinstance(best_params, list):
            for entry in best_params:
                sym = entry.get("symbol", "Unknown")
                with st.expander(f"📌 {sym.replace('.NS', '')}"):
                    st.json(entry)
        elif isinstance(best_params, dict):
            for stock, params in best_params.items():
                with st.expander(f"📌 {stock.replace('.NS', '')}"):
                    st.json(params)
    else:
        st.info("Not yet tuned. Run `python src/learner.py`.")


# ══════════════════════════════════════════════════
#  TAB 5 — TRADE HISTORY (Live Sniper)
# ══════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🧾 <h3>Live Trade History (from Sniper)</h3></div>', unsafe_allow_html=True)

    trades_df = load_csv(FILE_TRADES)

    if trades_df.empty:
        st.info("No live trades yet. Sniper runs during market hours.")
        st.code(
            "# Train models first\npython src/scholar.py\n\n"
            "# Start the Sniper during market hours\npython src/sniper.py",
            language="bash",
        )
    else:
        for col in ["Actual_Profit", "Entry_Price", "Exit_Price", "Qty", "AI_Confidence"]:
            if col in trades_df.columns:
                trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")

        if "Actual_Profit" in trades_df.columns:
            total_pnl = trades_df["Actual_Profit"].sum()
            closed = trades_df[trades_df["Action"] != "BUY"] if "Action" in trades_df.columns else trades_df
            total_trades = len(closed)
            wins = len(trades_df[trades_df["Actual_Profit"] > 0])
            losses = len(trades_df[trades_df["Actual_Profit"] < 0])
            wr = (wins / total_trades * 100) if total_trades > 0 else 0

            if total_pnl >= 0:
                st.success(f"### ✅ Total P&L: ₹{total_pnl:,.2f} ({total_pnl / CAPITAL * 100:+.2f}%)")
            else:
                st.error(f"### ❌ Total P&L: ₹{total_pnl:,.2f} ({total_pnl / CAPITAL * 100:+.2f}%)")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Trades", total_trades)
            with c2:
                st.metric("Wins / Losses", f"{wins} / {losses}")
            with c3:
                st.metric("Win Rate", f"{wr:.1f}%")
            with c4:
                avg = total_pnl / total_trades if total_trades > 0 else 0
                st.metric("Avg P&L/Trade", f"₹{avg:,.2f}")

            st.divider()

        # Cumulative P&L chart
        if "Date" in trades_df.columns and "Actual_Profit" in trades_df.columns:
            pnl_by_date = trades_df.groupby("Date")["Actual_Profit"].sum().cumsum()
            if len(pnl_by_date) > 1:
                st.markdown("### 📈 Cumulative P&L Over Time")
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pnl_by_date.index, y=pnl_by_date.values,
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color="#42a5f5", width=2),
                    fillcolor="rgba(66,165,245,0.1)",
                ))
                fig.update_layout(
                    template="plotly_dark", height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14,17,23,0.9)",
                    yaxis_title="Cumulative P&L (₹)",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Per-stock
        if "Stock" in trades_df.columns and "Actual_Profit" in trades_df.columns:
            st.markdown("### 🏢 P&L by Stock")
            stock_pnl = trades_df.groupby("Stock")["Actual_Profit"].agg(["sum", "count", "mean"])
            stock_pnl.columns = ["Total P&L ₹", "Trades", "Avg P&L ₹"]
            stock_pnl = stock_pnl.round(2)
            st.dataframe(stock_pnl, use_container_width=True)

        st.divider()

        # Full table
        st.markdown("### 📋 All Trades")
        if "Stock" in trades_df.columns:
            stocks = ["All"] + sorted(trades_df["Stock"].dropna().unique().tolist())
            selected = st.selectbox("Filter by Stock", stocks, key="live_filter")
            if selected != "All":
                trades_df = trades_df[trades_df["Stock"] == selected]

        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Trade History",
            trades_df.to_csv(index=False), "aegis_trades.csv", "text/csv",
        )


# ══════════════════════════════════════════════════
#  TAB 6 — LEARNER & GUARDIAN REPORT
# ══════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">🧠 <h3>Learner Report — What the AI Learned</h3></div>', unsafe_allow_html=True)

    learner = load_json(FILE_LEARNER)

    if not learner:
        st.info("Learner has not run yet. Run `python src/learner.py` during off-market hours.")
        st.markdown("""
**What the Learner does (meta-learning):**

1. **Trade Review** — Patterns in wins/losses
2. **Hyperparameter Tuning** — Walk-forward optimisation
3. **Regime Detection** — TRENDING / RANGING / VOLATILE
4. **Confidence Calibration** — Ensures probabilities match reality
5. **Ensemble Weight Optimization** — Per-stock model trust
6. **Risk Adaptation** — Adjusts stop-loss / targets
7. **Health Report** — Status for next morning's Sniper
        """)
    else:
        # Health Banner
        health = learner.get("model_health", "UNKNOWN")
        trading = learner.get("trading_allowed", True)
        gen_at = learner.get("generated_at", "N/A")
        if health == "HEALTHY" and trading:
            st.success(f"### ✅ Health: {health} | Trading: ALLOWED | {gen_at}")
        elif health == "DEGRADED":
            st.warning(f"### ⚠️ Health: {health} | Trading: {'ALLOWED' if trading else 'PAUSED'} | {gen_at}")
        else:
            st.error(f"### ❌ Health: {health} | Trading: PAUSED | {gen_at}")

        reasons = learner.get("unhealthy_reasons", [])
        for r in reasons:
            st.markdown(f"- ⚠️ {r}")

        st.divider()

        # Trade Review
        tr = learner.get("trade_review", {})
        st.markdown("### 📊 Trade Review Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Trades Reviewed", tr.get("total_trades", 0))
        with c2:
            st.metric("Win Rate", f"{tr.get('win_rate', 0):.1f}%")
        with c3:
            st.metric("Profit Factor", f"{tr.get('profit_factor', 0):.2f}")
        with c4:
            st.metric("Optimal Conf", f"{tr.get('optimal_confidence', CONFIDENCE_THRESHOLD)}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Avg Win", f"₹{tr.get('avg_win', 0):,.2f}")
        with c2:
            st.metric("Avg Loss", f"₹{tr.get('avg_loss', 0):,.2f}")

        if tr.get("best_stocks"):
            st.markdown(f"**Best Stocks:** {', '.join(tr['best_stocks'])}")
        if tr.get("worst_stocks"):
            st.markdown(f"**Worst Stocks:** {', '.join(tr['worst_stocks'])}")

        st.divider()

        # Risk Parameters
        risk = learner.get("risk_params", {})
        if risk:
            st.markdown("### 🎚️ Risk Parameter Adaptation")
            c1, c2, c3, c4 = st.columns(4)
            rl = risk.get("risk_level", "NORMAL")
            icons = {"EMERGENCY": "🔴", "CONSERVATIVE": "🟡", "NORMAL": "🟢", "CONFIDENT": "💚"}
            with c1:
                st.metric("Risk Level", f"{icons.get(rl, '⚪')} {rl}")
            with c2:
                st.metric("Position Size", f"{risk.get('position_size_multiplier', 1.0):.0%}")
            with c3:
                st.metric("ATR Stop", f"{risk.get('recommended_atr_stop', ATR_STOP_MULTIPLIER)}x")
            with c4:
                st.metric("Max Bullets", risk.get("recommended_max_bullets", MAX_BULLETS))

        st.divider()

        # Market Regimes
        regimes = learner.get("market_regimes", {})
        if regimes:
            st.markdown("### 🌡️ Market Regime Detection")
            rc = {"TRENDING_UP": "🟢", "TRENDING_DOWN": "🔴", "RANGING": "🟡", "HIGH_VOLATILE": "🟠"}
            nifty = regimes.get("NIFTY50", regimes.get("^NSEI", "UNKNOWN"))
            st.markdown(f"**NIFTY 50:** {rc.get(nifty, '⚪')} **{nifty}**")

            rrows = []
            for stk, reg in regimes.items():
                if stk not in ("NIFTY50", "^NSEI"):
                    rrows.append({"Stock": stk.replace(".NS", ""), "Regime": f"{rc.get(reg, '⚪')} {reg}"})
            if rrows:
                st.dataframe(pd.DataFrame(rrows), use_container_width=True, hide_index=True)

        st.divider()

        # Calibrations
        cals = learner.get("calibrations", [])
        if cals:
            st.markdown("### 🎯 Confidence Calibration")
            cal_rows = []
            for cal in cals:
                if isinstance(cal, dict):
                    cal_rows.append({
                        "Stock": str(cal.get("symbol", "")).replace(".NS", ""),
                        "RF OK": "✅" if cal.get("rf_reliable") else "❌",
                        "XGB OK": "✅" if cal.get("xgb_reliable") else "❌",
                        "RF Brier": round(cal.get("rf_brier", 0), 4),
                        "XGB Brier": round(cal.get("xgb_brier", 0), 4),
                    })
            if cal_rows:
                st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Tuned params
        tp = learner.get("tuned_params", [])
        if tp:
            st.markdown("### 🎛️ Tuned Hyperparameters")
            tp_rows = []
            for entry in tp:
                if isinstance(entry, dict):
                    tp_rows.append({
                        "Stock": str(entry.get("symbol", "")).replace(".NS", ""),
                        "RF Prec": entry.get("rf_precision", 0),
                        "XGB Prec": entry.get("xgb_precision", 0),
                    })
            if tp_rows:
                st.dataframe(pd.DataFrame(tp_rows), use_container_width=True, hide_index=True)

        st.divider()

        with st.expander("🔍 Full Learner Report (Raw JSON)"):
            st.json(learner)

    st.divider()

    # Risk Guardian
    st.markdown('<div class="section-header">🛡️ <h3>Risk Guardian — 10 Safety Layers</h3></div>', unsafe_allow_html=True)
    guardian = load_json(FILE_GUARDIAN)

    if guardian:
        if isinstance(guardian, list):
            st.dataframe(pd.DataFrame(guardian), use_container_width=True, hide_index=True)
        elif isinstance(guardian, dict):
            for key, val in guardian.items():
                if isinstance(val, (dict, list)):
                    with st.expander(f"📌 {key}"):
                        st.json(val)
                else:
                    st.markdown(f"**{key}:** {val}")
    else:
        st.info("Risk Guardian log is empty. Records safety decisions during live trading.")
        st.markdown("""
| Layer | Protection |
|-------|------------|
| 1 | Learner Health Check — Is AI healthy? |
| 2 | Market Regime Filter — Block TRENDING_DOWN / HIGH_VOLATILE |
| 3 | Drawdown Circuit Breakers (-2%, -4%, -5%, -8%) |
| 4 | Position Limits — Max 2% risk/trade, 30%/stock |
| 5 | Sector Correlation Filter |
| 6 | Volatility Guard — Reject if ATR > 5% |
| 7 | Adaptive Confidence Gate (0.78 for real money) |
| 8 | Time Filters — Skip first 15 min & last 30 min |
| 9 | Consecutive Loss Lock — Pause after 3 losses |
| 10 | Weekly (-8%) / Monthly (-15%) Loss Caps |
        """)


# ══════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════
st.divider()
st.markdown(
    '<div class="footer">'
    '🛡️ '
    'Project Aegis v4.0 · Live Dashboard · '
    'Plotly Charts · Streamlit Fragments · '
    '100% Local — No data leaves your PC'
    '</div>',
    unsafe_allow_html=True,
)
