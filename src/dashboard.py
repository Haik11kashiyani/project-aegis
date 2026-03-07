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

Run:w
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

# Force sidebar open on every load
if "sidebar_opened" not in st.session_state:
    st.session_state["sidebar_opened"] = True

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
FILE_VOTER_HIST = os.path.join(DATA, "voter_history.json")
FILE_VOTER_ACC  = os.path.join(DATA, "voter_accuracy.json")
FILE_ALERTS     = os.path.join(DATA, "alert_history.json")
FILE_BREADTH    = os.path.join(DATA, "market_breadth.json")
FILE_SECTOR     = os.path.join(DATA, "sector_rotation.json")
FILE_CORR       = os.path.join(DATA, "correlation_matrix.json")
FILE_INTRADAY   = os.path.join(DATA, "intraday_patterns.json")
FILE_DECAY      = os.path.join(DATA, "confidence_decay.json")
FILE_BROKER_LOG = os.path.join(DATA, "broker_orders.json")
REPORTS_DIR     = os.path.join(DATA, "reports")
FILE_KELLY      = os.path.join(DATA, "kelly_state.json")
FILE_EARNINGS   = os.path.join(DATA, "earnings_calendar.json")
FILE_DRIFT      = os.path.join(DATA, "drift_state.json")
FILE_DRIFT_LOG  = os.path.join(DATA, "drift_log.json")
FILE_WF_RESULTS = os.path.join(DATA, "walk_forward_results.json")
FILE_WF_TRADES  = os.path.join(DATA, "walk_forward_trades.csv")
FILE_HEDGE      = os.path.join(DATA, "hedge_state.json")
FILE_MTF        = os.path.join(DATA, "multi_timeframe.json")
FILE_NEWS       = os.path.join(DATA, "news_events.json")
FILE_REBALANCE  = os.path.join(DATA, "rebalance_state.json")
FILE_RP         = os.path.join(DATA, "risk_parity.json")
FILE_AB         = os.path.join(DATA, "ab_backtest.json")
FILE_REGIME     = os.path.join(DATA, "regime_state.json")
FILE_VAR        = os.path.join(DATA, "var_stress.json")
FILE_FINBERT    = os.path.join(DATA, "finbert_sentiment.json")
FILE_EQ         = os.path.join(DATA, "execution_quality.json")
FILE_OC         = os.path.join(DATA, "option_chain.json")
FILE_VP         = os.path.join(DATA, "volume_profile.json")
FILE_JOURNAL    = os.path.join(DATA, "trade_journal.json")
FILE_SMART_ALERT = os.path.join(DATA, "smart_alerts.json")
FILE_RL_SIZER    = os.path.join(DATA, "rl_sizer_state.json")
FILE_RL_QTABLE   = os.path.join(DATA, "rl_qtable.json")
FILE_INTERMARKET = os.path.join(DATA, "intermarket.json")
FILE_LIQUIDITY   = os.path.join(DATA, "liquidity_state.json")
FILE_GREEKS      = os.path.join(DATA, "greeks_heatmap.json")
FILE_BAYESIAN    = os.path.join(DATA, "bayesian_fusion.json")
FILE_BAYES_PRIOR = os.path.join(DATA, "bayesian_priors.json")
FILE_SCALPER     = os.path.join(DATA, "scalper_state.json")
FILE_TUNER       = os.path.join(DATA, "tuner_state.json")
FILE_TRANSFORMER = os.path.join(DATA, "transformer_state.json")
FILE_ORDERBOOK   = os.path.join(DATA, "orderbook_state.json")
FILE_DYN_SL      = os.path.join(DATA, "dynamic_stoploss_state.json")
FILE_PAIRS       = os.path.join(DATA, "pair_trading_state.json")
FILE_ANOMALY     = os.path.join(DATA, "anomaly_state.json")
FILE_VERSIONING  = os.path.join(DATA, "model_versioning_state.json")
FILE_EXECUTOR    = os.path.join(DATA, "adaptive_executor_state.json")
FILE_RL_REBAL    = os.path.join(DATA, "rl_rebalancer_state.json")
FILE_OPT_SYNTH   = os.path.join(DATA, "options_synth_state.json")
FILE_CAUSAL      = os.path.join(DATA, "causal_engine_state.json")
FILE_GA_EVOLVER  = os.path.join(DATA, "ga_evolver_state.json")
FILE_DEBATE      = os.path.join(DATA, "debate_system_state.json")
FILE_RL_AGENT    = os.path.join(DATA, "rl_trade_agent_state.json")
FILE_SMI         = os.path.join(DATA, "sentiment_momentum_state.json")

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
        TRADING_PERSONALITY, SMART_EXIT_ENABLED, SMART_ENTRY_ENABLED,
        GLOBAL_MOOD_ENABLED,
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
    TRADING_PERSONALITY = "MODERATE"
    SMART_EXIT_ENABLED = True
    SMART_ENTRY_ENABLED = True
    GLOBAL_MOOD_ENABLED = True

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

/* ── Sidebar Navigation Radio Styling ── */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    padding: 6px 12px !important; border-radius: 6px; cursor: pointer;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background: #1a1a2e !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
    background: #162040 !important; border-left: 3px solid #3b82f6;
}

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
#  SIDEBAR — NAVIGATION + CONFIG
# ══════════════════════════════════════════════════
_NAV = {
    "📊 Core Trading": [
        "Live Trading",
        "AI Brain & Voters",
        "Performance & Alerts",
    ],
    "📈 Analysis": [
        "Backtest P&L",
        "Ranking & Signals",
        "Model Details",
        "Trade History",
    ],
    "🧠 Intelligence": [
        "Learner & Guardian",
        "Market Intelligence",
        "Broker & Reports",
    ],
    "💼 Portfolio": [
        "Portfolio Treemap",
        "Walk-Forward & Drift",
        "Risk Parity & A/B",
        "News & Rebalancer",
    ],
    "⚠️ Risk & Regime": [
        "Regime & VaR",
        "Volume & Options",
        "Execution & Journal",
    ],
    "🤖 ML & Sizing": [
        "RL Sizer & Bayesian",
        "Intermarket & Liquidity",
        "Greeks Heat Map",
        "Scalper & Tuner",
    ],
    "🔬 Advanced Models": [
        "Transformer & Anomaly",
        "Order Book & Pairs",
        "Dynamic Stops",
        "Model Versioning",
    ],
    "🚀 Phase 12-14": [
        "Adaptive Executor",
        "RL Rebalancer",
        "Options Synthesizer",
        "Causal Engine",
        "GA Strategy Evolver",
        "Multi-Agent Debate",
        "RL Trade Agent",
        "Sentiment Momentum",
    ],
}

with st.sidebar:
    st.markdown('<div class="section-header">🧭 <h3>Navigation</h3></div>', unsafe_allow_html=True)
    _cat = st.selectbox("Category", list(_NAV.keys()), label_visibility="collapsed")
    page = st.radio("Page", _NAV[_cat], label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="section-header">⚙️ <h3>Configuration</h3></div>', unsafe_allow_html=True)
    st.markdown(f"""
- **Watchlist:** {len(STOCK_WATCHLIST)} stocks
- **Top-N:** {TOP_N_STOCKS}
- **Capital:** ₹{CAPITAL:,}
- **Bullet Size:** ₹{CAPITAL // MAX_BULLETS}
- **Models:** RF + XGB + LSTM + Intraday
- **NeuroVoters:** 6 AI voters
- **Voting:** Weighted consensus
- **Confidence:** {CONFIDENCE_THRESHOLD}
- **Stop Loss:** {ATR_STOP_MULTIPLIER}× ATR
- **Target:** {ATR_TARGET_MULTIPLIER}× ATR
- **Daily Target:** {DAILY_TARGET * 100:.0f}%
- **Kill Switch:** -{MAX_DAILY_LOSS_PCT * 100:.0f}%
    """)

    st.divider()
    st.markdown('<div class="section-header">🧠 <h3>AI Brain Config</h3></div>', unsafe_allow_html=True)
    _pers_icon = {"AGGRESSIVE": "🔴", "MODERATE": "🟡", "CONSERVATIVE": "🟢"}.get(TRADING_PERSONALITY, "⚪")
    st.markdown(f"""
- **Personality:** {_pers_icon} {TRADING_PERSONALITY}
- **Smart Exit:** {"✅ ON" if SMART_EXIT_ENABLED else "❌ OFF"}
- **Smart Entry:** {"✅ ON" if SMART_ENTRY_ENABLED else "❌ OFF"}
- **Global Mood:** {"✅ ON" if GLOBAL_MOOD_ENABLED else "❌ OFF"}
- **Anti-Overfit:** ✅ Active
- **Regime Detection:** ✅ Active

**6 NeuroVoters:**
1. 🏃 Momentum Mike (1.2×)
2. 💎 Value Vinay (0.9×)
3. 🛡️ Risk Raj (1.5× VETO)
4. 📰 Sentiment Sanjay (1.0× VETO)
5. 📊 Pattern Priya (1.0×)
6. 🤖 Quant Qasim (1.3×)
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
    <div class="hero-sub">6 NeuroVoter AI Ensemble · Smart Entry/Exit Brain · Anti-Overfit · Regime-Aware · Real-Time Charts · 100% Local</div>
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
#  PAGE ROUTING (sidebar-driven)
# ══════════════════════════════════════════════════


# ══════════════════════════════════════════════════
#  TAB 1 — LIVE TRADING & CHARTS
# ══════════════════════════════════════════════════
if page == "Live Trading":

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

    # ── MINI EQUITY CURVE (today's running P&L) ──
    st.markdown('<div class="section-header">💰 <h3>Portfolio Equity Curve</h3></div>', unsafe_allow_html=True)
    _th_eq = load_csv(FILE_TRADES)
    if not _th_eq.empty and "Date" in _th_eq.columns and "Profit" in _th_eq.columns:
        try:
            _eq = _th_eq.copy()
            _eq["Date"] = pd.to_datetime(_eq["Date"], errors="coerce")
            _eq = _eq.dropna(subset=["Date"])
            _eq["Profit"] = pd.to_numeric(_eq["Profit"], errors="coerce").fillna(0)
            _eq = _eq.sort_values("Date")
            _eq["Cumulative"] = _eq["Profit"].cumsum()
            _eq["Balance"] = 1000 + _eq["Cumulative"]
            _daily_eq = _eq.groupby(_eq["Date"].dt.date).agg({"Profit": "sum", "Balance": "last"}).reset_index()
            _daily_eq.columns = ["Date", "Daily_PnL", "Balance"]

            _total_pnl = _eq["Cumulative"].iloc[-1] if len(_eq) > 0 else 0
            _pnl_color = "#00ff88" if _total_pnl >= 0 else "#ff4444"
            _total_trades = len(_eq)
            _wins = (_eq["Profit"] > 0).sum()
            _wr = (_wins / _total_trades * 100) if _total_trades > 0 else 0

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(f'<div style="background:#0d1117;padding:10px;border-radius:8px;text-align:center;border:1px solid {_pnl_color}">'
                         f'<span style="color:#888;font-size:11px">Total P&L</span><br>'
                         f'<span style="color:{_pnl_color};font-size:20px;font-weight:bold">₹{_total_pnl:+.2f}</span></div>', unsafe_allow_html=True)
            mc2.markdown(f'<div style="background:#0d1117;padding:10px;border-radius:8px;text-align:center;border:1px solid #29b6f6">'
                         f'<span style="color:#888;font-size:11px">Total Trades</span><br>'
                         f'<span style="color:#29b6f6;font-size:20px;font-weight:bold">{_total_trades}</span></div>', unsafe_allow_html=True)
            mc3.markdown(f'<div style="background:#0d1117;padding:10px;border-radius:8px;text-align:center;border:1px solid #ffd700">'
                         f'<span style="color:#888;font-size:11px">Win Rate</span><br>'
                         f'<span style="color:#ffd700;font-size:20px;font-weight:bold">{_wr:.0f}%</span></div>', unsafe_allow_html=True)
            mc4.markdown(f'<div style="background:#0d1117;padding:10px;border-radius:8px;text-align:center;border:1px solid #ab47bc">'
                         f'<span style="color:#888;font-size:11px">Current Balance</span><br>'
                         f'<span style="color:#ab47bc;font-size:20px;font-weight:bold">₹{_eq["Balance"].iloc[-1]:.2f}</span></div>', unsafe_allow_html=True)

            import plotly.graph_objects as go
            _fig_mini = go.Figure()
            _fig_mini.add_trace(go.Scatter(
                x=_daily_eq["Date"], y=_daily_eq["Balance"],
                mode="lines+markers", line=dict(color="#00ff88", width=2),
                marker=dict(size=5, color=["#00ff88" if p >= 0 else "#ff4444" for p in _daily_eq["Daily_PnL"]]),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.05)",
                hovertemplate="Date: %{x}<br>Balance: ₹%{y:.2f}<extra></extra>"
            ))
            _fig_mini.add_hline(y=1000, line_dash="dash", line_color="rgba(255,255,255,0.2)")
            _fig_mini.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=10, r=10, t=5, b=10),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                font=dict(color="#e0e0e0", size=10)
            )
            st.plotly_chart(_fig_mini, use_container_width=True)
        except Exception as _e:
            st.caption(f"Equity curve: {_e}")
    else:
        st.caption("📭 Equity curve will appear after trades are executed.")

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

        # ── Trading Brain & Global Mood Bar ──
        _personality = _analysis.get("personality", "N/A")
        _smart_exit = "ON" if _analysis.get("smart_exit", False) else "OFF"
        _smart_entry = "ON" if _analysis.get("smart_entry", False) else "OFF"
        _gmood = _analysis.get("global_mood", {})
        _mood_score = _gmood.get("overall_mood", 0) if _gmood else 0
        _mood_label = "🟢 RISK-ON" if _mood_score > 0.2 else ("🔴 RISK-OFF" if _mood_score < -0.2 else "🟡 NEUTRAL")
        _vix = _gmood.get("india_vix", "N/A") if _gmood else "N/A"
        _sp500 = _gmood.get("sp500_change", "N/A") if _gmood else "N/A"

        st.markdown(f"""
        <div style="border: 1px solid #27272a; border-radius: 6px; padding: 10px 16px; margin-bottom: 12px; background: linear-gradient(135deg, #09090b 0%, #18181b 100%); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <span style="font-size: 0.78rem; color: #a1a1aa;">🧠 <b style="color: #c084fc;">{_personality}</b> Brain</span>
            <span style="font-size: 0.75rem; color: #a1a1aa;">🗳️ <b style="color: #c084fc;">6 NeuroVoters</b></span>
            <span style="font-size: 0.75rem; color: #a1a1aa;">Smart Exit: <b style="color: {'#10b981' if _smart_exit=='ON' else '#ef4444'};">{_smart_exit}</b></span>
            <span style="font-size: 0.75rem; color: #a1a1aa;">Smart Entry: <b style="color: {'#10b981' if _smart_entry=='ON' else '#ef4444'};">{_smart_entry}</b></span>
            <span style="font-size: 0.75rem; color: #a1a1aa;">Global Mood: <b>{_mood_label}</b> ({_mood_score:+.2f})</span>
            <span style="font-size: 0.75rem; color: #a1a1aa;">VIX: <b style="color: #f59e0b;">{_vix}</b></span>
        </div>
        """, unsafe_allow_html=True)

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

                # ── NeuroVoter Panel: Show 6 human-like AI voters ──
                neuro = s.get("neuro_decision", {})
                if neuro:
                    n_score = neuro.get("score", 0)
                    n_regime = neuro.get("regime", "UNKNOWN")
                    n_vetoed = neuro.get("vetoed", False)
                    n_buy = neuro.get("buy_voters", 0)
                    n_sell = neuro.get("sell_voters", 0)
                    n_hold = neuro.get("hold_voters", 0)
                    n_conviction = neuro.get("conviction", 0)
                    n_reasoning = neuro.get("reasoning", "")

                    regime_colors = {
                        "TRENDING_UP": "#10b981", "TRENDING_DOWN": "#ef4444",
                        "HIGH_VOLATILE": "#f59e0b", "LOW_VOLATILE": "#3b82f6",
                        "SIDEWAYS": "#a1a1aa",
                    }
                    rc = regime_colors.get(n_regime, "#a1a1aa")

                    score_color = "#10b981" if n_score > 0.15 else ("#ef4444" if n_score < -0.15 else "#f59e0b")

                    voters_html = f"""
                    <div style="margin-top: 10px; border: 1px solid #27272a; border-radius: 6px; padding: 10px 12px; background: #0a0a0a;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 0.78rem; color: #c084fc; font-weight: 600;">🗳️ NeuroVoter Panel</span>
                            <span style="font-size: 0.7rem; background: #18181b; color: {rc}; padding: 2px 8px; border-radius: 3px; border: 1px solid #27272a;">{n_regime}</span>
                        </div>
                        <div style="display: flex; gap: 12px; margin-bottom: 8px; flex-wrap: wrap;">
                            <span style="font-size: 0.72rem; color: #a1a1aa;">Score: <b style="color: {score_color};">{n_score:+.3f}</b></span>
                            <span style="font-size: 0.72rem; color: #10b981;">Buy: <b>{n_buy}</b></span>
                            <span style="font-size: 0.72rem; color: #ef4444;">Sell: <b>{n_sell}</b></span>
                            <span style="font-size: 0.72rem; color: #a1a1aa;">Hold: <b>{n_hold}</b></span>
                            <span style="font-size: 0.72rem; color: #a1a1aa;">Conviction: <b>{n_conviction:.0%}</b></span>
                        </div>
                    """

                    if n_vetoed:
                        veto_reasons = neuro.get("veto_reasons", [])
                        voters_html += f"""
                        <div style="padding: 4px 8px; background: #450a0a; border: 1px solid #7f1d1d; border-radius: 4px; margin-bottom: 8px;">
                            <span style="font-size: 0.7rem; color: #f87171;">🚫 VETOED: {', '.join(veto_reasons[:2])}</span>
                        </div>
                        """

                    # Individual voter bars
                    voters_list = neuro.get("voters", [])
                    for v in voters_list:
                        v_name = v.get("name", "?")
                        v_vote = v.get("vote", 0)
                        v_conv = v.get("conviction", 0)
                        v_reason = v.get("reasoning", "")
                        v_veto = v.get("veto", False)

                        if v_vote > 0.1:
                            v_color = "#10b981"
                            v_icon = "🟢"
                        elif v_vote < -0.1:
                            v_color = "#ef4444"
                            v_icon = "🔴"
                        else:
                            v_color = "#a1a1aa"
                            v_icon = "⚪"

                        # Vote bar: centered at 50%, extends left for sell, right for buy
                        bar_pct = min(abs(v_vote) * 50, 50)
                        bar_dir = "right" if v_vote >= 0 else "left"
                        bar_style = f"width: {bar_pct:.0f}%; height: 100%; background: {v_color}; border-radius: 2px; float: {'left' if v_vote >= 0 else 'right'}; margin-{'left' if v_vote >= 0 else 'right'}: 50%;"

                        veto_badge = '<span style="font-size: 0.6rem; color: #f87171; margin-left: 4px;">VETO</span>' if v_veto else ""

                        voters_html += f"""
                        <div style="display: flex; align-items: center; margin-bottom: 4px;" title="{v_reason}">
                            <div style="width: 130px; font-size: 0.68rem; color: #a1a1aa;">{v_icon} {v_name}{veto_badge}</div>
                            <div style="flex: 1; background: #18181b; border-radius: 2px; height: 10px; margin: 0 6px; border: 1px solid #27272a; position: relative; overflow: hidden;">
                                <div style="{bar_style}"></div>
                            </div>
                            <div style="width: 65px; font-size: 0.66rem; color: {v_color}; text-align: right;">{v_vote:+.2f} ({v_conv:.0%})</div>
                        </div>
                        """

                    # Reasoning line
                    if n_reasoning:
                        voters_html += f"""
                        <div style="margin-top: 6px; font-size: 0.68rem; color: #71717a; font-style: italic;">↳ {n_reasoning[:120]}</div>
                        """

                    voters_html += "</div>"
                    st.markdown(voters_html, unsafe_allow_html=True)

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

                    # Smart Exit Type badge
                    _exit_type = str(crow.get('Exit_Type', crow.get('exit_type', '')))
                    _exit_type_colors = {
                        'TARGET_HIT': ('#10b981', '#052e16', '#064e3b'),
                        'STOP_LOSS': ('#ef4444', '#450a0a', '#7f1d1d'),
                        'MOMENTUM_EXIT': ('#f59e0b', '#451a03', '#92400e'),
                        'RSI_EXIT': ('#8b5cf6', '#2e1065', '#4c1d95'),
                        'VOLUME_EXIT': ('#06b6d4', '#083344', '#155e75'),
                        'TIME_DECAY': ('#a1a1aa', '#27272a', '#3f3f46'),
                        'SENTIMENT_EXIT': ('#ec4899', '#500724', '#831843'),
                        'MAX_HOLD': ('#84cc16', '#1a2e05', '#365314'),
                    }
                    _et_colors = _exit_type_colors.get(_exit_type, ('#a1a1aa', '#27272a', '#3f3f46'))
                    _exit_badge_html = f'<span style="font-size: 0.65rem; background: {_et_colors[1]}; color: {_et_colors[0]}; padding: 2px 7px; border-radius: 3px; border: 1px solid {_et_colors[2]}; margin-left: 4px;">{_exit_type}</span>' if _exit_type and _exit_type not in ('', 'nan', 'None') else ''

                    st.markdown(f"""
                    <div style="border: 1px solid #27272a; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; background: #0c0c10; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <span style="font-size: 1rem; font-weight: 600; color: #fff;">{_sym_name}</span>
                        <span style="font-size: 0.8rem; color: #a1a1aa;">{_action}{_exit_badge_html}</span>
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
#  TAB 2 — AI BRAIN & VOTERS
# ══════════════════════════════════════════════════
if page == "AI Brain & Voters":
    st.markdown('<div class="section-header">🧠 <h3>AI Brain & NeuroVoter System — Full Architecture</h3></div>', unsafe_allow_html=True)

    # ── System Architecture Overview ──
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 100%); border: 1px solid #27272a; border-radius: 10px; padding: 20px 24px; margin-bottom: 16px;">
        <div style="text-align: center; font-size: 1.1rem; color: #c084fc; font-weight: 700; margin-bottom: 14px;">🏗️ System Architecture — How Decisions Are Made</div>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; align-items: center; padding: 0 12px;">
            <div style="background: #172554; padding: 8px 14px; border-radius: 6px; border: 1px solid #1d4ed8; text-align: center;">
                <div style="font-size: 0.7rem; color: #93c5fd;">Model 1</div>
                <div style="font-size: 0.8rem; color: #60a5fa; font-weight: 600;">🌲 Random Forest</div>
            </div>
            <div style="background: #172554; padding: 8px 14px; border-radius: 6px; border: 1px solid #1d4ed8; text-align: center;">
                <div style="font-size: 0.7rem; color: #93c5fd;">Model 2</div>
                <div style="font-size: 0.8rem; color: #60a5fa; font-weight: 600;">⚡ XGBoost</div>
            </div>
            <div style="background: #172554; padding: 8px 14px; border-radius: 6px; border: 1px solid #1d4ed8; text-align: center;">
                <div style="font-size: 0.7rem; color: #93c5fd;">Model 3</div>
                <div style="font-size: 0.8rem; color: #60a5fa; font-weight: 600;">🔮 Daily LSTM</div>
            </div>
            <div style="background: #172554; padding: 8px 14px; border-radius: 6px; border: 1px solid #1d4ed8; text-align: center;">
                <div style="font-size: 0.7rem; color: #93c5fd;">Model 4</div>
                <div style="font-size: 0.8rem; color: #60a5fa; font-weight: 600;">📈 Intraday LSTM</div>
            </div>

            <div style="color: #c084fc; font-size: 1.4rem; padding: 0 6px;">→</div>

            <div style="background: #1a0a2e; padding: 10px 16px; border-radius: 8px; border: 2px solid #7c3aed; text-align: center;">
                <div style="font-size: 0.7rem; color: #c4b5fd;">Ensemble</div>
                <div style="font-size: 0.85rem; color: #a78bfa; font-weight: 700;">🗳️ 6 NeuroVoters</div>
            </div>

            <div style="color: #c084fc; font-size: 1.4rem; padding: 0 6px;">→</div>

            <div style="background: #052e16; padding: 10px 16px; border-radius: 8px; border: 2px solid #16a34a; text-align: center;">
                <div style="font-size: 0.7rem; color: #86efac;">Decision</div>
                <div style="font-size: 0.85rem; color: #4ade80; font-weight: 700;">✅ BUY / ❌ SKIP</div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 12px; color: #71717a; font-size: 0.72rem;">
            4 ML models generate confidence scores → 6 human-like AI voters analyse with unique strategies → Weighted consensus with veto power → Final decision
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 6 NeuroVoter Profiles ──
    st.markdown("#### 🗳️ The 6 NeuroVoters")
    voter_profiles = [
        {
            "icon": "📈", "name": "Momentum Trader", "weight": 0.20,
            "specialty": "Trend strength, RSI momentum, MACD crossovers",
            "style": "Aggressive — rides strong trends, quick to act on momentum shifts",
            "veto": False, "color": "#10b981",
        },
        {
            "icon": "💎", "name": "Value Hunter", "weight": 0.15,
            "specialty": "Price vs SMA-200, Bollinger Band position, mean reversion",
            "style": "Patient — waits for deep value opportunities, contrarian at extremes",
            "veto": False, "color": "#3b82f6",
        },
        {
            "icon": "🛡️", "name": "Risk Manager", "weight": 0.20,
            "specialty": "ATR volatility, volume anomalies, drawdown protection",
            "style": "Cautious — primary safety gate, vetoes dangerous trades",
            "veto": True, "color": "#ef4444",
        },
        {
            "icon": "📰", "name": "Sentiment Analyst", "weight": 0.15,
            "specialty": "News sentiment, sector mood, global market conditions",
            "style": "Contextual — reads the room before acting, vetoes on bad news",
            "veto": True, "color": "#f59e0b",
        },
        {
            "icon": "🔬", "name": "Pattern Recognizer", "weight": 0.15,
            "specialty": "LSTM confidence analysis, model agreement patterns, technical setups",
            "style": "Analytical — compares models and looks for consensus patterns",
            "veto": False, "color": "#8b5cf6",
        },
        {
            "icon": "🤖", "name": "Quant Brain", "weight": 0.15,
            "specialty": "Statistical edge, expected value calculation, Kelly criterion sizing",
            "style": "Mathematical — only acts when the numbers provide a clear advantage",
            "veto": False, "color": "#06b6d4",
        },
    ]

    v_cols = st.columns(3)
    for i, vp in enumerate(voter_profiles):
        with v_cols[i % 3]:
            veto_badge = '<span style="background: #7f1d1d; color: #f87171; font-size: 0.65rem; padding: 2px 6px; border-radius: 3px; margin-left: 6px;">🚫 VETO POWER</span>' if vp["veto"] else ""
            st.markdown(f"""
            <div style="background: #09090b; border: 1px solid #27272a; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; border-left: 3px solid {vp["color"]};">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">{vp["icon"]}</span>
                    <span style="font-size: 0.88rem; color: #e4e4e7; font-weight: 700;">{vp["name"]}</span>
                    {veto_badge}
                </div>
                <div style="font-size: 0.72rem; color: #a1a1aa; margin-bottom: 6px;">
                    <b style="color: #c084fc;">Weight:</b> {vp["weight"]:.0%} &nbsp;│&nbsp;
                    <b style="color: #c084fc;">Specialty:</b> {vp["specialty"]}
                </div>
                <div style="font-size: 0.7rem; color: #71717a; font-style: italic;">{vp["style"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Trading Brain Personality ──
    st.markdown("#### 🎭 Trading Brain Personality")
    _personality = TRADING_PERSONALITY.upper() if hasattr(TRADING_PERSONALITY, "upper") else "MODERATE"
    pers_map = {
        "AGGRESSIVE": {"icon": "🔴", "color": "#ef4444", "desc": "More trades, wider stops, lower confidence bar. Accepts higher risk for more opportunities."},
        "MODERATE": {"icon": "🟡", "color": "#f59e0b", "desc": "Balanced approach — recommended for most users. Good risk/reward tradeoff."},
        "CONSERVATIVE": {"icon": "🟢", "color": "#10b981", "desc": "Fewer trades, tighter stops, higher confidence needed. Maximizes safety."},
    }
    pp = pers_map.get(_personality, pers_map["MODERATE"])

    pc1, pc2 = st.columns([1, 2])
    with pc1:
        st.markdown(f"""
        <div style="background: #09090b; border: 2px solid {pp["color"]}; border-radius: 10px; padding: 24px; text-align: center;">
            <div style="font-size: 3rem;">{pp["icon"]}</div>
            <div style="font-size: 1.2rem; color: {pp["color"]}; font-weight: 800; margin-top: 8px;">{_personality}</div>
        </div>
        """, unsafe_allow_html=True)
    with pc2:
        st.markdown(f"""
        <div style="background: #09090b; border: 1px solid #27272a; border-radius: 8px; padding: 16px 20px;">
            <div style="font-size: 0.85rem; color: #e4e4e7; margin-bottom: 12px;">{pp["desc"]}</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div style="font-size: 0.75rem; color: #a1a1aa;">✅ Smart Exit: <b style="color: #10b981;">{"ENABLED" if SMART_EXIT_ENABLED else "DISABLED"}</b></div>
                <div style="font-size: 0.75rem; color: #a1a1aa;">✅ Smart Entry: <b style="color: #10b981;">{"ENABLED" if SMART_ENTRY_ENABLED else "DISABLED"}</b></div>
                <div style="font-size: 0.75rem; color: #a1a1aa;">🌐 Global Mood: <b style="color: #10b981;">{"ENABLED" if GLOBAL_MOOD_ENABLED else "DISABLED"}</b></div>
                <div style="font-size: 0.75rem; color: #a1a1aa;">🛡️ Risk Guardian: <b style="color: #10b981;">ENABLED</b></div>
                <div style="font-size: 0.75rem; color: #a1a1aa;">📊 Min Votes: <b style="color: #60a5fa;">{MIN_VOTES_TO_BUY}/4</b></div>
                <div style="font-size: 0.75rem; color: #a1a1aa;">🎯 Confidence: <b style="color: #60a5fa;">{CONFIDENCE_THRESHOLD:.0%}</b></div>
                <div style="font-size: 0.75rem; color: #a1a1aa;">🔴 Stop Loss: <b style="color: #ef4444;">{ATR_STOP_MULTIPLIER}× ATR</b></div>
                <div style="font-size: 0.75rem; color: #a1a1aa;">🟢 Target: <b style="color: #10b981;">{ATR_TARGET_MULTIPLIER}× ATR</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Smart Exit Types ──
    st.markdown("#### 🚪 Smart Exit System")
    st.markdown("""
    <div style="background: #09090b; border: 1px solid #27272a; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
        <div style="font-size: 0.8rem; color: #a1a1aa; margin-bottom: 12px;">
            The Trading Brain uses <b style="color: #c084fc;">8 intelligent exit triggers</b> instead of just stop-loss/target. Each exit type adapts to market conditions:
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #10b981; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;"></span>
                <span style="font-size: 0.75rem; color: #e4e4e7;"><b>TARGET_HIT</b> — Price reached profit target</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #ef4444; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;"></span>
                <span style="font-size: 0.75rem; color: #e4e4e7;"><b>STOP_LOSS</b> — Price hit stop-loss level</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #f59e0b; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;"></span>
                <span style="font-size: 0.75rem; color: #e4e4e7;"><b>MOMENTUM_EXIT</b> — Momentum reversed (RSI, MACD)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #8b5cf6; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;"></span>
                <span style="font-size: 0.75rem; color: #e4e4e7;"><b>RSI_EXIT</b> — RSI entered overbought zone (&gt;70)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #06b6d4; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;"></span>
                <span style="font-size: 0.75rem; color: #e4e4e7;"><b>VOLUME_EXIT</b> — Volume dried up (distribution)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #a1a1aa; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;"></span>
                <span style="font-size: 0.75rem; color: #e4e4e7;"><b>TIME_DECAY</b> — Held too long without progress</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #ec4899; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;"></span>
                <span style="font-size: 0.75rem; color: #e4e4e7;"><b>SENTIMENT_EXIT</b> — News turned negative</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #84cc16; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;"></span>
                <span style="font-size: 0.75rem; color: #e4e4e7;"><b>MAX_HOLD</b> — Market closing, force exit</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Anti-Overfit Engine ──
    st.markdown("#### 🛡️ Anti-Overfit Engine")
    st.markdown("""
    <div style="background: #09090b; border: 1px solid #27272a; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
        <div style="font-size: 0.8rem; color: #a1a1aa; margin-bottom: 12px;">
            The <b style="color: #c084fc;">OverfitDetector</b> runs 5 real-time checks to ensure AI models aren't memorizing noise:
        </div>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #27272a;">
                <td style="padding: 8px 10px; font-size: 0.75rem; color: #f59e0b; font-weight: 600;">Check</td>
                <td style="padding: 8px 10px; font-size: 0.75rem; color: #a1a1aa;">What It Detects</td>
                <td style="padding: 8px 10px; font-size: 0.75rem; color: #a1a1aa;">Action</td>
            </tr>
            <tr style="border-bottom: 1px solid #1a1a1a;">
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #e4e4e7;">🎯 Extreme Confidence</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #a1a1aa;">Model outputting &gt;95% or &lt;5% confidence consistently</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #ef4444;">Compress to neutral</td>
            </tr>
            <tr style="border-bottom: 1px solid #1a1a1a;">
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #e4e4e7;">🔁 Stuck Predictions</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #a1a1aa;">Same direction 10+ times in a row</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #ef4444;">Shift toward opposite</td>
            </tr>
            <tr style="border-bottom: 1px solid #1a1a1a;">
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #e4e4e7;">🔄 Flip-Flop Detection</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #a1a1aa;">Confidence swinging wildly between calls</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #ef4444;">Dampen to average</td>
            </tr>
            <tr style="border-bottom: 1px solid #1a1a1a;">
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #e4e4e7;">📉 Reality Mismatch</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #a1a1aa;">Predicting BUY but stock is actually falling</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #ef4444;">Reduce confidence</td>
            </tr>
            <tr>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #e4e4e7;">🌊 Volatility Mismatch</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #a1a1aa;">High confidence in highly volatile market</td>
                <td style="padding: 8px 10px; font-size: 0.73rem; color: #ef4444;">Shrink toward 0.5</td>
            </tr>
        </table>
        <div style="margin-top: 10px; font-size: 0.72rem; color: #71717a; text-align: center;">
            + <b>Ensemble-level overfit detection</b>: If all 4 models agree with &gt;90% confidence, the system applies skepticism dampening
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Regime Detector ──
    st.markdown("#### 🌊 Market Regime Detector")
    st.markdown("""
    <div style="background: #09090b; border: 1px solid #27272a; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
        <div style="font-size: 0.8rem; color: #a1a1aa; margin-bottom: 12px;">
            Automatically detects the current market regime using SMA trends, ATR volatility, and price action. Each voter adapts behaviour based on the detected regime.
        </div>
        <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;">
            <div style="background: #052e16; border: 1px solid #16a34a; border-radius: 6px; padding: 10px 16px; text-align: center; min-width: 120px;">
                <div style="font-size: 1.2rem;">📈</div>
                <div style="font-size: 0.78rem; color: #4ade80; font-weight: 600;">TRENDING UP</div>
                <div style="font-size: 0.65rem; color: #86efac;">SMA20 &gt; SMA50</div>
            </div>
            <div style="background: #450a0a; border: 1px solid #b91c1c; border-radius: 6px; padding: 10px 16px; text-align: center; min-width: 120px;">
                <div style="font-size: 1.2rem;">📉</div>
                <div style="font-size: 0.78rem; color: #f87171; font-weight: 600;">TRENDING DOWN</div>
                <div style="font-size: 0.65rem; color: #fca5a5;">SMA20 &lt; SMA50</div>
            </div>
            <div style="background: #451a03; border: 1px solid #c2410c; border-radius: 6px; padding: 10px 16px; text-align: center; min-width: 120px;">
                <div style="font-size: 1.2rem;">🌊</div>
                <div style="font-size: 0.78rem; color: #fb923c; font-weight: 600;">HIGH VOLATILE</div>
                <div style="font-size: 0.65rem; color: #fdba74;">ATR &gt; 1.5× avg</div>
            </div>
            <div style="background: #172554; border: 1px solid #1d4ed8; border-radius: 6px; padding: 10px 16px; text-align: center; min-width: 120px;">
                <div style="font-size: 1.2rem;">🧊</div>
                <div style="font-size: 0.78rem; color: #60a5fa; font-weight: 600;">LOW VOLATILE</div>
                <div style="font-size: 0.65rem; color: #93c5fd;">ATR &lt; 0.5× avg</div>
            </div>
            <div style="background: #18181b; border: 1px solid #3f3f46; border-radius: 6px; padding: 10px 16px; text-align: center; min-width: 120px;">
                <div style="font-size: 1.2rem;">➡️</div>
                <div style="font-size: 0.78rem; color: #a1a1aa; font-weight: 600;">SIDEWAYS</div>
                <div style="font-size: 0.65rem; color: #d4d4d8;">No clear trend</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Live Cross-Stock NeuroVoter Consensus ──
    st.markdown("#### 📊 Live Cross-Stock NeuroVoter Consensus")
    _analysis = load_json(FILE_ANALYSIS)
    _stocks_data = _analysis.get("stocks", []) if _analysis else []

    has_neuro_data = any(s.get("neuro_decision") for s in _stocks_data)

    if has_neuro_data:
        # Build consensus table
        consensus_rows = []
        for s in _stocks_data:
            nd = s.get("neuro_decision", {})
            if not nd:
                continue
            consensus_rows.append({
                "Stock": s.get("name", s.get("symbol", "?")),
                "NeuroScore": nd.get("score", 0),
                "Conviction": nd.get("conviction", 0),
                "Buy Voters": nd.get("buy_voters", 0),
                "Sell Voters": nd.get("sell_voters", 0),
                "Hold Voters": nd.get("hold_voters", 0),
                "Regime": nd.get("regime", "?"),
                "Vetoed": "🚫 YES" if nd.get("vetoed", False) else "✅ No",
                "Decision": "BUY" if nd.get("should_buy", False) else "SKIP",
            })

        if consensus_rows:
            cdf = pd.DataFrame(consensus_rows)

            # Heatmap-style display
            for _, row in cdf.iterrows():
                sc = row["NeuroScore"]
                sc_color = "#10b981" if sc > 0.15 else ("#ef4444" if sc < -0.15 else "#f59e0b")
                dec_color = "#10b981" if row["Decision"] == "BUY" else "#ef4444"
                regime_colors = {
                    "TRENDING_UP": "#10b981", "TRENDING_DOWN": "#ef4444",
                    "HIGH_VOLATILE": "#f59e0b", "LOW_VOLATILE": "#3b82f6",
                    "SIDEWAYS": "#a1a1aa",
                }
                rc = regime_colors.get(row["Regime"], "#a1a1aa")

                st.markdown(f"""
                <div style="display: flex; align-items: center; background: #09090b; border: 1px solid #27272a; border-radius: 6px; padding: 10px 14px; margin-bottom: 6px; gap: 16px; flex-wrap: wrap;">
                    <div style="min-width: 90px; font-size: 0.88rem; color: #e4e4e7; font-weight: 700;">{row["Stock"]}</div>
                    <div style="font-size: 0.75rem; color: {sc_color}; min-width: 100px;">Score: <b>{sc:+.3f}</b></div>
                    <div style="font-size: 0.75rem; color: #a1a1aa; min-width: 90px;">Conv: <b>{row["Conviction"]:.0%}</b></div>
                    <div style="font-size: 0.75rem; min-width: 130px;">
                        <span style="color: #10b981;">🟢{row["Buy Voters"]}</span> &nbsp;
                        <span style="color: #ef4444;">🔴{row["Sell Voters"]}</span> &nbsp;
                        <span style="color: #a1a1aa;">⚪{row["Hold Voters"]}</span>
                    </div>
                    <div style="font-size: 0.72rem; color: {rc}; background: #18181b; padding: 2px 8px; border-radius: 3px; border: 1px solid #27272a;">{row["Regime"]}</div>
                    <div style="font-size: 0.72rem; color: #a1a1aa;">{row["Vetoed"]}</div>
                    <div style="font-size: 0.78rem; color: {dec_color}; font-weight: 700; margin-left: auto;">{row["Decision"]}</div>
                </div>
                """, unsafe_allow_html=True)

            # Aggregate summary
            total_buy = sum(1 for r in consensus_rows if r["Decision"] == "BUY")
            total_skip = sum(1 for r in consensus_rows if r["Decision"] == "SKIP")
            avg_score = np.mean([r["NeuroScore"] for r in consensus_rows])
            avg_conv = np.mean([r["Conviction"] for r in consensus_rows])

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 100%); border: 1px solid #7c3aed; border-radius: 8px; padding: 14px 20px; margin-top: 12px; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 12px; text-align: center;">
                <div>
                    <div style="font-size: 0.7rem; color: #a1a1aa;">Avg NeuroScore</div>
                    <div style="font-size: 1.1rem; color: {"#10b981" if avg_score > 0 else "#ef4444"}; font-weight: 700;">{avg_score:+.3f}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: #a1a1aa;">Avg Conviction</div>
                    <div style="font-size: 1.1rem; color: #c084fc; font-weight: 700;">{avg_conv:.0%}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: #a1a1aa;">BUY Signals</div>
                    <div style="font-size: 1.1rem; color: #10b981; font-weight: 700;">{total_buy}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: #a1a1aa;">SKIP Signals</div>
                    <div style="font-size: 1.1rem; color: #ef4444; font-weight: 700;">{total_skip}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ NeuroVoter data not yet available in analysis. Run the sniper to generate decisions.")
    else:
        st.info("ℹ️ No NeuroVoter decisions yet. The sniper will populate this after its first scan with the new AI system. Run: `python src/sniper.py`")

    st.divider()

    # ── Live Per-Voter Accuracy (from analysis data) ──
    st.markdown("#### 🎯 Per-Voter Breakdown (Latest Scan)")
    if has_neuro_data:
        # Collect all voter names and their votes across stocks
        voter_summary = {}
        for s in _stocks_data:
            nd = s.get("neuro_decision", {})
            for v in nd.get("voters", []):
                vn = v.get("name", "?")
                if vn not in voter_summary:
                    voter_summary[vn] = {"buy": 0, "sell": 0, "hold": 0, "total_conv": 0, "count": 0}
                vote_val = v.get("vote", 0)
                if vote_val > 0.1:
                    voter_summary[vn]["buy"] += 1
                elif vote_val < -0.1:
                    voter_summary[vn]["sell"] += 1
                else:
                    voter_summary[vn]["hold"] += 1
                voter_summary[vn]["total_conv"] += v.get("conviction", 0)
                voter_summary[vn]["count"] += 1

        if voter_summary:
            voter_icons = {"MomentumTrader": "📈", "ValueHunter": "💎", "RiskManager": "🛡️",
                           "SentimentAnalyst": "📰", "PatternRecognizer": "🔬", "QuantBrain": "🤖"}

            for vn, vs in voter_summary.items():
                avg_c = vs["total_conv"] / vs["count"] if vs["count"] > 0 else 0
                icon = voter_icons.get(vn, "🗳️")
                total = vs["buy"] + vs["sell"] + vs["hold"]
                buy_pct = (vs["buy"] / total * 100) if total > 0 else 0
                sell_pct = (vs["sell"] / total * 100) if total > 0 else 0
                hold_pct = (vs["hold"] / total * 100) if total > 0 else 0

                st.markdown(f"""
                <div style="display: flex; align-items: center; background: #09090b; border: 1px solid #27272a; border-radius: 6px; padding: 8px 14px; margin-bottom: 4px; gap: 12px;">
                    <div style="min-width: 160px; font-size: 0.8rem; color: #e4e4e7; font-weight: 600;">{icon} {vn}</div>
                    <div style="flex: 1; display: flex; height: 16px; border-radius: 3px; overflow: hidden; border: 1px solid #27272a;">
                        <div style="width: {buy_pct:.0f}%; background: #10b981;" title="Buy {vs['buy']}"></div>
                        <div style="width: {hold_pct:.0f}%; background: #71717a;" title="Hold {vs['hold']}"></div>
                        <div style="width: {sell_pct:.0f}%; background: #ef4444;" title="Sell {vs['sell']}"></div>
                    </div>
                    <div style="min-width: 130px; font-size: 0.7rem; color: #a1a1aa; text-align: right;">
                        🟢{vs["buy"]} ⚪{vs["hold"]} 🔴{vs["sell"]} | Conv: {avg_c:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("Voter breakdown will appear after the first NeuroVoter scan.")


# ══════════════════════════════════════════════════
#  TAB 3 — PERFORMANCE & ALERTS
# ══════════════════════════════════════════════════
if page == "Performance & Alerts":
    st.markdown('<div class="section-header">📈 <h3>Performance Dashboard & Alert Centre</h3></div>', unsafe_allow_html=True)

    # ── Section 1: Multi-Day Equity Curve ──────────────────
    st.markdown("### 💰 Multi-Day P&L — Equity Curve")
    th = load_csv(FILE_TRADES)
    if not th.empty and "Date" in th.columns and "Profit" in th.columns:
        try:
            eq = th.copy()
            eq["Date"] = pd.to_datetime(eq["Date"], errors="coerce")
            eq = eq.dropna(subset=["Date"])
            eq["Profit"] = pd.to_numeric(eq["Profit"], errors="coerce").fillna(0)
            daily_pnl = eq.groupby(eq["Date"].dt.date)["Profit"].sum().reset_index()
            daily_pnl.columns = ["Date", "Daily_PnL"]
            daily_pnl = daily_pnl.sort_values("Date")
            daily_pnl["Cumulative"] = daily_pnl["Daily_PnL"].cumsum()
            daily_pnl["Running_Balance"] = 1000 + daily_pnl["Cumulative"]

            # Key stats
            total_pnl = daily_pnl["Cumulative"].iloc[-1] if len(daily_pnl) > 0 else 0
            max_dd = 0
            peak = daily_pnl["Running_Balance"].iloc[0] if len(daily_pnl) > 0 else 1000
            for bal in daily_pnl["Running_Balance"]:
                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            win_days = (daily_pnl["Daily_PnL"] > 0).sum()
            loss_days = (daily_pnl["Daily_PnL"] < 0).sum()
            total_days = len(daily_pnl)
            avg_daily = daily_pnl["Daily_PnL"].mean() if total_days > 0 else 0
            best_day = daily_pnl["Daily_PnL"].max() if total_days > 0 else 0
            worst_day = daily_pnl["Daily_PnL"].min() if total_days > 0 else 0

            # Win/Loss streak
            streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            cur_type = None
            for p in daily_pnl["Daily_PnL"]:
                if p > 0:
                    if cur_type == "win":
                        streak += 1
                    else:
                        streak = 1
                        cur_type = "win"
                    max_win_streak = max(max_win_streak, streak)
                elif p < 0:
                    if cur_type == "loss":
                        streak += 1
                    else:
                        streak = 1
                        cur_type = "loss"
                    max_loss_streak = max(max_loss_streak, streak)

            pc1, pc2, pc3, pc4 = st.columns(4)
            pnl_color = "#00ff88" if total_pnl >= 0 else "#ff4444"
            pc1.markdown(f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:15px;border-radius:10px;text-align:center;border:1px solid {pnl_color}">'
                         f'<div style="font-size:12px;color:#aaa">Total P&L</div>'
                         f'<div style="font-size:24px;font-weight:bold;color:{pnl_color}">₹{total_pnl:+.2f}</div></div>', unsafe_allow_html=True)
            pc2.markdown(f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:15px;border-radius:10px;text-align:center;border:1px solid #ff9800">'
                         f'<div style="font-size:12px;color:#aaa">Max Drawdown</div>'
                         f'<div style="font-size:24px;font-weight:bold;color:#ff9800">{max_dd:.1f}%</div></div>', unsafe_allow_html=True)
            pc3.markdown(f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:15px;border-radius:10px;text-align:center;border:1px solid #29b6f6">'
                         f'<div style="font-size:12px;color:#aaa">Win/Loss Days</div>'
                         f'<div style="font-size:24px;font-weight:bold;color:#29b6f6">{win_days}W / {loss_days}L</div></div>', unsafe_allow_html=True)
            pc4.markdown(f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:15px;border-radius:10px;text-align:center;border:1px solid #ab47bc">'
                         f'<div style="font-size:12px;color:#aaa">Avg Daily P&L</div>'
                         f'<div style="font-size:24px;font-weight:bold;color:#ab47bc">₹{avg_daily:+.2f}</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Equity curve chart
            import plotly.graph_objects as go
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=daily_pnl["Date"], y=daily_pnl["Running_Balance"],
                mode="lines+markers", name="Balance",
                line=dict(color="#00ff88", width=3),
                marker=dict(size=6, color=["#00ff88" if p >= 0 else "#ff4444" for p in daily_pnl["Daily_PnL"]]),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.08)",
                hovertemplate="Date: %{x}<br>Balance: ₹%{y:.2f}<extra></extra>"
            ))
            fig_eq.add_hline(y=1000, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Starting ₹1000")
            fig_eq.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=350, margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Date", yaxis_title="Portfolio Balance (₹)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # Daily P&L bar chart
            fig_bar = go.Figure()
            colors_bar = ["#00ff88" if p >= 0 else "#ff4444" for p in daily_pnl["Daily_PnL"]]
            fig_bar.add_trace(go.Bar(
                x=daily_pnl["Date"], y=daily_pnl["Daily_PnL"],
                marker_color=colors_bar, name="Daily P&L",
                hovertemplate="Date: %{x}<br>P&L: ₹%{y:+.2f}<extra></extra>"
            ))
            fig_bar.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=250, margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="Date", yaxis_title="Daily P&L (₹)",
                font=dict(color="#e0e0e0")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Streak section
            st.markdown(f"""
            <div style="display:flex;gap:15px;margin-bottom:15px">
                <div style="flex:1;background:linear-gradient(135deg,#1a1a2e,#16213e);padding:12px;border-radius:10px;text-align:center;border:1px solid #444">
                    <div style="color:#aaa;font-size:11px">Best Day</div>
                    <div style="color:#00ff88;font-size:18px;font-weight:bold">₹{best_day:+.2f}</div>
                </div>
                <div style="flex:1;background:linear-gradient(135deg,#1a1a2e,#16213e);padding:12px;border-radius:10px;text-align:center;border:1px solid #444">
                    <div style="color:#aaa;font-size:11px">Worst Day</div>
                    <div style="color:#ff4444;font-size:18px;font-weight:bold">₹{worst_day:+.2f}</div>
                </div>
                <div style="flex:1;background:linear-gradient(135deg,#1a1a2e,#16213e);padding:12px;border-radius:10px;text-align:center;border:1px solid #444">
                    <div style="color:#aaa;font-size:11px">🔥 Best Win Streak</div>
                    <div style="color:#00ff88;font-size:18px;font-weight:bold">{max_win_streak} days</div>
                </div>
                <div style="flex:1;background:linear-gradient(135deg,#1a1a2e,#16213e);padding:12px;border-radius:10px;text-align:center;border:1px solid #444">
                    <div style="color:#aaa;font-size:11px">💀 Worst Loss Streak</div>
                    <div style="color:#ff4444;font-size:18px;font-weight:bold">{max_loss_streak} days</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Equity curve error: {e}")
    else:
        st.info("📭 No trade history yet. Equity curve will appear after trades are executed.")

    st.markdown("---")

    # ── Section 2: Portfolio Heatmap ───────────────────────
    st.markdown("### 🗺️ Portfolio Heatmap — All Stocks at a Glance")
    la = load_json(FILE_LIVE)
    if la:
        try:
            stocks_data = la if isinstance(la, list) else la.get("stocks", la.get("analysis", []))
            if isinstance(stocks_data, dict):
                stocks_data = list(stocks_data.values()) if stocks_data else []

            if stocks_data and len(stocks_data) > 0:
                hmap_symbols = []
                hmap_signals = []
                hmap_neuro = []
                hmap_regimes = []
                hmap_colors = []

                for s in stocks_data:
                    if isinstance(s, dict):
                        sym = s.get("symbol", s.get("Symbol", "?"))
                        sig = s.get("ensemble_signal", s.get("signal", s.get("Signal", "HOLD")))
                        ns = s.get("neuro_score", s.get("NeuroScore", 0.5))
                        regime = s.get("regime", s.get("Regime", "Unknown"))
                        hmap_symbols.append(sym.replace("_NS", "").replace(".NS", ""))
                        hmap_signals.append(str(sig).upper())
                        try:
                            ns_val = float(ns)
                        except (ValueError, TypeError):
                            ns_val = 0.5
                        hmap_neuro.append(ns_val)
                        hmap_regimes.append(str(regime))
                        if str(sig).upper() == "BUY":
                            hmap_colors.append("#00ff88")
                        elif str(sig).upper() == "SELL":
                            hmap_colors.append("#ff4444")
                        else:
                            hmap_colors.append("#ffa726")

                if hmap_symbols:
                    # Build a grid
                    n_cols = min(5, len(hmap_symbols))
                    rows_needed = (len(hmap_symbols) + n_cols - 1) // n_cols
                    idx = 0
                    for r in range(rows_needed):
                        cols = st.columns(n_cols)
                        for c in range(n_cols):
                            if idx < len(hmap_symbols):
                                sym = hmap_symbols[idx]
                                sig = hmap_signals[idx]
                                ns_val = hmap_neuro[idx]
                                regime = hmap_regimes[idx]
                                color = hmap_colors[idx]
                                ns_bar = int(ns_val * 100)
                                sig_emoji = "🟢" if sig == "BUY" else ("🔴" if sig == "SELL" else "🟡")
                                cols[c].markdown(f"""
                                <div style="background:linear-gradient(135deg,#0d1117,#161b22);padding:15px;border-radius:12px;
                                    border-left:4px solid {color};margin-bottom:8px;text-align:center">
                                    <div style="font-size:16px;font-weight:bold;color:#e0e0e0">{sym}</div>
                                    <div style="font-size:28px;margin:5px 0">{sig_emoji}</div>
                                    <div style="color:{color};font-weight:bold;font-size:13px">{sig}</div>
                                    <div style="margin:8px 0;background:#333;border-radius:5px;height:8px;overflow:hidden">
                                        <div style="width:{ns_bar}%;height:100%;background:linear-gradient(90deg,#ff4444,#ffa726,#00ff88);border-radius:5px"></div>
                                    </div>
                                    <div style="color:#888;font-size:10px">NeuroScore: {ns_val:.2f} | {regime}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                idx += 1
                else:
                    st.info("No stock data available for heatmap.")
            else:
                st.info("📭 Live analysis data format not recognised. Run sniper first.")
        except Exception as e:
            st.error(f"Heatmap error: {e}")
    else:
        st.info("📭 No live analysis data. Run `python src/sniper.py` to generate.")

    st.markdown("---")

    # ── Section 3: Voter Accuracy Leaderboard ─────────────
    st.markdown("### 🎯 NeuroVoter Accuracy Leaderboard")
    va = load_json(FILE_VOTER_ACC)
    if va:
        try:
            voters_acc = va.get("voters", va) if isinstance(va, dict) else {}
            if isinstance(voters_acc, list):
                voters_acc = {v.get("name", f"V{i}"): v for i, v in enumerate(voters_acc)}

            if voters_acc:
                # Sort by accuracy descending
                sorted_voters = sorted(voters_acc.items(), key=lambda x: x[1].get("accuracy", 0) if isinstance(x[1], dict) else 0, reverse=True)

                # Podium display
                voter_names = []
                voter_accs = []
                voter_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#29b6f6", "#ab47bc", "#ff7043"]
                for i, (vname, vdata) in enumerate(sorted_voters):
                    acc = vdata.get("accuracy", 0) if isinstance(vdata, dict) else 0
                    voter_names.append(vname)
                    voter_accs.append(acc * 100 if acc <= 1 else acc)

                fig_va = go.Figure()
                fig_va.add_trace(go.Bar(
                    x=voter_names, y=voter_accs,
                    marker_color=voter_colors[:len(voter_names)],
                    text=[f"{a:.1f}%" for a in voter_accs],
                    textposition="outside",
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
                ))
                fig_va.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=300, margin=dict(l=20, r=20, t=10, b=20),
                    yaxis_title="Accuracy %", yaxis_range=[0, 105],
                    font=dict(color="#e0e0e0")
                )
                st.plotly_chart(fig_va, use_container_width=True)

                # Recommended weights
                rec_weights = va.get("recommended_weights", {})
                if rec_weights:
                    st.markdown("**🔧 Adaptive Weight Recommendations** (from Learner)")
                    wt_cols = st.columns(min(6, len(rec_weights)))
                    for i, (wn, wv) in enumerate(rec_weights.items()):
                        if i < len(wt_cols):
                            wt_cols[i].markdown(f"""
                            <div style="background:#1a1a2e;padding:10px;border-radius:8px;text-align:center;border:1px solid #444">
                                <div style="color:#aaa;font-size:10px">{wn}</div>
                                <div style="color:#ffd700;font-size:16px;font-weight:bold">{wv:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No voter accuracy data found.")
        except Exception as e:
            st.error(f"Voter accuracy display error: {e}")
    else:
        st.info("📭 Voter accuracy not yet computed. Run `python src/learner.py` after some trades.")

    st.markdown("---")

    # ── Section 4: Alert History Feed ─────────────────────
    st.markdown("### 🔔 Alert History — Recent Notifications")
    alerts = load_json(FILE_ALERTS)
    if alerts:
        try:
            alert_list = alerts if isinstance(alerts, list) else alerts.get("alerts", [])
            if alert_list:
                # Show last 25 alerts, newest first
                recent = list(reversed(alert_list[-25:]))
                alert_icons = {
                    "BUY": "🟢", "EXIT": "🔴", "VETO": "🚫", "GUARDIAN_STOP": "🛡️",
                    "REGIME_CHANGE": "🌊", "DAILY_SUMMARY": "📊", "CUSTOM": "📌"
                }
                for a in recent:
                    atype = a.get("type", "CUSTOM")
                    icon = alert_icons.get(atype, "📌")
                    ts = a.get("timestamp", "")[:19]
                    msg = a.get("message", str(a.get("data", "")))
                    sym = a.get("symbol", "")
                    sym_badge = f'<span style="background:#1e3a5f;color:#29b6f6;padding:2px 8px;border-radius:4px;font-size:11px;margin-right:5px">{sym}</span>' if sym else ""

                    type_colors = {
                        "BUY": "#00ff88", "EXIT": "#ff4444", "VETO": "#ff9800",
                        "GUARDIAN_STOP": "#f44336", "REGIME_CHANGE": "#ab47bc",
                        "DAILY_SUMMARY": "#29b6f6", "CUSTOM": "#888"
                    }
                    tc = type_colors.get(atype, "#888")

                    st.markdown(f"""
                    <div style="background:linear-gradient(90deg,rgba({','.join(str(int(tc.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.08),transparent);
                        padding:8px 12px;border-radius:8px;border-left:3px solid {tc};margin-bottom:4px;font-size:13px">
                        <span style="color:#666;font-size:11px">{ts}</span>
                        {sym_badge}
                        <span style="font-size:14px">{icon}</span>
                        <span style="color:{tc};font-weight:bold;font-size:11px">{atype}</span>
                        <span style="color:#ccc;margin-left:8px">{msg[:120]}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No alerts in history yet.")
        except Exception as e:
            st.error(f"Alert feed error: {e}")
    else:
        st.info("📭 No alert history. Alerts will appear here after sniper runs with notifications enabled.")

    # ── Section 5: Voter Decision History ─────────────────
    st.markdown("---")
    st.markdown("### 📜 Voter Decision Log — Recent Decisions")
    vh = load_json(FILE_VOTER_HIST)
    if vh:
        try:
            hist_list = vh if isinstance(vh, list) else vh.get("history", [])
            if hist_list:
                recent_vh = list(reversed(hist_list[-20:]))
                for entry in recent_vh:
                    ts = entry.get("timestamp", "")[:19]
                    sym = entry.get("symbol", "?")
                    action = entry.get("trade_action", "?")
                    price = entry.get("price", 0)
                    voters = entry.get("voters", {})

                    action_colors = {"BUY": "#00ff88", "SKIP": "#ffa726", "VETOED": "#ff4444"}
                    ac = action_colors.get(action, "#888")

                    voter_chips = ""
                    if isinstance(voters, dict):
                        for vn, vd in voters.items():
                            if isinstance(vd, dict):
                                vote = vd.get("vote", "?")
                                conf = vd.get("conviction", 0)
                                veto = vd.get("veto", False)
                                vc = "#00ff88" if vote == "BUY" else ("#ff4444" if vote == "SELL" else "#ffa726")
                                veto_mark = "🚫" if veto else ""
                                voter_chips += f'<span style="background:rgba({",".join(str(int(vc.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.15);color:{vc};padding:2px 6px;border-radius:4px;font-size:10px;margin:1px">{vn}:{vote}({conf:.0%}){veto_mark}</span> '

                    st.markdown(f"""
                    <div style="background:#0d1117;padding:8px 12px;border-radius:8px;border-left:3px solid {ac};margin-bottom:4px">
                        <span style="color:#666;font-size:11px">{ts}</span>
                        <span style="background:#1e3a5f;color:#29b6f6;padding:2px 8px;border-radius:4px;font-size:11px;margin:0 5px">{sym}</span>
                        <span style="color:{ac};font-weight:bold">{action}</span>
                        <span style="color:#888;margin-left:5px">@ ₹{price}</span>
                        <div style="margin-top:4px">{voter_chips}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No voter decisions recorded yet.")
        except Exception as e:
            st.error(f"Voter history error: {e}")
    else:
        st.info("📭 No voter history. This will populate after sniper runs with NeuroVoter enabled.")


# ══════════════════════════════════════════════════
#  TAB 4 — BACKTEST P&L ANALYSIS
# ══════════════════════════════════════════════════
if page == "Backtest P&L":
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
#  TAB 5 — RANKING & SIGNALS
# ══════════════════════════════════════════════════
if page == "Ranking & Signals":
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
#  TAB 6 — MODEL DETAILS & HEALTH
# ══════════════════════════════════════════════════
if page == "Model Details":
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
#  TAB 7 — TRADE HISTORY (Live Sniper)
# ══════════════════════════════════════════════════
if page == "Trade History":
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

        # ── Exit Type Breakdown ──
        _exit_col = "Exit_Type" if "Exit_Type" in trades_df.columns else ("exit_type" if "exit_type" in trades_df.columns else None)
        if _exit_col and trades_df[_exit_col].notna().any():
            st.markdown("### 🚪 Smart Exit Breakdown")
            exit_counts = trades_df[_exit_col].value_counts()
            _et_color_map = {
                'TARGET_HIT': '#10b981', 'STOP_LOSS': '#ef4444', 'MOMENTUM_EXIT': '#f59e0b',
                'RSI_EXIT': '#8b5cf6', 'VOLUME_EXIT': '#06b6d4', 'TIME_DECAY': '#a1a1aa',
                'SENTIMENT_EXIT': '#ec4899', 'MAX_HOLD': '#84cc16',
            }
            e_cols = st.columns(min(len(exit_counts), 4))
            for i, (etype, ecount) in enumerate(exit_counts.items()):
                with e_cols[i % len(e_cols)]:
                    ec = _et_color_map.get(str(etype), '#a1a1aa')
                    st.markdown(f"""
                    <div style="background: #09090b; border: 1px solid #27272a; border-left: 3px solid {ec}; border-radius: 6px; padding: 12px 14px; text-align: center; margin-bottom: 8px;">
                        <div style="font-size: 0.72rem; color: {ec}; font-weight: 600;">{etype}</div>
                        <div style="font-size: 1.3rem; color: #e4e4e7; font-weight: 700;">{ecount}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Exit type vs average P&L
            if "Actual_Profit" in trades_df.columns:
                exit_pnl = trades_df.groupby(_exit_col)["Actual_Profit"].agg(["mean", "sum", "count"])
                exit_pnl.columns = ["Avg P&L ₹", "Total P&L ₹", "Count"]
                exit_pnl = exit_pnl.round(2)
                st.dataframe(exit_pnl, use_container_width=True)

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
#  TAB 8 — LEARNER & GUARDIAN REPORT
# ══════════════════════════════════════════════════
if page == "Learner & Guardian":
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
#  TAB 9 — MARKET INTELLIGENCE
# ══════════════════════════════════════════════════
if page == "Market Intelligence":
    st.markdown('<div class="section-header">🌐 <h3>Market Intelligence — Breadth, Sectors, Correlation & Timing</h3></div>', unsafe_allow_html=True)

    # ── A) Market Breadth ──
    st.markdown("### 📊 Market Breadth")
    breadth = load_json(FILE_BREADTH)
    if breadth:
        sig = breadth.get("signal", "UNKNOWN")
        score = breadth.get("composite_score", 50)
        sf = breadth.get("position_size_factor", 1.0)
        vix = breadth.get("india_vix", "N/A")
        sig_icons = {"HEALTHY": "🟢", "CAUTION": "🟡", "WEAK": "🔴"}

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Signal", f"{sig_icons.get(sig, '⚪')} {sig}")
        with c2:
            st.metric("Composite Score", f"{score}/100")
        with c3:
            st.metric("Position Size Factor", f"{sf:.0%}")
        with c4:
            st.metric("India VIX", f"{vix}")

        # Breadth components
        components = breadth.get("components", {})
        if components:
            c1, c2, c3 = st.columns(3)
            with c1:
                ad = components.get("advance_decline_ratio", 0)
                st.metric("A/D Ratio", f"{ad:.2f}")
            with c2:
                above_50 = components.get("pct_above_sma50", 0)
                st.metric("% Above SMA50", f"{above_50:.0f}%")
            with c3:
                above_200 = components.get("pct_above_sma200", 0)
                st.metric("% Above SMA200", f"{above_200:.0f}%")

        # Breadth gauge chart
        try:
            import plotly.graph_objects as go
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "Market Health"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00d4aa"},
                    "steps": [
                        {"range": [0, 30], "color": "#ff4444"},
                        {"range": [30, 60], "color": "#ffa500"},
                        {"range": [60, 100], "color": "#00cc66"},
                    ],
                },
            ))
            fig_gauge.update_layout(
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception:
            pass
    else:
        st.info("Market breadth data not available. Run sniper to generate.")

    st.divider()

    # ── B) Sector Rotation ──
    st.markdown("### 🔄 Sector Rotation Heatmap")
    sector = load_json(FILE_SECTOR)
    if sector and sector.get("sectors"):
        sectors_data = sector["sectors"]
        rows = []
        for sname, sdata in sectors_data.items():
            state_icon = {
                "STRONG": "🟢", "ROTATING_IN": "🔵", "NEUTRAL": "⚪",
                "WEAKENING": "🟡", "ROTATING_OUT": "🔴",
            }
            rows.append({
                "Sector": sname,
                "State": f"{state_icon.get(sdata.get('state', ''), '⚪')} {sdata.get('state', 'N/A')}",
                "5D Return": f"{sdata.get('return_5d', 0):.2f}%",
                "10D Return": f"{sdata.get('return_10d', 0):.2f}%",
                "20D Return": f"{sdata.get('return_20d', 0):.2f}%",
                "Momentum": f"{sdata.get('acceleration', 0):+.3f}",
                "Volume": f"{sdata.get('volume_trend', 0):.2f}x",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Heatmap using plotly
        try:
            import plotly.express as px
            hm_data = []
            for sname, sdata in sectors_data.items():
                hm_data.append({
                    "Sector": sname,
                    "5D": sdata.get("return_5d", 0),
                    "10D": sdata.get("return_10d", 0),
                    "20D": sdata.get("return_20d", 0),
                })
            hm_df = pd.DataFrame(hm_data).set_index("Sector")
            fig_hm = px.imshow(
                hm_df.values, x=hm_df.columns.tolist(), y=hm_df.index.tolist(),
                color_continuous_scale="RdYlGn", text_auto=".2f",
                labels={"color": "Return %"},
            )
            fig_hm.update_layout(
                title="Sector Returns Heatmap",
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception:
            pass

        # Sector recommendations
        recs = sector.get("stock_recommendations", {})
        if recs:
            st.markdown("#### Stock Sector Adjustments")
            rec_rows = []
            for stk, rec in recs.items():
                rec_icon = {"OVERWEIGHT": "⬆️", "UNDERWEIGHT": "⬇️", "NEUTRAL": "➡️"}
                rec_rows.append({
                    "Stock": stk.replace(".NS", ""),
                    "Sector": rec.get("sector", ""),
                    "Recommendation": f"{rec_icon.get(rec.get('recommendation', ''), '❓')} {rec.get('recommendation', 'N/A')}",
                    "Sector State": rec.get("sector_state", ""),
                })
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Sector rotation data not available yet.")

    st.divider()

    # ── C) Correlation Matrix ──
    st.markdown("### 🔗 Correlation Guard")
    corr = load_json(FILE_CORR)
    if corr:
        c1, c2 = st.columns(2)
        with c1:
            div_score = corr.get("diversification_score", "N/A")
            st.metric("Diversification Score", f"{div_score}/100")
        with c2:
            high_pairs = corr.get("high_correlation_pairs", [])
            st.metric("High Correlation Pairs", len(high_pairs))

        if high_pairs:
            st.markdown("**⚠️ High Correlation Pairs:**")
            pair_rows = []
            for p in high_pairs[:10]:
                pair_rows.append({
                    "Stock A": str(p.get("stock_a", "")).replace(".NS", ""),
                    "Stock B": str(p.get("stock_b", "")).replace(".NS", ""),
                    "Correlation": f"{p.get('correlation', 0):.3f}",
                    "Risk": "🔴 CRITICAL" if p.get("correlation", 0) >= 0.9 else "🟡 HIGH",
                })
            st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)

        # Correlation heatmap
        matrix = corr.get("matrix", {})
        if matrix:
            try:
                import plotly.express as px
                corr_df = pd.DataFrame(matrix)
                # Clean column names
                corr_df.columns = [c.replace(".NS", "") for c in corr_df.columns]
                corr_df.index = [c.replace(".NS", "") for c in corr_df.index]
                fig_corr = px.imshow(
                    corr_df, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, text_auto=".2f",
                )
                fig_corr.update_layout(
                    title="Stock Correlation Matrix",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception:
                pass
    else:
        st.info("Correlation data not available yet.")

    st.divider()

    # ── D) Intraday Patterns ──
    st.markdown("### ⏰ Intraday Time Patterns")
    intraday = load_json(FILE_INTRADAY)
    if intraday:
        # Current window
        analysis = load_json(FILE_ANALYSIS)
        current_w = {}
        if analysis:
            current_w = analysis.get("intraday_window", {})
        if current_w:
            st.markdown(f"**Current Window:** {current_w.get('window', 'N/A')} — "
                        f"Entry: {current_w.get('entry_signal', 'N/A')}")

        # Per-stock patterns
        patterns = intraday.get("patterns", {})
        for stk, windows in patterns.items():
            if isinstance(windows, dict):
                with st.expander(f"📈 {stk.replace('.NS', '')} — Time Windows"):
                    win_rows = []
                    for wname, wdata in windows.items():
                        entry_icons = {
                            "STRONG_ENTRY": "🟢", "GOOD_ENTRY": "🔵",
                            "NEUTRAL": "⚪", "WEAK_ENTRY": "🟡", "AVOID": "🔴",
                        }
                        es = wdata.get("entry_signal", "NEUTRAL")
                        win_rows.append({
                            "Window": wname,
                            "Avg Return": f"{wdata.get('avg_return', 0):.3f}%",
                            "Win Rate": f"{wdata.get('win_rate', 0):.0f}%",
                            "Volatility": f"{wdata.get('volatility', 0):.3f}",
                            "Entry Score": f"{wdata.get('entry_score', 0):.2f}",
                            "Signal": f"{entry_icons.get(es, '❓')} {es}",
                        })
                    st.dataframe(pd.DataFrame(win_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Intraday pattern data not available yet.")

    st.divider()

    # ── E) Confidence Decay ──
    st.markdown("### 📉 AI Confidence Decay Status")
    decay = load_json(FILE_DECAY)
    if decay:
        applied = decay.get("decay_applied", False)
        st.markdown(f"**Decay Active:** {'🔴 YES — weights adjusted' if applied else '🟢 NO — all voters healthy'}")

        voter_states = decay.get("voter_states", {})
        if voter_states:
            decay_rows = []
            for vname, vs in voter_states.items():
                status_icons = {
                    "SEVERE_DECAY": "🔴", "MILD_DECAY": "🟡",
                    "RECOVERING": "🟢", "STABLE": "⚪",
                    "INSUFFICIENT_DATA": "⏳",
                }
                decay_rows.append({
                    "Voter": vname,
                    "Accuracy": f"{vs.get('accuracy', 0):.1%}",
                    "Weight": f"{vs.get('weight', 0):.2%}",
                    "Status": f"{status_icons.get(vs.get('status', ''), '❓')} {vs.get('status', 'N/A')}",
                    "Decay Streak": vs.get("decay_streak", 0),
                    "Factor": f"{vs.get('factor_applied', 1.0):.2f}x",
                })
            st.dataframe(pd.DataFrame(decay_rows), use_container_width=True, hide_index=True)

        retrain = decay.get("retrain_flags", [])
        if retrain:
            st.warning(f"⚠️ {len(retrain)} voter(s) flagged for retraining!")
            for rf in retrain:
                st.markdown(f"- **{rf.get('voter', '?')}**: {rf.get('reason', '')}")
    else:
        st.info("Confidence decay data not available. Run learner.py to generate.")


# ══════════════════════════════════════════════════
#  TAB 10 — BROKER & REPORTS
# ══════════════════════════════════════════════════
if page == "Broker & Reports":
    st.markdown('<div class="section-header">🏦 <h3>Broker Bridge & Daily Reports</h3></div>', unsafe_allow_html=True)

    # ── A) Broker Status ──
    st.markdown("### 🔌 Broker Connection Status")
    analysis = load_json(FILE_ANALYSIS)
    broker_mode = "PAPER"
    broker_name = "PaperBroker"
    if analysis:
        broker_mode = analysis.get("broker_mode", "PAPER")
        broker_name = analysis.get("broker_name", "PaperBroker")

    c1, c2, c3 = st.columns(3)
    with c1:
        mode_icons = {"PAPER": "📝", "DRY_RUN": "🔍", "LIVE": "🔴"}
        st.metric("Trade Mode", f"{mode_icons.get(broker_mode, '❓')} {broker_mode}")
    with c2:
        st.metric("Broker", broker_name)
    with c3:
        broker_icons = {"PaperBroker": "🟢", "ZerodhaBroker": "🔵", "AngelOneBroker": "🟠", "GrowwBroker": "🟣"}
        st.metric("Status", f"{broker_icons.get(broker_name, '⚪')} Connected")

    # Mode explanation
    mode_desc = {
        "PAPER": "Simulated trades — no real money involved. Perfect for testing.",
        "DRY_RUN": "Order preview mode — generates real orders but does NOT send them. Check broker_orders.json.",
        "LIVE": "⚠️ REAL MONEY — Orders are sent to your broker. Double-check everything!",
    }
    st.info(mode_desc.get(broker_mode, "Unknown mode"))

    st.divider()

    # ── B) Recent Broker Orders ──
    st.markdown("### 📋 Recent Broker Orders")
    broker_orders = load_json(FILE_BROKER_LOG)
    if broker_orders and isinstance(broker_orders, list):
        order_rows = []
        for order in broker_orders[-20:][::-1]:  # Last 20, newest first
            order_rows.append({
                "Time": order.get("timestamp", "")[:19],
                "Symbol": str(order.get("symbol", "")).replace(".NS", ""),
                "Side": order.get("side", ""),
                "Qty": order.get("qty", 0),
                "Price": f"₹{order.get('price', 0):,.2f}",
                "Mode": order.get("mode", ""),
                "Status": order.get("status", ""),
            })
        st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No broker orders logged yet. Orders appear here once sniper fires bullets.")

    st.divider()

    # ── C) Broker Setup Guide ──
    with st.expander("🔧 How to Connect a Real Broker"):
        st.markdown("""
**Step 1:** Set environment variables before running sniper:

```bash
# Choose mode
set AEGIS_TRADE_MODE=DRY_RUN    # or LIVE

# Choose broker
set AEGIS_BROKER=ZERODHA         # or ANGEL_ONE or GROWW
```

**Step 2 (Zerodha):**
```bash
set ZERODHA_API_KEY=your_api_key
set ZERODHA_API_SECRET=your_secret
set ZERODHA_ACCESS_TOKEN=your_token
```

**Step 2 (Angel One):**
```bash
set ANGEL_API_KEY=your_api_key
set ANGEL_CLIENT_ID=your_client_id
set ANGEL_PASSWORD=your_password
set ANGEL_TOTP_SECRET=your_totp_secret
```

**Step 2 (Groww):**
```bash
set GROWW_ACCESS_TOKEN=your_bearer_token
```

**Step 3:** Install broker SDK:
```bash
pip install kiteconnect       # Zerodha
pip install smartapi-python pyotp  # Angel One
# Groww uses requests (already installed)
```

**⚠️ IMPORTANT:** Always start with `DRY_RUN` to verify order logic before going `LIVE`!
        """)

    st.divider()

    # ── D) Daily Reports ──
    st.markdown("### 📄 Daily Reports")
    if os.path.exists(REPORTS_DIR):
        reports = sorted(
            [f for f in os.listdir(REPORTS_DIR) if f.endswith((".html", ".pdf"))],
            reverse=True,
        )
        if reports:
            st.success(f"**{len(reports)} report(s) available**")
            for rname in reports[:10]:
                rpath = os.path.join(REPORTS_DIR, rname)
                with open(rpath, "rb") as rf:
                    data = rf.read()
                ext = rname.split(".")[-1].upper()
                st.download_button(
                    label=f"📥 Download {rname}",
                    data=data,
                    file_name=rname,
                    mime="text/html" if ext == "HTML" else "application/pdf",
                    key=f"dl_{rname}",
                )
        else:
            st.info("No reports generated yet. Reports are created at end-of-day automatically.")
    else:
        st.info("Reports directory doesn't exist yet. It will be created when the first report is generated.")

    st.divider()

    # ── E) Manual Report Generation ──
    st.markdown("### 🔄 Generate Report Now")
    if st.button("📊 Generate Today's Report", key="gen_report"):
        try:
            sys.path.insert(0, os.path.join(BASE, "src"))
            from report_exporter import generate_report as gen_rpt
            path = gen_rpt()
            st.success(f"✅ Report generated: {path}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate report: {e}")


# ══════════════════════════════════════════════════
#  TAB 11 — PORTFOLIO TREEMAP & KELLY SIZING
# ══════════════════════════════════════════════════
if page == "Portfolio Treemap":
    st.markdown('<div class="section-header">🗺️ <h3>Portfolio Treemap & Kelly Sizing</h3></div>', unsafe_allow_html=True)

    # ── A) Portfolio Treemap ──
    st.markdown("### 🗺️ Open Positions Treemap")
    try:
        import plotly.express as px
        analysis_tm = load_json(FILE_ANALYSIS)
        state_tm = load_json(FILE_STATE)
        have_positions = False

        if analysis_tm:
            active = analysis_tm.get("active_positions", [])
            if not active and state_tm:
                active = [
                    t for t in state_tm.get("active_trades", [])
                    if t.get("status") == "OPEN"
                ]

            if active:
                have_positions = True
                tm_data = []
                for p in active:
                    sym = p.get("stock", p.get("symbol", "?"))
                    entry = p.get("price", p.get("entry_price", 0))
                    qty = p.get("qty", 0)
                    current = p.get("current_price", entry)
                    value = current * qty if current else entry * qty
                    pnl = (current - entry) * qty if current and entry else 0
                    pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                    tm_data.append({
                        "Symbol": sym,
                        "Value": max(value, 1),
                        "PnL": round(pnl, 2),
                        "PnL%": round(pnl_pct, 2),
                        "Qty": qty,
                        "Entry": round(entry, 2),
                        "Current": round(current, 2),
                    })

                if tm_data:
                    df_tm = pd.DataFrame(tm_data)
                    fig_tm = px.treemap(
                        df_tm,
                        path=["Symbol"],
                        values="Value",
                        color="PnL%",
                        color_continuous_scale=["#ff4444", "#ffbb33", "#00C851"],
                        color_continuous_midpoint=0,
                        hover_data=["Entry", "Current", "PnL", "Qty"],
                        title="Position Size by Capital Allocated, Coloured by P&L%",
                    )
                    fig_tm.update_layout(height=500, margin=dict(t=40, l=10, r=10, b=10))
                    st.plotly_chart(fig_tm, use_container_width=True)

                    # Summary table
                    st.dataframe(df_tm, use_container_width=True, hide_index=True)
        if not have_positions:
            st.info("No open positions to display. Treemap appears when positions are open.")
    except Exception as e:
        st.warning(f"Treemap error: {e}")

    st.divider()

    # ── B) Kelly Criterion Sizing ──
    st.markdown("### 💰 Kelly Criterion Sizing")
    kelly_data = load_json(FILE_KELLY)
    if kelly_data:
        overall = kelly_data.get("overall", {})
        per_stock = kelly_data.get("per_stock", {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Portfolio Win Rate", f"{overall.get('win_rate', 0)*100:.1f}%")
        with c2:
            st.metric("Payoff Ratio", f"{overall.get('payoff_ratio', 0):.2f}")
        with c3:
            st.metric("Kelly %", f"{overall.get('kelly_adj', 0)*100:.2f}%")
        with c4:
            st.metric("Trade Sample", str(overall.get("trades", 0)))

        if per_stock:
            kelly_rows = []
            for sym, info in sorted(per_stock.items(), key=lambda x: x[1].get("kelly_adj", 0), reverse=True):
                kelly_rows.append({
                    "Symbol": sym,
                    "Win Rate": f"{info.get('win_rate', 0)*100:.1f}%",
                    "Payoff": f"{info.get('payoff_ratio', 0):.2f}",
                    "Raw Kelly": f"{info.get('kelly_raw', 0)*100:.2f}%",
                    "Adjusted Kelly": f"{info.get('kelly_adj', 0)*100:.2f}%",
                    "Recommended %": f"{info.get('recommended_pct', 0):.2f}%",
                    "Trades": info.get("trades", 0),
                })
            st.dataframe(pd.DataFrame(kelly_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No per-stock Kelly data yet. Will populate after enough trades.")

        st.caption(f"Updated: {kelly_data.get('timestamp', 'N/A')}")
    else:
        st.info("Kelly sizing data not available yet. Run at least 10 trades.")

    st.divider()

    # ── C) Earnings Calendar ──
    st.markdown("### 📅 Earnings Calendar Guard")
    earnings_data = load_json(FILE_EARNINGS)
    if earnings_data:
        calendar = earnings_data.get("calendar", {})
        if calendar:
            today = datetime.now()
            earn_rows = []
            for sym, info in calendar.items():
                ed = info.get("next_earnings", "?")
                source = info.get("source", "?")
                try:
                    dt = datetime.strptime(ed, "%Y-%m-%d")
                    days_to = (dt - today).days
                except Exception:
                    days_to = 999
                status = "🚫 BLOCKED" if days_to <= 3 else ("⚠️ NEAR" if days_to <= 5 else "✅ CLEAR")
                earn_rows.append({
                    "Symbol": sym,
                    "Earnings Date": ed,
                    "Days Away": days_to,
                    "Status": status,
                    "Source": source,
                })
            earn_df = pd.DataFrame(earn_rows).sort_values("Days Away")
            st.dataframe(earn_df, use_container_width=True, hide_index=True)
        st.caption(f"Updated: {earnings_data.get('timestamp', 'N/A')}")
    else:
        st.info("Earnings calendar not loaded yet. Will populate on next Sniper run.")


# ══════════════════════════════════════════════════
#  TAB 12 — WALK-FORWARD SIMULATOR & MODEL DRIFT
# ══════════════════════════════════════════════════
if page == "Walk-Forward & Drift":
    st.markdown('<div class="section-header">🔄 <h3>Walk-Forward Simulator & Model Drift</h3></div>', unsafe_allow_html=True)

    # ── A) Walk-Forward Results ──
    st.markdown("### 🔄 Walk-Forward Simulation Results")
    wf_data = load_json(FILE_WF_RESULTS)
    if wf_data:
        summary = wf_data.get("summary", {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ret = summary.get("total_return", 0)
            st.metric("Total Return", f"₹{ret:,.2f}", delta=f"{summary.get('total_return_pct', 0):+.2f}%")
        with c2:
            st.metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
        with c3:
            st.metric("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")
        with c4:
            st.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Total Trades", str(summary.get("total_trades", 0)))
        with c6:
            st.metric("Max Drawdown", f"{summary.get('max_drawdown_pct', 0):.2f}%")
        with c7:
            st.metric("Avg Hold Days", f"{summary.get('avg_hold_days', 0):.1f}")
        with c8:
            st.metric("Avg Win", f"₹{summary.get('avg_win', 0):,.2f}")

        # Equity curve
        eq_curve = wf_data.get("equity_curve", [])
        if eq_curve:
            import plotly.graph_objects as go
            eq_df = pd.DataFrame(eq_curve)
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_df["date"], y=eq_df["equity"],
                mode="lines", name="Equity",
                line=dict(color="#00C851", width=2),
                fill="tozeroy", fillcolor="rgba(0,200,81,0.1)",
            ))
            fig_eq.update_layout(
                title="Walk-Forward Equity Curve",
                xaxis_title="Date", yaxis_title="Equity (₹)",
                height=400, template="plotly_dark",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # Per-stock breakdown
        stock_stats = wf_data.get("stock_stats", {})
        if stock_stats:
            st.markdown("#### Per-Stock Breakdown")
            ss_rows = []
            for sym, info in sorted(stock_stats.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
                ss_rows.append({
                    "Symbol": sym,
                    "Trades": info.get("trades", 0),
                    "Wins": info.get("wins", 0),
                    "Win Rate": f"{info.get('win_rate', 0):.1f}%",
                    "P&L": f"₹{info.get('pnl', 0):,.2f}",
                })
            st.dataframe(pd.DataFrame(ss_rows), use_container_width=True, hide_index=True)

        # Run simulator button
        with st.expander("🔄 Run Walk-Forward Simulation"):
            wf_days = st.slider("Simulation Days", 30, 365, 60)
            if st.button("▶️ Run Walk-Forward"):
                with st.spinner("Running walk-forward simulation... (may take a few minutes)"):
                    try:
                        sys.path.insert(0, os.path.join(BASE, "src"))
                        from walk_forward import run_walk_forward
                        results = run_walk_forward(days=wf_days, verbose=False)
                        if "error" not in results:
                            st.success(f"✅ Simulation complete! {results['summary']['total_trades']} trades, "
                                       f"P&L: ₹{results['summary']['total_return']:,.2f}")
                            st.rerun()
                        else:
                            st.error(f"Simulation failed: {results.get('error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.caption(f"Last run: {wf_data.get('simulation_date', 'N/A')}")
    else:
        st.info("No walk-forward results yet. Click 'Run Walk-Forward' below to simulate.")
        if st.button("▶️ Run Walk-Forward Simulation"):
            with st.spinner("Running walk-forward simulation..."):
                try:
                    sys.path.insert(0, os.path.join(BASE, "src"))
                    from walk_forward import run_walk_forward
                    results = run_walk_forward(days=60, verbose=False)
                    if "error" not in results:
                        st.success(f"✅ Done! {results['summary']['total_trades']} trades")
                        st.rerun()
                    else:
                        st.error(f"Failed: {results.get('error')}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # ── B) Model Drift Detector ──
    st.markdown("### 🔬 Model Drift Detector")
    drift_data = load_json(FILE_DRIFT)
    if drift_data:
        models_info = drift_data.get("models", {})
        alerts = drift_data.get("alerts", [])
        retrain = drift_data.get("retrain_needed", [])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Models Analysed", str(len(models_info)))
        with c2:
            st.metric("Predictions Logged", str(drift_data.get("total_predictions", 0)))
        with c3:
            if retrain:
                st.metric("⚠️ Retrain Needed", str(len(retrain)))
            else:
                st.metric("✅ Status", "All Stable")

        if alerts:
            st.markdown("#### 🚨 Drift Alerts")
            for a in alerts:
                if "SEVERE" in a:
                    st.error(a)
                else:
                    st.warning(a)

        if models_info:
            st.markdown("#### Model Health")
            drift_rows = []
            for model, info in models_info.items():
                status = info.get("status", "?")
                icon = {"STABLE": "✅", "MILD_DRIFT": "⚠️", "SEVERE_DRIFT": "🚨"}.get(status, "❓")
                drift_rows.append({
                    "Model": model,
                    "Status": f"{icon} {status}",
                    "KL Divergence": f"{info.get('kl_divergence', 0):.4f}",
                    "PSI": f"{info.get('psi', 0):.4f}",
                    "Baseline Conf": f"{info.get('baseline_mean_conf', 0):.4f}",
                    "Recent Conf": f"{info.get('recent_mean_conf', 0):.4f}",
                    "Predictions": info.get("predictions", 0),
                })
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

            # Confidence trend chart
            try:
                import plotly.graph_objects as go
                fig_drift = go.Figure()
                for model, info in models_info.items():
                    conf_trend = info.get("confidence_trend", {})
                    baseline = info.get("baseline_mean_conf", 0)
                    recent = info.get("recent_mean_conf", 0)
                    fig_drift.add_trace(go.Bar(
                        name=model,
                        x=["Baseline Conf", "Recent Conf"],
                        y=[baseline, recent],
                    ))
                fig_drift.update_layout(
                    title="Model Confidence: Baseline vs Recent",
                    barmode="group", height=350, template="plotly_dark",
                )
                st.plotly_chart(fig_drift, use_container_width=True)
            except Exception:
                pass

        # Run drift analysis button
        if st.button("🔬 Run Drift Analysis Now"):
            try:
                sys.path.insert(0, os.path.join(BASE, "src"))
                from model_drift import analyse_model_drift as run_drift
                result = run_drift()
                if result.get("retrain_needed"):
                    st.warning(f"⚠️ Retrain needed: {', '.join(result['retrain_needed'])}")
                else:
                    st.success("✅ All models stable")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        st.caption(f"Last analysis: {drift_data.get('timestamp', 'N/A')}")
    else:
        st.info("No drift data yet. Predictions are logged during live trading.")
        if st.button("🔬 Run Drift Analysis"):
            try:
                sys.path.insert(0, os.path.join(BASE, "src"))
                from model_drift import analyse_model_drift as run_drift
                run_drift()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════
#  TAB 13 — RISK PARITY & A/B BACKTEST
# ══════════════════════════════════════════════════
if page == "Risk Parity & A/B":
    st.markdown('<div class="section-header">⚖️ <h3>Risk Parity Allocator & A/B Backtest</h3></div>', unsafe_allow_html=True)

    # ── A) Risk Parity ──
    st.markdown("### ⚖️ Risk Parity Allocation")
    rp_data = load_json(FILE_RP)
    if rp_data:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Parity Score", f"{rp_data.get('parity_score', 0):.0f}/100")
        with c2:
            st.metric("Portfolio Vol", f"{rp_data.get('total_portfolio_vol', 0):.4f}")
        with c3:
            st.metric("Stocks", len(rp_data.get("symbols", [])))
        with c4:
            st.metric("Capital", f"₹{rp_data.get('capital', 0):,.0f}")

        # Weights table
        weights = rp_data.get("weights", {})
        vols = rp_data.get("volatilities", {})
        allocs = rp_data.get("allocations", {})
        risk_contr = rp_data.get("risk_contributions", {})

        if weights:
            rows = []
            for sym in rp_data.get("symbols", []):
                rows.append({
                    "Stock": sym.replace(".NS", ""),
                    "Weight %": round(weights.get(sym, 0) * 100, 2),
                    "Allocation ₹": allocs.get(sym, 0),
                    "Volatility %": round(vols.get(sym, 0) * 100, 2),
                    "Risk Contribution %": round(risk_contr.get(sym, 0) * 100, 2),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            try:
                import plotly.express as px
                df_rp = pd.DataFrame(rows)
                fig = px.bar(df_rp, x="Stock", y=["Weight %", "Risk Contribution %"],
                             barmode="group", title="Risk Parity: Weight vs Risk Contribution")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        st.caption(f"Updated: {rp_data.get('timestamp', 'N/A')}")
        st.info(rp_data.get("summary", ""))
    else:
        st.info("Risk parity data not available yet. Will populate on next Sniper run.")

    st.divider()

    # ── B) A/B Backtest Results ──
    st.markdown("### 🧪 A/B Strategy Comparison")
    ab_data = load_json(FILE_AB)
    if ab_data:
        var_a = ab_data.get("variant_a", {})
        var_b = ab_data.get("variant_b", {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Strategy A: {var_a.get('name', 'Baseline')}**")
            st.metric("Return", f"{var_a.get('total_return_pct', 0):+.2f}%")
            st.metric("Sharpe", f"{var_a.get('sharpe', 0):.3f}")
            st.metric("Win Rate", f"{var_a.get('win_rate', 0):.1f}%")
            st.metric("Max Drawdown", f"{var_a.get('max_drawdown', 0):.2f}%")
            st.metric("Profit Factor", f"{var_a.get('profit_factor', 0):.2f}")
        with c2:
            st.markdown(f"**Strategy B: {var_b.get('name', 'Modified')}**")
            st.metric("Return", f"{var_b.get('total_return_pct', 0):+.2f}%")
            st.metric("Sharpe", f"{var_b.get('sharpe', 0):.3f}")
            st.metric("Win Rate", f"{var_b.get('win_rate', 0):.1f}%")
            st.metric("Max Drawdown", f"{var_b.get('max_drawdown', 0):.2f}%")
            st.metric("Profit Factor", f"{var_b.get('profit_factor', 0):.2f}")

        # Statistical tests
        tests = ab_data.get("statistical_tests", {})
        ttest = tests.get("paired_ttest", {})
        sharpe_t = tests.get("sharpe_test", {})

        st.markdown("#### 📊 Statistical Significance")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            sig = "✅ Yes" if ttest.get("significant") else "❌ No"
            st.metric("T-Test Significant?", sig)
            st.caption(f"p-value: {ttest.get('p_value', 'N/A')}")
        with tc2:
            st.metric("T-Statistic", f"{ttest.get('t_stat', 0):.4f}")
            st.caption(f"n = {ttest.get('n', 0)} observations")
        with tc3:
            st.metric("Winner", ab_data.get("winner", "TIE"))
            scores = ab_data.get("scores", {})
            st.caption(f"Score: {scores.get('a', 0)}-{scores.get('b', 0)} / {scores.get('total', 5)}")

        # Equity curves
        eq_a = var_a.get("equity_curve", [])
        eq_b = var_b.get("equity_curve", [])
        if eq_a and eq_b:
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=eq_a, name=var_a.get("name", "A"), line=dict(color="#00d4ff")))
                fig.add_trace(go.Scatter(y=eq_b, name=var_b.get("name", "B"), line=dict(color="#ff6b6b")))
                fig.update_layout(title="A/B Equity Curves", yaxis_title="Equity ₹", template="plotly_dark",
                                  height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        st.info(ab_data.get("summary", ""))
        st.caption(f"Last run: {ab_data.get('timestamp', 'N/A')}")
    else:
        st.info("No A/B backtest results yet. Run `python src/ab_backtest.py` to generate.")

    # Run A/B from trade history button
    if st.button("🧪 Run A/B Test (First Half vs Second Half)"):
        try:
            sys.path.insert(0, os.path.join(BASE, "src"))
            from ab_backtest import run_ab_from_csv, save_ab_results
            result = run_ab_from_csv(FILE_TRADES)
            if "error" not in result:
                save_ab_results(result)
                st.success(f"✅ A/B test complete! Winner: {result.get('winner', 'TIE')}")
                st.rerun()
            else:
                st.error(result["error"])
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════
#  TAB 14 — NEWS EVENTS & PORTFOLIO REBALANCER
# ══════════════════════════════════════════════════
if page == "News & Rebalancer":
    st.markdown('<div class="section-header">📰 <h3>News Events & Portfolio Rebalancer</h3></div>', unsafe_allow_html=True)

    # ── A) News Event Detector ──
    st.markdown("### 📰 News Event Detector (NLP)")
    news_data = load_json(FILE_NEWS)
    if news_data:
        summary = news_data.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🚫 Blocked", summary.get("blocked", 0))
        with c2:
            st.metric("🚀 Boosted", summary.get("boosted", 0))
        with c3:
            st.metric("🔧 Adjusted", summary.get("adjusted", 0))
        with c4:
            st.metric("✅ Clear", summary.get("clear", 0))

        # Per-stock news events
        stocks_news = news_data.get("stocks", {})
        for sym, data in stocks_news.items():
            events = data.get("events", [])
            action = data.get("action", "ALLOW")
            headlines = data.get("headlines_found", 0)

            icon = "🚫" if action == "BLOCK" else "🚀" if action == "BOOST" else "✅"
            with st.expander(f"{icon} {sym.replace('.NS', '')} — {action} ({headlines} headlines)"):
                if events:
                    for evt in events:
                        st.markdown(f"  {evt.get('icon', '•')} **{evt.get('event_type', '')}** "
                                    f"[{evt.get('severity', '')}]: {evt.get('headline', '')[:100]}")
                else:
                    st.success("No significant events detected")

                # Show recent headlines
                for h in data.get("headlines", [])[:5]:
                    st.caption(f"📄 {h.get('title', '')[:100]} — _{h.get('source', '')}_")

        st.caption(f"Updated: {news_data.get('timestamp', 'N/A')}")
    else:
        st.info("News data not available yet. Will populate on next Sniper run.")

    st.divider()

    # ── B) Options Hedge Status ──
    st.markdown("### 🛡️ Options Hedging Layer")
    hedge_data = load_json(FILE_HEDGE)
    if hedge_data:
        exposure = hedge_data.get("exposure", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Portfolio Exposure", f"{exposure.get('exposure_pct', 0):.1f}%")
        with c2:
            needed = "Yes ⚠️" if hedge_data.get("hedge_needed") else "No ✅"
            st.metric("Hedge Needed?", needed)
        with c3:
            st.metric("Est. Hedge Cost", f"₹{hedge_data.get('total_hedge_cost', 0):,.2f}")

        recs = hedge_data.get("recommendations", [])
        if recs:
            st.markdown("#### Hedge Recommendations")
            rows = []
            for r in recs:
                rows.append({
                    "Stock": r.get("symbol", "").replace(".NS", ""),
                    "Spot": f"₹{r.get('spot_price', 0):,.2f}",
                    "Strike": f"₹{r.get('strike', 0):,.2f}",
                    "Put Premium": f"₹{r.get('put_premium', 0):,.2f}",
                    "Lots": r.get("lots_needed", 0),
                    "Total Cost": f"₹{r.get('total_cost', 0):,.2f}",
                    "Delta": r.get("delta", 0),
                    "Protection": f"{r.get('protection_pct', 0):.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.info(hedge_data.get("summary", ""))
    else:
        st.info("Hedge analysis will appear when portfolio exposure exceeds threshold.")

    st.divider()

    # ── C) Portfolio Rebalancer ──
    st.markdown("### 🔄 Portfolio Rebalancer")
    rebal_data = load_json(FILE_REBALANCE)
    if rebal_data:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Orders", rebal_data.get("num_orders", 0))
        with c2:
            st.metric("Turnover", f"₹{rebal_data.get('turnover', 0):,.2f}")
        with c3:
            st.metric("Turnover %", f"{rebal_data.get('turnover_pct', 0):.1f}%")
        with c4:
            current = rebal_data.get("current_allocation", {})
            st.metric("Cash", f"{current.get('cash_pct', 0):.1f}%")

        # Target weights
        targets = rebal_data.get("target_weights", {})
        if targets:
            st.markdown("#### Target Allocation")
            rows = []
            for sym, wt in sorted(targets.items(), key=lambda x: x[1], reverse=True):
                if sym == "_cash":
                    rows.append({"Asset": "💵 Cash", "Target %": wt})
                else:
                    cur_w = rebal_data.get("current_allocation", {}).get("stock_weights", {}).get(sym, 0)
                    rows.append({
                        "Asset": sym.replace(".NS", ""),
                        "Target %": wt,
                        "Current %": round(cur_w, 2),
                        "Drift": round(wt - cur_w, 2),
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Rebalance orders
        orders = rebal_data.get("orders", [])
        if orders:
            st.markdown("#### Rebalance Orders")
            for o in orders:
                icon = "🟢" if o["action"] == "BUY" else "🔴"
                st.markdown(f"  {icon} **{o['action']}** {o['symbol'].replace('.NS', '')} "
                            f"× {o['qty']} @ ₹{o['price']:,.2f} ({o['drift']:+.1f}%)")

        st.info(rebal_data.get("summary", ""))
        st.caption(f"Last rebalance: {rebal_data.get('timestamp', 'N/A')}")
    else:
        st.info("Rebalance data not available yet. Runs weekly (Monday after market open).")

    # ── D) Multi-Timeframe Status ──
    st.divider()
    st.markdown("### 📊 Multi-Timeframe Consensus")
    mtf_data = load_json(FILE_MTF)
    if mtf_data:
        stocks_mtf = mtf_data.get("stocks", {})
        if stocks_mtf:
            rows = []
            for sym, data in stocks_mtf.items():
                tfs = data.get("timeframes", {})
                rows.append({
                    "Stock": sym.replace(".NS", ""),
                    "Daily": tfs.get("daily", {}).get("direction", "?"),
                    "Weekly": tfs.get("weekly", {}).get("direction", "?"),
                    "Monthly": tfs.get("monthly", {}).get("direction", "?"),
                    "Consensus": data.get("consensus", "?"),
                    "Conviction": f"{data.get('conviction', 0):.2f}",
                    "Trade OK": "✅" if data.get("should_trade") else "❌",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Updated: {mtf_data.get('timestamp', 'N/A')}")
    else:
        st.info("Multi-timeframe data will appear after next Sniper scan.")


# ══════════════════════════════════════════════════
#  TAB 15 — REGIME DETECTION & VaR / STRESS TESTING
# ══════════════════════════════════════════════════
if page == "Regime & VaR":
    st.markdown('<div class="section-header"><h3>🎯 Market Regime Detection (HMM)</h3></div>', unsafe_allow_html=True)

    regime_data = load_json(FILE_REGIME)
    if regime_data:
        regime = regime_data.get("regime", "SIDEWAYS")
        regime_icons = {"BULL": "🐂", "BEAR": "🐻", "SIDEWAYS": "➡️"}
        regime_colors = {"BULL": "green", "BEAR": "red", "SIDEWAYS": "orange"}

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Current Regime", f"{regime_icons.get(regime, '')} {regime}")
        with c2:
            st.metric("Confidence", f"{regime_data.get('confidence', 0):.0%}")
        with c3:
            st.metric("Sizing Multiplier", f"{regime_data.get('sizing_multiplier', 1.0):.0%}")
        with c4:
            override = regime_data.get("personality_override") or "Default"
            st.metric("Brain Override", override)

        # State probabilities
        probs = regime_data.get("state_probs", {})
        if probs:
            st.markdown("#### State Probabilities")
            try:
                import plotly.graph_objects as go
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        marker_color=["#ef4444", "#f59e0b", "#22c55e"],
                        text=[f"{v:.1%}" for v in probs.values()],
                        textposition="auto",
                    )
                ])
                fig.update_layout(height=250, margin=dict(t=20, b=20), yaxis_title="Probability")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                for state, prob in probs.items():
                    st.write(f"  {state}: {prob:.1%}")

        # Regime history timeline
        history = regime_data.get("regime_history", [])
        if history:
            st.markdown("#### Regime History (Last 60 Days)")
            hist_colors = {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡"}
            # Show as compacted string
            display = " ".join(hist_colors.get(r, "⚪") for r in history[-60:])
            st.markdown(f"<small>{display}</small>", unsafe_allow_html=True)
            st.caption("🟢 Bull · 🔴 Bear · 🟡 Sideways")

        st.caption(f"Updated: {regime_data.get('timestamp', 'N/A')} | Observations: {regime_data.get('observations', 0)}")
    else:
        st.info("Regime data not available yet. Will populate on next Sniper run.")

    st.divider()

    # ── VaR & Stress Testing ──
    st.markdown('<div class="section-header"><h3>📉 VaR & Stress Testing</h3></div>', unsafe_allow_html=True)

    var_data = load_json(FILE_VAR)
    if var_data:
        c1, c2, c3 = st.columns(3)
        with c1:
            risk_lvl = var_data.get("risk_level", "UNKNOWN")
            risk_clr = "🟢" if risk_lvl == "LOW" else ("🟡" if risk_lvl == "MODERATE" else "🔴")
            st.metric("Risk Level", f"{risk_clr} {risk_lvl}")
        with c2:
            st.metric("Risk Score", f"{var_data.get('risk_score', 0)}/100")
        with c3:
            mc_var = var_data.get("monte_carlo_var_95", {})
            st.metric("MC VaR (95%)", f"₹{mc_var.get('var_rupees', 0):,.2f}")

        # VaR comparison table
        st.markdown("#### Value-at-Risk Comparison")
        var_rows = []
        for key, label in [("parametric_var_95", "Parametric 95%"), ("historical_var_95", "Historical 95%"),
                           ("monte_carlo_var_95", "Monte Carlo 95%"), ("parametric_var_99", "Parametric 99%"),
                           ("historical_var_99", "Historical 99%"), ("monte_carlo_var_99", "Monte Carlo 99%")]:
            v = var_data.get(key, {})
            if v:
                var_rows.append({
                    "Method": label,
                    "VaR %": f"{v.get('var_pct', 0):.2f}%",
                    "VaR ₹": f"₹{v.get('var_rupees', 0):,.2f}",
                    "CVaR %": f"{v.get('cvar_pct', 0):.2f}%" if "cvar_pct" in v else "N/A",
                })
        if var_rows:
            st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

        # Stress tests
        stress = var_data.get("stress_tests", [])
        if stress:
            st.markdown("#### Stress Test Scenarios")
            stress_rows = []
            for s in stress:
                sev_icon = "🔴" if s.get("severity") == "EXTREME" else ("🟡" if s.get("severity") == "HIGH" else "🟢")
                stress_rows.append({
                    "Scenario": s.get("name", ""),
                    "Shock": f"{s.get('equity_shock_pct', 0):+.1f}%",
                    "Est. Loss": f"₹{s.get('estimated_loss', 0):,.2f}",
                    "After": f"₹{s.get('portfolio_after', 0):,.2f}",
                    "Severity": f"{sev_icon} {s.get('severity', '')}",
                })
            st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

        # Drawdowns
        dd = var_data.get("drawdowns", {})
        if dd:
            st.markdown("#### Drawdown Analysis")
            dc1, dc2, dc3, dc4 = st.columns(4)
            with dc1:
                st.metric("Max Drawdown", f"{dd.get('max_drawdown_pct', 0):.2f}%")
            with dc2:
                st.metric("Current DD", f"{dd.get('current_drawdown_pct', 0):.2f}%")
            with dc3:
                st.metric("Longest DD", f"{dd.get('longest_drawdown_days', 0)} days")
            with dc4:
                st.metric("Avg DD", f"{dd.get('avg_drawdown_pct', 0):.2f}%")

        st.caption(f"Updated: {var_data.get('timestamp', 'N/A')}")
    else:
        st.info("VaR & stress data not available yet. Will populate on next Sniper run.")

    st.divider()

    # ── FinBERT Sentiment ──
    st.markdown('<div class="section-header"><h3>🧠 FinBERT Sentiment Analysis</h3></div>', unsafe_allow_html=True)

    fb_data = load_json(FILE_FINBERT)
    if fb_data:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            mkt_label = fb_data.get("market_label", "NEUTRAL")
            mkt_icon = "🟢" if mkt_label == "BULLISH" else ("🔴" if mkt_label == "BEARISH" else "⚪")
            st.metric("Market Sentiment", f"{mkt_icon} {mkt_label}")
        with c2:
            st.metric("Sentiment Score", f"{fb_data.get('market_sentiment', 0):+.3f}")
        with c3:
            st.metric("Method", fb_data.get("method", "keyword").upper())
        with c4:
            st.metric("Headlines", fb_data.get("total_headlines", 0))

        stocks_fb = fb_data.get("stocks", {})
        if stocks_fb:
            rows = []
            for sym, data in stocks_fb.items():
                if isinstance(data, dict) and "error" not in data:
                    rows.append({
                        "Stock": sym.replace(".NS", ""),
                        "Sentiment": data.get("overall", "neutral").title(),
                        "🟢 Pos": data.get("positive", 0),
                        "🔴 Neg": data.get("negative", 0),
                        "⚪ Neu": data.get("neutral", 0),
                        "Score": f"{data.get('avg_score', 0):+.3f}",
                        "Headlines": data.get("headline_count", 0),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(f"Updated: {fb_data.get('timestamp', 'N/A')}")
    else:
        st.info("FinBERT data not available yet. Install `transformers torch` for best results.")


# ══════════════════════════════════════════════════
#  TAB 16 — VOLUME PROFILE & OPTIONS CHAIN
# ══════════════════════════════════════════════════
if page == "Volume & Options":
    st.markdown('<div class="section-header"><h3>📊 Volume Profile & Order Flow</h3></div>', unsafe_allow_html=True)

    vp_data = load_json(FILE_VP)
    if vp_data:
        mkt_p = vp_data.get("market_pressure", "NEUTRAL")
        p_icon = "🟢" if mkt_p == "BUY" else ("🔴" if mkt_p == "SELL" else "⚪")
        c1, c2    = st.columns(2)
        with c1:
            st.metric("Market Pressure", f"{p_icon} {mkt_p}")
        with c2:
            st.metric("Buy vs Sell",
                      f"{vp_data.get('buy_pressure_count', 0)} buy / {vp_data.get('sell_pressure_count', 0)} sell")

        stocks_vp = vp_data.get("stocks", {})
        if stocks_vp:
            rows = []
            for sym, data in stocks_vp.items():
                if isinstance(data, dict) and "error" not in data:
                    vwap_info = data.get("vwap", {})
                    delta_info = data.get("volume_delta", {})
                    profile = data.get("volume_profile", {})
                    rows.append({
                        "Stock": sym.replace(".NS", ""),
                        "Price": f"₹{data.get('current_price', 0):,.2f}",
                        "VWAP": f"₹{vwap_info.get('vwap', 0):,.2f}",
                        "vs VWAP": "↑" if data.get("above_vwap") else "↓",
                        "VWAP Dist": f"{data.get('vwap_distance_pct', 0):+.2f}%",
                        "POC": f"₹{profile.get('poc', 0):,.2f}",
                        "Pressure": delta_info.get("pressure", "?"),
                        "Strength": f"{delta_info.get('strength_pct', 0):.1f}%",
                        "In VA": "✅" if data.get("in_value_area") else "❌",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Volume profile chart for selected stock
            vp_syms = list(stocks_vp.keys())
            if vp_syms:
                sel = st.selectbox("Volume profile detail", vp_syms, format_func=lambda x: x.replace(".NS", ""), key="vp_sel")
                sel_data = stocks_vp.get(sel, {})
                hist = sel_data.get("volume_profile", {}).get("histogram", [])
                if hist:
                    try:
                        import plotly.graph_objects as go
                        prices = [h["price"] for h in hist]
                        vols = [h["volume"] for h in hist]
                        colors = ["#22c55e" if h.get("is_poc") else ("#3b82f6" if h.get("in_value_area") else "#6b7280") for h in hist]
                        fig = go.Figure(data=go.Bar(y=prices, x=vols, orientation="h", marker_color=colors))
                        fig.update_layout(height=350, margin=dict(t=20, b=20),
                                          xaxis_title="Volume", yaxis_title="Price ₹",
                                          title=f"{sel.replace('.NS', '')} Volume Profile")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.write("Install plotly for volume profile chart.")

        st.caption(f"Updated: {vp_data.get('timestamp', 'N/A')}")
    else:
        st.info("Volume profile data not available yet. Will populate on next Sniper run.")

    st.divider()

    # ── Option Chain ──
    st.markdown('<div class="section-header"><h3>📜 Live Option Chain</h3></div>', unsafe_allow_html=True)

    oc_data = load_json(FILE_OC)
    if oc_data:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("NSE Live Chains", oc_data.get("nse_live", 0))
        with c2:
            st.metric("BS Fallback", oc_data.get("bs_fallback", 0))
        with c3:
            st.metric("Total", oc_data.get("total", 0))

        chains = oc_data.get("chains", {})
        if chains:
            rows = []
            for sym, chain in chains.items():
                if isinstance(chain, dict):
                    rows.append({
                        "Stock": sym.replace(".NS", ""),
                        "Spot": f"₹{chain.get('spot', 0):,.2f}",
                        "PCR": f"{chain.get('pcr', 0):.2f}",
                        "Sentiment": chain.get("sentiment", "?"),
                        "Support": f"₹{chain.get('support', chain.get('max_pe_oi_strike', 0)):,.0f}",
                        "Resistance": f"₹{chain.get('resistance', chain.get('max_ce_oi_strike', 0)):,.0f}",
                        "Source": chain.get("source", "?"),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(f"Updated: {oc_data.get('timestamp', 'N/A')}")
    else:
        st.info("Option chain data not available yet. Will populate on next Sniper run.")


# ══════════════════════════════════════════════════
#  TAB 17 — EXECUTION QUALITY & TRADE JOURNAL
# ══════════════════════════════════════════════════
if page == "Execution & Journal":
    st.markdown('<div class="section-header"><h3>🎯 Execution Quality Analytics</h3></div>', unsafe_allow_html=True)

    eq_data = load_json(FILE_EQ)
    if eq_data:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            grade = eq_data.get("overall_grade", "N/A")
            grade_clr = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🔴"}
            st.metric("Overall Grade", f"{grade_clr.get(grade, '⚪')} {grade}")
        with c2:
            st.metric("Avg Timing Score", f"{eq_data.get('avg_timing_score', 0):.0f}/100")
        with c3:
            st.metric("Avg Slippage", f"{eq_data.get('avg_slippage_bps', 0):.1f} bps")
        with c4:
            st.metric("Slippage Cost", f"₹{eq_data.get('total_slippage_cost', 0):,.2f}")

        # Grade distribution
        grades = eq_data.get("grade_distribution", {})
        if grades:
            st.markdown("#### Grade Distribution")
            try:
                import plotly.graph_objects as go
                labels = list(grades.keys())
                values = list(grades.values())
                colors = ["#22c55e", "#86efac", "#fbbf24", "#ef4444", "#991b1b", "#6b7280"]
                fig = go.Figure(data=go.Pie(labels=labels, values=values, marker_colors=colors[:len(labels)]))
                fig.update_layout(height=250, margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                for g, cnt in grades.items():
                    st.write(f"  Grade {g}: {cnt}")

        # Improvement tip
        tip = eq_data.get("improvement_tip", "")
        if tip:
            st.success(f"💡 **Tip:** {tip}")

        # Trade details
        details = eq_data.get("trade_details", [])
        if details:
            st.markdown("#### Recent Trade Execution")
            rows = []
            for d in details:
                if isinstance(d, dict) and d.get("grade", "N/A") != "N/A":
                    rows.append({
                        "Stock": d.get("symbol", "").replace(".NS", ""),
                        "Entry": f"₹{d.get('entry_price', 0):,.2f}",
                        "VWAP": f"₹{d.get('vwap_estimate', 0):,.2f}",
                        "Day Low": f"₹{d.get('day_low', 0):,.2f}",
                        "Timing": f"{d.get('timing_score', 0):.0f}/100",
                        "Slip bps": f"{d.get('slippage_bps', 0):.1f}",
                        "Grade": d.get("grade", "?"),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(f"Updated: {eq_data.get('timestamp', 'N/A')} | Trades: {eq_data.get('trades_analysed', 0)}")
    else:
        st.info("Execution quality data will appear after trading activity.")

    st.divider()

    # ── Trade Journal ──
    st.markdown('<div class="section-header"><h3>📝 Auto Trade Journal</h3></div>', unsafe_allow_html=True)

    journal_data = load_json(FILE_JOURNAL)
    if journal_data:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Journal Period", journal_data.get("period", "N/A"))
        with c2:
            st.metric("Total Trades", journal_data.get("total_trades", 0))
        with c3:
            st.metric("Net P&L", f"₹{journal_data.get('total_pnl', 0):,.2f}")

        # Summary stats
        stats = journal_data.get("stats", {})
        if stats:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
            with c2:
                st.metric("Avg Win", f"₹{stats.get('avg_win', 0):,.2f}")
            with c3:
                st.metric("Avg Loss", f"₹{stats.get('avg_loss', 0):,.2f}")
            with c4:
                st.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")

        # Per-stock breakdown
        stocks_j = journal_data.get("per_stock", {})
        if stocks_j:
            st.markdown("#### Per-Stock Performance")
            rows = []
            for sym, data in stocks_j.items():
                if isinstance(data, dict):
                    rows.append({
                        "Stock": sym.replace(".NS", ""),
                        "Trades": data.get("trades", 0),
                        "Wins": data.get("wins", 0),
                        "Losses": data.get("losses", 0),
                        "P&L": f"₹{data.get('pnl', 0):,.2f}",
                        "Win Rate": f"{data.get('win_rate', 0):.0f}%",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Journal file link
        journal_path = journal_data.get("file_path", "")
        if journal_path:
            st.info(f"Full journal: `{journal_path}`")

        st.caption(f"Generated: {journal_data.get('timestamp', 'N/A')}")
    else:
        st.info("Trade journal will be generated automatically at end of day.")

    st.divider()

    # ── Smart Alerts Summary ──
    st.markdown('<div class="section-header"><h3>📣 Smart Alerts</h3></div>', unsafe_allow_html=True)

    alert_smart = load_json(FILE_SMART_ALERT)
    if alert_smart:
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.metric("🔴 Critical", alert_smart.get("critical_count", 0))
        with ac2:
            st.metric("🟡 Warning", alert_smart.get("warning_count", 0))
        with ac3:
            st.metric("🟢 Info", alert_smart.get("info_count", 0))

        recent = alert_smart.get("recent_alerts", [])
        if recent:
            for a in recent[-10:]:
                tier = a.get("tier", "info")
                icon = "🔴" if tier == "critical" else ("🟡" if tier == "warning" else "🟢")
                st.markdown(f"  {icon} **[{a.get('time', '')}]** {a.get('message', '')}")
    else:
        st.info("Smart alerts will populate during live trading.")


# ══════════════════════════════════════════════════
#  TAB 18 — RL SIZER & BAYESIAN FUSION
# ══════════════════════════════════════════════════
if page == "RL Sizer & Bayesian":
    st.markdown('<div class="section-header"><h3>🎲 RL Position Sizer & Bayesian Fusion</h3></div>', unsafe_allow_html=True)

    col_rl, col_bay = st.columns(2)

    with col_rl:
        st.subheader("Q-Learning Sizer")
        rl_data = load_json(FILE_RL_SIZER)
        if rl_data:
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("Episodes", rl_data.get("total_episodes", 0))
            with rc2:
                st.metric("States", rl_data.get("states_explored", 0))
            with rc3:
                st.metric("Epsilon", f"{rl_data.get('epsilon', 0):.3f}")
            with rc4:
                st.metric("Method", rl_data.get("last_method", "—"))

            # Q-value distribution
            qtable = load_json(FILE_RL_QTABLE)
            if qtable:
                st.caption(f"Q-table size: {len(qtable)} state entries")
                # Show top-5 states by max Q-value
                top_states = sorted(
                    qtable.items(),
                    key=lambda x: max(x[1]) if isinstance(x[1], list) else 0,
                    reverse=True
                )[:5]
                if top_states:
                    rows = []
                    for skey, qvals in top_states:
                        if isinstance(qvals, list):
                            best_action = qvals.index(max(qvals))
                            action_pcts = [2, 5, 8, 12, 18, 25]
                            rows.append({
                                "State": skey[:40],
                                "Best Action": f"{action_pcts[best_action] if best_action < len(action_pcts) else '?'}%",
                                "Max Q": f"{max(qvals):.3f}",
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

            last_action = rl_data.get("last_action", {})
            if last_action:
                st.markdown(f"**Last sizing:** {last_action.get('pct', 0):.1f}% "
                            f"(₹{last_action.get('amount', 0):,.0f})")
        else:
            st.info("RL sizer data will appear after trades are executed.")

    with col_bay:
        st.subheader("Bayesian Fusion")
        bay_data = load_json(FILE_BAYESIAN)
        if bay_data:
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Fused Score", f"{bay_data.get('fused_score', 0):.3f}")
            with bc2:
                st.metric("Action", bay_data.get("action", "—"))
            with bc3:
                st.metric("Sources", bay_data.get("n_sources", 0))

            # Source weights
            sources = bay_data.get("source_weights", {})
            if sources:
                src_df = pd.DataFrame([
                    {"Source": k, "Weight": round(v, 4)}
                    for k, v in sorted(sources.items(), key=lambda x: -x[1])
                ])
                st.dataframe(src_df, use_container_width=True)
        else:
            st.info("Bayesian fusion data will appear after signals are processed.")

        # Bayesian priors
        priors = load_json(FILE_BAYES_PRIOR)
        if priors:
            st.caption("Posterior Parameters (α, β)")
            prior_rows = []
            for src, vals in priors.items():
                if isinstance(vals, dict):
                    prior_rows.append({
                        "Source": src,
                        "Alpha": round(vals.get("alpha", 1), 2),
                        "Beta": round(vals.get("beta", 1), 2),
                        "Accuracy": f"{vals.get('alpha', 1) / (vals.get('alpha', 1) + vals.get('beta', 1)):.1%}",
                    })
            if prior_rows:
                st.dataframe(pd.DataFrame(prior_rows), use_container_width=True)


# ══════════════════════════════════════════════════
#  TAB 19 — INTERMARKET & LIQUIDITY
# ══════════════════════════════════════════════════
if page == "Intermarket & Liquidity":
    st.markdown('<div class="section-header"><h3>🌍 Intermarket Engine & Liquidity Filter</h3></div>', unsafe_allow_html=True)

    col_im, col_lq = st.columns(2)

    with col_im:
        st.subheader("Intermarket Correlations")
        im_data = load_json(FILE_INTERMARKET)
        if im_data:
            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                st.metric("Exposure Mult", f"{im_data.get('exposure_multiplier', 1.0):.2f}")
            with ic2:
                st.metric("Status", im_data.get("status", "—"))
            with ic3:
                st.metric("Signals", im_data.get("n_signals", 0))

            # Instrument z-scores
            instruments = im_data.get("instruments", {})
            if instruments:
                inst_rows = []
                for name, vals in instruments.items():
                    if isinstance(vals, dict):
                        inst_rows.append({
                            "Instrument": name,
                            "Price": f"{vals.get('price', 0):,.2f}",
                            "Z-Score": f"{vals.get('z_score', 0):+.2f}",
                            "Trend": vals.get("trend", "—"),
                        })
                if inst_rows:
                    st.dataframe(pd.DataFrame(inst_rows), use_container_width=True)

            # Active rules
            rules = im_data.get("active_rules", [])
            if rules:
                st.markdown("**Active Rules:**")
                for r in rules:
                    st.markdown(f"  - {r}")
        else:
            st.info("Intermarket data will appear during live trading.")

    with col_lq:
        st.subheader("Liquidity Scores")
        lq_data = load_json(FILE_LIQUIDITY)
        if lq_data:
            lq_rows = []
            for sym, vals in lq_data.items():
                if isinstance(vals, dict):
                    score = vals.get("score", vals.get("composite_score", 0))
                    ok = vals.get("ok", True)
                    reason = vals.get("reason", "")
                    status_icon = "✅" if ok else "❌"
                    lq_rows.append({
                        "Stock": sym.replace(".NS", ""),
                        "Score": round(score, 1),
                        "Gate": status_icon,
                        "Reason": reason[:50],
                    })
            if lq_rows:
                lq_df = pd.DataFrame(lq_rows).sort_values("Score", ascending=False)
                st.dataframe(lq_df, use_container_width=True)
        else:
            st.info("Liquidity data will appear during live trading.")


# ══════════════════════════════════════════════════
#  TAB 20 — GREEKS HEAT MAP
# ══════════════════════════════════════════════════
if page == "Greeks Heat Map":
    st.markdown('<div class="section-header"><h3>📊 Portfolio Greeks Heat Map</h3></div>', unsafe_allow_html=True)

    greeks_data = load_json(FILE_GREEKS)
    if greeks_data:
        totals = greeks_data.get("totals", {})
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            st.metric("Portfolio Delta", f"{totals.get('delta', 0):.3f}")
        with gc2:
            st.metric("Portfolio Gamma", f"{totals.get('gamma', 0):.4f}")
        with gc3:
            st.metric("Portfolio Theta", f"₹{totals.get('theta', 0):.2f}/day")
        with gc4:
            st.metric("Portfolio Vega", f"₹{totals.get('vega', 0):.2f}")

        # Traffic lights
        alerts = greeks_data.get("alerts", {})
        if alerts:
            al_cols = st.columns(len(alerts))
            for i, (greek, level) in enumerate(alerts.items()):
                color_map = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
                icon = color_map.get(level, "⚪")
                with al_cols[i]:
                    st.markdown(f"**{greek}**: {icon} {level}")

        # Per-stock Greeks table
        positions = greeks_data.get("positions", [])
        if positions:
            st.subheader("Per-Stock Greeks")
            pos_rows = []
            for p in positions:
                pos_rows.append({
                    "Stock": p.get("symbol", "").replace(".NS", ""),
                    "Qty": p.get("qty", 0),
                    "Delta": f"{p.get('delta', 0):.3f}",
                    "Gamma": f"{p.get('gamma', 0):.4f}",
                    "Theta": f"₹{p.get('theta', 0):.2f}",
                    "Vega": f"₹{p.get('vega', 0):.2f}",
                    "Premium": f"₹{p.get('premium', 0):,.0f}",
                })
            if pos_rows:
                st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)

        # Heat map matrix
        heatmap = greeks_data.get("heatmap_matrix", {})
        if heatmap and "z" in heatmap:
            try:
                import plotly.graph_objects as go
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap["z"],
                    x=heatmap.get("x", []),
                    y=heatmap.get("y", []),
                    colorscale="RdYlGn",
                    text=heatmap.get("text", []),
                    texttemplate="%{text}",
                ))
                fig.update_layout(
                    title="Greeks Heat Map (normalised)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Plotly import failed for heatmap")
    else:
        st.info("Greeks data will appear after positions are opened.")


# ══════════════════════════════════════════════════
#  TAB 21 — SCALPER & AUTO TUNER
# ══════════════════════════════════════════════════
if page == "Scalper & Tuner":
    st.markdown('<div class="section-header"><h3>⚡ Intraday Scalper & Auto Tuner</h3></div>', unsafe_allow_html=True)

    col_sc, col_tu = st.columns(2)

    with col_sc:
        st.subheader("Intraday VWAP Scalper")
        sc_data = load_json(FILE_SCALPER)
        if sc_data:
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Scalp Trades", sc_data.get("trades", 0))
            with sc2:
                pnl = sc_data.get("total_pnl", 0)
                st.metric("Scalp P&L", f"₹{pnl:,.2f}")
            with sc3:
                st.metric("Wins", sc_data.get("wins", 0))
            with sc4:
                st.metric("Losses", sc_data.get("losses", 0))

            details = sc_data.get("details", [])
            if details:
                st.caption("Recent Scalp Trades")
                sc_df = pd.DataFrame(details[-15:])
                st.dataframe(sc_df, use_container_width=True)
        else:
            st.info("Scalper data will appear during live market hours.")

    with col_tu:
        st.subheader("Auto Hyperparameter Tuner")
        tu_data = load_json(FILE_TUNER)
        if tu_data:
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric("Status", tu_data.get("status", "—"))
            with tc2:
                st.metric("Best Sharpe", f"{tu_data.get('best_sharpe', 0):.3f}")
            with tc3:
                st.metric("Trials", tu_data.get("n_trials", 0))

            best = tu_data.get("best_params", {})
            if best:
                st.markdown("**Best Parameters:**")
                bp_rows = [{"Parameter": k, "Value": round(v, 4) if isinstance(v, float) else v}
                           for k, v in best.items() if not k.startswith("_")]
                if bp_rows:
                    st.dataframe(pd.DataFrame(bp_rows), use_container_width=True)

            top5 = tu_data.get("top_5", [])
            if top5:
                st.caption("Top 5 Trials")
                for i, t in enumerate(top5):
                    st.markdown(f"  **#{i+1}** Sharpe: {t.get('sharpe', 0):.4f}")
        else:
            st.info("Tuner data will appear after a sweep is run (weekends).")


# ══════════════════════════════════════════════════
#  TAB 22 — TRANSFORMER & ANOMALY (Phase 11)
# ══════════════════════════════════════════════════
if page == "Transformer & Anomaly":
    col_tf, col_an = st.columns(2)

    with col_tf:
        st.subheader("Portfolio Transformer (Attention)")
        tf_data = load_json(FILE_TRANSFORMER)
        if tf_data:
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric("Method", tf_data.get("method", "—"))
            with tc2:
                st.metric("Total Stocks", tf_data.get("n_stocks", 0))
            with tc3:
                top_picks = tf_data.get("top_picks", [])
                st.metric("Top Picks", len(top_picks))

            preds = tf_data.get("predictions", {})
            if preds:
                st.markdown("**Per-Stock BUY Probability:**")
                pred_rows = [
                    {"Stock": sym, "Probability": f"{p:.3f}",
                     "Signal": "🟢 BUY" if p > 0.45 else "🔴 SKIP"}
                    for sym, p in preds.items()
                ]
                st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)

            if top_picks:
                st.success(f"**Top Picks:** {', '.join(top_picks)}")

            attn = tf_data.get("attention_scores", {})
            if attn:
                st.markdown("**Attention Scores (Cross-Stock Dependencies):**")
                try:
                    attn_df = pd.DataFrame(attn)
                    import plotly.express as px
                    fig = px.imshow(attn_df, color_continuous_scale="Blues",
                                    title="Attention Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.json(attn)
        else:
            st.info("Transformer data will appear during live trading.")

    with col_an:
        st.subheader("Anomaly Detector (Isolation Forest)")
        an_data = load_json(FILE_ANOMALY)
        if an_data:
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                st.metric("Stocks Scanned", an_data.get("n_total", 0))
            with ac2:
                n_anom = an_data.get("n_anomalies", 0)
                st.metric("Anomalies Found", n_anom,
                          delta=f"{'⚠️' if n_anom > 0 else '✅'}")
            with ac3:
                st.metric("Last Scan", str(an_data.get("timestamp", "—"))[:16])

            stocks = an_data.get("stocks", {})
            if stocks:
                st.markdown("**Anomaly Scores:**")
                anom_rows = []
                for sym, info in stocks.items():
                    score = info.get("score", 0)
                    is_anom = info.get("is_anomaly", False)
                    anom_rows.append({
                        "Stock": sym,
                        "Score": f"{score:.4f}",
                        "Status": "🔴 ANOMALY" if is_anom else "🟢 Normal",
                        "Method": info.get("method", "—"),
                    })
                st.dataframe(pd.DataFrame(anom_rows), use_container_width=True)

                # Anomaly score chart
                scores_vals = {s: i.get("score", 0) for s, i in stocks.items() if "score" in i}
                if scores_vals:
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(
                        x=list(scores_vals.keys()),
                        y=list(scores_vals.values()),
                        marker_color=["red" if v < -0.3 else "green" for v in scores_vals.values()],
                    ))
                    fig.update_layout(title="Anomaly Scores", yaxis_title="Score", height=300)
                    fig.add_hline(y=-0.3, line_dash="dash", line_color="red",
                                  annotation_text="Block Threshold")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Anomaly data will appear during live trading.")


# ══════════════════════════════════════════════════
#  TAB 23 — ORDER BOOK & PAIRS (Phase 11)
# ══════════════════════════════════════════════════
if page == "Order Book & Pairs":
    col_ob, col_pt = st.columns(2)

    with col_ob:
        st.subheader("Order Book Simulator")
        ob_data = load_json(FILE_ORDERBOOK)
        if ob_data:
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.metric("Stocks Analysed", ob_data.get("n_analysed", 0))
            with oc2:
                st.metric("Total Impact ₹", f"₹{ob_data.get('total_est_impact', 0):,.2f}")
            with oc3:
                st.metric("Last Update", str(ob_data.get("timestamp", "—"))[:16])

            stocks_ob = ob_data.get("stocks", {})
            if stocks_ob:
                st.markdown("**Impact Cost Analysis:**")
                ob_rows = []
                for sym, info in stocks_ob.items():
                    if "error" in info:
                        continue
                    imp = info.get("impact", {})
                    book = info.get("book_summary", {})
                    ob_rows.append({
                        "Stock": sym,
                        "Price": f"₹{info.get('price', 0):,.2f}",
                        "Spread (bps)": info.get("spread_bps", 0),
                        "Impact Score": imp.get("impact_score", 0),
                        "Slippage %": f"{imp.get('slippage_pct', 0):.4f}%",
                        "Imbalance": book.get("imbalance", 0),
                    })
                if ob_rows:
                    st.dataframe(pd.DataFrame(ob_rows), use_container_width=True)

                # Impact score chart
                import plotly.graph_objects as go
                impact_scores = {s: info.get("impact", {}).get("impact_score", 0)
                                for s, info in stocks_ob.items() if "error" not in info}
                if impact_scores:
                    fig = go.Figure(go.Bar(
                        x=list(impact_scores.keys()),
                        y=list(impact_scores.values()),
                        marker_color=["red" if v > 60 else "orange" if v > 30 else "green"
                                      for v in impact_scores.values()],
                    ))
                    fig.update_layout(title="Impact Cost Scores", yaxis_title="Score (0-100)", height=300)
                    fig.add_hline(y=60, line_dash="dash", line_color="red",
                                  annotation_text="Block Threshold")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Order book data will appear during live trading.")

    with col_pt:
        st.subheader("Pair Trading (Co-Integration)")
        pt_data = load_json(FILE_PAIRS)
        if pt_data:
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.metric("Pairs Tested", pt_data.get("n_tested", 0))
            with pc2:
                st.metric("Co-Integrated", pt_data.get("n_found", 0))
            with pc3:
                st.metric("Last Scan", str(pt_data.get("timestamp", "—"))[:16])

            pairs = pt_data.get("pairs", [])
            if pairs:
                st.markdown("**Detected Pairs:**")
                pair_rows = []
                for p in pairs:
                    spread = p.get("spread_data", {})
                    pair_rows.append({
                        "Leg 1": p.get("leg1", ""),
                        "Leg 2": p.get("leg2", ""),
                        "Correlation": p.get("correlation", 0),
                        "Hedge Ratio": f"{p.get('hedge_ratio', 0):.3f}",
                        "Half-Life": f"{p.get('half_life', 0):.1f}d",
                        "ADF p-val": p.get("adf_pvalue", 1),
                        "Z-Score": spread.get("z_score", "—"),
                        "Signal": spread.get("signal", "—"),
                    })
                st.dataframe(pd.DataFrame(pair_rows), use_container_width=True)

                # Spread chart for top pair
                top_pair = pairs[0]
                spread_history = top_pair.get("spread_data", {}).get("spread_history", [])
                if spread_history:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=spread_history, mode="lines",
                                            name="Spread", line=dict(color="blue")))
                    spread_mean = top_pair.get("spread_data", {}).get("spread_mean", 0)
                    spread_std = top_pair.get("spread_data", {}).get("spread_std", 1)
                    fig.add_hline(y=spread_mean, line_dash="dash", line_color="green",
                                  annotation_text="Mean")
                    fig.add_hline(y=spread_mean + 2 * spread_std, line_dash="dot",
                                  line_color="red", annotation_text="+2σ")
                    fig.add_hline(y=spread_mean - 2 * spread_std, line_dash="dot",
                                  line_color="red", annotation_text="-2σ")
                    fig.update_layout(
                        title=f"Spread: {top_pair.get('leg1', '')} vs {top_pair.get('leg2', '')}",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No co-integrated pairs found in current scan.")
        else:
            st.info("Pair trading data will appear during live trading.")


# ══════════════════════════════════════════════════
#  TAB 24 — DYNAMIC STOPS (Phase 11)
# ══════════════════════════════════════════════════
if page == "Dynamic Stops":
    st.subheader("Dynamic Trailing Stop-Loss (Chandelier Exit)")
    sl_data = load_json(FILE_DYN_SL)
    if sl_data:
        stops = sl_data.get("stops", {})
        if stops:
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.metric("Active Stops", len(stops))
            with dc2:
                avg_mult = sum(s.get("current_multiplier", 3.0) for s in stops.values()) / max(1, len(stops))
                st.metric("Avg ATR Multiplier", f"{avg_mult:.2f}×")
            with dc3:
                trailing_ct = sum(1 for s in stops.values() if s.get("trailing_active"))
                st.metric("Trailing Active", trailing_ct)

            st.markdown("**Position Stop States:**")
            sl_rows = []
            for sym, s in stops.items():
                entry = s.get("entry_price", 0)
                stop = s.get("current_stop", 0)
                high = s.get("highest_price", 0)
                pct_risk = ((entry - stop) / max(0.01, entry)) * 100 if entry > 0 else 0
                sl_rows.append({
                    "Stock": sym,
                    "Entry ₹": f"₹{entry:,.2f}",
                    "Current Stop ₹": f"₹{stop:,.2f}",
                    "Highest ₹": f"₹{high:,.2f}",
                    "Risk %": f"{pct_risk:.2f}%",
                    "ATR Mult": s.get("current_multiplier", "—"),
                    "Days Held": s.get("days_held", 0),
                    "Remaining Qty": s.get("remaining_qty", "—"),
                    "Partials": s.get("partial_exits", 0) if isinstance(s.get("partial_exits"), int) else len(s.get("partial_exits", [])),
                    "Trailing": "✅" if s.get("trailing_active") else "❌",
                })
            st.dataframe(pd.DataFrame(sl_rows), use_container_width=True)

            # Trailing stop chart for each position
            for sym, s in stops.items():
                history = s.get("history", [])
                if len(history) > 3:
                    import plotly.graph_objects as go
                    dates = [h.get("date", "") for h in history]
                    prices = [h.get("price", 0) for h in history]
                    stop_levels = [h.get("stop", 0) for h in history]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines",
                                            name="Price", line=dict(color="blue")))
                    fig.add_trace(go.Scatter(x=dates, y=stop_levels, mode="lines",
                                            name="Trailing Stop", line=dict(color="red", dash="dash")))
                    fig.add_hline(y=s.get("entry_price", 0), line_dash="dot",
                                  line_color="gray", annotation_text="Entry")
                    fig.update_layout(title=f"Trailing Stop: {sym}", height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active trailing stops — will populate when trades are opened.")
    else:
        st.info("Dynamic stop-loss data will appear during live trading.")


# ══════════════════════════════════════════════════
#  TAB 25 — MODEL VERSIONING (Phase 11)
# ══════════════════════════════════════════════════
if page == "Model Versioning":
    st.subheader("Federated Model Versioning & A/B Deployment")
    vs_data = load_json(FILE_VERSIONING)
    if vs_data:
        reg = vs_data.get("registry", {})
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            versions = reg.get("versions", [])
            st.metric("Total Versions", len(versions))
        with vc2:
            prod = reg.get("production", {})
            st.metric("Production Models", len(prod))
        with vc3:
            shadow = reg.get("shadow", {})
            st.metric("Shadow Deployments", len(shadow))

        # Production models
        if prod:
            st.markdown("**🟢 Production Models:**")
            prod_rows = [{"Model Type": mt, "Version": info.get("version_id", "—"),
                          "Promoted At": str(info.get("promoted_at", "—"))[:16]}
                         for mt, info in prod.items()]
            st.dataframe(pd.DataFrame(prod_rows), use_container_width=True)

        # Shadow deployments
        if shadow:
            st.markdown("**🔵 Shadow Models (A/B Testing):**")
            shad_rows = []
            for mt, info in shadow.items():
                shad_rows.append({
                    "Model Type": mt,
                    "Version": info.get("version_id", "—"),
                    "Predictions": info.get("n_predictions", 0),
                    "Trades": info.get("n_trades", 0),
                    "PnL": f"₹{info.get('total_pnl', 0):,.2f}",
                    "Deployed": str(info.get("deployed_at", "—"))[:16],
                })
            st.dataframe(pd.DataFrame(shad_rows), use_container_width=True)

        # Comparison history
        comparisons = reg.get("comparison_history", [])
        if comparisons:
            st.markdown("**Recent A/B Comparisons:**")
            comp_rows = []
            for c in comparisons[-10:]:
                sm = c.get("shadow_metrics", {})
                pm = c.get("production_metrics", {})
                comp = c.get("comparison", {})
                comp_rows.append({
                    "Model Type": c.get("model_type", "—"),
                    "Shadow Sharpe": sm.get("sharpe", 0),
                    "Prod Sharpe": pm.get("sharpe", 0),
                    "Sharpe Δ": f"{comp.get('sharpe_diff', 0):+.3f}",
                    "Shadow Acc": f"{sm.get('accuracy', 0):.1%}",
                    "Prod Acc": f"{pm.get('accuracy', 0):.1%}",
                    "Recommendation": c.get("recommendation", "—"),
                    "Auto-Promoted": "✅" if c.get("auto_promoted") else "",
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        # Version history
        if versions:
            st.markdown("**Version Registry (Recent):**")
            ver_rows = []
            for v in versions[-15:]:
                ver_rows.append({
                    "Version ID": v.get("version_id", "—"),
                    "Type": v.get("model_type", "—"),
                    "Status": v.get("status", "—"),
                    "Tag": v.get("tag", "—"),
                    "Created": str(v.get("created_at", "—"))[:16],
                })
            st.dataframe(pd.DataFrame(ver_rows), use_container_width=True)
    else:
        st.info("Model versioning data will appear after models are registered.")


# ══════════════════════════════════════════════════
#  TAB 26 — ADAPTIVE EXECUTOR
# ══════════════════════════════════════════════════
if page == "Adaptive Executor":
    st.markdown('<div class="section-header">⚡ <h3>Adaptive Execution Engine (VWAP / TWAP)</h3></div>', unsafe_allow_html=True)
    ex_data = load_json(FILE_EXECUTOR)
    if ex_data:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Active Plans", ex_data.get("active_plans", 0))
        with c2:
            st.metric("Completed Plans", ex_data.get("completed_plans", 0))
        with c3:
            st.metric("Avg Slippage (bps)", f"{ex_data.get('avg_slippage_bps', 0):.1f}")

        plans = ex_data.get("plans", {})
        if plans:
            st.markdown("**Execution Plans:**")
            plan_rows = []
            for sym, p in plans.items():
                plan_rows.append({
                    "Symbol": sym,
                    "Strategy": p.get("strategy", "—"),
                    "Total Qty": p.get("total_qty", 0),
                    "Filled": p.get("filled_qty", 0),
                    "Remaining": p.get("remaining_qty", 0),
                    "Avg Fill ₹": p.get("avg_fill_price", 0),
                    "Slippage (bps)": p.get("slippage_bps", 0),
                    "Slices": f"{p.get('slices_filled', 0)}/{p.get('n_slices', 0)}",
                    "Status": p.get("status", "—"),
                })
            st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

        st.caption(f"Last updated: {ex_data.get('timestamp', '—')}")
    else:
        st.info("Adaptive Executor data will appear once orders are sliced.")


# ══════════════════════════════════════════════════
#  TAB 27 — RL REBALANCER
# ══════════════════════════════════════════════════
if page == "RL Rebalancer":
    st.markdown('<div class="section-header">🤖 <h3>RL Portfolio Rebalancer</h3></div>', unsafe_allow_html=True)
    rl_rb = load_json(FILE_RL_REBAL)
    if rl_rb:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Q-Table States", rl_rb.get("qtable_size", 0))
        with c2:
            st.metric("Epsilon (ε)", f"{rl_rb.get('epsilon', 0):.4f}")
        with c3:
            st.metric("History Length", rl_rb.get("history_len", 0))

        recent = rl_rb.get("recent_actions", [])
        if recent:
            st.markdown("**Recent RL Actions:**")
            act_names = {0: "HOLD", 1: "LIGHT", 2: "FULL", 3: "DEFENSIVE"}
            act_rows = []
            for a in recent[-15:]:
                act_rows.append({
                    "Time": str(a.get("timestamp", "—"))[:19],
                    "State": a.get("state", "—"),
                    "Action": act_names.get(a.get("action", 0), "?"),
                    "Reward": f"{a.get('reward', 0):+.4f}",
                })
            st.dataframe(pd.DataFrame(act_rows), use_container_width=True, hide_index=True)

        st.caption(f"Last updated: {rl_rb.get('timestamp', '—')}")
    else:
        st.info("RL Rebalancer data will appear after the first trading session.")


# ══════════════════════════════════════════════════
#  TAB 28 — OPTIONS SYNTHESIZER
# ══════════════════════════════════════════════════
if page == "Options Synthesizer":
    st.markdown('<div class="section-header">🛡️ <h3>Options Strategy Synthesizer</h3></div>', unsafe_allow_html=True)
    synth = load_json(FILE_OPT_SYNTH)
    if synth and synth.get("recommendation") not in (None, "NO_DATA"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Recommendation", synth.get("recommendation", "—"))
        with c2:
            st.metric("Hedge Cost ₹", f"{synth.get('total_hedge_cost', 0):,.2f}")
        with c3:
            st.metric("Cost % Capital", f"{synth.get('cost_pct_of_capital', 0):.2f}%")
        with c4:
            st.metric("Positions Hedged", synth.get("n_positions_hedged", 0))

        top = synth.get("top_strategy", {})
        if top:
            st.markdown(f"**Top Strategy: {top.get('strategy', '—')}**")
            cols_top = st.columns(4)
            for i, (k, v) in enumerate(top.items()):
                if k in ("strategy",):
                    continue
                with cols_top[i % 4]:
                    st.markdown(f"**{k}:** {v}")

        all_strats = synth.get("all_strategies", [])
        if all_strats:
            st.markdown("**All Candidate Strategies (top 10):**")
            strat_rows = []
            for s in all_strats[:10]:
                cost = abs(s.get("net_cost", s.get("premium", s.get("total_cost", s.get("net_debit", 0)))))
                strat_rows.append({
                    "Strategy": s.get("strategy", "—"),
                    "Symbol": s.get("symbol", "—"),
                    "Score": s.get("score", 0),
                    "Cost ₹": f"{cost:,.2f}",
                    "Max Loss ₹": f"{s.get('max_loss', 0):,.2f}",
                    "DTE": s.get("dte", 0),
                    "IV": f"{s.get('iv', 0):.0%}",
                })
            st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

        st.caption(f"Regime: {synth.get('regime', '—')} | VaR: {synth.get('var_pct', 0):.1f}% | {synth.get('timestamp', '—')}")
    else:
        st.info("Options Synthesizer data will appear after positions are opened.")


# ══════════════════════════════════════════════════
#  TAB 29 — CAUSAL ENGINE
# ══════════════════════════════════════════════════
if page == "Causal Engine":
    st.markdown('<div class="section-header">🔬 <h3>Causal Inference Engine</h3></div>', unsafe_allow_html=True)
    causal = load_json(FILE_CAUSAL)
    if causal and causal.get("total_tested"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Features Tested", causal.get("total_tested", 0))
        with c2:
            st.metric("Approved (Causal)", causal.get("total_approved", 0))
        with c3:
            st.metric("Graph Density", f"{causal.get('graph_density', 0):.2f}")

        approved = causal.get("approved_features", [])
        if approved:
            st.success(f"**Causally-approved features:** {', '.join(approved)}")

        edges = causal.get("edges", [])
        if edges:
            st.markdown("**Causal Edge Analysis:**")
            edge_rows = []
            for e in edges:
                edge_rows.append({
                    "Feature": e.get("feature", "—"),
                    "Granger-Caused": "✅" if e.get("granger_caused") else "❌",
                    "F-Stat": f"{e.get('f_stat', 0):.3f}",
                    "P-Value": f"{e.get('p_value', 1):.5f}",
                    "Best Lag": e.get("best_lag", 0),
                    "Partial Corr": f"{e.get('partial_corr', 0):.4f}",
                    "Survives PCorr": "✅" if e.get("survives_pcorr") else "❌",
                    "Approved": "✅" if e.get("approved") else "❌",
                })
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

        # Visualise as a simple network table
        rejected = causal.get("rejected", [])
        if rejected:
            with st.expander("Rejected Features", expanded=False):
                rej_rows = [{"Feature": r.get("feature", "—"), "Reason": r.get("reason", "—")} for r in rejected]
                st.dataframe(pd.DataFrame(rej_rows), use_container_width=True, hide_index=True)

        st.caption(f"Last scan: {causal.get('timestamp', '—')}")
    else:
        st.info("Causal Engine data will appear after the first market scan.")


# ══════════════════════════════════════════════════
#  TAB 30 — GA STRATEGY EVOLVER
# ══════════════════════════════════════════════════
if page == "GA Strategy Evolver":
    st.markdown('<div class="section-header">🧬 <h3>Genetic Algorithm Strategy Evolver</h3></div>', unsafe_allow_html=True)
    ga = load_json(FILE_GA_EVOLVER)
    if ga and ga.get("generation"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Generation", ga.get("generation", 0))
        with c2:
            st.metric("Best Sharpe", f"{ga.get('best_fitness', 0):.3f}")
        with c3:
            st.metric("Symbols Evolved", len(ga.get("symbols_used", [])))
        with c4:
            st.metric("Status", ga.get("status", "—"))

        # Fitness history chart
        fh = ga.get("fitness_history", [])
        if fh:
            st.markdown("**Fitness Evolution Over Generations:**")
            fig_fh = go.Figure()
            fig_fh.add_trace(go.Scatter(
                x=list(range(1, len(fh) + 1)), y=fh,
                mode="lines+markers", name="Best Sharpe",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=4),
            ))
            fig_fh.update_layout(
                template="plotly_dark", height=350,
                xaxis_title="Generation", yaxis_title="Sharpe Ratio",
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_fh, use_container_width=True)

        # Best chromosome parameters
        best_chrom = ga.get("best_chromosome", {})
        if best_chrom:
            st.markdown("**Best Strategy Parameters (Evolved):**")
            param_rows = [{"Gene": k, "Value": v} for k, v in best_chrom.items()]
            st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

        # Per-stock backtest results
        bt = ga.get("best_backtest", {})
        if bt:
            st.markdown("**Walk-Forward Backtest (Best Chromosome):**")
            bt_rows = []
            for s, r in bt.items():
                bt_rows.append({
                    "Symbol": s,
                    "Sharpe": r.get("sharpe", 0),
                    "Return %": r.get("total_return", 0),
                    "Win Rate %": r.get("win_rate", 0),
                    "Trades": r.get("n_trades", 0),
                })
            st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)

        # Generation stats table
        gs = ga.get("generation_stats", [])
        if gs:
            with st.expander("Generation-by-Generation Stats", expanded=False):
                st.dataframe(pd.DataFrame(gs), use_container_width=True, hide_index=True)

        st.caption(f"Last evolution: {ga.get('timestamp', '—')}")
    else:
        st.info("GA Evolver data will appear after the first EOD evolution cycle.")


# ══════════════════════════════════════════════════
#  TAB 31 — MULTI-AGENT DEBATE
# ══════════════════════════════════════════════════
if page == "Multi-Agent Debate":
    st.markdown('<div class="section-header">🗣️ <h3>Multi-Agent Debate System</h3></div>', unsafe_allow_html=True)
    debate = load_json(FILE_DEBATE)
    if debate and debate.get("total_debates"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Debates", debate.get("total_debates", 0))
        with c2:
            st.metric("Approvals", debate.get("approvals", 0))
        with c3:
            st.metric("Rejections", debate.get("rejections", 0))
        with c4:
            st.metric("Vetoes (Risk Bear)", debate.get("veto_count", 0))

        # Approval rate
        total = debate.get("total_debates", 1)
        app_rate = debate.get("approvals", 0) / max(total, 1) * 100
        st.progress(min(app_rate / 100, 1.0), text=f"Approval Rate: {app_rate:.1f}%")

        # Recent debates
        recent = debate.get("recent_debates", [])
        if recent:
            st.markdown("**Recent Debate Results:**")
            deb_rows = []
            for d in reversed(recent[-20:]):
                vc = d.get("vote_counts", {})
                deb_rows.append({
                    "Symbol": d.get("symbol", "—"),
                    "Verdict": d.get("verdict", "—"),
                    "Score": f"{d.get('consensus_score', 0):.3f}",
                    "Vetoed": "⚠️" if d.get("vetoed") else "",
                    "FOR": vc.get("FOR", 0),
                    "AGAINST": vc.get("AGAINST", 0),
                    "ABSTAIN": vc.get("ABSTAIN", 0),
                    "Time": d.get("timestamp", "—")[:19],
                })
            st.dataframe(pd.DataFrame(deb_rows), use_container_width=True, hide_index=True)

            # Verdict distribution pie
            verdicts = [d.get("verdict", "UNKNOWN") for d in recent]
            from collections import Counter
            vc_counts = Counter(verdicts)
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(vc_counts.keys()),
                values=list(vc_counts.values()),
                hole=0.4,
                marker=dict(colors=["#00ff88", "#ff4444", "#ffaa00", "#888888"]),
            )])
            fig_pie.update_layout(
                template="plotly_dark", height=300,
                title="Verdict Distribution",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Consensus score histogram
            scores = [d.get("consensus_score", 0) for d in recent if d.get("consensus_score") is not None]
            if scores:
                fig_hist = go.Figure(data=[go.Histogram(
                    x=scores, nbinsx=20,
                    marker_color="#00d4ff",
                )])
                fig_hist.update_layout(
                    template="plotly_dark", height=280,
                    title="Consensus Score Distribution",
                    xaxis_title="Score", yaxis_title="Count",
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        st.caption(f"Last update: {debate.get('timestamp', '—')}")
    else:
        st.info("Debate System data will appear after the first trade signal is debated.")


# ══════════════════════════════════════════════════
#  TAB 32 — RL TRADE AGENT
# ══════════════════════════════════════════════════
if page == "RL Trade Agent":
    st.markdown('<div class="section-header">🤖 <h3>RL Trade Agent (Deep Q-Network)</h3></div>', unsafe_allow_html=True)
    rl_agent = load_json(FILE_RL_AGENT)
    if rl_agent and rl_agent.get("train_steps", 0) > 0:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Train Steps", f"{rl_agent.get('train_steps', 0):,}")
        with c2:
            st.metric("Epsilon (ε)", f"{rl_agent.get('epsilon', 1):.4f}")
        with c3:
            st.metric("Win Rate", f"{rl_agent.get('win_rate', 0):.1f}%")
        with c4:
            st.metric("Total P&L", f"{rl_agent.get('total_pnl_pct', 0):+.2f}%")

        c5, c6, c7 = st.columns(3)
        with c5:
            st.metric("Total Trades", rl_agent.get("total_trades", 0))
        with c6:
            st.metric("Wins", rl_agent.get("wins", 0))
        with c7:
            st.metric("Losses", rl_agent.get("losses", 0))

        # Recent actions
        actions = rl_agent.get("recent_actions", [])
        if actions:
            st.markdown("**Recent DQN Actions:**")
            act_rows = []
            for a in reversed(actions[-20:]):
                qv = a.get("q_values", {})
                act_rows.append({
                    "Symbol": a.get("symbol", "—"),
                    "Action": a.get("action", "—"),
                    "Q(HOLD)": qv.get("HOLD", 0),
                    "Q(BUY)": qv.get("BUY", 0),
                    "Q(SELL)": qv.get("SELL", 0),
                    "ε": a.get("epsilon", 0),
                    "Time": a.get("timestamp", "—")[:19],
                })
            st.dataframe(pd.DataFrame(act_rows), use_container_width=True, hide_index=True)

        # Trade history
        trades = rl_agent.get("trade_history", [])
        if trades:
            st.markdown("**DQN Trade History:**")
            tr_rows = []
            for t in reversed(trades[-20:]):
                pnl = t.get("pnl_pct", 0)
                tr_rows.append({
                    "Symbol": t.get("symbol", "—"),
                    "Action": t.get("action", "—"),
                    "P&L %": f"{pnl*100:+.2f}%",
                    "Entry": f"₹{t.get('entry', 0):,.2f}",
                    "Exit": f"₹{t.get('exit', 0):,.2f}",
                    "Time": t.get("timestamp", "—")[:19],
                })
            st.dataframe(pd.DataFrame(tr_rows), use_container_width=True, hide_index=True)

            # P&L histogram
            pnls = [t.get("pnl_pct", 0) * 100 for t in trades]
            if pnls:
                fig_pnl = go.Figure(data=[go.Histogram(
                    x=pnls, nbinsx=20,
                    marker_color="#00d4ff",
                )])
                fig_pnl.update_layout(
                    template="plotly_dark", height=280,
                    title="DQN Trade P&L Distribution",
                    xaxis_title="P&L %", yaxis_title="Count",
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

        st.caption(f"Last update: {rl_agent.get('timestamp', '—')}")
    else:
        st.info("RL Trade Agent data will appear after first training steps.")


# ══════════════════════════════════════════════════
#  TAB 33 — SENTIMENT MOMENTUM INDEX
# ══════════════════════════════════════════════════
if page == "Sentiment Momentum":
    st.markdown('<div class="section-header">📊 <h3>Sentiment Momentum Index (SMI)</h3></div>', unsafe_allow_html=True)
    smi = load_json(FILE_SMI)
    if smi and smi.get("market_smi") is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Market SMI", f"{smi.get('market_smi', 0):+.3f}")
        with c2:
            st.metric("Label", smi.get("market_label", "NEUTRAL"))
        with c3:
            st.metric("Best Sentiment", smi.get("best_sentiment", "—"))
        with c4:
            st.metric("Worst Sentiment", smi.get("worst_sentiment", "—"))

        # SMI history time-series
        history = smi.get("history", [])
        if history:
            st.markdown("**SMI Over Time:**")
            hist_smi = [h.get("smi", 0) for h in history]
            hist_ts = [h.get("timestamp", "")[:19] for h in history]
            fig_smi = go.Figure()
            fig_smi.add_trace(go.Scatter(
                x=list(range(len(hist_smi))), y=hist_smi,
                mode="lines+markers", name="Market SMI",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=3),
            ))
            fig_smi.add_hline(y=0.5, line_dash="dash", line_color="#00ff88",
                              annotation_text="Strong Bullish")
            fig_smi.add_hline(y=-0.5, line_dash="dash", line_color="#ff4444",
                              annotation_text="Strong Bearish")
            fig_smi.add_hline(y=0, line_dash="dot", line_color="#888888")
            fig_smi.update_layout(
                template="plotly_dark", height=350,
                xaxis_title="Snapshot", yaxis_title="SMI",
                yaxis=dict(range=[-1.1, 1.1]),
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_smi, use_container_width=True)

        # Per-stock SMI table
        per_stock = smi.get("per_stock", {})
        if per_stock:
            st.markdown("**Per-Stock Sentiment Breakdown:**")
            ps_rows = []
            for sym_name, sd in per_stock.items():
                ps_rows.append({
                    "Symbol": sym_name,
                    "SMI": f"{sd.get('smi', 0):+.3f}",
                    "Label": sd.get("label", "—"),
                })
            st.dataframe(pd.DataFrame(ps_rows), use_container_width=True, hide_index=True)

            # Component breakdown for first stock
            first_sym = list(per_stock.keys())[0]
            first_data = per_stock[first_sym]
            comps = first_data.get("components", [])
            if comps:
                with st.expander(f"Component Breakdown — {first_sym}", expanded=False):
                    comp_rows = []
                    for c in comps:
                        comp_rows.append({
                            "Source": c.get("source", "—"),
                            "Raw": f"{c.get('raw', 0):+.4f}",
                            "Decayed": f"{c.get('decayed', 0):+.4f}",
                            "Weight": c.get("weight", 0),
                            "Age (hrs)": c.get("age_hours", 0),
                            "Detail": c.get("detail", "—"),
                        })
                    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # SMI gauge
        mkt_smi = smi.get("market_smi", 0)
        gauge_color = "#00ff88" if mkt_smi > 0.2 else ("#ff4444" if mkt_smi < -0.2 else "#ffaa00")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mkt_smi,
            title={"text": "Market Sentiment Momentum"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [-1, -0.5], "color": "#8b0000"},
                    {"range": [-0.5, -0.2], "color": "#cc3333"},
                    {"range": [-0.2, 0.2], "color": "#444444"},
                    {"range": [0.2, 0.5], "color": "#228b22"},
                    {"range": [0.5, 1], "color": "#00ff88"},
                ],
            },
        ))
        fig_gauge.update_layout(template="plotly_dark", height=260,
                                margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(f"Last update: {smi.get('timestamp', '—')}")
    else:
        st.info("Sentiment Momentum data will appear after the first market scan.")


# ══════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════
st.divider()
st.markdown(
    '<div class="footer">'
    '🛡️ '
    'Project Aegis v14.0 · NeuroVoter AI · Smart Brain · '
    'Broker Bridge · Market Intelligence · Kelly Sizing · '
    'Earnings Guard · Model Drift · Walk-Forward · '
    'Risk Parity · News Detector · Rebalancer · Options Hedge · '
    'Regime HMM · VaR Stress · FinBERT · Volume Profile · '
    'Execution Quality · Trade Journal · Option Chain · '
    'RL Sizer · Intermarket · Liquidity · Greeks · '
    'Bayesian Fusion · Scalper · Auto Tuner · '
    'Portfolio Transformer · Order Book Sim · Dynamic Stops · '
    'Pair Trading · Anomaly Detector · Model Versioning · '
    'Adaptive Executor · RL Rebalancer · Options Synthesizer · '
    'Causal Engine · GA Strategy Evolver · Multi-Agent Debate · '
    'RL Trade Agent · Sentiment Momentum · '
    '100% Local — No data leaves your PC'
    '</div>',
    unsafe_allow_html=True,
)
