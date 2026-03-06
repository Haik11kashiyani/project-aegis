"""
====================================================
🛡️ PROJECT AEGIS - Configuration
====================================================
Central configuration loaded from environment variables (GitHub Secrets).
If someone copies this repo, they get an empty shell without knowing
which stock you trade or what your profit targets are.
====================================================
"""

import os


# ──────────────────────────────────────────────────
# 🔐  SECURE CONFIG  (Loaded from GitHub Secrets)
# ──────────────────────────────────────────────────
# Multi-stock watchlist: comma-separated tickers in a single secret.
# Scholar scans ALL of them, ranks by AI probability, and picks the top N.
_RAW_WATCHLIST = os.getenv(
    "STOCK_WATCHLIST",
    "TATASTEEL.NS,SBIN.NS,RELIANCE.NS,HDFCBANK.NS,ICICIBANK.NS,"
    "NTPC.NS,POWERGRID.NS,COALINDIA.NS,INFY.NS,TCS.NS"
)
STOCK_WATCHLIST: list[str] = [s.strip() for s in _RAW_WATCHLIST.split(",") if s.strip()]
TOP_N_STOCKS   = int(os.getenv("TOP_N_STOCKS", "3"))           # Trade only top-N ranked stocks

# Legacy single-stock fallback (still used where needed)
TARGET_STOCK = os.getenv("TARGET_STOCK", STOCK_WATCHLIST[0])

DAILY_TARGET = float(os.getenv("DAILY_TARGET", "0.02"))       # 2 % daily target
MAX_BULLETS  = int(os.getenv("MAX_BULLETS", "5"))              # Split capital into 5 shots
TIME_GAP     = int(os.getenv("TIME_GAP", "600"))               # 10 min between bullets
CAPITAL      = float(os.getenv("CAPITAL", "wwwwwwwwwwwww"))              # Starting capital ₹

# ──────────────────────────────────────────────────
#   MODEL / TRAINING PARAMETERS
# ──────────────────────────────────────────────────
RF_ESTIMATORS       = 200        # Number of trees in Random Forest
RF_MIN_SPLIT        = 50         # Minimum samples to split (anti-overfit)
RF_MAX_DEPTH        = 12         # Tree depth cap (anti-overfit)

# XGBoost parameters
XGB_ESTIMATORS      = 200        # Number of boosting rounds
XGB_MAX_DEPTH       = 6          # Shallower than RF (XGB is more powerful per tree)
XGB_LEARNING_RATE   = 0.05       # Low LR + high estimators = better generalisation
XGB_SUBSAMPLE       = 0.8        # Row sampling (anti-overfit)
XGB_COLSAMPLE       = 0.8        # Feature sampling (anti-overfit)

# Daily LSTM parameters
LSTM_LOOKBACK       = 60         # Days of history the daily LSTM "sees"
LSTM_EPOCHS         = 10         # Training epochs (keep low to avoid overfit)
LSTM_BATCH_SIZE     = 32
LSTM_DROPOUT        = 0.25       # Dropout rate (anti-overfit)

# Intraday LSTM parameters (15-minute candles)
INTRADAY_LOOKBACK   = 48         # 48 × 15min = 12 hours of candle history
INTRADAY_EPOCHS     = 8
INTRADAY_BATCH_SIZE = 32
INTRADAY_DROPOUT    = 0.20
INTRADAY_PERIOD     = "60d"      # Max period for 15-min data on Yahoo

DATA_PERIOD         = "5y"       # Historical data period for training
CONFIDENCE_THRESHOLD = 0.60      # Minimum confidence for RF/XGB to vote BUY

# Ensemble voting: require at least N out of 4 models to agree
MIN_VOTES_TO_BUY    = 2          # 2-out-of-4 consensus (RF+XGB enough when LSTMs are weak)

# ──────────────────────────────────────────────────
#   SENTIMENT ANALYSIS
# ──────────────────────────────────────────────────
SENTIMENT_ENABLED    = True       # Toggle news sentiment feature
SENTIMENT_MAX_ARTICLES = 10       # Max headlines to analyse per stock
SENTIMENT_LOOKBACK_DAYS = 3       # How many days of news to consider

# ──────────────────────────────────────────────────
#   STRATEGY FILTERS (Technical Entry Gates)
# ──────────────────────────────────────────────────
STRATEGY_FILTERS_ENABLED = True  # Master switch for all filters below

# Trend Filter: Only buy when price is above key moving averages
FILTER_TREND_ENABLED     = True  # Price > EMA_20 > SMA_50 required

# RSI Filter: Skip overbought / freefall stocks
FILTER_RSI_ENABLED       = True
FILTER_RSI_MAX           = 70    # Don't buy if RSI above this (overbought)
FILTER_RSI_MIN           = 25    # Don't buy if RSI below this (freefall)

# MACD Confirmation: Momentum must be positive
FILTER_MACD_ENABLED      = True  # MACD > MACD_Signal required

# Volume Confirmation: Need above-average volume
FILTER_VOLUME_ENABLED    = True
FILTER_VOLUME_MIN_RATIO  = 0.8   # Volume must be >= 80% of 20-day average

# Bollinger Band Resistance: Don't buy near upper band
FILTER_BB_ENABLED        = True
FILTER_BB_UPPER_PCT      = 0.98  # Don't buy if price > 98% toward BB upper

# Per-Stock Cooldown: Max bullets per stock per day
MAX_BULLETS_PER_STOCK    = 1     # No more averaging down blindly

# Partial Profit Taking (scale-out strategy)
PARTIAL_EXIT_ENABLED     = True
PARTIAL_EXIT_ATR_MULT    = 1.5   # Take partial at 1.5× ATR profit
PARTIAL_EXIT_PCT         = 0.50  # Exit 50% of position at partial target

# Prefer cheaper stocks (better position sizing with small capital)
PREFER_CHEAP_STOCKS      = True
MAX_PRICE_FOR_PREFERENCE = 800   # Stocks under this get ranking boost
CHEAP_STOCK_BOOST        = 0.10  # +10% boost to ranking score for cheap stocks

# ──────────────────────────────────────────────────
#   RISK MANAGEMENT
# ──────────────────────────────────────────────────
ATR_STOP_MULTIPLIER   = 1.5     # Stop loss = 1.5 × ATR below entry
ATR_TARGET_MULTIPLIER = 3.0     # Target   = 3.0 × ATR above entry
MAX_DAILY_LOSS_PCT    = 0.05    # Kill-switch: stop if total loss > 5 % of capital

# ──────────────────────────────────────────────────
#   REAL MONEY MODE (Risk Guardian)
# ──────────────────────────────────────────────────
REAL_MONEY_MODE       = True     # Enable all safety restrictions
RISK_GUARDIAN_ENABLED  = True    # Must pass Risk Guardian before every trade
LEARNER_ENABLED        = True    # Enable off-market continuous learning

# ──────────────────────────────────────────────────
#   FILE PATHS
# ──────────────────────────────────────────────────
# Per-stock model paths (multi-stock support)
def model_paths(symbol: str) -> dict:
    """Return model file paths for a given stock ticker."""
    safe = symbol.replace(".", "_").replace("-", "_")
    return {
        "rf":              f"models/{safe}_rf.pkl",
        "xgb":             f"models/{safe}_xgb.pkl",
        "lstm":            f"models/{safe}_lstm.h5",
        "lstm_scaler":     f"models/{safe}_scaler.pkl",
        "intraday_lstm":   f"models/{safe}_intraday_lstm.h5",
        "intraday_scaler": f"models/{safe}_intraday_scaler.pkl",
    }

# Legacy single-stock paths (backward compat)
RF_MODEL_PATH     = "models/rf_brain.pkl"
XGB_MODEL_PATH    = "models/xgb_brain.pkl"
LSTM_MODEL_PATH   = "models/lstm_brain.h5"
SCALER_PATH       = "models/scaler.pkl"
INTRADAY_LSTM_PATH = "models/intraday_lstm_brain.h5"
INTRADAY_SCALER_PATH = "models/intraday_scaler.pkl"

STATE_FILE        = "data/daily_state.json"
TRADE_LOG_FILE    = "data/trade_history.csv"
RANKING_FILE      = "data/daily_ranking.csv"
DASHBOARD_FILE    = "data/dashboard_state.json"
ANALYSIS_FILE     = "data/live_analysis.json"
LEARNER_REPORT    = "data/learner_report.json"
GUARDIAN_LOG      = "data/guardian_log.json"
ENSEMBLE_WEIGHTS  = "data/ensemble_weights.json"
BEST_PARAMS_FILE  = "data/best_params.json"

# ──────────────────────────────────────────────────
#   TRADING BRAIN (Human-Like Intelligence)
# ──────────────────────────────────────────────────
# Choose your trading personality:
#   AGGRESSIVE   — More trades, wider stops, lower confidence bar
#   MODERATE     — Balanced (recommended for most users)
#   CONSERVATIVE — Fewer trades, tighter stops, higher confidence needed
TRADING_PERSONALITY = os.getenv("TRADING_PERSONALITY", "MODERATE")

# Smart exit system (the fix for "never sells")
SMART_EXIT_ENABLED     = True   # Enable human-like exit logic (momentum, RSI, volume, time)
SMART_ENTRY_ENABLED    = True   # Enable market-context-aware entries

# Global market mood check (VIX, S&P500, crude oil, USD/INR)
GLOBAL_MOOD_ENABLED    = True   # Check global indicators before trading

# ──────────────────────────────────────────────────
#   FEATURE LIST (must match between Scholar & Sniper)
# ──────────────────────────────────────────────────
RF_FEATURES = ["RSI", "SMA_50", "SMA_200", "EMA_20", "ATR",
               "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
               "Volume_Ratio", "OBV", "Sentiment_Score"]

# ──────────────────────────────────────────────────
#   BROKER BRIDGE (Paper-to-Live Trading)
# ──────────────────────────────────────────────────
# TRADE_MODE: PAPER (default) | DRY_RUN (shows orders but doesn't send) | LIVE
BROKER_MODE = os.getenv("AEGIS_TRADE_MODE", "PAPER")
# BROKER: PAPER | ZERODHA | ANGEL_ONE | GROWW
BROKER_NAME = os.getenv("AEGIS_BROKER", "PAPER")

# ──────────────────────────────────────────────────
#   NEW MODULE TOGGLES (Phase 6 Features)
# ──────────────────────────────────────────────────
SECTOR_ROTATION_ENABLED    = True   # Adjust rankings by sector momentum
MARKET_BREADTH_ENABLED     = True   # Macro breadth filter for position sizing
CORRELATION_GUARD_ENABLED  = True   # Block correlated positions
INTRADAY_PATTERNS_ENABLED  = True   # Time-of-day entry optimisation
CONFIDENCE_DECAY_ENABLED   = True   # Auto-penalise bad voters in learner
REPORT_AUTO_GENERATE       = True   # Generate daily HTML report at EOD

# ──────────────────────────────────────────────────
#   NEW MODULE TOGGLES (Phase 7 Features)
# ──────────────────────────────────────────────────
KELLY_SIZING_ENABLED       = True   # Adaptive position sizing (Kelly Criterion)
EARNINGS_GUARD_ENABLED     = True   # Block/reduce trades near quarterly results
MODEL_DRIFT_ENABLED        = True   # Monitor prediction drift & flag retraining
TELEGRAM_BOT_ENABLED       = True   # Two-way Telegram bot (needs TELEGRAM_BOT_TOKEN)
WALK_FORWARD_ENABLED       = True   # Walk-forward simulator (manual / scheduled)

# ──────────────────────────────────────────────────
#   NEW MODULE TOGGLES (Phase 8 Features)
# ──────────────────────────────────────────────────
OPTIONS_HEDGE_ENABLED      = True   # Auto protective-put calculator & exposure gate
MULTI_TIMEFRAME_ENABLED    = True   # Weekly + monthly trend consensus gate
NEWS_DETECTOR_ENABLED      = True   # NLP news event detector (M&A, regulatory, etc.)
REBALANCER_ENABLED         = True   # Weekly portfolio rebalancing engine
RISK_PARITY_ENABLED        = True   # Inverse-volatility position sizing
AB_BACKTEST_ENABLED        = True   # A/B strategy comparison framework

# ──────────────────────────────────────────────────
#   NEW MODULE TOGGLES (Phase 9 Features)
# ──────────────────────────────────────────────────
REGIME_DETECTOR_ENABLED    = True   # HMM-based bull/bear/sideways regime detection
VAR_STRESS_ENABLED         = True   # VaR & stress testing (Monte Carlo, historical)
FINBERT_ENABLED            = True   # FinBERT transformer sentiment (falls back to keyword)
EXECUTION_QUALITY_ENABLED  = True   # Trade execution grading & slippage tracking
OPTION_CHAIN_ENABLED       = True   # Live NSE option chain (falls back to BS)
VOLUME_PROFILE_ENABLED     = True   # Volume-at-price & order flow analysis
SMART_ALERTS_ENABLED       = True   # Graduated tiered Telegram alerts
TRADE_JOURNAL_ENABLED      = True   # Auto weekly HTML trade journal

# ──────────────────────────────────────────────────
#   NEW MODULE TOGGLES (Phase 10 Features)
# ──────────────────────────────────────────────────
RL_SIZER_ENABLED           = True   # Q-learning adaptive position sizing
INTERMARKET_ENABLED        = True   # Gold/USD/Crude/US10Y correlation engine
LIQUIDITY_FILTER_ENABLED   = True   # Corwin-Schultz spread + volume filter
GREEKS_HEATMAP_ENABLED     = True   # Portfolio-level Greeks heat map
BAYESIAN_FUSION_ENABLED    = True   # Bayesian posterior signal fusion
INTRADAY_SCALPER_ENABLED   = True   # Sub-15-min VWAP-cross scalp overlay
AUTO_TUNER_ENABLED         = True   # Optuna-based hyper-parameter sweep

# ── Phase 11 Toggles ─────────────────────────────
TRANSFORMER_ENABLED        = True   # Attention-based portfolio transformer
ORDERBOOK_SIM_ENABLED      = True   # Tick-level order-book impact simulator
DYNAMIC_STOPLOSS_ENABLED   = True   # Trailing ATR + Chandelier Exit stops
PAIR_TRADING_ENABLED       = True   # Co-integration pair-trading engine
ANOMALY_DETECTOR_ENABLED   = True   # Isolation Forest anomaly guard
MODEL_VERSIONING_ENABLED   = True   # Shadow A/B model deployment

# ── Phase 12 Toggles ─────────────────────────────
ADAPTIVE_EXECUTOR_ENABLED  = True   # VWAP/TWAP order slicing engine
RL_REBALANCER_ENABLED      = True   # RL-based portfolio rebalancer
OPTIONS_SYNTH_ENABLED      = True   # Auto options strategy synthesizer
CAUSAL_ENGINE_ENABLED      = True   # Causal inference feature filter

# ── Phase 13 Toggles ─────────────────────────────
GA_EVOLVER_ENABLED         = True   # Genetic Algorithm strategy evolver
DEBATE_SYSTEM_ENABLED      = True   # Multi-Agent Debate consensus gate

# ── Phase 14 Toggles ─────────────────────────────
RL_TRADE_AGENT_ENABLED     = True   # Deep Q-Network entry/exit agent
SENTIMENT_MOMENTUM_ENABLED = True   # Composite sentiment momentum index
