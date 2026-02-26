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
CAPITAL      = float(os.getenv("CAPITAL", "1000"))              # Starting capital ₹

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
LEARNER_REPORT    = "data/learner_report.json"
GUARDIAN_LOG      = "data/guardian_log.json"
ENSEMBLE_WEIGHTS  = "data/ensemble_weights.json"
BEST_PARAMS_FILE  = "data/best_params.json"

# ──────────────────────────────────────────────────
#   FEATURE LIST (must match between Scholar & Sniper)
# ──────────────────────────────────────────────────
RF_FEATURES = ["RSI", "SMA_50", "SMA_200", "EMA_20", "ATR",
               "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
               "Volume_Ratio", "OBV", "Sentiment_Score"]
