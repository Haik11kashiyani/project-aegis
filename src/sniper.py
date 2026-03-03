"""
====================================================
PROJECT AEGIS - The Sniper v2 (Live Paper-Trading)
====================================================
Runs during Indian market hours (9:15 AM - 3:15 PM IST).
  1. Loads today's top-N stocks from Scholar ranking
  2. Loads 4 AI brains per stock (RF + XGB + daily LSTM + intraday LSTM)
  3. Splits capital into N bullets (position sizing)
  4. Fires a bullet ONLY when 3-of-4 models agree (ensemble voting)
  5. Uses ATR-based dynamic stop-loss and target
  6. Enforces time-diversity (10-min gap between shots)
  7. Stops when daily target (2%) is hit OR max loss reached
  8. Force-closes all positions before market close
  9. Logs every trade to CSV for the Scholar to review tonight
  10. Writes dashboard state JSON for the Streamlit dashboard
====================================================
"""

import os
import sys

# Suppress all TensorFlow and ABSL warnings before loading any TF modules
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "ABSL_LOG_LEVEL" not in os.environ:
    os.environ["ABSL_LOG_LEVEL"] = "3"
    
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import json
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
import indicators as ta
from indicators import flatten_yf_columns
import joblib
from datetime import datetime
import pytz

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras.models import load_model

from config import (
    STOCK_WATCHLIST, TOP_N_STOCKS,
    CAPITAL, MAX_BULLETS, TIME_GAP, DAILY_TARGET,
    CONFIDENCE_THRESHOLD, RF_FEATURES,
    ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER, MAX_DAILY_LOSS_PCT,
    LSTM_LOOKBACK, INTRADAY_LOOKBACK,
    MIN_VOTES_TO_BUY,
    SENTIMENT_ENABLED, SENTIMENT_MAX_ARTICLES, SENTIMENT_LOOKBACK_DAYS,
    RISK_GUARDIAN_ENABLED,
    model_paths, STATE_FILE, TRADE_LOG_FILE, RANKING_FILE, DASHBOARD_FILE,
    ANALYSIS_FILE,
    # Strategy filters
    STRATEGY_FILTERS_ENABLED,
    FILTER_TREND_ENABLED, FILTER_RSI_ENABLED, FILTER_RSI_MAX, FILTER_RSI_MIN,
    FILTER_MACD_ENABLED, FILTER_VOLUME_ENABLED, FILTER_VOLUME_MIN_RATIO,
    FILTER_BB_ENABLED, FILTER_BB_UPPER_PCT,
    MAX_BULLETS_PER_STOCK,
    PARTIAL_EXIT_ENABLED, PARTIAL_EXIT_ATR_MULT, PARTIAL_EXIT_PCT,
)
from sentiment import get_sentiment_score
from risk_guardian import RiskGuardian, NeuralSafetyNet, quick_safety_check

IST = pytz.timezone("Asia/Kolkata")

# --------------------------------------------------
#   PREMIUM CONSOLE LOGGING (ANSI)
# --------------------------------------------------
class Log:
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    @staticmethod
    def info(msg):
        print(f"{Log.BLUE}[INFO]{Log.RESET} {msg}")

    @staticmethod
    def success(msg):
        print(f"{Log.GREEN}[OK]{Log.RESET} {msg}")

    @staticmethod
    def warn(msg):
        print(f"{Log.YELLOW}[WARN]{Log.RESET} {msg}")

    @staticmethod
    def error(msg):
        print(f"{Log.RED}[ERROR]{Log.RESET} {msg}")

    @staticmethod
    def vote(symbol, rf, xgb, lstm, intra, votes, needed):
        color = Log.GREEN if votes >= needed else (Log.RED if votes == 0 else Log.YELLOW)
        print(f"   {Log.DIM}»{Log.RESET} {Log.BOLD}{symbol:12}{Log.RESET} | RF={rf:.2f} XGB={xgb:.2f} LSTM={lstm:.2f} Intra={intra:.2f} | {color}Votes: {votes}/{needed}{Log.RESET}")

    @staticmethod
    def highlight(msg):
        print(f"{Log.CYAN}{Log.BOLD}{msg}{Log.RESET}")

    @staticmethod
    def section(title):
        print(f"\n{Log.MAGENTA}╔{'═' * 58}╗{Log.RESET}")
        print(f"{Log.MAGENTA}║ {Log.BOLD}{title.center(56)}{Log.RESET} {Log.MAGENTA}║{Log.RESET}")
        print(f"{Log.MAGENTA}╚{'═' * 58}╝{Log.RESET}")


# --------------------------------------------------
#   STATE MANAGEMENT  (survives GitHub Action restarts)
# --------------------------------------------------
def _default_state() -> dict:
    return {
        "date": datetime.now(IST).strftime("%Y-%m-%d"),
        "total_profit": 0.0,
        "total_profit_pct": 0.0,
        "trades_taken": 0,
        "trades_won": 0,
        "trades_lost": 0,
        "status": "ACTIVE",
        "active_trades": [],
        "stocks_traded": [],
    }


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        if state.get("date") != datetime.now(IST).strftime("%Y-%m-%d"):
            return _default_state()
        return state
    return _default_state()


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    # Also write dashboard state
    try:
        dash = {
            "last_updated": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
            "status": state["status"],
            "total_profit": state["total_profit"],
            "total_profit_pct": state.get("total_profit_pct", 0),
            "trades_taken": state["trades_taken"],
            "trades_won": state["trades_won"],
            "trades_lost": state["trades_lost"],
            "stocks_traded": state.get("stocks_traded", []),
            "active_count": sum(1 for t in state["active_trades"] if t.get("status") == "OPEN"),
        }
        with open(DASHBOARD_FILE, "w") as f:
            json.dump(dash, f, indent=2, default=str)
    except Exception:
        pass


# --------------------------------------------------
#   TRADE LOGGING
# --------------------------------------------------
def log_trade(entry: dict):
    os.makedirs(os.path.dirname(TRADE_LOG_FILE) or ".", exist_ok=True)
    df_new = pd.DataFrame([entry])
    if os.path.exists(TRADE_LOG_FILE):
        df_new.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(TRADE_LOG_FILE, index=False)


# --------------------------------------------------
#   LIVE ANALYSIS OUTPUT (for dashboard)
# --------------------------------------------------
def save_analysis(analysis_data: dict):
    """Write live analysis JSON so the dashboard can show real-time AI signals."""
    os.makedirs(os.path.dirname(ANALYSIS_FILE) or ".", exist_ok=True)
    try:
        with open(ANALYSIS_FILE, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)
    except Exception:
        pass


# --------------------------------------------------
#   LOAD TOP-N STOCKS FROM RANKING
# --------------------------------------------------
def load_top_stocks() -> list:
    """Load today's top-N ranked stocks from Scholar output."""
    if os.path.exists(RANKING_FILE):
        try:
            df = pd.read_csv(RANKING_FILE)
            today = datetime.now(IST).strftime("%Y-%m-%d")
            df_today = df[df["date"] == today] if "date" in df.columns else df
            if not df_today.empty:
                return df_today.head(TOP_N_STOCKS)["symbol"].tolist()
            # Fallback: use whatever ranking exists
            return df.head(TOP_N_STOCKS)["symbol"].tolist()
        except Exception as e:
            Log.warn(f"Could not load ranking: {e}")
    # Fallback: use first N from watchlist
    return STOCK_WATCHLIST[:TOP_N_STOCKS]


# --------------------------------------------------
#   LOAD 4 MODELS FOR A STOCK
# --------------------------------------------------
def load_models(symbol: str) -> dict:
    """Load RF, XGB, daily LSTM, intraday LSTM for one stock."""
    paths = model_paths(symbol)
    models = {"rf": None, "xgb": None, "lstm": None, "lstm_scaler": None,
              "intraday_lstm": None, "intraday_scaler": None}

    try:
        models["rf"] = joblib.load(paths["rf"])
    except Exception:
        print(f"   [WARN] No RF model for {symbol}")

    try:
        models["xgb"] = joblib.load(paths["xgb"])
    except Exception:
        print(f"   [WARN] No XGB model for {symbol}")

    try:
        models["lstm"] = load_model(paths["lstm"])
        models["lstm_scaler"] = joblib.load(paths["lstm_scaler"])
    except Exception:
        print(f"   [WARN] No daily LSTM for {symbol}")

    try:
        models["intraday_lstm"] = load_model(paths["intraday_lstm"])
        models["intraday_scaler"] = joblib.load(paths["intraday_scaler"])
    except Exception:
        print(f"   [WARN] No intraday LSTM for {symbol}")

    loaded = sum(1 for k in ["rf", "xgb", "lstm", "intraday_lstm"] if models[k] is not None)
    Log.success(f"{symbol:10} | {loaded}/4 models loaded")
    return models


# --------------------------------------------------
#   DATA FETCHER (daily + indicators + sentiment)
# --------------------------------------------------
def _yf_download_with_retry(symbol: str, period: str, interval: str, max_retries: int = 3) -> pd.DataFrame:
    """Download from yfinance with retry logic for intermittent API failures."""
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            df = flatten_yf_columns(df)
            if df is not None and not df.empty:
                return df
            if attempt < max_retries:
                time.sleep(2 * attempt)  # Backoff: 2s, 4s
        except Exception as e:
            if attempt < max_retries:
                Log.warn(f"{symbol} download attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                time.sleep(2 * attempt)
            else:
                Log.warn(f"{symbol} download failed after {max_retries} attempts: {e}")
    return pd.DataFrame()


def get_live_data(symbol: str) -> pd.DataFrame:
    """Fetch recent daily data + indicators for the ensemble decision."""
    try:
        # Need at least 200+ trading days for SMA_200 — use 2 years
        df = _yf_download_with_retry(symbol, period="2y", interval="1d")
        if df.empty:
            print(f"   [WARN] No data downloaded for {symbol}")
            return None

        df["RSI"]         = ta.rsi(df["Close"], length=14)
        df["SMA_50"]      = ta.sma(df["Close"], length=50)
        df["SMA_200"]     = ta.sma(df["Close"], length=200)
        df["EMA_20"]      = ta.ema(df["Close"], length=20)
        df["ATR"]         = ta.atr(df["High"], df["Low"], df["Close"], length=14)

        macd = ta.macd(df["Close"])
        if macd is not None and not macd.empty:
            df["MACD"]        = macd.iloc[:, 0]
            df["MACD_Signal"] = macd.iloc[:, 2]
        else:
            df["MACD"] = 0.0
            df["MACD_Signal"] = 0.0

        bb = ta.bbands(df["Close"], length=20, std=2)
        if bb is not None and not bb.empty:
            df["BB_Upper"] = bb.iloc[:, 2]
            df["BB_Lower"] = bb.iloc[:, 0]
        else:
            df["BB_Upper"] = df["Close"]
            df["BB_Lower"] = df["Close"]

        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["OBV"]          = ta.obv(df["Close"], df["Volume"])

        # Sentiment
        if SENTIMENT_ENABLED:
            score = get_sentiment_score(symbol, SENTIMENT_MAX_ARTICLES, SENTIMENT_LOOKBACK_DAYS)
            df["Sentiment_Score"] = score
        else:
            df["Sentiment_Score"] = 0.0

        # Fill NaN from indicator warm-up (forward-fill then back-fill)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Safety: if the last row still has NaN, drop those rows only
        if df.iloc[-1].isna().any():
            df.dropna(inplace=True)

        if df.empty:
            print(f"   [WARN] {symbol}: All rows NaN after indicators — data too short")
            return None

        return df
    except Exception as e:
        print(f"   [WARN] Data fetch error for {symbol}: {e}")
        return None


# --------------------------------------------------
#   GET INTRADAY DATA (15-min)
# --------------------------------------------------
def get_intraday_data(symbol: str) -> pd.DataFrame:
    """Fetch recent 15-min candle data for intraday LSTM."""
    try:
        df = _yf_download_with_retry(symbol, period="5d", interval="15m")
        if df.empty:
            return None
        return df
    except Exception:
        return None


# --------------------------------------------------
#   4-MODEL ENSEMBLE VOTING
# --------------------------------------------------
# One NeuralSafetyNet per day — tracks prediction history across all stocks
_neural_safety = NeuralSafetyNet()


def get_ensemble_signal(symbol: str, df: pd.DataFrame, models: dict) -> tuple:
    """
    Returns (should_buy, votes, rf_conf, xgb_conf, lstm_conf, intra_conf)
    Buy only if >= MIN_VOTES_TO_BUY models agree.

    All LSTM outputs pass through the NeuralSafetyNet which:
      - Rejects NaN / Inf / out-of-range outputs
      - Blocks extreme (>0.99 / <0.01) overconfidence
      - Detects stuck models (same prediction N times)
      - Flags unstable predictions (>35% jump between ticks)
      - Validates input arrays before feeding them to the network
    If any check fails the model effectively abstains (conf=0.50).
    """
    global _neural_safety
    votes = 0
    rf_conf = 0.0
    xgb_conf = 0.0
    lstm_conf = 0.0
    intra_conf = 0.0

    last_row = df.iloc[-1:]

    # ---- Random Forest Vote ----
    if models["rf"] is not None:
        try:
            rf_conf = float(models["rf"].predict_proba(last_row[RF_FEATURES])[0][1])
            if rf_conf > CONFIDENCE_THRESHOLD:
                votes += 1
        except Exception as e:
            print(f"   [WARN] {symbol} RF prediction failed: {e}")

    # ---- XGBoost Vote ----
    if models["xgb"] is not None:
        try:
            xgb_conf = float(models["xgb"].predict_proba(last_row[RF_FEATURES])[0][1])
            if xgb_conf > CONFIDENCE_THRESHOLD:
                votes += 1
        except Exception as e:
            print(f"   [WARN] {symbol} XGB prediction failed: {e}")

    # ---- Daily LSTM Vote (with NeuralSafetyNet) ----
    if models["lstm"] is not None and models["lstm_scaler"] is not None:
        try:
            close_prices = df["Close"].values[-LSTM_LOOKBACK:].reshape(-1, 1)
            scaled = models["lstm_scaler"].transform(close_prices)
            X_input = scaled.reshape(1, LSTM_LOOKBACK, 1)

            # ─── Neural Safety: validate input BEFORE inference ───
            raw_lstm = float(models["lstm"].predict(X_input, verbose=0)[0][0])
            lstm_conf = _neural_safety.validate(
                f"daily_lstm_{symbol}", raw_lstm, input_array=X_input
            )
            if lstm_conf > 0.55:
                votes += 1
        except Exception as e:
            print(f"   [WARN] {symbol} Daily LSTM prediction failed: {e}")

    # ---- Intraday LSTM Vote (with NeuralSafetyNet) ----
    if models["intraday_lstm"] is not None and models["intraday_scaler"] is not None:
        try:
            df_intra = get_intraday_data(symbol)
            if df_intra is not None and len(df_intra) >= INTRADAY_LOOKBACK:
                close_intra = df_intra["Close"].values[-INTRADAY_LOOKBACK:].reshape(-1, 1)
                scaled = models["intraday_scaler"].transform(close_intra)
                X_input = scaled.reshape(1, INTRADAY_LOOKBACK, 1)

                # ─── Neural Safety: validate input BEFORE inference ───
                raw_intra = float(models["intraday_lstm"].predict(X_input, verbose=0)[0][0])
                intra_conf = _neural_safety.validate(
                    f"intra_lstm_{symbol}", raw_intra, input_array=X_input
                )
                if intra_conf > 0.55:
                    votes += 1
        except Exception as e:
            print(f"   [WARN] {symbol} Intraday LSTM prediction failed: {e}")

    # ---- Neural Safety: ensemble disagreement check ----
    ensemble_valid, ens_reason = _neural_safety.validate_ensemble({
        "rf": rf_conf, "xgb": xgb_conf,
        "lstm": lstm_conf, "intra": intra_conf,
    })
    if not ensemble_valid:
        print(f"   [NEURAL_SAFETY] {symbol}: {ens_reason}")
        return False, 0, rf_conf, xgb_conf, lstm_conf, intra_conf

    Log.vote(symbol, rf_conf, xgb_conf, lstm_conf, intra_conf, votes, MIN_VOTES_TO_BUY)

    should_buy = (votes >= MIN_VOTES_TO_BUY)
    return should_buy, votes, rf_conf, xgb_conf, lstm_conf, intra_conf


# --------------------------------------------------
#   POSITION SIZING
# --------------------------------------------------
def calculate_position(price: float, atr: float, bullet_size: float) -> dict:
    stop_loss = price - (ATR_STOP_MULTIPLIER * atr)
    target = price + (ATR_TARGET_MULTIPLIER * atr)
    risk_per_share = price - stop_loss

    if risk_per_share <= 0:
        return None

    qty_by_capital = int(bullet_size / price)
    max_risk = bullet_size * 0.10
    qty_by_risk = int(max_risk / risk_per_share) if risk_per_share > 0 else 0
    qty = max(1, min(qty_by_capital, qty_by_risk))

    return {
        "qty": qty,
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "risk_per_share": round(risk_per_share, 2),
        "total_risk": round(qty * risk_per_share, 2),
        "potential_profit": round(qty * (target - price), 2),
    }


# --------------------------------------------------
#   STRATEGY FILTERS (7 Technical Entry Gates)
# --------------------------------------------------
def check_strategy_filters(symbol: str, df: pd.DataFrame) -> tuple:
    """
    Run all 7 technical filters on the latest data.
    Returns (passed: bool, reason: str, filter_details: dict)
    
    Filters:
    1. Trend: Price > EMA_20 > SMA_50  (uptrend confirmation)
    2. RSI: Not overbought (>70) or in freefall (<25)
    3. MACD: MACD line > Signal line  (positive momentum)
    4. Volume: Above-average volume   (institutional participation)
    5. Bollinger: Not near upper band  (not at resistance)
    6. Per-stock cooldown (checked externally)
    7. Partial exit (managed in trade monitoring)
    """
    if not STRATEGY_FILTERS_ENABLED:
        return True, "Filters disabled", {}

    last = df.iloc[-1]
    details = {}
    reasons_blocked = []

    # ── Filter 1: Trend ──
    if FILTER_TREND_ENABLED:
        price = float(last.get("Close", 0))
        ema20 = float(last.get("EMA_20", 0))
        sma50 = float(last.get("SMA_50", 0))
        trend_ok = (price > ema20) and (ema20 > sma50) if ema20 > 0 and sma50 > 0 else False
        details["trend"] = {"price": round(price, 2), "ema20": round(ema20, 2), "sma50": round(sma50, 2), "ok": trend_ok}
        if not trend_ok:
            reasons_blocked.append(f"Trend: Price({price:.0f}) > EMA20({ema20:.0f}) > SMA50({sma50:.0f}) FAILED")

    # ── Filter 2: RSI ──
    if FILTER_RSI_ENABLED:
        rsi = float(last.get("RSI", 50))
        rsi_ok = FILTER_RSI_MIN <= rsi <= FILTER_RSI_MAX
        details["rsi"] = {"value": round(rsi, 1), "min": FILTER_RSI_MIN, "max": FILTER_RSI_MAX, "ok": rsi_ok}
        if not rsi_ok:
            if rsi > FILTER_RSI_MAX:
                reasons_blocked.append(f"RSI: {rsi:.1f} > {FILTER_RSI_MAX} (overbought)")
            else:
                reasons_blocked.append(f"RSI: {rsi:.1f} < {FILTER_RSI_MIN} (freefall)")

    # ── Filter 3: MACD ──
    if FILTER_MACD_ENABLED:
        macd_val = float(last.get("MACD", 0))
        macd_sig = float(last.get("MACD_Signal", 0))
        macd_ok = macd_val > macd_sig
        details["macd"] = {"macd": round(macd_val, 4), "signal": round(macd_sig, 4), "ok": macd_ok}
        if not macd_ok:
            reasons_blocked.append(f"MACD: {macd_val:.4f} < Signal {macd_sig:.4f} (bearish momentum)")

    # ── Filter 4: Volume ──
    if FILTER_VOLUME_ENABLED:
        vol_ratio = float(last.get("Volume_Ratio", 0))
        vol_ok = vol_ratio >= FILTER_VOLUME_MIN_RATIO
        details["volume"] = {"ratio": round(vol_ratio, 2), "min": FILTER_VOLUME_MIN_RATIO, "ok": vol_ok}
        if not vol_ok:
            reasons_blocked.append(f"Volume: {vol_ratio:.2f}x < {FILTER_VOLUME_MIN_RATIO}x (low participation)")

    # ── Filter 5: Bollinger Band Resistance ──
    if FILTER_BB_ENABLED:
        price = float(last.get("Close", 0))
        bb_upper = float(last.get("BB_Upper", 0))
        bb_lower = float(last.get("BB_Lower", 0))
        bb_range = bb_upper - bb_lower if bb_upper > bb_lower else 1
        bb_position = (price - bb_lower) / bb_range  # 0.0 = at lower, 1.0 = at upper
        bb_ok = bb_position < FILTER_BB_UPPER_PCT
        details["bb"] = {"position": round(bb_position, 3), "limit": FILTER_BB_UPPER_PCT, "ok": bb_ok}
        if not bb_ok:
            reasons_blocked.append(f"BB: Price at {bb_position:.0%} of band (near resistance)")

    if reasons_blocked:
        return False, " | ".join(reasons_blocked), details
    return True, "All filters passed", details


def count_stock_bullets_today(state: dict, symbol: str) -> int:
    """Count how many bullets have been fired for a specific stock today."""
    return sum(
        1 for t in state.get("active_trades", [])
        if t.get("stock") == symbol
    )


# --------------------------------------------------
#   MAIN SNIPER LOOP
# --------------------------------------------------
def run_sniper():
    Log.section("PROJECT AEGIS - THE SNIPER v2")
    print(f"   {Log.DIM}Capital  :{Log.RESET} {Log.BOLD}Rs.{CAPITAL:,}{Log.RESET}")
    print(f"   {Log.DIM}Bullets  :{Log.RESET} {MAX_BULLETS} (Rs.{CAPITAL/MAX_BULLETS:,.0f} ea.)")
    print(f"   {Log.DIM}Target   :{Log.RESET} {Log.GREEN}{DAILY_TARGET*100:.1f}% daily{Log.RESET}")
    print(f"   {Log.DIM}Voting   :{Log.RESET} {MIN_VOTES_TO_BUY}/4 agreement")
    print(f"   {Log.DIM}Guardian :{Log.RESET} {'✅ ENABLED' if RISK_GUARDIAN_ENABLED else '❌ DISABLED'}")
    print(f"   {Log.DIM}Time     :{Log.RESET} {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    #  RISK GUARDIAN — Pre-flight safety check
    # ═══════════════════════════════════════════════════════
    guardian = None
    if RISK_GUARDIAN_ENABLED:
        Log.info("Running pre-flight safety checks ...")
        allowed, reason = quick_safety_check()
        if not allowed:
            Log.error(f"TRADING BLOCKED: {reason}")
            return
        guardian = RiskGuardian(capital=CAPITAL)
        print(f"   [GUARDIAN] Pre-flight passed. Risk level: "
              f"{guardian.get_status().get('risk_level', 'NORMAL')}")

    # Load top-N stocks from Scholar ranking
    top_stocks = load_top_stocks()
    Log.highlight(f"Targeting: {', '.join(top_stocks)}")

    # Load models for each stock
    all_models = {}
    for sym in top_stocks:
        all_models[sym] = load_models(sym)

    state = load_state()
    state["stocks_traded"] = top_stocks
    bullet_size = CAPITAL / MAX_BULLETS
    last_fire_time = 0
    scan_interval = 60
    stock_idx = 0  # Round-robin through stocks

    while True:
        now = datetime.now(IST)

        # ---- Market Window Check ----
        if now.hour < 9 or (now.hour == 9 and now.minute < 15):
            print(f"   {Log.DIM}[WAIT]{Log.RESET} Market not open yet ({now.strftime('%H:%M IST')})", end="\r")
            time.sleep(60)
            continue

        # ---- Force-close before market close ----
        if (now.hour == 15 and now.minute >= 10) or now.hour >= 16:
            Log.highlight("MARKET CLOSING. FORCE-EXITING ALL POSITIONS.")
            for trade in state["active_trades"]:
                if trade["status"] != "OPEN":
                    continue
                sym = trade.get("stock", top_stocks[0])
                df = get_live_data(sym)
                if df is not None:
                    current_price = float(df["Close"].iloc[-1])
                    pnl = (current_price - trade["price"]) * trade["qty"]
                    state["total_profit"] += pnl
                    trade["status"] = "FORCE_CLOSED"
                    trade["exit_price"] = current_price
                    trade["pnl"] = round(pnl, 2)
                    state["trades_won" if pnl > 0 else "trades_lost"] += 1
                    log_trade({
                        "Date": now.strftime("%Y-%m-%d"),
                        "Time": now.strftime("%H:%M"),
                        "Stock": sym,
                        "Action": "FORCE_CLOSE",
                        "Entry_Price": trade["price"],
                        "Exit_Price": current_price,
                        "Qty": trade["qty"],
                        "Stop_Loss": trade["stop_loss"],
                        "Target": trade["target"],
                        "AI_Confidence": trade.get("confidence", 0),
                        "Votes": trade.get("votes", 0),
                        "Actual_Profit": round(pnl, 2),
                        "Status": "FORCE_CLOSED",
                    })

            state["status"] = "MARKET_CLOSED"
            state["total_profit_pct"] = round((state["total_profit"] / CAPITAL) * 100, 2)
            save_state(state)
            _print_summary(state)
            break

        # ---- Daily Target Check ----
        if state["total_profit"] >= (CAPITAL * DAILY_TARGET):
            state["status"] = "TARGET_HIT"
            state["total_profit_pct"] = round((state["total_profit"] / CAPITAL) * 100, 2)
            save_state(state)
            print(f"\n   [WIN] DAILY TARGET HIT!  +Rs.{state['total_profit']:.2f}  "
                  f"({state['total_profit_pct']:.2f}%)")
            _print_summary(state)
            break

        # ---- Kill-Switch ----
        if state["total_profit"] <= -(CAPITAL * MAX_DAILY_LOSS_PCT):
            state["status"] = "STOP_LOSS_HIT"
            save_state(state)
            print(f"\n   [STOP] MAX DAILY LOSS!  Rs.{state['total_profit']:.2f}")
            _print_summary(state)
            break

        # ---- Monitor Active Trades (with partial profit taking) ----
        for trade in state["active_trades"]:
            if trade["status"] != "OPEN":
                continue
            sym = trade.get("stock", top_stocks[0])
            df = get_live_data(sym)
            if df is None:
                continue

            current_price = float(df["Close"].iloc[-1])
            current_atr = float(df["ATR"].iloc[-1])
            pnl_per_share = current_price - trade["price"]
            pnl_total = pnl_per_share * trade["qty"]

            # Trailing stop
            new_stop = current_price - (ATR_STOP_MULTIPLIER * current_atr)
            if new_stop > trade["stop_loss"]:
                trade["stop_loss"] = round(new_stop, 2)

            # ── PARTIAL PROFIT TAKING ──
            # If price moved 1.5×ATR in our favor and we haven't taken partial yet
            partial_target = trade["price"] + (PARTIAL_EXIT_ATR_MULT * current_atr)
            if (PARTIAL_EXIT_ENABLED
                and current_price >= partial_target
                and not trade.get("partial_taken", False)
                and trade["qty"] > 1):
                # Take partial profit (exit ~50% of position)
                partial_qty = max(1, int(trade["qty"] * PARTIAL_EXIT_PCT))
                remaining_qty = trade["qty"] - partial_qty
                partial_pnl = pnl_per_share * partial_qty

                state["total_profit"] += partial_pnl
                trade["qty"] = remaining_qty
                trade["partial_taken"] = True
                # After partial, move stop to breakeven (entry price)
                trade["stop_loss"] = max(trade["stop_loss"], trade["price"])

                state["trades_taken"] += 1
                state["trades_won"] += 1

                Log.success(f"PARTIAL EXIT {sym}: {partial_qty} shares @ ₹{current_price:,.2f} | P&L: ₹{partial_pnl:,.2f}")

                log_trade({
                    "Date": now.strftime("%Y-%m-%d"),
                    "Time": now.strftime("%H:%M"),
                    "Stock": sym,
                    "Action": "PARTIAL_EXIT",
                    "Entry_Price": trade["price"],
                    "Exit_Price": current_price,
                    "Qty": partial_qty,
                    "Stop_Loss": trade["stop_loss"],
                    "Target": trade["target"],
                    "AI_Confidence": trade.get("confidence", 0),
                    "Votes": trade.get("votes", 0),
                    "Actual_Profit": round(partial_pnl, 2),
                    "Status": "PARTIAL_EXIT",
                })
                continue  # Check full exit next cycle

            if current_price >= trade["target"]:
                action = "TARGET_HIT"
            elif current_price <= trade["stop_loss"]:
                action = "STOP_LOSS"
            else:
                continue

            state["total_profit"] += pnl_total
            trade["status"] = action
            trade["exit_price"] = current_price
            trade["pnl"] = round(pnl_total, 2)
            state["trades_taken"] += 1
            state["trades_won" if pnl_total > 0 else "trades_lost"] += 1
            pnl_pct = (pnl_per_share / trade["price"]) * 100

            # ── Update Risk Guardian with trade result ──
            if guardian:
                guardian.record_trade_result(
                    symbol=sym, pnl=pnl_total, was_win=(pnl_total > 0)
                )
                if not guardian.guardian_active:
                    state["status"] = "GUARDIAN_STOP"
                    save_state(state)
                    Log.error("🚨 Trading HALTED by Risk Guardian")
                    _print_summary(state)
                    return

            result_icon = f"{Log.GREEN}WIN{Log.RESET}" if pnl_total > 0 else f"{Log.RED}LOSS{Log.RESET}"
            print(f"   {Log.DIM}«{Log.RESET} [{result_icon}] {Log.BOLD}{sym:10}{Log.RESET} | "
                  f"Exit: Rs.{current_price:,.2f} | P&L: {Log.BOLD}Rs.{pnl_total:,.2f}{Log.RESET} "
                  f"({Log.GREEN if pnl_total > 0 else Log.RED}{pnl_pct:+.2f}%{Log.RESET})")

            log_trade({
                "Date": now.strftime("%Y-%m-%d"),
                "Time": now.strftime("%H:%M"),
                "Stock": sym,
                "Action": action,
                "Entry_Price": trade["price"],
                "Exit_Price": current_price,
                "Qty": trade["qty"],
                "Stop_Loss": trade["stop_loss"],
                "Target": trade["target"],
                "AI_Confidence": trade.get("confidence", 0),
                "Votes": trade.get("votes", 0),
                "Actual_Profit": round(pnl_total, 2),
                "Status": action,
            })

        # ---- Scan ALL stocks & write LIVE ANALYSIS ----
        analysis = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "scan_number": getattr(run_sniper, '_scan_count', 0) + 1,
            "status": state["status"],
            "day_pnl": round(state["total_profit"], 2),
            "day_pnl_pct": round((state["total_profit"] / CAPITAL) * 100, 2),
            "open_positions": sum(1 for t in state["active_trades"] if t.get("status") == "OPEN"),
            "bullets_left": MAX_BULLETS - sum(1 for t in state["active_trades"] if t.get("status") == "OPEN"),
            "stocks": [],
        }
        run_sniper._scan_count = analysis["scan_number"]

        for scan_sym in top_stocks:
            stock_analysis = {
                "symbol": scan_sym,
                "name": scan_sym.replace(".NS", "").replace(".BO", ""),
                "price": 0.0, "change_pct": 0.0,
                "rf_conf": 0.0, "xgb_conf": 0.0, "lstm_conf": 0.0, "intra_conf": 0.0,
                "avg_conf": 0.0, "votes": 0,
                "signal": "WAIT", "reason": "",
                "guardian": "", "guardian_reason": "",
                "rsi": 0.0, "atr": 0.0, "macd": 0.0, "sma_50": 0.0, "sma_200": 0.0,
                "sentiment": 0.0, "volume_ratio": 0.0,
                "stop_loss": 0.0, "target": 0.0,
                "filters": {},  # Strategy filter details
            }
            try:
                scan_df = get_live_data(scan_sym)
                if scan_df is not None and not scan_df.empty:
                    cur_price = float(scan_df["Close"].iloc[-1])
                    prev_price = float(scan_df["Close"].iloc[-2]) if len(scan_df) > 1 else cur_price
                    stock_analysis["price"] = round(cur_price, 2)
                    stock_analysis["change_pct"] = round(((cur_price - prev_price) / prev_price) * 100, 2) if prev_price > 0 else 0.0

                    # Technical indicators from latest row
                    last = scan_df.iloc[-1]
                    stock_analysis["rsi"] = round(float(last.get("RSI", 0)), 2)
                    stock_analysis["atr"] = round(float(last.get("ATR", 0)), 2)
                    stock_analysis["macd"] = round(float(last.get("MACD", 0)), 4)
                    stock_analysis["sma_50"] = round(float(last.get("SMA_50", 0)), 2)
                    stock_analysis["sma_200"] = round(float(last.get("SMA_200", 0)), 2)
                    stock_analysis["sentiment"] = round(float(last.get("Sentiment_Score", 0)), 3)
                    stock_analysis["volume_ratio"] = round(float(last.get("Volume_Ratio", 0)), 2)

                    # Run ensemble (only if models loaded)
                    if scan_sym in all_models:
                        should_buy, votes, rf_c, xgb_c, lstm_c, intra_c = get_ensemble_signal(
                            scan_sym, scan_df, all_models[scan_sym]
                        )
                        stock_analysis["rf_conf"] = round(rf_c, 4)
                        stock_analysis["xgb_conf"] = round(xgb_c, 4)
                        stock_analysis["lstm_conf"] = round(lstm_c, 4)
                        stock_analysis["intra_conf"] = round(intra_c, 4)
                        stock_analysis["avg_conf"] = round((rf_c + xgb_c + lstm_c + intra_c) / 4, 4)
                        stock_analysis["votes"] = votes

                        if should_buy:
                            stock_analysis["signal"] = "BUY"
                            stock_analysis["reason"] = f"{votes}/4 models agree"
                            # Calculate stop/target for display
                            cur_atr = float(last.get("ATR", 0))
                            if cur_atr > 0:
                                stock_analysis["stop_loss"] = round(cur_price - ATR_STOP_MULTIPLIER * cur_atr, 2)
                                stock_analysis["target"] = round(cur_price + ATR_TARGET_MULTIPLIER * cur_atr, 2)

                            # Lightweight Guardian pre-check (doesn't modify state)
                            avg_conf_check = (rf_c + xgb_c + lstm_c + intra_c) / 4
                            from risk_guardian import MIN_CONFIDENCE_REAL_MONEY, MIN_VOTES_REAL_MONEY
                            if avg_conf_check < MIN_CONFIDENCE_REAL_MONEY:
                                stock_analysis["guardian"] = "BLOCKED"
                                stock_analysis["guardian_reason"] = f"Avg conf {avg_conf_check:.2f} < {MIN_CONFIDENCE_REAL_MONEY}"
                            elif votes < MIN_VOTES_REAL_MONEY:
                                stock_analysis["guardian"] = "BLOCKED"
                                stock_analysis["guardian_reason"] = f"Votes {votes} < {MIN_VOTES_REAL_MONEY}"
                            else:
                                stock_analysis["guardian"] = "APPROVED"
                                stock_analysis["guardian_reason"] = "All checks passed"
                        else:
                            stock_analysis["signal"] = "WAIT"
                            stock_analysis["guardian"] = "N/A"
                            if votes < MIN_VOTES_TO_BUY:
                                stock_analysis["reason"] = f"Only {votes}/{MIN_VOTES_TO_BUY} votes"
                                stock_analysis["guardian_reason"] = "Not enough model consensus"
                            else:
                                stock_analysis["reason"] = "Below confidence threshold"
                                stock_analysis["guardian_reason"] = "Confidence too low"
            except Exception as e:
                stock_analysis["reason"] = f"Error: {str(e)[:50]}"

            analysis["stocks"].append(stock_analysis)

        save_analysis(analysis)

        # ---- Fire New Bullet (round-robin through stocks) ----
        open_count = sum(1 for t in state["active_trades"] if t["status"] == "OPEN")
        time_since_last = time.time() - last_fire_time

        if open_count < MAX_BULLETS and time_since_last >= TIME_GAP:
            sym = top_stocks[stock_idx % len(top_stocks)]
            stock_idx += 1

            # ── Per-Stock Cooldown: max N bullets per stock per day ──
            stock_bullets = count_stock_bullets_today(state, sym)
            if stock_bullets >= MAX_BULLETS_PER_STOCK:
                Log.warn(f"{sym}: Already {stock_bullets}/{MAX_BULLETS_PER_STOCK} bullets today. Skipping.")
            elif sym in all_models:
                df = get_live_data(sym)
                if df is not None and not df.empty:
                    should_buy, votes, rf_c, xgb_c, lstm_c, intra_c = get_ensemble_signal(
                        sym, df, all_models[sym]
                    )
                    avg_conf = round((rf_c + xgb_c + lstm_c + intra_c) / 4, 3)

                    if should_buy:
                        current_price = float(df["Close"].iloc[-1])
                        current_atr = float(df["ATR"].iloc[-1])

                        # ═══════════════════════════════════════════
                        #  STRATEGY FILTER GATE — 7 Technical checks
                        # ═══════════════════════════════════════════
                        filters_ok, filter_reason, filter_details = check_strategy_filters(sym, df)
                        if not filters_ok:
                            Log.warn(f"{sym}: BLOCKED by Strategy Filters — {filter_reason}")
                            # Update analysis data with filter rejection
                            for s_entry in analysis.get("stocks", []):
                                if s_entry.get("symbol") == sym:
                                    s_entry["guardian"] = "BLOCKED"
                                    s_entry["guardian_reason"] = f"Strategy: {filter_reason}"
                            continue

                        Log.info(f"{sym}: Strategy filters PASSED ✅")

                        pos = calculate_position(current_price, current_atr, bullet_size)
                        if pos:
                            # ═══════════════════════════════════════════
                            #  RISK GUARDIAN GATE — Must approve trade
                            # ═══════════════════════════════════════════
                            final_qty = pos["qty"]
                            if guardian:
                                approved, g_reason, adj_qty = guardian.approve_trade(
                                    symbol=sym,
                                    price=current_price,
                                    qty=pos["qty"],
                                    stop_loss=pos["stop_loss"],
                                    target=pos["target"],
                                    atr=current_atr,
                                    rf_conf=rf_c, xgb_conf=xgb_c,
                                    lstm_conf=lstm_c, intra_conf=intra_c,
                                    votes=votes,
                                )
                                if not approved:
                                    Log.error(f"{sym}: REJECTED by Guardian - {g_reason}")
                                    continue
                                final_qty = adj_qty
                                Log.success(f"{sym}: APPROVED (qty adjusted: {pos['qty']} → {adj_qty})")

                            Log.highlight(f"FIRE BULLET on {sym} @ Rs.{current_price:,.2f}")
                            print(f"     {Log.DIM}Qty: {final_qty} | SL: {pos['stop_loss']} | Target: {pos['target']} | Votes: {votes}/4{Log.RESET}")

                            trade_entry = {
                                "stock": sym,
                                "price": current_price,
                                "qty": final_qty,
                                "stop_loss": pos["stop_loss"],
                                "target": pos["target"],
                                "status": "OPEN",
                                "time": now.strftime("%H:%M"),
                                "confidence": avg_conf,
                                "votes": votes,
                            }
                            state["active_trades"].append(trade_entry)
                            last_fire_time = time.time()

                            # Register with guardian
                            if guardian:
                                guardian.register_position(sym, final_qty, current_price)

                            log_trade({
                                "Date": now.strftime("%Y-%m-%d"),
                                "Time": now.strftime("%H:%M"),
                                "Stock": sym,
                                "Action": "BUY",
                                "Entry_Price": current_price,
                                "Exit_Price": 0,
                                "Qty": final_qty,
                                "Stop_Loss": pos["stop_loss"],
                                "Target": pos["target"],
                                "AI_Confidence": avg_conf,
                                "Votes": votes,
                                "Actual_Profit": 0,
                                "Status": "OPEN",
                            })

        # ---- Periodic Status ----
        total_pnl_pct = round((state["total_profit"] / CAPITAL) * 100, 2)
        pnl_color = Log.GREEN if state['total_profit'] >= 0 else Log.RED
        print(f"   {Log.DIM}[SCAN] {now.strftime('%H:%M IST')} | Day P&L: {pnl_color}Rs.{state['total_profit']:,.2f} ({total_pnl_pct}%){Log.RESET} | Open: {open_count}/{MAX_BULLETS}", end="\r")

        save_state(state)
        time.sleep(scan_interval)


# --------------------------------------------------
#   END-OF-DAY SUMMARY
# --------------------------------------------------
def _print_summary(state: dict):
    Log.section("END-OF-DAY SUMMARY")
    print(f"   {Log.DIM}Status       :{Log.RESET} {Log.BOLD}{state['status']}{Log.RESET}")
    print(f"   {Log.DIM}Stocks       :{Log.RESET} {', '.join(state.get('stocks_traded', []))}")
    print(f"   {Log.DIM}Total Trades :{Log.RESET} {state['trades_taken']}")
    print(f"   {Log.DIM}Wins         :{Log.RESET} {Log.GREEN}{state['trades_won']}{Log.RESET}")
    print(f"   {Log.DIM}Losses       :{Log.RESET} {Log.RED}{state['trades_lost']}{Log.RESET}")
    
    win_rate = (state["trades_won"] / state["trades_taken"] * 100) if state["trades_taken"] > 0 else 0
    pnl_color = Log.GREEN if state['total_profit'] >= 0 else Log.RED
    
    print(f"   {Log.DIM}Win Rate     :{Log.RESET} {Log.BOLD}{win_rate:.1f}%{Log.RESET}")
    print(f"   {Log.DIM}Total P&L    :{Log.RESET} {pnl_color}{Log.BOLD}Rs.{state['total_profit']:,.2f}{Log.RESET} ({state.get('total_profit_pct', 0):.2f}%)")
    print(f"{Log.MAGENTA}╚{'═' * 58}╝{Log.RESET}")


if __name__ == "__main__":
    run_sniper()
