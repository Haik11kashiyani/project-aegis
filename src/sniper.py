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
    # Trading Brain (new)
    TRADING_PERSONALITY, SMART_EXIT_ENABLED, SMART_ENTRY_ENABLED,
    GLOBAL_MOOD_ENABLED,
    # Phase 6 modules
    BROKER_MODE, BROKER_NAME,
    SECTOR_ROTATION_ENABLED, MARKET_BREADTH_ENABLED,
    CORRELATION_GUARD_ENABLED, INTRADAY_PATTERNS_ENABLED,
    REPORT_AUTO_GENERATE,
    # Phase 7 modules
    KELLY_SIZING_ENABLED, EARNINGS_GUARD_ENABLED,
    MODEL_DRIFT_ENABLED, TELEGRAM_BOT_ENABLED,
    # Phase 8 modules
    OPTIONS_HEDGE_ENABLED, MULTI_TIMEFRAME_ENABLED,
    NEWS_DETECTOR_ENABLED, REBALANCER_ENABLED,
    RISK_PARITY_ENABLED, AB_BACKTEST_ENABLED,
    # Phase 9 modules
    REGIME_DETECTOR_ENABLED, VAR_STRESS_ENABLED,
    FINBERT_ENABLED, EXECUTION_QUALITY_ENABLED,
    OPTION_CHAIN_ENABLED, VOLUME_PROFILE_ENABLED,
    SMART_ALERTS_ENABLED, TRADE_JOURNAL_ENABLED,
    # Phase 10 modules
    RL_SIZER_ENABLED, INTERMARKET_ENABLED,
    LIQUIDITY_FILTER_ENABLED, GREEKS_HEATMAP_ENABLED,
    BAYESIAN_FUSION_ENABLED, INTRADAY_SCALPER_ENABLED,
    AUTO_TUNER_ENABLED,
    # Phase 11 modules
    TRANSFORMER_ENABLED, ORDERBOOK_SIM_ENABLED,
    DYNAMIC_STOPLOSS_ENABLED, PAIR_TRADING_ENABLED,
    ANOMALY_DETECTOR_ENABLED, MODEL_VERSIONING_ENABLED,
    # Phase 12 modules
    ADAPTIVE_EXECUTOR_ENABLED, RL_REBALANCER_ENABLED,
    OPTIONS_SYNTH_ENABLED, CAUSAL_ENGINE_ENABLED,
    # Phase 13 modules
    GA_EVOLVER_ENABLED, DEBATE_SYSTEM_ENABLED,
    # Phase 14 modules
    RL_TRADE_AGENT_ENABLED, SENTIMENT_MOMENTUM_ENABLED,
)
from sentiment import get_sentiment_score, get_global_market_mood, get_macro_sentiment
from risk_guardian import RiskGuardian, NeuralSafetyNet, quick_safety_check
from trading_brain import TradingBrain
from neuro_voter import NeuroVoterEnsemble
from notifier import (
    alert_trade_buy, alert_trade_exit, alert_veto,
    alert_guardian_stop, alert_daily_summary, alert_regime_change,
)
from broker_bridge import create_bridge
from sector_rotation import analyse_sector_rotation, get_sector_recommendations
from market_breadth import analyse_market_breadth
from correlation_guard import check_before_buy, analyse_correlations
from intraday_patterns import should_delay_entry, get_current_window
from report_exporter import generate_report
from kelly_sizing import get_kelly_position_size, compute_stock_kelly_map, save_kelly_state
from earnings_guard import check_earnings_guard, refresh_earnings_calendar
from model_drift import log_prediction, update_prediction_actuals, get_drift_penalty, analyse_model_drift
from telegram_bot import get_bot as get_telegram_bot
from options_hedge import check_hedge_gate, compute_hedge_requirements, save_hedge_state
from multi_timeframe import check_multi_timeframe_gate, save_mtf_state
from news_detector import check_news_gate, scan_all_news, save_news_state
from rebalancer import run_rebalance, should_rebalance, save_rebalance_state, load_rebalance_state
from risk_parity import analyse_risk_parity, get_risk_parity_size, compute_risk_parity_weights, save_rp_state
from regime_detector import detect_regime, get_regime_gate, get_regime_sizing, save_regime_state
from var_stress import analyse_var_stress, save_var_state
from finbert_sentiment import get_finbert_gate, analyse_all_stocks as finbert_scan, save_finbert_state
from execution_quality import analyse_execution_quality, log_execution, save_eq_state
from option_chain import analyse_option_chain, save_oc_state
from volume_profile import check_volume_gate, analyse_all_volume, save_vp_state
from smart_alerts import (
    enqueue_alert, alert_trade_opened, alert_trade_closed,
    alert_drawdown, alert_guardian_halt, alert_regime_shift,
    alert_daily_digest, detect_anomalies, dispatch_alerts,
    Severity,
)
from trade_journal import generate_journal, save_journal_state
from rl_position_sizer import get_rl_position_size, record_trade_result, batch_train as rl_batch_train, save_rl_state
from intermarket_engine import analyse_intermarket, get_intermarket_gate, get_exposure_multiplier, save_intermarket_state
from liquidity_filter import check_liquidity_gate, get_liquidity_sizing_factor, compute_liquidity_score, save_liquidity_state
from greeks_heatmap import analyse_portfolio_greeks, save_greeks_state
from bayesian_fusion import bayesian_fuse, record_bayesian_outcome, save_bayesian_state
from intraday_scalper import scan_scalp_opportunities, record_scalp, save_scalper_state
from auto_tuner import run_tuning_sweep, get_tuner_status, save_tuner_state
from portfolio_transformer import predict_portfolio, check_transformer_gate, save_transformer_state, load_transformer_state
from orderbook_simulator import analyse_orderbook, check_orderbook_gate, save_orderbook_state, load_orderbook_state
from dynamic_stoploss import init_stop, update_trailing_stop, check_dynamic_exit, remove_stop, get_all_stop_states, save_stoploss_state, load_stoploss_state, compute_chandelier_exit
from pair_trading import analyse_pairs, check_pair_signal, save_pairs_state, load_pairs_state
from anomaly_detector import detect_anomalies as detect_anomalies_iforest, check_anomaly_gate, train_isolation_forest, load_anomaly_models, save_anomaly_state, load_anomaly_state
from model_versioning import register_model_version, deploy_shadow, shadow_predict, auto_promote_check, get_versioning_status, save_versioning_state, load_versioning_state
from adaptive_executor import should_slice_order, create_execution_plan, execute_next_slice, get_executor_status, save_executor_state, load_executor_state
from rl_rebalancer import get_rl_rebalance_action, compute_rl_weights, record_rebalance_outcome, train_rl_rebalancer, save_rl_rebal_state, load_rl_rebal_state, get_rebalancer_status
from options_synthesizer import synthesize_strategy, get_protection_cost, get_synth_status, save_synth_state, load_synth_state
from causal_engine import discover_causal_graph, get_causal_features, filter_spurious_signals, get_causal_status, save_causal_state, load_causal_state
from genetic_evolver import evolve_strategies, get_best_strategy, get_evolver_status, save_evolver_state, load_evolver_state
from debate_system import run_debate, get_debate_verdict, record_debate_result, get_debate_status, save_debate_state, load_debate_state
from rl_trade_agent import get_rl_action, record_rl_experience, batch_train_agent, record_rl_trade, get_agent_status, save_agent_state, load_agent_state
from sentiment_momentum import compute_smi, compute_market_smi, check_smi_gate, get_smi_sizing_factor, get_smi_status, record_smi_snapshot, save_smi_state, load_smi_state

IST = pytz.timezone("Asia/Kolkata")

# Voter history file for accuracy tracking
VOTER_HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "voter_history.json")

def save_voter_history(symbol: str, decision, trade_action: str, price: float):
    """Save each voter's decision alongside the trade for accuracy tracking."""
    try:
        history = []
        if os.path.exists(VOTER_HISTORY_FILE):
            with open(VOTER_HISTORY_FILE, "r") as f:
                history = json.load(f)

        entry = {
            "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": trade_action,
            "price": price,
            "neuro_score": round(decision.weighted_score, 3),
            "conviction": round(decision.total_conviction, 3),
            "regime": decision.regime,
            "should_buy": decision.should_buy,
            "vetoed": decision.vetoed,
            "voters": [
                {
                    "name": vr.name,
                    "vote": round(vr.vote, 3),
                    "conviction": round(vr.conviction, 3),
                    "veto": vr.veto,
                }
                for vr in decision.voter_results
            ],
        }
        history.append(entry)

        # Keep last 2000 entries
        if len(history) > 2000:
            history = history[-2000:]

        os.makedirs(os.path.dirname(VOTER_HISTORY_FILE), exist_ok=True)
        with open(VOTER_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception:
        pass

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
    Log.section("PROJECT AEGIS - THE SNIPER v3 (Brain Edition)")
    print(f"   {Log.DIM}Capital  :{Log.RESET} {Log.BOLD}Rs.{CAPITAL:,}{Log.RESET}")
    print(f"   {Log.DIM}Bullets  :{Log.RESET} {MAX_BULLETS} (Rs.{CAPITAL/MAX_BULLETS:,.0f} ea.)")
    print(f"   {Log.DIM}Target   :{Log.RESET} {Log.GREEN}{DAILY_TARGET*100:.1f}% daily{Log.RESET}")
    print(f"   {Log.DIM}Voting   :{Log.RESET} {MIN_VOTES_TO_BUY}/4 agreement")
    print(f"   {Log.DIM}Guardian :{Log.RESET} {'✅ ENABLED' if RISK_GUARDIAN_ENABLED else '❌ DISABLED'}")
    print(f"   {Log.DIM}Brain    :{Log.RESET} {TRADING_PERSONALITY} | SmartExit: {'✅' if SMART_EXIT_ENABLED else '❌'} | SmartEntry: {'✅' if SMART_ENTRY_ENABLED else '❌'}")
    print(f"   {Log.DIM}Mood     :{Log.RESET} {'✅ Global events tracked' if GLOBAL_MOOD_ENABLED else '❌ DISABLED'}")
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

    # ═══════════════════════════════════════════════════════
    #  TRADING BRAIN — Human-like decision engine
    # ═══════════════════════════════════════════════════════
    brain = TradingBrain(personality=TRADING_PERSONALITY, capital=CAPITAL)
    print(f"   [BRAIN] Personality: {TRADING_PERSONALITY} | "
          f"Smart Entry: {'ON' if SMART_ENTRY_ENABLED else 'OFF'} | "
          f"Smart Exit: {'ON' if SMART_EXIT_ENABLED else 'OFF'}")

    # ═══════════════════════════════════════════════════════
    #  NEURO-VOTER ENSEMBLE — 6 Human-like AI Voters
    # ═══════════════════════════════════════════════════════
    neuro_ensemble = NeuroVoterEnsemble()
    print(f"   [VOTERS] 6 AI voters active: "
          f"{', '.join(v.name for v in neuro_ensemble.voters)}")
    print(f"   [VOTERS] Anti-overfit: ON | Regime-aware: ON")

    # ═══════════════════════════════════════════════════════
    #  BROKER BRIDGE — Paper / DryRun / Live Order Router
    # ═══════════════════════════════════════════════════════
    broker = create_bridge()
    print(f"   [BROKER] Mode: {broker.mode.value} | Broker: {broker.broker.__class__.__name__}")

    # ═══════════════════════════════════════════════════════
    #  MARKET BREADTH — Macro Health (refreshed every 15 min)
    # ═══════════════════════════════════════════════════════
    breadth_cache = {"data": {}, "last_update": 0, "size_factor": 1.0}
    if MARKET_BREADTH_ENABLED:
        try:
            breadth_cache["data"] = analyse_market_breadth()
            breadth_cache["size_factor"] = breadth_cache["data"].get("position_size_factor", 1.0)
            breadth_cache["last_update"] = time.time()
            score = breadth_cache["data"].get("composite_score", 50)
            sig = breadth_cache["data"].get("signal", "UNKNOWN")
            print(f"   [BREADTH] Market: {sig} (score={score}) | Size factor: {breadth_cache['size_factor']:.0%}")
        except Exception as e:
            Log.warn(f"[BREADTH] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  SECTOR ROTATION — Sector Momentum Tracking
    # ═══════════════════════════════════════════════════════
    sector_cache = {"data": {}, "last_update": 0}
    if SECTOR_ROTATION_ENABLED:
        try:
            sector_cache["data"] = analyse_sector_rotation()
            sector_cache["last_update"] = time.time()
            hot = [s for s, d in sector_cache["data"].get("sectors", {}).items()
                   if d.get("state") in ("STRONG", "ROTATING_IN")]
            cold = [s for s, d in sector_cache["data"].get("sectors", {}).items()
                    if d.get("state") in ("ROTATING_OUT", "WEAKENING")]
            print(f"   [SECTOR] Hot: {', '.join(hot) or 'None'} | Cold: {', '.join(cold) or 'None'}")
        except Exception as e:
            Log.warn(f"[SECTOR] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  CORRELATION GUARD — Pre-compute correlation matrix
    # ═══════════════════════════════════════════════════════
    corr_cache = {"data": {}, "last_update": 0}
    if CORRELATION_GUARD_ENABLED:
        try:
            corr_cache["data"] = analyse_correlations()
            corr_cache["last_update"] = time.time()
            div_score = corr_cache["data"].get("diversification_score", "N/A")
            print(f"   [CORR] Diversification score: {div_score}")
        except Exception as e:
            Log.warn(f"[CORR] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  KELLY SIZING — Adaptive position sizing (Phase 7)
    # ═══════════════════════════════════════════════════════
    kelly_cache = {"map": {}, "last_update": 0}
    if KELLY_SIZING_ENABLED:
        try:
            kelly_cache["map"] = compute_stock_kelly_map()
            kelly_cache["last_update"] = time.time()
            save_kelly_state(kelly_cache["map"])
            kt = len(kelly_cache["map"])
            print(f"   [KELLY] Loaded Kelly fractions for {kt} stocks")
        except Exception as e:
            Log.warn(f"[KELLY] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  EARNINGS GUARD — Block near quarterly results (Phase 7)
    # ═══════════════════════════════════════════════════════
    earnings_cache = {"calendar": {}, "last_update": 0}
    if EARNINGS_GUARD_ENABLED:
        try:
            earnings_cache["calendar"] = refresh_earnings_calendar(STOCK_WATCHLIST[:TOP_N_STOCKS])
            earnings_cache["last_update"] = time.time()
            blocked = [s for s, c in earnings_cache["calendar"].items()
                       if check_earnings_guard(s, earnings_cache["calendar"]).get("blocked")]
            if blocked:
                print(f"   [EARNINGS] ⚠️ Blocked near earnings: {', '.join(blocked)}")
            else:
                print(f"   [EARNINGS] All clear — no stocks near results")
        except Exception as e:
            Log.warn(f"[EARNINGS] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  MODEL DRIFT — Check prediction distribution (Phase 7)
    # ═══════════════════════════════════════════════════════
    if MODEL_DRIFT_ENABLED:
        try:
            drift_result = analyse_model_drift()
            retrain = drift_result.get("retrain_needed", [])
            if retrain:
                Log.warn(f"[DRIFT] 🚨 Models need retraining: {', '.join(retrain)}")
            else:
                print(f"   [DRIFT] All models stable ✅")
        except Exception as e:
            Log.warn(f"[DRIFT] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  TELEGRAM BOT — Start background listener (Phase 7)
    # ═══════════════════════════════════════════════════════
    tg_bot = None
    if TELEGRAM_BOT_ENABLED:
        try:
            tg_bot = get_telegram_bot()
            started = tg_bot.start(background=True)
            if started:
                print(f"   [TELEGRAM] Bot started in background ✅")
            else:
                print(f"   [TELEGRAM] Bot disabled (no token)")
        except Exception as e:
            Log.warn(f"[TELEGRAM] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  RISK PARITY — Inverse-vol allocation (Phase 8)
    # ═══════════════════════════════════════════════════════
    rp_cache = {"weights": {}, "last_update": 0}
    if RISK_PARITY_ENABLED:
        try:
            rp_result = analyse_risk_parity(
                STOCK_WATCHLIST[:TOP_N_STOCKS], CAPITAL
            )
            rp_cache["weights"] = rp_result.get("weights", {})
            rp_cache["result"] = rp_result
            rp_cache["last_update"] = time.time()
            save_rp_state(rp_result)
            print(f"   [RISK PARITY] Parity score: {rp_result.get('parity_score', 0):.0f}/100")
        except Exception as e:
            Log.warn(f"[RISK PARITY] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  NEWS DETECTOR — Scan headlines at startup (Phase 8)
    # ═══════════════════════════════════════════════════════
    news_cache = {"data": {}, "last_update": 0}
    if NEWS_DETECTOR_ENABLED:
        try:
            news_cache["data"] = scan_all_news(STOCK_WATCHLIST[:TOP_N_STOCKS])
            news_cache["last_update"] = time.time()
            save_news_state(news_cache["data"])
            summary = news_cache["data"].get("summary", {})
            blocked = summary.get("blocked", 0)
            boosted = summary.get("boosted", 0)
            if blocked:
                Log.warn(f"[NEWS] ⚠️ {blocked} stocks have blocking events")
            else:
                print(f"   [NEWS] All clear — {boosted} boosted, 0 blocked")
        except Exception as e:
            Log.warn(f"[NEWS] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  REBALANCER — Check weekly rebalance (Phase 8)
    # ═══════════════════════════════════════════════════════
    if REBALANCER_ENABLED:
        try:
            last_state = load_rebalance_state()
            should, reason = should_rebalance(last_state.get("timestamp"))
            if should:
                print(f"   [REBALANCE] Weekly rebalance due — {reason}")
            else:
                print(f"   [REBALANCE] {reason}")
        except Exception as e:
            Log.warn(f"[REBALANCE] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  REGIME DETECTOR — HMM Bull/Bear/Sideways (Phase 9)
    # ═══════════════════════════════════════════════════════
    regime_cache = {"data": {}, "last_update": 0}
    if REGIME_DETECTOR_ENABLED:
        try:
            regime_cache["data"] = detect_regime()
            regime_cache["last_update"] = time.time()
            save_regime_state(regime_cache["data"])
            regime = regime_cache["data"].get("regime", "SIDEWAYS")
            r_conf = regime_cache["data"].get("confidence", 0)
            override = regime_cache["data"].get("personality_override")
            sizing_mult = regime_cache["data"].get("sizing_multiplier", 1.0)
            print(f"   [REGIME] {regime} (conf={r_conf:.0%}) | "
                  f"Sizing: {sizing_mult:.0%} | Override: {override or 'None'}")
            # Auto-switch brain personality if regime suggests it
            if override and SMART_ENTRY_ENABLED:
                brain.personality = override
                print(f"   [REGIME] Brain personality → {override}")
        except Exception as e:
            Log.warn(f"[REGIME] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  VaR & STRESS TEST — Portfolio risk (Phase 9)
    # ═══════════════════════════════════════════════════════
    if VAR_STRESS_ENABLED:
        try:
            var_result = analyse_var_stress(
                symbols=STOCK_WATCHLIST[:TOP_N_STOCKS], capital=CAPITAL,
            )
            save_var_state(var_result)
            risk_lvl = var_result.get("risk_level", "UNKNOWN")
            risk_scr = var_result.get("risk_score", 0)
            var_95 = var_result.get("monte_carlo_var_95", {}).get("var_pct", 0)
            print(f"   [VaR] Risk: {risk_lvl} (score={risk_scr}/100) | "
                  f"MC-VaR(95%): {var_95:.2f}%")
        except Exception as e:
            Log.warn(f"[VaR] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  FINBERT SENTIMENT — Transformer NLP (Phase 9)
    # ═══════════════════════════════════════════════════════
    finbert_cache = {"data": {}, "last_update": 0}
    if FINBERT_ENABLED:
        try:
            finbert_cache["data"] = finbert_scan(STOCK_WATCHLIST[:TOP_N_STOCKS])
            finbert_cache["last_update"] = time.time()
            save_finbert_state(finbert_cache["data"])
            mkt = finbert_cache["data"].get("market_label", "NEUTRAL")
            method = finbert_cache["data"].get("method", "keyword")
            print(f"   [FINBERT] Market: {mkt} | Method: {method}")
        except Exception as e:
            Log.warn(f"[FINBERT] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  VOLUME PROFILE — Order flow analysis (Phase 9)
    # ═══════════════════════════════════════════════════════
    volume_cache = {"data": {}, "last_update": 0}
    if VOLUME_PROFILE_ENABLED:
        try:
            volume_cache["data"] = analyse_all_volume(STOCK_WATCHLIST[:TOP_N_STOCKS])
            volume_cache["last_update"] = time.time()
            save_vp_state(volume_cache["data"])
            mkt_pressure = volume_cache["data"].get("market_pressure", "NEUTRAL")
            print(f"   [VOLUME] Market pressure: {mkt_pressure}")
        except Exception as e:
            Log.warn(f"[VOLUME] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  OPTION CHAIN — Live NSE data (Phase 9)
    # ═══════════════════════════════════════════════════════
    if OPTION_CHAIN_ENABLED:
        try:
            oc_data = analyse_option_chain(STOCK_WATCHLIST[:TOP_N_STOCKS])
            save_oc_state(oc_data)
            nse_hits = oc_data.get("nse_live", 0)
            bs_fb = oc_data.get("bs_fallback", 0)
            print(f"   [OC] Option chains: {nse_hits} live NSE + {bs_fb} BS fallback")
        except Exception as e:
            Log.warn(f"[OC] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  RL POSITION SIZER — Q-Learning batch train (Phase 10)
    # ═══════════════════════════════════════════════════════
    if RL_SIZER_ENABLED:
        try:
            rl_stats = rl_batch_train()
            episodes = rl_stats.get("episodes_trained", 0)
            states_n = rl_stats.get("states_explored", 0)
            print(f"   [RL] Batch trained: {episodes} episodes, {states_n} states")
        except Exception as e:
            Log.warn(f"[RL] Batch train failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  INTERMARKET ENGINE — Gold/USD/Crude/US10Y (Phase 10)
    # ═══════════════════════════════════════════════════════
    intermarket_cache = {"data": {}, "last_update": 0}
    if INTERMARKET_ENABLED:
        try:
            intermarket_cache["data"] = analyse_intermarket()
            intermarket_cache["last_update"] = time.time()
            save_intermarket_state(intermarket_cache["data"])
            mult = intermarket_cache["data"].get("exposure_multiplier", 1.0)
            status = intermarket_cache["data"].get("status", "NORMAL")
            print(f"   [MACRO] Intermarket: {status} | Exposure mult: {mult:.2f}")
        except Exception as e:
            Log.warn(f"[MACRO] Intermarket init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  LIQUIDITY FILTER — Spread + Volume (Phase 10)
    # ═══════════════════════════════════════════════════════
    liquidity_cache = {"data": {}, "last_update": 0}
    if LIQUIDITY_FILTER_ENABLED:
        try:
            liq_results = {}
            for _sym in STOCK_WATCHLIST[:TOP_N_STOCKS]:
                _ok, _reason, _ldata = check_liquidity_gate(_sym)
                liq_results[_sym] = {"ok": _ok, "reason": _reason, "score": _ldata.get("composite_score", 0)}
            liquidity_cache["data"] = liq_results
            liquidity_cache["last_update"] = time.time()
            save_liquidity_state(liq_results)
            blocked_ct = sum(1 for v in liq_results.values() if not v["ok"])
            print(f"   [LIQUIDITY] Scanned {len(liq_results)} stocks, {blocked_ct} blocked")
        except Exception as e:
            Log.warn(f"[LIQUIDITY] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  BAYESIAN FUSION — Posterior signal weights (Phase 10)
    # ═══════════════════════════════════════════════════════
    bayesian_cache = {"last_update": 0}
    if BAYESIAN_FUSION_ENABLED:
        try:
            # Just initialize — actual fusion happens at trade time
            from bayesian_fusion import BayesianFusion
            _bf = BayesianFusion()
            n_sources = len(_bf.sources)
            print(f"   [BAYES] Fusion engine ready: {n_sources} sources loaded")
        except Exception as e:
            Log.warn(f"[BAYES] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  AUTO TUNER — Check last sweep results (Phase 10)
    # ═══════════════════════════════════════════════════════
    if AUTO_TUNER_ENABLED:
        try:
            tuner_status = get_tuner_status()
            if tuner_status.get("status") == "COMPLETED":
                best_sharpe = tuner_status.get("best_sharpe", 0)
                print(f"   [TUNER] Last sweep: Sharpe {best_sharpe:.3f}")
            else:
                print(f"   [TUNER] Status: {tuner_status.get('status', 'NO_DATA')}")
        except Exception as e:
            Log.warn(f"[TUNER] Status check failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  PORTFOLIO TRANSFORMER — Attention model (Phase 11)
    # ═══════════════════════════════════════════════════════
    transformer_cache = {"data": {}, "last_update": 0}
    if TRANSFORMER_ENABLED:
        try:
            saved_tf = load_transformer_state()
            if saved_tf:
                transformer_cache["data"] = saved_tf
            print(f"   [TRANSFORMER] Attention model ready")
        except Exception as e:
            Log.warn(f"[TRANSFORMER] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  ORDER BOOK SIMULATOR — Impact cost (Phase 11)
    # ═══════════════════════════════════════════════════════
    orderbook_cache = {"data": {}, "last_update": 0}
    if ORDERBOOK_SIM_ENABLED:
        try:
            ob_saved = load_orderbook_state()
            if ob_saved:
                orderbook_cache["data"] = ob_saved
            print(f"   [ORDERBOOK] Impact simulator ready")
        except Exception as e:
            Log.warn(f"[ORDERBOOK] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  DYNAMIC STOP-LOSS — Chandelier trailing (Phase 11)
    # ═══════════════════════════════════════════════════════
    if DYNAMIC_STOPLOSS_ENABLED:
        try:
            sl_state = load_stoploss_state()
            tracked = len(sl_state.get("stops", {}))
            print(f"   [DYN-SL] Trailing stops loaded: {tracked} positions")
        except Exception as e:
            Log.warn(f"[DYN-SL] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  PAIR TRADING — Co-integration engine (Phase 11)
    # ═══════════════════════════════════════════════════════
    pairs_cache = {"data": {}, "last_update": 0}
    if PAIR_TRADING_ENABLED:
        try:
            saved_pairs = load_pairs_state()
            if saved_pairs:
                pairs_cache["data"] = saved_pairs
                n_pairs = len(saved_pairs.get("pairs", []))
                print(f"   [PAIRS] Co-integrated pairs loaded: {n_pairs}")
            else:
                print(f"   [PAIRS] No cached pairs — will scan at first cycle")
        except Exception as e:
            Log.warn(f"[PAIRS] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  ANOMALY DETECTOR — Isolation Forest (Phase 11)
    # ═══════════════════════════════════════════════════════
    anomaly_cache = {"data": {}, "last_update": 0}
    if ANOMALY_DETECTOR_ENABLED:
        try:
            loaded = load_anomaly_models()
            saved_a = load_anomaly_state()
            if saved_a:
                anomaly_cache["data"] = saved_a
            print(f"   [ANOMALY] Isolation Forest {'loaded' if loaded else 'will train on first scan'}")
        except Exception as e:
            Log.warn(f"[ANOMALY] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  MODEL VERSIONING — Shadow A/B deploy (Phase 11)
    # ═══════════════════════════════════════════════════════
    if MODEL_VERSIONING_ENABLED:
        try:
            vs_state = load_versioning_state()
            vs_status = get_versioning_status()
            n_versions = vs_status.get("versions_count", 0)
            n_shadow = len(vs_status.get("shadow", {}))
            print(f"   [VERSIONING] Registry: {n_versions} versions, {n_shadow} shadows")
        except Exception as e:
            Log.warn(f"[VERSIONING] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  ADAPTIVE EXECUTOR — VWAP/TWAP slicer (Phase 12)
    # ═══════════════════════════════════════════════════════
    executor_cache = {"data": {}, "last_update": 0}
    if ADAPTIVE_EXECUTOR_ENABLED:
        try:
            ex_saved = load_executor_state()
            if ex_saved:
                executor_cache["data"] = ex_saved
            print(f"   [EXECUTOR] Adaptive execution engine ready")
        except Exception as e:
            Log.warn(f"[EXECUTOR] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  RL REBALANCER — Q-learning rebalance agent (Phase 12)
    # ═══════════════════════════════════════════════════════
    rl_rebal_cache = {"data": {}, "last_update": 0}
    if RL_REBALANCER_ENABLED:
        try:
            rl_rebal_cache["data"] = load_rl_rebal_state()
            train_res = train_rl_rebalancer()
            episodes = train_res.get("episodes", 0)
            print(f"   [RL-REBAL] Agent ready — {episodes} replay episodes")
        except Exception as e:
            Log.warn(f"[RL-REBAL] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  OPTIONS SYNTHESIZER — Strategy generation (Phase 12)
    # ═══════════════════════════════════════════════════════
    synth_cache = {"data": {}, "last_update": 0}
    if OPTIONS_SYNTH_ENABLED:
        try:
            synth_cache["data"] = load_synth_state()
            print(f"   [OPT-SYNTH] Strategy synthesizer ready")
        except Exception as e:
            Log.warn(f"[OPT-SYNTH] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  CAUSAL ENGINE — Feature causality filter (Phase 12)
    # ═══════════════════════════════════════════════════════
    causal_cache = {"data": {}, "last_update": 0}
    if CAUSAL_ENGINE_ENABLED:
        try:
            causal_cache["data"] = load_causal_state()
            n_approved = len(causal_cache["data"].get("approved_features", []))
            if n_approved:
                print(f"   [CAUSAL] Loaded: {n_approved} causally-approved features")
            else:
                print(f"   [CAUSAL] Will discover causal graph on first scan")
        except Exception as e:
            Log.warn(f"[CAUSAL] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  GA EVOLVER — Genetic algorithm strategy evolver (Phase 13)
    # ═══════════════════════════════════════════════════════
    ga_state = {}
    if GA_EVOLVER_ENABLED:
        try:
            ga_state = load_evolver_state()
            gen = ga_state.get("generation", 0)
            fit = ga_state.get("best_fitness", 0)
            if gen:
                print(f"   [GA] Loaded: gen {gen}, best Sharpe {fit:.3f}")
            else:
                print(f"   [GA] Will evolve strategies at first EOD")
        except Exception as e:
            Log.warn(f"[GA] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  DEBATE SYSTEM — Multi-agent consensus (Phase 13)
    # ═══════════════════════════════════════════════════════
    debate_state = {}
    if DEBATE_SYSTEM_ENABLED:
        try:
            debate_state = load_debate_state()
            n_debates = debate_state.get("total_debates", 0)
            if n_debates:
                print(f"   [DEBATE] Loaded: {n_debates} debates recorded")
            else:
                print(f"   [DEBATE] Will run debates on first buy signal")
        except Exception as e:
            Log.warn(f"[DEBATE] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  RL TRADE AGENT — DQN entry/exit agent (Phase 14)
    # ═══════════════════════════════════════════════════════
    if RL_TRADE_AGENT_ENABLED:
        try:
            _rl_agent_state = load_agent_state()
            _rl_steps = _rl_agent_state.get("train_steps", 0)
            if _rl_steps:
                print(f"   [RL-AGENT] Loaded: {_rl_steps} training steps, ε={_rl_agent_state.get('epsilon', 1):.3f}")
            else:
                print(f"   [RL-AGENT] Will begin learning on first trade signal")
        except Exception as e:
            Log.warn(f"[RL-AGENT] Init failed: {e}")

    # ═══════════════════════════════════════════════════════
    #  SENTIMENT MOMENTUM INDEX — Composite SMI (Phase 14)
    # ═══════════════════════════════════════════════════════
    smi_state = {}
    if SENTIMENT_MOMENTUM_ENABLED:
        try:
            smi_state = load_smi_state()
            if smi_state.get("market_smi") is not None:
                print(f"   [SMI] Loaded: market SMI={smi_state.get('market_smi', 0):.3f} ({smi_state.get('market_label', 'NEUTRAL')})")
            else:
                print(f"   [SMI] Will compute sentiment index on first scan")
        except Exception as e:
            Log.warn(f"[SMI] Init failed: {e}")

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
    market_mood_cache = {"mood": {}, "last_update": 0}  # Cache global mood (refresh every 5 min)

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
                        "Exit_Type": "FORCE_CLOSE",
                    })
                    alert_trade_exit(sym, trade["price"], current_price, round(pnl, 2), "FORCE_CLOSE")
                    try:
                        broker.sell(symbol=sym, qty=trade["qty"], price=current_price)
                    except Exception:
                        pass

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

        # ═══════════════════════════════════════════════════════
        #  GLOBAL MOOD REFRESH — Assess real-world context
        # ═══════════════════════════════════════════════════════
        if GLOBAL_MOOD_ENABLED and (time.time() - market_mood_cache["last_update"] > 300):
            try:
                market_mood_cache["mood"] = get_global_market_mood()
                market_mood_cache["last_update"] = time.time()
                mood_score = market_mood_cache["mood"].get("overall_mood", 0)
                mood_label = "RISK-ON" if mood_score > 0.2 else ("RISK-OFF" if mood_score < -0.2 else "NEUTRAL")
                Log.info(f"[MOOD] Global: {mood_label} ({mood_score:+.2f}) | "
                         f"VIX: {market_mood_cache['mood'].get('india_vix', 'N/A')}")
            except Exception as e:
                Log.warn(f"[MOOD] Failed to refresh: {e}")

        # Monitor Active Trades (with partial profit taking + SMART EXIT) ----
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

            # Dynamic trailing stop (Phase 11) — replaces static ATR trail
            if DYNAMIC_STOPLOSS_ENABLED:
                try:
                    dyn_exit, dyn_reason, dyn_data = check_dynamic_exit(sym, current_price, df)
                    dyn_stop = dyn_data.get("current_stop", trade["stop_loss"])
                    if dyn_stop > trade["stop_loss"]:
                        trade["stop_loss"] = round(dyn_stop, 2)
                except Exception:
                    # Fallback to static trailing
                    new_stop = current_price - (ATR_STOP_MULTIPLIER * current_atr)
                    if new_stop > trade["stop_loss"]:
                        trade["stop_loss"] = round(new_stop, 2)
            else:
                # Static trailing stop
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
            elif SMART_EXIT_ENABLED:
                # ═══════════════════════════════════════════
                #  SMART EXIT — Brain decides when to leave
                # ═══════════════════════════════════════════
                last = df.iloc[-1]
                trade_indicators = {
                    "rsi": float(last.get("RSI", 50)),
                    "macd": float(last.get("MACD", 0)),
                    "macd_signal": float(last.get("MACD_Signal", 0)),
                    "volume_ratio": float(last.get("Volume_Ratio", 1.0)),
                    "atr": current_atr,
                    "sma_50": float(last.get("SMA_50", 0)),
                    "ema_20": float(last.get("EMA_20", 0)),
                    "bb_upper": float(last.get("BB_Upper", 0)),
                    "bb_lower": float(last.get("BB_Lower", 0)),
                }
                # Get sentiment for smart exit
                trade_sentiment = 0.0
                try:
                    trade_sentiment = float(last.get("Sentiment_Score", 0))
                except Exception:
                    pass

                should_exit, exit_type, exit_reason = brain.should_exit(
                    entry_price=trade["price"],
                    current_price=current_price,
                    stop_loss=trade["stop_loss"],
                    target=trade["target"],
                    entry_time=trade.get("time", "09:30"),
                    indicators=trade_indicators,
                    sentiment_score=trade_sentiment,
                    market_mood=market_mood_cache.get("mood", {}),
                )
                if should_exit:
                    action = exit_type  # e.g. MOMENTUM_EXIT, RSI_EXIT, TIME_DECAY, etc.
                    Log.info(f"{sym}: Smart Exit → {exit_type} | {exit_reason}")
                else:
                    continue
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
                    alert_guardian_stop(f"Risk Guardian halted after losing trade on {sym}")
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
                "Exit_Type": action,
            })

            # Send exit alert
            alert_trade_exit(
                symbol=sym, entry_price=trade["price"],
                exit_price=current_price, pnl=round(pnl_total, 2),
                exit_type=action,
            )

            # Route SELL through broker bridge
            try:
                broker.sell(symbol=sym, qty=trade["qty"], price=current_price)
            except Exception:
                pass

            # Update model drift actuals (Phase 7)
            if MODEL_DRIFT_ENABLED:
                try:
                    update_prediction_actuals(sym, pnl_total > 0)
                except Exception:
                    pass

        # ---- Scan ALL stocks & write LIVE ANALYSIS ----
        analysis = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "scan_number": getattr(run_sniper, '_scan_count', 0) + 1,
            "status": state["status"],
            "day_pnl": round(state["total_profit"], 2),
            "day_pnl_pct": round((state["total_profit"] / CAPITAL) * 100, 2),
            "open_positions": sum(1 for t in state["active_trades"] if t.get("status") == "OPEN"),
            "bullets_left": MAX_BULLETS - sum(1 for t in state["active_trades"] if t.get("status") == "OPEN"),
            "personality": TRADING_PERSONALITY,
            "smart_exit": SMART_EXIT_ENABLED,
            "smart_entry": SMART_ENTRY_ENABLED,
            "global_mood": market_mood_cache.get("mood", {}),
            "neuro_voters": neuro_ensemble.get_voter_summary(),
            "broker_mode": broker.mode.value,
            "broker_name": broker.broker.__class__.__name__,
            "market_breadth": breadth_cache.get("data", {}),
            "sector_rotation": sector_cache.get("data", {}),
            "correlation": corr_cache.get("data", {}),
            "intraday_window": {},
            "stocks": [],
        }

        # Refresh intraday window info
        try:
            analysis["intraday_window"] = get_current_window()
        except Exception:
            pass
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

                    # Run ensemble + NeuroVoter (only if models loaded)
                    if scan_sym in all_models:
                        _, votes, rf_c, xgb_c, lstm_c, intra_c = get_ensemble_signal(
                            scan_sym, scan_df, all_models[scan_sym]
                        )
                        stock_analysis["rf_conf"] = round(rf_c, 4)
                        stock_analysis["xgb_conf"] = round(xgb_c, 4)
                        stock_analysis["lstm_conf"] = round(lstm_c, 4)
                        stock_analysis["intra_conf"] = round(intra_c, 4)
                        stock_analysis["avg_conf"] = round((rf_c + xgb_c + lstm_c + intra_c) / 4, 4)
                        stock_analysis["votes"] = votes

                        # ── NeuroVoter: 6 human-like AI voters debate ──
                        scan_sentiment = float(last.get("Sentiment_Score", 0))
                        scan_mood_score = market_mood_cache.get("mood", {}).get("overall_mood", 0)
                        scan_mood_details = market_mood_cache.get("mood", {})

                        neuro_decision = neuro_ensemble.decide(
                            symbol=scan_sym,
                            df=scan_df,
                            rf_conf=rf_c,
                            xgb_conf=xgb_c,
                            lstm_conf=lstm_c,
                            intra_conf=intra_c,
                            original_votes=votes,
                            sentiment_score=scan_sentiment,
                            global_mood=scan_mood_score,
                            mood_details=scan_mood_details,
                        )

                        # Store voter details for dashboard
                        stock_analysis["neuro_decision"] = {
                            "should_buy": neuro_decision.should_buy,
                            "score": round(neuro_decision.weighted_score, 3),
                            "conviction": round(neuro_decision.total_conviction, 3),
                            "buy_voters": neuro_decision.buy_voters,
                            "sell_voters": neuro_decision.sell_voters,
                            "hold_voters": neuro_decision.hold_voters,
                            "vetoed": neuro_decision.vetoed,
                            "veto_reasons": neuro_decision.veto_reasons,
                            "regime": neuro_decision.regime,
                            "reasoning": neuro_decision.reasoning,
                            "voters": [
                                {
                                    "name": vr.name,
                                    "vote": round(vr.vote, 2),
                                    "conviction": round(vr.conviction, 2),
                                    "reasoning": vr.reasoning,
                                    "veto": vr.veto,
                                    "veto_reason": vr.veto_reason or "",
                                }
                                for vr in neuro_decision.voter_results
                            ],
                        }

                        if neuro_decision.should_buy:
                            stock_analysis["signal"] = "BUY"
                            stock_analysis["reason"] = neuro_decision.reasoning
                            cur_atr = float(last.get("ATR", 0))
                            if cur_atr > 0:
                                stock_analysis["stop_loss"] = round(cur_price - ATR_STOP_MULTIPLIER * cur_atr, 2)
                                stock_analysis["target"] = round(cur_price + ATR_TARGET_MULTIPLIER * cur_atr, 2)

                            # Lightweight Guardian pre-check
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
                            stock_analysis["reason"] = neuro_decision.reasoning
                            stock_analysis["guardian_reason"] = neuro_decision.reasoning
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
                    _, votes, rf_c, xgb_c, lstm_c, intra_c = get_ensemble_signal(
                        sym, df, all_models[sym]
                    )
                    avg_conf = round((rf_c + xgb_c + lstm_c + intra_c) / 4, 3)

                    # ═══════════════════════════════════════════
                    #  NEURO-VOTER GATE — 6 human-like AI voters
                    # ═══════════════════════════════════════════
                    fire_sentiment = 0.0
                    try:
                        fire_sentiment = float(df.iloc[-1].get("Sentiment_Score", 0))
                    except Exception:
                        pass
                    fire_mood_score = market_mood_cache.get("mood", {}).get("overall_mood", 0)
                    fire_mood_details = market_mood_cache.get("mood", {})

                    fire_decision = neuro_ensemble.decide(
                        symbol=sym,
                        df=df,
                        rf_conf=rf_c,
                        xgb_conf=xgb_c,
                        lstm_conf=lstm_c,
                        intra_conf=intra_c,
                        original_votes=votes,
                        sentiment_score=fire_sentiment,
                        global_mood=fire_mood_score,
                        mood_details=fire_mood_details,
                    )

                    should_buy = fire_decision.should_buy

                    # Log NeuroVoter decision
                    voter_line = " | ".join(
                        f"{vr.name}:{vr.vote:+.1f}" for vr in fire_decision.voter_results
                    )
                    if fire_decision.vetoed:
                        Log.warn(f"{sym}: VETOED — {fire_decision.reasoning}")
                        alert_veto(sym, fire_decision.veto_reasons)
                        save_voter_history(sym, fire_decision, "VETOED", float(df["Close"].iloc[-1]))
                    elif should_buy:
                        Log.success(f"{sym}: NEURO-BUY (score {fire_decision.weighted_score:+.3f}) — {voter_line}")
                        save_voter_history(sym, fire_decision, "BUY", float(df["Close"].iloc[-1]))
                    else:
                        Log.info(f"{sym}: NEURO-WAIT — {fire_decision.reasoning}")
                        save_voter_history(sym, fire_decision, "SKIP", float(df["Close"].iloc[-1]))

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

                        # ═══════════════════════════════════════════
                        #  TRADING BRAIN GATE — Human-like entry check
                        # ═══════════════════════════════════════════
                        if SMART_ENTRY_ENABLED:
                            last_row = df.iloc[-1]
                            entry_indicators = {
                                "rsi": float(last_row.get("RSI", 50)),
                                "macd": float(last_row.get("MACD", 0)),
                                "macd_signal": float(last_row.get("MACD_Signal", 0)),
                                "volume_ratio": float(last_row.get("Volume_Ratio", 1.0)),
                                "atr": current_atr,
                                "sma_50": float(last_row.get("SMA_50", 0)),
                                "ema_20": float(last_row.get("EMA_20", 0)),
                                "bb_upper": float(last_row.get("BB_Upper", 0)),
                                "bb_lower": float(last_row.get("BB_Lower", 0)),
                            }
                            entry_sentiment = float(last_row.get("Sentiment_Score", 0))

                            brain_ok, brain_reason = brain.should_enter(
                                votes=votes,
                                avg_confidence=avg_conf,
                                indicators=entry_indicators,
                                sentiment_score=entry_sentiment,
                                market_mood=market_mood_cache.get("mood", {}),
                            )
                            if not brain_ok:
                                Log.warn(f"{sym}: BLOCKED by Brain — {brain_reason}")
                                for s_entry in analysis.get("stocks", []):
                                    if s_entry.get("symbol") == sym:
                                        s_entry["guardian"] = "BLOCKED"
                                        s_entry["guardian_reason"] = f"Brain: {brain_reason}"
                                continue
                            Log.info(f"{sym}: Brain APPROVED ✅ — {brain_reason}")

                        # ═══════════════════════════════════════════
                        #  INTRADAY PATTERN GATE — Time-of-day check
                        # ═══════════════════════════════════════════
                        if INTRADAY_PATTERNS_ENABLED:
                            try:
                                delay, delay_reason, window_info = should_delay_entry(sym)
                                if delay:
                                    Log.warn(f"{sym}: DELAYED by Intraday Pattern — {delay_reason}")
                                    continue
                                else:
                                    Log.info(f"{sym}: Intraday window OK ✅ ({window_info.get('window', '?')})")
                            except Exception as e:
                                Log.warn(f"[INTRADAY] Check failed: {e}")

                        # ═══════════════════════════════════════════
                        #  CORRELATION GUARD — Block if too similar
                        # ═══════════════════════════════════════════
                        if CORRELATION_GUARD_ENABLED:
                            try:
                                held = [t["stock"] for t in state["active_trades"] if t["status"] == "OPEN"]
                                corr_ok, corr_reason = check_before_buy(sym, held)
                                if not corr_ok:
                                    Log.warn(f"{sym}: BLOCKED by Correlation Guard — {corr_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Correlation: {corr_reason}"
                                    continue
                                Log.info(f"{sym}: Correlation check PASSED ✅")
                            except Exception as e:
                                Log.warn(f"[CORR] Check failed: {e}")

                        # ═══════════════════════════════════════════
                        #  EARNINGS GUARD — Block near quarterly results
                        # ═══════════════════════════════════════════
                        if EARNINGS_GUARD_ENABLED:
                            try:
                                eg = check_earnings_guard(sym, earnings_cache.get("calendar"))
                                if eg["blocked"]:
                                    Log.warn(f"{sym}: BLOCKED by Earnings Guard — {eg['reason']} "
                                             f"(earnings on {eg['earnings_date']})")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Earnings: {eg['reason']}"
                                    continue
                                elif eg["reduce_target"]:
                                    Log.info(f"{sym}: Earnings near — targets tightened by {eg['reduce_factor']:.0%}")
                            except Exception as e:
                                Log.warn(f"[EARNINGS] Check failed: {e}")

                        # ═══════════════════════════════════════════
                        #  MODEL DRIFT — Log prediction & apply penalty
                        # ═══════════════════════════════════════════
                        if MODEL_DRIFT_ENABLED:
                            try:
                                for mname, mconf in [("RF", rf_c), ("XGB", xgb_c)]:
                                    log_prediction(sym, mname, mconf, 1 if mconf >= CONFIDENCE_THRESHOLD else 0)
                                    drift_pen = get_drift_penalty(mname)
                                    if drift_pen < 1.0:
                                        Log.info(f"{sym}: {mname} drift penalty {drift_pen:.0%}")
                            except Exception as e:
                                Log.warn(f"[DRIFT] Log failed: {e}")

                        # ═══════════════════════════════════════════
                        #  MULTI-TIMEFRAME GATE — Weekly+Monthly consensus
                        # ═══════════════════════════════════════════
                        if MULTI_TIMEFRAME_ENABLED:
                            try:
                                mtf_ok, mtf_reason, mtf_data = check_multi_timeframe_gate(sym, df)
                                if not mtf_ok:
                                    Log.warn(f"{sym}: BLOCKED by Multi-Timeframe — {mtf_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"MTF: {mtf_reason}"
                                    continue
                                Log.info(f"{sym}: Multi-TF {mtf_reason}")
                            except Exception as e:
                                Log.warn(f"[MTF] Check failed: {e}")

                        # ═══════════════════════════════════════════
                        #  NEWS DETECTOR GATE — NLP event check
                        # ═══════════════════════════════════════════
                        if NEWS_DETECTOR_ENABLED:
                            try:
                                news_ok, news_reason, news_data = check_news_gate(sym)
                                if not news_ok:
                                    Log.warn(f"{sym}: BLOCKED by News — {news_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"News: {news_reason}"
                                    continue
                                if "boost" in news_reason.lower():
                                    Log.success(f"{sym}: News BOOST ✅ — {news_reason[:80]}")
                                else:
                                    Log.info(f"{sym}: News clear ✅")
                            except Exception as e:
                                Log.warn(f"[NEWS] Check failed: {e}")

                        # ═══════════════════════════════════════════
                        #  REGIME GATE — HMM Bull/Bear/Sideways (P9)
                        # ═══════════════════════════════════════════
                        if REGIME_DETECTOR_ENABLED:
                            try:
                                # Refresh regime every 30 min
                                if time.time() - regime_cache.get("last_update", 0) > 1800:
                                    regime_cache["data"] = detect_regime()
                                    regime_cache["last_update"] = time.time()
                                    save_regime_state(regime_cache["data"])
                                rg_ok, rg_reason, rg_data = get_regime_gate(regime_cache.get("data"))
                                if not rg_ok:
                                    Log.warn(f"{sym}: BLOCKED by Regime — {rg_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Regime: {rg_reason}"
                                    continue
                                if "caution" in rg_reason.lower() or "weak" in rg_reason.lower():
                                    Log.info(f"{sym}: Regime — {rg_reason}")
                            except Exception as e:
                                Log.warn(f"[REGIME] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  FINBERT GATE — Transformer sentiment (P9)
                        # ═══════════════════════════════════════════
                        if FINBERT_ENABLED:
                            try:
                                fb_ok, fb_reason, fb_data = get_finbert_gate(sym)
                                if not fb_ok:
                                    Log.warn(f"{sym}: BLOCKED by FinBERT — {fb_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"FinBERT: {fb_reason}"
                                    continue
                                if "positive" in fb_reason.lower() or "boost" in fb_reason.lower():
                                    Log.success(f"{sym}: FinBERT POSITIVE ✅")
                            except Exception as e:
                                Log.warn(f"[FINBERT] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  VOLUME PROFILE GATE — Order flow (P9)
                        # ═══════════════════════════════════════════
                        if VOLUME_PROFILE_ENABLED:
                            try:
                                vp_ok, vp_reason, vp_data = check_volume_gate(sym, df)
                                if not vp_ok:
                                    Log.warn(f"{sym}: BLOCKED by Volume Profile — {vp_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Volume: {vp_reason}"
                                    continue
                                Log.info(f"{sym}: Volume — {vp_reason[:60]}")
                            except Exception as e:
                                Log.warn(f"[VOLUME] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  INTERMARKET GATE — Macro risk-off (P10)
                        # ═══════════════════════════════════════════
                        if INTERMARKET_ENABLED:
                            try:
                                # Refresh every 30 min
                                if time.time() - intermarket_cache.get("last_update", 0) > 1800:
                                    intermarket_cache["data"] = analyse_intermarket()
                                    intermarket_cache["last_update"] = time.time()
                                    save_intermarket_state(intermarket_cache["data"])
                                im_ok, im_reason, im_data = get_intermarket_gate(intermarket_cache.get("data"))
                                if not im_ok:
                                    Log.warn(f"{sym}: BLOCKED by Intermarket — {im_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Macro: {im_reason}"
                                    continue
                                Log.info(f"{sym}: Intermarket — {im_reason[:60]}")
                            except Exception as e:
                                Log.warn(f"[MACRO] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  LIQUIDITY GATE — Spread+Volume filter (P10)
                        # ═══════════════════════════════════════════
                        if LIQUIDITY_FILTER_ENABLED:
                            try:
                                lq_ok, lq_reason, lq_data = check_liquidity_gate(sym)
                                if not lq_ok:
                                    Log.warn(f"{sym}: BLOCKED by Liquidity — {lq_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Liquidity: {lq_reason}"
                                    continue
                                Log.info(f"{sym}: Liquidity score {lq_data.get('composite_score', 0):.0f} ✅")
                            except Exception as e:
                                Log.warn(f"[LIQUIDITY] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  BAYESIAN FUSION — Posterior vote override (P10)
                        # ═══════════════════════════════════════════
                        if BAYESIAN_FUSION_ENABLED:
                            try:
                                bfuse = bayesian_fuse(
                                    rf_conf=rf_c, xgb_conf=xgb_c,
                                    lstm_conf=lstm_c, intra_conf=intra_c,
                                    voter_signals=None,  # uses NeuroVoter data if available
                                )
                                bfuse_score = bfuse.get("fused_score", 0.5)
                                bfuse_action = bfuse.get("action", "HOLD")
                                if bfuse_action == "SELL":
                                    Log.warn(f"{sym}: BLOCKED by Bayesian Fusion — score {bfuse_score:.3f}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Bayes: score {bfuse_score:.3f}"
                                    continue
                                Log.info(f"{sym}: Bayesian fusion score {bfuse_score:.3f} → {bfuse_action}")
                            except Exception as e:
                                Log.warn(f"[BAYES] Fusion failed: {e}")

                        # ═══════════════════════════════════════════
                        #  TRANSFORMER GATE — Attention portfolio (P11)
                        # ═══════════════════════════════════════════
                        # ═══════════════════════════════════════════
                        #  CAUSAL ENGINE — Feature filter (P12)
                        # ═══════════════════════════════════════════
                        if CAUSAL_ENGINE_ENABLED:
                            try:
                                # Refresh causal graph every 60 min
                                if time.time() - causal_cache.get("last_update", 0) > 3600:
                                    causal_cache["data"] = discover_causal_graph(df)
                                    causal_cache["last_update"] = time.time()
                                    save_causal_state(causal_cache["data"])
                                    n_app = causal_cache["data"].get("total_approved", 0)
                                    Log.info(f"{sym}: Causal graph — {n_app} features approved")
                            except Exception as e:
                                Log.warn(f"[CAUSAL] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  SENTIMENT MOMENTUM — SMI gate (P14)
                        # ═══════════════════════════════════════════
                        if SENTIMENT_MOMENTUM_ENABLED:
                            try:
                                _smi_fb = finbert_cache.get("data", {}) if 'finbert_cache' in dir() else {}
                                _smi_news = news_cache.get("data", {}) if 'news_cache' in dir() else {}
                                _smi_mood = market_mood_cache.get("mood", {})
                                _smi_result = compute_smi(sym, df, _smi_fb, _smi_news, _smi_mood)
                                _smi_pass, _smi_reason, _smi_val = check_smi_gate(sym, _smi_result)
                                if not _smi_pass:
                                    Log.warn(f"{sym}: SMI REJECT — {_smi_reason}")
                                    continue
                                Log.info(f"{sym}: SMI={_smi_val:.3f} ({_smi_result.get('label', '')})")
                            except Exception as e:
                                Log.warn(f"[SMI] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  DEBATE SYSTEM — Multi-agent consensus (P13)
                        # ═══════════════════════════════════════════
                        if DEBATE_SYSTEM_ENABLED:
                            try:
                                # Assemble live_data from real module states
                                _debate_live = {
                                    "regime_state": regime_cache.get("data", {}),
                                    "finbert_state": finbert_cache.get("data", {}) if 'finbert_cache' in dir() else {},
                                    "news_state": news_cache.get("data", {}) if 'news_cache' in dir() else {},
                                    "sentiment_score": sentiment_cache.get("score", 0) if 'sentiment_cache' in dir() else 0,
                                    "intermarket_state": intermarket_cache.get("data", {}) if 'intermarket_cache' in dir() else {},
                                    "breadth_state": breadth_cache.get("data", {}),
                                    "var_state": var_cache.get("data", {}) if 'var_cache' in dir() else {},
                                    "guardian_status": {"halted": guardian.stats.get("halted", False)} if guardian else {},
                                    "anomaly_score": 0,
                                    "volume_gate_pass": True,
                                    "market_mood": market_mood_cache.get("mood", {}),
                                }
                                _approved, _reason, _cscore = get_debate_verdict(sym, df, current_price, _debate_live)
                                debate_result = run_debate(sym, df, current_price, _debate_live)
                                debate_state = record_debate_result(debate_result, debate_state)
                                if not _approved:
                                    Log.warn(f"{sym}: DEBATE REJECT — {_reason} (score={_cscore:.3f})")
                                    continue
                                Log.info(f"{sym}: DEBATE APPROVE — score={_cscore:.3f}")
                            except Exception as e:
                                Log.warn(f"[DEBATE] Gate failed: {e}")

                        if TRANSFORMER_ENABLED:
                            try:
                                # Refresh transformer every 30 min
                                if time.time() - transformer_cache.get("last_update", 0) > 1800:
                                    stock_dfs = {}
                                    for _tsym in top_stocks:
                                        _tdf = get_live_data(_tsym)
                                        if _tdf is not None:
                                            stock_dfs[_tsym] = _tdf
                                    if stock_dfs:
                                        transformer_cache["data"] = predict_portfolio(stock_dfs, list(stock_dfs.keys()))
                                        transformer_cache["last_update"] = time.time()
                                        save_transformer_state(transformer_cache["data"])
                                tf_ok, tf_reason = check_transformer_gate(sym, transformer_cache.get("data", {}))
                                if not tf_ok:
                                    Log.warn(f"{sym}: BLOCKED by Transformer — {tf_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Transformer: {tf_reason}"
                                    continue
                                Log.info(f"{sym}: Transformer — {tf_reason[:60]}")
                            except Exception as e:
                                Log.warn(f"[TRANSFORMER] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  ANOMALY GATE — Isolation Forest guard (P11)
                        # ═══════════════════════════════════════════
                        if ANOMALY_DETECTOR_ENABLED:
                            try:
                                # Refresh anomaly detection every 15 min
                                if time.time() - anomaly_cache.get("last_update", 0) > 900:
                                    stock_dfs = {}
                                    for _asym in top_stocks:
                                        _adf = get_live_data(_asym)
                                        if _adf is not None:
                                            stock_dfs[_asym] = _adf
                                    if stock_dfs:
                                        anomaly_cache["data"] = detect_anomalies_iforest(stock_dfs)
                                        anomaly_cache["last_update"] = time.time()
                                        save_anomaly_state(anomaly_cache["data"])
                                an_ok, an_reason, an_data = check_anomaly_gate(sym, anomaly_cache.get("data", {}))
                                if not an_ok:
                                    Log.warn(f"{sym}: BLOCKED by Anomaly Detector — {an_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Anomaly: {an_reason}"
                                    continue
                                Log.info(f"{sym}: Anomaly check — {an_reason[:60]}")
                            except Exception as e:
                                Log.warn(f"[ANOMALY] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  PAIR TRADING GATE — Conflict check (P11)
                        # ═══════════════════════════════════════════
                        if PAIR_TRADING_ENABLED:
                            try:
                                # Refresh pairs every 60 min
                                if time.time() - pairs_cache.get("last_update", 0) > 3600:
                                    pairs_cache["data"] = analyse_pairs(top_stocks)
                                    pairs_cache["last_update"] = time.time()
                                    save_pairs_state(pairs_cache["data"])
                                pt_ok, pt_reason, pt_data = check_pair_signal(sym, pairs_cache.get("data", {}))
                                if not pt_ok:
                                    Log.warn(f"{sym}: BLOCKED by Pair Trading — {pt_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"Pairs: {pt_reason}"
                                    continue
                                Log.info(f"{sym}: Pair check — {pt_reason[:60]}")
                            except Exception as e:
                                Log.warn(f"[PAIRS] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  ORDERBOOK GATE — Impact cost filter (P11)
                        # ═══════════════════════════════════════════
                        if ORDERBOOK_SIM_ENABLED:
                            try:
                                # Refresh order book every 10 min
                                if time.time() - orderbook_cache.get("last_update", 0) > 600:
                                    orderbook_cache["data"] = analyse_orderbook(top_stocks)
                                    orderbook_cache["last_update"] = time.time()
                                    save_orderbook_state(orderbook_cache["data"])
                                ob_ok, ob_reason, ob_data = check_orderbook_gate(sym, 0, orderbook_cache.get("data", {}))
                                if not ob_ok:
                                    Log.warn(f"{sym}: BLOCKED by Order Book — {ob_reason}")
                                    for s_entry in analysis.get("stocks", []):
                                        if s_entry.get("symbol") == sym:
                                            s_entry["guardian"] = "BLOCKED"
                                            s_entry["guardian_reason"] = f"OrderBook: {ob_reason}"
                                    continue
                                Log.info(f"{sym}: Order book — {ob_reason[:60]}")
                            except Exception as e:
                                Log.warn(f"[ORDERBOOK] Gate failed: {e}")

                        # ═══════════════════════════════════════════
                        #  OPTIONS HEDGE GATE — Exposure limit check
                        # ═══════════════════════════════════════════
                        if OPTIONS_HEDGE_ENABLED:
                            try:
                                hedge_ok, hedge_reason, hedge_info = check_hedge_gate(
                                    state["active_trades"], CAPITAL
                                )
                                if not hedge_ok:
                                    Log.warn(f"{sym}: BLOCKED by Hedge Gate — {hedge_reason}")
                                    # Save hedge recommendation for dashboard
                                    hedge_data = compute_hedge_requirements(state["active_trades"], CAPITAL)
                                    save_hedge_state(hedge_data)
                                    continue
                                elif "caution" in hedge_reason.lower():
                                    Log.info(f"{sym}: Hedge warning — {hedge_reason}")
                            except Exception as e:
                                Log.warn(f"[HEDGE] Check failed: {e}")

                        # ═══════════════════════════════════════════
                        #  MARKET BREADTH — Adjust position size
                        # ═══════════════════════════════════════════
                        adjusted_bullet = bullet_size

                        # Kelly Criterion sizing override
                        if KELLY_SIZING_ENABLED:
                            try:
                                ks = get_kelly_position_size(
                                    sym, current_price, CAPITAL,
                                    regime="neutral", kelly_map=kelly_cache.get("map"),
                                )
                                adjusted_bullet = ks["amount"]
                                if ks["method"] != "DEFAULT":
                                    Log.info(f"{sym}: Kelly sizing → ₹{adjusted_bullet:,.0f} "
                                             f"({ks['pct']:.1f}% via {ks['method']})")
                            except Exception as e:
                                Log.warn(f"[KELLY] Sizing failed: {e}")

                        if MARKET_BREADTH_ENABLED:
                            # Refresh breadth every 15 min
                            if time.time() - breadth_cache["last_update"] > 900:
                                try:
                                    breadth_cache["data"] = analyse_market_breadth()
                                    breadth_cache["size_factor"] = breadth_cache["data"].get("position_size_factor", 1.0)
                                    breadth_cache["last_update"] = time.time()
                                except Exception:
                                    pass
                            sf = breadth_cache.get("size_factor", 1.0)
                            adjusted_bullet = bullet_size * sf
                            if sf < 1.0:
                                Log.info(f"{sym}: Breadth size factor {sf:.0%} → bullet ₹{adjusted_bullet:,.0f}")

                        # Regime sizing multiplier (Phase 9)
                        if REGIME_DETECTOR_ENABLED and regime_cache.get("data"):
                            try:
                                regime_mult = regime_cache["data"].get("sizing_multiplier", 1.0)
                                adjusted_bullet = adjusted_bullet * regime_mult
                                if regime_mult != 1.0:
                                    Log.info(f"{sym}: Regime sizing {regime_mult:.0%} → ₹{adjusted_bullet:,.0f}")
                            except Exception:
                                pass

                        # Risk parity sizing blend
                        if RISK_PARITY_ENABLED and rp_cache.get("weights"):
                            try:
                                rp_size = get_risk_parity_size(
                                    sym, current_price, CAPITAL,
                                    top_stocks, rp_cache["weights"],
                                )
                                # Blend: 60% current + 40% risk-parity
                                rp_bullet = rp_size["amount"]
                                adjusted_bullet = adjusted_bullet * 0.60 + rp_bullet * 0.40
                                if abs(rp_bullet - adjusted_bullet) > 10:
                                    Log.info(f"{sym}: Risk parity blend → ₹{adjusted_bullet:,.0f}")
                            except Exception as e:
                                Log.warn(f"[RP] Sizing failed: {e}")

                        # RL position sizer override (Phase 10)
                        if RL_SIZER_ENABLED:
                            try:
                                rl_size = get_rl_position_size(
                                    sym, current_price, CAPITAL,
                                    regime=regime_cache.get("data", {}).get("regime", "sideways"),
                                    breadth=breadth_cache.get("data", {}),
                                )
                                # Blend: 50% current + 50% RL
                                rl_bullet = rl_size["amount"]
                                adjusted_bullet = adjusted_bullet * 0.50 + rl_bullet * 0.50
                                Log.info(f"{sym}: RL sizing ({rl_size['method']}) "
                                         f"→ ₹{adjusted_bullet:,.0f} ({rl_size['pct']:.1f}%)")
                            except Exception as e:
                                Log.warn(f"[RL] Sizing failed: {e}")

                        # Intermarket exposure multiplier (Phase 10)
                        if INTERMARKET_ENABLED and intermarket_cache.get("data"):
                            try:
                                im_mult = intermarket_cache["data"].get("exposure_multiplier", 1.0)
                                adjusted_bullet = adjusted_bullet * im_mult
                                if im_mult != 1.0:
                                    Log.info(f"{sym}: Intermarket mult {im_mult:.2f} → ₹{adjusted_bullet:,.0f}")
                            except Exception:
                                pass

                        # Liquidity sizing factor (Phase 10)
                        if LIQUIDITY_FILTER_ENABLED:
                            try:
                                lq_factor = get_liquidity_sizing_factor(sym)
                                if lq_factor < 1.0:
                                    adjusted_bullet = adjusted_bullet * lq_factor
                                    Log.info(f"{sym}: Liquidity factor {lq_factor:.0%} → ₹{adjusted_bullet:,.0f}")
                            except Exception:
                                pass

                        pos = calculate_position(current_price, current_atr, adjusted_bullet)
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

                            # Initialize dynamic trailing stop (Phase 11)
                            if DYNAMIC_STOPLOSS_ENABLED:
                                try:
                                    init_stop(sym, current_price, now.isoformat(), pos["stop_loss"], final_qty, df)
                                    Log.info(f"{sym}: Dynamic trailing stop initialized")
                                except Exception as e:
                                    Log.warn(f"[DYN-SL] Init stop failed: {e}")

                            # Shadow model prediction record (Phase 11)
                            if MODEL_VERSIONING_ENABLED:
                                try:
                                    shadow_predict("random_forest", sym, rf_c, metadata={"action": "BUY"})
                                    shadow_predict("xgboost", sym, xgb_c, metadata={"action": "BUY"})
                                except Exception:
                                    pass

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
                                "NeuroScore": round(fire_decision.weighted_score, 3),
                                "Regime": fire_decision.regime,
                            })

                            # Send buy alert
                            alert_trade_buy(
                                symbol=sym, price=current_price, qty=final_qty,
                                confidence=avg_conf,
                                neuro_score=fire_decision.weighted_score,
                                regime=fire_decision.regime,
                            )

                            # Route order through broker bridge
                            # Adaptive Executor: slice large orders (Phase 12)
                            if ADAPTIVE_EXECUTOR_ENABLED and should_slice_order(final_qty, avg_conf):
                                try:
                                    plan = create_execution_plan(
                                        sym, final_qty, current_price,
                                        confidence=avg_conf,
                                    )
                                    # Fire first slice immediately
                                    first = execute_next_slice(sym, current_price)
                                    if first.get("fired"):
                                        broker.buy(
                                            symbol=sym, qty=first["qty"], price=current_price,
                                            stop_loss=pos["stop_loss"], target=pos["target"],
                                        )
                                        Log.info(f"{sym}: Exec slice 1/{len(plan['slices'])} "
                                                 f"({plan['strategy']}) — {first['qty']} shares")
                                    else:
                                        # Fallback: single order
                                        broker.buy(
                                            symbol=sym, qty=final_qty, price=current_price,
                                            stop_loss=pos["stop_loss"], target=pos["target"],
                                        )
                                except Exception as e:
                                    Log.warn(f"[EXECUTOR] Slicing failed, single order: {e}")
                                    try:
                                        broker.buy(
                                            symbol=sym, qty=final_qty, price=current_price,
                                            stop_loss=pos["stop_loss"], target=pos["target"],
                                        )
                                    except Exception as e2:
                                        Log.warn(f"[BROKER] Order routing failed: {e2}")
                            else:
                                try:
                                    broker.buy(
                                        symbol=sym, qty=final_qty, price=current_price,
                                        stop_loss=pos["stop_loss"], target=pos["target"],
                                    )
                                except Exception as e:
                                    Log.warn(f"[BROKER] Order routing failed: {e}")

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

    # Send daily summary alert
    alert_daily_summary(
        total_pnl=state["total_profit"],
        total_trades=state["trades_taken"],
        wins=state["trades_won"],
        losses=state["trades_lost"],
        capital=CAPITAL,
    )

    # Auto-generate daily report (HTML/PDF)
    if REPORT_AUTO_GENERATE:
        try:
            report_path = generate_report()
            Log.success(f"Daily report → {report_path}")
        except Exception as e:
            Log.warn(f"Report generation failed: {e}")

    # Execution quality analysis (Phase 9)
    if EXECUTION_QUALITY_ENABLED:
        try:
            eq_data = analyse_execution_quality()
            save_eq_state(eq_data)
            grade = eq_data.get("overall_grade", "N/A")
            timing = eq_data.get("avg_timing_score", 0)
            Log.info(f"Execution grade: {grade} | Avg timing: {timing:.0f}/100")
        except Exception as e:
            Log.warn(f"Execution quality failed: {e}")

    # Auto trade journal (Phase 9)
    if TRADE_JOURNAL_ENABLED:
        try:
            from trade_journal import generate_journal
            journal_path = generate_journal(capital=CAPITAL)
            if journal_path:
                Log.success(f"Trade journal → {journal_path}")
        except Exception as e:
            Log.warn(f"Trade journal failed: {e}")

    # RL sizer: record outcomes for closed trades (Phase 10)
    if RL_SIZER_ENABLED:
        try:
            for trade in state.get("active_trades", []):
                if trade.get("status") in ("EXITED", "FORCE_CLOSED", "STOPPED"):
                    record_trade_result(
                        trade.get("stock", ""),
                        trade.get("pnl", 0),
                        trade.get("price", 0),
                    )
            save_rl_state()
            Log.info("[RL] Trade outcomes recorded & Q-table saved")
        except Exception as e:
            Log.warn(f"[RL] EOD record failed: {e}")

    # Bayesian fusion: record outcomes (Phase 10)
    if BAYESIAN_FUSION_ENABLED:
        try:
            for trade in state.get("active_trades", []):
                if trade.get("status") in ("EXITED", "FORCE_CLOSED", "STOPPED"):
                    pnl_val = trade.get("pnl", 0)
                    record_bayesian_outcome(success=(pnl_val > 0))
            save_bayesian_state()
            Log.info("[BAYES] Posterior weights updated")
        except Exception as e:
            Log.warn(f"[BAYES] EOD failed: {e}")

    # Greeks heatmap snapshot (Phase 10)
    if GREEKS_HEATMAP_ENABLED:
        try:
            greeks_data = analyse_portfolio_greeks(state.get("active_trades", []), CAPITAL)
            save_greeks_state(greeks_data)
            Log.info(f"[GREEKS] Portfolio delta: {greeks_data.get('totals', {}).get('delta', 0):.2f}")
        except Exception as e:
            Log.warn(f"[GREEKS] EOD failed: {e}")

    # Scalper summary (Phase 10)
    if INTRADAY_SCALPER_ENABLED:
        try:
            save_scalper_state()
            Log.info("[SCALPER] Scalp summary saved")
        except Exception as e:
            Log.warn(f"[SCALPER] EOD failed: {e}")

    # Auto tuner: run sweep on weekends (Phase 10)
    if AUTO_TUNER_ENABLED:
        try:
            import datetime as dt_mod
            if dt_mod.datetime.now().weekday() >= 5:  # Saturday=5, Sunday=6
                Log.info("[TUNER] Weekend detected — running sweep...")
                tuner_result = run_tuning_sweep()
                Log.info(f"[TUNER] Best Sharpe: {tuner_result.get('best_sharpe', 0):.3f}")
            else:
                Log.info("[TUNER] Weekday — skipping sweep")
        except Exception as e:
            Log.warn(f"[TUNER] Sweep failed: {e}")

    # ── Phase 11 EOD hooks ──

    # Dynamic stop-loss: save all trailing stop states
    if DYNAMIC_STOPLOSS_ENABLED:
        try:
            save_stoploss_state()
            stops = get_all_stop_states()
            Log.info(f"[DYN-SL] Trailing stops saved: {len(stops)} positions")
            # Clean exited positions
            for trade in state.get("active_trades", []):
                if trade.get("status") in ("EXITED", "FORCE_CLOSED", "STOPPED"):
                    remove_stop(trade.get("stock", ""))
        except Exception as e:
            Log.warn(f"[DYN-SL] EOD failed: {e}")

    # Pair trading: save pairs state
    if PAIR_TRADING_ENABLED:
        try:
            save_pairs_state()
            Log.info("[PAIRS] Co-integration state saved")
        except Exception as e:
            Log.warn(f"[PAIRS] EOD failed: {e}")

    # Anomaly detector: retrain Isolation Forest EOD
    if ANOMALY_DETECTOR_ENABLED:
        try:
            stock_dfs = {}
            for _sym in top_stocks:
                _df = get_live_data(_sym)
                if _df is not None:
                    stock_dfs[_sym] = _df
            if stock_dfs:
                train_result = train_isolation_forest(stock_dfs)
                Log.info(f"[ANOMALY] Retrained: {train_result.get('n_samples', 0)} samples")
            save_anomaly_state()
        except Exception as e:
            Log.warn(f"[ANOMALY] EOD failed: {e}")

    # Transformer: save state
    if TRANSFORMER_ENABLED:
        try:
            save_transformer_state()
            Log.info("[TRANSFORMER] State saved")
        except Exception as e:
            Log.warn(f"[TRANSFORMER] EOD failed: {e}")

    # Order book: save state
    if ORDERBOOK_SIM_ENABLED:
        try:
            save_orderbook_state()
            Log.info("[ORDERBOOK] State saved")
        except Exception as e:
            Log.warn(f"[ORDERBOOK] EOD failed: {e}")

    # Model versioning: auto-promote check
    if MODEL_VERSIONING_ENABLED:
        try:
            for mtype in ["random_forest", "xgboost", "lstm_daily"]:
                result = auto_promote_check(mtype)
                rec = result.get("recommendation", "HOLD")
                if result.get("auto_promoted"):
                    Log.success(f"[VERSIONING] {mtype} → PROMOTED to production!")
                elif rec != "HOLD":
                    Log.info(f"[VERSIONING] {mtype}: {rec}")
            save_versioning_state()
            Log.info("[VERSIONING] A/B state saved")
        except Exception as e:
            Log.warn(f"[VERSIONING] EOD failed: {e}")

    # ── Phase 12 EOD hooks ──

    # Adaptive Executor: save execution plans & fire pending slices
    if ADAPTIVE_EXECUTOR_ENABLED:
        try:
            ex_status = get_executor_status()
            save_executor_state(ex_status)
            active_plans = ex_status.get("active_plans", 0)
            avg_slip = ex_status.get("avg_slippage_bps", 0)
            Log.info(f"[EXECUTOR] Plans: {active_plans} active | Avg slippage: {avg_slip:.1f} bps")
        except Exception as e:
            Log.warn(f"[EXECUTOR] EOD failed: {e}")

    # RL Rebalancer: check rebalance & record outcomes
    if RL_REBALANCER_ENABLED:
        try:
            # Compute current portfolio weights
            open_trades = [t for t in state.get("active_trades", []) if t.get("status") == "OPEN"]
            total_val = sum(t.get("price", 0) * t.get("qty", 0) for t in open_trades) or 1.0
            cur_weights = {t["stock"]: (t.get("price", 0) * t.get("qty", 0)) / total_val for t in open_trades}

            rl_action = get_rl_rebalance_action(
                symbols=top_stocks,
                current_weights=cur_weights,
                regime=regime_cache.get("data", {}).get("regime", "SIDEWAYS"),
                drawdown_pct=abs(state.get("total_profit_pct", 0)),
                breadth_signal=breadth_cache.get("data", {}).get("signal", "NEUTRAL"),
            )
            action_name = rl_action.get("action_name", "HOLD")
            if rl_action.get("should_rebalance"):
                Log.info(f"[RL-REBAL] Recommends: {action_name} (HHI={rl_action.get('herfindahl', 0):.3f})")
            else:
                Log.info(f"[RL-REBAL] Action: HOLD")

            # Record outcome: reward = today's P&L normalised
            day_pnl_pct = state.get("total_profit_pct", 0)
            reward = day_pnl_pct * 10  # scale for Q-learning
            record_rebalance_outcome(
                rl_action["state_key"], rl_action["action"], reward,
            )
            save_rl_rebal_state()
            Log.info("[RL-REBAL] State saved")
        except Exception as e:
            Log.warn(f"[RL-REBAL] EOD failed: {e}")

    # Options Synthesizer: generate hedge recommendations
    if OPTIONS_SYNTH_ENABLED:
        try:
            regime_str = regime_cache.get("data", {}).get("regime", "SIDEWAYS")
            var_pct = 2.0
            if VAR_STRESS_ENABLED:
                try:
                    from var_stress import analyse_var_stress
                    vr = analyse_var_stress(symbols=top_stocks, capital=CAPITAL)
                    var_pct = abs(vr.get("monte_carlo_var_95", {}).get("var_pct", 2.0))
                except Exception:
                    pass
            portfolio_delta = 0.0
            if GREEKS_HEATMAP_ENABLED:
                try:
                    _greeks_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "greeks_heatmap.json")
                    if os.path.exists(_greeks_path):
                        with open(_greeks_path, "r") as _gf:
                            greeks_dat = json.load(_gf)
                        portfolio_delta = greeks_dat.get("totals", {}).get("delta", 0)
                except Exception:
                    pass

            synth_result = synthesize_strategy(
                state.get("active_trades", []),
                CAPITAL,
                regime=regime_str,
                var_pct=var_pct,
                portfolio_delta=portfolio_delta,
            )
            save_synth_state(synth_result)
            rec = synth_result.get("recommendation", "NONE")
            cost_pct = synth_result.get("cost_pct_of_capital", 0)
            Log.info(f"[OPT-SYNTH] Best: {rec} | Hedge cost: {cost_pct:.2f}% of capital")
        except Exception as e:
            Log.warn(f"[OPT-SYNTH] EOD failed: {e}")

    # Causal Engine: save discovered graph
    if CAUSAL_ENGINE_ENABLED:
        try:
            save_causal_state()
            status = get_causal_status()
            n_app = status.get("total_approved", 0)
            density = status.get("graph_density", 0)
            Log.info(f"[CAUSAL] Graph saved — {n_app} approved features, density {density:.2f}")
        except Exception as e:
            Log.warn(f"[CAUSAL] EOD failed: {e}")

    # GA Evolver: run evolution cycle on accumulated data
    if GA_EVOLVER_ENABLED:
        try:
            ga_state = evolve_strategies(
                symbols=top_stocks,
                existing_state=ga_state if ga_state else None,
            )
            save_evolver_state(ga_state)
            gen = ga_state.get("generation", 0)
            fit = ga_state.get("best_fitness", 0)
            Log.info(f"[GA] Evolution complete — gen {gen}, best Sharpe {fit:.3f}")
            best = get_best_strategy(ga_state)
            if best:
                Log.info(f"[GA] Best params: RSI-buy={best.get('rsi_buy_threshold')}, "
                         f"RSI-sell={best.get('rsi_sell_threshold')}, "
                         f"ATR-SL={best.get('atr_sl_multiplier'):.1f}x")
        except Exception as e:
            Log.warn(f"[GA] EOD failed: {e}")

    # Debate System: save accumulated debate state
    if DEBATE_SYSTEM_ENABLED:
        try:
            save_debate_state(debate_state)
            ds = get_debate_status(debate_state)
            n_total = ds.get("total_debates", 0)
            n_app = ds.get("approvals", 0)
            n_rej = ds.get("rejections", 0)
            n_veto = ds.get("veto_count", 0)
            Log.info(f"[DEBATE] Saved — {n_total} debates ({n_app} approved, {n_rej} rejected, {n_veto} vetoes)")
        except Exception as e:
            Log.warn(f"[DEBATE] EOD failed: {e}")

    # RL Trade Agent: batch-train on replay buffer and save
    if RL_TRADE_AGENT_ENABLED:
        try:
            avg_loss = batch_train_agent(n_steps=30)
            save_agent_state()
            status = get_agent_status()
            Log.info(f"[RL-AGENT] Trained — steps={status.get('train_steps', 0)}, "
                     f"ε={status.get('epsilon', 0):.3f}, trades={status.get('total_trades', 0)}, "
                     f"win_rate={status.get('win_rate', 0):.1f}%, loss={avg_loss:.6f}")
        except Exception as e:
            Log.warn(f"[RL-AGENT] EOD failed: {e}")

    # Sentiment Momentum: compute market-wide SMI and save
    if SENTIMENT_MOMENTUM_ENABLED:
        try:
            _smi_fb = finbert_cache.get("data", {}) if 'finbert_cache' in dir() else {}
            _smi_news = news_cache.get("data", {}) if 'news_cache' in dir() else {}
            _smi_mood = market_mood_cache.get("mood", {})
            _stock_dfs = {}
            for _ss in top_stocks:
                try:
                    _sdf = get_live_data(_ss)
                    if _sdf is not None:
                        _stock_dfs[_ss] = _sdf
                except Exception:
                    pass
            mkt_smi = compute_market_smi(top_stocks, _stock_dfs, _smi_fb, _smi_news, _smi_mood)
            smi_state = record_smi_snapshot(smi_state, mkt_smi)
            save_smi_state(smi_state)
            Log.info(f"[SMI] Market SMI={mkt_smi.get('market_smi', 0):.3f} "
                     f"({mkt_smi.get('market_label', 'NEUTRAL')}), "
                     f"best={mkt_smi.get('best_sentiment', '—')}, "
                     f"worst={mkt_smi.get('worst_sentiment', '—')}")
        except Exception as e:
            Log.warn(f"[SMI] EOD failed: {e}")


if __name__ == "__main__":
    run_sniper()
