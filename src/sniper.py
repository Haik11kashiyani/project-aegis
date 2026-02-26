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
)
from sentiment import get_sentiment_score
from risk_guardian import RiskGuardian, NeuralSafetyNet, quick_safety_check

IST = pytz.timezone("Asia/Kolkata")


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
            print(f"   [WARN] Could not load ranking: {e}")
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
    print(f"   [OK] {symbol}: {loaded}/4 models loaded")
    return models


# --------------------------------------------------
#   DATA FETCHER (daily + indicators + sentiment)
# --------------------------------------------------
def get_live_data(symbol: str) -> pd.DataFrame:
    """Fetch recent daily data + indicators for the ensemble decision."""
    try:
        # Need at least 200+ trading days for SMA_200 — use 2 years
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        df = flatten_yf_columns(df)
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
        df = yf.download(symbol, period="5d", interval="15m", progress=False)
        df = flatten_yf_columns(df)
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

    print(f"   [VOTE] {symbol}: RF={rf_conf:.2f} XGB={xgb_conf:.2f} "
          f"LSTM={lstm_conf:.2f} Intra={intra_conf:.2f} => {votes}/{MIN_VOTES_TO_BUY} needed")

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
#   MAIN SNIPER LOOP
# --------------------------------------------------
def run_sniper():
    print("=" * 60)
    print("  PROJECT AEGIS - THE SNIPER v2 (Live Paper-Trading)")
    print(f"   Capital : Rs.{CAPITAL}")
    print(f"   Bullets : {MAX_BULLETS}  (Rs.{CAPITAL / MAX_BULLETS:.0f} each)")
    print(f"   Target  : {DAILY_TARGET * 100:.1f}% daily")
    print(f"   Voting  : {MIN_VOTES_TO_BUY}/4 models must agree")
    print(f"   Guardian: {'ENABLED' if RISK_GUARDIAN_ENABLED else 'DISABLED'}")
    print(f"   Date    : {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    #  RISK GUARDIAN — Pre-flight safety check
    # ═══════════════════════════════════════════════════════
    guardian = None
    if RISK_GUARDIAN_ENABLED:
        print("\n   [GUARDIAN] Running pre-flight safety checks ...")
        allowed, reason = quick_safety_check()
        if not allowed:
            print(f"   [GUARDIAN] TRADING BLOCKED: {reason}")
            print("   [GUARDIAN] The Risk Guardian has determined it is not safe")
            print("              to trade today. This protects your real money.")
            print("              Review data/learner_report.json for details.")
            return
        guardian = RiskGuardian(capital=CAPITAL)
        print(f"   [GUARDIAN] Pre-flight passed. Risk level: "
              f"{guardian.get_status().get('risk_level', 'NORMAL')}")

    # Load top-N stocks from Scholar ranking
    top_stocks = load_top_stocks()
    print(f"\n   Today's stocks: {', '.join(top_stocks)}")

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
            print(f"   [WAIT] Market not open yet ({now.strftime('%H:%M IST')})")
            time.sleep(60)
            continue

        # ---- Force-close before market close ----
        if (now.hour == 15 and now.minute >= 10) or now.hour >= 16:
            print("\n   [CLOSE] Market closing. Force-exiting all open positions.")
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

        # ---- Monitor Active Trades ----
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
                # Check if guardian wants to stop trading
                if not guardian.guardian_active:
                    state["status"] = "GUARDIAN_STOP"
                    save_state(state)
                    print(f"\n   [GUARDIAN] 🚨 Trading HALTED by Risk Guardian")
                    _print_summary(state)
                    return

            result_icon = "[WIN]" if pnl_total > 0 else "[LOSS]"
            print(f"   {result_icon} {sym} closed ({action}): "
                  f"Entry Rs.{trade['price']} -> Exit Rs.{current_price}  "
                  f"P&L: Rs.{pnl_total:.2f} ({pnl_pct:.2f}%)")

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

        # ---- Fire New Bullet (round-robin through stocks) ----
        open_count = sum(1 for t in state["active_trades"] if t["status"] == "OPEN")
        time_since_last = time.time() - last_fire_time

        if open_count < MAX_BULLETS and time_since_last >= TIME_GAP:
            sym = top_stocks[stock_idx % len(top_stocks)]
            stock_idx += 1

            if sym in all_models:
                df = get_live_data(sym)
                if df is not None and not df.empty:
                    should_buy, votes, rf_c, xgb_c, lstm_c, intra_c = get_ensemble_signal(
                        sym, df, all_models[sym]
                    )
                    avg_conf = round((rf_c + xgb_c + lstm_c + intra_c) / 4, 3)

                    if should_buy:
                        current_price = float(df["Close"].iloc[-1])
                        current_atr = float(df["ATR"].iloc[-1])
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
                                    print(f"   [GUARDIAN] ❌ REJECTED: {g_reason}")
                                    continue
                                final_qty = adj_qty  # Use guardian-adjusted qty
                                print(f"   [GUARDIAN] ✅ APPROVED (qty adjusted: {pos['qty']} → {adj_qty})")

                            print(f"\n   [FIRE] BULLET #{open_count + 1} on {sym} @ Rs.{current_price:.2f}")
                            print(f"     Qty: {final_qty}  SL: Rs.{pos['stop_loss']}  "
                                  f"Target: Rs.{pos['target']}  Votes: {votes}/4")

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
        print(f"   [SCAN] {now.strftime('%H:%M IST')}  Day P&L: Rs.{state['total_profit']:.2f} "
              f"({total_pnl_pct}%)  Open: {open_count}/{MAX_BULLETS}")

        save_state(state)
        time.sleep(scan_interval)


# --------------------------------------------------
#   END-OF-DAY SUMMARY
# --------------------------------------------------
def _print_summary(state: dict):
    print("\n" + "=" * 60)
    print("  END-OF-DAY SUMMARY")
    print("=" * 60)
    print(f"   Status       : {state['status']}")
    print(f"   Stocks       : {', '.join(state.get('stocks_traded', []))}")
    print(f"   Total Trades : {state['trades_taken']}")
    print(f"   Won          : {state['trades_won']}")
    print(f"   Lost         : {state['trades_lost']}")
    win_rate = (
        (state["trades_won"] / state["trades_taken"] * 100)
        if state["trades_taken"] > 0 else 0
    )
    print(f"   Win Rate     : {win_rate:.1f}%")
    print(f"   Total P&L    : Rs.{state['total_profit']:.2f}  "
          f"({state.get('total_profit_pct', 0):.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_sniper()
