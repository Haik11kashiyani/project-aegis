"""
====================================================
PROJECT AEGIS - Backtester v2 (Safety Check)
====================================================
Run this BEFORE deploying to verify the strategy works
on historical data.

Tests RF + XGBoost with Walk-Forward Validation so the
model never peeks into the future. Multi-stock support.

Usage:
    python src/backtest.py
====================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
import indicators as ta
from indicators import flatten_yf_columns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, accuracy_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from config import (
    STOCK_WATCHLIST, TOP_N_STOCKS, DATA_PERIOD,
    RF_ESTIMATORS, RF_MIN_SPLIT, RF_MAX_DEPTH, RF_FEATURES,
    XGB_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE,
    ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER,
    CAPITAL, MAX_BULLETS, CONFIDENCE_THRESHOLD,
)


def fetch_and_engineer(symbol: str, period: str) -> pd.DataFrame:
    """Same feature engineering as scholar.py -- must stay in sync."""
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    df = flatten_yf_columns(df)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")

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
    df["Sentiment_Score"] = 0.0   # Neutral in backtest (no historical sentiment)
    df["Target"]       = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df


def walk_forward_backtest(df: pd.DataFrame, symbol: str, n_splits: int = 5):
    """Walk-forward validation with RF + XGBoost."""
    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD BACKTEST: {symbol}")
    print(f"{'=' * 60}")

    X = df[RF_FEATURES]
    y = df["Target"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # --- RF ---
    rf_preds_all, rf_actuals_all = [], []
    print("\n   [RF] Random Forest:")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, min_samples_split=RF_MIN_SPLIT,
            max_depth=RF_MAX_DEPTH, random_state=42, n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        preds = (probs > CONFIDENCE_THRESHOLD).astype(int)
        prec = precision_score(y.iloc[test_idx], preds, zero_division=0)
        acc = accuracy_score(y.iloc[test_idx], preds)
        rf_preds_all.extend(preds)
        rf_actuals_all.extend(y.iloc[test_idx])
        print(f"      Fold {fold}: Prec={prec:.3f}  Acc={acc:.3f}  Signals={preds.sum()}")

    rf_preds_all = np.array(rf_preds_all)
    rf_actuals_all = np.array(rf_actuals_all)
    rf_prec = precision_score(rf_actuals_all, rf_preds_all, zero_division=0)
    print(f"   [RF] Overall Precision: {rf_prec:.3f}  Signals: {rf_preds_all.sum()}")

    # --- XGBoost ---
    xgb_preds_all, xgb_actuals_all = [], []
    print("\n   [XGB] XGBoost:")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = XGBClassifier(
            n_estimators=XGB_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE, eval_metric="logloss",
            random_state=42, use_label_encoder=False, verbosity=0,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        preds = (probs > CONFIDENCE_THRESHOLD).astype(int)
        prec = precision_score(y.iloc[test_idx], preds, zero_division=0)
        acc = accuracy_score(y.iloc[test_idx], preds)
        xgb_preds_all.extend(preds)
        xgb_actuals_all.extend(y.iloc[test_idx])
        print(f"      Fold {fold}: Prec={prec:.3f}  Acc={acc:.3f}  Signals={preds.sum()}")

    xgb_preds_all = np.array(xgb_preds_all)
    xgb_actuals_all = np.array(xgb_actuals_all)
    xgb_prec = precision_score(xgb_actuals_all, xgb_preds_all, zero_division=0)
    print(f"   [XGB] Overall Precision: {xgb_prec:.3f}  Signals: {xgb_preds_all.sum()}")

    # --- Combined (both agree) ---
    combined = (rf_preds_all & xgb_preds_all)
    comb_prec = precision_score(rf_actuals_all, combined, zero_division=0)
    print(f"\n   [COMBINED RF+XGB] Precision: {comb_prec:.3f}  Signals: {combined.sum()}")

    return rf_prec, xgb_prec, comb_prec


def simulate_pnl(df: pd.DataFrame, symbol: str):
    """Simulate P&L using RF+XGB consensus on historical data."""
    print(f"\n{'=' * 60}")
    print(f"  P&L SIMULATION: {symbol}")
    print(f"{'=' * 60}")

    split = int(len(df) * 0.7)
    train_df = df.iloc[:split]
    sim_df = df.iloc[split:].copy()

    # Train both models
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, min_samples_split=RF_MIN_SPLIT,
        max_depth=RF_MAX_DEPTH, random_state=42, n_jobs=-1,
    )
    rf.fit(train_df[RF_FEATURES], train_df["Target"])

    xgb = XGBClassifier(
        n_estimators=XGB_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE, eval_metric="logloss",
        random_state=42, use_label_encoder=False, verbosity=0,
    )
    xgb.fit(train_df[RF_FEATURES], train_df["Target"])

    bullet_size = CAPITAL / MAX_BULLETS
    total_profit = 0.0
    trades = []

    for i in range(len(sim_df) - 1):
        row = sim_df.iloc[i:i + 1]
        rf_prob = rf.predict_proba(row[RF_FEATURES])[0][1]
        xgb_prob = xgb.predict_proba(row[RF_FEATURES])[0][1]

        # Need both to agree
        if rf_prob < CONFIDENCE_THRESHOLD or xgb_prob < CONFIDENCE_THRESHOLD:
            continue

        entry_price = float(row["Close"].values[0])
        atr_val = float(row["ATR"].values[0])
        stop_loss = entry_price - (ATR_STOP_MULTIPLIER * atr_val)
        target = entry_price + (ATR_TARGET_MULTIPLIER * atr_val)
        qty = max(1, int(bullet_size / entry_price))

        next_row = sim_df.iloc[i + 1]
        next_high = float(next_row["High"])
        next_low = float(next_row["Low"])
        next_close = float(next_row["Close"])

        if next_low <= stop_loss:
            exit_price = stop_loss
            result = "STOP_LOSS"
        elif next_high >= target:
            exit_price = target
            result = "TARGET_HIT"
        else:
            exit_price = next_close
            result = "HOLD_CLOSE"

        pnl = (exit_price - entry_price) * qty
        total_profit += pnl
        trades.append({
            "Date": sim_df.index[i].strftime("%Y-%m-%d"),
            "Stock": symbol,
            "Entry": round(entry_price, 2),
            "Exit": round(exit_price, 2),
            "Qty": qty,
            "PnL": round(pnl, 2),
            "Result": result,
            "RF_Conf": round(rf_prob, 3),
            "XGB_Conf": round(xgb_prob, 3),
        })

    if not trades:
        print("   No trades triggered during the simulation.")
        return 0.0

    trades_df = pd.DataFrame(trades)
    wins = len(trades_df[trades_df["PnL"] > 0])
    losses = len(trades_df[trades_df["PnL"] <= 0])
    total = len(trades_df)
    win_rate = (wins / total * 100) if total > 0 else 0

    print(f"   Period       : {trades_df['Date'].iloc[0]} -> {trades_df['Date'].iloc[-1]}")
    print(f"   Total Trades : {total}")
    print(f"   Wins         : {wins}  ({win_rate:.1f}%)")
    print(f"   Losses       : {losses}")
    print(f"   Total P&L    : Rs.{total_profit:.2f}")
    print(f"   Avg P&L/Trade: Rs.{total_profit / total:.2f}")
    print(f"   Best Trade   : Rs.{trades_df['PnL'].max():.2f}")
    print(f"   Worst Trade  : Rs.{trades_df['PnL'].min():.2f}")

    print("\n   Last 10 Simulated Trades:")
    print(trades_df.tail(10).to_string(index=False))
    return total_profit


def main():
    print("=" * 60)
    print("  PROJECT AEGIS - BACKTESTER v2")
    print(f"   Stocks  : {len(STOCK_WATCHLIST)} in watchlist")
    print(f"   Period  : {DATA_PERIOD}")
    print(f"   Models  : RF + XGBoost (walk-forward)")
    print(f"   Conf    : {CONFIDENCE_THRESHOLD}")
    print("=" * 60)

    grand_total_pnl = 0.0
    stock_results = []

    for symbol in STOCK_WATCHLIST[:TOP_N_STOCKS]:
        try:
            df = fetch_and_engineer(symbol, DATA_PERIOD)
            print(f"\n   {symbol}: {len(df)} rows")

            rf_prec, xgb_prec, comb_prec = walk_forward_backtest(df, symbol)
            pnl = simulate_pnl(df, symbol)
            grand_total_pnl += pnl

            stock_results.append({
                "Stock": symbol,
                "RF_Prec": round(rf_prec, 3),
                "XGB_Prec": round(xgb_prec, 3),
                "Combined_Prec": round(comb_prec, 3),
                "Sim_PnL": round(pnl, 2),
            })
        except Exception as e:
            print(f"   [FAIL] {symbol}: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  BACKTEST SUMMARY (All Stocks)")
    print(f"{'=' * 60}")
    if stock_results:
        summary_df = pd.DataFrame(stock_results)
        print(summary_df.to_string(index=False))
        print(f"\n   Grand Total Simulated P&L: Rs.{grand_total_pnl:.2f}")

    print(f"\n{'=' * 60}")
    print("  RECOMMENDATION:")
    print("    Run this backtest weekly. If combined precision drops")
    print("    below 50%, STOP the sniper and retrain.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
