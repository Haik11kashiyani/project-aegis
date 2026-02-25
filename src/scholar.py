"""
====================================================
PROJECT AEGIS - The Scholar (Night Training) v2
====================================================
Runs every night AFTER market close.
  1. Scans ALL stocks in the watchlist
  2. Engineers 12 technical features + sentiment score
  3. Trains 4 models per stock:
       - Random Forest  (ML - logic-based voter)
       - XGBoost        (ML - gradient-boosted voter)
       - Daily LSTM     (DL - 60-day pattern voter)
       - Intraday LSTM  (DL - 15-min candle pattern voter)
  4. Validates with Time-Series Split (anti-overfit)
  5. Ranks stocks by predicted probability -> picks top N
  6. Saves all Brain files for the Sniper
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
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Local config
from config import (
    STOCK_WATCHLIST, TOP_N_STOCKS, DATA_PERIOD,
    RF_ESTIMATORS, RF_MIN_SPLIT, RF_MAX_DEPTH, RF_FEATURES,
    XGB_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE,
    LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_DROPOUT,
    INTRADAY_LOOKBACK, INTRADAY_EPOCHS, INTRADAY_BATCH_SIZE,
    INTRADAY_DROPOUT, INTRADAY_PERIOD,
    CONFIDENCE_THRESHOLD,
    SENTIMENT_ENABLED, SENTIMENT_MAX_ARTICLES, SENTIMENT_LOOKBACK_DAYS,
    model_paths, RANKING_FILE, TRADE_LOG_FILE,
)
from sentiment import get_sentiment_score


# --------------------------------------------------
#  1.  DATA DOWNLOAD + FEATURE ENGINEERING (Daily)
# --------------------------------------------------
def fetch_and_engineer(symbol: str, period: str = DATA_PERIOD) -> pd.DataFrame:
    """Download daily OHLCV data and compute 12 features (incl. sentiment)."""
    print(f"\n[DOWN] Downloading {symbol} ({period}) ...")
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    df = flatten_yf_columns(df)

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}.")

    # --- 11 Technical Indicators ---
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

    # --- 12th Feature: Sentiment Score ---
    if SENTIMENT_ENABLED:
        score = get_sentiment_score(symbol, SENTIMENT_MAX_ARTICLES, SENTIMENT_LOOKBACK_DAYS)
        df["Sentiment_Score"] = score
    else:
        df["Sentiment_Score"] = 0.0

    # --- Target: 1 if TOMORROW's close > TODAY's close ---
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    print(f"   [OK] {len(df)} rows after feature engineering.")
    return df


# --------------------------------------------------
#  1b. INTRADAY DATA (15-min candles)
# --------------------------------------------------
def fetch_intraday(symbol: str) -> pd.DataFrame:
    """Download 15-minute candle data for intraday LSTM."""
    print(f"   [DOWN] Downloading {symbol} (15m, {INTRADAY_PERIOD}) ...")
    df = yf.download(symbol, period=INTRADAY_PERIOD, interval="15m", progress=False)
    df = flatten_yf_columns(df)
    if df.empty:
        raise RuntimeError(f"No intraday data for {symbol}.")
    print(f"   [OK] {len(df)} intraday candles.")
    return df


# --------------------------------------------------
#  2.  RANDOM FOREST
# --------------------------------------------------
def train_random_forest(df: pd.DataFrame) -> RandomForestClassifier:
    print("   [RF] Training Random Forest ...")
    X = df[RF_FEATURES]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        min_samples_split=RF_MIN_SPLIT,
        max_depth=RF_MAX_DEPTH,
        random_state=42, n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    precisions = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        prec = precision_score(y.iloc[te], preds, zero_division=0)
        precisions.append(prec)
        print(f"      Fold {fold}: Precision={prec:.3f}")

    avg = np.mean(precisions)
    print(f"      Avg Precision: {avg:.3f}")

    model.fit(X.iloc[:-1], y.iloc[:-1])
    return model


# --------------------------------------------------
#  3.  XGBOOST
# --------------------------------------------------
def train_xgboost(df: pd.DataFrame) -> XGBClassifier:
    print("   [XGB] Training XGBoost ...")
    X = df[RF_FEATURES]
    y = df["Target"]

    model = XGBClassifier(
        n_estimators=XGB_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    precisions = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        prec = precision_score(y.iloc[te], preds, zero_division=0)
        precisions.append(prec)
        print(f"      Fold {fold}: Precision={prec:.3f}")

    avg = np.mean(precisions)
    print(f"      Avg Precision: {avg:.3f}")

    model.fit(X.iloc[:-1], y.iloc[:-1])
    return model


# --------------------------------------------------
#  4.  DAILY LSTM (60-day lookback on daily closes)
# --------------------------------------------------
def train_daily_lstm(df: pd.DataFrame):
    print("   [LSTM] Training Daily LSTM ...")
    close = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(LSTM_LOOKBACK, len(scaled) - 1):
        X.append(scaled[i - LSTM_LOOKBACK:i, 0])
        y.append(1 if scaled[i + 1, 0] > scaled[i, 0] else 0)

    X = np.array(X).reshape(-1, LSTM_LOOKBACK, 1)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_LOOKBACK, 1)),
        Dropout(LSTM_DROPOUT),
        LSTM(64, return_sequences=False),
        Dropout(LSTM_DROPOUT),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=0,
    )
    val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"      Daily LSTM Val Accuracy: {val_acc:.3f}")
    return model, scaler


# --------------------------------------------------
#  5.  INTRADAY LSTM (48 x 15-min candle lookback)
# --------------------------------------------------
def train_intraday_lstm(df_intraday: pd.DataFrame):
    print("   [INTRA] Training Intraday LSTM (15-min candles) ...")
    close = df_intraday["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(INTRADAY_LOOKBACK, len(scaled) - 1):
        X.append(scaled[i - INTRADAY_LOOKBACK:i, 0])
        y.append(1 if scaled[i + 1, 0] > scaled[i, 0] else 0)

    if len(X) < 50:
        print("      WARNING: Not enough intraday data. Skipping.")
        return None, scaler

    X = np.array(X).reshape(-1, INTRADAY_LOOKBACK, 1)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(48, return_sequences=True, input_shape=(INTRADAY_LOOKBACK, 1)),
        Dropout(INTRADAY_DROPOUT),
        LSTM(48, return_sequences=False),
        Dropout(INTRADAY_DROPOUT),
        Dense(24, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=INTRADAY_EPOCHS, batch_size=INTRADAY_BATCH_SIZE,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=0,
    )
    val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"      Intraday LSTM Val Accuracy: {val_acc:.3f}")
    return model, scaler


# --------------------------------------------------
#  6.  SELF-CORRECTION
# --------------------------------------------------
def review_yesterday(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not os.path.exists(TRADE_LOG_FILE):
        return df
    try:
        log = pd.read_csv(TRADE_LOG_FILE)
        log = log[log["Stock"] == symbol]
        bad_calls = log[(log["AI_Confidence"] > 0.75) & (log["Actual_Profit"] < 0)]
        if bad_calls.empty:
            return df
        bad_dates = pd.to_datetime(bad_calls["Date"])
        mask = df.index.isin(bad_dates)
        flipped = mask.sum()
        df.loc[mask, "Target"] = 0
        if flipped:
            print(f"   [LEARN] Self-correction ({symbol}): flipped {flipped} targets.")
    except Exception:
        pass
    return df


# --------------------------------------------------
#  7.  TRAIN ALL 4 MODELS FOR ONE STOCK
# --------------------------------------------------
def train_stock(symbol: str) -> dict:
    """Train RF, XGB, daily LSTM, intraday LSTM for one stock."""
    paths = model_paths(symbol)
    result = {"symbol": symbol, "rf_prob": 0, "xgb_prob": 0,
              "lstm_prob": 0, "intraday_prob": 0, "sentiment": 0}

    # -- Daily data --
    try:
        df = fetch_and_engineer(symbol)
    except Exception as e:
        print(f"   [FAIL] Skipping {symbol}: {e}")
        return result

    df = review_yesterday(df, symbol)
    result["sentiment"] = float(df["Sentiment_Score"].iloc[-1])

    # -- Train RF --
    try:
        rf = train_random_forest(df)
        joblib.dump(rf, paths["rf"])
        last = df.iloc[-1:][RF_FEATURES]
        result["rf_prob"] = float(rf.predict_proba(last)[0][1])
    except Exception as e:
        print(f"   [WARN] RF failed for {symbol}: {e}")

    # -- Train XGBoost --
    try:
        xgb = train_xgboost(df)
        joblib.dump(xgb, paths["xgb"])
        last = df.iloc[-1:][RF_FEATURES]
        result["xgb_prob"] = float(xgb.predict_proba(last)[0][1])
    except Exception as e:
        print(f"   [WARN] XGB failed for {symbol}: {e}")

    # -- Train Daily LSTM --
    try:
        lstm, scaler = train_daily_lstm(df)
        lstm.save(paths["lstm"])
        joblib.dump(scaler, paths["lstm_scaler"])
        close_last = df["Close"].values[-LSTM_LOOKBACK:].reshape(-1, 1)
        scaled = scaler.transform(close_last).reshape(1, LSTM_LOOKBACK, 1)
        result["lstm_prob"] = float(lstm.predict(scaled, verbose=0)[0][0])
    except Exception as e:
        print(f"   [WARN] Daily LSTM failed for {symbol}: {e}")

    # -- Train Intraday LSTM --
    try:
        df_intra = fetch_intraday(symbol)
        intra_model, intra_scaler = train_intraday_lstm(df_intra)
        if intra_model is not None:
            intra_model.save(paths["intraday_lstm"])
            joblib.dump(intra_scaler, paths["intraday_scaler"])
            close_last = df_intra["Close"].values[-INTRADAY_LOOKBACK:].reshape(-1, 1)
            scaled = intra_scaler.transform(close_last).reshape(1, INTRADAY_LOOKBACK, 1)
            result["intraday_prob"] = float(intra_model.predict(scaled, verbose=0)[0][0])
    except Exception as e:
        print(f"   [WARN] Intraday LSTM failed for {symbol}: {e}")

    return result


# --------------------------------------------------
#  8.  RANK STOCKS -> PICK TOP N
# --------------------------------------------------
def rank_stocks(results: list) -> pd.DataFrame:
    """Rank all stocks by average model probability. Save to CSV."""
    df = pd.DataFrame(results)
    df["avg_prob"] = (df["rf_prob"] + df["xgb_prob"] +
                      df["lstm_prob"] + df["intraday_prob"]) / 4.0
    df["votes"] = ((df["rf_prob"] > CONFIDENCE_THRESHOLD).astype(int) +
                   (df["xgb_prob"] > CONFIDENCE_THRESHOLD).astype(int) +
                   (df["lstm_prob"] > 0.55).astype(int) +
                   (df["intraday_prob"] > 0.55).astype(int))
    df = df.sort_values("avg_prob", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["date"] = datetime.now().strftime("%Y-%m-%d")

    os.makedirs("data", exist_ok=True)
    df.to_csv(RANKING_FILE, index=False)
    return df


# --------------------------------------------------
#  9.  MAIN - ORCHESTRATOR
# --------------------------------------------------
def main():
    print("=" * 60)
    print("  PROJECT AEGIS - THE SCHOLAR v2 (Night Training)")
    print(f"   Watchlist : {len(STOCK_WATCHLIST)} stocks")
    print(f"   Top-N     : {TOP_N_STOCKS}")
    print(f"   Models    : RF + XGBoost + Daily LSTM + Intraday LSTM")
    print(f"   Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Train all stocks
    results = []
    for sym in STOCK_WATCHLIST:
        print(f"\n{'=' * 50}")
        print(f"  Processing: {sym}")
        print(f"{'=' * 50}")
        res = train_stock(sym)
        results.append(res)
        print(f"   RF={res['rf_prob']:.3f}  XGB={res['xgb_prob']:.3f}  "
              f"LSTM={res['lstm_prob']:.3f}  Intra={res['intraday_prob']:.3f}  "
              f"Sent={res['sentiment']:.3f}")

    # Rank and pick top N
    ranking_df = rank_stocks(results)

    print(f"\n{'=' * 60}")
    print(f"  STOCK RANKING (Top {TOP_N_STOCKS} will be traded)")
    print(f"{'=' * 60}")
    for _, row in ranking_df.iterrows():
        star = ">>>" if row["rank"] <= TOP_N_STOCKS else "   "
        print(f"   {star} #{int(row['rank']):<2}  {row['symbol']:<16}  "
              f"AvgProb={row['avg_prob']:.3f}  Votes={int(row['votes'])}/4")

    top_picks = ranking_df.head(TOP_N_STOCKS)["symbol"].tolist()
    print(f"\n  Today's picks: {', '.join(top_picks)}")
    print("\n[OK] Scholar v2 complete. All brains ready for the Sniper.")


if __name__ == "__main__":
    main()
