"""
====================================================
🧠 PROJECT AEGIS - The Learner (Off-Market Continuous Learning)
====================================================
Runs when the market is CLOSED (nights, weekends, holidays).
Instead of sitting idle, the bot learns and adapts:

  1. TRADE REVIEW  — Analyses every past trade to find patterns
                     in wins/losses (what went wrong, what worked)
  2. HYPERPARAMETER TUNING — Tests different model configs via
                     Bayesian-style random search + walk-forward CV
  3. REGIME DETECTION — Deep learning model that classifies the
                     current market as TRENDING / RANGING / VOLATILE
                     (adjusts strategy aggressively vs defensively)
  4. CONFIDENCE CALIBRATION — Checks if model probabilities match
                     real outcomes (if 80% confident → should win ~80%)
  5. ENSEMBLE WEIGHT OPTIMIZATION — Learns which of the 4 models
                     performs best for each stock → weighted voting
  6. RISK PARAMETER ADAPTATION — Adjusts stop-loss / targets based
                     on recent volatility and win-rate trends
  7. HEALTH REPORT — Generates a JSON report for the Sniper to read
                     next morning (which models to trust, risk settings)

This is the "brain that improves the brain" — meta-learning.
====================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
import indicators as ta
from indicators import flatten_yf_columns
import joblib
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from config import (
    STOCK_WATCHLIST, TOP_N_STOCKS, DATA_PERIOD,
    RF_ESTIMATORS, RF_MIN_SPLIT, RF_MAX_DEPTH, RF_FEATURES,
    XGB_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE,
    LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_DROPOUT,
    INTRADAY_LOOKBACK, INTRADAY_EPOCHS, INTRADAY_BATCH_SIZE,
    INTRADAY_DROPOUT, INTRADAY_PERIOD,
    CONFIDENCE_THRESHOLD, ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER,
    CAPITAL, MAX_BULLETS,
    model_paths, TRADE_LOG_FILE, RANKING_FILE,
)

# ──────────────────────────────────────────────────
#   PATHS
# ──────────────────────────────────────────────────
LEARNER_REPORT_FILE = "data/learner_report.json"
LEARNER_WEIGHTS_FILE = "data/ensemble_weights.json"
LEARNER_REGIME_MODEL = "models/regime_detector.h5"
LEARNER_REGIME_SCALER = "models/regime_scaler.pkl"
LEARNER_BEST_PARAMS_FILE = "data/best_params.json"


# ══════════════════════════════════════════════════
#  1. TRADE REVIEW — Analyse Past Trade Outcomes
# ══════════════════════════════════════════════════
def review_past_trades() -> dict:
    """
    Analyse ALL historical trades to find patterns:
    - Which stocks are consistently profitable?
    - Which time-of-day has best win-rate?
    - What confidence threshold actually works?
    - Average loss size vs average win size
    """
    report = {
        "total_trades": 0, "total_wins": 0, "total_losses": 0,
        "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "profit_factor": 0.0,
        "best_stocks": [], "worst_stocks": [],
        "optimal_confidence": CONFIDENCE_THRESHOLD,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    if not os.path.exists(TRADE_LOG_FILE):
        print("   [LEARN] No trade history found. Skipping review.")
        return report

    try:
        df = pd.read_csv(TRADE_LOG_FILE)
    except Exception as e:
        print(f"   [LEARN] Could not read trade log: {e}")
        return report

    closed = df[df["Status"].isin(["TARGET_HIT", "STOP_LOSS", "FORCE_CLOSED", "HOLD_CLOSE"])]
    if closed.empty:
        print("   [LEARN] No closed trades to review.")
        return report

    report["total_trades"] = len(closed)
    wins = closed[closed["Actual_Profit"] > 0]
    losses = closed[closed["Actual_Profit"] <= 0]
    report["total_wins"] = len(wins)
    report["total_losses"] = len(losses)
    report["win_rate"] = round(len(wins) / len(closed) * 100, 2) if len(closed) > 0 else 0

    report["avg_win"] = round(float(wins["Actual_Profit"].mean()), 2) if not wins.empty else 0
    report["avg_loss"] = round(float(losses["Actual_Profit"].mean()), 2) if not losses.empty else 0

    total_win_amt = float(wins["Actual_Profit"].sum()) if not wins.empty else 0
    total_loss_amt = abs(float(losses["Actual_Profit"].sum())) if not losses.empty else 0.01
    report["profit_factor"] = round(total_win_amt / total_loss_amt, 2)

    # Per-stock analysis
    if "Stock" in closed.columns:
        stock_stats = closed.groupby("Stock").agg(
            trades=("Actual_Profit", "count"),
            wins=("Actual_Profit", lambda x: (x > 0).sum()),
            total_pnl=("Actual_Profit", "sum"),
        ).reset_index()
        stock_stats["win_rate"] = (stock_stats["wins"] / stock_stats["trades"] * 100).round(1)
        stock_stats = stock_stats.sort_values("total_pnl", ascending=False)
        report["best_stocks"] = stock_stats.head(3)["Stock"].tolist()
        report["worst_stocks"] = stock_stats.tail(3)["Stock"].tolist()

    # Optimal confidence threshold discovery
    if "AI_Confidence" in closed.columns:
        best_thresh = CONFIDENCE_THRESHOLD
        best_pf = 0
        for thresh in np.arange(0.55, 0.95, 0.05):
            above = closed[closed["AI_Confidence"] >= thresh]
            if len(above) < 5:
                continue
            w = above[above["Actual_Profit"] > 0]["Actual_Profit"].sum()
            l = abs(above[above["Actual_Profit"] <= 0]["Actual_Profit"].sum()) + 0.01
            pf = w / l
            if pf > best_pf:
                best_pf = pf
                best_thresh = round(thresh, 2)
        report["optimal_confidence"] = best_thresh
        print(f"   [LEARN] Optimal confidence threshold: {best_thresh} (PF={best_pf:.2f})")

    print(f"   [LEARN] Trade Review: {report['total_trades']} trades, "
          f"WR={report['win_rate']}%, PF={report['profit_factor']}")
    return report


# ══════════════════════════════════════════════════
#  2. HYPERPARAMETER TUNING — Find Best Model Config
# ══════════════════════════════════════════════════
def tune_hyperparameters(symbol: str, df: pd.DataFrame) -> dict:
    """
    Random search over hyperparameter space with walk-forward CV.
    Returns the best params found for this stock.
    """
    print(f"   [TUNE] Hyperparameter tuning for {symbol} ...")

    X = df[RF_FEATURES]
    y = df["Target"]
    tscv = TimeSeriesSplit(n_splits=3)

    # ── RF Search Space ──
    rf_configs = [
        {"n_estimators": 150, "max_depth": 8, "min_samples_split": 30},
        {"n_estimators": 200, "max_depth": 10, "min_samples_split": 50},
        {"n_estimators": 250, "max_depth": 12, "min_samples_split": 40},
        {"n_estimators": 300, "max_depth": 15, "min_samples_split": 60},
        {"n_estimators": 200, "max_depth": 8, "min_samples_split": 80},
    ]

    best_rf_score = 0
    best_rf_params = rf_configs[1]  # default
    for cfg in rf_configs:
        scores = []
        for tr, te in tscv.split(X):
            model = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                min_samples_split=cfg["min_samples_split"],
                random_state=42, n_jobs=-1,
            )
            model.fit(X.iloc[tr], y.iloc[tr])
            probs = model.predict_proba(X.iloc[te])[:, 1]
            preds = (probs > CONFIDENCE_THRESHOLD).astype(int)
            scores.append(precision_score(y.iloc[te], preds, zero_division=0))
        avg = np.mean(scores)
        if avg > best_rf_score:
            best_rf_score = avg
            best_rf_params = cfg

    # ── XGB Search Space ──
    xgb_configs = [
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.7},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.04, "subsample": 0.85},
        {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.02, "subsample": 0.75},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.9},
    ]

    best_xgb_score = 0
    best_xgb_params = xgb_configs[1]  # default
    for cfg in xgb_configs:
        scores = []
        for tr, te in tscv.split(X):
            model = XGBClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                subsample=cfg["subsample"],
                colsample_bytree=XGB_COLSAMPLE,
                eval_metric="logloss",
                random_state=42, use_label_encoder=False, verbosity=0,
            )
            model.fit(X.iloc[tr], y.iloc[tr])
            probs = model.predict_proba(X.iloc[te])[:, 1]
            preds = (probs > CONFIDENCE_THRESHOLD).astype(int)
            scores.append(precision_score(y.iloc[te], preds, zero_division=0))
        avg = np.mean(scores)
        if avg > best_xgb_score:
            best_xgb_score = avg
            best_xgb_params = cfg

    result = {
        "symbol": symbol,
        "rf_params": best_rf_params,
        "rf_precision": round(best_rf_score, 4),
        "xgb_params": best_xgb_params,
        "xgb_precision": round(best_xgb_score, 4),
    }

    print(f"   [TUNE] {symbol} — Best RF precision: {best_rf_score:.3f}, "
          f"Best XGB precision: {best_xgb_score:.3f}")
    return result


# ══════════════════════════════════════════════════
#  3. MARKET REGIME DETECTION (Deep Learning)
# ══════════════════════════════════════════════════
def train_regime_detector(symbol: str) -> str:
    """
    Train a deep learning model to classify market regime:
      0 = TRENDING_UP   → aggressive trading OK
      1 = TRENDING_DOWN → avoid buying / go defensive
      2 = RANGING       → tight stops, quick profits
      3 = HIGH_VOLATILE → reduce position sizes or skip

    Uses 30-day rolling features to detect the regime.
    """
    print(f"   [REGIME] Training regime detector on {symbol} ...")

    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    df = flatten_yf_columns(df)
    if df.empty or len(df) < 200:
        print("   [REGIME] Not enough data. Skipping.")
        return "UNKNOWN"

    # Feature engineering for regime detection
    df["Returns"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Returns"].rolling(20).std()
    df["Volatility_60"] = df["Returns"].rolling(60).std()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_60"] = df["Close"].rolling(60).mean()
    df["Trend_Strength"] = (df["SMA_20"] - df["SMA_60"]) / df["SMA_60"]
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX"] = adx_df["ADX_14"] if adx_df is not None else 0
    df["Range_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["Volume_Change"] = df["Volume"].pct_change(5)

    df.dropna(inplace=True)
    if len(df) < 100:
        return "UNKNOWN"

    # Label regimes using rule-based approach (for training the DL model)
    conditions = []
    for i in range(len(df)):
        trend = df["Trend_Strength"].iloc[i]
        vol = df["Volatility_20"].iloc[i]
        vol_median = df["Volatility_20"].median()

        if trend > 0.02 and vol < vol_median * 1.5:
            conditions.append(0)  # TRENDING_UP
        elif trend < -0.02 and vol < vol_median * 1.5:
            conditions.append(1)  # TRENDING_DOWN
        elif vol >= vol_median * 1.5:
            conditions.append(3)  # HIGH_VOLATILE
        else:
            conditions.append(2)  # RANGING

    df["Regime"] = conditions

    # Prepare sequences for LSTM
    features = ["Returns", "Volatility_20", "Volatility_60", "Trend_Strength",
                "Range_Pct", "Volume_Change"]
    feature_data = df[features].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_data)

    lookback = 30
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(df["Regime"].iloc[i])

    X = np.array(X)
    y = np.array(y)

    # One-hot encode targets
    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y, num_classes=4)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_cat[:split], y_cat[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, len(features))),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15, batch_size=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=0,
    )

    val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"   [REGIME] Regime detector accuracy: {val_acc:.3f}")

    # Save
    os.makedirs("models", exist_ok=True)
    model.save(LEARNER_REGIME_MODEL)
    joblib.dump(scaler, LEARNER_REGIME_SCALER)

    # Predict current regime
    current_features = scaled[-lookback:].reshape(1, lookback, len(features))
    pred = model.predict(current_features, verbose=0)[0]
    regime_idx = int(np.argmax(pred))
    regime_map = {0: "TRENDING_UP", 1: "TRENDING_DOWN", 2: "RANGING", 3: "HIGH_VOLATILE"}
    current_regime = regime_map[regime_idx]
    confidence = float(pred[regime_idx])

    print(f"   [REGIME] Current market: {current_regime} (confidence: {confidence:.2f})")
    return current_regime


# ══════════════════════════════════════════════════
#  4. CONFIDENCE CALIBRATION
# ══════════════════════════════════════════════════
def calibrate_confidence(symbol: str, df: pd.DataFrame) -> dict:
    """
    Check if model confidence matches real outcomes.
    If model says 80% → it should be right ~80% of the time.
    Returns calibration metrics + recommended adjustment.
    """
    print(f"   [CALIB] Confidence calibration for {symbol} ...")

    X = df[RF_FEATURES]
    y = df["Target"]

    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # RF calibration
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SPLIT, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_brier = brier_score_loss(y_test, rf_probs)

    # XGB calibration
    xgb = XGBClassifier(
        n_estimators=XGB_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE, eval_metric="logloss",
        random_state=42, use_label_encoder=False, verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_brier = brier_score_loss(y_test, xgb_probs)

    # Check probability calibration
    try:
        rf_frac_pos, rf_mean_pred = calibration_curve(y_test, rf_probs, n_bins=5, strategy="uniform")
        xgb_frac_pos, xgb_mean_pred = calibration_curve(y_test, xgb_probs, n_bins=5, strategy="uniform")

        rf_calibration_error = float(np.mean(np.abs(rf_frac_pos - rf_mean_pred)))
        xgb_calibration_error = float(np.mean(np.abs(xgb_frac_pos - xgb_mean_pred)))
    except Exception:
        rf_calibration_error = 0.5
        xgb_calibration_error = 0.5

    result = {
        "symbol": symbol,
        "rf_brier_score": round(float(rf_brier), 4),
        "xgb_brier_score": round(float(xgb_brier), 4),
        "rf_calibration_error": round(rf_calibration_error, 4),
        "xgb_calibration_error": round(xgb_calibration_error, 4),
        "rf_reliable": rf_calibration_error < 0.15,
        "xgb_reliable": xgb_calibration_error < 0.15,
    }

    print(f"   [CALIB] {symbol}: RF Brier={rf_brier:.3f} (calib_err={rf_calibration_error:.3f}), "
          f"XGB Brier={xgb_brier:.3f} (calib_err={xgb_calibration_error:.3f})")
    return result


# ══════════════════════════════════════════════════
#  5. ENSEMBLE WEIGHT OPTIMIZATION
# ══════════════════════════════════════════════════
def optimize_ensemble_weights(symbol: str, df: pd.DataFrame) -> dict:
    """
    Learn optimal weights for the 4-model ensemble for each stock.
    Instead of equal voting, give more weight to models that perform
    better on recent data for this specific stock.
    """
    print(f"   [WEIGHTS] Optimizing ensemble weights for {symbol} ...")

    X = df[RF_FEATURES]
    y = df["Target"]
    close = df["Close"].values

    # Use last 20% as validation
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # Train and get probabilities on validation set
    # RF
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SPLIT, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_val)[:, 1]
    rf_acc = accuracy_score(y_val, (rf_probs > CONFIDENCE_THRESHOLD).astype(int))

    # XGB
    xgb = XGBClassifier(
        n_estimators=XGB_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE, eval_metric="logloss",
        random_state=42, use_label_encoder=False, verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_val)[:, 1]
    xgb_acc = accuracy_score(y_val, (xgb_probs > CONFIDENCE_THRESHOLD).astype(int))

    # Daily LSTM (approximate using close prices)
    lstm_acc = 0.5  # Default if can't compute
    try:
        scaler = MinMaxScaler()
        close_scaled = scaler.fit_transform(close.reshape(-1, 1))
        Xl, yl = [], []
        for i in range(LSTM_LOOKBACK, len(close_scaled) - 1):
            Xl.append(close_scaled[i - LSTM_LOOKBACK:i, 0])
            yl.append(1 if close_scaled[i + 1, 0] > close_scaled[i, 0] else 0)
        Xl = np.array(Xl).reshape(-1, LSTM_LOOKBACK, 1)
        yl = np.array(yl)
        split_l = int(len(Xl) * 0.8)
        if split_l > LSTM_LOOKBACK:
            lstm_model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(LSTM_LOOKBACK, 1)),
                Dropout(LSTM_DROPOUT),
                LSTM(64), Dropout(LSTM_DROPOUT),
                Dense(32, activation="relu"), Dense(1, activation="sigmoid"),
            ])
            lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            lstm_model.fit(Xl[:split_l], yl[:split_l], epochs=5, batch_size=32, verbose=0)
            lstm_preds = (lstm_model.predict(Xl[split_l:], verbose=0).flatten() > 0.55).astype(int)
            lstm_acc = accuracy_score(yl[split_l:], lstm_preds)
    except Exception:
        pass

    # Compute weights proportional to accuracy (softmax-style)
    intraday_acc = 0.5  # approximate baseline
    accs = np.array([rf_acc, xgb_acc, lstm_acc, intraday_acc])
    accs = np.clip(accs, 0.01, 1.0)  # avoid zero
    weights = accs / accs.sum()

    result = {
        "rf_weight": round(float(weights[0]), 4),
        "xgb_weight": round(float(weights[1]), 4),
        "lstm_weight": round(float(weights[2]), 4),
        "intraday_weight": round(float(weights[3]), 4),
        "rf_accuracy": round(float(rf_acc), 4),
        "xgb_accuracy": round(float(xgb_acc), 4),
        "lstm_accuracy": round(float(lstm_acc), 4),
    }

    print(f"   [WEIGHTS] {symbol}: RF={weights[0]:.2f} XGB={weights[1]:.2f} "
          f"LSTM={weights[2]:.2f} Intra={weights[3]:.2f}")
    return result


# ══════════════════════════════════════════════════
#  6. RISK PARAMETER ADAPTATION
# ══════════════════════════════════════════════════
def adapt_risk_parameters() -> dict:
    """
    Analyse recent trade outcomes and volatility to recommend
    adjustments to stop-loss multipliers, position sizes, etc.
    Designed for REAL MONEY safety.
    """
    print("   [RISK] Adapting risk parameters ...")

    risk_params = {
        "recommended_atr_stop": ATR_STOP_MULTIPLIER,
        "recommended_atr_target": ATR_TARGET_MULTIPLIER,
        "recommended_max_bullets": MAX_BULLETS,
        "recommended_confidence": CONFIDENCE_THRESHOLD,
        "risk_level": "NORMAL",  # CONSERVATIVE / NORMAL / AGGRESSIVE
        "position_size_multiplier": 1.0,  # Scale down if losing streak
    }

    if not os.path.exists(TRADE_LOG_FILE):
        risk_params["risk_level"] = "CONSERVATIVE"
        risk_params["position_size_multiplier"] = 0.5  # Start cautious with real money
        print("   [RISK] No trade history. Starting CONSERVATIVE.")
        return risk_params

    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        closed = df[df["Status"].isin(["TARGET_HIT", "STOP_LOSS", "FORCE_CLOSED"])]
        if closed.empty:
            risk_params["risk_level"] = "CONSERVATIVE"
            risk_params["position_size_multiplier"] = 0.5
            return risk_params
    except Exception:
        return risk_params

    # Recent performance (last 20 trades)
    recent = closed.tail(20)
    recent_wins = len(recent[recent["Actual_Profit"] > 0])
    recent_wr = recent_wins / len(recent) * 100 if len(recent) > 0 else 0

    # Consecutive losses detection
    recent_results = (recent["Actual_Profit"] > 0).tolist()
    max_consecutive_losses = 0
    current_streak = 0
    for won in recent_results:
        if not won:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    # ── Decision Logic ──
    if max_consecutive_losses >= 5 or recent_wr < 30:
        risk_params["risk_level"] = "EMERGENCY"
        risk_params["position_size_multiplier"] = 0.25  # Quarter positions
        risk_params["recommended_atr_stop"] = 2.0       # Wider stops
        risk_params["recommended_confidence"] = 0.85     # Higher threshold
        risk_params["recommended_max_bullets"] = 2       # Max 2 trades
        print(f"   [RISK] EMERGENCY: {max_consecutive_losses} consecutive losses, WR={recent_wr:.0f}%")

    elif max_consecutive_losses >= 3 or recent_wr < 45:
        risk_params["risk_level"] = "CONSERVATIVE"
        risk_params["position_size_multiplier"] = 0.5    # Half positions
        risk_params["recommended_atr_stop"] = 1.8        # Slightly wider
        risk_params["recommended_confidence"] = 0.80
        risk_params["recommended_max_bullets"] = 3
        print(f"   [RISK] CONSERVATIVE: WR={recent_wr:.0f}%, max streak={max_consecutive_losses}")

    elif recent_wr >= 65 and max_consecutive_losses <= 1:
        risk_params["risk_level"] = "CONFIDENT"
        risk_params["position_size_multiplier"] = 1.0
        risk_params["recommended_atr_stop"] = 1.5
        risk_params["recommended_confidence"] = 0.70
        risk_params["recommended_max_bullets"] = 5
        print(f"   [RISK] CONFIDENT: WR={recent_wr:.0f}% — normal operation")

    else:
        risk_params["risk_level"] = "NORMAL"
        risk_params["position_size_multiplier"] = 0.75   # Slightly conservative (real money)
        print(f"   [RISK] NORMAL: WR={recent_wr:.0f}%")

    return risk_params


# ══════════════════════════════════════════════════
#  7. GENERATE HEALTH REPORT (for Sniper to read)
# ══════════════════════════════════════════════════
def generate_health_report(trade_review: dict, calibrations: list,
                           weights: dict, risk_params: dict,
                           regimes: dict, tuned_params: list) -> dict:
    """
    Create a comprehensive JSON report that the Sniper reads
    every morning to adjust its behaviour.
    """
    # Determine if models are healthy enough to trade
    model_health = "HEALTHY"
    unhealthy_reasons = []

    if trade_review["win_rate"] < 40 and trade_review["total_trades"] >= 10:
        model_health = "DEGRADED"
        unhealthy_reasons.append(f"Win rate only {trade_review['win_rate']}%")

    if trade_review["profit_factor"] < 0.8 and trade_review["total_trades"] >= 10:
        model_health = "DEGRADED"
        unhealthy_reasons.append(f"Profit factor only {trade_review['profit_factor']}")

    # Check if any calibrations are poor
    for cal in calibrations:
        if not cal.get("rf_reliable") and not cal.get("xgb_reliable"):
            model_health = "DEGRADED"
            unhealthy_reasons.append(f"Models poorly calibrated for {cal['symbol']}")

    # Safety lock: if DEGRADED → Sniper should reduce risk
    trading_allowed = True
    if model_health == "DEGRADED" and risk_params["risk_level"] == "EMERGENCY":
        trading_allowed = False
        unhealthy_reasons.append("EMERGENCY + DEGRADED → Trading paused")

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_health": model_health,
        "trading_allowed": trading_allowed,
        "unhealthy_reasons": unhealthy_reasons,
        "trade_review": trade_review,
        "risk_params": risk_params,
        "ensemble_weights": weights,
        "market_regimes": regimes,
        "calibrations": calibrations,
        "tuned_params": tuned_params,
    }

    os.makedirs("data", exist_ok=True)
    with open(LEARNER_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n   [REPORT] Health: {model_health} | Trading: {'ALLOWED' if trading_allowed else 'PAUSED'}")
    if unhealthy_reasons:
        for reason in unhealthy_reasons:
            print(f"   [WARN] {reason}")

    return report


# ══════════════════════════════════════════════════
#  UTILITY: Fetch + Engineer (same as scholar.py)
# ══════════════════════════════════════════════════
def fetch_and_engineer(symbol: str, period: str = DATA_PERIOD) -> pd.DataFrame:
    """Download and compute features — mirrors scholar.py logic."""
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
    df["Sentiment_Score"] = 0.0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════
#  MAIN — ORCHESTRATOR
# ══════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PROJECT AEGIS - THE LEARNER (Off-Market Learning)")
    print(f"   Watchlist : {len(STOCK_WATCHLIST)} stocks")
    print(f"   Mode      : Continuous Learning & Adaptation")
    print(f"   Date      : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ── Step 1: Review past trades ──
    print(f"\n{'─' * 50}")
    print("  STEP 1: Trade Review")
    print(f"{'─' * 50}")
    trade_review = review_past_trades()

    # ── Step 2: Hyperparameter tuning for each stock ──
    print(f"\n{'─' * 50}")
    print("  STEP 2: Hyperparameter Tuning")
    print(f"{'─' * 50}")
    tuned_params = []
    for sym in STOCK_WATCHLIST[:TOP_N_STOCKS]:
        try:
            df = fetch_and_engineer(sym)
            params = tune_hyperparameters(sym, df)
            tuned_params.append(params)
        except Exception as e:
            print(f"   [FAIL] Tuning failed for {sym}: {e}")

    # Save best params
    if tuned_params:
        with open(LEARNER_BEST_PARAMS_FILE, "w") as f:
            json.dump(tuned_params, f, indent=2)

    # ── Step 3: Market regime detection ──
    print(f"\n{'─' * 50}")
    print("  STEP 3: Market Regime Detection")
    print(f"{'─' * 50}")
    regimes = {}
    # Use NIFTY 50 as overall market indicator
    try:
        regime = train_regime_detector("^NSEI")
        regimes["NIFTY50"] = regime
    except Exception as e:
        print(f"   [FAIL] Regime detection failed: {e}")
        regimes["NIFTY50"] = "UNKNOWN"

    # Per-stock regime
    for sym in STOCK_WATCHLIST[:TOP_N_STOCKS]:
        try:
            regime = train_regime_detector(sym)
            regimes[sym] = regime
        except Exception as e:
            regimes[sym] = "UNKNOWN"

    # ── Step 4: Confidence calibration ──
    print(f"\n{'─' * 50}")
    print("  STEP 4: Confidence Calibration")
    print(f"{'─' * 50}")
    calibrations = []
    for sym in STOCK_WATCHLIST[:TOP_N_STOCKS]:
        try:
            df = fetch_and_engineer(sym)
            cal = calibrate_confidence(sym, df)
            calibrations.append(cal)
        except Exception as e:
            print(f"   [FAIL] Calibration failed for {sym}: {e}")

    # ── Step 5: Ensemble weight optimization ──
    print(f"\n{'─' * 50}")
    print("  STEP 5: Ensemble Weight Optimization")
    print(f"{'─' * 50}")
    all_weights = {}
    for sym in STOCK_WATCHLIST[:TOP_N_STOCKS]:
        try:
            df = fetch_and_engineer(sym)
            w = optimize_ensemble_weights(sym, df)
            all_weights[sym] = w
        except Exception as e:
            print(f"   [FAIL] Weight optimization failed for {sym}: {e}")

    # Save weights
    if all_weights:
        with open(LEARNER_WEIGHTS_FILE, "w") as f:
            json.dump(all_weights, f, indent=2)

    # ── Step 6: Risk parameter adaptation ──
    print(f"\n{'─' * 50}")
    print("  STEP 6: Risk Parameter Adaptation")
    print(f"{'─' * 50}")
    risk_params = adapt_risk_parameters()

    # ── Step 7: Generate health report ──
    print(f"\n{'─' * 50}")
    print("  STEP 7: Health Report Generation")
    print(f"{'─' * 50}")
    report = generate_health_report(
        trade_review, calibrations, all_weights, risk_params, regimes, tuned_params
    )

    print(f"\n{'=' * 60}")
    print("  LEARNER COMPLETE — Brain is now smarter!")
    print(f"   Model Health  : {report['model_health']}")
    print(f"   Trading Status: {'ALLOWED' if report['trading_allowed'] else 'PAUSED'}")
    print(f"   Risk Level    : {risk_params['risk_level']}")
    print(f"   Market Regime : {regimes.get('NIFTY50', 'UNKNOWN')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
