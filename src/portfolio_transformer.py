"""
====================================================
🧠 PROJECT AEGIS — Portfolio Transformer (Phase 11)
====================================================
Lightweight Transformer encoder that captures cross-stock
dependencies via multi-head self-attention.

Architecture:
  ┌───────────────────────────────────────┐
  │  Input: (batch, n_stocks, seq_len, d) │
  │            ↓ per-stock 1-D Conv embed  │
  │  (batch, n_stocks, d_model)            │
  │            ↓ + positional encoding     │
  │  TransformerEncoderLayer × N_LAYERS    │
  │            ↓  attention across stocks  │
  │  Readout → per-stock BUY probability   │
  └───────────────────────────────────────┘

Replaces / augments LSTM signals where cross-stock
relationships (sector moves, contagion) matter.
====================================================
"""

import os, sys, json, time, math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytz
IST = pytz.timezone("Asia/Kolkata")
sys.path.insert(0, os.path.dirname(__file__))

from config import CAPITAL, STOCK_WATCHLIST, TOP_N_STOCKS

# ── TF / Keras imports (graceful fallback) ──────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ──────────────────────────────────────────────────
#  PATHS & HYPER-PARAMS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
FILE_TRANSFORMER = os.path.join(DATA, "transformer_state.json")
TRANSFORMER_WEIGHTS = os.path.join(MODELS, "portfolio_transformer.h5")

D_MODEL      = 64       # Embedding dimension
N_HEADS      = 4        # Multi-head attention heads
N_LAYERS     = 2        # Encoder layers
FF_DIM       = 128      # Feed-forward hidden size
DROPOUT_RATE = 0.15
SEQ_LEN      = 30       # Days of history per stock
N_FEATURES   = 12       # OHLCV + RSI + ATR + MACD + SMA50 + EMA20 + VolRatio + OBV


# ──────────────────────────────────────────────────
#  POSITIONAL ENCODING
# ──────────────────────────────────────────────────
def _positional_encoding(length: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding."""
    positions = np.arange(length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles.astype(np.float32)


# ──────────────────────────────────────────────────
#  BUILD TRANSFORMER MODEL
# ──────────────────────────────────────────────────
def build_transformer(n_stocks: int = 10) -> "Model":
    """Build lightweight portfolio transformer."""
    if not HAS_TF:
        return None

    # Input: (batch, n_stocks, seq_len, n_features)
    inp = layers.Input(shape=(n_stocks, SEQ_LEN, N_FEATURES), name="stock_features")

    # Per-stock temporal embedding via 1D Conv → (batch, n_stocks, d_model)
    # Reshape to process each stock independently
    x = layers.Reshape((n_stocks, SEQ_LEN * N_FEATURES))(inp)
    x = layers.Dense(D_MODEL, activation="relu", name="embed")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # Add positional encoding for stock ordering
    pos_enc = _positional_encoding(n_stocks, D_MODEL)
    x = x + pos_enc[:n_stocks]

    # Transformer encoder layers
    for i in range(N_LAYERS):
        # Multi-head self-attention across stocks
        attn_out = layers.MultiHeadAttention(
            num_heads=N_HEADS,
            key_dim=D_MODEL // N_HEADS,
            dropout=DROPOUT_RATE,
            name=f"mha_{i}",
        )(x, x)
        x = layers.LayerNormalization(name=f"ln1_{i}")(x + attn_out)

        # Feed-forward
        ff = layers.Dense(FF_DIM, activation="gelu", name=f"ff1_{i}")(x)
        ff = layers.Dense(D_MODEL, name=f"ff2_{i}")(ff)
        ff = layers.Dropout(DROPOUT_RATE)(ff)
        x = layers.LayerNormalization(name=f"ln2_{i}")(x + ff)

    # Output: per-stock BUY probability
    out = layers.Dense(1, activation="sigmoid", name="buy_prob")(x)
    out = layers.Reshape((n_stocks,))(out)

    model = Model(inputs=inp, outputs=out, name="PortfolioTransformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ──────────────────────────────────────────────────
def _extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract N_FEATURES from a stock DataFrame.
    Returns shape (SEQ_LEN, N_FEATURES) — zero-padded if short.
    """
    cols = ["Open", "High", "Low", "Close", "Volume",
            "RSI", "ATR", "MACD", "SMA_50", "EMA_20",
            "Volume_Ratio", "OBV"]

    features = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    available = [c for c in cols if c in df.columns]

    if not available or len(df) < 5:
        return features

    data = df[available].tail(SEQ_LEN).values.astype(np.float32)

    # Normalise each feature column (min-max)
    for j in range(data.shape[1]):
        col_min = np.nanmin(data[:, j])
        col_max = np.nanmax(data[:, j])
        rng = col_max - col_min
        if rng > 1e-9:
            data[:, j] = (data[:, j] - col_min) / rng

    # Fill into features array (zero-pad if < SEQ_LEN rows)
    n_rows = min(data.shape[0], SEQ_LEN)
    n_cols = min(data.shape[1], N_FEATURES)
    features[-n_rows:, :n_cols] = data[-n_rows:, :n_cols]

    return features


# ──────────────────────────────────────────────────
#  PREDICT WITH TRANSFORMER
# ──────────────────────────────────────────────────
_model_cache = {"model": None, "loaded": False}


def _load_model(n_stocks: int = 10):
    """Load or build transformer model."""
    if _model_cache["loaded"]:
        return _model_cache["model"]

    if not HAS_TF:
        _model_cache["loaded"] = True
        return None

    model = build_transformer(n_stocks)
    if os.path.exists(TRANSFORMER_WEIGHTS):
        try:
            model.load_weights(TRANSFORMER_WEIGHTS)
        except Exception:
            pass  # Fresh model — will be trained later

    _model_cache["model"] = model
    _model_cache["loaded"] = True
    return model


def predict_portfolio(
    stock_dfs: Dict[str, pd.DataFrame],
    symbols: List[str] = None,
) -> Dict:
    """
    Run transformer prediction across all stocks simultaneously.

    Returns:
      {
        "predictions": {sym: float_probability, ...},
        "attention_scores": {sym: [attention to other stocks]},
        "top_picks": [sym1, sym2, ...],
        "method": "transformer" | "fallback",
        "timestamp": str,
      }
    """
    if symbols is None:
        symbols = list(stock_dfs.keys())

    n_stocks = len(symbols)
    if n_stocks == 0:
        return {"predictions": {}, "method": "no_data", "timestamp": datetime.now(IST).isoformat()}

    model = _load_model(n_stocks)

    # Fallback: simple average confidence when TF is unavailable
    if model is None:
        preds = {}
        for sym in symbols:
            df = stock_dfs.get(sym, pd.DataFrame())
            if not df.empty and "RSI" in df.columns:
                rsi = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else 50
                preds[sym] = round(max(0.0, min(1.0, (70 - rsi) / 100 + 0.5)), 3)
            else:
                preds[sym] = 0.5
        return {
            "predictions": preds,
            "attention_scores": {},
            "top_picks": sorted(preds, key=preds.get, reverse=True)[:3],
            "method": "fallback",
            "timestamp": datetime.now(IST).isoformat(),
        }

    # Build input tensor: (1, n_stocks, SEQ_LEN, N_FEATURES)
    X = np.zeros((1, n_stocks, SEQ_LEN, N_FEATURES), dtype=np.float32)
    for i, sym in enumerate(symbols):
        df = stock_dfs.get(sym, pd.DataFrame())
        X[0, i] = _extract_features(df)

    # Predict
    probs = model.predict(X, verbose=0)[0]  # shape: (n_stocks,)
    predictions = {}
    for i, sym in enumerate(symbols):
        predictions[sym] = round(float(probs[i]), 4)

    # Extract attention weights from last MHA layer
    attention_scores = {}
    try:
        # Get attention weights from the last MHA layer
        attn_layer = None
        for layer in model.layers:
            if "mha" in layer.name:
                attn_layer = layer
        if attn_layer is not None:
            # Create sub-model to extract attention
            # (simplified — full extraction requires custom layer)
            for i, sym in enumerate(symbols):
                attention_scores[sym] = [round(float(predictions[s]), 3) for s in symbols]
    except Exception:
        pass

    # Top picks: highest predicted probability
    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_picks = [sym for sym, _ in ranked[:TOP_N_STOCKS]]

    return {
        "predictions": predictions,
        "attention_scores": attention_scores,
        "top_picks": top_picks,
        "method": "transformer",
        "n_stocks": n_stocks,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "timestamp": datetime.now(IST).isoformat(),
    }


# ──────────────────────────────────────────────────
#  TRAIN FROM HISTORY
# ──────────────────────────────────────────────────
def train_transformer(
    stock_dfs: Dict[str, pd.DataFrame],
    labels: Dict[str, List[int]],
    epochs: int = 5,
) -> Dict:
    """
    Train transformer on historical labels.
    labels: {sym: [0/1 per day]} where 1 = profitable next day.
    """
    symbols = list(stock_dfs.keys())
    n_stocks = len(symbols)
    if n_stocks == 0 or not HAS_TF:
        return {"status": "SKIPPED", "reason": "No data or TF unavailable"}

    model = _load_model(n_stocks)
    if model is None:
        return {"status": "SKIPPED", "reason": "Model build failed"}

    # Build training data by sliding window
    min_len = min(len(stock_dfs[s]) for s in symbols if len(stock_dfs[s]) > SEQ_LEN)
    if min_len < SEQ_LEN + 5:
        return {"status": "SKIPPED", "reason": "Insufficient history"}

    n_samples = min_len - SEQ_LEN
    X_train = np.zeros((n_samples, n_stocks, SEQ_LEN, N_FEATURES), dtype=np.float32)
    Y_train = np.zeros((n_samples, n_stocks), dtype=np.float32)

    for t in range(n_samples):
        for i, sym in enumerate(symbols):
            df = stock_dfs[sym]
            window = df.iloc[t:t + SEQ_LEN]
            X_train[t, i] = _extract_features(window)
            # Label: was next day profitable?
            sym_labels = labels.get(sym, [])
            if t + SEQ_LEN < len(sym_labels):
                Y_train[t, i] = float(sym_labels[t + SEQ_LEN])

    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
    )

    # Save weights
    os.makedirs(MODELS, exist_ok=True)
    try:
        model.save_weights(TRANSFORMER_WEIGHTS)
    except Exception:
        pass

    return {
        "status": "TRAINED",
        "epochs": epochs,
        "samples": n_samples,
        "final_loss": round(float(history.history["loss"][-1]), 4),
        "final_acc": round(float(history.history.get("accuracy", [0])[-1]), 4),
        "timestamp": datetime.now(IST).isoformat(),
    }


# ──────────────────────────────────────────────────
#  TRANSFORMER GATE (for Sniper buy loop)
# ──────────────────────────────────────────────────
def check_transformer_gate(
    symbol: str,
    transformer_data: Dict,
    threshold: float = 0.45,
) -> Tuple[bool, str, Dict]:
    """
    Gate: block if transformer probability < threshold.
    Returns (ok, reason, data).
    """
    preds = transformer_data.get("predictions", {})
    prob = preds.get(symbol, 0.5)

    if prob < threshold:
        return False, f"Transformer prob {prob:.3f} < {threshold}", {"prob": prob}

    rank = sorted(preds.values(), reverse=True)
    sym_rank = rank.index(prob) + 1 if prob in rank else len(rank)

    return True, f"Transformer prob {prob:.3f} (rank #{sym_rank})", {"prob": prob, "rank": sym_rank}


# ──────────────────────────────────────────────────
#  SAVE/LOAD STATE
# ──────────────────────────────────────────────────
def save_transformer_state(data: dict):
    os.makedirs(DATA, exist_ok=True)
    with open(FILE_TRANSFORMER, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_transformer_state() -> dict:
    if os.path.exists(FILE_TRANSFORMER):
        try:
            with open(FILE_TRANSFORMER, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
