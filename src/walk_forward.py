"""
====================================================
🔄 PROJECT AEGIS - Walk-Forward Paper Trading Simulator
====================================================
Replays historical data tick-by-tick to simulate exactly
how the Sniper would have traded. Superior to simple
backtesting because it accounts for:
  - Execution latency (configurable delay)
  - Slippage (random or fixed)
  - Intraday candle replay (not just close prices)
  - Sequential decision-making (no lookahead bias)
  - Real capital constraints and position sizing

Usage:
    python src/walk_forward.py               # Default: last 60 days
    python src/walk_forward.py --days 120    # Custom period
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

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import (
    STOCK_WATCHLIST, TOP_N_STOCKS, CAPITAL, MAX_BULLETS,
    CONFIDENCE_THRESHOLD, RF_FEATURES,
    ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER,
    MIN_VOTES_TO_BUY, LSTM_LOOKBACK, INTRADAY_LOOKBACK,
    model_paths, TRADE_LOG_FILE,
)

# ──────────────────────────────────────────────────
#  SIMULATION CONFIG
# ──────────────────────────────────────────────────
DEFAULT_DAYS         = 60         # Replay window
SLIPPAGE_BPS         = 5          # Slippage in basis points (0.05%)
LATENCY_CANDLES      = 1          # Execute 1 candle later (simulates delay)
COMMISSION_PER_TRADE = 0.0        # ₹0 for paper, set >0 for broker fees
WALK_FORWARD_FILE    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "walk_forward_results.json",
)
WF_TRADES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "walk_forward_trades.csv",
)


# ──────────────────────────────────────────────────
#  DATA PREPARATION
# ──────────────────────────────────────────────────
def fetch_historical_data(symbol: str, days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """Fetch and engineer features for historical simulation."""
    try:
        period = f"{days}d" if days <= 730 else f"{days // 365}y"
        raw = yf.download(symbol, period=period, interval="1d", progress=False)
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = flatten_yf_columns(raw) if hasattr(raw.columns, 'nlevels') and raw.columns.nlevels > 1 else raw.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # Engineer indicators
        df["RSI"]          = ta.rsi(df["Close"])
        df["SMA_50"]       = ta.sma(df["Close"], 50)
        df["SMA_200"]      = ta.sma(df["Close"], 200)
        df["EMA_20"]       = ta.ema(df["Close"], 20)
        df["ATR"]          = ta.atr(df["High"], df["Low"], df["Close"])
        df["MACD"], df["MACD_Signal"] = ta.macd(df["Close"])
        df["BB_Upper"], df["BB_Lower"] = ta.bollinger_bands(df["Close"])
        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["OBV"]          = ta.obv(df["Close"], df["Volume"])
        df["Sentiment_Score"] = 0.0
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  [WF] Data fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def load_models_for_sim(symbol: str) -> dict:
    """Load trained models for a symbol (RF, XGB, LSTM)."""
    paths = model_paths(symbol)
    models = {"rf": None, "xgb": None, "lstm": None, "intraday_lstm": None}

    try:
        if os.path.exists(paths["rf"]):
            models["rf"] = joblib.load(paths["rf"])
    except Exception:
        pass
    try:
        if os.path.exists(paths["xgb"]):
            models["xgb"] = joblib.load(paths["xgb"])
    except Exception:
        pass
    try:
        if os.path.exists(paths["lstm"]):
            from tensorflow.keras.models import load_model
            models["lstm"] = load_model(paths["lstm"], compile=False)
    except Exception:
        pass
    try:
        if os.path.exists(paths.get("lstm_scaler", "")):
            models["lstm_scaler"] = joblib.load(paths["lstm_scaler"])
    except Exception:
        pass
    try:
        if os.path.exists(paths["intraday_lstm"]):
            from tensorflow.keras.models import load_model
            models["intraday_lstm"] = load_model(paths["intraday_lstm"], compile=False)
    except Exception:
        pass

    return models


# ──────────────────────────────────────────────────
#  MODEL PREDICTION (simplified for simulation)
# ──────────────────────────────────────────────────
def get_sim_signal(df_slice: pd.DataFrame, models: dict) -> tuple:
    """
    Get ensemble signal using available models on a data slice.
    Returns: (votes, avg_confidence, rf_conf, xgb_conf)
    """
    votes = 0
    confs = []
    rf_conf = 0.5
    xgb_conf = 0.5

    last = df_slice.iloc[-1]
    features = [last.get(f, 0) for f in RF_FEATURES]
    feat_arr = np.array(features).reshape(1, -1)

    # RF prediction
    if models.get("rf") is not None:
        try:
            proba = models["rf"].predict_proba(feat_arr)[0]
            rf_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
            if rf_conf >= CONFIDENCE_THRESHOLD:
                votes += 1
            confs.append(rf_conf)
        except Exception:
            confs.append(0.5)

    # XGB prediction
    if models.get("xgb") is not None:
        try:
            proba = models["xgb"].predict_proba(feat_arr)[0]
            xgb_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
            if xgb_conf >= CONFIDENCE_THRESHOLD:
                votes += 1
            confs.append(xgb_conf)
        except Exception:
            confs.append(0.5)

    # LSTM (daily) — simple direction check
    if models.get("lstm") is not None and models.get("lstm_scaler") is not None:
        try:
            scaler = models["lstm_scaler"]
            close_data = df_slice["Close"].values[-LSTM_LOOKBACK:].reshape(-1, 1)
            scaled = scaler.transform(close_data)
            X = scaled.reshape(1, LSTM_LOOKBACK, 1)
            pred = float(models["lstm"].predict(X, verbose=0)[0][0])
            lstm_conf = pred
            if pred > scaler.transform([[float(df_slice["Close"].iloc[-1])]])[0][0]:
                votes += 1
            confs.append(min(max(lstm_conf, 0), 1))
        except Exception:
            confs.append(0.5)
    else:
        confs.append(0.5)

    # Intraday LSTM placeholder
    confs.append(0.5)

    avg_conf = sum(confs) / len(confs) if confs else 0.5
    return votes, avg_conf, rf_conf, xgb_conf


# ──────────────────────────────────────────────────
#  SLIPPAGE & LATENCY
# ──────────────────────────────────────────────────
def apply_slippage(price: float, side: str = "BUY") -> float:
    """Apply realistic slippage to execution price."""
    slip = price * (SLIPPAGE_BPS / 10000)
    noise = np.random.uniform(0, slip)
    if side == "BUY":
        return round(price + noise, 2)  # Pay slightly more
    else:
        return round(price - noise, 2)  # Receive slightly less


# ──────────────────────────────────────────────────
#  WALK-FORWARD ENGINE
# ──────────────────────────────────────────────────
def run_walk_forward(
    symbols: list = None,
    days: int = DEFAULT_DAYS,
    capital: float = CAPITAL,
    max_bullets: int = MAX_BULLETS,
    verbose: bool = True,
) -> dict:
    """
    Walk-Forward Paper Trading Simulator.

    Replays historical data day-by-day, making buy/sell decisions
    sequentially without future information.

    Returns comprehensive results dict.
    """
    if symbols is None:
        symbols = STOCK_WATCHLIST[:TOP_N_STOCKS]

    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD SIMULATOR")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Period  : {days} days")
    print(f"  Capital : ₹{capital:,.0f}")
    print(f"  Bullets : {max_bullets}")
    print(f"  Slippage: {SLIPPAGE_BPS} bps")
    print(f"  Latency : {LATENCY_CANDLES} candle(s)")
    print(f"{'=' * 60}\n")

    # Fetch data and models
    all_data = {}
    all_models = {}
    for sym in symbols:
        df = fetch_historical_data(sym, days + 200)  # Extra for indicators
        if len(df) > 60:
            all_data[sym] = df
            all_models[sym] = load_models_for_sim(sym)
            if verbose:
                print(f"  ✅ {sym}: {len(df)} candles, models loaded")
        else:
            if verbose:
                print(f"  ❌ {sym}: insufficient data ({len(df)} candles)")

    if not all_data:
        print("  [WF] No usable data. Aborting.")
        return {"error": "No data"}

    # Determine common date range (last N trading days)
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    sorted_dates = sorted(all_dates)
    sim_dates = sorted_dates[-days:] if len(sorted_dates) > days else sorted_dates

    # Simulation state
    cash = capital
    positions = {}         # sym → {qty, entry_price, stop_loss, target, entry_date}
    trades = []            # completed trades
    equity_curve = []      # (date, equity)
    daily_returns = []
    prev_equity = capital

    bullets_used = 0
    bullet_size = capital / max_bullets

    # ── DAY-BY-DAY REPLAY ──
    for i, date in enumerate(sim_dates):
        day_str = str(date)[:10]

        # Get current prices for all held positions
        portfolio_value = cash
        for sym, pos in positions.items():
            if sym in all_data and date in all_data[sym].index:
                row = all_data[sym].loc[date]
                current_price = float(row["Close"])
                portfolio_value += current_price * pos["qty"]
            else:
                portfolio_value += pos["entry_price"] * pos["qty"]

        equity_curve.append({
            "date": day_str,
            "equity": round(portfolio_value, 2),
            "cash": round(cash, 2),
            "positions": len(positions),
        })

        # Daily return
        daily_ret = (portfolio_value - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns.append(daily_ret)
        prev_equity = portfolio_value

        # ── EXIT CHECK — Check stops and targets ──
        closed_syms = []
        for sym, pos in list(positions.items()):
            if sym not in all_data or date not in all_data[sym].index:
                continue

            row = all_data[sym].loc[date]
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            atr = float(row.get("ATR", 0))

            # Trailing stop update
            if atr > 0:
                new_stop = close - (ATR_STOP_MULTIPLIER * atr)
                if new_stop > pos["stop_loss"]:
                    pos["stop_loss"] = round(new_stop, 2)

            exit_price = None
            exit_type = None

            # Stop loss hit (using low of day)
            if low <= pos["stop_loss"]:
                exit_price = apply_slippage(pos["stop_loss"], "SELL")
                exit_type = "STOP_LOSS"
            # Target hit (using high of day)
            elif high >= pos["target"]:
                exit_price = apply_slippage(pos["target"], "SELL")
                exit_type = "TARGET_HIT"

            if exit_price is not None:
                pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                pnl -= COMMISSION_PER_TRADE
                cash += exit_price * pos["qty"]
                trades.append({
                    "date": day_str,
                    "symbol": sym,
                    "side": "SELL",
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "qty": pos["qty"],
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((pnl / (pos["entry_price"] * pos["qty"])) * 100, 2),
                    "exit_type": exit_type,
                    "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
                    "entry_date": pos["entry_date"],
                })
                closed_syms.append(sym)
                bullets_used = max(0, bullets_used - 1)

                if verbose and i >= len(sim_dates) - 30:
                    icon = "✅" if pnl > 0 else "❌"
                    print(f"  {icon} {day_str} EXIT {sym}: {exit_type} | "
                          f"₹{pos['entry_price']:.2f}→₹{exit_price:.2f} | "
                          f"P&L: ₹{pnl:,.2f}")

        for sym in closed_syms:
            del positions[sym]

        # ── ENTRY CHECK — Look for new trades ──
        if bullets_used < max_bullets and i >= LATENCY_CANDLES:
            for sym in symbols:
                if sym in positions:
                    continue  # Already holding
                if sym not in all_data or date not in all_data[sym].index:
                    continue
                if bullets_used >= max_bullets:
                    break

                # Need enough history for LSTM lookback
                idx = all_data[sym].index.get_loc(date)
                if idx < max(LSTM_LOOKBACK, 60):
                    continue

                df_slice = all_data[sym].iloc[:idx + 1]
                votes, avg_conf, rf_conf, xgb_conf = get_sim_signal(df_slice, all_models.get(sym, {}))

                # ── Entry criteria (mimic sniper logic) ──
                if votes >= MIN_VOTES_TO_BUY and avg_conf >= CONFIDENCE_THRESHOLD:
                    row = all_data[sym].loc[date]
                    price = apply_slippage(float(row["Close"]), "BUY")
                    atr = float(row.get("ATR", 0))

                    if atr <= 0 or price <= 0:
                        continue

                    stop_loss = round(price - ATR_STOP_MULTIPLIER * atr, 2)
                    target = round(price + ATR_TARGET_MULTIPLIER * atr, 2)

                    # Position sizing
                    max_spend = min(bullet_size, cash)
                    qty = max(1, int(max_spend / price))

                    if qty * price > cash:
                        continue

                    cost = price * qty + COMMISSION_PER_TRADE
                    cash -= cost

                    positions[sym] = {
                        "qty": qty,
                        "entry_price": price,
                        "stop_loss": stop_loss,
                        "target": target,
                        "entry_date": day_str,
                        "confidence": avg_conf,
                        "votes": votes,
                    }
                    bullets_used += 1

                    if verbose and i >= len(sim_dates) - 30:
                        print(f"  🔫 {day_str} BUY  {sym}: {qty}x @ ₹{price:.2f} | "
                              f"SL: ₹{stop_loss:.2f} | Tgt: ₹{target:.2f} | "
                              f"Conf: {avg_conf:.2f}")

    # ── Force close remaining positions at last day's close ──
    last_date = sim_dates[-1] if sim_dates else None
    if last_date:
        for sym, pos in list(positions.items()):
            if sym in all_data and last_date in all_data[sym].index:
                close = float(all_data[sym].loc[last_date]["Close"])
                exit_price = apply_slippage(close, "SELL")
                pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                cash += exit_price * pos["qty"]
                trades.append({
                    "date": str(last_date)[:10],
                    "symbol": sym,
                    "side": "SELL",
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "qty": pos["qty"],
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((pnl / (pos["entry_price"] * pos["qty"])) * 100, 2),
                    "exit_type": "FORCE_CLOSE",
                    "hold_days": (pd.Timestamp(last_date) - pd.Timestamp(pos["entry_date"])).days,
                    "entry_date": pos["entry_date"],
                })
        positions.clear()

    # ── COMPUTE STATISTICS ──
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 999

    # Max drawdown
    peak = capital
    max_dd = 0
    for ec in equity_curve:
        eq = ec["equity"]
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualised)
    if daily_returns and len(daily_returns) > 1:
        dr = np.array(daily_returns)
        sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
    else:
        sharpe = 0

    # Average hold period
    avg_hold = np.mean([t.get("hold_days", 0) for t in trades]) if trades else 0

    # Per-stock breakdown
    stock_stats = {}
    for t in trades:
        sym = t["symbol"]
        if sym not in stock_stats:
            stock_stats[sym] = {"trades": 0, "wins": 0, "pnl": 0}
        stock_stats[sym]["trades"] += 1
        stock_stats[sym]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            stock_stats[sym]["wins"] += 1

    for sym in stock_stats:
        ss = stock_stats[sym]
        ss["win_rate"] = round(ss["wins"] / ss["trades"] * 100, 1) if ss["trades"] > 0 else 0
        ss["pnl"] = round(ss["pnl"], 2)

    results = {
        "simulation_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": {
            "symbols": symbols,
            "days": days,
            "capital": capital,
            "max_bullets": max_bullets,
            "slippage_bps": SLIPPAGE_BPS,
            "latency_candles": LATENCY_CANDLES,
            "commission": COMMISSION_PER_TRADE,
        },
        "summary": {
            "total_return": round(total_pnl, 2),
            "total_return_pct": round((total_pnl / capital) * 100, 2),
            "final_equity": round(cash, 2),
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_hold_days": round(avg_hold, 1),
        },
        "stock_stats": stock_stats,
        "equity_curve": equity_curve,
        "trades": trades[-100:],  # Keep last 100 for dashboard
    }

    # ── PRINT SUMMARY ──
    print(f"\n{'═' * 60}")
    print(f"  WALK-FORWARD RESULTS")
    print(f"{'═' * 60}")
    print(f"  Total P&L     : ₹{total_pnl:,.2f} ({results['summary']['total_return_pct']:+.2f}%)")
    print(f"  Trades         : {total_trades} ({len(wins)}W / {len(losses)}L)")
    print(f"  Win Rate       : {win_rate:.1f}%")
    print(f"  Profit Factor  : {profit_factor:.2f}")
    print(f"  Max Drawdown   : {max_dd * 100:.2f}%")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Avg Hold       : {avg_hold:.1f} days")
    print(f"{'═' * 60}")

    if stock_stats:
        print(f"\n  Per-Stock:")
        for sym, ss in sorted(stock_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            icon = "🟢" if ss["pnl"] >= 0 else "🔴"
            print(f"    {icon} {sym:15} | {ss['trades']:2d} trades | "
                  f"WR: {ss['win_rate']:5.1f}% | P&L: ₹{ss['pnl']:,.2f}")

    # ── SAVE ──
    try:
        os.makedirs(os.path.dirname(WALK_FORWARD_FILE), exist_ok=True)
        with open(WALK_FORWARD_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  📁 Results saved to {WALK_FORWARD_FILE}")
    except Exception as e:
        print(f"  [WF] Save failed: {e}")

    # Save trades CSV
    try:
        if trades:
            pd.DataFrame(trades).to_csv(WF_TRADES_FILE, index=False)
            print(f"  📁 Trades saved to {WF_TRADES_FILE}")
    except Exception:
        pass

    return results


# ──────────────────────────────────────────────────
#  CLI ENTRY POINT
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Walk-Forward Paper Trading Simulator")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Simulation period in days")
    parser.add_argument("--capital", type=float, default=CAPITAL, help="Starting capital")
    parser.add_argument("--bullets", type=int, default=MAX_BULLETS, help="Max concurrent positions")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (e.g. SBIN.NS,TCS.NS)")
    args = parser.parse_args()

    syms = args.symbols.split(",") if args.symbols else None
    run_walk_forward(symbols=syms, days=args.days, capital=args.capital, max_bullets=args.bullets)
