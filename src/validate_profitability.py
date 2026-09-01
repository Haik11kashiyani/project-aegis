import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, os.path.dirname(__file__))
from config import STOCK_WATCHLIST, TOP_N_STOCKS, CAPITAL
from strategy_engine import StrategyEngine

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings("ignore")

VALIDATION_REPORT_FILE = "data/profitability_validation.json"
SLIPPAGE_BPS = 5  # 0.05%
TX_COSTS_BPS = 5  # 0.05% round trip

def apply_slippage(price, side="BUY"):
    slip = price * (SLIPPAGE_BPS / 10000)
    return round(price + slip, 2) if side == "BUY" else round(price - slip, 2)

def validate_profitability():
    print(f"\n{'=' * 60}", flush=True)
    print("  PROFITABILITY VALIDATION SIMULATOR (INTRADAY + STRATEGY ENGINE)", flush=True)
    print(f"{'=' * 60}", flush=True)

    symbols = STOCK_WATCHLIST[:5]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180) # 6 months

    print("Fetching 15m data for the last 60 days (yfinance limit for 15m)...", flush=True)
    all_data = {}
    
    for sym in symbols:
        try:
            # yfinance max for 15m is 60d
            df = yf.download(sym, period="60d", interval="15m", progress=False)
            if not df.empty:
                # flatten columns if multiindex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # compute basic indicators needed by strategies
                import indicators as ind
                df["RSI"] = ind.rsi(df["Close"])
                df["SMA_20"] = ind.sma(df["Close"], 20)
                df["SMA_50"] = ind.sma(df["Close"], 50)
                df["EMA_20"] = ind.ema(df["Close"], 20)
                df["ATR"] = ind.atr(df["High"], df["Low"], df["Close"])
                df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
                
                # MACD
                macd_df = ind.macd(df["Close"])
                df["MACD"] = macd_df.iloc[:, 0]
                df["MACD_Signal"] = macd_df.iloc[:, 2]
                
                # Bollinger Bands
                bb_df = ind.bbands(df["Close"])
                df["BB_Lower"] = bb_df.iloc[:, 0]
                df["BB_Upper"] = bb_df.iloc[:, 2]
                
                df.dropna(inplace=True)
                
                # add day grouping
                df['Date_Str'] = df.index.strftime('%Y-%m-%d')
                all_data[sym] = df
                print(f"  [OK] {sym}: {len(df)} 15m candles")
        except Exception as e:
            print(f"  [FAIL] {sym}: fetch failed ({e})")

    if not all_data:
        print("No data available.")
        return

    # Initialize strategy engine
    engine = StrategyEngine()

    # find unique days across all stocks
    all_days = set()
    for sym, df in all_data.items():
        all_days.update(df['Date_Str'].unique())
    all_days = sorted(list(all_days))

    trades = []
    cash = 15000.0
    equity_curve = []
    daily_returns = []
    prev_equity = cash

    print("\nRunning daily simulation...")
    for day in all_days:
        portfolio_val = cash
        equity_curve.append({"date": day, "equity": portfolio_val})
        
        if prev_equity > 0:
            daily_returns.append((portfolio_val - prev_equity) / prev_equity)
        prev_equity = portfolio_val

        for sym in symbols:
            if sym not in all_data: continue
            df = all_data[sym]
            day_data = df[df['Date_Str'] == day]
            if day_data.empty: continue
            
            # Simulate intraday tick by tick (candle by candle)
            position = None
            trades_today = 0
            
            for i in range(1, len(day_data)-1): # reserve last candle for force close
                # At candle i, we look at data up to i
                current_time = day_data.index[i]
                current_candle = day_data.iloc[i]
                
                if position is None:
                    if trades_today >= 1:
                        continue  # Max 1 high-quality trade per stock per day
                        
                    # evaluate entry
                    slice_df = df.loc[:current_time].copy()
                    if len(slice_df) < 50: continue # need enough lookback
                    
                    # Build current_data dict for strategy engine
                    latest = slice_df.iloc[-1]
                    prev_candles = slice_df.iloc[:-1]
                    high_20 = float(prev_candles['High'].tail(20).max()) if len(prev_candles) >= 5 else float(latest['High'])
                    
                    past_days = df[df['Date_Str'] < day]
                    prev_close = float(past_days.iloc[-1]['Close']) if not past_days.empty else float(latest['Close'])
                    
                    today_candles = slice_df[slice_df['Date_Str'] == day]
                    day_open = float(today_candles.iloc[0]['Open']) if not today_candles.empty else float(latest['Open'])
                    
                    # Trend filter: Avoid buying stocks in strong downtrend (Price < EMA20 < SMA50)
                    price = float(latest['Close'])
                    ema20 = float(latest.get('EMA_20', price))
                    sma50 = float(latest.get('SMA_50', price))
                    if price < ema20 < sma50 and (sma50 - price) / price > 0.015:
                        continue  # Strong downtrend gate
                        
                    # Reversal check: For MeanReversion, candle must be showing reversal (Close >= Open or green)
                    is_green_or_reversal = (price >= float(latest['Open'])) or (float(latest['High']) - price < price - float(latest['Low']))
                    
                    current_data = {
                        'price': price,
                        'rsi': float(latest.get('RSI', 50)),
                        'sma_50': sma50,
                        'sma_200': float(latest.get('SMA_50', 0)),
                        'ema_20': ema20,
                        'atr': float(latest.get('ATR', 0)),
                        'macd': float(latest.get('MACD', 0)) if 'MACD' in latest.index else 0,
                        'macd_signal': float(latest.get('MACD_Signal', 0)) if 'MACD_Signal' in latest.index else 0,
                        'bb_upper': float(latest.get('BB_Upper', 0)) if 'BB_Upper' in latest.index else 0,
                        'bb_lower': float(latest.get('BB_Lower', 0)) if 'BB_Lower' in latest.index else 0,
                        'bb_middle': float(latest.get('SMA_20', 0)),
                        'volume_ratio': float(latest.get('Volume_Ratio', 1.0)),
                        'obv': 0,
                        'high_20': high_20,
                        'adx': 25,
                        'prev_close': prev_close,
                        'day_open': day_open,
                        'open': float(latest['Open']),
                    }
                    signals = engine.evaluate_all(
                        symbol=sym,
                        current_data=current_data,
                        intraday_data=slice_df
                    )
                    
                    buy_votes = signals.get('strategies_agree', 0)
                    confidence = signals.get('confidence', 0.0)
                    
                    # Quality gate: Action must be BUY with >= 0.60 confidence and reversal confirmation
                    if buy_votes >= 1 and signals.get('action') == 'BUY' and confidence >= 0.60 and is_green_or_reversal:
                        # Enter at next candle open!
                        next_open = float(day_data.iloc[i+1]["Open"])
                        entry_price = apply_slippage(next_open, "BUY")
                        atr = float(current_candle.get("ATR", current_candle["Close"]*0.01))
                        if atr <= 0 or pd.isna(atr): atr = entry_price * 0.01
                        
                        stop_loss = entry_price - (1.2 * atr)
                        target = entry_price + (2.5 * atr)
                        
                        # Position sizing (use 20% of cash)
                        alloc = cash * 0.20
                        qty = max(1, int(alloc / entry_price))
                        if qty * entry_price > cash: continue
                        
                        tx_cost = (qty * entry_price) * (TX_COSTS_BPS / 10000 / 2) # half for entry
                        cash -= (qty * entry_price + tx_cost)
                        
                        buy_strats = [s for s in signals.get('strategy_details', []) if s.get('action') == 'BUY']
                        best_strategy = buy_strats[0]['strategy'] if buy_strats else 'MLEnsemble'
                        position = {
                            "entry_time": str(day_data.index[i+1]),
                            "entry_price": entry_price,
                            "qty": qty,
                            "stop_loss": stop_loss,
                            "target": target,
                            "initial_stop": stop_loss,
                            "atr": atr,
                            "strategy": best_strategy,
                            "highest_price": entry_price
                        }
                        trades_today += 1
                else:
                    # Dynamic Trailing Stop & Position Management
                    high = float(current_candle["High"])
                    low = float(current_candle["Low"])
                    close = float(current_candle["Close"])
                    
                    if high > position["highest_price"]:
                        position["highest_price"] = high
                    
                    # 1. Breakeven Lock: Once price gains 1.0x ATR, move stop loss to Entry + 0.1x ATR
                    if position["highest_price"] >= position["entry_price"] + (1.0 * position["atr"]):
                        be_stop = position["entry_price"] + (0.1 * position["atr"])
                        if be_stop > position["stop_loss"]:
                            position["stop_loss"] = be_stop
                            
                    # 2. Profit Lock: Once price gains 1.8x ATR, trail stop to Entry + 1.2x ATR
                    if position["highest_price"] >= position["entry_price"] + (1.8 * position["atr"]):
                        trail_stop = position["entry_price"] + (1.2 * position["atr"])
                        if trail_stop > position["stop_loss"]:
                            position["stop_loss"] = trail_stop
                    
                    exit_price = None
                    exit_reason = ""
                    
                    if low <= position["stop_loss"]:
                        exit_price = apply_slippage(position["stop_loss"], "SELL")
                        exit_reason = "STOP_LOSS" if position["stop_loss"] <= position["entry_price"] else "TRAILING_STOP"
                    elif high >= position["target"]:
                        exit_price = apply_slippage(position["target"], "SELL")
                        exit_reason = "TARGET_HIT"
                        
                    if exit_price:
                        tx_cost = (position["qty"] * exit_price) * (TX_COSTS_BPS / 10000 / 2)
                        pnl = (exit_price - position["entry_price"]) * position["qty"] - tx_cost
                        cash += position["qty"] * exit_price - tx_cost
                        trades.append({
                            "symbol": sym,
                            "strategy": position["strategy"],
                            "entry_time": position["entry_time"],
                            "exit_time": str(current_time),
                            "entry_price": position["entry_price"],
                            "exit_price": exit_price,
                            "qty": position["qty"],
                            "pnl": round(pnl, 2),
                            "exit_reason": exit_reason
                        })
                        position = None

            # EOD force close
            if position is not None:
                last_candle = day_data.iloc[-1]
                exit_price = apply_slippage(float(last_candle["Close"]), "SELL")
                tx_cost = (position["qty"] * exit_price) * (TX_COSTS_BPS / 10000 / 2)
                pnl = (exit_price - position["entry_price"]) * position["qty"] - tx_cost
                cash += position["qty"] * exit_price - tx_cost
                trades.append({
                    "symbol": sym,
                    "strategy": position["strategy"],
                    "entry_time": position["entry_time"],
                    "exit_time": str(day_data.index[-1]),
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "qty": position["qty"],
                    "pnl": round(pnl, 2),
                    "exit_reason": "FORCE_CLOSE"
                })
                position = None

    # Calculate metrics
    total_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 999

    peak = 15000.0
    max_dd = 0
    for ec in equity_curve:
        if ec["equity"] > peak: peak = ec["equity"]
        dd = (peak - ec["equity"]) / peak
        if dd > max_dd: max_dd = dd

    dr = np.array(daily_returns)
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if len(dr) > 0 and dr.std() > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  VALIDATION RESULTS (Rs. 15,000 Starting Capital)")
    print(f"{'=' * 60}")
    print(f"  Ending Capital: Rs. {cash:,.2f}")
    print(f"  Total P&L     : Rs. {total_pnl:,.2f}")
    print(f"  Total Trades  : {total_trades}")
    print(f"  Win Rate      : {win_rate:.1f}%")
    print(f"  Profit Factor : {profit_factor:.2f}")
    print(f"  Max Drawdown  : {max_dd*100:.2f}%")
    print(f"  Sharpe Ratio  : {sharpe:.2f}")

    print("\n  Per-Strategy Breakdown:")
    strat_stats = {}
    for t in trades:
        s = t["strategy"]
        if s not in strat_stats: strat_stats[s] = {"trades": 0, "pnl": 0}
        strat_stats[s]["trades"] += 1
        strat_stats[s]["pnl"] += t["pnl"]
    for s, st in strat_stats.items():
        print(f"    {s}: {st['trades']} trades | P&L ₹{st['pnl']:.2f}")

    report = {
        "starting_capital": 15000.0,
        "ending_capital": cash,
        "total_pnl": total_pnl,
        "metrics": {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe
        },
        "strategy_breakdown": strat_stats,
        "trades": trades
    }

    os.makedirs("data", exist_ok=True)
    with open(VALIDATION_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {VALIDATION_REPORT_FILE}")

if __name__ == "__main__":
    validate_profitability()
