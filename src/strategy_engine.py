"""
====================================================
🛡️ PROJECT AEGIS - Multi-Strategy Engine
====================================================
Orchestrates 6 independent trading strategies.
====================================================
"""

import os
import json
import logging
from datetime import datetime, time
import numpy as np
import pandas as pd
import pytz

import config
import indicators as ta

IST = pytz.timezone("Asia/Kolkata")
WEIGHTS_FILE = "data/strategy_weights.json"

logger = logging.getLogger("StrategyEngine")

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Intraday VWAP from 15-min candles (vectorized, 100x faster)."""
    if df.empty or 'Volume' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
        
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    pv = typical_price * df['Volume']
    
    if hasattr(df.index, 'date'):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df.index).date
        
    df_temp = pd.DataFrame({'pv': pv, 'vol': df['Volume'], 'date': dates}, index=df.index)
    cum_pv = df_temp.groupby('date')['pv'].cumsum()
    cum_vol = df_temp.groupby('date')['vol'].cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    return vwap.fillna(typical_price)

class BaseStrategy:
    def __init__(self, name: str, default_weight: float = 1.0):
        self.name = name
        self.weight = default_weight
        self.enabled = True
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0

    def evaluate(self, symbol: str, data: dict, **kwargs) -> dict:
        """Returns {'action': 'BUY'|'SELL'|'HOLD', 'confidence': float, 'reason': str}"""
        return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Not implemented'}

    def get_exit(self, symbol: str, entry_price: float, data: dict, **kwargs) -> dict:
        """Returns {'should_exit': bool, 'reason': str}"""
        return {'should_exit': False, 'reason': 'Not implemented'}

    def update_performance(self, win: bool, pnl: float):
        """Track strategy performance"""
        self.trades += 1
        if win:
            self.wins += 1
        self.total_pnl += pnl

class MLEnsembleStrategy(BaseStrategy):
    def __init__(self, name="MLEnsembleStrategy", default_weight=1.0):
        super().__init__(name, default_weight)

    def evaluate(self, symbol, data, **kwargs):
        preds = kwargs.get('model_predictions', {})
        votes = 0
        total_conf = 0
        models_count = 0
        
        for m, conf in preds.items():
            models_count += 1
            if conf >= config.CONFIDENCE_THRESHOLD:
                votes += 1
                total_conf += conf
                
        if models_count == 0:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No model predictions'}
            
        avg_conf = total_conf / votes if votes > 0 else 0
        if votes >= config.MIN_VOTES_TO_BUY:
            return {'action': 'BUY', 'confidence': avg_conf, 'reason': f'ML Consensus ({votes}/{models_count})'}
        return {'action': 'HOLD', 'confidence': avg_conf, 'reason': 'Insufficient ML votes'}

    def get_exit(self, symbol, entry_price, data, **kwargs):
        return {'should_exit': False, 'reason': 'ML Exit logic delegated to TradingBrain'}

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, name="MeanReversionStrategy", default_weight=1.0):
        super().__init__(name, default_weight)

    def evaluate(self, symbol, data, **kwargs):
        try:
            price = data.get('price', 0)
            bbl = data.get('bb_lower', 0)
            rsi = data.get('rsi', 50)
            vol_ratio = data.get('volume_ratio', 1.0)
            open_p = data.get('open', price)
            
            if price == 0 or bbl == 0: return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Missing data'}
            
            # Must be near lower band, oversold, and bouncing green (Close >= Open)
            if price <= bbl * 1.006 and rsi <= 38 and price >= open_p:
                dist_bb = max(0, (bbl - price) / (bbl + 1e-6))
                conf = min(0.90, 0.60 + dist_bb * 10 + (38 - rsi) * 0.01)
                return {'action': 'BUY', 'confidence': conf, 'reason': f'Mean Reversion bounce at BBL ({price:.2f}<={bbl:.2f}), RSI={rsi:.1f}'}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
        return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Conditions not met'}

    def get_exit(self, symbol, entry_price, data, **kwargs):
        price = data.get('price', 0)
        bbm = data.get('bb_middle', 0)
        rsi = data.get('rsi', 50)
        
        if price >= bbm and bbm > 0:
            return {'should_exit': True, 'reason': 'Price touched SMA20 (BB Middle)'}
        if rsi > 60:
            return {'should_exit': True, 'reason': 'RSI > 60 recovery'}
        return {'should_exit': False, 'reason': 'Holding'}

class MomentumBreakoutStrategy(BaseStrategy):
    """
    Trend-Pullback Strategy:
    Buys shallow dips to EMA20/VWAP during confirmed uptrends (EMA20 > SMA50).
    Captures trend continuation with tight stops and high reward-to-risk.
    """
    def __init__(self, name="MomentumBreakoutStrategy", default_weight=1.0):
        super().__init__(name, default_weight)

    def evaluate(self, symbol, data, **kwargs):
        try:
            price = data.get('price', 0)
            ema20 = data.get('ema_20', 0)
            sma50 = data.get('sma_50', 0)
            rsi = data.get('rsi', 50)
            macd = data.get('macd', 0)
            macd_sig = data.get('macd_signal', 0)
            open_p = data.get('open', price)
            
            if price == 0 or ema20 == 0 or sma50 == 0:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Missing data'}
            
            # Uptrend: EMA20 above SMA50
            is_uptrend = ema20 > sma50 * 0.998
            # Pullback to EMA20: price near EMA20 (within 0.4%)
            dist_to_ema = abs(price - ema20) / ema20
            at_support = dist_to_ema <= 0.005 or (price >= ema20 * 0.995 and price <= ema20 * 1.005)
            # Health check: RSI between 38 and 60 (healthy dip, not oversold crash, not overbought)
            healthy_rsi = 38 <= rsi <= 60
            # Bounce confirmation: candle is green (Close >= Open) and MACD > signal
            bounce = (price >= open_p) and (macd >= macd_sig * 0.98)
            
            if is_uptrend and at_support and healthy_rsi and bounce:
                conf = min(0.92, 0.65 + (1 - dist_to_ema * 100) * 0.1)
                return {'action': 'BUY', 'confidence': conf, 'reason': f'Trend Pullback to EMA20 (dist={dist_to_ema:.2%}), RSI={rsi:.1f}'}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
        return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Conditions not met'}

    def get_exit(self, symbol, entry_price, data, **kwargs):
        price = data.get('price', 0)
        rsi = data.get('rsi', 50)
        high_20 = data.get('high_20', 0)
        
        # Take profit near recent 20-candle high or overbought RSI
        if high_20 > 0 and price >= high_20 * 0.998:
            return {'should_exit': True, 'reason': 'Reached 20-period swing high target'}
        if rsi >= 68:
            return {'should_exit': True, 'reason': 'RSI overbought (>= 68)'}
        return {'should_exit': False, 'reason': 'Holding trend continuation'}

class VWAPReversionStrategy(BaseStrategy):
    def __init__(self, name="VWAPReversionStrategy", default_weight=1.0):
        super().__init__(name, default_weight)

    def evaluate(self, symbol, data, **kwargs):
        try:
            intraday_data = kwargs.get('intraday_data', pd.DataFrame())
            if intraday_data.empty or 'VWAP' not in intraday_data.columns:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No VWAP data'}
                
            current_vwap = float(intraday_data['VWAP'].iloc[-1])
            price = data.get('price', 0)
            rsi = data.get('rsi', 50)
            
            if price == 0 or current_vwap == 0: return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Zero price/VWAP'}
            
            dev = (current_vwap - price) / current_vwap
            if dev >= 0.005 and rsi <= 45:
                conf = min(0.90, 0.55 + (dev - 0.005) * 20)
                return {'action': 'BUY', 'confidence': conf, 'reason': f'VWAP Deviation {dev:.2%}, RSI={rsi:.1f}'}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
        return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Conditions not met'}

    def get_exit(self, symbol, entry_price, data, **kwargs):
        intraday_data = kwargs.get('intraday_data', pd.DataFrame())
        if not intraday_data.empty and 'VWAP' in intraday_data.columns:
            current_vwap = intraday_data['VWAP'].iloc[-1]
            price = data.get('price', 0)
            if price >= current_vwap * 1.005:
                return {'should_exit': True, 'reason': 'Exceeded VWAP by 0.5%'}
            if price >= current_vwap:
                return {'should_exit': True, 'reason': 'Returned to VWAP'}
        return {'should_exit': False, 'reason': 'Holding for VWAP reversion'}

class ORBStrategy(BaseStrategy):
    def __init__(self, name="ORBStrategy", default_weight=1.0):
        super().__init__(name, default_weight)

    def evaluate(self, symbol, data, **kwargs):
        try:
            intraday_data = kwargs.get('intraday_data', pd.DataFrame())
            if intraday_data.empty: return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No intraday data'}
            
            last_idx = intraday_data.index[-1]
            candle_time = last_idx.time() if hasattr(last_idx, 'time') else datetime.now(IST).time()
            
            if not (time(9, 30) <= candle_time <= time(12, 30)):
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': f'Outside ORB window ({candle_time})'}
                
            if hasattr(last_idx, 'date'):
                today_date = last_idx.date()
                today_candles = intraday_data[intraday_data.index.date == today_date]
            else:
                today_candles = intraday_data.tail(25)
                
            if len(today_candles) < 2:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Waiting for first candle'}
            
            first_candle = today_candles.iloc[0]
            orb_high = float(first_candle['High'])
            price = data.get('price', 0)
            vol_ratio = data.get('volume_ratio', 1.0)
            
            if price > orb_high and vol_ratio >= 1.1:
                return {'action': 'BUY', 'confidence': 0.72, 'reason': f'ORB Breakout above {orb_high:.2f}'}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
        return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Conditions not met'}

    def get_exit(self, symbol, entry_price, data, **kwargs):
        price = data.get('price', 0)
        atr = data.get('atr', 0)
        
        if atr > 0:
            if price >= entry_price + 1.5 * atr:
                return {'should_exit': True, 'reason': 'ORB Target hit (1.5x ATR)'}
            if price <= entry_price - 0.5 * atr:
                return {'should_exit': True, 'reason': 'ORB Stop hit (0.5x ATR)'}
        return {'should_exit': False, 'reason': 'Inside ORB range'}

class GapStrategy(BaseStrategy):
    def __init__(self, name="GapStrategy", default_weight=1.0):
        super().__init__(name, default_weight)

    def evaluate(self, symbol, data, **kwargs):
        try:
            prev_close = data.get('prev_close', 0)
            day_open = data.get('day_open', data.get('open', 0))
            price = data.get('price', 0)
            ema20 = data.get('ema_20', 0)
            sma50 = data.get('sma_50', 0)
            vol_ratio = data.get('volume_ratio', 1.0)
            
            if prev_close == 0 or day_open == 0: return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Missing price data'}
            
            gap_pct = (day_open - prev_close) / prev_close
            
            if gap_pct >= 0.004 and price >= day_open and ema20 >= sma50 * 0.99:
                conf = min(0.85, 0.60 + gap_pct * 10)
                return {'action': 'BUY', 'confidence': conf, 'reason': f'Gap Up {gap_pct:.2%} with trend support'}
        except Exception as e:
            logger.error(f"{self.name} error: {e}")
        return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Conditions not met'}

    def get_exit(self, symbol, entry_price, data, **kwargs):
        price = data.get('price', 0)
        prev_close = data.get('prev_close', 0)
        
        if price <= prev_close and prev_close > 0:
            return {'should_exit': True, 'reason': 'Gap filled'}
        
        open_price = data.get('day_open', data.get('open', 0))
        if open_price > prev_close > 0:
            gap_mid = prev_close + (open_price - prev_close) / 2
            if price <= gap_mid:
                return {'should_exit': True, 'reason': 'Gap 50% stop'}
        return {'should_exit': False, 'reason': 'Gap holding'}

class StrategyEngine:
    def __init__(self):
        self.strategies = {
            "MLEnsemble": MLEnsembleStrategy(),
            "MeanReversion": MeanReversionStrategy(),
            "MomentumBreakout": MomentumBreakoutStrategy(),
            "VWAPReversion": VWAPReversionStrategy(),
            "ORB": ORBStrategy(),
            "Gap": GapStrategy()
        }
        self.load_weights()

    def load_weights(self):
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE, 'r') as f:
                    weights = json.load(f)
                for name, strat in self.strategies.items():
                    if name in weights:
                        strat.weight = weights[name].get('weight', 1.0)
                        strat.enabled = weights[name].get('enabled', True)
                        strat.trades = weights[name].get('trades', 0)
                        strat.wins = weights[name].get('wins', 0)
                        strat.total_pnl = weights[name].get('total_pnl', 0.0)
            except Exception as e:
                logger.error(f"Error loading strategy weights: {e}")

    def save_weights(self):
        os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
        data = {}
        for name, strat in self.strategies.items():
            data[name] = {
                'weight': strat.weight,
                'enabled': strat.enabled,
                'trades': strat.trades,
                'wins': strat.wins,
                'total_pnl': strat.total_pnl
            }
        try:
            with open(WEIGHTS_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving strategy weights: {e}")

    def evaluate_all(self, symbol: str, current_data: dict, intraday_data: pd.DataFrame, 
                     model_predictions: dict = None, sentiment_score: float = 0.0, market_mood: float = 0.0) -> dict:
        """
        Run all enabled strategies and return aggregate signal.
        """
        
        # Calculate VWAP if intraday data exists
        if intraday_data is not None and not intraday_data.empty:
            if 'VWAP' not in intraday_data.columns:
                intraday_data['VWAP'] = calculate_vwap(intraday_data)

        strategy_details = []
        buy_votes = 0
        total_confidence = 0.0
        active_strats = 0
        
        kwargs = {
            'intraday_data': intraday_data,
            'model_predictions': model_predictions or {},
            'sentiment_score': sentiment_score,
            'market_mood': market_mood
        }

        for name, strat in self.strategies.items():
            if not strat.enabled:
                continue
            
            res = strat.evaluate(symbol, current_data, **kwargs)
            res['strategy'] = name
            res['weight'] = strat.weight
            strategy_details.append(res)
            
            if res['action'] == 'BUY':
                buy_votes += 1
                total_confidence += res['confidence'] * strat.weight
                active_strats += strat.weight

        avg_confidence = (total_confidence / active_strats) if active_strats > 0 else 0.0
        action = 'BUY' if buy_votes > 0 and avg_confidence >= 0.5 else 'HOLD'

        return {
            'action': action,
            'confidence': avg_confidence,
            'strategies_agree': buy_votes,
            'strategy_details': strategy_details,
            'position_size_modifier': 1.0 + (avg_confidence - 0.5) if action == 'BUY' else 1.0
        }

    def update_weights(self, strategy_name: str, win: bool, pnl: float):
        if strategy_name in self.strategies:
            strat = self.strategies[strategy_name]
            strat.update_performance(win, pnl)
            
            # Simple adaptive weighting logic
            if win:
                strat.weight = min(2.0, strat.weight + 0.05)
            else:
                strat.weight = max(0.5, strat.weight - 0.05)
                
            self.save_weights()

    def get_exit_signal(self, symbol: str, entry_price: float, current_data: dict, strategy_name: str, intraday_data: pd.DataFrame = None) -> dict:
        if strategy_name in self.strategies and self.strategies[strategy_name].enabled:
            kwargs = {'intraday_data': intraday_data}
            return self.strategies[strategy_name].get_exit(symbol, entry_price, current_data, **kwargs)
        return {'should_exit': False, 'reason': 'Strategy not found or disabled'}

    def get_performance_report(self) -> dict:
        report = {}
        for name, strat in self.strategies.items():
            win_rate = (strat.wins / strat.trades) if strat.trades > 0 else 0
            report[name] = {
                'enabled': strat.enabled,
                'weight': strat.weight,
                'trades': strat.trades,
                'win_rate': win_rate,
                'total_pnl': strat.total_pnl
            }
        return report

if __name__ == "__main__":
    engine = StrategyEngine()
    print("Multi-Strategy Engine Initialized")
    print(engine.get_performance_report())
