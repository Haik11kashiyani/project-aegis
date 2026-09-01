import json
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger('PerformanceTracker')

class PerformanceTracker:
    """
    Tracks, analyzes, and reports on trading system performance in real-time.
    Provides degradation detection, benchmark comparison, and evolution snapshots.
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.metrics_file = os.path.join(data_dir, 'performance_metrics.json')
        self.snapshots_file = os.path.join(data_dir, 'evolution_snapshots.json')
        self.trade_history_file = os.path.join(data_dir, 'trade_history.csv')
        self.initial_capital = 15000  # Will be loaded from config
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> dict:
        """Load persisted metrics or initialize defaults"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
                
        initial_state = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'peak_equity': self.initial_capital,
            'current_equity': self.initial_capital,
            'strategies': {},
            'last_updated': datetime.now(IST).isoformat()
        }
        # Bootstrap past memory from trade_history.csv if metrics file is missing
        if os.path.exists(self.trade_history_file) and os.path.getsize(self.trade_history_file) > 0:
            try:
                df = pd.read_csv(self.trade_history_file)
                pnl_col = next((c for c in ["Actual_Profit", "pnl", "PnL", "profit"] if c in df.columns), None)
                if pnl_col and not df.empty:
                    pnl_vals = pd.to_numeric(df[pnl_col], errors='coerce').fillna(0)
                    total_pnl = float(pnl_vals.sum())
                    initial_state['total_trades'] = len(df)
                    initial_state['wins'] = int((pnl_vals > 0).sum())
                    initial_state['losses'] = int((pnl_vals < 0).sum())
                    initial_state['total_pnl'] = round(total_pnl, 2)
                    initial_state['current_equity'] = round(self.initial_capital + total_pnl, 2)
                    initial_state['peak_equity'] = max(self.initial_capital, initial_state['current_equity'])
            except Exception as e:
                logger.warning(f"Failed to bootstrap metrics from history: {e}")
        return initial_state
    
    def _save_metrics(self):
        """Persist metrics to disk"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            
    def update_from_trade(self, trade_data: dict):
        """
        Called after every closed trade to update metrics.
        trade_data: {'symbol', 'entry_price', 'exit_price', 'quantity', 'pnl', 
                     'strategy', 'entry_time', 'exit_time', 'confidence'}
        """
        pnl = trade_data.get('pnl', 0)
        strategy = trade_data.get('strategy', 'unknown')
        
        self.metrics['total_trades'] += 1
        if pnl > 0:
            self.metrics['wins'] += 1
        else:
            self.metrics['losses'] += 1
            
        self.metrics['total_pnl'] += pnl
        self.metrics['current_equity'] = self.initial_capital + self.metrics['total_pnl']
        self.metrics['peak_equity'] = max(self.metrics['peak_equity'], self.metrics['current_equity'])
        
        if strategy not in self.metrics['strategies']:
            self.metrics['strategies'][strategy] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            
        self.metrics['strategies'][strategy]['trades'] += 1
        if pnl > 0:
            self.metrics['strategies'][strategy]['wins'] += 1
        self.metrics['strategies'][strategy]['pnl'] += pnl
        
        self.metrics['last_updated'] = datetime.now(IST).isoformat()
        self._save_metrics()
        
    def get_rolling_metrics(self, window_days=20) -> dict:
        """Calculate rolling performance metrics"""
        if not os.path.exists(self.trade_history_file):
            return {}
            
        try:
            df = pd.read_csv(self.trade_history_file)
            if df.empty:
                return {}
                
            date_col = 'exit_date' if 'exit_date' in df.columns else 'Date'
            if date_col not in df.columns:
                return {}
                
            pnl_col = 'pnl' if 'pnl' in df.columns else 'Actual_Profit'
            
            df[date_col] = pd.to_datetime(df[date_col])
            cutoff_date = datetime.now(IST) - timedelta(days=window_days)  # M6-FIX: Use IST timezone
            
            if df[date_col].dt.tz is not None:
                cutoff_date = pytz.UTC.localize(cutoff_date)
            
            recent_trades = df[df[date_col] >= cutoff_date]
            
            if recent_trades.empty:
                return {}
                
            wins = len(recent_trades[recent_trades[pnl_col] > 0])
            total = len(recent_trades)
            win_rate = wins / total if total > 0 else 0
            
            recent_trades['date_only'] = recent_trades[date_col].dt.date
            daily_pnl = recent_trades.groupby('date_only')[pnl_col].sum()
            
            returns = daily_pnl / self.initial_capital
            rolling_sharpe = self._calculate_sharpe(returns) if len(returns) > 1 else 0
            
            profit_factor = self._calculate_profit_factor(recent_trades[pnl_col])
            
            return {
                'rolling_win_rate': round(win_rate, 4),
                'rolling_sharpe': round(rolling_sharpe, 4),
                'rolling_profit_factor': round(profit_factor, 4),
                'trades_in_window': total
            }
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return {}

    def _calculate_sharpe(self, returns, risk_free_rate=0.065):
        """Annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        return np.sqrt(252) * (excess_returns.mean() / returns.std())
        
    def _calculate_max_drawdown(self, equity_curve):
        """Maximum peak-to-trough drawdown"""
        if len(equity_curve) == 0:
            return 0.0
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return abs(drawdown.min())
        
    def _calculate_profit_factor(self, pnls):
        """Gross Profits / Gross Losses"""
        gross_profit = pnls[pnls > 0].sum()
        gross_loss = abs(pnls[pnls < 0].sum())
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def detect_degradation(self) -> dict:
        """Early warning system for performance degradation"""
        rolling = self.get_rolling_metrics(window_days=20)
        
        status = 'HEALTHY'
        issues = []
        recommendations = []
        
        if not rolling:
            return {'status': status, 'issues': ['Not enough data'], 'recommendations': []}
            
        win_rate = rolling.get('rolling_win_rate', 0)
        sharpe = rolling.get('rolling_sharpe', 0)
        pf = rolling.get('rolling_profit_factor', 0)
        
        peak = self.metrics.get('peak_equity', self.initial_capital)
        current = self.metrics.get('current_equity', self.initial_capital)
        current_dd = (peak - current) / peak if peak > 0 else 0
        
        if current_dd > 0.15:
            status = 'CRITICAL'
            issues.append(f"Drawdown {current_dd*100:.1f}% exceeds 15% critical limit")
            recommendations.append("Halt all trading immediately")
        elif current_dd > 0.10:
            if status != 'CRITICAL': status = 'WARNING'
            issues.append(f"Drawdown {current_dd*100:.1f}% exceeds 10% warning limit")
            recommendations.append("Reduce position sizes, review recent trades")
            
        if win_rate < 0.35 and rolling.get('trades_in_window', 0) >= 30:
            if status != 'CRITICAL': status = 'WARNING'
            issues.append(f"Win rate {win_rate*100:.1f}% below 35% threshold")
            recommendations.append("Check market regime, consider tightening entry conditions")
            
        if pf < 1.0 and rolling.get('trades_in_window', 0) >= 30:
            status = 'CRITICAL'
            issues.append(f"Profit factor {pf:.2f} below 1.0")
            recommendations.append("System is losing money. Pause and retrain.")
            
        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations
        }

    def compare_live_vs_backtest(self, backtest_file='data/backtest_results.csv') -> dict:
        """Compare live trading performance vs backtest expectations."""
        return {
            'status': 'unknown',
            'live_sharpe': 0.0,
            'backtest_sharpe': 0.0,
            'ratio': 0.0
        }

    def get_strategy_breakdown(self) -> dict:
        """Per-strategy performance analysis"""
        breakdown = {}
        for strategy, data in self.metrics.get('strategies', {}).items():
            trades = data['trades']
            wins = data['wins']
            win_rate = wins / trades if trades > 0 else 0
            avg_pnl = data['pnl'] / trades if trades > 0 else 0
            
            recommendation = "keep"
            if trades > 10:
                if win_rate < 0.4 and data['pnl'] < 0:
                    recommendation = "disable"
                elif win_rate > 0.6:
                    recommendation = "increase_weight"
                    
            breakdown[strategy] = {
                'trades': trades,
                'win_rate': round(win_rate, 4),
                'total_pnl': round(data['pnl'], 2),
                'avg_pnl': round(avg_pnl, 2),
                'recommendation': recommendation
            }
        return breakdown

    def get_capital_curve(self) -> dict:
        """Calculate and return capital growth curve"""
        if not os.path.exists(self.trade_history_file):
            return {}
            
        try:
            df = pd.read_csv(self.trade_history_file)
            date_col = 'exit_date' if 'exit_date' in df.columns else 'Date'
            pnl_col = 'pnl' if 'pnl' in df.columns else 'Actual_Profit'
            
            if date_col not in df.columns or pnl_col not in df.columns:
                return {}
                
            df[date_col] = pd.to_datetime(df[date_col])
            df['date_only'] = df[date_col].dt.date
            
            daily_pnl = df.groupby('date_only')[pnl_col].sum().sort_index()
            daily_equity = self.initial_capital + daily_pnl.cumsum()
            
            return {
                'dates': [str(d) for d in daily_equity.index],
                'equity': daily_equity.tolist()
            }
        except Exception as e:
            logger.error(f"Error calculating capital curve: {e}")
            return {}

    def take_evolution_snapshot(self):
        """Weekly snapshot of system state for rollback"""
        snapshot = {
            'timestamp': datetime.now(IST).isoformat(),
            'metrics': self.metrics.copy(),
            'capital': self.metrics.get('current_equity', self.initial_capital)
        }
        
        snapshots = []
        if os.path.exists(self.snapshots_file):
            try:
                with open(self.snapshots_file, 'r') as f:
                    snapshots = json.load(f)
            except Exception:
                pass
                
        snapshots.append(snapshot)
        snapshots = snapshots[-52:]
        
        try:
            with open(self.snapshots_file, 'w') as f:
                json.dump(snapshots, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

    def get_capital_tier(self, current_capital) -> str:
        """Determine current capital tier and allowed features"""
        if current_capital < 25000:
            return "CONSERVATIVE"
        elif current_capital <= 50000:
            return "MODERATE"
        else:
            return "FULL"

    def generate_dashboard_data(self) -> dict:
        """Generate JSON for dashboard display"""
        dashboard_file = os.path.join(self.data_dir, 'performance_dashboard.json')
        
        data = {
            'metrics': self.metrics,
            'rolling': self.get_rolling_metrics(),
            'degradation': self.detect_degradation(),
            'strategies': self.get_strategy_breakdown(),
            'tier': self.get_capital_tier(self.metrics.get('current_equity', self.initial_capital)),
            'timestamp': datetime.now(IST).isoformat()
        }
        
        try:
            with open(dashboard_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dashboard data: {e}")
            
        return data
