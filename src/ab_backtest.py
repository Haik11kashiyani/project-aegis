"""
====================================================
PROJECT AEGIS — A/B Backtesting Framework (Phase 8)
====================================================
Compare two strategy variants over the same historical
period with statistical significance testing.

Features:
 ● Run two backtest variants (A = baseline, B = modified)
 ● Sharpe ratio comparison
 ● Paired t-test on daily returns
 ● Win rate / drawdown / profit factor analysis
 ● Bootstrap confidence intervals
 ● Results saved to data/ab_backtest.json for dashboard

Run:
    python src/ab_backtest.py
====================================================
"""

import os
import json
import math
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("aegis.ab_backtest")

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FILE_AB = os.path.join(DATA, "ab_backtest.json")
FILE_AB_LOG = os.path.join(DATA, "ab_backtest_log.json")


# ══════════════════════════════════════════════════
#  BACKTEST RESULT CONTAINER
# ══════════════════════════════════════════════════
class BacktestResult:
    """Container for a single backtest run."""

    def __init__(self, name: str):
        self.name = name
        self.trades: List[Dict] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
        self.total_pnl: float = 0.0
        self.total_return_pct: float = 0.0
        self.win_rate: float = 0.0
        self.max_drawdown: float = 0.0
        self.sharpe: float = 0.0
        self.profit_factor: float = 0.0
        self.num_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.avg_win: float = 0.0
        self.avg_loss: float = 0.0

    def compute_metrics(self, capital: float = 1000.0):
        """Compute all performance metrics from trade list."""
        self.num_trades = len(self.trades)
        if self.num_trades == 0:
            return

        pnls = [t.get("pnl", 0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        self.wins = len(wins)
        self.losses = len(losses)
        self.win_rate = (self.wins / self.num_trades * 100) if self.num_trades > 0 else 0
        self.total_pnl = sum(pnls)
        self.total_return_pct = (self.total_pnl / capital * 100) if capital > 0 else 0

        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = abs(np.mean(losses)) if losses else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.01
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity curve
        equity = capital
        self.equity_curve = [capital]
        for pnl in pnls:
            equity += pnl
            self.equity_curve.append(equity)

        # Max drawdown
        peak = capital
        max_dd = 0
        for val in self.equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        self.max_drawdown = max_dd

        # Daily returns → Sharpe
        if len(pnls) >= 2:
            returns = np.array(pnls) / capital
            self.daily_returns = returns.tolist()
            avg_r = np.mean(returns)
            std_r = np.std(returns, ddof=1)
            self.sharpe = (avg_r / std_r * math.sqrt(252)) if std_r > 0 else 0
        else:
            self.sharpe = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "num_trades": self.num_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "profit_factor": round(self.profit_factor, 3) if self.profit_factor != float("inf") else 999.0,
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "equity_curve": [round(e, 2) for e in self.equity_curve[-100:]],  # last 100
        }


# ══════════════════════════════════════════════════
#  STATISTICAL TESTS
# ══════════════════════════════════════════════════
def paired_ttest(returns_a: List[float], returns_b: List[float]) -> Dict:
    """
    Paired t-test on daily returns.
    H0: mean(A) = mean(B)
    """
    n = min(len(returns_a), len(returns_b))
    if n < 5:
        return {"t_stat": 0, "p_value": 1.0, "significant": False, "n": n}

    a = np.array(returns_a[:n])
    b = np.array(returns_b[:n])
    diff = a - b

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se = std_diff / math.sqrt(n)

    t_stat = mean_diff / se if se > 0 else 0

    # Approximate p-value using t-distribution (Student's)
    # Simple approximation for two-tailed test
    df = n - 1
    x = abs(t_stat)
    # Using normal approximation for large df
    if df >= 30:
        from math import erf
        p_value = 1 - erf(x / math.sqrt(2))
    else:
        # Rough approximation for small df
        p_value = 2 * (1 - _student_cdf(x, df))

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(max(p_value, 0.0001), 4),
        "significant": p_value < 0.05,
        "confidence": round((1 - p_value) * 100, 2),
        "n": n,
        "mean_diff": round(mean_diff, 6),
    }


def _student_cdf(x: float, df: int) -> float:
    """Approximate Student's t CDF using beta incomplete function approximation."""
    # Simple normal approximation for moderate df
    if df >= 10:
        z = x * (1 - 1 / (4 * df)) / math.sqrt(1 + x**2 / (2 * df))
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    # For small df, rough approximation
    t2 = x * x
    y = t2 / (df + t2)
    p = 0.5 * (1 + math.copysign(1, x) * (1 - (1 - y) ** (df / 2)))
    return min(max(p, 0), 1)


def sharpe_comparison(sharpe_a: float, sharpe_b: float, n_a: int, n_b: int) -> Dict:
    """
    Test if two Sharpe ratios are statistically different.
    Uses Jobson-Korkie test approximation.
    """
    diff = sharpe_a - sharpe_b
    # Standard error of Sharpe difference (approximate)
    se = math.sqrt((2 / max(n_a, 2)) + (sharpe_a**2 + sharpe_b**2) / (4 * max(n_a, 2)))
    z_stat = diff / se if se > 0 else 0
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_stat) / math.sqrt(2))))

    return {
        "sharpe_diff": round(diff, 4),
        "z_stat": round(z_stat, 4),
        "p_value": round(max(p_value, 0.0001), 4),
        "significant": p_value < 0.05,
        "winner": "A" if diff > 0 else "B" if diff < 0 else "TIE",
    }


def bootstrap_confidence(returns: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
    """Bootstrap confidence interval for mean return."""
    if len(returns) < 5:
        return {"mean": 0, "lower": 0, "upper": 0, "std": 0}

    arr = np.array(returns)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    means = sorted(means)
    alpha = (1 - ci) / 2
    lower = means[int(alpha * n_bootstrap)]
    upper = means[int((1 - alpha) * n_bootstrap)]

    return {
        "mean": round(float(np.mean(arr)), 6),
        "lower": round(float(lower), 6),
        "upper": round(float(upper), 6),
        "std": round(float(np.std(arr)), 6),
        "ci": ci,
    }


# ══════════════════════════════════════════════════
#  A/B COMPARISON ENGINE
# ══════════════════════════════════════════════════
def compare_strategies(
    result_a: BacktestResult,
    result_b: BacktestResult,
) -> Dict:
    """
    Full statistical comparison of two backtest results.
    """
    comparison = {
        "timestamp": datetime.now(IST).isoformat(),
        "variant_a": result_a.to_dict(),
        "variant_b": result_b.to_dict(),
        "statistical_tests": {},
        "winner": "",
        "summary": "",
    }

    # Paired t-test on returns
    if result_a.daily_returns and result_b.daily_returns:
        comparison["statistical_tests"]["paired_ttest"] = paired_ttest(
            result_a.daily_returns, result_b.daily_returns
        )

    # Sharpe comparison
    comparison["statistical_tests"]["sharpe_test"] = sharpe_comparison(
        result_a.sharpe, result_b.sharpe,
        result_a.num_trades, result_b.num_trades,
    )

    # Bootstrap CI
    if result_a.daily_returns:
        comparison["statistical_tests"]["bootstrap_a"] = bootstrap_confidence(result_a.daily_returns)
    if result_b.daily_returns:
        comparison["statistical_tests"]["bootstrap_b"] = bootstrap_confidence(result_b.daily_returns)

    # Determine winner across multiple metrics
    score_a, score_b = 0, 0

    # Sharpe
    if result_a.sharpe > result_b.sharpe:
        score_a += 1
    else:
        score_b += 1

    # Win rate
    if result_a.win_rate > result_b.win_rate:
        score_a += 1
    else:
        score_b += 1

    # Max drawdown (lower is better)
    if result_a.max_drawdown < result_b.max_drawdown:
        score_a += 1
    else:
        score_b += 1

    # Total return
    if result_a.total_return_pct > result_b.total_return_pct:
        score_a += 1
    else:
        score_b += 1

    # Profit factor
    if result_a.profit_factor > result_b.profit_factor:
        score_a += 1
    else:
        score_b += 1

    if score_a > score_b:
        comparison["winner"] = result_a.name
    elif score_b > score_a:
        comparison["winner"] = result_b.name
    else:
        comparison["winner"] = "TIE"

    comparison["scores"] = {"a": score_a, "b": score_b, "total": 5}

    ttest = comparison["statistical_tests"].get("paired_ttest", {})
    sig = "YES" if ttest.get("significant", False) else "NO"

    comparison["summary"] = (
        f"Winner: {comparison['winner']} ({score_a}-{score_b}/5 metrics). "
        f"Sharpe: {result_a.sharpe:.3f} vs {result_b.sharpe:.3f}. "
        f"Returns: {result_a.total_return_pct:.2f}% vs {result_b.total_return_pct:.2f}%. "
        f"Statistically significant: {sig} (p={ttest.get('p_value', 'N/A')})."
    )

    return comparison


# ══════════════════════════════════════════════════
#  QUICK A/B FROM TRADE HISTORY
# ══════════════════════════════════════════════════
def run_ab_from_trades(
    trades_a: List[Dict],
    trades_b: List[Dict],
    name_a: str = "Baseline (A)",
    name_b: str = "Modified (B)",
    capital: float = 1000.0,
) -> Dict:
    """
    Quick A/B test from two lists of trade dicts.
    Each trade must have 'pnl' key.
    """
    result_a = BacktestResult(name_a)
    result_a.trades = trades_a
    result_a.compute_metrics(capital)

    result_b = BacktestResult(name_b)
    result_b.trades = trades_b
    result_b.compute_metrics(capital)

    return compare_strategies(result_a, result_b)


def run_ab_from_csv(
    csv_path: str,
    split_column: str = "strategy",
    value_a: str = "baseline",
    value_b: str = "modified",
    capital: float = 1000.0,
) -> Dict:
    """
    Run A/B test from a single CSV with a strategy label column.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Cannot read CSV: {e}"}

    if split_column not in df.columns:
        # Fall back: split by date (first half vs second half)
        mid = len(df) // 2
        trades_a = df.iloc[:mid].to_dict("records")
        trades_b = df.iloc[mid:].to_dict("records")
        return run_ab_from_trades(trades_a, trades_b, "First Half (A)", "Second Half (B)", capital)

    trades_a = df[df[split_column] == value_a].to_dict("records")
    trades_b = df[df[split_column] == value_b].to_dict("records")
    return run_ab_from_trades(trades_a, trades_b, value_a, value_b, capital)


# ══════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════
def save_ab_results(data: Dict):
    """Save A/B test results to JSON."""
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(FILE_AB, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save A/B results: {e}")


def load_ab_results() -> Dict:
    """Load latest A/B results."""
    try:
        if os.path.exists(FILE_AB):
            with open(FILE_AB) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(42)

    print("═" * 60)
    print("  A/B BACKTESTING FRAMEWORK — Test")
    print("═" * 60)

    # Simulate two strategies
    n = 50
    trades_a = [{"pnl": float(np.random.normal(5, 20))} for _ in range(n)]
    trades_b = [{"pnl": float(np.random.normal(8, 25))} for _ in range(n)]

    comparison = run_ab_from_trades(trades_a, trades_b, "Conservative", "Aggressive")

    print(f"\n  Strategy A ({comparison['variant_a']['name']}):")
    print(f"    Trades: {comparison['variant_a']['num_trades']} | Win Rate: {comparison['variant_a']['win_rate']}%")
    print(f"    Return: {comparison['variant_a']['total_return_pct']}% | Sharpe: {comparison['variant_a']['sharpe']}")
    print(f"    Max DD: {comparison['variant_a']['max_drawdown']}% | PF: {comparison['variant_a']['profit_factor']}")

    print(f"\n  Strategy B ({comparison['variant_b']['name']}):")
    print(f"    Trades: {comparison['variant_b']['num_trades']} | Win Rate: {comparison['variant_b']['win_rate']}%")
    print(f"    Return: {comparison['variant_b']['total_return_pct']}% | Sharpe: {comparison['variant_b']['sharpe']}")
    print(f"    Max DD: {comparison['variant_b']['max_drawdown']}% | PF: {comparison['variant_b']['profit_factor']}")

    print(f"\n  {comparison['summary']}")

    ttest = comparison['statistical_tests'].get('paired_ttest', {})
    print(f"  T-Test: t={ttest.get('t_stat', 'N/A')}, p={ttest.get('p_value', 'N/A')}")

    # Save results
    save_ab_results(comparison)
    print(f"\n  Results saved to {FILE_AB}")
