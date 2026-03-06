"""
====================================================
🧬 PROJECT AEGIS — Genetic Algorithm Strategy Evolver
====================================================
Phase 13 · Fully Dynamic

Evolves trading strategy parameters using a real genetic algorithm
with tournament selection, single-point crossover, and mutation.
Fitness is measured by walk-forward Sharpe ratio on REAL historical
price data fetched via yfinance. No static or fake values.

Chromosome genes:
  - RSI buy/sell thresholds
  - MACD fast/slow/signal periods
  - EMA short/long crossover windows
  - ATR multiplier for stop-loss
  - Bollinger Band period and std-dev width
  - Volume spike threshold
  - Position sizing factor (fraction of capital)

====================================================
"""

import os, json, time, random, math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    IST = None

# ── Paths ────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "ga_evolver_state.json")

# ── Gene Definitions (ranges for each parameter) ─
GENE_SPEC = {
    "rsi_buy":           {"min": 15,  "max": 45,  "type": "int"},
    "rsi_sell":          {"min": 55,  "max": 85,  "type": "int"},
    "macd_fast":         {"min": 6,   "max": 18,  "type": "int"},
    "macd_slow":         {"min": 18,  "max": 40,  "type": "int"},
    "macd_signal":       {"min": 5,   "max": 15,  "type": "int"},
    "ema_short":         {"min": 5,   "max": 20,  "type": "int"},
    "ema_long":          {"min": 30,  "max": 80,  "type": "int"},
    "atr_sl_mult":       {"min": 1.0, "max": 4.0, "type": "float"},
    "bb_period":         {"min": 10,  "max": 30,  "type": "int"},
    "bb_std":            {"min": 1.5, "max": 3.0, "type": "float"},
    "volume_spike":      {"min": 1.2, "max": 3.0, "type": "float"},
    "position_frac":     {"min": 0.05,"max": 0.25,"type": "float"},
}

GENE_NAMES = list(GENE_SPEC.keys())

# ── GA Config ────────────────────────────────────
POP_SIZE         = 60       # Population size
N_GENERATIONS    = 30       # Max generations per evolution cycle
TOURNAMENT_K     = 3        # Tournament selection size
CROSSOVER_PROB   = 0.85     # Probability of crossover
MUTATION_PROB    = 0.15     # Per-gene mutation probability
ELITE_SIZE       = 4        # Number of elites preserved per generation
LOOKBACK_DAYS    = 180      # Historical data for fitness eval
WALK_FWD_SPLIT   = 0.7      # 70% train, 30% test (walk-forward)


# ═══════════════════════════════════════════════════
#  CHROMOSOME HELPERS
# ═══════════════════════════════════════════════════

def _random_gene(name: str):
    spec = GENE_SPEC[name]
    if spec["type"] == "int":
        return random.randint(spec["min"], spec["max"])
    return round(random.uniform(spec["min"], spec["max"]), 3)


def create_random_chromosome() -> Dict:
    chrom = {}
    for name in GENE_NAMES:
        chrom[name] = _random_gene(name)
    # Enforce macd_slow > macd_fast
    if chrom["macd_slow"] <= chrom["macd_fast"]:
        chrom["macd_slow"] = chrom["macd_fast"] + random.randint(2, 10)
    # Enforce ema_long > ema_short
    if chrom["ema_long"] <= chrom["ema_short"]:
        chrom["ema_long"] = chrom["ema_short"] + random.randint(10, 30)
    return chrom


def crossover(parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
    """Single-point crossover on the gene list."""
    if random.random() > CROSSOVER_PROB:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(GENE_NAMES) - 1)
    child1, child2 = {}, {}
    for i, name in enumerate(GENE_NAMES):
        if i < point:
            child1[name] = parent1[name]
            child2[name] = parent2[name]
        else:
            child1[name] = parent2[name]
            child2[name] = parent1[name]
    # Repair constraints
    for c in (child1, child2):
        if c["macd_slow"] <= c["macd_fast"]:
            c["macd_slow"] = c["macd_fast"] + 2
        if c["ema_long"] <= c["ema_short"]:
            c["ema_long"] = c["ema_short"] + 10
    return child1, child2


def mutate(chrom: Dict) -> Dict:
    """Per-gene Gaussian mutation within allowed ranges."""
    mutated = chrom.copy()
    for name in GENE_NAMES:
        if random.random() < MUTATION_PROB:
            spec = GENE_SPEC[name]
            spread = (spec["max"] - spec["min"]) * 0.15
            new_val = mutated[name] + random.gauss(0, spread)
            new_val = max(spec["min"], min(spec["max"], new_val))
            if spec["type"] == "int":
                new_val = int(round(new_val))
            else:
                new_val = round(new_val, 3)
            mutated[name] = new_val
    # Repair constraints
    if mutated["macd_slow"] <= mutated["macd_fast"]:
        mutated["macd_slow"] = mutated["macd_fast"] + 2
    if mutated["ema_long"] <= mutated["ema_short"]:
        mutated["ema_long"] = mutated["ema_short"] + 10
    return mutated


def tournament_select(population: List[Dict], fitnesses: List[float]) -> Dict:
    """Tournament selection: pick K random, return the fittest."""
    indices = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# ═══════════════════════════════════════════════════
#  FITNESS EVALUATION — Real Data Backtest
# ═══════════════════════════════════════════════════

def _fetch_price_data(symbols: List[str], days: int = LOOKBACK_DAYS) -> Dict[str, pd.DataFrame]:
    """Fetch REAL historical OHLCV data for all symbols via yfinance."""
    result = {}
    if yf is None:
        return result
    end = datetime.now()
    start = end - timedelta(days=days + 30)  # extra buffer for indicators
    for sym in symbols:
        try:
            df = yf.download(sym, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            if df is not None and len(df) > 60:
                df = df.copy()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                result[sym] = df
        except Exception:
            pass
    return result


def _compute_indicators(df: pd.DataFrame, chrom: Dict) -> pd.DataFrame:
    """Compute technical indicators from chromosome genes on REAL price data."""
    df = df.copy()
    close = df["Close"].values.astype(float)

    # RSI
    period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().values
    avg_loss = pd.Series(loss).rolling(period).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    fast_ema = pd.Series(close).ewm(span=chrom["macd_fast"], adjust=False).mean()
    slow_ema = pd.Series(close).ewm(span=chrom["macd_slow"], adjust=False).mean()
    df["MACD"] = fast_ema - slow_ema
    df["MACD_Signal"] = df["MACD"].ewm(span=chrom["macd_signal"], adjust=False).mean()

    # EMA crossover
    df["EMA_Short"] = pd.Series(close).ewm(span=chrom["ema_short"], adjust=False).mean()
    df["EMA_Long"] = pd.Series(close).ewm(span=chrom["ema_long"], adjust=False).mean()

    # ATR
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    df["ATR"] = pd.Series(tr).rolling(14).mean().values

    # Bollinger Bands
    df["BB_Mid"] = pd.Series(close).rolling(chrom["bb_period"]).mean()
    bb_std = pd.Series(close).rolling(chrom["bb_period"]).std()
    df["BB_Lower"] = df["BB_Mid"] - chrom["bb_std"] * bb_std

    # Volume spike
    if "Volume" in df.columns:
        df["Vol_MA"] = df["Volume"].rolling(20).mean()
        df["Vol_Spike"] = df["Volume"] / df["Vol_MA"].replace(0, 1)
    else:
        df["Vol_Spike"] = 1.0

    return df.dropna()


def _simulate_strategy(df: pd.DataFrame, chrom: Dict) -> Dict:
    """Run a simple BUY/SELL simulation using chromosome rules on REAL data.
    Returns dict with sharpe, total_return, win_rate, n_trades."""
    if len(df) < 20:
        return {"sharpe": -999, "total_return": 0, "win_rate": 0, "n_trades": 0}

    close = df["Close"].values.astype(float)
    rsi = df["RSI"].values.astype(float)
    macd = df["MACD"].values.astype(float)
    macd_sig = df["MACD_Signal"].values.astype(float)
    ema_s = df["EMA_Short"].values.astype(float)
    ema_l = df["EMA_Long"].values.astype(float)
    atr = df["ATR"].values.astype(float)
    bb_lower = df["BB_Lower"].values.astype(float)
    vol_spike = df["Vol_Spike"].values.astype(float)

    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    trades = []
    daily_returns = []

    for i in range(1, len(close)):
        if not in_position:
            buy_signal = (
                rsi[i] < chrom["rsi_buy"]
                and macd[i] > macd_sig[i]
                and ema_s[i] > ema_l[i]
                and close[i] < bb_lower[i] * 1.02  # near lower band
                and vol_spike[i] > chrom["volume_spike"]
            )
            if buy_signal:
                entry_price = close[i]
                stop_loss = entry_price - chrom["atr_sl_mult"] * atr[i]
                in_position = True
        else:
            sell_signal = (
                rsi[i] > chrom["rsi_sell"]
                or macd[i] < macd_sig[i]
                or close[i] <= stop_loss
            )
            if sell_signal:
                pnl_pct = (close[i] - entry_price) / entry_price
                trades.append(pnl_pct)
                daily_returns.append(pnl_pct)
                in_position = False
        # Track daily returns for Sharpe
        if i > 0 and not in_position:
            daily_returns.append(0)
        elif in_position:
            daily_returns.append((close[i] - close[i - 1]) / close[i - 1])

    if not daily_returns:
        return {"sharpe": -999, "total_return": 0, "win_rate": 0, "n_trades": 0}

    returns = np.array(daily_returns)
    mean_r = np.mean(returns)
    std_r = np.std(returns) if np.std(returns) > 0 else 1e-8
    sharpe = (mean_r / std_r) * math.sqrt(252)
    total_ret = float(np.sum(returns))
    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / len(trades) if trades else 0

    return {
        "sharpe": round(sharpe, 4),
        "total_return": round(total_ret * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": len(trades),
    }


def evaluate_fitness(chrom: Dict, price_data: Dict[str, pd.DataFrame]) -> float:
    """Walk-forward fitness: train on first 70%, test on last 30%.
    Uses REAL price data. Returns average Sharpe across all symbols."""
    sharpes = []
    for sym, df in price_data.items():
        split_idx = int(len(df) * WALK_FWD_SPLIT)
        test_df = df.iloc[split_idx:]
        if len(test_df) < 20:
            continue
        ind_df = _compute_indicators(test_df, chrom)
        result = _simulate_strategy(ind_df, chrom)
        sharpes.append(result["sharpe"])
    if not sharpes:
        return -999.0
    return float(np.mean(sharpes))


# ═══════════════════════════════════════════════════
#  MAIN EVOLUTION ENGINE
# ═══════════════════════════════════════════════════

def evolve_strategies(symbols: List[str],
                      n_generations: int = N_GENERATIONS,
                      pop_size: int = POP_SIZE,
                      existing_state: Optional[Dict] = None) -> Dict:
    """Run a full GA evolution cycle on REAL stock data.
    Returns state dict with best chromosome, fitness history, population stats."""

    price_data = _fetch_price_data(symbols)
    if not price_data:
        return {
            "status": "NO_DATA",
            "message": "Could not fetch price data for any symbol",
            "timestamp": datetime.now().isoformat(),
        }

    # Initialize or continue population
    if existing_state and existing_state.get("population"):
        population = existing_state["population"][:pop_size]
        # Top up if needed
        while len(population) < pop_size:
            population.append(create_random_chromosome())
        gen_start = existing_state.get("generation", 0)
    else:
        population = [create_random_chromosome() for _ in range(pop_size)]
        gen_start = 0

    best_ever = None
    best_ever_fitness = -999.0
    fitness_history = existing_state.get("fitness_history", []) if existing_state else []
    gen_stats = []

    for gen in range(gen_start, gen_start + n_generations):
        # Evaluate all chromosomes
        fitnesses = [evaluate_fitness(c, price_data) for c in population]

        # Track stats
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best = fitnesses[gen_best_idx]
        gen_mean = float(np.mean(fitnesses))
        gen_std = float(np.std(fitnesses))

        if gen_best > best_ever_fitness:
            best_ever_fitness = gen_best
            best_ever = population[gen_best_idx].copy()

        gen_stats.append({
            "generation": gen + 1,
            "best_fitness": round(gen_best, 4),
            "mean_fitness": round(gen_mean, 4),
            "std_fitness": round(gen_std, 4),
        })
        fitness_history.append(round(gen_best, 4))

        # Elitism: keep top ELITE_SIZE
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        new_pop = [population[i].copy() for i in sorted_indices[:ELITE_SIZE]]

        # Fill rest via tournament + crossover + mutation
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    # Final evaluation
    final_fitnesses = [evaluate_fitness(c, price_data) for c in population]
    final_best_idx = int(np.argmax(final_fitnesses))
    if final_fitnesses[final_best_idx] > best_ever_fitness:
        best_ever = population[final_best_idx].copy()
        best_ever_fitness = final_fitnesses[final_best_idx]

    # Detailed backtest of best chromosome
    best_results = {}
    for sym, df in price_data.items():
        split_idx = int(len(df) * WALK_FWD_SPLIT)
        test_df = df.iloc[split_idx:]
        if len(test_df) < 20:
            continue
        ind_df = _compute_indicators(test_df, best_ever)
        best_results[sym] = _simulate_strategy(ind_df, best_ever)

    state = {
        "status": "EVOLVED",
        "generation": gen_start + n_generations,
        "best_chromosome": best_ever,
        "best_fitness": round(best_ever_fitness, 4),
        "fitness_history": fitness_history,
        "generation_stats": gen_stats,
        "best_backtest": best_results,
        "population": population[:10],  # Keep top 10 for continuation
        "symbols_used": list(price_data.keys()),
        "data_points": {s: len(d) for s, d in price_data.items()},
        "timestamp": datetime.now().isoformat(),
    }
    return state


def get_best_strategy(state: Optional[Dict] = None) -> Dict:
    """Return the best chromosome as strategy parameter overrides."""
    if state is None:
        state = load_evolver_state()
    chrom = state.get("best_chromosome")
    if not chrom:
        return {}
    return {
        "rsi_buy_threshold": chrom.get("rsi_buy", 30),
        "rsi_sell_threshold": chrom.get("rsi_sell", 70),
        "macd_fast": chrom.get("macd_fast", 12),
        "macd_slow": chrom.get("macd_slow", 26),
        "macd_signal": chrom.get("macd_signal", 9),
        "ema_short": chrom.get("ema_short", 12),
        "ema_long": chrom.get("ema_long", 50),
        "atr_sl_multiplier": chrom.get("atr_sl_mult", 2.0),
        "bb_period": chrom.get("bb_period", 20),
        "bb_std_dev": chrom.get("bb_std", 2.0),
        "volume_spike_threshold": chrom.get("volume_spike", 1.5),
        "position_fraction": chrom.get("position_frac", 0.10),
        "ga_fitness": state.get("best_fitness", 0),
        "ga_generation": state.get("generation", 0),
    }


def get_evolver_status(state: Optional[Dict] = None) -> Dict:
    """Return a summary for dashboard display."""
    if state is None:
        state = load_evolver_state()
    if not state or state.get("status") == "NO_DATA":
        return {"status": "NO_DATA"}
    return {
        "status": state.get("status", "UNKNOWN"),
        "generation": state.get("generation", 0),
        "best_fitness": state.get("best_fitness", 0),
        "fitness_history": state.get("fitness_history", []),
        "generation_stats": state.get("generation_stats", []),
        "best_chromosome": state.get("best_chromosome", {}),
        "best_backtest": state.get("best_backtest", {}),
        "symbols_used": state.get("symbols_used", []),
        "timestamp": state.get("timestamp", "—"),
    }


# ── Persistence ──────────────────────────────────

def save_evolver_state(state: Dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_evolver_state() -> Dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
