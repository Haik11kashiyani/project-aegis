# 🛡️ Project Aegis — Complete End-to-End Guide

> **Start with ₹1,000 → Use AI to paper-trade Indian stocks → Grow ~2% per day**
>
> Everything runs automatically via GitHub Actions (free). After setup, you don't need to touch anything.

---

## 📋 Table of Contents

1. [The Goal](#the-goal)
2. [Phase 1: Scholar (Training)](#phase-1-scholar-training--scholarpy)
3. [Phase 2: Sniper (Live Trading)](#phase-2-sniper-live-trading--sniperpy)
4. [Phase 3: Dashboard (Monitoring)](#phase-3-dashboard-monitoring--dashboardpy)
5. [Phase 4: Backtest (Validation)](#phase-4-backtest-validation--backtestpy)
6. [Daily Timeline](#daily-timeline)
7. [Money Flow](#money-flow-with-1000)
8. [Where It Runs (Free)](#where-it-all-runs-free)
9. [GitHub Secrets Setup](#github-secrets-one-time-setup)
10. [How to Run Locally](#how-to-run-locally)

---

## The Goal

Start with **₹1,000**, use AI to paper-trade Indian stocks, and grow that capital by **~2% per day**. Everything runs automatically via **GitHub Actions** (free) — you don't need to touch anything after setup.

---

## Phase 1: Scholar (Training) — `scholar.py`

> 🕗 **Runs daily at 8:00 AM IST** (before market opens)

### What it does:

1. **Downloads 5 years of stock data** from Yahoo Finance for all 10 stocks in your watchlist:
   - TATA STEEL, SBI, RELIANCE, HDFC BANK, ICICI BANK, NTPC, POWER GRID, COAL INDIA, INFOSYS, TCS

2. **Engineers 12 features** for each stock:
   - **Technical Indicators:** RSI, SMA_50, SMA_200, EMA_20, MACD, Bollinger Bands, ATR, OBV, Volume Ratio
   - **Sentiment Score:** Scrapes recent Google News headlines about each stock and scores them positive/negative using TextBlob

3. **Trains 4 AI models** on each stock:

   | Model | What It Learns | How |
   |-------|---------------|-----|
   | **Random Forest** | Pattern recognition from 12 features | 200 decision trees voting together |
   | **XGBoost** | Same 12 features, learns from mistakes iteratively | 200 boosting rounds |
   | **Daily LSTM** | Price sequences (last 60 days) | Neural network with memory |
   | **Intraday LSTM** | 15-minute candle patterns (last 12 hours) | Short-term neural network |

4. **Each model outputs a probability** (0–100%) that the stock will go **UP** tomorrow

5. **Ranks all 10 stocks** by average probability across all 4 models

6. **Picks the Top 3** (configurable) and saves the ranking to `data/daily_ranking.csv`

### Result:
> ✅ 4 trained models per stock + a ranking of which stocks look best today.

---

## Phase 2: Sniper (Live Trading) — `sniper.py`

> 🕘 **Runs at 9:20 AM IST** (just after market opens) and monitors until 3:15 PM

### What it does:

1. **Loads the Top 3 stocks** from Scholar's ranking

2. **Loads all 4 trained models** for each of those stocks

3. **Enters a monitoring loop** (checks every few minutes):

   For each stock:
   - Fetches the **latest live price** from Yahoo Finance
   - Feeds current features into all 4 models
   - Each model votes: **BUY** or **WAIT**
   - Needs **3-out-of-4 models to agree** before buying (consensus voting)

4. **When 3+ models say BUY:**
   - 🔫 Fires a **"bullet"** (1/5th of your capital = ₹200)
   - 🛑 Sets an **ATR-based stop loss** (1.5× ATR below entry) — auto-exit if price drops
   - 🎯 Sets an **ATR-based target** (3.0× ATR above entry) — auto-exit if price hits profit target
   - ⏳ Waits **10 minutes** before the next bullet (diversification)
   - 🔄 Distributes bullets **round-robin** across the top 3 stocks (not all on one stock)

5. **Risk Management (always running):**
   - If any position hits stop loss → **auto-sells** (limits damage)
   - If any position hits target → **auto-sells** (locks profit)
   - If total day loss exceeds **-5%** → **KILL SWITCH** activates, closes everything
   - At **3:15 PM** → force-closes all remaining positions

6. **Saves state** to files after every action:
   - `data/daily_state.json` — active trades, bullets used, P&L
   - `data/dashboard_state.json` — summary for the dashboard
   - `data/trade_history.csv` — permanent log of all trades

### Result:
> ✅ Paper trades executed, P&L tracked, all logged.

---

## Phase 3: Dashboard (Monitoring) — `dashboard.py`

> 🖥️ **Run locally anytime:** `streamlit run src/dashboard.py`

### Shows you in real-time (refreshes every 30 seconds):

| Section | What You See |
|---------|-------------|
| 📡 **Live Stock Prices** | Current price + % change for top stocks (from Yahoo Finance) |
| 🔫 **Sniper Status** | Today's P&L, win rate, number of trades, open positions |
| 💰 **Open Position P&L** | Live unrealised profit/loss (current price vs entry price) |
| 📊 **Stock Rankings** | Which stocks the AI picked today and their model probabilities |
| 📋 **Trade History** | Cumulative P&L chart, per-stock breakdown, last 50 trades |

### Privacy:
> 🔒 **Everything runs 100% on your PC** — no data leaves your machine. No cloud. No sharing.

---

## Phase 4: Backtest (Validation) — `backtest.py`

> 📅 **Runs weekly (Saturday)** to check if the AI is actually working

### What it does:

1. Takes the last several months of historical data
2. Uses **walk-forward validation** (train on past → test on future, sliding window)
3. Tests **RF + XGBoost** on each stock
4. Simulates trades with the same rules as the Sniper
5. Reports: **accuracy, precision, simulated P&L**

### Purpose:
> If the backtest shows poor results, you know to adjust parameters before risking more capital.

---

## Daily Timeline

```
 ┌─────────────────────────────────────────────────────────┐
 │  8:00 AM   →  Scholar runs (trains AI, ranks stocks)    │
 │  9:15 AM   →  Market opens                              │
 │  9:20 AM   →  Sniper starts (monitors top 3 stocks)     │
 │  9:20-3:15 →  Sniper loop (buy/sell on 4-model vote)    │
 │  3:15 PM   →  Force-close all positions, save P&L       │
 │  3:30 PM   →  Market closes                             │
 │  Saturday  →  Backtest runs (weekly validation)          │
 └─────────────────────────────────────────────────────────┘
```

---

## Money Flow (with ₹1,000)

```
 ₹1,000 capital
     │
     ▼
 Split into 5 bullets (₹200 each)
     │
     ▼
 Round-robin across top 3 stocks
     │
     ▼
 Each trade: risk = 1.5×ATR, reward = 3.0×ATR (2:1 ratio)
     │
     ▼
 Target: +2% per day = ₹20/day
     │
     ▼
 Kill switch: if -5% (₹50 loss) → stop everything
```

---

## Where It All Runs (Free)

**GitHub Actions** provides free compute for public repos. Two workflows:

| Workflow | Schedule | What It Does |
|----------|----------|-------------|
| `aegis_protocol.yml` | Daily 8:00 AM + 9:20 AM IST | Runs Scholar then Sniper |
| `weekly_backtest.yml` | Saturday | Runs Backtest validation |

> Your stock list and settings are stored as **GitHub Secrets** (not in code), so even though the repo is public, your configuration is **private**.

---

## GitHub Secrets (One-Time Setup)

Go to: **https://github.com/Haik11kashiyani/project-aegis/settings/secrets/actions**

Click **"New repository secret"** for each:

| Secret Name | Value | What It Does |
|-------------|-------|-------------|
| `STOCK_WATCHLIST` | `TATASTEEL.NS,SBIN.NS,RELIANCE.NS,HDFCBANK.NS,ICICIBANK.NS,NTPC.NS,POWERGRID.NS,COALINDIA.NS,INFY.NS,TCS.NS` | 10 Indian stocks to scan |
| `TOP_N_STOCKS` | `3` | Trade only best 3 each day |
| `CAPITAL` | `1000` | Starting paper money (₹) |
| `DAILY_TARGET` | `0.02` | 2% daily profit target |
| `MAX_BULLETS` | `5` | Split capital into 5 shots |
| `TARGET_STOCK` | `TATASTEEL.NS` | Fallback stock |

### Stock List Explained:

| Ticker | Company | Sector |
|--------|---------|--------|
| TATASTEEL.NS | Tata Steel | Metal |
| SBIN.NS | State Bank of India | Banking |
| RELIANCE.NS | Reliance Industries | Conglomerate |
| HDFCBANK.NS | HDFC Bank | Banking |
| ICICIBANK.NS | ICICI Bank | Banking |
| NTPC.NS | NTPC Ltd | Power |
| POWERGRID.NS | Power Grid Corp | Power |
| COALINDIA.NS | Coal India | Mining |
| INFY.NS | Infosys | IT |
| TCS.NS | TCS | IT |

---

## How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the AI (Scholar)
```bash
python src/scholar.py
```

### 3. Start the Sniper
```bash
python src/sniper.py
```

### 4. View the Dashboard
```bash
streamlit run src/dashboard.py
```
Opens at **http://localhost:8501** — auto-refreshes every 30 seconds with live prices.

### 5. Run Backtest
```bash
python src/backtest.py
```

---

## Project Files

```
project-aegis/
├── src/
│   ├── config.py        ← All settings (reads from GitHub Secrets)
│   ├── scholar.py       ← Phase 1: Train 4 AI models, rank stocks
│   ├── sniper.py        ← Phase 2: Live paper trading with consensus voting
│   ├── backtest.py      ← Phase 4: Weekly validation
│   ├── dashboard.py     ← Phase 3: Live Streamlit dashboard
│   └── sentiment.py     ← News sentiment scoring
├── .github/workflows/
│   ├── aegis_protocol.yml    ← Daily automation
│   └── weekly_backtest.yml   ← Saturday backtest
├── requirements.txt     ← Python dependencies
├── README.md            ← Project overview
├── PROJECT_GUIDE.md     ← This file (full guide)
└── .gitignore           ← Files to ignore in git
```

---

## ⚠️ Important Notes

- This is **paper trading only** — no real money is involved
- Past AI performance does NOT guarantee future results
- The 2% daily target is ambitious — real results will vary
- Always run the backtest to validate before trusting the AI
- The kill switch (-5%) protects against catastrophic days
- All data stays on your PC / GitHub Actions — nothing is shared externally

---

*Project Aegis v2.1 — Built with ❤️ for learning AI-powered trading*
