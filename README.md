# PROJECT AEGIS v2 -- AI Paper-Trading Bot

> **Automated Indian Stock Paper-Trading System** using a 4-Model Ensemble (Random Forest + XGBoost + Daily LSTM + Intraday LSTM) with ATR-based risk management, multi-stock scanning, sentiment analysis, and a Streamlit dashboard. Runs entirely on GitHub Actions for free.

---

## Architecture -- "The Quad Engine"

| Component | Role | Schedule |
|-----------|------|----------|
| **Scholar v2** | Scans 10 stocks, trains 4 models each, ranks by AI probability | Pre-market (8:45 AM IST) |
| **Sniper v2** | Paper-trades top-N stocks with 4-model ensemble voting | 9:15 AM - 3:15 PM IST |
| **Backtester v2** | Walk-forward validation for RF + XGBoost across all stocks | Weekly (Sunday midnight) |
| **Dashboard** | Streamlit web app showing live P&L, rankings, trade history | Always-on (Streamlit Cloud) |

`
Scholar v2 (Night)                    Sniper v2 (Day)
+----------------------+              +--------------------------+
| Scan 10 stocks       |              | Load top-3 ranked stocks |
| Engineer 12 features |              | Load 4 models per stock  |
| + Sentiment Score    |              |                          |
| Train per stock:     | --> models/  | For each stock:          |
|   Random Forest      |              |   RF vote (>75%?)        |
|   XGBoost            |              |   XGB vote (>75%?)       |
|   Daily LSTM (60d)   |              |   LSTM vote (>55%?)      |
|   Intraday LSTM(15m) |              |   Intraday vote (>55%?)  |
| Rank by avg prob     |              |                          |
| Pick top-3           | --> ranking  | BUY if 3/4 agree         |
| Self-correct errors  |              | ATR trailing stop        |
+----------------------+              | 10-min diversity gap     |
                                      | 2% daily target          |
                                      | 5% max-loss killswitch   |
                                      +--------------------------+
`

---

## Features

### 4-Model Ensemble
1. **Random Forest** (200 trees, depth 12, min_split 50)
2. **XGBoost** (200 rounds, depth 6, LR 0.05, subsample 0.8)
3. **Daily LSTM** (60-day lookback, 2-layer 64-unit, 25% dropout)
4. **Intraday LSTM** (48x15-min candles, 2-layer 48-unit, 20% dropout)

**Consensus voting**: 3-out-of-4 models must agree to fire a bullet.

### Multi-Stock Scanning
- Watchlist of 10 Indian stocks (configurable via secrets)
- Scholar trains all 10, ranks by average model probability
- Sniper trades only the top-N (default: 3)
- Round-robin bullet allocation across top stocks

### 12 Technical Features + Sentiment

| # | Feature | Purpose |
|---|---------|---------|
| 1 | RSI (14) | Momentum / overbought-oversold |
| 2 | SMA 50 | Medium-term trend |
| 3 | SMA 200 | Long-term trend |
| 4 | EMA 20 | Short-term trend |
| 5 | ATR (14) | Volatility (dynamic stop-loss) |
| 6 | MACD | Trend momentum |
| 7 | MACD Signal | Crossover signals |
| 8 | BB Upper | Resistance band |
| 9 | BB Lower | Support band |
| 10| Volume Ratio | Volume vs 20-day avg |
| 11| OBV | Accumulation/distribution |
| 12| **Sentiment Score** | News sentiment via GoogleNews + TextBlob |

### Risk Management
- **ATR Trailing Stop Loss**: Stop = Entry - 1.5x ATR
- **ATR Target**: Target = Entry + 3.0x ATR (1:2 risk-reward)
- **Bullet System**: Capital split into 5 equal parts
- **Time Diversity**: 10-minute minimum gap between trades
- **Daily Kill-Switch**: Stops trading if loss > 5% of capital
- **Force Close**: All positions closed 20 min before market close

### Streamlit Dashboard
- Real-time P&L tracking and win rate
- Stock ranking table with per-model probabilities
- Cumulative P&L chart over time
- Active positions monitor
- Filter by stock
- Free hosting on Streamlit Community Cloud

---

## Project Structure

`
Project-Aegis/
+-- .gitignore                 # Privacy shield
+-- requirements.txt           # Python dependencies (RF, XGB, TF, Streamlit)
+-- .github/workflows/
|   +-- aegis_protocol.yml     # Daily: Scholar -> Sniper pipeline
|   +-- weekly_backtest.yml    # Sunday: RF+XGB validation
+-- src/
    +-- config.py              # Central config (env vars + defaults)
    +-- scholar.py             # 4-model training engine
    +-- sniper.py              # Live paper-trading with ensemble voting
    +-- backtest.py            # Walk-forward backtest (RF + XGB)
    +-- sentiment.py           # GoogleNews + TextBlob sentiment
    +-- dashboard.py           # Streamlit web dashboard
`

---

## Setup (5 Minutes)

### 1. Create Public GitHub Repo
Public repos get **unlimited** free GitHub Actions minutes.

### 2. Push This Code
`ash
git init
git add .
git commit -m "Initial commit - Project Aegis v2"
git remote add origin https://github.com/YOUR_USERNAME/project-aegis.git
git push -u origin main
`

### 3. Add GitHub Secrets
Go to **Settings -> Secrets and variables -> Actions -> New repository secret**:

| Secret Name | Example Value | Description |
|-------------|---------------|-------------|
| `STOCK_WATCHLIST` | `TATASTEEL.NS,SBIN.NS,RELIANCE.NS,...` | Comma-separated tickers |
| `TOP_N_STOCKS` | `3` | Trade only top-N ranked stocks |
| `DAILY_TARGET` | `0.02` | 2% daily profit target |
| `MAX_BULLETS` | `5` | Number of position splits |
| `TIME_GAP` | `600` | Seconds between trades |
| `CAPITAL` | `1000` | Starting capital in Rs. |

### 4. Enable Workflow Permissions
**Settings -> Actions -> General -> Workflow permissions -> Read and write permissions**

### 5. Deploy Dashboard (Free)
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set main file path: `src/dashboard.py`
4. Deploy -- your dashboard is now live!

### 6. Test Manually
Go to **Actions -> Aegis Protocol -> Run workflow** to trigger a test run.

---

## Running Locally

`ash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run backtest first
python src/backtest.py

# Train models (scans all 10 stocks)
python src/scholar.py

# Run paper-trading (during market hours)
python src/sniper.py

# Launch dashboard
streamlit run src/dashboard.py
`

---

## Security and Privacy

| Threat | Protection |
|--------|------------|
| Someone copies your repo | All trading params in **GitHub Secrets** |
| Trade logs exposed | `.gitignore` blocks all `data/`, `models/`, `.csv`, `.pkl`, `.h5` |
| Model files stolen | Passed via **GitHub Artifacts** (encrypted, 7-day TTL) |
| Strategy reverse-engineered | Feature names visible, thresholds hidden in Secrets |

---

## Disclaimer

> This project is for **educational and paper-trading purposes only**.
> It does NOT place real trades or connect to any broker API.
> There is **no guarantee of profit**. Stock trading involves risk.
> Past performance does not indicate future results.
> Always consult a SEBI-registered advisor before investing real money.
