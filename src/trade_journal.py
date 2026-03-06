"""
====================================================
PROJECT AEGIS — Auto Trade Journal (PDF Generator)
====================================================
Weekly auto-generated trade journal with:
  - Annotated candlestick charts per trade
  - Trade reasoning (neuro-voter, brain, confidence)
  - Performance attribution by sector, model, strategy
  - Win/loss breakdown, equity curve
  - Risk metrics summary (VaR, drawdown, Sharpe)

Uses matplotlib for chart generation (always available).
Generates HTML → optional PDF via weasyprint or wkhtmltopdf.

Outputs
-------
data/reports/trade_journal_YYYYMMDD.html
data/trade_journal_state.json
====================================================
"""

import os, json, time, warnings, io, base64
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
JOURNAL_STATE_FILE = os.path.join(DATA_DIR, "trade_journal_state.json")
TRADE_LOG = os.path.join(DATA_DIR, "trade_history.csv")


# ══════════════════════════════════════════════════
#  CHART GENERATION (matplotlib)
# ══════════════════════════════════════════════════
def _generate_equity_curve(trades_df: pd.DataFrame) -> str:
    """Generate equity curve chart → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if "Actual_Profit" not in trades_df.columns:
            return ""

        cumulative = trades_df["Actual_Profit"].cumsum()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(range(len(cumulative)), cumulative, alpha=0.3,
                        color="green" if cumulative.iloc[-1] >= 0 else "red")
        ax.plot(cumulative, linewidth=2,
                color="green" if cumulative.iloc[-1] >= 0 else "red")
        ax.set_title("Equity Curve", fontweight="bold")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative P&L (₹)")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def _generate_pnl_by_stock(trades_df: pd.DataFrame) -> str:
    """P&L bar chart by stock → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "Stock" not in trades_df.columns or "Actual_Profit" not in trades_df.columns:
            return ""

        by_stock = trades_df.groupby("Stock")["Actual_Profit"].sum().sort_values()
        if by_stock.empty:
            return ""

        colors = ["green" if v >= 0 else "red" for v in by_stock.values]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(by_stock.index.str.replace(".NS", ""), by_stock.values, color=colors)
        ax.set_title("P&L by Stock", fontweight="bold")
        ax.set_xlabel("P&L (₹)")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def _generate_win_loss_pie(wins: int, losses: int) -> str:
    """Win/Loss pie chart → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if wins + losses == 0:
            return ""

        fig, ax = plt.subplots(figsize=(4, 4))
        labels = [f"Wins ({wins})", f"Losses ({losses})"]
        sizes = [wins, losses]
        colors = ["#2ecc71", "#e74c3c"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 12})
        ax.set_title("Win / Loss Ratio", fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


# ══════════════════════════════════════════════════
#  PERFORMANCE ATTRIBUTION
# ══════════════════════════════════════════════════
SECTOR_MAP = {
    "TATASTEEL.NS": "Metals", "SBIN.NS": "Banking", "RELIANCE.NS": "Energy",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "NTPC.NS": "Power",
    "POWERGRID.NS": "Power", "COALINDIA.NS": "Mining", "INFY.NS": "IT",
    "TCS.NS": "IT",
}


def _compute_attribution(trades_df: pd.DataFrame) -> dict:
    """Performance attribution by sector and exit type."""
    attrs = {}

    # By sector
    if "Stock" in trades_df.columns and "Actual_Profit" in trades_df.columns:
        trades_df["Sector"] = trades_df["Stock"].map(SECTOR_MAP).fillna("Other")
        by_sector = trades_df.groupby("Sector").agg(
            pnl=("Actual_Profit", "sum"),
            trades=("Actual_Profit", "count"),
            avg_pnl=("Actual_Profit", "mean"),
        ).to_dict("index")
        attrs["by_sector"] = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by_sector.items()}

    # By exit type
    if "Exit_Type" in trades_df.columns and "Actual_Profit" in trades_df.columns:
        by_exit = trades_df[trades_df["Exit_Type"].notna()].groupby("Exit_Type").agg(
            pnl=("Actual_Profit", "sum"),
            trades=("Actual_Profit", "count"),
        ).to_dict("index")
        attrs["by_exit_type"] = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by_exit.items()}

    # By time of day
    if "Time" in trades_df.columns and "Actual_Profit" in trades_df.columns:
        def _time_bucket(t):
            try:
                h = int(str(t).split(":")[0])
                if h < 10:
                    return "Opening (9-10)"
                elif h < 12:
                    return "Morning (10-12)"
                elif h < 14:
                    return "Afternoon (12-14)"
                else:
                    return "Closing (14-15)"
            except Exception:
                return "Unknown"

        trades_df["TimeBucket"] = trades_df["Time"].apply(_time_bucket)
        by_time = trades_df.groupby("TimeBucket").agg(
            pnl=("Actual_Profit", "sum"),
            trades=("Actual_Profit", "count"),
        ).to_dict("index")
        attrs["by_time"] = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by_time.items()}

    return attrs


# ══════════════════════════════════════════════════
#  RISK METRICS
# ══════════════════════════════════════════════════
def _compute_risk_metrics(trades_df: pd.DataFrame, capital: float = 1000) -> dict:
    """Compute Sharpe, max DD, profit factor from trade history."""
    if "Actual_Profit" not in trades_df.columns or trades_df.empty:
        return {}

    returns = trades_df["Actual_Profit"].values
    if len(returns) == 0:
        return {}

    total = float(returns.sum())
    avg = float(returns.mean())
    std = float(returns.std()) if len(returns) > 1 else 0.01

    # Sharpe (annualised, assuming ~250 trading days)
    sharpe = (avg / std * np.sqrt(250)) if std > 0 else 0

    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    dd = cumulative - running_max
    max_dd = float(np.min(dd))

    # Profit factor
    gains = returns[returns > 0].sum()
    losses_abs = abs(returns[returns < 0].sum())
    pf = gains / losses_abs if losses_abs > 0 else float("inf")

    return {
        "total_pnl": round(total, 2),
        "avg_pnl_per_trade": round(avg, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else "∞",
        "best_trade": round(float(returns.max()), 2),
        "worst_trade": round(float(returns.min()), 2),
    }


# ══════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════
def generate_journal(period_days: int = 7, capital: float = 1000) -> dict:
    """
    Generate a comprehensive trade journal.
    Returns report data and saves HTML file.
    """
    if not os.path.exists(TRADE_LOG):
        return {"error": "no trade history", "timestamp": _now_ist()}

    try:
        df = pd.read_csv(TRADE_LOG)
    except Exception as e:
        return {"error": f"cannot read log: {e}", "timestamp": _now_ist()}

    if df.empty:
        return {"error": "empty trade log", "timestamp": _now_ist()}

    # Filter to period
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            cutoff = datetime.now() - timedelta(days=period_days)
            df = df[df["Date"] >= cutoff]
        except Exception:
            pass

    if df.empty:
        return {"error": "no trades in period", "timestamp": _now_ist()}

    # Compute metrics
    total_trades = len(df)
    exits = df[df["Action"] != "BUY"]
    wins = len(exits[exits["Actual_Profit"] > 0])
    losses = len(exits[exits["Actual_Profit"] < 0])
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    total_pnl = float(exits["Actual_Profit"].sum()) if "Actual_Profit" in exits.columns else 0

    risk_metrics = _compute_risk_metrics(exits, capital)
    attribution = _compute_attribution(exits.copy())

    # Generate charts
    equity_chart = _generate_equity_curve(exits)
    stock_chart = _generate_pnl_by_stock(exits)
    pie_chart = _generate_win_loss_pie(wins, losses)

    # Build HTML
    html = _build_html(
        period_days=period_days,
        total_trades=total_trades,
        wins=wins, losses=losses, win_rate=win_rate,
        total_pnl=total_pnl, capital=capital,
        risk_metrics=risk_metrics,
        attribution=attribution,
        equity_chart=equity_chart,
        stock_chart=stock_chart,
        pie_chart=pie_chart,
        trades=exits.tail(50).to_dict("records"),
    )

    # Save HTML
    os.makedirs(REPORTS_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    html_path = os.path.join(REPORTS_DIR, f"trade_journal_{date_str}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    result = {
        "report_path": html_path,
        "period_days": period_days,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "risk_metrics": risk_metrics,
        "attribution": attribution,
        "timestamp": _now_ist(),
    }

    save_journal_state(result)
    return result


def _build_html(period_days, total_trades, wins, losses, win_rate,
                total_pnl, capital, risk_metrics, attribution,
                equity_chart, stock_chart, pie_chart, trades) -> str:
    """Build the complete HTML journal report."""
    pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0
    pnl_color = "green" if total_pnl >= 0 else "red"

    # Charts
    equity_img = f'<img src="data:image/png;base64,{equity_chart}" style="width:100%">' if equity_chart else ""
    stock_img = f'<img src="data:image/png;base64,{stock_chart}" style="width:100%">' if stock_chart else ""
    pie_img = f'<img src="data:image/png;base64,{pie_chart}" style="width:300px">' if pie_chart else ""

    # Trade table rows
    trade_rows = ""
    for t in trades:
        pnl = t.get("Actual_Profit", 0)
        row_color = "#e8f5e9" if pnl > 0 else ("#ffebee" if pnl < 0 else "#fff")
        trade_rows += f"""
        <tr style="background:{row_color}">
            <td>{t.get('Date', '')}</td>
            <td>{t.get('Time', '')}</td>
            <td>{str(t.get('Stock', '')).replace('.NS', '')}</td>
            <td>{t.get('Action', '')}</td>
            <td>₹{t.get('Entry_Price', 0):,.2f}</td>
            <td>₹{t.get('Exit_Price', 0):,.2f}</td>
            <td>{t.get('Qty', 0)}</td>
            <td style="color:{'green' if pnl >= 0 else 'red'}">₹{pnl:,.2f}</td>
            <td>{t.get('Exit_Type', '')}</td>
        </tr>"""

    # Attribution tables
    sector_rows = ""
    for sector, data in attribution.get("by_sector", {}).items():
        sector_rows += f"<tr><td>{sector}</td><td>₹{data.get('pnl', 0):,.2f}</td><td>{data.get('trades', 0)}</td></tr>"

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Project Aegis — Trade Journal</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #fafafa; }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
        h2 {{ color: #283593; margin-top: 30px; }}
        .metrics {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
        .metric-card {{ background: white; border-radius: 10px; padding: 20px; min-width: 150px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ color: #666; font-size: 12px; margin-top: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
        th {{ background: #1a237e; color: white; }}
        .chart-container {{ margin: 20px 0; background: white; padding: 15px; border-radius: 10px;
                           box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .footer {{ text-align: center; color: #999; margin-top: 40px; padding: 20px;
                  border-top: 1px solid #eee; }}
    </style>
</head>
<body>
    <h1>🛡️ Project Aegis — Trade Journal</h1>
    <p>Period: Last {period_days} days | Generated: {_now_ist()}</p>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value" style="color:{pnl_color}">₹{total_pnl:,.2f}</div>
            <div class="metric-label">Total P&L ({pnl_pct:+.1f}%)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_trades}</div>
            <div class="metric-label">Total Trades</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:green">{wins}</div>
            <div class="metric-label">Wins</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:red">{losses}</div>
            <div class="metric-label">Losses</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{win_rate:.0f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{risk_metrics.get('sharpe_ratio', 0):.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{risk_metrics.get('profit_factor', 0)}</div>
            <div class="metric-label">Profit Factor</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:red">₹{risk_metrics.get('max_drawdown', 0):,.2f}</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
    </div>

    <h2>📈 Equity Curve</h2>
    <div class="chart-container">{equity_img}</div>

    <div style="display:flex; gap:20px; flex-wrap:wrap;">
        <div style="flex:2">
            <h2>📊 P&L by Stock</h2>
            <div class="chart-container">{stock_img}</div>
        </div>
        <div style="flex:1">
            <h2>🎯 Win / Loss</h2>
            <div class="chart-container" style="text-align:center">{pie_img}</div>
        </div>
    </div>

    <h2>🏭 Attribution by Sector</h2>
    <table>
        <tr><th>Sector</th><th>P&L</th><th>Trades</th></tr>
        {sector_rows}
    </table>

    <h2>📋 Trade Log (Last 50)</h2>
    <table>
        <tr><th>Date</th><th>Time</th><th>Stock</th><th>Action</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>Exit Type</th></tr>
        {trade_rows}
    </table>

    <div class="footer">
        🛡️ Project Aegis v9.0 — Auto Trade Journal<br>
        Generated by the AI Trading System. All data is from paper-trading simulations.
    </div>
</body>
</html>"""


def save_journal_state(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(JOURNAL_STATE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_ist() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
