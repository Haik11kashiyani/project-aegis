"""
====================================================
PROJECT AEGIS — PDF Report Exporter v1.0
====================================================
Generates a professional daily PDF report with:
  - P&L summary & equity curve
  - Trade table with entry/exit details
  - Voter accuracy & decision breakdown
  - Market breadth & sector rotation summary
  - Portfolio correlation heatmap
  - AI system health status

Does NOT require heavy dependencies — uses only HTML → PDF
via a lightweight approach (generates styled HTML, optionally
converts to PDF if weasyprint/pdfkit is available, otherwise
saves as .html which opens nicely in any browser).

Output: data/reports/aegis_report_YYYY-MM-DD.html (or .pdf)
====================================================
"""

import os
import json
import pandas as pd
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")


def _load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _load_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _css() -> str:
    """Professional report CSS (dark mode)."""
    return """
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #0b0e1a; color: #e0e0e0;
            padding: 30px; max-width: 1200px; margin: auto;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            padding: 30px; border-radius: 15px; margin-bottom: 25px;
            border: 1px solid #1e3a5f; text-align: center;
        }
        .header h1 { font-size: 28px; color: #00ff88; margin-bottom: 5px; }
        .header p { color: #888; font-size: 14px; }
        .section {
            background: #111827; border: 1px solid #1e3a5f;
            border-radius: 12px; padding: 20px; margin-bottom: 20px;
        }
        .section h2 {
            color: #29b6f6; font-size: 18px; margin-bottom: 15px;
            border-bottom: 1px solid #1e3a5f; padding-bottom: 8px;
        }
        .metrics { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px; }
        .metric {
            flex: 1; min-width: 150px; background: #0d1117;
            padding: 15px; border-radius: 10px; text-align: center;
            border: 1px solid #333;
        }
        .metric .label { font-size: 11px; color: #888; margin-bottom: 5px; }
        .metric .value { font-size: 22px; font-weight: bold; }
        .green { color: #00ff88; }
        .red { color: #ff4444; }
        .yellow { color: #ffa726; }
        .blue { color: #29b6f6; }
        .purple { color: #ab47bc; }
        table {
            width: 100%; border-collapse: collapse;
            font-size: 13px; margin-top: 10px;
        }
        th {
            background: #1a1a2e; color: #29b6f6;
            padding: 10px 8px; text-align: left;
            border-bottom: 2px solid #1e3a5f;
        }
        td {
            padding: 8px; border-bottom: 1px solid #1a1a2e;
        }
        tr:hover { background: rgba(29, 78, 137, 0.15); }
        .badge {
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; font-weight: bold;
        }
        .badge-green { background: rgba(0,255,136,0.15); color: #00ff88; }
        .badge-red { background: rgba(255,68,68,0.15); color: #ff4444; }
        .badge-yellow { background: rgba(255,167,38,0.15); color: #ffa726; }
        .footer {
            text-align: center; color: #555; font-size: 11px;
            margin-top: 30px; padding: 15px;
        }
        @media print {
            body { background: white; color: #333; }
            .section { border-color: #ddd; background: #fafafa; }
            .header { background: #f0f0f0; }
            .header h1 { color: #1a73e8; }
        }
    </style>
    """


def generate_report(date_str: str = None) -> str:
    """
    Generate HTML report for a given date (default: today).
    Returns the file path of the created report.
    """
    if date_str is None:
        date_str = datetime.now(IST).strftime("%Y-%m-%d")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load all data
    trades = _load_csv(os.path.join(DATA_DIR, "trade_history.csv"))
    state = _load_json(os.path.join(DATA_DIR, "daily_state.json"))
    learner = _load_json(os.path.join(DATA_DIR, "learner_report.json"))
    voter_acc = _load_json(os.path.join(DATA_DIR, "voter_accuracy.json"))
    sector = _load_json(os.path.join(DATA_DIR, "sector_rotation.json"))
    breadth = _load_json(os.path.join(DATA_DIR, "market_breadth.json"))
    corr = _load_json(os.path.join(DATA_DIR, "correlation_matrix.json"))
    alerts = _load_json(os.path.join(DATA_DIR, "alert_history.json"))
    live = _load_json(os.path.join(DATA_DIR, "live_analysis.json"))

    # Filter trades for today
    today_trades = pd.DataFrame()
    if not trades.empty and "Date" in trades.columns:
        today_trades = trades[trades["Date"].astype(str) == date_str]

    # P&L stats
    total_pnl = 0
    total_trades = 0
    wins = 0
    losses = 0
    if not today_trades.empty and "Actual_Profit" in today_trades.columns:
        profits = pd.to_numeric(today_trades["Actual_Profit"], errors="coerce").fillna(0)
        total_pnl = profits.sum()
        total_trades = len(today_trades)
        wins = (profits > 0).sum()
        losses = (profits < 0).sum()

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl_color = "green" if total_pnl >= 0 else "red"

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Aegis — Daily Report {date_str}</title>
    {_css()}
</head>
<body>

<div class="header">
    <h1>🛡️ PROJECT AEGIS — Daily Report</h1>
    <p>{date_str} | Generated at {datetime.now(IST).strftime('%H:%M IST')} | AI Paper Trading System</p>
</div>

<!-- P&L Summary -->
<div class="section">
    <h2>💰 P&L Summary</h2>
    <div class="metrics">
        <div class="metric">
            <div class="label">Total P&L</div>
            <div class="value {pnl_color}">₹{total_pnl:+,.2f}</div>
        </div>
        <div class="metric">
            <div class="label">Total Trades</div>
            <div class="value blue">{total_trades}</div>
        </div>
        <div class="metric">
            <div class="label">Win Rate</div>
            <div class="value yellow">{win_rate:.0f}%</div>
        </div>
        <div class="metric">
            <div class="label">Wins / Losses</div>
            <div class="value"><span class="green">{wins}W</span> / <span class="red">{losses}L</span></div>
        </div>
        <div class="metric">
            <div class="label">Return on Capital</div>
            <div class="value {pnl_color}">{(total_pnl / 1000 * 100):+.2f}%</div>
        </div>
    </div>
</div>
"""

    # Trade Table
    if not today_trades.empty:
        html += """<div class="section"><h2>📋 Trade Details</h2><table><tr>"""
        cols = ["Time", "Stock", "Action", "Entry_Price", "Exit_Price", "Qty", "Actual_Profit", "Status"]
        available_cols = [c for c in cols if c in today_trades.columns]
        for c in available_cols:
            html += f"<th>{c}</th>"
        html += "</tr>"
        for _, row in today_trades.iterrows():
            html += "<tr>"
            for c in available_cols:
                val = row.get(c, "")
                if c == "Actual_Profit":
                    pnl = float(val) if val else 0
                    cls = "badge-green" if pnl > 0 else "badge-red"
                    html += f'<td><span class="badge {cls}">₹{pnl:+.2f}</span></td>'
                elif c == "Status":
                    html += f'<td><span class="badge badge-yellow">{val}</span></td>'
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>"
        html += "</table></div>"

    # Voter Accuracy
    if voter_acc:
        html += """<div class="section"><h2>🎯 NeuroVoter Accuracy</h2><div class="metrics">"""
        voters = voter_acc.get("voters", voter_acc)
        if isinstance(voters, dict):
            for vname, vdata in voters.items():
                if isinstance(vdata, dict):
                    acc = vdata.get("accuracy", 0)
                    acc_pct = acc * 100 if acc <= 1 else acc
                    color = "green" if acc_pct >= 60 else ("yellow" if acc_pct >= 45 else "red")
                    html += f"""<div class="metric">
                        <div class="label">{vname}</div>
                        <div class="value {color}">{acc_pct:.1f}%</div>
                    </div>"""
        html += "</div></div>"

    # Market Breadth
    if breadth:
        b = breadth.get("breadth", {})
        v = breadth.get("vix", {})
        html += f"""<div class="section"><h2>📊 Market Breadth</h2>
        <div class="metrics">
            <div class="metric"><div class="label">A/D Ratio</div><div class="value blue">{b.get('advance_decline_ratio', 'N/A')}</div></div>
            <div class="metric"><div class="label">% > SMA50</div><div class="value">{b.get('pct_above_sma50', 'N/A')}%</div></div>
            <div class="metric"><div class="label">Breadth Score</div><div class="value">{b.get('breadth_score', 'N/A')}/100</div></div>
            <div class="metric"><div class="label">VIX</div><div class="value">{v.get('vix', 'N/A')}</div></div>
            <div class="metric"><div class="label">Size Factor</div><div class="value">{breadth.get('position_size_factor', 'N/A')}</div></div>
        </div></div>"""

    # Sector Rotation
    if sector:
        sectors = sector.get("sectors", {})
        if sectors:
            html += """<div class="section"><h2>🔄 Sector Rotation</h2><table>
            <tr><th>Sector</th><th>5D Return</th><th>10D Return</th><th>Score</th><th>State</th></tr>"""
            sorted_secs = sorted(sectors.items(),
                                 key=lambda x: x[1].get("momentum_score", -99) if isinstance(x[1], dict) else -99,
                                 reverse=True)
            for sec_name, sec_data in sorted_secs:
                if isinstance(sec_data, dict) and "momentum_score" in sec_data:
                    state_cls = "badge-green" if sec_data["state"] in ("ROTATING_IN", "STRONG") else (
                        "badge-red" if sec_data["state"] == "ROTATING_OUT" else "badge-yellow")
                    html += f"""<tr>
                        <td><strong>{sec_name}</strong></td>
                        <td>{sec_data.get('ret_5d', 'N/A'):+.2f}%</td>
                        <td>{sec_data.get('ret_10d', 'N/A'):+.2f}%</td>
                        <td>{sec_data.get('momentum_score', 'N/A'):+.2f}</td>
                        <td><span class="badge {state_cls}">{sec_data['state']}</span></td>
                    </tr>"""
            html += "</table></div>"

    # Correlation
    if corr:
        conc = corr.get("concentration", {})
        high_pairs = corr.get("correlation", {}).get("high_pairs", [])
        html += f"""<div class="section"><h2>🔗 Portfolio Correlation</h2>
        <div class="metrics">
            <div class="metric"><div class="label">Concentration</div><div class="value">{conc.get('concentration_score', 'N/A')}/100</div></div>
            <div class="metric"><div class="label">Rating</div><div class="value blue">{conc.get('rating', 'N/A')}</div></div>
            <div class="metric"><div class="label">High-Corr Pairs</div><div class="value yellow">{len(high_pairs)}</div></div>
        </div>"""
        if high_pairs:
            html += "<table><tr><th>Stock 1</th><th>Stock 2</th><th>Correlation</th></tr>"
            for p in high_pairs[:5]:
                html += f"<tr><td>{p.get('stock1','').replace('.NS','')}</td><td>{p.get('stock2','').replace('.NS','')}</td><td>{p.get('correlation', 0):.3f}</td></tr>"
            html += "</table>"
        html += "</div>"

    # Recent Alerts
    if alerts:
        alert_list = alerts if isinstance(alerts, list) else alerts.get("alerts", [])
        recent = alert_list[-10:]
        if recent:
            html += """<div class="section"><h2>🔔 Recent Alerts</h2><table>
            <tr><th>Time</th><th>Type</th><th>Symbol</th><th>Message</th></tr>"""
            for a in reversed(recent):
                atype = a.get("type", "")
                ts = a.get("timestamp", "")[:19]
                sym = a.get("symbol", "")
                msg = a.get("message", "")[:80]
                html += f"<tr><td>{ts}</td><td><span class='badge badge-yellow'>{atype}</span></td><td>{sym}</td><td>{msg}</td></tr>"
            html += "</table></div>"

    # Footer
    html += f"""
<div class="footer">
    <p>🛡️ Project Aegis — AI Paper Trading System | Report generated automatically</p>
    <p>This is NOT financial advice. Paper trading only.</p>
</div>
</body></html>"""

    # Save
    report_path = os.path.join(REPORTS_DIR, f"aegis_report_{date_str}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[REPORT] Saved: {report_path}")

    # Try PDF conversion (optional)
    pdf_path = report_path.replace(".html", ".pdf")
    try:
        from weasyprint import HTML
        HTML(filename=report_path).write_pdf(pdf_path)
        print(f"[REPORT] PDF: {pdf_path}")
        return pdf_path
    except ImportError:
        pass

    try:
        import pdfkit
        pdfkit.from_file(report_path, pdf_path)
        print(f"[REPORT] PDF: {pdf_path}")
        return pdf_path
    except (ImportError, Exception):
        pass

    print("[REPORT] PDF libraries not available. HTML report created (opens in any browser).")
    return report_path


if __name__ == "__main__":
    path = generate_report()
    print(f"\nReport: {path}")
