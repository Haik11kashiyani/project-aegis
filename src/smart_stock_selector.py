import os
import json
import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SmartStockSelector:
    """
    Dynamically selects the best stocks to trade based on:
    - Available capital (filters out stocks too expensive)
    - Liquidity (minimum average daily volume)
    - Volatility (ATR as % of price — sweet spot for algo trading)
    - Sector diversification (max 2 stocks per sector)
    - Price range (affordable enough for position sizing)
    """
    
    # Universe of 30+ liquid NSE stocks to scan
    STOCK_UNIVERSE = [
        # Metal
        ('TATASTEEL.NS', 'Metal'), ('HINDALCO.NS', 'Metal'), ('JSWSTEEL.NS', 'Metal'),
        # Banking
        ('SBIN.NS', 'Banking'), ('BANKBARODA.NS', 'Banking'), ('PNB.NS', 'Banking'),
        # Power/Energy  
        ('NTPC.NS', 'Power'), ('POWERGRID.NS', 'Power'), ('TATAPOWER.NS', 'Power'), ('NHPC.NS', 'Power'),
        # Oil & Gas
        ('ONGC.NS', 'Oil&Gas'), ('BPCL.NS', 'Oil&Gas'), ('GAIL.NS', 'Oil&Gas'),
        # Mining
        ('COALINDIA.NS', 'Mining'), ('NMDC.NS', 'Mining'),
        # FMCG
        ('ITC.NS', 'FMCG'), ('HINDUNILVR.NS', 'FMCG'),
        # PSU
        ('IRFC.NS', 'PSU'), ('IRCTC.NS', 'PSU'), ('BEL.NS', 'PSU'),
        # Auto
        ('TATAMOTORS.NS', 'Auto'), ('M&M.NS', 'Auto'), ('MARUTI.NS', 'Auto'),
        # ETFs
        ('NIFTYBEES.NS', 'ETF'), ('BANKBEES.NS', 'ETF'),
        # Infra
        ('ADANIPORTS.NS', 'Infra'), ('LT.NS', 'Infra'),
        # Pharma
        ('SUNPHARMA.NS', 'Pharma'), ('CIPLA.NS', 'Pharma'),
        # Telecom
        ('BHARTIARTL.NS', 'Telecom'),
    ]
    
    def __init__(self):
        self.cache_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "stock_selection_cache.json"
        )
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    def _fetch_stock_data(self):
        """Fetch data from yfinance and cache it for 24 hours."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                cache_time = datetime.fromisoformat(cache.get("timestamp"))
                if datetime.now() - cache_time < timedelta(hours=24):
                    return cache.get("data", [])
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

        stock_data = []
        for symbol, sector in self.STOCK_UNIVERSE:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
                if hist.empty or len(hist) < 20:
                    continue

                current_price = hist['Close'].iloc[-1]
                if current_price <= 0:
                    continue
                avg_volume = hist['Volume'].tail(20).mean()
                
                # Calculate ATR (14 period)
                high = hist['High']
                low = hist['Low']
                close = hist['Close']
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                atr_pct = atr / current_price

                stock_data.append({
                    "symbol": symbol,
                    "sector": sector,
                    "price": current_price,
                    "avg_volume": avg_volume,
                    "atr_pct": atr_pct
                })
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")

        # Save to cache
        try:
            with open(self.cache_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "data": stock_data
                }, f)
        except Exception as e:
            logger.error(f"Failed to write cache: {e}")

        return stock_data

    def _filter_by_price(self, capital, stocks) -> list:
        """Remove stocks where 1 share > capital/3"""
        max_price = capital / 3
        return [s for s in stocks if s["price"] <= max_price]
        
    def _filter_by_liquidity(self, stocks, min_volume=500000) -> list:
        """Remove stocks with avg volume below threshold"""
        return [s for s in stocks if s["avg_volume"] >= min_volume]
        
    def _score_by_volatility(self, stocks) -> list:
        """Score stocks by ATR% — want 1.5% to 4% daily range"""
        scored = []
        for s in stocks:
            atr = s["atr_pct"]
            score = 0
            if 0.015 <= atr <= 0.04:
                score = 100
            elif 0.01 <= atr < 0.015:
                score = 70
            elif 0.04 < atr <= 0.06:
                score = 50
            else:
                score = 0
            
            s["volatility_score"] = score
            scored.append(s)
        return sorted(scored, key=lambda x: x["volatility_score"], reverse=True)
        
    def _enforce_sector_diversity(self, stocks, max_per_sector=2) -> list:
        """Limit to max 2 stocks per sector"""
        sector_counts = {}
        diverse = []
        for s in stocks:
            sector = s["sector"]
            if sector_counts.get(sector, 0) < max_per_sector:
                diverse.append(s)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return diverse
        
    def _rank_and_select(self, stocks, max_stocks) -> list:
        """Final ranking by composite score"""
        ranked = sorted(stocks, key=lambda x: x.get("volatility_score", 0), reverse=True)
        final_selection = ranked[:max_stocks]
        return [(s["symbol"], s["sector"], s.get("volatility_score", 0), s["price"]) for s in final_selection]

    def select_stocks(self, capital, max_stocks=10) -> list:
        """
        Select top stocks for today based on capital and market conditions.
        Returns list of (symbol, sector, score, price) tuples.
        """
        data = self._fetch_stock_data()
        
        filtered = self._filter_by_price(capital, data)
        filtered = self._filter_by_liquidity(filtered)
        scored = self._score_by_volatility(filtered)
        diverse = self._enforce_sector_diversity(scored)
        
        return self._rank_and_select(diverse, max_stocks)
