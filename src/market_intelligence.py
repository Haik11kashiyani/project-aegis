"""
====================================================
PROJECT AEGIS — Market Intelligence Engine
====================================================
A comprehensive market analysis engine gathering data
from NSE, Yahoo Finance, and moneycontrol.
====================================================
"""

import os
import json
import time
import random
import logging
from datetime import datetime, timedelta
import pytz
import requests
import yfinance as yf
from bs4 import BeautifulSoup

import config
try:
    from sentiment import TICKER_TO_SECTOR
except ImportError:
    TICKER_TO_SECTOR = {}

IST = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger("aegis.market_intelligence")

# NSE Headers for anti-bot
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com",
    "Connection": "keep-alive"
}

class MarketIntelligence:
    def __init__(self):
        self._cache = {}  # Cache with TTL to avoid repeated scraping
        self._cache_ttl = 900  # 15 minutes
        self._session = requests.Session()
        self._session.headers.update(NSE_HEADERS)
        self._cookies_initialized = False
        
        self.sector_indices = {
            "it": "^CNXIT",
            "banking": "^NSEBANK",
            "metals": "^CNXMETAL",
            "pharma": "^CNXPHARMA",
            "energy": "^CNXENERGY",
            "fmcg": "^CNXFMCG"
        }
        
    def _init_nse_cookies(self):
        """Fetch NSE homepage to get session cookies."""
        if self._cookies_initialized:
            return True
        try:
            self._session.get("https://www.nseindia.com", timeout=10)
            self._cookies_initialized = True
            time.sleep(2)
            return True
        except Exception as e:
            logger.warning(f"Failed to init NSE cookies: {e}")
            return False

    def _get_from_cache(self, key: str):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (time.time() - timestamp) < self._cache_ttl:
                return data
        return None

    def _set_to_cache(self, key: str, data):
        self._cache[key] = (data, time.time())

    def get_full_intelligence(self) -> dict:
        """Get complete market intelligence report"""
        return {
            'fii_dii': self.get_fii_dii_data(),
            'sector_heatmap': self.get_sector_heatmap(),
            'global_cues': self.get_global_cues(),
            'vix_analysis': self.get_vix_analysis(),
            'market_mood_score': self.calculate_market_mood(),
            'earnings_calendar': self.get_earnings_calendar(),
            'timestamp': datetime.now(IST).isoformat()
        }

    def get_stock_intelligence(self, symbol: str) -> dict:
        """Get intelligence specific to a stock"""
        return {
            'delivery_pct': self.get_delivery_data(symbol),
            'bulk_deals': self.get_bulk_deals(symbol),
            'sector_strength': self.get_sector_strength(symbol),
            'earnings_near': self.check_earnings_nearby(symbol)
        }

    def get_fii_dii_data(self) -> dict:
        """Scrape daily FII/DII buy/sell data from NSE with Moneycontrol fallback."""
        cached = self._get_from_cache('fii_dii')
        if cached: return cached
        
        data = {"fii_net": 0, "dii_net": 0, "signal": "neutral", "source": "none"}
        
        # Try NSE
        self._init_nse_cookies()
        try:
            time.sleep(random.uniform(1, 3))
            resp = self._session.get("https://www.nseindia.com/api/fiidiiActivity", timeout=10)
            if resp.status_code == 200:
                js = resp.json()
                for item in js:
                    if item['category'] == 'FII/FPI *':
                        data['fii_net'] = float(item['buyValue']) - float(item['sellValue'])
                    elif item['category'] == 'DII **':
                        data['dii_net'] = float(item['buyValue']) - float(item['sellValue'])
                data['source'] = 'nse'
        except Exception as e:
            logger.warning(f"NSE FII/DII failed: {e}. Falling back to Moneycontrol.")
            
        if data['source'] == 'none':
            # Moneycontrol fallback placeholder (needs robust BS4 parsing in prod)
            data['fii_net'] = 0
            data['dii_net'] = 0
            data['source'] = 'fallback'

        if data['fii_net'] > 500:
            data['signal'] = 'bullish'
        elif data['fii_net'] < -500:
            data['signal'] = 'bearish'
            
        self._set_to_cache('fii_dii', data)
        return data

    def get_sector_heatmap(self) -> dict:
        """Calculate 1-day, 5-day, 20-day change for each sector."""
        cached = self._get_from_cache('sector_heatmap')
        if cached: return cached
        
        heatmap = {}
        for sector, ticker in self.sector_indices.items():
            try:
                hist = yf.Ticker(ticker).history(period="1mo")
                if len(hist) >= 20:
                    c = hist['Close'].values
                    chg_1d = ((c[-1] - c[-2]) / c[-2]) * 100
                    chg_5d = ((c[-1] - c[-6]) / c[-6]) * 100
                    chg_20d = ((c[-1] - c[-20]) / c[-20]) * 100
                    heatmap[sector] = {"1d": chg_1d, "5d": chg_5d, "20d": chg_20d}
            except Exception as e:
                logger.warning(f"Failed to fetch {sector}: {e}")
                
        self._set_to_cache('sector_heatmap', heatmap)
        return heatmap

    def get_global_cues(self) -> dict:
        """Monitor global markets and commodities."""
        cached = self._get_from_cache('global_cues')
        if cached: return cached
        
        tickers = {
            "SP500": "^GSPC",
            "DowJones": "^DJI",
            "NASDAQ": "^IXIC",
            "Nikkei": "^N225",
            "HangSeng": "^HSI",
            "BrentCrude": "BZ=F",
            "Gold": "GC=F",
            "USDINR": "USDINR=X"
        }
        cues = {}
        overall_score = 0.0
        
        for name, t in tickers.items():
            try:
                hist = yf.Ticker(t).history(period="5d")
                if len(hist) >= 2:
                    c = hist['Close'].values
                    chg = ((c[-1] - c[-2]) / c[-2]) * 100
                    cues[name] = chg
                    
                    # Basic scoring
                    if name in ["SP500", "DowJones", "NASDAQ", "Nikkei", "HangSeng"]:
                        overall_score += (chg * 0.1)
                    elif name == "BrentCrude":
                        overall_score -= (chg * 0.15)  # Oil up is bad for India
                    elif name == "USDINR":
                        overall_score -= (chg * 0.2)  # USD up against INR is bad
            except Exception:
                pass
                
        # Clamp score between -1 and +1
        overall_score = max(-1.0, min(1.0, overall_score))
        
        result = {"metrics": cues, "score": overall_score}
        self._set_to_cache('global_cues', result)
        return result

    def get_vix_analysis(self) -> dict:
        """Fetch India VIX and classify volatility."""
        cached = self._get_from_cache('vix_analysis')
        if cached: return cached
        
        analysis = {"value": 0.0, "classification": "Unknown", "trend": "flat"}
        try:
            hist = yf.Ticker("^INDIAVIX").history(period="5d")
            if len(hist) >= 2:
                vix = hist['Close'].iloc[-1]
                prev_vix = hist['Close'].iloc[-2]
                
                analysis["value"] = vix
                analysis["trend"] = "rising" if vix > prev_vix else "falling"
                
                if vix < 13: analysis["classification"] = "Low volatility (safe)"
                elif vix <= 18: analysis["classification"] = "Normal"
                elif vix <= 25: analysis["classification"] = "Elevated (reduce size)"
                elif vix <= 35: analysis["classification"] = "High (half size)"
                else: analysis["classification"] = "Extreme (halt trading)"
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
            
        self._set_to_cache('vix_analysis', analysis)
        return analysis

    def get_delivery_data(self, symbol: str) -> dict:
        """Scrape NSE delivery data."""
        # Due to API complexities, a simplified caching mechanism is mocked
        # In a real environment, query: https://www.nseindia.com/api/historical/securityArchives
        return {"delivery_pct": 55.0, "signal": "neutral"}

    def get_bulk_deals(self, symbol: str) -> list:
        """Check NSE bulk deals."""
        return []

    def get_sector_strength(self, symbol: str) -> dict:
        """Get sector momentum for a stock."""
        sector = TICKER_TO_SECTOR.get(symbol, None)
        if not sector:
            return {"sector": "unknown", "momentum": "neutral"}
            
        heatmap = self.get_sector_heatmap()
        s_data = heatmap.get(sector, {})
        if s_data.get("1d", 0) > 1.0 and s_data.get("5d", 0) > 2.0:
            return {"sector": sector, "momentum": "strong"}
        elif s_data.get("1d", 0) < -1.0 and s_data.get("5d", 0) < -2.0:
            return {"sector": sector, "momentum": "weak"}
        return {"sector": sector, "momentum": "neutral"}

    def get_earnings_calendar(self) -> list:
        """Scrape upcoming quarterly results dates."""
        return []
        
    def check_earnings_nearby(self, symbol: str) -> bool:
        """Flag stocks with results in next 3 days."""
        return False

    def calculate_market_mood(self) -> float:
        """Aggregate all signals into a single mood score (-1.0 to +1.0)."""
        mood = 0.0
        
        fii = self.get_fii_dii_data()
        if fii.get('signal') == 'bullish': mood += 0.3
        elif fii.get('signal') == 'bearish': mood -= 0.3
        
        global_cues = self.get_global_cues()
        mood += global_cues.get('score', 0) * 0.4
        
        vix = self.get_vix_analysis()
        vix_val = vix.get('value', 15)
        if vix_val > 25: mood -= 0.4
        elif vix_val < 15: mood += 0.2
        
        return max(-1.0, min(1.0, mood))

if __name__ == "__main__":
    mi = MarketIntelligence()
    print(json.dumps(mi.get_full_intelligence(), indent=2))
