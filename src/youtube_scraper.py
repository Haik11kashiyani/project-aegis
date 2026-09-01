"""
====================================================
PROJECT AEGIS — YouTube Sentiment Scraper
====================================================
Extract market sentiment from popular YouTube trading
channels via FREE RSS feeds.
====================================================
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
import feedparser

try:
    import finbert_sentiment
except ImportError:
    finbert_sentiment = None

logger = logging.getLogger("aegis.youtube_scraper")

class YouTubeSentimentScraper:
    # Popular Indian trading/finance channels
    CHANNELS = {
        'CA Rachana Ranade': 'UCe3qdG0A_gr-sEdat5y2twQ',
        'Pranjal Kamra': 'UCnQJ3p19_s043g7n4P01_qA',
        'Akshat Shrivastava': 'UCqW8jxh4tH1Z1sWPbkGWL4g',
        'Ankur Warikoo': 'UCRzYN32xtBf3Yxsx5BvJWJw',
        'Asset Yogi': 'UCW0j1003Vp4D-aC-S8tY19w',
        'Groww': 'UCw51rEDL5e51xosUuL0lTkw',
    }
    
    def __init__(self):
        self.rss_base_url = "https://www.youtube.com/feeds/videos.xml?channel_id="

    def get_recent_videos(self, hours=48) -> list:
        """Fetch recent video titles from all channels via RSS"""
        recent_videos = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        for name, channel_id in self.CHANNELS.items():
            feed_url = f"{self.rss_base_url}{channel_id}"
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    # Parse published time
                    pub_time = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%S%z")
                    if pub_time >= cutoff_time:
                        recent_videos.append({
                            'channel': name,
                            'title': entry.title,
                            'link': entry.link,
                            'published': entry.published
                        })
            except Exception as e:
                logger.error(f"Error fetching RSS for {name}: {e}")
                
        return recent_videos
        
    def analyze_sentiment(self, videos) -> dict:
        """Run sentiment analysis on video titles"""
        stock_mentions = {}
        overall_score = 0.0
        
        # Simple keywords
        bullish_words = ['buy', 'bull', 'rally', 'surge', 'multibagger', 'breakout', 'growth']
        bearish_words = ['sell', 'bear', 'crash', 'plunge', 'fall', 'bubble', 'warning']
        
        total_videos = len(videos)
        if not total_videos:
            return {"overall_score": 0.0, "stock_mentions": {}}

        for v in videos:
            title = v['title'].lower()
            score = 0
            
            # 1. Try FinBERT if available
            if finbert_sentiment:
                try:
                    fb_res = finbert_sentiment.analyse_text(title)
                    if fb_res.get('label') == 'positive':
                        score = fb_res.get('score', 0.5)
                    elif fb_res.get('label') == 'negative':
                        score = -fb_res.get('score', 0.5)
                except Exception:
                    pass
            
            # 2. Fallback to keywords if score is 0
            if score == 0:
                for bw in bullish_words:
                    if bw in title: score += 0.5
                for bw in bearish_words:
                    if bw in title: score -= 0.5
            
            v['sentiment_score'] = max(-1.0, min(1.0, score))
            overall_score += v['sentiment_score']
            
        return {
            "overall_score": overall_score / total_videos if total_videos > 0 else 0,
            "stock_mentions": stock_mentions  # Logic to map against watchlists could go here
        }

    def get_youtube_mood(self) -> dict:
        """Get aggregate YouTube sentiment"""
        videos = self.get_recent_videos(hours=48)
        analysis = self.analyze_sentiment(videos)
        
        # Determine channel consensus
        score = analysis['overall_score']
        consensus = "mixed"
        if score > 0.3: consensus = "bullish"
        elif score < -0.3: consensus = "bearish"
        
        # Contrarian signal if all channels strongly agree
        contrarian = False
        if score > 0.8 or score < -0.8:
            contrarian = True

        return {
            'overall_mood': score,
            'stock_mentions': analysis['stock_mentions'],
            'trending_topics': [v['title'] for v in videos[:5]],
            'channel_consensus': consensus,
            'contrarian_signal': contrarian
        }

if __name__ == "__main__":
    yt = YouTubeSentimentScraper()
    print(yt.get_youtube_mood())
