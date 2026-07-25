"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 7
Chapter 3: Sentiment Analysis - News Scraper

File: backend/sentiment/news_scraper.py
Purpose: Scrape news for sentiment analysis and Fear & Greed Index integration.
         Use lightweight NLP without heavy LLMs to respect 8GB RAM limit.

Features:
- Multi-source news aggregation (CoinDesk, Cointelegraph, Bloomberg Crypto)
- VADER-based sentiment scoring (lightweight, no GPU needed)
- Fear & Greed Index integration
- Event extraction for market-moving headlines
- Memory-efficient streaming processing

Design Patterns:
- Strategy: Different sentiment scoring methods
- Adapter: Normalize news from different sources
- Observer: Notify on sentiment shifts

Author: Opus 4.8
Domain: Sentiment Analysis, NLP, Alternative Data
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Deque
from collections import deque
from enum import Enum
import logging
import math

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER not installed. Install with: pip install vaderSentiment")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class NewsCategory(Enum):
    """News category classification."""
    REGULATORY = "regulatory"
    TECHNOLOGY = "technology"
    MARKET = "market"
    SECURITY = "security"
    ADOPTION = "adoption"
    MACRO = "macro"
    EXCHANGE = "exchange"
    OTHER = "other"


@dataclass(slots=True)
class NewsArticle:
    """
    Represents a scraped news article.
    Uses __slots__ for memory efficiency.
    """
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    category: NewsCategory
    mentioned_assets: List[str]
    sentiment_score: float = 0.0
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    relevance_score: float = 0.0
    is_breaking: bool = False


@dataclass(slots=True)
class SentimentAggregate:
    """Aggregated sentiment metrics."""
    timestamp: datetime
    overall_score: float
    overall_label: SentimentLabel
    score_by_asset: Dict[str, float]
    score_by_category: Dict[str, float]
    article_count: int
    breaking_news_count: int
    momentum: float  # Rate of change in sentiment


@dataclass(slots=True)
class FearGreedReading:
    """Fear & Greed Index reading."""
    timestamp: datetime
    value: int  # 0-100
    classification: str  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    previous_value: int
    change: int


class LightweightSentimentAnalyzer:
    """
    Lightweight sentiment analyzer using rule-based methods.
    Falls back to simple keyword scoring if VADER unavailable.
    """
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
        
        # Crypto-specific sentiment lexicon
        self.positive_keywords = {
            'bullish', 'surge', 'rally', 'breakout', 'moon', 'gain', 'profit',
            'adoption', 'partnership', 'upgrade', 'launch', 'approval', 'etf',
            'institutional', 'record', 'high', 'milestone', 'innovation'
        }
        
        self.negative_keywords = {
            'bearish', 'crash', 'dump', 'plunge', 'collapse', 'loss', 'hack',
            'exploit', 'ban', 'regulation', 'lawsuit', 'investigation', 'fraud',
            'scam', 'ftx', 'collapse', 'bankruptcy', 'liquidation', 'warning'
        }
        
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.4, 'significantly': 1.3,
            'massively': 1.8, 'slightly': 0.7, 'marginally': 0.6
        }
    
    def analyze(self, text: str) -> Tuple[float, SentimentLabel]:
        """
        Analyze sentiment of text.
        
        Returns:
            Tuple of (score from -1 to +1, label)
        """
        if self.analyzer:
            # Use VADER
            scores = self.analyzer.polarity_scores(text)
            compound = scores['compound']
        else:
            # Fallback to keyword-based scoring
            compound = self._keyword_score(text)
        
        # Convert to label
        label = self._score_to_label(compound)
        
        return compound, label
    
    def _keyword_score(self, text: str) -> float:
        """Simple keyword-based sentiment scoring."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        score = 0.0
        intensity_multiplier = 1.0
        
        for i, word in enumerate(words):
            # Check for intensifier before keyword
            if i > 0 and words[i-1] in self.intensifiers:
                intensity_multiplier = self.intensifiers[words[i-1]]
            
            if word in self.positive_keywords:
                score += 1.0 * intensity_multiplier
            elif word in self.negative_keywords:
                score -= 1.0 * intensity_multiplier
            
            intensity_multiplier = 1.0  # Reset
        
        # Normalize to -1 to +1 range
        if len(words) > 0:
            score = math.tanh(score / (len(words) * 0.1))
        
        return max(-1.0, min(1.0, score))
    
    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert numeric score to label."""
        if score >= 0.6:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentLabel.POSITIVE
        elif score >= -0.2:
            return SentimentLabel.NEUTRAL
        elif score >= -0.6:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.VERY_NEGATIVE


class NewsScraper:
    """
    Main news scraper with sentiment analysis.
    
    Features:
    - Multi-source scraping (mock implementations)
    - Real-time sentiment scoring
    - Asset mention detection
    - Breaking news detection
    """
    
    def __init__(self, max_articles: int = 1000):
        self.max_articles = max_articles
        self.analyzer = LightweightSentimentAnalyzer()
        
        # Article storage (bounded for memory safety)
        self.articles: Deque[NewsArticle] = deque(maxlen=max_articles)
        self.articles_by_id: Dict[str, NewsArticle] = {}
        
        # Cached aggregates
        self._latest_aggregate: Optional[SentimentAggregate] = None
        
        # Monitored assets
        self.monitored_assets = {'BTC', 'ETH', 'SOL', 'USDT', 'Bitcoin', 'Ethereum', 'Solana'}
        
        logger.info("NewsScraper initialized")
    
    async def scrape_coindesk(self) -> List[NewsArticle]:
        """Scrape CoinDesk (mock implementation)."""
        try:
            logger.debug("Scraping CoinDesk (mock mode)")
            
            # Mock articles for demonstration
            now = datetime.now(timezone.utc)
            
            return [
                NewsArticle(
                    id="cd_001",
                    title="Bitcoin Surges Past $70K as ETF Inflows Hit Record",
                    content="Bitcoin reached a new all-time high today as institutional demand continues...",
                    source="coindesk",
                    url="https://coindesk.com/example1",
                    published_at=now - timedelta(minutes=30),
                    category=NewsCategory.MARKET,
                    mentioned_assets=['BTC', 'Bitcoin'],
                    is_breaking=True
                ),
                NewsArticle(
                    id="cd_002",
                    title="SEC Delays Decision on Ethereum ETF Applications",
                    content="The Securities and Exchange Commission has postponed its decision...",
                    source="coindesk",
                    url="https://coindesk.com/example2",
                    published_at=now - timedelta(hours=2),
                    category=NewsCategory.REGULATORY,
                    mentioned_assets=['ETH', 'Ethereum']
                ),
            ]
            
        except Exception as e:
            logger.error(f"Error scraping CoinDesk: {e}")
            return []
    
    async def scrape_cointelegraph(self) -> List[NewsArticle]:
        """Scrape Cointelegraph (mock implementation)."""
        try:
            logger.debug("Scraping Cointelegraph (mock mode)")
            
            now = datetime.now(timezone.utc)
            
            return [
                NewsArticle(
                    id="ct_001",
                    title="Solana Network Processes Record 65M Transactions in Single Day",
                    content="The Solana blockchain achieved a new milestone...",
                    source="cointelegraph",
                    url="https://cointelegraph.com/example1",
                    published_at=now - timedelta(hours=1),
                    category=NewsCategory.TECHNOLOGY,
                    mentioned_assets=['SOL', 'Solana']
                ),
            ]
            
        except Exception as e:
            logger.error(f"Error scraping Cointelegraph: {e}")
            return []
    
    async def scrape_bloomberg_crypto(self) -> List[NewsArticle]:
        """Scrape Bloomberg Crypto (mock implementation)."""
        try:
            logger.debug("Scraping Bloomberg Crypto (mock mode)")
            
            now = datetime.now(timezone.utc)
            
            return [
                NewsArticle(
                    id="bb_001",
                    title="Crypto Hedge Funds Report Strong Q1 Performance",
                    content="Digital asset hedge funds posted average gains of 25%...",
                    source="bloomberg",
                    url="https://bloomberg.com/example1",
                    published_at=now - timedelta(hours=4),
                    category=NewsCategory.MACRO,
                    mentioned_assets=['BTC', 'ETH']
                ),
            ]
            
        except Exception as e:
            logger.error(f"Error scraping Bloomberg: {e}")
            return []
    
    def _analyze_article_sentiment(self, article: NewsArticle) -> NewsArticle:
        """Analyze sentiment for an article."""
        # Combine title and content for analysis
        full_text = f"{article.title}. {article.content}"
        
        score, label = self.analyzer.analyze(full_text)
        article.sentiment_score = score
        article.sentiment_label = label
        
        # Calculate relevance score based on monitored assets
        mentions = len([a for a in article.mentioned_assets if a in self.monitored_assets])
        article.relevance_score = min(1.0, mentions * 0.3 + (0.5 if article.is_breaking else 0))
        
        return article
    
    async def process_article(self, article: NewsArticle):
        """Process a single article."""
        try:
            # Analyze sentiment
            article = self._analyze_article_sentiment(article)
            
            # Store article
            if article.id not in self.articles_by_id:
                self.articles.append(article)
                self.articles_by_id[article.id] = article
                
                logger.info(
                    f"📰 {article.source}: {article.title[:50]}... "
                    f"(Sentiment: {article.sentiment_label.value}, Score: {article.sentiment_score:.2f})"
                )
            
            # Update aggregate
            self._update_aggregate()
            
        except Exception as e:
            logger.error(f"Error processing article: {e}", exc_info=True)
    
    def _update_aggregate(self):
        """Update aggregated sentiment metrics."""
        if not self.articles:
            return
        
        now = datetime.now(timezone.utc)
        
        # Recent articles (last hour)
        cutoff = now - timedelta(hours=1)
        recent = [a for a in self.articles if a.published_at >= cutoff]
        
        if not recent:
            return
        
        # Overall sentiment
        scores = [a.sentiment_score for a in recent]
        overall_score = sum(scores) / len(scores)
        overall_label = self.analyzer._score_to_label(overall_score)
        
        # By asset
        asset_scores: Dict[str, List[float]] = {}
        for article in recent:
            for asset in article.mentioned_assets:
                if asset not in asset_scores:
                    asset_scores[asset] = []
                asset_scores[asset].append(article.sentiment_score)
        
        score_by_asset = {
            asset: sum(s)/len(s) for asset, s in asset_scores.items() if s
        }
        
        # By category
        category_scores: Dict[str, List[float]] = {}
        for article in recent:
            cat = article.category.value
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(article.sentiment_score)
        
        score_by_category = {
            cat: sum(s)/len(s) for cat, s in category_scores.items() if s
        }
        
        # Momentum (compare to previous hour)
        old_cutoff = now - timedelta(hours=2)
        older = [a for a in self.articles if old_cutoff <= a.published_at < cutoff]
        
        if older:
            old_avg = sum(a.sentiment_score for a in older) / len(older)
            momentum = overall_score - old_avg
        else:
            momentum = 0.0
        
        # Breaking news count
        breaking_count = sum(1 for a in recent if a.is_breaking)
        
        self._latest_aggregate = SentimentAggregate(
            timestamp=now,
            overall_score=overall_score,
            overall_label=overall_label,
            score_by_asset=score_by_asset,
            score_by_category=score_by_category,
            article_count=len(recent),
            breaking_news_count=breaking_count,
            momentum=momentum
        )
    
    def get_latest_sentiment(self) -> Optional[SentimentAggregate]:
        """Get latest sentiment aggregate."""
        return self._latest_aggregate
    
    def get_asset_sentiment(self, asset: str) -> Optional[float]:
        """Get current sentiment for a specific asset."""
        if not self._latest_aggregate:
            return None
        return self._latest_aggregate.score_by_asset.get(asset)
    
    def detect_sentiment_shift(
        self,
        threshold: float = 0.3,
        window_minutes: int = 30
    ) -> Optional[Tuple[str, float]]:
        """Detect significant sentiment shifts."""
        if not self._latest_aggregate:
            return None
        
        # Compare current momentum to threshold
        if abs(self._latest_aggregate.momentum) > threshold:
            direction = "positive" if self._latest_aggregate.momentum > 0 else "negative"
            return (direction, self._latest_aggregate.momentum)
        
        return None
    
    async def run_scraping_cycle(self):
        """Run one complete scraping cycle from all sources."""
        sources = [
            self.scrape_coindesk(),
            self.scrape_cointelegraph(),
            self.scrape_bloomberg_crypto(),
        ]
        
        results = await asyncio.gather(*sources, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                for article in result:
                    await self.process_article(article)
            elif isinstance(result, Exception):
                logger.error(f"Scraping error: {result}")
    
    async def start_continuous_scraping(self, interval_minutes: float = 5.0):
        """Start continuous scraping loop."""
        while True:
            await self.run_scraping_cycle()
            await asyncio.sleep(interval_minutes * 60)


# Example usage
async def main():
    """Demonstration of NewsScraper functionality."""
    scraper = NewsScraper()
    
    # Run one scraping cycle
    await scraper.run_scraping_cycle()
    
    # Show results
    print("\n📊 Latest Sentiment Aggregate:")
    agg = scraper.get_latest_sentiment()
    if agg:
        print(f"  Overall: {agg.overall_label.value} ({agg.overall_score:.2f})")
        print(f"  Articles (1h): {agg.article_count}")
        print(f"  Breaking: {agg.breaking_news_count}")
        print(f"  Momentum: {agg.momentum:.2f}")
        
        print("\n  By Asset:")
        for asset, score in agg.score_by_asset.items():
            print(f"    {asset}: {score:.2f}")
    
    # Check for sentiment shifts
    shift = scraper.detect_sentiment_shift()
    if shift:
        print(f"\n⚠️ SENTIMENT SHIFT DETECTED: {shift[0]} ({shift[1]:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
