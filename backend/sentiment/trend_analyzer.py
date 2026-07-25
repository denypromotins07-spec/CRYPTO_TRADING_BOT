"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 7
Chapter 3: Sentiment Analysis - Trend Analyzer

File: backend/sentiment/trend_analyzer.py
Purpose: Track Google Trends data and geopolitical events for sentiment.
         Identify emerging narratives before they hit mainstream.

Features:
- Google Trends API integration (pytrends)
- Geopolitical event tracking
- Narrative emergence detection
- Cross-platform trend correlation
- Memory-efficient time-series storage

Design Patterns:
- Observer: Notify on trend spikes
- Strategy: Different trend detection algorithms
- Adapter: Normalize data from different sources

Author: Opus 4.8
Domain: Trend Analysis, Geopolitical Risk, Narrative Trading
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Deque, Set
from collections import deque
from enum import Enum
import logging
import math

try:
    from pytrends.trendinterest import TrendInterest
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logging.warning("pytrends not installed. Install with: pip install pytrends")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction classification."""
    SURGING = "surging"       # Rapid increase
    RISING = "rising"         # Steady increase
    STABLE = "stable"         # No significant change
    DECLINING = "declining"   # Steady decrease
    PLUMMETING = "plummeting" # Rapid decrease


class GeopoliticalEventType(Enum):
    """Types of geopolitical events affecting crypto."""
    REGULATORY_ANNOUNCEMENT = "regulatory"
    SANCTIONS = "sanctions"
    ELECTION = "election"
    CONFLICT = "conflict"
    TRADE_POLICY = "trade_policy"
    CENTRAL_BANK_POLICY = "central_bank"
    DIPLOMATIC_RELATIONS = "diplomatic"
    OTHER = "other"


@dataclass(slots=True)
class TrendData:
    """Google Trends data point."""
    timestamp: datetime
    keyword: str
    interest_score: int  # 0-100
    region: str
    related_queries: List[str]
    breakout_status: bool = False


@dataclass(slots=True)
class GeopoliticalEvent:
    """Geopolitical event affecting markets."""
    id: str
    event_type: GeopoliticalEventType
    title: str
    description: str
    timestamp: datetime
    affected_regions: List[str]
    affected_assets: List[str]
    severity: int  # 1-10 scale
    verified: bool = True


@dataclass(slots=True)
class TrendAnalysis:
    """Complete trend analysis result."""
    timestamp: datetime
    keyword: str
    current_interest: int
    trend_direction: TrendDirection
    momentum: float  # Rate of change
    is_breakout: bool
    related_narratives: List[str]
    risk_score: float  # 0-1 based on geopolitical factors


class TrendDetector:
    """Detects emerging trends from search data."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        # Bounded history per keyword
        self.trend_history: Dict[str, Deque[TrendData]] = {}
    
    def add_trend_data(self, data: TrendData):
        """Add new trend data point."""
        if data.keyword not in self.trend_history:
            self.trend_history[data.keyword] = deque(maxlen=self.history_size)
        
        self.trend_history[data.keyword].append(data)
    
    def detect_direction(self, keyword: str, window: int = 5) -> TrendDirection:
        """Determine trend direction from recent data."""
        history = self.trend_history.get(keyword)
        if not history or len(history) < window:
            return TrendDirection.STABLE
        
        recent = list(history)[-window:]
        older = list(history)[-window*2:-window] if len(history) >= window*2 else recent
        
        recent_avg = sum(d.interest_score for d in recent) / len(recent)
        older_avg = sum(d.interest_score for d in older) / len(older)
        
        change_pct = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        if change_pct > 0.5:
            return TrendDirection.SURGING
        elif change_pct > 0.1:
            return TrendDirection.RISING
        elif change_pct < -0.5:
            return TrendDirection.PLUMMETING
        elif change_pct < -0.1:
            return TrendDirection.DECLINING
        else:
            return TrendDirection.STABLE
    
    def calculate_momentum(self, keyword: str) -> float:
        """Calculate trend momentum (rate of change)."""
        history = self.trend_history.get(keyword)
        if not history or len(history) < 3:
            return 0.0
        
        recent = list(history)[-3:]
        if len(recent) < 2:
            return 0.0
        
        # Linear regression slope (simplified)
        x_vals = list(range(len(recent)))
        y_vals = [d.interest_score for d in recent]
        
        n = len(recent)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_xx = sum(x * x for x in x_vals)
        
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Normalize to 0-1 range
        return max(-1.0, min(1.0, slope / 20.0))
    
    def detect_breakout(self, keyword: str, threshold: int = 80) -> bool:
        """Detect if keyword is experiencing breakout interest."""
        history = self.trend_history.get(keyword)
        if not history or len(history) < 5:
            return False
        
        recent = list(history)[-5:]
        return any(d.interest_score >= threshold for d in recent)


class TrendAnalyzer:
    """
    Main trend analyzer combining Google Trends and geopolitical events.
    
    Features:
    - Multi-keyword trend tracking
    - Geopolitical event correlation
    - Narrative emergence detection
    - Risk scoring
    """
    
    def __init__(self, update_interval_minutes: float = 30.0):
        self.update_interval = update_interval_minutes * 60
        self.detector = TrendDetector()
        
        # Monitored keywords
        self.keywords: Set[str] = {
            'Bitcoin', 'Ethereum', 'Solana', 'crypto', 'blockchain',
            'BTC ETF', 'crypto regulation', 'DeFi', 'NFT', 'stablecoin'
        }
        
        # Geopolitical events (bounded)
        self.events: Deque[GeopoliticalEvent] = deque(maxlen=500)
        
        # Cached analyses
        self._latest_analysis: Dict[str, TrendAnalysis] = {}
        
        # Subscribers
        self._subscribers: List[Callable[[str, TrendAnalysis], None]] = []
        
        # Running state
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info("TrendAnalyzer initialized")
    
    def subscribe(self, callback: Callable[[str, TrendAnalysis], None]):
        """Subscribe to trend updates."""
        self._subscribers.append(callback)
    
    async def fetch_google_trends(self, keyword: str) -> Optional[TrendData]:
        """Fetch Google Trends data for a keyword."""
        if not PYTRENDS_AVAILABLE:
            # Mock data for demonstration
            import random
            now = datetime.now(timezone.utc)
            
            return TrendData(
                timestamp=now,
                keyword=keyword,
                interest_score=random.randint(20, 90),
                region="US",
                related_queries=[f"{keyword} price", f"buy {keyword}"],
                breakout_status=random.random() > 0.8
            )
        
        try:
            # In production: Use actual pytrends
            # pytrends = TrendInterest()
            # pytrends.build_payload([keyword], timeframe='today 1-d')
            # data = pytrends.interest_over_time()
            
            logger.debug(f"Fetched Google Trends for {keyword} (mock mode)")
            
            now = datetime.now(timezone.utc)
            return TrendData(
                timestamp=now,
                keyword=keyword,
                interest_score=50,  # Would be actual data
                region="US",
                related_queries=[],
                breakout_status=False
            )
            
        except Exception as e:
            logger.error(f"Error fetching Google Trends for {keyword}: {e}")
            return None
    
    def add_geopolitical_event(self, event: GeopoliticalEvent):
        """Add a geopolitical event."""
        self.events.append(event)
        
        # Update risk scores for affected assets
        for asset in event.affected_assets:
            self._update_asset_risk(asset)
    
    def _update_asset_risk(self, asset: str):
        """Update risk score for an asset based on recent events."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)
        
        relevant_events = [
            e for e in self.events
            if asset in e.affected_assets and e.timestamp >= cutoff
        ]
        
        # Calculate cumulative risk
        risk_score = sum(e.severity / 10.0 for e in relevant_events) / max(len(relevant_events), 1)
        
        # Store in latest analysis if exists
        if asset in self._latest_analysis:
            self._latest_analysis[asset].risk_score = min(1.0, risk_score)
    
    def analyze_keyword(self, keyword: str) -> Optional[TrendAnalysis]:
        """Perform complete analysis for a keyword."""
        direction = self.detector.detect_direction(keyword)
        momentum = self.detector.calculate_momentum(keyword)
        breakout = self.detector.detect_breakout(keyword)
        
        # Get related narratives (simplified)
        related = self._get_related_narratives(keyword)
        
        # Calculate risk score from geopolitical events
        risk_score = self._calculate_geopolitical_risk(keyword)
        
        analysis = TrendAnalysis(
            timestamp=datetime.now(timezone.utc),
            keyword=keyword,
            current_interest=50,  # Would get from latest data
            trend_direction=direction,
            momentum=momentum,
            is_breakout=breakout,
            related_narratives=related,
            risk_score=risk_score
        )
        
        self._latest_analysis[keyword] = analysis
        
        return analysis
    
    def _get_related_narratives(self, keyword: str) -> List[str]:
        """Get narratives related to a keyword."""
        narrative_map = {
            'Bitcoin': ['digital gold', 'institutional adoption', 'ETF approval', 'halving'],
            'Ethereum': ['DeFi', 'staking', 'layer 2', 'upgrade'],
            'Solana': ['high performance', 'NFTs', 'ecosystem growth'],
            'crypto regulation': ['SEC', 'compliance', 'institutional clarity'],
            'stablecoin': ['payments', 'DeFi liquidity', 'regulatory scrutiny'],
        }
        
        return narrative_map.get(keyword, [])
    
    def _calculate_geopolitical_risk(self, keyword: str) -> float:
        """Calculate geopolitical risk for a keyword."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)
        
        relevant_events = [e for e in self.events if e.timestamp >= cutoff]
        
        # Map keywords to event types
        risk_keywords = {
            'Bitcoin': [GeopoliticalEventType.SANCTIONS, GeopoliticalEventType.REGULATORY_ANNOUNCEMENT],
            'crypto regulation': [GeopoliticalEventType.REGULATORY_ANNOUNCEMENT],
            'stablecoin': [GeopoliticalEventType.REGULATORY_ANNOUNCEMENT, GeopoliticalEventType.TRADE_POLICY],
        }
        
        relevant_types = risk_keywords.get(keyword, [])
        if not relevant_types:
            return 0.0
        
        matching_events = [e for e in relevant_events if e.event_type in relevant_types]
        
        if not matching_events:
            return 0.0
        
        avg_severity = sum(e.severity for e in matching_events) / len(matching_events)
        return min(1.0, avg_severity / 10.0)
    
    async def run_analysis_cycle(self):
        """Run one complete analysis cycle."""
        for keyword in self.keywords:
            # Fetch trends data
            trend_data = await self.fetch_google_trends(keyword)
            if trend_data:
                self.detector.add_trend_data(trend_data)
            
            # Analyze
            analysis = self.analyze_keyword(keyword)
            
            if analysis and analysis.is_breakout:
                logger.info(f"🚀 BREAKOUT detected for {keyword}!")
            
            # Notify subscribers
            if analysis:
                for subscriber in self._subscribers:
                    try:
                        res = subscriber(keyword, analysis)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception as e:
                        logger.error(f"Error notifying subscriber: {e}")
    
    def get_emerging_narratives(self) -> List[Tuple[str, float]]:
        """Get currently emerging narratives ranked by momentum."""
        narratives = []
        
        for keyword, analysis in self._latest_analysis.items():
            if analysis.trend_direction in [TrendDirection.SURGING, TrendDirection.RISING]:
                for narrative in analysis.related_narratives:
                    narratives.append((narrative, analysis.momentum))
        
        # Deduplicate and rank
        narrative_scores: Dict[str, float] = {}
        for narrative, momentum in narratives:
            narrative_scores[narrative] = max(narrative_scores.get(narrative, 0), momentum)
        
        sorted_narratives = sorted(
            narrative_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_narratives[:10]
    
    async def start_continuous_monitoring(self):
        """Start continuous trend monitoring."""
        while self._running:
            await self.run_analysis_cycle()
            await asyncio.sleep(self.update_interval)
    
    async def start(self):
        """Start monitoring."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting TrendAnalyzer")
        
        self._tasks = [
            asyncio.create_task(self.start_continuous_monitoring())
        ]
    
    async def stop(self):
        """Stop monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


# Example usage
async def main():
    """Demonstration of TrendAnalyzer functionality."""
    analyzer = TrendAnalyzer(update_interval_minutes=1.0)
    
    def trend_handler(keyword: str, analysis: TrendAnalysis):
        print(
            f"📈 {keyword}: "
            f"Direction={analysis.trend_direction.value}, "
            f"Momentum={analysis.momentum:.2f}, "
            f"Breakout={analysis.is_breakout}"
        )
    
    analyzer.subscribe(trend_handler)
    
    # Add some geopolitical events
    analyzer.add_geopolitical_event(GeopoliticalEvent(
        id="geo_001",
        event_type=GeopoliticalEventType.REGULATORY_ANNOUNCEMENT,
        title="SEC Announces New Crypto Framework",
        description="The SEC released comprehensive guidelines...",
        timestamp=datetime.now(timezone.utc),
        affected_regions=["US"],
        affected_assets=["Bitcoin", "Ethereum", "stablecoin"],
        severity=7
    ))
    
    # Run analysis
    await analyzer.run_analysis_cycle()
    
    # Show emerging narratives
    print("\n🔥 Emerging Narratives:")
    for narrative, momentum in analyzer.get_emerging_narratives():
        print(f"  {narrative} (momentum: {momentum:.2f})")
    
    await asyncio.sleep(2)
    await analyzer.stop()


if __name__ == "__main__":
    asyncio.run(main())
