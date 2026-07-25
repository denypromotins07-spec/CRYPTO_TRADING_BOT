"""
Regime Adapter: Dynamic strategy weight adjustment based on SOUL.md data.
Reads learned patterns from SOUL.md and adapts trading strategies accordingly.
Implements regime detection and strategy selection for optimal performance.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import time
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Detected market regimes."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"
    CRISIS = "crisis"


class StrategyType(Enum):
    """Available strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    TREND_FOLLOWING = "trend_following"


@dataclass
class RegimeConfig:
    """Configuration for a specific regime."""
    regime: MarketRegime
    preferred_strategies: List[StrategyType]
    avoided_strategies: List[StrategyType]
    position_size_multiplier: float
    risk_multiplier: float
    max_correlation_exposure: float
    stop_loss_adjustment: float  # Multiplier for stop loss distance


@dataclass
class SoulInsight:
    """Parsed insight from SOUL.md."""
    timestamp: float
    asset: Optional[str]
    insight_type: str
    content: str
    relevance_score: float
    actionable: bool


@dataclass
class StrategyWeights:
    """Dynamic weights for different strategies."""
    momentum: float = 0.2
    mean_reversion: float = 0.2
    breakout: float = 0.2
    trend_following: float = 0.2
    market_making: float = 0.1
    arbitrage: float = 0.1
    
    def normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum([
            self.momentum,
            self.mean_reversion,
            self.breakout,
            self.trend_following,
            self.market_making,
            self.arbitrage
        ])
        if total > 0:
            self.momentum /= total
            self.mean_reversion /= total
            self.breakout /= total
            self.trend_following /= total
            self.market_making /= total
            self.arbitrage /= total
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "momentum": round(self.momentum, 4),
            "mean_reversion": round(self.mean_reversion, 4),
            "breakout": round(self.breakout, 4),
            "trend_following": round(self.trend_following, 4),
            "market_making": round(self.market_making, 4),
            "arbitrage": round(self.arbitrage, 4),
        }


class RegimeAdapter:
    """
    Adapts strategy weights based on detected market regime and SOUL.md insights.
    Continuously learns from past performance and adjusts parameters.
    """
    
    # Default regime configurations
    DEFAULT_REGIME_CONFIGS: Dict[MarketRegime, RegimeConfig] = {
        MarketRegime.TRENDING_BULL: RegimeConfig(
            regime=MarketRegime.TRENDING_BULL,
            preferred_strategies=[StrategyType.MOMENTUM, StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT],
            avoided_strategies=[StrategyType.MEAN_REVERSION],
            position_size_multiplier=1.2,
            risk_multiplier=1.1,
            max_correlation_exposure=0.8,
            stop_loss_adjustment=1.2,
        ),
        MarketRegime.TRENDING_BEAR: RegimeConfig(
            regime=MarketRegime.TRENDING_BEAR,
            preferred_strategies=[StrategyType.TREND_FOLLOWING],
            avoided_strategies=[StrategyType.MOMENTUM, StrategyType.BREAKOUT],
            position_size_multiplier=0.8,
            risk_multiplier=0.9,
            max_correlation_exposure=0.6,
            stop_loss_adjustment=1.0,
        ),
        MarketRegime.RANGING: RegimeConfig(
            regime=MarketRegime.RANGING,
            preferred_strategies=[StrategyType.MEAN_REVERSION, StrategyType.MARKET_MAKING],
            avoided_strategies=[StrategyType.TREND_FOLLOWING],
            position_size_multiplier=1.0,
            risk_multiplier=1.0,
            max_correlation_exposure=0.7,
            stop_loss_adjustment=0.8,
        ),
        MarketRegime.HIGH_VOLATILITY: RegimeConfig(
            regime=MarketRegime.HIGH_VOLATILITY,
            preferred_strategies=[StrategyType.BREAKOUT],
            avoided_strategies=[StrategyType.MARKET_MAKING, StrategyType.MEAN_REVERSION],
            position_size_multiplier=0.5,
            risk_multiplier=0.7,
            max_correlation_exposure=0.4,
            stop_loss_adjustment=1.5,
        ),
        MarketRegime.LOW_VOLATILITY: RegimeConfig(
            regime=MarketRegime.LOW_VOLATILITY,
            preferred_strategies=[StrategyType.MARKET_MAKING, StrategyType.ARBITRAGE],
            avoided_strategies=[],
            position_size_multiplier=1.3,
            risk_multiplier=1.2,
            max_correlation_exposure=0.9,
            stop_loss_adjustment=0.7,
        ),
        MarketRegime.CRISIS: RegimeConfig(
            regime=MarketRegime.CRISIS,
            preferred_strategies=[],
            avoided_strategies=list(StrategyType),
            position_size_multiplier=0.1,
            risk_multiplier=0.3,
            max_correlation_exposure=0.2,
            stop_loss_adjustment=2.0,
        ),
    }
    
    def __init__(
        self,
        soul_md_path: str = "SOUL.md",
        update_interval_seconds: float = 60.0,
    ):
        self.soul_md_path = Path(soul_md_path)
        self.update_interval = update_interval_seconds
        
        # Current state
        self._current_regime: MarketRegime = MarketRegime.TRANSITION
        self._regime_configs = self.DEFAULT_REGIME_CONFIGS.copy()
        self._strategy_weights = StrategyWeights()
        
        # Parsed insights from SOUL.md
        self._insights: List[SoulInsight] = []
        self._last_soul_read_time: float = 0.0
        self._soul_file_hash: int = 0
        
        # Asset-specific adaptations
        self._asset_regimes: Dict[str, MarketRegime] = {}
        self._asset_weights: Dict[str, StrategyWeights] = {}
        
        # Performance tracking for learning
        self._regime_performance: Dict[MarketRegime, List[float]] = defaultdict(list)
        self._strategy_performance: Dict[StrategyType, List[float]] = defaultdict(list)
        
        logger.info("RegimeAdapter initialized")
    
    def detect_regime(
        self,
        volatility: float,
        trend_strength: float,
        correlation_avg: float,
        volume_ratio: float,
    ) -> MarketRegime:
        """
        Detect current market regime from indicators.
        
        Args:
            volatility: Annualized volatility (e.g., 0.5 for 50%)
            trend_strength: ADX or similar trend indicator (0-1)
            correlation_avg: Average correlation between assets (-1 to 1)
            volume_ratio: Current volume vs average volume
        
        Returns:
            Detected MarketRegime
        """
        # Crisis detection (extreme conditions)
        if volatility > 1.5 or correlation_avg > 0.9:
            regime = MarketRegime.CRISIS
        
        # High volatility regime
        elif volatility > 0.8:
            regime = MarketRegime.HIGH_VOLATILITY
        
        # Low volatility regime
        elif volatility < 0.2:
            regime = MarketRegime.LOW_VOLATILITY
        
        # Trending regimes
        elif trend_strength > 0.6:
            if trend_strength > 0 and volume_ratio > 1.2:
                regime = MarketRegime.TRENDING_BULL
            else:
                regime = MarketRegime.TRENDING_BEAR
        
        # Ranging regime
        elif trend_strength < 0.3:
            regime = MarketRegime.RANGING
        
        # Transition (unclear signals)
        else:
            regime = MarketRegime.TRANSITION
        
        self._current_regime = regime
        logger.info(f"Detected regime: {regime.value}")
        
        return regime
    
    def detect_asset_specific_regime(self, asset: str, price_data: List[float]) -> MarketRegime:
        """Detect regime for a specific asset."""
        if len(price_data) < 20:
            return MarketRegime.TRANSITION
        
        # Calculate asset-specific metrics
        returns = [(price_data[i] - price_data[i-1]) / price_data[i-1] 
                   for i in range(1, len(price_data))]
        
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 * (252 ** 0.5)
        
        # Simple trend detection
        sma_short = sum(price_data[-7:]) / 7
        sma_long = sum(price_data[-25:]) / 25 if len(price_data) >= 25 else sma_short
        trend_strength = (sma_short - sma_long) / sma_long if sma_long > 0 else 0
        
        return self.detect_regime(
            volatility=volatility,
            trend_strength=abs(trend_strength),
            correlation_avg=0.5,  # Would calculate from multi-asset data
            volume_ratio=1.0,
        )
    
    def update_strategy_weights(self) -> StrategyWeights:
        """Update strategy weights based on current regime."""
        config = self._regime_configs.get(self._current_regime)
        
        if not config:
            return self._strategy_weights
        
        # Start with base weights
        weights = StrategyWeights()
        
        # Boost preferred strategies
        boost_factor = 1.5
        for strategy in config.preferred_strategies:
            if strategy == StrategyType.MOMENTUM:
                weights.momentum *= boost_factor
            elif strategy == StrategyType.MEAN_REVERSION:
                weights.mean_reversion *= boost_factor
            elif strategy == StrategyType.BREAKOUT:
                weights.breakout *= boost_factor
            elif strategy == StrategyType.TREND_FOLLOWING:
                weights.trend_following *= boost_factor
            elif strategy == StrategyType.MARKET_MAKING:
                weights.market_making *= boost_factor
            elif strategy == StrategyType.ARBITRAGE:
                weights.arbitrage *= boost_factor
        
        # Reduce avoided strategies
        reduction_factor = 0.3
        for strategy in config.avoided_strategies:
            if strategy == StrategyType.MOMENTUM:
                weights.momentum *= reduction_factor
            elif strategy == StrategyType.MEAN_REVERSION:
                weights.mean_reversion *= reduction_factor
            elif strategy == StrategyType.BREAKOUT:
                weights.breakout *= reduction_factor
            elif strategy == StrategyType.TREND_FOLLOWING:
                weights.trend_following *= reduction_factor
            elif strategy == StrategyType.MARKET_MAKING:
                weights.market_making *= reduction_factor
            elif strategy == StrategyType.ARBITRAGE:
                weights.arbitrage *= reduction_factor
        
        # Apply SOUL.md insights
        self._apply_soul_insights(weights)
        
        # Normalize
        weights.normalize()
        
        self._strategy_weights = weights
        logger.debug(f"Updated strategy weights: {weights.to_dict()}")
        
        return weights
    
    def _apply_soul_insights(self, weights: StrategyWeights) -> None:
        """Apply insights from SOUL.md to adjust weights."""
        if not self._insights:
            return
        
        # Count relevant insights per strategy
        strategy_mentions: Dict[str, int] = defaultdict(int)
        
        for insight in self._insights[-100:]:  # Last 100 insights
            if not insight.actionable:
                continue
            
            content_lower = insight.content.lower()
            
            if "momentum" in content_lower:
                strategy_mentions["momentum"] += 1 if "work" in content_lower or "effective" in content_lower else -1
            if "mean reversion" in content_lower:
                strategy_mentions["mean_reversion"] += 1 if "work" in content_lower else -1
            if "breakout" in content_lower:
                strategy_mentions["breakout"] += 1 if "success" in content_lower else -1
            if "trend" in content_lower:
                strategy_mentions["trend_following"] += 1 if "follow" in content_lower else -1
        
        # Apply adjustments based on insight sentiment
        adjustment_factor = 0.1
        
        if strategy_mentions.get("momentum", 0) > 2:
            weights.momentum *= (1 + adjustment_factor)
        elif strategy_mentions.get("momentum", 0) < -2:
            weights.momentum *= (1 - adjustment_factor)
        
        if strategy_mentions.get("mean_reversion", 0) > 2:
            weights.mean_reversion *= (1 + adjustment_factor)
        elif strategy_mentions.get("mean_reversion", 0) < -2:
            weights.mean_reversion *= (1 - adjustment_factor)
    
    def parse_soul_md(self) -> List[SoulInsight]:
        """Parse SOUL.md file for actionable insights."""
        if not self.soul_md_path.exists():
            return []
        
        try:
            content = self.soul_md_path.read_text()
            
            # Check if file has changed
            current_hash = hash(content)
            if current_hash == self._soul_file_hash:
                return self._insights
            
            self._soul_file_hash = current_hash
            
            insights = []
            
            # Parse markdown entries
            entry_pattern = r'### ([^-]+) - ([^\n]+)\n+(.*?)(?=---|$)'
            matches = re.findall(entry_pattern, content, re.DOTALL)
            
            for match in matches:
                entry_type = match[0].strip()
                title = match[1].strip()
                body = match[2].strip()
                
                # Extract asset if present
                asset_match = re.search(r'\*\*Asset\*\*: (\w+)', body)
                asset = asset_match.group(1) if asset_match else None
                
                # Determine if actionable
                actionable_keywords = ["learned", "discovered", "avoid", "prefer", 
                                      "adjust", "increase", "decrease", "use"]
                is_actionable = any(kw in body.lower() for kw in actionable_keywords)
                
                insight = SoulInsight(
                    timestamp=time.time(),  # Would parse from entry
                    asset=asset,
                    insight_type=entry_type,
                    content=f"{title}: {body}",
                    relevance_score=self._calculate_relevance(body),
                    actionable=is_actionable,
                )
                insights.append(insight)
            
            # Sort by relevance
            insights.sort(key=lambda x: x.relevance_score, reverse=True)
            
            self._insights = insights[:50]  # Keep top 50 most relevant
            self._last_soul_read_time = time.time()
            
            logger.info(f"Parsed {len(self._insights)} insights from SOUL.md")
            
            return self._insights
            
        except Exception as e:
            logger.error(f"Failed to parse SOUL.md: {e}")
            return []
    
    def _calculate_relevance(self, content: str) -> float:
        """Calculate relevance score for an insight."""
        score = 1.0
        
        # Higher relevance for recent entries (would use actual timestamp)
        score *= 1.0
        
        # Higher relevance for entries with specific recommendations
        if "should" in content.lower() or "must" in content.lower():
            score *= 1.5
        
        # Higher relevance for entries with quantitative data
        if any(char.isdigit() for char in content):
            score *= 1.2
        
        # Lower relevance for generic statements
        generic_words = ["maybe", "possibly", "sometimes"]
        if any(word in content.lower() for word in generic_words):
            score *= 0.7
        
        return score
    
    def get_regime_config(self, regime: Optional[MarketRegime] = None) -> RegimeConfig:
        """Get configuration for current or specified regime."""
        target_regime = regime or self._current_regime
        return self._regime_configs.get(
            target_regime,
            self.DEFAULT_REGIME_CONFIGS.get(MarketRegime.TRANSITION, 
                RegimeConfig(
                    regime=MarketRegime.TRANSITION,
                    preferred_strategies=[],
                    avoided_strategies=[],
                    position_size_multiplier=1.0,
                    risk_multiplier=1.0,
                    max_correlation_exposure=0.7,
                    stop_loss_adjustment=1.0,
                ))
        )
    
    def record_strategy_performance(self, strategy: StrategyType, pnl: float) -> None:
        """Record performance for a strategy to enable learning."""
        self._strategy_performance[strategy].append(pnl)
        
        # Keep only last 100 trades
        if len(self._strategy_performance[strategy]) > 100:
            self._strategy_performance[strategy] = self._strategy_performance[strategy][-100:]
    
    def record_regime_performance(self, pnl: float) -> None:
        """Record performance for current regime."""
        self._regime_performance[self._current_regime].append(pnl)
        
        if len(self._regime_performance[self._current_regime]) > 100:
            self._regime_performance[self._current_regime] = \
                self._regime_performance[self._current_regime][-100:]
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of current adaptation state."""
        return {
            "current_regime": self._current_regime.value,
            "strategy_weights": self._strategy_weights.to_dict(),
            "insights_count": len(self._insights),
            "asset_regimes": {k: v.value for k, v in self._asset_regimes.items()},
            "regime_performance": {
                k.value: sum(v) / len(v) if v else 0
                for k, v in self._regime_performance.items()
            },
        }


# Singleton instance
_adapter_instance: Optional[RegimeAdapter] = None


def get_regime_adapter() -> RegimeAdapter:
    """Get singleton instance of RegimeAdapter."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = RegimeAdapter()
    return _adapter_instance


if __name__ == "__main__":
    # Example usage
    adapter = get_regime_adapter()
    
    # Simulate regime detection
    regime = adapter.detect_regime(
        volatility=0.5,
        trend_strength=0.7,
        correlation_avg=0.6,
        volume_ratio=1.3,
    )
    print(f"Detected regime: {regime.value}")
    
    # Update strategy weights
    weights = adapter.update_strategy_weights()
    print(f"Strategy weights: {weights.to_dict()}")
    
    # Get regime config
    config = adapter.get_regime_config()
    print(f"Position size multiplier: {config.position_size_multiplier}")
    
    # Get adaptation summary
    summary = adapter.get_adaptation_summary()
    print(f"\nAdaptation Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nRegime Adapter module initialized successfully.")
