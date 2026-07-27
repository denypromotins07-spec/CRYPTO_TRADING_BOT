"""
Behavioral Finance Module for Agent-Based Simulation

Models psychological biases and collective behaviors in crypto markets:
- Herding: Following the crowd regardless of fundamentals
- Panic Selling: Disproportionate reaction to downside
- FOMO (Fear Of Missing Out): Chasing rallies
- Loss Aversion: Pain of losses > pleasure of gains
- Anchoring: Fixating on reference prices

Optimized for 8GB RAM with efficient NumPy operations.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification for behavioral adaptation."""
    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"
    CRASH = "crash"
    RECOVERY = "recovery"


@dataclass
class BehavioralParameters:
    """Calibrated behavioral parameters for crypto markets."""
    herding_strength: float = 0.35          # Tendency to follow crowd (0-1)
    panic_sensitivity: float = 0.6          # Reaction to negative news
    fomo_intensity: float = 0.45            # Chase momentum intensity
    loss_aversion_coeff: float = 2.25       # Kahneman-Tversky coefficient
    anchoring_weight: float = 0.3           # Weight on reference price
    disposition_effect: float = 0.4         # Sell winners too early
    overconfidence_bias: float = 0.2        # Overestimate signal accuracy
    recency_bias: float = 0.5               # overweight recent events
    
    # Regime-specific adjustments
    panic_multiplier_crash: float = 2.5
    fomo_multiplier_bull: float = 1.8
    
    @classmethod
    def crypto_defaults(cls) -> BehavioralParameters:
        """Parameters calibrated to BTC/ETH/SOL markets."""
        return cls(
            herding_strength=0.42,
            panic_sensitivity=0.65,
            fomo_intensity=0.55,
            loss_aversion_coeff=2.5,
            anchoring_weight=0.25,
            disposition_effect=0.45,
            overconfidence_bias=0.3,
            recency_bias=0.6,
        )


@dataclass
class AgentSentiment:
    """Individual agent sentiment state."""
    fear_level: float = 0.0         # 0 (calm) to 1 (terrified)
    greed_level: float = 0.0        # 0 (neutral) to 1 (greedy)
    confidence: float = 0.5         # 0 (doubt) to 1 (certain)
    reference_price: float = 0.0    # Anchored price
    recent_pnl: List[float] = field(default_factory=list)
    
    def update_reference(self, price: float, adaptation_rate: float = 0.1) -> None:
        """Slowly adapt reference price (anchoring effect)."""
        if self.reference_price == 0.0:
            self.reference_price = price
        else:
            self.reference_price = (1 - adaptation_rate) * self.reference_price + adaptation_rate * price
    
    def add_pnl(self, pnl: float, window_size: int = 20) -> None:
        """Track recent PnL for behavioral feedback."""
        self.recent_pnl.append(pnl)
        if len(self.recent_pnl) > window_size:
            self.recent_pnl.pop(0)
    
    def get_cumulative_pnl(self) -> float:
        return sum(self.recent_pnl)


class BehavioralFinanceEngine:
    """
    Core engine for simulating behavioral biases in market agents.
    
    Implements:
    - Prospect Theory value function
    - Herding dynamics via social contagion
    - Panic/FOMO triggers based on price action
    - Sentiment propagation through agent network
    """
    
    def __init__(self, params: Optional[BehavioralParameters] = None, seed: int = 42):
        self.params = params or BehavioralParameters.crypto_defaults()
        self.rng = np.random.default_rng(seed)
        self.market_regime = MarketRegime.CHOP
        self.agent_sentiments: Dict[int, AgentSentiment] = {}
        
        # Market-wide metrics
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.volatility_estimate: float = 0.02
        
    def register_agent(self, agent_id: int, initial_price: float) -> None:
        """Register a new agent with initial sentiment state."""
        self.agent_sentiments[agent_id] = AgentSentiment(reference_price=initial_price)
    
    def update_market_regime(self, returns: np.ndarray) -> MarketRegime:
        """Classify current market regime from recent returns."""
        if len(returns) < 5:
            return self.market_regime
            
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        min_ret = np.min(returns)
        
        # Volatility-adjusted regime detection
        if min_ret < -0.05 and std_ret > 0.03:
            self.market_regime = MarketRegime.CRASH
        elif mean_ret > 0.02 and std_ret < 0.02:
            self.market_regime = MarketRegime.BULL
        elif mean_ret < -0.01 and std_ret < 0.02:
            self.market_regime = MarketRegime.BEAR
        elif abs(mean_ret) < 0.005:
            self.market_regime = MarketRegime.CHOP
        elif mean_ret > 0.01 and self.market_regime == MarketRegime.BEAR:
            self.market_regime = MarketRegime.RECOVERY
            
        return self.market_regime
    
    def prospect_value(self, x: float) -> float:
        """
        Kahneman-Tversky Prospect Theory value function.
        v(x) = x^α for gains, -λ(-x)^β for losses
        """
        alpha = 0.88  # Diminishing sensitivity
        beta = 0.88
        lambda_ = self.params.loss_aversion_coeff
        
        if x >= 0:
            return x ** alpha
        else:
            return -lambda_ * ((-x) ** beta)
    
    def calculate_herding_pressure(
        self, 
        agent_id: int, 
        buy_ratio: float,
        price_change: float
    ) -> float:
        """
        Calculate herding pressure on an agent.
        
        Args:
            agent_id: Target agent
            buy_ratio: Fraction of agents buying (0-1)
            price_change: Recent price change (%)
            
        Returns:
            Herding impulse (-1 to 1), positive = buy pressure
        """
        base_herding = (buy_ratio - 0.5) * 2  # Normalize to [-1, 1]
        
        # Amplify by regime
        multiplier = 1.0
        if self.market_regime == MarketRegime.CRASH:
            multiplier = self.params.panic_multiplier_crash
        elif self.market_regime == MarketRegime.BULL:
            multiplier = self.params.fomo_multiplier_bull
        
        # Price momentum reinforces herding
        momentum_signal = np.tanh(price_change * 10)
        
        combined = (base_herding * self.params.herding_strength * multiplier 
                   + momentum_signal * self.params.recency_bias)
        
        return np.clip(combined, -1, 1)
    
    def trigger_panic_response(
        self, 
        agent_id: int, 
        drawdown: float
    ) -> Tuple[bool, float]:
        """
        Determine if agent panics given current drawdown.
        
        Returns:
            (panic_triggered, panic_intensity)
        """
        if agent_id not in self.agent_sentiments:
            return False, 0.0
            
        sentiment = self.agent_sentiments[agent_id]
        
        # Base panic probability increases with drawdown
        base_prob = 1 / (1 + np.exp(-10 * (drawdown + self.params.panic_sensitivity * 0.05)))
        
        # Amplify in crash regime
        if self.market_regime == MarketRegime.CRASH:
            base_prob *= self.params.panic_multiplier_crash
        
        # Fear level increases panic susceptibility
        fear_adjusted_prob = base_prob * (1 + sentiment.fear_level)
        
        # Check cumulative losses
        cumulative_pnl = sentiment.get_cumulative_pnl()
        if cumulative_pnl < -0.1:  # Down 10%
            fear_adjusted_prob *= 1.5
        
        triggered = self.rng.random() < np.clip(fear_adjusted_prob, 0, 1)
        intensity = min(abs(drawdown) * 5, 1.0) if triggered else 0.0
        
        if triggered:
            sentiment.fear_level = min(sentiment.fear_level + 0.3, 1.0)
            logger.debug(f"Agent {agent_id} PANIC triggered: intensity={intensity:.2f}")
        
        return triggered, intensity
    
    def trigger_fomo_response(
        self,
        agent_id: int,
        rally: float
    ) -> Tuple[bool, float]:
        """
        Determine if agent experiences FOMO given recent rally.
        
        Returns:
            (fomo_triggered, fomo_intensity)
        """
        if agent_id not in self.agent_sentiments:
            return False, 0.0
            
        sentiment = self.agent_sentiments[agent_id]
        
        # FOMO threshold
        threshold = 0.03 / (1 + sentiment.greed_level)  # Greedy agents need less provocation
        
        base_prob = 0.0
        if rally > threshold:
            excess_rally = rally - threshold
            base_prob = 1 - np.exp(-20 * excess_rally)
        
        # Amplify in bull regime
        if self.market_regime == MarketRegime.BULL:
            base_prob *= self.params.fomo_multiplier_bull
        
        # Missed opportunity pain
        if sentiment.reference_price > 0:
            missed_gain = (rally * sentiment.reference_price) / sentiment.reference_price
            if missed_gain > 0.05:  # Missed 5%+ move
                base_prob *= 1.3
        
        triggered = self.rng.random() < np.clip(base_prob, 0, 1)
        intensity = min((rally / threshold - 1) * self.params.fomo_intensity, 1.0) if triggered else 0.0
        
        if triggered:
            sentiment.greed_level = min(sentiment.greed_level + 0.2, 1.0)
            logger.debug(f"Agent {agent_id} FOMO triggered: intensity={intensity:.2f}")
        
        return triggered, intensity
    
    def apply_disposition_effect(
        self,
        agent_id: int,
        entry_price: float,
        current_price: float,
        holding_period: int
    ) -> float:
        """
        Calculate tendency to sell due to disposition effect.
        
        Returns probability of premature selling.
        """
        if agent_id not in self.agent_sentiments:
            return 0.0
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Disposition: sell winners too early, hold losers too long
        if pnl_pct > 0:  # Winner
            # Probability increases with gain size and holding period
            sell_prob = self.params.disposition_effect * (1 - np.exp(-pnl_pct * 10))
            sell_prob *= (1 + 0.1 * holding_period)  # More likely as time passes
        else:  # Loser
            # Reluctance to realize losses
            sell_prob = 0.1 * np.exp(pnl_pct * 5)  # Decreases with loss size
        
        return np.clip(sell_prob, 0, 0.9)
    
    def propagate_sentiment(
        self,
        adjacency_matrix: np.ndarray,
        initial_sentiments: np.ndarray
    ) -> np.ndarray:
        """
        Propagate sentiment through agent network using diffusion.
        
        Args:
            adjacency_matrix: NxN connectivity matrix
            initial_sentiments: Initial sentiment vector
            
        Returns:
            Updated sentiment vector after one diffusion step
        """
        n = len(initial_sentiments)
        if adjacency_matrix.shape != (n, n):
            raise ValueError("Adjacency matrix dimension mismatch")
        
        # Normalize adjacency matrix (row-stochastic)
        row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_adj = adjacency_matrix / row_sums
        
        # Diffusion step
        social_influence = normalized_adj @ initial_sentiments
        
        # Blend with own sentiment (stubbornness factor)
        stubbornness = 0.3
        updated = (1 - stubbornness) * social_influence + stubbornness * initial_sentiments
        
        return np.clip(updated, 0, 1)
    
    def update_agent_sentiment(
        self,
        agent_id: int,
        price_return: float,
        position_pnl: float
    ) -> None:
        """Update agent sentiment based on market action and PnL."""
        if agent_id not in self.agent_sentiments:
            return
        
        sentiment = self.agent_sentiments[agent_id]
        
        # Update fear/greed based on returns
        if price_return < -0.02:
            sentiment.fear_level = min(sentiment.fear_level + 0.15, 1.0)
            sentiment.greed_level = max(sentiment.greed_level - 0.1, 0.0)
        elif price_return > 0.02:
            sentiment.greed_level = min(sentiment.greed_level + 0.12, 1.0)
            sentiment.fear_level = max(sentiment.fear_level - 0.08, 0.0)
        
        # Decay towards neutral over time
        sentiment.fear_level *= 0.98
        sentiment.greed_level *= 0.98
        
        # Track PnL
        sentiment.add_pnl(position_pnl)
    
    def generate_behavioral_order_bias(
        self,
        agent_id: int,
        base_probability: float,
        market_context: Dict[str, float]
    ) -> float:
        """
        Generate final order probability adjusted for all behavioral biases.
        
        Args:
            agent_id: Target agent
            base_probability: Rational trading probability
            market_context: Current market state
            
        Returns:
            Behaviorally-adjusted probability
        """
        if agent_id not in self.agent_sentiments:
            return base_probability
        
        sentiment = self.agent_sentiments[agent_id]
        adjustment = 1.0
        
        # Herding adjustment
        buy_ratio = market_context.get('buy_ratio', 0.5)
        herding_pressure = self.calculate_herding_pressure(agent_id, buy_ratio, 
                                                           market_context.get('price_change', 0))
        adjustment += herding_pressure * 0.3
        
        # Fear/Greed adjustment
        adjustment -= sentiment.fear_level * 0.2
        adjustment += sentiment.greed_level * 0.15
        
        # Overconfidence adjustment
        if sentiment.confidence > 0.7:
            adjustment += self.params.overconfidence_bias
        
        # Anchoring adjustment (resistance to move away from reference)
        ref_price = sentiment.reference_price
        current_price = market_context.get('current_price', ref_price)
        if ref_price > 0:
            distance_from_anchor = abs(current_price - ref_price) / ref_price
            if distance_from_anchor > 0.05:
                adjustment *= (1 - 0.1 * np.exp(-distance_from_anchor * 5))
        
        return np.clip(base_probability * adjustment, 0, 1)
    
    def reset_agent_sentiment(self, agent_id: int) -> None:
        """Reset sentiment for an agent (e.g., after strategy change)."""
        if agent_id in self.agent_sentiments:
            self.agent_sentiments[agent_id].fear_level = 0.0
            self.agent_sentiments[agent_id].greed_level = 0.0
            self.agent_sentiments[agent_id].confidence = 0.5
    
    def get_aggregate_sentiment(self) -> Dict[str, float]:
        """Calculate market-wide aggregate sentiment metrics."""
        if not self.agent_sentiments:
            return {'fear': 0.0, 'greed': 0.0, 'confidence': 0.5}
        
        sentiments = list(self.agent_sentiments.values())
        return {
            'fear': np.mean([s.fear_level for s in sentiments]),
            'greed': np.mean([s.greed_level for s in sentiments]),
            'confidence': np.mean([s.confidence for s in sentiments]),
            'cumulative_pnl': np.mean([s.get_cumulative_pnl() for s in sentiments])
        }


# Example usage for testing
if __name__ == "__main__":
    engine = BehavioralFinanceEngine(seed=42)
    
    # Register test agents
    for i in range(100):
        engine.register_agent(i, 50000.0)
    
    # Simulate market stress
    returns = np.array([-0.03, -0.02, -0.04, -0.01, -0.05])
    regime = engine.update_market_regime(returns)
    print(f"Market regime: {regime}")
    
    # Test panic triggering
    panic_count = 0
    for agent_id in range(100):
        triggered, intensity = engine.trigger_panic_response(agent_id, -0.08)
        if triggered:
            panic_count += 1
    
    print(f"Panic triggered: {panic_count}/100 agents")
    print(f"Aggregate sentiment: {engine.get_aggregate_sentiment()}")
