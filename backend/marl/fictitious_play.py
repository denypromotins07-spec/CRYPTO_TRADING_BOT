#!/usr/bin/env python3
"""
Fictitious Play Implementation for Opponent Strategy Learning

This module implements fictitious play algorithm to track historical 
opponent strategies and exploit predictable patterns in market behavior.

Features:
- Historical frequency tracking of opponent actions
- Best response computation against empirical distributions
- Convergence detection for stationary opponents
- Adaptive forgetting factor for non-stationary environments
- Memory-efficient counting structures

Integrates quantitative finance domains:
- Game theory and learning in games
- Behavioral finance (pattern exploitation)
- Statistical inference
- Time series analysis
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray


class ActionType(Enum):
    """Types of market actions that can be tracked."""
    AGGRESSIVE_BUY = auto()
    AGGRESSIVE_SELL = auto()
    PASSIVE_PROVIDE = auto()
    PASSIVE_CANCEL = auto()
    MOMENTUM_FOLLOW = auto()
    CONTRARIAN = auto()
    SPOOF = auto()
    LAYER = auto()


@dataclass
class ActionFrequency:
    """Tracks frequency of a specific action type."""
    count: int = 0
    total_value: float = 0.0
    last_seen_timestamp: int = 0
    recent_counts: List[int] = field(default_factory=list)
    
    @property
    def mean_value(self) -> float:
        """Get mean value of this action."""
        return self.total_value / max(self.count, 1)
    
    def update(self, value: float, timestamp: int, window_size: int = 100) -> None:
        """Update frequency with new observation."""
        self.count += 1
        self.total_value += value
        self.last_seen_timestamp = timestamp
        
        # Track recent counts for trend detection
        self.recent_counts.append(1)
        if len(self.recent_counts) > window_size:
            self.recent_counts.pop(0)
    
    def get_recent_frequency(self, window_size: int = 50) -> float:
        """Get frequency over recent window."""
        if not self.recent_counts:
            return 0.0
        recent = self.recent_counts[-window_size:]
        return sum(recent) / len(recent)


@dataclass
class OpponentProfile:
    """Complete profile of an opponent's strategy."""
    opponent_id: str
    action_frequencies: Dict[ActionType, ActionFrequency] = field(default_factory=dict)
    total_actions: int = 0
    first_seen: int = 0
    last_seen: int = 0
    detected_patterns: List[str] = field(default_factory=list)
    
    def get_empirical_distribution(self) -> Dict[ActionType, float]:
        """Compute empirical probability distribution over actions."""
        if self.total_actions == 0:
            return {}
        
        distribution = {}
        for action_type, freq in self.action_frequencies.items():
            distribution[action_type] = freq.count / self.total_actions
        
        return distribution
    
    def get_weighted_distribution(
        self,
        forget_factor: float = 0.95
    ) -> Dict[ActionType, float]:
        """
        Compute exponentially weighted distribution.
        
        Args:
            forget_factor: Discount factor for older observations (0-1)
            
        Returns:
            Weighted probability distribution
        """
        if not self.action_frequencies:
            return {}
        
        weights = {}
        current_time = self.last_seen
        
        for action_type, freq in self.action_frequencies.items():
            # Exponential decay based on recency
            time_diff = current_time - freq.last_seen_timestamp
            weight = (forget_factor ** time_diff) * freq.count
            weights[action_type] = weight
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {}
        
        return {k: v / total_weight for k, v in weights.items()}


class FictitiousPlay:
    """
    Fictitious play learner for exploiting predictable opponents.
    
    This class implements the classical fictitious play algorithm where
    each player assumes opponents are playing according to their historical
    frequency distribution and plays best response.
    
    Memory footprint: ~2MB per tracked opponent
    """
    
    def __init__(
        self,
        forget_factor: float = 0.98,
        min_observations: int = 30,
        confidence_threshold: float = 0.7
    ):
        self.opponents: Dict[str, OpponentProfile] = {}
        self.forget_factor = forget_factor
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold
        
        # Payoff matrix for best response computation
        self.payoff_matrix: NDArray[np.float32] = np.zeros((8, 8), dtype=np.float32)
        self._initialize_payoffs()
        
        # Convergence tracking
        self.convergence_history: Dict[str, List[float]] = defaultdict(list)
        self.convergence_window = 50
    
    def _initialize_payoffs(self) -> None:
        """Initialize payoff matrix for action interactions."""
        # Simplified payoff structure (can be learned)
        # Rows: our actions, Columns: opponent actions
        payoffs = [
            # AggBuy, AggSell, Passive, Cancel, Momentum, Contrarian, Spoof, Layer
            [ 0.0,   0.3,   0.1,    -0.1,   0.2,     -0.2,      0.4,   0.3 ],  # AggBuy
            [ 0.3,   0.0,   0.1,    -0.1,   -0.2,    0.2,       0.4,   0.3 ],  # AggSell
            [ 0.1,   0.1,   0.0,    0.05,   0.0,     0.0,       -0.1,  0.0 ],  # Passive
            [-0.1,  -0.1,   0.05,   0.0,    -0.05,  -0.05,      0.1,   0.1 ],  # Cancel
            [ 0.2,  -0.2,   0.0,    -0.05,  0.0,     0.1,       0.15,  0.1 ],  # Momentum
            [-0.2,   0.2,   0.0,    -0.05,  0.1,     0.0,       0.1,   0.15],  # Contrarian
            [ 0.4,   0.4,  -0.1,     0.1,   0.15,    0.1,       0.0,   0.2 ],  # Spoof
            [ 0.3,   0.3,   0.0,     0.1,   0.1,     0.15,      0.2,   0.0 ],  # Layer
        ]
        self.payoff_matrix = np.array(payoffs, dtype=np.float32)
    
    def observe_action(
        self,
        opponent_id: str,
        action: ActionType,
        value: float = 1.0,
        timestamp: int = 0
    ) -> None:
        """Record observed action from opponent."""
        if opponent_id not in self.opponents:
            self.opponents[opponent_id] = OpponentProfile(
                opponent_id=opponent_id,
                first_seen=timestamp
            )
        
        profile = self.opponents[opponent_id]
        
        if action not in profile.action_frequencies:
            profile.action_frequencies[action] = ActionFrequency()
        
        profile.action_frequencies[action].update(value, timestamp)
        profile.total_actions += 1
        profile.last_seen_timestamp = timestamp
        
        # Update pattern detection
        self._detect_patterns(profile)
    
    def _detect_patterns(self, profile: OpponentProfile) -> None:
        """Detect recurring patterns in opponent behavior."""
        patterns = []
        
        # Check for momentum following
        if len(profile.action_frequencies) >= 2:
            agg_buy = profile.action_frequencies.get(ActionType.AGGRESSIVE_BUY, ActionFrequency())
            momentum = profile.action_frequencies.get(ActionType.MOMENTUM_FOLLOW, ActionFrequency())
            
            if momentum.count > agg_buy.count * 0.5:
                patterns.append("momentum_follower")
        
        # Check for contrarian behavior
        contrarian = profile.action_frequencies.get(ActionType.CONTRARIAN, ActionFrequency())
        if contrarian.count > profile.total_actions * 0.3:
            patterns.append("contrarian")
        
        # Check for spoofing tendency
        spoof = profile.action_frequencies.get(ActionType.SPOOF, ActionFrequency())
        if spoof.count > profile.total_actions * 0.2:
            patterns.append("potential_spoofer")
        
        profile.detected_patterns = patterns
    
    def get_best_response(self, opponent_id: str) -> Optional[ActionType]:
        """
        Compute best response action against opponent's historical play.
        
        Returns:
            Best response action type, or None if insufficient data
        """
        if opponent_id not in self.opponents:
            return None
        
        profile = self.opponents[opponent_id]
        
        # Check minimum observations
        if profile.total_actions < self.min_observations:
            return None
        
        # Get opponent's empirical distribution
        distribution = profile.get_weighted_distribution(self.forget_factor)
        
        if not distribution:
            return None
        
        # Convert distribution to vector
        opp_vector = np.zeros(8, dtype=np.float32)
        for action_type, prob in distribution.items():
            idx = list(ActionType).index(action_type)
            opp_vector[idx] = prob
        
        # Compute expected payoffs for each of our actions
        expected_payoffs = self.payoff_matrix @ opp_vector
        
        # Return action with highest expected payoff
        best_idx = int(np.argmax(expected_payoffs))
        return list(ActionType)[best_idx]
    
    def get_exploitability(self, opponent_id: str) -> float:
        """
        Measure how exploitable the opponent is.
        
        Returns:
            Exploitability score (0-1), higher means more exploitable
        """
        if opponent_id not in self.opponents:
            return 0.0
        
        profile = self.opponents[opponent_id]
        
        if profile.total_actions < self.min_observations:
            return 0.0
        
        distribution = profile.get_weighted_distribution(self.forget_factor)
        
        if not distribution:
            return 0.0
        
        # Exploitability = variance in distribution (predictable = exploitable)
        probs = list(distribution.values())
        if len(probs) < 2:
            return 0.0
        
        variance = np.var(probs)
        max_variance = 0.25  # Maximum variance for binary distribution
        return min(1.0, variance / max_variance)
    
    def check_convergence(self, opponent_id: str) -> bool:
        """
        Check if opponent's strategy has converged (become stationary).
        
        Returns:
            True if strategy appears stationary
        """
        if opponent_id not in self.opponents:
            return False
        
        profile = self.opponents[opponent_id]
        
        if profile.total_actions < self.convergence_window * 2:
            return False
        
        # Compare recent vs older distribution
        recent_dist = profile.get_weighted_distribution(0.9)  # More weight on recent
        older_dist = profile.get_weighted_distribution(0.7)   # More balanced
        
        if not recent_dist or not older_dist:
            return False
        
        # Compute KL divergence approximation
        all_actions = set(recent_dist.keys()) | set(older_dist.keys())
        kl_div = 0.0
        
        for action in all_actions:
            p = recent_dist.get(action, 1e-10)
            q = older_dist.get(action, 1e-10)
            if p > 1e-10 and q > 1e-10:
                kl_div += p * np.log(p / q)
        
        # Store in history
        self.convergence_history[opponent_id].append(kl_div)
        if len(self.convergence_history[opponent_id]) > self.convergence_window:
            self.convergence_history[opponent_id].pop(0)
        
        # Converged if KL divergence consistently low
        if len(self.convergence_history[opponent_id]) >= self.convergence_window // 2:
            recent_kl = np.mean(self.convergence_history[opponent_id][-10:])
            return recent_kl < 0.1
        
        return False
    
    def get_opponent_summary(self, opponent_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary of opponent's strategy."""
        if opponent_id not in self.opponents:
            return None
        
        profile = self.opponents[opponent_id]
        distribution = profile.get_weighted_distribution(self.forget_factor)
        
        return {
            "opponent_id": opponent_id,
            "total_actions": profile.total_actions,
            "dominant_action": max(distribution.items(), key=lambda x: x[1])[0] if distribution else None,
            "exploitability": self.get_exploitability(opponent_id),
            "converged": self.check_convergence(opponent_id),
            "patterns": profile.detected_patterns,
            "distribution": {k.name: v for k, v in distribution.items()}
        }
    
    def clear_opponent(self, opponent_id: str) -> bool:
        """Remove opponent from tracking."""
        if opponent_id in self.opponents:
            del self.opponents[opponent_id]
            if opponent_id in self.convergence_history:
                del self.convergence_history[opponent_id]
            return True
        return False
    
    def reset(self) -> None:
        """Clear all opponent tracking."""
        self.opponents.clear()
        self.convergence_history.clear()


def analyze_market_maker_behavior(
    actions: List[Tuple[str, ActionType, float]],
    timestamps: List[int]
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to analyze market maker behavior.
    
    Args:
        actions: List of (opponent_id, action_type, value) tuples
        timestamps: Corresponding timestamps
        
    Returns:
        Dictionary of opponent summaries
    """
    fp = FictitiousPlay()
    
    for (opp_id, action, value), ts in zip(actions, timestamps):
        fp.observe_action(opp_id, action, value, ts)
    
    summaries = {}
    for opp_id in fp.opponents:
        summary = fp.get_opponent_summary(opp_id)
        if summary:
            summaries[opp_id] = summary
    
    return summaries


if __name__ == "__main__":
    # Example usage
    import random
    
    fp = FictitiousPlay()
    
    # Simulate observing a momentum-following opponent
    opponent_actions = [
        ActionType.MOMENTUM_FOLLOW,
        ActionType.AGGRESSIVE_BUY,
        ActionType.MOMENTUM_FOLLOW,
        ActionType.AGGRESSIVE_SELL,
        ActionType.MOMENTUM_FOLLOW,
    ]
    
    for i, action in enumerate(opponent_actions * 20):  # Repeat to build history
        value = random.uniform(0.5, 2.0)
        fp.observe_action("MM1", action, value, i)
    
    # Get best response
    best_response = fp.get_best_response("MM1")
    print(f"Best response to MM1: {best_response}")
    
    # Get exploitability
    exploitability = fp.get_exploitability("MM1")
    print(f"Exploitability score: {exploitability:.3f}")
    
    # Get summary
    summary = fp.get_opponent_summary("MM1")
    if summary:
        print(f"\nOpponent Summary:")
        print(f"  Total actions: {summary['total_actions']}")
        print(f"  Dominant action: {summary['dominant_action']}")
        print(f"  Patterns: {summary['patterns']}")
        print(f"  Converged: {summary['converged']}")
