#!/usr/bin/env python3
"""
VPIN Engine - Volume-Synchronized Probability of Informed Trading

This module calculates VPIN (Volume-Synchronized Probability of Informed Trading),
a key metric for detecting informed trading and adverse selection risk.
Unlike time-based metrics, VPIN buckets trades by volume for more accurate analysis.

Designed for the ZAID PERSONAL CRYPTO TRADING BOT with strict 8GB RAM constraints.
Implements efficient volume bucketing and Easley-O'Hara methodology.

Author: Opus 4.8
Stage: 21/100 - Advanced Market Microstructure
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Deque
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time


@dataclass(slots=True)
class TradeRecord:
    """Individual trade record."""
    price: float
    volume: float
    timestamp_ns: int
    is_buyer_maker: bool  # True = sell, False = buy
    
    @property
    def signed_volume(self) -> float:
        """Return signed volume (positive for buys, negative for sells)."""
        return self.volume if not self.is_buyer_maker else -self.volume


@dataclass(slots=True)
class VolumeBucket:
    """A bucket of trades with fixed volume."""
    trades: List[TradeRecord] = field(default_factory=list)
    total_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    start_time_ns: int = 0
    end_time_ns: int = 0
    
    def add_trade(self, trade: TradeRecord) -> None:
        """Add a trade to this bucket."""
        self.trades.append(trade)
        self.total_volume += trade.volume
        if trade.is_buyer_maker:
            self.sell_volume += trade.volume
        else:
            self.buy_volume += trade.volume
        
        if self.start_time_ns == 0:
            self.start_time_ns = trade.timestamp_ns
        self.end_time_ns = trade.timestamp_ns
    
    @property
    def is_complete(self) -> bool:
        """Check if bucket has sufficient volume."""
        return False  # Determined externally
    
    @property
    def imbalance(self) -> float:
        """Calculate volume imbalance within bucket."""
        if self.total_volume == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / self.total_volume
    
    def reset(self) -> None:
        """Reset bucket for reuse."""
        self.trades.clear()
        self.total_volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.start_time_ns = 0
        self.end_time_ns = 0


@dataclass(slots=True)
class VPINResult:
    """VPIN calculation result."""
    vpin_value: float
    bucket_count: int
    avg_imbalance: float
    std_imbalance: float
    toxicity_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    timestamp_ns: int
    recommended_spread_adjustment: float  # Basis points
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'vpin': self.vpin_value,
            'buckets': self.bucket_count,
            'avg_imbalance': self.avg_imbalance,
            'std_imbalance': self.std_imbalance,
            'toxicity': self.toxicity_level,
            'spread_adjustment_bps': self.recommended_spread_adjustment,
        }


class VPINEngine:
    """
    Volume-Synchronized Probability of Informed Trading calculator.
    
    VPIN measures the probability that a trade comes from an informed trader
    by analyzing order flow imbalances over volume buckets rather than time.
    
    High VPIN indicates:
    - Elevated adverse selection risk
    - Potential for toxic order flow
    - Need for wider spreads
    """
    
    # Default parameters
    DEFAULT_BUCKET_SIZE: float = 1000.0  # Volume units per bucket
    DEFAULT_NUM_BUCKETS: int = 50  # Number of buckets for VPIN calculation
    MIN_TRADES_PER_BUCKET: int = 5
    
    # Toxicity thresholds
    LOW_TOXICITY: float = 0.3
    MEDIUM_TOXICITY: float = 0.5
    HIGH_TOXICITY: float = 0.7
    
    def __init__(
        self,
        symbol: str,
        bucket_size: float = DEFAULT_BUCKET_SIZE,
        num_buckets: int = DEFAULT_NUM_BUCKETS,
    ) -> None:
        """
        Initialize VPIN engine.
        
        Args:
            symbol: Trading pair symbol
            bucket_size: Target volume per bucket
            num_buckets: Number of buckets to use in VPIN calculation
        """
        self.symbol: str = symbol
        self.bucket_size: float = bucket_size
        self.num_buckets: int = num_buckets
        
        # Current active bucket
        self._current_bucket: VolumeBucket = VolumeBucket()
        
        # Completed buckets (circular buffer)
        self._buckets: Deque[VolumeBucket] = deque(maxlen=num_buckets + 10)
        
        # All trades (for analysis)
        self._trades: Deque[TradeRecord] = deque(maxlen=10000)
        
        # VPIN history
        self._vpin_history: Deque[float] = deque(maxlen=100)
        
        # Statistics
        self._total_trades: int = 0
        self._total_volume: float = 0.0
        
        # Last VPIN calculation
        self._last_vpin: float = 0.0
        self._last_calculation_ns: int = 0
    
    @staticmethod
    def _now_ns() -> int:
        """Get current time in nanoseconds."""
        return time.time_ns()
    
    def add_trade(
        self,
        price: float,
        volume: float,
        is_buyer_maker: bool,
        timestamp_ns: Optional[int] = None,
    ) -> Optional[VPINResult]:
        """
        Add a trade and update VPIN calculation.
        
        Args:
            price: Trade price
            volume: Trade volume
            is_buyer_maker: True if buyer was maker (sell), False if taker (buy)
            timestamp_ns: Trade timestamp (defaults to now)
        
        Returns:
            VPINResult if a new VPIN was calculated, None otherwise
        """
        if timestamp_ns is None:
            timestamp_ns = self._now_ns()
        
        trade = TradeRecord(
            price=price,
            volume=volume,
            timestamp_ns=timestamp_ns,
            is_buyer_maker=is_buyer_maker,
        )
        
        self._trades.append(trade)
        self._total_trades += 1
        self._total_volume += volume
        
        # Add to current bucket
        self._current_bucket.add_trade(trade)
        
        # Check if bucket is complete
        if self._current_bucket.total_volume >= self.bucket_size:
            self._complete_bucket()
            
            # Calculate VPIN if we have enough buckets
            if len(self._buckets) >= self.num_buckets:
                return self.calculate_vpin()
        
        return None
    
    def _complete_bucket(self) -> None:
        """Move current bucket to completed buckets."""
        if self._current_bucket.total_volume == 0:
            return
        
        # Store completed bucket
        completed = VolumeBucket(
            trades=self._current_bucket.trades.copy(),
            total_volume=self._current_bucket.total_volume,
            buy_volume=self._current_bucket.buy_volume,
            sell_volume=self._current_bucket.sell_volume,
            start_time_ns=self._current_bucket.start_time_ns,
            end_time_ns=self._current_bucket.end_time_ns,
        )
        
        self._buckets.append(completed)
        
        # Reset current bucket
        self._current_bucket.reset()
    
    def calculate_vpin(self) -> VPINResult:
        """
        Calculate VPIN using Easley-O'Hara methodology.
        
        VPIN = (1/n) * sum(|V_buy - V_sell|) / (V_buy + V_sell)
        
        where the sum is over n volume buckets.
        
        Returns:
            VPINResult with calculated value and interpretation
        """
        if len(self._buckets) < self.num_buckets:
            return VPINResult(
                vpin_value=0.0,
                bucket_count=len(self._buckets),
                avg_imbalance=0.0,
                std_imbalance=0.0,
                toxicity_level='INSUFFICIENT_DATA',
                timestamp_ns=self._now_ns(),
                recommended_spread_adjustment=0.0,
            )
        
        # Get recent buckets
        recent_buckets = list(self._buckets)[-self.num_buckets:]
        
        # Calculate absolute imbalances
        abs_imbalances: List[float] = []
        for bucket in recent_buckets:
            if bucket.total_volume > 0:
                imbalance = abs(bucket.buy_volume - bucket.sell_volume) / bucket.total_volume
                abs_imbalances.append(imbalance)
        
        if not abs_imbalances:
            return VPINResult(
                vpin_value=0.0,
                bucket_count=len(recent_buckets),
                avg_imbalance=0.0,
                std_imbalance=0.0,
                toxicity_level='NO_IMBALANCE',
                timestamp_ns=self._now_ns(),
                recommended_spread_adjustment=0.0,
            )
        
        # VPIN is the average absolute imbalance
        vpin = np.mean(abs_imbalances)
        std_imbalance = np.std(abs_imbalances)
        avg_imbalance = np.mean([b.imbalance for b in recent_buckets])
        
        # Determine toxicity level
        if vpin >= self.HIGH_TOXICITY:
            toxicity = 'EXTREME'
        elif vpin >= self.MEDIUM_TOXICITY:
            toxicity = 'HIGH'
        elif vpin >= self.LOW_TOXICITY:
            toxicity = 'MEDIUM'
        else:
            toxicity = 'LOW'
        
        # Calculate recommended spread adjustment
        # Higher VPIN = more adverse selection = wider spread needed
        spread_adjustment = vpin * 10  # Basis points (e.g., VPIN=0.5 => 5bps)
        
        result = VPINResult(
            vpin_value=vpin,
            bucket_count=len(recent_buckets),
            avg_imbalance=avg_imbalance,
            std_imbalance=std_imbalance,
            toxicity_level=toxicity,
            timestamp_ns=self._now_ns(),
            recommended_spread_adjustment=spread_adjustment,
        )
        
        # Store in history
        self._vpin_history.append(vpin)
        self._last_vpin = vpin
        self._last_calculation_ns = self._now_ns()
        
        return result
    
    def get_current_vpin(self) -> float:
        """Get the most recent VPIN value."""
        return self._last_vpin
    
    def get_vpin_trend(self, periods: int = 10) -> float:
        """
        Calculate VPIN trend over recent periods.
        
        Positive trend = increasing toxicity
        Negative trend = decreasing toxicity
        """
        if len(self._vpin_history) < 2:
            return 0.0
        
        recent = list(self._vpin_history)[-periods:]
        if len(recent) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
    
    def is_toxic(self, threshold: float = MEDIUM_TOXICITY) -> bool:
        """Check if current order flow is toxic."""
        return self._last_vpin >= threshold
    
    def get_dynamic_spread_multiplier(self) -> float:
        """
        Get spread multiplier based on VPIN.
        
        Returns factor to multiply base spread by.
        Higher VPIN = higher multiplier = wider spreads.
        """
        if self._last_vpin < self.LOW_TOXICITY:
            return 1.0
        elif self._last_vpin < self.MEDIUM_TOXICITY:
            return 1.0 + (self._last_vpin - self.LOW_TOXICITY) * 0.5
        elif self._last_vpin < self.HIGH_TOXICITY:
            return 1.2 + (self._last_vpin - self.MEDIUM_TOXICITY) * 1.0
        else:
            return 1.5 + (self._last_vpin - self.HIGH_TOXICITY) * 2.0
    
    def get_statistics(self) -> Dict:
        """Get VPIN statistics."""
        if not self._vpin_history:
            return {
                'mean_vpin': 0.0,
                'std_vpin': 0.0,
                'max_vpin': 0.0,
                'min_vpin': 0.0,
                'percent_toxic': 0.0,
            }
        
        vpin_array = np.array(self._vpin_history)
        return {
            'mean_vpin': float(np.mean(vpin_array)),
            'std_vpin': float(np.std(vpin_array)),
            'max_vpin': float(np.max(vpin_array)),
            'min_vpin': float(np.min(vpin_array)),
            'percent_toxic': float(np.mean(vpin_array >= self.MEDIUM_TOXICITY) * 100),
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self._current_bucket.reset()
        self._buckets.clear()
        self._trades.clear()
        self._vpin_history.clear()
        self._total_trades = 0
        self._total_volume = 0.0
        self._last_vpin = 0.0
        self._last_calculation_ns = 0
    
    @property
    def bucket_count(self) -> int:
        """Get number of completed buckets."""
        return len(self._buckets)
    
    @property
    def trade_count(self) -> int:
        """Get total trade count."""
        return self._total_trades
    
    @property
    def total_volume(self) -> float:
        """Get total volume processed."""
        return self._total_volume


def main() -> None:
    """Example usage of VPINEngine."""
    engine = VPINEngine(symbol='BTCUSDT', bucket_size=100.0, num_buckets=30)
    
    import random
    random.seed(42)
    
    # Simulate trades with varying imbalance
    print("Simulating trades...")
    
    for i in range(500):
        # Generate realistic trade pattern
        base_price = 50000.0
        
        # Create periods of high imbalance
        if 100 <= i < 200:
            # High buy pressure period
            is_buyer_maker = random.random() > 0.7  # More buys
        elif 300 <= i < 400:
            # High sell pressure period
            is_buyer_maker = random.random() < 0.7  # More sells
        else:
            # Normal period
            is_buyer_maker = random.random() > 0.5
        
        price = base_price + random.uniform(-10, 10)
        volume = random.uniform(0.1, 5.0)
        
        result = engine.add_trade(price, volume, is_buyer_maker)
        
        if result and i % 50 == 0:
            print(f"\nTrade {i}: VPIN = {result.vpin_value:.3f} ({result.toxicity_level})")
            print(f"  Spread adjustment: {result.recommended_spread_adjustment:.1f} bps")
    
    # Print final statistics
    stats = engine.get_statistics()
    print(f"\n=== VPIN Statistics ===")
    print(f"Mean VPIN: {stats['mean_vpin']:.3f}")
    print(f"Std VPIN: {stats['std_vpin']:.3f}")
    print(f"Max VPIN: {stats['max_vpin']:.3f}")
    print(f"% Toxic: {stats['percent_toxic']:.1f}%")
    print(f"Current VPIN: {engine.get_current_vpin():.3f}")
    print(f"Spread Multiplier: {engine.get_dynamic_spread_multiplier():.2f}x")


if __name__ == "__main__":
    main()
