#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
VPIN Engine Module
Calculates Volume-Synchronized Probability of Informed Trading
Buckets trades by volume rather than chronological time for accurate VPIN
Optimized for 8GB RAM with strict type hinting
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum
from collections import deque
import time
import numpy as np


class TradeClassification(Enum):
    """Classification of trade initiation."""
    BUY_INITIATED = "buy"
    SELL_INITIATED = "sell"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class VolumeBucket:
    """
    Represents a volume-synchronized bucket for VPIN calculation.
    Uses __slots__ for memory efficiency.
    """
    bucket_id: int
    total_volume: float
    buy_volume: float
    sell_volume: float
    trade_count: int = 0
    timestamp_start_ns: int = 0
    timestamp_end_ns: int = 0
    price_start: float = 0.0
    price_end: float = 0.0
    
    @property
    def imbalance(self) -> float:
        """Calculate volume imbalance within bucket."""
        if self.total_volume == 0:
            return 0.0
        return abs(self.buy_volume - self.sell_volume) / self.total_volume


@dataclass(slots=True)
class VPINSnapshot:
    """Snapshot of VPIN calculation results."""
    timestamp_ns: int
    vpin_value: float
    bucket_count: int
    avg_bucket_imbalance: float
    toxicity_level: str
    recommended_spread_adjustment_bps: float
    informed_trading_probability: float


class VPINEngine:
    """
    Calculates Volume-Synchronized Probability of Informed Trading (VPIN).
    
    VPIN measures the probability that trades are initiated by informed traders.
    High VPIN indicates potential adverse selection and toxic order flow.
    
    Key features:
    - Buckets trades by volume (not time) for accurate synchronization
    - Uses tick rule for trade classification
    - Implements Easley-Kiefer-O'Hara-Paperman (EKOP) methodology
    - Dynamically adjusts spread based on VPIN level
    """
    
    # Default bucket size (adjust based on asset liquidity)
    DEFAULT_BUCKET_SIZE = 1000.0  # Volume units per bucket
    
    # Number of buckets for rolling VPIN calculation
    DEFAULT_BUCKET_COUNT = 50
    
    # VPIN thresholds for toxicity levels
    LOW_TOXICITY_THRESHOLD = 0.2
    MEDIUM_TOXICITY_THRESHOLD = 0.4
    HIGH_TOXICITY_THRESHOLD = 0.6
    EXTREME_TOXICITY_THRESHOLD = 0.8
    
    def __init__(self, bucket_size: float = None, bucket_count: int = None):
        """
        Initialize VPIN engine.
        
        Args:
            bucket_size: Target volume per bucket
            bucket_count: Number of buckets for rolling calculation
        """
        self.bucket_size = bucket_size or self.DEFAULT_BUCKET_SIZE
        self.bucket_count = bucket_count or self.DEFAULT_BUCKET_COUNT
        
        # Current accumulating bucket
        self.current_bucket: Optional[VolumeBucket] = None
        self.current_bucket_buy_vol: float = 0.0
        self.current_bucket_sell_vol: float = 0.0
        
        # Completed buckets for VPIN calculation
        self.buckets: Deque[VolumeBucket] = deque(maxlen=self.bucket_count)
        
        # Trade history for classification
        self.recent_trades: Deque[Tuple[float, float, int]] = deque(maxlen=100)  # (price, volume, timestamp)
        
        # Last price for tick rule
        self.last_price: Optional[float] = None
        self.last_trade_direction: int = 0  # +1 (buy), -1 (sell), 0 (unknown)
        
        # Statistics
        self.stats = {
            'total_buckets': 0,
            'total_volume_processed': 0.0,
            'avg_vpin': 0.0,
            'high_toxicity_events': 0,
        }
        
        # VPIN smoothing parameter
        self.vpin_smoothing_alpha = 0.1
        self.smoothed_vpin = 0.0
        
        # Last calculation timestamp
        self.last_calculation_ns: int = time.time_ns()
    
    def process_trade(self, price: float, volume: float, timestamp_ns: int = None) -> Optional[VPINSnapshot]:
        """
        Process a new trade and update VPIN calculation.
        
        Args:
            price: Trade price
            volume: Trade volume
            timestamp_ns: Trade timestamp in nanoseconds
            
        Returns:
            VPINSnapshot if bucket completed and VPIN calculated, None otherwise
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        
        # Classify trade using tick rule
        direction = self._classify_trade(price, timestamp_ns)
        
        # Update current bucket
        if self.current_bucket is None:
            self.current_bucket = VolumeBucket(
                bucket_id=len(self.buckets) + len(self.buckets),
                total_volume=0.0,
                buy_volume=0.0,
                sell_volume=0.0,
                timestamp_start_ns=timestamp_ns,
                price_start=price,
            )
        
        # Add volume to appropriate side
        if direction == 1:  # Buy initiated
            self.current_bucket.buy_volume += volume
        elif direction == -1:  # Sell initiated
            self.current_bucket.sell_volume += volume
        
        self.current_bucket.total_volume += volume
        self.current_bucket.trade_count += 1
        self.current_bucket.price_end = price
        self.current_bucket.timestamp_end_ns = timestamp_ns
        
        # Store recent trade for classification
        self.recent_trades.append((price, volume, timestamp_ns))
        
        # Check if bucket is full
        if self.current_bucket.total_volume >= self.bucket_size:
            # Complete current bucket
            completed_bucket = self.current_bucket
            self.buckets.append(completed_bucket)
            self.stats['total_buckets'] += 1
            self.stats['total_volume_processed'] += completed_bucket.total_volume
            
            # Reset current bucket
            self.current_bucket = None
            self.current_bucket_buy_vol = 0.0
            self.current_bucket_sell_vol = 0.0
            
            # Calculate VPIN if enough buckets
            if len(self.buckets) >= min(10, self.bucket_count // 5):
                return self.calculate_vpin()
        
        return None
    
    def _classify_trade(self, price: float, timestamp_ns: int) -> int:
        """
        Classify trade as buy or sell initiated using tick rule.
        
        Tick Rule:
        - If price > last_price: buy initiated (+1)
        - If price < last_price: sell initiated (-1)
        - If price == last_price: same direction as previous trade
        
        Returns:
            +1 for buy, -1 for sell, 0 for unknown
        """
        if self.last_price is None:
            self.last_price = price
            return 0
        
        if price > self.last_price:
            self.last_trade_direction = 1
        elif price < self.last_price:
            self.last_trade_direction = -1
        # else: keep last_trade_direction
        
        self.last_price = price
        return self.last_trade_direction
    
    def calculate_vpin(self) -> VPINSnapshot:
        """
        Calculate VPIN from accumulated buckets.
        
        VPIN = (1/n) * Σ|V_buy - V_sell| / (V_buy + V_sell)
        
        Returns:
            VPINSnapshot with current VPIN value and metrics
        """
        if len(self.buckets) == 0:
            return self._empty_snapshot()
        
        current_ns = time.time_ns()
        
        # Calculate sum of absolute imbalances
        total_imbalance_sum = 0.0
        total_volume_sum = 0.0
        imbalances = []
        
        for bucket in self.buckets:
            if bucket.total_volume > 0:
                imbalance = abs(bucket.buy_volume - bucket.sell_volume)
                total_imbalance_sum += imbalance
                total_volume_sum += bucket.total_volume
                imbalances.append(bucket.imbalance)
        
        if total_volume_sum == 0:
            return self._empty_snapshot()
        
        # Raw VPIN calculation
        raw_vpin = total_imbalance_sum / total_volume_sum
        
        # Apply smoothing
        self.smoothed_vpin = (
            self.vpin_smoothing_alpha * raw_vpin +
            (1 - self.vpin_smoothing_alpha) * self.smoothed_vpin
        )
        
        # Determine toxicity level
        toxicity_level = self._get_toxicity_level(self.smoothed_vpin)
        
        # Calculate recommended spread adjustment
        spread_adjustment = self._calculate_spread_adjustment(self.smoothed_vpin)
        
        # Estimate informed trading probability
        informed_prob = self._estimate_informed_trading_probability(toxicity_level, imbalances)
        
        # Update statistics
        self.stats['avg_vpin'] = (
            self.stats['avg_vpin'] * (self.stats['total_buckets'] - 1) + self.smoothed_vpin
        ) / self.stats['total_buckets']
        
        if self.smoothed_vpin > self.HIGH_TOXICITY_THRESHOLD:
            self.stats['high_toxicity_events'] += 1
        
        snapshot = VPINSnapshot(
            timestamp_ns=current_ns,
            vpin_value=self.smoothed_vpin,
            bucket_count=len(self.buckets),
            avg_bucket_imbalance=np.mean(imbalances) if imbalances else 0.0,
            toxicity_level=toxicity_level,
            recommended_spread_adjustment_bps=spread_adjustment,
            informed_trading_probability=informed_prob,
        )
        
        self.last_calculation_ns = current_ns
        return snapshot
    
    def get_current_vpin(self) -> float:
        """Get current smoothed VPIN value."""
        return self.smoothed_vpin
    
    def get_toxicity_level(self) -> str:
        """Get current toxicity level string."""
        return self._get_toxicity_level(self.smoothed_vpin)
    
    def should_widen_spread(self) -> bool:
        """Check if spread should be widened due to high VPIN."""
        return self.smoothed_vpin > self.MEDIUM_TOXICITY_THRESHOLD
    
    def get_spread_adjustment_bps(self) -> float:
        """Get recommended spread adjustment in basis points."""
        return self._calculate_spread_adjustment(self.smoothed_vpin)
    
    def get_stats(self) -> Dict:
        """Get VPIN engine statistics."""
        return self.stats.copy()
    
    def clear(self) -> None:
        """Clear all accumulated data."""
        self.current_bucket = None
        self.current_bucket_buy_vol = 0.0
        self.current_bucket_sell_vol = 0.0
        self.buckets.clear()
        self.recent_trades.clear()
        self.last_price = None
        self.last_trade_direction = 0
        self.smoothed_vpin = 0.0
        self.stats = {
            'total_buckets': 0,
            'total_volume_processed': 0.0,
            'avg_vpin': 0.0,
            'high_toxicity_events': 0,
        }
    
    def _empty_snapshot(self) -> VPINSnapshot:
        """Create empty VPIN snapshot."""
        return VPINSnapshot(
            timestamp_ns=time.time_ns(),
            vpin_value=0.0,
            bucket_count=0,
            avg_bucket_imbalance=0.0,
            toxicity_level="UNKNOWN",
            recommended_spread_adjustment_bps=0.0,
            informed_trading_probability=0.0,
        )
    
    def _get_toxicity_level(self, vpin: float) -> str:
        """Map VPIN value to toxicity level."""
        if vpin < self.LOW_TOXICITY_THRESHOLD:
            return "LOW"
        elif vpin < self.MEDIUM_TOXICITY_THRESHOLD:
            return "MEDIUM"
        elif vpin < self.HIGH_TOXICITY_THRESHOLD:
            return "HIGH"
        elif vpin < self.EXTREME_TOXICITY_THRESHOLD:
            return "EXTREME"
        else:
            return "CRITICAL"
    
    def _calculate_spread_adjustment_bps(self, vpin: float) -> float:
        """
        Calculate recommended spread adjustment based on VPIN.
        
        Higher VPIN indicates more informed trading, requiring wider spreads
        to compensate for adverse selection risk.
        """
        if vpin < self.LOW_TOXICITY_THRESHOLD:
            return 0.0
        elif vpin < self.MEDIUM_TOXICITY_THRESHOLD:
            return 1.0  # 1 bps
        elif vpin < self.HIGH_TOXICITY_THRESHOLD:
            return 3.0  # 3 bps
        elif vpin < self.EXTREME_TOXICITY_THRESHOLD:
            return 7.0  # 7 bps
        else:
            return 15.0  # 15 bps - consider halting trading
    
    def _estimate_informed_trading_probability(self, toxicity_level: str, 
                                                imbalances: List[float]) -> float:
        """
        Estimate probability of informed trading based on VPIN and patterns.
        
        Uses Bayesian updating based on observed order flow patterns.
        """
        # Base probability from VPIN
        base_prob = self.smoothed_vpin
        
        # Adjust for consistency of imbalances
        if imbalances:
            imbalance_std = np.std(imbalances)
            # Consistent high imbalances suggest informed trading
            if imbalance_std < 0.2 and np.mean(imbalances) > 0.5:
                base_prob = min(1.0, base_prob * 1.2)
        
        # Adjust for toxicity level
        level_multipliers = {
            "LOW": 0.8,
            "MEDIUM": 1.0,
            "HIGH": 1.3,
            "EXTREME": 1.5,
            "CRITICAL": 2.0,
            "UNKNOWN": 1.0,
        }
        
        adjusted_prob = base_prob * level_multipliers.get(toxicity_level, 1.0)
        return min(1.0, adjusted_prob)


if __name__ == "__main__":
    # Example usage and testing
    engine = VPINEngine(bucket_size=100.0, bucket_count=20)
    
    # Simulate trades with mixed initiation
    import random
    base_price = 50000.0
    
    print("Simulating trades for VPIN calculation...")
    
    for i in range(500):
        # Random walk price with some trend
        price_change = random.gauss(0, 0.5)
        if i % 50 < 25:  # Periodic buying pressure
            price_change += 0.3
        
        price = base_price + price_change
        volume = random.uniform(1.0, 20.0)
        
        snapshot = engine.process_trade(price, volume)
        
        if snapshot and i % 100 == 0:
            print(f"\nVPIN Snapshot at trade {i}:")
            print(f"  VPIN Value: {snapshot.vpin_value:.4f}")
            print(f"  Toxicity Level: {snapshot.toxicity_level}")
            print(f"  Spread Adjustment: {snapshot.recommended_spread_adjustment_bps:.1f} bps")
            print(f"  Informed Trading Prob: {snapshot.informed_trading_probability:.2%}")
    
    # Final statistics
    stats = engine.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total Buckets: {stats['total_buckets']}")
    print(f"  Total Volume: {stats['total_volume_processed']:.2f}")
    print(f"  Avg VPIN: {stats['avg_vpin']:.4f}")
    print(f"  High Toxicity Events: {stats['high_toxicity_events']}")
