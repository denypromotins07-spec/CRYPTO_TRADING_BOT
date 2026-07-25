#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
File: backend/microstructure/vpin_engine.py
Chapter 3: Order Book Imbalance, VPIN, and Toxicity Metrics

Purpose: Calculate Volume-Synchronized Probability of Informed Trading (VPIN)
Constraints: Must bucket trades by volume, not time; optimized for 8GB RAM
Target: AMD Ryzen AI 5 laptop

VPIN measures the probability that a trade is initiated by an informed trader.
High VPIN indicates toxic order flow and potential adverse selection.

Key insight: Bucket trades by equal volume, not equal time intervals.
This synchronizes the metric with actual trading activity.

Design Patterns: Strategy Pattern for bucketing algorithms
Type Hinting: Strict typing for production reliability
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from collections import deque
import numpy as np


class VPINSignal(Enum):
    """VPIN-based trading signals"""
    LOW_TOXICITY = "low_toxicity"       # Safe to trade
    MODERATE_TOXICITY = "moderate"      # Exercise caution
    HIGH_TOXICITY = "high_toxicity"     # Widen spreads significantly
    EXTREME_TOXICITY = "extreme"        # Consider pausing trading


@dataclass(slots=True)
class TradeBucket:
    """A bucket of trades with fixed volume"""
    bucket_id: int
    total_volume: float
    buy_volume: float
    sell_volume: float
    trade_count: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    
    @property
    def buy_ratio(self) -> float:
        """Ratio of buy volume to total volume"""
        if self.total_volume <= 0:
            return 0.5
        return self.buy_volume / self.total_volume
    
    @property
    def imbalance(self) -> float:
        """Absolute imbalance between buy and sell volume"""
        if self.total_volume <= 0:
            return 0.0
        return abs(self.buy_volume - self.sell_volume) / self.total_volume


@dataclass(slots=True)
class VPINSnapshot:
    """Current VPIN calculation result"""
    vpin_value: float
    signal: VPINSignal
    buckets_used: int
    average_bucket_volume: float
    timestamp_ns: int
    confidence_score: float
    recommended_spread_adjustment_bps: float


class VPINEngine:
    """
    Volume-Synchronized Probability of Informed Trading calculator.
    
    Implements the Easley, López de Prado, and O'Hara (2012) VPIN methodology
    with optimizations for real-time cryptocurrency trading.
    """
    
    # Default configuration
    DEFAULT_BUCKET_SIZE: float = 1000.0  # Volume units per bucket
    DEFAULT_NUM_BUCKETS: int = 50        # Number of buckets for VPIN calculation
    MIN_BUCKETS_FOR_SIGNAL: int = 10     # Minimum buckets needed for valid signal
    
    # Thresholds for signal generation
    VPIN_LOW_THRESHOLD: float = 0.2
    VPIN_MODERATE_THRESHOLD: float = 0.4
    VPIN_HIGH_THRESHOLD: float = 0.6
    VPIN_EXTREME_THRESHOLD: float = 0.8
    
    def __init__(self, bucket_size: float = DEFAULT_BUCKET_SIZE, 
                 num_buckets: int = DEFAULT_NUM_BUCKETS):
        # Volume bucketing
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        
        # Current incomplete bucket
        self.current_bucket_volume: float = 0.0
        self.current_bucket_buy_volume: float = 0.0
        self.current_bucket_sell_volume: float = 0.0
        self.current_bucket_trade_count: int = 0
        self.current_bucket_start_ns: Optional[int] = None
        
        # Completed buckets (circular buffer)
        self.completed_buckets: deque = deque(maxlen=num_buckets + 10)
        self.bucket_counter: int = 0
        
        # VPIN history for smoothing
        self.vpin_history: deque = deque(maxlen=100)
        
        # Timing
        self.last_update_ns: int = 0
        
        # Configuration
        self.enabled: bool = True
        self.use_smoothing: bool = True
        self.smoothing_alpha: float = 0.3  # Exponential smoothing factor
    
    def process_trade(self, price: float, volume: float, is_buy: bool, 
                      timestamp_ns: Optional[int] = None) -> Optional[VPINSnapshot]:
        """
        Process a single trade and update VPIN calculation.
        
        Trades are accumulated into volume buckets. When a bucket is full,
        it's sealed and added to the bucket history. VPIN is recalculated
        whenever a new bucket is completed.
        
        Returns a VPINSnapshot when a bucket completes and we have enough history.
        """
        if not self.enabled:
            return None
        
        import time
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        
        self.last_update_ns = timestamp_ns
        
        # Initialize bucket start time if needed
        if self.current_bucket_start_ns is None:
            self.current_bucket_start_ns = timestamp_ns
        
        # Add trade to current bucket
        self.current_bucket_volume += volume
        self.current_bucket_trade_count += 1
        
        if is_buy:
            self.current_bucket_buy_volume += volume
        else:
            self.current_bucket_sell_volume += volume
        
        # Check if bucket is complete
        if self.current_bucket_volume >= self.bucket_size:
            self._complete_bucket(timestamp_ns)
            
            # Calculate VPIN if we have enough buckets
            if len(self.completed_buckets) >= self.MIN_BUCKETS_FOR_SIGNAL:
                return self.calculate_vpin()
        
        return None
    
    def _complete_bucket(self, end_timestamp_ns: int) -> None:
        """Seal the current bucket and start a new one"""
        bucket = TradeBucket(
            bucket_id=self.bucket_counter,
            total_volume=self.current_bucket_volume,
            buy_volume=self.current_bucket_buy_volume,
            sell_volume=self.current_bucket_sell_volume,
            trade_count=self.current_bucket_trade_count,
            start_timestamp_ns=self.current_bucket_start_ns or 0,
            end_timestamp_ns=end_timestamp_ns
        )
        
        self.completed_buckets.append(bucket)
        self.bucket_counter += 1
        
        # Handle any excess volume (carry over to next bucket)
        excess_volume = self.current_bucket_volume - self.bucket_size
        
        # Start new bucket
        self.current_bucket_volume = excess_volume
        self.current_bucket_buy_volume = 0.0
        self.current_bucket_sell_volume = 0.0
        self.current_bucket_trade_count = 0
        self.current_bucket_start_ns = end_timestamp_ns
        
        # If there was excess, allocate it appropriately
        if excess_volume > 0:
            # Simplified: assume same buy/sell ratio as completed portion
            # More sophisticated implementations would track this precisely
            pass
    
    def calculate_vpin(self) -> VPINSnapshot:
        """
        Calculate VPIN from completed buckets.
        
        VPIN = (1/n) * sum(|buy_volume - sell_volume|) / total_volume
        across all buckets.
        
        Higher VPIN indicates more imbalanced order flow, suggesting
        informed trading and higher adverse selection risk.
        """
        if len(self.completed_buckets) < self.MIN_BUCKETS_FOR_SIGNAL:
            raise ValueError("Not enough buckets for VPIN calculation")
        
        # Use most recent N buckets
        recent_buckets = list(self.completed_buckets)[-self.num_buckets:]
        
        # Calculate sum of absolute imbalances
        total_imbalance = sum(b.imbalance * b.total_volume for b in recent_buckets)
        total_volume = sum(b.total_volume for b in recent_buckets)
        
        raw_vpin = total_imbalance / max(total_volume, 1e-10)
        
        # Apply exponential smoothing if enabled
        if self.use_smoothing and self.vpin_history:
            previous_vpin = self.vpin_history[-1]
            smoothed_vpin = self.smoothing_alpha * raw_vpin + (1 - self.smoothing_alpha) * previous_vpin
        else:
            smoothed_vpin = raw_vpin
        
        # Clamp to valid range
        vpin_value = max(0.0, min(1.0, smoothed_vpin))
        
        # Store in history
        self.vpin_history.append(vpin_value)
        
        # Determine signal
        signal = self._determine_signal(vpin_value)
        
        # Calculate confidence based on bucket consistency
        confidence = self._calculate_confidence(recent_buckets)
        
        # Recommend spread adjustment based on VPIN
        spread_adjustment = self._recommend_spread_adjustment(vpin_value)
        
        return VPINSnapshot(
            vpin_value=vpin_value,
            signal=signal,
            buckets_used=len(recent_buckets),
            average_bucket_volume=np.mean([b.total_volume for b in recent_buckets]),
            timestamp_ns=self.last_update_ns,
            confidence_score=confidence,
            recommended_spread_adjustment_bps=spread_adjustment
        )
    
    def _determine_signal(self, vpin_value: float) -> VPINSignal:
        """Map VPIN value to trading signal"""
        if vpin_value >= self.VPIN_EXTREME_THRESHOLD:
            return VPINSignal.EXTREME_TOXICITY
        elif vpin_value >= self.VPIN_HIGH_THRESHOLD:
            return VPINSignal.HIGH_TOXICITY
        elif vpin_value >= self.VPIN_MODERATE_THRESHOLD:
            return VPINSignal.MODERATE_TOXICITY
        else:
            return VPINSignal.LOW_TOXICITY
    
    def _calculate_confidence(self, buckets: List[TradeBucket]) -> float:
        """
        Calculate confidence in VPIN estimate.
        
        Confidence is higher when:
        1. More buckets available
        2. Bucket volumes are consistent
        3. Recent data is available
        """
        if len(buckets) < self.MIN_BUCKETS_FOR_SIGNAL:
            return 0.0
        
        # Base confidence from bucket count
        count_confidence = min(1.0, len(buckets) / self.num_buckets)
        
        # Consistency of bucket volumes
        volumes = [b.total_volume for b in buckets]
        volume_cv = np.std(volumes) / max(np.mean(volumes), 1e-10)  # Coefficient of variation
        consistency_confidence = max(0.0, 1.0 - volume_cv)
        
        # Weighted average
        confidence = 0.6 * count_confidence + 0.4 * consistency_confidence
        
        return min(1.0, confidence)
    
    def _recommend_spread_adjustment_bps(self, vpin_value: float) -> float:
        """
        Recommend spread adjustment in basis points based on VPIN.
        
        Higher VPIN = wider spreads to compensate for adverse selection risk.
        """
        # Linear scaling with VPIN
        base_adjustment = vpin_value * 50  # Up to 50 bps at VPIN=1.0
        
        # Additional penalty for extreme toxicity
        if vpin_value >= self.VPIN_EXTREME_THRESHOLD:
            base_adjustment += 25
        
        return min(base_adjustment, 100.0)  # Cap at 100 bps
    
    def get_toxic_flow_indicator(self) -> float:
        """
        Get a simplified toxic flow indicator.
        
        This is a faster, less precise version of VPIN for quick decisions.
        """
        if len(self.completed_buckets) < 5:
            return 0.5
        
        recent = list(self.completed_buckets)[-5:]
        avg_imbalance = np.mean([b.imbalance for b in recent])
        
        return avg_imbalance
    
    def reset(self) -> None:
        """Reset all state for fresh calculation"""
        self.current_bucket_volume = 0.0
        self.current_bucket_buy_volume = 0.0
        self.current_bucket_sell_volume = 0.0
        self.current_bucket_trade_count = 0
        self.current_bucket_start_ns = None
        self.completed_buckets.clear()
        self.bucket_counter = 0
        self.vpin_history.clear()
    
    def get_statistics(self) -> dict:
        """Get VPIN engine statistics"""
        if not self.completed_buckets:
            return {
                'buckets_completed': 0,
                'current_bucket_progress': 0.0,
                'vpin_current': None,
            }
        
        current_vpin = self.vpin_history[-1] if self.vpin_history else None
        
        return {
            'buckets_completed': len(self.completed_buckets),
            'current_bucket_progress': self.current_bucket_volume / self.bucket_size,
            'vpin_current': current_vpin,
            'average_bucket_volume': np.mean([b.total_volume for b in self.completed_buckets]),
            'total_trades_processed': sum(b.trade_count for b in self.completed_buckets),
        }


def main():
    """Example usage of VPIN engine"""
    engine = VPINEngine(bucket_size=500.0, num_buckets=30)
    
    import random
    import time
    
    print("Simulating trades for VPIN calculation...")
    
    # Simulate some trades
    base_price = 50000.0
    base_time = time.time_ns()
    
    last_snapshot = None
    
    for i in range(500):
        # Random price movement
        price = base_price + random.uniform(-50, 50)
        
        # Random volume
        volume = random.uniform(10, 100)
        
        # Slightly more buys than sells to create imbalance
        is_buy = random.random() < 0.6
        
        snapshot = engine.process_trade(price, volume, is_buy, base_time + i * 1_000_000)
        
        if snapshot:
            last_snapshot = snapshot
    
    if last_snapshot:
        print(f"\n=== VPIN ANALYSIS ===")
        print(f"VPIN Value: {last_snapshot.vpin_value:.4f}")
        print(f"Signal: {last_snapshot.signal.value}")
        print(f"Buckets Used: {last_snapshot.buckets_used}")
        print(f"Confidence: {last_snapshot.confidence_score:.2%}")
        print(f"Recommended Spread Adjustment: {last_snapshot.recommended_spread_adjustment_bps:.1f} bps")
    
    stats = engine.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    main()
