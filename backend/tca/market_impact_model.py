#!/usr/bin/env python3
"""
Market Impact Model: Bot's Own Footprint Estimation on L2 Order Book
Estimates the bot's trading footprint and scales penalties based on ADV.
Provides real-time market impact predictions for order sizing decisions.

Stage 13: Advanced Execution Algorithms
Target: Minimize market impact to secure 8k-20k INR/hour
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import threading


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class MarketImpactEstimate:
    """Estimated market impact for a proposed order"""
    order_size: float
    side: OrderSide
    estimated_impact_bps: float
    estimated_slippage_bps: float
    adv_consumption_pct: float
    recommended_slice_size: float
    confidence_score: float


@dataclass
class AssetProfile:
    """Trading profile for an asset"""
    symbol: str
    avg_daily_volume: float
    avg_daily_trades: int
    typical_spread_bps: float
    volatility_1h: float
    liquidity_score: float  # 0.0 to 1.0


class MarketImpactModel:
    """
    Estimates the bot's own footprint on the L2 order book.
    Scales penalties based on asset's average daily volume.
    
    Features:
    - Square-root market impact model
    - Volume participation rate tracking
    - Real-time ADV consumption monitoring
    - Dynamic slice size recommendations
    - Thread-safe state management
    """
    
    def __init__(
        self,
        impact_coefficient: float = 0.1,
        max_participation_rate: float = 0.10,  # 10% of ADV
        warning_participation_rate: float = 0.05,  # 5% threshold
        lookback_days: int = 30,
    ):
        self.impact_coefficient = impact_coefficient
        self.max_participation_rate = max_participation_rate
        self.warning_participation_rate = warning_participation_rate
        self.lookback_days = lookback_days
        
        # Asset profiles
        self._asset_profiles: Dict[str, AssetProfile] = {}
        
        # Recent trading activity by asset
        self._trading_history: Dict[str, List[Tuple[int, float]]] = {}  # symbol -> [(timestamp, volume)]
        
        # Current session tracking
        self._session_start_time: int = 0
        self._session_volume_by_asset: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def register_asset(
        self,
        symbol: str,
        avg_daily_volume: float,
        avg_daily_trades: int = 100000,
        typical_spread_bps: float = 5.0,
        volatility_1h: float = 0.02,
        liquidity_score: float = 0.9,
    ) -> None:
        """Register or update an asset's trading profile"""
        with self._lock:
            self._asset_profiles[symbol] = AssetProfile(
                symbol=symbol,
                avg_daily_volume=avg_daily_volume,
                avg_daily_trades=avg_daily_trades,
                typical_spread_bps=typical_spread_bps,
                volatility_1h=volatility_1h,
                liquidity_score=liquidity_score,
            )
            self._trading_history[symbol] = []
            self._session_volume_by_asset[symbol] = 0.0
    
    def start_session(self) -> None:
        """Start a new trading session"""
        with self._lock:
            self._session_start_time = time.time_ns()
            self._session_volume_by_asset = {symbol: 0.0 for symbol in self._asset_profiles}
    
    def record_trade(self, symbol: str, volume: float, side: OrderSide) -> None:
        """Record a trade for impact tracking"""
        with self._lock:
            timestamp = time.time_ns()
            
            # Update session volume
            if symbol in self._session_volume_by_asset:
                self._session_volume_by_asset[symbol] += volume
            
            # Update trading history
            if symbol not in self._trading_history:
                self._trading_history[symbol] = []
            
            self._trading_history[symbol].append((timestamp, volume))
            
            # Prune old history (keep last 24 hours)
            cutoff_ns = timestamp - (24 * 60 * 60 * 1_000_000_000)
            self._trading_history[symbol] = [
                (ts, vol) for ts, vol in self._trading_history[symbol]
                if ts > cutoff_ns
            ]
    
    def estimate_impact(
        self,
        symbol: str,
        order_size: float,
        side: OrderSide,
    ) -> Optional[MarketImpactEstimate]:
        """
        Estimate market impact for a proposed order.
        Uses square-root model: Impact = coefficient * sqrt(order_size / ADV)
        """
        with self._lock:
            if symbol not in self._asset_profiles:
                return None
            
            profile = self._asset_profiles[symbol]
            
            # Calculate participation rate
            participation_rate = order_size / profile.avg_daily_volume if profile.avg_daily_volume > 0 else 0
            
            # Check if exceeds limits
            if participation_rate > self.max_participation_rate:
                # Order too large - would consume too much ADV
                pass  # Still calculate but flag in result
            
            # Square-root market impact model
            # Base impact scaled by liquidity score
            base_impact = self.impact_coefficient * np.sqrt(order_size / max(profile.avg_daily_volume, 1))
            
            # Adjust for liquidity
            liquidity_adjustment = 1.0 / max(profile.liquidity_score, 0.1)
            adjusted_impact_bps = base_impact * liquidity_adjustment * 10000  # Convert to bps
            
            # Add spread cost estimate
            spread_cost_bps = profile.typical_spread_bps * 0.5  # Assume half-spread capture
            
            # Total estimated slippage
            total_slippage_bps = adjusted_impact_bps + spread_cost_bps
            
            # Calculate recommended slice size
            recommended_slice = profile.avg_daily_volume * self.warning_participation_rate
            if order_size > recommended_slice:
                recommended_slice = min(recommended_slice, order_size / 10)
            else:
                recommended_slice = order_size
            
            # Confidence score based on model uncertainty
            confidence = max(0.5, 1.0 - (participation_rate * 2))  # Lower confidence for large orders
            
            return MarketImpactEstimate(
                order_size=order_size,
                side=side,
                estimated_impact_bps=adjusted_impact_bps,
                estimated_slippage_bps=total_slippage_bps,
                adv_consumption_pct=participation_rate * 100,
                recommended_slice_size=recommended_slice,
                confidence_score=confidence,
            )
    
    def get_current_participation_rate(self, symbol: str) -> Optional[float]:
        """Get current session participation rate for an asset"""
        with self._lock:
            if symbol not in self._asset_profiles:
                return None
            
            profile = self._asset_profiles[symbol]
            session_volume = self._session_volume_by_asset.get(symbol, 0.0)
            
            # Estimate daily volume pro-rated by time elapsed
            if self._session_start_time == 0:
                return 0.0
            
            elapsed_hours = (time.time_ns() - self._session_start_time) / (60 * 60 * 1_000_000_000)
            projected_daily_volume = session_volume / max(elapsed_hours / 24, 0.01)
            
            return projected_daily_volume / profile.avg_daily_volume if profile.avg_daily_volume > 0 else 0
    
    def is_within_limits(self, symbol: str, order_size: float) -> bool:
        """Check if an order is within acceptable participation limits"""
        with self._lock:
            if symbol not in self._asset_profiles:
                return True
            
            profile = self._asset_profiles[symbol]
            current_session_volume = self._session_volume_by_asset.get(symbol, 0.0)
            
            # Project total session volume including this order
            elapsed_hours = max((time.time_ns() - self._session_start_time) / (60 * 60 * 1_000_000_000), 0.01)
            projected_total = ((current_session_volume + order_size) / elapsed_hours) * 24
            
            participation_rate = projected_total / profile.avg_daily_volume
            
            return participation_rate <= self.max_participation_rate
    
    def get_penalty_multiplier(self, symbol: str, order_size: float) -> float:
        """
        Get penalty multiplier for order sizing based on market impact.
        Higher multiplier = more aggressive penalty against large orders.
        """
        with self._lock:
            if symbol not in self._asset_profiles:
                return 1.0
            
            profile = self._asset_profiles[symbol]
            participation_rate = order_size / profile.avg_daily_volume if profile.avg_daily_volume > 0 else 0
            
            # Penalty increases exponentially with participation rate
            if participation_rate < 0.01:
                return 1.0
            elif participation_rate < 0.05:
                return 1.0 + (participation_rate * 10)
            elif participation_rate < 0.10:
                return 1.5 + ((participation_rate - 0.05) * 20)
            else:
                return 2.5 + ((participation_rate - 0.10) * 50)
    
    def get_optimal_execution_schedule(
        self,
        symbol: str,
        total_quantity: float,
        max_duration_minutes: int = 60,
    ) -> Optional[List[Dict]]:
        """
        Generate optimal execution schedule to minimize market impact.
        Returns list of slices with timing recommendations.
        """
        with self._lock:
            if symbol not in self._asset_profiles:
                return None
            
            profile = self._asset_profiles[symbol]
            
            # Calculate number of slices needed
            max_slice_size = profile.avg_daily_volume * self.warning_participation_rate
            num_slices = int(np.ceil(total_quantity / max_slice_size))
            
            # Ensure reasonable number of slices
            num_slices = max(1, min(num_slices, max_duration_minutes))
            
            slice_size = total_quantity / num_slices
            interval_seconds = (max_duration_minutes * 60) / num_slices
            
            schedule = []
            for i in range(num_slices):
                # Add some randomization to avoid predictable patterns
                jitter = np.random.uniform(-0.1, 0.1) * interval_seconds
                
                schedule.append({
                    "slice_id": i,
                    "quantity": slice_size,
                    "scheduled_offset_seconds": i * interval_seconds + jitter,
                    "estimated_impact_bps": self.estimate_impact(symbol, slice_size, OrderSide.BUY).estimated_impact_bps if self.estimate_impact(symbol, slice_size, OrderSide.BUY) else 0,
                })
            
            return schedule
    
    def reset(self) -> None:
        """Reset session tracking"""
        with self._lock:
            self._session_start_time = 0
            self._session_volume_by_asset = {symbol: 0.0 for symbol in self._asset_profiles}


if __name__ == "__main__":
    # Test market impact model
    model = MarketImpactModel(
        impact_coefficient=0.1,
        max_participation_rate=0.10,
    )
    
    # Register BTC with typical stats
    model.register_asset(
        symbol="BTCUSDT",
        avg_daily_volume=1000000000,  # $1B daily volume
        avg_daily_trades=500000,
        typical_spread_bps=5.0,
        volatility_1h=0.02,
        liquidity_score=0.95,
    )
    
    # Start session
    model.start_session()
    
    # Estimate impact for various order sizes
    test_sizes = [10000, 100000, 1000000, 10000000]
    
    print("Market Impact Analysis for BTCUSDT:")
    print("-" * 60)
    
    for size in test_sizes:
        estimate = model.estimate_impact("BTCUSDT", size, OrderSide.BUY)
        if estimate:
            print(f"\nOrder Size: ${size:,.0f}")
            print(f"  Estimated Impact: {estimate.estimated_impact_bps:.2f} bps")
            print(f"  Total Slippage: {estimate.estimated_slippage_bps:.2f} bps")
            print(f"  ADV Consumption: {estimate.adv_consumption_pct:.2f}%")
            print(f"  Recommended Slice: ${estimate.recommended_slice_size:,.0f}")
            print(f"  Confidence: {estimate.confidence_score:.2f}")
    
    # Test participation rate tracking
    model.record_trade("BTCUSDT", 5000000, OrderSide.BUY)
    model.record_trade("BTCUSDT", 3000000, OrderSide.SELL)
    
    participation = model.get_current_participation_rate("BTCUSDT")
    print(f"\nCurrent Participation Rate: {participation:.2%}" if participation else "N/A")
    
    # Check limits
    is_ok = model.is_within_limits("BTCUSDT", 50000000)
    print(f"Within limits for $50M order: {is_ok}")
    
    # Get optimal schedule
    schedule = model.get_optimal_execution_schedule("BTCUSDT", 100000000, max_duration_minutes=30)
    if schedule:
        print(f"\nOptimal Execution Schedule for $100M:")
        for slice_info in schedule[:5]:  # Show first 5 slices
            print(f"  Slice {slice_info['slice_id']}: ${slice_info['quantity']:,.0f} @ +{slice_info['scheduled_offset_seconds']:.0f}s")
