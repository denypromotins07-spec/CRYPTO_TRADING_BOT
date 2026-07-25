#!/usr/bin/env python3
"""
VWAP Tracker: Historical Volume Profile Tracking for VWAP Targets
Tracks and analyzes historical volume profiles to execute orders along the VWAP curve.
Uses numpy C-extensions for high-performance calculations.

Stage 13: Advanced Execution Algorithms
Target: Minimize market impact to secure 8k-20k INR/hour
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class VolumeProfileBucket:
    """Single bucket in the volume profile"""
    timestamp_bucket: int  # Minute of day (0-1439)
    avg_volume: float
    avg_price: float
    trade_count: int
    vwap: float = 0.0
    
    def __post_init__(self):
        if self.avg_volume > 0 and self.avg_price > 0:
            self.vwap = self.avg_price  # Simplified; real VWAP needs PV/sum(V)


@dataclass
class VWAPExecutionState:
    """Current state of VWAP execution"""
    target_vwap: float
    current_vwap: float
    deviation_bps: float
    fill_progress: float
    time_progress: float
    next_target_volume: float


class VWAPTracker:
    """
    Real-time VWAP tracker with historical volume profile analysis.
    Ensures execution adheres strictly to the VWAP curve without detectable volume spikes.
    
    Features:
    - Historical volume profile construction (minute-level granularity)
    - Real-time VWAP calculation
    - Deviation monitoring and correction
    - Adaptive volume scheduling
    - Thread-safe state management
    """
    
    MINUTES_PER_DAY = 1440
    
    def __init__(
        self,
        history_days: int = 30,
        min_volume_threshold: float = 1000.0,
        max_deviation_bps: float = 50.0,
    ):
        self.history_days = history_days
        self.min_volume_threshold = min_volume_threshold
        self.max_deviation_bps = max_deviation_bps
        
        # Historical volume profile: minute_of_day -> list of (volume, price, pv)
        self._volume_profile: Dict[int, List[Tuple[float, float, float]]] = {
            i: [] for i in range(self.MINUTES_PER_DAY)
        }
        
        # Current session tracking
        self._session_start_time: Optional[int] = None
        self._session_pv: float = 0.0  # Price * Volume sum
        self._session_volume: float = 0.0
        self._session_trades: int = 0
        
        # Real-time VWAP
        self._current_vwap: float = 0.0
        self._target_vwap: Optional[float] = None
        
        # Execution tracking
        self._target_volume_profile: Optional[np.ndarray] = None
        self._executed_volume_profile: np.ndarray = np.zeros(self.MINUTES_PER_DAY, dtype=np.float64)
        self._total_target_volume: float = 0.0
        self._total_executed_volume: float = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Recent trades for smoothing
        self._recent_trades: deque = deque(maxlen=1000)
    
    def add_historical_trade(
        self, 
        minute_of_day: int, 
        volume: float, 
        price: float
    ) -> None:
        """Add historical trade data to build volume profile"""
        with self._lock:
            if minute_of_day < 0 or minute_of_day >= self.MINUTES_PER_DAY:
                return
            
            pv = price * volume
            self._volume_profile[minute_of_day].append((volume, price, pv))
            
            # Limit history size per bucket
            if len(self._volume_profile[minute_of_day]) > 10000:
                self._volume_profile[minute_of_day].pop(0)
    
    def build_volume_profile(self) -> np.ndarray:
        """
        Build expected volume profile from historical data.
        Returns normalized volume distribution (sums to 1.0).
        """
        with self._lock:
            profile = np.zeros(self.MINUTES_PER_DAY, dtype=np.float64)
            
            for minute in range(self.MINUTES_PER_DAY):
                trades = self._volume_profile[minute]
                if not trades:
                    continue
                
                total_volume = sum(t[0] for t in trades)
                if total_volume >= self.min_volume_threshold:
                    profile[minute] = total_volume / len(trades)  # Average volume per day
            
            # Normalize to sum to 1.0
            total = profile.sum()
            if total > 0:
                profile /= total
            
            return profile
    
    def start_session(
        self, 
        target_volume: float, 
        side: OrderSide,
        start_time_ns: Optional[int] = None
    ) -> None:
        """Start a new VWAP execution session"""
        with self._lock:
            self._session_start_time = start_time_ns or time.time_ns()
            self._session_pv = 0.0
            self._session_volume = 0.0
            self._session_trades = 0
            self._total_target_volume = target_volume
            self._total_executed_volume = 0.0
            self._executed_volume_profile.fill(0.0)
            
            # Build target volume profile
            self._target_volume_profile = self.build_volume_profile()
    
    def record_trade(self, price: float, volume: float) -> None:
        """Record a trade in the current session"""
        with self._lock:
            if self._session_start_time is None:
                return
            
            pv = price * volume
            self._session_pv += pv
            self._session_volume += volume
            self._session_trades += 1
            
            # Update current VWAP
            if self._session_volume > 0:
                self._current_vwap = self._session_pv / self._session_volume
            
            # Track recent trades for analysis
            self._recent_trades.append({
                'timestamp_ns': time.time_ns(),
                'price': price,
                'volume': volume,
                'pv': pv,
            })
    
    def record_execution(self, minute_of_day: int, volume: float) -> None:
        """Record executed volume for a specific minute bucket"""
        with self._lock:
            if 0 <= minute_of_day < self.MINUTES_PER_DAY:
                self._executed_volume_profile[minute_of_day] += volume
                self._total_executed_volume += volume
    
    def get_current_vwap(self) -> float:
        """Get current session VWAP"""
        with self._lock:
            return self._current_vwap
    
    def get_vwap_deviation_bps(self, market_vwap: float) -> float:
        """Calculate deviation from market VWAP in basis points"""
        with self._lock:
            if self._current_vwap == 0 or market_vwap == 0:
                return 0.0
            
            deviation = ((self._current_vwap - market_vwap) / market_vwap) * 10000
            return deviation
    
    def get_execution_state(self, current_minute: int, market_vwap: float) -> Optional[VWAPExecutionState]:
        """Get current VWAP execution state"""
        with self._lock:
            if self._target_volume_profile is None or self._total_target_volume == 0:
                return None
            
            # Calculate time progress
            elapsed_minutes = current_minute
            if self._session_start_time:
                elapsed_ns = time.time_ns() - self._session_start_time
                elapsed_minutes = int(elapsed_ns / 60_000_000_000)  # ns to minutes
            
            time_progress = min(1.0, elapsed_minutes / self.MINUTES_PER_DAY)
            
            # Calculate expected volume by now
            cumulative_target = self._target_volume_profile[:current_minute + 1].sum()
            expected_volume = cumulative_target * self._total_target_volume
            
            # Calculate fill progress
            fill_progress = self._total_executed_volume / self._total_target_volume if self._total_target_volume > 0 else 0.0
            
            # Calculate deviation
            deviation_bps = self.get_vwap_deviation_bps(market_vwap)
            
            # Calculate next target volume
            next_target = 0.0
            if current_minute + 1 < self.MINUTES_PER_DAY:
                next_bucket_ratio = self._target_volume_profile[current_minute + 1]
                next_target = next_bucket_ratio * self._total_target_volume
            
            return VWAPExecutionState(
                target_vwap=self._target_vwap or market_vwap,
                current_vwap=self._current_vwap,
                deviation_bps=deviation_bps,
                fill_progress=fill_progress,
                time_progress=time_progress,
                next_target_volume=next_target,
            )
    
    def check_volume_spike(self, current_volume: float, window_minutes: int = 5) -> bool:
        """
        Check if current execution would cause detectable volume spike.
        Compares against historical average for the time window.
        """
        with self._lock:
            if self._target_volume_profile is None:
                return False
            
            current_minute = int((time.time_ns() - (self._session_start_time or 0)) / 60_000_000_000) % self.MINUTES_PER_DAY
            
            # Get historical average for window
            hist_avg = 0.0
            count = 0
            for offset in range(-window_minutes // 2, window_minutes // 2 + 1):
                bucket = (current_minute + offset) % self.MINUTES_PER_DAY
                hist_avg += self._target_volume_profile[bucket]
                count += 1
            
            if count == 0 or hist_avg == 0:
                return False
            
            hist_avg /= count
            
            # Check if current volume exceeds 2x historical average
            return current_volume > (hist_avg * 2.0)
    
    def get_adaptive_slice_volume(
        self, 
        remaining_volume: float,
        time_remaining_minutes: int,
        deviation_bps: float
    ) -> float:
        """
        Calculate adaptive volume slice to correct VWAP deviation.
        Increases volume if behind schedule, decreases if ahead.
        """
        with self._lock:
            if time_remaining_minutes <= 0:
                return remaining_volume
            
            # Base slice
            base_slice = remaining_volume / time_remaining_minutes
            
            # Adjustment factor based on deviation
            if abs(deviation_bps) > self.max_deviation_bps:
                # Need to catch up - increase volume
                adjustment = 1.0 + (abs(deviation_bps) / 100.0)
                adjustment = min(adjustment, 3.0)  # Cap at 3x
            else:
                adjustment = 1.0
            
            return base_slice * adjustment
    
    def reset(self) -> None:
        """Reset tracker state"""
        with self._lock:
            self._session_start_time = None
            self._session_pv = 0.0
            self._session_volume = 0.0
            self._session_trades = 0
            self._current_vwap = 0.0
            self._target_vwap = None
            self._target_volume_profile = None
            self._executed_volume_profile.fill(0.0)
            self._total_target_volume = 0.0
            self._total_executed_volume = 0.0


class VWAPCurveAnalyzer:
    """
    Analyzes VWAP curve characteristics for optimal execution scheduling.
    Uses numpy for fast statistical analysis.
    """
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self._daily_vwaps: List[np.ndarray] = []
    
    def add_daily_vwap(self, minute_vwaps: np.ndarray) -> None:
        """Add daily VWAP curve for analysis"""
        if len(minute_vwaps) != 1440:
            raise ValueError("Expected 1440 minute VWAP values")
        self._daily_vwaps.append(minute_vwaps.copy())
        if len(self._daily_vwaps) > self.lookback_days:
            self._daily_vwaps.pop(0)
    
    def get_expected_vwap_curve(self) -> Optional[np.ndarray]:
        """Get expected VWAP curve from historical average"""
        if not self._daily_vwaps:
            return None
        
        return np.mean(self._daily_vwaps, axis=0)
    
    def get_vwap_volatility(self) -> Optional[np.ndarray]:
        """Get VWAP volatility profile throughout the day"""
        if not self._daily_vwaps:
            return None
        
        return np.std(self._daily_vwaps, axis=0)
    
    def identify_high_volume_periods(self, threshold_percentile: float = 75.0) -> List[Tuple[int, int]]:
        """Identify high volume periods for optimal execution"""
        if not self._daily_vwaps:
            return []
        
        avg_profile = np.mean([np.diff(v) for v in self._daily_vwaps], axis=0)
        threshold = np.percentile(avg_profile, threshold_percentile)
        
        periods = []
        in_period = False
        start_minute = 0
        
        for i, vol in enumerate(avg_profile):
            if vol >= threshold and not in_period:
                in_period = True
                start_minute = i
            elif vol < threshold and in_period:
                in_period = False
                periods.append((start_minute, i))
        
        if in_period:
            periods.append((start_minute, len(avg_profile)))
        
        return periods


if __name__ == "__main__":
    # Test VWAP tracker
    tracker = VWAPTracker(history_days=30)
    
    # Add some historical data
    for day in range(30):
        for minute in range(1440):
            # Simulate higher volume during market hours
            volume = 10000 if 9 * 60 <= minute <= 16 * 60 else 1000
            price = 50000 + np.random.randn() * 100
            tracker.add_historical_trade(minute, volume, price)
    
    # Build profile
    profile = tracker.build_volume_profile()
    print(f"Volume profile built: {profile.sum():.4f} (should be ~1.0)")
    
    # Start session
    tracker.start_session(target_volume=100000, side=OrderSide.BUY)
    
    # Simulate trades
    for i in range(100):
        price = 50000 + np.random.randn() * 50
        volume = 100 + np.random.rand() * 200
        tracker.record_trade(price, volume)
        tracker.record_execution(i % 1440, volume)
    
    # Get execution state
    state = tracker.get_execution_state(60, 50000.0)
    if state:
        print(f"Current VWAP: {state.current_vwap:.2f}")
        print(f"Deviation: {state.deviation_bps:.2f} bps")
        print(f"Fill progress: {state.fill_progress:.2%}")
