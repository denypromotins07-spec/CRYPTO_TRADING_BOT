#!/usr/bin/env python3
"""
Poisson Tracker for Market Shock Detection

This module estimates the arrival rate (lambda) of sudden market shocks
using Poisson process theory. Critical for the ZAID bot to detect when
jump intensity exceeds historical norms, signaling elevated tail risk.

Features:
- Online lambda estimation with exponential weighting
- Historical percentile calculation for anomaly detection
- Multi-timescale analysis (short/medium/long term)
- Real-time alerting when intensity exceeds 99th percentile
- Separation of upward and downward shock intensities

Designed for 8GB RAM constraint with efficient streaming updates.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class ShockEvent:
    """Represents a detected market shock."""
    timestamp: int
    magnitude: float  # Absolute log return
    direction: int    # +1 for up, -1 for down
    is_extreme: bool  # Beyond 3-sigma


@dataclass
class IntensityEstimate:
    """Container for jump intensity estimates."""
    lambda_short: float   # Recent intensity (fast response)
    lambda_medium: float  # Medium-term intensity
    lambda_long: float    # Long-term baseline
    percentile_99: float  # 99th percentile threshold
    current_z_score: float  # How many std devs above mean
    
    @property
    def is_elevated(self) -> bool:
        """Check if current intensity exceeds 99th percentile."""
        return self.lambda_short > self.percentile_99
    
    @property
    def elevation_ratio(self) -> float:
        """Ratio of current to baseline intensity."""
        if self.lambda_long < 1e-10:
            return 1.0
        return self.lambda_short / self.lambda_long


class PoissonTracker:
    """
    Tracks and estimates Poisson arrival rates for market shocks.
    
    This implementation uses multiple estimation windows to capture
    both transient spikes and persistent changes in jump intensity.
    
    Mathematical Background:
    - For a Poisson process, inter-arrival times are exponentially distributed
    - MLE for lambda = n / T (events per unit time)
    - Exponential weighting gives more importance to recent events
    """
    
    def __init__(self,
                 short_window_seconds: float = 3600,      # 1 hour
                 medium_window_seconds: float = 86400,     # 24 hours
                 long_window_seconds: float = 604800,      # 7 days
                 shock_threshold_sigma: float = 3.0,
                 decay_factor: float = 0.99):
        """
        Initialize the Poisson tracker.
        
        Args:
            short_window_seconds: Window for short-term intensity
            medium_window_seconds: Window for medium-term intensity
            long_window_seconds: Window for long-term baseline
            shock_threshold_sigma: Number of std devs to define a shock
            decay_factor: Exponential decay factor (0-1) for weighting
        """
        self.short_window = short_window_seconds
        self.medium_window = medium_window_seconds
        self.long_window = long_window_seconds
        self.shock_threshold_sigma = shock_threshold_sigma
        self.decay_factor = decay_factor
        
        # Event storage (timestamps in seconds)
        self.all_shocks: Deque[ShockEvent] = deque(maxlen=10000)
        self.upward_shocks: Deque[ShockEvent] = deque(maxlen=10000)
        self.downward_shocks: Deque[ShockEvent] = deque(maxlen=10000)
        
        # Running statistics for online estimation
        self.weighted_event_count: float = 0.0
        self.weighted_time: float = 0.0
        
        # Historical lambda values for percentile calculation
        self.lambda_history: Deque[float] = deque(maxlen=10000)
        
        # Baseline statistics
        self.baseline_lambda: float = 0.0
        self.baseline_std: float = 0.0
        
        # Current time tracking
        self.current_time: Optional[int] = None
        self.last_update_time: Optional[int] = None
        
        # Alert state
        self.alert_active: bool = False
        self.alert_timestamps: List[int] = []
    
    def update_market_state(self, 
                            timestamp: int, 
                            price: float, 
                            prev_price: float) -> Optional[ShockEvent]:
        """
        Update with new price and detect if a shock occurred.
        
        Args:
            timestamp: Current timestamp (nanoseconds)
            price: Current price
            prev_price: Previous price
            
        Returns:
            ShockEvent if a shock was detected, None otherwise
        """
        if prev_price <= 0 or price <= 0:
            return None
        
        # Calculate log return
        log_ret = math.log(price / prev_price)
        abs_ret = abs(log_ret)
        
        # Update current time
        self.current_time = timestamp // 1_000_000_000  # Convert to seconds
        
        # Simple volatility estimate for shock detection
        # In production, would use a more sophisticated estimator
        recent_vol = self._estimate_recent_volatility()
        threshold = self.shock_threshold_sigma * max(recent_vol, 0.001)
        
        # Detect shock
        if abs_ret > threshold:
            direction = 1 if log_ret > 0 else -1
            is_extreme = abs_ret > (self.shock_threshold_sigma * 2) * max(recent_vol, 0.001)
            
            shock = ShockEvent(
                timestamp=self.current_time,
                magnitude=abs_ret,
                direction=direction,
                is_extreme=is_extreme
            )
            
            self._record_shock(shock)
            return shock
        
        return None
    
    def _estimate_recent_volatility(self) -> float:
        """Estimate recent volatility from shock magnitudes."""
        if len(self.all_shocks) < 5:
            return 0.01  # Default assumption
        
        # Use RMS of recent shock magnitudes
        recent_mags = [s.magnitude for s in list(self.all_shocks)[-20:]]
        if not recent_mags:
            return 0.01
        
        return math.sqrt(sum(m**2 for m in recent_mags) / len(recent_mags))
    
    def _record_shock(self, shock: ShockEvent):
        """Record a shock event and update statistics."""
        self.all_shocks.append(shock)
        
        if shock.direction > 0:
            self.upward_shocks.append(shock)
        else:
            self.downward_shocks.append(shock)
        
        # Update weighted counts with exponential decay
        self._update_weighted_counts()
        
        # Recalculate intensity estimates
        self._update_intensity_estimates()
    
    def _update_weighted_counts(self):
        """Update exponentially weighted event counts."""
        if self.current_time is None or len(self.all_shocks) == 0:
            return
        
        # Decay previous weights
        if self.last_update_time is not None:
            dt = self.current_time - self.last_update_time
            if dt > 0:
                decay = self.decay_factor ** (dt / 3600)  # Hourly decay
                self.weighted_event_count *= decay
                self.weighted_time *= decay
        
        # Add new event with weight 1
        self.weighted_event_count += 1.0
        self.weighted_time += 1.0  # Normalized time unit
        
        self.last_update_time = self.current_time
    
    def _update_intensity_estimates(self):
        """Update all intensity estimates and percentiles."""
        if self.current_time is None:
            return
        
        # Calculate lambda for different windows
        lambda_short = self._calculate_intensity(self.short_window)
        lambda_medium = self._calculate_intensity(self.medium_window)
        lambda_long = self._calculate_intensity(self.long_window)
        
        # Store in history for percentile calculation
        current_lambda = lambda_short
        self.lambda_history.append(current_lambda)
        
        # Update baseline statistics
        if len(self.lambda_history) >= 100:
            history_array = np.array(self.lambda_history)
            self.baseline_lambda = np.mean(history_array)
            self.baseline_std = np.std(history_array)
    
    def _calculate_intensity(self, window_seconds: float) -> float:
        """
        Calculate jump intensity within a time window.
        
        Uses exponential weighting for smoother estimates.
        
        Args:
            window_seconds: Size of the time window
            
        Returns:
            Estimated lambda (events per second)
        """
        if self.current_time is None or len(self.all_shocks) == 0:
            return 0.0
        
        window_start = self.current_time - window_seconds
        
        # Count events in window with exponential weighting
        weighted_count = 0.0
        for shock in reversed(self.all_shocks):
            if shock.timestamp < window_start:
                break
            
            # Weight by recency
            age = self.current_time - shock.timestamp
            weight = math.exp(-age / (window_seconds / 5))  # Decay over 1/5 of window
            weighted_count += weight
        
        return weighted_count / window_seconds
    
    def get_intensity_estimate(self) -> Optional[IntensityEstimate]:
        """
        Get comprehensive intensity estimates.
        
        Returns:
            IntensityEstimate object or None if insufficient data
        """
        if len(self.lambda_history) < 10:
            return None
        
        lambda_short = self._calculate_intensity(self.short_window)
        lambda_medium = self._calculate_intensity(self.medium_window)
        lambda_long = self._calculate_intensity(self.long_window)
        
        # Calculate 99th percentile from history
        history_array = np.array(self.lambda_history)
        percentile_99 = float(np.percentile(history_array, 99))
        
        # Calculate z-score relative to baseline
        if self.baseline_std > 1e-10:
            z_score = (lambda_short - self.baseline_lambda) / self.baseline_std
        else:
            z_score = 0.0
        
        estimate = IntensityEstimate(
            lambda_short=lambda_short,
            lambda_medium=lambda_medium,
            lambda_long=lambda_long,
            percentile_99=percentile_99,
            current_z_score=z_score
        )
        
        # Check for alerts
        if estimate.is_elevated and not self.alert_active:
            self._trigger_alert()
        elif not estimate.is_elevated:
            self.alert_active = False
        
        return estimate
    
    def _trigger_alert(self):
        """Trigger an alert for elevated jump intensity."""
        self.alert_active = True
        if self.current_time is not None:
            self.alert_timestamps.append(self.current_time)
    
    def get_separate_intensities(self) -> Tuple[float, float]:
        """
        Get separate intensity estimates for upward and downward shocks.
        
        Returns:
            Tuple of (upward_lambda, downward_lambda)
        """
        if self.current_time is None:
            return 0.0, 0.0
        
        def calc_directional_intensity(shocks: Deque[ShockEvent]) -> float:
            if len(shocks) == 0:
                return 0.0
            
            weighted_count = 0.0
            window_start = self.current_time - self.short_window
            
            for shock in reversed(shocks):
                if shock.timestamp < window_start:
                    break
                age = self.current_time - shock.timestamp
                weight = math.exp(-age / (self.short_window / 5))
                weighted_count += weight
            
            return weighted_count / self.short_window
        
        return (calc_directional_intensity(self.upward_shocks),
                calc_directional_intensity(self.downward_shocks))
    
    def get_expected_next_arrival(self) -> Optional[float]:
        """
        Calculate expected time until next shock.
        
        For a Poisson process, E[T] = 1/lambda
        
        Returns:
            Expected seconds until next shock, or None if no data
        """
        estimate = self.get_intensity_estimate()
        if estimate is None or estimate.lambda_short < 1e-10:
            return None
        
        return 1.0 / estimate.lambda_short
    
    def get_shock_statistics(self) -> Dict:
        """Get comprehensive statistics about shock behavior."""
        if len(self.all_shocks) == 0:
            return {"total_shocks": 0}
        
        now = self.current_time or 0
        
        # Time since last shock
        last_shock_age = now - self.all_shocks[-1].timestamp if self.all_shocks else float('inf')
        
        # Extreme shock count
        extreme_count = sum(1 for s in self.all_shocks if s.is_extreme)
        
        # Directional breakdown
        up_count = len(self.upward_shocks)
        down_count = len(self.downward_shocks)
        
        # Average magnitude
        avg_magnitude = np.mean([s.magnitude for s in self.all_shocks])
        max_magnitude = max(s.magnitude for s in self.all_shocks)
        
        return {
            "total_shocks": len(self.all_shocks),
            "extreme_shocks": extreme_count,
            "upward_shocks": up_count,
            "downward_shocks": down_count,
            "up_down_ratio": up_count / max(down_count, 1),
            "avg_magnitude": avg_magnitude,
            "max_magnitude": max_magnitude,
            "time_since_last_shock_sec": last_shock_age,
            "alerts_triggered": len(self.alert_timestamps),
        }
    
    def reset(self):
        """Clear all stored data."""
        self.all_shocks.clear()
        self.upward_shocks.clear()
        self.downward_shocks.clear()
        self.lambda_history.clear()
        self.weighted_event_count = 0.0
        self.weighted_time = 0.0
        self.baseline_lambda = 0.0
        self.baseline_std = 0.0
        self.alert_active = False
        self.alert_timestamps.clear()


# Example usage and testing
if __name__ == "__main__":
    print("Poisson Tracker Test - Market Shock Detection")
    print("=" * 50)
    
    # Create tracker
    tracker = PoissonTracker(
        short_window_seconds=3600,
        medium_window_seconds=86400,
        long_window_seconds=604800,
        shock_threshold_sigma=3.0
    )
    
    # Simulate price series with occasional shocks
    np.random.seed(42)
    base_price = 50000.0
    current_time = 0
    prev_price = base_price
    
    print("\nSimulating price series with shocks...")
    
    for i in range(10000):
        # Generate realistic returns
        if np.random.random() < 0.005:  # 0.5% chance of shock
            # Shock return (3+ sigma)
            shock_mag = np.random.uniform(0.03, 0.10)
            shock_dir = np.random.choice([-1, 1])
            ret = shock_dir * shock_mag
        else:
            # Normal return
            ret = np.random.normal(0, 0.001)
        
        current_price = prev_price * (1 + ret)
        
        # Update tracker
        shock = tracker.update_market_state(
            timestamp=current_time,
            price=current_price,
            prev_price=prev_price
        )
        
        if shock and i % 500 == 0:
            print(f"  t={i}: Shock detected! Magnitude={shock.magnitude:.4f}, "
                  f"Direction={'UP' if shock.direction > 0 else 'DOWN'}")
        
        prev_price = current_price
        current_time += 60  # 1 minute intervals
    
    # Get final estimates
    estimate = tracker.get_intensity_estimate()
    
    if estimate:
        print(f"\nIntensity Estimates:")
        print(f"  Short-term lambda: {estimate.lambda_short:.6f} per sec")
        print(f"  Medium-term lambda: {estimate.lambda_medium:.6f} per sec")
        print(f"  Long-term lambda:  {estimate.lambda_long:.6f} per sec")
        print(f"  99th Percentile:   {estimate.percentile_99:.6f}")
        print(f"  Z-score:           {estimate.current_z_score:.2f}")
        print(f"  Elevated Risk:     {'YES' if estimate.is_elevated else 'NO'}")
        print(f"  Elevation Ratio:   {estimate.elevation_ratio:.2f}x")
        
        expected_wait = tracker.get_expected_next_arrival()
        if expected_wait:
            print(f"\n  Expected time to next shock: {expected_wait/60:.1f} minutes")
    
    # Get statistics
    stats = tracker.get_shock_statistics()
    print(f"\nShock Statistics:")
    print(f"  Total Shocks:      {stats['total_shocks']}")
    print(f"  Extreme Shocks:    {stats['extreme_shocks']}")
    print(f"  Up/Down Ratio:     {stats['up_down_ratio']:.2f}")
    print(f"  Avg Magnitude:     {stats['avg_magnitude']:.4f}")
    print(f"  Max Magnitude:     {stats['max_magnitude']:.4f}")
    print(f"  Alerts Triggered:  {stats['alerts_triggered']}")
