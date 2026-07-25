#!/usr/bin/env python3
"""
Realized Volatility and Bipower Variation Calculator

This module calculates high-frequency realized volatility measures from
tick-level data, providing more accurate volatility estimates than traditional
methods by utilizing the full information content of intraday prices.

Features:
- Realized Volatility (RV): Sum of squared intraday returns
- Bipower Variation (BV): Robust to jumps, captures continuous variation
- Jump detection: Separates continuous and jump components
- Microstructure noise correction
- Optimized for microsecond timestamp alignments

Designed for the ZAID bot's 8GB RAM constraint with efficient streaming.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Deque
from collections import deque
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class TickData:
    """Represents a single tick observation."""
    timestamp: int  # Microsecond precision timestamp
    price: float
    volume: float
    
    def log_return(self, prev_price: float) -> float:
        """Calculate log return from previous price."""
        if prev_price <= 0 or self.price <= 0:
            return 0.0
        return math.log(self.price / prev_price)


@dataclass
class VolatilityMeasures:
    """Container for various volatility measures."""
    realized_volatility: float
    bipower_variation: float
    jump_component: float
    continuous_component: float
    n_observations: int
    time_span_seconds: float
    
    @property
    def annualized_rv(self) -> float:
        """Annualize realized volatility (assuming crypto 24/7 trading)."""
        # Crypto trades 24/7 = 365 days * 24 hours * 3600 seconds
        seconds_per_year = 31536000
        return self.realized_volatility * math.sqrt(seconds_per_year / self.time_span_seconds)
    
    @property
    def jump_ratio(self) -> float:
        """Ratio of jump variation to total variation."""
        if self.realized_volatility < 1e-10:
            return 0.0
        return max(0.0, self.jump_component / self.realized_volatility)


class RealizedVolatilityCalculator:
    """
    Calculates realized volatility and related measures from high-frequency data.
    
    This implementation handles:
    - Irregular sampling intervals
    - Microstructure noise correction
    - Jump detection via bipower variation
    - Streaming updates for real-time computation
    
    Mathematical Background:
    - Realized Volatility: RV_t = sum(r_i^2) where r_i are intraday returns
    - Bipower Variation: BV_t = (pi/2) * sum(|r_i| * |r_{i-1}|)
    - Jump Component: J_t = RV_t - BV_t (if positive)
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 sampling_interval_ms: int = 100,
                 noise_correction: bool = True):
        """
        Initialize the realized volatility calculator.
        
        Args:
            window_size: Maximum number of ticks to keep in memory
            sampling_interval_ms: Target sampling interval in milliseconds
            noise_correction: Apply microstructure noise correction
        """
        self.window_size = window_size
        self.sampling_interval_ns = sampling_interval_ms * 1_000_000  # Convert to nanoseconds
        self.noise_correction = noise_correction
        
        # Tick buffer for streaming
        self.tick_buffer: Deque[TickData] = deque(maxlen=window_size)
        
        # Sampled returns (at regular intervals)
        self.sampled_returns: Deque[float] = deque(maxlen=window_size)
        
        # Last sampled price and time
        self.last_sampled_price: Optional[float] = None
        self.last_sampled_time: Optional[int] = None
        
        # Precomputed constants
        self.PI_HALF = math.pi / 2.0
        self.SQRT_PI_HALF = math.sqrt(math.pi / 2.0)
        
        # Statistics
        self.total_ticks_processed = 0
        self.last_calculation_time: Optional[int] = None
    
    def add_tick(self, tick: TickData) -> bool:
        """
        Add a new tick and update volatility calculations.
        
        Args:
            tick: New tick data
            
        Returns:
            True if this tick triggered a new sampled return
        """
        self.total_ticks_processed += 1
        self.tick_buffer.append(tick)
        
        # Check if we should sample a new return
        if self._should_sample(tick.timestamp):
            return self._sample_return(tick)
        
        return False
    
    def _should_sample(self, current_time: int) -> bool:
        """Determine if enough time has passed to sample a new return."""
        if self.last_sampled_time is None:
            return True
        
        return (current_time - self.last_sampled_time) >= self.sampling_interval_ns
    
    def _sample_return(self, tick: TickData) -> bool:
        """
        Sample a return at the current tick.
        
        Uses linear interpolation if the exact sampling time doesn't align
        with a tick (handles microsecond timestamp alignment).
        """
        if self.last_sampled_price is None:
            # First sample
            self.last_sampled_price = tick.price
            self.last_sampled_time = tick.timestamp
            return False
        
        # Simple sampling: use latest price
        # For production, could implement linear interpolation between ticks
        current_price = tick.price
        
        if current_price <= 0 or self.last_sampled_price <= 0:
            return False
        
        # Calculate log return
        log_ret = math.log(current_price / self.last_sampled_price)
        
        # Skip extreme outliers (likely data errors)
        if abs(log_ret) > 0.5:  # >50% return between samples
            return False
        
        self.sampled_returns.append(log_ret)
        
        # Update state
        self.last_sampled_price = current_price
        self.last_sampled_time = tick.timestamp
        self.last_calculation_time = tick.timestamp
        
        return True
    
    def calculate_realized_volatility(self, n_periods: Optional[int] = None) -> float:
        """
        Calculate realized volatility from sampled returns.
        
        RV = sqrt(sum(r_i^2))
        
        Args:
            n_periods: Number of periods to use (None = all available)
            
        Returns:
            Realized volatility (standard deviation over the period)
        """
        if len(self.sampled_returns) < 2:
            return 0.0
        
        returns = list(self.sampled_returns)[-n_periods:] if n_periods else list(self.sampled_returns)
        
        # Sum of squared returns
        rv_squared = sum(r * r for r in returns)
        
        # Apply microstructure noise correction if enabled
        if self.noise_correction and len(returns) > 10:
            rv_squared = self._correct_for_microstructure_noise(rv_squared, returns)
        
        return math.sqrt(max(0.0, rv_squared))
    
    def _correct_for_microstructure_noise(self, rv_squared: float, 
                                           returns: List[float]) -> float:
        """
        Correct realized volatility for microstructure noise.
        
        Uses the Zhang et al. (2005) estimator which subtracts an estimate
        of the noise variance based on the autocorrelation of returns.
        """
        n = len(returns)
        if n < 5:
            return rv_squared
        
        # Estimate first-order autocorrelation
        mean_ret = sum(returns) / n
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / n
        
        if var_ret < 1e-15:
            return rv_squared
        
        # Autocovariance at lag 1
        autocov = sum((returns[i] - mean_ret) * (returns[i+1] - mean_ret) 
                      for i in range(n-1)) / (n - 1)
        
        # Noise variance estimate (negative autocorrelation indicates noise)
        if autocov >= 0:
            return rv_squared
        
        # Correction term
        noise_correction = -2 * n * autocov
        
        return max(0.0, rv_squared - noise_correction)
    
    def calculate_bipower_variation(self, n_periods: Optional[int] = None) -> float:
        """
        Calculate bipower variation, robust to jumps.
        
        BV_t = (pi/2) * sum(|r_i| * |r_{i-1}|)
        
        Bipower variation isolates the continuous component of price variation
        by using the product of adjacent absolute returns.
        
        Args:
            n_periods: Number of periods to use
            
        Returns:
            Bipower variation
        """
        if len(self.sampled_returns) < 3:
            return 0.0
        
        returns = list(self.sampled_returns)[-n_periods:] if n_periods else list(self.sampled_returns)
        
        # Sum of products of adjacent absolute returns
        bpv_sum = sum(abs(returns[i]) * abs(returns[i+1]) 
                      for i in range(len(returns) - 1))
        
        return self.PI_HALF * bpv_sum
    
    def detect_jumps(self, n_periods: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Decompose total variation into continuous and jump components.
        
        Based on Barndorff-Nielsen and Shephard (2006):
        - Total variation (RV) = Continuous + Jump
        - Continuous ≈ Bipower Variation
        - Jump = max(0, RV - BV)
        
        Args:
            n_periods: Number of periods to analyze
            
        Returns:
            Tuple of (jump_component, continuous_component, total_rv)
        """
        rv = self.calculate_realized_volatility(n_periods)
        bv = self.calculate_bipower_variation(n_periods)
        
        # Jump component (only positive differences indicate jumps)
        jump = max(0.0, rv - bv)
        continuous = min(rv, bv)  # Continuous can't exceed total
        
        return jump, continuous, rv
    
    def get_volatility_measures(self, 
                                 n_periods: Optional[int] = None,
                                 time_window_seconds: Optional[float] = None) -> Optional[VolatilityMeasures]:
        """
        Get comprehensive volatility measures.
        
        Args:
            n_periods: Override number of periods
            time_window_seconds: Specify time window instead of period count
            
        Returns:
            VolatilityMeasures object or None if insufficient data
        """
        if len(self.sampled_returns) < 3:
            return None
        
        # Determine number of periods to use
        if time_window_seconds is not None:
            # Estimate based on sampling interval
            n_periods = int(time_window_seconds * 1000 / (self.sampling_interval_ns / 1_000_000))
        
        rv = self.calculate_realized_volatility(n_periods)
        bv = self.calculate_bipower_variation(n_periods)
        jump, continuous, _ = self.detect_jumps(n_periods)
        
        # Calculate time span
        if len(self.tick_buffer) >= 2:
            time_span_ns = self.tick_buffer[-1].timestamp - self.tick_buffer[0].timestamp
            time_span_seconds = time_span_ns / 1_000_000_000
        else:
            time_span_seconds = 0.0
        
        return VolatilityMeasures(
            realized_volatility=rv,
            bipower_variation=bv,
            jump_component=jump,
            continuous_component=continuous,
            n_observations=len(self.sampled_returns),
            time_span_seconds=max(time_span_seconds, 1e-10)
        )
    
    def get_jump_test_statistic(self, n_periods: Optional[int] = None) -> Optional[float]:
        """
        Calculate test statistic for presence of jumps.
        
        Based on Huang and Tauchen (2005):
        Z = (RV - BV) / sqrt(var(RV - BV))
        
        Under null of no jumps, Z ~ N(0,1)
        
        Args:
            n_periods: Number of periods
            
        Returns:
            Z-statistic or None if insufficient data
        """
        if len(self.sampled_returns) < 10:
            return None
        
        returns = list(self.sampled_returns)[-n_periods:] if n_periods else list(self.sampled_returns)
        n = len(returns)
        
        rv = self.calculate_realized_volatility(n_periods)
        bv = self.calculate_bipower_variation(n_periods)
        
        # Estimate asymptotic variance
        # Var(RV - BV) ≈ (pi^2/4 - 1) * sum(r_i^4)
        quarticity = sum(r ** 4 for r in returns)
        asymptotic_var = (math.pi ** 2 / 4 - 1) * quarticity
        
        if asymptotic_var < 1e-20:
            return None
        
        z_stat = (rv - bv) / math.sqrt(asymptotic_var)
        
        return z_stat
    
    def reset(self):
        """Clear all stored data and reset statistics."""
        self.tick_buffer.clear()
        self.sampled_returns.clear()
        self.last_sampled_price = None
        self.last_sampled_time = None
        self.total_ticks_processed = 0
        self.last_calculation_time = None


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Realized Volatility Calculator Test")
    print("=" * 50)
    
    # Create calculator with 100ms sampling
    calc = RealizedVolatilityCalculator(
        window_size=1000,
        sampling_interval_ms=100,
        noise_correction=True
    )
    
    # Simulate tick data with occasional jumps
    np.random.seed(42)
    base_price = 50000.0  # BTC-like price
    current_time = 0
    current_price = base_price
    
    print("\nSimulating high-frequency tick data...")
    
    for i in range(2000):
        # Generate realistic price movement
        if np.random.random() < 0.02:  # 2% chance of jump
            # Jump component
            jump = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.02)
            current_price *= (1 + jump)
        else:
            # Continuous diffusion
            ret = np.random.normal(0, 0.0005)
            current_price *= (1 + ret)
        
        # Add some noise
        current_price *= (1 + np.random.normal(0, 0.0001))
        
        tick = TickData(
            timestamp=current_time,
            price=current_price,
            volume=np.random.uniform(0.1, 10.0)
        )
        
        calc.add_tick(tick)
        current_time += np.random.randint(10_000_000, 100_000_000)  # 10-100ms in ns
    
    # Get results
    measures = calc.get_volatility_measures()
    
    if measures:
        print(f"\nVolatility Measures:")
        print(f"  Realized Volatility: {measures.realized_volatility:.6f}")
        print(f"  Bipower Variation:   {measures.bipower_variation:.6f}")
        print(f"  Jump Component:      {measures.jump_component:.6f}")
        print(f"  Continuous Component: {measures.continuous_component:.6f}")
        print(f"  Jump Ratio:          {measures.jump_ratio:.4f}")
        print(f"  Annualized RV:       {measures.annualized_rv:.4f}")
        print(f"  Observations:        {measures.n_observations}")
        print(f"  Time Span:           {measures.time_span_seconds:.2f}s")
        
        # Jump test
        z_stat = calc.get_jump_test_statistic()
        if z_stat:
            significant = abs(z_stat) > 2.576  # 99% confidence
            print(f"\nJump Test:")
            print(f"  Z-statistic: {z_stat:.4f}")
            print(f"  Significant jumps detected: {'YES' if significant else 'NO'}")
    
    print(f"\nTotal ticks processed: {calc.total_ticks_processed}")
    print(f"Sampled returns: {len(calc.sampled_returns)}")
