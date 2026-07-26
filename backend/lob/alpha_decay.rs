#!/usr/bin/env python3
"""
Alpha Decay Measurement for Predictive Signal Degradation

This module measures how fast predictive signals degrade due to market impact,
providing optimal holding period estimates for alpha-generating strategies.

Key Features:
- Real-time alpha decay tracking
- Half-life calculation for signals
- Impact-adjusted return estimation
- Optimal holding period optimization
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import math
from threading import Lock


@dataclass
class SignalDecayMetrics:
    """Metrics for signal decay analysis."""
    # Current signal strength
    signal_strength: float
    # Decay rate (per second)
    decay_rate: float
    # Half-life in seconds
    half_life_s: float
    # Time since signal generation (seconds)
    age_s: float
    # Remaining alpha (fraction of original)
    remaining_alpha: float
    # Optimal exit time (seconds)
    optimal_exit_s: float
    # Confidence in decay estimate
    confidence: float


@dataclass
class AlphaSignal:
    """Represents a predictive alpha signal."""
    signal_id: str
    timestamp_us: int
    initial_strength: float
    direction: int  # 1 = long, -1 = short
    predicted_return: float
    current_strength: float
    realized_pnl: float = 0.0


class AlphaDecayTracker:
    """
    Track alpha decay of predictive signals over time.
    
    This class measures how quickly trading signals lose their
    predictive power due to market impact and information diffusion.
    """
    
    def __init__(
        self,
        default_half_life_s: float = 30.0,
        min_samples: int = 50,
    ):
        """
        Initialize the alpha decay tracker.
        
        Args:
            default_half_life_s: Default signal half-life estimate
            min_samples: Minimum samples for reliable estimation
        """
        self.default_half_life_s = default_half_life_s
        self.min_samples = min_samples
        
        # Active signals
        self.active_signals: Dict[str, AlphaSignal] = {}
        
        # Historical decay observations
        self.decay_observations: List[Tuple[float, float]] = []  # (age, remaining_strength)
        
        # Estimated decay parameters
        self.estimated_decay_rate: Optional[float] = None
        self.estimated_half_life: float = default_half_life_s
        
        # Thread safety
        self._lock = Lock()
    
    def add_signal(
        self,
        signal_id: str,
        strength: float,
        direction: int,
        predicted_return: float,
    ) -> None:
        """Record a new alpha signal."""
        with self._lock:
            self.active_signals[signal_id] = AlphaSignal(
                signal_id=signal_id,
                timestamp_us=int(time.time() * 1_000_000),
                initial_strength=strength,
                direction=direction,
                predicted_return=predicted_return,
                current_strength=strength,
            )
    
    def update_signal(self, signal_id: str, current_strength: float) -> Optional[SignalDecayMetrics]:
        """
        Update signal strength and calculate decay metrics.
        
        Args:
            signal_id: ID of the signal to update
            current_strength: Current signal strength
            
        Returns:
            SignalDecayMetrics if signal exists
        """
        with self._lock:
            if signal_id not in self.active_signals:
                return None
            
            signal = self.active_signals[signal_id]
            now_us = int(time.time() * 1_000_000)
            
            # Calculate age
            age_s = (now_us - signal.timestamp_us) / 1_000_000.0
            
            # Record decay observation
            if signal.initial_strength > 0:
                remaining = current_strength / signal.initial_strength
                self.decay_observations.append((age_s, remaining))
                
                # Prune old observations
                if len(self.decay_observations) > 1000:
                    self.decay_observations = self.decay_observations[-500:]
                
                # Update decay estimate
                self._update_decay_estimate()
            
            # Update current strength
            signal.current_strength = current_strength
            
            # Calculate metrics
            return self._calculate_metrics(signal, age_s)
    
    def _update_decay_estimate(self) -> None:
        """Update decay rate estimate from observations."""
        if len(self.decay_observations) < self.min_samples:
            return
        
        # Fit exponential decay: remaining = exp(-λ * t)
        # Linearized: log(remaining) = -λ * t
        
        sum_t = 0.0
        sum_log_r = 0.0
        sum_t2 = 0.0
        n = 0
        
        for age, remaining in self.decay_observations:
            if remaining > 0:
                sum_t += age
                sum_log_r += math.log(remaining)
                sum_t2 += age * age
                n += 1
        
        if n < self.min_samples or sum_t2 == 0:
            return
        
        # OLS estimate: λ = -Σ(t * log(r)) / Σ(t²)
        sum_t_log_r = sum(
            age * math.log(remaining) 
            for age, remaining in self.decay_observations 
            if remaining > 0
        )
        
        decay_rate = -sum_t_log_r / sum_t2
        
        if decay_rate > 0:
            self.estimated_decay_rate = decay_rate
            self.estimated_half_life = math.log(2) / decay_rate
    
    def _calculate_metrics(
        self, 
        signal: AlphaSignal, 
        age_s: float
    ) -> SignalDecayMetrics:
        """Calculate decay metrics for a signal."""
        # Remaining alpha
        if self.estimated_decay_rate is not None:
            remaining_alpha = math.exp(-self.estimated_decay_rate * age_s)
            decay_rate = self.estimated_decay_rate
        else:
            # Use default
            decay_rate = math.log(2) / self.default_half_life_s
            remaining_alpha = math.exp(-decay_rate * age_s)
        
        # Half-life
        half_life = math.log(2) / decay_rate if decay_rate > 0 else self.default_half_life_s
        
        # Optimal exit time (when marginal alpha equals transaction cost)
        # Simplified: exit at 2x half-life for typical costs
        optimal_exit = half_life * 2.0
        
        # Confidence based on sample size
        confidence = min(1.0, len(self.decay_observations) / (self.min_samples * 2))
        
        return SignalDecayMetrics(
            signal_strength=signal.current_strength,
            decay_rate=decay_rate,
            half_life_s=half_life,
            age_s=age_s,
            remaining_alpha=remaining_alpha,
            optimal_exit_s=optimal_exit,
            confidence=confidence,
        )
    
    def close_signal(
        self, 
        signal_id: str, 
        realized_pnl: float
    ) -> Optional[SignalDecayMetrics]:
        """Close a signal and record realized PnL."""
        with self._lock:
            if signal_id not in self.active_signals:
                return None
            
            signal = self.active_signals.pop(signal_id)
            now_us = int(time.time() * 1_000_000)
            age_s = (now_us - signal.timestamp_us) / 1_000_000.0
            
            signal.realized_pnl = realized_pnl
            
            return self._calculate_metrics(signal, age_s)
    
    def get_average_half_life(self) -> float:
        """Get average signal half-life across all observations."""
        with self._lock:
            return self.estimated_half_life
    
    def should_exit_signal(self, signal_id: str) -> bool:
        """Check if a signal has decayed past optimal exit point."""
        with self._lock:
            if signal_id not in self.active_signals:
                return False
            
            signal = self.active_signals[signal_id]
            now_us = int(time.time() * 1_000_000)
            age_s = (now_us - signal.timestamp_us) / 1_000_000.0
            
            optimal_exit = self.estimated_half_life * 2.0
            
            return age_s > optimal_exit
    
    def get_active_signals_count(self) -> int:
        """Get number of active signals."""
        with self._lock:
            return len(self.active_signals)
    
    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self.active_signals.clear()
            self.decay_observations.clear()
            self.estimated_decay_rate = None
            self.estimated_half_life = self.default_half_life_s


if __name__ == "__main__":
    # Example usage
    tracker = AlphaDecayTracker(default_half_life_s=30.0)
    
    base_time = time.time()
    
    # Add some signals
    for i in range(5):
        tracker.add_signal(
            signal_id=f"signal_{i}",
            strength=1.0 - i * 0.1,
            direction=1,
            predicted_return=0.001,
        )
    
    # Simulate decay over time
    import time as time_module
    for t in range(10):
        time_module.sleep(0.1)  # Short delay for demo
        
        for i in range(5):
            # Simulate exponential decay
            age = t * 0.1
            decay_rate = math.log(2) / 30.0
            current = (1.0 - i * 0.1) * math.exp(-decay_rate * age)
            
            metrics = tracker.update_signal(f"signal_{i}", current)
            
            if metrics and i == 0 and t % 3 == 0:
                print(f"t={t*0.1:.1f}s: Strength={metrics.signal_strength:.3f}, "
                      f"Remaining={metrics.remaining_alpha:.2%}, "
                      f"Half-life={metrics.half_life_s:.1f}s")
    
    print(f"\nEstimated Half-Life: {tracker.get_average_half_life():.1f}s")
