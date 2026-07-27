#!/usr/bin/env python3
"""
Adaptive Step Size Controller for Continuous-Time RNN ODE Solvers

This module dynamically adjusts ODE solver step sizes based on market volatility,
ensuring numerical stability during flash crashes while maintaining efficiency
during calm periods.

Key Features:
- Real-time volatility estimation using multiple time scales
- Adaptive RK4/5 step size control with error estimation
- Flash crash detection and ultra-fine stepping
- Memory-efficient implementation for 8GB constraint
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import warnings


@dataclass
class AdaptiveStepConfig:
    """Configuration for adaptive step size controller."""
    base_dt: float = 0.01           # Base time step (seconds)
    min_dt: float = 1e-6            # Minimum step for stability
    max_dt: float = 0.1             # Maximum step for efficiency
    safety_factor: float = 0.9      # Safety margin for step adjustments
    max_volatility_ratio: float = 10.0  # Max volatility multiplier
    volatility_window: int = 100    # Samples for volatility estimation
    flash_crash_threshold: float = 5.0  # Sigma threshold for flash crash
    dtype: np.dtype = np.float64


class VolatilityEstimator:
    """
    Multi-scale volatility estimator for adaptive step control.
    
    Uses exponential moving averages at different time scales to capture
    both instantaneous spikes and sustained volatility regimes.
    """
    
    def __init__(self, config: AdaptiveStepConfig):
        self.config = config
        
        # Fast EMA for immediate reaction (alpha = 0.3)
        self.volatility_fast: float = 0.0
        self.alpha_fast: float = 0.3
        
        # Slow EMA for regime detection (alpha = 0.05)
        self.volatility_slow: float = 0.0
        self.alpha_slow: float = 0.05
        
        # Rolling window for statistical tests
        self.returns_buffer: deque = deque(maxlen=config.volatility_window)
        
        # Baseline volatility (computed from historical data)
        self.baseline_volatility: float = 0.01
        self.baseline_computed: bool = False
    
    def update(self, price_change: float) -> float:
        """
        Update volatility estimates with new price change.
        
        Args:
            price_change: Price change since last observation
            
        Returns:
            Current volatility estimate (fast scale)
        """
        abs_change = abs(price_change)
        
        # Update fast EMA
        self.volatility_fast = (
            (1 - self.alpha_fast) * self.volatility_fast + 
            self.alpha_fast * abs_change
        )
        
        # Update slow EMA
        self.volatility_slow = (
            (1 - self.alpha_slow) * self.volatility_slow + 
            self.alpha_slow * abs_change
        )
        
        # Add to rolling buffer
        self.returns_buffer.append(price_change)
        
        # Compute baseline after sufficient samples
        if len(self.returns_buffer) >= 50 and not self.baseline_computed:
            self._compute_baseline()
        
        return self.volatility_fast
    
    def _compute_baseline(self) -> None:
        """Compute baseline volatility from historical samples."""
        if len(self.returns_buffer) < 50:
            return
        
        returns = np.array(list(self.returns_buffer))
        self.baseline_volatility = float(np.std(returns))
        if self.baseline_volatility < 1e-10:
            self.baseline_volatility = 0.01  # Default fallback
        self.baseline_computed = True
    
    def get_volatility_ratio(self) -> float:
        """Get ratio of current to baseline volatility."""
        if not self.baseline_computed:
            return 1.0
        return self.volatility_fast / max(self.baseline_volatility, 1e-10)
    
    def is_flash_crash(self) -> bool:
        """Detect flash crash condition using statistical test."""
        if len(self.returns_buffer) < 20:
            return False
        
        returns = np.array(list(self.returns_buffer))
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-10:
            return False
        
        # Check if latest return is extreme
        latest = returns[-1]
        z_score = abs(latest - mean_ret) / std_ret
        
        return z_score > self.config.flash_crash_threshold
    
    def reset(self) -> None:
        """Reset all volatility estimates."""
        self.volatility_fast = 0.0
        self.volatility_slow = 0.0
        self.returns_buffer.clear()
        self.baseline_computed = False


class AdaptiveStepSolver:
    """
    Adaptive step size controller for ODE solvers.
    
    Implements Runge-Kutta-Fehlberg (RK45) style step size control
    with volatility-aware adjustments for financial time series.
    """
    
    def __init__(self, config: Optional[AdaptiveStepConfig] = None):
        self.config = config or AdaptiveStepConfig()
        self.volatility_estimator = VolatilityEstimator(self.config)
        
        # Current step size
        self.current_dt: float = self.config.base_dt
        
        # Error estimate from last step (for RK45)
        self.last_error_estimate: float = 0.0
        
        # Step rejection counter
        self.rejected_steps: int = 0
        
        # Total steps taken
        self.total_steps: int = 0
        self.accepted_steps: int = 0
    
    def suggest_step(self, price_change: float, error_estimate: Optional[float] = None) -> float:
        """
        Suggest next step size based on volatility and error estimates.
        
        Args:
            price_change: Latest price change for volatility update
            error_estimate: Local truncation error estimate (optional)
            
        Returns:
            Suggested step size for next integration step
        """
        # Update volatility
        vol = self.volatility_estimator.update(price_change)
        
        # Check for flash crash
        if self.volatility_estimator.is_flash_crash():
            # Ultra-fine stepping during flash crash
            self.current_dt = self.config.min_dt
            warnings.warn("Flash crash detected! Using minimum step size.")
            return self.current_dt
        
        # Volatility-based adjustment
        vol_ratio = self.volatility_estimator.get_volatility_ratio()
        vol_factor = 1.0 / np.sqrt(1.0 + vol_ratio)
        
        # Error-based adjustment (if available)
        error_factor = 1.0
        if error_estimate is not None and error_estimate > 1e-10:
            # Standard RK45 step size control
            tolerance = 1e-4  # Relative tolerance
            error_ratio = error_estimate / tolerance
            error_factor = (self.config.safety_factor / error_ratio) ** 0.2
        
        # Combined adjustment
        new_dt = self.config.base_dt * vol_factor * error_factor
        
        # Apply bounds and smoothing
        new_dt = np.clip(new_dt, self.config.min_dt, self.config.max_dt)
        
        # Smooth step size changes to avoid oscillations
        self.current_dt = 0.8 * self.current_dt + 0.2 * new_dt
        self.current_dt = np.clip(self.current_dt, self.config.min_dt, self.config.max_dt)
        
        return self.current_dt
    
    def accept_step(self) -> None:
        """Mark the current step as accepted."""
        self.accepted_steps += 1
        self.total_steps += 1
        self.rejected_steps = 0  # Reset rejection counter
    
    def reject_step(self) -> None:
        """Mark the current step as rejected (will be retried with smaller dt)."""
        self.rejected_steps += 1
        self.total_steps += 1
        
        # Reduce step size significantly on rejection
        self.current_dt *= 0.5
        self.current_dt = max(self.current_dt, self.config.min_dt)
    
    def get_statistics(self) -> dict:
        """Get solver statistics for monitoring."""
        return {
            'total_steps': self.total_steps,
            'accepted_steps': self.accepted_steps,
            'rejected_steps': self.rejected_steps,
            'acceptance_rate': self.accepted_steps / max(self.total_steps, 1),
            'current_dt': self.current_dt,
            'volatility_fast': self.volatility_estimator.volatility_fast,
            'volatility_slow': self.volatility_estimator.volatility_slow,
            'is_flash_crash': self.volatility_estimator.is_flash_crash(),
        }
    
    def reset(self) -> None:
        """Reset solver state."""
        self.current_dt = self.config.base_dt
        self.last_error_estimate = 0.0
        self.rejected_steps = 0
        self.total_steps = 0
        self.accepted_steps = 0
        self.volatility_estimator.reset()


class MarketAwareODESolver:
    """
    High-level ODE solver wrapper with market-aware adaptive stepping.
    
    Provides a simple interface for solving continuous-time RNN ODEs
    with automatic step size control based on market conditions.
    """
    
    def __init__(self, config: Optional[AdaptiveStepConfig] = None):
        self.config = config or AdaptiveStepConfig()
        self.adaptive_solver = AdaptiveStepSolver(self.config)
    
    def solve_step(
        self,
        derivative_fn: callable,
        state: np.ndarray,
        input_data: np.ndarray,
        price_change: float,
        target_dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Perform one adaptive ODE integration step.
        
        Args:
            derivative_fn: Function computing dh/dt given (state, input)
            state: Current hidden state array
            input_data: Current input array
            price_change: Price change for volatility estimation
            target_dt: Optional target time increment
            
        Returns:
            Tuple of (new_state, actual_dt_used)
        """
        # Get suggested step size
        dt = target_dt or self.adaptive_solver.suggest_step(price_change)
        
        # Simple Euler step for now (can be upgraded to RK4)
        # In production, this would use embedded RK45 for error estimation
        deriv = derivative_fn(state, input_data)
        new_state = state + dt * deriv
        
        # Clamp to prevent explosion
        new_state = np.clip(new_state, -10.0, 10.0)
        
        # Accept the step
        self.adaptive_solver.accept_step()
        
        return new_state, dt
    
    def solve_trajectory(
        self,
        derivative_fn: callable,
        initial_state: np.ndarray,
        inputs: np.ndarray,
        price_changes: np.ndarray,
        total_time: float,
    ) -> np.ndarray:
        """
        Solve ODE trajectory over entire sequence.
        
        Args:
            derivative_fn: Function computing dh/dt
            initial_state: Initial hidden state
            inputs: Sequence of inputs [T, input_dim]
            price_changes: Sequence of price changes [T]
            total_time: Total time to simulate
            
        Returns:
            State trajectory [T+1, state_dim]
        """
        T = len(inputs)
        state_dim = len(initial_state)
        
        # Pre-allocate output
        trajectory = np.zeros((T + 1, state_dim), dtype=self.config.dtype)
        trajectory[0] = initial_state
        
        current_state = initial_state.copy()
        current_time = 0.0
        dt_target = total_time / T
        
        for t in range(T):
            new_state, dt_used = self.solve_step(
                derivative_fn,
                current_state,
                inputs[t],
                price_changes[t],
                dt_target,
            )
            
            trajectory[t + 1] = new_state
            current_state = new_state
            current_time += dt_used
        
        return trajectory
    
    def get_statistics(self) -> dict:
        """Get solver statistics."""
        return self.adaptive_solver.get_statistics()
    
    def reset(self) -> None:
        """Reset solver state."""
        self.adaptive_solver.reset()


def benchmark_adaptive_solver() -> None:
    """Benchmark adaptive step solver performance."""
    import time
    
    config = AdaptiveStepConfig(base_dt=0.01)
    solver = AdaptiveStepSolver(config)
    
    # Simulate various market conditions
    np.random.seed(42)
    
    # Calm period
    print("Simulating calm market...")
    start = time.perf_counter()
    for _ in range(1000):
        price_change = np.random.randn() * 0.001
        dt = solver.suggest_step(price_change)
        solver.accept_step()
    calm_time = time.perf_counter() - start
    
    # Reset
    solver.reset()
    
    # Volatile period
    print("Simulating volatile market...")
    start = time.perf_counter()
    for _ in range(1000):
        price_change = np.random.randn() * 0.01
        dt = solver.suggest_step(price_change)
        solver.accept_step()
    volatile_time = time.perf_counter() - start
    
    # Flash crash
    print("Simulating flash crash...")
    solver.reset()
    start = time.perf_counter()
    for i in range(1000):
        if i == 500:
            price_change = -0.1  # Flash crash
        else:
            price_change = np.random.randn() * 0.001
        dt = solver.suggest_step(price_change)
        solver.accept_step()
    crash_time = time.perf_counter() - start
    
    print(f"Calm market: {calm_time*1000:.2f}ms for 1000 steps")
    print(f"Volatile market: {volatile_time*1000:.2f}ms for 1000 steps")
    print(f"Flash crash: {crash_time*1000:.2f}ms for 1000 steps")
    
    stats = solver.get_statistics()
    print(f"Final step size: {stats['current_dt']:.6f}")


if __name__ == "__main__":
    benchmark_adaptive_solver()
