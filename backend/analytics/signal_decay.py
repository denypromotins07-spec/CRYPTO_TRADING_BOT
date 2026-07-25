#!/usr/bin/env python3
"""
Signal Decay Analytics - Measure Half-Life of Predictive Alpha

This module measures how quickly predictive signals decay across different
timeframes, critical for determining optimal holding periods and entry/exit timing.

Key Features:
- Exponential decay fitting for alpha half-life estimation
- Multi-timeframe analysis (1min to 4hr windows)
- Signal quality scoring based on decay characteristics
- Memory-efficient rolling window implementation

Optimized for crypto HFT where alpha decays within seconds to minutes.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.optimize import curve_fit
import warnings


class DecayQuality(Enum):
    """Quality rating for signal decay characteristics."""
    EXCELLENT = "excellent"  # Slow decay, high predictability
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"  # Fast decay, low utility
    NOISE = "noise"  # No discernible pattern


@dataclass
class DecayResult:
    """Results from signal decay analysis."""
    timeframe: str
    half_life_seconds: float
    decay_rate: float
    initial_alpha: float
    r_squared: float
    quality: DecayQuality
    confidence_interval: Tuple[float, float]
    n_observations: int
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Signal Decay Analysis ({self.timeframe})\n"
            f"{'='*50}\n"
            f"Half-life: {self.half_life_seconds:.2f} seconds\n"
            f"Decay Rate: {self.decay_rate:.4f} per second\n"
            f"Initial Alpha: {self.initial_alpha:.4f}\n"
            f"R-squared: {self.r_squared:.4f}\n"
            f"Quality: {self.quality.value.upper()}\n"
            f"95% CI: [{self.confidence_interval[0]:.2f}, {self.confidence_interval[1]:.2f}] seconds\n"
        )


@dataclass
class MultiTimeframeAnalysis:
    """Combined analysis across multiple timeframes."""
    results: Dict[str, DecayResult]
    optimal_timeframe: str
    max_half_life: float
    min_decay_rate: float
    
    def summary(self) -> str:
        """Generate summary across all timeframes."""
        lines = ["Multi-Timeframe Signal Decay Analysis", "=" * 60]
        
        for tf, result in sorted(self.results.items()):
            icon = "★" if tf == self.optimal_timeframe else " "
            lines.append(
                f"{icon} {tf:12s}: HL={result.half_life_seconds:7.1f}s, "
                f"α₀={result.initial_alpha:6.4f}, R²={result.r_squared:.3f}, "
                f"{result.quality.value.upper():8s}"
            )
        
        lines.extend([
            "=" * 60,
            f"Optimal Timeframe: {self.optimal_timeframe}",
            f"Maximum Half-life: {self.max_half_life:.1f} seconds",
        ])
        
        return "\n".join(lines)


def exponential_decay(t: np.ndarray, alpha_0: float, decay_rate: float, 
                      baseline: float = 0.0) -> np.ndarray:
    """Exponential decay model: α(t) = α₀ * exp(-λt) + baseline"""
    return alpha_0 * np.exp(-decay_rate * t) + baseline


def linear_decay(t: np.ndarray, alpha_0: float, decay_rate: float,
                 baseline: float = 0.0) -> np.ndarray:
    """Linear decay model (alternative): α(t) = α₀ - λt + baseline"""
    return np.maximum(baseline, alpha_0 - decay_rate * t)


class SignalDecayAnalyzer:
    """
    Analyze decay characteristics of predictive signals.
    
    Measures how quickly IC (Information Coefficient) or alpha
    decays over time to determine optimal holding periods.
    """
    
    def __init__(self, min_half_life: float = 1.0, 
                 max_half_life: float = 3600.0):
        """
        Initialize analyzer.
        
        Args:
            min_half_life: Minimum meaningful half-life (seconds)
            max_half_life: Maximum meaningful half-life (seconds)
        """
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self._results: Dict[str, DecayResult] = {}
    
    def analyze_signal(self, 
                       signal_values: np.ndarray,
                       future_returns: np.ndarray,
                       timestamps: np.ndarray,
                       timeframe: str = "default") -> DecayResult:
        """
        Analyze decay of a predictive signal.
        
        Args:
            signal_values: Historical signal values
            future_returns: Forward returns aligned with signals
            timestamps: Time offsets for each observation (seconds)
            timeframe: Name for this analysis
            
        Returns:
            DecayResult with half-life and quality metrics
        """
        # Convert inputs
        signal_values = np.asarray(signal_values, dtype=np.float64)
        future_returns = np.asarray(future_returns, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        
        # Remove NaN values
        mask = ~(np.isnan(signal_values) | np.isnan(future_returns) | np.isnan(timestamps))
        signal_values = signal_values[mask]
        future_returns = future_returns[mask]
        timestamps = timestamps[mask]
        
        if len(signal_values) < 50:
            warnings.warn(f"Insufficient observations ({len(signal_values)}) for decay analysis")
            return self._null_result(timeframe, len(signal_values))
        
        # Compute IC (correlation) at different time lags
        lags = np.unique(np.clip(timestamps.astype(int), 1, int(self.max_half_life * 2)))
        if len(lags) < 5:
            lags = np.linspace(1, int(self.max_half_life), 20).astype(int)
        
        ic_by_lag = []
        valid_lags = []
        
        for lag in lags:
            # Shift returns by lag
            if lag < len(future_returns):
                shifted_returns = future_returns[:-lag]
                lagged_signals = signal_values[lag:]
                
                if len(shifted_returns) > 10:
                    ic = np.corrcoef(lagged_signals, shifted_returns)[0, 1]
                    if not np.isnan(ic):
                        ic_by_lag.append(abs(ic))  # Use absolute IC
                        valid_lags.append(lag)
        
        if len(valid_lags) < 3:
            return self._null_result(timeframe, len(signal_values))
        
        valid_lags = np.array(valid_lags, dtype=np.float64)
        ic_by_lag = np.array(ic_by_lag)
        
        # Fit exponential decay model
        try:
            # Initial guesses
            p0 = [ic_by_lag[0], 0.01, 0.0]
            
            # Bound constraints
            bounds = (
                [0, 1e-6, -0.1],  # Lower bounds
                [2, 1.0, 0.1]     # Upper bounds
            )
            
            params, covariance = curve_fit(
                exponential_decay,
                valid_lags,
                ic_by_lag,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            
            alpha_0, decay_rate, baseline = params
            
        except (RuntimeError, ValueError) as e:
            warnings.warn(f"Curve fitting failed: {e}")
            return self._null_result(timeframe, len(signal_values))
        
        # Calculate half-life: t_{1/2} = ln(2) / λ
        if decay_rate > 0:
            half_life = np.log(2) / decay_rate
        else:
            half_life = float('inf')
        
        # Clamp to reasonable range
        half_life = np.clip(half_life, self.min_half_life, self.max_half_life)
        
        # Calculate R-squared
        predicted = exponential_decay(valid_lags, *params)
        ss_res = np.sum((ic_by_lag - predicted) ** 2)
        ss_tot = np.sum((ic_by_lag - np.mean(ic_by_lag)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Confidence interval for half-life
        try:
            param_std = np.sqrt(np.diag(covariance))
            decay_std = param_std[1]
            if decay_std > 0 and decay_rate > 0:
                # Delta method approximation
                hl_std = (np.log(2) / decay_rate**2) * decay_std
                ci_lower = max(0, half_life - 1.96 * hl_std)
                ci_upper = half_life + 1.96 * hl_std
            else:
                ci_lower, ci_upper = half_life * 0.5, half_life * 2.0
        except:
            ci_lower, ci_upper = half_life * 0.5, half_life * 2.0
        
        # Determine quality
        quality = self._assess_quality(half_life, r_squared, alpha_0)
        
        result = DecayResult(
            timeframe=timeframe,
            half_life_seconds=float(half_life),
            decay_rate=float(decay_rate),
            initial_alpha=float(alpha_0),
            r_squared=float(r_squared),
            quality=quality,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            n_observations=len(signal_values)
        )
        
        self._results[timeframe] = result
        return result
    
    def _null_result(self, timeframe: str, n_obs: int) -> DecayResult:
        """Return null result when analysis fails."""
        return DecayResult(
            timeframe=timeframe,
            half_life_seconds=0.0,
            decay_rate=0.0,
            initial_alpha=0.0,
            r_squared=0.0,
            quality=DecayQuality.NOISE,
            confidence_interval=(0.0, 0.0),
            n_observations=n_obs
        )
    
    def _assess_quality(self, half_life: float, r_squared: float, 
                        alpha_0: float) -> DecayQuality:
        """Assess signal quality based on decay characteristics."""
        # Score based on multiple factors
        score = 0.0
        
        # Half-life scoring (prefer moderate half-lives for HFT)
        if 10 <= half_life <= 300:  # 10s to 5min is ideal for HFT
            score += 3.0
        elif 5 <= half_life < 10 or 300 < half_life <= 600:
            score += 2.0
        elif half_life < 5 or half_life > 600:
            score += 1.0
        
        # R-squared scoring
        if r_squared >= 0.8:
            score += 3.0
        elif r_squared >= 0.6:
            score += 2.0
        elif r_squared >= 0.4:
            score += 1.0
        
        # Initial alpha scoring
        if alpha_0 >= 0.3:
            score += 2.0
        elif alpha_0 >= 0.15:
            score += 1.0
        
        # Map score to quality
        if score >= 7:
            return DecayQuality.EXCELLENT
        elif score >= 5:
            return DecayQuality.GOOD
        elif score >= 3:
            return DecayQuality.MODERATE
        elif score >= 1:
            return DecayQuality.POOR
        else:
            return DecayQuality.NOISE
    
    def analyze_multiple_timeframes(self,
                                    signals: Dict[str, np.ndarray],
                                    returns: Dict[str, np.ndarray],
                                    timestamps: Dict[str, np.ndarray]) -> MultiTimeframeAnalysis:
        """
        Analyze signal decay across multiple timeframes.
        
        Args:
            signals: Dict mapping timeframe to signal values
            returns: Dict mapping timeframe to future returns
            timestamps: Dict mapping timeframe to timestamps
            
        Returns:
            MultiTimeframeAnalysis with combined results
        """
        results = {}
        
        for tf in signals.keys():
            if tf in returns and tf in timestamps:
                result = self.analyze_signal(
                    signals[tf],
                    returns[tf],
                    timestamps[tf],
                    timeframe=tf
                )
                results[tf] = result
        
        # Find optimal timeframe
        if results:
            valid_results = {
                tf: r for tf, r in results.items() 
                if r.quality != DecayQuality.NOISE and r.half_life > 0
            }
            
            if valid_results:
                optimal_tf = max(valid_results.keys(), 
                                key=lambda tf: valid_results[tf].half_life)
                max_hl = valid_results[optimal_tf].half_life
                min_decay = valid_results[optimal_tf].decay_rate
            else:
                optimal_tf = list(results.keys())[0]
                max_hl = 0.0
                min_decay = 0.0
        else:
            optimal_tf = ""
            max_hl = 0.0
            min_decay = 0.0
        
        return MultiTimeframeAnalysis(
            results=results,
            optimal_timeframe=optimal_tf,
            max_half_life=max_hl,
            min_decay_rate=min_decay
        )
    
    def get_optimal_holding_period(self, timeframe: str) -> float:
        """
        Get recommended holding period based on half-life.
        
        Rule of thumb: Hold for ~1 half-life to capture 50% of alpha.
        """
        result = self._results.get(timeframe)
        if result is None or result.half_life <= 0:
            return 0.0
        
        # Optimal hold time is approximately one half-life
        return result.half_life


def compute_signal_half_life(signals: np.ndarray, 
                             returns: np.ndarray,
                             lags: Optional[np.ndarray] = None) -> float:
    """
    Convenience function to compute signal half-life.
    
    Args:
        signals: Predictive signal values
        returns: Forward returns
        lags: Time lags to evaluate (optional)
        
    Returns:
        Half-life in same units as lags
    """
    analyzer = SignalDecayAnalyzer()
    
    if lags is None:
        lags = np.arange(len(signals))
    
    result = analyzer.analyze_signal(signals, returns, lags)
    return result.half_life_seconds


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n = 10000
    
    # Generate synthetic signal with known decay
    true_decay_rate = 0.01  # Half-life ≈ 69 seconds
    true_alpha_0 = 0.3
    
    lags = np.arange(1, 200)
    true_ic = true_alpha_0 * np.exp(-true_decay_rate * lags)
    
    # Simulate noisy observations
    signals = np.random.randn(n)
    returns = np.zeros(n)
    
    # Create returns with decaying predictability
    for i in range(n - 200):
        for lag in range(1, min(200, n - i)):
            returns[i] += signals[i] * true_alpha_0 * np.exp(-true_decay_rate * lag)
        returns[i] += np.random.randn() * 0.1
    
    timestamps = np.arange(n) % 200
    
    analyzer = SignalDecayAnalyzer()
    result = analyzer.analyze_signal(signals, returns, timestamps, "synthetic")
    
    print(result.summary())
    
    print(f"\nRecommended holding period: {analyzer.get_optimal_holding_period('synthetic'):.1f} seconds")
