#!/usr/bin/env python3
"""
Dependency Tracker for Rolling Correlation Analysis

Calculates rolling Kendall's tau and Spearman rank correlations
to track dynamic dependencies between crypto assets.
Optimized with NumPy for 8GB RAM constraints.

Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from scipy.stats import kendalltau, spearmanr
import warnings

# Suppress numerical warnings for production stability
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class CorrelationWindow:
    """Sliding window for correlation calculations."""
    
    max_size: int
    data: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add(self, returns: np.ndarray) -> None:
        """Add a new observation to the window."""
        self.data.append(returns.copy())
    
    def get_matrix(self) -> Optional[np.ndarray]:
        """Get all data as a 2D array."""
        if len(self.data) < 2:
            return None
        return np.array(list(self.data))
    
    def is_ready(self, min_samples: int = 30) -> bool:
        """Check if window has enough samples."""
        return len(self.data) >= min_samples
    
    def clear(self) -> None:
        """Clear the window."""
        self.data.clear()


@dataclass
class CorrelationMetrics:
    """Container for correlation metrics."""
    
    kendall_tau: np.ndarray
    spearman_rho: np.ndarray
    pearson_corr: np.ndarray
    timestamp: float
    sample_count: int
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for logging."""
        return {
            'kendall_tau': self.kendall_tau.tolist(),
            'spearman_rho': self.spearman_rho.tolist(),
            'pearson_corr': self.pearson_corr.tolist(),
            'timestamp': self.timestamp,
            'sample_count': self.sample_count,
        }


class DependencyTracker:
    """
    Tracks rolling rank-based correlations between crypto assets.
    
    Uses Kendall's tau and Spearman's rho for robustness to outliers
    and non-linear relationships common in crypto markets.
    """
    
    def __init__(
        self,
        assets: List[str],
        window_size: int = 252,
        min_samples: int = 30,
        memory_limit_mb: int = 512,
    ) -> None:
        """
        Initialize the dependency tracker.
        
        Args:
            assets: List of asset names (e.g., ['BTC', 'ETH', 'SOL'])
            window_size: Maximum window size for rolling calculations
            min_samples: Minimum samples required before computing correlations
            memory_limit_mb: Memory limit in MB for data storage
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Calculate max observations based on memory limit
        max_obs_by_memory = (memory_limit_mb * 1024 * 1024) // (self.n_assets * 8)
        effective_window = min(window_size, int(max_obs_by_memory))
        
        self.window = CorrelationWindow(max_size=effective_window)
        
        # Cache for latest correlation matrices
        self._latest_metrics: Optional[CorrelationMetrics] = None
        self._update_count: int = 0
        
        # Asset name to index mapping
        self.asset_to_idx = {asset: idx for idx, asset in enumerate(assets)}
    
    def update(self, returns: Dict[str, float], timestamp: float) -> Optional[CorrelationMetrics]:
        """
        Update tracker with new returns data.
        
        Args:
            returns: Dictionary mapping asset names to returns
            timestamp: Unix timestamp of the observation
            
        Returns:
            Updated correlation metrics if enough samples, else None
        """
        # Convert to ordered array
        returns_array = np.zeros(self.n_assets)
        for i, asset in enumerate(self.assets):
            returns_array[i] = returns.get(asset, 0.0)
        
        # Add to window
        self.window.add(returns_array)
        
        # Check if ready to compute
        if not self.window.is_ready(self.min_samples):
            return None
        
        # Compute correlations
        metrics = self._compute_correlations(timestamp)
        self._latest_metrics = metrics
        self._update_count += 1
        
        return metrics
    
    def _compute_correlations(self, timestamp: float) -> CorrelationMetrics:
        """Compute all correlation metrics from current window."""
        data = self.window.get_matrix()
        if data is None:
            raise ValueError("Insufficient data for correlation calculation")
        
        n_samples = data.shape[0]
        
        # Initialize matrices
        kendall_matrix = np.eye(self.n_assets)
        spearman_matrix = np.eye(self.n_assets)
        pearson_matrix = np.corrcoef(data.T)
        
        # Compute pairwise rank correlations
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                # Kendall's tau
                tau_result = kendalltau(data[:, i], data[:, j])
                kendall_matrix[i, j] = tau_result.correlation
                kendall_matrix[j, i] = tau_result.correlation
                
                # Spearman's rho
                rho_result = spearmanr(data[:, i], data[:, j])
                spearman_matrix[i, j] = rho_result.correlation
                spearman_matrix[j, i] = rho_result.correlation
        
        return CorrelationMetrics(
            kendall_tau=kendall_matrix,
            spearman_rho=spearman_matrix,
            pearson_corr=pearson_matrix,
            timestamp=timestamp,
            sample_count=n_samples,
        )
    
    def get_latest_metrics(self) -> Optional[CorrelationMetrics]:
        """Get the most recent correlation metrics."""
        return self._latest_metrics
    
    def get_correlation_between(
        self,
        asset1: str,
        asset2: str,
        method: str = 'kendall'
    ) -> Optional[float]:
        """
        Get correlation between two specific assets.
        
        Args:
            asset1: First asset name
            asset2: Second asset name
            method: 'kendall', 'spearman', or 'pearson'
            
        Returns:
            Correlation value or None if not computed yet
        """
        if self._latest_metrics is None:
            return None
        
        idx1 = self.asset_to_idx.get(asset1)
        idx2 = self.asset_to_idx.get(asset2)
        
        if idx1 is None or idx2 is None:
            raise ValueError(f"Unknown asset: {asset1 if idx1 is None else asset2}")
        
        if method == 'kendall':
            return self._latest_metrics.kendall_tau[idx1, idx2]
        elif method == 'spearman':
            return self._latest_metrics.spearman_rho[idx1, idx2]
        elif method == 'pearson':
            return self._latest_metrics.pearson_corr[idx1, idx2]
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_correlation_breakdown(
        self,
        threshold: float = 0.7,
        lookback_periods: int = 10
    ) -> List[Tuple[str, str, float, float]]:
        """
        Detect pairs where correlation has broken down significantly.
        
        Identifies pairs where recent correlation differs substantially
        from historical average, indicating potential regime change.
        
        Args:
            threshold: Minimum absolute change to flag as breakdown
            lookback_periods: Number of periods to compare
            
        Returns:
            List of tuples (asset1, asset2, old_corr, new_corr)
        """
        if self._latest_metrics is None:
            return []
        
        # This would require storing historical metrics
        # Simplified implementation for production
        breakdowns = []
        current_kendall = self._latest_metrics.kendall_tau
        
        # Compare to long-term average (would be stored in production)
        # Placeholder: flag extreme correlations
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                corr = current_kendall[i, j]
                if abs(corr) > threshold:
                    breakdowns.append((
                        self.assets[i],
                        self.assets[j],
                        0.0,  # Historical average (placeholder)
                        corr
                    ))
        
        return breakdowns
    
    def get_tail_dependence_estimate(self) -> np.ndarray:
        """
        Estimate lower tail dependence from Kendall's tau.
        
        For elliptical copulas, tail dependence can be approximated
        from the correlation matrix.
        
        Returns:
            Estimated lower tail dependence matrix
        """
        if self._latest_metrics is None:
            return np.eye(self.n_assets)
        
        tau = self._latest_metrics.kendall_tau
        
        # Approximation for Gaussian copula tail dependence
        # lambda_L = 2 * Phi(-sqrt((1-rho)/(1+rho)) * q_alpha)
        # Simplified: use transformation of tau
        
        # For Clayton copula: lambda_L = 2^(-1/theta), theta from tau
        # tau = theta / (theta + 2) => theta = 2*tau/(1-tau)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.where(
                tau != 1,
                2 * tau / (1 - tau),
                np.inf
            )
            lambda_lower = np.where(
                theta > 0,
                np.power(2, -1 / theta),
                0
            )
        
        # Ensure diagonal is 1
        np.fill_diagonal(lambda_lower, 1.0)
        
        return lambda_lower
    
    def reset(self) -> None:
        """Reset all state."""
        self.window.clear()
        self._latest_metrics = None
        self._update_count = 0
    
    def get_statistics(self) -> Dict[str, any]:
        """Get tracker statistics for monitoring."""
        return {
            'assets': self.assets,
            'window_size': self.window_size,
            'current_samples': len(self.window.data),
            'min_samples_required': self.min_samples,
            'is_ready': self.window.is_ready(self.min_samples),
            'update_count': self._update_count,
        }


class RollingCorrelationAnalyzer:
    """
    Advanced analyzer for rolling correlation dynamics.
    
    Provides additional metrics like correlation volatility,
    regime detection, and correlation clustering.
    """
    
    def __init__(self, tracker: DependencyTracker) -> None:
        """Initialize with a dependency tracker."""
        self.tracker = tracker
        self.correlation_history: deque = deque(maxlen=500)
    
    def record_metrics(self, metrics: CorrelationMetrics) -> None:
        """Record metrics for historical analysis."""
        self.correlation_history.append(metrics)
    
    def get_correlation_volatility(self, window: int = 60) -> Optional[np.ndarray]:
        """
        Calculate volatility of correlations over time.
        
        Args:
            window: Number of periods for volatility calculation
            
        Returns:
            Standard deviation of correlation for each pair
        """
        if len(self.correlation_history) < window:
            return None
        
        # Extract correlation matrices
        correlations = np.array([
            m.kendall_tau for m in list(self.correlation_history)[-window:]
        ])
        
        # Compute standard deviation (upper triangle only)
        n = self.tracker.n_assets
        vol_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                vol_matrix[i, j] = np.std(correlations[:, i, j])
                vol_matrix[j, i] = vol_matrix[i, j]
        
        return vol_matrix
    
    def detect_regime_change(
        self,
        short_window: int = 20,
        long_window: int = 100,
        threshold: float = 0.15
    ) -> Optional[bool]:
        """
        Detect if correlation regime has changed.
        
        Compares short-term and long-term average correlations.
        
        Args:
            short_window: Short-term window size
            long_window: Long-term window size
            threshold: Minimum difference to flag regime change
            
        Returns:
            True if regime change detected, None if insufficient data
        """
        if len(self.correlation_history) < long_window:
            return None
        
        recent = list(self.correlation_history)[-short_window:]
        historical = list(self.correlation_history)[-long_window:-short_window]
        
        recent_avg = np.mean([m.kendall_tau for m in recent], axis=0)
        historical_avg = np.mean([m.kendall_tau for m in historical], axis=0)
        
        max_diff = np.max(np.abs(recent_avg - historical_avg))
        
        return max_diff > threshold
    
    def get_average_correlation(self) -> Optional[float]:
        """Get average pairwise correlation."""
        if self._latest_metrics is None:
            return None
        
        metrics = self.tracker.get_latest_metrics()
        if metrics is None:
            return None
        
        # Average of upper triangle
        tau = metrics.kendall_tau
        n = self.tracker.n_assets
        
        upper_tri = tau[np.triu_indices(n, k=1)]
        return float(np.mean(upper_tri))


def create_tracker_for_crypto(
    assets: List[str] = None,
    high_frequency: bool = False
) -> DependencyTracker:
    """
    Factory function to create optimized tracker for crypto trading.
    
    Args:
        assets: List of crypto assets (default: BTC, ETH, SOL)
        high_frequency: If True, use smaller windows for faster response
        
    Returns:
        Configured DependencyTracker instance
    """
    if assets is None:
        assets = ['BTC', 'ETH', 'SOL']
    
    if high_frequency:
        # Faster response for HFT environments
        return DependencyTracker(
            assets=assets,
            window_size=100,
            min_samples=20,
            memory_limit_mb=256,
        )
    else:
        # Standard configuration for swing trading
        return DependencyTracker(
            assets=assets,
            window_size=252,
            min_samples=30,
            memory_limit_mb=512,
        )


if __name__ == '__main__':
    # Example usage
    tracker = create_tracker_for_crypto(['BTC', 'ETH', 'SOL'])
    
    # Simulate some returns data
    np.random.seed(42)
    for t in range(100):
        returns = {
            'BTC': np.random.normal(0, 0.02),
            'ETH': np.random.normal(0, 0.03),
            'SOL': np.random.normal(0, 0.05),
        }
        metrics = tracker.update(returns, timestamp=t * 60)
        
        if metrics is not None and t % 20 == 0:
            print(f"t={t}: BTC-ETH Kendall's tau = "
                  f"{tracker.get_correlation_between('BTC', 'ETH'):.4f}")
    
    print("\nFinal statistics:", tracker.get_statistics())
