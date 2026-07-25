#!/usr/bin/env python3
"""
Covariance Engine Module for ZAID Trading Bot
Computes rolling covariance matrices for portfolio mathematics
Optimized for 4-asset portfolio (BTC, SOL, ETH, USDT) under 8GB RAM

This module provides efficient rolling window covariance calculations
with support for exponential weighting and online updates.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass
import threading


@dataclass
class CovarianceResult:
    """Container for covariance computation results"""
    matrix: np.ndarray
    assets: Tuple[str, ...]
    window_size: int
    is_positive_definite: bool
    computation_time_ns: int
    condition_number: float


class RollingCovarianceEngine:
    """
    High-performance rolling covariance calculator for multi-asset portfolios.
    
    Implements both standard rolling window and exponentially weighted
    covariance estimation with thread-safe operations.
    """
    
    def __init__(self, 
                 assets: Tuple[str, ...] = ("BTC", "SOL", "ETH", "USDT"),
                 window_size: int = 252,
                 min_samples: int = 30):
        """
        Initialize the covariance engine.
        
        Args:
            assets: Tuple of asset names in order
            window_size: Number of samples for rolling window
            min_samples: Minimum samples required before output is valid
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Circular buffer for returns data
        self._buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
        # Cached results
        self._cached_cov: Optional[np.ndarray] = None
        self._cache_valid = False
    
    def add_returns(self, returns: np.ndarray) -> None:
        """
        Add a new return vector to the rolling window.
        
        Args:
            returns: Array of returns for all assets (shape: n_assets,)
        """
        returns = np.asarray(returns, dtype=np.float64).flatten()
        
        if returns.shape[0] != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} returns, got {returns.shape[0]}"
            )
        
        with self._lock:
            self._buffer.append(returns.copy())
            self._cache_valid = False
    
    def add_returns_batch(self, returns_matrix: np.ndarray) -> None:
        """
        Add multiple return vectors at once.
        
        Args:
            returns_matrix: Matrix of returns (shape: n_samples x n_assets)
        """
        returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
        
        if returns_matrix.ndim == 1:
            returns_matrix = returns_matrix.reshape(1, -1)
        
        if returns_matrix.shape[1] != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} assets, got {returns_matrix.shape[1]}"
            )
        
        with self._lock:
            for row in returns_matrix:
                self._buffer.append(row.copy())
            self._cache_valid = False
    
    def compute(self, use_cache: bool = True) -> Optional[CovarianceResult]:
        """
        Compute the current rolling covariance matrix.
        
        Args:
            use_cache: Whether to return cached result if available
            
        Returns:
            CovarianceResult or None if insufficient samples
        """
        import time
        
        with self._lock:
            if len(self._buffer) < self.min_samples:
                return None
            
            if use_cache and self._cache_valid and self._cached_cov is not None:
                return CovarianceResult(
                    matrix=self._cached_cov.copy(),
                    assets=self.assets,
                    window_size=len(self._buffer),
                    is_positive_definite=self._is_positive_definite(self._cached_cov),
                    computation_time_ns=0,
                    condition_number=self._condition_number(self._cached_cov)
                )
            
            start_ns = time.perf_counter_ns()
            
            # Convert buffer to array
            data = np.array(list(self._buffer))
            n_samples = data.shape[0]
            
            # Compute covariance using numpy (uses optimized BLAS)
            cov_matrix = np.cov(data, rowvar=False, ddof=1)
            
            elapsed_ns = time.perf_counter_ns() - start_ns
            
            # Cache result
            self._cached_cov = cov_matrix.copy()
            self._cache_valid = True
            
            return CovarianceResult(
                matrix=cov_matrix,
                assets=self.assets,
                window_size=n_samples,
                is_positive_definite=self._is_positive_definite(cov_matrix),
                computation_time_ns=elapsed_ns,
                condition_number=self._condition_number(cov_matrix)
            )
    
    def _is_positive_definite(self, matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite via Cholesky decomposition"""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def _condition_number(self, matrix: np.ndarray) -> float:
        """Compute condition number for numerical stability assessment"""
        try:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            if singular_values[-1] < 1e-15:
                return float('inf')
            return float(singular_values[0] / singular_values[-1])
        except Exception:
            return float('inf')
    
    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Convert covariance to correlation matrix"""
        result = self.compute()
        if result is None:
            return None
        
        cov = result.matrix
        std_devs = np.sqrt(np.diag(cov))
        
        if np.any(std_devs < 1e-15):
            return None
        
        # Correlation = Cov / (std_i * std_j)
        outer = np.outer(std_devs, std_devs)
        corr = cov / outer
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(corr, 1.0)
        
        return corr
    
    def clear(self) -> None:
        """Clear all stored data"""
        with self._lock:
            self._buffer.clear()
            self._cached_cov = None
            self._cache_valid = False


class ExponentialWeightedCovariance(RollingCovarianceEngine):
    """
    Exponentially weighted covariance estimator.
    
    Gives more weight to recent observations, useful for
    rapidly changing market conditions.
    """
    
    def __init__(self,
                 assets: Tuple[str, ...] = ("BTC", "SOL", "ETH", "USDT"),
                 span: int = 63,
                 min_samples: int = 30):
        """
        Initialize EWMA covariance estimator.
        
        Args:
            assets: Tuple of asset names
            span: Decay span for exponential weights
            min_samples: Minimum samples before output is valid
        """
        super().__init__(assets=assets, window_size=span * 5, min_samples=min_samples)
        self.span = span
        self._ewma_cov: Optional[np.ndarray] = None
        self._ewma_mean: Optional[np.ndarray] = None
        self._decay = 1.0 - 2.0 / (span + 1.0)
    
    def add_returns(self, returns: np.ndarray) -> None:
        """Add returns and update EWMA estimate"""
        returns = np.asarray(returns, dtype=np.float64).flatten()
        
        if returns.shape[0] != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} returns, got {returns.shape[0]}"
            )
        
        with self._lock:
            self._buffer.append(returns.copy())
            
            if len(self._buffer) >= self.min_samples:
                # Update EWMA mean
                if self._ewma_mean is None:
                    self._ewma_mean = returns.copy()
                else:
                    self._ewma_mean = self._decay * self._ewma_mean + \
                                      (1 - self._decay) * returns
                
                # Update EWMA covariance
                diff = returns - self._ewma_mean
                outer = np.outer(diff, diff)
                
                if self._ewma_cov is None:
                    self._ewma_cov = outer
                else:
                    self._ewma_cov = self._decay * self._ewma_cov + \
                                     (1 - self._decay) * outer
                
                self._cache_valid = False
    
    def compute(self, use_cache: bool = True) -> Optional[CovarianceResult]:
        """Return current EWMA covariance estimate"""
        import time
        
        with self._lock:
            if self._ewma_cov is None or len(self._buffer) < self.min_samples:
                return None
            
            if use_cache and self._cache_valid and self._cached_cov is not None:
                return CovarianceResult(
                    matrix=self._cached_cov.copy(),
                    assets=self.assets,
                    window_size=len(self._buffer),
                    is_positive_definite=self._is_positive_definite(self._cached_cov),
                    computation_time_ns=0,
                    condition_number=self._condition_number(self._cached_cov)
                )
            
            start_ns = time.perf_counter_ns()
            
            # Scale EWMA to be comparable to sample covariance
            scale_factor = 1.0 / (1 - self._decay ** len(self._buffer))
            cov_matrix = self._ewma_cov * scale_factor
            
            elapsed_ns = time.perf_counter_ns() - start_ns
            
            self._cached_cov = cov_matrix.copy()
            self._cache_valid = True
            
            return CovarianceResult(
                matrix=cov_matrix,
                assets=self.assets,
                window_size=len(self._buffer),
                is_positive_definite=self._is_positive_definite(cov_matrix),
                computation_time_ns=elapsed_ns,
                condition_number=self._condition_number(cov_matrix)
            )


def compute_portfolio_variance(cov_matrix: np.ndarray, 
                               weights: np.ndarray) -> float:
    """
    Compute portfolio variance given covariance matrix and weights.
    
    Args:
        cov_matrix: Covariance matrix (n_assets x n_assets)
        weights: Portfolio weights (n_assets,)
        
    Returns:
        Portfolio variance
    """
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights.T @ cov_matrix @ weights)


def compute_marginal_contribution(cov_matrix: np.ndarray,
                                   weights: np.ndarray) -> np.ndarray:
    """
    Compute marginal contribution to risk for each asset.
    
    Args:
        cov_matrix: Covariance matrix
        weights: Portfolio weights
        
    Returns:
        Array of marginal contributions
    """
    weights = np.asarray(weights, dtype=np.float64)
    return cov_matrix @ weights


if __name__ == "__main__":
    # Self-test
    np.random.seed(42)
    
    # Generate synthetic returns
    n_samples = 100
    returns = np.random.randn(n_samples, 4) * 0.02  # 2% daily vol
    
    # Test rolling covariance
    engine = RollingCovarianceEngine(window_size=50)
    
    for i in range(n_samples):
        engine.add_returns(returns[i])
    
    result = engine.compute()
    if result:
        print(f"Covariance shape: {result.matrix.shape}")
        print(f"Positive definite: {result.is_positive_definite}")
        print(f"Condition number: {result.condition_number:.2f}")
        print(f"Computation time: {result.computation_time_ns}ns")
    
    # Test EWMA
    ewma = ExponentialWeightedCovariance(span=21)
    
    for i in range(n_samples):
        ewma.add_returns(returns[i])
    
    ewma_result = ewma.compute()
    if ewma_result:
        print(f"\nEWMA Covariance computed")
        print(f"Diagonal: {np.diag(ewma_result.matrix)}")
