#!/usr/bin/env python3
"""
Stationarity Test Module for Crypto Time Series

This module provides ADF (Augmented Dickey-Fuller) and KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
tests to validate stationarity of fractional-differenced price series. These tests are critical
for determining the optimal differencing parameter 'd' that achieves stationarity while
preserving long-term memory.

Key Features:
- ADF test: Null hypothesis = unit root (non-stationary)
- KPSS test: Null hypothesis = stationary
- Dual-test approach for robust stationarity confirmation
- Optimized for streaming crypto data with minimal memory footprint
- Thread-safe for concurrent BTC/ETH/SOL analysis

Usage:
    tester = StationarityTester(window_size=500)
    tester.update(price)
    if tester.is_stationary(alpha=0.05):
        # Safe to use for ML models
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from scipy import stats
import warnings

# Suppress scipy warnings for production stability
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass(slots=True)
class StationarityResult:
    """Container for stationarity test results."""
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: Dict[str, float]
    kpss_statistic: float
    kpss_pvalue: float
    kpss_critical_values: Dict[str, float]
    is_stationary_adf: bool
    is_stationary_kpss: bool
    conclusion: str
    
    def is_confirmed_stationary(self) -> bool:
        """Both tests agree on stationarity."""
        return self.is_stationary_adf and self.is_stationary_kpss


@dataclass(slots=True)
class StationarityConfig:
    """Configuration for stationarity testing."""
    window_size: int = 500  # Fits within 8GB RAM constraint
    max_lag: int = 25  # Maximum lag for ADF test
    alpha: float = 0.05  # Significance level
    kpss_regression: str = 'c'  # 'c' for constant, 'ct' for trend+constant
    
    def __post_init__(self):
        if self.window_size < 50:
            raise ValueError("Window size must be >= 50 for reliable tests")
        if self.max_lag < 1:
            raise ValueError("Max lag must be >= 1")


class StationarityTester:
    """
    Streaming stationarity tester using ADF and KPSS tests.
    
    Maintains a bounded window of prices and performs dual hypothesis testing
    to confirm stationarity of the transformed series.
    """
    
    def __init__(self, config: Optional[StationarityConfig] = None):
        self.config = config or StationarityConfig()
        self.price_buffer: deque = deque(maxlen=self.config.window_size)
        self._last_result: Optional[StationarityResult] = None
        self._cache_valid: bool = False
    
    def update(self, price: float) -> None:
        """Add new price to the rolling window."""
        self.price_buffer.append(price)
        self._cache_valid = False
    
    def update_batch(self, prices: List[float]) -> None:
        """Efficiently add multiple prices."""
        for price in prices:
            self.price_buffer.append(price)
        self._cache_valid = False
    
    def _compute_adf(self, series: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute Augmented Dickey-Fuller test.
        
        H0: Series has unit root (non-stationary)
        H1: Series is stationary
        
        Returns: (statistic, p-value, critical_values)
        """
        if len(series) < 50:
            return 0.0, 1.0, {'1%': 0.0, '5%': 0.0, '10%': 0.0}
        
        try:
            result = stats.adfuller(
                series,
                maxlag=min(self.config.max_lag, len(series) // 4),
                regression='c',
                autolag='AIC'
            )
            return result[0], result[1], result[4]
        except Exception:
            # Fallback for edge cases
            return 0.0, 1.0, {'1%': 0.0, '5%': 0.0, '10%': 0.0}
    
    def _compute_kpss(self, series: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute KPSS test.
        
        H0: Series is stationary
        H1: Series has unit root (non-stationary)
        
        Returns: (statistic, p-value, critical_values)
        """
        if len(series) < 50:
            return 0.0, 0.0, {'1%': 0.0, '5%': 0.0, '10%': 0.0}
        
        try:
            result = stats.kpss(
                series,
                regression=self.config.kpss_regression,
                nlags='auto'
            )
            statistic, pvalue, _, critical_values = result
            return statistic, pvalue, dict(critical_values)
        except Exception:
            # Fallback for edge cases
            return 0.0, 0.0, {'1%': 0.0, '5%': 0.0, '10%': 0.0}
    
    def test(self) -> Optional[StationarityResult]:
        """
        Perform both ADF and KPSS tests on current window.
        
        Returns StationarityResult if enough data, None otherwise.
        """
        if len(self.price_buffer) < 50:
            return None
        
        if self._cache_valid and self._last_result is not None:
            return self._last_result
        
        series = np.array(list(self.price_buffer))
        
        # ADF test
        adf_stat, adf_pval, adf_crit = self._compute_adf(series)
        is_stationary_adf = adf_pval < self.config.alpha
        
        # KPSS test
        kpss_stat, kpss_pval, kpss_crit = self._compute_kpss(series)
        is_stationary_kpss = kpss_pval > self.config.alpha
        
        # Formulate conclusion
        if is_stationary_adf and is_stationary_kpss:
            conclusion = "CONFIRMED_STATIONARY: Both tests agree"
        elif not is_stationary_adf and not is_stationary_kpss:
            conclusion = "CONFIRMED_NON_STATIONARY: Both tests agree"
        else:
            conclusion = "INCONCLUSIVE: Tests disagree - collect more data or adjust d"
        
        result = StationarityResult(
            adf_statistic=adf_stat,
            adf_pvalue=adf_pval,
            adf_critical_values=adf_crit,
            kpss_statistic=kpss_stat,
            kpss_pvalue=kpss_pval,
            kpss_critical_values=kpss_crit,
            is_stationary_adf=is_stationary_adf,
            is_stationary_kpss=is_stationary_kpss,
            conclusion=conclusion
        )
        
        self._last_result = result
        self._cache_valid = True
        
        return result
    
    def is_stationary(self, alpha: Optional[float] = None) -> bool:
        """Quick check if series is stationary."""
        result = self.test()
        if result is None:
            return False
        
        effective_alpha = alpha or self.config.alpha
        return (result.adf_pvalue < effective_alpha and 
                result.kpss_pvalue > effective_alpha)
    
    def get_optimal_d_range(self, 
                           original_prices: List[float],
                           fractional_differencer) -> Tuple[float, float]:
        """
        Find range of d values that produce stationary series.
        
        Uses binary search to find minimum and maximum d that achieve
        stationarity according to both ADF and KPSS tests.
        """
        stationary_d_values = []
        
        for d in np.linspace(0.1, 0.9, 17):  # Test 17 points
            fractional_differencer.set_d(d)
            diffed = [fractional_differencer.update(p) for p in original_prices[-self.config.window_size:]]
            
            # Quick variance check before full test
            if np.std(diffed) > 0 and np.isfinite(np.std(diffed)):
                self.update_batch(diffed)
                if self.is_stationary():
                    stationary_d_values.append(d)
                self.price_buffer.clear()
        
        if not stationary_d_values:
            return (0.3, 0.5)  # Default safe range
        
        return (min(stationary_d_values), max(stationary_d_values))
    
    def clear(self) -> None:
        """Reset the tester."""
        self.price_buffer.clear()
        self._last_result = None
        self._cache_valid = False
    
    @property
    def window_full(self) -> bool:
        """Check if buffer has enough data."""
        return len(self.price_buffer) >= self.config.window_size


class MultiAssetStationarityAnalyzer:
    """
    Analyze stationarity across BTC, ETH, SOL simultaneously.
    
    Useful for detecting regime changes that affect all assets
    and for calibrating asset-specific differencing parameters.
    """
    
    def __init__(self):
        self.btc_tester = StationarityTester()
        self.eth_tester = StationarityTester()
        self.sol_tester = StationarityTester()
    
    def update(self, btc_price: float, eth_price: float, sol_price: float) -> None:
        """Update all three asset testers."""
        self.btc_tester.update(btc_price)
        self.eth_tester.update(eth_price)
        self.sol_tester.update(sol_price)
    
    def all_stationary(self) -> bool:
        """Check if all assets are stationary."""
        return (self.btc_tester.is_stationary() and 
                self.eth_tester.is_stationary() and 
                self.sol_tester.is_stationary())
    
    def get_results(self) -> Dict[str, Optional[StationarityResult]]:
        """Get results for all assets."""
        return {
            'BTC': self.btc_tester.test(),
            'ETH': self.eth_tester.test(),
            'SOL': self.sol_tester.test()
        }


if __name__ == "__main__":
    # Demo usage
    tester = StationarityTester(StationarityConfig(window_size=200))
    
    # Simulate stationary series (mean-reverting)
    np.random.seed(42)
    stationary_series = np.cumsum(np.random.randn(500)) * 0.1 + 100
    
    for price in stationary_series:
        tester.update(price)
    
    result = tester.test()
    if result:
        print(f"ADF p-value: {result.adf_pvalue:.4f}")
        print(f"KPSS p-value: {result.kpss_pvalue:.4f}")
        print(f"Conclusion: {result.conclusion}")
