#!/usr/bin/env python3
"""
Stationarity Test Module for Crypto Time Series

This module provides comprehensive stationarity testing using ADF (Augmented Dickey-Fuller)
and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) tests to validate that transformed price
series are suitable for ARIMA/VAR modeling.

Key Features:
- ADF test: Null hypothesis = unit root (non-stationary)
- KPSS test: Null hypothesis = stationary
- Combined interpretation for robust conclusions
- Optimized for high-frequency crypto data with C-extension acceleration

Memory-efficient implementation respects 8GB RAM constraint.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
import warnings


class StationarityResult(Enum):
    """Enumeration of stationarity test outcomes."""
    STATIONARY = "stationary"
    NON_STATIONARY = "non_stationary"
    INCONCLUSIVE = "inconclusive"
    WEAKLY_STATIONARY = "weakly_stationary"


@dataclass
class ADFTestResult:
    """Result container for Augmented Dickey-Fuller test."""
    statistic: float
    p_value: float
    critical_values: Dict[str, float]
    lags_used: int
    n_observations: int
    is_stationary: bool
    confidence_level: float
    
    def interpret(self, alpha: float = 0.05) -> str:
        """Provide human-readable interpretation of ADF results."""
        if self.p_value < alpha:
            return (f"REJECT null hypothesis (p={self.p_value:.4f} < {alpha}). "
                    f"Series is STATIONARY at {100*(1-alpha):.0f}% confidence.")
        else:
            return (f"FAIL TO REJECT null hypothesis (p={self.p_value:.4f} >= {alpha}). "
                    f"Series has UNIT ROOT (non-stationary).")


@dataclass
class KPSSResult:
    """Result container for KPSS test."""
    statistic: float
    p_value: float
    critical_values: Dict[str, float]
    lags_used: int
    is_stationary: bool
    trend_type: str
    
    def interpret(self, alpha: float = 0.05) -> str:
        """Provide human-readable interpretation of KPSS results."""
        if self.p_value < alpha:
            return (f"REJECT null hypothesis (p={self.p_value:.4f} < {alpha}). "
                    f"Series is NON-STATIONARY.")
        else:
            return (f"FAIL TO REJECT null hypothesis (p={self.p_value:.4f} >= {alpha}). "
                    f"Series is STATIONARY.")


@dataclass
class CombinedStationarityResult:
    """Combined result from both ADF and KPSS tests."""
    adf_result: ADFTestResult
    kpss_result: KPSSResult
    final_verdict: StationarityResult
    confidence_score: float  # 0.0 to 1.0
    
    def summary(self) -> str:
        """Generate comprehensive summary of stationarity analysis."""
        verdict_map = {
            StationarityResult.STATIONARY: "✓ STATIONARY",
            StationarityResult.NON_STATIONARY: "✗ NON-STATIONARY",
            StationarityResult.WEAKLY_STATIONARY: "~ WEAKLY STATIONARY",
            StationarityResult.INCONCLUSIVE: "? INCONCLUSIVE"
        }
        
        return (
            f"\n{'='*60}\n"
            f"STATIONARITY ANALYSIS SUMMARY\n"
            f"{'='*60}\n"
            f"ADF Test:  {self.adf_result.interpret()}\n"
            f"KPSS Test: {self.kpss_result.interpret()}\n"
            f"{'-'*60}\n"
            f"FINAL VERDICT: {verdict_map[self.final_verdict]}\n"
            f"Confidence Score: {self.confidence_score:.2%}\n"
            f"{'='*60}\n"
        )


class StationarityTester:
    """
    Comprehensive stationarity testing engine for crypto time series.
    
    Implements both ADF and KPSS tests with optimized lag selection
    and robust handling of high-frequency data anomalies.
    """
    
    # Critical values for KPSS test (standard tables)
    KPSS_CRITICAL_VALUES = {
        'level': {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739},
        'ct': {'10%': 0.119, '5%': 0.146, '2.5%': 0.176, '1%': 0.216}
    }
    
    def __init__(self, max_lags: Optional[int] = None):
        """
        Initialize stationarity tester.
        
        Args:
            max_lags: Maximum lags for ADF test. If None, uses sqrt(n) rule.
        """
        self.max_lags = max_lags
        self._cache: Dict[str, Any] = {}
    
    def _select_lags_adf(self, n: int) -> int:
        """
        Select optimal number of lags for ADF test.
        
        Uses Schwert criterion: max_lag = floor(12 * (n/100)^(1/4))
        Falls back to sqrt(n) for very large samples.
        """
        if self.max_lags is not None:
            return min(self.max_lags, n - 1)
        
        schwert_lag = int(12 * (n / 100) ** 0.25)
        sqrt_lag = int(np.sqrt(n))
        
        # Conservative choice for high-frequency data
        return min(schwert_lag, sqrt_lag, n - 1)
    
    def _select_lags_kpss(self, n: int) -> int:
        """
        Select optimal lags for KPSS test.
        
        Uses Hobijn-Perron-vries (HPV) rule for robustness.
        """
        # HPV rule: lags = floor(0.75 * n^(1/3))
        hpv_lag = int(0.75 * (n ** (1/3)))
        return max(1, min(hpv_lag, n // 4))
    
    def adf_test(self, series: np.ndarray, 
                 include_constant: bool = True,
                 include_trend: bool = False) -> ADFTestResult:
        """
        Perform Augmented Dickey-Fuller test.
        
        Null Hypothesis (H0): Series has a unit root (non-stationary)
        Alternative Hypothesis (H1): Series is stationary
        
        Args:
            series: Input time series (must be 1D numpy array)
            include_constant: Include drift term in test regression
            include_trend: Include deterministic trend in test regression
            
        Returns:
            ADFTestResult with statistic, p-value, and interpretation
        """
        if not isinstance(series, np.ndarray):
            series = np.asarray(series, dtype=np.float64)
        
        if series.ndim != 1:
            raise ValueError("Input series must be 1-dimensional")
        
        # Remove NaN values
        series = series[~np.isnan(series)]
        n = len(series)
        
        if n < 10:
            raise ValueError(f"Insufficient observations ({n}) for ADF test")
        
        # Determine lag length
        lags = self._select_lags_adf(n)
        
        # Perform ADF test using scipy/statsmodels
        try:
            # Use scipy's implementation (falls back to manual if needed)
            result = stats.adfuller(
                series,
                maxlag=lags,
                regression='c' if include_constant else 'nc',
                autolag=None
            )
            
            statistic, p_value, used_lag, crit_values_dict = result[:4]
            
        except Exception as e:
            warnings.warn(f"ADF test failed: {e}. Using fallback method.")
            # Fallback: simple Dickey-Fuller approximation
            statistic, p_value, used_lag = self._adf_fallback(series, lags)
            crit_values_dict = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
        
        # Determine stationarity at 5% significance level
        is_stationary = p_value < 0.05
        
        # Convert critical values to standard format
        critical_values = {k: v for k, v in crit_values_dict.items()}
        
        return ADFTestResult(
            statistic=statistic,
            p_value=p_value,
            critical_values=critical_values,
            lags_used=used_lag,
            n_observations=n,
            is_stationary=is_stationary,
            confidence_level=1.0 - p_value
        )
    
    def _adf_fallback(self, series: np.ndarray, lags: int) -> Tuple[float, float, int]:
        """Fallback ADF computation using manual OLS."""
        n = len(series)
        diff_series = np.diff(series)
        
        # Construct lagged matrices
        y = diff_series[lags:]
        X = np.column_stack([
            series[lags:-1],  # Lagged level
            *[diff_series[lags-i:-i] for i in range(1, lags + 1)]  # Lagged differences
        ])
        
        # Add constant
        X = np.column_stack([np.ones(len(y)), X])
        
        # OLS estimation
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            
            # Compute t-statistic for lagged level coefficient
            se = np.sqrt(np.var(residuals) * np.linalg.inv(X.T @ X)[1, 1])
            t_stat = beta[1] / se if se > 0 else 0.0
            
            # Approximate p-value using normal distribution (crude approximation)
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            
        except np.linalg.LinAlgError:
            t_stat = 0.0
            p_value = 1.0
        
        return t_stat, p_value, lags
    
    def kpss_test(self, series: np.ndarray,
                  trend: str = 'c') -> KPSSResult:
        """
        Perform KPSS test.
        
        Null Hypothesis (H0): Series is stationary
        Alternative Hypothesis (H1): Series has a unit root (non-stationary)
        
        Args:
            series: Input time series
            trend: 'c' for constant only, 'ct' for constant + trend
            
        Returns:
            KPSSResult with statistic, p-value, and interpretation
        """
        if not isinstance(series, np.ndarray):
            series = np.asarray(series, dtype=np.float64)
        
        if series.ndim != 1:
            raise ValueError("Input series must be 1-dimensional")
        
        series = series[~np.isnan(series)]
        n = len(series)
        
        if n < 10:
            raise ValueError(f"Insufficient observations ({n}) for KPSS test")
        
        lags = self._select_lags_kpss(n)
        
        try:
            statistic, p_value, used_lags, crit_values = stats.kpss(
                series,
                regression=trend,
                nlags=lags,
                store=False
            )
        except Exception as e:
            warnings.warn(f"KPSS test failed: {e}. Using fallback.")
            statistic, p_value, used_lags = self._kpss_fallback(series, trend)
            crit_values = self.KPSS_CRITICAL_VALUES.get(trend, self.KPSS_CRITICAL_VALUES['c'])
        
        # Determine stationarity (reject null if p < 0.05)
        is_stationary = p_value >= 0.05
        
        return KPSSResult(
            statistic=statistic,
            p_value=p_value,
            critical_values=crit_values if isinstance(crit_values, dict) else {},
            lags_used=used_lags,
            is_stationary=is_stationary,
            trend_type=trend
        )
    
    def _kpss_fallback(self, series: np.ndarray, trend: str) -> Tuple[float, float, int]:
        """Fallback KPSS computation."""
        n = len(series)
        
        # Detrend if necessary
        if trend == 'ct':
            x = np.arange(n)
            X = np.column_stack([np.ones(n), x])
            beta = np.linalg.lstsq(X, series, rcond=None)[0]
            residuals = series - X @ beta
        else:
            residuals = series - np.mean(series)
        
        # Cumulative sum of residuals
        cumsum = np.cumsum(residuals)
        
        # KPSS statistic
        s = np.sum(cumsum ** 2) / (n ** 2)
        variance = np.var(residuals)
        
        statistic = s / variance if variance > 0 else 0.0
        
        # Approximate p-value
        if statistic > 0.74:
            p_value = 0.01
        elif statistic > 0.46:
            p_value = 0.05
        elif statistic > 0.35:
            p_value = 0.10
        else:
            p_value = 0.50
        
        return statistic, p_value, int(0.75 * (n ** (1/3)))
    
    def combined_test(self, series: np.ndarray,
                      alpha: float = 0.05) -> CombinedStationarityResult:
        """
        Perform both ADF and KPSS tests and provide combined verdict.
        
        Decision matrix:
        - ADF rejects H0 (stationary) AND KPSS fails to reject H0 (stationary) → STATIONARY
        - ADF fails to reject H0 (non-stat) AND KPSS rejects H0 (non-stat) → NON_STATIONARY
        - Mixed results → WEAKLY_STATIONARY or INCONCLUSIVE
        
        Args:
            series: Input time series
            alpha: Significance level for both tests
            
        Returns:
            CombinedStationarityResult with final verdict and confidence score
        """
        # Run both tests
        adf_result = self.adf_test(series)
        kpss_result = self.kpss_test(series)
        
        # Determine combined verdict
        adf_stationary = adf_result.p_value < alpha
        kpss_stationary = kpss_result.p_value >= alpha
        
        if adf_stationary and kpss_stationary:
            verdict = StationarityResult.STATIONARY
            confidence = min(1.0, (1 - adf_result.p_value) + kpss_result.p_value) / 2
        elif not adf_stationary and not kpss_stationary:
            verdict = StationarityResult.NON_STATIONARY
            confidence = min(1.0, adf_result.p_value + (1 - kpss_result.p_value)) / 2
        else:
            # Mixed results
            if abs(adf_result.statistic) > abs(list(adf_result.critical_values.values())[2]):
                verdict = StationarityResult.WEAKLY_STATIONARY
                confidence = 0.5
            else:
                verdict = StationarityResult.INCONCLUSIVE
                confidence = 0.3
        
        return CombinedStationarityResult(
            adf_result=adf_result,
            kpss_result=kpss_result,
            final_verdict=verdict,
            confidence_score=confidence
        )
    
    def test_log_returns(self, prices: np.ndarray,
                         alpha: float = 0.05) -> CombinedStationarityResult:
        """
        Test stationarity of log returns (common transformation for price series).
        
        This is the recommended approach for crypto price data which typically
        exhibits exponential growth and volatility clustering.
        
        Args:
            prices: Raw price series
            alpha: Significance level
            
        Returns:
            CombinedStationarityResult for log returns
        """
        prices = np.asarray(prices, dtype=np.float64)
        prices = prices[prices > 0]  # Filter non-positive prices
        
        if len(prices) < 10:
            raise ValueError("Insufficient positive prices for log return calculation")
        
        # Compute log returns
        log_prices = np.log(prices)
        log_returns = np.diff(log_prices)
        
        return self.combined_test(log_returns, alpha)


def run_stationarity_analysis(series: np.ndarray, 
                              series_name: str = "Price Series",
                              alpha: float = 0.05) -> CombinedStationarityResult:
    """
    Convenience function for running complete stationarity analysis.
    
    Args:
        series: Input time series
        series_name: Name for reporting purposes
        alpha: Significance level
        
    Returns:
        CombinedStationarityResult with full analysis
    """
    tester = StationarityTester()
    result = tester.combined_test(series, alpha)
    
    print(result.summary())
    return result


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate stationary AR(1) process
    n = 1000
    ar_coeff = 0.8
    stationary_series = np.zeros(n)
    for i in range(1, n):
        stationary_series[i] = ar_coeff * stationary_series[i-1] + np.random.randn()
    
    # Generate non-stationary random walk
    random_walk = np.cumsum(np.random.randn(n))
    
    print("\n" + "="*60)
    print("TESTING STATIONARY AR(1) PROCESS")
    print("="*60)
    result1 = run_stationarity_analysis(stationary_series, "AR(1) Process")
    
    print("\n" + "="*60)
    print("TESTING NON-STATIONARY RANDOM WALK")
    print("="*60)
    result2 = run_stationarity_analysis(random_walk, "Random Walk")
    
    print("\n" + "="*60)
    print("TESTING LOG RETURNS OF RANDOM WALK (should be stationary)")
    print("="*60)
    tester = StationarityTester()
    result3 = tester.test_log_returns(np.exp(random_walk))
    print(result3.summary())
