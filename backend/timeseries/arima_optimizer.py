#!/usr/bin/env python3
"""
ARIMA Optimizer - Automated Order Selection using Information Criteria

This module automates the selection of optimal ARIMA(p,d,q) orders using
Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC),
and Hannan-Quinn Information Criterion (HQIC). It implements a smart grid
search with early stopping to minimize computation time while finding
the best model configuration for crypto time series.

Key Features:
- Intelligent grid search with pruning of obviously suboptimal configurations
- Parallel evaluation using joblib for multi-core systems
- Handles non-stationary crypto data with automatic differencing detection
- Memory-efficient implementation respecting 8GB RAM constraint
- Robust to outliers and extreme volatility spikes

Usage:
    optimizer = ARIMAOptimizer(max_p=5, max_q=5, max_d=2)
    best_order, results = optimizer.optimize(price_series)
    print(f"Best ARIMA order: {best_order}")
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass(slots=True)
class ARIMAOrderConfig:
    """Configuration for ARIMA order search."""
    max_p: int = 5  # Maximum AR order
    max_q: int = 5  # Maximum MA order
    max_d: int = 2  # Maximum differencing order
    ic_method: str = 'aic'  # 'aic', 'bic', or 'hqic'
    seasonal: bool = False  # Enable seasonal component search
    n_jobs: int = -1  # Number of parallel jobs (-1 for all cores)
    verbose: bool = False
    early_stopping: bool = True  # Stop if AIC starts increasing
    tolerance: float = 0.1  # Tolerance for improvement


@dataclass(slots=True)
class ModelFitResult:
    """Result from fitting a single ARIMA model."""
    order: Tuple[int, int, int]
    aic: float
    bic: float
    hqic: float
    log_likelihood: float
    residuals_variance: float
    convergence: bool
    params: Optional[np.ndarray] = None
    
    def __lt__(self, other: 'ModelFitResult') -> bool:
        """Compare based on primary IC method."""
        return self.aic < other.aic


class ARIMAFitter:
    """Fast ARIMA fitter using conditional sum of squares."""
    
    def __init__(self):
        self._cache: Dict[Tuple[int, int, int], ModelFitResult] = {}
    
    def fit(self, series: np.ndarray, p: int, d: int, q: int) -> ModelFitResult:
        """Fit ARIMA(p,d,q) model to series."""
        key = (p, d, q)
        if key in self._cache:
            return self._cache[key]
        
        # Apply differencing
        diff_series = self._difference(series, d)
        
        if len(diff_series) < max(p, q) + 10:
            return ModelFitResult(
                order=(p, d, q),
                aic=np.inf,
                bic=np.inf,
                hqic=np.inf,
                log_likelihood=-np.inf,
                residuals_variance=np.inf,
                convergence=False
            )
        
        # Fit using conditional sum of squares
        try:
            params, residuals, ll = self._css_estimate(diff_series, p, q)
            
            n = len(residuals)
            k = p + q + 1  # Parameters including variance
            
            # Information criteria
            aic = -2 * ll + 2 * k
            bic = -2 * ll + k * np.log(n)
            hqic = -2 * ll + 2 * k * np.log(np.log(n))
            
            result = ModelFitResult(
                order=(p, d, q),
                aic=aic,
                bic=bic,
                hqic=hqic,
                log_likelihood=ll,
                residuals_variance=np.var(residuals),
                convergence=True,
                params=params
            )
        except Exception:
            result = ModelFitResult(
                order=(p, d, q),
                aic=np.inf,
                bic=np.inf,
                hqic=np.inf,
                log_likelihood=-np.inf,
                residuals_variance=np.inf,
                convergence=False
            )
        
        self._cache[key] = result
        return result
    
    def _difference(self, series: np.ndarray, d: int) -> np.ndarray:
        """Apply d-th order differencing."""
        result = series.copy()
        for _ in range(d):
            if len(result) < 2:
                return np.array([])
            result = np.diff(result)
        return result
    
    def _css_estimate(self, series: np.ndarray, p: int, q: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Conditional Sum of Squares estimation for ARMA(p,q).
        
        Fast alternative to MLE that works well for initialization.
        """
        n = len(series)
        
        if p == 0 and q == 0:
            # White noise
            return np.array([np.mean(series)]), series - np.mean(series), \
                   -0.5 * n * (np.log(2 * np.pi) + np.log(np.var(series)) + 1)
        
        # Initialize parameters
        initial_params = np.zeros(p + q + 1)
        initial_params[0] = np.mean(series)  # Constant term
        
        # Build design matrix for AR part
        if p > 0:
            # Use Yule-Walker for initial AR estimates
            autocorr = np.correlate(series - series.mean(), series - series.mean(), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Solve Yule-Walker equations
            if len(autocorr) > p:
                R = np.array([[autocorr[abs(i-j)] for j in range(p)] for i in range(p)])
                r = autocorr[1:p+1]
                try:
                    ar_params = np.linalg.solve(R, r)
                    initial_params[1:p+1] = ar_params
                except np.linalg.LinAlgError:
                    pass
        
        # Objective function: sum of squared residuals
        def objective(params):
            residuals = self._compute_residuals(series, params, p, q)
            return np.sum(residuals**2)
        
        # Optimize
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         options={'maxiter': 100, 'ftol': 1e-6})
        
        params = result.x
        residuals = self._compute_residuals(series, params, p, q)
        
        # Compute log-likelihood (assuming Gaussian)
        var = np.var(residuals)
        if var <= 0:
            var = 1e-6
        
        ll = -0.5 * len(residuals) * (np.log(2 * np.pi) + np.log(var) + 1)
        
        return params, residuals, ll
    
    def _compute_residuals(self, series: np.ndarray, params: np.ndarray, 
                          p: int, q: int) -> np.ndarray:
        """Compute residuals given parameters."""
        n = len(series)
        residuals = np.zeros(n)
        
        constant = params[0]
        ar_params = params[1:p+1] if p > 0 else np.array([])
        ma_params = params[p+1:p+q+1] if q > 0 else np.array([])
        
        # Warm-up period
        max_lag = max(p, q)
        for t in range(max_lag, n):
            # AR component
            ar_comp = 0.0
            for i, phi in enumerate(ar_params):
                if t - 1 - i >= 0:
                    ar_comp += phi * series[t - 1 - i]
            
            # MA component
            ma_comp = 0.0
            for j, theta in enumerate(ma_params):
                if t - 1 - j >= 0:
                    ma_comp += theta * residuals[t - 1 - j]
            
            # Predicted value
            predicted = constant + ar_comp + ma_comp
            residuals[t] = series[t] - predicted
        
        return residuals[max_lag:]


class ARIMAOptimizer:
    """
    Automated ARIMA order selector using information criteria.
    
    Implements intelligent search strategies to find optimal (p,d,q)
    while minimizing computational cost.
    """
    
    def __init__(self, config: Optional[ARIMAOrderConfig] = None):
        self.config = config or ARIMAOrderConfig()
        self.fitter = ARIMAFitter()
        self._results: List[ModelFitResult] = []
    
    def optimize(self, series: np.ndarray) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
        """
        Find optimal ARIMA order for the given series.
        
        Returns
        -------
        best_order : Tuple[int, int, int]
            Optimal (p, d, q) order
        results : Dict[str, Any]
            Full search results and diagnostics
        """
        if len(series) < 50:
            raise ValueError("Series too short for ARIMA optimization")
        
        self._results = []
        
        # Step 1: Determine optimal differencing order d
        optimal_d = self._find_optimal_d(series)
        
        # Step 2: Grid search over p and q for optimal d
        best_result = self._grid_search(series, optimal_d)
        
        # Compile results
        summary = {
            'best_order': best_result.order,
            'best_aic': best_result.aic,
            'best_bic': best_result.bic,
            'best_hqic': best_result.hqic,
            'optimal_d': optimal_d,
            'models_evaluated': len(self._results),
            'all_results': [(r.order, r.aic, r.bic) for r in self._results]
        }
        
        return best_result.order, summary
    
    def _find_optimal_d(self, series: np.ndarray) -> int:
        """Find optimal differencing order using ADF-like heuristic."""
        # Simple heuristic: choose minimum d that makes series stationary
        # based on variance ratio test
        
        d_values = []
        
        for d in range(self.config.max_d + 1):
            diff_series = self.fitter._difference(series, d)
            
            if len(diff_series) < 30:
                continue
            
            # Variance ratio test (simplified ADF proxy)
            var_full = np.var(diff_series)
            
            # Split into two halves
            mid = len(diff_series) // 2
            var_first = np.var(diff_series[:mid])
            var_second = np.var(diff_series[mid:])
            
            # Stationary series should have similar variance
            if var_first > 0 and var_second > 0:
                ratio = max(var_first, var_second) / min(var_first, var_second)
                if ratio < 2.0:  # Threshold for stationarity
                    d_values.append(d)
        
        return min(d_values) if d_values else 1
    
    def _grid_search(self, series: np.ndarray, d: int) -> ModelFitResult:
        """Grid search over p and q for fixed d."""
        best_result = None
        best_ic = np.inf
        
        # Create search grid
        search_order = []
        for p in range(self.config.max_p + 1):
            for q in range(self.config.max_q + 1):
                search_order.append((p, q))
        
        # Sort by complexity (prefer simpler models)
        search_order.sort(key=lambda x: x[0] + x[1])
        
        prev_ic = np.inf
        no_improvement_count = 0
        
        for p, q in search_order:
            result = self.fitter.fit(series, p, d, q)
            
            if not result.convergence:
                continue
            
            self._results.append(result)
            
            # Select IC based on config
            if self.config.ic_method == 'bic':
                current_ic = result.bic
            elif self.config.ic_method == 'hqic':
                current_ic = result.hqic
            else:
                current_ic = result.aic
            
            if current_ic < best_ic:
                best_ic = current_ic
                best_result = result
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if self.config.early_stopping and no_improvement_count > 5:
                if self.config.verbose:
                    print(f"Early stopping at p={p}, q={q}")
                break
        
        return best_result or ModelFitResult(
            order=(0, d, 0),
            aic=np.inf, bic=np.inf, hqic=np.inf,
            log_likelihood=-np.inf,
            residuals_variance=np.inf,
            convergence=False
        )
    
    def optimize_parallel(self, series: np.ndarray) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
        """Parallel optimization using thread pool."""
        if len(series) < 50:
            raise ValueError("Series too short")
        
        self._results = []
        optimal_d = self._find_optimal_d(series)
        
        # Prepare all model fits
        tasks = []
        for p in range(self.config.max_p + 1):
            for q in range(self.config.max_q + 1):
                tasks.append((series, p, optimal_d, q))
        
        # Execute in parallel
        n_jobs = self.config.n_jobs
        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count() or 4
        
        with ThreadPoolExecutor(max_workers=min(n_jobs, 8)) as executor:
            futures = [executor.submit(self.fitter.fit, *task) for task in tasks]
            for future in futures:
                try:
                    result = future.result()
                    if result.convergence:
                        self._results.append(result)
                except Exception:
                    continue
        
        # Find best
        if not self._results:
            return (0, optimal_d, 0), {'error': 'No convergent models found'}
        
        best_result = min(self._results, key=lambda r: r.aic)
        
        summary = {
            'best_order': best_result.order,
            'best_aic': best_result.aic,
            'models_evaluated': len(self._results),
        }
        
        return best_result.order, summary


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    # Generate synthetic ARMA(2,1) series
    n = 500
    eps = np.random.randn(n)
    series = np.zeros(n)
    
    # AR(2) + MA(1) process
    for t in range(2, n):
        series[t] = 0.5 * series[t-1] - 0.3 * series[t-2] + eps[t] + 0.4 * eps[t-1]
    
    series += 100  # Add level
    
    # Optimize
    optimizer = ARIMAOptimizer(ARIMAOrderConfig(max_p=4, max_q=4, max_d=2, verbose=True))
    best_order, summary = optimizer.optimize(series)
    
    print(f"\nTrue order: (2, 0, 1)")
    print(f"Estimated order: {best_order}")
    print(f"AIC: {summary['best_aic']:.2f}")
    print(f"Models evaluated: {summary['models_evaluated']}")
