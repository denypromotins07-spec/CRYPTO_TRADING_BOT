#!/usr/bin/env python3
"""
Vector Autoregression (VAR) Model for Cross-Asset Lead-Lag Analysis

This module implements a high-performance VAR model to capture instantaneous
lead-lag relationships between BTC, ETH, and SOL. The VAR framework allows
us to model how shocks to one asset propagate through the entire crypto ecosystem.

Key Features:
- Microsecond-level matrix solving using NumPy's LAPACK bindings
- Handles singular covariance matrices during extreme correlation spikes
- Dynamic lag order selection using AIC/BIC criteria
- Impulse response function analysis for shock propagation
- Forecast error variance decomposition (FEVD) for attribution

Mathematical Foundation:
    Y_t = c + A_1 * Y_{t-1} + ... + A_p * Y_{t-p} + ε_t
    
    where Y_t = [BTC_return, ETH_return, SOL_return]'
          A_i = coefficient matrices
          p = lag order
          ε_t = white noise

Usage:
    var = VARModel(max_lags=10)
    var.fit(data)  # data shape: (n_samples, 3)
    forecasts = var.forecast(horizon=5)
    irf = var.impulse_response(periods=20)
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from numpy.linalg import inv, pinv, eigvals, LinAlgError
import warnings

# Suppress warnings for production stability
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass(slots=True)
class VARConfig:
    """Configuration for VAR model."""
    max_lags: int = 10  # Maximum lag order to consider
    ic_method: str = 'aic'  # 'aic', 'bic', or 'hqic' for lag selection
    trend: str = 'c'  # 'c' for constant, 'ct' for constant+trend, 'nc' for none
    min_lags: int = 1
    robust_cov: bool = True  # Use robust covariance estimation


@dataclass(slots=True)
class VARResult:
    """Container for VAR estimation results."""
    coefficients: np.ndarray  # Shape: (max_lags * n_vars + intercept, n_vars)
    residuals: np.ndarray
    sigma_u: np.ndarray  # Covariance matrix of residuals
    lag_order: int
    nobs: int
    aic: float
    bic: float
    hqic: float
    is_stable: bool
    eigenvalues: np.ndarray
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
VAR Model Summary
=================
Lag Order: {self.lag_order}
Observations: {self.nobs}
AIC: {self.aic:.4f}
BIC: {self.bic:.4f}
HQIC: {self.hqic:.4f}
Stable: {self.is_stable}
Max Eigenvalue: {np.abs(self.eigenvalues).max():.4f}
"""


class VARModel:
    """
    Vector Autoregression model for multi-asset time series.
    
    Optimized for crypto market microstructure with:
    - Fast OLS estimation using normal equations
    - Automatic lag order selection
    - Stability checking via eigenvalue analysis
    """
    
    ASSET_NAMES = ['BTC', 'ETH', 'SOL']
    
    def __init__(self, config: Optional[VARConfig] = None):
        self.config = config or VARConfig()
        self._result: Optional[VARResult] = None
        self._data: Optional[np.ndarray] = None
        self._mean: np.ndarray = np.zeros(3)
        self._is_fitted: bool = False
    
    def _select_lag_order(self, Y: np.ndarray) -> int:
        """Select optimal lag order using information criteria."""
        best_ic = np.inf
        best_lag = self.config.min_lags
        
        for p in range(self.config.min_lags, self.config.max_lags + 1):
            if len(Y) <= p * len(self.ASSET_NAMES) + 10:
                continue
            
            # Estimate VAR(p)
            X, y = self._prepare_data(Y, p)
            if X.shape[0] < X.shape[1] + 1:
                continue
            
            try:
                # OLS estimation
                beta = pinv(X.T @ X) @ X.T @ y
                residuals = y - X @ beta
                
                # Compute log-likelihood
                n = len(residuals)
                k = len(self.ASSET_NAMES)
                sigma_u = (residuals.T @ residuals) / n
                log_det = np.linalg.slogdet(sigma_u)[1]
                loglik = -0.5 * (k * np.log(2 * np.pi) + log_det + k)
                
                # Information criteria
                df = p * k**2 + k  # Parameters per equation
                
                if self.config.ic_method == 'aic':
                    ic = -2 * loglik * n + 2 * df
                elif self.config.ic_method == 'bic':
                    ic = -2 * loglik * n + df * np.log(n)
                else:  # hqic
                    ic = -2 * loglik * n + 2 * df * np.log(np.log(n))
                
                if ic < best_ic:
                    best_ic = ic
                    best_lag = p
                    
            except (LinAlgError, ValueError):
                continue
        
        return best_lag
    
    def _prepare_data(self, Y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare design matrix and response vector for VAR(p)."""
        n = len(Y) - p
        
        # Design matrix: [1, Y_{t-1}, ..., Y_{t-p}]
        X_blocks = [np.ones(n)]  # Intercept
        for lag in range(1, p + 1):
            X_blocks.append(Y[lag:n + lag])
        
        X = np.column_stack(X_blocks)
        
        # Response: Y_t
        y = Y[p:]
        
        return X, y
    
    def fit(self, data: np.ndarray) -> 'VARModel':
        """
        Fit VAR model to data.
        
        Parameters
        ----------
        data : np.ndarray
            Shape: (n_samples, 3) for BTC, ETH, SOL returns
        
        Returns
        -------
        self : VARModel
            Fitted model
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if data.shape[1] != 3:
            raise ValueError(f"Expected 3 columns (BTC, ETH, SOL), got {data.shape[1]}")
        
        self._data = data.copy()
        self._mean = data.mean(axis=0)
        
        # Demean data for better numerical stability
        Y = data - self._mean
        
        # Select optimal lag order
        optimal_lag = self._select_lag_order(Y)
        
        # Prepare data for optimal lag
        X, y = self._prepare_data(Y, optimal_lag)
        
        # OLS estimation: β = (X'X)^{-1} X'y
        try:
            XtX = X.T @ X
            # Add small regularization for singular matrices
            XtX += np.eye(XtX.shape[0]) * 1e-8
            beta = inv(XtX) @ X.T @ y
        except LinAlgError:
            # Fall back to pseudo-inverse for extreme cases
            beta = pinv(X) @ y
        
        # Compute residuals
        residuals = y - X @ beta
        
        # Residual covariance matrix
        n = len(residuals)
        sigma_u = (residuals.T @ residuals) / n
        
        # Check stability (all eigenvalues inside unit circle)
        companion = self._build_companion_matrix(beta, optimal_lag)
        eigenvalues = eigvals(companion)
        is_stable = np.all(np.abs(eigenvalues) < 1.0)
        
        # Information criteria
        k = len(self.ASSET_NAMES)
        df = optimal_lag * k**2 + k
        log_det = np.linalg.slogdet(sigma_u)[1]
        loglik = -0.5 * (k * np.log(2 * np.pi) + log_det + k) * n
        
        aic = -2 * loglik + 2 * df
        bic = -2 * loglik + df * np.log(n)
        hqic = -2 * loglik + 2 * df * np.log(np.log(n))
        
        self._result = VARResult(
            coefficients=beta,
            residuals=residuals,
            sigma_u=sigma_u,
            lag_order=optimal_lag,
            nobs=n,
            aic=aic,
            bic=bic,
            hqic=hqic,
            is_stable=is_stable,
            eigenvalues=eigenvalues
        )
        
        self._is_fitted = True
        return self
    
    def _build_companion_matrix(self, beta: np.ndarray, p: int) -> np.ndarray:
        """Build companion matrix for stability analysis."""
        k = len(self.ASSET_NAMES)
        
        # Extract coefficient matrices A_1, ..., A_p
        A_matrices = []
        for i in range(p):
            start_idx = 1 + i * k  # Skip intercept
            end_idx = start_idx + k
            A_matrices.append(beta[start_idx:end_idx, :].T)
        
        # Build companion matrix
        companion = np.zeros((k * p, k * p))
        
        # First block row: [A_1, A_2, ..., A_p]
        for i, A in enumerate(A_matrices):
            companion[:k, i*k:(i+1)*k] = A
        
        # Remaining rows: identity structure
        for i in range(1, p):
            companion[i*k:(i+1)*k, (i-1)*k:i*k] = np.eye(k)
        
        return companion
    
    def forecast(self, horizon: int = 5) -> np.ndarray:
        """
        Generate forecasts for all assets.
        
        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast
        
        Returns
        -------
        forecasts : np.ndarray
            Shape: (horizon, 3) for BTC, ETH, SOL
        """
        if not self._is_fitted or self._result is None:
            raise RuntimeError("Model must be fitted before forecasting")
        
        p = self._result.lag_order
        k = len(self.ASSET_NAMES)
        
        # Get last p observations
        if self._data is None or len(self._data) < p:
            raise ValueError("Insufficient data for forecasting")
        
        recent = self._data[-p:] - self._mean
        
        forecasts = np.zeros((horizon, k))
        
        for h in range(horizon):
            # Build predictor vector: [1, Y_{t-1}, ..., Y_{t-p}]
            if h == 0:
                pred = np.concatenate([[1], recent.flatten()])
            else:
                # Use previous forecasts
                needed = max(0, p - h)
                hist = recent[-needed:].flatten() if needed > 0 else np.array([])
                fc_hist = forecasts[max(0, h-p):h].flatten()
                pred = np.concatenate([[1], fc_hist, hist])
            
            # Ensure correct length
            expected_len = 1 + p * k
            if len(pred) < expected_len:
                pred = np.pad(pred, (0, expected_len - len(pred)), mode='edge')
            pred = pred[:expected_len]
            
            # Forecast
            forecasts[h] = pred @ self._result.coefficients
        
        # Add mean back
        forecasts += self._mean
        
        return forecasts
    
    def impulse_response(self, periods: int = 20) -> np.ndarray:
        """
        Compute impulse response functions.
        
        Measures how a one-standard-deviation shock to each variable
        affects all variables over time.
        
        Returns
        -------
        irf : np.ndarray
            Shape: (periods, n_vars, n_vars)
            irf[t, i, j] = response of variable i to shock in variable j at time t
        """
        if not self._is_fitted or self._result is None:
            raise RuntimeError("Model must be fitted first")
        
        p = self._result.lag_order
        k = len(self.ASSET_NAMES)
        
        # Cholesky decomposition for orthogonalization
        try:
            P = np.linalg.cholesky(self._result.sigma_u)
        except LinAlgError:
            # Use eigenvalue decomposition if not positive definite
            evals, evecs = np.linalg.eigh(self._result.sigma_u)
            evals = np.maximum(evals, 1e-8)
            P = evecs @ np.diag(np.sqrt(evals))
        
        # Extract coefficient matrices
        A = []
        for i in range(p):
            start_idx = 1 + i * k
            end_idx = start_idx + k
            A.append(self._result.coefficients[start_idx:end_idx, :].T)
        
        # Compute IRF recursively
        irf = np.zeros((periods, k, k))
        
        # Period 0: immediate impact
        irf[0] = P.T
        
        # Subsequent periods
        for t in range(1, periods):
            for j in range(min(t, p)):
                if t - 1 - j >= 0:
                    irf[t] += A[j] @ irf[t - 1 - j]
        
        return irf
    
    def fevd(self, periods: int = 20) -> np.ndarray:
        """
        Forecast Error Variance Decomposition.
        
        Attributes forecast error variance to shocks from each variable.
        
        Returns
        -------
        fevd : np.ndarray
            Shape: (periods, n_vars, n_vars)
            fevd[t, i, j] = % of variance in variable i due to shock in j
        """
        irf = self.impulse_response(periods)
        k = len(self.ASSET_NAMES)
        
        fevd = np.zeros((periods, k, k))
        
        for t in range(periods):
            # Cumulative squared IRF
            cum_irf_sq = sum(irf[s]**2 for s in range(t + 1))
            
            # Normalize to percentages
            row_sums = cum_irf_sq.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            
            fevd[t] = cum_irf_sq / row_sums * 100
        
        return fevd
    
    def get_lead_lag_matrix(self) -> np.ndarray:
        """
        Extract lead-lag relationship matrix.
        
        Returns coefficient matrix showing how each asset leads/lags others.
        Positive value in [i,j] means asset j leads asset i.
        """
        if not self._is_fitted or self._result is None:
            raise RuntimeError("Model must be fitted first")
        
        # Sum coefficient matrices across all lags
        p = self._result.lag_order
        k = len(self.ASSET_NAMES)
        
        total_effect = np.zeros((k, k))
        for i in range(p):
            start_idx = 1 + i * k
            end_idx = start_idx + k
            total_effect += self._result.coefficients[start_idx:end_idx, :].T
        
        return total_effect
    
    @property
    def is_stable(self) -> bool:
        """Check if VAR process is stable."""
        return self._result.is_stable if self._result else False
    
    @property
    def residual_covariance(self) -> np.ndarray:
        """Get residual covariance matrix."""
        return self._result.sigma_u if self._result else np.eye(3)


if __name__ == "__main__":
    # Demo usage with synthetic data
    np.random.seed(42)
    n = 500
    
    # Generate correlated returns
    btc = np.cumsum(np.random.randn(n) * 0.02)
    eth = btc * 0.8 + np.random.randn(n) * 0.03
    sol = btc * 0.6 + eth * 0.3 + np.random.randn(n) * 0.05
    
    data = np.column_stack([btc, eth, sol])
    
    # Fit VAR model
    var = VARModel(VARConfig(max_lags=5))
    var.fit(data)
    
    print(var._result.summary())
    
    # Forecast
    forecasts = var.forecast(5)
    print("\n5-step forecasts:")
    for i, name in enumerate(VARModel.ASSET_NAMES):
        print(f"  {name}: {forecasts[-1, i]:.4f}")
    
    # Check lead-lag
    llm = var.get_lead_lag_matrix()
    print("\nLead-Lag Matrix (row led by column):")
    print(llm)
