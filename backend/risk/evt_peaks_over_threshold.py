#!/usr/bin/env python3
"""
Extreme Value Theory - Peaks Over Threshold (POT) Method

Fits Generalized Pareto Distributions (GPD) to extreme losses
for accurate fat-tail risk quantification in crypto markets.
Optimized for limited extreme sample data scenarios.

Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import genpareto
from scipy.optimize import minimize_scalar, brentq
import warnings

# Suppress numerical warnings for production stability
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class GPDParameters:
    """Fitted Generalized Pareto Distribution parameters."""
    
    xi: float  # Shape parameter (tail index)
    sigma: float  # Scale parameter
    threshold: float  # Threshold used for fitting
    n_exceedances: int  # Number of exceedances used
    log_likelihood: float  # Log-likelihood at optimum
    
    @property
    def has_heavy_tail(self) -> bool:
        """Check if distribution has heavy tail (xi > 0)."""
        return self.xi > 0
    
    @property
    def tail_index(self) -> float:
        """Get the tail index (1/xi for xi > 0)."""
        if self.xi <= 0:
            return float('inf')
        return 1.0 / self.xi
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'xi': self.xi,
            'sigma': self.sigma,
            'threshold': self.threshold,
            'n_exceedances': self.n_exceedances,
            'log_likelihood': self.log_likelihood,
            'tail_index': self.tail_index,
        }


@dataclass
class EVTFitResult:
    """Complete EVT fitting results."""
    
    gpd_params: GPDParameters
    threshold_method: str
    goodness_of_fit: Dict[str, float]
    return_levels: Dict[float, float]  # probability -> loss level
    expected_shortfall: float  # ES at high confidence
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for logging."""
        return {
            'gpd_params': self.gpd_params.to_dict(),
            'threshold_method': self.threshold_method,
            'goodness_of_fit': self.goodness_of_fit,
            'return_levels': self.return_levels,
            'expected_shortfall': self.expected_shortfall,
        }


class PeaksOverThreshold:
    """
    Peaks Over Threshold (POT) method for Extreme Value Theory.
    
    Fits Generalized Pareto Distribution to exceedances over
    a high threshold, providing accurate tail risk estimates
    even with limited extreme observations.
    """
    
    def __init__(
        self,
        min_exceedances: int = 30,
        max_threshold_percentile: float = 95.0,
        min_threshold_percentile: float = 70.0,
    ) -> None:
        """
        Initialize POT estimator.
        
        Args:
            min_exceedances: Minimum number of exceedances required
            max_threshold_percentile: Maximum threshold percentile to consider
            min_threshold_percentile: Minimum threshold percentile to consider
        """
        self.min_exceedances = min_exceedances
        self.max_threshold_percentile = max_threshold_percentile
        self.min_threshold_percentile = min_threshold_percentile
        
        # Cache for fitted results
        self._latest_result: Optional[EVTFitResult] = None
        self._excess_data: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None
    
    def fit(
        self,
        returns: np.ndarray,
        threshold: Optional[float] = None,
        threshold_percentile: Optional[float] = None,
        method: str = 'mle'
    ) -> EVTFitResult:
        """
        Fit GPD to exceedances over threshold.
        
        Args:
            returns: Array of returns (negative values are losses)
            threshold: Absolute threshold value (optional)
            threshold_percentile: Percentile for threshold selection (optional)
            method: Fitting method ('mle' or 'pwm')
            
        Returns:
            Fitted EVT results
        """
        # Work with losses (negative returns)
        losses = -returns
        extreme_losses = losses[losses > 0]  # Only positive losses
        
        if len(extreme_losses) < self.min_exceedances * 2:
            raise ValueError(
                f"Insufficient data for EVT fitting. "
                f"Need at least {self.min_exceedances * 2} observations."
            )
        
        # Determine threshold
        if threshold is not None:
            self._threshold = threshold
        elif threshold_percentile is not None:
            self._threshold = np.percentile(extreme_losses, threshold_percentile)
        else:
            # Automatic threshold selection
            self._threshold = self._select_threshold_optimal(extreme_losses)
        
        # Extract exceedances
        exceedances = extreme_losses[extreme_losses > self._threshold] - self._threshold
        self._excess_data = exceedances
        
        if len(exceedances) < self.min_exceedances:
            raise ValueError(
                f"Too few exceedances ({len(exceedances)}). "
                f"Lower the threshold or collect more data."
            )
        
        # Fit GPD
        if method == 'mle':
            xi, sigma = self._fit_mle(exceedances)
        elif method == 'pwm':
            xi, sigma = self._fit_pwm(exceedances)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        # Compute log-likelihood
        log_likelihood = self._log_likelihood(exceedances, xi, sigma)
        
        # Create GPD parameters
        gpd_params = GPDParameters(
            xi=xi,
            sigma=sigma,
            threshold=self._threshold,
            n_exceedances=len(exceedances),
            log_likelihood=log_likelihood,
        )
        
        # Compute goodness of fit
        gof = self._goodness_of_fit_test(exceedances, xi, sigma)
        
        # Compute return levels
        return_levels = self._compute_return_levels(gpd_params)
        
        # Compute Expected Shortfall
        es = self._compute_expected_shortfall(gpd_params, confidence=0.99)
        
        result = EVTFitResult(
            gpd_params=gpd_params,
            threshold_method='automatic' if threshold_percentile is None else 'percentile',
            goodness_of_fit=gof,
            return_levels=return_levels,
            expected_shortfall=es,
        )
        
        self._latest_result = result
        return result
    
    def _select_threshold_optimal(self, losses: np.ndarray) -> float:
        """
        Select optimal threshold using goodness-of-fit criterion.
        
        Balances bias (low threshold) vs variance (high threshold).
        """
        percentiles = np.linspace(
            self.min_threshold_percentile,
            self.max_threshold_percentile,
            10
        )
        
        best_threshold = losses.min()
        best_score = float('-inf')
        
        for p in percentiles:
            threshold = np.percentile(losses, p)
            exceedances = losses[losses > threshold] - threshold
            
            if len(exceedances) < self.min_exceedances:
                continue
            
            # Fit and evaluate
            try:
                xi, sigma = self._fit_mle(exceedances)
                
                # Score based on stability and fit quality
                if xi < 0 or sigma <= 0:
                    continue
                
                # Penalize extreme xi values
                score = -abs(xi - 0.3)**2 + len(exceedances) * 0.01
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except Exception:
                continue
        
        return best_threshold
    
    def _fit_mle(self, exceedances: np.ndarray) -> Tuple[float, float]:
        """Fit GPD using Maximum Likelihood Estimation."""
        if len(exceedances) < 3:
            return 0.0, np.std(exceedances)
        
        # Initial guesses
        xi0 = 0.3
        sigma0 = np.mean(exceedances)
        
        def neg_log_likelihood(params):
            xi, sigma = params
            if sigma <= 0:
                return 1e10
            
            try:
                ll = genpareto.loglikelihood(xi, loc=0, scale=sigma, data=exceedances)
                return -ll
            except Exception:
                return 1e10
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[xi0, sigma0],
            method='Nelder-Mead',
            bounds=[(-0.5, 1.0), (1e-6, None)]
        )
        
        if not result.success:
            # Fallback to method of moments
            return self._fit_method_of_moments(exceedances)
        
        xi, sigma = result.x
        return max(-0.5, min(xi, 1.0)), max(1e-6, sigma)
    
    def _fit_pwm(self, exceedances: np.ndarray) -> Tuple[float, float]:
        """Fit GPD using Probability Weighted Moments."""
        n = len(exceedances)
        if n < 3:
            return 0.0, np.std(exceedances)
        
        # Sort exceedances
        x_sorted = np.sort(exceedances)
        
        # Compute PWMs
        b1 = np.mean(x_sorted)
        b2 = np.mean(np.arange(1, n + 1) * x_sorted) / n
        
        # Estimate parameters
        if b1 > 0:
            xi = 2.0 - b1**2 / (b2 - b1**2 / 2) if (b2 - b1**2 / 2) != 0 else 0.0
            sigma = b1 * (1 - xi) if xi < 1 else b1
        else:
            xi, sigma = 0.0, b1
        
        return max(-0.5, min(xi, 1.0)), max(1e-6, abs(sigma))
    
    def _fit_method_of_moments(self, exceedances: np.ndarray) -> Tuple[float, float]:
        """Fallback: Method of moments estimation."""
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)
        
        if var_exc <= 0 or mean_exc <= 0:
            return 0.0, max(1e-6, mean_exc)
        
        cv_sq = var_exc / (mean_exc ** 2)
        
        xi = (cv_sq - 1) / (cv_sq + 1)
        sigma = mean_exc * (1 - xi)
        
        return max(-0.5, min(xi, 1.0)), max(1e-6, abs(sigma))
    
    def _log_likelihood(
        self,
        exceedances: np.ndarray,
        xi: float,
        sigma: float
    ) -> float:
        """Compute log-likelihood of GPD fit."""
        try:
            return float(genpareto.loglikelihood(xi, loc=0, scale=sigma, data=exceedances))
        except Exception:
            return float('-inf')
    
    def _goodness_of_fit_test(
        self,
        exceedances: np.ndarray,
        xi: float,
        sigma: float
    ) -> Dict[str, float]:
        """Perform goodness-of-fit tests."""
        result = {}
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, p_value = self._ks_test(exceedances, xi, sigma)
            result['ks_statistic'] = ks_stat
            result['ks_pvalue'] = p_value
            result['ks_pass'] = p_value > 0.05
        except Exception:
            result['ks_statistic'] = float('nan')
            result['ks_pvalue'] = float('nan')
            result['ks_pass'] = False
        
        # Anderson-Darling (simplified)
        try:
            ad_stat = self._ad_statistic(exceedances, xi, sigma)
            result['ad_statistic'] = ad_stat
            result['ad_pass'] = ad_stat < 2.5
        except Exception:
            result['ad_statistic'] = float('nan')
            result['ad_pass'] = False
        
        return result
    
    def _ks_test(
        self,
        data: np.ndarray,
        xi: float,
        sigma: float
    ) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for GPD fit."""
        n = len(data)
        
        # Empirical CDF
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Theoretical CDF
        theoretical_cdf = genpareto.cdf(sorted_data, xi, loc=0, scale=sigma)
        
        # KS statistic
        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        # Approximate p-value (asymptotic)
        lambda_ks = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * ks_stat
        p_value = 2 * np.exp(-2 * lambda_ks**2)
        
        return ks_stat, min(1.0, max(0.0, p_value))
    
    def _ad_statistic(
        self,
        data: np.ndarray,
        xi: float,
        sigma: float
    ) -> float:
        """Anderson-Darling statistic (simplified)."""
        n = len(data)
        sorted_data = np.sort(data)
        
        # Theoretical CDF values
        u = genpareto.cdf(sorted_data, xi, loc=0, scale=sigma)
        u = np.clip(u, 1e-10, 1 - 1e-10)
        
        # AD statistic
        i = np.arange(1, n + 1)
        ad = -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n
        
        return max(0.0, ad)
    
    def _compute_return_levels(
        self,
        params: GPDParameters
    ) -> Dict[float, float]:
        """Compute return levels for various probabilities."""
        return_levels = {}
        
        for prob in [0.90, 0.95, 0.99, 0.995, 0.999]:
            try:
                # Return level = threshold + GPD quantile
                z_q = genpareto.ppf(prob, params.xi, loc=0, scale=params.sigma)
                return_level = params.threshold + z_q
                return_levels[prob] = float(return_level)
            except Exception:
                return_levels[prob] = float('nan')
        
        return return_levels
    
    def _compute_expected_shortfall(
        self,
        params: GPDParameters,
        confidence: float = 0.99
    ) -> float:
        """
        Compute Expected Shortfall (ES) at given confidence level.
        
        For GPD: ES = VaR * (1/(1-xi) + sigma/(xi*VaR) - sigma/xi)
                 when xi != 0 and xi < 1
        """
        xi = params.xi
        sigma = params.sigma
        threshold = params.threshold
        
        if xi >= 1:
            # ES undefined for xi >= 1 (infinite mean)
            return float('inf')
        
        try:
            # VaR at confidence level
            var = threshold + genpareto.ppf(confidence, xi, loc=0, scale=sigma)
            
            if abs(xi) < 1e-10:
                # Exponential case (xi = 0)
                es = threshold + sigma + sigma * np.log(1 - confidence) / (1 - confidence)
            else:
                # General GPD case
                es = var / (1 - xi) + (sigma - xi * threshold) / (1 - xi)
            
            return float(es)
        except Exception:
            return float('nan')
    
    def get_tail_probability(self, loss_level: float) -> Optional[float]:
        """
        Get probability of exceeding a given loss level.
        
        Args:
            loss_level: Loss threshold of interest
            
        Returns:
            P(Loss > loss_level) or None if not fitted
        """
        if self._latest_result is None or self._threshold is None:
            return None
        
        params = self._latest_result.gpd_params
        
        if loss_level <= params.threshold:
            # Use empirical probability
            if self._excess_data is None:
                return None
            total_obs = len(self._excess_data) + sum(1 for x in [-1] if False)  # Placeholder
            return None
        
        # GPD tail probability
        excess = loss_level - params.threshold
        try:
            survival = genpareto.sf(excess, params.xi, loc=0, scale=params.sigma)
            return float(survival)
        except Exception:
            return None
    
    def reset(self) -> None:
        """Reset all state."""
        self._latest_result = None
        self._excess_data = None
        self._threshold = None


def fit_crypto_tail_risk(
    returns: np.ndarray,
    asset_name: str = 'Portfolio'
) -> EVTFitResult:
    """
    Convenience function to fit EVT model for crypto returns.
    
    Args:
        returns: Array of crypto returns
        asset_name: Name for logging
        
    Returns:
        Complete EVT fit results
    """
    pot = PeaksOverThreshold(
        min_exceedances=30,
        max_threshold_percentile=95.0,
        min_threshold_percentile=75.0,
    )
    
    result = pot.fit(returns, method='mle')
    
    print(f"EVT Fit for {asset_name}:")
    print(f"  Threshold: {result.gpd_params.threshold:.4f}")
    print(f"  Shape (xi): {result.gpd_params.xi:.4f}")
    print(f"  Scale (sigma): {result.gpd_params.sigma:.4f}")
    print(f"  Tail Index: {result.gpd_params.tail_index:.2f}")
    print(f"  Heavy Tail: {result.gpd_params.has_heavy_tail}")
    print(f"  ES(99%): {result.expected_shortfall:.4f}")
    
    return result


if __name__ == '__main__':
    # Example usage with simulated crypto returns
    np.random.seed(42)
    
    # Simulate fat-tailed returns (Student-t with df=3)
    n_samples = 5000
    returns = np.random.standard_t(df=3, size=n_samples) * 0.02
    
    # Add some extreme events
    returns[np.random.choice(n_samples, 20)] *= 5
    
    result = fit_crypto_tail_risk(returns, 'BTC/USDT')
    
    print("\nReturn Levels:")
    for prob, level in result.return_levels.items():
        print(f"  {prob*100:.1f}%: {level:.4f}")
