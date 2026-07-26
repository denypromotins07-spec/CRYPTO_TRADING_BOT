#!/usr/bin/env python3
"""
Return Level Calculator for Extreme Value Theory

Estimates 1-in-100 day catastrophic loss probabilities using
fitted EVT models (GPD and GEV) for accurate tail risk assessment.
Optimized for crypto market fat-tail characteristics.

Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.stats import genpareto, genextreme
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ReturnPeriod(Enum):
    """Common return periods for extreme events."""
    ONE_DAY = 1
    ONE_WEEK = 7
    ONE_MONTH = 21
    ONE_QUARTER = 63
    ONE_YEAR = 252
    TWO_YEAR = 504
    FIVE_YEAR = 1260
    TEN_YEAR = 2520
    HUNDRED_DAY = 100


@dataclass
class ReturnLevelResult:
    """Result of return level calculation."""
    
    return_period: int  # In days
    return_level: float  # Loss level
    confidence_interval: Tuple[float, float]
    probability: float  # Daily exceedance probability
    method: str  # 'GPD' or 'GEV'
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for logging."""
        return {
            'return_period_days': self.return_period,
            'return_level': self.return_level,
            'confidence_lower': self.confidence_interval[0],
            'confidence_upper': self.confidence_interval[1],
            'daily_probability': self.probability,
            'method': self.method,
        }


@dataclass
class CatastrophicLossEstimate:
    """Estimate of catastrophic loss probabilities."""
    
    one_in_100_day: float  # 1% daily exceedance
    one_in_500_day: float  # 0.2% daily exceedance
    one_in_1000_day: float  # 0.1% daily exceedance
    one_in_2520_day: float  # ~0.04% (10-year event)
    expected_shortfall_99: float
    expected_shortfall_99_9: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'p1_in_100': self.one_in_100_day,
            'p1_in_500': self.one_in_500_day,
            'p1_in_1000': self.one_in_1000_day,
            'p1_in_2520': self.one_in_2520_day,
            'es_99': self.expected_shortfall_99,
            'es_99_9': self.expected_shortfall_99_9,
        }


class ReturnLevelCalculator:
    """
    Calculate return levels for extreme loss events.
    
    Supports both Peaks Over Threshold (GPD) and 
    Block Maxima (GEV) approaches.
    """
    
    def __init__(
        self,
        gpd_params: Optional[Dict[str, float]] = None,
        gev_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize calculator with fitted parameters.
        
        Args:
            gpd_params: Dictionary with 'xi', 'sigma', 'threshold' for GPD
            gev_params: Dictionary with 'xi', 'mu', 'sigma' for GEV
        """
        self.gpd_params = gpd_params
        self.gev_params = gev_params
        
        if gpd_params is None and gev_params is None:
            raise ValueError("Must provide either GPD or GEV parameters")
    
    def calculate_return_level(
        self,
        return_period: int,
        method: str = 'auto',
        n_blocks: int = 252,
    ) -> ReturnLevelResult:
        """
        Calculate return level for specified return period.
        
        Args:
            return_period: Return period in days
            method: 'GPD', 'GEV', or 'auto'
            n_blocks: Number of blocks per year for GEV
            
        Returns:
            Return level result with confidence intervals
        """
        if method == 'auto':
            method = 'GPD' if self.gpd_params is not None else 'GEV'
        
        if method == 'GPD' and self.gpd_params is not None:
            return self._gpd_return_level(return_period)
        elif method == 'GEV' and self.gev_params is not None:
            return self._gev_return_level(return_period, n_blocks)
        else:
            raise ValueError(f"Cannot use method {method}: parameters not available")
    
    def _gpd_return_level(self, return_period: int) -> ReturnLevelResult:
        """Calculate return level using GPD (POT approach)."""
        xi = self.gpd_params['xi']
        sigma = self.gpd_params['sigma']
        threshold = self.gpd_params['threshold']
        
        # Daily exceedance probability
        p_daily = 1.0 / return_period
        
        # For POT: need to account for threshold exceedance rate
        # Assuming roughly 5% of observations exceed threshold
        zeta_u = 0.05
        
        # Effective probability for GPD
        p_gpd = p_daily / zeta_u
        p_gpd = min(p_gpd, 0.999)  # Cap at reasonable value
        
        # GPD quantile
        try:
            z_q = genpareto.ppf(1 - p_gpd, xi, loc=0, scale=sigma)
            return_level = threshold + z_q
        except Exception:
            # Fallback approximation
            if abs(xi) < 1e-10:
                z_q = -sigma * np.log(p_gpd)
            else:
                z_q = sigma * ((p_gpd ** (-xi) - 1) / xi)
            return_level = threshold + z_q
        
        # Confidence interval (delta method approximation)
        ci_lower, ci_upper = self._gpd_confidence_interval(
            return_level, return_period
        )
        
        return ReturnLevelResult(
            return_period=return_period,
            return_level=float(return_level),
            confidence_interval=(ci_lower, ci_upper),
            probability=p_daily,
            method='GPD',
        )
    
    def _gev_return_level(
        self,
        return_period: int,
        n_blocks: int = 252
    ) -> ReturnLevelResult:
        """Calculate return level using GEV (Block Maxima approach)."""
        xi = self.gev_params['xi']
        mu = self.gev_params['mu']
        sigma = self.gev_params['sigma']
        
        # Convert return period to number of blocks
        # If block size is 1 day, then m = return_period
        # If block size is different, adjust accordingly
        m = return_period  # Assuming daily blocks
        
        # GEV return level formula
        try:
            if abs(xi) < 1e-10:
                # Gumbel case
                return_level = mu - sigma * np.log(-np.log(1 - 1/m))
            else:
                return_level = mu + sigma * (m**xi - 1) / xi
        except Exception:
            # Fallback
            return_level = mu + sigma * np.log(m)
        
        # Confidence interval
        ci_lower, ci_upper = self._gev_confidence_interval(
            return_level, return_period, n_blocks
        )
        
        return ReturnLevelResult(
            return_period=return_period,
            return_level=float(return_level),
            confidence_interval=(ci_lower, ci_upper),
            probability=1.0 / return_period,
            method='GEV',
        )
    
    def _gpd_confidence_interval(
        self,
        return_level: float,
        return_period: int
    ) -> Tuple[float, float]:
        """Approximate confidence interval for GPD return level."""
        # Delta method approximation
        # Simplified: use 10% relative error as placeholder
        rel_error = 0.1 + 0.05 * np.log10(return_period)
        rel_error = min(rel_error, 0.5)  # Cap at 50%
        
        lower = return_level * (1 - rel_error)
        upper = return_level * (1 + rel_error)
        
        return max(0, lower), upper
    
    def _gev_confidence_interval(
        self,
        return_level: float,
        return_period: int,
        n_blocks: int
    ) -> Tuple[float, float]:
        """Approximate confidence interval for GEV return level."""
        # Delta method approximation
        rel_error = 0.15 + 0.08 * np.log10(return_period)
        rel_error = min(rel_error, 0.6)
        
        lower = return_level * (1 - rel_error)
        upper = return_level * (1 + rel_error)
        
        return max(0, lower), upper
    
    def get_catastrophic_estimates(self) -> CatastrophicLossEstimate:
        """
        Get comprehensive catastrophic loss estimates.
        
        Returns estimates for various extreme return periods
        and expected shortfall measures.
        """
        # Calculate return levels for key periods
        rl_100 = self.calculate_return_level(100)
        rl_500 = self.calculate_return_level(500)
        rl_1000 = self.calculate_return_level(1000)
        rl_2520 = self.calculate_return_level(2520)
        
        # Expected Shortfall calculations
        es_99 = self._compute_expected_shortfall(0.99)
        es_999 = self._compute_expected_shortfall(0.999)
        
        return CatastrophicLossEstimate(
            one_in_100_day=rl_100.return_level,
            one_in_500_day=rl_500.return_level,
            one_in_1000_day=rl_1000.return_level,
            one_in_2520_day=rl_2520.return_level,
            expected_shortfall_99=es_99,
            expected_shortfall_99_9=es_999,
        )
    
    def _compute_expected_shortfall(self, confidence: float) -> float:
        """Compute Expected Shortfall at given confidence level."""
        if self.gpd_params is not None:
            return self._es_gpd(confidence)
        elif self.gev_params is not None:
            return self._es_gev(confidence)
        else:
            return float('nan')
    
    def _es_gpd(self, confidence: float) -> float:
        """Expected Shortfall using GPD."""
        xi = self.gpd_params['xi']
        sigma = self.gpd_params['sigma']
        threshold = self.gpd_params['threshold']
        
        if xi >= 1:
            return float('inf')
        
        # VaR at confidence level
        p = 1 - confidence
        try:
            var = threshold + genpareto.ppf(1 - p, xi, loc=0, scale=sigma)
        except Exception:
            if abs(xi) < 1e-10:
                var = threshold - sigma * np.log(p)
            else:
                var = threshold + sigma * (p**(-xi) - 1) / xi
        
        # ES formula for GPD
        if abs(xi) < 1e-10:
            es = threshold + sigma + sigma * np.log(p) / p
        else:
            es = var / (1 - xi) + (sigma - xi * threshold) / (1 - xi)
        
        return float(es)
    
    def _es_gev(self, confidence: float) -> float:
        """Expected Shortfall using GEV (approximation)."""
        xi = self.gev_params['xi']
        mu = self.gev_params['mu']
        sigma = self.gev_params['sigma']
        
        if xi >= 1:
            return float('inf')
        
        # VaR approximation
        p = 1 - confidence
        try:
            var = genextreme.ppf(1 - p, xi, loc=mu, scale=sigma)
        except Exception:
            if abs(xi) < 1e-10:
                var = mu - sigma * np.log(-np.log(1 - p))
            else:
                var = mu + sigma * ((-np.log(1 - p))**(-xi) - 1) / xi
        
        # ES approximation (GEV ES has no closed form)
        es = var + sigma / (1 - xi) if xi < 1 else float('inf')
        
        return float(es)
    
    def plot_return_level_curve(
        self,
        max_period: int = 2520,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for return level curve plotting.
        
        Args:
            max_period: Maximum return period to compute
            n_points: Number of points to generate
            
        Returns:
            Arrays of (periods, return_levels) for plotting
        """
        periods = np.logspace(0, np.log10(max_period), n_points).astype(int)
        return_levels = np.array([
            self.calculate_return_level(int(p)).return_level
            for p in periods
        ])
        
        return periods, return_levels


def estimate_crypto_catastrophic_risk(
    returns: np.ndarray,
    asset_name: str = 'Portfolio',
    use_gpd: bool = True
) -> CatastrophicLossEstimate:
    """
    Convenience function to estimate catastrophic risk for crypto assets.
    
    Args:
        returns: Array of crypto returns
        asset_name: Name for logging
        use_gpd: If True, use GPD (POT); else use GEV (Block Maxima)
        
    Returns:
        Comprehensive catastrophic loss estimates
    """
    from evt_peaks_over_threshold import PeaksOverThreshold
    
    if use_gpd:
        # Fit GPD using POT
        pot = PeaksOverThreshold(min_exceedances=30)
        result = pot.fit(returns)
        
        calculator = ReturnLevelCalculator(
            gpd_params={
                'xi': result.gpd_params.xi,
                'sigma': result.gpd_params.sigma,
                'threshold': result.gpd_params.threshold,
            }
        )
    else:
        # Use simple GEV approximation
        # In production, would use proper block maxima fitting
        losses = -returns
        block_size = 21  # Monthly blocks
        n_blocks = len(losses) // block_size
        block_maxima = [
            np.max(losses[i*block_size:(i+1)*block_size])
            for i in range(n_blocks)
        ]
        
        # Simple moment-based GEV fit
        mu = np.mean(block_maxima)
        sigma = np.std(block_maxima)
        xi = 0.2  # Typical for crypto
        
        calculator = ReturnLevelCalculator(
            gev_params={'xi': xi, 'mu': mu, 'sigma': sigma}
        )
    
    estimates = calculator.get_catastrophic_estimates()
    
    print(f"\nCatastrophic Risk Estimates for {asset_name}:")
    print(f"  1-in-100 day loss: {estimates.one_in_100_day:.4f}")
    print(f"  1-in-500 day loss: {estimates.one_in_500_day:.4f}")
    print(f"  1-in-1000 day loss: {estimates.one_in_1000_day:.4f}")
    print(f"  1-in-2520 day loss: {estimates.one_in_2520_day:.4f}")
    print(f"  ES(99%): {estimates.expected_shortfall_99:.4f}")
    print(f"  ES(99.9%): {estimates.expected_shortfall_99_9:.4f}")
    
    return estimates


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Simulate crypto returns with fat tails
    n_samples = 5000
    returns = np.random.standard_t(df=3, size=n_samples) * 0.02
    
    # Add extreme events
    extreme_indices = np.random.choice(n_samples, 30)
    returns[extreme_indices] *= 4
    
    estimates = estimate_crypto_catastrophic_risk(returns, 'BTC/USDT')
    
    print("\nDetailed breakdown:")
    for key, value in estimates.to_dict().items():
        print(f"  {key}: {value:.6f}")
