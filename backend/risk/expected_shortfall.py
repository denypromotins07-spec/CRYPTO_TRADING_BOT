#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Advanced Risk Management
Chapter 1: Expected Shortfall (Conditional VaR) Calculator

This module computes Conditional Value at Risk (CVaR), also known as Expected Shortfall,
which measures the expected loss given that losses exceed the VaR threshold.
Unlike VaR, CVaR captures tail risk and satisfies coherence properties.

Memory Budget: <30MB for historical data structures
Target Latency: <500μs for CVaR calculation
Assets: BTC, SOL, ETH, USDT parallel processing
Integration: Feeds results to SOUL.md for risk lessons

Author: Opus 4.8
Stage: 5/100 - Advanced Risk Management and Order Book Microstructure
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress numpy warnings for production stability
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class CVaRConfig:
    """Configuration for Expected Shortfall calculations."""
    confidence_level: float = 0.99  # 99% confidence
    time_horizon_days: int = 1
    historical_window_size: int = 252  # ~1 trading year
    assets: List[str] = field(default_factory=lambda: ["BTC", "SOL", "ETH", "USDT"])
    use_ewma: bool = True  # Exponentially weighted moving average
    ewma_lambda: float = 0.94  # Decay factor for EWMA
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.9 <= self.confidence_level <= 0.999:
            raise ValueError("Confidence level must be between 0.9 and 0.999")
        if not 0.5 <= self.ewma_lambda <= 0.99:
            raise ValueError("EWMA lambda must be between 0.5 and 0.99")


@dataclass
class CVaRResult(TypedDict):
    """Type-safe result structure for CVaR calculations."""
    asset: str
    var: float  # Value at Risk
    cvar: float  # Conditional VaR (Expected Shortfall)
    tail_mean: float  # Mean of tail losses
    tail_std: float  # Standard deviation of tail losses
    observation_count: int
    timestamp: float


class ExpectedShortfallCalculator:
    """
    High-performance Expected Shortfall (CVaR) calculator.
    
    CVaR represents the expected loss given that losses exceed VaR.
    It provides a more comprehensive view of tail risk than VaR alone.
    
    Mathematical definition:
    CVaR_α = E[L | L > VaR_α]
    
    Where:
    - L is the loss distribution
    - α is the confidence level
    - VaR_α is the Value at Risk at confidence level α
    """
    
    __slots__ = (
        '_config', '_returns_history', '_cvar_cache', 
        '_ewma_weights', '_thread_pool'
    )
    
    def __init__(self, config: CVaRConfig = None) -> None:
        """
        Initialize the Expected Shortfall calculator.
        
        Args:
            config: CVaR configuration parameters
        """
        self._config = config or CVaRConfig()
        self._returns_history: Dict[str, deque] = {
            asset: deque(maxlen=self._config.historical_window_size)
            for asset in self._config.assets
        }
        self._cvar_cache: Dict[str, CVaRResult] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=4)  # One per asset
        
        # Precompute EWMA weights if enabled
        if self._config.use_ewma:
            self._ewma_weights = self._compute_ewma_weights()
        else:
            self._ewma_weights = None
    
    def _compute_ewma_weights(self) -> np.ndarray:
        """
        Compute exponentially weighted moving average weights.
        
        EWMA gives more weight to recent observations, making CVaR
        more responsive to changing market conditions.
        
        Returns:
            Array of normalized EWMA weights
        """
        n = self._config.historical_window_size
        lambda_val = self._config.ewma_lambda
        
        # EWMA weights: w_i = (1-λ) * λ^(n-i) for i = 1, ..., n
        weights = (1 - lambda_val) * np.power(lambda_val, np.arange(n - 1, -1, -1))
        
        # Normalize weights to sum to 1
        weights /= weights.sum()
        
        return weights
    
    def add_return(self, asset: str, return_value: float) -> None:
        """
        Add a new return observation for an asset.
        
        Args:
            asset: Asset identifier (BTC, SOL, ETH, USDT)
            return_value: Log return or simple return
        """
        if asset in self._returns_history:
            self._returns_history[asset].append(return_value)
            # Invalidate cache for this asset
            self._cvar_cache.pop(asset, None)
    
    def add_returns_batch(self, asset: str, returns: List[float]) -> None:
        """
        Add multiple return observations efficiently.
        
        Args:
            asset: Asset identifier
            returns: List of return values
        """
        if asset in self._returns_history:
            history = self._returns_history[asset]
            for ret in returns:
                history.append(ret)
            self._cvar_cache.pop(asset, None)
    
    def compute_var(self, returns: np.ndarray, confidence_level: float = None) -> float:
        """
        Compute Value at Risk using historical simulation.
        
        Args:
            returns: Array of historical returns
            confidence_level: Override default confidence level
            
        Returns:
            VaR as a positive number representing potential loss
        """
        if len(returns) == 0:
            return 0.0
        
        conf = confidence_level or self._config.confidence_level
        
        # VaR is the negative of the (1-confidence) percentile
        # For 99% confidence, we look at the 1st percentile
        percentile = (1 - conf) * 100
        var_threshold = np.percentile(returns, percentile)
        
        # VaR is expressed as positive loss
        return -var_threshold if var_threshold < 0 else 0.0
    
    def compute_cvar(self, returns: np.ndarray, confidence_level: float = None) -> CVaRResult:
        """
        Compute Conditional VaR (Expected Shortfall).
        
        This is the main computation method that calculates:
        1. VaR threshold
        2. All losses exceeding VaR
        3. Expected value of those tail losses
        
        Args:
            returns: Array of historical returns
            confidence_level: Override default confidence level
            
        Returns:
            CVaRResult with detailed risk metrics
        """
        if len(returns) == 0:
            return CVaRResult(
                asset="",
                var=0.0,
                cvar=0.0,
                tail_mean=0.0,
                tail_std=0.0,
                observation_count=0,
                timestamp=0.0
            )
        
        import time
        conf = confidence_level or self._config.confidence_level
        
        # Apply EWMA weights if configured
        if self._ewma_weights is not None and len(returns) == len(self._ewma_weights):
            # Weighted percentile calculation
            sorted_indices = np.argsort(returns)
            sorted_returns = returns[sorted_indices]
            sorted_weights = self._ewma_weights[sorted_indices]
            
            # Find VaR threshold with weights
            cumulative_weights = np.cumsum(sorted_weights)
            target_weight = 1 - conf
            var_index = np.searchsorted(cumulative_weights, target_weight)
            var_index = min(var_index, len(sorted_returns) - 1)
            var_threshold = sorted_returns[var_index]
        else:
            # Simple historical simulation
            percentile = (1 - conf) * 100
            var_threshold = np.percentile(returns, percentile)
        
        # Identify tail losses (losses exceeding VaR)
        tail_losses = returns[returns <= var_threshold]
        
        # Compute CVaR as mean of tail losses
        if len(tail_losses) > 0:
            if self._ewma_weights is not None and len(tail_losses) == len(self._ewma_weights[:len(tail_losses)]):
                # Weighted CVaR
                tail_weights = self._ewma_weights[:len(tail_losses)]
                tail_weights_normalized = tail_weights / tail_weights.sum()
                cvar_value = -np.sum(tail_losses * tail_weights_normalized)
            else:
                cvar_value = -np.mean(tail_losses)
        else:
            cvar_value = 0.0
        
        # Additional tail statistics
        tail_mean = -np.mean(tail_losses) if len(tail_losses) > 0 else 0.0
        tail_std = np.std(tail_losses) if len(tail_losses) > 1 else 0.0
        
        return CVaRResult(
            asset="",
            var=-var_threshold if var_threshold < 0 else 0.0,
            cvar=max(cvar_value, 0.0),
            tail_mean=max(tail_mean, 0.0),
            tail_std=tail_std,
            observation_count=len(returns),
            timestamp=time.time()
        )
    
    def get_cvar(self, asset: str) -> Optional[CVaRResult]:
        """
        Get cached CVaR result for an asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            CVaRResult or None if not computed
        """
        return self._cvar_cache.get(asset)
    
    def update_cvar(self, asset: str) -> Optional[CVaRResult]:
        """
        Compute and cache CVaR for an asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Updated CVaRResult
        """
        if asset not in self._returns_history:
            return None
        
        returns_array = np.array(self._returns_history[asset], dtype=np.float64)
        
        if len(returns_array) < 10:  # Minimum samples for meaningful CVaR
            return None
        
        result = self.compute_cvar(returns_array)
        result['asset'] = asset
        
        self._cvar_cache[asset] = result
        return result
    
    def get_portfolio_cvar(self, weights: Dict[str, float]) -> float:
        """
        Compute portfolio-level CVaR with diversification effects.
        
        This uses the historical simulation approach on portfolio returns
        rather than simply summing individual CVaRs.
        
        Args:
            weights: Portfolio weights for each asset
            
        Returns:
            Portfolio CVaR
        """
        # Ensure all assets have sufficient data
        min_length = min(
            len(self._returns_history.get(asset, []))
            for asset in weights.keys()
        )
        
        if min_length < 10:
            return 0.0
        
        # Build aligned return series
        returns_matrix = np.zeros((min_length, len(weights)))
        
        for i, (asset, weight) in enumerate(weights.items()):
            if asset in self._returns_history:
                asset_returns = list(self._returns_history[asset])[-min_length:]
                returns_matrix[:, i] = np.array(asset_returns) * weight
        
        # Portfolio returns (sum of weighted asset returns)
        portfolio_returns = returns_matrix.sum(axis=1)
        
        # Compute CVaR on portfolio returns
        result = self.compute_cvar(portfolio_returns)
        return result['cvar']
    
    async def update_all_assets_async(self) -> Dict[str, CVaRResult]:
        """
        Asynchronously update CVaR for all assets.
        
        Uses thread pool for parallel computation without blocking.
        
        Returns:
            Dictionary of asset -> CVaRResult
        """
        loop = asyncio.get_event_loop()
        
        def compute_for_asset(asset: str) -> Tuple[str, Optional[CVaRResult]]:
            result = self.update_cvar(asset)
            return (asset, result)
        
        tasks = [
            loop.run_in_executor(self._thread_pool, compute_for_asset, asset)
            for asset in self._config.assets
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            asset: result
            for asset, result in results
            if result is not None
        }
    
    def stress_test_cvar(self, asset: str, shock_scenarios: List[float]) -> Dict[float, CVaRResult]:
        """
        Stress test CVaR under various shock scenarios.
        
        Args:
            asset: Asset to stress test
            shock_scenarios: List of shock percentages (e.g., [-0.1, -0.2, -0.3])
            
        Returns:
            Dictionary mapping shock levels to stressed CVaR
        """
        if asset not in self._returns_history:
            return {}
        
        base_returns = np.array(self._returns_history[asset], dtype=np.float64)
        results = {}
        
        for shock in shock_scenarios:
            # Apply shock to historical returns
            stressed_returns = base_returns + shock
            
            # Compute CVaR under stress
            stressed_result = self.compute_cvar(stressed_returns)
            stressed_result['asset'] = f"{asset}_shock_{shock:.2f}"
            
            results[shock] = stressed_result
        
        return results
    
    def get_tail_risk_ratio(self, asset: str) -> Optional[float]:
        """
        Compute the ratio of CVaR to VaR (tail risk indicator).
        
        A higher ratio indicates fatter tails and more extreme tail risk.
        For normal distributions, this ratio is approximately 1.0.
        For fat-tailed distributions (crypto), this can be 1.5-3.0+.
        
        Args:
            asset: Asset identifier
            
        Returns:
            CVaR/VaR ratio or None
        """
        if asset not in self._cvar_cache:
            self.update_cvar(asset)
        
        result = self._cvar_cache.get(asset)
        if result is None or result['var'] == 0:
            return None
        
        return result['cvar'] / result['var']
    
    def export_to_soul_md(self, output_path: str = "SOUL.md") -> None:
        """
        Export CVaR analysis results to SOUL.md for learning integration.
        
        Args:
            output_path: Path to SOUL.md file
        """
        import os
        
        # Update all CVaR values first
        for asset in self._config.assets:
            self.update_cvar(asset)
        
        # Generate report section
        timestamp = __import__('datetime').datetime.now().isoformat()
        
        report_lines = [
            f"## CVaR Analysis Report - {timestamp}",
            "",
            "| Asset | VaR (99%) | CVaR (99%) | Tail Risk Ratio | Observations |",
            "|-------|-----------|------------|-----------------|--------------|"
        ]
        
        for asset in self._config.assets:
            result = self._cvar_cache.get(asset)
            if result:
                tail_ratio = self.get_tail_risk_ratio(asset)
                report_lines.append(
                    f"| {asset} | {result['var']:.4f} | {result['cvar']:.4f} | "
                    f"{tail_ratio:.2f} | {result['observation_count']} |"
                )
        
        report_lines.extend([
            "",
            "### Tail Risk Lessons",
            "- CVaR > VaR indicates fat-tailed loss distributions",
            "- Higher tail risk ratios suggest need for wider stop-losses",
            "- Monitor changes in tail risk during high volatility periods",
            ""
        ])
        
        # Append to SOUL.md
        try:
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            else:
                existing_content = "# ZAID BOT SOUL.md - Trading Intelligence\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(existing_content + '\n'.join(report_lines))
        except Exception as e:
            # Silently fail to avoid disrupting trading operations
            print(f"Warning: Could not write to SOUL.md: {e}")


class RollingCVaREstimator(ExpectedShortfallCalculator):
    """
    Rolling window CVaR estimator for real-time risk monitoring.
    
    Maintains a rolling window of CVaR estimates to detect
    regime changes in tail risk.
    """
    
    __slots__ = ('_rolling_window', '_cvar_history')
    
    def __init__(self, config: CVaRConfig = None, rolling_window: int = 20) -> None:
        """
        Initialize rolling CVaR estimator.
        
        Args:
            config: CVaR configuration
            rolling_window: Number of CVaR estimates to keep in rolling window
        """
        super().__init__(config)
        self._rolling_window = rolling_window
        self._cvar_history: Dict[str, deque] = {
            asset: deque(maxlen=rolling_window)
            for asset in config.assets
        }
    
    def update_and_record(self, asset: str) -> Optional[float]:
        """
        Update CVaR and record in rolling history.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Current CVaR value or None
        """
        result = self.update_cvar(asset)
        if result:
            self._cvar_history[asset].append(result['cvar'])
            return result['cvar']
        return None
    
    def get_cvar_trend(self, asset: str) -> Optional[str]:
        """
        Determine CVaR trend from rolling history.
        
        Args:
            asset: Asset identifier
            
        Returns:
            'INCREASING', 'DECREASING', 'STABLE', or None
        """
        history = self._cvar_history.get(asset)
        if not history or len(history) < 3:
            return None
        
        recent_avg = np.mean(list(history)[-3:])
        older_avg = np.mean(list(history)[:3])
        
        change_pct = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
        
        if change_pct > 0.1:
            return 'INCREASING'
        elif change_pct < -0.1:
            return 'DECREASING'
        else:
            return 'STABLE'
    
    def is_tail_risk_escalating(self, asset: str, threshold: float = 0.2) -> bool:
        """
        Check if tail risk is escalating beyond threshold.
        
        Args:
            asset: Asset identifier
            threshold: Percentage increase threshold
            
        Returns:
            True if tail risk is escalating
        """
        history = self._cvar_history.get(asset)
        if not history or len(history) < 5:
            return False
        
        current_cvar = history[-1]
        baseline_cvar = np.mean(list(history)[:5])
        
        if baseline_cvar == 0:
            return False
        
        return (current_cvar - baseline_cvar) / baseline_cvar > threshold


if __name__ == "__main__":
    # Example usage and validation
    import random
    
    config = CVaRConfig(confidence_level=0.99)
    calculator = ExpectedShortfallCalculator(config)
    
    # Simulate historical returns (normal market conditions)
    for _ in range(200):
        for asset in ["BTC", "ETH", "SOL"]:
            # Normal returns with occasional large moves
            ret = random.gauss(0.001, 0.02)
            if random.random() < 0.05:  # 5% chance of large move
                ret *= 3
            calculator.add_return(asset, ret)
    
    # Compute CVaR for all assets
    for asset in ["BTC", "ETH", "SOL"]:
        result = calculator.update_cvar(asset)
        if result:
            print(f"{asset}: VaR={result['var']:.4f}, CVaR={result['cvar']:.4f}, "
                  f"Tail Ratio={calculator.get_tail_risk_ratio(asset):.2f}")
    
    # Portfolio CVaR
    weights = {"BTC": 0.4, "ETH": 0.4, "SOL": 0.2}
    port_cvar = calculator.get_portfolio_cvar(weights)
    print(f"\nPortfolio CVaR (99%): {port_cvar:.4f}")
    
    # Stress testing
    print("\nStress Testing BTC:")
    stress_results = calculator.stress_test_cvar("BTC", [-0.05, -0.10, -0.20])
    for shock, result in stress_results.items():
        print(f"  Shock {shock:.0%}: CVaR = {result['cvar']:.4f}")
