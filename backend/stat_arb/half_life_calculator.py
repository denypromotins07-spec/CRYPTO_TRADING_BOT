#!/usr/bin/env python3
"""
Half-Life Calculator - Estimating Mean-Reversion Speed for Dynamic Exits

This module calculates the half-life of mean reversion for spread series,
which determines optimal holding periods and exit timing for pairs trades.
Uses Ornstein-Uhlenbeck process estimation.

Chapter 3: Statistical Arbitrage, Pairs Trading, and Cointegration Execution
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
import time
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HalfLifeResult:
    """Result of half-life calculation"""
    symbol_pair: Tuple[str, str]
    half_life_periods: float  # In number of periods (e.g., bars)
    half_life_seconds: float  # In seconds
    mean_reversion_speed: float  # theta parameter
    long_term_mean: float  # mu parameter
    volatility: float  # sigma parameter
    r_squared: float  # Goodness of fit
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_mean_reverting(self) -> bool:
        """Check if series is mean-reverting (positive half-life)"""
        return self.half_life_periods > 0 and self.half_life_periods < 1000
    
    @property
    def optimal_exit_periods(self) -> float:
        """Suggest optimal exit timing (2x half-life)"""
        return max(1.0, self.half_life_periods * 2.0)


class HalfLifeCalculator:
    """
    Calculates mean reversion half-life using Ornstein-Uhlenbeck process.
    
    The OU process is defined as:
    dx_t = θ(μ - x_t)dt + σdW_t
    
    Where:
    - θ (theta): Mean reversion speed
    - μ (mu): Long-term mean
    - σ (sigma): Volatility
    - Half-life = ln(2) / θ
    
    Uses linear regression on lagged differences to estimate θ.
    """
    
    # Minimum data points for reliable estimation
    MIN_DATA_POINTS = 50
    
    # Maximum half-life to consider valid (in periods)
    MAX_HALF_LIFE = 500
    
    def __init__(self, window_size: int = 100):
        """
        Initialize half-life calculator.
        
        Args:
            window_size: Rolling window size for calculations
        """
        self.window_size = window_size
        
        # Storage for spread series per pair
        self._spread_history: Dict[Tuple[str, str], List[float]] = {}
        
        # Cached results
        self._last_results: Dict[Tuple[str, str], HalfLifeResult] = {}
        
        # Calculation timestamps
        self._last_calc_time: Dict[Tuple[str, str], float] = {}
        
        logger.info(f"HalfLifeCalculator initialized (window={window_size})")
    
    def _get_pair_key(self, symbol_a: str, symbol_b: str) -> Tuple[str, str]:
        """Get canonical pair key"""
        return tuple(sorted([symbol_a, symbol_b]))
    
    def add_spread_observation(
        self, 
        symbol_a: str, 
        symbol_b: str, 
        spread: float
    ) -> None:
        """Add a new spread observation"""
        pair_key = self._get_pair_key(symbol_a, symbol_b)
        
        if pair_key not in self._spread_history:
            self._spread_history[pair_key] = []
        
        history = self._spread_history[pair_key]
        history.append(spread)
        
        # Maintain window size
        if len(history) > self.window_size:
            history.pop(0)
    
    def calculate_half_life(
        self,
        symbol_a: str,
        symbol_b: str,
        period_seconds: float = 60.0  # Default: 1 minute bars
    ) -> Optional[HalfLifeResult]:
        """
        Calculate half-life for a pair.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            period_seconds: Time period of each observation in seconds
            
        Returns:
            HalfLifeResult or None if insufficient data
        """
        pair_key = self._get_pair_key(symbol_a, symbol_b)
        
        if pair_key not in self._spread_history:
            return None
        
        spread_series = self._spread_history[pair_key]
        
        if len(spread_series) < self.MIN_DATA_POINTS:
            logger.debug(f"Insufficient data for {pair_key}: {len(spread_series)} < {self.MIN_DATA_POINTS}")
            return None
        
        # Estimate OU parameters using linear regression
        # Δy_t = y_t - y_{t-1} regressed against y_{t-1}
        # Model: Δy_t = α + β * y_{t-1} + ε
        # Then: θ = -β, μ = α/θ
        
        y_lagged = spread_series[:-1]  # y_{t-1}
        y_diff = [spread_series[i] - spread_series[i-1] for i in range(1, len(spread_series))]  # Δy_t
        
        # Linear regression using least squares
        n = len(y_lagged)
        
        sum_x = sum(y_lagged)
        sum_y = sum(y_diff)
        sum_xy = sum(x * y for x, y in zip(y_lagged, y_diff))
        sum_xx = sum(x * x for x in y_lagged)
        
        # Calculate coefficients
        denominator = n * sum_xx - sum_x * sum_x
        
        if abs(denominator) < 1e-10:
            return None
        
        beta = (n * sum_xy - sum_x * sum_y) / denominator
        alpha = (sum_y - beta * sum_x) / n
        
        # OU parameters
        # θ = -β (mean reversion speed)
        theta = -beta
        
        # Check for valid mean reversion
        if theta <= 0:
            # Series is not mean-reverting (explosive or random walk)
            result = HalfLifeResult(
                symbol_pair=pair_key,
                half_life_periods=float('inf'),
                half_life_seconds=float('inf'),
                mean_reversion_speed=theta,
                long_term_mean=0.0,
                volatility=0.0,
                r_squared=0.0
            )
            self._last_results[pair_key] = result
            return result
        
        # Half-life = ln(2) / θ
        half_life_periods = math.log(2) / theta
        
        # Validate half-life
        if half_life_periods > self.MAX_HALF_LIFE:
            logger.debug(f"Half-life too long for {pair_key}: {half_life_periods:.1f} periods")
            result = HalfLifeResult(
                symbol_pair=pair_key,
                half_life_periods=half_life_periods,
                half_life_seconds=half_life_periods * period_seconds,
                mean_reversion_speed=theta,
                long_term_mean=alpha / theta if theta != 0 else 0.0,
                volatility=self._calculate_volatility(spread_series),
                r_squared=self._calculate_r_squared(y_lagged, y_diff, alpha, beta),
                timestamp=datetime.now(timezone.utc)
            )
            self._last_results[pair_key] = result
            return result
        
        # Long-term mean μ = α / θ
        mu = alpha / theta
        
        # Calculate volatility (σ)
        sigma = self._calculate_volatility(spread_series)
        
        # Calculate R-squared
        r_squared = self._calculate_r_squared(y_lagged, y_diff, alpha, beta)
        
        half_life_seconds = half_life_periods * period_seconds
        
        result = HalfLifeResult(
            symbol_pair=pair_key,
            half_life_periods=half_life_periods,
            half_life_seconds=half_life_seconds,
            mean_reversion_speed=theta,
            long_term_mean=mu,
            volatility=sigma,
            r_squared=r_squared,
            timestamp=datetime.now(timezone.utc)
        )
        
        self._last_results[pair_key] = result
        self._last_calc_time[pair_key] = time.time()
        
        logger.info(
            f"Half-life for {pair_key}: {half_life_periods:.2f} periods "
            f"({half_life_seconds:.1f}s), θ={theta:.4f}, R²={r_squared:.3f}"
        )
        
        return result
    
    def _calculate_volatility(self, series: List[float]) -> float:
        """Calculate standard deviation of series"""
        if len(series) < 2:
            return 0.0
        
        mean = sum(series) / len(series)
        variance = sum((x - mean) ** 2 for x in series) / (len(series) - 1)
        return math.sqrt(variance)
    
    def _calculate_r_squared(
        self, 
        x: List[float], 
        y: List[float], 
        alpha: float, 
        beta: float
    ) -> float:
        """Calculate R-squared for the regression"""
        if len(x) < 2:
            return 0.0
        
        # Predicted values
        y_pred = [alpha + beta * xi for xi in x]
        
        # Total sum of squares
        y_mean = sum(y) / len(y)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        
        # Residual sum of squares
        ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
        
        if ss_tot < 1e-10:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    
    def get_optimal_exit_time(
        self,
        symbol_a: str,
        symbol_b: str,
        confidence_level: float = 0.95
    ) -> Optional[float]:
        """
        Get optimal exit time based on half-life.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            confidence_level: Confidence level for mean reversion (default 95%)
            
        Returns:
            Optimal exit time in seconds, or None if not calculable
        """
        pair_key = self._get_pair_key(symbol_a, symbol_b)
        
        if pair_key not in self._last_results:
            # Need to calculate first
            result = self.calculate_half_life(symbol_a, symbol_b)
            if not result:
                return None
        else:
            result = self._last_results[pair_key]
        
        if not result.is_mean_reverting:
            return None
        
        # For 95% confidence, we need ~4.32 half-lives
        # For 99% confidence, we need ~6.64 half-lives
        confidence_multiplier = {
            0.90: 3.32,
            0.95: 4.32,
            0.99: 6.64
        }.get(confidence_level, 4.32)
        
        return result.half_life_seconds * confidence_multiplier
    
    def should_exit_position(
        self,
        symbol_a: str,
        symbol_b: str,
        holding_periods: float
    ) -> bool:
        """
        Determine if a position should be exited based on holding time.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            holding_periods: How long the position has been held (in periods)
            
        Returns:
            True if should exit
        """
        pair_key = self._get_pair_key(symbol_a, symbol_b)
        
        if pair_key not in self._last_results:
            return False
        
        result = self._last_results[pair_key]
        
        if not result.is_mean_reverting:
            return False
        
        # Exit if held longer than 2x half-life (diminishing returns)
        return holding_periods > (result.half_life_periods * 2.0)
    
    def get_all_results(self) -> Dict[Tuple[str, str], HalfLifeResult]:
        """Get all cached half-life results"""
        return dict(self._last_results)
    
    def clear_cache(self) -> None:
        """Clear cached results"""
        self._last_results.clear()
        self._last_calc_time.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get calculator metrics"""
        valid_pairs = sum(1 for r in self._last_results.values() if r.is_mean_reverting)
        
        avg_half_life = 0.0
        if valid_pairs > 0:
            avg_half_life = sum(
                r.half_life_periods for r in self._last_results.values() 
                if r.is_mean_reverting
            ) / valid_pairs
        
        return {
            'pairs_tracked': len(self._spread_history),
            'pairs_with_results': len(self._last_results),
            'valid_mean_reverting_pairs': valid_pairs,
            'average_half_life_periods': avg_half_life,
            'cache_age_seconds': {
                k: time.time() - v for k, v in self._last_calc_time.items()
            }
        }


# Example usage
if __name__ == "__main__":
    async def test_half_life():
        calculator = HalfLifeCalculator(window_size=100)
        
        # Simulate mean-reverting spread (OU process)
        print("Simulating mean-reverting spread...")
        
        # Parameters for simulated OU process
        true_theta = 0.05  # Mean reversion speed
        true_mu = 0.0  # Long-term mean
        true_sigma = 0.1  # Volatility
        
        spread = 0.0
        dt = 1.0  # Time step
        
        for i in range(150):
            # OU process: dx = θ(μ - x)dt + σ√dt * Z
            z = (hash(i) % 1000) / 1000.0 - 0.5  # Pseudo-random
            dx = true_theta * (true_mu - spread) * dt + true_sigma * math.sqrt(dt) * z
            spread += dx
            
            calculator.add_spread_observation('BTC', 'ETH', spread)
        
        # Calculate half-life
        print("\nCalculating half-life...")
        result = calculator.calculate_half_life('BTC', 'ETH', period_seconds=60.0)
        
        if result:
            print(f"\nHalf-Life Results:")
            print(f"  Pair: {result.symbol_pair}")
            print(f"  Half-life: {result.half_life_periods:.2f} periods ({result.half_life_seconds:.1f}s)")
            print(f"  Mean reversion speed (θ): {result.mean_reversion_speed:.4f}")
            print(f"  Long-term mean (μ): {result.long_term_mean:.4f}")
            print(f"  Volatility (σ): {result.volatility:.4f}")
            print(f"  R-squared: {result.r_squared:.3f}")
            print(f"  Is mean-reverting: {result.is_mean_reverting}")
            print(f"  Optimal exit (periods): {result.optimal_exit_periods:.1f}")
            
            # Get optimal exit time
            exit_time = calculator.get_optimal_exit_time('BTC', 'ETH')
            if exit_time:
                print(f"  Optimal exit time (95% conf): {exit_time:.1f}s")
        
        print(f"\nMetrics: {calculator.get_metrics()}")
    
    asyncio.run(test_half_life())
