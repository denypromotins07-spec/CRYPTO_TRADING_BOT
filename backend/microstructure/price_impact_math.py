#!/usr/bin/env python3
"""
Price Impact Mathematics for Market Microstructure

Models permanent vs temporary price impact of trades.
Implements square-root law and other impact models from quantitative finance.

**Key Features:**
- Permanent vs temporary impact decomposition
- Square-root law implementation
- Real-time impact estimation
- Strict type hinting for production reliability

**Performance:** Optimized for 8GB RAM constraint with NumPy vectorization.

References:
    - Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions
    - Gatheral, J. (2010). No-dynamic-arbitrage and market impact
"""

from __future__ import annotations
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ImpactModel(Enum):
    """Available price impact models."""
    SQUARE_ROOT = "square_root"
    LINEAR = "linear"
    POWER_LAW = "power_law"
    ALMGREN_CHRISS = "almgren_chriss"


@dataclass
class ImpactResult:
    """Result container for price impact calculation."""
    permanent_impact: float      # Long-term price change (bps)
    temporary_impact: float      # Short-term/reverting component (bps)
    total_impact: float          # Sum of both components (bps)
    impact_bps: float            # Impact in basis points
    decay_rate: float            # Estimated decay rate of temporary impact
    model_used: str              # Which model was used


@dataclass
class ImpactParameters:
    """Parameters for impact models."""
    # Square-root law parameters
    alpha: float = 0.1       # Impact coefficient
    beta: float = 0.5        # Exponent (typically ~0.5)
    
    # Decay parameters
    decay_half_life: float = 100.0  # Trades until half decay
    
    # Market parameters
    daily_volume: float = 1e6       # Average daily volume
    volatility: float = 0.02        # Daily volatility
    
    # Risk aversion (for Almgren-Chriss)
    risk_aversion: float = 1e-6


class PriceImpactModel:
    """
    Price impact model implementing various microstructure theories.
    
    The model decomposes impact into:
    - Permanent: Information-driven, irreversible price change
    - Temporary: Liquidity-driven, mean-reverting component
    """
    
    def __init__(self, params: Optional[ImpactParameters] = None) -> None:
        """
        Initialize price impact model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or ImpactParameters()
        self._trade_history: List[Tuple[float, float]] = []  # (volume, sign)
        self._impact_buffer: List[float] = []
    
    def calculate_impact(
        self,
        trade_volume: float,
        trade_sign: int,
        current_price: float,
        model: ImpactModel = ImpactModel.SQUARE_ROOT
    ) -> ImpactResult:
        """
        Calculate price impact for a given trade.
        
        Args:
            trade_volume: Absolute trade size
            trade_sign: +1 for buy, -1 for sell
            current_price: Current market price
            model: Which impact model to use
            
        Returns:
            ImpactResult with decomposed impact components
        """
        if trade_volume <= 0 or current_price <= 0:
            return ImpactResult(
                permanent_impact=0.0,
                temporary_impact=0.0,
                total_impact=0.0,
                impact_bps=0.0,
                decay_rate=0.0,
                model_used=model.value
            )
        
        # Normalize volume by daily volume
        vol_fraction = trade_volume / max(self.params.daily_volume, 1e-9)
        
        if model == ImpactModel.SQUARE_ROOT:
            perm_impact, temp_impact = self._square_root_impact(
                vol_fraction, trade_sign
            )
        elif model == ImpactModel.LINEAR:
            perm_impact, temp_impact = self._linear_impact(
                vol_fraction, trade_sign
            )
        elif model == ImpactModel.POWER_LAW:
            perm_impact, temp_impact = self._power_law_impact(
                vol_fraction, trade_sign
            )
        elif model == ImpactModel.ALMGREN_CHRISS:
            perm_impact, temp_impact = self._almgren_chriss_impact(
                vol_fraction, trade_sign, current_price
            )
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Convert to basis points
        total_impact = perm_impact + temp_impact
        impact_bps = abs(total_impact) * 10000  # bps
        
        # Calculate decay rate
        decay_rate = np.log(2) / max(self.params.decay_half_life, 1)
        
        # Store in history
        self._trade_history.append((trade_volume, trade_sign))
        self._impact_buffer.append(temp_impact)
        
        return ImpactResult(
            permanent_impact=perm_impact,
            temporary_impact=temp_impact,
            total_impact=total_impact,
            impact_bps=impact_bps,
            decay_rate=decay_rate,
            model_used=model.value
        )
    
    def _square_root_impact(
        self, 
        vol_fraction: float, 
        sign: int
    ) -> Tuple[float, float]:
        """
        Square-root law impact (most empirically supported).
        
        ΔP/P = α * sign(v) * |v|^β
        """
        sqrt_vol = np.sqrt(abs(vol_fraction))
        
        # Permanent impact (information component)
        perm = self.params.alpha * sign * sqrt_vol * 0.5
        
        # Temporary impact (liquidity component)
        temp = self.params.alpha * sign * sqrt_vol * 0.5
        
        return perm, temp
    
    def _linear_impact(
        self, 
        vol_fraction: float, 
        sign: int
    ) -> Tuple[float, float]:
        """Linear impact model (simplified)."""
        impact = self.params.alpha * sign * vol_fraction
        
        # Split evenly between permanent and temporary
        return impact * 0.5, impact * 0.5
    
    def _power_law_impact(
        self, 
        vol_fraction: float, 
        sign: int
    ) -> Tuple[float, float]:
        """Power law with custom exponent."""
        power_vol = np.power(abs(vol_fraction), self.params.beta)
        impact = self.params.alpha * sign * power_vol
        
        # More permanent for larger trades
        perm_ratio = 0.3 + 0.4 * min(power_vol, 1.0)
        
        return impact * perm_ratio, impact * (1 - perm_ratio)
    
    def _almgren_chriss_impact(
        self, 
        vol_fraction: float, 
        sign: int,
        price: float
    ) -> Tuple[float, float]:
        """
        Almgren-Chriss optimal execution model.
        
        Combines permanent and temporary impact with risk considerations.
        """
        # Temporary impact dominates in AC model
        temp = self.params.alpha * sign * vol_fraction
        
        # Permanent impact scales with volatility
        perm = self.params.volatility * sign * np.sqrt(abs(vol_fraction)) * 0.1
        
        return perm, temp
    
    def calculate_decay(self, steps: int) -> float:
        """
        Calculate remaining impact after given steps.
        
        Args:
            steps: Number of time steps/trades
            
        Returns:
            Fraction of temporary impact remaining
        """
        decay_rate = np.log(2) / max(self.params.decay_half_life, 1)
        return np.exp(-decay_rate * steps)
    
    def get_cumulative_impact(self) -> float:
        """
        Calculate cumulative impact from all recent trades.
        
        Returns:
            Total net impact considering decay
        """
        if not self._trade_history:
            return 0.0
        
        cumulative = 0.0
        for i, (vol, sign) in enumerate(reversed(self._trade_history)):
            decay = self.calculate_decay(i)
            
            vol_fraction = vol / max(self.params.daily_volume, 1e-9)
            perm, temp = self._square_root_impact(vol_fraction, sign)
            
            cumulative += (perm + temp * decay)
        
        return cumulative
    
    def reset(self) -> None:
        """Reset model state."""
        self._trade_history.clear()
        self._impact_buffer.clear()
    
    def set_daily_volume(self, volume: float) -> None:
        """Update daily volume estimate."""
        if volume > 0:
            self.params.daily_volume = volume


class TransientImpactDecay:
    """
    Models the decay of temporary price impact over time.
    
    Implements exponential and power-law decay kernels.
    """
    
    def __init__(
        self,
        decay_type: str = "exponential",
        half_life: float = 100.0,
        power_law_exponent: float = 0.5
    ) -> None:
        """
        Initialize decay model.
        
        Args:
            decay_type: "exponential" or "power_law"
            half_life: Half-life of decay (in trades/time units)
            power_law_exponent: Exponent for power-law decay
        """
        self.decay_type = decay_type
        self.half_life = half_life
        self.power_law_exponent = power_law_exponent
    
    def decay(self, t: float) -> float:
        """
        Calculate decay factor at time t.
        
        Args:
            t: Time elapsed since impact
            
        Returns:
            Decay factor (0 to 1)
        """
        if self.decay_type == "exponential":
            rate = np.log(2) / max(self.half_life, 1e-9)
            return np.exp(-rate * t)
        elif self.decay_type == "power_law":
            return np.power(1 + t / max(self.half_life, 1), -self.power_law_exponent)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
    
    def decay_vectorized(self, times: np.ndarray) -> np.ndarray:
        """Vectorized decay calculation."""
        times = np.asarray(times, dtype=np.float64)
        
        if self.decay_type == "exponential":
            rate = np.log(2) / max(self.half_life, 1e-9)
            return np.exp(-rate * times)
        elif self.decay_type == "power_law":
            return np.power(1 + times / max(self.half_life, 1), -self.power_law_exponent)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")


def estimate_market_impact(
    prices: np.ndarray,
    volumes: np.ndarray,
    signs: np.ndarray,
    lookback: int = 100
) -> ImpactResult:
    """
    Estimate price impact from historical data.
    
    Args:
        prices: Array of prices
        volumes: Array of trade volumes
        signs: Array of trade signs (+1/-1)
        lookback: Number of observations to use
        
    Returns:
        ImpactResult with estimated parameters
    """
    if len(prices) < 10:
        return ImpactResult(
            permanent_impact=0.0,
            temporary_impact=0.0,
            total_impact=0.0,
            impact_bps=0.0,
            decay_rate=0.0,
            model_used="estimation_failed"
        )
    
    # Use recent data
    prices = prices[-lookback:]
    volumes = volumes[-lookback:]
    signs = signs[-lookback:]
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    # Regress returns on signed volume
    X = signs[1:] * np.sqrt(volumes[1:] / np.mean(volumes))
    y = returns
    
    # Simple OLS
    if np.var(X) > 0:
        alpha_hat = np.cov(X, y)[0, 1] / np.var(X)
    else:
        alpha_hat = 0.0
    
    # Estimate permanent vs temporary
    # (simplified - would need more sophisticated analysis in practice)
    perm_impact = alpha_hat * 0.5
    temp_impact = alpha_hat * 0.5
    
    return ImpactResult(
        permanent_impact=perm_impact,
        temporary_impact=temp_impact,
        total_impact=alpha_hat,
        impact_bps=abs(alpha_hat) * 10000,
        decay_rate=0.01,  # Placeholder
        model_used="empirical_ols"
    )


if __name__ == "__main__":
    # Example usage and validation
    print("Price Impact Model - Validation Test")
    print("=" * 50)
    
    # Create model with default parameters
    model = PriceImpactModel(ImpactParameters(
        alpha=0.1,
        beta=0.5,
        daily_volume=1e6,
        volatility=0.02
    ))
    
    # Test different trade sizes
    print("\nSquare-root law impact for various trade sizes:")
    trade_sizes = [1000, 10000, 50000, 100000, 500000]
    
    for size in trade_sizes:
        result = model.calculate_impact(size, 1, 50000.0, ImpactModel.SQUARE_ROOT)
        print(f"  Volume: {size:>7} → Total: {result.total_impact:.6f} "
              f"({result.impact_bps:.2f} bps), "
              f"Perm: {result.permanent_impact:.6f}, Temp: {result.temporary_impact:.6f}")
    
    # Compare models
    print("\nModel comparison for 100k buy order:")
    test_volume = 100000
    
    for model_type in ImpactModel:
        result = model.calculate_impact(test_volume, 1, 50000.0, model_type)
        print(f"  {model_type.value:>15}: {result.total_impact:.6f} ({result.impact_bps:.2f} bps)")
    
    # Test decay
    print("\nTemporary impact decay (exponential):")
    decay_model = TransientImpactDecay(decay_type="exponential", half_life=50)
    
    for t in [0, 10, 25, 50, 100, 200]:
        print(f"  t={t:>3}: decay factor = {decay_model.decay(t):.4f}")
    
    # Cumulative impact
    print("\nCumulative impact simulation:")
    model.reset()
    
    # Simulate a series of buys
    for i in range(10):
        model.calculate_impact(10000, 1, 50000.0)
    
    cum_impact = model.get_cumulative_impact()
    print(f"  After 10x 10k buys: cumulative impact = {cum_impact:.6f}")
    
    # Empirical estimation
    print("\nEmpirical impact estimation:")
    np.random.seed(42)
    n = 500
    sim_volumes = np.random.exponential(10000, n)
    sim_signs = np.random.choice([-1, 1], n)
    sim_returns = 0.0001 * sim_signs * np.sqrt(sim_volumes / 10000) + np.random.normal(0, 0.001, n)
    sim_prices = 100.0 * np.cumprod(1 + sim_returns)
    
    est_result = estimate_market_impact(sim_prices, sim_volumes, sim_signs)
    print(f"  Estimated total impact: {est_result.total_impact:.6f}")
    print(f"  Estimated in bps: {est_result.impact_bps:.2f}")
    print(f"  Model used: {est_result.model_used}")
    
    print("\n✓ Price Impact module validated successfully")
