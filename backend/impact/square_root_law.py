#!/usr/bin/env python3
"""
Square Root Law Market Impact Model

This module implements the square root law of market impact, which models
the non-linear relationship between order size and price impact. The model
dynamically switches between linear and square-root impact based on order
size relative to average daily volume.

Key Features:
- Dynamic model selection (linear vs square-root)
- Asset-specific coefficient calibration (BTC vs SOL)
- Volume-participation aware impact estimation
- Strict type hinting for memory safety
- C-extension compatible structures

Mathematical Foundation:
For small orders (participation < threshold):
    Impact = a * (Q / ADV)  [Linear regime]

For large orders (participation >= threshold):
    Impact = b * σ * sign(Q) * sqrt(|Q| / ADV)  [Square-root regime]

where:
    Q = order size
    ADV = average daily volume
    σ = volatility
    a, b = calibrated coefficients
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum, auto
import math
from threading import Lock


class ImpactRegime(Enum):
    """Market impact regime classification."""
    LINEAR = auto()      # Small orders, proportional impact
    SQUARE_ROOT = auto() # Large orders, square-root scaling
    TRANSITION = auto()  # Between regimes


@dataclass
class AssetCoefficients:
    """Calibrated impact coefficients for a specific asset."""
    # Linear regime coefficient
    alpha: float
    # Square-root regime coefficient  
    beta: float
    # Volatility scaling factor
    vol_scale: float
    # Participation threshold for regime switch
    threshold: float
    # Asset-specific adjustment factor
    adjustment: float
    
    @classmethod
    def default_btc(cls) -> 'AssetCoefficients':
        """Default coefficients for Bitcoin (high liquidity)."""
        return cls(
            alpha=0.1,       # Lower linear impact
            beta=0.8,        # Standard square-root coefficient
            vol_scale=1.0,   # Normal vol scaling
            threshold=0.01,  # 1% ADV threshold
            adjustment=1.0,  # No adjustment
        )
    
    @classmethod
    def default_eth(cls) -> 'AssetCoefficients':
        """Default coefficients for Ethereum."""
        return cls(
            alpha=0.12,
            beta=0.85,
            vol_scale=1.1,   # Slightly higher vol impact
            threshold=0.01,
            adjustment=1.0,
        )
    
    @classmethod
    def default_sol(cls) -> 'AssetCoefficients':
        """Default coefficients for Solana (lower liquidity)."""
        return cls(
            alpha=0.2,       # Higher linear impact
            beta=1.0,        # Higher square-root coefficient
            vol_scale=1.3,   # Higher vol sensitivity
            threshold=0.005, # Lower threshold (switches earlier)
            adjustment=1.2,  # Adjustment for lower cap
        )


@dataclass
class ImpactResult:
    """Result of market impact calculation."""
    # Estimated price impact (fractional)
    impact_fraction: float
    # Impact in basis points
    impact_bps: float
    # Regime used for calculation
    regime: ImpactRegime
    # Participation rate
    participation_rate: float
    # Confidence score (0-1)
    confidence: float
    # Permanent impact component
    permanent_impact: float
    # Temporary impact component
    temporary_impact: float


class SquareRootImpactModel:
    """
    Square root law market impact model with dynamic regime switching.
    
    This model captures the empirical observation that market impact
    scales with the square root of order size for large orders, while
    being approximately linear for small orders.
    """
    
    def __init__(
        self,
        default_volatility: float = 0.02,
        min_participation: float = 1e-6,
        max_participation: float = 0.2,
    ):
        """
        Initialize the impact model.
        
        Args:
            default_volatility: Default annualized volatility
            min_participation: Minimum participation rate
            max_participation: Maximum participation rate (circuit breaker)
        """
        self.default_volatility = default_volatility
        self.min_participation = min_participation
        self.max_participation = max_participation
        
        # Asset-specific coefficients
        self.coefficients: Dict[str, AssetCoefficients] = {
            'BTC': AssetCoefficients.default_btc(),
            'ETH': AssetCoefficients.default_eth(),
            'SOL': AssetCoefficients.default_sol(),
            'USDT': AssetCoefficients(
                alpha=0.01, beta=0.1, vol_scale=0.1,
                threshold=0.05, adjustment=0.5
            ),
        }
        
        # Current volatility estimates (updated from market data)
        self.volatility_estimates: Dict[str, float] = {}
        
        # Thread safety
        self._lock = Lock()
    
    def set_coefficients(self, symbol: str, coeffs: AssetCoefficients) -> None:
        """Set custom coefficients for an asset."""
        with self._lock:
            self.coefficients[symbol] = coeffs
    
    def set_volatility(self, symbol: str, vol: float) -> None:
        """Update volatility estimate for an asset."""
        with self._lock:
            self.volatility_estimates[symbol] = vol
    
    def get_volatility(self, symbol: str) -> float:
        """Get current volatility estimate."""
        return self.volatility_estimates.get(symbol, self.default_volatility)
    
    def calculate_impact(
        self,
        symbol: str,
        order_size: float,
        adv: float,
        volatility: Optional[float] = None,
        side: int = 1,  # 1 for buy, -1 for sell
    ) -> ImpactResult:
        """
        Calculate market impact for an order.
        
        Args:
            symbol: Asset symbol (BTC, ETH, SOL, USDT)
            order_size: Absolute order size
            adv: Average daily volume
            volatility: Optional volatility override
            side: Order side (1=buy, -1=sell)
            
        Returns:
            ImpactResult with impact estimates and regime info
        """
        if adv <= 0:
            return ImpactResult(
                impact_fraction=0.0,
                impact_bps=0.0,
                regime=ImpactRegime.LINEAR,
                participation_rate=0.0,
                confidence=0.0,
                permanent_impact=0.0,
                temporary_impact=0.0,
            )
        
        with self._lock:
            coeffs = self.coefficients.get(
                symbol, AssetCoefficients.default_btc()
            )
        
        vol = volatility if volatility is not None else self.get_volatility(symbol)
        
        # Calculate participation rate
        participation = abs(order_size) / adv
        participation = max(self.min_participation, min(participation, self.max_participation))
        
        # Determine regime
        if participation < coeffs.threshold * 0.5:
            regime = ImpactRegime.LINEAR
        elif participation > coeffs.threshold * 2.0:
            regime = ImpactRegime.SQUARE_ROOT
        else:
            regime = ImpactRegime.TRANSITION
        
        # Calculate impact based on regime
        if regime == ImpactRegime.LINEAR:
            # Linear impact for small orders
            impact = coeffs.alpha * participation
        else:
            # Square root impact for larger orders
            # Impact = β * σ * sqrt(Q / ADV)
            impact = coeffs.beta * vol * math.sqrt(participation)
        
        # Apply asset-specific adjustment
        impact *= coeffs.adjustment
        
        # Ensure correct sign
        impact *= side
        
        # Decompose into permanent and temporary components
        # Permanent ≈ 20-40% of total, Temporary ≈ 60-80%
        permanent_ratio = 0.3
        permanent_impact = impact * permanent_ratio
        temporary_impact = impact * (1.0 - permanent_ratio)
        
        # Calculate confidence based on participation rate
        # Lower confidence for extreme participation rates
        if participation < 0.001:
            confidence = 0.9
        elif participation > 0.1:
            confidence = 0.5
        else:
            confidence = 0.95 - 5.0 * participation
        
        confidence = max(0.3, min(1.0, confidence))
        
        return ImpactResult(
            impact_fraction=abs(impact),
            impact_bps=abs(impact) * 10000,
            regime=regime,
            participation_rate=participation,
            confidence=confidence,
            permanent_impact=abs(permanent_impact),
            temporary_impact=abs(temporary_impact),
        )
    
    def calculate_execution_cost(
        self,
        symbol: str,
        order_size: float,
        adv: float,
        price: float,
        volatility: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate total execution cost including spread and impact.
        
        Args:
            symbol: Asset symbol
            order_size: Order size
            adv: Average daily volume
            price: Current price
            volatility: Optional volatility
            
        Returns:
            Dictionary with cost breakdown
        """
        impact = self.calculate_impact(symbol, order_size, adv, volatility)
        
        # Typical spread cost (half spread for entry)
        spread_bps = 5.0 if symbol == 'BTC' else 10.0
        spread_cost = spread_bps / 10000 * price * abs(order_size)
        
        # Impact cost
        impact_cost = impact.impact_fraction * price * abs(order_size)
        
        # Total cost
        total_cost = spread_cost + impact_cost
        
        return {
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'permanent_cost': impact.permanent_impact * price * abs(order_size),
            'temporary_cost': impact.temporary_impact * price * abs(order_size),
            'total_cost': total_cost,
            'cost_bps': (total_cost / (price * abs(order_size))) * 10000 if price * abs(order_size) > 0 else 0,
        }
    
    def optimal_order_size(
        self,
        symbol: str,
        adv: float,
        max_impact_bps: float = 10.0,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate maximum order size for a given impact tolerance.
        
        Inverts the square root law to find Q such that impact <= max_impact.
        
        Args:
            symbol: Asset symbol
            adv: Average daily volume
            max_impact_bps: Maximum acceptable impact in basis points
            volatility: Optional volatility
            
        Returns:
            Maximum recommended order size
        """
        with self._lock:
            coeffs = self.coefficients.get(
                symbol, AssetCoefficients.default_btc()
            )
        
        vol = volatility if volatility is not None else self.get_volatility(symbol)
        max_impact = max_impact_bps / 10000
        
        # Solve for participation in square root regime
        # max_impact = β * σ * sqrt(participation)
        # participation = (max_impact / (β * σ))²
        
        participation = (max_impact / (coeffs.beta * vol)) ** 2
        participation = min(participation, self.max_participation)
        
        return participation * adv


class ImpactCalibrator:
    """
    Calibrate impact model coefficients from historical execution data.
    
    Uses regression to fit the square root law model to observed impacts.
    """
    
    def __init__(self):
        """Initialize the calibrator."""
        self.executions: List[Dict] = []
        self._lock = Lock()
    
    def add_execution(
        self,
        symbol: str,
        order_size: float,
        adv: float,
        realized_impact: float,
        volatility: float,
    ) -> None:
        """Record an execution for calibration."""
        with self._lock:
            self.executions.append({
                'symbol': symbol,
                'order_size': abs(order_size),
                'adv': adv,
                'participation': abs(order_size) / adv if adv > 0 else 0,
                'realized_impact': abs(realized_impact),
                'volatility': volatility,
            })
    
    def calibrate_beta(self, symbol: str) -> Optional[float]:
        """
        Calibrate beta coefficient for a symbol using OLS.
        
        Fits: impact = β * σ * sqrt(participation)
        
        Returns:
            Calibrated beta or None if insufficient data
        """
        with self._lock:
            relevant = [e for e in self.executions if e['symbol'] == symbol]
        
        if len(relevant) < 10:
            return None
        
        # Simple OLS: minimize Σ(impact - β * σ * sqrt(part))²
        numerator = 0.0
        denominator = 0.0
        
        for e in relevant:
            if e['participation'] > 0 and e['volatility'] > 0:
                x = e['volatility'] * math.sqrt(e['participation'])
                y = e['realized_impact']
                numerator += x * y
                denominator += x * x
        
        if denominator > 0:
            return numerator / denominator
        return None
    
    def calibrate_threshold(
        self, 
        symbol: str,
        default_threshold: float = 0.01,
    ) -> float:
        """
        Estimate regime transition threshold from data.
        
        Finds the participation rate where impact transitions from
        linear to square-root scaling.
        """
        with self._lock:
            relevant = sorted(
                [e for e in self.executions if e['symbol'] == symbol],
                key=lambda x: x['participation']
            )
        
        if len(relevant) < 20:
            return default_threshold
        
        # Find point where R² improves most by switching models
        best_threshold = default_threshold
        best_improvement = 0.0
        
        for i in range(5, len(relevant) - 5):
            threshold = relevant[i]['participation']
            
            # Compare model fits before and after threshold
            before = relevant[:i]
            after = relevant[i:]
            
            if not before or not after:
                continue
            
            # Simple heuristic: variance ratio
            var_before = self._variance([e['realized_impact'] for e in before])
            var_after = self._variance([e['realized_impact'] for e in after])
            
            improvement = var_before - var_after
            if improvement > best_improvement:
                best_improvement = improvement
                best_threshold = threshold
        
        return best_threshold
    
    def _variance(self, values: List[float]) -> float:
        """Calculate variance of a list."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


if __name__ == "__main__":
    # Example usage
    model = SquareRootImpactModel()
    
    # Calculate impact for BTC order
    result = model.calculate_impact(
        symbol='BTC',
        order_size=10.0,  # 10 BTC
        adv=50000.0,      # 50k BTC daily volume
        volatility=0.02,
        side=1,
    )
    
    print(f"Impact: {result.impact_bps:.2f} bps")
    print(f"Regime: {result.regime.name}")
    print(f"Participation: {result.participation_rate*100:.3f}%")
    print(f"Permanent: {result.permanent_impact*10000:.2f} bps")
    print(f"Temporary: {result.temporary_impact*10000:.2f} bps")
    
    # Calculate for SOL (higher impact expected)
    sol_result = model.calculate_impact(
        symbol='SOL',
        order_size=1000.0,
        adv=500000.0,
        volatility=0.04,
        side=-1,
    )
    
    print(f"\nSOL Impact: {sol_result.impact_bps:.2f} bps")
    print(f"SOL Regime: {sol_result.regime.name}")
