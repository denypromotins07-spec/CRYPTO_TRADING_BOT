#!/usr/bin/env python3
"""
Square Root Law Market Impact Model

This module implements the square-root law for modeling non-linear market impact
of aggressive taker orders. The model captures the empirical observation that
market impact scales with the square root of traded volume relative to average
daily volume.

Key Features:
- Square-root impact: impact ~ sigma * (Q/ADV)^0.5
- Asset-specific calibration for BTC, ETH, SOL
- Dynamic coefficient adjustment based on volatility regime
- Memory-efficient implementation for 8GB constraint

References:
- Bouchaud, J.-P., et al. "Equivariance of the price impact"
- Almgren, R., et al. "Direct estimation of equity market impact"

Author: ZAID Personal Crypto Trading Bot
Stage: 22 - LOB Physics & Hawkes Processes
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from threading import Lock


class ImpactRegime(Enum):
    """Market impact regime classification."""
    LOW_VOLATILITY = "low_vol"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_vol"
    EXTREME = "extreme"


@dataclass
class ImpactResult:
    """Result of market impact calculation."""
    # Input parameters
    order_size: float
    adv: float  # Average daily volume
    volatility: float
    
    # Impact metrics
    temporary_impact: float  # In basis points
    permanent_impact: float  # In basis points
    total_impact_bps: float  # Total in bps
    impact_coefficient: float
    
    # Execution metrics
    expected_slippage: float  # In price units
    execution_price_estimate: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    
    # Metadata
    regime: ImpactRegime
    timestamp: float


@dataclass
class AssetImpactParams:
    """Asset-specific impact parameters."""
    base_coefficient: float  # Base sqrt impact coefficient
    vol_adjustment: float  # Volatility sensitivity
    min_adv_threshold: float  # Minimum ADV for valid calculation
    max_order_fraction: float  # Maximum order as fraction of ADV
    
    @classmethod
    def default_params(cls, asset: str) -> 'AssetImpactParams':
        """Get default parameters for common assets."""
        params = {
            "BTC": cls(
                base_coefficient=0.10,  # 10% of daily vol
                vol_adjustment=0.5,
                min_adv_threshold=1e9,  # $1B
                max_order_fraction=0.05,  # 5% of ADV
            ),
            "ETH": cls(
                base_coefficient=0.12,
                vol_adjustment=0.6,
                min_adv_threshold=5e8,
                max_order_fraction=0.05,
            ),
            "SOL": cls(
                base_coefficient=0.15,
                vol_adjustment=0.7,
                min_adv_threshold=1e8,
                max_order_fraction=0.03,  # Lower for higher vol asset
            ),
            "DEFAULT": cls(
                base_coefficient=0.10,
                vol_adjustment=0.5,
                min_adv_threshold=1e7,
                max_order_fraction=0.05,
            ),
        }
        return params.get(asset.upper(), params["DEFAULT"])


class SquareRootImpactModel:
    """
    Square-root law market impact model.
    
    The core formula is:
        impact = coefficient * sigma * (Q / ADV)^gamma
    
    Where:
        - coefficient: asset-specific scaling factor
        - sigma: daily volatility
        - Q: order size
        - ADV: average daily volume
        - gamma: typically 0.5 (square root)
    """
    
    def __init__(
        self,
        asset: str = "BTC",
        exponent: float = 0.5,
        use_adaptive_coefficient: bool = True,
    ):
        """
        Initialize the square root impact model.
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL)
            exponent: Power law exponent (default 0.5 for square root)
            use_adaptive_coefficient: Adjust coefficient based on volatility
        """
        self.asset = asset.upper()
        self.exponent = exponent
        self.use_adaptive_coefficient = use_adaptive_coefficient
        
        # Load asset-specific parameters
        self.params = AssetImpactParams.default_params(self.asset)
        
        # Volatility tracking for adaptive coefficient
        self._recent_volatilities: List[float] = []
        self._vol_window_size: int = 100
        
        # Thread safety
        self._lock = Lock()
        
        # Pre-compute constants
        self._sqrt_2pi = np.sqrt(2 * np.pi)
    
    def _get_current_volatility(self) -> float:
        """Get current volatility estimate from recent observations."""
        if not self._recent_volatilities:
            # Default volatility based on asset
            defaults = {"BTC": 0.04, "ETH": 0.06, "SOL": 0.10}
            return defaults.get(self.asset, 0.05)
        
        return np.mean(self._recent_volatilities[-self._vol_window_size:])
    
    def _update_volatility(self, new_vol: float) -> None:
        """Update volatility estimate with new observation."""
        with self._lock:
            self._recent_volatilities.append(new_vol)
            if len(self._recent_volatilities) > self._vol_window_size:
                self._recent_volatilities.pop(0)
    
    def _determine_regime(self, volatility: float) -> ImpactRegime:
        """Classify volatility regime."""
        thresholds = {
            "BTC": [0.02, 0.04, 0.08],
            "ETH": [0.03, 0.06, 0.12],
            "SOL": [0.05, 0.10, 0.20],
            "DEFAULT": [0.025, 0.05, 0.10],
        }
        thresh = thresholds.get(self.asset, thresholds["DEFAULT"])
        
        if volatility < thresh[0]:
            return ImpactRegime.LOW_VOLATILITY
        elif volatility < thresh[1]:
            return ImpactRegime.NORMAL
        elif volatility < thresh[2]:
            return ImpactRegime.HIGH_VOLATILITY
        else:
            return ImpactRegime.EXTREME
    
    def _calculate_coefficient(self, volatility: float, regime: ImpactRegime) -> float:
        """Calculate adaptive impact coefficient."""
        base_coef = self.params.base_coefficient
        
        if not self.use_adaptive_coefficient:
            return base_coef
        
        # Adjust for volatility regime
        vol_multiplier = 1.0 + self.params.vol_adjustment * (
            volatility / 0.05 - 1.0
        )
        
        # Additional adjustment for extreme regimes
        regime_multipliers = {
            ImpactRegime.LOW_VOLATILITY: 0.8,
            ImpactRegime.NORMAL: 1.0,
            ImpactRegime.HIGH_VOLATILITY: 1.3,
            ImpactRegime.EXTREME: 1.8,
        }
        
        return base_coef * vol_multiplier * regime_multipliers.get(regime, 1.0)
    
    def calculate_impact(
        self,
        order_size: float,
        adv: float,
        current_price: float,
        volatility: Optional[float] = None,
        side: str = "buy",
    ) -> ImpactResult:
        """
        Calculate market impact using square-root law.
        
        Args:
            order_size: Absolute size of order in base currency
            adv: Average daily volume in quote currency
            current_price: Current market price
            volatility: Daily volatility (optional, will estimate if not provided)
            side: Order side ("buy" or "sell")
            
        Returns:
            ImpactResult with detailed impact metrics
        """
        if volatility is None:
            volatility = self._get_current_volatility()
        
        # Update volatility tracking
        self._update_volatility(volatility)
        
        # Determine regime
        regime = self._determine_regime(volatility)
        
        # Validate inputs
        if adv <= 0:
            raise ValueError("ADV must be positive")
        if order_size <= 0:
            raise ValueError("Order size must be positive")
        
        # Check order size relative to ADV
        order_fraction = order_size / adv
        if order_fraction > self.params.max_order_fraction:
            # Warning: large order may have unpredictable impact
            pass
        
        # Calculate coefficient
        coefficient = self._calculate_coefficient(volatility, regime)
        
        # Core square-root formula
        # impact_bps = coefficient * (Q / ADV)^exponent * 10000
        raw_impact = coefficient * (order_fraction ** self.exponent)
        temporary_impact_bps = raw_impact * 10000  # Convert to basis points
        
        # Permanent impact is typically ~20-40% of temporary
        permanent_impact_bps = temporary_impact_bps * 0.3
        
        # Total impact
        total_impact_bps = temporary_impact_bps + permanent_impact_bps
        
        # Convert to price units
        sign = 1.0 if side.lower() == "buy" else -1.0
        expected_slippage = sign * (total_impact_bps / 10000) * current_price
        
        # Execution price estimate
        execution_price = current_price + expected_slippage
        
        # Confidence interval (assuming normal distribution of impact)
        # Standard error scales with volatility and order size
        std_error = volatility * np.sqrt(order_fraction) * current_price * 0.5
        confidence_lower = execution_price - 1.96 * std_error
        confidence_upper = execution_price + 1.96 * std_error
        
        return ImpactResult(
            order_size=order_size,
            adv=adv,
            volatility=volatility,
            temporary_impact=temporary_impact_bps,
            permanent_impact=permanent_impact_bps,
            total_impact_bps=total_impact_bps,
            impact_coefficient=coefficient,
            expected_slippage=expected_slippage,
            execution_price_estimate=execution_price,
            confidence_interval_lower=confidence_lower,
            confidence_interval_upper=confidence_upper,
            regime=regime,
            timestamp=time.time(),
        )
    
    def calculate_impact_vectorized(
        self,
        order_sizes: np.ndarray,
        adv: float,
        current_price: float,
        volatility: Optional[float] = None,
    ) -> np.ndarray:
        """
        Vectorized impact calculation for multiple order sizes.
        
        Args:
            order_sizes: Array of order sizes
            adv: Average daily volume
            current_price: Current market price
            volatility: Daily volatility
            
        Returns:
            Array of impact values in basis points
        """
        if volatility is None:
            volatility = self._get_current_volatility()
        
        coefficient = self._calculate_coefficient(
            volatility, self._determine_regime(volatility)
        )
        
        order_fractions = order_sizes / adv
        impacts = coefficient * np.power(order_fractions, self.exponent)
        
        return impacts * 10000  # Return in bps
    
    def get_marginal_impact(
        self,
        current_filled: float,
        additional_size: float,
        adv: float,
        current_price: float,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate marginal impact of additional order on top of already filled amount.
        
        This is useful for determining the cost of increasing an order size.
        
        Args:
            current_filled: Amount already executed
            additional_size: Additional amount to execute
            adv: Average daily volume
            current_price: Current market price
            volatility: Daily volatility
            
        Returns:
            Marginal impact in basis points
        """
        if volatility is None:
            volatility = self._get_current_volatility()
        
        coefficient = self._calculate_coefficient(
            volatility, self._determine_regime(volatility)
        )
        
        # Impact of total order
        total_fraction = (current_filled + additional_size) / adv
        total_impact = coefficient * (total_fraction ** self.exponent)
        
        # Impact of current filled
        current_fraction = current_filled / adv
        current_impact = coefficient * (current_fraction ** self.exponent) if current_filled > 0 else 0
        
        # Marginal impact (average over additional size)
        marginal_impact = (total_impact * (current_filled + additional_size) 
                          - current_impact * current_filled) / additional_size
        
        return marginal_impact * 10000  # In bps
    
    def update_asset(self, asset: str) -> None:
        """Update asset class for recalibration."""
        with self._lock:
            self.asset = asset.upper()
            self.params = AssetImpactParams.default_params(self.asset)
    
    def reset_volatility_history(self) -> None:
        """Clear volatility history."""
        with self._lock:
            self._recent_volatilities.clear()


class MultiAssetImpactAnalyzer:
    """
    Analyzes market impact across multiple assets simultaneously.
    
    Optimized for BTC, SOL, ETH portfolio execution.
    """
    
    def __init__(self, assets: List[str] = None):
        """
        Initialize multi-asset analyzer.
        
        Args:
            assets: List of asset symbols to analyze
        """
        if assets is None:
            assets = ["BTC", "ETH", "SOL"]
        
        self._models: Dict[str, SquareRootImpactModel] = {}
        for asset in assets:
            self._models[asset] = SquareRootImpactModel(asset=asset)
        
        self._lock = Lock()
    
    def add_asset(self, asset: str) -> None:
        """Add a new asset to the analyzer."""
        with self._lock:
            if asset.upper() not in self._models:
                self._models[asset.upper()] = SquareRootImpactModel(asset=asset)
    
    def calculate_portfolio_impact(
        self,
        orders: Dict[str, float],
        adv_map: Dict[str, float],
        price_map: Dict[str, float],
        volatility_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, ImpactResult]:
        """
        Calculate impact for a portfolio of orders.
        
        Args:
            orders: Dict mapping asset to order size
            adv_map: Dict mapping asset to ADV
            price_map: Dict mapping asset to current price
            volatility_map: Optional dict mapping asset to volatility
            
        Returns:
            Dict mapping asset to ImpactResult
        """
        results = {}
        
        with self._lock:
            for asset, size in orders.items():
                if asset not in self._models:
                    self.add_asset(asset)
                
                model = self._models[asset]
                adv = adv_map.get(asset, 1e9)
                price = price_map.get(asset, 50000.0)
                vol = volatility_map.get(asset) if volatility_map else None
                
                result = model.calculate_impact(
                    order_size=size,
                    adv=adv,
                    current_price=price,
                    volatility=vol,
                )
                results[asset] = result
        
        return results
    
    def get_aggregate_impact(
        self,
        results: Dict[str, ImpactResult],
    ) -> float:
        """
        Calculate aggregate impact across portfolio in USD terms.
        
        Args:
            results: Dict of ImpactResults per asset
            
        Returns:
            Total expected slippage in USD
        """
        total_slippage = 0.0
        for asset, result in results.items():
            total_slippage += abs(result.expected_slippage * result.order_size)
        return total_slippage


if __name__ == "__main__":
    # Demo usage
    model = SquareRootImpactModel(asset="BTC")
    
    # Example: Calculate impact for 10 BTC order
    result = model.calculate_impact(
        order_size=10.0,  # 10 BTC
        adv=50000.0,  # 50k BTC daily volume
        current_price=50000.0,
        volatility=0.04,
        side="buy",
    )
    
    print(f"Order Size: {result.order_size} BTC")
    print(f"Temporary Impact: {result.temporary_impact:.2f} bps")
    print(f"Permanent Impact: {result.permanent_impact:.2f} bps")
    print(f"Total Impact: {result.total_impact_bps:.2f} bps")
    print(f"Expected Slippage: ${result.expected_slippage:.2f}")
    print(f"Execution Price: ${result.execution_price_estimate:.2f}")
    print(f"Regime: {result.regime.value}")
