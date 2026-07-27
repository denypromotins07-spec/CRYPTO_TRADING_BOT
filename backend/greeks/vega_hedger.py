#!/usr/bin/env python3
"""
Vega Hedger - Volatility Risk Neutralization

This module neutralizes volatility risk using options or VIX proxies.
It manages vega exposure across the portfolio and executes hedges when
volatility breaches predefined thresholds.

Key Features:
- Real-time vega calculation and monitoring
- Volatility surface interpolation
- VIX proxy hedging for crypto portfolios
- Strict type hinting for memory safety
- C-extension ready for performance-critical paths

Target: Maintain vega neutrality while optimizing vol trading costs
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable, Any
import threading
import math
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Asset(Enum):
    """Supported trading assets."""
    BTC = "BTC"
    ETH = "ETH"
    SOL = "SOL"
    USDT = "USDT"
    VIX = "VIX"  # Volatility proxy


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "CALL"
    PUT = "PUT"


class VolatilityInstrument(Enum):
    """Available instruments for vega hedging."""
    OPTIONS = auto()
    VARIANCE_SWAP = auto()
    VOL_FUTURE = auto()
    VIX_PROXY = auto()


@dataclass
class VolatilitySurface:
    """
    Represents the volatility surface for an asset.
    
    The surface maps strike and expiry to implied volatility.
    """
    asset: Asset
    strikes: List[Decimal]
    expiries: List[int]  # Days to expiry
    vol_matrix: List[List[Decimal]]  # [expiry][strike] -> IV
    
    def get_implied_vol(
        self,
        strike: Decimal,
        days_to_expiry: int
    ) -> Optional[Decimal]:
        """
        Interpolate implied volatility from the surface.
        
        Uses bilinear interpolation for smooth estimates.
        """
        if not self.strikes or not self.expiries:
            return None
        
        # Find bracketing strikes
        strike_idx = None
        for i, s in enumerate(self.strikes):
            if s >= strike:
                strike_idx = i
                break
        
        if strike_idx is None or strike_idx == 0:
            return None
        
        # Find bracketing expiries
        expiry_idx = None
        for i, e in enumerate(self.expiries):
            if e >= days_to_expiry:
                expiry_idx = i
                break
        
        if expiry_idx is None or expiry_idx == 0:
            return None
        
        # Bilinear interpolation
        try:
            s1, s2 = self.strikes[strike_idx - 1], self.strikes[strike_idx]
            e1, e2 = self.expiries[expiry_idx - 1], self.expiries[expiry_idx]
            
            v11 = self.vol_matrix[expiry_idx - 1][strike_idx - 1]
            v12 = self.vol_matrix[expiry_idx - 1][strike_idx]
            v21 = self.vol_matrix[expiry_idx][strike_idx - 1]
            v22 = self.vol_matrix[expiry_idx][strike_idx]
            
            # Weight factors
            w_strike = float(strike - s1) / float(s2 - s1) if s2 != s1 else 0.5
            w_expiry = float(days_to_expiry - e1) / float(e2 - e1) if e2 != e1 else 0.5
            
            # Interpolate
            v_bottom = v11 * (1 - w_strike) + v12 * w_strike
            v_top = v21 * (1 - w_strike) + v22 * w_strike
            v_interp = v_bottom * (1 - w_expiry) + v_top * w_expiry
            
            return v_interp
        except (IndexError, ZeroDivisionError):
            return None


@dataclass
class VegaExposure:
    """Vega exposure for a position or portfolio."""
    asset: Asset
    total_vega: Decimal
    vega_by_strike: Dict[Decimal, Decimal]
    vega_by_expiry: Dict[int, Decimal]
    net_notional: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def vega_per_1pct_move(self) -> Decimal:
        """P&L impact of 1% volatility move."""
        return self.total_vega * Decimal('0.01')


@dataclass
class VegaHedgeExecution:
    """Result of a vega hedge execution."""
    success: bool
    execution_time_us: int
    vega_before: Decimal
    vega_after: Decimal
    hedge_instrument: VolatilityInstrument
    hedge_quantity: Decimal
    hedge_cost: Decimal
    slippage_bps: Decimal
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VegaHedgingStrategy(ABC):
    """Abstract base class for vega hedging strategies."""
    
    @abstractmethod
    def calculate_hedge_quantity(
        self,
        portfolio_vega: Decimal,
        instrument_vega: Decimal,
        target_vega: Decimal
    ) -> Decimal:
        """Calculate quantity needed to reach target vega."""
        pass
    
    @abstractmethod
    def should_execute_hedge(
        self,
        portfolio_vega: Decimal,
        threshold: Decimal,
        market_conditions: Dict[str, Any]
    ) -> bool:
        """Determine if hedge should be executed."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name for logging."""
        pass


class NeutralVegaStrategy(VegaHedgingStrategy):
    """Strategy targeting zero vega exposure."""
    
    def __init__(self, tolerance: Decimal = Decimal('100')):
        self.tolerance = tolerance
    
    def calculate_hedge_quantity(
        self,
        portfolio_vega: Decimal,
        instrument_vega: Decimal,
        target_vega: Decimal = Decimal('0')
    ) -> Decimal:
        if instrument_vega == 0:
            return Decimal('0')
        
        vega_gap = target_vega - portfolio_vega
        return vega_gap / instrument_vega
    
    def should_execute_hedge(
        self,
        portfolio_vega: Decimal,
        threshold: Decimal,
        market_conditions: Dict[str, Any]
    ) -> bool:
        return abs(portfolio_vega) > max(threshold, self.tolerance)
    
    def get_strategy_name(self) -> str:
        return "NeutralVega"


class DirectionalVolStrategy(VegaHedgingStrategy):
    """
    Strategy that maintains directional vol exposure based on view.
    
    Allows controlled vega exposure when expecting vol changes.
    """
    
    def __init__(
        self,
        target_vega: Decimal,
        vol_view: str = 'neutral'  # 'long', 'short', 'neutral'
    ):
        self.target_vega = target_vega
        self.vol_view = vol_view
    
    def calculate_hedge_quantity(
        self,
        portfolio_vega: Decimal,
        instrument_vega: Decimal,
        target_vega: Optional[Decimal] = None
    ) -> Decimal:
        target = target_vega if target_vega is not None else self.target_vega
        
        if instrument_vega == 0:
            return Decimal('0')
        
        vega_gap = target - portfolio_vega
        return vega_gap / instrument_vega
    
    def should_execute_hedge(
        self,
        portfolio_vega: Decimal,
        threshold: Decimal,
        market_conditions: Dict[str, Any]
    ) -> bool:
        deviation = abs(portfolio_vega - self.target_vega)
        return deviation > threshold
    
    def get_strategy_name(self) -> str:
        return f"DirectionalVol_{self.vol_view}"


class VegaObserver(ABC):
    """Observer interface for vega monitoring."""
    
    @abstractmethod
    def on_vega_change(
        self,
        old_vega: Decimal,
        new_vega: Decimal,
        timestamp: datetime
    ) -> None:
        pass
    
    @abstractmethod
    def on_hedge_executed(self, execution: VegaHedgeExecution) -> None:
        pass


class VegaHedger:
    """
    Main vega hedging engine.
    
    Manages volatility exposure across the portfolio and executes
    hedges using options or VIX proxies.
    """
    
    def __init__(
        self,
        strategy: VegaHedgingStrategy,
        vega_threshold: Decimal = Decimal('500'),
        max_execution_time_ms: int = 10
    ):
        self.strategy = strategy
        self.vega_threshold = vega_threshold
        self.max_execution_time_ms = max_execution_time_ms
        
        # Portfolio state
        self._vega_exposures: Dict[Asset, VegaExposure] = {}
        self._vol_surfaces: Dict[Asset, VolatilitySurface] = {}
        self._available_instruments: List[VolatilityInstrument] = []
        
        # Observers
        self._observers: List[VegaObserver] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics
        self.total_hedges: int = 0
        self.total_hedge_cost: Decimal = Decimal('0')
        self.last_rebalance: Optional[datetime] = None
    
    def add_observer(self, observer: VegaObserver) -> None:
        """Add observer for vega monitoring."""
        self._observers.append(observer)
    
    def _notify_vega_change(
        self,
        old_vega: Decimal,
        new_vega: Decimal,
        timestamp: datetime
    ) -> None:
        for obs in self._observers:
            try:
                obs.on_vega_change(old_vega, new_vega, timestamp)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")
    
    def update_vol_surface(self, surface: VolatilitySurface) -> None:
        """Update volatility surface for an asset."""
        with self._lock:
            self._vol_surfaces[surface.asset] = surface
    
    def update_vega_exposure(self, exposure: VegaExposure) -> None:
        """Update vega exposure for an asset."""
        with self._lock:
            self._vega_exposures[exposure.asset] = exposure
    
    def add_hedging_instrument(self, instrument: VolatilityInstrument) -> None:
        """Add available hedging instrument."""
        if instrument not in self._available_instruments:
            self._available_instruments.append(instrument)
    
    def calculate_total_vega(self) -> Decimal:
        """Calculate total portfolio vega."""
        with self._lock:
            total = Decimal('0')
            for exposure in self._vega_exposures.values():
                total += exposure.total_vega
            return total
    
    def get_vega_by_asset(self, asset: Asset) -> Optional[Decimal]:
        """Get vega exposure for specific asset."""
        with self._lock:
            if asset in self._vega_exposures:
                return self._vega_exposures[asset].total_vega
        return None
    
    def execute_vega_hedge(self) -> VegaHedgeExecution:
        """
        Execute vega hedge to neutralize exposure.
        
        Returns:
            VegaHedgeExecution result with metrics
        """
        import time
        start_time = time.perf_counter()
        
        with self._lock:
            # Calculate current vega
            vega_before = self.calculate_total_vega()
            
            # Get market conditions
            market_conditions = self._get_market_conditions()
            
            # Check if hedge is needed
            if not self.strategy.should_execute_hedge(
                vega_before,
                self.vega_threshold,
                market_conditions
            ):
                return VegaHedgeExecution(
                    success=True,
                    execution_time_us=int((time.perf_counter() - start_time) * 1e6),
                    vega_before=vega_before,
                    vega_after=vega_before,
                    hedge_instrument=VolatilityInstrument.OPTIONS,
                    hedge_quantity=Decimal('0'),
                    hedge_cost=Decimal('0'),
                    slippage_bps=Decimal('0')
                )
            
            # Select best hedging instrument
            instrument = self._select_hedging_instrument(vega_before)
            
            # Estimate instrument vega (simplified)
            instrument_vega = self._estimate_instrument_vega(instrument)
            
            if instrument_vega == 0:
                return VegaHedgeExecution(
                    success=False,
                    execution_time_us=int((time.perf_counter() - start_time) * 1e6),
                    vega_before=vega_before,
                    vega_after=vega_before,
                    hedge_instrument=instrument,
                    hedge_quantity=Decimal('0'),
                    hedge_cost=Decimal('0'),
                    slippage_bps=Decimal('0'),
                    error_message="Could not estimate instrument vega"
                )
            
            # Calculate hedge quantity
            hedge_qty = self.strategy.calculate_hedge_quantity(
                vega_before,
                instrument_vega,
                Decimal('0')  # Target zero vega
            )
            
            # Execute hedge (simulated)
            execution = self._execute_internal(
                hedge_qty=hedge_qty,
                instrument=instrument,
                vega_before=vega_before,
                instrument_vega=instrument_vega
            )
            
            # Update metrics
            exec_time = int((time.perf_counter() - start_time) * 1e6)
            execution.execution_time_us = exec_time
            
            if execution.success:
                self.total_hedges += 1
                self.total_hedge_cost += execution.hedge_cost
                self.last_rebalance = datetime.now(timezone.utc)
                
                # Notify observers
                self._notify_vega_change(
                    vega_before,
                    execution.vega_after,
                    execution.timestamp
                )
            
            return execution
    
    def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions."""
        conditions: Dict[str, Any] = {}
        
        # Average implied vol across surfaces
        avg_iv = Decimal('0')
        count = 0
        for surface in self._vol_surfaces.values():
            for row in surface.vol_matrix:
                for vol in row:
                    avg_iv += vol
                    count += 1
        
        if count > 0:
            conditions['avg_implied_vol'] = float(avg_iv / count)
        
        # Vol regime indicator
        conditions['vol_regime'] = 'high' if avg_iv > Decimal('0.8') else 'normal'
        
        return conditions
    
    def _select_hedging_instrument(self, vega: Decimal) -> VolatilityInstrument:
        """Select best instrument for hedging."""
        if not self._available_instruments:
            return VolatilityInstrument.OPTIONS
        
        # Prefer VIX proxy for large hedges, options for precision
        if abs(vega) > Decimal('1000'):
            if VolatilityInstrument.VIX_PROXY in self._available_instruments:
                return VolatilityInstrument.VIX_PROXY
        
        return VolatilityInstrument.OPTIONS
    
    def _estimate_instrument_vega(self, instrument: VolatilityInstrument) -> Decimal:
        """Estimate vega per unit of hedging instrument."""
        # Simplified estimates - in production, calculate from actual contracts
        estimates = {
            VolatilityInstrument.OPTIONS: Decimal('0.1'),
            VolatilityInstrument.VARIANCE_SWAP: Decimal('1.0'),
            VolatilityInstrument.VOL_FUTURE: Decimal('0.5'),
            VolatilityInstrument.VIX_PROXY: Decimal('0.2'),
        }
        return estimates.get(instrument, Decimal('0.1'))
    
    def _execute_internal(
        self,
        hedge_qty: Decimal,
        instrument: VolatilityInstrument,
        vega_before: Decimal,
        instrument_vega: Decimal
    ) -> VegaHedgeExecution:
        """Internal hedge execution."""
        import time
        exec_start = time.perf_counter()
        
        # Simulate execution latency
        time.sleep(0.0001)
        
        exec_time = int((time.perf_counter() - exec_start) * 1e6)
        
        # Calculate hedge cost (simplified)
        hedge_cost = abs(hedge_qty) * Decimal('0.001')  # 0.1% of notional
        
        # Calculate slippage
        slippage_bps = min(abs(hedge_qty) * Decimal('0.01'), Decimal('5'))
        
        # New vega after hedge
        vega_reduction = hedge_qty * instrument_vega
        vega_after = vega_before + vega_reduction
        
        return VegaHedgeExecution(
            success=True,
            execution_time_us=exec_time,
            vega_before=vega_before,
            vega_after=vega_after,
            hedge_instrument=instrument,
            hedge_quantity=hedge_qty,
            hedge_cost=hedge_cost,
            slippage_bps=slippage_bps
        )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio vega summary."""
        with self._lock:
            return {
                'total_vega': float(self.calculate_total_vega()),
                'assets': list(self._vega_exposures.keys()),
                'total_hedges': self.total_hedges,
                'total_hedge_cost': float(self.total_hedge_cost),
                'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None
            }


def main() -> None:
    """Example usage."""
    from decimal import Decimal
    
    # Create strategy
    strategy = NeutralVegaStrategy(tolerance=Decimal('100'))
    
    # Initialize hedger
    hedger = VegaHedger(
        strategy=strategy,
        vega_threshold=Decimal('500'),
        max_execution_time_ms=10
    )
    
    # Add hedging instrument
    hedger.add_hedging_instrument(VolatilityInstrument.OPTIONS)
    
    # Add vega exposure
    exposure = VegaExposure(
        asset=Asset.BTC,
        total_vega=Decimal('1000'),
        vega_by_strike={},
        vega_by_expiry={},
        net_notional=Decimal('50000')
    )
    hedger.update_vega_exposure(exposure)
    
    # Execute hedge
    result = hedger.execute_vega_hedge()
    
    print(f"Hedge executed: {result.success}")
    print(f"Vega before: {result.vega_before}")
    print(f"Vega after: {result.vega_after}")
    print(f"Hedge quantity: {result.hedge_quantity}")


if __name__ == "__main__":
    main()
