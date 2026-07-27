#!/usr/bin/env python3
"""
Perpetual Swap Hedger for Delta-Neutral Trading

This module uses perpetual swaps to hedge spot exposure dynamically.
It accounts for Binance's specific auto-deleveraging (ADL) queues and
funding rate mechanics to optimize hedge execution.

Key Features:
- Dynamic delta hedging using perpetual futures
- ADL queue position tracking and avoidance
- Funding rate arbitrage integration
- Strict type hinting for memory safety
- C-extension ready for performance-critical paths

Target: Maintain delta neutrality while minimizing funding costs
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable, Any
import threading
import time
import logging

# Configure logging for production use
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


class PositionSide(Enum):
    """Position direction for perpetual contracts."""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class HedgeStrategy(Enum):
    """Available hedging strategies."""
    AGGRESSIVE = auto()
    CONSERVATIVE = auto()
    ADAPTIVE = auto()


@dataclass
class MarketData:
    """Real-time market data snapshot."""
    asset: Asset
    price: Decimal
    bid: Decimal
    ask: Decimal
    volume_24h: Decimal
    funding_rate: Optional[Decimal]
    next_funding_time: Optional[datetime]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid-price from bid/ask."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread_bps(self) -> Decimal:
        """Calculate bid-ask spread in basis points."""
        if self.mid_price == 0:
            return Decimal('0')
        return ((self.ask - self.bid) / self.mid_price) * 10000


@dataclass
class Position:
    """Represents a trading position."""
    asset: Asset
    quantity: Decimal
    entry_price: Decimal
    side: PositionSide
    position_type: str  # 'spot', 'perp', 'futures', 'options'
    unrealized_pnl: Decimal = Decimal('0')
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value in USDT."""
        return abs(self.quantity * self.entry_price)
    
    @property
    def delta_exposure(self) -> Decimal:
        """Calculate delta exposure (simplified for spot/perp)."""
        if self.position_type in ('spot', 'perp'):
            return self.quantity
        return Decimal('0')


@dataclass
class ADLQueuePosition:
    """
    Auto-Deleveraging Queue Position tracker.
    
    Binance uses ADL queues to liquidate positions when insurance funds are depleted.
    This class tracks our position in the queue to avoid adverse selection.
    """
    asset: Asset
    queue_position: int  # 1-5, where 1 is highest priority for ADL
    total_queue_size: int
    estimated_fill_probability: Decimal
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def adl_risk_score(self) -> Decimal:
        """
        Calculate ADL risk score (0-1).
        Higher score means higher risk of being auto-deleveraged.
        """
        if self.total_queue_size == 0:
            return Decimal('0')
        return Decimal(self.queue_position) / Decimal(self.total_queue_size)
    
    def is_safe_to_hedge(self, threshold: Decimal = Decimal('0.5')) -> bool:
        """Check if it's safe to execute hedge given ADL risk."""
        return self.adl_risk_score < threshold


@dataclass
class HedgeExecution:
    """Result of a hedge execution."""
    success: bool
    execution_time_us: int
    delta_before: Decimal
    delta_after: Decimal
    hedge_quantity: Decimal
    hedge_asset: Asset
    hedge_side: PositionSide
    execution_price: Decimal
    slippage_bps: Decimal
    funding_rate_impact: Decimal
    adl_queue_position: Optional[int]
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HedgingStrategyInterface(ABC):
    """Abstract base class for hedging strategies (Strategy Pattern)."""
    
    @abstractmethod
    def calculate_hedge_quantity(
        self,
        portfolio_delta: Decimal,
        current_price: Decimal,
        asset: Asset,
        market_data: Dict[Asset, MarketData]
    ) -> Decimal:
        """Calculate required hedge quantity to neutralize delta."""
        pass
    
    @abstractmethod
    def should_execute_hedge(
        self,
        portfolio_delta: Decimal,
        threshold_bps: Decimal,
        market_conditions: Dict[str, Any]
    ) -> bool:
        """Determine if hedge should be executed based on threshold."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name for logging."""
        pass


class AggressiveHedgingStrategy(HedgingStrategyInterface):
    """Aggressive hedging with tight delta bands."""
    
    def __init__(
        self,
        delta_threshold_bps: Decimal = Decimal('50'),
        min_hedge_size_usdt: Decimal = Decimal('100')
    ):
        self.delta_threshold_bps = delta_threshold_bps
        self.min_hedge_size_usdt = min_hedge_size_usdt
    
    def calculate_hedge_quantity(
        self,
        portfolio_delta: Decimal,
        current_price: Decimal,
        asset: Asset,
        market_data: Dict[Asset, MarketData]
    ) -> Decimal:
        """Calculate inverse position to neutralize delta."""
        if current_price <= 0:
            return Decimal('0')
        
        hedge_qty = -portfolio_delta / current_price
        
        # Round down to exchange precision
        precision = Decimal('0.001') if asset == Asset.BTC else Decimal('0.01')
        return hedge_qty.quantize(precision, rounding=ROUND_DOWN)
    
    def should_execute_hedge(
        self,
        portfolio_delta: Decimal,
        threshold_bps: Decimal,
        market_conditions: Dict[str, Any]
    ) -> bool:
        """Execute on any significant delta breach."""
        effective_threshold = max(threshold_bps, self.delta_threshold_bps)
        return abs(portfolio_delta) > effective_threshold
    
    def get_strategy_name(self) -> str:
        return "AggressiveHedge"


class ConservativeHedgingStrategy(HedgingStrategyInterface):
    """Conservative hedging with wider bands to reduce transaction costs."""
    
    def __init__(
        self,
        delta_threshold_bps: Decimal = Decimal('200'),
        rebalance_interval_seconds: int = 300
    ):
        self.delta_threshold_bps = delta_threshold_bps
        self.rebalance_interval_seconds = rebalance_interval_seconds
        self.last_rebalance: Optional[datetime] = None
    
    def calculate_hedge_quantity(
        self,
        portfolio_delta: Decimal,
        current_price: Decimal,
        asset: Asset,
        market_data: Dict[Asset, MarketData]
    ) -> Decimal:
        """Calculate hedge with size optimization."""
        if current_price <= 0:
            return Decimal('0')
        
        hedge_qty = -portfolio_delta / current_price
        
        # More conservative rounding
        precision = Decimal('0.001') if asset == Asset.BTC else Decimal('0.01')
        return hedge_qty.quantize(precision, rounding=ROUND_DOWN)
    
    def should_execute_hedge(
        self,
        portfolio_delta: Decimal,
        threshold_bps: Decimal,
        market_conditions: Dict[str, Any]
    ) -> bool:
        """Execute only on large delta breaches or time-based rebalance."""
        now = datetime.now(timezone.utc)
        
        # Time-based rebalance
        if self.last_rebalance:
            elapsed = (now - self.last_rebalance).total_seconds()
            if elapsed >= self.rebalance_interval_seconds:
                return True
        
        # Delta-based rebalance
        effective_threshold = max(threshold_bps, self.delta_threshold_bps)
        return abs(portfolio_delta) > effective_threshold
    
    def record_rebalance(self) -> None:
        """Record last rebalance time."""
        self.last_rebalance = datetime.now(timezone.utc)
    
    def get_strategy_name(self) -> str:
        return "ConservativeHedge"


class AdaptiveHedgingStrategy(HedgingStrategyInterface):
    """
    Adaptive hedging that adjusts thresholds based on volatility and funding rates.
    """
    
    def __init__(
        self,
        base_threshold_bps: Decimal = Decimal('100'),
        volatility_multiplier: Decimal = Decimal('1.5'),
        funding_rate_threshold: Decimal = Decimal('0.0001')
    ):
        self.base_threshold_bps = base_threshold_bps
        self.volatility_multiplier = volatility_multiplier
        self.funding_rate_threshold = funding_rate_threshold
        self.current_volatility: Decimal = Decimal('0')
    
    def update_volatility(self, volatility: Decimal) -> None:
        """Update current volatility estimate."""
        self.current_volatility = volatility
    
    def calculate_hedge_quantity(
        self,
        portfolio_delta: Decimal,
        current_price: Decimal,
        asset: Asset,
        market_data: Dict[Asset, MarketData]
    ) -> Decimal:
        """Calculate hedge with volatility-adjusted sizing."""
        if current_price <= 0:
            return Decimal('0')
        
        # Adjust hedge size based on volatility
        vol_factor = Decimal('1') + (self.current_volatility * Decimal('0.1'))
        hedge_qty = (-portfolio_delta / current_price) * vol_factor
        
        precision = Decimal('0.001') if asset == Asset.BTC else Decimal('0.01')
        return hedge_qty.quantize(precision, rounding=ROUND_DOWN)
    
    def should_execute_hedge(
        self,
        portfolio_delta: Decimal,
        threshold_bps: Decimal,
        market_conditions: Dict[str, Any]
    ) -> bool:
        """Adjust threshold based on market conditions."""
        # Increase threshold during high volatility to avoid whipsaws
        adjusted_threshold = self.base_threshold_bps * (
            Decimal('1') + self.current_volatility * self.volatility_multiplier
        )
        
        # Check funding rate impact
        funding_rate = market_conditions.get('funding_rate', Decimal('0'))
        if abs(funding_rate) > self.funding_rate_threshold:
            # Be more aggressive when funding is favorable
            adjusted_threshold *= Decimal('0.7')
        
        effective_threshold = max(threshold_bps, adjusted_threshold)
        return abs(portfolio_delta) > effective_threshold
    
    def get_strategy_name(self) -> str:
        return "AdaptiveHedge"


class DeltaObserver(ABC):
    """Observer interface for delta monitoring (Observer Pattern)."""
    
    @abstractmethod
    def on_delta_change(
        self,
        old_delta: Decimal,
        new_delta: Decimal,
        timestamp: datetime
    ) -> None:
        """Called when portfolio delta changes."""
        pass
    
    @abstractmethod
    def on_hedge_executed(self, execution: HedgeExecution) -> None:
        """Called when a hedge is executed."""
        pass


class PerpHedger:
    """
    Main perpetual swap hedger with ADL awareness and funding optimization.
    
    This class manages delta-neutral hedging using perpetual futures,
    accounting for Binance's ADL queues and funding rate mechanics.
    """
    
    def __init__(
        self,
        strategy: HedgingStrategyInterface,
        delta_threshold_bps: Decimal = Decimal('100'),
        max_execution_time_ms: int = 5
    ):
        self.strategy = strategy
        self.delta_threshold_bps = delta_threshold_bps
        self.max_execution_time_ms = max_execution_time_ms
        
        # Portfolio state
        self._positions: Dict[Asset, List[Position]] = {}
        self._market_data: Dict[Asset, MarketData] = {}
        self._adl_queues: Dict[Asset, ADLQueuePosition] = {}
        
        # Observers
        self._observers: List[DeltaObserver] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics
        self.total_hedges_executed: int = 0
        self.total_slippage_bps: Decimal = Decimal('0')
        self.last_rebalance_time: Optional[datetime] = None
    
    def add_observer(self, observer: DeltaObserver) -> None:
        """Add observer for delta monitoring."""
        self._observers.append(observer)
    
    def remove_observer(self, observer: DeltaObserver) -> None:
        """Remove observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers_delta_change(
        self,
        old_delta: Decimal,
        new_delta: Decimal,
        timestamp: datetime
    ) -> None:
        """Notify all observers of delta change."""
        for observer in self._observers:
            try:
                observer.on_delta_change(old_delta, new_delta, timestamp)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")
    
    def _notify_observers_hedge_executed(self, execution: HedgeExecution) -> None:
        """Notify all observers of hedge execution."""
        for observer in self._observers:
            try:
                observer.on_hedge_executed(execution)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")
    
    def update_market_data(self, market_data: MarketData) -> None:
        """Update market data for an asset (thread-safe)."""
        with self._lock:
            self._market_data[market_data.asset] = market_data
    
    def update_adl_queue(self, adl_position: ADLQueuePosition) -> None:
        """Update ADL queue position for an asset."""
        with self._lock:
            self._adl_queues[adl_position.asset] = adl_position
    
    def add_position(self, position: Position) -> None:
        """Add position to portfolio (thread-safe)."""
        with self._lock:
            if position.asset not in self._positions:
                self._positions[position.asset] = []
            self._positions[position.asset].append(position)
    
    def remove_position(self, asset: Asset, index: int) -> Optional[Position]:
        """Remove position from portfolio."""
        with self._lock:
            if asset in self._positions and index < len(self._positions[asset]):
                return self._positions[asset].pop(index)
        return None
    
    def calculate_portfolio_delta(self) -> Decimal:
        """
        Calculate total portfolio delta exposure.
        
        Returns:
            Total delta in USDT terms
        """
        with self._lock:
            total_delta = Decimal('0')
            
            for asset, positions in self._positions.items():
                if asset not in self._market_data:
                    continue
                
                market = self._market_data[asset]
                
                for position in positions:
                    if position.position_type in ('spot', 'perp'):
                        # Delta = quantity * price for linear products
                        delta = position.quantity * market.price
                        total_delta += delta
                    elif position.position_type == 'futures':
                        # Futures may have different delta characteristics
                        delta = position.quantity * market.price
                        total_delta += delta
                    # Options delta handled separately in greeks module
            
            return total_delta
    
    def execute_rebalance(self) -> HedgeExecution:
        """
        Execute delta-neutral rebalancing using perpetual swaps.
        
        CRITICAL: Must complete within max_execution_time_ms
        
        Returns:
            HedgeExecution result with timing and slippage metrics
        """
        start_time = time.perf_counter()
        
        with self._lock:
            # Calculate current delta
            delta_before = self.calculate_portfolio_delta()
            
            # Get market conditions
            market_conditions = self._get_market_conditions()
            
            # Check if hedge is needed
            if not self.strategy.should_execute_hedge(
                delta_before,
                self.delta_threshold_bps,
                market_conditions
            ):
                return HedgeExecution(
                    success=True,
                    execution_time_us=int((time.perf_counter() - start_time) * 1e6),
                    delta_before=delta_before,
                    delta_after=delta_before,
                    hedge_quantity=Decimal('0'),
                    hedge_asset=Asset.USDT,
                    hedge_side=PositionSide.NONE,
                    execution_price=Decimal('0'),
                    slippage_bps=Decimal('0'),
                    funding_rate_impact=Decimal('0'),
                    adl_queue_position=None
                )
            
            # Determine hedge asset (use most liquid perpetual)
            hedge_asset = self._select_hedge_asset()
            
            if hedge_asset not in self._market_data:
                return HedgeExecution(
                    success=False,
                    execution_time_us=int((time.perf_counter() - start_time) * 1e6),
                    delta_before=delta_before,
                    delta_after=delta_before,
                    hedge_quantity=Decimal('0'),
                    hedge_asset=hedge_asset,
                    hedge_side=PositionSide.NONE,
                    execution_price=Decimal('0'),
                    slippage_bps=Decimal('0'),
                    funding_rate_impact=Decimal('0'),
                    adl_queue_position=None,
                    error_message=f"No market data for hedge asset {hedge_asset}"
                )
            
            market = self._market_data[hedge_asset]
            
            # Check ADL queue safety
            adl_safe = True
            adl_queue_pos = None
            
            if hedge_asset in self._adl_queues:
                adl_info = self._adl_queues[hedge_asset]
                adl_queue_pos = adl_info.queue_position
                adl_safe = adl_info.is_safe_to_hedge()
                
                if not adl_safe:
                    logger.warning(
                        f"ADL risk elevated for {hedge_asset.value}: "
                        f"queue position {adl_queue_pos}"
                    )
            
            # Calculate hedge quantity
            hedge_quantity = self.strategy.calculate_hedge_quantity(
                delta_before,
                market.price,
                hedge_asset,
                self._market_data
            )
            
            # Determine hedge side
            hedge_side = PositionSide.SHORT if delta_before > 0 else PositionSide.LONG
            
            # Execute hedge (simulated - in production, call exchange API)
            execution_result = self._execute_hedge_internal(
                hedge_quantity=hedge_quantity,
                hedge_asset=hedge_asset,
                hedge_side=hedge_side,
                market_data=market,
                delta_before=delta_before,
                adl_safe=adl_safe
            )
            
            # Record metrics
            execution_time_us = int((time.perf_counter() - start_time) * 1e6)
            execution_result.execution_time_us = execution_time_us
            
            if execution_result.success:
                self.total_hedges_executed += 1
                self.total_slippage_bps += execution_result.slippage_bps
                self.last_rebalance_time = datetime.now(timezone.utc)
                
                # Notify observers
                self._notify_observers_delta_change(
                    delta_before,
                    execution_result.delta_after,
                    execution_result.timestamp
                )
                self._notify_observers_hedge_executed(execution_result)
            
            # Verify timing constraint
            if execution_time_us > self.max_execution_time_ms * 1000:
                logger.warning(
                    f"Rebalance exceeded {self.max_execution_time_ms}ms limit "
                    f"(took {execution_time_us / 1000:.2f}ms)"
                )
            
            return execution_result
    
    def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions for strategy decisions."""
        conditions: Dict[str, Any] = {}
        
        # Average funding rate across assets
        funding_rates = [
            m.funding_rate for m in self._market_data.values()
            if m.funding_rate is not None
        ]
        
        if funding_rates:
            avg_funding = sum(funding_rates) / len(funding_rates)
            conditions['funding_rate'] = avg_funding
            conditions['avg_funding_bps'] = float(avg_funding * 10000)
        
        # Volatility estimate (simplified)
        spreads = [m.spread_bps for m in self._market_data.values()]
        if spreads:
            conditions['avg_spread_bps'] = float(sum(spreads) / len(spreads))
        
        return conditions
    
    def _select_hedge_asset(self) -> Asset:
        """Select best asset for hedging based on liquidity and funding."""
        if not self._market_data:
            return Asset.USDT
        
        # Prefer assets with lowest funding cost and highest liquidity
        best_asset = Asset.USDT
        best_score = Decimal('-1e9')
        
        for asset, market in self._market_data.items():
            if asset == Asset.USDT:
                continue
            
            # Score = liquidity - funding_cost * weight
            liquidity_score = market.volume_24h / Decimal('1e9')  # Normalize
            
            funding_cost = market.funding_rate or Decimal('0')
            funding_penalty = abs(funding_cost) * Decimal('1e6')
            
            score = liquidity_score - funding_penalty
            
            if score > best_score:
                best_score = score
                best_asset = asset
        
        return best_asset
    
    def _execute_hedge_internal(
        self,
        hedge_quantity: Decimal,
        hedge_asset: Asset,
        hedge_side: PositionSide,
        market_data: MarketData,
        delta_before: Decimal,
        adl_safe: bool
    ) -> HedgeExecution:
        """
        Internal hedge execution logic.
        
        In production, this would call the exchange API with proper
        error handling, retry logic, and ADL queue management.
        """
        exec_start = time.perf_counter()
        
        # Simulate network latency and execution (microseconds)
        time.sleep(0.0001)  # 100 microseconds
        
        exec_time = int((time.perf_counter() - exec_start) * 1e6)
        
        # Calculate simulated slippage
        # Slippage increases with trade size and decreases with liquidity
        size_impact = abs(hedge_quantity) / Decimal('1000')  # Normalize
        liquidity_factor = Decimal('1e9') / max(market_data.volume_24h, Decimal('1'))
        slippage_bps = (size_impact * liquidity_factor * Decimal('0.01')).quantize(
            Decimal('0.01'), rounding=ROUND_DOWN
        )
        slippage_bps = min(slippage_bps, Decimal('10'))  # Cap at 10 bps
        
        # Calculate funding rate impact
        funding_impact = market_data.funding_rate or Decimal('0')
        if hedge_side == PositionSide.LONG and funding_impact > 0:
            funding_impact = -abs(funding_impact)  # Pay funding
        elif hedge_side == PositionSide.SHORT and funding_impact > 0:
            funding_impact = abs(funding_impact)  # Receive funding
        
        # Simulate delta after hedge
        hedge_value = hedge_quantity * market_data.price
        delta_after = delta_before + hedge_value
        
        # Add hedge position to portfolio
        hedge_position = Position(
            asset=hedge_asset,
            quantity=hedge_quantity if hedge_side == PositionSide.LONG else -hedge_quantity,
            entry_price=market_data.price,
            side=hedge_side,
            position_type='perp',
            unrealized_pnl=Decimal('0')
        )
        
        if hedge_asset not in self._positions:
            self._positions[hedge_asset] = []
        self._positions[hedge_asset].append(hedge_position)
        
        return HedgeExecution(
            success=True,
            execution_time_us=exec_time,
            delta_before=delta_before,
            delta_after=delta_after,
            hedge_quantity=hedge_quantity,
            hedge_asset=hedge_asset,
            hedge_side=hedge_side,
            execution_price=market_data.price,
            slippage_bps=slippage_bps,
            funding_rate_impact=funding_impact,
            adl_queue_position=None if adl_safe else 1
        )
    
    def get_average_slippage_bps(self) -> Decimal:
        """Get average slippage across all executed hedges."""
        if self.total_hedges_executed == 0:
            return Decimal('0')
        return self.total_slippage_bps / Decimal(self.total_hedges_executed)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for monitoring."""
        with self._lock:
            return {
                'total_positions': sum(len(positions) for positions in self._positions.values()),
                'assets': list(self._positions.keys()),
                'delta_exposure': float(self.calculate_portfolio_delta()),
                'total_hedges': self.total_hedges_executed,
                'avg_slippage_bps': float(self.get_average_slippage_bps()),
                'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
            }


def main() -> None:
    """Example usage of the PerpHedger."""
    # Create hedging strategy
    strategy = AggressiveHedgingStrategy(
        delta_threshold_bps=Decimal('50'),
        min_hedge_size_usdt=Decimal('100')
    )
    
    # Initialize hedger
    hedger = PerpHedger(
        strategy=strategy,
        delta_threshold_bps=Decimal('100'),
        max_execution_time_ms=5
    )
    
    # Add market data
    btc_market = MarketData(
        asset=Asset.BTC,
        price=Decimal('50000'),
        bid=Decimal('49999'),
        ask=Decimal('50001'),
        volume_24h=Decimal('1e9'),
        funding_rate=Decimal('0.0001'),
        next_funding_time=datetime.now(timezone.utc)
    )
    hedger.update_market_data(btc_market)
    
    # Add spot position (long BTC)
    spot_position = Position(
        asset=Asset.BTC,
        quantity=Decimal('1.0'),
        entry_price=Decimal('50000'),
        side=PositionSide.LONG,
        position_type='spot'
    )
    hedger.add_position(spot_position)
    
    # Execute rebalance
    result = hedger.execute_rebalance()
    
    print(f"Hedge executed: {result.success}")
    print(f"Delta before: {result.delta_before}")
    print(f"Delta after: {result.delta_after}")
    print(f"Hedge quantity: {result.hedge_quantity}")
    print(f"Slippage: {result.slippage_bps} bps")
    print(f"Execution time: {result.execution_time_us} μs")


if __name__ == "__main__":
    main()
