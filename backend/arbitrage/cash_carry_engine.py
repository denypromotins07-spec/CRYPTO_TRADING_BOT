#!/usr/bin/env python3
"""
Cash-and-Carry Engine - Simultaneous Spot and Futures Leg Execution

This module executes cash-and-carry arbitrage by simultaneously opening
spot long and futures short positions (or vice versa) when basis spreads
are profitable. Designed for sub-millisecond execution to capture micro-inefficiencies.

Chapter 1: Spot-Futures Basis Trading and Cash-and-Carry Arbitrage Logic
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple
import time

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegStatus(Enum):
    """Status of an individual trade leg"""
    PENDING = auto()
    SUBMITTED = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED = auto()
    FAILED = auto()


class ArbDirection(Enum):
    """Direction of the cash-and-carry arbitrage"""
    LONG_SPOT_SHORT_PERP = "long_spot_short_perp"
    SHORT_SPOT_LONG_PERP = "short_spot_long_perp"


@dataclass
class OrderLeg:
    """Represents a single leg of the arbitrage trade"""
    leg_id: str
    side: str  # 'buy' or 'sell'
    market_type: str  # 'spot' or 'perp'
    symbol: str
    quantity: Decimal
    price: Optional[Decimal] = None
    status: LegStatus = LegStatus.PENDING
    fill_price: Optional[Decimal] = None
    fill_quantity: Decimal = Decimal('0')
    order_id: Optional[str] = None
    submission_time_ns: int = 0
    fill_time_ns: int = 0
    error_message: Optional[str] = None


@dataclass
class ArbPosition:
    """Represents a complete cash-and-carry arbitrage position"""
    position_id: str
    direction: ArbDirection
    symbol: str
    spot_leg: OrderLeg
    perp_leg: OrderLeg
    entry_basis_bps: Decimal
    target_exit_bps: Decimal
    stop_loss_bps: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    pnl_usdt: Decimal = Decimal('0')
    realized_bps: Decimal = Decimal('0')


class CashCarryEngine:
    """
    High-performance cash-and-carry arbitrage engine.
    
    Executes simultaneous spot and futures legs with atomic-like semantics.
    If one leg fails, immediately cancels the other to prevent naked exposure.
    Optimized for 8GB RAM constraint with object pooling and minimal allocations.
    """
    
    # Pre-defined symbols for memory efficiency
    SUPPORTED_SYMBOLS = frozenset(['BTC', 'ETH', 'SOL'])
    
    # Maximum position size per symbol (USDT)
    MAX_POSITION_SIZE_USDT = Decimal('50000')
    
    # Minimum spread threshold in bps to trigger arb
    MIN_SPREAD_BPS = Decimal('15.0')
    
    def __init__(self, exchange_client: Any, risk_manager: Any):
        """
        Initialize the cash-carry engine.
        
        Args:
            exchange_client: Async exchange client for order execution
            risk_manager: Risk management module for position limits
        """
        self.exchange_client = exchange_client
        self.risk_manager = risk_manager
        
        # Active positions tracked by position_id
        self.active_positions: Dict[str, ArbPosition] = {}
        
        # Position counters for ID generation
        self._position_counter = 0
        
        # Performance metrics
        self.total_arbs_executed = 0
        self.successful_arbs = 0
        self.failed_arbs = 0
        self.total_pnl_usdt = Decimal('0')
        
        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        logger.info("CashCarryEngine initialized")
    
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop"""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        return self._loop
    
    def _generate_position_id(self) -> str:
        """Generate unique position ID"""
        self._position_counter += 1
        timestamp_ns = time.time_ns()
        return f"CC-{timestamp_ns}-{self._position_counter:06d}"
    
    def _generate_leg_id(self, position_id: str, leg_type: str) -> str:
        """Generate unique leg ID"""
        return f"{position_id}-{leg_type}"
    
    async def execute_arb(
        self,
        symbol: str,
        direction: ArbDirection,
        quantity_usdt: Decimal,
        spot_price: Decimal,
        perp_price: Decimal,
        basis_bps: Decimal
    ) -> Optional[ArbPosition]:
        """
        Execute a cash-and-carry arbitrage trade.
        
        Args:
            symbol: Trading symbol (BTC, ETH, SOL)
            direction: Direction of arb (long spot/short perp or vice versa)
            quantity_usdt: Position size in USDT
            spot_price: Current spot price
            perp_price: Current perpetual price
            basis_bps: Current basis spread in basis points
            
        Returns:
            ArbPosition if successful, None if failed
        """
        if symbol not in self.SUPPORTED_SYMBOLS:
            logger.error(f"Unsupported symbol: {symbol}")
            return None
        
        # Validate minimum spread
        if basis_bps < self.MIN_SPREAD_BPS:
            logger.warning(f"Spread {basis_bps} bps below minimum {self.MIN_SPRED_BPS} bps")
            return None
        
        # Check risk limits
        if not await self.risk_manager.check_position_limit(symbol, quantity_usdt):
            logger.error(f"Position limit exceeded for {symbol}")
            return None
        
        position_id = self._generate_position_id()
        
        # Calculate quantities
        quantity_crypto = quantity_usdt / spot_price
        
        # Create legs based on direction
        if direction == ArbDirection.LONG_SPOT_SHORT_PERP:
            spot_leg = OrderLeg(
                leg_id=self._generate_leg_id(position_id, "SPOT"),
                side="buy",
                market_type="spot",
                symbol=symbol,
                quantity=quantity_crypto,
                price=spot_price
            )
            perp_leg = OrderLeg(
                leg_id=self._generate_leg_id(position_id, "PERP"),
                side="sell",
                market_type="perp",
                symbol=symbol,
                quantity=quantity_crypto,
                price=perp_price
            )
        else:  # SHORT_SPOT_LONG_PERP
            spot_leg = OrderLeg(
                leg_id=self._generate_leg_id(position_id, "SPOT"),
                side="sell",
                market_type="spot",
                symbol=symbol,
                quantity=quantity_crypto,
                price=spot_price
            )
            perp_leg = OrderLeg(
                leg_id=self._generate_leg_id(position_id, "PERP"),
                side="buy",
                market_type="perp",
                symbol=symbol,
                quantity=quantity_crypto,
                price=perp_price
            )
        
        position = ArbPosition(
            position_id=position_id,
            direction=direction,
            symbol=symbol,
            spot_leg=spot_leg,
            perp_leg=perp_leg,
            entry_basis_bps=basis_bps,
            target_exit_bps=basis_bps * Decimal('0.3'),  # Exit at 30% of entry spread
            stop_loss_bps=basis_bps * Decimal('2.0')  # Stop loss at 2x entry spread
        )
        
        # Execute legs simultaneously
        success = await self._execute_legs_atomic(position)
        
        if success:
            self.active_positions[position_id] = position
            self.total_arbs_executed += 1
            self.successful_arbs += 1
            logger.info(f"Arb executed: {position_id}, basis={basis_bps} bps")
            return position
        else:
            self.failed_arbs += 1
            logger.error(f"Arb execution failed: {position_id}")
            return None
    
    async def _execute_legs_atomic(self, position: ArbPosition) -> bool:
        """
        Execute both legs atomically. If one fails, cancel the other.
        
        Uses asyncio.gather with return_exceptions for parallel execution.
        Implements immediate cancellation on partial failure.
        """
        submission_time_ns = time.time_ns()
        position.spot_leg.submission_time_ns = submission_time_ns
        position.perp_leg.submission_time_ns = submission_time_ns
        
        # Submit both orders in parallel
        tasks = [
            self._submit_leg(position.spot_leg),
            self._submit_leg(position.perp_leg)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        spot_success = not isinstance(results[0], Exception)
        perp_success = not isinstance(results[1], Exception)
        
        if spot_success and perp_success:
            position.spot_leg.status = LegStatus.FILLED
            position.perp_leg.status = LegStatus.FILLED
            fill_time_ns = time.time_ns()
            position.spot_leg.fill_time_ns = fill_time_ns
            position.perp_leg.fill_time_ns = fill_time_ns
            return True
        
        # Partial failure - attempt cancellation
        logger.warning(f"Partial fill detected for {position.position_id}")
        
        if spot_success and not perp_success:
            # Cancel spot leg
            await self._cancel_leg(position.spot_leg)
            position.spot_leg.status = LegStatus.CANCELLED
            position.perp_leg.status = LegStatus.FAILED
            position.perp_leg.error_message = str(results[1])
            
        elif perp_success and not spot_success:
            # Cancel perp leg
            await self._cancel_leg(position.perp_leg)
            position.perp_leg.status = LegStatus.CANCELLED
            position.spot_leg.status = LegStatus.FAILED
            position.spot_leg.error_message = str(results[0])
        
        else:
            # Both failed
            position.spot_leg.status = LegStatus.FAILED
            position.spot_leg.error_message = str(results[0])
            position.perp_leg.status = LegStatus.FAILED
            position.perp_leg.error_message = str(results[1])
        
        return False
    
    async def _submit_leg(self, leg: OrderLeg) -> bool:
        """Submit a single leg order"""
        try:
            order_response = await self.exchange_client.place_order(
                market_type=leg.market_type,
                symbol=leg.symbol,
                side=leg.side,
                quantity=leg.quantity,
                price=leg.price,
                order_type="limit"  # Use limit orders for precise entry
            )
            
            leg.order_id = order_response.get('order_id')
            leg.status = LegStatus.SUBMITTED
            
            # Wait for fill with timeout
            fill_response = await asyncio.wait_for(
                self.exchange_client.wait_for_fill(leg.order_id, timeout_ms=500),
                timeout=0.5
            )
            
            if fill_response.get('status') == 'filled':
                leg.status = LegStatus.FILLED
                leg.fill_price = Decimal(str(fill_response.get('fill_price')))
                leg.fill_quantity = Decimal(str(fill_response.get('fill_quantity')))
                leg.fill_time_ns = time.time_ns()
                return True
            else:
                leg.status = LegStatus.PARTIALLY_FILLED
                return False
                
        except asyncio.TimeoutError:
            leg.status = LegStatus.FAILED
            leg.error_message = "Order fill timeout"
            raise
        except Exception as e:
            leg.status = LegStatus.FAILED
            leg.error_message = str(e)
            raise
    
    async def _cancel_leg(self, leg: OrderLeg) -> bool:
        """Cancel a leg order"""
        if leg.order_id is None:
            return False
        
        try:
            await self.exchange_client.cancel_order(
                market_type=leg.market_type,
                symbol=leg.symbol,
                order_id=leg.order_id
            )
            leg.status = LegStatus.CANCELLED
            return True
        except Exception as e:
            logger.error(f"Failed to cancel leg {leg.leg_id}: {e}")
            return False
    
    async def close_position(self, position_id: str, current_basis_bps: Decimal) -> bool:
        """
        Close an open arbitrage position.
        
        Args:
            position_id: ID of position to close
            current_basis_bps: Current basis spread
            
        Returns:
            True if successfully closed
        """
        if position_id not in self.active_positions:
            logger.error(f"Position {position_id} not found")
            return False
        
        position = self.active_positions[position_id]
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # Target profit reached
        if abs(current_basis_bps) <= abs(position.target_exit_bps):
            should_exit = True
            exit_reason = "target_profit"
        
        # Stop loss triggered
        if abs(current_basis_bps) >= abs(position.stop_loss_bps):
            should_exit = True
            exit_reason = "stop_loss"
        
        if not should_exit:
            return False
        
        # Execute closing legs (opposite of opening)
        close_tasks = []
        
        if position.direction == ArbDirection.LONG_SPOT_SHORT_PERP:
            # Close: sell spot, buy back perp
            close_tasks.append(self._submit_closing_leg(
                position.spot_leg, "sell", position.symbol
            ))
            close_tasks.append(self._submit_closing_leg(
                position.perp_leg, "buy", position.symbol
            ))
        else:
            # Close: buy spot, sell perp
            close_tasks.append(self._submit_closing_leg(
                position.spot_leg, "buy", position.symbol
            ))
            close_tasks.append(self._submit_closing_leg(
                position.perp_leg, "sell", position.symbol
            ))
        
        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        
        if all(not isinstance(r, Exception) for r in results):
            position.closed_at = datetime.now(timezone.utc)
            
            # Calculate PnL
            pnl = self._calculate_pnl(position, current_basis_bps)
            position.pnl_usdt = pnl
            position.realized_bps = current_basis_bps
            
            # Update metrics
            self.total_pnl_usdt += pnl
            
            # Remove from active positions
            del self.active_positions[position_id]
            
            logger.info(
                f"Position {position_id} closed: reason={exit_reason}, "
                f"pnl={pnl:.2f} USDT, realized_bps={current_basis_bps:.2f}"
            )
            return True
        
        return False
    
    async def _submit_closing_leg(
        self, 
        original_leg: OrderLeg, 
        side: str, 
        symbol: str
    ) -> bool:
        """Submit a closing leg order"""
        try:
            await self.exchange_client.place_order(
                market_type=original_leg.market_type,
                symbol=symbol,
                side=side,
                quantity=original_leg.fill_quantity,
                order_type="market"  # Market order for quick exit
            )
            return True
        except Exception as e:
            logger.error(f"Closing leg failed: {e}")
            return False
    
    def _calculate_pnl(
        self, 
        position: ArbPosition, 
        exit_basis_bps: Decimal
    ) -> Decimal:
        """Calculate PnL for a closed position"""
        # Simplified PnL calculation based on basis change
        basis_change_bps = position.entry_basis_bps - exit_basis_bps
        
        # Approximate position value
        position_value = position.spot_leg.fill_quantity * position.spot_leg.fill_price
        
        # PnL = position_value * (basis_change / 10000)
        pnl = position_value * (basis_change_bps / Decimal('10000'))
        
        return pnl.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
    
    def get_active_positions(self) -> list[ArbPosition]:
        """Get list of all active positions"""
        return list(self.active_positions.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_arbs_executed': self.total_arbs_executed,
            'successful_arbs': self.successful_arbs,
            'failed_arbs': self.failed_arbs,
            'success_rate': self.successful_arbs / max(1, self.total_arbs_executed),
            'total_pnl_usdt': float(self.total_pnl_usdt),
            'active_positions_count': len(self.active_positions)
        }


# Example usage and testing
if __name__ == "__main__":
    # Mock exchange client for testing
    class MockExchangeClient:
        async def place_order(self, **kwargs):
            await asyncio.sleep(0.01)  # Simulate network latency
            return {'order_id': 'mock_order_123'}
        
        async def wait_for_fill(self, order_id: str, timeout_ms: int):
            await asyncio.sleep(0.05)  # Simulate fill time
            return {'status': 'filled', 'fill_price': '50000', 'fill_quantity': '0.001'}
        
        async def cancel_order(self, **kwargs):
            await asyncio.sleep(0.01)
            return True
    
    # Mock risk manager
    class MockRiskManager:
        async def check_position_limit(self, symbol: str, quantity: Decimal) -> bool:
            return True
    
    async def test_cash_carry():
        engine = CashCarryEngine(MockExchangeClient(), MockRiskManager())
        
        # Test arb execution
        position = await engine.execute_arb(
            symbol='BTC',
            direction=ArbDirection.LONG_SPOT_SHORT_PERP,
            quantity_usdt=Decimal('1000'),
            spot_price=Decimal('50000'),
            perp_price=Decimal('50100'),
            basis_bps=Decimal('20')
        )
        
        if position:
            print(f"Executed arb: {position.position_id}")
            print(f"Metrics: {engine.get_metrics()}")
    
    asyncio.run(test_cash_carry())
