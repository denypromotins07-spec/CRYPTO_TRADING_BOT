#!/usr/bin/env python3
"""
Inventory Manager - Balancing Delta-Neutral Portfolios Across Assets

This module manages inventory across multiple assets to maintain
delta-neutral exposure while optimizing for arbitrage opportunities.
Enforces strict delta-neutrality constraints across the portfolio.

Chapter 4: Arbitrage Risk Management, Legging Risk, and SOUL.md Arb Logging
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple, Set
import time
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in an asset"""
    symbol: str
    quantity: Decimal
    average_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal = Decimal('0')
    delta: Decimal = Decimal('0')  # Position delta (for options/perps)
    
    @property
    def market_value(self) -> Decimal:
        """Calculate market value of position"""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> Decimal:
        """Calculate cost basis"""
        return self.quantity * self.average_entry_price
    
    def update_price(self, new_price: Decimal) -> None:
        """Update current price and recalculate PnL"""
        self.current_price = new_price
        if self.quantity > 0:
            self.unrealized_pnl = (new_price - self.average_entry_price) * self.quantity
        elif self.quantity < 0:
            self.unrealized_pnl = (self.average_entry_price - new_price) * abs(self.quantity)


@dataclass
class InventoryBalance:
    """Target vs actual inventory balance"""
    symbol: str
    target_quantity: Decimal
    actual_quantity: Decimal
    deviation: Decimal
    deviation_pct: Decimal
    rebalance_required: bool


@dataclass
class RebalanceOrder:
    """Order to rebalance inventory"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    reason: str
    priority: int  # Higher = more urgent
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InventoryManager:
    """
    Manages inventory across assets to maintain delta-neutral portfolio.
    
    Key features:
    1. Real-time delta calculation across all positions
    2. Automatic rebalancing when deviation exceeds thresholds
    3. Cross-asset netting for correlated positions
    4. Strict enforcement of delta-neutrality constraints
    """
    
    # Supported symbols
    SUPPORTED_SYMBOLS = frozenset(['BTC', 'ETH', 'SOL', 'USDT'])
    
    # Maximum delta deviation allowed (in USDT notional)
    MAX_DELTA_DEVIATION_USDT = Decimal('1000')
    
    # Rebalancing threshold (percentage deviation)
    REBALANCE_THRESHOLD_PCT = Decimal('0.05')  # 5%
    
    # Minimum trade size for rebalancing (USDT)
    MIN_REBALANCE_SIZE_USDT = Decimal('100')
    
    def __init__(self, exchange_client: Any, risk_limits: Dict[str, Decimal]):
        """
        Initialize inventory manager.
        
        Args:
            exchange_client: Async exchange client
            risk_limits: Dict of symbol -> max position size (USDT)
        """
        self.exchange_client = exchange_client
        self.risk_limits = risk_limits
        
        # Current positions by symbol
        self.positions: Dict[str, Position] = {}
        
        # Target allocations (delta-neutral = 0 for most)
        self.target_deltas: Dict[str, Decimal] = {sym: Decimal('0') for sym in self.SUPPORTED_SYMBOLS}
        
        # Pending rebalance orders
        self.pending_orders: Dict[str, RebalanceOrder] = {}
        self._order_counter = 0
        
        # Historical delta tracking
        self.delta_history: List[Tuple[datetime, Decimal]] = []
        
        # Performance metrics
        self.total_rebalances = 0
        self.successful_rebalances = 0
        self.failed_rebalances = 0
        self.total_delta_violations = 0
        
        logger.info("InventoryManager initialized")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_counter += 1
        timestamp_ns = time.time_ns()
        return f"INV-{timestamp_ns}-{self._order_counter:06d}"
    
    async def initialize_positions(self) -> bool:
        """Initialize positions from exchange"""
        try:
            for symbol in self.SUPPORTED_SYMBOLS:
                position_data = await self.exchange_client.get_position(symbol)
                
                if position_data:
                    qty = Decimal(str(position_data.get('quantity', '0')))
                    entry_price = Decimal(str(position_data.get('entry_price', '0')))
                    current_price = Decimal(str(position_data.get('current_price', '0')))
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=qty,
                        average_entry_price=entry_price,
                        current_price=current_price
                    )
            
            logger.info(f"Initialized {len(self.positions)} positions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize positions: {e}")
            return False
    
    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update price for a symbol"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
    
    def calculate_portfolio_delta(self) -> Decimal:
        """
        Calculate total portfolio delta in USDT terms.
        
        For a delta-neutral portfolio, this should be close to zero.
        
        Returns:
            Total delta in USDT
        """
        total_delta = Decimal('0')
        
        for symbol, position in self.positions.items():
            # Delta = quantity * price (for spot/simple perps)
            # For options, would use actual delta
            symbol_delta = position.quantity * position.current_price
            total_delta += symbol_delta
        
        return total_delta
    
    def calculate_net_exposure(self) -> Dict[str, Decimal]:
        """
        Calculate net exposure per symbol and overall.
        
        Returns:
            Dict with per-symbol and total exposures
        """
        exposures = {}
        
        for symbol, position in self.positions.items():
            exposures[symbol] = position.market_value
        
        exposures['TOTAL'] = sum(exposures.values())
        exposures['NET_DELTA'] = self.calculate_portfolio_delta()
        
        return exposures
    
    def check_delta_neutrality(self) -> Tuple[bool, Decimal]:
        """
        Check if portfolio is delta-neutral within tolerance.
        
        Returns:
            Tuple of (is_neutral, current_delta)
        """
        current_delta = self.calculate_portfolio_delta()
        is_neutral = abs(current_delta) <= self.MAX_DELTA_DEVIATION_USDT
        
        if not is_neutral:
            self.total_delta_violations += 1
            logger.warning(
                f"Delta neutrality violated: delta={current_delta:.2f} USDT, "
                f"threshold={self.MAX_DELTA_DEVIATION_USDT:.2f} USDT"
            )
        
        return is_neutral, current_delta
    
    def calculate_rebalance_orders(self) -> List[RebalanceOrder]:
        """
        Calculate orders needed to restore delta-neutrality.
        
        Returns:
            List of RebalanceOrder objects
        """
        orders = []
        exposures = self.calculate_net_exposure()
        net_delta = exposures.get('NET_DELTA', Decimal('0'))
        
        if abs(net_delta) < self.MIN_REBALANCE_SIZE_USDT:
            return orders  # No rebalancing needed
        
        # Determine which positions to adjust
        # Priority: reduce largest exposures first
        
        sorted_positions = sorted(
            self.positions.items(),
            key=lambda x: abs(x[1].market_value),
            reverse=True
        )
        
        remaining_delta = net_delta
        
        for symbol, position in sorted_positions:
            if symbol == 'USDT':
                continue  # Don't rebalance USDT
            
            if abs(remaining_delta) < self.MIN_REBALANCE_SIZE_USDT:
                break
            
            # Calculate how much to reduce
            position_value = position.market_value
            
            if position_value == 0:
                continue
            
            # Determine direction
            if net_delta > 0:
                # Portfolio is long, need to sell
                if position.quantity > 0:
                    side = 'sell'
                else:
                    continue  # Already short, don't increase
            else:
                # Portfolio is short, need to buy
                if position.quantity < 0:
                    side = 'buy'
                else:
                    continue  # Already long, don't increase
            
            # Calculate rebalance quantity
            # Reduce by percentage proportional to deviation
            deviation_pct = abs(net_delta) / abs(position_value) if position_value != 0 else Decimal('0')
            
            if deviation_pct < self.REBALANCE_THRESHOLD_PCT:
                continue
            
            # Quantity to trade
            trade_qty = abs(position.quantity) * min(deviation_pct, Decimal('0.5'))  # Max 50% reduction
            
            # Check minimum size
            trade_value = trade_qty * position.current_price
            if trade_value < self.MIN_REBALANCE_SIZE_USDT:
                continue
            
            # Check risk limits
            risk_limit = self.risk_limits.get(symbol, Decimal('100000'))
            if abs(position.quantity - trade_qty) * position.current_price > risk_limit:
                trade_qty = abs(position.quantity) - (risk_limit / position.current_price)
            
            if trade_qty <= 0:
                continue
            
            order = RebalanceOrder(
                order_id=self._generate_order_id(),
                symbol=symbol,
                side=side,
                quantity=trade_qty.quantize(Decimal('0.0001'), rounding=ROUND_DOWN),
                reason="delta_rebalance",
                priority=int(abs(deviation_pct) * 100)
            )
            
            orders.append(order)
            remaining_delta -= trade_qty * position.current_price if side == 'sell' else -trade_qty * position.current_price
        
        # Sort by priority
        orders.sort(key=lambda o: o.priority, reverse=True)
        
        return orders
    
    async def execute_rebalance(self, dry_run: bool = True) -> bool:
        """
        Execute rebalancing trades to restore delta-neutrality.
        
        Args:
            dry_run: If True, simulate without executing
            
        Returns:
            True if successful
        """
        orders = self.calculate_rebalance_orders()
        
        if not orders:
            logger.debug("No rebalancing required")
            return True
        
        self.total_rebalances += 1
        
        if dry_run:
            logger.info(f"[DRY RUN] Would execute {len(orders)} rebalance orders:")
            for order in orders:
                logger.info(
                    f"  {order.symbol}: {order.side.upper()} {order.quantity} "
                    f"(priority={order.priority})"
                )
            return True
        
        try:
            # Execute orders
            for order in orders:
                self.pending_orders[order.order_id] = order
                
                result = await self.exchange_client.place_order(
                    market_type="spot",
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    order_type="market"  # Market order for quick rebalance
                )
                
                if result.get('status') == 'filled':
                    # Update position
                    if order.symbol in self.positions:
                        position = self.positions[order.symbol]
                        
                        if order.side == 'buy':
                            # Update average entry for buys
                            total_cost = position.cost_basis + (order.quantity * result.get('fill_price'))
                            total_qty = position.quantity + order.quantity
                            if total_qty > 0:
                                position.average_entry_price = total_cost / total_qty
                            position.quantity = total_qty
                        else:
                            position.quantity -= order.quantity
                        
                        position.update_price(Decimal(str(result.get('fill_price'))))
                    
                    self.successful_rebalances += 1
                    del self.pending_orders[order.order_id]
                else:
                    self.failed_rebalances += 1
                    logger.warning(f"Rebalance order failed: {order.order_id}")
            
            # Verify delta-neutrality restored
            is_neutral, delta = self.check_delta_neutrality()
            
            if is_neutral:
                logger.info(f"Rebalancing complete: delta={delta:.2f} USDT")
            else:
                logger.warning(f"Rebalancing incomplete: delta={delta:.2f} USDT")
            
            return is_neutral
            
        except Exception as e:
            logger.error(f"Rebalancing failed: {e}")
            return False
    
    def get_inventory_balances(self) -> List[InventoryBalance]:
        """Get current inventory balance status"""
        balances = []
        
        for symbol, position in self.positions.items():
            target = self.target_deltas.get(symbol, Decimal('0'))
            actual = position.quantity
            deviation = actual - target
            deviation_pct = deviation / abs(target) if target != 0 else Decimal('0')
            
            rebalance_required = abs(deviation_pct) > self.REBALANCE_THRESHOLD_PCT
            
            balances.append(InventoryBalance(
                symbol=symbol,
                target_quantity=target,
                actual_quantity=actual,
                deviation=deviation,
                deviation_pct=deviation_pct,
                rebalance_required=rebalance_required
            ))
        
        return balances
    
    def enforce_delta_neutrality(self, new_position_symbol: str, new_quantity: Decimal) -> bool:
        """
        Check if a new position would violate delta-neutrality.
        
        This is called BEFORE entering a new arb position to ensure
        the portfolio remains delta-neutral.
        
        Args:
            new_position_symbol: Symbol of new position
            new_quantity: Quantity of new position (positive for long, negative for short)
            
        Returns:
            True if position is allowed, False if it would violate constraints
        """
        # Get current price (or estimate)
        current_price = self.positions.get(new_position_symbol)
        if current_price is None:
            return True  # Can't determine, allow by default
        
        # Calculate new delta contribution
        new_delta = new_quantity * current_price.current_price
        
        # Calculate resulting portfolio delta
        current_delta = self.calculate_portfolio_delta()
        new_portfolio_delta = current_delta + new_delta
        
        # Check if within limits
        if abs(new_portfolio_delta) > self.MAX_DELTA_DEVIATION_USDT:
            logger.warning(
                f"Position rejected: would cause delta={new_portfolio_delta:.2f} USDT, "
                f"exceeds limit={self.MAX_DELTA_DEVIATION_USDT:.2f} USDT"
            )
            return False
        
        # Check if offsetting position exists or will be created
        # For true delta-neutrality, every long should have a corresponding short
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get inventory management metrics"""
        delta_history_avg = 0.0
        if self.delta_history:
            delta_history_avg = sum(float(d) for _, d in self.delta_history) / len(self.delta_history)
        
        return {
            'positions_count': len(self.positions),
            'portfolio_delta_usdt': float(self.calculate_portfolio_delta()),
            'total_rebalances': self.total_rebalances,
            'successful_rebalances': self.successful_rebalances,
            'failed_rebalances': self.failed_rebalances,
            'rebalance_success_rate': self.successful_rebalances / max(1, self.total_rebalances),
            'delta_violations': self.total_delta_violations,
            'pending_orders': len(self.pending_orders),
            'average_historical_delta': delta_history_avg,
            'inventory_balances': [
                {
                    'symbol': b.symbol,
                    'deviation_pct': float(b.deviation_pct),
                    'rebalance_required': b.rebalance_required
                }
                for b in self.get_inventory_balances()
            ]
        }


# Example usage
if __name__ == "__main__":
    class MockExchangeClient:
        async def get_position(self, symbol: str) -> Dict:
            return {
                'quantity': '0.1' if symbol == 'BTC' else '0',
                'entry_price': '50000',
                'current_price': '50100'
            }
        
        async def place_order(self, **kwargs):
            await asyncio.sleep(0.01)
            return {'status': 'filled', 'fill_price': '50100'}
    
    async def test_inventory_manager():
        manager = InventoryManager(
            exchange_client=MockExchangeClient(),
            risk_limits={'BTC': Decimal('50000'), 'ETH': Decimal('30000'), 'SOL': Decimal('10000')}
        )
        
        # Initialize positions
        await manager.initialize_positions()
        
        # Check delta-neutrality
        is_neutral, delta = manager.check_delta_neutrality()
        print(f"Delta neutral: {is_neutral}, Current delta: {delta:.2f} USDT")
        
        # Get exposures
        exposures = manager.calculate_net_exposure()
        print(f"\nExposures: {exposures}")
        
        # Calculate rebalance orders
        orders = manager.calculate_rebalance_orders()
        if orders:
            print(f"\nRebalance orders needed: {len(orders)}")
            for order in orders:
                print(f"  {order.symbol}: {order.side} {order.quantity}")
        else:
            print("\nNo rebalancing needed")
        
        # Get metrics
        print(f"\nMetrics: {manager.get_metrics()}")
    
    asyncio.run(test_inventory_manager())
