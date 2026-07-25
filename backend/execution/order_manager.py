"""
Order Manager for ZAID Personal Crypto Trading Bot
===================================================
Chapter 3: Smart Order Routing, Slippage Modeling, and Order Lifecycle Management

Strict state machine for order lifecycle management.
Ensures orders transition through valid states only with full audit trail.

Features:
- Finite state machine for order lifecycle
- Real-time order status tracking
- Fill aggregation and averaging
- Partial fill handling
- SOUL.md integration for failed orders
- Support for BTC, SOL, ETH, USDT pairs

Author: Opus 4.8
Stage: 2 of 100
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Valid order states in the lifecycle."""
    CREATED = "created"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Valid state transitions
VALID_TRANSITIONS: Dict[OrderState, List[OrderState]] = {
    OrderState.CREATED: [OrderState.SUBMITTED, OrderState.CANCELLED],
    OrderState.SUBMITTED: [OrderState.ACKNOWLEDGED, OrderState.REJECTED, OrderState.CANCELLED],
    OrderState.ACKNOWLEDGED: [OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED],
    OrderState.PARTIALLY_FILLED: [OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED],
    OrderState.FILLED: [],  # Terminal state
    OrderState.CANCELLED: [],  # Terminal state
    OrderState.REJECTED: [],  # Terminal state
    OrderState.EXPIRED: [],  # Terminal state
}


@dataclass
class Fill:
    """Represents a single fill event."""
    fill_id: str
    order_id: str
    price: Decimal
    quantity: Decimal
    commission: Decimal
    commission_asset: str
    timestamp: int
    trade_id: Optional[int] = None
    
    @property
    def notional_value(self) -> Decimal:
        return self.price * self.quantity


@dataclass
class Order:
    """Represents a trading order with full lifecycle tracking."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    price: Optional[Decimal]
    time_in_force: str = "GTC"
    state: OrderState = OrderState.CREATED
    filled_quantity: Decimal = Decimal('0')
    avg_fill_price: Optional[Decimal] = None
    fills: List[Fill] = field(default_factory=list)
    created_at: int = 0
    submitted_at: Optional[int] = None
    acknowledged_at: Optional[int] = None
    updated_at: int = 0
    exchange_order_id: Optional[str] = None
    reject_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        return self.state not in [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED]
    
    @property
    def fill_pct(self) -> float:
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)
    
    @property
    def total_commission(self) -> Decimal:
        return sum(f.commission for f in self.fills)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price else None,
            'time_in_force': self.time_in_force,
            'state': self.state.value,
            'filled_quantity': str(self.filled_quantity),
            'avg_fill_price': str(self.avg_fill_price) if self.avg_fill_price else None,
            'fill_pct': self.fill_pct,
            'remaining_quantity': str(self.remaining_quantity),
            'created_at': self.created_at,
            'exchange_order_id': self.exchange_order_id,
            'reject_reason': self.reject_reason,
        }


class OrderStateMachine:
    """
    Finite state machine for order lifecycle management.
    
    Enforces valid state transitions and maintains audit trail.
    Thread-safe for concurrent order updates.
    """
    
    def __init__(self, soul_logger_callback: Optional[Callable] = None):
        """
        Initialize order state machine.
        
        Args:
            soul_logger_callback: Function to log failed orders to SOUL.md
        """
        self._soul_logger = soul_logger_callback
        self._lock = asyncio.Lock()
    
    async def transition(self, order: Order, new_state: OrderState, 
                         reason: Optional[str] = None) -> bool:
        """
        Transition order to new state.
        
        Args:
            order: Order to transition
            new_state: Target state
            reason: Reason for transition (for rejections/cancellations)
        
        Returns:
            True if transition successful, False otherwise
        """
        async with self._lock:
            current_state = order.state
            
            # Check if transition is valid
            if new_state not in VALID_TRANSITIONS.get(current_state, []):
                logger.warning(f"Invalid state transition: {current_state.value} -> {new_state.value}")
                return False
            
            # Execute transition
            order.state = new_state
            order.updated_at = int(time.time() * 1000)
            
            # Set timestamps for specific states
            if new_state == OrderState.SUBMITTED:
                order.submitted_at = order.updated_at
            elif new_state == OrderState.ACKNOWLEDGED:
                order.acknowledged_at = order.updated_at
            elif new_state == OrderState.REJECTED:
                order.reject_reason = reason
                await self._log_to_soul(order, reason or "Order rejected")
            elif new_state == OrderState.CANCELLED:
                if reason:
                    order.reject_reason = reason
                    await self._log_to_soul(order, f"Order cancelled: {reason}")
            
            logger.debug(f"Order {order.order_id} transitioned: {current_state.value} -> {new_state.value}")
            return True
    
    async def add_fill(self, order: Order, fill: Fill) -> bool:
        """
        Add a fill to an order.
        
        Args:
            order: Order to add fill to
            fill: Fill event to add
        
        Returns:
            True if fill added successfully
        """
        async with self._lock:
            if not order.is_active:
                logger.warning(f"Cannot add fill to inactive order {order.order_id}")
                return False
            
            # Add fill
            order.fills.append(fill)
            
            # Update filled quantity
            old_filled = order.filled_quantity
            order.filled_quantity += fill.quantity
            
            # Update average fill price
            total_notional = sum(f.notional_value for f in order.fills)
            if order.filled_quantity > 0:
                order.avg_fill_price = total_notional / order.filled_quantity
            
            # Update state based on fill
            if order.filled_quantity >= order.quantity:
                await self.transition(order, OrderState.FILLED)
            elif order.filled_quantity > old_filled:
                if order.state != OrderState.PARTIALLY_FILLED:
                    await self.transition(order, OrderState.PARTIALLY_FILLED)
            
            order.updated_at = int(time.time() * 1000)
            logger.info(f"Fill added to {order.order_id}: {fill.quantity}@{fill.price}")
            return True
    
    async def _log_to_soul(self, order: Order, reason: str):
        """Log failed/cancelled order to SOUL.md for learning."""
        if self._soul_logger:
            try:
                lesson = {
                    'type': 'order_failure',
                    'timestamp': int(time.time() * 1000),
                    'order': order.to_dict(),
                    'reason': reason,
                    'learning': self._extract_learning(order, reason)
                }
                await self._soul_logger(lesson)
            except Exception as e:
                logger.error(f"Failed to log to SOUL: {e}")
    
    def _extract_learning(self, order: Order, reason: str) -> str:
        """Extract learning point from order failure."""
        if 'latency' in reason.lower():
            return "Consider reducing order size or improving execution timing during high volatility"
        elif 'liquidity' in reason.lower():
            return "Check order book depth before submitting large orders"
        elif 'slippage' in reason.lower():
            return "Implement tighter slippage tolerance or use limit orders"
        else:
            return f"Review order parameters: {order.symbol} {order.side} {order.quantity}"


class OrderManager:
    """
    Central order manager for lifecycle tracking.
    
    Manages all orders, handles state transitions,
    and provides real-time order status queries.
    """
    
    def __init__(self, soul_logger_callback: Optional[Callable] = None):
        """Initialize order manager."""
        self._orders: Dict[str, Order] = {}
        self._orders_by_exchange_id: Dict[str, str] = {}  # exchange_id -> order_id
        self._orders_by_client_id: Dict[str, str] = {}  # client_order_id -> order_id
        self._state_machine = OrderStateMachine(soul_logger_callback)
        self._lock = asyncio.Lock()
        
        logger.info("Order manager initialized")
    
    async def create_order(self,
                          symbol: str,
                          side: str,
                          quantity: Decimal,
                          order_type: str,
                          price: Optional[Decimal] = None,
                          time_in_force: str = "GTC",
                          metadata: Optional[Dict[str, Any]] = None) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity
            order_type: LIMIT, MARKET, etc.
            price: Limit price (optional)
            time_in_force: GTC, IOC, FOK
            metadata: Additional order metadata
        
        Returns:
            Created Order object
        """
        order_id = f"ORD_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        client_order_id = f"CLI_{int(time.time() * 1000)}"
        
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol.upper(),
            side=side.upper(),
            order_type=order_type.upper(),
            quantity=quantity,
            price=price,
            time_in_force=time_in_force.upper(),
            created_at=int(time.time() * 1000),
            metadata=metadata or {},
        )
        
        async with self._lock:
            self._orders[order_id] = order
            self._orders_by_client_id[client_order_id] = order_id
        
        logger.info(f"Order created: {order_id} - {order_type} {quantity} {symbol}")
        return order
    
    async def submit_order(self, order_id: str, exchange_order_id: str) -> bool:
        """Mark order as submitted to exchange."""
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False
        
        order.exchange_order_id = exchange_order_id
        
        async with self._lock:
            self._orders_by_exchange_id[exchange_order_id] = order_id
        
        success = await self._state_machine.transition(order, OrderState.SUBMITTED)
        if success:
            logger.info(f"Order submitted: {order_id} -> {exchange_order_id}")
        return success
    
    async def acknowledge_order(self, order_id: str) -> bool:
        """Mark order as acknowledged by exchange."""
        order = self._orders.get(order_id)
        if not order:
            return False
        
        success = await self._state_machine.transition(order, OrderState.ACKNOWLEDGED)
        if success:
            logger.debug(f"Order acknowledged: {order_id}")
        return success
    
    async def add_fill(self,
                      order_id: str,
                      price: Decimal,
                      quantity: Decimal,
                      commission: Decimal,
                      commission_asset: str,
                      trade_id: Optional[int] = None) -> bool:
        """
        Add a fill to an order.
        
        Args:
            order_id: Order ID
            price: Fill price
            quantity: Fill quantity
            commission: Commission amount
            commission_asset: Commission asset
            trade_id: Exchange trade ID
        
        Returns:
            True if fill added successfully
        """
        order = self._orders.get(order_id)
        if not order:
            logger.error(f"Order not found: {order_id}")
            return False
        
        fill = Fill(
            fill_id=f"FILL_{int(time.time() * 1000)}_{os.urandom(2).hex()}",
            order_id=order_id,
            price=price,
            quantity=quantity,
            commission=commission,
            commission_asset=commission_asset,
            timestamp=int(time.time() * 1000),
            trade_id=trade_id,
        )
        
        return await self._state_machine.add_fill(order, fill)
    
    async def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """Cancel an active order."""
        order = self._orders.get(order_id)
        if not order:
            return False
        
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active, cannot cancel")
            return False
        
        success = await self._state_machine.transition(order, OrderState.CANCELLED, reason)
        if success:
            logger.info(f"Order cancelled: {order_id} - {reason}")
        return success
    
    async def reject_order(self, order_id: str, reason: str) -> bool:
        """Mark order as rejected."""
        order = self._orders.get(order_id)
        if not order:
            return False
        
        success = await self._state_machine.transition(order, OrderState.REJECTED, reason)
        if success:
            logger.warning(f"Order rejected: {order_id} - {reason}")
        return success
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[Order]:
        """Get order by exchange order ID."""
        order_id = self._orders_by_exchange_id.get(exchange_order_id)
        return self._orders.get(order_id) if order_id else None
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        order_id = self._orders_by_client_id.get(client_order_id)
        return self._orders.get(order_id) if order_id else None
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        orders = [o for o in self._orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        return orders
    
    def get_all_orders(self) -> Dict[str, Order]:
        """Get all orders."""
        return self._orders.copy()
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        total = len(self._orders)
        active = len([o for o in self._orders.values() if o.is_active])
        filled = len([o for o in self._orders.values() if o.state == OrderState.FILLED])
        cancelled = len([o for o in self._orders.values() if o.state == OrderState.CANCELLED])
        rejected = len([o for o in self._orders.values() if o.state == OrderState.REJECTED])
        
        return {
            'total_orders': total,
            'active_orders': active,
            'filled_orders': filled,
            'cancelled_orders': cancelled,
            'rejected_orders': rejected,
            'fill_rate': filled / total if total > 0 else 0.0,
        }


async def mock_soul_logger(lesson: Dict[str, Any]):
    """Mock SOUL logger for testing."""
    print(f"SOUL Lesson: {json.dumps(lesson, indent=2)}")


async def main():
    """Example usage of order manager."""
    manager = OrderManager(mock_soul_logger)
    
    # Create order
    order = await manager.create_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal('0.5'),
        order_type="LIMIT",
        price=Decimal('50000'),
    )
    
    print(f"Created order: {order.order_id}")
    print(f"State: {order.state.value}")
    
    # Submit order
    await manager.submit_order(order.order_id, "EXCH_12345")
    print(f"After submit: {order.state.value}")
    
    # Acknowledge
    await manager.acknowledge_order(order.order_id)
    print(f"After ack: {order.state.value}")
    
    # Add fill
    await manager.add_fill(
        order.order_id,
        price=Decimal('50000'),
        quantity=Decimal('0.3'),
        commission=Decimal('0.0003'),
        commission_asset="BTC",
    )
    print(f"After partial fill: {order.state.value} ({order.fill_pct:.1f}%)")
    
    # Add another fill
    await manager.add_fill(
        order.order_id,
        price=Decimal('50001'),
        quantity=Decimal('0.2'),
        commission=Decimal('0.0002'),
        commission_asset="BTC",
    )
    print(f"After full fill: {order.state.value} ({order.fill_pct:.1f}%)")
    
    # Print statistics
    stats = manager.get_order_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
