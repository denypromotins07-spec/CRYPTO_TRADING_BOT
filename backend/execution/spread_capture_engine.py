"""
Spread Capture Engine
Manages limit order cancellations and replacements for optimal spread capture.
Optimized for sub-50 microsecond quote updates to avoid being run over.

This module implements:
- Fast order cancellation and replacement
- Spread monitoring and capture optimization
- Quote refresh logic based on market conditions
- Performance tracking for spread capture rates

Target: Capture bid-ask spreads while minimizing adverse selection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
from datetime import datetime, timezone


class OrderStatus(Enum):
    """Status of a limit order."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class QuoteAction(Enum):
    """Action to take on quotes."""
    HOLD = "hold"
    UPDATE = "update"
    CANCEL = "cancel"
    WIDEN = "widen"
    NARROW = "narrow"


@dataclass
class LimitOrder:
    """Represents a limit order in the book."""
    order_id: str
    symbol: str
    side: str  # 'bid' or 'ask'
    price: float
    quantity: float
    filled_quantity: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at_ms: int = 0
    updated_at_ms: int = 0
    fill_count: int = 0
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED)
    
    @property
    def age_ms(self) -> int:
        current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        return current_ms - self.created_at_ms


@dataclass
class SpreadCaptureConfig:
    """Configuration for spread capture strategy."""
    # Minimum spread to quote (basis points)
    min_spread_bps: float = 2.0
    
    # Target spread (basis points)
    target_spread_bps: float = 5.0
    
    # Maximum quote age before refresh (milliseconds)
    max_quote_age_ms: int = 5000
    
    # Fast cancel threshold (microseconds)
    fast_cancel_threshold_us: int = 50
    
    # Queue position threshold for cancellation
    queue_position_threshold: float = 0.7
    
    # Profit target per cycle (basis points)
    profit_target_bps: float = 3.0
    
    # Maximum inventory per symbol
    max_inventory: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.5,
        "ETH": 5.0,
        "SOL": 50.0,
    })


@dataclass
class SpreadCaptureEvent:
    """Records a spread capture event."""
    timestamp_ms: int
    symbol: str
    action: str
    bid_price: Optional[float]
    ask_price: Optional[float]
    spread_bps: float
    captured_amount: float = 0.0
    pnl: float = 0.0


class SpreadCaptureEngine:
    """
    Main engine for capturing bid-ask spreads.
    
    Manages the lifecycle of limit orders, optimizing for:
    - Fast reaction to market changes
    - Minimal latency in quote updates
    - Maximum spread capture with minimal risk
    """
    
    def __init__(self, config: Optional[SpreadCaptureConfig] = None):
        self.config = config or SpreadCaptureConfig()
        
        # Active orders by symbol
        self.active_orders: Dict[str, Dict[str, LimitOrder]] = {}  # symbol -> {order_id -> order}
        
        # Current quotes by symbol
        self.current_quotes: Dict[str, Dict] = {}  # symbol -> {bid_price, ask_price, bid_size, ask_size}
        
        # Event history
        self.events: List[SpreadCaptureEvent] = []
        
        # Statistics
        self.stats: Dict[str, Dict] = {}
        
        # Latency tracking
        self.cancel_latencies: List[float] = []
        self.update_latencies: List[float] = []
        
        # Inventory tracking
        self.inventory: Dict[str, float] = {}
    
    def initialize_symbol(self, symbol: str) -> None:
        """Initialize tracking for a new symbol."""
        if symbol not in self.active_orders:
            self.active_orders[symbol] = {}
        if symbol not in self.current_quotes:
            self.current_quotes[symbol] = {
                'bid_price': None,
                'ask_price': None,
                'bid_size': 0.0,
                'ask_size': 0.0,
                'last_update_ms': 0,
            }
        if symbol not in self.stats:
            self.stats[symbol] = {
                'total_captures': 0,
                'total_volume': 0.0,
                'total_pnl': 0.0,
                'avg_spread_captured': 0.0,
            }
        if symbol not in self.inventory:
            self.inventory[symbol] = 0.0
    
    def place_quote(
        self,
        symbol: str,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float,
        timestamp_ms: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Place new bid and ask quotes.
        
        Returns order IDs for bid and ask orders.
        """
        self.initialize_symbol(symbol)
        
        bid_order_id = None
        ask_order_id = None
        
        # Place bid order
        if bid_size > 0:
            bid_order_id = f"{symbol}_BID_{timestamp_ms}"
            bid_order = LimitOrder(
                order_id=bid_order_id,
                symbol=symbol,
                side='bid',
                price=bid_price,
                quantity=bid_size,
                status=OrderStatus.ACTIVE,
                created_at_ms=timestamp_ms,
                updated_at_ms=timestamp_ms,
            )
            self.active_orders[symbol][bid_order_id] = bid_order
        
        # Place ask order
        if ask_size > 0:
            ask_order_id = f"{symbol}_ASK_{timestamp_ms}"
            ask_order = LimitOrder(
                order_id=ask_order_id,
                symbol=symbol,
                side='ask',
                price=ask_price,
                quantity=ask_size,
                status=OrderStatus.ACTIVE,
                created_at_ms=timestamp_ms,
                updated_at_ms=timestamp_ms,
            )
            self.active_orders[symbol][ask_order_id] = ask_order
        
        # Update current quotes
        self.current_quotes[symbol].update({
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'last_update_ms': timestamp_ms,
        })
        
        return bid_order_id, ask_order_id
    
    def cancel_order(self, symbol: str, order_id: str, timestamp_ms: int) -> bool:
        """
        Cancel an existing order.
        
        Returns True if cancellation was successful.
        """
        if symbol not in self.active_orders:
            return False
        
        order = self.active_orders[symbol].get(order_id)
        if order is None or not order.is_active:
            return False
        
        # Simulate cancellation (in production, this would call exchange API)
        start_time = time.perf_counter_ns()
        
        order.status = OrderStatus.CANCELLED
        order.updated_at_ms = timestamp_ms
        
        elapsed_us = (time.perf_counter_ns() - start_time) / 1000
        self.cancel_latencies.append(elapsed_us)
        
        # Keep only recent latencies
        if len(self.cancel_latencies) > 1000:
            self.cancel_latencies.pop(0)
        
        return True
    
    def cancel_both_quotes(self, symbol: str, timestamp_ms: int) -> bool:
        """Cancel both bid and ask quotes for a symbol."""
        if symbol not in self.active_orders:
            return False
        
        cancelled = False
        for order_id, order in list(self.active_orders[symbol].items()):
            if order.is_active:
                if self.cancel_order(symbol, order_id, timestamp_ms):
                    cancelled = True
        
        return cancelled
    
    def update_quotes(
        self,
        symbol: str,
        new_bid_price: float,
        new_ask_price: float,
        timestamp_ms: int
    ) -> bool:
        """
        Update existing quotes (cancel and replace).
        
        Returns True if update was successful.
        """
        start_time = time.perf_counter_ns()
        
        # Cancel existing orders
        self.cancel_both_quotes(symbol, timestamp_ms)
        
        # Get sizes from current quotes
        current = self.current_quotes.get(symbol, {})
        bid_size = current.get('bid_size', 0.0)
        ask_size = current.get('ask_size', 0.0)
        
        # Place new orders
        self.place_quote(symbol, new_bid_price, new_ask_price, bid_size, ask_size, timestamp_ms)
        
        elapsed_us = (time.perf_counter_ns() - start_time) / 1000
        self.update_latencies.append(elapsed_us)
        
        # Keep only recent latencies
        if len(self.update_latencies) > 1000:
            self.update_latencies.pop(0)
        
        return True
    
    def should_refresh_quotes(self, symbol: str, current_time_ms: int) -> QuoteAction:
        """
        Determine if quotes should be refreshed.
        
        Returns recommended action based on market conditions.
        """
        if symbol not in self.current_quotes:
            return QuoteAction.UPDATE
        
        current = self.current_quotes[symbol]
        last_update = current.get('last_update_ms', 0)
        age_ms = current_time_ms - last_update
        
        # Check if quotes are stale
        if age_ms > self.config.max_quote_age_ms:
            return QuoteAction.UPDATE
        
        # Check if we're too far from mid (would need market data integration)
        # For now, just check age
        
        return QuoteAction.HOLD
    
    def record_fill(
        self,
        symbol: str,
        order_id: str,
        fill_price: float,
        fill_quantity: float,
        timestamp_ms: int
    ) -> None:
        """Record a fill and update statistics."""
        if symbol not in self.active_orders:
            return
        
        order = self.active_orders[symbol].get(order_id)
        if order is None:
            return
        
        # Update order
        order.filled_quantity += fill_quantity
        order.fill_count += 1
        order.updated_at_ms = timestamp_ms
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Update inventory
        if order.side == 'bid':
            self.inventory[symbol] = self.inventory.get(symbol, 0.0) + fill_quantity
        else:
            self.inventory[symbol] = self.inventory.get(symbol, 0.0) - fill_quantity
        
        # Record event
        event = SpreadCaptureEvent(
            timestamp_ms=timestamp_ms,
            symbol=symbol,
            action='fill',
            bid_price=order.price if order.side == 'bid' else None,
            ask_price=order.price if order.side == 'ask' else None,
            spread_bps=0.0,  # Would calculate from current spread
            captured_amount=fill_quantity,
        )
        self.events.append(event)
        
        # Update stats
        self.stats[symbol]['total_captures'] += 1
        self.stats[symbol]['total_volume'] += fill_quantity
    
    def get_statistics(self) -> Dict:
        """Get comprehensive spread capture statistics."""
        avg_cancel_latency = sum(self.cancel_latencies) / len(self.cancel_latencies) if self.cancel_latencies else 0
        avg_update_latency = sum(self.update_latencies) / len(self.update_latencies) if self.update_latencies else 0
        
        return {
            'symbols': list(self.active_orders.keys()),
            'total_events': len(self.events),
            'active_orders_count': sum(
                sum(1 for o in orders.values() if o.is_active)
                for orders in self.active_orders.values()
            ),
            'average_cancel_latency_us': avg_cancel_latency,
            'average_update_latency_us': avg_update_latency,
            'per_symbol_stats': dict(self.stats),
            'inventory': dict(self.inventory),
        }
    
    def get_pnl_summary(self) -> Dict:
        """Get P&L summary for spread capture activity."""
        total_pnl = sum(e.pnl for e in self.events)
        total_captured = sum(e.captured_amount for e in self.events)
        
        return {
            'total_pnl': total_pnl,
            'total_captured_volume': total_captured,
            'event_count': len(self.events),
            'avg_pnl_per_event': total_pnl / len(self.events) if self.events else 0,
        }


# Example usage and testing
if __name__ == "__main__":
    engine = SpreadCaptureEngine()
    
    base_ts = int(time.time() * 1000)
    
    # Place initial quotes
    bid_id, ask_id = engine.place_quote(
        symbol="BTC",
        bid_price=49990.0,
        ask_price=50010.0,
        bid_size=0.1,
        ask_size=0.1,
        timestamp_ms=base_ts
    )
    
    print(f"Placed orders: BID={bid_id}, ASK={ask_id}")
    print(f"Spread: {50010.0 - 49990.0} USD ({(20/50000)*10000:.1f} bps)")
    
    # Simulate a fill
    if bid_id:
        engine.record_fill("BTC", bid_id, 49990.0, 0.1, base_ts + 100)
    
    # Check statistics
    stats = engine.get_statistics()
    print(f"\nStatistics:")
    print(f"  Active Orders: {stats['active_orders_count']}")
    print(f"  Avg Cancel Latency: {stats['average_cancel_latency_us']:.1f} μs")
    print(f"  Avg Update Latency: {stats['average_update_latency_us']:.1f} μs")
    
    # Test quote refresh decision
    action = engine.should_refresh_quotes("BTC", base_ts + 6000)
    print(f"\nQuote Refresh Action: {action.value}")
    
    # P&L summary
    pnl = engine.get_pnl_summary()
    print(f"\nP&L Summary:")
    print(f"  Total Captured: {pnl['total_captured_volume']}")
    print(f"  Events: {pnl['event_count']}")
