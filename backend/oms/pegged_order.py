"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 32
Advanced Order Management System (OMS) - Chapter 2
File: backend/oms/pegged_order.py

Dynamically adjusts limit prices to track the mid-market or best bid.
Updates prices only when the mid-market moves by a full tick size.
Optimized for AMD Ryzen AI 5, strictly respecting 8GB RAM limit.
Targets 8k-20k INR/hour in a 4hr trading window.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Tuple
from threading import Lock, RLock
from collections import defaultdict
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PegType(Enum):
    """Types of price pegging strategies."""
    MID_MARKET = auto()      # Peg to mid-point of bid-ask
    BEST_BID = auto()        # Peg to best bid (for sells)
    BEST_ASK = auto()        # Peg to best ask (for buys)
    WEIGHTED_MID = auto()    # Weighted mid based on order book depth
    LAST_TRADE = auto()      # Peg to last traded price


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book for peg calculations."""
    timestamp: float
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    mid_price: float = field(init=False)
    spread: float = field(init=False)
    spread_pct: float = field(init=False)

    def __post_init__(self):
        self.mid_price = (self.best_bid + self.best_ask) / 2.0
        self.spread = self.best_ask - self.best_bid
        self.spread_pct = (self.spread / self.mid_price * 100.0) if self.mid_price > 0 else 0.0


@dataclass
class PeggedOrderConfig:
    """Configuration for a pegged order."""
    symbol: str
    side: str  # BUY or SELL
    base_quantity: float
    peg_type: PegType = PegType.MID_MARKET
    offset_ticks: int = 0  # Number of ticks to offset from peg
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    update_threshold_ticks: int = 1  # Minimum move to trigger update
    max_updates_per_second: float = 10.0  # Rate limiting
    cancel_on_wide_spread: bool = True
    spread_threshold_pct: float = 0.5  # Cancel if spread exceeds this


@dataclass
class PeggedOrder:
    """Represents an active pegged order."""
    order_id: str
    config: PeggedOrderConfig
    current_price: float
    current_peg_reference: float
    last_update_time: float = field(default_factory=time.time)
    update_count: int = 0
    status: str = "ACTIVE"  # ACTIVE, SUSPENDED, CANCELLED, FILLED
    exchange_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    _lock: RLock = field(default_factory=RLock, repr=False)

    def __post_init__(self):
        self.remaining_quantity = self.config.base_quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"

    def get_effective_price(self, tick_size: float) -> float:
        """Calculate effective price with offset applied."""
        offset = self.config.offset_ticks * tick_size
        
        if self.config.side == "BUY":
            # For buys, offset downward (better price)
            return self.current_price - offset
        else:
            # For sells, offset upward (better price)
            return self.current_price + offset


class PeggedOrderManager:
    """
    Manages dynamically pegged orders that track market prices.
    
    Features:
    - Multiple peg types (mid, best bid/ask, weighted)
    - Tick-size-based update threshold to prevent noise
    - Rate limiting to avoid API spam
    - Wide spread detection and automatic suspension
    - Thread-safe price updates
    """

    def __init__(
        self,
        tick_sizes: Dict[str, float],
        max_orders: int = 500,
    ):
        """
        Initialize the pegged order manager.
        
        Args:
            tick_sizes: Mapping of symbol to minimum tick size
            max_orders: Maximum concurrent pegged orders
        """
        self.tick_sizes = tick_sizes
        self.max_orders = max_orders
        
        # Storage
        self._orders: Dict[str, PeggedOrder] = {}
        self._symbol_index: Dict[str, set] = defaultdict(set)
        
        # Rate limiting: order_id -> last_update_time
        self._rate_limits: Dict[str, float] = {}
        
        # Last known peg reference per order: order_id -> reference_price
        self._peg_references: Dict[str, float] = {}
        
        # Locks
        self._global_lock = Lock()
        
        logger.info(f"PeggedOrderManager initialized with max_orders={max_orders}")

    def create_pegged_order(
        self,
        config: PeggedOrderConfig,
        initial_book: OrderBookSnapshot,
    ) -> PeggedOrder:
        """
        Create a new pegged order with initial pricing.
        
        Args:
            config: Order configuration
            initial_book: Initial order book snapshot
            
        Returns:
            PeggedOrder instance
        """
        with self._global_lock:
            if len(self._orders) >= self.max_orders:
                raise RuntimeError(f"Maximum pegged orders ({self.max_orders}) reached")

        # Calculate initial peg price
        peg_price = self._calculate_peg_price(config.peg_type, initial_book)
        
        # Apply offset
        tick_size = self.tick_sizes.get(config.symbol, 0.01)
        effective_price = self._apply_offset_and_constraints(
            peg_price, config.side, config.offset_ticks, tick_size,
            config.min_price, config.max_price
        )

        order_id = f"PEG_{uuid.uuid4().hex[:12]}"
        order = PeggedOrder(
            order_id=order_id,
            config=config,
            current_price=effective_price,
            current_peg_reference=peg_price,
            remaining_quantity=config.base_quantity,
        )

        # Register order
        with self._global_lock:
            self._orders[order_id] = order
            self._symbol_index[config.symbol].add(order_id)
            self._peg_references[order_id] = peg_price

        logger.info(
            f"Created pegged order {order_id}: {config.side} {config.base_quantity} "
            f"{config.symbol} @ {effective_price} (peg: {peg_price})"
        )
        
        return order

    def update_order_prices(
        self,
        symbol: str,
        book: OrderBookSnapshot,
    ) -> List[Tuple[str, float]]:
        """
        Update all pegged orders for a symbol based on new order book.
        
        CRITICAL: Only updates if mid-market moves by full tick size.
        This prevents whipsaw from microsecond spread noise.
        
        Args:
            symbol: Symbol to update
            book: New order book snapshot
            
        Returns:
            List of (order_id, new_price) tuples for updated orders
        """
        tick_size = self.tick_sizes.get(symbol, 0.01)
        updates = []
        
        now = time.time()
        order_ids = list(self._symbol_index.get(symbol, set()))
        
        for order_id in order_ids:
            order = self._orders.get(order_id)
            if not order or not order.is_active:
                continue

            with order._lock:
                # Check rate limit
                last_update = self._rate_limits.get(order_id, 0.0)
                min_interval = 1.0 / order.config.max_updates_per_second
                if now - last_update < min_interval:
                    continue

                # Check spread threshold
                if order.config.cancel_on_wide_spread:
                    if book.spread_pct > order.config.spread_threshold_pct:
                        logger.warning(
                            f"Order {order_id}: Spread too wide ({book.spread_pct:.2f}%), suspending"
                        )
                        order.status = "SUSPENDED"
                        continue

                # Calculate new peg price
                new_peg = self._calculate_peg_price(order.config.peg_type, book)
                
                # Check if move exceeds threshold
                old_peg = self._peg_references.get(order_id, new_peg)
                price_move = abs(new_peg - old_peg)
                
                if price_move < tick_size * order.config.update_threshold_ticks:
                    # Move too small, skip update
                    continue

                # Apply offset and constraints
                new_price = self._apply_offset_and_constraints(
                    new_peg,
                    order.config.side,
                    order.config.offset_ticks,
                    tick_size,
                    order.config.min_price,
                    order.config.max_price,
                )

                # Check if price actually changed
                if new_price != order.current_price:
                    order.current_price = new_price
                    order.current_peg_reference = new_peg
                    order.last_update_time = now
                    order.update_count += 1
                    
                    self._peg_references[order_id] = new_peg
                    self._rate_limits[order_id] = now
                    
                    updates.append((order_id, new_price))
                    
                    logger.debug(
                        f"Updated order {order_id}: {old_peg:.2f} -> {new_peg:.2f} "
                        f"(effective: {new_price:.2f})"
                    )

        return updates

    def _calculate_peg_price(
        self,
        peg_type: PegType,
        book: OrderBookSnapshot,
    ) -> float:
        """Calculate peg reference price based on type."""
        if peg_type == PegType.MID_MARKET:
            return book.mid_price
        elif peg_type == PegType.BEST_BID:
            return book.best_bid
        elif peg_type == PegType.BEST_ASK:
            return book.best_ask
        elif peg_type == PegType.WEIGHTED_MID:
            # Volume-weighted mid price
            if book.bid_size + book.ask_size > 0:
                weight = book.bid_size / (book.bid_size + book.ask_size)
                return book.best_bid * weight + book.best_ask * (1 - weight)
            return book.mid_price
        elif peg_type == PegType.LAST_TRADE:
            # Would need last trade price from market data
            return book.mid_price
        else:
            return book.mid_price

    def _apply_offset_and_constraints(
        self,
        peg_price: float,
        side: str,
        offset_ticks: int,
        tick_size: float,
        min_price: Optional[float],
        max_price: Optional[float],
    ) -> float:
        """Apply offset and enforce price constraints."""
        offset = offset_ticks * tick_size
        
        if side == "BUY":
            # Buy orders: offset below peg (more aggressive)
            price = peg_price - offset
        else:
            # Sell orders: offset above peg
            price = peg_price + offset

        # Round to tick size
        price = round(price / tick_size) * tick_size

        # Apply constraints
        if min_price is not None:
            price = max(price, min_price)
        if max_price is not None:
            price = min(price, max_price)

        return price

    def handle_fill(
        self,
        order_id: str,
        filled_qty: float,
        fill_price: float,
    ) -> bool:
        """Handle partial or full fill of a pegged order."""
        order = self._orders.get(order_id)
        if not order:
            return False

        with order._lock:
            order.filled_quantity += filled_qty
            order.remaining_quantity = order.config.base_quantity - order.filled_quantity

            if order.remaining_quantity <= 0:
                order.status = "FILLED"
                logger.info(f"Pegged order {order_id} fully filled @ {fill_price}")
            else:
                logger.debug(
                    f"Pegged order {order_id} partially filled: {filled_qty} @ {fill_price}, "
                    f"remaining: {order.remaining_quantity}"
                )

        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pegged order."""
        order = self._orders.get(order_id)
        if not order:
            return False

        with order._lock:
            order.status = "CANCELLED"
            
            # Clean up indices
            self._symbol_index[order.config.symbol].discard(order_id)
            self._peg_references.pop(order_id, None)
            self._rate_limits.pop(order_id, None)

        logger.info(f"Cancelled pegged order {order_id}")
        return True

    def suspend_order(self, order_id: str) -> bool:
        """Suspend a pegged order (e.g., due to wide spread)."""
        order = self._orders.get(order_id)
        if not order:
            return False

        with order._lock:
            order.status = "SUSPENDED"

        logger.info(f"Suspended pegged order {order_id}")
        return True

    def resume_order(self, order_id: str, current_book: OrderBookSnapshot) -> bool:
        """Resume a suspended pegged order with fresh pricing."""
        order = self._orders.get(order_id)
        if not order or order.status != "SUSPENDED":
            return False

        with order._lock:
            tick_size = self.tick_sizes.get(order.config.symbol, 0.01)
            new_peg = self._calculate_peg_price(order.config.peg_type, current_book)
            
            new_price = self._apply_offset_and_constraints(
                new_peg,
                order.config.side,
                order.config.offset_ticks,
                tick_size,
                order.config.min_price,
                order.config.max_price,
            )

            order.current_price = new_price
            order.current_peg_reference = new_peg
            order.status = "ACTIVE"
            order.last_update_time = time.time()

            self._peg_references[order_id] = new_peg

        logger.info(f"Resumed pegged order {order_id} @ {new_price}")
        return True

    def get_order(self, order_id: str) -> Optional[PeggedOrder]:
        """Get pegged order by ID."""
        return self._orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[PeggedOrder]:
        """Get all active pegged orders, optionally filtered by symbol."""
        if symbol:
            order_ids = self._symbol_index.get(symbol, set())
        else:
            order_ids = set(self._orders.keys())

        return [
            self._orders[oid] 
            for oid in order_ids 
            if (order := self._orders.get(oid)) and order.is_active
        ]

    def get_total_exposure(self, symbol: Optional[str] = None) -> float:
        """Calculate total quantity exposure from active pegged orders."""
        orders = self.get_active_orders(symbol)
        return sum(o.remaining_quantity for o in orders)

    def cleanup_inactive(self) -> int:
        """Remove filled/cancelled orders from memory."""
        inactive_ids = [
            oid for oid, order in self._orders.items()
            if order.status in ("FILLED", "CANCELLED")
        ]

        for oid in inactive_ids:
            order = self._orders.pop(oid, None)
            if order:
                self._symbol_index[order.config.symbol].discard(oid)
                self._peg_references.pop(oid, None)
                self._rate_limits.pop(oid, None)

        if inactive_ids:
            logger.info(f"Cleaned up {len(inactive_ids)} inactive pegged orders")

        return len(inactive_ids)


# Example usage and testing
if __name__ == "__main__":
    # Configure tick sizes for common symbols
    TICK_SIZES = {
        "BTCUSDT": 0.01,
        "ETHUSDT": 0.01,
        "SOLUSDT": 0.001,
    }

    manager = PeggedOrderManager(tick_sizes=TICK_SIZES)

    # Create initial order book
    book = OrderBookSnapshot(
        timestamp=time.time(),
        best_bid=49950.0,
        best_ask=49960.0,
        bid_size=10.0,
        ask_size=8.0,
    )

    # Create a pegged buy order at mid-market
    config = PeggedOrderConfig(
        symbol="BTCUSDT",
        side="BUY",
        base_quantity=0.5,
        peg_type=PegType.MID_MARKET,
        offset_ticks=-1,  # 1 tick better than mid
        update_threshold_ticks=1,
    )

    order = manager.create_pegged_order(config, book)
    print(f"Created order: {order.order_id} @ {order.current_price}")

    # Simulate order book movement
    new_book = OrderBookSnapshot(
        timestamp=time.time(),
        best_bid=49960.0,
        best_ask=49970.0,
        bid_size=12.0,
        ask_size=9.0,
    )

    updates = manager.update_order_prices("BTCUSDT", new_book)
    print(f"Updates: {updates}")

    # Cleanup
    manager.cleanup_inactive()
