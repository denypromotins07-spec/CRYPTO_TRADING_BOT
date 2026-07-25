"""
Order Book Manager for ZAID Personal Crypto Trading Bot
========================================================
Chapter 2: High-Frequency WebSocket Market Data Feeds and Order Book State

Maintains a local L2 order book state with real-time updates from Binance WebSocket.
Optimized for microsecond access patterns on AMD Ryzen AI 5.

Features:
- Real-time L2 order book maintenance
- Snapshot + delta update reconciliation
- O(1) best bid/ask access
- Depth aggregation at multiple levels
- Support for BTC, SOL, ETH, USDT pairs
- Thread-safe concurrent access

Author: Opus 4.8
Stage: 2 of 100
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from decimal import Decimal
from collections import defaultdict
import logging
from sortedcontainers import SortedDict
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class OrderBookState(Enum):
    """Order book synchronization state."""
    DISCONNECTED = "disconnected"
    INITIALIZING = "initializing"
    SYNCHRONIZING = "synchronizing"
    READY = "ready"
    STALE = "stale"


@dataclass
class PriceLevel:
    """Represents a single price level in the order book."""
    price: Decimal
    quantity: Decimal
    order_count: int = 1
    
    def __post_init__(self):
        if isinstance(self.price, (int, float, str)):
            self.price = Decimal(str(self.price))
        if isinstance(self.quantity, (int, float, str)):
            self.quantity = Decimal(str(self.quantity))


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book at a point in time."""
    symbol: str
    bids: List[Tuple[Decimal, Decimal]]  # (price, quantity)
    asks: List[Tuple[Decimal, Decimal]]
    timestamp: int
    last_update_id: int


class LocalOrderBook:
    """
    High-performance local order book implementation.
    
    Uses SortedDict for O(log n) insert/delete and O(1) best bid/ask access.
    Maintains separate books for bids (descending) and asks (ascending).
    
    Thread-safe for concurrent read/write operations across trading threads.
    """
    
    def __init__(self, symbol: str, max_depth: int = 1000):
        """
        Initialize order book for a symbol.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            max_depth: Maximum depth levels to maintain
        """
        self.symbol = symbol
        self.max_depth = max_depth
        
        # Bids: sorted descending (highest first)
        # We use negative prices for descending sort
        self._bids: SortedDict = SortedDict()
        
        # Asks: sorted ascending (lowest first)
        self._asks: SortedDict = SortedDict()
        
        # Update tracking
        self._last_update_id: int = 0
        self._first_update_id: Optional[int] = None
        self._timestamp: int = 0
        
        # State
        self._state: OrderBookState = OrderBookState.DISCONNECTED
        self._lock = threading.RLock()
        
        # Statistics
        self._update_count: int = 0
        self._snapshot_count: int = 0
        self._last_message_time: float = 0
        
        logger.debug(f"Order book initialized for {symbol} (max_depth={max_depth})")
    
    @property
    def state(self) -> OrderBookState:
        """Get current order book state."""
        return self._state
    
    @state.setter
    def state(self, value: OrderBookState):
        """Set order book state."""
        self._state = value
    
    def set_snapshot(self, snapshot: OrderBookSnapshot):
        """
        Initialize order book from a snapshot.
        
        Args:
            snapshot: Complete order book snapshot
        """
        with self._lock:
            self._bids.clear()
            self._asks.clear()
            
            # Add bids (use negative price for descending sort)
            for price, qty in snapshot.bids:
                neg_price = -float(price)
                self._bids[neg_price] = PriceLevel(price=price, quantity=qty)
            
            # Add asks
            for price, qty in snapshot.asks:
                pos_price = float(price)
                self._asks[pos_price] = PriceLevel(price=price, quantity=qty)
            
            self._last_update_id = snapshot.last_update_id
            self._first_update_id = snapshot.last_update_id
            self._timestamp = snapshot.timestamp
            self._last_message_time = time.time()
            self._state = OrderBookState.READY
            
            logger.info(f"Order book snapshot loaded for {self.symbol}: "
                       f"{len(self._bids)} bids, {len(self._asks)} asks")
    
    def apply_delta(self, 
                    bids: List[Tuple[str, str]], 
                    asks: List[Tuple[str, str]], 
                    update_id: int,
                    timestamp: int) -> bool:
        """
        Apply incremental update to order book.
        
        Args:
            bids: List of (price, quantity) updates for bids
            asks: List of (price, quantity) updates for asks
            update_id: Update sequence ID
            timestamp: Update timestamp
        
        Returns:
            True if update applied successfully, False if skipped
        """
        with self._lock:
            # Check for gaps in update sequence
            if self._first_update_id is not None:
                if update_id <= self._last_update_id:
                    # Duplicate update, skip
                    logger.debug(f"Skipping duplicate update {update_id} <= {self._last_update_id}")
                    return False
                
                if update_id > self._last_update_id + 1:
                    # Gap detected
                    logger.warning(f"Update gap detected: expected {self._last_update_id + 1}, got {update_id}")
                    self._state = OrderBookState.STALE
            
            # Apply bid updates
            for price_str, qty_str in bids:
                price = Decimal(price_str)
                quantity = Decimal(qty_str)
                neg_price = -float(price)
                
                if quantity == 0:
                    # Remove level
                    self._bids.pop(neg_price, None)
                else:
                    # Update/add level
                    self._bids[neg_price] = PriceLevel(price=price, quantity=quantity)
            
            # Apply ask updates
            for price_str, qty_str in asks:
                price = Decimal(price_str)
                quantity = Decimal(qty_str)
                pos_price = float(price)
                
                if quantity == 0:
                    # Remove level
                    self._asks.pop(pos_price, None)
                else:
                    # Update/add level
                    self._asks[pos_price] = PriceLevel(price=price, quantity=quantity)
            
            # Trim to max depth
            self._trim_depth()
            
            # Update metadata
            self._last_update_id = update_id
            self._timestamp = timestamp
            self._last_message_time = time.time()
            self._update_count += 1
            self._state = OrderBookState.READY
            
            return True
    
    def _trim_depth(self):
        """Trim order book to maximum depth."""
        # Trim bids
        while len(self._bids) > self.max_depth:
            self._bids.popitem(index=-1)  # Remove lowest bid (highest negative)
        
        # Trim asks
        while len(self._asks) > self.max_depth:
            self._asks.popitem(index=-1)  # Remove highest ask
    
    def get_best_bid(self) -> Optional[PriceLevel]:
        """Get best (highest) bid."""
        with self._lock:
            if self._bids:
                return self._bids.peekitem(index=0)[1]  # First item (highest bid)
            return None
    
    def get_best_ask(self) -> Optional[PriceLevel]:
        """Get best (lowest) ask."""
        with self._lock:
            if self._asks:
                return self._asks.peekitem(index=0)[1]  # First item (lowest ask)
            return None
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price (average of best bid and ask)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None
    
    def get_spread(self) -> Optional[Decimal]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        mid = self.get_mid_price()
        spread = self.get_spread()
        
        if mid and spread and mid > 0:
            return float(spread / mid * 10000)
        return None
    
    def get_bids(self, depth: int = 10) -> List[Tuple[Decimal, Decimal]]:
        """Get top N bid levels."""
        with self._lock:
            result = []
            for i in range(min(depth, len(self._bids))):
                level = self._bids.peekitem(index=i)[1]
                result.append((level.price, level.quantity))
            return result
    
    def get_asks(self, depth: int = 10) -> List[Tuple[Decimal, Decimal]]:
        """Get top N ask levels."""
        with self._lock:
            result = []
            for i in range(min(depth, len(self._asks))):
                level = self._asks.peekitem(index=i)[1]
                result.append((level.price, level.quantity))
            return result
    
    def get_volume_at_price(self, price: Decimal, side: str = 'bid') -> Decimal:
        """Get total volume at a specific price level."""
        with self._lock:
            if side == 'bid':
                neg_price = -float(price)
                level = self._bids.get(neg_price)
            else:
                pos_price = float(price)
                level = self._asks.get(pos_price)
            
            return level.quantity if level else Decimal('0')
    
    def get_cumulative_volume(self, side: str = 'bid', depth: int = 10) -> Decimal:
        """Get cumulative volume for top N levels."""
        with self._lock:
            total = Decimal('0')
            levels = self._bids if side == 'bid' else self._asks
            
            for i in range(min(depth, len(levels))):
                level = levels.peekitem(index=i)[1]
                total += level.quantity
            
            return total
    
    def get_order_book_state(self) -> Dict[str, Any]:
        """Get comprehensive order book state."""
        with self._lock:
            return {
                'symbol': self.symbol,
                'state': self._state.value,
                'last_update_id': self._last_update_id,
                'timestamp': self._timestamp,
                'bid_count': len(self._bids),
                'ask_count': len(self._asks),
                'update_count': self._update_count,
                'last_message_age_ms': int((time.time() - self._last_message_time) * 1000),
                'best_bid': str(self.get_best_bid().price) if self.get_best_bid() else None,
                'best_ask': str(self.get_best_ask().price) if self.get_best_ask() else None,
                'mid_price': str(self.get_mid_price()) if self.get_mid_price() else None,
                'spread_bps': self.get_spread_bps(),
            }
    
    def is_stale(self, max_age_ms: int = 5000) -> bool:
        """Check if order book is stale (no updates for specified time)."""
        age_ms = (time.time() - self._last_message_time) * 1000
        return age_ms > max_age_ms or self._state == OrderBookState.STALE


class OrderBookManager:
    """
    Manages multiple order books for different symbols.
    
    Provides centralized access to all order books with thread-safe operations.
    Handles snapshot initialization and delta update routing.
    """
    
    def __init__(self, symbols: List[str], max_depth: int = 1000):
        """
        Initialize order book manager.
        
        Args:
            symbols: List of trading pairs to track
            max_depth: Maximum depth per order book
        """
        self.symbols = symbols
        self.max_depth = max_depth
        
        self._books: Dict[str, LocalOrderBook] = {}
        self._lock = threading.RLock()
        
        # Initialize books for all symbols
        for symbol in symbols:
            self._books[symbol.upper()] = LocalOrderBook(symbol.upper(), max_depth)
        
        logger.info(f"Order book manager initialized for {len(symbols)} symbols: {symbols}")
    
    def get_book(self, symbol: str) -> Optional[LocalOrderBook]:
        """Get order book for a symbol."""
        return self._books.get(symbol.upper())
    
    def get_all_books(self) -> Dict[str, LocalOrderBook]:
        """Get all order books."""
        return self._books.copy()
    
    def update_snapshot(self, symbol: str, snapshot: OrderBookSnapshot):
        """Update snapshot for a symbol."""
        book = self.get_book(symbol)
        if book:
            book.set_snapshot(snapshot)
    
    def apply_delta(self, 
                    symbol: str, 
                    bids: List[Tuple[str, str]], 
                    asks: List[Tuple[str, str]],
                    update_id: int,
                    timestamp: int) -> bool:
        """Apply delta update to a symbol's order book."""
        book = self.get_book(symbol)
        if book:
            return book.apply_delta(bids, asks, update_id, timestamp)
        return False
    
    def get_best_prices(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Get best bid/ask for all symbols."""
        result = {}
        for symbol, book in self._books.items():
            best_bid = book.get_best_bid()
            best_ask = book.get_best_ask()
            result[symbol] = {
                'bid': str(best_bid.price) if best_bid else None,
                'ask': str(best_ask.price) if best_ask else None,
            }
        return result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all order books."""
        status = {
            'total_symbols': len(self._books),
            'ready_count': 0,
            'stale_count': 0,
            'books': {}
        }
        
        for symbol, book in self._books.items():
            book_state = book.get_order_book_state()
            status['books'][symbol] = book_state
            
            if book.state == OrderBookState.READY:
                status['ready_count'] += 1
            elif book.state == OrderBookState.STALE:
                status['stale_count'] += 1
        
        return status


async def main():
    """Example usage of order book manager."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    manager = OrderBookManager(symbols, max_depth=500)
    
    # Simulate snapshot
    snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=[(Decimal("50000"), Decimal("1.5")), (Decimal("49999"), Decimal("2.0"))],
        asks=[(Decimal("50001"), Decimal("1.0")), (Decimal("50002"), Decimal("2.5"))],
        timestamp=int(time.time() * 1000),
        last_update_id=1000
    )
    
    manager.update_snapshot("BTCUSDT", snapshot)
    
    # Get book state
    book = manager.get_book("BTCUSDT")
    if book:
        print(f"BTCUSDT State: {book.get_order_book_state()}")
        print(f"Best Bid: {book.get_best_bid()}")
        print(f"Best Ask: {book.get_best_ask()}")
        print(f"Mid Price: {book.get_mid_price()}")
        print(f"Spread: {book.get_spread()} bps")


if __name__ == "__main__":
    asyncio.run(main())
