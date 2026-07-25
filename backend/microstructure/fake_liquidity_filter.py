#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
Fake Liquidity Filter Module
Purges spoofed walls from the local order book
Optimized for 8GB RAM with strict type hinting
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import deque
import time
import numpy as np


class LiquidityType(Enum):
    """Classification of liquidity quality."""
    GENUINE = "genuine"
    SUSPICIOUS = "suspicious"
    SPOOFED = "spoofed"
    PHANTOM = "phantom"


@dataclass(slots=True)
class OrderBookLevel:
    """
    Represents a single price level in the filtered order book.
    Uses __slots__ for memory efficiency.
    """
    price: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    bid_order_count: int = 0
    ask_order_count: int = 0
    liquidity_type: LiquidityType = LiquidityType.GENUINE
    confidence_score: float = 1.0
    last_update_ns: int = 0
    # Tracking for spoofing detection
    rapid_cancellations: int = 0
    total_cancelled_volume: float = 0.0
    avg_order_lifetime_ms: float = 0.0


@dataclass(slots=True)
class FilteredOrderBook:
    """Complete filtered order book snapshot."""
    timestamp_ns: int
    levels: Dict[float, OrderBookLevel]
    best_bid: Optional[float]
    best_ask: Optional[float]
    filtered_volume_bid: float
    filtered_volume_ask: float
    spoofed_volume_removed: float
    liquidity_quality_score: float


class FakeLiquidityFilter:
    """
    Filters out spoofed and fake liquidity from the order book.
    
    Detection criteria:
    - Orders cancelled within 5ms of placement
    - Volume that disappears before execution
    - Abnormal cancellation patterns
    - Phantom walls (large orders that vanish)
    
    Implements Strategy pattern for different filtering strategies.
    """
    
    # Thresholds for spoofing detection
    RAPID_CANCEL_THRESHOLD_MS = 5.0
    WALL_VOLUME_THRESHOLD = 50.0  # Minimum volume to be considered a wall
    LIFETIME_THRESHOLD_MS = 100.0  # Genuine orders typically last >100ms
    
    def __init__(self, max_levels: int = 100, tick_size: float = 0.01):
        """
        Initialize fake liquidity filter.
        
        Args:
            max_levels: Maximum price levels to track
            tick_size: Price tick size for the asset
        """
        self.max_levels = max_levels
        self.tick_size = tick_size
        
        # Order book levels
        self.levels: Dict[float, OrderBookLevel] = {}
        
        # Active orders being tracked
        self.active_orders: Dict[str, Tuple[float, str, float, int]] = {}  # (price, side, volume, timestamp)
        
        # Historical data for pattern analysis
        self.order_history: deque = deque(maxlen=10000)
        self.cancellation_history: deque = deque(maxlen=10000)
        
        # Best bid/ask cache
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
        
        # Statistics
        self.stats = {
            'total_orders_tracked': 0,
            'spoofed_orders_detected': 0,
            'fake_volume_filtered': 0.0,
            'phantom_walls_removed': 0,
            'false_positives_corrected': 0,
        }
        
        # Filtering strategy (Strategy pattern)
        self.filtering_strategy: FilteringStrategy = AggressiveFilteringStrategy()
        
        # Last update timestamp
        self.last_update_ns: int = time.time_ns()
    
    def set_filtering_strategy(self, strategy: 'FilteringStrategy') -> None:
        """Set the filtering strategy (Strategy pattern)."""
        self.filtering_strategy = strategy
    
    def add_order(self, order_id: str, price: float, quantity: float, side: str) -> None:
        """
        Add an order to tracking.
        
        Args:
            order_id: Unique order identifier
            price: Order price
            quantity: Order quantity
            side: 'bid' or 'ask'
        """
        current_ns = time.time_ns()
        
        # Track active order
        self.active_orders[order_id] = (price, side, quantity, current_ns)
        self.stats['total_orders_tracked'] += 1
        
        # Update price level
        self._update_level(price, side, quantity, is_addition=True)
        
        self.last_update_ns = current_ns
    
    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """
        Process order cancellation and check for spoofing.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dict with spoofing info if detected, None otherwise
        """
        current_ns = time.time_ns()
        
        if order_id not in self.active_orders:
            return None
        
        price, side, quantity, placement_ns = self.active_orders.pop(order_id)
        
        # Calculate lifetime
        lifetime_ms = (current_ns - placement_ns) / 1_000_000.0
        
        # Record cancellation
        self.cancellation_history.append({
            'order_id': order_id,
            'price': price,
            'side': side,
            'quantity': quantity,
            'lifetime_ms': lifetime_ms,
            'timestamp_ns': current_ns,
        })
        
        # Check for spoofing
        spoofing_info = self.filtering_strategy.evaluate_cancellation(
            price, side, quantity, lifetime_ms, self.levels.get(price)
        )
        
        if spoofing_info and spoofing_info['is_spoofed']:
            self.stats['spoofed_orders_detected'] += 1
            self.stats['fake_volume_filtered'] += quantity
            
            # Mark level as suspicious/spoofed
            self._mark_level_spoofed(price, side, quantity)
            
            # Update level
            self._update_level(price, side, quantity, is_addition=False, apply_filter=True)
            
            return spoofing_info
        else:
            # Normal cancellation
            self._update_level(price, side, quantity, is_addition=False)
        
        return None
    
    def fill_order(self, order_id: str, filled_quantity: float) -> None:
        """
        Process order fill.
        
        Args:
            order_id: Order identifier
            filled_quantity: Quantity filled
        """
        if order_id in self.active_orders:
            price, side, _, _ = self.active_orders.pop(order_id)
            self._update_level(price, side, filled_quantity, is_addition=False)
    
    def get_filtered_book(self) -> FilteredOrderBook:
        """
        Get the current filtered order book.
        
        Returns:
            FilteredOrderBook with spoofed liquidity removed
        """
        current_ns = time.time_ns()
        
        # Calculate filtered volumes and best prices
        filtered_bid_vol = 0.0
        filtered_ask_vol = 0.0
        best_bid = None
        best_ask = None
        spoofed_removed = 0.0
        quality_scores = []
        
        for level in self.levels.values():
            # Apply filtering based on liquidity type
            if level.liquidity_type == LiquidityType.SPOOFED:
                spoofed_removed += level.bid_volume + level.ask_volume
                continue
            
            # Weight volume by confidence
            bid_factor = level.confidence_score if level.liquidity_type == LiquidityType.SUSPICIOUS else 1.0
            ask_factor = level.confidence_score if level.liquidity_type == LiquidityType.SUSPICIOUS else 1.0
            
            filtered_bid = level.bid_volume * bid_factor
            filtered_ask = level.ask_volume * ask_factor
            
            filtered_bid_vol += filtered_bid
            filtered_ask_vol += filtered_ask
            
            # Track best prices
            if filtered_bid > 0:
                if best_bid is None or level.price > best_bid:
                    best_bid = level.price
            
            if filtered_ask > 0:
                if best_ask is None or level.price < best_ask:
                    best_ask = level.price
            
            quality_scores.append(level.confidence_score)
        
        # Calculate overall quality score
        liquidity_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return FilteredOrderBook(
            timestamp_ns=current_ns,
            levels=self.levels.copy(),
            best_bid=best_bid,
            best_ask=best_ask,
            filtered_volume_bid=filtered_bid_vol,
            filtered_volume_ask=filtered_ask_vol,
            spoofed_volume_removed=spoofed_removed,
            liquidity_quality_score=liquidity_quality,
        )
    
    def detect_phantom_wall(self, price: float, side: str) -> bool:
        """
        Detect if a price level contains a phantom wall.
        
        Args:
            price: Price level to check
            side: 'bid' or 'ask'
            
        Returns:
            True if phantom wall detected
        """
        if price not in self.levels:
            return False
        
        level = self.levels[price]
        
        # Check for high cancellation rate
        if level.rapid_cancellations < 3:
            return False
        
        # Check if significant volume was cancelled
        if level.total_cancelled_volume < self.WALL_VOLUME_THRESHOLD:
            return False
        
        # Check average lifetime
        if level.avg_order_lifetime_ms > self.LIFETIME_THRESHOLD_MS:
            return False
        
        return True
    
    def purge_spoofed_levels(self) -> int:
        """
        Remove all spoofed liquidity from the book.
        
        Returns:
            Number of levels purged
        """
        purged_count = 0
        
        for price, level in list(self.levels.items()):
            if level.liquidity_type == LiquidityType.SPOOFED:
                level.bid_volume = 0.0
                level.ask_volume = 0.0
                level.bid_order_count = 0
                level.ask_order_count = 0
                purged_count += 1
        
        self._update_best_prices()
        return purged_count
    
    def get_genuine_liquidity(self, side: str, depth: int = 10) -> List[Tuple[float, float]]:
        """
        Get genuine liquidity levels (excluding spoofed).
        
        Args:
            side: 'bid' or 'ask'
            depth: Number of levels to return
            
        Returns:
            List of (price, volume) tuples
        """
        valid_levels = [
            (price, level) for price, level in self.levels.items()
            if level.liquidity_type != LiquidityType.SPOOFED
        ]
        
        if side == 'bid':
            valid_levels.sort(key=lambda x: x[0], reverse=True)
        else:
            valid_levels.sort(key=lambda x: x[0])
        
        result = []
        for price, level in valid_levels[:depth]:
            volume = level.bid_volume if side == 'bid' else level.ask_volume
            if volume > 0:
                result.append((price, volume * level.confidence_score))
        
        return result
    
    def get_stats(self) -> Dict:
        """Get filter statistics."""
        return self.stats.copy()
    
    def clear(self) -> None:
        """Clear all tracked data."""
        self.levels.clear()
        self.active_orders.clear()
        self.order_history.clear()
        self.cancellation_history.clear()
        self._best_bid = None
        self._best_ask = None
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}
    
    def _update_level(self, price: float, side: str, quantity: float,
                      is_addition: bool = True, apply_filter: bool = False) -> None:
        """Update a price level."""
        current_ns = time.time_ns()
        
        if price not in self.levels:
            if len(self.levels) >= self.max_levels:
                self._prune_farthest_level()
            self.levels[price] = OrderBookLevel(price=price)
        
        level = self.levels[price]
        
        if side == 'bid':
            if is_addition and not apply_filter:
                level.bid_volume += quantity
                level.bid_order_count += 1
            else:
                level.bid_volume = max(0.0, level.bid_volume - quantity)
                level.bid_order_count = max(0, level.bid_order_count - 1)
        else:
            if is_addition and not apply_filter:
                level.ask_volume += quantity
                level.ask_order_count += 1
            else:
                level.ask_volume = max(0.0, level.ask_volume - quantity)
                level.ask_order_count = max(0, level.ask_order_count - 1)
        
        level.last_update_ns = current_ns
        self._update_best_prices()
    
    def _mark_level_spoofed(self, price: float, side: str, quantity: float) -> None:
        """Mark a level as containing spoofed liquidity."""
        if price in self.levels:
            level = self.levels[price]
            level.rapid_cancellations += 1
            level.total_cancelled_volume += quantity
            level.liquidity_type = LiquidityType.SPOOFED
            level.confidence_score = max(0.0, level.confidence_score - 0.3)
    
    def _update_best_prices(self) -> None:
        """Update cached best bid and ask."""
        bid_prices = [p for p, l in self.levels.items() if l.bid_volume > 0 and l.liquidity_type != LiquidityType.SPOOFED]
        ask_prices = [p for p, l in self.levels.items() if l.ask_volume > 0 and l.liquidity_type != LiquidityType.SPOOFED]
        
        self._best_bid = max(bid_prices) if bid_prices else None
        self._best_ask = min(ask_prices) if ask_prices else None
    
    def _prune_farthest_level(self) -> None:
        """Remove the level farthest from mid (memory management)."""
        if not self.levels:
            return
        
        mid = self._calculate_mid()
        if mid is None:
            return
        
        farthest = max(self.levels.keys(), key=lambda p: abs(p - mid))
        del self.levels[farthest]
    
    def _calculate_mid(self) -> Optional[float]:
        """Calculate mid price."""
        if self._best_bid is not None and self._best_ask is not None:
            return (self._best_bid + self._best_ask) / 2.0
        return None


# Strategy Pattern for Filtering
class FilteringStrategy:
    """Base class for filtering strategies."""
    
    def evaluate_cancellation(self, price: float, side: str, quantity: float,
                              lifetime_ms: float, level: Optional[OrderBookLevel]) -> Optional[Dict]:
        """Evaluate if a cancellation indicates spoofing."""
        raise NotImplementedError


class AggressiveFilteringStrategy(FilteringStrategy):
    """Aggressive filtering - marks anything suspicious as spoofed."""
    
    def evaluate_cancellation(self, price: float, side: str, quantity: float,
                              lifetime_ms: float, level: Optional[OrderBookLevel]) -> Optional[Dict]:
        if lifetime_ms < FakeLiquidityFilter.RAPID_CANCEL_THRESHOLD_MS:
            return {
                'is_spoofed': True,
                'reason': 'rapid_cancel',
                'lifetime_ms': lifetime_ms,
                'confidence': 1.0,
            }
        elif lifetime_ms < 50.0 and quantity > FakeLiquidityFilter.WALL_VOLUME_THRESHOLD:
            return {
                'is_spoofed': True,
                'reason': 'short_lived_wall',
                'lifetime_ms': lifetime_ms,
                'confidence': 0.8,
            }
        return None


class ConservativeFilteringStrategy(FilteringStrategy):
    """Conservative filtering - only marks clear spoofing."""
    
    def evaluate_cancellation(self, price: float, side: str, quantity: float,
                              lifetime_ms: float, level: Optional[OrderBookLevel]) -> Optional[Dict]:
        # Only flag very rapid cancellations
        if lifetime_ms < 2.0:
            return {
                'is_spoofed': True,
                'reason': 'very_rapid_cancel',
                'lifetime_ms': lifetime_ms,
                'confidence': 0.95,
            }
        return None


class AdaptiveFilteringStrategy(FilteringStrategy):
    """Adaptive filtering - adjusts based on market conditions."""
    
    def __init__(self):
        self.spoofing_rate = 0.0
        self.recent_evaluations: deque = deque(maxlen=100)
    
    def evaluate_cancellation(self, price: float, side: str, quantity: float,
                              lifetime_ms: float, level: Optional[OrderBookLevel]) -> Optional[Dict]:
        # Adjust threshold based on recent spoofing rate
        dynamic_threshold = FakeLiquidityFilter.RAPID_CANCEL_THRESHOLD_MS * (1.0 - self.spoofing_rate)
        
        is_spoofed = lifetime_ms < dynamic_threshold
        
        result = None
        if is_spoofed:
            result = {
                'is_spoofed': True,
                'reason': 'adaptive_rapid_cancel',
                'lifetime_ms': lifetime_ms,
                'dynamic_threshold': dynamic_threshold,
                'confidence': 1.0 - (lifetime_ms / dynamic_threshold),
            }
        
        self.recent_evaluations.append(is_spoofed)
        self.spoofing_rate = sum(self.recent_evaluations) / len(self.recent_evaluations)
        
        return result


if __name__ == "__main__":
    # Example usage and testing
    filter_instance = FakeLiquidityFilter()
    
    # Add some orders
    filter_instance.add_order("ORD001", 50000.0, 10.0, "bid")
    filter_instance.add_order("ORD002", 49999.5, 5.0, "bid")
    filter_instance.add_order("ORD003", 50000.5, 8.0, "ask")
    
    # Simulate rapid cancellation (spoofing)
    time.sleep(0.003)  # 3ms
    spoofing_result = filter_instance.cancel_order("ORD001")
    
    if spoofing_result:
        print(f"Spoofing Detected: {spoofing_result['reason']}")
        print(f"Confidence: {spoofing_result['confidence']:.2%}")
    
    # Get filtered book
    filtered_book = filter_instance.get_filtered_book()
    print(f"\nFiltered Book:")
    print(f"Best Bid: {filtered_book.best_bid}")
    print(f"Best Ask: {filtered_book.best_ask}")
    print(f"Liquidity Quality: {filtered_book.liquidity_quality_score:.2%}")
    print(f"Spoofed Volume Removed: {filtered_book.spoofed_volume_removed}")
    
    # Get stats
    stats = filter_instance.get_stats()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
