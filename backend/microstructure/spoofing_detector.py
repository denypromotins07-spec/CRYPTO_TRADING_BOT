#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
Spoofing Detection Module
Identifies phantom orders that vanish before fills (cancelled within 5ms)
Optimized for 8GB RAM with strict type hinting
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import deque
import time
import numpy as np


class SpoofType(Enum):
    """Types of spoofing patterns detected."""
    RAPID_CANCEL = "rapid_cancel"  # Order cancelled within 5ms
    LAYERING = "layering"  # Multiple orders at different levels
    MOMENTUM_IGNITION = "momentum_ignition"  # Fake orders to trigger movement
    WASH_TRADING = "wash_trading"  # Self-matching attempts
    PHANTOM_WALL = "phantom_wall"  # Large order that disappears


@dataclass(slots=True)
class OrderEvent:
    """
    Represents an order lifecycle event.
    Uses __slots__ for memory efficiency.
    """
    order_id: str
    timestamp_ns: int
    price: float
    quantity: float
    side: str  # 'bid' or 'ask'
    event_type: str  # 'new', 'cancel', 'fill', 'modify'
    exchange_order_id: Optional[str] = None
    
    @property
    def age_ms(self) -> float:
        """Calculate order age in milliseconds since placement."""
        return (time.time_ns() - self.timestamp_ns) / 1_000_000.0


@dataclass(slots=True)
class SpoofingAlert:
    """Alert generated when spoofing is detected."""
    timestamp_ns: int
    spoof_type: SpoofType
    confidence: float  # 0.0 to 1.0
    order_ids: List[str]
    price_levels: List[float]
    total_volume: float
    description: str
    action_taken: str
    missed_fill_prevented: bool = False
    adverse_selection_avoided: bool = False


class SpoofingDetector:
    """
    Detects spoofing and manipulative trading patterns.
    
    Key detection criteria:
    - Orders cancelled within 5ms of placement (Chapter 2 requirement)
    - Multi-level layering patterns
    - Phantom liquidity walls
    - Momentum ignition attempts
    
    Uses Observer pattern for alert distribution.
    """
    
    # Threshold for rapid cancellation detection (5ms as per requirements)
    RAPID_CANCEL_THRESHOLD_MS = 5.0
    
    # Minimum volume ratio for wall detection
    WALL_VOLUME_RATIO = 10.0
    
    # Lookback window for pattern detection
    LOOKBACK_WINDOW_MS = 1000.0
    
    def __init__(self, max_events: int = 10000, max_alerts: int = 100):
        """
        Initialize spoofing detector.
        
        Args:
            max_events: Maximum order events to track (memory constraint)
            max_alerts: Maximum alerts to retain in history
        """
        # Active orders awaiting completion
        self.active_orders: Dict[str, OrderEvent] = {}
        
        # Event history for pattern analysis
        self.event_history: deque[OrderEvent] = deque(maxlen=max_events)
        
        # Generated alerts
        self.alerts: deque[SpoofingAlert] = deque(maxlen=max_alerts)
        
        # Price level tracking for layering detection
        self.price_level_events: Dict[float, List[OrderEvent]] = {}
        
        # Statistics for calibration
        self.stats = {
            'total_orders': 0,
            'spoofed_orders': 0,
            'rapid_cancels': 0,
            'layering_detected': 0,
            'phantom_walls': 0,
        }
        
        # Observer callbacks
        self._observers: List[callable] = []
        
        # Last cleanup timestamp
        self.last_cleanup_ns: int = time.time_ns()
        
    def register_observer(self, callback: callable) -> None:
        """Register an observer for spoofing alerts (Observer pattern)."""
        self._observers.append(callback)
    
    def remove_observer(self, callback: callable) -> None:
        """Remove an observer."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self, alert: SpoofingAlert) -> None:
        """Notify all registered observers of a new alert."""
        for observer in self._observers:
            try:
                observer(alert)
            except Exception:
                pass  # Don't let observer errors break detection
    
    def process_new_order(self, order_id: str, price: float, quantity: float,
                          side: str, exchange_order_id: Optional[str] = None) -> None:
        """
        Process a new order event.
        
        Args:
            order_id: Internal order identifier
            price: Order price
            quantity: Order quantity
            side: 'bid' or 'ask'
            exchange_order_id: Exchange-provided order ID
        """
        current_ns = time.time_ns()
        
        event = OrderEvent(
            order_id=order_id,
            timestamp_ns=current_ns,
            price=price,
            quantity=quantity,
            side=side,
            event_type='new',
            exchange_order_id=exchange_order_id
        )
        
        self.active_orders[order_id] = event
        self.event_history.append(event)
        self.stats['total_orders'] += 1
        
        # Track by price level for layering detection
        if price not in self.price_level_events:
            self.price_level_events[price] = []
        self.price_level_events[price].append(event)
        
        # Periodic cleanup
        self._maybe_cleanup()
    
    def process_cancellation(self, order_id: str) -> Optional[SpoofingAlert]:
        """
        Process an order cancellation and check for spoofing.
        
        Args:
            order_id: Order identifier to cancel
            
        Returns:
            SpoofingAlert if spoofing detected, None otherwise
        """
        current_ns = time.time_ns()
        
        if order_id not in self.active_orders:
            return None
        
        order_event = self.active_orders.pop(order_id)
        
        # Calculate order age
        age_ns = current_ns - order_event.timestamp_ns
        age_ms = age_ns / 1_000_000.0
        
        # Create cancellation event
        cancel_event = OrderEvent(
            order_id=order_id,
            timestamp_ns=current_ns,
            price=order_event.price,
            quantity=order_event.quantity,
            side=order_event.side,
            event_type='cancel'
        )
        self.event_history.append(cancel_event)
        
        # Check for rapid cancellation (primary spoofing indicator)
        if age_ms <= self.RAPID_CANCEL_THRESHOLD_MS:
            self.stats['rapid_cancels'] += 1
            self.stats['spoofed_orders'] += 1
            
            alert = SpoofingAlert(
                timestamp_ns=current_ns,
                spoof_type=SpoofType.RAPID_CANCEL,
                confidence=self._calculate_rapid_cancel_confidence(age_ms),
                order_ids=[order_id],
                price_levels=[order_event.price],
                total_volume=order_event.quantity,
                description=f"Order cancelled within {age_ms:.2f}ms (threshold: {self.RAPID_CANCEL_THRESHOLD_MS}ms)",
                action_taken="IGNORED_FROM_BOOK",
                missed_fill_prevented=False,
                adverse_selection_avoided=True
            )
            
            self._record_alert(alert)
            return alert
        
        return None
    
    def process_fill(self, order_id: str, filled_quantity: float) -> None:
        """
        Process an order fill event.
        
        Args:
            order_id: Order identifier
            filled_quantity: Quantity filled
        """
        current_ns = time.time_ns()
        
        if order_id in self.active_orders:
            del self.active_orders[order_id]
        
        fill_event = OrderEvent(
            order_id=order_id,
            timestamp_ns=current_ns,
            price=0.0,  # Fill price would come from execution report
            quantity=filled_quantity,
            side='unknown',
            event_type='fill'
        )
        self.event_history.append(fill_event)
    
    def detect_layering(self, side: str, min_levels: int = 3) -> Optional[SpoofingAlert]:
        """
        Detect layering patterns (multiple orders at consecutive price levels).
        
        Args:
            side: 'bid' or 'ask'
            min_levels: Minimum number of levels for layering pattern
            
        Returns:
            SpoofingAlert if layering detected
        """
        current_ns = time.time_ns()
        
        # Get recent active orders on this side
        recent_orders = [
            e for e in self.active_orders.values()
            if e.side == side and (current_ns - e.timestamp_ns) / 1_000_000.0 < self.LOOKBACK_WINDOW_MS
        ]
        
        if len(recent_orders) < min_levels:
            return None
        
        # Group by price level
        price_groups: Dict[float, float] = {}
        for order in recent_orders:
            price_groups[order.price] = price_groups.get(order.price, 0.0) + order.quantity
        
        # Check for consecutive price levels
        sorted_prices = sorted(price_groups.keys())
        consecutive_count = 1
        max_consecutive = 1
        start_idx = 0
        
        for i in range(1, len(sorted_prices)):
            price_diff = abs(sorted_prices[i] - sorted_prices[i-1])
            if price_diff <= 0.05:  # Within 5 cents (adjust for asset)
                consecutive_count += 1
                if consecutive_count > max_consecutive:
                    max_consecutive = consecutive_count
                    start_idx = i - consecutive_count + 1
            else:
                consecutive_count = 1
        
        if max_consecutive >= min_levels:
            layer_prices = sorted_prices[start_idx:start_idx + max_consecutive]
            layer_volume = sum(price_groups[p] for p in layer_prices)
            
            self.stats['layering_detected'] += 1
            
            alert = SpoofingAlert(
                timestamp_ns=current_ns,
                spoof_type=SpoofType.LAYERING,
                confidence=min(1.0, max_consecutive / 10.0),
                order_ids=[o.order_id for o in recent_orders if o.price in layer_prices],
                price_levels=layer_prices,
                total_volume=layer_volume,
                description=f"Layering detected: {max_consecutive} consecutive levels on {side}",
                action_taken="FLAGGED_FOR_REVIEW",
                missed_fill_prevented=False,
                adverse_selection_avoided=False
            )
            
            self._record_alert(alert)
            return alert
        
        return None
    
    def detect_phantom_wall(self, price: float, side: str, 
                            min_volume_ratio: float = None) -> Optional[SpoofingAlert]:
        """
        Detect phantom liquidity walls (large orders that disappear).
        
        Args:
            price: Price level to check
            side: 'bid' or 'ask'
            min_volume_ratio: Minimum ratio vs average volume
            
        Returns:
            SpoofingAlert if phantom wall detected
        """
        if min_volume_ratio is None:
            min_volume_ratio = self.WALL_VOLUME_RATIO
        
        current_ns = time.time_ns()
        
        # Calculate average volume at this price level
        level_orders = [
            e for e in self.event_history
            if abs(e.price - price) < 0.01 and e.side == side
        ]
        
        if len(level_orders) < 5:
            return None
        
        recent_volumes = [
            e.quantity for e in level_orders
            if (current_ns - e.timestamp_ns) / 1_000_000.0 < self.LOOKBACK_WINDOW_MS
        ]
        
        if not recent_volumes:
            return None
        
        avg_volume = np.mean(recent_volumes)
        max_volume = max(recent_volumes)
        
        if max_volume > avg_volume * min_volume_ratio:
            # Check if the large order was cancelled
            large_orders = [e for e in level_orders if e.quantity == max_volume]
            cancelled_large = any(
                e.event_type == 'cancel' for e in large_orders
            )
            
            if cancelled_large:
                self.stats['phantom_walls'] += 1
                
                alert = SpoofingAlert(
                    timestamp_ns=current_ns,
                    spoof_type=SpoofType.PHANTOM_WALL,
                    confidence=min(1.0, max_volume / (avg_volume * min_volume_ratio)),
                    order_ids=[e.order_id for e in large_orders],
                    price_levels=[price],
                    total_volume=max_volume,
                    description=f"Phantom wall detected: {max_volume:.4f} vs avg {avg_volume:.4f}",
                    action_taken="IGNORED_LIQUIDITY",
                    missed_fill_prevented=True,
                    adverse_selection_avoided=True
                )
                
                self._record_alert(alert)
                return alert
        
        return None
    
    def get_spoofing_rate(self) -> float:
        """Get the current spoofing rate (spoofed orders / total orders)."""
        if self.stats['total_orders'] == 0:
            return 0.0
        return self.stats['spoofed_orders'] / self.stats['total_orders']
    
    def get_recent_alerts(self, count: int = 10) -> List[SpoofingAlert]:
        """Get the most recent spoofing alerts."""
        return list(self.alerts)[-count:]
    
    def clear(self) -> None:
        """Clear all tracked data."""
        self.active_orders.clear()
        self.event_history.clear()
        self.alerts.clear()
        self.price_level_events.clear()
        self.stats = {k: 0 for k in self.stats}
    
    def _calculate_rapid_cancel_confidence(self, age_ms: float) -> float:
        """Calculate confidence score for rapid cancellation spoofing."""
        if age_ms <= 1.0:
            return 1.0
        elif age_ms >= self.RAPID_CANCEL_THRESHOLD_MS:
            return 0.0
        else:
            # Linear interpolation
            return 1.0 - (age_ms / self.RAPID_CANCEL_THRESHOLD_MS)
    
    def _record_alert(self, alert: SpoofingAlert) -> None:
        """Record an alert and notify observers."""
        self.alerts.append(alert)
        self._notify_observers(alert)
    
    def _maybe_cleanup(self) -> None:
        """Periodically clean up old data (memory management)."""
        current_ns = time.time_ns()
        
        # Cleanup every 10 seconds
        if current_ns - self.last_cleanup_ns > 10_000_000_000:
            cutoff_ns = current_ns - int(self.LOOKBACK_WINDOW_MS * 1_000_000)
            
            # Clean old price level events
            for price in list(self.price_level_events.keys()):
                self.price_level_events[price] = [
                    e for e in self.price_level_events[price]
                    if e.timestamp_ns > cutoff_ns
                ]
                if not self.price_level_events[price]:
                    del self.price_level_events[price]
            
            self.last_cleanup_ns = current_ns


def differentiate_genuine_market_making(detector: SpoofingDetector, 
                                         order_id: str) -> bool:
    """
    Differentiate between genuine market making and malicious layering.
    
    Genuine market makers typically:
    - Keep orders alive longer (>100ms)
    - Have balanced bid/ask presence
    - Don't rapidly cancel and replace
    
    Args:
        detector: SpoofingDetector instance
        order_id: Order to evaluate
        
    Returns:
        True if likely genuine market making, False if potentially malicious
    """
    if order_id not in detector.active_orders:
        # Check historical data
        for event in reversed(detector.event_history):
            if event.order_id == order_id:
                age_ms = (time.time_ns() - event.timestamp_ns) / 1_000_000.0
                return age_ms > 100.0  # Genuine MM keeps orders longer
        return False
    
    order = detector.active_orders[order_id]
    age_ms = order.age_ms
    
    # Genuine market makers keep orders alive
    if age_ms < 50.0:
        return False
    
    # Check for balanced presence (simplified heuristic)
    bid_count = sum(1 for o in detector.active_orders.values() if o.side == 'bid')
    ask_count = sum(1 for o in detector.active_orders.values() if o.side == 'ask')
    
    if bid_count > 0 and ask_count > 0:
        balance_ratio = min(bid_count, ask_count) / max(bid_count, ask_count)
        if balance_ratio > 0.5:  # Relatively balanced
            return True
    
    return False


if __name__ == "__main__":
    # Example usage and testing
    detector = SpoofingDetector()
    
    # Simulate rapid cancellation (spoofing)
    detector.process_new_order("ORD001", 50000.0, 1.5, "bid")
    time.sleep(0.003)  # 3ms
    alert = detector.process_cancellation("ORD001")
    
    if alert:
        print(f"Spoofing Detected: {alert.spoof_type.value}")
        print(f"Confidence: {alert.confidence:.2%}")
        print(f"Description: {alert.description}")
        print(f"Action: {alert.action_taken}")
    
    # Test layering detection
    for i in range(5):
        detector.process_new_order(f"ORD{i+10}", 50000.0 - i * 0.01, 2.0, "bid")
    
    layering_alert = detector.detect_layering("bid")
    if layering_alert:
        print(f"\nLayering Detected: {layering_alert.description}")
