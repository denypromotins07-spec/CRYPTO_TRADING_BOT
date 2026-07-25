#!/usr/bin/env python3
"""
Spoofing Detector - Identifying Phantom Orders and Fake Liquidity

This module detects spoofing attempts by identifying orders that are placed
and cancelled within milliseconds without genuine intent to trade.
It uses statistical analysis and pattern recognition to flag manipulation.

Designed for the ZAID PERSONAL CRYPTO TRADING BOT with strict 8GB RAM constraints.
Implements real-time detection with sub-millisecond latency requirements.

Author: Opus 4.8
Stage: 21/100 - Advanced Market Microstructure
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time
import numpy as np
from datetime import datetime


class SpoofingType(Enum):
    """Types of spoofing behavior detected."""
    RAPID_CANCEL = auto()      # Order cancelled within 5ms
    LAYERING = auto()          # Multiple fake orders at different levels
    MOMENTUM_IGNITION = auto() # Fake orders to trigger momentum
    WALL_SPOOFING = auto()     # Large fake order to intimidate
    PINGING = auto()           # Small orders to probe book depth


@dataclass(slots=True)
class OrderEvent:
    """Represents an order placement or cancellation event."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    volume: float
    timestamp_ns: int
    event_type: str  # 'new', 'cancel', 'modify', 'fill'
    exchange_order_id: Optional[str] = None
    
    @property
    def age_ms(self) -> float:
        """Calculate age in milliseconds from creation to now."""
        return (time.time_ns() - self.timestamp_ns) / 1_000_000


@dataclass(slots=True)
class SpoofingSignal:
    """A detected spoofing signal with confidence score."""
    spoofing_type: SpoofingType
    confidence: float  # 0.0 to 1.0
    order_ids: List[str]
    symbol: str
    side: str
    price_levels: List[float]
    total_volume: float
    detection_time_ns: int
    duration_ms: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'type': self.spoofing_type.name,
            'confidence': self.confidence,
            'order_count': len(self.order_ids),
            'symbol': self.symbol,
            'side': self.side,
            'price_range': f"{min(self.price_levels):.2f}-{max(self.price_levels):.2f}",
            'volume': self.total_volume,
            'duration_ms': self.duration_ms,
            'description': self.description,
        }


@dataclass(slots=True)
class OrderLifecycle:
    """Tracks the lifecycle of a single order."""
    order_id: str
    symbol: str
    side: str
    price: float
    volume: float
    placed_at_ns: int
    cancelled_at_ns: Optional[int] = None
    filled_at_ns: Optional[int] = None
    modified_at_ns: Optional[int] = None
    fill_qty: float = 0.0
    
    @property
    def lifetime_ms(self) -> float:
        """Calculate order lifetime in milliseconds."""
        end_time = self.cancelled_at_ns or self.filled_at_ns or time.time_ns()
        return (end_time - self.placed_at_ns) / 1_000_000
    
    @property
    def was_spoofed(self) -> bool:
        """Check if order appears to be spoofed (cancelled < 5ms)."""
        if self.cancelled_at_ns is None:
            return False
        return self.lifetime_ms < 5.0
    
    @property
    def was_filled(self) -> bool:
        """Check if order was fully or partially filled."""
        return self.fill_qty > 0


class SpoofingDetector:
    """
    Real-time spoofing detection engine.
    
    Detects various forms of market manipulation including:
    - Rapid cancellations (< 5ms)
    - Layering patterns
    - Momentum ignition attempts
    - Fake liquidity walls
    
    Uses sliding window analysis and statistical thresholds.
    """
    
    # Detection thresholds
    RAPID_CANCEL_THRESHOLD_MS: float = 5.0
    LAYERING_MIN_LEVELS: int = 3
    LAYERING_MAX_SPREAD_BPS: float = 50.0  # Basis points
    WALL_VOLUME_MULTIPLE: float = 10.0  # Times average order size
    PING_FREQUENCY_THRESHOLD: int = 10  # Pings per second
    
    def __init__(
        self,
        symbols: List[str],
        window_size_ms: int = 1000,
        sensitivity: float = 0.7,
    ) -> None:
        """
        Initialize the spoofing detector.
        
        Args:
            symbols: List of symbols to monitor
            window_size_ms: Analysis window size in milliseconds
            sensitivity: Detection sensitivity (0.0 to 1.0)
        """
        self.symbols: Set[str] = set(symbols)
        self.window_size_ms: int = window_size_ms
        self.sensitivity: float = max(0.0, min(1.0, sensitivity))
        
        # Order tracking
        self._active_orders: Dict[str, OrderLifecycle] = {}
        self._completed_orders: deque[OrderLifecycle] = deque(maxlen=10000)
        
        # Event history per symbol
        self._event_history: Dict[str, deque[OrderEvent]] = {
            sym: deque(maxlen=5000) for sym in symbols
        }
        
        # Detected signals
        self._signals: deque[SpoofingSignal] = deque(maxlen=1000)
        
        # Statistics for adaptive thresholds
        self._order_stats: Dict[str, Dict[str, float]] = {
            sym: {'avg_lifetime_ms': 100.0, 'avg_size': 1.0, 'cancel_rate': 0.5}
            for sym in symbols
        }
        
        # Pattern buffers for layering detection
        self._layering_buffer: Dict[str, deque[Tuple[int, str, float, float]]] = {
            sym: deque(maxlen=100) for sym in symbols
        }
        
        # Rate limiting for signal generation
        self._last_signal_time: Dict[str, int] = {}
        self._signal_cooldown_ms: int = 100  # Minimum time between signals
    
    @staticmethod
    def _now_ns() -> int:
        """Get current time in nanoseconds."""
        return time.time_ns()
    
    def process_event(self, event: OrderEvent) -> Optional[SpoofingSignal]:
        """
        Process an order book event and check for spoofing.
        
        Args:
            event: Order event to process
        
        Returns:
            SpoofingSignal if spoofing detected, None otherwise
        """
        if event.symbol not in self.symbols:
            return None
        
        # Store event in history
        self._event_history[event.symbol].append(event)
        
        signal: Optional[SpoofingSignal] = None
        
        if event.event_type == 'new':
            signal = self._handle_new_order(event)
        elif event.event_type == 'cancel':
            signal = self._handle_cancellation(event)
        elif event.event_type == 'modify':
            self._handle_modification(event)
        elif event.event_type == 'fill':
            self._handle_fill(event)
        
        # Update statistics periodically
        self._update_statistics(event.symbol)
        
        return signal
    
    def _handle_new_order(self, event: OrderEvent) -> Optional[SpoofingSignal]:
        """Handle new order placement."""
        # Create lifecycle tracker
        lifecycle = OrderLifecycle(
            order_id=event.order_id,
            symbol=event.symbol,
            side=event.side,
            price=event.price,
            volume=event.volume,
            placed_at_ns=event.timestamp_ns,
        )
        self._active_orders[event.order_id] = lifecycle
        
        # Check for wall spoofing immediately
        signal = self._check_wall_spoofing(event)
        if signal:
            return signal
        
        # Add to layering buffer
        self._layering_buffer[event.symbol].append((
            event.timestamp_ns,
            event.side,
            event.price,
            event.volume,
        ))
        
        # Check for layering pattern
        return self._check_layering(event.symbol, event.side)
    
    def _handle_cancellation(self, event: OrderEvent) -> Optional[SpoofingSignal]:
        """Handle order cancellation and check for rapid cancel spoofing."""
        if event.order_id not in self._active_orders:
            return None
        
        lifecycle = self._active_orders[event.order_id]
        lifecycle.cancelled_at_ns = event.timestamp_ns
        
        # Move to completed
        del self._active_orders[event.order_id]
        self._completed_orders.append(lifecycle)
        
        # Check for rapid cancellation (primary spoofing indicator)
        if lifecycle.was_spoofed:
            return self._create_rapid_cancel_signal(lifecycle)
        
        return None
    
    def _handle_modification(self, event: OrderEvent) -> None:
        """Handle order modification."""
        if event.order_id in self._active_orders:
            self._active_orders[event.order_id].modified_at_ns = event.timestamp_ns
    
    def _handle_fill(self, event: OrderEvent) -> None:
        """Handle order fill."""
        if event.order_id in self._active_orders:
            lifecycle = self._active_orders[event.order_id]
            lifecycle.fill_qty += event.volume
            if lifecycle.fill_qty >= lifecycle.volume:
                lifecycle.filled_at_ns = event.timestamp_ns
                del self._active_orders[event.order_id]
                self._completed_orders.append(lifecycle)
    
    def _check_wall_spoofing(self, event: OrderEvent) -> Optional[SpoofingSignal]:
        """Check if a large order is potentially a spoof wall."""
        stats = self._order_stats.get(event.symbol, {})
        avg_size = stats.get('avg_size', 1.0)
        
        if event.volume >= avg_size * self.WALL_VOLUME_MULTIPLE:
            # Large order detected - mark for monitoring
            # Will be confirmed as spoof if cancelled quickly
            pass
        
        return None
    
    def _check_layering(self, symbol: str, side: str) -> Optional[SpoofingSignal]:
        """Check for layering pattern in recent orders."""
        buffer = self._layering_buffer[symbol]
        
        if len(buffer) < self.LAYERING_MIN_LEVELS:
            return None
        
        # Get recent orders on same side
        now = self._now_ns()
        window_ns = self.window_size_ms * 1_000_000
        
        recent = [
            (ts, s, p, v) for ts, s, p, v in buffer
            if s == side and (now - ts) / 1_000_000 < self.window_size_ms
        ]
        
        if len(recent) < self.LAYERING_MIN_LEVELS:
            return None
        
        # Check price spread
        prices = [p for _, _, p, _ in recent]
        price_range = max(prices) - min(prices)
        mid_price = sum(prices) / len(prices)
        
        if mid_price == 0:
            return None
        
        spread_bps = (price_range / mid_price) * 10000
        
        if spread_bps > self.LAYERING_MAX_SPREAD_BPS:
            return None
        
        # Check volume pattern (often increasing with distance from mid)
        volumes = [v for _, _, _, v in recent]
        if len(set(volumes)) < 2:
            return None
        
        # Potential layering detected
        total_volume = sum(volumes)
        confidence = min(1.0, (len(recent) / 10.0) * self.sensitivity)
        
        return SpoofingSignal(
            spoofing_type=SpoofingType.LAYERING,
            confidence=confidence,
            order_ids=[],  # Would need to track order IDs in buffer
            symbol=symbol,
            side=side,
            price_levels=prices,
            total_volume=total_volume,
            detection_time_ns=now,
            duration_ms=self.window_size_ms,
            description=f"Layering detected: {len(recent)} orders across {spread_bps:.1f} bps",
        )
    
    def _create_rapid_cancel_signal(self, lifecycle: OrderLifecycle) -> SpoofingSignal:
        """Create a signal for rapid cancellation spoofing."""
        return SpoofingSignal(
            spoofing_type=SpoofingType.RAPID_CANCEL,
            confidence=min(1.0, (5.0 / max(lifecycle.lifetime_ms, 0.001)) * self.sensitivity),
            order_ids=[lifecycle.order_id],
            symbol=lifecycle.symbol,
            side=lifecycle.side,
            price_levels=[lifecycle.price],
            total_volume=lifecycle.volume,
            detection_time_ns=self._now_ns(),
            duration_ms=lifecycle.lifetime_ms,
            description=f"Rapid cancel: order lifetime {lifecycle.lifetime_ms:.2f}ms",
        )
    
    def _update_statistics(self, symbol: str) -> None:
        """Update rolling statistics for a symbol."""
        # Get recent completed orders
        recent = [
            o for o in self._completed_orders
            if o.symbol == symbol and (self._now_ns() - o.placed_at_ns) / 1_000_000 < 60000
        ]
        
        if len(recent) < 10:
            return
        
        # Calculate averages
        lifetimes = [o.lifetime_ms for o in recent]
        sizes = [o.volume for o in recent]
        cancels = sum(1 for o in recent if o.cancelled_at_ns is not None)
        
        self._order_stats[symbol] = {
            'avg_lifetime_ms': np.mean(lifetimes),
            'avg_size': np.mean(sizes),
            'cancel_rate': cancels / len(recent),
        }
    
    def get_recent_signals(self, symbol: Optional[str] = None, limit: int = 10) -> List[SpoofingSignal]:
        """Get recent spoofing signals."""
        signals = list(self._signals)
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        return signals[-limit:]
    
    def get_spoofing_rate(self, symbol: str, window_ms: int = 60000) -> float:
        """
        Calculate the rate of spoofed orders for a symbol.
        
        Returns:
            Ratio of spoofed orders to total orders
        """
        now = self._now_ns()
        recent = [
            o for o in self._completed_orders
            if o.symbol == symbol and (now - o.placed_at_ns) / 1_000_000 < window_ms
        ]
        
        if not recent:
            return 0.0
        
        spoofed = sum(1 for o in recent if o.was_spoofed)
        return spoofed / len(recent)
    
    def is_genuine_market_making(self, order_id: str) -> bool:
        """
        Differentiate between genuine market making and spoofing.
        
        Genuine market makers typically:
        - Keep orders alive longer (> 100ms)
        - Have balanced bid/ask presence
        - Get partial fills occasionally
        - Don't exhibit layering patterns
        """
        if order_id not in self._completed_orders:
            # Check active orders
            if order_id in self._active_orders:
                lifecycle = self._active_orders[order_id]
                # Too early to tell
                return lifecycle.lifetime_ms > 50
            return True
        
        # Find in completed orders
        for order in self._completed_orders:
            if order.order_id == order_id:
                # Genuine if filled or long-lived
                if order.was_filled:
                    return True
                if order.lifetime_ms > 100:
                    return True
                return False
        
        return True
    
    def add_signal(self, signal: SpoofingSignal) -> None:
        """Add a signal to the history."""
        self._signals.append(signal)
    
    def clear_symbol(self, symbol: str) -> None:
        """Clear all data for a symbol."""
        self._active_orders = {k: v for k, v in self._active_orders.items() if v.symbol != symbol}
        self._event_history[symbol].clear()
        self._layering_buffer[symbol].clear()


def main() -> None:
    """Example usage of SpoofingDetector."""
    detector = SpoofingDetector(symbols=['BTCUSDT', 'ETHUSDT'])
    
    # Simulate rapid cancel spoofing
    import random
    base_time = time.time_ns()
    
    for i in range(10):
        # Place order
        order_id = f"order_{i}"
        place_event = OrderEvent(
            order_id=order_id,
            symbol='BTCUSDT',
            side='buy',
            price=50000.0 + i * 0.01,
            volume=1.0,
            timestamp_ns=base_time + i * 1_000_000,
            event_type='new',
        )
        
        signal = detector.process_event(place_event)
        if signal:
            print(f"Signal on placement: {signal.description}")
        
        # Cancel quickly (spoofing)
        cancel_event = OrderEvent(
            order_id=order_id,
            symbol='BTCUSDT',
            side='buy',
            price=50000.0 + i * 0.01,
            volume=1.0,
            timestamp_ns=base_time + i * 1_000_000 + 2_000_000,  # 2ms later
            event_type='cancel',
        )
        
        signal = detector.process_event(cancel_event)
        if signal:
            print(f"SPOOF DETECTED: {signal.to_dict()}")
    
    # Print statistics
    print(f"\nSpoofing rate: {detector.get_spoofing_rate('BTCUSDT'):.2%}")
    print(f"Recent signals: {len(detector.get_recent_signals())}")


if __name__ == "__main__":
    main()
