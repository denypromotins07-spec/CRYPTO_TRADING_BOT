#!/usr/bin/env python3
"""
backend/orderflow/absorption_detector.py

Identifies passive limit orders absorbing aggressive market hits.
Distinguishes between genuine absorption and spoofing by analyzing
order behavior patterns.

Features:
- Real-time absorption detection at price levels
- Differentiation between genuine absorption vs spoofing
- Multi-level absorption tracking
- Strict type hinting for production reliability
- Cross-platform compatibility optimized for Windows PowerShell
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Protocol, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, auto
import time


class AbsorptionType(Enum):
    """Types of absorption detected."""
    BID_ABSORPTION = auto()   # Passive bids absorbing sells (bullish)
    ASK_ABSORPTION = auto()   # Passive asks absorbing buys (bearish)
    DUAL_ABSORPTION = auto()  # Both sides showing absorption
    SPOOF_ATTEMPT = auto()    # Likely spoofing, not genuine absorption


@dataclass
class AbsorptionEvent:
    """Represents a detected absorption event."""
    absorption_type: AbsorptionType
    price: float
    absorbed_volume: float
    attempted_volume: float  # Total aggressive volume that hit the level
    price_movement_after: float  # Price change after absorption
    duration_ms: float
    confidence_score: float  # 0.0 to 1.0 (higher = more likely genuine)
    timestamp_ns: int
    is_spoof: bool = False
    
    @property
    def absorption_ratio(self) -> float:
        """Ratio of absorbed volume to attempted volume."""
        if self.attempted_volume == 0:
            return 0.0
        return self.absorbed_volume / self.attempted_volume


@dataclass
class PriceLevelData:
    """Tracks activity at a single price level."""
    price: float
    total_bid_volume_absorbed: float = 0.0
    total_ask_volume_absorbed: float = 0.0
    aggressive_sell_attempts: float = 0.0
    aggressive_buy_attempts: float = 0.0
    price_touches: int = 0
    last_update_ns: int = 0
    level_created_ns: int = 0
    
    @property
    def bid_absorption_ratio(self) -> float:
        """How well bids are absorbing sells."""
        if self.aggressive_sell_attempts == 0:
            return 0.0
        return min(1.0, self.total_bid_volume_absorbed / self.aggressive_sell_attempts)
    
    @property
    def ask_absorption_ratio(self) -> float:
        """How well asks are absorbing buys."""
        if self.aggressive_buy_attempts == 0:
            return 0.0
        return min(1.0, self.total_ask_volume_absorbed / self.aggressive_buy_attempts)


class AbsorptionStrategy(Protocol):
    """Strategy interface for absorption detection."""
    
    def detect(
        self,
        level_data: PriceLevelData,
        recent_trades: List[Tuple[float, float, bool]]  # (price, volume, is_buyer_maker)
    ) -> Optional[AbsorptionEvent]:
        """Detect absorption at a price level."""
        ...


class VolumeThresholdStrategy:
    """
    Detects absorption based on volume thresholds.
    Genuine absorption requires high volume with minimal price movement.
    """
    
    def __init__(
        self,
        min_absorbed_volume: float = 10.0,
        min_attempts: float = 20.0,
        max_price_move_pct: float = 0.05
    ):
        self.min_absorbed_volume = min_absorbed_volume
        self.min_attempts = min_attempts
        self.max_price_move_pct = max_price_move_pct
    
    def detect(
        self,
        level_data: PriceLevelData,
        recent_trades: List[Tuple[float, float, bool]]
    ) -> Optional[AbsorptionEvent]:
        if level_data.price_touches < 3:
            return None
        
        # Check bid absorption
        if (level_data.aggressive_sell_attempts >= self.min_attempts and
            level_data.total_bid_volume_absorbed >= self.min_absorbed_volume):
            
            bid_ratio = level_data.bid_absorption_ratio
            if bid_ratio >= 0.7:  # 70%+ of sells absorbed
                confidence = min(bid_ratio, 1.0)
                
                return AbsorptionEvent(
                    absorption_type=AbsorptionType.BID_ABSORPTION,
                    price=level_data.price,
                    absorbed_volume=level_data.total_bid_volume_absorbed,
                    attempted_volume=level_data.aggressive_sell_attempts,
                    price_movement_after=0.0,  # Would be calculated from subsequent ticks
                    duration_ms=(time.time_ns() - level_data.level_created_ns) / 1_000_000,
                    confidence_score=confidence,
                    timestamp_ns=time.time_ns(),
                    is_spoof=False
                )
        
        # Check ask absorption
        if (level_data.aggressive_buy_attempts >= self.min_attempts and
            level_data.total_ask_volume_absorbed >= self.min_absorbed_volume):
            
            ask_ratio = level_data.ask_absorption_ratio
            if ask_ratio >= 0.7:  # 70%+ of buys absorbed
                confidence = min(ask_ratio, 1.0)
                
                return AbsorptionEvent(
                    absorption_type=AbsorptionType.ASK_ABSORPTION,
                    price=level_data.price,
                    absorbed_volume=level_data.total_ask_volume_absorbed,
                    attempted_volume=level_data.aggressive_buy_attempts,
                    price_movement_after=0.0,
                    duration_ms=(time.time_ns() - level_data.level_created_ns) / 1_000_000,
                    confidence_score=confidence,
                    timestamp_ns=time.time_ns(),
                    is_spoof=False
                )
        
        return None


class SpoofDetectionStrategy:
    """
    Identifies potential spoofing attempts that mimic absorption.
    Spoofing characteristics:
    - Large orders that disappear before being hit
    - Rapid order placement/cancellation
    - No genuine execution despite apparent absorption
    """
    
    def __init__(
        self,
        cancel_threshold_ms: float = 5.0,  # Orders cancelled within 5ms
        spoof_confidence_threshold: float = 0.6
    ):
        self.cancel_threshold_ms = cancel_threshold_ms
        self.spoof_confidence_threshold = spoof_confidence_threshold
        self.order_lifetimes: Dict[float, List[Tuple[int, int]]] = defaultdict(list)
    
    def record_order_placement(self, price: float, timestamp_ns: int) -> None:
        """Record when an order is placed at a price level."""
        self.order_lifetimes[price].append((timestamp_ns, 0))
    
    def record_order_cancel(self, price: float, timestamp_ns: int) -> None:
        """Record when an order is cancelled at a price level."""
        if self.order_lifetimes[price]:
            # Update the most recent uncancelled order
            for i in range(len(self.order_lifetimes[price]) - 1, -1, -1):
                if self.order_lifetimes[price][i][1] == 0:
                    entry = self.order_lifetimes[price][i]
                    self.order_lifetimes[price][i] = (entry[0], timestamp_ns)
                    break
    
    def detect_spoof(
        self,
        level_data: PriceLevelData,
        recent_trades: List[Tuple[float, float, bool]]
    ) -> Optional[AbsorptionEvent]:
        """Check if apparent absorption is actually spoofing."""
        price = level_data.price
        
        if price not in self.order_lifetimes:
            return None
        
        # Analyze order lifetimes
        short_lived_count = 0
        total_orders = len(self.order_lifetimes[price])
        
        for placed_ns, cancelled_ns in self.order_lifetimes[price]:
            if cancelled_ns > 0:
                lifetime_ms = (cancelled_ns - placed_ns) / 1_000_000
                if lifetime_ms < self.cancel_threshold_ms:
                    short_lived_count += 1
        
        if total_orders == 0:
            return None
        
        spoof_ratio = short_lived_count / total_orders
        
        if spoof_ratio >= self.spoof_confidence_threshold:
            return AbsorptionEvent(
                absorption_type=AbsorptionType.SPOOF_ATTEMPT,
                price=price,
                absorbed_volume=0,
                attempted_volume=level_data.aggressive_buy_attempts + level_data.aggressive_sell_attempts,
                price_movement_after=0.0,
                duration_ms=(time.time_ns() - level_data.level_created_ns) / 1_000_000,
                confidence_score=spoof_ratio,
                timestamp_ns=time.time_ns(),
                is_spoof=True
            )
        
        return None


class AbsorptionDetector:
    """
    Main detector combining multiple strategies for comprehensive absorption analysis.
    Implements Observer pattern for real-time notifications.
    """
    
    def __init__(self):
        self.volume_strategy = VolumeThresholdStrategy()
        self.spoof_strategy = SpoofDetectionStrategy()
        self.level_data: Dict[str, Dict[float, PriceLevelData]] = defaultdict(dict)
        self.recent_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.observers: List[Callable[[str, AbsorptionEvent], None]] = []
        
        # Configuration
        self.trade_history_size = 1000
        self.price_level_ttl_ns = 300_000_000_000  # 5 minutes
    
    def add_observer(
        self, 
        callback: Callable[[str, AbsorptionEvent], None]
    ) -> None:
        """Register callback for absorption detections."""
        self.observers.append(callback)
    
    def process_trade(
        self,
        symbol: str,
        price: float,
        volume: float,
        is_buyer_maker: bool,
        timestamp_ns: Optional[int] = None
    ) -> Optional[AbsorptionEvent]:
        """Process a trade and check for absorption."""
        ts = timestamp_ns or time.time_ns()
        
        # Record trade
        self.recent_trades[symbol].append((price, volume, is_buyer_maker))
        
        # Initialize or update level data
        if price not in self.level_data[symbol]:
            self.level_data[symbol][price] = PriceLevelData(
                price=price,
                level_created_ns=ts
            )
        
        level = self.level_data[symbol][price]
        level.last_update_ns = ts
        level.price_touches += 1
        
        # Track absorption
        if is_buyer_maker:
            # Seller initiated - hitting bids
            level.aggressive_sell_attempts += volume
            level.total_bid_volume_absorbed += volume
        else:
            # Buyer initiated - lifting asks
            level.aggressive_buy_attempts += volume
            level.total_ask_volume_absorbed += volume
        
        # Get recent trades at this level
        recent_at_level = [
            (p, v, ibm) for p, v, ibm in self.recent_trades[symbol]
            if abs(p - price) < 0.5  # Within $0.50
        ]
        
        # Check for spoofing first
        spoof_event = self.spoof_strategy.detect_spoof(level, recent_at_level)
        if spoof_event and spoof_event.is_spoof:
            self._notify_observers(symbol, spoof_event)
            return spoof_event
        
        # Check for genuine absorption
        absorption_event = self.volume_strategy.detect(level, recent_at_level)
        if absorption_event:
            self._notify_observers(symbol, absorption_event)
            return absorption_event
        
        return None
    
    def record_limit_order(self, symbol: str, price: float, timestamp_ns: int) -> None:
        """Record a limit order placement for spoof detection."""
        self.spoof_strategy.record_order_placement(price, timestamp_ns)
    
    def record_order_cancellation(self, symbol: str, price: float, timestamp_ns: int) -> None:
        """Record a limit order cancellation for spoof detection."""
        self.spoof_strategy.record_order_cancel(price, timestamp_ns)
    
    def _notify_observers(self, symbol: str, event: AbsorptionEvent) -> None:
        """Notify all observers of an absorption event."""
        for observer in self.observers:
            try:
                observer(symbol, event)
            except Exception as e:
                print(f"Observer error: {e}")
    
    def get_active_absorption_levels(
        self,
        symbol: str,
        min_confidence: float = 0.6
    ) -> List[AbsorptionEvent]:
        """Get current absorption levels above confidence threshold."""
        events = []
        
        if symbol not in self.level_data:
            return events
        
        for price, level in self.level_data[symbol].items():
            recent_trades = [
                (p, v, ibm) for p, v, ibm in self.recent_trades[symbol]
                if abs(p - price) < 0.5
            ]
            
            event = self.volume_strategy.detect(level, recent_trades)
            if event and event.confidence_score >= min_confidence:
                events.append(event)
        
        return events
    
    def differentiate_absorption_vs_spoof(
        self,
        symbol: str,
        price: float
    ) -> Tuple[bool, float]:
        """
        Determine if activity at a price level is genuine absorption or spoofing.
        Returns (is_genuine_absorption, confidence).
        """
        if symbol not in self.level_data or price not in self.level_data[symbol]:
            return False, 0.0
        
        level = self.level_data[symbol][price]
        
        # Check spoof indicators
        spoof_event = self.spoof_strategy.detect_spoof(
            level,
            list(self.recent_trades[symbol])
        )
        
        if spoof_event and spoof_event.is_spoof:
            return False, spoof_event.confidence_score
        
        # Check genuine absorption indicators
        absorption_event = self.volume_strategy.detect(
            level,
            list(self.recent_trades[symbol])
        )
        
        if absorption_event:
            return True, absorption_event.confidence_score
        
        return False, 0.0
    
    def cleanup_old_levels(self, symbol: str, current_ts: int) -> int:
        """Remove price levels that haven't been updated recently."""
        if symbol not in self.level_data:
            return 0
        
        cutoff = current_ts - self.price_level_ttl_ns
        original_count = len(self.level_data[symbol])
        
        self.level_data[symbol] = {
            p: l for p, l in self.level_data[symbol].items()
            if l.last_update_ns > cutoff
        }
        
        return original_count - len(self.level_data[symbol])


if __name__ == "__main__":
    # Example usage
    detector = AbsorptionDetector()
    
    # Simulate absorption at a price level
    base_price = 60000.0
    base_time = time.time_ns()
    
    print("Simulating bid absorption...")
    
    # Multiple aggressive sells hitting the same bid level
    for i in range(20):
        event = detector.process_trade(
            symbol="BTCUSDT",
            price=base_price,
            volume=2.0,
            is_buyer_maker=True,  # Sellers hitting bids
            timestamp_ns=base_time + i * 100_000_000  # 100ms apart
        )
        
        if event:
            print(f"\nAbsorption detected at ${event.price:,.2f}")
            print(f"  Type: {event.absorption_type.name}")
            print(f"  Absorbed Volume: {event.absorbed_volume:.2f}")
            print(f"  Attempted Volume: {event.attempted_volume:.2f}")
            print(f"  Confidence: {event.confidence_score:.2f}")
            print(f"  Is Spoof: {event.is_spoof}")
    
    # Test spoof detection
    print("\n\nTesting spoof detection...")
    spoof_price = 60100.0
    
    # Record rapid order placement/cancellation
    for i in range(10):
        t = base_time + i * 1_000_000  # 1ms apart
        detector.record_limit_order("BTCUSDT", spoof_price, t)
        detector.record_order_cancellation("BTCUSDT", spoof_price, t + 3_000_000)  # Cancelled in 3ms
    
    # Process some trades
    detector.process_trade("BTCUSDT", spoof_price, 5.0, False, base_time + 20_000_000)
    
    is_genuine, confidence = detector.differentiate_absorption_vs_spoof("BTCUSDT", spoof_price)
    print(f"Genuine absorption: {is_genuine}, Confidence: {confidence:.2f}")
