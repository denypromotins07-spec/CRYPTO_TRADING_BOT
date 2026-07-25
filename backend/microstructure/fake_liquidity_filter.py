#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
File: backend/microstructure/fake_liquidity_filter.py
Chapter 2: Spoofing Detection, Layering, and Fake Liquidity Filtering

Purpose: Purge spoofed walls from the local order book representation
Constraints: Must operate in real-time with minimal latency
Target: AMD Ryzen AI 5 laptop with 8GB RAM limit

This filter maintains a "clean" view of the order book by removing
liquidity that has been identified as fake/spoofed through the
spoofing detection system.

Design Patterns: Decorator Pattern for order book wrapping, Strategy for filtering
Type Hinting: Strict typing for production reliability
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum
from collections import deque, defaultdict
import time
import numpy as np


class LiquidityClassification(Enum):
    """Classification of liquidity quality"""
    GENUINE = "genuine"           # Verified real liquidity
    SUSPECTED_FAKE = "suspected"  # Potentially spoofed
    CONFIRMED_FAKE = "confirmed"  # Known spoofed liquidity
    UNKNOWN = "unknown"           # Not yet classified


@dataclass(slots=True)
class OrderBookLevel:
    """Represents a single price level with liquidity metadata"""
    price: float
    quantity: float
    order_count: int
    timestamp_ns: int
    classification: LiquidityClassification = LiquidityClassification.UNKNOWN
    fake_quantity_estimate: float = 0.0  # Estimated amount of fake liquidity
    
    @property
    def genuine_quantity(self) -> float:
        """Get estimated genuine (real) quantity at this level"""
        return max(0.0, self.quantity - self.fake_quantity_estimate)
    
    @property
    def fake_ratio(self) -> float:
        """Get ratio of fake to total quantity"""
        if self.quantity <= 0:
            return 0.0
        return self.fake_quantity_estimate / self.quantity


@dataclass(slots=True)
class FilteredOrderBookSnapshot:
    """Snapshot of filtered order book with only genuine liquidity"""
    timestamp_ns: int
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    total_genuine_bid_qty: float
    total_genuine_ask_qty: float
    total_fake_bid_qty: float
    total_fake_ask_qty: float
    fake_liquidity_ratio: float  # Ratio of fake to total liquidity


class FakeLiquidityFilter:
    """
    Filters out spoofed/fake liquidity from order book representation.
    
    Maintains a clean view of market depth by tracking which orders
    have been identified as spoofed and adjusting visible quantities.
    """
    
    # Configuration thresholds
    CONFIRMATION_THRESHOLD: float = 0.8  # Confidence needed to mark as fake
    SUSPICION_THRESHOLD: float = 0.5     # Confidence to mark as suspected
    DECAY_RATE_PER_SECOND: float = 0.1   # How fast fake labels decay
    
    def __init__(self):
        # Track classification confidence per price level
        self.bid_confidence: Dict[float, float] = defaultdict(float)
        self.ask_confidence: Dict[float, float] = defaultdict(float)
        
        # Track specific order IDs identified as fake
        self.confirmed_fake_orders: Set[str] = set()
        self.suspected_orders: Set[str] = set()
        
        # Historical spoofing alerts for pattern recognition
        self.spoofing_alerts: deque = deque(maxlen=10_000)
        
        # Per-level statistics
        self.level_history: Dict[float, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Timing
        self.last_update_ns: int = 0
        self.start_time_ns: int = time.time_ns()
        
        # Configuration
        self.enabled: bool = True
        self.aggressive_mode: bool = False  # More aggressive filtering
        
    def record_spoofing_alert(self, alert_data: dict) -> None:
        """
        Record a spoofing alert from the spoofing detector.
        
        This is the primary input for identifying fake liquidity.
        """
        self.spoofing_alerts.append({
            'timestamp_ns': time.time_ns(),
            'price_levels': alert_data.get('price_levels', []),
            'total_quantity': alert_data.get('total_quantity', 0),
            'confidence': alert_data.get('confidence_score', 0),
            'side': alert_data.get('side', 'UNKNOWN'),
        })
        
        # Update confidence for affected price levels
        side = alert_data.get('side', 'BID')
        confidence_map = self.bid_confidence if side == 'BID' else self.ask_confidence
        
        for price in alert_data.get('price_levels', []):
            current_conf = confidence_map[price]
            new_conf = min(1.0, current_conf + alert_data.get('confidence_score', 0) * 0.3)
            confidence_map[price] = new_conf
    
    def classify_level(self, price: float, quantity: float, 
                       side: str, recent_events: list) -> LiquidityClassification:
        """
        Classify the quality of liquidity at a price level.
        
        Returns classification based on historical patterns and recent alerts.
        """
        confidence_map = self.bid_confidence if side == 'BID' else self.ask_confidence
        confidence = confidence_map.get(price, 0.0)
        
        # Check for recent rapid cancellations at this level
        recent_cancels = sum(
            1 for evt in recent_events[-50:]
            if abs(evt.get('price', 0) - price) < 2.0 
            and evt.get('event_type') == 'CANCEL'
            and evt.get('lifetime_ms', 100) < 10
        )
        
        # Adjust confidence based on recent activity
        if recent_cancels >= 3:
            confidence = min(1.0, confidence + 0.2)
        
        # Large size relative to history is suspicious
        avg_size = self._get_average_level_size(price, side)
        if avg_size > 0 and quantity > avg_size * 5:
            confidence = min(1.0, confidence + 0.15)
        
        # Update classification
        if confidence >= self.CONFIRMATION_THRESHOLD:
            return LiquidityClassification.CONFIRMED_FAKE
        elif confidence >= self.SUSPICION_THRESHOLD:
            return LiquidityClassification.SUSPECTED_FAKE
        else:
            return LiquidityClassification.GENUINE
    
    def _get_average_level_size(self, price: float, side: str) -> float:
        """Get historical average size at this price level"""
        history = self.level_history.get(price)
        if not history:
            return 0.0
        
        sizes = [h['quantity'] for h in history if h.get('side') == side]
        if not sizes:
            return 0.0
        
        return np.mean(sizes)
    
    def estimate_fake_quantity(self, price: float, total_quantity: float,
                                side: str) -> float:
        """
        Estimate how much of the visible quantity is fake.
        
        Returns the estimated fake portion of the total quantity.
        """
        confidence_map = self.bid_confidence if side == 'BID' else self.ask_confidence
        confidence = confidence_map.get(price, 0.0)
        
        # Apply decay based on time since last spoofing signal
        elapsed_s = (time.time_ns() - self.start_time_ns) / 1e9
        decayed_confidence = max(0.0, confidence - elapsed_s * self.DECAY_RATE_PER_SECOND)
        
        # In aggressive mode, be more suspicious
        if self.aggressive_mode:
            decayed_confidence = min(1.0, decayed_confidence * 1.5)
        
        return total_quantity * decayed_confidence
    
    def filter_order_book(self, raw_bids: List[OrderBookLevel], 
                          raw_asks: List[OrderBookLevel]) -> FilteredOrderBookSnapshot:
        """
        Create a filtered view of the order book with fake liquidity removed.
        
        This is the main method called before making trading decisions.
        """
        if not self.enabled:
            # Return unfiltered snapshot
            return self._create_snapshot(raw_bids, raw_asks, 0.0, 0.0)
        
        timestamp_ns = time.time_ns()
        self.last_update_ns = timestamp_ns
        
        filtered_bids = []
        filtered_asks = []
        total_fake_bid = 0.0
        total_fake_ask = 0.0
        
        # Filter bids
        for level in raw_bids:
            classification = self.classify_level(
                level.price, level.quantity, 'BID', []
            )
            
            fake_qty = self.estimate_fake_quantity(level.price, level.quantity, 'BID')
            
            # Create filtered level
            filtered_level = OrderBookLevel(
                price=level.price,
                quantity=level.quantity,
                order_count=level.order_count,
                timestamp_ns=timestamp_ns,
                classification=classification,
                fake_quantity_estimate=fake_qty
            )
            
            total_fake_bid += fake_qty
            
            # Only include if there's genuine liquidity
            if filtered_level.genuine_quantity > 0:
                filtered_bids.append(filtered_level)
        
        # Filter asks
        for level in raw_asks:
            classification = self.classify_level(
                level.price, level.quantity, 'ASK', []
            )
            
            fake_qty = self.estimate_fake_quantity(level.price, level.quantity, 'ASK')
            
            filtered_level = OrderBookLevel(
                price=level.price,
                quantity=level.quantity,
                order_count=level.order_count,
                timestamp_ns=timestamp_ns,
                classification=classification,
                fake_quantity_estimate=fake_qty
            )
            
            total_fake_ask += fake_qty
            
            if filtered_level.genuine_quantity > 0:
                filtered_asks.append(filtered_level)
        
        return self._create_snapshot(
            filtered_bids, filtered_asks, total_fake_bid, total_fake_ask
        )
    
    def _create_snapshot(self, bids: List[OrderBookLevel], asks: List[OrderBookLevel],
                         fake_bid_qty: float, fake_ask_qty: float) -> FilteredOrderBookSnapshot:
        """Create a filtered snapshot object"""
        genuine_bid_qty = sum(b.genuine_quantity for b in bids)
        genuine_ask_qty = sum(a.genuine_quantity for a in asks)
        
        total_qty = genuine_bid_qty + genuine_ask_qty + fake_bid_qty + fake_ask_qty
        fake_ratio = (fake_bid_qty + fake_ask_qty) / max(total_qty, 1e-10)
        
        return FilteredOrderBookSnapshot(
            timestamp_ns=time.time_ns(),
            bids=bids,
            asks=asks,
            total_genuine_bid_qty=genuine_bid_qty,
            total_genuine_ask_qty=genuine_ask_qty,
            total_fake_bid_qty=fake_bid_qty,
            total_fake_ask_qty=fake_ask_qty,
            fake_liquidity_ratio=fake_ratio
        )
    
    def should_ignore_level(self, price: float, side: str) -> bool:
        """
        Check if a price level should be completely ignored for trading.
        
        Used when deciding where to place orders or execute trades.
        """
        confidence_map = self.bid_confidence if side == 'BID' else self.ask_confidence
        confidence = confidence_map.get(price, 0.0)
        
        # Ignore if confirmed fake
        if confidence >= self.CONFIRMATION_THRESHOLD:
            return True
        
        # In aggressive mode, also ignore suspected fake
        if self.aggressive_mode and confidence >= self.SUSPICION_THRESHOLD:
            return True
        
        return False
    
    def get_adjusted_spread(self, best_bid: float, best_ask: float,
                            bid_qty: float, ask_qty: float) -> Tuple[float, float]:
        """
        Get spread adjusted for fake liquidity.
        
        Returns (adjusted_best_bid, adjusted_best_ask) considering only genuine liquidity.
        """
        # If best levels have significant fake liquidity, look deeper
        bid_fake_ratio = self.estimate_fake_quantity(best_bid, bid_qty, 'BID') / max(bid_qty, 1)
        ask_fake_ratio = self.estimate_fake_quantity(best_ask, ask_qty, 'ASK') / max(ask_qty, 1)
        
        adjusted_bid = best_bid
        adjusted_ask = best_ask
        
        # If more than 50% fake, consider the level unreliable
        if bid_fake_ratio > 0.5:
            # Would need to look at next level - simplified here
            adjusted_bid = best_bid * 0.9999  # Slightly worse
        
        if ask_fake_ratio > 0.5:
            adjusted_ask = best_ask * 1.0001  # Slightly worse
        
        return adjusted_bid, adjusted_ask
    
    def update_from_fill(self, price: float, quantity: float, side: str) -> None:
        """
        Update internal state when a fill occurs.
        
        Genuine fills reduce our confidence that liquidity at that level is fake.
        """
        confidence_map = self.bid_confidence if side == 'BID' else self.ask_confidence
        
        # Reduce confidence (genuine fill suggests real liquidity)
        current = confidence_map.get(price, 0.0)
        confidence_map[price] = max(0.0, current - 0.2)
        
        # Record in history
        self.level_history[price].append({
            'timestamp_ns': time.time_ns(),
            'price': price,
            'quantity': quantity,
            'side': side,
            'event_type': 'FILL'
        })
    
    def reset_confidence(self, price: Optional[float] = None) -> None:
        """Reset confidence scores, optionally for a specific price level"""
        if price is not None:
            self.bid_confidence[price] = 0.0
            self.ask_confidence[price] = 0.0
        else:
            self.bid_confidence.clear()
            self.ask_confidence.clear()
    
    def get_statistics(self) -> dict:
        """Get filter statistics for monitoring"""
        confirmed_fake_levels = sum(
            1 for conf in self.bid_confidence.values() 
            if conf >= self.CONFIRMATION_THRESHOLD
        ) + sum(
            1 for conf in self.ask_confidence.values()
            if conf >= self.CONFIRMATION_THRESHOLD
        )
        
        suspected_levels = sum(
            1 for conf in self.bid_confidence.values()
            if self.SUSPICION_THRESHOLD <= conf < self.CONFIRMATION_THRESHOLD
        ) + sum(
            1 for conf in self.ask_confidence.values()
            if self.SUSPICION_THRESHOLD <= conf < self.CONFIRMATION_THRESHOLD
        )
        
        return {
            'confirmed_fake_levels': confirmed_fake_levels,
            'suspected_fake_levels': suspected_levels,
            'total_tracked_levels': len(self.bid_confidence) + len(self.ask_confidence),
            'spoofing_alerts_recorded': len(self.spoofing_alerts),
            'filter_enabled': self.enabled,
            'aggressive_mode': self.aggressive_mode,
        }


def main():
    """Example usage of fake liquidity filter"""
    filter_obj = FakeLiquidityFilter()
    
    # Simulate receiving a spoofing alert
    filter_obj.record_spoofing_alert({
        'price_levels': [50000.0, 50001.0],
        'total_quantity': 5000.0,
        'confidence_score': 0.85,
        'side': 'ASK'
    })
    
    # Create raw order book
    raw_bids = [
        OrderBookLevel(price=49999.0, quantity=1000.0, order_count=5, timestamp_ns=time.time_ns()),
        OrderBookLevel(price=49998.0, quantity=2000.0, order_count=8, timestamp_ns=time.time_ns()),
    ]
    
    raw_asks = [
        OrderBookLevel(price=50000.0, quantity=5000.0, order_count=3, timestamp_ns=time.time_ns()),  # Suspected fake
        OrderBookLevel(price=50001.0, quantity=3000.0, order_count=2, timestamp_ns=time.time_ns()),  # Suspected fake
        OrderBookLevel(price=50002.0, quantity=1500.0, order_count=6, timestamp_ns=time.time_ns()),
    ]
    
    # Get filtered view
    filtered = filter_obj.filter_order_book(raw_bids, raw_asks)
    
    print("=== FAKE LIQUIDITY FILTER RESULTS ===")
    print(f"Total Genuine Bid Qty: {filtered.total_genuine_bid_qty:.2f}")
    print(f"Total Genuine Ask Qty: {filtered.total_genuine_ask_qty:.2f}")
    print(f"Total Fake Bid Qty: {filtered.total_fake_bid_qty:.2f}")
    print(f"Total Fake Ask Qty: {filtered.total_fake_ask_qty:.2f}")
    print(f"Fake Liquidity Ratio: {filtered.fake_liquidity_ratio:.2%}")
    
    print("\nFiltered Bids:")
    for bid in filtered.bids:
        print(f"  ${bid.price}: {bid.genuine_quantity:.2f} genuine / {bid.quantity:.2f} total ({bid.classification.value})")
    
    print("\nFiltered Asks:")
    for ask in filtered.asks:
        print(f"  ${ask.price}: {ask.genuine_quantity:.2f} genuine / {ask.quantity:.2f} total ({ask.classification.value})")
    
    # Statistics
    stats = filter_obj.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    main()
