#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
File: backend/microstructure/spoofing_detector.py
Chapter 2: Spoofing Detection, Layering, and Fake Liquidity Filtering

Purpose: Identify phantom orders that vanish before fills (spoofing detection)
Constraints: Must flag orders cancelled within 5ms of placement instantly
Target: AMD Ryzen AI 5 laptop with 8GB RAM limit

Spoofing is a manipulative practice where traders place large orders to create
false impression of demand/supply, then cancel before execution.

Detection criteria:
1. Order lifetime < 5ms (ultra-fast cancellation)
2. Large size relative to average
3. Price movement after placement favors spoofer
4. Pattern repetition across multiple price levels

Design Patterns: Observer Pattern for real-time alerts, Strategy Pattern for detection algorithms
Type Hinting: Strict typing for production reliability
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum
from collections import deque, defaultdict
import time
import numpy as np


class SpoofingSeverity(Enum):
    """Severity levels for spoofing detection"""
    NONE = "none"
    SUSPICIOUS = "suspicious"
    LIKELY = "likely"
    CONFIRMED = "confirmed"
    CRITICAL = "critical"


class SpoofingPattern(Enum):
    """Types of spoofing patterns detected"""
    FAST_CANCEL = "fast_cancel"  # Order cancelled within threshold
    MOMENTUM_IGNITION = "momentum_ignition"  # Orders trigger price move then cancel
    MIRROR_TRADING = "mirror_trading"  # Symmetric orders on both sides
    LAYERING = "layering"  # Multiple stacked fake orders
    PINGING = "pinging"  # Rapid small orders to probe book


@dataclass(slots=True)
class OrderEvent:
    """Represents an order lifecycle event"""
    order_id: str
    timestamp_ns: int
    event_type: str  # 'NEW', 'CANCEL', 'FILL', 'AMEND'
    price: float
    quantity: float
    side: str  # 'BID' or 'ASK'
    exchange_timestamp_ns: int = 0
    
    def lifetime_ms(self, end_timestamp_ns: int) -> float:
        """Calculate order lifetime in milliseconds"""
        return (end_timestamp_ns - self.timestamp_ns) / 1_000_000.0


@dataclass(slots=True)
class SpoofingAlert:
    """Alert generated when spoofing is detected"""
    alert_id: str
    timestamp_ns: int
    severity: SpoofingSeverity
    pattern: SpoofingPattern
    order_ids: List[str]
    price_levels: List[float]
    total_quantity: float
    estimated_impact_bps: float  # Estimated price impact in basis points
    confidence_score: float  # 0.0 to 1.0
    details: str
    action_recommended: str  # What the bot should do


@dataclass
class OrderLifecycleTracker:
    """Tracks the lifecycle of individual orders for spoofing analysis"""
    order_id: str
    creation_time_ns: int
    last_update_time_ns: int
    price: float
    quantity: float
    side: str
    status: str  # 'ACTIVE', 'CANCELLED', 'FILLED'
    cancel_time_ns: Optional[int] = None
    fill_time_ns: Optional[int] = None
    
    @property
    def lifetime_ns(self) -> Optional[int]:
        """Get order lifetime in nanoseconds"""
        if self.cancel_time_ns:
            return self.cancel_time_ns - self.creation_time_ns
        elif self.fill_time_ns:
            return self.fill_time_ns - self.creation_time_ns
        return None
    
    @property
    def lifetime_ms(self) -> Optional[float]:
        """Get order lifetime in milliseconds"""
        if self.lifetime_ns:
            return self.lifetime_ns / 1_000_000.0
        return None
    
    def was_fast_cancelled(self, threshold_ms: float = 5.0) -> bool:
        """Check if order was cancelled within threshold"""
        if self.lifetime_ms is None:
            return False
        return self.status == 'CANCELLED' and self.lifetime_ms < threshold_ms


class SpoofingDetector:
    """
    Main detector for spoofing/manipulation patterns in order flow.
    
    Detects orders placed with no intention to execute, designed to 
    manipulate other market participants.
    """
    
    # Detection thresholds
    FAST_CANCEL_THRESHOLD_MS: float = 5.0  # Orders cancelled within 5ms
    LARGE_ORDER_MULTIPLIER: float = 3.0  # Orders >3x average size suspicious
    MIN_CONFIDENCE_FOR_ALERT: float = 0.6
    
    def __init__(self):
        # Active order tracking
        self.active_orders: Dict[str, OrderLifecycleTracker] = {}
        
        # Historical data for pattern recognition
        self.order_history: deque = deque(maxlen=100_000)
        self.cancelled_orders: deque = deque(maxlen=50_000)
        
        # Per-price-level statistics
        self.level_stats: Dict[float, LevelStatistics] = defaultdict(LevelStatistics)
        
        # Alert generation
        self.alert_counter: int = 0
        self.recent_alerts: deque = deque(maxlen=1000)
        
        # Timing
        self.start_time_ns: int = time.time_ns()
        
        # Configuration
        self.enabled_patterns: Set[SpoofingPattern] = set(SpoofingPattern)
        self.severity_threshold: SpoofingSeverity = SpoofingSeverity.SUSPICIOUS
        
        # Sliding window for rate calculations
        self.recent_cancellations: deque = deque(maxlen=10_000)
        
    def process_order_event(self, event: OrderEvent) -> Optional[SpoofingAlert]:
        """
        Process an order event and check for spoofing patterns.
        
        Returns an alert if spoofing is detected, None otherwise.
        This is the main entry point for real-time order flow processing.
        """
        if event.event_type == 'NEW':
            return self._handle_new_order(event)
        elif event.event_type == 'CANCEL':
            return self._handle_cancellation(event)
        elif event.event_type == 'FILL':
            return self._handle_fill(event)
        elif event.event_type == 'AMEND':
            return self._handle_amendment(event)
        return None
    
    def _handle_new_order(self, event: OrderEvent) -> Optional[SpoofingAlert]:
        """Track new order for potential spoofing detection"""
        tracker = OrderLifecycleTracker(
            order_id=event.order_id,
            creation_time_ns=event.timestamp_ns,
            last_update_time_ns=event.timestamp_ns,
            price=event.price,
            quantity=event.quantity,
            side=event.side,
            status='ACTIVE'
        )
        self.active_orders[event.order_id] = tracker
        
        # Update level statistics
        stats = self.level_stats[event.price]
        stats.record_order(event.quantity)
        
        return None
    
    def _handle_cancellation(self, event: OrderEvent) -> Optional[SpoofingAlert]:
        """
        Check cancellation for spoofing patterns.
        
        This is the primary detection point for fast cancellations.
        """
        tracker = self.active_orders.pop(event.order_id, None)
        if not tracker:
            return None
        
        tracker.status = 'CANCELLED'
        tracker.cancel_time_ns = event.timestamp_ns
        tracker.last_update_time_ns = event.timestamp_ns
        
        # Record in history
        self.order_history.append(tracker)
        self.cancelled_orders.append(tracker)
        self.recent_cancellations.append((event.timestamp_ns, event.price, event.quantity))
        
        # Check for fast cancellation (primary spoofing signal)
        if tracker.was_fast_cancelled(self.FAST_CANCEL_THRESHOLD_MS):
            return self._generate_fast_cancel_alert(tracker, event)
        
        # Check for other patterns
        return self._check_cancellation_patterns(tracker, event)
    
    def _handle_fill(self, event: OrderEvent) -> Optional[SpoofingAlert]:
        """Handle order fill - genuine orders get filled, spoofed ones don't"""
        tracker = self.active_orders.pop(event.order_id, None)
        if not tracker:
            return None
        
        tracker.status = 'FILLED'
        tracker.fill_time_ns = event.timestamp_ns
        tracker.last_update_time_ns = event.timestamp_ns
        
        self.order_history.append(tracker)
        
        # Update statistics (genuine fill)
        stats = self.level_stats[event.price]
        stats.record_genuine_fill(event.quantity)
        
        return None
    
    def _handle_amendment(self, event: OrderEvent) -> Optional[SpoofingAlert]:
        """Handle order amendment - can be sign of manipulation"""
        tracker = self.active_orders.get(event.order_id)
        if not tracker:
            return None
        
        tracker.last_update_time_ns = event.timestamp_ns
        
        # Significant amendments can be suspicious
        if abs(event.quantity - tracker.quantity) / max(tracker.quantity, 1) > 0.5:
            # Large size change - could be adjusting spoof
            pass
        
        return None
    
    def _generate_fast_cancel_alert(self, tracker: OrderLifecycleTracker, 
                                     event: OrderEvent) -> SpoofingAlert:
        """Generate alert for fast-cancelled order (high-confidence spoofing)"""
        self.alert_counter += 1
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_spoofing_confidence(tracker)
        
        # Determine severity
        if tracker.lifetime_ms < 1.0:  # <1ms is almost certainly spoofing
            severity = SpoofingSeverity.CRITICAL
        elif tracker.lifetime_ms < 3.0:
            severity = SpoofingSeverity.CONFIRMED
        elif tracker.lifetime_ms < 5.0:
            severity = SpoofingSeverity.LIKELY
        else:
            severity = SpoofingSeverity.SUSPICIOUS
        
        # Estimate market impact
        stats = self.level_stats.get(tracker.price, LevelStatistics())
        avg_size = stats.average_order_size
        size_ratio = tracker.quantity / max(avg_size, 1e-10)
        estimated_impact = min(size_ratio * 0.5, 50.0)  # Cap at 50 bps
        
        alert = SpoofingAlert(
            alert_id=f"SPOOF-{self.alert_counter:06d}",
            timestamp_ns=event.timestamp_ns,
            severity=severity,
            pattern=SpoofingPattern.FAST_CANCEL,
            order_ids=[tracker.order_id],
            price_levels=[tracker.price],
            total_quantity=tracker.quantity,
            estimated_impact_bps=estimated_impact,
            confidence_score=confidence,
            details=f"Order cancelled in {tracker.lifetime_ms:.2f}ms (threshold: {self.FAST_CANCEL_THRESHOLD_MS}ms). "
                    f"Size: {tracker.quantity}, Side: {tracker.side}",
            action_recommended="IGNORE_LIQUIDITY"
        )
        
        self.recent_alerts.append(alert)
        return alert
    
    def _check_cancellation_patterns(self, tracker: OrderLifecycleTracker,
                                      event: OrderEvent) -> Optional[SpoofingAlert]:
        """Check for more complex spoofing patterns beyond fast cancellation"""
        alerts = []
        
        # Check for layering pattern (multiple orders at different prices)
        layering_alert = self._detect_layering_pattern(tracker)
        if layering_alert:
            return layering_alert
        
        # Check for momentum ignition
        ignition_alert = self._detect_momentum_ignition(tracker)
        if ignition_alert:
            return ignition_alert
        
        return None
    
    def _detect_layering_pattern(self, tracker: OrderLifecycleTracker) -> Optional[SpoofingAlert]:
        """
        Detect layering: multiple orders at consecutive price levels
        that are all cancelled quickly.
        """
        # Look for other recent cancellations at nearby prices
        nearby_cancels = []
        for cancelled in list(self.cancelled_orders)[-50:]:
            if cancelled.order_id == tracker.order_id:
                continue
            price_diff = abs(cancelled.price - tracker.price)
            if price_diff <= 5.0:  # Within 5 price units
                if cancelled.lifetime_ms and cancelled.lifetime_ms < 10.0:
                    nearby_cancels.append(cancelled)
        
        # If multiple nearby orders also cancelled quickly, likely layering
        if len(nearby_cancels) >= 3:
            self.alert_counter += 1
            
            total_qty = tracker.quantity + sum(c.quantity for c in nearby_cancels)
            price_levels = sorted(set([tracker.price] + [c.price for c in nearby_cancels]))
            
            return SpoofingAlert(
                alert_id=f"LAYER-{self.alert_counter:06d}",
                timestamp_ns=tracker.cancel_time_ns or 0,
                severity=SpoofingSeverity.LIKELY,
                pattern=SpoofingPattern.LAYERING,
                order_ids=[tracker.order_id] + [c.order_id for c in nearby_cancels],
                price_levels=price_levels,
                total_quantity=total_qty,
                estimated_impact_bps=15.0,
                confidence_score=0.75,
                details=f"Layering detected: {len(nearby_cancels)+1} orders at nearby levels cancelled quickly",
                action_recommended="WIDEN_SPREAD"
            )
        
        return None
    
    def _detect_momentum_ignition(self, tracker: OrderLifecycleTracker) -> Optional[SpoofingAlert]:
        """
        Detect momentum ignition: large orders placed to trigger price movement,
        then cancelled once momentum starts.
        """
        stats = self.level_stats.get(tracker.price, LevelStatistics())
        
        # Check if order was significantly larger than average
        if tracker.quantity < stats.average_order_size * self.LARGE_ORDER_MULTIPLIER:
            return None
        
        # Check if there was price movement after order placement
        # (This would need price data integration - simplified here)
        
        return None
    
    def _calculate_spoofing_confidence(self, tracker: OrderLifecycleTracker) -> float:
        """
        Calculate confidence score for spoofing detection.
        
        Factors:
        1. Order lifetime (shorter = more suspicious)
        2. Order size (larger = more suspicious)
        3. Historical behavior at this price level
        4. Time of day patterns
        """
        confidence = 0.5  # Base confidence
        
        # Lifetime factor
        if tracker.lifetime_ms:
            if tracker.lifetime_ms < 1.0:
                confidence += 0.3
            elif tracker.lifetime_ms < 3.0:
                confidence += 0.2
            elif tracker.lifetime_ms < 5.0:
                confidence += 0.1
        
        # Size factor
        stats = self.level_stats.get(tracker.price, LevelStatistics())
        if stats.average_order_size > 0:
            size_ratio = tracker.quantity / stats.average_order_size
            if size_ratio > 5.0:
                confidence += 0.2
            elif size_ratio > 3.0:
                confidence += 0.1
        
        # Historical cancellation rate at this level
        if stats.cancellation_rate > 0.8:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def get_spoofing_statistics(self) -> Dict[str, float]:
        """Get aggregate spoofing statistics"""
        if not self.cancelled_orders:
            return {}
        
        cancelled_list = list(self.cancelled_orders)
        fast_cancels = [o for o in cancelled_list if o.was_fast_cancelled(self.FAST_CANCEL_THRESHOLD_MS)]
        
        return {
            "total_orders_tracked": len(self.order_history),
            "total_cancellations": len(cancelled_list),
            "fast_cancellations": len(fast_cancels),
            "fast_cancel_ratio": len(fast_cancels) / max(len(cancelled_list), 1),
            "alerts_generated": len(self.recent_alerts),
            "average_order_lifetime_ms": np.mean([o.lifetime_ms for o in cancelled_list if o.lifetime_ms]),
            "median_order_lifetime_ms": np.median([o.lifetime_ms for o in cancelled_list if o.lifetime_ms]),
        }
    
    def is_liquidity_genuine(self, price: float, quantity: float, 
                              recent_orders: List[OrderEvent]) -> bool:
        """
        Check if observed liquidity is genuine or likely spoofed.
        
        Used before placing orders to avoid trading against fake liquidity.
        """
        # Check recent cancellation rate at this price level
        stats = self.level_stats.get(price, LevelStatistics())
        if stats.cancellation_rate > 0.7:
            return False
        
        # Check if quantity is abnormally large (potential spoof)
        if quantity > stats.average_order_size * self.LARGE_ORDER_MULTIPLIER:
            return False
        
        # Check recent fast cancellations at this level
        recent_fast_cancels = sum(
            1 for ts, p, qty in self.recent_cancellations
            if abs(p - price) < 2.0 and ts > time.time_ns() - 1_000_000_000  # Last 1 second
        )
        
        if recent_fast_cancels >= 3:
            return False
        
        return True


@dataclass
class LevelStatistics:
    """Statistics for a single price level"""
    total_orders: int = 0
    total_quantity: float = 0.0
    cancelled_orders: int = 0
    genuine_fills: int = 0
    order_sizes: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    @property
    def average_order_size(self) -> float:
        if not self.order_sizes:
            return 0.0
        return np.mean(self.order_sizes)
    
    @property
    def cancellation_rate(self) -> float:
        if self.total_orders == 0:
            return 0.0
        return self.cancelled_orders / self.total_orders
    
    def record_order(self, quantity: float) -> None:
        self.total_orders += 1
        self.total_quantity += quantity
        self.order_sizes.append(quantity)
    
    def record_cancellation(self) -> None:
        self.cancelled_orders += 1
    
    def record_genuine_fill(self, quantity: float) -> None:
        self.genuine_fills += 1


def main():
    """Example usage of spoofing detector"""
    detector = SpoofingDetector()
    
    # Simulate a spoofing attack
    base_time = time.time_ns()
    
    # Place large order
    spoof_order = OrderEvent(
        order_id="SPOOF-001",
        timestamp_ns=base_time,
        event_type="NEW",
        price=50000.0,
        quantity=100.0,  # Large size
        side="BID"
    )
    detector.process_order_event(spoof_order)
    
    # Cancel it 2ms later (spoofing!)
    cancel_event = OrderEvent(
        order_id="SPOOF-001",
        timestamp_ns=base_time + 2_000_000,  # 2ms later
        event_type="CANCEL",
        price=50000.0,
        quantity=100.0,
        side="BID"
    )
    alert = detector.process_order_event(cancel_event)
    
    if alert:
        print(f"🚨 SPOOFING DETECTED!")
        print(f"Alert ID: {alert.alert_id}")
        print(f"Severity: {alert.severity.value}")
        print(f"Pattern: {alert.pattern.value}")
        print(f"Confidence: {alert.confidence_score:.2%}")
        print(f"Details: {alert.details}")
        print(f"Action: {alert.action_recommended}")
    
    # Show statistics
    stats = detector.get_spoofing_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    main()
