#!/usr/bin/env python3
"""
Cancellation Intensity Analyzer

This module analyzes the ratio of cancellations to executions in the order queue,
providing insights into market participant behavior and potential manipulation.

Key Features:
- Real-time cancellation rate tracking
- Queue position decay analysis
- Spoofing pattern detection
- Strict type hinting for memory safety
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum, auto
import time
import math
from threading import Lock


class CancellationPattern(Enum):
    """Classification of cancellation patterns."""
    NORMAL = auto()       # Typical market making cancellations
    ELEVATED = auto()     # Above normal cancellation rate
    SUSPICIOUS = auto()   # Potential spoofing pattern
    MANIPULATIVE = auto() # Clear manipulation signature


@dataclass
class QueueOrder:
    """Represents an order in the queue."""
    order_id: str
    timestamp_us: int
    volume: float
    price: float
    side: str  # 'bid' or 'ask'
    queue_position: int


@dataclass
class CancellationEvent:
    """Records a cancellation event."""
    order_id: str
    timestamp_us: int
    lifetime_us: int  # Time from placement to cancellation
    volume: float
    was_at_top: bool  # Was order at top of queue?
    fill_probability_before_cancel: float


@dataclass
class CancellationMetrics:
    """Aggregated cancellation metrics."""
    # Cancellation rate (cancellations / total orders)
    cancellation_rate: float
    # Average order lifetime (microseconds)
    avg_lifetime_us: float
    # Cancellation rate at top of queue
    top_queue_cancel_rate: float
    # Fast cancel rate (<5ms)
    fast_cancel_rate: float
    # Volume-weighted cancellation rate
    volume_cancel_rate: float
    # Pattern classification
    pattern: CancellationPattern
    # Manipulation score (0-1)
    manipulation_score: float


class CancellationIntensityAnalyzer:
    """
    Analyze cancellation intensity and patterns in the order queue.
    
    This class tracks the ratio of cancellations to executions,
    identifies suspicious patterns, and provides early warning
    of potential market manipulation.
    """
    
    def __init__(
        self,
        window_size_s: float = 60.0,
        fast_cancel_threshold_ms: float = 5.0,
        suspicious_rate_threshold: float = 0.5,
    ):
        """
        Initialize the cancellation analyzer.
        
        Args:
            window_size_s: Analysis window size in seconds
            fast_cancel_threshold_ms: Threshold for "fast" cancellation (ms)
            suspicious_rate_threshold: Rate above which pattern is suspicious
        """
        self.window_size_s = window_size_s
        self.fast_cancel_threshold_us = int(fast_cancel_threshold_ms * 1000)
        self.suspicious_rate_threshold = suspicious_rate_threshold
        
        # Active orders in queue
        self.active_orders: Dict[str, QueueOrder] = {}
        
        # Historical cancellations
        self.cancellations: Deque[CancellationEvent] = deque(maxlen=10000)
        
        # Historical executions
        self.executions: Deque[Dict] = deque(maxlen=10000)
        
        # Rolling metrics
        self.metrics_history: Deque[Tuple[int, CancellationMetrics]] = deque(maxlen=1000)
        
        # Thread safety
        self._lock = Lock()
    
    def add_order(self, order: QueueOrder) -> None:
        """Record a new order entering the queue."""
        with self._lock:
            self.active_orders[order.order_id] = order
    
    def cancel_order(
        self, 
        order_id: str, 
        timestamp_us: int,
    ) -> Optional[CancellationEvent]:
        """
        Record an order cancellation.
        
        Args:
            order_id: ID of the cancelled order
            timestamp_us: Cancellation timestamp
            
        Returns:
            CancellationEvent if order was found, None otherwise
        """
        with self._lock:
            if order_id not in self.active_orders:
                return None
            
            order = self.active_orders.pop(order_id)
            lifetime_us = timestamp_us - order.timestamp_us
            
            # Estimate fill probability before cancellation
            # (simplified: based on queue position and time)
            fill_prob = self._estimate_fill_probability(order, lifetime_us)
            
            event = CancellationEvent(
                order_id=order_id,
                timestamp_us=timestamp_us,
                lifetime_us=lifetime_us,
                volume=order.volume,
                was_at_top=order.queue_position <= 3,
                fill_probability_before_cancel=fill_prob,
            )
            
            self.cancellations.append(event)
            
            # Prune old events
            self._prune_old_events(timestamp_us)
            
            return event
    
    def execute_order(
        self,
        order_id: str,
        timestamp_us: int,
        filled_volume: float,
    ) -> None:
        """Record an order execution (fill)."""
        with self._lock:
            if order_id in self.active_orders:
                order = self.active_orders.pop(order_id)
                self.executions.append({
                    'order_id': order_id,
                    'timestamp_us': timestamp_us,
                    'volume': filled_volume,
                    'side': order.side,
                    'price': order.price,
                })
    
    def _estimate_fill_probability(
        self, 
        order: QueueOrder, 
        lifetime_us: int
    ) -> float:
        """Estimate fill probability based on queue dynamics."""
        # Simplified model: probability decays with queue position
        # and increases with time in queue
        position_factor = 1.0 / (order.queue_position + 1)
        time_factor = min(1.0, lifetime_us / (60_000_000))  # Max at 60s
        
        return 0.5 * position_factor + 0.5 * time_factor
    
    def _prune_old_events(self, current_us: int) -> None:
        """Remove events outside the analysis window."""
        cutoff_us = int((current_us / 1_000_000 - self.window_size_s) * 1_000_000)
        
        while self.cancellations and self.cancellations[0].timestamp_us < cutoff_us:
            self.cancellations.popleft()
        
        while self.executions and self.executions[0]['timestamp_us'] < cutoff_us:
            self.executions.popleft()
    
    def get_metrics(self, timestamp_us: Optional[int] = None) -> CancellationMetrics:
        """
        Calculate current cancellation metrics.
        
        Args:
            timestamp_us: Current timestamp (uses time.time() if None)
            
        Returns:
            CancellationMetrics with current analysis
        """
        if timestamp_us is None:
            timestamp_us = int(time.time() * 1_000_000)
        
        with self._lock:
            self._prune_old_events(timestamp_us)
            
            total_orders = len(self.cancellations) + len(self.executions)
            if total_orders == 0:
                return CancellationMetrics(
                    cancellation_rate=0.0,
                    avg_lifetime_us=0.0,
                    top_queue_cancel_rate=0.0,
                    fast_cancel_rate=0.0,
                    volume_cancel_rate=0.0,
                    pattern=CancellationPattern.NORMAL,
                    manipulation_score=0.0,
                )
            
            # Basic cancellation rate
            cancellation_rate = len(self.cancellations) / total_orders
            
            # Average lifetime
            if self.cancellations:
                avg_lifetime = sum(c.lifetime_us for c in self.cancellations) / len(self.cancellations)
            else:
                avg_lifetime = 0.0
            
            # Top queue cancellation rate
            top_cancels = sum(1 for c in self.cancellations if c.was_at_top)
            top_cancel_rate = top_cancels / max(1, len(self.cancellations))
            
            # Fast cancellation rate (<5ms)
            fast_cancels = sum(
                1 for c in self.cancellations 
                if c.lifetime_us < self.fast_cancel_threshold_us
            )
            fast_cancel_rate = fast_cancels / max(1, len(self.cancellations))
            
            # Volume-weighted cancellation rate
            cancel_volume = sum(c.volume for c in self.cancellations)
            exec_volume = sum(e['volume'] for e in self.executions)
            total_volume = cancel_volume + exec_volume
            volume_cancel_rate = cancel_volume / max(1, total_volume)
            
            # Classify pattern
            pattern, manipulation_score = self._classify_pattern(
                cancellation_rate,
                fast_cancel_rate,
                top_cancel_rate,
                volume_cancel_rate,
            )
            
            metrics = CancellationMetrics(
                cancellation_rate=cancellation_rate,
                avg_lifetime_us=avg_lifetime,
                top_queue_cancel_rate=top_cancel_rate,
                fast_cancel_rate=fast_cancel_rate,
                volume_cancel_rate=volume_cancel_rate,
                pattern=pattern,
                manipulation_score=manipulation_score,
            )
            
            # Store in history
            self.metrics_history.append((timestamp_us, metrics))
            
            return metrics
    
    def _classify_pattern(
        self,
        cancel_rate: float,
        fast_rate: float,
        top_rate: float,
        volume_rate: float,
    ) -> Tuple[CancellationPattern, float]:
        """Classify cancellation pattern and calculate manipulation score."""
        # Manipulation score based on multiple factors
        score = 0.0
        
        # High overall cancellation rate
        if cancel_rate > 0.7:
            score += 0.3
        elif cancel_rate > 0.5:
            score += 0.15
        
        # High fast cancellation rate (spoofing indicator)
        if fast_rate > 0.5:
            score += 0.35
        elif fast_rate > 0.3:
            score += 0.2
        
        # Cancelling at top of queue (blocking behavior)
        if top_rate > 0.4:
            score += 0.2
        elif top_rate > 0.2:
            score += 0.1
        
        # High volume cancellation rate
        if volume_rate > 0.6:
            score += 0.15
        
        score = min(1.0, score)
        
        # Classify based on score
        if score > 0.7:
            pattern = CancellationPattern.MANIPULATIVE
        elif score > 0.5:
            pattern = CancellationPattern.SUSPICIOUS
        elif score > 0.3:
            pattern = CancellationPattern.ELEVATED
        else:
            pattern = CancellationPattern.NORMAL
        
        return pattern, score
    
    def is_spoofing_likely(self) -> bool:
        """Check if current patterns suggest spoofing."""
        metrics = self.get_metrics()
        return metrics.manipulation_score > 0.5
    
    def get_queue_decay_rate(self) -> float:
        """
        Calculate the rate at which orders decay from the queue.
        
        Returns:
            Decay rate (orders per second)
        """
        with self._lock:
            if not self.cancellations:
                return 0.0
            
            # Get time range
            if len(self.cancellations) < 2:
                return 0.0
            
            time_range_s = (
                self.cancellations[-1].timestamp_us - self.cancellations[0].timestamp_us
            ) / 1_000_000.0
            
            if time_range_s <= 0:
                return 0.0
            
            return len(self.cancellations) / time_range_s
    
    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self.active_orders.clear()
            self.cancellations.clear()
            self.executions.clear()
            self.metrics_history.clear()


if __name__ == "__main__":
    # Example usage
    analyzer = CancellationIntensityAnalyzer()
    
    base_time = int(time.time() * 1_000_000)
    
    # Add some orders
    for i in range(10):
        order = QueueOrder(
            order_id=f"order_{i}",
            timestamp_us=base_time + i * 1000,
            volume=1.0 + i * 0.1,
            price=50000.0 + i,
            side='bid',
            queue_position=i + 1,
        )
        analyzer.add_order(order)
    
    # Cancel some quickly (potential spoofing)
    for i in range(5):
        analyzer.cancel_order(f"order_{i}", base_time + 10_000 + i * 1000)
    
    # Execute some
    for i in range(5, 8):
        analyzer.execute_order(f"order_{i}", base_time + 20_000, 1.0)
    
    # Get metrics
    metrics = analyzer.get_metrics(base_time + 30_000)
    
    print(f"Cancellation Rate: {metrics.cancellation_rate:.2%}")
    print(f"Fast Cancel Rate: {metrics.fast_cancel_rate:.2%}")
    print(f"Pattern: {metrics.pattern.name}")
    print(f"Manipulation Score: {metrics.manipulation_score:.2f}")
    print(f"Spoofing Likely: {analyzer.is_spoofing_likely()}")
