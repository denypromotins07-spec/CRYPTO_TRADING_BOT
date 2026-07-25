"""
Queue Position Tracker
Estimates fill probability based on historical queue decay and order book dynamics.
Optimized for tracking exact queue positions on Binance to avoid phantom fills.

This module implements:
- Queue position estimation from L2/L3 data
- Historical decay pattern analysis
- Fill probability calculation
- Phantom fill detection and prevention

Target: Avoid latency arbitrage and accurately predict order execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum
import numpy as np
from datetime import datetime, timezone


class QueuePosition(Enum):
    """Relative position in the order queue."""
    FRONT = "front"  # Top of queue, high fill probability
    MIDDLE = "middle"  # Middle of queue, moderate probability
    BACK = "back"  # Back of queue, low probability
    UNKNOWN = "unknown"  # Cannot determine


@dataclass
class QueueEvent:
    """Represents an event affecting queue position."""
    timestamp_ms: int
    event_type: str  # 'new_order', 'cancel', 'fill'
    volume: float
    price: float
    side: str  # 'bid' or 'ask'


@dataclass
class QueueState:
    """Current state of the order queue at a price level."""
    price: float
    side: str  # 'bid' or 'ask'
    total_volume: float
    our_volume: float
    estimated_position: float  # Volume ahead of us
    queue_length: int  # Number of orders
    last_update_ms: int
    
    @property
    def position_ratio(self) -> float:
        """Get our position as a ratio (0 = front, 1 = back)."""
        if self.total_volume == 0:
            return 0.0
        return self.estimated_position / self.total_volume
    
    @property
    def fill_probability(self) -> float:
        """Estimate probability of fill in next time window."""
        # Simple model: probability decreases linearly with position
        base_prob = 0.8
        position_penalty = self.position_ratio * 0.6
        return max(0.0, min(1.0, base_prob - position_penalty))


@dataclass
class QueueConfig:
    """Configuration for queue position tracking."""
    # Time window for decay analysis (milliseconds)
    decay_window_ms: int = 5000
    
    # Minimum samples for reliable estimation
    min_samples: int = 20
    
    # Maximum age for queue data (milliseconds)
    max_data_age_ms: int = 1000
    
    # Decay rate estimation lookback (number of events)
    decay_lookback: int = 100
    
    # Probability thresholds
    high_prob_threshold: float = 0.7
    low_prob_threshold: float = 0.3


class QueuePositionTracker:
    """
    Tracks queue position and estimates fill probabilities.
    
    Uses historical decay patterns and real-time order book updates
    to estimate where our orders sit in the queue.
    """
    
    def __init__(self, config: Optional[QueueConfig] = None):
        self.config = config or QueueConfig()
        
        # Per-price-level queue states
        self.queue_states: Dict[Tuple[float, str], QueueState] = {}
        
        # Event history for decay analysis (per symbol-side)
        self.event_history: Dict[str, Deque[QueueEvent]] = {}
        
        # Historical decay rates (volume cancelled/filled per ms)
        self.decay_rates: Dict[str, float] = {}
        
        # Our active orders tracking
        self.our_orders: Dict[str, Dict] = {}  # order_id -> order_info
        
        # Fill statistics
        self.fill_stats: Dict[str, Dict] = {}
        
        # Phantom fill counter
        self.phantom_fill_count: int = 0
    
    def update_queue_state(
        self,
        price: float,
        side: str,
        total_volume: float,
        our_volume: float,
        queue_length: int,
        timestamp_ms: int
    ) -> QueueState:
        """Update the queue state for a specific price level."""
        key = (price, side)
        
        # Get previous state for position estimation
        prev_state = self.queue_states.get(key)
        
        if prev_state is not None and our_volume > 0:
            # Estimate position based on changes since last update
            volume_change = total_volume - prev_state.total_volume
            our_volume_change = our_volume - prev_state.our_volume
            
            # If our volume increased, we're likely at the back of new volume
            if our_volume_change > 0:
                estimated_ahead = prev_state.estimated_position + prev_state.our_volume
            else:
                # Estimate based on proportional position
                if prev_state.total_volume > 0:
                    position_ratio = prev_state.estimated_position / prev_state.total_volume
                    estimated_ahead = position_ratio * (total_volume - our_volume)
                else:
                    estimated_ahead = 0.0
        else:
            # New order, assume random position in queue
            estimated_ahead = (total_volume - our_volume) * 0.5
        
        state = QueueState(
            price=price,
            side=side,
            total_volume=total_volume,
            our_volume=our_volume,
            estimated_position=estimated_ahead,
            queue_length=queue_length,
            last_update_ms=timestamp_ms
        )
        
        self.queue_states[key] = state
        return state
    
    def record_event(self, symbol: str, event: QueueEvent) -> None:
        """Record a queue event for decay analysis."""
        key = f"{symbol}_{event.side}"
        
        if key not in self.event_history:
            self.event_history[key] = deque(maxlen=self.config.decay_lookback)
        
        self.event_history[key].append(event)
        
        # Update decay rate if enough events
        if len(self.event_history[key]) >= self.config.min_samples:
            self._update_decay_rate(key)
    
    def _update_decay_rate(self, key: str) -> None:
        """Calculate decay rate from historical events."""
        events = list(self.event_history[key])
        if len(events) < 2:
            return
        
        # Calculate volume reduction over time windows
        cancels_and_fills = [e for e in events if e.event_type in ('cancel', 'fill')]
        
        if len(cancels_and_fills) >= 2:
            time_span_ms = events[-1].timestamp_ms - events[0].timestamp_ms
            if time_span_ms > 0:
                total_volume = sum(e.volume for e in cancels_and_fills)
                decay_rate = total_volume / time_span_ms
                self.decay_rates[key] = decay_rate
    
    def estimate_time_to_fill(
        self,
        symbol: str,
        price: float,
        side: str,
        volume: float
    ) -> Optional[float]:
        """
        Estimate time until order fill in milliseconds.
        
        Returns None if fill is unlikely within reasonable time.
        """
        key = (price, side)
        state = self.queue_states.get(key)
        
        if state is None or state.our_volume == 0:
            return None
        
        decay_key = f"{symbol}_{side}"
        decay_rate = self.decay_rates.get(decay_key, 0.0)
        
        if decay_rate <= 0:
            # No decay observed, use default estimate
            # Assume 1% of queue clears per second
            decay_rate = state.total_volume * 0.01 / 1000
        
        # Time = volume ahead / decay rate
        time_to_fill_ms = state.estimated_position / decay_rate
        
        # Sanity check: cap at reasonable maximum (5 minutes)
        return min(time_to_fill_ms, 300000)
    
    def get_fill_probability(
        self,
        symbol: str,
        price: float,
        side: str,
        time_horizon_ms: int = 1000
    ) -> float:
        """
        Calculate probability of fill within given time horizon.
        
        Uses exponential decay model based on historical patterns.
        """
        key = (price, side)
        state = self.queue_states.get(key)
        
        if state is None or state.our_volume == 0:
            return 0.0
        
        # Check data freshness
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        data_age = current_time_ms - state.last_update_ms
        
        if data_age > self.config.max_data_age_ms:
            # Data too stale, reduce probability
            return state.fill_probability * 0.5
        
        decay_key = f"{symbol}_{side}"
        decay_rate = self.decay_rates.get(decay_key, 0.0)
        
        if decay_rate <= 0:
            return state.fill_probability * 0.5
        
        # Volume expected to clear in time horizon
        expected_clearance = decay_rate * time_horizon_ms
        
        # Probability based on whether expected clearance exceeds our position
        if expected_clearance >= state.estimated_position:
            return min(1.0, expected_clearance / (state.estimated_position + state.our_volume))
        else:
            # Partial probability based on ratio
            return 0.3 * (expected_clearance / max(state.estimated_position, 1))
    
    def detect_phantom_fill(
        self,
        order_id: str,
        expected_status: str,
        reported_status: str
    ) -> bool:
        """
        Detect potential phantom fills (reported fill without actual execution).
        
        This can happen due to exchange bugs or latency issues.
        """
        if expected_status == 'open' and reported_status == 'filled':
            # Check if queue position suggested fill was unlikely
            order_info = self.our_orders.get(order_id)
            if order_info:
                prob = self.get_fill_probability(
                    order_info['symbol'],
                    order_info['price'],
                    order_info['side'],
                    time_horizon_ms=100
                )
                
                if prob < self.config.low_prob_threshold:
                    self.phantom_fill_count += 1
                    return True
        
        return False
    
    def register_order(
        self,
        order_id: str,
        symbol: str,
        price: float,
        side: str,
        volume: float
    ) -> None:
        """Register a new order for tracking."""
        self.our_orders[order_id] = {
            'symbol': symbol,
            'price': price,
            'side': side,
            'volume': volume,
            'status': 'open',
            'created_ms': int(datetime.now(timezone.utc).timestamp() * 1000)
        }
        
        # Initialize fill stats
        if symbol not in self.fill_stats:
            self.fill_stats[symbol] = {'fills': 0, 'total_volume': 0.0}
    
    def update_order_status(
        self,
        order_id: str,
        new_status: str,
        filled_volume: float = 0.0
    ) -> None:
        """Update order status and track fill statistics."""
        if order_id not in self.our_orders:
            return
        
        order = self.our_orders[order_id]
        old_status = order['status']
        
        # Check for phantom fills
        if self.detect_phantom_fill(order_id, old_status, new_status):
            print(f"WARNING: Potential phantom fill detected for order {order_id}")
        
        order['status'] = new_status
        
        if new_status == 'filled' or filled_volume > 0:
            symbol = order['symbol']
            self.fill_stats[symbol]['fills'] += 1
            self.fill_stats[symbol]['total_volume'] += filled_volume or order['volume']
    
    def get_queue_position_category(self, state: QueueState) -> QueuePosition:
        """Categorize queue position into discrete buckets."""
        ratio = state.position_ratio
        
        if ratio < 0.2:
            return QueuePosition.FRONT
        elif ratio < 0.6:
            return QueuePosition.MIDDLE
        else:
            return QueuePosition.BACK
    
    def get_optimal_price_for_quick_fill(
        self,
        symbol: str,
        side: str,
        current_book: Dict
    ) -> Optional[float]:
        """
        Determine optimal price to achieve quick fill based on queue analysis.
        
        Returns price level where fill probability exceeds threshold.
        """
        if side == 'bid':
            levels = sorted(current_book.get('bids', {}).keys(), reverse=True)
        else:
            levels = sorted(current_book.get('asks', {}).keys())
        
        for price in levels:
            key = (price, side)
            state = self.queue_states.get(key)
            
            if state is None:
                continue
            
            prob = self.get_fill_probability(symbol, price, side, time_horizon_ms=500)
            
            if prob >= self.config.high_prob_threshold:
                return price
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get comprehensive queue tracking statistics."""
        active_orders = sum(1 for o in self.our_orders.values() if o['status'] == 'open')
        
        avg_fill_prob = 0.0
        if self.queue_states:
            probs = [s.fill_probability for s in self.queue_states.values() if s.our_volume > 0]
            if probs:
                avg_fill_prob = np.mean(probs)
        
        return {
            'active_orders': active_orders,
            'tracked_price_levels': len(self.queue_states),
            'average_fill_probability': avg_fill_prob,
            'phantom_fill_count': self.phantom_fill_count,
            'decay_rates': dict(self.decay_rates),
            'fill_statistics': dict(self.fill_stats),
        }


# Example usage and testing
if __name__ == "__main__":
    import time
    
    tracker = QueuePositionTracker()
    
    # Simulate queue state
    base_ts = int(time.time() * 1000)
    
    # Update queue state at best bid
    state = tracker.update_queue_state(
        price=50000.0,
        side='bid',
        total_volume=100.0,
        our_volume=5.0,
        queue_length=20,
        timestamp_ms=base_ts
    )
    
    print(f"Queue State:")
    print(f"  Position Ratio: {state.position_ratio:.2%}")
    print(f"  Fill Probability: {state.fill_probability:.2%}")
    print(f"  Category: {tracker.get_queue_position_category(state).value}")
    
    # Record some events
    for i in range(10):
        event = QueueEvent(
            timestamp_ms=base_ts + i * 100,
            event_type='cancel' if i % 2 == 0 else 'fill',
            volume=2.0,
            price=50000.0,
            side='bid'
        )
        tracker.record_event("BTC", event)
    
    # Estimate time to fill
    est_time = tracker.estimate_time_to_fill("BTC", 50000.0, 'bid', 5.0)
    print(f"\nEstimated Time to Fill: {est_time:.0f}ms" if est_time else "Fill unlikely")
    
    # Statistics
    stats = tracker.get_statistics()
    print(f"\nStatistics:")
    print(f"  Tracked Levels: {stats['tracked_price_levels']}")
    print(f"  Avg Fill Prob: {stats['average_fill_probability']:.2%}")
    print(f"  Phantom Fills: {stats['phantom_fill_count']}")
