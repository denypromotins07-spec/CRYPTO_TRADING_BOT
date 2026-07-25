#!/usr/bin/env python3
"""
Iceberg Manager: Hidden Order Size Management
Hides large order sizes to prevent front-running and market impact.
Implements iceberg order detection and protection mechanisms.

Stage 13: Advanced Execution Algorithms
Target: Minimize market impact to secure 8k-20k INR/hour
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import threading


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class IcebergOrder:
    """Represents an iceberg order with visible and hidden quantities"""
    order_id: str
    side: OrderSide
    total_quantity: float
    visible_quantity: float
    filled_quantity: float
    price: float
    venue: str
    created_at_ns: int
    last_refresh_ns: int
    is_active: bool
    detection_risk_score: float  # 0.0 to 1.0, higher = more likely to be detected


@dataclass
class LiquiditySweepEvent:
    """Detected liquidity sweep event"""
    timestamp_ns: int
    venue: str
    side: OrderSide
    swept_volume: float
    price_impact_bps: float
    duration_us: int
    is_spoofing: bool


class IcebergManager:
    """
    Manages iceberg orders to hide large order sizes.
    Detects liquidity sweeps and spoofing attacks.
    Instantly pulls iceberg orders when threats are detected.
    
    Features:
    - Dynamic visible quantity adjustment
    - Refresh rate optimization
    - Liquidity sweep detection
    - Spoofing identification
    - Emergency order cancellation
    - Thread-safe state management
    """
    
    def __init__(
        self,
        default_visible_pct: float = 0.1,
        min_visible_quantity: float = 100.0,
        max_visible_quantity: float = 10000.0,
        refresh_interval_ms: int = 5000,
        sweep_detection_threshold: float = 5.0,  # Standard deviations
        spoofing_time_threshold_us: int = 100000,  # 100ms
    ):
        self.default_visible_pct = default_visible_pct
        self.min_visible_quantity = min_visible_quantity
        self.max_visible_quantity = max_visible_quantity
        self.refresh_interval_ms = refresh_interval_ms
        self.sweep_detection_threshold = sweep_detection_threshold
        self.spoofing_time_threshold_us = spoofing_time_threshold_us
        
        # Active iceberg orders
        self._iceberg_orders: Dict[str, IcebergOrder] = {}
        
        # Historical volume baseline for sweep detection
        self._volume_history: List[Tuple[int, float]] = []  # (timestamp_ns, volume)
        self._volume_baseline_mean: float = 0.0
        self._volume_baseline_std: float = 0.0
        
        # Recent order book events for spoofing detection
        self._order_book_events: List[Dict] = []
        
        # Detected sweep events
        self._sweep_events: List[LiquiditySweepEvent] = []
        
        # Threat state
        self._threat_detected: bool = False
        self._last_threat_time_ns: int = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def create_iceberg_order(
        self,
        order_id: str,
        side: OrderSide,
        total_quantity: float,
        price: float,
        venue: str,
        visible_pct: Optional[float] = None,
    ) -> IcebergOrder:
        """Create a new iceberg order"""
        with self._lock:
            visible_pct = visible_pct or self.default_visible_pct
            
            visible_qty = max(
                self.min_visible_quantity,
                min(
                    self.max_visible_quantity,
                    total_quantity * visible_pct
                )
            )
            
            order = IcebergOrder(
                order_id=order_id,
                side=side,
                total_quantity=total_quantity,
                visible_quantity=visible_qty,
                filled_quantity=0.0,
                price=price,
                venue=venue,
                created_at_ns=time.time_ns(),
                last_refresh_ns=time.time_ns(),
                is_active=True,
                detection_risk_score=0.0,
            )
            
            self._iceberg_orders[order_id] = order
            return order
    
    def record_fill(self, order_id: str, fill_quantity: float) -> Optional[IcebergOrder]:
        """Record a fill against an iceberg order"""
        with self._lock:
            if order_id not in self._iceberg_orders:
                return None
            
            order = self._iceberg_orders[order_id]
            order.filled_quantity += fill_quantity
            
            # Check if fully filled
            if order.filled_quantity >= order.total_quantity:
                order.is_active = False
                return order
            
            # Refresh visible quantity if needed
            remaining = order.total_quantity - order.filled_quantity
            if order.visible_quantity > remaining:
                order.visible_quantity = remaining
            
            order.last_refresh_ns = time.time_ns()
            
            # Update detection risk
            self._update_detection_risk(order)
            
            return order
    
    def _update_detection_risk(self, order: IcebergOrder) -> None:
        """Update the detection risk score for an order"""
        # Factors that increase detection risk:
        # 1. Repeated fills at same price level
        # 2. Large total quantity relative to market
        # 3. Long time on book without full fill
        
        elapsed_ms = (time.time_ns() - order.created_at_ns) / 1_000_000
        fill_rate = order.filled_quantity / order.total_quantity if order.total_quantity > 0 else 0
        
        # Simple risk model
        time_risk = min(1.0, elapsed_ms / 60000)  # Max risk after 1 minute
        fill_pattern_risk = fill_rate * 0.5  # Higher fill rate = higher risk
        
        order.detection_risk_score = (time_risk + fill_pattern_risk) / 2.0
    
    def adjust_visible_quantity(
        self, 
        order_id: str, 
        new_visible_pct: Optional[float] = None
    ) -> Optional[float]:
        """Dynamically adjust visible quantity based on market conditions"""
        with self._lock:
            if order_id not in self._iceberg_orders:
                return None
            
            order = self._iceberg_orders[order_id]
            remaining = order.total_quantity - order.filled_quantity
            
            if new_visible_pct is not None:
                # Use provided percentage
                visible_qty = max(
                    self.min_visible_quantity,
                    min(self.max_visible_quantity, remaining * new_visible_pct)
                )
            elif self._threat_detected:
                # Reduce visibility during threats
                visible_qty = max(
                    self.min_visible_quantity,
                    remaining * (self.default_visible_pct * 0.5)
                )
            elif order.detection_risk_score > 0.7:
                # High detection risk - reduce visibility
                visible_qty = max(
                    self.min_visible_quantity,
                    remaining * (self.default_visible_pct * 0.7)
                )
            else:
                # Normal operation
                visible_qty = order.visible_quantity
            
            order.visible_quantity = min(visible_qty, remaining)
            order.last_refresh_ns = time.time_ns()
            
            return order.visible_quantity
    
    def add_volume_sample(self, volume: float) -> None:
        """Add volume sample for baseline calculation"""
        with self._lock:
            self._volume_history.append((time.time_ns(), volume))
            
            # Keep last 1000 samples
            if len(self._volume_history) > 1000:
                self._volume_history.pop(0)
            
            # Update baseline statistics
            if len(self._volume_history) >= 10:
                volumes = [v for _, v in self._volume_history]
                self._volume_baseline_mean = np.mean(volumes)
                self._volume_baseline_std = np.std(volumes)
    
    def detect_liquidity_sweep(
        self,
        current_volume: float,
        side: OrderSide,
        venue: str,
        price_impact_bps: float,
        duration_us: int,
    ) -> Optional[LiquiditySweepEvent]:
        """Detect potential liquidity sweep"""
        with self._lock:
            if self._volume_baseline_std == 0:
                return None
            
            # Calculate z-score of current volume
            z_score = (current_volume - self._volume_baseline_mean) / self._volume_baseline_std
            
            if z_score > self.sweep_detection_threshold:
                # Determine if this is likely spoofing
                is_spoofing = (
                    duration_us < self.spoofing_time_threshold_us and
                    price_impact_bps > 50  # Large impact that quickly reverses
                )
                
                event = LiquiditySweepEvent(
                    timestamp_ns=time.time_ns(),
                    venue=venue,
                    side=side,
                    swept_volume=current_volume,
                    price_impact_bps=price_impact_bps,
                    duration_us=duration_us,
                    is_spoofing=is_spoofing,
                )
                
                self._sweep_events.append(event)
                self._threat_detected = True
                self._last_threat_time_ns = event.timestamp_ns
                
                # Keep only recent sweep events
                if len(self._sweep_events) > 100:
                    self._sweep_events.pop(0)
                
                return event
            
            return None
    
    def pull_all_icebergs(self) -> List[str]:
        """Emergency pull all iceberg orders"""
        with self._lock:
            pulled_orders = []
            
            for order_id, order in self._iceberg_orders.items():
                if order.is_active:
                    order.is_active = False
                    pulled_orders.append(order_id)
            
            self._threat_detected = True
            self._last_threat_time_ns = time.time_ns()
            
            return pulled_orders
    
    def pull_iceberg_for_venue(self, venue: str) -> List[str]:
        """Pull all iceberg orders for a specific venue"""
        with self._lock:
            pulled_orders = []
            
            for order_id, order in self._iceberg_orders.items():
                if order.is_active and order.venue == venue:
                    order.is_active = False
                    pulled_orders.append(order_id)
            
            return pulled_orders
    
    def is_threat_detected(self) -> bool:
        """Check if a threat is currently detected"""
        with self._lock:
            if not self._threat_detected:
                return False
            
            # Clear threat after cooldown period (5 seconds)
            elapsed_ms = (time.time_ns() - self._last_threat_time_ns) / 1_000_000
            if elapsed_ms > 5000:
                self._threat_detected = False
                return False
            
            return True
    
    def get_recent_sweeps(self, limit: int = 10) -> List[LiquiditySweepEvent]:
        """Get recent liquidity sweep events"""
        with self._lock:
            return self._sweep_events[-limit:]
    
    def get_iceberg_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an iceberg order"""
        with self._lock:
            if order_id not in self._iceberg_orders:
                return None
            
            order = self._iceberg_orders[order_id]
            remaining = order.total_quantity - order.filled_quantity
            
            return {
                "order_id": order.order_id,
                "side": order.side.value,
                "total_quantity": order.total_quantity,
                "visible_quantity": order.visible_quantity,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": remaining,
                "fill_progress": order.filled_quantity / order.total_quantity if order.total_quantity > 0 else 0,
                "detection_risk": order.detection_risk_score,
                "is_active": order.is_active,
                "venue": order.venue,
            }
    
    def reset(self) -> None:
        """Reset manager state"""
        with self._lock:
            self._iceberg_orders.clear()
            self._volume_history.clear()
            self._sweep_events.clear()
            self._threat_detected = False
            self._last_threat_time_ns = 0
            self._volume_baseline_mean = 0.0
            self._volume_baseline_std = 0.0


if __name__ == "__main__":
    # Test iceberg manager
    manager = IcebergManager(
        default_visible_pct=0.1,
        min_visible_quantity=100.0,
    )
    
    # Create iceberg order
    order = manager.create_iceberg_order(
        order_id="IBG-001",
        side=OrderSide.BUY,
        total_quantity=10000,
        price=50000.0,
        venue="binance_spot",
    )
    print(f"Created iceberg: visible={order.visible_quantity}, total={order.total_quantity}")
    
    # Simulate fills
    for i in range(5):
        manager.record_fill("IBG-001", 500)
        status = manager.get_iceberg_status("IBG-001")
        print(f"Fill {i+1}: filled={status['filled_quantity']}, risk={status['detection_risk']:.3f}")
    
    # Add volume samples
    for _ in range(20):
        manager.add_volume_sample(1000 + np.random.randn() * 100)
    
    # Simulate liquidity sweep detection
    sweep = manager.detect_liquidity_sweep(
        current_volume=5000,  # Much higher than baseline
        side=OrderSide.SELL,
        venue="binance_spot",
        price_impact_bps=30,
        duration_us=50000,
    )
    
    if sweep:
        print(f"\nLiquidity sweep detected! Volume={sweep.swept_volume}, Spoofing={sweep.is_spoofing}")
        
        # Pull icebergs
        pulled = manager.pull_all_icebergs()
        print(f"Pulled orders: {pulled}")
        print(f"Threat detected: {manager.is_threat_detected()}")
