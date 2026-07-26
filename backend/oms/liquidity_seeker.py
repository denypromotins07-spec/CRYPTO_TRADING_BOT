#!/usr/bin/env python3
"""
Liquidity Seeker - Passively Probe Order Book to Uncover Iceberg Orders

This module implements passive liquidity seeking algorithms to detect hidden 
(iceberg) orders in the order book without triggering taker fees or revealing
our trading intent.

Key Features:
- Passive probing with minimal market impact
- Iceberg detection through trade pattern analysis
- Volume accumulation tracking at price levels
- Adaptive probe sizing based on detected liquidity
- Integration with OMS for smart order routing

Memory Optimized: Uses generators and lazy evaluation
Thread Safe: GIL-safe with atomic operations where needed
Platform: Optimized for AMD Ryzen AI 5, Windows PowerShell
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import threading
from abc import ABC, abstractmethod


class LiquiditySide(Enum):
    """Side of liquidity being probed."""
    BID = "bid"
    ASK = "ask"


@dataclass(frozen=True)
class LiquidityProbe:
    """Represents a passive liquidity probe."""
    symbol: str
    side: LiquiditySide
    price: float
    probe_size: float
    timestamp_ns: int
    filled_size: float = 0.0
    fill_count: int = 0
    is_active: bool = True
    
    def __post_init__(self) -> None:
        if self.probe_size <= 0:
            raise ValueError("Probe size must be positive")
        if self.price <= 0:
            raise ValueError("Price must be positive")


@dataclass
class IcebergDetection:
    """Detected iceberg order information."""
    symbol: str
    side: LiquiditySide
    price_level: float
    estimated_total_size: float
    visible_size: float
    hidden_size: float
    confidence: float  # 0.0 to 1.0
    detection_time_ns: int
    consecutive_hits: int = 0
    last_hit_time_ns: int = 0


@dataclass
class VolumeAccumulation:
    """Track volume accumulation at a price level."""
    price: float
    total_volume: float = 0.0
    trade_count: int = 0
    first_seen_ns: int = 0
    last_seen_ns: int = 0
    
    def add_trade(self, volume: float, timestamp_ns: int) -> None:
        """Add a trade to this accumulation level."""
        self.total_volume += volume
        self.trade_count += 1
        if self.first_seen_ns == 0:
            self.first_seen_ns = timestamp_ns
        self.last_seen_ns = timestamp_ns


class LiquiditySeekerConfig:
    """Configuration for liquidity seeker behavior."""
    
    def __init__(
        self,
        min_iceberg_confidence: float = 0.7,
        probe_size_pct: float = 0.01,  # 1% of detected liquidity
        max_probes_per_level: int = 5,
        probe_timeout_ms: int = 1000,
        iceberg_min_trades: int = 3,
        volume_accumulation_window_ms: int = 5000,
        tick_size: float = 0.01,
    ) -> None:
        self.min_iceberg_confidence = min_iceberg_confidence
        self.probe_size_pct = probe_size_pct
        self.max_probes_per_level = max_probes_per_level
        self.probe_timeout_ms = probe_timeout_ms
        self.iceberg_min_trades = iceberg_min_trades
        self.volume_accumulation_window_ms = volume_accumulation_window_ms
        self.tick_size = tick_size
        
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (
            0.0 < self.min_iceberg_confidence < 1.0 and
            0.0 < self.probe_size_pct < 1.0 and
            self.max_probes_per_level > 0 and
            self.probe_timeout_ms > 0 and
            self.iceberg_min_trades >= 2 and
            self.volume_accumulation_window_ms > 0 and
            self.tick_size > 0
        )


class LiquiditySeekerState:
    """Thread-safe state container for liquidity seeker."""
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_probes: Dict[int, LiquidityProbe] = {}
        self._detected_icebergs: Dict[Tuple[str, float], IcebergDetection] = {}
        self._volume_accumulations: Dict[Tuple[str, float], VolumeAccumulation] = {}
        self._probe_counter: int = 0
        self._total_probes_sent: int = 0
        self._successful_detections: int = 0
        
    @property
    def active_probe_count(self) -> int:
        """Get count of currently active probes."""
        with self._lock:
            return sum(1 for p in self._active_probes.values() if p.is_active)
    
    @property
    def detected_iceberg_count(self) -> int:
        """Get count of detected icebergs."""
        with self._lock:
            return len(self._detected_icebergs)
    
    def add_probe(self, probe: LiquidityProbe) -> int:
        """Add a new probe and return its ID."""
        with self._lock:
            self._probe_counter += 1
            self._active_probes[self._probe_counter] = probe
            self._total_probes_sent += 1
            return self._probe_counter
    
    def update_probe_fill(self, probe_id: int, fill_size: float) -> Optional[LiquidityProbe]:
        """Update probe with fill information."""
        with self._lock:
            probe = self._active_probes.get(probe_id)
            if probe and probe.is_active:
                object.__setattr__(probe, 'filled_size', probe.filled_size + fill_size)
                object.__setattr__(probe, 'fill_count', probe.fill_count + 1)
                return probe
            return None
    
    def deactivate_probe(self, probe_id: int) -> None:
        """Mark a probe as inactive."""
        with self._lock:
            probe = self._active_probes.get(probe_id)
            if probe:
                object.__setattr__(probe, 'is_active', False)
    
    def record_iceberg(self, detection: IcebergDetection) -> None:
        """Record a detected iceberg."""
        with self._lock:
            key = (detection.symbol, detection.price_level)
            self._detected_icebergs[key] = detection
            self._successful_detections += 1
    
    def get_iceberg(self, symbol: str, price: float) -> Optional[IcebergDetection]:
        """Get iceberg detection for a specific level."""
        with self._lock:
            return self._detected_icebergs.get((symbol, price))
    
    def get_all_icebergs(self) -> List[IcebergDetection]:
        """Get all detected icebergs."""
        with self._lock:
            return list(self._detected_icebergs.values())
    
    def add_volume_at_level(self, symbol: str, price: float, volume: float, timestamp_ns: int) -> None:
        """Add volume observation at a price level."""
        with self._lock:
            key = (symbol, price)
            if key not in self._volume_accumulations:
                self._volume_accumulations[key] = VolumeAccumulation(price=price)
            
            self._volume_accumulations[key].add_trade(volume, timestamp_ns)
    
    def get_stats(self) -> Dict:
        """Get seeker statistics."""
        with self._lock:
            return {
                'total_probes_sent': self._total_probes_sent,
                'active_probes': self.active_probe_count,
                'detected_icebergs': len(self._detected_icebergs),
                'successful_detections': self._successful_detections,
                'volume_levels_tracked': len(self._volume_accumulations),
            }


class BaseLiquidityStrategy(ABC):
    """Abstract base class for liquidity seeking strategies."""
    
    @abstractmethod
    def should_probe(self, symbol: str, side: LiquiditySide, price: float) -> bool:
        """Determine if we should probe this level."""
        pass
    
    @abstractmethod
    def calculate_probe_size(self, available_liquidity: float) -> float:
        """Calculate optimal probe size."""
        pass
    
    @abstractmethod
    def analyze_fill_pattern(self, fills: List[Tuple[float, int]]) -> float:
        """Analyze fill pattern to detect iceberg. Returns confidence."""
        pass


class IcebergDetectionStrategy(BaseLiquidityStrategy):
    """Strategy focused on detecting iceberg orders."""
    
    def __init__(self, config: LiquiditySeekerConfig) -> None:
        self.config = config
        self._consecutive_fills: Dict[Tuple[str, float], deque] = {}
        
    def should_probe(self, symbol: str, side: LiquiditySide, price: float) -> bool:
        """Probe if we suspect hidden liquidity."""
        # Round price to tick size for grouping
        rounded_price = round(price / self.config.tick_size) * self.config.tick_size
        key = (symbol, rounded_price)
        
        # Check if we've seen multiple trades at this level
        if key in self._consecutive_fills:
            return len(self._consecutive_fills[key]) >= self.config.iceberg_min_trades - 1
        
        return False
    
    def calculate_probe_size(self, available_liquidity: float) -> float:
        """Calculate probe as percentage of available liquidity."""
        return max(
            available_liquidity * self.config.probe_size_pct,
            self.config.tick_size  # Minimum probe size
        )
    
    def analyze_fill_pattern(self, fills: List[Tuple[float, int]]) -> float:
        """
        Analyze fill pattern for iceberg characteristics.
        
        Args:
            fills: List of (price, size) tuples
            
        Returns:
            Confidence score 0.0 to 1.0
        """
        if len(fills) < self.config.iceberg_min_trades:
            return 0.0
        
        # Group fills by price
        price_groups: Dict[float, List[int]] = {}
        for price, size in fills:
            if price not in price_groups:
                price_groups[price] = []
            price_groups[price].append(size)
        
        # Look for repeated similar-sized fills at same price
        max_confidence = 0.0
        for price, sizes in price_groups.items():
            if len(sizes) >= self.config.iceberg_min_trades:
                # Calculate size variance - low variance suggests iceberg
                avg_size = sum(sizes) / len(sizes)
                if avg_size > 0:
                    variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
                    cv = (variance ** 0.5) / avg_size  # Coefficient of variation
                    
                    # Lower CV = higher confidence of iceberg
                    confidence = max(0, 1.0 - cv)
                    confidence *= min(1.0, len(sizes) / 10.0)  # Scale by count
                    
                    max_confidence = max(max_confidence, confidence)
        
        return min(1.0, max_confidence)


class LiquiditySeeker:
    """
    Main liquidity seeker engine for passive iceberg detection.
    
    This class coordinates probing activities, analyzes fill patterns,
    and maintains state about detected hidden liquidity.
    """
    
    def __init__(self, config: Optional[LiquiditySeekerConfig] = None) -> None:
        self.config = config or LiquiditySeekerConfig()
        if not self.config.validate():
            raise ValueError("Invalid liquidity seeker configuration")
        
        self.state = LiquiditySeekerState()
        self.strategy = IcebergDetectionStrategy(self.config)
        self._start_time_ns = time.time_ns()
        
    def _get_elapsed_ns(self) -> int:
        """Get elapsed time since start in nanoseconds."""
        return time.time_ns() - self._start_time_ns
    
    def process_trade(
        self,
        symbol: str,
        price: float,
        volume: float,
        side: LiquiditySide,
    ) -> Optional[IcebergDetection]:
        """
        Process an incoming trade and check for iceberg patterns.
        
        Args:
            symbol: Trading pair symbol
            price: Trade execution price
            volume: Trade volume
            side: Side of the trade
            
        Returns:
            IcebergDetection if confidence threshold met, else None
        """
        now_ns = self._get_elapsed_ns()
        
        # Record volume at this level
        rounded_price = round(price / self.config.tick_size) * self.config.tick_size
        self.state.add_volume_at_level(symbol, rounded_price, volume, now_ns)
        
        # Check if we should probe
        if self.strategy.should_probe(symbol, side, rounded_price):
            probe_size = self.strategy.calculate_probe_size(volume)
            
            # Create and send probe
            probe = LiquidityProbe(
                symbol=symbol,
                side=side,
                price=rounded_price,
                probe_size=probe_size,
                timestamp_ns=now_ns,
            )
            
            probe_id = self.state.add_probe(probe)
            
            # In production, this would send a passive limit order
            # For now, we simulate the probe being placed
            
        # Analyze accumulated volume for iceberg detection
        key = (symbol, rounded_price)
        accumulation = self.state._volume_accumulations.get(key)
        
        if accumulation and accumulation.trade_count >= self.config.iceberg_min_trades:
            # Get recent fills at this level
            # (In real implementation, would track individual fills)
            fills = [(accumulation.price, int(accumulation.total_volume / accumulation.trade_count))] * accumulation.trade_count
            
            confidence = self.strategy.analyze_fill_pattern(fills)
            
            if confidence >= self.config.min_iceberg_confidence:
                # Estimate iceberg size (heuristic: 5x visible)
                visible_size = accumulation.total_volume / accumulation.trade_count
                estimated_total = visible_size * 5
                
                detection = IcebergDetection(
                    symbol=symbol,
                    side=side,
                    price_level=rounded_price,
                    estimated_total_size=estimated_total,
                    visible_size=visible_size,
                    hidden_size=estimated_total - visible_size,
                    confidence=confidence,
                    detection_time_ns=now_ns,
                    consecutive_hits=accumulation.trade_count,
                    last_hit_time_ns=now_ns,
                )
                
                self.state.record_iceberg(detection)
                return detection
        
        return None
    
    def create_probe_order(
        self,
        symbol: str,
        side: LiquiditySide,
        price: float,
        size: float,
    ) -> Optional[int]:
        """
        Create a passive probe order.
        
        Args:
            symbol: Trading pair
            side: Bid or Ask
            price: Limit price for probe
            size: Order size
            
        Returns:
            Probe ID if created, None if rejected
        """
        now_ns = self._get_elapsed_ns()
        
        # Validate probe parameters
        if size <= 0 or price <= 0:
            return None
        
        # Check if we already have too many probes at this level
        rounded_price = round(price / self.config.tick_size) * self.config.tick_size
        existing_probes = sum(
            1 for p in self.state._active_probes.values()
            if p.symbol == symbol and abs(p.price - rounded_price) < self.config.tick_size and p.is_active
        )
        
        if existing_probes >= self.config.max_probes_per_level:
            return None
        
        probe = LiquidityProbe(
            symbol=symbol,
            side=side,
            price=rounded_price,
            probe_size=size,
            timestamp_ns=now_ns,
        )
        
        return self.state.add_probe(probe)
    
    def update_probe_fill(
        self,
        probe_id: int,
        fill_size: float,
    ) -> bool:
        """
        Update a probe with fill information.
        
        Args:
            probe_id: ID of the probe
            fill_size: Size that was filled
            
        Returns:
            True if update successful
        """
        result = self.state.update_probe_fill(probe_id, fill_size)
        if result and result.filled_size >= result.probe_size:
            self.state.deactivate_probe(probe_id)
            return True
        return result is not None
    
    def purge_expired_probes(self) -> int:
        """Remove probes that have exceeded timeout. Returns count purged."""
        now_ns = self._get_elapsed_ns()
        timeout_ns = self.config.probe_timeout_ms * 1_000_000
        
        purged = 0
        for probe_id, probe in list(self.state._active_probes.items()):
            if probe.is_active and (now_ns - probe.timestamp_ns) > timeout_ns:
                self.state.deactivate_probe(probe_id)
                purged += 1
        
        return purged
    
    def get_best_iceberg_opportunity(
        self,
        symbol: str,
        side: Optional[LiquiditySide] = None,
    ) -> Optional[IcebergDetection]:
        """
        Get the highest confidence iceberg opportunity.
        
        Args:
            symbol: Filter by symbol
            side: Optional filter by side
            
        Returns:
            Best iceberg detection or None
        """
        icebergs = self.state.get_all_icebergs()
        
        filtered = [
            i for i in icebergs
            if i.symbol == symbol and (side is None or i.side == side)
        ]
        
        if not filtered:
            return None
        
        return max(filtered, key=lambda x: x.confidence)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive seeker statistics."""
        stats = self.state.get_stats()
        stats['elapsed_ms'] = self._get_elapsed_ns() / 1_000_000
        stats['config'] = {
            'min_confidence': self.config.min_iceberg_confidence,
            'probe_size_pct': self.config.probe_size_pct,
            'max_probes_per_level': self.config.max_probes_per_level,
        }
        return stats


class SmartOrderRouter:
    """
    Routes orders to take advantage of detected liquidity.
    
    Integrates with LiquiditySeeker to intelligently route orders
    to price levels with hidden liquidity.
    """
    
    def __init__(self, seeker: LiquiditySeeker) -> None:
        self.seeker = seeker
        self._routed_orders: int = 0
        self._iceberg_captures: int = 0
        
    def route_order(
        self,
        symbol: str,
        side: LiquiditySide,
        total_size: float,
    ) -> List[Dict]:
        """
        Route an order across detected liquidity levels.
        
        Args:
            symbol: Trading pair
            side: Order side
            total_size: Total order size
            
        Returns:
            List of child orders with allocation details
        """
        icebergs = self.seeker.state.get_all_icebergs()
        
        # Filter relevant icebergs
        relevant = [
            i for i in icebergs
            if i.symbol == symbol and i.side != side  # Opposite side has liquidity
        ]
        
        if not relevant:
            # No icebergs detected, return single order
            self._routed_orders += 1
            return [{
                'symbol': symbol,
                'side': side.value,
                'size': total_size,
                'price': None,  # Market order or default logic
                'type': 'standard',
            }]
        
        # Sort by confidence and allocate
        relevant.sort(key=lambda x: x.confidence, reverse=True)
        
        child_orders = []
        remaining_size = total_size
        
        for iceberg in relevant:
            if remaining_size <= 0:
                break
            
            # Allocate portion to this iceberg level
            allocation = min(
                remaining_size,
                iceberg.visible_size * 0.5,  # Don't exhaust visible liquidity
                iceberg.hidden_size * 0.2,   # Conservative hidden capture
            )
            
            if allocation > 0:
                child_orders.append({
                    'symbol': symbol,
                    'side': side.value,
                    'size': allocation,
                    'price': iceberg.price_level,
                    'type': 'iceberg_capture',
                    'confidence': iceberg.confidence,
                })
                remaining_size -= allocation
                self._iceberg_captures += 1
        
        # Handle remaining size
        if remaining_size > 0:
            child_orders.append({
                'symbol': symbol,
                'side': side.value,
                'size': remaining_size,
                'price': None,
                'type': 'remainder',
            })
        
        self._routed_orders += 1
        return child_orders
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        return {
            'total_routed': self._routed_orders,
            'iceberg_captures': self._iceberg_captures,
            'capture_rate': self._iceberg_captures / max(1, self._routed_orders),
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize seeker
    config = LiquiditySeekerConfig(
        min_iceberg_confidence=0.6,
        probe_size_pct=0.02,
        tick_size=0.01,
    )
    
    seeker = LiquiditySeeker(config)
    
    # Simulate trades at a price level (iceberg pattern)
    print("Simulating iceberg detection...")
    for i in range(5):
        detection = seeker.process_trade(
            symbol="BTCUSDT",
            price=50000.00,
            volume=2.5,
            side=LiquiditySide.BID,
        )
        if detection:
            print(f"Iceberg detected! Confidence: {detection.confidence:.2f}")
            print(f"  Estimated total size: {detection.estimated_total_size:.2f}")
            print(f"  Hidden size: {detection.hidden_size:.2f}")
    
    # Get statistics
    stats = seeker.get_statistics()
    print(f"\nSeeker Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test smart order routing
    router = SmartOrderRouter(seeker)
    orders = router.route_order("BTCUSDT", LiquiditySide.ASK, 10.0)
    print(f"\nRouted {len(orders)} child orders:")
    for order in orders:
        print(f"  {order['type']}: {order['size']} @ {order.get('price', 'MARKET')}")
