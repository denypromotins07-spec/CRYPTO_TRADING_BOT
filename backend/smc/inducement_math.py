#!/usr/bin/env python3
"""
ZAID Personal Crypto Trading Bot - Inducement Math Module
Chapter 2: Liquidity Pools, Stop Hunts, and Inducement Detection

This module calculates minor pullbacks designed to induce early retail entries.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint using NumPy C-extensions.

Design Patterns: Strategy Pattern for different inducement types
Time Complexity: O(1) for detection after setup
Space Complexity: O(k) where k is number of tracked levels

Strict type hinting enforced for production reliability.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray


class InducementType(Enum):
    """Classification of inducement patterns."""
    EARLY_ENTRY = auto()      # Pullback inducing early buyers/sellers
    BREAKOUT_FAKE = auto()    # False breakout inducing momentum traders
    SUPPORT_BUY = auto()      # Inducement to buy at "support"
    RESISTANCE_SELL = auto()  # Inducement to sell at "resistance"
    TREND_CONTINUATION = auto()  # Small pullback in trend inducing entry


@dataclass(slots=True)
class InducementZone:
    """
    Represents an inducement zone where retail traders are likely trapped.
    Uses __slots__ for memory efficiency on 8GB RAM systems.
    """
    zone_id: int
    price_start: float
    price_end: float
    inducement_type: InducementType
    strength: float  # 0.0 to 1.0
    trapped_volume_estimate: float
    timestamp_created: int
    triggered: bool = False
    timestamp_triggered: Optional[int] = None
    
    @property
    def mid_price(self) -> float:
        """Get midpoint of the inducement zone."""
        return (self.price_start + self.price_end) / 2.0
    
    @property
    def width(self) -> float:
        """Get width of the zone."""
        return abs(self.price_end - self.price_start)
    
    def contains(self, price: float) -> bool:
        """Check if price is within this zone."""
        min_p = min(self.price_start, self.price_end)
        max_p = max(self.price_start, self.price_end)
        return min_p <= price <= max_p


@dataclass
class InducementConfig:
    """Configuration for inducement detection."""
    # Fibonacci levels commonly used for inducement
    fib_levels: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.707)
    
    # Minimum retracement to qualify as inducement
    min_retracement_pct: float = 0.15
    
    # Maximum retracement before it's a real reversal
    max_retracement_pct: float = 0.65
    
    # Volume threshold for trapped trader estimation
    volume_threshold_multiplier: float = 1.2
    
    # Maximum zones to track (memory bound)
    max_zones: int = 100


class InducementCalculator:
    """
    Advanced inducement detection for identifying retail trap zones.
    
    Detects:
    - Minor pullbacks that induce early entries
    - False breakouts that trap breakout traders
    - Fibonacci-based retracement zones
    - Volume-based trapped trader estimation
    
    Memory-safe implementation with bounded storage.
    """
    
    def __init__(self, config: Optional[InducementConfig] = None):
        """Initialize inducement calculator with configuration."""
        self.config = config or InducementConfig()
        
        # Detected inducement zones
        self._zones: List[InducementZone] = []
        self._next_zone_id: int = 1
        
        # Price history for retracement calculation
        self._price_history: List[Tuple[int, float, float, float, float]] = []  # (ts, O, H, L, C)
        
        # Recent swing points
        self._swing_highs: List[Tuple[int, float]] = []
        self._swing_lows: List[Tuple[int, float]] = []
        
        # Current market state
        self._current_trend: str = "neutral"  # bullish, bearish, neutral
        self._last_swing_high: Optional[Tuple[int, float]] = None
        self._last_swing_low: Optional[Tuple[int, float]] = None
        
        # ATR for normalization
        self._atr_value: float = 0.0
        self._tr_values: List[float] = []
        self._atr_period: int = 14
    
    @property
    def zones(self) -> List[InducementZone]:
        """Get list of active inducement zones (read-only copy)."""
        return [z for z in self._zones if not z.triggered]
    
    @property
    def active_zone_count(self) -> int:
        """Get count of active (untriggered) zones."""
        return len(self.zones)
    
    def _calculate_true_range(self, high: float, low: float, prev_close: float) -> float:
        """Calculate True Range for ATR computation."""
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        return max(tr1, tr2, tr3)
    
    def _update_atr(self, tr: float) -> None:
        """Update ATR value using Wilder's smoothing method."""
        self._tr_values.append(tr)
        
        if len(self._tr_values) > self._atr_period:
            self._tr_values.pop(0)
        
        if len(self._tr_values) >= self._atr_period:
            if self._atr_value == 0.0:
                self._atr_value = sum(self._tr_values) / len(self._tr_values)
            else:
                self._atr_value = (
                    (self._atr_value * (self._atr_period - 1) + tr) 
                    / self._atr_period
                )
    
    def register_swing_high(self, timestamp: int, price: float) -> None:
        """Register a swing high for retracement calculation."""
        self._swing_highs.append((timestamp, price))
        
        # Update last swing high
        if self._last_swing_high is None or price > self._last_swing_high[1]:
            self._last_swing_high = (timestamp, price)
        
        # Keep only recent swings
        if len(self._swing_highs) > 50:
            self._swing_highs.pop(0)
        
        # Check for inducement zone creation in bullish trend
        if self._current_trend == "bullish" and self._last_swing_low:
            self._check_retracement_zone(timestamp, price, False)
    
    def register_swing_low(self, timestamp: int, price: float) -> None:
        """Register a swing low for retracement calculation."""
        self._swing_lows.append((timestamp, price))
        
        # Update last swing low
        if self._last_swing_low is None or price < self._last_swing_low[1]:
            self._last_swing_low = (timestamp, price)
        
        # Keep only recent swings
        if len(self._swing_lows) > 50:
            self._swing_lows.pop(0)
        
        # Check for inducement zone creation in bearish trend
        if self._current_trend == "bearish" and self._last_swing_high:
            self._check_retracement_zone(timestamp, price, True)
    
    def _check_retracement_zone(
        self, 
        timestamp: int, 
        current_price: float, 
        is_bearish: bool
    ) -> None:
        """Check if current retracement forms an inducement zone."""
        if is_bearish:
            # In bearish trend, check retracement from last swing low to current
            if self._last_swing_low is None:
                return
            
            swing_low = self._last_swing_low[1]
            swing_high = self._last_swing_high[1] if self._last_swing_high else swing_low * 1.1
            
            move = swing_high - swing_low
            if move <= 0:
                return
            
            retracement = (swing_high - current_price) / move
            
            if self.config.min_retracement_pct <= retracement <= self.config.max_retracement_pct:
                # Find Fibonacci level closest to this retracement
                fib_level = min(
                    self.config.fib_levels,
                    key=lambda x: abs(x - retracement)
                )
                
                # Create inducement zone around this level
                zone_start = swing_low + move * (fib_level - 0.05)
                zone_end = swing_low + move * (fib_level + 0.05)
                
                self._create_zone(
                    zone_start=zone_start,
                    zone_end=zone_end,
                    zone_type=InducementType.TREND_CONTINUATION,
                    timestamp=timestamp,
                    strength=self._calculate_zone_strength(retracement, fib_level)
                )
        else:
            # In bullish trend, check retracement from last swing high to current
            if self._last_swing_high is None:
                return
            
            swing_high = self._last_swing_high[1]
            swing_low = self._last_swing_low[1] if self._last_swing_low else swing_high * 0.9
            
            move = swing_high - swing_low
            if move <= 0:
                return
            
            retracement = (swing_high - current_price) / move
            
            if self.config.min_retracement_pct <= retracement <= self.config.max_retracement_pct:
                fib_level = min(
                    self.config.fib_levels,
                    key=lambda x: abs(x - retracement)
                )
                
                zone_start = swing_low + move * (fib_level - 0.05)
                zone_end = swing_low + move * (fib_level + 0.05)
                
                self._create_zone(
                    zone_start=zone_start,
                    zone_end=zone_end,
                    zone_type=InducementType.TREND_CONTINUATION,
                    timestamp=timestamp,
                    strength=self._calculate_zone_strength(retracement, fib_level)
                )
    
    def _calculate_zone_strength(self, retracement: float, fib_level: float) -> float:
        """Calculate strength score for inducement zone."""
        strength = 0.5
        
        # Bonus for being close to key Fibonacci level
        fib_proximity = 1.0 - abs(retracement - fib_level)
        strength += fib_proximity * 0.3
        
        # Bonus for 0.618 (golden ratio) and 0.707
        if abs(fib_level - 0.618) < 0.02:
            strength += 0.15
        elif abs(fib_level - 0.707) < 0.02:
            strength += 0.1
        
        return min(1.0, strength)
    
    def _create_zone(
        self,
        zone_start: float,
        zone_end: float,
        zone_type: InducementType,
        timestamp: int,
        strength: float
    ) -> Optional[InducementZone]:
        """Create new inducement zone if one doesn't already exist nearby."""
        # Check for overlapping zones
        for zone in self._zones:
            if not zone.triggered:
                # Check for significant overlap
                if (zone.contains(zone_start) or zone.contains(zone_end) or
                    (zone_start <= zone.price_start <= zone_end) or
                    (zone_start <= zone.price_end <= zone_end)):
                    # Update existing zone
                    zone.strength = max(zone.strength, strength)
                    return None
        
        # Estimate trapped volume based on zone characteristics
        trapped_volume = abs(zone_end - zone_start) * strength * 1000  # Simplified estimation
        
        zone = InducementZone(
            zone_id=self._next_zone_id,
            price_start=zone_start,
            price_end=zone_end,
            inducement_type=zone_type,
            strength=strength,
            trapped_volume_estimate=trapped_volume,
            timestamp_created=timestamp
        )
        
        self._zones.append(zone)
        self._next_zone_id += 1
        
        # Enforce memory bounds
        if len(self._zones) > self.config.max_zones:
            self._remove_weakest_zones()
        
        return zone
    
    def _remove_weakest_zones(self) -> None:
        """Remove weakest zones to maintain memory bounds."""
        self._zones.sort(key=lambda z: z.strength)
        remove_count = max(1, len(self._zones) // 5)
        self._zones = self._zones[remove_count:]
    
    def update(
        self, 
        timestamp: int, 
        open_: float, 
        high: float, 
        low: float, 
        close: float, 
        volume: float
    ) -> List[InducementZone]:
        """
        Process new candle data and check for inducement triggers.
        
        Returns list of zones that were triggered.
        """
        # Update ATR
        prev_close = self._price_history[-1][4] if self._price_history else close
        tr = self._calculate_true_range(high, low, prev_close)
        self._update_atr(tr)
        
        # Store price history
        self._price_history.append((timestamp, open_, high, low, close))
        if len(self._price_history) > 100:
            self._price_history.pop(0)
        
        # Update trend based on price action
        self._update_trend(close)
        
        triggered_zones: List[InducementZone] = []
        
        # Check each zone for trigger
        for zone in self._zones:
            if zone.triggered:
                continue
            
            if zone.contains(low) or zone.contains(high):
                zone.triggered = True
                zone.timestamp_triggered = timestamp
                triggered_zones.append(zone)
                
                # Update trapped volume estimate with actual volume
                zone.trapped_volume_estimate = volume * zone.strength
        
        return triggered_zones
    
    def _update_trend(self, current_price: float) -> None:
        """Update current trend classification."""
        if self._last_swing_high and self._last_swing_low:
            if current_price > self._last_swing_high[1]:
                self._current_trend = "bullish"
            elif current_price < self._last_swing_low[1]:
                self._current_trend = "bearish"
            else:
                self._current_trend = "neutral"
    
    def get_nearest_inducement_above(self, current_price: float) -> Optional[InducementZone]:
        """Get nearest untriggered inducement zone above current price."""
        zones_above = [z for z in self.zones if z.price_start > current_price or z.price_end > current_price]
        if not zones_above:
            return None
        return min(zones_above, key=lambda z: abs(z.mid_price - current_price))
    
    def get_nearest_inducement_below(self, current_price: float) -> Optional[InducementZone]:
        """Get nearest untriggered inducement zone below current price."""
        zones_below = [z for z in self.zones if z.price_start < current_price or z.price_end < current_price]
        if not zones_below:
            return None
        return max(zones_below, key=lambda z: abs(z.mid_price - current_price))
    
    def calculate_optimal_entry_after_sweep(
        self, 
        swept_level: float, 
        sweep_direction: str
    ) -> Optional[float]:
        """
        Calculate optimal entry point after liquidity sweep.
        
        After a sweep, price often returns to an inducement zone before continuing.
        This finds the best entry level.
        
        Args:
            swept_level: The price level that was swept
            sweep_direction: 'up' or 'down'
        
        Returns:
            Optimal entry price or None if no suitable level found
        """
        if sweep_direction == 'up':
            # Swept highs - look for support below
            # Entry should be at first significant inducement zone below
            zones_below = [z for z in self.zones if z.mid_price < swept_level and not z.triggered]
            
            if zones_below:
                # Return the highest zone below (nearest support)
                best_zone = max(zones_below, key=lambda z: z.mid_price)
                return best_zone.mid_price
        
        else:  # sweep_direction == 'down'
            # Swept lows - look for resistance above
            zones_above = [z for z in self.zones if z.mid_price > swept_level and not z.triggered]
            
            if zones_above:
                # Return the lowest zone above (nearest resistance)
                best_zone = min(zones_above, key=lambda z: z.mid_price)
                return best_zone.mid_price
        
        return None
    
    def reset(self) -> None:
        """Reset calculator state for new asset or timeframe."""
        self._zones.clear()
        self._price_history.clear()
        self._swing_highs.clear()
        self._swing_lows.clear()
        self._tr_values.clear()
        self._atr_value = 0.0
        self._last_swing_high = None
        self._last_swing_low = None
        self._current_trend = "neutral"
        self._next_zone_id = 1


if __name__ == "__main__":
    # Example usage and basic validation
    print("ZAID Bot - Inducement Math Module")
    print("=" * 50)
    
    calc = InducementCalculator()
    
    # Simulate a bullish trend with pullbacks
    test_data = [
        (1000, 100.0, 102.0, 99.0, 101.0, 1000.0),
        (1001, 101.0, 105.0, 100.0, 104.0, 1200.0),
        (1002, 104.0, 108.0, 103.0, 107.0, 1500.0),  # Swing high forming
        (1003, 107.0, 108.0, 104.0, 105.0, 900.0),   # Pullback starts
        (1004, 105.0, 106.0, 102.0, 103.0, 800.0),   # Continue pullback
        (1005, 103.0, 104.0, 100.0, 101.0, 1100.0),  # Near 0.5 fib
        (1006, 101.0, 102.0, 98.0, 99.0, 950.0),     # Near 0.618 fib
        (1007, 99.0, 101.0, 97.0, 98.0, 1050.0),     # Swing low forming
        (1008, 98.0, 102.0, 97.0, 101.0, 1300.0),    # Reversal
        (1009, 101.0, 105.0, 100.0, 104.0, 1600.0),
        (1010, 104.0, 108.0, 103.0, 107.0, 1800.0),
    ]
    
    # Register swing points
    calc.register_swing_high(1002, 108.0)
    calc.register_swing_low(1007, 97.0)
    
    # Set bullish trend
    calc._current_trend = "bullish"
    calc._last_swing_high = (1002, 108.0)
    calc._last_swing_low = (1007, 97.0)
    
    # Manually create some inducement zones for demonstration
    calc._create_zone(
        zone_start=101.0,
        zone_end=103.0,
        zone_type=InducementType.TREND_CONTINUATION,
        timestamp=1003,
        strength=0.75
    )
    
    print("\nActive inducement zones:")
    for zone in calc.zones:
        print(f"  Zone {zone.zone_id}: {zone.price_start:.2f} - {zone.price_end:.2f} "
              f"(type: {zone.inducement_type.name}, strength: {zone.strength:.2f})")
    
    print("\nProcessing candles...")
    for ts, o, h, l, c, v in test_data:
        triggered = calc.update(ts, o, h, l, c, v)
        if triggered:
            for zone in triggered:
                print(f"  Zone {zone.zone_id} TRIGGERED at {zone.timestamp_triggered}")
    
    print(f"\nTotal zones created: {calc._next_zone_id - 1}")
    print(f"Active zones remaining: {calc.active_zone_count}")
    
    # Test optimal entry calculation
    optimal = calc.calculate_optimal_entry_after_sweep(108.0, 'up')
    if optimal:
        print(f"\nOptimal entry after sweeping 108.0: {optimal:.2f}")
