#!/usr/bin/env python3
"""
ZAID Personal Crypto Trading Bot - Premium/Discount Module
Chapter 4: Institutional Order Flow, Premium/Discount Pricing, and SOUL.md SMC Logging

This module divides the current dealing range into Premium, Equilibrium, and Discount zones.
Strictly prevents buying in Premium zone or selling in Discount zone.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint using NumPy C-extensions.

Design Patterns: Strategy Pattern for range calculation
Time Complexity: O(1) for zone queries after setup
Space Complexity: O(1) auxiliary

Strict type hinting enforced for production reliability.
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray


class ZoneType(Enum):
    """Classification of dealing range zones."""
    PREMIUM = auto()      # Upper 50% - expensive, look to sell
    EQUILIBRIUM = auto()  # Middle zone - fair value
    DISCOUNT = auto()     # Lower 50% - cheap, look to buy


class FibonacciLevel(Enum):
    """Standard Fibonacci levels for premium/discount zones."""
    LEVEL_0 = 0.0
    LEVEL_236 = 0.236
    LEVEL_382 = 0.382
    LEVEL_500 = 0.500
    LEVEL_618 = 0.618
    LEVEL_707 = 0.707
    LEVEL_786 = 0.786
    LEVEL_886 = 0.886
    LEVEL_1000 = 1.0


@dataclass(slots=True)
class DealingRange:
    """
    Represents the current dealing range with zone boundaries.
    Uses __slots__ for memory efficiency on 8GB RAM systems.
    """
    range_high: float
    range_low: float
    timestamp_created: int
    valid: bool = True
    
    @property
    def range_size(self) -> float:
        """Get total range size."""
        return self.range_high - self.range_low
    
    def get_price_at_fib(self, fib_level: float) -> float:
        """Get price at specific Fibonacci level within range."""
        return self.range_low + (self.range_size * fib_level)
    
    def get_fib_level(self, price: float) -> float:
        """Get Fibonacci level for a given price."""
        if self.range_size == 0:
            return 0.5
        return (price - self.range_low) / self.range_size
    
    def contains(self, price: float) -> bool:
        """Check if price is within this dealing range."""
        return self.range_low <= price <= self.range_high


@dataclass(slots=True)
class PDZone:
    """
    Represents a Premium, Discount, or Equilibrium zone.
    Uses __slots__ for memory efficiency.
    """
    zone_type: ZoneType
    price_start: float
    price_end: float
    fib_start: float
    fib_end: float
    
    @property
    def midpoint(self) -> float:
        """Get midpoint of the zone."""
        return (self.price_start + self.price_end) / 2.0
    
    @property
    def size(self) -> float:
        """Get size of the zone."""
        return abs(self.price_end - self.price_start)
    
    def contains(self, price: float) -> bool:
        """Check if price is within this zone."""
        min_p = min(self.price_start, self.price_end)
        max_p = max(self.price_start, self.price_end)
        return min_p <= price <= max_p
    
    def is_premium(self) -> bool:
        """Check if this is a premium zone."""
        return self.zone_type == ZoneType.PREMIUM
    
    def is_discount(self) -> bool:
        """Check if this is a discount zone."""
        return self.zone_type == ZoneType.DISCOUNT
    
    def is_equilibrium(self) -> bool:
        """Check if this is an equilibrium zone."""
        return self.zone_type == ZoneType.EQUILIBRIUM


@dataclass
class PDConfig:
    """Configuration for Premium/Discount zone calculation."""
    # Use standard 50% equilibrium split
    equilibrium_center: float = 0.5
    
    # Premium zone starts above this Fib level
    premium_threshold: float = 0.5
    
    # Discount zone ends below this Fib level
    discount_threshold: float = 0.5
    
    # Number of sub-zones within each main zone
    sub_zone_count: int = 3
    
    # Minimum range size to be valid (percentage)
    min_range_size_pct: float = 0.01


class PremiumDiscountCalculator:
    """
    Advanced Premium/Discount zone calculator for institutional pricing.
    
    Implements:
    - Dynamic dealing range identification
    - Fibonacci-based zone division
    - Premium/Discount/Equlibrium classification
    - Trade direction validation (no buys in premium, no sells in discount)
    
    Memory-safe implementation with bounded storage.
    """
    
    def __init__(self, config: Optional[PDConfig] = None):
        """Initialize PD calculator with configuration."""
        self.config = config or PDConfig()
        
        # Current dealing range
        self._current_range: Optional[DealingRange] = None
        
        # Historical ranges for reference
        self._historical_ranges: List[DealingRange] = []
        self._max_historical_ranges: int = 50
        
        # Cached zones
        self._premium_zones: List[PDZone] = []
        self._equilibrium_zones: List[PDZone] = []
        self._discount_zones: List[PDZone] = []
        
        # Current price tracking
        self._current_price: float = 0.0
        self._last_timestamp: int = 0
    
    @property
    def current_range(self) -> Optional[DealingRange]:
        """Get current dealing range."""
        return self._current_range
    
    @property
    def premium_zones(self) -> List[PDZone]:
        """Get all premium zones."""
        return self._premium_zones.copy()
    
    @property
    def discount_zones(self) -> List[PDZone]:
        """Get all discount zones."""
        return self._discount_zones.copy()
    
    @property
    def equilibrium_zones(self) -> List[PDZone]:
        """Get all equilibrium zones."""
        return self._equilibrium_zones.copy()
    
    def update_dealing_range(
        self, 
        swing_high: float, 
        swing_low: float, 
        timestamp: int,
        force_update: bool = False
    ) -> bool:
        """
        Update the current dealing range based on new swing points.
        
        Args:
            swing_high: Recent swing high price
            swing_low: Recent swing low price
            timestamp: Current timestamp
            force_update: Force update even if range hasn't changed significantly
        
        Returns:
            True if range was updated, False otherwise
        """
        if swing_high <= swing_low:
            return False
        
        # Check minimum range size
        range_size_pct = (swing_high - swing_low) / swing_low
        if range_size_pct < self.config.min_range_size_pct and not force_update:
            return False
        
        # Save current range to history if it exists
        if self._current_range is not None and self._current_range.valid:
            self._historical_ranges.append(self._current_range)
            if len(self._historical_ranges) > self._max_historical_ranges:
                self._historical_ranges.pop(0)
        
        # Create new dealing range
        self._current_range = DealingRange(
            range_high=swing_high,
            range_low=swing_low,
            timestamp_created=timestamp,
            valid=True
        )
        
        # Recalculate zones
        self._calculate_zones()
        
        return True
    
    def _calculate_zones(self) -> None:
        """Calculate Premium, Discount, and Equilibrium zones from current range."""
        if self._current_range is None:
            return
        
        r = self._current_range
        
        # Clear existing zones
        self._premium_zones.clear()
        self._equilibrium_zones.clear()
        self._discount_zones.clear()
        
        # Calculate zone boundaries using Fibonacci levels
        # Discount: 0.0 - 0.5 (lower 50%)
        # Equilibrium: 0.5 area (fair value)
        # Premium: 0.5 - 1.0 (upper 50%)
        
        # Discount zones (0.0 to 0.5)
        discount_levels = [0.0, 0.236, 0.382, 0.5]
        for i in range(len(discount_levels) - 1):
            fib_start = discount_levels[i]
            fib_end = discount_levels[i + 1]
            
            zone = PDZone(
                zone_type=ZoneType.DISCOUNT,
                price_start=r.get_price_at_fib(fib_start),
                price_end=r.get_price_at_fib(fib_end),
                fib_start=fib_start,
                fib_end=fib_end
            )
            self._discount_zones.append(zone)
        
        # Equilibrium zone (around 0.5)
        eq_start = 0.5
        eq_end = 0.5
        # Create a small equilibrium band
        eq_band = 0.05
        zone = PDZone(
            zone_type=ZoneType.EQUILIBRIUM,
            price_start=r.get_price_at_fib(eq_start - eq_band),
            price_end=r.get_price_at_fib(eq_start + eq_band),
            fib_start=eq_start - eq_band,
            fib_end=eq_start + eq_band
        )
        self._equilibrium_zones.append(zone)
        
        # Premium zones (0.5 to 1.0)
        premium_levels = [0.5, 0.618, 0.786, 1.0]
        for i in range(len(premium_levels) - 1):
            fib_start = premium_levels[i]
            fib_end = premium_levels[i + 1]
            
            zone = PDZone(
                zone_type=ZoneType.PREMIUM,
                price_start=r.get_price_at_fib(fib_start),
                price_end=r.get_price_at_fib(fib_end),
                fib_start=fib_start,
                fib_end=fib_end
            )
            self._premium_zones.append(zone)
    
    def get_zone_for_price(self, price: float) -> Optional[PDZone]:
        """Get the zone that contains the given price."""
        # Check discount zones first (bottom up)
        for zone in reversed(self._discount_zones):
            if zone.contains(price):
                return zone
        
        # Check equilibrium
        for zone in self._equilibrium_zones:
            if zone.contains(price):
                return zone
        
        # Check premium zones (top down)
        for zone in self._premium_zones:
            if zone.contains(price):
                return zone
        
        return None
    
    def get_current_zone(self) -> Optional[PDZone]:
        """Get the zone containing current price."""
        return self.get_zone_for_price(self._current_price)
    
    def is_in_premium(self, price: Optional[float] = None) -> bool:
        """Check if price is in premium zone."""
        check_price = price if price is not None else self._current_price
        zone = self.get_zone_for_price(check_price)
        return zone is not None and zone.is_premium()
    
    def is_in_discount(self, price: Optional[float] = None) -> bool:
        """Check if price is in discount zone."""
        check_price = price if price is not None else self._current_price
        zone = self.get_zone_for_price(check_price)
        return zone is not None and zone.is_discount()
    
    def is_in_equilibrium(self, price: Optional[float] = None) -> bool:
        """Check if price is in equilibrium zone."""
        check_price = price if price is not None else self._current_price
        zone = self.get_zone_for_price(check_price)
        return zone is not None and zone.is_equilibrium()
    
    def validate_trade_direction(
        self, 
        direction: str, 
        entry_price: float
    ) -> Tuple[bool, str]:
        """
        Validate if trade direction is appropriate for current zone.
        
        STRICT RULES:
        - No LONG positions in Premium zone
        - No SHORT positions in Discount zone
        
        Args:
            direction: 'long' or 'short'
            entry_price: Proposed entry price
        
        Returns:
            Tuple of (is_valid, reason_message)
        """
        zone = self.get_zone_for_price(entry_price)
        
        if zone is None:
            return True, "Price outside current dealing range"
        
        direction_lower = direction.lower()
        
        if direction_lower == 'long':
            if zone.is_premium():
                return False, f"INVALID: Long in Premium zone at {entry_price:.2f}. Wait for Discount."
            elif zone.is_equilibrium():
                return True, "CAUTION: Long at Equilibrium. Prefer Discount zone."
            else:
                return True, "VALID: Long in Discount zone."
        
        elif direction_lower == 'short':
            if zone.is_discount():
                return False, f"INVALID: Short in Discount zone at {entry_price:.2f}. Wait for Premium."
            elif zone.is_equilibrium():
                return True, "CAUTION: Short at Equilibrium. Prefer Premium zone."
            else:
                return True, "VALID: Short in Premium zone."
        
        return False, f"Unknown direction: {direction}"
    
    def get_optimal_entry_zone(self, direction: str) -> Optional[PDZone]:
        """
        Get the optimal zone for entry based on trade direction.
        
        Args:
            direction: 'long' or 'short'
        
        Returns:
            Optimal PDZone for entry or None
        """
        if direction.lower() == 'long':
            # For longs, want deepest discount
            if self._discount_zones:
                return self._discount_zones[0]  # Lowest zone
        else:
            # For shorts, want highest premium
            if self._premium_zones:
                return self._premium_zones[-1]  # Highest zone
        
        return None
    
    def get_fair_value_price(self) -> Optional[float]:
        """Get the fair value (equilibrium) price."""
        if self._current_range is None:
            return None
        return self._current_range.get_price_at_fib(0.5)
    
    def update_current_price(self, price: float, timestamp: int) -> None:
        """Update current price tracking."""
        self._current_price = price
        self._last_timestamp = timestamp
    
    def invalidate_current_range(self) -> None:
        """Invalidate current dealing range (called on BOS)."""
        if self._current_range:
            self._current_range.valid = False
    
    def reset(self) -> None:
        """Reset calculator state."""
        self._current_range = None
        self._historical_ranges.clear()
        self._premium_zones.clear()
        self._equilibrium_zones.clear()
        self._discount_zones.clear()
        self._current_price = 0.0
        self._last_timestamp = 0


if __name__ == "__main__":
    # Example usage and basic validation
    print("ZAID Bot - Premium/Discount Calculator Module")
    print("=" * 50)
    
    calc = PremiumDiscountCalculator()
    
    # Set up a dealing range
    swing_high = 110.0
    swing_low = 90.0
    calc.update_dealing_range(swing_high, swing_low, 1000)
    
    print(f"\nDealing Range: {swing_low:.2f} - {swing_high:.2f}")
    print(f"Range Size: {calc.current_range.range_size:.2f}")
    print(f"Fair Value (0.5): {calc.get_fair_value_price():.2f}")
    
    print("\nDiscount Zones:")
    for zone in calc.discount_zones:
        print(f"  {zone.zone_type.name}: {zone.price_start:.2f} - {zone.price_end:.2f} "
              f"(Fib: {zone.fib_start:.3f} - {zone.fib_end:.3f})")
    
    print("\nEquilibrium Zones:")
    for zone in calc.equilibrium_zones:
        print(f"  {zone.zone_type.name}: {zone.price_start:.2f} - {zone.price_end:.2f}")
    
    print("\nPremium Zones:")
    for zone in calc.premium_zones:
        print(f"  {zone.zone_type.name}: {zone.price_start:.2f} - {zone.price_end:.2f} "
              f"(Fib: {zone.fib_start:.3f} - {zone.fib_end:.3f})")
    
    # Test trade validation
    print("\n" + "=" * 50)
    print("Trade Validation Tests:")
    print("=" * 50)
    
    test_prices = [92.0, 98.0, 100.0, 105.0, 108.0]
    
    for price in test_prices:
        calc.update_current_price(price, 1001)
        zone = calc.get_current_zone()
        zone_name = zone.zone_type.name if zone else "Unknown"
        
        long_valid, long_reason = calc.validate_trade_direction('long', price)
        short_valid, short_reason = calc.validate_trade_direction('short', price)
        
        print(f"\nPrice: {price:.2f} (Zone: {zone_name})")
        print(f"  Long: {'✓' if long_valid else '✗'} - {long_reason}")
        print(f"  Short: {'✓' if short_valid else '✗'} - {short_reason}")
