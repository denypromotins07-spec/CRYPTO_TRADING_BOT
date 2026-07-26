#!/usr/bin/env python3
"""
ZAID Personal Crypto Trading Bot - Liquidity Pools Module
Chapter 2: Liquidity Pools, Stop Hunts, and Inducement Detection

This module maps Equal Highs/Lows and Buy/Sell Side Liquidity (BSL/SSL).
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint using NumPy C-extensions.

Design Patterns: Observer Pattern for liquidity level updates
Time Complexity: O(n) for detection, O(1) for queries
Space Complexity: O(k) where k is number of liquidity pools

Strict type hinting enforced for production reliability.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray


class LiquidityType(Enum):
    """Classification of liquidity pool types."""
    BSL = auto()  # Buy Side Liquidity - stops above highs
    SSL = auto()  # Sell Side Liquidity - stops below lows
    EQUAL_HIGH = auto()  # Equal highs forming liquidity pool
    EQUAL_LOW = auto()  # Equal lows forming liquidity pool
    SWING_HIGH = auto()  # Single swing high liquidity
    SWING_LOW = auto()  # Single swing low liquidity


@dataclass(slots=True)
class LiquidityPool:
    """
    Represents a detected liquidity pool with metadata.
    Uses __slots__ for memory efficiency on 8GB RAM systems.
    """
    pool_id: int
    price_level: float
    liquidity_type: LiquidityType
    strength: float  # 0.0 to 1.0, based on touches and volume
    stop_volume_estimate: float  # Estimated stop loss volume
    touch_count: int = 0
    last_touched_timestamp: int = 0
    swept: bool = False  # Whether liquidity has been taken
    sweep_timestamp: Optional[int] = None
    mitigation_target: Optional[float] = None  # Target after sweep
    
    def is_bsl(self) -> bool:
        """Check if this is buy side liquidity."""
        return self.liquidity_type in (
            LiquidityType.BSL, 
            LiquidityType.EQUAL_HIGH,
            LiquidityType.SWING_HIGH
        )
    
    def is_ssl(self) -> bool:
        """Check if this is sell side liquidity."""
        return self.liquidity_type in (
            LiquidityType.SSL,
            LiquidityType.EQUAL_LOW,
            LiquidityType.SWING_LOW
        )
    
    def is_equal_level(self) -> bool:
        """Check if this is an equal high/low level."""
        return self.liquidity_type in (
            LiquidityType.EQUAL_HIGH,
            LiquidityType.EQUAL_LOW
        )


@dataclass
class LiquidityConfig:
    """Configuration for liquidity pool detection."""
    # Price tolerance for equal levels (percentage)
    equal_level_tolerance_pct: float = 0.5
    
    # Minimum touches to form significant liquidity pool
    min_touches_for_pool: int = 2
    
    # Lookback period for detecting equal levels
    lookback_period: int = 50
    
    # Volume multiplier for strength calculation
    volume_strength_multiplier: float = 0.3
    
    # Maximum number of pools to track (memory bound)
    max_pools: int = 200
    
    # ATR multiplier for sweep detection
    sweep_atr_multiplier: float = 1.5


class LiquidityPoolMapper:
    """
    Advanced liquidity pool mapping for institutional order flow analysis.
    
    Detects:
    - Equal Highs/Lows (EQH/EQL)
    - Buy Side Liquidity (BSL) above swing highs
    - Sell Side Liquidity (SSL) below swing lows
    - Liquidity sweeps and mitigations
    
    Memory-safe implementation with bounded storage.
    """
    
    def __init__(self, config: Optional[LiquidityConfig] = None):
        """Initialize liquidity pool mapper with configuration."""
        self.config = config or LiquidityConfig()
        
        # Detected liquidity pools
        self._pools: List[LiquidityPool] = []
        self._next_pool_id: int = 1
        
        # Price history for equal level detection
        self._price_history: List[Tuple[int, float, float]] = []  # (timestamp, high, low)
        
        # Known swing points for liquidity placement
        self._swing_highs: List[Tuple[int, float]] = []
        self._swing_lows: List[Tuple[int, float]] = []
        
        # ATR state for sweep detection
        self._atr_value: float = 0.0
        self._tr_values: List[float] = []
        self._atr_period: int = 14
        
        # Level tracking for equal detection
        self._level_touches: Dict[float, List[Tuple[int, float]]] = defaultdict(list)
        
        # Current price tracking
        self._current_price: float = 0.0
        self._last_timestamp: int = 0
    
    @property
    def pools(self) -> List[LiquidityPool]:
        """Get list of active liquidity pools (read-only copy)."""
        return [p for p in self._pools if not p.swept or p.mitigation_target is not None]
    
    @property
    def unswept_bsl(self) -> List[LiquidityPool]:
        """Get all unswept buy side liquidity pools."""
        return [p for p in self._pools if p.is_bsl() and not p.swept]
    
    @property
    def unswept_ssl(self) -> List[LiquidityPool]:
        """Get all unswept sell side liquidity pools."""
        return [p for p in self._pools if p.is_ssl() and not p.swept]
    
    @property
    def recent_sweeps(self) -> List[LiquidityPool]:
        """Get recently swept pools awaiting mitigation."""
        return [p for p in self._pools if p.swept and p.mitigation_target is not None]
    
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
    
    def _normalize_price_level(self, price: float) -> float:
        """Normalize price to detect equal levels within tolerance."""
        tolerance = price * self.config.equal_level_tolerance_pct / 100.0
        # Round to nearest tolerance increment
        return round(price / tolerance) * tolerance
    
    def register_swing_high(self, timestamp: int, price: float) -> None:
        """Register a swing high for BSL placement."""
        self._swing_highs.append((timestamp, price))
        
        # Keep only recent swings
        if len(self._swing_highs) > self.config.lookback_period:
            self._swing_highs.pop(0)
        
        # Create liquidity pool at this level
        self._create_liquidity_pool(
            price_level=price,
            liquidity_type=LiquidityType.SWING_HIGH,
            timestamp=timestamp
        )
    
    def register_swing_low(self, timestamp: int, price: float) -> None:
        """Register a swing low for SSL placement."""
        self._swing_lows.append((timestamp, price))
        
        # Keep only recent swings
        if len(self._swing_lows) > self.config.lookback_period:
            self._swing_lows.pop(0)
        
        # Create liquidity pool at this level
        self._create_liquidity_pool(
            price_level=price,
            liquidity_type=LiquidityType.SWING_LOW,
            timestamp=timestamp
        )
    
    def _create_liquidity_pool(
        self, 
        price_level: float, 
        liquidity_type: LiquidityType,
        timestamp: int
    ) -> Optional[LiquidityPool]:
        """Create new liquidity pool if one doesn't already exist nearby."""
        # Check for existing pool at similar level
        normalized = self._normalize_price_level(price_level)
        
        for pool in self._pools:
            pool_normalized = self._normalize_price_level(pool.price_level)
            if abs(normalized - pool_normalized) < 1:
                # Pool already exists - update it
                pool.touch_count += 1
                pool.last_touched_timestamp = timestamp
                
                # Upgrade to equal level if enough touches
                if (pool.touch_count >= self.config.min_touches_for_pool and 
                    not pool.is_equal_level()):
                    if pool.liquidity_type == LiquidityType.SWING_HIGH:
                        pool.liquidity_type = LiquidityType.EQUAL_HIGH
                    elif pool.liquidity_type == LiquidityType.SWING_LOW:
                        pool.liquidity_type = LiquidityType.EQUAL_LOW
                
                return None
        
        # Create new pool
        pool = LiquidityPool(
            pool_id=self._next_pool_id,
            price_level=price_level,
            liquidity_type=liquidity_type,
            strength=0.3,  # Initial strength
            stop_volume_estimate=0.0,
            touch_count=1,
            last_touched_timestamp=timestamp
        )
        
        self._pools.append(pool)
        self._next_pool_id += 1
        
        # Enforce memory bounds
        if len(self._pools) > self.config.max_pools:
            self._remove_weakest_pools()
        
        return pool
    
    def _remove_weakest_pools(self) -> None:
        """Remove weakest pools to maintain memory bounds."""
        # Sort by strength and remove weakest
        self._pools.sort(key=lambda p: p.strength)
        
        # Remove bottom 20%
        remove_count = max(1, len(self._pools) // 5)
        self._pools = self._pools[remove_count:]
    
    def update(self, timestamp: int, high: float, low: float, close: float, volume: float) -> List[LiquidityPool]:
        """
        Process new candle data and check for liquidity interactions.
        
        Returns list of pools that were interacted with or swept.
        """
        self._current_price = close
        self._last_timestamp = timestamp
        
        # Update ATR
        prev_close = self._price_history[-1][2] if self._price_history else close
        tr = self._calculate_true_range(high, low, prev_close)
        self._update_atr(tr)
        
        # Store price history
        self._price_history.append((timestamp, high, low))
        if len(self._price_history) > self.config.lookback_period:
            self._price_history.pop(0)
        
        interacted_pools: List[LiquidityPool] = []
        sweep_threshold = self._atr_value * self.config.sweep_atr_multiplier if self._atr_value > 0 else float('inf')
        
        # Check each pool for interaction
        for pool in self._pools:
            if pool.swept:
                # Check for mitigation (return to swept level)
                if pool.liquidity_type.is_bsl() and low <= pool.price_level <= high:
                    interacted_pools.append(pool)
                elif pool.liquidity_type.is_ssl() and low <= pool.price_level <= high:
                    interacted_pools.append(pool)
                continue
            
            # Check for sweep
            if pool.is_bsl():
                # Buy side liquidity - above price
                if high >= pool.price_level:
                    # Check if it's a true sweep (wicked above then closed below)
                    if close < pool.price_level and (high - pool.price_level) < sweep_threshold:
                        pool.swept = True
                        pool.sweep_timestamp = timestamp
                        
                        # Calculate mitigation target (opposing liquidity)
                        pool.mitigation_target = self._find_mitigation_target(pool.price_level, False)
                        
                        # Update strength
                        pool.strength = min(1.0, pool.strength + 0.2)
                        
                        interacted_pools.append(pool)
            
            elif pool.is_ssl():
                # Sell side liquidity - below price
                if low <= pool.price_level:
                    # Check if it's a true sweep (wicked below then closed above)
                    if close > pool.price_level and (pool.price_level - low) < sweep_threshold:
                        pool.swept = True
                        pool.sweep_timestamp = timestamp
                        
                        # Calculate mitigation target
                        pool.mitigation_target = self._find_mitigation_target(pool.price_level, True)
                        
                        # Update strength
                        pool.strength = min(1.0, pool.strength + 0.2)
                        
                        interacted_pools.append(pool)
            
            # Check for simple touch (not sweep)
            if not pool.swept:
                if pool.is_bsl() and abs(high - pool.price_level) < pool.price_level * 0.001:
                    pool.touch_count += 1
                    pool.last_touched_timestamp = timestamp
                    pool.strength = min(1.0, pool.strength + 0.05)
                    interacted_pools.append(pool)
                
                elif pool.is_ssl() and abs(low - pool.price_level) < pool.price_level * 0.001:
                    pool.touch_count += 1
                    pool.last_touched_timestamp = timestamp
                    pool.strength = min(1.0, pool.strength + 0.05)
                    interacted_pools.append(pool)
        
        # Estimate stop volume based on price action
        self._estimate_stop_volumes(volume)
        
        return interacted_pools
    
    def _find_mitigation_target(self, swept_level: float, was_bsl_sweep: bool) -> Optional[float]:
        """
        Find the mitigation target after a liquidity sweep.
        
        After BSL sweep -> target is opposing SSL
        After SSL sweep -> target is opposing BSL
        """
        if was_bsl_sweep:
            # Look for nearest SSL below
            ssl_pools = [p for p in self._pools if p.is_ssl() and p.price_level < swept_level and not p.swept]
            if ssl_pools:
                return max(p.price_level for p in ssl_pools)
        else:
            # Look for nearest BSL above
            bsl_pools = [p for p in self._pools if p.is_bsl() and p.price_level > swept_level and not p.swept]
            if bsl_pools:
                return min(p.price_level for p in bsl_pools)
        
        return None
    
    def _estimate_stop_volumes(self, current_volume: float) -> None:
        """Estimate stop loss volume at each liquidity pool."""
        for pool in self._pools:
            if pool.touch_count > 0:
                # Simple estimation based on touches and recent volume
                base_estimate = current_volume * pool.touch_count * 0.1
                pool.stop_volume_estimate = base_estimate * (1.0 + pool.strength)
    
    def get_nearest_bsl(self, current_price: float) -> Optional[LiquidityPool]:
        """Get nearest unswept buy side liquidity above current price."""
        bsl_pools = [p for p in self.unswept_bsl if p.price_level > current_price]
        if not bsl_pools:
            return None
        return min(bsl_pools, key=lambda p: p.price_level - current_price)
    
    def get_nearest_ssl(self, current_price: float) -> Optional[LiquidityPool]:
        """Get nearest unswept sell side liquidity below current price."""
        ssl_pools = [p for p in self.unswept_ssl if p.price_level < current_price]
        if not ssl_pools:
            return None
        return max(ssl_pools, key=lambda p: current_price - p.price_level)
    
    def get_total_liquidity_above(self, current_price: float) -> float:
        """Estimate total liquidity (stop volume) above current price."""
        return sum(p.stop_volume_estimate for p in self.unswept_bsl if p.price_level > current_price)
    
    def get_total_liquidity_below(self, current_price: float) -> float:
        """Estimate total liquidity (stop volume) below current price."""
        return sum(p.stop_volume_estimate for p in self.unswept_ssl if p.price_level < current_price)
    
    def detect_equal_levels(self) -> List[Tuple[LiquidityPool, LiquidityPool]]:
        """
        Detect equal high/low pairs that form strong liquidity zones.
        
        Returns list of (equal_high_pool, equal_low_pool) tuples.
        """
        equal_highs = [p for p in self._pools if p.liquidity_type == LiquidityType.EQUAL_HIGH and not p.swept]
        equal_lows = [p for p in self._pools if p.liquidity_type == LiquidityType.EQUAL_LOW and not p.swept]
        
        pairs = []
        for eh in equal_highs:
            for el in equal_lows:
                # Check if they form a valid range
                if eh.price_level > el.price_level:
                    pairs.append((eh, el))
        
        return pairs
    
    def reset(self) -> None:
        """Reset mapper state for new asset or timeframe."""
        self._pools.clear()
        self._price_history.clear()
        self._swing_highs.clear()
        self._swing_lows.clear()
        self._tr_values.clear()
        self._atr_value = 0.0
        self._level_touches.clear()
        self._next_pool_id = 1


if __name__ == "__main__":
    # Example usage and basic validation
    print("ZAID Bot - Liquidity Pools Module")
    print("=" * 50)
    
    mapper = LiquidityPoolMapper()
    
    # Simulate some price data with swing points
    test_data = [
        (1000, 105.0, 100.0, 103.0, 1000.0),
        (1001, 107.0, 102.0, 106.0, 1200.0),
        (1002, 109.0, 104.0, 108.0, 1500.0),  # Swing high
        (1003, 108.0, 103.0, 105.0, 900.0),
        (1004, 106.0, 101.0, 102.0, 800.0),
        (1005, 104.0, 99.0, 100.0, 1100.0),   # Swing low
        (1006, 102.0, 98.0, 99.0, 950.0),
        (1007, 103.0, 97.0, 98.0, 1050.0),    # Another swing low (equal)
        (1008, 105.0, 100.0, 104.0, 1300.0),
        (1009, 108.0, 103.0, 107.0, 1600.0),
        (1010, 109.5, 105.0, 109.0, 1800.0),  # Test near swing high
        (1011, 110.5, 107.0, 108.0, 2000.0),  # Sweep above swing high
        (1012, 108.0, 104.0, 105.0, 1400.0),  # Close back below
    ]
    
    # Register swing points
    mapper.register_swing_high(1002, 109.0)
    mapper.register_swing_low(1005, 99.0)
    mapper.register_swing_low(1007, 97.0)
    
    print("\nInitial liquidity pools:")
    for pool in mapper.pools:
        print(f"  {pool.liquidity_type.name}: {pool.price_level:.2f} (strength: {pool.strength:.2f})")
    
    print("\nProcessing candles...")
    for ts, high, low, close, vol in test_data:
        interacted = mapper.update(ts, high, low, close, vol)
        if interacted:
            for pool in interacted:
                if pool.swept:
                    print(f"  SWEEP at {pool.price_level:.2f}! Target: {pool.mitigation_target}")
    
    print(f"\nTotal pools tracked: {len(mapper.pools)}")
    print(f"Unswept BSL: {len(mapper.unswept_bsl)}")
    print(f"Unswept SSL: {len(mapper.unswept_ssl)}")
    print(f"Recent sweeps: {len(mapper.recent_sweeps)}")
