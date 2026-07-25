#!/usr/bin/env python3
"""
backend/volume_profile/value_area.py

Dynamically updates Value Area High (VAH) and Low (VAL) based on real-time volume.
Implements Observer pattern for notifications and Strategy pattern for different
value area calculation methods.

Features:
- Real-time VAH/VAL updates as trades occur
- Handles low-volume nodes and high-volume nodes during sideways chop
- Multiple value area calculation strategies (70%, 80%, custom)
- Memory-efficient rolling profile windows
- Strict type hinting for production reliability
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Protocol, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
import time


class MarketRegime(Enum):
    """Detected market regime affecting value area behavior."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    SIDEWAYS_CHOP = auto()
    HIGH_VOLATILITY = auto()
    LOW_LIQUIDITY = auto()


@dataclass
class VolumeNode:
    """Single price level with volume information."""
    price: float
    total_volume: float
    bid_volume: float
    ask_volume: float
    trade_count: int = 0
    is_poc: bool = False
    is_vah: bool = False
    is_val: bool = False
    timestamp_ns: int = 0
    
    @property
    def delta(self) -> float:
        """Net delta (ask - bid)."""
        return self.ask_volume - self.bid_volume
    
    @property
    def imbalance_ratio(self) -> float:
        """Ratio of net delta to total volume."""
        if self.total_volume == 0:
            return 0.0
        return self.delta / self.total_volume


@dataclass
class ValueAreaResult:
    """Result of value area calculation."""
    vah: float  # Value Area High
    val: float  # Value Area Low
    poc: float  # Point of Control
    total_volume: float
    value_area_volume: float
    percentage: float
    node_count: int
    timestamp_ns: int


class ValueAreaStrategy(Protocol):
    """Strategy interface for value area calculation."""
    
    def calculate(
        self, 
        nodes: Dict[float, VolumeNode],
        poc_price: float
    ) -> Optional[ValueAreaResult]:
        """Calculate value area from volume nodes."""
        ...


class PercentageValueAreaStrategy:
    """
    Standard percentage-based value area calculation.
    Typically uses 70% (one standard deviation in normal distribution).
    """
    
    def __init__(self, percentage: float = 0.70):
        if not 0.0 < percentage <= 1.0:
            raise ValueError("Percentage must be between 0 and 1")
        self.percentage = percentage
    
    def calculate(
        self,
        nodes: Dict[float, VolumeNode],
        poc_price: float
    ) -> Optional[ValueAreaResult]:
        if not nodes or poc_price not in nodes:
            return None
        
        total_volume = sum(n.total_volume for n in nodes.values())
        if total_volume == 0:
            return None
        
        target_volume = total_volume * self.percentage
        poc_node = nodes[poc_price]
        
        # Clear previous flags
        for node in nodes.values():
            node.is_vah = False
            node.is_val = False
        
        # Start from POC and expand outward
        sorted_prices = sorted(nodes.keys())
        poc_index = sorted_prices.index(poc_price)
        
        accumulated_volume = poc_node.total_volume
        left_idx = poc_index
        right_idx = poc_index
        
        # Mark POC
        nodes[poc_price].is_vah = True
        nodes[poc_price].is_val = True
        
        # Expand until we reach target volume
        while accumulated_volume < target_volume:
            left_vol = nodes[sorted_prices[left_idx - 1]].total_volume if left_idx > 0 else 0
            right_vol = (
                nodes[sorted_prices[right_idx + 1]].total_volume 
                if right_idx < len(sorted_prices) - 1 
                else 0
            )
            
            if left_vol >= right_vol and left_idx > 0:
                left_idx -= 1
                accumulated_volume += left_vol
                nodes[sorted_prices[left_idx]].is_val = True
            elif right_idx < len(sorted_prices) - 1:
                right_idx += 1
                accumulated_volume += right_vol
                nodes[sorted_prices[right_idx]].is_vah = True
            else:
                break
        
        vah = sorted_prices[right_idx]
        val = sorted_prices[left_idx]
        
        return ValueAreaResult(
            vah=vah,
            val=val,
            poc=poc_price,
            total_volume=total_volume,
            value_area_volume=accumulated_volume,
            percentage=self.percentage,
            node_count=right_idx - left_idx + 1,
            timestamp_ns=time.time_ns()
        )


class AdaptiveValueAreaStrategy:
    """
    Adapts value area percentage based on market regime.
    - Sideways chop: wider value area (more price acceptance)
    - Trending: narrower value area (price discovery)
    """
    
    def __init__(self):
        self.base_percentage = 0.70
        self.regime_adjustments: Dict[MarketRegime, float] = {
            MarketRegime.SIDEWAYS_CHOP: 0.10,   # 80% total
            MarketRegime.TRENDING_UP: -0.10,    # 60% total
            MarketRegime.TRENDING_DOWN: -0.10,  # 60% total
            MarketRegime.HIGH_VOLATILITY: 0.05,  # 75% total
            MarketRegime.LOW_LIQUIDITY: 0.15,   # 85% total
        }
        self._percentage_strategy = PercentageValueAreaStrategy()
    
    def calculate(
        self,
        nodes: Dict[float, VolumeNode],
        poc_price: float,
        regime: MarketRegime = MarketRegime.SIDEWAYS_CHOP
    ) -> Optional[ValueAreaResult]:
        adjustment = self.regime_adjustments.get(regime, 0.0)
        adjusted_pct = max(0.4, min(0.9, self.base_percentage + adjustment))
        
        self._percentage_strategy.percentage = adjusted_pct
        return self._percentage_strategy.calculate(nodes, poc_price)


class ValueAreaTracker:
    """
    Main tracker for dynamic value area updates.
    Implements Observer pattern for real-time notifications.
    """
    
    def __init__(self, strategy: Optional[ValueAreaStrategy] = None):
        self.strategy = strategy or PercentageValueAreaStrategy(0.70)
        self.nodes: Dict[str, Dict[float, VolumeNode]] = defaultdict(dict)
        self.poc_cache: Dict[str, float] = {}
        self.observers: List[Callable[[str, ValueAreaResult], None]] = []
        self.regime_history: Dict[str, List[MarketRegime]] = defaultdict(list)
    
    def add_observer(
        self, 
        callback: Callable[[str, ValueAreaResult], None]
    ) -> None:
        """Register callback for value area updates."""
        self.observers.append(callback)
    
    def add_trade(
        self,
        symbol: str,
        price: float,
        volume: float,
        is_buyer_maker: bool,
        timestamp_ns: Optional[int] = None
    ) -> None:
        """Add a trade and update volume profile."""
        ts = timestamp_ns or time.time_ns()
        
        if price not in self.nodes[symbol]:
            self.nodes[symbol][price] = VolumeNode(
                price=price,
                total_volume=0,
                bid_volume=0,
                ask_volume=0
            )
        
        node = self.nodes[symbol][price]
        if is_buyer_maker:
            node.bid_volume += volume
        else:
            node.ask_volume += volume
        node.total_volume = node.bid_volume + node.ask_volume
        node.trade_count += 1
        node.timestamp_ns = ts
        
        # Update POC if this node has highest volume
        current_poc = self.poc_cache.get(symbol)
        if current_poc is None or node.total_volume > self.nodes[symbol][current_poc].total_volume:
            # Clear old POC flag
            if current_poc and current_poc in self.nodes[symbol]:
                self.nodes[symbol][current_poc].is_poc = False
            
            # Set new POC
            node.is_poc = True
            self.poc_cache[symbol] = price
    
    def get_current_poc(self, symbol: str) -> Optional[float]:
        """Get current Point of Control for symbol."""
        return self.poc_cache.get(symbol)
    
    def calculate_value_area(
        self,
        symbol: str,
        use_adaptive: bool = False,
        regime: Optional[MarketRegime] = None
    ) -> Optional[ValueAreaResult]:
        """Calculate current value area for symbol."""
        if symbol not in self.nodes or not self.nodes[symbol]:
            return None
        
        poc = self.poc_cache.get(symbol)
        if poc is None:
            return None
        
        if use_adaptive:
            if regime is None:
                regime = self._detect_regime(symbol)
            strategy = AdaptiveValueAreaStrategy()
            result = strategy.calculate(self.nodes[symbol], poc, regime)
        else:
            result = self.strategy.calculate(self.nodes[symbol], poc)
        
        if result:
            # Notify observers
            for observer in self.observers:
                try:
                    observer(symbol, result)
                except Exception as e:
                    print(f"Observer error: {e}")
        
        return result
    
    def _detect_regime(self, symbol: str) -> MarketRegime:
        """Detect market regime from volume profile characteristics."""
        if symbol not in self.nodes or not self.nodes[symbol]:
            return MarketRegime.SIDEWAYS_CHOP
        
        nodes = list(self.nodes[symbol].values())
        if len(nodes) < 5:
            return MarketRegime.LOW_LIQUIDITY
        
        # Calculate profile width and concentration
        prices = [n.price for n in nodes if n.total_volume > 0]
        if not prices:
            return MarketRegime.LOW_LIQUIDITY
        
        price_range = max(prices) - min(prices)
        avg_volume = sum(n.total_volume for n in nodes) / len(nodes)
        
        # High volume nodes
        high_vol_nodes = [n for n in nodes if n.total_volume > avg_volume * 2]
        
        # Regime detection logic
        if len(high_vol_nodes) <= 2 and price_range > 100:
            # Concentrated volume, wide range = trending
            poc = self.poc_cache.get(symbol)
            if poc:
                current_price = max(nodes, key=lambda n: n.timestamp_ns).price
                if current_price > poc:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
        
        if len(high_vol_nodes) >= 5 and price_range < 50:
            # Many high volume nodes, narrow range = sideways chop
            return MarketRegime.SIDEWAYS_CHOP
        
        if price_range > 200:
            return MarketRegime.HIGH_VOLATILITY
        
        return MarketRegime.SIDEWAYS_CHOP
    
    def get_vah_val(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get current VAH and VAL for symbol."""
        result = self.calculate_value_area(symbol)
        if result:
            return result.vah, result.val
        return None, None
    
    def identify_low_volume_nodes(
        self,
        symbol: str,
        threshold_percentile: float = 0.3
    ) -> List[float]:
        """Identify low-volume nodes (potential breakout points)."""
        if symbol not in self.nodes:
            return []
        
        volumes = sorted([n.total_volume for n in self.nodes[symbol].values()])
        if not volumes:
            return []
        
        threshold_idx = int(len(volumes) * threshold_percentile)
        threshold = volumes[min(threshold_idx, len(volumes) - 1)]
        
        return [
            n.price for n in self.nodes[symbol].values()
            if n.total_volume <= threshold and n.total_volume > 0
        ]
    
    def identify_high_volume_nodes(
        self,
        symbol: str,
        threshold_percentile: float = 0.7
    ) -> List[float]:
        """Identify high-volume nodes (strong support/resistance)."""
        if symbol not in self.nodes:
            return []
        
        volumes = sorted([n.total_volume for n in self.nodes[symbol].values()])
        if not volumes:
            return []
        
        threshold_idx = int(len(volumes) * threshold_percentile)
        threshold = volumes[min(threshold_idx, len(volumes) - 1)]
        
        return [
            n.price for n in self.nodes[symbol].values()
            if n.total_volume >= threshold
        ]
    
    def reset_profile(self, symbol: str) -> None:
        """Reset volume profile for a symbol (new session)."""
        if symbol in self.nodes:
            self.nodes[symbol].clear()
        if symbol in self.poc_cache:
            del self.poc_cache[symbol]


def create_test_data() -> Dict[float, VolumeNode]:
    """Create test volume profile data."""
    nodes = {}
    
    # Create a bell-curve-like distribution
    center = 100.0
    for i in range(-10, 11):
        price = center + i * 0.5
        volume = max(1, 20 - abs(i) * 2)  # Higher volume near center
        nodes[price] = VolumeNode(
            price=price,
            total_volume=float(volume),
            bid_volume=float(volume) * 0.4,
            ask_volume=float(volume) * 0.6,
            timestamp_ns=time.time_ns()
        )
    
    return nodes


if __name__ == "__main__":
    # Example usage
    tracker = ValueAreaTracker()
    
    # Simulate trades
    for i in range(100):
        price = 100.0 + (i % 20 - 10) * 0.5
        volume = (i % 10) + 1
        tracker.add_trade("BTCUSDT", price, volume, i % 2 == 0)
    
    # Calculate value area
    result = tracker.calculate_value_area("BTCUSDT")
    if result:
        print(f"VPOC: {result.poc}")
        print(f"VAH: {result.vah}")
        print(f"VAL: {result.val}")
        print(f"Value Area Volume: {result.value_area_volume:.2f}")
        print(f"Total Volume: {result.total_volume:.2f}")
        
        # Identify key levels
        lvns = tracker.identify_low_volume_nodes("BTCUSDT")
        hvns = tracker.identify_high_volume_nodes("BTCUSDT")
        print(f"Low Volume Nodes: {lvns[:5]}...")
        print(f"High Volume Nodes: {hvns[:5]}...")
