#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
Micro-Price Calculation Module
Calculates true fair value using queue dynamics and order book imbalance
Optimized for 8GB RAM constraint with C-extension compatible structures
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
from collections import deque
import numpy as np


class OrderSide(Enum):
    """Order side enumeration for type safety."""
    BID = "bid"
    ASK = "ask"


@dataclass(slots=True)
class PriceLevel:
    """
    Represents a single price level in the order book.
    Uses __slots__ for memory efficiency on 8GB RAM systems.
    """
    price: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    bid_order_count: int = 0
    ask_order_count: int = 0
    last_update_ns: int = 0
    # Queue decay metrics
    avg_cancel_rate: float = 0.0
    avg_fill_rate: float = 0.0


@dataclass(slots=True)
class MicroPriceSnapshot:
    """Snapshot of micro-price calculation results."""
    timestamp_ns: int
    micro_price: float
    mid_price: float
    weighted_mid: float
    imbalance_ratio: float
    bid_pressure: float
    ask_pressure: float
    fair_value_estimate: float
    confidence_score: float
    spread_bps: float


class MicroPriceCalculator:
    """
    Calculates true fair value using queue dynamics and order book microstructure.
    
    Implements multiple micro-price models:
    1. Volume-weighted micro-price
    2. Order-flow weighted micro-price
    3. Queue-position adjusted micro-price
    4. Toxicity-adjusted fair value
    
    All calculations are optimized for O(1) updates where possible.
    """
    
    def __init__(self, max_levels: int = 50, lookback_windows: int = 100):
        """
        Initialize micro-price calculator.
        
        Args:
            max_levels: Maximum price levels to track (memory constraint)
            lookback_windows: Number of snapshots for historical analysis
        """
        self.max_levels = max_levels
        self.lookback_windows = lookback_windows
        
        # Price level storage - pre-allocated for memory efficiency
        self.levels: Dict[float, PriceLevel] = {}
        
        # Historical snapshots for trend analysis
        self.snapshots: deque[MicroPriceSnapshot] = deque(maxlen=lookback_windows)
        
        # Best bid/ask cache for O(1) access
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
        
        # Tick size for the trading pair (configurable per asset)
        self.tick_size: float = 0.01
        
        # Last calculation timestamp
        self.last_calculation_ns: int = 0
        
        # Calibration parameters (can be learned from historical data)
        self.imbalance_weight: float = 0.6
        self.queue_weight: float = 0.3
        self.flow_weight: float = 0.1
        
    def update_level(self, price: float, side: OrderSide, volume_delta: float,
                     is_addition: bool = True) -> None:
        """
        Update a price level with new order book data.
        O(1) operation for existing levels.
        
        Args:
            price: Price level to update
            side: Bid or Ask
            volume_delta: Change in volume at this level
            is_addition: True if adding volume, False if removing
        """
        current_time_ns = time.time_ns()
        
        if price not in self.levels:
            if len(self.levels) >= self.max_levels:
                # Remove farthest level if at capacity (memory constraint)
                self._prune_farthest_level()
            self.levels[price] = PriceLevel(price=price)
        
        level = self.levels[price]
        
        if side == OrderSide.BID:
            if is_addition:
                level.bid_volume += volume_delta
                level.bid_order_count += 1
            else:
                level.bid_volume = max(0.0, level.bid_volume - volume_delta)
                level.bid_order_count = max(0, level.bid_order_count - 1)
        else:
            if is_addition:
                level.ask_volume += volume_delta
                level.ask_order_count += 1
            else:
                level.ask_volume = max(0.0, level.ask_volume - volume_delta)
                level.ask_order_count = max(0, level.ask_order_count - 1)
        
        level.last_update_ns = current_time_ns
        
        # Update best bid/ask cache
        self._update_best_prices()
    
    def _prune_farthest_level(self) -> None:
        """Remove the price level farthest from mid (memory management)."""
        if not self.levels:
            return
            
        mid = self._calculate_raw_mid()
        if mid is None:
            return
            
        farthest_price = max(
            self.levels.keys(),
            key=lambda p: abs(p - mid)
        )
        del self.levels[farthest_price]
        self._update_best_prices()
    
    def _update_best_prices(self) -> None:
        """Update cached best bid and ask prices."""
        bid_prices = [p for p, l in self.levels.items() if l.bid_volume > 0]
        ask_prices = [p for p, l in self.levels.items() if l.ask_volume > 0]
        
        self._best_bid = max(bid_prices) if bid_prices else None
        self._best_ask = min(ask_prices) if ask_prices else None
    
    def _calculate_raw_mid(self) -> Optional[float]:
        """Calculate raw mid price from best bid/ask."""
        if self._best_bid is not None and self._best_ask is not None:
            return (self._best_bid + self._best_ask) / 2.0
        return None
    
    def calculate_volume_weighted_micro_price(self) -> Optional[float]:
        """
        Calculate volume-weighted micro-price using top-of-book volumes.
        
        Formula: MicroPrice = (Bid * AskVol + Ask * BidVol) / (BidVol + AskVol)
        
        Returns:
            Micro-price value or None if insufficient data
        """
        if self._best_bid is None or self._best_ask is None:
            return None
        
        bid_vol = self.levels[self._best_bid].bid_volume
        ask_vol = self.levels[self._best_ask].ask_volume
        
        total_vol = bid_vol + ask_vol
        if total_vol <= 0:
            return (self._best_bid + self._best_ask) / 2.0
        
        micro_price = (
            self._best_bid * ask_vol + self._best_ask * bid_vol
        ) / total_vol
        
        return micro_price
    
    def calculate_imbalance_adjusted_micro_price(self, depth_levels: int = 5) -> Optional[float]:
        """
        Calculate micro-price adjusted for order book imbalance across multiple levels.
        
        Args:
            depth_levels: Number of levels to include in imbalance calculation
            
        Returns:
            Imbalance-adjusted micro-price
        """
        if self._best_bid is None or self._best_ask is None:
            return None
        
        # Sort prices
        sorted_prices = sorted(self.levels.keys())
        mid_idx = len(sorted_prices) // 2
        
        # Get levels around mid
        bid_prices = sorted([p for p in sorted_prices[:mid_idx] 
                            if self.levels[p].bid_volume > 0], reverse=True)
        ask_prices = sorted([p for p in sorted_prices[mid_idx:] 
                            if self.levels[p].ask_volume > 0])
        
        if not bid_prices or not ask_prices:
            return self.calculate_volume_weighted_micro_price()
        
        # Calculate cumulative volumes
        total_bid_vol = sum(self.levels[p].bid_volume for p in bid_prices[:depth_levels])
        total_ask_vol = sum(self.levels[p].ask_volume for p in ask_prices[:depth_levels])
        
        # Volume-weighted average prices
        if total_bid_vol > 0:
            vwap_bid = sum(p * self.levels[p].bid_volume 
                          for p in bid_prices[:depth_levels]) / total_bid_vol
        else:
            vwap_bid = self._best_bid
        
        if total_ask_vol > 0:
            vwap_ask = sum(p * self.levels[p].ask_volume 
                          for p in ask_prices[:depth_levels]) / total_ask_vol
        else:
            vwap_ask = self._best_ask
        
        # Imbalance ratio
        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-9)
        
        # Adjust mid price based on imbalance
        base_mid = (vwap_bid + vwap_ask) / 2.0
        spread = vwap_ask - vwap_bid
        
        adjusted_micro_price = base_mid + (imbalance * spread * self.imbalance_weight / 2.0)
        
        return adjusted_micro_price
    
    def calculate_queue_position_adjusted_price(self, our_queue_position: float = 0.5) -> Optional[float]:
        """
        Adjust micro-price based on our queue position.
        
        Args:
            our_queue_position: Fraction of orders ahead of us (0.0 = front, 1.0 = back)
            
        Returns:
            Queue-position adjusted price
        """
        base_micro = self.calculate_volume_weighted_micro_price()
        if base_micro is None:
            return None
        
        # If we're far back in queue, adjust price to reflect lower fill probability
        queue_discount = our_queue_position * 0.0001  # 1 tick adjustment factor
        
        return base_micro - queue_discount
    
    def calculate_fair_value(self, vpin: float = 0.0, toxicity_score: float = 0.0) -> Optional[MicroPriceSnapshot]:
        """
        Calculate comprehensive fair value estimate incorporating all factors.
        
        Args:
            vpin: Volume-Synchronized Probability of Informed Trading (0-1)
            toxicity_score: Order flow toxicity metric (0-1)
            
        Returns:
            MicroPriceSnapshot with all calculated metrics
        """
        current_time_ns = time.time_ns()
        
        # Base calculations
        mid_price = self._calculate_raw_mid()
        if mid_price is None:
            return None
        
        vol_weighted = self.calculate_volume_weighted_micro_price()
        imbalance_adjusted = self.calculate_imbalance_adjusted_micro_price()
        
        if vol_weighted is None:
            vol_weighted = mid_price
        if imbalance_adjusted is None:
            imbalance_adjusted = mid_price
        
        # Combine micro-price estimates
        micro_price = (
            self.imbalance_weight * imbalance_adjusted +
            self.queue_weight * vol_weighted +
            self.flow_weight * mid_price
        )
        
        # Calculate imbalance ratio
        if self._best_bid is not None and self._best_ask is not None:
            bid_vol = self.levels[self._best_bid].bid_volume
            ask_vol = self.levels[self._best_ask].ask_volume
            total_vol = bid_vol + ask_vol + 1e-9
            imbalance_ratio = (bid_vol - ask_vol) / total_vol
        else:
            imbalance_ratio = 0.0
        
        # Calculate bid/ask pressure
        bid_pressure = max(0.0, imbalance_ratio)
        ask_pressure = max(0.0, -imbalance_ratio)
        
        # Adjust for toxicity (informed trading reduces fair value certainty)
        toxicity_adjustment = toxicity_score * 0.0005 * mid_price
        fair_value_estimate = micro_price - (toxicity_adjustment * (vpin - 0.5))
        
        # Calculate spread in basis points
        if self._best_bid is not None and self._best_ask is not None:
            spread_bps = ((self._best_ask - self._best_bid) / mid_price) * 10000
        else:
            spread_bps = 0.0
        
        # Confidence score based on data quality
        confidence_score = self._calculate_confidence_score(vpin, toxicity_score)
        
        snapshot = MicroPriceSnapshot(
            timestamp_ns=current_time_ns,
            micro_price=micro_price,
            mid_price=mid_price,
            weighted_mid=vol_weighted,
            imbalance_ratio=imbalance_ratio,
            bid_pressure=bid_pressure,
            ask_pressure=ask_pressure,
            fair_value_estimate=fair_value_estimate,
            confidence_score=confidence_score,
            spread_bps=spread_bps
        )
        
        # Store snapshot for historical analysis
        self.snapshots.append(snapshot)
        self.last_calculation_ns = current_time_ns
        
        return snapshot
    
    def _calculate_confidence_score(self, vpin: float, toxicity_score: float) -> float:
        """
        Calculate confidence score for fair value estimate.
        
        Returns value between 0.0 and 1.0
        """
        # Base confidence from spread tightness
        if self._best_bid is not None and self._best_ask is not None:
            mid = (self._best_bid + self._best_ask) / 2.0
            spread_ratio = (self._best_ask - self._best_bid) / mid
            spread_confidence = max(0.0, 1.0 - spread_ratio * 100)
        else:
            spread_confidence = 0.0
        
        # Reduce confidence for high VPIN/toxicity
        info_confidence = 1.0 - (vpin * 0.5 + toxicity_score * 0.5)
        
        # Data sufficiency confidence
        data_confidence = min(1.0, len(self.levels) / 10.0)
        
        # Weighted combination
        confidence = (
            0.4 * spread_confidence +
            0.4 * info_confidence +
            0.2 * data_confidence
        )
        
        return max(0.0, min(1.0, confidence))
    
    def get_micro_price_trend(self, window_size: int = 10) -> float:
        """
        Calculate trend in micro-price over recent snapshots.
        
        Args:
            window_size: Number of snapshots to include
            
        Returns:
            Trend as price change per nanosecond
        """
        if len(self.snapshots) < 2:
            return 0.0
        
        recent = list(self.snapshots)[-window_size:]
        if len(recent) < 2:
            return 0.0
        
        first = recent[0]
        last = recent[-1]
        
        time_delta = last.timestamp_ns - first.timestamp_ns
        if time_delta <= 0:
            return 0.0
        
        price_delta = last.micro_price - first.micro_price
        
        return price_delta / time_delta
    
    def clear(self) -> None:
        """Clear all stored data (for symbol changes)."""
        self.levels.clear()
        self.snapshots.clear()
        self._best_bid = None
        self._best_ask = None


# C-extension compatible interface for performance-critical paths
def calculate_micro_price_c_interface(
    bid_prices: List[float],
    bid_volumes: List[float],
    ask_prices: List[float],
    ask_volumes: List[float]
) -> Tuple[float, float, float]:
    """
    C-extension compatible micro-price calculation.
    
    Args:
        bid_prices: List of bid prices
        bid_volumes: List of corresponding bid volumes
        ask_prices: List of ask prices
        ask_volumes: List of corresponding ask volumes
        
    Returns:
        Tuple of (micro_price, mid_price, imbalance_ratio)
    """
    if not bid_prices or not ask_prices:
        return (0.0, 0.0, 0.0)
    
    best_bid = max(bid_prices)
    best_ask = min(ask_prices)
    mid_price = (best_bid + best_ask) / 2.0
    
    best_bid_idx = bid_prices.index(best_bid)
    best_ask_idx = ask_prices.index(best_ask)
    
    bid_vol = bid_volumes[best_bid_idx]
    ask_vol = ask_volumes[best_ask_idx]
    
    total_vol = bid_vol + ask_vol
    if total_vol > 0:
        micro_price = (best_bid * ask_vol + best_ask * bid_vol) / total_vol
    else:
        micro_price = mid_price
    
    imbalance_ratio = (bid_vol - ask_vol) / (total_vol + 1e-9)
    
    return (micro_price, mid_price, imbalance_ratio)


if __name__ == "__main__":
    # Example usage and testing
    calculator = MicroPriceCalculator()
    
    # Simulate order book updates
    calculator.update_level(49999.5, OrderSide.BID, 1.5)
    calculator.update_level(49999.0, OrderSide.BID, 2.0)
    calculator.update_level(50000.0, OrderSide.ASK, 1.0)
    calculator.update_level(50000.5, OrderSide.ASK, 1.5)
    
    # Calculate fair value
    snapshot = calculator.calculate_fair_value(vpin=0.3, toxicity_score=0.2)
    
    if snapshot:
        print(f"Micro-Price: {snapshot.micro_price:.4f}")
        print(f"Mid-Price: {snapshot.mid_price:.4f}")
        print(f"Fair Value: {snapshot.fair_value_estimate:.4f}")
        print(f"Confidence: {snapshot.confidence_score:.2%}")
        print(f"Spread (bps): {snapshot.spread_bps:.2f}")
