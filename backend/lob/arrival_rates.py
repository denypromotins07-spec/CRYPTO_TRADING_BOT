#!/usr/bin/env python3
"""
Arrival Rates Calculator for Market vs Limit Orders

This module calculates the intensity of different order types (market orders, limit orders,
cancellations) arriving at the exchange. It integrates with the Hawkes process model to
provide real-time arrival rate estimates for optimal execution timing.

Key Features:
- Real-time intensity estimation using exponential moving averages
- Separation of aggressive (taker) vs passive (maker) order flows
- Volume-weighted arrival rate calculations
- Integration with Hawkes processes for self-excitation modeling
- Memory-efficient design for 8GB RAM constraint

Author: ZAID PERSONAL CRYPTO TRADING BOT - Stage 22
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum
import time
import math
from threading import Lock


class OrderType(Enum):
    """Enumeration of order types for arrival tracking."""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    CANCEL_BUY = "cancel_buy"
    CANCEL_SELL = "cancel_sell"
    AMENDMENT = "amendment"


class EventSide(Enum):
    """Side of the order book event."""
    BID = "bid"
    ASK = "ask"
    BOTH = "both"


@dataclass(slots=True)
class OrderEvent:
    """
    Represents a single order book event for arrival rate calculation.
    
    Uses __slots__ for memory efficiency on 8GB RAM systems.
    """
    timestamp: float  # Unix timestamp in seconds
    order_type: OrderType
    side: EventSide
    volume: float  # Order volume in base currency
    price: float  # Order price in quote currency
    aggressor: bool  # True if taker/market order, False if maker/limit
    
    @property
    def is_aggressive(self) -> bool:
        """Check if this is an aggressive (taker) order."""
        return self.aggressor
    
    @property
    def is_passive(self) -> bool:
        """Check if this is a passive (maker) order."""
        return not self.aggressor


@dataclass(slots=True)
class ArrivalRateStats:
    """Statistics snapshot for arrival rates."""
    market_buy_rate: float
    market_sell_rate: float
    limit_buy_rate: float
    limit_sell_rate: float
    cancel_rate: float
    total_arrival_rate: float
    aggressive_ratio: float
    passive_ratio: float
    imbalance: float  # (buy - sell) / (buy + sell)
    window_seconds: float
    event_count: int


class ExponentialMovingAverage:
    """
    Memory-efficient EMA calculator for real-time rate estimation.
    
    Provides smooth estimates without storing full history.
    """
    __slots__ = ['alpha', 'value', 'last_update', 'count']
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize EMA with smoothing factor.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Higher = more responsive.
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be in (0, 1]")
        self.alpha = alpha
        self.value: float = 0.0
        self.last_update: float = 0.0
        self.count: int = 0
    
    def update(self, new_value: float, timestamp: float) -> None:
        """Update EMA with new observation."""
        if self.count == 0:
            self.value = new_value
        else:
            dt = timestamp - self.last_update if self.last_update > 0 else 1.0
            # Time-decay adjustment for irregular arrivals
            effective_alpha = 1.0 - math.exp(-self.alpha * dt)
            self.value = self.value + effective_alpha * (new_value - self.value)
        
        self.last_update = timestamp
        self.count += 1
    
    def get_value(self) -> float:
        """Get current EMA value."""
        return self.value
    
    def reset(self) -> None:
        """Reset EMA state."""
        self.value = 0.0
        self.last_update = 0.0
        self.count = 0


class ArrivalRateCalculator:
    """
    Main class for calculating order arrival rates.
    
    Implements Observer pattern for real-time updates and uses
    multiple time windows for robust rate estimation.
    
    Thread-safe design for concurrent order book processing.
    """
    
    def __init__(
        self,
        short_window: float = 1.0,      # 1 second for HFT signals
        medium_window: float = 10.0,    # 10 seconds for tactical adjustments
        long_window: float = 60.0,      # 60 seconds for strategic view
        ema_alpha: float = 0.3,
        max_events_buffer: int = 10000,
    ):
        """
        Initialize arrival rate calculator.
        
        Args:
            short_window: Short-term analysis window (seconds)
            medium_window: Medium-term analysis window (seconds)
            long_window: Long-term analysis window (seconds)
            ema_alpha: EMA smoothing factor
            max_events_buffer: Maximum events to store for analysis
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.ema_alpha = ema_alpha
        self.max_events_buffer = max_events_buffer
        
        # Event storage with bounded size
        self.events: Deque[OrderEvent] = deque(maxlen=max_events_buffer)
        
        # Per-type arrival rate EMAs
        self.market_buy_ema = ExponentialMovingAverage(ema_alpha)
        self.market_sell_ema = ExponentialMovingAverage(ema_alpha)
        self.limit_buy_ema = ExponentialMovingAverage(ema_alpha)
        self.limit_sell_ema = ExponentialMovingAverage(ema_alpha)
        self.cancel_ema = ExponentialMovingAverage(ema_alpha)
        
        # Aggressive vs passive EMAs
        self.aggressive_ema = ExponentialMovingAverage(ema_alpha)
        self.passive_ema = ExponentialMovingAverage(ema_alpha)
        
        # Thread safety
        self._lock = Lock()
        
        # Last update timestamp
        self.last_update: float = 0.0
        
        # Callbacks for Observer pattern
        self._observers: List[callable] = []
        
        # Asset-specific coefficients (for BTC vs SOL differentiation)
        self.asset_coefficients: Dict[str, float] = {
            "BTC": 1.0,
            "ETH": 1.2,
            "SOL": 1.5,  # Higher volatility adjustment
            "USDT": 0.8,
        }
    
    def add_event(self, event: OrderEvent) -> None:
        """
        Record a new order event and update arrival rates.
        
        This is the main entry point for the Observer pattern.
        Thread-safe implementation.
        
        Args:
            event: The order event to record
        """
        with self._lock:
            current_time = event.timestamp
            
            # Calculate instantaneous rate contribution
            rate_contribution = event.volume  # Volume-weighted
            
            # Update appropriate EMA based on order type
            if event.order_type == OrderType.MARKET_BUY:
                self.market_buy_ema.update(rate_contribution, current_time)
                self.aggressive_ema.update(rate_contribution, current_time)
            elif event.order_type == OrderType.MARKET_SELL:
                self.market_sell_ema.update(rate_contribution, current_time)
                self.aggressive_ema.update(rate_contribution, current_time)
            elif event.order_type == OrderType.LIMIT_BUY:
                self.limit_buy_ema.update(rate_contribution, current_time)
                self.passive_ema.update(rate_contribution, current_time)
            elif event.order_type == OrderType.LIMIT_SELL:
                self.limit_sell_ema.update(rate_contribution, current_time)
                self.passive_ema.update(rate_contribution, current_time)
            elif event.order_type in (OrderType.CANCEL_BUY, OrderType.CANCEL_SELL):
                self.cancel_ema.update(rate_contribution, current_time)
            
            # Store event for window-based analysis
            self.events.append(event)
            self.last_update = current_time
            
            # Notify observers
            self._notify_observers(event)
    
    def add_market_order(
        self,
        side: EventSide,
        volume: float,
        price: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method to add a market order."""
        if timestamp is None:
            timestamp = time.time()
        
        order_type = OrderType.MARKET_BUY if side == EventSide.BID else OrderType.MARKET_SELL
        event = OrderEvent(
            timestamp=timestamp,
            order_type=order_type,
            side=side,
            volume=volume,
            price=price,
            aggressor=True,
        )
        self.add_event(event)
    
    def add_limit_order(
        self,
        side: EventSide,
        volume: float,
        price: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method to add a limit order."""
        if timestamp is None:
            timestamp = time.time()
        
        order_type = OrderType.LIMIT_BUY if side == EventSide.BID else OrderType.LIMIT_SELL
        event = OrderEvent(
            timestamp=timestamp,
            order_type=order_type,
            side=side,
            volume=volume,
            price=price,
            aggressor=False,
        )
        self.add_event(event)
    
    def add_cancellation(
        self,
        side: EventSide,
        volume: float,
        price: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Convenience method to add a cancellation."""
        if timestamp is None:
            timestamp = time.time()
        
        order_type = OrderType.CANCEL_BUY if side == EventSide.BID else OrderType.CANCEL_SELL
        event = OrderEvent(
            timestamp=timestamp,
            order_type=order_type,
            side=side,
            volume=volume,
            price=price,
            aggressor=False,
        )
        self.add_event(event)
    
    def get_current_rates(self, asset: str = "BTC") -> ArrivalRateStats:
        """
        Get current arrival rate statistics.
        
        Applies asset-specific coefficients for accurate cross-asset comparison.
        
        Args:
            asset: Asset symbol for coefficient adjustment
            
        Returns:
            ArrivalRateStats with current rates
        """
        with self._lock:
            coef = self.asset_coefficients.get(asset, 1.0)
            
            market_buy_rate = self.market_buy_ema.get_value() * coef
            market_sell_rate = self.market_sell_ema.get_value() * coef
            limit_buy_rate = self.limit_buy_ema.get_value() * coef
            limit_sell_rate = self.limit_sell_ema.get_value() * coef
            cancel_rate = self.cancel_ema.get_value() * coef
            
            total_market = market_buy_rate + market_sell_rate
            total_limit = limit_buy_rate + limit_sell_rate
            total_arrival = total_market + total_limit + cancel_rate
            
            aggressive_total = self.aggressive_ema.get_value() * coef
            passive_total = self.passive_ema.get_value() * coef
            
            # Calculate ratios safely
            if total_arrival > 0:
                aggressive_ratio = aggressive_total / total_arrival
                passive_ratio = passive_total / total_arrival
            else:
                aggressive_ratio = 0.0
                passive_ratio = 0.0
            
            # Order flow imbalance: (buy_volume - sell_volume) / (buy_volume + sell_volume)
            buy_volume = market_buy_rate + limit_buy_rate
            sell_volume = market_sell_rate + limit_sell_rate
            if buy_volume + sell_volume > 0:
                imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            else:
                imbalance = 0.0
            
            return ArrivalRateStats(
                market_buy_rate=market_buy_rate,
                market_sell_rate=market_sell_rate,
                limit_buy_rate=limit_buy_rate,
                limit_sell_rate=limit_sell_rate,
                cancel_rate=cancel_rate,
                total_arrival_rate=total_arrival,
                aggressive_ratio=aggressive_ratio,
                passive_ratio=passive_ratio,
                imbalance=imbalance,
                window_seconds=self.medium_window,
                event_count=len(self.events),
            )
    
    def get_window_rates(self, window_seconds: float) -> Dict[str, float]:
        """
        Calculate arrival rates over a specific time window.
        
        Uses actual event counts within the window for accuracy.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary of rate types to rates (events per second)
        """
        with self._lock:
            if not self.events:
                return {
                    "market_buy": 0.0,
                    "market_sell": 0.0,
                    "limit_buy": 0.0,
                    "limit_sell": 0.0,
                    "cancel": 0.0,
                }
            
            current_time = self.last_update
            cutoff = current_time - window_seconds
            
            # Count events in window
            counts = {
                OrderType.MARKET_BUY: 0.0,
                OrderType.MARKET_SELL: 0.0,
                OrderType.LIMIT_BUY: 0.0,
                OrderType.LIMIT_SELL: 0.0,
                OrderType.CANCEL_BUY: 0.0,
                OrderType.CANCEL_SELL: 0.0,
            }
            
            for event in self.events:
                if event.timestamp >= cutoff:
                    counts[event.order_type] += event.volume
            
            # Convert to rates (volume per second)
            return {
                "market_buy": counts[OrderType.MARKET_BUY] / window_seconds,
                "market_sell": counts[OrderType.MARKET_SELL] / window_seconds,
                "limit_buy": counts[OrderType.LIMIT_BUY] / window_seconds,
                "limit_sell": counts[OrderType.LIMIT_SELL] / window_seconds,
                "cancel": (counts[OrderType.CANCEL_BUY] + counts[OrderType.CANCEL_SELL]) / window_seconds,
            }
    
    def get_intensity_ratio(self) -> float:
        """
        Calculate the ratio of market to limit order intensity.
        
        High ratio indicates aggressive market conditions.
        Low ratio indicates passive, range-bound conditions.
        
        Returns:
            Ratio of market order rate to limit order rate
        """
        with self._lock:
            market_rate = (
                self.market_buy_ema.get_value() + 
                self.market_sell_ema.get_value()
            )
            limit_rate = (
                self.limit_buy_ema.get_value() + 
                self.limit_sell_ema.get_value()
            )
            
            if limit_rate > 0:
                return market_rate / limit_rate
            return float('inf') if market_rate > 0 else 0.0
    
    def predict_arrival_probability(
        self,
        order_type: OrderType,
        time_horizon: float,
    ) -> float:
        """
        Predict probability of at least one order arrival in time horizon.
        
        Uses Poisson process approximation with time-varying intensity.
        
        Args:
            order_type: Type of order to predict
            time_horizon: Time horizon in seconds
            
        Returns:
            Probability of at least one arrival (0 to 1)
        """
        with self._lock:
            # Get current intensity for this order type
            if order_type == OrderType.MARKET_BUY:
                intensity = self.market_buy_ema.get_value()
            elif order_type == OrderType.MARKET_SELL:
                intensity = self.market_sell_ema.get_value()
            elif order_type == OrderType.LIMIT_BUY:
                intensity = self.limit_buy_ema.get_value()
            elif order_type == OrderType.LIMIT_SELL:
                intensity = self.limit_sell_ema.get_value()
            else:
                intensity = self.cancel_ema.get_value()
            
            # Poisson probability: P(N >= 1) = 1 - exp(-lambda * t)
            if intensity <= 0:
                return 0.0
            
            return 1.0 - math.exp(-intensity * time_horizon)
    
    def register_observer(self, callback: callable) -> None:
        """Register an observer callback for real-time updates."""
        self._observers.append(callback)
    
    def unregister_observer(self, callback: callable) -> None:
        """Unregister an observer callback."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self, event: OrderEvent) -> None:
        """Notify all registered observers of new event."""
        for observer in self._observers:
            try:
                observer(event)
            except Exception as e:
                # Log error but continue notifying other observers
                print(f"Observer notification error: {e}")
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self.events.clear()
            self.market_buy_ema.reset()
            self.market_sell_ema.reset()
            self.limit_buy_ema.reset()
            self.limit_sell_ema.reset()
            self.cancel_ema.reset()
            self.aggressive_ema.reset()
            self.passive_ema.reset()
            self.last_update = 0.0


class MultiAssetArrivalTracker:
    """
    Track arrival rates across multiple assets simultaneously.
    
    Useful for relative value trading and cross-asset arbitrage.
    """
    
    def __init__(self, assets: List[str]):
        """
        Initialize multi-asset tracker.
        
        Args:
            assets: List of asset symbols to track
        """
        self.assets = assets
        self.trackers: Dict[str, ArrivalRateCalculator] = {
            asset: ArrivalRateCalculator() for asset in assets
        }
    
    def add_event(self, asset: str, event: OrderEvent) -> None:
        """Add event for specific asset."""
        if asset in self.trackers:
            self.trackers[asset].add_event(event)
    
    def get_relative_intensity(self, asset1: str, asset2: str) -> float:
        """
        Get relative arrival intensity between two assets.
        
        Returns ratio of total arrival rates.
        """
        if asset1 not in self.trackers or asset2 not in self.trackers:
            return 1.0
        
        stats1 = self.trackers[asset1].get_current_rates(asset1)
        stats2 = self.trackers[asset2].get_current_rates(asset2)
        
        if stats2.total_arrival_rate > 0:
            return stats1.total_arrival_rate / stats2.total_arrival_rate
        return float('inf') if stats1.total_arrival_rate > 0 else 1.0


if __name__ == "__main__":
    # Example usage and testing
    calculator = ArrivalRateCalculator()
    
    # Simulate some events
    base_time = time.time()
    for i in range(100):
        calculator.add_market_order(
            side=EventSide.BID if i % 2 == 0 else EventSide.ASK,
            volume=1.0 + (i % 10) * 0.1,
            price=50000.0 + i,
            timestamp=base_time + i * 0.1,
        )
        calculator.add_limit_order(
            side=EventSide.ASK if i % 2 == 0 else EventSide.BID,
            volume=2.0 + (i % 5) * 0.2,
            price=50000.0 + i + 10,
            timestamp=base_time + i * 0.1 + 0.05,
        )
    
    stats = calculator.get_current_rates("BTC")
    print(f"Market Buy Rate: {stats.market_buy_rate:.4f}")
    print(f"Market Sell Rate: {stats.market_sell_rate:.4f}")
    print(f"Imbalance: {stats.imbalance:.4f}")
    print(f"Aggressive Ratio: {stats.aggressive_ratio:.4f}")
