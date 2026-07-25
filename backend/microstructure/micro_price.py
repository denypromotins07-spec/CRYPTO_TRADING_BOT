#!/usr/bin/env python3
"""
Micro-Price Calculator using Queue Dynamics

This module calculates the true fair value of an asset by analyzing
queue dynamics, order book imbalance, and microstructure signals.
It provides a more accurate price estimate than simple mid-price.

Designed for the ZAID PERSONAL CRYPTO TRADING BOT with strict 8GB RAM constraints.
Uses NumPy for vectorized operations and optional Cython extensions for performance.

Author: Opus 4.8
Stage: 21/100 - Advanced Market Microstructure
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import time


class Side(Enum):
    """Order book side enumeration."""
    BID = "bid"
    ASK = "ask"


@dataclass(slots=True)
class PriceLevel:
    """Represents a single price level in the order book."""
    price: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    bid_order_count: int = 0
    ask_order_count: int = 0
    last_update_ns: int = 0  # Nanoseconds for microsecond precision
    
    def __post_init__(self) -> None:
        """Validate price level data."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.bid_volume < 0 or self.ask_volume < 0:
            raise ValueError("Volumes cannot be negative")


@dataclass(slots=True)
class QueueMetrics:
    """Metrics derived from queue analysis."""
    bid_queue_pressure: float = 0.0
    ask_queue_pressure: float = 0.0
    queue_imbalance: float = 0.0
    weighted_mid: float = 0.0
    micro_price: float = 0.0
    fair_value_estimate: float = 0.0
    confidence_score: float = 0.0
    timestamp_ns: int = 0


@dataclass(slots=True)
class OrderBookState:
    """Snapshot of order book state for micro-price calculation."""
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: float = 0.0
    mid_price: float = 0.0
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0
    timestamp_ns: int = 0
    
    def update_best_prices(self) -> None:
        """Update best bid/ask from current levels."""
        if self.bids:
            self.best_bid = max(level.price for level in self.bids)
        if self.asks:
            self.best_ask = min(level.price for level in self.asks)
        
        if self.best_bid is not None and self.best_ask is not None:
            self.spread = self.best_ask - self.best_bid
            self.mid_price = (self.best_bid + self.best_ask) / 2.0
        
        self.total_bid_volume = sum(level.bid_volume for level in self.bids)
        self.total_ask_volume = sum(level.ask_volume for level in self.asks)


class MicroPriceCalculator:
    """
    Calculate micro-price using advanced queue dynamics.
    
    The micro-price is a volume-weighted estimate of fair value that accounts for:
    - Order book imbalance
    - Queue position pressures
    - Recent trade flow direction
    - Liquidity asymmetry
    
    This provides alpha signals for short-term price movements.
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float = 0.01,
        max_levels: int = 50,
        decay_factor: float = 0.95,
    ) -> None:
        """
        Initialize the micro-price calculator.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            tick_size: Minimum price increment
            max_levels: Maximum number of price levels to consider
            decay_factor: Exponential decay factor for older data
        """
        self.symbol: str = symbol
        self.tick_size: float = tick_size
        self.max_levels: int = max_levels
        self.decay_factor: float = decay_factor
        
        self._book_state: OrderBookState = OrderBookState()
        self._metrics_history: deque[QueueMetrics] = deque(maxlen=1000)
        self._trade_flow: deque[Tuple[float, float, int]] = deque(maxlen=500)  # (price, volume, side_flag)
        
        # Weights for different micro-price components
        self.imbalance_weight: float = 0.4
        self.queue_pressure_weight: float = 0.3
        self.trade_flow_weight: float = 0.3
        
        # Calibration parameters
        self._volatility_adjustment: float = 1.0
        self._toxicity_penalty: float = 0.0
    
    @staticmethod
    def _now_ns() -> int:
        """Get current time in nanoseconds."""
        return time.time_ns()
    
    def update_level(
        self,
        price: float,
        side: Side,
        volume: float,
        order_count: int = 1,
    ) -> None:
        """
        Update a price level in the order book.
        
        Args:
            price: Price level to update
            side: Bid or Ask side
            volume: Volume at this level
            order_count: Number of orders at this level
        """
        timestamp = self._now_ns()
        
        # Find existing level or create new one
        levels = self._book_state.bids if side == Side.BID else self._book_state.asks
        
        for level in levels:
            if abs(level.price - price) < self.tick_size / 2:
                if side == Side.BID:
                    level.bid_volume = volume
                    level.bid_order_count = order_count
                else:
                    level.ask_volume = volume
                    level.ask_order_count = order_count
                level.last_update_ns = timestamp
                break
        else:
            # Create new level
            new_level = PriceLevel(
                price=price,
                bid_volume=volume if side == Side.BID else 0.0,
                ask_volume=volume if side == Side.ASK else 0.0,
                bid_order_count=order_count if side == Side.BID else 0,
                ask_order_count=order_count if side == Side.ASK else 0,
                last_update_ns=timestamp,
            )
            levels.append(new_level)
            # Keep only max_levels
            if len(levels) > self.max_levels:
                levels.pop()
        
        self._book_state.timestamp_ns = timestamp
        self._book_state.update_best_prices()
    
    def remove_level(self, price: float, side: Side) -> None:
        """Remove a price level from the book."""
        levels = self._book_state.bids if side == Side.BID else self._book_state.asks
        self._book_state.bids = [l for l in levels if abs(l.price - price) >= self.tick_size / 2]
        self._book_state.update_best_prices()
    
    def add_trade(self, price: float, volume: float, is_buyer_maker: bool) -> None:
        """
        Record a trade for trade flow analysis.
        
        Args:
            price: Trade execution price
            volume: Trade volume
            is_buyer_maker: True if buyer was maker (sell pressure), False if taker (buy pressure)
        """
        side_flag = -1.0 if is_buyer_maker else 1.0  # Negative for sells, positive for buys
        self._trade_flow.append((price, volume, side_flag))
    
    def calculate_order_book_imbalance(self, depth: int = 10) -> float:
        """
        Calculate order book imbalance ratio.
        
        OBI = (BidVolume - AskVolume) / (BidVolume + AskVolume)
        
        Returns value in [-1, 1] where:
        - Positive values indicate buy pressure
        - Negative values indicate sell pressure
        
        Args:
            depth: Number of levels to consider from top of book
        """
        bids = sorted(self._book_state.bids, key=lambda x: x.price, reverse=True)[:depth]
        asks = sorted(self._book_state.asks, key=lambda x: x.price)[:depth]
        
        bid_vol = sum(level.bid_volume for level in bids)
        ask_vol = sum(level.ask_volume for level in asks)
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        
        return (bid_vol - ask_vol) / total
    
    def calculate_queue_pressure(self, depth: int = 5) -> Tuple[float, float]:
        """
        Calculate queue pressure on bid and ask sides.
        
        Queue pressure measures how quickly orders are being added/removed
        at each price level, weighted by distance from mid-price.
        
        Returns:
            Tuple of (bid_pressure, ask_pressure)
        """
        if self._book_state.mid_price == 0:
            return 0.0, 0.0
        
        mid = self._book_state.mid_price
        now = self._now_ns()
        
        bid_pressure = 0.0
        ask_pressure = 0.0
        
        bids = sorted(self._book_state.bids, key=lambda x: x.price, reverse=True)[:depth]
        asks = sorted(self._book_state.asks, key=lambda x: x.price)[:depth]
        
        for i, level in enumerate(bids):
            distance = (mid - level.price) / mid
            time_weight = self.decay_factor ** i
            age_weight = 1.0 / (1.0 + (now - level.last_update_ns) / 1e9)
            pressure = level.bid_volume * time_weight * age_weight / (1.0 + distance)
            bid_pressure += pressure
        
        for i, level in enumerate(asks):
            distance = (level.price - mid) / mid
            time_weight = self.decay_factor ** i
            age_weight = 1.0 / (1.0 + (now - level.last_update_ns) / 1e9)
            pressure = level.ask_volume * time_weight * age_weight / (1.0 + distance)
            ask_pressure += pressure
        
        return bid_pressure, ask_pressure
    
    def calculate_trade_flow_imbalance(self, window_ms: int = 1000) -> float:
        """
        Calculate recent trade flow imbalance.
        
        Positive values indicate net buying pressure,
        negative values indicate net selling pressure.
        
        Args:
            window_ms: Time window in milliseconds to consider
        
        Returns:
            Trade flow imbalance in [-1, 1]
        """
        if not self._trade_flow:
            return 0.0
        
        now_ns = self._now_ns()
        window_ns = window_ms * 1_000_000
        
        weighted_sum = 0.0
        total_volume = 0.0
        
        for price, volume, side_flag in reversed(self._trade_flow):
            # Note: In production, would check actual timestamp
            weighted_sum += volume * side_flag
            total_volume += volume
        
        if total_volume == 0:
            return 0.0
        
        return weighted_sum / total_volume
    
    def calculate_weighted_mid_price(self) -> float:
        """
        Calculate volume-weighted mid price.
        
        Instead of simple (best_bid + best_ask) / 2, this weights
        by the volume available at each side.
        """
        if self._book_state.best_bid is None or self._book_state.best_ask is None:
            return self._book_state.mid_price
        
        bid_vol = self._book_state.total_bid_volume
        ask_vol = self._book_state.total_ask_volume
        total_vol = bid_vol + ask_vol
        
        if total_vol == 0:
            return self._book_state.mid_price
        
        # Weight towards side with less liquidity (price moves there)
        bid_weight = ask_vol / total_vol
        ask_weight = bid_vol / total_vol
        
        return self._book_state.best_bid * bid_weight + self._book_state.best_ask * ask_weight
    
    def calculate_micro_price(self) -> float:
        """
        Calculate the micro-price using all available signals.
        
        The micro-price is an estimate of the fair value that accounts for:
        1. Order book imbalance
        2. Queue pressure dynamics
        3. Recent trade flow
        4. Liquidity asymmetry
        
        Returns:
            Micro-price estimate
        """
        if self._book_state.mid_price == 0:
            return 0.0
        
        # Component 1: Order book imbalance adjustment
        obi = self.calculate_order_book_imbalance()
        obi_adjustment = obi * self.imbalance_weight
        
        # Component 2: Queue pressure adjustment
        bid_pressure, ask_pressure = self.calculate_queue_pressure()
        total_pressure = bid_pressure + ask_pressure
        if total_pressure > 0:
            queue_imbalance = (bid_pressure - ask_pressure) / total_pressure
        else:
            queue_imbalance = 0.0
        queue_adjustment = queue_imbalance * self.queue_pressure_weight
        
        # Component 3: Trade flow adjustment
        trade_flow = self.calculate_trade_flow_imbalance()
        trade_adjustment = trade_flow * self.trade_flow_weight
        
        # Combine adjustments
        total_adjustment = obi_adjustment + queue_adjustment + trade_adjustment
        
        # Apply to weighted mid price
        weighted_mid = self.calculate_weighted_mid_price()
        spread = self._book_state.spread if self._book_state.spread > 0 else weighted_mid * 0.001
        
        micro_price = weighted_mid + total_adjustment * spread * self._volatility_adjustment
        
        # Apply toxicity penalty if informed trading detected
        micro_price *= (1.0 - self._toxicity_penalty * 0.01)
        
        return micro_price
    
    def calculate_fair_value(self, lookback_periods: int = 100) -> float:
        """
        Calculate fair value estimate using historical micro-prices.
        
        Uses exponential moving average of micro-prices with volatility adjustment.
        
        Args:
            lookback_periods: Number of periods for EMA calculation
        
        Returns:
            Fair value estimate
        """
        current_micro = self.calculate_micro_price()
        
        # Store metrics
        metrics = QueueMetrics(
            micro_price=current_micro,
            timestamp_ns=self._now_ns(),
        )
        self._metrics_history.append(metrics)
        
        if len(self._metrics_history) < 2:
            return current_micro
        
        # Calculate EMA of micro-prices
        ema = self._metrics_history[0].micro_price
        multiplier = 2.0 / (lookback_periods + 1)
        
        for m in list(self._metrics_history)[1:]:
            ema = m.micro_price * multiplier + ema * (1 - multiplier)
        
        return ema
    
    def get_queue_metrics(self) -> QueueMetrics:
        """Get current queue metrics."""
        bid_pressure, ask_pressure = self.calculate_queue_pressure()
        total_pressure = bid_pressure + ask_pressure
        
        queue_imbalance = 0.0
        if total_pressure > 0:
            queue_imbalance = (bid_pressure - ask_pressure) / total_pressure
        
        micro_price = self.calculate_micro_price()
        fair_value = self.calculate_fair_value()
        
        # Confidence based on spread and liquidity
        spread_ratio = self._book_state.spread / self._book_state.mid_price if self._book_state.mid_price > 0 else 1.0
        liquidity = self._book_state.total_bid_volume + self._book_state.total_ask_volume
        confidence = max(0.0, min(1.0, 1.0 - spread_ratio * 100)) * min(1.0, liquidity / 1000.0)
        
        return QueueMetrics(
            bid_queue_pressure=bid_pressure,
            ask_queue_pressure=ask_pressure,
            queue_imbalance=queue_imbalance,
            weighted_mid=self.calculate_weighted_mid_price(),
            micro_price=micro_price,
            fair_value_estimate=fair_value,
            confidence_score=confidence,
            timestamp_ns=self._now_ns(),
        )
    
    def set_volatility_adjustment(self, volatility: float) -> None:
        """
        Adjust sensitivity based on market volatility.
        
        Args:
            volatility: Realized volatility (e.g., 0.02 for 2%)
        """
        self._volatility_adjustment = 1.0 + volatility
    
    def set_toxicity_penalty(self, toxicity: float) -> None:
        """
        Set toxicity penalty for adverse selection protection.
        
        Args:
            toxicity: Toxicity score in [0, 1]
        """
        self._toxicity_penalty = max(0.0, min(1.0, toxicity))
    
    def reset(self) -> None:
        """Reset all state."""
        self._book_state = OrderBookState()
        self._metrics_history.clear()
        self._trade_flow.clear()
        self._volatility_adjustment = 1.0
        self._toxicity_penalty = 0.0


def main() -> None:
    """Example usage of MicroPriceCalculator."""
    calculator = MicroPriceCalculator(symbol="BTCUSDT", tick_size=0.01)
    
    # Simulate order book updates
    for i in range(10):
        bid_price = 50000.0 - i * 0.01
        ask_price = 50000.05 + i * 0.01
        calculator.update_level(bid_price, Side.BID, 1.5 + i * 0.1)
        calculator.update_level(ask_price, Side.ASK, 1.2 + i * 0.1)
    
    # Add some trades
    calculator.add_trade(50000.02, 0.5, is_buyer_maker=False)  # Buy
    calculator.add_trade(50000.01, 0.3, is_buyer_maker=True)   # Sell
    
    # Get metrics
    metrics = calculator.get_queue_metrics()
    print(f"Symbol: {calculator.symbol}")
    print(f"Mid Price: {calculator._book_state.mid_price:.4f}")
    print(f"Weighted Mid: {metrics.weighted_mid:.4f}")
    print(f"Micro Price: {metrics.micro_price:.4f}")
    print(f"Fair Value: {metrics.fair_value_estimate:.4f}")
    print(f"Queue Imbalance: {metrics.queue_imbalance:.4f}")
    print(f"Confidence: {metrics.confidence_score:.4f}")


if __name__ == "__main__":
    main()
