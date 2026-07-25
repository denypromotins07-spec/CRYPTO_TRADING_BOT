#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
File: backend/microstructure/micro_price.py
Chapter 1: Queue Position Tracking, Fill Probability, and Micro-Price Calculation

Purpose: Calculate true fair value using queue dynamics (micro-price)
Constraints: Optimized for 8GB RAM, uses NumPy for vectorized operations
Target: AMD Ryzen AI 5 laptop

The micro-price is a weighted average of bid and ask prices where weights
are determined by queue sizes. It provides a more accurate fair value than
the mid-price, especially when order book imbalance exists.

Formula: micro_price = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)

Design Patterns: Strategy Pattern for different micro-price calculation methods
Type Hinting: Strict typing for production reliability
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Protocol
from enum import Enum
import numpy as np
from collections import deque
import time


class MicroPriceStrategy(Enum):
    """Different strategies for calculating micro-price"""
    VOLUME_WEIGHTED = "volume_weighted"
    QUEUE_IMBALANCE = "queue_imbalance"
    KYLE_LAMBDA = "kyle_lambda"
    ADAPTIVE = "adaptive"


@dataclass(slots=True)
class OrderBookLevel:
    """Represents a single price level in the order book"""
    price: float
    quantity: float
    order_count: int = 0
    timestamp_ns: int = 0
    
    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.quantity < 0:
            raise ValueError("Quantity cannot be negative")


@dataclass(slots=True)
class MicroPriceSnapshot:
    """Snapshot of micro-price calculation with metadata"""
    micro_price: float
    mid_price: float
    spread: float
    spread_bps: float
    bid_imbalance: float  # -1 to 1, where 1 means all bids
    queue_pressure: float  # Pressure from queue dynamics
    timestamp_ns: int
    confidence_score: float  # 0 to 1, based on liquidity depth
    strategy_used: MicroPriceStrategy


class MicroPriceCalculator(Protocol):
    """Protocol for micro-price calculation strategies"""
    
    def calculate(self, best_bid: float, best_ask: float, 
                  bid_volume: float, ask_volume: float) -> float:
        """Calculate micro-price given best bid/ask and volumes"""
        ...


class VolumeWeightedMicroPrice:
    """Standard volume-weighted micro-price calculation"""
    
    def calculate(self, best_bid: float, best_ask: float,
                  bid_volume: float, ask_volume: float) -> float:
        """
        Calculate micro-price using volume weighting.
        
        The micro-price moves toward the side with less liquidity,
        as that side is more likely to be depleted first.
        """
        if bid_volume + ask_volume == 0:
            return (best_bid + best_ask) / 2.0
        
        # Weight inversely proportional to volume
        # Less volume = more pressure = price moves toward that side
        bid_weight = ask_volume / (bid_volume + ask_volume)
        ask_weight = bid_volume / (bid_volume + ask_volume)
        
        return bid_weight * best_bid + ask_weight * best_ask


class QueueImbalanceMicroPrice:
    """Micro-price adjusted for queue imbalance"""
    
    def __init__(self, imbalance_sensitivity: float = 0.5):
        self.imbalance_sensitivity = imbalance_sensitivity
    
    def calculate(self, best_bid: float, best_ask: float,
                  bid_volume: float, ask_volume: float) -> float:
        """
        Calculate micro-price with queue imbalance adjustment.
        
        When bid volume >> ask volume, micro-price > mid-price
        When ask volume >> bid volume, micro-price < mid-price
        """
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        
        if bid_volume + ask_volume == 0:
            return mid_price
        
        # Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # Adjust mid-price based on imbalance
        adjustment = imbalance * spread * self.imbalance_sensitivity
        
        return mid_price + adjustment


class KyleLambdaMicroPrice:
    """
    Micro-price using Kyle's Lambda model for price impact.
    
    Incorporates informed trading probability into fair value estimation.
    """
    
    def __init__(self, lambda_param: float = 0.001):
        self.lambda_param = lambda_param  # Price impact coefficient
    
    def calculate(self, best_bid: float, best_ask: float,
                  bid_volume: float, ask_volume: float) -> float:
        """
        Calculate micro-price using Kyle's Lambda model.
        
        Lambda measures how much price moves per unit of order flow.
        """
        mid_price = (best_bid + best_ask) / 2.0
        
        # Net order flow (positive = buying pressure)
        net_flow = bid_volume - ask_volume
        
        # Price adjustment based on order flow toxicity
        adjustment = self.lambda_param * net_flow
        
        return mid_price + adjustment


class AdaptiveMicroPriceCalculator:
    """
    Adaptive micro-price calculator that selects the best strategy
    based on current market conditions.
    """
    
    def __init__(self):
        self.volume_strategy = VolumeWeightedMicroPrice()
        self.imbalance_strategy = QueueImbalanceMicroPrice()
        self.kyle_strategy = KyleLambdaMicroPrice()
        self.current_strategy = MicroPriceStrategy.VOLUME_WEIGHTED
        self.volatility_window: deque = deque(maxlen=100)
        self.imbalance_history: deque = deque(maxlen=100)
    
    def calculate(self, best_bid: float, best_ask: float,
                  bid_volume: float, ask_volume: float,
                  volatility: Optional[float] = None) -> Tuple[float, MicroPriceStrategy]:
        """
        Calculate micro-price using adaptive strategy selection.
        
        In high volatility: Use Kyle's Lambda (accounts for informed trading)
        In normal conditions: Use volume-weighted
        When strong imbalance: Use queue imbalance strategy
        """
        # Track metrics for strategy selection
        if volatility is not None:
            self.volatility_window.append(volatility)
        
        imbalance = (bid_volume - ask_volume) / max(bid_volume + ask_volume, 1e-10)
        self.imbalance_history.append(imbalance)
        
        # Strategy selection logic
        if len(self.volatility_window) >= 10:
            avg_volatility = np.mean(self.volatility_window)
            if avg_volatility > 0.02:  # High volatility (>2%)
                self.current_strategy = MicroPriceStrategy.KYLE_LAMBDA
                price = self.kyle_strategy.calculate(best_bid, best_ask, 
                                                      bid_volume, ask_volume)
            elif abs(imbalance) > 0.7:  # Strong imbalance
                self.current_strategy = MicroPriceStrategy.QUEUE_IMBALANCE
                price = self.imbalance_strategy.calculate(best_bid, best_ask,
                                                           bid_volume, ask_volume)
            else:
                self.current_strategy = MicroPriceStrategy.VOLUME_WEIGHTED
                price = self.volume_strategy.calculate(best_bid, best_ask,
                                                        bid_volume, ask_volume)
        else:
            # Default to volume-weighted until we have enough history
            self.current_strategy = MicroPriceStrategy.VOLUME_WEIGHTED
            price = self.volume_strategy.calculate(best_bid, best_ask,
                                                    bid_volume, ask_volume)
        
        return price, self.current_strategy


@dataclass
class MicroPriceEngine:
    """
    Main engine for micro-price calculation with full order book integration.
    
    Tracks multiple levels of the order book for more accurate calculations.
    Uses pre-allocated arrays to minimize memory allocations during updates.
    """
    
    # Configuration
    max_levels: int = 20  # Number of levels to consider
    decay_factor: float = 0.9  # Exponential decay for deeper levels
    
    # State
    bid_prices: np.ndarray = field(default_factory=lambda: np.zeros(20))
    bid_volumes: np.ndarray = field(default_factory=lambda: np.zeros(20))
    ask_prices: np.ndarray = field(default_factory=lambda: np.zeros(20))
    ask_volumes: np.ndarray = field(default_factory=lambda: np.zeros(20))
    last_update_ns: int = 0
    
    # Strategies
    adaptive_calculator: AdaptiveMicroPriceCalculator = field(
        default_factory=AdaptiveMicroPriceCalculator
    )
    
    # History for analytics
    price_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def __post_init__(self):
        # Pre-allocate numpy arrays for zero-copy updates
        self.bid_prices = np.zeros(self.max_levels, dtype=np.float64)
        self.bid_volumes = np.zeros(self.max_levels, dtype=np.float64)
        self.ask_prices = np.zeros(self.max_levels, dtype=np.float64)
        self.ask_volumes = np.zeros(self.max_levels, dtype=np.float64)
    
    def update_order_book(self, bids: List[OrderBookLevel], asks: List[OrderBookLevel]) -> None:
        """
        Update internal state with new order book snapshot.
        
        Uses vectorized operations for efficiency.
        No heap allocations after initialization.
        """
        self.last_update_ns = time.time_ns()
        
        # Update bids (vectorized)
        num_bids = min(len(bids), self.max_levels)
        if num_bids > 0:
            self.bid_prices[:num_bids] = [b.price for b in bids[:num_bids]]
            self.bid_volumes[:num_bids] = [b.quantity for b in bids[:num_bids]]
        
        # Update asks (vectorized)
        num_asks = min(len(asks), self.max_levels)
        if num_asks > 0:
            self.ask_prices[:num_asks] = [a.price for a in asks[:num_asks]]
            self.ask_volumes[:num_asks] = [a.quantity for a in asks[:num_asks]]
    
    def calculate_micro_price(self, 
                              strategy: Optional[MicroPriceStrategy] = None) -> MicroPriceSnapshot:
        """
        Calculate comprehensive micro-price snapshot.
        
        Returns detailed metrics including:
        - Micro-price (fair value estimate)
        - Bid-ask spread in basis points
        - Queue imbalance metrics
        - Confidence score based on liquidity
        """
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            raise ValueError("Order book not initialized")
        
        best_bid = self.bid_prices[0]
        best_ask = self.ask_prices[0]
        
        if best_bid <= 0 or best_ask <= 0:
            raise ValueError("Invalid prices in order book")
        
        # Calculate effective volumes with exponential decay
        # Deeper levels have less impact on micro-price
        decay_weights = np.array([self.decay_factor ** i for i in range(self.max_levels)])
        
        effective_bid_volume = np.sum(self.bid_volumes * decay_weights)
        effective_ask_volume = np.sum(self.ask_volumes * decay_weights)
        
        # Mid-price and spread
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Queue imbalance: ranges from -1 (all asks) to 1 (all bids)
        total_volume = effective_bid_volume + effective_ask_volume
        if total_volume > 0:
            bid_imbalance = (effective_bid_volume - effective_ask_volume) / total_volume
        else:
            bid_imbalance = 0.0
        
        # Calculate micro-price using selected or adaptive strategy
        if strategy is None:
            micro_price, used_strategy = self.adaptive_calculator.calculate(
                best_bid, best_ask, effective_bid_volume, effective_ask_volume
            )
        elif strategy == MicroPriceStrategy.VOLUME_WEIGHTED:
            calc = VolumeWeightedMicroPrice()
            micro_price = calc.calculate(best_bid, best_ask, 
                                         effective_bid_volume, effective_ask_volume)
            used_strategy = MicroPriceStrategy.VOLUME_WEIGHTED
        elif strategy == MicroPriceStrategy.QUEUE_IMBALANCE:
            calc = QueueImbalanceMicroPrice()
            micro_price = calc.calculate(best_bid, best_ask,
                                         effective_bid_volume, effective_ask_volume)
            used_strategy = MicroPriceStrategy.QUEUE_IMBALANCE
        elif strategy == MicroPriceStrategy.KYLE_LAMBDA:
            calc = KyleLambdaMicroPrice()
            micro_price = calc.calculate(best_bid, best_ask,
                                         effective_bid_volume, effective_ask_volume)
            used_strategy = MicroPriceStrategy.KYLE_LAMBDA
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Queue pressure: directional pressure from queue dynamics
        queue_pressure = bid_imbalance * (spread / mid_price) if mid_price > 0 else 0
        
        # Confidence score based on liquidity depth
        # Higher volume = higher confidence in micro-price
        total_liquidity = np.sum(self.bid_volumes[:5]) + np.sum(self.ask_volumes[:5])
        confidence_score = min(1.0, total_liquidity / 1000000)  # Normalize to 1M units
        
        snapshot = MicroPriceSnapshot(
            micro_price=micro_price,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            bid_imbalance=bid_imbalance,
            queue_pressure=queue_pressure,
            timestamp_ns=self.last_update_ns,
            confidence_score=confidence_score,
            strategy_used=used_strategy
        )
        
        # Store in history for analytics
        self.price_history.append(snapshot)
        
        return snapshot
    
    def get_weighted_mid_price(self) -> float:
        """
        Calculate volume-weighted mid-price across multiple levels.
        
        More robust than simple mid-price during temporary imbalances.
        """
        # Get cumulative volumes
        bid_cumvol = np.cumsum(self.bid_volumes)
        ask_cumvol = np.cumsum(self.ask_volumes)
        
        # Find price levels where cumulative volume reaches threshold
        threshold = min(bid_cumvol[-1], ask_cumvol[-1]) * 0.5
        
        bid_idx = np.searchsorted(bid_cumvol, threshold)
        ask_idx = np.searchsorted(ask_cumvol, threshold)
        
        if bid_idx < self.max_levels and ask_idx < self.max_levels:
            return (self.bid_prices[bid_idx] + self.ask_prices[ask_idx]) / 2.0
        
        return (self.bid_prices[0] + self.ask_prices[0]) / 2.0
    
    def get_effective_spread(self, volume_threshold: float = 10000) -> float:
        """
        Calculate effective spread for executing a given volume.
        
        Considers market depth to estimate true execution cost.
        """
        bid_cumvol = np.cumsum(self.bid_volumes)
        ask_cumvol = np.cumsum(self.ask_volumes)
        
        # Find levels needed to fill threshold volume
        bid_level = np.searchsorted(bid_cumvol, volume_threshold)
        ask_level = np.searchsorted(ask_cumvol, volume_threshold)
        
        if bid_level >= self.max_levels:
            bid_level = self.max_levels - 1
        if ask_level >= self.max_levels:
            ask_level = self.max_levels - 1
        
        # Effective spread is the spread at the required depth
        effective_bid = self.bid_prices[bid_level]
        effective_ask = self.ask_prices[ask_level]
        
        return effective_ask - effective_bid
    
    def get_analytics(self) -> Dict[str, float]:
        """Get statistical analytics from price history"""
        if len(self.price_history) < 2:
            return {}
        
        snapshots = list(self.price_history)
        micro_prices = np.array([s.micro_price for s in snapshots])
        spreads = np.array([s.spread_bps for s in snapshots])
        imbalances = np.array([s.bid_imbalance for s in snapshots])
        
        return {
            "micro_price_mean": float(np.mean(micro_prices)),
            "micro_price_std": float(np.std(micro_prices)),
            "micro_price_min": float(np.min(micro_prices)),
            "micro_price_max": float(np.max(micro_prices)),
            "spread_mean_bps": float(np.mean(spreads)),
            "spread_std_bps": float(np.std(spreads)),
            "imbalance_mean": float(np.mean(imbalances)),
            "imbalance_std": float(np.std(imbalances)),
            "price_change_pct": float((micro_prices[-1] - micro_prices[0]) / micro_prices[0] * 100),
        }


def main():
    """Example usage demonstrating micro-price calculation"""
    # Initialize engine
    engine = MicroPriceEngine(max_levels=10)
    
    # Create sample order book
    bids = [
        OrderBookLevel(price=49999.0, quantity=5.0),
        OrderBookLevel(price=49998.0, quantity=10.0),
        OrderBookLevel(price=49997.0, quantity=15.0),
    ]
    
    asks = [
        OrderBookLevel(price=50001.0, quantity=3.0),  # Less liquidity on ask
        OrderBookLevel(price=50002.0, quantity=8.0),
        OrderBookLevel(price=50003.0, quantity=12.0),
    ]
    
    # Update order book
    engine.update_order_book(bids, asks)
    
    # Calculate micro-price
    snapshot = engine.calculate_micro_price()
    
    print(f"Mid Price: ${snapshot.mid_price:.2f}")
    print(f"Micro Price: ${snapshot.micro_price:.2f}")
    print(f"Spread: {snapshot.spread_bps:.2f} bps")
    print(f"Bid Imbalance: {snapshot.bid_imbalance:.4f}")
    print(f"Strategy Used: {snapshot.strategy_used.value}")
    print(f"Confidence: {snapshot.confidence_score:.2%}")
    
    # Analytics
    analytics = engine.get_analytics()
    print(f"\nAnalytics: {analytics}")


if __name__ == "__main__":
    main()
