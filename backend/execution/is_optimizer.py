#!/usr/bin/env python3
"""
IS Optimizer: Dynamic Aggression Adjustment Based on Volatility
Implements Implementation Shortfall optimization with real-time volatility scaling.
Uses C-extensions where needed for performance-critical calculations.

Stage 13: Advanced Execution Algorithms
Target: Minimize market impact to secure 8k-20k INR/hour
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import threading


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class ArrivalPriceSnapshot:
    """Immutable snapshot of arrival price benchmark"""
    timestamp_ns: int
    price: float
    volume: float
    side: OrderSide
    volatility: float


@dataclass
class FillEvent:
    """Record of individual fill execution"""
    timestamp_ns: int
    price: float
    volume: float
    fee_bps: float = 0.0


@dataclass
class ISMetrics:
    """Implementation Shortfall metrics"""
    total_is_bps: float
    timing_cost_bps: float
    market_impact_bps: float
    opportunity_cost_bps: float
    elapsed_us: int
    fill_rate: float


class ISOptimizer:
    """
    Dynamic Implementation Shortfall optimizer.
    Adjusts execution aggression based on real-time volatility and market movement.
    
    Features:
    - Real-time volatility scaling using exponential moving average
    - Adaptive aggression control (0.1 to 1.0)
    - Market move acceleration when price moves against position
    - Thread-safe state management
    """
    
    def __init__(
        self,
        base_aggression: float = 0.5,
        max_slippage_bps: float = 50.0,
        volatility_window: int = 100,
        acceleration_threshold_bps: float = 10.0,
        volatility_multiplier: float = 0.15,
    ):
        self.base_aggression = base_aggression
        self.current_aggression = base_aggression
        self.max_slippage_bps = max_slippage_bps
        self.acceleration_threshold_bps = acceleration_threshold_bps
        self.volatility_multiplier = volatility_multiplier
        
        # State tracking
        self._arrival_price: Optional[ArrivalPriceSnapshot] = None
        self._fills: List[FillEvent] = []
        self._start_time_ns: int = 0
        self._volatility_history: List[float] = []
        self._volatility_window = volatility_window
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self._total_filled_value = 0.0
        self._total_filled_cost = 0.0
    
    def set_arrival_price(self, snapshot: ArrivalPriceSnapshot) -> None:
        """Set the arrival price benchmark for IS calculation"""
        with self._lock:
            self._arrival_price = snapshot
            self._start_time_ns = time.time_ns()
            self._fills.clear()
            self._total_filled_value = 0.0
            self._total_filled_cost = 0.0
            self.current_aggression = self.base_aggression
    
    def record_fill(self, price: float, volume: float, fee_bps: float = 0.0) -> None:
        """Record a fill event"""
        with self._lock:
            fill = FillEvent(
                timestamp_ns=time.time_ns(),
                price=price,
                volume=volume,
                fee_bps=fee_bps,
            )
            self._fills.append(fill)
            self._total_filled_value += volume
            self._total_filled_cost += price * volume
    
    def update_volatility(self, volatility: float) -> None:
        """Update rolling volatility estimate"""
        with self._lock:
            self._volatility_history.append(volatility)
            if len(self._volatility_history) > self._volatility_window:
                self._volatility_history.pop(0)
    
    def get_current_volatility(self) -> float:
        """Get current EMA volatility estimate"""
        with self._lock:
            if not self._volatility_history:
                return 0.0
            # Exponential moving average for recent volatility
            ema_alpha = 2.0 / (min(len(self._volatility_history), 20) + 1)
            ema = self._volatility_history[0]
            for vol in self._volatility_history[1:]:
                ema = ema_alpha * vol + (1 - ema_alpha) * ema
            return ema
    
    def adjust_aggression(self, current_price: float) -> float:
        """
        Dynamically adjust execution aggression.
        Accelerates if market moves against position.
        """
        with self._lock:
            if self._arrival_price is None:
                return self.current_aggression
            
            arrival = self._arrival_price
            volatility = self.get_current_volatility()
            
            # Calculate market move in bps
            if arrival.price > 0:
                market_move_bps = ((current_price - arrival.price) / arrival.price) * 10000
            else:
                market_move_bps = 0.0
            
            # Determine if market moved against us
            adverse_move = False
            if arrival.side == OrderSide.BUY and market_move_bps > 0:
                adverse_move = True  # Price went up against buy order
            elif arrival.side == OrderSide.SELL and market_move_bps < 0:
                adverse_move = True  # Price went down against sell order
            
            # Adjust aggression
            vol_adjustment = volatility * self.volatility_multiplier
            
            if adverse_move and abs(market_move_bps) > self.acceleration_threshold_bps:
                # Accelerate execution when market moves against us
                self.current_aggression = min(
                    1.0,
                    max(0.1, self.base_aggression * (1.0 + vol_adjustment + abs(market_move_bps) / 100))
                )
            else:
                self.current_aggression = self.base_aggression
            
            return self.current_aggression
    
    def calculate_is_bps(self) -> Optional[float]:
        """Calculate current Implementation Shortfall in basis points"""
        with self._lock:
            if self._arrival_price is None or self._total_filled_value == 0:
                return None
            
            arrival = self._arrival_price
            avg_execution_price = self._total_filled_cost / self._total_filled_value
            
            if arrival.side == OrderSide.BUY:
                is_bps = ((avg_execution_price - arrival.price) / arrival.price) * 10000
            else:  # SELL
                is_bps = ((arrival.price - avg_execution_price) / arrival.price) * 10000
            
            return is_bps
    
    def check_slippage_exceeded(self) -> bool:
        """Check if slippage exceeds maximum threshold"""
        is_bps = self.calculate_is_bps()
        if is_bps is None:
            return False
        return abs(is_bps) > self.max_slippage_bps
    
    def get_metrics(self) -> Optional[ISMetrics]:
        """Get comprehensive IS metrics"""
        with self._lock:
            if self._arrival_price is None:
                return None
            
            elapsed_us = (time.time_ns() - self._start_time_ns) // 1000
            fill_rate = len(self._fills) / max(1, elapsed_us / 1000000)  # fills per second
            
            total_is_bps = self.calculate_is_bps() or 0.0
            
            # Decompose IS into components (simplified model)
            timing_cost_bps = total_is_bps * 0.6  # 60% attributed to timing
            market_impact_bps = total_is_bps * 0.3  # 30% to market impact
            opportunity_cost_bps = total_is_bps * 0.1  # 10% opportunity cost
            
            return ISMetrics(
                total_is_bps=total_is_bps,
                timing_cost_bps=timing_cost_bps,
                market_impact_bps=market_impact_bps,
                opportunity_cost_bps=opportunity_cost_bps,
                elapsed_us=elapsed_us,
                fill_rate=fill_rate,
            )
    
    def reset(self) -> None:
        """Reset optimizer state for new order"""
        with self._lock:
            self._arrival_price = None
            self._fills.clear()
            self._total_filled_value = 0.0
            self._total_filled_cost = 0.0
            self.current_aggression = self.base_aggression


class VolatilityEstimator:
    """
    High-performance volatility estimator using numpy C-extensions.
    Calculates realized volatility from tick data.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._prices: np.ndarray = np.zeros(window_size, dtype=np.float64)
        self._index = 0
        self._count = 0
    
    def update(self, price: float) -> float:
        """Update with new price and return current volatility"""
        self._prices[self._index] = price
        self._index = (self._index + 1) % self.window_size
        self._count = min(self._count + 1, self.window_size)
        
        if self._count < 2:
            return 0.0
        
        # Calculate log returns
        valid_prices = self._prices[:self._count]
        log_returns = np.diff(np.log(valid_prices))
        
        if len(log_returns) == 0:
            return 0.0
        
        # Annualized realized volatility (assuming crypto trades 24/7)
        volatility = np.std(log_returns) * np.sqrt(365 * 24 * 60)  # per minute
        return float(volatility)


if __name__ == "__main__":
    # Test the IS optimizer
    optimizer = ISOptimizer(base_aggression=0.5, max_slippage_bps=50.0)
    
    # Set arrival price
    arrival = ArrivalPriceSnapshot(
        timestamp_ns=time.time_ns(),
        price=50000.0,
        volume=1.0,
        side=OrderSide.BUY,
        volatility=0.02,
    )
    optimizer.set_arrival_price(arrival)
    
    # Simulate fills
    optimizer.record_fill(50010.0, 0.3)
    optimizer.record_fill(50015.0, 0.4)
    optimizer.record_fill(50020.0, 0.3)
    
    # Update volatility
    vol_estimator = VolatilityEstimator()
    for p in [50000, 50010, 50005, 50015, 50020]:
        vol_estimator.update(p)
    
    current_vol = vol_estimator.update(50025)
    optimizer.update_volatility(current_vol)
    
    # Adjust aggression
    new_aggression = optimizer.adjust_aggression(50030.0)
    print(f"Adjusted aggression: {new_aggression:.3f}")
    
    # Get metrics
    metrics = optimizer.get_metrics()
    if metrics:
        print(f"Total IS: {metrics.total_is_bps:.2f} bps")
        print(f"Elapsed: {metrics.elapsed_us} μs")
