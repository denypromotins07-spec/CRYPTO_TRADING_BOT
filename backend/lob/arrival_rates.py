#!/usr/bin/env python3
"""
Arrival Rates Calculator for Order Book Events

This module calculates the intensity of market vs limit order arrivals
using Hawkes process outputs and empirical observations. It differentiates
between aggressive (taker) and passive (maker) order flow.

Key Features:
- Real-time intensity estimation for multiple order types
- Volume-weighted arrival rate calculations
- Adaptive baseline estimation using rolling windows
- Strict type hinting for memory safety
- C-extension compatible structures for performance

Mathematical Foundation:
- Market order intensity: λ_m(t) = μ_m + α_m * Σ exp(-β_m * (t - t_i^m))
- Limit order intensity: λ_l(t) = μ_l + α_l * Σ exp(-β_l * (t - t_i^l))
- Cancellation intensity: λ_c(t) = μ_c + α_c * Σ exp(-β_c * (t - t_i^c))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum, auto
import time
import math
from threading import Lock


class OrderType(Enum):
    """Enumeration of order types for arrival tracking."""
    MARKET_BUY = auto()
    MARKET_SELL = auto()
    LIMIT_BUY = auto()
    LIMIT_SELL = auto()
    CANCEL_BUY = auto()
    CANCEL_SELL = auto()
    AMEND_BUY = auto()
    AMEND_SELL = auto()


@dataclass
class OrderEvent:
    """Represents a single order book event."""
    timestamp_us: int
    order_type: OrderType
    volume: float
    price: float
    order_id: str
    
    @property
    def timestamp_s(self) -> float:
        """Convert microsecond timestamp to seconds."""
        return self.timestamp_us / 1_000_000.0


@dataclass
class HawkesParameters:
    """Parameters for Hawkes process intensity calculation."""
    mu: float  # Baseline intensity
    alpha: float  # Excitation coefficient
    beta: float  # Decay rate
    
    def validate(self) -> bool:
        """Validate parameters for stationarity."""
        if self.mu <= 0:
            return False
        if self.alpha <= 0 or self.alpha >= self.beta:
            return False
        if self.beta <= 0:
            return False
        return True
    
    @property
    def branching_ratio(self) -> float:
        """Calculate branching ratio (should be < 1 for stationarity)."""
        return self.alpha / self.beta


@dataclass
class IntensityState:
    """Current state of intensity calculation for one order type."""
    current_intensity: float
    last_update_us: int
    event_count: int
    total_volume: float
    params: HawkesParameters


class ArrivalRateCalculator:
    """
    Calculate arrival rates for different order types using Hawkes processes.
    
    This class maintains separate Hawkes processes for each order type and
    provides O(1) intensity queries after each event update.
    """
    
    def __init__(
        self,
        window_size_s: float = 60.0,
        max_events: int = 10000,
    ):
        """
        Initialize the arrival rate calculator.
        
        Args:
            window_size_s: Size of rolling window for baseline estimation
            max_events: Maximum events to store per order type
        """
        self.window_size_s = window_size_s
        self.max_events = max_events
        
        # Event queues per order type
        self.event_queues: Dict[OrderType, Deque[OrderEvent]] = {
            ot: deque(maxlen=max_events) for ot in OrderType
        }
        
        # Intensity states per order type
        self.intensity_states: Dict[OrderType, IntensityState] = {}
        
        # Default parameters (should be calibrated from historical data)
        self._init_default_parameters()
        
        # Thread safety
        self._lock = Lock()
        
        # Global start time
        self.start_time_us: int = 0
    
    def _init_default_parameters(self) -> None:
        """Initialize default Hawkes parameters for each order type."""
        # Market orders: higher excitation, faster decay
        market_params = HawkesParameters(mu=2.0, alpha=0.8, beta=3.0)
        
        # Limit orders: moderate excitation, slower decay
        limit_params = HawkesParameters(mu=5.0, alpha=0.5, beta=1.5)
        
        # Cancellations: high baseline, moderate excitation
        cancel_params = HawkesParameters(mu=8.0, alpha=0.6, beta=2.0)
        
        # Amendments: low baseline, low excitation
        amend_params = HawkesParameters(mu=1.0, alpha=0.3, beta=1.0)
        
        self.default_params: Dict[OrderType, HawkesParameters] = {
            OrderType.MARKET_BUY: market_params,
            OrderType.MARKET_SELL: market_params,
            OrderType.LIMIT_BUY: limit_params,
            OrderType.LIMIT_SELL: limit_params,
            OrderType.CANCEL_BUY: cancel_params,
            OrderType.CANCEL_SELL: cancel_params,
            OrderType.AMEND_BUY: amend_params,
            OrderType.AMEND_SELL: amend_params,
        }
    
    def add_event(self, event: OrderEvent) -> None:
        """
        Add a new event and update intensity calculations.
        
        Uses recursive formula for O(1) intensity update:
        λ(t) = μ + (λ(t-Δt) - μ) * exp(-β * Δt) + α
        
        Args:
            event: The order event to process
        """
        with self._lock:
            if self.start_time_us == 0:
                self.start_time_us = event.timestamp_us
            
            order_type = event.order_type
            
            # Initialize state if needed
            if order_type not in self.intensity_states:
                params = self.default_params.get(
                    order_type, HawkesParameters(1.0, 0.5, 2.0)
                )
                self.intensity_states[order_type] = IntensityState(
                    current_intensity=params.mu,
                    last_update_us=event.timestamp_us,
                    event_count=0,
                    total_volume=0.0,
                    params=params,
                )
            
            state = self.intensity_states[order_type]
            
            # Update intensity with time decay
            if state.last_update_us > 0:
                dt_us = event.timestamp_us - state.last_update_us
                dt_s = dt_us / 1_000_000.0
                
                # Exponential decay
                decay_factor = math.exp(-state.params.beta * dt_s)
                state.current_intensity = (
                    state.params.mu 
                    + (state.current_intensity - state.params.mu) * decay_factor
                )
                
                # Add excitation
                state.current_intensity += state.params.alpha
            
            # Update state
            state.last_update_us = event.timestamp_us
            state.event_count += 1
            state.total_volume += event.volume
            
            # Store event
            self.event_queues[order_type].append(event)
    
    def get_intensity(self, order_type: OrderType) -> float:
        """
        Get current intensity for specified order type in O(1) time.
        
        Args:
            order_type: The type of order to query
            
        Returns:
            Current intensity (events per second)
        """
        with self._lock:
            if order_type not in self.intensity_states:
                return 0.0
            return self.intensity_states[order_type].current_intensity
    
    def get_all_intensities(self) -> Dict[OrderType, float]:
        """Get intensities for all order types."""
        with self._lock:
            return {
                ot: state.current_intensity 
                for ot, state in self.intensity_states.items()
            }
    
    def get_intensity_at(self, order_type: OrderType, future_us: int) -> float:
        """
        Predict intensity at a future time.
        
        Args:
            order_type: The type of order
            future_us: Future timestamp in microseconds
            
        Returns:
            Predicted intensity at future time
        """
        with self._lock:
            if order_type not in self.intensity_states:
                return 0.0
            
            state = self.intensity_states[order_type]
            
            if future_us <= state.last_update_us:
                return state.current_intensity
            
            dt_us = future_us - state.last_update_us
            dt_s = dt_us / 1_000_000.0
            
            decay_factor = math.exp(-state.params.beta * dt_s)
            return state.params.mu + (state.current_intensity - state.params.mu) * decay_factor
    
    def get_expected_events(
        self, 
        order_type: OrderType, 
        horizon_s: float
    ) -> float:
        """
        Calculate expected number of events in next T seconds.
        
        E[N(T)] = μ*T + (λ(0) - μ) * (1 - exp(-β*T)) / β
        
        Args:
            order_type: The type of order
            horizon_s: Time horizon in seconds
            
        Returns:
            Expected number of events
        """
        with self._lock:
            if order_type not in self.intensity_states:
                return 0.0
            
            state = self.intensity_states[order_type]
            params = state.params
            
            lambda_0 = state.current_intensity
            mu = params.mu
            beta = params.beta
            
            return (
                mu * horizon_s 
                + (lambda_0 - mu) * (1.0 - math.exp(-beta * horizon_s)) / beta
            )
    
    def get_market_order_intensity(self) -> Tuple[float, float]:
        """
        Get combined market order intensity for buys and sells.
        
        Returns:
            Tuple of (buy_intensity, sell_intensity)
        """
        buy = self.get_intensity(OrderType.MARKET_BUY)
        sell = self.get_intensity(OrderType.MARKET_SELL)
        return buy, sell
    
    def get_limit_order_intensity(self) -> Tuple[float, float]:
        """
        Get combined limit order intensity for bids and asks.
        
        Returns:
            Tuple of (bid_intensity, ask_intensity)
        """
        bid = self.get_intensity(OrderType.LIMIT_BUY)
        ask = self.get_intensity(OrderType.LIMIT_SELL)
        return bid, ask
    
    def get_arrival_rate_ratio(self) -> float:
        """
        Calculate ratio of market to limit order arrivals.
        
        High ratio indicates aggressive trading (potential trend).
        Low ratio indicates passive liquidity provision.
        
        Returns:
            Ratio of market order intensity to limit order intensity
        """
        market_buy, market_sell = self.get_market_order_intensity()
        limit_bid, limit_ask = self.get_limit_order_intensity()
        
        market_total = market_buy + market_sell
        limit_total = limit_bid + limit_ask
        
        if limit_total == 0:
            return float('inf')
        
        return market_total / limit_total
    
    def get_volume_weighted_intensity(
        self, 
        order_type: OrderType
    ) -> float:
        """
        Calculate volume-weighted intensity.
        
        This gives more weight to large orders which have greater market impact.
        
        Args:
            order_type: The type of order
            
        Returns:
            Volume-weighted intensity
        """
        intensity = self.get_intensity(order_type)
        
        with self._lock:
            if order_type not in self.intensity_states:
                return 0.0
            
            state = self.intensity_states[order_type]
            if state.event_count == 0:
                return 0.0
            
            avg_volume = state.total_volume / state.event_count
        
        return intensity * avg_volume
    
    def reset(self) -> None:
        """Reset all intensity calculations."""
        with self._lock:
            for queue in self.event_queues.values():
                queue.clear()
            self.intensity_states.clear()
            self.start_time_us = 0


class AdaptiveBaselineEstimator:
    """
    Adaptively estimate baseline intensity (μ) using rolling windows.
    
    This allows the model to adjust to changing market conditions
    without manual recalibration.
    """
    
    def __init__(
        self,
        window_size_s: float = 300.0,  # 5 minute window
        min_events: int = 100,
    ):
        """
        Initialize the adaptive baseline estimator.
        
        Args:
            window_size_s: Size of rolling window in seconds
            min_events: Minimum events required for estimation
        """
        self.window_size_s = window_size_s
        self.min_events = min_events
        
        # Event timestamps per order type
        self.timestamps: Dict[OrderType, Deque[int]] = {
            ot: deque() for ot in OrderType
        }
        
        # Current baseline estimates
        self.baselines: Dict[OrderType, float] = {}
    
    def add_timestamp(self, order_type: OrderType, timestamp_us: int) -> None:
        """Add event timestamp and update baseline if enough data."""
        self.timestamps[order_type].append(timestamp_us)
        
        # Prune old timestamps
        cutoff_us = int((time.time() - self.window_size_s) * 1_000_000)
        while (
            self.timestamps[order_type] 
            and self.timestamps[order_type][0] < cutoff_us
        ):
            self.timestamps[order_type].popleft()
        
        # Update baseline if enough events
        if len(self.timestamps[order_type]) >= self.min_events:
            self._update_baseline(order_type)
    
    def _update_baseline(self, order_type: OrderType) -> None:
        """Update baseline estimate using maximum likelihood."""
        timestamps = list(self.timestamps[order_type])
        if len(timestamps) < 2:
            return
        
        # Simple MLE for Poisson process: μ = N / T
        duration_s = (timestamps[-1] - timestamps[0]) / 1_000_000.0
        if duration_s > 0:
            self.baselines[order_type] = len(timestamps) / duration_s
    
    def get_baseline(self, order_type: OrderType) -> Optional[float]:
        """Get current baseline estimate for order type."""
        return self.baselines.get(order_type)


if __name__ == "__main__":
    # Example usage
    calculator = ArrivalRateCalculator()
    
    # Simulate some events
    base_time = int(time.time() * 1_000_000)
    
    for i in range(10):
        event = OrderEvent(
            timestamp_us=base_time + i * 100_000,  # 100ms apart
            order_type=OrderType.MARKET_BUY,
            volume=1.0 + i * 0.1,
            price=50000.0 + i * 10,
            order_id=f"order_{i}",
        )
        calculator.add_event(event)
    
    print(f"Market Buy Intensity: {calculator.get_intensity(OrderType.MARKET_BUY):.4f} events/s")
    print(f"Expected events in 10s: {calculator.get_expected_events(OrderType.MARKET_BUY, 10.0):.2f}")
    print(f"Branching ratio: {calculator.default_params[OrderType.MARKET_BUY].branching_ratio:.4f}")
