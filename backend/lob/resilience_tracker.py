#!/usr/bin/env python3
"""
Resilience Tracker for Order Book Liquidity Replenishment

This module measures the speed of liquidity replenishment after large sweeps,
detecting potential flash crash conditions when replenishment stalls.

Key Features:
- Real-time resilience scoring
- Flash crash early warning detection
- Asset-specific baseline calibration
- Memory-efficient sliding window calculations

Author: ZAID Personal Crypto Trading Bot
Stage: 22 - LOB Physics & Hawkes Processes
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
from threading import Lock


class ResilienceState(Enum):
    """Order book resilience classification."""
    HIGHLY_RESILIENT = "highly_resilient"
    RESILIENT = "resilient"
    WEAK = "weak"
    FRAGILE = "fragile"
    CRITICAL = "critical"  # Potential flash crash condition


@dataclass
class LiquidityEvent:
    """Record of a liquidity depletion event."""
    timestamp: float
    side: str  # 'bid' or 'ask'
    depleted_volume: float
    price_level: float
    depletion_pct: float  # Percentage of depth removed
    replenishment_start: float = 0.0
    full_replenishment_time: float = 0.0
    is_fully_replenished: bool = False


@dataclass
class ResilienceMetrics:
    """Current resilience metrics snapshot."""
    bid_resilience_score: float  # 0-1 scale
    ask_resilience_score: float
    combined_resilience: float
    state: ResilienceState
    avg_replenishment_time_ms: float
    recent_depletion_events: int
    flash_crash_risk: float  # 0-1 probability
    last_update: float


@dataclass
class AssetResilienceParams:
    """Asset-specific resilience parameters."""
    normal_replenishment_time_ms: float  # Expected replenishment time
    flash_crash_threshold_ms: float  # Time beyond which indicates stress
    min_liquidity_depth: float  # Minimum expected depth
    
    @classmethod
    def default_params(cls, asset: str) -> 'AssetResilienceParams':
        """Get default parameters for common assets."""
        params = {
            "BTC": cls(
                normal_replenishment_time_ms=500,  # 500ms typical
                flash_crash_threshold_ms=5000,  # 5s indicates severe stress
                min_liquidity_depth=100.0,  # 100 BTC at top levels
            ),
            "ETH": cls(
                normal_replenishment_time_ms=400,
                flash_crash_threshold_ms=4000,
                min_liquidity_depth=500.0,
            ),
            "SOL": cls(
                normal_replenishment_time_ms=300,
                flash_crash_threshold_ms=3000,
                min_liquidity_depth=5000.0,
            ),
            "DEFAULT": cls(
                normal_replenishment_time_ms=500,
                flash_crash_threshold_ms=5000,
                min_liquidity_depth=100.0,
            ),
        }
        return params.get(asset.upper(), params["DEFAULT"])


class ResilienceTracker:
    """
    Tracks order book resilience and liquidity replenishment dynamics.
    
    Monitors how quickly the order book recovers after large trades,
    providing early warning signals for potential flash crashes.
    """
    
    def __init__(
        self,
        asset: str = "BTC",
        window_size: int = 1000,
        alert_threshold: float = 0.7,
    ):
        """
        Initialize resilience tracker.
        
        Args:
            asset: Asset symbol for parameter calibration
            window_size: Number of events to track in sliding window
            alert_threshold: Resilience score below which to trigger alerts
        """
        self.asset = asset.upper()
        self.params = AssetResilienceParams.default_params(self.asset)
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Event tracking
        self._depletion_events: Deque[LiquidityEvent] = deque(maxlen=window_size)
        self._replenishment_times: Deque[float] = deque(maxlen=window_size)
        
        # Current state tracking
        self._current_bid_depth: float = 0.0
        self._current_ask_depth: float = 0.0
        self._baseline_bid_depth: float = 0.0
        self._baseline_ask_depth: float = 0.0
        
        # Timing
        self._last_bid_depletion: float = 0.0
        self._last_ask_depletion: float = 0.0
        self._last_update: float = time.time()
        
        # Thread safety
        self._lock = Lock()
        
        # Pre-compute constants
        self._decay_factor = 0.99  # For exponential weighting
    
    def set_baseline_depth(self, bid_depth: float, ask_depth: float) -> None:
        """Set baseline depth levels for comparison."""
        with self._lock:
            self._baseline_bid_depth = bid_depth
            self._baseline_ask_depth = ask_depth
            self._current_bid_depth = bid_depth
            self._current_ask_depth = ask_depth
    
    def record_liquidity_sweep(
        self,
        side: str,
        depleted_volume: float,
        price_level: float,
        current_depth: float,
    ) -> Optional[LiquidityEvent]:
        """
        Record a liquidity sweep/depletion event.
        
        Args:
            side: 'bid' or 'ask'
            depleted_volume: Volume removed from the book
            price_level: Price at which sweep occurred
            current_depth: Remaining depth after sweep
            
        Returns:
            LiquidityEvent if this qualifies as a significant depletion
        """
        with self._lock:
            now = time.time()
            
            # Determine baseline for this side
            baseline = (self._baseline_bid_depth if side.lower() == 'bid' 
                       else self._baseline_ask_depth)
            
            if baseline <= 0:
                return None
            
            # Calculate depletion percentage
            depletion_pct = depleted_volume / baseline
            
            # Only track significant depletions (>5% of depth)
            if depletion_pct < 0.05:
                return None
            
            event = LiquidityEvent(
                timestamp=now,
                side=side.lower(),
                depleted_volume=depleted_volume,
                price_level=price_level,
                depletion_pct=depletion_pct,
            )
            
            self._depletion_events.append(event)
            
            # Update last depletion time
            if side.lower() == 'bid':
                self._last_bid_depletion = now
            else:
                self._last_ask_depletion = now
            
            # Update current depth
            if side.lower() == 'bid':
                self._current_bid_depth = current_depth
            else:
                self._current_ask_depth = current_depth
            
            return event
    
    def update_current_depth(self, bid_depth: float, ask_depth: float) -> None:
        """Update current depth measurements."""
        with self._lock:
            now = time.time()
            
            # Check for replenishment completion
            for event in list(self._depletion_events):
                if event.is_fully_replenished:
                    continue
                
                baseline = (self._baseline_bid_depth if event.side == 'bid' 
                           else self._baseline_ask_depth)
                current = (bid_depth if event.side == 'bid' else ask_depth)
                
                # Check if depth has recovered to >90% of baseline
                if current >= 0.9 * baseline:
                    event.is_fully_replenished = True
                    event.full_replenishment_time = (now - event.timestamp) * 1000  # ms
                    
                    if not event.replenishment_start:
                        event.replenishment_start = now
                    
                    self._replenishment_times.append(event.full_replenishment_time)
            
            self._current_bid_depth = bid_depth
            self._current_ask_depth = ask_depth
            self._last_update = now
    
    def calculate_resilience_score(self) -> ResilienceMetrics:
        """
        Calculate current resilience metrics.
        
        Returns:
            ResilienceMetrics with current state assessment
        """
        with self._lock:
            now = time.time()
            
            # Calculate average replenishment time
            if self._replenishment_times:
                # Exponentially weighted average
                weights = np.array([self._decay_factor ** i 
                                   for i in range(len(self._replenishment_times))])
                weights /= weights.sum()
                avg_replenishment = np.dot(
                    list(self._replenishment_times), weights
                )
            else:
                avg_replenishment = self.params.normal_replenishment_time_ms
            
            # Calculate time since last depletion
            time_since_bid = now - self._last_bid_depletion if self._last_bid_depletion else float('inf')
            time_since_ask = now - self._last_ask_depletion if self._last_ask_depletion else float('inf')
            
            # Calculate depth recovery ratios
            bid_recovery = (self._current_bid_depth / self._baseline_bid_depth 
                          if self._baseline_bid_depth > 0 else 1.0)
            ask_recovery = (self._current_ask_depth / self._baseline_ask_depth 
                          if self._baseline_ask_depth > 0 else 1.0)
            
            # Resilience scores (higher is better)
            # Based on: recovery ratio, replenishment speed, time since last event
            bid_resilience = self._calculate_single_resilience(
                recovery=bid_recovery,
                time_since_event=time_since_bid,
                avg_replenishment=avg_replenishment,
            )
            ask_resilience = self._calculate_single_resilience(
                recovery=ask_recovery,
                time_since_event=time_since_ask,
                avg_replenishment=avg_replenishment,
            )
            
            combined = (bid_resilience + ask_resilience) / 2
            
            # Determine state
            state = self._classify_state(combined, avg_replenishment)
            
            # Flash crash risk assessment
            flash_crash_risk = self._calculate_flash_crash_risk(
                avg_replenishment, bid_recovery, ask_recovery
            )
            
            return ResilienceMetrics(
                bid_resilience_score=bid_resilience,
                ask_resilience_score=ask_resilience,
                combined_resilience=combined,
                state=state,
                avg_replenishment_time_ms=avg_replenishment,
                recent_depletion_events=len(self._depletion_events),
                flash_crash_risk=flash_crash_risk,
                last_update=now,
            )
    
    def _calculate_single_resilience(
        self,
        recovery: float,
        time_since_event: float,
        avg_replenishment: float,
    ) -> float:
        """Calculate resilience score for one side."""
        # Recovery component (0-0.5)
        recovery_score = min(0.5, recovery * 0.5)
        
        # Speed component (0-0.3)
        if avg_replenishment <= self.params.normal_replenishment_time_ms:
            speed_score = 0.3
        elif avg_replenishment >= self.params.flash_crash_threshold_ms:
            speed_score = 0.0
        else:
            ratio = avg_replenishment / self.params.flash_crash_threshold_ms
            speed_score = 0.3 * (1 - ratio)
        
        # Stability component (0-0.2) - time since last event
        if time_since_event >= 10.0:  # 10 seconds
            stability_score = 0.2
        else:
            stability_score = 0.2 * (time_since_event / 10.0)
        
        return min(1.0, recovery_score + speed_score + stability_score)
    
    def _classify_state(
        self,
        combined_score: float,
        avg_replenishment: float,
    ) -> ResilienceState:
        """Classify resilience state based on score and timing."""
        if combined_score >= 0.8:
            return ResilienceState.HIGHLY_RESILIENT
        elif combined_score >= 0.6:
            return ResilienceState.RESILIENT
        elif combined_score >= 0.4:
            return ResilienceState.WEAK
        elif combined_score >= 0.2:
            return ResilienceState.FRAGILE
        else:
            return ResilienceState.CRITICAL
    
    def _calculate_flash_crash_risk(
        self,
        avg_replenishment: float,
        bid_recovery: float,
        ask_recovery: float,
    ) -> float:
        """Calculate probability of flash crash condition."""
        risk = 0.0
        
        # Replenishment time risk
        if avg_replenishment > self.params.flash_crash_threshold_ms:
            risk += 0.4
        elif avg_replenishment > self.params.normal_replenishment_time_ms * 3:
            risk += 0.2
        
        # Depth recovery risk
        if bid_recovery < 0.5 or ask_recovery < 0.5:
            risk += 0.3
        
        # Recent event frequency risk
        recent_events = sum(
            1 for e in self._depletion_events 
            if time.time() - e.timestamp < 1.0  # Last second
        )
        if recent_events >= 5:
            risk += 0.3
        
        return min(1.0, risk)
    
    def is_flash_crash_warning(self) -> bool:
        """Check if current conditions indicate flash crash risk."""
        metrics = self.calculate_resilience_score()
        return (metrics.state == ResilienceState.CRITICAL or 
                metrics.flash_crash_risk > 0.5)
    
    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._depletion_events.clear()
            self._replenishment_times.clear()
            self._current_bid_depth = self._baseline_bid_depth
            self._current_ask_depth = self._baseline_ask_depth
            self._last_bid_depletion = 0.0
            self._last_ask_depletion = 0.0
    
    def update_asset(self, asset: str) -> None:
        """Update asset class for recalibration."""
        with self._lock:
            self.asset = asset.upper()
            self.params = AssetResilienceParams.default_params(self.asset)


if __name__ == "__main__":
    # Demo usage
    tracker = ResilienceTracker(asset="BTC")
    tracker.set_baseline_depth(bid_depth=100.0, ask_depth=100.0)
    
    # Simulate a liquidity sweep
    event = tracker.record_liquidity_sweep(
        side="bid",
        depleted_volume=30.0,  # 30% of depth
        price_level=49500.0,
        current_depth=70.0,
    )
    
    print(f"Sweep recorded: {event.depletion_pct:.1%} depletion")
    
    # Update with recovering depth
    tracker.update_current_depth(bid_depth=85.0, ask_depth=100.0)
    
    # Get resilience metrics
    metrics = tracker.calculate_resilience_score()
    print(f"Bid Resilience: {metrics.bid_resilience_score:.2f}")
    print(f"Ask Resilience: {metrics.ask_resilience_score:.2f}")
    print(f"State: {metrics.state.value}")
    print(f"Flash Crash Risk: {metrics.flash_crash_risk:.2%}")
