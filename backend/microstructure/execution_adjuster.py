#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
File: backend/microstructure/execution_adjuster.py
Chapter 4: Microstructure Analytics, Execution Adjustments, and SOUL.md Logging

Purpose: Pull quotes instantly if spoofing is detected; adjust execution based on microstructure
Constraints: Must react within microseconds to spoofing alerts
Target: AMD Ryzen AI 5 laptop with 8GB RAM limit

This module provides real-time execution adjustments based on:
1. Spoofing detection alerts
2. VPIN toxicity levels
3. Order book imbalance signals
4. Queue position analysis

Design Patterns: Strategy Pattern for execution adjustments, Observer for alerts
Type Hinting: Strict typing for production reliability
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from enum import Enum
from collections import deque
import time


class ExecutionAction(Enum):
    """Actions the adjuster can recommend"""
    HOLD = "hold"                    # No change needed
    PULL_QUOTES = "pull_quotes"      # Immediately cancel resting orders
    WIDEN_SPREAD = "widen_spread"    # Increase bid-ask spread
    REDUCE_SIZE = "reduce_size"      # Decrease order sizes
    AGGRESSIVE_FILL = "aggressive"   # Use market orders instead of limit
    PAUSE_TRADING = "pause"          # Stop all trading activity
    SWITCH_VENUE = "switch_venue"    # Move to different exchange


@dataclass(slots=True)
class MicrostructureSignal:
    """Combined signal from microstructure analysis"""
    timestamp_ns: int
    spoofing_detected: bool
    vpin_level: float  # 0.0 to 1.0
    imbalance_signal: float  # -1.0 to 1.0
    toxicity_score: float  # 0.0 to 1.0
    queue_position_risk: float  # 0.0 to 1.0


@dataclass(slots=True)
class ExecutionAdjustment:
    """Recommended execution adjustment"""
    action: ExecutionAction
    confidence: float  # 0.0 to 1.0
    reason: str
    suggested_spread_bps: Optional[float] = None
    suggested_size_multiplier: Optional[float] = None
    duration_ms: Optional[int] = None  # How long to maintain adjustment
    timestamp_ns: int = 0


class ExecutionAdjuster:
    """
    Real-time execution adjustment engine.
    
    Monitors microstructure signals and recommends immediate actions
    to protect against adverse selection and manipulation.
    """
    
    # Thresholds for action triggers
    SPOOFING_PULL_THRESHOLD: float = 0.7  # Confidence to pull quotes
    VPIN_PAUSE_THRESHOLD: float = 0.8     # VPIN level to pause trading
    TOXICITY_WIDEN_THRESHOLD: float = 0.5 # Toxicity to widen spreads
    
    def __init__(self):
        # Recent signals
        self.signal_history: deque = deque(maxlen=1000)
        
        # Active adjustments
        self.active_adjustments: List[ExecutionAdjustment] = []
        
        # Alert callbacks (Observer Pattern)
        self.alert_callbacks: List[Callable[[ExecutionAdjustment], None]] = []
        
        # Timing
        self.last_update_ns: int = 0
        self.start_time_ns: int = time.time_ns()
        
        # Configuration
        self.enabled: bool = True
        self.auto_execute: bool = False  # If True, execute adjustments automatically
        
        # State tracking
        self.current_spread_multiplier: float = 1.0
        self.current_size_multiplier: float = 1.0
        self.trading_paused: bool = False
        
        # Statistics
        self.total_adjustments: int = 0
        self.pull_quote_count: int = 0
        self.pause_count: int = 0
    
    def register_callback(self, callback: Callable[[ExecutionAdjustment], None]) -> None:
        """Register a callback for when adjustments are recommended"""
        self.alert_callbacks.append(callback)
    
    def process_signal(self, signal: MicrostructureSignal) -> Optional[ExecutionAdjustment]:
        """
        Process a microstructure signal and determine if adjustment is needed.
        
        This is the main entry point called whenever new microstructure data arrives.
        """
        if not self.enabled:
            return None
        
        self.last_update_ns = signal.timestamp_ns
        self.signal_history.append(signal)
        
        # Evaluate conditions in priority order
        adjustment = self._evaluate_conditions(signal)
        
        if adjustment:
            self._apply_adjustment(adjustment)
            self._notify_callbacks(adjustment)
        
        return adjustment
    
    def _evaluate_conditions(self, signal: MicrostructureSignal) -> Optional[ExecutionAdjustment]:
        """Evaluate all conditions and return highest-priority adjustment"""
        
        # Priority 1: Spoofing detected - pull quotes immediately
        if signal.spoofing_detected:
            return ExecutionAdjustment(
                action=ExecutionAction.PULL_QUOTES,
                confidence=0.95,
                reason="Spoofing detected - protecting against fake liquidity",
                suggested_spread_bps=None,
                suggested_size_multiplier=0.0,
                duration_ms=500,  # Pull for 500ms
                timestamp_ns=self.last_update_ns
            )
        
        # Priority 2: Extreme toxicity - pause trading
        if signal.toxicity_score >= 0.8 or signal.vpin_level >= self.VPIN_PAUSE_THRESHOLD:
            return ExecutionAdjustment(
                action=ExecutionAction.PAUSE_TRADING,
                confidence=min(signal.toxicity_score, signal.vpin_level),
                reason=f"Extreme toxicity (VPIN={signal.vpin_level:.2f}, Toxicity={signal.toxicity_score:.2f})",
                suggested_spread_bps=None,
                suggested_size_multiplier=0.0,
                duration_ms=5000,  # Pause for 5 seconds
                timestamp_ns=self.last_update_ns
            )
        
        # Priority 3: High toxicity - widen spreads significantly
        if signal.toxicity_score >= self.TOXICITY_WIDEN_THRESHOLD:
            spread_adjustment = 1.0 + signal.toxicity_score * 2.0  # Up to 3x spread
            return ExecutionAdjustment(
                action=ExecutionAction.WIDEN_SPREAD,
                confidence=signal.toxicity_score,
                reason=f"High toxicity requiring wider spreads",
                suggested_spread_bps=spread_adjustment * 10,  # Convert to bps
                suggested_size_multiplier=0.5,
                duration_ms=2000,
                timestamp_ns=self.last_update_ns
            )
        
        # Priority 4: Moderate VPIN - reduce size
        if signal.vpin_level >= 0.5:
            return ExecutionAdjustment(
                action=ExecutionAction.REDUCE_SIZE,
                confidence=signal.vpin_level,
                reason=f"Elevated VPIN ({signal.vpin_level:.2f}) - reducing exposure",
                suggested_spread_bps=None,
                suggested_size_multiplier=0.6,
                duration_ms=3000,
                timestamp_ns=self.last_update_ns
            )
        
        # Priority 5: Strong imbalance - consider aggressive fill
        if abs(signal.imbalance_signal) >= 0.7:
            direction = "buy" if signal.imbalance_signal > 0 else "sell"
            return ExecutionAdjustment(
                action=ExecutionAction.AGGRESSIVE_FILL,
                confidence=abs(signal.imbalance_signal),
                reason=f"Strong {direction} imbalance - opportunity for immediate fill",
                suggested_spread_bps=None,
                suggested_size_multiplier=1.2,  # Can increase size for favorable conditions
                duration_ms=1000,
                timestamp_ns=self.last_update_ns
            )
        
        # No adjustment needed
        return None
    
    def _apply_adjustment(self, adjustment: ExecutionAdjustment) -> None:
        """Apply the adjustment to current state"""
        self.total_adjustments += 1
        
        if adjustment.action == ExecutionAction.PULL_QUOTES:
            self.pull_quote_count += 1
            self.current_size_multiplier = 0.0
        elif adjustment.action == ExecutionAction.PAUSE_TRADING:
            self.pause_count += 1
            self.trading_paused = True
        elif adjustment.action == ExecutionAction.WIDEN_SPREAD:
            if adjustment.suggested_spread_bps:
                self.current_spread_multiplier = adjustment.suggested_spread_bps / 10.0
        elif adjustment.action == ExecutionAction.REDUCE_SIZE:
            if adjustment.suggested_size_multiplier:
                self.current_size_multiplier = adjustment.suggested_size_multiplier
        
        self.active_adjustments.append(adjustment)
        
        # Clean up old adjustments
        cutoff_ns = self.last_update_ns - 10_000_000_000  # 10 seconds ago
        self.active_adjustments = [a for a in self.active_adjustments if a.timestamp_ns > cutoff_ns]
    
    def _notify_callbacks(self, adjustment: ExecutionAdjustment) -> None:
        """Notify all registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(adjustment)
            except Exception as e:
                print(f"Error in adjustment callback: {e}")
    
    def get_current_state(self) -> ExecutionState:
        """Get current execution state"""
        return ExecutionState(
            spread_multiplier=self.current_spread_multiplier,
            size_multiplier=self.current_size_multiplier,
            trading_paused=self.trading_paused,
            active_adjustment_count=len(self.active_adjustments),
            last_signal_time_ns=self.last_update_ns
        )
    
    def reset_state(self) -> None:
        """Reset all multipliers to normal"""
        self.current_spread_multiplier = 1.0
        self.current_size_multiplier = 1.0
        self.trading_paused = False
        self.active_adjustments.clear()
    
    def should_cancel_order(self, order_age_ms: float) -> bool:
        """
        Determine if a resting order should be cancelled based on current state.
        
        Returns True if order should be pulled.
        """
        if self.trading_paused:
            return True
        
        if self.current_size_multiplier <= 0.0:
            return True
        
        # Check recent signals for spoofing
        recent_spoofing = any(
            s.spoofing_detected 
            for s in list(self.signal_history)[-10:]
        )
        
        if recent_spoofing:
            return True
        
        return False
    
    def get_adjusted_price(self, base_price: float, side: str) -> float:
        """
        Get price adjusted for current spread multiplier.
        
        For bids: lower the price (more conservative)
        For asks: raise the price (more conservative)
        """
        if self.current_spread_multiplier <= 1.0:
            return base_price
        
        # Calculate adjustment based on spread multiplier
        adjustment_factor = (self.current_spread_multiplier - 1.0) / 2.0
        
        if side.lower() == "bid":
            return base_price * (1.0 - adjustment_factor * 0.0001)
        else:  # ask
            return base_price * (1.0 + adjustment_factor * 0.0001)
    
    def get_statistics(self) -> dict:
        """Get execution adjuster statistics"""
        return {
            'total_adjustments': self.total_adjustments,
            'pull_quote_count': self.pull_quote_count,
            'pause_count': self.pause_count,
            'current_spread_multiplier': self.current_spread_multiplier,
            'current_size_multiplier': self.current_size_multiplier,
            'trading_paused': self.trading_paused,
            'signals_processed': len(self.signal_history),
        }


@dataclass
class ExecutionState:
    """Current state of execution adjustments"""
    spread_multiplier: float
    size_multiplier: float
    trading_paused: bool
    active_adjustment_count: int
    last_signal_time_ns: int


def main():
    """Example usage of execution adjuster"""
    adjuster = ExecutionAdjuster()
    
    # Simulate a spoofing alert
    signal = MicrostructureSignal(
        timestamp_ns=time.time_ns(),
        spoofing_detected=True,
        vpin_level=0.4,
        imbalance_signal=0.2,
        toxicity_score=0.3,
        queue_position_risk=0.1
    )
    
    adjustment = adjuster.process_signal(signal)
    
    if adjustment:
        print(f"🚨 EXECUTION ADJUSTMENT RECOMMENDED")
        print(f"Action: {adjustment.action.value}")
        print(f"Confidence: {adjustment.confidence:.2%}")
        print(f"Reason: {adjustment.reason}")
        if adjustment.suggested_spread_bps:
            print(f"Suggested Spread: {adjustment.suggested_spread_bps:.1f} bps")
        if adjustment.suggested_size_multiplier:
            print(f"Size Multiplier: {adjustment.suggested_size_multiplier:.2f}")
    
    # Show current state
    state = adjuster.get_current_state()
    print(f"\nCurrent State:")
    print(f"  Spread Multiplier: {state.spread_multiplier:.2f}x")
    print(f"  Size Multiplier: {state.size_multiplier:.2f}x")
    print(f"  Trading Paused: {state.trading_paused}")
    
    # Statistics
    stats = adjuster.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    main()
