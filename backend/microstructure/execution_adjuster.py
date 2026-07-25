#!/usr/bin/env python3
"""
Execution Adjuster - Dynamic Execution Based on Microstructure Signals

This module adjusts order execution parameters in real-time based on
detected spoofing, toxicity, and liquidity conditions. It can instantly
pull quotes when adverse conditions are detected.

Designed for the ZAID PERSONAL CRYPTO TRADING BOT with strict 8GB RAM constraints.
Implements sub-millisecond reaction to market microstructure changes.

Author: Opus 4.8
Stage: 21/100 - Advanced Market Microstructure
"""

from __future__ import annotations
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time


class ExecutionAction(Enum):
    """Possible execution adjustments."""
    CONTINUE = auto()          # No change needed
    WIDEN_SPREAD = auto()      # Increase spread
    REDUCE_SIZE = auto()       # Decrease order size
    PULL_QUOTES = auto()       # Cancel all quotes
    PAUSE_TRADING = auto()     # Temporarily halt trading
    AGGRESSIVE = auto()        # Tighten spread for priority


@dataclass(slots=True)
class ExecutionState:
    """Current execution state and parameters."""
    action: ExecutionAction = ExecutionAction.CONTINUE
    spread_adjustment_bps: float = 0.0
    size_multiplier: float = 1.0
    max_position_pct: float = 1.0
    pause_duration_ms: int = 0
    reason: str = ""
    timestamp_ns: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action': self.action.name,
            'spread_adjustment_bps': self.spread_adjustment_bps,
            'size_multiplier': self.size_multiplier,
            'max_position_pct': self.max_position_pct,
            'pause_duration_ms': self.pause_duration_ms,
            'reason': self.reason,
            'timestamp_ns': self.timestamp_ns,
        }


@dataclass(slots=True)
class MicrostructureSignals:
    """Collection of microstructure signals."""
    vpin: float = 0.0
    toxicity_score: float = 0.0
    spoofing_detected: bool = False
    fake_liquidity_ratio: float = 0.0
    order_imbalance: float = 0.0
    spread_anomaly: bool = False
    volume_anomaly: bool = False
    queue_decay_rate: float = 0.0
    
    def is_critical(self) -> bool:
        """Check if any signal is at critical level."""
        return (
            self.vpin > 0.7 or
            self.toxicity_score > 0.8 or
            self.spoofing_detected or
            self.fake_liquidity_ratio > 0.5
        )
    
    def is_warning(self) -> bool:
        """Check if any signal is at warning level."""
        return (
            self.vpin > 0.4 or
            self.toxicity_score > 0.5 or
            self.fake_liquidity_ratio > 0.3 or
            abs(self.order_imbalance) > 0.6
        )


class ExecutionAdjuster:
    """
    Real-time execution parameter adjuster.
    
    Monitors microstructure signals and adjusts:
    - Spread width
    - Order sizes
    - Quote placement
    - Trading pauses
    
    Reacts within microseconds to protect against adverse selection.
    """
    
    # Thresholds for actions
    SPOOFING_PULL_THRESHOLD: float = 0.8
    TOXICITY_PAUSE_THRESHOLD: float = 0.85
    VPIN_WIDEN_THRESHOLD: float = 0.5
    FAKE_LIQUIDITY_REDUCE_THRESHOLD: float = 0.4
    
    def __init__(
        self,
        symbol: str,
        base_spread_bps: float = 10.0,
        max_spread_bps: float = 50.0,
    ) -> None:
        """
        Initialize execution adjuster.
        
        Args:
            symbol: Trading pair symbol
            base_spread_bps: Normal spread in basis points
            max_spread_bps: Maximum allowed spread
        """
        self.symbol: str = symbol
        self.base_spread_bps: float = base_spread_bps
        self.max_spread_bps: float = max_spread_bps
        
        # Current state
        self._current_state: ExecutionState = ExecutionState()
        self._signals: MicrostructureSignals = MicrostructureSignals()
        
        # State history
        self._state_history: List[ExecutionState] = []
        self._max_history: int = 100
        
        # Callbacks for external systems
        self._on_pull_quotes: Optional[Callable[[], None]] = None
        self._on_pause: Optional[Callable[[int], None]] = None
        self._on_spread_change: Optional[Callable[[float], None]] = None
        
        # Timing
        self._last_adjustment_ns: int = 0
        self._adjustment_cooldown_ns: int = 1_000_000  # 1ms minimum between adjustments
        
        # Active flags
        self._quotes_pulled: bool = False
        self._trading_paused: bool = False
        self._pause_end_ns: int = 0
    
    @staticmethod
    def _now_ns() -> int:
        """Get current time in nanoseconds."""
        return time.time_ns()
    
    def update_signals(self, signals: MicrostructureSignals) -> ExecutionState:
        """
        Update signals and recalculate execution state.
        
        Args:
            signals: Current microstructure signals
        
        Returns:
            Updated execution state
        """
        now = self._now_ns()
        
        # Check cooldown
        if now - self._last_adjustment_ns < self._adjustment_cooldown_ns:
            return self._current_state
        
        self._signals = signals
        self._current_state = ExecutionState(timestamp_ns=now)
        
        # Evaluate conditions in priority order
        
        # 1. Critical: Spoofing detected - pull quotes immediately
        if signals.spoofing_detected:
            self._handle_spoofing(now)
        
        # 2. Critical: Extreme toxicity - pause trading
        elif signals.toxicity_score >= self.TOXICITY_PAUSE_THRESHOLD:
            self._handle_extreme_toxicity(now)
        
        # 3. High VPIN - widen spread significantly
        elif signals.vpin >= self.VPIN_WIDEN_THRESHOLD:
            self._handle_high_vpin(now)
        
        # 4. High fake liquidity ratio - reduce size
        elif signals.fake_liquidity_ratio >= self.FAKE_LIQUIDITY_REDUCE_THRESHOLD:
            self._handle_fake_liquidity(now)
        
        # 5. Warning level - moderate adjustments
        elif signals.is_warning():
            self._handle_warning(now)
        
        # 6. Normal conditions
        else:
            self._handle_normal(now)
        
        self._last_adjustment_ns = now
        self._state_history.append(self._current_state)
        
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)
        
        # Execute callbacks
        self._execute_callbacks()
        
        return self._current_state
    
    def _handle_spoofing(self, now: int) -> None:
        """Handle spoofing detection - pull all quotes."""
        self._current_state.action = ExecutionAction.PULL_QUOTES
        self._current_state.spread_adjustment_bps = self.max_spread_bps
        self._current_state.size_multiplier = 0.0
        self._current_state.reason = "Spoofing detected - pulling quotes"
        self._quotes_pulled = True
    
    def _handle_extreme_toxicity(self, now: int) -> None:
        """Handle extreme toxicity - pause trading."""
        self._current_state.action = ExecutionAction.PAUSE_TRADING
        self._current_state.spread_adjustment_bps = self.max_spread_bps
        self._current_state.size_multiplier = 0.0
        self._current_state.pause_duration_ms = 5000  # 5 second pause
        self._current_state.reason = f"Extreme toxicity ({self._signals.toxicity_score:.2f})"
        self._trading_paused = True
        self._pause_end_ns = now + 5_000_000_000  # 5 seconds in ns
    
    def _handle_high_vpin(self, now: int) -> None:
        """Handle high VPIN - widen spread."""
        vpin = self._signals.vpin
        # Scale spread adjustment with VPIN
        adjustment = self.base_spread_bps * (1.0 + vpin * 2.0)
        adjustment = min(adjustment, self.max_spread_bps)
        
        self._current_state.action = ExecutionAction.WIDEN_SPREAD
        self._current_state.spread_adjustment_bps = adjustment - self.base_spread_bps
        self._current_state.size_multiplier = max(0.5, 1.0 - vpin)
        self._current_state.reason = f"High VPIN ({vpin:.2f}) - widening spread"
    
    def _handle_fake_liquidity(self, now: int) -> None:
        """Handle high fake liquidity - reduce order size."""
        ratio = self._signals.fake_liquidity_ratio
        
        self._current_state.action = ExecutionAction.REDUCE_SIZE
        self._current_state.size_multiplier = max(0.3, 1.0 - ratio)
        self._current_state.spread_adjustment_bps = self.base_spread_bps * 0.5
        self._current_state.reason = f"High fake liquidity ({ratio:.2f}) - reducing size"
    
    def _handle_warning(self, now: int) -> None:
        """Handle warning level signals - cautious adjustments."""
        self._current_state.action = ExecutionAction.WIDEN_SPREAD
        self._current_state.spread_adjustment_bps = self.base_spread_bps * 0.3
        self._current_state.size_multiplier = 0.8
        self._current_state.reason = "Warning signals - cautious execution"
    
    def _handle_normal(self, now: int) -> None:
        """Handle normal conditions - standard execution."""
        self._current_state.action = ExecutionAction.CONTINUE
        self._current_state.spread_adjustment_bps = 0.0
        self._current_state.size_multiplier = 1.0
        
        # Check if we're recovering from a pause
        if self._trading_paused and now >= self._pause_end_ns:
            self._trading_paused = False
            self._current_state.reason = "Resuming normal trading"
        elif self._quotes_pulled:
            self._quotes_pulled = False
            self._current_state.reason = "Re-establishing quotes"
        else:
            self._current_state.reason = "Normal conditions"
    
    def _execute_callbacks(self) -> None:
        """Execute registered callbacks for state changes."""
        state = self._current_state
        
        if state.action == ExecutionAction.PULL_QUOTES and self._on_pull_quotes:
            self._on_pull_quotes()
        
        if state.action == ExecutionAction.PAUSE_TRADING and self._on_pause:
            self._on_pause(state.pause_duration_ms)
        
        if state.action == ExecutionAction.WIDEN_SPREAD and self._on_spread_change:
            self._on_spread_change(state.spread_adjustment_bps)
    
    def get_effective_spread_bps(self) -> float:
        """Get the effective spread including adjustments."""
        return self.base_spread_bps + self._current_state.spread_adjustment_bps
    
    def get_effective_size_multiplier(self) -> float:
        """Get the effective size multiplier."""
        return self._current_state.size_multiplier
    
    def should_place_quotes(self) -> bool:
        """Check if quotes should be placed."""
        if self._quotes_pulled:
            return False
        if self._trading_paused and self._now_ns() < self._pause_end_ns:
            return False
        return self._current_state.action not in (
            ExecutionAction.PULL_QUOTES,
            ExecutionAction.PAUSE_TRADING,
        )
    
    def should_cancel_order(self, order_age_ms: float) -> bool:
        """
        Check if an existing order should be cancelled.
        
        Args:
            order_age_ms: Age of the order in milliseconds
        
        Returns:
            True if order should be cancelled
        """
        # Always cancel if quotes pulled
        if self._quotes_pulled:
            return True
        
        # Cancel old orders during high toxicity
        if self._signals.toxicity_score > 0.6 and order_age_ms > 100:
            return True
        
        # Cancel if spoofing detected after order placement
        if self._signals.spoofing_detected:
            return True
        
        return False
    
    def register_callbacks(
        self,
        on_pull_quotes: Optional[Callable[[], None]] = None,
        on_pause: Optional[Callable[[int], None]] = None,
        on_spread_change: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Register callback functions for state changes."""
        self._on_pull_quotes = on_pull_quotes
        self._on_pause = on_pause
        self._on_spread_change = on_spread_change
    
    def get_current_state(self) -> ExecutionState:
        """Get current execution state."""
        return self._current_state
    
    def get_state_history(self, limit: int = 10) -> List[ExecutionState]:
        """Get recent state history."""
        return self._state_history[-limit:]
    
    def is_trading_active(self) -> bool:
        """Check if trading is currently active."""
        if self._trading_paused and self._now_ns() < self._pause_end_ns:
            return False
        return not self._quotes_pulled
    
    def manual_pull_quotes(self, reason: str = "Manual override") -> None:
        """Manually trigger quote pull."""
        now = self._now_ns()
        self._current_state = ExecutionState(
            action=ExecutionAction.PULL_QUOTES,
            spread_adjustment_bps=self.max_spread_bps,
            size_multiplier=0.0,
            reason=reason,
            timestamp_ns=now,
        )
        self._quotes_pulled = True
        self._execute_callbacks()
    
    def manual_resume(self) -> None:
        """Manually resume trading after pause."""
        self._trading_paused = False
        self._quotes_pulled = False
        self._pause_end_ns = 0
        self._handle_normal(self._now_ns())
    
    def reset(self) -> None:
        """Reset all state."""
        self._current_state = ExecutionState()
        self._signals = MicrostructureSignals()
        self._state_history.clear()
        self._quotes_pulled = False
        self._trading_paused = False
        self._pause_end_ns = 0


def main() -> None:
    """Example usage of ExecutionAdjuster."""
    adjuster = ExecutionAdjuster(symbol='BTCUSDT', base_spread_bps=10.0)
    
    # Simulate various market conditions
    scenarios = [
        # Normal
        MicrostructureSignals(vpin=0.2, toxicity_score=0.1, fake_liquidity_ratio=0.1),
        # Warning
        MicrostructureSignals(vpin=0.45, toxicity_score=0.4, fake_liquidity_ratio=0.25),
        # High VPIN
        MicrostructureSignals(vpin=0.6, toxicity_score=0.5, fake_liquidity_ratio=0.2),
        # Spoofing detected!
        MicrostructureSignals(vpin=0.3, toxicity_score=0.2, spoofing_detected=True, fake_liquidity_ratio=0.1),
        # Extreme toxicity
        MicrostructureSignals(vpin=0.8, toxicity_score=0.9, fake_liquidity_ratio=0.3),
    ]
    
    print("=== Execution Adjuster Simulation ===\n")
    
    for i, signals in enumerate(scenarios):
        state = adjuster.update_signals(signals)
        print(f"Scenario {i+1}:")
        print(f"  VPIN: {signals.vpin:.2f}, Toxicity: {signals.toxicity_score:.2f}")
        print(f"  Spoofing: {signals.spoofing_detected}, Fake Liq: {signals.fake_liquidity_ratio:.2f}")
        print(f"  Action: {state.action.name}")
        print(f"  Spread Adjustment: +{state.spread_adjustment_bps:.1f} bps")
        print(f"  Size Multiplier: {state.size_multiplier:.2f}x")
        print(f"  Reason: {state.reason}")
        print(f"  Quotes Active: {adjuster.should_place_quotes()}")
        print()


if __name__ == "__main__":
    main()
