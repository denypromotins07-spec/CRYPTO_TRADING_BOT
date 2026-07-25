#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
Execution Adjuster Module
Pulls quotes instantly if spoofing is detected
Optimized for 8GB RAM with strict type hinting
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import time


class ExecutionAction(Enum):
    """Actions the execution adjuster can take."""
    HOLD = "hold"
    PULL_QUOTES = "pull_quotes"
    WIDEN_SPREAD = "widen_spread"
    REDUCE_SIZE = "reduce_size"
    HALT_TRADING = "halt_trading"
    ADJUST_PRICE = "adjust_price"


@dataclass(slots=True)
class SpoofingEvent:
    """Represents a detected spoofing event."""
    timestamp_ns: int
    event_type: str
    confidence: float
    price_level: float
    volume: float
    side: str


@dataclass(slots=True)
class ExecutionAdjustment:
    """Result of execution adjustment decision."""
    timestamp_ns: int
    action: ExecutionAction
    reason: str
    original_bid: Optional[float]
    original_ask: Optional[float]
    adjusted_bid: Optional[float]
    adjusted_ask: Optional[float]
    size_multiplier: float
    halt_duration_ms: int


class ExecutionAdjuster:
    """
    Adjusts execution parameters in real-time based on microstructure signals.
    
    Key triggers:
    - Spoofing detection -> Pull quotes immediately
    - High VPIN -> Widen spread
    - Toxicity > threshold -> Reduce size or halt
    - Layering detected -> Adjust pricing
    
    Implements Strategy pattern for different adjustment strategies.
    """
    
    # Response time thresholds (microseconds)
    MAX_RESPONSE_TIME_US = 100  # Must respond within 100us
    
    # Default adjustment parameters
    DEFAULT_SPREAD_WIDENING_BPS = 5.0
    DEFAULT_SIZE_REDUCTION_FACTOR = 0.5
    DEFAULT_HALT_DURATION_MS = 5000
    
    def __init__(self):
        """Initialize execution adjuster."""
        # Current state
        self.active_quotes: Dict[str, Tuple[float, float, float]] = {}  # symbol: (bid, ask, size)
        self.current_spread_bps: float = 0.0
        self.current_size_multiplier: float = 1.0
        self.is_trading_halted: bool = False
        self.halt_until_ns: int = 0
        
        # Recent spoofing events
        self.spoofing_events: List[SpoofingEvent] = []
        self.max_spoofing_history: int = 100
        
        # Adjustment strategy
        self.adjustment_strategy: AdjustmentStrategy = AggressiveAdjustmentStrategy()
        
        # Statistics
        self.stats = {
            'total_adjustments': 0,
            'quotes_pulled': 0,
            'spread_widened': 0,
            'trades_halted': 0,
            'adverse_selection_prevented': 0,
            'avg_response_time_us': 0.0,
        }
        
        # Callbacks for quote management
        self._quote_pull_callback: Optional[Callable] = None
        self._quote_update_callback: Optional[Callable] = None
        self._halt_callback: Optional[Callable] = None
        
        # Last adjustment timestamp
        self.last_adjustment_ns: int = time.time_ns()
    
    def set_adjustment_strategy(self, strategy: 'AdjustmentStrategy') -> None:
        """Set the adjustment strategy (Strategy pattern)."""
        self.adjustment_strategy = strategy
    
    def register_callbacks(
        self,
        pull_callback: Optional[Callable] = None,
        update_callback: Optional[Callable] = None,
        halt_callback: Optional[Callable] = None
    ) -> None:
        """Register callbacks for execution actions."""
        self._quote_pull_callback = pull_callback
        self._quote_update_callback = update_callback
        self._halt_callback = halt_callback
    
    def on_spoofing_detected(self, event: SpoofingEvent) -> ExecutionAdjustment:
        """
        Handle spoofing detection - must respond within microseconds.
        
        Args:
            event: SpoofingEvent that was detected
            
        Returns:
            ExecutionAdjustment with recommended action
        """
        start_ns = time.time_ns()
        
        # Record spoofing event
        self.spoofing_events.append(event)
        if len(self.spoofing_events) > self.max_spoofing_history:
            self.spoofing_events.pop(0)
        
        # Get adjustment from strategy
        adjustment = self.adjustment_strategy.adjust_for_spoofing(
            event,
            self.active_quotes,
            self.current_spread_bps,
            self.current_size_multiplier
        )
        
        # Execute adjustment
        self._execute_adjustment(adjustment)
        
        # Update statistics
        response_time_us = (time.time_ns() - start_ns) / 1000.0
        self._update_stats(adjustment, response_time_us)
        
        return adjustment
    
    def on_vpin_spike(self, vpin_value: float, spread_adjustment_bps: float) -> ExecutionAdjustment:
        """
        Handle VPIN spike - widen spread to compensate for adverse selection.
        
        Args:
            vpin_value: Current VPIN value
            spread_adjustment_bps: Recommended spread adjustment
            
        Returns:
            ExecutionAdjustment with spread widening
        """
        start_ns = time.time_ns()
        
        if self.is_trading_halted:
            return self._create_hold_adjustment("Trading halted")
        
        adjustment = self.adjustment_strategy.adjust_for_vpin(
            vpin_value,
            spread_adjustment_bps,
            self.current_spread_bps
        )
        
        self._execute_adjustment(adjustment)
        
        response_time_us = (time.time_ns() - start_ns) / 1000.0
        self._update_stats(adjustment, response_time_us)
        
        return adjustment
    
    def on_toxicity_alert(self, toxicity_score: float, risk_level: str) -> ExecutionAdjustment:
        """
        Handle toxicity alert - reduce size or halt trading.
        
        Args:
            toxicity_score: Current toxicity score (0-1)
            risk_level: Risk level string
            
        Returns:
            ExecutionAdjustment with appropriate action
        """
        start_ns = time.time_ns()
        
        adjustment = self.adjustment_strategy.adjust_for_toxicity(
            toxicity_score,
            risk_level,
            self.current_size_multiplier,
            self.is_trading_halted
        )
        
        self._execute_adjustment(adjustment)
        
        response_time_us = (time.time_ns() - start_ns) / 1000.0
        self._update_stats(adjustment, response_time_us)
        
        return adjustment
    
    def on_layering_detected(self, spread_adjustment_bps: float) -> ExecutionAdjustment:
        """
        Handle layering detection - adjust spread and pricing.
        
        Args:
            spread_adjustment_bps: Recommended spread adjustment
            
        Returns:
            ExecutionAdjustment with pricing changes
        """
        start_ns = time.time_ns()
        
        if self.is_trading_halted:
            return self._create_hold_adjustment("Trading halted")
        
        adjustment = self.adjustment_strategy.adjust_for_layering(
            spread_adjustment_bps,
            self.current_spread_bps
        )
        
        self._execute_adjustment(adjustment)
        
        response_time_us = (time.time_ns() - start_ns) / 1000.0
        self._update_stats(adjustment, response_time_us)
        
        return adjustment
    
    def update_quote(self, symbol: str, bid: float, ask: float, size: float) -> None:
        """Update active quote for a symbol."""
        self.active_quotes[symbol] = (bid, ask, size)
    
    def remove_quote(self, symbol: str) -> None:
        """Remove quote for a symbol."""
        if symbol in self.active_quotes:
            del self.active_quotes[symbol]
    
    def resume_trading(self) -> None:
        """Manually resume trading after halt."""
        self.is_trading_halted = False
        self.halt_until_ns = 0
    
    def get_status(self) -> Dict:
        """Get current execution adjuster status."""
        return {
            'is_trading_halted': self.is_trading_halted,
            'current_spread_bps': self.current_spread_bps,
            'current_size_multiplier': self.current_size_multiplier,
            'active_quotes_count': len(self.active_quotes),
            'recent_spoofing_events': len(self.spoofing_events),
            'stats': self.stats.copy(),
        }
    
    def clear(self) -> None:
        """Clear all state."""
        self.active_quotes.clear()
        self.spoofing_events.clear()
        self.current_spread_bps = 0.0
        self.current_size_multiplier = 1.0
        self.is_trading_halted = False
        self.halt_until_ns = 0
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}
    
    def _execute_adjustment(self, adjustment: ExecutionAdjustment) -> None:
        """Execute the adjustment action."""
        if adjustment.action == ExecutionAction.PULL_QUOTES:
            if self._quote_pull_callback:
                self._quote_pull_callback(list(self.active_quotes.keys()))
            self.active_quotes.clear()
            
        elif adjustment.action == ExecutionAction.WIDEN_SPREAD:
            self.current_spread_bps = adjustment.adjusted_bid  # Reusing field for spread
            if self._quote_update_callback:
                self._quote_update_callback(adjustment)
                
        elif adjustment.action == ExecutionAction.REDUCE_SIZE:
            self.current_size_multiplier = adjustment.size_multiplier
            
        elif adjustment.action == ExecutionAction.HALT_TRADING:
            self.is_trading_halted = True
            self.halt_until_ns = time.time_ns() + adjustment.halt_duration_ms * 1_000_000
            if self._halt_callback:
                self._halt_callback(adjustment.halt_duration_ms)
        
        self.last_adjustment_ns = time.time_ns()
    
    def _create_hold_adjustment(self, reason: str) -> ExecutionAdjustment:
        """Create a hold adjustment."""
        return ExecutionAdjustment(
            timestamp_ns=time.time_ns(),
            action=ExecutionAction.HOLD,
            reason=reason,
            original_bid=None,
            original_ask=None,
            adjusted_bid=None,
            adjusted_ask=None,
            size_multiplier=1.0,
            halt_duration_ms=0
        )
    
    def _update_stats(self, adjustment: ExecutionAdjustment, response_time_us: float) -> None:
        """Update statistics after adjustment."""
        self.stats['total_adjustments'] += 1
        
        if adjustment.action == ExecutionAction.PULL_QUOTES:
            self.stats['quotes_pulled'] += 1
        elif adjustment.action == ExecutionAction.WIDEN_SPREAD:
            self.stats['spread_widened'] += 1
        elif adjustment.action == ExecutionAction.HALT_TRADING:
            self.stats['trades_halted'] += 1
        
        # Update running average response time
        n = self.stats['total_adjustments']
        self.stats['avg_response_time_us'] = (
            (self.stats['avg_response_time_us'] * (n - 1) + response_time_us) / n
        )


# Strategy Pattern for Adjustment Strategies
class AdjustmentStrategy:
    """Base class for adjustment strategies."""
    
    def adjust_for_spoofing(self, event: SpoofingEvent, active_quotes: Dict,
                           current_spread_bps: float, size_mult: float) -> ExecutionAdjustment:
        raise NotImplementedError
    
    def adjust_for_vpin(self, vpin: float, spread_adj_bps: float,
                       current_spread_bps: float) -> ExecutionAdjustment:
        raise NotImplementedError
    
    def adjust_for_toxicity(self, toxicity: float, risk_level: str,
                           size_mult: float, is_halted: bool) -> ExecutionAdjustment:
        raise NotImplementedError
    
    def adjust_for_layering(self, spread_adj_bps: float,
                           current_spread_bps: float) -> ExecutionAdjustment:
        raise NotImplementedError


class AggressiveAdjustmentStrategy(AdjustmentStrategy):
    """Aggressive strategy - quick to pull quotes and halt."""
    
    def adjust_for_spoofing(self, event: SpoofingEvent, active_quotes: Dict,
                           current_spread_bps: float, size_mult: float) -> ExecutionAdjustment:
        # Always pull quotes on spoofing detection
        return ExecutionAdjustment(
            timestamp_ns=time.time_ns(),
            action=ExecutionAction.PULL_QUOTES,
            reason=f"Spoofing detected at {event.price_level} with {event.confidence:.0%} confidence",
            original_bid=None,
            original_ask=None,
            adjusted_bid=None,
            adjusted_ask=None,
            size_multiplier=1.0,
            halt_duration_ms=1000
        )
    
    def adjust_for_vpin(self, vpin: float, spread_adj_bps: float,
                       current_spread_bps: float) -> ExecutionAdjustment:
        new_spread = current_spread_bps + spread_adj_bps
        return ExecutionAdjustment(
            timestamp_ns=time.time_ns(),
            action=ExecutionAction.WIDEN_SPREAD,
            reason=f"VPIN spike: {vpin:.2%}",
            original_bid=current_spread_bps,
            original_ask=None,
            adjusted_bid=new_spread,
            adjusted_ask=None,
            size_multiplier=1.0,
            halt_duration_ms=0
        )
    
    def adjust_for_toxicity(self, toxicity: float, risk_level: str,
                           size_mult: float, is_halted: bool) -> ExecutionAdjustment:
        if toxicity > 0.8 or risk_level == "CRITICAL":
            return ExecutionAdjustment(
                timestamp_ns=time.time_ns(),
                action=ExecutionAction.HALT_TRADING,
                reason=f"Critical toxicity: {toxicity:.2%}",
                original_bid=None,
                original_ask=None,
                adjusted_bid=None,
                adjusted_ask=None,
                size_multiplier=0.0,
                halt_duration_ms=5000
            )
        elif toxicity > 0.5:
            return ExecutionAdjustment(
                timestamp_ns=time.time_ns(),
                action=ExecutionAction.REDUCE_SIZE,
                reason=f"High toxicity: {toxicity:.2%}",
                original_bid=None,
                original_ask=None,
                adjusted_bid=None,
                adjusted_ask=None,
                size_multiplier=0.25,
                halt_duration_ms=0
            )
        return self._create_hold("Low toxicity")
    
    def adjust_for_layering(self, spread_adj_bps: float,
                           current_spread_bps: float) -> ExecutionAdjustment:
        new_spread = current_spread_bps + spread_adj_bps
        return ExecutionAdjustment(
            timestamp_ns=time.time_ns(),
            action=ExecutionAction.WIDEN_SPREAD,
            reason="Layering detected",
            original_bid=current_spread_bps,
            original_ask=None,
            adjusted_bid=new_spread,
            adjusted_ask=None,
            size_multiplier=1.0,
            halt_duration_ms=0
        )
    
    def _create_hold(self, reason: str) -> ExecutionAdjustment:
        return ExecutionAdjustment(
            timestamp_ns=time.time_ns(),
            action=ExecutionAction.HOLD,
            reason=reason,
            original_bid=None,
            original_ask=None,
            adjusted_bid=None,
            adjusted_ask=None,
            size_multiplier=1.0,
            halt_duration_ms=0
        )


class ConservativeAdjustmentStrategy(AdjustmentStrategy):
    """Conservative strategy - prefers widening spread over pulling quotes."""
    
    def adjust_for_spoofing(self, event: SpoofingEvent, active_quotes: Dict,
                           current_spread_bps: float, size_mult: float) -> ExecutionAdjustment:
        # Only pull if high confidence
        if event.confidence > 0.9:
            return ExecutionAdjustment(
                timestamp_ns=time.time_ns(),
                action=ExecutionAction.PULL_QUOTES,
                reason=f"High-confidence spoofing: {event.confidence:.0%}",
                original_bid=None,
                original_ask=None,
                adjusted_bid=None,
                adjusted_ask=None,
                size_multiplier=1.0,
                halt_duration_ms=500
            )
        else:
            # Just widen spread
            return ExecutionAdjustment(
                timestamp_ns=time.time.ns(),
                action=ExecutionAction.WIDEN_SPREAD,
                reason=f"Spoofing suspected: {event.confidence:.0%}",
                original_bid=current_spread_bps,
                original_ask=None,
                adjusted_bid=current_spread_bps + 3.0,
                adjusted_ask=None,
                size_multiplier=0.8,
                halt_duration_ms=0
            )
    
    def adjust_for_vpin(self, vpin: float, spread_adj_bps: float,
                       current_spread_bps: float) -> ExecutionAdjustment:
        new_spread = current_spread_bps + spread_adj_bps * 0.5  # More conservative
        return ExecutionAdjustment(
            timestamp_ns=time.time_ns(),
            action=ExecutionAction.WIDEN_SPREAD,
            reason=f"VPIN elevated: {vpin:.2%}",
            original_bid=current_spread_bps,
            original_ask=None,
            adjusted_bid=new_spread,
            adjusted_ask=None,
            size_multiplier=0.8,
            halt_duration_ms=0
        )
    
    def adjust_for_toxicity(self, toxicity: float, risk_level: str,
                           size_mult: float, is_halted: bool) -> ExecutionAdjustment:
        if toxicity > 0.9:
            return ExecutionAdjustment(
                timestamp_ns=time.time_ns(),
                action=ExecutionAction.HALT_TRADING,
                reason=f"Extreme toxicity: {toxicity:.2%}",
                original_bid=None,
                original_ask=None,
                adjusted_bid=None,
                adjusted_ask=None,
                size_multiplier=0.0,
                halt_duration_ms=10000
            )
        elif toxicity > 0.6:
            return ExecutionAdjustment(
                timestamp_ns=time.time_ns(),
                action=ExecutionAction.REDUCE_SIZE,
                reason=f"Elevated toxicity: {toxicity:.2%}",
                original_bid=None,
                original_ask=None,
                adjusted_bid=None,
                adjusted_ask=None,
                size_multiplier=0.5,
                halt_duration_ms=0
            )
        return self._create_hold("Acceptable toxicity")
    
    def adjust_for_layering(self, spread_adj_bps: float,
                           current_spread_bps: float) -> ExecutionAdjustment:
        return ExecutionAdjustment(
            timestamp_ns=time.time_ns(),
            action=ExecutionAction.WIDEN_SPREAD,
            reason="Layering suspected",
            original_bid=current_spread_bps,
            original_ask=None,
            adjusted_bid=current_spread_bps + spread_adj_bps * 0.5,
            adjusted_ask=None,
            size_multiplier=0.9,
            halt_duration_ms=0
        )
    
    def _create_hold(self, reason: str) -> ExecutionAdjustment:
        return ExecutionAdjustment(
            timestamp_ns=time.time_ns(),
            action=ExecutionAction.HOLD,
            reason=reason,
            original_bid=None,
            original_ask=None,
            adjusted_bid=None,
            adjusted_ask=None,
            size_multiplier=1.0,
            halt_duration_ms=0
        )


if __name__ == "__main__":
    # Example usage
    adjuster = ExecutionAdjuster()
    
    # Set aggressive strategy
    adjuster.set_adjustment_strategy(AggressiveAdjustmentStrategy())
    
    # Add some quotes
    adjuster.update_quote("BTCUSDT", 49999.0, 50001.0, 1.0)
    adjuster.update_quote("ETHUSDT", 2999.0, 3001.0, 10.0)
    
    # Simulate spoofing detection
    spoof_event = SpoofingEvent(
        timestamp_ns=time.time_ns(),
        event_type="rapid_cancel",
        confidence=0.95,
        price_level=50000.0,
        volume=100.0,
        side="bid"
    )
    
    adjustment = adjuster.on_spoofing_detected(spoof_event)
    print(f"Adjustment Action: {adjustment.action.value}")
    print(f"Reason: {adjustment.reason}")
    
    # Get status
    status = adjuster.get_status()
    print(f"\nStatus: {status}")
