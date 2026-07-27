#!/usr/bin/env python3
"""
Hedging Soul Logger - SOUL.md Logging for Hedging Events

This module logs hedging failures and gamma profits to SOUL.md.
It updates the log when correlation breakdowns cause temporary
directional exposure, ensuring complete audit trail of all
hedging events.

Key Features:
- Structured logging to SOUL.md format
- Correlation breakdown event tracking
- Gamma profit/loss recording
- Directional exposure alerts
- Strict type hinting for memory safety

Target: Maintain complete audit trail in SOUL.md
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import threading
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of hedging events."""
    HEDGE_EXECUTED = auto()
    HEDGE_FAILED = auto()
    CORRELATION_BREAKDOWN = auto()
    GAMMA_PROFIT = auto()
    GAMMA_LOSS = auto()
    DELTA_BREACH = auto()
    THETA_DECAY_ALERT = auto()
    REBALANCE_COMPLETE = auto()
    DIRECTIONAL_EXPOSURE = auto()
    HEDGING_SUSPENDED = auto()
    HEDGING_RESUMED = auto()


class SeverityLevel(Enum):
    """Event severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class HedgingEvent:
    """A single hedging event for logging."""
    event_id: str
    event_type: EventType
    severity: SeverityLevel
    timestamp: datetime
    asset: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    delta_before: Optional[Decimal] = None
    delta_after: Optional[Decimal] = None
    pnl_impact: Optional[Decimal] = None
    correlation_breakdown: bool = False
    directional_exposure: bool = False


@dataclass
class SoulEntry:
    """Entry in SOUL.md log."""
    event: HedgingEvent
    formatted_entry: str


class HedgingSoulLogger:
    """
    Logger for hedging events that writes to SOUL.md.
    
    Maintains a complete audit trail of all hedging activities,
    failures, and gamma scalping results.
    """
    
    def __init__(
        self,
        soul_file_path: str = "SOUL.md",
        max_entries_in_memory: int = 1000
    ):
        self.soul_file_path = soul_file_path
        self.max_entries_in_memory = max_entries_in_memory
        
        # In-memory event buffer
        self._events: List[HedgingEvent] = []
        self._entries: List[SoulEntry] = []
        
        # Statistics
        self.total_events: int = 0
        self.correlation_breakdowns: int = 0
        self.hedge_failures: int = 0
        self.gamma_profits: Decimal = Decimal('0')
        self.gamma_losses: Decimal = Decimal('0')
        self.directional_exposures: int = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize SOUL.md file
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize or append to SOUL.md file."""
        header = """# ZAID PERSONAL CRYPTO TRADING BOT - SOUL.md

## Hedging Strategy Audit Log

This document contains the complete audit trail of all hedging events,
including delta-neutral rebalancing, gamma scalping, correlation breakdowns,
and directional exposure incidents.

---

"""
        try:
            if not os.path.exists(self.soul_file_path):
                with open(self.soul_file_path, 'w', encoding='utf-8') as f:
                    f.write(header)
            else:
                # Append separator for new session
                with open(self.soul_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n---\n\n## New Session: {datetime.now(timezone.utc).isoformat()}\n\n")
        except Exception as e:
            logger.error(f"Failed to initialize SOUL.md: {e}")
    
    def log_event(self, event: HedgingEvent) -> None:
        """Log a hedging event."""
        with self._lock:
            self._events.append(event)
            self.total_events += 1
            
            # Update statistics
            if event.correlation_breakdown:
                self.correlation_breakdowns += 1
            
            if event.event_type == EventType.HEDGE_FAILED:
                self.hedge_failures += 1
            
            if event.event_type == EventType.GAMMA_PROFIT and event.pnl_impact:
                self.gamma_profits += event.pnl_impact
            
            if event.event_type == EventType.GAMMA_LOSS and event.pnl_impact:
                self.gamma_losses += event.pnl_impact.abs()
            
            if event.directional_exposure:
                self.directional_exposures += 1
            
            # Create formatted entry
            entry = self._format_entry(event)
            self._entries.append(entry)
            
            # Trim old entries if needed
            if len(self._entries) > self.max_entries_in_memory:
                self._entries.pop(0)
            
            # Write to file
            self._write_to_soul(entry)
    
    def _format_entry(self, event: HedgingEvent) -> SoulEntry:
        """Format event as markdown entry."""
        timestamp_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC'
        
        # Build details section
        details_lines = []
        for key, value in event.details.items():
            details_lines.append(f"- **{key}**: {value}")
        
        # Delta info
        delta_info = ""
        if event.delta_before is not None and event.delta_after is not None:
            delta_info = f"\n**Delta Change**: {event.delta_before} → {event.delta_after}"
        
        # PnL info
        pnl_info = ""
        if event.pnl_impact is not None:
            pnl_sign = "+" if event.pnl_impact >= 0 else ""
            pnl_info = f"\n**P&L Impact**: {pnl_sign}{event.pnl_impact}"
        
        # Warning flags
        flags = []
        if event.correlation_breakdown:
            flags.append("⚠️ CORRELATION BREAKDOWN")
        if event.directional_exposure:
            flags.append("🔴 DIRECTIONAL EXPOSURE")
        
        flag_str = " ".join(flags)
        if flag_str:
            flag_str = f"\n\n{flag_str}"
        
        formatted = f"""### [{event.severity.value}] {event.event_type.name}

**Event ID**: `{event.event_id}`  
**Timestamp**: {timestamp_str}  
**Asset**: {event.asset}  
**Message**: {event.message}
{delta_info}{pnl_info}{flag_str}

**Details**:
{chr(10).join(details_lines) if details_lines else '- None'}

---

"""
        return SoulEntry(event=event, formatted_entry=formatted)
    
    def _write_to_soul(self, entry: SoulEntry) -> None:
        """Write entry to SOUL.md file."""
        try:
            with open(self.soul_file_path, 'a', encoding='utf-8') as f:
                f.write(entry.formatted_entry)
        except Exception as e:
            logger.error(f"Failed to write to SOUL.md: {e}")
    
    def log_correlation_breakdown(
        self,
        asset: str,
        pair: str,
        correlation: float,
        expected_correlation: float,
        z_score: float,
        action_taken: str
    ) -> None:
        """Log a correlation breakdown event."""
        event = HedgingEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.CORRELATION_BREAKDOWN,
            severity=SeverityLevel.CRITICAL,
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            message=f"Correlation breakdown detected for {pair}",
            details={
                'pair': pair,
                'current_correlation': correlation,
                'expected_correlation': expected_correlation,
                'z_score': z_score,
                'action_taken': action_taken,
            },
            correlation_breakdown=True,
            directional_exposure=True,
        )
        self.log_event(event)
    
    def log_hedge_executed(
        self,
        asset: str,
        hedge_id: str,
        delta_before: Decimal,
        delta_after: Decimal,
        quantity: Decimal,
        slippage_bps: Decimal,
        execution_time_us: int
    ) -> None:
        """Log a successful hedge execution."""
        event = HedgingEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.HEDGE_EXECUTED,
            severity=SeverityLevel.INFO,
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            message=f"Hedge executed for {asset}",
            details={
                'hedge_id': hedge_id,
                'quantity': str(quantity),
                'slippage_bps': str(slippage_bps),
                'execution_time_us': execution_time_us,
            },
            delta_before=delta_before,
            delta_after=delta_after,
        )
        self.log_event(event)
    
    def log_hedge_failed(
        self,
        asset: str,
        reason: str,
        delta_exposure: Decimal,
        error_message: str
    ) -> None:
        """Log a failed hedge execution."""
        event = HedgingEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.HEDGE_FAILED,
            severity=SeverityLevel.ERROR,
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            message=f"Hedge execution failed: {reason}",
            details={
                'reason': reason,
                'error_message': error_message,
            },
            delta_before=delta_exposure,
            delta_after=delta_exposure,
        )
        self.log_event(event)
    
    def log_gamma_profit(
        self,
        asset: str,
        scalp_id: str,
        profit: Decimal,
        gamma_at_execution: float,
        holding_time_ms: int
    ) -> None:
        """Log a gamma scalping profit."""
        event = HedgingEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.GAMMA_PROFIT,
            severity=SeverityLevel.INFO,
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            message=f"Gamma scalp profit: {profit}",
            details={
                'scalp_id': scalp_id,
                'gamma': gamma_at_execution,
                'holding_time_ms': holding_time_ms,
            },
            pnl_impact=profit,
        )
        self.log_event(event)
    
    def log_gamma_loss(
        self,
        asset: str,
        scalp_id: str,
        loss: Decimal,
        gamma_at_execution: float,
        theta_decay: Decimal
    ) -> None:
        """Log a gamma scalping loss."""
        event = HedgingEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.GAMMA_LOSS,
            severity=SeverityLevel.WARNING,
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            message=f"Gamma scalp loss: {loss}",
            details={
                'scalp_id': scalp_id,
                'gamma': gamma_at_execution,
                'theta_decay': str(theta_decay),
            },
            pnl_impact=loss,
        )
        self.log_event(event)
    
    def log_directional_exposure(
        self,
        asset: str,
        exposure_amount: Decimal,
        exposure_side: str,
        reason: str,
        expected_duration_seconds: int
    ) -> None:
        """Log temporary directional exposure."""
        event = HedgingEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.DIRECTIONAL_EXPOSURE,
            severity=SeverityLevel.WARNING,
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            message=f"Directional exposure detected: {exposure_side} {exposure_amount}",
            details={
                'exposure_amount': str(exposure_amount),
                'exposure_side': exposure_side,
                'reason': reason,
                'expected_duration_seconds': expected_duration_seconds,
            },
            directional_exposure=True,
        )
        self.log_event(event)
    
    def log_rebalance_complete(
        self,
        portfolio_delta: Decimal,
        hedges_executed: int,
        total_cost: Decimal,
        time_taken_ms: int
    ) -> None:
        """Log completion of portfolio rebalancing."""
        event = HedgingEvent(
            event_id=self._generate_event_id(),
            event_type=EventType.REBALANCE_COMPLETE,
            severity=SeverityLevel.INFO,
            timestamp=datetime.now(timezone.utc),
            asset="PORTFOLIO",
            message=f"Rebalancing complete: {hedges_executed} hedges executed",
            details={
                'hedges_executed': hedges_executed,
                'total_cost': str(total_cost),
                'time_taken_ms': time_taken_ms,
            },
            delta_after=portfolio_delta,
        )
        self.log_event(event)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return f"EVT-{uuid.uuid4().hex[:12].upper()}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            net_gamma_pnl = self.gamma_profits - self.gamma_losses
            
            return {
                'total_events': self.total_events,
                'correlation_breakdowns': self.correlation_breakdowns,
                'hedge_failures': self.hedge_failures,
                'gamma_profits': float(self.gamma_profits),
                'gamma_losses': float(self.gamma_losses),
                'net_gamma_pnl': float(net_gamma_pnl),
                'directional_exposures': self.directional_exposures,
                'events_in_memory': len(self._events),
            }
    
    def get_recent_events(self, limit: int = 10) -> List[HedgingEvent]:
        """Get recent events from memory."""
        with self._lock:
            return self._events[-limit:]
    
    def flush_to_file(self) -> None:
        """Force flush all pending entries to file."""
        with self._lock:
            # All entries are already written, but we can add a summary
            stats = self.get_statistics()
            summary = f"""
## Session Summary

**Generated**: {datetime.now(timezone.utc).isoformat()}

- Total Events: {stats['total_events']}
- Correlation Breakdowns: {stats['correlation_breakdowns']}
- Hedge Failures: {stats['hedge_failures']}
- Net Gamma P&L: {stats['net_gamma_pnl']}
- Directional Exposures: {stats['directional_exposures']}

---
"""
            try:
                with open(self.soul_file_path, 'a', encoding='utf-8') as f:
                    f.write(summary)
            except Exception as e:
                logger.error(f"Failed to write summary: {e}")


def main() -> None:
    """Example usage."""
    from decimal import Decimal
    
    # Initialize logger
    logger = HedgingSoulLogger(soul_file_path="SOUL.md")
    
    # Log various events
    logger.log_hedge_executed(
        asset="BTC",
        hedge_id="HEDGE-001",
        delta_before=Decimal('50000'),
        delta_after=Decimal('100'),
        quantity=Decimal('0.98'),
        slippage_bps=Decimal('2.5'),
        execution_time_us=1500,
    )
    
    logger.log_correlation_breakdown(
        asset="ETH",
        pair="ETH-BTC",
        correlation=0.3,
        expected_correlation=0.85,
        z_score=-3.5,
        action_taken="Suspended cross-asset hedging",
    )
    
    logger.log_gamma_profit(
        asset="BTC",
        scalp_id="SCALP-001",
        profit=Decimal('150.00'),
        gamma_at_execution=0.002,
        holding_time_ms=450,
    )
    
    logger.log_directional_exposure(
        asset="SOL",
        exposure_amount=Decimal('5000'),
        exposure_side="LONG",
        reason="Correlation breakdown prevented hedge execution",
        expected_duration_seconds=300,
    )
    
    logger.flush_to_file()
    
    # Print statistics
    stats = logger.get_statistics()
    print(f"Logging Statistics: {stats}")


if __name__ == "__main__":
    main()
