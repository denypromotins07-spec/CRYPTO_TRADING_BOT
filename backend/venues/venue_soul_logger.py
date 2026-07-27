#!/usr/bin/env python3
"""
backend/venues/venue_soul_logger.py

ZAID PERSONAL CRYPTO TRADING BOT - Stage 33: Multi-Venue SOR
Chapter 4: Venue Soul Logger

Logs routing failures and latency spikes to SOUL.md.
Updates SOUL.md if the SOR successfully saves a trade via cross-venue routing.
Strictly respects 8GB RAM limit on AMD Ryzen AI 5 laptop.

Features:
- Structured logging of all SOR events
- Latency spike detection and logging
- Routing failure tracking with root cause analysis
- Successful cross-venue save documentation
- Automatic SOUL.md file management
- Event categorization and severity levels

Type hints enforced for memory safety and IDE support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import threading
import os
from datetime import datetime


class EventType(Enum):
    """Types of SOR events to log."""
    ROUTING_SUCCESS = "routing_success"
    ROUTING_FAILURE = "routing_failure"
    LATENCY_SPIKE = "latency_spike"
    VENUE_FAILOVER = "venue_failover"
    ARBITRAGE_EXECUTED = "arbitrage_executed"
    RECONCILIATION_ALERT = "reconciliation_alert"
    RATE_LIMIT_WARNING = "rate_limit_warning"
    ORDER_SPLIT = "order_split"
    LIQUIDITY_DETECTED = "liquidity_detected"
    SYSTEM_HEARTBEAT = "system_heartbeat"


class Severity(Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SorEvent:
    """Represents a single SOR event for logging."""
    event_id: int
    event_type: EventType
    severity: Severity
    timestamp_ns: int
    venue: Optional[str]
    symbol: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    latency_us: Optional[int] = None
    saved_bps: Optional[float] = None


class VenueSoulLogger:
    """
    Logs SOR events to SOUL.md file.
    
    Captures routing decisions, failures, and successful saves.
    Provides audit trail for cross-venue trading operations.
    """
    
    def __init__(self, soul_file_path: str = "SOUL.md"):
        """
        Initialize the venue soul logger.
        
        Args:
            soul_file_path: Path to the SOUL.md log file
        """
        self._lock = threading.RLock()
        self.soul_file_path = soul_file_path
        
        # Event tracking
        self.events: List[SorEvent] = []
        self.event_counter: int = 0
        
        # Statistics
        self.stats: Dict[str, int] = {
            "total_events": 0,
            "routing_successes": 0,
            "routing_failures": 0,
            "latency_spikes": 0,
            "successful_saves": 0,
            "failovers": 0,
        }
        
        # Thresholds
        self.latency_spike_threshold_us: int = 1000  # 1ms = spike
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize or append to SOUL.md file."""
        header = """# ZAID PERSONAL CRYPTO TRADING BOT - SOUL Log

## Multi-Venue Smart Order Routing (SOR) Audit Trail

This file contains the complete audit trail of the SOR system, including:
- Cross-venue routing decisions
- Latency arbitrage executions
- Venue failovers and health events
- Reconciliation alerts
- Successful trade saves via smart routing

---

"""
        try:
            if not os.path.exists(self.soul_file_path):
                with open(self.soul_file_path, 'w', encoding='utf-8') as f:
                    f.write(header)
            else:
                # Append separator for new session
                with open(self.soul_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n---\n\n## Session Started: {datetime.utcnow().isoformat()}Z\n\n")
        except Exception as e:
            print(f"[WARN] Could not initialize SOUL.md: {e}")
    
    def log_event(
        self,
        event_type: EventType,
        message: str,
        venue: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: Severity = Severity.INFO,
        latency_us: Optional[int] = None,
        saved_bps: Optional[float] = None,
    ) -> int:
        """
        Log an SOR event.
        
        Args:
            event_type: Type of event
            message: Human-readable message
            venue: Related venue (if applicable)
            symbol: Related symbol (if applicable)
            details: Additional structured data
            severity: Event severity level
            latency_us: Event latency in microseconds (if applicable)
            saved_bps: Basis points saved by SOR (if applicable)
            
        Returns:
            Event ID
        """
        now_ns = time.time_ns()
        
        with self._lock:
            self.event_counter += 1
            
            event = SorEvent(
                event_id=self.event_counter,
                event_type=event_type,
                severity=severity,
                timestamp_ns=now_ns,
                venue=venue,
                symbol=symbol,
                message=message,
                details=details or {},
                latency_us=latency_us,
                saved_bps=saved_bps,
            )
            
            self.events.append(event)
            self._update_stats(event)
            self._write_to_soul_file(event)
            
            # Keep only last 10000 events in memory
            if len(self.events) > 10000:
                self.events = self.events[-10000:]
        
        return event.event_id
    
    def _update_stats(self, event: SorEvent) -> None:
        """Update statistics based on event type."""
        self.stats["total_events"] += 1
        
        if event.event_type == EventType.ROUTING_SUCCESS:
            self.stats["routing_successes"] += 1
        elif event.event_type == EventType.ROUTING_FAILURE:
            self.stats["routing_failures"] += 1
        elif event.event_type == EventType.LATENCY_SPIKE:
            self.stats["latency_spikes"] += 1
        elif event.event_type == EventType.ARBITRAGE_EXECUTED:
            self.stats["successful_saves"] += 1
        elif event.event_type == EventType.VENUE_FAILOVER:
            self.stats["failovers"] += 1
    
    def _write_to_soul_file(self, event: SorEvent) -> None:
        """Write event to SOUL.md file."""
        timestamp = datetime.utcfromtimestamp(event.timestamp_ns / 1_000_000_000)
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
        
        # Format event entry
        entry_lines = [
            f"### Event #{event.event_id} | {event.event_type.value.upper()}",
            f"**Timestamp:** {ts_str}",
            f"**Severity:** {event.severity.value.upper()}",
        ]
        
        if event.venue:
            entry_lines.append(f"**Venue:** {event.venue}")
        if event.symbol:
            entry_lines.append(f"**Symbol:** {event.symbol}")
        
        entry_lines.append(f"**Message:** {event.message}")
        
        # Add metrics if present
        metrics = []
        if event.latency_us is not None:
            metrics.append(f"Latency: {event.latency_us}μs")
        if event.saved_bps is not None:
            metrics.append(f"Saved: {event.saved_bps:.2f} bps")
        
        if metrics:
            entry_lines.append(f"**Metrics:** {' | '.join(metrics)}")
        
        # Add details if present
        if event.details:
            entry_lines.append("**Details:**")
            for key, value in event.details.items():
                entry_lines.append(f"- `{key}`: {value}")
        
        entry_lines.append("")  # Empty line between entries
        
        try:
            with open(self.soul_file_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(entry_lines))
        except Exception as e:
            print(f"[WARN] Could not write to SOUL.md: {e}")
    
    def log_routing_success(
        self,
        venue: str,
        symbol: str,
        saved_bps: float,
        original_venue: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Log successful SOR routing that saved basis points."""
        msg = f"SOR routed to {venue}"
        if original_venue:
            msg += f" instead of {original_venue}"
        msg += f", saving {saved_bps:.2f} bps"
        
        return self.log_event(
            event_type=EventType.ROUTING_SUCCESS,
            message=msg,
            venue=venue,
            symbol=symbol,
            details=details,
            severity=Severity.INFO,
            saved_bps=saved_bps,
        )
    
    def log_routing_failure(
        self,
        venue: str,
        symbol: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Log routing failure."""
        return self.log_event(
            event_type=EventType.ROUTING_FAILURE,
            message=f"Routing failed on {venue}: {reason}",
            venue=venue,
            symbol=symbol,
            details=details,
            severity=Severity.ERROR,
        )
    
    def log_latency_spike(
        self,
        venue: str,
        latency_us: int,
        threshold_us: int,
        symbol: Optional[str] = None,
    ) -> int:
        """Log latency spike detection."""
        return self.log_event(
            event_type=EventType.LATENCY_SPIKE,
            message=f"Latency spike on {venue}: {latency_us}μs (threshold: {threshold_us}μs)",
            venue=venue,
            symbol=symbol,
            latency_us=latency_us,
            severity=Severity.WARNING,
        )
    
    def log_arbitrage_save(
        self,
        buy_venue: str,
        sell_venue: str,
        symbol: str,
        profit_bps: float,
        size: float,
    ) -> int:
        """Log successful arbitrage execution."""
        return self.log_event(
            event_type=EventType.ARBITRAGE_EXECUTED,
            message=f"Cross-venue arb: Buy {buy_venue} -> Sell {sell_venue}, profit {profit_bps:.2f} bps",
            venue=f"{buy_venue}/{sell_venue}",
            symbol=symbol,
            details={"size": size, "profit_bps": profit_bps},
            severity=Severity.INFO,
            saved_bps=profit_bps,
        )
    
    def log_venue_failover(
        self,
        from_venue: str,
        to_venue: str,
        reason: str,
        symbol: Optional[str] = None,
    ) -> int:
        """Log venue failover event."""
        return self.log_event(
            event_type=EventType.VENUE_FAILOVER,
            message=f"Failover from {from_venue} to {to_venue}: {reason}",
            venue=to_venue,
            symbol=symbol,
            details={"from_venue": from_venue, "reason": reason},
            severity=Severity.WARNING,
        )
    
    def log_reconciliation_alert(
        self,
        venue: str,
        asset_or_symbol: str,
        drift_pct: float,
        internal_value: float,
        external_value: float,
    ) -> int:
        """Log reconciliation alert."""
        return self.log_event(
            event_type=EventType.RECONCILIATION_ALERT,
            message=f"Reconciliation drift on {venue}/{asset_or_symbol}: {drift_pct:.2f}%",
            venue=venue,
            symbol=asset_or_symbol,
            details={
                "drift_pct": drift_pct,
                "internal_value": internal_value,
                "external_value": external_value,
            },
            severity=Severity.CRITICAL if drift_pct > 1.0 else Severity.WARNING,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            return {
                **self.stats,
                "events_in_memory": len(self.events),
                "last_event_id": self.event_counter,
            }
    
    def get_recent_events(self, count: int = 100) -> List[SorEvent]:
        """Get most recent events."""
        with self._lock:
            return self.events[-count:]
    
    def get_events_by_type(self, event_type: EventType) -> List[SorEvent]:
        """Get all events of a specific type."""
        with self._lock:
            return [e for e in self.events if e.event_type == event_type]
    
    def set_latency_threshold(self, threshold_us: int) -> None:
        """Set latency spike detection threshold."""
        self.latency_spike_threshold_us = threshold_us


# Example usage and testing
if __name__ == "__main__":
    logger = VenueSoulLogger()
    
    # Log various events
    logger.log_routing_success(
        venue="binance",
        symbol="BTCUSDT",
        saved_bps=2.5,
        original_venue="coinbase",
    )
    
    logger.log_latency_spike(
        venue="kraken",
        latency_us=5000,
        threshold_us=1000,
        symbol="ETHUSDT",
    )
    
    logger.log_arbitrage_save(
        buy_venue="binance",
        sell_venue="coinbase",
        symbol="BTCUSDT",
        profit_bps=5.2,
        size=0.1,
    )
    
    logger.log_venue_failover(
        from_venue="kraken",
        to_venue="bybit",
        reason="WebSocket disconnect",
        symbol="SOLUSDT",
    )
    
    logger.log_reconciliation_alert(
        venue="binance",
        asset_or_symbol="BTC",
        drift_pct=0.5,
        internal_value=1.5,
        external_value=1.5075,
    )
    
    # Print statistics
    stats = logger.get_statistics()
    print("Logging Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
