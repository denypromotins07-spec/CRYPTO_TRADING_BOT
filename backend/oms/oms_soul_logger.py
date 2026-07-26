#!/usr/bin/env python3
"""
OMS Soul Logger - Log Synthetic Order Failures and Reconciliation Drifts to SOUL.md

This module provides comprehensive logging for OMS events including:
- Synthetic order failures and successes
- Reconciliation drifts and corrections
- Trailing stop saves during sudden reversals
- Shadow model discoveries of new profitable regimes
- All critical OMS state transitions

Memory Optimized: Async file writes, buffered logging
Thread Safe: Proper locking for concurrent access
Platform: Optimized for AMD Ryzen AI 5, Windows PowerShell
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import time
import threading
import json
import os
from pathlib import Path


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    ALERT = "ALERT"  # For immediate attention required


class EventType(Enum):
    """Types of OMS events to log."""
    # Order Events
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_EXPIRED = "order_expired"
    
    # Synthetic Order Events
    BRACKET_ORDER_CREATED = "bracket_order_created"
    BRACKET_ORDER_TRIGGERED = "bracket_order_triggered"
    BRACKET_ORDER_COMPLETED = "bracket_order_completed"
    OCO_ORDER_CREATED = "oco_order_created"
    OCO_ORDER_EXECUTED = "oco_order_executed"
    OCO_ORDER_CANCELLED = "oco_order_cancelled"
    TRAILING_STOP_CREATED = "trailing_stop_created"
    TRAILING_STOP_UPDATED = "trailing_stop_updated"
    TRAILING_STOP_SAVED = "trailing_stop_saved"  # Saved trade from reversal
    
    # Reconciliation Events
    RECONCILIATION_STARTED = "reconciliation_started"
    RECONCILIATION_COMPLETED = "reconciliation_completed"
    DRIFT_DETECTED = "drift_detected"
    DRIFT_RESOLVED = "drift_resolved"
    FATAL_DESYNC = "fatal_desync"
    POSITION_FLATTENED = "position_flattened"
    
    # Post-Only Events
    POST_ONLY_ORDER_POSTED = "post_only_posted"
    POST_ONLY_ORDER_REJECTED = "post_only_rejected"
    FEE_SAVINGS_RECORDED = "fee_savings_recorded"
    
    # Sniper/Liquidity Events
    HIDDEN_LIQUIDITY_DETECTED = "hidden_liquidity_detected"
    SNIPER_EXECUTION = "sniper_execution"
    ICEBERG_DETECTED = "iceberg_detected"
    
    # State Machine Events
    STATE_TRANSITION = "state_transition"
    INVALID_TRANSITION_ATTEMPT = "invalid_transition_attempt"
    STALE_ORDER_DETECTED = "stale_order_detected"
    
    # Shadow/AB Testing Events
    SHADOW_MODEL_ACTIVATED = "shadow_model_activated"
    SHADOW_MODEL_DISCOVERY = "shadow_model_discovery"  # New profitable regime
    AB_TEST_STARTED = "ab_test_started"
    AB_TEST_COMPLETED = "ab_test_completed"
    AB_TEST_RESULT = "ab_test_result"
    
    # System Events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGE = "config_change"
    PERFORMANCE_ALERT = "performance_alert"


@dataclass
class OmsEvent:
    """Represents an OMS event for logging."""
    event_type: EventType
    timestamp_ms: int
    level: LogLevel
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    pnl_impact: Optional[float] = None
    latency_ns: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary for JSON serialization."""
        return {
            'event_type': self.event_type.value,
            'timestamp_ms': self.timestamp_ms,
            'timestamp_iso': datetime.fromtimestamp(
                self.timestamp_ms / 1000.0
            ).isoformat(),
            'level': self.level.value,
            'symbol': self.symbol,
            'order_id': self.order_id,
            'message': self.message,
            'details': self.details,
            'pnl_impact': self.pnl_impact,
            'latency_ns': self.latency_ns,
        }


@dataclass
class SoulEntry:
    """A single entry in the SOUL.md log."""
    entry_id: int
    event: OmsEvent
    tags: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: str = ""
    
    def to_markdown(self) -> str:
        """Convert entry to Markdown format for SOUL.md."""
        lines = [
            f"## Entry #{self.entry_id}",
            "",
            f"**Timestamp:** {self.event.timestamp_iso if hasattr(self.event, 'timestamp_iso') else datetime.fromtimestamp(self.event.timestamp_ms / 1000.0).isoformat()}",
            f"**Level:** {self.event.level.value}",
            f"**Type:** {self.event.event_type.value}",
            f"**Symbol:** {self.event.symbol or 'N/A'}",
            f"**Order ID:** {self.event.order_id or 'N/A'}",
            "",
            f"**Message:** {self.event.message}",
            "",
        ]
        
        if self.event.details:
            lines.append("**Details:**")
            lines.append("```json")
            lines.append(json.dumps(self.event.details, indent=2))
            lines.append("```")
            lines.append("")
        
        if self.event.pnl_impact is not None:
            lines.append(f"**PnL Impact:** {self.event.pnl_impact:.2f} USDT")
            lines.append("")
        
        if self.event.latency_ns is not None:
            lines.append(f"**Latency:** {self.event.latency_ns / 1_000_000:.3f} ms")
            lines.append("")
        
        if self.tags:
            lines.append(f"**Tags:** {', '.join(self.tags)}")
            lines.append("")
        
        if self.resolved:
            lines.append(f"**Status:** ✅ RESOLVED")
            if self.resolution_notes:
                lines.append(f"**Resolution Notes:** {self.resolution_notes}")
            lines.append("")
        elif self.acknowledged:
            lines.append("**Status:** ⏳ ACKNOWLEDGED (pending resolution)")
            lines.append("")
        else:
            lines.append("**Status:** 🔴 NEW")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return "\n".join(lines)


class SoulLoggerConfig:
    """Configuration for SOUL logger behavior."""
    
    def __init__(
        self,
        soul_file_path: str = "./SOUL.md",
        max_entries_in_memory: int = 10000,
        flush_interval_s: int = 5,
        auto_flush_on_critical: bool = True,
        include_debug_events: bool = False,
        alert_levels: Optional[List[LogLevel]] = None,
        tag_prefixes: Optional[Dict[EventType, List[str]]] = None,
    ) -> None:
        self.soul_file_path = soul_file_path
        self.max_entries_in_memory = max_entries_in_memory
        self.flush_interval_s = flush_interval_s
        self.auto_flush_on_critical = auto_flush_on_critical
        self.include_debug_events = include_debug_events
        self.alert_levels = alert_levels or [LogLevel.ERROR, LogLevel.CRITICAL, LogLevel.ALERT]
        self.tag_prefixes = tag_prefixes or self._default_tag_prefixes()
        
    def _default_tag_prefixes(self) -> Dict[EventType, List[str]]:
        """Default tag prefixes for event types."""
        return {
            EventType.TRAILING_STOP_SAVED: ["risk_management", "profit_protection"],
            EventType.FATAL_DESYNC: ["critical", "reconciliation", "auto_flatten"],
            EventType.SHADOW_MODEL_DISCOVERY: ["alpha", "regime_change", "profitable"],
            EventType.DRIFT_DETECTED: ["reconciliation", "drift"],
            EventType.ORDER_REJECTED: ["order_failure"],
            EventType.SNIPER_EXECUTION: ["execution", "sniper"],
        }


class OmsSoulLogger:
    """
    Main SOUL logger for OMS events.
    
    This logger writes all critical OMS events to SOUL.md in a structured
    Markdown format that can be easily reviewed and searched.
    """
    
    def __init__(self, config: Optional[SoulLoggerConfig] = None) -> None:
        self.config = config or SoulLoggerConfig()
        
        self._lock = threading.RLock()
        self._entries: List[SoulEntry] = []
        self._entry_counter: int = 0
        self._last_flush_time: float = time.time()
        self._running: bool = False
        self._flush_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._total_events: int = 0
        self._events_by_type: Dict[EventType, int] = {}
        self._events_by_level: Dict[LogLevel, int] = {}
        self._total_pnl_impact: float = 0.0
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md file with header if it doesn't exist."""
        path = Path(self.config.soul_file_path)
        
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            
            header = """# ZAID Trading Bot - SOUL.md

**System Operational Understanding Log**

This file contains the complete audit trail of OMS events, including:
- Order lifecycle events
- Synthetic order executions
- Reconciliation drifts and corrections
- Risk management actions (trailing stops, position flattens)
- Shadow model discoveries
- System alerts and performance metrics

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Entries | 0 |
| Critical Events | 0 |
| PnL Impact | 0.00 USDT |
| Last Update | N/A |

---

## Event Log

"""
            with open(path, 'w') as f:
                f.write(header)
    
    def start(self) -> None:
        """Start the background flush thread."""
        if self._running:
            return
        
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def stop(self) -> None:
        """Stop the logger and flush remaining entries."""
        self._running = False
        
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        
        # Final flush
        self._flush_to_file()
    
    def _flush_loop(self) -> None:
        """Background loop to periodically flush entries."""
        while self._running:
            time.sleep(self.config.flush_interval_s)
            self._flush_to_file()
    
    def log_event(
        self,
        event_type: EventType,
        message: str,
        level: LogLevel = LogLevel.INFO,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        pnl_impact: Optional[float] = None,
        latency_ns: Optional[int] = None,
    ) -> SoulEntry:
        """
        Log an OMS event to SOUL.md.
        
        Args:
            event_type: Type of event
            message: Human-readable message
            level: Severity level
            symbol: Related trading symbol
            order_id: Related order ID
            details: Additional structured data
            pnl_impact: PnL impact in USDT
            latency_ns: Event latency in nanoseconds
            
        Returns:
            The created SoulEntry
        """
        now_ms = int(time.time() * 1000)
        
        event = OmsEvent(
            event_type=event_type,
            timestamp_ms=now_ms,
            level=level,
            symbol=symbol,
            order_id=order_id,
            message=message,
            details=details or {},
            pnl_impact=pnl_impact,
            latency_ns=latency_ns,
        )
        
        with self._lock:
            self._entry_counter += 1
            self._total_events += 1
            
            # Update statistics
            self._events_by_type[event_type] = self._events_by_type.get(event_type, 0) + 1
            self._events_by_level[level] = self._events_by_level.get(level, 0) + 1
            
            if pnl_impact is not None:
                self._total_pnl_impact += pnl_impact
            
            # Create entry
            tags = self.config.tag_prefixes.get(event_type, [])
            entry = SoulEntry(
                entry_id=self._entry_counter,
                event=event,
                tags=tags,
            )
            
            self._entries.append(entry)
            
            # Trim memory if needed
            if len(self._entries) > self.config.max_entries_in_memory:
                self._entries = self._entries[-self.config.max_entries_in_memory:]
        
        # Auto-flush on critical events
        if level in self.config.alert_levels and self.config.auto_flush_on_critical:
            self._flush_to_file()
        
        return entry
    
    def _flush_to_file(self) -> None:
        """Flush pending entries to SOUL.md file."""
        with self._lock:
            if not self._entries:
                return
            
            # Get entries to flush (all unflushed)
            entries_to_flush = self._entries.copy()
        
        if not entries_to_flush:
            return
        
        # Sort by entry_id to ensure order
        entries_to_flush.sort(key=lambda e: e.entry_id)
        
        # Append to file
        path = Path(self.config.soul_file_path)
        
        try:
            with open(path, 'a') as f:
                for entry in entries_to_flush:
                    f.write(entry.to_markdown())
            
            self._last_flush_time = time.time()
            
            # Clear flushed entries (keep only recent in memory)
            with self._lock:
                if entries_to_flush:
                    last_id = entries_to_flush[-1].entry_id
                    self._entries = [e for e in self._entries if e.entry_id > last_id]
        
        except Exception as e:
            # Log error but don't crash
            print(f"Error flushing to SOUL.md: {e}")
    
    def update_summary(self) -> None:
        """Update the summary statistics at the top of SOUL.md."""
        path = Path(self.config.soul_file_path)
        
        if not path.exists():
            return
        
        with self._lock:
            critical_count = self._events_by_level.get(LogLevel.CRITICAL, 0)
            critical_count += self._events_by_level.get(LogLevel.ALERT, 0)
            
            summary = f"""## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Entries | {self._entry_counter} |
| Critical Events | {critical_count} |
| PnL Impact | {self._total_pnl_impact:.2f} USDT |
| Last Update | {datetime.now().isoformat()} |

---

## Event Type Breakdown

| Event Type | Count |
|------------|-------|
"""
            for event_type, count in sorted(
                self._events_by_type.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]:  # Top 20
                summary += f"| {event_type.value} | {count} |\n"
            
            summary += "\n---\n\n## Event Log\n\n"
        
        # Read existing file and replace summary section
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            # Find and replace summary section
            if "## Summary Statistics" in content:
                start = content.index("## Summary Statistics")
                end = content.index("## Event Log", start)
                content = content[:start] + summary + content[end:]
            
            with open(path, 'w') as f:
                f.write(content)
        
        except Exception as e:
            print(f"Error updating summary: {e}")
    
    def acknowledge_entry(self, entry_id: int) -> bool:
        """Mark an entry as acknowledged."""
        with self._lock:
            for entry in self._entries:
                if entry.entry_id == entry_id:
                    entry.acknowledged = True
                    return True
            return False
    
    def resolve_entry(
        self,
        entry_id: int,
        notes: str = ""
    ) -> bool:
        """Mark an entry as resolved with optional notes."""
        with self._lock:
            for entry in self._entries:
                if entry.entry_id == entry_id:
                    entry.resolved = True
                    entry.resolution_notes = notes
                    return True
            return False
    
    def get_unresolved_alerts(self) -> List[SoulEntry]:
        """Get all unresolved alert-level entries."""
        with self._lock:
            return [
                e for e in self._entries
                if e.event.level in self.config.alert_levels
                and not e.resolved
            ]
    
    def get_statistics(self) -> Dict:
        """Get logger statistics."""
        with self._lock:
            return {
                'total_events': self._total_events,
                'total_entries': self._entry_counter,
                'entries_in_memory': len(self._entries),
                'events_by_type': {k.value: v for k, v in self._events_by_type.items()},
                'events_by_level': {k.value: v for k, v in self._events_by_level.items()},
                'total_pnl_impact': self._total_pnl_impact,
                'last_flush_time': self._last_flush_time,
                'running': self._running,
            }
    
    # Convenience methods for common event types
    
    def log_trailing_stop_save(
        self,
        symbol: str,
        saved_pnl: float,
        details: Optional[Dict] = None,
    ) -> SoulEntry:
        """Log a trailing stop that saved a trade from reversal."""
        return self.log_event(
            event_type=EventType.TRAILING_STOP_SAVED,
            message=f"Trailing stop saved {saved_pnl:.2f} USDT on {symbol} from sudden reversal",
            level=LogLevel.INFO,
            symbol=symbol,
            pnl_impact=saved_pnl,
            details=details,
        )
    
    def log_fatal_desync(
        self,
        description: str,
        positions_flattened: int,
        details: Optional[Dict] = None,
    ) -> SoulEntry:
        """Log a fatal reconciliation desync."""
        return self.log_event(
            event_type=EventType.FATAL_DESYNC,
            message=f"Fatal desync detected: {description}. Flattened {positions_flattened} positions.",
            level=LogLevel.CRITICAL,
            pnl_impact=None,
            details={
                'positions_flattened': positions_flattened,
                **(details or {}),
            },
        )
    
    def log_shadow_discovery(
        self,
        model_name: str,
        regime_description: str,
        estimated_alpha: float,
        details: Optional[Dict] = None,
    ) -> SoulEntry:
        """Log a shadow model discovery of new profitable regime."""
        return self.log_event(
            event_type=EventType.SHADOW_MODEL_DISCOVERY,
            message=f"Shadow model '{model_name}' discovered new regime: {regime_description}. Est. alpha: {estimated_alpha:.4f}",
            level=LogLevel.ALERT,
            pnl_impact=None,
            details={
                'model_name': model_name,
                'regime': regime_description,
                'estimated_alpha': estimated_alpha,
                **(details or {}),
            },
        )
    
    def log_drift_detected(
        self,
        drift_type: str,
        severity: str,
        details: Optional[Dict] = None,
    ) -> SoulEntry:
        """Log a reconciliation drift detection."""
        return self.log_event(
            event_type=EventType.DRIFT_DETECTED,
            message=f"Reconciliation drift detected: {drift_type} (Severity: {severity})",
            level=LogLevel.WARNING if severity == "HIGH" else LogLevel.INFO,
            details={
                'drift_type': drift_type,
                'severity': severity,
                **(details or {}),
            },
        )
    
    def log_synthetic_order_failure(
        self,
        order_type: str,
        symbol: str,
        failure_reason: str,
        details: Optional[Dict] = None,
    ) -> SoulEntry:
        """Log a synthetic order failure."""
        return self.log_event(
            event_type=EventType.ORDER_REJECTED,
            message=f"Synthetic {order_type} order failed on {symbol}: {failure_reason}",
            level=LogLevel.ERROR,
            symbol=symbol,
            details={
                'order_type': order_type,
                'failure_reason': failure_reason,
                **(details or {}),
            },
        )


# Example usage
if __name__ == "__main__":
    # Initialize logger
    config = SoulLoggerConfig(
        soul_file_path="./SOUL.md",
        auto_flush_on_critical=True,
    )
    
    logger = OmsSoulLogger(config)
    logger.start()
    
    # Log various events
    print("Logging OMS events to SOUL.md...")
    
    # Trailing stop save
    logger.log_trailing_stop_save(
        symbol="BTCUSDT",
        saved_pnl=1250.50,
        details={'trigger_price': 51200.0, 'exit_price': 51150.0},
    )
    
    # Shadow model discovery
    logger.log_shadow_discovery(
        model_name="regime_detector_v3",
        regime_description="High volatility mean-reversion in Asian session",
        estimated_alpha=0.0023,
    )
    
    # Drift detection
    logger.log_drift_detected(
        drift_type="POSITION_MISMATCH",
        severity="HIGH",
        details={'oms_size': 1.5, 'exchange_size': 1.49},
    )
    
    # Synthetic order failure
    logger.log_synthetic_order_failure(
        order_type="BRACKET",
        symbol="ETHUSDT",
        failure_reason="Insufficient margin",
    )
    
    # Fatal desync
    logger.log_fatal_desync(
        description="Exchange reported different fill prices",
        positions_flattened=3,
    )
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"\nLogger Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Stop logger (flushes remaining)
    logger.stop()
    
    print(f"\nEvents logged to {config.soul_file_path}")
