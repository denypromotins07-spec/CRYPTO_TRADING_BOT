"""
CQRS Soul Logger - Logs state mismatches and rebuild times to SOUL.md.

This module provides comprehensive logging for the CQRS architecture,
tracking state consistency, rebuild performance, and any anomalies that
could indicate system issues. All logs are written to SOUL.md for audit
and debugging purposes.

Features:
- State mismatch detection and logging
- Rebuild time tracking with SLA monitoring
- Event chain integrity verification logging
- Performance metrics aggregation
- Alert generation for critical issues

Integrates with 152 domains including:
- System health monitoring
- Performance optimization
- Compliance reporting
- Incident response
"""

from __future__ import annotations
import os
import json
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from threading import RLock
from pathlib import Path


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Types of events logged by the soul logger."""
    STATE_MISMATCH = "state_mismatch"
    REBUILD_COMPLETE = "rebuild_complete"
    REBUILD_SLA_BREACH = "rebuild_sla_breach"
    EVENT_CHAIN_BROKEN = "event_chain_broken"
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_VERIFIED = "checkpoint_verified"
    SNAPSHOT_TRIGGERED = "snapshot_triggered"
    QUERY_PERFORMANCE = "query_performance"
    COMMAND_EXECUTED = "command_executed"
    SYSTEM_HEALTH = "system_health"


@dataclass
class LogEntry:
    """A single log entry in the soul log."""
    timestamp: datetime
    level: LogLevel
    event_type: EventType
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    correlation_id: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Format log entry as Markdown."""
        header = f"### [{self.timestamp.isoformat()}] {self.level.value}: {self.event_type.value}"
        
        content = [
            header,
            "",
            f"**Message:** {self.message}",
            f"**Source:** {self.source or 'system'}",
        ]
        
        if self.correlation_id:
            content.append(f"**Correlation ID:** {self.correlation_id}")
        
        if self.details:
            content.extend([
                "",
                "**Details:**",
                "```json",
                json.dumps(self.details, indent=2, default=str),
                "```",
            ])
        
        content.extend(["", "---", ""])
        return "\n".join(content)


@dataclass
class StateMismatch:
    """Represents a detected state mismatch."""
    component_a: str
    component_b: str
    expected_value: Any
    actual_value: Any
    sequence_number: int
    severity: str = "high"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_a": self.component_a,
            "component_b": self.component_b,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "sequence_number": self.sequence_number,
            "severity": self.severity,
        }


@dataclass
class RebuildMetrics:
    """Metrics from a state rebuild operation."""
    start_sequence: int
    end_sequence: int
    events_processed: int
    rebuild_time_ms: float
    sla_target_ms: float = 500.0
    gaps_detected: int = 0
    
    @property
    def met_sla(self) -> bool:
        return self.rebuild_time_ms <= self.sla_target_ms
    
    @property
    def throughput_events_per_sec(self) -> float:
        if self.rebuild_time_ms <= 0:
            return 0.0
        return (self.events_processed / self.rebuild_time_ms) * 1000.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "events_processed": self.events_processed,
            "rebuild_time_ms": round(self.rebuild_time_ms, 3),
            "sla_target_ms": self.sla_target_ms,
            "met_sla": self.met_sla,
            "throughput_events_per_sec": round(self.throughput_events_per_sec, 2),
            "gaps_detected": self.gaps_detected,
        }


class CQRSSoulLogger:
    """
    Central logging system for CQRS architecture.
    
    Tracks all state-related events, mismatches, and performance metrics,
    writing them to SOUL.md for audit and debugging.
    """
    
    def __init__(
        self,
        soul_path: str = "./SOUL.md",
        max_log_entries: int = 10000,
        enable_console_output: bool = True,
    ):
        self.soul_path = Path(soul_path)
        self.max_log_entries = max_log_entries
        self.enable_console_output = enable_console_output
        
        self._lock = RLock()
        self._entries: List[LogEntry] = []
        self._mismatches: List[StateMismatch] = []
        self._rebuild_history: List[RebuildMetrics] = []
        
        # Metrics
        self._total_logs = 0
        self._error_count = 0
        self._critical_count = 0
        
        # Initialize SOUL.md
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize or verify the SOUL.md file."""
        if not self.soul_path.exists():
            self._write_header()
    
    def _write_header(self) -> None:
        """Write the SOUL.md header."""
        header = """# ZAID Trading Bot - SOUL (System Observation & Utility Log)

## CQRS Architecture State Log

This file contains the complete audit trail of state management operations,
including rebuilds, snapshots, mismatches, and performance metrics.

---

## System Information

- **Bot Name:** ZAID Personal Crypto Trading Bot
- **Architecture:** CQRS with Event Sourcing
- **Target Rebuild SLA:** < 500ms
- **Trading Assets:** BTC, SOL, ETH, USDT
- **Hardware:** AMD Ryzen AI 5 (8GB RAM)

---

## Log Entries

"""
        with open(self.soul_path, 'w') as f:
            f.write(header)
    
    def log(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "",
        correlation_id: Optional[str] = None,
    ) -> LogEntry:
        """
        Create and store a log entry.
        
        Args:
            level: Log severity level
            event_type: Type of event
            message: Human-readable message
            details: Additional structured data
            source: Component that generated the log
            correlation_id: ID for tracing related events
            
        Returns:
            The created LogEntry
        """
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            event_type=event_type,
            message=message,
            details=details or {},
            source=source,
            correlation_id=correlation_id,
        )
        
        with self._lock:
            self._entries.append(entry)
            self._total_logs += 1
            
            if level == LogLevel.ERROR:
                self._error_count += 1
            elif level == LogLevel.CRITICAL:
                self._critical_count += 1
            
            # Trim old entries if over limit
            if len(self._entries) > self.max_log_entries:
                self._entries = self._entries[-self.max_log_entries:]
        
        # Write to file
        self._append_to_soul(entry)
        
        # Console output
        if self.enable_console_output:
            self._print_to_console(entry)
        
        return entry
    
    def _append_to_soul(self, entry: LogEntry) -> None:
        """Append a log entry to SOUL.md."""
        with self._lock:
            with open(self.soul_path, 'a') as f:
                f.write(entry.to_markdown())
    
    def _print_to_console(self, entry: LogEntry) -> None:
        """Print log entry to console."""
        color_codes = {
            LogLevel.DEBUG: "\033[36m",      # Cyan
            LogLevel.INFO: "\033[32m",       # Green
            LogLevel.WARNING: "\033[33m",    # Yellow
            LogLevel.ERROR: "\033[31m",      # Red
            LogLevel.CRITICAL: "\033[35m",   # Magenta
        }
        reset = "\033[0m"
        
        color = color_codes.get(entry.level, "")
        print(f"{color}[{entry.timestamp.strftime('%H:%M:%S.%f')}] "
              f"[{entry.level.value}] [{entry.event_type.value}] "
              f"{entry.message}{reset}")
    
    def log_state_mismatch(
        self,
        mismatch: StateMismatch,
        context: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """
        Log a state mismatch between components.
        
        Args:
            mismatch: StateMismatch object with details
            context: Additional context about the mismatch
            
        Returns:
            The created LogEntry
        """
        with self._lock:
            self._mismatches.append(mismatch)
        
        return self.log(
            level=LogLevel.CRITICAL,
            event_type=EventType.STATE_MISMATCH,
            message=f"State mismatch detected between {mismatch.component_a} and {mismatch.component_b}",
            details={
                "mismatch": mismatch.to_dict(),
                "context": context or {},
            },
            source="state_validator",
        )
    
    def log_rebuild_complete(self, metrics: RebuildMetrics) -> LogEntry:
        """
        Log completion of a state rebuild operation.
        
        Args:
            metrics: RebuildMetrics with performance data
            
        Returns:
            The created LogEntry
        """
        with self._lock:
            self._rebuild_history.append(metrics)
        
        level = LogLevel.INFO if metrics.met_sla else LogLevel.WARNING
        
        # Log SLA breach separately if applicable
        if not metrics.met_sla:
            self.log(
                level=LogLevel.WARNING,
                event_type=EventType.REBUILD_SLA_BREACH,
                message=f"Rebuild exceeded SLA target: {metrics.rebuild_time_ms:.2f}ms > {metrics.sla_target_ms}ms",
                details=metrics.to_dict(),
                source="state_rebuilder",
            )
        
        return self.log(
            level=level,
            event_type=EventType.REBUILD_COMPLETE,
            message=f"State rebuild completed: {metrics.events_processed} events in {metrics.rebuild_time_ms:.2f}ms",
            details=metrics.to_dict(),
            source="state_rebuilder",
        )
    
    def log_event_chain_broken(
        self,
        sequence: int,
        expected_hash: str,
        actual_hash: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """
        Log a broken event chain (hash mismatch).
        
        Args:
            sequence: Sequence number where break was detected
            expected_hash: Expected hash value
            actual_hash: Actual hash value found
            context: Additional context
            
        Returns:
            The created LogEntry
        """
        return self.log(
            level=LogLevel.CRITICAL,
            event_type=EventType.EVENT_CHAIN_BROKEN,
            message=f"Event chain integrity violated at sequence {sequence}",
            details={
                "sequence": sequence,
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
                "context": context or {},
            },
            source="audit_trail",
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about logged events."""
        with self._lock:
            # Count by event type
            type_counts: Dict[str, int] = {}
            for entry in self._entries:
                key = entry.event_type.value
                type_counts[key] = type_counts.get(key, 0) + 1
            
            # Calculate average rebuild time
            avg_rebuild_time = 0.0
            if self._rebuild_history:
                avg_rebuild_time = sum(
                    m.rebuild_time_ms for m in self._rebuild_history
                ) / len(self._rebuild_history)
            
            # SLA compliance rate
            sla_compliance = 1.0
            if self._rebuild_history:
                compliant = sum(1 for m in self._rebuild_history if m.met_sla)
                sla_compliance = compliant / len(self._rebuild_history)
            
            return {
                "total_logs": self._total_logs,
                "error_count": self._error_count,
                "critical_count": self._critical_count,
                "mismatch_count": len(self._mismatches),
                "rebuild_count": len(self._rebuild_history),
                "average_rebuild_time_ms": round(avg_rebuild_time, 3),
                "sla_compliance_rate": round(sla_compliance, 4),
                "logs_by_type": type_counts,
                "soul_file_path": str(self.soul_path),
            }
    
    def get_recent_mismatches(self, limit: int = 10) -> List[StateMismatch]:
        """Get recent state mismatches."""
        with self._lock:
            return self._mismatches[-limit:]
    
    def get_rebuild_history(self, limit: int = 100) -> List[RebuildMetrics]:
        """Get rebuild history."""
        with self._lock:
            return self._rebuild_history[-limit:]
    
    def export_report(self, output_path: str) -> None:
        """Export a comprehensive report to a JSON file."""
        with self._lock:
            report = {
                "generated_at": datetime.utcnow().isoformat(),
                "summary": self.get_summary(),
                "recent_mismatches": [m.to_dict() for m in self._mismatches[-50:]],
                "rebuild_history": [m.to_dict() for m in self._rebuild_history[-100:]],
                "recent_errors": [
                    entry.to_dict() if hasattr(entry, 'to_dict') else {
                        "timestamp": entry.timestamp.isoformat(),
                        "level": entry.level.value,
                        "event_type": entry.event_type.value,
                        "message": entry.message,
                    }
                    for entry in self._entries[-100:]
                    if entry.level in (LogLevel.ERROR, LogLevel.CRITICAL)
                ],
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def clear(self) -> None:
        """Clear all logged data (use with caution)."""
        with self._lock:
            self._entries.clear()
            self._mismatches.clear()
            self._rebuild_history.clear()
            self._total_logs = 0
            self._error_count = 0
            self._critical_count = 0
        
        # Reinitialize the soul file
        self._write_header()


# Global instance for convenience
_soul_logger: Optional[CQRSSoulLogger] = None


def get_soul_logger() -> CQRSSoulLogger:
    """Get or create the global soul logger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = CQRSSoulLogger()
    return _soul_logger


def log_state_mismatch(
    component_a: str,
    component_b: str,
    expected: Any,
    actual: Any,
    sequence: int,
) -> LogEntry:
    """Convenience function to log a state mismatch."""
    mismatch = StateMismatch(
        component_a=component_a,
        component_b=component_b,
        expected_value=expected,
        actual_value=actual,
        sequence_number=sequence,
    )
    return get_soul_logger().log_state_mismatch(mismatch)


def log_rebuild_complete(
    events_processed: int,
    rebuild_time_ms: float,
    start_sequence: int,
    end_sequence: int,
    gaps_detected: int = 0,
) -> LogEntry:
    """Convenience function to log rebuild completion."""
    metrics = RebuildMetrics(
        start_sequence=start_sequence,
        end_sequence=end_sequence,
        events_processed=events_processed,
        rebuild_time_ms=rebuild_time_ms,
        gaps_detected=gaps_detected,
    )
    return get_soul_logger().log_rebuild_complete(metrics)


if __name__ == "__main__":
    # Demo usage
    logger = CQRSSoulLogger()
    
    # Log various events
    logger.log(
        level=LogLevel.INFO,
        event_type=EventType.SYSTEM_HEALTH,
        message="System startup complete",
        details={"version": "1.0.0", "mode": "production"},
    )
    
    # Log a state mismatch
    log_state_mismatch(
        component_a="position_aggregator",
        component_b="read_model",
        expected=100.0,
        actual=99.5,
        sequence=12345,
    )
    
    # Log rebuild completion
    log_rebuild_complete(
        events_processed=50000,
        rebuild_time_ms=250.5,
        start_sequence=1,
        end_sequence=50000,
    )
    
    # Get summary
    summary = logger.get_summary()
    print(f"\nLogger Summary: {json.dumps(summary, indent=2)}")
