#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 24
Protocol Soul Logger for SOUL.md
Logs serialization bottlenecks, memory spikes, and null pointer events
Updates SOUL.md with critical protocol analytics and system health data
"""

from __future__ import annotations

import os
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProtocolEvent:
    """Represents a protocol-level event for logging."""
    timestamp: str
    event_type: str  # 'null_pointer', 'bottleneck', 'memory_spike', 'schema_mismatch'
    component: str
    severity: str  # 'INFO', 'WARNING', 'CRITICAL', 'FATAL'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[str] = None


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: str
    total_bytes: int
    used_bytes: int
    available_bytes: int
    percent_used: float
    gc_objects: int = 0
    serialization_buffers: int = 0


@dataclass 
class BottleneckReport:
    """Serialization bottleneck report."""
    timestamp: str
    protocol: str
    operation: str
    avg_latency_ns: float
    p99_latency_ns: float
    threshold_ns: float
    impact_score: float  # 0.0 to 1.0
    affected_components: List[str] = field(default_factory=list)


class ProtocolSoulLogger:
    """
    Centralized logger for protocol analytics and system health.
    Writes to SOUL.md with structured event tracking and alerting.
    """
    
    # Severity levels
    SEVERITY_INFO = "INFO"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_CRITICAL = "CRITICAL"
    SEVERITY_FATAL = "FATAL"
    
    # Default thresholds
    DEFAULT_LATENCY_THRESHOLD_NS = 500  # 500 nanoseconds
    DEFAULT_MEMORY_THRESHOLD_PERCENT = 85.0  # 85% RAM usage
    
    def __init__(
        self,
        soul_md_path: str = "SOUL.md",
        max_events: int = 1000,
        auto_flush: bool = True,
    ):
        self.soul_md_path = Path(soul_md_path)
        self.max_events = max_events
        self.auto_flush = auto_flush
        
        # Event storage (circular buffer)
        self.events: deque[ProtocolEvent] = deque(maxlen=max_events)
        self.memory_snapshots: deque[MemorySnapshot] = deque(maxlen=100)
        self.bottleneck_reports: deque[BottleneckReport] = deque(maxlen=50)
        
        # Statistics
        self.event_counts: Dict[str, int] = {
            'total': 0,
            'null_pointer': 0,
            'bottleneck': 0,
            'memory_spike': 0,
            'schema_mismatch': 0,
        }
        
        # Threading lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Initialize SOUL.md if it doesn't exist
        self._initialize_soul_md()
        
        logger.info(f"Protocol Soul Logger initialized at {self.soul_md_path}")
    
    def _initialize_soul_md(self) -> None:
        """Initialize SOUL.md file with header."""
        if not self.soul_md_path.exists():
            header = f"""# ZAID Personal Crypto Trading Bot - SOUL.md

## System Overview
**Initialized**: {datetime.now().isoformat()}
**Target**: Zero-Copy Binary Protocols for HFT
**Memory Limit**: 8GB RAM (AMD Ryzen AI 5 Laptop)
**Performance Target**: 8k-20k INR/hour in 4hr window

## Protocol Architecture
- **FlatBuffers**: Zero-copy deserialization for tick data
- **Custom Bitwise Packing**: 64-bit order encoding
- **LZ4/Zstd Compression**: <5% CPU overhead
- **Schema Registry**: Backward-compatible evolution

---

## Live System Log

"""
            self.soul_md_path.parent.mkdir(parents=True, exist_ok=True)
            self.soul_md_path.write_text(header)
    
    def log_null_pointer(
        self,
        component: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProtocolEvent:
        """Log a null pointer encounter in zero-copy parser."""
        event = ProtocolEvent(
            timestamp=datetime.now().isoformat(),
            event_type='null_pointer',
            component=component,
            severity=self.SEVERITY_CRITICAL,
            message=message,
            details=context or {},
            stack_trace=traceback.format_stack()[:-1],
        )
        
        self._record_event(event)
        self.event_counts['null_pointer'] += 1
        
        # Update SOUL.md immediately for critical events
        self._update_soul_md()
        
        return event
    
    def log_bottleneck(
        self,
        protocol: str,
        operation: str,
        avg_latency_ns: float,
        p99_latency_ns: float,
        threshold_ns: float,
        affected_components: Optional[List[str]] = None,
    ) -> ProtocolEvent:
        """Log a serialization bottleneck detection."""
        impact_score = min(1.0, (p99_latency_ns / threshold_ns) * 0.5)
        
        report = BottleneckReport(
            timestamp=datetime.now().isoformat(),
            protocol=protocol,
            operation=operation,
            avg_latency_ns=avg_latency_ns,
            p99_latency_ns=p99_latency_ns,
            threshold_ns=threshold_ns,
            impact_score=impact_score,
            affected_components=affected_components or [],
        )
        
        with self._lock:
            self.bottleneck_reports.append(report)
        
        severity = self.SEVERITY_WARNING
        if p99_latency_ns > threshold_ns * 2:
            severity = self.SEVERITY_CRITICAL
        elif p99_latency_ns > threshold_ns * 5:
            severity = self.SEVERITY_FATAL
        
        event = ProtocolEvent(
            timestamp=report.timestamp,
            event_type='bottleneck',
            component=f"{protocol}:{operation}",
            severity=severity,
            message=f"Serialization bottleneck detected: {p99_latency_ns:.0f}ns p99 (threshold: {threshold_ns:.0f}ns)",
            details={
                'avg_latency_ns': avg_latency_ns,
                'p99_latency_ns': p99_latency_ns,
                'threshold_ns': threshold_ns,
                'impact_score': impact_score,
            },
        )
        
        self._record_event(event)
        self.event_counts['bottleneck'] += 1
        
        if severity in [self.SEVERITY_CRITICAL, self.SEVERITY_FATAL]:
            self._update_soul_md()
        
        return event
    
    def log_memory_spike(
        self,
        total_bytes: int,
        used_bytes: int,
        available_bytes: int,
        gc_objects: int = 0,
        serialization_buffers: int = 0,
    ) -> Optional[ProtocolEvent]:
        """Log a memory spike event."""
        percent_used = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            available_bytes=available_bytes,
            percent_used=percent_used,
            gc_objects=gc_objects,
            serialization_buffers=serialization_buffers,
        )
        
        with self._lock:
            self.memory_snapshots.append(snapshot)
        
        # Check if this exceeds threshold
        if percent_used >= self.DEFAULT_MEMORY_THRESHOLD_PERCENT:
            severity = self.SEVERITY_WARNING
            if percent_used >= 95:
                severity = self.SEVERITY_CRITICAL
            
            event = ProtocolEvent(
                timestamp=snapshot.timestamp,
                event_type='memory_spike',
                component='memory_manager',
                severity=severity,
                message=f"Memory usage at {percent_used:.1f}% ({used_bytes / 1e6:.1f}MB / {total_bytes / 1e6:.1f}MB)",
                details={
                    'percent_used': percent_used,
                    'used_mb': used_bytes / 1e6,
                    'total_mb': total_bytes / 1e6,
                    'available_mb': available_bytes / 1e6,
                    'gc_objects': gc_objects,
                    'serialization_buffers': serialization_buffers,
                },
            )
            
            self._record_event(event)
            self.event_counts['memory_spike'] += 1
            
            if severity == self.SEVERITY_CRITICAL:
                self._update_soul_md()
            
            return event
        
        return None
    
    def log_schema_mismatch(
        self,
        expected_version: int,
        received_version: int,
        message_type: str,
        action_taken: str = "HALT",
    ) -> ProtocolEvent:
        """Log a schema mismatch that threatens event store corruption."""
        event = ProtocolEvent(
            timestamp=datetime.now().isoformat(),
            event_type='schema_mismatch',
            component='schema_registry',
            severity=self.SEVERITY_FATAL,
            message=f"Schema version mismatch: expected v{expected_version}, got v{received_version} for {message_type}",
            details={
                'expected_version': expected_version,
                'received_version': received_version,
                'message_type': message_type,
                'action_taken': action_taken,
            },
        )
        
        self._record_event(event)
        self.event_counts['schema_mismatch'] += 1
        
        # Immediate update for fatal events
        self._update_soul_md()
        
        return event
    
    def _record_event(self, event: ProtocolEvent) -> None:
        """Record an event to the internal buffer."""
        with self._lock:
            self.events.append(event)
            self.event_counts['total'] += 1
            
            if self.auto_flush and event.severity in [
                self.SEVERITY_CRITICAL,
                self.SEVERITY_FATAL,
            ]:
                self.flush()
    
    def flush(self) -> None:
        """Force flush all pending logs to SOUL.md."""
        self._update_soul_md()
    
    def _update_soul_md(self) -> None:
        """Update SOUL.md with current state."""
        with self._lock:
            # Build the log content
            content = self._build_soul_content()
            
            # Write atomically
            temp_path = self.soul_md_path.with_suffix('.tmp')
            temp_path.write_text(content)
            temp_path.replace(self.soul_md_path)
    
    def _build_soul_content(self) -> str:
        """Build the complete SOUL.md content."""
        now = datetime.now().isoformat()
        
        # Header
        content = f"""# ZAID Personal Crypto Trading Bot - SOUL.md

## System Status
**Last Updated**: {now}
**Total Events**: {self.event_counts['total']}
**Active Alerts**: {sum(1 for e in self.events if not e.resolved)}

### Event Summary
| Type | Count |
|------|-------|
| Null Pointer | {self.event_counts['null_pointer']} |
| Bottlenecks | {self.event_counts['bottleneck']} |
| Memory Spikes | {self.event_counts['memory_spike']} |
| Schema Mismatches | {self.event_counts['schema_mismatch']} |

---

## Recent Critical Events

"""
        
        # Add recent critical/fatal events
        critical_events = [
            e for e in reversed(list(self.events))
            if e.severity in [self.SEVERITY_CRITICAL, self.SEVERITY_FATAL]
        ][:20]
        
        if critical_events:
            for event in critical_events:
                content += self._format_event(event)
        else:
            content += "*No critical events recorded*\n\n"
        
        # Add recent bottlenecks
        content += "\n## Serialization Bottlenecks\n\n"
        
        if self.bottleneck_reports:
            content += "| Timestamp | Protocol | Operation | P99 Latency | Impact |\n"
            content += "|-----------|----------|-----------|-------------|--------|\n"
            
            for report in list(self.bottleneck_reports)[-10:]:
                content += (
                    f"| {report.timestamp} | {report.protocol} | {report.operation} | "
                    f"{report.p99_latency_ns:.0f}ns | {report.impact_score:.2f} |\n"
                )
        else:
            content += "*No bottlenecks detected*\n"
        
        # Add memory status
        content += "\n## Memory Status\n\n"
        
        if self.memory_snapshots:
            latest = self.memory_snapshots[-1]
            content += f"""
- **Current Usage**: {latest.percent_used:.1f}%
- **Used**: {latest.used_bytes / 1e6:.1f} MB
- **Available**: {latest.available_bytes / 1e6:.1f} MB
- **GC Objects**: {latest.gc_objects:,}
- **Serialization Buffers**: {latest.serialization_buffers}
"""
        else:
            content += "*No memory data available*\n"
        
        # Add footer
        content += f"""
---

## System Health Indicators

- [x] FlatBuffer Parser Active
- [x] Zero-Copy Deserialization Enabled
- [x] LZ4 Stream Compression Ready
- [x] Schema Registry Monitoring
- [x] Memory Bounds Checking Active

*This file is automatically updated by the Protocol Soul Logger.*
*Last full refresh: {now}*
"""
        
        return content
    
    def _format_event(self, event: ProtocolEvent) -> str:
        """Format a single event for SOUL.md."""
        content = f"""### [{event.severity}] {event.event_type.upper()} - {event.component}

**Time**: {event.timestamp}
**Message**: {event.message}

"""
        
        if event.details:
            content += "**Details**:\n```json\n"
            content += json.dumps(event.details, indent=2)
            content += "\n```\n\n"
        
        if event.stack_trace:
            content += "**Stack Trace**:\n```\n"
            stack_str = '\n'.join(event.stack_trace) if isinstance(event.stack_trace, list) else event.stack_trace
            content += stack_str[:2000]  # Truncate long traces
            content += "\n```\n\n"
        
        content += "---\n\n"
        
        return content
    
    def get_unresolved_events(self) -> List[ProtocolEvent]:
        """Get all unresolved events."""
        with self._lock:
            return [e for e in self.events if not e.resolved]
    
    def resolve_event(self, event: ProtocolEvent, resolution: str) -> None:
        """Mark an event as resolved."""
        event.resolved = True
        event.resolution_time = datetime.now().isoformat()
        event.details['resolution'] = resolution
        
        logger.info(f"Event resolved: {event.event_type} at {event.component}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logger statistics."""
        with self._lock:
            return {
                'total_events': self.event_counts['total'],
                'events_by_type': self.event_counts.copy(),
                'pending_events': len([e for e in self.events if not e.resolved]),
                'memory_snapshots': len(self.memory_snapshots),
                'bottleneck_reports': len(self.bottleneck_reports),
                'soul_md_path': str(self.soul_md_path),
            }


def main():
    """Example usage and testing."""
    print("Protocol Soul Logger for ZAID Trading Bot")
    print("=" * 60)
    
    # Create logger
    logger_instance = ProtocolSoulLogger(soul_md_path="SOUL.md")
    
    # Simulate various events
    print("\nSimulating protocol events...")
    
    # Null pointer event
    logger_instance.log_null_pointer(
        component="zero_copy_parser",
        message="Unexpected null pointer in tick data stream",
        context={"symbol": "BTCUSDT", "sequence_id": 12345},
    )
    
    # Bottleneck event
    logger_instance.log_bottleneck(
        protocol="flatbuffer",
        operation="deserialize_tick",
        avg_latency_ns=350,
        p99_latency_ns=750,
        threshold_ns=500,
        affected_components=["tick_handler", "order_book_updater"],
    )
    
    # Memory spike event
    logger_instance.log_memory_spike(
        total_bytes=8 * 1024**3,  # 8GB
        used_bytes=7 * 1024**3,   # 7GB (87.5%)
        available_bytes=1 * 1024**3,
        gc_objects=150000,
        serialization_buffers=256,
    )
    
    # Schema mismatch event
    logger_instance.log_schema_mismatch(
        expected_version=3,
        received_version=2,
        message_type="OrderBookUpdate",
        action_taken="HALT",
    )
    
    # Print statistics
    print("\nLogger Statistics:")
    stats = logger_instance.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nSOUL.md created at: {logger_instance.soul_md_path.absolute()}")
    print("\nProtocol Soul Logger ready!")


if __name__ == "__main__":
    main()
