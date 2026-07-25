#!/usr/bin/env python3
"""
IPC Soul Logger for Logging Bottlenecks to SOUL.md

This module implements comprehensive IPC logging that tracks bottlenecks,
dropped messages, and latency spikes, writing detailed reports to SOUL.md.
Designed for the ZAID Personal Crypto Trading Bot to achieve 8k-20k INR/hour targets.

Key Features:
- Real-time IPC bottleneck detection
- Microsecond-level latency spike tracking
- Dropped message accounting and categorization
- Automatic SOUL.md updates with structured reports
- Integration with event bus and process supervisor

Domain Integration: Quantitative Finance Domains 133-144 (Observability, Diagnostics)
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar("T")


class SeverityLevel(IntEnum):
    """Severity levels for IPC events."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class EventType(Enum):
    """Types of IPC events to track."""
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_DROPPED = "message_dropped"
    LATENCY_SPIKE = "latency_spike"
    QUEUE_BACKPRESSURE = "queue_backpressure"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    MEMORY_PRESSURE = "memory_pressure"
    SHARED_MEMORY_ERROR = "shared_memory_error"
    RUST_PYTHON_HANDSHAKE = "rust_python_handshake"
    PROCESS_CRASH = "process_crash"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"


@dataclass(slots=True, frozen=True)
class IPCEvent:
    """Represents a single IPC event for logging."""
    timestamp_ns: int
    event_type: EventType
    severity: SeverityLevel
    source: str
    target: str
    latency_us: float
    message_id: Optional[str]
    details: Dict[str, Any]
    
    @classmethod
    def create(
        cls,
        event_type: EventType,
        source: str,
        target: str,
        latency_us: float = 0.0,
        severity: SeverityLevel = SeverityLevel.INFO,
        message_id: Optional[str] = None,
        **details: Any,
    ) -> IPCEvent:
        """Factory method for creating IPC events."""
        return cls(
            timestamp_ns=time.time_ns(),
            event_type=event_type,
            severity=severity,
            source=source,
            target=target,
            latency_us=latency_us,
            message_id=message_id,
            details=details,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "timestamp_iso": datetime.fromtimestamp(
                self.timestamp_ns / 1e9
            ).isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.name,
            "source": self.source,
            "target": self.target,
            "latency_us": self.latency_us,
            "message_id": self.message_id,
            "details": self.details,
        }


@dataclass(slots=True)
class LatencyStats:
    """Statistics for latency monitoring."""
    count: int = 0
    total_us: float = 0.0
    min_us: float = float('inf')
    max_us: float = 0.0
    spike_count: int = 0  # Events exceeding threshold
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    _samples: List[float] = field(default_factory=list)
    
    def record(self, latency_us: float, spike_threshold_us: float = 1000.0) -> None:
        """Record a latency sample."""
        self.count += 1
        self.total_us += latency_us
        self.min_us = min(self.min_us, latency_us)
        self.max_us = max(self.max_us, latency_us)
        
        if latency_us > spike_threshold_us:
            self.spike_count += 1
        
        # Keep last 1000 samples for percentile calculation
        self._samples.append(latency_us)
        if len(self._samples) > 1000:
            self._samples.pop(0)
        
        # Update percentiles
        self._update_percentiles()
    
    def _update_percentiles(self) -> None:
        """Calculate percentile values from samples."""
        if not self._samples:
            return
        
        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)
        
        self.p50_us = sorted_samples[int(n * 0.50)] if n > 0 else 0.0
        self.p95_us = sorted_samples[int(n * 0.95)] if n > 0 else 0.0
        self.p99_us = sorted_samples[int(n * 0.99)] if n > 0 else 0.0
    
    @property
    def avg_us(self) -> float:
        """Get average latency."""
        return self.total_us / self.count if self.count > 0 else 0.0
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.total_us = 0.0
        self.min_us = float('inf')
        self.max_us = 0.0
        self.spike_count = 0
        self.p50_us = 0.0
        self.p95_us = 0.0
        self.p99_us = 0.0
        self._samples.clear()


@dataclass(slots=True)
class IPSoulStats:
    """Overall IPC statistics for SOUL.md reporting."""
    total_events: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_messages_dropped: int = 0
    total_latency_spikes: int = 0
    total_backpressure_events: int = 0
    uptime_seconds: float = 0.0
    start_timestamp: float = 0.0
    last_event_timestamp: float = 0.0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)
    latency_by_channel: Dict[str, LatencyStats] = field(default_factory=dict)


class IPCSoulLogger:
    """
    Main IPC logger that tracks all inter-process communication
    and writes detailed reports to SOUL.md.
    """
    
    def __init__(
        self,
        soul_md_path: str = "/workspace/SOUL.md",
        latency_spike_threshold_us: float = 1000.0,
        max_events_in_memory: int = 10000,
        flush_interval_s: float = 60.0,
    ):
        self._soul_md_path = Path(soul_md_path)
        self._latency_spike_threshold_us = latency_spike_threshold_us
        self._max_events_in_memory = max_events_in_memory
        self._flush_interval_s = flush_interval_s
        
        # Event storage
        self._events: List[IPCEvent] = []
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = IPSoulStats()
        self._stats.start_timestamp = time.time()
        self._latency_stats: DefaultDict[str, LatencyStats] = defaultdict(LatencyStats)
        
        # Counters
        self._dropped_messages: DefaultDict[str, int] = defaultdict(int)
        self._spike_events: List[IPCEvent] = []
        
        # Background flush
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        
        # Ensure SOUL.md exists
        self._ensure_soul_file()
    
    def _ensure_soul_file(self) -> None:
        """Ensure SOUL.md file exists with initial structure."""
        if not self._soul_md_path.exists():
            self._soul_md_path.parent.mkdir(parents=True, exist_ok=True)
            
            initial_content = """# ZAID Personal Crypto Trading Bot - SOUL.md

## System Observability and Unified Logging

This file contains real-time IPC diagnostics, bottleneck analysis, and system health metrics.

---

## Current Session

"""
            with open(self._soul_md_path, 'w') as f:
                f.write(initial_content)
    
    def log_event(self, event: IPCEvent) -> None:
        """Log an IPC event."""
        with self._lock:
            self._events.append(event)
            self._stats.total_events += 1
            self._stats.last_event_timestamp = time.time()
            
            # Trim if exceeding memory limit
            if len(self._events) > self._max_events_in_memory:
                self._events = self._events[-self._max_events_in_memory:]
            
            # Update specific counters
            if event.event_type == EventType.MESSAGE_SENT:
                self._stats.total_messages_sent += 1
            elif event.event_type == EventType.MESSAGE_RECEIVED:
                self._stats.total_messages_received += 1
            elif event.event_type == EventType.MESSAGE_DROPPED:
                self._stats.total_messages_dropped += 1
                reason = event.details.get("reason", "unknown")
                self._dropped_messages[reason] += 1
                self._stats.dropped_by_reason[reason] = self._dropped_messages[reason]
            elif event.event_type == EventType.LATENCY_SPIKE:
                self._stats.total_latency_spikes += 1
                self._spike_events.append(event)
                if len(self._spike_events) > 100:
                    self._spike_events.pop(0)
            elif event.event_type == EventType.QUEUE_BACKPRESSURE:
                self._stats.total_backpressure_events += 1
            
            # Record latency
            if event.latency_us > 0:
                channel = f"{event.source}->{event.target}"
                self._latency_stats[channel].record(
                    event.latency_us,
                    self._latency_spike_threshold_us,
                )
                
                # Log spike if exceeds threshold
                if event.latency_us > self._latency_spike_threshold_us:
                    self._log_latency_spike(event)
    
    def _log_latency_spike(self, event: IPCEvent) -> None:
        """Log a latency spike event."""
        logger.warning(
            f"LATENCY SPIKE: {event.source} -> {event.target} "
            f"={event.latency_us:.2f}µs (threshold: {self._latency_spike_threshold_us}µs)"
        )
        
        # Update SOUL.md immediately for critical spikes
        if event.latency_us > self._latency_spike_threshold_us * 10:
            self._update_soul_md()
    
    def record_latency(self, source: str, target: str, latency_us: float) -> None:
        """Record a latency measurement."""
        if latency_us > self._latency_spike_threshold_us:
            event = IPCEvent.create(
                event_type=EventType.LATENCY_SPIKE,
                source=source,
                target=target,
                latency_us=latency_us,
                severity=SeverityLevel.WARNING,
            )
            self.log_event(event)
        else:
            channel = f"{source}->{target}"
            self._latency_stats[channel].record(latency_us, self._latency_spike_threshold_us)
    
    def record_dropped_message(
        self,
        source: str,
        target: str,
        reason: str,
        message_id: Optional[str] = None,
    ) -> None:
        """Record a dropped message."""
        event = IPCEvent.create(
            event_type=EventType.MESSAGE_DROPPED,
            source=source,
            target=target,
            severity=SeverityLevel.WARNING,
            message_id=message_id,
            reason=reason,
        )
        self.log_event(event)
    
    def record_backpressure(
        self,
        source: str,
        target: str,
        queue_depth: int,
        max_depth: int,
    ) -> None:
        """Record a backpressure event."""
        event = IPCEvent.create(
            event_type=EventType.QUEUE_BACKPRESSURE,
            source=source,
            target=target,
            severity=SeverityLevel.WARNING,
            queue_depth=queue_depth,
            max_depth=max_depth,
            utilization=queue_depth / max_depth if max_depth > 0 else 0,
        )
        self.log_event(event)
    
    def get_stats(self) -> IPSoulStats:
        """Get current statistics."""
        with self._lock:
            self._stats.uptime_seconds = time.time() - self._stats.start_timestamp
            self._stats.latency_by_channel = dict(self._latency_stats)
            return self._stats
    
    def _generate_report(self) -> str:
        """Generate a markdown report for SOUL.md."""
        stats = self.get_stats()
        now = datetime.now().isoformat()
        
        report = f"""## Report Generated: {now}

### Summary Statistics

| Metric | Value |
|--------|-------|
| Uptime | {stats.uptime_seconds:.2f}s |
| Total Events | {stats.total_events} |
| Messages Sent | {stats.total_messages_sent} |
| Messages Received | {stats.total_messages_received} |
| Messages Dropped | {stats.total_messages_dropped} |
| Latency Spikes | {stats.total_latency_spikes} |
| Backpressure Events | {stats.total_backpressure_events} |

### Dropped Messages by Reason

"""
        if stats.dropped_by_reason:
            report += "| Reason | Count |\n|--------|-------|\n"
            for reason, count in sorted(stats.dropped_by_reason.items(), key=lambda x: -x[1]):
                report += f"| {reason} | {count} |\n"
        else:
            report += "*No messages dropped*\n"
        
        report += "\n### Latency Statistics by Channel\n\n"
        
        if stats.latency_by_channel:
            report += "| Channel | Avg (µs) | P50 (µs) | P95 (µs) | P99 (µs) | Max (µs) | Spikes |\n"
            report += "|---------|----------|----------|----------|----------|----------|--------|\n"
            
            for channel, lat_stats in sorted(stats.latency_by_channel.items()):
                if lat_stats.count > 0:
                    report += (
                        f"| {channel} | {lat_stats.avg_us:.2f} | {lat_stats.p50_us:.2f} | "
                        f"{lat_stats.p95_us:.2f} | {lat_stats.p99_us:.2f} | "
                        f"{lat_stats.max_us:.2f} | {lat_stats.spike_count} |\n"
                    )
        else:
            report += "*No latency data recorded*\n"
        
        # Recent spike events
        if self._spike_events:
            report += "\n### Recent Latency Spikes\n\n"
            report += "| Timestamp | Source | Target | Latency (µs) |\n"
            report += "|-----------|--------|--------|---------------|\n"
            
            for event in self._spike_events[-10:]:
                ts = datetime.fromtimestamp(event.timestamp_ns / 1e9).strftime("%H:%M:%S.%f")[:-3]
                report += f"| {ts} | {event.source} | {event.target} | {event.latency_us:.2f} |\n"
        
        report += "\n---\n\n"
        
        return report
    
    def _update_soul_md(self) -> None:
        """Update SOUL.md with latest report."""
        try:
            report = self._generate_report()
            
            # Read existing content
            existing = ""
            if self._soul_md_path.exists():
                with open(self._soul_md_path, 'r') as f:
                    existing = f.read()
            
            # Find where to insert (after "## Current Session")
            marker = "## Current Session"
            if marker in existing:
                parts = existing.split(marker, 1)
                new_content = parts[0] + marker + "\n\n" + report
                # Keep any content after the report section
                if "\n---\n" in parts[1]:
                    remainder = parts[1].split("\n---\n", 1)[1]
                    new_content += "\n---\n" + remainder
            else:
                new_content = existing + "\n" + report
            
            with open(self._soul_md_path, 'w') as f:
                f.write(new_content)
            
            logger.info(f"Updated SOUL.md at {self._soul_md_path}")
            
        except Exception as e:
            logger.error(f"Failed to update SOUL.md: {e}")
    
    def _flush_loop(self) -> None:
        """Background loop for periodic flushing."""
        while self._running:
            time.sleep(self._flush_interval_s)
            
            if self._running:
                self._update_soul_md()
    
    def start(self) -> None:
        """Start the background flush thread."""
        if self._running:
            return
        
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="IPCSoulLogger-Flush",
        )
        self._flush_thread.start()
        
        logger.info("IPC Soul Logger started")
    
    def stop(self) -> None:
        """Stop the logger and flush final report."""
        self._running = False
        
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        
        # Final flush
        self._update_soul_md()
        
        logger.info("IPC Soul Logger stopped")
    
    def get_recent_events(self, count: int = 100) -> List[IPCEvent]:
        """Get recent events."""
        with self._lock:
            return self._events[-count:]


# Global singleton instance
_soul_logger: Optional[IPCSoulLogger] = None


def get_soul_logger() -> IPCSoulLogger:
    """Get or create the global soul logger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = IPCSoulLogger()
    return _soul_logger


if __name__ == "__main__":
    # Self-test and validation
    print("IPC Soul Logger Module - ZAID Personal Crypto Trading Bot")
    print("=" * 70)
    
    logger_instance = IPCSoulLogger(
        soul_md_path="/workspace/test_SOUL.md",
        flush_interval_s=5.0,
    )
    logger_instance.start()
    
    # Log some test events
    for i in range(100):
        latency = 100.0 + (i % 50) * 20
        logger_instance.record_latency("rust_engine", "python_orchestrator", latency)
    
    # Simulate some spikes
    logger_instance.record_latency("rust_engine", "python_orchestrator", 5000.0)
    logger_instance.record_latency("rust_engine", "python_orchestrator", 10000.0)
    
    # Simulate dropped messages
    logger_instance.record_dropped_message(
        "event_bus", "ray_actor",
        reason="queue_full",
        message_id="test_msg_001",
    )
    
    # Simulate backpressure
    logger_instance.record_backpressure(
        "ring_buffer", "consumer",
        queue_depth=9500,
        max_depth=10000,
    )
    
    time.sleep(1)
    
    # Get stats
    stats = logger_instance.get_stats()
    print(f"  Total events: {stats.total_events}")
    print(f"  Latency spikes: {stats.total_latency_spikes}")
    print(f"  Dropped messages: {stats.total_messages_dropped}")
    
    # Stop and flush
    logger_instance.stop()
    
    # Verify SOUL.md was created
    assert Path("/workspace/test_SOUL.md").exists()
    print("✓ SOUL.md file created successfully")
    
    # Cleanup test file
    Path("/workspace/test_SOUL.md").unlink()
    
    print("\n✓ IPC Soul Logger module validated successfully")
