#!/usr/bin/env python3
"""
Telemetry Soul Logger for SOUL.md Updates

This module logs system degradation and telemetry events to SOUL.md,
providing a persistent record of the bot's internal health and performance.

Key Features:
- Automatic detection of microsecond-level degradation
- Persistent logging to SOUL.md file
- Event categorization and severity tracking
- Trend analysis for system health
- Integration with other telemetry components

Designed for the ZAID Personal Crypto Trading Bot to maintain
a comprehensive audit trail of system behavior.
"""

from __future__ import annotations

import time
import threading
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from enum import Enum
from datetime import datetime
import json


class DegradationType(Enum):
    """Types of system degradation."""
    LATENCY_SPIKE = "latency_spike"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_OVERLOAD = "cpu_overload"
    DISK_IO_DELAY = "disk_io_delay"
    NETWORK_LATENCY = "network_latency"
    QUEUE_BACKLOG = "queue_backlog"
    ERROR_RATE_INCREASE = "error_rate_increase"
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"


class SeverityLevel(Enum):
    """Severity levels for degradation events."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class TelemetryEvent:
    """A single telemetry event."""
    timestamp: float
    event_type: DegradationType
    severity: SeverityLevel
    metric_name: str
    current_value: float
    threshold_value: float
    duration_ms: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'duration_ms': self.duration_ms,
            'message': self.message,
            'metadata': self.metadata
        }
    
    def to_markdown(self) -> str:
        """Convert event to markdown format for SOUL.md."""
        emoji = {
            SeverityLevel.INFO: "ℹ️",
            SeverityLevel.LOW: "🟡",
            SeverityLevel.MEDIUM: "🟠",
            SeverityLevel.HIGH: "🔴",
            SeverityLevel.CRITICAL: "🚨"
        }.get(self.severity, "⚪")
        
        return (
            f"### {emoji} [{self.severity.value.upper()}] {self.event_type.value.replace('_', ' ').title()}\n\n"
            f"- **Time:** {datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
            f"- **Metric:** `{self.metric_name}`\n"
            f"- **Current Value:** `{self.current_value:.6f}`\n"
            f"- **Threshold:** `{self.threshold_value:.6f}`\n"
            f"- **Duration:** `{self.duration_ms:.2f}ms`\n"
            f"- **Message:** {self.message}\n"
        )


@dataclass
class SystemHealthSnapshot:
    """Snapshot of system health metrics."""
    timestamp: float
    avg_latency_us: float
    p99_latency_us: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_positions: int
    error_rate: float
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'avg_latency_us': self.avg_latency_us,
            'p99_latency_us': self.p99_latency_us,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'active_positions': self.active_positions,
            'error_rate': self.error_rate,
            'uptime_seconds': self.uptime_seconds
        }


class TelemetrySoulLogger:
    """
    Logs system degradation events to SOUL.md.
    
    Provides X-ray vision into the bot's internal latency and health,
    maintaining a persistent audit trail for post-mortem analysis.
    """
    
    def __init__(
        self,
        soul_path: str = "SOUL.md",
        max_events: int = 10_000,
        auto_flush_interval: float = 60.0
    ) -> None:
        self.soul_path = soul_path
        self.max_events = max_events
        self.auto_flush_interval = auto_flush_interval
        
        # Event storage
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.RLock()
        
        # Baseline metrics for degradation detection
        self._baseline_latency_us: float = 100.0  # Expected baseline
        self._latency_threshold_multiplier: float = 2.0  # 2x baseline = degradation
        
        # Health snapshots
        self._health_snapshots: deque = deque(maxlen=1000)
        
        # Statistics
        self._total_events = 0
        self._events_by_severity: Dict[SeverityLevel, int] = {level: 0 for level in SeverityLevel}
        self._events_by_type: Dict[DegradationType, int] = {}
        
        # Running state
        self._running = True
        
        # Auto-flush thread
        self._flush_thread = threading.Thread(target=self._auto_flush_loop, daemon=True)
        self._flush_thread.start()
        
        # Initialize SOUL.md if needed
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize or load the SOUL.md file."""
        if not os.path.exists(self.soul_path):
            self._write_header()
        else:
            # Load existing events count
            try:
                with open(self.soul_path, 'r') as f:
                    content = f.read()
                    # Count existing events by counting ### headers
                    self._total_events = content.count('### ')
            except Exception:
                pass
    
    def _write_header(self) -> None:
        """Write the SOUL.md header."""
        header = """# ZAID Trading Bot - SOUL.md

## System Observability and Utility Log

This file contains the persistent telemetry record of the ZAID Personal Crypto Trading Bot,
documenting all system degradation events, latency anomalies, and health transitions.

---

## Configuration

- **Baseline Latency:** {} μs
- **Degradation Threshold:** {}x baseline
- **Max Events Stored:** {}

---

## Event Summary

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |
| Info | 0 |

---

## Recent Events

""".format(
    self._baseline_latency_us,
    self._latency_threshold_multiplier,
    self.max_events
)
        
        with open(self.soul_path, 'w') as f:
            f.write(header)
    
    def set_baseline_latency(self, latency_us: float) -> None:
        """Set the baseline latency for degradation detection."""
        self._baseline_latency_us = latency_us
    
    def detect_and_log_degradation(
        self,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        event_type: Optional[DegradationType] = None,
        severity: Optional[SeverityLevel] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[TelemetryEvent]:
        """
        Detect degradation and log an event if thresholds are breached.
        
        Returns the created event if logged, None otherwise.
        """
        # Determine severity based on how much threshold is exceeded
        if severity is None:
            ratio = current_value / threshold_value if threshold_value > 0 else 1.0
            
            if ratio >= 5.0:
                severity = SeverityLevel.CRITICAL
            elif ratio >= 3.0:
                severity = SeverityLevel.HIGH
            elif ratio >= 2.0:
                severity = SeverityLevel.MEDIUM
            elif ratio >= 1.5:
                severity = SeverityLevel.LOW
            else:
                severity = SeverityLevel.INFO
        
        # Determine event type if not provided
        if event_type is None:
            if 'latency' in metric_name.lower():
                event_type = DegradationType.LATENCY_SPIKE
            elif 'memory' in metric_name.lower():
                event_type = DegradationType.MEMORY_PRESSURE
            elif 'cpu' in metric_name.lower():
                event_type = DegradationType.CPU_OVERLOAD
            elif 'error' in metric_name.lower():
                event_type = DegradationType.ERROR_RATE_INCREASE
            else:
                event_type = DegradationType.LATENCY_SPIKE
        
        # Create event
        event = TelemetryEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            duration_ms=0.0,  # Will be updated when event completes
            message=message or f"{metric_name} exceeded threshold: {current_value:.6f} > {threshold_value:.6f}",
            metadata=metadata or {}
        )
        
        self._log_event(event)
        return event
    
    def log_latency_degradation(
        self,
        current_latency_us: float,
        span_name: str = ""
    ) -> Optional[TelemetryEvent]:
        """Log latency degradation specifically."""
        threshold = self._baseline_latency_us * self._latency_threshold_multiplier
        
        if current_latency_us <= threshold:
            return None  # No degradation
        
        return self.detect_and_log_degradation(
            metric_name=f"latency_{span_name}" if span_name else "latency_us",
            current_value=current_latency_us,
            threshold_value=threshold,
            event_type=DegradationType.LATENCY_SPIKE,
            message=f"Event loop latency spike detected: {current_latency_us:.2f}μs (threshold: {threshold:.2f}μs)",
            metadata={
                'baseline_latency_us': self._baseline_latency_us,
                'spike_ratio': current_latency_us / self._baseline_latency_us
            }
        )
    
    def _log_event(self, event: TelemetryEvent) -> None:
        """Log an event internally."""
        with self._lock:
            self._events.append(event)
            self._total_events += 1
            self._events_by_severity[event.severity] += 1
            
            event_type_key = event.event_type.value
            self._events_by_type[event_type_key] = self._events_by_type.get(event_type_key, 0) + 1
        
        # Write to file immediately for critical events
        if event.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
            self._append_to_soul(event)
    
    def _append_to_soul(self, event: TelemetryEvent) -> None:
        """Append an event to the SOUL.md file."""
        try:
            with open(self.soul_path, 'a') as f:
                f.write(event.to_markdown())
                f.write("\n---\n\n")
        except Exception as e:
            print(f"Error writing to SOUL.md: {e}")
    
    def _auto_flush_loop(self) -> None:
        """Background thread to periodically flush events to SOUL.md."""
        while self._running:
            time.sleep(self.auto_flush_interval)
            self.flush_to_soul()
    
    def flush_to_soul(self) -> int:
        """Flush all pending events to SOUL.md."""
        with self._lock:
            events_to_write = list(self._events)[-100:]  # Last 100 events
        
        count = 0
        for event in events_to_write:
            # Check if already written (by checking file)
            # For simplicity, we just append new ones
            self._append_to_soul(event)
            count += 1
        
        return count
    
    def record_health_snapshot(
        self,
        avg_latency_us: float,
        p99_latency_us: float,
        memory_usage_mb: float,
        cpu_usage_percent: float,
        active_positions: int,
        error_rate: float,
        uptime_seconds: float
    ) -> None:
        """Record a system health snapshot."""
        snapshot = SystemHealthSnapshot(
            timestamp=time.time(),
            avg_latency_us=avg_latency_us,
            p99_latency_us=p99_latency_us,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            active_positions=active_positions,
            error_rate=error_rate,
            uptime_seconds=uptime_seconds
        )
        
        with self._lock:
            self._health_snapshots.append(snapshot)
        
        # Check for degradation
        self.log_latency_degradation(p99_latency_us, "p99")
    
    def get_recent_events(self, count: int = 50) -> List[TelemetryEvent]:
        """Get recent telemetry events."""
        with self._lock:
            return list(self._events)[-count:]
    
    def get_events_by_severity(self, severity: SeverityLevel) -> List[TelemetryEvent]:
        """Get events filtered by severity."""
        with self._lock:
            return [e for e in self._events if e.severity == severity]
    
    def get_events_by_type(self, event_type: DegradationType) -> List[TelemetryEvent]:
        """Get events filtered by type."""
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        with self._lock:
            return {
                'total_events': self._total_events,
                'events_by_severity': {
                    level.value: count 
                    for level, count in self._events_by_severity.items()
                },
                'events_by_type': dict(self._events_by_type),
                'recent_event_count': len(self._events),
                'health_snapshots': len(self._health_snapshots)
            }
    
    def get_health_trend(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze health trends over a time window."""
        cutoff = time.time() - (window_minutes * 60)
        
        with self._lock:
            recent_snapshots = [
                s for s in self._health_snapshots 
                if s.timestamp >= cutoff
            ]
        
        if not recent_snapshots:
            return {'trend': 'insufficient_data'}
        
        # Calculate averages
        avg_latency = sum(s.avg_latency_us for s in recent_snapshots) / len(recent_snapshots)
        avg_memory = sum(s.memory_usage_mb for s in recent_snapshots) / len(recent_snapshots)
        avg_cpu = sum(s.cpu_usage_percent for s in recent_snapshots) / len(recent_snapshots)
        
        # Determine trend direction (compare first half to second half)
        mid = len(recent_snapshots) // 2
        if mid > 0:
            first_half_latency = sum(s.avg_latency_us for s in recent_snapshots[:mid]) / mid
            second_half_latency = sum(s.avg_latency_us for s in recent_snapshots[mid:]) / (len(recent_snapshots) - mid)
            
            if second_half_latency > first_half_latency * 1.1:
                trend = "degrading"
            elif second_half_latency < first_half_latency * 0.9:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            'trend': trend,
            'window_minutes': window_minutes,
            'snapshots_analyzed': len(recent_snapshots),
            'avg_latency_us': avg_latency,
            'avg_memory_mb': avg_memory,
            'avg_cpu_percent': avg_cpu
        }
    
    def update_summary_table(self) -> None:
        """Update the summary table in SOUL.md."""
        stats = self.get_statistics()
        
        table = """## Event Summary

| Severity | Count |
|----------|-------|
| Critical | {} |
| High | {} |
| Medium | {} |
| Low | {} |
| Info | {} |

""".format(
    stats['events_by_severity'].get('critical', 0),
    stats['events_by_severity'].get('high', 0),
    stats['events_by_severity'].get('medium', 0),
    stats['events_by_severity'].get('low', 0),
    stats['events_by_severity'].get('info', 0)
)
        
        # Read current file and replace summary section
        try:
            with open(self.soul_path, 'r') as f:
                content = f.read()
            
            # Find and replace summary section
            start_marker = "## Event Summary"
            end_marker = "## Recent Events"
            
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                new_content = content[:start_idx] + table + content[end_idx:]
                
                with open(self.soul_path, 'w') as f:
                    f.write(new_content)
        except Exception as e:
            print(f"Error updating summary: {e}")
    
    def shutdown(self) -> None:
        """Shutdown the logger gracefully."""
        self._running = False
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2.0)
        
        # Final flush
        self.flush_to_soul()
        self.update_summary_table()


# Global instance
_soul_logger: Optional[TelemetrySoulLogger] = None
_soul_lock = threading.Lock()


def get_soul_logger() -> TelemetrySoulLogger:
    """Get or create the global soul logger instance."""
    global _soul_logger
    
    with _soul_lock:
        if _soul_logger is None:
            _soul_logger = TelemetrySoulLogger()
        return _soul_logger


def initialize_soul_logger(soul_path: str = "SOUL.md") -> TelemetrySoulLogger:
    """Initialize the global soul logger with custom settings."""
    global _soul_logger
    
    with _soul_lock:
        if _soul_logger is not None:
            _soul_logger.shutdown()
        
        _soul_logger = TelemetrySoulLogger(soul_path=soul_path)
        return _soul_logger


if __name__ == '__main__':
    # Example usage
    print("Initializing Telemetry Soul Logger...")
    
    logger = initialize_soul_logger("SOUL.md")
    
    # Set baseline
    logger.set_baseline_latency(100.0)
    
    print("\nSimulating telemetry events...")
    
    # Normal operation
    logger.record_health_snapshot(
        avg_latency_us=50,
        p99_latency_us=80,
        memory_usage_mb=2048,
        cpu_usage_percent=25,
        active_positions=5,
        error_rate=0.001,
        uptime_seconds=3600
    )
    print("  Recorded normal health snapshot")
    
    # Latency spike
    logger.record_health_snapshot(
        avg_latency_us=500,
        p99_latency_us=2500,
        memory_usage_mb=2100,
        cpu_usage_percent=45,
        active_positions=8,
        error_rate=0.005,
        uptime_seconds=3660
    )
    print("  Recorded degraded health snapshot (should trigger alert)")
    
    # Manual degradation log
    logger.detect_and_log_degradation(
        metric_name="order_queue_depth",
        current_value=1500,
        threshold_value=1000,
        event_type=DegradationType.QUEUE_BACKLOG,
        severity=SeverityLevel.MEDIUM,
        message="Order queue depth exceeded threshold during high volatility"
    )
    print("  Logged manual degradation event")
    
    # Wait a moment
    time.sleep(0.5)
    
    print("\nStatistics:")
    stats = logger.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\nHealth Trend:")
    trend = logger.get_health_trend(window_minutes=60)
    print(json.dumps(trend, indent=2))
    
    print("\nRecent Events:")
    for event in logger.get_recent_events(5):
        print(f"  [{event.severity.value}] {event.event_type.value}: {event.message[:50]}...")
    
    print("\nFlushing to SOUL.md...")
    logger.flush_to_soul()
    logger.update_summary_table()
    
    logger.shutdown()
    print(f"\nTelemetry Soul Logger test complete. Check SOUL.md for output.")
