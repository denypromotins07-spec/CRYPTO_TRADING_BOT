#!/usr/bin/env python3
"""
System Soul Logger for ZAID Crypto Trading Bot

This module logs system bottlenecks, OS throttling events, and Windows
background task interference to SOUL.md, creating a permanent record of
performance issues and their resolutions.

Features:
- Automatic detection of CPU stealing by background tasks
- Logging of context switch anomalies
- Documentation of GC pauses and memory pressure events
- Integration with SOUL.md knowledge base
- Actionable recommendations for each issue type

Target: AMD Ryzen AI 5 laptop with 8GB RAM on Windows PowerShell
"""

from __future__ import annotations
import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemEventType(Enum):
    """Types of system events to log"""
    CPU_THROTTLING = "cpu_throttling"
    HIGH_CONTEXT_SWITCHES = "high_context_switches"
    PAGE_FAULTS = "page_faults"
    MEMORY_PRESSURE = "memory_pressure"
    GC_PAUSE = "gc_pause"
    NETWORK_LATENCY = "network_latency"
    DISK_IO_BOTTLENECK = "disk_io_bottleneck"
    WINDOWS_UPDATE_INTERFERENCE = "windows_update_interference"
    ANTIVIRUS_SCAN = "antivirus_scan"
    BACKGROUND_TASK_CPU_STEAL = "background_task_cpu_steal"


@dataclass
class SystemEvent:
    """A system performance event"""
    event_type: SystemEventType
    timestamp: float
    severity: str  # "info", "warning", "critical"
    details: Dict[str, Any]
    recommendation: str = ""
    resolved: bool = False


@dataclass
class PerformanceSession:
    """A trading session's performance summary"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    events: List[SystemEvent] = field(default_factory=list)
    total_trades: int = 0
    avg_latency_us: float = 0.0
    max_latency_us: float = 0.0
    profit_loss_inr: float = 0.0
    
    def add_event(self, event: SystemEvent):
        """Add an event to the session"""
        self.events.append(event)
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def event_count(self) -> int:
        """Get number of recorded events"""
        return len(self.events)


class SystemSoulLogger:
    """
    Logs system performance events to SOUL.md
    
    This class creates a permanent record of system bottlenecks,
    their impact on trading, and lessons learned for future optimization.
    """
    
    def __init__(
        self,
        soul_md_path: Optional[str] = None,
        auto_log: bool = True,
    ):
        # Default to workspace SOUL.md
        if soul_md_path is None:
            workspace = Path.cwd()
            soul_md_path = str(workspace / "SOUL.md")
        
        self.soul_md_path = Path(soul_md_path)
        self.auto_log = auto_log
        
        self._current_session: Optional[PerformanceSession] = None
        self._lock = threading.Lock()
        self._event_history: List[SystemEvent] = []
        
        logger.info(f"System Soul Logger initialized, writing to {self.soul_md_path}")
    
    def start_session(self, session_id: str) -> PerformanceSession:
        """Start a new trading session"""
        with self._lock:
            self._current_session = PerformanceSession(
                session_id=session_id,
                start_time=time.time(),
            )
            logger.info(f"Started performance session: {session_id}")
            return self._current_session
    
    def end_session(self) -> Optional[PerformanceSession]:
        """End the current trading session and write to SOUL.md"""
        with self._lock:
            if not self._current_session:
                return None
            
            self._current_session.end_time = time.time()
            session = self._current_session
            
            # Write session summary to SOUL.md
            if self.auto_log:
                self._write_session_to_soul(session)
            
            self._current_session = None
            return session
    
    def log_event(
        self,
        event_type: SystemEventType,
        severity: str,
        details: Dict[str, Any],
        recommendation: str = "",
    ):
        """Log a system performance event"""
        event = SystemEvent(
            event_type=event_type,
            timestamp=time.time(),
            severity=severity,
            details=details,
            recommendation=recommendation,
        )
        
        with self._lock:
            self._event_history.append(event)
            
            if self._current_session:
                self._current_session.add_event(event)
        
        # Auto-log critical events immediately
        if severity == "critical" and self.auto_log:
            self._write_event_to_soul(event)
        
        logger.info(
            f"Logged {event_type.value} event ({severity}): {details}"
        )
    
    def log_cpu_throttling(self, frequency_mhz: int, expected_mhz: int):
        """Log CPU throttling detection"""
        self.log_event(
            SystemEventType.CPU_THROTTLING,
            "critical",
            {
                "current_frequency_mhz": frequency_mhz,
                "expected_frequency_mhz": expected_mhz,
                "reduction_percent": round((1 - frequency_mhz / expected_mhz) * 100, 2),
            },
            recommendation=(
                "Check Windows Power Plan settings. Set to 'High Performance'. "
                "Disable CPU parking. Consider disabling Intel SpeedStep/AMD Cool'n'Quiet "
                "during trading hours."
            ),
        )
    
    def log_high_context_switches(self, rate_per_sec: int, threshold: int):
        """Log excessive context switches"""
        self.log_event(
            SystemEventType.HIGH_CONTEXT_SWITCHES,
            "warning",
            {
                "rate_per_second": rate_per_sec,
                "threshold": threshold,
                "excess_percent": round((rate_per_sec / threshold - 1) * 100, 2),
            },
            recommendation=(
                "Identify processes causing context switches using Process Explorer. "
                "Common culprits: antivirus, cloud sync, browser tabs. "
                "Consider isolating CPU cores for trading using Task Manager affinity."
            ),
        )
    
    def log_background_task_cpu_steal(
        self,
        process_name: str,
        cpu_percent: float,
        timestamp: str,
    ):
        """Log Windows background task stealing CPU cycles"""
        self.log_event(
            SystemEventType.BACKGROUND_TASK_CPU_STEAL,
            "warning",
            {
                "process_name": process_name,
                "cpu_percent": cpu_percent,
                "detection_timestamp": timestamp,
            },
            recommendation=(
                f"Background process '{process_name}' detected consuming {cpu_percent}% CPU. "
                "Use Game Mode or Focus Assist during trading. "
                "Disable unnecessary startup programs. "
                "Schedule Windows updates outside trading hours."
            ),
        )
    
    def log_gc_pause(self, pause_ms: float, triggered_by: str):
        """Log garbage collection pause"""
        self.log_event(
            SystemEventType.GC_PAUSE,
            "warning" if pause_ms < 10 else "critical",
            {
                "pause_duration_ms": pause_ms,
                "triggered_by": triggered_by,
            },
            recommendation=(
                "GC paused trading thread. Ensure GC is disabled during active trading. "
                "If emergency GC triggered, review memory allocation patterns. "
                "Consider pre-allocating objects before trading window."
            ),
        )
    
    def _write_event_to_soul(self, event: SystemEvent):
        """Write a single event to SOUL.md"""
        timestamp = datetime.fromtimestamp(event.timestamp).isoformat()
        
        content = f"""
## System Event: {event.event_type.value.upper()}

**Timestamp:** {timestamp}  
**Severity:** {event.severity.upper()}  

### Details
{json.dumps(event.details, indent=2)}

### Recommendation
{event.recommendation or "No specific recommendation"}

---
"""
        self._append_to_soul(content)
    
    def _write_session_to_soul(self, session: PerformanceSession):
        """Write a complete session summary to SOUL.md"""
        timestamp = datetime.fromtimestamp(session.start_time).isoformat()
        duration = session.duration_seconds
        
        # Categorize events by severity
        critical_events = [e for e in session.events if e.severity == "critical"]
        warning_events = [e for e in session.events if e.severity == "warning"]
        
        # Generate lessons learned
        lessons = self._generate_lessons(session)
        
        content = f"""
# Performance Session: {session.session_id}

**Date:** {timestamp}  
**Duration:** {duration:.2f} seconds ({duration/60:.2f} minutes)  
**Total Trades:** {session.total_trades}  
**P&L:** ₹{session.profit_loss_inr:,.2f}  

## Latency Statistics
- **Average:** {session.avg_latency_us:.2f} μs
- **Maximum:** {session.max_latency_us:.2f} μs

## Events Summary
- **Critical Events:** {len(critical_events)}
- **Warnings:** {len(warning_events)}
- **Total Events:** {session.event_count}

"""
        if critical_events:
            content += "### Critical Events\n\n"
            for event in critical_events:
                content += f"- **{event.event_type.value}**: {json.dumps(event.details)}\n"
            content += "\n"
        
        if warning_events:
            content += "### Warnings\n\n"
            for event in warning_events[:10]:  # Limit to first 10
                content += f"- **{event.event_type.value}**: {json.dumps(event.details)}\n"
            if len(warning_events) > 10:
                content += f"\n... and {len(warning_events) - 10} more warnings\n"
        
        content += "\n## Lessons Learned\n\n"
        for i, lesson in enumerate(lessons, 1):
            content += f"{i}. {lesson}\n"
        
        content += "\n---\n"
        
        self._append_to_soul(content)
    
    def _generate_lessons(self, session: PerformanceSession) -> List[str]:
        """Generate actionable lessons from session events"""
        lessons = []
        
        event_types = set(e.event_type for e in session.events)
        
        if SystemEventType.CPU_THROTTLING in event_types:
            lessons.append(
                "CPU throttling was detected during this session. Ensure power plan is set "
                "to 'High Performance' before next trading session to prevent frequency reduction."
            )
        
        if SystemEventType.BACKGROUND_TASK_CPU_STEAL in event_types:
            lessons.append(
                "Background tasks interfered with trading. Consider enabling Windows Game Mode "
                "or using a dedicated trading profile that disables non-essential services."
            )
        
        if SystemEventType.GC_PAUSE in event_types:
            lessons.append(
                "Garbage collection pauses occurred. Review the GC tuner configuration and "
                "ensure GC is fully disabled during the 4-hour trading window."
            )
        
        if SystemEventType.HIGH_CONTEXT_SWITCHES in event_types:
            lessons.append(
                "Excessive context switches detected. Use Process Explorer to identify the "
                "culprit processes and consider CPU core isolation for the trading bot."
            )
        
        if not lessons:
            lessons.append("No significant system performance issues detected. Continue monitoring.")
        
        return lessons
    
    def _append_to_soul(self, content: str):
        """Append content to SOUL.md file"""
        try:
            # Create file if it doesn't exist
            if not self.soul_md_path.exists():
                self.soul_md_path.parent.mkdir(parents=True, exist_ok=True)
                self.soul_md_path.write_text("# ZAID Bot SOUL.md\n\nSystem Performance Knowledge Base\n")
            
            # Append new content
            with open(self.soul_md_path, 'a', encoding='utf-8') as f:
                f.write(content)
            
            logger.debug(f"Appended to SOUL.md: {len(content)} bytes")
            
        except Exception as e:
            logger.error(f"Failed to write to SOUL.md: {e}")
    
    def get_session_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics for current session"""
        with self._lock:
            if not self._current_session:
                return None
            
            return {
                "session_id": self._current_session.session_id,
                "duration_seconds": self._current_session.duration_seconds,
                "event_count": self._current_session.event_count,
                "critical_events": len([e for e in self._current_session.events if e.severity == "critical"]),
                "warning_events": len([e for e in self._current_session.events if e.severity == "warning"]),
            }


# Global instance
_soul_logger: Optional[SystemSoulLogger] = None
_logger_lock = threading.Lock()


def get_system_soul_logger() -> SystemSoulLogger:
    """Get or create the global system soul logger"""
    global _soul_logger
    
    if _soul_logger is None:
        with _logger_lock:
            if _soul_logger is None:
                _soul_logger = SystemSoulLogger()
    
    return _soul_logger


if __name__ == "__main__":
    # Test the system soul logger
    print("=== System Soul Logger Test ===\n")
    
    logger_instance = SystemSoulLogger(auto_log=False)
    
    # Start a session
    session = logger_instance.start_session("test-session-001")
    
    # Log some events
    logger_instance.log_cpu_throttling(frequency_mhz=2400, expected_mhz=3200)
    logger_instance.log_high_context_switches(rate_per_sec=15000, threshold=10000)
    logger_instance.log_background_task_cpu_steal(
        process_name="OneDrive.exe",
        cpu_percent=12.5,
        timestamp=datetime.now().isoformat(),
    )
    logger_instance.log_gc_pause(pause_ms=15.3, triggered_by="memory_pressure")
    
    # Update session stats
    session.total_trades = 47
    session.avg_latency_us = 85.2
    session.max_latency_us = 450.0
    session.profit_loss_inr = 12500.00
    
    # End session (would write to SOUL.md if auto_log=True)
    logger_instance.end_session()
    
    print(f"Session ended with {session.event_count} events logged")
    print(f"Events would be written to: {logger_instance.soul_md_path}")
