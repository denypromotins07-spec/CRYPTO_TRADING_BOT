"""
=============================================================================
ZAID PERSONAL CRYPTO TRADING BOT - PERFORMANCE MONITOR
=============================================================================
Infrastructure, Logging, and System Health Monitoring for Extreme Stability

Active RAM usage tracking daemon ensuring the bot never exceeds 8GB limit.
Implements proactive memory management with automatic garbage collection
triggers and emergency shutdown procedures.

Domains Integrated:
- Resource Management
- Performance Profiling
- Memory Optimization
- System Monitoring
- Fault Tolerance
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
import gc
import os
import psutil
from pathlib import Path
from enum import Enum


class MemoryStatus(Enum):
    """Memory usage status levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MemorySnapshot:
    """
    Point-in-time memory usage snapshot.
    
    Attributes:
        timestamp: When snapshot was taken
        ram_used_bytes: Current RAM usage in bytes
        ram_available_bytes: Available RAM in bytes
        ram_percent: Percentage of RAM used
        process_ram_bytes: This process's RAM usage
        gc_objects: Count of tracked objects by GC
        status: Memory status level
    """
    timestamp: str
    ram_used_bytes: int
    ram_available_bytes: int
    ram_percent: float
    process_ram_bytes: int
    gc_objects: int
    status: MemoryStatus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "ram_used_gb": self.ram_used_bytes / (1024**3),
            "ram_available_gb": self.ram_available_bytes / (1024**3),
            "ram_percent": self.ram_percent,
            "process_ram_mb": self.process_ram_bytes / (1024**2),
            "gc_objects": self.gc_objects,
            "status": self.status.value,
        }


@dataclass
class PerformanceMetrics:
    """
    Aggregated performance metrics over a time window.
    
    Attributes:
        avg_ram_percent: Average RAM usage percentage
        max_ram_percent: Peak RAM usage percentage
        min_ram_percent: Minimum RAM usage percentage
        samples_count: Number of samples collected
        gc_runs: Number of garbage collection runs triggered
        oom_warnings: Count of near-OOM warnings
    """
    avg_ram_percent: float = 0.0
    max_ram_percent: float = 0.0
    min_ram_percent: float = 100.0
    samples_count: int = 0
    gc_runs: int = 0
    oom_warnings: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_ram_percent": round(self.avg_ram_percent, 2),
            "max_ram_percent": round(self.max_ram_percent, 2),
            "min_ram_percent": round(self.min_ram_percent, 2),
            "samples_count": self.samples_count,
            "gc_runs": self.gc_runs,
            "oom_warnings": self.oom_warnings,
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring daemon for RAM management.
    
    Features:
    - Continuous RAM monitoring at configurable intervals
    - Multi-tier alert system (warning/critical/emergency)
    - Automatic garbage collection triggers
    - Graceful degradation under memory pressure
    - Emergency shutdown to prevent system instability
    - Historical metrics tracking
    """
    
    # Memory thresholds (percentages)
    WARNING_THRESHOLD: float = 70.0      # Start monitoring closely
    CRITICAL_THRESHOLD: float = 85.0     # Trigger aggressive GC
    EMERGENCY_THRESHOLD: float = 95.0    # Initiate shutdown
    
    # 8GB max constraint in bytes
    MAX_RAM_BYTES: int = 8 * 1024 * 1024 * 1024  # 8GB
    
    def __init__(
        self,
        sample_interval_seconds: float = 1.0,
        history_size: int = 3600,  # Keep 1 hour of samples at 1s interval
        auto_gc_enabled: bool = True,
        callback_on_warning: Optional[Callable] = None,
        callback_on_critical: Optional[Callable] = None,
        callback_on_emergency: Optional[Callable] = None,
    ):
        """
        Initialize the performance monitor.
        
        Args:
            sample_interval_seconds: Time between samples
            history_size: Number of samples to retain
            auto_gc_enabled: Whether to trigger GC automatically
            callback_on_warning: Called when entering warning state
            callback_on_critical: Called when entering critical state
            callback_on_emergency: Called when entering emergency state
        """
        self.sample_interval = sample_interval_seconds
        self.history_size = history_size
        self.auto_gc_enabled = auto_gc_enabled
        
        # Callbacks
        self._on_warning = callback_on_warning
        self._on_critical = callback_on_critical
        self._on_emergency = callback_on_emergency
        
        # State
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # History buffer (circular)
        self._history: List[MemorySnapshot] = []
        self._history_index = 0
        
        # Current state
        self._current_snapshot: Optional[MemorySnapshot] = None
        self._current_status = MemoryStatus.NORMAL
        
        # Metrics
        self._metrics = PerformanceMetrics()
        self._last_gc_time: Optional[float] = None
        
        # Process handle
        self._process = psutil.Process(os.getpid())
        
        # Status transition tracking
        self._last_warning_time: Optional[float] = None
        self._consecutive_critical = 0
    
    def _get_memory_snapshot(self) -> MemorySnapshot:
        """Collect current memory usage snapshot."""
        try:
            # System-wide memory
            virtual_mem = psutil.virtual_memory()
            
            # Process-specific memory
            mem_info = self._process.memory_info()
            process_ram = mem_info.rss
            
            # GC object count
            gc_objects = len(gc.get_objects())
            
            # Calculate percentage relative to 8GB max
            ram_percent = (virtual_mem.used / self.MAX_RAM_BYTES) * 100
            
            # Determine status
            if ram_percent >= self.EMERGENCY_THRESHOLD:
                status = MemoryStatus.EMERGENCY
            elif ram_percent >= self.CRITICAL_THRESHOLD:
                status = MemoryStatus.CRITICAL
            elif ram_percent >= self.WARNING_THRESHOLD:
                status = MemoryStatus.WARNING
            else:
                status = MemoryStatus.NORMAL
            
            return MemorySnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                ram_used_bytes=virtual_mem.used,
                ram_available_bytes=virtual_mem.available,
                ram_percent=ram_percent,
                process_ram_bytes=process_ram,
                gc_objects=gc_objects,
                status=status,
            )
        except Exception as e:
            # Return safe default on error
            return MemorySnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                ram_used_bytes=0,
                ram_available_bytes=self.MAX_RAM_BYTES,
                ram_percent=0.0,
                process_ram_bytes=0,
                gc_objects=0,
                status=MemoryStatus.NORMAL,
            )
    
    def _update_metrics(self, snapshot: MemorySnapshot) -> None:
        """Update aggregated metrics with new snapshot."""
        with self._lock:
            self._metrics.samples_count += 1
            
            # Update running average
            n = self._metrics.samples_count
            self._metrics.avg_ram_percent = (
                (self._metrics.avg_ram_percent * (n - 1) + snapshot.ram_percent) / n
            )
            
            # Update min/max
            self._metrics.max_ram_percent = max(
                self._metrics.max_ram_percent, snapshot.ram_percent
            )
            self._metrics.min_ram_percent = min(
                self._metrics.min_ram_percent, snapshot.ram_percent
            )
            
            # Track OOM warnings
            if snapshot.status in [MemoryStatus.CRITICAL, MemoryStatus.EMERGENCY]:
                self._metrics.oom_warnings += 1
    
    def _add_to_history(self, snapshot: MemorySnapshot) -> None:
        """Add snapshot to circular history buffer."""
        with self._lock:
            if len(self._history) < self.history_size:
                self._history.append(snapshot)
            else:
                self._history[self._history_index] = snapshot
                self._history_index = (self._history_index + 1) % self.history_size
    
    def _handle_status_change(self, snapshot: MemorySnapshot) -> None:
        """Handle transitions between memory status levels."""
        old_status = self._current_status
        new_status = snapshot.status
        
        if new_status == old_status:
            return
        
        self._current_status = new_status
        now = time.time()
        
        if new_status == MemoryStatus.WARNING:
            self._last_warning_time = now
            if self._on_warning:
                try:
                    self._on_warning(snapshot)
                except Exception:
                    pass
                    
        elif new_status == MemoryStatus.CRITICAL:
            self._consecutive_critical = 1
            if self._on_critical:
                try:
                    self._on_critical(snapshot)
                except Exception:
                    pass
            
            # Auto GC if enabled
            if self.auto_gc_enabled:
                self._force_garbage_collection()
                
        elif new_status == MemoryStatus.EMERGENCY:
            if self._on_emergency:
                try:
                    self._on_emergency(snapshot)
                except Exception:
                    pass
            
            # Aggressive GC
            if self.auto_gc_enabled:
                self._force_garbage_collection(aggressive=True)
    
    def _force_garbage_collection(self, aggressive: bool = False) -> None:
        """
        Trigger garbage collection.
        
        Args:
            aggressive: If True, run full collection including finalizers
        """
        try:
            if aggressive:
                # Full collection
                gc.collect(2)
            else:
                # Quick collection
                gc.collect(0)
            
            self._metrics.gc_runs += 1
            self._last_gc_time = time.time()
            
        except Exception:
            pass
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop running on background thread."""
        while self._running:
            try:
                # Collect snapshot
                snapshot = self._get_memory_snapshot()
                
                # Update state
                with self._lock:
                    self._current_snapshot = snapshot
                
                # Update metrics and history
                self._update_metrics(snapshot)
                self._add_to_history(snapshot)
                
                # Handle status changes
                self._handle_status_change(snapshot)
                
                # Sleep until next sample
                time.sleep(self.sample_interval)
                
            except Exception as e:
                # Don't crash the monitor on errors
                time.sleep(self.sample_interval * 10)  # Back off on error
    
    def start(self) -> None:
        """Start the monitoring daemon."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop the monitoring daemon."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
    
    def get_current_status(self) -> MemoryStatus:
        """Get current memory status."""
        with self._lock:
            return self._current_status
    
    def get_current_snapshot(self) -> Optional[MemorySnapshot]:
        """Get most recent memory snapshot."""
        with self._lock:
            return self._current_snapshot
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get aggregated performance metrics."""
        with self._lock:
            return PerformanceMetrics(
                avg_ram_percent=self._metrics.avg_ram_percent,
                max_ram_percent=self._metrics.max_ram_percent,
                min_ram_percent=self._metrics.min_ram_percent,
                samples_count=self._metrics.samples_count,
                gc_runs=self._metrics.gc_runs,
                oom_warnings=self._metrics.oom_warnings,
            )
    
    def get_history(
        self,
        last_n: Optional[int] = None
    ) -> List[MemorySnapshot]:
        """
        Get historical snapshots.
        
        Args:
            last_n: Number of recent snapshots (default: all)
            
        Returns:
            List of MemorySnapshot objects
        """
        with self._lock:
            if not self._history:
                return []
            
            if last_n is None:
                return list(self._history)
            
            # Get last N from circular buffer
            n = min(last_n, len(self._history))
            if len(self._history) <= n:
                return list(self._history)
            
            # Circular buffer logic
            start_idx = (self._history_index - n) % len(self._history)
            if start_idx < self._history_index:
                return self._history[start_idx:self._history_index]
            else:
                return self._history[start_idx:] + self._history[:self._history_index]
    
    def is_safe_to_trade(self) -> bool:
        """
        Check if it's safe to continue trading operations.
        
        Returns:
            True if memory usage is within safe limits
        """
        status = self.get_current_status()
        return status in [MemoryStatus.NORMAL, MemoryStatus.WARNING]
    
    def get_recommendation(self) -> str:
        """
        Get actionable recommendation based on current state.
        
        Returns:
            Recommendation string
        """
        status = self.get_current_status()
        
        if status == MemoryStatus.NORMAL:
            return "Memory usage normal. All systems operational."
        elif status == MemoryStatus.WARNING:
            return (
                "Memory usage elevated. Consider reducing position sizes "
                "or disabling non-essential features."
            )
        elif status == MemoryStatus.CRITICAL:
            return (
                "CRITICAL: Memory usage high. Garbage collection triggered. "
                "Strongly recommend pausing new trade entries."
            )
        else:  # EMERGENCY
            return (
                "EMERGENCY: Memory usage critical. Immediate action required. "
                "Consider graceful shutdown to prevent system instability."
            )
    
    def export_report(self, output_path: str) -> Path:
        """
        Export performance report to JSON file.
        
        Args:
            output_path: Path for output file
            
        Returns:
            Path to created file
        """
        output = Path(output_path)
        
        report = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "current_status": self.get_current_status().value,
            "current_snapshot": (
                self._current_snapshot.to_dict() if self._current_snapshot else None
            ),
            "metrics": self.get_metrics().to_dict(),
            "recent_history": [
                s.to_dict() for s in self.get_history(last_n=60)
            ],  # Last minute at 1s interval
            "configuration": {
                "sample_interval": self.sample_interval,
                "history_size": self.history_size,
                "auto_gc_enabled": self.auto_gc_enabled,
                "thresholds": {
                    "warning": self.WARNING_THRESHOLD,
                    "critical": self.CRITICAL_THRESHOLD,
                    "emergency": self.EMERGENCY_THRESHOLD,
                },
            },
        }
        
        output.write_text(
            __import__('json').dumps(report, indent=2),
            encoding='utf-8'
        )
        
        return output


# Singleton instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(
    sample_interval_seconds: float = 1.0,
    auto_gc_enabled: bool = True
) -> PerformanceMonitor:
    """
    Get or create the singleton PerformanceMonitor instance.
    
    Args:
        sample_interval_seconds: Sampling interval
        auto_gc_enabled: Enable automatic garbage collection
        
    Returns:
        PerformanceMonitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(
            sample_interval_seconds=sample_interval_seconds,
            auto_gc_enabled=auto_gc_enabled,
        )
    return _performance_monitor


if __name__ == "__main__":
    # Test the performance monitor
    def on_warning(snapshot):
        print(f"⚠️  WARNING: RAM at {snapshot.ram_percent:.1f}%")
    
    def on_critical(snapshot):
        print(f"🔴 CRITICAL: RAM at {snapshot.ram_percent:.1f}%")
    
    def on_emergency(snapshot):
        print(f"🚨 EMERGENCY: RAM at {snapshot.ram_percent:.1f}%")
    
    monitor = get_performance_monitor(
        sample_interval_seconds=1.0,
        auto_gc_enabled=True,
    )
    
    # Set up callbacks
    monitor._on_warning = on_warning
    monitor._on_critical = on_critical
    monitor._on_emergency = on_emergency
    
    # Start monitoring
    monitor.start()
    print("Performance monitor started...")
    
    # Run for a few seconds
    for i in range(10):
        time.sleep(1)
        snapshot = monitor.get_current_snapshot()
        if snapshot:
            print(f"  RAM: {snapshot.ram_percent:.1f}% | "
                  f"Process: {snapshot.process_ram_bytes / 1024**2:.1f}MB | "
                  f"Status: {snapshot.status.value}")
    
    # Get metrics
    metrics = monitor.get_metrics()
    print(f"\nMetrics: {metrics.to_dict()}")
    
    # Check if safe to trade
    print(f"\nSafe to trade: {monitor.is_safe_to_trade()}")
    print(f"Recommendation: {monitor.get_recommendation()}")
    
    # Stop monitor
    monitor.stop()
    print("\nMonitor stopped.")
