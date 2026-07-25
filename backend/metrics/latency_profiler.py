#!/usr/bin/env python3
"""
Latency Profiler for ZAID Crypto Trading Bot

This module provides microsecond-resolution profiling of the Nautilus
event loop and all critical trading operations to identify performance
bottlenecks and ensure the bot meets its 8k-20k INR/hour target.

Features:
- Microsecond-resolution timing
- Event loop latency tracking
- Operation-level profiling
- Statistical analysis (p50, p95, p99)
- Real-time alerting on latency spikes
- Integration with SOUL.md for performance logging

Target: AMD Ryzen AI 5 laptop with 8GB RAM
"""

from __future__ import annotations
import time
import logging
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable, TypeVar
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from contextlib import contextmanager
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class LatencyUnit(Enum):
    """Time unit for latency measurements"""
    MICROSECONDS = "μs"
    MILLISECONDS = "ms"
    NANOSECONDS = "ns"


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements"""
    count: int = 0
    min_us: float = float('inf')
    max_us: float = 0.0
    mean_us: float = 0.0
    median_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    stddev_us: float = 0.0
    _samples: List[float] = field(default_factory=list, repr=False)
    
    def add_sample(self, latency_us: float):
        """Add a latency sample"""
        self.count += 1
        self._samples.append(latency_us)
        
        # Update running statistics
        self.min_us = min(self.min_us, latency_us)
        self.max_us = max(self.max_us, latency_us)
        
        # Calculate mean incrementally
        old_mean = self.mean_us
        self.mean_us = old_mean + (latency_us - old_mean) / self.count
        
        # Recalculate percentiles periodically (every 100 samples)
        if self.count % 100 == 0 and len(self._samples) > 0:
            self._update_percentiles()
    
    def _update_percentiles(self):
        """Update percentile calculations"""
        if not self._samples:
            return
        
        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)
        
        self.median_us = sorted_samples[n // 2]
        self.p95_us = sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1]
        self.p99_us = sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1]
        
        if n > 1:
            self.stddev_us = statistics.stdev(self._samples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "count": self.count,
            "min_us": round(self.min_us, 2),
            "max_us": round(self.max_us, 2),
            "mean_us": round(self.mean_us, 2),
            "median_us": round(self.median_us, 2),
            "p95_us": round(self.p95_us, 2),
            "p99_us": round(self.p99_us, 2),
            "stddev_us": round(self.stddev_us, 2),
        }
    
    def reset(self):
        """Reset all statistics"""
        self.count = 0
        self.min_us = float('inf')
        self.max_us = 0.0
        self.mean_us = 0.0
        self.median_us = 0.0
        self.p95_us = 0.0
        self.p99_us = 0.0
        self.stddev_us = 0.0
        self._samples.clear()


@dataclass
class LatencyThreshold:
    """Threshold configuration for alerting"""
    warning_us: float = 100.0  # 100 microseconds
    critical_us: float = 500.0  # 500 microseconds
    disaster_us: float = 1000.0  # 1 millisecond


class LatencyProfiler:
    """
    Microsecond-resolution latency profiler for the trading bot
    
    This class tracks latency across all critical operations and
    provides real-time alerting when thresholds are exceeded.
    """
    
    def __init__(
        self,
        config: Optional[LatencyThreshold] = None,
        max_samples: int = 10000,
    ):
        self.config = config or LatencyThreshold()
        self.max_samples = max_samples
        
        # Per-operation latency tracking
        self._operation_stats: Dict[str, LatencyStats] = {}
        self._lock = threading.RLock()
        
        # Callbacks for threshold violations
        self._warning_callback: Optional[Callable] = None
        self._critical_callback: Optional[Callable] = None
        
        # Overall event loop stats
        self._event_loop_stats = LatencyStats()
        self._last_event_time: Optional[float] = None
        
        logger.info("Latency Profiler initialized")
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """
        Context manager for profiling an operation
        
        Usage:
            with profiler.profile_operation("order_execution"):
                execute_order()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_us = (time.perf_counter() - start) * 1_000_000
            self.record_latency(operation_name, elapsed_us)
    
    def record_latency(self, operation: str, latency_us: float):
        """Record a latency measurement"""
        with self._lock:
            if operation not in self._operation_stats:
                self._operation_stats[operation] = LatencyStats()
            
            self._operation_stats[operation].add_sample(latency_us)
            
            # Check thresholds and trigger alerts
            self._check_thresholds(operation, latency_us)
    
    def _check_thresholds(self, operation: str, latency_us: float):
        """Check if latency exceeds thresholds"""
        if latency_us >= self.config.disaster_us:
            logger.error(
                f"DISASTER LATENCY: {operation} took {latency_us:.2f}μs "
                f"(threshold: {self.config.disaster_us}μs)"
            )
            if self._critical_callback:
                self._critical_callback(operation, latency_us, "disaster")
        elif latency_us >= self.config.critical_us:
            logger.warning(
                f"CRITICAL LATENCY: {operation} took {latency_us:.2f}μs "
                f"(threshold: {self.config.critical_us}μs)"
            )
            if self._critical_callback:
                self._critical_callback(operation, latency_us, "critical")
        elif latency_us >= self.config.warning_us:
            logger.debug(
                f"HIGH LATENCY: {operation} took {latency_us:.2f}μs "
                f"(threshold: {self.config.warning_us}μs)"
            )
            if self._warning_callback:
                self._warning_callback(operation, latency_us, "warning")
    
    def record_event_loop_tick(self):
        """Record an event loop iteration latency"""
        current_time = time.perf_counter()
        
        if self._last_event_time is not None:
            elapsed_us = (current_time - self._last_event_time) * 1_000_000
            self._event_loop_stats.add_sample(elapsed_us)
            self.record_latency("event_loop_tick", elapsed_us)
        
        self._last_event_time = current_time
    
    def set_warning_callback(self, callback: Callable[[str, float, str], None]):
        """Set callback for warning-level latency"""
        self._warning_callback = callback
    
    def set_critical_callback(self, callback: Callable[[str, float, str], None]):
        """Set callback for critical-level latency"""
        self._critical_callback = callback
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get latency statistics"""
        with self._lock:
            if operation:
                if operation in self._operation_stats:
                    return self._operation_stats[operation].to_dict()
                return {}
            
            # Return all stats
            return {
                op: stats.to_dict() 
                for op, stats in self._operation_stats.items()
            }
    
    def get_event_loop_stats(self) -> Dict[str, Any]:
        """Get event loop specific statistics"""
        return self._event_loop_stats.to_dict()
    
    def get_slowest_operations(self, top_n: int = 10) -> List[tuple]:
        """Get the slowest operations by p99 latency"""
        with self._lock:
            sorted_ops = sorted(
                self._operation_stats.items(),
                key=lambda x: x[1].p99_us,
                reverse=True
            )
            return [(op, stats.p99_us) for op, stats in sorted_ops[:top_n]]
    
    def reset_stats(self, operation: Optional[str] = None):
        """Reset statistics"""
        with self._lock:
            if operation:
                if operation in self._operation_stats:
                    self._operation_stats[operation].reset()
            else:
                for stats in self._operation_stats.values():
                    stats.reset()
                self._event_loop_stats.reset()
    
    def export_report(self, filepath: str) -> str:
        """Export a latency report to a file"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_loop": self.get_event_loop_stats(),
            "operations": self.get_stats(),
            "slowest_operations": self.get_slowest_operations(10),
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Latency report exported to {filepath}")
        return filepath


# Global profiler instance
_profiler: Optional[LatencyProfiler] = None
_profiler_lock = threading.Lock()


def get_latency_profiler() -> LatencyProfiler:
    """Get or create the global latency profiler"""
    global _profiler
    
    if _profiler is None:
        with _profiler_lock:
            if _profiler is None:
                _profiler = LatencyProfiler()
    
    return _profiler


# Convenience decorators
def profiled(operation_name: str):
    """Decorator to profile a function"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            profiler = get_latency_profiler()
            with profiler.profile_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the latency profiler
    print("=== Latency Profiler Test ===\n")
    
    profiler = LatencyProfiler(LatencyThreshold(
        warning_us=50,
        critical_us=200,
        disaster_us=500,
    ))
    
    # Simulate some operations
    print("Simulating operations...")
    
    for i in range(100):
        with profiler.profile_operation("websocket_message"):
            time.sleep(0.00005)  # 50 μs simulation
        
        with profiler.profile_operation("order_placement"):
            time.sleep(0.00015)  # 150 μs simulation
        
        profiler.record_event_loop_tick()
    
    # Print statistics
    print("\n=== Event Loop Stats ===")
    print(json.dumps(profiler.get_event_loop_stats(), indent=2))
    
    print("\n=== Operation Stats ===")
    for op, stats in profiler.get_stats().items():
        print(f"\n{op}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean_us']:.2f} μs")
        print(f"  P95: {stats['p95_us']:.2f} μs")
        print(f"  P99: {stats['p99_us']:.2f} μs")
    
    print("\n=== Slowest Operations ===")
    for op, p99 in profiler.get_slowest_operations(5):
        print(f"  {op}: {p99:.2f} μs")
