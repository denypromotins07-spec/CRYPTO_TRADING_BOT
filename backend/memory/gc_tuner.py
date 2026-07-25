#!/usr/bin/env python3
"""
GC Tuner for ZAID Crypto Trading Bot

This module disables and tunes Python's garbage collector during the
4-hour trading window to prevent GC-induced latency spikes that could
cost crucial microseconds in high-frequency trading.

Features:
- Complete GC disable during active trading
- Generational tuning for minimal pause times
- Memory monitoring to prevent unbounded growth
- Automatic GC re-enable after trading session
- Emergency GC triggers when memory thresholds exceeded

Target: AMD Ryzen AI 5 laptop with 8GB RAM constraint
"""

from __future__ import annotations
import gc
import logging
import time
import threading
import sys
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GCTunerConfig:
    """Configuration for GC tuning"""
    
    def __init__(
        self,
        # Disable GC completely during trading
        disable_gc_during_trading: bool = True,
        # Generation 0 threshold (objects before collection)
        gen0_threshold: int = 700,
        # Generation 1 threshold
        gen1_threshold: int = 10,
        # Generation 2 threshold  
        gen2_threshold: int = 10,
        # Maximum memory usage before emergency GC (MB)
        max_memory_mb: int = 6 * 1024,  # 6GB of 8GB total
        # Memory check interval in seconds
        memory_check_interval: float = 1.0,
        # Enable GC statistics tracking
        track_stats: bool = True,
        # Emergency GC callback
        emergency_callback: Optional[Callable] = None,
    ):
        self.disable_gc_during_trading = disable_gc_during_trading
        self.gen0_threshold = gen0_threshold
        self.gen1_threshold = gen1_threshold
        self.gen2_threshold = gen2_threshold
        self.max_memory_mb = max_memory_mb
        self.memory_check_interval = memory_check_interval
        self.track_stats = track_stats
        self.emergency_callback = emergency_callback


class GCState(Enum):
    """Current state of the GC tuner"""
    NORMAL = "normal"
    TRADING_DISABLED = "trading_disabled"
    EMERGENCY = "emergency"
    MONITORING = "monitoring"


@dataclass
class GCStats:
    """Statistics about GC behavior"""
    collections_before_disable: int = 0
    collections_during_disabled: int = 0
    emergency_collections: int = 0
    peak_memory_mb: float = 0.0
    avg_collection_time_ms: float = 0.0
    total_time_in_gc_ms: float = 0.0
    _collection_times: List[float] = field(default_factory=list, repr=False)
    
    def add_collection_time(self, time_ms: float):
        """Record a collection time"""
        self._collection_times.append(time_ms)
        n = len(self._collection_times)
        old_avg = self.avg_collection_time_ms
        self.avg_collection_time_ms = old_avg + (time_ms - old_avg) / n
        self.total_time_in_gc_ms += time_ms
    
    @property
    def num_collections(self) -> int:
        return len(self._collection_times)


class GCTuner:
    """
    Garbage Collector tuner for high-frequency trading
    
    This class manages Python's GC to eliminate pause times during
    the critical 4-hour trading window while preventing memory bloat.
    """
    
    def __init__(self, config: Optional[GCTunerConfig] = None):
        self.config = config or GCTunerConfig()
        self._state = GCState.NORMAL
        self._stats = GCStats()
        self._original_thresholds: Optional[tuple] = None
        self._gc_was_enabled: bool = gc.isenabled()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
        logger.info("GC Tuner initialized")
    
    def configure_for_trading(self):
        """Configure GC for minimal-latency trading"""
        with self._lock:
            if self._state == GCState.TRADING_DISABLED:
                logger.debug("GC already configured for trading")
                return
            
            # Save original thresholds
            self._original_thresholds = gc.get_threshold()
            
            if self.config.disable_gc_during_trading:
                # Disable automatic GC
                gc.disable()
                self._state = GCState.TRADING_DISABLED
                logger.info("GC DISABLED for trading session")
            else:
                # Tune thresholds for minimal pauses
                gc.set_threshold(
                    self.config.gen0_threshold,
                    self.config.gen1_threshold,
                    self.config.gen2_threshold,
                )
                self._state = GCState.MONITORING
                logger.info(f"GC tuned: thresholds={gc.get_threshold()}")
            
            # Start memory monitoring
            self._start_memory_monitoring()
            
            # Record initial stats
            self._stats.collections_before_disable = gc.get_count()[0]
    
    def restore_normal(self):
        """Restore GC to normal operation after trading"""
        with self._lock:
            if self._state == GCState.NORMAL:
                return
            
            # Stop monitoring
            self._stop_memory_monitoring()
            
            # Restore original settings
            if self._original_thresholds:
                gc.set_threshold(*self._original_thresholds)
            
            # Re-enable GC if it was enabled before
            if self._gc_was_enabled:
                gc.enable()
            
            # Force a collection to clean up accumulated objects
            logger.info("Running post-trading GC cleanup...")
            start = time.perf_counter()
            gc.collect()
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._stats.add_collection_time(elapsed_ms)
            
            self._state = GCState.NORMAL
            logger.info(f"GC restored to normal (cleanup took {elapsed_ms:.2f}ms)")
    
    def _start_memory_monitoring(self):
        """Start background thread to monitor memory usage"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            daemon=True,
            name="GC-Memory-Monitor"
        )
        self._monitor_thread.start()
        logger.debug("Memory monitoring started")
    
    def _stop_memory_monitoring(self):
        """Stop the memory monitoring thread"""
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.debug("Memory monitoring stopped")
    
    def _memory_monitor_loop(self):
        """Background loop to monitor memory and trigger emergency GC if needed"""
        while not self._stop_monitoring.is_set():
            try:
                current_memory_mb = self._get_process_memory_mb()
                
                # Update peak memory
                if current_memory_mb > self._stats.peak_memory_mb:
                    self._stats.peak_memory_mb = current_memory_mb
                
                # Check if we need emergency GC
                if current_memory_mb > self.config.max_memory_mb:
                    self._trigger_emergency_gc(current_memory_mb)
                
                time.sleep(self.config.memory_check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
    
    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _trigger_emergency_gc(self, current_memory_mb: float):
        """Trigger emergency garbage collection"""
        logger.warning(
            f"EMERGENCY GC: Memory at {current_memory_mb:.2f}MB "
            f"(threshold: {self.config.max_memory_mb}MB)"
        )
        
        self._state = GCState.EMERGENCY
        
        # Temporarily enable GC and force full collection
        gc.enable()
        start = time.perf_counter()
        collected = gc.collect()
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        self._stats.emergency_collections += 1
        self._stats.add_collection_time(elapsed_ms)
        
        logger.warning(
            f"Emergency GC completed: collected {collected} objects "
            f"in {elapsed_ms:.2f}ms"
        )
        
        # Restore trading state
        if self.config.disable_gc_during_trading:
            gc.disable()
        else:
            gc.set_threshold(
                self.config.gen0_threshold,
                self.config.gen1_threshold,
                self.config.gen2_threshold,
            )
        
        self._state = GCState.TRADING_DISABLED if self.config.disable_gc_during_trading else GCState.MONITORING
        
        # Call emergency callback if configured
        if self.config.emergency_callback:
            try:
                self.config.emergency_callback(current_memory_mb, collected)
            except Exception as e:
                logger.error(f"Emergency callback error: {e}")
    
    def manual_collect(self) -> int:
        """Manually trigger garbage collection and return objects collected"""
        if self._state == GCState.TRADING_DISABLED:
            logger.debug("Manual GC requested during disabled period")
            # Temporarily enable for manual collection
            gc.enable()
        
        start = time.perf_counter()
        collected = gc.collect()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._stats.add_collection_time(elapsed_ms)
        
        logger.debug(f"Manual GC: collected {collected} objects in {elapsed_ms:.2f}ms")
        
        if self._state == GCState.TRADING_DISABLED:
            gc.disable()
        
        return collected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get GC statistics"""
        gc_counts = gc.get_count()
        gc_stats_raw = gc.get_stats() if hasattr(gc, 'get_stats') else []
        
        return {
            "state": self._state.value,
            "gc_enabled": gc.isenabled(),
            "thresholds": gc.get_threshold(),
            "gc_counts": gc_counts,
            "collections_before_disable": self._stats.collections_before_disable,
            "emergency_collections": self._stats.emergency_collections,
            "peak_memory_mb": round(self._stats.peak_memory_mb, 2),
            "avg_collection_time_ms": round(self._stats.avg_collection_time_ms, 2),
            "total_time_in_gc_ms": round(self._stats.total_time_in_gc_ms, 2),
            "num_collections": self._stats.num_collections,
            "current_memory_mb": round(self._get_process_memory_mb(), 2),
        }
    
    @contextmanager
    def trading_session(self):
        """Context manager for automatic GC management during trading"""
        try:
            self.configure_for_trading()
            yield
        finally:
            self.restore_normal()


# Global GC tuner instance
_gc_tuner: Optional[GCTuner] = None
_tuner_lock = threading.Lock()


def get_gc_tuner(config: Optional[GCTunerConfig] = None) -> GCTuner:
    """Get or create the global GC tuner instance"""
    global _gc_tuner
    
    if _gc_tuner is None:
        with _tuner_lock:
            if _gc_tuner is None:
                _gc_tuner = GCTuner(config)
    
    return _gc_tuner


# Convenience functions
def disable_gc_for_trading():
    """Quick function to disable GC for trading"""
    tuner = get_gc_tuner()
    tuner.configure_for_trading()


def enable_gc_after_trading():
    """Quick function to restore GC after trading"""
    tuner = get_gc_tuner()
    tuner.restore_normal()


if __name__ == "__main__":
    # Test GC tuner
    print("=== GC Tuner Test ===\n")
    
    tuner = GCTuner(GCTunerConfig(
        disable_gc_during_trading=True,
        max_memory_mb=1024,  # 1GB for testing
        track_stats=True,
    ))
    
    print(f"Initial stats: {tuner.get_stats()}")
    
    # Simulate trading session
    print("\nStarting trading session (GC disabled)...")
    tuner.configure_for_trading()
    print(f"During trading: {tuner.get_stats()}")
    
    # Allocate some memory
    data = [i for i in range(100000)]
    print(f"Allocated 100k integers, memory: {tuner._get_process_memory_mb():.2f}MB")
    
    # End trading session
    print("\nEnding trading session...")
    tuner.restore_normal()
    print(f"After restore: {tuner.get_stats()}")
    
    # Clean up
    del data
