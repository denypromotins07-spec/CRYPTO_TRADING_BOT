"""
=============================================================================
ZAID PERSONAL CRYPTO TRADING BOT - ASYNCHRONOUS SYSTEM LOGGER
=============================================================================
Infrastructure, Logging, and System Health Monitoring for Extreme Stability

Non-blocking, asynchronous logging implementation for order flow and system events.
Uses Python's asyncio with queue-based batching to ensure zero blocking of the
main trading event loop.

Domains Integrated:
- Asynchronous I/O Patterns
- Log Aggregation Systems
- Event Stream Processing
- Distributed Tracing
- Audit Trail Management
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import json
import threading
import logging
from enum import Enum
from collections import deque


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class EventType(Enum):
    """Classification of system events."""
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    SIGNAL_GENERATED = "signal_generated"
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    MARKET_DATA_UPDATE = "market_data_update"
    SYSTEM_HEALTH = "system_health"
    MEMORY_UPDATE = "memory_update"
    ERROR_EXCEPTION = "error_exception"


@dataclass
class LogEntry:
    """
    Represents a single log entry with structured metadata.
    
    Attributes:
        timestamp: ISO format UTC timestamp
        level: Log severity level
        event_type: Classification of the event
        message: Human-readable log message
        context: Additional structured data
        correlation_id: For tracing related events
        source: Component that generated the log
    """
    timestamp: str
    level: int
    event_type: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    source: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "level_name": LogLevel(self.level).name,
            "event_type": self.event_type,
            "message": self.message,
            "context": self.context,
            "correlation_id": self.correlation_id,
            "source": self.source,
        }
    
    def to_json_line(self) -> str:
        """Convert to JSON line format."""
        return json.dumps(self.to_dict())


class AsyncLogQueue:
    """
    Thread-safe async queue for log entries.
    
    Provides non-blocking enqueue with backpressure handling.
    """
    
    def __init__(self, max_size: int = 10_000):
        """
        Initialize the async log queue.
        
        Args:
            max_size: Maximum queue size before dropping logs
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._dropped_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
    
    async def put(self, entry: LogEntry) -> bool:
        """
        Add entry to queue (non-blocking).
        
        Args:
            entry: Log entry to add
            
        Returns:
            True if successful, False if queue full
        """
        try:
            # Use put_nowait to avoid blocking
            self._queue.put_nowait(entry)
            with self._lock:
                self._total_count += 1
            return True
        except asyncio.QueueFull:
            with self._lock:
                self._dropped_count += 1
            return False
    
    async def get(self) -> Optional[LogEntry]:
        """Get entry from queue (async)."""
        try:
            return await self._queue.get()
        except asyncio.CancelledError:
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self._lock:
            return {
                "current_size": self._queue.qsize(),
                "total_enqueued": self._total_count,
                "dropped": self._dropped_count,
                "max_size": self._queue.maxsize,
            }


class SystemLogger:
    """
    High-performance asynchronous system logger.
    
    Features:
    - Non-blocking log submission from main trading loop
    - Batched writes to disk for I/O efficiency
    - Multiple output handlers (file, console, remote)
    - Automatic log rotation
    - Structured JSON logging for easy parsing
    """
    
    def __init__(
        self,
        log_dir: str = "./logs/",
        log_level: int = LogLevel.INFO.value,
        max_queue_size: int = 10_000,
        batch_size: int = 100,
        flush_interval_seconds: float = 1.0,
        retention_days: int = 30,
        max_file_size_mb: int = 100
    ):
        """
        Initialize the system logger.
        
        Args:
            log_dir: Directory for log files
            log_level: Minimum log level to capture
            max_queue_size: Maximum pending logs in queue
            batch_size: Number of logs per write batch
            flush_interval_seconds: Time between flushes
            retention_days: Days to keep old logs
            max_file_size_mb: Max size before rotation
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.retention_days = retention_days
        self.max_file_size = max_file_size_mb * 1024 * 1024
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "trades").mkdir(exist_ok=True)
        (self.log_dir / "system").mkdir(exist_ok=True)
        (self.log_dir / "errors").mkdir(exist_ok=True)
        
        # Async queue for non-blocking logs
        self._queue = AsyncLogQueue(max_queue_size)
        
        # Running state
        self._running = False
        self._writer_task: Optional[asyncio.Task] = None
        self._rotation_task: Optional[asyncio.Task] = None
        
        # File handles
        self._current_file: Optional[Any] = None
        self._current_file_path: Optional[Path] = None
        self._current_file_size = 0
        
        # Statistics
        self._stats = {
            "logs_written": 0,
            "rotations": 0,
            "errors": 0,
        }
        
        # Standard Python logger for fallback
        self._py_logger = logging.getLogger("zaid_bot")
        self._py_logger.setLevel(logging.DEBUG)
        
        # Setup correlation ID generator
        self._correlation_counter = 0
        self._correlation_lock = threading.Lock()
    
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracing."""
        with self._correlation_lock:
            self._correlation_counter = (self._correlation_counter + 1) % 1_000_000
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            return f"{timestamp}_{self._correlation_counter:06d}"
    
    def _get_log_file_path(self, event_type: str) -> Path:
        """Determine appropriate log file for event type."""
        base_name = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if event_type in [e.value for e in EventType if "order" in e.value]:
            return self.log_dir / "trades" / f"orders_{base_name}.jsonl"
        elif event_type == EventType.ERROR_EXCEPTION.value:
            return self.log_dir / "errors" / f"errors_{base_name}.jsonl"
        else:
            return self.log_dir / "system" / f"system_{base_name}.jsonl"
    
    async def _rotate_logs(self) -> None:
        """Background task for log rotation and cleanup."""
        while self._running:
            try:
                # Close current file if it exceeds size limit
                if self._current_file and self._current_file_size > self.max_file_size:
                    self._current_file.close()
                    self._current_file = None
                    self._stats["rotations"] += 1
                
                # Cleanup old logs
                await self._cleanup_old_logs()
                
                await asyncio.sleep(3600)  # Check hourly
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats["errors"] += 1
                self._py_logger.error(f"Log rotation error: {e}")
    
    async def _cleanup_old_logs(self) -> None:
        """Remove log files older than retention period."""
        try:
            cutoff = datetime.now(timezone.utc).timestamp() - (self.retention_days * 86400)
            
            for subdir in ["trades", "system", "errors"]:
                dir_path = self.log_dir / subdir
                if not dir_path.exists():
                    continue
                
                for log_file in dir_path.glob("*.jsonl"):
                    if log_file.stat().st_mtime < cutoff:
                        log_file.unlink()
        except Exception as e:
            self._py_logger.warning(f"Log cleanup error: {e}")
    
    async def _write_batch(self, entries: List[LogEntry]) -> None:
        """Write a batch of log entries to their respective files."""
        if not entries:
            return
        
        # Group by target file
        by_file: Dict[Path, List[LogEntry]] = {}
        for entry in entries:
            file_path = self._get_log_file_path(entry.event_type)
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(entry)
        
        # Write to each file
        for file_path, file_entries in by_file.items():
            try:
                # Append mode
                with open(file_path, 'a', encoding='utf-8') as f:
                    for entry in file_entries:
                        line = entry.to_json_line()
                        f.write(line + '\n')
                        self._current_file_size = file_path.stat().st_size
                
                self._stats["logs_written"] += len(file_entries)
                
            except Exception as e:
                self._stats["errors"] += 1
                self._py_logger.error(f"Log write error: {e}")
    
    async def _writer_loop(self) -> None:
        """Main writer loop - processes queue in batches."""
        batch: List[LogEntry] = []
        last_flush = asyncio.get_event_loop().time()
        
        while self._running:
            try:
                # Wait for entries or timeout
                try:
                    entry = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=self.flush_interval
                    )
                    if entry:
                        batch.append(entry)
                except asyncio.TimeoutError:
                    pass
                
                # Flush if batch is full or timeout reached
                now = asyncio.get_event_loop().time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and now - last_flush >= self.flush_interval)
                )
                
                if should_flush and batch:
                    await self._write_batch(batch)
                    batch.clear()
                    last_flush = now
                    
            except asyncio.CancelledError:
                # Flush remaining on shutdown
                if batch:
                    await self._write_batch(batch)
                break
            except Exception as e:
                self._stats["errors"] += 1
                self._py_logger.error(f"Writer loop error: {e}")
    
    def log(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        source: str = "unknown"
    ) -> bool:
        """
        Submit a log entry (non-blocking).
        
        Args:
            level: Log severity level
            event_type: Event classification
            message: Log message
            context: Additional structured data
            correlation_id: Tracing ID (auto-generated if None)
            source: Component name
            
        Returns:
            True if successfully queued, False if dropped
        """
        if level.value < self.log_level:
            return True  # Silently skip below threshold
        
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            event_type=event_type.value,
            message=message,
            context=context or {},
            correlation_id=correlation_id or self._generate_correlation_id(),
            source=source,
        )
        
        # Get or create event loop for queue submission
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - need to schedule
            asyncio.create_task(self._queue.put(entry))
            return True
        except RuntimeError:
            # No running loop - create one for this call
            return asyncio.run(self._queue.put(entry))
    
    def info(
        self,
        event_type: EventType,
        message: str,
        **context: Any
    ) -> bool:
        """Convenience method for INFO level logs."""
        return self.log(LogLevel.INFO, event_type, message, context)
    
    def warning(
        self,
        event_type: EventType,
        message: str,
        **context: Any
    ) -> bool:
        """Convenience method for WARNING level logs."""
        return self.log(LogLevel.WARNING, event_type, message, context)
    
    def error(
        self,
        event_type: EventType,
        message: str,
        **context: Any
    ) -> bool:
        """Convenience method for ERROR level logs."""
        return self.log(LogLevel.ERROR, event_type, message, context)
    
    def debug(
        self,
        event_type: EventType,
        message: str,
        **context: Any
    ) -> bool:
        """Convenience method for DEBUG level logs."""
        return self.log(LogLevel.DEBUG, event_type, message, context)
    
    def start(self) -> None:
        """Start the background writer tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start async event loop in background thread
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def main():
                self._writer_task = asyncio.create_task(self._writer_loop())
                self._rotation_task = asyncio.create_task(self._rotate_logs())
                
                # Keep running until cancelled
                try:
                    await asyncio.gather(
                        self._writer_task,
                        self._rotation_task,
                    )
                except asyncio.CancelledError:
                    pass
            
            loop.run_until_complete(main())
            loop.close()
        
        self._background_thread = threading.Thread(
            target=run_async_loop,
            daemon=True
        )
        self._background_thread.start()
    
    def stop(self) -> None:
        """Gracefully shutdown the logger."""
        if not self._running:
            return
        
        self._running = False
        
        # Wait for background thread
        if hasattr(self, '_background_thread'):
            self._background_thread.join(timeout=5.0)
        
        # Close any open files
        if self._current_file:
            self._current_file.close()
            self._current_file = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        queue_stats = self._queue.get_stats()
        return {
            **self._stats,
            "queue": queue_stats,
            "running": self._running,
        }


# Singleton instance
_system_logger: Optional[SystemLogger] = None


def get_system_logger(
    log_dir: str = "./logs/",
    log_level: int = LogLevel.INFO.value
) -> SystemLogger:
    """
    Get or create the singleton SystemLogger instance.
    
    Args:
        log_dir: Directory for log files
        log_level: Minimum log level to capture
        
    Returns:
        SystemLogger instance
    """
    global _system_logger
    if _system_logger is None:
        _system_logger = SystemLogger(log_dir=log_dir, log_level=log_level)
    return _system_logger


if __name__ == "__main__":
    # Test the system logger
    logger = get_system_logger()
    logger.start()
    
    # Log some test events
    logger.info(
        EventType.ORDER_SUBMITTED,
        "Test order submitted",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=45000.0
    )
    
    logger.info(
        EventType.ORDER_FILLED,
        "Order filled successfully",
        symbol="BTCUSDT",
        fill_price=45000.0,
        fill_quantity=0.1
    )
    
    logger.warning(
        EventType.RISK_CHECK_FAILED,
        "Position size exceeds limit",
        requested_size=1.5,
        max_allowed=1.0
    )
    
    # Give time for async processing
    import time
    time.sleep(2)
    
    print(f"Logger stats: {logger.get_stats()}")
    
    logger.stop()
