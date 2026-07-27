#!/usr/bin/env python3
"""
Log Router Module

This module routes logs to different sinks based on severity levels,
implementing the Observer and Sink design patterns.

Key Features:
- Severity-based routing rules
- Multiple sink support (file, console, network, memory)
- Asynchronous log processing
- Configurable filtering and transformation
- Thread-safe operation

Designed for the ZAID Personal Crypto Trading Bot to ensure
logs are properly categorized and delivered to appropriate destinations.
"""

from __future__ import annotations

import time
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import IntEnum
from collections import deque
import json


class LogLevel(IntEnum):
    """Log severity levels."""
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    FATAL = 5


@dataclass(slots=True)
class LogRecord:
    """A single log record."""
    timestamp: float
    level: LogLevel
    target: str
    message: str
    fields: Dict[str, Any] = field(default_factory=dict)
    span_id: Optional[int] = None
    trace_id: Optional[int] = None
    thread_id: int = field(default_factory=lambda: threading.current_thread().ident or 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self.timestamp)),
            'level': self.level.name,
            'target': self.target,
            'message': self.message,
            'fields': self.fields,
            'span_id': self.span_id,
            'trace_id': self.trace_id,
            'thread_id': self.thread_id
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class LogSink(ABC):
    """Abstract base class for log sinks."""
    
    @abstractmethod
    def write(self, record: LogRecord) -> bool:
        """Write a log record. Returns True if successful."""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered logs."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the sink."""
        pass


class ConsoleSink(LogSink):
    """Sink that writes logs to console/stderr."""
    
    def __init__(self, min_level: LogLevel = LogLevel.INFO) -> None:
        self.min_level = min_level
        self._lock = threading.Lock()
    
    def write(self, record: LogRecord) -> bool:
        if record.level < self.min_level:
            return True
        
        with self._lock:
            output = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.timestamp))}] [{record.level.name}] {record.target}: {record.message}"
            if record.level >= LogLevel.ERROR:
                print(output, file=__import__('sys').stderr)
            else:
                print(output)
        return True
    
    def flush(self) -> None:
        pass
    
    def close(self) -> None:
        pass


class FileSink(LogSink):
    """Sink that writes logs to a file."""
    
    def __init__(
        self,
        filepath: str,
        min_level: LogLevel = LogLevel.DEBUG,
        max_size_mb: int = 100,
        backup_count: int = 5
    ) -> None:
        self.filepath = filepath
        self.min_level = min_level
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        
        self._file: Optional[Any] = None
        self._current_size = 0
        self._lock = threading.Lock()
        
        self._open_file()
    
    def _open_file(self) -> None:
        """Open or rotate the log file."""
        import os
        
        # Check if rotation needed
        if self._file is not None and self._current_size >= self.max_size_bytes:
            self._rotate_file()
        
        if self._file is None:
            self._file = open(self.filepath, 'a')
            self._current_size = self._file.tell()
    
    def _rotate_file(self) -> None:
        """Rotate log files."""
        import os
        
        if self._file:
            self._file.close()
            self._file = None
        
        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.filepath}.{i}"
            dst = f"{self.filepath}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)
        
        # Move current to .1
        if os.path.exists(self.filepath):
            os.rename(self.filepath, f"{self.filepath}.1")
    
    def write(self, record: LogRecord) -> bool:
        if record.level < self.min_level:
            return True
        
        with self._lock:
            self._open_file()
            if self._file:
                line = record.to_json() + '\n'
                self._file.write(line)
                self._current_size += len(line.encode('utf-8'))
        return True
    
    def flush(self) -> None:
        with self._lock:
            if self._file:
                self._file.flush()
    
    def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None


class NetworkSink(LogSink):
    """Sink that sends logs to a remote server via UDP/TCP."""
    
    def __init__(
        self,
        host: str,
        port: int,
        protocol: str = 'udp',
        min_level: LogLevel = LogLevel.WARN
    ) -> None:
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.min_level = min_level
        
        self._socket: Optional[Any] = None
        self._lock = threading.Lock()
        self._connect()
    
    def _connect(self) -> None:
        """Establish network connection."""
        import socket
        
        if self.protocol == 'udp':
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self._socket.connect((self.host, self.port))
            except Exception:
                self._socket = None
    
    def write(self, record: LogRecord) -> bool:
        if record.level < self.min_level:
            return True
        
        with self._lock:
            if not self._socket:
                self._connect()
            
            if self._socket:
                try:
                    data = (record.to_json() + '\n').encode('utf-8')
                    if self.protocol == 'udp':
                        self._socket.sendto(data, (self.host, self.port))
                    else:
                        self._socket.sendall(data)
                    return True
                except Exception:
                    self._socket = None
                    return False
        return False
    
    def flush(self) -> None:
        pass
    
    def close(self) -> None:
        with self._lock:
            if self._socket:
                self._socket.close()
                self._socket = None


class MemorySink(LogSink):
    """In-memory sink for recent logs (crash dump support)."""
    
    def __init__(self, max_records: int = 10_000, min_level: LogLevel = LogLevel.DEBUG) -> None:
        self.max_records = max_records
        self.min_level = min_level
        self._records: deque = deque(maxlen=max_records)
        self._lock = threading.Lock()
    
    def write(self, record: LogRecord) -> bool:
        if record.level < self.min_level:
            return True
        
        with self._lock:
            self._records.append(record)
        return True
    
    def flush(self) -> None:
        pass
    
    def close(self) -> None:
        with self._lock:
            self._records.clear()
    
    def get_recent(self, count: int = 100) -> List[LogRecord]:
        """Get recent log records."""
        with self._lock:
            return list(self._records)[-count:]
    
    def get_all(self) -> List[LogRecord]:
        """Get all stored log records."""
        with self._lock:
            return list(self._records)


class RoutingRule:
    """Rule for routing logs to sinks."""
    
    def __init__(
        self,
        sink: LogSink,
        min_level: LogLevel = LogLevel.DEBUG,
        max_level: LogLevel = LogLevel.FATAL,
        targets: Optional[List[str]] = None,
        exclude_targets: Optional[List[str]] = None
    ) -> None:
        self.sink = sink
        self.min_level = min_level
        self.max_level = max_level
        self.targets = set(targets) if targets else None
        self.exclude_targets = set(exclude_targets) if exclude_targets else None
    
    def matches(self, record: LogRecord) -> bool:
        """Check if a record matches this rule."""
        # Check level
        if record.level < self.min_level or record.level > self.max_level:
            return False
        
        # Check target inclusion
        if self.targets and record.target not in self.targets:
            return False
        
        # Check target exclusion
        if self.exclude_targets and record.target in self.exclude_targets:
            return False
        
        return True


class LogRouter:
    """
    Central log router that distributes logs to multiple sinks.
    
    Implements the Observer pattern for log distribution.
    """
    
    def __init__(self, queue_size: int = 10_000) -> None:
        self._rules: List[RoutingRule] = []
        self._rule_lock = threading.RLock()
        
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._running = True
        
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()
        
        self._dropped_count = 0
        self._processed_count = 0
    
    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        with self._rule_lock:
            self._rules.append(rule)
    
    def remove_rule(self, sink: LogSink) -> None:
        """Remove all rules for a sink."""
        with self._rule_lock:
            self._rules = [r for r in self._rules if r.sink != sink]
    
    def route(self, record: LogRecord) -> bool:
        """Route a log record to matching sinks."""
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            self._dropped_count += 1
            return False
    
    def _process_loop(self) -> None:
        """Background processing loop."""
        while self._running:
            try:
                record = self._queue.get(timeout=0.1)
                self._dispatch(record)
                self._processed_count += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Log router error: {e}")
    
    def _dispatch(self, record: LogRecord) -> None:
        """Dispatch record to matching sinks."""
        with self._rule_lock:
            for rule in self._rules:
                if rule.matches(record):
                    try:
                        rule.sink.write(record)
                    except Exception as e:
                        print(f"Sink write error: {e}")
    
    def flush_all(self) -> None:
        """Flush all sinks."""
        with self._rule_lock:
            for rule in self._rules:
                try:
                    rule.sink.flush()
                except Exception:
                    pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            'processed_count': self._processed_count,
            'dropped_count': self._dropped_count,
            'queue_size': self._queue.qsize(),
            'rule_count': len(self._rules)
        }
    
    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown the router gracefully."""
        self._running = False
        
        # Process remaining items
        start_time = time.time()
        while not self._queue.empty() and (time.time() - start_time) < timeout:
            try:
                record = self._queue.get_nowait()
                self._dispatch(record)
            except queue.Empty:
                break
        
        # Flush all sinks
        self.flush_all()
        
        # Close all sinks
        with self._rule_lock:
            for rule in self._rules:
                try:
                    rule.sink.close()
                except Exception:
                    pass


# Convenience functions for creating common routing configurations
def create_default_router(log_directory: str = './logs') -> LogRouter:
    """Create a router with default configuration."""
    router = LogRouter()
    
    # Console sink for INFO and above
    console_sink = ConsoleSink(min_level=LogLevel.INFO)
    router.add_rule(RoutingRule(
        sink=console_sink,
        min_level=LogLevel.INFO
    ))
    
    # File sink for all logs
    file_sink = FileSink(
        filepath=f"{log_directory}/bot.log",
        min_level=LogLevel.DEBUG
    )
    router.add_rule(RoutingRule(
        sink=file_sink,
        min_level=LogLevel.DEBUG
    ))
    
    # Memory sink for crash dumps (ERROR and above)
    memory_sink = MemorySink(max_records=1000, min_level=LogLevel.ERROR)
    router.add_rule(RoutingRule(
        sink=memory_sink,
        min_level=LogLevel.ERROR
    ))
    
    return router


def create_minimal_router() -> LogRouter:
    """Create a minimal router with only console output."""
    router = LogRouter()
    
    console_sink = ConsoleSink(min_level=LogLevel.INFO)
    router.add_rule(RoutingRule(
        sink=console_sink,
        min_level=LogLevel.INFO
    ))
    
    return router


if __name__ == '__main__':
    # Example usage
    print("Initializing Log Router...")
    
    router = create_default_router()
    
    # Create some test log records
    test_records = [
        LogRecord(timestamp=time.time(), level=LogLevel.DEBUG, target="test", message="Debug message"),
        LogRecord(timestamp=time.time(), level=LogLevel.INFO, target="test", message="Info message"),
        LogRecord(timestamp=time.time(), level=LogLevel.WARN, target="trading", message="Warning message"),
        LogRecord(timestamp=time.time(), level=LogLevel.ERROR, target="oms", message="Error message"),
    ]
    
    print("\nRouting test records...")
    for record in test_records:
        success = router.route(record)
        print(f"  Routed {record.level.name}: {success}")
    
    # Wait for processing
    time.sleep(0.5)
    
    print("\nRouter Statistics:")
    stats = router.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\nShutting down...")
    router.shutdown()
    print("Log Router test complete.")
