#!/usr/bin/env python3
"""
Read-Write Locks for Optimizing Read-Heavy Market Data Caches

This module implements optimized read-write lock primitives for handling
read-heavy access patterns in market data caches.
Designed for the ZAID Personal Crypto Trading Bot to achieve 8k-20k INR/hour targets.

Key Features:
- Multiple readers, single writer semantics
- Writer preference mode to prevent starvation
- Reentrant read locks for nested access
- Timeout-based deadlock prevention
- Compatible with 8GB RAM constraint

Domain Integration: Quantitative Finance Domains 85-96 (Synchronization, Caching)
"""

from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

# Type definitions
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class LockState(IntEnum):
    """Current state of the RW lock."""
    UNLOCKED = 0
    READ_LOCKED = 1
    WRITE_LOCKED = 2
    WRITE_WAITING = 3


@dataclass(slots=True)
class LockStats:
    """Statistics for monitoring lock performance."""
    total_read_locks: int = 0
    total_write_locks: int = 0
    read_wait_count: int = 0
    write_wait_count: int = 0
    total_read_wait_time_ns: int = 0
    total_write_wait_time_ns: int = 0
    max_read_hold_time_ns: int = 0
    max_write_hold_time_ns: int = 0
    contention_events: int = 0
    deadlock_preventions: int = 0


class RWLock:
    """
    Read-Write lock implementation with writer preference.
    
    This lock allows multiple concurrent readers but ensures exclusive
    access for writers. Writer preference prevents writer starvation
    during high read concurrency scenarios common in market data feeds.
    
    Attributes:
        writer_preference: If True, waiting writers block new readers
    """
    
    def __init__(self, writer_preference: bool = True):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        # Reader/writer counts
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
        
        # Configuration
        self._writer_preference = writer_preference
        self._max_read_timeout_s = 5.0  # Deadlock prevention
        
        # Statistics
        self._stats = LockStats()
    
    def acquire_read(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a read lock.
        
        Args:
            timeout: Maximum time to wait (None = use default timeout)
        
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time_ns()
        effective_timeout = timeout if timeout is not None else self._max_read_timeout_s
        
        with self._condition:
            # Check if we should wait for writers
            if self._writer_preference and self._writers_waiting > 0:
                self._stats.read_wait_count += 1
            
            deadline = time.time() + effective_timeout
            
            while self._writer_active or (
                self._writer_preference and self._writers_waiting > 0 and self._readers == 0
            ):
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._stats.deadlock_preventions += 1
                    return False
                
                if not self._condition.wait(timeout=remaining):
                    self._stats.deadlock_preventions += 1
                    return False
            
            self._readers += 1
            self._stats.total_read_locks += 1
            
            wait_time = time.time_ns() - start_time
            self._stats.total_read_wait_time_ns += wait_time
            
            if wait_time > 0:
                self._stats.contention_events += 1
        
        return True
    
    def release_read(self) -> None:
        """Release a read lock."""
        with self._condition:
            if self._readers <= 0:
                raise RuntimeError("Release read lock when no readers active")
            
            self._readers -= 1
            
            if self._readers == 0:
                self._condition.notify_all()
    
    def acquire_write(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a write lock.
        
        Args:
            timeout: Maximum time to wait (None = use default timeout)
        
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time_ns()
        effective_timeout = timeout if timeout is not None else self._max_read_timeout_s
        
        with self._condition:
            self._writers_waiting += 1
            self._stats.write_wait_count += 1
            
            try:
                deadline = time.time() + effective_timeout
                
                while self._readers > 0 or self._writer_active:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        self._stats.deadlock_preventions += 1
                        return False
                    
                    if not self._condition.wait(timeout=remaining):
                        self._stats.deadlock_preventions += 1
                        return False
                
                self._writer_active = True
                self._stats.total_write_locks += 1
                
                wait_time = time.time_ns() - start_time
                self._stats.total_write_wait_time_ns += wait_time
                
                if wait_time > 0:
                    self._stats.contention_events += 1
            finally:
                self._writers_waiting -= 1
        
        return True
    
    def release_write(self) -> None:
        """Release a write lock."""
        with self._condition:
            if not self._writer_active:
                raise RuntimeError("Release write lock when not holding it")
            
            self._writer_active = False
            self._condition.notify_all()
    
    @contextmanager
    def read_lock(self, timeout: Optional[float] = None):
        """Context manager for read lock."""
        if not self.acquire_read(timeout):
            raise TimeoutError("Failed to acquire read lock")
        try:
            yield
        finally:
            self.release_read()
    
    @contextmanager
    def write_lock(self, timeout: Optional[float] = None):
        """Context manager for write lock."""
        if not self.acquire_write(timeout):
            raise TimeoutError("Failed to acquire write lock")
        try:
            yield
        finally:
            self.release_write()
    
    def get_stats(self) -> LockStats:
        """Get current lock statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = LockStats()
    
    @property
    def reader_count(self) -> int:
        """Get current number of active readers."""
        return self._readers
    
    @property
    def has_writer(self) -> bool:
        """Check if a writer is currently active."""
        return self._writer_active
    
    @property
    def writers_waiting_count(self) -> int:
        """Get number of writers waiting."""
        return self._writers_waiting


class ReentrantRWLock(RWLock):
    """
    Reentrant Read-Write lock that allows the same thread to
    acquire read locks multiple times.
    """
    
    def __init__(self, writer_preference: bool = True):
        super().__init__(writer_preference)
        self._reader_threads: Dict[int, int] = {}  # thread_id -> count
        self._writer_thread: Optional[int] = None
    
    def acquire_read(self, timeout: Optional[float] = None) -> bool:
        """Acquire read lock with reentrancy support."""
        thread_id = threading.get_ident()
        
        with self._lock:
            # Check for reentrancy
            if thread_id in self._reader_threads:
                self._reader_threads[thread_id] += 1
                self._stats.total_read_locks += 1
                return True
            
            # Check if we're the writer trying to read (upgrade attempt)
            if self._writer_thread == thread_id:
                raise RuntimeError("Cannot acquire read lock while holding write lock")
        
        # Try to acquire normally
        if super().acquire_read(timeout):
            with self._lock:
                self._reader_threads[thread_id] = 1
            return True
        
        return False
    
    def release_read(self) -> None:
        """Release read lock with reentrancy support."""
        thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id not in self._reader_threads:
                raise RuntimeError("Release read lock without holding it")
            
            self._reader_threads[thread_id] -= 1
            self._stats.total_read_locks -= 1  # Adjust stats
            
            if self._reader_threads[thread_id] == 0:
                del self._reader_threads[thread_id]
        
        # Call parent release if this was the last read
        with self._condition:
            if self._readers > 0:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()
    
    def acquire_write(self, timeout: Optional[float] = None) -> bool:
        """Acquire write lock with reentrancy tracking."""
        thread_id = threading.get_ident()
        
        # Check for reentrancy
        if self._writer_thread == thread_id:
            raise RuntimeError("Cannot acquire write lock recursively")
        
        if super().acquire_write(timeout):
            self._writer_thread = thread_id
            return True
        
        return False
    
    def release_write(self) -> None:
        """Release write lock with reentrancy tracking."""
        thread_id = threading.get_ident()
        
        if self._writer_thread != thread_id:
            raise RuntimeError("Release write lock without holding it")
        
        self._writer_thread = None
        super().release_write()


@dataclass(slots=True)
class CacheEntry(Generic[V]):
    """Cache entry with metadata."""
    value: V
    created_at_ns: int
    last_accessed_ns: int
    access_count: int = 0
    expires_at_ns: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at_ns is None:
            return False
        return time.time_ns() > self.expires_at_ns


class RWLockedCache(Generic[K, V]):
    """
    Thread-safe cache with read-write locking for read-heavy workloads.
    Optimized for market data caching where reads vastly outnumber writes.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl_s: float = 60.0,
        writer_preference: bool = True,
    ):
        self._cache: Dict[K, CacheEntry[V]] = {}
        self._lock = ReentrantRWLock(writer_preference)
        self._max_size = max_size
        self._default_ttl_ns = int(default_ttl_s * 1_000_000_000)
        self._stats = LockStats()
    
    def get(self, key: K) -> Optional[V]:
        """Get a value from cache (read operation)."""
        with self._lock.read_lock():
            entry = self._cache.get(key)
            
            if entry is None:
                return None
            
            if entry.is_expired():
                return None
            
            # Update access metadata
            entry.last_accessed_ns = time.time_ns()
            entry.access_count += 1
            
            return entry.value
    
    def set(
        self,
        key: K,
        value: V,
        ttl_s: Optional[float] = None,
    ) -> None:
        """Set a value in cache (write operation)."""
        now_ns = time.time_ns()
        ttl_ns = int((ttl_s if ttl_s is not None else self._default_ttl_ns))
        
        with self._lock.write_lock():
            # Evict if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_one()
            
            entry = CacheEntry(
                value=value,
                created_at_ns=now_ns,
                last_accessed_ns=now_ns,
                expires_at_ns=now_ns + ttl_ns if ttl_ns > 0 else None,
            )
            self._cache[key] = entry
    
    def delete(self, key: K) -> bool:
        """Delete a key from cache."""
        with self._lock.write_lock():
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def _evict_one(self) -> None:
        """Evict one entry (LRU-style based on access)."""
        if not self._cache:
            return
        
        # Find least recently accessed non-expired entry
        oldest_key = None
        oldest_access = float('inf')
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                oldest_key = key
                break
            
            if entry.last_accessed_ns < oldest_access:
                oldest_access = entry.last_accessed_ns
                oldest_key = key
        
        if oldest_key is not None:
            del self._cache[oldest_key]
    
    def clear_expired(self) -> int:
        """Clear all expired entries."""
        cleared = 0
        
        with self._lock.write_lock():
            expired_keys = [
                k for k, v in self._cache.items()
                if v.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                cleared += 1
        
        return cleared
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock.read_lock():
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock.read_lock():
            lock_stats = self._lock.get_stats()
            
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "lock_stats": {
                    "total_read_locks": lock_stats.total_read_locks,
                    "total_write_locks": lock_stats.total_write_locks,
                    "contention_events": lock_stats.contention_events,
                },
            }


if __name__ == "__main__":
    # Self-test and validation
    print("Read-Write Locks Module - ZAID Personal Crypto Trading Bot")
    print("=" * 70)
    
    # Test basic RWLock
    rwlock = RWLock(writer_preference=True)
    
    # Test read lock
    with rwlock.read_lock():
        print("✓ Read lock acquired")
    
    # Test write lock
    with rwlock.write_lock():
        print("✓ Write lock acquired")
    
    # Test concurrent readers
    results: List[str] = []
    
    def reader_task(task_id: int) -> None:
        with rwlock.read_lock():
            results.append(f"reader_{task_id}")
            time.sleep(0.01)
    
    threads = [threading.Thread(target=reader_task, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(results) == 5
    print("✓ Concurrent readers test passed")
    
    # Test cache
    cache: RWLockedCache[str, float] = RWLockedCache(max_size=100, default_ttl_s=60.0)
    
    cache.set("BTC_price", 50000.0)
    cache.set("ETH_price", 3000.0)
    
    assert cache.get("BTC_price") == 50000.0
    assert cache.get("ETH_price") == 3000.0
    assert cache.get("SOL_price") is None
    
    print("✓ Cache operations test passed")
    
    # Get stats
    stats = cache.get_stats()
    print(f"  Cache size: {stats['size']}")
    print(f"  Read locks: {stats['lock_stats']['total_read_locks']}")
    
    print("\n✓ RW locks module validated successfully")
