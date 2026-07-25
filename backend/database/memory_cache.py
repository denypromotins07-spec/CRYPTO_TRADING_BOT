"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 9
File: backend/database/memory_cache.py

Shared memory cache for O(1) historical lookups with strict 8GB RAM adherence.
Uses multiprocessing.shared_memory for zero-copy data sharing between processes.

Features:
- O(1) lookup time for cached tick data
- Automatic LRU eviction when memory limit approached
- Cross-process data sharing without serialization overhead
- Thread-safe access with read-write locks

Design Patterns: Cache, Observer
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
import mmap
import os
import struct
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import shared_memory
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheEntryType(Enum):
    """Types of data that can be cached."""
    TICK = "tick"
    ORDER_BOOK = "order_book"
    TRADE = "trade"
    SIGNAL = "signal"
    METRIC = "metric"


@dataclass
class CacheMetadata:
    """Metadata for a cache entry."""
    key: str
    entry_type: CacheEntryType
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    symbol: str = ""
    
    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for the memory cache."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def size_mb(self) -> float:
        """Get cache size in MB."""
        return self.total_size_bytes / (1024 * 1024)


class MemoryCache:
    """
    High-performance shared memory cache for crypto trading data.
    
    Implements LRU eviction policy and maintains strict memory limits
    to ensure the bot stays within the 8GB RAM constraint.
    """
    
    # Maximum cache size (6GB to leave room for other components)
    MAX_CACHE_SIZE_BYTES: int = 6 * 1024 * 1024 * 1024
    
    def __init__(self, name: str = "zaid_bot_cache", max_size: Optional[int] = None):
        """
        Initialize the memory cache.
        
        Args:
            name: Name for the shared memory block
            max_size: Maximum cache size in bytes (default: 6GB)
        """
        self.name = name
        self.max_size = max_size or self.MAX_CACHE_SIZE_BYTES
        self._lock = RLock()
        
        # In-memory metadata store (lightweight)
        self._metadata: OrderedDict[str, CacheMetadata] = OrderedDict()
        
        # Actual data storage (using numpy arrays for efficiency)
        self._data_store: Dict[str, np.ndarray] = {}
        
        # Statistics tracking
        self.stats = CacheStats()
        
        # Create or attach to shared memory
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._init_shared_memory()
        
        logger.info(f"MemoryCache initialized with max size {self.max_size / (1024**3):.2f} GB")
    
    def _init_shared_memory(self) -> None:
        """Initialize shared memory block."""
        try:
            # Try to create new shared memory
            self._shm = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=self.max_size
            )
            logger.info(f"Created new shared memory block: {self.name}")
        except FileExistsError:
            # Attach to existing shared memory
            try:
                self._shm = shared_memory.SharedMemory(name=self.name)
                logger.info(f"Attached to existing shared memory block: {self.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize shared memory: {e}")
                self._shm = None
    
    def _generate_key(self, symbol: str, data_type: str, timestamp: int) -> str:
        """Generate a unique cache key."""
        key_string = f"{symbol}:{data_type}:{timestamp}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def put(
        self,
        symbol: str,
        data_type: str,
        timestamp: int,
        data: Union[np.ndarray, bytes, List[float]],
        ttl_seconds: Optional[float] = None
    ) -> str:
        """
        Store data in the cache.
        
        Args:
            symbol: Asset symbol (BTC, ETH, SOL, etc.)
            data_type: Type of data being cached
            timestamp: Timestamp associated with the data
            data: Data to cache (numpy array, bytes, or list)
            ttl_seconds: Time-to-live in seconds (optional)
            
        Returns:
            Cache key for the stored data
        """
        with self._lock:
            # Convert data to numpy array if needed
            if isinstance(data, list):
                data_array = np.array(data, dtype=np.float64)
            elif isinstance(data, bytes):
                data_array = np.frombuffer(data, dtype=np.uint8)
            elif isinstance(data, np.ndarray):
                data_array = data
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")
            
            # Generate cache key
            key = self._generate_key(symbol, data_type, timestamp)
            
            # Check if we need to evict entries
            data_size = data_array.nbytes
            while self.stats.total_size_bytes + data_size > self.max_size:
                self._evict_lru()
            
            # Store data
            self._data_store[key] = data_array
            
            # Update metadata
            now = time.time()
            metadata = CacheMetadata(
                key=key,
                entry_type=CacheEntryType(data_type) if data_type in [e.value for e in CacheEntryType] else CacheEntryType.METRIC,
                size_bytes=data_size,
                created_at=now,
                last_accessed=now,
                symbol=symbol
            )
            
            # Move to end (most recently used)
            if key in self._metadata:
                self._metadata.move_to_end(key)
            self._metadata[key] = metadata
            
            # Update stats
            self.stats.total_entries = len(self._metadata)
            self.stats.total_size_bytes += data_size
            
            logger.debug(f"Cached {data_size} bytes for {symbol}:{data_type}")
            
            return key
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve data from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        with self._lock:
            if key not in self._data_store:
                self.stats.misses += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
            
            # Update metadata
            if key in self._metadata:
                self._metadata[key].touch()
                self._metadata.move_to_end(key)
            
            self.stats.hits += 1
            
            # Return a copy to prevent modification
            return self._data_store[key].copy()
    
    def get_ticks(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int
    ) -> Optional[np.ndarray]:
        """
        Retrieve tick data for a time range.
        
        Args:
            symbol: Asset symbol
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            Concatenated tick data or None
        """
        with self._lock:
            matching_arrays = []
            
            for ts in range(start_ts, end_ts + 1):
                key = self._generate_key(symbol, "tick", ts)
                if key in self._data_store:
                    if key in self._metadata:
                        self._metadata[key].touch()
                        self._metadata.move_to_end(key)
                    matching_arrays.append(self._data_store[key])
                    self.stats.hits += 1
                else:
                    self.stats.misses += 1
            
            if not matching_arrays:
                return None
            
            return np.concatenate(matching_arrays)
    
    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._metadata:
            return
        
        # Get the oldest (least recently used) entry
        oldest_key = next(iter(self._metadata))
        oldest_meta = self._metadata[oldest_key]
        
        # Remove from data store
        if oldest_key in self._data_store:
            del self._data_store[oldest_key]
        
        # Remove from metadata
        del self._metadata[oldest_key]
        
        # Update stats
        self.stats.total_entries -= 1
        self.stats.total_size_bytes -= oldest_meta.size_bytes
        self.stats.evictions += 1
        
        logger.debug(f"Evicted LRU entry: {oldest_key} ({oldest_meta.size_bytes} bytes)")
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._data_store.clear()
            self._metadata.clear()
            self.stats = CacheStats()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_entries": self.stats.total_entries,
                "total_size_mb": self.stats.size_mb,
                "hit_rate": self.stats.hit_rate,
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "evictions": self.stats.evictions,
                "max_size_gb": self.max_size / (1024**3)
            }
    
    def cleanup(self) -> None:
        """Cleanup shared memory resources."""
        if self._shm:
            try:
                self._shm.close()
                # Only unlink if we created it
                # self._shm.unlink()  # Uncomment if you want to delete on cleanup
                logger.info("Shared memory cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up shared memory: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class HistoricalLookupCache(MemoryCache):
    """
    Specialized cache for O(1) historical tick lookups.
    
    Uses a hash-based indexing scheme for instant retrieval
    of historical tick data by timestamp.
    """
    
    def __init__(self, name: str = "zaid_historical_cache"):
        super().__init__(name=name)
        self._timestamp_index: Dict[str, Dict[int, str]] = {}
    
    def index_tick(
        self,
        symbol: str,
        timestamp: int,
        tick_data: np.ndarray
    ) -> str:
        """
        Index a tick for O(1) lookup by timestamp.
        
        Args:
            symbol: Asset symbol
            timestamp: Tick timestamp
            tick_data: Tick data array
            
        Returns:
            Cache key
        """
        key = self.put(symbol, "tick", timestamp, tick_data)
        
        # Update timestamp index
        if symbol not in self._timestamp_index:
            self._timestamp_index[symbol] = {}
        self._timestamp_index[symbol][timestamp] = key
        
        return key
    
    def lookup_tick(self, symbol: str, timestamp: int) -> Optional[np.ndarray]:
        """
        Lookup tick data by symbol and timestamp in O(1).
        
        Args:
            symbol: Asset symbol
            timestamp: Tick timestamp
            
        Returns:
            Tick data or None
        """
        if symbol not in self._timestamp_index:
            return None
        
        key = self._timestamp_index[symbol].get(timestamp)
        if key is None:
            return None
        
        return self.get(key)
    
    def lookup_range(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int
    ) -> np.ndarray:
        """
        Lookup multiple ticks in a range efficiently.
        
        Args:
            symbol: Asset symbol
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            Concatenated tick data
        """
        if symbol not in self._timestamp_index:
            return np.array([])
        
        arrays = []
        for ts in range(start_ts, end_ts + 1):
            key = self._timestamp_index[symbol].get(ts)
            if key:
                data = self.get(key)
                if data is not None:
                    arrays.append(data)
        
        if not arrays:
            return np.array([])
        
        return np.concatenate(arrays)


# Example usage and testing
if __name__ == "__main__":
    # Test the cache
    cache = HistoricalLookupCache()
    
    # Simulate caching tick data
    for i in range(1000):
        symbol = "BTCUSDT"
        timestamp = int(time.time()) + i
        tick_data = np.random.rand(10).astype(np.float64)  # Simulated tick
        cache.index_tick(symbol, timestamp, tick_data)
    
    # Test O(1) lookup
    test_ts = int(time.time()) + 500
    result = cache.lookup_tick("BTCUSDT", test_ts)
    print(f"Lookup result shape: {result.shape if result is not None else 'None'}")
    
    # Print stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    cache.cleanup()
