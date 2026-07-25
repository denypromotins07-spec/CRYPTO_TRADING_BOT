#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 24
Memory-Mapped Binary File Loader for Historical Data
Provides instant Python access to FlatBuffer-serialized tick data
Uses memory mapping for zero-copy file access and minimal RAM footprint
"""

from __future__ import annotations

import mmap
import struct
import os
import logging
from pathlib import Path
from typing import Optional, Iterator, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """Message type identifiers matching FlatBuffer schema."""
    TICK = 0
    ORDER_BOOK_SNAPSHOT = 1
    ORDER_BOOK_UPDATE = 2
    DEPTH_STATS = 3
    TRADING_SIGNAL = 4
    EXECUTION_ORDER = 5
    EXECUTION_REPORT = 6
    RISK_METRICS = 7
    SYSTEM_HEALTH = 8


@dataclass(slots=True)
class TickRecord:
    """Zero-copy tick record from memory-mapped file."""
    timestamp_ns: int
    price: float
    quantity: float
    side: str
    trade_id: int
    
    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> TickRecord:
        """Parse tick from binary data at given offset."""
        # Unpack fixed-size fields (matching Rust structure)
        timestamp_ns, price_fp, quantity_fp, side_byte, _, _, _ = struct.unpack_from(
            '<qQQBBBxQ', data, offset
        )
        
        # Convert fixed-point to float
        price = price_fp / 1e8
        quantity = quantity_fp / 1e8
        side = 'BUY' if side_byte == 1 else 'SELL'
        
        return cls(
            timestamp_ns=timestamp_ns,
            price=price,
            quantity=quantity,
            side=side,
            trade_id=0  # Would need additional parsing
        )


@dataclass
class MMapFile:
    """Memory-mapped file wrapper with metadata."""
    path: Path
    size: int
    mmap_obj: Optional[mmap.mmap] = None
    header_size: int = 64  # Reserved for file header
    record_size: int = 64  # Fixed tick record size
    
    def __post_init__(self):
        self.size = self.path.stat().st_size if self.path.exists() else 0
        
    def open(self, access: int = mmap.ACCESS_READ) -> None:
        """Open memory-mapped file."""
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
            
        fd = os.open(str(self.path), os.O_RDONLY)
        try:
            self.mmap_obj = mmap.mmap(fd, 0, access=access)
            logger.info(f"Mapped {self.path.name}: {self.size:,} bytes")
        finally:
            os.close(fd)
    
    def close(self) -> None:
        """Close memory-mapped file."""
        if self.mmap_obj is not None:
            self.mmap_obj.close()
            self.mmap_obj = None
            logger.debug(f"Unmapped {self.path.name}")
    
    def __enter__(self) -> MMapFile:
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class MemoryMappedLoader:
    """
    High-performance loader for memory-mapped binary files.
    Supports random access, slicing, and iteration without loading full file into RAM.
    """
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.loaded_files: Dict[str, MMapFile] = {}
        self.file_index: Dict[str, List[int]] = {}  # symbol -> [offsets]
        
    def load_file(self, filepath: str | Path, symbol: Optional[str] = None) -> MMapFile:
        """Load and memory-map a binary file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        mmap_file = MMapFile(path=path, size=path.stat().st_size)
        mmap_file.open()
        
        if symbol is None:
            symbol = path.stem.split('_')[0].upper()
            
        self.loaded_files[symbol] = mmap_file
        logger.info(f"Loaded {symbol}: {mmap_file.size:,} bytes")
        
        return mmap_file
    
    def unload_file(self, symbol: str) -> None:
        """Unload a memory-mapped file."""
        if symbol in self.loaded_files:
            self.loaded_files[symbol].close()
            del self.loaded_files[symbol]
            logger.debug(f"Unloaded {symbol}")
    
    def unload_all(self) -> None:
        """Unload all memory-mapped files."""
        for symbol in list(self.loaded_files.keys()):
            self.unload_file(symbol)
    
    def get_tick_count(self, symbol: str) -> int:
        """Get number of ticks in loaded file."""
        if symbol not in self.loaded_files:
            raise KeyError(f"Symbol not loaded: {symbol}")
            
        mmap_file = self.loaded_files[symbol]
        data_size = mmap_file.size - mmap_file.header_size
        return max(0, data_size // mmap_file.record_size)
    
    def get_tick(self, symbol: str, index: int) -> Optional[TickRecord]:
        """Get tick at specific index using zero-copy access."""
        if symbol not in self.loaded_files:
            raise KeyError(f"Symbol not loaded: {symbol}")
            
        mmap_file = self.loaded_files[symbol]
        if mmap_file.mmap_obj is None:
            raise ValueError("File not opened")
            
        tick_count = self.get_tick_count(symbol)
        if index < 0 or index >= tick_count:
            raise IndexError(f"Index {index} out of range [0, {tick_count})")
            
        offset = mmap_file.header_size + (index * mmap_file.record_size)
        data = mmap_file.mmap_obj[offset:offset + mmap_file.record_size]
        
        return TickRecord.from_bytes(data)
    
    def get_ticks_slice(
        self,
        symbol: str,
        start_idx: int,
        end_idx: int,
        as_numpy: bool = True
    ) -> np.ndarray | List[TickRecord]:
        """Get slice of ticks efficiently."""
        if symbol not in self.loaded_files:
            raise KeyError(f"Symbol not loaded: {symbol}")
            
        mmap_file = self.loaded_files[symbol]
        if mmap_file.mmap_obj is None:
            raise ValueError("File not opened")
            
        tick_count = self.get_tick_count(symbol)
        start_idx = max(0, start_idx)
        end_idx = min(tick_count, end_idx)
        
        if start_idx >= end_idx:
            return np.array([]) if as_numpy else []
        
        offset = mmap_file.header_size + (start_idx * mmap_file.record_size)
        size = (end_idx - start_idx) * mmap_file.record_size
        
        # Zero-copy view into mmap
        data_view = mmap_file.mmap_obj[offset:offset + size]
        
        if as_numpy:
            # Create numpy array from buffer
            arr = np.frombuffer(data_view, dtype=np.uint8)
            return arr.reshape(-1, mmap_file.record_size)
        else:
            ticks = []
            for i in range(start_idx, end_idx):
                tick = self.get_tick(symbol, i)
                if tick:
                    ticks.append(tick)
            return ticks
    
    def iterate_ticks(self, symbol: str) -> Iterator[TickRecord]:
        """Iterate over all ticks in file."""
        tick_count = self.get_tick_count(symbol)
        for i in range(tick_count):
            tick = self.get_tick(symbol, i)
            if tick:
                yield tick
    
    def search_by_timestamp(
        self,
        symbol: str,
        target_ts_ns: int,
        find_first: bool = True
    ) -> Optional[int]:
        """Binary search for timestamp in sorted data."""
        if symbol not in self.loaded_files:
            raise KeyError(f"Symbol not loaded: {symbol}")
            
        tick_count = self.get_tick_count(symbol)
        if tick_count == 0:
            return None
            
        # Binary search
        left, right = 0, tick_count - 1
        result = None
        
        while left <= right:
            mid = (left + right) // 2
            tick = self.get_tick(symbol, mid)
            if tick is None:
                break
                
            if tick.timestamp_ns == target_ts_ns:
                if find_first:
                    # Find first occurrence
                    result = mid
                    right = mid - 1
                else:
                    return mid
            elif tick.timestamp_ns < target_ts_ns:
                left = mid + 1
            else:
                right = mid - 1
        
        # If not exact match, return closest
        if result is None and not find_first:
            return left if left < tick_count else None
            
        return result
    
    def get_time_range(self, symbol: str) -> Tuple[int, int]:
        """Get min and max timestamps in file."""
        if symbol not in self.loaded_files:
            raise KeyError(f"Symbol not loaded: {symbol}")
            
        tick_count = self.get_tick_count(symbol)
        if tick_count == 0:
            return (0, 0)
            
        first_tick = self.get_tick(symbol, 0)
        last_tick = self.get_tick(symbol, tick_count - 1)
        
        if first_tick and last_tick:
            return (first_tick.timestamp_ns, last_tick.timestamp_ns)
        return (0, 0)
    
    def get_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistical summary of loaded data."""
        if symbol not in self.loaded_files:
            raise KeyError(f"Symbol not loaded: {symbol}")
            
        tick_count = self.get_tick_count(symbol)
        time_range = self.get_time_range(symbol)
        
        # Sample first and last ticks for price range
        first_tick = self.get_tick(symbol, 0)
        last_tick = self.get_tick(symbol, tick_count - 1) if tick_count > 1 else first_tick
        
        min_price = first_tick.price if first_tick else 0.0
        max_price = last_tick.price if last_tick else 0.0
        
        return {
            'symbol': symbol,
            'tick_count': tick_count,
            'file_size_bytes': self.loaded_files[symbol].size,
            'time_start_ns': time_range[0],
            'time_end_ns': time_range[1],
            'duration_hours': (time_range[1] - time_range[0]) / 3.6e12 if time_range[1] > 0 else 0,
            'min_price': min_price,
            'max_price': max_price,
            'avg_tick_size_bytes': self.loaded_files[symbol].record_size,
        }


class HistoricalDataStore:
    """
    High-level interface for managing historical binary data files.
    Supports multiple symbols, automatic file discovery, and batch operations.
    """
    
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.loader = MemoryMappedLoader(base_dir)
        self.symbols: List[str] = []
        
    def discover_files(self, pattern: str = "*.ztb") -> List[Path]:
        """Discover all binary data files matching pattern."""
        files = list(self.base_dir.glob(pattern))
        logger.info(f"Discovered {len(files)} data files")
        return files
    
    def load_symbol(self, symbol: str, filename: Optional[str] = None) -> None:
        """Load data for specific symbol."""
        if filename is None:
            # Auto-discover file
            pattern = f"{symbol.upper()}*.ztb"
            files = list(self.base_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No data file found for {symbol}")
            filename = files[0]
            
        self.loader.load_file(filename, symbol)
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def load_all_symbols(self, pattern: str = "*.ztb") -> None:
        """Load all discovered symbol files."""
        files = self.discover_files(pattern)
        for filepath in files:
            symbol = filepath.stem.split('_')[0].upper()
            try:
                self.load_symbol(symbol, filepath)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    
    def get_cross_symbol_ticks(
        self,
        symbols: List[str],
        start_ts_ns: int,
        end_ts_ns: int
    ) -> Dict[str, List[TickRecord]]:
        """Get synchronized ticks across multiple symbols for same time window."""
        result = {}
        for symbol in symbols:
            if symbol not in self.loader.loaded_files:
                continue
                
            # Find time range indices
            loader = self.loader
            start_idx = loader.search_by_timestamp(symbol, start_ts_ns)
            end_idx = loader.search_by_timestamp(symbol, end_ts_ns, find_first=False)
            
            if start_idx is not None and end_idx is not None:
                ticks = loader.get_ticks_slice(symbol, start_idx, end_idx + 1, as_numpy=False)
                result[symbol] = ticks if isinstance(ticks, list) else []
                
        return result
    
    def close(self) -> None:
        """Close all loaded files."""
        self.loader.unload_all()
        self.symbols.clear()
    
    def __enter__(self) -> HistoricalDataStore:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def main():
    """Example usage of memory-mapped loader."""
    # Create test directory
    test_dir = Path("/tmp/zaid_test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize store
    store = HistoricalDataStore(test_dir)
    
    # In production, files would be pre-generated by the Rust writer
    print(f"Data directory: {test_dir}")
    print(f"Discovered files: {store.discover_files()}")
    
    # Example statistics (would require actual data files)
    # stats = store.loader.get_statistics("BTCUSDT")
    # print(f"Statistics: {stats}")
    
    store.close()
    print("Memory-mapped loader ready")


if __name__ == "__main__":
    main()
