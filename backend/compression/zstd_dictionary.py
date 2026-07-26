#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 24
Zstandard Dictionary Training for Order Book Snapshots
Dynamically retrains custom Zstd dictionaries when new assets are added
Optimizes compression for repetitive order book structures
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logger.warning("zstandard library not available. Install with: pip install zstandard")


@dataclass
class CompressionStats:
    """Statistics for compression operations."""
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    
    @property
    def ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return self.compressed_size / self.original_size
    
    @property
    def savings_percent(self) -> float:
        return (1.0 - self.ratio) * 100.0


@dataclass
class AssetCompressionConfig:
    """Compression configuration for a specific asset."""
    symbol: str
    dictionary_hash: str = ""
    dictionary_size: int = 0
    training_samples: int = 0
    last_trained: Optional[datetime] = None
    avg_compression_ratio: float = 0.0
    is_active: bool = True


class ZstdDictionaryTrainer:
    """
    Trains and manages Zstandard dictionaries for order book snapshot compression.
    Dictionaries capture common patterns in order book structures for better compression.
    """
    
    # Default dictionary size (128KB recommended for order books)
    DEFAULT_DICT_SIZE = 128 * 1024
    
    # Minimum samples required for training
    MIN_TRAINING_SAMPLES = 100
    
    # Maximum samples to use for training (to avoid memory issues)
    MAX_TRAINING_SAMPLES = 10000
    
    def __init__(self, dict_dir: str | Path, dict_size: int = DEFAULT_DICT_SIZE):
        """
        Initialize the dictionary trainer.
        
        Args:
            dict_dir: Directory to store trained dictionaries
            dict_size: Target dictionary size in bytes
        """
        if not ZSTD_AVAILABLE:
            raise ImportError("zstandard library required for dictionary training")
            
        self.dict_dir = Path(dict_dir)
        self.dict_dir.mkdir(parents=True, exist_ok=True)
        self.dict_size = dict_size
        self.dictionaries: Dict[str, bytes] = {}
        self.configs: Dict[str, AssetCompressionConfig] = {}
        
        # Load existing dictionaries
        self._load_existing_dictionaries()
        
    def _load_existing_dictionaries(self) -> None:
        """Load previously trained dictionaries from disk."""
        for dict_file in self.dict_dir.glob("*.zdict"):
            symbol = dict_file.stem.replace("_dictionary", "")
            try:
                dict_data = dict_file.read_bytes()
                self.dictionaries[symbol] = dict_data
                
                # Create config entry
                self.configs[symbol] = AssetCompressionConfig(
                    symbol=symbol,
                    dictionary_hash=hashlib.sha256(dict_data).hexdigest()[:16],
                    dictionary_size=len(dict_data),
                    is_active=True,
                )
                logger.info(f"Loaded dictionary for {symbol}: {len(dict_data)} bytes")
            except Exception as e:
                logger.error(f"Failed to load dictionary for {symbol}: {e}")
    
    def collect_training_samples(
        self,
        symbol: str,
        samples: List[Dict[str, Any]],
        force_retrain: bool = False
    ) -> bool:
        """
        Collect samples and train/update dictionary for an asset.
        
        Args:
            symbol: Asset symbol (e.g., "BTCUSDT")
            samples: List of order book snapshots as dictionaries
            force_retrain: Force retraining even if dictionary exists
            
        Returns:
            True if training was successful
        """
        if len(samples) < self.MIN_TRAINING_SAMPLES:
            logger.warning(
                f"Insufficient samples for {symbol}: "
                f"{len(samples)} < {self.MIN_TRAINING_SAMPLES}"
            )
            return False
        
        # Limit samples to avoid memory issues
        if len(samples) > self.MAX_TRAINING_SAMPLES:
            samples = samples[:self.MAX_TRAINING_SAMPLES]
            logger.info(f"Using {len(samples)} samples for training (limited from more)")
        
        # Convert samples to JSON lines format for zstd training
        sample_bytes = []
        for sample in samples:
            # Serialize order book snapshot
            json_str = json.dumps(sample, separators=(',', ':'))
            sample_bytes.append(json_str.encode('utf-8'))
        
        try:
            # Train dictionary using zstd
            dict_data = zstd.train_dictionary(
                self.dict_size,
                sample_bytes,
                # Optimize for order book patterns
                level=3,
                notifications=False,
            )
            
            # Save dictionary to disk
            dict_path = self.dict_dir / f"{symbol}_dictionary.zdict"
            dict_path.write_bytes(dict_data)
            
            # Update in-memory cache
            self.dictionaries[symbol] = dict_data
            
            # Update configuration
            self.configs[symbol] = AssetCompressionConfig(
                symbol=symbol,
                dictionary_hash=hashlib.sha256(dict_data).hexdigest()[:16],
                dictionary_size=len(dict_data),
                training_samples=len(samples),
                last_trained=datetime.now(),
                is_active=True,
            )
            
            logger.info(
                f"Trained dictionary for {symbol}: "
                f"{len(dict_data)} bytes, {len(samples)} samples"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to train dictionary for {symbol}: {e}")
            return False
    
    def get_dictionary(self, symbol: str) -> Optional[bytes]:
        """Get the trained dictionary for a symbol."""
        return self.dictionaries.get(symbol)
    
    def get_config(self, symbol: str) -> Optional[AssetCompressionConfig]:
        """Get compression configuration for a symbol."""
        return self.configs.get(symbol)
    
    def needs_retraining(self, symbol: str, current_ratio: float, threshold: float = 0.7) -> bool:
        """
        Check if dictionary needs retraining based on compression ratio.
        
        Args:
            symbol: Asset symbol
            current_ratio: Current compression ratio achieved
            threshold: Ratio below which retraining is triggered
            
        Returns:
            True if retraining is recommended
        """
        config = self.configs.get(symbol)
        if config is None:
            return True  # No dictionary exists
        
        if not config.is_active:
            return True
        
        # If compression ratio degraded significantly, recommend retraining
        if current_ratio > threshold:
            logger.info(
                f"Compression ratio {current_ratio:.2f} exceeded threshold {threshold}, "
                f"retraining recommended for {symbol}"
            )
            return True
        
        return False
    
    def remove_dictionary(self, symbol: str) -> bool:
        """Remove dictionary for a symbol (when asset is removed from portfolio)."""
        if symbol in self.dictionaries:
            del self.dictionaries[symbol]
        
        if symbol in self.configs:
            self.configs[symbol].is_active = False
        
        dict_path = self.dict_dir / f"{symbol}_dictionary.zdict"
        if dict_path.exists():
            try:
                dict_path.unlink()
                logger.info(f"Removed dictionary for {symbol}")
                return True
            except Exception as e:
                logger.error(f"Failed to remove dictionary for {symbol}: {e}")
                return False
        
        return False
    
    def get_all_configs(self) -> Dict[str, AssetCompressionConfig]:
        """Get all compression configurations."""
        return self.configs.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall dictionary statistics."""
        total_dicts = len(self.dictionaries)
        total_size = sum(len(d) for d in self.dictionaries.values())
        active_count = sum(1 for c in self.configs.values() if c.is_active)
        
        return {
            'total_dictionaries': total_dicts,
            'active_dictionaries': active_count,
            'total_size_bytes': total_size,
            'average_dict_size': total_size / total_dicts if total_dicts > 0 else 0,
            'dict_directory': str(self.dict_dir),
        }


class OrderBookCompressor:
    """
    High-performance order book snapshot compressor using trained Zstd dictionaries.
    Provides automatic dictionary selection and fallback compression.
    """
    
    def __init__(self, dict_dir: str | Path, default_level: int = 3):
        """
        Initialize the compressor.
        
        Args:
            dict_dir: Directory containing trained dictionaries
            default_level: Default compression level (0-22)
        """
        if not ZSTD_AVAILABLE:
            raise ImportError("zstandard library required")
            
        self.trainer = ZstdDictionaryTrainer(dict_dir)
        self.default_level = default_level
        self.stats: Dict[str, CompressionStats] = {}
        
        # Pre-create compressors for common levels
        self.compressors: Dict[int, zstd.ZstdCompressor] = {}
        self.decompressors: Dict[int, zstd.ZstdDecompressor] = {}
        
        for level in range(0, 10):
            self.compressors[level] = zstd.ZstdCompressor(level=level)
            self.decompressors[level] = zstd.ZstdDecompressor()
    
    def compress(
        self,
        symbol: str,
        order_book: Dict[str, Any],
        level: Optional[int] = None
    ) -> Tuple[bytes, CompressionStats]:
        """
        Compress an order book snapshot.
        
        Args:
            symbol: Asset symbol
            order_book: Order book snapshot dictionary
            level: Compression level (uses default if None)
            
        Returns:
            Tuple of (compressed_bytes, stats)
        """
        import time
        
        level = level if level is not None else self.default_level
        start = time.perf_counter()
        
        # Serialize order book
        json_data = json.dumps(order_book, separators=(',', ':')).encode('utf-8')
        original_size = len(json_data)
        
        # Get dictionary if available
        dictionary = self.trainer.get_dictionary(symbol)
        
        # Select compressor
        compressor = self.compressors.get(level, self.compressors[self.default_level])
        
        # Compress with or without dictionary
        if dictionary:
            compressed = compressor.compress(json_data, dictionary=dictionary)
        else:
            compressed = compressor.compress(json_data)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Update statistics
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=len(compressed),
            compression_time_ms=elapsed_ms,
        )
        self.stats[symbol] = stats
        
        return compressed, stats
    
    def decompress(
        self,
        symbol: str,
        compressed_data: bytes
    ) -> Tuple[Dict[str, Any], CompressionStats]:
        """
        Decompress an order book snapshot.
        
        Args:
            symbol: Asset symbol
            compressed_data: Compressed bytes
            
        Returns:
            Tuple of (decompressed_dict, stats)
        """
        import time
        
        start = time.perf_counter()
        
        # Get dictionary if available
        dictionary = self.trainer.get_dictionary(symbol)
        
        # Select decompressor
        decompressor = self.decompressors.get(
            self.default_level,
            self.decompressors[3]
        )
        
        # Decompress with or without dictionary
        if dictionary:
            decompressed = decompressor.decompress(compressed_data, dictionary=dictionary)
        else:
            decompressed = decompressor.decompress(compressed_data)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Parse JSON
        order_book = json.loads(decompressed.decode('utf-8'))
        
        stats = CompressionStats(
            compressed_size=len(compressed_data),
            original_size=len(decompressed),
            decompression_time_ms=elapsed_ms,
        )
        
        return order_book, stats
    
    def train_for_symbol(
        self,
        symbol: str,
        samples: List[Dict[str, Any]]
    ) -> bool:
        """Train a dictionary for a specific symbol."""
        return self.trainer.collect_training_samples(symbol, samples)
    
    def get_statistics(self, symbol: str) -> Optional[CompressionStats]:
        """Get compression statistics for a symbol."""
        return self.stats.get(symbol)
    
    def should_retrain(self, symbol: str, threshold: float = 0.7) -> bool:
        """Check if dictionary should be retrained."""
        stats = self.stats.get(symbol)
        if stats is None:
            return True
        
        return self.trainer.needs_retraining(symbol, stats.ratio, threshold)


def main():
    """Example usage and testing."""
    import tempfile
    
    print("Zstandard Dictionary Trainer for Order Book Compression")
    print("=" * 60)
    
    # Create temporary directory for dictionaries
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize compressor
        compressor = OrderBookCompressor(tmpdir)
        
        # Generate synthetic order book samples for training
        symbol = "BTCUSDT"
        samples = []
        
        for i in range(200):
            base_price = 50000 + (i % 100)
            sample = {
                "symbol": symbol,
                "timestamp": 1234567890 + i,
                "bids": [[base_price - j * 0.5, 1.5 + j * 0.1] for j in range(20)],
                "asks": [[base_price + j * 0.5, 1.5 + j * 0.1] for j in range(20)],
                "sequence": i,
            }
            samples.append(sample)
        
        # Train dictionary
        print(f"\nTraining dictionary for {symbol}...")
        success = compressor.train_for_symbol(symbol, samples)
        print(f"Training {'successful' if success else 'failed'}")
        
        # Test compression
        test_book = samples[-1]
        compressed, stats = compressor.compress(symbol, test_book)
        
        print(f"\nCompression Results:")
        print(f"  Original size: {stats.original_size:,} bytes")
        print(f"  Compressed size: {stats.compressed_size:,} bytes")
        print(f"  Compression ratio: {stats.ratio:.2%}")
        print(f"  Savings: {stats.savings_percent:.1f}%")
        print(f"  Compression time: {stats.compression_time_ms:.3f} ms")
        
        # Test decompression
        decompressed, dec_stats = compressor.decompress(symbol, compressed)
        print(f"\nDecompression time: {dec_stats.decompression_time_ms:.3f} ms")
        print(f"Decompression successful: {decompressed == test_book}")
        
        # Get dictionary statistics
        dict_stats = compressor.trainer.get_statistics()
        print(f"\nDictionary Statistics:")
        for key, value in dict_stats.items():
            print(f"  {key}: {value}")
    
    print("\nZstandard dictionary trainer ready!")


if __name__ == "__main__":
    main()
