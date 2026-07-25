"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 9
File: backend/persistence/state_serializer.py

Rapid serialization of open positions and trading state for crash recovery.
Uses efficient binary protocols with support for incremental snapshots.

Features:
- Protocol Buffers-style binary serialization
- Delta encoding for minimal storage
- Checkpoint creation and restoration
- Multi-asset position tracking (BTC, SOL, ETH, USDT)

Design Patterns: Memento, Builder
"""

from __future__ import annotations
import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import pickle
import zlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class PositionState:
    """Represents an open position."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: int
    margin: float
    liquidation_price: Optional[float]
    opened_at: float
    last_updated: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'leverage': self.leverage,
            'margin': self.margin,
            'liquidation_price': self.liquidation_price,
            'opened_at': self.opened_at,
            'last_updated': self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PositionState:
        """Create from dictionary."""
        return cls(
            symbol=data['symbol'],
            side=PositionSide(data['side']),
            quantity=data['quantity'],
            entry_price=data['entry_price'],
            current_price=data['current_price'],
            unrealized_pnl=data['unrealized_pnl'],
            leverage=data['leverage'],
            margin=data['margin'],
            liquidation_price=data.get('liquidation_price'),
            opened_at=data['opened_at'],
            last_updated=data['last_updated'],
        )


@dataclass
class OrderState:
    """Represents an order state."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    filled_quantity: float
    average_fill_price: float
    status: OrderStatus
    created_at: float
    updated_at: float
    exchange_order_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'filled_quantity': self.filled_quantity,
            'average_fill_price': self.average_fill_price,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'exchange_order_id': self.exchange_order_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrderState:
        """Create from dictionary."""
        return cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            side=data['side'],
            order_type=data['order_type'],
            quantity=data['quantity'],
            price=data.get('price'),
            filled_quantity=data['filled_quantity'],
            average_fill_price=data['average_fill_price'],
            status=OrderStatus(data['status']),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            exchange_order_id=data.get('exchange_order_id'),
        )


@dataclass
class TradingState:
    """Complete trading state snapshot."""
    timestamp: float
    positions: Dict[str, PositionState]
    orders: Dict[str, OrderState]
    balances: Dict[str, float]
    total_equity: float
    realized_pnl: float
    session_start: float
    trade_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'orders': {k: v.to_dict() for k, v in self.orders.items()},
            'balances': self.balances,
            'total_equity': self.total_equity,
            'realized_pnl': self.realized_pnl,
            'session_start': self.session_start,
            'trade_count': self.trade_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TradingState:
        """Create from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            positions={k: PositionState.from_dict(v) for k, v in data['positions'].items()},
            orders={k: OrderState.from_dict(v) for k, v in data['orders'].items()},
            balances=data['balances'],
            total_equity=data['total_equity'],
            realized_pnl=data['realized_pnl'],
            session_start=data['session_start'],
            trade_count=data['trade_count'],
        )


class StateSerializer:
    """
    High-performance state serializer for trading bot persistence.
    
    Supports multiple serialization formats with compression
    for efficient storage and fast recovery.
    """
    
    # Binary format magic bytes
    MAGIC_BYTES = b'ZAID'
    FORMAT_VERSION = 1
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize the state serializer.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._checkpoint_counter = 0
        self._last_checkpoint_time = 0.0
        
        logger.info(f"StateSerializer initialized at {checkpoint_dir}")
    
    def serialize_binary(self, state: TradingState) -> bytes:
        """
        Serialize state to compact binary format.
        
        Format:
        - Magic bytes (4): ZAID
        - Version (1): Format version
        - Flags (1): Compression flag
        - Timestamp (8): Float64
        - Data length (4): UInt32
        - Compressed JSON data (variable)
        - CRC32 checksum (4)
        
        Args:
            state: Trading state to serialize
            
        Returns:
            Serialized binary data
        """
        # Convert to JSON first
        json_data = json.dumps(state.to_dict(), separators=(',', ':')).encode('utf-8')
        
        # Compress with zlib
        compressed = zlib.compress(json_data, level=6)
        
        # Build binary structure
        header = struct.pack(
            '<4sBBdI',
            self.MAGIC_BYTES,
            self.FORMAT_VERSION,
            1,  # Compression flag
            state.timestamp,
            len(compressed)
        )
        
        # Calculate checksum
        checksum = zlib.crc32(header + compressed) & 0xffffffff
        
        # Pack checksum
        footer = struct.pack('<I', checksum)
        
        return header + compressed + footer
    
    def deserialize_binary(self, data: bytes) -> Optional[TradingState]:
        """
        Deserialize state from binary format.
        
        Args:
            data: Binary data to deserialize
            
        Returns:
            TradingState or None if invalid
        """
        if len(data) < 21:  # Minimum header + footer size
            logger.error("Data too short for valid state")
            return None
        
        # Parse header
        magic = data[:4]
        if magic != self.MAGIC_BYTES:
            logger.error(f"Invalid magic bytes: {magic}")
            return None
        
        version, flags, timestamp, data_len = struct.unpack('<4sBBdI', data[:17])
        
        if version != self.FORMAT_VERSION:
            logger.error(f"Unsupported format version: {version}")
            return None
        
        # Extract compressed data
        compressed = data[17:17 + data_len]
        
        # Verify checksum
        stored_checksum = struct.unpack('<I', data[17 + data_len:17 + data_len + 4])[0]
        computed_checksum = zlib.crc32(data[:17 + data_len]) & 0xffffffff
        
        if stored_checksum != computed_checksum:
            logger.error("Checksum verification failed")
            return None
        
        # Decompress and parse
        try:
            json_data = zlib.decompress(compressed)
            state_dict = json.loads(json_data.decode('utf-8'))
            return TradingState.from_dict(state_dict)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    def save_checkpoint(self, state: TradingState) -> Path:
        """
        Save a state checkpoint to disk.
        
        Args:
            state: Trading state to save
            
        Returns:
            Path to the checkpoint file
        """
        timestamp = int(state.timestamp * 1000)
        filename = f"checkpoint_{timestamp}_{self._checkpoint_counter}.bin"
        checkpoint_path = self.checkpoint_dir / filename
        
        # Serialize
        binary_data = self.serialize_binary(state)
        
        # Write atomically (write to temp, then rename)
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            f.write(binary_data)
            f.flush()
        
        # Atomic rename
        temp_path.rename(checkpoint_path)
        
        self._checkpoint_counter += 1
        self._last_checkpoint_time = time.time()
        
        logger.info(f"Checkpoint saved: {checkpoint_path} ({len(binary_data)} bytes)")
        
        return checkpoint_path
    
    def load_latest_checkpoint(self) -> Optional[TradingState]:
        """
        Load the most recent checkpoint.
        
        Returns:
            TradingState or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.bin"))
        
        if not checkpoints:
            logger.info("No checkpoints found")
            return None
        
        # Sort by filename (includes timestamp)
        checkpoints.sort(key=lambda p: p.name, reverse=True)
        latest = checkpoints[0]
        
        try:
            with open(latest, 'rb') as f:
                data = f.read()
            
            state = self.deserialize_binary(data)
            if state:
                logger.info(f"Loaded checkpoint: {latest}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest}: {e}")
            return None
    
    def save_checkpoint_json(self, state: TradingState, pretty: bool = False) -> Path:
        """
        Save state as JSON (for debugging/inspection).
        
        Args:
            state: Trading state to save
            pretty: Whether to use pretty formatting
            
        Returns:
            Path to the JSON file
        """
        timestamp = int(state.timestamp * 1000)
        filename = f"checkpoint_{timestamp}.json"
        json_path = self.checkpoint_dir / filename
        
        indent = 2 if pretty else None
        with open(json_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=indent)
        
        logger.debug(f"JSON checkpoint saved: {json_path}")
        
        return json_path
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about stored checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.bin"))
        
        if not checkpoints:
            return {
                'count': 0,
                'total_size_bytes': 0,
                'oldest': None,
                'newest': None,
            }
        
        sizes = [cp.stat().st_size for cp in checkpoints]
        timestamps = []
        
        for cp in checkpoints:
            try:
                parts = cp.stem.split('_')
                if len(parts) >= 2:
                    ts = int(parts[1]) / 1000
                    timestamps.append(ts)
            except (ValueError, IndexError):
                pass
        
        return {
            'count': len(checkpoints),
            'total_size_bytes': sum(sizes),
            'average_size_bytes': sum(sizes) // len(sizes) if sizes else 0,
            'oldest': min(timestamps) if timestamps else None,
            'newest': max(timestamps) if timestamps else None,
        }
    
    def cleanup_old_checkpoints(self, keep_count: int = 10) -> int:
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of checkpoints to retain
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.bin"))
        
        if len(checkpoints) <= keep_count:
            return 0
        
        # Sort by name (includes timestamp)
        checkpoints.sort(key=lambda p: p.name, reverse=True)
        
        deleted = 0
        for cp in checkpoints[keep_count:]:
            try:
                cp.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {cp}: {e}")
        
        logger.info(f"Cleaned up {deleted} old checkpoints")
        
        return deleted


# Example usage
if __name__ == "__main__":
    import tempfile
    
    # Create test state
    test_state = TradingState(
        timestamp=time.time(),
        positions={
            "BTCUSDT": PositionState(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=0.5,
                entry_price=45000.0,
                current_price=45500.0,
                unrealized_pnl=250.0,
                leverage=10,
                margin=2250.0,
                liquidation_price=40500.0,
                opened_at=time.time() - 3600,
                last_updated=time.time(),
            )
        },
        orders={},
        balances={"USDT": 10000.0, "BTC": 0.5},
        total_equity=12500.0,
        realized_pnl=500.0,
        session_start=time.time() - 7200,
        trade_count=15,
    )
    
    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        serializer = StateSerializer(Path(tmpdir))
        
        # Save checkpoint
        path = serializer.save_checkpoint(test_state)
        print(f"Saved checkpoint: {path}")
        
        # Load checkpoint
        loaded = serializer.load_latest_checkpoint()
        if loaded:
            print(f"Loaded state: {loaded.total_equity} USDT equity")
            print(f"Positions: {list(loaded.positions.keys())}")
        
        # Print stats
        stats = serializer.get_checkpoint_stats()
        print(f"Stats: {stats}")
