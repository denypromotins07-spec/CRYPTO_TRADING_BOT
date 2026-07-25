#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 24
Endianness Handler for Cross-Platform Byte Order Consistency
Ensures correct byte ordering across Windows (x86_64), Linux, and macOS
Handles conversion between host byte order and network byte order
"""

from __future__ import annotations

import struct
import sys
import logging
from typing import TypeVar, Union, List, Tuple
from enum import IntEnum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
NumericType = TypeVar('NumericType', int, float)

class ByteOrder(IntEnum):
    """Byte order enumeration."""
    LITTLE_ENDIAN = 0  # Least significant byte first (x86, x64)
    BIG_ENDIAN = 1     # Most significant byte first (network order)


# Detect system byte order
def get_system_byte_order() -> ByteOrder:
    """Detect the native byte order of the current system."""
    if sys.byteorder == 'little':
        return ByteOrder.LITTLE_ENDIAN
    else:
        return ByteOrder.BIG_ENDIAN


# System byte order constant
SYSTEM_BYTE_ORDER = get_system_byte_order()
NETWORK_BYTE_ORDER = ByteOrder.BIG_ENDIAN  # Network order is always big-endian

# Format character mappings for struct module
FORMAT_MAP = {
    'int8': 'b',
    'uint8': 'B',
    'int16': 'h',
    'uint16': 'H',
    'int32': 'i',
    'uint32': 'I',
    'int64': 'q',
    'uint64': 'Q',
    'float32': 'f',
    'float64': 'd',
}

# Size mappings in bytes
SIZE_MAP = {
    'int8': 1,
    'uint8': 1,
    'int16': 2,
    'uint16': 2,
    'int32': 4,
    'uint32': 4,
    'int64': 8,
    'uint64': 8,
    'float32': 4,
    'float64': 8,
}


class EndiannessHandler:
    """
    Handles byte order conversions for cross-platform compatibility.
    Ensures consistent data representation regardless of host architecture.
    """
    
    def __init__(self, target_order: ByteOrder = NETWORK_BYTE_ORDER):
        """
        Initialize with target byte order.
        
        Args:
            target_order: Target byte order for serialization (default: network/big-endian)
        """
        self.target_order = target_order
        self.is_native_little = SYSTEM_BYTE_ORDER == ByteOrder.LITTLE_ENDIAN
        
        # Pre-compute format prefixes
        if self.target_order == ByteOrder.BIG_ENDIAN:
            self.format_prefix = '>'
        else:
            self.format_prefix = '<'
            
        logger.info(f"EndiannessHandler initialized: system={SYSTEM_BYTE_ORDER.name}, "
                   f"target={target_order.name}")
    
    def needs_conversion(self) -> bool:
        """Check if conversion is needed based on system vs target byte order."""
        if self.target_order == ByteOrder.BIG_ENDIAN:
            return self.is_native_little
        else:
            return not self.is_native_little
    
    def pack(self, value: NumericType, dtype: str) -> bytes:
        """
        Pack a numeric value into bytes with target byte order.
        
        Args:
            value: Numeric value to pack
            dtype: Data type string ('int16', 'uint32', 'float64', etc.)
            
        Returns:
            Bytes representation in target byte order
            
        Raises:
            ValueError: If dtype is not supported
        """
        if dtype not in FORMAT_MAP:
            raise ValueError(f"Unsupported data type: {dtype}")
            
        fmt = f"{self.format_prefix}{FORMAT_MAP[dtype]}"
        try:
            return struct.pack(fmt, value)
        except struct.error as e:
            raise ValueError(f"Failed to pack value {value} as {dtype}: {e}")
    
    def unpack(self, data: bytes, dtype: str) -> NumericType:
        """
        Unpack bytes to numeric value assuming target byte order.
        
        Args:
            data: Bytes to unpack
            dtype: Data type string
            
        Returns:
            Unpacked numeric value
            
        Raises:
            ValueError: If dtype is not supported or data length is incorrect
        """
        if dtype not in FORMAT_MAP:
            raise ValueError(f"Unsupported data type: {dtype}")
            
        expected_size = SIZE_MAP[dtype]
        if len(data) != expected_size:
            raise ValueError(f"Expected {expected_size} bytes for {dtype}, got {len(data)}")
            
        fmt = f"{self.format_prefix}{FORMAT_MAP[dtype]}"
        try:
            return struct.unpack(fmt, data)[0]
        except struct.error as e:
            raise ValueError(f"Failed to unpack data as {dtype}: {e}")
    
    def pack_array(self, values: List[NumericType], dtype: str) -> bytes:
        """
        Pack an array of values into bytes.
        
        Args:
            values: List of numeric values
            dtype: Data type string
            
        Returns:
            Packed bytes array
        """
        if not values:
            return b''
            
        if dtype not in FORMAT_MAP:
            raise ValueError(f"Unsupported data type: {dtype}")
            
        fmt = f"{self.format_prefix}{len(values)}{FORMAT_MAP[dtype]}"
        try:
            return struct.pack(fmt, *values)
        except struct.error as e:
            raise ValueError(f"Failed to pack array as {dtype}: {e}")
    
    def unpack_array(self, data: bytes, dtype: str, count: int) -> List[NumericType]:
        """
        Unpack bytes to array of numeric values.
        
        Args:
            data: Bytes to unpack
            dtype: Data type string
            count: Number of elements to unpack
            
        Returns:
            List of unpacked values
        """
        if count == 0:
            return []
            
        if dtype not in FORMAT_MAP:
            raise ValueError(f"Unsupported data type: {dtype}")
            
        expected_size = SIZE_MAP[dtype] * count
        if len(data) != expected_size:
            raise ValueError(f"Expected {expected_size} bytes, got {len(data)}")
            
        fmt = f"{self.format_prefix}{count}{FORMAT_MAP[dtype]}"
        try:
            return list(struct.unpack(fmt, data))
        except struct.error as e:
            raise ValueError(f"Failed to unpack array as {dtype}: {e}")
    
    def convert_to_network_order(self, data: bytes, dtype: str) -> bytes:
        """
        Convert bytes from host order to network order (big-endian).
        
        Args:
            data: Bytes in host byte order
            dtype: Data type string
            
        Returns:
            Bytes in network byte order
        """
        if self.target_order != ByteOrder.BIG_ENDIAN:
            raise ValueError("This method requires target_order to be BIG_ENDIAN")
            
        # If system is already big-endian, no conversion needed
        if not self.is_native_little:
            return data
            
        # Reverse bytes for each element
        element_size = SIZE_MAP[dtype]
        result = bytearray()
        
        for i in range(0, len(data), element_size):
            element = data[i:i + element_size]
            result.extend(reversed(element))
            
        return bytes(result)
    
    def convert_from_network_order(self, data: bytes, dtype: str) -> bytes:
        """
        Convert bytes from network order to host order.
        
        Args:
            data: Bytes in network byte order
            dtype: Data type string
            
        Returns:
            Bytes in host byte order
        """
        # Network order is big-endian, so reverse if host is little-endian
        if not self.is_native_little:
            return data
            
        element_size = SIZE_MAP[dtype]
        result = bytearray()
        
        for i in range(0, len(data), element_size):
            element = data[i:i + element_size]
            result.extend(reversed(element))
            
        return bytes(result)


class BinaryProtocolHandler:
    """
    High-level handler for binary protocol serialization/deserialization.
    Manages complex message structures with proper endianness handling.
    """
    
    def __init__(self):
        self.endianness = EndiannessHandler(target_order=ByteOrder.BIG_ENDIAN)
        self.buffer = bytearray()
        
    def write_uint8(self, value: int) -> BinaryProtocolHandler:
        """Write uint8 to buffer."""
        self.buffer.extend(self.endianness.pack(value, 'uint8'))
        return self
    
    def write_uint16(self, value: int) -> BinaryProtocolHandler:
        """Write uint16 to buffer."""
        self.buffer.extend(self.endianness.pack(value, 'uint16'))
        return self
    
    def write_uint32(self, value: int) -> BinaryProtocolHandler:
        """Write uint32 to buffer."""
        self.buffer.extend(self.endianness.pack(value, 'uint32'))
        return self
    
    def write_uint64(self, value: int) -> BinaryProtocolHandler:
        """Write uint64 to buffer."""
        self.buffer.extend(self.endianness.pack(value, 'uint64'))
        return self
    
    def write_int64(self, value: int) -> BinaryProtocolHandler:
        """Write int64 to buffer."""
        self.buffer.extend(self.endianness.pack(value, 'int64'))
        return self
    
    def write_float64(self, value: float) -> BinaryProtocolHandler:
        """Write float64 to buffer."""
        self.buffer.extend(self.endianness.pack(value, 'float64'))
        return self
    
    def write_string(self, value: str) -> BinaryProtocolHandler:
        """Write length-prefixed string to buffer."""
        encoded = value.encode('utf-8')
        self.write_uint32(len(encoded))
        self.buffer.extend(encoded)
        return self
    
    def read_uint8(self, offset: int) -> Tuple[int, int]:
        """Read uint8 from buffer at offset. Returns (value, new_offset)."""
        value = self.endianness.unpack(self.buffer[offset:offset+1], 'uint8')
        return value, offset + 1
    
    def read_uint16(self, offset: int) -> Tuple[int, int]:
        """Read uint16 from buffer at offset. Returns (value, new_offset)."""
        value = self.endianness.unpack(self.buffer[offset:offset+2], 'uint16')
        return value, offset + 2
    
    def read_uint32(self, offset: int) -> Tuple[int, int]:
        """Read uint32 from buffer at offset. Returns (value, new_offset)."""
        value = self.endianness.unpack(self.buffer[offset:offset+4], 'uint32')
        return value, offset + 4
    
    def read_uint64(self, offset: int) -> Tuple[int, int]:
        """Read uint64 from buffer at offset. Returns (value, new_offset)."""
        value = self.endianness.unpack(self.buffer[offset:offset+8], 'uint64')
        return value, offset + 8
    
    def read_int64(self, offset: int) -> Tuple[int, int]:
        """Read int64 from buffer at offset. Returns (value, new_offset)."""
        value = self.endianness.unpack(self.buffer[offset:offset+8], 'int64')
        return value, offset + 8
    
    def read_float64(self, offset: int) -> Tuple[float, int]:
        """Read float64 from buffer at offset. Returns (value, new_offset)."""
        value = self.endianness.unpack(self.buffer[offset:offset+8], 'float64')
        return value, offset + 8
    
    def read_string(self, offset: int) -> Tuple[str, int]:
        """Read length-prefixed string from buffer. Returns (value, new_offset)."""
        length, offset = self.read_uint32(offset)
        value = self.buffer[offset:offset+length].decode('utf-8')
        return value, offset + length
    
    def get_buffer(self) -> bytes:
        """Get current buffer contents."""
        return bytes(self.buffer)
    
    def clear(self) -> BinaryProtocolHandler:
        """Clear the buffer."""
        self.buffer.clear()
        return self
    
    def __len__(self) -> int:
        return len(self.buffer)


def test_sign_extension_prevention():
    """Test that unsigned types prevent sign extension bugs."""
    handler = EndiannessHandler()
    
    # Test maximum uint16 value (should not be interpreted as negative)
    max_uint16 = 0xFFFF
    packed = handler.pack(max_uint16, 'uint16')
    unpacked = handler.unpack(packed, 'uint16')
    assert unpacked == max_uint16, f"Sign extension bug detected: {unpacked}"
    
    # Test maximum uint32 value
    max_uint32 = 0xFFFFFFFF
    packed = handler.pack(max_uint32, 'uint32')
    unpacked = handler.unpack(packed, 'uint32')
    assert unpacked == max_uint32, f"Sign extension bug detected: {unpacked}"
    
    # Test maximum uint64 value
    max_uint64 = 0xFFFFFFFFFFFFFFFF
    packed = handler.pack(max_uint64, 'uint64')
    unpacked = handler.unpack(packed, 'uint64')
    assert unpacked == max_uint64, f"Sign extension bug detected: {unpacked}"
    
    logger.info("All sign extension tests passed!")


def main():
    """Example usage and testing."""
    print(f"System byte order: {SYSTEM_BYTE_ORDER.name}")
    print(f"Network byte order: {NETWORK_BYTE_ORDER.name}")
    
    # Create handler
    handler = EndiannessHandler()
    
    # Test packing/unpacking
    test_value = 0x12345678
    packed = handler.pack(test_value, 'uint32')
    unpacked = handler.unpack(packed, 'uint32')
    
    print(f"Original: 0x{test_value:08X}")
    print(f"Packed: {packed.hex()}")
    print(f"Unpacked: 0x{unpacked:08X}")
    print(f"Match: {test_value == unpacked}")
    
    # Test protocol handler
    protocol = BinaryProtocolHandler()
    protocol.write_uint32(12345)\
            .write_float64(3.14159)\
            .write_string("BTCUSDT")
    
    buffer = protocol.get_buffer()
    print(f"\nProtocol buffer ({len(buffer)} bytes): {buffer.hex()}")
    
    # Read back
    val1, off = protocol.read_uint32(0)
    val2, off = protocol.read_float64(off)
    val3, off = protocol.read_string(off)
    
    print(f"Read back: {val1}, {val2:.5f}, {val3}")
    
    # Run sign extension tests
    test_sign_extension_prevention()
    
    print("\nEndianness handler ready for cross-platform operation!")


if __name__ == "__main__":
    main()
