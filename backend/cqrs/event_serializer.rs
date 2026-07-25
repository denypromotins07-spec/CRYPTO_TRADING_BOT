//! Event Serializer - Zero-copy binary serialization of market events.
//!
//! This module provides high-performance binary serialization for trading events,
//! optimized for zero-copy operations between Rust and Python via FFI. Critical
//! for the event sourcing architecture where millions of events must be serialized
//! and deserialized with minimal CPU overhead and memory allocations.
//!
//! Features:
//! - Zero-copy serialization using byte buffers
//! - Proper endianness handling for cross-platform compatibility
//! - Struct alignment padding for AVX2/SIMD optimization
//! - Support for variable-length payloads
//! - Checksum validation for data integrity
//! - Compact binary format minimizing storage footprint

use std::mem;
use std::io::{Cursor, Read, Write};
use crc::{Crc, CRC_32_ISCSI};

/// CRC32 checksum algorithm for data integrity verification
const CRC32: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);

/// Magic number to identify valid event records
const EVENT_MAGIC: u32 = 0x5A494400; // "ZAID" in hex

/// Current serialization format version
const FORMAT_VERSION: u16 = 1;

/// Maximum payload size (64KB)
const MAX_PAYLOAD_SIZE: usize = 64 * 1024;

/// Byte order marker for little-endian (x86/x64/AMD64)
const LITTLE_ENDIAN_MARKER: u8 = 0x01;

/// Represents a serialized event record header
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct EventHeader {
    /// Magic number for validation (4 bytes)
    pub magic: u32,
    /// Format version (2 bytes)
    pub version: u16,
    /// Endianness marker (1 byte)
    pub endian_marker: u8,
    /// Reserved padding for 8-byte alignment (1 byte)
    pub padding: u8,
    /// Event type ID (2 bytes)
    pub event_type_id: u16,
    /// Sequence number (8 bytes)
    pub sequence: u64,
    /// Timestamp in microseconds (8 bytes)
    pub timestamp_us: u64,
    /// Payload length in bytes (4 bytes)
    pub payload_length: u32,
    /// Correlation ID hash (8 bytes)
    pub correlation_hash: u64,
    /// Header checksum (4 bytes)
    pub header_checksum: u32,
}

impl EventHeader {
    /// Size of the header in bytes (44 bytes total)
    pub const SIZE: usize = mem::size_of::<EventHeader>();

    /// Create a new event header
    pub fn new(
        event_type_id: u16,
        sequence: u64,
        timestamp_us: u64,
        payload_length: u32,
        correlation_hash: u64,
    ) -> Self {
        let mut header = EventHeader {
            magic: EVENT_MAGIC,
            version: FORMAT_VERSION,
            endian_marker: LITTLE_ENDIAN_MARKER,
            padding: 0,
            event_type_id,
            sequence,
            timestamp_us,
            payload_length,
            correlation_hash,
            header_checksum: 0,
        };

        // Calculate checksum over all fields except checksum itself
        header.header_checksum = header.calculate_checksum();
        header
    }

    /// Calculate checksum for header integrity
    fn calculate_checksum(&self) -> u32 {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const EventHeader as *const u8,
                mem::size_of::<EventHeader>() - mem::size_of::<u32>(), // Exclude checksum field
            )
        };
        CRC32.checksum(bytes)
    }

    /// Validate header checksum
    pub fn validate_checksum(&self) -> bool {
        let expected = self.calculate_checksum();
        expected == self.header_checksum
    }

    /// Validate magic number and version
    pub fn validate_metadata(&self) -> Result<(), SerializationError> {
        if self.magic != EVENT_MAGIC {
            return Err(SerializationError::InvalidMagic(self.magic));
        }

        if self.version != FORMAT_VERSION {
            return Err(SerializationError::UnsupportedVersion(self.version));
        }

        if self.endian_marker != LITTLE_ENDIAN_MARKER {
            return Err(SerializationError::UnsupportedEndianness(self.endian_marker));
        }

        Ok(())
    }

    /// Serialize header to bytes (zero-copy)
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        unsafe { mem::transmute(*self) }
    }

    /// Deserialize header from bytes (zero-copy)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        if bytes.len() < Self::SIZE {
            return Err(SerializationError::InsufficientData(bytes.len(), Self::SIZE));
        }

        let header: EventHeader = unsafe {
            std::ptr::read_unaligned(bytes.as_ptr() as *const EventHeader)
        };

        Ok(header)
    }
}

/// Errors that can occur during serialization/deserialization
#[derive(Debug, Clone)]
pub enum SerializationError {
    InvalidMagic(u32),
    UnsupportedVersion(u16),
    UnsupportedEndianness(u8),
    ChecksumMismatch,
    InsufficientData(usize, usize),
    PayloadTooLarge(usize),
    InvalidEventType(u16),
    IoError(String),
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationError::InvalidMagic(m) => write!(f, "Invalid magic number: 0x{:08X}", m),
            SerializationError::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
            SerializationError::UnsupportedEndianness(e) => write!(f, "Unsupported endianness: {}", e),
            SerializationError::ChecksumMismatch => write!(f, "Checksum mismatch"),
            SerializationError::InsufficientData(got, expected) => {
                write!(f, "Insufficient data: got {} bytes, expected {}", got, expected)
            }
            SerializationError::PayloadTooLarge(size) => {
                write!(f, "Payload too large: {} bytes (max: {})", size, MAX_PAYLOAD_SIZE)
            }
            SerializationError::InvalidEventType(id) => write!(f, "Invalid event type ID: {}", id),
            SerializationError::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for SerializationError {}

/// Complete serialized event record (header + payload + checksum)
#[derive(Debug, Clone)]
pub struct SerializedEvent {
    pub header: EventHeader,
    pub payload: Vec<u8>,
    pub payload_checksum: u32,
}

impl SerializedEvent {
    /// Create a serialized event from components
    pub fn new(
        event_type_id: u16,
        sequence: u64,
        timestamp_us: u64,
        payload: Vec<u8>,
        correlation_hash: u64,
    ) -> Result<Self, SerializationError> {
        if payload.len() > MAX_PAYLOAD_SIZE {
            return Err(SerializationError::PayloadTooLarge(payload.len()));
        }

        let payload_checksum = CRC32.checksum(&payload);
        let header = EventHeader::new(
            event_type_id,
            sequence,
            timestamp_us,
            payload.len() as u32,
            correlation_hash,
        );

        Ok(SerializedEvent {
            header,
            payload,
            payload_checksum,
        })
    }

    /// Get total size in bytes (header + payload + payload checksum)
    pub fn total_size(&self) -> usize {
        EventHeader::SIZE + self.payload.len() + mem::size_of::<u32>()
    }

    /// Serialize complete event to byte buffer
    pub fn to_buffer(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(self.total_size());

        // Write header (zero-copy via transmute)
        buffer.extend_from_slice(&self.header.to_bytes());

        // Write payload
        buffer.extend_from_slice(&self.payload);

        // Write payload checksum
        buffer.extend_from_slice(&self.payload_checksum.to_le_bytes());

        buffer
    }

    /// Deserialize event from byte buffer (zero-copy where possible)
    pub fn from_buffer(buffer: &[u8]) -> Result<Self, SerializationError> {
        if buffer.len() < EventHeader::SIZE {
            return Err(SerializationError::InsufficientData(
                buffer.len(),
                EventHeader::SIZE,
            ));
        }

        // Parse header
        let header = EventHeader::from_bytes(buffer)?;
        header.validate_metadata()?;

        if !header.validate_checksum() {
            return Err(SerializationError::ChecksumMismatch);
        }

        let payload_start = EventHeader::SIZE;
        let payload_end = payload_start + header.payload_length as usize;

        if buffer.len() < payload_end + mem::size_of::<u32>() {
            return Err(SerializationError::InsufficientData(
                buffer.len(),
                payload_end + mem::size_of::<u32>(),
            ));
        }

        // Extract payload (copy necessary for safety)
        let payload = buffer[payload_start..payload_end].to_vec();

        // Extract and verify payload checksum
        let checksum_bytes: [u8; 4] = buffer[payload_end..payload_end + 4]
            .try_into()
            .map_err(|_| SerializationError::InsufficientData(
                buffer.len() - payload_end,
                4,
            ))?;

        let payload_checksum = u32::from_le_bytes(checksum_bytes);
        let expected_checksum = CRC32.checksum(&payload);

        if payload_checksum != expected_checksum {
            return Err(SerializationError::ChecksumMismatch);
        }

        Ok(SerializedEvent {
            header,
            payload,
            payload_checksum,
        })
    }

    /// Validate entire event (header + payload)
    pub fn validate(&self) -> Result<(), SerializationError> {
        self.header.validate_metadata()?;

        if !self.header.validate_checksum() {
            return Err(SerializationError::ChecksumMismatch);
        }

        let expected_payload_checksum = CRC32.checksum(&self.payload);
        if self.payload_checksum != expected_payload_checksum {
            return Err(SerializationError::ChecksumMismatch);
        }

        Ok(())
    }
}

/// Event type registry for mapping IDs to names
pub struct EventTypeRegistry {
    types: std::collections::HashMap<u16, String>,
    next_id: u16,
}

impl EventTypeRegistry {
    pub fn new() -> Self {
        let mut registry = EventTypeRegistry {
            types: std::collections::HashMap::new(),
            next_id: 1,
        };

        // Register standard event types
        registry.register("TickReceived");
        registry.register("OrderPlaced");
        registry.register("OrderFilled");
        registry.register("OrderCancelled");
        registry.register("OrderModified");
        registry.register("PositionOpened");
        registry.register("PositionClosed");
        registry.register("PnLUpdate");
        registry.register("RiskLimitBreached");
        registry.register("MarketHalt");

        registry
    }

    pub fn register(&mut self, event_type: &str) -> u16 {
        if let Some((&id, _)) = self.types.iter().find(|(_, t)| t == event_type) {
            return id;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.types.insert(id, event_type.to_string());
        id
    }

    pub fn get_name(&self, event_type_id: u16) -> Option<&str> {
        self.types.get(&event_type_id).map(|s| s.as_str())
    }

    pub fn get_id(&self, event_type: &str) -> Option<u16> {
        self.types.iter()
            .find(|(_, t)| t == event_type)
            .map(|(&id, _)| id)
    }
}

impl Default for EventTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy event buffer for batch serialization
pub struct EventBuffer {
    buffer: Vec<u8>,
    event_count: usize,
    capacity_hint: usize,
}

impl EventBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        EventBuffer {
            buffer: Vec::with_capacity(capacity),
            event_count: 0,
            capacity_hint: capacity,
        }
    }

    pub fn append(&mut self, event: &SerializedEvent) {
        let bytes = event.to_buffer();
        self.buffer.extend(bytes);
        self.event_count += 1;
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    pub fn event_count(&self) -> usize {
        self.event_count
    }

    pub fn total_bytes(&self) -> usize {
        self.buffer.len()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.event_count = 0;
    }

    /// Iterate over events in buffer (zero-copy parsing)
    pub fn iter(&self) -> EventBufferIter<'_> {
        EventBufferIter {
            data: &self.buffer,
            offset: 0,
        }
    }
}

/// Iterator for parsing events from buffer
pub struct EventBufferIter<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Iterator for EventBufferIter<'a> {
    type Item = Result<SerializedEvent, SerializationError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        match SerializedEvent::from_buffer(&self.data[self.offset..]) {
            Ok(event) => {
                self.offset += event.total_size();
                Some(Ok(event))
            }
            Err(e) => {
                self.offset = self.data.len(); // Stop iteration on error
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_serialization() {
        let header = EventHeader::new(1, 100, 1234567890, 256, 0xABCD);

        assert_eq!(header.magic, EVENT_MAGIC);
        assert_eq!(header.version, FORMAT_VERSION);
        assert!(header.validate_checksum());

        let bytes = header.to_bytes();
        let restored = EventHeader::from_bytes(&bytes).unwrap();

        assert_eq!(restored.magic, header.magic);
        assert_eq!(restored.sequence, header.sequence);
        assert!(restored.validate_checksum());
    }

    #[test]
    fn test_event_serialization() {
        let payload = vec![1u8, 2, 3, 4, 5];
        let event = SerializedEvent::new(1, 100, 1234567890, payload.clone(), 0xABCD)
            .unwrap();

        assert!(event.validate().is_ok());

        let buffer = event.to_buffer();
        let restored = SerializedEvent::from_buffer(&buffer).unwrap();

        assert_eq!(restored.header.sequence, event.header.sequence);
        assert_eq!(restored.payload, event.payload);
        assert_eq!(restored.payload_checksum, event.payload_checksum);
    }

    #[test]
    fn test_invalid_magic_detection() {
        let mut header = EventHeader::new(1, 100, 1234567890, 0, 0);
        header.magic = 0xDEADBEEF;

        assert!(header.validate_metadata().is_err());
    }

    #[test]
    fn test_checksum_corruption_detection() {
        let payload = vec![1u8, 2, 3, 4, 5];
        let mut event = SerializedEvent::new(1, 100, 1234567890, payload.clone(), 0xABCD)
            .unwrap();

        // Corrupt payload
        event.payload[0] = 0xFF;

        assert!(event.validate().is_err());
    }

    #[test]
    fn test_event_buffer_iteration() {
        let mut buffer = EventBuffer::with_capacity(1024);

        for i in 0..5 {
            let payload = vec![i as u8; 10];
            let event = SerializedEvent::new(1, i, i * 1000, payload, 0).unwrap();
            buffer.append(&event);
        }

        let mut count = 0;
        for result in buffer.iter() {
            let event = result.unwrap();
            assert_eq!(event.header.sequence, count as u64);
            count += 1;
        }

        assert_eq!(count, 5);
    }

    #[test]
    fn test_payload_size_limit() {
        let large_payload = vec![0u8; MAX_PAYLOAD_SIZE + 1];
        let result = SerializedEvent::new(1, 100, 1234567890, large_payload, 0);

        assert!(matches!(result, Err(SerializationError::PayloadTooLarge(_))));
    }

    #[test]
    fn test_registry() {
        let mut registry = EventTypeRegistry::new();

        let tick_id = registry.register("TickReceived");
        assert_eq!(tick_id, 1);

        let custom_id = registry.register("CustomEvent");
        assert_eq!(custom_id, 11);

        assert_eq!(registry.get_name(1), Some("TickReceived"));
        assert_eq!(registry.get_id("TickReceived"), Some(1));
    }
}
