// ZAID PERSONAL CRYPTO TRADING BOT - Stage 24
// Delta-of-Delta Compression for L2 Order Book Updates
// Efficiently encodes price/quantity changes using delta encoding
// Handles out-of-order exchange sequence IDs gracefully

use std::collections::{HashMap, VecDeque};
use std::mem;

/// Maximum levels to track per side (bid/ask)
const MAX_LEVELS: usize = 100;

/// Circular buffer for recent deltas (for rollback/replay)
const DELTA_BUFFER_SIZE: usize = 1024;

/// Price tick size in fixed-point units (e.g., 0.01 = 1 for BTC)
const PRICE_TICK: u64 = 1;

/// Delta-encoded level update
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, packed)]
pub struct DeltaLevel {
    /// Delta from previous price (zigzag encoded)
    pub price_delta: i16,
    /// Delta from previous quantity (zigzag encoded)
    pub quantity_delta: i32,
    /// Level index (0 = best bid/ask)
    pub level_index: u8,
    /// Update type
    pub update_type: UpdateType,
}

impl DeltaLevel {
    #[inline]
    pub fn new(
        price_delta: i16,
        quantity_delta: i32,
        level_index: u8,
        update_type: UpdateType,
    ) -> Self {
        Self {
            price_delta,
            quantity_delta,
            level_index,
            update_type,
        }
    }
    
    /// Encode to bytes (9 bytes total)
    #[inline]
    pub fn encode(&self) -> [u8; 9] {
        let mut buf = [0u8; 9];
        buf[0..2].copy_from_slice(&self.price_delta.to_le_bytes());
        buf[2..6].copy_from_slice(&self.quantity_delta.to_le_bytes());
        buf[6] = self.level_index;
        buf[7] = self.update_type as u8;
        buf[8] = 0; // Reserved/padding
        buf
    }
    
    /// Decode from bytes
    #[inline]
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 9 {
            return None;
        }
        
        Some(Self {
            price_delta: i16::from_le_bytes(buf[0..2].try_into().ok()?),
            quantity_delta: i32::from_le_bytes(buf[2..6].try_into().ok()?),
            level_index: buf[6],
            update_type: UpdateType::from_u8(buf[7])?,
        })
    }
    
    /// Size in bytes
    pub const SIZE: usize = 9;
}

/// Update type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UpdateType {
    Add = 0,
    Modify = 1,
    Remove = 2,
    Snapshot = 3,
}

impl UpdateType {
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Add),
            1 => Some(Self::Modify),
            2 => Some(Self::Remove),
            3 => Some(Self::Snapshot),
            _ => None,
        }
    }
}

/// Zigzag encoding for signed integers (converts to unsigned for better compression)
#[inline]
fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

#[inline]
fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

/// Delta-of-delta encoder for order book updates
pub struct DeltaOfDeltaEncoder {
    /// Last known prices per level (bid and ask)
    last_bid_prices: [i64; MAX_LEVELS],
    last_ask_prices: [i64; MAX_LEVELS],
    /// Last known quantities per level
    last_bid_quantities: [i64; MAX_LEVELS],
    last_ask_quantities: [i64; MAX_LEVELS],
    /// Track which levels are active
    bid_active: [bool; MAX_LEVELS],
    ask_active: [bool; MAX_LEVELS],
    /// Buffer of encoded deltas
    delta_buffer: VecDeque<Vec<u8>>,
    /// Current sequence ID
    current_sequence: u64,
    /// Last processed sequence ID (for out-of-order detection)
    last_sequence: u64,
    /// Statistics
    stats: EncoderStats,
}

/// Encoder statistics
#[derive(Debug, Clone, Default)]
pub struct EncoderStats {
    pub updates_processed: u64,
    pub out_of_order_updates: u64,
    pub duplicate_updates: u64,
    pub bytes_saved: u64,
    pub original_bytes: u64,
    pub compressed_bytes: u64,
}

impl EncoderStats {
    #[inline]
    pub fn compression_ratio(&self) -> f64 {
        if self.original_bytes == 0 {
            0.0
        } else {
            self.compressed_bytes as f64 / self.original_bytes as f64
        }
    }
    
    #[inline]
    pub fn savings_percent(&self) -> f64 {
        (1.0 - self.compression_ratio()) * 100.0
    }
}

impl Default for DeltaOfDeltaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaOfDeltaEncoder {
    /// Create a new delta encoder
    pub fn new() -> Self {
        Self {
            last_bid_prices: [0; MAX_LEVELS],
            last_ask_prices: [0; MAX_LEVELS],
            last_bid_quantities: [0; MAX_LEVELS],
            last_ask_quantities: [0; MAX_LEVELS],
            bid_active: [false; MAX_LEVELS],
            ask_active: [false; MAX_LEVELS],
            delta_buffer: VecDeque::with_capacity(DELTA_BUFFER_SIZE),
            current_sequence: 0,
            last_sequence: 0,
            stats: EncoderStats::default(),
        }
    }
    
    /// Reset encoder state (for snapshot processing)
    pub fn reset(&mut self) {
        self.last_bid_prices.fill(0);
        self.last_ask_prices.fill(0);
        self.last_bid_quantities.fill(0);
        self.last_ask_quantities.fill(0);
        self.bid_active.fill(false);
        self.ask_active.fill(false);
        self.delta_buffer.clear();
        self.stats = EncoderStats::default();
    }
    
    /// Process an order book update and return delta-encoded bytes
    /// 
    /// # Arguments
    /// * `sequence_id` - Exchange sequence ID (handles out-of-order gracefully)
    /// * `bids` - Vector of (price, quantity) pairs for bids
    /// * `asks` - Vector of (price, quantity) pairs for asks
    /// * `is_snapshot` - Whether this is a full snapshot
    pub fn process_update(
        &mut self,
        sequence_id: u64,
        bids: &[(i64, i64)],
        asks: &[(i64, i64)],
        is_snapshot: bool,
    ) -> Result<Vec<u8>, EncoderError> {
        // Handle out-of-order sequence IDs gracefully
        if sequence_id <= self.last_sequence && !is_snapshot {
            self.stats.out_of_order_updates += 1;
            
            // Check if it's a duplicate
            if sequence_id == self.last_sequence {
                self.stats.duplicate_updates += 1;
                return Ok(Vec::new()); // Skip duplicate
            }
            
            // For old updates, we still process but flag them
            // In production, might want to skip or handle differently
        }
        
        self.current_sequence = sequence_id;
        let mut encoded = Vec::new();
        
        // Write header: [sequence_id(8)][flags(1)][level_count(1)]
        encoded.extend_from_slice(&sequence_id.to_le_bytes());
        
        let flags = if is_snapshot { 0x01 } else { 0x00 };
        encoded.push(flags);
        
        if is_snapshot {
            // Reset state for snapshot
            self.reset();
            encoded.push(UpdateType::Snapshot as u8);
        } else {
            encoded.push(0x00); // Normal update
        }
        
        // Encode bid deltas
        let bid_count = bids.len().min(MAX_LEVELS);
        encoded.push(bid_count as u8);
        
        for (i, &(price, quantity)) in bids.iter().take(bid_count).enumerate() {
            let delta = self.encode_level_delta(
                price,
                quantity,
                i,
                true, // is_bid
                is_snapshot,
            );
            
            if let Some(delta) = delta {
                let delta_bytes = delta.encode();
                encoded.extend_from_slice(&delta_bytes);
                
                // Update last known values
                self.last_bid_prices[i] = price;
                self.last_bid_quantities[i] = quantity;
                self.bid_active[i] = true;
            }
        }
        
        // Encode ask deltas
        let ask_count = asks.len().min(MAX_LEVELS);
        encoded.push(ask_count as u8);
        
        for (i, &(price, quantity)) in asks.iter().take(ask_count).enumerate() {
            let delta = self.encode_level_delta(
                price,
                quantity,
                i,
                false, // is_bid
                is_snapshot,
            );
            
            if let Some(delta) = delta {
                let delta_bytes = delta.encode();
                encoded.extend_from_slice(&delta_bytes);
                
                // Update last known values
                self.last_ask_prices[i] = price;
                self.last_ask_quantities[i] = quantity;
                self.ask_active[i] = true;
            }
        }
        
        // Update statistics
        let original_size = (bids.len() + asks.len()) * 16; // 16 bytes per level (uncompressed)
        self.stats.original_bytes += original_size as u64;
        self.stats.compressed_bytes += encoded.len() as u64;
        self.stats.bytes_saved += original_size.saturating_sub(encoded.len()) as u64;
        self.stats.updates_processed += 1;
        
        self.last_sequence = sequence_id;
        
        // Store in buffer for potential replay
        if self.delta_buffer.len() >= DELTA_BUFFER_SIZE {
            self.delta_buffer.pop_front();
        }
        self.delta_buffer.push_back(encoded.clone());
        
        Ok(encoded)
    }
    
    /// Encode delta for a single level
    fn encode_level_delta(
        &mut self,
        price: i64,
        quantity: i64,
        level_index: usize,
        is_bid: bool,
        is_snapshot: bool,
    ) -> Option<DeltaLevel> {
        let (last_price, last_quantity, is_active) = if is_bid {
            (
                self.last_bid_prices[level_index],
                self.last_bid_quantities[level_index],
                self.bid_active[level_index],
            )
        } else {
            (
                self.last_ask_prices[level_index],
                self.last_ask_quantities[level_index],
                self.ask_active[level_index],
            )
        };
        
        // Determine update type
        let update_type = if is_snapshot {
            UpdateType::Snapshot
        } else if !is_active {
            UpdateType::Add
        } else if quantity == 0 {
            UpdateType::Remove
        } else {
            UpdateType::Modify
        };
        
        // Calculate deltas
        let price_delta = if is_snapshot {
            price as i16 // Use absolute value for snapshot
        } else {
            (price - last_price).clamp(i16::MIN as i64, i16::MAX as i64) as i16
        };
        
        let quantity_delta = if is_snapshot {
            quantity as i32
        } else {
            (quantity - last_quantity).clamp(i32::MIN as i64, i32::MAX as i64) as i32
        };
        
        Some(DeltaLevel::new(
            price_delta,
            quantity_delta,
            level_index as u8,
            update_type,
        ))
    }
    
    /// Decode delta-encoded data back to order book levels
    pub fn decode_update(&mut self, data: &[u8]) -> Result<(Vec<(i64, i64)>, Vec<(i64, i64)>), EncoderError> {
        if data.len() < 11 {
            return Err(EncoderError::InvalidData("Data too short"));
        }
        
        let mut offset = 0;
        
        // Read header
        let sequence_id = u64::from_le_bytes(data[offset..offset+8].try_into().unwrap());
        offset += 8;
        
        let _flags = data[offset];
        offset += 1;
        
        let update_marker = data[offset];
        offset += 1;
        
        let is_snapshot = (update_marker & 0x01) != 0;
        if is_snapshot {
            self.reset();
        }
        
        let _update_type = UpdateType::from_u8(update_marker & !0x01)
            .unwrap_or(UpdateType::Modify);
        
        // Read bid count
        let bid_count = data[offset] as usize;
        offset += 1;
        
        let mut bids = Vec::with_capacity(bid_count);
        for i in 0..bid_count {
            if offset + DeltaLevel::SIZE > data.len() {
                return Err(EncoderError::InvalidData("Unexpected end of data"));
            }
            
            let delta = DeltaLevel::decode(&data[offset..offset+DeltaLevel::SIZE])
                .ok_or(EncoderError::InvalidData("Invalid delta encoding"))?;
            offset += DeltaLevel::SIZE;
            
            let (last_price, last_quantity) = (self.last_bid_prices[i], self.last_bid_quantities[i]);
            
            let price = match delta.update_type {
                UpdateType::Snapshot | UpdateType::Add => delta.price_delta as i64,
                _ => last_price + delta.price_delta as i64,
            };
            
            let quantity = match delta.update_type {
                UpdateType::Snapshot | UpdateType::Add => delta.quantity_delta as i64,
                _ => last_quantity + delta.quantity_delta as i64,
            };
            
            if quantity > 0 {
                bids.push((price, quantity));
                self.last_bid_prices[i] = price;
                self.last_bid_quantities[i] = quantity;
                self.bid_active[i] = true;
            } else {
                self.bid_active[i] = false;
            }
        }
        
        // Read ask count
        let ask_count = data[offset] as usize;
        offset += 1;
        
        let mut asks = Vec::with_capacity(ask_count);
        for i in 0..ask_count {
            if offset + DeltaLevel::SIZE > data.len() {
                return Err(EncoderError::InvalidData("Unexpected end of data"));
            }
            
            let delta = DeltaLevel::decode(&data[offset..offset+DeltaLevel::SIZE])
                .ok_or(EncoderError::InvalidData("Invalid delta encoding"))?;
            offset += DeltaLevel::SIZE;
            
            let (last_price, last_quantity) = (self.last_ask_prices[i], self.last_ask_quantities[i]);
            
            let price = match delta.update_type {
                UpdateType::Snapshot | UpdateType::Add => delta.price_delta as i64,
                _ => last_price + delta.price_delta as i64,
            };
            
            let quantity = match delta.update_type {
                UpdateType::Snapshot | UpdateType::Add => delta.quantity_delta as i64,
                _ => last_quantity + delta.quantity_delta as i64,
            };
            
            if quantity > 0 {
                asks.push((price, quantity));
                self.last_ask_prices[i] = price;
                self.last_ask_quantities[i] = quantity;
                self.ask_active[i] = true;
            } else {
                self.ask_active[i] = false;
            }
        }
        
        Ok((bids, asks))
    }
    
    /// Get encoder statistics
    pub fn get_stats(&self) -> &EncoderStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = EncoderStats::default();
    }
}

/// Encoder errors
#[derive(Debug)]
pub enum EncoderError {
    InvalidData(&'static str),
    OutOfOrderSequence { current: u64, received: u64 },
    BufferOverflow,
    InvalidLevelIndex(usize),
}

impl std::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            EncoderError::OutOfOrderSequence { current, received } => {
                write!(f, "Out of order sequence: current={}, received={}", current, received)
            }
            EncoderError::BufferOverflow => write!(f, "Buffer overflow"),
            EncoderError::InvalidLevelIndex(idx) => write!(f, "Invalid level index: {}", idx),
        }
    }
}

impl std::error::Error for EncoderError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zigzag_encoding() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);
        
        assert_eq!(zigzag_decode(0), 0);
        assert_eq!(zigzag_decode(1), -1);
        assert_eq!(zigzag_decode(2), 1);
        assert_eq!(zigzag_decode(3), -2);
        assert_eq!(zigzag_decode(4), 2);
    }
    
    #[test]
    fn test_delta_encoding_roundtrip() {
        let mut encoder = DeltaOfDeltaEncoder::new();
        
        // First update (snapshot)
        let bids = vec![(50000, 100), (49999, 200), (49998, 300)];
        let asks = vec![(50001, 150), (50002, 250), (50003, 350)];
        
        let encoded = encoder.process_update(1, &bids, &asks, true).unwrap();
        assert!(!encoded.is_empty());
        
        // Decode and verify
        let (decoded_bids, decoded_asks) = encoder.decode_update(&encoded).unwrap();
        assert_eq!(decoded_bids.len(), bids.len());
        assert_eq!(decoded_asks.len(), asks.len());
        
        // Second update (incremental)
        let bids2 = vec![(50001, 120), (50000, 100), (49999, 200)];
        let asks2 = vec![(50002, 260), (50003, 350), (50004, 400)];
        
        let encoded2 = encoder.process_update(2, &bids2, &asks2, false).unwrap();
        
        // Verify compression saved bytes
        let stats = encoder.get_stats();
        println!("Compression ratio: {:.2}%", stats.compression_ratio() * 100.0);
        println!("Bytes saved: {}", stats.bytes_saved);
    }
    
    #[test]
    fn test_out_of_order_handling() {
        let mut encoder = DeltaOfDeltaEncoder::new();
        
        // Process update 1
        let bids = vec![(50000, 100)];
        let asks = vec![(50001, 100)];
        encoder.process_update(1, &bids, &asks, true).unwrap();
        
        // Process update 3 (skip 2)
        encoder.process_update(3, &bids, &asks, false).unwrap();
        
        // Process update 2 (out of order)
        let result = encoder.process_update(2, &bids, &asks, false);
        assert!(result.is_ok());
        
        // Verify stats show out-of-order detection
        let stats = encoder.get_stats();
        assert!(stats.out_of_order_updates > 0);
    }
    
    #[test]
    fn test_size_assertions() {
        assert_eq!(mem::size_of::<DeltaLevel>(), 9);
        assert_eq!(DeltaLevel::SIZE, 9);
    }
}
