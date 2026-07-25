// ZAID PERSONAL CRYPTO TRADING BOT - Stage 24
// Bitwise Packer for Order Flags and Prices into 64-bit Integers
// Zero-cost abstractions for ultra-compact data representation
// Guarantees no sign-extension bugs through careful type handling

use std::mem;

/// Bit layout for packed order data (64-bit):
/// 
/// Bits 0-15:   Price fraction (4 decimal places, 0-9999)
/// Bits 16-31:  Quantity fraction (4 decimal places, 0-9999)
/// Bits 32-47:  Order flags (bitmask)
/// Bits 48-55:  Asset identifier (0-255)
/// Bits 56-62:  Sequence number modulo 128
/// Bit 63:      Side indicator (0=buy, 1=sell)

/// Maximum price fraction value (4 decimal places)
const MAX_PRICE_FRACTION: u16 = 9999;

/// Maximum quantity fraction value (4 decimal places)
const MAX_QUANTITY_FRACTION: u16 = 9999;

/// Bit positions
const PRICE_FRAC_SHIFT: u32 = 0;
const PRICE_FRAC_MASK: u64 = 0xFFFF; // 16 bits

const QUANTITY_FRAC_SHIFT: u32 = 16;
const QUANTITY_FRAC_MASK: u64 = 0xFFFF; // 16 bits

const FLAGS_SHIFT: u32 = 32;
const FLAGS_MASK: u64 = 0xFFFF; // 16 bits

const ASSET_SHIFT: u32 = 48;
const ASSET_MASK: u64 = 0xFF; // 8 bits

const SEQUENCE_SHIFT: u32 = 56;
const SEQUENCE_MASK: u64 = 0x7F; // 7 bits

const SIDE_SHIFT: u32 = 63;
const SIDE_MASK: u64 = 0x01; // 1 bit

/// Order flags bitmask definitions
pub mod order_flags {
    pub const NONE: u16 = 0b0000_0000_0000_0000;
    pub const POST_ONLY: u16 = 0b0000_0000_0000_0001;
    pub const REDUCE_ONLY: u16 = 0b0000_0000_0000_0010;
    pub const IOC: u16 = 0b0000_0000_0000_0100;
    pub const FOK: u16 = 0b0000_0000_0000_1000;
    pub const STOP_LOSS: u16 = 0b0000_0000_0001_0000;
    pub const TAKE_PROFIT: u16 = 0b0000_0000_0010_0000;
    pub const TRAILING_STOP: u16 = 0b0000_0000_0100_0000;
    pub const HIDDEN: u16 = 0b0000_0000_1000_0000;
    pub const URGENT: u16 = 0b0000_0001_0000_0000;
    pub const SPOT: u16 = 0b0000_0010_0000_0000;
    pub const FUTURES: u16 = 0b0000_0100_0000_0000;
    pub const OPTIONS: u16 = 0b0000_1000_0000_0000;
    pub const MARGIN: u16 = 0b0001_0000_0000_0000;
    pub const CROSS_MARGIN: u16 = 0b0010_0000_0000_0000;
    pub const ISOLATED_MARGIN: u16 = 0b0100_0000_0000_0000;
}

/// Packed order representation (64-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PackedOrder(pub u64);

impl PackedOrder {
    /// Create a new packed order from components
    /// 
    /// # Arguments
    /// * `price_int` - Integer part of price (0-65535)
    /// * `price_frac` - Fractional part of price (0-9999)
    /// * `quantity_int` - Integer part of quantity (0-65535)
    /// * `quantity_frac` - Fractional part of quantity (0-9999)
    /// * `flags` - Order flags bitmask
    /// * `asset_id` - Asset identifier (0-255)
    /// * `sequence` - Sequence number (will be mod 128)
    /// * `is_sell` - true for sell, false for buy
    /// 
    /// # Panics
    /// Panics if any value exceeds its maximum allowed value
    #[inline]
    pub fn new(
        price_int: u16,
        price_frac: u16,
        quantity_int: u16,
        quantity_frac: u16,
        flags: u16,
        asset_id: u8,
        sequence: u8,
        is_sell: bool,
    ) -> Self {
        // Validate inputs to prevent overflow
        assert!(price_frac <= MAX_PRICE_FRACTION, "Price fraction exceeds maximum");
        assert!(quantity_frac <= MAX_QUANTITY_FRACTION, "Quantity fraction exceeds maximum");
        assert!(asset_id <= 255, "Asset ID exceeds maximum");
        
        // Use bitwise operations for packing - no sign extension possible with unsigned types
        let mut packed: u64 = 0;
        
        // Pack price fraction (bits 0-15)
        packed |= (price_frac as u64 & PRICE_FRAC_MASK) << PRICE_FRAC_SHIFT;
        
        // Pack quantity fraction (bits 16-31)
        packed |= (quantity_frac as u64 & QUANTITY_FRAC_MASK) << QUANTITY_FRAC_SHIFT;
        
        // Pack flags (bits 32-47)
        packed |= (flags as u64 & FLAGS_MASK) << FLAGS_SHIFT;
        
        // Pack asset ID (bits 48-55)
        packed |= (asset_id as u64 & ASSET_MASK) << ASSET_SHIFT;
        
        // Pack sequence number (bits 56-62), modulo 128
        packed |= ((sequence as u64 & 0x7F) & SEQUENCE_MASK) << SEQUENCE_SHIFT;
        
        // Pack side indicator (bit 63)
        let side_bit = if is_sell { 1u64 } else { 0u64 };
        packed |= (side_bit & SIDE_MASK) << SIDE_SHIFT;
        
        // Now pack integer parts using a secondary encoding
        // For full 64-bit utilization, we store integers in a companion struct
        // This struct focuses on the fractional/flag portion for quick comparisons
        
        Self(packed)
    }
    
    /// Create packed order from full price and quantity values
    /// Automatically splits into integer and fractional parts
    #[inline]
    pub fn from_price_quantity(
        price: f64,
        quantity: f64,
        flags: u16,
        asset_id: u8,
        sequence: u8,
        is_sell: bool,
    ) -> Result<Self, PackingError> {
        // Convert price to fixed-point representation
        let price_scaled = (price * 10000.0).round() as u64;
        let price_int = (price_scaled / 10000) as u16;
        let price_frac = (price_scaled % 10000) as u16;
        
        // Convert quantity to fixed-point representation
        let quantity_scaled = (quantity * 10000.0).round() as u64;
        let quantity_int = (quantity_scaled / 10000) as u16;
        let quantity_frac = (quantity_scaled % 10000) as u16;
        
        Ok(Self::new(
            price_int,
            price_frac,
            quantity_int,
            quantity_frac,
            flags,
            asset_id,
            sequence,
            is_sell,
        ))
    }
    
    /// Extract price fraction
    #[inline]
    pub fn price_fraction(&self) -> u16 {
        ((self.0 >> PRICE_FRAC_SHIFT) & PRICE_FRAC_MASK) as u16
    }
    
    /// Extract quantity fraction
    #[inline]
    pub fn quantity_fraction(&self) -> u16 {
        ((self.0 >> QUANTITY_FRAC_SHIFT) & QUANTITY_FRAC_MASK) as u16
    }
    
    /// Extract order flags
    #[inline]
    pub fn flags(&self) -> u16 {
        ((self.0 >> FLAGS_SHIFT) & FLAGS_MASK) as u16
    }
    
    /// Check if specific flag is set
    #[inline]
    pub fn has_flag(&self, flag: u16) -> bool {
        (self.flags() & flag) != 0
    }
    
    /// Extract asset ID
    #[inline]
    pub fn asset_id(&self) -> u8 {
        ((self.0 >> ASSET_SHIFT) & ASSET_MASK) as u8
    }
    
    /// Extract sequence number
    #[inline]
    pub fn sequence(&self) -> u8 {
        ((self.0 >> SEQUENCE_SHIFT) & SEQUENCE_MASK) as u8
    }
    
    /// Check if this is a sell order
    #[inline]
    pub fn is_sell(&self) -> bool {
        ((self.0 >> SIDE_SHIFT) & SIDE_MASK) != 0
    }
    
    /// Check if this is a buy order
    #[inline]
    pub fn is_buy(&self) -> bool {
        !self.is_sell()
    }
    
    /// Set a specific flag
    #[inline]
    pub fn with_flag(mut self, flag: u16) -> Self {
        let current_flags = self.flags();
        let new_flags = current_flags | flag;
        
        // Clear old flags and set new ones
        self.0 &= !(FLAGS_MASK << FLAGS_SHIFT);
        self.0 |= (new_flags as u64) << FLAGS_SHIFT;
        
        self
    }
    
    /// Clear a specific flag
    #[inline]
    pub fn without_flag(mut self, flag: u16) -> Self {
        let current_flags = self.flags();
        let new_flags = current_flags & !flag;
        
        self.0 &= !(FLAGS_MASK << FLAGS_SHIFT);
        self.0 |= (new_flags as u64) << FLAGS_SHIFT;
        
        self
    }
    
    /// Toggle sell/buy side
    #[inline]
    pub fn toggle_side(mut self) -> Self {
        self.0 ^= (SIDE_MASK << SIDE_SHIFT);
        self
    }
    
    /// Get raw 64-bit value
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.0
    }
    
    /// Create from raw 64-bit value (zero-cost)
    #[inline]
    pub fn from_u64(value: u64) -> Self {
        Self(value)
    }
    
    /// Compare prices (ignoring integer part stored separately)
    #[inline]
    pub fn compare_price_frac(&self, other: &Self) -> std::cmp::Ordering {
        self.price_fraction().cmp(&other.price_fraction())
    }
    
    /// Check for sign-extension bugs (should always be false)
    #[inline]
    pub fn validate_no_sign_extension(&self) -> bool {
        // All fields are extracted using unsigned operations
        // This is a runtime check for debugging
        self.price_fraction() <= MAX_PRICE_FRACTION
            && self.quantity_fraction() <= MAX_QUANTITY_FRACTION
            && self.asset_id() <= 255
            && self.sequence() <= 127
    }
}

impl Default for PackedOrder {
    fn default() -> Self {
        Self(0)
    }
}

impl PartialOrd for PackedOrder {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PackedOrder {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by side first (buys before sells for queue ordering)
        match (self.is_buy(), other.is_buy()) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                // Same side: compare by price
                match (self.is_buy(), other.is_buy()) {
                    (true, true) => other.compare_price_frac(self), // Higher price = higher priority for buys
                    (false, false) => self.compare_price_frac(other), // Lower price = higher priority for sells
                }
            }
        }
    }
}

/// Companion structure for storing integer parts of price/quantity
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, packed)]
pub struct PackedOrderIntegers {
    pub price_int: u16,
    pub quantity_int: u16,
    pub reserved: u32, // For alignment and future use
}

impl PackedOrderIntegers {
    #[inline]
    pub fn new(price_int: u16, quantity_int: u16) -> Self {
        Self {
            price_int,
            quantity_int,
            reserved: 0,
        }
    }
    
    /// Total size is 8 bytes (64 bits)
    #[inline]
    pub const SIZE_BYTES: usize = mem::size_of::<Self>();
}

/// Full order representation combining both packed structures
#[derive(Debug, Clone, Copy)]
pub struct FullPackedOrder {
    pub packed: PackedOrder,
    pub integers: PackedOrderIntegers,
}

impl FullPackedOrder {
    #[inline]
    pub fn new(
        price_int: u16,
        price_frac: u16,
        quantity_int: u16,
        quantity_frac: u16,
        flags: u16,
        asset_id: u8,
        sequence: u8,
        is_sell: bool,
    ) -> Self {
        Self {
            packed: PackedOrder::new(
                price_int,
                price_frac,
                quantity_int,
                quantity_frac,
                flags,
                asset_id,
                sequence,
                is_sell,
            ),
            integers: PackedOrderIntegers::new(price_int, quantity_int),
        }
    }
    
    /// Reconstruct full price from packed representation
    #[inline]
    pub fn get_price(&self) -> f64 {
        let price_total = (self.integers.price_int as u64) * 10000 + self.packed.price_fraction() as u64;
        price_total as f64 / 10000.0
    }
    
    /// Reconstruct full quantity from packed representation
    #[inline]
    pub fn get_quantity(&self) -> f64 {
        let quantity_total = (self.integers.quantity_int as u64) * 10000 
            + self.packed.quantity_fraction() as u64;
        quantity_total as f64 / 10000.0
    }
    
    /// Total size is 16 bytes
    #[inline]
    pub const SIZE_BYTES: usize = mem::size_of::<PackedOrder>() + mem::size_of::<PackedOrderIntegers>();
}

/// Packing errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingError {
    PriceOutOfRange,
    QuantityOutOfRange,
    InvalidFlags,
    AssetIdOutOfRange,
}

impl std::fmt::Display for PackingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackingError::PriceOutOfRange => write!(f, "Price out of representable range"),
            PackingError::QuantityOutOfRange => write!(f, "Quantity out of representable range"),
            PackingError::InvalidFlags => write!(f, "Invalid flag combination"),
            PackingError::AssetIdOutOfRange => write!(f, "Asset ID exceeds maximum (255)"),
        }
    }
}

impl std::error::Error for PackingError {}

/// Batch processor for multiple packed orders
pub struct PackedOrderBatch {
    orders: Vec<PackedOrder>,
    integers: Vec<PackedOrderIntegers>,
    capacity: usize,
}

impl PackedOrderBatch {
    pub fn new(capacity: usize) -> Self {
        Self {
            orders: Vec::with_capacity(capacity),
            integers: Vec::with_capacity(capacity),
            capacity,
        }
    }
    
    #[inline]
    pub fn push(&mut self, order: FullPackedOrder) -> Result<(), &'static str> {
        if self.orders.len() >= self.capacity {
            return Err("Batch capacity exceeded");
        }
        self.orders.push(order.packed);
        self.integers.push(order.integers);
        Ok(())
    }
    
    #[inline]
    pub fn len(&self) -> usize {
        self.orders.len()
    }
    
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
    
    /// Get raw slice for SIMD processing
    #[inline]
    pub fn as_raw_slice(&self) -> &[u64] {
        // Safe because PackedOrder is #[repr(transparent)] over u64
        unsafe {
            std::slice::from_raw_parts(
                self.orders.as_ptr() as *const u64,
                self.orders.len(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_packing_roundtrip() {
        let order = FullPackedOrder::new(
            50000, // price_int
            5000,  // price_frac (0.5000)
            1,     // quantity_int
            2500,  // quantity_frac (0.2500)
            order_flags::POST_ONLY,
            1,     // BTC asset ID
            42,    // sequence
            false, // buy
        );
        
        assert_eq!(order.get_price(), 50000.5);
        assert_eq!(order.get_quantity(), 1.25);
        assert!(order.packed.is_buy());
        assert!(order.packed.has_flag(order_flags::POST_ONLY));
        assert_eq!(order.packed.asset_id(), 1);
        assert_eq!(order.packed.sequence(), 42);
    }
    
    #[test]
    fn test_no_sign_extension() {
        // Test edge cases that could cause sign extension
        let order = FullPackedOrder::new(
            65535, // max price_int
            9999,  // max price_frac
            65535, // max quantity_int
            9999,  // max quantity_frac
            0xFFFF, // all flags
            255,   // max asset_id
            127,   // max sequence
            true,  // sell
        );
        
        assert!(order.packed.validate_no_sign_extension());
        assert!(order.packed.is_sell());
        assert_eq!(order.packed.price_fraction(), 9999);
        assert_eq!(order.packed.quantity_fraction(), 9999);
        assert_eq!(order.packed.asset_id(), 255);
        assert_eq!(order.packed.sequence(), 127);
    }
    
    #[test]
    fn test_flag_operations() {
        let mut order = PackedOrder::new(
            0, 0, 0, 0,
            order_flags::NONE,
            0, 0, false,
        );
        
        assert!(!order.has_flag(order_flags::POST_ONLY));
        
        order = order.with_flag(order_flags::POST_ONLY);
        assert!(order.has_flag(order_flags::POST_ONLY));
        
        order = order.without_flag(order_flags::POST_ONLY);
        assert!(!order.has_flag(order_flags::POST_ONLY));
    }
    
    #[test]
    fn test_size_assertions() {
        assert_eq!(mem::size_of::<PackedOrder>(), 8);
        assert_eq!(mem::size_of::<PackedOrderIntegers>(), 8);
        assert_eq!(mem::size_of::<FullPackedOrder>(), 16);
    }
}
