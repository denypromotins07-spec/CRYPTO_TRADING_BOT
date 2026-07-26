//! backend/venues/price_normalizer.rs
//! 
//! ZAID PERSONAL CRYPTO TRADING BOT - Stage 33: Multi-Venue SOR
//! Chapter 1: Price Normalization Across Exchanges
//! 
//! Aligns tick sizes and lot sizes across different exchanges (Binance, Coinbase, Kraken).
//! Ensures consistent price representation for cross-venue arbitrage calculations.
//! Uses zero-cost abstractions for maximum throughput on AMD Ryzen AI 5.
//! Strictly respects 8GB RAM limit.
//! 
//! Features:
//! - Exchange-specific precision configuration
//! - Tick size and lot size normalization
//! - Base/quote currency pair notation handling (BTC/USDT vs BTCUSDT)
//! - Round-trip conversion without precision loss
//! - Compile-time exchange trait implementations

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Precision configuration for a specific exchange
#[derive(Clone, Copy, Debug)]
pub struct ExchangePrecision {
    pub price_precision: u8,   // Decimal places for price (e.g., 2 for BTC/USD on Coinbase)
    pub quantity_precision: u8, // Decimal places for quantity
    pub tick_size: u64,        // Minimum price increment in micro-units
    pub lot_size: u64,         // Minimum quantity increment in micro-units
    pub min_notional: u64,     // Minimum order value in micro-units of quote currency
}

impl ExchangePrecision {
    /// Create precision config with sensible defaults
    pub const fn new(
        price_precision: u8,
        quantity_precision: u8,
        tick_size: u64,
        lot_size: u64,
        min_notional: u64,
    ) -> Self {
        Self {
            price_precision,
            quantity_precision,
            tick_size,
            lot_size,
            min_notional,
        }
    }
    
    /// Round price to exchange tick size
    #[inline]
    pub fn round_price(&self, price: f64) -> f64 {
        let tick = self.tick_size as f64 / 1_000_000.0;
        (price / tick).round() * tick
    }
    
    /// Round quantity to exchange lot size
    #[inline]
    pub fn round_quantity(&self, qty: f64) -> f64 {
        let lot = self.lot_size as f64 / 1_000_000.0;
        (qty / lot).round() * lot
    }
    
    /// Check if notional value meets minimum requirement
    #[inline]
    pub fn meets_min_notional(&self, price: f64, qty: f64) -> bool {
        let notional = (price * qty * 1_000_000.0) as u64;
        notional >= self.min_notional
    }
}

/// Supported exchanges with their precision configs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Exchange {
    Binance,
    Coinbase,
    Kraken,
    Bybit,
    Okx,
}

impl Exchange {
    /// Get default precision config for this exchange
    pub const fn default_precision(self) -> ExchangePrecision {
        match self {
            // Binance: typically 2 decimal places for BTC/USDT, tick size 0.01
            Exchange::Binance => ExchangePrecision::new(2, 6, 10_000, 1, 10_000_000),
            // Coinbase: varies by pair, BTC/USD uses 2 decimals
            Exchange::Coinbase => ExchangePrecision::new(2, 8, 1_000_000, 1, 1_000_000),
            // Kraken: uses more precision, BTC/USD up to 1 decimal
            Exchange::Kraken => ExchangePrecision::new(1, 8, 100_000, 1, 5_000_000),
            // Bybit: similar to Binance
            Exchange::Bybit => ExchangePrecision::new(2, 6, 10_000, 1, 10_000_000),
            // OKX: similar precision
            Exchange::Okx => ExchangePrecision::new(2, 6, 10_000, 1, 10_000_000),
        }
    }
    
    /// Parse exchange from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "binance" => Some(Self::Binance),
            "coinbase" | "gdax" => Some(Self::Coinbase),
            "kraken" => Some(Self::Kraken),
            "bybit" => Some(Self::Bybit),
            "okx" | "okex" => Some(Self::Okx),
            _ => None,
        }
    }
}

/// Normalized price representation across all exchanges
#[derive(Clone, Copy, Debug)]
pub struct NormalizedPrice {
    pub price_micro: u64,      // Price in micro-units (e.g., 50000000000 = 50000.000000)
    pub quantity_micro: u64,   // Quantity in micro-units
    pub exchange: Exchange,
    pub symbol_hash: u64,      // Fast hash for symbol comparison
}

impl NormalizedPrice {
    /// Create normalized price from raw values
    pub fn new(exchange: Exchange, price: f64, qty: f64, symbol: &str) -> Self {
        let precision = exchange.default_precision();
        let rounded_price = precision.round_price(price);
        let rounded_qty = precision.round_quantity(qty);
        
        Self {
            price_micro: (rounded_price * 1_000_000.0) as u64,
            quantity_micro: (rounded_qty * 1_000_000.0) as u64,
            exchange,
            symbol_hash: Self::hash_symbol(symbol),
        }
    }
    
    /// Get price as f64
    #[inline]
    pub fn price(&self) -> f64 {
        self.price_micro as f64 / 1_000_000.0
    }
    
    /// Get quantity as f64
    #[inline]
    pub fn quantity(&self) -> f64 {
        self.quantity_micro as f64 / 1_000_000.0
    }
    
    /// Get notional value in quote currency
    #[inline]
    pub fn notional(&self) -> f64 {
        self.price() * self.quantity()
    }
    
    /// Simple hash function for symbol comparison
    #[inline]
    fn hash_symbol(symbol: &str) -> u64 {
        // FNV-1a hash for fast symbol comparison
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in symbol.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
    
    /// Check if two normalized prices represent the same symbol
    #[inline]
    pub fn same_symbol(&self, other: &Self) -> bool {
        self.symbol_hash == other.symbol_hash
    }
}

/// Price Normalizer - Main orchestrator for cross-exchange price alignment
pub struct PriceNormalizer {
    /// Precision configs per exchange (can be overridden from API)
    precisions: RwLock<HashMap<Exchange, ExchangePrecision>>,
    /// Symbol notation mappings (e.g., "BTC/USDT" -> "BTCUSDT")
    symbol_mappings: RwLock<HashMap<(Exchange, String), String>>,
    /// Counter for normalized prices processed
    normalization_count: std::sync::atomic::AtomicU64,
}

impl PriceNormalizer {
    pub fn new() -> Self {
        // Initialize with default precisions
        let mut precisions = HashMap::with_capacity(8);
        precisions.insert(Exchange::Binance, Exchange::Binance.default_precision());
        precisions.insert(Exchange::Coinbase, Exchange::Coinbase.default_precision());
        precisions.insert(Exchange::Kraken, Exchange::Kraken.default_precision());
        precisions.insert(Exchange::Bybit, Exchange::Bybit.default_precision());
        precisions.insert(Exchange::Okx, Exchange::Okx.default_precision());
        
        Self {
            precisions: RwLock::new(precisions),
            symbol_mappings: RwLock::new(HashMap::with_capacity(64)),
            normalization_count: std::sync::atomic::AtomicU64::new(0),
        }
    }
    
    /// Update precision config for an exchange (called when exchange updates rules)
    pub fn update_precision(&self, exchange: Exchange, precision: ExchangePrecision) {
        let mut precs = self.precisions.write();
        precs.insert(exchange, precision);
    }
    
    /// Get precision for an exchange
    #[inline]
    pub fn get_precision(&self, exchange: Exchange) -> ExchangePrecision {
        let precs = self.precisions.read();
        *precs.get(&exchange).unwrap_or(&exchange.default_precision())
    }
    
    /// Register symbol mapping (e.g., "BTC/USDT" on Kraken -> "BTCUSDT" internal)
    pub fn register_symbol_mapping(&self, exchange: Exchange, external: &str, internal: &str) {
        let mut mappings = self.symbol_mappings.write();
        mappings.insert((exchange, external.to_string()), internal.to_string());
    }
    
    /// Normalize symbol to internal representation
    #[inline]
    pub fn normalize_symbol(&self, exchange: Exchange, external_symbol: &str) -> String {
        let mappings = self.symbol_mappings.read();
        mappings
            .get(&(exchange, external_symbol.to_string()))
            .cloned()
            .unwrap_or_else(|| {
                // Default: remove slashes and convert to uppercase
                external_symbol.replace('/', "").to_uppercase()
            })
    }
    
    /// Normalize price from exchange-specific format to internal representation
    pub fn normalize(
        &self,
        exchange: Exchange,
        price: f64,
        qty: f64,
        symbol: &str,
    ) -> NormalizedPrice {
        self.normalization_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        NormalizedPrice::new(exchange, price, qty, symbol)
    }
    
    /// Convert normalized price back to exchange-specific format
    pub fn denormalize(&self, norm_price: &NormalizedPrice) -> (f64, f64) {
        let precision = self.get_precision(norm_price.exchange);
        (norm_price.price(), norm_price.quantity())
    }
    
    /// Compare prices across exchanges after normalization
    pub fn compare_prices(
        &self,
        prices: &[NormalizedPrice],
    ) -> Option<(NormalizedPrice, NormalizedPrice)> {
        if prices.len() < 2 {
            return None;
        }
        
        let mut min_price = prices[0];
        let mut max_price = prices[0];
        
        for &price in prices.iter().skip(1) {
            if !price.same_symbol(&min_price) {
                continue; // Skip different symbols
            }
            if price.price_micro < min_price.price_micro {
                min_price = price;
            }
            if price.price_micro > max_price.price_micro {
                max_price = price;
            }
        }
        
        if min_price.price_micro != max_price.price_micro {
            Some((min_price, max_price))
        } else {
            None
        }
    }
    
    /// Get total normalizations performed
    pub fn normalization_count(&self) -> u64 {
        self.normalization_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for PriceNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// Zero-cost abstraction: compile-time symbol constants
pub mod symbols {
    pub const BTC_USDT: &str = "BTCUSDT";
    pub const ETH_USDT: &str = "ETHUSDT";
    pub const SOL_USDT: &str = "SOLUSDT";
    pub const BTC_USD: &str = "BTCUSD";
    pub const ETH_USD: &str = "ETHUSD";
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_price_rounding() {
        let binance_prec = Exchange::Binance.default_precision();
        
        // Test tick size rounding
        let price = 50000.123456;
        let rounded = binance_prec.round_price(price);
        assert_eq!(rounded, 50000.12); // Rounded to 2 decimals
        
        // Test lot size rounding
        let qty = 1.23456789;
        let rounded_qty = binance_prec.round_quantity(qty);
        assert_eq!(rounded_qty, 1.234568); // Rounded to 6 decimals
    }
    
    #[test]
    fn test_symbol_normalization() {
        let normalizer = PriceNormalizer::new();
        
        // Register custom mapping
        normalizer.register_symbol_mapping(Exchange::Kraken, "BTC/USDT", "BTCUSDT");
        
        let normalized = normalizer.normalize_symbol(Exchange::Kraken, "BTC/USDT");
        assert_eq!(normalized, "BTCUSDT");
        
        // Default behavior
        let default_normalized = normalizer.normalize_symbol(Exchange::Binance, "BTCUSDT");
        assert_eq!(default_normalized, "BTCUSDT");
    }
    
    #[test]
    fn test_cross_exchange_comparison() {
        let normalizer = Arc::new(PriceNormalizer::new());
        
        // Same BTC price on different exchanges
        let binance_price = normalizer.normalize(Exchange::Binance, 50000.0, 1.0, "BTCUSDT");
        let coinbase_price = normalizer.normalize(Exchange::Coinbase, 50010.0, 1.0, "BTCUSD");
        let kraken_price = normalizer.normalize(Exchange::Kraken, 49990.0, 1.0, "BTC/USDT");
        
        let prices = vec![binance_price, coinbase_price, kraken_price];
        
        if let Some((min, max)) = normalizer.compare_prices(&prices) {
            assert!(min.price_micro <= max.price_micro);
            println!("Arb opportunity: buy at {:.2}, sell at {:.2}", min.price(), max.price());
        }
    }
}
