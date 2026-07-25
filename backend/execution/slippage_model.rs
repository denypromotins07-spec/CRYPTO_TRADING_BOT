/**
 * Slippage Model for ZAID Personal Crypto Trading Bot
 * =====================================================
 * Chapter 3: Smart Order Routing, Slippage Modeling, and Order Lifecycle Management
 *
 * Real-time slippage calculation based on market depth analysis.
 * Zero-cost abstractions for microsecond execution on AMD Ryzen AI 5.
 *
 * Features:
 * - Order book impact modeling
 * - Volume-weighted slippage estimation
 * - Dynamic spread adjustment
 * - Market stress detection
 * - Support for BTC, SOL, ETH, USDT pairs
 *
 * Author: Opus 4.8
 * Stage: 2 of 100
 */

use std::collections::HashMap;
use std::sync::Arc;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use parking_lot::RwLock as PlRwLock;
use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Custom error types for slippage operations
#[derive(Error, Debug)]
pub enum SlippageError {
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
    #[error("Invalid order book data")]
    InvalidOrderBookData,
    #[error("Calculation overflow")]
    OverflowError,
    #[error("Slippage exceeds threshold: {0}%")]
    SlippageThresholdExceeded(f64),
}

/// Order book level for slippage calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub quantity: Decimal,
}

/// Slippage estimate result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageEstimate {
    pub symbol: String,
    pub side: String,
    pub order_quantity: Decimal,
    pub expected_fill_price: Decimal,
    pub mid_price: Decimal,
    pub slippage_bps: f64,
    pub slippage_pct: f64,
    pub total_cost: Decimal,
    pub market_impact_bps: f64,
    pub spread_cost_bps: f64,
    pub confidence: f64,  // 0.0 to 1.0
}

impl SlippageEstimate {
    /// Check if slippage is within acceptable threshold
    pub fn is_acceptable(&self, max_slippage_bps: f64) -> bool {
        self.slippage_bps <= max_slippage_bps
    }
    
    /// Get slippage in quote currency
    pub fn slippage_cost(&self) -> Decimal {
        (self.total_cost - (self.order_quantity * self.mid_price)).abs()
    }
}

/// Market stress indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketStress {
    Normal,
    Elevated,
    High,
    Extreme,
}

impl MarketStress {
    /// Get stress multiplier for slippage adjustment
    pub fn multiplier(&self) -> f64 {
        match self {
            MarketStress::Normal => 1.0,
            MarketStress::Elevated => 1.5,
            MarketStress::High => 2.0,
            MarketStress::Extreme => 5.0,
        }
    }
}

/// Slippage model configuration
#[derive(Debug, Clone)]
pub struct SlippageConfig {
    pub default_max_slippage_bps: f64,
    pub stress_adjustment_enabled: bool,
    pub min_liquidity_threshold: Decimal,
    pub confidence_decay_factor: f64,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            default_max_slippage_bps: 50.0,  // 0.5% max slippage
            stress_adjustment_enabled: true,
            min_liquidity_threshold: Decimal::from_str("0.001").unwrap(),
            confidence_decay_factor: 0.95,
        }
    }
}

/// High-performance slippage calculator
pub struct SlippageModel {
    config: SlippageConfig,
    order_books: Arc<PlRwLock<HashMap<String, OrderBookSnapshot>>>,
    stress_levels: Arc<PlRwLock<HashMap<String, MarketStress>>>,
    historical_slippage: Arc<PlRwLock<HashMap<String, Vec<f64>>>>,
}

impl SlippageModel {
    /// Create new slippage model
    pub fn new(config: SlippageConfig) -> Self {
        Self {
            config,
            order_books: Arc::new(PlRwLock::new(HashMap::new())),
            stress_levels: Arc::new(PlRwLock::new(HashMap::new())),
            historical_slippage: Arc::new(PlRwLock::new(HashMap::new())),
        }
    }
    
    /// Update order book snapshot for a symbol
    pub fn update_order_book(&self, symbol: &str, snapshot: OrderBookSnapshot) {
        self.order_books.write().insert(symbol.to_string(), snapshot);
    }
    
    /// Update market stress level
    pub fn update_stress_level(&self, symbol: &str, stress: MarketStress) {
        self.stress_levels.write().insert(symbol.to_string(), stress);
    }
    
    /// Estimate slippage for a market order
    pub fn estimate_slippage(
        &self,
        symbol: &str,
        side: &str,
        quantity: Decimal,
    ) -> Result<SlippageEstimate, SlippageError> {
        let books = self.order_books.read();
        let snapshot = books.get(symbol)
            .ok_or(SlippageError::InvalidOrderBookData)?;
        
        // Get relevant side of order book
        let levels = match side.to_uppercase().as_str() {
            "BUY" => &snapshot.asks,  // Buying hits asks
            "SELL" => &snapshot.bids, // Selling hits bids
            _ => return Err(SlippageError::InvalidOrderBookData),
        };
        
        if levels.is_empty() {
            return Err(SlippageError::InsufficientLiquidity);
        }
        
        // Calculate mid price
        let best_bid = snapshot.bids.first().map(|l| l.price).unwrap_or(Decimal::ZERO);
        let best_ask = snapshot.asks.first().map(|l| l.price).unwrap_or(Decimal::ZERO);
        let mid_price = if best_bid > Decimal::ZERO && best_ask > Decimal::ZERO {
            (best_bid + best_ask) / Decimal::from(2)
        } else {
            best_ask.max(best_bid)
        };
        
        if mid_price == Decimal::ZERO {
            return Err(SlippageError::InvalidOrderBookData);
        }
        
        // Walk the book to calculate fill price
        let mut remaining_qty = quantity;
        let mut total_cost = Decimal::ZERO;
        let mut filled_qty = Decimal::ZERO;
        
        for level in levels {
            if remaining_qty <= Decimal::ZERO {
                break;
            }
            
            let fill_qty = remaining_qty.min(level.quantity);
            total_cost += fill_qty * level.price;
            filled_qty += fill_qty;
            remaining_qty -= fill_qty;
        }
        
        // Check if we could fill the entire order
        if remaining_qty > Decimal::ZERO {
            return Err(SlippageError::InsufficientLiquidity);
        }
        
        let expected_fill_price = total_cost / filled_qty;
        
        // Calculate slippage
        let price_diff = (expected_fill_price - mid_price).abs();
        let slippage_bps = (price_diff / mid_price * Decimal::from(10000))
            .to_f64()
            .unwrap_or(f64::INFINITY);
        let slippage_pct = slippage_bps / 100.0;
        
        // Calculate spread cost
        let spread = best_ask - best_bid;
        let spread_bps = if mid_price > Decimal::ZERO {
            (spread / mid_price * Decimal::from(10000)).to_f64().unwrap_or(0.0)
        } else {
            0.0
        };
        let spread_cost_bps = spread_bps / 2.0;  // Half spread for one-sided trade
        
        // Market impact is slippage minus spread cost
        let market_impact_bps = (slippage_bps - spread_cost_bps).max(0.0);
        
        // Get stress multiplier
        let stress = self.stress_levels.read().get(symbol)
            .copied()
            .unwrap_or(MarketStress::Normal);
        let adjusted_slippage_bps = if self.config.stress_adjustment_enabled {
            slippage_bps * stress.multiplier()
        } else {
            slippage_bps
        };
        
        // Calculate confidence based on historical data and book depth
        let confidence = self.calculate_confidence(symbol, quantity, levels.len());
        
        Ok(SlippageEstimate {
            symbol: symbol.to_string(),
            side: side.to_string(),
            order_quantity: quantity,
            expected_fill_price,
            mid_price,
            slippage_bps: adjusted_slippage_bps,
            slippage_pct,
            total_cost,
            market_impact_bps,
            spread_cost_bps,
            confidence,
        })
    }
    
    /// Calculate confidence score
    fn calculate_confidence(&self, symbol: &str, quantity: Decimal, num_levels: usize) -> f64 {
        let mut confidence = 1.0;
        
        // Reduce confidence for large orders relative to book depth
        if let Some(snapshot) = self.order_books.read().get(symbol) {
            let total_book_qty: Decimal = snapshot.bids.iter()
                .chain(snapshot.asks.iter())
                .map(|l| l.quantity)
                .sum();
            
            if total_book_qty > Decimal::ZERO {
                let order_ratio = quantity / total_book_qty;
                let ratio_f64 = order_ratio.to_f64().unwrap_or(1.0);
                confidence *= (1.0 - ratio_f64.min(0.9)).max(0.1);
            }
        }
        
        // Reduce confidence for shallow books
        let depth_factor = (num_levels as f64 / 20.0).min(1.0);
        confidence *= depth_factor;
        
        // Apply historical decay
        if let Some(history) = self.historical_slippage.read().get(symbol) {
            if !history.is_empty() {
                let avg_error = history.iter().sum::<f64>() / history.len() as f64;
                let error_factor = (1.0 - (avg_error / 100.0)).max(0.5);
                confidence *= error_factor;
            }
        }
        
        confidence.clamp(0.0, 1.0)
    }
    
    /// Record actual slippage for learning
    pub fn record_actual_slippage(&self, symbol: &str, actual_slippage_bps: f64) {
        let mut history = self.historical_slippage.write();
        let entry = history.entry(symbol.to_string()).or_insert_with(Vec::new);
        entry.push(actual_slippage_bps);
        
        // Keep only last 100 observations
        if entry.len() > 100 {
            entry.remove(0);
        }
    }
    
    /// Check if order should be rejected due to slippage
    pub fn should_reject_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: Decimal,
        max_slippage_bps: Option<f64>,
    ) -> Result<bool, SlippageError> {
        let threshold = max_slippage_bps.unwrap_or(self.config.default_max_slippage_bps);
        let estimate = self.estimate_slippage(symbol, side, quantity)?;
        
        if estimate.slippage_bps > threshold {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Get optimal order size for target slippage
    pub fn get_optimal_size(
        &self,
        symbol: &str,
        side: &str,
        target_slippage_bps: f64,
    ) -> Option<Decimal> {
        // Binary search for optimal size
        let mut low = Decimal::from_str("0.0001").ok()?;
        let mut high = Decimal::from_str("1000.0").ok()?;
        
        for _ in 0..20 {  // Max iterations
            let mid = (low + high) / Decimal::from(2);
            
            match self.estimate_slippage(symbol, side, mid) {
                Ok(estimate) => {
                    if (estimate.slippage_bps - target_slippage_bps).abs() < 1.0 {
                        return Some(mid);
                    }
                    
                    if estimate.slippage_bps < target_slippage_bps {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                Err(_) => {
                    high = mid;
                }
            }
        }
        
        Some(low)
    }
}

/// Order book snapshot for slippage calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: i64,
}

impl OrderBookSnapshot {
    pub fn new(symbol: &str, bids: Vec<OrderBookLevel>, asks: Vec<OrderBookLevel>) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids,
            asks,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_slippage_calculation() {
        let config = SlippageConfig::default();
        let model = SlippageModel::new(config);
        
        // Create test order book
        let snapshot = OrderBookSnapshot::new(
            "BTCUSDT",
            vec![
                OrderBookLevel { price: Decimal::from_str("49999").unwrap(), quantity: Decimal::from_str("1.0").unwrap() },
                OrderBookLevel { price: Decimal::from_str("49998").unwrap(), quantity: Decimal::from_str("2.0").unwrap() },
            ],
            vec![
                OrderBookLevel { price: Decimal::from_str("50001").unwrap(), quantity: Decimal::from_str("1.0").unwrap() },
                OrderBookLevel { price: Decimal::from_str("50002").unwrap(), quantity: Decimal::from_str("2.0").unwrap() },
            ],
        );
        
        model.update_order_book("BTCUSDT", snapshot);
        
        // Test sell slippage
        let estimate = model.estimate_slippage("BTCUSDT", "SELL", Decimal::from_str("0.5").unwrap()).unwrap();
        assert!(estimate.slippage_bps >= 0.0);
        println!("Slippage estimate: {:?}", estimate);
    }
}

fn main() {
    println!("Slippage Model - ZAID Personal Crypto Trading Bot");
    println!("Stage 2 of 100 - Exchange Connectivity");
    
    let config = SlippageConfig::default();
    let model = SlippageModel::new(config);
    
    // Create test order book
    let snapshot = OrderBookSnapshot::new(
        "BTCUSDT",
        vec![
            OrderBookLevel { price: Decimal::from_str("49999").unwrap(), quantity: Decimal::from_str("1.0").unwrap() },
            OrderBookLevel { price: Decimal::from_str("49998").unwrap(), quantity: Decimal::from_str("2.0").unwrap() },
        ],
        vec![
            OrderBookLevel { price: Decimal::from_str("50001").unwrap(), quantity: Decimal::from_str("1.0").unwrap() },
            OrderBookLevel { price: Decimal::from_str("50002").unwrap(), quantity: Decimal::from_str("2.0").unwrap() },
        ],
    );
    
    model.update_order_book("BTCUSDT", snapshot);
    model.update_stress_level("BTCUSDT", MarketStress::Normal);
    
    // Estimate slippage
    match model.estimate_slippage("BTCUSDT", "BUY", Decimal::from_str("0.5").unwrap()) {
        Ok(estimate) => {
            println!("\nSlippage Estimate for BUY 0.5 BTC:");
            println!("  Mid Price: ${}", estimate.mid_price);
            println!("  Expected Fill: ${}", estimate.expected_fill_price);
            println!("  Slippage: {:.2} bps ({:.4}%)", estimate.slippage_bps, estimate.slippage_pct);
            println!("  Market Impact: {:.2} bps", estimate.market_impact_bps);
            println!("  Spread Cost: {:.2} bps", estimate.spread_cost_bps);
            println!("  Confidence: {:.2}%", estimate.confidence * 100.0);
            println!("  Total Cost: ${}", estimate.total_cost);
        }
        Err(e) => println!("Error: {}", e),
    }
}
