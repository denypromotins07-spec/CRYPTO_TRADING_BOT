/*
 * ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
 * Chapter 2: NautilusTrader Backtesting Engine
 *
 * File: backend/backtest/matching_engine.rs
 * Purpose: Simulate realistic Binance order matching for backtesting.
 * Features:
 *     - Accurate maker/taker fee calculation
 *     - Funding rate simulation for perpetual swaps
 *     - Tick size and lot size validation
 *     - Order book simulation with partial fills
 */

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use log::{info, debug, warn, error};
use serde::{Serialize, Deserialize};

/// Error types for matching engine operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatchingError {
    InvalidPrice { price: f64, reason: String },
    InvalidQuantity { quantity: f64, reason: String },
    InsufficientBalance { asset: String, required: f64, available: f64 },
    OrderNotFound { order_id: u64 },
    MarketClosed,
    RateLimitExceeded,
}

impl std::fmt::Display for MatchingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchingError::InvalidPrice { price, reason } => 
                write!(f, "Invalid price {}: {}", price, reason),
            MatchingError::InvalidQuantity { quantity, reason } => 
                write!(f, "Invalid quantity {}: {}", quantity, reason),
            MatchingError::InsufficientBalance { asset, required, available } => 
                write!(f, "Insufficient {} balance: required {}, available {}", asset, required, available),
            MatchingError::OrderNotFound { order_id } => 
                write!(f, "Order not found: {}", order_id),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Order side (buy/sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
}

/// Time in force for orders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC, // Good Till Cancel
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
    GTD, // Good Till Date
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Represents a single order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: u64,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub price: f64,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub status: OrderStatus,
    pub timestamp: u128,
    pub client_order_id: Option<String>,
}

impl Order {
    pub fn new(
        order_id: u64,
        symbol: String,
        side: Side,
        order_type: OrderType,
        price: f64,
        quantity: f64,
        time_in_force: TimeInForce,
    ) -> Self {
        Self {
            order_id,
            symbol,
            side,
            order_type,
            time_in_force,
            price,
            quantity,
            filled_quantity: 0.0,
            remaining_quantity: quantity,
            status: OrderStatus::New,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            client_order_id: None,
        }
    }

    pub fn is_filled(&self) -> bool {
        self.remaining_quantity <= 0.0 || self.status == OrderStatus::Filled
    }

    pub fn fill(&mut self, quantity: f64, fill_price: f64) -> f64 {
        let fill_qty = quantity.min(self.remaining_quantity);
        self.filled_quantity += fill_qty;
        self.remaining_quantity -= fill_qty;

        if self.remaining_quantity <= 0.0 {
            self.status = OrderStatus::Filled;
        } else if self.filled_quantity > 0.0 {
            self.status = OrderStatus::PartiallyFilled;
        }

        fill_qty * fill_price
    }
}

/// Trade execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub trade_id: u64,
    pub order_id: u64,
    pub symbol: String,
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub commission_asset: String,
    pub timestamp: u128,
    pub is_maker: bool,
}

/// Symbol-specific trading rules (Binance specifications)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolRules {
    pub symbol: String,
    pub base_asset: String,
    pub quote_asset: String,
    pub price_precision: u8,
    pub quantity_precision: u8,
    pub tick_size: f64,      // Minimum price increment
    pub lot_size: f64,       // Minimum quantity increment
    pub min_quantity: f64,
    pub max_quantity: f64,
    pub min_price: f64,
    pub max_price: f64,
    pub min_notional: f64,
    pub maker_fee_rate: f64,
    pub taker_fee_rate: f64,
}

impl SymbolRules {
    /// Create default rules for common crypto symbols
    pub fn for_symbol(symbol: &str) -> Self {
        match symbol {
            "BTCUSDT" => Self {
                symbol: "BTCUSDT".to_string(),
                base_asset: "BTC".to_string(),
                quote_asset: "USDT".to_string(),
                price_precision: 2,
                quantity_precision: 5,
                tick_size: 0.01,
                lot_size: 0.001,
                min_quantity: 0.001,
                max_quantity: 1000.0,
                min_price: 0.01,
                max_price: 1000000.0,
                min_notional: 10.0,
                maker_fee_rate: 0.0002,  // 0.02%
                taker_fee_rate: 0.0004,  // 0.04%
            },
            "ETHUSDT" => Self {
                symbol: "ETHUSDT".to_string(),
                base_asset: "ETH".to_string(),
                quote_asset: "USDT".to_string(),
                price_precision: 2,
                quantity_precision: 4,
                tick_size: 0.01,
                lot_size: 0.001,
                min_quantity: 0.001,
                max_quantity: 10000.0,
                min_price: 0.01,
                max_price: 100000.0,
                min_notional: 10.0,
                maker_fee_rate: 0.0002,
                taker_fee_rate: 0.0004,
            },
            "SOLUSDT" => Self {
                symbol: "SOLUSDT".to_string(),
                base_asset: "SOL".to_string(),
                quote_asset: "USDT".to_string(),
                price_precision: 3,
                quantity_precision: 2,
                tick_size: 0.001,
                lot_size: 0.01,
                min_quantity: 0.01,
                max_quantity: 100000.0,
                min_price: 0.001,
                max_price: 10000.0,
                min_notional: 10.0,
                maker_fee_rate: 0.0002,
                taker_fee_rate: 0.0004,
            },
            _ => Self {
                symbol: symbol.to_string(),
                base_asset: "BASE".to_string(),
                quote_asset: "USDT".to_string(),
                price_precision: 3,
                quantity_precision: 3,
                tick_size: 0.001,
                lot_size: 0.001,
                min_quantity: 0.001,
                max_quantity: 10000.0,
                min_price: 0.001,
                max_price: 100000.0,
                min_notional: 10.0,
                maker_fee_rate: 0.0002,
                taker_fee_rate: 0.0004,
            },
        }
    }

    /// Validate and normalize price according to tick size
    pub fn normalize_price(&self, price: f64) -> f64 {
        (price / self.tick_size).round() * self.tick_size
    }

    /// Validate and normalize quantity according to lot size
    pub fn normalize_quantity(&self, quantity: f64) -> f64 {
        (quantity / self.lot_size).round() * self.lot_size
    }

    /// Validate order parameters
    pub fn validate_order(
        &self,
        price: f64,
        quantity: f64,
    ) -> Result<(), MatchingError> {
        if price < self.min_price || price > self.max_price {
            return Err(MatchingError::InvalidPrice {
                price,
                reason: format!("Price must be between {} and {}", self.min_price, self.max_price),
            });
        }

        if quantity < self.min_quantity || quantity > self.max_quantity {
            return Err(MatchingError::InvalidQuantity {
                quantity,
                reason: format!("Quantity must be between {} and {}", self.min_quantity, self.max_quantity),
            });
        }

        let notional = price * quantity;
        if notional < self.min_notional {
            return Err(MatchingError::InvalidQuantity {
                quantity,
                reason: format!("Notional value {} below minimum {}", notional, self.min_notional),
            });
        }

        Ok(())
    }
}

/// Funding rate data for perpetual swaps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: String,
    pub rate: f64,
    pub timestamp: u128,
    next_funding_time: u128,
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
    pub order_count: usize,
}

/// Realistic matching engine simulating Binance behavior
pub struct MatchingEngine {
    symbol_rules: SymbolRules,
    bids: BTreeMap<f64, PriceLevel>,  // Sorted descending by price
    asks: BTreeMap<f64, PriceLevel>,  // Sorted ascending by price
    orders: BTreeMap<u64, Order>,
    trades: VecDeque<Trade>,
    order_id_counter: u64,
    trade_id_counter: u64,
    current_price: f64,
    funding_rate: Option<FundingRate>,
    last_funding_time: u128,
    funding_interval_ms: u128,
}

impl MatchingEngine {
    /// Create a new matching engine for a specific symbol
    pub fn new(symbol: &str, initial_price: f64) -> Self {
        let rules = SymbolRules::for_symbol(symbol);
        let normalized_price = rules.normalize_price(initial_price);

        Self {
            symbol_rules: rules,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: BTreeMap::new(),
            trades: VecDeque::with_capacity(1000),
            order_id_counter: 1,
            trade_id_counter: 1,
            current_price: normalized_price,
            funding_rate: None,
            last_funding_time: 0,
            funding_interval_ms: 8 * 60 * 60 * 1000, // 8 hours in milliseconds
        }
    }

    /// Get the symbol rules
    pub fn rules(&self) -> &SymbolRules {
        &self.symbol_rules
    }

    /// Get current mid price
    pub fn mid_price(&self) -> f64 {
        self.current_price
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.last_key_value().map(|(k, _)| *k)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first_key_value().map(|(k, _)| *k)
    }

    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Submit a new order
    pub fn submit_order(
        &mut self,
        symbol: String,
        side: Side,
        order_type: OrderType,
        price: f64,
        quantity: f64,
        time_in_force: TimeInForce,
    ) -> Result<Order, MatchingError> {
        // Validate against symbol rules
        let normalized_price = self.symbol_rules.normalize_price(price);
        let normalized_quantity = self.symbol_rules.normalize_quantity(quantity);

        self.symbol_rules.validate_order(normalized_price, normalized_quantity)?;

        let order_id = self.order_id_counter;
        self.order_id_counter += 1;

        let mut order = Order::new(
            order_id,
            symbol,
            side,
            order_type,
            normalized_price,
            normalized_quantity,
            time_in_force,
        );

        // Match the order immediately
        let fills = self.match_order(&mut order);

        // Update order status based on fills
        if order.filled_quantity > 0.0 && order.remaining_quantity > 0.0 {
            order.status = OrderStatus::PartiallyFilled;
        }

        // Store resting orders (limit orders that weren't fully filled)
        if order.remaining_quantity > 0.0 && order.order_type == OrderType::Limit {
            match order.time_in_force {
                TimeInForce::GTC | TimeInForce::GTD => {
                    self.add_to_book(&order);
                }
                TimeInForce::IOC | TimeInForce::FOK => {
                    // IOC/FOK orders should be cancelled if not fully filled
                    if order.order_type == OrderType::Limit && order.time_in_force == TimeInForce::FOK {
                        order.status = OrderStatus::Cancelled;
                    }
                }
            }
        }

        self.orders.insert(order_id, order.clone());

        Ok(order)
    }

    /// Match an incoming order against the book
    fn match_order(&mut self, order: &mut Order) -> Vec<Trade> {
        let mut fills = Vec::new();

        match order.side {
            Side::Buy => {
                // Match against asks (lowest first)
                while order.remaining_quantity > 0.0 {
                    if let Some((ask_price, ask_level)) = self.asks.first_entry() {
                        if order.order_type == OrderType::Limit && order.price < *ask_price {
                            break; // Price crossed, stop matching
                        }

                        let fill_qty = order.remaining_quantity.min(ask_level.quantity);
                        let fill_value = self.execute_fill(
                            order.order_id,
                            order.symbol.clone(),
                            Side::Buy,
                            *ask_price,
                            fill_qty,
                            false, // Taker
                        );

                        order.fill(fill_qty, *ask_price);
                        fills.push(fill_value);

                        // Update or remove ask level
                        if ask_level.quantity <= fill_qty {
                            self.asks.remove(ask_price);
                        } else {
                            ask_level.quantity -= fill_qty;
                        }
                    } else {
                        break; // No more asks
                    }
                }
            }
            Side::Sell => {
                // Match against bids (highest first)
                while order.remaining_quantity > 0.0 {
                    if let Some((bid_price, bid_level)) = self.bids.last_entry() {
                        if order.order_type == OrderType::Limit && order.price > *bid_price {
                            break; // Price crossed, stop matching
                        }

                        let fill_qty = order.remaining_quantity.min(bid_level.quantity);
                        let fill_value = self.execute_fill(
                            order.order_id,
                            order.symbol.clone(),
                            Side::Sell,
                            *bid_price,
                            fill_qty,
                            false, // Taker
                        );

                        order.fill(fill_qty, *bid_price);
                        fills.push(fill_value);

                        // Update or remove bid level
                        if bid_level.quantity <= fill_qty {
                            self.bids.remove(bid_price);
                        } else {
                            bid_level.quantity -= fill_qty;
                        }
                    } else {
                        break; // No more bids
                    }
                }
            }
        }

        fills
    }

    /// Execute a fill and create a trade record
    fn execute_fill(
        &mut self,
        order_id: u64,
        symbol: String,
        side: Side,
        price: f64,
        quantity: f64,
        is_maker: bool,
    ) -> Trade {
        let fee_rate = if is_maker {
            self.symbol_rules.maker_fee_rate
        } else {
            self.symbol_rules.taker_fee_rate
        };

        let commission = price * quantity * fee_rate;
        let commission_asset = self.symbol_rules.quote_asset.clone();

        let trade = Trade {
            trade_id: self.trade_id_counter,
            order_id,
            symbol,
            side,
            price,
            quantity,
            commission,
            commission_asset,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            is_maker,
        };

        self.trade_id_counter += 1;

        // Update current price
        self.current_price = price;

        // Store trade (with limited history)
        if self.trades.len() >= 1000 {
            self.trades.pop_front();
        }
        self.trades.push_back(trade.clone());

        trade
    }

    /// Add a resting order to the book
    fn add_to_book(&mut self, order: &Order) {
        let levels = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        let level = levels.entry(order.price).or_insert_with(|| PriceLevel {
            price: order.price,
            quantity: 0.0,
            order_count: 0,
        });

        level.quantity += order.remaining_quantity;
        level.order_count += 1;
    }

    /// Cancel an order
    pub fn cancel_order(&mut self, order_id: u64) -> Result<Order, MatchingError> {
        let order = self.orders.get_mut(&order_id)
            .ok_or(MatchingError::OrderNotFound { order_id })?;

        if order.is_filled() {
            return Err(MatchingError::OrderNotFound { order_id });
        }

        // Remove from book
        let levels = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        if let Some(level) = levels.get_mut(&order.price) {
            level.quantity -= order.remaining_quantity;
            level.order_count = level.order_count.saturating_sub(1);

            if level.quantity <= 0.0 {
                levels.remove(&order.price);
            }
        }

        order.status = OrderStatus::Cancelled;
        Ok(order.clone())
    }

    /// Get recent trades
    pub fn recent_trades(&self, limit: usize) -> Vec<Trade> {
        self.trades.iter().rev().take(limit).cloned().collect()
    }

    /// Get order book depth
    pub fn get_book_depth(&self, levels: usize) -> (Vec<PriceLevel>, Vec<PriceLevel>) {
        let bids: Vec<PriceLevel> = self.bids.values().rev().take(levels).cloned().collect();
        let asks: Vec<PriceLevel> = self.asks.values().take(levels).cloned().collect();
        (bids, asks)
    }

    /// Update funding rate (called periodically)
    pub fn update_funding_rate(&mut self, rate: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();

        self.funding_rate = Some(FundingRate {
            symbol: self.symbol_rules.symbol.clone(),
            rate,
            timestamp: now,
            next_funding_time: now + self.funding_interval_ms,
        });
    }

    /// Calculate funding payment for a position
    pub fn calculate_funding_payment(
        &self,
        position_size: f64,
        entry_price: f64,
    ) -> f64 {
        if let Some(funding) = &self.funding_rate {
            let notional = position_size * entry_price;
            notional * funding.rate
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matching_engine_creation() {
        let engine = MatchingEngine::new("BTCUSDT", 50000.0);
        assert_eq!(engine.mid_price(), 50000.0);
        assert_eq!(engine.rules().maker_fee_rate, 0.0002);
    }

    #[test]
    fn test_limit_order_placement() {
        let mut engine = MatchingEngine::new("BTCUSDT", 50000.0);

        // Place a buy limit order below market
        let order = engine.submit_order(
            "BTCUSDT".to_string(),
            Side::Buy,
            OrderType::Limit,
            49000.0,
            0.1,
            TimeInForce::GTC,
        ).unwrap();

        assert_eq!(order.status, OrderStatus::New);
        assert_eq!(order.remaining_quantity, 0.1);

        // Check it's in the book
        let (bids, _) = engine.get_book_depth(10);
        assert!(!bids.is_empty());
    }

    #[test]
    fn test_market_order_execution() {
        let mut engine = MatchingEngine::new("BTCUSDT", 50000.0);

        // First place a sell limit order
        engine.submit_order(
            "BTCUSDT".to_string(),
            Side::Sell,
            OrderType::Limit,
            50100.0,
            0.5,
            TimeInForce::GTC,
        ).unwrap();

        // Now place a buy market order
        let order = engine.submit_order(
            "BTCUSDT".to_string(),
            Side::Buy,
            OrderType::Market,
            50100.0,
            0.3,
            TimeInForce::IOC,
        ).unwrap();

        assert_eq!(order.status, OrderStatus::Filled);
        assert_eq!(order.filled_quantity, 0.3);

        // Verify trade was recorded
        let trades = engine.recent_trades(10);
        assert!(!trades.is_empty());
        assert!(trades[0].commission > 0.0);
    }

    #[test]
    fn test_symbol_rules_normalization() {
        let rules = SymbolRules::for_symbol("BTCUSDT");

        // Test price normalization
        let normalized = rules.normalize_price(50123.456);
        assert_eq!(normalized, 50123.46);

        // Test quantity normalization
        let normalized = rules.normalize_quantity(0.12345);
        assert_eq!(normalized, 0.123);
    }
}
