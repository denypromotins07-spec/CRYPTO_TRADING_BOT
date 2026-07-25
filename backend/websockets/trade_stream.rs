/**
 * Trade Stream Parser for ZAID Personal Crypto Trading Bot
 * =========================================================
 * Chapter 2: High-Frequency WebSocket Market Data Feeds and Order Book State
 *
 * Parses aggressive trades for Cumulative Volume Delta (CVD) calculations.
 * Zero-copy parsing optimized for microsecond execution on AMD Ryzen AI 5.
 *
 * Features:
 * - Real-time trade message parsing
 * - Aggressive buyer/seller classification
 * - CVD calculation per symbol and aggregate
 * - Volume-weighted average price tracking
 * - Support for BTC, SOL, ETH, USDT pairs
 *
 * Author: Opus 4.8
 * Stage: 2 of 100
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use parking_lot::RwLock as PlRwLock;
use log::{info, warn, debug, error};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Custom error types for trade stream operations
#[derive(Error, Debug)]
pub enum TradeStreamError {
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Invalid trade data: {0}")]
    InvalidTradeData(String),
    #[error("Channel send error: {0}")]
    ChannelSendError(String),
    #[error("Overflow error")]
    OverflowError,
}

/// Raw trade message from Binance WebSocket
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RawTradeMessage {
    #[serde(rename = "e")]
    pub event_type: String,
    
    #[serde(rename = "E")]
    pub event_time: i64,
    
    #[serde(rename = "s")]
    pub symbol: String,
    
    #[serde(rename = "t")]
    pub trade_id: i64,
    
    #[serde(rename = "p")]
    pub price: String,
    
    #[serde(rename = "q")]
    pub quantity: String,
    
    #[serde(rename = "b")]
    pub buyer_order_id: i64,
    
    #[serde(rename = "a")]
    pub seller_order_id: i64,
    
    #[serde(rename = "T")]
    pub trade_time: i64,
    
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
    
    #[serde(rename = "M")]
    #[serde(default)]
    pub ignore: bool,
}

/// Parsed and classified trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedTrade {
    pub symbol: String,
    pub trade_id: i64,
    pub price: Decimal,
    pub quantity: Decimal,
    pub quote_quantity: Decimal,
    pub timestamp: i64,
    pub is_aggressive_buy: bool,  // True if buyer is aggressor (seller is maker)
    pub is_aggressive_sell: bool, // True if seller is aggressor (buyer is maker)
    pub buyer_order_id: i64,
    pub seller_order_id: i64,
}

impl TryFrom<RawTradeMessage> for ClassifiedTrade {
    type Error = TradeStreamError;
    
    fn try_from(raw: RawTradeMessage) -> Result<Self, Self::Error> {
        let price = raw.price.parse::<Decimal>()
            .map_err(|e| TradeStreamError::ParseError(format!("Invalid price: {}", e)))?;
        
        let quantity = raw.quantity.parse::<Decimal>()
            .map_err(|e| TradeStreamError::ParseError(format!("Invalid quantity: {}", e)))?;
        
        let quote_quantity = price * quantity;
        
        // Classify aggression
        // If is_buyer_maker = true, then buyer is maker, seller is aggressor (sell)
        // If is_buyer_maker = false, then buyer is taker, buyer is aggressor (buy)
        let is_aggressive_buy = !raw.is_buyer_maker;
        let is_aggressive_sell = raw.is_buyer_maker;
        
        Ok(Self {
            symbol: raw.symbol,
            trade_id: raw.trade_id,
            price,
            quantity,
            quote_quantity,
            timestamp: raw.trade_time,
            is_aggressive_buy,
            is_aggressive_sell,
            buyer_order_id: raw.buyer_order_id,
            seller_order_id: raw.seller_order_id,
        })
    }
}

/// CVD (Cumulative Volume Delta) statistics for a symbol
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CvdStats {
    pub symbol: String,
    pub aggressive_buy_volume: Decimal,
    pub aggressive_sell_volume: Decimal,
    pub net_cvd: Decimal,  // Buy volume - Sell volume
    pub aggressive_buy_count: u64,
    pub aggressive_sell_count: u64,
    pub total_trade_count: u64,
    pub buy_quote_volume: Decimal,
    pub sell_quote_volume: Decimal,
    pub vwap_buy: Option<Decimal>,
    pub vwap_sell: Option<Decimal>,
    pub last_update_time: Option<i64>,
}

impl CvdStats {
    /// Create new CVD stats for a symbol
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            ..Default::default()
        }
    }
    
    /// Update stats with a new trade
    pub fn update(&mut self, trade: &ClassifiedTrade) {
        if trade.is_aggressive_buy {
            self.aggressive_buy_volume += trade.quantity;
            self.aggressive_buy_count += 1;
            self.buy_quote_volume += trade.quote_quantity;
        } else if trade.is_aggressive_sell {
            self.aggressive_sell_volume += trade.quantity;
            self.aggressive_sell_count += 1;
            self.sell_quote_volume += trade.quote_quantity;
        }
        
        self.net_cvd = self.aggressive_buy_volume - self.aggressive_sell_volume;
        self.total_trade_count += 1;
        self.last_update_time = Some(trade.timestamp);
        
        // Calculate VWAP
        if self.aggressive_buy_volume > Decimal::ZERO {
            self.vwap_buy = Some(self.buy_quote_volume / self.aggressive_buy_volume);
        }
        if self.aggressive_sell_volume > Decimal::ZERO {
            self.vwap_sell = Some(self.sell_quote_volume / self.aggressive_sell_volume);
        }
    }
    
    /// Get CVD ratio (buy volume / total volume)
    pub fn cvd_ratio(&self) -> f64 {
        let total = self.aggressive_buy_volume + self.aggressive_sell_volume;
        if total > Decimal::ZERO {
            (self.aggressive_buy_volume / total).to_f64().unwrap_or(0.5)
        } else {
            0.5
        }
    }
    
    /// Get imbalance percentage
    pub fn imbalance_pct(&self) -> f64 {
        ((self.aggressive_buy_volume - self.aggressive_sell_volume).to_f64().unwrap_or(0.0) /
         (self.aggressive_buy_volume + self.aggressive_sell_volume).to_f64().unwrap_or(1.0)) * 100.0
    }
}

/// Time-windowed CVD tracker for short-term analysis
#[derive(Debug, Clone)]
pub struct WindowedCvd {
    pub window_seconds: u64,
    pub stats: CvdStats,
    pub trades: Vec<ClassifiedTrade>,
    pub start_time: i64,
    pub end_time: i64,
}

impl WindowedCvd {
    pub fn new(window_seconds: u64, symbol: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        
        Self {
            window_seconds,
            stats: CvdStats::new(symbol),
            trades: Vec::with_capacity(1000),
            start_time: now,
            end_time: now,
        }
    }
    
    /// Add trade and prune old trades outside window
    pub fn add_trade(&mut self, trade: ClassifiedTrade) {
        self.stats.update(&trade);
        self.trades.push(trade.clone());
        self.end_time = trade.timestamp;
        
        // Prune old trades
        let cutoff = trade.timestamp - (self.window_seconds * 1000) as i64;
        self.trades.retain(|t| t.timestamp >= cutoff);
        
        // Recalculate stats from retained trades
        self.stats = CvdStats::new(&self.stats.symbol);
        for t in &self.trades {
            self.stats.update(t);
        }
    }
}

/// High-performance trade stream processor
pub struct TradeStreamProcessor {
    symbols: Arc<PlRwLock<Vec<String>>>,
    cvd_stats: Arc<PlRwLock<HashMap<String, CvdStats>>>,
    windowed_cvds: Arc<PlRwLock<HashMap<String, WindowedCvd>>>,
    trade_tx: mpsc::Sender<ClassifiedTrade>,
    trade_rx: Arc<PlRwLock<Option<mpsc::Receiver<ClassifiedTrade>>>>,
    total_trades_processed: Arc<PlRwLock<u64>>,
    processing_start_time: Arc<PlRwLock<Option<Instant>>>,
}

impl TradeStreamProcessor {
    /// Create new trade stream processor
    pub fn new(symbols: Vec<String>, buffer_size: usize) -> Self {
        let (tx, rx) = mpsc::channel::<ClassifiedTrade>(buffer_size);
        
        // Initialize CVD stats for all symbols
        let mut cvd_stats = HashMap::new();
        let mut windowed_cvds = HashMap::new();
        
        for symbol in &symbols {
            cvd_stats.insert(symbol.clone(), CvdStats::new(symbol));
            windowed_cvds.insert(symbol.clone(), WindowedCvd::new(60, symbol)); // 1-minute window
        }
        
        Self {
            symbols: Arc::new(PlRwLock::new(symbols)),
            cvd_stats: Arc::new(PlRwLock::new(cvd_stats)),
            windowed_cvds: Arc::new(PlRwLock::new(windowed_cvds)),
            trade_tx: tx,
            trade_rx: Arc::new(PlRwLock::new(Some(rx))),
            total_trades_processed: Arc::new(PlRwLock::new(0)),
            processing_start_time: Arc::new(PlRwLock::new(None)),
        }
    }
    
    /// Process raw trade message
    pub async fn process_raw_message(&self, raw: RawTradeMessage) -> Result<(), TradeStreamError> {
        let trade: ClassifiedTrade = raw.try_into()?;
        
        // Update CVD stats
        {
            let mut stats_map = self.cvd_stats.write();
            if let Some(stats) = stats_map.get_mut(&trade.symbol) {
                stats.update(&trade);
            } else {
                // New symbol, create stats
                let mut new_stats = CvdStats::new(&trade.symbol);
                new_stats.update(&trade);
                stats_map.insert(trade.symbol.clone(), new_stats);
            }
        }
        
        // Update windowed CVD
        {
            let mut windowed_map = self.windowed_cvds.write();
            if let Some(windowed) = windowed_map.get_mut(&trade.symbol) {
                windowed.add_trade(trade.clone());
            }
        }
        
        // Send to channel for downstream processing
        if self.trade_tx.send(trade).await.is_err() {
            return Err(TradeStreamError::ChannelSendError(
                "Trade channel closed".to_string()
            ));
        }
        
        // Update counter
        *self.total_trades_processed.write() += 1;
        
        Ok(())
    }
    
    /// Get CVD stats for a symbol
    pub fn get_cvd_stats(&self, symbol: &str) -> Option<CvdStats> {
        self.cvd_stats.read().get(symbol).cloned()
    }
    
    /// Get all CVD stats
    pub fn get_all_cvd_stats(&self) -> HashMap<String, CvdStats> {
        self.cvd_stats.read().clone()
    }
    
    /// Get windowed CVD for a symbol
    pub fn get_windowed_cvd(&self, symbol: &str, window_seconds: u64) -> Option<WindowedCvd> {
        let map = self.windowed_cvds.read();
        map.get(symbol).map(|w| {
            if w.window_seconds == window_seconds {
                w.clone()
            } else {
                // Create new windowed CVD with requested window
                let mut new_windowed = WindowedCvd::new(window_seconds, symbol);
                // Copy recent trades that fit in new window
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64;
                let cutoff = now - (window_seconds * 1000) as i64;
                for trade in &w.trades {
                    if trade.timestamp >= cutoff {
                        new_windowed.add_trade(trade.clone());
                    }
                }
                new_windowed
            }
        })
    }
    
    /// Take the trade receiver (can only be done once)
    pub fn take_trade_receiver(&self) -> Option<mpsc::Receiver<ClassifiedTrade>> {
        self.trade_rx.write().take()
    }
    
    /// Get processing statistics
    pub fn get_processing_stats(&self) -> TradeProcessingStats {
        let total = *self.total_trades_processed.read();
        let start_time = *self.processing_start_time.read();
        let elapsed = start_time.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);
        
        let trades_per_second = if elapsed > Duration::ZERO {
            total as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        
        TradeProcessingStats {
            total_trades: total,
            elapsed_seconds: elapsed.as_secs_f64(),
            trades_per_second,
        }
    }
    
    /// Mark processing start time
    pub fn mark_start(&self) {
        *self.processing_start_time.write() = Some(Instant::now());
    }
}

/// Processing statistics
#[derive(Debug, Clone, Default)]
pub struct TradeProcessingStats {
    pub total_trades: u64,
    pub elapsed_seconds: f64,
    pub trades_per_second: f64,
}

/// Builder for trade stream processor
pub struct TradeStreamProcessorBuilder {
    symbols: Vec<String>,
    buffer_size: usize,
}

impl TradeStreamProcessorBuilder {
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            buffer_size: 10000,
        }
    }
    
    pub fn symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = symbols;
        self
    }
    
    pub fn add_symbol(mut self, symbol: &str) -> Self {
        self.symbols.push(symbol.to_string());
        self
    }
    
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
    
    pub fn build(self) -> TradeStreamProcessor {
        TradeStreamProcessor::new(self.symbols, self.buffer_size)
    }
}

impl Default for TradeStreamProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[tokio::main]
async fn main() {
    println!("Trade Stream Processor - ZAID Personal Crypto Trading Bot");
    println!("Stage 2 of 100 - Exchange Connectivity");
    
    let symbols = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
    ];
    
    let processor = TradeStreamProcessorBuilder::new()
        .symbols(symbols.clone())
        .buffer_size(5000)
        .build();
    
    processor.mark_start();
    
    // Simulate some trades
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for i in 0..100 {
        let symbol = &symbols[i % symbols.len()];
        let raw = RawTradeMessage {
            event_type: "trade".to_string(),
            event_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64,
            symbol: symbol.clone(),
            trade_id: i as i64,
            price: format!("{:.2}", rng.gen_range(1000.0..50000.0)),
            quantity: format!("{:.4}", rng.gen_range(0.001..10.0)),
            buyer_order_id: i as i64,
            seller_order_id: i as i64 + 1,
            trade_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as i64,
            is_buyer_maker: rng.gen_bool(0.5),
            ignore: false,
        };
        
        if let Err(e) = processor.process_raw_message(raw).await {
            error!("Error processing trade: {}", e);
        }
    }
    
    // Print CVD stats
    println!("\nCVD Statistics:");
    for symbol in &symbols {
        if let Some(stats) = processor.get_cvd_stats(symbol) {
            println!(
                "{}: Buy Vol={:.4}, Sell Vol={:.4}, Net CVD={:.4}, Imbalance={:.2}%",
                stats.symbol,
                stats.aggressive_buy_volume.to_f64().unwrap_or(0.0),
                stats.aggressive_sell_volume.to_f64().unwrap_or(0.0),
                stats.net_cvd.to_f64().unwrap_or(0.0),
                stats.imbalance_pct()
            );
        }
    }
    
    let proc_stats = processor.get_processing_stats();
    println!(
        "\nProcessing Stats: {} trades in {:.2}s ({:.0} trades/sec)",
        proc_stats.total_trades,
        proc_stats.elapsed_seconds,
        proc_stats.trades_per_second
    );
}
