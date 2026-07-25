//! ZAID PERSONAL CRYPTO TRADING BOT - Order Book Microstructure
//! Chapter 2: Order Imbalance Calculator
//! 
//! This module implements real-time bid/ask pressure analysis using L2 order book data.
//! It identifies hidden institutional order flow by tracking order book dynamics,
//! volume imbalances, and price impact patterns.
//! 
//! Memory Budget: <100MB for order book snapshots
//! Target Latency: <50μs for imbalance calculation
//! Assets: BTC, SOL, ETH parallel processing with shared memory pools
//! Integration: Feeds signals to execution engine for optimal entry timing

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};
use rayon::prelude::*;

/// Maximum order book depth to track (levels)
const MAX_BOOK_DEPTH: usize = 50;

/// Rolling window for imbalance history
const IMBALANCE_WINDOW: usize = 100;

/// Configuration for order imbalance detection
#[derive(Debug, Clone)]
pub struct ImbalanceConfig {
    pub assets: Vec<String>,
    pub depth_levels: usize,
    pub imbalance_threshold: f64,    // Threshold for significant imbalance
    pub volume_weight_factor: f64,   // Weight for volume vs count
    pub decay_factor: f64,           // Exponential decay for older updates
}

impl Default for ImbalanceConfig {
    fn default() -> Self {
        Self {
            assets: vec!["BTC".to_string(), "SOL".to_string(), "ETH".to_string()],
            depth_levels: 20,
            imbalance_threshold: 0.3,
            volume_weight_factor: 0.7,
            decay_factor: 0.95,
        }
    }
}

/// Single order book level
#[derive(Debug, Clone, Copy)]
pub struct BookLevel {
    pub price: f64,
    pub quantity: f64,
    pub order_count: u32,
}

/// Snapshot of order book state
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub timestamp_ns: u64,
    pub sequence_number: u64,
}

/// Order flow direction indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderFlowDirection {
    StrongBuy,
    Buy,
    Neutral,
    Sell,
    StrongSell,
}

/// Result of order imbalance analysis
#[derive(Debug, Clone)]
pub struct ImbalanceResult {
    pub asset: String,
    pub bid_ask_ratio: f64,
    pub volume_imbalance: f64,
    pub weighted_imbalance: f64,
    pub order_flow_direction: OrderFlowDirection,
    pub pressure_score: f64,      // -1.0 to +1.0 (sell to buy pressure)
    pub hidden_liquidity_estimate: f64,
    pub timestamp_ns: u64,
}

/// High-performance order imbalance calculator
pub struct OrderImbalanceCalculator {
    config: ImbalanceConfig,
    /// Latest order book snapshots per asset
    book_snapshots: HashMap<String, OrderBookSnapshot>,
    /// Rolling history of imbalance results
    imbalance_history: HashMap<String, VecDeque<ImbalanceResult>>,
    /// Cumulative volume trackers
    cumulative_bid_volume: HashMap<String, f64>,
    cumulative_ask_volume: HashMap<String, f64>,
    /// Sequence number for ordering updates
    sequence_counter: AtomicU64,
    /// Active flag for circuit breaker integration
    is_active: AtomicBool,
}

impl OrderImbalanceCalculator {
    /// Create new imbalance calculator with default config
    pub fn new() -> Self {
        Self::with_config(ImbalanceConfig::default())
    }
    
    /// Create new imbalance calculator with custom config
    pub fn with_config(config: ImbalanceConfig) -> Self {
        let mut book_snapshots = HashMap::new();
        let mut imbalance_history = HashMap::new();
        let mut cumulative_bid_volume = HashMap::new();
        let mut cumulative_ask_volume = HashMap::new();
        
        for asset in &config.assets {
            book_snapshots.insert(asset.clone(), OrderBookSnapshot {
                bids: Vec::with_capacity(config.depth_levels),
                asks: Vec::with_capacity(config.depth_levels),
                timestamp_ns: 0,
                sequence_number: 0,
            });
            
            imbalance_history.insert(asset.clone(), VecDeque::with_capacity(IMBALANCE_WINDOW));
            cumulative_bid_volume.insert(asset.clone(), 0.0);
            cumulative_ask_volume.insert(asset.clone(), 0.0);
        }
        
        Self {
            config,
            book_snapshots,
            imbalance_history,
            cumulative_bid_volume,
            cumulative_ask_volume,
            sequence_counter: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }
    
    /// Get current sequence number
    #[inline]
    fn next_sequence(&self) -> u64 {
        self.sequence_counter.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Update order book snapshot for an asset
    /// 
    /// This is the main entry point for L2 data updates.
    /// Expected input format: vectors of (price, quantity, order_count)
    pub fn update_book(&mut self, asset: &str, 
                       bids: Vec<(f64, f64, u32)>, 
                       asks: Vec<(f64, f64, u32)>) -> Option<ImbalanceResult> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }
        
        let snapshot = self.book_snapshots.get_mut(asset)?;
        
        // Convert to BookLevel format
        let bid_levels: Vec<BookLevel> = bids.into_iter()
            .take(self.config.depth_levels)
            .map(|(price, qty, count)| BookLevel { price, quantity: qty, order_count: count })
            .collect();
        
        let ask_levels: Vec<BookLevel> = asks.into_iter()
            .take(self.config.depth_levels)
            .map(|(price, qty, count)| BookLevel { price, quantity: qty, order_count: count })
            .collect();
        
        // Get timestamp in nanoseconds
        let timestamp_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;
        
        *snapshot = OrderBookSnapshot {
            bids: bid_levels,
            asks: ask_levels,
            timestamp_ns,
            sequence_number: self.next_sequence(),
        };
        
        // Calculate and return imbalance
        Some(self.calculate_imbalance(asset))
    }
    
    /// Calculate order imbalance from current book state
    #[inline]
    pub fn calculate_imbalance(&mut self, asset: &str) -> ImbalanceResult {
        let snapshot = match self.book_snapshots.get(asset) {
            Some(s) => s,
            None => return self.create_neutral_result(asset),
        };
        
        if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
            return self.create_neutral_result(asset);
        }
        
        // Calculate volume-weighted metrics
        let (bid_volume, bid_weighted_price) = self.calculate_weighted_volume(&snapshot.bids);
        let (ask_volume, ask_weighted_price) = self.calculate_weighted_volume(&snapshot.asks);
        
        // Update cumulative volumes with decay
        self.update_cumulative_volumes(asset, bid_volume, ask_volume);
        
        // Calculate bid-ask ratio
        let bid_ask_ratio = if ask_volume > 0.0 {
            bid_volume / ask_volume
        } else {
            1.0
        };
        
        // Calculate volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        let total_volume = bid_volume + ask_volume;
        let volume_imbalance = if total_volume > 0.0 {
            (bid_volume - ask_volume) / total_volume
        } else {
            0.0
        };
        
        // Calculate weighted imbalance (incorporates price levels)
        let weighted_imbalance = self.calculate_weighted_imbalance(
            bid_volume, ask_volume, bid_weighted_price, ask_weighted_price
        );
        
        // Estimate hidden liquidity (large orders split across levels)
        let hidden_liquidity = self.estimate_hidden_liquidity(&snapshot.bids, &snapshot.asks);
        
        // Determine order flow direction
        let order_flow_direction = self.determine_flow_direction(weighted_imbalance);
        
        // Calculate pressure score (-1.0 to +1.0)
        let pressure_score = self.calculate_pressure_score(asset, volume_imbalance, weighted_imbalance);
        
        let result = ImbalanceResult {
            asset: asset.to_string(),
            bid_ask_ratio,
            volume_imbalance,
            weighted_imbalance,
            order_flow_direction,
            pressure_score,
            hidden_liquidity_estimate: hidden_liquidity,
            timestamp_ns: snapshot.timestamp_ns,
        };
        
        // Store in rolling history
        if let Some(history) = self.imbalance_history.get_mut(asset) {
            if history.len() >= IMBALANCE_WINDOW {
                history.pop_front();
            }
            history.push_back(result.clone());
        }
        
        result
    }
    
    /// Calculate weighted volume and average price for book levels
    #[inline]
    fn calculate_weighted_volume(&self, levels: &[BookLevel]) -> (f64, f64) {
        let mut total_volume = 0.0;
        let mut weighted_sum = 0.0;
        
        // Apply exponential decay to deeper levels
        for (i, level) in levels.iter().enumerate() {
            let weight = self.config.decay_factor.powi(i as i32);
            let weighted_qty = level.quantity * weight;
            total_volume += weighted_qty;
            weighted_sum += weighted_qty * level.price;
        }
        
        let avg_price = if total_volume > 0.0 {
            weighted_sum / total_volume
        } else {
            0.0
        };
        
        (total_volume, avg_price)
    }
    
    /// Update cumulative volume trackers with exponential decay
    #[inline]
    fn update_cumulative_volumes(&mut self, asset: &str, bid_vol: f64, ask_vol: f64) {
        if let (Some(cum_bid), Some(cum_ask)) = (
            self.cumulative_bid_volume.get_mut(asset),
            self.cumulative_ask_volume.get_mut(asset)
        ) {
            // Apply decay to historical volumes
            *cum_bid = *cum_bid * self.config.decay_factor + bid_vol;
            *cum_ask = *cum_ask * self.config.decay_factor + ask_vol;
        }
    }
    
    /// Calculate weighted imbalance incorporating price information
    #[inline]
    fn calculate_weighted_imbalance(&self, bid_vol: f64, ask_vol: f64, 
                                     bid_price: f64, ask_price: f64) -> f64 {
        // Volume component
        let vol_component = if bid_vol + ask_vol > 0.0 {
            (bid_vol - ask_vol) / (bid_vol + ask_vol)
        } else {
            0.0
        };
        
        // Price pressure component (mid-price deviation)
        let mid_price = (bid_price + ask_price) / 2.0;
        let price_component = if mid_price > 0.0 {
            (bid_price - ask_price) / mid_price
        } else {
            0.0
        };
        
        // Combine with configured weights
        vol_component * self.config.volume_weight_factor 
            + price_component * (1.0 - self.config.volume_weight_factor)
    }
    
    /// Estimate hidden liquidity by detecting unusual order size patterns
    fn estimate_hidden_liquidity(&self, bids: &[BookLevel], asks: &[BookLevel]) -> f64 {
        // Look for iceberging patterns: similar sizes at multiple levels
        let mut hidden_estimate = 0.0;
        
        // Analyze bid side
        if bids.len() >= 3 {
            let avg_size: f64 = bids.iter().map(|l| l.quantity).sum::<f64>() / bids.len() as f64;
            let variance: f64 = bids.iter()
                .map(|l| (l.quantity - avg_size).powi(2))
                .sum::<f64>() / bids.len() as f64;
            
            // Low variance suggests potential iceberg orders
            if variance < avg_size * 0.1 {
                hidden_estimate += avg_size * bids.len() as f64 * 0.5;
            }
        }
        
        // Analyze ask side
        if asks.len() >= 3 {
            let avg_size: f64 = asks.iter().map(|l| l.quantity).sum::<f64>() / asks.len() as f64;
            let variance: f64 = asks.iter()
                .map(|l| (l.quantity - avg_size).powi(2))
                .sum::<f64>() / asks.len() as f64;
            
            if variance < avg_size * 0.1 {
                hidden_estimate += avg_size * asks.len() as f64 * 0.5;
            }
        }
        
        hidden_estimate
    }
    
    /// Determine order flow direction from imbalance score
    #[inline]
    fn determine_flow_direction(&self, imbalance: f64) -> OrderFlowDirection {
        match imbalance {
            x if x > 0.6 => OrderFlowDirection::StrongBuy,
            x if x > 0.2 => OrderFlowDirection::Buy,
            x if x < -0.6 => OrderFlowDirection::StrongSell,
            x if x < -0.2 => OrderFlowDirection::Sell,
            _ => OrderFlowDirection::Neutral,
        }
    }
    
    /// Calculate composite pressure score
    fn calculate_pressure_score(&mut self, asset: &str, 
                                 vol_imbalance: f64, weighted_imbalance: f64) -> f64 {
        // Get historical average for normalization
        let historical_avg = self.get_historical_pressure(asset);
        
        // Combine current signals
        let current_signal = (vol_imbalance + weighted_imbalance) / 2.0;
        
        // Blend with historical context
        let blended_score = current_signal * 0.7 + historical_avg * 0.3;
        
        // Clamp to [-1.0, 1.0]
        blended_score.max(-1.0).min(1.0)
    }
    
    /// Get average pressure from recent history
    fn get_historical_pressure(&self, asset: &str) -> f64 {
        match self.imbalance_history.get(asset) {
            Some(history) if !history.is_empty() => {
                history.iter()
                    .map(|r| r.pressure_score)
                    .sum::<f64>() / history.len() as f64
            }
            _ => 0.0,
        }
    }
    
    /// Create neutral result when no data available
    fn create_neutral_result(&self, asset: &str) -> ImbalanceResult {
        ImbalanceResult {
            asset: asset.to_string(),
            bid_ask_ratio: 1.0,
            volume_imbalance: 0.0,
            weighted_imbalance: 0.0,
            order_flow_direction: OrderFlowDirection::Neutral,
            pressure_score: 0.0,
            hidden_liquidity_estimate: 0.0,
            timestamp_ns: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64,
        }
    }
    
    /// Get latest imbalance result for an asset
    pub fn get_latest_imbalance(&self, asset: &str) -> Option<ImbalanceResult> {
        self.imbalance_history
            .get(asset)
            .and_then(|h| h.back())
            .cloned()
    }
    
    /// Check if imbalance exceeds threshold (trading signal)
    pub fn is_significant_imbalance(&self, asset: &str) -> bool {
        match self.get_latest_imbalance(asset) {
            Some(result) => result.pressure_score.abs() > self.config.imbalance_threshold,
            None => false,
        }
    }
    
    /// Get aggregate imbalance across all assets (for portfolio-level decisions)
    pub fn get_portfolio_imbalance(&self) -> f64 {
        let mut total_pressure = 0.0;
        let mut count = 0;
        
        for asset in &self.config.assets {
            if let Some(result) = self.get_latest_imbalance(asset) {
                total_pressure += result.pressure_score;
                count += 1;
            }
        }
        
        if count > 0 {
            total_pressure / count as f64
        } else {
            0.0
        }
    }
    
    /// Detect institutional order flow patterns
    pub fn detect_institutional_flow(&self, asset: &str) -> Option<InstitutionalFlowPattern> {
        let history = self.imbalance_history.get(asset)?;
        
        if history.len() < 10 {
            return None;
        }
        
        // Look for sustained one-sided pressure (institutional accumulation/distribution)
        let recent_scores: Vec<f64> = history.iter()
            .rev()
            .take(10)
            .map(|r| r.pressure_score)
            .collect();
        
        let avg_score: f64 = recent_scores.iter().sum::<f64>() / recent_scores.len() as f64;
        let consistency = recent_scores.iter()
            .filter(|&&s| s.signum() == avg_score.signum())
            .count() as f64 / recent_scores.len() as f64;
        
        // Hidden liquidity spike suggests large player
        let latest_hidden = history.back()?.hidden_liquidity_estimate;
        let avg_hidden: f64 = history.iter()
            .map(|r| r.hidden_liquidity_estimate)
            .sum::<f64>() / history.len() as f64;
        
        if consistency > 0.8 && avg_score.abs() > 0.3 {
            if latest_hidden > avg_hidden * 2.0 {
                return Some(if avg_score > 0.0 {
                    InstitutionalFlowPattern::Accumulation
                } else {
                    InstitutionalFlowPattern::Distribution
                });
            }
        }
        
        None
    }
    
    /// Activate/deactivate calculator (circuit breaker integration)
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Relaxed);
    }
    
    /// Check if calculator is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }
}

/// Institutional order flow pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstitutionalFlowPattern {
    Accumulation,    // Quiet buying over time
    Distribution,    // Quiet selling over time
    BlockTrade,      // Large single transaction
    Spoofing,        // Fake orders to manipulate
}

/// Parallel imbalance calculator for multiple assets
pub struct ParallelImbalanceAnalyzer {
    calculators: HashMap<String, OrderImbalanceCalculator>,
}

impl ParallelImbalanceAnalyzer {
    pub fn new(assets: Vec<String>) -> Self {
        let mut calculators = HashMap::new();
        
        for asset in assets {
            calculators.insert(asset.clone(), OrderImbalanceCalculator::with_config(
                ImbalanceConfig {
                    assets: vec![asset.clone()],
                    ..Default::default()
                }
            ));
        }
        
        Self { calculators }
    }
    
    /// Process updates for multiple assets in parallel
    pub fn process_parallel(&mut self, updates: Vec<(&str, Vec<(f64, f64, u32)>, Vec<(f64, f64, u32)>)>) 
                            -> HashMap<String, ImbalanceResult> {
        updates.into_par_iter()
            .filter_map(|(asset, bids, asks)| {
                if let Some(calc) = self.calculators.get_mut(asset) {
                    if let Some(result) = calc.update_book(asset, bids, asks) {
                        return Some((asset.to_string(), result));
                    }
                }
                None
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_imbalance_calculation() {
        let mut calculator = OrderImbalanceCalculator::new();
        
        // Simulate order book with buy pressure
        let bids = vec![
            (49900.0, 5.0, 10),
            (49850.0, 3.0, 5),
            (49800.0, 8.0, 15),
        ];
        
        let asks = vec![
            (50000.0, 2.0, 3),
            (50050.0, 1.5, 2),
            (50100.0, 3.0, 5),
        ];
        
        let result = calculator.update_book("BTC", bids, asks);
        
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.pressure_score > 0.0); // Should show buy pressure
        assert_eq!(r.order_flow_direction, OrderFlowDirection::Buy);
    }
    
    #[test]
    fn test_institutional_flow_detection() {
        let mut calculator = OrderImbalanceCalculator::new();
        
        // Feed consistent buy pressure
        for i in 0..20 {
            let base_price = 50000.0 - (i as f64 * 10.0);
            let bids = vec![
                (base_price - 100.0, 10.0, 20),
                (base_price - 200.0, 10.0, 20),
                (base_price - 300.0, 10.0, 20),
            ];
            let asks = vec![
                (base_price + 100.0, 2.0, 3),
                (base_price + 200.0, 2.0, 3),
            ];
            
            calculator.update_book("BTC", bids, asks);
        }
        
        let pattern = calculator.detect_institutional_flow("BTC");
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap(), InstitutionalFlowPattern::Accumulation);
    }
}
