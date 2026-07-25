//! ZAID PERSONAL CRYPTO TRADING BOT - Transaction Cost Analysis (TCA)
//! Chapter 4: Market Impact Model
//! 
//! This module estimates the bot's footprint on the order book, calculating
//! how much our own trading affects market prices. Essential for optimal
//! execution sizing and timing decisions.
//! 
//! Memory Budget: <30MB for impact tracking
//! Target Latency: <100μs for impact estimation
//! Assets: BTC, SOL, ETH parallel processing
//! Integration: Feeds into execution scorecard

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, SystemTime};
use std::sync::{Arc, RwLock};

/// Default market depth levels to analyze
const DEFAULT_DEPTH_LEVELS: usize = 20;

/// Maximum trade history to retain per asset
const MAX_TRADE_HISTORY: usize = 1000;

/// Configuration for market impact model
#[derive(Debug, Clone)]
pub struct MarketImpactConfig {
    pub assets: Vec<String>,
    pub depth_levels: usize,
    pub decay_factor: f64,      // For exponential weighting
    pub base_slippage_bps: f64,  // Baseline slippage assumption
}

impl Default for MarketImpactConfig {
    fn default() -> Self {
        Self {
            assets: vec!["BTC".to_string(), "SOL".to_string(), "ETH".to_string()],
            depth_levels: DEFAULT_DEPTH_LEVELS,
            decay_factor: 0.95,
            base_slippage_bps: 5.0,
        }
    }
}

/// Order book liquidity snapshot
#[derive(Debug, Clone)]
pub struct LiquiditySnapshot {
    pub asset: String,
    pub bid_liquidity: Vec<f64>,  // Cumulative liquidity at each level
    pub ask_liquidity: Vec<f64>,
    pub bid_prices: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub timestamp_ns: u64,
}

/// Market impact estimate result
#[derive(Debug, Clone)]
pub struct ImpactEstimate {
    pub asset: String,
    pub side: TradeSide,
    pub order_size: f64,
    pub estimated_impact_bps: f64,
    pub estimated_slippage_usd: f64,
    pub price_move_estimate: f64,
    pub recommended_split: usize,  // How many child orders
    pub confidence_score: f64,     // 0-1 confidence in estimate
}

/// Trade side indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Historical impact observation
#[derive(Debug, Clone)]
struct ImpactObservation {
    order_size: f64,
    actual_impact_bps: f64,
    timestamp_ns: u64,
}

/// High-performance market impact calculator
pub struct MarketImpactModel {
    config: MarketImpactConfig,
    /// Latest liquidity snapshots
    liquidity_cache: Arc<RwLock<HashMap<String, LiquiditySnapshot>>>,
    /// Historical impact observations for calibration
    impact_history: Arc<RwLock<HashMap<String, VecDeque<ImpactObservation>>>>,
    /// Asset-specific volatility (for impact scaling)
    asset_volatility: Arc<RwLock<HashMap<String, f64>>>,
    /// Active flag
    is_active: AtomicBool,
}

unsafe impl Send for MarketImpactModel {}
unsafe impl Sync for MarketImpactModel {}

impl MarketImpactModel {
    /// Create new market impact model with default config
    pub fn new() -> Self {
        Self::with_config(MarketImpactConfig::default())
    }
    
    /// Create model with custom config
    pub fn with_config(config: MarketImpactConfig) -> Self {
        let mut liquidity_cache = HashMap::new();
        let mut impact_history = HashMap::new();
        let mut asset_volatility = HashMap::new();
        
        for asset in &config.assets {
            liquidity_cache.insert(asset.clone(), LiquiditySnapshot {
                asset: asset.clone(),
                bid_liquidity: Vec::with_capacity(config.depth_levels),
                ask_liquidity: Vec::with_capacity(config.depth_levels),
                bid_prices: Vec::with_capacity(config.depth_levels),
                ask_prices: Vec::with_capacity(config.depth_levels),
                timestamp_ns: 0,
            });
            
            impact_history.insert(asset.clone(), VecDeque::with_capacity(MAX_TRADE_HISTORY));
            asset_volatility.insert(asset.clone(), 0.02);  // Default 2% daily vol
        }
        
        Self {
            config,
            liquidity_cache: Arc::new(RwLock::new(liquidity_cache)),
            impact_history: Arc::new(RwLock::new(impact_history)),
            asset_volatility: Arc::new(RwLock::new(asset_volatility)),
            is_active: AtomicBool::new(true),
        }
    }
    
    /// Update liquidity snapshot for an asset
    pub fn update_liquidity(&self, asset: &str, 
                            bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        if !self.is_active.load(Ordering::Relaxed) {
            return;
        }
        
        let mut cache = match self.liquidity_cache.write() {
            Ok(c) => c,
            Err(e) => e.into_inner(),
        };
        
        let snapshot = match cache.get_mut(asset) {
            Some(s) => s,
            None => return,
        };
        
        // Process bids
        snapshot.bid_prices.clear();
        snapshot.bid_liquidity.clear();
        let mut cumulative_bid = 0.0;
        
        for (price, qty) in bids.iter().take(self.config.depth_levels) {
            snapshot.bid_prices.push(*price);
            cumulative_bid += qty;
            snapshot.bid_liquidity.push(cumulative_bid);
        }
        
        // Process asks
        snapshot.ask_prices.clear();
        snapshot.ask_liquidity.clear();
        let mut cumulative_ask = 0.0;
        
        for (price, qty) in asks.iter().take(self.config.depth_levels) {
            snapshot.ask_prices.push(*price);
            cumulative_ask += qty;
            snapshot.ask_liquidity.push(cumulative_ask);
        }
        
        snapshot.timestamp_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
    }
    
    /// Estimate market impact for a given order size
    pub fn estimate_impact(&self, asset: &str, side: TradeSide, 
                           order_size: f64, current_price: f64) -> Option<ImpactEstimate> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }
        
        let cache = match self.liquidity_cache.read() {
            Ok(c) => c,
            Err(e) => e.into_inner(),
        };
        
        let snapshot = cache.get(asset)?;
        
        // Select appropriate liquidity based on side
        let (liquidity, prices) = match side {
            TradeSide::Buy => (&snapshot.ask_liquidity, &snapshot.ask_prices),
            TradeSide::Sell => (&snapshot.bid_liquidity, &snapshot.bid_prices),
        };
        
        if liquidity.is_empty() || prices.is_empty() {
            return None;
        }
        
        // Find how many levels we need to walk
        let mut remaining_size = order_size;
        let mut notional_sum = 0.0;
        let mut quantity_sum = 0.0;
        let mut levels_consumed = 0;
        
        for (i, &avail_qty) in liquidity.iter().enumerate() {
            let prev_qty = if i == 0 { 0.0 } else { liquidity[i - 1] };
            let level_qty = avail_qty - prev_qty;
            
            let fill_qty = remaining_size.min(level_qty);
            remaining_size -= fill_qty;
            
            notional_sum += fill_qty * prices[i];
            quantity_sum += fill_qty;
            levels_consumed = i + 1;
            
            if remaining_size <= 0.0 {
                break;
            }
        }
        
        // Calculate average fill price
        let avg_fill_price = if quantity_sum > 0.0 {
            notional_sum / quantity_sum
        } else {
            current_price
        };
        
        // Calculate impact in basis points
        let price_move = match side {
            TradeSide::Buy => avg_fill_price - current_price,
            TradeSide::Sell => current_price - avg_fill_price,
        };
        
        let impact_bps = if current_price > 0.0 {
            (price_move / current_price) * 10000.0
        } else {
            0.0
        };
        
        // Calculate USD impact
        let slippage_usd = price_move * order_size;
        
        // Get historical calibration factor
        let calibration = self.get_calibration_factor(asset);
        let calibrated_impact = impact_bps * calibration;
        
        // Recommend order splitting if impact is high
        let recommended_split = if calibrated_impact > 10.0 {
            ((calibrated_impact / 5.0).ceil() as usize).max(2)
        } else {
            1
        };
        
        // Confidence based on available liquidity
        let total_liquidity = liquidity.last().copied().unwrap_or(0.0);
        let confidence = if total_liquidity > 0.0 {
            (order_size / total_liquidity).min(1.0)
        } else {
            0.5
        };
        
        Some(ImpactEstimate {
            asset: asset.to_string(),
            side,
            order_size,
            estimated_impact_bps: calibrated_impact.max(0.0),
            estimated_slippage_usd: slippage_usd.abs(),
            price_move_estimate: price_move.abs(),
            recommended_split,
            confidence_score: 1.0 - confidence,
        })
    }
    
    /// Get calibration factor from historical observations
    fn get_calibration_factor(&self, asset: &str) -> f64 {
        let history = match self.impact_history.read() {
            Ok(h) => h,
            Err(e) => e.into_inner(),
        };
        
        let obs = match history.get(asset) {
            Some(o) => o,
            None => return 1.0,  // Default factor
        };
        
        if obs.is_empty() {
            return 1.0;
        }
        
        // Calculate average ratio of actual to predicted impact
        let ratios: Vec<f64> = obs.iter()
            .filter(|o| o.actual_impact_bps > 0.0)
            .map(|o| o.actual_impact_bps / self.config.base_slippage_bps)
            .collect();
        
        if ratios.is_empty() {
            return 1.0;
        }
        
        // Simple average with decay weighting
        let sum: f64 = ratios.iter()
            .enumerate()
            .map(|(i, &r)| r * self.config.decay_factor.powi(i as i32))
            .sum();
        
        let weight_sum: f64 = ratios.iter()
            .enumerate()
            .map(|(i, _)| self.config.decay_factor.powi(i as i32))
            .sum();
        
        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            1.0
        }
    }
    
    /// Record actual impact after trade execution for calibration
    pub fn record_actual_impact(&self, asset: &str, order_size: f64, 
                                 actual_impact_bps: f64) {
        let mut history = match self.impact_history.write() {
            Ok(h) => h,
            Err(e) => e.into_inner(),
        };
        
        let obs_queue = match history.get_mut(asset) {
            Some(q) => q,
            None => return,
        };
        
        let observation = ImpactObservation {
            order_size,
            actual_impact_bps,
            timestamp_ns: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };
        
        if obs_queue.len() >= MAX_TRADE_HISTORY {
            obs_queue.pop_front();
        }
        
        obs_queue.push_back(observation);
    }
    
    /// Set volatility for an asset (affects impact scaling)
    pub fn set_asset_volatility(&self, asset: &str, volatility: f64) {
        let mut vol = match self.asset_volatility.write() {
            Ok(v) => v,
            Err(e) => e.into_inner(),
        };
        
        vol.insert(asset.to_string(), volatility);
    }
    
    /// Calculate optimal order size for target impact
    pub fn optimal_order_size(&self, asset: &str, target_impact_bps: f64,
                               current_price: f64) -> Option<f64> {
        let cache = match self.liquidity_cache.read() {
            Ok(c) => c,
            Err(e) => e.into_inner(),
        };
        
        let snapshot = cache.get(asset)?;
        
        // Binary search for order size that achieves target impact
        let mut low = 0.0;
        let mut high = snapshot.ask_liquidity.last().copied()
            .max(snapshot.bid_liquidity.last().copied().unwrap_or(0.0))
            .unwrap_or(100.0);
        
        for _ in 0..20 {  // 20 iterations for precision
            let mid = (low + high) / 2.0;
            
            if let Some(estimate) = self.estimate_impact(asset, TradeSide::Buy, mid, current_price) {
                if estimate.estimated_impact_bps < target_impact_bps {
                    low = mid;
                } else {
                    high = mid;
                }
            } else {
                break;
            }
        }
        
        Some((low + high) / 2.0)
    }
    
    /// Get aggregate market impact across portfolio
    pub fn portfolio_impact(&self, orders: Vec<(&str, TradeSide, f64)>) -> PortfolioImpact {
        let mut total_impact_usd = 0.0;
        let mut total_notional = 0.0;
        
        for (asset, side, size) in orders {
            // Use mid-price approximation
            let mid_price = self.get_mid_price(asset).unwrap_or(50000.0);
            
            if let Some(estimate) = self.estimate_impact(asset, side, size, mid_price) {
                total_impact_usd += estimate.estimated_slippage_usd;
                total_notional += size * mid_price;
            }
        }
        
        let portfolio_impact_bps = if total_notional > 0.0 {
            (total_impact_usd / total_notional) * 10000.0
        } else {
            0.0
        };
        
        PortfolioImpact {
            total_impact_usd,
            total_notional,
            portfolio_impact_bps,
            order_count: orders.len(),
        }
    }
    
    /// Get mid price from liquidity cache
    fn get_mid_price(&self, asset: &str) -> Option<f64> {
        let cache = match self.liquidity_cache.read() {
            Ok(c) => c,
            Err(e) => e.into_inner(),
        };
        
        let snapshot = cache.get(asset)?;
        
        let best_bid = *snapshot.bid_prices.first()?;
        let best_ask = *snapshot.ask_prices.first()?;
        
        Some((best_bid + best_ask) / 2.0)
    }
    
    /// Activate/deactivate model
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Relaxed);
    }
}

/// Portfolio-level impact summary
#[derive(Debug, Clone)]
pub struct PortfolioImpact {
    pub total_impact_usd: f64,
    pub total_notional: f64,
    pub portfolio_impact_bps: f64,
    pub order_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_impact_estimation() {
        let model = MarketImpactModel::new();
        
        // Set up liquidity
        let bids = vec![
            (49900.0, 5.0),
            (49850.0, 10.0),
            (49800.0, 15.0),
        ];
        let asks = vec![
            (50000.0, 5.0),
            (50050.0, 10.0),
            (50100.0, 15.0),
        ];
        
        model.update_liquidity("BTC", bids, asks);
        
        // Estimate impact for buying 3 BTC
        let estimate = model.estimate_impact("BTC", TradeSide::Buy, 3.0, 50000.0);
        
        assert!(estimate.is_some());
        let e = estimate.unwrap();
        assert!(e.estimated_impact_bps > 0.0);
        assert!(e.confidence_score > 0.0);
    }
    
    #[test]
    fn test_calibration_update() {
        let model = MarketImpactModel::new();
        
        // Record some actual impacts
        model.record_actual_impact("BTC", 1.0, 8.0);
        model.record_actual_impact("BTC", 2.0, 15.0);
        model.record_actual_impact("BTC", 1.5, 12.0);
        
        // Calibration factor should be updated
        let factor = model.get_calibration_factor("BTC");
        assert!(factor > 0.0);
    }
}
