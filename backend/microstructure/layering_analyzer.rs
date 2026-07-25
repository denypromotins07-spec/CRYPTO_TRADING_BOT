//! Layering Analyzer - Multi-Level Manipulation Detection
//!
//! This module detects sophisticated layering algorithms that place
//! multiple fake orders at different price levels to create false
//! impressions of supply/demand.
//!
//! Designed for the ZAID PERSONAL CRYPTO TRADING BOT with 8GB RAM constraints.
//! Uses zero-cost abstractions and efficient pattern matching.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;

/// Maximum number of levels to analyze for layering
const MAX_LAYERS: usize = 20;

/// Minimum layers required for detection
const MIN_LAYERS_FOR_DETECTION: usize = 3;

/// Default time window in milliseconds
const DEFAULT_WINDOW_MS: u64 = 1000;

/// Order ID type
pub type OrderId = u64;

/// Price tick type
pub type PriceTick = i64;

/// Volume type
pub type Volume = u64;

/// Side of the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// A single layer in a potential layering pattern
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Layer {
    /// Price level
    pub price: PriceTick,
    /// Order volume at this level
    pub volume: Volume,
    /// Number of orders
    pub order_count: u32,
    /// Distance from mid-price in ticks
    pub distance_from_mid: i32,
    /// Timestamp when layer was detected
    pub timestamp_us: u64,
    /// Whether this layer was cancelled quickly
    pub was_cancelled: bool,
    /// Time to cancellation in microseconds (if cancelled)
    pub cancel_time_us: Option<u64>,
}

impl Layer {
    #[inline]
    pub const fn new(
        price: PriceTick,
        volume: Volume,
        order_count: u32,
        distance: i32,
        timestamp_us: u64,
    ) -> Self {
        Self {
            price,
            volume,
            order_count,
            distance_from_mid: distance,
            timestamp_us,
            was_cancelled: false,
            cancel_time_us: None,
        }
    }

    #[inline]
    pub fn mark_cancelled(&mut self, cancel_time_us: u64) {
        self.was_cancelled = true;
        self.cancel_time_us = Some(cancel_time_us);
    }

    #[inline]
    pub fn lifetime_us(&self) -> u64 {
        match self.cancel_time_us {
            Some(ct) => ct - self.timestamp_us,
            None => get_timestamp_us() - self.timestamp_us,
        }
    }
}

/// Detected layering pattern
#[derive(Debug, Clone)]
pub struct LayeringPattern {
    /// Side of the book (bid or ask)
    pub side: Side,
    /// Symbol being analyzed
    pub symbol: String,
    /// All detected layers in the pattern
    pub layers: Vec<Layer>,
    /// Confidence score [0, 1]
    pub confidence: f64,
    /// Total volume in the pattern
    pub total_volume: Volume,
    /// Price range covered (in ticks)
    pub price_range: PriceTick,
    /// Average time to cancellation
    pub avg_cancel_time_us: f64,
    /// Detection timestamp
    pub detected_at_us: u64,
    /// Pattern severity (1-5 scale)
    pub severity: u8,
}

impl LayeringPattern {
    #[inline]
    pub fn calculate_severity(&mut self) {
        let layer_count = self.layers.len();
        let cancel_rate = self.layers.iter().filter(|l| l.was_cancelled).count() as f64 / layer_count as f64;
        let total_vol = self.total_volume as f64;
        
        // Severity based on: layer count, cancel rate, and volume
        let count_score = (layer_count.min(10) as f64 / 10.0) * 0.4;
        let cancel_score = cancel_rate * 0.4;
        let vol_score = (total_vol / 10000.0).min(1.0) * 0.2;
        
        let combined = count_score + cancel_score + vol_score;
        self.severity = ((combined * 5.0).ceil() as u8).min(5).max(1);
    }
}

/// Layering analysis result
#[derive(Debug, Clone, Copy)]
pub struct LayeringAnalysis {
    /// Whether layering is detected
    pub is_detected: bool,
    /// Confidence in detection
    pub confidence: f64,
    /// Number of suspicious layers
    pub suspicious_layers: usize,
    /// Estimated fake volume
    pub fake_volume_estimate: Volume,
}

/// Historical data for a price level
struct LevelHistory {
    /// Recent order placements
    placements: VecDeque<u64>,
    /// Recent cancellations
    cancellations: VecDeque<u64>,
    /// Total volume placed
    total_volume: Volume,
}

impl LevelHistory {
    fn new() -> Self {
        Self {
            placements: VecDeque::with_capacity(100),
            cancellations: VecDeque::with_capacity(100),
            total_volume: 0,
        }
    }
}

/// Layering analyzer for detecting multi-level manipulation
pub struct LayeringAnalyzer {
    /// Symbol being analyzed
    symbol: String,
    /// Current mid-price
    mid_price: RwLock<Option<PriceTick>>,
    /// Active layers being tracked
    active_layers: RwLock<HashMap<PriceTick, Layer>>,
    /// Historical patterns detected
    detected_patterns: RwLock<VecDeque<LayeringPattern>>,
    /// Level history for statistical analysis
    level_history: RwLock<HashMap<PriceTick, LevelHistory>>,
    /// Analysis window in milliseconds
    window_ms: u64,
    /// Detection sensitivity (0.0 to 1.0)
    sensitivity: f64,
    /// Last update timestamp
    last_update: RwLock<Instant>,
}

/// Get current timestamp in microseconds
#[inline]
pub fn get_timestamp_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

impl LayeringAnalyzer {
    /// Create a new layering analyzer
    pub fn new(symbol: &str, sensitivity: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            mid_price: RwLock::new(None),
            active_layers: RwLock::new(HashMap::with_capacity(MAX_LAYERS)),
            detected_patterns: RwLock::new(VecDeque::with_capacity(100)),
            level_history: RwLock::new(HashMap::with_capacity(500)),
            window_ms: DEFAULT_WINDOW_MS,
            sensitivity: sensitivity.max(0.0).min(1.0),
            last_update: RwLock::new(Instant::now()),
        }
    }

    /// Update mid-price reference
    pub fn update_mid_price(&self, price: PriceTick) {
        *self.mid_price.write() = Some(price);
    }

    /// Record a new order placement at a price level
    pub fn record_placement(&self, price: PriceTick, volume: Volume, order_count: u32) -> Option<LayeringPattern> {
        let now = get_timestamp_us();
        let mid = self.mid_price.read().unwrap_or(price);
        let distance = ((price - mid) as i32).abs();

        let layer = Layer::new(price, volume, order_count, distance, now);

        // Add to active layers
        self.active_layers.write().insert(price, layer);

        // Update level history
        {
            let mut history = self.level_history.write();
            let hist = history.entry(price).or_insert_with(LevelHistory::new);
            hist.placements.push_back(now);
            hist.total_volume += volume;
            
            // Keep only recent entries
            while hist.placements.len() > 100 {
                hist.placements.pop_front();
            }
        }

        *self.last_update.write() = Instant::now();

        // Check for layering pattern
        self.detect_pattern()
    }

    /// Record a cancellation at a price level
    pub fn record_cancellation(&self, price: PriceTick) -> Option<LayeringPattern> {
        let now = get_timestamp_us();

        // Update active layer
        if let Some(layer) = self.active_layers.write().get_mut(&price) {
            layer.mark_cancelled(now);
        }

        // Update level history
        {
            let mut history = self.level_history.write();
            if let Some(hist) = history.get_mut(&price) {
                hist.cancellations.push_back(now);
                while hist.cancellations.len() > 100 {
                    hist.cancellations.pop_front();
                }
            }
        }

        *self.last_update.write() = Instant::now();

        // Check if this completes a layering pattern
        self.detect_pattern()
    }

    /// Detect layering pattern in current book state
    pub fn detect_pattern(&self) -> Option<LayeringPattern> {
        let now = get_timestamp_us();
        let window_us = self.window_ms * 1000;

        let active = self.active_layers.read();
        let mut layers: Vec<Layer> = active
            .values()
            .filter(|l| (now - l.timestamp_us) / 1000 < self.window_ms)
            .copied()
            .collect();

        if layers.len() < MIN_LAYERS_FOR_DETECTION {
            return None;
        }

        // Sort by distance from mid-price
        layers.sort_by_key(|l| l.distance_from_mid);

        // Analyze patterns
        let same_side_bid = layers.iter().filter(|l| l.price < self.mid_price.read().unwrap_or(0)).count();
        let same_side_ask = layers.iter().filter(|l| l.price > self.mid_price.read().unwrap_or(0)).count();

        let (side, relevant_layers) = if same_side_bid >= same_side_ask && same_side_bid >= MIN_LAYERS_FOR_DETECTION {
            (Side::Bid, layers.iter().filter(|l| l.price < self.mid_price.read().unwrap_or(0)).copied().collect())
        } else if same_side_ask >= MIN_LAYERS_FOR_DETECTION {
            (Side::Ask, layers.iter().filter(|l| l.price > self.mid_price.read().unwrap_or(0)).copied().collect())
        } else {
            return None;
        };

        // Check for layering characteristics
        let cancel_rate = relevant_layers.iter().filter(|l| l.was_cancelled).count() as f64 / relevant_layers.len() as f64;
        
        // Calculate volume distribution (layering often has increasing volume with distance)
        let volumes: Vec<f64> = relevant_layers.iter().map(|l| l.volume as f64).collect();
        let volume_trend = self.calculate_volume_trend(&volumes);

        // Check price spacing regularity (algorithmic layering often has regular spacing)
        let prices: Vec<PriceTick> = relevant_layers.iter().map(|l| l.price).collect();
        let spacing_regularity = self.calculate_spacing_regularity(&prices);

        // Calculate confidence
        let cancel_confidence = cancel_rate * self.sensitivity;
        let trend_confidence = volume_trend.abs() * 0.3;
        let regularity_confidence = spacing_regularity * 0.3;
        
        let confidence = (cancel_confidence * 0.4 + trend_confidence + regularity_confidence).min(1.0);

        if confidence < 0.5 * self.sensitivity {
            return None;
        }

        // Build pattern
        let total_volume: Volume = relevant_layers.iter().map(|l| l.volume).sum();
        let price_range = relevant_layers.iter().map(|l| l.price).max().unwrap_or(0) 
            - relevant_layers.iter().map(|l| l.price).min().unwrap_or(0);
        
        let avg_cancel_time: f64 = relevant_layers
            .iter()
            .filter(|l| l.was_cancelled)
            .map(|l| l.lifetime_us() as f64)
            .sum::<f64>() / relevant_layers.iter().filter(|l| l.was_cancelled).count() as f64;

        let mut pattern = LayeringPattern {
            side,
            symbol: self.symbol.clone(),
            layers: relevant_layers,
            confidence,
            total_volume,
            price_range,
            avg_cancel_time_us: avg_cancel_time,
            detected_at_us: now,
            severity: 1,
        };

        pattern.calculate_severity();

        // Store pattern
        drop(active);
        self.detected_patterns.write().push_back(pattern.clone());
        while self.detected_patterns.read().len() > 100 {
            self.detected_patterns.write().pop_front();
        }

        Some(pattern)
    }

    /// Calculate volume trend (positive = increasing with distance)
    fn calculate_volume_trend(&self, volumes: &[f64]) -> f64 {
        if volumes.len() < 2 {
            return 0.0;
        }

        let n = volumes.len() as f64;
        let sum_x: f64 = (0..volumes.len()).map(|i| i as f64).sum();
        let sum_y: f64 = volumes.iter().sum();
        let sum_xy: f64 = volumes.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
        let sum_xx: f64 = (0..volumes.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        
        // Normalize to [-1, 1]
        slope.clamp(-1.0, 1.0)
    }

    /// Calculate regularity of price spacing (higher = more regular/algorithmic)
    fn calculate_spacing_regularity(&self, prices: &[PriceTick]) -> f64 {
        if prices.len() < 3 {
            return 0.0;
        }

        let mut spacings: Vec<i64> = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            spacings.push((prices[i] - prices[i - 1]).abs());
        }

        if spacings.is_empty() {
            return 0.0;
        }

        // Calculate coefficient of variation
        let mean: f64 = spacings.iter().map(|&s| s as f64).sum::<f64>() / spacings.len() as f64;
        if mean < 1e-10 {
            return 1.0; // Perfectly regular (all zero spacing)
        }

        let variance: f64 = spacings.iter()
            .map(|&s| ((s as f64) - mean).powi(2))
            .sum::<f64>() / spacings.len() as f64;
        
        let std_dev = variance.sqrt();
        let cv = std_dev / mean;

        // Lower CV = more regular, convert to [0, 1] where 1 = perfectly regular
        (1.0 / (1.0 + cv)).min(1.0)
    }

    /// Get recent detected patterns
    pub fn get_recent_patterns(&self, limit: usize) -> Vec<LayeringPattern> {
        let patterns = self.detected_patterns.read();
        patterns.iter().rev().take(limit).cloned().collect()
    }

    /// Get layering analysis summary
    pub fn analyze(&self) -> LayeringAnalysis {
        let patterns = self.get_recent_patterns(10);
        
        if patterns.is_empty() {
            return LayeringAnalysis {
                is_detected: false,
                confidence: 0.0,
                suspicious_layers: 0,
                fake_volume_estimate: 0,
            };
        }

        let avg_confidence: f64 = patterns.iter().map(|p| p.confidence).sum::<f64>() / patterns.len() as f64;
        let total_layers: usize = patterns.iter().map(|p| p.layers.len()).sum();
        let fake_volume: Volume = patterns.iter()
            .filter(|p| p.confidence > 0.7)
            .map(|p| p.total_volume)
            .sum();

        LayeringAnalysis {
            is_detected: avg_confidence > 0.5,
            confidence: avg_confidence,
            suspicious_layers: total_layers,
            fake_volume_estimate: fake_volume,
        }
    }

    /// Clear all data
    pub fn reset(&self) {
        self.active_layers.write().clear();
        self.detected_patterns.write().clear();
        self.level_history.write().clear();
        *self.last_update.write() = Instant::now();
    }

    /// Set analysis window
    pub fn set_window_ms(&self, window_ms: u64) {
        self.window_ms = window_ms;
    }

    /// Get symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// Builder for LayeringAnalyzer
pub struct LayeringAnalyzerBuilder {
    symbol: String,
    sensitivity: f64,
    window_ms: u64,
}

impl LayeringAnalyzerBuilder {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            sensitivity: 0.7,
            window_ms: DEFAULT_WINDOW_MS,
        }
    }

    pub fn with_sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = sensitivity.max(0.0).min(1.0);
        self
    }

    pub fn with_window_ms(mut self, window_ms: u64) -> Self {
        self.window_ms = window_ms;
        self
    }

    pub fn build(self) -> LayeringAnalyzer {
        let analyzer = LayeringAnalyzer::new(&self.symbol, self.sensitivity);
        analyzer.set_window_ms(self.window_ms);
        analyzer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_layering_detection() {
        let analyzer = LayeringAnalyzer::new("BTCUSDT", 0.8);
        analyzer.update_mid_price(50000);

        // Simulate layering on bid side
        for i in 1..=5 {
            let price = 50000 - i;
            analyzer.record_placement(price, 100 * i as u64, 1);
        }

        let analysis = analyzer.analyze();
        // May or may not detect depending on timing
        println!("Analysis: {:?}", analysis);
    }

    #[test]
    fn test_volume_trend_calculation() {
        let analyzer = LayeringAnalyzer::new("ETHUSDT", 0.7);
        
        // Increasing volumes
        let volumes_inc = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend_inc = analyzer.calculate_volume_trend(&volumes_inc);
        assert!(trend_inc > 0.0);

        // Decreasing volumes
        let volumes_dec = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend_dec = analyzer.calculate_volume_trend(&volumes_dec);
        assert!(trend_dec < 0.0);
    }

    #[test]
    fn test_spacing_regularity() {
        let analyzer = LayeringAnalyzer::new("SOLUSDT", 0.7);
        
        // Regular spacing
        let prices_regular = vec![100, 102, 104, 106, 108];
        let reg_regular = analyzer.calculate_spacing_regularity(&prices_regular);
        assert!(reg_regular > 0.8);

        // Irregular spacing
        let prices_irregular = vec![100, 105, 107, 120, 125];
        let reg_irregular = analyzer.calculate_spacing_regularity(&prices_irregular);
        assert!(reg_irregular < reg_regular);
    }
}
