//! Premium Index Calculator for Perpetual Swaps
//! 
//! This module calculates real-time premium/discount spreads between perpetual
//! swap prices and the underlying index price. It implements Binance's specific
//! moving average windows and handles edge cases during extreme volatility.
//!
//! Features:
//! - Real-time premium index calculation
//! - Binance-specific MA window handling (1h, 4h)
//! - Clamp logic for funding rate components
//! - Zero-cost abstractions for high-frequency updates

use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;

/// Price sample with timestamp for moving average calculations
#[derive(Debug, Clone)]
pub struct PriceSample {
    pub timestamp_us: u64,
    pub mark_price: f64,
    pub index_price: f64,
}

/// Moving average calculator with configurable window
pub struct MovingAverage {
    /// Window size in seconds
    window_seconds: u64,
    /// Rolling window of price samples
    samples: VecDeque<PriceSample>,
    /// Cached sum for O(1) average calculation
    cached_sum: f64,
}

impl MovingAverage {
    /// Create a new moving average calculator
    pub fn new(window_seconds: u64) -> Self {
        Self {
            window_seconds,
            samples: VecDeque::with_capacity(1000),
            cached_sum: 0.0,
        }
    }
    
    /// Add a new price sample, evicting old samples outside the window
    pub fn add_sample(&mut self, sample: PriceSample) {
        let window_us = self.window_seconds * 1_000_000;
        
        // Remove samples outside the window
        while let Some(front) = self.samples.front() {
            if sample.timestamp_us.saturating_sub(front.timestamp_us) > window_us {
                if let Some(evicted) = self.samples.pop_front() {
                    self.cached_sum -= evicted.index_price;
                }
            } else {
                break;
            }
        }
        
        // Add new sample
        self.cached_sum += sample.index_price;
        self.samples.push_back(sample);
    }
    
    /// Get the current moving average in O(1) time
    pub fn get_average(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.cached_sum / self.samples.len() as f64
    }
    
    /// Get the number of samples in the window
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
    
    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
        self.cached_sum = 0.0;
    }
}

/// Premium index calculator tracking mark vs index price divergence
pub struct PremiumIndex {
    /// Short-term MA (Binance uses 1-hour)
    ma_short: MovingAverage,
    /// Long-term MA (Binance uses 4-hour)
    ma_long: MovingAverage,
    /// Current mark price
    current_mark_price: f64,
    /// Current index price
    current_index_price: f64,
    /// Last update timestamp
    last_update_us: u64,
    /// Asset symbol
    symbol: String,
}

impl PremiumIndex {
    /// Create a new premium index calculator for an asset
    pub fn new(symbol: &str) -> Self {
        let now_us = Self::get_timestamp_us();
        
        Self {
            ma_short: MovingAverage::new(3600),  // 1-hour window
            ma_long: MovingAverage::new(14400),  // 4-hour window
            current_mark_price: 0.0,
            current_index_price: 0.0,
            last_update_us: now_us,
            symbol: symbol.to_string(),
        }
    }
    
    /// Get current timestamp in microseconds
    fn get_timestamp_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
    
    /// Update with new price data
    /// This is the main entry point for real-time price feeds
    pub fn update(&mut self, mark_price: f64, index_price: f64) {
        let now_us = Self::get_timestamp_us();
        
        self.current_mark_price = mark_price;
        self.current_index_price = index_price;
        self.last_update_us = now_us;
        
        // Create price sample
        let sample = PriceSample {
            timestamp_us: now_us,
            mark_price,
            index_price,
        };
        
        // Update both moving averages
        self.ma_short.add_sample(sample.clone());
        self.ma_long.add_sample(sample);
    }
    
    /// Calculate the current premium index
    /// Premium Index = (Mark Price - Index Price) / Index Price
    pub fn calculate_premium(&self) -> f64 {
        if self.current_index_price <= 0.0 {
            return 0.0;
        }
        (self.current_mark_price - self.current_index_price) / self.current_index_price
    }
    
    /// Calculate premium in basis points (bps)
    pub fn calculate_premium_bps(&self) -> f64 {
        self.calculate_premium() * 10000.0
    }
    
    /// Get the short-term MA of index price
    pub fn get_ma_short(&self) -> f64 {
        self.ma_short.get_average()
    }
    
    /// Get the long-term MA of index price
    pub fn get_ma_long(&self) -> f64 {
        self.ma_long.get_average()
    }
    
    /// Calculate the predicted funding rate component
    /// 
    /// Binance formula:
    /// Funding Rate = Premium Index + clamp(Interest Rate - Premium Index, -0.05%, 0.05%)
    /// 
    /// Interest Rate is typically 0.01% per 8 hours
    pub fn calculate_predicted_funding(&self, interest_rate: f64) -> f64 {
        let premium = self.calculate_premium();
        
        // Clamp component: clamp(interest_rate - premium, -0.0005, 0.0005)
        let clamp_component = (interest_rate - premium).clamp(-0.0005, 0.0005);
        
        // Final predicted funding rate
        premium + clamp_component
    }
    
    /// Check if premium indicates contango (perp trading at premium)
    pub fn is_contango(&self) -> bool {
        self.calculate_premium() > 0.0
    }
    
    /// Check if premium indicates backwardation (perp trading at discount)
    pub fn is_backwardation(&self) -> bool {
        self.calculate_premium() < 0.0
    }
    
    /// Get premium status as a string
    pub fn premium_status(&self) -> &'static str {
        let premium = self.calculate_premium();
        if premium > 0.0001 {
            "contango"
        } else if premium < -0.0001 {
            "backwardation"
        } else {
            "neutral"
        }
    }
    
    /// Detect abnormal premium spike (potential manipulation)
    pub fn detect_abnormal_premium(&self, threshold_bps: f64) -> bool {
        self.calculate_premium_bps().abs() > threshold_bps
    }
    
    /// Get statistics about the premium index
    pub fn get_statistics(&self) -> PremiumStatistics {
        let premium = self.calculate_premium();
        let premium_bps = self.calculate_premium_bps();
        
        PremiumStatistics {
            premium: premium,
            premium_bps: premium_bps,
            mark_price: self.current_mark_price,
            index_price: self.current_index_price,
            ma_short: self.ma_short.get_average(),
            ma_long: self.ma_long.get_average(),
            status: self.premium_status(),
            sample_count_short: self.ma_short.sample_count(),
            sample_count_long: self.ma_long.sample_count(),
        }
    }
}

/// Statistics snapshot for the premium index
#[derive(Debug)]
pub struct PremiumStatistics {
    pub premium: f64,
    pub premium_bps: f64,
    pub mark_price: f64,
    pub index_price: f64,
    pub ma_short: f64,
    pub ma_long: f64,
    pub status: &'static str,
    pub sample_count_short: usize,
    pub sample_count_long: usize,
}

/// Premium spread tracker for multiple assets
pub struct PremiumSpreadTracker {
    /// Trackers for individual assets
    trackers: std::collections::HashMap<String, PremiumIndex>,
}

impl PremiumSpreadTracker {
    /// Create a new multi-asset premium tracker
    pub fn new() -> Self {
        Self {
            trackers: std::collections::HashMap::new(),
        }
    }
    
    /// Add or get existing tracker for an asset
    pub fn get_or_create_tracker(&mut self, symbol: &str) -> &mut PremiumIndex {
        use std::collections::hash_map::Entry;
        
        match self.trackers.entry(symbol.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(PremiumIndex::new(symbol))
            }
            Entry::Occupied(entry) => entry.into_mut(),
        }
    }
    
    /// Update premium for a specific asset
    pub fn update_asset(&mut self, symbol: &str, mark_price: f64, index_price: f64) {
        let tracker = self.get_or_create_tracker(symbol);
        tracker.update(mark_price, index_price);
    }
    
    /// Get the asset with the highest premium (best for short perp arb)
    pub fn get_highest_premium_asset(&self) -> Option<(String, f64)> {
        self.trackers
            .iter()
            .map(|(symbol, tracker)| (symbol.clone(), tracker.calculate_premium_bps()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get the asset with the lowest premium (best for long perp arb)
    pub fn get_lowest_premium_asset(&self) -> Option<(String, f64)> {
        self.trackers
            .iter()
            .map(|(symbol, tracker)| (symbol.clone(), tracker.calculate_premium_bps()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get all premiums sorted by value
    pub fn get_sorted_premiums(&self) -> Vec<(String, f64)> {
        let mut premiums: Vec<_> = self.trackers
            .iter()
            .map(|(symbol, tracker)| (symbol.clone(), tracker.calculate_premium_bps()))
            .collect();
        
        premiums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        premiums
    }
}

impl Default for PremiumSpreadTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3600);
        
        let base_time = 1_000_000_000_000_000;
        for i in 0..10 {
            let sample = PriceSample {
                timestamp_us: base_time + i * 100_000,
                mark_price: 50000.0,
                index_price: 49900.0 + (i as f64),
            };
            ma.add_sample(sample);
        }
        
        assert_eq!(ma.sample_count(), 10);
        assert!(ma.get_average() > 0.0);
    }
    
    #[test]
    fn test_premium_calculation() {
        let mut premium = PremiumIndex::new("BTCUSDT");
        premium.update(50100.0, 50000.0);
        
        let expected_premium = (50100.0 - 50000.0) / 50000.0;
        assert!((premium.calculate_premium() - expected_premium).abs() < 1e-10);
        assert!(premium.is_contango());
    }
    
    #[test]
    fn test_funding_prediction() {
        let mut premium = PremiumIndex::new("BTCUSDT");
        premium.update(50100.0, 50000.0);
        
        let interest_rate = 0.0001;
        let predicted = premium.calculate_predicted_funding(interest_rate);
        
        // Should be positive since premium is positive
        assert!(predicted > 0.0);
    }
}
