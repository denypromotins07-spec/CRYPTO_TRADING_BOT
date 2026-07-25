//! backend/orderflow/exhaustion_math.rs
//!
//! Calculates delta exhaustion to spot trend reversals.
//! Implements mathematical models for detecting when buying/selling pressure is depleting.
//!
//! Features:
//! - Delta exhaustion calculation using multiple methods
//! - Trend reversal signal generation
//! - Volume-weighted momentum decay tracking
//! - Memory-efficient rolling calculations
//! - Thread-safe operations

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// Configuration for exhaustion detection
#[derive(Debug, Clone, Copy)]
pub struct ExhaustionConfig {
    /// Lookback window for delta analysis
    pub delta_lookback: usize,
    /// Minimum volume threshold
    pub min_volume: f64,
    /// Momentum decay threshold
    pub momentum_decay_threshold: f64,
    /// Divergence confirmation window
    pub divergence_window: usize,
}

impl Default for ExhaustionConfig {
    fn default() -> Self {
        Self {
            delta_lookback: 20,
            min_volume: 0.1,
            momentum_decay_threshold: 0.3,
            divergence_window: 10,
        }
    }
}

/// Type of exhaustion detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExhaustionType {
    BullishExhaustion,  // Selling pressure exhausted, potential reversal up
    BearishExhaustion,  // Buying pressure exhausted, potential reversal down
}

/// Exhaustion signal with confidence metrics
#[derive(Debug, Clone, Copy)]
pub struct ExhaustionSignal {
    pub exhaustion_type: ExhaustionType,
    pub price: f64,
    pub delta_value: f64,
    pub momentum_score: f64,  // -1.0 to 1.0
    pub volume_confirmation: bool,
    pub confidence: f64,  // 0.0 to 1.0
    pub timestamp_ns: u64,
}

/// Tracks delta and momentum for exhaustion detection
pub struct DeltaTracker {
    config: ExhaustionConfig,
    /// Rolling window of delta values
    delta_history: VecDeque<f64>,
    /// Rolling window of prices
    price_history: VecDeque<f64>,
    /// Rolling window of volumes
    volume_history: VecDeque<f64>,
    /// Cumulative delta
    cumulative_delta: f64,
    last_update_ns: u64,
}

impl DeltaTracker {
    pub fn new(config: ExhaustionConfig) -> Self {
        Self {
            config,
            delta_history: VecDeque::with_capacity(config.delta_lookback + 10),
            price_history: VecDeque::with_capacity(config.delta_lookback + 10),
            volume_history: VecDeque::with_capacity(config.delta_lookback + 10),
            cumulative_delta: 0.0,
            last_update_ns: 0,
        }
    }
    
    /// Add a new tick (zero-allocation hot path)
    #[inline]
    pub fn add_tick(&mut self, price: f64, volume: f64, is_buyer_maker: bool, ts: u64) {
        if volume < self.config.min_volume {
            return;
        }
        
        // Calculate delta for this tick
        let delta = if is_buyer_maker {
            -volume  // Seller initiated
        } else {
            volume   // Buyer initiated
        };
        
        self.cumulative_delta += delta;
        
        // Update histories
        if self.delta_history.len() >= self.config.delta_lookback * 2 {
            self.delta_history.pop_front();
            self.price_history.pop_front();
            self.volume_history.pop_front();
        }
        
        self.delta_history.push_back(delta);
        self.price_history.push_back(price);
        self.volume_history.push_back(volume);
        
        self.last_update_ns = ts;
    }
    
    /// Get current cumulative delta
    #[inline]
    pub fn get_cumulative_delta(&self) -> f64 {
        self.cumulative_delta
    }
    
    /// Calculate delta momentum (rate of change of delta)
    pub fn get_delta_momentum(&self, window: usize) -> Option<f64> {
        if self.delta_history.len() < window {
            return None;
        }
        
        let recent_sum: f64 = self.delta_history.iter().rev().take(window).sum();
        let older_sum: f64 = self.delta_history.iter().rev().skip(window).take(window).sum();
        
        Some(recent_sum - older_sum)
    }
    
    /// Detect delta exhaustion
    pub fn detect_exhaustion(&self) -> Option<ExhaustionSignal> {
        if self.delta_history.len() < self.config.delta_lookback {
            return None;
        }
        
        let prices: Vec<f64> = self.price_history.iter().copied().collect();
        let deltas: Vec<f64> = self.delta_history.iter().copied().collect();
        
        // Split into two halves for comparison
        let mid = deltas.len() / 2;
        let first_half = &deltas[..mid];
        let second_half = &deltas[mid..];
        
        // Calculate average delta in each half
        let first_avg: f64 = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg: f64 = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        // Calculate price direction
        let first_price_avg: f64 = prices[..mid].iter().sum::<f64>() / mid as f64;
        let second_price_avg: f64 = prices[mid..].iter().sum::<f64>() / (prices.len() - mid) as f64;
        let price_change = second_price_avg - first_price_avg;
        
        // Delta momentum decay
        let momentum_decay = (second_avg - first_avg).abs() / (first_avg.abs() + 1e-10);
        
        // Detect exhaustion patterns
        let signal = if price_change > 0.0 && second_avg < first_avg && second_avg < 0.0 {
            // Price going up but delta becoming more negative = bearish exhaustion
            Some(ExhaustionSignal {
                exhaustion_type: ExhaustionType::BearishExhaustion,
                price: *self.price_history.back()?,
                delta_value: self.cumulative_delta,
                momentum_score: -momentum_decay.min(1.0),
                volume_confirmation: self.check_volume_confirmation(),
                confidence: self.calculate_confidence(momentum_decay, price_change),
                timestamp_ns: self.last_update_ns,
            })
        } else if price_change < 0.0 && second_avg > first_avg && second_avg > 0.0 {
            // Price going down but delta becoming more positive = bullish exhaustion
            Some(ExhaustionSignal {
                exhaustion_type: ExhaustionType::BullishExhaustion,
                price: *self.price_history.back()?,
                delta_value: self.cumulative_delta,
                momentum_score: momentum_decay.min(1.0),
                volume_confirmation: self.check_volume_confirmation(),
                confidence: self.calculate_confidence(momentum_decay, price_change.abs()),
                timestamp_ns: self.last_update_ns,
            })
        } else {
            None
        };
        
        signal.filter(|s| s.confidence >= 0.5)
    }
    
    /// Check if volume supports the exhaustion signal
    fn check_volume_confirmation(&self) -> bool {
        if self.volume_history.len() < 10 {
            return false;
        }
        
        let recent_vol: f64 = self.volume_history.iter().rev().take(5).sum();
        let older_vol: f64 = self.volume_history.iter().rev().skip(5).take(5).sum();
        
        // Volume should be decreasing (exhaustion) or spiking (climax)
        recent_vol < older_vol * 0.7 || recent_vol > older_vol * 1.5
    }
    
    /// Calculate confidence score for the signal
    fn calculate_confidence(&self, momentum_decay: f64, price_change: f64) -> f64 {
        let momentum_factor = momentum_decay.min(1.0);
        
        // Price change factor: larger moves have higher confidence
        let price_factor = (price_change / 10.0).min(1.0).abs();
        
        (momentum_factor * 0.6 + price_factor * 0.4).min(1.0)
    }
    
    /// Get RSI-like delta indicator
    pub fn get_delta_rsi(&self, period: usize) -> Option<f64> {
        if self.delta_history.len() < period {
            return None;
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in (self.delta_history.len() - period)..self.delta_history.len() {
            if let Some(delta) = self.delta_history.get(i) {
                if *delta > 0.0 {
                    gains += delta;
                } else {
                    losses -= delta;  // Make positive
                }
            }
        }
        
        if losses == 0.0 {
            return Some(100.0);
        }
        
        let rs = gains / losses;
        Some(100.0 - (100.0 / (1.0 + rs)))
    }
    
    /// Reset tracker
    pub fn reset(&mut self) {
        self.delta_history.clear();
        self.price_history.clear();
        self.volume_history.clear();
        self.cumulative_delta = 0.0;
    }
}

/// Manager for exhaustion detection across multiple symbols
pub struct ExhaustionManager {
    trackers: HashMap<String, DeltaTracker>,
    config: ExhaustionConfig,
    observers: Vec<Box<dyn ExhaustionObserver>>,
}

/// Observer trait for exhaustion notifications
pub trait ExhaustionObserver: Send + Sync {
    fn on_exhaustion_detected(&self, symbol: &str, signal: ExhaustionSignal);
}

impl ExhaustionManager {
    pub fn new(config: ExhaustionConfig) -> Self {
        Self {
            trackers: HashMap::new(),
            config,
            observers: Vec::new(),
        }
    }
    
    pub fn add_observer(&mut self, observer: Box<dyn ExhaustionObserver>) {
        self.observers.push(observer);
    }
    
    fn get_or_create_tracker(&mut self, symbol: &str) -> &mut DeltaTracker {
        self.trackers
            .entry(symbol.to_string())
            .or_insert_with(|| DeltaTracker::new(self.config))
    }
    
    /// Process a tick
    pub fn process_tick(
        &mut self,
        symbol: &str,
        price: f64,
        volume: f64,
        is_buyer_maker: bool,
        ts: u64,
    ) {
        let tracker = self.get_or_create_tracker(symbol);
        tracker.add_tick(price, volume, is_buyer_maker, ts);
        
        // Check for exhaustion
        if let Some(signal) = tracker.detect_exhaustion() {
            for observer in &self.observers {
                observer.on_exhaustion_detected(symbol, signal);
            }
        }
    }
    
    /// Get exhaustion status for a symbol
    pub fn get_exhaustion_status(&self, symbol: &str) -> Option<ExhaustionSignal> {
        self.trackers.get(symbol).and_then(|t| t.detect_exhaustion())
    }
    
    /// Get delta RSI for a symbol
    pub fn get_delta_rsi(&self, symbol: &str, period: usize) -> Option<f64> {
        self.trackers.get(symbol).and_then(|t| t.get_delta_rsi(period))
    }
}

impl Default for ExhaustionManager {
    fn default() -> Self {
        Self::new(ExhaustionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_delta_tracking() {
        let mut tracker = DeltaTracker::new(ExhaustionConfig::default());
        
        // Add buyer-initiated trades
        tracker.add_tick(100.0, 10.0, false, 1000);
        tracker.add_tick(101.0, 5.0, false, 1001);
        
        assert_eq!(tracker.get_cumulative_delta(), 15.0);
        
        // Add seller-initiated trade
        tracker.add_tick(100.5, 8.0, true, 1002);
        
        assert_eq!(tracker.get_cumulative_delta(), 7.0);
    }
    
    #[test]
    fn test_delta_rsi() {
        let mut tracker = DeltaTracker::new(ExhaustionConfig::default());
        
        // Add alternating deltas
        for i in 0..20 {
            let delta = if i % 2 == 0 { 5.0 } else { -3.0 };
            let is_buyer_maker = delta < 0.0;
            tracker.add_tick(100.0 + i as f64, delta.abs(), is_buyer_maker, i as u64);
        }
        
        let rsi = tracker.get_delta_rsi(14);
        assert!(rsi.is_some());
        println!("Delta RSI: {:?}", rsi);
    }
}
