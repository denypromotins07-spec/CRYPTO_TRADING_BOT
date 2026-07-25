// ZAID PERSONAL CRYPTO TRADING BOT - Stage 7
// Chapter 4: Data Fusion - Signal Normalizer
//
// File: backend/fusion/signal_normalizer.rs
// Purpose: Z-score normalize alternative data against order flow.
//          Ensure alt-data doesn't overpower hard order flow signals.
//
// Features:
// - Z-score normalization across signal types
// - Order flow dominance weighting
// - Adaptive signal scaling
// - Memory-efficient rolling statistics
//
// Design Patterns:
// - Strategy: Different normalization methods
// - Adapter: Convert signals to normalized form
//
// Author: Opus 4.8
// Domain: Signal Processing, Statistical Normalization, Quant Finance

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use log::{info, debug};

/// Maximum history for rolling statistics (memory bound)
const MAX_ROLLING_HISTORY: usize = 2000;

/// Normalized signal output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedSignal {
    pub source: String,
    pub asset: String,
    pub timestamp: u64,
    pub raw_value: f64,
    pub z_score: f64,
    pub percentile_rank: f64,
    pub order_flow_weight: f64,
    pub final_score: f64,
}

/// Rolling statistics calculator
#[derive(Debug, Clone)]
pub struct RollingStats {
    values: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
    count: usize,
}

impl RollingStats {
    pub fn new(max_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(max_size),
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
        }
    }
    
    pub fn add(&mut self, value: f64) {
        if self.values.len() >= self.values.capacity() {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
                self.count -= 1;
            }
        }
        
        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
    }
    
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        Some(self.sum / self.count as f64)
    }
    
    pub fn std_dev(&self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        
        let mean = self.mean()?;
        let variance = (self.sum_sq / self.count as f64) - (mean * mean);
        
        if variance < 0.0 {
            return Some(0.0);
        }
        
        Some(variance.sqrt())
    }
    
    pub fn z_score(&self, value: f64) -> Option<f64> {
        let mean = self.mean()?;
        let std = self.std_dev()?;
        
        if std == 0.0 {
            return Some(0.0);
        }
        
        Some((value - mean) / std)
    }
    
    pub fn percentile_rank(&self, value: f64) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        
        let count_below = self.values.iter().filter(|&&v| v < value).count();
        Some(count_below as f64 / self.values.len() as f64)
    }
}

/// Signal normalizer with order flow integration
pub struct SignalNormalizer {
    /// Rolling stats per signal source
    signal_stats: DashMap<String, RollingStats>,
    
    /// Order flow strength per asset
    order_flow_strength: DashMap<String, f64>,
    
    /// Configuration
    config: NormalizerConfig,
}

#[derive(Debug, Clone)]
pub struct NormalizerConfig {
    pub max_history: usize,
    pub order_flow_dominance: f64,  // 0-1, how much order flow dominates
    pub min_z_score: f64,
    pub max_z_score: f64,
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        Self {
            max_history: MAX_ROLLING_HISTORY,
            order_flow_dominance: 0.6,  // Order flow gets 60% weight minimum
            min_z_score: -3.0,
            max_z_score: 3.0,
        }
    }
}

impl SignalNormalizer {
    /// Create new signal normalizer
    pub fn new(config: NormalizerConfig) -> Self {
        info!("Initializing SignalNormalizer");
        
        Self {
            signal_stats: DashMap::new(),
            order_flow_strength: DashMap::new(),
            config,
        }
    }
    
    /// Update order flow strength for an asset
    pub fn update_order_flow(&self, asset: String, strength: f64) {
        // Strength is typically 0-1 based on order book depth/volume
        let normalized = strength.max(0.0).min(1.0);
        self.order_flow_strength.insert(asset, normalized);
    }
    
    /// Normalize a signal with order flow integration
    pub fn normalize_signal(
        &self,
        source: &str,
        asset: &str,
        raw_value: f64,
        timestamp: u64
    ) -> Option<NormalizedSignal> {
        // Get or create rolling stats for this source
        let mut stats = self.signal_stats
            .entry(source.to_string())
            .or_insert_with(|| RollingStats::new(self.config.max_history));
        
        // Calculate z-score before adding new value
        let z_score = stats.z_score(raw_value).unwrap_or(0.0);
        let percentile = stats.percentile_rank(raw_value).unwrap_or(0.5);
        
        // Add new value to rolling stats
        stats.add(raw_value);
        
        // Clamp z-score
        let clamped_z = z_score.max(self.config.min_z_score).min(self.config.max_z_score);
        
        // Get order flow strength
        let of_strength = self.order_flow_strength.get(asset).map(|r| *r).unwrap_or(0.5);
        
        // Calculate order flow weight (prevents alt-data from overpowering)
        // When order flow is strong, reduce alt-data influence
        let order_flow_weight = self.config.order_flow_dominance + (1.0 - self.config.order_flow_dominance) * (1.0 - of_strength);
        
        // Final score combines z-score with order flow adjustment
        // Normalize z-score to 0-1 range, then apply order flow weight
        let normalized_z = (clamped_z + 3.0) / 6.0;  // Convert -3..3 to 0..1
        let final_score = normalized_z * (1.0 - order_flow_weight * 0.5);
        
        Some(NormalizedSignal {
            source: source.to_string(),
            asset: asset.to_string(),
            timestamp,
            raw_value,
            z_score: clamped_z,
            percentile_rank: percentile,
            order_flow_weight,
            final_score,
        })
    }
    
    /// Normalize multiple signals and return weighted composite
    pub fn normalize_composite(
        &self,
        signals: Vec<(&str, &str, f64)>,  // (source, asset, raw_value)
        timestamp: u64
    ) -> HashMap<String, NormalizedSignal> {
        let mut results = HashMap::new();
        
        for (source, asset, raw_value) in signals {
            if let Some(normalized) = self.normalize_signal(source, asset, raw_value, timestamp) {
                results.insert(format!("{}:{}", asset, source), normalized);
            }
        }
        
        results
    }
    
    /// Get current statistics for a signal source
    pub fn get_stats(&self, source: &str) -> Option<(f64, f64)> {
        let stats = self.signal_stats.get(source)?;
        Some((stats.mean()?, stats.std_dev()?))
    }
    
    /// Reset statistics for a source (e.g., after regime change)
    pub fn reset_source(&self, source: &str) {
        self.signal_stats.remove(source);
        info!("Reset statistics for signal source: {}", source);
    }
    
    /// Detect signal anomalies (extreme z-scores)
    pub fn detect_anomalies(&self, threshold: f64) -> Vec<(String, f64)> {
        let mut anomalies = Vec::new();
        
        for entry in self.signal_stats.iter() {
            let source = entry.key();
            let stats = entry.value();
            
            if let Some(mean) = stats.mean() {
                if let Some(std) = stats.std_dev() {
                    if std > 0.0 {
                        // Check if recent values are anomalous
                        let recent_values: Vec<f64> = stats.values.iter().rev().take(5).cloned().collect();
                        
                        for value in recent_values {
                            let z = (value - mean) / std;
                            if z.abs() > threshold {
                                anomalies.push((source.clone(), z));
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        anomalies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(100);
        
        // Add some test data
        for i in 0..50 {
            stats.add(i as f64);
        }
        
        let mean = stats.mean().unwrap();
        assert!(mean > 20.0 && mean < 30.0);
        
        let std = stats.std_dev().unwrap();
        assert!(std > 0.0);
    }
    
    #[test]
    fn test_z_score_normalization() {
        let normalizer = SignalNormalizer::new(NormalizerConfig::default());
        
        // Add baseline data
        for i in 0..100 {
            normalizer.normalize_signal("test_source", "BTC", 50.0 + (i % 10) as f64, i);
        }
        
        // Test extreme value
        let result = normalizer.normalize_signal("test_source", "BTC", 100.0, 101).unwrap();
        assert!(result.z_score > 2.0);
        
        // Test normal value
        let result = normalizer.normalize_signal("test_source", "BTC", 55.0, 102).unwrap();
        assert!(result.z_score.abs() < 1.0);
    }
    
    #[test]
    fn test_order_flow_weighting() {
        let normalizer = SignalNormalizer::new(NormalizerConfig::default());
        
        // Set strong order flow
        normalizer.update_order_flow("BTC".to_string(), 0.9);
        
        let result_weak_of = normalizer.normalize_signal("alt_data", "BTC", 75.0, 1).unwrap();
        
        // Set weak order flow
        normalizer.update_order_flow("BTC".to_string(), 0.1);
        
        let result_strong_of = normalizer.normalize_signal("alt_data", "BTC", 75.0, 2).unwrap();
        
        // With weak order flow, alt-data should have more influence
        assert!(result_strong_of.final_score > result_weak_of.final_score);
    }
}

fn main() {
    env_logger::init();
    
    let config = NormalizerConfig::default();
    let normalizer = SignalNormalizer::new(config);
    
    info!("SignalNormalizer initialized and ready");
}
