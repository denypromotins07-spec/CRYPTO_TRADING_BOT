//! Toxicity Scorer - Order Flow Toxicity Grading
//!
//! This module grades order flow toxicity to prevent adverse selection.
//! It combines multiple signals including VPIN, order imbalance, and
//! cancellation patterns to produce a comprehensive toxicity score.
//!
//! Designed for the ZAID PERSONAL CRYPTO TRADING BOT with 8GB RAM constraints.
//! Uses zero-cost abstractions and efficient signal aggregation.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;

/// Maximum history size
const MAX_HISTORY_SIZE: usize = 500;

/// Default lookback period for calculations
const DEFAULT_LOOKBACK: usize = 100;

/// Price tick type
pub type PriceTick = i64;

/// Volume type
pub type Volume = u64;

/// Toxicity level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityLevel {
    /// Safe to trade normally
    Low,
    /// Exercise caution
    Medium,
    /// High adverse selection risk
    High,
    /// Extreme toxicity - avoid trading
    Extreme,
}

impl ToxicityLevel {
    #[inline]
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            ToxicityLevel::Extreme
        } else if score >= 0.6 {
            ToxicityLevel::High
        } else if score >= 0.4 {
            ToxicityLevel::Medium
        } else {
            ToxicityLevel::Low
        }
    }

    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            ToxicityLevel::Low => "LOW",
            ToxicityLevel::Medium => "MEDIUM",
            ToxicityLevel::High => "HIGH",
            ToxicityLevel::Extreme => "EXTREME",
        }
    }
}

/// Toxicity snapshot with all component scores
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ToxicitySnapshot {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Overall toxicity score [0, 1]
    pub overall_score: f64,
    /// VPIN-based toxicity component
    pub vpin_component: f64,
    /// Order imbalance component
    pub imbalance_component: f64,
    /// Cancellation rate component
    pub cancel_component: f64,
    /// Spread anomaly component
    pub spread_component: f64,
    /// Volume anomaly component
    pub volume_component: f64,
    /// Classified toxicity level
    pub level: ToxicityLevel,
    /// Recommended spread adjustment (basis points)
    pub spread_adjustment_bps: f64,
    /// Whether to pause trading
    pub should_pause: bool,
}

impl ToxicitySnapshot {
    #[inline]
    pub const fn new() -> Self {
        Self {
            timestamp_us: 0,
            overall_score: 0.0,
            vpin_component: 0.0,
            imbalance_component: 0.0,
            cancel_component: 0.0,
            spread_component: 0.0,
            volume_component: 0.0,
            level: ToxicityLevel::Low,
            spread_adjustment_bps: 0.0,
            should_pause: false,
        }
    }

    /// Calculate weighted overall score
    #[inline]
    pub fn calculate_overall(
        vpin: f64,
        imbalance: f64,
        cancel: f64,
        spread: f64,
        volume: f64,
    ) -> f64 {
        // Weights for different components
        const VPIN_WEIGHT: f64 = 0.35;
        const IMBALANCE_WEIGHT: f64 = 0.20;
        const CANCEL_WEIGHT: f64 = 0.20;
        const SPREAD_WEIGHT: f64 = 0.15;
        const VOLUME_WEIGHT: f64 = 0.10;

        let score = vpin * VPIN_WEIGHT
            + imbalance * IMBALANCE_WEIGHT
            + cancel * CANCEL_WEIGHT
            + spread * SPREAD_WEIGHT
            + volume * VOLUME_WEIGHT;

        score.clamp(0.0, 1.0)
    }

    /// Calculate recommended spread adjustment
    #[inline]
    pub fn calculate_spread_adjustment(score: f64) -> f64 {
        // Linear scaling: 0 score = 0 bps, 1.0 score = 20 bps
        score * 20.0
    }
}

impl Default for ToxicitySnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Order flow toxicity scorer
pub struct ToxicityScorer {
    /// Symbol being tracked
    symbol: String,
    /// Recent VPIN values
    vpin_history: RwLock<VecDeque<f64>>,
    /// Recent order imbalance values
    imbalance_history: RwLock<VecDeque<f64>>,
    /// Cancellation rate
    cancel_rate: RwLock<f64>,
    /// Normal spread baseline
    normal_spread: RwLock<f64>,
    /// Current spread
    current_spread: RwLock<f64>,
    /// Normal volume baseline
    normal_volume: RwLock<f64>,
    /// Current volume
    current_volume: RwLock<f64>,
    /// History of toxicity snapshots
    history: RwLock<VecDeque<ToxicitySnapshot>>,
    /// Current snapshot
    current_snapshot: RwLock<ToxicitySnapshot>,
    /// Lookback period
    lookback: usize,
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

impl ToxicityScorer {
    /// Create a new toxicity scorer
    pub fn new(symbol: &str, lookback: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            vpin_history: RwLock::new(VecDeque::with_capacity(lookback)),
            imbalance_history: RwLock::new(VecDeque::with_capacity(lookback)),
            cancel_rate: RwLock::new(0.0),
            normal_spread: RwLock::new(0.01), // Default 1%
            current_spread: RwLock::new(0.01),
            normal_volume: RwLock::new(1000.0),
            current_volume: RwLock::new(0.0),
            history: RwLock::new(VecDeque::with_capacity(MAX_HISTORY_SIZE)),
            current_snapshot: RwLock::new(ToxicitySnapshot::new()),
            lookback: lookback.min(DEFAULT_LOOKBACK),
            last_update: RwLock::new(Instant::now()),
        }
    }

    /// Update VPIN value
    pub fn update_vpin(&self, vpin: f64) {
        let mut history = self.vpin_history.write();
        history.push_back(vpin.clamp(0.0, 1.0));
        while history.len() > self.lookback {
            history.pop_front();
        }
        
        self.update_snapshot();
    }

    /// Update order imbalance
    pub fn update_imbalance(&self, imbalance: f64) {
        let mut history = self.imbalance_history.write();
        // Convert imbalance from [-1, 1] to [0, 1] toxicity
        let toxic_imbalance = imbalance.abs();
        history.push_back(toxic_imbalance);
        while history.len() > self.lookback {
            history.pop_front();
        }
        
        self.update_snapshot();
    }

    /// Update cancellation rate
    pub fn update_cancel_rate(&self, rate: f64) {
        *self.cancel_rate.write() = rate.clamp(0.0, 1.0);
        self.update_snapshot();
    }

    /// Update spread information
    pub fn update_spread(&self, current: f64, normal: Option<f64>) {
        *self.current_spread.write() = current;
        if let Some(n) = normal {
            *self.normal_spread.write() = n;
        }
        self.update_snapshot();
    }

    /// Update volume information
    pub fn update_volume(&self, current: f64, normal: Option<f64>) {
        *self.current_volume.write() = current;
        if let Some(n) = normal {
            *self.normal_volume.write() = n;
        }
        self.update_snapshot();
    }

    /// Update the toxicity snapshot
    fn update_snapshot(&self) {
        let now = get_timestamp_us();
        
        let vpin_history = self.vpin_history.read();
        let imbalance_history = self.imbalance_history.read();
        let cancel_rate = *self.cancel_rate.read();
        let current_spread = *self.current_spread.read();
        let normal_spread = *self.normal_spread.read();
        let current_volume = *self.current_volume.read();
        let normal_volume = *self.normal_volume.read();

        let mut snapshot = ToxicitySnapshot {
            timestamp_us: now,
            ..ToxicitySnapshot::new()
        };

        // Component 1: VPIN toxicity
        snapshot.vpin_component = if vpin_history.is_empty() {
            0.0
        } else {
            vpin_history.iter().sum::<f64>() / vpin_history.len() as f64
        };

        // Component 2: Imbalance toxicity
        snapshot.imbalance_component = if imbalance_history.is_empty() {
            0.0
        } else {
            imbalance_history.iter().sum::<f64>() / imbalance_history.len() as f64
        };

        // Component 3: Cancellation toxicity
        snapshot.cancel_component = cancel_rate;

        // Component 4: Spread anomaly
        if normal_spread > 0.0 {
            let spread_ratio = current_spread / normal_spread;
            snapshot.spread_component = if spread_ratio > 2.0 {
                ((spread_ratio - 1.0) / 3.0).min(1.0)
            } else {
                0.0
            };
        }

        // Component 5: Volume anomaly
        if normal_volume > 0.0 && current_volume > 0.0 {
            let volume_ratio = current_volume / normal_volume;
            // Both unusually high and low volume can be toxic
            if volume_ratio > 3.0 {
                snapshot.volume_component = ((volume_ratio - 1.0) / 5.0).min(1.0);
            } else if volume_ratio < 0.3 {
                snapshot.volume_component = (1.0 - volume_ratio) * 0.5;
            } else {
                snapshot.volume_component = 0.0;
            }
        }

        // Calculate overall score
        snapshot.overall_score = ToxicitySnapshot::calculate_overall(
            snapshot.vpin_component,
            snapshot.imbalance_component,
            snapshot.cancel_component,
            snapshot.spread_component,
            snapshot.volume_component,
        );

        // Classify level
        snapshot.level = ToxicityLevel::from_score(snapshot.overall_score);

        // Calculate spread adjustment
        snapshot.spread_adjustment_bps = ToxicitySnapshot::calculate_spread_adjustment(snapshot.overall_score);

        // Determine if trading should pause
        snapshot.should_pause = snapshot.level == ToxicityLevel::Extreme 
            || (snapshot.level == ToxicityLevel::High && snapshot.overall_score > 0.7);

        // Store snapshot
        drop(vpin_history);
        drop(imbalance_history);

        let mut history = self.history.write();
        *self.current_snapshot.write() = snapshot;
        
        history.push_back(snapshot);
        while history.len() > MAX_HISTORY_SIZE {
            history.pop_front();
        }

        *self.last_update.write() = Instant::now();
    }

    /// Get current toxicity snapshot
    pub fn get_snapshot(&self) -> ToxicitySnapshot {
        *self.current_snapshot.read()
    }

    /// Get overall toxicity score
    pub fn get_toxicity_score(&self) -> f64 {
        self.current_snapshot.read().overall_score
    }

    /// Get toxicity level
    pub fn get_toxicity_level(&self) -> ToxicityLevel {
        self.current_snapshot.read().level
    }

    /// Check if order flow is toxic
    pub fn is_toxic(&self) -> bool {
        let level = self.get_toxicity_level();
        level == ToxicityLevel::High || level == ToxicityLevel::Extreme
    }

    /// Check if trading should be paused
    pub fn should_pause_trading(&self) -> bool {
        self.current_snapshot.read().should_pause
    }

    /// Get recommended spread adjustment in basis points
    pub fn get_spread_adjustment_bps(&self) -> f64 {
        self.current_snapshot.read().spread_adjustment_bps
    }

    /// Get average toxicity over recent period
    pub fn get_average_toxicity(&self, periods: usize) -> f64 {
        let history = self.history.read();
        if history.is_empty() {
            return 0.0;
        }

        let take = periods.min(history.len());
        let sum: f64 = history.iter().rev().take(take).map(|s| s.overall_score).sum();
        sum / take as f64
    }

    /// Get toxicity trend (positive = increasing toxicity)
    pub fn get_toxicity_trend(&self, periods: usize) -> f64 {
        let history = self.history.read();
        if history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = history.iter().rev().take(periods.min(history.len())).map(|s| s.overall_score).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }

    /// Get VPIN component
    pub fn get_vpin_component(&self) -> f64 {
        self.current_snapshot.read().vpin_component
    }

    /// Reset all data
    pub fn reset(&self) {
        self.vpin_history.write().clear();
        self.imbalance_history.write().clear();
        *self.cancel_rate.write() = 0.0;
        self.history.write().clear();
        *self.current_snapshot.write() = ToxicitySnapshot::new();
        *self.last_update.write() = Instant::now();
    }

    /// Get symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// Builder for ToxicityScorer
pub struct ToxicityScorerBuilder {
    symbol: String,
    lookback: usize,
}

impl ToxicityScorerBuilder {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            lookback: DEFAULT_LOOKBACK,
        }
    }

    pub fn with_lookback(mut self, lookback: usize) -> Self {
        self.lookback = lookback;
        self
    }

    pub fn build(self) -> ToxicityScorer {
        ToxicityScorer::new(&self.symbol, self.lookback)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toxicity_level_classification() {
        assert_eq!(ToxicityLevel::from_score(0.0), ToxicityLevel::Low);
        assert_eq!(ToxicityLevel::from_score(0.3), ToxicityLevel::Low);
        assert_eq!(ToxicityLevel::from_score(0.5), ToxicityLevel::Medium);
        assert_eq!(ToxicityLevel::from_score(0.7), ToxicityLevel::High);
        assert_eq!(ToxicityLevel::from_score(0.9), ToxicityLevel::Extreme);
    }

    #[test]
    fn test_basic_scoring() {
        let scorer = ToxicityScorer::new("BTCUSDT", 50);
        
        // Initially should be low toxicity
        assert_eq!(scorer.get_toxicity_level(), ToxicityLevel::Low);
        assert!(!scorer.is_toxic());
        assert!(!scorer.should_pause_trading());
        
        // Add high VPIN
        scorer.update_vpin(0.8);
        scorer.update_imbalance(0.9);
        scorer.update_cancel_rate(0.8);
        
        let snapshot = scorer.get_snapshot();
        assert!(snapshot.overall_score > 0.5);
    }

    #[test]
    fn test_spread_adjustment() {
        let adjustment = ToxicitySnapshot::calculate_spread_adjustment(0.5);
        assert_eq!(adjustment, 10.0); // 0.5 * 20 = 10 bps
        
        let adjustment = ToxicitySnapshot::calculate_spread_adjustment(1.0);
        assert_eq!(adjustment, 20.0);
    }
}
