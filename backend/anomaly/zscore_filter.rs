//! Z-Score Filter with Dynamic Standard Deviation Bands
//! 
//! This module implements adaptive z-score filtering that dynamically adjusts
//! standard deviation bands during volatility spikes to prevent false anomaly
//! detection during flash crashes and market stress.
//! 
//! Key Features:
//! - Exponentially weighted moving statistics for rapid adaptation
//! - Volatility regime detection for band adjustment
//! - Multi-scale z-score computation (short/medium/long term)
//! - Memory-efficient O(1) updates per observation

use std::collections::VecDeque;

/// Configuration for the dynamic z-score filter
#[derive(Debug, Clone)]
pub struct ZScoreFilterConfig {
    /// Short-term EMA span for rapid changes
    pub short_span: usize,
    /// Medium-term EMA span
    pub medium_span: usize,
    /// Long-term EMA span for baseline
    pub long_span: usize,
    /// Base threshold for outlier detection (standard deviations)
    pub base_threshold: f64,
    /// Maximum allowed threshold during extreme volatility
    pub max_threshold: f64,
    /// Minimum samples before activation
    pub min_samples: usize,
    /// Volatility sensitivity factor
    pub volatility_sensitivity: f64,
}

impl Default for ZScoreFilterConfig {
    fn default() -> Self {
        Self {
            short_span: 10,
            medium_span: 50,
            long_span: 200,
            base_threshold: 3.0,
            max_threshold: 6.0,
            min_samples: 20,
            volatility_sensitivity: 0.5,
        }
    }
}

/// Result of z-score analysis
#[derive(Debug, Clone)]
pub struct ZScoreResult {
    /// Current z-score value
    pub z_score: f64,
    /// Adaptive threshold at this moment
    pub adaptive_threshold: f64,
    /// Whether this is an outlier
    pub is_outlier: bool,
    /// Volatility regime classification
    pub regime: VolatilityRegime,
    /// Which timescale detected the anomaly
    pub detection_scale: DetectionScale,
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    /// Normal market conditions
    Calm,
    /// Elevated but manageable volatility
    Elevated,
    /// Extreme volatility (flash crash, news events)
    Extreme,
}

/// Timescale of detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionScale {
    Short,
    Medium,
    Long,
    None,
}

/// Exponentially Weighted Moving Average tracker
struct EWMATracker {
    span: usize,
    /// Smoothing factor α = 2/(span+1)
    alpha: f64,
    /// Current mean estimate
    mean: f64,
    /// Current variance estimate (EWMA of squared deviations)
    variance: f64,
    /// Sample count
    n: usize,
    /// Whether initialized
    initialized: bool,
}

impl EWMATracker {
    fn new(span: usize) -> Self {
        let alpha = 2.0 / (span + 1) as f64;
        
        Self {
            span,
            alpha,
            mean: 0.0,
            variance: 0.0,
            n: 0,
            initialized: false,
        }
    }

    /// Update with new observation
    #[inline]
    fn update(&mut self, value: f64) {
        self.n += 1;
        
        if !self.initialized {
            // Initialize with first value
            self.mean = value;
            self.variance = 0.0;
            self.initialized = true;
        } else {
            // EWMA update for mean
            let diff = value - self.mean;
            self.mean += self.alpha * diff;
            
            // EWMA update for variance (Welford-style for EWMA)
            let diff_after = value - self.mean;
            self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff_after);
        }
    }

    /// Get current standard deviation
    #[inline]
    fn std(&self) -> f64 {
        if self.variance < 0.0 {
            return 0.0;
        }
        self.variance.sqrt()
    }

    /// Compute z-score for a value
    #[inline]
    fn z_score(&self, value: f64) -> f64 {
        let std = self.std();
        if std < 1e-10 {
            return 0.0;
        }
        (value - self.mean) / std
    }

    /// Check if tracker has enough samples
    #[inline]
    fn is_ready(&self) -> bool {
        self.n >= self.span / 2
    }
}

/// Dynamic z-score filter with adaptive thresholds
pub struct ZScoreFilter {
    config: ZScoreFilterConfig,
    /// Short-term tracker for rapid detection
    short_tracker: EWMATracker,
    /// Medium-term tracker
    medium_tracker: EWMATracker,
    /// Long-term tracker for baseline
    long_tracker: EWMATracker,
    /// Recent volatility measurements
    recent_volatility: VecDeque<f64>,
    /// Current adaptive threshold multiplier
    threshold_multiplier: f64,
    /// Last processed value
    last_value: Option<f64>,
}

impl ZScoreFilter {
    /// Create new z-score filter with configuration
    pub fn new(config: ZScoreFilterConfig) -> Self {
        Self {
            config: config.clone(),
            short_tracker: EWMATracker::new(config.short_span),
            medium_tracker: EWMATracker::new(config.medium_span),
            long_tracker: EWMATracker::new(config.long_span),
            recent_volatility: VecDeque::with_capacity(20),
            threshold_multiplier: 1.0,
            last_value: None,
        }
    }

    /// Process new observation and compute adaptive z-score
    pub fn update(&mut self, value: f64) -> Option<ZScoreResult> {
        // Calculate return if we have previous value
        if let Some(last) = self.last_value {
            let return_val = (value - last) / last.max(1e-10);
            
            // Update all trackers
            self.short_tracker.update(return_val);
            self.medium_tracker.update(return_val);
            self.long_tracker.update(return_val);
            
            // Track recent volatility
            let abs_return = return_val.abs();
            self.recent_volatility.push_back(abs_return);
            if self.recent_volatility.len() > 20 {
                self.recent_volatility.pop_front();
            }
        }
        
        self.last_value = Some(value);
        
        // Skip if not enough samples
        if !self.short_tracker.is_ready() || !self.medium_tracker.is_ready() {
            return None;
        }
        
        // Determine volatility regime
        let regime = self._classify_regime();
        
        // Compute adaptive threshold
        let adaptive_threshold = self._compute_adaptive_threshold(regime);
        
        // Compute z-scores at different scales
        let z_short = self.short_tracker.z_score(
            (value - self.last_value.unwrap_or(value)) / self.last_value.unwrap_or(value).max(1e-10)
        );
        let z_medium = self.medium_tracker.z_score(
            (value - self.last_value.unwrap_or(value)) / self.last_value.unwrap_or(value).max(1e-10)
        );
        let z_long = self.long_tracker.z_score(
            (value - self.last_value.unwrap_or(value)) / self.last_value.unwrap_or(value).max(1e-10)
        );
        
        // Determine if outlier and at which scale
        let (is_outlier, detection_scale) = self._check_outlier(z_short, z_medium, z_long, adaptive_threshold);
        
        Some(ZScoreResult {
            z_score: z_medium, // Use medium-term as primary
            adaptive_threshold,
            is_outlier,
            regime,
            detection_scale,
        })
    }

    /// Classify current volatility regime
    fn _classify_regime(&self) -> VolatilityRegime {
        if self.recent_volatility.len() < 5 {
            return VolatilityRegime::Calm;
        }
        
        let recent_avg: f64 = self.recent_volatility.iter().sum::<f64>() / self.recent_volatility.len() as f64;
        let baseline_std = self.long_tracker.std();
        
        if baseline_std < 1e-10 {
            return VolatilityRegime::Calm;
        }
        
        let vol_ratio = recent_avg / baseline_std;
        
        if vol_ratio > 3.0 {
            VolatilityRegime::Extreme
        } else if vol_ratio > 1.5 {
            VolatilityRegime::Elevated
        } else {
            VolatilityRegime::Calm
        }
    }

    /// Compute adaptive threshold based on regime
    fn _compute_adaptive_threshold(&mut self, regime: VolatilityRegime) -> f64 {
        let base = self.config.base_threshold;
        
        // Adjust threshold based on regime
        let regime_multiplier = match regime {
            VolatilityRegime::Calm => 1.0,
            VolatilityRegime::Elevated => 1.3,
            VolatilityRegime::Extreme => 1.8,
        };
        
        // Also adjust based on recent volatility trend
        let vol_trend = self._compute_volatility_trend();
        let trend_adjustment = 1.0 + vol_trend * self.config.volatility_sensitivity;
        
        self.threshold_multiplier = regime_multiplier * trend_adjustment.min(2.0);
        
        let adaptive = base * self.threshold_multiplier;
        adaptive.min(self.config.max_threshold)
    }

    /// Compute volatility trend (positive = increasing)
    fn _compute_volatility_trend(&self) -> f64 {
        if self.recent_volatility.len() < 10 {
            return 0.0;
        }
        
        let first_half: f64 = self.recent_volatility.iter().take(5).sum::<f64>() / 5.0;
        let second_half: f64 = self.recent_volatility.iter().rev().take(5).sum::<f64>() / 5.0;
        
        if first_half < 1e-10 {
            return 0.0;
        }
        
        (second_half - first_half) / first_half
    }

    /// Check if value is outlier at any scale
    fn _check_outlier(&self, z_short: f64, z_medium: f64, z_long: f64, 
                      threshold: f64) -> (bool, DetectionScale) {
        // Check short-term first (most sensitive)
        if z_short.abs() > threshold * 0.8 {
            return (true, DetectionScale::Short);
        }
        
        // Check medium-term
        if z_medium.abs() > threshold {
            return (true, DetectionScale::Medium);
        }
        
        // Check long-term (most conservative)
        if z_long.abs() > threshold * 1.2 {
            return (true, DetectionScale::Long);
        }
        
        (false, DetectionScale::None)
    }

    /// Get current volatility regime
    pub fn current_regime(&self) -> VolatilityRegime {
        self._classify_regime()
    }

    /// Get current adaptive threshold
    pub fn current_threshold(&self) -> f64 {
        self.config.base_threshold * self.threshold_multiplier
    }

    /// Get z-score without updating (for peeking)
    pub fn peek_z_score(&self, value: f64) -> Option<f64> {
        if !self.medium_tracker.is_ready() {
            return None;
        }
        
        let last = self.last_value?;
        let ret = (value - last) / last.max(1e-10);
        
        Some(self.medium_tracker.z_score(ret))
    }

    /// Reset all trackers
    pub fn reset(&mut self) {
        self.short_tracker = EWMATracker::new(self.config.short_span);
        self.medium_tracker = EWMATracker::new(self.config.medium_span);
        self.long_tracker = EWMATracker::new(self.config.long_span);
        self.recent_volatility.clear();
        self.threshold_multiplier = 1.0;
        self.last_value = None;
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.medium_tracker.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_normal_data() {
        let config = ZScoreFilterConfig::default();
        let mut filter = ZScoreFilter::new(config);
        
        // Feed normal data
        let mut base_price = 100.0;
        for i in 0..100 {
            base_price *= 1.0 + (i as f64 * 0.001).sin() * 0.01;
            let result = filter.update(base_price);
            
            if i > 30 {
                assert!(result.is_some());
                let r = result.unwrap();
                // Normal data should not trigger outliers often
                if r.is_outlier {
                    assert!(r.regime != VolatilityRegime::Calm);
                }
            }
        }
    }

    #[test]
    fn test_zscore_spike_detection() {
        let config = ZScoreFilterConfig::default();
        let mut filter = ZScoreFilter::new(config);
        
        // Establish baseline
        let mut price = 100.0;
        for _ in 0..50 {
            price *= 1.0 + rand::random::<f64>() * 0.01 - 0.005;
            filter.update(price);
        }
        
        // Inject sharp spike
        price *= 1.15; // 15% jump
        let result = filter.update(price);
        
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.is_outlier, "Should detect large spike");
        assert!(r.z_score.abs() > 3.0);
    }

    #[test]
    fn test_adaptive_threshold_during_volatility() {
        let config = ZScoreFilterConfig::default();
        let mut filter = ZScoreFilter::new(config);
        
        // High volatility period
        let mut price = 100.0;
        for i in 0..50 {
            let change = (i as f64 * 0.5).sin() * 0.05; // 5% swings
            price *= 1.0 + change;
            filter.update(price);
        }
        
        // Threshold should be elevated
        let threshold = filter.current_threshold();
        assert!(threshold > config.base_threshold, 
                "Threshold should increase during volatile period");
    }

    #[test]
    fn test_regime_classification() {
        let config = ZScoreFilterConfig::default();
        let mut filter = ZScoreFilter::new(config);
        
        // Calm period
        let mut price = 100.0;
        for _ in 0..30 {
            price *= 1.0 + rand::random::<f64>() * 0.002 - 0.001;
            filter.update(price);
        }
        
        assert_eq!(filter.current_regime(), VolatilityRegime::Calm);
        
        // Simulate volatility spike
        for _ in 0..20 {
            price *= 1.0 + rand::random::<f64>() * 0.04 - 0.02;
            filter.update(price);
        }
        
        let regime = filter.current_regime();
        assert!(regime == VolatilityRegime::Elevated || regime == VolatilityRegime::Extreme);
    }
}
