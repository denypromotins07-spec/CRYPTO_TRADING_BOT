//! Drawdown-Adaptive Position Scaler
//!
//! Dynamically reduces position sizes as equity curves decline.
//! Implements defensive scaling to prevent catastrophic losses.
//! Critical for surviving crypto market drawdowns while preserving capital.
//!
//! # Key Features:
//! - Progressive size reduction based on drawdown depth
//! - Faster recovery path with reduced risk exposure
//! - Configurable sensitivity and thresholds
//! - Integration with Kelly criterion for optimal sizing

use std::f64;

/// Fixed portfolio size
const N_ASSETS: usize = 4;

/// Asset labels
pub const ASSET_LABELS: [&str; N_ASSETS] = ["BTC", "SOL", "ETH", "USDT"];

/// Drawdown severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawdownLevel {
    /// No significant drawdown (< 5%)
    Normal,
    /// Mild drawdown (5-10%)
    Mild,
    /// Moderate drawdown (10-20%)
    Moderate,
    /// Severe drawdown (20-35%)
    Severe,
    /// Crisis mode (> 35%)
    Crisis,
}

impl DrawdownLevel {
    /// Get the recommended position scale factor for this level
    pub fn scale_factor(&self) -> f64 {
        match self {
            DrawdownLevel::Normal => 1.0,
            DrawdownLevel::Mild => 0.75,
            DrawdownLevel::Moderate => 0.50,
            DrawdownLevel::Severe => 0.25,
            DrawdownLevel::Crisis => 0.10,
        }
    }

    /// Get minimum drawdown percentage for this level
    pub fn min_drawdown(&self) -> f64 {
        match self {
            DrawdownLevel::Normal => 0.0,
            DrawdownLevel::Mild => 0.05,
            DrawdownLevel::Moderate => 0.10,
            DrawdownLevel::Severe => 0.20,
            DrawdownLevel::Crisis => 0.35,
        }
    }
}

/// Result of drawdown-based position scaling
#[derive(Debug, Clone)]
pub struct DrawdownScalingResult {
    /// Current drawdown percentage
    pub current_drawdown: f64,
    /// Drawdown severity level
    pub drawdown_level: DrawdownLevel,
    /// Scale factor to apply to positions
    pub scale_factor: f64,
    /// Original position sizes (before scaling)
    pub original_sizes: [f64; N_ASSETS],
    /// Scaled position sizes (after applying drawdown reduction)
    pub scaled_sizes: [f64; N_ASSETS],
    /// Maximum allowed position after scaling
    pub max_position: [f64; N_ASSETS],
    /// Whether trading should be halted entirely
    pub halt_trading: bool,
    /// Recommended action
    pub recommendation: ScalingRecommendation,
}

/// Trading recommendation based on drawdown analysis
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingRecommendation {
    /// Continue normal trading
    ContinueNormal,
    /// Reduce position sizes
    ReduceSizes,
    /// Only close positions, no new entries
    CloseOnly,
    /// Halt all trading immediately
    HaltAll,
}

/// Drawdown-Adaptive Position Scaler
///
/// Monitors portfolio drawdown and automatically scales position sizes
/// to reduce risk during losing streaks. Helps prevent ruin and enables
/// gradual recovery.
pub struct DrawdownScaler {
    /// Peak portfolio value (high water mark)
    peak_value: f64,
    /// Current portfolio value
    current_value: f64,
    /// Drawdown thresholds for each level
    thresholds: DrawdownThresholds,
    /// Hard stop loss (halt trading if exceeded)
    hard_stop_loss: f64,
    /// Recovery threshold (return to normal trading)
    recovery_threshold: f64,
    /// Whether currently in recovery mode
    in_recovery: bool,
}

/// Configurable drawdown thresholds
#[derive(Debug, Clone)]
pub struct DrawdownThresholds {
    /// Mild drawdown threshold (default 5%)
    pub mild_threshold: f64,
    /// Moderate drawdown threshold (default 10%)
    pub moderate_threshold: f64,
    /// Severe drawdown threshold (default 20%)
    pub severe_threshold: f64,
    /// Crisis threshold (default 35%)
    pub crisis_threshold: f64,
    /// Hard stop loss (default 50%)
    pub hard_stop: f64,
}

impl Default for DrawdownThresholds {
    fn default() -> Self {
        Self {
            mild_threshold: 0.05,
            moderate_threshold: 0.10,
            severe_threshold: 0.20,
            crisis_threshold: 0.35,
            hard_stop: 0.50,
        }
    }
}

impl DrawdownScaler {
    /// Create new scaler with default thresholds
    pub fn new(initial_value: f64) -> Self {
        Self {
            peak_value: initial_value,
            current_value: initial_value,
            thresholds: DrawdownThresholds::default(),
            hard_stop_loss: 0.50,
            recovery_threshold: 0.02, // 2% from peak to exit recovery
            in_recovery: false,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(initial_value: f64, thresholds: DrawdownThresholds) -> Self {
        Self {
            peak_value: initial_value,
            current_value: initial_value,
            thresholds,
            hard_stop_loss: thresholds.hard_stop,
            in_recovery: false,
        }
    }

    /// Update current portfolio value
    pub fn update_value(&mut self, new_value: f64) {
        if new_value <= 0.0 {
            return; // Invalid value
        }
        
        self.current_value = new_value;
        
        // Update peak if we've exceeded it
        if new_value > self.peak_value {
            self.peak_value = new_value;
            
            // Check if we've recovered enough to exit recovery mode
            let drawdown = self.compute_drawdown();
            if drawdown < self.recovery_threshold {
                self.in_recovery = false;
            }
        }
    }

    /// Set hard stop loss level
    pub fn set_hard_stop(&mut self, stop_loss: f64) {
        self.hard_stop_loss = stop_loss.clamp(0.1, 0.9);
    }

    /// Compute current drawdown percentage
    pub fn compute_drawdown(&self) -> f64 {
        if self.peak_value <= 0.0 {
            return 0.0;
        }
        (self.peak_value - self.current_value) / self.peak_value
    }

    /// Determine current drawdown level
    pub fn get_drawdown_level(&self) -> DrawdownLevel {
        let dd = self.compute_drawdown();
        
        if dd >= self.thresholds.crisis_threshold {
            DrawdownLevel::Crisis
        } else if dd >= self.thresholds.severe_threshold {
            DrawdownLevel::Severe
        } else if dd >= self.thresholds.moderate_threshold {
            DrawdownLevel::Moderate
        } else if dd >= self.thresholds.mild_threshold {
            DrawdownLevel::Mild
        } else {
            DrawdownLevel::Normal
        }
    }

    /// Compute scaled position sizes based on current drawdown
    ///
    /// # Arguments
    /// * `original_sizes` - Desired position sizes before drawdown adjustment
    /// * `max_positions` - Maximum allowed position per asset
    ///
    /// # Returns
    /// DrawdownScalingResult with scaled positions and recommendations
    pub fn compute_scaled_positions(
        &self,
        original_sizes: [f64; N_ASSETS],
        max_positions: [f64; N_ASSETS],
    ) -> DrawdownScalingResult {
        let drawdown = self.compute_drawdown();
        let level = self.get_drawdown_level();
        let base_scale = level.scale_factor();
        
        // Additional scaling if in recovery mode (be more conservative)
        let recovery_scale = if self.in_recovery { 0.8 } else { 1.0 };
        let scale_factor = base_scale * recovery_scale;
        
        let mut scaled_sizes = [0.0; N_ASSETS];
        let mut max_scaled = [0.0; N_ASSETS];
        
        for i in 0..N_ASSETS {
            // Apply drawdown scaling
            let scaled = original_sizes[i] * scale_factor;
            
            // Apply maximum position limit (also scaled by drawdown)
            let max_allowed = max_positions[i] * scale_factor;
            max_scaled[i] = max_allowed;
            
            // Take minimum
            scaled_sizes[i] = scaled.min(max_allowed);
        }
        
        // Determine if trading should halt
        let halt_trading = drawdown >= self.hard_stop_loss;
        
        // Get recommendation
        let recommendation = self._get_recommendation(level, halt_trading);
        
        DrawdownScalingResult {
            current_drawdown: drawdown,
            drawdown_level: level,
            scale_factor,
            original_sizes,
            scaled_sizes,
            max_position: max_scaled,
            halt_trading,
            recommendation,
        }
    }

    /// Get recommendation based on drawdown state
    fn _get_recommendation(&self, level: DrawdownLevel, halt: bool) -> ScalingRecommendation {
        if halt || level == DrawdownLevel::Crisis {
            ScalingRecommendation::HaltAll
        } else if level == DrawdownLevel::Severe {
            ScalingRecommendation::CloseOnly
        } else if level == DrawdownLevel::Moderate {
            ScalingRecommendation::ReduceSizes
        } else {
            ScalingRecommendation::ContinueNormal
        }
    }

    /// Instantly scale down positions during emergency
    ///
    /// Used when rapid market deterioration is detected.
    pub fn emergency_decrease(&mut self, severity: f64) {
        // Artificially increase perceived drawdown
        let artificial_dd = self.compute_drawdown() + severity * 0.2;
        
        // If severe enough, enter recovery mode immediately
        if severity > 0.3 {
            self.in_recovery = true;
        }
        
        // Note: We don't actually change peak/current values here,
        // just the behavior through recovery mode flag
    }

    /// Get the maximum allowed new position size
    pub fn max_new_position(&self, base_max: f64) -> f64 {
        let level = self.get_drawdown_level();
        base_max * level.scale_factor()
    }

    /// Check if new positions are allowed
    pub fn can_open_positions(&self) -> bool {
        let level = self.get_drawdown_level();
        !self.in_recovery && 
        level != DrawdownLevel::Crisis && 
        self.compute_drawdown() < self.hard_stop_loss
    }

    /// Get recovery progress (0 = at bottom, 1 = fully recovered)
    pub fn recovery_progress(&self) -> f64 {
        let current_dd = self.compute_drawdown();
        
        // Find the starting drawdown (when recovery began)
        // For simplicity, use the max drawdown level
        let max_dd = self.thresholds.crisis_threshold;
        
        if current_dd >= max_dd {
            0.0
        } else if current_dd < self.recovery_threshold {
            1.0
        } else {
            1.0 - (current_dd - self.recovery_threshold) / (max_dd - self.recovery_threshold)
        }
    }

    /// Get historical peak value
    pub fn peak_value(&self) -> f64 {
        self.peak_value
    }

    /// Get current value
    pub fn current_value(&self) -> f64 {
        self.current_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drawdown_computation() {
        let mut scaler = DrawdownScaler::new(100_000.0);
        
        // Initial state: no drawdown
        assert_eq!(scaler.compute_drawdown(), 0.0);
        assert_eq!(scaler.get_drawdown_level(), DrawdownLevel::Normal);
        
        // Simulate 15% drawdown
        scaler.update_value(85_000.0);
        let dd = scaler.compute_drawdown();
        assert!((dd - 0.15).abs() < 0.001);
        assert_eq!(scaler.get_drawdown_level(), DrawdownLevel::Moderate);
    }

    #[test]
    fn test_position_scaling() {
        let mut scaler = DrawdownScaler::new(100_000.0);
        
        let original = [10_000.0, 8_000.0, 9_000.0, 5_000.0];
        let max_pos = [30_000.0, 25_000.0, 30_000.0, 10_000.0];
        
        // Normal market: full size
        let result = scaler.compute_scaled_positions(original, max_pos);
        assert_eq!(result.drawdown_level, DrawdownLevel::Normal);
        assert!((result.scale_factor - 1.0).abs() < 0.01);
        
        // Simulate 25% drawdown (severe)
        scaler.update_value(75_000.0);
        let result = scaler.compute_scaled_positions(original, max_pos);
        assert_eq!(result.drawdown_level, DrawdownLevel::Severe);
        assert!((result.scale_factor - 0.25).abs() < 0.01);
        
        // Positions should be scaled down
        for i in 0..N_ASSETS {
            assert!(result.scaled_sizes[i] < original[i]);
        }
    }

    #[test]
    fn test_peak_update_and_recovery() {
        let mut scaler = DrawdownScaler::new(100_000.0);
        
        // Drop to 80k
        scaler.update_value(80_000.0);
        assert_eq!(scaler.peak_value(), 100_000.0);
        assert!(scaler.compute_drawdown() > 0.15);
        
        // Recover to 95k (peak stays at 100k)
        scaler.update_value(95_000.0);
        assert_eq!(scaler.peak_value(), 100_000.0);
        assert!((scaler.compute_drawdown() - 0.05).abs() < 0.001);
        
        // Exceed peak
        scaler.update_value(105_000.0);
        assert_eq!(scaler.peak_value(), 105_000.0);
        assert_eq!(scaler.compute_drawdown(), 0.0);
    }

    #[test]
    fn test_hard_stop() {
        let mut scaler = DrawdownScaler::new(100_000.0);
        scaler.set_hard_stop(0.40); // 40% hard stop
        
        let original = [10_000.0, 8_000.0, 9_000.0, 5_000.0];
        let max_pos = [30_000.0, 25_000.0, 30_000.0, 10_000.0];
        
        // 45% drawdown exceeds hard stop
        scaler.update_value(55_000.0);
        let result = scaler.compute_scaled_positions(original, max_pos);
        
        assert!(result.halt_trading);
        assert_eq!(result.recommendation, ScalingRecommendation::HaltAll);
    }

    #[test]
    fn test_emergency_decrease() {
        let mut scaler = DrawdownScaler::new(100_000.0);
        
        assert!(!scaler.in_recovery);
        assert!(scaler.can_open_positions());
        
        // Trigger emergency
        scaler.emergency_decrease(0.5);
        
        assert!(scaler.in_recovery);
        assert!(!scaler.can_open_positions());
    }

    #[test]
    fn test_drawdown_levels() {
        let levels = [
            (0.03, DrawdownLevel::Normal),
            (0.07, DrawdownLevel::Mild),
            (0.15, DrawdownLevel::Moderate),
            (0.25, DrawdownLevel::Severe),
            (0.40, DrawdownLevel::Crisis),
        ];

        for (dd, expected_level) in levels {
            let mut scaler = DrawdownScaler::new(100_000.0);
            scaler.update_value(100_000.0 * (1.0 - dd));
            assert_eq!(scaler.get_drawdown_level(), expected_level);
        }
    }
}
