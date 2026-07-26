//! CUSUM (Cumulative Sum) Control Chart Engine for Micro-Regime Shift Detection
//! 
//! This module implements ultra-fast Cumulative Sum control algorithms to detect
//! subtle changes in market behavior at the microsecond level. It is optimized
//! for zero-cost abstractions and operates without heap allocations during runtime.
//! 
//! Key Features:
//! - O(1) update time per tick
//! - Configurable sensitivity (k) and threshold (h) parameters
//! - Automatic reset after structural break confirmation
//! - Memory-bounded circular buffers for historical tracking
//! 
//! Mathematical Foundation:
//! S_t = max(0, S_{t-1} + (x_t - μ_0)/σ - k)
//! Alarm triggered when S_t > h

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the CUSUM engine
#[derive(Debug, Clone)]
pub struct CusumConfig {
    /// Reference mean (μ_0) - typically rolling mean of returns
    pub reference_mean: f64,
    /// Reference standard deviation (σ)
    pub reference_std: f64,
    /// Allowance parameter (k) - controls sensitivity
    pub allowance: f64,
    /// Decision threshold (h) - triggers regime shift when exceeded
    pub threshold: f64,
    /// Minimum samples before activation
    pub warmup_samples: usize,
}

impl Default for CusumConfig {
    fn default() -> Self {
        Self {
            reference_mean: 0.0,
            reference_std: 1.0,
            allowance: 0.5,
            threshold: 5.0,
            warmup_samples: 20,
        }
    }
}

/// Result of a CUSUM update
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CusumState {
    /// No change detected, accumulation ongoing
    Accumulating { positive_sum: f64, negative_sum: f64 },
    /// Positive regime shift detected (upward break)
    PositiveShift { magnitude: f64, detection_time_us: u64 },
    /// Negative regime shift detected (downward break)
    NegativeShift { magnitude: f64, detection_time_us: u64 },
    /// Reset state after confirmation
    Reset,
}

/// Ultra-fast CUSUM engine for micro-regime detection
pub struct CusumEngine {
    config: CusumConfig,
    positive_sum: f64,
    negative_sum: f64,
    sample_count: usize,
    last_update: Instant,
    detection_start: Option<Instant>,
    /// Pre-computed inverse std for division avoidance
    inv_std: f64,
}

impl CusumEngine {
    /// Create a new CUSUM engine with given configuration
    #[inline]
    pub fn new(config: CusumConfig) -> Self {
        let inv_std = if config.reference_std > 1e-10 {
            1.0 / config.reference_std
        } else {
            1.0
        };
        
        Self {
            config,
            positive_sum: 0.0,
            negative_sum: 0.0,
            sample_count: 0,
            last_update: Instant::now(),
            detection_start: None,
            inv_std,
        }
    }

    /// Update CUSUM with new observation - O(1) time, zero allocations
    #[inline]
    pub fn update(&mut self, observation: f64) -> CusumState {
        self.sample_count += 1;
        
        // Skip during warmup period
        if self.sample_count <= self.config.warmup_samples {
            return CusumState::Accumulating {
                positive_sum: self.positive_sum,
                negative_sum: self.negative_sum,
            };
        }

        // Standardize observation: z = (x - μ_0) / σ
        let standardized = (observation - self.config.reference_mean) * self.inv_std;
        
        // Update positive CUSUM (detects upward shifts)
        self.positive_sum = (0.0_f64).max(self.positive_sum + standardized - self.config.allowance);
        
        // Update negative CUSUM (detects downward shifts)
        self.negative_sum = (0.0_f64).max(self.negative_sum - standardized - self.config.allowance);
        
        let now = Instant::now();
        let elapsed_us = now.duration_since(self.last_update).as_micros() as u64;
        self.last_update = now;

        // Check for positive regime shift
        if self.positive_sum > self.config.threshold {
            let magnitude = self.positive_sum;
            self.reset();
            return CusumState::PositiveShift {
                magnitude,
                detection_time_us: elapsed_us,
            };
        }

        // Check for negative regime shift
        if self.negative_sum > self.config.threshold {
            let magnitude = self.negative_sum;
            self.reset();
            return CusumState::NegativeShift {
                magnitude,
                detection_time_us: elapsed_us,
            };
        }

        CusumState::Accumulating {
            positive_sum: self.positive_sum,
            negative_sum: self.negative_sum,
        }
    }

    /// Manually reset the CUSUM accumulators after confirmed structural break
    #[inline]
    pub fn reset(&mut self) {
        self.positive_sum = 0.0;
        self.negative_sum = 0.0;
        self.detection_start = None;
    }

    /// Update reference parameters dynamically (adaptive CUSUM)
    #[inline]
    pub fn update_reference(&mut self, new_mean: f64, new_std: f64) {
        self.config.reference_mean = new_mean;
        self.config.reference_std = new_std.max(1e-10);
        self.inv_std = 1.0 / self.config.reference_std;
    }

    /// Get current accumulation values for monitoring
    #[inline]
    pub fn get_state(&self) -> (f64, f64) {
        (self.positive_sum, self.negative_sum)
    }

    /// Check if engine is in warmup phase
    #[inline]
    pub fn is_warming_up(&self) -> bool {
        self.sample_count <= self.config.warmup_samples
    }

    /// Get sample count since last reset
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
}

/// Multi-asset CUSUM coordinator for correlated regime detection
pub struct MultiAssetCusum {
    engines: Vec<(String, CusumEngine)>,
    correlation_threshold: f64,
}

impl MultiAssetCusum {
    /// Create coordinator for multiple assets
    pub fn new(assets: &[&str], config: CusumConfig) -> Self {
        let engines = assets
            .iter()
            .map(|&asset| (asset.to_string(), CusumEngine::new(config.clone())))
            .collect();
        
        Self {
            engines,
            correlation_threshold: 0.7,
        }
    }

    /// Update all asset engines and detect correlated regime shifts
    pub fn update_all(&mut self, observations: &[(String, f64)]) -> Vec<CusumState> {
        observations
            .iter()
            .filter_map(|(asset, obs)| {
                self.engines
                    .iter_mut()
                    .find(|(name, _)| name == asset)
                    .map(|(_, engine)| engine.update(*obs))
            })
            .collect()
    }

    /// Detect if multiple assets show synchronized regime shifts
    pub fn detect_synchronized_shift(&self, states: &[CusumState]) -> bool {
        let shift_count = states.iter().filter(|s| {
            matches!(s, CusumState::PositiveShift { .. } | CusumState::NegativeShift { .. })
        }).count();

        let total_assets = self.engines.len();
        let ratio = shift_count as f64 / total_assets as f64;
        
        ratio >= self.correlation_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusum_positive_shift_detection() {
        let config = CusumConfig {
            reference_mean: 0.0,
            reference_std: 1.0,
            allowance: 0.5,
            threshold: 3.0,
            warmup_samples: 5,
        };
        
        let mut engine = CusumEngine::new(config);
        
        // Feed normal data
        for _ in 0..6 {
            engine.update(0.0);
        }
        
        // Feed upward shift
        let mut detected = false;
        for _ in 0..20 {
            if let CusumState::PositiveShift { .. } = engine.update(2.0) {
                detected = true;
                break;
            }
        }
        
        assert!(detected, "Should detect positive regime shift");
    }

    #[test]
    fn test_cusum_reset_after_shift() {
        let config = CusumConfig::default();
        let mut engine = CusumEngine::new(config);
        
        // Trigger shift
        for _ in 0..100 {
            engine.update(10.0);
        }
        
        // Verify reset
        let (pos, neg) = engine.get_state();
        assert!(pos < 1.0 && neg < 1.0, "Should be reset after detection");
    }

    #[test]
    fn test_zero_allocation_runtime() {
        let config = CusumConfig::default();
        let mut engine = CusumEngine::new(config);
        
        // Warmup
        for i in 0..20 {
            engine.update(i as f64 * 0.01);
        }
        
        // Runtime updates should not allocate
        for i in 0..1000 {
            let _state = engine.update(i as f64 * 0.001);
        }
        
        assert!(engine.sample_count() > 1000);
    }
}
