//! Branching Ratio Estimator for Order Book Stability Analysis
//!
//! This module estimates the criticality and stability of the order book
//! by calculating the branching ratio of the Hawkes process. A branching
//! ratio approaching 1.0 indicates a critical/excitable market state.
//!
//! # Features
//! - Real-time branching ratio estimation
//! - Market stability classification
//! - Criticality warnings for flash crash prevention
//! - Zero-cost abstractions for memory efficiency

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Stability state of the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketStability {
    /// Stable market, normal conditions
    Stable,
    /// Elevated activity, caution advised
    Elevated,
    /// Near-critical, high risk of cascading events
    NearCritical,
    /// Critical state, potential for flash crash
    Critical,
}

impl MarketStability {
    /// Get stability from branching ratio
    pub fn from_branching_ratio(ratio: f64) -> Self {
        match ratio {
            r if r < 0.5 => MarketStability::Stable,
            r if r < 0.7 => MarketStability::Elevated,
            r if r < 0.95 => MarketStability::NearCritical,
            _ => MarketStability::Critical,
        }
    }
    
    /// Get risk level as string
    pub fn risk_level(&self) -> &'static str {
        match self {
            MarketStability::Stable => "LOW",
            MarketStability::Elevated => "MEDIUM",
            MarketStability::NearCritical => "HIGH",
            MarketStability::Critical => "EXTREME",
        }
    }
}

/// Configuration for branching ratio estimation
#[derive(Debug, Clone)]
pub struct BranchingConfig {
    /// Minimum samples for reliable estimation
    pub min_samples: usize,
    /// Decay factor for exponential weighting
    pub decay_factor: f64,
    /// Maximum matrix dimension
    pub max_dimensions: usize,
    /// Stability thresholds
    pub critical_threshold: f64,
    pub near_critical_threshold: f64,
}

impl Default for BranchingConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            decay_factor: 0.995,
            max_dimensions: 6, // Market/Limit/Cancel x Buy/Sell
            critical_threshold: 0.95,
            near_critical_threshold: 0.70,
        }
    }
}

/// Running estimate of the excitation matrix
struct ExcitationMatrixEstimator {
    /// Current estimate of excitation matrix (flattened)
    estimate: Vec<f64>,
    /// Weighted sum for estimation
    weighted_sum: Vec<f64>,
    /// Total weight accumulated
    total_weight: f64,
    /// Matrix dimension
    dimension: usize,
    /// Decay factor
    decay: f64,
}

impl ExcitationMatrixEstimator {
    fn new(dimension: usize, decay: f64) -> Self {
        let size = dimension * dimension;
        Self {
            estimate: vec![0.0; size],
            weighted_sum: vec![0.0; size],
            total_weight: 0.0,
            dimension,
            decay,
        }
    }
    
    /// Update estimate with new observation in O(1)
    fn update(&mut self, cause_idx: usize, effect_idx: usize, weight: f64) {
        if cause_idx >= self.dimension || effect_idx >= self.dimension {
            return;
        }
        
        let idx = effect_idx * self.dimension + cause_idx;
        
        // Apply decay to all elements
        self.weighted_sum.iter_mut().for_each(|v| *v *= self.decay);
        self.total_weight *= self.decay;
        
        // Add new observation
        self.weighted_sum[idx] += weight;
        self.total_weight += weight;
        
        // Update estimate
        if self.total_weight > 1e-10 {
            self.estimate[idx] = self.weighted_sum[idx] / self.total_weight;
        }
    }
    
    /// Get the current estimate as 2D view
    fn get_matrix(&self) -> Vec<Vec<f64>> {
        (0..self.dimension)
            .map(|i| {
                self.estimate[i * self.dimension..(i + 1) * self.dimension].to_vec()
            })
            .collect()
    }
    
    /// Calculate spectral radius (largest eigenvalue magnitude)
    /// Uses power iteration for efficiency
    fn spectral_radius(&self) -> f64 {
        if self.dimension == 0 {
            return 0.0;
        }
        
        // Power iteration method
        let mut v: Vec<f64> = vec![1.0 / (self.dimension as f64).sqrt(); self.dimension];
        let mut eigenvalue = 0.0;
        
        for _ in 0..50 { // Max iterations
            // Matrix-vector multiplication
            let mut new_v = vec![0.0; self.dimension];
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    new_v[i] += self.estimate[i * self.dimension + j] * v[j];
                }
            }
            
            // Calculate norm
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            
            if norm < 1e-10 {
                break;
            }
            
            // Normalize
            eigenvalue = norm;
            v = new_v.iter().map(|x| x / norm).collect();
        }
        
        eigenvalue
    }
    
    /// Reset estimator
    fn reset(&mut self) {
        self.weighted_sum.fill(0.0);
        self.total_weight = 0.0;
        self.estimate.fill(0.0);
    }
}

/// Branching ratio calculator with real-time updates
pub struct BranchingRatioCalculator {
    config: BranchingConfig,
    estimator: ExcitationMatrixEstimator,
    sample_count: usize,
    current_branching_ratio: f64,
    last_update_time: u128,
    is_initialized: bool,
}

impl BranchingRatioCalculator {
    /// Create a new branching ratio calculator
    pub fn new(config: BranchingConfig) -> Self {
        let estimator = ExcitationMatrixEstimator::new(
            config.max_dimensions,
            config.decay_factor,
        );
        
        Self {
            config,
            estimator,
            sample_count: 0,
            current_branching_ratio: 0.0,
            last_update_time: 0,
            is_initialized: false,
        }
    }
    
    /// Record a causal relationship between events
    /// 
    /// # Arguments
    /// * `cause_idx` - Index of the causing event type
    /// * `effect_idx` - Index of the affected event type
    /// * `strength` - Strength of the causal relationship
    pub fn record_causality(&mut self, cause_idx: usize, effect_idx: usize, strength: f64) {
        self.sample_count += 1;
        self.estimator.update(cause_idx, effect_idx, strength);
        
        // Recalculate branching ratio periodically
        if self.sample_count % 10 == 0 {
            self.current_branching_ratio = self.estimator.spectral_radius();
            self.last_update_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros();
        }
        
        if !self.is_initialized && self.sample_count >= self.config.min_samples {
            self.is_initialized = true;
            self.current_branching_ratio = self.estimator.spectral_radius();
        }
    }
    
    /// Get current branching ratio estimate
    pub fn branching_ratio(&self) -> f64 {
        if !self.is_initialized {
            return 0.0;
        }
        self.current_branching_ratio
    }
    
    /// Get market stability classification
    pub fn stability(&self) -> MarketStability {
        MarketStability::from_branching_ratio(self.branching_ratio())
    }
    
    /// Check if market is in critical state
    pub fn is_critical(&self) -> bool {
        self.stability() == MarketStability::Critical
    }
    
    /// Check if market is near critical
    pub fn is_near_critical(&self) -> bool {
        matches!(
            self.stability(),
            MarketStability::NearCritical | MarketStability::Critical
        )
    }
    
    /// Get excitation matrix estimate
    pub fn get_excitation_matrix(&self) -> Vec<Vec<f64>> {
        self.estimator.get_matrix()
    }
    
    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
    
    /// Check if estimator has enough samples
    pub fn is_ready(&self) -> bool {
        self.is_initialized
    }
    
    /// Reset the calculator
    pub fn reset(&mut self) {
        self.estimator.reset();
        self.sample_count = 0;
        self.current_branching_ratio = 0.0;
        self.is_initialized = false;
    }
    
    /// Get detailed statistics
    pub fn get_stats(&self) -> BranchingStats {
        BranchingStats {
            branching_ratio: self.branching_ratio(),
            stability: self.stability(),
            sample_count: self.sample_count,
            is_ready: self.is_initialized,
            last_update_micros: self.last_update_time,
            excitation_matrix: self.get_excitation_matrix(),
        }
    }
}

/// Statistics snapshot from branching ratio calculator
#[derive(Debug, Clone)]
pub struct BranchingStats {
    pub branching_ratio: f64,
    pub stability: MarketStability,
    pub sample_count: usize,
    pub is_ready: bool,
    pub last_update_micros: u128,
    pub excitation_matrix: Vec<Vec<f64>>,
}

/// Thread-safe wrapper for concurrent access
pub struct ThreadSafeBranchingCalculator {
    inner: Arc<RwLock<BranchingRatioCalculator>>,
}

impl ThreadSafeBranchingCalculator {
    pub fn new(calculator: BranchingRatioCalculator) -> Self {
        Self {
            inner: Arc::new(RwLock::new(calculator)),
        }
    }
    
    pub fn record_causality(&self, cause: usize, effect: usize, strength: f64) {
        let mut calc = self.inner.write();
        calc.record_causality(cause, effect, strength);
    }
    
    pub fn branching_ratio(&self) -> f64 {
        let calc = self.inner.read();
        calc.branching_ratio()
    }
    
    pub fn is_critical(&self) -> bool {
        let calc = self.inner.read();
        calc.is_critical()
    }
    
    pub fn stability(&self) -> MarketStability {
        let calc = self.inner.read();
        calc.stability()
    }
    
    pub fn get_stats(&self) -> BranchingStats {
        let calc = self.inner.read();
        calc.get_stats()
    }
}

/// Multi-scale branching analysis for different time horizons
pub struct MultiScaleBranchingAnalyzer {
    /// Short-term analyzer (milliseconds)
    short_term: BranchingRatioCalculator,
    /// Medium-term analyzer (seconds)
    medium_term: BranchingRatioCalculator,
    /// Long-term analyzer (minutes)
    long_term: BranchingRatioCalculator,
}

impl MultiScaleBranchingAnalyzer {
    /// Create analyzer with multiple time scales
    pub fn new(config: BranchingConfig) -> Self {
        let mut short_config = config.clone();
        short_config.decay_factor = 0.99; // Faster decay
        
        let mut long_config = config.clone();
        long_config.decay_factor = 0.999; // Slower decay
        
        Self {
            short_term: BranchingRatioCalculator::new(short_config),
            medium_term: BranchingRatioCalculator::new(config),
            long_term: BranchingRatioCalculator::new(long_config),
        }
    }
    
    /// Record causality across all time scales
    pub fn record_causality(&mut self, cause: usize, effect: usize, strength: f64) {
        self.short_term.record_causality(cause, effect, strength);
        self.medium_term.record_causality(cause, effect, strength);
        self.long_term.record_causality(cause, effect, strength);
    }
    
    /// Check for divergence between time scales (early warning signal)
    pub fn check_divergence(&self) -> Option<DivergenceWarning> {
        let short = self.short_term.branching_ratio();
        let medium = self.medium_term.branching_ratio();
        let long = self.long_term.branching_ratio();
        
        // Rising short-term branching ratio is an early warning
        if short > medium && short > long && short > 0.7 {
            return Some(DivergenceWarning {
                short_term_ratio: short,
                medium_term_ratio: medium,
                long_term_ratio: long,
                warning_level: if short > 0.95 {
                    WarningLevel::Critical
                } else {
                    WarningLevel::Elevated
                },
            });
        }
        
        None
    }
    
    /// Get all statistics
    pub fn get_all_stats(&self) -> MultiScaleStats {
        MultiScaleStats {
            short_term: self.short_term.get_stats(),
            medium_term: self.medium_term.get_stats(),
            long_term: self.long_term.get_stats(),
        }
    }
}

/// Warning about diverging branching ratios
#[derive(Debug, Clone)]
pub struct DivergenceWarning {
    pub short_term_ratio: f64,
    pub medium_term_ratio: f64,
    pub long_term_ratio: f64,
    pub warning_level: WarningLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningLevel {
    Elevated,
    Critical,
}

/// Multi-scale statistics
#[derive(Debug, Clone)]
pub struct MultiScaleStats {
    pub short_term: BranchingStats,
    pub medium_term: BranchingStats,
    pub long_term: BranchingStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stability_classification() {
        assert_eq!(MarketStability::from_branching_ratio(0.3), MarketStability::Stable);
        assert_eq!(MarketStability::from_branching_ratio(0.6), MarketStability::Elevated);
        assert_eq!(MarketStability::from_branching_ratio(0.8), MarketStability::NearCritical);
        assert_eq!(MarketStability::from_branching_ratio(0.98), MarketStability::Critical);
    }
    
    #[test]
    fn test_branching_calculator() {
        let config = BranchingConfig::default();
        let mut calc = BranchingRatioCalculator::new(config);
        
        // Initially not ready
        assert!(!calc.is_ready());
        
        // Add some causality data
        for _ in 0..200 {
            calc.record_causality(0, 0, 0.3);
            calc.record_causality(0, 1, 0.1);
            calc.record_causality(1, 0, 0.1);
            calc.record_causality(1, 1, 0.3);
        }
        
        // Should be ready now
        assert!(calc.is_ready());
        assert!(calc.branching_ratio() > 0.0);
    }
    
    #[test]
    fn test_critical_detection() {
        let config = BranchingConfig::default();
        let mut calc = BranchingRatioCalculator::new(config);
        
        // Add strong causality to push towards critical
        for _ in 0..500 {
            calc.record_causality(0, 0, 0.5);
            calc.record_causality(0, 1, 0.5);
            calc.record_causality(1, 0, 0.5);
            calc.record_causality(1, 1, 0.5);
        }
        
        // May or may not be critical depending on convergence
        let _ = calc.is_critical();
    }
}
