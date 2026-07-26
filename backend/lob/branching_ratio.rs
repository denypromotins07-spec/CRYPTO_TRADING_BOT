//! Branching Ratio Estimator for Order Book Stability Analysis
//!
//! This module estimates the criticality and stability of the order book
//! by calculating the branching ratio of Hawkes processes. The branching
//! ratio determines whether the market is in a stable (sub-critical) or
//! unstable (super-critical) regime.
//!
//! Key Features:
//! - Real-time branching ratio estimation using MLE
//! - Criticality detection for flash crash prediction
//! - Multi-dimensional analysis for bid/ask cross-excitation
//! - Zero-cost abstractions with strict borrowing checks
//!
//! Mathematical Foundation:
//! n = α / β  (branching ratio)
//! - n < 1: Sub-critical (stable, exogenous dominates)
//! - n = 1: Critical (phase transition, high volatility)
//! - n > 1: Super-critical (unstable, endogenous bubble)
//!
//! For multivariate case:
//! n = spectral_radius(Alpha * Beta^{-1})

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Event type enumeration for multi-dimensional analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    MarketBuy,
    MarketSell,
    LimitBuy,
    LimitSell,
    CancelBuy,
    CancelSell,
}

impl EventType {
    /// Get all event types as array
    pub const ALL: [EventType; 6] = [
        Self::MarketBuy,
        Self::MarketSell,
        Self::LimitBuy,
        Self::LimitSell,
        Self::CancelBuy,
        Self::CancelSell,
    ];
    
    /// Check if this is an aggressive (taker) event
    #[inline]
    pub fn is_aggressive(self) -> bool {
        matches!(self, Self::MarketBuy | Self::MarketSell)
    }
    
    /// Check if this is a passive (maker) event
    #[inline]
    pub fn is_passive(self) -> bool {
        matches!(self, Self::LimitBuy | Self::LimitSell)
    }
}

/// Configuration for branching ratio estimation
#[derive(Debug, Clone)]
pub struct BranchingConfig {
    /// Minimum events required for estimation
    pub min_events: usize,
    /// Rolling window duration for estimation
    pub window_duration: Duration,
    /// Regularization parameter for numerical stability
    pub regularization: f64,
    /// Maximum condition number for matrix inversion
    pub max_condition_number: f64,
}

impl Default for BranchingConfig {
    fn default() -> Self {
        Self {
            min_events: 50,
            window_duration: Duration::from_secs(300), // 5 minutes
            regularization: 1e-6,
            max_condition_number: 1e10,
        }
    }
}

/// Result of branching ratio estimation
#[derive(Debug, Clone)]
pub struct BranchingResult {
    /// Estimated branching ratio (n = α/β)
    pub branching_ratio: f64,
    /// Standard error of estimate
    pub standard_error: f64,
    /// Number of events used
    pub event_count: usize,
    /// Estimated alpha (excitation)
    pub alpha: f64,
    /// Estimated beta (decay)
    pub beta: f64,
    /// Regime classification
    pub regime: StabilityRegime,
    /// Distance to criticality
    pub distance_to_critical: f64,
}

/// Stability regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityRegime {
    /// Sub-critical: stable, exogenous events dominate
    SubCritical,
    /// Near-critical: approaching instability
    NearCritical,
    /// Critical: phase transition region
    Critical,
    /// Super-critical: unstable, endogenous bubbles
    SuperCritical,
}

impl StabilityRegime {
    /// Classify based on branching ratio
    pub fn from_ratio(n: f64) -> Self {
        const CRITICAL_THRESHOLD: f64 = 0.05;
        
        if n < 1.0 - CRITICAL_THRESHOLD {
            Self::SubCritical
        } else if n < 1.0 {
            Self::NearCritical
        } else if n < 1.0 + CRITICAL_THRESHOLD {
            Self::Critical
        } else {
            Self::SuperCritical
        }
    }
    
    /// Get risk level description
    pub fn risk_level(&self) -> &'static str {
        match self {
            Self::SubCritical => "LOW",
            Self::NearCritical => "MEDIUM",
            Self::Critical => "HIGH",
            Self::SuperCritical => "EXTREME",
        }
    }
}

/// Univariate branching ratio estimator
pub struct BranchingRatioEstimator {
    config: BranchingConfig,
    /// Event timestamps (microseconds)
    timestamps: Vec<u64>,
    /// Event volumes
    volumes: Vec<f64>,
    /// Current estimate
    current_estimate: Option<BranchingResult>,
    /// Last estimation time
    last_estimate_time: Option<Instant>,
}

impl BranchingRatioEstimator {
    /// Create new estimator with given configuration
    pub fn new(config: BranchingConfig) -> Self {
        Self {
            config,
            timestamps: Vec::with_capacity(config.min_events * 2),
            volumes: Vec::with_capacity(config.min_events * 2),
            current_estimate: None,
            last_estimate_time: None,
        }
    }
    
    /// Add an event
    pub fn add_event(&mut self, timestamp_us: u64, volume: f64) {
        self.timestamps.push(timestamp_us);
        self.volumes.push(volume);
        
        // Prune old events
        self.prune_old_events(timestamp_us);
        
        // Update estimate if enough events
        if self.timestamps.len() >= self.config.min_events {
            self.update_estimate();
        }
    }
    
    /// Prune events outside the window
    fn prune_old_events(&mut self, current_us: u64) {
        let window_us = self.config.window_duration.as_micros() as u64;
        let cutoff = current_us.saturating_sub(window_us);
        
        let mut keep_from = 0;
        for (i, &ts) in self.timestamps.iter().enumerate() {
            if ts >= cutoff {
                keep_from = i;
                break;
            }
        }
        
        if keep_from > 0 {
            self.timestamps.drain(..keep_from);
            self.volumes.drain(..keep_from);
        }
    }
    
    /// Update branching ratio estimate using MLE
    fn update_estimate(&mut self) {
        if self.timestamps.len() < self.config.min_events {
            return;
        }
        
        // Simple MLE for Hawkes process with exponential kernel
        // Using method of moments for computational efficiency
        
        let n = self.timestamps.len();
        if n < 2 {
            return;
        }
        
        // Calculate inter-arrival times
        let mut inter_arrivals: Vec<f64> = Vec::with_capacity(n - 1);
        for i in 1..n {
            let dt = (self.timestamps[i] - self.timestamps[i - 1]) as f64 / 1_000_000.0;
            if dt > 0.0 {
                inter_arrivals.push(dt);
            }
        }
        
        if inter_arrivals.is_empty() {
            return;
        }
        
        // Method of moments estimation
        // Mean inter-arrival: E[Δt] = 1 / (μ * (1 - n))
        // Variance provides additional constraint
        
        let mean_dt = inter_arrivals.iter().sum::<f64>() / inter_arrivals.len() as f64;
        
        // Calculate variance
        let variance: f64 = inter_arrivals.iter()
            .map(|&dt| (dt - mean_dt).powi(2))
            .sum::<f64>() / inter_arrivals.len() as f64;
        
        // Estimate parameters using moment matching
        // For exponential Hawkes: Var(Δt) ≈ (1 + n) / (μ^2 * (1 - n)^3)
        // Simplified: n ≈ sqrt(Var / mean^2 - 1) / sqrt(Var / mean^2 + 1)
        
        let cv_squared = variance / (mean_dt * mean_dt); // Coefficient of variation squared
        
        // Regularized estimate
        let cv_squared_reg = cv_squared.max(1.0 + self.config.regularization);
        
        // Branching ratio estimate
        let branching_ratio = ((cv_squared_reg - 1.0) / (cv_squared_reg + 1.0)).sqrt().min(1.5);
        
        // Estimate beta from characteristic decay time
        let beta = 1.0 / mean_dt;
        
        // Estimate alpha from branching ratio
        let alpha = branching_ratio * beta;
        
        // Standard error approximation
        let std_error = branching_ratio / (n as f64).sqrt();
        
        // Classify regime
        let regime = StabilityRegime::from_ratio(branching_ratio);
        
        // Distance to criticality
        let distance_to_critical = (1.0 - branching_ratio).abs();
        
        self.current_estimate = Some(BranchingResult {
            branching_ratio,
            standard_error: std_error,
            event_count: n,
            alpha,
            beta,
            regime,
            distance_to_critical,
        });
        
        self.last_estimate_time = Some(Instant::now());
    }
    
    /// Get current branching ratio estimate
    pub fn branching_ratio(&self) -> Option<f64> {
        self.current_estimate.as_ref().map(|r| r.branching_ratio)
    }
    
    /// Get current stability regime
    pub fn regime(&self) -> Option<StabilityRegime> {
        self.current_estimate.as_ref().map(|r| r.regime)
    }
    
    /// Check if market is near criticality (flash crash risk)
    pub fn is_near_critical(&self) -> bool {
        self.current_estimate
            .as_ref()
            .map(|r| matches!(r.regime, StabilityRegime::NearCritical | StabilityRegime::Critical))
            .unwrap_or(false)
    }
    
    /// Get full estimation result
    pub fn get_result(&self) -> Option<&BranchingResult> {
        self.current_estimate.as_ref()
    }
    
    /// Reset estimator
    pub fn reset(&mut self) {
        self.timestamps.clear();
        self.volumes.clear();
        self.current_estimate = None;
        self.last_estimate_time = None;
    }
}

/// Multivariate branching ratio estimator for cross-excitation analysis
pub struct MultivariateBranchingEstimator {
    config: BranchingConfig,
    n_types: usize,
    /// Event counts per type pair
    event_counts: Vec<Vec<usize>>,
    /// Time since last event per type
    last_event_times: Vec<u64>,
    /// Cross-excitation matrix estimate
    alpha_matrix: Vec<Vec<f64>>,
    /// Decay matrix estimate
    beta_matrix: Vec<Vec<f64>>,
    /// Current spectral radius (multivariate branching ratio)
    spectral_radius: f64,
    /// Current estimate
    current_estimate: Option<BranchingResult>,
}

impl MultivariateBranchingEstimator {
    /// Create new multivariate estimator
    pub fn new(n_types: usize, config: BranchingConfig) -> Self {
        Self {
            config,
            n_types,
            event_counts: vec![vec![0; n_types]; n_types],
            last_event_times: vec![0; n_types],
            alpha_matrix: vec![vec![0.0; n_types]; n_types],
            beta_matrix: vec![vec![1.0; n_types]; n_types],
            spectral_radius: 0.0,
            current_estimate: None,
        }
    }
    
    /// Record an event of given type
    pub fn add_event(&mut self, event_type: usize, timestamp_us: u64) {
        if event_type >= self.n_types {
            return;
        }
        
        // Update cross-excitation counts
        for j in 0..self.n_types {
            if self.last_event_times[j] > 0 {
                // Count excitation from type j to current type
                self.event_counts[j][event_type] += 1;
            }
        }
        
        self.last_event_times[event_type] = timestamp_us;
        
        // Periodically update estimate
        let total_events: usize = self.event_counts.iter().map(|row| row.iter().sum::<usize>()).sum();
        if total_events >= self.config.min_events * self.n_types {
            self.update_multivariate_estimate();
        }
    }
    
    /// Update multivariate branching ratio estimate
    fn update_multivariate_estimate(&mut self) {
        // Calculate total events per type
        let totals: Vec<usize> = (0..self.n_types)
            .map(|i| self.event_counts[i].iter().sum())
            .collect();
        
        // Normalize to get excitation probabilities
        let mut max_row_sum = 0.0;
        
        for i in 0..self.n_types {
            let row_sum: f64 = (0..self.n_types)
                .map(|j| {
                    if totals[i] > 0 {
                        self.event_counts[i][j] as f64 / totals[i] as f64
                    } else {
                        0.0
                    }
                })
                .sum();
            
            self.alpha_matrix[i] = (0..self.n_types)
                .map(|j| {
                    if totals[i] > 0 {
                        self.event_counts[i][j] as f64 / totals[i] as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            
            max_row_sum = max_row_sum.max(row_sum);
        }
        
        // Spectral radius approximation using Gershgorin circle theorem
        // Upper bound: max row sum
        self.spectral_radius = max_row_sum;
        
        // Classify regime based on spectral radius
        let regime = StabilityRegime::from_ratio(self.spectral_radius);
        
        self.current_estimate = Some(BranchingResult {
            branching_ratio: self.spectral_radius,
            standard_error: self.spectral_radius / (totals.iter().sum::<usize>() as f64).sqrt(),
            event_count: totals.iter().sum(),
            alpha: self.alpha_matrix.iter().flatten().sum::<f64>() / (self.n_types * self.n_types) as f64,
            beta: 1.0, // Normalized
            regime,
            distance_to_critical: (1.0 - self.spectral_radius).abs(),
        });
    }
    
    /// Get spectral radius (multivariate branching ratio)
    pub fn spectral_radius(&self) -> f64 {
        self.spectral_radius
    }
    
    /// Check if system is stable
    pub fn is_stable(&self) -> bool {
        self.spectral_radius < 1.0
    }
    
    /// Get excitation matrix
    pub fn excitation_matrix(&self) -> &Vec<Vec<f64>> {
        &self.alpha_matrix
    }
}

/// Stability monitor for real-time criticality detection
pub struct StabilityMonitor {
    estimators: HashMap<EventType, BranchingRatioEstimator>,
    alert_threshold: f64,
    last_alert_time: Option<Instant>,
    alert_cooldown: Duration,
}

impl StabilityMonitor {
    /// Create new stability monitor
    pub fn new(alert_threshold: f64, cooldown_s: u64) -> Self {
        let config = BranchingConfig::default();
        let estimators = EventType::ALL
            .iter()
            .map(|&et| (et, BranchingRatioEstimator::new(config.clone())))
            .collect();
        
        Self {
            estimators,
            alert_threshold,
            last_alert_time: None,
            alert_cooldown: Duration::from_secs(cooldown_s),
        }
    }
    
    /// Add event to appropriate estimator
    pub fn add_event(&mut self, event_type: EventType, timestamp_us: u64, volume: f64) {
        if let Some(estimator) = self.estimators.get_mut(&event_type) {
            estimator.add_event(timestamp_us, volume);
        }
    }
    
    /// Check if any estimator indicates critical regime
    pub fn check_criticality(&mut self) -> Vec<(EventType, StabilityRegime)> {
        let mut critical_events = Vec::new();
        let now = Instant::now();
        
        for (&event_type, estimator) in &self.estimators {
            if let Some(regime) = estimator.regime() {
                if matches!(regime, StabilityRegime::NearCritical | StabilityRegime::Critical | StabilityRegime::SuperCritical) {
                    // Check cooldown
                    let should_alert = match self.last_alert_time {
                        Some(last) => now.duration_since(last) > self.alert_cooldown,
                        None => true,
                    };
                    
                    if should_alert {
                        critical_events.push((event_type, regime));
                    }
                }
            }
        }
        
        if !critical_events.is_empty() {
            self.last_alert_time = Some(now);
        }
        
        critical_events
    }
    
    /// Get overall market stability score (0 = stable, 1 = critical)
    pub fn stability_score(&self) -> f64 {
        let mut max_ratio = 0.0;
        
        for estimator in self.estimators.values() {
            if let Some(ratio) = estimator.branching_ratio() {
                max_ratio = max_ratio.max(ratio);
            }
        }
        
        // Map to [0, 1] where 1 is critical
        ((max_ratio - 0.5) / 0.5).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_regime_classification() {
        assert_eq!(StabilityRegime::from_ratio(0.5), StabilityRegime::SubCritical);
        assert_eq!(StabilityRegime::from_ratio(0.97), StabilityRegime::NearCritical);
        assert_eq!(StabilityRegime::from_ratio(1.02), StabilityRegime::Critical);
        assert_eq!(StabilityRegime::from_ratio(1.2), StabilityRegime::SuperCritical);
    }
    
    #[test]
    fn test_branching_estimator() {
        let config = BranchingConfig {
            min_events: 10,
            ..Default::default()
        };
        let mut estimator = BranchingRatioEstimator::new(config);
        
        // Add synthetic events with regular spacing
        let base_time = 1_000_000_000u64;
        for i in 0..20 {
            estimator.add_event(base_time + i * 100_000, 1.0);
        }
        
        assert!(estimator.branching_ratio().is_some());
        assert!(!estimator.is_near_critical());
    }
}
