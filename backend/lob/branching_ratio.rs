//! Branching Ratio Estimator for Order Book Stability Analysis
//! 
//! This module estimates the branching ratio of the Hawkes process, which is a critical
//! measure of market stability. The branching ratio indicates how close the market is
//! to a critical state where small perturbations can cause cascading effects.
//! 
//! Key Concepts:
//! - Branching ratio n = alpha / beta (excitation / decay)
//! - n < 1: Sub-critical (stable) - shocks decay over time
//! - n = 1: Critical - market at tipping point
//! - n > 1: Super-critical (unstable) - shocks amplify, potential flash crash
//! 
//! Optimized for real-time estimation with minimal memory footprint.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for branching ratio estimation
#[derive(Debug, Clone)]
pub struct BranchingRatioConfig {
    /// Minimum events required for reliable estimation
    pub min_events: usize,
    /// Time window for estimation (milliseconds)
    pub estimation_window_ms: u64,
    /// Smoothing factor for EMA-based estimation
    pub smoothing_alpha: f64,
    /// Critical threshold (typically 0.8-1.0)
    pub critical_threshold: f64,
    /// Warning threshold (before critical)
    pub warning_threshold: f64,
}

impl Default for BranchingRatioConfig {
    fn default() -> Self {
        Self {
            min_events: 50,
            estimation_window_ms: 5000, // 5 seconds
            smoothing_alpha: 0.1,
            critical_threshold: 0.95,
            warning_threshold: 0.75,
        }
    }
}

/// Stability state of the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityState {
    /// Stable, far from critical
    Stable,
    /// Approaching critical state
    Warning,
    /// Near critical, high risk
    Critical,
    /// Unstable, super-critical regime
    Unstable,
}

impl StabilityState {
    /// Get risk level as numeric value (0-3)
    pub fn risk_level(&self) -> u8 {
        match self {
            StabilityState::Stable => 0,
            StabilityState::Warning => 1,
            StabilityState::Critical => 2,
            StabilityState::Unstable => 3,
        }
    }
    
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            StabilityState::Stable => "Market stable, normal conditions",
            StabilityState::Warning => "Caution: approaching instability",
            StabilityState::Critical => "High risk: near critical state",
            StabilityState::Unstable => "Danger: super-critical, flash crash risk",
        }
    }
}

/// Snapshot of branching ratio estimation
#[derive(Debug, Clone)]
pub struct BranchingRatioSnapshot {
    /// Current estimated branching ratio
    pub branching_ratio: f64,
    /// Confidence in the estimate (0-1)
    pub confidence: f64,
    /// Current stability state
    pub stability_state: StabilityState,
    /// Number of events used in estimation
    pub event_count: usize,
    /// Estimated excitation parameter (alpha)
    pub alpha_estimate: f64,
    /// Estimated decay parameter (beta)
    pub beta_estimate: f64,
    /// Timestamp of estimation
    pub timestamp: Instant,
}

/// Real-time branching ratio estimator using multiple methods
pub struct BranchingRatioEstimator {
    config: BranchingRatioConfig,
    
    /// Event timestamps for inter-arrival analysis
    event_timestamps: VecDeque<Instant>,
    
    /// Event magnitudes (volumes)
    event_magnitudes: VecDeque<f64>,
    
    /// EMA-smoothed branching ratio estimate
    ema_branching_ratio: f64,
    
    /// Maximum likelihood estimate (more accurate but slower)
    mle_branching_ratio: Option<f64>,
    
    /// Current stability state
    current_state: StabilityState,
    
    /// Last update timestamp
    last_update: Instant,
    
    /// Total events processed
    total_events: usize,
    
    /// Thread safety
    inner: Arc<Mutex<()>>,
}

impl BranchingRatioEstimator {
    /// Create a new branching ratio estimator
    pub fn new(config: BranchingRatioConfig) -> Self {
        let capacity = (config.estimation_window_ms / 10) as usize; // Approximate capacity
        
        Self {
            config,
            event_timestamps: VecDeque::with_capacity(capacity),
            event_magnitudes: VecDeque::with_capacity(capacity),
            ema_branching_ratio: 0.0,
            mle_branching_ratio: None,
            current_state: StabilityState::Stable,
            last_update: Instant::now(),
            total_events: 0,
            inner: Arc::new(Mutex::new(())),
        }
    }
    
    /// Record a new event for branching ratio estimation
    pub fn record_event(&mut self, magnitude: f64) {
        let now = Instant::now();
        
        // Store event data
        self.event_timestamps.push_back(now);
        self.event_magnitudes.push_back(magnitude);
        self.total_events += 1;
        
        // Cleanup old events outside window
        self.cleanup_old_events(now);
        
        // Update estimates if enough events
        if self.event_timestamps.len() >= self.config.min_events {
            self.update_estimates();
        }
        
        self.last_update = now;
    }
    
    /// Get current branching ratio snapshot
    pub fn get_snapshot(&self) -> Option<BranchingRatioSnapshot> {
        let event_count = self.event_timestamps.len();
        
        if event_count < self.config.min_events {
            return None;
        }
        
        // Use MLE if available, otherwise EMA
        let branching_ratio = self.mle_branching_ratio.unwrap_or(self.ema_branching_ratio);
        
        // Calculate confidence based on event count and recency
        let confidence = self.calculate_confidence(event_count);
        
        Some(BranchingRatioSnapshot {
            branching_ratio,
            confidence,
            stability_state: self.current_state,
            event_count,
            alpha_estimate: self.estimate_alpha(),
            beta_estimate: self.estimate_beta(),
            timestamp: self.last_update,
        })
    }
    
    /// Check if market is in critical state
    pub fn is_critical(&self) -> bool {
        self.current_state == StabilityState::Critical || 
        self.current_state == StabilityState::Unstable
    }
    
    /// Get current stability state
    pub fn get_stability_state(&self) -> StabilityState {
        self.current_state
    }
    
    /// Get raw branching ratio estimate
    pub fn get_branching_ratio(&self) -> f64 {
        self.mle_branching_ratio.unwrap_or(self.ema_branching_ratio)
    }
    
    /// Reset estimator state
    pub fn reset(&mut self) {
        self.event_timestamps.clear();
        self.event_magnitudes.clear();
        self.ema_branching_ratio = 0.0;
        self.mle_branching_ratio = None;
        self.current_state = StabilityState::Stable;
        self.last_update = Instant::now();
        self.total_events = 0;
    }
    
    /// Cleanup events outside the estimation window
    fn cleanup_old_events(&mut self, now: Instant) {
        let cutoff = Duration::from_millis(self.config.estimation_window_ms);
        
        while let Some(&ts) = self.event_timestamps.front() {
            if now.duration_since(ts) > cutoff {
                self.event_timestamps.pop_front();
                self.event_magnitudes.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Update branching ratio estimates
    fn update_estimates(&mut self) {
        // Update EMA estimate
        let current_estimate = self.estimate_branching_ratio_ema();
        self.ema_branching_ratio = self.ema_branching_ratio 
            + self.config.smoothing_alpha * (current_estimate - self.ema_branching_ratio);
        
        // Update MLE estimate periodically (more computationally expensive)
        if self.total_events % 10 == 0 {
            self.mle_branching_ratio = self.estimate_branching_ratio_mle();
        }
        
        // Use MLE if available, otherwise EMA for state determination
        let effective_ratio = self.mle_branching_ratio.unwrap_or(self.ema_branching_ratio);
        
        // Update stability state
        self.current_state = self.determine_stability_state(effective_ratio);
    }
    
    /// Estimate branching ratio using EMA method (fast, less accurate)
    fn estimate_branching_ratio_ema(&self) -> f64 {
        if self.event_timestamps.len() < 2 {
            return 0.0;
        }
        
        // Simple ratio of recent excitations to decays
        let mut excitation_sum = 0.0;
        let mut decay_sum = 0.0;
        
        let timestamps: Vec<_> = self.event_timestamps.iter().copied().collect();
        let magnitudes: Vec<_> = self.event_magnitudes.iter().copied().collect();
        
        for i in 1..timestamps.len() {
            let dt = timestamps[i].duration_since(timestamps[i-1]).as_secs_f64();
            
            // Excitation from event magnitude
            excitation_sum += magnitudes[i];
            
            // Decay over time
            decay_sum += dt * 2.0; // Assume baseline decay rate of 2.0
        }
        
        if decay_sum > 0.0 {
            (excitation_sum / decay_sum).min(2.0) // Cap at 2.0 for numerical stability
        } else {
            0.0
        }
    }
    
    /// Estimate branching ratio using Maximum Likelihood Estimation (accurate, slower)
    /// 
    /// Uses the likelihood function for Hawkes processes with exponential kernel
    fn estimate_branching_ratio_mle(&self) -> Option<f64> {
        if self.event_timestamps.len() < self.config.min_events {
            return None;
        }
        
        let timestamps: Vec<_> = self.event_timestamps.iter().copied().collect();
        let n = timestamps.len();
        
        // Grid search for optimal parameters (can be optimized with gradient descent)
        let mut best_ratio = 0.0;
        let mut best_likelihood = f64::NEG_INFINITY;
        
        // Search branching ratio from 0.1 to 1.5
        for ratio_candidate in (1..=30).map(|i| i as f64 / 20.0) {
            let likelihood = self.compute_log_likelihood(&timestamps, ratio_candidate);
            
            if likelihood > best_likelihood {
                best_likelihood = likelihood;
                best_ratio = ratio_candidate;
            }
        }
        
        Some(best_ratio)
    }
    
    /// Compute log-likelihood for given branching ratio
    fn compute_log_likelihood(&self, timestamps: &[Instant], ratio: f64) -> f64 {
        if timestamps.len() < 2 {
            return f64::NEG_INFINITY;
        }
        
        let t_start = timestamps[0];
        let t_end = timestamps[timestamps.len() - 1];
        let total_time = t_end.duration_since(t_start).as_secs_f64();
        
        if total_time <= 0.0 {
            return f64::NEG_INFINITY;
        }
        
        // Simplified likelihood calculation
        // Full implementation would use the complete Hawkes likelihood
        let mu = timestamps.len() as f64 / total_time; // Background rate
        let beta = 2.0; // Fixed decay for simplicity
        let alpha = ratio * beta;
        
        let mut log_likelihood = 0.0;
        
        for i in 1..timestamps.len() {
            let dt = timestamps[i].duration_since(timestamps[i-1]).as_secs_f64();
            
            // Conditional intensity
            let lambda_t = mu + alpha * mu / beta * (1.0 - (-beta * dt).exp());
            
            if lambda_t > 0.0 {
                log_likelihood += lambda_t.ln();
            }
        }
        
        // Subtract integral of intensity
        log_likelihood -= mu * total_time + alpha * mu / beta * total_time;
        
        log_likelihood
    }
    
    /// Estimate alpha parameter
    fn estimate_alpha(&self) -> f64 {
        let beta = self.estimate_beta();
        self.get_branching_ratio() * beta
    }
    
    /// Estimate beta parameter (decay rate)
    fn estimate_beta(&self) -> f64 {
        if self.event_timestamps.len() < 2 {
            return 2.0; // Default
        }
        
        // Estimate from average inter-arrival time
        let timestamps: Vec<_> = self.event_timestamps.iter().copied().collect();
        let mut total_dt = 0.0;
        let mut count = 0;
        
        for i in 1..timestamps.len() {
            let dt = timestamps[i].duration_since(timestamps[i-1]).as_secs_f64();
            total_dt += dt;
            count += 1;
        }
        
        if count > 0 && total_dt > 0.0 {
            let avg_dt = total_dt / count as f64;
            // Beta is inversely related to typical inter-arrival time
            (1.0 / avg_dt).clamp(0.5, 10.0)
        } else {
            2.0
        }
    }
    
    /// Determine stability state from branching ratio
    fn determine_stability_state(&self, ratio: f64) -> StabilityState {
        if ratio >= 1.0 {
            StabilityState::Unstable
        } else if ratio >= self.config.critical_threshold {
            StabilityState::Critical
        } else if ratio >= self.config.warning_threshold {
            StabilityState::Warning
        } else {
            StabilityState::Stable
        }
    }
    
    /// Calculate confidence in the estimate
    fn calculate_confidence(&self, event_count: usize) -> f64 {
        // Confidence increases with event count up to a maximum
        let count_factor = (event_count as f64 / self.config.min_events as f64).min(1.0);
        
        // Confidence decreases if data is stale
        let elapsed = self.last_update.elapsed().as_secs_f64();
        let freshness_factor = (-elapsed / 5.0).exp(); // Decay over 5 seconds
        
        count_factor * freshness_factor
    }
    
    /// Get early warning signal for approaching criticality
    pub fn get_criticality_warning(&self) -> Option<CriticalityWarning> {
        let snapshot = self.get_snapshot()?;
        
        if snapshot.stability_state == StabilityState::Stable {
            return None;
        }
        
        Some(CriticalityWarning {
            state: snapshot.stability_state,
            branching_ratio: snapshot.branching_ratio,
            distance_to_critical: 1.0 - snapshot.branching_ratio,
            recommended_action: self.get_recommended_action(snapshot.stability_state),
        })
    }
    
    /// Get recommended action based on stability state
    fn get_recommended_action(&self, state: StabilityState) -> &'static str {
        match state {
            StabilityState::Stable => "Normal trading operations",
            StabilityState::Warning => "Reduce position sizes, tighten stops",
            StabilityState::Critical => "Pause new entries, reduce exposure",
            StabilityState::Unstable => "Emergency exit, avoid market orders",
        }
    }
}

/// Warning information for approaching criticality
#[derive(Debug, Clone)]
pub struct CriticalityWarning {
    pub state: StabilityState,
    pub branching_ratio: f64,
    pub distance_to_critical: f64,
    pub recommended_action: &'static str,
}

/// Multi-asset branching ratio tracker
pub struct MultiAssetBranchingTracker {
    trackers: std::collections::HashMap<String, BranchingRatioEstimator>,
}

impl MultiAssetBranchingTracker {
    /// Create tracker for multiple assets
    pub fn new(assets: Vec<&str>, config: BranchingRatioConfig) -> Self {
        let trackers = assets
            .into_iter()
            .map(|asset| (asset.to_string(), BranchingRatioEstimator::new(config.clone())))
            .collect();
        
        Self { trackers }
    }
    
    /// Record event for specific asset
    pub fn record_event(&mut self, asset: &str, magnitude: f64) {
        if let Some(tracker) = self.trackers.get_mut(asset) {
            tracker.record_event(magnitude);
        }
    }
    
    /// Get aggregate stability across all assets
    pub fn get_aggregate_stability(&self) -> StabilityState {
        let mut max_risk = 0u8;
        
        for tracker in self.trackers.values() {
            let risk = tracker.get_stability_state().risk_level();
            max_risk = max_risk.max(risk);
        }
        
        match max_risk {
            0 => StabilityState::Stable,
            1 => StabilityState::Warning,
            2 => StabilityState::Critical,
            _ => StabilityState::Unstable,
        }
    }
    
    /// Check if any asset is in critical state
    pub fn any_critical(&self) -> bool {
        self.trackers.values().any(|t| t.is_critical())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stable_market() {
        let config = BranchingRatioConfig {
            min_events: 10,
            ..Default::default()
        };
        let mut estimator = BranchingRatioEstimator::new(config);
        
        // Simulate stable market events (low excitation)
        for i in 0..50 {
            std::thread::sleep(Duration::from_millis(100));
            estimator.record_event(0.5); // Small, consistent magnitudes
        }
        
        let snapshot = estimator.get_snapshot();
        assert!(snapshot.is_some());
        
        let snap = snapshot.unwrap();
        // In stable market, branching ratio should be well below 1
        assert!(snap.branching_ratio < 0.8 || snap.confidence < 0.5);
    }
    
    #[test]
    fn test_stability_state_transitions() {
        let estimator = BranchingRatioEstimator::new(BranchingRatioConfig::default());
        
        // Test state determination logic
        assert_eq!(estimator.determine_stability_state(0.5), StabilityState::Stable);
        assert_eq!(estimator.determine_stability_state(0.8), StabilityState::Warning);
        assert_eq!(estimator.determine_stability_state(0.96), StabilityState::Critical);
        assert_eq!(estimator.determine_stability_state(1.1), StabilityState::Unstable);
    }
    
    #[test]
    fn test_risk_levels() {
        assert_eq!(StabilityState::Stable.risk_level(), 0);
        assert_eq!(StabilityState::Warning.risk_level(), 1);
        assert_eq!(StabilityState::Critical.risk_level(), 2);
        assert_eq!(StabilityState::Unstable.risk_level(), 3);
    }
}
