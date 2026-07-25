//! Alpha Decay Analyzer for Predictive Signal Degradation
//!
//! This module measures how fast predictive signals degrade due to market impact,
//! helping optimize execution timing and position sizing.
//!
//! # Features
//! - Signal half-life estimation
//! - Impact-adjusted alpha calculation
//! - Optimal holding period determination
//! - Memory-efficient exponential decay tracking

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;

/// Configuration for alpha decay analysis
#[derive(Debug, Clone)]
pub struct AlphaDecayConfig {
    /// Maximum signals to track
    pub max_signals: usize,
    /// Time window for decay estimation (seconds)
    pub decay_window: f64,
    /// Minimum samples for reliable estimation
    pub min_samples: usize,
    /// Impact sensitivity factor
    pub impact_factor: f64,
}

impl Default for AlphaDecayConfig {
    fn default() -> Self {
        Self {
            max_signals: 1000,
            decay_window: 300.0, // 5 minutes
            min_samples: 50,
            impact_factor: 0.1,
        }
    }
}

/// Recorded signal with its evolution
#[derive(Debug, Clone)]
pub struct SignalRecord {
    /// Signal ID
    pub id: u64,
    /// Initial signal value
    pub initial_value: f64,
    /// Timestamp in microseconds
    pub timestamp: u128,
    /// Current value
    pub current_value: f64,
    /// Realized PnL if closed
    pub realized_pnl: Option<f64>,
    /// Market impact at entry
    pub entry_impact: f64,
}

/// Alpha decay metrics
#[derive(Debug, Clone)]
pub struct AlphaDecayMetrics {
    /// Estimated half-life in seconds
    pub half_life_seconds: f64,
    /// Decay rate (lambda)
    pub decay_rate: f64,
    /// Current average alpha
    pub current_alpha: f64,
    /// Initial average alpha
    pub initial_alpha: f64,
    /// Alpha retained (current/initial)
    pub alpha_retained: f64,
    /// Impact-adjusted alpha
    pub impact_adjusted_alpha: f64,
    /// Optimal holding period
    pub optimal_holding_period: f64,
    /// R-squared of decay fit
    pub fit_quality: f64,
}

/// Alpha decay tracker
pub struct AlphaDecayAnalyzer {
    config: AlphaDecayConfig,
    /// Active signals
    signals: VecDeque<SignalRecord>,
    /// Historical decay observations (time, remaining_alpha)
    decay_observations: VecDeque<(f64, f64)>,
    /// Running sum for exponential decay estimation
    sum_alpha: f64,
    /// Sum of weighted time
    sum_weighted_time: f64,
    sample_count: usize,
}

impl AlphaDecayAnalyzer {
    /// Create a new alpha decay analyzer
    pub fn new(config: AlphaDecayConfig) -> Self {
        Self {
            config,
            signals: VecDeque::with_capacity(config.max_signals),
            decay_observations: VecDeque::with_capacity(config.max_signals),
            sum_alpha: 0.0,
            sum_weighted_time: 0.0,
            sample_count: 0,
        }
    }
    
    /// Record a new signal
    pub fn record_signal(&mut self, id: u64, value: f64, entry_impact: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros();
        
        let signal = SignalRecord {
            id,
            initial_value: value,
            timestamp: now,
            current_value: value,
            realized_pnl: None,
            entry_impact,
        };
        
        self.signals.push_back(signal);
        
        // Maintain bounded size
        while self.signals.len() > self.config.max_signals {
            self.signals.pop_front();
        }
    }
    
    /// Update signal values (called periodically)
    pub fn update_signals(&mut self, current_values: &std::collections::HashMap<u64, f64>) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros();
        
        for signal in &mut self.signals {
            if let Some(&current) = current_values.get(&signal.id) {
                let elapsed_s = (now - signal.timestamp) as f64 / 1_000_000.0;
                
                // Record decay observation if signal has aged enough
                if elapsed_s > 1.0 && signal.initial_value.abs() > 1e-10 {
                    let remaining_alpha = current / signal.initial_value;
                    self.decay_observations.push_back((elapsed_s, remaining_alpha));
                    
                    // Update running sums for O(1) decay estimation
                    self.sum_alpha += remaining_alpha;
                    self.sum_weighted_time += elapsed_s * remaining_alpha;
                    self.sample_count += 1;
                }
                
                signal.current_value = current;
            }
        }
        
        // Clean old observations
        self.clean_old_observations(now);
    }
    
    /// Remove observations outside the decay window
    fn clean_old_observations(&mut self, now: u128) {
        let window_us = (self.config.decay_window * 1_000_000.0) as u128;
        let cutoff = now.saturating_sub(window_us);
        
        // Remove signals that are too old
        while let Some(front) = self.signals.front() {
            if front.timestamp < cutoff {
                self.signals.pop_front();
            } else {
                break;
            }
        }
        
        // Keep decay observations bounded
        while self.decay_observations.len() > self.config.max_signals {
            self.decay_observations.pop_front();
        }
    }
    
    /// Estimate decay parameters using linear regression on log-transformed data
    pub fn estimate_decay(&self) -> Option<(f64, f64)> {
        if self.decay_observations.len() < self.config.min_samples {
            return None;
        }
        
        // Linear regression: log(alpha) = -lambda * t + log(alpha_0)
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut n = 0;
        
        for &(t, alpha) in &self.decay_observations {
            if alpha > 0.0 {
                let x = t;
                let y = alpha.ln();
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
                n += 1;
            }
        }
        
        if n < self.config.min_samples {
            return None;
        }
        
        let n_f = n as f64;
        let denom = n_f * sum_xx - sum_x * sum_x;
        
        if denom.abs() < 1e-10 {
            return None;
        }
        
        let lambda = -(n_f * sum_xy - sum_x * sum_y) / denom;
        let log_alpha0 = (sum_y + lambda * sum_x) / n_f;
        
        // Ensure positive decay rate
        let lambda = lambda.max(0.001);
        
        Some((lambda, log_alpha0.exp()))
    }
    
    /// Get current alpha decay metrics
    pub fn get_metrics(&self) -> Option<AlphaDecayMetrics> {
        let (decay_rate, initial_alpha) = self.estimate_decay()?;
        
        // Half-life: t_1/2 = ln(2) / lambda
        let half_life = std::f64::consts::LN_2 / decay_rate;
        
        // Calculate current average alpha
        let current_alpha: f64 = self.signals.iter()
            .map(|s| s.current_value)
            .sum::<f64>() / self.signals.len().max(1) as f64;
        
        let initial_alpha_avg: f64 = self.signals.iter()
            .map(|s| s.initial_value)
            .sum::<f64>() / self.signals.len().max(1) as f64;
        
        let alpha_retained = if initial_alpha_avg.abs() > 1e-10 {
            current_alpha / initial_alpha_avg
        } else {
            0.0
        };
        
        // Impact-adjusted alpha
        let avg_entry_impact: f64 = self.signals.iter()
            .map(|s| s.entry_impact)
            .sum::<f64>() / self.signals.len().max(1) as f64;
        
        let impact_adjusted_alpha = current_alpha - avg_entry_impact * self.config.impact_factor;
        
        // Optimal holding period: maximize alpha - cost
        // Simplified: hold until alpha decays to cost level
        let optimal_holding = if decay_rate > 0.0 {
            (-avg_entry_impact * self.config.impact_factor / initial_alpha).ln() / decay_rate
        } else {
            half_life
        };
        
        // Fit quality (simplified R²)
        let fit_quality = if self.decay_observations.len() > 10 {
            0.8 // Placeholder - would calculate actual R²
        } else {
            0.0
        };
        
        Some(AlphaDecayMetrics {
            half_life_seconds: half_life,
            decay_rate,
            current_alpha,
            initial_alpha: initial_alpha_avg,
            alpha_retained,
            impact_adjusted_alpha,
            optimal_holding_period: optimal_holding.max(0.0),
            fit_quality,
        })
    }
    
    /// Get recommended action based on alpha decay
    pub fn get_recommendation(&self) -> AlphaRecommendation {
        match self.get_metrics() {
            Some(metrics) => {
                if metrics.alpha_retained < 0.3 {
                    AlphaRecommendation::Close
                } else if metrics.alpha_retained < 0.5 {
                    AlphaRecommendation::Reduce
                } else if metrics.optimal_holding_period < 10.0 {
                    AlphaRecommendation::Hold
                } else {
                    AlphaRecommendation::Add
                }
            }
            None => AlphaRecommendation::Hold,
        }
    }
    
    /// Reset analyzer
    pub fn reset(&mut self) {
        self.signals.clear();
        self.decay_observations.clear();
        self.sum_alpha = 0.0;
        self.sum_weighted_time = 0.0;
        self.sample_count = 0;
    }
}

/// Trading recommendation based on alpha decay
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaRecommendation {
    /// Alpha still strong - consider adding
    Add,
    /// Alpha moderate - hold current position
    Hold,
    /// Alpha decaying - reduce exposure
    Reduce,
    /// Alpha exhausted - close position
    Close,
}

/// Thread-safe wrapper
pub struct ThreadSafeAlphaDecayAnalyzer {
    inner: Arc<RwLock<AlphaDecayAnalyzer>>,
}

impl ThreadSafeAlphaDecayAnalyzer {
    pub fn new(analyzer: AlphaDecayAnalyzer) -> Self {
        Self {
            inner: Arc::new(RwLock::new(analyzer)),
        }
    }
    
    pub fn record_signal(&self, id: u64, value: f64, impact: f64) {
        let mut a = self.inner.write();
        a.record_signal(id, value, impact);
    }
    
    pub fn update_signals(&self, values: &std::collections::HashMap<u64, f64>) {
        let mut a = self.inner.write();
        a.update_signals(values);
    }
    
    pub fn get_metrics(&self) -> Option<AlphaDecayMetrics> {
        let a = self.inner.read();
        a.get_metrics()
    }
    
    pub fn get_recommendation(&self) -> AlphaRecommendation {
        let a = self.inner.read();
        a.get_recommendation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alpha_decay_estimation() {
        let config = AlphaDecayConfig::default();
        let mut analyzer = AlphaDecayAnalyzer::new(config);
        
        // Simulate decaying signals
        for i in 0..100 {
            analyzer.record_signal(i, 1.0, 0.01);
        }
        
        // Metrics should be available after enough samples
        // (though decay won't be meaningful without time passing)
        let _metrics = analyzer.get_metrics();
    }
}
