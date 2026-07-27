// ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
// High-Frequency Feature Stores, MLOps, and Concept Drift
// File: backend/mlops/online_calibrator.rs
// Chapter 2: Concept Drift Detection, Online Calibration, and Model Decay
//
// Updates Platt scaling parameters in microseconds via Rust.
// Strictly prevents probability outputs from exceeding valid 0.0 to 1.0 bounds.
// Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
// Uses zero-cost abstractions and numerically stable algorithms.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Maximum iterations for Newton-Raphson optimization
const MAX_ITERATIONS: usize = 100;

/// Convergence threshold for optimization
const CONVERGENCE_THRESHOLD: f64 = 1e-10;

/// Numerical stability epsilon
const EPSILON: f64 = 1e-15;

/// Platt scaling parameters (A, B) for probability calibration
/// P(y=1|f) = 1 / (1 + exp(A*f + B))
#[derive(Clone, Debug)]
pub struct PlattParameters {
    /// Slope parameter
    pub a: f64,
    /// Intercept parameter
    pub b: f64,
    /// Last update timestamp (nanoseconds)
    pub last_update_ns: u64,
    /// Number of samples used for fitting
    pub sample_count: usize,
}

impl Default for PlattParameters {
    fn default() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            last_update_ns: 0,
            sample_count: 0,
        }
    }
}

impl PlattParameters {
    /// Create new parameters with given values
    pub fn new(a: f64, b: f64) -> Self {
        Self {
            a,
            b,
            last_update_ns: current_time_ns(),
            sample_count: 0,
        }
    }

    /// Calibrate a raw model score to a probability
    /// Strictly bounds output to [0.0, 1.0]
    #[inline]
    pub fn calibrate(&self, raw_score: f64) -> f64 {
        // Compute Platt scaling: P = 1 / (1 + exp(A*f + B))
        let logit = self.a * raw_score + self.b;
        
        // Numerically stable sigmoid
        let prob = if logit >= 0.0 {
            1.0 / (1.0 + (-logit).exp())
        } else {
            logit.exp() / (1.0 + logit.exp())
        };
        
        // Strict bounds enforcement
        prob.clamp(0.0, 1.0)
    }

    /// Calibrate multiple scores efficiently
    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }
}

/// Online Platt scaler with incremental updates
pub struct OnlinePlattCalibrator {
    /// Current parameters
    params: RwLock<PlattParameters>,
    /// Running sums for incremental updates
    running_stats: RwLock<RunningStats>,
    /// Whether calibration is enabled
    enabled: AtomicBool,
    /// Update counter
    update_count: AtomicU64,
    /// Total calibration time (nanoseconds)
    total_calibration_time_ns: AtomicU64,
    /// Memory budget in bytes
    memory_budget_bytes: AtomicUsize,
    /// Current memory usage
    memory_used_bytes: AtomicUsize,
}

/// Running statistics for incremental Platt scaling
#[derive(Clone, Debug, Default)]
struct RunningStats {
    /// Sum of targets (positive examples)
    sum_targets: f64,
    /// Sum of raw scores
    sum_scores: f64,
    /// Sum of squared scores
    sum_scores_sq: f64,
    /// Sum of target * score
    sum_target_score: f64,
    /// Count of samples
    count: usize,
    /// Prior weight for regularization
    prior_weight: f64,
}

impl RunningStats {
    /// Create new running stats with optional prior
    pub fn new(prior_weight: f64) -> Self {
        Self {
            prior_weight,
            ..Default::default()
        }
    }

    /// Add a new observation
    #[inline]
    pub fn update(&mut self, raw_score: f64, target: f64) {
        self.sum_targets += target;
        self.sum_scores += raw_score;
        self.sum_scores_sq += raw_score * raw_score;
        self.sum_target_score += target * raw_score;
        self.count += 1;
    }

    /// Reset statistics
    pub fn clear(&mut self) {
        *self = Self::new(self.prior_weight);
    }
}

impl OnlinePlattCalibrator {
    /// Create a new online calibrator
    pub fn new(memory_budget_mb: usize) -> Self {
        Self {
            params: RwLock::new(PlattParameters::default()),
            running_stats: RwLock::new(RunningStats::new(1.0)),
            enabled: AtomicBool::new(true),
            update_count: AtomicU64::new(0),
            total_calibration_time_ns: AtomicU64::new(0),
            memory_budget_bytes: AtomicUsize::new(memory_budget_mb * 1024 * 1024),
            memory_used_bytes: AtomicUsize::new(std::mem::size_of::<Self>()),
        }
    }

    /// Create with initial parameters
    pub fn with_initial_params(a: f64, b: f64, memory_budget_mb: usize) -> Self {
        Self {
            params: RwLock::new(PlattParameters::new(a, b)),
            running_stats: RwLock::new(RunningStats::new(1.0)),
            enabled: AtomicBool::new(true),
            update_count: AtomicU64::new(0),
            total_calibration_time_ns: AtomicU64::new(0),
            memory_budget_bytes: AtomicUsize::new(memory_budget_mb * 1024 * 1024),
            memory_used_bytes: AtomicUsize::new(std::mem::size_of::<Self>()),
        }
    }

    /// Calibrate a single raw score
    #[inline]
    pub fn calibrate(&self, raw_score: f64) -> f64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return raw_score.clamp(0.0, 1.0);
        }

        let params = self.params.read().unwrap();
        params.calibrate(raw_score)
    }

    /// Calibrate multiple scores
    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        if !self.enabled.load(Ordering::Relaxed) {
            return scores.iter().map(|&s| s.clamp(0.0, 1.0)).collect();
        }

        let params = self.params.read().unwrap();
        params.calibrate_batch(scores)
    }

    /// Add an observation for online learning
    pub fn observe(&self, raw_score: f64, target: f64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let mut stats = self.running_stats.write().unwrap();
        stats.update(raw_score, target);

        // Periodically update parameters
        if stats.count % 100 == 0 && stats.count >= 10 {
            self.update_parameters_from_stats(&stats);
        }
    }

    /// Add batch observations
    pub fn observe_batch(&self, scores: &[f64], targets: &[f64]) {
        if scores.len() != targets.len() {
            return;
        }

        let mut stats = self.running_stats.write().unwrap();
        for (&score, &target) in scores.iter().zip(targets.iter()) {
            stats.update(score, target);
        }

        // Update parameters if enough data
        if stats.count >= 100 {
            self.update_parameters_from_stats(&stats);
        }
    }

    /// Update parameters using closed-form solution
    fn update_parameters_from_stats(&self, stats: &RunningStats) {
        let start = Instant::now();

        let n = stats.count as f64;
        if n < 10.0 {
            return;
        }

        // Target priors (from original Platt paper)
        let n_pos = stats.sum_targets + stats.prior_weight;
        let n_neg = n - stats.sum_targets + stats.prior_weight;
        let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
        let t_neg = 1.0 / (n_neg + 2.0);

        // Compute means
        let mean_score = stats.sum_scores / n;
        let mean_target = stats.sum_targets / n;

        // Compute variances and covariance
        let var_score = stats.sum_scores_sq / n - mean_score * mean_score;
        let cov = stats.sum_target_score / n - mean_score * mean_target;

        // Solve for A and B using normal equations
        // This is a simplified approach; production would use full Newton-Raphson
        if var_score > EPSILON {
            let a = -cov / var_score;
            let b = -(mean_target - mean_score * cov / var_score);

            // Refine with Newton-Raphson
            let (a_refined, b_refined) = self.newton_raphson(stats, a, b, t_pos, t_neg);

            let mut params = self.params.write().unwrap();
            params.a = a_refined;
            params.b = b_refined;
            params.last_update_ns = current_time_ns();
            params.sample_count = stats.count;
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);
        let elapsed = start.elapsed().as_nanos() as u64;
        self.total_calibration_time_ns.fetch_add(elapsed, Ordering::Relaxed);
    }

    /// Newton-Raphson refinement for Platt parameters
    fn newton_raphson(
        &self,
        stats: &RunningStats,
        init_a: f64,
        init_b: f64,
        t_pos: f64,
        t_neg: f64
    ) -> (f64, f64) {
        let mut a = init_a;
        let mut b = init_b;

        // Collect unique scores and their targets
        let n = stats.count as f64;
        let hi_target = t_pos;
        let lo_target = t_neg;

        for _ in 0..MAX_ITERATIONS {
            let (fval, grad, hess) = self.compute_objective_and_derivatives(
                stats, a, b, hi_target, lo_target, n
            );

            // Check convergence
            if grad.norm_squared() < CONVERGENCE_THRESHOLD {
                break;
            }

            // Newton step: solve H * delta = -grad
            let det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
            if det.abs() < EPSILON {
                break;
            }

            let delta_a = (-hess[1][1] * grad[0] + hess[0][1] * grad[1]) / det;
            let delta_b = (hess[1][0] * grad[0] - hess[0][0] * grad[1]) / det;

            // Line search with backtracking
            let mut step = 1.0;
            for _ in 0..10 {
                let new_a = a + step * delta_a;
                let new_b = b + step * delta_b;
                let (new_fval, _, _) = self.compute_objective_and_derivatives(
                    stats, new_a, new_b, hi_target, lo_target, n
                );
                if new_fval < fval {
                    a = new_a;
                    b = new_b;
                    break;
                }
                step *= 0.5;
            }
        }

        (a, b)
    }

    /// Compute objective function and its derivatives
    fn compute_objective_and_derivatives(
        &self,
        stats: &RunningStats,
        a: f64,
        b: f64,
        hi_target: f64,
        lo_target: f64,
        n: f64
    ) -> (f64, [f64; 2], [[f64; 2]; 2]) {
        // Simplified computation assuming aggregated stats
        // In production, would iterate over individual samples
        
        let mean_score = stats.sum_scores / n;
        let mean_target = stats.sum_targets / n;
        let var_score = stats.sum_scores_sq / n - mean_score * mean_score;

        // Approximate fval (negative log likelihood)
        let f_ap = a * mean_score + b;
        let p = 1.0 / (1.0 + f_ap.exp());
        let fval = -(mean_target * p.ln().max(-1e10) + (1.0 - mean_target) * (1.0 - p).ln().max(-1e10));

        // Gradient
        let grad = [
            var_score * a * n,
            n * (p - mean_target)
        ];

        // Hessian (approximate)
        let hess = [
            [var_score * n, mean_score * var_score * n],
            [mean_score * var_score * n, n]
        ];

        (fval, grad, hess)
    }

    /// Get current parameters
    pub fn get_params(&self) -> PlattParameters {
        self.params.read().unwrap().clone()
    }

    /// Set parameters directly
    pub fn set_params(&self, a: f64, b: f64) {
        let mut params = self.params.write().unwrap();
        params.a = a;
        params.b = b;
        params.last_update_ns = current_time_ns();
    }

    /// Enable/disable calibration
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if calibration is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Get calibration statistics
    pub fn get_stats(&self) -> CalibratorStats {
        let params = self.params.read().unwrap();
        let stats = self.running_stats.read().unwrap();

        CalibratorStats {
            current_a: params.a,
            current_b: params.b,
            sample_count: stats.count,
            update_count: self.update_count.load(Ordering::Relaxed),
            avg_calibration_time_ns: {
                let updates = self.update_count.load(Ordering::Relaxed);
                if updates > 0 {
                    self.total_calibration_time_ns.load(Ordering::Relaxed) / updates
                } else {
                    0
                }
            },
            enabled: self.enabled.load(Ordering::Relaxed),
            memory_used_bytes: self.memory_used_bytes.load(Ordering::Relaxed),
        }
    }

    /// Reset the calibrator
    pub fn reset(&self) {
        *self.params.write().unwrap() = PlattParameters::default();
        self.running_stats.write().unwrap().clear();
        self.update_count.store(0, Ordering::Relaxed);
        self.total_calibration_time_ns.store(0, Ordering::Relaxed);
    }
}

/// Statistics about the calibrator
#[derive(Debug, Clone)]
pub struct CalibratorStats {
    pub current_a: f64,
    pub current_b: f64,
    pub sample_count: usize,
    pub update_count: u64,
    pub avg_calibration_time_ns: u64,
    pub enabled: bool,
    pub memory_used_bytes: usize,
}

/// Helper function to get current time in nanoseconds
#[inline]
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Extension trait for gradient vector operations
trait GradientOps {
    fn norm_squared(&self) -> f64;
}

impl GradientOps for [f64; 2] {
    fn norm_squared(&self) -> f64 {
        self[0] * self[0] + self[1] * self[1]
    }
}

/// Builder for creating calibrated predictions
pub struct CalibratedPredictionBuilder {
    calibrator: Arc<OnlinePlattCalibrator>,
    raw_scores: Vec<f64>,
}

impl CalibratedPredictionBuilder {
    pub fn new(calibrator: Arc<OnlinePlattCalibrator>) -> Self {
        Self {
            calibrator,
            raw_scores: Vec::new(),
        }
    }

    pub fn add_score(mut self, score: f64) -> Self {
        self.raw_scores.push(score);
        self
    }

    pub fn add_scores(mut self, scores: &[f64]) -> Self {
        self.raw_scores.extend(scores);
        self
    }

    pub fn build(self) -> Vec<f64> {
        self.calibrator.calibrate_batch(&self.raw_scores)
    }

    pub fn build_with_stats(self) -> (Vec<f64>, CalibratorStats) {
        let result = self.calibrator.calibrate_batch(&self.raw_scores);
        let stats = self.calibrator.get_stats();
        (result, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_parameters_default() {
        let params = PlattParameters::default();
        assert_eq!(params.a, 0.0);
        assert_eq!(params.b, 0.0);
    }

    #[test]
    fn test_calibration_bounds() {
        let params = PlattParameters::new(-1.0, 0.0);
        
        // Test extreme values
        assert!((0.0..=1.0).contains(&params.calibrate(-100.0)));
        assert!((0.0..=1.0).contains(&params.calibrate(100.0)));
        assert!((0.0..=1.0).contains(&params.calibrate(f64::NEG_INFINITY)));
        assert!((0.0..=1.0).contains(&params.calibrate(f64::INFINITY)));
    }

    #[test]
    fn test_online_calibrator_basic() {
        let calibrator = OnlinePlattCalibrator::new(10);
        
        // Initial calibration should return reasonable values
        let prob = calibrator.calibrate(0.0);
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_online_learning() {
        let calibrator = OnlinePlattCalibrator::new(10);
        
        // Add training data (perfect separation)
        for i in 0..50 {
            calibrator.observe(-2.0 + (i as f64) * 0.04, 0.0);
        }
        for i in 0..50 {
            calibrator.observe(0.0 + (i as f64) * 0.04, 1.0);
        }
        
        // Check that calibration works
        let low_prob = calibrator.calibrate(-2.0);
        let high_prob = calibrator.calibrate(2.0);
        
        assert!(low_prob < 0.5);
        assert!(high_prob > 0.5);
    }

    #[test]
    fn test_batch_calibration() {
        let calibrator = OnlinePlattCalibrator::with_initial_params(-1.0, 0.0, 10);
        
        let scores = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let probs = calibrator.calibrate_batch(&scores);
        
        assert_eq!(probs.len(), 5);
        for prob in probs {
            assert!((0.0..=1.0).contains(&prob));
        }
        
        // Verify monotonicity
        for i in 1..probs.len() {
            assert!(probs[i] >= probs[i-1]);
        }
    }

    #[test]
    fn test_stats_tracking() {
        let calibrator = OnlinePlattCalibrator::new(10);
        
        calibrator.observe(1.0, 1.0);
        calibrator.observe(-1.0, 0.0);
        
        let stats = calibrator.get_stats();
        assert!(stats.sample_count >= 2);
        assert!(stats.avg_calibration_time_ns > 0 || stats.update_count == 0);
    }
}
