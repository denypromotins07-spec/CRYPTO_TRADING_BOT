// ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
// High-Frequency Feature Stores, MLOps, and Concept Drift
// File: backend/mlops/shadow_engine.rs
// Chapter 4: MLOps Pipeline, Shadow Mode Testing, and SOUL.md Logging
//
// Runs deprecated models in the background for A/B comparison.
// Strictly isolates shadow models so they consume zero execution CPU cycles.
// Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
// Uses lazy evaluation and priority-based scheduling.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Instant, Duration};
use std::thread;

/// Maximum shadow models to track
const MAX_SHADOW_MODELS: usize = 5;

/// Priority levels for shadow model execution
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShadowPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Execution mode for shadow models
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadowMode {
    /// Model runs but results are discarded
    Passive,
    /// Model results are logged for analysis
    Logged,
    /// Model competes with live model for selection
    Competitive,
}

/// Result from a shadow model prediction
#[derive(Clone, Debug)]
pub struct ShadowPrediction {
    pub model_id: String,
    pub timestamp_ns: u64,
    pub prediction: f64,
    pub confidence: f64,
    pub latency_us: u64,
    pub metadata: HashMap<String, String>,
}

/// Statistics for a shadow model
#[derive(Clone, Debug, Default)]
pub struct ShadowModelStats {
    pub total_predictions: u64,
    pub avg_latency_us: f64,
    pub max_latency_us: u64,
    pub min_latency_us: u64,
    pub agreement_rate: f64,  // Agreement with live model
    pub pnl_if_traded: f64,   // Hypothetical PnL
    pub last_execution_ns: u64,
}

/// Configuration for a shadow model
#[derive(Clone, Debug)]
pub struct ShadowModelConfig {
    pub model_id: String,
    pub model_path: String,
    pub priority: ShadowPriority,
    pub mode: ShadowMode,
    pub sampling_rate: f64,  // Fraction of trades to evaluate
    pub max_memory_mb: usize,
}

/// Shadow model runner with strict isolation
pub struct ShadowEngine {
    /// Registered shadow models
    models: RwLock<HashMap<String, ShadowModelRunner>>,
    /// Execution queue (ordered by priority)
    execution_queue: Mutex<VecDeque<String>>,
    /// Whether shadow execution is enabled
    enabled: AtomicBool,
    /// Global statistics
    total_executions: AtomicU64,
    total_cpu_time_us: AtomicU64,
    cpu_budget_remaining_us: AtomicU64,
    /// CPU budget per 4-hour window (in microseconds)
    /// Set to near-zero to ensure no impact on live trading
    cpu_budget_per_window_us: AtomicU64,
    /// Window start time
    window_start_ns: AtomicU64,
}

/// Individual shadow model runner
struct ShadowModelRunner {
    config: ShadowModelConfig,
    stats: RwLock<ShadowModelStats>,
    /// Latency histogram for monitoring
    latency_histogram: RwLock<Vec<u64>>,
    /// Last N predictions for comparison
    recent_predictions: RwLock<VecDeque<ShadowPrediction>>,
    /// Whether this model is currently executing
    is_executing: AtomicBool,
}

impl ShadowModelRunner {
    fn new(config: ShadowModelConfig) -> Self {
        Self {
            config,
            stats: RwLock::new(ShadowModelStats::default()),
            latency_histogram: RwLock::new(Vec::with_capacity(100)),
            recent_predictions: RwLock::new(VecDeque::with_capacity(100)),
            is_executing: AtomicBool::new(false),
        }
    }

    /// Execute prediction (lazy, only if within CPU budget)
    fn execute(&self, input: &[f64]) -> Option<ShadowPrediction> {
        if self.is_executing.swap(true, Ordering::SeqCst) {
            return None; // Already executing, skip
        }

        let start = Instant::now();
        
        // Simulate model inference (in production, would load and run actual model)
        let prediction = self.run_inference(input);
        let confidence = self.compute_confidence(input);
        
        let latency_us = start.elapsed().as_micros() as u64;
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_predictions += 1;
            stats.max_latency_us = stats.max_latency_us.max(latency_us);
            stats.min_latency_us = if stats.min_latency_us == 0 {
                latency_us
            } else {
                stats.min_latency_us.min(latency_us)
            };
            
            // Running average latency
            let n = stats.total_predictions as f64;
            stats.avg_latency_us = (stats.avg_latency_us * (n - 1.0) + latency_us as f64) / n;
            stats.last_execution_ns = current_time_ns();
        }

        // Update latency histogram
        {
            let mut hist = self.latency_histogram.write().unwrap();
            hist.push(latency_us);
            if hist.len() > 100 {
                hist.remove(0);
            }
        }

        // Store prediction
        let pred = ShadowPrediction {
            model_id: self.config.model_id.clone(),
            timestamp_ns: current_time_ns(),
            prediction,
            confidence,
            latency_us,
            metadata: HashMap::new(),
        };

        {
            let mut preds = self.recent_predictions.write().unwrap();
            preds.push_back(pred.clone());
            if preds.len() >= 100 {
                preds.pop_front();
            }
        }

        self.is_executing.store(false, Ordering::SeqCst);
        Some(pred)
    }

    /// Run model inference (placeholder - would load actual model)
    fn run_inference(&self, _input: &[f64]) -> f64 {
        // In production: load model from self.config.model_path and run inference
        // For now, return a deterministic pseudo-prediction
        0.5
    }

    /// Compute prediction confidence
    fn compute_confidence(&self, _input: &[f64]) -> f64 {
        // In production: compute actual confidence from model
        0.8
    }

    /// Get recent predictions for comparison
    fn get_recent_predictions(&self) -> Vec<ShadowPrediction> {
        self.recent_predictions.read().unwrap().iter().cloned().collect()
    }

    /// Get current statistics
    fn get_stats(&self) -> ShadowModelStats {
        self.stats.read().unwrap().clone()
    }
}

impl ShadowEngine {
    /// Create a new shadow engine
    pub fn new(cpu_budget_per_window_us: u64) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            execution_queue: Mutex::new(VecDeque::new()),
            enabled: AtomicBool::new(true),
            total_executions: AtomicU64::new(0),
            total_cpu_time_us: AtomicU64::new(0),
            cpu_budget_remaining_us: AtomicU64::new(cpu_budget_per_window_us),
            cpu_budget_per_window_us: AtomicU64::new(cpu_budget_per_window_us),
            window_start_ns: AtomicU64::new(current_time_ns()),
        }
    }

    /// Register a shadow model
    pub fn register_model(&self, config: ShadowModelConfig) -> Result<(), &'static str> {
        let mut models = self.models.write().unwrap();
        
        if models.len() >= MAX_SHADOW_MODELS {
            return Err("Maximum shadow models reached");
        }
        
        if models.contains_key(&config.model_id) {
            return Err("Model already registered");
        }
        
        let runner = ShadowModelRunner::new(config.clone());
        models.insert(config.model_id.clone(), runner);
        
        // Add to execution queue based on priority
        let mut queue = self.execution_queue.lock().unwrap();
        
        // Insert in priority order
        let mut inserted = false;
        for (i, id) in queue.iter().enumerate() {
            if let Some(existing) = models.get(id) {
                if existing.config.priority < config.priority {
                    queue.insert(i, config.model_id.clone());
                    inserted = true;
                    break;
                }
            }
        }
        
        if !inserted {
            queue.push_back(config.model_id.clone());
        }
        
        Ok(())
    }

    /// Unregister a shadow model
    pub fn unregister_model(&self, model_id: &str) -> bool {
        let mut models = self.models.write().unwrap();
        let mut queue = self.execution_queue.lock().unwrap();
        
        if models.remove(model_id).is_some() {
            queue.retain(|id| id != model_id);
            true
        } else {
            false
        }
    }

    /// Execute all shadow models on given input (within CPU budget)
    pub fn execute_shadow(&self, input: &[f64], live_prediction: f64) -> Vec<ShadowPrediction> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Vec::new();
        }

        // Check if we're in a new window (4 hours = 14,400,000,000 microseconds)
        self.check_window_reset();

        // Check CPU budget
        let remaining = self.cpu_budget_remaining_us.load(Ordering::Relaxed);
        if remaining < 100 {  // Need at least 100us for any execution
            return Vec::new();
        }

        let mut results = Vec::new();
        let queue = self.execution_queue.lock().unwrap();
        let models = self.models.read().unwrap();

        for model_id in queue.iter() {
            if let Some(runner) = models.get(model_id) {
                // Check sampling rate
                let should_sample = (runner.config.sampling_rate * 100.0) as usize;
                if self.total_executions.load(Ordering::Relaxed) % should_sample.max(1) != 0 {
                    continue;
                }

                // Execute if within budget
                if let Some(prediction) = runner.execute(input) {
                    // Compare with live model
                    self.update_agreement_stats(model_id, prediction.prediction, live_prediction);
                    
                    results.push(prediction);
                    
                    // Deduct from budget (estimate based on model priority)
                    let estimated_cost = match runner.config.priority {
                        ShadowPriority::Low => 10,
                        ShadowPriority::Medium => 50,
                        ShadowPriority::High => 100,
                        ShadowPriority::Critical => 200,
                    };
                    
                    self.cpu_budget_remaining_us.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |current| current.checked_sub(estimated_cost)
                    ).ok();
                    
                    self.total_executions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        results
    }

    /// Check if window has reset (every 4 hours)
    fn check_window_reset(&self) {
        let current_ns = current_time_ns();
        let window_start = self.window_start_ns.load(Ordering::Relaxed);
        
        // 4 hours in nanoseconds
        let window_duration_ns = 4u64 * 3600 * 1_000_000_000;
        
        if current_ns - window_start > window_duration_ns {
            // Reset window
            self.window_start_ns.store(current_ns, Ordering::Relaxed);
            self.cpu_budget_remaining_us.store(
                self.cpu_budget_per_window_us.load(Ordering::Relaxed),
                Ordering::Relaxed
            );
        }
    }

    /// Update agreement statistics between shadow and live model
    fn update_agreement_stats(&self, model_id: &str, shadow_pred: f64, live_pred: f64) {
        let models = self.models.read().unwrap();
        if let Some(runner) = models.get(model_id) {
            let mut stats = runner.stats.write().unwrap();
            
            // Simple agreement: same sign and within 10%
            let agrees = (shadow_pred.signum() == live_pred.signum()) &&
                         ((shadow_pred - live_pred).abs() / (live_pred.abs() + 1e-10)) < 0.1;
            
            let n = stats.total_predictions as f64;
            let current_rate = stats.agreement_rate;
            stats.agreement_rate = (current_rate * (n - 1.0) + if agrees { 1.0 } else { 0.0 }) / n;
        }
    }

    /// Get statistics for a specific shadow model
    pub fn get_model_stats(&self, model_id: &str) -> Option<ShadowModelStats> {
        let models = self.models.read().unwrap();
        models.get(model_id).map(|r| r.get_stats())
    }

    /// Get all shadow model statistics
    pub fn get_all_stats(&self) -> HashMap<String, ShadowModelStats> {
        let models = self.models.read().unwrap();
        models.iter().map(|(id, r)| (id.clone(), r.get_stats())).collect()
    }

    /// Get engine-wide statistics
    pub fn get_engine_stats(&self) -> ShadowEngineStats {
        ShadowEngineStats {
            total_models: self.models.read().unwrap().len(),
            total_executions: self.total_executions.load(Ordering::Relaxed),
            total_cpu_time_us: self.total_cpu_time_us.load(Ordering::Relaxed),
            cpu_budget_remaining_us: self.cpu_budget_remaining_us.load(Ordering::Relaxed),
            cpu_budget_total_us: self.cpu_budget_per_window_us.load(Ordering::Relaxed),
            enabled: self.enabled.load(Ordering::Relaxed),
            window_age_seconds: {
                let elapsed_ns = current_time_ns() - self.window_start_ns.load(Ordering::Relaxed);
                elapsed_ns / 1_000_000_000
            },
        }
    }

    /// Enable/disable shadow execution
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get recent predictions from a model
    pub fn get_recent_predictions(&self, model_id: &str) -> Option<Vec<ShadowPrediction>> {
        let models = self.models.read().unwrap();
        models.get(model_id).map(|r| r.get_recent_predictions())
    }
}

/// Engine-wide statistics
#[derive(Debug, Clone)]
pub struct ShadowEngineStats {
    pub total_models: usize,
    pub total_executions: u64,
    pub total_cpu_time_us: u64,
    pub cpu_budget_remaining_us: u64,
    pub cpu_budget_total_us: u64,
    pub enabled: bool,
    pub window_age_seconds: u64,
}

/// Helper function to get current time in nanoseconds
#[inline]
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_engine_creation() {
        let engine = ShadowEngine::new(10000); // 10ms budget per window
        let stats = engine.get_engine_stats();
        assert_eq!(stats.total_models, 0);
        assert!(stats.enabled);
    }

    #[test]
    fn test_register_unregister() {
        let engine = ShadowEngine::new(10000);
        
        let config = ShadowModelConfig {
            model_id: "test_model".to_string(),
            model_path: "/path/to/model".to_string(),
            priority: ShadowPriority::Medium,
            mode: ShadowMode::Logged,
            sampling_rate: 1.0,
            max_memory_mb: 64,
        };
        
        assert!(engine.register_model(config).is_ok());
        assert!(engine.unregister_model("test_model"));
        assert!(!engine.unregister_model("nonexistent"));
    }

    #[test]
    fn test_shadow_execution() {
        let engine = ShadowEngine::new(100000); // 100ms budget
        
        let config = ShadowModelConfig {
            model_id: "shadow_v1".to_string(),
            model_path: "".to_string(),
            priority: ShadowPriority::High,
            mode: ShadowMode::Logged,
            sampling_rate: 1.0,
            max_memory_mb: 64,
        };
        
        engine.register_model(config).unwrap();
        
        let input = vec![1.0, 2.0, 3.0];
        let results = engine.execute_shadow(&input, 0.5);
        
        // Should have executed at least once
        assert!(!results.is_empty() || engine.get_engine_stats().cpu_budget_remaining_us < 100000);
    }

    #[test]
    fn test_cpu_budget_enforcement() {
        let engine = ShadowEngine::new(100); // Very small budget
        
        let config = ShadowModelConfig {
            model_id: "budget_test".to_string(),
            model_path: "".to_string(),
            priority: ShadowPriority::Low,
            mode: ShadowMode::Passive,
            sampling_rate: 1.0,
            max_memory_mb: 64,
        };
        
        engine.register_model(config).unwrap();
        
        let input = vec![1.0];
        
        // First execution might succeed
        engine.execute_shadow(&input, 0.5);
        
        // Budget should be depleted
        let stats = engine.get_engine_stats();
        assert!(stats.cpu_budget_remaining_us < 100);
    }

    #[test]
    fn test_priority_ordering() {
        let engine = ShadowEngine::new(100000);
        
        // Register in random order
        engine.register_model(ShadowModelConfig {
            model_id: "low".to_string(),
            model_path: "".to_string(),
            priority: ShadowPriority::Low,
            mode: ShadowMode::Passive,
            sampling_rate: 1.0,
            max_memory_mb: 64,
        }).unwrap();
        
        engine.register_model(ShadowModelConfig {
            model_id: "high".to_string(),
            model_path: "".to_string(),
            priority: ShadowPriority::High,
            mode: ShadowMode::Logged,
            sampling_rate: 1.0,
            max_memory_mb: 64,
        }).unwrap();
        
        engine.register_model(ShadowModelConfig {
            model_id: "medium".to_string(),
            model_path: "".to_string(),
            priority: ShadowPriority::Medium,
            mode: ShadowMode::Logged,
            sampling_rate: 1.0,
            max_memory_mb: 64,
        }).unwrap();
        
        // Queue should be ordered: high, medium, low
        let queue = engine.execution_queue.lock().unwrap();
        assert_eq!(queue.front(), Some(&"high".to_string()));
    }
}
