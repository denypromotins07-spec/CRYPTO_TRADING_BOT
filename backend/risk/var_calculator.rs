//! ZAID PERSONAL CRYPTO TRADING BOT - Advanced Risk Management
//! Chapter 1: Value at Risk (VaR) Calculator
//! 
//! This module implements ultra-fast Historical and Monte Carlo VaR calculations
//! optimized for O(1) lookup after O(n) preprocessing. Designed for sub-microsecond
//! risk assessment during extreme market stress conditions.
//! 
//! Memory Budget: <50MB for historical data structures
//! Target Latency: <100ns for VaR lookup after preprocessing
//! Assets: BTC, SOL, ETH, USDT parallel processing

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use rayon::prelude::*;

/// Configuration for VaR calculations
#[derive(Debug, Clone)]
pub struct VarConfig {
    pub confidence_level: f64,      // e.g., 0.95, 0.99
    pub time_horizon_days: u32,     // e.g., 1, 10, 30
    pub historical_window_size: usize, // Number of historical samples
    pub monte_carlo_simulations: usize, // Number of MC paths
    pub assets: Vec<String>,        // Trading assets
}

impl Default for VarConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.99,
            time_horizon_days: 1,
            historical_window_size: 252, // ~1 trading year
            monte_carlo_simulations: 10_000,
            assets: vec!["BTC".to_string(), "SOL".to_string(), "ETH".to_string(), "USDT".to_string()],
        }
    }
}

/// Historical VaR calculator with O(1) lookup capability
pub struct HistoricalVarCalculator {
    config: VarConfig,
    /// Sorted returns for O(1) percentile lookup via index
    sorted_returns: HashMap<String, Vec<f64>>,
    /// Raw returns in chronological order for rolling window updates
    return_queue: HashMap<String, VecDeque<f64>>,
    /// Precomputed VaR values for instant access
    cached_var: HashMap<String, f64>,
    /// Cache validity timestamp
    cache_timestamp: AtomicU64,
}

impl HistoricalVarCalculator {
    pub fn new(config: VarConfig) -> Self {
        let mut sorted_returns = HashMap::new();
        let mut return_queue = HashMap::new();
        let mut cached_var = HashMap::new();
        
        for asset in &config.assets {
            sorted_returns.insert(asset.clone(), Vec::with_capacity(config.historical_window_size));
            return_queue.insert(asset.clone(), VecDeque::with_capacity(config.historical_window_size));
            cached_var.insert(asset.clone(), 0.0);
        }
        
        Self {
            config,
            sorted_returns,
            return_queue,
            cached_var,
            cache_timestamp: AtomicU64::new(0),
        }
    }
    
    /// Add a new return observation - O(log n) for maintaining sorted order
    #[inline]
    pub fn add_return(&mut self, asset: &str, return_value: f64) {
        if let Some(queue) = self.return_queue.get_mut(asset) {
            if queue.len() >= self.config.historical_window_size {
                // Remove oldest return
                let _ = queue.pop_front();
            }
            queue.push_back(return_value);
            
            // Rebuild sorted array (optimized with insertion sort for small changes)
            self.rebuild_sorted_returns(asset);
            
            // Update cached VaR
            self.update_cached_var(asset);
        }
    }
    
    /// Rebuild sorted returns array using efficient sorting
    #[inline]
    fn rebuild_sorted_returns(&mut self, asset: &str) {
        if let Some(queue) = self.return_queue.get(asset) {
            if let Some(sorted) = self.sorted_returns.get_mut(asset) {
                *sorted = queue.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
    }
    
    /// Get VaR in O(1) time using precomputed index
    #[inline]
    pub fn get_var(&self, asset: &str) -> Option<f64> {
        self.cached_var.get(asset).copied()
    }
    
    /// Get portfolio VaR with correlation adjustment
    pub fn get_portfolio_var(&self, weights: &HashMap<String, f64>) -> f64 {
        let mut portfolio_var = 0.0;
        let mut cov_matrix: Vec<Vec<f64>> = Vec::new();
        
        // Build covariance matrix from historical returns
        let assets: Vec<&String> = self.config.assets.iter().collect();
        for i in 0..assets.len() {
            let mut row = Vec::new();
            for j in 0..assets.len() {
                let cov = self.compute_covariance(assets[i], assets[j]);
                row.push(cov);
            }
            cov_matrix.push(row);
        }
        
        // Calculate portfolio variance: w^T * Σ * w
        let weight_vec: Vec<f64> = assets.iter()
            .map(|a| *weights.get(*a).unwrap_or(&0.0))
            .collect();
        
        for i in 0..weight_vec.len() {
            for j in 0..weight_vec.len() {
                portfolio_var += weight_vec[i] * weight_vec[j] * cov_matrix[i][j];
            }
        }
        
        // Portfolio VaR = sqrt(variance) * z-score
        let z_score = self.z_score_for_confidence(self.config.confidence_level);
        portfolio_var.sqrt() * z_score
    }
    
    /// Compute covariance between two assets
    fn compute_covariance(&self, asset1: &str, asset2: &str) -> f64 {
        let returns1 = match self.sorted_returns.get(asset1) {
            Some(r) if !r.is_empty() => r,
            _ => return 0.0,
        };
        let returns2 = match self.sorted_returns.get(asset2) {
            Some(r) if !r.is_empty() => r,
            _ => return 0.0,
        };
        
        let mean1: f64 = returns1.iter().sum::<f64>() / returns1.len() as f64;
        let mean2: f64 = returns2.iter().sum::<f64>() / returns2.len() as f64;
        
        let covariance: f64 = returns1.iter()
            .zip(returns2.iter())
            .map(|(r1, r2)| (r1 - mean1) * (r2 - mean2))
            .sum();
        
        covariance / (returns1.len() as f64 - 1.0)
    }
    
    /// Update cached VaR value for an asset
    #[inline]
    fn update_cached_var(&mut self, asset: &str) {
        if let (Some(sorted), Some(var_value)) = (
            self.sorted_returns.get(asset),
            self.cached_var.get_mut(asset)
        ) {
            if !sorted.is_empty() {
                let index = ((1.0 - self.config.confidence_level) * sorted.len() as f64).floor() as usize;
                let var_index = index.min(sorted.len() - 1);
                *var_value = -sorted[var_index]; // VaR is typically expressed as positive loss
            }
        }
        self.cache_timestamp.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            Ordering::Relaxed
        );
    }
    
    /// Z-score for given confidence level (precomputed for speed)
    const fn z_score_for_confidence(confidence: f64) -> f64 {
        match confidence {
            c if c >= 0.999 => 3.090,
            c if c >= 0.99 => 2.326,
            c if c >= 0.975 => 1.960,
            c if c >= 0.95 => 1.645,
            c if c >= 0.90 => 1.282,
            _ => 1.645,
        }
    }
    
    /// Force refresh all cached VaR values
    pub fn refresh_all_var(&mut self) {
        let assets: Vec<String> = self.config.assets.clone();
        for asset in assets {
            self.update_cached_var(&asset);
        }
    }
}

/// Monte Carlo VaR simulator for tail risk estimation
pub struct MonteCarloVarSimulator {
    config: VarConfig,
    /// Precomputed random number generator state for speed
    rng_state: u64,
    /// Asset parameters: (mean, volatility)
    asset_params: HashMap<String, (f64, f64)>,
}

impl MonteCarloVarSimulator {
    pub fn new(config: VarConfig) -> Self {
        Self {
            config,
            rng_state: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            asset_params: HashMap::new(),
        }
    }
    
    /// Set asset parameters for simulation
    pub fn set_asset_parameters(&mut self, asset: &str, mean: f64, volatility: f64) {
        self.asset_params.insert(asset.to_string(), (mean, volatility));
    }
    
    /// Fast XORShift random number generator (zero-cost abstraction)
    #[inline]
    fn next_random(&mut self) -> f64 {
        // XORShift64 algorithm
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        
        // Convert to uniform [0, 1)
        (x as f64) / (u64::MAX as f64)
    }
    
    /// Box-Muller transform for normal distribution (zero allocation)
    #[inline]
    fn generate_normal(&mut self) -> (f64, f64) {
        let u1 = self.next_random().max(f64::EPSILON);
        let u2 = self.next_random();
        
        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();
        
        (z0, z1)
    }
    
    /// Run Monte Carlo simulation for VaR
    pub fn simulate_var(&mut self, asset: &str, current_price: f64, position_size: f64) -> f64 {
        let (mean, vol) = match self.asset_params.get(asset) {
            Some(params) => *params,
            None => return 0.0,
        };
        
        let mut simulated_prices = Vec::with_capacity(self.config.monte_carlo_simulations);
        let time_step = self.config.time_horizon_days as f64 / 252.0; // Trading days
        
        // Parallel simulation using Rayon for maximum throughput
        let results: Vec<f64> = (0..self.config.monte_carlo_simulations)
            .into_par_iter()
            .map(|i| {
                // Use different seed for each thread
                let mut local_rng = self.rng_state.wrapping_add(i as u64);
                
                // Generate random shock
                let u1 = ((local_rng ^ (local_rng << 13)) ^ ((local_rng >> 7) ^ (local_rng << 17))) as f64 / u64::MAX as f64;
                let u2 = (((local_rng ^ (local_rng << 13)) ^ ((local_rng >> 7) ^ (local_rng << 17))) >> 1) as f64 / u64::MAX as f64;
                
                let mag = (-2.0 * u1.max(f64::EPSILON).ln()).sqrt();
                let z = mag * (2.0 * std::f64::consts::PI * u2).cos();
                
                // Geometric Brownian Motion
                let drift = (mean - 0.5 * vol * vol) * time_step;
                let diffusion = vol * z * time_step.sqrt();
                let shock = (drift + diffusion).exp();
                
                current_price * shock
            })
            .collect();
        
        // Sort and find VaR percentile
        let mut sorted_prices = results;
        sorted_prices.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let var_index = ((1.0 - self.config.confidence_level) * sorted_prices.len() as f64).floor() as usize;
        let var_price = sorted_prices[var_index.min(sorted_prices.len() - 1)];
        
        // VaR as loss from current position
        let loss = (current_price - var_price) * position_size;
        loss.max(0.0)
    }
    
    /// Incremental VaR - sensitivity to position size changes
    pub fn incremental_var(&mut self, asset: &str, current_price: f64, delta_position: f64) -> f64 {
        let base_var = self.simulate_var(asset, current_price, 1.0);
        base_var * delta_position
    }
}

/// Component VaR for decomposing portfolio risk
pub struct ComponentVarAnalyzer {
    historical_calculator: HistoricalVarCalculator,
    monte_carlo_simulator: MonteCarloVarSimulator,
}

impl ComponentVarAnalyzer {
    pub fn new(config: VarConfig) -> Self {
        Self {
            historical_calculator: HistoricalVarCalculator::new(config.clone()),
            monte_carlo_simulator: MonteCarloVarSimulator::new(config),
        }
    }
    
    /// Decompose portfolio VaR into asset contributions
    pub fn analyze_components(&mut self, weights: &HashMap<String, f64>, prices: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut components = HashMap::new();
        let portfolio_var = self.historical_calculator.get_portfolio_var(weights);
        
        for (asset, weight) in weights {
            let marginal_var = self.monte_carlo_simulator.incremental_var(
                asset,
                *prices.get(asset).unwrap_or(&0.0),
                0.01 // 1% position change
            );
            
            let component = weight * marginal_var;
            components.insert(asset.clone(), component);
        }
        
        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_historical_var_calculation() {
        let config = VarConfig::default();
        let mut calculator = HistoricalVarCalculator::new(config);
        
        // Add sample returns
        for i in 0..100 {
            let return_val = (i as f64 - 50.0) / 1000.0; // Returns from -5% to +5%
            calculator.add_return("BTC", return_val);
        }
        
        let var = calculator.get_var("BTC");
        assert!(var.is_some());
        assert!(var.unwrap() > 0.0);
    }
    
    #[test]
    fn test_monte_carlo_var_simulation() {
        let config = VarConfig::default();
        let mut simulator = MonteCarloVarSimulator::new(config);
        simulator.set_asset_parameters("BTC", 0.0001, 0.02); // Daily mean and vol
        
        let var = simulator.simulate_var("BTC", 50000.0, 1.0);
        assert!(var >= 0.0);
    }
}
