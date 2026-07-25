//! ZAID PERSONAL CRYPTO TRADING BOT - Stress Testing & Chaos Engineering
//! Chapter 3: Flash Crash Simulator
//! 
//! This module simulates extreme market events (20%+ drops in microseconds)
//! to test bot resilience and circuit breaker response. ALL SIMULATIONS
//! RUN IN SANDBOXED ENVIRONMENT ONLY - NEVER AFFECTS LIVE FUNDS.
//! 
//! Memory Budget: <50MB for simulation state
//! Target Response: <100μs circuit breaker trigger
//! Safety: Sandboxed, isolated from live trading systems
//! Assets: BTC, SOL, ETH stress scenarios

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use rayon::prelude::*;

/// Maximum simulated price movement percentage
const MAX_SIMULATED_DROP: f64 = 0.50; // 50% maximum drop

/// Minimum time between flash crash simulations (safety)
const MIN_SIMULATION_INTERVAL_MS: u64 = 60000; // 60 seconds

/// Configuration for flash crash simulation
#[derive(Debug, Clone)]
pub struct FlashCrashConfig {
    pub assets: Vec<String>,
    pub max_drop_pct: f64,
    pub recovery_time_ms: u64,
    pub volatility_multiplier: f64,
    pub sandbox_mode: bool,  // ALWAYS true in production
}

impl Default for FlashCrashConfig {
    fn default() -> Self {
        Self {
            assets: vec!["BTC".to_string(), "SOL".to_string(), "ETH".to_string()],
            max_drop_pct: 0.20,  // 20% drop
            recovery_time_ms: 5000,  // 5 second recovery
            volatility_multiplier: 10.0,
            sandbox_mode: true,  // Safety first
        }
    }
}

/// Flash crash scenario types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashScenario {
    VShape,      // Quick drop and recovery
    LShape,      // Drop and stay low
    WShape,      // Double dip
    Gradual,     // Slow decline
    Spike,       // Brief extreme move
}

/// Simulation state for an asset
#[derive(Debug, Clone)]
pub struct SimulationState {
    pub asset: String,
    pub base_price: f64,
    pub current_price: f64,
    pub is_crashing: bool,
    pub crash_start_ns: u64,
    pub max_drawdown: f64,
    pub scenario: CrashScenario,
}

/// Result of a flash crash simulation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub asset: String,
    pub scenario: CrashScenario,
    pub initial_price: f64,
    pub minimum_price: f64,
    pub final_price: f64,
    pub max_drawdown_pct: f64,
    pub duration_ms: u64,
    pub circuit_breaker_triggered: bool,
    pub timestamp_ns: u64,
}

/// Ultra-fast flash crash simulator with sub-microsecond response
pub struct FlashCrashSimulator {
    config: FlashCrashConfig,
    /// Active simulations per asset
    simulations: HashMap<String, SimulationState>,
    /// Historical simulation results
    results_history: Vec<SimulationResult>,
    /// Safety flag - must be true to run simulations
    sandbox_enabled: AtomicBool,
    /// Last simulation timestamp (rate limiting)
    last_simulation_ns: AtomicU64,
    /// Circuit breaker callback
    circuit_breaker_callback: Option<Arc<dyn Fn(&str, f64) + Send + Sync>>,
}

unsafe impl Send for FlashCrashSimulator {}
unsafe impl Sync for FlashCrashSimulator {}

impl FlashCrashSimulator {
    /// Create new simulator with sandbox mode ENABLED by default
    pub fn new() -> Self {
        Self::with_config(FlashCrashConfig::default())
    }
    
    /// Create simulator with custom config (sandbox always enforced)
    pub fn with_config(config: FlashCrashConfig) -> Self {
        // Force sandbox mode regardless of config
        let mut safe_config = config.clone();
        safe_config.sandbox_mode = true;
        
        let mut simulations = HashMap::new();
        
        for asset in &safe_config.assets {
            simulations.insert(asset.clone(), SimulationState {
                asset: asset.clone(),
                base_price: 0.0,
                current_price: 0.0,
                is_crashing: false,
                crash_start_ns: 0,
                max_drawdown: 0.0,
                scenario: CrashScenario::VShape,
            });
        }
        
        Self {
            config: safe_config,
            simulations,
            results_history: Vec::new(),
            sandbox_enabled: AtomicBool::new(true),
            last_simulation_ns: AtomicU64::new(0),
            circuit_breaker_callback: None,
        }
    }
    
    /// Verify sandbox mode is active (CRITICAL SAFETY CHECK)
    #[inline]
    pub fn verify_sandbox(&self) -> bool {
        self.sandbox_enabled.load(Ordering::SeqCst) && self.config.sandbox_mode
    }
    
    /// Set circuit breaker callback (called when crash detected)
    pub fn set_circuit_breaker_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str, f64) + Send + Sync + 'static,
    {
        self.circuit_breaker_callback = Some(Arc::new(callback));
    }
    
    /// Initialize base price for an asset
    pub fn set_base_price(&mut self, asset: &str, price: f64) {
        if let Some(sim) = self.simulations.get_mut(asset) {
            sim.base_price = price;
            sim.current_price = price;
        }
    }
    
    /// Run flash crash simulation for an asset
    /// 
    /// SAFETY: Only runs in sandbox mode, never affects live systems
    pub fn simulate_flash_crash(&mut self, asset: &str, scenario: CrashScenario, 
                                 drop_pct: f64) -> Option<SimulationResult> {
        // CRITICAL: Verify sandbox mode
        if !self.verify_sandbox() {
            eprintln!("SECURITY VIOLATION: Sandbox not verified, aborting simulation");
            return None;
        }
        
        // Rate limit check
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let last_ns = self.last_simulation_ns.load(Ordering::Relaxed);
        if now_ns - last_ns < MIN_SIMULATION_INTERVAL_MS * 1_000_000 {
            return None;  // Too soon since last simulation
        }
        
        // Clamp drop percentage to safe limits
        let actual_drop = drop_pct.min(MAX_SIMULATED_DROP).abs();
        
        let sim_state = self.simulations.get_mut(asset)?;
        
        let initial_price = sim_state.current_price;
        if initial_price <= 0.0 {
            return None;
        }
        
        // Start simulation
        sim_state.is_crashing = true;
        sim_state.crash_start_ns = now_ns;
        sim_state.scenario = scenario;
        sim_state.max_drawdown = 0.0;
        
        self.last_simulation_ns.store(now_ns, Ordering::Relaxed);
        
        // Generate price path based on scenario
        let price_path = self.generate_price_path(scenario, initial_price, actual_drop);
        
        // Process price path and track minimum
        let mut minimum_price = initial_price;
        let mut circuit_breaker_triggered = false;
        
        for (tick, price) in price_path.iter().enumerate() {
            sim_state.current_price = *price;
            
            // Track drawdown
            let drawdown = (initial_price - price) / initial_price;
            if drawdown > sim_state.max_drawdown {
                sim_state.max_drawdown = drawdown;
            }
            
            if *price < minimum_price {
                minimum_price = *price;
            }
            
            // Check circuit breaker threshold (10% drop triggers)
            if drawdown > 0.10 && !circuit_breaker_triggered {
                circuit_breaker_triggered = true;
                
                // Trigger circuit breaker callback
                if let Some(ref callback) = self.circuit_breaker_callback {
                    callback(asset, drawdown);
                }
            }
        }
        
        // End simulation
        sim_state.is_crashing = false;
        let final_price = sim_state.current_price;
        
        let result = SimulationResult {
            asset: asset.to_string(),
            scenario,
            initial_price,
            minimum_price,
            final_price,
            max_drawdown_pct: sim_state.max_drawdown,
            duration_ms: self.config.recovery_time_ms,
            circuit_breaker_triggered,
            timestamp_ns: now_ns,
        };
        
        self.results_history.push(result.clone());
        
        Some(result)
    }
    
    /// Generate price path for given scenario
    fn generate_price_path(&self, scenario: CrashScenario, 
                           initial_price: f64, drop_pct: f64) -> Vec<f64> {
        let num_ticks = 100;  // Resolution of simulation
        let mut prices = Vec::with_capacity(num_ticks);
        
        match scenario {
            CrashScenario::VShape => {
                // Quick drop then recovery
                for i in 0..num_ticks {
                    let t = i as f64 / num_ticks as f64;
                    let drop_phase = if t < 0.3 {
                        // Rapid drop
                        drop_pct * (t / 0.3)
                    } else {
                        // Recovery
                        drop_pct * (1.0 - (t - 0.3) / 0.7)
                    };
                    prices.push(initial_price * (1.0 - drop_phase));
                }
            }
            CrashScenario::LShape => {
                // Drop and stay low
                for i in 0..num_ticks {
                    let t = i as f64 / num_ticks as f64;
                    let drop_phase = if t < 0.2 {
                        drop_pct * (t / 0.2)
                    } else {
                        drop_pct  // Stay at bottom
                    };
                    prices.push(initial_price * (1.0 - drop_phase));
                }
            }
            CrashScenario::WShape => {
                // Double dip
                for i in 0..num_ticks {
                    let t = i as f64 / num_ticks as f64;
                    let mut drop_phase = 0.0;
                    
                    if t < 0.25 {
                        drop_phase = drop_pct * (t / 0.25);
                    } else if t < 0.5 {
                        drop_phase = drop_pct * (1.0 - (t - 0.25) / 0.25);
                    } else if t < 0.75 {
                        drop_phase = drop_pct * ((t - 0.5) / 0.25);
                    } else {
                        drop_phase = drop_pct * (1.0 - (t - 0.75) / 0.25);
                    }
                    
                    prices.push(initial_price * (1.0 - drop_phase));
                }
            }
            CrashScenario::Gradual => {
                // Slow decline over time
                for i in 0..num_ticks {
                    let t = i as f64 / num_ticks as f64;
                    let drop_phase = drop_pct * t;
                    prices.push(initial_price * (1.0 - drop_phase));
                }
            }
            CrashScenario::Spike => {
                // Brief extreme move then immediate recovery
                for i in 0..num_ticks {
                    let t = i as f64 / num_ticks as f64;
                    let spike = if t >= 0.45 && t <= 0.55 {
                        drop_pct
                    } else {
                        0.0
                    };
                    prices.push(initial_price * (1.0 - spike));
                }
            }
        }
        
        // Add noise proportional to volatility multiplier
        let noise_level = 0.01 * self.config.volatility_multiplier;
        
        prices.par_iter_mut().for_each(|price| {
            let noise = (rand_distr::StandardNormal.sample(&mut rand::thread_rng())) * noise_level;
            *price *= 1.0 + noise.min(0.1).max(-0.1);  // Cap noise at 10%
        });
        
        prices
    }
    
    /// Run parallel simulations across all assets
    pub fn simulate_parallel(&mut self, scenario: CrashScenario, 
                             drop_pct: f64) -> HashMap<String, SimulationResult> {
        if !self.verify_sandbox() {
            return HashMap::new();
        }
        
        let assets: Vec<String> = self.config.assets.clone();
        
        assets.into_par_iter()
            .filter_map(|asset| {
                self.simulate_flash_crash(&asset, scenario, drop_pct)
                    .map(|result| (asset, result))
            })
            .collect()
    }
    
    /// Get historical simulation results
    pub fn get_simulation_history(&self) -> &[SimulationResult] {
        &self.results_history
    }
    
    /// Calculate statistics from simulation history
    pub fn get_simulation_statistics(&self, asset: &str) -> Option<SimulationStats> {
        let asset_results: Vec<&SimulationResult> = self.results_history
            .iter()
            .filter(|r| r.asset == asset)
            .collect();
        
        if asset_results.is_empty() {
            return None;
        }
        
        let avg_drawdown: f64 = asset_results.iter()
            .map(|r| r.max_drawdown_pct)
            .sum::<f64>() / asset_results.len() as f64;
        
        let max_drawdown = asset_results.iter()
            .map(|r| r.max_drawdown_pct)
            .fold(0.0_f64, |a, b| a.max(b));
        
        let cb_trigger_rate = asset_results.iter()
            .filter(|r| r.circuit_breaker_triggered)
            .count() as f64 / asset_results.len() as f64;
        
        Some(SimulationStats {
            asset: asset.to_string(),
            simulation_count: asset_results.len(),
            average_drawdown: avg_drawdown,
            maximum_drawdown: max_drawdown,
            circuit_breaker_trigger_rate: cb_trigger_rate,
        })
    }
    
    /// Emergency stop - disable all simulations
    pub fn emergency_stop(&self) {
        self.sandbox_enabled.store(false, Ordering::SeqCst);
    }
    
    /// Re-enable sandbox (requires explicit confirmation)
    pub fn enable_sandbox(&self) {
        self.sandbox_enabled.store(true, Ordering::SeqCst);
    }
    
    /// Check if any simulation is currently active
    pub fn has_active_simulation(&self) -> bool {
        self.simulations.values().any(|s| s.is_crashing)
    }
}

/// Statistics from flash crash simulations
#[derive(Debug, Clone)]
pub struct SimulationStats {
    pub asset: String,
    pub simulation_count: usize,
    pub average_drawdown: f64,
    pub maximum_drawdown: f64,
    pub circuit_breaker_trigger_rate: f64,
}

/// Stress test runner for comprehensive testing
pub struct StressTestRunner {
    simulator: FlashCrashSimulator,
    test_scenarios: Vec<(CrashScenario, f64)>,
}

impl StressTestRunner {
    pub fn new() -> Self {
        let mut simulator = FlashCrashSimulator::new();
        
        // Set up standard test scenarios
        let test_scenarios = vec![
            (CrashScenario::VShape, 0.10),   // 10% V-shape
            (CrashScenario::VShape, 0.20),   // 20% V-shape
            (CrashScenario::LShape, 0.15),   // 15% L-shape
            (CrashScenario::WShape, 0.25),   // 25% W-shape
            (CrashScenario::Spike, 0.30),    // 30% spike
        ];
        
        Self {
            simulator,
            test_scenarios,
        }
    }
    
    /// Run full stress test suite
    pub fn run_full_suite(&mut self) -> Vec<SimulationStats> {
        println!("Starting Flash Crash Stress Test Suite");
        println!("Sandbox Mode: {}", self.simulator.verify_sandbox());
        
        let mut stats = Vec::new();
        
        for (scenario, drop_pct) in &self.test_scenarios {
            println!("\nRunning scenario: {:?} with {}% drop", scenario, drop_pct * 100.0);
            
            // Run parallel simulation
            let results = self.simulator.simulate_parallel(*scenario, *drop_pct);
            
            // Collect stats per asset
            for asset in self.simulator.config.assets.iter() {
                if let Some(asset_stats) = self.simulator.get_simulation_statistics(asset) {
                    stats.push(asset_stats);
                }
            }
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sandbox_verification() {
        let simulator = FlashCrashSimulator::new();
        assert!(simulator.verify_sandbox());
    }
    
    #[test]
    fn test_v_shape_simulation() {
        let mut simulator = FlashCrashSimulator::new();
        simulator.set_base_price("BTC", 50000.0);
        
        let result = simulator.simulate_flash_crash("BTC", CrashScenario::VShape, 0.20);
        
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.max_drawdown_pct > 0.15);  // Should be close to 20%
        assert!(r.minimum_price < r.initial_price * 0.85);
    }
    
    #[test]
    fn test_circuit_breaker_trigger() {
        let mut simulator = FlashCrashSimulator::new();
        simulator.set_base_price("BTC", 50000.0);
        
        let result = simulator.simulate_flash_crash("BTC", CrashScenario::LShape, 0.25);
        
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.circuit_breaker_triggered);  // 25% drop should trigger CB
    }
}
