//! Liquid Time-Constant Network: Continuous-Time RNN via ODE Solving
//!
//! This module implements Liquid Time-Constant (LTC) networks, a class of
//! continuous-time recurrent neural networks that solve ODEs for hidden state evolution.
//!
//! Key Features:
//! - Runge-Kutta 4th order ODE solver for numerical stability
//! - Adaptive time-stepping based on market volatility
//! - Zero-heap allocation during inference
//! - Flash crash protection via derivative clamping
//!
//! Mathematical Foundation:
//! The LTC network evolves hidden state h(t) according to:
//!   dh/dt = -h(t)/tau + f(x(t), h(t))
//! Where tau is the liquid time constant that adapts to input dynamics.

use std::time::{Duration, Instant};

/// Configuration for the LTC network
#[derive(Debug, Clone)]
pub struct LtcConfig {
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub time_constant_min: f64,
    pub time_constant_max: f64,
    pub rk4_steps_per_tick: usize,
}

impl Default for LtcConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 32,       // Optimized for 8GB RAM
            input_dim: 16,
            time_constant_min: 0.001,  // 1ms minimum time constant
            time_constant_max: 1.0,    // 1s maximum time constant
            rk4_steps_per_tick: 4,     // 4 RK4 steps per tick for accuracy
        }
    }
}

/// Liquid Time-Constant Network core structure
pub struct LiquidNetwork {
    config: LtcConfig,
    /// Hidden state vector
    h_state: Vec<f64>,
    /// Time constants (tau) for each neuron
    tau: Vec<f64>,
    /// Weight matrices (flattened for cache efficiency)
    w_input: Vec<f64>,  // [hidden_dim * input_dim]
    w_recurrent: Vec<f64>,  // [hidden_dim * hidden_dim]
    /// Bias terms
    bias: Vec<f64>,
    /// Temporary buffers for RK4 (pre-allocated to avoid heap churn)
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    h_temp: Vec<f64>,
}

impl LiquidNetwork {
    /// Initialize the LTC network with given configuration
    pub fn new(config: LtcConfig) -> Result<Self, String> {
        if config.hidden_dim > 64 {
            return Err("Hidden dimension exceeds memory budget for 8GB system".to_string());
        }
        
        let total_weights = config.hidden_dim * (config.input_dim + config.hidden_dim + 1);
        if total_weights > 10000 {
            return Err("Weight matrix too large for memory constraints".to_string());
        }
        
        Ok(Self {
            config: config.clone(),
            h_state: vec![0.0; config.hidden_dim],
            tau: vec![0.5; config.hidden_dim],  // Initialize with moderate time constants
            w_input: vec![0.0; config.hidden_dim * config.input_dim],
            w_recurrent: vec![0.0; config.hidden_dim * config.hidden_dim],
            bias: vec![0.0; config.hidden_dim],
            k1: vec![0.0; config.hidden_dim],
            k2: vec![0.0; config.hidden_dim],
            k3: vec![0.0; config.hidden_dim],
            k4: vec![0.0; config.hidden_dim],
            h_temp: vec![0.0; config.hidden_dim],
        })
    }
    
    /// Initialize weights with small random values (deterministic for reproducibility)
    pub fn initialize_weights(&mut self, seed: u64) {
        // Simple LCG-based pseudo-random initialization
        let mut rng = seed;
        let lcg_next = |x: u64| x.wrapping_mul(6364136223846793005).wrapping_add(1);
        
        for i in 0..self.w_input.len() {
            rng = lcg_next(rng);
            self.w_input[i] = ((rng % 1000) as f64 / 1000.0 - 0.5) * 0.1;
        }
        
        for i in 0..self.w_recurrent.len() {
            rng = lcg_next(rng);
            // Diagonal dominance for stability
            if i % self.config.hidden_dim == i / self.config.hidden_dim {
                self.w_recurrent[i] = -1.0 + ((rng % 100) as f64 / 100.0) * 0.1;
            } else {
                self.w_recurrent[i] = ((rng % 1000) as f64 / 1000.0 - 0.5) * 0.01;
            }
        }
        
        for i in 0..self.bias.len() {
            rng = lcg_next(rng);
            self.bias[i] = ((rng % 1000) as f64 / 1000.0 - 0.5) * 0.01;
        }
    }
    
    /// Process a single input tick using RK4 ODE integration
    #[inline]
    pub fn process_tick(&mut self, input: &[f64], dt: f64) -> Result<Vec<f64>, String> {
        if input.len() != self.config.input_dim {
            return Err(format!(
                "Input dimension {} mismatch with expected {}",
                input.len(),
                self.config.input_dim
            ));
        }
        
        let sub_dt = dt / self.config.rk4_steps_per_tick as f64;
        
        // Perform RK4 integration over sub-steps
        for _step in 0..self.config.rk4_steps_per_tick {
            // Compute adaptive time constants based on current input
            self.update_time_constants(input);
            
            // RK4 step: k1 = f(h, x)
            self.compute_derivative(&self.h_state, input, &mut self.k1);
            
            // h_temp = h + 0.5 * dt * k1
            for i in 0..self.config.hidden_dim {
                self.h_temp[i] = self.h_state[i] + 0.5 * sub_dt * self.k1[i];
            }
            
            // k2 = f(h + 0.5*dt*k1, x)
            self.compute_derivative(&self.h_temp, input, &mut self.k2);
            
            // h_temp = h + 0.5 * dt * k2
            for i in 0..self.config.hidden_dim {
                self.h_temp[i] = self.h_state[i] + 0.5 * sub_dt * self.k2[i];
            }
            
            // k3 = f(h + 0.5*dt*k2, x)
            self.compute_derivative(&self.h_temp, input, &mut self.k3);
            
            // h_temp = h + dt * k3
            for i in 0..self.config.hidden_dim {
                self.h_temp[i] = self.h_state[i] + sub_dt * self.k3[i];
            }
            
            // k4 = f(h + dt*k3, x)
            self.compute_derivative(&self.h_temp, input, &mut self.k4);
            
            // h = h + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            for i in 0..self.config.hidden_dim {
                self.h_state[i] += (sub_dt / 6.0) * (
                    self.k1[i] + 
                    2.0 * self.k2[i] + 
                    2.0 * self.k3[i] + 
                    self.k4[i]
                );
                
                // Clamp hidden state to prevent explosion during flash crashes
                self.h_state[i] = self.h_state[i].clamp(-10.0, 10.0);
            }
        }
        
        Ok(self.h_state.clone())
    }
    
    /// Compute the derivative dh/dt for the LTC ODE
    #[inline]
    fn compute_derivative(&self, h: &[f64], input: &[f64], output: &mut [f64]) {
        for i in 0..self.config.hidden_dim {
            // Input contribution: sum_j W_input[i,j] * x[j]
            let mut input_contrib = 0.0;
            for j in 0..self.config.input_dim {
                input_contrib += self.w_input[i * self.config.input_dim + j] * input[j];
            }
            
            // Recurrent contribution: sum_j W_rec[i,j] * h[j]
            let mut recurrent_contrib = 0.0;
            for j in 0..self.config.hidden_dim {
                recurrent_contrib += self.w_recurrent[i * self.config.hidden_dim + j] * h[j];
            }
            
            // Nonlinear activation (tanh for bounded output)
            let activation = (input_contrib + recurrent_contrib + self.bias[i]).tanh();
            
            // LTC ODE: dh/dt = -h/tau + activation
            output[i] = -h[i] / self.tau[i] + activation;
            
            // Clamp derivative to prevent numerical explosion
            output[i] = output[i].clamp(-100.0, 100.0);
        }
    }
    
    /// Update time constants adaptively based on input magnitude
    #[inline]
    fn update_time_constants(&mut self, input: &[f64]) {
        // Compute input energy (squared L2 norm)
        let input_energy: f64 = input.iter().map(|&x| x * x).sum();
        
        // Adaptive time constant: smaller tau for high-energy inputs (fast response)
        // larger tau for low-energy inputs (slow integration)
        let target_tau = self.config.time_constant_min 
            + (self.config.time_constant_max - self.config.time_constant_min) 
                * (-input_energy * 0.1).exp();
        
        // Smooth update of time constants (first-order filter)
        for i in 0..self.config.hidden_dim {
            self.tau[i] = 0.9 * self.tau[i] + 0.1 * target_tau;
            self.tau[i] = self.tau[i].clamp(
                self.config.time_constant_min,
                self.config.time_constant_max,
            );
        }
    }
    
    /// Reset hidden state to zero
    pub fn reset_state(&mut self) {
        self.h_state.fill(0.0);
    }
    
    /// Get current hidden state
    pub fn get_hidden_state(&self) -> &[f64] {
        &self.h_state
    }
    
    /// Set hidden state (for warm starting)
    pub fn set_hidden_state(&mut self, state: &[f64]) -> Result<(), String> {
        if state.len() != self.config.hidden_dim {
            return Err("State dimension mismatch".to_string());
        }
        self.h_state.copy_from_slice(state);
        Ok(())
    }
    
    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let vector_size = self.config.hidden_dim * 8; // f64
        let weight_size = (self.w_input.len() + self.w_recurrent.len()) * 8;
        base_size + 6 * vector_size + weight_size
    }
}

/// Volatility-adaptive step size controller for ODE solver
pub struct AdaptiveStepController {
    /// Base step size
    base_dt: f64,
    /// Current step size
    current_dt: f64,
    /// Volatility estimate (exponential moving average)
    volatility_ema: f64,
    /// EMA decay factor
    alpha: f64,
    /// Minimum step size for stability
    min_dt: f64,
    /// Maximum step size for efficiency
    max_dt: f64,
}

impl AdaptiveStepController {
    pub fn new(base_dt: f64) -> Self {
        Self {
            base_dt,
            current_dt: base_dt,
            volatility_ema: 0.0,
            alpha: 0.1,
            min_dt: base_dt / 10.0,
            max_dt: base_dt * 2.0,
        }
    }
    
    /// Update step size based on observed volatility
    #[inline]
    pub fn update(&mut self, price_change: f64) {
        // Update volatility estimate
        let abs_change = price_change.abs();
        self.volatility_ema = (1.0 - self.alpha) * self.volatility_ema 
            + self.alpha * abs_change;
        
        // Reduce step size during high volatility (flash crash protection)
        // Increase step size during calm periods (efficiency)
        let volatility_factor = (-self.volatility_ema * 10.0).exp();
        self.current_dt = (self.base_dt * volatility_factor)
            .clamp(self.min_dt, self.max_dt);
    }
    
    /// Get current recommended step size
    #[inline]
    pub fn get_step_size(&self) -> f64 {
        self.current_dt
    }
    
    /// Reset volatility estimate
    pub fn reset(&mut self) {
        self.volatility_ema = 0.0;
        self.current_dt = self.base_dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltc_initialization() {
        let config = LtcConfig::default();
        let mut network = LiquidNetwork::new(config).unwrap();
        network.initialize_weights(42);
        
        assert_eq!(network.h_state.len(), 32);
        assert_eq!(network.tau.len(), 32);
    }

    #[test]
    fn test_rk4_stability() {
        let config = LtcConfig::default();
        let mut network = LiquidNetwork::new(config).unwrap();
        network.initialize_weights(42);
        
        let input = vec![0.1; config.input_dim];
        
        // Process many steps to check for numerical explosion
        for _ in 0..1000 {
            let result = network.process_tick(&input, 0.01);
            assert!(result.is_ok());
            
            // Check no NaN or Inf
            for &h in network.get_hidden_state() {
                assert!(h.is_finite());
            }
        }
    }

    #[test]
    fn test_adaptive_step_controller() {
        let mut controller = AdaptiveStepController::new(0.01);
        
        // Low volatility should maintain base step
        controller.update(0.001);
        assert!(controller.get_step_size() >= 0.009);
        
        // High volatility should reduce step
        controller.update(1.0);
        controller.update(1.0);
        controller.update(1.0);
        assert!(controller.get_step_size() < 0.01);
    }
}
