//! Counterfactual Simulation for Macroeconomic Events
//!
//! This module runs "what-if" scenarios for macroeconomic events
//! in the ZAID PERSONAL CRYPTO TRADING BOT. Enables hedging against
//! unseen black swan events through counterfactual reasoning.
//!
//! Features:
//! - Parallel counterfactual simulation across multiple scenarios
//! - Memory-efficient state representation
//! - Black swan event modeling with fat-tailed distributions
//! - Real-time scenario evaluation for hedging decisions

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Represents a macroeconomic event scenario
#[derive(Debug, Clone)]
pub struct MacroScenario {
    pub name: String,
    pub event_type: MacroEventType,
    pub magnitude: f64,      // Standard deviations from mean
    pub duration_steps: u32,  // How many time steps the effect lasts
    pub affected_variables: HashMap<String, f64>,
}

/// Types of macroeconomic events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroEventType {
    FedRateChange,
    InflationSurprise,
    EmploymentReport,
    GeopoliticalShock,
    RegulatoryAnnouncement,
    BankFailure,
    CurrencyCrisis,
    BlackSwan,
}

/// Result of a counterfactual simulation
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    pub scenario_name: String,
    pub baseline_outcome: f64,
    pub counterfactual_outcome: f64,
    pub causal_effect: f64,
    pub confidence_interval: (f64, f64),
    pub probability_of_extreme: f64,
}

/// State representation for counterfactual worlds
#[derive(Debug, Clone)]
pub struct WorldState {
    variables: HashMap<String, f64>,
    timestamp: u64,
    is_counterfactual: bool,
}

impl WorldState {
    pub fn new(variables: HashMap<String, f64>) -> Self {
        Self {
            variables,
            timestamp: 0,
            is_counterfactual: false,
        }
    }

    pub fn get(&self, var: &str) -> Option<f64> {
        self.variables.get(var).copied()
    }

    pub fn set(&mut self, var: &str, value: f64) {
        self.variables.insert(var.to_string(), value);
    }

    pub fn mark_counterfactual(&mut self) {
        self.is_counterfactual = true;
    }
}

/// Structural equation for variable dynamics
pub type StructuralFn = Arc<dyn Fn(&WorldState, f64) -> f64 + Send + Sync>;

/// Counterfactual simulation engine
pub struct CounterfactualSimulator {
    /// Base structural equations for each variable
    structural_equations: HashMap<String, StructuralFn>,
    /// Causal ordering for evaluation
    causal_order: Vec<String>,
    /// Exogenous noise distributions
    noise_params: HashMap<String, (f64, f64)>, // mean, std
    /// Current observed state
    observed_state: WorldState,
    /// Historical states for reference
    history: Vec<WorldState>,
}

impl CounterfactualSimulator {
    pub fn new(causal_order: Vec<String>) -> Self {
        Self {
            structural_equations: HashMap::new(),
            causal_order,
            noise_params: HashMap::new(),
            observed_state: WorldState::new(HashMap::new()),
            history: Vec::with_capacity(1000),
        }
    }

    /// Add a structural equation for a variable
    pub fn add_equation<F>(&mut self, var: &str, equation: F, noise_std: f64)
    where
        F: Fn(&WorldState, f64) -> f64 + Send + Sync + 'static,
    {
        self.structural_equations
            .insert(var.to_string(), Arc::new(equation));
        self.noise_params.insert(var.to_string(), (0.0, noise_std));
    }

    /// Set the current observed state
    pub fn set_observed_state(&mut self, state: HashMap<String, f64>) {
        self.observed_state = WorldState::new(state);
    }

    /// Record current state to history
    pub fn record_state(&mut self) {
        self.history.push(self.observed_state.clone());
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    /// Run a single counterfactual simulation
    pub fn simulate_counterfactual(
        &self,
        scenario: &MacroScenario,
        n_steps: u32,
    ) -> CounterfactualResult {
        let start = Instant::now();

        // Create counterfactual world starting from observed state
        let mut cf_world = self.observed_state.clone();
        cf_world.mark_counterfactual();

        // Apply initial intervention
        for (var, shock) in &scenario.affected_variables {
            let base_value = cf_world.get(var).unwrap_or(0.0);
            cf_world.set(var, base_value + shock * scenario.magnitude);
        }

        // Track outcomes
        let mut baseline_values = Vec::with_capacity(n_steps as usize);
        let mut counterfactual_values = Vec::with_capacity(n_steps as usize);

        // Simulate forward
        for step in 0..n_steps {
            // Evolve counterfactual world
            self.evolve_state(&mut cf_world, step as f64);

            // Store outcome (e.g., portfolio value)
            if let Some(outcome) = cf_world.get("portfolio_return") {
                counterfactual_values.push(outcome);
            }

            // Decay intervention effect over time
            let decay_factor = 1.0 - (step as f64) / (scenario.duration_steps as f64);
            if decay_factor > 0.0 {
                for (var, shock) in &scenario.affected_variables {
                    if let Some(current) = cf_world.get(var) {
                        let original = self.observed_state.get(var).unwrap_or(0.0);
                        let new_value = original + (current - original) * decay_factor * 0.9;
                        cf_world.set(var, new_value);
                    }
                }
            }
        }

        // Compute baseline (no intervention)
        let mut baseline_world = self.observed_state.clone();
        for _ in 0..n_steps {
            self.evolve_state(&mut baseline_world, 0.0);
            if let Some(outcome) = baseline_world.get("portfolio_return") {
                baseline_values.push(outcome);
            }
        }

        // Calculate statistics
        let baseline_mean = self.mean(&baseline_values);
        let cf_mean = self.mean(&counterfactual_values);
        let causal_effect = cf_mean - baseline_mean;

        // Confidence interval via bootstrap (simplified)
        let cf_std = self.std(&counterfactual_values);
        let ci_margin = 1.96 * cf_std / (n_steps as f64).sqrt();

        // Probability of extreme outcome
        let threshold = baseline_mean - 3.0 * self.std(&baseline_values);
        let extreme_count = counterfactual_values
            .iter()
            .filter(|&&v| v < threshold)
            .count();
        let prob_extreme = extreme_count as f64 / counterfactual_values.len() as f64;

        log::info!(
            "Counterfactual '{}' completed in {:?}: effect={:.4}, P(extreme)={:.4}",
            scenario.name,
            start.elapsed(),
            causal_effect,
            prob_extreme
        );

        CounterfactualResult {
            scenario_name: scenario.name.clone(),
            baseline_outcome: baseline_mean,
            counterfactual_outcome: cf_mean,
            causal_effect,
            confidence_interval: (cf_mean - ci_margin, cf_mean + ci_margin),
            probability_of_extreme: prob_extreme,
        }
    }

    /// Run multiple scenarios in parallel
    pub fn simulate_scenarios_batch(
        &self,
        scenarios: &[MacroScenario],
        n_steps: u32,
    ) -> Vec<CounterfactualResult> {
        scenarios
            .par_iter()
            .map(|s| self.simulate_counterfactual(s, n_steps))
            .collect()
    }

    /// Evolve state one timestep using structural equations
    fn evolve_state(&self, state: &mut WorldState, time: f64) {
        let mut new_values = HashMap::new();

        for var in &self.causal_order {
            if let Some(eq) = self.structural_equations.get(var) {
                // Generate noise
                let (mean, std) = self.noise_params.get(var).unwrap_or(&(0.0, 0.01));
                let noise = rand_distr::Distribution::sample(
                    rand_distr::Normal::new(*mean, *std).unwrap(),
                );

                // Evaluate structural equation
                let new_value = eq(state, noise);
                new_values.insert(var.clone(), new_value);
            }
        }

        // Update state
        for (var, value) in new_values {
            state.set(&var, value);
        }

        state.timestamp += 1;
    }

    /// Generate black swan scenarios
    pub fn generate_black_swan_scenarios(&self) -> Vec<MacroScenario> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut scenarios = Vec::new();

        // Fat-tailed shock generation (using Pareto distribution approximation)
        let shock_magnitude: f64 = {
            let alpha = 2.5; // Pareto shape parameter
            let u = rng.gen::<f64>();
            u.powf(-1.0 / alpha)
        };

        // Black swan: simultaneous multi-asset crash
        scenarios.push(MacroScenario {
            name: "BlackSwan_CryptoCrash".to_string(),
            event_type: MacroEventType::BlackSwan,
            magnitude: -shock_magnitude,
            duration_steps: 20,
            affected_variables: HashMap::from([
                ("BTC_price".to_string(), -0.15),
                ("ETH_price".to_string(), -0.20),
                ("SOL_price".to_string(), -0.30),
                ("market_liquidity".to_string(), -0.50),
                ("volatility_index".to_string(), 2.0),
            ]),
        });

        // Regulatory shock
        scenarios.push(MacroScenario {
            name: "RegulatoryBan_China".to_string(),
            event_type: MacroEventType::RegulatoryAnnouncement,
            magnitude: -1.0,
            duration_steps: 10,
            affected_variables: HashMap::from([
                ("BTC_price".to_string(), -0.10),
                ("trading_volume".to_string(), -0.40),
                ("fear_greed_index".to_string(), -30.0),
            ]),
        });

        // Liquidity crisis
        scenarios.push(MacroScenario {
            name: "LiquidityCrisis_Stablecoin".to_string(),
            event_type: MacroEventType::BankFailure,
            magnitude: -2.0,
            duration_steps: 15,
            affected_variables: HashMap::from([
                ("USDT_premium".to_string(), -0.05),
                ("funding_rate".to_string(), -0.001),
                ("open_interest".to_string(), -0.25),
            ]),
        });

        scenarios
    }

    /// Compute optimal hedge based on counterfactual analysis
    pub fn compute_optimal_hedge(
        &self,
        scenarios: &[CounterfactualResult],
        risk_tolerance: f64,
    ) -> HedgeRecommendation {
        // Find worst-case scenario
        let worst_scenario = scenarios
            .iter()
            .min_by(|a, b| a.causal_effect.partial_cmp(&b.causal_effect).unwrap())
            .unwrap();

        // Calculate required hedge size
        let max_loss = -worst_scenario.causal_effect.min(0.0);
        let hedge_ratio = (max_loss / risk_tolerance).min(1.0);

        HedgeRecommendation {
            recommended_hedge_ratio: hedge_ratio,
            worst_case_scenario: worst_scenario.scenario_name.clone(),
            expected_loss_reduction: max_loss * hedge_ratio,
            confidence: 1.0 - worst_scenario.probability_of_extreme,
        }
    }

    fn mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn std(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean(values);
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Hedge recommendation based on counterfactual analysis
#[derive(Debug, Clone)]
pub struct HedgeRecommendation {
    pub recommended_hedge_ratio: f64,
    pub worst_case_scenario: String,
    pub expected_loss_reduction: f64,
    pub confidence: f64,
}

// Helper module for random number generation
mod rand_distr {
    use rand::Rng;
    
    pub struct Normal {
        mean: f64,
        std: f64,
    }

    impl Normal {
        pub fn new(mean: f64, std: f64) -> Result<Self, &'static str> {
            if std < 0.0 {
                return Err("Standard deviation must be non-negative");
            }
            Ok(Self { mean, std })
        }
    }

    pub trait Distribution {
        fn sample<R: Rng>(dist: Self) -> f64 where Self: Sized;
    }

    impl Distribution for Normal {
        fn sample<R: Rng>(dist: Self) -> f64 {
            // Box-Muller transform
            let u1 = R::gen::<f64>();
            let u2 = R::gen::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            dist.mean + dist.std * z
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counterfactual_simulation() {
        let mut simulator = CounterfactualSimulator::new(vec![
            "market_return".to_string(),
            "portfolio_return".to_string(),
        ]);

        // Simple linear structural equations
        simulator.add_equation(
            "market_return",
            |state, noise| noise * 0.02,
            1.0,
        );

        simulator.add_equation(
            "portfolio_return",
            |state, noise| {
                let market = state.get("market_return").unwrap_or(0.0);
                market * 1.2 + noise * 0.01
            },
            0.5,
        );

        // Set observed state
        let initial_state = HashMap::from([
            ("market_return".to_string(), 0.001),
            ("portfolio_return".to_string(), 0.0012),
        ]);
        simulator.set_observed_state(initial_state);

        // Create a stress scenario
        let scenario = MacroScenario {
            name: "MarketCrash".to_string(),
            event_type: MacroEventType::BlackSwan,
            magnitude: -3.0,
            duration_steps: 5,
            affected_variables: HashMap::from([
                ("market_return".to_string(), -0.05),
            ]),
        };

        let result = simulator.simulate_counterfactual(&scenario, 10);

        assert!(result.causal_effect < 0.0, "Crash should have negative effect");
        println!("Causal effect: {:.6}", result.causal_effect);
    }

    #[test]
    fn test_black_swan_generation() {
        let simulator = CounterfactualSimulator::new(vec![]);
        let scenarios = simulator.generate_black_swan_scenarios();

        assert!(!scenarios.is_empty());
        assert!(scenarios.iter().any(|s| s.event_type == MacroEventType::BlackSwan));
        
        for scenario in &scenarios {
            println!("Scenario: {}, Type: {:?}", scenario.name, scenario.event_type);
        }
    }
}
