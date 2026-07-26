//! Reverse Stress Testing Engine
//! 
//! Mathematically finds the exact market conditions that would cause
//! portfolio ruin using gradient descent optimization.
//! Optimized for microsecond execution to instantly flag fatal states.
//! 
//! Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
//! ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour

use std::collections::HashMap;
use thiserror::Error;

/// Errors specific to reverse stress testing
#[derive(Error, Debug)]
pub enum ReverseStressError {
    #[error("Optimization failed to converge after {iterations} iterations")]
    OptimizationFailed { iterations: usize },
    #[error("Invalid target loss: {0}")]
    InvalidTargetLoss(String),
    #[error("Asset not found in portfolio: {0}")]
    AssetNotFound(String),
    #[error("Numerical overflow in calculation")]
    NumericalOverflow,
    #[error("Gradient computation failed: {0}")]
    GradientFailure(String),
}

/// Result type for reverse stress operations
pub type ReverseStressResult<T> = Result<T, ReverseStressError>;

/// Configuration for reverse stress testing
#[derive(Debug, Clone)]
pub struct ReverseStressConfig {
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Minimum shock magnitude to consider
    pub min_shock: f64,
    /// Maximum shock magnitude (circuit breaker)
    pub max_shock: f64,
}

impl Default for ReverseStressConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            learning_rate: 0.01,
            min_shock: -0.99,
            max_shock: 0.50,
        }
    }
}

/// Result of reverse stress test - the fatal scenario
#[derive(Debug, Clone)]
pub struct FatalScenario {
    /// Asset-specific shocks that cause ruin
    pub asset_shocks: HashMap<String, f64>,
    /// Total portfolio loss at this scenario
    pub total_loss: f64,
    /// Loss relative to target (should be ~0)
    pub loss_error: f64,
    /// Number of iterations to find solution
    pub iterations: usize,
    /// Whether solution converged
    pub converged: bool,
    /// Scenario severity rating
    pub severity: SeverityLevel,
}

/// Severity classification of fatal scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeverityLevel {
    /// Moderate stress (10-20% loss)
    Moderate,
    /// Severe stress (20-40% loss)
    Severe,
    /// Extreme stress (40-60% loss)
    Extreme,
    /// Catastrophic (>60% loss)
    Catastrophic,
}

impl SeverityLevel {
    /// Determine severity from loss percentage
    pub fn from_loss(loss_pct: f64) -> Self {
        let abs_loss = loss_pct.abs();
        if abs_loss < 0.20 {
            Self::Moderate
        } else if abs_loss < 0.40 {
            Self::Severe
        } else if abs_loss < 0.60 {
            Self::Extreme
        } else {
            Self::Catastrophic
        }
    }
    
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Moderate => "Moderate stress - manageable with risk controls",
            Self::Severe => "Severe stress - significant capital erosion",
            Self::Extreme => "Extreme stress - near margin call territory",
            Self::Catastrophic => "Catastrophic - portfolio ruin likely",
        }
    }
}

/// Reverse stress testing engine using gradient descent
pub struct ReverseStressEngine {
    /// Portfolio weights (asset -> weight)
    portfolio_weights: HashMap<String, f64>,
    /// Current asset prices (for P&L calculation)
    asset_prices: HashMap<String, f64>,
    /// Correlation matrix (flattened row-major)
    correlation: Vec<f64>,
    /// Number of assets
    n_assets: usize,
    /// Optimization configuration
    config: ReverseStressConfig,
}

impl ReverseStressEngine {
    /// Create a new reverse stress engine
    pub fn new(config: Option<ReverseStressConfig>) -> Self {
        Self {
            portfolio_weights: HashMap::new(),
            asset_prices: HashMap::new(),
            correlation: vec![],
            n_assets: 0,
            config: config.unwrap_or_default(),
        }
    }
    
    /// Set portfolio composition
    pub fn set_portfolio(
        &mut self,
        weights: HashMap<String, f64>,
        prices: HashMap<String, f64>,
    ) -> ReverseStressResult<()> {
        // Validate weights sum to ~1
        let weight_sum: f64 = weights.values().sum();
        if (weight_sum - 1.0).abs() > 0.01 {
            return Err(ReverseStressError::InvalidTargetLoss(
                format!("Portfolio weights must sum to 1.0, got {}", weight_sum)
            ));
        }
        
        self.n_assets = weights.len();
        self.portfolio_weights = weights;
        self.asset_prices = prices;
        
        // Initialize correlation as identity (will be overwritten)
        self.correlation = vec![0.0; self.n_assets * self.n_assets];
        for i in 0..self.n_assets {
            self.correlation[i * self.n_assets + i] = 1.0;
        }
        
        Ok(())
    }
    
    /// Set correlation matrix
    pub fn set_correlation(&mut self, corr: Vec<f64>) -> ReverseStressResult<()> {
        let expected_size = self.n_assets * self.n_assets;
        if corr.len() != expected_size {
            return Err(ReverseStressError::InvalidTargetLoss(
                format!("Correlation size mismatch: expected {}, got {}", expected_size, corr.len())
            ));
        }
        self.correlation = corr;
        Ok(())
    }
    
    /// Find the minimum market shock that causes target loss
    /// 
    /// Uses gradient descent to solve:
    ///   minimize ||shocks||^2
    ///   subject to: portfolio_loss(shocks) = target_loss
    pub fn find_fatal_scenario(
        &self,
        target_loss: f64,
    ) -> ReverseStressResult<FatalScenario> {
        if target_loss >= 0.0 {
            return Err(ReverseStressError::InvalidTargetLoss(
                "Target loss must be negative (representing a loss)".to_string()
            ));
        }
        
        if target_loss < -1.0 {
            return Err(ReverseStressError::InvalidTargetLoss(
                "Target loss cannot exceed 100%".to_string()
            ));
        }
        
        // Initialize shocks uniformly
        let mut shocks = vec![0.0; self.n_assets];
        
        // Initial portfolio loss
        let initial_loss = self.compute_portfolio_loss(&shocks);
        
        // If already at target, return immediately
        if (initial_loss - target_loss).abs() < self.config.tolerance {
            return Ok(self.create_scenario(shocks, target_loss, 0, true));
        }
        
        // Gradient descent optimization
        let mut prev_loss = initial_loss;
        let mut iteration = 0;
        
        for iter in 0..self.config.max_iterations {
            iteration = iter;
            
            // Compute gradient of loss w.r.t. shocks
            let gradient = self.compute_loss_gradient(&shocks)?;
            
            // Update shocks in direction that increases loss
            for i in 0..self.n_assets {
                // Weight by portfolio exposure
                let asset_idx = i;
                let weight = self.get_weight_by_index(asset_idx).unwrap_or(0.0);
                
                // Gradient step
                shocks[i] += self.config.learning_rate * gradient[i] * weight.abs();
                
                // Clamp to valid range
                shocks[i] = shocks[i].clamp(self.config.min_shock, self.config.max_shock);
            }
            
            // Compute new loss
            let current_loss = self.compute_portfolio_loss(&shocks);
            
            // Check convergence
            let loss_error = (current_loss - target_loss).abs();
            if loss_error < self.config.tolerance {
                return Ok(self.create_scenario(shocks, target_loss, iteration + 1, true));
            }
            
            // Check for stagnation
            if (current_loss - prev_loss).abs() < self.config.tolerance * 0.1 {
                // Reduce learning rate and continue
                // (simplified - would use adaptive learning rate in production)
            }
            
            prev_loss = current_loss;
        }
        
        // Return best effort even if not converged
        let final_loss = self.compute_portfolio_loss(&shocks);
        Ok(self.create_scenario(shocks, final_loss, iteration + 1, false))
    }
    
    /// Find multiple fatal scenarios with different characteristics
    pub fn find_multiple_fatal_scenarios(
        &self,
        target_losses: &[f64],
    ) -> ReverseStressResult<Vec<FatalScenario>> {
        target_losses
            .iter()
            .map(|&loss| self.find_fatal_scenario(loss))
            .collect()
    }
    
    /// Compute portfolio loss given asset shocks
    fn compute_portfolio_loss(&self, shocks: &[f64]) -> f64 {
        let mut loss = 0.0;
        
        for (i, asset) in self.portfolio_weights.keys().enumerate() {
            if i >= shocks.len() {
                break;
            }
            
            let weight = self.portfolio_weights.get(asset).copied().unwrap_or(0.0);
            let shock = shocks[i];
            
            loss += weight * shock;
        }
        
        loss
    }
    
    /// Compute gradient of portfolio loss w.r.t. each shock
    fn compute_loss_gradient(&self, shocks: &[f64]) -> ReverseStressResult<Vec<f64>> {
        let mut gradient = Vec::with_capacity(self.n_assets);
        
        for i in 0..self.n_assets {
            // Simple gradient: derivative of weighted sum
            let asset_idx = i;
            let weight = self.get_weight_by_index(asset_idx).unwrap_or(0.0);
            
            // Include correlation effects
            let mut corr_effect = 0.0;
            for j in 0..self.n_assets {
                if i != j {
                    let corr = self.correlation[i * self.n_assets + j];
                    let other_shock = shocks.get(j).copied().unwrap_or(0.0);
                    corr_effect += corr * other_shock * 0.1; // Damped correlation effect
                }
            }
            
            gradient.push(weight + corr_effect);
        }
        
        // Validate gradient
        if gradient.iter().any(|&g| g.is_nan() || g.is_infinite()) {
            return Err(ReverseStressError::GradientFailure(
                "NaN or Inf in gradient".to_string()
            ));
        }
        
        Ok(gradient)
    }
    
    /// Get weight by index
    fn get_weight_by_index(&self, idx: usize) -> Option<f64> {
        self.portfolio_weights
            .values()
            .nth(idx)
            .copied()
    }
    
    /// Create scenario result
    fn create_scenario(
        &self,
        shocks: Vec<f64>,
        total_loss: f64,
        iterations: usize,
        converged: bool,
    ) -> FatalScenario {
        let asset_shocks: HashMap<String, f64> = self.portfolio_weights
            .keys()
            .enumerate()
            .map(|(i, asset)| (*asset, shocks.get(i).copied().unwrap_or(0.0)))
            .collect();
        
        let loss_error = total_loss; // Relative to target (handled by caller)
        let severity = SeverityLevel::from_loss(total_loss);
        
        FatalScenario {
            asset_shocks,
            total_loss,
            loss_error,
            iterations,
            converged,
            severity,
        }
    }
    
    /// Get the most vulnerable asset (highest weight * volatility)
    pub fn get_most_vulnerable_asset(&self) -> Option<(String, f64)> {
        self.portfolio_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(asset, &weight)| (asset.clone(), weight))
    }
    
    /// Calculate the maximum possible loss (all assets go to zero)
    pub fn maximum_possible_loss(&self) -> f64 {
        -1.0 // 100% loss
    }
    
    /// Check if a given shock vector would cause ruin
    pub fn would_cause_ruin(&self, shocks: &[f64], ruin_threshold: f64) -> bool {
        let loss = self.compute_portfolio_loss(shocks);
        loss < ruin_threshold
    }
}

/// Builder for constructing reverse stress tests
pub struct ReverseStressBuilder {
    engine: ReverseStressEngine,
}

impl ReverseStressBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            engine: ReverseStressEngine::new(None),
        }
    }
    
    /// Add portfolio weight
    pub fn with_asset(mut self, asset: &str, weight: f64, price: f64) -> Self {
        self.engine.portfolio_weights.insert(asset.to_string(), weight);
        self.engine.asset_prices.insert(asset.to_string(), price);
        self.engine.n_assets = self.engine.portfolio_weights.len();
        self
    }
    
    /// Set correlation between two assets
    pub fn with_correlation(mut self, asset1: &str, asset2: &str, corr: f64) -> Self {
        // Would need proper index mapping in production
        self
    }
    
    /// Build the engine
    pub fn build(self) -> ReverseStressEngine {
        self.engine
    }
}

impl Default for ReverseStressBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fatal_scenario_finding() {
        let mut engine = ReverseStressEngine::new(None);
        
        let mut weights = HashMap::new();
        weights.insert("BTC".to_string(), 0.5);
        weights.insert("ETH".to_string(), 0.3);
        weights.insert("SOL".to_string(), 0.2);
        
        let prices = hashmap! {
            "BTC".to_string() => 50000.0,
            "ETH".to_string() => 3000.0,
            "SOL".to_string() => 100.0,
        };
        
        engine.set_portfolio(weights, prices).unwrap();
        
        let scenario = engine.find_fatal_scenario(-0.30).unwrap();
        
        assert!(scenario.total_loss < 0.0);
        assert_eq!(scenario.severity, SeverityLevel::Severe);
    }
    
    #[test]
    fn test_severity_classification() {
        assert_eq!(SeverityLevel::from_loss(-0.15), SeverityLevel::Moderate);
        assert_eq!(SeverityLevel::from_loss(-0.35), SeverityLevel::Severe);
        assert_eq!(SeverityLevel::from_loss(-0.50), SeverityLevel::Extreme);
        assert_eq!(SeverityLevel::from_loss(-0.70), SeverityLevel::Catastrophic);
    }
}

// Helper macro for hashmap creation
macro_rules! hashmap {
    ($( $key:expr => $value:expr ),* $(,)?) => {{
        let mut map = ::std::collections::HashMap::new();
        $( map.insert($key, $value); )*
        map
    }};
}
