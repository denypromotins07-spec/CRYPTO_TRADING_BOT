//! Historical Scenario Stress Testing Engine
//! 
//! Replays historical crypto market crises (Black Thursday, LUNA, FTX)
//! to assess portfolio resilience under extreme conditions.
//! Includes liquidity evaporation and slippage modeling.
//! 
//! Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
//! ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour

use std::collections::HashMap;
use thiserror::Error;

/// Errors specific to stress testing
#[derive(Error, Debug)]
pub enum StressTestError {
    #[error("Scenario not found: {0}")]
    ScenarioNotFound(String),
    #[error("Invalid scenario data: {0}")]
    InvalidScenarioData(String),
    #[error("Portfolio asset not in scenario: {0}")]
    AssetNotInScenario(String),
    #[error("Numerical overflow in stress calculation")]
    NumericalOverflow,
}

/// Result type for stress test operations
pub type StressTestResult<T> = Result<T, StressTestError>;

/// Predefined historical crisis scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistoricalScenario {
    /// March 12, 2020 - COVID crash
    BlackThursday,
    /// May 2022 - Terra/LUNA collapse
    LunaCollapse,
    /// November 2022 - FTX contagion
    FTXContagion,
    /// May 2021 - China mining ban
    ChinaMiningBan,
    /// June 2022 - 3AC liquidation
    ThreeArrowsLiquidation,
    /// Custom scenario
    Custom,
}

impl HistoricalScenario {
    /// Get the scenario name
    pub fn name(&self) -> &'static str {
        match self {
            Self::BlackThursday => "Black Thursday (COVID Crash)",
            Self::LunaCollapse => "Terra/LUNA Collapse",
            Self::FTXContagion => "FTX Contagion",
            Self::ChinaMiningBan => "China Mining Ban",
            Self::ThreeArrowsLiquidation => "3AC Liquidation",
            Self::Custom => "Custom Scenario",
        }
    }
    
    /// Get approximate duration in days
    pub fn duration_days(&self) -> usize {
        match self {
            Self::BlackThursday => 7,
            Self::LunaCollapse => 5,
            Self::FTXContagion => 10,
            Self::ChinaMiningBan => 14,
            Self::ThreeArrowsLiquidation => 21,
            Self::Custom => 0,
        }
    }
}

/// Stress scenario definition with asset-specific shocks
#[derive(Debug, Clone)]
pub struct StressScenario {
    /// Scenario identifier
    pub name: String,
    /// Asset returns during the scenario (asset -> return %)
    pub asset_shocks: HashMap<String, f64>,
    /// Correlation breakdown factor (1.0 = normal, >1.0 = increased correlation)
    pub correlation_multiplier: f64,
    /// Liquidity reduction factor (1.0 = normal, <1.0 = reduced liquidity)
    pub liquidity_factor: f64,
    /// Slippage multiplier (1.0 = normal, >1.0 = increased slippage)
    pub slippage_multiplier: f64,
    /// Volatility multiplier
    pub volatility_multiplier: f64,
}

impl StressScenario {
    /// Create a new stress scenario
    pub fn new(
        name: String,
        asset_shocks: HashMap<String, f64>,
        correlation_multiplier: f64,
        liquidity_factor: f64,
        slippage_multiplier: f64,
        volatility_multiplier: f64,
    ) -> StressTestResult<Self> {
        if correlation_multiplier < 0.0 || liquidity_factor <= 0.0 || slippage_multiplier < 0.0 {
            return Err(StressTestError::InvalidScenarioData(
                "Invalid scenario parameters".to_string()
            ));
        }
        
        Ok(Self {
            name,
            asset_shocks,
            correlation_multiplier,
            liquidity_factor,
            slippage_multiplier,
            volatility_multiplier,
        })
    }
    
    /// Get shock for a specific asset
    pub fn get_shock(&self, asset: &str) -> Option<f64> {
        self.asset_shocks.get(asset).copied()
    }
    
    /// Check if scenario contains an asset
    pub fn contains_asset(&self, asset: &str) -> bool {
        self.asset_shocks.contains_key(asset)
    }
}

/// Historical scenario replay engine
pub struct HistoricalScenarioEngine {
    /// Pre-built scenarios
    scenarios: HashMap<HistoricalScenario, StressScenario>,
    /// Current portfolio weights
    portfolio_weights: HashMap<String, f64>,
}

impl HistoricalScenarioEngine {
    /// Create a new engine with default historical scenarios
    pub fn new() -> Self {
        let mut engine = Self {
            scenarios: HashMap::new(),
            portfolio_weights: HashMap::new(),
        };
        
        // Initialize with predefined historical scenarios
        engine.initialize_scenarios();
        
        engine
    }
    
    /// Set portfolio weights for stress testing
    pub fn set_portfolio(&mut self, weights: HashMap<String, f64>) {
        self.portfolio_weights = weights;
    }
    
    /// Initialize predefined historical scenarios
    fn initialize_scenarios(&mut self) {
        // Black Thursday (March 12, 2020)
        let black_thursday = StressScenario::new(
            HistoricalScenario::BlackThursday.name().to_string(),
            hashmap! {
                "BTC".to_string() => -0.50,
                "ETH".to_string() => -0.55,
                "SOL".to_string() => -0.60,
                "USDT".to_string() => 0.02,
            },
            1.8,   // Correlations spiked to near 1
            0.3,   // Liquidity dried up significantly
            3.0,   // Extreme slippage
            2.5,   // Volatility exploded
        ).unwrap();
        self.scenarios.insert(HistoricalScenario::BlackThursday, black_thursday);
        
        // LUNA Collapse (May 2022)
        let luna_collapse = StressScenario::new(
            HistoricalScenario::LunaCollapse.name().to_string(),
            hashmap! {
                "BTC".to_string() => -0.35,
                "ETH".to_string() => -0.40,
                "SOL".to_string() => -0.45,
                "USDT".to_string() => 0.01,
            },
            1.5,
            0.4,
            2.5,
            2.0,
        ).unwrap();
        self.scenarios.insert(HistoricalScenario::LunaCollapse, luna_collapse);
        
        // FTX Contagion (November 2022)
        let ftx_contagion = StressScenario::new(
            HistoricalScenario::FTXContagion.name().to_string(),
            hashmap! {
                "BTC".to_string() => -0.25,
                "ETH".to_string() => -0.30,
                "SOL".to_string() => -0.55,
                "USDT".to_string() => 0.005,
            },
            1.6,
            0.35,
            4.0,   // Extreme slippage due to panic
            2.2,
        ).unwrap();
        self.scenarios.insert(HistoricalScenario::FTXContagion, ftx_contagion);
        
        // China Mining Ban (May 2021)
        let china_ban = StressScenario::new(
            HistoricalScenario::ChinaMiningBan.name().to_string(),
            hashmap! {
                "BTC".to_string() => -0.45,
                "ETH".to_string() => -0.35,
                "SOL".to_string() => -0.40,
                "USDT".to_string() => 0.01,
            },
            1.4,
            0.5,
            2.0,
            1.8,
        ).unwrap();
        self.scenarios.insert(HistoricalScenario::ChinaMiningBan, china_ban);
        
        // 3AC Liquidation (June 2022)
        let three_ac = StressScenario::new(
            HistoricalScenario::ThreeArrowsLiquidation.name().to_string(),
            hashmap! {
                "BTC".to_string() => -0.30,
                "ETH".to_string() => -0.35,
                "SOL".to_string() => -0.50,
                "USDT".to_string() => 0.005,
            },
            1.5,
            0.4,
            2.5,
            2.0,
        ).unwrap();
        self.scenarios.insert(HistoricalScenario::ThreeArrowsLiquidation, three_ac);
    }
    
    /// Run stress test for a specific scenario
    pub fn run_scenario(
        &self,
        scenario: HistoricalScenario
    ) -> StressTestResult<StressTestOutput> {
        let scenario_def = self.scenarios.get(&scenario)
            .ok_or_else(|| StressTestError::ScenarioNotFound(scenario.name().to_string()))?;
        
        // Calculate portfolio-level impact
        let mut portfolio_return = 0.0;
        let mut max_asset_loss = f64::NEG_INFINITY;
        let mut worst_asset = String::new();
        
        for (asset, weight) in &self.portfolio_weights {
            let shock = scenario_def.get_shock(asset)
                .ok_or_else(|| StressTestError::AssetNotInScenario(asset.clone()))?;
            
            let contribution = weight * shock;
            portfolio_return += contribution;
            
            if shock < max_asset_loss {
                max_asset_loss = shock;
                worst_asset = asset.clone();
            }
        }
        
        // Calculate liquidity-adjusted loss
        let liquidity_adjusted_loss = portfolio_return / scenario_def.liquidity_factor;
        
        // Calculate slippage cost
        let slippage_cost = portfolio_return.abs() * (scenario_def.slippage_multiplier - 1.0);
        
        // Total stressed loss
        let total_loss = liquidity_adjusted_loss - slippage_cost;
        
        Ok(StressTestOutput {
            scenario_name: scenario_def.name.clone(),
            portfolio_return,
            total_loss,
            max_asset_loss,
            worst_asset,
            liquidity_factor: scenario_def.liquidity_factor,
            slippage_multiplier: scenario_def.slippage_multiplier,
            volatility_multiplier: scenario_def.volatility_multiplier,
            correlation_breakdown: scenario_def.correlation_multiplier,
        })
    }
    
    /// Run all historical scenarios
    pub fn run_all_scenarios(&self) -> StressTestResult<Vec<StressTestOutput>> {
        let mut outputs = Vec::new();
        
        for scenario in [
            HistoricalScenario::BlackThursday,
            HistoricalScenario::LunaCollapse,
            HistoricalScenario::FTXContagion,
            HistoricalScenario::ChinaMiningBan,
            HistoricalScenario::ThreeArrowsLiquidation,
        ] {
            let output = self.run_scenario(scenario)?;
            outputs.push(output);
        }
        
        Ok(outputs)
    }
    
    /// Find the worst-case scenario for current portfolio
    pub fn find_worst_case(&self) -> StressTestResult<StressTestOutput> {
        let outputs = self.run_all_scenarios()?;
        
        outputs.into_iter()
            .min_by(|a, b| a.total_loss.partial_cmp(&b.total_loss).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| StressTestError::InvalidScenarioData(
                "No scenarios available".to_string()
            ))
    }
    
    /// Add a custom scenario
    pub fn add_custom_scenario(
        &mut self,
        name: String,
        asset_shocks: HashMap<String, f64>,
        correlation_mult: f64,
        liquidity_factor: f64,
        slippage_mult: f64,
        vol_mult: f64,
    ) -> StressTestResult<()> {
        let scenario = StressScenario::new(
            name.clone(),
            asset_shocks,
            correlation_mult,
            liquidity_factor,
            slippage_mult,
            vol_mult,
        )?;
        
        self.scenarios.insert(HistoricalScenario::Custom, scenario);
        Ok(())
    }
}

/// Output from a stress test run
#[derive(Debug, Clone)]
pub struct StressTestOutput {
    /// Name of the scenario
    pub scenario_name: String,
    /// Raw portfolio return under stress
    pub portfolio_return: f64,
    /// Total loss including liquidity and slippage
    pub total_loss: f64,
    /// Maximum single asset loss
    pub max_asset_loss: f64,
    /// Worst performing asset
    pub worst_asset: String,
    /// Liquidity factor applied
    pub liquidity_factor: f64,
    /// Slippage multiplier applied
    pub slippage_multiplier: f64,
    /// Volatility multiplier
    pub volatility_multiplier: f64,
    /// Correlation breakdown factor
    pub correlation_breakdown: f64,
}

impl StressTestOutput {
    /// Check if loss exceeds threshold
    pub fn exceeds_threshold(&self, threshold: f64) -> bool {
        self.total_loss < -threshold
    }
    
    /// Get risk rating based on loss magnitude
    pub fn risk_rating(&self) -> &'static str {
        if self.total_loss > -0.10 {
            "LOW"
        } else if self.total_loss > -0.25 {
            "MEDIUM"
        } else if self.total_loss > -0.40 {
            "HIGH"
        } else {
            "CRITICAL"
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_engine_initialization() {
        let engine = HistoricalScenarioEngine::new();
        assert!(engine.scenarios.len() >= 5);
    }
    
    #[test]
    fn test_portfolio_stress() {
        let mut engine = HistoricalScenarioEngine::new();
        engine.set_portfolio(hashmap! {
            "BTC".to_string() => 0.5,
            "ETH".to_string() => 0.3,
            "SOL".to_string() => 0.2,
        });
        
        let output = engine.run_scenario(HistoricalScenario::BlackThursday).unwrap();
        assert!(output.total_loss < 0.0);
        assert_eq!(output.risk_rating(), "CRITICAL");
    }
}
