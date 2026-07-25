// ZAID PERSONAL CRYPTO TRADING BOT - Stage 7
// Chapter 2: Macro-Economic Indicators - Cross-Asset Math
//
// File: backend/macro/cross_asset_math.rs
// Purpose: Calculate DXY, bond yields, gold correlations with crypto assets.
//          Map cross-asset flows to predict crypto movements.
//
// Features:
// - Real-time correlation matrix calculation
// - DXY (Dollar Index) computation from currency pairs
// - Bond yield curve analysis
// - Gold/crypto correlation tracking
// - Lead-lag relationship detection
//
// Design Patterns:
// - Strategy: Different correlation calculation methods
// - Observer: Notify on correlation regime changes
// - Builder: Construct complex correlation matrices
//
// Author: Opus 4.8
// Domain: Cross-Asset Analysis, Macro Trading, Correlation Trading

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use log::{info, warn, debug};
use chrono::{DateTime, Utc};

/// Maximum historical data points per asset (memory bound)
const MAX_DATA_POINTS: usize = 5000;

/// Supported asset classes for correlation analysis
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetClass {
    Crypto,
    Forex,
    Commodities,
    FixedIncome,
    Equity,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetType {
    // Crypto
    BTC,
    ETH,
    SOL,
    // Forex
    EURUSD,
    GBPUSD,
    USDJPY,
    // Commodities
    GOLD,
    SILVER,
    OIL,
    // Fixed Income
    US10Y,  // 10-year Treasury yield
    US2Y,   // 2-year Treasury yield
    // Equity
    SPX,    // S&P 500
    NDX,    // Nasdaq 100
}

impl AssetType {
    pub fn asset_class(&self) -> AssetClass {
        match self {
            AssetType::BTC | AssetType::ETH | AssetType::SOL => AssetClass::Crypto,
            AssetType::EURUSD | AssetType::GBPUSD | AssetType::USDJPY => AssetClass::Forex,
            AssetType::GOLD | AssetType::SILVER | AssetType::OIL => AssetClass::Commodities,
            AssetType::US10Y | AssetType::US2Y => AssetClass::FixedIncome,
            AssetType::SPX | AssetType::NDX => AssetClass::Equity,
        }
    }
}

/// Price data point with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub timestamp: u64,
    pub price: f64,
    pub volume: Option<f64>,
}

/// Yield curve data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YieldCurve {
    pub timestamp: u64,
    pub yields: HashMap<String, f64>, // maturity -> yield
}

/// Correlation calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    pub asset1: AssetType,
    pub asset2: AssetType,
    pub correlation: f64,
    pub p_value: f64,
    pub sample_size: usize,
    pub period_days: u32,
    pub is_significant: bool,
}

/// DXY (Dollar Index) calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DXYResult {
    pub timestamp: u64,
    pub dxy_value: f64,
    pub components: HashMap<String, f64>,
    pub change_24h_pct: f64,
    pub trend: TrendDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Rising,
    Falling,
    Neutral,
}

/// Correlation regime classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRegime {
    pub regime_type: RegimeType,
    pub crypto_correlation_avg: f64,
    pub risk_on_correlation: f64,
    pub safe_haven_correlation: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeType {
    RiskOn,      // Positive stock-crypto correlation
    RiskOff,     // Negative stock-crypto correlation  
    Decoupled,   // Low correlations across board
    Crisis,      // High correlations (everything correlated)
}

/// Rolling window statistics
#[derive(Debug, Clone)]
pub struct RollingStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl RollingStats {
    pub fn calculate(data: &[f64]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Skewness
        let skewness = if std_dev > 0.0 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>() * n / ((n - 1.0) * (n - 2.0))
        } else {
            0.0
        };

        // Kurtosis (excess)
        let kurtosis = if std_dev > 0.0 && n > 3 {
            let m4 = data.iter()
                .map(|x| (x - mean).powi(4))
                .sum::<f64>() / n;
            let m2 = variance * (n - 1.0) / n;
            (m4 / (m2 * m2)) - 3.0
        } else {
            0.0
        };

        Some(Self {
            mean,
            std_dev,
            min,
            max,
            skewness,
            kurtosis,
        })
    }
}

/// Thread-safe cross-asset correlation engine
pub struct CrossAssetEngine {
    /// Price history by asset type
    price_history: DashMap<AssetType, VecDeque<PricePoint>>,
    
    /// Latest prices
    latest_prices: DashMap<AssetType, f64>,
    
    /// Cached correlation matrix
    correlation_matrix: DashMap<(AssetType, AssetType), f64>,
    
    /// DXY history
    dxy_history: VecDeque<DXYResult>,
    
    /// Configuration
    config: CrossAssetConfig,
}

#[derive(Debug, Clone)]
pub struct CrossAssetConfig {
    pub max_history_points: usize,
    pub correlation_window_days: u32,
    pub significance_threshold: f64,
}

impl Default for CrossAssetConfig {
    fn default() -> Self {
        Self {
            max_history_points: MAX_DATA_POINTS,
            correlation_window_days: 30,
            significance_threshold: 0.05,
        }
    }
}

impl CrossAssetEngine {
    /// Create new cross-asset engine
    pub fn new(config: CrossAssetConfig) -> Self {
        info!("Initializing CrossAssetEngine");
        
        Self {
            price_history: DashMap::new(),
            latest_prices: DashMap::new(),
            correlation_matrix: DashMap::new(),
            dxy_history: VecDeque::with_capacity(1000),
            config,
        }
    }
    
    /// Add price data point
    pub fn add_price(&self, asset: AssetType, point: PricePoint) {
        let mut history = self.price_history
            .entry(asset.clone())
            .or_insert_with(|| VecDeque::with_capacity(self.config.max_history_points));
        
        history.push_back(point);
        
        // Enforce memory limit
        while history.len() > self.config.max_history_points {
            history.pop_front();
        }
        
        // Update latest price
        self.latest_prices.insert(asset, point.price);
    }
    
    /// Get latest price for an asset
    pub fn get_latest_price(&self, asset: &AssetType) -> Option<f64> {
        self.latest_prices.get(asset).map(|r| *r)
    }
    
    /// Calculate Pearson correlation between two assets
    pub fn calculate_correlation(
        &self,
        asset1: &AssetType,
        asset2: &AssetType,
        window_days: Option<u32>
    ) -> Option<CorrelationResult> {
        let history1 = self.price_history.get(asset1)?;
        let history2 = self.price_history.get(asset2)?;
        
        let window = window_days.unwrap_or(self.config.correlation_window_days);
        let cutoff_time = current_timestamp() - (window as u64 * 24 * 3600);
        
        // Get aligned returns
        let returns1 = self.calculate_returns(&history1, cutoff_time);
        let returns2 = self.calculate_returns(&history2, cutoff_time);
        
        if returns1.len() < 10 || returns2.len() < 10 {
            return None;
        }
        
        // Align the returns (use minimum length)
        let n = returns1.len().min(returns2.len());
        let aligned1 = &returns1[returns1.len()-n..];
        let aligned2 = &returns2[returns2.len()-n..];
        
        // Calculate Pearson correlation
        let correlation = self.pearson_correlation(aligned1, aligned2);
        
        // Calculate p-value (simplified t-test approximation)
        let t_stat = if correlation.abs() < 1.0 {
            correlation * ((n as f64 - 2.0) / (1.0 - correlation.powi(2))).sqrt()
        } else {
            f64::INFINITY
        };
        
        // Simplified p-value (would use proper t-distribution in production)
        let p_value = (1.0 / (1.0 + t_stat.abs())).min(1.0);
        
        let result = CorrelationResult {
            asset1: asset1.clone(),
            asset2: asset2.clone(),
            correlation,
            p_value,
            sample_size: n,
            period_days: window,
            is_significant: p_value < self.config.significance_threshold,
        };
        
        // Cache result
        self.correlation_matrix.insert((asset1.clone(), asset2.clone()), correlation);
        
        Some(result)
    }
    
    /// Calculate full correlation matrix
    pub fn calculate_full_correlation_matrix(
        &self,
        assets: &[AssetType]
    ) -> HashMap<(AssetType, AssetType), f64> {
        let assets_clone = assets.to_vec();
        
        // Parallel calculation for efficiency
        let results: Vec<_> = assets_clone
            .par_iter()
            .flat_map(|a1| {
                assets_clone.par_iter()
                    .filter(|a2| a1 != a2)
                    .filter_map(|a2| {
                        self.calculate_correlation(a1, a2, None)
                            .map(|r| ((a1.clone(), a2.clone()), r.correlation))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        
        results.into_iter().collect()
    }
    
    /// Calculate DXY (Dollar Index) from component currencies
    pub fn calculate_dxy(&self, currency_pairs: &HashMap<String, f64>) -> Option<DXYResult> {
        // DXY weights (geometric weighted average)
        let weights: HashMap<&str, f64> = [
            ("EUR", 0.576),
            ("JPY", 0.136),
            ("GBP", 0.119),
            ("CAD", 0.091),
            ("SEK", 0.042),
            ("CHF", 0.036),
        ].iter().cloned().collect();
        
        // Base values for normalization (1973 = 100)
        let base_rates: HashMap<&str, f64> = [
            ("EURUSD", 1.0),
            ("USDJPY", 1.0),
            ("GBPUSD", 1.0),
            ("USDCAD", 1.0),
            ("USDSEK", 1.0),
            ("USDCHF", 1.0),
        ].iter().cloned().collect();
        
        let mut dxy_value = 100.0; // Base value
        
        for (currency, &weight) in &weights {
            let pair_name = match *currency {
                "EUR" | "GBP" | "SEK" => format!("{}USD", currency),
                "JPY" | "CAD" | "CHF" => format!("USD{}", currency),
                _ => continue,
            };
            
            if let Some(&current_rate) = currency_pairs.get(&pair_name) {
                if let Some(&base) = base_rates.get(pair_name.as_str()) {
                    let change_ratio = current_rate / base;
                    // For USD/XXX pairs, inverse the ratio
                    let adjusted_ratio = if currency == "JPY" || currency == "CAD" || currency == "CHF" {
                        1.0 / change_ratio
                    } else {
                        change_ratio
                    };
                    
                    dxy_value *= adjusted_ratio.powf(weight);
                }
            }
        }
        
        // Calculate 24h change
        let change_24h = if self.dxy_history.len() > 24 {
            let old_value = self.dxy_history[self.dxy_history.len() - 25].dxy_value;
            ((dxy_value - old_value) / old_value) * 100.0
        } else {
            0.0
        };
        
        let trend = if change_24h > 0.5 {
            TrendDirection::Rising
        } else if change_24h < -0.5 {
            TrendDirection::Falling
        } else {
            TrendDirection::Neutral
        };
        
        let result = DXYResult {
            timestamp: current_timestamp(),
            dxy_value,
            components: currency_pairs.clone(),
            change_24h_pct: change_24h,
            trend,
        };
        
        // Store in history
        self.dxy_history.push_back(result.clone());
        if self.dxy_history.len() > 1000 {
            self.dxy_history.pop_front();
        }
        
        Some(result)
    }
    
    /// Detect correlation regime
    pub fn detect_correlation_regime(&self) -> Option<CorrelationRegime> {
        // Get key correlations
        let btc_spx = self.calculate_correlation(&AssetType::BTC, &AssetType::SPX, Some(30))?;
        let btc_gold = self.calculate_correlation(&AssetType::BTC, &AssetType::GOLD, Some(30))?;
        let btc_bonds = self.calculate_correlation(&AssetType::BTC, &AssetType::US10Y, Some(30))?;
        
        // Calculate average crypto-stock correlation
        let crypto_equity_corr = btc_spx.correlation;
        
        // Safe haven correlation (gold and bonds)
        let safe_haven_corr = (btc_gold.correlation + btc_bonds.correlation) / 2.0;
        
        // Determine regime
        let regime = if crypto_equity_corr > 0.5 {
            RegimeType::Crisis  // Everything correlated
        } else if crypto_equity_corr > 0.2 {
            RegimeType::RiskOn
        } else if crypto_equity_corr < -0.2 {
            RegimeType::RiskOff
        } else {
            RegimeType::Decoupled
        };
        
        let confidence = if btc_spx.sample_size > 20 { 0.9 } else { 0.6 };
        
        Some(CorrelationRegime {
            regime_type: regime,
            crypto_correlation_avg: crypto_equity_corr,
            risk_on_correlation: crypto_equity_corr,
            safe_haven_correlation: safe_haven_corr,
            confidence,
        })
    }
    
    /// Calculate lead-lag relationship
    pub fn analyze_lead_lag(
        &self,
        leader: &AssetType,
        follower: &AssetType,
        max_lag_hours: u32
    ) -> Option<(u32, f64)> {
        let history_leader = self.price_history.get(leader)?;
        let history_follower = self.price_history.get(follower)?;
        
        let returns_leader = self.calculate_returns(&history_leader, 0);
        let returns_follower = self.calculate_returns(&history_follower, 0);
        
        if returns_leader.len() < 50 || returns_follower.len() < 50 {
            return None;
        }
        
        let mut best_lag = 0;
        let mut best_correlation = 0.0;
        
        for lag in 0..max_lag_hours {
            if lag >= returns_leader.len() as u32 {
                break;
            }
            
            let lagged_leader = &returns_leader[..returns_leader.len() - lag as usize];
            let follower_trimmed = &returns_follower[lag as usize..];
            
            let n = lagged_leader.len().min(follower_trimmed.len());
            if n < 10 {
                continue;
            }
            
            let corr = self.pearson_correlation(
                &lagged_leader[lagged_leader.len()-n..],
                &follower_trimmed[follower_trimmed.len()-n..]
            );
            
            if corr.abs() > best_correlation.abs() {
                best_correlation = corr;
                best_lag = lag;
            }
        }
        
        Some((best_lag, best_correlation))
    }
    
    /// Helper: Calculate returns from price history
    fn calculate_returns(
        &self,
        history: &dashmap::mapref::one::Ref<AssetType, VecDeque<PricePoint>>,
        cutoff_time: u64
    ) -> Vec<f64> {
        let prices: Vec<f64> = history
            .iter()
            .filter(|p| p.timestamp > cutoff_time)
            .map(|p| p.price)
            .collect();
        
        if prices.len() < 2 {
            return vec![];
        }
        
        prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
    
    /// Helper: Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 2 {
            return 0.0;
        }
        
        let x = &x[..n];
        let y = &y[..n];
        
        let mean_x = x.iter().sum::<f64>() / n as f64;
        let mean_y = y.iter().sum::<f64>() / n as f64;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    /// Get rolling statistics for an asset
    pub fn get_rolling_stats(&self, asset: &AssetType, window: usize) -> Option<RollingStats> {
        let history = self.price_history.get(asset)?;
        
        let prices: Vec<f64> = history
            .iter()
            .rev()
            .take(window)
            .map(|p| p.price)
            .collect();
        
        RollingStats::calculate(&prices)
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_correlation_calculation() {
        let engine = CrossAssetEngine::new(CrossAssetConfig::default());
        
        // Add some correlated test data
        let now = current_timestamp();
        for i in 0..100 {
            let ts = now - (100 - i) * 3600;
            
            // BTC prices (simulated)
            engine.add_price(AssetType::BTC, PricePoint {
                timestamp: ts,
                price: 50000.0 + (i as f64 * 100.0),
                volume: None,
            });
            
            // ETH prices (correlated with BTC)
            engine.add_price(AssetType::ETH, PricePoint {
                timestamp: ts,
                price: 3000.0 + (i as f64 * 5.0),
                volume: None,
            });
        }
        
        let corr = engine.calculate_correlation(&AssetType::BTC, &AssetType::ETH, Some(30));
        assert!(corr.is_some());
        let result = corr.unwrap();
        assert!(result.correlation > 0.5); // Should be highly correlated
    }
    
    #[test]
    fn test_rolling_stats() {
        let engine = CrossAssetEngine::new(CrossAssetConfig::default());
        
        let now = current_timestamp();
        for i in 0..50 {
            engine.add_price(AssetType::BTC, PricePoint {
                timestamp: now - (50 - i) * 3600,
                price: 50000.0 + (i as f64 * 100.0),
                volume: None,
            });
        }
        
        let stats = engine.get_rolling_stats(&AssetType::BTC, 30);
        assert!(stats.is_some());
        let s = stats.unwrap();
        assert!(s.mean > 0.0);
        assert!(s.std_dev > 0.0);
    }
}

fn main() {
    env_logger::init();
    
    let config = CrossAssetConfig::default();
    let engine = CrossAssetEngine::new(config);
    
    info!("CrossAssetEngine initialized and ready");
}
