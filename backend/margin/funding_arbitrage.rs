//! Funding Arbitrage - Optimal Hedge Routing by Funding Rate
//! 
//! This module routes hedges to contracts with the most favorable funding rates.
//! It monitors funding rates across exchanges and tenors to minimize hedging costs
//! and potentially capture funding arbitrage opportunities.
//! 
//! Key Features:
//! - Real-time funding rate monitoring across venues
//! - Cost-optimal hedge routing algorithm
//! - Funding arbitrage opportunity detection
//! - Annualized funding yield calculation
//! - Zero-cost abstractions for memory efficiency
//! 
//! Target: Minimize hedge costs by routing to lowest funding rate venues

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::time::Instant;

/// Asset types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Asset {
    BTC,
    ETH,
    SOL,
}

impl Display for Asset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Asset::BTC => write!(f, "BTC"),
            Asset::ETH => write!(f, "ETH"),
            Asset::SOL => write!(f, "SOL"),
        }
    }
}

/// Exchange/venue identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Exchange {
    Binance,
    Bybit,
    OKX,
    Deribit,
}

impl Display for Exchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exchange::Binance => write!(f, "Binance"),
            Exchange::Bybit => write!(f, "Bybit"),
            Exchange::OKX => write!(f, "OKX"),
            Exchange::Deribit => write!(f, "Deribit"),
        }
    }
}

/// Contract tenor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tenor {
    Perpetual,
    Weekly,
    Biweekly,
    Monthly,
    Quarterly,
}

impl Display for Tenor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tenor::Perpetual => write!(f, "PERP"),
            Tenor::Weekly => write!(f, "1W"),
            Tenor::Biweekly => write!(f, "2W"),
            Tenor::Monthly => write!(f, "1M"),
            Tenor::Quarterly => write!(f, "3M"),
        }
    }
}

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

impl Display for PositionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionSide::Long => write!(f, "Long"),
            PositionSide::Short => write!(f, "Short"),
        }
    }
}

/// Funding rate data point
#[derive(Debug, Clone)]
pub struct FundingRate {
    pub exchange: Exchange,
    pub asset: Asset,
    pub tenor: Tenor,
    pub current_rate: f64,      // 8-hour funding rate (decimal)
    pub annualized_rate: f64,   // Annualized funding rate
    pub next_funding_time: u64, // Unix timestamp
    pub mark_price: f64,
    pub index_price: f64,
    pub basis_bps: f64,         // Basis in basis points
    pub open_interest: f64,
    pub volume_24h: f64,
    pub last_updated: Instant,
}

impl FundingRate {
    /// Calculate effective cost for a position over holding period
    pub fn effective_cost(&self, notional: f64, holding_hours: f64) -> f64 {
        // Funding occurs every 8 hours
        let funding_periods = holding_hours / 8.0;
        notional * self.current_rate * funding_periods
    }
    
    /// Check if rate is favorable for a given side
    pub fn is_favorable_for(&self, side: PositionSide) -> bool {
        match side {
            // Longs prefer negative funding (receive payments)
            PositionSide::Long => self.current_rate < 0.0,
            // Shorts prefer positive funding (receive payments)
            PositionSide::Short => self.current_rate > 0.0,
        }
    }
    
    /// Get funding yield for a side (positive = receive, negative = pay)
    pub fn yield_for_side(&self, side: PositionSide) -> f64 {
        match side {
            PositionSide::Long => -self.current_rate,
            PositionSide::Short => self.current_rate,
        }
    }
}

/// Hedge route recommendation
#[derive(Debug, Clone)]
pub struct HedgeRoute {
    pub exchange: Exchange,
    pub asset: Asset,
    pub tenor: Tenor,
    pub side: PositionSide,
    pub quantity: f64,
    pub expected_funding_cost: f64,
    pub expected_slippage: f64,
    pub total_cost: f64,
    pub confidence: f64,
}

impl HedgeRoute {
    /// Quality score (lower is better)
    pub fn quality_score(&self) -> f64 {
        self.total_cost / self.confidence.max(0.01)
    }
}

/// Funding arbitrage opportunity
#[derive(Debug, Clone)]
pub struct FundingArbitrage {
    pub long_exchange: Exchange,
    pub short_exchange: Exchange,
    pub asset: Asset,
    pub spread_bps: f64,
    pub annualized_return: f64,
    pub capacity: f64,
    pub risk_score: f64,
}

impl PartialOrd for FundingArbitrage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FundingArbitrage {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher annualized return is better
        self.annualized_return
            .partial_cmp(&other.annualized_return)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialEq for FundingArbitrage {
    fn eq(&self, other: &Self) -> bool {
        self.annualized_return == other.annualized_return
    }
}

impl Eq for FundingArbitrage {}

/// Priority queue item for best rates
#[derive(Debug, Clone)]
struct RatePriority {
    exchange: Exchange,
    rate: f64,
    score: f64,
}

impl PartialEq for RatePriority {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for RatePriority {}

impl PartialOrd for RatePriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RatePriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score = higher priority
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

/// Main funding arbitrage engine
pub struct FundingArbitrageEngine {
    /// Funding rates by exchange/asset/tenor
    funding_rates: HashMap<(Exchange, Asset, Tenor), FundingRate>,
    
    /// Historical rates for trend analysis
    rate_history: HashMap<(Exchange, Asset, Tenor), Vec<f64>>,
    
    /// Configuration
    max_history_length: usize,
    min_confidence: f64,
    
    /// Metrics
    opportunities_found: u64,
    total_arb_profit: f64,
}

impl FundingArbitrageEngine {
    /// Create new engine with default settings
    pub fn new(max_history_length: usize, min_confidence: f64) -> Self {
        Self {
            funding_rates: HashMap::new(),
            rate_history: HashMap::new(),
            max_history_length,
            min_confidence,
            opportunities_found: 0,
            total_arb_profit: 0.0,
        }
    }
    
    /// Update funding rate for a contract
    pub fn update_rate(&mut self, rate: FundingRate) {
        let key = (rate.exchange, rate.asset, rate.tenor);
        
        // Update history
        let history = self.rate_history.entry(key).or_insert_with(Vec::new);
        history.push(rate.current_rate);
        
        // Trim history
        if history.len() > self.max_history_length {
            history.remove(0);
        }
        
        self.funding_rates.insert(key, rate);
    }
    
    /// Get funding rate for specific contract
    pub fn get_rate(&self, exchange: Exchange, asset: Asset, tenor: Tenor) -> Option<&FundingRate> {
        self.funding_rates.get(&(exchange, asset, tenor))
    }
    
    /// Find best route for a hedge
    pub fn find_best_route(
        &self,
        asset: Asset,
        side: PositionSide,
        quantity: f64,
        holding_hours: f64,
    ) -> Option<HedgeRoute> {
        let mut best_route: Option<HedgeRoute> = None;
        let mut best_score = f64::INFINITY;
        
        for ((exchange, rate_asset, tenor), rate) in &self.funding_rates {
            if *rate_asset != asset {
                continue;
            }
            
            // Calculate costs
            let funding_cost = rate.effective_cost(quantity * rate.mark_price, holding_hours);
            
            // Estimate slippage based on volume
            let slippage_estimate = if rate.volume_24h > 0.0 {
                (quantity * rate.mark_price / rate.volume_24h).abs() * 0.001
            } else {
                0.005 // Default 5 bps
            };
            
            let total_cost = funding_cost + slippage_estimate * quantity * rate.mark_price;
            
            // Calculate confidence based on liquidity and rate stability
            let confidence = self.calculate_confidence(exchange, rate);
            
            if confidence < self.min_confidence {
                continue;
            }
            
            let score = total_cost / confidence;
            
            if score < best_score {
                best_score = score;
                best_route = Some(HedgeRoute {
                    exchange: *exchange,
                    asset,
                    tenor: *tenor,
                    side,
                    quantity,
                    expected_funding_cost: funding_cost,
                    expected_slippage: slippage_estimate,
                    total_cost,
                    confidence,
                });
            }
        }
        
        best_route
    }
    
    /// Find all funding arbitrage opportunities
    pub fn find_arbitrage_opportunities(&self, min_spread_bps: f64) -> Vec<FundingArbitrage> {
        let mut opportunities = Vec::new();
        
        // Group rates by asset
        let mut rates_by_asset: HashMap<Asset, Vec<&FundingRate>> = HashMap::new();
        for rate in self.funding_rates.values() {
            rates_by_asset.entry(rate.asset).or_default().push(rate);
        }
        
        // Find cross-exchange arb opportunities
        for (asset, rates) in rates_by_asset {
            for i in 0..rates.len() {
                for j in (i + 1)..rates.len() {
                    let rate_a = rates[i];
                    let rate_b = rates[j];
                    
                    if rate_a.exchange == rate_b.exchange {
                        continue;
                    }
                    
                    // Calculate spread
                    let spread = (rate_a.current_rate - rate_b.current_rate).abs();
                    let spread_bps = spread * 10000.0;
                    
                    if spread_bps < min_spread_bps {
                        continue;
                    }
                    
                    // Annualized return (assuming continuous arb)
                    let annual_periods = 365.0 * 3.0; // 3 funding periods per day
                    let annualized_return = spread * annual_periods * 100.0;
                    
                    // Capacity limited by lower OI
                    let capacity = rate_a.open_interest.min(rate_b.open_interest) * 0.1;
                    
                    // Risk score based on exchange reliability and rate volatility
                    let risk_score = self.calculate_arb_risk(rate_a, rate_b);
                    
                    opportunities.push(FundingArbitrage {
                        long_exchange: if rate_a.current_rate < rate_b.current_rate {
                            rate_a.exchange
                        } else {
                            rate_b.exchange
                        },
                        short_exchange: if rate_a.current_rate < rate_b.current_rate {
                            rate_b.exchange
                        } else {
                            rate_a.exchange
                        },
                        asset,
                        spread_bps,
                        annualized_return,
                        capacity,
                        risk_score,
                    });
                    
                    self.opportunities_found += 1;
                }
            }
        }
        
        // Sort by annualized return
        opportunities.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        
        opportunities
    }
    
    /// Get optimal hedge distribution across venues
    pub fn optimize_hedge_distribution(
        &self,
        asset: Asset,
        side: PositionSide,
        total_quantity: f64,
        holding_hours: f64,
    ) -> Vec<HedgeRoute> {
        let mut routes = Vec::new();
        let mut remaining = total_quantity;
        
        // Get all available venues sorted by cost
        let mut venue_costs: Vec<_> = self.funding_rates
            .iter()
            .filter(|((_, rate_asset, _), _)| *rate_asset == asset)
            .map(|((exchange, _, tenor), rate)| {
                let cost_per_unit = rate.yield_for_side(side);
                let liquidity_score = rate.volume_24h.sqrt();
                (exchange, tenor, rate, cost_per_unit, liquidity_score)
            })
            .collect();
        
        // Sort by yield (higher is better for receiving funding)
        venue_costs.sort_by(|a, b| {
            b.3.partial_cmp(&a.3).unwrap_or(Ordering::Equal)
        });
        
        // Distribute quantity across venues
        for (exchange, tenor, rate, _, liquidity) in venue_costs {
            if remaining <= 0.0 {
                break;
            }
            
            // Max allocation based on liquidity
            let max_allocation = liquidity * 0.01; // 1% of daily volume
            let allocation = remaining.min(max_allocation);
            
            if allocation > 0.0 {
                let funding_cost = rate.effective_cost(allocation * rate.mark_price, holding_hours);
                
                routes.push(HedgeRoute {
                    exchange: *exchange,
                    asset,
                    tenor: *tenor,
                    side,
                    quantity: allocation,
                    expected_funding_cost: funding_cost,
                    expected_slippage: 0.0,
                    total_cost: funding_cost,
                    confidence: 0.9,
                });
                
                remaining -= allocation;
            }
        }
        
        routes
    }
    
    /// Calculate confidence score for a rate
    fn calculate_confidence(&self, exchange: &Exchange, rate: &FundingRate) -> f64 {
        let mut confidence = 0.5;
        
        // Liquidity factor
        if rate.volume_24h > 1e9 {
            confidence += 0.3;
        } else if rate.volume_24h > 1e8 {
            confidence += 0.2;
        } else if rate.volume_24h > 1e7 {
            confidence += 0.1;
        }
        
        // Exchange reliability factor
        match exchange {
            Exchange::Binance | Exchange::Deribit => confidence += 0.2,
            Exchange::Bybit | Exchange::OKX => confidence += 0.15,
        }
        
        // Rate stability factor
        let key = (rate.exchange, rate.asset, rate.tenor);
        if let Some(history) = self.rate_history.get(&key) {
            if history.len() >= 10 {
                let variance: f64 = history.iter()
                    .map(|r| (r - rate.current_rate).powi(2))
                    .sum::<f64>() / history.len() as f64;
                
                if variance < 0.0001 {
                    confidence += 0.1;
                }
            }
        }
        
        confidence.min(1.0)
    }
    
    /// Calculate arbitrage risk score
    fn calculate_arb_risk(&self, rate_a: &FundingRate, rate_b: &FundingRate) -> f64 {
        let mut risk = 0.5;
        
        // Exchange risk
        match (rate_a.exchange, rate_b.exchange) {
            (Exchange::Binance, Exchange::Deribit) | (Exchange::Deribit, Exchange::Binance) => {
                risk -= 0.2; // Lower risk for reputable exchanges
            }
            _ => {}
        }
        
        // Liquidation risk based on basis
        let avg_basis = (rate_a.basis_bps.abs() + rate_b.basis_bps.abs()) / 2.0;
        if avg_basis > 100.0 {
            risk += 0.2; // Higher basis = higher liquidation risk
        }
        
        risk.min(1.0)
    }
    
    /// Get summary statistics
    pub fn get_summary(&self) -> FundingSummary {
        let total_rates = self.funding_rates.len();
        
        let avg_rate = self.funding_rates.values()
            .map(|r| r.current_rate)
            .sum::<f64>() / total_rates.max(1) as f64;
        
        let best_long_rate = self.funding_rates.values()
            .min_by(|a, b| a.current_rate.partial_cmp(&b.current_rate).unwrap_or(Ordering::Equal))
            .map(|r| r.current_rate);
        
        let best_short_rate = self.funding_rates.values()
            .max_by(|a, b| a.current_rate.partial_cmp(&b.current_rate).unwrap_or(Ordering::Equal))
            .map(|r| r.current_rate);
        
        FundingSummary {
            total_contracts: total_rates,
            average_funding_rate: avg_rate,
            best_rate_for_longs: best_long_rate,
            best_rate_for_shorts: best_short_rate,
            opportunities_tracked: self.opportunities_found,
            total_arb_profit: self.total_arb_profit,
        }
    }
}

/// Summary statistics
#[derive(Debug)]
pub struct FundingSummary {
    pub total_contracts: usize,
    pub average_funding_rate: f64,
    pub best_rate_for_longs: Option<f64>,
    pub best_rate_for_shorts: Option<f64>,
    pub opportunities_tracked: u64,
    pub total_arb_profit: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_funding_rate_cost() {
        let rate = FundingRate {
            exchange: Exchange::Binance,
            asset: Asset::BTC,
            tenor: Tenor::Perpetual,
            current_rate: 0.0001, // 1 bp per 8 hours
            annualized_rate: 0.1095,
            next_funding_time: 0,
            mark_price: 50000.0,
            index_price: 50000.0,
            basis_bps: 0.0,
            open_interest: 1e9,
            volume_24h: 5e9,
            last_updated: Instant::now(),
        };
        
        let notional = 100000.0;
        let cost = rate.effective_cost(notional, 24.0); // 24 hours
        
        assert!((cost - 30.0).abs() < 0.01); // 3 funding periods * 100000 * 0.0001
    }
    
    #[test]
    fn test_best_route_selection() {
        let mut engine = FundingArbitrageEngine::new(100, 0.5);
        
        // Add two exchanges with different rates
        engine.update_rate(FundingRate {
            exchange: Exchange::Binance,
            asset: Asset::BTC,
            tenor: Tenor::Perpetual,
            current_rate: 0.0001,
            annualized_rate: 0.1095,
            next_funding_time: 0,
            mark_price: 50000.0,
            index_price: 50000.0,
            basis_bps: 0.0,
            open_interest: 1e9,
            volume_24h: 5e9,
            last_updated: Instant::now(),
        });
        
        engine.update_rate(FundingRate {
            exchange: Exchange::Bybit,
            asset: Asset::BTC,
            tenor: Tenor::Perpetual,
            current_rate: 0.00005, // Lower rate
            annualized_rate: 0.05475,
            next_funding_time: 0,
            mark_price: 50000.0,
            index_price: 50000.0,
            basis_bps: 0.0,
            open_interest: 5e8,
            volume_24h: 2e9,
            last_updated: Instant::now(),
        });
        
        let route = engine.find_best_route(Asset::BTC, PositionSide::Long, 1.0, 24.0);
        
        assert!(route.is_some());
        // Bybit should be preferred for longs (lower funding cost)
        assert_eq!(route.unwrap().exchange, Exchange::Bybit);
    }
    
    #[test]
    fn test_arbitrage_detection() {
        let mut engine = FundingArbitrageEngine::new(100, 0.5);
        
        // Create large spread
        engine.update_rate(FundingRate {
            exchange: Exchange::Binance,
            asset: Asset::BTC,
            tenor: Tenor::Perpetual,
            current_rate: 0.0005,
            annualized_rate: 0.5475,
            next_funding_time: 0,
            mark_price: 50000.0,
            index_price: 50000.0,
            basis_bps: 10.0,
            open_interest: 1e9,
            volume_24h: 5e9,
            last_updated: Instant::now(),
        });
        
        engine.update_rate(FundingRate {
            exchange: Exchange::Bybit,
            asset: Asset::BTC,
            tenor: Tenor::Perpetual,
            current_rate: -0.0003,
            annualized_rate: -0.3285,
            next_funding_time: 0,
            mark_price: 50000.0,
            index_price: 50000.0,
            basis_bps: -5.0,
            open_interest: 1e9,
            volume_24h: 5e9,
            last_updated: Instant::now(),
        });
        
        let opportunities = engine.find_arbitrage_opportunities(10.0);
        
        assert!(!opportunities.is_empty());
        assert!(opportunities[0].spread_bps > 10.0);
    }
}
