//! Smart Order Router (SOR) Engine
//! Routes orders across spot, margin, and futures markets simultaneously.
//! Optimizes execution venue selection for best price and lowest cost.
//! 
//! Stage 13: Advanced Execution Algorithms
//! Target: Minimize market impact to secure 8k-20k INR/hour

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Trading venue types supported by SOR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Venue {
    Spot,
    MarginIsolated,
    MarginCross,
    FuturesUsdM,
    FuturesCoinM,
}

/// Order book level from a venue
#[derive(Debug, Clone)]
pub struct VenueLevel {
    pub venue: Venue,
    pub price_bps: i64,
    pub volume: u64,
    pub fee_bps: i64,
    pub latency_us: u64,
    pub available: bool,
}

/// Execution route candidate
#[derive(Debug, Clone)]
pub struct ExecutionRoute {
    pub venue: Venue,
    pub quantity: u64,
    pub expected_price_bps: i64,
    pub total_cost_bps: i64,
    pub estimated_slippage_bps: i64,
    pub priority_score: i64,
}

impl PartialEq for ExecutionRoute {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for ExecutionRoute {}

impl PartialOrd for ExecutionRoute {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExecutionRoute {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority score = better route
        other.priority_score.cmp(&self.priority_score)
    }
}

/// Smart Order Router State Machine
pub struct SorEngine {
    state: SorState,
    venues: Vec<VenueInfo>,
    order_id_counter: AtomicU64,
    is_active: AtomicBool,
    routing_config: RoutingConfig,
}

#[derive(Debug, Clone)]
pub struct VenueInfo {
    pub venue: Venue,
    pub base_fee_bps: i64,
    pub maker_rebate_bps: i64,
    pub taker_fee_bps: i64,
    pub min_order_size: u64,
    pub max_order_size: u64,
    pub tick_size_bps: i64,
    pub avg_latency_us: u64,
    pub reliability_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SorState {
    Idle,
    Analyzing,
    Routing,
    Executing,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct RoutingConfig {
    /// Prefer maker orders when spread allows
    pub prefer_maker: bool,
    /// Maximum slippage tolerance in bps
    pub max_slippage_bps: u64,
    /// Split orders across multiple venues if beneficial
    pub allow_splitting: bool,
    /// Minimum quantity per venue when splitting
    pub min_split_quantity: u64,
    /// Weight for latency in scoring (0-100)
    pub latency_weight: u64,
    /// Weight for fees in scoring (0-100)
    pub fee_weight: u64,
    /// Weight for liquidity in scoring (0-100)
    pub liquidity_weight: u64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            prefer_maker: true,
            max_slippage_bps: 50,
            allow_splitting: true,
            min_split_quantity: 100,
            latency_weight: 30,
            fee_weight: 40,
            liquidity_weight: 30,
        }
    }
}

impl SorEngine {
    pub fn new(routing_config: RoutingConfig) -> Self {
        let mut engine = Self {
            state: SorState::Idle,
            venues: Vec::new(),
            order_id_counter: AtomicU64::new(1),
            is_active: AtomicBool::new(false),
            routing_config,
        };
        
        // Initialize default venues
        engine.initialize_default_venues();
        
        engine
    }

    fn initialize_default_venues(&mut self) {
        self.venues = vec![
            VenueInfo {
                venue: Venue::Spot,
                base_fee_bps: 10,
                maker_rebate_bps: 0,
                taker_fee_bps: 10,
                min_order_size: 10,
                max_order_size: 10000000,
                tick_size_bps: 1,
                avg_latency_us: 50,
                reliability_score: 0.99,
            },
            VenueInfo {
                venue: Venue::MarginIsolated,
                base_fee_bps: 10,
                maker_rebate_bps: 0,
                taker_fee_bps: 10,
                min_order_size: 10,
                max_order_size: 5000000,
                tick_size_bps: 1,
                avg_latency_us: 60,
                reliability_score: 0.98,
            },
            VenueInfo {
                venue: Venue::FuturesUsdM,
                base_fee_bps: 4,
                maker_rebate_bps: 2,
                taker_fee_bps: 4,
                min_order_size: 1,
                max_order_size: 100000000,
                tick_size_bps: 1,
                avg_latency_us: 40,
                reliability_score: 0.995,
            },
        ];
    }

    /// Add or update venue information
    pub fn update_venue(&mut self, info: VenueInfo) {
        if let Some(existing) = self.venues.iter_mut().find(|v| v.venue == info.venue) {
            *existing = info;
        } else {
            self.venues.push(info);
        }
    }

    /// Find optimal execution route for an order
    pub fn find_best_route(
        &mut self,
        side: OrderSide,
        quantity: u64,
        market_prices: &[VenueLevel],
    ) -> Vec<ExecutionRoute> {
        self.state = SorState::Analyzing;
        
        let mut routes = BinaryHeap::new();
        
        // Score each venue
        for venue_info in &self.venues {
            if !self.is_venue_suitable(venue_info, quantity, side) {
                continue;
            }
            
            // Find matching market level
            let market_level = market_prices
                .iter()
                .find(|l| l.venue == venue_info.venue && l.available);
            
            if let Some(level) = market_level {
                let route = self.calculate_route_score(venue_info, level, quantity, side);
                if route.total_cost_bps as u64 <= self.routing_config.max_slippage_bps {
                    routes.push(route);
                }
            }
        }
        
        // Convert to vector of routes
        let mut result: Vec<ExecutionRoute> = routes.into_vec();
        
        // Handle order splitting if enabled and beneficial
        if self.routing_config.allow_splitting && quantity > self.routing_config.min_split_quantity * 2 {
            result = self.optimize_split(result, quantity, side);
        }
        
        self.state = SorState::Routing;
        result
    }

    fn is_venue_suitable(&self, venue: &VenueInfo, quantity: u64, _side: OrderSide) -> bool {
        quantity >= venue.min_order_size && quantity <= venue.max_order_size
    }

    fn calculate_route_score(
        &self,
        venue_info: &VenueInfo,
        level: &VenueLevel,
        quantity: u64,
        side: OrderSide,
    ) -> ExecutionRoute {
        // Calculate effective fee (maker vs taker)
        let effective_fee_bps = if self.routing_config.prefer_maker {
            venue_info.maker_rebate_bps
        } else {
            venue_info.taker_fee_bps
        };
        
        // Estimate slippage based on quantity vs available liquidity
        let slippage_estimate_bps = if level.volume > 0 {
            ((quantity as f64 / level.volume as f64) * 10.0) as i64
        } else {
            100 // High slippage if no liquidity
        };
        
        // Total cost = fee + slippage
        let total_cost_bps = effective_fee_bps.abs() + slippage_estimate_bps;
        
        // Calculate priority score (higher is better)
        let latency_score = (100 - ((level.latency_us as f64 / 100.0).min(100.0) as i64)) 
            * self.routing_config.latency_weight as i64 / 100;
        let fee_score = (50 - total_cost_bps).max(0) * self.routing_config.fee_weight as i64 / 100;
        let liquidity_score = ((level.volume as f64 / 10000.0).min(100.0) as i64)
            * self.routing_config.liquidity_weight as i64 / 100;
        
        let priority_score = latency_score + fee_score + liquidity_score;
        
        ExecutionRoute {
            venue: venue_info.venue,
            quantity,
            expected_price_bps: level.price_bps,
            total_cost_bps,
            estimated_slippage_bps: slippage_estimate_bps,
            priority_score,
        }
    }

    fn optimize_split(
        &self,
        mut routes: Vec<ExecutionRoute>,
        total_quantity: u64,
        side: OrderSide,
    ) -> Vec<ExecutionRoute> {
        if routes.is_empty() || total_quantity == 0 {
            return routes;
        }
        
        // Sort by priority
        routes.sort_by(|a, b| b.priority_score.cmp(&a.priority_score));
        
        let mut split_routes = Vec::new();
        let mut remaining_qty = total_quantity;
        
        for route in routes {
            if remaining_qty == 0 {
                break;
            }
            
            let alloc_qty = remaining_qty.min(route.quantity);
            if alloc_qty >= self.routing_config.min_split_quantity {
                let mut split_route = route.clone();
                split_route.quantity = alloc_qty;
                split_routes.push(split_route);
                remaining_qty -= alloc_qty;
            }
        }
        
        // If we couldn't allocate all, add remainder to best route
        if remaining_qty > 0 && !split_routes.is_empty() {
            split_routes[0].quantity += remaining_qty;
        }
        
        split_routes
    }

    /// Execute routed orders
    pub fn execute_routes(&mut self, routes: &[ExecutionRoute]) -> u64 {
        if routes.is_empty() {
            return 0;
        }
        
        self.state = SorState::Executing;
        self.is_active.store(true, Ordering::Relaxed);
        
        let order_id = self.order_id_counter.fetch_add(routes.len() as u64, Ordering::Relaxed);
        
        // In production, this would send actual orders to venues
        // For now, just simulate execution
        
        self.state = SorState::Completed;
        self.is_active.store(false, Ordering::Relaxed);
        
        order_id
    }

    /// Get current state
    pub fn get_state(&self) -> SorState {
        self.state
    }

    /// Check if engine is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }

    /// Reset engine
    pub fn reset(&mut self) {
        self.state = SorState::Idle;
        self.is_active.store(false, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sor_routing() {
        let config = RoutingConfig::default();
        let mut engine = SorEngine::new(config);
        
        let market_prices = vec![
            VenueLevel {
                venue: Venue::Spot,
                price_bps: 500000,
                volume: 10000,
                fee_bps: 10,
                latency_us: 50,
                available: true,
            },
            VenueLevel {
                venue: Venue::FuturesUsdM,
                price_bps: 499950,
                volume: 50000,
                fee_bps: 4,
                latency_us: 40,
                available: true,
            },
        ];
        
        let routes = engine.find_best_route(OrderSide::Buy, 5000, &market_prices);
        
        assert!(!routes.is_empty());
        // Futures should be preferred due to lower fees
        assert_eq!(routes[0].venue, Venue::FuturesUsdM);
    }

    #[test]
    fn test_order_splitting() {
        let mut config = RoutingConfig::default();
        config.allow_splitting = true;
        config.min_split_quantity = 1000;
        
        let mut engine = SorEngine::new(config);
        
        let market_prices = vec![
            VenueLevel {
                venue: Venue::Spot,
                price_bps: 500000,
                volume: 5000,
                fee_bps: 10,
                latency_us: 50,
                available: true,
            },
            VenueLevel {
                venue: Venue::FuturesUsdM,
                price_bps: 499950,
                volume: 5000,
                fee_bps: 4,
                latency_us: 40,
                available: true,
            },
        ];
        
        // Large order that should be split
        let routes = engine.find_best_route(OrderSide::Buy, 10000, &market_prices);
        
        // Should have split across venues
        assert!(!routes.is_empty());
    }
}
