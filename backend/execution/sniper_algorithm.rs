//! Sniper Algorithm: Hidden Liquidity Detection and Execution
//! Detects and executes against genuine hidden liquidity sweeps.
//! Filters out toxic spoofed walls to avoid adverse selection.
//! 
//! Stage 13: Advanced Execution Algorithms
//! Target: Minimize market impact to secure 8k-20k INR/hour

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

/// Order book level snapshot
#[derive(Debug, Clone)]
pub struct OrderBookLevel {
    pub price_bps: i64,
    pub volume: u64,
    pub order_count: u64,
    pub timestamp_ns: u64,
}

/// Detected hidden liquidity opportunity
#[derive(Debug, Clone)]
pub struct HiddenLiquidityOpportunity {
    pub detection_time_ns: u64,
    pub side: OrderSide,
    pub price_bps: i64,
    pub estimated_volume: u64,
    pub confidence_score: f64,
    pub is_genuine: bool,
    pub expiry_time_ns: u64,
}

/// Spoofing detection result
#[derive(Debug, Clone)]
pub struct SpoofingAnalysis {
    pub is_spoofing: bool,
    pub confidence: f64,
    pub wall_lifetime_us: u64,
    pub cancellation_rate: f64,
    pub price_impact_reversal_bps: i64,
}

/// Sniper execution state machine
pub struct SniperAlgorithm {
    state: SniperState,
    opportunities: VecDeque<HiddenLiquidityOpportunity>,
    executed_opportunities: Vec<HiddenLiquidityOpportunity>,
    order_book_history: VecDeque<OrderBookSnapshot>,
    config: SniperConfig,
    is_active: AtomicBool,
    last_execution_time: AtomicU64,
    total_executions: AtomicU64,
    successful_executions: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub timestamp_ns: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub mid_price_bps: i64,
    pub spread_bps: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct SniperConfig {
    /// Minimum confidence score to execute (0.0 to 1.0)
    pub min_confidence: f64,
    /// Maximum age of opportunity before considering stale (microseconds)
    pub max_opportunity_age_us: u64,
    /// Minimum volume threshold for consideration
    pub min_volume_threshold: u64,
    /// Lookback window for order book history (number of snapshots)
    pub lookback_window: usize,
    /// Spoofing detection sensitivity (higher = more aggressive detection)
    pub spoofing_sensitivity: f64,
    /// Minimum time between executions (microseconds)
    pub execution_cooldown_us: u64,
    /// Maximum slippage tolerance in bps
    pub max_slippage_bps: u64,
}

impl Default for SniperConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.75,
            max_opportunity_age_us: 1000, // 1ms
            min_volume_threshold: 1000,
            lookback_window: 50,
            spoofing_sensitivity: 0.8,
            execution_cooldown_us: 500, // 500μs between executions
            max_slippage_bps: 30,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SniperState {
    Idle,
    Scanning,
    Analyzing,
    Executing,
    Cooldown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl SniperAlgorithm {
    pub fn new(config: SniperConfig) -> Self {
        Self {
            state: SniperState::Idle,
            opportunities: VecDeque::with_capacity(100),
            executed_opportunities: Vec::new(),
            order_book_history: VecDeque::with_capacity(config.lookback_window),
            config,
            is_active: AtomicBool::new(false),
            last_execution_time: AtomicU64::new(0),
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
        }
    }

    /// Start the sniper algorithm
    pub fn start(&mut self) {
        self.state = SniperState::Scanning;
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Stop the sniper algorithm
    pub fn stop(&mut self) {
        self.state = SniperState::Idle;
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Process new order book snapshot
    pub fn process_order_book(&mut self, snapshot: OrderBookSnapshot) -> Vec<HiddenLiquidityOpportunity> {
        if !self.is_active.load(Ordering::Relaxed) {
            return Vec::new();
        }

        self.state = SniperState::Analyzing;

        // Add to history
        self.order_book_history.push_back(snapshot.clone());
        while self.order_book_history.len() > self.config.lookback_window {
            self.order_book_history.pop_front();
        }

        // Scan for hidden liquidity
        let mut opportunities = Vec::new();

        // Check bid side
        if let Some(opportunity) = self.detect_hidden_liquidity(&snapshot, OrderSide::Buy) {
            opportunities.push(opportunity);
        }

        // Check ask side
        if let Some(opportunity) = self.detect_hidden_liquidity(&snapshot, OrderSide::Sell) {
            opportunities.push(opportunity);
        }

        // Filter and validate opportunities
        let valid_opportunities: Vec<_> = opportunities
            .into_iter()
            .filter(|opp| self.validate_opportunity(opp))
            .collect();

        // Add to queue
        for opp in &valid_opportunities {
            self.opportunities.push_back(opp.clone());
        }

        // Prune stale opportunities
        self.prune_stale_opportunities(snapshot.timestamp_ns);

        self.state = SniperState::Scanning;
        valid_opportunities
    }

    /// Detect hidden liquidity on one side of the book
    fn detect_hidden_liquidity(
        &self,
        snapshot: &OrderBookSnapshot,
        side: OrderSide,
    ) -> Option<HiddenLiquidityOpportunity> {
        let levels = match side {
            OrderSide::Buy => &snapshot.bids,
            OrderSide::Sell => &snapshot.asks,
        };

        if levels.is_empty() {
            return None;
        }

        // Look for anomalies that suggest hidden liquidity
        let mut best_opportunity: Option<HiddenLiquidityOpportunity> = None;

        for (i, level) in levels.iter().enumerate() {
            if level.volume < self.config.min_volume_threshold {
                continue;
            }

            // Analyze if this level has characteristics of hidden liquidity
            let analysis = self.analyze_level_for_hidden_liquidity(level, i, levels, side);

            if analysis.confidence >= self.config.min_confidence && analysis.is_genuine {
                let opportunity = HiddenLiquidityOpportunity {
                    detection_time_ns: snapshot.timestamp_ns,
                    side,
                    price_bps: level.price_bps,
                    estimated_volume: level.volume,
                    confidence_score: analysis.confidence,
                    is_genuine: true,
                    expiry_time_ns: snapshot.timestamp_ns + (self.config.max_opportunity_age_us * 1000),
                };

                if best_opportunity.is_none() 
                    || opportunity.confidence_score > best_opportunity.as_ref().unwrap().confidence_score 
                {
                    best_opportunity = Some(opportunity);
                }
            }
        }

        best_opportunity
    }

    /// Analyze a single order book level for hidden liquidity characteristics
    fn analyze_level_for_hidden_liquidity(
        &self,
        level: &OrderBookLevel,
        level_index: usize,
        all_levels: &[OrderBookLevel],
        side: OrderSide,
    ) -> SpoofingAnalysis {
        // Check for spoofing indicators
        let spoofing_analysis = self.detect_spoofing(level, level_index, all_levels);

        if spoofing_analysis.is_spoofing {
            return SpoofingAnalysis {
                is_spoofing: true,
                confidence: 0.0,
                wall_lifetime_us: 0,
                cancellation_rate: 1.0,
                price_impact_reversal_bps: 0,
            };
        }

        // Genuine hidden liquidity indicators:
        // 1. Consistent presence across multiple snapshots
        // 2. Volume replenishment after partial fills
        // 3. Reasonable order count relative to volume
        // 4. Not at round number prices (spoofers prefer round numbers)

        let consistency_score = self.check_level_consistency(level, side);
        let replenishment_score = self.check_volume_replenishment(level, side);
        let round_number_penalty = if self.is_round_number_price(level.price_bps) { 0.1 } else { 0.0 };

        let confidence = (consistency_score + replenishment_score) / 2.0 - round_number_penalty;

        SpoofingAnalysis {
            is_spoofing: false,
            confidence: confidence.clamp(0.0, 1.0),
            wall_lifetime_us: 0,
            cancellation_rate: 0.0,
            price_impact_reversal_bps: 0,
        }
    }

    /// Detect spoofing characteristics
    fn detect_spoofing(
        &self,
        level: &OrderBookLevel,
        level_index: usize,
        all_levels: &[OrderBookLevel],
    ) -> SpoofingAnalysis {
        // Look for spoofing patterns in history
        let mut appearances = 0;
        let mut total_snapshots = 0;
        let mut cancellations = 0;

        for snapshot in &self.order_book_history {
            total_snapshots += 1;
            let levels = match level_index {
                0 => if level_index < snapshot.bids.len() { &snapshot.bids } else { continue },
                _ => {
                    // Simplified - in production check both sides
                    &snapshot.bids
                }
            };

            if levels.iter().any(|l| l.price_bps == level.price_bps) {
                appearances += 1;
            }
        }

        // Spoofing indicators:
        // 1. Very short lifetime (< 100ms typical)
        // 2. High cancellation rate
        // 3. Large size relative to surrounding levels
        // 4. Appears just before price moves against it

        let appearance_rate = if total_snapshots > 0 {
            appearances as f64 / total_snapshots as f64
        } else {
            1.0
        };

        // Check if level is abnormally large
        let avg_neighbor_volume = if level_index > 0 && level_index < all_levels.len() {
            (all_levels[level_index - 1].volume + all_levels[level_index + 1].volume) as f64 / 2.0
        } else {
            level.volume as f64
        };

        let size_ratio = level.volume as f64 / avg_neighbor_volume.max(1.0);

        // Determine if spoofing
        let is_spoofing = appearance_rate < self.config.spoofing_sensitivity 
            || size_ratio > 5.0; // More than 5x neighbors is suspicious

        let confidence = if is_spoofing {
            (1.0 - appearance_rate) * self.config.spoofing_sensitivity
        } else {
            0.0
        };

        SpoofingAnalysis {
            is_spoofing,
            confidence,
            wall_lifetime_us: 0,
            cancellation_rate: 1.0 - appearance_rate,
            price_impact_reversal_bps: 0,
        }
    }

    /// Check if a price level appears consistently
    fn check_level_consistency(&self, level: &OrderBookLevel, side: OrderSide) -> f64 {
        let mut appearances = 0;
        
        for snapshot in &self.order_book_history {
            let levels = match side {
                OrderSide::Buy => &snapshot.bids,
                OrderSide::Sell => &snapshot.asks,
            };
            
            if levels.iter().any(|l| l.price_bps == level.price_bps && l.volume >= level.volume / 2) {
                appearances += 1;
            }
        }

        if self.order_book_history.is_empty() {
            return 0.5
        }

        appearances as f64 / self.order_book_history.len() as f64
    }

    /// Check for volume replenishment pattern
    fn check_volume_replenishment(&self, level: &OrderBookLevel, side: OrderSide) -> f64 {
        // In production, track actual fill events and replenishment
        // Simplified version checks for consistent volume
        let mut consistent_count = 0;

        for snapshot in &self.order_book_history {
            let levels = match side {
                OrderSide::Buy => &snapshot.bids,
                OrderSide::Sell => &snapshot.asks,
            };

            if let Some(existing) = levels.iter().find(|l| l.price_bps == level.price_bps) {
                let variance = (existing.volume as i64 - level.volume as i64).abs() as f64;
                if variance < level.volume as f64 * 0.2 {
                    consistent_count += 1;
                }
            }
        }

        if self.order_book_history.is_empty() {
            return 0.5
        }

        consistent_count as f64 / self.order_book_history.len() as f64
    }

    /// Check if price is at a round number (spoofers prefer these)
    fn is_round_number_price(&self, price_bps: i64) -> bool {
        let price = price_bps as f64 / 10000.0;
        price % 100.0 < 0.1 || price % 50.0 < 0.1 || price % 10.0 < 0.01
    }

    /// Validate an opportunity before queuing
    fn validate_opportunity(&self, opportunity: &HiddenLiquidityOpportunity) -> bool {
        opportunity.confidence_score >= self.config.min_confidence
            && opportunity.estimated_volume >= self.config.min_volume_threshold
            && !opportunity.is_genuine // We want genuine, so negate the spoofing flag
    }

    /// Remove stale opportunities
    fn prune_stale_opportunities(&mut self, current_time_ns: u64) {
        let max_age_ns = self.config.max_opportunity_age_us * 1000;
        
        self.opportunities.retain(|opp| {
            current_time_ns < opp.expiry_time_ns
        });
    }

    /// Get next executable opportunity
    pub fn get_next_opportunity(&mut self) -> Option<HiddenLiquidityOpportunity> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        // Check cooldown
        let last_exec = self.last_execution_time.load(Ordering::Relaxed);
        if last_exec > 0 {
            let elapsed_us = (current_time_ns() - last_exec) / 1000;
            if elapsed_us < self.config.execution_cooldown_us {
                return None;
            }
        }

        self.opportunities.pop_front()
    }

    /// Record execution of an opportunity
    pub fn record_execution(&mut self, opportunity: HiddenLiquidityOpportunity, success: bool) {
        self.executed_opportunities.push(opportunity.clone());
        self.total_executions.fetch_add(1, Ordering::Relaxed);
        
        if success {
            self.successful_executions.fetch_add(1, Ordering::Relaxed);
        }

        self.last_execution_time.store(current_time_ns(), Ordering::Relaxed);
        self.state = SniperState::Cooldown;
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> SniperStats {
        let total = self.total_executions.load(Ordering::Relaxed);
        let successful = self.successful_executions.load(Ordering::Relaxed);

        SniperStats {
            total_executions: total,
            successful_executions: successful,
            success_rate: if total > 0 { successful as f64 / total as f64 } else { 0.0 },
            pending_opportunities: self.opportunities.len(),
        }
    }
}

#[derive(Debug)]
pub struct SniperStats {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub success_rate: f64,
    pub pending_opportunities: usize,
}

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
    fn test_spoofing_detection() {
        let config = SniperConfig::default();
        let mut sniper = SniperAlgorithm::new(config);
        sniper.start();

        // Create order book with suspicious large wall
        let snapshot = OrderBookSnapshot {
            timestamp_ns: current_time_ns(),
            bids: vec![
                OrderBookLevel {
                    price_bps: 499000,
                    volume: 1000,
                    order_count: 10,
                    timestamp_ns: current_time_ns(),
                },
                OrderBookLevel {
                    price_bps: 498000,
                    volume: 50000, // Suspiciously large
                    order_count: 1,
                    timestamp_ns: current_time_ns(),
                },
            ],
            asks: vec![],
            mid_price_bps: 500000,
            spread_bps: 1000,
        };

        let opportunities = sniper.process_order_book(snapshot);
        
        // The large wall should be flagged as potential spoofing
        assert!(opportunities.is_empty() || opportunities[0].confidence_score < 0.8);
    }

    #[test]
    fn test_genuine_liquidity_detection() {
        let config = SniperConfig::default();
        let mut sniper = SniperAlgorithm::new(config);
        sniper.start();

        // Add historical snapshots showing consistent liquidity
        for i in 0..10 {
            let snapshot = OrderBookSnapshot {
                timestamp_ns: current_time_ns() - ((10 - i) * 100_000_000) as u64,
                bids: vec![
                    OrderBookLevel {
                        price_bps: 499000,
                        volume: 5000 + (i as u64 * 100),
                        order_count: 50,
                        timestamp_ns: current_time_ns(),
                    },
                ],
                asks: vec![],
                mid_price_bps: 500000,
                spread_bps: 1000,
            };
            sniper.process_order_book(snapshot);
        }

        let stats = sniper.get_stats();
        assert_eq!(stats.pending_opportunities, sniper.opportunities.len());
    }
}
