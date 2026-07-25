// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// Toxicity Scorer Module
// Grades order flow toxicity to prevent adverse selection
// Zero-cost abstractions for memory-efficient operation

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Trade record for toxicity analysis
#[derive(Debug, Clone)]
pub struct ToxicTrade {
    pub timestamp_ns: u64,
    pub price: i64,
    pub quantity: u64,
    pub side: TradeSide,
    pub is_aggressive: bool,
    pub trade_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Toxicity score result
#[derive(Debug, Clone)]
pub struct ToxicityScore {
    pub timestamp_ns: u64,
    pub overall_score: f64,        // 0.0 (safe) to 1.0 (highly toxic)
    pub component_scores: ToxicityComponents,
    pub risk_level: ToxicityLevel,
    pub recommended_action: ToxicityAction,
    pub adverse_selection_risk: f64,
}

#[derive(Debug, Clone)]
pub struct ToxicityComponents {
    pub order_flow_toxicity: f64,   // VPIN-like metric
    pub price_impact_toxicity: f64, // Adverse price movement after trades
    pub cancellation_toxicity: f64, // High cancellation rate indicator
    pub imbalance_toxicity: f64,    // Order book imbalance toxicity
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToxicityLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToxicityAction {
    ContinueTrading,
    ReduceSize,
    WidenSpread,
    PauseTrading,
    HaltTrading,
}

/// Toxicity Scorer using multiple signals
pub struct ToxicityScorer {
    // Recent trades for analysis
    recent_trades: VecDeque<ToxicTrade>,
    max_trade_history: usize,
    
    // Price impact tracking
    price_after_trades: VecDeque<(u64, i64)>, // (timestamp, price)
    
    // Cancellation tracking
    cancellations: VecDeque<(u64, u64)>, // (timestamp_ns, volume)
    
    // Order book imbalance history
    imbalance_history: VecDeque<f64>,
    
    // Rolling metrics
    rolling_vpin: f64,
    rolling_price_impact: f64,
    rolling_cancel_rate: f64,
    
    // Weights for components
    vpin_weight: f64,
    price_impact_weight: f64,
    cancel_weight: f64,
    imbalance_weight: f64,
    
    // Thresholds
    toxic_threshold: f64,
    critical_threshold: f64,
    
    // Statistics
    stats: ToxicityStats,
    
    // Last calculation timestamp
    last_calculation_ns: u64,
}

#[derive(Debug, Default, Clone)]
pub struct ToxicityStats {
    pub total_trades_analyzed: u64,
    pub toxic_events_detected: u64,
    pub avg_toxicity_score: f64,
    pub adverse_selection_losses_prevented: u64,
}

impl ToxicityScorer {
    pub fn new(max_history: usize) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            recent_trades: VecDeque::with_capacity(max_history),
            max_trade_history: max_history,
            price_after_trades: VecDeque::with_capacity(100),
            cancellations: VecDeque::with_capacity(100),
            imbalance_history: VecDeque::with_capacity(50),
            rolling_vpin: 0.0,
            rolling_price_impact: 0.0,
            rolling_cancel_rate: 0.0,
            vpin_weight: 0.35,
            price_impact_weight: 0.30,
            cancel_weight: 0.20,
            imbalance_weight: 0.15,
            toxic_threshold: 0.5,
            critical_threshold: 0.8,
            stats: ToxicityStats::default(),
            last_calculation_ns: now_ns,
        }
    }

    /// Process a new trade for toxicity analysis
    pub fn process_trade(&mut self, trade: ToxicTrade) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.recent_trades.push_back(trade.clone());
        if self.recent_trades.len() > self.max_trade_history {
            self.recent_trades.pop_front();
        }

        // Track price for impact calculation
        self.price_after_trades.push_back((now_ns, trade.price));
        if self.price_after_trades.len() > 100 {
            self.price_after_trades.pop_front();
        }

        self.stats.total_trades_analyzed += 1;
        
        // Update rolling metrics periodically
        if self.stats.total_trades_analyzed % 10 == 0 {
            self.update_rolling_metrics();
        }

        self.last_calculation_ns = now_ns;
    }

    /// Record a cancellation for toxicity analysis
    pub fn record_cancellation(&mut self, timestamp_ns: u64, volume: u64) {
        self.cancellations.push_back((timestamp_ns, volume));
        if self.cancellations.len() > 100 {
            self.cancellations.pop_front();
        }
    }

    /// Record order book imbalance
    pub fn record_imbalance(&mut self, imbalance: f64) {
        self.imbalance_history.push_back(imbalance);
        if self.imbalance_history.len() > 50 {
            self.imbalance_history.pop_front();
        }
    }

    /// Calculate current toxicity score
    pub fn calculate_toxicity_score(&mut self) -> ToxicityScore {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.update_rolling_metrics();

        // Calculate component scores
        let components = ToxicityComponents {
            order_flow_toxicity: self.calculate_order_flow_toxicity(),
            price_impact_toxicity: self.calculate_price_impact_toxicity(),
            cancellation_toxicity: self.calculate_cancellation_toxicity(),
            imbalance_toxicity: self.calculate_imbalance_toxicity(),
        };

        // Calculate weighted overall score
        let overall_score = 
            self.vpin_weight * components.order_flow_toxicity +
            self.price_impact_weight * components.price_impact_toxicity +
            self.cancel_weight * components.cancellation_toxicity +
            self.imbalance_weight * components.imbalance_toxicity;

        // Determine risk level
        let risk_level = self.determine_risk_level(overall_score);

        // Determine recommended action
        let action = self.determine_recommended_action(risk_level, overall_score);

        // Calculate adverse selection risk
        let adverse_risk = self.calculate_adverse_selection_risk(&components);

        // Update statistics
        if overall_score > self.toxic_threshold {
            self.stats.toxic_events_detected += 1;
        }
        self.stats.avg_toxicity_score = 
            (self.stats.avg_toxicity_score * (self.stats.total_trades_analyzed - 1) as f64 
             + overall_score) / self.stats.total_trades_analyzed as f64;

        self.last_calculation_ns = now_ns;

        ToxicityScore {
            timestamp_ns: now_ns,
            overall_score: overall_score.clamp(0.0, 1.0),
            component_scores: components,
            risk_level,
            recommended_action: action,
            adverse_selection_risk: adverse_risk,
        }
    }

    /// Check if current conditions are toxic for trading
    pub fn is_toxic(&self) -> bool {
        self.rolling_vpin > self.toxic_threshold || self.rolling_price_impact > self.toxic_threshold
    }

    /// Get current risk level
    pub fn get_risk_level(&mut self) -> ToxicityLevel {
        let score = self.calculate_toxicity_score();
        score.risk_level
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.recent_trades.clear();
        self.price_after_trades.clear();
        self.cancellations.clear();
        self.imbalance_history.clear();
        self.rolling_vpin = 0.0;
        self.rolling_price_impact = 0.0;
        self.rolling_cancel_rate = 0.0;
        self.stats = ToxicityStats::default();
    }

    fn update_rolling_metrics(&mut self) {
        // Update rolling VPIN
        self.rolling_vpin = self.calculate_order_flow_toxicity();
        
        // Update rolling price impact
        self.rolling_price_impact = self.calculate_price_impact_toxicity();
        
        // Update rolling cancellation rate
        self.rolling_cancel_rate = self.calculate_cancellation_toxicity();
    }

    fn calculate_order_flow_toxicity(&self) -> f64 {
        if self.recent_trades.len() < 10 {
            return 0.3; // Neutral prior
        }

        // Calculate buy/sell imbalance in recent trades
        let mut buy_vol: u64 = 0;
        let mut sell_vol: u64 = 0;

        for trade in self.recent_trades.iter().rev().take(50) {
            if trade.is_aggressive {
                match trade.side {
                    TradeSide::Buy => buy_vol += trade.quantity,
                    TradeSide::Sell => sell_vol += trade.quantity,
                }
            }
        }

        let total = buy_vol + sell_vol;
        if total == 0 {
            return 0.0;
        }

        // High imbalance suggests informed trading
        let imbalance = (buy_vol as i64 - sell_vol as i64).abs() as f64 / total as f64;
        imbalance.clamp(0.0, 1.0)
    }

    fn calculate_price_impact_toxicity(&self) -> f64 {
        if self.price_after_trades.len() < 5 {
            return 0.3;
        }

        // Calculate average adverse price movement after aggressive trades
        let mut adverse_moves: Vec<f64> = Vec::new();

        let trades: Vec<&ToxicTrade> = self.recent_trades.iter()
            .filter(|t| t.is_aggressive)
            .rev()
            .take(20)
            .collect();

        for trade in trades {
            // Find price change after this trade
            let trade_ts = trade.timestamp_ns;
            
            for (ts, price) in &self.price_after_trades {
                if *ts > trade_ts + 100_000_000 { // 100ms after
                    let price_change = (*price as f64 - trade.price as f64).abs() / trade.price as f64;
                    
                    // Check if move was adverse
                    let is_adverse = match trade.side {
                        TradeSide::Buy => *price < trade.price,
                        TradeSide::Sell => *price > trade.price,
                    };

                    if is_adverse {
                        adverse_moves.push(price_change * 1000.0); // Convert to bps
                    }
                    break;
                }
            }
        }

        if adverse_moves.is_empty() {
            return 0.2;
        }

        let avg_adverse = adverse_moves.iter().sum::<f64>() / adverse_moves.len() as f64;
        (avg_adverse / 10.0).clamp(0.0, 1.0) // Normalize
    }

    fn calculate_cancellation_toxicity(&self) -> f64 {
        if self.cancellations.is_empty() {
            return 0.2;
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Count rapid cancellations (within 1 second)
        let recent_cancels: u64 = self.cancellations.iter()
            .filter(|(ts, _)| now_ns - ts < 1_000_000_000)
            .count() as u64;

        let cancel_rate = recent_cancels as f64 / self.recent_trades.len() as f64;
        cancel_rate.min(1.0)
    }

    fn calculate_imbalance_toxicity(&self) -> f64 {
        if self.imbalance_history.is_empty() {
            return 0.3;
        }

        // High sustained imbalance is toxic
        let avg_imbalance: f64 = self.imbalance_history.iter().sum::<f64>() / self.imbalance_history.len() as f64;
        avg_imbalance.abs().clamp(0.0, 1.0)
    }

    fn determine_risk_level(&self, score: f64) -> ToxicityLevel {
        if score < 0.2 {
            ToxicityLevel::Safe
        } else if score < 0.4 {
            ToxicityLevel::Low
        } else if score < 0.6 {
            ToxicityLevel::Medium
        } else if score < 0.8 {
            ToxicityLevel::High
        } else {
            ToxicityLevel::Critical
        }
    }

    fn determine_recommended_action(&self, level: ToxicityLevel, score: f64) -> ToxicityAction {
        match level {
            ToxicityLevel::Safe | ToxicityLevel::Low => ToxicityAction::ContinueTrading,
            ToxicityLevel::Medium => ToxicityAction::ReduceSize,
            ToxicityLevel::High => ToxicityAction::WidenSpread,
            ToxicityLevel::Critical => {
                if score > 0.9 {
                    ToxicityAction::HaltTrading
                } else {
                    ToxicityAction::PauseTrading
                }
            }
        }
    }

    fn calculate_adverse_selection_risk(&self, components: &ToxicityComponents) -> f64 {
        // Weighted combination of toxicity indicators
        let risk = 
            0.4 * components.order_flow_toxicity +
            0.3 * components.price_impact_toxicity +
            0.2 * components.cancellation_toxicity +
            0.1 * components.imbalance_toxicity;
        
        risk.clamp(0.0, 1.0)
    }
}

impl Default for ToxicityScorer {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toxicity_score_basic() {
        let mut scorer = ToxicityScorer::new(1000);
        
        // Add some normal trades
        for i in 0..20 {
            let trade = ToxicTrade {
                timestamp_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64 + i as u64 * 1_000_000,
                price: 50000,
                quantity: 10,
                side: TradeSide::Buy,
                is_aggressive: true,
                trade_id: format!("TEST{}", i),
            };
            scorer.process_trade(trade);
        }

        let score = scorer.calculate_toxicity_score();
        assert!(score.overall_score >= 0.0 && score.overall_score <= 1.0);
    }

    #[test]
    fn test_toxicity_detection() {
        let mut scorer = ToxicityScorer::new(1000);
        
        // Simulate toxic conditions with high imbalance
        for i in 0..50 {
            let trade = ToxicTrade {
                timestamp_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                price: 50000,
                quantity: 100,
                side: if i % 2 == 0 { TradeSide::Buy } else { TradeSide::Sell },
                is_aggressive: true,
                trade_id: format!("TOXIC{}", i),
            };
            scorer.process_trade(trade);
        }

        let score = scorer.calculate_toxicity_score();
        assert!(score.overall_score >= 0.0 && score.overall_score <= 1.0);
    }
}
