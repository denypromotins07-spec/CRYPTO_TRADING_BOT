// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// Order Book Imbalance Module
// Tracks aggressive bid/ask volume for short-term alpha signals
// Zero-cost abstractions for memory-efficient operation

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// Trade record with aggression information
#[derive(Debug, Clone)]
pub struct AggressiveTrade {
    pub timestamp_ns: u64,
    pub price: i64,
    pub quantity: u64,
    pub side: Side,
    pub is_aggressive: bool,
    pub trade_id: String,
}

/// Order book imbalance metrics
#[derive(Debug, Clone)]
pub struct ImbalanceMetrics {
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub aggressive_bid_volume: u64,
    pub aggressive_ask_volume: u64,
    pub imbalance_ratio: f64,
    pub aggressive_ratio: f64,
    pub order_flow_imbalance: f64,
    pub tick_imbalance: i64,
}

/// Short-term alpha signal from imbalance
#[derive(Debug, Clone)]
pub struct AlphaSignal {
    pub timestamp_ns: u64,
    pub signal_strength: f64,
    pub signal_type: SignalType,
    pub confidence: f64,
    pub recommended_action: TradingAction,
    pub expected_move_bps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalType {
    BullishImbalance,
    BearishImbalance,
    Neutral,
    Reversal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingAction {
    Buy,
    Sell,
    Hold,
    ReduceExposure,
}

/// Order Book Imbalance Tracker using Observer pattern
pub struct OrderImbalanceTracker {
    bid_levels: HashMap<i64, u64>,
    ask_levels: HashMap<i64, u64>,
    aggressive_trades: VecDeque<AggressiveTrade>,
    max_trade_history: usize,
    cumulative_bid_delta: i64,
    cumulative_ask_delta: i64,
    last_trade_price: Option<i64>,
    last_tick_direction: i8,
    tick_imbalance_sum: i64,
    volume_buckets: VecDeque<(u64, u64)>,
    bucket_duration_ms: u64,
    last_bucket_time_ns: u64,
    current_metrics: ImbalanceMetrics,
    stats: ImbalanceStats,
    last_update_ns: u64,
}

#[derive(Debug, Default, Clone)]
pub struct ImbalanceStats {
    pub total_trades_processed: u64,
    pub aggressive_trades_count: u64,
    pub total_bid_volume: u64,
    pub total_ask_volume: u64,
    pub avg_imbalance_ratio: f64,
    pub extreme_imbalance_events: u64,
}

impl OrderImbalanceTracker {
    pub fn new(max_history: usize, bucket_duration_ms: u64) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            bid_levels: HashMap::with_capacity(50),
            ask_levels: HashMap::with_capacity(50),
            aggressive_trades: VecDeque::with_capacity(max_history),
            max_trade_history: max_history,
            cumulative_bid_delta: 0,
            cumulative_ask_delta: 0,
            last_trade_price: None,
            last_tick_direction: 0,
            tick_imbalance_sum: 0,
            volume_buckets: VecDeque::with_capacity(60),
            bucket_duration_ms,
            last_bucket_time_ns: now_ns,
            current_metrics: ImbalanceMetrics {
                bid_volume: 0,
                ask_volume: 0,
                aggressive_bid_volume: 0,
                aggressive_ask_volume: 0,
                imbalance_ratio: 0.0,
                aggressive_ratio: 0.5,
                order_flow_imbalance: 0.0,
                tick_imbalance: 0,
            },
            stats: ImbalanceStats::default(),
            last_update_ns: now_ns,
        }
    }

    pub fn process_book_update(&mut self, price: i64, quantity: u64, side: Side, is_addition: bool) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.maybe_rotate_bucket(now_ns);

        let levels = match side {
            Side::Bid => &mut self.bid_levels,
            Side::Ask => &mut self.ask_levels,
        };

        let current_vol = levels.entry(price).or_insert(0);
        
        if is_addition {
            *current_vol += quantity;
            if side == Side::Bid {
                self.cumulative_bid_delta += quantity as i64;
            } else {
                self.cumulative_ask_delta += quantity as i64;
            }
        } else {
            *current_vol = current_vol.saturating_sub(quantity);
        }

        self.update_metrics();
        self.last_update_ns = now_ns;
    }

    pub fn process_aggressive_trade(&mut self, trade: AggressiveTrade) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.maybe_rotate_bucket(now_ns);

        if let Some(last_price) = self.last_trade_price {
            if trade.price > last_price {
                self.last_tick_direction = 1;
            } else if trade.price < last_price {
                self.last_tick_direction = -1;
            }
            self.tick_imbalance_sum += self.last_tick_direction as i64;
        }
        self.last_trade_price = Some(trade.price);

        self.aggressive_trades.push_back(trade.clone());
        if self.aggressive_trades.len() > self.max_trade_history {
            self.aggressive_trades.pop_front();
        }

        match trade.side {
            Side::Bid => self.cumulative_bid_delta -= trade.quantity as i64,
            Side::Ask => self.cumulative_ask_delta -= trade.quantity as i64,
        }

        self.update_metrics();
        self.last_update_ns = now_ns;
        self.stats.total_trades_processed += 1;
        self.stats.aggressive_trades_count += 1;
    }

    pub fn get_metrics(&self) -> &ImbalanceMetrics {
        &self.current_metrics
    }

    pub fn calculate_alpha_signal(&self, lookback_trades: usize) -> AlphaSignal {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut aggressive_buy_vol: u64 = 0;
        let mut aggressive_sell_vol: u64 = 0;

        for trade in self.aggressive_trades.iter().rev().take(lookback_trades) {
            match trade.side {
                Side::Bid => aggressive_buy_vol += trade.quantity,
                Side::Ask => aggressive_sell_vol += trade.quantity,
            }
        }

        let total_aggressive = aggressive_buy_vol + aggressive_sell_vol;
        let order_flow_imbalance = if total_aggressive > 0 {
            (aggressive_buy_vol as f64 - aggressive_sell_vol as f64) / total_aggressive as f64
        } else {
            0.0
        };

        let combined_signal = 0.6 * order_flow_imbalance + 0.4 * self.current_metrics.imbalance_ratio;

        let (signal_type, signal_strength) = if combined_signal > 0.3 {
            (SignalType::BullishImbalance, combined_signal.min(1.0))
        } else if combined_signal < -0.3 {
            (SignalType::BearishImbalance, combined_signal.max(-1.0))
        } else {
            (SignalType::Neutral, combined_signal)
        };

        let is_reversal = self.detect_reversal_pattern();
        let final_signal_type = if is_reversal { SignalType::Reversal } else { signal_type };
        let confidence = self.calculate_signal_confidence(lookback_trades);
        let expected_move_bps = signal_strength.abs() * 10.0;

        let action = match final_signal_type {
            SignalType::BullishImbalance => TradingAction::Buy,
            SignalType::BearishImbalance => TradingAction::Sell,
            SignalType::Reversal => TradingAction::ReduceExposure,
            SignalType::Neutral => TradingAction::Hold,
        };

        AlphaSignal {
            timestamp_ns: now_ns,
            signal_strength,
            signal_type: final_signal_type,
            confidence,
            recommended_action: action,
            expected_move_bps,
        }
    }

    pub fn get_cumulative_delta(&self) -> i64 {
        self.cumulative_bid_delta - self.cumulative_ask_delta
    }

    pub fn clear(&mut self) {
        self.bid_levels.clear();
        self.ask_levels.clear();
        self.aggressive_trades.clear();
        self.volume_buckets.clear();
        self.cumulative_bid_delta = 0;
        self.cumulative_ask_delta = 0;
        self.last_trade_price = None;
        self.last_tick_direction = 0;
        self.tick_imbalance_sum = 0;
        self.stats = ImbalanceStats::default();
    }

    fn update_metrics(&mut self) {
        let total_bid: u64 = self.bid_levels.values().sum();
        let total_ask: u64 = self.ask_levels.values().sum();
        
        let aggressive_bid: u64 = self.aggressive_trades.iter()
            .filter(|t| t.side == Side::Bid && t.is_aggressive)
            .map(|t| t.quantity)
            .sum();
        let aggressive_ask: u64 = self.aggressive_trades.iter()
            .filter(|t| t.side == Side::Ask && t.is_aggressive)
            .map(|t| t.quantity)
            .sum();

        let sum = total_bid + total_ask;
        let imbalance_ratio = if sum > 0 {
            (total_bid as f64 - total_ask as f64) / sum as f64
        } else {
            0.0
        };

        let aggressive_sum = aggressive_bid + aggressive_ask;
        let aggressive_ratio = if aggressive_sum > 0 {
            aggressive_bid as f64 / aggressive_sum as f64
        } else {
            0.5
        };

        self.current_metrics = ImbalanceMetrics {
            bid_volume: total_bid,
            ask_volume: total_ask,
            aggressive_bid_volume: aggressive_bid,
            aggressive_ask_volume: aggressive_ask,
            imbalance_ratio,
            aggressive_ratio,
            order_flow_imbalance: aggressive_ratio * 2.0 - 1.0,
            tick_imbalance: self.last_tick_direction as i64,
        };
    }

    fn maybe_rotate_bucket(&mut self, now_ns: u64) {
        let bucket_duration_ns = self.bucket_duration_ms * 1_000_000;
        
        if now_ns - self.last_bucket_time_ns >= bucket_duration_ns {
            let bucket_bid: u64 = self.bid_levels.values().sum();
            let bucket_ask: u64 = self.ask_levels.values().sum();
            
            self.volume_buckets.push_back((bucket_bid, bucket_ask));
            
            if self.volume_buckets.len() > 60 {
                self.volume_buckets.pop_front();
            }
            
            self.last_bucket_time_ns = now_ns;
        }
    }

    fn detect_reversal_pattern(&self) -> bool {
        if self.aggressive_trades.len() < 10 {
            return false;
        }

        let recent_imbalance = self.current_metrics.imbalance_ratio;
        let mut counter_aggression: u64 = 0;
        let mut total_recent: u64 = 0;

        for trade in self.aggressive_trades.iter().rev().take(20) {
            total_recent += trade.quantity;
            
            let is_counter = if recent_imbalance > 0.5 {
                trade.side == Side::Ask
            } else if recent_imbalance < -0.5 {
                trade.side == Side::Bid
            } else {
                false
            };

            if is_counter {
                counter_aggression += trade.quantity;
            }
        }

        if total_recent > 0 {
            let counter_ratio = counter_aggression as f64 / total_recent as f64;
            return counter_ratio > 0.6;
        }

        false
    }

    fn calculate_signal_confidence(&self, lookback_trades: usize) -> f64 {
        let mut confidence = 0.0;

        let data_factor = (self.aggressive_trades.len() as f64 / lookback_trades as f64).min(1.0);
        confidence += 0.4 * data_factor;

        if !self.aggressive_trades.is_empty() {
            let volumes: Vec<u64> = self.aggressive_trades.iter()
                .take(lookback_trades)
                .map(|t| t.quantity)
                .collect();
            
            if volumes.len() > 1 {
                let mean_vol = volumes.iter().sum::<u64>() as f64 / volumes.len() as f64;
                let variance: f64 = volumes.iter()
                    .map(|v| (*v as f64 - mean_vol).powi(2))
                    .sum::<f64>() / volumes.len() as f64;
                
                let cv = variance.sqrt() / mean_vol;
                let consistency_factor = (1.0 - cv.min(1.0)).max(0.0);
                confidence += 0.3 * consistency_factor;
            }
        }

        confidence.clamp(0.0, 1.0)
    }
}

impl Default for OrderImbalanceTracker {
    fn default() -> Self {
        Self::new(1000, 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imbalance_calculation() {
        let mut tracker = OrderImbalanceTracker::new(1000, 1000);
        
        tracker.process_book_update(50000, 100, Side::Bid, true);
        tracker.process_book_update(49999, 200, Side::Bid, true);
        tracker.process_book_update(50001, 50, Side::Ask, true);
        
        let metrics = tracker.get_metrics();
        assert!(metrics.imbalance_ratio > 0.0);
    }

    #[test]
    fn test_aggressive_trade_processing() {
        let mut tracker = OrderImbalanceTracker::new(1000, 1000);
        
        let trade = AggressiveTrade {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            price: 50000,
            quantity: 50,
            side: Side::Bid,
            is_aggressive: true,
            trade_id: "TEST1".to_string(),
        };
        
        tracker.process_aggressive_trade(trade);
        assert_eq!(tracker.stats.aggressive_trades_count, 1);
    }
}
