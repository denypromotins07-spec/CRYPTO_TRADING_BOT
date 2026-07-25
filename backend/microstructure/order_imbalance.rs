// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// File: backend/microstructure/order_imbalance.rs
// Chapter 3: Order Book Imbalance, VPIN, and Toxicity Metrics
//
// Purpose: Track aggressive bid/ask volume for short-term alpha signals
// Constraints: Zero-cost abstractions, minimal latency for HFT-style signals
// Target: AMD Ryzen AI 5 laptop with 8GB RAM limit
//
// Design Patterns: Observer Pattern for imbalance notifications, Strategy for calculation
// Memory Model: Pre-allocated circular buffers, stack-based computation

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Maximum number of trades to track in sliding window
const MAX_TRADE_HISTORY: usize = 10_000;

/// Default time window for imbalance calculation (in milliseconds)
const DEFAULT_WINDOW_MS: u64 = 1000;

#[derive(Debug, Clone)]
pub struct OrderFlowImbalance {
    pub net_imbalance: i64,
    pub normalized_imbalance: f64,
    pub bid_volume: i64,
    pub ask_volume: i64,
    pub trade_count: usize,
    pub timestamp_us: u64,
}

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp_us: u64,
    pub price: i64,
    pub quantity: i64,
    pub aggressor_side: AggressorSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggressorSide {
    Buyer,
    Seller,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ImbalanceAlert {
    pub alert_id: u64,
    pub timestamp_us: u64,
    pub imbalance_value: f64,
    pub signal_direction: SignalDirection,
    pub recommended_action: ImbalanceAction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalDirection {
    Bullish,
    Bearish,
    Neutral,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImbalanceAction {
    ConsiderLong,
    ConsiderShort,
    ReduceExposure,
    WidenSpreads,
    Monitor,
}

pub struct OrderImbalanceTracker {
    trade_history: VecDeque<TradeRecord>,
    running_bid_volume: i64,
    running_ask_volume: i64,
    sequence_number: AtomicU64,
    start_time: Instant,
    window_ms: u64,
    high_threshold: f64,
    enabled: AtomicBool,
    recent_alerts: VecDeque<ImbalanceAlert>,
}

impl OrderImbalanceTracker {
    pub fn new() -> Self {
        Self {
            trade_history: VecDeque::with_capacity(MAX_TRADE_HISTORY),
            running_bid_volume: 0,
            running_ask_volume: 0,
            sequence_number: AtomicU64::new(0),
            start_time: Instant::now(),
            window_ms: DEFAULT_WINDOW_MS,
            high_threshold: 0.7,
            enabled: AtomicBool::new(true),
            recent_alerts: VecDeque::with_capacity(100),
        }
    }
    
    fn now_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }
    
    pub fn record_trade(&mut self, price: i64, quantity: i64, aggressor_side: AggressorSide) -> Option<ImbalanceAlert> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }
        
        let timestamp_us = self.now_us();
        
        let trade = TradeRecord {
            timestamp_us,
            price,
            quantity,
            aggressor_side,
        };
        
        self.trade_history.push_back(trade);
        
        match aggressor_side {
            AggressorSide::Buyer => self.running_bid_volume += quantity,
            AggressorSide::Seller => self.running_ask_volume += quantity,
            _ => {}
        }
        
        self.prune_old_trades(timestamp_us);
        self.check_imbalance_threshold()
    }
    
    fn prune_old_trades(&mut self, current_time_us: u64) {
        let cutoff_us = current_time_us.saturating_sub(self.window_ms * 1000);
        
        while let Some(front) = self.trade_history.front() {
            if front.timestamp_us < cutoff_us {
                match front.aggressor_side {
                    AggressorSide::Buyer => self.running_bid_volume -= front.quantity,
                    AggressorSide::Seller => self.running_ask_volume -= front.quantity,
                    _ => {}
                }
                self.trade_history.pop_front();
            } else {
                break;
            }
        }
    }
    
    pub fn calculate_imbalance(&self) -> OrderFlowImbalance {
        let total_volume = self.running_bid_volume + self.running_ask_volume;
        let net_imbalance = self.running_bid_volume - self.running_ask_volume;
        
        let normalized_imbalance = if total_volume > 0 {
            (self.running_bid_volume as f64 - self.running_ask_volume as f64) / total_volume as f64
        } else {
            0.0
        };
        
        OrderFlowImbalance {
            net_imbalance,
            normalized_imbalance,
            bid_volume: self.running_bid_volume.max(0),
            ask_volume: self.running_ask_volume.max(0),
            trade_count: self.trade_history.len(),
            timestamp_us: self.now_us(),
        }
    }
    
    fn check_imbalance_threshold(&mut self) -> Option<ImbalanceAlert> {
        let imbalance = self.calculate_imbalance();
        let abs_imbalance = imbalance.normalized_imbalance.abs();
        
        if abs_imbalance < self.high_threshold {
            return None;
        }
        
        let direction = if imbalance.normalized_imbalance > 0 {
            SignalDirection::Bullish
        } else {
            SignalDirection::Bearish
        };
        
        let action = match direction {
            SignalDirection::Bullish => {
                if abs_imbalance > 0.9 { ImbalanceAction::ConsiderLong } 
                else { ImbalanceAction::WidenSpreads }
            }
            SignalDirection::Bearish => {
                if abs_imbalance > 0.9 { ImbalanceAction::ConsiderShort } 
                else { ImbalanceAction::ReduceExposure }
            }
            SignalDirection::Neutral => ImbalanceAction::Monitor,
        };
        
        let alert = ImbalanceAlert {
            alert_id: self.sequence_number.fetch_add(1, Ordering::SeqCst),
            timestamp_us: self.now_us(),
            imbalance_value: imbalance.normalized_imbalance,
            signal_direction: direction,
            recommended_action: action,
        };
        
        self.recent_alerts.push_back(alert.clone());
        Some(alert)
    }
    
    pub fn get_cumulative_delta(&self) -> i64 {
        self.running_bid_volume - self.running_ask_volume
    }
    
    pub fn get_statistics(&self) -> ImbalanceStatistics {
        ImbalanceStatistics {
            total_trades: self.trade_history.len(),
            total_bid_volume: self.running_bid_volume.max(0),
            total_ask_volume: self.running_ask_volume.max(0),
            current_imbalance: self.calculate_imbalance().normalized_imbalance,
            alerts_generated: self.recent_alerts.len(),
        }
    }
}

impl Default for OrderImbalanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ImbalanceStatistics {
    pub total_trades: usize,
    pub total_bid_volume: i64,
    pub total_ask_volume: i64,
    pub current_imbalance: f64,
    pub alerts_generated: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imbalance_calculation() {
        let mut tracker = OrderImbalanceTracker::new();
        
        for _ in 0..10 {
            tracker.record_trade(50000, 100, AggressorSide::Buyer);
        }
        for _ in 0..3 {
            tracker.record_trade(50001, 100, AggressorSide::Seller);
        }
        
        let imbalance = tracker.calculate_imbalance();
        assert!(imbalance.normalized_imbalance > 0.5);
    }
}
