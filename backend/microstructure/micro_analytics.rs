//! Microstructure Analytics - Historical Queue Analysis
//!
//! This module profiles historical queue cancellation rates, fill patterns,
//! and other microstructure metrics for strategy optimization.
//!
//! Designed for the ZAID PERSONAL CRYPTO TRADING BOT with 8GB RAM constraints.
//! Uses efficient data structures for historical analysis.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;

/// Maximum history entries to retain
const MAX_HISTORY_ENTRIES: usize = 10000;

/// Default analysis window in milliseconds
const DEFAULT_WINDOW_MS: u64 = 60000;

/// Price tick type
pub type PriceTick = i64;

/// Volume type
pub type Volume = u64;

/// Side of the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// Queue event types
#[derive(Debug, Clone, Copy)]
pub enum QueueEventType {
    OrderPlaced,
    OrderCancelled,
    OrderFilled,
    OrderModified,
    QueueDecay,
}

/// A single queue event record
#[derive(Debug, Clone, Copy)]
pub struct QueueEventRecord {
    pub timestamp_us: u64,
    pub event_type: QueueEventType,
    pub price: PriceTick,
    pub volume: Volume,
    pub queue_position: u32,
    pub time_to_event_us: Option<u64>, // Time from placement to this event
}

/// Cancellation statistics for a price level
#[derive(Debug, Clone, Default)]
pub struct CancellationStats {
    /// Total orders placed
    pub total_orders: u64,
    /// Orders cancelled before fill
    pub cancelled_orders: u64,
    /// Orders fully filled
    pub filled_orders: u64,
    /// Average time to cancellation (microseconds)
    pub avg_cancel_time_us: f64,
    /// Average time to fill (microseconds)
    pub avg_fill_time_us: f64,
    /// Cancellation rate [0, 1]
    pub cancel_rate: f64,
    /// Fill rate [0, 1]
    pub fill_rate: f64,
}

impl CancellationStats {
    #[inline]
    pub fn update_cancel(&mut self, time_to_cancel_us: u64) {
        self.cancelled_orders += 1;
        // Running average
        let n = self.cancelled_orders as f64;
        self.avg_cancel_time_us += (time_to_cancel_us as f64 - self.avg_cancel_time_us) / n;
        self.cancel_rate = self.cancelled_orders as f64 / self.total_orders.max(1) as f64;
    }

    #[inline]
    pub fn update_fill(&mut self, time_to_fill_us: u64) {
        self.filled_orders += 1;
        let n = self.filled_orders as f64;
        self.avg_fill_time_us += (time_to_fill_us as f64 - self.avg_fill_time_us) / n;
        self.fill_rate = self.filled_orders as f64 / self.total_orders.max(1) as f64;
    }

    #[inline]
    pub fn add_order(&mut self) {
        self.total_orders += 1;
    }
}

/// Historical analytics snapshot
#[derive(Debug, Clone)]
pub struct AnalyticsSnapshot {
    pub timestamp_us: u64,
    pub symbol: String,
    /// Overall cancellation rate
    pub overall_cancel_rate: f64,
    /// Overall fill rate
    pub overall_fill_rate: f64,
    /// Average queue decay rate
    pub avg_decay_rate: f64,
    /// Spoofing detection rate
    pub spoofing_rate: f64,
    /// Average time to fill
    pub avg_time_to_fill_ms: f64,
    /// Volume-weighted fill probability
    pub vw_fill_probability: f64,
}

/// Microstructure analytics engine
pub struct MicroAnalytics {
    /// Symbol being analyzed
    symbol: String,
    /// Event history
    events: RwLock<VecDeque<QueueEventRecord>>,
    /// Cancellation stats per price level
    cancel_stats: RwLock<HashMap<PriceTick, CancellationStats>>,
    /// Active orders (placement time)
    active_orders: RwLock<HashMap<u64, (PriceTick, u64, Volume)>>, // order_id -> (price, time, volume)
    /// Recent decay rates
    decay_rates: RwLock<VecDeque<f64>>,
    /// Detected spoofing events
    spoofing_events: RwLock<VecDeque<u64>>,
    /// Analysis window
    window_ms: u64,
    /// Last snapshot
    last_snapshot: RwLock<Option<AnalyticsSnapshot>>,
    /// Last update
    last_update: RwLock<Instant>,
}

/// Get current timestamp in microseconds
#[inline]
pub fn get_timestamp_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

impl MicroAnalytics {
    /// Create a new analytics engine
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            events: RwLock::new(VecDeque::with_capacity(MAX_HISTORY_ENTRIES)),
            cancel_stats: RwLock::new(HashMap::with_capacity(500)),
            active_orders: RwLock::new(HashMap::with_capacity(1000)),
            decay_rates: RwLock::new(VecDeque::with_capacity(1000)),
            spoofing_events: RwLock::new(VecDeque::with_capacity(100)),
            window_ms: DEFAULT_WINDOW_MS,
            last_snapshot: RwLock::new(None),
            last_update: RwLock::new(Instant::now()),
        }
    }

    /// Record an order placement
    pub fn record_placement(&self, order_id: u64, price: PriceTick, volume: Volume) {
        let now = get_timestamp_us();
        
        self.active_orders.write().insert(order_id, (price, now, volume));
        
        if let Some(stats) = self.cancel_stats.write().get_mut(&price) {
            stats.add_order();
        } else {
            let mut stats = CancellationStats::default();
            stats.add_order();
            self.cancel_stats.write().insert(price, stats);
        }

        self.record_event(QueueEventRecord {
            timestamp_us: now,
            event_type: QueueEventType::OrderPlaced,
            price,
            volume,
            queue_position: 0,
            time_to_event_us: None,
        });
    }

    /// Record an order cancellation
    pub fn record_cancellation(&self, order_id: u64, is_spoof: bool) {
        let now = get_timestamp_us();
        
        if let Some((price, placed_at, volume)) = self.active_orders.write().remove(&order_id) {
            let time_to_cancel = now - placed_at;
            
            if let Some(stats) = self.cancel_stats.write().get_mut(&price) {
                stats.update_cancel(time_to_cancel);
            }

            if is_spoof {
                self.spoofing_events.write().push_back(now);
                while self.spoofing_events.read().len() > 100 {
                    self.spoofing_events.write().pop_front();
                }
            }

            self.record_event(QueueEventRecord {
                timestamp_us: now,
                event_type: QueueEventType::OrderCancelled,
                price,
                volume,
                queue_position: 0,
                time_to_event_us: Some(time_to_cancel),
            });
        }
    }

    /// Record an order fill
    pub fn record_fill(&self, order_id: u64, fill_volume: Volume) {
        let now = get_timestamp_us();
        
        if let Some((price, placed_at, _volume)) = self.active_orders.write().remove(&order_id) {
            let time_to_fill = now - placed_at;
            
            if let Some(stats) = self.cancel_stats.write().get_mut(&price) {
                stats.update_fill(time_to_fill);
            }

            self.record_event(QueueEventRecord {
                timestamp_us: now,
                event_type: QueueEventType::OrderFilled,
                price,
                volume: fill_volume,
                queue_position: 0,
                time_to_event_us: Some(time_to_fill),
            });
        }
    }

    /// Record a queue decay observation
    pub fn record_decay_rate(&self, rate: f64) {
        self.decay_rates.write().push_back(rate);
        while self.decay_rates.read().len() > 1000 {
            self.decay_rates.write().pop_front();
        }
    }

    /// Record a queue event
    fn record_event(&self, event: QueueEventRecord) {
        self.events.write().push_back(event);
        while self.events.read().len() > MAX_HISTORY_ENTRIES {
            self.events.write().pop_front();
        }
        
        *self.last_update.write() = Instant::now();
        
        // Update snapshot periodically
        self.update_snapshot();
    }

    /// Update the analytics snapshot
    fn update_snapshot(&self) {
        let now = get_timestamp_us();
        let window_us = self.window_ms * 1000;

        // Calculate overall stats
        let stats = self.cancel_stats.read();
        let mut total_orders = 0u64;
        let mut total_cancelled = 0u64;
        let mut total_filled = 0u64;
        let mut weighted_fill_time = 0.0;
        let mut total_fill_weight = 0.0;

        for (_, s) in stats.iter() {
            total_orders += s.total_orders;
            total_cancelled += s.cancelled_orders;
            total_filled += s.filled_orders;
            
            if s.filled_orders > 0 {
                let weight = s.filled_orders as f64;
                weighted_fill_time += s.avg_fill_time_us * weight;
                total_fill_weight += weight;
            }
        }

        let overall_cancel_rate = if total_orders > 0 {
            total_cancelled as f64 / total_orders as f64
        } else {
            0.0
        };

        let overall_fill_rate = if total_orders > 0 {
            total_filled as f64 / total_orders as f64
        } else {
            0.0
        };

        let avg_time_to_fill_ms = if total_fill_weight > 0 {
            weighted_fill_time / total_fill_weight / 1000.0
        } else {
            0.0
        };

        // Average decay rate
        let decay_rates = self.decay_rates.read();
        let avg_decay_rate = if !decay_rates.is_empty() {
            decay_rates.iter().sum::<f64>() / decay_rates.len() as f64
        } else {
            0.0
        };

        // Spoofing rate (spoofing events per minute)
        let spoofing_events = self.spoofing_events.read();
        let recent_spoofs = spoofing_events
            .iter()
            .filter(|&&t| (now - t) / 1000 < self.window_ms)
            .count();
        let spoofing_rate = recent_spoofs as f64 / (self.window_ms as f64 / 60000.0);

        // VW fill probability
        let vw_fill_prob = overall_fill_rate * (1.0 - overall_cancel_rate);

        let snapshot = AnalyticsSnapshot {
            timestamp_us: now,
            symbol: self.symbol.clone(),
            overall_cancel_rate,
            overall_fill_rate,
            avg_decay_rate,
            spoofing_rate,
            avg_time_to_fill_ms,
            vw_fill_probability: vw_fill_prob,
        };

        *self.last_snapshot.write() = Some(snapshot);
    }

    /// Get the latest analytics snapshot
    pub fn get_snapshot(&self) -> Option<AnalyticsSnapshot> {
        self.last_snapshot.read().clone()
    }

    /// Get cancellation stats for a price level
    pub fn get_level_stats(&self, price: PriceTick) -> Option<CancellationStats> {
        self.cancel_stats.read().get(&price).cloned()
    }

    /// Get overall cancellation rate
    pub fn get_cancel_rate(&self) -> f64 {
        if let Some(snapshot) = self.last_snapshot.read().as_ref() {
            snapshot.overall_cancel_rate
        } else {
            0.0
        }
    }

    /// Get overall fill rate
    pub fn get_fill_rate(&self) -> f64 {
        if let Some(snapshot) = self.last_snapshot.read().as_ref() {
            snapshot.overall_fill_rate
        } else {
            0.0
        }
    }

    /// Get average time to fill in milliseconds
    pub fn get_avg_time_to_fill_ms(&self) -> f64 {
        if let Some(snapshot) = self.last_snapshot.read().as_ref() {
            snapshot.avg_time_to_fill_ms
        } else {
            0.0
        }
    }

    /// Get spoofing rate (events per minute)
    pub fn get_spoofing_rate(&self) -> f64 {
        if let Some(snapshot) = self.last_snapshot.read().as_ref() {
            snapshot.spoofing_rate
        } else {
            0.0
        }
    }

    /// Get expected fill probability for a queue position
    pub fn get_fill_probability(&self, position: u32, decay_rate: f64) -> f64 {
        // Probability decreases with position and increases with decay
        let position_factor = 1.0 / (1.0 + position as f64 * 0.1);
        let decay_factor = decay_rate.min(1.0);
        
        let base_rate = self.get_fill_rate();
        let cancel_rate = self.get_cancel_rate();
        
        base_rate * position_factor * decay_factor * (1.0 - cancel_rate * 0.5)
    }

    /// Set analysis window
    pub fn set_window_ms(&self, window_ms: u64) {
        self.window_ms = window_ms;
    }

    /// Get symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Reset all data
    pub fn reset(&self) {
        self.events.write().clear();
        self.cancel_stats.write().clear();
        self.active_orders.write().clear();
        self.decay_rates.write().clear();
        self.spoofing_events.write().clear();
        *self.last_snapshot.write() = None;
        *self.last_update.write() = Instant::now();
    }
}

/// Builder for MicroAnalytics
pub struct MicroAnalyticsBuilder {
    symbol: String,
    window_ms: u64,
}

impl MicroAnalyticsBuilder {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            window_ms: DEFAULT_WINDOW_MS,
        }
    }

    pub fn with_window_ms(mut self, window_ms: u64) -> Self {
        self.window_ms = window_ms;
        self
    }

    pub fn build(self) -> MicroAnalytics {
        let analytics = MicroAnalytics::new(&self.symbol);
        analytics.set_window_ms(self.window_ms);
        analytics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_analytics() {
        let analytics = MicroAnalytics::new("BTCUSDT");
        
        // Record some placements
        analytics.record_placement(1, 50000, 100);
        analytics.record_placement(2, 50000, 200);
        
        // One fill, one cancel
        analytics.record_fill(1, 100);
        analytics.record_cancellation(2, false);
        
        let snapshot = analytics.get_snapshot().unwrap();
        assert_eq!(snapshot.overall_cancel_rate, 0.5);
        assert_eq!(snapshot.overall_fill_rate, 0.5);
    }

    #[test]
    fn test_spoofing_detection() {
        let analytics = MicroAnalytics::new("ETHUSDT");
        
        analytics.record_placement(1, 3000, 50);
        analytics.record_cancellation(1, true); // Mark as spoof
        
        let rate = analytics.get_spoofing_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_fill_probability() {
        let analytics = MicroAnalytics::new("SOLUSDT");
        
        // With no data, should return reasonable default
        let prob = analytics.get_fill_probability(0, 0.5);
        assert!(prob >= 0.0 && prob <= 1.0);
        
        // Higher position = lower probability
        let prob_pos_0 = analytics.get_fill_probability(0, 0.5);
        let prob_pos_10 = analytics.get_fill_probability(10, 0.5);
        assert!(prob_pos_0 >= prob_pos_10);
    }
}
