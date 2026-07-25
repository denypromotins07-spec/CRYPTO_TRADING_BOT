// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// File: backend/microstructure/micro_analytics.rs
// Chapter 4: Microstructure Analytics, Execution Adjustments, and SOUL.md Logging
//
// Purpose: Profile historical queue cancellation rates and microstructure patterns
// Constraints: Memory-efficient historical analysis, fast aggregations
// Target: AMD Ryzen AI 5 laptop with 8GB RAM limit

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Maximum historical events to track
const MAX_HISTORY_SIZE: usize = 100_000;

/// Time buckets for analytics (in seconds)
const BUCKET_SIZES: [u64; 4] = [60, 300, 900, 3600]; // 1min, 5min, 15min, 1hour

#[derive(Debug, Clone)]
pub struct CancellationProfile {
    /// Overall cancellation rate (0.0 to 1.0)
    pub overall_rate: f64,
    /// Rate by price level
    pub by_price_level: HashMap<i64, f64>,
    /// Rate by time of day bucket
    pub by_time_bucket: HashMap<u8, f64>,
    /// Average lifetime before cancellation (microseconds)
    pub avg_lifetime_us: u64,
    /// Fast cancellation rate (<5ms)
    pub fast_cancel_rate: f64,
}

#[derive(Debug, Clone)]
pub struct QueueDecayProfile {
    /// Average decay rate (quantity removed per second)
    pub avg_decay_rate: f64,
    /// Decay rate percentiles
    pub decay_percentiles: DecayPercentiles,
    /// Time-to-fill distribution
    pub time_to_fill_distribution: TimeDistribution,
}

#[derive(Debug, Clone)]
pub struct DecayPercentiles {
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Default)]
pub struct TimeDistribution {
    pub buckets: Vec<u64>, // Count in each time bucket
    pub labels: Vec<String>, // Bucket labels
}

/// Event types for microstructure tracking
#[derive(Debug, Clone)]
pub enum MicroEvent {
    OrderNew {
        order_id: String,
        price: i64,
        quantity: i64,
        timestamp_us: u64,
    },
    OrderCancel {
        order_id: String,
        price: i64,
        lifetime_us: u64,
        timestamp_us: u64,
    },
    OrderFill {
        order_id: String,
        price: i64,
        filled_quantity: i64,
        timestamp_us: u64,
    },
    QueueUpdate {
        price: i64,
        quantity_change: i64,
        timestamp_us: u64,
    },
}

/// Aggregated statistics for a time period
#[derive(Debug, Clone, Default)]
pub struct PeriodStats {
    pub total_orders: u64,
    pub total_cancellations: u64,
    pub total_fills: u64,
    pub cancellation_rate: f64,
    pub fill_rate: f64,
    pub avg_order_lifetime_us: u64,
    pub total_volume: i64,
}

/// Main microstructure analytics engine
pub struct MicroAnalytics {
    /// Event history
    event_history: VecDeque<MicroEvent>,
    
    /// Per-level statistics
    level_stats: HashMap<i64, LevelAnalytics>,
    
    /// Time-bucketed statistics
    time_buckets: HashMap<u64, PeriodStats>, // bucket_start -> stats
    
    /// Sequence number
    sequence_number: AtomicU64,
    
    /// Start time
    start_time: Instant,
    
    /// Configuration
    max_events: usize,
}

/// Analytics for a single price level
#[derive(Debug, Clone, Default)]
pub struct LevelAnalytics {
    pub total_orders: u64,
    pub cancellations: u64,
    pub fills: u64,
    pub total_lifetime_us: u64,
    pub lifetimes: VecDeque<u64>, // Recent lifetimes for percentile calculation
}

impl LevelAnalytics {
    pub fn cancellation_rate(&self) -> f64 {
        if self.total_orders == 0 {
            return 0.0;
        }
        self.cancellations as f64 / self.total_orders as f64
    }
    
    pub fn average_lifetime_us(&self) -> u64 {
        if self.lifetimes.is_empty() {
            return 0;
        }
        self.lifetimes.iter().sum::<u64>() / self.lifetimes.len() as u64
    }
}

impl MicroAnalytics {
    pub fn new() -> Self {
        Self {
            event_history: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            level_stats: HashMap::with_capacity(1000),
            time_buckets: HashMap::with_capacity(BUCKET_SIZES.len() * 10),
            sequence_number: AtomicU64::new(0),
            start_time: Instant::now(),
            max_events: MAX_HISTORY_SIZE,
        }
    }
    
    fn now_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }
    
    /// Record a microstructure event
    pub fn record_event(&mut self, event: MicroEvent) {
        // Add to history
        if self.event_history.len() >= self.max_events {
            self.event_history.pop_front();
        }
        self.event_history.push_back(event.clone());
        
        // Update level stats
        match &event {
            MicroEvent::OrderNew { price, .. } => {
                let level = self.level_stats.entry(*price).or_default();
                level.total_orders += 1;
            }
            MicroEvent::OrderCancel { price, lifetime_us, .. } => {
                let level = self.level_stats.entry(*price).or_default();
                level.cancellations += 1;
                level.total_lifetime_us += lifetime_us;
                if level.lifetimes.len() >= 1000 {
                    level.lifetimes.pop_front();
                }
                level.lifetimes.push_back(*lifetime_us);
            }
            MicroEvent::OrderFill { price, .. } => {
                let level = self.level_stats.entry(*price).or_default();
                level.fills += 1;
            }
            _ => {}
        }
        
        // Update time buckets
        self.update_time_buckets(&event);
    }
    
    fn update_time_buckets(&mut self, event: &MicroEvent) {
        let timestamp_us = match event {
            MicroEvent::OrderNew { timestamp_us, .. } => *timestamp_us,
            MicroEvent::OrderCancel { timestamp_us, .. } => *timestamp_us,
            MicroEvent::OrderFill { timestamp_us, .. } => *timestamp_us,
            MicroEvent::QueueUpdate { timestamp_us, .. } => *timestamp_us,
        };
        
        for bucket_size in BUCKET_SIZES.iter() {
            let bucket_start = (timestamp_us / (bucket_size * 1_000_000)) * (bucket_size * 1_000_000);
            let stats = self.time_buckets.entry(bucket_start).or_default();
            
            match event {
                MicroEvent::OrderNew { quantity, .. } => {
                    stats.total_orders += 1;
                    stats.total_volume += quantity;
                }
                MicroEvent::OrderCancel { .. } => {
                    stats.total_cancellations += 1;
                }
                MicroEvent::OrderFill { filled_quantity, .. } => {
                    stats.total_fills += 1;
                    stats.total_volume += filled_quantity;
                }
                _ => {}
            }
            
            // Update rates
            let total = stats.total_orders + stats.total_cancellations;
            if total > 0 {
                stats.cancellation_rate = stats.total_cancellations as f64 / total as f64;
                stats.fill_rate = stats.total_fills as f64 / total as f64;
            }
        }
    }
    
    /// Get cancellation profile across all tracked data
    pub fn get_cancellation_profile(&self) -> CancellationProfile {
        let mut total_orders = 0u64;
        let mut total_cancellations = 0u64;
        let mut total_lifetime = 0u64;
        let mut fast_cancels = 0u64;
        
        let mut by_price: HashMap<i64, (u64, u64)> = HashMap::new();
        
        for (price, level) in &self.level_stats {
            total_orders += level.total_orders;
            total_cancellations += level.cancellations;
            total_lifetime += level.total_lifetime_us;
            
            by_price.insert(*price, (level.total_orders, level.cancellations));
            
            // Count fast cancels from recent lifetimes
            for &lt in &level.lifetimes {
                if lt < 5000 {
                    // <5ms
                    fast_cancels += 1;
                }
            }
        }
        
        let overall_rate = if total_orders > 0 {
            total_cancellations as f64 / total_orders as f64
        } else {
            0.0
        };
        
        let avg_lifetime = if total_cancellations > 0 {
            total_lifetime / total_cancellations
        } else {
            0
        };
        
        let fast_cancel_rate = if total_cancellations > 0 {
            fast_cancels as f64 / total_cancellations as f64
        } else {
            0.0
        };
        
        let by_price_level: HashMap<i64, f64> = by_price
            .into_iter()
            .map(|(p, (o, c))| (p, if o > 0 { c as f64 / o as f64 } else { 0.0 }))
            .collect();
        
        CancellationProfile {
            overall_rate,
            by_price_level,
            by_time_bucket: HashMap::new(), // Would calculate from time_buckets
            avg_lifetime_us: avg_lifetime,
            fast_cancel_rate,
        }
    }
    
    /// Get queue decay profile
    pub fn get_queue_decay_profile(&self) -> QueueDecayProfile {
        // Collect all lifetimes for percentile calculation
        let mut all_lifetimes: Vec<f64> = Vec::new();
        
        for level in self.level_stats.values() {
            for &lt in &level.lifetimes {
                all_lifetimes.push(lt as f64);
            }
        }
        
        if all_lifetimes.is_empty() {
            return QueueDecayProfile {
                avg_decay_rate: 0.0,
                decay_percentiles: DecayPercentiles {
                    p10: 0.0,
                    p50: 0.0,
                    p90: 0.0,
                    p99: 0.0,
                },
                time_to_fill_distribution: TimeDistribution::default(),
            };
        }
        
        all_lifetimes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = all_lifetimes.len();
        let percentiles = DecayPercentiles {
            p10: all_lifetimes[(n as f64 * 0.1) as usize],
            p50: all_lifetimes[(n as f64 * 0.5) as usize],
            p90: all_lifetimes[(n as f64 * 0.9) as usize],
            p99: all_lifetimes[((n as f64 * 0.99) as usize).min(n - 1)],
        };
        
        QueueDecayProfile {
            avg_decay_rate: 0.0, // Would calculate from queue updates
            decay_percentiles: percentiles,
            time_to_fill_distribution: TimeDistribution::default(),
        }
    }
    
    /// Get statistics for a specific time period
    pub fn get_period_stats(&self, bucket_size_seconds: u64) -> Vec<(u64, PeriodStats)> {
        let bucket_us = bucket_size_seconds * 1_000_000;
        let current_us = self.now_us();
        
        let mut results: Vec<_> = self.time_buckets
            .iter()
            .filter(|(&k, _)| k >= current_us.saturating_sub(bucket_us * 10))
            .filter(|(&k, _)| k % bucket_us == 0)
            .map(|(&k, v)| (k, v.clone()))
            .collect();
        
        results.sort_by_key(|(k, _)| *k);
        results
    }
    
    /// Get summary statistics
    pub fn get_summary(&self) -> MicroSummary {
        let cancel_profile = self.get_cancellation_profile();
        let decay_profile = self.get_queue_decay_profile();
        
        MicroSummary {
            total_events: self.event_history.len(),
            total_levels_tracked: self.level_stats.len(),
            overall_cancellation_rate: cancel_profile.overall_rate,
            fast_cancel_ratio: cancel_profile.fast_cancel_rate,
            avg_lifetime_us: cancel_profile.avg_lifetime_us,
            decay_p50_us: decay_profile.decay_percentiles.p50 as u64,
        }
    }
}

impl Default for MicroAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct MicroSummary {
    pub total_events: usize,
    pub total_levels_tracked: usize,
    pub overall_cancellation_rate: f64,
    pub fast_cancel_ratio: f64,
    pub avg_lifetime_us: u64,
    pub decay_p50_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_tracking() {
        let mut analytics = MicroAnalytics::new();
        
        // Record some orders
        analytics.record_event(MicroEvent::OrderNew {
            order_id: "1".to_string(),
            price: 50000,
            quantity: 100,
            timestamp_us: 1000,
        });
        
        // Cancel one quickly
        analytics.record_event(MicroEvent::OrderCancel {
            order_id: "1".to_string(),
            price: 50000,
            lifetime_us: 2000, // 2ms
            timestamp_us: 3000,
        });
        
        let profile = analytics.get_cancellation_profile();
        assert!(profile.overall_rate > 0.0);
    }
}
