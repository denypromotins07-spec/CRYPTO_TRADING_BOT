// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// Micro Analytics Module
// Profiles historical queue cancellation rates and microstructure patterns
// Zero-cost abstractions for memory-efficient operation

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Queue event types for tracking
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueueEventType {
    OrderAdded,
    OrderCancelled,
    OrderFilled,
    OrderModified,
}

/// Single queue event record
#[derive(Debug, Clone)]
pub struct QueueEvent {
    pub timestamp_ns: u64,
    pub event_type: QueueEventType,
    pub price: i64,
    pub quantity: u64,
    pub order_id: u64,
    pub lifetime_ms: Option<f64>, // For cancellations/fills
}

/// Cancellation rate statistics for a price level
#[derive(Debug, Clone)]
pub struct CancellationStats {
    pub price: i64,
    pub total_orders: u64,
    pub cancelled_orders: u64,
    pub filled_orders: u64,
    pub modified_orders: u64,
    pub cancellation_rate: f64,
    pub avg_lifetime_ms: f64,
    pub median_lifetime_ms: f64,
    pub std_lifetime_ms: f64,
    pub rapid_cancels_under_5ms: u64,
}

/// Historical pattern analysis result
#[derive(Debug, Clone)]
pub struct PatternAnalysis {
    pub timestamp_ns: u64,
    pub dominant_pattern: MarketPattern,
    pub confidence: f64,
    pub recommended_strategy: TradingStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketPattern {
    NormalTrading,
    HighCancellation,
    SpoofingDetected,
    LiquidityCrisis,
    MeanReverting,
    Trending,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingStrategy {
    MarketMaking,
    Momentum,
    MeanReversion,
    Passive,
    Halt,
}

/// Micro Analytics Engine for historical profiling
pub struct MicroAnalyticsEngine {
    // Event history by price level
    events_by_price: HashMap<i64, VecDeque<QueueEvent>>,
    max_events_per_level: usize,
    
    // Global event history
    all_events: VecDeque<QueueEvent>,
    max_global_events: usize,
    
    // Cancellation tracking
    cancellation_times: VecDeque<f64>, // Lifetime in ms
    
    // Pre-computed statistics cache
    stats_cache: HashMap<i64, CancellationStats>,
    cache_valid: bool,
    
    // Time-based buckets for trend analysis
    time_buckets: VecDeque<TimeBucket>,
    bucket_duration_ms: u64,
    
    // Statistics
    analytics_stats: AnalyticsStats,
    
    // Last calculation timestamp
    last_calculation_ns: u64,
}

#[derive(Debug, Clone)]
struct TimeBucket {
    start_ns: u64,
    end_ns: u64,
    add_count: u64,
    cancel_count: u64,
    fill_count: u64,
    total_volume_added: u64,
    total_volume_cancelled: u64,
}

#[derive(Debug, Default, Clone)]
pub struct AnalyticsStats {
    pub total_events_processed: u64,
    pub total_cancellations: u64,
    pub overall_cancellation_rate: f64,
    pub patterns_detected: u64,
    pub avg_analysis_time_us: f64,
}

impl MicroAnalyticsEngine {
    pub fn new(max_events_per_level: usize, max_global_events: usize) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            events_by_price: HashMap::with_capacity(100),
            max_events_per_level,
            all_events: VecDeque::with_capacity(max_global_events),
            max_global_events,
            cancellation_times: VecDeque::with_capacity(1000),
            stats_cache: HashMap::with_capacity(100),
            cache_valid: false,
            time_buckets: VecDeque::with_capacity(60),
            bucket_duration_ms: 1000, // 1 second buckets
            analytics_stats: AnalyticsStats::default(),
            last_calculation_ns: now_ns,
        }
    }

    /// Record a queue event
    pub fn record_event(&mut self, event: QueueEvent) {
        let price = event.price;
        
        // Add to price-level history
        let level_events = self.events_by_price.entry(price).or_insert_with(|| {
            VecDeque::with_capacity(self.max_events_per_level)
        });
        level_events.push_back(event.clone());
        
        if level_events.len() > self.max_events_per_level {
            level_events.pop_front();
        }
        
        // Add to global history
        self.all_events.push_back(event.clone());
        if self.all_events.len() > self.max_global_events {
            self.all_events.pop_front();
        }
        
        // Track cancellation lifetimes
        if event.event_type == QueueEventType::OrderCancelled {
            if let Some(lifetime) = event.lifetime_ms {
                self.cancellation_times.push_back(lifetime);
                if self.cancellation_times.len() > 1000 {
                    self.cancellation_times.pop_front();
                }
            }
        }
        
        // Update time buckets
        self.update_time_buckets(&event);
        
        // Invalidate cache
        self.cache_valid = false;
        
        self.analytics_stats.total_events_processed += 1;
        if event.event_type == QueueEventType::OrderCancelled {
            self.analytics_stats.total_cancellations += 1;
        }
        
        self.last_calculation_ns = event.timestamp_ns;
    }

    /// Get cancellation statistics for a price level
    pub fn get_cancellation_stats(&mut self, price: i64) -> Option<CancellationStats> {
        if !self.cache_valid {
            self.rebuild_cache();
        }
        self.stats_cache.get(&price).cloned()
    }

    /// Get overall cancellation rate
    pub fn get_overall_cancellation_rate(&self) -> f64 {
        if self.analytics_stats.total_events_processed == 0 {
            return 0.0;
        }
        self.analytics_stats.total_cancellations as f64 / 
            self.analytics_stats.total_events_processed as f64
    }

    /// Analyze market patterns
    pub fn analyze_patterns(&mut self) -> PatternAnalysis {
        let start = Instant::now();
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Calculate metrics for pattern detection
        let cancel_rate = self.get_overall_cancellation_rate();
        let rapid_cancel_rate = self.calculate_rapid_cancel_rate();
        let volume_cancel_ratio = self.calculate_volume_cancel_ratio();
        
        // Determine dominant pattern
        let (pattern, confidence) = if rapid_cancel_rate > 0.3 {
            (MarketPattern::SpoofingDetected, rapid_cancel_rate.min(1.0))
        } else if cancel_rate > 0.5 {
            (MarketPattern::HighCancellation, cancel_rate.min(1.0))
        } else if volume_cancel_ratio > 0.7 {
            (MarketPattern::LiquidityCrisis, volume_cancel_ratio.min(1.0))
        } else if self.is_mean_reverting() {
            (MarketPattern::MeanReverting, 0.7)
        } else if self.is_trending() {
            (MarketPattern::Trending, 0.7)
        } else {
            (MarketPattern::NormalTrading, 0.8)
        };

        // Determine recommended strategy
        let strategy = match pattern {
            MarketPattern::SpoofingDetected | MarketPattern::HighCancellation => TradingStrategy::Passive,
            MarketPattern::LiquidityCrisis => TradingStrategy::Halt,
            MarketPattern::MeanReverting => TradingStrategy::MeanReversion,
            MarketPattern::Trending => TradingStrategy::Momentum,
            MarketPattern::NormalTrading => TradingStrategy::MarketMaking,
        };

        let analysis_time_us = start.elapsed().as_micros() as f64;
        self.analytics_stats.avg_analysis_time_us = 
            (self.analytics_stats.avg_analysis_time_us * self.analytics_stats.patterns_detected as f64 
             + analysis_time_us) / (self.analytics_stats.patterns_detected + 1) as f64;
        self.analytics_stats.patterns_detected += 1;

        self.last_calculation_ns = now_ns;

        PatternAnalysis {
            timestamp_ns: now_ns,
            dominant_pattern: pattern,
            confidence,
            recommended_strategy: strategy,
        }
    }

    /// Get average cancellation lifetime
    pub fn get_avg_cancellation_lifetime(&self) -> f64 {
        if self.cancellation_times.is_empty() {
            return 0.0;
        }
        self.cancellation_times.iter().sum::<f64>() / self.cancellation_times.len() as f64
    }

    /// Get percentage of rapid cancellations (under 5ms)
    pub fn get_rapid_cancel_percentage(&self) -> f64 {
        if self.cancellation_times.is_empty() {
            return 0.0;
        }
        let rapid_count = self.cancellation_times.iter()
            .filter(|&&t| t < 5.0)
            .count();
        rapid_count as f64 / self.cancellation_times.len() as f64
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.events_by_price.clear();
        self.all_events.clear();
        self.cancellation_times.clear();
        self.stats_cache.clear();
        self.time_buckets.clear();
        self.cache_valid = false;
        self.analytics_stats = AnalyticsStats::default();
    }

    fn rebuild_cache(&mut self) {
        self.stats_cache.clear();
        
        for (&price, events) in &self.events_by_price {
            let stats = self.calculate_level_stats(price, events);
            self.stats_cache.insert(price, stats);
        }
        
        self.cache_valid = true;
        
        // Update overall rate
        self.analytics_stats.overall_cancellation_rate = self.get_overall_cancellation_rate();
    }

    fn calculate_level_stats(&self, price: i64, events: &VecDeque<QueueEvent>) -> CancellationStats {
        let mut total = 0u64;
        let mut cancelled = 0u64;
        let mut filled = 0u64;
        let mut modified = 0u64;
        let mut lifetimes: Vec<f64> = Vec::new();
        let mut rapid_cancels = 0u64;

        for event in events {
            total += 1;
            match event.event_type {
                QueueEventType::OrderCancelled => {
                    cancelled += 1;
                    if let Some(lt) = event.lifetime_ms {
                        lifetimes.push(lt);
                        if lt < 5.0 {
                            rapid_cancels += 1;
                        }
                    }
                }
                QueueEventType::OrderFilled => {
                    filled += 1;
                    if let Some(lt) = event.lifetime_ms {
                        lifetimes.push(lt);
                    }
                }
                QueueEventType::OrderModified => modified += 1,
                _ => {}
            }
        }

        let cancel_rate = if total > 0 {
            cancelled as f64 / total as f64
        } else {
            0.0
        };

        let (avg_lt, median_lt, std_lt) = if lifetimes.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let avg = lifetimes.iter().sum::<f64>() / lifetimes.len() as f64;
            
            let mut sorted = lifetimes.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            };

            let variance = lifetimes.iter()
                .map(|x| (x - avg).powi(2))
                .sum::<f64>() / lifetimes.len() as f64;
            let std = variance.sqrt();

            (avg, median, std)
        };

        CancellationStats {
            price,
            total_orders: total,
            cancelled_orders: cancelled,
            filled_orders: filled,
            modified_orders: modified,
            cancellation_rate: cancel_rate,
            avg_lifetime_ms: avg_lt,
            median_lifetime_ms: median_lt,
            std_lifetime_ms: std_lt,
            rapid_cancels_under_5ms: rapid_cancels,
        }
    }

    fn update_time_buckets(&mut self, event: &QueueEvent) {
        let now_ns = event.timestamp_ns;
        let bucket_duration_ns = self.bucket_duration_ms * 1_000_000;

        // Get or create current bucket
        let bucket = if self.time_buckets.is_empty() || 
            now_ns >= self.time_buckets.back().unwrap().end_ns {
            let new_bucket = TimeBucket {
                start_ns: now_ns,
                end_ns: now_ns + bucket_duration_ns,
                add_count: 0,
                cancel_count: 0,
                fill_count: 0,
                total_volume_added: 0,
                total_volume_cancelled: 0,
            };
            self.time_buckets.push_back(new_bucket);
            self.time_buckets.back_mut().unwrap()
        } else {
            self.time_buckets.back_mut().unwrap()
        };

        // Update bucket stats
        match event.event_type {
            QueueEventType::OrderAdded => {
                bucket.add_count += 1;
                bucket.total_volume_added += event.quantity;
            }
            QueueEventType::OrderCancelled => {
                bucket.cancel_count += 1;
                bucket.total_volume_cancelled += event.quantity;
            }
            QueueEventType::OrderFilled => {
                bucket.fill_count += 1;
            }
            _ => {}
        }

        // Prune old buckets
        while self.time_buckets.len() > 60 {
            self.time_buckets.pop_front();
        }
    }

    fn calculate_rapid_cancel_rate(&self) -> f64 {
        if self.cancellation_times.is_empty() {
            return 0.0;
        }
        self.cancellation_times.iter()
            .filter(|&&t| t < 5.0)
            .count() as f64 / self.cancellation_times.len() as f64
    }

    fn calculate_volume_cancel_ratio(&self) -> f64 {
        if self.time_buckets.is_empty() {
            return 0.0;
        }

        let total_added: u64 = self.time_buckets.iter()
            .map(|b| b.total_volume_added)
            .sum();
        let total_cancelled: u64 = self.time_buckets.iter()
            .map(|b| b.total_volume_cancelled)
            .sum();

        if total_added == 0 {
            return 0.0;
        }

        total_cancelled as f64 / total_added as f64
    }

    fn is_mean_reverting(&self) -> bool {
        // Check if cancellation rate oscillates around mean
        if self.cancellation_times.len() < 20 {
            return false;
        }

        let mean = self.get_avg_cancellation_lifetime();
        let mut crossings = 0;
        let mut above = self.cancellation_times.front().map(|&x| x > mean).unwrap_or(false);

        for &lt in self.cancellation_times.iter().skip(1) {
            let currently_above = lt > mean;
            if currently_above != above {
                crossings += 1;
                above = currently_above;
            }
        }

        // High crossing rate suggests mean reversion
        crossings > self.cancellation_times.len() / 4
    }

    fn is_trending(&self) -> bool {
        // Check for sustained direction in cancellation patterns
        if self.cancellation_times.len() < 20 {
            return false;
        }

        let recent_avg: f64 = self.cancellation_times.iter()
            .rev()
            .take(10)
            .sum::<f64>() / 10.0;
        
        let older_avg: f64 = self.cancellation_times.iter()
            .rev()
            .skip(10)
            .take(10)
            .sum::<f64>() / 10.0;

        // Significant change suggests trending
        (recent_avg - older_avg).abs() > older_avg * 0.3
    }
}

impl Default for MicroAnalyticsEngine {
    fn default() -> Self {
        Self::new(1000, 10000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_stats() {
        let mut engine = MicroAnalyticsEngine::new(100, 1000);
        
        // Record some events
        for i in 0..50 {
            let event = QueueEvent {
                timestamp_ns: 1000000000 + i as u64 * 1000000,
                event_type: if i % 3 == 0 { 
                    QueueEventType::OrderCancelled 
                } else { 
                    QueueEventType::OrderAdded 
                },
                price: 50000,
                quantity: 100,
                order_id: i,
                lifetime_ms: if i % 3 == 0 { Some((i as f64) * 0.5) } else { None },
            };
            engine.record_event(event);
        }

        let stats = engine.get_cancellation_stats(50000);
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert!(stats.cancellation_rate > 0.0);
    }

    #[test]
    fn test_pattern_analysis() {
        let mut engine = MicroAnalyticsEngine::new(100, 1000);
        
        // Simulate high cancellation pattern
        for i in 0..100 {
            let event = QueueEvent {
                timestamp_ns: 1000000000 + i as u64 * 100000,
                event_type: QueueEventType::OrderCancelled,
                price: 50000,
                quantity: 100,
                order_id: i,
                lifetime_ms: Some(2.0), // Rapid cancellation
            };
            engine.record_event(event);
        }

        let analysis = engine.analyze_patterns();
        assert_eq!(analysis.dominant_pattern, MarketPattern::SpoofingDetected);
    }
}
