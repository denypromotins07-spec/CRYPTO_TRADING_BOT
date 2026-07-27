//! Lock-Free Metrics using Atomic Counters for Latency Histograms
//!
//! This module provides zero-cost abstractions for recording and querying
//! latency metrics without locks, ensuring O(1) operations that never
//! trigger garbage collection pauses.
//!
//! Key Features:
//! - Atomic counters for thread-safe increment operations
//! - Fixed-size histogram buckets to avoid dynamic allocation
//! - Percentile calculations with minimal CPU overhead
//! - Memory-bounded data structures (no unbounded growth)
//! - Zero-copy metric export for Prometheus integration
//!
//! Designed for the ZAID Personal Crypto Trading Bot to maintain
//! microsecond observability during high-frequency trading windows.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::array;

/// Number of histogram buckets (logarithmic scale)
const BUCKET_COUNT: usize = 16;

/// Bucket boundaries in microseconds (powers of 2 approximately)
const BUCKET_BOUNDARIES: [u64; BUCKET_COUNT] = [
    10,       // 10μs
    50,       // 50μs
    100,      // 100μs
    250,      // 250μs
    500,      // 500μs
    1_000,    // 1ms
    2_500,    // 2.5ms
    5_000,    // 5ms
    10_000,   // 10ms
    25_000,   // 25ms
    50_000,   // 50ms
    100_000,  // 100ms
    250_000,  // 250ms
    500_000,  // 500ms
    1_000_000,// 1s
    u64::MAX, // Overflow bucket
];

/// A single atomic counter for lock-free increments
#[derive(Debug)]
pub struct AtomicCounter {
    value: AtomicU64,
}

impl AtomicCounter {
    /// Create a new counter with initial value
    #[inline]
    pub fn new(initial: u64) -> Self {
        AtomicCounter {
            value: AtomicU64::new(initial),
        }
    }

    /// Increment the counter atomically and return the new value
    #[inline]
    pub fn increment(&self, delta: u64) -> u64 {
        self.value.fetch_add(delta, Ordering::Relaxed) + delta
    }

    /// Get the current value
    #[inline]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero
    #[inline]
    pub fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::Relaxed)
    }

    /// Add a value atomically (for sum tracking)
    #[inline]
    pub fn add(&self, value: u64) -> u64 {
        self.value.fetch_add(value, Ordering::Relaxed) + value
    }
}

/// Lock-free latency histogram using atomic counters
///
/// This structure maintains fixed-size buckets to record latency distributions
/// without any heap allocation after initialization. All operations are O(1).
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Bucket counts (atomic for lock-free updates)
    buckets: [AtomicU64; BUCKET_COUNT],
    
    /// Total count of recorded values
    count: AtomicU64,
    
    /// Sum of all recorded values (for mean calculation)
    sum: AtomicU64,
    
    /// Minimum recorded value
    min: AtomicU64,
    
    /// Maximum recorded value
    max: AtomicU64,
}

impl LatencyHistogram {
    /// Create a new empty histogram
    #[inline]
    pub fn new() -> Self {
        LatencyHistogram {
            buckets: array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    /// Record a latency value in microseconds (O(1) operation)
    #[inline]
    pub fn record(&self, duration_us: u64) {
        // Find the appropriate bucket using binary search
        let bucket_idx = self.find_bucket(duration_us);
        
        // Atomically increment the bucket count
        self.buckets[bucket_idx].fetch_add(1, Ordering::Relaxed);
        
        // Update count and sum
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(duration_us, Ordering::Relaxed);
        
        // Update min/max (may have contention but still lock-free)
        self.update_min_max(duration_us);
    }

    /// Find the bucket index for a given duration
    #[inline]
    fn find_bucket(&self, duration_us: u64) -> usize {
        // Linear search is fast enough for 16 buckets
        // Could use binary search for larger bucket counts
        for (i, &boundary) in BUCKET_BOUNDARIES.iter().enumerate() {
            if duration_us <= boundary {
                return i;
            }
        }
        BUCKET_COUNT - 1 // Overflow bucket
    }

    /// Update min/max values atomically
    #[inline]
    fn update_min_max(&self, duration_us: u64) {
        // Update min (CAS loop)
        let mut current_min = self.min.load(Ordering::Relaxed);
        while duration_us < current_min {
            match self.min.compare_exchange_weak(
                current_min,
                duration_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }

        // Update max (CAS loop)
        let mut current_max = self.max.load(Ordering::Relaxed);
        while duration_us > current_max {
            match self.max.compare_exchange_weak(
                current_max,
                duration_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    /// Get the count of recorded values
    #[inline]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the sum of all recorded values
    #[inline]
    pub fn sum(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
    }

    /// Get the mean latency in microseconds
    #[inline]
    pub fn mean(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            return 0.0;
        }
        self.sum() as f64 / count as f64
    }

    /// Get the minimum recorded latency
    #[inline]
    pub fn min(&self) -> u64 {
        let min = self.min.load(Ordering::Relaxed);
        if min == u64::MAX {
            0
        } else {
            min
        }
    }

    /// Get the maximum recorded latency
    #[inline]
    pub fn max(&self) -> u64 {
        self.max.load(Ordering::Relaxed)
    }

    /// Calculate approximate percentile from histogram
    ///
    /// Returns the midpoint of the bucket containing the percentile
    #[inline]
    pub fn percentile(&self, p: f64) -> Option<f64> {
        let count = self.count();
        if count == 0 {
            return None;
        }

        let target = ((count as f64) * p).ceil() as u64;
        let mut cumulative = 0u64;

        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                // Return bucket midpoint
                let lower = if i == 0 {
                    0
                } else {
                    BUCKET_BOUNDARIES[i - 1]
                };
                let upper = BUCKET_BOUNDARIES[i];
                
                // Handle overflow bucket specially
                if i == BUCKET_COUNT - 1 {
                    return Some(upper as f64);
                }
                
                return Some((lower as f64 + upper as f64) / 2.0);
            }
        }

        // Should not reach here, but return max just in case
        Some(BUCKET_BOUNDARIES[BUCKET_COUNT - 2] as f64 * 2.0)
    }

    /// Get P50 (median) latency
    #[inline]
    pub fn p50(&self) -> Option<f64> {
        self.percentile(0.50)
    }

    /// Get P95 latency
    #[inline]
    pub fn p95(&self) -> Option<f64> {
        self.percentile(0.95)
    }

    /// Get P99 latency
    #[inline]
    pub fn p99(&self) -> Option<f64> {
        self.percentile(0.99)
    }

    /// Get comprehensive statistics
    #[inline]
    pub fn stats(&self) -> HistogramStats {
        HistogramStats {
            count: self.count(),
            mean: self.mean(),
            min: self.min(),
            max: self.max(),
            p50: self.p50().unwrap_or(0.0),
            p95: self.p95().unwrap_or(0.0),
            p99: self.p99().unwrap_or(0.0),
        }
    }

    /// Get bucket counts as a vector (for export)
    #[inline]
    pub fn bucket_counts(&self) -> [u64; BUCKET_COUNT] {
        array::from_fn(|i| self.buckets[i].load(Ordering::Relaxed))
    }

    /// Reset all counters
    #[inline]
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
    }
}

impl Default for LatencyHistogram {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics snapshot from a histogram
#[derive(Debug, Clone, Copy)]
pub struct HistogramStats {
    pub count: u64,
    pub mean: f64,
    pub min: u64,
    pub max: u64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl HistogramStats {
    /// Convert to JSON string (for logging/export)
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"count":{},"mean":{:.2},"min":{},"max":{},"p50":{:.2},"p95":{:.2},"p99":{:.2}}}"#,
            self.count, self.mean, self.min, self.max, self.p50, self.p95, self.p99
        )
    }
}

/// Multi-dimensional metrics registry for tracking multiple operations
#[derive(Debug)]
pub struct MetricsRegistry {
    /// Named histograms for different operations
    histograms: parking_lot::RwLock<std::collections::HashMap<String, LatencyHistogram>>,
    
    /// Global counters
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    start_time: Instant,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        MetricsRegistry {
            histograms: parking_lot::RwLock::new(std::collections::HashMap::new()),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Get or create a histogram for a named operation
    #[inline]
    pub fn histogram(&self, name: &str) -> std::sync::Arc<LatencyHistogram> {
        // Fast path: check if exists (read lock)
        {
            let histograms = self.histograms.read();
            if let Some(hist) = histograms.get(name) {
                // Clone the Arc (cheap)
                return std::sync::Arc::clone(hist);
            }
        }

        // Slow path: create if not exists (write lock)
        let mut histograms = self.histograms.write();
        
        // Double-check after acquiring write lock
        if let Some(hist) = histograms.get(name) {
            return std::sync::Arc::clone(hist);
        }

        let hist = std::sync::Arc::new(LatencyHistogram::new());
        histograms.insert(name.to_string(), std::sync::Arc::clone(&hist));
        hist
    }

    /// Record a latency for a named operation
    #[inline]
    pub fn record(&self, name: &str, duration_us: u64) {
        self.total_requests.increment(1);
        self.histogram(name).record(duration_us);
    }

    /// Record an error
    #[inline]
    pub fn record_error(&self) {
        self.total_errors.increment(1);
    }

    /// Get uptime
    #[inline]
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get total request count
    #[inline]
    pub fn total_requests(&self) -> u64 {
        self.total_requests.get()
    }

    /// Get total error count
    #[inline]
    pub fn total_errors(&self) -> u64 {
        self.total_errors.get()
    }

    /// Get all histogram names
    #[inline]
    pub fn histogram_names(&self) -> Vec<String> {
        self.histograms.read().keys().cloned().collect()
    }

    /// Export all metrics in Prometheus format
    #[inline]
    pub fn export_prometheus(&self) -> String {
        let mut output = String::with_capacity(4096);
        
        // Uptime
        output.push_str(&format!(
            "# HELP bot_uptime_seconds Bot uptime in seconds\n# TYPE bot_uptime_seconds counter\nbot_uptime_seconds {:.3}\n\n",
            self.uptime().as_secs_f64()
        ));

        // Request counts
        output.push_str(&format!(
            "# HELP bot_requests_total Total number of requests\n# TYPE bot_requests_total counter\nbot_requests_total {}\n\n",
            self.total_requests()
        ));

        output.push_str(&format!(
            "# HELP bot_errors_total Total number of errors\n# TYPE bot_errors_total counter\nbot_errors_total {}\n\n",
            self.total_errors()
        ));

        // Histograms
        let histograms = self.histograms.read();
        for (name, hist) in histograms.iter() {
            let stats = hist.stats();
            
            output.push_str(&format!(
                "# HELP bot_{}_latency_us Latency histogram for {}\n# TYPE bot_{}_latency_us summary\n",
                name, name, name
            ));
            output.push_str(&format!(
                "bot_{}_latency_us{{quantile=\"0.5\"}} {:.3}\n",
                name, stats.p50
            ));
            output.push_str(&format!(
                "bot_{}_latency_us{{quantile=\"0.95\"}} {:.3}\n",
                name, stats.p95
            ));
            output.push_str(&format!(
                "bot_{}_latency_us{{quantile=\"0.99\"}} {:.3}\n",
                name, stats.p99
            ));
            output.push_str(&format!(
                "bot_{}_latency_us_sum {}\n",
                name, stats.sum
            ));
            output.push_str(&format!(
                "bot_{}_latency_us_count {}\n\n",
                name, stats.count
            ));
        }

        output
    }
}

impl Default for MetricsRegistry {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// RAII timer for automatically recording operation latency
pub struct LatencyTimer<'a> {
    registry: &'a MetricsRegistry,
    name: String,
    start: Instant,
    recorded: bool,
}

impl<'a> LatencyTimer<'a> {
    /// Create a new timer that will record to the given registry
    #[inline]
    pub fn new(registry: &'a MetricsRegistry, name: &str) -> Self {
        LatencyTimer {
            registry,
            name: name.to_string(),
            start: Instant::now(),
            recorded: false,
        }
    }

    /// Get elapsed time in microseconds
    #[inline]
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// Record the latency manually (usually automatic on drop)
    #[inline]
    pub fn record(mut self) {
        let duration = self.elapsed_us();
        self.registry.record(&self.name, duration);
        self.recorded = true;
    }
}

impl<'a> Drop for LatencyTimer<'a> {
    #[inline]
    fn drop(&mut self) {
        if !self.recorded {
            let duration = self.elapsed_us();
            self.registry.record(&self.name, duration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new(0);
        assert_eq!(counter.get(), 0);
        
        counter.increment(1);
        assert_eq!(counter.get(), 1);
        
        counter.add(5);
        assert_eq!(counter.get(), 6);
        
        let old = counter.reset();
        assert_eq!(old, 6);
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_latency_histogram() {
        let hist = LatencyHistogram::new();
        
        // Record some values
        hist.record(100);
        hist.record(200);
        hist.record(300);
        hist.record(400);
        hist.record(500);
        
        assert_eq!(hist.count(), 5);
        assert_eq!(hist.sum(), 1500);
        assert!((hist.mean() - 300.0).abs() < 0.01);
        assert_eq!(hist.min(), 100);
        assert_eq!(hist.max(), 500);
        
        // Test percentiles
        assert!(hist.p50().is_some());
        assert!(hist.p95().is_some());
        assert!(hist.p99().is_some());
    }

    #[test]
    fn test_metrics_registry() {
        let registry = MetricsRegistry::new();
        
        // Record some latencies
        registry.record("order_execution", 100);
        registry.record("order_execution", 200);
        registry.record("market_data", 50);
        
        assert_eq!(registry.total_requests(), 3);
        
        // Record an error
        registry.record_error();
        assert_eq!(registry.total_errors(), 1);
        
        // Check histogram exists
        let names = registry.histogram_names();
        assert!(names.contains(&"order_execution".to_string()));
        assert!(names.contains(&"market_data".to_string()));
    }

    #[test]
    fn test_latency_timer() {
        let registry = MetricsRegistry::new();
        
        {
            let _timer = LatencyTimer::new(&registry, "test_op");
            std::thread::sleep(Duration::from_millis(1));
        } // Timer records on drop
        
        assert_eq!(registry.total_requests(), 1);
    }

    #[test]
    fn test_prometheus_export() {
        let registry = MetricsRegistry::new();
        
        registry.record("test", 100);
        registry.record("test", 200);
        
        let export = registry.export_prometheus();
        
        assert!(export.contains("bot_test_latency_us"));
        assert!(export.contains("quantile=\"0.5\""));
        assert!(export.contains("quantile=\"0.95\""));
        assert!(export.contains("quantile=\"0.99\""));
    }
}
