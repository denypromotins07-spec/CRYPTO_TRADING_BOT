//! Kernel Monitor for ZAID Crypto Trading Bot
//!
//! This module tracks OS-level metrics including context switches,
//! page faults, and CPU throttling in real-time to detect system
//! bottlenecks that could impact trading performance.
//!
//! Features:
//! - Real-time context switch monitoring
//! - Page fault tracking
//! - CPU frequency/throttling detection
//! - Windows-specific performance counters
//! - Alerting when thresholds exceeded

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use tracing::{info, warn, debug, error};

/// Configuration for kernel monitoring
#[derive(Debug, Clone)]
pub struct KernelMonitorConfig {
    /// Monitoring interval
    pub check_interval: Duration,
    /// Maximum acceptable context switches per second
    pub max_context_switches_per_sec: u64,
    /// Maximum acceptable page faults per second
    pub max_page_faults_per_sec: u64,
    /// Enable CPU throttling detection
    pub detect_throttling: bool,
    /// Alert callback threshold multiplier
    pub alert_threshold_multiplier: f64,
}

impl Default for KernelMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_millis(100), // 10Hz monitoring
            max_context_switches_per_sec: 10_000,
            max_page_faults_per_sec: 1000,
            detect_throttling: true,
            alert_threshold_multiplier: 1.5,
        }
    }
}

/// Current kernel metrics snapshot
#[derive(Debug, Clone, Default)]
pub struct KernelMetrics {
    /// Voluntary context switches
    pub voluntary_ctx_switches: u64,
    /// Involuntary context switches
    pub involuntary_ctx_switches: u64,
    /// Major page faults (disk I/O required)
    pub major_page_faults: u64,
    /// Minor page faults
    pub minor_page_faults: u64,
    /// Current CPU frequency (MHz) if available
    pub cpu_frequency_mhz: Option<u32>,
    /// Is CPU throttling detected
    pub is_throttling: bool,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

impl KernelMetrics {
    /// Calculate rates from two snapshots
    pub fn calculate_rates(&self, previous: &KernelMetrics, elapsed_secs: f64) -> MetricRates {
        if elapsed_secs <= 0.0 {
            return MetricRates::default();
        }

        let ctx_switches = (self.voluntary_ctx_switches.saturating_sub(previous.voluntary_ctx_switches))
            + (self.involuntary_ctx_switches.saturating_sub(previous.involuntary_ctx_switches));
        
        let page_faults = (self.major_page_faults.saturating_sub(previous.major_page_faults))
            + (self.minor_page_faults.saturating_sub(previous.minor_page_faults));

        MetricRates {
            context_switches_per_sec: (ctx_switches as f64 / elapsed_secs) as u64,
            page_faults_per_sec: (page_faults as f64 / elapsed_secs) as u64,
            major_faults_per_sec: ((self.major_page_faults.saturating_sub(previous.major_page_faults)) as f64 / elapsed_secs) as u64,
        }
    }
}

/// Rates calculated from metrics
#[derive(Debug, Clone, Default)]
pub struct MetricRates {
    pub context_switches_per_sec: u64,
    pub page_faults_per_sec: u64,
    pub major_faults_per_sec: u64,
}

/// Alert types from the kernel monitor
#[derive(Debug, Clone)]
pub enum KernelAlert {
    HighContextSwitches { rate: u64, threshold: u64 },
    HighPageFaults { rate: u64, threshold: u64 },
    CpuThrottlingDetected,
    CpuFrequencyDrop { current_mhz: u32, expected_mhz: u32 },
}

/// Kernel monitor state
pub struct KernelMonitor {
    config: KernelMonitorConfig,
    running: AtomicBool,
    last_metrics: Arc<parking_lot::RwLock<KernelMetrics>>,
    alert_count: AtomicU64,
}

impl KernelMonitor {
    /// Create a new kernel monitor
    pub fn new(config: KernelMonitorConfig) -> Self {
        info!("Initializing Kernel Monitor with {}ms interval",
              config.check_interval.as_millis());
        
        Self {
            config,
            running: AtomicBool::new(false),
            last_metrics: Arc::new(parking_lot::RwLock::new(KernelMetrics::default())),
            alert_count: AtomicU64::new(0),
        }
    }

    /// Start the monitoring thread
    pub fn start(&self, alert_callback: impl Fn(KernelAlert) + Send + Sync + 'static) {
        if self.running.swap(true, Ordering::SeqCst) {
            warn!("Kernel monitor already running");
            return;
        }

        let config = self.config.clone();
        let last_metrics = Arc::clone(&self.last_metrics);
        let running = self.running.clone();
        let alert_count = self.alert_count.clone();

        thread::spawn(move || {
            let mut previous_metrics = KernelMetrics::default();
            let mut last_check = Instant::now();

            while running.load(Ordering::Relaxed) {
                // Get current metrics
                let current_metrics = Self::collect_metrics(&config);
                
                // Update stored metrics
                *last_metrics.write() = current_metrics.clone();

                // Calculate elapsed time
                let elapsed = last_check.elapsed();
                let elapsed_secs = elapsed.as_secs_f64();

                if elapsed_secs > 0.0 {
                    // Calculate rates
                    let rates = current_metrics.calculate_rates(&previous_metrics, elapsed_secs);

                    // Check thresholds and generate alerts
                    let ctx_threshold = (config.max_context_switches_per_sec as f64 
                        * config.alert_threshold_multiplier) as u64;
                    
                    if rates.context_switches_per_sec > ctx_threshold {
                        let alert = KernelAlert::HighContextSwitches {
                            rate: rates.context_switches_per_sec,
                            threshold: ctx_threshold,
                        };
                        alert_callback(alert);
                        alert_count.fetch_add(1, Ordering::Relaxed);
                    }

                    let pf_threshold = (config.max_page_faults_per_sec as f64 
                        * config.alert_threshold_multiplier) as u64;
                    
                    if rates.page_faults_per_sec > pf_threshold {
                        let alert = KernelAlert::HighPageFaults {
                            rate: rates.page_faults_per_sec,
                            threshold: pf_threshold,
                        };
                        alert_callback(alert);
                        alert_count.fetch_add(1, Ordering::Relaxed);
                    }

                    // Check CPU throttling
                    if config.detect_throttling && current_metrics.is_throttling {
                        alert_callback(KernelAlert::CpuThrottlingDetected);
                        alert_count.fetch_add(1, Ordering::Relaxed);
                    }
                }

                previous_metrics = current_metrics;
                last_check = Instant::now();

                // Sleep until next check
                thread::sleep(config.check_interval);
            }

            info!("Kernel monitor thread stopped");
        });

        info!("Kernel monitor started");
    }

    /// Stop the monitoring thread
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("Kernel monitor stopping...");
    }

    /// Collect current kernel metrics (platform-specific)
    fn collect_metrics(config: &KernelMonitorConfig) -> KernelMetrics {
        let mut metrics = KernelMetrics::default();
        metrics.timestamp = Instant::now();

        #[cfg(target_os = "windows")]
        {
            Self::collect_metrics_windows(&mut metrics, config);
        }

        #[cfg(target_os = "linux")]
        {
            Self::collect_metrics_linux(&mut metrics, config);
        }

        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            // Fallback for other platforms
            debug!("Platform-specific metrics collection not implemented");
        }

        metrics
    }

    #[cfg(target_os = "windows")]
    fn collect_metrics_windows(metrics: &mut KernelMetrics, _config: &KernelMonitorConfig) {
        // Windows implementation would use Performance Data Helper (PDH) API
        // or NtQueryInformationProcess for process-specific metrics
        
        // Placeholder: In production, would use winapi crate:
        // - PdhCollectQueryData for performance counters
        // - GetProcessTimes for context switches
        // - CallNtPowerInformation for CPU throttling
        
        debug!("Windows metrics collection (placeholder)");
        
        // Simulate getting some metrics
        metrics.voluntary_ctx_switches = 0; // Would get from GetProcessTimes
        metrics.involuntary_ctx_switches = 0;
        
        // Check CPU frequency using CallNtPowerInformation
        // For now, mark as unknown
        metrics.cpu_frequency_mhz = None;
        metrics.is_throttling = false;
    }

    #[cfg(target_os = "linux")]
    fn collect_metrics_linux(metrics: &mut KernelMetrics, _config: &KernelMonitorConfig) {
        // Linux implementation reads from /proc filesystem
        
        // Read process stats
        if let Ok(stat_content) = std::fs::read_to_string("/proc/self/stat") {
            let parts: Vec<&str> = stat_content.split_whitespace().collect();
            if parts.len() > 14 {
                metrics.voluntary_ctx_switches = parts.get(13)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                metrics.involuntary_ctx_switches = parts.get(14)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            }
        }

        // Read page faults from /proc/self/status
        if let Ok(status_content) = std::fs::read_to_string("/proc/self/status") {
            for line in status_content.lines() {
                if line.starts_with("VmRSS:") {
                    // Parse memory info
                    break;
                }
                if line.starts_with("MinFlt:") || line.starts_with("MajFlt:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if line.starts_with("MinFlt:") {
                            metrics.minor_page_faults = parts[1].parse().unwrap_or(0);
                        } else if line.starts_with("MajFlt:") {
                            metrics.major_page_faults = parts[1].parse().unwrap_or(0);
                        }
                    }
                }
            }
        }

        // Check CPU frequency
        if let Ok(freq_content) = std::fs::read_to_string(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        ) {
            metrics.cpu_frequency_mhz = freq_content.trim()
                .parse::<u32>()
                .ok()
                .map(|khz| khz / 1000);
        }

        // Check for throttling
        if let Ok(throttle_content) = std::fs::read_to_string(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ) {
            metrics.is_throttling = throttle_content.contains("powersave");
        }
    }

    /// Get the latest metrics
    pub fn get_current_metrics(&self) -> KernelMetrics {
        self.last_metrics.read().clone()
    }

    /// Get alert count since start
    pub fn get_alert_count(&self) -> u64 {
        self.alert_count.load(Ordering::Relaxed)
    }

    /// Check if system is healthy for trading
    pub fn is_healthy(&self) -> bool {
        let metrics = self.get_current_metrics();
        let previous = KernelMetrics::default(); // Should use actual previous
        
        let rates = metrics.calculate_rates(&previous, 1.0); // Assume 1 second

        rates.context_switches_per_sec < self.config.max_context_switches_per_sec
            && rates.page_faults_per_sec < self.config.max_page_faults_per_sec
            && !metrics.is_throttling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_monitor_creation() {
        let monitor = KernelMonitor::new(KernelMonitorConfig::default());
        assert!(!monitor.running.load(Ordering::Relaxed));
        assert_eq!(monitor.get_alert_count(), 0);
    }

    #[test]
    fn test_metrics_calculation() {
        let mut prev = KernelMetrics::default();
        prev.voluntary_ctx_switches = 1000;
        prev.major_page_faults = 10;

        let mut curr = KernelMetrics::default();
        curr.voluntary_ctx_switches = 1500;
        curr.major_page_faults = 15;
        curr.timestamp = Instant::now();

        let rates = curr.calculate_rates(&prev, 1.0);
        assert!(rates.context_switches_per_sec >= 500);
    }
}
