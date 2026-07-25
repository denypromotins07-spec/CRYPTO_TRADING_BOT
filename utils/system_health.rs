// =============================================================================
// ZAID PERSONAL CRYPTO TRADING BOT - SYSTEM HEALTH MONITOR
// =============================================================================
// Infrastructure, Logging, and System Health Monitoring for Extreme Stability
//
// Rust implementation of CPU thread monitoring and runaway process killer.
// Provides instant detection and termination of processes exceeding resource limits.
// Uses zero-cost abstractions for minimal overhead during normal operation.
//
// Domains Integrated:
// - System Resource Management
// - Process Control
// - Fault Detection and Recovery
// - Real-time Monitoring
// - Safety-Critical Systems
// =============================================================================

use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use std::thread;
use std::process;
use sysinfo::{ProcessExt, System, SystemExt, ProcessStatus};
use thiserror::Error;
use chrono::{DateTime, Utc};

/// Maximum allowed CPU usage percentage per thread
const MAX_CPU_PERCENT_PER_THREAD: f32 = 90.0;

/// Maximum allowed memory per process (bytes) - 2GB default
const DEFAULT_MAX_PROCESS_MEMORY: u64 = 2 * 1024 * 1024 * 1024;

/// Check interval in milliseconds
const CHECK_INTERVAL_MS: u64 = 500;

/// Error types for system health operations
#[derive(Error, Debug)]
pub enum SystemHealthError {
    #[error("System information error: {0}")]
    SystemInfo(String),
    
    #[error("Process not found: PID {pid}")]
    ProcessNotFound { pid: u32 },
    
    #[error("Resource limit exceeded: {resource} at {value}")]
    ResourceLimitExceeded {
        resource: String,
        value: String,
    },
    
    #[error("Critical failure: {message}")]
    CriticalFailure { message: String },
}

/// Result type alias for health check operations
pub type SystemHealthResult<T> = Result<T, SystemHealthError>;

/// Snapshot of a single process's resource usage
#[derive(Debug, Clone)]
pub struct ProcessSnapshot {
    /// Process ID
    pub pid: u32,
    
    /// Process name
    pub name: String,
    
    /// CPU usage percentage (0-100)
    pub cpu_usage: f32,
    
    /// Memory usage in bytes
    pub memory_bytes: u64,
    
    /// Process status
    pub status: ProcessStatus,
    
    /// Number of threads
    pub thread_count: u32,
    
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
}

impl ProcessSnapshot {
    /// Create snapshot from sysinfo process
    pub fn from_process(process: &dyn ProcessExt, pid: u32) -> Self {
        Self {
            pid,
            name: process.name().to_string(),
            cpu_usage: process.cpu_usage(),
            memory_bytes: process.memory(),
            status: process.status(),
            thread_count: process.thread_count(),
            timestamp: Utc::now(),
        }
    }
    
    /// Check if process exceeds CPU threshold
    pub fn exceeds_cpu_threshold(&self, max_per_thread: f32) -> bool {
        let max_allowed = max_per_thread * self.thread_count as f32;
        self.cpu_usage > max_allowed
    }
    
    /// Check if process exceeds memory threshold
    pub fn exceeds_memory_threshold(&self, max_bytes: u64) -> bool {
        self.memory_bytes > max_bytes
    }
}

/// System-wide health snapshot
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    /// Total CPU usage percentage
    pub total_cpu_usage: f32,
    
    /// Number of physical cores
    pub physical_cores: usize,
    
    /// Number of logical CPUs
    pub logical_cpus: usize,
    
    /// Total system memory in bytes
    pub total_memory: u64,
    
    /// Used system memory in bytes
    pub used_memory: u64,
    
    /// Available memory in bytes
    pub available_memory: u64,
    
    /// Number of running processes
    pub process_count: usize,
    
    /// Timestamp of snapshot
    pub timestamp: DateTime<Utc>,
    
    /// Top processes by CPU usage
    pub top_cpu_processes: Vec<ProcessSnapshot>,
    
    /// Top processes by memory usage
    pub top_memory_processes: Vec<ProcessSnapshot>,
}

impl SystemSnapshot {
    /// Collect current system state
    pub fn collect(top_n: usize) -> SystemHealthResult<Self> {
        let mut sys = System::new_all();
        sys.refresh_all();
        
        let total_cpu = sys.global_cpu_usage();
        let total_mem = sys.total_memory();
        let used_mem = sys.used_memory();
        
        // Get top processes
        let mut processes: Vec<(u32, &dyn ProcessExt)> = sys.processes().iter()
            .map(|(pid, proc)| (*pid, proc as &dyn ProcessExt))
            .collect();
        
        // Sort by CPU usage
        processes.sort_by(|a, b| {
            b.1.cpu_usage().partial_cmp(&a.1.cpu_usage()).unwrap()
        });
        
        let top_cpu: Vec<ProcessSnapshot> = processes.iter()
            .take(top_n)
            .map(|(pid, proc)| ProcessSnapshot::from_process(*proc, *pid))
            .collect();
        
        // Sort by memory usage
        processes.sort_by(|a, b| {
            b.1.memory().cmp(&a.1.memory())
        });
        
        let top_memory: Vec<ProcessSnapshot> = processes.iter()
            .take(top_n)
            .map(|(pid, proc)| ProcessSnapshot::from_process(*proc, *pid))
            .collect();
        
        Ok(Self {
            total_cpu_usage: total_cpu,
            physical_cores: sys.physical_core_count().unwrap_or(0),
            logical_cpus: sys.cpus().len(),
            total_memory: total_mem,
            used_memory: used_mem,
            available_memory: total_mem.saturating_sub(used_mem),
            process_count: sys.processes().len(),
            timestamp: Utc::now(),
            top_cpu_processes: top_cpu,
            top_memory_processes: top_memory,
        })
    }
    
    /// Get memory usage percentage
    pub fn memory_percent(&self) -> f32 {
        (self.used_memory as f32 / self.total_memory as f32) * 100.0
    }
}

/// Statistics for the health monitor
#[derive(Debug, Default)]
pub struct HealthMonitorStats {
    /// Number of checks performed
    pub checks_performed: AtomicU64,
    
    /// Number of warnings issued
    pub warnings_issued: AtomicU64,
    
    /// Number of processes terminated
    pub processes_terminated: AtomicU64,
    
    /// Last check timestamp (Unix seconds)
    pub last_check_timestamp: AtomicU64,
}

/// Configuration for the health monitor
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// Maximum CPU percentage per thread
    pub max_cpu_per_thread: f32,
    
    /// Maximum memory per process
    pub max_memory_per_process: u64,
    
    /// Maximum system memory percentage
    pub max_system_memory_percent: f32,
    
    /// Whether to auto-kill offending processes
    pub auto_kill_enabled: bool,
    
    /// Grace period before killing (milliseconds)
    pub grace_period_ms: u64,
    
    /// PIDs to exclude from monitoring (e.g., critical system processes)
    pub excluded_pids: Vec<u32>,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            max_cpu_per_thread: MAX_CPU_PERCENT_PER_THREAD,
            max_memory_per_process: DEFAULT_MAX_PROCESS_MEMORY,
            max_system_memory_percent: 90.0,
            auto_kill_enabled: true,
            grace_period_ms: 5000,
            excluded_pids: vec![1], // Exclude init/systemd
        }
    }
}

/// High-performance system health monitor daemon
pub struct SystemHealthMonitor {
    /// Configuration
    config: HealthMonitorConfig,
    
    /// Running flag
    running: Arc<AtomicBool>,
    
    /// Statistics
    stats: Arc<HealthMonitorStats>,
    
    /// Handle to monitor thread
    monitor_handle: Option<thread::JoinHandle<()>>,
    
    /// System info instance (reused)
    system: Arc<std::sync::Mutex<System>>,
    
    /// Processes that have been warned (for grace period tracking)
    warned_processes: Arc<std::sync::Mutex<Vec<(u32, Instant)>>>,
}

impl SystemHealthMonitor {
    /// Create a new health monitor with default configuration
    pub fn new() -> SystemHealthResult<Self> {
        Self::with_config(HealthMonitorConfig::default())
    }
    
    /// Create a new health monitor with custom configuration
    pub fn with_config(config: HealthMonitorConfig) -> SystemHealthResult<Self> {
        Ok(Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(HealthMonitorStats::default()),
            monitor_handle: None,
            system: Arc::new(std::sync::Mutex::new(System::new())),
            warned_processes: Arc::new(std::sync::Mutex::new(Vec::new())),
        })
    }
    
    /// Start the background monitoring loop
    pub fn start(&mut self) -> SystemHealthResult<()> {
        if self.running.load(Ordering::Relaxed) {
            return Err(SystemHealthError::CriticalFailure {
                message: "Monitor already running".to_string(),
            });
        }
        
        self.running.store(true, Ordering::Relaxed);
        
        let running = Arc::clone(&self.running);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();
        let system = Arc::clone(&self.system);
        let warned = Arc::clone(&self.warned_processes);
        
        let handle = thread::spawn(move || {
            Self::monitor_loop(running, stats, config, system, warned);
        });
        
        self.monitor_handle = Some(handle);
        Ok(())
    }
    
    /// Main monitoring loop
    fn monitor_loop(
        running: Arc<AtomicBool>,
        stats: Arc<HealthMonitorStats>,
        config: HealthMonitorConfig,
        system: Arc<std::sync::Mutex<System>>,
        warned: Arc<std::sync::Mutex<Vec<(u32, Instant)>>>,
    ) {
        let check_interval = Duration::from_millis(CHECK_INTERVAL_MS);
        
        while running.load(Ordering::Relaxed) {
            let start = Instant::now();
            
            // Refresh system info
            {
                let mut sys = system.lock().unwrap();
                sys.refresh_all();
                
                // Check each process
                for (pid, process) in sys.processes() {
                    let snapshot = ProcessSnapshot::from_process(process, *pid);
                    
                    // Skip excluded PIDs
                    if config.excluded_pids.contains(pid) {
                        continue;
                    }
                    
                    // Check CPU usage
                    if snapshot.exceeds_cpu_threshold(config.max_cpu_per_thread) {
                        Self::handle_violation(
                            &snapshot,
                            "CPU",
                            format!("{:.1}% (threads: {})", 
                                snapshot.cpu_usage, snapshot.thread_count),
                            &config,
                            &warned,
                            &stats,
                        );
                    }
                    
                    // Check memory usage
                    if snapshot.exceeds_memory_threshold(config.max_memory_per_process) {
                        Self::handle_violation(
                            &snapshot,
                            "Memory",
                            format!("{} MB", snapshot.memory_bytes / 1024 / 1024),
                            &config,
                            &warned,
                            &stats,
                        );
                    }
                }
                
                // Check system-wide memory
                let mem_percent = (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0;
                if mem_percent > config.max_system_memory_percent {
                    eprintln!(
                        "⚠️  SYSTEM WARNING: Memory usage at {:.1}% ({} GB / {} GB)",
                        mem_percent,
                        sys.used_memory() / 1024 / 1024 / 1024,
                        sys.total_memory() / 1024 / 1024 / 1024
                    );
                    stats.warnings_issued.fetch_add(1, Ordering::Relaxed);
                }
            }
            
            // Update stats
            stats.checks_performed.fetch_add(1, Ordering::Relaxed);
            stats.last_check_timestamp.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                Ordering::Relaxed,
            );
            
            // Clean up old warnings
            {
                let mut warned_lock = warned.lock().unwrap();
                warned_lock.retain(|(_, time)| time.elapsed().as_millis() < 60000);
            }
            
            // Sleep until next check
            let elapsed = start.elapsed();
            if elapsed < check_interval {
                thread::sleep(check_interval - elapsed);
            }
        }
    }
    
    /// Handle a resource violation
    fn handle_violation(
        snapshot: &ProcessSnapshot,
        resource_type: &str,
        value: String,
        config: &HealthMonitorConfig,
        warned: &std::sync::Mutex<Vec<(u32, Instant)>>,
        stats: &HealthMonitorStats,
    ) {
        let now = Instant::now();
        
        // Check if already warned
        let mut warned_lock = warned.lock().unwrap();
        let existing = warned_lock.iter_mut().find(|(pid, _)| pid == &snapshot.pid);
        
        match existing {
            Some((_, warn_time)) => {
                // Check if grace period has elapsed
                if warn_time.elapsed().as_millis() >= config.grace_period_ms as u128 {
                    // Grace period expired - take action
                    eprintln!(
                        "🚨 CRITICAL: Process {} (PID {}) exceeded {} limit: {} - TERMINATING",
                        snapshot.name, snapshot.pid, resource_type, value
                    );
                    
                    if config.auto_kill_enabled {
                        #[cfg(unix)]
                        {
                            // Send SIGTERM first
                            unsafe {
                                libc::kill(snapshot.pid as i32, libc::SIGTERM);
                            }
                        }
                        
                        stats.processes_terminated.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    // Remove from warned list
                    warned_lock.retain(|(pid, _)| pid != &snapshot.pid);
                }
            }
            None => {
                // First violation - issue warning and start grace period
                eprintln!(
                    "⚠️  WARNING: Process {} (PID {}) exceeded {} limit: {}",
                    snapshot.name, snapshot.pid, resource_type, value
                );
                warned_lock.push((snapshot.pid, now));
                stats.warnings_issued.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    
    /// Stop the monitoring loop
    pub fn stop(&mut self) -> SystemHealthResult<()> {
        self.running.store(false, Ordering::Relaxed);
        
        if let Some(handle) = self.monitor_handle.take() {
            handle.join().map_err(|_| SystemHealthError::CriticalFailure {
                message: "Failed to join monitor thread".to_string(),
            })?;
        }
        
        Ok(())
    }
    
    /// Get current system snapshot
    pub fn get_snapshot(&self, top_n: usize) -> SystemHealthResult<SystemSnapshot> {
        SystemSnapshot::collect(top_n)
    }
    
    /// Get monitor statistics
    pub fn get_stats(&self) -> HealthMonitorStatsSnapshot {
        HealthMonitorStatsSnapshot {
            checks_performed: self.stats.checks_performed.load(Ordering::Relaxed),
            warnings_issued: self.stats.warnings_issued.load(Ordering::Relaxed),
            processes_terminated: self.stats.processes_terminated.load(Ordering::Relaxed),
            last_check_timestamp: self.stats.last_check_timestamp.load(Ordering::Relaxed),
            is_running: self.running.load(Ordering::Relaxed),
        }
    }
    
    /// Check if a specific PID is healthy
    pub fn check_process_health(&self, pid: u32) -> SystemHealthResult<ProcessSnapshot> {
        let mut sys = self.system.lock().unwrap();
        sys.refresh_process(pid.into());
        
        sys.process(pid.into())
            .map(|proc| ProcessSnapshot::from_process(proc, pid))
            .ok_or(SystemHealthError::ProcessNotFound { pid })
    }
    
    /// Emergency kill function for immediate termination
    pub fn emergency_kill(pid: u32) -> SystemHealthResult<()> {
        eprintln!("🚨 EMERGENCY KILL: Terminating process {}", pid);
        
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
        }
        
        #[cfg(windows)]
        {
            use std::process::Command;
            Command::new("taskkill")
                .args(&["/F", "/PID", &pid.to_string()])
                .output()
                .map_err(|e| SystemHealthError::SystemInfo(e.to_string()))?;
        }
        
        Ok(())
    }
}

impl Default for SystemHealthMonitor {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Snapshot of monitor statistics
#[derive(Debug, Clone)]
pub struct HealthMonitorStatsSnapshot {
    pub checks_performed: u64,
    pub warnings_issued: u64,
    pub processes_terminated: u64,
    pub last_check_timestamp: u64,
    pub is_running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_system_snapshot() {
        let snapshot = SystemSnapshot::collect(5).expect("Failed to collect snapshot");
        
        assert!(snapshot.logical_cpus > 0);
        assert!(snapshot.total_memory > 0);
        assert!(snapshot.timestamp <= Utc::now());
    }
    
    #[test]
    fn test_process_snapshot() {
        let current_pid = process::id();
        let snapshot = SystemSnapshot::collect(100).expect("Failed to collect snapshot");
        
        // Find our own process
        let our_process = snapshot.top_cpu_processes.iter()
            .find(|p| p.pid == current_pid);
        
        assert!(our_process.is_some());
    }
    
    #[test]
    fn test_monitor_lifecycle() {
        let mut monitor = SystemHealthMonitor::new().expect("Failed to create monitor");
        
        assert!(monitor.start().is_ok());
        
        // Let it run briefly
        thread::sleep(Duration::from_millis(100));
        
        let stats = monitor.get_stats();
        assert!(stats.is_running);
        assert!(stats.checks_performed > 0);
        
        assert!(monitor.stop().is_ok());
        
        let stats = monitor.get_stats();
        assert!(!stats.is_running);
    }
}
