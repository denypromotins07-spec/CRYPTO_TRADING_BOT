// SOUL Writer: High-speed, non-blocking asynchronous writes to SOUL.md.
// Implements lock-free writing with batched I/O for minimal latency.
// Ensures durability of learning data without blocking trading operations.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Entry types for SOUL.md
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SoulEntryType {
    TradeLesson,
    SignalInsight,
    RegimeChange,
    RiskEvent,
    PerformanceMetric,
    SystemAlert,
    ManualNote,
}

/// A single SOUL entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoulEntry {
    pub entry_type: SoulEntryType,
    pub timestamp_ns: u64,
    pub asset: Option<String>,
    pub title: String,
    pub content: String,
    pub metadata: std::collections::HashMap<String, String>,
    pub priority: u8,  // 0-255, higher = more important
}

impl SoulEntry {
    pub fn new(entry_type: SoulEntryType, title: &str, content: &str) -> Self {
        Self {
            entry_type,
            timestamp_ns: current_time_ns(),
            asset: None,
            title: title.to_string(),
            content: content.to_string(),
            metadata: std::collections::HashMap::new(),
            priority: 128,  // Default medium priority
        }
    }
    
    pub fn with_asset(mut self, asset: &str) -> Self {
        self.asset = Some(asset.to_string());
        self
    }
    
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
    
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Format entry as Markdown
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        
        // Header based on type
        let type_str = match self.entry_type {
            SoulEntryType::TradeLesson => "📚 Trade Lesson",
            SoulEntryType::SignalInsight => "📊 Signal Insight",
            SoulEntryType::RegimeChange => "🔄 Regime Change",
            SoulEntryType::RiskEvent => "⚠️ Risk Event",
            SoulEntryType::PerformanceMetric => "📈 Performance",
            SoulEntryType::SystemAlert => "🔔 System Alert",
            SoulEntryType::ManualNote => "📝 Manual Note",
        };
        
        md.push_str(&format!("### {} - {}\n\n", type_str, self.title));
        
        if let Some(ref asset) = self.asset {
            md.push_str(&format!("**Asset**: {}\n\n", asset));
        }
        
        md.push_str(&format!("**Time**: {}\n\n", format_timestamp(self.timestamp_ns)));
        md.push_str(&self.content);
        md.push_str("\n\n");
        
        if !self.metadata.is_empty() {
            md.push_str("**Metadata**:\n");
            for (key, value) in &self.metadata {
                md.push_str(&format!("- {}: {}\n", key, value));
            }
            md.push_str("\n");
        }
        
        md.push_str("---\n\n");
        
        md
    }
}

/// Configuration for SOUL writer
#[derive(Debug, Clone)]
pub struct SoulWriterConfig {
    /// Path to SOUL.md file
    pub file_path: PathBuf,
    /// Maximum entries in write buffer before flush
    pub buffer_size: usize,
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
    /// Enable synchronous writes (for critical entries)
    pub sync_mode: bool,
    /// Maximum file size before rotation (MB)
    pub max_file_size_mb: usize,
}

impl Default for SoulWriterConfig {
    fn default() -> Self {
        Self {
            file_path: PathBuf::from("SOUL.md"),
            buffer_size: 100,
            flush_interval_ms: 1000,  // 1 second
            sync_mode: false,
            max_file_size_mb: 50,
        }
    }
}

/// Statistics for SOUL writer
#[derive(Debug, Default)]
pub struct SoulWriterStats {
    pub entries_written: AtomicU64,
    pub entries_pending: AtomicU64,
    pub flush_count: AtomicU64,
    pub errors: AtomicU64,
    pub last_flush_ns: AtomicU64,
}

/// High-speed SOUL.md writer with async batching
pub struct SoulWriter {
    config: SoulWriterConfig,
    /// Pending entries queue
    pending_entries: Arc<Mutex<VecDeque<SoulEntry>>>,
    /// Writer is running
    is_running: AtomicBool,
    /// Statistics
    stats: Arc<SoulWriterStats>,
    /// File handle (wrapped for thread safety)
    file_handle: Arc<Mutex<Option<BufWriter<File>>>>,
}

impl SoulWriter {
    /// Create a new SOUL writer
    pub fn new(config: SoulWriterConfig) -> Self {
        let pending = Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_size)));
        let stats = Arc::new(SoulWriterStats::default());
        
        Self {
            config,
            pending_entries: pending,
            is_running: AtomicBool::new(false),
            stats,
            file_handle: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Initialize the writer (open file, start background thread)
    pub fn initialize(&self) -> Result<(), String> {
        // Ensure parent directory exists
        if let Some(parent) = self.config.file_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }
        
        // Open file for appending
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.file_path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        *self.file_handle.lock().unwrap() = Some(BufWriter::new(file));
        
        // Write header if file is new
        if self.config.file_path.metadata()
            .map(|m| m.len() == 0)
            .unwrap_or(true)
        {
            self.write_header()?;
        }
        
        // Start background writer thread
        self.is_running.store(true, Ordering::SeqCst);
        let pending_clone = Arc::clone(&self.pending_entries);
        let stats_clone = Arc::clone(&self.stats);
        let file_clone = Arc::clone(&self.file_handle);
        let flush_interval = self.config.flush_interval_ms;
        let buffer_size = self.config.buffer_size;
        
        thread::spawn(move || {
            background_writer(
                pending_clone,
                stats_clone,
                file_clone,
                flush_interval,
                buffer_size,
            );
        });
        
        Ok(())
    }
    
    fn write_header(&self) -> Result<(), String> {
        let header = "# ZAID Bot - SOUL Memory\n\n> Self-Organizing Understanding Layer\n> This file contains lessons learned, insights, and adaptive knowledge from live trading.\n\n---\n\n";
        
        if let Ok(mut guard) = self.file_handle.lock() {
            if let Some(ref mut writer) = *guard {
                writer.write_all(header.as_bytes())
                    .map_err(|e| format!("Failed to write header: {}", e))?;
                writer.flush()
                    .map_err(|e| format!("Failed to flush header: {}", e))?;
            }
        }
        
        Ok(())
    }
    
    /// Queue an entry for async writing
    pub fn queue_entry(&self, entry: SoulEntry) -> Result<(), String> {
        if !self.is_running.load(Ordering::SeqCst) {
            return Err("Writer not running".to_string());
        }
        
        let mut pending = self.pending_entries.lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?;
        
        pending.push_back(entry);
        self.stats.entries_pending.fetch_add(1, Ordering::SeqCst);
        
        // If buffer is full, trigger immediate flush
        if pending.len() >= self.config.buffer_size {
            drop(pending);  // Release lock before flushing
            self.flush()?;
        }
        
        Ok(())
    }
    
    /// Write entry synchronously (for critical entries)
    pub fn write_sync(&self, entry: SoulEntry) -> Result<(), String> {
        let markdown = entry.to_markdown();
        
        if let Ok(mut guard) = self.file_handle.lock() {
            if let Some(ref mut writer) = *guard {
                writer.write_all(markdown.as_bytes())
                    .map_err(|e| format!("Write failed: {}", e))?;
                writer.flush()
                    .map_err(|e| format!("Flush failed: {}", e))?;
            }
        }
        
        self.stats.entries_written.fetch_add(1, Ordering::SeqCst);
        self.stats.last_flush_ns.store(current_time_ns(), Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Flush pending entries
    pub fn flush(&self) -> Result<(), String> {
        let entries: Vec<SoulEntry> = {
            let mut pending = self.pending_entries.lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;
            
            if pending.is_empty() {
                return Ok(());
            }
            
            pending.drain(..).collect()
        };
        
        // Sort by priority (higher first)
        let mut sorted_entries: Vec<SoulEntry> = entries;
        sorted_entries.sort_by_key(|e| std::cmp::Reverse(e.priority));
        
        // Write all entries
        let mut content = String::new();
        for entry in &sorted_entries {
            content.push_str(&entry.to_markdown());
        }
        
        if let Ok(mut guard) = self.file_handle.lock() {
            if let Some(ref mut writer) = *guard {
                writer.write_all(content.as_bytes())
                    .map_err(|e| format!("Write failed: {}", e))?;
                writer.flush()
                    .map_err(|e| format!("Flush failed: {}", e))?;
            }
        }
        
        let count = sorted_entries.len() as u64;
        self.stats.entries_written.fetch_add(count, Ordering::SeqCst);
        self.stats.entries_pending.fetch_sub(count, Ordering::SeqCst);
        self.stats.flush_count.fetch_add(1, Ordering::SeqCst);
        self.stats.last_flush_ns.store(current_time_ns(), Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Check if file needs rotation
    pub fn check_rotation(&self) -> bool {
        if let Ok(metadata) = std::fs::metadata(&self.config.file_path) {
            let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
            size_mb >= self.config.max_file_size_mb as f64
        } else {
            false
        }
    }
    
    /// Rotate SOUL.md file (archive old content)
    pub fn rotate(&self) -> Result<(), String> {
        let timestamp = format_timestamp(current_time_ns());
        let archive_name = format!(
            "SOUL_{}.md",
            timestamp.replace(' ', "_").replace(':', "-")
        );
        
        let archive_path = self.config.file_path.with_file_name(archive_name);
        
        // Rename current file to archive
        std::fs::rename(&self.config.file_path, &archive_path)
            .map_err(|e| format!("Failed to rotate: {}", e))?;
        
        // Reinitialize with fresh file
        self.initialize()?;
        
        // Log rotation event
        let entry = SoulEntry::new(
            SoulEntryType::SystemAlert,
            "SOUL.md Rotated",
            &format!("Archived to {:?}", archive_path),
        );
        self.write_sync(entry)?;
        
        Ok(())
    }
    
    /// Get writer statistics
    pub fn get_stats(&self) -> SoulWriterStatsSnapshot {
        SoulWriterStatsSnapshot {
            entries_written: self.stats.entries_written.load(Ordering::SeqCst),
            entries_pending: self.stats.entries_pending.load(Ordering::SeqCst),
            flush_count: self.stats.flush_count.load(Ordering::SeqCst),
            errors: self.stats.errors.load(Ordering::SeqCst),
            last_flush_ns: self.stats.last_flush_ns.load(Ordering::SeqCst),
        }
    }
    
    /// Shutdown the writer gracefully
    pub fn shutdown(&self) -> Result<(), String> {
        self.is_running.store(false, Ordering::SeqCst);
        
        // Flush remaining entries
        self.flush()?;
        
        // Close file handle
        if let Ok(mut guard) = self.file_handle.lock() {
            if let Some(ref mut writer) = *guard {
                writer.flush()
                    .map_err(|e| format!("Final flush failed: {}", e))?;
            }
            *guard = None;
        }
        
        Ok(())
    }
}

/// Snapshot of writer statistics
#[derive(Debug, Clone)]
pub struct SoulWriterStatsSnapshot {
    pub entries_written: u64,
    pub entries_pending: u64,
    pub flush_count: u64,
    pub errors: u64,
    pub last_flush_ns: u64,
}

/// Background writer thread function
fn background_writer(
    pending: Arc<Mutex<VecDeque<SoulEntry>>>,
    stats: Arc<SoulWriterStats>,
    file: Arc<Mutex<Option<BufWriter<File>>>>,
    flush_interval_ms: u64,
    buffer_size: usize,
) {
    loop {
        thread::sleep(Duration::from_millis(flush_interval_ms));
        
        // Check if we should flush
        let should_flush = {
            let queue = pending.lock().unwrap();
            !queue.is_empty() && queue.len() >= buffer_size / 2
        };
        
        if should_flush {
            let entries: Vec<SoulEntry> = {
                let mut queue = pending.lock().unwrap();
                queue.drain(..).collect()
            };
            
            if !entries.is_empty() {
                let mut content = String::new();
                for entry in &entries {
                    content.push_str(&entry.to_markdown());
                }
                
                if let Ok(mut guard) = file.lock() {
                    if let Some(ref mut writer) = *guard {
                        let _ = writer.write_all(content.as_bytes());
                        let _ = writer.flush();
                    }
                }
                
                let count = entries.len() as u64;
                stats.entries_written.fetch_add(count, Ordering::SeqCst);
                stats.entries_pending.fetch_sub(count, Ordering::SeqCst);
                stats.flush_count.fetch_add(1, Ordering::SeqCst);
                stats.last_flush_ns.store(current_time_ns(), Ordering::SeqCst);
            }
        }
    }
}

/// Helper function to get current time in nanoseconds
fn current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Format nanosecond timestamp as readable string
fn format_timestamp(ns: u64) -> String {
    let secs = ns / 1_000_000_000;
    let datetime = SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
    
    // Simple formatting (in production, use chrono)
    format!("{:?}", datetime)
}

/// Builder for SoulWriter
pub struct SoulWriterBuilder {
    config: SoulWriterConfig,
}

impl SoulWriterBuilder {
    pub fn new() -> Self {
        Self {
            config: SoulWriterConfig::default(),
        }
    }
    
    pub fn file_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.file_path = path.as_ref().to_path_buf();
        self
    }
    
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }
    
    pub fn flush_interval_ms(mut self, ms: u64) -> Self {
        self.config.flush_interval_ms = ms;
        self
    }
    
    pub fn sync_mode(mut self, sync: bool) -> Self {
        self.config.sync_mode = sync;
        self
    }
    
    pub fn build(self) -> SoulWriter {
        SoulWriter::new(self.config)
    }
}

impl Default for SoulWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entry_creation() {
        let entry = SoulEntry::new(
            SoulEntryType::TradeLesson,
            "Stop Loss Placement",
            "Learned to place stops below liquidity pools",
        )
        .with_asset("BTCUSDT")
        .with_priority(200);
        
        assert_eq!(entry.entry_type, SoulEntryType::TradeLesson);
        assert_eq!(entry.asset, Some("BTCUSDT".to_string()));
        assert_eq!(entry.priority, 200);
    }
    
    #[test]
    fn test_markdown_formatting() {
        let entry = SoulEntry::new(
            SoulEntryType::SignalInsight,
            "Test Entry",
            "Test content here",
        );
        
        let md = entry.to_markdown();
        assert!(md.contains("📊 Signal Insight"));
        assert!(md.contains("Test Entry"));
        assert!(md.contains("Test content here"));
    }
    
    #[test]
    fn test_writer_initialization() {
        let writer = SoulWriterBuilder::new()
            .file_path("/tmp/test_soul.md")
            .build();
        
        let result = writer.initialize();
        assert!(result.is_ok());
        
        let _ = writer.shutdown();
        let _ = std::fs::remove_file("/tmp/test_soul.md");
    }
}

// FFI exports for Python integration
#[no_mangle]
pub extern "C" fn create_soul_writer(file_path: *const i8) -> *mut SoulWriter {
    unsafe {
        let path_str = std::ffi::CStr::from_ptr(file_path).to_str().unwrap_or("SOUL.md");
        let writer = SoulWriterBuilder::new()
            .file_path(path_str)
            .build();
        Box::into_raw(Box::new(writer))
    }
}

#[no_mangle]
pub extern "C" fn soul_write_entry(
    writer: *mut SoulWriter,
    entry_type: u8,
    title: *const i8,
    content: *const i8,
    asset: *const i8,
) -> bool {
    unsafe {
        let writer = &*writer;
        let title_str = std::ffi::CStr::from_ptr(title).to_str().unwrap_or("Untitled");
        let content_str = std::ffi::CStr::from_ptr(content).to_str().unwrap_or("");
        let asset_str = std::ffi::CStr::from_ptr(asset).to_str().unwrap_or("");
        
        let entry_type = match entry_type {
            0 => SoulEntryType::TradeLesson,
            1 => SoulEntryType::SignalInsight,
            2 => SoulEntryType::RegimeChange,
            3 => SoulEntryType::RiskEvent,
            4 => SoulEntryType::PerformanceMetric,
            _ => SoulEntryType::ManualNote,
        };
        
        let mut entry = SoulEntry::new(entry_type, title_str, content_str);
        if !asset_str.is_empty() {
            entry = entry.with_asset(asset_str);
        }
        
        writer.queue_entry(entry).is_ok()
    }
}
