//! Replay Simulator - Isolates and replays specific event slices for debugging.
//!
//! This module provides controlled event replay capabilities for debugging
//! and testing specific scenarios. Critical for isolating bugs, reproducing
//! market conditions, and validating strategy behavior.
//!
//! Features:
//! - Selective event filtering by type, sequence range, or symbol
//! - Variable replay speed (real-time, accelerated, step-by-step)
//! - Event injection for scenario testing
//! - Replay state capture and comparison
//! - Breakpoint support for interactive debugging

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, Duration};
use parking_lot::RwLock;

/// Replay mode options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplayMode {
    /// Replay as fast as possible
    MaximumSpeed,
    /// Replay at original event timestamps
    RealTime,
    /// Replay with specified delay between events (milliseconds)
    FixedDelay(u64),
    /// Step-by-step manual advancement
    StepByStep,
}

/// Filter criteria for event selection
#[derive(Debug, Clone, Default)]
pub struct ReplayFilter {
    /// Include only events in this sequence range
    pub sequence_range: Option<(u64, u64)>,
    /// Include only these event types
    pub event_types: HashSet<String>,
    /// Include only these symbols
    pub symbols: HashSet<String>,
    /// Exclude these event types
    pub exclude_types: HashSet<String>,
    /// Minimum timestamp (microseconds)
    pub min_timestamp_us: Option<u64>,
    /// Maximum timestamp (microseconds)
    pub max_timestamp_us: Option<u64>,
}

impl ReplayFilter {
    pub fn new() -> Self {
        ReplayFilter::default()
    }

    pub fn with_sequence_range(mut self, start: u64, end: u64) -> Self {
        self.sequence_range = Some((start, end));
        self
    }

    pub fn with_event_type(mut self, event_type: &str) -> Self {
        self.event_types.insert(event_type.to_string());
        self
    }

    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbols.insert(symbol.to_string());
        self
    }

    pub fn excludes_event_type(mut self, event_type: &str) -> Self {
        self.exclude_types.insert(event_type.to_string());
        self
    }

    /// Check if an event passes the filter
    pub fn matches(&self, event: &crate::event_store::Event) -> bool {
        // Check sequence range
        if let Some((start, end)) = self.sequence_range {
            if event.sequence < start || event.sequence > end {
                return false;
            }
        }

        // Check event type inclusion
        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }

        // Check event type exclusion
        if self.exclude_types.contains(&event.event_type) {
            return false;
        }

        // Check symbol filter
        if !self.symbols.is_empty() {
            if let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&event.payload) {
                if let Some(symbol) = payload.get("symbol").and_then(|s| s.as_str()) {
                    if !self.symbols.contains(symbol) {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check timestamp range
        if let Some(min_ts) = self.min_timestamp_us {
            if event.timestamp_us < min_ts {
                return false;
            }
        }
        if let Some(max_ts) = self.max_timestamp_us {
            if event.timestamp_us > max_ts {
                return false;
            }
        }

        true
    }
}

/// Result of a replay operation
#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub total_events: u64,
    pub filtered_events: u64,
    pub replayed_events: u64,
    pub skipped_events: u64,
    pub duration_ms: f64,
    pub mode: ReplayMode,
    pub error_message: Option<String>,
}

impl ReplayResult {
    pub fn success(
        total: u64,
        filtered: u64,
        replayed: u64,
        skipped: u64,
        duration_ms: f64,
        mode: ReplayMode,
    ) -> Self {
        ReplayResult {
            total_events: total,
            filtered_events: filtered,
            replayed_events: replayed,
            skipped_events: skipped,
            duration_ms,
            mode,
            error_message: None,
        }
    }

    pub fn failure(error: &str) -> Self {
        ReplayResult {
            total_events: 0,
            filtered_events: 0,
            replayed_events: 0,
            skipped_events: 0,
            duration_ms: 0.0,
            mode: ReplayMode::MaximumSpeed,
            error_message: Some(error.to_string()),
        }
    }
}

/// State snapshot for comparison during replay
#[derive(Debug, Clone)]
pub struct ReplaySnapshot {
    pub sequence: u64,
    pub timestamp_us: u64,
    pub positions: HashMap<String, PositionState>,
    pub metrics: PortfolioMetrics,
}

#[derive(Debug, Clone)]
pub struct PositionState {
    pub quantity: i64,
    pub average_price: u64,
    pub realized_pnl: i64,
}

#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub total_pnl: i64,
    pub gross_exposure: u64,
    pub position_count: usize,
}

/// Event handler callback type
pub type EventHandler = Box<dyn Fn(&crate::event_store::Event) + Send + Sync>;

/// Replay simulator for controlled event playback
pub struct ReplaySimulator {
    /// Event store reference
    event_store: Arc<dyn EventStoreTrait + Send + Sync>,
    /// Current replay mode
    mode: RwLock<ReplayMode>,
    /// Current filter
    filter: RwLock<ReplayFilter>,
    /// Replay in progress flag
    is_replaying: AtomicBool,
    /// Pause flag for step-by-step mode
    is_paused: AtomicBool,
    /// Current replay position
    current_sequence: AtomicU64,
    /// Event handlers to call during replay
    handlers: Arc<RwLock<Vec<EventHandler>>>,
    /// Captured snapshots for comparison
    snapshots: Arc<RwLock<HashMap<u64, ReplaySnapshot>>>,
    /// Breakpoints (sequences where replay should pause)
    breakpoints: Arc<RwLock<HashSet<u64>>>,
}

/// Trait for event store integration
pub trait EventStoreTrait {
    fn get_events_from(&self, start_sequence: u64) -> Vec<crate::event_store::Event>;
    fn get_event(&self, sequence: u64) -> Option<crate::event_store::Event>;
    fn latest_sequence(&self) -> u64;
}

unsafe impl Send for ReplaySimulator {}
unsafe impl Sync for ReplaySimulator {}

impl ReplaySimulator {
    pub fn new(event_store: Arc<dyn EventStoreTrait + Send + Sync>) -> Self {
        ReplaySimulator {
            event_store,
            mode: RwLock::new(ReplayMode::MaximumSpeed),
            filter: RwLock::new(ReplayFilter::default()),
            is_replaying: AtomicBool::new(false),
            is_paused: AtomicBool::new(false),
            current_sequence: AtomicU64::new(0),
            handlers: Arc::new(RwLock::new(Vec::new())),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            breakpoints: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Set replay mode
    pub fn set_mode(&self, mode: ReplayMode) {
        *self.mode.write() = mode;
    }

    /// Set replay filter
    pub fn set_filter(&self, filter: ReplayFilter) {
        *self.filter.write() = filter;
    }

    /// Add event handler
    pub fn add_handler(&self, handler: EventHandler) {
        self.handlers.write().push(handler);
    }

    /// Clear all handlers
    pub fn clear_handlers(&self) {
        self.handlers.write().clear();
    }

    /// Add breakpoint at sequence
    pub fn add_breakpoint(&self, sequence: u64) {
        self.breakpoints.write().insert(sequence);
    }

    /// Remove breakpoint
    pub fn remove_breakpoint(&self, sequence: u64) {
        self.breakpoints.write().remove(&sequence);
    }

    /// Start replay from given sequence
    pub fn replay(&self, start_sequence: u64) -> ReplayResult {
        if self.is_replaying.swap(true, Ordering::SeqCst) {
            return ReplayResult::failure("Replay already in progress");
        }

        let start_time = Instant::now();
        let mode = *self.mode.read();
        let filter = self.filter.read().clone();

        // Fetch events
        let all_events = self.event_store.get_events_from(start_sequence);
        let total_events = all_events.len() as u64;

        // Apply filter
        let filtered_events: Vec<_> = all_events
            .into_iter()
            .filter(|e| filter.matches(e))
            .collect();
        let filtered_count = filtered_events.len() as u64;

        let mut replayed = 0u64;
        let mut skipped = 0u64;
        let mut last_timestamp_us = 0u64;

        // Replay events
        for event in &filtered_events {
            // Check for breakpoint
            if self.breakpoints.read().contains(&event.sequence) {
                self.is_paused.store(true, Ordering::Relaxed);
                
                // Wait until unpaused
                while self.is_paused.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(10));
                    
                    // Check if replay was cancelled
                    if !self.is_replaying.load(Ordering::Relaxed) {
                        self.is_replaying.store(false, Ordering::SeqCst);
                        return ReplayResult::failure("Replay cancelled at breakpoint");
                    }
                }
            }

            // Handle timing based on mode
            match mode {
                ReplayMode::RealTime => {
                    if last_timestamp_us > 0 && event.timestamp_us > last_timestamp_us {
                        let delay_us = event.timestamp_us - last_timestamp_us;
                        std::thread::sleep(Duration::from_micros(delay_us));
                    }
                }
                ReplayMode::FixedDelay(delay_ms) => {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                ReplayMode::StepByStep => {
                    self.is_paused.store(true, Ordering::Relaxed);
                    while self.is_paused.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
                ReplayMode::MaximumSpeed => {
                    // No delay
                }
            }

            // Call handlers
            let handlers = self.handlers.read();
            for handler in handlers.iter() {
                handler(event);
            }

            last_timestamp_us = event.timestamp_us;
            self.current_sequence.store(event.sequence, Ordering::Relaxed);
            replayed += 1;
        }

        let duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        skipped = filtered_count - replayed;

        self.is_replaying.store(false, Ordering::SeqCst);

        ReplayResult::success(
            total_events,
            filtered_count,
            replayed,
            skipped,
            duration_ms,
            mode,
        )
    }

    /// Cancel ongoing replay
    pub fn cancel(&self) {
        self.is_replaying.store(false, Ordering::SeqCst);
        self.is_paused.store(false, Ordering::Relaxed);
    }

    /// Resume paused replay
    pub fn resume(&self) {
        self.is_paused.store(false, Ordering::Relaxed);
    }

    /// Step forward one event (in StepByStep mode)
    pub fn step(&self) {
        self.is_paused.store(false, Ordering::Relaxed);
    }

    /// Capture state snapshot at current sequence
    pub fn capture_snapshot(
        &self,
        sequence: u64,
        positions: HashMap<String, PositionState>,
        metrics: PortfolioMetrics,
    ) {
        let snapshot = ReplaySnapshot {
            sequence,
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            positions,
            metrics,
        };

        self.snapshots.write().insert(sequence, snapshot);
    }

    /// Compare two snapshots
    pub fn compare_snapshots(&self, seq1: u64, seq2: u64) -> Option<SnapshotComparison> {
        let snapshots = self.snapshots.read();
        let snap1 = snapshots.get(&seq1)?;
        let snap2 = snapshots.get(&seq2)?;

        Some(SnapshotComparison {
            sequence_delta: seq2 as i64 - seq1 as i64,
            pnl_change: snap2.metrics.total_pnl - snap1.metrics.total_pnl,
            exposure_change: snap2.metrics.gross_exposure as i64 - snap1.metrics.gross_exposure as i64,
            position_changes: self.compare_positions(&snap1.positions, &snap2.positions),
        })
    }

    fn compare_positions(
        &self,
        pos1: &HashMap<String, PositionState>,
        pos2: &HashMap<String, PositionState>,
    ) -> Vec<PositionChange> {
        let mut changes = Vec::new();

        // Find changed and removed positions
        for (symbol, state1) in pos1 {
            if let Some(state2) = pos2.get(symbol) {
                if state1.quantity != state2.quantity
                    || state1.average_price != state2.average_price
                    || state1.realized_pnl != state2.realized_pnl
                {
                    changes.push(PositionChange {
                        symbol: symbol.clone(),
                        old_quantity: state1.quantity,
                        new_quantity: state2.quantity,
                        change_type: "modified",
                    });
                }
            } else {
                changes.push(PositionChange {
                    symbol: symbol.clone(),
                    old_quantity: state1.quantity,
                    new_quantity: 0,
                    change_type: "closed",
                });
            }
        }

        // Find new positions
        for (symbol, state2) in pos2 {
            if !pos1.contains_key(symbol) {
                changes.push(PositionChange {
                    symbol: symbol.clone(),
                    old_quantity: 0,
                    new_quantity: state2.quantity,
                    change_type: "opened",
                });
            }
        }

        changes
    }

    /// Get current replay status
    pub fn get_status(&self) -> ReplayStatus {
        ReplayStatus {
            is_replaying: self.is_replaying.load(Ordering::Relaxed),
            is_paused: self.is_paused.load(Ordering::Relaxed),
            current_sequence: self.current_sequence.load(Ordering::Relaxed),
            mode: *self.mode.read(),
            breakpoint_count: self.breakpoints.read().len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnapshotComparison {
    pub sequence_delta: i64,
    pub pnl_change: i64,
    pub exposure_change: i64,
    pub position_changes: Vec<PositionChange>,
}

#[derive(Debug, Clone)]
pub struct PositionChange {
    pub symbol: String,
    pub old_quantity: i64,
    pub new_quantity: i64,
    pub change_type: &'static str,
}

#[derive(Debug, Clone)]
pub struct ReplayStatus {
    pub is_replaying: bool,
    pub is_paused: bool,
    pub current_sequence: u64,
    pub mode: ReplayMode,
    pub breakpoint_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEventStore {
        events: Vec<crate::event_store::Event>,
    }

    impl EventStoreTrait for MockEventStore {
        fn get_events_from(&self, start: u64) -> Vec<crate::event_store::Event> {
            self.events.iter()
                .filter(|e| e.sequence >= start)
                .cloned()
                .collect()
        }

        fn get_event(&self, sequence: u64) -> Option<crate::event_store::Event> {
            self.events.iter().find(|e| e.sequence == sequence).cloned()
        }

        fn latest_sequence(&self) -> u64 {
            self.events.last().map(|e| e.sequence).unwrap_or(0)
        }
    }

    #[test]
    fn test_replay_filter() {
        let filter = ReplayFilter::new()
            .with_sequence_range(10, 20)
            .with_event_type("OrderFilled")
            .with_symbol("BTC/USDT");

        assert!(filter.sequence_range == Some((10, 20)));
        assert!(filter.event_types.contains("OrderFilled"));
        assert!(filter.symbols.contains("BTC/USDT"));
    }

    #[test]
    fn test_replay_status() {
        let events = vec![];
        let store = Arc::new(MockEventStore { events });
        let simulator = ReplaySimulator::new(store);

        let status = simulator.get_status();
        assert!(!status.is_replaying);
        assert!(!status.is_paused);
        assert_eq!(status.current_sequence, 0);
    }
}
