//! TWAP Scheduler: Time-Weighted Average Price Execution
//! Slices large parent orders into micro-child orders for minimal market impact.
//! Optimized for zero-cost abstractions and deterministic execution.
//! 
//! Stage 13: Advanced Execution Algorithms
//! Target: Minimize market impact to secure 8k-20k INR/hour

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;

/// Configuration for TWAP execution
#[derive(Debug, Clone)]
pub struct TwapConfig {
    /// Total duration of execution in seconds
    pub total_duration_secs: u64,
    /// Number of slices to divide the order into
    pub num_slices: u64,
    /// Minimum time between slices in milliseconds
    pub min_interval_ms: u64,
    /// Randomization window in milliseconds (to avoid predictable patterns)
    pub randomization_window_ms: u64,
    /// Minimum slice size in base units
    pub min_slice_size: u64,
    /// Maximum slice size in base units
    pub max_slice_size: u64,
}

impl Default for TwapConfig {
    fn default() -> Self {
        Self {
            total_duration_secs: 3600, // 1 hour default
            num_slices: 60,            // One slice per minute
            min_interval_ms: 1000,     // Minimum 1 second between slices
            randomization_window_ms: 500, // ±500ms randomization
            min_slice_size: 100,
            max_slice_size: 1000000,
        }
    }
}

/// TWAP slice request
#[derive(Debug, Clone)]
pub struct TwapSlice {
    pub slice_id: u64,
    pub scheduled_time_ns: u64,
    pub quantity: u64,
    pub side: OrderSide,
    pub is_final_slice: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// TWAP Execution State Machine
pub struct TwapScheduler {
    config: TwapConfig,
    state: TwapState,
    parent_order_id: u64,
    total_quantity: u64,
    executed_quantity: u64,
    slices_created: u64,
    next_slice_index: u64,
    start_time_ns: u64,
    end_time_ns: u64,
    slice_quantities: Vec<u64>,
    pending_slices: VecDeque<TwapSlice>,
    is_active: AtomicBool,
    last_execution_time: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TwapState {
    Idle,
    Scheduled,
    Running,
    Paused,
    Completed,
    Cancelled,
    Failed,
}

impl TwapScheduler {
    pub fn new(config: TwapConfig, parent_order_id: u64, total_quantity: u64) -> Self {
        let mut scheduler = Self {
            config,
            state: TwapState::Idle,
            parent_order_id,
            total_quantity,
            executed_quantity: 0,
            slices_created: 0,
            next_slice_index: 0,
            start_time_ns: 0,
            end_time_ns: 0,
            slice_quantities: Vec::new(),
            pending_slices: VecDeque::new(),
            is_active: AtomicBool::new(false),
            last_execution_time: AtomicU64::new(0),
        };
        
        // Pre-calculate slice quantities
        scheduler.calculate_slice_quantities();
        
        scheduler
    }

    /// Calculate optimal slice quantities
    fn calculate_slice_quantities(&mut self) {
        let num_slices = self.config.num_slices.min(self.total_quantity / self.config.min_slice_size.max(1));
        
        if num_slices == 0 {
            self.slice_quantities = vec![self.total_quantity];
            return;
        }
        
        let base_quantity = self.total_quantity / num_slices;
        let remainder = self.total_quantity % num_slices;
        
        self.slice_quantities = (0..num_slices)
            .map(|i| {
                let qty = base_quantity + if i < remainder { 1 } else { 0 };
                qty.clamp(self.config.min_slice_size, self.config.max_slice_size)
            })
            .collect();
    }

    /// Schedule the TWAP execution
    pub fn schedule(&mut self, start_time_ns: u64) -> bool {
        if self.total_quantity == 0 || self.state != TwapState::Idle {
            return false;
        }

        self.start_time_ns = start_time_ns;
        self.end_time_ns = start_time_ns + (self.config.total_duration_secs * 1_000_000_000);
        
        // Calculate interval between slices
        let interval_ns = (self.config.total_duration_secs * 1_000_000_000) / self.slice_quantities.len() as u64;
        
        // Create all slices
        for (i, &qty) in self.slice_quantities.iter().enumerate() {
            // Add randomization to scheduled time
            let randomization_ns = if self.config.randomization_window_ms > 0 {
                // In production, use proper RNG seeded securely
                ((i as u64) * 17 % self.config.randomization_window_ms) * 1_000_000
            } else {
                0
            };
            
            let scheduled_time = self.start_time_ns + (i as u64 * interval_ns) + randomization_ns;
            
            let slice = TwapSlice {
                slice_id: i as u64,
                scheduled_time_ns: scheduled_time,
                quantity: qty,
                side: OrderSide::Buy, // Would be set from parent order
                is_final_slice: i == self.slice_quantities.len() - 1,
            };
            
            self.pending_slices.push_back(slice);
            self.slices_created += 1;
        }
        
        self.state = TwapState::Scheduled;
        true
    }

    /// Start TWAP execution
    pub fn start(&mut self) -> bool {
        if self.state != TwapState::Scheduled {
            return false;
        }
        
        self.state = TwapState::Running;
        self.is_active.store(true, Ordering::Relaxed);
        true
    }

    /// Get next slice ready for execution
    pub fn get_next_slice(&mut self, current_time_ns: u64) -> Option<TwapSlice> {
        if !self.is_active.load(Ordering::Relaxed) || self.pending_slices.is_empty() {
            return None;
        }

        // Check minimum interval since last execution
        let last_exec = self.last_execution_time.load(Ordering::Relaxed);
        if last_exec > 0 {
            let elapsed_since_last = current_time_ns.saturating_sub(last_exec);
            let min_interval_ns = self.config.min_interval_ms * 1_000_000;
            
            if elapsed_since_last < min_interval_ns {
                return None;
            }
        }

        // Peek at next slice
        if let Some(next_slice) = self.pending_slices.front() {
            // Check if it's time to execute (with some tolerance)
            if current_time_ns >= next_slice.scheduled_time_ns {
                if let Some(slice) = self.pending_slices.pop_front() {
                    self.next_slice_index += 1;
                    self.last_execution_time.store(current_time_ns, Ordering::Relaxed);
                    return Some(slice);
                }
            }
        }

        None
    }

    /// Record a fill for a slice
    pub fn record_fill(&mut self, slice_id: u64, filled_quantity: u64) {
        self.executed_quantity += filled_quantity;
        
        // Check completion
        if self.executed_quantity >= self.total_quantity {
            self.state = TwapState::Completed;
            self.is_active.store(false, Ordering::Relaxed);
        }
    }

    /// Pause TWAP execution
    pub fn pause(&mut self) {
        if self.state == TwapState::Running {
            self.state = TwapState::Paused;
            self.is_active.store(false, Ordering::Relaxed);
        }
    }

    /// Resume paused TWAP execution
    pub fn resume(&mut self) -> bool {
        if self.state == TwapState::Paused {
            self.state = TwapState::Running;
            self.is_active.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Cancel TWAP execution
    pub fn cancel(&mut self) {
        self.state = TwapState::Cancelled;
        self.is_active.store(false, Ordering::Relaxed);
        self.pending_slices.clear();
    }

    /// Get execution progress (0.0 to 1.0)
    pub fn get_progress(&self) -> f64 {
        if self.total_quantity == 0 {
            return 0.0;
        }
        self.executed_quantity as f64 / self.total_quantity as f64
    }

    /// Get time progress (0.0 to 1.0)
    pub fn get_time_progress(&self, current_time_ns: u64) -> f64 {
        if self.start_time_ns == 0 || self.end_time_ns == 0 {
            return 0.0;
        }
        
        let total_duration = self.end_time_ns - self.start_time_ns;
        let elapsed = current_time_ns.saturating_sub(self.start_time_ns);
        
        (elapsed as f64 / total_duration as f64).min(1.0)
    }

    /// Get remaining quantity
    pub fn remaining_quantity(&self) -> u64 {
        self.total_quantity.saturating_sub(self.executed_quantity)
    }

    /// Get number of pending slices
    pub fn pending_slice_count(&self) -> usize {
        self.pending_slices.len()
    }

    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        self.state == TwapState::Completed
    }

    /// Get estimated completion time
    pub fn estimated_completion_ns(&self) -> u64 {
        self.end_time_ns
    }

    /// Reset scheduler for new order
    pub fn reset(&mut self, new_parent_id: u64, new_total_quantity: u64) {
        self.state = TwapState::Idle;
        self.parent_order_id = new_parent_id;
        self.total_quantity = new_total_quantity;
        self.executed_quantity = 0;
        self.slices_created = 0;
        self.next_slice_index = 0;
        self.start_time_ns = 0;
        self.end_time_ns = 0;
        self.pending_slices.clear();
        self.is_active.store(false, Ordering::Relaxed);
        self.last_execution_time.store(0, Ordering::Relaxed);
        
        self.calculate_slice_quantities();
    }
}

/// Dead zone detector for TWAP scheduling
pub struct DeadZoneDetector {
    min_volume_threshold: u64,
    dead_zone_windows: Vec<(u64, u64)>, // (start_ns, end_ns) pairs
}

impl DeadZoneDetector {
    pub fn new(min_volume_threshold: u64) -> Self {
        Self {
            min_volume_threshold,
            dead_zone_windows: Vec::new(),
        }
    }

    /// Detect dead zones based on historical volume
    pub fn detect_dead_zones(&mut self, volume_profile: &[(u64, u64)]) {
        self.dead_zone_windows.clear();
        
        let mut in_dead_zone = false;
        let mut zone_start = 0;
        
        for &(time_ns, volume) in volume_profile {
            if volume < self.min_volume_threshold && !in_dead_zone {
                in_dead_zone = true;
                zone_start = time_ns;
            } else if volume >= self.min_volume_threshold && in_dead_zone {
                in_dead_zone = false;
                self.dead_zone_windows.push((zone_start, time_ns));
            }
        }
    }

    /// Check if current time is in a dead zone
    pub fn is_in_dead_zone(&self, current_time_ns: u64) -> bool {
        self.dead_zone_windows.iter().any(|&(start, end)| {
            current_time_ns >= start && current_time_ns <= end
        })
    }

    /// Get next active period after dead zone
    pub fn next_active_period(&self, current_time_ns: u64) -> Option<u64> {
        for &(start, end) in &self.dead_zone_windows {
            if current_time_ns < end {
                return Some(end);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_scheduling() {
        let config = TwapConfig {
            total_duration_secs: 60,
            num_slices: 10,
            ..Default::default()
        };
        
        let mut scheduler = TwapScheduler::new(config, 12345, 10000);
        let start_time = 1000000000000;
        
        assert!(scheduler.schedule(start_time));
        assert_eq!(scheduler.pending_slice_count(), 10);
        
        assert!(scheduler.start());
        
        // Get first slice
        let slice = scheduler.get_next_slice(start_time);
        assert!(slice.is_some());
        assert_eq!(slice.unwrap().quantity, 1000); // 10000 / 10
    }

    #[test]
    fn test_dead_zone_detection() {
        let mut detector = DeadZoneDetector::new(1000);
        
        // Volume profile with dead zone in the middle
        let profile = vec![
            (1000, 5000),
            (2000, 4000),
            (3000, 500),   // Dead zone starts
            (4000, 300),
            (5000, 600),
            (6000, 2000),  // Dead zone ends
            (7000, 3000),
        ];
        
        detector.detect_dead_zones(&profile);
        
        assert!(detector.is_in_dead_zone(4000));
        assert!(!detector.is_in_dead_zone(2000));
        assert!(!detector.is_in_dead_zone(7000));
    }
}
