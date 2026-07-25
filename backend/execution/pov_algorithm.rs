//! POV (Percentage of Volume) Algorithm
//! Executes orders as a percentage of real-time market volume.
//! Tracks Binance's trade stream for accurate volume pacing.
//! 
//! Stage 13: Advanced Execution Algorithms
//! Target: Minimize market impact to secure 8k-20k INR/hour

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crossbeam_channel::{Receiver, Sender, try_recv};

/// Configuration for POV execution
#[derive(Debug, Clone)]
pub struct PovConfig {
    /// Target percentage of market volume (0.0 to 1.0)
    pub target_percentage: f64,
    /// Minimum order size in base units
    pub min_order_size: u64,
    /// Maximum order size in base units
    pub max_order_size: u64,
    /// Aggression multiplier for adverse moves
    pub aggression_multiplier: f64,
    /// Maximum deviation from target percentage
    pub max_deviation_bps: u64,
}

impl Default for PovConfig {
    fn default() -> Self {
        Self {
            target_percentage: 0.10, // 10% of market volume
            min_order_size: 1000,    // Minimum 1000 units (adjusted per asset)
            max_order_size: 1000000, // Maximum 1M units
            aggression_multiplier: 1.5,
            max_deviation_bps: 200,  // 2% max deviation
        }
    }
}

/// Real-time market volume tick
#[derive(Debug, Clone, Copy)]
pub struct VolumeTick {
    pub timestamp_ns: u64,
    pub price_bps: i64,
    pub volume: u64,
    pub side: TradeSide,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
    Unknown,
}

/// POV Execution State Machine
pub struct PovExecutor {
    config: PovConfig,
    state: PovState,
    parent_order_size: u64,
    executed_size: u64,
    market_volume_tracked: u64,
    last_adjustment_time: Instant,
    is_active: AtomicBool,
    current_aggression: AtomicU64, // Stored as basis points * 100
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PovState {
    Idle,
    Tracking,
    Executing,
    Paused,
    Completed,
    Cancelled,
}

impl PovExecutor {
    pub fn new(config: PovConfig, parent_order_size: u64) -> Self {
        Self {
            config,
            state: PovState::Idle,
            parent_order_size,
            executed_size: 0,
            market_volume_tracked: 0,
            last_adjustment_time: Instant::now(),
            is_active: AtomicBool::new(false),
            current_aggression: AtomicU64::new((config.target_percentage * 10000.0) as u64),
        }
    }

    /// Start POV execution
    #[inline]
    pub fn start(&mut self) {
        if self.parent_order_size == 0 {
            return;
        }
        self.state = PovState::Tracking;
        self.is_active.store(true, Ordering::Relaxed);
        self.last_adjustment_time = Instant::now();
    }

    /// Process incoming market volume tick
    #[inline]
    pub fn process_tick(&mut self, tick: VolumeTick) -> Option<u64> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        match self.state {
            PovState::Idle | PovState::Completed | PovState::Cancelled => return None,
            PovState::Paused => return None,
            PovState::Tracking | PovState::Executing => {}
        }

        // Update tracked market volume
        self.market_volume_tracked += tick.volume;

        // Calculate target execution based on market volume
        let target_executed = (self.market_volume_tracked as f64 
            * self.current_aggression.load(Ordering::Relaxed) as f64 / 10000.0) as u64;

        // Determine if we need to execute
        let remaining_parent = self.parent_order_size.saturating_sub(self.executed_size);
        
        if remaining_parent == 0 {
            self.state = PovState::Completed;
            self.is_active.store(false, Ordering::Relaxed);
            return None;
        }

        // Calculate execution quantity
        let target_qty = target_executed.saturating_sub(self.executed_size);
        
        if target_qty >= self.config.min_order_size && target_qty <= remaining_parent {
            let exec_qty = target_qty.min(self.config.max_order_size).min(remaining_parent);
            
            // Execute the order
            self.executed_size += exec_qty;
            self.state = PovState::Executing;
            
            // Check completion
            if self.executed_size >= self.parent_order_size {
                self.state = PovState::Completed;
                self.is_active.store(false, Ordering::Relaxed);
            }
            
            return Some(exec_qty);
        }

        None
    }

    /// Adjust aggression based on execution progress
    #[inline]
    pub fn adjust_aggression(&mut self, execution_deficit_bps: u64) {
        let current = self.current_aggression.load(Ordering::Relaxed) as f64 / 10000.0;
        
        if execution_deficit_bps > self.config.max_deviation_bps {
            // Increase aggression if falling behind
            let new_aggression = (current * self.config.aggression_multiplier)
                .min(1.0)
                .max(0.01);
            self.current_aggression.store((new_aggression * 10000.0) as u64, Ordering::Relaxed);
        } else if execution_deficit_bps == 0 && current > self.config.target_percentage {
            // Reduce aggression if ahead of schedule
            let new_aggression = current.max(self.config.target_percentage);
            self.current_aggression.store((new_aggression * 10000.0) as u64, Ordering::Relaxed);
        }
        
        self.last_adjustment_time = Instant::now();
    }

    /// Pause execution temporarily
    #[inline]
    pub fn pause(&mut self) {
        if self.state == PovState::Executing || self.state == PovState::Tracking {
            self.state = PovState::Paused;
        }
    }

    /// Resume execution
    #[inline]
    pub fn resume(&mut self) {
        if self.state == PovState::Paused {
            self.state = PovState::Executing;
            self.is_active.store(true, Ordering::Relaxed);
        }
    }

    /// Cancel execution
    #[inline]
    pub fn cancel(&mut self) {
        self.state = PovState::Cancelled;
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Get execution progress
    #[inline]
    pub fn get_progress(&self) -> f64 {
        if self.parent_order_size == 0 {
            return 0.0;
        }
        self.executed_size as f64 / self.parent_order_size as f64
    }

    /// Check if execution is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.state == PovState::Completed
    }

    /// Get remaining quantity
    #[inline]
    pub fn remaining_qty(&self) -> u64 {
        self.parent_order_size.saturating_sub(self.executed_size)
    }

    /// Reset executor for new order
    #[inline]
    pub fn reset(&mut self, new_parent_size: u64) {
        self.state = PovState::Idle;
        self.parent_order_size = new_parent_size;
        self.executed_size = 0;
        self.market_volume_tracked = 0;
        self.is_active.store(false, Ordering::Relaxed);
        self.current_aggression.store((self.config.target_percentage * 10000.0) as u64, Ordering::Relaxed);
    }
}

/// Volume stream tracker for Binance real-time trade data
pub struct VolumeStreamTracker {
    recent_volume_sum: AtomicU64,
    tick_count: AtomicU64,
    window_size_ns: u64,
    oldest_tick_time: AtomicU64,
}

impl VolumeStreamTracker {
    pub fn new(window_size_ms: u64) -> Self {
        Self {
            recent_volume_sum: AtomicU64::new(0),
            tick_count: AtomicU64::new(0),
            window_size_ns: window_size_ms * 1_000_000,
            oldest_tick_time: AtomicU64::new(0),
        }
    }

    /// Add tick to rolling window (simplified - in production use ring buffer)
    #[inline]
    pub fn add_tick(&self, tick: VolumeTick) -> u64 {
        // In production, maintain a proper ring buffer
        // This is a simplified atomic version
        self.recent_volume_sum.fetch_add(tick.volume, Ordering::Relaxed);
        self.tick_count.fetch_add(1, Ordering::Relaxed);
        
        if self.oldest_tick_time.load(Ordering::Relaxed) == 0 {
            self.oldest_tick_time.store(tick.timestamp_ns, Ordering::Relaxed);
        }
        
        self.recent_volume_sum.load(Ordering::Relaxed)
    }

    /// Get current volume rate (volume per second)
    #[inline]
    pub fn get_volume_rate(&self, current_time_ns: u64) -> f64 {
        let oldest = self.oldest_tick_time.load(Ordering::Relaxed);
        if oldest == 0 {
            return 0.0;
        }
        
        let elapsed_ns = current_time_ns.saturating_sub(oldest);
        if elapsed_ns == 0 {
            return 0.0;
        }
        
        let volume = self.recent_volume_sum.load(Ordering::Relaxed) as f64;
        let elapsed_sec = elapsed_ns as f64 / 1_000_000_000.0;
        
        volume / elapsed_sec
    }

    /// Reset tracker
    #[inline]
    pub fn reset(&self) {
        self.recent_volume_sum.store(0, Ordering::Relaxed);
        self.tick_count.store(0, Ordering::Relaxed);
        self.oldest_tick_time.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pov_execution() {
        let config = PovConfig {
            target_percentage: 0.10,
            min_order_size: 100,
            max_order_size: 10000,
            ..Default::default()
        };
        
        let mut executor = PovExecutor::new(config, 10000);
        executor.start();
        
        // Simulate market volume ticks
        let ticks = vec![
            VolumeTick { timestamp_ns: 1000, price_bps: 500000, volume: 5000, side: TradeSide::Buy },
            VolumeTick { timestamp_ns: 2000, price_bps: 500100, volume: 5000, side: TradeSide::Sell },
            VolumeTick { timestamp_ns: 3000, price_bps: 500200, volume: 5000, side: TradeSide::Buy },
        ];
        
        let mut total_executed = 0;
        for tick in ticks {
            if let Some(qty) = executor.process_tick(tick) {
                total_executed += qty;
            }
        }
        
        // Should have executed ~10% of 15000 = 1500
        assert!(total_executed >= 1400 && total_executed <= 1600);
    }

    #[test]
    fn test_aggression_adjustment() {
        let config = PovConfig::default();
        let mut executor = PovExecutor::new(config, 10000);
        executor.start();
        
        // Simulate falling behind (200 bps deficit)
        executor.adjust_aggression(200);
        
        let new_aggression = executor.current_aggression.load(Ordering::Relaxed) as f64 / 10000.0;
        assert!(new_aggression > config.target_percentage);
    }
}
