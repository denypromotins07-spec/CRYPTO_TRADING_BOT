//! ZAID Personal Crypto Trading Bot - Mitigation Tracker Module
//! Chapter 3: Order Blocks, Fair Value Gaps (FVG), and Mitigation Zones
//! 
//! This module tracks when price returns to mitigate unmitigated blocks.
//! Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
//! 
//! Design Patterns: Observer Pattern, State Machine for mitigation states
//! Time Complexity: O(1) for updates and queries
//! Space Complexity: O(k) where k is number of tracked mitigation levels

use std::collections::{HashMap, VecDeque};

/// Mitigation event type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationType {
    /// Order block mitigation
    OrderBlock,
    /// Fair Value Gap fill
    FVGFill,
    /// Breaker block mitigation
    BreakerBlock,
    /// Liquidity pool sweep mitigation
    LiquidityMitigation,
}

/// Mitigation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationStatus {
    /// Waiting for price to return
    Pending,
    /// Price currently in mitigation zone
    InProgress,
    /// Fully mitigated
    Complete,
    /// Cancelled/invalidated
    Cancelled,
}

/// Mitigation level representation
#[derive(Debug, Clone)]
pub struct MitigationLevel {
    pub id: u64,
    pub mitigation_type: MitigationType,
    pub status: MitigationStatus,
    /// Target price level for mitigation
    pub target_price: f64,
    /// Upper bound of mitigation zone
    pub zone_upper: f64,
    /// Lower bound of mitigation zone
    pub zone_lower: f64,
    /// Timestamp when level was created
    pub created_timestamp: u64,
    /// Timestamp when mitigation started (if applicable)
    pub start_timestamp: Option<u64>,
    /// Timestamp when fully mitigated
    pub complete_timestamp: Option<u64>,
    /// Fill percentage (0.0 to 1.0)
    pub fill_percentage: f64,
    /// Priority score for ordering
    pub priority: f64,
    /// Associated volume at creation
    pub associated_volume: f64,
}

impl MitigationLevel {
    /// Create new mitigation level
    pub fn new(
        id: u64,
        mitigation_type: MitigationType,
        target_price: f64,
        zone_width_pct: f64,
        timestamp: u64,
        volume: f64,
        priority: f64,
    ) -> Self {
        let zone_half_width = target_price * zone_width_pct / 100.0 / 2.0;
        
        Self {
            id,
            mitigation_type,
            status: MitigationStatus::Pending,
            target_price,
            zone_upper: target_price + zone_half_width,
            zone_lower: target_price - zone_half_width,
            created_timestamp: timestamp,
            start_timestamp: None,
            complete_timestamp: None,
            fill_percentage: 0.0,
            priority,
            associated_volume: volume,
        }
    }
    
    /// Check if price is within the mitigation zone - O(1)
    #[inline]
    pub fn contains(&self, price: f64) -> bool {
        price >= self.zone_lower && price <= self.zone_upper
    }
    
    /// Update mitigation progress based on price action - O(1)
    pub fn update_progress(&mut self, high: f64, low: f64, close: f64, current_timestamp: u64) {
        if self.status == MitigationStatus::Complete || self.status == MitigationStatus::Cancelled {
            return;
        }
        
        // Check if price entered the zone
        if self.contains(low) || self.contains(high) {
            if self.status == MitigationStatus::Pending {
                self.status = MitigationStatus::InProgress;
                self.start_timestamp = Some(current_timestamp);
            }
            
            // Calculate fill percentage based on how much of the zone was touched
            let zone_size = self.zone_upper - self.zone_lower;
            if zone_size > 0.0 {
                let touch_low = low.min(self.zone_upper).max(self.zone_lower);
                let touch_high = high.min(self.zone_upper).max(self.zone_lower);
                
                let touched_range = touch_high - touch_low;
                let new_fill = touched_range / zone_size;
                
                self.fill_percentage = self.fill_percentage.max(new_fill);
                
                // Mark complete if mostly filled
                if self.fill_percentage >= 0.95 {
                    self.status = MitigationStatus::Complete;
                    self.complete_timestamp = Some(current_timestamp);
                }
            }
        }
    }
    
    /// Cancel this mitigation level
    #[inline]
    pub fn cancel(&mut self) {
        self.status = MitigationStatus::Cancelled;
    }
}

/// Configuration for mitigation tracking
pub struct MitigationConfig {
    /// Default zone width as percentage of price
    pub default_zone_width_pct: f64,
    /// Maximum mitigation levels to track
    pub max_levels: usize,
    /// Minimum priority threshold for tracking
    pub min_priority_threshold: f64,
    /// Expiration time for pending mitigations (in timestamps)
    pub expiration_time: u64,
}

impl Default for MitigationConfig {
    fn default() -> Self {
        Self {
            default_zone_width_pct: 0.5,
            max_levels: 100,
            min_priority_threshold: 0.3,
            expiration_time: 500,
        }
    }
}

/// Core mitigation tracker
pub struct MitigationTracker {
    config: MitigationConfig,
    /// Active mitigation levels
    levels: HashMap<u64, MitigationLevel>,
    /// Next ID for new levels
    next_id: u64,
    /// Priority queue of levels by urgency (stored separately for efficient access)
    priority_queue: Vec<u64>,
    /// Current timestamp for expiration tracking
    current_timestamp: u64,
    /// Statistics
    total_mitigations: u64,
    completed_mitigations: u64,
}

impl MitigationTracker {
    /// Create new mitigation tracker with default configuration
    pub fn new() -> Self {
        Self::with_config(MitigationConfig::default())
    }
    
    /// Create new mitigation tracker with custom configuration
    pub fn with_config(config: MitigationConfig) -> Self {
        Self {
            config,
            levels: HashMap::new(),
            next_id: 1,
            priority_queue: Vec::new(),
            current_timestamp: 0,
            total_mitigations: 0,
            completed_mitigations: 0,
        }
    }
    
    /// Add new mitigation level to track
    /// Returns the ID of the new level
    pub fn add_mitigation_level(
        &mut self,
        mitigation_type: MitigationType,
        target_price: f64,
        timestamp: u64,
        volume: f64,
        priority: f64,
    ) -> Option<u64> {
        // Check priority threshold
        if priority < self.config.min_priority_threshold {
            return None;
        }
        
        // Enforce memory bounds
        if self.levels.len() >= self.config.max_levels {
            self.remove_lowest_priority();
        }
        
        let level = MitigationLevel::new(
            self.next_id,
            mitigation_type,
            target_price,
            self.config.default_zone_width_pct,
            timestamp,
            volume,
            priority,
        );
        
        let id = level.id;
        self.levels.insert(id, level);
        self.priority_queue.push(id);
        self.next_id += 1;
        self.total_mitigations += 1;
        
        // Sort priority queue by priority (descending)
        self.priority_queue.sort_by(|a, b| {
            let priority_a = self.levels.get(a).map(|l| l.priority).unwrap_or(0.0);
            let priority_b = self.levels.get(b).map(|l| l.priority).unwrap_or(0.0);
            priority_b.partial_cmp(&priority_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Some(id)
    }
    
    /// Remove lowest priority level to make room
    fn remove_lowest_priority(&mut self) {
        if let Some(lowest_id) = self.priority_queue.last().copied() {
            self.levels.remove(&lowest_id);
            self.priority_queue.pop();
        }
    }
    
    /// Update all mitigation levels with new price data
    /// Returns vector of IDs that were completed in this update
    pub fn update(&mut self, high: f64, low: f64, close: f64, timestamp: u64) -> Vec<u64> {
        self.current_timestamp = timestamp;
        let mut completed = Vec::new();
        
        for (&id, level) in &mut self.levels {
            if level.status != MitigationStatus::Complete && level.status != MitigationStatus::Cancelled {
                let prev_status = level.status;
                level.update_progress(high, low, close, timestamp);
                
                if level.status == MitigationStatus::Complete && prev_status != MitigationStatus::Complete {
                    completed.push(id);
                    self.completed_mitigations += 1;
                }
            }
        }
        
        // Expire old pending levels
        self.expire_old_levels(timestamp);
        
        completed
    }
    
    /// Expire old pending mitigation levels
    fn expire_old_levels(&mut self, current_timestamp: u64) {
        let mut to_remove = Vec::new();
        
        for (&id, level) in &self.levels {
            if level.status == MitigationStatus::Pending {
                let age = current_timestamp - level.created_timestamp;
                if age > self.config.expiration_time {
                    to_remove.push(id);
                }
            }
        }
        
        for id in to_remove {
            if let Some(level) = self.levels.get_mut(&id) {
                level.cancel();
            }
            self.priority_queue.retain(|&qid| qid != id);
        }
    }
    
    /// Get all active (pending or in-progress) mitigation levels
    pub fn get_active_levels(&self) -> impl Iterator<Item = &MitigationLevel> {
        self.levels.values().filter(|l| {
            l.status == MitigationStatus::Pending || l.status == MitigationStatus::InProgress
        })
    }
    
    /// Get nearest mitigation level above current price
    pub fn get_nearest_above(&self, current_price: f64) -> Option<&MitigationLevel> {
        self.get_active_levels()
            .filter(|l| l.zone_lower > current_price)
            .min_by(|a, b| a.zone_lower.partial_cmp(&b.zone_lower).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get nearest mitigation level below current price
    pub fn get_nearest_below(&self, current_price: f64) -> Option<&MitigationLevel> {
        self.get_active_levels()
            .filter(|l| l.zone_upper < current_price)
            .max_by(|a, b| a.zone_upper.partial_cmp(&b.zone_upper).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get highest priority active mitigation level
    pub fn get_highest_priority(&self) -> Option<&MitigationLevel> {
        self.priority_queue
            .iter()
            .filter_map(|&id| self.levels.get(&id))
            .find(|l| l.status == MitigationStatus::Pending || l.status == MitigationStatus::InProgress)
    }
    
    /// Get mitigation level by ID
    pub fn get_level(&self, id: u64) -> Option<&MitigationLevel> {
        self.levels.get(&id)
    }
    
    /// Get completion statistics
    pub fn get_stats(&self) -> (u64, u64, f64) {
        let success_rate = if self.total_mitigations > 0 {
            self.completed_mitigations as f64 / self.total_mitigations as f64
        } else {
            0.0
        };
        (self.total_mitigations, self.completed_mitigations, success_rate)
    }
    
    /// Clear all mitigation levels
    pub fn clear(&mut self) {
        self.levels.clear();
        self.priority_queue.clear();
        self.current_timestamp = 0;
    }
}

impl Default for MitigationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mitigation_level_creation() {
        let level = MitigationLevel::new(
            1,
            MitigationType::OrderBlock,
            100.0,
            1.0,  // 1% zone width
            1000,
            1000.0,
            0.8,
        );
        
        assert_eq!(level.target_price, 100.0);
        assert_eq!(level.status, MitigationStatus::Pending);
        assert!(level.contains(99.5));
        assert!(level.contains(100.5));
        assert!(!level.contains(99.0));
        assert!(!level.contains(101.0));
    }

    #[test]
    fn test_mitigation_tracker_operations() {
        let mut tracker = MitigationTracker::new();
        
        let id = tracker.add_mitigation_level(
            MitigationType::FVGFill,
            100.0,
            1000,
            500.0,
            0.7,
        );
        
        assert!(id.is_some());
        assert_eq!(tracker.levels.len(), 1);
        
        // Simulate price entering the zone
        let completed = tracker.update(100.5, 99.5, 100.0, 1001);
        
        // Should be in progress but not complete yet
        let level = tracker.get_level(id.unwrap()).unwrap();
        assert_eq!(level.status, MitigationStatus::InProgress);
        assert!(completed.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        let mut tracker = MitigationTracker::new();
        
        tracker.add_mitigation_level(MitigationType::OrderBlock, 100.0, 1000, 100.0, 0.3);
        tracker.add_mitigation_level(MitigationType::OrderBlock, 105.0, 1001, 100.0, 0.9);
        tracker.add_mitigation_level(MitigationType::OrderBlock, 95.0, 1002, 100.0, 0.5);
        
        // Highest priority should be first
        let highest = tracker.get_highest_priority();
        assert!(highest.is_some());
        assert_eq!(highest.unwrap().target_price, 105.0);
    }
}
