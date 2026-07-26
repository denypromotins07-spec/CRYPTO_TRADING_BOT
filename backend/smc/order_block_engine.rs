//! ZAID Personal Crypto Trading Bot - Order Block Engine Module
//! Chapter 3: Order Blocks, Fair Value Gaps (FVG), and Mitigation Zones
//! 
//! This module pinpoints institutional supply/demand order blocks.
//! Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
//! 
//! Design Patterns: Strategy Pattern, Flyweight for block storage
//! Time Complexity: O(1) for updates and queries
//! Space Complexity: O(k) where k is number of active order blocks

use std::collections::VecDeque;

/// Order block type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderBlockType {
    /// Bullish order block - last down candle before strong up move
    Bullish,
    /// Bearish order block - last up candle before strong down move
    Bearish,
    /// Breaker block - failed order block that flips role
    BreakerBullish,
    /// Breaker block - failed order block that flips role
    BreakerBearish,
}

/// Order block status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderBlockStatus {
    /// Fresh - price hasn't returned yet
    Fresh,
    /// Partially mitigated - some orders filled
    PartiallyMitigated,
    /// Fully mitigated - all orders filled
    FullyMitigated,
    /// Invalidated - price closed through with high volume
    Invalidated,
}

/// Order block representation
#[derive(Debug, Clone)]
pub struct OrderBlock {
    pub id: u64,
    pub block_type: OrderBlockType,
    pub status: OrderBlockStatus,
    /// High of the order block candle
    pub high: f64,
    /// Low of the order block candle
    pub low: f64,
    /// Open of the order block candle
    pub open: f64,
    /// Close of the order block candle
    pub close: f64,
    /// Timestamp when block was formed
    pub timestamp: u64,
    /// Number of times price has touched this block
    pub touch_count: u32,
    /// Volume at formation (institutional footprint)
    pub formation_volume: f64,
    /// Strength score 0.0 to 1.0
    pub strength: f64,
    /// Validity flag
    pub valid: bool,
}

impl OrderBlock {
    /// Create new order block from candle data
    pub fn new(
        id: u64,
        block_type: OrderBlockType,
        high: f64,
        low: f64,
        open: f64,
        close: f64,
        timestamp: u64,
        volume: f64,
    ) -> Self {
        // Calculate initial strength based on candle characteristics
        let body = (close - open).abs();
        let range = high - low;
        let body_ratio = if range > 0.0 { body / range } else { 0.0 };
        
        Self {
            id,
            block_type,
            status: OrderBlockStatus::Fresh,
            high,
            low,
            open,
            close,
            timestamp,
            touch_count: 0,
            formation_volume: volume,
            strength: Self::calculate_initial_strength(body_ratio, volume),
            valid: true,
        }
    }
    
    /// Calculate initial strength based on candle properties
    #[inline]
    fn calculate_initial_strength(body_ratio: f64, volume: f64) -> f64 {
        let mut strength = 0.5;
        
        // Strong body ratio adds strength
        strength += body_ratio * 0.3;
        
        // Higher volume adds strength (simplified)
        if volume > 1000.0 {
            strength += 0.2;
        }
        
        strength.min(1.0)
    }
    
    /// Get the key mitigation zone (50% level of the block)
    #[inline]
    pub fn conduction_zone(&self) -> f64 {
        (self.high + self.low) / 2.0
    }
    
    /// Get the open price (often acts as limit)
    #[inline]
    pub fn open_limit(&self) -> f64 {
        self.open
    }
    
    /// Check if price is within the order block zone
    #[inline]
    pub fn contains(&self, price: f64) -> bool {
        price >= self.low && price <= self.high
    }
    
    /// Update mitigation status based on price action
    pub fn update_mitigation(&mut self, current_price: f64, current_volume: f64) {
        if !self.valid {
            return;
        }
        
        if self.contains(current_price) {
            self.touch_count += 1;
            
            // Update status based on touches
            self.status = match self.touch_count {
                0 => OrderBlockStatus::Fresh,
                1..=3 => OrderBlockStatus::PartiallyMitigated,
                _ => OrderBlockStatus::FullyMitigated,
            };
        }
    }
    
    /// Invalidate order block if price closes through with high volume
    pub fn check_invalidation(&mut self, candle_close: f64, candle_volume: f64, avg_volume: f64) {
        if !self.valid {
            return;
        }
        
        match self.block_type {
            OrderBlockType::Bullish | OrderBlockType::BreakerBullish => {
                // Bullish OB invalidated if price closes below low with high volume
                if candle_close < self.low && candle_volume > avg_volume * 1.5 {
                    self.valid = false;
                    self.status = OrderBlockStatus::Invalidated;
                }
            }
            OrderBlockType::Bearish | OrderBlockType::BreakerBearish => {
                // Bearish OB invalidated if price closes above high with high volume
                if candle_close > self.high && candle_volume > avg_volume * 1.5 {
                    self.valid = false;
                    self.status = OrderBlockStatus::Invalidated;
                }
            }
        }
    }
    
    /// Convert to breaker block after failure
    pub fn convert_to_breaker(&mut self) {
        self.block_type = match self.block_type {
            OrderBlockType::Bullish => OrderBlockType::BreakerBearish,
            OrderBlockType::Bearish => OrderBlockType::BreakerBullish,
            other => other,
        };
        self.strength *= 0.7; // Breakers typically weaker than original
    }
}

/// Configuration for order block detection
pub struct OrderBlockConfig {
    /// Minimum body size ratio for significant candle
    pub min_body_ratio: f64,
    /// Minimum displacement (move after OB) in ATR
    pub min_displacement_atr: f64,
    /// Maximum order blocks to track per type
    pub max_blocks_per_type: usize,
}

impl Default for OrderBlockConfig {
    fn default() -> Self {
        Self {
            min_body_ratio: 0.5,
            min_displacement_atr: 2.0,
            max_blocks_per_type: 20,
        }
    }
}

/// Candle data for analysis
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    #[inline]
    pub fn new(timestamp: u64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self { timestamp, open, high, low, close, volume }
    }
    
    #[inline]
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
    
    #[inline]
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
    
    #[inline]
    pub fn body_ratio(&self) -> f64 {
        let range = self.range();
        if range > 0.0 { self.body() / range } else { 0.0 }
    }
    
    #[inline]
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
    
    #[inline]
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

/// Core order block engine
pub struct OrderBlockEngine {
    config: OrderBlockConfig,
    /// Detected order blocks
    blocks: Vec<OrderBlock>,
    /// Next ID for new blocks
    next_id: u64,
    /// Recent candles for pattern detection
    recent_candles: VecDeque<Candle>,
    /// ATR value for displacement calculation
    atr_value: f64,
    /// Average volume
    avg_volume: f64,
    /// Pending block awaiting displacement confirmation
    pending_block: Option<(Candle, OrderBlockType)>,
}

impl OrderBlockEngine {
    /// Create new order block engine with default configuration
    pub fn new() -> Self {
        Self::with_config(OrderBlockConfig::default())
    }
    
    /// Create new order block engine with custom configuration
    pub fn with_config(config: OrderBlockConfig) -> Self {
        Self {
            config,
            blocks: Vec::new(),
            next_id: 1,
            recent_candles: VecDeque::with_capacity(50),
            atr_value: 0.0,
            avg_volume: 0.0,
            pending_block: None,
        }
    }
    
    /// Update engine with new candle data
    /// Returns vector of newly created order blocks
    pub fn update(&mut self, candle: Candle) -> Vec<&OrderBlock> {
        let mut new_blocks = Vec::new();
        
        // Add to history
        if self.recent_candles.len() >= 50 {
            self.recent_candles.pop_front();
        }
        self.recent_candles.push_back(candle);
        
        // Update ATR
        self.update_atr(&candle);
        
        // Update average volume
        self.update_avg_volume(candle.volume);
        
        // Check for order block formation
        if let Some(block) = self.detect_order_block(&candle) {
            new_blocks.push(block);
        }
        
        // Check pending block for displacement confirmation
        if let Some((pending_candle, block_type)) = self.pending_block.take() {
            if self.check_displacement(&pending_candle, &candle, block_type) {
                let block = OrderBlock::new(
                    self.next_id,
                    block_type,
                    pending_candle.high,
                    pending_candle.low,
                    pending_candle.open,
                    pending_candle.close,
                    pending_candle.timestamp,
                    pending_candle.volume,
                );
                
                self.add_block(block);
                new_blocks.push(self.blocks.last().unwrap());
                self.next_id += 1;
            }
        }
        
        // Update existing blocks
        for block in &mut self.blocks {
            block.update_mitigation(candle.close, candle.volume);
            block.check_invalidation(candle.close, candle.volume, self.avg_volume);
        }
        
        // Remove invalid blocks if too many
        self.cleanup_blocks();
        
        new_blocks
    }
    
    /// Update ATR using Wilder's smoothing
    fn update_atr(&mut self, candle: &Candle) {
        let tr = if let Some(prev) = self.recent_candles.back() {
            let tr1 = candle.high - candle.low;
            let tr2 = (candle.high - prev.close).abs();
            let tr3 = (candle.low - prev.close).abs();
            tr1.max(tr2).max(tr3)
        } else {
            candle.high - candle.low
        };
        
        if self.atr_value == 0.0 {
            self.atr_value = tr;
        } else {
            self.atr_value = (self.atr_value * 13.0 + tr) / 14.0;
        }
    }
    
    /// Update average volume
    fn update_avg_volume(&mut self, volume: f64) {
        if self.avg_volume == 0.0 {
            self.avg_volume = volume;
        } else {
            self.avg_volume = (self.avg_volume * 9.0 + volume) / 10.0;
        }
    }
    
    /// Detect potential order block formation
    fn detect_order_block(&mut self, candle: &Candle) -> Option<OrderBlock> {
        if self.recent_candles.len() < 2 {
            return None;
        }
        
        let candles: Vec<&Candle> = self.recent_candles.iter().collect();
        let prev = candles[candles.len() - 2];
        
        // Check for bullish OB: down candle followed by strong up move
        if prev.is_bearish() && candle.is_bullish() && candle.body_ratio() >= self.config.min_body_ratio {
            self.pending_block = Some((*prev, OrderBlockType::Bullish));
        }
        
        // Check for bearish OB: up candle followed by strong down move
        if prev.is_bullish() && candle.is_bearish() && candle.body_ratio() >= self.config.min_body_ratio {
            self.pending_block = Some((*prev, OrderBlockType::Bearish));
        }
        
        None
    }
    
    /// Check if displacement occurred after potential OB
    fn check_displacement(&self, ob_candle: &Candle, current: &Candle, block_type: OrderBlockType) -> bool {
        if self.atr_value <= 0.0 {
            return false;
        }
        
        match block_type {
            OrderBlockType::Bullish => {
                let displacement = current.high - ob_candle.high;
                displacement >= self.atr_value * self.config.min_displacement_atr
            }
            OrderBlockType::Bearish => {
                let displacement = ob_candle.low - current.low;
                displacement >= self.atr_value * self.config.min_displacement_atr
            }
            _ => false,
        }
    }
    
    /// Add block to internal storage with memory management
    fn add_block(&mut self, block: OrderBlock) {
        // Count blocks by type
        let type_count = self.blocks.iter()
            .filter(|b| b.block_type == block.block_type && b.valid)
            .count();
        
        if type_count >= self.config.max_blocks_per_type {
            // Remove oldest block of same type
            if let Some(pos) = self.blocks.iter().position(|b| {
                b.block_type == block.block_type && !b.valid
            }) {
                self.blocks.remove(pos);
            } else if let Some(pos) = self.blocks.iter().position(|b| {
                b.block_type == block.block_type && b.status == OrderBlockStatus::FullyMitigated
            }) {
                self.blocks.remove(pos);
            }
        }
        
        self.blocks.push(block);
    }
    
    /// Cleanup invalid and fully mitigated blocks
    fn cleanup_blocks(&mut self) {
        self.blocks.retain(|b| {
            b.valid && b.status != OrderBlockStatus::FullyMitigated
        });
    }
    
    /// Get all valid bullish order blocks
    pub fn get_bullish_blocks(&self) -> impl Iterator<Item = &OrderBlock> {
        self.blocks.iter().filter(|b| {
            b.valid && matches!(b.block_type, OrderBlockType::Bullish | OrderBlockType::BreakerBullish)
        })
    }
    
    /// Get all valid bearish order blocks
    pub fn get_bearish_blocks(&self) -> impl Iterator<Item = &OrderBlock> {
        self.blocks.iter().filter(|b| {
            b.valid && matches!(b.block_type, OrderBlockType::Bearish | OrderBlockType::BreakerBearish)
        })
    }
    
    /// Get nearest bullish order block below current price
    pub fn get_nearest_bullish_block(&self, current_price: f64) -> Option<&OrderBlock> {
        self.get_bullish_blocks()
            .filter(|b| b.high < current_price)
            .max_by(|a, b| a.high.partial_cmp(&b.high).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get nearest bearish order block above current price
    pub fn get_nearest_bearish_block(&self, current_price: f64) -> Option<&OrderBlock> {
        self.get_bearish_blocks()
            .filter(|b| b.low > current_price)
            .min_by(|a, b| a.low.partial_cmp(&b.low).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get current ATR
    #[inline]
    pub fn get_atr(&self) -> f64 {
        self.atr_value
    }
    
    /// Clear all state
    #[inline]
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.recent_candles.clear();
        self.pending_block = None;
        self.atr_value = 0.0;
        self.avg_volume = 0.0;
    }
}

impl Default for OrderBlockEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_block_creation() {
        let mut engine = OrderBlockEngine::new();
        
        // Create a bearish candle followed by strong bullish candle
        let bearish = Candle::new(1, 105.0, 106.0, 103.0, 103.5, 1000.0);
        let bullish = Candle::new(2, 103.5, 110.0, 103.0, 109.5, 2000.0);
        
        engine.update(bearish);
        let blocks = engine.update(bullish);
        
        // Should have pending block awaiting displacement
        assert!(engine.pending_block.is_some());
    }

    #[test]
    fn test_order_block_contains() {
        let block = OrderBlock::new(
            1,
            OrderBlockType::Bullish,
            105.0,
            100.0,
            104.0,
            103.0,
            1000,
            1000.0,
        );
        
        assert!(block.contains(102.0));
        assert!(!block.contains(106.0));
        assert!(!block.contains(99.0));
    }
}
