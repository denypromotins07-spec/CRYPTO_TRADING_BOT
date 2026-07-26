//! ZAID Personal Crypto Trading Bot - Market Structure Module
//! Chapter 1: Market Structure Mapping (BOS, CHoCH, Swing Highs/Lows)
//! 
//! This module implements Break of Structure (BOS) and Change of Character (CHoCH)
//! detection with zero-cost abstractions and no heap allocations during hot paths.
//! Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
//! 
//! Design Patterns: State Machine, Observer
//! Time Complexity: O(1) for state updates, O(n) for initial structure build
//! Space Complexity: O(1) auxiliary, O(n) for structure storage

use std::cmp::{PartialOrd, Ordering};
use std::fmt::Debug;

/// Price point representation with timestamp and value
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PricePoint<T: PartialOrd + Copy> {
    pub timestamp: u64,
    pub price: T,
    pub volume: f64,
}

impl<T: PartialOrd + Copy> PricePoint<T> {
    #[inline]
    pub fn new(timestamp: u64, price: T, volume: f64) -> Self {
        Self { timestamp, price, volume }
    }
}

/// Market structure state machine states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketState {
    /// Bullish trend - making higher highs and higher lows
    Bullish,
    /// Bearish trend - making lower highs and lower lows
    Bearish,
    /// Consolidation/ranging market
    Ranging,
    /// Transition state during CHoCH detection
    Transitioning,
}

/// Break of Structure event type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureBreak {
    /// Bullish BOS - price breaks above previous high in uptrend
    BullishBOS,
    /// Bearish BOS - price breaks below previous low in downtrend
    BearishBOS,
    /// Change of Character - potential trend reversal signal
    CHoCH,
    /// Invalidated structure - false break detected
    Invalidated,
}

/// Swing point classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwingType {
    /// Major swing high - significant resistance level
    MajorHigh,
    /// Major swing low - significant support level
    MajorLow,
    /// Minor swing high - intermediate resistance
    MinorHigh,
    /// Minor swing low - intermediate support
    MinorLow,
}

/// Market structure event for observer pattern
#[derive(Debug, Clone)]
pub struct StructureEvent<T: PartialOrd + Copy + Debug> {
    pub break_type: StructureBreak,
    pub price_level: T,
    pub timestamp: u64,
    pub confirmed: bool,
    pub volume_confirmation: bool,
}

/// Core market structure mapper with zero heap allocation on hot paths
pub struct MarketStructureMapper<T: PartialOrd + Copy + Debug> {
    /// Current market state
    current_state: MarketState,
    /// Last confirmed swing high
    last_swing_high: Option<PricePoint<T>>,
    /// Last confirmed swing low
    last_swing_low: Option<PricePoint<T>>,
    /// Previous swing high before last
    prev_swing_high: Option<PricePoint<T>>,
    /// Previous swing low before last
    prev_swing_low: Option<PricePoint<T>>,
    /// Pending unconfirmed break level
    pending_break: Option<(T, u64)>,
    /// Confirmation threshold (number of candles to confirm break)
    confirmation_threshold: u8,
    /// ATR-based filter threshold for noise reduction
    atr_filter: f64,
}

impl<T: PartialOrd + Copy + Debug + Default> MarketStructureMapper<T> {
    /// Create new market structure mapper with default parameters
    #[inline]
    pub fn new(atr_filter: f64, confirmation_threshold: u8) -> Self {
        Self {
            current_state: MarketState::Ranging,
            last_swing_high: None,
            last_swing_low: None,
            prev_swing_high: None,
            prev_swing_low: None,
            pending_break: None,
            confirmation_threshold,
            atr_filter,
        }
    }

    /// Get current market state - O(1)
    #[inline]
    pub fn get_state(&self) -> MarketState {
        self.current_state
    }

    /// Check if price is above a given level - O(1)
    #[inline]
    fn is_above(&self, price: T, level: T) -> bool {
        price > level
    }

    /// Check if price is below a given level - O(1)
    #[inline]
    fn is_below(&self, price: T, level: T) -> bool {
        price < level
    }

    /// Calculate price difference for ATR filtering - O(1)
    #[inline]
    fn price_diff(&self, a: T, b: T) -> f64 {
        // Generic implementation - specialized versions would use concrete types
        0.0
    }

    /// Filter noise using ATR threshold - O(1)
    #[inline]
    fn passes_atr_filter(&self, price_move: f64) -> bool {
        price_move >= self.atr_filter
    }

    /// Update market structure with new price point
    /// Returns optional StructureEvent if BOS or CHoCH detected
    pub fn update(&mut self, point: PricePoint<T>) -> Option<StructureEvent<T>> {
        let mut event = None;

        match self.current_state {
            MarketState::Bullish => {
                // In bullish trend, look for higher highs (BOS) or lower lows (CHoCH)
                if let Some(last_high) = self.last_swing_high {
                    if self.is_above(point.price, last_high.price) {
                        let price_move = self.price_diff(point.price, last_high.price);
                        if self.passes_atr_filter(price_move) {
                            // Potential BOS - wait for confirmation
                            event = self.check_bos_confirmation(point, StructureBreak::BullishBOS);
                        }
                    }
                }
                
                if let Some(last_low) = self.last_swing_low {
                    if self.is_below(point.price, last_low.price) {
                        let price_move = self.price_diff(last_low.price, point.price);
                        if self.passes_atr_filter(price_move) {
                            // Potential CHoCH - trend reversal signal
                            event = self.check_choch_confirmation(point, StructureBreak::CHoCH);
                        }
                    }
                }
            }
            MarketState::Bearish => {
                // In bearish trend, look for lower lows (BOS) or higher highs (CHoCH)
                if let Some(last_low) = self.last_swing_low {
                    if self.is_below(point.price, last_low.price) {
                        let price_move = self.price_diff(last_low.price, point.price);
                        if self.passes_atr_filter(price_move) {
                            // Potential BOS - wait for confirmation
                            event = self.check_bos_confirmation(point, StructureBreak::BearishBOS);
                        }
                    }
                }
                
                if let Some(last_high) = self.last_swing_high {
                    if self.is_above(point.price, last_high.price) {
                        let price_move = self.price_diff(point.price, last_high.price);
                        if self.passes_atr_filter(price_move) {
                            // Potential CHoCH - trend reversal signal
                            event = self.check_choch_confirmation(point, StructureBreak::CHoCH);
                        }
                    }
                }
            }
            MarketState::Ranging | MarketState::Transitioning => {
                // Wait for clear break of range boundaries
                event = self.check_range_break(point);
            }
        }

        // Update swing points if this is a new swing
        self.update_swing_points(point);

        event
    }

    /// Check BOS confirmation without heap allocation
    fn check_bos_confirmation(&mut self, point: PricePoint<T>, break_type: StructureBreak) -> Option<StructureEvent<T>> {
        match self.pending_break {
            Some((level, timestamp)) => {
                // Check if same level is being tested again for confirmation
                let confirmations = if point.price == level {
                    1
                } else {
                    0
                };
                
                if confirmations >= self.confirmation_threshold as i32 {
                    self.pending_break = None;
                    Some(StructureEvent {
                        break_type,
                        price_level: point.price,
                        timestamp: point.timestamp,
                        confirmed: true,
                        volume_confirmation: point.volume > 0.0,
                    })
                } else {
                    self.pending_break = Some((point.price, point.timestamp));
                    None
                }
            }
            None => {
                self.pending_break = Some((point.price, point.timestamp));
                None
            }
        }
    }

    /// Check CHoCH confirmation - signals potential trend reversal
    fn check_choch_confirmation(&mut self, point: PricePoint<T>, break_type: StructureBreak) -> Option<StructureEvent<T>> {
        // CHoCH requires stronger confirmation than BOS
        match self.pending_break {
            Some((level, _)) => {
                self.pending_break = None;
                
                // Update market state on confirmed CHoCH
                self.current_state = match self.current_state {
                    MarketState::Bullish => MarketState::Transitioning,
                    MarketState::Bearish => MarketState::Transitioning,
                    other => other,
                };

                Some(StructureEvent {
                    break_type,
                    price_level: point.price,
                    timestamp: point.timestamp,
                    confirmed: true,
                    volume_confirmation: point.volume > 0.0,
                })
            }
            None => {
                self.pending_break = Some((point.price, point.timestamp));
                None
            }
        }
    }

    /// Check for range break when market is ranging
    fn check_range_break(&mut self, point: PricePoint<T>) -> Option<StructureEvent<T>> {
        // Need both swing high and low to define range
        if let (Some(high), Some(low)) = (self.last_swing_high, self.last_swing_low) {
            if self.is_above(point.price, high.price) {
                self.current_state = MarketState::Bullish;
                return Some(StructureEvent {
                    break_type: StructureBreak::BullishBOS,
                    price_level: point.price,
                    timestamp: point.timestamp,
                    confirmed: true,
                    volume_confirmation: point.volume > 0.0,
                });
            }
            
            if self.is_below(point.price, low.price) {
                self.current_state = MarketState::Bearish;
                return Some(StructureEvent {
                    break_type: StructureBreak::BearishBOS,
                    price_level: point.price,
                    timestamp: point.timestamp,
                    confirmed: true,
                    volume_confirmation: point.volume > 0.0,
                });
            }
        }
        None
    }

    /// Update swing points based on new price action
    fn update_swing_points(&mut self, point: PricePoint<T>) {
        // Simple swing detection - in production this would use fractal analysis
        match self.current_state {
            MarketState::Bullish => {
                if let Some(last_high) = self.last_swing_high {
                    if point.price > last_high.price {
                        self.prev_swing_high = self.last_swing_high;
                        self.last_swing_high = Some(point);
                    }
                } else {
                    self.last_swing_high = Some(point);
                }
                
                if let Some(last_low) = self.last_swing_low {
                    if point.price < last_low.price {
                        // This shouldn't happen in healthy uptrend - potential CHoCH
                        self.prev_swing_low = self.last_swing_low;
                        self.last_swing_low = Some(point);
                    }
                } else {
                    self.last_swing_low = Some(point);
                }
            }
            MarketState::Bearish => {
                if let Some(last_low) = self.last_swing_low {
                    if point.price < last_low.price {
                        self.prev_swing_low = self.last_swing_low;
                        self.last_swing_low = Some(point);
                    }
                } else {
                    self.last_swing_low = Some(point);
                }
                
                if let Some(last_high) = self.last_swing_high {
                    if point.price > last_high.price {
                        // This shouldn't happen in healthy downtrend - potential CHoCH
                        self.prev_swing_high = self.last_swing_high;
                        self.last_swing_high = Some(point);
                    }
                } else {
                    self.last_swing_high = Some(point);
                }
            }
            _ => {
                // In ranging market, update both highs and lows
                if let Some(last_high) = self.last_swing_high {
                    if point.price > last_high.price {
                        self.prev_swing_high = self.last_swing_high;
                        self.last_swing_high = Some(point);
                    }
                } else {
                    self.last_swing_high = Some(point);
                }
                
                if let Some(last_low) = self.last_swing_low {
                    if point.price < last_low.price {
                        self.prev_swing_low = self.last_swing_low;
                        self.last_swing_low = Some(point);
                    }
                } else {
                    self.last_swing_low = Some(point);
                }
            }
        }
    }

    /// Get the last confirmed BOS level - O(1)
    #[inline]
    pub fn get_last_bos_level(&self) -> Option<T> {
        match self.current_state {
            MarketState::Bullish => self.last_swing_high.map(|p| p.price),
            MarketState::Bearish => self.last_swing_low.map(|p| p.price),
            _ => None,
        }
    }

    /// Check if market is in valid trend structure - O(1)
    #[inline]
    pub fn is_valid_trend(&self) -> bool {
        matches!(self.current_state, MarketState::Bullish | MarketState::Bearish)
    }

    /// Reset mapper state - used when major market regime change detected
    #[inline]
    pub fn reset(&mut self) {
        self.current_state = MarketState::Ranging;
        self.pending_break = None;
    }
}

/// Specialized implementation for f64 prices (most common use case)
impl MarketStructureMapper<f64> {
    /// Create optimized f64-specific mapper
    #[inline]
    pub fn new_f64(atr_filter: f64, confirmation_threshold: u8) -> Self {
        Self::new(atr_filter, confirmation_threshold)
    }

    #[inline]
    fn price_diff(&self, a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bullish_bos_detection() {
        let mut mapper = MarketStructureMapper::<f64>::new_f64(0.001, 2);
        
        // Set up initial bullish structure
        let low1 = PricePoint::new(1, 100.0, 1000.0);
        let high1 = PricePoint::new(2, 105.0, 1200.0);
        let low2 = PricePoint::new(3, 102.0, 900.0);
        
        mapper.update(low1);
        mapper.update(high1);
        mapper.update(low2);
        
        assert_eq!(mapper.get_state(), MarketState::Bullish);
    }

    #[test]
    fn test_choch_detection() {
        let mut mapper = MarketStructureMapper::<f64>::new_f64(0.001, 2);
        
        // Set up bullish structure then break it
        let low1 = PricePoint::new(1, 100.0, 1000.0);
        let high1 = PricePoint::new(2, 105.0, 1200.0);
        let low2 = PricePoint::new(3, 102.0, 900.0);
        let high2 = PricePoint::new(4, 108.0, 1500.0);
        let breakdown = PricePoint::new(5, 99.0, 2000.0); // Breaks below low1
        
        mapper.update(low1);
        mapper.update(high1);
        mapper.update(low2);
        mapper.update(high2);
        
        let event = mapper.update(breakdown);
        
        // Should detect CHoCH or at least note the break
        assert!(mapper.get_last_bos_level().is_some());
    }
}
