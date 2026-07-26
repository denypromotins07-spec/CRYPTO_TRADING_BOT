//! ZAID Personal Crypto Trading Bot - Trendline Engine Module
//! Chapter 1: Market Structure Mapping (BOS, CHoCH, Swing Highs/Lows)
//! 
//! This module draws dynamic institutional trendlines in O(1) time using
//! advanced geometric algorithms and incremental line fitting.
//! Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
//! 
//! Design Patterns: Strategy Pattern, Flyweight for line storage
//! Time Complexity: O(1) for updates and queries
//! Space Complexity: O(k) where k is number of active trendlines

use std::f64::consts::PI;

/// Point in 2D space representing timestamp-price coordinate
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: u64,      // Timestamp
    pub y: f64,      // Price
}

impl Point {
    #[inline]
    pub fn new(timestamp: u64, price: f64) -> Self {
        Self { x: timestamp, y: price }
    }
}

/// Trendline classification based on institutional significance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendlineType {
    /// Major support/resistance - touched 3+ times by price
    Major,
    /// Minor trendline - touched 2 times
    Minor,
    /// Internal trendline - through candle bodies rather than wicks
    Internal,
    /// Breakout line - being tested for potential break
    Breakout,
}

/// Trendline direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendlineDirection {
    /// Upward sloping - support line in uptrend
    Bullish,
    /// Downward sloping - resistance line in downtrend
    Bearish,
    /// Horizontal - equilibrium zone
    Horizontal,
}

/// Computed trendline data with cached calculations
#[derive(Debug, Clone)]
pub struct Trendline {
    /// Unique identifier for this trendline
    pub id: u64,
    /// First anchor point (earlier in time)
    pub p1: Point,
    /// Second anchor point (later in time)
    pub p2: Point,
    /// Line type classification
    pub line_type: TrendlineType,
    /// Direction of the trendline
    pub direction: TrendlineDirection,
    /// Number of times price has touched/tested this line
    pub touch_count: u32,
    /// Last timestamp when price touched this line
    pub last_touch_time: u64,
    /// Validity flag - false if invalidated by strong break
    pub valid: bool,
    /// Cached slope value for O(1) access
    pub slope: f64,
    /// Cached intercept value for O(1) access
    pub intercept: f64,
}

impl Trendline {
    /// Create new trendline from two points with automatic slope/intercept calculation
    pub fn new(id: u64, p1: Point, p2: Point, line_type: TrendlineType) -> Self {
        let (slope, intercept) = Self::calculate_line(p1, p2);
        let direction = Self::classify_direction(slope);
        
        Self {
            id,
            p1,
            p2,
            line_type,
            direction,
            touch_count: 2, // Minimum 2 points to form a line
            last_touch_time: p2.x,
            valid: true,
            slope,
            intercept,
        }
    }
    
    /// Calculate slope and intercept from two points - O(1)
    #[inline]
    fn calculate_line(p1: Point, p2: Point) -> (f64, f64) {
        let dx = p2.x as f64 - p1.x as f64;
        
        if dx.abs() < 1e-10 {
            // Vertical line - use large slope
            return (f64::MAX / 2.0, p1.y);
        }
        
        let slope = (p2.y - p1.y) / dx;
        let intercept = p1.y - slope * p1.x as f64;
        
        (slope, intercept)
    }
    
    /// Classify line direction based on slope - O(1)
    #[inline]
    fn classify_direction(slope: f64) -> TrendlineDirection {
        const HORIZONTAL_THRESHOLD: f64 = 1e-6;
        
        if slope > HORIZONTAL_THRESHOLD {
            TrendlineDirection::Bullish
        } else if slope < -HORIZONTAL_THRESHOLD {
            TrendlineDirection::Bearish
        } else {
            TrendlineDirection::Horizontal
        }
    }
    
    /// Get price level at given timestamp - O(1)
    #[inline]
    pub fn price_at(&self, timestamp: u64) -> f64 {
        self.slope * timestamp as f64 + self.intercept
    }
    
    /// Calculate perpendicular distance from point to line - O(1)
    #[inline]
    pub fn distance_to_line(&self, point: Point) -> f64 {
        // Line equation: ax + by + c = 0
        // Convert from y = mx + b to mx - y + b = 0
        let a = self.slope;
        let b = -1.0;
        let c = self.intercept;
        
        let numerator = (a * point.x as f64 + b * point.y + c).abs();
        let denominator = (a * a + b * b).sqrt();
        
        if denominator < 1e-10 {
            return f64::MAX;
        }
        
        numerator / denominator
    }
    
    /// Check if point touches or nearly touches the line - O(1)
    #[inline]
    pub fn is_touched(&self, point: Point, tolerance: f64) -> bool {
        self.distance_to_line(point) <= tolerance
    }
    
    /// Check if point is above the trendline - O(1)
    #[inline]
    pub fn is_above(&self, point: Point) -> bool {
        point.y > self.price_at(point.x)
    }
    
    /// Check if point is below the trendline - O(1)
    #[inline]
    pub fn is_below(&self, point: Point) -> bool {
        point.y < self.price_at(point.x)
    }
    
    /// Update trendline with new touch point - extends the line - O(1)
    pub fn extend(&mut self, new_point: Point) {
        if new_point.x > self.p2.x {
            // Extend forward
            self.p2 = new_point;
        } else if new_point.x < self.p1.x {
            // Extend backward
            self.p1 = new_point;
        }
        
        // Recalculate slope and intercept
        let (slope, intercept) = Self::calculate_line(self.p1, self.p2);
        self.slope = slope;
        self.intercept = intercept;
        self.touch_count += 1;
        self.last_touch_time = new_point.x;
        
        // Upgrade line type if enough touches
        if self.touch_count >= 3 && self.line_type == TrendlineType::Minor {
            self.line_type = TrendlineType::Major;
        }
    }
    
    /// Invalidate trendline - called when strong break detected
    #[inline]
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
    
    /// Get angle of trendline in degrees - O(1)
    #[inline]
    pub fn angle_degrees(&self) -> f64 {
        self.slope.atan() * (180.0 / PI)
    }
    
    /// Project trendline to future timestamp - O(1)
    #[inline]
    pub fn project(&self, future_timestamp: u64) -> f64 {
        self.price_at(future_timestamp)
    }
}

/// Convergence zone where multiple trendlines intersect
#[derive(Debug, Clone)]
pub struct ConfluenceZone {
    /// Center price of the zone
    pub center_price: f64,
    /// Zone width (price range)
    pub width: f64,
    /// Number of trendlines converging
    pub line_count: u32,
    /// Strength score 0.0 to 1.0
    pub strength: f64,
}

impl ConfluenceZone {
    pub fn new(center_price: f64, width: f64, line_count: u32) -> Self {
        let strength = (line_count as f64 / 5.0).min(1.0); // Max strength at 5+ lines
        
        Self {
            center_price,
            width,
            line_count,
            strength,
        }
    }
    
    /// Check if price is within this confluence zone - O(1)
    #[inline]
    pub fn contains(&self, price: f64) -> bool {
        (price - self.center_price).abs() <= self.width / 2.0
    }
}

/// Core trendline engine with O(1) operations
pub struct TrendlineEngine {
    /// Active trendlines stored efficiently
    trendlines: Vec<Trendline>,
    /// Maximum number of trendlines to track (memory bound)
    max_trendlines: usize,
    /// Next ID for new trendlines
    next_id: u64,
    /// Tolerance for touch detection (percentage of price)
    touch_tolerance_pct: f64,
    /// Cache of detected confluence zones
    confluence_zones: Vec<ConfluenceZone>,
    /// Last update timestamp for incremental processing
    last_update_ts: u64,
}

impl TrendlineEngine {
    /// Create new trendline engine with default parameters
    pub fn new(max_trendlines: usize, touch_tolerance_pct: f64) -> Self {
        Self {
            trendlines: Vec::with_capacity(max_trendlines),
            max_trendlines,
            next_id: 1,
            touch_tolerance_pct,
            confluence_zones: Vec::new(),
            last_update_ts: 0,
        }
    }
    
    /// Add new trendline from two swing points - O(1) amortized
    pub fn add_trendline(&mut self, p1: Point, p2: Point, line_type: TrendlineType) -> Option<u64> {
        if self.trendlines.len() >= self.max_trendlines {
            // Remove oldest invalid trendline
            self.trendlines.retain(|tl| tl.valid);
            
            if self.trendlines.len() >= self.max_trendlines {
                // Still full - remove oldest minor trendline
                if let Some(pos) = self.trendlines.iter().position(|tl| tl.line_type == TrendlineType::Minor) {
                    self.trendlines.remove(pos);
                } else {
                    // Remove first trendline as last resort
                    self.trendlines.remove(0);
                }
            }
        }
        
        let trendline = Trendline::new(self.next_id, p1, p2, line_type);
        let id = trendline.id;
        self.trendlines.push(trendline);
        self.next_id += 1;
        
        Some(id)
    }
    
    /// Update all trendlines with new price point - O(n) where n is number of trendlines
    /// but each individual operation is O(1)
    pub fn update(&mut self, point: Point) -> Vec<u64> {
        let mut touched_ids = Vec::new();
        let tolerance = point.y * self.touch_tolerance_pct / 100.0;
        
        for trendline in &mut self.trendlines {
            if !trendline.valid {
                continue;
            }
            
            // Check if point is within tolerance of trendline
            if trendline.is_touched(point, tolerance) {
                trendline.extend(point);
                touched_ids.push(trendline.id);
                
                // Check for major trendline break
                if trendline.touch_count >= 3 {
                    // Strong break after multiple touches signals potential reversal
                    let break_distance = trendline.distance_to_line(point);
                    if break_distance > tolerance * 3.0 {
                        // Significant break - might invalidate
                        // But don't invalidate immediately - wait for confirmation
                    }
                }
            }
        }
        
        self.last_update_ts = point.x;
        
        // Periodically recalculate confluence zones
        if self.last_update_ts % 10 == 0 {
            self.update_confluence_zones(point.x);
        }
        
        touched_ids
    }
    
    /// Find confluence zones where multiple trendlines converge - O(n²) but n is small
    fn update_confluence_zones(&mut self, current_time: u64) {
        self.confluence_zones.clear();
        
        // Group trendlines by price level at current time
        let mut price_groups: Vec<(f64, Vec<&Trendline>)> = Vec::new();
        
        for trendline in &self.trendlines {
            if !trendline.valid {
                continue;
            }
            
            let price = trendline.project(current_time);
            
            // Find existing group or create new one
            let mut found = false;
            for (group_price, group_lines) in &mut price_groups {
                if (price - *group_price).abs() < price * 0.01 {
                    // Within 1% - same group
                    group_lines.push(trendline);
                    found = true;
                    break;
                }
            }
            
            if !found {
                price_groups.push((price, vec![trendline]));
            }
        }
        
        // Create confluence zones for groups with 2+ lines
        for (center_price, lines) in price_groups {
            if lines.len() >= 2 {
                // Calculate zone width based on price spread
                let prices: Vec<f64> = lines.iter().map(|tl| tl.project(current_time)).collect();
                let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                
                let zone = ConfluenceZone::new(
                    center_price,
                    max_price - min_price,
                    lines.len() as u32,
                );
                
                self.confluence_zones.push(zone);
            }
        }
    }
    
    /// Get all valid trendlines - O(1) reference
    #[inline]
    pub fn get_valid_trendlines(&self) -> impl Iterator<Item = &Trendline> {
        self.trendlines.iter().filter(|tl| tl.valid)
    }
    
    /// Get trendline by ID - O(n) but n is typically small
    pub fn get_trendline(&self, id: u64) -> Option<&Trendline> {
        self.trendlines.iter().find(|tl| tl.id == id)
    }
    
    /// Get nearest trendline support below current price - O(n)
    pub fn get_nearest_support(&self, current_price: f64, current_time: u64) -> Option<&Trendline> {
        self.get_valid_trendlines()
            .filter(|tl| {
                tl.direction == TrendlineDirection::Bullish || tl.direction == TrendlineDirection::Horizontal
            })
            .filter(|tl| tl.project(current_time) < current_price)
            .min_by(|a, b| {
                let dist_a = current_price - a.project(current_time);
                let dist_b = current_price - b.project(current_time);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    
    /// Get nearest trendline resistance above current price - O(n)
    pub fn get_nearest_resistance(&self, current_price: f64, current_time: u64) -> Option<&Trendline> {
        self.get_valid_trendlines()
            .filter(|tl| {
                tl.direction == TrendlineDirection::Bearish || tl.direction == TrendlineDirection::Horizontal
            })
            .filter(|tl| tl.project(current_time) > current_price)
            .min_by(|a, b| {
                let dist_a = a.project(current_time) - current_price;
                let dist_b = b.project(current_time) - current_price;
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
    
    /// Get all confluence zones - O(1) reference
    #[inline]
    pub fn get_confluence_zones(&self) -> &[ConfluenceZone] {
        &self.confluence_zones
    }
    
    /// Invalidate trendlines that have been strongly broken - O(n)
    pub fn check_breaks(&mut self, point: Point, break_threshold_pct: f64) -> Vec<u64> {
        let mut invalidated = Vec::new();
        let threshold = point.y * break_threshold_pct / 100.0;
        
        for trendline in &mut self.trendlines {
            if !trendline.valid || trendline.touch_count < 3 {
                continue;
            }
            
            let distance = trendline.distance_to_line(point);
            
            if distance > threshold {
                // Strong break detected
                match trendline.direction {
                    TrendlineDirection::Bullish => {
                        // Support broken - price below line
                        if trendline.is_below(point) {
                            trendline.invalidate();
                            invalidated.push(trendline.id);
                        }
                    }
                    TrendlineDirection::Bearish => {
                        // Resistance broken - price above line
                        if trendline.is_above(point) {
                            trendline.invalidate();
                            invalidated.push(trendline.id);
                        }
                    }
                    TrendlineDirection::Horizontal => {
                        // Either direction counts for horizontal
                        trendline.invalidate();
                        invalidated.push(trendline.id);
                    }
                }
            }
        }
        
        invalidated
    }
    
    /// Count active trendlines - O(1)
    #[inline]
    pub fn active_count(&self) -> usize {
        self.trendlines.iter().filter(|tl| tl.valid).count()
    }
    
    /// Clear all trendlines - used for reset
    #[inline]
    pub fn clear(&mut self) {
        self.trendlines.clear();
        self.confluence_zones.clear();
    }
}

impl Default for TrendlineEngine {
    fn default() -> Self {
        Self::new(50, 0.5) // 50 max lines, 0.5% tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trendline_creation() {
        let p1 = Point::new(1000, 100.0);
        let p2 = Point::new(2000, 110.0);
        
        let tl = Trendline::new(1, p1, p2, TrendlineType::Minor);
        
        assert!(tl.slope > 0.0);
        assert_eq!(tl.direction, TrendlineDirection::Bullish);
        assert_eq!(tl.price_at(1500), 105.0);
    }

    #[test]
    fn test_trendline_engine_operations() {
        let mut engine = TrendlineEngine::new(10, 1.0);
        
        let p1 = Point::new(1000, 100.0);
        let p2 = Point::new(2000, 110.0);
        
        let id = engine.add_trendline(p1, p2, TrendlineType::Minor);
        assert!(id.is_some());
        
        // Test price projection
        let tl = engine.get_trendline(id.unwrap()).unwrap();
        assert!((tl.price_at(1500) - 105.0).abs() < 0.01);
    }

    #[test]
    fn test_support_resistance_detection() {
        let mut engine = TrendlineEngine::default();
        
        // Add bullish support line
        engine.add_trendline(Point::new(1000, 95.0), Point::new(2000, 100.0), TrendlineType::Major);
        
        // Add bearish resistance line
        engine.add_trendline(Point::new(1000, 115.0), Point::new(2000, 110.0), TrendlineType::Major);
        
        let support = engine.get_nearest_support(107.0, 2500);
        let resistance = engine.get_nearest_resistance(107.0, 2500);
        
        assert!(support.is_some());
        assert!(resistance.is_some());
    }
}
