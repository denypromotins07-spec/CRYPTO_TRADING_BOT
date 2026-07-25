// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// Layering Analyzer Module
// Detects multi-level manipulation algorithms
// Zero-cost abstractions for memory-efficient operation

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// Single order in the layering analysis
#[derive(Debug, Clone)]
pub struct LayerOrder {
    pub order_id: u64,
    pub price: i64,
    pub quantity: u64,
    pub side: Side,
    pub timestamp_ns: u64,
    pub exchange_order_id: String,
    pub is_cancelled: bool,
    pub cancel_timestamp_ns: Option<u64>,
}

impl LayerOrder {
    pub fn age_ms(&self) -> f64 {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        (now_ns.saturating_sub(self.timestamp_ns)) as f64 / 1_000_000.0
    }

    pub fn lifetime_ms(&self) -> Option<f64> {
        self.cancel_timestamp_ns.map(|cancel_ts| {
            (cancel_ts.saturating_sub(self.timestamp_ns)) as f64 / 1_000_000.0
        })
    }
}

/// Detected layering pattern
#[derive(Debug, Clone)]
pub struct LayeringPattern {
    pub detection_timestamp_ns: u64,
    pub side: Side,
    pub levels: Vec<i64>,           // Price levels involved
    pub order_ids: Vec<u64>,        // Orders in the pattern
    pub total_volume: u64,
    pub avg_lifetime_ms: f64,
    pub confidence_score: f64,      // 0.0 to 1.0
    pub pattern_type: PatternType,
    pub action_recommended: LayeringAction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternType {
    ConsecutiveLevels,      // Orders at consecutive price levels
    StackedLiquidity,       // Large volume at multiple levels
    MomentumIgnition,       // Attempts to trigger price movement
    QuoteStuffing,          // Rapid order placement/cancellation
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayeringAction {
    Monitor,
    FlagForReview,
    IgnoreLiquidity,
    WidenSpread,
    HaltTrading,
}

/// Layering detection result
#[derive(Debug, Clone)]
pub struct LayeringAnalysisResult {
    pub is_layering_detected: bool,
    pub patterns: Vec<LayeringPattern>,
    pub spoofing_probability: f64,
    pub recommended_spread_adjustment_bps: f64,
}

/// Layering Analyzer using State pattern for detection states
pub struct LayeringAnalyzer {
    // Active orders being tracked
    active_orders: HashMap<u64, LayerOrder>,
    
    // Order history for pattern analysis
    order_history: VecDeque<LayerOrder>,
    max_history_size: usize,
    
    // Price level tracking
    level_orders: HashMap<(i64, Side), Vec<u64>>,
    
    // Detection thresholds
    min_levels_for_pattern: usize,
    max_level_spacing_ticks: i64,
    rapid_cancel_threshold_ms: f64,
    
    // State tracking
    current_state: AnalyzerState,
    state_history: VecDeque<(AnalyzerState, u64)>,
    
    // Statistics
    stats: LayeringStats,
    
    // Last analysis timestamp
    last_analysis_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalyzerState {
    Normal,
    Suspicious,
    ConfirmedLayering,
    Critical,
}

#[derive(Debug, Default, Clone)]
pub struct LayeringStats {
    pub total_orders_analyzed: u64,
    pub layering_patterns_detected: u64,
    pub false_positive_corrections: u64,
    pub avg_pattern_confidence: f64,
    pub total_spoofed_volume: u64,
}

impl LayeringAnalyzer {
    pub fn new(max_history: usize) -> Self {
        Self {
            active_orders: HashMap::with_capacity(500),
            order_history: VecDeque::with_capacity(max_history),
            max_history_size: max_history,
            level_orders: HashMap::with_capacity(100),
            min_levels_for_pattern: 3,
            max_level_spacing_ticks: 5,
            rapid_cancel_threshold_ms: 5.0,
            current_state: AnalyzerState::Normal,
            state_history: VecDeque::with_capacity(100),
            stats: LayeringStats::default(),
            last_analysis_ns: 0,
        }
    }

    /// Add a new order to tracking
    pub fn add_order(&mut self, order: LayerOrder) {
        let key = (order.price, order.side);
        
        // Track by level
        self.level_orders.entry(key).or_insert_with(Vec::new).push(order.order_id);
        
        // Add to active orders
        self.active_orders.insert(order.order_id, order.clone());
        
        // Update stats
        self.stats.total_orders_analyzed += 1;
        
        self.last_analysis_ns = order.timestamp_ns;
    }

    /// Mark an order as cancelled and check for layering
    pub fn cancel_order(&mut self, order_id: u64) -> Option<LayeringAnalysisResult> {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        if let Some(order) = self.active_orders.get_mut(&order_id) {
            order.is_cancelled = true;
            order.cancel_timestamp_ns = Some(now_ns);
            
            // Check lifetime for rapid cancellation
            if let Some(lifetime) = order.lifetime_ms() {
                if lifetime < self.rapid_cancel_threshold_ms {
                    self.stats.total_spoofed_volume += order.quantity;
                }
            }
            
            // Move to history
            let cloned_order = order.clone();
            self.order_history.push_back(cloned_order);
            
            if self.order_history.len() > self.max_history_size {
                self.order_history.pop_front();
            }
            
            // Remove from active
            let order = self.active_orders.remove(&order_id).unwrap();
            
            // Clean up level tracking
            let key = (order.price, order.side);
            if let Some(level_orders) = self.level_orders.get_mut(&key) {
                level_orders.retain(|&id| id != order_id);
                if level_orders.is_empty() {
                    self.level_orders.remove(&key);
                }
            }
            
            // Analyze for layering patterns
            return Some(self.analyze_layering(order.side));
        }
        None
    }

    /// Analyze for layering patterns on a specific side
    pub fn analyze_layering(&mut self, side: Side) -> LayeringAnalysisResult {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.last_analysis_ns = now_ns;

        // Collect recent orders on this side
        let recent_orders: Vec<&LayerOrder> = self.active_orders.values()
            .filter(|o| o.side == side && !o.is_cancelled)
            .collect();

        if recent_orders.len() < self.min_levels_for_pattern {
            return LayeringAnalysisResult {
                is_layering_detected: false,
                patterns: Vec::new(),
                spoofing_probability: 0.0,
                recommended_spread_adjustment_bps: 0.0,
            };
        }

        // Group by price level
        let mut level_map: HashMap<i64, Vec<&LayerOrder>> = HashMap::new();
        for order in &recent_orders {
            level_map.entry(order.price).or_insert_with(Vec::new).push(order);
        }

        // Sort levels
        let mut levels: Vec<i64> = level_map.keys().copied().collect();
        levels.sort();

        // Detect consecutive level patterns
        let mut patterns = Vec::new();
        let mut current_pattern_levels = Vec::new();
        let mut current_pattern_orders = Vec::new();

        for i in 0..levels.len() {
            let current_level = levels[i];
            
            if i == 0 {
                current_pattern_levels.push(current_level);
                current_pattern_orders.extend(level_map[&current_level].iter().map(|o| o.order_id));
                continue;
            }

            let prev_level = levels[i - 1];
            let spacing = (current_level - prev_level).abs();

            if spacing <= self.max_level_spacing_ticks {
                current_pattern_levels.push(current_level);
                current_pattern_orders.extend(level_map[&current_level].iter().map(|o| o.order_id));
            } else {
                // End of potential pattern
                if current_pattern_levels.len() >= self.min_levels_for_pattern {
                    if let Some(pattern) = self.create_pattern(
                        side,
                        current_pattern_levels.clone(),
                        current_pattern_orders.clone(),
                        &level_map,
                    ) {
                        patterns.push(pattern);
                    }
                }
                current_pattern_levels.clear();
                current_pattern_orders.clear();
                current_pattern_levels.push(current_level);
                current_pattern_orders.extend(level_map[&current_level].iter().map(|o| o.order_id));
            }
        }

        // Check final pattern
        if current_pattern_levels.len() >= self.min_levels_for_pattern {
            if let Some(pattern) = self.create_pattern(
                side,
                current_pattern_levels.clone(),
                current_pattern_orders.clone(),
                &level_map,
            ) {
                patterns.push(pattern);
            }
        }

        // Calculate overall spoofing probability
        let spoofing_prob = if patterns.is_empty() {
            0.0
        } else {
            patterns.iter().map(|p| p.confidence_score).sum::<f64>() / patterns.len() as f64
        };

        // Determine state and spread adjustment
        self.update_state(spoofing_prob, patterns.len());
        let spread_adjustment = self.calculate_spread_adjustment();

        // Update stats
        if !patterns.is_empty() {
            self.stats.layering_patterns_detected += patterns.len() as u64;
            self.stats.avg_pattern_confidence = 
                (self.stats.avg_pattern_confidence * (self.stats.layering_patterns_detected - 1) as f64 
                 + spoofing_prob) / self.stats.layering_patterns_detected as f64;
        }

        LayeringAnalysisResult {
            is_layering_detected: !patterns.is_empty(),
            patterns,
            spoofing_probability: spoofing_prob,
            recommended_spread_adjustment_bps: spread_adjustment,
        }
    }

    /// Differentiate genuine market making from malicious layering
    pub fn differentiate_market_making(&self, order_id: u64) -> bool {
        // Genuine market makers exhibit:
        // 1. Longer order lifetimes (>100ms)
        // 2. Balanced bid/ask presence
        // 3. Consistent quoting behavior
        
        if let Some(order) = self.active_orders.get(&order_id) {
            let age = order.age_ms();
            
            // Check lifetime threshold
            if age < 100.0 {
                return false;
            }

            // Check for balanced presence
            let bid_count = self.active_orders.values()
                .filter(|o| o.side == Side::Bid && !o.is_cancelled)
                .count();
            let ask_count = self.active_orders.values()
                .filter(|o| o.side == Side::Ask && !o.is_cancelled)
                .count();

            if bid_count > 0 && ask_count > 0 {
                let balance_ratio = (bid_count.min(ask_count) as f64) / (bid_count.max(ask_count) as f64);
                if balance_ratio > 0.5 {
                    return true;
                }
            }

            // Check historical behavior
            let historical_orders: Vec<&LayerOrder> = self.order_history.iter()
                .filter(|o| o.exchange_order_id.starts_with(&order.exchange_order_id[..4.min(order.exchange_order_id.len())]))
                .collect();

            if !historical_orders.is_empty() {
                let avg_lifetime: f64 = historical_orders.iter()
                    .filter_map(|o| o.lifetime_ms())
                    .sum::<f64>() / historical_orders.len() as f64;

                if avg_lifetime > 200.0 {
                    return true;
                }
            }
        }

        false
    }

    /// Get current analyzer state
    pub fn get_state(&self) -> AnalyzerState {
        self.current_state
    }

    /// Get statistics
    pub fn get_stats(&self) -> &LayeringStats {
        &self.stats
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.active_orders.clear();
        self.order_history.clear();
        self.level_orders.clear();
        self.state_history.clear();
        self.stats = LayeringStats::default();
        self.current_state = AnalyzerState::Normal;
    }

    fn create_pattern(
        &self,
        side: Side,
        levels: Vec<i64>,
        order_ids: Vec<u64>,
        level_map: &HashMap<i64, Vec<&LayerOrder>>,
    ) -> Option<LayeringPattern> {
        if levels.is_empty() {
            return None;
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Calculate total volume
        let total_volume: u64 = order_ids.iter()
            .filter_map(|id| self.active_orders.get(id))
            .map(|o| o.quantity)
            .sum();

        // Calculate average lifetime
        let lifetimes: Vec<f64> = order_ids.iter()
            .filter_map(|id| self.active_orders.get(id))
            .filter_map(|o| o.lifetime_ms())
            .collect();
        
        let avg_lifetime = if lifetimes.is_empty() {
            order_ids.iter()
                .filter_map(|id| self.active_orders.get(id))
                .map(|o| o.age_ms())
                .sum::<f64>() / order_ids.len() as f64
        } else {
            lifetimes.iter().sum::<f64>() / lifetimes.len() as f64
        };

        // Calculate confidence based on pattern characteristics
        let level_confidence = (levels.len() as f64 / 10.0).min(1.0);
        let volume_confidence = (total_volume as f64 / 100.0).min(1.0);
        let lifetime_confidence = if avg_lifetime < self.rapid_cancel_threshold_ms {
            1.0
        } else {
            (1.0 - avg_lifetime / 100.0).max(0.0)
        };

        let confidence_score = 0.4 * level_confidence + 0.3 * volume_confidence + 0.3 * lifetime_confidence;

        // Determine pattern type
        let pattern_type = if levels.len() >= 5 {
            PatternType::StackedLiquidity
        } else if avg_lifetime < 10.0 {
            PatternType::QuoteStuffing
        } else {
            PatternType::ConsecutiveLevels
        };

        // Determine recommended action
        let action = if confidence_score > 0.8 {
            LayeringAction::IgnoreLiquidity
        } else if confidence_score > 0.6 {
            LayeringAction::WidenSpread
        } else if confidence_score > 0.4 {
            LayeringAction::FlagForReview
        } else {
            LayeringAction::Monitor
        };

        Some(LayeringPattern {
            detection_timestamp_ns: now_ns,
            side,
            levels,
            order_ids,
            total_volume,
            avg_lifetime_ms: avg_lifetime,
            confidence_score,
            pattern_type,
            action_recommended: action,
        })
    }

    fn update_state(&mut self, spoofing_prob: f64, pattern_count: usize) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let new_state = if spoofing_prob > 0.8 || pattern_count >= 3 {
            AnalyzerState::Critical
        } else if spoofing_prob > 0.6 || pattern_count >= 2 {
            AnalyzerState::ConfirmedLayering
        } else if spoofing_prob > 0.3 {
            AnalyzerState::Suspicious
        } else {
            AnalyzerState::Normal
        };

        if new_state != self.current_state {
            self.current_state = new_state;
            self.state_history.push_back((new_state, now_ns));
            
            if self.state_history.len() > 100 {
                self.state_history.pop_front();
            }
        }
    }

    fn calculate_spread_adjustment(&self) -> f64 {
        match self.current_state {
            AnalyzerState::Normal => 0.0,
            AnalyzerState::Suspicious => 2.0,      // 2 bps
            AnalyzerState::ConfirmedLayering => 5.0, // 5 bps
            AnalyzerState::Critical => 10.0,        // 10 bps
        }
    }
}

impl Default for LayeringAnalyzer {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layering_detection() {
        let mut analyzer = LayeringAnalyzer::new(1000);
        
        // Add orders at consecutive levels (potential layering)
        for i in 0..5 {
            let order = LayerOrder {
                order_id: i,
                price: 50000 - i as i64,
                quantity: 100,
                side: Side::Bid,
                timestamp_ns: 1000000000,
                exchange_order_id: format!("TEST{}", i),
                is_cancelled: false,
                cancel_timestamp_ns: None,
            };
            analyzer.add_order(order);
        }

        // Cancel one order to trigger analysis
        let result = analyzer.cancel_order(0);
        
        assert!(result.is_some());
        let result = result.unwrap();
        
        // Should detect layering pattern
        assert_eq!(result.is_layering_detected, true);
        assert!(!result.patterns.is_empty());
    }

    #[test]
    fn test_genuine_market_maker_differentiation() {
        let mut analyzer = LayeringAnalyzer::new(1000);
        
        // Add a long-lived order (genuine MM behavior)
        let order = LayerOrder {
            order_id: 1,
            price: 50000,
            quantity: 100,
            side: Side::Bid,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64 - 200_000_000, // 200ms ago
            exchange_order_id: "MM001".to_string(),
            is_cancelled: false,
            cancel_timestamp_ns: None,
        };
        analyzer.add_order(order);

        let is_genuine = analyzer.differentiate_market_making(1);
        assert!(is_genuine);
    }
}
