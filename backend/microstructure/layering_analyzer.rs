// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// File: backend/microstructure/layering_analyzer.rs
// Chapter 2: Spoofing Detection, Layering, and Fake Liquidity Filtering
//
// Purpose: Detect multi-level manipulation algorithms (layering detection)
// Constraints: Real-time analysis, memory-efficient pattern storage
// Target: AMD Ryzen AI 5 laptop with 8GB RAM limit
//
// Design Patterns: State Pattern for layering state machine, Observer for alerts
// Memory Model: Pre-allocated buffers, efficient interval trees for price levels

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Maximum number of price levels to track for layering patterns
const MAX_LEVELS_TRACKED: usize = 50;

/// Minimum number of stacked orders to consider as potential layering
const MIN_LAYER_COUNT: usize = 3;

/// Time window for detecting layering patterns (in milliseconds)
const LAYERING_WINDOW_MS: u64 = 100;

/// Unique identifier for layering detection sessions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayeringSessionId(pub u64);

/// Represents a single order in the layering analysis
#[derive(Debug, Clone)]
pub struct LayerOrder {
    pub order_id: String,
    pub price: i64,
    pub quantity: i64,
    pub side: OrderSide,
    pub timestamp_us: u64,
    pub cancelled_us: Option<u64>,
    pub filled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Bid,
    Ask,
}

/// A detected layer (stack of orders at consecutive price levels)
#[derive(Debug, Clone)]
pub struct DetectedLayer {
    pub side: OrderSide,
    pub start_price: i64,
    pub end_price: i64,
    pub order_count: usize,
    pub total_quantity: i64,
    pub creation_time_us: u64,
    pub completion_time_us: Option<u64>,
    pub all_cancelled: bool,
    pub confidence_score: f64,
}

/// Alert generated when layering is detected
#[derive(Debug, Clone)]
pub struct LayeringAlert {
    pub alert_id: u64,
    pub timestamp_us: u64,
    pub session_id: LayeringSessionId,
    pub layers: Vec<DetectedLayer>,
    pub severity: LayeringSeverity,
    pub estimated_manipulation_impact_bps: f64,
    pub recommended_action: LayeringAction,
    pub evidence: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayeringSeverity {
    Low,      // Suspicious but not confirmed
    Medium,   // Likely layering
    High,     // Confirmed layering pattern
    Critical, // Active manipulation in progress
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayeringAction {
    Monitor,           // Keep watching
    WidenSpread,       // Increase spread to avoid manipulation
    ReduceSize,        // Trade smaller sizes
    PauseTrading,      // Temporarily stop trading
    SwitchVenue,       // Move to different exchange
}

/// Tracks potential layering patterns across price levels
pub struct LayeringAnalyzer {
    /// Sequence number for generating unique IDs
    sequence_number: AtomicU64,
    
    /// Active sessions being analyzed
    active_sessions: HashMap<LayeringSessionId, LayeringSession>,
    
    /// Completed/closed sessions for historical analysis
    completed_sessions: VecDeque<LayeringSession>,
    
    /// Recent alerts generated
    recent_alerts: VecDeque<LayeringAlert>,
    
    /// Start time for relative timestamps
    start_time: Instant,
    
    /// Configuration
    min_layers_for_alert: usize,
    enabled: AtomicBool,
}

/// A session tracking potential layering activity
struct LayeringSession {
    pub id: LayeringSessionId,
    pub created_us: u64,
    pub last_update_us: u64,
    pub bid_orders: HashMap<i64, Vec<LayerOrder>>, // Price -> Orders
    pub ask_orders: HashMap<i64, Vec<LayerOrder>>,
    pub detected_bid_layers: Vec<DetectedLayer>,
    pub detected_ask_layers: Vec<DetectedLayer>,
    pub cancellation_pattern: CancellationPattern,
}

#[derive(Debug, Clone, Default)]
struct CancellationPattern {
    pub rapid_cancellations: u32,
    pub sequential_cancellations: u32,
    pub average_lifetime_us: u64,
}

impl LayeringSession {
    fn new(id: LayeringSessionId, timestamp_us: u64) -> Self {
        Self {
            id,
            created_us: timestamp_us,
            last_update_us: timestamp_us,
            bid_orders: HashMap::with_capacity(MAX_LEVELS_TRACKED),
            ask_orders: HashMap::with_capacity(MAX_LEVELS_TRACKED),
            detected_bid_layers: Vec::new(),
            detected_ask_layers: Vec::new(),
            cancellation_pattern: CancellationPattern::default(),
        }
    }
    
    /// Add an order to the session
    fn add_order(&mut self, order: LayerOrder) {
        self.last_update_us = order.timestamp_us;
        
        let map = match order.side {
            OrderSide::Bid => &mut self.bid_orders,
            OrderSide::Ask => &mut self.ask_orders,
        };
        
        map.entry(order.price).or_insert_with(Vec::new).push(order);
    }
    
    /// Mark an order as cancelled
    fn cancel_order(&mut self, order_id: &str, cancel_time_us: u64) {
        self.last_update_us = cancel_time_us;
        
        // Search in both bid and ask maps
        for map in [&mut self.bid_orders, &mut self.ask_orders] {
            for orders in map.values_mut() {
                if let Some(order) = orders.iter_mut().find(|o| o.order_id == order_id) {
                    order.cancelled_us = Some(cancel_time_us);
                    self.cancellation_pattern.rapid_cancellations += 1;
                    return;
                }
            }
        }
    }
    
    /// Mark an order as filled
    fn fill_order(&mut self, order_id: &str, fill_time_us: u64) {
        for map in [&mut self.bid_orders, &mut self.ask_orders] {
            for orders in map.values_mut() {
                if let Some(order) = orders.iter_mut().find(|o| o.order_id == order_id) {
                    order.filled = true;
                    return;
                }
            }
        }
    }
    
    /// Analyze current state for layering patterns
    fn analyze_for_layers(&mut self) -> (Vec<DetectedLayer>, Vec<DetectedLayer>) {
        let bid_layers = self.detect_layers_in_map(&self.bid_orders, OrderSide::Bid);
        let ask_layers = self.detect_layers_in_map(&self.ask_orders, OrderSide::Ask);
        
        self.detected_bid_layers = bid_layers.clone();
        self.detected_ask_layers = ask_layers.clone();
        
        (bid_layers, ask_layers)
    }
    
    /// Detect layered patterns in a price level map
    fn detect_layers_in_map(
        &self,
        orders: &HashMap<i64, Vec<LayerOrder>>,
        side: OrderSide,
    ) -> Vec<DetectedLayer> {
        if orders.is_empty() {
            return Vec::new();
        }
        
        let mut layers = Vec::new();
        let mut sorted_prices: Vec<i64> = orders.keys().copied().collect();
        sorted_prices.sort();
        
        // Find consecutive price levels with orders
        let mut current_layer_start: Option<i64> = None;
        let mut current_layer_end: i64 = 0;
        let mut current_layer_orders: Vec<&LayerOrder> = Vec::new();
        
        for (i, &price) in sorted_prices.iter().enumerate() {
            let order_list = orders.get(&price).unwrap();
            
            if current_layer_start.is_none() {
                current_layer_start = Some(price);
                current_layer_end = price;
            } else {
                let prev_price = current_layer_end;
                
                // Check if this price is consecutive (within tick size)
                let is_consecutive = (price - prev_price).abs() <= 5; // Assuming tick size of 1-5
                
                if is_consecutive {
                    current_layer_end = price;
                } else {
                    // End current layer, check if it qualifies
                    if current_layer_orders.len() >= MIN_LAYER_COUNT {
                        layers.push(self.create_layer(
                            current_layer_start.unwrap(),
                            current_layer_end,
                            &current_layer_orders,
                            side,
                        ));
                    }
                    
                    // Start new layer
                    current_layer_start = Some(price);
                    current_layer_end = price;
                    current_layer_orders.clear();
                }
            }
            
            // Add orders at this price to current layer
            current_layer_orders.extend(order_list.iter());
        }
        
        // Don't forget the last layer
        if let Some(start) = current_layer_start {
            if current_layer_orders.len() >= MIN_LAYER_COUNT {
                layers.push(self.create_layer(start, current_layer_end, &current_layer_orders, side));
            }
        }
        
        layers
    }
    
    fn create_layer(
        &self,
        start_price: i64,
        end_price: i64,
        orders: &[&LayerOrder],
        side: OrderSide,
    ) -> DetectedLayer {
        let total_quantity: i64 = orders.iter().map(|o| o.quantity).sum();
        
        // Calculate confidence based on various factors
        let mut confidence = 0.5;
        
        // More orders = higher confidence
        if orders.len() >= 5 {
            confidence += 0.2;
        } else if orders.len() >= 3 {
            confidence += 0.1;
        }
        
        // Check if orders are similar size (typical of layering)
        let avg_size = total_quantity as f64 / orders.len() as f64;
        let size_variance: f64 = orders.iter()
            .map(|o| ((o.quantity as f64 - avg_size).powi(2)))
            .sum::<f64>() / orders.len() as f64;
        
        if size_variance < (avg_size * 0.2).powi(2) {
            // Low variance = more likely algorithmic layering
            confidence += 0.15;
        }
        
        // Check cancellation rate
        let cancelled_count = orders.iter().filter(|o| o.cancelled_us.is_some()).count();
        if cancelled_count == orders.len() {
            confidence += 0.15; // All cancelled = strong signal
        }
        
        DetectedLayer {
            side,
            start_price,
            end_price,
            order_count: orders.len(),
            total_quantity,
            creation_time_us: orders.first().map(|o| o.timestamp_us).unwrap_or(0),
            completion_time_us: None,
            all_cancelled: cancelled_count == orders.len(),
            confidence_score: confidence.min(1.0),
        }
    }
}

impl LayeringAnalyzer {
    pub fn new() -> Self {
        Self {
            sequence_number: AtomicU64::new(0),
            active_sessions: HashMap::with_capacity(100),
            completed_sessions: VecDeque::with_capacity(1000),
            recent_alerts: VecDeque::with_capacity(500),
            start_time: Instant::now(),
            min_layers_for_alert: MIN_LAYER_COUNT,
            enabled: AtomicBool::new(true),
        }
    }
    
    /// Generate a new session ID
    fn next_session_id(&self) -> LayeringSessionId {
        LayeringSessionId(self.sequence_number.fetch_add(1, Ordering::SeqCst))
    }
    
    /// Get current timestamp in microseconds
    fn now_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }
    
    /// Process a new order event
    pub fn process_new_order(
        &mut self,
        order_id: String,
        price: i64,
        quantity: i64,
        side: OrderSide,
    ) -> Option<LayeringAlert> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }
        
        let timestamp_us = self.now_us();
        
        // Get or create active session
        let session_id = self.get_or_create_active_session();
        let session = self.active_sessions.get_mut(&session_id)?;
        
        let order = LayerOrder {
            order_id,
            price,
            quantity,
            side,
            timestamp_us,
            cancelled_us: None,
            filled: false,
        };
        
        session.add_order(order);
        
        // Analyze for layering patterns
        let (bid_layers, ask_layers) = session.analyze_for_layers();
        
        // Check if we should generate an alert
        self.check_and_generate_alert(session_id, &bid_layers, &ask_layers)
    }
    
    /// Process an order cancellation
    pub fn process_cancellation(
        &mut self,
        order_id: &str,
    ) -> Option<LayeringAlert> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }
        
        let timestamp_us = self.now_us();
        
        // Update all active sessions
        for session in self.active_sessions.values_mut() {
            session.cancel_order(order_id, timestamp_us);
        }
        
        // Re-analyze sessions after cancellation
        for (session_id, session) in &mut self.active_sessions {
            let (bid_layers, ask_layers) = session.analyze_for_layers();
            
            if let Some(alert) = self.check_and_generate_alert(*session_id, &bid_layers, &ask_layers) {
                return Some(alert);
            }
        }
        
        None
    }
    
    /// Process an order fill
    pub fn process_fill(&mut self, order_id: &str) {
        let timestamp_us = self.now_us();
        
        for session in self.active_sessions.values_mut() {
            session.fill_order(order_id, timestamp_us);
        }
    }
    
    /// Get or create an active session
    fn get_or_create_active_session(&mut self) -> LayeringSessionId {
        // Check if we have any active sessions
        if let Some((&id, _)) = self.active_sessions.iter().next() {
            return id;
        }
        
        // Create new session
        let id = self.next_session_id();
        let session = LayeringSession::new(id, self.now_us());
        self.active_sessions.insert(id, session);
        id
    }
    
    /// Check if layering patterns warrant an alert
    fn check_and_generate_alert(
        &mut self,
        session_id: LayeringSessionId,
        bid_layers: &[DetectedLayer],
        ask_layers: &[DetectedLayer],
    ) -> Option<LayeringAlert> {
        let mut all_layers = Vec::new();
        all_layers.extend(bid_layers.iter().cloned());
        all_layers.extend(ask_layers.iter().cloned());
        
        // Filter for high-confidence layers
        let suspicious_layers: Vec<_> = all_layers
            .into_iter()
            .filter(|l| l.confidence_score > 0.6 && l.order_count >= self.min_layers_for_alert)
            .collect();
        
        if suspicious_layers.is_empty() {
            return None;
        }
        
        // Determine severity
        let severity = if suspicious_layers.iter().any(|l| l.confidence_score > 0.85) {
            LayeringSeverity::Critical
        } else if suspicious_layers.iter().any(|l| l.confidence_score > 0.75) {
            LayeringSeverity::High
        } else if suspicious_layers.iter().any(|l| l.confidence_score > 0.65) {
            LayeringSeverity::Medium
        } else {
            LayeringSeverity::Low
        };
        
        // Calculate estimated impact
        let total_manipulated_qty: i64 = suspicious_layers.iter().map(|l| l.total_quantity).sum();
        let estimated_impact = (total_manipulated_qty as f64 / 100000.0).min(50.0); // Cap at 50 bps
        
        // Determine recommended action
        let action = match severity {
            LayeringSeverity::Low => LayeringAction::Monitor,
            LayeringSeverity::Medium => LayeringAction::WidenSpread,
            LayeringSeverity::High => LayeringAction::ReduceSize,
            LayeringSeverity::Critical => LayeringAction::PauseTrading,
        };
        
        // Build evidence string
        let evidence = format!(
            "Detected {} suspicious layers. Total manipulated quantity: {}. Bid layers: {}, Ask layers: {}",
            suspicious_layers.len(),
            total_manipulated_qty,
            bid_layers.iter().filter(|l| l.confidence_score > 0.6).count(),
            ask_layers.iter().filter(|l| l.confidence_score > 0.6).count()
        );
        
        let alert = LayeringAlert {
            alert_id: self.sequence_number.fetch_add(1, Ordering::SeqCst),
            timestamp_us: self.now_us(),
            session_id,
            layers: suspicious_layers,
            severity,
            estimated_manipulation_impact_bps: estimated_impact,
            recommended_action: action,
            evidence,
        };
        
        self.recent_alerts.push_back(alert.clone());
        Some(alert)
    }
    
    /// Differentiate between genuine market making and malicious layering
    pub fn is_genuine_market_making(&self, session: &LayeringSession) -> bool {
        // Genuine market makers:
        // 1. Have balanced bid/ask presence
        // 2. Don't cancel all orders simultaneously
        // 3. Get filled occasionally
        // 4. Maintain quotes for reasonable time
        
        let bid_count: usize = session.bid_orders.values().map(|v| v.len()).sum();
        let ask_count: usize = session.ask_orders.values().map(|v| v.len()).sum();
        
        // Check balance (genuine MM has roughly equal sides)
        let ratio = (bid_count as f64 / ask_count as f64).max(1.0) / (bid_count as f64 / ask_count as f64).min(1.0);
        if ratio > 3.0 {
            return false; // Heavily imbalanced = suspicious
        }
        
        // Check fill rate
        let total_orders = bid_count + ask_count;
        let filled_orders = session.bid_orders.values()
            .chain(session.ask_orders.values())
            .flat_map(|v| v.iter())
            .filter(|o| o.filled)
            .count();
        
        if total_orders > 0 && (filled_orders as f64 / total_orders as f64) > 0.1 {
            return true; // >10% fill rate suggests genuine activity
        }
        
        // Check cancellation pattern
        if session.cancellation_pattern.rapid_cancellations > 10 {
            return false; // Too many rapid cancellations
        }
        
        true
    }
    
    /// Get statistics about layering detection
    pub fn get_statistics(&self) -> LayeringStatistics {
        let total_sessions = self.active_sessions.len() + self.completed_sessions.len();
        let alerts_count = self.recent_alerts.len();
        
        let critical_alerts = self.recent_alerts.iter()
            .filter(|a| a.severity == LayeringSeverity::Critical)
            .count();
        
        LayeringStatistics {
            total_sessions,
            active_sessions: self.active_sessions.len(),
            total_alerts: alerts_count,
            critical_alerts,
            average_layers_per_session: 0.0, // Would calculate from sessions
        }
    }
    
    /// Enable/disable the analyzer
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
}

impl Default for LayeringAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about layering detection
#[derive(Debug, Clone, Default)]
pub struct LayeringStatistics {
    pub total_sessions: usize,
    pub active_sessions: usize,
    pub total_alerts: usize,
    pub critical_alerts: usize,
    pub average_layers_per_session: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layering_detection_basic() {
        let mut analyzer = LayeringAnalyzer::new();
        
        // Simulate layering: multiple orders at consecutive prices
        for i in 0..5 {
            let price = 50000 + i;
            let _alert = analyzer.process_new_order(
                format!("ORDER-{}", i),
                price,
                1000,
                OrderSide::Bid,
            );
        }
        
        let stats = analyzer.get_statistics();
        assert!(stats.active_sessions > 0);
    }

    #[test]
    fn test_genuine_vs_spoof_differentiation() {
        // This test would verify that genuine market making
        // is not flagged as layering while spoofing is
    }
}
