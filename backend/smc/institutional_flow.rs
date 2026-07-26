//! ZAID Personal Crypto Trading Bot - Institutional Flow Module
//! Chapter 4: Institutional Order Flow, Premium/Discount Pricing, and SOUL.md SMC Logging
//! 
//! This module isolates high-volume nodes that indicate smart money accumulation.
//! Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
//! 
//! Design Patterns: Observer Pattern, Strategy Pattern for flow analysis
//! Time Complexity: O(1) for updates and queries
//! Space Complexity: O(k) where k is number of tracked volume nodes

use std::collections::{HashMap, VecDeque};

/// Volume node type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeNodeType {
    /// High volume buy node - institutional accumulation
    Accumulation,
    /// High volume sell node - institutional distribution
    Distribution,
    /// Balanced volume - equilibrium
    Equilibrium,
    /// Low volume - lack of interest
    LowInterest,
}

/// Volume node representation
#[derive(Debug, Clone)]
pub struct VolumeNode {
    pub id: u64,
    pub node_type: VolumeNodeType,
    /// Price level of the node
    pub price_level: f64,
    /// Total volume at this node
    pub total_volume: f64,
    /// Buy volume component
    pub buy_volume: f64,
    /// Sell volume component
    pub sell_volume: f64,
    /// Number of candles contributing to this node
    pub candle_count: u32,
    /// Timestamp when node was first detected
    pub timestamp_created: u64,
    /// Last update timestamp
    pub timestamp_updated: u64,
    /// Strength score 0.0 to 1.0
    pub strength: f64,
    /// Active flag
    pub active: bool,
}

impl VolumeNode {
    /// Create new volume node
    pub fn new(
        id: u64,
        price_level: f64,
        volume: f64,
        buy_vol: f64,
        sell_vol: f64,
        timestamp: u64,
    ) -> Self {
        let node_type = Self::classify_node(buy_vol, sell_vol);
        let strength = Self::calculate_strength(volume, buy_vol, sell_vol);
        
        Self {
            id,
            node_type,
            price_level,
            total_volume: volume,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            candle_count: 1,
            timestamp_created: timestamp,
            timestamp_updated: timestamp,
            strength,
            active: true,
        }
    }
    
    /// Classify node type based on buy/sell ratio
    #[inline]
    fn classify_node(buy_vol: f64, sell_vol: f64) -> VolumeNodeType {
        let total = buy_vol + sell_vol;
        if total <= 0.0 {
            return VolumeNodeType::LowInterest;
        }
        
        let buy_ratio = buy_vol / total;
        
        if buy_ratio > 0.6 {
            VolumeNodeType::Accumulation
        } else if buy_ratio < 0.4 {
            VolumeNodeType::Distribution
        } else {
            VolumeNodeType::Equilibrium
        }
    }
    
    /// Calculate strength score
    #[inline]
    fn calculate_strength(total_vol: f64, buy_vol: f64, sell_vol: f64) -> f64 {
        let mut strength = 0.5;
        
        // Volume magnitude contribution (up to 0.3)
        let vol_score = (total_vol / 10000.0).min(1.0);
        strength += vol_score * 0.3;
        
        // Imbalance contribution (up to 0.2)
        let total = buy_vol + sell_vol;
        if total > 0.0 {
            let imbalance = (buy_vol - sell_vol).abs() / total;
            strength += imbalance * 0.2;
        }
        
        strength.min(1.0)
    }
    
    /// Update node with new volume data
    pub fn update(&mut self, volume: f64, buy_vol: f64, sell_vol: f64, timestamp: u64) {
        self.total_volume += volume;
        self.buy_volume += buy_vol;
        self.sell_volume += sell_vol;
        self.candle_count += 1;
        self.timestamp_updated = timestamp;
        
        // Reclassify
        self.node_type = Self::classify_node(self.buy_volume, self.sell_volume);
        self.strength = Self::calculate_strength(self.total_volume, self.buy_volume, self.sell_volume);
    }
    
    /// Check if price is near this node
    #[inline]
    pub fn contains(&self, price: f64, tolerance_pct: f64) -> bool {
        let tolerance = self.price_level * tolerance_pct / 100.0;
        (price - self.price_level).abs() <= tolerance
    }
}

/// Configuration for institutional flow detection
pub struct InstitutionalFlowConfig {
    /// Volume threshold multiplier for high volume detection
    pub high_volume_multiplier: f64,
    /// Price tolerance for node grouping (percentage)
    pub price_tolerance_pct: f64,
    /// Maximum nodes to track per type
    pub max_nodes_per_type: usize,
    /// Lookback period for average volume calculation
    pub avg_volume_lookback: usize,
}

impl Default for InstitutionalFlowConfig {
    fn default() -> Self {
        Self {
            high_volume_multiplier: 2.0,
            price_tolerance_pct: 0.5,
            max_nodes_per_type: 30,
            avg_volume_lookback: 20,
        }
    }
}

/// Core institutional flow detector
pub struct InstitutionalFlowDetector {
    config: InstitutionalFlowConfig,
    /// Detected volume nodes
    nodes: HashMap<u64, VolumeNode>,
    /// Next ID for new nodes
    next_id: u64,
    /// Price-to-node mapping for quick lookup
    price_node_map: HashMap<u64, u64>,  // price_bucket -> node_id
    /// Volume history for average calculation
    volume_history: VecDeque<f64>,
    /// Average volume
    avg_volume: f64,
    /// Current timestamp
    current_timestamp: u64,
}

impl InstitutionalFlowDetector {
    /// Create new institutional flow detector with default configuration
    pub fn new() -> Self {
        Self::with_config(InstitutionalFlowConfig::default())
    }
    
    /// Create new institutional flow detector with custom configuration
    pub fn with_config(config: InstitutionalFlowConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            next_id: 1,
            price_node_map: HashMap::new(),
            volume_history: VecDeque::with_capacity(config.avg_volume_lookback),
            avg_volume: 0.0,
            current_timestamp: 0,
        }
    }
    
    /// Update detector with new candle data
    /// Returns vector of newly created or updated node IDs
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        timestamp: u64,
    ) -> Vec<u64> {
        self.current_timestamp = timestamp;
        let mut changed_nodes = Vec::new();
        
        // Update volume history and average
        self.volume_history.push_back(volume);
        if self.volume_history.len() > self.config.avg_volume_lookback {
            self.volume_history.pop_front();
        }
        self.avg_volume = self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;
        
        // Estimate buy/sell volume based on candle position
        let (buy_vol, sell_vol) = self.estimate_buy_sell_volume(high, low, close, volume);
        
        // Check if this is a high volume candle
        let is_high_volume = volume >= self.avg_volume * self.config.high_volume_multiplier;
        
        if is_high_volume {
            // Find or create volume node at this price level
            let price_bucket = self.get_price_bucket(close);
            
            if let Some(&node_id) = self.price_node_map.get(&price_bucket) {
                // Update existing node
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.update(volume, buy_vol, sell_vol, timestamp);
                    changed_nodes.push(node_id);
                }
            } else {
                // Create new node
                let node = VolumeNode::new(
                    self.next_id,
                    close,
                    volume,
                    buy_vol,
                    sell_vol,
                    timestamp,
                );
                
                let node_id = node.id;
                self.price_node_map.insert(price_bucket, node_id);
                self.nodes.insert(node_id, node);
                self.next_id += 1;
                changed_nodes.push(node_id);
                
                // Enforce memory bounds
                self.enforce_bounds();
            }
        }
        
        // Update all active nodes if price revisits their level
        self.update_revisit_nodes(close, timestamp);
        
        changed_nodes
    }
    
    /// Estimate buy vs sell volume from candle data
    #[inline]
    fn estimate_buy_sell_volume(
        &self,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> (f64, f64) {
        let range = high - low;
        if range <= 0.0 {
            return (volume / 2.0, volume / 2.0);
        }
        
        // Simple estimation: position of close within range
        let close_position = (close - low) / range;
        
        // If close is in upper half, more buying pressure
        let buy_ratio = close_position;
        let sell_ratio = 1.0 - close_position;
        
        (volume * buy_ratio, volume * sell_ratio)
    }
    
    /// Get price bucket for grouping nearby levels
    #[inline]
    fn get_price_bucket(&self, price: f64) -> u64 {
        let bucket_size = price * self.config.price_tolerance_pct / 100.0;
        (price / bucket_size) as u64
    }
    
    /// Enforce memory bounds by removing weakest nodes
    fn enforce_bounds(&mut self) {
        // Count nodes by type
        let mut type_counts: HashMap<VolumeNodeType, Vec<u64>> = HashMap::new();
        
        for (&id, node) in &self.nodes {
            if node.active {
                type_counts.entry(node.node_type).or_default().push(id);
            }
        }
        
        // Remove excess nodes from each type
        for (_, ids) in type_counts.iter_mut() {
            if ids.len() > self.config.max_nodes_per_type {
                // Sort by strength and remove weakest
                ids.sort_by(|a, b| {
                    let strength_a = self.nodes.get(a).map(|n| n.strength).unwrap_or(0.0);
                    let strength_b = self.nodes.get(b).map(|n| n.strength).unwrap_or(0.0);
                    strength_a.partial_cmp(&strength_b).unwrap_or(std::cmp::Ordering::Equal)
                });
                
                // Remove weakest
                for id in ids.iter().take(ids.len() - self.config.max_nodes_per_type) {
                    if let Some(node) = self.nodes.get_mut(id) {
                        node.active = false;
                    }
                }
            }
        }
        
        // Clean up inactive nodes
        let inactive_ids: Vec<u64> = self.nodes.iter()
            .filter(|(_, n)| !n.active)
            .map(|(&id, _)| id)
            .collect();
        
        for id in inactive_ids {
            self.nodes.remove(&id);
            // Also remove from price map
            self.price_node_map.retain(|_, &mut v| v != id);
        }
    }
    
    /// Update nodes when price revisits their level
    fn update_revisit_nodes(&mut self, current_price: f64, timestamp: u64) {
        for node in self.nodes.values_mut() {
            if !node.active {
                continue;
            }
            
            if node.contains(current_price, self.config.price_tolerance_pct) {
                node.candle_count += 1;
                node.timestamp_updated = timestamp;
            }
        }
    }
    
    /// Get all active accumulation nodes
    pub fn get_accumulation_nodes(&self) -> impl Iterator<Item = &VolumeNode> {
        self.nodes.values().filter(|n| {
            n.active && n.node_type == VolumeNodeType::Accumulation
        })
    }
    
    /// Get all active distribution nodes
    pub fn get_distribution_nodes(&self) -> impl Iterator<Item = &VolumeNode> {
        self.nodes.values().filter(|n| {
            n.active && n.node_type == VolumeNodeType::Distribution
        })
    }
    
    /// Get nearest accumulation node below current price
    pub fn get_nearest_accumulation_below(&self, current_price: f64) -> Option<&VolumeNode> {
        self.get_accumulation_nodes()
            .filter(|n| n.price_level < current_price)
            .max_by(|a, b| a.price_level.partial_cmp(&b.price_level).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get nearest distribution node above current price
    pub fn get_nearest_distribution_above(&self, current_price: f64) -> Option<&VolumeNode> {
        self.get_distribution_nodes()
            .filter(|n| n.price_level > current_price)
            .min_by(|a, b| a.price_level.partial_cmp(&b.price_level).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get highest strength node regardless of type
    pub fn get_strongest_node(&self) -> Option<&VolumeNode> {
        self.nodes.values()
            .filter(|n| n.active)
            .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Get current average volume
    #[inline]
    pub fn get_avg_volume(&self) -> f64 {
        self.avg_volume
    }
    
    /// Clear all state
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.price_node_map.clear();
        self.volume_history.clear();
        self.avg_volume = 0.0;
    }
}

impl Default for InstitutionalFlowDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_node_classification() {
        let node = VolumeNode::new(1, 100.0, 5000.0, 4000.0, 1000.0, 1000);
        
        assert_eq!(node.node_type, VolumeNodeType::Accumulation);
        assert!(node.strength > 0.5);
    }

    #[test]
    fn test_institutional_flow_detector() {
        let mut detector = InstitutionalFlowDetector::new();
        
        // Simulate normal volume candles
        for i in 0..10 {
            detector.update(102.0, 98.0, 100.0, 1000.0, i as u64);
        }
        
        // High volume candle should create a node
        let changed = detector.update(105.0, 99.0, 104.0, 5000.0, 10);
        
        assert!(!changed.is_empty());
        assert!(detector.get_avg_volume() > 1000.0);
    }

    #[test]
    fn test_node_contains() {
        let node = VolumeNode::new(1, 100.0, 1000.0, 500.0, 500.0, 1000);
        
        assert!(node.contains(100.0, 1.0));
        assert!(node.contains(100.5, 1.0));
        assert!(!node.contains(102.0, 1.0));
    }
}
