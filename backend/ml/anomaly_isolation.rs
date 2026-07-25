//! Anomaly Isolation Forest for Spoofing Detection
//! 
//! Implements Isolation Forest algorithm for detecting statistical outliers
//! in order book data, including spoofing patterns and exchange API lag.
//! Optimized for O(N) time complexity and minimal memory footprint.

use std::collections::VecDeque;
use rayon::prelude::*;

/// Configuration for Isolation Forest
#[derive(Clone, Debug)]
pub struct IsolationForestConfig {
    pub n_trees: usize,
    pub sample_size: usize,
    pub max_depth: usize,
    pub contamination: f64,  // Expected proportion of outliers
    pub n_features_sample: usize,
}

impl Default for IsolationForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            sample_size: 256,
            max_depth: 8,
            contamination: 0.05,
            n_features_sample: 10,
        }
    }
}

/// Single isolation tree node
#[derive(Clone, Debug)]
pub enum IsolationNode {
    Internal {
        feature_idx: usize,
        split_value: f64,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
        size: usize,
    },
    Leaf {
        size: usize,
        height: usize,
    },
}

impl IsolationNode {
    pub fn new_leaf(size: usize, height: usize) -> Self {
        IsolationNode::Leaf { size, height }
    }
    
    pub fn new_internal(
        feature_idx: usize,
        split_value: f64,
        left: IsolationNode,
        right: IsolationNode,
        size: usize,
    ) -> Self {
        IsolationNode::Internal {
            feature_idx,
            split_value,
            left: Box::new(left),
            right: Box::new(right),
            size,
        }
    }
}

/// Isolation Tree implementation
pub struct IsolationTree {
    root: Option<IsolationNode>,
    height_limit: usize,
}

impl IsolationTree {
    pub fn new(height_limit: usize) -> Self {
        Self {
            root: None,
            height_limit,
        }
    }
    
    /// Build tree from sample data
    pub fn fit(&mut self, data: &[Vec<f64>], current_height: usize) {
        if data.is_empty() || current_height >= self.height_limit {
            self.root = Some(IsolationNode::new_leaf(data.len(), current_height));
            return;
        }
        
        if data.len() == 1 {
            self.root = Some(IsolationNode::new_leaf(1, current_height));
            return;
        }
        
        // Randomly select feature
        let n_features = data[0].len();
        let feature_idx = rand::random::<usize>() % n_features;
        
        // Get min/max for selected feature
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        
        for row in data {
            let val = row[feature_idx];
            if val < min_val { min_val = val; }
            if val > max_val { max_val = val; }
        }
        
        // If all values same, create leaf
        if min_val == max_val {
            self.root = Some(IsolationNode::new_leaf(data.len(), current_height));
            return;
        }
        
        // Random split value
        let split_value = {
            let mut rng = rand::thread_rng();
            let random_val = rng.gen::<f64>();
            min_val + random_val * (max_val - min_val)
        };
        
        // Partition data
        let mut left_data: Vec<Vec<f64>> = Vec::with_capacity(data.len());
        let mut right_data: Vec<Vec<f64>> = Vec::with_capacity(data.len());
        
        for row in data {
            if row[feature_idx] < split_value {
                left_data.push(row.clone());
            } else {
                right_data.push(row.clone());
            }
        }
        
        // Recursively build subtrees
        let mut left_tree = IsolationTree::new(self.height_limit);
        left_tree.fit(&left_data, current_height + 1);
        
        let mut right_tree = IsolationTree::new(self.height_limit);
        right_tree.fit(&right_data, current_height + 1);
        
        self.root = Some(IsolationNode::new_internal(
            feature_idx,
            split_value,
            left_tree.root.unwrap_or_else(|| IsolationNode::new_leaf(left_data.len(), current_height + 1)),
            right_tree.root.unwrap_or_else(|| IsolationNode::new_leaf(right_data.len(), current_height + 1)),
            data.len(),
        ));
    }
    
    /// Get path length for a single observation
    pub fn path_length(&self, x: &[f64]) -> f64 {
        self.path_length_recursive(&self.root, x, 0)
    }
    
    fn path_length_recursive(&self, node: &Option<IsolationNode>, x: &[f64], current_height: usize) -> f64 {
        match node {
            Some(IsolationNode::Leaf { size, height }) => {
                let c_factor = Self::c_factor(*size);
                current_height as f64 + c_factor
            },
            Some(IsolationNode::Internal { feature_idx, split_value, left, right, .. }) => {
                if x[*feature_idx] < *split_value {
                    self.path_length_recursive(&Some(*left.clone()), x, current_height + 1)
                } else {
                    self.path_length_recursive(&Some(*right.clone()), x, current_height + 1)
                }
            },
            None => current_height as f64,
        }
    }
    
    /// Average path length for normalization
    fn c_factor(n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        if n == 2 {
            return 1.0;
        }
        
        let n_f = n as f64;
        2.0 * (n_f.ln() + 0.5772156649) - 2.0 * (n_f - 1.0) / n_f
    }
}

/// Isolation Forest for anomaly detection
pub struct AnomalyIsolationForest {
    config: IsolationForestConfig,
    trees: Vec<IsolationTree>,
    
    // Running statistics for normalization
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
    n_observations: u64,
    
    // History for adaptive threshold
    score_history: VecDeque<f64>,
    
    // Detection flags
    api_lag_detected: bool,
    spoofing_detected: bool,
}

impl AnomalyIsolationForest {
    pub fn new(config: IsolationForestConfig) -> Self {
        Self {
            config,
            trees: Vec::with_capacity(config.n_trees),
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
            n_observations: 0,
            score_history: VecDeque::with_capacity(1000),
            api_lag_detected: false,
            spoofing_detected: false,
        }
    }
    
    /// Fit forest on training data
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }
        
        let n_features = data[0].len();
        
        // Initialize running statistics
        self.feature_means = vec![0.0; n_features];
        self.feature_stds = vec![1.0; n_features];
        self.n_observations = 0;
        
        // Update statistics incrementally
        for row in data {
            self.update_statistics(row);
        }
        
        // Normalize data
        let normalized_data: Vec<Vec<f64>> = data
            .iter()
            .map(|row| self.normalize_row(row))
            .collect();
        
        // Build trees in parallel
        self.trees = (0..self.config.n_trees)
            .into_par_iter()
            .map(|_| {
                // Bootstrap sample
                let sample: Vec<Vec<f64>> = (0..self.config.sample_size.min(normalized_data.len()))
                    .map(|_| {
                        let idx = rand::random::<usize>() % normalized_data.len();
                        normalized_data[idx].clone()
                    })
                    .collect();
                
                let mut tree = IsolationTree::new(self.config.max_depth);
                tree.fit(&sample, 0);
                tree
            })
            .collect();
    }
    
    /// Update running statistics incrementally (Welford's algorithm)
    fn update_statistics(&mut self, x: &[f64]) {
        self.n_observations += 1;
        let n = self.n_observations as f64;
        
        if self.feature_means.is_empty() {
            self.feature_means = vec![0.0; x.len()];
            self.feature_stds = vec![1.0; x.len()];
        }
        
        for (i, &val) in x.iter().enumerate() {
            let delta = val - self.feature_means[i];
            self.feature_means[i] += delta / n;
            
            if i < self.feature_stds.len() {
                let delta2 = val - self.feature_means[i];
                self.feature_stds[i] = self.feature_stds[i] + delta * delta2;
            }
        }
        
        // Convert variance to std
        if self.n_observations > 1 {
            for i in 0..self.feature_stds.len() {
                self.feature_stds[i] = (self.feature_stds[i] / (n - 1.0)).sqrt().max(1e-8);
            }
        }
    }
    
    /// Normalize a row using running statistics
    fn normalize_row(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.feature_means)
            .zip(&self.feature_stds)
            .map(|((&val, &mean), &std)| (val - mean) / std)
            .collect()
    }
    
    /// Compute anomaly score for single observation
    pub fn anomaly_score(&self, x: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.5;  // Default score
        }
        
        let normalized_x = self.normalize_row(x);
        
        // Average path length across all trees
        let avg_path_length: f64 = self.trees
            .par_iter()
            .map(|tree| tree.path_length(&normalized_x))
            .sum::<f64>() / self.trees.len() as f64;
        
        // Normalize by expected path length
        let c_factor = IsolationTree::c_factor(self.config.sample_size);
        let score = 2.0_f64.powf(-avg_path_length / (c_factor + 1e-8));
        
        score
    }
    
    /// Classify observation as normal or anomaly
    pub fn classify(&self, x: &[f64]) -> (bool, f64) {
        let score = self.anomaly_score(x);
        
        // Adaptive threshold based on contamination
        let threshold = 1.0 - self.config.contamination;
        let is_anomaly = score > threshold;
        
        (is_anomaly, score)
    }
    
    /// Detect specific anomaly types
    pub fn detect_spoofing(&self, orderbook_features: &[f64]) -> bool {
        let (is_anomaly, score) = self.classify(orderbook_features);
        
        if !is_anomaly {
            self.spoofing_detected = false;
            return false;
        }
        
        // Additional spoofing-specific checks
        // High orderbook imbalance + rapid cancellation pattern
        let imbalance = orderbook_features.get(0).copied().unwrap_or(0.0);
        let cancellation_rate = orderbook_features.get(1).copied().unwrap_or(0.0);
        
        self.spoofing_detected = imbalance.abs() > 2.0 && cancellation_rate > 0.8;
        self.spoofing_detected
    }
    
    /// Detect exchange API lag anomalies
    pub fn detect_api_lag(&self, latency_features: &[f64]) -> bool {
        let latency = latency_features.get(0).copied().unwrap_or(0.0);
        let jitter = latency_features.get(1).copied().unwrap_or(0.0);
        
        // Latency spike detection
        let latency_threshold = 3.0;  // 3 standard deviations
        self.api_lag_detected = latency > latency_threshold || jitter > latency_threshold * 2.0;
        
        self.api_lag_detected
    }
    
    /// Batch anomaly scoring
    pub fn batch_score(&self, data: &[Vec<f64>]) -> Vec<f64> {
        data.par_iter()
            .map(|x| self.anomaly_score(x))
            .collect()
    }
    
    /// Get summary statistics
    pub fn get_summary(&self) -> serde_json::Value {
        use serde_json::json;
        
        json!({
            "n_trees": self.trees.len(),
            "n_observations": self.n_observations,
            "contamination": self.config.contamination,
            "api_lag_detected": self.api_lag_detected,
            "spoofing_detected": self.spoofing_detected,
            "score_history_len": self.score_history.len(),
        })
    }
    
    /// Reset detection flags
    pub fn reset_flags(&mut self) {
        self.api_lag_detected = false;
        self.spoofing_detected = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_forest_creation() {
        let config = IsolationForestConfig::default();
        let forest = AnomalyIsolationForest::new(config);
        
        assert_eq!(forest.config.n_trees, 100);
        assert!(forest.trees.is_empty());
    }
    
    #[test]
    fn test_anomaly_detection() {
        let config = IsolationForestConfig {
            n_trees: 10,
            sample_size: 64,
            ..Default::default()
        };
        
        let mut forest = AnomalyIsolationForest::new(config);
        
        // Generate normal data
        let normal_data: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![rand::random::<f64>() * 0.1, rand::random::<f64>() * 0.1])
            .collect();
        
        forest.fit(&normal_data);
        
        // Test normal point
        let normal_point = vec![0.05, 0.05];
        let (is_anomaly_normal, score_normal) = forest.classify(&normal_point);
        
        // Test anomalous point
        let anomaly_point = vec![5.0, 5.0];
        let (is_anomaly_extreme, score_extreme) = forest.classify(&anomaly_point);
        
        // Anomaly should have higher score
        assert!(score_extreme > score_normal);
    }
    
    #[test]
    fn test_api_lag_detection() {
        let config = IsolationForestConfig::default();
        let forest = AnomalyIsolationForest::new(config);
        
        // Normal latency
        let normal_latency = vec![0.5, 0.1];
        assert!(!forest.detect_api_lag(&normal_latency));
        
        // High latency
        let high_latency = vec![5.0, 0.5];
        assert!(forest.detect_api_lag(&high_latency));
    }
}
