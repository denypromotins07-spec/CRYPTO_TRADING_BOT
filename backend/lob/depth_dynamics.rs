//! Depth Dynamics Tracker for Order Book Elasticity
//! Tracks elasticity and shape changes of the L2 order book.

use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DepthSnapshot {
    pub timestamp: Instant,
    pub bid_depth: Vec<f64>,
    pub ask_depth: Vec<f64>,
    pub total_bid_volume: f64,
    pub total_ask_volume: f64,
}

#[derive(Debug)]
pub struct DepthDynamics {
    snapshots: VecDeque<DepthSnapshot>,
    max_snapshots: usize,
    num_levels: usize,
}

impl DepthDynamics {
    pub fn new(num_levels: usize, max_snapshots: usize) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(max_snapshots),
            max_snapshots,
            num_levels,
        }
    }
    
    pub fn add_snapshot(&mut self, bid_depth: Vec<f64>, ask_depth: Vec<f64>) {
        let snapshot = DepthSnapshot {
            timestamp: Instant::now(),
            total_bid_volume: bid_depth.iter().sum(),
            total_ask_volume: ask_depth.iter().sum(),
            bid_depth,
            ask_depth,
        };
        
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }
    
    pub fn get_elasticity(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 1.0;
        }
        
        let depths: Vec<_> = self.snapshots.iter().collect();
        let mut elasticity_sum = 0.0;
        
        for i in 1..depths.len() {
            let prev_total = depths[i-1].total_bid_volume + depths[i-1].total_ask_volume;
            let curr_total = depths[i].total_bid_volume + depths[i].total_ask_volume;
            
            if prev_total > 0.0 {
                let change = (curr_total - prev_total).abs() / prev_total;
                elasticity_sum += 1.0 / (1.0 + change);
            }
        }
        
        elasticity_sum / (depths.len() - 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_elasticity_calculation() {
        let mut dynamics = DepthDynamics::new(10, 100);
        dynamics.add_snapshot(vec![100.0; 10], vec![100.0; 10]);
        dynamics.add_snapshot(vec![95.0; 10], vec![105.0; 10]);
        
        let elasticity = dynamics.get_elasticity();
        assert!(elasticity > 0.0 && elasticity <= 1.0);
    }
}
