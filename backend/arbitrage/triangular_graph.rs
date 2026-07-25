//! Triangular Arbitrage Graph - Directed Graph for BTC-ETH-USDT Loops
//!
//! This module builds and maintains a directed graph representation of
//! triangular arbitrage opportunities. Optimized for zero allocations
//! during tick updates using pre-allocated edge arrays.
//!
//! Chapter 2: Triangular Arbitrage and Cross-Margin Efficiency Optimization

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of symbols supported (fixed for zero allocation)
pub const MAX_SYMBOLS: usize = 10;

/// Maximum number of edges in the graph (complete graph: n*(n-1))
pub const MAX_EDGES: usize = MAX_SYMBOLS * (MAX_SYMBOLS - 1);

/// Represents a trading pair edge in the arbitrage graph
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    /// Source symbol index
    pub from: u8,
    /// Destination symbol index
    pub to: u8,
    /// Exchange rate (how much of `to` per unit of `from`)
    pub rate: f64,
    /// Log rate (negative for Bellman-Ford): -ln(rate)
    pub log_rate_neg: f64,
    /// Trading fee in basis points
    pub fee_bps: u32,
    /// Last update timestamp in nanoseconds
    pub last_update_ns: u64,
    /// Whether this edge is currently active
    pub active: bool,
}

impl Edge {
    /// Create a new edge with given parameters
    #[inline]
    pub fn new(from: u8, to: u8, rate: f64, fee_bps: u32) -> Self {
        let rate_after_fee = rate * (1.0 - (fee_bps as f64 / 10000.0));
        let log_rate_neg = -rate_after_fee.ln();
        
        Self {
            from,
            to,
            rate,
            log_rate_neg,
            fee_bps,
            last_update_ns: 0,
            active: true,
        }
    }

    /// Update edge rate atomically
    #[inline]
    pub fn update_rate(&mut self, new_rate: f64) {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        self.rate = new_rate;
        let rate_after_fee = new_rate * (1.0 - (self.fee_bps as f64 / 10000.0));
        self.log_rate_neg = -rate_after_fee.ln();
        self.last_update_ns = now_ns;
    }

    /// Get effective rate after fees
    #[inline]
    pub const fn effective_rate(&self) -> f64 {
        self.rate * (1.0 - (self.fee_bps as f64 / 10000.0))
    }
}

/// Pre-allocated triangular arbitrage graph
/// Uses fixed-size arrays to avoid heap allocations during critical paths
pub struct TriangularGraph {
    /// Symbol name to index mapping
    symbol_map: HashMap<&'static str, u8>,
    /// Index to symbol name (reverse mapping)
    index_map: [&'static str; MAX_SYMBOLS],
    /// Number of active symbols
    symbol_count: u8,
    /// All edges stored in contiguous array
    edges: [Option<Edge>; MAX_EDGES],
    /// Edge count for quick iteration
    edge_count: AtomicU64,
    /// Adjacency matrix for O(1) edge lookup
    adjacency: [[Option<usize>; MAX_SYMBOLS]; MAX_SYMBOLS],
    /// Last graph update timestamp
    last_update_ns: AtomicU64,
}

impl TriangularGraph {
    /// Create a new triangular graph with standard crypto symbols
    pub fn new() -> Self {
        // Initialize with common arbitrage symbols
        let mut symbol_map = HashMap::new();
        let mut index_map: [&str; MAX_SYMBOLS] = Default::default();
        
        let symbols: [&str; 4] = ["BTC", "ETH", "SOL", "USDT"];
        
        for (i, &sym) in symbols.iter().enumerate() {
            symbol_map.insert(sym, i as u8);
            index_map[i] = sym;
        }
        
        Self {
            symbol_map,
            index_map,
            symbol_count: symbols.len() as u8,
            edges: [None; MAX_EDGES],
            edge_count: AtomicU64::new(0),
            adjacency: [[None; MAX_SYMBOLS]; MAX_SYMBOLS],
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Register a new symbol (up to MAX_SYMBOLS)
    #[inline]
    pub fn add_symbol(&mut self, symbol: &'static str) -> Option<u8> {
        if self.symbol_count >= MAX_SYMBOLS as u8 {
            return None;
        }
        
        if let Some(&idx) = self.symbol_map.get(symbol) {
            return Some(idx);
        }
        
        let idx = self.symbol_count;
        self.symbol_map.insert(symbol, idx);
        self.index_map[idx as usize] = symbol;
        self.symbol_count += 1;
        
        Some(idx)
    }

    /// Get symbol index by name
    #[inline]
    pub fn get_symbol_index(&self, symbol: &str) -> Option<u8> {
        self.symbol_map.get(symbol).copied()
    }

    /// Get symbol name by index
    #[inline]
    pub fn get_symbol_name(&self, index: u8) -> Option<&'static str> {
        if index < self.symbol_count {
            Some(self.index_map[index as usize])
        } else {
            None
        }
    }

    /// Add or update an edge in the graph - O(1)
    #[inline]
    pub fn update_edge(&mut self, from: &str, to: &str, rate: f64, fee_bps: u32) -> bool {
        let from_idx = match self.symbol_map.get(from) {
            Some(&idx) => idx,
            None => return false,
        };
        
        let to_idx = match self.symbol_map.get(to) {
            Some(&idx) => idx,
            None => return false,
        };
        
        let edge_idx = (from_idx as usize) * MAX_SYMBOLS + (to_idx as usize);
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        if let Some(ref mut edge) = &mut self.edges[edge_idx] {
            edge.update_rate(rate);
        } else {
            let mut edge = Edge::new(from_idx, to_idx, rate, fee_bps);
            edge.last_update_ns = now_ns;
            self.edges[edge_idx] = Some(edge);
            
            self.edge_count.fetch_add(1, Ordering::SeqCst);
        }
        
        self.adjacency[from_idx as usize][to_idx as usize] = Some(edge_idx);
        self.last_update_ns.store(now_ns, Ordering::SeqCst);
        
        true
    }

    /// Get edge between two symbols - O(1)
    #[inline]
    pub fn get_edge(&self, from: u8, to: u8) -> Option<&Edge> {
        let edge_idx = self.adjacency[from as usize][to as usize]?;
        self.edges[edge_idx].as_ref()
    }

    /// Get all outgoing edges from a symbol
    #[inline]
    pub fn get_outgoing_edges(&self, from: u8) -> Vec<&Edge> {
        let mut edges = Vec::with_capacity(self.symbol_count as usize);
        
        for to in 0..self.symbol_count {
            if from == to {
                continue;
            }
            
            if let Some(edge_idx) = self.adjacency[from as usize][to as usize] {
                if let Some(ref edge) = self.edges[edge_idx] {
                    if edge.active {
                        edges.push(edge);
                    }
                }
            }
        }
        
        edges
    }

    /// Find all triangular arbitrage cycles starting from a given symbol
    /// Returns vector of (symbol_a, symbol_b, symbol_c, profit_bps)
    #[inline]
    pub fn find_triangles_from(&self, start: u8) -> Vec<(u8, u8, u8, f64)> {
        let mut triangles = Vec::new();
        
        for mid in 0..self.symbol_count {
            if start == mid {
                continue;
            }
            
            // Check edge start -> mid
            let edge1 = match self.get_edge(start, mid) {
                Some(e) => e,
                None => continue,
            };
            
            for end in 0..self.symbol_count {
                if end == start || end == mid {
                    continue;
                }
                
                // Check edge mid -> end
                let edge2 = match self.get_edge(mid, end) {
                    Some(e) => e,
                    None => continue,
                };
                
                // Check edge end -> start (completing the triangle)
                let edge3 = match self.get_edge(end, start) {
                    Some(e) => e,
                    None => continue,
                };
                
                // Calculate total log return
                // Negative sum of log rates = positive means profitable
                let total_log_return = -(edge1.log_rate_neg + edge2.log_rate_neg + edge3.log_rate_neg);
                
                // Convert to profit percentage
                let profit_factor = total_log_return.exp();
                let profit_bps = (profit_factor - 1.0) * 10000.0;
                
                if profit_bps > 0.0 {
                    triangles.push((start, mid, end, profit_bps));
                }
            }
        }
        
        triangles
    }

    /// Find all triangular arbitrage opportunities in the graph
    #[inline]
    pub fn find_all_triangles(&self) -> Vec<(u8, u8, u8, f64)> {
        let mut all_triangles = Vec::new();
        
        for start in 0..self.symbol_count {
            all_triangles.extend(self.find_triangles_from(start));
        }
        
        // Sort by profit (highest first)
        all_triangles.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        
        all_triangles
    }

    /// Get best triangular arbitrage opportunity
    #[inline]
    pub fn get_best_triangle(&self) -> Option<(u8, u8, u8, f64)> {
        self.find_all_triangles().into_iter().next()
    }

    /// Check if a specific triangle is profitable
    #[inline]
    pub fn check_triangle_profitability(&self, a: u8, b: u8, c: u8) -> Option<f64> {
        let edge_ab = self.get_edge(a, b)?;
        let edge_bc = self.get_edge(b, c)?;
        let edge_ca = self.get_edge(c, a)?;
        
        let total_log_return = -(edge_ab.log_rate_neg + edge_bc.log_rate_neg + edge_ca.log_rate_neg);
        let profit_factor = total_log_return.exp();
        let profit_bps = (profit_factor - 1.0) * 10000.0;
        
        Some(profit_bps)
    }

    /// Get current symbol count
    #[inline]
    pub const fn symbol_count(&self) -> u8 {
        self.symbol_count
    }

    /// Get current edge count
    #[inline]
    pub fn edge_count(&self) -> u64 {
        self.edge_count.load(Ordering::Acquire)
    }

    /// Get last update timestamp
    #[inline]
    pub fn last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Acquire)
    }

    /// Reset the graph (clear all edges)
    #[inline]
    pub fn reset(&mut self) {
        self.edges = [None; MAX_EDGES];
        self.adjacency = [[None; MAX_SYMBOLS]; MAX_SYMBOLS];
        self.edge_count.store(0, Ordering::Release);
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_update_ns.store(now_ns, Ordering::Release);
    }
}

impl Default for TriangularGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache-line aligned graph for multi-threaded access
#[repr(align(64))]
pub struct AlignedTriangularGraph {
    inner: TriangularGraph,
    _padding: [u8; 64],
}

impl AlignedTriangularGraph {
    pub fn new() -> Self {
        Self {
            inner: TriangularGraph::new(),
            _padding: [0; 64],
        }
    }

    #[inline]
    pub fn update_edge(&mut self, from: &str, to: &str, rate: f64, fee_bps: u32) -> bool {
        self.inner.update_edge(from, to, rate, fee_bps)
    }

    #[inline]
    pub fn get_best_triangle(&self) -> Option<(u8, u8, u8, f64)> {
        self.inner.get_best_triangle()
    }

    #[inline]
    pub fn find_all_triangles(&self) -> Vec<(u8, u8, u8, f64)> {
        self.inner.find_all_triangles()
    }
}

impl Default for AlignedTriangularGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = TriangularGraph::new();
        assert_eq!(graph.symbol_count(), 4); // BTC, ETH, SOL, USDT
    }

    #[test]
    fn test_edge_update() {
        let mut graph = TriangularGraph::new();
        
        // Update BTC/USDT edge
        let success = graph.update_edge("BTC", "USDT", 50000.0, 10);
        assert!(success);
        
        let btc_idx = graph.get_symbol_index("BTC").unwrap();
        let usdt_idx = graph.get_symbol_index("USDT").unwrap();
        
        let edge = graph.get_edge(btc_idx, usdt_idx).unwrap();
        assert!((edge.rate - 50000.0).abs() < 0.001);
    }

    #[test]
    fn test_triangle_detection() {
        let mut graph = TriangularGraph::new();
        
        // Set up a profitable triangle: BTC -> ETH -> USDT -> BTC
        // BTC/ETH = 0.05 (1 BTC = 0.05 ETH... wait, that's wrong)
        // Let's use realistic rates:
        // BTC/USDT = 50000
        // ETH/USDT = 3000
        // BTC/ETH = 16.67 (1 BTC = 16.67 ETH)
        
        graph.update_edge("BTC", "USDT", 50000.0, 10);
        graph.update_edge("USDT", "BTC", 1.0 / 50000.0, 10);
        graph.update_edge("ETH", "USDT", 3000.0, 10);
        graph.update_edge("USDT", "ETH", 1.0 / 3000.0, 10);
        graph.update_edge("BTC", "ETH", 16.67, 10);
        graph.update_edge("ETH", "BTC", 1.0 / 16.67, 10);
        
        let triangles = graph.find_all_triangles();
        
        // Should detect triangles (may or may not be profitable depending on fees)
        println!("Found {} triangles", triangles.len());
    }
}
