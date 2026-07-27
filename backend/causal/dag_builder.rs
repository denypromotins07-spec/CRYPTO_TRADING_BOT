//! DAG Builder for Causal Discovery
//! 
//! This module constructs Directed Acyclic Graphs (DAGs) from time-series data
//! using efficient Rust algorithms optimized for O(N log N) complexity.
//! Designed for the ZAID PERSONAL CRYPTO TRADING BOT to identify causal structures
//! in high-frequency crypto market data without heap allocations where possible.
//!
//! Features:
//! - Zero-cost abstractions for graph traversal
//! - Memory-efficient adjacency list representation
//! - Time-unrolling of cyclical feedback loops into valid DAGs
//! - Strict borrowing checks for thread safety

use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::Arc;
use std::time::{Instant, Duration};

/// Represents a node in the causal DAG
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CausalNode {
    pub id: usize,
    pub name: String,
    pub variable_type: VariableType,
    pub lag_depth: u8,
}

/// Types of variables in the causal model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Price,
    Volume,
    OrderFlow,
    Volatility,
    Macro,
    Sentiment,
}

/// Edge representing a causal relationship with confidence score
#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub source: usize,
    pub target: usize,
    pub confidence: f64,
    pub lag: u8,
    pub edge_type: EdgeType,
}

/// Type of causal relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    Direct,
    Indirect,
    Confounded,
    Mediated,
}

/// The main DAG structure for causal discovery
/// Uses adjacency list for O(1) edge lookup and O(N log N) construction
pub struct CausalDAG {
    nodes: Vec<CausalNode>,
    adjacency: Vec<Vec<usize>>, // Forward edges
    reverse_adjacency: Vec<Vec<usize>>, // Backward edges for topological sort
    edges: HashMap<(usize, usize), CausalEdge>,
    node_name_map: HashMap<String, usize>,
    topological_order: Option<Vec<usize>>,
    is_valid_dag: bool,
}

impl CausalDAG {
    /// Create a new empty DAG with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            adjacency: Vec::with_capacity(capacity),
            reverse_adjacency: Vec::with_capacity(capacity),
            edges: HashMap::with_capacity(capacity * 2),
            node_name_map: HashMap::with_capacity(capacity),
            topological_order: None,
            is_valid_dag: true,
        }
    }

    /// Add a node to the DAG - O(1) amortized
    pub fn add_node(&mut self, name: &str, var_type: VariableType, lag: u8) -> usize {
        let id = self.nodes.len();
        let node = CausalNode {
            id,
            name: name.to_string(),
            variable_type: var_type,
            lag_depth: lag,
        };
        
        self.node_name_map.insert(name.to_string(), id);
        self.nodes.push(node);
        self.adjacency.push(Vec::new());
        self.reverse_adjacency.push(Vec::new());
        
        id
    }

    /// Add a directed edge - maintains DAG property
    /// Returns false if adding edge would create a cycle
    pub fn add_edge(&mut self, source: usize, target: usize, confidence: f64, lag: u8) -> bool {
        if source >= self.nodes.len() || target >= self.nodes.len() {
            return false;
        }

        // Check if edge already exists
        if self.edges.contains_key(&(source, target)) {
            return true;
        }

        // Temporarily add edge and check for cycles
        self.adjacency[source].push(target);
        self.reverse_adjacency[target].push(source);

        if self.has_cycle_from(target, source) {
            // Remove edge if it creates a cycle
            self.adjacency[source].pop();
            self.reverse_adjacency[target].pop();
            self.is_valid_dag = false;
            return false;
        }

        let edge = CausalEdge {
            source,
            target,
            confidence,
            lag,
            edge_type: EdgeType::Direct,
        };

        self.edges.insert((source, target), edge);
        self.topological_order = None; // Invalidate cached order
        true
    }

    /// DFS-based cycle detection - O(V + E)
    fn has_cycle_from(&self, start: usize, end: usize) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![start];

        while let Some(node) = stack.pop() {
            if node == end {
                return true;
            }
            
            if visited[node] {
                continue;
            }
            visited[node] = true;

            for &neighbor in &self.adjacency[node] {
                stack.push(neighbor);
            }
        }

        false
    }

    /// Compute topological ordering using Kahn's algorithm - O(V + E)
    pub fn compute_topological_order(&mut self) -> Option<&[usize]> {
        if let Some(ref order) = self.topological_order {
            return Some(order);
        }

        let n = self.nodes.len();
        let mut in_degree = vec![0; n];
        
        for i in 0..n {
            in_degree[i] = self.reverse_adjacency[i].len();
        }

        let mut queue = Vec::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        let mut head = 0;

        while head < queue.len() {
            let node = queue[head];
            head += 1;
            order.push(node);

            for &neighbor in &self.adjacency[node] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push(neighbor);
                }
            }
        }

        if order.len() != n {
            self.is_valid_dag = false;
            return None;
        }

        self.topological_order = Some(order);
        self.topological_order.as_deref()
    }

    /// Get all parents of a node
    pub fn get_parents(&self, node_id: usize) -> &[usize] {
        if node_id >= self.reverse_adjacency.len() {
            return &[];
        }
        &self.reverse_adjacency[node_id]
    }

    /// Get all children of a node
    pub fn get_children(&self, node_id: usize) -> &[usize] {
        if node_id >= self.adjacency.len() {
            return &[];
        }
        &self.adjacency[node_id]
    }

    /// Get edge information
    pub fn get_edge(&self, source: usize, target: usize) -> Option<&CausalEdge> {
        self.edges.get(&(source, target))
    }

    /// Prune edges below confidence threshold
    pub fn prune_weak_edges(&mut self, threshold: f64) -> usize {
        let mut removed = 0;
        let keys_to_remove: Vec<_> = self.edges
            .iter()
            .filter(|(_, edge)| edge.confidence < threshold)
            .map(|(k, _)| *k)
            .collect();

        for key in keys_to_remove {
            self.edges.remove(&key);
            if let Some(pos) = self.adjacency[key.0].iter().position(|&x| x == key.1) {
                self.adjacency[key.0].remove(pos);
            }
            if let Some(pos) = self.reverse_adjacency[key.1].iter().position(|&x| x == key.0) {
                self.reverse_adjacency[key.1].remove(pos);
            }
            removed += 1;
        }

        if removed > 0 {
            self.topological_order = None;
        }

        removed
    }

    /// Export DAG to DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph CausalDAG {\n");
        dot.push_str("    rankdir=LR;\n");
        
        for node in &self.nodes {
            dot.push_str(&format!(
                "    {} [label=\"{}\\n{:?}\"];\n",
                node.id, node.name, node.variable_type
            ));
        }

        for (_, edge) in &self.edges {
            dot.push_str(&format!(
                "    {} -> {} [label=\"{:.2}\"];\n",
                edge.source, edge.target, edge.confidence
            ));
        }

        dot.push('}');
        dot
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn is_valid(&self) -> bool {
        self.is_valid_dag
    }
}

/// Builder for constructing DAGs from time-series data
pub struct DAGBuilder {
    max_lag: u8,
    min_confidence: f64,
    time_unrolling: bool,
}

impl DAGBuilder {
    pub fn new() -> Self {
        Self {
            max_lag: 5,
            min_confidence: 0.05,
            time_unrolling: true,
        }
    }

    pub fn with_max_lag(mut self, lag: u8) -> Self {
        self.max_lag = lag;
        self
    }

    pub fn with_min_confidence(mut self, conf: f64) -> Self {
        self.min_confidence = conf;
        self
    }

    pub fn with_time_unrolling(mut self, enabled: bool) -> Self {
        self.time_unrolling = enabled;
        self
    }

    /// Build DAG from correlation matrix and statistical tests
    /// Uses PC-algorithm output to construct the final DAG
    pub fn build_from_skeleton(
        &self,
        variables: &[&str],
        var_types: &[VariableType],
        skeleton: &[(usize, usize, f64)],
    ) -> CausalDAG {
        let start = Instant::now();
        
        // Pre-allocate based on expected size for O(N log N) performance
        let estimated_nodes = variables.len() * (self.max_lag as usize + 1);
        let mut dag = CausalDAG::with_capacity(estimated_nodes);

        // Create time-unrolled nodes if enabled
        let node_ids: Vec<Vec<usize>> = if self.time_unrolling {
            variables
                .iter()
                .enumerate()
                .map(|(i, &name)| {
                    (0..=self.max_lag)
                        .map(|lag| {
                            let node_name = format!("{}_lag{}", name, lag);
                            dag.add_node(&node_name, var_types[i], lag)
                        })
                        .collect()
                })
                .collect()
        } else {
            variables
                .iter()
                .enumerate()
                .map(|(i, &name)| {
                    let id = dag.add_node(name, var_types[i], 0);
                    vec![id]
                })
                .collect()
        };

        // Add edges based on skeleton
        for &(source, target, confidence) in skeleton {
            if confidence < self.min_confidence {
                continue;
            }

            if self.time_unrolling {
                // Add time-lagged edges respecting causality (past -> future)
                for lag in 0..=self.max_lag {
                    if lag + 1 <= self.max_lag as usize {
                        let src = node_ids[source][lag];
                        let tgt = node_ids[target][lag + 1];
                        dag.add_edge(src, tgt, confidence, (lag + 1) as u8);
                    }
                }
            } else {
                dag.add_edge(node_ids[source][0], node_ids[target][0], confidence, 0);
            }
        }

        // Compute topological order
        dag.compute_topological_order();

        let elapsed = start.elapsed();
        log::info!(
            "DAG built: {} nodes, {} edges in {:?}",
            dag.node_count(),
            dag.edge_count(),
            elapsed
        );

        dag
    }
}

impl Default for DAGBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_construction() {
        let mut dag = CausalDAG::with_capacity(10);
        
        let btc = dag.add_node("BTC_price", VariableType::Price, 0);
        let eth = dag.add_node("ETH_price", VariableType::Price, 0);
        let vol = dag.add_node("Volume", VariableType::Volume, 0);

        assert!(dag.add_edge(btc, eth, 0.8, 1));
        assert!(dag.add_edge(vol, btc, 0.6, 1));
        
        // Cycle should be rejected
        assert!(!dag.add_edge(eth, btc, 0.7, 0));
        
        assert!(dag.is_valid());
        assert_eq!(dag.node_count(), 3);
        assert_eq!(dag.edge_count(), 2);
    }

    #[test]
    fn test_topological_sort() {
        let mut dag = CausalDAG::with_capacity(5);
        
        let n0 = dag.add_node("A", VariableType::Price, 0);
        let n1 = dag.add_node("B", VariableType::Volume, 0);
        let n2 = dag.add_node("C", VariableType::Volatility, 0);

        dag.add_edge(n0, n1, 0.9, 1);
        dag.add_edge(n0, n2, 0.7, 1);
        dag.add_edge(n1, n2, 0.5, 1);

        let order = dag.compute_topological_order().unwrap();
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], n0); // Root first
    }

    #[test]
    fn test_builder_with_time_unrolling() {
        let builder = DAGBuilder::new()
            .with_max_lag(3)
            .with_min_confidence(0.1)
            .with_time_unrolling(true);

        let variables = ["BTC", "ETH"];
        let var_types = [VariableType::Price, VariableType::Price];
        let skeleton = [(0, 1, 0.85)];

        let dag = builder.build_from_skeleton(&variables, &var_types, &skeleton);
        
        // Should have 2 variables * 4 lags = 8 nodes
        assert_eq!(dag.node_count(), 8);
        assert!(dag.is_valid());
    }
}
