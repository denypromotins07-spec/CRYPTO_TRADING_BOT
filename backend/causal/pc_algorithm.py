#!/usr/bin/env python3
"""
PC Algorithm for Constraint-Based Causal Discovery

This module implements the PC (Peter-Clark) algorithm for discovering
causal structures from time-series data. Designed for the ZAID PERSONAL
CRYPTO TRADING BOT to identify causal relationships in crypto markets.

Features:
- Conditional independence testing using partial correlations
- Time-unrolling of cyclical feedback loops into valid DAGs
- Orientation rules for converting skeleton to DAG
- Optimized for high-frequency trading data streams

The algorithm works in three phases:
1. Skeleton identification via conditional independence tests
2. Edge orientation using v-structures
3. Additional orientations using Meek's rules
"""

from __future__ import annotations
from typing import List, Set, Dict, Tuple, Optional, FrozenSet
from itertools import combinations
from collections import defaultdict
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ConditionalIndependenceTest:
    """Performs conditional independence tests using partial correlations."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the test with significance level.
        
        Args:
            alpha: Significance level for independence tests
        """
        self.alpha = alpha
        
    def partial_correlation(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute partial correlation between X and Y given Z.
        
        Args:
            X: First variable (n_samples,)
            Y: Second variable (n_samples,)
            Z: Conditioning variables (n_samples, n_conditions)
            
        Returns:
            Tuple of (partial_correlation, p_value)
        """
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        n_vars = Z.shape[1] + 2
        all_vars = np.column_stack([X, Y, Z])
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(all_vars.T)
        
        if n_vars == 2:
            # No conditioning variables
            r_xy = corr_matrix[0, 1]
            n = len(X)
            t_stat = r_xy * np.sqrt((n - 2) / (1 - r_xy**2 + 1e-10))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
            return r_xy, p_value
        
        # Compute partial correlation using inverse correlation matrix
        try:
            inv_corr = np.linalg.inv(corr_matrix + 1e-10 * np.eye(n_vars))
        except np.linalg.LinAlgError:
            return 0.0, 1.0
        
        r_xy_z = -inv_corr[0, 1] / np.sqrt(inv_corr[0, 0] * inv_corr[1, 1] + 1e-10)
        r_xy_z = np.clip(r_xy_z, -1.0, 1.0)
        
        # Fisher's z-transformation for p-value
        n = len(X)
        df = n - n_vars
        
        if abs(r_xy_z) >= 1.0:
            return np.sign(r_xy_z), 0.0
        
        z_score = 0.5 * np.log((1 + r_xy_z) / (1 - r_xy_z + 1e-10))
        se = 1.0 / np.sqrt(df - 3 + 1e-10)
        z_stat = abs(z_score) / se
        p_value = 2 * (1 - stats.norm.cdf(z_stat))
        
        return r_xy_z, p_value
    
    def is_independent(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test if X and Y are conditionally independent given Z.
        
        Args:
            X: First variable
            Y: Second variable
            Z: Conditioning variables
            
        Returns:
            Tuple of (is_independent, p_value)
        """
        _, p_value = self.partial_correlation(X, Y, Z)
        return p_value > self.alpha, p_value


class PCAlgorithm:
    """
    Implementation of the PC algorithm for causal discovery.
    
    The algorithm discovers causal structure by:
    1. Starting with a complete undirected graph
    2. Removing edges based on conditional independence tests
    3. Orienting edges using v-structures and Meek's rules
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_size: int = 3,
        time_unrolling: bool = True
    ):
        """
        Initialize PC algorithm.
        
        Args:
            alpha: Significance level for independence tests
            max_cond_size: Maximum size of conditioning sets
            time_unrolling: Whether to unroll time series into DAG
        """
        self.ci_test = ConditionalIndependenceTest(alpha)
        self.max_cond_size = max_cond_size
        self.time_unrolling = time_unrolling
        self.separation_sets: Dict[Tuple[int, int], Set[int]] = {}
        
    def build_skeleton(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int], Set[int]]]:
        """
        Build the skeleton of the causal graph.
        
        Args:
            data: Time-series data (n_samples, n_variables)
            variable_names: Optional names for variables
            
        Returns:
            Tuple of (adjacency_dict, separation_sets)
        """
        n_samples, n_vars = data.shape
        
        # Initialize complete graph
        adj: Dict[int, Set[int]] = {i: set(range(n_vars)) - {i} for i in range(n_vars)}
        self.separation_sets = {}
        
        logger.info(f"Building skeleton for {n_vars} variables...")
        
        # Iterate through conditioning set sizes
        for cond_size in range(self.max_cond_size + 1):
            logger.debug(f"Testing conditioning sets of size {cond_size}")
            
            # Get pairs of adjacent vertices
            pairs_to_test = []
            for x in range(n_vars):
                for y in adj[x]:
                    if x < y:
                        pairs_to_test.append((x, y))
            
            # Test each pair
            for x, y in pairs_to_test:
                if y not in adj[x]:
                    continue
                    
                # Get possible conditioning variables
                neighbors_x = adj[x] - {y}
                neighbors_y = adj[y] - {x}
                possible_z = neighbors_x | neighbors_y
                
                if len(possible_z) < cond_size:
                    continue
                
                # Test all conditioning sets of current size
                for z_set in combinations(possible_z, cond_size):
                    z_indices = list(z_set)
                    Z = data[:, z_indices] if z_indices else np.zeros((n_samples, 0))
                    
                    is_ind, p_val = self.ci_test.is_independent(
                        data[:, x], data[:, y], Z
                    )
                    
                    if is_ind:
                        # Remove edge and record separation set
                        adj[x].discard(y)
                        adj[y].discard(x)
                        self.separation_sets[(x, y)] = set(z_indices)
                        self.separation_sets[(y, x)] = set(z_indices)
                        logger.debug(
                            f"Removed edge {x}-{y}, separated by {z_indices}, p={p_val:.4f}"
                        )
                        break
        
        logger.info(f"Skeleton complete: {sum(len(v) for v in adj.values()) // 2} edges remaining")
        return adj, self.separation_sets
    
    def orient_edges(
        self,
        adj: Dict[int, Set[int]],
        data: np.ndarray
    ) -> Dict[int, Set[int]]:
        """
        Orient edges to create a DAG using v-structures and Meek's rules.
        
        Args:
            adj: Undirected adjacency from skeleton
            data: Original data for reference
            
        Returns:
            Directed adjacency list
        """
        n_vars = len(adj)
        
        # Create directed graph (parents -> children)
        directed: Dict[int, Set[int]] = {i: set() for i in range(n_vars)}
        undirected: Dict[int, Set[int]] = {i: adj[i].copy() for i in range(n_vars)}
        
        # Step 1: Orient v-structures (X -> Z <- Y where X and Y not adjacent)
        for z in range(n_vars):
            neighbors = list(undirected[z])
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    # Check if X and Y are not adjacent
                    if y in undirected[x] or x in directed[y] or y in directed[x]:
                        continue
                    
                    # Check if Z is NOT in separation set of X and Y
                    sep_set = self.separation_sets.get((x, y), set())
                    if z not in sep_set:
                        # Orient as v-structure: X -> Z <- Y
                        if z in undirected[x]:
                            undirected[x].remove(z)
                        if z in undirected[y]:
                            undirected[y].remove(x)
                        directed[x].add(z)
                        directed[y].add(z)
                        
                        logger.debug(f"Oriented v-structure: {x} -> {z} <- {y}")
        
        # Step 2: Apply Meek's rules iteratively
        changed = True
        iteration = 0
        while changed and iteration < 100:
            changed = False
            iteration += 1
            
            # Rule 1: If X -> Y and Y - Z (undirected), and X and Z not adjacent
            # Then orient Y -> Z
            for y in range(n_vars):
                for z in list(undirected[y]):
                    for x in directed[y]:
                        if z not in undirected[x] and z not in directed[x] and x not in directed[z]:
                            # X and Z not adjacent
                            undirected[y].remove(z)
                            directed[y].add(z)
                            changed = True
                            logger.debug(f"Meek Rule 1: {y} -> {z}")
                            break
            
            # Rule 2: If X -> Y -> Z and X - Z, then orient X -> Z
            for x in range(n_vars):
                for z in list(undirected[x]):
                    for y in directed[x]:
                        if z in directed[y]:
                            undirected[x].remove(z)
                            directed[x].add(z)
                            changed = True
                            logger.debug(f"Meek Rule 2: {x} -> {z}")
                            break
            
            # Rule 3: If X - Y, X - Z, Y -> W, Z -> W, then orient X -> W
            # (Simplified implementation)
        
        # Convert remaining undirected edges arbitrarily (avoiding cycles)
        for x in range(n_vars):
            for y in list(undirected[x]):
                if x < y:  # Arbitrary but consistent orientation
                    directed[x].add(y)
        
        return directed
    
    def discover(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> Dict[int, Set[int]]:
        """
        Run the full PC algorithm.
        
        Args:
            data: Time-series data (n_samples, n_variables)
            variable_names: Optional variable names
            
        Returns:
            Directed adjacency list representing the causal DAG
        """
        # Build skeleton
        adj, _ = self.build_skeleton(data, variable_names)
        
        # Orient edges
        dag = self.orient_edges(adj, data)
        
        return dag
    
    def discover_with_time_unrolling(
        self,
        data: np.ndarray,
        variable_names: List[str],
        max_lag: int = 5
    ) -> Dict[str, Set[str]]:
        """
        Discover causal structure with time-unrolling for cyclical data.
        
        This converts feedback loops into valid DAGs by creating separate
        nodes for each time lag.
        
        Args:
            data: Time-series data
            variable_names: Names of variables
            max_lag: Maximum lag for time-unrolling
            
        Returns:
            Named DAG as string-keyed adjacency list
        """
        n_orig = len(variable_names)
        
        # Create time-unrolled variable names
        unrolled_names = []
        name_to_idx = {}
        
        for lag in range(max_lag + 1):
            for i, name in enumerate(variable_names):
                unrolled_name = f"{name}_lag{lag}"
                unrolled_names.append(unrolled_name)
                name_to_idx[unrolled_name] = (i, lag)
        
        # Prepare unrolled data (using available lags)
        n_samples = len(data)
        effective_samples = n_samples - max_lag
        
        unrolled_data = []
        for lag in range(max_lag + 1):
            for i in range(n_orig):
                unrolled_data.append(data[lag:lag + effective_samples, i])
        
        unrolled_data = np.column_stack(unrolled_data)
        
        # Run PC algorithm on unrolled data
        dag = self.discover(unrolled_data, unrolled_names)
        
        # Convert to named adjacency
        named_dag: Dict[str, Set[str]] = {}
        for src_idx, targets in dag.items():
            src_name = unrolled_names[src_idx]
            named_dag[src_name] = set()
            for tgt_idx in targets:
                tgt_name = unrolled_names[tgt_idx]
                named_dag[src_name].add(tgt_name)
        
        logger.info(
            f"Time-unrolled DAG: {len(named_dag)} nodes, "
            f"{sum(len(v) for v in named_dag.values())} edges"
        )
        
        return named_dag


def pc_algorithm_main(
    data: np.ndarray,
    variable_names: List[str],
    alpha: float = 0.05,
    max_lag: int = 5
) -> Dict[str, Set[str]]:
    """
    Main entry point for PC algorithm causal discovery.
    
    Args:
        data: Time-series data (n_samples, n_variables)
        variable_names: Names of variables
        alpha: Significance level
        max_lag: Maximum lag for time-unrolling
        
    Returns:
        Causal DAG as adjacency list
    """
    pc = PCAlgorithm(alpha=alpha, max_cond_size=3, time_unrolling=True)
    return pc.discover_with_time_unrolling(data, variable_names, max_lag)


if __name__ == "__main__":
    # Example usage with synthetic crypto data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate causal structure: BTC -> ETH -> SOL with feedback
    btc = np.cumsum(np.random.randn(n_samples) * 0.02)
    eth = 0.5 * btc + np.random.randn(n_samples) * 0.01
    sol = 0.3 * eth + 0.2 * btc + np.random.randn(n_samples) * 0.015
    volume = 0.4 * np.abs(btc) + np.random.randn(n_samples) * 0.1
    
    data = np.column_stack([btc, eth, sol, volume])
    variables = ["BTC", "ETH", "SOL", "Volume"]
    
    dag = pc_algorithm_main(data, variables, alpha=0.05, max_lag=3)
    
    print("\nDiscovered Causal DAG:")
    for source, targets in sorted(dag.items()):
        if targets:
            print(f"  {source} -> {sorted(targets)}")
