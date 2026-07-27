#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
High-Frequency Feature Stores, MLOps, and Concept Drift
File: backend/features/vector_db.py
Chapter 1: Real-Time Feature Store and Low-Latency Vector Retrieval

Builds a lightweight HNSW (Hierarchical Navigable Small World) index for 
nearest-neighbor lookups of high-dimensional SMC and order flow embeddings.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
Uses NumPy C-extensions for performance-critical operations.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import threading
import math


@dataclass
class HNSWNode:
    """A node in the HNSW graph representing a feature vector."""
    id: int
    vector: np.ndarray
    level: int
    edges: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    
    def __post_init__(self):
        # Ensure vector is contiguous for cache efficiency
        self.vector = np.ascontiguousarray(self.vector, dtype=np.float64)


@dataclass
class SearchResult:
    """Result from a nearest neighbor search."""
    node_id: int
    distance: float
    metadata: Optional[Dict[str, Any]] = None


class HNSWIndex:
    """
    Lightweight HNSW index for high-dimensional feature vectors.
    
    Implements the Hierarchical Navigable Small World algorithm for 
    approximate nearest neighbor search with O(log N) complexity.
    
    Optimized for:
    - High-dimensional SMC (Smart Money Concepts) embeddings
    - Order flow feature vectors
    - Sub-millisecond query latency
    - 8GB RAM constraint on AMD Ryzen AI 5
    """
    
    def __init__(
        self,
        dimension: int,
        max_connections: int = 16,
        max_level_multiplier: float = 1.0 / math.log(2),
        ef_construction: int = 200,
        memory_budget_mb: int = 512
    ):
        """
        Initialize the HNSW index.
        
        Args:
            dimension: Dimensionality of feature vectors
            max_connections: Maximum connections per node (M parameter)
            max_level_multiplier: Controls level distribution
            ef_construction: Size of dynamic candidate list during construction
            memory_budget_mb: Memory budget in MB for the index
        """
        self.dimension = dimension
        self.max_connections = max_connections
        self.max_connections_level_0 = max_connections * 2
        self.max_level_multiplier = max_level_multiplier
        self.ef_construction = ef_construction
        self.memory_budget_bytes = memory_budget_mb * 1024 * 1024
        
        # Node storage
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.node_count = 0
        
        # Metadata storage
        self.metadata: Dict[int, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_insertions': 0,
            'total_queries': 0,
            'avg_query_latency_ns': 0.0,
            'memory_used_bytes': 0
        }
        
        # Estimate memory per node
        self._bytes_per_node = (
            dimension * 8 +  # vector data
            64 +  # node overhead
            max_connections * 8 * 2  # edge lists
        )
    
    def _random_level(self) -> int:
        """Generate a random level for a new node using exponential distribution."""
        level = 0
        while np.random.random() < self.max_level_multiplier and level < 32:
            level += 1
        return level
    
    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute squared Euclidean distance (faster than sqrt for comparisons)."""
        diff = v1 - v2
        return float(np.dot(diff, diff))
    
    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine distance between vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1.0 - float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _search_layer(
        self,
        query: np.ndarray,
        entry_point: int,
        ef: int,
        level: int
    ) -> List[Tuple[float, int]]:
        """
        Search within a single layer of the HNSW graph.
        
        Uses a priority queue to maintain the closest candidates.
        """
        visited = set()
        candidates = []
        closest = []
        
        # Initialize with entry point
        entry_dist = self._euclidean_distance(query, self.nodes[entry_point].vector)
        heapq.heappush(candidates, (-entry_dist, entry_point))
        heapq.heappush(closest, (entry_dist, entry_point))
        visited.add(entry_point)
        
        while candidates:
            neg_dist, current = heapq.heappop(candidates)
            current_dist = -neg_dist
            
            # Check if we can stop exploring
            if closest and current_dist > -closest[0][0]:
                continue
            
            # Explore neighbors
            node = self.nodes[current]
            neighbors = node.edges.get(level, [])
            
            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                
                neighbor_dist = self._euclidean_distance(
                    query, 
                    self.nodes[neighbor_id].vector
                )
                
                if len(closest) < ef or neighbor_dist < -closest[0][0]:
                    heapq.heappush(candidates, (-neighbor_dist, neighbor_id))
                    heapq.heappush(closest, (neighbor_dist, neighbor_id))
                    
                    if len(closest) > ef:
                        heapq.heappop(closest)
        
        return sorted(closest, key=lambda x: x[0])
    
    def insert(
        self,
        vector: np.ndarray,
        node_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Insert a new vector into the HNSW index.
        
        Args:
            vector: Feature vector to insert
            node_id: Optional explicit node ID (auto-generated if None)
            metadata: Optional metadata dictionary
            
        Returns:
            The node ID of the inserted vector
        """
        with self._lock:
            # Validate input
            vector = np.asarray(vector, dtype=np.float64).flatten()
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} doesn't match "
                    f"index dimension {self.dimension}"
                )
            
            # Check memory budget
            estimated_memory = (self.node_count + 1) * self._bytes_per_node
            if estimated_memory > self.memory_budget_bytes:
                self._evict_oldest()
            
            # Generate node ID
            if node_id is None:
                node_id = self.node_count
            self.node_count = max(self.node_count, node_id + 1)
            
            # Create new node
            level = self._random_level()
            node = HNSWNode(
                id=node_id,
                vector=vector,
                level=level
            )
            self.nodes[node_id] = node
            
            # Store metadata
            if metadata:
                self.metadata[node_id] = metadata
            
            # Find insertion point
            if self.entry_point is None:
                self.entry_point = node_id
            else:
                self._insert_with_links(node, level)
            
            self.stats['total_insertions'] += 1
            self.stats['memory_used_bytes'] = len(self.nodes) * self._bytes_per_node
            
            return node_id
    
    def _insert_with_links(self, new_node: HNSWNode, level: int) -> None:
        """Insert a node and create appropriate links."""
        current_entry = self.entry_point
        current_level = self.nodes[current_entry].level if current_entry else 0
        
        # Descend through levels
        for lvl in range(current_level, level, -1):
            candidates = self._search_layer(
                new_node.vector,
                current_entry,
                ef=1,
                level=lvl
            )
            if candidates:
                current_entry = candidates[0][1]
        
        # Insert at each level from min(current_level, level) to 0
        for lvl in range(min(current_level, level), -1, -1):
            candidates = self._search_layer(
                new_node.vector,
                current_entry,
                ef=self.ef_construction,
                level=lvl
            )
            
            # Select neighbors using heuristic
            neighbors = self._select_neighbors_heuristic(
                new_node,
                candidates,
                lvl,
                self.max_connections if lvl > 0 else self.max_connections_level_0
            )
            
            # Add bidirectional links
            new_node.edges[lvl] = [n[1] for n in neighbors]
            for _, neighbor_id in neighbors:
                neighbor_node = self.nodes[neighbor_id]
                neighbor_node.edges[lvl].append(new_node.id)
                
                # Prune if necessary
                max_conn = self.max_connections if lvl > 0 else self.max_connections_level_0
                if len(neighbor_node.edges[lvl]) > max_conn:
                    self._prune_edges(neighbor_node, lvl, max_conn)
        
        # Update entry point if new node has higher level
        if level > current_level:
            self.entry_point = new_node.id
    
    def _select_neighbors_heuristic(
        self,
        node: HNSWNode,
        candidates: List[Tuple[float, int]],
        level: int,
        max_neighbors: int
    ) -> List[Tuple[float, int]]:
        """Select neighbors using the heuristic from the HNSW paper."""
        selected = []
        candidates_sorted = sorted(candidates, key=lambda x: x[0])
        
        for dist, cand_id in candidates_sorted:
            if len(selected) >= max_neighbors:
                break
            
            # Check if adding this candidate would disconnect the graph
            should_add = True
            for _, sel_id in selected:
                sel_node = self.nodes[sel_id]
                cand_node = self.nodes[cand_id]
                
                # Triangle inequality check
                dist_sel_cand = self._euclidean_distance(
                    sel_node.vector,
                    cand_node.vector
                )
                if dist_sel_cand < dist:
                    should_add = False
                    break
            
            if should_add:
                selected.append((dist, cand_id))
        
        return selected[:max_neighbors]
    
    def _prune_edges(
        self,
        node: HNSWNode,
        level: int,
        max_edges: int
    ) -> None:
        """Prune edges to maintain maximum connection limit."""
        if len(node.edges[level]) <= max_edges:
            return
        
        # Keep the closest neighbors
        distances = []
        for neighbor_id in node.edges[level]:
            dist = self._euclidean_distance(
                node.vector,
                self.nodes[neighbor_id].vector
            )
            distances.append((dist, neighbor_id))
        
        distances.sort(key=lambda x: x[0])
        node.edges[level] = [nid for _, nid in distances[:max_edges]]
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 100
    ) -> List[SearchResult]:
        """
        Find k nearest neighbors for a query vector.
        
        Args:
            query: Query feature vector
            k: Number of neighbors to return
            ef: Size of dynamic candidate list during search
            
        Returns:
            List of SearchResult objects sorted by distance
        """
        import time
        start_ns = time.perf_counter_ns()
        
        with self._lock:
            if self.entry_point is None or not self.nodes:
                return []
            
            query = np.asarray(query, dtype=np.float64).flatten()
            
            # Search from top level to level 0
            current_entry = self.entry_point
            current_level = self.nodes[current_entry].level
            
            for lvl in range(current_level, 0, -1):
                candidates = self._search_layer(
                    query,
                    current_entry,
                    ef=1,
                    level=lvl
                )
                if candidates:
                    current_entry = candidates[0][1]
            
            # Final search at level 0
            final_candidates = self._search_layer(
                query,
                current_entry,
                ef=max(ef, k),
                level=0
            )
            
            # Get top k results
            results = []
            for dist, node_id in final_candidates[:k]:
                results.append(SearchResult(
                    node_id=node_id,
                    distance=math.sqrt(dist),  # Convert to actual Euclidean distance
                    metadata=self.metadata.get(node_id)
                ))
            
            # Update statistics
            elapsed_ns = time.perf_counter_ns() - start_ns
            self.stats['total_queries'] += 1
            # Running average
            n = self.stats['total_queries']
            self.stats['avg_query_latency_ns'] = (
                (self.stats['avg_query_latency_ns'] * (n - 1) + elapsed_ns) / n
            )
            
            return results
    
    def _evict_oldest(self) -> None:
        """Evict oldest nodes when memory budget is exceeded."""
        if not self.nodes:
            return
        
        # Simple eviction: remove first 10% of nodes
        evict_count = max(1, len(self.nodes) // 10)
        keys_to_remove = list(self.nodes.keys())[:evict_count]
        
        for key in keys_to_remove:
            del self.nodes[key]
            self.metadata.pop(key, None)
        
        # Rebuild entry point if necessary
        if self.entry_point not in self.nodes and self.nodes:
            self.entry_point = min(self.nodes.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            **self.stats,
            'node_count': len(self.nodes),
            'dimension': self.dimension,
            'max_connections': self.max_connections,
            'entry_point': self.entry_point
        }
    
    def save(self, filepath: str) -> None:
        """Save the index to disk."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'nodes': self.nodes,
                'metadata': self.metadata,
                'entry_point': self.entry_point,
                'stats': self.stats
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'HNSWIndex':
        """Load an index from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new instance
        if data['nodes']:
            first_vector = next(iter(data['nodes'].values())).vector
            dimension = len(first_vector)
        else:
            dimension = 0
        
        index = cls(dimension=dimension)
        index.nodes = data['nodes']
        index.metadata = data['metadata']
        index.entry_point = data['entry_point']
        index.stats = data['stats']
        index.node_count = len(index.nodes)
        
        return index


# Convenience function for creating SMC-specific index
def create_smc_index(
    smc_features: int = 64,
    order_flow_features: int = 32,
    memory_mb: int = 256
) -> HNSWIndex:
    """
    Create an HNSW index optimized for SMC and order flow features.
    
    Args:
        smc_features: Number of Smart Money Concepts features
        order_flow_features: Number of order flow features
        memory_mb: Memory budget in MB
        
    Returns:
        Configured HNSWIndex instance
    """
    total_dimension = smc_features + order_flow_features
    return HNSWIndex(
        dimension=total_dimension,
        max_connections=24,  # Higher for better accuracy in financial data
        ef_construction=400,  # More thorough construction
        memory_budget_mb=memory_mb
    )


if __name__ == '__main__':
    # Test the HNSW implementation
    print("Testing HNSW Index for ZAID Trading Bot...")
    
    # Create index for 96-dimensional features (64 SMC + 32 order flow)
    index = create_smc_index(memory_mb=128)
    
    # Insert sample vectors
    np.random.seed(42)
    for i in range(1000):
        vector = np.random.randn(96)
        metadata = {'asset': 'BTC', 'timestamp': i, 'regime': 'trending'}
        index.insert(vector, node_id=i, metadata=metadata)
    
    # Search for nearest neighbors
    query = np.random.randn(96)
    results = index.search(query, k=5)
    
    print(f"Index stats: {index.get_stats()}")
    print(f"Found {len(results)} nearest neighbors")
    for r in results:
        print(f"  Node {r.node_id}: distance={r.distance:.4f}, metadata={r.metadata}")
    
    print("\nHNSW Index test completed successfully!")
