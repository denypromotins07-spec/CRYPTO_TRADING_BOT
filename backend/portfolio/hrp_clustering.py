#!/usr/bin/env python3
"""
Hierarchical Risk Parity (HRP) Portfolio Allocator

Uses hierarchical clustering to build diversified crypto portfolios.
Superior to traditional methods during correlation regime changes.
Implements dendrogram-based allocation with recursive bisection.

# Key Advantages:
- No matrix inversion required (handles singular covariance)
- Naturally groups correlated assets before allocation
- Robust to estimation error in covariance matrix
- Adapts dynamically to changing market regimes

# Algorithm Steps:
1. Compute distance matrix from correlation
2. Build hierarchical tree (dendrogram) using linkage
3. Quasi-diagonalize the covariance matrix
4. Recursively bisect and allocate capital
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform


@dataclass
class HRPResult:
    """Result of HRP optimization."""
    # Optimal portfolio weights
    weights: dict[str, float]
    # Linkage matrix used for clustering
    linkage_matrix: np.ndarray
    # Sorted asset order (quasi-diagonalized)
    sorted_assets: list[str]
    # Cophenetic correlation (measure of clustering quality)
    cophenetic_corr: float
    # Cluster assignments at each level
    cluster_structure: list[dict]


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio optimizer.
    
    Uses agglomerative clustering to group similar assets,
    then allocates capital recursively based on inverse variance.
    
    This approach is particularly effective for crypto portfolios
    where correlations change rapidly during market stress.
    
    # Attributes:
        asset_labels: List of asset tickers [BTC, SOL, ETH, USDT]
        covariance: Covariance matrix of returns
        correlation: Correlation matrix (computed from covariance)
    """
    
    ASSET_LABELS: list[str] = ["BTC", "SOL", "ETH", "USDT"]
    NUM_ASSETS: int = 4
    
    def __init__(self, covariance: np.ndarray) -> None:
        """
        Initialize HRP optimizer.
        
        Args:
            covariance: 4x4 covariance matrix of returns
        """
        if covariance.shape != (self.NUM_ASSETS, self.NUM_ASSETS):
            raise ValueError(f"Expected 4x4 covariance, got {covariance.shape}")
        
        self.covariance = np.asarray(covariance, dtype=np.float64)
        
        # Compute correlation matrix
        self.correlation = self._cov_to_corr(self.covariance)
        
        # Precompute volatilities
        self.volatilities = np.sqrt(np.diag(self.covariance))
        
        # Cache for linkage
        self._linkage: Optional[np.ndarray] = None
        self._sorted_indices: Optional[np.ndarray] = None
    
    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        std[std < 1e-10] = 1e-10  # Avoid division by zero
        outer_std = np.outer(std, std)
        return cov / outer_std
    
    @property
    def linkage_matrix(self) -> np.ndarray:
        """Lazy-computed linkage matrix."""
        if self._linkage is None:
            # Convert correlation to distance
            distance_matrix = np.sqrt(0.5 * (1 - self.correlation))
            
            # Extract upper triangle for linkage
            condensed_dist = squareform(distance_matrix, checks=False)
            
            # Use Ward's method for minimum variance clustering
            self._linkage = linkage(condensed_dist, method='ward')
        
        return self._linkage
    
    @property
    def sorted_indices(self) -> np.ndarray:
        """Indices that quasi-diagonalize the covariance matrix."""
        if self._sorted_indices is None:
            self._sorted_indices = self._quasi_diag(self.linkage_matrix)
        return self._sorted_indices
    
    def optimize(self) -> HRPResult:
        """
        Compute HRP optimal weights.
        
        Returns:
            HRPResult with weights and clustering information
        """
        # Get sorted order
        sorted_idx = self.sorted_indices
        
        # Recursive bisection to allocate weights
        weights_raw = self._recursive_bisection(np.ones(self.NUM_ASSETS), sorted_idx)
        
        # Ensure weights_raw is 1D array of correct length
        if weights_raw.ndim > 1:
            weights_raw = weights_raw.flatten()
        
        # Truncate or pad to correct length
        if len(weights_raw) != self.NUM_ASSETS:
            weights_raw = weights_raw[:self.NUM_ASSETS]
            if len(weights_raw) < self.NUM_ASSETS:
                weights_raw = np.pad(weights_raw, (0, self.NUM_ASSETS - len(weights_raw)))
        
        # Map back to original order
        weights = np.zeros(self.NUM_ASSETS)
        for i, idx in enumerate(sorted_idx):
            weights[idx] = weights_raw[i]
        
        # Normalize
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        
        # Compute cophenetic correlation
        coph_corr, _ = cophenet(self.linkage_matrix, squareform(
            np.sqrt(0.5 * (1 - self.correlation)), checks=False
        ))
        
        # Build cluster structure description
        cluster_structure = self._describe_clusters()
        
        # Create result
        weight_dict = dict(zip(self.ASSET_LABELS, weights.tolist()))
        sorted_assets = [self.ASSET_LABELS[i] for i in sorted_idx]
        
        return HRPResult(
            weights=weight_dict,
            linkage_matrix=self.linkage_matrix.copy(),
            sorted_assets=sorted_assets,
            cluster_structure=cluster_structure,
            cophenetic_corr=float(coph_corr),
        )
    
    def _quasi_diag(self, linkage: np.ndarray) -> np.ndarray:
        """
        Sort assets by hierarchical tree structure.
        
        Produces ordering that quasi-diagonalizes the covariance matrix,
        grouping similar assets together.
        """
        n = self.NUM_ASSETS
        
        # Start with leaf nodes
        sort_idx = [np.array([i]) for i in range(n)]
        
        # Traverse the tree and merge clusters
        for i in range(n - 1):
            left_cluster = int(linkage[i, 0])
            right_cluster = int(linkage[i, 1])
            
            # Get leaves in each cluster
            if left_cluster < n:
                left_leaves = np.array([left_cluster])
            else:
                left_leaves = sort_idx[left_cluster - n]
            
            if right_cluster < n:
                right_leaves = np.array([right_cluster])
            else:
                right_leaves = sort_idx[right_cluster - n]
            
            # Sort within cluster by variance (lower variance first)
            left_leaves = sorted(left_leaves, key=lambda x: self.covariance[x, x])
            right_leaves = sorted(right_leaves, key=lambda x: self.covariance[x, x])
            
            # Store merged cluster
            sort_idx.append(np.concatenate([left_leaves, right_leaves]))
        
        return sort_idx[-1]
    
    def _get_leaves(self, cluster_id: int, linkage: np.ndarray, n: int) -> np.ndarray:
        """Get all leaf nodes (original assets) in a cluster."""
        if cluster_id < n:
            return np.array([cluster_id])
        
        # Internal node - get children
        left_child = int(linkage[cluster_id - n, 0])
        right_child = int(linkage[cluster_id - n, 1])
        
        left_leaves = self._get_leaves(left_child, linkage, n)
        right_leaves = self._get_leaves(right_child, linkage, n)
        
        return np.concatenate([left_leaves, right_leaves])
    
    def _recursive_bisection(self, alpha: np.ndarray, sorted_idx: np.ndarray) -> np.ndarray:
        """
        Recursively allocate capital using bisection.
        
        At each step, splits cluster into two sub-clusters and
        allocates based on inverse variance weighting.
        """
        n = len(sorted_idx)
        
        if n == 1:
            return np.array([alpha])
        
        if n == 2:
            # Base case: allocate between two assets
            var_0 = self.covariance[sorted_idx[0], sorted_idx[0]]
            var_1 = self.covariance[sorted_idx[1], sorted_idx[1]]
            
            # Inverse variance weighting
            w_0 = 1.0 / var_0 if var_0 > 1e-10 else 1.0
            w_1 = 1.0 / var_1 if var_1 > 1e-10 else 1.0
            
            total = w_0 + w_1
            alpha_0 = alpha * w_0 / total
            alpha_1 = alpha * w_1 / total
            
            return np.array([alpha_0, alpha_1])
        
        # Split cluster in half
        mid = n // 2
        left_idx = sorted_idx[:mid]
        right_idx = sorted_idx[mid:]
        
        # Compute cluster variances
        left_var = self._cluster_variance(left_idx)
        right_var = self._cluster_variance(right_idx)
        
        # Allocate between clusters (inverse variance)
        if left_var + right_var > 1e-10:
            left_alpha = alpha * right_var / (left_var + right_var)
            right_alpha = alpha * left_var / (left_var + right_var)
        else:
            left_alpha = alpha * 0.5
            right_alpha = alpha * 0.5
        
        # Recurse
        left_weights = self._recursive_bisection(left_alpha, left_idx)
        right_weights = self._recursive_bisection(right_alpha, right_idx)
        
        return np.concatenate([left_weights, right_weights])
    
    def _cluster_variance(self, indices: np.ndarray) -> float:
        """Compute average variance of a cluster."""
        if len(indices) == 0:
            return 1.0
        
        vars_ = [self.covariance[i, i] for i in indices]
        return np.mean(vars_)
    
    def _describe_clusters(self) -> list[dict]:
        """Describe the hierarchical cluster structure."""
        n = self.NUM_ASSETS
        linkage = self.linkage_matrix
        
        structure = []
        for i in range(n - 1):
            left_id = int(linkage[i, 0])
            right_id = int(linkage[i, 1])
            distance = linkage[i, 2]
            count = int(linkage[i, 3])
            
            # Get asset names in each sub-cluster
            left_assets = self._get_asset_names(left_id)
            right_assets = self._get_asset_names(right_id)
            
            structure.append({
                'merge_step': i,
                'left_cluster': left_assets,
                'right_cluster': right_assets,
                'distance': float(distance),
                'cluster_size': count,
            })
        
        return structure
    
    def _get_asset_names(self, cluster_id: int) -> list[str]:
        """Get asset names for a cluster ID."""
        if cluster_id < self.NUM_ASSETS:
            return [self.ASSET_LABELS[cluster_id]]
        
        # Internal node
        linkage = self.linkage_matrix
        n = self.NUM_ASSETS
        left = int(linkage[cluster_id - n, 0])
        right = int(linkage[cluster_id - n, 1])
        
        return self._get_asset_names(left) + self._get_asset_names(right)
    
    def get_diversification_ratio(self, weights: Optional[np.ndarray] = None) -> float:
        """
        Compute portfolio diversification ratio.
        
        DR = (sum of weighted volatilities) / portfolio volatility
        Higher values indicate better diversification.
        """
        if weights is None:
            result = self.optimize()
            weights = np.array([result.weights[a] for a in self.ASSET_LABELS])
        
        # Weighted average volatility
        avg_vol = np.dot(weights, self.volatilities)
        
        # Portfolio volatility
        port_vol = np.sqrt(np.dot(weights, np.dot(self.covariance, weights)))
        
        if port_vol < 1e-10:
            return float('inf')
        
        return avg_vol / port_vol


def example_usage() -> None:
    """Demonstrate HRP optimization."""
    # Sample covariance for BTC, SOL, ETH, USDT
    covariance = np.array([
        [0.04, 0.02, 0.025, 0.0001],
        [0.02, 0.09, 0.04, 0.0002],
        [0.025, 0.04, 0.05, 0.0001],
        [0.0001, 0.0002, 0.0001, 0.0001],
    ])
    
    hrp = HierarchicalRiskParity(covariance)
    result = hrp.optimize()
    
    print("HRP Optimal Weights:")
    for asset, weight in result.weights.items():
        print(f"  {asset}: {weight:.2%}")
    
    print(f"\nCluster Structure:")
    for cluster in result.cluster_structure:
        left = ', '.join(cluster['left_cluster'])
        right = ', '.join(cluster['right_cluster'])
        print(f"  Merge {cluster['merge_step']}: [{left}] + [{right}]")
    
    print(f"\nCophenetic Correlation: {result.cophenetic_corr:.3f}")
    print(f"Diversification Ratio: {hrp.get_diversification_ratio():.3f}")


if __name__ == "__main__":
    example_usage()
