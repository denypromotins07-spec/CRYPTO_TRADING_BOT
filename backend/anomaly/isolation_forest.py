"""
Isolation Forest for Streaming Anomaly Detection

Lightweight streaming isolation trees for detecting spoofing and
market manipulation patterns in real-time order book data.

Key Features:
- Incremental tree updates for streaming data
- Memory-bounded with fixed tree count and depth
- Optimized for high-dimensional order book features
- Fast anomaly scoring using path length analysis

Author: Opus 4.8 for ZAID PERSONAL CRYPTO TRADING BOT
Stage: 29/100 - Statistical Anomaly Detection
"""

from __future__ import annotations
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import random


@dataclass
class IsolationForestConfig:
    """Configuration for Isolation Forest."""
    
    # Number of isolation trees
    n_trees: int = 100
    
    # Maximum tree depth (log2 of sample size recommended)
    max_depth: int = 8
    
    # Sample size per tree (subsample for efficiency)
    sample_size: int = 256
    
    # Contamination rate (expected proportion of outliers)
    contamination: float = 0.01
    
    # Minimum samples before activation
    min_samples: int = 50
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None


@dataclass
class IsolationTree:
    """Single isolation tree for anomaly detection."""
    
    # Split feature index
    split_feature: Optional[int] = None
    
    # Split value
    split_value: Optional[float] = None
    
    # Left child
    left: Optional['IsolationTree'] = None
    
    # Right child
    right: Optional['IsolationTree'] = None
    
    # Node depth
    depth: int = 0
    
    # Is leaf node
    is_leaf: bool = False
    
    # Number of samples at this node
    size: int = 0


class StreamingIsolationForest:
    """
    Streaming Isolation Forest for real-time anomaly detection.
    
    Efficiently detects spoofing, layering, and other manipulative
    patterns in order book data streams.
    """
    
    def __init__(self, config: Optional[IsolationForestConfig] = None):
        self.config = config or IsolationForestConfig()
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        self.trees: List[IsolationTree] = []
        self.data_buffer: List[NDArray[np.float64]] = []
        self.sample_count: int = 0
        self.is_fitted: bool = False
        
        # Feature bounds for adaptive splitting
        self.feature_mins: Optional[NDArray[np.float64]] = None
        self.feature_maxs: Optional[NDArray[np.float64]] = None
        
    def partial_fit(self, data: NDArray[np.float64]) -> None:
        """
        Incrementally update the forest with new data.
        
        Args:
            data: 2D array of shape (n_samples, n_features) or 1D for single sample
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_samples, n_features = data.shape
        
        # Initialize feature bounds
        if self.feature_mins is None:
            self.feature_mins = np.min(data, axis=0)
            self.feature_maxs = np.max(data, axis=0)
        else:
            self.feature_mins = np.minimum(self.feature_mins, np.min(data, axis=0))
            self.feature_maxs = np.maximum(self.feature_maxs, np.max(data, axis=0))
        
        # Buffer data for batch fitting
        for sample in data:
            self.data_buffer.append(sample.copy())
            self.sample_count += 1
            
            # Rebuild trees when buffer reaches sample_size
            if len(self.data_buffer) >= self.config.sample_size:
                self._rebuild_trees()
    
    def _rebuild_trees(self) -> None:
        """Rebuild all trees with current buffered data."""
        if len(self.data_buffer) < self.config.min_samples:
            return
        
        # Subsample for each tree
        data = np.array(self.data_buffer[-self.config.sample_size:])
        
        self.trees = []
        for _ in range(self.config.n_trees):
            # Bootstrap sample
            indices = np.random.choice(
                len(data), 
                size=min(len(data), self.config.sample_size),
                replace=True
            )
            sample = data[indices]
            
            # Build tree
            tree = self._build_tree(sample, depth=0)
            self.trees.append(tree)
        
        self.is_fitted = True
        # Clear buffer after building
        self.data_buffer.clear()
    
    def _build_tree(self, X: NDArray[np.float64], depth: int) -> IsolationTree:
        """Recursively build an isolation tree."""
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if depth >= self.config.max_depth or n_samples <= 1:
            return IsolationTree(depth=depth, is_leaf=True, size=n_samples)
        
        # Check if all samples are identical
        if np.all(X == X[0]):
            return IsolationTree(depth=depth, is_leaf=True, size=n_samples)
        
        # Random feature selection
        feature_idx = np.random.randint(n_features)
        
        # Get feature bounds
        feature_min = self.feature_mins[feature_idx] if self.feature_mins is not None else X[:, feature_idx].min()
        feature_max = self.feature_maxs[feature_idx] if self.feature_maxs is not None else X[:, feature_idx].max()
        
        # Random split value
        if feature_min == feature_max:
            return IsolationTree(depth=depth, is_leaf=True, size=n_samples)
        
        split_value = np.random.uniform(feature_min, feature_max)
        
        # Split data
        left_mask = X[:, feature_idx] < split_value
        right_mask = ~left_mask
        
        X_left = X[left_mask]
        X_right = X[right_mask]
        
        # Handle edge case where all samples go to one side
        if len(X_left) == 0 or len(X_right) == 0:
            return IsolationTree(depth=depth, is_leaf=True, size=n_samples)
        
        # Recursively build children
        left_child = self._build_tree(X_left, depth + 1)
        right_child = self._build_tree(X_right, depth + 1)
        
        return IsolationTree(
            split_feature=feature_idx,
            split_value=split_value,
            left=left_child,
            right=right_child,
            depth=depth,
            is_leaf=False,
            size=n_samples
        )
    
    def score(self, observation: NDArray[np.float64]) -> float:
        """
        Compute anomaly score for a single observation.
        
        Returns:
            Score in [0, 1] where higher means more anomalous
        """
        if not self.is_fitted or len(self.trees) == 0:
            return 0.5  # Neutral score when not fitted
        
        obs = np.asarray(observation)
        
        # Average path length across all trees
        path_lengths = []
        for tree in self.trees:
            path_len = self._path_length(obs, tree, 0)
            path_lengths.append(path_len)
        
        avg_path_length = np.mean(path_lengths)
        
        # Normalize using expected path length for random BST
        n = self.config.sample_size
        c_n = self._c_factor(n)
        
        # Anomaly score: s(x, n) = 2^(-E(h(x))/c(n))
        score = 2 ** (-avg_path_length / c_n) if c_n > 0 else 0.5
        
        return float(np.clip(score, 0, 1))
    
    def _path_length(self, observation: NDArray[np.float64], 
                     node: IsolationTree, current_depth: int) -> float:
        """Compute path length for observation through a tree."""
        if node.is_leaf:
            # Add adjustment for unbuilt subtree
            return current_depth + self._c_factor(node.size)
        
        feature_idx = node.split_feature
        if feature_idx is None or feature_idx >= len(observation):
            return current_depth + self._c_factor(node.size)
        
        if observation[feature_idx] < (node.split_value or 0):
            if node.left is not None:
                return self._path_length(observation, node.left, current_depth + 1)
        else:
            if node.right is not None:
                return self._path_length(observation, node.right, current_depth + 1)
        
        return current_depth + self._c_factor(node.size)
    
    @staticmethod
    def _c_factor(n: int) -> float:
        """
        Average path length of unsuccessful search in BST.
        
        Used for normalization of anomaly scores.
        """
        if n <= 1:
            return 0.0
        elif n == 2:
            return 1.0
        else:
            # H(n-1) ≈ ln(n-1) + γ (Euler-Mascheroni constant)
            harmonic = np.log(n - 1) + 0.5772156649
            return 2.0 * harmonic - 2.0 * (n - 1) / n
    
    def predict(self, observation: NDArray[np.float64], 
                threshold: Optional[float] = None) -> bool:
        """
        Predict if observation is anomalous.
        
        Args:
            observation: Feature vector
            threshold: Score threshold (default from contamination)
            
        Returns:
            True if anomalous
        """
        if threshold is None:
            # Default threshold based on contamination
            threshold = 0.6 + 0.4 * self.config.contamination
        
        score = self.score(observation)
        return score > threshold
    
    def score_batch(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score multiple observations efficiently."""
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        
        scores = np.array([self.score(obs) for obs in observations])
        return scores
    
    def get_feature_importance(self) -> Optional[NDArray[np.float64]]:
        """Estimate feature importance based on split frequencies."""
        if not self.trees:
            return None
        
        n_features = len(self.feature_mins) if self.feature_mins is not None else 0
        if n_features == 0:
            return None
        
        importance = np.zeros(n_features)
        
        for tree in self.trees:
            self._accumulate_importance(tree, importance)
        
        # Normalize
        total = importance.sum()
        if total > 0:
            importance /= total
        
        return importance
    
    def _accumulate_importance(self, node: IsolationTree, 
                               importance: NDArray[np.float64]) -> None:
        """Recursively accumulate feature importance."""
        if node.is_leaf:
            return
        
        if node.split_feature is not None:
            # Weight by inverse depth (earlier splits more important)
            weight = 1.0 / (node.depth + 1)
            importance[node.split_feature] += weight
        
        if node.left:
            self._accumulate_importance(node.left, importance)
        if node.right:
            self._accumulate_importance(node.right, importance)
    
    def reset(self) -> None:
        """Reset the forest for retraining."""
        self.trees.clear()
        self.data_buffer.clear()
        self.sample_count = 0
        self.is_fitted = False
        self.feature_mins = None
        self.feature_maxs = None


if __name__ == "__main__":
    # Example usage for spoofing detection
    np.random.seed(42)
    
    config = IsolationForestConfig(
        n_trees=50,
        max_depth=6,
        sample_size=128,
        contamination=0.02
    )
    
    forest = StreamingIsolationForest(config)
    
    # Simulate normal order book features
    # [bid_ask_spread, order_imbalance, volume_ratio, price_momentum]
    normal_data = np.random.normal(
        loc=[0.01, 0.0, 1.0, 0.0],
        scale=[0.005, 0.2, 0.3, 0.02],
        size=(200, 4)
    )
    
    # Train incrementally
    for i in range(0, len(normal_data), 10):
        batch = normal_data[i:i+10]
        forest.partial_fit(batch)
    
    # Test with normal observation
    normal_obs = np.array([0.01, 0.05, 1.1, 0.01])
    normal_score = forest.score(normal_obs)
    print(f"Normal observation score: {normal_score:.3f}")
    
    # Test with spoofing pattern (abnormal spread + imbalance)
    spoof_obs = np.array([0.05, 0.8, 5.0, -0.02])
    spoof_score = forest.score(spoof_obs)
    print(f"Spoofing pattern score: {spoof_score:.3f}")
    print(f"Is spoofing detected: {forest.predict(spoof_obs)}")
    
    # Feature importance
    importance = forest.get_feature_importance()
    if importance is not None:
        print(f"\nFeature importance: {importance}")
