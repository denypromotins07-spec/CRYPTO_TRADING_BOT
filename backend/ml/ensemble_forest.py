"""
Ensemble Forest for Signal Fusion
Aggregates lightweight decision trees for robust signal generation.
Optimized for 8GB RAM with incremental updates.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreeConfig:
    """Configuration for individual decision tree."""
    max_depth: int = 4  # Shallow trees for speed
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    n_features_sample: Optional[int] = None  # For random forest behavior


@dataclass
class EnsembleConfig:
    """Configuration for ensemble forest."""
    n_trees: int = 20  # Number of trees
    tree_config: TreeConfig = field(default_factory=TreeConfig)
    
    # Bootstrap sampling
    bootstrap: bool = True
    sample_ratio: float = 0.8
    
    # Feature subsampling (random subspace)
    feature_ratio: float = 0.7
    
    # Voting weights
    weight_by_accuracy: bool = True
    
    # Confidence threshold for trading
    confidence_threshold: float = 0.6
    
    # Maximum history
    max_history: int = 5000


@dataclass
class DecisionNode:
    """Single decision node in tree."""
    feature_idx: int
    threshold: float
    left_child: Optional['DecisionNode']
    right_child: Optional['DecisionNode']
    value: Optional[float] = None  # Leaf prediction
    is_leaf: bool = False


class LightweightDecisionTree:
    """
    Lightweight decision tree for fast inference.
    
    Implements:
    - Greedy splitting on information gain
    - Depth limiting for speed
    - Pre-pruning via min_samples constraints
    """
    
    def __init__(self, config: TreeConfig):
        self.config = config
        self.root: Optional[DecisionNode] = None
        self.n_features: int = 0
        self.feature_subset: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: Optional[np.ndarray] = None
    ) -> 'LightweightDecisionTree':
        """
        Fit tree to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (continuous or binary)
            feature_indices: Subset of features to consider
        """
        self.n_features = X.shape[1]
        
        # Feature subsampling
        if feature_indices is not None:
            self.feature_subset = feature_indices
            X_subset = X[:, feature_indices]
        else:
            self.feature_subset = np.arange(self.n_features)
            X_subset = X
        
        # Build tree recursively
        self.root = self._build_tree(X_subset, y, depth=0)
        
        return self
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ) -> DecisionNode:
        """Recursively build tree."""
        n_samples = len(y)
        
        # Check stopping conditions
        if (depth >= self.config.max_depth or
            n_samples < self.config.min_samples_split or
            len(np.unique(y)) == 1):
            # Create leaf
            return DecisionNode(
                feature_idx=0,
                threshold=0.0,
                left_child=None,
                right_child=None,
                value=float(np.mean(y)),
                is_leaf=True
            )
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_gain <= 0:
            # No improvement possible
            return DecisionNode(
                feature_idx=0,
                threshold=0.0,
                left_child=None,
                right_child=None,
                value=float(np.mean(y)),
                is_leaf=True
            )
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if left_mask.sum() < self.config.min_samples_leaf or \
           right_mask.sum() < self.config.min_samples_leaf:
            # Can't split further
            return DecisionNode(
                feature_idx=0,
                threshold=0.0,
                left_child=None,
                right_child=None,
                value=float(np.mean(y)),
                is_leaf=True
            )
        
        # Recursively build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(
            feature_idx=self.feature_subset[best_feature] if self.feature_subset is not None else best_feature,
            threshold=best_threshold,
            left_child=left_child,
            right_child=right_child,
            is_leaf=False
        )
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[int, float, float]:
        """Find best feature and threshold for splitting."""
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0.0
        
        parent_variance = np.var(y) * n_samples
        
        for feature_idx in range(n_features):
            # Get unique thresholds
            values = np.unique(X[:, feature_idx])
            if len(values) < 2:
                continue
            
            # Try midpoints between consecutive values
            thresholds = (values[:-1] + values[1:]) / 2
            
            # Limit number of thresholds to check
            if len(thresholds) > 20:
                indices = np.linspace(0, len(thresholds) - 1, 20, dtype=int)
                thresholds = thresholds[indices]
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                
                if n_left < self.config.min_samples_leaf or \
                   n_right < self.config.min_samples_leaf:
                    continue
                
                # Compute variance reduction
                var_left = np.var(y[left_mask]) * n_left if n_left > 0 else 0
                var_right = np.var(y[right_mask]) * n_right if n_right > 0 else 0
                
                gain = parent_variance - (var_left + var_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def predict_one(self, x: np.ndarray) -> float:
        """Predict for single sample."""
        if self.root is None:
            return 0.0
        
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        
        return node.value if node.value is not None else 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for multiple samples."""
        return np.array([self.predict_one(x) for x in X])


class EnsembleForest:
    """
    Ensemble of lightweight decision trees for signal fusion.
    
    Implements:
    - Bootstrap aggregating (bagging)
    - Random subspace method
    - Weighted voting by tree accuracy
    - Confidence estimation via vote dispersion
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.trees: List[LightweightDecisionTree] = []
        self.tree_weights: np.ndarray = np.ones(config.n_trees) / config.n_trees
        
        # Training history
        self.X_history: deque = deque(maxlen=config.max_history)
        self.y_history: deque = deque(maxlen=config.max_history)
        
        # Out-of-bag estimates
        self.oob_predictions: Dict[int, List[float]] = {}
        
        # Enabled flag
        self.enabled: bool = True
        self.disable_reason: Optional[str] = None
        
        logger.info(f"EnsembleForest initialized: n_trees={config.n_trees}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'EnsembleForest':
        """
        Fit ensemble to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values
        """
        n_samples, n_features = X.shape
        self.trees = []
        oob_counts = np.zeros(n_samples)
        oob_predictions = np.zeros(n_samples)
        
        # Determine feature subset size
        n_features_sample = self.config.tree_config.n_features_sample
        if n_features_sample is None:
            n_features_sample = max(1, int(n_features * self.config.feature_ratio))
        
        for i in range(self.config.n_trees):
            # Bootstrap sample
            if self.config.bootstrap:
                n_sample = int(n_samples * self.config.sample_ratio)
                indices = np.random.choice(n_samples, size=n_sample, replace=True)
                oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
            else:
                indices = np.arange(n_samples)
                oob_indices = np.array([])
            
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Random feature subset
            feature_indices = np.random.choice(
                n_features,
                size=min(n_features_sample, n_features),
                replace=False
            )
            
            # Fit tree
            tree = LightweightDecisionTree(self.config.tree_config)
            tree.fit(X_boot, y_boot, feature_indices)
            self.trees.append(tree)
            
            # OOB predictions
            if len(oob_indices) > 0:
                oob_preds = tree.predict(X[oob_indices])
                oob_predictions[oob_indices] += oob_preds
                oob_counts[oob_indices] += 1
        
        # Compute tree weights based on OOB accuracy
        if self.config.weight_by_accuracy:
            valid_oob = oob_counts > 0
            if valid_oob.any():
                oob_errors = np.abs(oob_predictions[valid_oob] / oob_counts[valid_oob] - y[valid_oob])
                tree_accuracies = 1.0 / (1.0 + np.mean(oob_errors))
                self.tree_weights = tree_accuracies / tree_accuracies.sum()
            else:
                self.tree_weights = np.ones(self.config.n_trees) / self.config.n_trees
        
        # Store history
        for x, yi in zip(X, y):
            self.X_history.append(x.copy())
            self.y_history.append(yi)
        
        logger.info(f"EnsembleForest fitted: {len(self.trees)} trees")
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.enabled:
            raise RuntimeError(f"Ensemble disabled: {self.disable_reason}")
        
        if len(self.trees) == 0:
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Collect predictions from all trees
        all_predictions = np.zeros((len(X), len(self.trees)))
        
        for i, tree in enumerate(self.trees):
            all_predictions[:, i] = tree.predict(X)
        
        # Weighted average
        predictions = np.average(all_predictions, axis=1, weights=self.tree_weights)
        
        # Confidence from vote dispersion (lower std = higher confidence)
        std_devs = np.std(all_predictions, axis=1)
        max_std = np.max(std_devs) if len(std_devs) > 0 else 1.0
        confidence = 1.0 - (std_devs / (max_std + 1e-8))
        
        return predictions, confidence
    
    def predict_one(
        self,
        x: np.ndarray
    ) -> Tuple[float, float, str]:
        """
        Predict single sample with interpretation.
        
        Returns:
            Tuple of (prediction, confidence, signal_direction)
        """
        if not self.enabled:
            return 0.0, 0.0, "DISABLED"
        
        preds = [tree.predict_one(x) for tree in self.trees]
        prediction = np.average(preds, weights=self.tree_weights)
        
        # Confidence
        std_dev = np.std(preds)
        confidence = 1.0 - (std_dev / (np.max(np.abs(preds)) + 1e-8))
        
        # Direction
        if prediction > self.config.confidence_threshold:
            direction = "STRONG_BUY"
        elif prediction > 0:
            direction = "WEAK_BUY"
        elif prediction < -self.config.confidence_threshold:
            direction = "STRONG_SELL"
        elif prediction < 0:
            direction = "WEAK_SELL"
        else:
            direction = "NEUTRAL"
        
        return prediction, confidence, direction
    
    def disable(self, reason: str) -> None:
        """Disable ensemble (e.g., due to model drift)."""
        self.enabled = False
        self.disable_reason = reason
        logger.warning(f"EnsembleForest disabled: {reason}")
    
    def enable(self) -> None:
        """Re-enable ensemble."""
        self.enabled = True
        self.disable_reason = None
        logger.info("EnsembleForest re-enabled")
    
    def get_feature_importance(self) -> np.ndarray:
        """Estimate feature importance across ensemble."""
        if len(self.trees) == 0:
            return np.array([])
        
        n_features = self.trees[0].n_features
        importance = np.zeros(n_features)
        
        for tree, weight in zip(self.trees, self.tree_weights):
            if tree.root is not None:
                self._accumulate_importance(tree.root, importance, weight)
        
        # Normalize
        total = importance.sum()
        if total > 0:
            importance /= total
        
        return importance
    
    def _accumulate_importance(
        self,
        node: DecisionNode,
        importance: np.ndarray,
        weight: float
    ) -> None:
        """Recursively accumulate feature importance."""
        if node.is_leaf:
            return
        
        importance[node.feature_idx] += weight
        
        if node.left_child:
            self._accumulate_importance(node.left_child, importance, weight)
        if node.right_child:
            self._accumulate_importance(node.right_child, importance, weight)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get ensemble summary."""
        if len(self.trees) == 0:
            return {'status': 'not_fitted'}
        
        return {
            'n_trees': len(self.trees),
            'enabled': self.enabled,
            'disable_reason': self.disable_reason,
            'mean_tree_weight': float(np.mean(self.tree_weights)),
            'max_tree_weight': float(np.max(self.tree_weights)),
            'min_tree_weight': float(np.min(self.tree_weights)),
            'feature_importance': self.get_feature_importance().tolist() if self.get_feature_importance().size > 0 else [],
        }


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # True signal: combination of features
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    config = EnsembleConfig(
        n_trees=20,
        tree_config=TreeConfig(max_depth=4),
    )
    
    ensemble = EnsembleForest(config)
    ensemble.fit(X, y)
    
    # Test prediction
    test_x = X[0]
    pred, conf, direction = ensemble.predict_one(test_x)
    print(f"Prediction: {pred:.4f}, Confidence: {conf:.4f}, Direction: {direction}")
    
    # Batch prediction
    preds, confidences = ensemble.predict(X[:10])
    print(f"Batch predictions: {preds[:5]}")
    print(f"Confidences: {confidences[:5]}")
    
    # Summary
    summary = ensemble.get_summary_statistics()
    print(f"\nSummary: {summary}")
