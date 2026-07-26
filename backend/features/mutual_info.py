#!/usr/bin/env python3
"""
Mutual Information Feature Selection - Drop Redundant Technical Indicators

This module calculates mutual information between features and target variable
to identify and remove redundant technical indicators, preventing overfitting.

Key Features:
- Non-parametric dependency measurement (captures non-linear relationships)
- K-nearest neighbor based MI estimation (Kraskov-Stögbauer-Grassberger)
- Memory-efficient implementation for high-dimensional feature spaces
- Iterative feature selection with redundancy penalty

Optimized for crypto feature engineering with 100+ technical indicators.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
import warnings


@dataclass
class FeatureImportance:
    """Container for feature importance scores."""
    name: str
    mutual_information: float
    redundancy_score: float
    relevance_score: float
    selected: bool = False
    
    def __str__(self) -> str:
        status = "✓" if self.selected else "✗"
        return f"{status} {self.name}: MI={self.mutual_information:.4f}, Relevance={self.relevance_score:.4f}"


@dataclass 
class FeatureSelectionResult:
    """Results from mutual information feature selection."""
    selected_features: List[str]
    all_importances: List[FeatureImportance]
    n_original: int
    n_selected: int
    total_mi: float
    redundancy_removed: float
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Feature Selection Summary\n"
            f"{'='*50}\n"
            f"Original features: {self.n_original}\n"
            f"Selected features: {self.n_selected}\n"
            f"Reduction: {(1 - self.n_selected/self.n_original)*100:.1f}%\n"
            f"Total Mutual Information: {self.total_mi:.4f}\n"
            f"Redundancy Removed: {self.redundancy_removed:.4f}\n"
            f"\nSelected Features:\n"
            f"{'-'*30}\n"
            + "\n".join(f"  • {f}" for f in self.selected_features[:20])
            + ("\n  ..." if len(self.selected_features) > 20 else "")
        )


class MutualInformationEstimator:
    """
    K-nearest neighbor based mutual information estimator.
    
    Implements Kraskov-Stögbauer-Grassberger (KSG) estimator which is
    consistent and works well for continuous variables.
    """
    
    def __init__(self, k: int = 3, add_noise: float = 1e-8):
        """
        Initialize MI estimator.
        
        Args:
            k: Number of neighbors for density estimation
            add_noise: Small noise to prevent numerical issues
        """
        self.k = k
        self.add_noise = add_noise
    
    def estimate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate mutual information between X and y.
        
        I(X;Y) = ψ(k) - <ψ(n_x + 1)> - <ψ(n_y + 1)> + ψ(N)
        
        where ψ is the digamma function.
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.float64)
        
        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        N = len(X)
        if N < self.k + 2:
            warnings.warn(f"Insufficient samples ({N}) for MI estimation")
            return 0.0
        
        # Add small noise to prevent ties
        X_noisy = X + np.random.randn(*X.shape) * self.add_noise
        y_noisy = y + np.random.randn(*y.shape) * self.add_noise
        
        # Joint space
        joint = np.hstack([X_noisy, y_noisy])
        
        # Build KD-tree for joint space
        tree_joint = cKDTree(joint)
        
        # Find k-th nearest neighbor distances in joint space
        distances, _ = tree_joint.query(joint, k=self.k + 1)
        eps = distances[:, -1] / 2.0  # Half the k-th neighbor distance
        
        # Count neighbors within eps in marginal spaces
        tree_X = cKDTree(X_noisy)
        tree_y = cKDTree(y_noisy)
        
        # n_x: number of points within eps in X space (excluding self)
        n_x = tree_X.query_ball_point(X_noisy, r=eps)
        n_x_counts = np.array([len(nx) - 1 for nx in n_x])
        
        # n_y: number of points within eps in y space (excluding self)
        n_y = tree_y.query_ball_point(y_noisy, r=eps)
        n_y_counts = np.array([len(ny) - 1 for ny in n_y])
        
        # Compute MI using KSG formula
        from scipy.special import digamma, psi
        
        mi = (digamma(self.k) 
              - np.mean(digamma(n_x_counts + 1)) 
              - np.mean(digamma(n_y_counts + 1)) 
              + digamma(N))
        
        return max(0.0, mi)  # MI should be non-negative
    
    def estimate_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate pairwise mutual information matrix for all features.
        
        Returns:
            Symmetric matrix of shape (n_features, n_features)
        """
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi = self.estimate(X[:, i], X[:, j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        return mi_matrix


class MIFeatureSelector:
    """
    Feature selector based on mutual information with redundancy penalty.
    
    Implements mRMR (minimum Redundancy Maximum Relevance) algorithm:
    - Maximize relevance: I(feature; target)
    - Minimize redundancy: mean(I(feature; selected_features))
    """
    
    def __init__(self, 
                 mi_estimator: Optional[MutualInformationEstimator] = None,
                 relevance_weight: float = 0.7):
        """
        Initialize feature selector.
        
        Args:
            mi_estimator: Custom MI estimator (default creates one)
            relevance_weight: Weight for relevance vs redundancy (0-1)
        """
        self.mi_estimator = mi_estimator or MutualInformationEstimator()
        self.relevance_weight = relevance_weight
        self._results: Optional[FeatureSelectionResult] = None
    
    def select_features(self, 
                       X: np.ndarray,
                       y: np.ndarray,
                       feature_names: Optional[List[str]] = None,
                       n_select: Optional[int] = None,
                       threshold: float = 0.01) -> FeatureSelectionResult:
        """
        Select optimal feature subset using mRMR criterion.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: Optional names for features
            n_select: Number of features to select (default: auto)
            threshold: Minimum MI threshold for inclusion
            
        Returns:
            FeatureSelectionResult with selected features
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if len(feature_names) != n_features:
            raise ValueError("feature_names length must match n_features")
        
        # Step 1: Compute MI between each feature and target (relevance)
        relevance_scores = np.zeros(n_features)
        for i in range(n_features):
            relevance_scores[i] = self.mi_estimator.estimate(X[:, i], y)
        
        # Step 2: Compute pairwise MI between features (redundancy)
        # For efficiency, compute incrementally during selection
        mi_matrix = self.mi_estimator.estimate_matrix(X)
        
        # Step 3: Greedy mRMR selection
        selected_indices: Set[int] = set()
        importance_list = []
        
        # Initialize with highest relevance feature
        first_idx = np.argmax(relevance_scores)
        selected_indices.add(first_idx)
        
        remaining = set(range(n_features)) - selected_indices
        
        while remaining:
            best_score = -np.inf
            best_idx = None
            
            for idx in remaining:
                # Relevance term
                rel = relevance_scores[idx]
                
                # Redundancy term (average MI with already selected features)
                if selected_indices:
                    redundancy = np.mean([mi_matrix[idx, s] for s in selected_indices])
                else:
                    redundancy = 0.0
                
                # mRMR score
                score = self.relevance_weight * rel - (1 - self.relevance_weight) * redundancy
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                # Check threshold
                if relevance_scores[best_idx] >= threshold:
                    selected_indices.add(best_idx)
                    remaining.remove(best_idx)
                else:
                    break  # No more features above threshold
            else:
                break
        
        # Create importance objects
        for i in range(n_features):
            # Compute average redundancy for this feature
            if selected_indices:
                avg_redundancy = np.mean([mi_matrix[i, s] for s in selected_indices if s != i])
            else:
                avg_redundancy = 0.0
            
            importance = FeatureImportance(
                name=feature_names[i],
                mutual_information=relevance_scores[i],
                redundancy_score=avg_redundancy,
                relevance_score=relevance_scores[i],
                selected=(i in selected_indices)
            )
            importance_list.append(importance)
        
        # Sort by MI descending
        importance_list.sort(key=lambda x: x.mutual_information, reverse=True)
        
        # Determine final selection
        if n_select is not None:
            # Override with fixed number
            for i, imp in enumerate(importance_list):
                imp.selected = (i < n_select)
            final_selected = [imp.name for imp in importance_list[:n_select]]
        else:
            final_selected = [imp.name for imp in importance_list if imp.selected]
        
        # Calculate statistics
        total_mi = sum(imp.mutual_information for imp in importance_list if imp.selected)
        redundancy_removed = sum(
            imp.redundancy_score for imp in importance_list if not imp.selected
        )
        
        self._results = FeatureSelectionResult(
            selected_features=final_selected,
            all_importances=importance_list,
            n_original=n_features,
            n_selected=len(final_selected),
            total_mi=total_mi,
            redundancy_removed=redundancy_removed
        )
        
        return self._results
    
    def get_selected_mask(self) -> Optional[np.ndarray]:
        """Get boolean mask for selected features."""
        if self._results is None:
            return None
        
        mask = np.zeros(self._results.n_original, dtype=bool)
        for imp in self._results.all_importances:
            if imp.selected:
                idx = next(
                    i for i, x in enumerate(self._results.all_importances) 
                    if x.name == imp.name
                )
                mask[idx] = True
        return mask
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to selected feature subset."""
        if self._results is None:
            raise ValueError("Must run select_features before transform")
        
        mask = self.get_selected_mask()
        if mask is None:
            return X
        
        return X[:, mask]


def select_features_mrmr(X: np.ndarray, 
                         y: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         n_select: Optional[int] = None) -> FeatureSelectionResult:
    """
    Convenience function for mRMR feature selection.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Optional feature names
        n_select: Number of features to select
        
    Returns:
        FeatureSelectionResult with selected features
    """
    selector = MIFeatureSelector()
    return selector.select_features(X, y, feature_names, n_select)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    n_samples = 500
    n_features = 20
    
    # Generate synthetic data with known structure
    X = np.random.randn(n_samples, n_features)
    
    # Create target with non-linear relationship to first 5 features
    y = (X[:, 0]**2 + 
         np.sin(X[:, 1] * 2) + 
         X[:, 2] * X[:, 3] + 
         np.random.randn(n_samples) * 0.1)
    
    # Add redundant features (copies with noise)
    X[:, 10] = X[:, 0] + np.random.randn(n_samples) * 0.01
    X[:, 11] = X[:, 1] + np.random.randn(n_samples) * 0.01
    X[:, 12] = X[:, 2] + np.random.randn(n_samples) * 0.01
    
    feature_names = [f"feat_{i}" for i in range(n_features)]
    
    print("Running mRMR feature selection...")
    result = select_features_mrmr(X, y, feature_names, n_select=8)
    
    print(result.summary())
    
    print("\nTop 10 Features by MI:")
    for imp in result.all_importances[:10]:
        print(f"  {imp}")
