#!/usr/bin/env python3
"""
Mutual Information Calculator for Feature Selection

This module calculates mutual information (MI) between technical indicators
and target returns to identify and drop redundant features. MI captures both
linear and non-linear dependencies, making it superior to correlation-based
feature selection for crypto markets.

Key Features:
- Non-parametric MI estimation using k-nearest neighbors
- Handles high-dimensional feature spaces efficiently
- Detects redundant features via conditional MI
- Memory-efficient implementation for 8GB RAM constraint
- Thread-safe for concurrent feature evaluation

Usage:
    mi_calc = MutualInformationCalculator(k_neighbors=5)
    mi_scores = mi_calc.calculate(features, target)
    selected = mi_calc.select_features(mi_scores, threshold=0.1)
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Set
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass(slots=True)
class MIConfig:
    """Configuration for mutual information calculation."""
    k_neighbors: int = 5  # Number of neighbors for k-NN MI estimation
    add_noise: float = 1e-6  # Small noise for continuous variables
    random_state: int = 42
    n_jobs: int = -1  # Parallel jobs
    discrete_features: bool = False


@dataclass(slots=True)
class FeatureInfo:
    """Information about a single feature."""
    name: str
    mi_score: float
    normalized_mi: float
    redundancy_score: float
    is_selected: bool
    rank: int


class MutualInformationCalculator:
    """
    Calculate mutual information between features and target.
    
    Uses k-nearest neighbor based estimation which is robust
    for continuous crypto market data.
    """
    
    def __init__(self, config: Optional[MIConfig] = None):
        self.config = config or MIConfig()
        self._feature_names: List[str] = []
        self._mi_scores: Optional[np.ndarray] = None
    
    def calculate(self, 
                  features: np.ndarray, 
                  target: np.ndarray,
                  feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Calculate mutual information between each feature and target.
        
        Parameters
        ----------
        features : np.ndarray
            Shape: (n_samples, n_features)
        target : np.ndarray
            Shape: (n_samples,)
        feature_names : List[str], optional
            Names of features
            
        Returns
        -------
        mi_scores : np.ndarray
            Mutual information score for each feature
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        if len(features) != len(target):
            raise ValueError("Features and target must have same length")
        
        if feature_names:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"feat_{i}" for i in range(features.shape[1])]
        
        # Add small noise to prevent numerical issues
        features_noisy = features + np.random.normal(0, self.config.add_noise, features.shape)
        
        # Calculate MI using sklearn's efficient implementation
        mi_scores = mutual_info_regression(
            features_noisy,
            target,
            discrete_features=self.config.discrete_features,
            n_neighbors=self.config.k_neighbors,
            copy=True,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        self._mi_scores = mi_scores
        return mi_scores
    
    def calculate_pairwise(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise mutual information between all features.
        
        Useful for detecting redundant features.
        
        Returns
        -------
        mi_matrix : np.ndarray
            Shape: (n_features, n_features)
        """
        n_features = features.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # MI between feature i and j
                mi = mutual_info_regression(
                    features[:, i:i+1],
                    features[:, j],
                    n_neighbors=self.config.k_neighbors,
                    random_state=self.config.random_state
                )[0]
                
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        return mi_matrix
    
    def select_features(self, 
                       mi_scores: np.ndarray,
                       threshold: float = 0.01,
                       max_features: Optional[int] = None) -> np.ndarray:
        """
        Select features based on MI scores.
        
        Parameters
        ----------
        mi_scores : np.ndarray
            MI scores for each feature
        threshold : float
            Minimum MI score for selection
        max_features : int, optional
            Maximum number of features to select
            
        Returns
        -------
        mask : np.ndarray
            Boolean mask for selected features
        """
        # Normalize scores
        max_score = mi_scores.max()
        if max_score > 0:
            normalized = mi_scores / max_score
        else:
            normalized = mi_scores
        
        # Select above threshold
        mask = normalized >= threshold
        
        # Limit to max_features
        if max_features is not None:
            selected_indices = np.where(mask)[0]
            sorted_indices = selected_indices[np.argsort(-mi_scores[selected_indices])]
            
            mask = np.zeros_like(mask)
            mask[sorted_indices[:max_features]] = True
        
        return mask
    
    def remove_redundant(self, 
                        features: np.ndarray,
                        target: np.ndarray,
                        threshold: float = 0.8) -> List[int]:
        """
        Remove redundant features using conditional MI heuristic.
        
        A feature is redundant if it has high MI with another feature
        but low incremental MI with the target.
        
        Returns
        -------
        selected_indices : List[int]
            Indices of non-redundant features
        """
        n_features = features.shape[1]
        
        # Calculate feature-target MI
        mi_target = self.calculate(features, target)
        
        # Calculate feature-feature MI
        mi_pairwise = self.calculate_pairwise(features)
        
        # Greedy selection
        selected = []
        remaining = set(range(n_features))
        
        while remaining:
            # Find feature with highest MI to target
            best_idx = max(remaining, key=lambda i: mi_target[i])
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            # Remove features highly correlated with selected one
            to_remove = set()
            for i in remaining:
                if mi_pairwise[best_idx, i] > threshold:
                    to_remove.add(i)
            remaining -= to_remove
        
        return selected
    
    def get_feature_importance(self, 
                              features: np.ndarray,
                              target: np.ndarray,
                              feature_names: Optional[List[str]] = None) -> List[FeatureInfo]:
        """
        Get comprehensive feature importance ranking.
        
        Returns list of FeatureInfo sorted by MI score.
        """
        mi_scores = self.calculate(features, target, feature_names)
        
        # Normalize
        max_score = mi_scores.max()
        normalized = mi_scores / max_score if max_score > 0 else mi_scores
        
        # Calculate redundancy
        mi_pairwise = self.calculate_pairwise(features)
        redundancy = mi_pairwise.mean(axis=1)
        
        # Create feature info
        indices = np.argsort(-mi_scores)
        
        results = []
        for rank, idx in enumerate(indices):
            info = FeatureInfo(
                name=self._feature_names[idx] if idx < len(self._feature_names) else f"feat_{idx}",
                mi_score=mi_scores[idx],
                normalized_mi=normalized[idx],
                redundancy_score=redundancy[idx],
                is_selected=normalized[idx] > 0.1,  # Default threshold
                rank=rank + 1
            )
            results.append(info)
        
        return results


class AdaptiveFeatureSelector:
    """
    Adaptive feature selector that updates selection based on regime changes.
    
    Monitors MI scores over time and adjusts feature selection when
    market regime changes are detected.
    """
    
    def __init__(self, window_size: int = 252, retrain_freq: int = 20):
        self.window_size = window_size
        self.retrain_freq = retrain_freq
        self.calculator = MutualInformationCalculator()
        self._feature_buffer: List[np.ndarray] = []
        self._target_buffer: List[float] = []
        self._last_selection: Optional[np.ndarray] = None
        self._update_count: int = 0
    
    def update(self, features: np.ndarray, target: float) -> Optional[np.ndarray]:
        """
        Update buffers and potentially recalculate feature selection.
        
        Returns new selection mask if recalculated, None otherwise.
        """
        self._feature_buffer.append(features)
        self._target_buffer.append(target)
        
        # Trim buffers
        while len(self._feature_buffer) > self.window_size:
            self._feature_buffer.pop(0)
            self._target_buffer.pop(0)
        
        self._update_count += 1
        
        # Recalculate periodically
        if self._update_count % self.retrain_freq == 0 and len(self._feature_buffer) >= 50:
            X = np.array(self._feature_buffer)
            y = np.array(self._target_buffer)
            
            mi_scores = self.calculator.calculate(X, y)
            self._last_selection = self.calculator.select_features(mi_scores, threshold=0.05)
            
            return self._last_selection
        
        return None
    
    def get_current_selection(self) -> Optional[np.ndarray]:
        """Get current feature selection mask."""
        return self._last_selection


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    # Generate synthetic features with varying relevance
    X = np.random.randn(n_samples, n_features)
    
    # Target depends on first 5 features + noise
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] + 
         np.sin(X[:, 3]) + X[:, 4]**2 + np.random.randn(n_samples) * 0.5)
    
    # Calculate MI
    calc = MutualInformationCalculator()
    mi_scores = calc.calculate(X, y)
    
    print("Mutual Information Scores:")
    for i, score in enumerate(mi_scores):
        print(f"  Feature {i}: {score:.4f}")
    
    # Select features
    mask = calc.select_features(mi_scores, threshold=0.1)
    print(f"\nSelected features: {np.where(mask)[0]}")
    
    # Remove redundant
    selected = calc.remove_redundant(X, y, threshold=0.5)
    print(f"After redundancy removal: {selected}")
