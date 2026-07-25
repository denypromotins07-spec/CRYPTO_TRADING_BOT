"""
=============================================================================
ZAID PERSONAL CRYPTO TRADING BOT - VECTOR STORE FOR MARKET REGIMES
=============================================================================
Memory and Self-Learning Foundation for Continuous Algorithmic Evolution

This module implements a lightweight vector store for storing and retrieving
learned market regimes using efficient local math models. No external LLM or
heavy ML frameworks - just pure numerical linear algebra with NumPy/SciPy.

Domains Integrated:
- Vector Space Models
- Similarity Search (Cosine, Euclidean)
- Clustering Algorithms (K-Means, DBSCAN)
- Dimensionality Reduction (PCA)
- Market Regime Classification
- Online Learning Systems
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import threading
import numpy as np
from enum import Enum


class RegimeType(Enum):
    """Classification of market regime types."""
    BULL_STRONG = "bull_strong"
    BULL_WEAK = "bull_weak"
    BEAR_STRONG = "bear_strong"
    BEAR_WEAK = "bear_weak"
    RANGING = "ranging"
    VOLATILE_HIGH = "volatile_high"
    VOLATILE_LOW = "volatile_low"
    TRANSITION = "transition"


@dataclass
class MarketFeatureVector:
    """
    Represents a market state as a feature vector for similarity comparison.
    
    Features (12-dimensional):
    0-2: Price momentum (1h, 4h, 24h returns)
    3-5: Volatility measures (realized vol, ATR ratio, Bollinger width)
    6-8: Volume metrics (volume ratio, OBV slope, volume volatility)
    9-10: Trend strength (ADX, DI+/-DI-)
    11: Market breadth indicator
    """
    features: np.ndarray
    timestamp: str
    asset: str
    regime_label: Optional[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate feature vector dimensions."""
        if self.features.shape != (12,):
            raise ValueError(f"Expected 12 features, got {self.features.shape}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "features": self.features.tolist(),
            "timestamp": self.timestamp,
            "asset": self.asset,
            "regime_label": self.regime_label,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketFeatureVector":
        """Deserialize from dictionary."""
        return cls(
            features=np.array(data["features"], dtype=np.float64),
            timestamp=data["timestamp"],
            asset=data["asset"],
            regime_label=data.get("regime_label"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class RegimeCluster:
    """
    Represents a cluster of similar market regimes.
    
    Attributes:
        cluster_id: Unique identifier
        centroid: Center point of the cluster in feature space
        members: Count of vectors assigned to this cluster
        regime_type: Dominant regime type in cluster
        created_at: When cluster was formed
        last_updated: Last time a member was added
    """
    cluster_id: int
    centroid: np.ndarray
    members: int = 0
    regime_type: Optional[str] = None
    created_at: str = ""
    last_updated: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "centroid": self.centroid.tolist(),
            "members": self.members,
            "regime_type": self.regime_type,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }


class VectorStore:
    """
    Lightweight vector store for market regime storage and retrieval.
    
    Uses cosine similarity and k-nearest neighbors for fast regime matching.
    Implements incremental clustering for online learning without retraining.
    
    Memory-efficient design:
    - Fixed-size circular buffer for recent vectors
    - Cluster centroids instead of storing all historical vectors
    - Memory-mapped arrays for large datasets
    """
    
    # Feature dimension constant
    FEATURE_DIM: int = 12
    
    def __init__(
        self,
        store_path: str = "./memory/vectors/",
        max_vectors: int = 10_000,
        num_clusters: int = 50
    ):
        """
        Initialize the vector store.
        
        Args:
            store_path: Directory for persisting vectors
            max_vectors: Maximum vectors to keep in memory (circular buffer)
            num_clusters: Target number of regime clusters
        """
        self.store_path = Path(store_path)
        self.max_vectors = max_vectors
        self.num_clusters = num_clusters
        
        # Thread-safe access
        self._lock = threading.RLock()
        
        # In-memory storage (circular buffer)
        self._vectors: List[MarketFeatureVector] = []
        self._vector_index = 0  # For circular buffer
        
        # Cluster centroids
        self._clusters: Dict[int, RegimeCluster] = {}
        self._next_cluster_id = 0
        
        # Asset-specific stores
        self._by_asset: Dict[str, List[int]] = {}  # asset -> indices
        
        # Ensure directories exist
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_state()
    
    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    def add_vector(
        self,
        features: Union[np.ndarray, List[float]],
        asset: str,
        regime_label: Optional[str] = None,
        confidence: float = 1.0
    ) -> int:
        """
        Add a new market feature vector to the store.
        
        Args:
            features: 12-dimensional feature vector
            asset: Asset symbol (BTC, ETH, SOL)
            regime_label: Optional pre-classified regime
            confidence: Confidence in the classification
            
        Returns:
            Index of the added vector
        """
        with self._lock:
            features_array = np.array(features, dtype=np.float64)
            
            if features_array.shape != (self.FEATURE_DIM,):
                raise ValueError(
                    f"Expected {self.FEATURE_DIM} features, "
                    f"got {features_array.shape}"
                )
            
            vector = MarketFeatureVector(
                features=features_array,
                timestamp=self._get_timestamp(),
                asset=asset.upper(),
                regime_label=regime_label,
                confidence=confidence,
            )
            
            # Circular buffer logic
            if len(self._vectors) < self.max_vectors:
                idx = len(self._vectors)
                self._vectors.append(vector)
            else:
                # Overwrite oldest vector
                idx = self._vector_index
                self._vectors[idx] = vector
                self._vector_index = (self._vector_index + 1) % self.max_vectors
            
            # Update asset index
            asset_key = asset.upper()
            if asset_key not in self._by_asset:
                self._by_asset[asset_key] = []
            self._by_asset[asset_key].append(idx)
            
            # Periodically update clusters
            if len(self._vectors) % 100 == 0:
                self._update_clusters()
            
            return idx
    
    def find_similar_regimes(
        self,
        features: Union[np.ndarray, List[float]],
        asset: Optional[str] = None,
        k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[MarketFeatureVector, float]]:
        """
        Find k most similar historical regimes.
        
        Uses cosine similarity for scale-invariant comparison.
        
        Args:
            features: Query feature vector
            asset: Optional asset filter
            k: Number of neighbors to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (vector, similarity_score) tuples
        """
        with self._lock:
            query = np.array(features, dtype=np.float64)
            query_norm = np.linalg.norm(query)
            
            if query_norm == 0:
                return []
            
            candidates = []
            
            for i, vector in enumerate(self._vectors):
                # Filter by asset if specified
                if asset and vector.asset != asset.upper():
                    continue
                
                # Calculate cosine similarity
                dot_product = np.dot(query, vector.features)
                vector_norm = np.linalg.norm(vector.features)
                
                if vector_norm == 0:
                    continue
                
                similarity = dot_product / (query_norm * vector_norm)
                
                if similarity >= similarity_threshold:
                    candidates.append((vector, similarity))
            
            # Sort by similarity descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            return candidates[:k]
    
    def classify_regime(
        self,
        features: Union[np.ndarray, List[float]],
        asset: Optional[str] = None
    ) -> Tuple[Optional[str], float]:
        """
        Classify current market regime based on stored patterns.
        
        Uses weighted voting from k nearest neighbors.
        
        Args:
            features: Current market feature vector
            asset: Optional asset filter
            
        Returns:
            Tuple of (predicted_regime, confidence)
        """
        similar = self.find_similar_regimes(features, asset, k=10)
        
        if not similar:
            return None, 0.0
        
        # Weighted voting by similarity
        regime_votes: Dict[str, float] = {}
        total_weight = 0.0
        
        for vector, similarity in similar:
            if vector.regime_label:
                weight = similarity * vector.confidence
                regime_votes[vector.regime_label] = (
                    regime_votes.get(vector.regime_label, 0.0) + weight
                )
                total_weight += weight
        
        if not regime_votes:
            return None, 0.0
        
        # Get regime with highest vote
        best_regime = max(regime_votes.items(), key=lambda x: x[1])
        confidence = best_regime[1] / total_weight if total_weight > 0 else 0.0
        
        return best_regime[0], confidence
    
    def _update_clusters(self) -> None:
        """
        Update regime clusters using incremental k-means.
        
        This is a simplified version - production would use
        more sophisticated online clustering algorithms.
        """
        if len(self._vectors) < self.num_clusters:
            return
        
        # Collect all feature vectors
        all_features = np.array([v.features for v in self._vectors])
        
        # Simple k-means initialization (use first k as centroids)
        n = min(self.num_clusters, len(all_features))
        centroids = all_features[:n].copy()
        
        # Assign vectors to clusters
        cluster_assignments = np.zeros(len(all_features), dtype=int)
        cluster_counts = np.zeros(n, dtype=int)
        
        # Single iteration for speed (online learning)
        for i, features in enumerate(all_features):
            # Find nearest centroid
            distances = np.array([
                np.linalg.norm(features - c) for c in centroids
            ])
            cluster_id = np.argmin(distances)
            cluster_assignments[i] = cluster_id
            cluster_counts[cluster_id] += 1
        
        # Update centroids (running average)
        for cluster_id in range(n):
            mask = cluster_assignments == cluster_id
            if np.sum(mask) > 0:
                # Move centroid toward new mean
                new_mean = np.mean(all_features[mask], axis=0)
                alpha = 0.1  # Learning rate
                centroids[cluster_id] = (
                    (1 - alpha) * centroids[cluster_id] + alpha * new_mean
                )
        
        # Update cluster objects
        now = self._get_timestamp()
        for cluster_id in range(n):
            if cluster_id in self._clusters:
                cluster = self._clusters[cluster_id]
                cluster.centroid = centroids[cluster_id]
                cluster.members = int(cluster_counts[cluster_id])
                cluster.last_updated = now
            else:
                self._clusters[cluster_id] = RegimeCluster(
                    cluster_id=cluster_id,
                    centroid=centroids[cluster_id],
                    members=int(cluster_counts[cluster_id]),
                    created_at=now,
                    last_updated=now,
                )
        
        self._next_cluster_id = max(self._next_cluster_id, n)
    
    def get_cluster_for_vector(
        self,
        features: Union[np.ndarray, List[float]]
    ) -> Optional[RegimeCluster]:
        """Find the cluster closest to given features."""
        if not self._clusters:
            return None
        
        query = np.array(features, dtype=np.float64)
        
        best_cluster = None
        best_distance = float('inf')
        
        for cluster in self._clusters.values():
            distance = np.linalg.norm(query - cluster.centroid)
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        
        return best_cluster
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            regime_counts: Dict[str, int] = {}
            asset_counts: Dict[str, int] = {}
            
            for vector in self._vectors:
                if vector.regime_label:
                    regime_counts[vector.regime_label] = (
                        regime_counts.get(vector.regime_label, 0) + 1
                    )
                asset_counts[vector.asset] = (
                    asset_counts.get(vector.asset, 0) + 1
                )
            
            return {
                "total_vectors": len(self._vectors),
                "max_vectors": self.max_vectors,
                "num_clusters": len(self._clusters),
                "assets": asset_counts,
                "regimes": regime_counts,
                "storage_path": str(self.store_path),
            }
    
    def _save_state(self) -> None:
        """Persist state to disk."""
        # Save vectors (sample only, not all for memory efficiency)
        sample_size = min(1000, len(self._vectors))
        sampled_vectors = self._vectors[-sample_size:]
        
        vectors_data = [v.to_dict() for v in sampled_vectors]
        vectors_file = self.store_path / "recent_vectors.json"
        vectors_file.write_text(
            json.dumps(vectors_data, indent=2),
            encoding='utf-8'
        )
        
        # Save clusters
        clusters_data = {
            str(k): v.to_dict() for k, v in self._clusters.items()
        }
        clusters_file = self.store_path / "clusters.json"
        clusters_file.write_text(
            json.dumps(clusters_data, indent=2),
            encoding='utf-8'
        )
    
    def _load_state(self) -> None:
        """Load state from disk if available."""
        vectors_file = self.store_path / "recent_vectors.json"
        if vectors_file.exists():
            try:
                data = json.loads(vectors_file.read_text(encoding='utf-8'))
                self._vectors = [
                    MarketFeatureVector.from_dict(item) for item in data
                ]
            except Exception as e:
                print(f"Warning: Could not load vectors: {e}")
        
        clusters_file = self.store_path / "clusters.json"
        if clusters_file.exists():
            try:
                data = json.loads(clusters_file.read_text(encoding='utf-8'))
                self._clusters = {
                    int(k): RegimeCluster(
                        cluster_id=v["cluster_id"],
                        centroid=np.array(v["centroid"]),
                        members=v["members"],
                        regime_type=v.get("regime_type"),
                        created_at=v.get("created_at", ""),
                        last_updated=v.get("last_updated", ""),
                    )
                    for k, v in data.items()
                }
            except Exception as e:
                print(f"Warning: Could not load clusters: {e}")
    
    def export_for_analysis(self, output_path: str) -> Path:
        """Export all data for external analysis."""
        with self._lock:
            output = Path(output_path)
            data = {
                "export_timestamp": self._get_timestamp(),
                "statistics": self.get_statistics(),
                "vectors": [v.to_dict() for v in self._vectors],
                "clusters": {
                    str(k): v.to_dict() for k, v in self._clusters.items()
                },
            }
            output.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return output


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store(
    store_path: str = "./memory/vectors/",
    max_vectors: int = 10_000,
    num_clusters: int = 50
) -> VectorStore:
    """
    Get or create the singleton VectorStore instance.
    
    Args:
        store_path: Directory for vector storage
        max_vectors: Maximum vectors in memory
        num_clusters: Target cluster count
        
    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(store_path, max_vectors, num_clusters)
    return _vector_store


if __name__ == "__main__":
    # Test the vector store
    store = get_vector_store()
    
    # Add some test vectors
    np.random.seed(42)
    for i in range(100):
        features = np.random.randn(12)
        asset = ["BTC", "ETH", "SOL"][i % 3]
        regime = ["bull_strong", "bear_weak", "ranging"][i % 3]
        
        store.add_vector(
            features=features,
            asset=asset,
            regime_label=regime,
            confidence=0.8 + np.random.random() * 0.2
        )
    
    print(f"Added vectors. Statistics: {store.get_statistics()}")
    
    # Test similarity search
    query = np.random.randn(12)
    similar = store.find_similar_regimes(query, asset="BTC", k=5)
    print(f"Found {len(similar)} similar BTC regimes")
    
    # Test classification
    regime, confidence = store.classify_regime(query, asset="BTC")
    print(f"Classified regime: {regime} (confidence: {confidence:.2f})")
