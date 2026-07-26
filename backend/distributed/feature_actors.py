#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
High-Frequency Feature Stores, MLOps, and Concept Drift
File: backend/distributed/feature_actors.py
Chapter 3: Distributed Feature Engineering and Ray Actor State Management

Spawns parallel Ray workers for cross-asset math operations.
Ensures Ray actors do not cause garbage collection pauses during the 4hr window.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
Uses Ray's actor model for distributed feature computation.
"""

from __future__ import annotations
import ray
from ray import actor
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
import threading
from collections import deque


# Initialize Ray with memory constraints for 8GB system
def initialize_ray(
    num_cpus: int = 4,
    object_store_memory_mb: int = 2048,
    heap_size_mb: int = 1024
) -> None:
    """
    Initialize Ray with constrained resources for 8GB system.
    
    Args:
        num_cpus: Number of CPUs to use
        object_store_memory_mb: Memory for Ray object store
        heap_size_mb: Heap size per worker
    """
    if not ray.is_initialized():
        ray.init(
            num_cpus=num_cpus,
            object_store_memory=object_store_memory_mb * 1024 * 1024,
            _system_config={
                "max_huge_pages": 0,  # Disable huge pages to reduce GC pressure
                "allocator_policy": "hybrid"
            }
        )


@dataclass
class FeatureComputeResult:
    """Result from a feature computation task."""
    asset_id: int
    feature_name: str
    values: np.ndarray
    timestamp: float
    compute_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@ray.remote(num_cpus=1, max_restarts=3)
class FeatureActor:
    """
    Ray actor for computing features on a specific asset.
    
    Each actor is responsible for:
    - Computing technical indicators for its assigned asset
    - Maintaining local state for rolling calculations
    - Synchronizing macro features with other actors
    - Managing memory to prevent GC pauses
    """
    
    def __init__(
        self,
        asset_id: int,
        asset_name: str,
        feature_configs: List[Dict[str, Any]],
        memory_limit_mb: int = 256
    ):
        """
        Initialize a feature actor.
        
        Args:
            asset_id: Unique asset identifier
            asset_name: Human-readable asset name (BTC, ETH, etc.)
            feature_configs: List of feature computation configurations
            memory_limit_mb: Memory limit for this actor
        """
        self.asset_id = asset_id
        self.asset_name = asset_name
        self.feature_configs = feature_configs
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Local state for rolling calculations
        self.price_history: deque = deque(maxlen=10000)
        self.volume_history: deque = deque(maxlen=10000)
        self.feature_cache: Dict[str, deque] = {}
        
        # Initialize feature caches
        for config in feature_configs:
            feature_name = config.get('name', 'unknown')
            self.feature_cache[feature_name] = deque(maxlen=10000)
        
        # Statistics
        self.compute_count = 0
        self.total_compute_time_ns = 0
        self.last_gc_time = time.time()
        
        # Pre-allocate arrays to reduce allocations
        self._temp_array = np.zeros(1000, dtype=np.float64)
    
    def update_tick(self, price: float, volume: float, timestamp: float) -> None:
        """
        Update with a new tick.
        
        Args:
            price: Current price
            volume: Current volume
            timestamp: Tick timestamp
        """
        self.price_history.append((timestamp, price))
        self.volume_history.append((timestamp, volume))
        
        # Periodic cleanup to prevent memory buildup
        if len(self.price_history) % 1000 == 0:
            self._cleanup_if_needed()
    
    def compute_features(
        self,
        window_size: int = 100
    ) -> List[FeatureComputeResult]:
        """
        Compute all configured features.
        
        Args:
            window_size: Lookback window for calculations
            
        Returns:
            List of FeatureComputeResult objects
        """
        start_time = time.perf_counter_ns()
        results = []
        
        if len(self.price_history) < window_size:
            return results
        
        # Extract price and volume arrays (zero-copy where possible)
        prices = np.array([p for _, p in self.price_history[-window_size:]])
        volumes = np.array([v for _, v in self.volume_history[-window_size:]])
        timestamps = np.array([t for t, _ in self.price_history[-window_size:]])
        
        for config in self.feature_configs:
            feature_name = config.get('name', 'unknown')
            feature_type = config.get('type', 'simple')
            
            try:
                values = self._compute_feature(
                    feature_type,
                    prices,
                    volumes,
                    config.get('params', {})
                )
                
                # Cache the result
                self.feature_cache[feature_name].extend(values)
                
                compute_time = (time.perf_counter_ns() - start_time) / 1e6
                self.compute_count += 1
                
                results.append(FeatureComputeResult(
                    asset_id=self.asset_id,
                    feature_name=feature_name,
                    values=values,
                    timestamp=timestamps[-1],
                    compute_time_ms=compute_time
                ))
            except Exception as e:
                # Log error but continue with other features
                print(f"Error computing {feature_name}: {e}")
        
        self.total_compute_time_ns += time.perf_counter_ns() - start_time
        return results
    
    def _compute_feature(
        self,
        feature_type: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute a specific feature type.
        
        Args:
            feature_type: Type of feature to compute
            prices: Price array
            volumes: Volume array
            params: Feature-specific parameters
            
        Returns:
            Computed feature values
        """
        if feature_type == 'returns':
            period = params.get('period', 1)
            return np.diff(prices) / prices[:-period]
        
        elif feature_type == 'volatility':
            period = params.get('period', 20)
            returns = np.diff(prices) / prices[:-1]
            return np.array([
                np.std(returns[max(0, i-period):i]) 
                for i in range(len(returns))
            ])
        
        elif feature_type == 'volume_profile':
            # Volume-weighted price distribution
            n_bins = params.get('bins', 10)
            price_range = prices.max() - prices.min()
            if price_range == 0:
                return np.zeros(n_bins)
            
            bins = np.linspace(prices.min(), prices.max(), n_bins + 1)
            volume_by_bin = np.zeros(n_bins)
            
            for i in range(len(prices)):
                bin_idx = min(
                    int((prices[i] - prices.min()) / price_range * n_bins),
                    n_bins - 1
                )
                volume_by_bin[bin_idx] += volumes[i]
            
            return volume_by_bin / (volume_by_bin.sum() + 1e-10)
        
        elif feature_type == 'momentum':
            period = params.get('period', 10)
            if len(prices) < period:
                return np.array([0.0])
            return np.array([(prices[-1] / prices[-period]) - 1])
        
        elif feature_type == 'order_flow_imbalance':
            # Simplified order flow imbalance
            buy_volume = volumes * (np.random.random(len(volumes)) > 0.5)
            sell_volume = volumes - buy_volume
            imbalance = (buy_volume - sell_volume) / (volumes + 1e-10)
            return imbalance
        
        else:
            # Default: simple moving average
            period = params.get('period', 20)
            return np.convolve(prices, np.ones(period)/period, mode='valid')
    
    def get_feature_value(
        self,
        feature_name: str,
        lag: int = 0
    ) -> Optional[float]:
        """
        Get the current value of a specific feature.
        
        Args:
            feature_name: Name of the feature
            lag: Number of periods to look back
            
        Returns:
            Feature value or None if not available
        """
        if feature_name not in self.feature_cache:
            return None
        
        cache = self.feature_cache[feature_name]
        if len(cache) <= lag:
            return None
        
        return cache[-(lag + 1)]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of actor state for monitoring."""
        return {
            'asset_id': self.asset_id,
            'asset_name': self.asset_name,
            'price_history_len': len(self.price_history),
            'features_cached': len(self.feature_cache),
            'compute_count': self.compute_count,
            'avg_compute_time_ms': (
                (self.total_compute_time_ns / 1e6) / self.compute_count
                if self.compute_count > 0 else 0
            ),
            'memory_estimate_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
    
    def _cleanup_if_needed(self) -> None:
        """Perform cleanup to prevent memory buildup and GC pauses."""
        current_time = time.time()
        
        # Force garbage collection every 60 seconds
        if current_time - self.last_gc_time > 60:
            import gc
            gc.collect()
            self.last_gc_time = current_time
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of this actor."""
        total = 0
        
        # Price/volume history
        total += len(self.price_history) * 16  # tuple overhead
        total += len(self.volume_history) * 16
        
        # Feature caches
        for cache in self.feature_cache.values():
            total += len(cache) * 8
        
        # Temp array
        total += self._temp_array.nbytes
        
        return total
    
    def reset(self) -> None:
        """Reset actor state (for retraining cycles)."""
        self.price_history.clear()
        self.volume_history.clear()
        for cache in self.feature_cache.values():
            cache.clear()
        self.compute_count = 0
        self.total_compute_time_ns = 0


class FeatureActorManager:
    """
    Manager for spawning and coordinating feature actors.
    
    Handles:
    - Actor lifecycle management
    - Load balancing across assets
    - Result aggregation
    - Memory monitoring
    """
    
    def __init__(
        self,
        max_actors: int = 4,
        memory_per_actor_mb: int = 256
    ):
        """
        Initialize the actor manager.
        
        Args:
            max_actors: Maximum number of actors to spawn
            memory_per_actor_mb: Memory limit per actor
        """
        self.max_actors = max_actors
        self.memory_per_actor_mb = memory_per_actor_mb
        self.actors: Dict[int, ray.actor.ActorHandle] = {}
        self.actor_refs: Dict[int, Any] = {}
        
        # Ensure Ray is initialized
        initialize_ray()
    
    def spawn_actor(
        self,
        asset_id: int,
        asset_name: str,
        feature_configs: List[Dict[str, Any]]
    ) -> ray.actor.ActorHandle:
        """
        Spawn a new feature actor.
        
        Args:
            asset_id: Asset identifier
            asset_name: Asset name
            feature_configs: Feature configurations
            
        Returns:
            Ray actor handle
        """
        if len(self.actors) >= self.max_actors:
            raise RuntimeError(f"Maximum actors ({self.max_actors}) reached")
        
        actor_handle = FeatureActor.options(
            name=f"feature_actor_{asset_name}",
            lifetime="detached"
        ).remote(
            asset_id=asset_id,
            asset_name=asset_name,
            feature_configs=feature_configs,
            memory_limit_mb=self.memory_per_actor_mb
        )
        
        self.actors[asset_id] = actor_handle
        return actor_handle
    
    async def compute_all_features(
        self,
        window_size: int = 100
    ) -> Dict[int, List[FeatureComputeResult]]:
        """
        Trigger feature computation on all actors.
        
        Args:
            window_size: Lookback window
            
        Returns:
            Dictionary mapping asset_id to list of results
        """
        tasks = [
            actor.compute_features.remote(window_size)
            for actor in self.actors.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        aggregated = {}
        for i, (asset_id, result) in enumerate(zip(self.actors.keys(), results)):
            if isinstance(result, Exception):
                print(f"Error computing features for asset {asset_id}: {result}")
                aggregated[asset_id] = []
            else:
                aggregated[asset_id] = result
        
        return aggregated
    
    def get_all_states(self) -> Dict[int, Dict[str, Any]]:
        """Get state summaries from all actors."""
        states = {}
        for asset_id, actor in self.actors.items():
            try:
                states[asset_id] = ray.get(actor.get_state_summary.remote())
            except Exception as e:
                states[asset_id] = {'error': str(e)}
        return states
    
    def shutdown(self) -> None:
        """Shutdown all actors and release resources."""
        for actor in self.actors.values():
            try:
                ray.kill(actor)
            except Exception:
                pass
        self.actors.clear()


# Import asyncio for async operations
import asyncio


if __name__ == '__main__':
    # Test the feature actors
    print("Testing Feature Actors for ZAID Trading Bot...")
    
    # Initialize Ray
    initialize_ray(num_cpus=2, object_store_memory_mb=512)
    
    # Create manager
    manager = FeatureActorManager(max_actors=2, memory_per_actor_mb=128)
    
    # Define feature configs
    btc_features = [
        {'name': 'returns_1', 'type': 'returns', 'params': {'period': 1}},
        {'name': 'volatility_20', 'type': 'volatility', 'params': {'period': 20}},
        {'name': 'momentum_10', 'type': 'momentum', 'params': {'period': 10}},
    ]
    
    eth_features = [
        {'name': 'returns_1', 'type': 'returns', 'params': {'period': 1}},
        {'name': 'volume_profile', 'type': 'volume_profile', 'params': {'bins': 10}},
    ]
    
    # Spawn actors
    print("\nSpawning actors...")
    manager.spawn_actor(0, 'BTC', btc_features)
    manager.spawn_actor(1, 'ETH', eth_features)
    
    # Simulate ticks
    print("Simulating ticks...")
    actor_btc = manager.actors[0]
    actor_eth = manager.actors[1]
    
    base_price_btc = 50000
    base_price_eth = 3000
    
    for i in range(200):
        # Random walk prices
        base_price_btc *= (1 + np.random.normal(0, 0.001))
        base_price_eth *= (1 + np.random.normal(0, 0.002))
        
        volume_btc = np.random.uniform(10, 100)
        volume_eth = np.random.uniform(50, 500)
        
        ray.get(actor_btc.update_tick.remote(base_price_btc, volume_btc, time.time()))
        ray.get(actor_eth.update_tick.remote(base_price_eth, volume_eth, time.time()))
    
    # Compute features
    print("\nComputing features...")
    results = ray.get(actor_btc.compute_features.remote())
    
    print(f"\nBTC Feature Results:")
    for result in results:
        print(f"  {result.feature_name}: {len(result.values)} values, "
              f"compute time: {result.compute_time_ms:.2f}ms")
    
    # Get state summaries
    states = manager.get_all_states()
    print(f"\nActor States:")
    for asset_id, state in states.items():
        print(f"  Asset {asset_id}: {state}")
    
    # Cleanup
    manager.shutdown()
    ray.shutdown()
    
    print("\nFeature Actors test completed!")
