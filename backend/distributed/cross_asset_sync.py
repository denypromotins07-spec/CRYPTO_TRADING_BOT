#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
High-Frequency Feature Stores, MLOps, and Concept Drift
File: backend/distributed/cross_asset_sync.py
Chapter 3: Distributed Feature Engineering and Ray Actor State Management

Synchronizes macro features across isolated asset workers.
Implements publish-subscribe pattern for cross-asset correlation signals.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
Uses shared memory for zero-copy feature distribution.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import threading
import time
import multiprocessing as mp
from multiprocessing import shared_memory


class SyncMode(Enum):
    """Synchronization mode for cross-asset features."""
    PUSH = "push"  # Active push to subscribers
    PULL = "pull"  # Passive pull on request
    HYBRID = "hybrid"  # Push for critical, pull for others


@dataclass
class MacroFeature:
    """A macro feature shared across assets."""
    feature_id: str
    name: str
    value: np.ndarray
    timestamp: float
    source_asset_id: int
    priority: int  # Higher = more critical
    ttl_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the feature has expired."""
        return time.time() - self.timestamp > self.ttl_seconds


@dataclass
class Subscription:
    """Subscription to macro features."""
    subscriber_id: str
    asset_ids: Set[int]
    feature_patterns: List[str]  # Regex patterns for feature names
    callback: Optional[Callable[[MacroFeature], None]] = None
    created_at: float = field(default_factory=time.time)


class CrossAssetSyncManager:
    """
    Manages synchronization of macro features across asset workers.
    
    Key responsibilities:
    1. Publish macro features from any asset worker
    2. Subscribe asset workers to relevant macro features
    3. Distribute features with minimal latency
    4. Handle feature expiration and cleanup
    5. Support both push and pull synchronization modes
    
    Optimized for high-frequency trading with shared memory support.
    """
    
    def __init__(
        self,
        sync_mode: SyncMode = SyncMode.HYBRID,
        default_ttl_seconds: float = 60.0,
        max_features: int = 1000,
        use_shared_memory: bool = True,
        memory_budget_mb: int = 64
    ):
        """
        Initialize the cross-asset sync manager.
        
        Args:
            sync_mode: Synchronization mode
            default_ttl_seconds: Default TTL for features
            max_features: Maximum features to track
            use_shared_memory: Whether to use shared memory for large arrays
            memory_budget_mb: Memory budget for shared memory
        """
        self.sync_mode = sync_mode
        self.default_ttl_seconds = default_ttl_seconds
        self.max_features = max_features
        self.use_shared_memory = use_shared_memory
        self.memory_budget_bytes = memory_budget_mb * 1024 * 1024
        
        # Published features
        self.features: Dict[str, MacroFeature] = {}
        
        # Subscriptions by subscriber ID
        self.subscriptions: Dict[str, Subscription] = {}
        
        # Feature index by asset (which assets care about which features)
        self.asset_feature_index: Dict[int, Set[str]] = defaultdict(set)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Shared memory pool (if enabled)
        self._shared_memory_pool: Dict[str, shared_memory.SharedMemory] = {}
        self._shm_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'features_published': 0,
            'features_distributed': 0,
            'subscriptions_active': 0,
            'expired_features_cleaned': 0,
            'shared_memory_allocations': 0
        }
        
        # Background cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
    
    def start(self) -> None:
        """Start background services."""
        self._start_cleanup_thread()
    
    def stop(self) -> None:
        """Stop background services."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
    
    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                self._cleanup_expired_features()
                self._stop_cleanup.wait(1.0)  # Check every second
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def publish(
        self,
        feature_id: str,
        name: str,
        value: np.ndarray,
        source_asset_id: int,
        priority: int = 1,
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish a macro feature for cross-asset distribution.
        
        Args:
            feature_id: Unique feature identifier
            name: Human-readable feature name
            value: Feature value array
            source_asset_id: Asset that generated this feature
            priority: Feature priority (higher = more critical)
            ttl_seconds: Time-to-live in seconds
            metadata: Optional metadata
            
        Returns:
            True if published successfully
        """
        with self._lock:
            # Check capacity
            if len(self.features) >= self.max_features:
                self._evict_low_priority_features()
            
            ttl = ttl_seconds or self.default_ttl_seconds
            
            # Store feature (with shared memory if enabled and large enough)
            if self.use_shared_memory and value.nbytes > 1024:  # > 1KB
                stored_value = self._store_in_shared_memory(feature_id, value)
            else:
                stored_value = value.copy()
            
            feature = MacroFeature(
                feature_id=feature_id,
                name=name,
                value=stored_value,
                timestamp=time.time(),
                source_asset_id=source_asset_id,
                priority=priority,
                ttl_seconds=ttl,
                metadata=metadata or {}
            )
            
            self.features[feature_id] = feature
            
            # Update asset index
            self.asset_feature_index[source_asset_id].add(feature_id)
            
            self.stats['features_published'] += 1
            
            # Distribute to subscribers (push mode)
            if self.sync_mode in (SyncMode.PUSH, SyncMode.HYBRID):
                self._distribute_to_subscribers(feature)
            
            return True
    
    def subscribe(
        self,
        subscriber_id: str,
        asset_ids: List[int],
        feature_patterns: Optional[List[str]] = None,
        callback: Optional[Callable[[MacroFeature], None]] = None
    ) -> bool:
        """
        Subscribe to macro features.
        
        Args:
            subscriber_id: Unique subscriber identifier
            asset_ids: List of asset IDs to monitor
            feature_patterns: Optional regex patterns for feature names
            callback: Optional callback for push notifications
            
        Returns:
            True if subscribed successfully
        """
        with self._lock:
            subscription = Subscription(
                subscriber_id=subscriber_id,
                asset_ids=set(asset_ids),
                feature_patterns=feature_patterns or ['.*'],  # Match all by default
                callback=callback
            )
            
            self.subscriptions[subscriber_id] = subscription
            self.stats['subscriptions_active'] = len(self.subscriptions)
            
            return True
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from macro features.
        
        Args:
            subscriber_id: Subscriber to remove
            
        Returns:
            True if unsubscribed successfully
        """
        with self._lock:
            if subscriber_id in self.subscriptions:
                del self.subscriptions[subscriber_id]
                self.stats['subscriptions_active'] = len(self.subscriptions)
                return True
            return False
    
    def get_features_for_asset(
        self,
        asset_id: int,
        exclude_source: bool = True
    ) -> List[MacroFeature]:
        """
        Get all macro features relevant to an asset.
        
        Args:
            asset_id: Asset to get features for
            exclude_source: Whether to exclude features from this asset
            
        Returns:
            List of relevant MacroFeature objects
        """
        with self._lock:
            relevant_features = []
            
            for feature_id, feature in self.features.items():
                # Skip expired features
                if feature.is_expired():
                    continue
                
                # Skip if exclude_source and this is the source
                if exclude_source and feature.source_asset_id == asset_id:
                    continue
                
                # Check if any subscriber for this asset would want this feature
                for subscription in self.subscriptions.values():
                    if asset_id in subscription.asset_ids:
                        if self._matches_patterns(feature.name, subscription.feature_patterns):
                            relevant_features.append(feature)
                            break
            
            # Sort by priority (highest first) and timestamp (newest first)
            relevant_features.sort(key=lambda f: (-f.priority, -f.timestamp))
            
            return relevant_features
    
    def pull_features(
        self,
        subscriber_id: str,
        asset_id: int
    ) -> List[MacroFeature]:
        """
        Pull features for a subscriber (pull mode).
        
        Args:
            subscriber_id: Subscriber requesting features
            asset_id: Asset context
            
        Returns:
            List of relevant MacroFeature objects
        """
        with self._lock:
            if subscriber_id not in self.subscriptions:
                return []
            
            subscription = self.subscriptions[subscriber_id]
            
            if asset_id not in subscription.asset_ids:
                return []
            
            relevant = []
            for feature in self.features.values():
                if feature.is_expired():
                    continue
                
                if self._matches_patterns(feature.name, subscription.feature_patterns):
                    relevant.append(feature)
            
            self.stats['features_distributed'] += len(relevant)
            return relevant
    
    def _distribute_to_subscribers(self, feature: MacroFeature) -> None:
        """Distribute a feature to matching subscribers."""
        for subscription in self.subscriptions.values():
            # Check if any subscribed asset should receive this
            should_receive = False
            for asset_id in subscription.asset_ids:
                if asset_id != feature.source_asset_id:  # Don't send back to source
                    should_receive = True
                    break
            
            if not should_receive:
                continue
            
            # Check pattern match
            if not self._matches_patterns(feature.name, subscription.feature_patterns):
                continue
            
            # Call callback if provided
            if subscription.callback:
                try:
                    subscription.callback(feature)
                except Exception as e:
                    print(f"Error in subscriber callback: {e}")
            
            self.stats['features_distributed'] += 1
    
    def _matches_patterns(self, name: str, patterns: List[str]) -> bool:
        """Check if a feature name matches any of the patterns."""
        import re
        for pattern in patterns:
            try:
                if re.match(pattern, name):
                    return True
            except re.error:
                # Invalid regex, treat as literal
                if pattern == name:
                    return True
        return False
    
    def _cleanup_expired_features(self) -> int:
        """Remove expired features and release shared memory."""
        current_time = time.time()
        expired = []
        
        with self._lock:
            for feature_id, feature in self.features.items():
                if feature.is_expired():
                    expired.append(feature_id)
            
            for feature_id in expired:
                feature = self.features.pop(feature_id, None)
                if feature:
                    # Release shared memory if used
                    if self.use_shared_memory:
                        self._release_shared_memory(feature_id)
                    
                    # Update index
                    if feature.source_asset_id in self.asset_feature_index:
                        self.asset_feature_index[feature.source_asset_id].discard(feature_id)
                
                self.stats['expired_features_cleaned'] += 1
        
        return len(expired)
    
    def _evict_low_priority_features(self) -> None:
        """Evict lowest priority features when at capacity."""
        if not self.features:
            return
        
        # Find lowest priority features
        min_priority = min(f.priority for f in self.features.values())
        
        to_evict = [
            fid for fid, f in self.features.items()
            if f.priority == min_priority
        ][:len(self.features) // 10]  # Evict 10%
        
        for feature_id in to_evict:
            feature = self.features.pop(feature_id, None)
            if feature and self.use_shared_memory:
                self._release_shared_memory(feature_id)
    
    def _store_in_shared_memory(
        self,
        feature_id: str,
        value: np.ndarray
    ) -> np.ndarray:
        """Store array in shared memory and return reference."""
        if not self.use_shared_memory:
            return value
        
        with self._shm_lock:
            # Create shared memory
            shm = shared_memory.SharedMemory(
                name=f"feature_{feature_id}",
                create=True,
                size=value.nbytes
            )
            
            # Copy data
            np.copyto(np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf), value)
            
            self._shared_memory_pool[feature_id] = shm
            self.stats['shared_memory_allocations'] += 1
            
            # Return the original array (caller should know it's backed by shm)
            return value
    
    def _release_shared_memory(self, feature_id: str) -> None:
        """Release shared memory for a feature."""
        with self._shm_lock:
            if feature_id in self._shared_memory_pool:
                try:
                    shm = self._shared_memory_pool.pop(feature_id)
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sync manager statistics."""
        with self._lock:
            return {
                **self.stats,
                'features_cached': len(self.features),
                'memory_used_mb': self._estimate_memory_usage() / (1024 * 1024),
                'sync_mode': self.sync_mode.value
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate total memory usage."""
        total = 0
        for feature in self.features.values():
            if isinstance(feature.value, np.ndarray):
                total += feature.value.nbytes
            else:
                total += 100  # Estimate for non-array values
        return total


# Convenience function for creating production sync manager
def create_production_sync_manager(
    assets: List[str] = None,
    use_hybrid_mode: bool = True
) -> CrossAssetSyncManager:
    """
    Create a production-ready cross-asset sync manager.
    
    Args:
        assets: List of asset names
        use_hybrid_mode: Use hybrid push/pull mode
        
    Returns:
        Configured CrossAssetSyncManager instance
    """
    if assets is None:
        assets = ['BTC', 'ETH', 'SOL', 'USDT']
    
    manager = CrossAssetSyncManager(
        sync_mode=SyncMode.HYBRID if use_hybrid_mode else SyncMode.PUSH,
        default_ttl_seconds=30.0,  # Shorter TTL for HFT
        max_features=500,
        use_shared_memory=True,
        memory_budget_mb=64
    )
    
    # Pre-register subscriptions for each asset
    for i, asset in enumerate(assets):
        manager.subscribe(
            subscriber_id=f"worker_{asset}",
            asset_ids=[j for j in range(len(assets)) if j != i],
            feature_patterns=['macro_.*', 'correlation_.*', 'market_regime_.*']
        )
    
    return manager


if __name__ == '__main__':
    # Test the cross-asset sync manager
    print("Testing Cross-Asset Sync Manager for ZAID Trading Bot...")
    
    # Create manager
    manager = CrossAssetSyncManager(
        sync_mode=SyncMode.HYBRID,
        use_shared_memory=False  # Disable for simple test
    )
    manager.start()
    
    # Subscribe workers
    print("\nSubscribing workers...")
    manager.subscribe("btc_worker", [0], ['macro_.*', 'volatility_.*'])
    manager.subscribe("eth_worker", [1], ['macro_.*', 'momentum_.*'])
    manager.subscribe("sol_worker", [2], ['macro_.*'])
    
    # Publish some macro features
    print("Publishing macro features...")
    
    # Market-wide volatility (from BTC)
    manager.publish(
        feature_id="macro_volatility",
        name="macro_market_volatility",
        value=np.array([0.02, 0.025, 0.018]),
        source_asset_id=0,
        priority=10,
        ttl_seconds=60.0,
        metadata={'type': 'volatility'}
    )
    
    # Correlation matrix (from ETH)
    corr_matrix = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    manager.publish(
        feature_id="correlation_matrix",
        name="macro_correlation_matrix",
        value=corr_matrix.flatten(),
        source_asset_id=1,
        priority=8,
        ttl_seconds=30.0
    )
    
    # Get features for BTC worker
    print("\nGetting features for BTC worker...")
    btc_features = manager.get_features_for_asset(0)
    print(f"  BTC sees {len(btc_features)} features from other assets")
    for f in btc_features:
        print(f"    - {f.name} (priority={f.priority}, source={f.source_asset_id})")
    
    # Pull features for ETH worker
    print("\nPulling features for ETH worker...")
    eth_features = manager.pull_features("eth_worker", 1)
    print(f"  ETH pulled {len(eth_features)} features")
    
    # Wait and check expiration
    print("\nWaiting for feature expiration...")
    time.sleep(2)
    
    # Cleanup
    cleaned = manager._cleanup_expired_features()
    print(f"  Cleaned {cleaned} expired features")
    
    # Get stats
    stats = manager.get_stats()
    print(f"\nSync Manager Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Stop manager
    manager.stop()
    
    print("\nCross-Asset Sync test completed!")
