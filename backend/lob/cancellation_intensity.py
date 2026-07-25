#!/usr/bin/env python3
"""
Cancellation Intensity Analyzer

This module analyzes the ratio of cancellations to executions in the order book,
providing insights into market participant behavior and potential manipulation.

Key Features:
- Real-time cancellation intensity tracking
- Order lifetime distribution analysis
- Spoofing pattern detection via cancellation clustering
- Memory-efficient sliding window calculations

Author: ZAID Personal Crypto Trading Bot
Stage: 22 - LOB Physics & Hawkes Processes
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
from threading import Lock


class CancellationType(Enum):
    """Classification of cancellation patterns."""
    NORMAL = "normal"  # Typical market making cancellation
    RAPID = "rapid"  # Very fast cancellation (<100ms)
    CLUSTERED = "clustered"  # Multiple cancellations together
    LAYERED = "layered"  # Sequential price level cancellations
    MASS = "mass"  # Large-scale cancellation event


@dataclass
class OrderRecord:
    """Record of an order for tracking lifecycle."""
    order_id: str
    timestamp: float  # Placement time
    side: str  # 'bid' or 'ask'
    price: float
    volume: float
    cancel_time: Optional[float] = None
    fill_time: Optional[float] = None
    fill_volume: float = 0.0
    is_cancelled: bool = False
    is_filled: bool = False


@dataclass
class CancellationMetrics:
    """Current cancellation metrics snapshot."""
    cancellation_ratio: float  # Cancels / (Cancels + Fills)
    cancel_intensity: float  # Cancels per second
    fill_intensity: float  # Fills per second
    avg_order_lifetime_ms: float
    rapid_cancel_ratio: float  # % cancelled within 100ms
    clustered_cancels: int  # Recent cluster count
    dominant_type: CancellationType
    spoofing_indicator: float  # 0-1 probability of spoofing
    last_update: float


@dataclass  
class AssetCancellationParams:
    """Asset-specific cancellation parameters."""
    normal_cancel_ratio: float  # Expected cancellation ratio
    rapid_threshold_ms: float  # Threshold for rapid cancellation
    cluster_window_ms: float  # Time window for cluster detection
    min_cluster_size: int  # Minimum cancels for cluster
    
    @classmethod
    def default_params(cls, asset: str) -> 'AssetCancellationParams':
        """Get default parameters for common assets."""
        params = {
            "BTC": cls(
                normal_cancel_ratio=0.60,  # 60% typical cancel rate
                rapid_threshold_ms=100,
                cluster_window_ms=50,
                min_cluster_size=5,
            ),
            "ETH": cls(
                normal_cancel_ratio=0.65,
                rapid_threshold_ms=80,
                cluster_window_ms=40,
                min_cluster_size=4,
            ),
            "SOL": cls(
                normal_cancel_ratio=0.70,
                rapid_threshold_ms=50,
                cluster_window_ms=30,
                min_cluster_size=3,
            ),
            "DEFAULT": cls(
                normal_cancel_ratio=0.60,
                rapid_threshold_ms=100,
                cluster_window_ms=50,
                min_cluster_size=5,
            ),
        }
        return params.get(asset.upper(), params["DEFAULT"])


class CancellationIntensityAnalyzer:
    """
    Analyzes cancellation patterns and intensity in the order book.
    
    Tracks order lifecycles to identify normal vs suspicious cancellation
    behavior that may indicate spoofing or layering manipulation.
    """
    
    def __init__(
        self,
        asset: str = "BTC",
        window_size: int = 10000,
        tracking_window_s: float = 60.0,
    ):
        """
        Initialize cancellation analyzer.
        
        Args:
            asset: Asset symbol for parameter calibration
            window_size: Maximum orders to track
            tracking_window_s: Time window for intensity calculation
        """
        self.asset = asset.upper()
        self.params = AssetCancellationParams.default_params(self.asset)
        self.window_size = window_size
        self.tracking_window_s = tracking_window_s
        
        # Order tracking
        self._active_orders: Dict[str, OrderRecord] = {}
        self._completed_orders: Deque[OrderRecord] = deque(maxlen=window_size)
        
        # Event tracking for intensity
        self._recent_cancels: Deque[float] = deque(maxlen=window_size)
        self._recent_fills: Deque[float] = deque(maxlen=window_size)
        
        # Cluster tracking
        self._cancel_timestamps: Deque[float] = deque(maxlen=1000)
        
        # Statistics
        self._total_cancels: int = 0
        self._total_fills: int = 0
        self._rapid_cancels: int = 0
        
        # Thread safety
        self._lock = Lock()
        
        # Exponential decay factor
        self._decay_factor = 0.99
    
    def record_order_placement(
        self,
        order_id: str,
        side: str,
        price: float,
        volume: float,
    ) -> None:
        """Record a new order placement."""
        with self._lock:
            now = time.time()
            order = OrderRecord(
                order_id=order_id,
                timestamp=now,
                side=side.lower(),
                price=price,
                volume=volume,
            )
            self._active_orders[order_id] = order
    
    def record_order_cancellation(self, order_id: str) -> Optional[OrderRecord]:
        """
        Record an order cancellation.
        
        Returns:
            The completed order record if found
        """
        with self._lock:
            now = time.time()
            
            if order_id not in self._active_orders:
                return None
            
            order = self._active_orders.pop(order_id)
            order.cancel_time = now
            order.is_cancelled = True
            
            # Track for intensity calculation
            self._recent_cancels.append(now)
            self._cancel_timestamps.append(now)
            
            # Check for rapid cancellation
            lifetime_ms = (now - order.timestamp) * 1000
            if lifetime_ms < self.params.rapid_threshold_ms:
                self._rapid_cancels += 1
            
            self._total_cancels += 1
            self._completed_orders.append(order)
            
            return order
    
    def record_order_fill(
        self,
        order_id: str,
        fill_volume: float,
        fill_price: Optional[float] = None,
    ) -> Optional[OrderRecord]:
        """
        Record an order fill (partial or complete).
        
        Returns:
            The completed order record if fully filled
        """
        with self._lock:
            now = time.time()
            
            if order_id not in self._active_orders:
                return None
            
            order = self._active_orders[order_id]
            order.fill_volume += fill_volume
            order.fill_time = now
            
            # Check if fully filled
            if order.fill_volume >= order.volume * 0.99:  # 99% threshold
                order.is_filled = True
                self._active_orders.pop(order_id)
                
                self._recent_fills.append(now)
                self._total_fills += 1
                self._completed_orders.append(order)
                
                return order
            
            return None
    
    def calculate_metrics(self) -> CancellationMetrics:
        """
        Calculate current cancellation metrics.
        
        Returns:
            CancellationMetrics with current state
        """
        with self._lock:
            now = time.time()
            
            # Clean old events outside tracking window
            cutoff = now - self.tracking_window_s
            while self._recent_cancels and self._recent_cancels[0] < cutoff:
                self._recent_cancels.popleft()
            while self._recent_fills and self._recent_fills[0] < cutoff:
                self._recent_fills.popleft()
            
            # Calculate intensities (events per second)
            cancel_intensity = len(self._recent_cancels) / self.tracking_window_s
            fill_intensity = len(self._recent_fills) / max(1, self.tracking_window_s)
            
            # Cancellation ratio
            total_events = self._total_cancels + self._total_fills
            if total_events > 0:
                cancellation_ratio = self._total_cancels / total_events
            else:
                cancellation_ratio = 0.0
            
            # Average order lifetime
            lifetimes = []
            for order in self._completed_orders:
                if order.cancel_time:
                    lifetime = (order.cancel_time - order.timestamp) * 1000
                    lifetimes.append(lifetime)
                elif order.fill_time:
                    lifetime = (order.fill_time - order.timestamp) * 1000
                    lifetimes.append(lifetime)
            
            avg_lifetime = np.mean(lifetimes) if lifetimes else 0.0
            
            # Rapid cancel ratio
            recent_total = len(self._recent_cancels)
            rapid_ratio = self._rapid_cancels / max(1, self._total_cancels)
            
            # Detect clusters
            clustered_cancels = self._detect_clusters(now)
            
            # Determine dominant type
            dominant_type = self._classify_type(
                rapid_ratio, clustered_cancels, cancellation_ratio
            )
            
            # Spoofing indicator
            spoofing_prob = self._calculate_spoofing_probability(
                rapid_ratio, clustered_cancels, cancellation_ratio
            )
            
            return CancellationMetrics(
                cancellation_ratio=cancellation_ratio,
                cancel_intensity=cancel_intensity,
                fill_intensity=fill_intensity,
                avg_order_lifetime_ms=avg_lifetime,
                rapid_cancel_ratio=rapid_ratio,
                clustered_cancels=clustered_cancels,
                dominant_type=dominant_type,
                spoofing_indicator=spoofing_prob,
                last_update=now,
            )
    
    def _detect_clusters(self, now: float) -> int:
        """Detect cancellation clusters in recent history."""
        if len(self._cancel_timestamps) < self.params.min_cluster_size:
            return 0
        
        clusters = 0
        window_s = self.params.cluster_window_ms / 1000.0
        
        timestamps = list(self._cancel_timestamps)
        i = 0
        while i < len(timestamps):
            cluster_start = timestamps[i]
            cluster_size = 1
            j = i + 1
            
            while j < len(timestamps) and timestamps[j] - cluster_start < window_s:
                cluster_size += 1
                j += 1
            
            if cluster_size >= self.params.min_cluster_size:
                clusters += 1
                i = j  # Skip past this cluster
            else:
                i += 1
        
        return clusters
    
    def _classify_type(
        self,
        rapid_ratio: float,
        clustered_cancels: int,
        cancel_ratio: float,
    ) -> CancellationType:
        """Classify the dominant cancellation type."""
        if clustered_cancels >= 3:
            return CancellationType.CLUSTERED
        
        if rapid_ratio > 0.5:
            return CancellationType.RAPID
        
        if cancel_ratio > self.params.normal_cancel_ratio * 1.5:
            return CancellationType.MASS
        
        return CancellationType.NORMAL
    
    def _calculate_spoofing_probability(
        self,
        rapid_ratio: float,
        clustered_cancels: int,
        cancel_ratio: float,
    ) -> float:
        """Calculate probability of spoofing behavior."""
        prob = 0.0
        
        # High rapid cancellation ratio is suspicious
        if rapid_ratio > 0.3:
            prob += 0.3
        elif rapid_ratio > 0.1:
            prob += 0.15
        
        # Clustering indicates potential layering
        if clustered_cancels >= 5:
            prob += 0.4
        elif clustered_cancels >= 2:
            prob += 0.2
        
        # Abnormally high cancel ratio
        if cancel_ratio > 0.8:
            prob += 0.3
        elif cancel_ratio > self.params.normal_cancel_ratio:
            prob += 0.1
        
        return min(1.0, prob)
    
    def get_order_lifetime_distribution(self) -> np.ndarray:
        """Get histogram of order lifetimes."""
        with self._lock:
            lifetimes = []
            for order in self._completed_orders:
                if order.cancel_time:
                    lifetime = (order.cancel_time - order.timestamp) * 1000
                    lifetimes.append(lifetime)
                elif order.fill_time:
                    lifetime = (order.fill_time - order.timestamp) * 1000
                    lifetimes.append(lifetime)
            
            if not lifetimes:
                return np.array([])
            
            return np.histogram(
                lifetimes,
                bins=np.logspace(0, 4, 50),  # 1ms to 10s
            )[0]
    
    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._active_orders.clear()
            self._completed_orders.clear()
            self._recent_cancels.clear()
            self._recent_fills.clear()
            self._cancel_timestamps.clear()
            self._total_cancels = 0
            self._total_fills = 0
            self._rapid_cancels = 0
    
    def update_asset(self, asset: str) -> None:
        """Update asset class for recalibration."""
        with self._lock:
            self.asset = asset.upper()
            self.params = AssetCancellationParams.default_params(self.asset)


if __name__ == "__main__":
    # Demo usage
    analyzer = CancellationIntensityAnalyzer(asset="BTC")
    
    # Simulate some orders
    import random
    
    for i in range(100):
        order_id = f"order_{i}"
        analyzer.record_order_placement(
            order_id=order_id,
            side=random.choice(["bid", "ask"]),
            price=50000.0 + random.uniform(-100, 100),
            volume=random.uniform(0.1, 10.0),
        )
        
        # Randomly cancel or fill
        if random.random() < 0.6:
            analyzer.record_order_cancellation(order_id)
        else:
            analyzer.record_order_fill(order_id, random.uniform(0.1, 5.0))
    
    metrics = analyzer.calculate_metrics()
    print(f"Cancellation Ratio: {metrics.cancellation_ratio:.2%}")
    print(f"Cancel Intensity: {metrics.cancel_intensity:.2f}/s")
    print(f"Avg Lifetime: {metrics.avg_order_lifetime_ms:.1f}ms")
    print(f"Rapid Cancel Ratio: {metrics.rapid_cancel_ratio:.2%}")
    print(f"Dominant Type: {metrics.dominant_type.value}")
    print(f"Spoofing Indicator: {metrics.spoofing_indicator:.2f}")
