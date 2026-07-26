#!/usr/bin/env python3
"""
backend/orderflow/cluster_analyzer.py

Detects stacked imbalances and unfinished auctions in footprint charts.
Implements Observer pattern for real-time notifications and Strategy pattern
for different clustering algorithms.

Features:
- Stacked imbalance detection (consecutive price levels with extreme delta)
- Unfinished auction identification (high volume nodes with no clear winner)
- Memory-efficient sliding window analysis
- Strict type hinting for production reliability
- Cross-platform compatibility optimized for Windows PowerShell
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Protocol, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import time


class ImbalanceType(Enum):
    """Types of order flow imbalances detected."""
    STACKED_BID = auto()  # Multiple consecutive levels with bid dominance
    STACKED_ASK = auto()  # Multiple consecutive levels with ask dominance
    UNFINISHED_AUCTION = auto()  # High volume with unclear direction
    SINGLE_LEVEL_SPIKE = auto()  # Extreme imbalance at single level


@dataclass
class ImbalanceCluster:
    """Represents a detected cluster of imbalanced price levels."""
    imbalance_type: ImbalanceType
    start_price: float
    end_price: float
    levels: List[float] = field(default_factory=list)
    total_volume: float = 0.0
    avg_imbalance_ratio: float = 0.0
    max_imbalance_ratio: float = 0.0
    timestamp_ns: int = 0
    confidence_score: float = 0.0  # 0.0 to 1.0
    
    def duration_ticks(self) -> int:
        """Number of price levels in the cluster."""
        return len(self.levels)


@dataclass
class FootprintLevelData:
    """Single price level data for analysis."""
    price: float
    bid_volume: float
    ask_volume: float
    trade_count: int
    delta: float
    imbalance_ratio: float
    timestamp_ns: int


class ClusterStrategy(Protocol):
    """Strategy interface for different clustering algorithms."""
    
    def analyze(self, levels: List[FootprintLevelData]) -> List[ImbalanceCluster]:
        """Analyze levels and return detected clusters."""
        ...


class StackedImbalanceStrategy:
    """
    Detects stacked imbalances - consecutive price levels with extreme delta.
    A stacked imbalance requires at least 3 consecutive levels with >70% imbalance.
    """
    
    def __init__(self, min_consecutive: int = 3, threshold: float = 0.7):
        self.min_consecutive = min_consecutive
        self.threshold = threshold
    
    def analyze(self, levels: List[FootprintLevelData]) -> List[ImbalanceCluster]:
        if len(levels) < self.min_consecutive:
            return []
        
        clusters: List[ImbalanceCluster] = []
        current_cluster: List[FootprintLevelData] = []
        cluster_type: Optional[ImbalanceType] = None
        
        for level in levels:
            abs_imbalance = abs(level.imbalance_ratio)
            
            if abs_imbalance >= self.threshold:
                # Determine direction
                direction = (
                    ImbalanceType.STACKED_ASK 
                    if level.imbalance_ratio > 0 
                    else ImbalanceType.STACKED_BID
                )
                
                if cluster_type is None:
                    cluster_type = direction
                    current_cluster = [level]
                elif cluster_type == direction:
                    current_cluster.append(level)
                else:
                    # Direction changed, save previous cluster if valid
                    if len(current_cluster) >= self.min_consecutive:
                        clusters.append(self._create_cluster(current_cluster, cluster_type))
                    current_cluster = [level]
                    cluster_type = direction
            else:
                # Break in sequence
                if len(current_cluster) >= self.min_consecutive:
                    clusters.append(self._create_cluster(current_cluster, cluster_type))
                current_cluster = []
                cluster_type = None
        
        # Handle trailing cluster
        if len(current_cluster) >= self.min_consecutive:
            clusters.append(self._create_cluster(current_cluster, cluster_type))
        
        return clusters
    
    def _create_cluster(
        self, 
        levels: List[FootprintLevelData], 
        cluster_type: Optional[ImbalanceType]
    ) -> ImbalanceCluster:
        if not levels or cluster_type is None:
            raise ValueError("Invalid cluster parameters")
        
        total_vol = sum(l.bid_volume + l.ask_volume for l in levels)
        avg_imbalance = sum(l.imbalance_ratio for l in levels) / len(levels)
        max_imbalance = max(abs(l.imbalance_ratio) for l in levels)
        
        # Confidence based on length and strength
        length_factor = min(len(levels) / 5.0, 1.0)  # Cap at 5 levels
        strength_factor = avg_imbalance / self.threshold
        confidence = min((length_factor + strength_factor) / 2.0, 1.0)
        
        return ImbalanceCluster(
            imbalance_type=cluster_type,
            start_price=levels[0].price,
            end_price=levels[-1].price,
            levels=[l.price for l in levels],
            total_volume=total_vol,
            avg_imbalance_ratio=avg_imbalance,
            max_imbalance_ratio=max_imbalance,
            timestamp_ns=levels[0].timestamp_ns,
            confidence_score=confidence
        )


class UnfinishedAuctionStrategy:
    """
    Identifies unfinished auctions - high volume nodes where neither buyers
    nor sellers gained clear control (imbalance ratio between -0.3 and 0.3).
    These often act as magnetic price targets.
    """
    
    def __init__(
        self, 
        min_volume_percentile: float = 0.8,
        imbalance_threshold: float = 0.3
    ):
        self.min_volume_percentile = min_volume_percentile
        self.imbalance_threshold = imbalance_threshold
    
    def analyze(self, levels: List[FootprintLevelData]) -> List[ImbalanceCluster]:
        if len(levels) < 3:
            return []
        
        # Calculate volume percentile threshold
        volumes = sorted([l.bid_volume + l.ask_volume for l in levels])
        if not volumes:
            return []
        
        vol_threshold_idx = int(len(volumes) * self.min_volume_percentile)
        vol_threshold = volumes[min(vol_threshold_idx, len(volumes) - 1)]
        
        clusters: List[ImbalanceCluster] = []
        current_cluster: List[FootprintLevelData] = []
        
        for level in levels:
            total_vol = level.bid_volume + level.ask_volume
            abs_imbalance = abs(level.imbalance_ratio)
            
            if (total_vol >= vol_threshold and 
                abs_imbalance <= self.imbalance_threshold):
                current_cluster.append(level)
            else:
                if len(current_cluster) >= 2:  # At least 2 consecutive levels
                    clusters.append(self._create_cluster(current_cluster))
                current_cluster = []
        
        # Handle trailing cluster
        if len(current_cluster) >= 2:
            clusters.append(self._create_cluster(current_cluster))
        
        return clusters
    
    def _create_cluster(self, levels: List[FootprintLevelData]) -> ImbalanceCluster:
        total_vol = sum(l.bid_volume + l.ask_volume for l in levels)
        avg_imbalance = sum(l.imbalance_ratio for l in levels) / len(levels)
        
        return ImbalanceCluster(
            imbalance_type=ImbalanceType.UNFINISHED_AUCTION,
            start_price=levels[0].price,
            end_price=levels[-1].price,
            levels=[l.price for l in levels],
            total_volume=total_vol,
            avg_imbalance_ratio=avg_imbalance,
            max_imbalance_ratio=max(abs(l.imbalance_ratio) for l in levels),
            timestamp_ns=levels[0].timestamp_ns,
            confidence_score=min(len(levels) / 3.0, 1.0)
        )


class ClusterAnalyzer:
    """
    Main analyzer combining multiple strategies for comprehensive cluster detection.
    Implements Observer pattern for real-time notifications.
    """
    
    def __init__(self):
        self.strategies: Dict[str, ClusterStrategy] = {
            'stacked': StackedImbalanceStrategy(min_consecutive=3, threshold=0.7),
            'unfinished': UnfinishedAuctionStrategy(min_volume_percentile=0.8),
        }
        self.observers: List[Callable[[ImbalanceCluster], None]] = []
        self.level_buffer: deque[FootprintLevelData] = deque(maxlen=500)
        self.analysis_window_ns: int = 60_000_000_000  # 60 seconds in nanoseconds
    
    def add_observer(self, callback: Callable[[ImbalanceCluster], None]) -> None:
        """Register callback for new cluster detections."""
        self.observers.append(callback)
    
    def add_level(self, level: FootprintLevelData) -> None:
        """Add a new price level to the analysis buffer."""
        self.level_buffer.append(level)
    
    def add_levels(self, levels: List[FootprintLevelData]) -> None:
        """Batch add multiple price levels."""
        for level in levels:
            self.level_buffer.append(level)
    
    def analyze(self) -> List[ImbalanceCluster]:
        """
        Run all strategies on current buffer and return detected clusters.
        Notifies observers of new detections.
        """
        if len(self.level_buffer) < 10:
            return []
        
        levels_list = list(self.level_buffer)
        all_clusters: List[ImbalanceCluster] = []
        
        for name, strategy in self.strategies.items():
            clusters = strategy.analyze(levels_list)
            all_clusters.extend(clusters)
            
            # Notify observers
            for cluster in clusters:
                for observer in self.observers:
                    try:
                        observer(cluster)
                    except Exception as e:
                        # Log error but don't crash the analyzer
                        print(f"Observer error in {name}: {e}")
        
        return all_clusters
    
    def get_recent_clusters(
        self, 
        since_ns: Optional[int] = None,
        min_confidence: float = 0.5
    ) -> List[ImbalanceCluster]:
        """Get recent high-confidence clusters."""
        current_time_ns = time.time_ns()
        cutoff = since_ns if since_ns else (current_time_ns - self.analysis_window_ns)
        
        return [
            c for c in self.analyze()
            if c.timestamp_ns >= cutoff and c.confidence_score >= min_confidence
        ]
    
    def find_nearest_imbalance(
        self, 
        current_price: float,
        search_range: float = 50.0
    ) -> Optional[ImbalanceCluster]:
        """Find nearest significant imbalance to current price."""
        clusters = self.get_recent_clusters(min_confidence=0.6)
        
        nearest: Optional[ImbalanceCluster] = None
        min_distance = float('inf')
        
        for cluster in clusters:
            # Check if cluster is within search range
            if (cluster.start_price <= current_price + search_range and
                cluster.end_price >= current_price - search_range):
                
                distance = min(
                    abs(cluster.start_price - current_price),
                    abs(cluster.end_price - current_price)
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest = cluster
        
        return nearest


def create_level_data(
    price: float,
    bid_vol: float,
    ask_vol: float,
    trade_count: int = 1
) -> FootprintLevelData:
    """Helper factory function for creating level data."""
    total = bid_vol + ask_vol
    imbalance = (ask_vol - bid_vol) / total if total > 0 else 0.0
    delta = ask_vol - bid_vol
    
    return FootprintLevelData(
        price=price,
        bid_volume=bid_vol,
        ask_volume=ask_vol,
        trade_count=trade_count,
        delta=delta,
        imbalance_ratio=imbalance,
        timestamp_ns=time.time_ns()
    )


if __name__ == "__main__":
    # Example usage and testing
    analyzer = ClusterAnalyzer()
    
    # Simulate stacked ask imbalance
    for i in range(5):
        level = create_level_data(
            price=60000.0 + i * 0.01,
            bid_vol=1.0,
            ask_vol=10.0  # Strong ask dominance
        )
        analyzer.add_level(level)
    
    clusters = analyzer.analyze()
    print(f"Detected {len(clusters)} clusters:")
    for c in clusters:
        print(f"  {c.imbalance_type.name}: {c.start_price} - {c.end_price} "
              f"(confidence: {c.confidence_score:.2f})")
