#!/usr/bin/env python3
"""
Liquidation Heatmap Estimator

Estimates retail liquidation clusters based on leverage distribution,
position data, and price levels. Helps identify potential cascade zones
where mass liquidations could trigger volatility spikes.

Features:
- Liquidation price estimation for leveraged positions
- Cluster detection for high-density liquidation zones
- Cascade risk scoring
- Real-time heatmap updates

Target: Estimate retail liquidation clusters to anticipate market moves.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LiquidationCluster:
    """Represents a cluster of liquidation orders at a price level"""
    price_level: float
    total_notional: float  # Total USD value of positions that would liquidate
    position_count: int
    avg_leverage: float
    side: str  # 'long' or 'short'
    intensity: str  # 'low', 'medium', 'high', 'extreme'
    cascade_risk: float  # 0.0 to 1.0
    
    @property
    def is_significant(self) -> bool:
        """Check if cluster is large enough to impact price"""
        return self.total_notional > 1_000_000  # $1M minimum


@dataclass
class EstimatedPosition:
    """Represents an estimated leveraged position in the market"""
    symbol: str
    side: str
    entry_price: float
    leverage: float
    size: float  # In base asset units
    estimated_liquidation: float
    confidence: float  # How confident we are in this estimate (0-1)


class LiquidationHeatmap:
    """
    Builds and maintains a heatmap of estimated liquidation levels.
    
    Uses statistical modeling to estimate where retail traders' 
    liquidation prices are clustered.
    """
    
    # Typical maintenance margin rates by asset
    MM_RATES = {
        'BTC': 0.004,
        'ETH': 0.005,
        'SOL': 0.0065,
    }
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.clusters: Dict[float, LiquidationCluster] = {}
        self.estimated_positions: List[EstimatedPosition] = []
        self.last_update_us: int = 0
        self._price_buckets: Dict[float, List[EstimatedPosition]] = defaultdict(list)
    
    def _calculate_long_liquidation(self, entry_price: float, leverage: float) -> float:
        """Calculate liquidation price for a long position"""
        mm_rate = self._get_mm_rate()
        initial_margin_rate = 1.0 / leverage
        
        liq_price = entry_price * (1 - initial_margin_rate + mm_rate)
        if mm_rate < 1.0:
            liq_price /= (1 - mm_rate)
        
        return max(0.0, liq_price)
    
    def _calculate_short_liquidation(self, entry_price: float, leverage: float) -> float:
        """Calculate liquidation price for a short position"""
        mm_rate = self._get_mm_rate()
        initial_margin_rate = 1.0 / leverage
        
        liq_price = entry_price * (1 + initial_margin_rate - mm_rate)
        if mm_rate < 1.0:
            liq_price /= (1 + mm_rate)
        
        return max(0.0, liq_price)
    
    def _get_mm_rate(self) -> float:
        """Get maintenance margin rate for this symbol"""
        base_asset = self.symbol.replace('USDT', '').replace('USD', '')
        return self.MM_RATES.get(base_asset, 0.01)
    
    def add_estimated_position(
        self,
        entry_price: float,
        leverage: float,
        size: float,
        side: str,
        confidence: float = 0.7,
    ) -> None:
        """
        Add an estimated position to the heatmap.
        
        This simulates knowledge of market positioning from various signals
        (OI data, funding rates, trade flow analysis).
        """
        if side == 'long':
            liq_price = self._calculate_long_liquidation(entry_price, leverage)
        else:
            liq_price = self._calculate_short_liquidation(entry_price, leverage)
        
        position = EstimatedPosition(
            symbol=self.symbol,
            side=side,
            entry_price=entry_price,
            leverage=leverage,
            size=size,
            estimated_liquidation=liq_price,
            confidence=confidence,
        )
        
        self.estimated_positions.append(position)
        
        # Bucket by liquidation price for clustering
        bucket = round(liq_price / 100) * 100  # $100 buckets
        self._price_buckets[bucket].append(position)
        
        self.last_update_us = int(time.time() * 1_000_000)
    
    def add_batch_positions(self, positions: List[Dict]) -> None:
        """Add multiple estimated positions"""
        for pos in positions:
            self.add_estimated_position(
                entry_price=pos['entry_price'],
                leverage=pos['leverage'],
                size=pos['size'],
                side=pos['side'],
                confidence=pos.get('confidence', 0.7),
            )
    
    def _calculate_cluster_intensity(self, notional: float) -> str:
        """Determine cluster intensity based on notional value"""
        if notional < 500_000:
            return 'low'
        elif notional < 2_000_000:
            return 'medium'
        elif notional < 10_000_000:
            return 'high'
        else:
            return 'extreme'
    
    def _calculate_cascade_risk(self, cluster: LiquidationCluster, current_price: float) -> float:
        """
        Calculate cascade risk score (0-1).
        
        Higher risk when:
        - Large notional relative to typical volume
        - Price is close to liquidation level
        - High average leverage
        """
        risk = 0.0
        
        # Distance factor (closer = higher risk)
        distance_pct = abs(cluster.price_level - current_price) / current_price
        if distance_pct < 0.01:
            risk += 0.4
        elif distance_pct < 0.03:
            risk += 0.25
        elif distance_pct < 0.05:
            risk += 0.1
        
        # Notional factor
        if cluster.total_notional > 10_000_000:
            risk += 0.3
        elif cluster.total_notional > 5_000_000:
            risk += 0.2
        elif cluster.total_notional > 1_000_000:
            risk += 0.1
        
        # Leverage factor
        if cluster.avg_leverage > 20:
            risk += 0.3
        elif cluster.avg_leverage > 10:
            risk += 0.15
        
        return min(1.0, risk)
    
    def rebuild_clusters(self) -> None:
        """Rebuild all clusters from estimated positions"""
        self.clusters.clear()
        
        for bucket, positions in self._price_buckets.items():
            if not positions:
                continue
            
            total_notional = sum(p.size * p.entry_price for p in positions)
            avg_leverage = sum(p.leverage for p in positions) / len(positions)
            
            # Determine dominant side
            long_notional = sum(p.size * p.entry_price for p in positions if p.side == 'long')
            short_notional = total_notional - long_notional
            
            dominant_side = 'long' if long_notional >= short_notional else 'short'
            
            cluster = LiquidationCluster(
                price_level=bucket,
                total_notional=total_notional,
                position_count=len(positions),
                avg_leverage=avg_leverage,
                side=dominant_side,
                intensity=self._calculate_cluster_intensity(total_notional),
                cascade_risk=0.0,  # Will be calculated with current price
            )
            
            self.clusters[bucket] = cluster
    
    def get_clusters_near_price(
        self,
        current_price: float,
        tolerance_pct: float = 0.05,
    ) -> List[LiquidationCluster]:
        """
        Get liquidation clusters within a percentage tolerance of current price.
        
        Args:
            current_price: Current market price
            tolerance_pct: Percentage tolerance (e.g., 0.05 = 5%)
            
        Returns:
            List of relevant clusters sorted by cascade risk
        """
        self.rebuild_clusters()
        
        lower_bound = current_price * (1 - tolerance_pct)
        upper_bound = current_price * (1 + tolerance_pct)
        
        relevant = []
        for cluster in self.clusters.values():
            if lower_bound <= cluster.price_level <= upper_bound:
                # Update cascade risk with current price
                cluster.cascade_risk = self._calculate_cascade_risk(cluster, current_price)
                relevant.append(cluster)
        
        # Sort by cascade risk descending
        relevant.sort(key=lambda c: c.cascade_risk, reverse=True)
        return relevant
    
    def get_heatmap_summary(self, current_price: float) -> Dict:
        """Get comprehensive heatmap summary"""
        self.rebuild_clusters()
        
        long_clusters = [c for c in self.clusters.values() if c.side == 'long']
        short_clusters = [c for c in self.clusters.values() if c.side == 'short']
        
        # Find largest clusters
        largest_long = max(long_clusters, key=lambda c: c.total_notional) if long_clusters else None
        largest_short = max(short_clusters, key=lambda c: c.total_notional) if short_clusters else None
        
        # Get nearby clusters
        nearby = self.get_clusters_near_price(current_price, 0.03)
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'total_estimated_positions': len(self.estimated_positions),
            'total_clusters': len(self.clusters),
            'long_clusters': len(long_clusters),
            'short_clusters': len(short_clusters),
            'largest_long_level': largest_long.price_level if largest_long else None,
            'largest_long_notional': largest_long.total_notional if largest_long else 0,
            'largest_short_level': largest_short.price_level if largest_short else None,
            'largest_short_notional': largest_short.total_notional if largest_short else 0,
            'nearby_clusters': [
                {
                    'price': c.price_level,
                    'notional': c.total_notional,
                    'side': c.side,
                    'cascade_risk': c.cascade_risk,
                }
                for c in nearby[:5]
            ],
            'timestamp_us': self.last_update_us,
        }
    
    def clear(self) -> None:
        """Clear all data"""
        self.estimated_positions.clear()
        self.clusters.clear()
        self._price_buckets.clear()
        self.last_update_us = 0


class MultiAssetLiquidationHeatmap:
    """Manage liquidation heatmaps for multiple assets"""
    
    def __init__(self):
        self.heatmaps: Dict[str, LiquidationHeatmap] = {}
    
    def get_heatmap(self, symbol: str) -> LiquidationHeatmap:
        """Get or create heatmap for a symbol"""
        if symbol not in self.heatmaps:
            self.heatmaps[symbol] = LiquidationHeatmap(symbol)
        return self.heatmaps[symbol]
    
    def update_from_oi_data(
        self,
        symbol: str,
        open_interest: float,
        long_ratio: float,
        avg_leverage: float,
        current_price: float,
    ) -> None:
        """
        Generate estimated positions from OI data.
        
        This creates synthetic positions that match observed market metrics.
        """
        heatmap = self.get_heatmap(symbol)
        heatmap.clear()
        
        # Estimate number of positions and distribute them
        # Assume average position size based on OI
        estimated_position_count = max(100, int(open_interest / 10))
        avg_position_size = open_interest / estimated_position_count
        
        # Distribute positions around current price
        for i in range(estimated_position_count):
            # Random entry price distribution (more near current price)
            price_offset = (i / estimated_position_count - 0.5) * current_price * 0.2
            entry_price = current_price + price_offset
            
            # Vary leverage
            leverage = 5 + (i % 20)  # 5x to 25x
            
            # Side based on long ratio
            side = 'long' if (i / estimated_position_count) < long_ratio else 'short'
            
            heatmap.add_estimated_position(
                entry_price=entry_price,
                leverage=leverage,
                size=avg_position_size,
                side=side,
                confidence=0.5,  # Lower confidence for synthetic data
            )
    
    def get_all_heatmaps_summary(self, prices: Dict[str, float]) -> Dict[str, Dict]:
        """Get summary for all heatmaps"""
        return {
            symbol: h.get_heatmap_summary(prices.get(symbol, 0))
            for symbol, h in self.heatmaps.items()
        }


if __name__ == "__main__":
    # Example usage
    print("Liquidation Heatmap Estimator initialized")
    
    heatmap = LiquidationHeatmap("BTCUSDT")
    
    # Add sample estimated positions
    sample_positions = [
        {'entry_price': 52000, 'leverage': 20, 'size': 0.5, 'side': 'long'},
        {'entry_price': 51000, 'leverage': 15, 'size': 1.0, 'side': 'long'},
        {'entry_price': 50000, 'leverage': 25, 'size': 2.0, 'side': 'long'},
        {'entry_price': 49000, 'leverage': 10, 'size': 1.5, 'side': 'long'},
        {'entry_price': 48000, 'leverage': 20, 'size': 3.0, 'side': 'long'},
        {'entry_price': 53000, 'leverage': 15, 'size': 0.8, 'side': 'short'},
        {'entry_price': 54000, 'leverage': 20, 'size': 1.2, 'side': 'short'},
        {'entry_price': 55000, 'leverage': 25, 'size': 2.5, 'side': 'short'},
    ]
    
    heatmap.add_batch_positions(sample_positions)
    
    # Get summary at current price
    summary = heatmap.get_heatmap_summary(current_price=51500)
    print(f"\nHeatmap Summary for {summary['symbol']}:")
    print(f"  Total Positions: {summary['total_estimated_positions']}")
    print(f"  Long Clusters: {summary['long_clusters']}")
    print(f"  Short Clusters: {summary['short_clusters']}")
    print(f"  Nearby Clusters: {summary['nearby_clusters']}")
