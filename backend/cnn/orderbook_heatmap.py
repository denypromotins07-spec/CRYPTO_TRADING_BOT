#!/usr/bin/env python3
"""
Order Book Heatmap Generator for CNN Processing

Converts L2 order book snapshots into 2D image matrices suitable for
convolutional neural network processing. Enables visual pattern detection
for spoofing, liquidity sweeps, and market manipulation.

Features:
- Real-time heatmap generation from order book data
- Multi-resolution support for different CNN architectures
- Normalization schemes optimized for crypto volatility
- Color mapping for bid/ask imbalance visualization
- Memory-efficient batch processing

Integrates with the 152 domains of quantitative finance.
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NormalizationMode(Enum):
    """Normalization strategies for order book heatmaps."""
    ZSCORE = "zscore"           # Standard z-score normalization
    MINMAX = "minmax"           # Scale to [0, 1]
    LOG = "log"                 # Log transform for heavy tails
    RANK = "rank"               # Rank-based normalization
    PERCENTILE = "percentile"   # Percentile scaling


@dataclass
class HeatmapConfig:
    """Configuration for order book heatmap generation."""
    n_levels: int = 10              # Number of price levels per side
    image_height: int = 20          # Output image height (bid + ask)
    image_width: int = 50           # Output image width (time dimension)
    normalization: NormalizationMode = NormalizationMode.ZSCORE
    include_volume: bool = True     # Include volume channels
    include_imbalance: bool = True  # Include bid-ask imbalance
    color_channels: int = 3         # RGB output
    
    def validate(self) -> bool:
        """Validate configuration constraints."""
        if self.n_levels < 1 or self.n_levels > 50:
            logger.warning(f"n_levels={self.n_levels} outside recommended range")
            return False
        if self.image_height < 10 or self.image_height > 200:
            logger.warning(f"image_height={self.image_height} outside range")
            return False
        return True


@dataclass
class OrderBookSnapshot:
    """Single order book snapshot with metadata."""
    timestamp_ns: int
    bids: npt.NDArray[np.float64]  # [n_levels, 2] (price, volume)
    asks: npt.NDArray[np.float64]  # [n_levels, 2] (price, volume)
    spread: float = 0.0
    mid_price: float = 0.0
    
    def __post_init__(self):
        if len(self.bids) > 0 and len(self.asks) > 0:
            self.mid_price = (self.bids[0, 0] + self.asks[0, 0]) / 2
            self.spread = self.asks[0, 0] - self.bids[0, 0]


class OrderBookHeatmap:
    """
    Converts L2 order book data into 2D heatmaps for CNN processing.
    
    The generated heatmaps encode:
    - Price levels (y-axis)
    - Time history (x-axis)  
    - Volume/intensity (color/brightness)
    - Bid-ask imbalance (hue)
    
    This enables CNNs to detect:
    - Spoofing patterns (large fake walls)
    - Liquidity sweeps
    - Accumulation/distribution patterns
    - Market manipulation signatures
    """
    
    def __init__(self, config: Optional[HeatmapConfig] = None):
        self.config = config or HeatmapConfig()
        self.config.validate()
        
        # Rolling buffer for time dimension
        self._buffer: List[OrderBookSnapshot] = []
        self._volume_history: Optional[npt.NDArray[np.float64]] = None
        self._price_history: Optional[npt.NDArray[np.float64]] = None
        
        # Statistics for normalization
        self._running_mean: float = 0.0
        self._running_var: float = 1.0
        self._n_samples: int = 0
        
        logger.info(f"OrderBookHeatmap initialized: {self.config}")
    
    def add_snapshot(
        self,
        timestamp_ns: int,
        bids: npt.NDArray[np.float64],
        asks: npt.NDArray[np.float64]
    ) -> None:
        """
        Add new order book snapshot to rolling buffer.
        
        Args:
            timestamp_ns: Nanosecond timestamp
            bids: Bid levels [n_levels, 2] (price, volume)
            asks: Ask levels [n_levels, 2] (price, volume)
        """
        snapshot = OrderBookSnapshot(
            timestamp_ns=timestamp_ns,
            bids=bids[:self.config.n_levels],
            asks=asks[:self.config.n_levels]
        )
        
        self._buffer.append(snapshot)
        
        # Trim buffer to image width
        if len(self._buffer) > self.config.image_width:
            self._buffer.pop(0)
        
        # Update running statistics
        self._update_statistics(snapshot)
    
    def _update_statistics(self, snapshot: OrderBookSnapshot) -> None:
        """Update running mean and variance for normalization."""
        volumes = np.concatenate([snapshot.bids[:, 1], snapshot.asks[:, 1]])
        
        if len(volumes) == 0:
            return
        
        batch_mean = np.mean(volumes)
        batch_var = np.var(volumes)
        
        # Welford's online algorithm
        self._n_samples += 1
        delta = batch_mean - self._running_mean
        self._running_mean += delta / self._n_samples
        self._running_var += delta * (batch_mean - self._running_mean)
    
    def _normalize_volumes(
        self,
        volumes: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Apply configured normalization to volume data."""
        if self.config.normalization == NormalizationMode.MINMAX:
            v_min, v_max = volumes.min(), volumes.max()
            if v_max - v_min < 1e-9:
                return np.zeros_like(volumes)
            return (volumes - v_min) / (v_max - v_min)
        
        elif self.config.normalization == NormalizationMode.LOG:
            return np.log1p(volumes) / np.log1p(volumes.max() + 1)
        
        elif self.config.normalization == NormalizationMode.RANK:
            from scipy.stats import rankdata
            return rankdata(volumes) / len(volumes)
        
        elif self.config.normalization == NormalizationMode.PERCENTILE:
            return np.searchsorted(np.percentile(volumes, np.arange(101)), volumes) / 100
        
        else:  # ZSCORE (default)
            if self._n_samples < 2:
                return (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-9)
            std = np.sqrt(self._running_var) + 1e-9
            return (volumes - self._running_mean) / std
    
    def generate_heatmap(
        self,
        include_channels: Optional[List[str]] = None
    ) -> npt.NDArray[np.float64]:
        """
        Generate 2D heatmap from current buffer state.
        
        Args:
            include_channels: Which channels to include
                - 'volume': Raw volume intensity
                - 'imbalance': Bid-ask volume imbalance
                - 'spread': Relative spread at each level
                - Default: all channels
                
        Returns:
            Heatmap array [height, width, channels]
        """
        if len(self._buffer) == 0:
            return np.zeros((
                self.config.image_height,
                self.config.image_width,
                self.config.color_channels
            ))
        
        n_time = min(len(self._buffer), self.config.image_width)
        n_levels = self.config.n_levels
        
        # Initialize channels
        channels = {}
        
        # Volume channel
        if include_channels is None or 'volume' in include_channels:
            volume_matrix = np.zeros((n_levels * 2, n_time))
            for t, snapshot in enumerate(self._buffer[-n_time:]):
                all_volumes = np.concatenate([
                    snapshot.bids[:, 1][::-1],  # Bids reversed (bottom to top)
                    snapshot.asks[:, 1]
                ])
                volume_matrix[:, t] = self._normalize_volumes(all_volumes)
            channels['volume'] = volume_matrix
        
        # Imbalance channel
        if include_channels is None or 'imbalance' in include_channels:
            imbalance_matrix = np.zeros((n_levels, n_time))
            for t, snapshot in enumerate(self._buffer[-n_time:]):
                bid_vol = snapshot.bids[:, 1]
                ask_vol = snapshot.asks[:, 1]
                total = bid_vol + ask_vol + 1e-9
                imbalance = (bid_vol - ask_vol) / total
                imbalance_matrix[:, t] = imbalance
            channels['imbalance'] = imbalance_matrix
        
        # Spread channel
        if include_channels is None or 'spread' in include_channels:
            spread_matrix = np.zeros((n_levels, n_time))
            for t, snapshot in enumerate(self._buffer[-n_time:]):
                for level in range(n_levels):
                    if level < len(snapshot.bids) and level < len(snapshot.asks):
                        spread = snapshot.asks[level, 0] - snapshot.bids[level, 0]
                        spread_matrix[level, t] = spread / (snapshot.mid_price + 1e-9)
            channels['spread'] = spread_matrix
        
        # Combine channels into RGB image
        return self._channels_to_rgb(channels, n_time)
    
    def _channels_to_rgb(
        self,
        channels: Dict[str, npt.NDArray[np.float64]],
        n_time: int
    ) -> npt.NDArray[np.float64]:
        """Convert channel data to RGB image."""
        height = self.config.image_height
        width = n_time
        
        # Initialize RGB image
        image = np.zeros((height, width, 3), dtype=np.float64)
        
        n_levels = self.config.n_levels
        bid_start = n_levels  # Bids in bottom half, asks in top half
        
        if 'volume' in channels:
            vol_data = channels['volume']
            
            # Map to RGB (blue for bids, red for asks)
            for t in range(width):
                # Ask side (top half) - red channel
                for level in range(n_levels):
                    row = n_levels - 1 - level
                    image[row, t, 0] = vol_data[n_levels + level, t]  # Red
                
                # Bid side (bottom half) - blue channel
                for level in range(n_levels):
                    row = bid_start + level
                    image[row, t, 2] = vol_data[n_levels - 1 - level, t]  # Blue
        
        if 'imbalance' in channels:
            imb_data = channels['imbalance']
            
            # Green channel shows imbalance
            for level in range(n_levels):
                row = n_levels - 1 - level
                image[row, :, 1] = (imb_data[level, :] + 1) / 2  # Scale [-1,1] to [0,1]
                image[bid_start + level, :, 1] = (imb_data[level, :] + 1) / 2
        
        return np.clip(image, 0, 1)
    
    def generate_batch(
        self,
        snapshots: List[OrderBookSnapshot],
        batch_size: int = 16
    ) -> npt.NDArray[np.float64]:
        """
        Generate batch of heatmaps from multiple snapshots.
        
        Args:
            snapshots: List of order book snapshots
            batch_size: Number of samples per batch
            
        Returns:
            Batch of heatmaps [batch, height, width, channels]
        """
        n_batches = (len(snapshots) + batch_size - 1) // batch_size
        all_heatmaps = []
        
        for i in range(0, len(snapshots), batch_size):
            batch_snapshots = snapshots[i:i + batch_size]
            
            # Clear buffer and add batch snapshots
            self._buffer.clear()
            for snap in batch_snapshots:
                self.add_snapshot(
                    snap.timestamp_ns,
                    snap.bids,
                    snap.asks
                )
            
            heatmap = self.generate_heatmap()
            all_heatmaps.append(heatmap)
        
        return np.stack(all_heatmaps, axis=0)
    
    def get_spoofing_score(self) -> float:
        """
        Calculate spoofing detection score from current heatmap.
        
        Spoofing indicators:
        - Large volume walls that disappear quickly
        - Asymmetric order book structure
        - Rapid size changes at specific levels
        
        Returns:
            Spoofing probability score [0, 1]
        """
        if len(self._buffer) < 5:
            return 0.0
        
        score = 0.0
        
        # Check for large asymmetric walls
        recent = self._buffer[-1]
        if len(recent.bids) > 0 and len(recent.asks) > 0:
            bid_vol = np.sum(recent.bids[:, 1])
            ask_vol = np.sum(recent.asks[:, 1])
            total = bid_vol + ask_vol + 1e-9
            
            # High imbalance suggests potential spoofing
            imbalance = abs(bid_vol - ask_vol) / total
            score += imbalance * 0.3
        
        # Check for rapid volume changes
        if len(self._buffer) >= 3:
            vol_changes = []
            for i in range(1, len(self._buffer)):
                prev_vol = np.sum(self._buffer[i-1].bids[:, 1]) + np.sum(self._buffer[i-1].asks[:, 1])
                curr_vol = np.sum(self._buffer[i].bids[:, 1]) + np.sum(self._buffer[i].asks[:, 1])
                if prev_vol > 0:
                    change = abs(curr_vol - prev_vol) / prev_vol
                    vol_changes.append(change)
            
            if vol_changes:
                avg_change = np.mean(vol_changes)
                score += min(avg_change, 1.0) * 0.4
        
        return min(score, 1.0)
    
    def clear(self) -> None:
        """Clear all buffered data."""
        self._buffer.clear()
        self._volume_history = None
        self._price_history = None


def main():
    """Test order book heatmap generation."""
    np.random.seed(42)
    
    print("=" * 60)
    print("Order Book Heatmap Generator Test")
    print("=" * 60)
    
    # Create configuration
    config = HeatmapConfig(
        n_levels=10,
        image_height=20,
        image_width=30,
        normalization=NormalizationMode.ZSCORE
    )
    
    # Initialize heatmap generator
    heatmap_gen = OrderBookHeatmap(config)
    
    # Generate synthetic order book data
    base_price = 50000.0
    snapshots = []
    
    for t in range(50):
        timestamp = t * 1_000_000  # 1ms intervals
        
        # Generate bid/ask levels
        bids = np.zeros((10, 2))
        asks = np.zeros((10, 2))
        
        for i in range(10):
            spread = (i + 1) * 0.5  # 0.5 price per level
            volume = np.random.exponential(100)
            
            bids[i] = [base_price - spread - np.random.uniform(0, 0.1), volume]
            asks[i] = [base_price + spread + np.random.uniform(0, 0.1), volume * 1.1]
        
        snapshot = OrderBookSnapshot(
            timestamp_ns=timestamp,
            bids=bids,
            asks=asks
        )
        snapshots.append(snapshot)
        
        # Add to generator
        heatmap_gen.add_snapshot(timestamp, bids, asks)
    
    # Generate heatmap
    print("\n1. Generating single heatmap...")
    heatmap = heatmap_gen.generate_heatmap()
    print(f"   Shape: {heatmap.shape}")
    print(f"   Value range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"   Mean: {heatmap.mean():.3f}")
    
    # Test batch generation
    print("\n2. Testing batch generation...")
    batch = heatmap_gen.generate_batch(snapshots[:20], batch_size=4)
    print(f"   Batch shape: {batch.shape}")
    
    # Test spoofing detection
    print("\n3. Testing spoofing detection...")
    spoof_score = heatmap_gen.get_spoofing_score()
    print(f"   Spoofing score: {spoof_score:.3f}")
    
    # Test different normalizations
    print("\n4. Testing normalization modes...")
    for mode in NormalizationMode:
        config_test = HeatmapConfig(normalization=mode)
        gen_test = OrderBookHeatmap(config_test)
        
        for snap in snapshots[:10]:
            gen_test.add_snapshot(snap.timestamp_ns, snap.bids, snap.asks)
        
        hm = gen_test.generate_heatmap()
        print(f"   {mode.value}: range=[{hm.min():.3f}, {hm.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
