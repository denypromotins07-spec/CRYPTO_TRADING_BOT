"""
Liquidity Engine for ZAID Personal Crypto Trading Bot
Uncovers hidden liquidity, iceberg orders, and institutional footprints
Uses order book analysis to detect large player activity
Memory-optimized with sliding window algorithms

Part of the 152 domains of quantitative finance implementation.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import time
import logging
from sortedcontainers import SortedDict

logger = logging.getLogger(__name__)


class LiquidityType(Enum):
    """Types of liquidity detected in the order book."""
    ICEBERG = "ICEBERG"
    HIDDEN = "HIDDEN"
    WALL = "WALL"
    SPREAD_IMBALANCE = "SPREAD_IMBALANCE"
    DEPTH_CLUSTER = "DEPTH_CLUSTER"


@dataclass
class OrderBookLevel:
    """Represents a single price level in the order book."""
    price: float
    volume: float
    order_count: int = 1
    visible_volume: float = 0.0
    hidden_volume: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    @property
    def total_volume(self) -> float:
        return self.visible_volume + self.hidden_volume
    
    @property
    def iceberg_ratio(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return self.hidden_volume / self.total_volume


@dataclass
class LiquidityEvent:
    """Detected liquidity event with metadata."""
    event_type: LiquidityType
    symbol: str
    price: float
    volume: float
    confidence: float
    timestamp: float
    side: str  # 'bid' or 'ask'
    duration_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class IcebergOrder:
    """Tracks a detected iceberg order."""
    symbol: str
    price: float
    side: str
    visible_size: float
    estimated_total: float
    refresh_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_refresh: float = field(default_factory=time.time)
    
    @property
    def iceberg_ratio(self) -> float:
        if self.estimated_total == 0:
            return 0.0
        return self.visible_size / self.estimated_total


class LiquidityDetector:
    """
    Detects various forms of hidden liquidity in the order book.
    Uses statistical analysis and pattern recognition.
    """
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
        self.volume_history: Dict[str, Dict[float, deque]] = {}
        self.refresh_patterns: Dict[str, Dict[float, List[float]]] = {}
        self.detected_icebergs: Dict[str, Dict[float, IcebergOrder]] = {}
        
    def update_level(self, symbol: str, price: float, volume: float, 
                     side: str, order_count: int = 1) -> Optional[LiquidityEvent]:
        """
        Update a price level and check for liquidity anomalies.
        Returns LiquidityEvent if anomaly detected.
        """
        if symbol not in self.volume_history:
            self.volume_history[symbol] = {}
            self.refresh_patterns[symbol] = {}
            self.detected_icebergs[symbol] = {}
        
        if price not in self.volume_history[symbol]:
            self.volume_history[symbol][price] = deque(maxlen=50)
            self.refresh_patterns[symbol][price] = []
        
        history = self.volume_history[symbol][price]
        current_time = time.time()
        
        # Store volume snapshot
        history.append({
            'volume': volume,
            'order_count': order_count,
            'timestamp': current_time,
            'side': side
        })
        
        # Check for iceberg pattern
        if len(history) >= 5:
            event = self._detect_iceberg(symbol, price, side, history)
            if event:
                return event
        
        # Check for hidden liquidity
        if len(history) >= 3:
            event = self._detect_hidden_liquidity(symbol, price, side, history)
            if event:
                return event
        
        return None
    
    def _detect_iceberg(self, symbol: str, price: float, side: str,
                       history: deque) -> Optional[LiquidityEvent]:
        """
        Detect iceberg order patterns.
        Iceberg shows consistent refresh after partial fills.
        """
        volumes = [h['volume'] for h in history]
        timestamps = [h['timestamp'] for h in history]
        
        # Check for volume refresh pattern
        refresh_detected = False
        refresh_times = []
        
        for i in range(1, len(volumes)):
            # Volume decreased significantly then returned to similar level
            if volumes[i] < volumes[i-1] * 0.7:  # 30%+ decrease
                if i + 1 < len(volumes) and volumes[i+1] > volumes[i] * 1.3:
                    refresh_detected = True
                    refresh_times.append(timestamps[i])
        
        if not refresh_detected or len(refresh_times) < 2:
            return None
        
        # Calculate refresh frequency
        time_diffs = [refresh_times[i+1] - refresh_times[i] 
                     for i in range(len(refresh_times)-1)]
        avg_refresh_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Estimate total iceberg size
        avg_volume = sum(volumes[-10:]) / min(10, len(volumes))
        estimated_total = avg_volume * (1 + len(refresh_times))
        
        # Confidence based on pattern consistency
        volume_variance = sum((v - avg_volume)**2 for v in volumes[-5:]) / min(5, len(volumes))
        confidence = 1.0 / (1.0 + volume_variance / (avg_volume ** 2))
        
        if confidence < self.sensitivity:
            return None
        
        # Store iceberg detection
        if price not in self.detected_icebergs[symbol]:
            self.detected_icebergs[symbol][price] = IcebergOrder(
                symbol=symbol,
                price=price,
                side=side,
                visible_size=avg_volume,
                estimated_total=estimated_total
            )
        else:
            iceberg = self.detected_icebergs[symbol][price]
            iceberg.refresh_count += 1
            iceberg.estimated_total = max(iceberg.estimated_total, estimated_total)
            iceberg.last_refresh = time.time()
        
        return LiquidityEvent(
            event_type=LiquidityType.ICEBERG,
            symbol=symbol,
            price=price,
            volume=estimated_total,
            confidence=confidence,
            timestamp=time.time(),
            side=side,
            metadata={
                'refresh_count': len(refresh_times),
                'avg_refresh_time': avg_refresh_time,
                'visible_size': avg_volume
            }
        )
    
    def _detect_hidden_liquidity(self, symbol: str, price: float, side: str,
                                 history: deque) -> Optional[LiquidityEvent]:
        """
        Detect hidden liquidity through volume anomalies.
        Hidden orders show unusual stability despite market pressure.
        """
        volumes = [h['volume'] for h in history]
        
        if len(volumes) < 5:
            return None
        
        # Check for abnormal volume stability
        avg_volume = sum(volumes) / len(volumes)
        variance = sum((v - avg_volume)**2 for v in volumes) / len(volumes)
        std_dev = variance ** 0.5
        cv = std_dev / avg_volume if avg_volume > 0 else float('inf')
        
        # Low coefficient of variation suggests hidden support/resistance
        if cv > 0.3:  # Too volatile
            return None
        
        # Check if level persists despite trades
        recent_volume_change = abs(volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
        
        if recent_volume_change > 0.5:  # Volume changed significantly
            return None
        
        # This level is unusually stable - likely hidden liquidity
        confidence = 1.0 - cv
        if confidence < self.sensitivity:
            return None
        
        return LiquidityEvent(
            event_type=LiquidityType.HIDDEN,
            symbol=symbol,
            price=price,
            volume=avg_volume,
            confidence=confidence,
            timestamp=time.time(),
            side=side,
            metadata={
                'coefficient_of_variation': cv,
                'stability_score': 1.0 - cv
            }
        )


class LiquidityWallDetector:
    """Detects large liquidity walls that may indicate institutional interest."""
    
    def __init__(self, wall_threshold_multiplier: float = 5.0):
        self.wall_threshold = wall_threshold_multiplier
        self.recent_volumes: Dict[str, deque] = {}
        
    def analyze_depth(self, symbol: str, bids: SortedDict, asks: SortedDict) -> List[LiquidityEvent]:
        """Analyze order book depth for liquidity walls."""
        events = []
        
        if symbol not in self.recent_volumes:
            self.recent_volumes[symbol] = deque(maxlen=100)
        
        # Calculate average volume at each side
        all_volumes = []
        for price, level in bids.items():
            if hasattr(level, 'volume'):
                all_volumes.append(level.volume)
            elif isinstance(level, dict):
                all_volumes.append(level.get('volume', 0))
            else:
                all_volumes.append(float(level) if isinstance(level, (int, float)) else 0)
        
        for price, level in asks.items():
            if hasattr(level, 'volume'):
                all_volumes.append(level.volume)
            elif isinstance(level, dict):
                all_volumes.append(level.get('volume', 0))
            else:
                all_volumes.append(float(level) if isinstance(level, (int, float)) else 0)
        
        if not all_volumes:
            return events
        
        avg_volume = sum(all_volumes) / len(all_volumes)
        threshold = avg_volume * self.wall_threshold
        
        # Check for bid walls
        for price, level in reversed(list(bids.items())):
            volume = level.volume if hasattr(level, 'volume') else (
                level.get('volume', 0) if isinstance(level, dict) else 
                (float(level) if isinstance(level, (int, float)) else 0))
            
            if volume >= threshold:
                events.append(LiquidityEvent(
                    event_type=LiquidityType.WALL,
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    confidence=min(1.0, volume / threshold),
                    timestamp=time.time(),
                    side='bid',
                    metadata={'wall_type': 'support'}
                ))
                break  # Only report largest wall
        
        # Check for ask walls
        for price, level in asks.items():
            volume = level.volume if hasattr(level, 'volume') else (
                level.get('volume', 0) if isinstance(level, dict) else 
                (float(level) if isinstance(level, (int, float)) else 0))
            
            if volume >= threshold:
                events.append(LiquidityEvent(
                    event_type=LiquidityType.WALL,
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    confidence=min(1.0, volume / threshold),
                    timestamp=time.time(),
                    side='ask',
                    metadata={'wall_type': 'resistance'}
                ))
                break
        
        return events


class SpreadImbalanceDetector:
    """Detects significant bid-ask spread imbalances."""
    
    def __init__(self, imbalance_threshold: float = 2.0):
        self.imbalance_threshold = imbalance_threshold
        
    def check_imbalance(self, symbol: str, best_bid: float, best_ask: float,
                       bid_volume: float, ask_volume: float) -> Optional[LiquidityEvent]:
        """Check for spread-based liquidity imbalance."""
        if best_bid >= best_ask or bid_volume == 0 or ask_volume == 0:
            return None
        
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate volume imbalance ratio
        imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        
        if imbalance_ratio < 1 / self.imbalance_threshold or \
           imbalance_ratio > self.imbalance_threshold:
            
            side = 'bid' if imbalance_ratio > 1 else 'ask'
            confidence = min(1.0, abs(imbalance_ratio - 1) / self.imbalance_threshold)
            
            return LiquidityEvent(
                event_type=LiquidityType.SPREAD_IMBALANCE,
                symbol=symbol,
                price=mid_price,
                volume=bid_volume + ask_volume,
                confidence=confidence,
                timestamp=time.time(),
                side=side,
                metadata={
                    'imbalance_ratio': imbalance_ratio,
                    'spread': spread,
                    'spread_pct': spread / mid_price * 100
                }
            )
        
        return None


class DepthClusterAnalyzer:
    """Identifies clusters of liquidity at specific price levels."""
    
    def __init__(self, cluster_tolerance: float = 0.001):
        self.cluster_tolerance = cluster_tolerance  # 0.1% price tolerance
        
    def find_clusters(self, symbol: str, levels: Dict[float, float], 
                     side: str) -> List[LiquidityEvent]:
        """Find price clusters where liquidity concentrates."""
        if not levels:
            return []
        
        events = []
        sorted_prices = sorted(levels.keys())
        
        i = 0
        while i < len(sorted_prices):
            cluster_start = sorted_prices[i]
            cluster_end = cluster_start
            cluster_volume = levels[sorted_prices[i]]
            
            # Expand cluster within tolerance
            j = i + 1
            while j < len(sorted_prices):
                if sorted_prices[j] <= cluster_end * (1 + self.cluster_tolerance):
                    cluster_end = sorted_prices[j]
                    cluster_volume += levels[sorted_prices[j]]
                    j += 1
                else:
                    break
            
            # Check if cluster is significant
            if j - i >= 3:  # At least 3 levels in cluster
                avg_price = (cluster_start + cluster_end) / 2
                events.append(LiquidityEvent(
                    event_type=LiquidityType.DEPTH_CLUSTER,
                    symbol=symbol,
                    price=avg_price,
                    volume=cluster_volume,
                    confidence=min(1.0, (j - i) / 10),
                    timestamp=time.time(),
                    side=side,
                    metadata={
                        'level_count': j - i,
                        'price_range': cluster_end - cluster_start,
                        'start_price': cluster_start,
                        'end_price': cluster_end
                    }
                ))
            
            i = j if j > i + 1 else i + 1
        
        return events


class LiquidityEngine:
    """
    Main liquidity analysis engine coordinating all detectors.
    Singleton pattern for global access.
    """
    
    _instance: Optional['LiquidityEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.iceberg_detector = LiquidityDetector(sensitivity=0.7)
        self.wall_detector = LiquidityWallDetector(wall_threshold_multiplier=5.0)
        self.spread_detector = SpreadImbalanceDetector(imbalance_threshold=2.0)
        self.cluster_analyzer = DepthClusterAnalyzer(cluster_tolerance=0.001)
        
        self.order_books: Dict[str, Dict[str, SortedDict]] = {}
        self.recent_events: Dict[str, deque] = {}
        self.max_events = 1000
        
        self._initialized = True
        logger.info("LiquidityEngine initialized")
    
    def update_order_book(self, symbol: str, bids: Dict[float, float], 
                         asks: Dict[float, float]) -> List[LiquidityEvent]:
        """
        Update order book and run all liquidity detectors.
        Returns list of detected liquidity events.
        """
        events = []
        
        if symbol not in self.order_books:
            self.order_books[symbol] = {
                'bids': SortedDict(lambda x: -x),  # Descending for bids
                'asks': SortedDict()
            }
            self.recent_events[symbol] = deque(maxlen=self.max_events)
        
        # Update stored order book
        for price, volume in bids.items():
            self.order_books[symbol]['bids'][price] = OrderBookLevel(
                price=price,
                volume=volume,
                visible_volume=volume * 0.7,  # Assume 70% visible initially
                hidden_volume=volume * 0.3
            )
        
        for price, volume in asks.items():
            self.order_books[symbol]['asks'][price] = OrderBookLevel(
                price=price,
                volume=volume,
                visible_volume=volume * 0.7,
                hidden_volume=volume * 0.3
            )
        
        # Run iceberg/hidden detection on top levels
        for price, level in list(self.order_books[symbol]['bids'].items())[:20]:
            event = self.iceberg_detector.update_level(
                symbol, price, level.volume, 'bid', level.order_count
            )
            if event:
                events.append(event)
        
        for price, level in list(self.order_books[symbol]['asks'].items())[:20]:
            event = self.iceberg_detector.update_level(
                symbol, price, level.volume, 'ask', level.order_count
            )
            if event:
                events.append(event)
        
        # Run wall detection
        wall_events = self.wall_detector.analyze_depth(
            symbol,
            self.order_books[symbol]['bids'],
            self.order_books[symbol]['asks']
        )
        events.extend(wall_events)
        
        # Run spread imbalance detection
        if self.order_books[symbol]['bids'] and self.order_books[symbol]['asks']:
            best_bid = self.order_books[symbol]['bids'].peekitem(0)[0]
            best_ask = self.order_books[symbol]['asks'].peekitem(0)[0]
            
            bid_level = self.order_books[symbol]['bids'][best_bid]
            ask_level = self.order_books[symbol]['asks'][best_ask]
            
            spread_event = self.spread_detector.check_imbalance(
                symbol, best_bid, best_ask,
                bid_level.volume, ask_level.volume
            )
            if spread_event:
                events.append(spread_event)
        
        # Run cluster analysis
        bid_levels = {p: l.volume for p, l in self.order_books[symbol]['bids'].items()}
        ask_levels = {p: l.volume for p, l in self.order_books[symbol]['asks'].items()}
        
        bid_clusters = self.cluster_analyzer.find_clusters(symbol, bid_levels, 'bid')
        ask_clusters = self.cluster_analyzer.find_clusters(symbol, ask_levels, 'ask')
        events.extend(bid_clusters)
        events.extend(ask_clusters)
        
        # Store events
        for event in events:
            self.recent_events[symbol].append(event)
        
        if events:
            logger.debug(f"Detected {len(events)} liquidity events for {symbol}")
        
        return events
    
    def get_recent_events(self, symbol: str, limit: int = 10) -> List[LiquidityEvent]:
        """Get recent liquidity events for a symbol."""
        if symbol not in self.recent_events:
            return []
        return list(self.recent_events[symbol])[-limit:]
    
    def get_iceberg_orders(self, symbol: str) -> List[IcebergOrder]:
        """Get currently tracked iceberg orders for a symbol."""
        if symbol not in self.iceberg_detector.detected_icebergs:
            return []
        return list(self.iceberg_detector.detected_icebergs[symbol].values())
    
    def get_liquidity_heatmap(self, symbol: str, price_range: Tuple[float, float],
                             num_bins: int = 50) -> Dict[float, float]:
        """
        Generate liquidity heatmap for a price range.
        Useful for visualizing support/resistance zones.
        """
        if symbol not in self.order_books:
            return {}
        
        min_price, max_price = price_range
        bin_size = (max_price - min_price) / num_bins
        heatmap: Dict[float, float] = {}
        
        for price, level in self.order_books[symbol]['bids'].items():
            if min_price <= price <= max_price:
                bin_price = round(price / bin_size) * bin_size
                heatmap[bin_price] = heatmap.get(bin_price, 0) + level.volume
        
        for price, level in self.order_books[symbol]['asks'].items():
            if min_price <= price <= max_price:
                bin_price = round(price / bin_size) * bin_size
                heatmap[bin_price] = heatmap.get(bin_price, 0) + level.volume
        
        return heatmap
    
    def clear_symbol(self, symbol: str):
        """Clear all liquidity data for a symbol."""
        if symbol in self.order_books:
            del self.order_books[symbol]
        if symbol in self.recent_events:
            del self.recent_events[symbol]
        if symbol in self.iceberg_detector.detected_icebergs:
            del self.iceberg_detector.detected_icebergs[symbol]
        if symbol in self.iceberg_detector.volume_history:
            del self.iceberg_detector.volume_history[symbol]
        
        logger.info(f"Cleared liquidity data for {symbol}")


# Async wrapper for integration
async def analyze_liquidity_async(engine: LiquidityEngine, symbol: str,
                                  bids: Dict[float, float], 
                                  asks: Dict[float, float]):
    """Async wrapper for liquidity analysis."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        engine.update_order_book,
        symbol, bids, asks
    )
    return result


if __name__ == "__main__":
    # Example usage
    engine = LiquidityEngine()
    
    # Simulate order book with potential iceberg
    bids = {45000 - i * 10: 10.0 if i != 5 else 50.0 for i in range(20)}
    asks = {45050 + i * 10: 8.0 if i != 3 else 40.0 for i in range(20)}
    
    # First update
    events = engine.update_order_book("BTCUSDT", bids, asks)
    print(f"Initial events: {len(events)}")
    
    # Simulate multiple updates to trigger iceberg detection
    for _ in range(10):
        # Reduce volume at key level (simulating fill)
        bids[44950] *= 0.6
        events = engine.update_order_book("BTCUSDT", bids, asks)
        if events:
            for event in events:
                print(f"Event: {event.event_type.value} at {event.price:.2f}, "
                      f"confidence: {event.confidence:.2f}")
