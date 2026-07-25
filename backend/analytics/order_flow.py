"""
Order Flow Analytics Engine for ZAID Personal Crypto Trading Bot
Computes CVD (Cumulative Volume Delta), Delta, and Footprint Charts
Optimized for O(1) updates from L2 order book data
Memory efficient: Uses ring buffers and incremental calculations

Part of the 152 domains of quantitative finance implementation.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class TradeSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


@dataclass
class TradeTick:
    """Represents a single trade tick with aggressive side detection."""
    timestamp: float
    price: float
    quantity: float
    side: TradeSide
    trade_id: str
    
    def __post_init__(self):
        if not isinstance(self.side, TradeSide):
            self.side = TradeSide.UNKNOWN


@dataclass
class FootprintCell:
    """Single cell in a footprint chart."""
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0
    high: float = 0.0
    low: float = float('inf')
    
    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume
    
    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume
    
    @property
    def imbalance_ratio(self) -> float:
        if self.sell_volume == 0:
            return float('inf') if self.buy_volume > 0 else 0.0
        return self.buy_volume / self.sell_volume


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state for delta calculations."""
    timestamp: float
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float


class CumulativeVolumeDelta:
    """
    Computes CVD in real-time with O(1) updates.
    Tracks buying vs selling pressure across all assets.
    """
    
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.cumulative_delta: Dict[str, float] = {}
        self.trade_history: Dict[str, Deque[TradeTick]] = {}
        self.delta_changes: Dict[str, Deque[float]] = {}
        
    def add_trade(self, symbol: str, tick: TradeTick) -> float:
        """
        Add a trade and return updated CVD value.
        O(1) operation using incremental update.
        """
        if symbol not in self.cumulative_delta:
            self.cumulative_delta[symbol] = 0.0
            self.trade_history[symbol] = deque(maxlen=self.window_size)
            self.delta_changes[symbol] = deque(maxlen=self.window_size)
        
        # Determine delta contribution
        delta = tick.quantity if tick.side == TradeSide.BUY else -tick.quantity
        
        # Update cumulative delta
        self.cumulative_delta[symbol] += delta
        
        # Store for potential rollback if needed
        self.trade_history[symbol].append(tick)
        self.delta_changes[symbol].append(delta)
        
        return self.cumulative_delta[symbol]
    
    def get_cvd(self, symbol: str) -> float:
        """Get current CVD for symbol."""
        return self.cumulative_delta.get(symbol, 0.0)
    
    def get_cvd_divergence(self, symbol: str, price_change: float) -> float:
        """
        Detect divergence between price action and CVD.
        Positive = bullish divergence, Negative = bearish divergence
        """
        if len(self.delta_changes.get(symbol, [])) < 100:
            return 0.0
        
        recent_cvd_change = sum(list(self.delta_changes[symbol])[-50:])
        expected_direction = 1 if price_change > 0 else -1
        actual_direction = 1 if recent_cvd_change > 0 else -1
        
        if expected_direction != actual_direction:
            return recent_cvd_change * expected_direction
        
        return 0.0
    
    def reset(self, symbol: str):
        """Reset CVD for specific symbol."""
        if symbol in self.cumulative_delta:
            self.cumulative_delta[symbol] = 0.0
            self.trade_history[symbol].clear()
            self.delta_changes[symbol].clear()


class DeltaAnalyzer:
    """
    Analyzes bid-ask delta and volume imbalances.
    Critical for detecting aggressive buying/selling.
    """
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback = lookback_periods
        self.volume_deltas: Dict[str, Deque[float]] = {}
        self.price_deltas: Dict[str, Deque[float]] = {}
        
    def update(self, symbol: str, buy_volume: float, sell_volume: float, 
               price_change: float) -> Dict[str, float]:
        """Update delta metrics and return analysis."""
        if symbol not in self.volume_deltas:
            self.volume_deltas[symbol] = deque(maxlen=self.lookback)
            self.price_deltas[symbol] = deque(maxlen=self.lookback)
        
        volume_delta = buy_volume - sell_volume
        self.volume_deltas[symbol].append(volume_delta)
        self.price_deltas[symbol].append(price_change)
        
        return {
            'current_delta': volume_delta,
            'avg_delta': sum(self.volume_deltas[symbol]) / len(self.volume_deltas[symbol]),
            'delta_sum': sum(self.volume_deltas[symbol]),
            'correlation': self._calculate_correlation(symbol)
        }
    
    def _calculate_correlation(self, symbol: str) -> float:
        """Calculate correlation between volume delta and price change."""
        if len(self.volume_deltas[symbol]) < 10:
            return 0.0
        
        vol_list = list(self.volume_deltas[symbol])
        price_list = list(self.price_deltas[symbol])
        
        n = len(vol_list)
        mean_vol = sum(vol_list) / n
        mean_price = sum(price_list) / n
        
        numerator = sum((v - mean_vol) * (p - mean_price) 
                       for v, p in zip(vol_list, price_list))
        
        var_vol = sum((v - mean_vol) ** 2 for v in vol_list)
        var_price = sum((p - mean_price) ** 2 for p in price_list)
        
        denominator = (var_vol * var_price) ** 0.5
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


class FootprintChart:
    """
    Generates footprint charts showing buy/sell volume at each price level.
    Memory optimized using sparse representation.
    """
    
    def __init__(self, price_precision: int = 2, max_levels: int = 100):
        self.price_precision = price_precision
        self.max_levels = max_levels
        self.footprint_data: Dict[str, Dict[float, FootprintCell]] = {}
        self.active_ranges: Dict[str, Tuple[float, float]] = {}
        
    def add_trade(self, symbol: str, price: float, quantity: float, 
                  side: TradeSide) -> None:
        """Add trade to footprint chart."""
        if symbol not in self.footprint_data:
            self.footprint_data[symbol] = {}
            self.active_ranges[symbol] = (price, price)
        
        # Round price to precision level
        rounded_price = round(price, self.price_precision)
        
        if rounded_price not in self.footprint_data[symbol]:
            self.footprint_data[symbol][rounded_price] = FootprintCell()
        
        cell = self.footprint_data[symbol][rounded_price]
        
        if side == TradeSide.BUY:
            cell.buy_volume += quantity
        elif side == TradeSide.SELL:
            cell.sell_volume += quantity
        
        cell.trade_count += 1
        cell.high = max(cell.high, price)
        cell.low = min(cell.low, price)
        
        # Update active range
        min_price, max_price = self.active_ranges[symbol]
        self.active_ranges[symbol] = (
            min(min_price, rounded_price),
            max(max_price, rounded_price)
        )
        
        # Prune old levels if exceeding max
        self._prune_levels(symbol)
    
    def _prune_levels(self, symbol: str) -> None:
        """Remove price levels outside active range to save memory."""
        if symbol not in self.footprint_data:
            return
            
        levels = self.footprint_data[symbol]
        if len(levels) <= self.max_levels:
            return
        
        # Keep only levels near current price range
        sorted_prices = sorted(levels.keys())
        min_p, max_p = self.active_ranges[symbol]
        range_size = max_p - min_p
        
        # Remove levels that are too far from current range
        to_remove = []
        for price in sorted_prices:
            if abs(price - max_p) > range_size * 2 or abs(price - min_p) > range_size * 2:
                to_remove.append(price)
        
        for price in to_remove[:len(levels) - self.max_levels]:
            del levels[price]
    
    def get_imbalance_levels(self, symbol: str, threshold: float = 3.0) -> List[float]:
        """Find price levels with significant buy/sell imbalance."""
        if symbol not in self.footprint_data:
            return []
        
        imbalances = []
        for price, cell in self.footprint_data[symbol].items():
            if cell.imbalance_ratio >= threshold or cell.imbalance_ratio <= 1/threshold:
                imbalances.append(price)
        
        return sorted(imbalances)
    
    def get_poc(self, symbol: str) -> Optional[float]:
        """Get Point of Control - price level with highest volume."""
        if symbol not in self.footprint_data or not self.footprint_data[symbol]:
            return None
        
        max_volume = 0
        poc = None
        
        for price, cell in self.footprint_data[symbol].items():
            if cell.total_volume > max_volume:
                max_volume = cell.total_volume
                poc = price
        
        return poc


class OrderFlowEngine:
    """
    Main engine coordinating all order flow analytics.
    Singleton pattern for global access.
    """
    
    _instance: Optional['OrderFlowEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.cvd = CumulativeVolumeDelta()
        self.delta_analyzer = DeltaAnalyzer()
        self.footprint = FootprintChart()
        self.last_prices: Dict[str, float] = {}
        self._initialized = True
        
        logger.info("OrderFlowEngine initialized")
    
    def process_trade(self, symbol: str, price: float, quantity: float, 
                      is_buyer_maker: bool, trade_id: str) -> Dict[str, any]:
        """
        Process incoming trade and update all analytics.
        Returns comprehensive order flow metrics.
        """
        start_time = time.perf_counter()
        
        # Determine trade side
        side = TradeSide.SELL if is_buyer_maker else TradeSide.BUY
        
        # Create trade tick
        tick = TradeTick(
            timestamp=time.time(),
            price=price,
            quantity=quantity,
            side=side,
            trade_id=trade_id
        )
        
        # Update CVD
        cvd_value = self.cvd.add_trade(symbol, tick)
        
        # Calculate price change
        price_change = 0.0
        if symbol in self.last_prices:
            price_change = price - self.last_prices[symbol]
        self.last_prices[symbol] = price
        
        # Estimate buy/sell volumes (simplified - would need order book for accuracy)
        buy_vol = quantity if side == TradeSide.BUY else 0
        sell_vol = quantity if side == TradeSide.SELL else 0
        
        # Update delta analyzer
        delta_metrics = self.delta_analyzer.update(symbol, buy_vol, sell_vol, price_change)
        
        # Update footprint
        self.footprint.add_trade(symbol, price, quantity, side)
        
        # Get key levels
        poc = self.footprint.get_poc(symbol)
        imbalances = self.footprint.get_imbalance_levels(symbol)
        
        # Check for divergence
        divergence = self.cvd.get_cvd_divergence(symbol, price_change)
        
        processing_time = time.perf_counter() - start_time
        
        return {
            'symbol': symbol,
            'cvd': cvd_value,
            'delta': delta_metrics['current_delta'],
            'avg_delta': delta_metrics['avg_delta'],
            'delta_correlation': delta_metrics['correlation'],
            'poc': poc,
            'imbalance_levels': imbalances,
            'divergence': divergence,
            'processing_time_us': processing_time * 1_000_000,
            'timestamp': time.time()
        }
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, any]:
        """Get complete order flow analysis for a symbol."""
        return {
            'cvd': self.cvd.get_cvd(symbol),
            'divergence': self.cvd.get_cvd_divergence(symbol, 
                            self.last_prices.get(symbol, 0)),
            'poc': self.footprint.get_poc(symbol),
            'imbalance_levels': self.footprint.get_imbalance_levels(symbol),
            'last_price': self.last_prices.get(symbol)
        }
    
    def reset_symbol(self, symbol: str):
        """Reset all analytics for a specific symbol."""
        self.cvd.reset(symbol)
        if symbol in self.delta_analyzer.volume_deltas:
            self.delta_analyzer.volume_deltas[symbol].clear()
            self.delta_analyzer.price_deltas[symbol].clear()
        if symbol in self.footprint.footprint_data:
            self.footprint.footprint_data[symbol].clear()
        if symbol in self.last_prices:
            del self.last_prices[symbol]
        
        logger.info(f"Reset order flow analytics for {symbol}")


# Async wrapper for integration with event loops
async def process_trade_async(engine: OrderFlowEngine, symbol: str, price: float,
                              quantity: float, is_buyer_maker: bool, trade_id: str):
    """Async wrapper for trade processing."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        engine.process_trade, 
        symbol, price, quantity, is_buyer_maker, trade_id
    )
    return result


if __name__ == "__main__":
    # Example usage and testing
    engine = OrderFlowEngine()
    
    # Simulate trades
    test_trades = [
        ("BTCUSDT", 45000.0, 0.5, False, "t1"),
        ("BTCUSDT", 45001.0, 0.3, True, "t2"),
        ("BTCUSDT", 45000.5, 0.8, False, "t3"),
        ("ETHUSDT", 3200.0, 2.0, False, "t4"),
        ("ETHUSDT", 3199.0, 1.5, True, "t5"),
    ]
    
    for symbol, price, qty, is_buyer_maker, tid in test_trades:
        result = engine.process_trade(symbol, price, qty, is_buyer_maker, tid)
        print(f"{symbol}: CVD={result['cvd']:.2f}, Delta={result['delta']:.2f}, "
              f"POC={result['poc']}, Time={result['processing_time_us']:.1f}µs")
