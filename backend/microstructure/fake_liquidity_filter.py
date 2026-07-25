#!/usr/bin/env python3
"""
Fake Liquidity Filter - Purging Spoofed Walls from Order Book

This module identifies and filters out fake liquidity (spoofed orders)
from the local order book representation to prevent adverse selection.
It works in conjunction with the SpoofingDetector and LayeringAnalyzer.

Designed for the ZAID PERSONAL CRYPTO TRADING BOT with strict 8GB RAM constraints.
Provides real-time filtering with minimal latency impact.

Author: Opus 4.8
Stage: 21/100 - Advanced Market Microstructure
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time
import numpy as np


class LiquidityType(Enum):
    """Classification of liquidity quality."""
    GENUINE = auto()       # Real, executable liquidity
    SUSPICIOUS = auto()    # Potentially fake, monitor closely
    SPOOFED = auto()       # Confirmed fake, should be ignored
    UNKNOWN = auto()       # Not yet classified


@dataclass(slots=True)
class LiquidityLevel:
    """Represents a price level with liquidity classification."""
    price: float
    side: str  # 'bid' or 'ask'
    volume: float
    order_count: int
    liquidity_type: LiquidityType = LiquidityType.UNKNOWN
    confidence: float = 0.0
    first_seen_ns: int = 0
    last_updated_ns: int = 0
    cancel_count: int = 0
    fill_count: int = 0
    
    @property
    def age_ms(self) -> float:
        """Age of this level in milliseconds."""
        return (time.time_ns() - self.first_seen_ns) / 1_000_000
    
    @property
    def is_reliable(self) -> bool:
        """Check if this liquidity is reliable for trading."""
        return self.liquidity_type == LiquidityType.GENUINE and self.confidence > 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'price': self.price,
            'side': self.side,
            'volume': self.volume,
            'order_count': self.order_count,
            'type': self.liquidity_type.name,
            'confidence': self.confidence,
            'age_ms': self.age_ms,
        }


@dataclass(slots=True)
class FilteredBook:
    """Order book with fake liquidity filtered out."""
    bids: List[LiquidityLevel]
    asks: List[LiquidityLevel]
    genuine_bid_volume: float
    genuine_ask_volume: float
    total_bid_volume: float
    total_ask_volume: float
    filtered_bid_volume: float  # Volume removed as fake
    filtered_ask_volume: float
    timestamp_ns: int
    
    @property
    def bid_filter_ratio(self) -> float:
        """Ratio of fake volume to total bid volume."""
        if self.total_bid_volume == 0:
            return 0.0
        return self.filtered_bid_volume / self.total_bid_volume
    
    @property
    def ask_filter_ratio(self) -> float:
        """Ratio of fake volume to total ask volume."""
        if self.total_ask_volume == 0:
            return 0.0
        return self.filtered_ask_volume / self.total_ask_volume


class FakeLiquidityFilter:
    """
    Filters fake/spoofed liquidity from the order book.
    
    Uses multiple signals to classify liquidity:
    - Order lifetime analysis
    - Cancel/fill ratios
    - Pattern recognition from spoofing detector
    - Volume anomalies
    
    Provides both raw and filtered book views.
    """
    
    # Classification thresholds
    SHORT_LIFETIME_THRESHOLD_MS: float = 50.0
    HIGH_CANCEL_RATIO: float = 0.8
    VOLUME_ANOMALY_MULTIPLE: float = 5.0
    CONFIDENCE_DECAY_RATE: float = 0.99  # Per update
    
    def __init__(
        self,
        symbols: List[str],
        sensitivity: float = 0.7,
    ) -> None:
        """
        Initialize the fake liquidity filter.
        
        Args:
            symbols: List of symbols to monitor
            sensitivity: Filter sensitivity (0.0 to 1.0)
        """
        self.symbols: Set[str] = set(symbols)
        self.sensitivity: float = max(0.0, min(1.0, sensitivity))
        
        # Current book state per symbol
        self._books: Dict[str, Dict[float, LiquidityLevel]] = {
            sym: {} for sym in symbols
        }
        
        # Historical statistics per price level
        self._level_stats: Dict[str, Dict[float, Dict[str, Any]]] = {
            sym: {} for sym in symbols
        }
        
        # Detected spoofing signals (from external detector)
        self._spoof_signals: deque[Dict[str, Any]] = deque(maxlen=1000)
        
        # Filtered book cache
        self._filtered_cache: Dict[str, FilteredBook] = {}
        
        # Volume statistics for anomaly detection
        self._volume_stats: Dict[str, Dict[str, float]] = {
            sym: {'mean_volume': 1.0, 'std_volume': 0.5}
            for sym in symbols
        }
    
    @staticmethod
    def _now_ns() -> int:
        """Get current time in nanoseconds."""
        return time.time_ns()
    
    def update_level(
        self,
        symbol: str,
        price: float,
        side: str,
        volume: float,
        order_count: int,
    ) -> LiquidityLevel:
        """
        Update a price level in the book.
        
        Args:
            symbol: Trading pair symbol
            price: Price level
            side: 'bid' or 'ask'
            volume: Volume at level
            order_count: Number of orders
        
        Returns:
            Updated LiquidityLevel with classification
        """
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not monitored")
        
        now = self._now_ns()
        book = self._books[symbol]
        
        if price in book:
            level = book[price]
            level.volume = volume
            level.order_count = order_count
            level.last_updated_ns = now
            
            # Decay confidence over time without confirmation
            level.confidence *= self.CONFIDENCE_DECAY_RATE
        else:
            level = LiquidityLevel(
                price=price,
                side=side,
                volume=volume,
                order_count=order_count,
                first_seen_ns=now,
                last_updated_ns=now,
                confidence=0.5,  # Start neutral
            )
            book[price] = level
        
        # Update statistics
        self._update_level_stats(symbol, price, level)
        
        # Classify the level
        self._classify_level(level, symbol)
        
        # Update filtered cache
        self._update_filtered_book(symbol)
        
        return level
    
    def remove_level(self, symbol: str, price: float, side: str) -> None:
        """Remove a price level (typically due to cancellation)."""
        if symbol in self._books and price in self._books[symbol]:
            level = self._books[symbol][price]
            
            # Record cancellation in stats
            self._record_cancellation(symbol, price)
            
            del self._books[symbol][price]
            self._update_filtered_book(symbol)
    
    def record_fill(self, symbol: str, price: float, side: str, volume: float) -> None:
        """Record a fill at a price level (confirms genuine liquidity)."""
        if symbol not in self._books or price not in self._books[symbol]:
            return
        
        level = self._books[symbol][price]
        level.fill_count += 1
        
        # Fills increase confidence significantly
        level.confidence = min(1.0, level.confidence + 0.2)
        
        if level.liquidity_type == LiquidityType.SPOOFED:
            level.liquidity_type = LiquidityType.SUSPICIOUS
        
        self._update_filtered_book(symbol)
    
    def add_spoof_signal(self, signal: Dict[str, Any]) -> None:
        """
        Add a spoofing signal from external detector.
        
        Args:
            signal: Spoofing signal dictionary with keys:
                - symbol, side, price_levels, confidence, type
        """
        self._spoof_signals.append(signal)
        
        # Mark affected levels as suspicious/spoofed
        symbol = signal.get('symbol', '')
        if symbol in self.symbols:
            price_levels = signal.get('price_levels', [])
            confidence = signal.get('confidence', 0.5)
            
            for price in price_levels:
                if price in self._books[symbol]:
                    level = self._books[symbol][price]
                    
                    if confidence > 0.8:
                        level.liquidity_type = LiquidityType.SPOOFED
                        level.confidence = confidence
                    elif confidence > 0.5:
                        level.liquidity_type = LiquidityType.SUSPICIOUS
                        level.confidence = max(level.confidence, confidence)
            
            self._update_filtered_book(symbol)
    
    def _update_level_stats(
        self,
        symbol: str,
        price: float,
        level: LiquidityLevel,
    ) -> None:
        """Update historical statistics for a price level."""
        if symbol not in self._level_stats:
            self._level_stats[symbol] = {}
        
        if price not in self._level_stats[symbol]:
            self._level_stats[symbol][price] = {
                'cancel_count': 0,
                'fill_count': 0,
                'total_updates': 0,
                'volumes': deque(maxlen=100),
                'lifetimes': deque(maxlen=50),
            }
        
        stats = self._level_stats[symbol][price]
        stats['total_updates'] += 1
        stats['volumes'].append(level.volume)
        
        # Update volume statistics
        volumes = list(stats['volumes'])
        if len(volumes) >= 10:
            self._volume_stats[symbol] = {
                'mean_volume': np.mean(volumes),
                'std_volume': np.std(volumes) + 1e-6,
            }
    
    def _record_cancellation(self, symbol: str, price: float) -> None:
        """Record a cancellation for statistics."""
        if symbol in self._level_stats and price in self._level_stats[symbol]:
            self._level_stats[symbol][price]['cancel_count'] += 1
    
    def _classify_level(self, level: LiquidityLevel, symbol: str) -> None:
        """
        Classify a liquidity level based on multiple factors.
        
        Updates level.liquidity_type and level.confidence in place.
        """
        now = self._now_ns()
        age_ms = level.age_ms
        
        # Get historical stats for this level
        stats = self._level_stats.get(symbol, {}).get(level.price, {})
        cancel_count = stats.get('cancel_count', 0)
        fill_count = stats.get('fill_count', 0)
        total_events = cancel_count + fill_count
        
        # Calculate cancel ratio
        if total_events > 0:
            cancel_ratio = cancel_count / total_events
        else:
            cancel_ratio = 0.5  # Unknown
        
        # Check for volume anomaly
        vol_stats = self._volume_stats.get(symbol, {'mean_volume': 1.0, 'std_volume': 0.5})
        z_score = abs(level.volume - vol_stats['mean_volume']) / vol_stats['std_volume']
        is_volume_anomaly = z_score > 2.0
        
        # Classification logic
        spoof_confidence = 0.0
        
        # Factor 1: Short lifetime without fills
        if age_ms < self.SHORT_LIFETIME_THRESHOLD_MS and fill_count == 0:
            spoof_confidence += 0.3 * self.sensitivity
        
        # Factor 2: High cancel ratio
        if cancel_ratio > self.HIGH_CANCEL_RATIO and total_events >= 3:
            spoof_confidence += 0.3 * self.sensitivity
        
        # Factor 3: Volume anomaly
        if is_volume_anomaly:
            spoof_confidence += 0.2 * self.sensitivity
        
        # Factor 4: Low order count with high volume (single large spoof order)
        if level.order_count == 1 and level.volume > vol_stats['mean_volume'] * 3:
            spoof_confidence += 0.2 * self.sensitivity
        
        # Determine classification
        if spoof_confidence > 0.7:
            level.liquidity_type = LiquidityType.SPOOFED
            level.confidence = min(1.0, spoof_confidence)
        elif spoof_confidence > 0.4:
            level.liquidity_type = LiquidityType.SUSPICIOUS
            level.confidence = spoof_confidence
        else:
            # Boost confidence for genuine-looking liquidity
            if fill_count > 0 or age_ms > 1000:
                level.liquidity_type = LiquidityType.GENUINE
                level.confidence = max(level.confidence, 0.8)
            else:
                level.liquidity_type = LiquidityType.UNKNOWN
                level.confidence = max(level.confidence, 0.5 - spoof_confidence)
    
    def _update_filtered_book(self, symbol: str) -> None:
        """Update the filtered book cache for a symbol."""
        if symbol not in self._books:
            return
        
        now = self._now_ns()
        
        # Filter bids and asks
        bids = []
        asks = []
        genuine_bid_vol = 0.0
        genuine_ask_vol = 0.0
        total_bid_vol = 0.0
        total_ask_vol = 0.0
        
        for level in self._books[symbol].values():
            if level.side == 'bid':
                total_bid_vol += level.volume
                if level.is_reliable:
                    bids.append(level)
                    genuine_bid_vol += level.volume
            else:
                total_ask_vol += level.volume
                if level.is_reliable:
                    asks.append(level)
                    genuine_ask_vol += level.volume
        
        # Sort by price
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        self._filtered_cache[symbol] = FilteredBook(
            bids=bids,
            asks=asks,
            genuine_bid_volume=genuine_bid_vol,
            genuine_ask_volume=genuine_ask_vol,
            total_bid_volume=total_bid_vol,
            total_ask_volume=total_ask_vol,
            filtered_bid_volume=total_bid_vol - genuine_bid_vol,
            filtered_ask_volume=total_ask_vol - genuine_ask_vol,
            timestamp_ns=now,
        )
    
    def get_filtered_book(self, symbol: str) -> Optional[FilteredBook]:
        """Get the filtered order book for a symbol."""
        return self._filtered_cache.get(symbol)
    
    def get_genuine_best_bid(self, symbol: str) -> Optional[LiquidityLevel]:
        """Get the best genuine bid (excluding fake liquidity)."""
        filtered = self.get_filtered_book(symbol)
        if filtered and filtered.bids:
            return filtered.bids[0]
        return None
    
    def get_genuine_best_ask(self, symbol: str) -> Optional[LiquidityLevel]:
        """Get the best genuine ask (excluding fake liquidity)."""
        filtered = self.get_filtered_book(symbol)
        if filtered and filtered.asks:
            return filtered.asks[0]
        return None
    
    def get_genuine_spread(self, symbol: str) -> Optional[float]:
        """Calculate spread using only genuine liquidity."""
        best_bid = self.get_genuine_best_bid(symbol)
        best_ask = self.get_genuine_best_ask(symbol)
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_fake_liquidity_ratio(self, symbol: str) -> Tuple[float, float]:
        """
        Get the ratio of fake liquidity on each side.
        
        Returns:
            Tuple of (bid_fake_ratio, ask_fake_ratio)
        """
        filtered = self.get_filtered_book(symbol)
        if not filtered:
            return 0.0, 0.0
        
        return filtered.bid_filter_ratio, filtered.ask_filter_ratio
    
    def should_ignore_level(self, symbol: str, price: float) -> bool:
        """Check if a price level should be ignored for trading decisions."""
        if symbol not in self._books or price not in self._books[symbol]:
            return False
        
        level = self._books[symbol][price]
        return level.liquidity_type == LiquidityType.SPOOFED
    
    def get_reliable_volume(self, symbol: str, side: str) -> float:
        """Get total reliable (genuine) volume on a side."""
        filtered = self.get_filtered_book(symbol)
        if not filtered:
            return 0.0
        
        return filtered.genuine_bid_volume if side == 'bid' else filtered.genuine_ask_volume
    
    def reset_symbol(self, symbol: str) -> None:
        """Reset all data for a symbol."""
        if symbol in self.symbols:
            self._books[symbol] = {}
            self._level_stats[symbol] = {}
            if symbol in self._filtered_cache:
                del self._filtered_cache[symbol]


def main() -> None:
    """Example usage of FakeLiquidityFilter."""
    filter_engine = FakeLiquidityFilter(symbols=['BTCUSDT', 'ETHUSDT'])
    
    # Simulate order book updates
    for i in range(10):
        bid_price = 50000.0 - i * 0.01
        ask_price = 50000.05 + i * 0.01
        
        # Add genuine liquidity (will age and potentially get fills)
        filter_engine.update_level('BTCUSDT', bid_price, 'bid', 1.5 + i * 0.1, 3)
        filter_engine.update_level('BTCUSDT', ask_price, 'ask', 1.2 + i * 0.1, 2)
    
    # Simulate a spoofing signal
    spoof_signal = {
        'symbol': 'BTCUSDT',
        'side': 'bid',
        'price_levels': [49999.95, 49999.94],
        'confidence': 0.85,
        'type': 'layering',
    }
    filter_engine.add_spoof_signal(spoof_signal)
    
    # Get filtered book
    filtered = filter_engine.get_filtered_book('BTCUSDT')
    if filtered:
        print(f"Genuine Bid Volume: {filtered.genuine_bid_volume:.2f}")
        print(f"Genuine Ask Volume: {filtered.genuine_ask_volume:.2f}")
        print(f"Filtered Bid Volume: {filtered.filtered_bid_volume:.2f}")
        print(f"Bid Filter Ratio: {filtered.bid_filter_ratio:.2%}")
        
        best_bid = filter_engine.get_genuine_best_bid('BTCUSDT')
        best_ask = filter_engine.get_genuine_best_ask('BTCUSDT')
        
        if best_bid and best_ask:
            print(f"Genuine Best Bid: {best_bid.price:.2f}")
            print(f"Genuine Best Ask: {best_ask.price:.2f}")
            print(f"Genuine Spread: {best_ask.price - best_bid.price:.2f}")


if __name__ == "__main__":
    main()
