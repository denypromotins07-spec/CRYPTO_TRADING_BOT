"""
Pattern Recognition Engine for ZAID Personal Crypto Trading Bot
Detects candlestick patterns and chart patterns automatically
Uses vectorized NumPy operations for efficiency
Memory-optimized sliding window implementation

Part of the 152 domains of quantitative finance implementation.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CandlestickPattern(Enum):
    """Single and multi-candle patterns."""
    DOJI = "DOJI"
    HAMMER = "HAMMER"
    INVERTED_HAMMER = "INVERTED_HAMMER"
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    MORNING_STAR = "MORNING_STAR"
    EVENING_STAR = "EVENING_STAR"
    THREE_WHITE_SOLDIERS = "THREE_WHITE_SOLDIERS"
    THREE_BLACK_CROWS = "THREE_BLACK_CROWS"
    HARAMI_BULLISH = "HARAMI_BULLISH"
    HARAMI_BEARISH = "HARAMI_BEARISH"
    PIERCING_LINE = "PIERCING_LINE"
    DARK_CLOUD_COVER = "DARK_CLOUD_COVER"
    SHOOTING_STAR = "SHOOTING_STAR"


class ChartPattern(Enum):
    """Multi-candle chart patterns."""
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    HEAD_AND_SHOULDERS = "HEAD_AND_SHOULDERS"
    INVERSE_HEAD_AND_SHOULDERS = "INVERSE_HEAD_AND_SHOULDERS"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRICAL_TRIANGLE = "SYMMETRICAL_TRIANGLE"
    FLAG_BULLISH = "FLAG_BULLISH"
    FLAG_BEARISH = "FLAG_BEARISH"
    WEDGE_RISING = "WEDGE_RISING"
    WEDGE_FALLING = "WEDGE_FALLING"


@dataclass
class PatternResult:
    """Detected pattern with metadata."""
    pattern_name: str
    pattern_type: str  # 'candlestick' or 'chart'
    signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float  # 0.0 to 1.0
    timestamp: float
    confirmation: bool
    metadata: Dict = field(default_factory=dict)


class CandlestickPatternDetector:
    """
    Detects single and multi-candle patterns.
    Optimized for real-time detection.
    """
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
        self.opens: Deque[float] = deque(maxlen=lookback + 5)
        self.highs: Deque[float] = deque(maxlen=lookback + 5)
        self.lows: Deque[float] = deque(maxlen=lookback + 5)
        self.closes: Deque[float] = deque(maxlen=lookback + 5)
        
    def update(self, open_: float, high: float, low: float, 
               close: float) -> List[PatternResult]:
        """Update with new candle and detect patterns."""
        self.opens.append(open_)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        patterns = []
        
        if len(self.closes) < 1:
            return patterns
        
        import time
        current_time = time.time()
        
        # Single candle patterns
        if len(self.closes) >= 1:
            patterns.extend(self._detect_single_candle(current_time))
        
        # Two-candle patterns
        if len(self.closes) >= 2:
            patterns.extend(self._detect_two_candle_patterns(current_time))
        
        # Three-candle patterns
        if len(self.closes) >= 3:
            patterns.extend(self._detect_three_candle_patterns(current_time))
        
        return patterns
    
    def _body_size(self, open_: float, close: float) -> float:
        return abs(close - open_)
    
    def _upper_shadow(self, high: float, open_: float, close: float) -> float:
        return high - max(open_, close)
    
    def _lower_shadow(self, low: float, open_: float, close: float) -> float:
        return min(open_, close) - low
    
    def _is_bullish(self, open_: float, close: float) -> bool:
        return close > open_
    
    def _is_bearish(self, open_: float, close: float) -> bool:
        return close < open_
    
    def _detect_single_candle(self, timestamp: float) -> List[PatternResult]:
        """Detect single candle patterns (Doji, Hammer, etc.)."""
        patterns = []
        
        o = self.opens[-1]
        h = self.highs[-1]
        l = self.lows[-1]
        c = self.closes[-1]
        
        body = self._body_size(o, c)
        upper_shadow = self._upper_shadow(h, o, c)
        lower_shadow = self._lower_shadow(l, o, c)
        total_range = h - l
        
        if total_range == 0:
            return patterns
        
        # Doji: Very small body relative to range
        if body / total_range < 0.1:
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.DOJI.value,
                pattern_type='candlestick',
                signal='NEUTRAL',
                strength=0.5,
                timestamp=timestamp,
                confirmation=True,
                metadata={'body_ratio': body / total_range}
            ))
        
        # Hammer: Small body at top, long lower shadow
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            signal = 'BULLISH' if self._is_bullish(o, c) else 'NEUTRAL'
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.HAMMER.value,
                pattern_type='candlestick',
                signal=signal,
                strength=min(1.0, lower_shadow / total_range),
                timestamp=timestamp,
                confirmation=signal == 'BULLISH',
                metadata={'lower_shadow_ratio': lower_shadow / total_range}
            ))
        
        # Shooting Star: Small body at bottom, long upper shadow
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            signal = 'BEARISH' if self._is_bearish(o, c) else 'NEUTRAL'
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.SHOOTING_STAR.value,
                pattern_type='candlestick',
                signal=signal,
                strength=min(1.0, upper_shadow / total_range),
                timestamp=timestamp,
                confirmation=signal == 'BEARISH',
                metadata={'upper_shadow_ratio': upper_shadow / total_range}
            ))
        
        return patterns
    
    def _detect_two_candle_patterns(self, timestamp: float) -> List[PatternResult]:
        """Detect two-candle patterns (Engulfing, Harami, etc.)."""
        patterns = []
        
        o1, h1, l1, c1 = self.opens[-2], self.highs[-2], self.lows[-2], self.closes[-2]
        o2, h2, l2, c2 = self.opens[-1], self.highs[-1], self.lows[-1], self.closes[-1]
        
        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)
        
        # Bullish Engulfing
        if (self._is_bearish(o1, c1) and self._is_bullish(o2, c2) and
            o2 < c1 and c2 > o1 and body2 > body1):
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.BULLISH_ENGULFING.value,
                pattern_type='candlestick',
                signal='BULLISH',
                strength=min(1.0, body2 / body1),
                timestamp=timestamp,
                confirmation=True,
                metadata={'engulfment_ratio': body2 / body1}
            ))
        
        # Bearish Engulfing
        if (self._is_bullish(o1, c1) and self._is_bearish(o2, c2) and
            o2 > c1 and c2 < o1 and body2 > body1):
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.BEARISH_ENGULFING.value,
                pattern_type='candlestick',
                signal='BEARISH',
                strength=min(1.0, body2 / body1),
                timestamp=timestamp,
                confirmation=True,
                metadata={'engulfment_ratio': body2 / body1}
            ))
        
        # Bullish Harami
        if (self._is_bearish(o1, c1) and self._is_bullish(o2, c2) and
            o2 > o1 and c2 < c1 and body2 < body1 * 0.5):
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.HARAMI_BULLISH.value,
                pattern_type='candlestick',
                signal='BULLISH',
                strength=0.6,
                timestamp=timestamp,
                confirmation=False,  # Needs confirmation
                metadata={}
            ))
        
        # Bearish Harami
        if (self._is_bullish(o1, c1) and self._is_bearish(o2, c2) and
            o2 < o1 and c2 > c1 and body2 < body1 * 0.5):
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.HARAMI_BEARISH.value,
                pattern_type='candlestick',
                signal='BEARISH',
                strength=0.6,
                timestamp=timestamp,
                confirmation=False,
                metadata={}
            ))
        
        return patterns
    
    def _detect_three_candle_patterns(self, timestamp: float) -> List[PatternResult]:
        """Detect three-candle patterns (Morning/Evening Star, etc.)."""
        patterns = []
        
        o1, c1 = self.opens[-3], self.closes[-3]
        o2, c2 = self.opens[-2], self.closes[-2]
        o3, c3 = self.opens[-1], self.closes[-1]
        
        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)
        body3 = self._body_size(o3, c3)
        
        # Morning Star
        if (self._is_bearish(o1, c1) and body2 < body1 * 0.5 and
            self._is_bullish(o3, c3) and c3 > (o1 + c1) / 2):
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.MORNING_STAR.value,
                pattern_type='candlestick',
                signal='BULLISH',
                strength=min(1.0, body3 / body1),
                timestamp=timestamp,
                confirmation=True,
                metadata={}
            ))
        
        # Evening Star
        if (self._is_bullish(o1, c1) and body2 < body1 * 0.5 and
            self._is_bearish(o3, c3) and c3 < (o1 + c1) / 2):
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.EVENING_STAR.value,
                pattern_type='candlestick',
                signal='BEARISH',
                strength=min(1.0, body3 / body1),
                timestamp=timestamp,
                confirmation=True,
                metadata={}
            ))
        
        # Three White Soldiers
        if (self._is_bullish(o1, c1) and self._is_bullish(o2, c2) and
            self._is_bullish(o3, c3) and
            c1 > o1 and c2 > c1 and c3 > c2):
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.THREE_WHITE_SOLDIERS.value,
                pattern_type='candlestick',
                signal='BULLISH',
                strength=0.8,
                timestamp=timestamp,
                confirmation=True,
                metadata={}
            ))
        
        # Three Black Crows
        if (self._is_bearish(o1, c1) and self._is_bearish(o2, c2) and
            self._is_bearish(o3, c3) and
            c1 < o1 and c2 < c1 and c3 < c2):
            patterns.append(PatternResult(
                pattern_name=CandlestickPattern.THREE_BLACK_CROWS.value,
                pattern_type='candlestick',
                signal='BEARISH',
                strength=0.8,
                timestamp=timestamp,
                confirmation=True,
                metadata={}
            ))
        
        return patterns


class ChartPatternDetector:
    """
    Detects multi-candle chart patterns.
    Uses swing point analysis for pattern recognition.
    """
    
    def __init__(self, lookback: int = 50, swing_lookback: int = 5):
        self.lookback = lookback
        self.swing_lookback = swing_lookback
        self.highs: Deque[float] = deque(maxlen=lookback)
        self.lows: Deque[float] = deque(maxlen=lookback)
        self.closes: Deque[float] = deque(maxlen=lookback)
        self.timestamps: Deque[float] = deque(maxlen=lookback)
        self.swing_highs: List[Tuple[float, float]] = []  # (price, timestamp)
        self.swing_lows: List[Tuple[float, float]] = []
        
    def update(self, high: float, low: float, close: float, 
               timestamp: float) -> List[PatternResult]:
        """Update with new candle and detect chart patterns."""
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.timestamps.append(timestamp)
        
        # Detect swing points
        self._update_swing_points()
        
        patterns = []
        
        # Need enough data for chart patterns
        if len(self.swing_highs) >= 2 or len(self.swing_lows) >= 2:
            patterns.extend(self._detect_reversal_patterns(timestamp))
            patterns.extend(self._detect_continuation_patterns(timestamp))
        
        return patterns
    
    def _update_swing_points(self):
        """Detect and store swing highs and lows."""
        if len(self.highs) < self.swing_lookback * 2 + 1:
            return
        
        idx = len(self.highs) - 1
        lookback = self.swing_lookback
        
        # Check for swing high
        is_swing_high = True
        current_high = self.highs[idx]
        
        for i in range(idx - lookback, idx):
            if i >= 0 and self.highs[i] >= current_high:
                is_swing_high = False
                break
        
        for i in range(idx + 1, min(idx + lookback + 1, len(self.highs))):
            if self.highs[i] >= current_high:
                is_swing_high = False
                break
        
        if is_swing_high and len(self.highs) >= idx + lookback:
            self.swing_highs.append((current_high, self.timestamps[idx]))
            if len(self.swing_highs) > 20:
                self.swing_highs.pop(0)
        
        # Check for swing low
        is_swing_low = True
        current_low = self.lows[idx]
        
        for i in range(idx - lookback, idx):
            if i >= 0 and self.lows[i] <= current_low:
                is_swing_low = False
                break
        
        for i in range(idx + 1, min(idx + lookback + 1, len(self.lows))):
            if self.lows[i] <= current_low:
                is_swing_low = False
                break
        
        if is_swing_low and len(self.lows) >= idx + lookback:
            self.swing_lows.append((current_low, self.timestamps[idx]))
            if len(self.swing_lows) > 20:
                self.swing_lows.pop(0)
    
    def _detect_reversal_patterns(self, timestamp: float) -> List[PatternResult]:
        """Detect reversal patterns (Double Top/Bottom, H&S)."""
        patterns = []
        
        # Double Top
        if len(self.swing_highs) >= 2:
            h1, t1 = self.swing_highs[-1]
            h2, t2 = self.swing_highs[-2]
            
            if abs(h1 - h2) / h1 < 0.01:  # Within 1%
                patterns.append(PatternResult(
                    pattern_name=ChartPattern.DOUBLE_TOP.value,
                    pattern_type='chart',
                    signal='BEARISH',
                    strength=0.7,
                    timestamp=timestamp,
                    confirmation=False,
                    metadata={'level1': h1, 'level2': h2}
                ))
        
        # Double Bottom
        if len(self.swing_lows) >= 2:
            l1, t1 = self.swing_lows[-1]
            l2, t2 = self.swing_lows[-2]
            
            if abs(l1 - l2) / l1 < 0.01:  # Within 1%
                patterns.append(PatternResult(
                    pattern_name=ChartPattern.DOUBLE_BOTTOM.value,
                    pattern_type='chart',
                    signal='BULLISH',
                    strength=0.7,
                    timestamp=timestamp,
                    confirmation=False,
                    metadata={'level1': l1, 'level2': l2}
                ))
        
        # Head and Shoulders (simplified)
        if len(self.swing_highs) >= 3:
            h1, _ = self.swing_highs[-3]
            h2, _ = self.swing_highs[-2]
            h3, _ = self.swing_highs[-1]
            
            if h2 > h1 and h2 > h3 and abs(h1 - h3) / h1 < 0.05:
                patterns.append(PatternResult(
                    pattern_name=ChartPattern.HEAD_AND_SHOULDERS.value,
                    pattern_type='chart',
                    signal='BEARISH',
                    strength=0.75,
                    timestamp=timestamp,
                    confirmation=False,
                    metadata={'left_shoulder': h1, 'head': h2, 'right_shoulder': h3}
                ))
        
        # Inverse Head and Shoulders
        if len(self.swing_lows) >= 3:
            l1, _ = self.swing_lows[-3]
            l2, _ = self.swing_lows[-2]
            l3, _ = self.swing_lows[-1]
            
            if l2 < l1 and l2 < l3 and abs(l1 - l3) / l1 < 0.05:
                patterns.append(PatternResult(
                    pattern_name=ChartPattern.INVERSE_HEAD_AND_SHOULDERS.value,
                    pattern_type='chart',
                    signal='BULLISH',
                    strength=0.75,
                    timestamp=timestamp,
                    confirmation=False,
                    metadata={'left_shoulder': l1, 'head': l2, 'right_shoulder': l3}
                ))
        
        return patterns
    
    def _detect_continuation_patterns(self, timestamp: float) -> List[PatternResult]:
        """Detect continuation patterns (Triangles, Flags)."""
        patterns = []
        
        # Simplified triangle detection
        if len(self.swing_highs) >= 2 and len(self.swing_lows) >= 2:
            # Check for converging highs and lows (symmetrical triangle)
            recent_highs = [h for h, _ in self.swing_highs[-3:]]
            recent_lows = [l for l, _ in self.swing_lows[-3:]]
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                high_slope = (recent_highs[-1] - recent_highs[0]) / max(1, len(recent_highs))
                low_slope = (recent_lows[-1] - recent_lows[0]) / max(1, len(recent_lows))
                
                # Symmetrical Triangle: Lower highs, higher lows
                if high_slope < -0.001 and low_slope > 0.001:
                    patterns.append(PatternResult(
                        pattern_name=ChartPattern.SYMMETRICAL_TRIANGLE.value,
                        pattern_type='chart',
                        signal='NEUTRAL',
                        strength=0.5,
                        timestamp=timestamp,
                        confirmation=False,
                        metadata={'high_slope': high_slope, 'low_slope': low_slope}
                    ))
                
                # Ascending Triangle: Flat highs, higher lows
                elif abs(high_slope) < 0.001 and low_slope > 0.001:
                    patterns.append(PatternResult(
                        pattern_name=ChartPattern.ASCENDING_TRIANGLE.value,
                        pattern_type='chart',
                        signal='BULLISH',
                        strength=0.6,
                        timestamp=timestamp,
                        confirmation=False,
                        metadata={}
                    ))
                
                # Descending Triangle: Lower highs, flat lows
                elif high_slope < -0.001 and abs(low_slope) < 0.001:
                    patterns.append(PatternResult(
                        pattern_name=ChartPattern.DESCENDING_TRIANGLE.value,
                        pattern_type='chart',
                        signal='BEARISH',
                        strength=0.6,
                        timestamp=timestamp,
                        confirmation=False,
                        metadata={}
                    ))
        
        return patterns


class PatternRecognitionEngine:
    """
    Main engine coordinating all pattern detection.
    Singleton pattern for global access.
    """
    
    _instance: Optional['PatternRecognitionEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.candlestick_detectors: Dict[str, CandlestickPatternDetector] = {}
        self.chart_detectors: Dict[str, ChartPatternDetector] = {}
        
        self.default_candle_lookback = 10
        self.default_chart_lookback = 50
        
        self.recent_patterns: Dict[str, deque] = {}
        self.max_recent = 100
        
        self._initialized = True
        logger.info("PatternRecognitionEngine initialized")
    
    def get_or_create_detectors(self, symbol: str):
        """Get or create pattern detectors for a symbol."""
        if symbol not in self.candlestick_detectors:
            self.candlestick_detectors[symbol] = CandlestickPatternDetector(
                self.default_candle_lookback
            )
            self.chart_detectors[symbol] = ChartPatternDetector(
                self.default_chart_lookback
            )
            self.recent_patterns[symbol] = deque(maxlen=self.max_recent)
    
    def update_candle(self, symbol: str, open_: float, high: float,
                      low: float, close: float, volume: float,
                      timestamp: float) -> List[PatternResult]:
        """Update all pattern detectors with new candle."""
        self.get_or_create_detectors(symbol)
        
        all_patterns = []
        
        # Candlestick patterns
        cs_patterns = self.candlestick_detectors[symbol].update(
            open_, high, low, close
        )
        all_patterns.extend(cs_patterns)
        
        # Chart patterns
        chart_patterns = self.chart_detectors[symbol].update(
            high, low, close, timestamp
        )
        all_patterns.extend(chart_patterns)
        
        # Store recent patterns
        for pattern in all_patterns:
            self.recent_patterns[symbol].append(pattern)
        
        return all_patterns
    
    def get_recent_patterns(self, symbol: str, limit: int = 10) -> List[PatternResult]:
        """Get recent patterns for a symbol."""
        if symbol not in self.recent_patterns:
            return []
        return list(self.recent_patterns[symbol])[-limit:]
    
    def get_pattern_summary(self, symbol: str) -> Dict[str, any]:
        """Get summary of recent pattern signals."""
        recent = self.get_recent_patterns(symbol, 20)
        
        bullish_count = sum(1 for p in recent if p.signal == 'BULLISH')
        bearish_count = sum(1 for p in recent if p.signal == 'BEARISH')
        
        total = bullish_count + bearish_count
        if total == 0:
            bias = 'NEUTRAL'
            confidence = 0.0
        else:
            if bullish_count > bearish_count:
                bias = 'BULLISH'
                confidence = bullish_count / total
            else:
                bias = 'BEARISH'
                confidence = bearish_count / total
        
        return {
            'symbol': symbol,
            'bias': bias,
            'confidence': confidence,
            'bullish_patterns': bullish_count,
            'bearish_patterns': bearish_count,
            'total_patterns': total
        }


if __name__ == "__main__":
    # Example usage
    engine = PatternRecognitionEngine()
    
    # Simulate candles
    test_candles = [
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
        (103, 105, 102, 95),  # Potential reversal
        (95, 97, 94, 96),
        (96, 98, 95, 97),
    ]
    
    import time
    for i, (o, h, l, c) in enumerate(test_candles):
        patterns = engine.update_candle("BTCUSDT", o, h, l, c, 1000, time.time())
        if patterns:
            print(f"Candle {i+1}: {[p.pattern_name for p in patterns]}")
    
    summary = engine.get_pattern_summary("BTCUSDT")
    print(f"Summary: {summary}")
