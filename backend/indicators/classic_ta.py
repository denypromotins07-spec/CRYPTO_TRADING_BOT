"""
Classic Technical Analysis Indicators for ZAID Personal Crypto Trading Bot
Calculates RSI, MACD, VWAP, Ichimoku, Fibonacci retracements
Optimized with NumPy for vectorized operations
Memory-efficient sliding window implementations

Part of the 152 domains of quantitative finance implementation.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Standardized result container for all indicators."""
    name: str
    value: float
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0.0 to 1.0
    timestamp: float
    metadata: Dict = field(default_factory=dict)


class RSICalculator:
    """
    Relative Strength Index with configurable periods.
    Uses Wilder's smoothing method for calculation.
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.gains: Deque[float] = deque(maxlen=period + 1)
        self.losses: Deque[float] = deque(maxlen=period + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.prices: Deque[float] = deque(maxlen=period + 1)
        
    def update(self, price: float) -> Optional[IndicatorResult]:
        """Update RSI with new price and return result if ready."""
        self.prices.append(price)
        
        if len(self.prices) < 2:
            return None
        
        # Calculate price change
        change = price - self.prices[-2]
        gain = max(0, change)
        loss = abs(min(0, change))
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        if len(self.gains) < self.period:
            return None
        
        # First calculation - simple average
        if self.avg_gain is None:
            self.avg_gain = sum(self.gains) / self.period
            self.avg_loss = sum(self.losses) / self.period
        else:
            # Wilder's smoothing
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
        
        # Calculate RSI
        if self.avg_loss == 0:
            rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Determine signal
        if rsi >= 70:
            signal = 'SELL'
            strength = (rsi - 70) / 30
        elif rsi <= 30:
            signal = 'BUY'
            strength = (30 - rsi) / 30
        else:
            signal = 'NEUTRAL'
            strength = 0.0
        
        import time
        return IndicatorResult(
            name='RSI',
            value=rsi,
            signal=signal,
            strength=min(1.0, strength),
            timestamp=time.time(),
            metadata={'period': self.period}
        )
    
    def get_rsi(self) -> Optional[float]:
        """Get current RSI value."""
        if self.avg_gain is None or self.avg_loss is None:
            return None
        if self.avg_loss == 0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100 - (100 / (1 + rs))


class MACDCalculator:
    """
    Moving Average Convergence Divergence indicator.
    Standard settings: 12, 26, 9
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
        
        self.prices: Deque[float] = deque(maxlen=slow * 2)
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.macd_line: Optional[float] = None
        self.signal_line: Optional[float] = None
        self.histogram: Optional[float] = None
        
        self.macd_values: Deque[float] = deque(maxlen=signal * 2)
        
    def _ema_update(self, prev_ema: Optional[float], price: float, 
                    period: int) -> float:
        """Calculate EMA update."""
        multiplier = 2 / (period + 1)
        if prev_ema is None:
            return price
        return (price * multiplier) + (prev_ema * (1 - multiplier))
    
    def update(self, price: float) -> Optional[IndicatorResult]:
        """Update MACD and return result if ready."""
        self.prices.append(price)
        
        if len(self.prices) < self.slow_period:
            return None
        
        # Update EMAs
        self.fast_ema = self._ema_update(self.fast_ema, price, self.fast_period)
        self.slow_ema = self._ema_update(self.slow_ema, price, self.slow_period)
        
        # MACD Line
        self.macd_line = self.fast_ema - self.slow_ema
        self.macd_values.append(self.macd_line)
        
        # Signal Line
        if len(self.macd_values) >= self.signal_period:
            macd_array = np.array(list(self.macd_values))
            self.signal_line = np.mean(macd_array[-self.signal_period:])
            self.histogram = self.macd_line - self.signal_line
            
            # Determine signal
            if self.histogram > 0 and self.macd_line > self.signal_line:
                signal = 'BUY'
                strength = min(1.0, abs(self.histogram) / abs(self.macd_line).max(0.0001))
            elif self.histogram < 0 and self.macd_line < self.signal_line:
                signal = 'SELL'
                strength = min(1.0, abs(self.histogram) / abs(self.macd_line).max(0.0001))
            else:
                signal = 'NEUTRAL'
                strength = 0.0
            
            import time
            return IndicatorResult(
                name='MACD',
                value=self.macd_line,
                signal=signal,
                strength=strength,
                timestamp=time.time(),
                metadata={
                    'signal_line': self.signal_line,
                    'histogram': self.histogram,
                    'fast_ema': self.fast_ema,
                    'slow_ema': self.slow_ema
                }
            )
        
        return None


class VWAPCalculator:
    """
    Volume Weighted Average Price calculator.
    Resets at session start (configurable).
    """
    
    def __init__(self, reset_interval: str = 'daily'):
        self.reset_interval = reset_interval
        self.cumulative_volume: float = 0.0
        self.cumulative_pv: float = 0.0  # price * volume
        self.vwap: Optional[float] = None
        self.session_start: Optional[float] = None
        
    def update(self, high: float, low: float, close: float, 
               volume: float, timestamp: float) -> Optional[IndicatorResult]:
        """Update VWAP with new candle data."""
        # Check for session reset
        if self.session_start is None:
            self.session_start = timestamp
        elif self._should_reset(timestamp):
            self.cumulative_volume = 0.0
            self.cumulative_pv = 0.0
            self.session_start = timestamp
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        self.cumulative_volume += volume
        self.cumulative_pv += typical_price * volume
        
        if self.cumulative_volume == 0:
            return None
        
        self.vwap = self.cumulative_pv / self.cumulative_volume
        
        # Determine signal based on price vs VWAP
        if close > self.vwap:
            signal = 'BUY'
            strength = min(1.0, (close - self.vwap) / self.vwap * 100)
        elif close < self.vwap:
            signal = 'SELL'
            strength = min(1.0, (self.vwap - close) / self.vwap * 100)
        else:
            signal = 'NEUTRAL'
            strength = 0.0
        
        import time
        return IndicatorResult(
            name='VWAP',
            value=self.vwap,
            signal=signal,
            strength=strength,
            timestamp=time.time(),
            metadata={
                'cumulative_volume': self.cumulative_volume,
                'typical_price': typical_price
            }
        )
    
    def _should_reset(self, timestamp: float) -> bool:
        """Check if VWAP should be reset based on interval."""
        if self.session_start is None:
            return False
        
        elapsed = timestamp - self.session_start
        
        if self.reset_interval == 'daily':
            return elapsed >= 86400  # 24 hours
        elif self.reset_interval == 'hourly':
            return elapsed >= 3600
        elif self.reset_interval == 'session':
            # Would need exchange session info
            return False
        
        return False


@dataclass
class IchimokuResult:
    """Complete Ichimoku Cloud analysis result."""
    tenkan_sen: float  # Conversion line
    kijun_sen: float   # Base line
    senkou_span_a: float  # Leading span A
    senkou_span_b: float  # Leading span B
    chikou_span: float  # Lagging span
    cloud_color: str  # 'BULLISH' or 'BEARISH'
    price_vs_cloud: str  # 'ABOVE', 'BELOW', 'INSIDE'
    signal: str
    timestamp: float


class IchimokuCalculator:
    """
    Ichimoku Kinko Hyo (Cloud) indicator.
    Standard settings: 9, 26, 52
    """
    
    def __init__(self, conversion: int = 9, base: int = 26, 
                 span_b: int = 52):
        self.conversion_period = conversion
        self.base_period = base
        self.span_b_period = span_b
        
        self.highs: Deque[float] = deque(maxlen=span_b * 2)
        self.lows: Deque[float] = deque(maxlen=span_b * 2)
        self.closes: Deque[float] = deque(maxlen=span_b * 2)
        
        self.tenkan_sen: Optional[float] = None
        self.kijun_sen: Optional[float] = None
        self.senkou_span_a: Optional[float] = None
        self.senkou_span_b: Optional[float] = None
        self.chikou_span: Optional[float] = None
        
    def update(self, high: float, low: float, close: float) -> Optional[IchimokuResult]:
        """Update Ichimoku and return complete result."""
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        if len(self.highs) < self.span_b_period:
            return None
        
        # Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2
        recent_highs = list(self.highs)[-self.conversion_period:]
        recent_lows = list(self.lows)[-self.conversion_period:]
        self.tenkan_sen = (max(recent_highs) + min(recent_lows)) / 2
        
        # Kijun-sen (Base Line): (Highest High + Lowest Low) / 2
        base_highs = list(self.highs)[-self.base_period:]
        base_lows = list(self.lows)[-self.base_period:]
        self.kijun_sen = (max(base_highs) + min(base_lows)) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 ahead
        self.senkou_span_a = (self.tenkan_sen + self.kijun_sen) / 2
        
        # Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2, 52 periods ago
        span_b_highs = list(self.highs)[-self.span_b_period:]
        span_b_lows = list(self.lows)[-self.span_b_period:]
        self.senkou_span_b = (max(span_b_highs) + min(span_b_lows)) / 2
        
        # Chikou Span (Lagging Span): Current close plotted 26 periods back
        self.chikou_span = close
        
        # Determine cloud color
        if self.senkou_span_a > self.senkou_span_b:
            cloud_color = 'BULLISH'
        else:
            cloud_color = 'BEARISH'
        
        # Price position relative to cloud
        if close > max(self.senkou_span_a, self.senkou_span_b):
            price_vs_cloud = 'ABOVE'
        elif close < min(self.senkou_span_a, self.senkou_span_b):
            price_vs_cloud = 'BELOW'
        else:
            price_vs_cloud = 'INSIDE'
        
        # Generate signal
        if cloud_color == 'BULLISH' and price_vs_cloud == 'ABOVE':
            signal = 'BUY'
        elif cloud_color == 'BEARISH' and price_vs_cloud == 'BELOW':
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
        
        import time
        return IchimokuResult(
            tenkan_sen=self.tenkan_sen,
            kijun_sen=self.kijun_sen,
            senkou_span_a=self.senkou_span_a,
            senkou_span_b=self.senkou_span_b,
            chikou_span=self.chikou_span,
            cloud_color=cloud_color,
            price_vs_cloud=price_vs_cloud,
            signal=signal,
            timestamp=time.time()
        )


class FibonacciRetracement:
    """
    Fibonacci retracement level calculator.
    Key levels: 0.236, 0.382, 0.5, 0.618, 0.786
    """
    
    LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    def __init__(self):
        self.swing_high: Optional[float] = None
        self.swing_low: Optional[float] = None
        self.levels: Dict[float, float] = {}
        
    def set_range(self, swing_high: float, swing_low: float):
        """Set the swing range for Fibonacci calculation."""
        self.swing_high = swing_high
        self.swing_low = swing_low
        self._calculate_levels()
    
    def _calculate_levels(self):
        """Calculate all Fibonacci levels."""
        if self.swing_high is None or self.swing_low is None:
            return
        
        diff = self.swing_high - self.swing_low
        
        self.levels = {
            level: self.swing_high - (diff * level)
            for level in self.LEVELS
        }
    
    def get_levels(self) -> Dict[str, float]:
        """Get labeled Fibonacci levels."""
        labels = {
            0.0: '0%',
            0.236: '23.6%',
            0.382: '38.2%',
            0.5: '50%',
            0.618: '61.8%',
            0.786: '78.6%',
            1.0: '100%'
        }
        
        return {
            labels[level]: price
            for level, price in self.levels.items()
        }
    
    def find_nearest_level(self, price: float) -> Tuple[str, float, float]:
        """Find nearest Fibonacci level to current price."""
        if not self.levels:
            return ('Unknown', 0.0, float('inf'))
        
        min_distance = float('inf')
        nearest_label = 'Unknown'
        nearest_price = 0.0
        
        labels = {
            0.0: '0%',
            0.236: '23.6%',
            0.382: '38.2%',
            0.5: '50%',
            0.618: '61.8%',
            0.786: '78.6%',
            1.0: '100%'
        }
        
        for level, fib_price in self.levels.items():
            distance = abs(price - fib_price)
            if distance < min_distance:
                min_distance = distance
                nearest_label = labels.get(level, 'Unknown')
                nearest_price = fib_price
        
        return (nearest_label, nearest_price, min_distance)


class ClassicTAEngine:
    """
    Main engine coordinating all classic technical indicators.
    Singleton pattern for global access.
    """
    
    _instance: Optional['ClassicTAEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.rsi_calculators: Dict[str, RSICalculator] = {}
        self.macd_calculators: Dict[str, MACDCalculator] = {}
        self.vwap_calculators: Dict[str, VWAPCalculator] = {}
        self.ichimoku_calculators: Dict[str, IchimokuCalculator] = {}
        self.fib_calculators: Dict[str, FibonacciRetracement] = {}
        
        self.default_rsi_period = 14
        self.default_macd_settings = (12, 26, 9)
        
        self._initialized = True
        logger.info("ClassicTAEngine initialized")
    
    def get_or_create_indicators(self, symbol: str):
        """Get or create indicator calculators for a symbol."""
        if symbol not in self.rsi_calculators:
            self.rsi_calculators[symbol] = RSICalculator(self.default_rsi_period)
            self.macd_calculators[symbol] = MACDCalculator(*self.default_macd_settings)
            self.vwap_calculators[symbol] = VWAPCalculator('daily')
            self.ichimoku_calculators[symbol] = IchimokuCalculator()
            self.fib_calculators[symbol] = FibonacciRetracement()
    
    def update_candle(self, symbol: str, open_: float, high: float, 
                      low: float, close: float, volume: float,
                      timestamp: float) -> Dict[str, any]:
        """Update all indicators with new candle data."""
        self.get_or_create_indicators(symbol)
        
        results = {
            'symbol': symbol,
            'timestamp': timestamp,
            'indicators': {}
        }
        
        # RSI
        rsi_result = self.rsi_calculators[symbol].update(close)
        if rsi_result:
            results['indicators']['rsi'] = {
                'value': rsi_result.value,
                'signal': rsi_result.signal,
                'strength': rsi_result.strength
            }
        
        # MACD
        macd_result = self.macd_calculators[symbol].update(close)
        if macd_result:
            results['indicators']['macd'] = {
                'value': macd_result.value,
                'signal': macd_result.signal,
                'strength': macd_result.strength,
                'histogram': macd_result.metadata.get('histogram'),
                'signal_line': macd_result.metadata.get('signal_line')
            }
        
        # VWAP
        vwap_result = self.vwap_calculators[symbol].update(
            high, low, close, volume, timestamp
        )
        if vwap_result:
            results['indicators']['vwap'] = {
                'value': vwap_result.value,
                'signal': vwap_result.signal,
                'strength': vwap_result.strength
            }
        
        # Ichimoku
        ichimoku_result = self.ichimoku_calculators[symbol].update(high, low, close)
        if ichimoku_result:
            results['indicators']['ichimoku'] = {
                'tenkan_sen': ichimoku_result.tenkan_sen,
                'kijun_sen': ichimoku_result.kijun_sen,
                'senkou_span_a': ichimoku_result.senkou_span_a,
                'senkou_span_b': ichimoku_result.senkou_span_b,
                'cloud_color': ichimoku_result.cloud_color,
                'price_vs_cloud': ichimoku_result.price_vs_cloud,
                'signal': ichimoku_result.signal
            }
        
        return results
    
    def set_fibonacci_range(self, symbol: str, swing_high: float, swing_low: float):
        """Set Fibonacci retracement range for a symbol."""
        self.get_or_create_indicators(symbol)
        self.fib_calculators[symbol].set_range(swing_high, swing_low)
    
    def get_fibonacci_levels(self, symbol: str) -> Dict[str, float]:
        """Get Fibonacci levels for a symbol."""
        if symbol not in self.fib_calculators:
            return {}
        return self.fib_calculators[symbol].get_levels()
    
    def get_comprehensive_ta(self, symbol: str, price: float) -> Dict[str, any]:
        """Get comprehensive TA summary including nearest Fib levels."""
        result = {'symbol': symbol, 'price': price, 'summary': {}}
        
        # Get Fib levels
        fib_levels = self.get_fibonacci_levels(symbol)
        if fib_levels:
            nearest = self.fib_calculators[symbol].find_nearest_level(price)
            result['fibonacci'] = {
                'levels': fib_levels,
                'nearest': {
                    'level': nearest[0],
                    'price': nearest[1],
                    'distance': nearest[2]
                }
            }
        
        # Get current indicator values
        if symbol in self.rsi_calculators:
            result['rsi'] = self.rsi_calculators[symbol].get_rsi()
        
        return result


if __name__ == "__main__":
    # Example usage
    engine = ClassicTAEngine()
    
    # Simulate candles
    test_candles = [
        (100, 102, 99, 101, 1000),
        (101, 103, 100, 102, 1200),
        (102, 104, 101, 103, 1100),
        (103, 105, 102, 104, 1300),
    ]
    
    for i, (o, h, l, c, v) in enumerate(test_candles):
        result = engine.update_candle("BTCUSDT", o, h, l, c, v, time.time())
        print(f"Candle {i+1}: {result['indicators']}")
