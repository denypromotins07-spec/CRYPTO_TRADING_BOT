#!/usr/bin/env python3
"""
backend/orderflow/flow_fusion.py

Blends footprint data with VWAP and SMC (Smart Money Concepts) order blocks.
Creates a unified view of order flow for execution decisions.

Features:
- Fusion of multiple order flow signals
- Integration with VWAP bands
- SMC order block identification
- Strict type hinting for production reliability
- Cross-platform compatibility optimized for Windows PowerShell
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, auto
import time


class SignalStrength(Enum):
    """Strength of a trading signal."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class OrderBlockType(Enum):
    """Type of SMC order block."""
    BULLISH_OB = auto()    # Bullish order block (support)
    BEARISH_OB = auto()    # Bearish order block (resistance)
    BREAKER_BLOCK = auto()  # Failed OB that flipped
    REJECTION_BLOCK = auto()  # Strong rejection zone


@dataclass
class FootprintSignal:
    """Signal from footprint analysis."""
    price: float
    imbalance_ratio: float
    stacked_imbalance: bool
    unfinished_auction: bool
    confidence: float
    timestamp_ns: int


@dataclass
class VWAPSignal:
    """Signal from VWAP analysis."""
    current_price: float
    vwap: float
    upper_band: float
    lower_band: float
    position: str  # 'above_upper', 'below_lower', 'near_vwap'
    deviation_pct: float


@dataclass
class OrderBlock:
    """SMC Order Block definition."""
    block_type: OrderBlockType
    start_price: float
    end_price: float
    mitigation_level: float  # Price where block should be mitigated
    is_tested: bool = False
    strength: SignalStrength = SignalStrength.MODERATE
    created_at: int = 0


@dataclass
class FusedSignal:
    """Combined signal from all sources."""
    price: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    strength: SignalStrength
    footprint_component: float  # 0-1 contribution
    vwap_component: float  # 0-1 contribution
    ob_component: float  # 0-1 contribution
    confluence_count: int  # Number of agreeing signals
    recommended_action: str  # 'buy', 'sell', 'wait'
    stop_loss_suggestion: float
    take_profit_suggestion: float
    timestamp_ns: int


class VWAPCalculator:
    """Calculates VWAP and standard deviation bands."""
    
    def __init__(self, session_start_ns: int):
        self.session_start_ns = session_start_ns
        self.cumulative_tp_volume: float = 0.0
        self.cumulative_volume: float = 0.0
        self.vwap: float = 0.0
        self.volume_sum_sq: float = 0.0
        self.trade_count: int = 0
        
        # For band calculation
        self.std_dev_multiplier: float = 2.0
        self.price_history: deque = deque(maxlen=1000)
    
    def add_trade(self, price: float, volume: float, timestamp_ns: int) -> None:
        """Add a trade and update VWAP."""
        tp = price  # Typical price (could use HLC average)
        
        self.cumulative_tp_volume += tp * volume
        self.cumulative_volume += volume
        self.vwap = self.cumulative_tp_volume / self.cumulative_volume if self.cumulative_volume > 0 else 0
        
        self.price_history.append((price, volume))
        self.trade_count += 1
    
    def get_vwap(self) -> float:
        """Get current VWAP value."""
        return self.vwap
    
    def get_bands(self) -> Tuple[float, float]:
        """Calculate VWAP bands based on standard deviation."""
        if len(self.price_history) < 10:
            return self.vwap, self.vwap
        
        prices = [p for p, v in self.price_history]
        volumes = [v for p, v in self.price_history]
        
        # Volume-weighted standard deviation
        mean = self.vwap
        variance = sum(v * (p - mean) ** 2 for p, v in self.price_history) / self.cumulative_volume
        std_dev = variance ** 0.5
        
        upper = mean + self.std_dev_multiplier * std_dev
        lower = mean - self.std_dev_multiplier * std_dev
        
        return upper, lower
    
    def reset(self, new_session_start_ns: int) -> None:
        """Reset for new session."""
        self.__init__(new_session_start_ns)


class FlowFusion:
    """
    Main fusion engine combining footprint, VWAP, and SMC order blocks.
    """
    
    def __init__(self):
        self.vwaps: Dict[str, VWAPCalculator] = {}
        self.order_blocks: Dict[str, List[OrderBlock]] = defaultdict(list)
        self.footprint_signals: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.observers: List = []
        
        # Configuration
        self.max_order_blocks = 20
        self.signal_decay_minutes = 5
    
    def set_session_start(self, symbol: str, session_start_ns: int) -> None:
        """Set session start for VWAP calculation."""
        self.vwaps[symbol] = VWAPCalculator(session_start_ns)
    
    def process_trade(
        self,
        symbol: str,
        price: float,
        volume: float,
        is_buyer_maker: bool,
        timestamp_ns: Optional[int] = None
    ) -> None:
        """Process a trade and update VWAP."""
        ts = timestamp_ns or time.time_ns()
        
        if symbol not in self.vwaps:
            self.set_session_start(symbol, ts)
        
        self.vwaps[symbol].add_trade(price, volume, ts)
    
    def add_footprint_signal(
        self,
        symbol: str,
        signal: FootprintSignal
    ) -> None:
        """Add a footprint signal for fusion."""
        self.footprint_signals[symbol].append(signal)
    
    def add_order_block(
        self,
        symbol: str,
        block: OrderBlock
    ) -> None:
        """Add an SMC order block."""
        block.created_at = time.time_ns()
        
        self.order_blocks[symbol].append(block)
        
        # Limit stored blocks
        if len(self.order_blocks[symbol]) > self.max_order_blocks:
            self.order_blocks[symbol].pop(0)
    
    def identify_order_blocks_from_candles(
        self,
        symbol: str,
        candles: List[Tuple[float, float, float, float]],  # OHLC
        threshold_pct: float = 0.02
    ) -> List[OrderBlock]:
        """
        Identify order blocks from candlestick patterns.
        Bullish OB: Last down candle before strong up move
        Bearish OB: Last up candle before strong down move
        """
        blocks = []
        
        for i in range(1, len(candles) - 1):
            prev_open, prev_high, prev_low, prev_close = candles[i - 1]
            curr_open, curr_high, curr_low, curr_close = candles[i]
            next_open, next_high, next_low, next_close = candles[i + 1]
            
            # Check for bullish order block
            if prev_close < prev_open:  # Previous candle was down
                move_up = (next_close - curr_close) / curr_close if curr_close > 0 else 0
                if move_up > threshold_pct:  # Strong move up after
                    block = OrderBlock(
                        block_type=OrderBlockType.BULLISH_OB,
                        start_price=prev_low,
                        end_price=prev_high,
                        mitigation_level=(prev_low + prev_high) / 2
                    )
                    blocks.append(block)
            
            # Check for bearish order block
            if prev_close > prev_open:  # Previous candle was up
                move_down = (curr_close - next_close) / curr_close if curr_close > 0 else 0
                if move_down > threshold_pct:  # Strong move down after
                    block = OrderBlock(
                        block_type=OrderBlockType.BEARISH_OB,
                        start_price=prev_low,
                        end_price=prev_high,
                        mitigation_level=(prev_low + prev_high) / 2
                    )
                    blocks.append(block)
        
        # Add to storage
        for block in blocks:
            self.add_order_block(symbol, block)
        
        return blocks
    
    def fuse_signals(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[FusedSignal]:
        """
        Fuse all signals into a single actionable signal.
        """
        ts = time.time_ns()
        
        # Get VWAP signal
        vwap_signal = self._get_vwap_signal(symbol, current_price)
        if not vwap_signal:
            return None
        
        # Get recent footprint signals
        footprint_signals = list(self.footprint_signals.get(symbol, []))
        footprint_score = self._calculate_footprint_score(footprint_signals, current_price)
        
        # Get relevant order blocks
        relevant_blocks = self._get_relevant_order_blocks(symbol, current_price)
        ob_score = self._calculate_ob_score(relevant_blocks, current_price)
        
        # Calculate components
        vwap_component = self._calculate_vwap_component(vwap_signal)
        
        # Determine direction and strength
        total_score = footprint_score + vwap_component + ob_score
        
        if total_score >= 2.0:
            direction = 'bullish'
            strength = self._score_to_strength(total_score / 3.0)
            action = 'buy'
        elif total_score <= -2.0:
            direction = 'bearish'
            strength = self._score_to_strength(abs(total_score) / 3.0)
            action = 'sell'
        else:
            direction = 'neutral'
            strength = SignalStrength.WEAK
            action = 'wait'
        
        # Count confluences
        confluence_count = sum([
            1 if abs(footprint_score) > 0.3 else 0,
            1 if abs(vwap_component) > 0.3 else 0,
            1 if abs(ob_score) > 0.3 else 0,
        ])
        
        # Calculate suggested levels
        vwap = self.vwaps[symbol].get_vwap() if symbol in self.vwaps else current_price
        stop_loss = current_price * (0.995 if direction == 'bullish' else 1.005)
        take_profit = current_price * (1.02 if direction == 'bullish' else 0.98)
        
        return FusedSignal(
            price=current_price,
            direction=direction,
            strength=strength,
            footprint_component=max(0, footprint_score),
            vwap_component=max(0, vwap_component),
            ob_component=max(0, ob_score),
            confluence_count=confluence_count,
            recommended_action=action,
            stop_loss_suggestion=stop_loss,
            take_profit_suggestion=take_profit,
            timestamp_ns=ts
        )
    
    def _get_vwap_signal(self, symbol: str, price: float) -> Optional[VWAPSignal]:
        """Get VWAP-based signal."""
        if symbol not in self.vwaps:
            return None
        
        calc = self.vwaps[symbol]
        vwap = calc.get_vwap()
        upper, lower = calc.get_bands()
        
        if price > upper:
            position = 'above_upper'
        elif price < lower:
            position = 'below_lower'
        else:
            position = 'near_vwap'
        
        deviation = ((price - vwap) / vwap * 100) if vwap > 0 else 0
        
        return VWAPSignal(
            current_price=price,
            vwap=vwap,
            upper_band=upper,
            lower_band=lower,
            position=position,
            deviation_pct=deviation
        )
    
    def _calculate_footprint_score(
        self,
        signals: List[FootprintSignal],
        current_price: float
    ) -> float:
        """Calculate aggregated footprint score (-1 to 1)."""
        if not signals:
            return 0.0
        
        # Weight recent signals more heavily
        recent_signals = signals[-5:]
        
        total_score = 0.0
        for signal in recent_signals:
            weight = 1.0
            if signal.stacked_imbalance:
                weight *= 1.5
            if signal.unfinished_auction:
                weight *= 1.2
            
            # Direction based on imbalance
            direction = 1.0 if signal.imbalance_ratio > 0 else -1.0
            total_score += direction * signal.confidence * weight
        
        return max(-1, min(1, total_score / len(recent_signals))) if recent_signals else 0.0
    
    def _get_relevant_order_blocks(
        self,
        symbol: str,
        current_price: float,
        range_pct: float = 0.05
    ) -> List[OrderBlock]:
        """Get order blocks within range of current price."""
        if symbol not in self.order_blocks:
            return []
        
        relevant = []
        for block in self.order_blocks[symbol]:
            mid_point = (block.start_price + block.end_price) / 2
            distance_pct = abs(mid_point - current_price) / current_price
            
            if distance_pct <= range_pct:
                relevant.append(block)
        
        return relevant
    
    def _calculate_ob_score(
        self,
        blocks: List[OrderBlock],
        current_price: float
    ) -> float:
        """Calculate order block score (-1 to 1)."""
        if not blocks:
            return 0.0
        
        total_score = 0.0
        for block in blocks:
            if block.block_type == OrderBlockType.BULLISH_OB:
                if current_price > block.end_price:
                    total_score += 0.5  # Price above OB, potential support
            elif block.block_type == OrderBlockType.BEARISH_OB:
                if current_price < block.start_price:
                    total_score -= 0.5  # Price below OB, potential resistance
        
        return max(-1, min(1, total_score))
    
    def _calculate_vwap_component(self, signal: VWAPSignal) -> float:
        """Calculate VWAP component score (-1 to 1)."""
        if signal.position == 'above_upper':
            return -0.5  # Overextended, potential mean reversion
        elif signal.position == 'below_lower':
            return 0.5  # Oversold, potential bounce
        elif signal.deviation_pct > 1.0:
            return -0.3 * min(signal.deviation_pct, 5.0) / 5.0
        elif signal.deviation_pct < -1.0:
            return 0.3 * min(abs(signal.deviation_pct), 5.0) / 5.0
        return 0.0
    
    def _score_to_strength(self, normalized_score: float) -> SignalStrength:
        """Convert normalized score to SignalStrength."""
        if normalized_score >= 0.8:
            return SignalStrength.VERY_STRONG
        elif normalized_score >= 0.6:
            return SignalStrength.STRONG
        elif normalized_score >= 0.4:
            return SignalStrength.MODERATE
        elif normalized_score >= 0.2:
            return SignalStrength.WEAK
        return SignalStrength.VERY_WEAK


if __name__ == "__main__":
    # Example usage
    fusion = FlowFusion()
    
    # Simulate trades
    base_time = time.time_ns()
    base_price = 60000.0
    
    for i in range(50):
        price = base_price + (i % 10 - 5) * 2
        fusion.process_trade("BTCUSDT", price, 1.0, i % 2 == 0, base_time + i * 100_000_000)
    
    # Add some footprint signals
    fusion.add_footprint_signal("BTCUSDT", FootprintSignal(
        price=base_price,
        imbalance_ratio=0.7,
        stacked_imbalance=True,
        unfinished_auction=False,
        confidence=0.8,
        timestamp_ns=base_time
    ))
    
    # Add order blocks
    fusion.add_order_block("BTCUSDT", OrderBlock(
        block_type=OrderBlockType.BULLISH_OB,
        start_price=59800.0,
        end_price=59900.0,
        mitigation_level=59850.0
    ))
    
    # Get fused signal
    signal = fusion.fuse_signals("BTCUSDT", base_price)
    if signal:
        print(f"Fused Signal:")
        print(f"  Direction: {signal.direction}")
        print(f"  Strength: {signal.strength.name}")
        print(f"  Action: {signal.recommended_action}")
        print(f"  Confluences: {signal.confluence_count}")
        print(f"  Stop Loss: ${signal.stop_loss_suggestion:,.2f}")
        print(f"  Take Profit: ${signal.take_profit_suggestion:,.2f}")
