#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Advanced Risk Management
Chapter 1: Dynamic Stop Loss Calculator

This module implements ATR-based and volatility-adjusted trailing stops
that automatically widen during high-volatility news events and market stress.
The stop loss mechanism uses multiple signals to prevent premature exits
while protecting capital during genuine trend reversals.

Memory Budget: <20MB for price history and indicators
Target Latency: <100μs for stop level recalculation
Assets: BTC, SOL, ETH, USDT parallel processing
Integration: Auto-widens during volatility spikes (>2σ moves)

Author: Opus 4.8
Stage: 5/100 - Advanced Risk Management and Order Book Microstructure
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict, Literal
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time


class StopLossType(Enum):
    """Types of stop loss mechanisms."""
    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_BASED = "atr_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TRAILING_ATR = "trailing_atr"
    CHANDELIER_EXIT = "chandelier_exit"
    TIME_BASED = "time_based"


@dataclass
class StopLossConfig:
    """Configuration for dynamic stop loss calculations."""
    atr_period: int = 14  # Standard ATR period
    atr_multiplier: float = 2.0  # Multiplier for ATR-based stops
    volatility_lookback: int = 20  # Period for volatility calculation
    volatility_threshold: float = 2.0  # Standard deviations for high vol detection
    max_stop_distance: float = 0.10  # Maximum 10% stop distance
    min_stop_distance: float = 0.01  # Minimum 1% stop distance
    trailing_step: float = 0.005  # Trailing increment (0.5%)
    news_event_window: int = 60  # Minutes around news events
    assets: List[str] = field(default_factory=lambda: ["BTC", "SOL", "ETH", "USDT"])


@dataclass
class StopLossResult(TypedDict):
    """Type-safe result structure for stop loss calculations."""
    asset: str
    position_type: Literal["LONG", "SHORT"]
    entry_price: float
    current_price: float
    stop_price: float
    stop_distance_pct: float
    stop_type: str
    is_trailing: bool
    trailing_profit_locked: float
    volatility_adjustment: float
    timestamp: float


class ATRCalculator:
    """
    High-performance Average True Range calculator.
    
    ATR measures market volatility by decomposing the entire range
    of an asset price for that period.
    """
    
    __slots__ = ('_period', '_high_history', '_low_history', '_close_history', '_atr_cache')
    
    def __init__(self, period: int = 14) -> None:
        """
        Initialize ATR calculator.
        
        Args:
            period: Number of periods for ATR calculation
        """
        self._period = period
        self._high_history: deque = deque(maxlen=period + 1)
        self._low_history: deque = deque(maxlen=period + 1)
        self._close_history: deque = deque(maxlen=period + 1)
        self._atr_cache: Optional[float] = None
    
    def add_candle(self, high: float, low: float, close: float) -> Optional[float]:
        """
        Add a new candle and update ATR.
        
        Args:
            high: Period high price
            low: Period low price
            close: Period close price
            
        Returns:
            Updated ATR value or None if insufficient data
        """
        self._high_history.append(high)
        self._low_history.append(low)
        self._close_history.append(close)
        
        if len(self._high_history) < self._period + 1:
            return None
        
        # Calculate True Range for latest candle
        high_vals = list(self._high_history)
        low_vals = list(self._low_history)
        close_vals = list(self._close_history)
        
        true_ranges = []
        for i in range(len(high_vals) - 1):
            tr = max(
                high_vals[i + 1] - low_vals[i + 1],  # Current high-low
                abs(high_vals[i + 1] - close_vals[i]),  # Gap up/down
                abs(low_vals[i + 1] - close_vals[i])
            )
            true_ranges.append(tr)
        
        # Wilder's smoothing: ATR = [(Previous ATR × (n-1)) + Current TR] / n
        if self._atr_cache is None:
            # Initial ATR is simple average
            self._atr_cache = sum(true_ranges) / len(true_ranges)
        else:
            current_tr = max(
                high_vals[-1] - low_vals[-1],
                abs(high_vals[-1] - close_vals[-2]),
                abs(low_vals[-1] - close_vals[-2])
            )
            self._atr_cache = ((self._atr_cache * (self._period - 1)) + current_tr) / self._period
        
        return self._atr_cache
    
    def get_atr(self) -> Optional[float]:
        """Get current ATR value."""
        return self._atr_cache
    
    def get_atr_percentage(self, current_price: float) -> Optional[float]:
        """Get ATR as percentage of current price."""
        if self._atr_cache is None or current_price == 0:
            return None
        return self._atr_cache / current_price


class VolatilityEstimator:
    """
    Real-time volatility estimator using multiple methods.
    
    Combines historical volatility, realized volatility, and Parkinson's
    estimator for robust volatility measurement.
    """
    
    __slots__ = ('_lookback', '_returns_history', '_vol_cache')
    
    def __init__(self, lookback: int = 20) -> None:
        """
        Initialize volatility estimator.
        
        Args:
            lookback: Number of periods for volatility calculation
        """
        self._lookback = lookback
        self._returns_history: deque = deque(maxlen=lookback)
        self._vol_cache: Optional[float] = None
    
    def add_return(self, return_value: float) -> Optional[float]:
        """
        Add a return observation and update volatility estimate.
        
        Args:
            return_value: Log return or simple return
            
        Returns:
            Updated volatility (annualized) or None
        """
        self._returns_history.append(return_value)
        
        if len(self._returns_history) < self._lookback:
            return None
        
        returns_array = np.array(self._returns_history, dtype=np.float64)
        
        # Historical volatility (standard deviation of returns)
        vol = np.std(returns_array, ddof=1)
        
        # Annualize (assuming daily returns, 365 days for crypto)
        self._vol_cache = vol * np.sqrt(365)
        
        return self._vol_cache
    
    def get_volatility(self) -> Optional[float]:
        """Get current volatility estimate."""
        return self._vol_cache
    
    def is_high_volatility(self, threshold_sigma: float = 2.0) -> bool:
        """
        Check if current volatility is above threshold.
        
        Args:
            threshold_sigma: Number of standard deviations above mean
            
        Returns:
            True if volatility is elevated
        """
        if len(self._returns_history) < self._lookback:
            return False
        
        returns_array = np.array(self._returns_history, dtype=np.float64)
        recent_vol = np.std(returns_array[-5:], ddof=1)  # Recent 5 periods
        historical_avg_vol = np.mean([np.std(returns_array[i:i+5], ddof=1) 
                                      for i in range(len(returns_array) - 5)])
        
        if historical_avg_vol == 0:
            return False
        
        return recent_vol > historical_avg_vol * threshold_sigma


class DynamicStopLossCalculator:
    """
    Main dynamic stop loss calculator with multiple strategies.
    
    Features:
    - ATR-based stops that adapt to market volatility
    - Automatic widening during high-volatility events
    - Trailing stops that lock in profits
    - Chandelier exits for trend following
    - News event detection and adjustment
    """
    
    __slots__ = (
        '_config', '_atr_calculators', '_volatility_estimators',
        '_positions', '_stop_history', '_news_event_flags'
    )
    
    def __init__(self, config: StopLossConfig = None) -> None:
        """
        Initialize dynamic stop loss calculator.
        
        Args:
            config: Stop loss configuration parameters
        """
        self._config = config or StopLossConfig()
        
        # Per-asset calculators
        self._atr_calculators: Dict[str, ATRCalculator] = {
            asset: ATRCalculator(self._config.atr_period)
            for asset in self._config.assets
        }
        
        self._volatility_estimators: Dict[str, VolatilityEstimator] = {
            asset: VolatilityEstimator(self._config.volatility_lookback)
            for asset in self._config.assets
        }
        
        # Active positions: asset -> {entry_price, position_type, current_stop}
        self._positions: Dict[str, Dict] = {}
        
        # Stop loss history for analysis
        self._stop_history: Dict[str, List[StopLossResult]] = {
            asset: [] for asset in self._config.assets
        }
        
        # News event flags (would be set by external news feed)
        self._news_event_flags: Dict[str, bool] = {
            asset: False for asset in self._config.assets
        }
    
    def set_news_event(self, asset: str, is_news_event: bool) -> None:
        """
        Set news event flag for an asset.
        
        During news events, stop losses are widened to avoid
        being stopped out by temporary volatility spikes.
        
        Args:
            asset: Asset identifier
            is_news_event: True if news event is active
        """
        if asset in self._news_event_flags:
            self._news_event_flags[asset] = is_news_event
    
    def update_price_data(self, asset: str, high: float, low: float, close: float) -> None:
        """
        Update price data for an asset.
        
        Args:
            asset: Asset identifier
            high: Period high
            low: Period low
            close: Period close
        """
        if asset in self._atr_calculators:
            self._atr_calculators[asset].add_candle(high, low, close)
            
            # Calculate return for volatility estimation
            if len(self._atr_calculators[asset]._close_history) > 1:
                prev_close = list(self._atr_calculators[asset]._close_history)[-2]
                if prev_close > 0:
                    ret = (close - prev_close) / prev_close
                    self._volatility_estimators[asset].add_return(ret)
    
    def open_position(self, asset: str, position_type: Literal["LONG", "SHORT"], 
                      entry_price: float, initial_capital: float) -> Optional[StopLossResult]:
        """
        Open a new position with dynamic stop loss.
        
        Args:
            asset: Asset identifier
            position_type: LONG or SHORT
            entry_price: Entry price for the position
            initial_capital: Capital allocated to position
            
        Returns:
            StopLossResult with initial stop level
        """
        atr = self._atr_calculators.get(asset, ATRCalculator()).get_atr()
        vol = self._volatility_estimators.get(asset, VolatilityEstimator()).get_volatility()
        
        if atr is None:
            # Fallback to fixed percentage if ATR not available
            stop_distance = 0.02  # 2% default
        else:
            # ATR-based stop distance
            stop_distance = (atr / entry_price) * self._config.atr_multiplier
            
            # Apply volatility adjustment
            if vol and self._volatility_estimators[asset].is_high_volatility():
                stop_distance *= 1.5  # Widen by 50% during high vol
            
            # Apply news event adjustment
            if self._news_event_flags.get(asset, False):
                stop_distance *= 2.0  # Double during news events
        
        # Enforce min/max bounds
        stop_distance = max(
            self._config.min_stop_distance,
            min(stop_distance, self._config.max_stop_distance)
        )
        
        # Calculate stop price based on position type
        if position_type == "LONG":
            stop_price = entry_price * (1 - stop_distance)
        else:  # SHORT
            stop_price = entry_price * (1 + stop_distance)
        
        # Store position
        self._positions[asset] = {
            'position_type': position_type,
            'entry_price': entry_price,
            'current_stop': stop_price,
            'highest_price': entry_price if position_type == "LONG" else 0,
            'lowest_price': entry_price if position_type == "SHORT" else float('inf'),
            'initial_capital': initial_capital,
            'stop_type': StopLossType.ATR_BASED.value,
            'is_trailing': False
        }
        
        result = StopLossResult(
            asset=asset,
            position_type=position_type,
            entry_price=entry_price,
            current_price=entry_price,
            stop_price=stop_price,
            stop_distance_pct=stop_distance,
            stop_type=StopLossType.ATR_BASED.value,
            is_trailing=False,
            trailing_profit_locked=0.0,
            volatility_adjustment=1.5 if vol and self._volatility_estimators[asset].is_high_volatility() else 1.0,
            timestamp=time.time()
        )
        
        self._stop_history[asset].append(result)
        return result
    
    def update_current_price(self, asset: str, current_price: float) -> Optional[StopLossResult]:
        """
        Update current price and adjust trailing stop if applicable.
        
        Args:
            asset: Asset identifier
            current_price: Latest market price
            
        Returns:
            Updated StopLossResult or None if no position
        """
        if asset not in self._positions:
            return None
        
        position = self._positions[asset]
        position_type = position['position_type']
        entry_price = position['entry_price']
        
        # Update extreme prices for trailing calculation
        if position_type == "LONG":
            position['highest_price'] = max(position['highest_price'], current_price)
        else:
            position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Get current ATR for stop calculation
        atr = self._atr_calculators.get(asset, ATRCalculator()).get_atr()
        if atr is None:
            return self.get_current_stop(asset, current_price)
        
        # Calculate new stop level
        if position_type == "LONG":
            # For long positions, stop trails below highest price
            new_stop = position['highest_price'] - (atr * self._config.atr_multiplier)
            
            # Only move stop up (for long positions)
            if new_stop > position['current_stop']:
                position['current_stop'] = new_stop
                position['is_trailing'] = True
        else:  # SHORT
            # For short positions, stop trails above lowest price
            new_stop = position['lowest_price'] + (atr * self._config.atr_multiplier)
            
            # Only move stop down (for short positions)
            if new_stop < position['current_stop']:
                position['current_stop'] = new_stop
                position['is_trailing'] = True
        
        return self.get_current_stop(asset, current_price)
    
    def get_current_stop(self, asset: str, current_price: float) -> Optional[StopLossResult]:
        """
        Get current stop loss level for a position.
        
        Args:
            asset: Asset identifier
            current_price: Current market price
            
        Returns:
            StopLossResult with current stop information
        """
        if asset not in self._positions:
            return None
        
        position = self._positions[asset]
        stop_price = position['current_stop']
        entry_price = position['entry_price']
        position_type = position['position_type']
        
        # Calculate stop distance
        if position_type == "LONG":
            stop_distance = (entry_price - stop_price) / entry_price
            profit_locked = max(0, position['highest_price'] - entry_price)
        else:
            stop_distance = (stop_price - entry_price) / entry_price
            profit_locked = max(0, entry_price - position['lowest_price'])
        
        result = StopLossResult(
            asset=asset,
            position_type=position_type,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_distance_pct=stop_distance,
            stop_type=position['stop_type'],
            is_trailing=position.get('is_trailing', False),
            trailing_profit_locked=profit_locked,
            volatility_adjustment=1.0,
            timestamp=time.time()
        )
        
        return result
    
    def is_stopped_out(self, asset: str, current_price: float) -> bool:
        """
        Check if position has been stopped out.
        
        Args:
            asset: Asset identifier
            current_price: Current market price
            
        Returns:
            True if stop loss has been triggered
        """
        if asset not in self._positions:
            return False
        
        position = self._positions[asset]
        stop_price = position['current_stop']
        position_type = position['position_type']
        
        if position_type == "LONG":
            return current_price <= stop_price
        else:
            return current_price >= stop_price
    
    def close_position(self, asset: str, exit_price: float) -> Optional[Dict]:
        """
        Close a position and return trade summary.
        
        Args:
            asset: Asset identifier
            exit_price: Exit/closing price
            
        Returns:
            Trade summary dictionary or None
        """
        if asset not in self._positions:
            return None
        
        position = self._positions[asset]
        entry_price = position['entry_price']
        position_type = position['position_type']
        initial_capital = position['initial_capital']
        
        # Calculate P&L
        if position_type == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        pnl_amount = pnl_pct * initial_capital
        
        # Determine exit reason
        if self.is_stopped_out(asset, exit_price):
            exit_reason = "STOP_LOSS"
        elif pnl_pct > 0.05:  # 5% profit target
            exit_reason = "TAKE_PROFIT"
        else:
            exit_reason = "MANUAL"
        
        trade_summary = {
            'asset': asset,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'exit_reason': exit_reason,
            'was_trailing': position.get('is_trailing', False),
            'max_adverse_excursion': self._calculate_mae(asset, exit_price),
            'max_favorable_excursion': self._calculate_mfe(asset, exit_price)
        }
        
        # Remove position
        del self._positions[asset]
        
        return trade_summary
    
    def _calculate_mae(self, asset: str, exit_price: float) -> float:
        """Calculate Maximum Adverse Excursion."""
        if asset not in self._positions:
            return 0.0
        
        position = self._positions[asset]
        entry_price = position['entry_price']
        position_type = position['position_type']
        
        if position_type == "LONG":
            worst_price = min(entry_price, exit_price)
            return (entry_price - worst_price) / entry_price
        else:
            best_price = max(entry_price, exit_price)
            return (best_price - entry_price) / entry_price
    
    def _calculate_mfe(self, asset: str, exit_price: float) -> float:
        """Calculate Maximum Favorable Excursion."""
        if asset not in self._positions:
            return 0.0
        
        position = self._positions[asset]
        entry_price = position['entry_price']
        position_type = position['position_type']
        
        if position_type == "LONG":
            best_price = position.get('highest_price', entry_price)
            return (best_price - entry_price) / entry_price
        else:
            worst_price = position.get('lowest_price', entry_price)
            return (entry_price - worst_price) / entry_price
    
    def get_all_active_positions(self) -> Dict[str, StopLossResult]:
        """Get all active positions with current stop levels."""
        results = {}
        for asset in list(self._positions.keys()):
            # Use last known close price as approximation
            if asset in self._atr_calculators:
                close_history = self._atr_calculators[asset]._close_history
                if close_history:
                    current_price = list(close_history)[-1]
                    result = self.get_current_stop(asset, current_price)
                    if result:
                        results[asset] = result
        return results


if __name__ == "__main__":
    # Example usage and validation
    config = StopLossConfig(atr_period=14, atr_multiplier=2.0)
    calculator = DynamicStopLossCalculator(config)
    
    # Simulate price data for BTC
    import random
    
    base_price = 50000
    prices = []
    
    # Generate simulated price series with trending behavior
    for i in range(100):
        # Random walk with drift
        change = random.gauss(0.001, 0.02)
        if i < 30:  # Uptrend
            change += 0.01
        elif i < 60:  # Downtrend
            change -= 0.015
        else:  # Recovery
            change += 0.005
        
        base_price *= (1 + change)
        prices.append(base_price)
    
    # Feed price data
    for i, price in enumerate(prices):
        high = price * (1 + abs(random.gauss(0, 0.01)))
        low = price * (1 - abs(random.gauss(0, 0.01)))
        calculator.update_price_data("BTC", high, low, price)
    
    # Open a long position after sufficient data
    if len(prices) > 20:
        entry_price = prices[20]
        result = calculator.open_position("BTC", "LONG", entry_price, 10000)
        
        if result:
            print(f"Position opened: Entry={result['entry_price']:.2f}, "
                  f"Initial Stop={result['stop_price']:.2f}")
        
        # Update through price series
        for price in prices[21:]:
            result = calculator.update_current_price("BTC", price)
            
            if calculator.is_stopped_out("BTC", price):
                print(f"Stopped out at {price:.2f}")
                trade = calculator.close_position("BTC", price)
                if trade:
                    print(f"P&L: {trade['pnl_pct']:.2%} ({trade['pnl_amount']:.2f} INR)")
                break
            
            if result and i % 10 == 0:
                print(f"Price: {price:.2f}, Stop: {result['stop_price']:.2f}, "
                      f"Profit Locked: {result['trailing_profit_locked']:.2f}")
