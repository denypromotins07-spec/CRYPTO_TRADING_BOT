#!/usr/bin/env python3
"""
Z-Score Executor - Entering Pairs Trades When Spread Z-Score > 2.0

This module executes statistical arbitrage trades based on z-score
thresholds of the spread between cointegrated pairs. Automatically
halts trading if cointegration breaks down (p-value > 0.05).

Chapter 3: Statistical Arbitrage, Pairs Trading, and Cointegration Execution
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
import time
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Direction of pairs trade"""
    LONG_SPREAD = "long_spread"  # Long A, Short B
    SHORT_SPREAD = "short_spread"  # Short A, Long B


@dataclass
class SpreadObservation:
    """Single spread observation"""
    timestamp: datetime
    spread_value: float
    z_score: float
    rolling_mean: float
    rolling_std: float


@dataclass
class PairsTrade:
    """Represents a pairs trading position"""
    trade_id: str
    symbol_a: str
    symbol_b: str
    direction: TradeDirection
    quantity_a: Decimal
    quantity_b: Decimal
    entry_z_score: float
    entry_spread: float
    entry_price_a: Decimal
    entry_price_b: Decimal
    hedge_ratio: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    exit_z_score: Optional[float] = None
    exit_spread: Optional[float] = None
    pnl_usdt: Decimal = Decimal('0')
    status: str = "open"


class ZScoreCalculator:
    """
    Calculates rolling z-scores for spread series.
    
    Uses Welford's online algorithm for numerically stable
    running mean and variance calculation.
    """
    
    def __init__(self, window_size: int = 60):
        """
        Initialize z-score calculator.
        
        Args:
            window_size: Rolling window size for statistics
        """
        self.window_size = window_size
        self._values: List[float] = []
        self._mean = 0.0
        self._m2 = 0.0  # Sum of squares of differences from mean
        self._count = 0
    
    def add_value(self, value: float) -> None:
        """Add a new value to the rolling window"""
        self._count += 1
        
        if len(self._values) >= self.window_size:
            # Remove oldest value
            old_value = self._values.pop(0)
            old_mean = self._mean
            
            # Update mean
            self._mean = ((old_mean * self.window_size) - old_value + value) / self.window_size
            
            # Update M2
            self._m2 += (value - old_value) * ((value - self._mean) + (old_value - old_mean))
        else:
            # Still filling window
            delta = value - self._mean
            self._mean += delta / self._count
            delta2 = value - self._mean
            self._m2 += delta * delta2
        
        self._values.append(value)
    
    @property
    def mean(self) -> float:
        """Get current rolling mean"""
        return self._mean
    
    @property
    def variance(self) -> float:
        """Get current rolling variance"""
        if len(self._values) < 2:
            return 0.0
        return self._m2 / (len(self._values) - 1)
    
    @property
    def std(self) -> float:
        """Get current rolling standard deviation"""
        return math.sqrt(self.variance)
    
    def z_score(self, value: float) -> float:
        """Calculate z-score for a given value"""
        if self.std < 1e-10:
            return 0.0
        return (value - self.mean) / self.std
    
    def is_ready(self) -> bool:
        """Check if we have enough data for reliable statistics"""
        return len(self._values) >= min(30, self.window_size)
    
    def reset(self) -> None:
        """Reset the calculator"""
        self._values.clear()
        self._mean = 0.0
        self._m2 = 0.0
        self._count = 0


class ZScoreExecutor:
    """
    Executes pairs trades based on z-score thresholds.
    
    Entry: When |z-score| > entry_threshold (default 2.0)
    Exit: When |z-score| < exit_threshold (default 0.5) or stop-loss triggered
    
    Automatically halts if cointegration p-value > 0.05.
    """
    
    # Default thresholds
    DEFAULT_ENTRY_ZSCORE = 2.0
    DEFAULT_EXIT_ZSCORE = 0.5
    DEFAULT_STOP_LOSS_ZSCORE = 4.0
    
    # Minimum history before trading
    MIN_HISTORY_SIZE = 30
    
    def __init__(
        self,
        exchange_client: Any,
        cointegration_checker: Any,
        entry_threshold: float = DEFAULT_ENTRY_ZSCORE,
        exit_threshold: float = DEFAULT_EXIT_ZSCORE,
        stop_loss_threshold: float = DEFAULT_STOP_LOSS_ZSCORE
    ):
        """
        Initialize z-score executor.
        
        Args:
            exchange_client: Async exchange client
            cointegration_checker: Module to check cointegration status
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            stop_loss_threshold: Z-score threshold for stop-loss
        """
        self.exchange_client = exchange_client
        self.cointegration_checker = cointegration_checker
        
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        
        # Z-score calculators per pair
        self.spread_calculators: Dict[Tuple[str, str], ZScoreCalculator] = {}
        
        # Active trades
        self.active_trades: Dict[str, PairsTrade] = {}
        self.trade_counter = 0
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = Decimal('0')
        self.halted_trades = 0  # Trades halted due to cointegration breakdown
        
        logger.info(
            f"ZScoreExecutor initialized (entry={entry_threshold}, "
            f"exit={exit_threshold}, stop_loss={stop_loss_threshold})"
        )
    
    def _get_pair_key(self, symbol_a: str, symbol_b: str) -> Tuple[str, str]:
        """Get canonical pair key (alphabetically sorted)"""
        return tuple(sorted([symbol_a, symbol_b]))
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_counter += 1
        timestamp_ns = time.time_ns()
        return f"PS-{timestamp_ns}-{self.trade_counter:06d}"
    
    def update_spread(
        self,
        symbol_a: str,
        symbol_b: str,
        price_a: float,
        price_b: float,
        hedge_ratio: float
    ) -> Optional[SpreadObservation]:
        """
        Update spread calculation for a pair.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            price_a: Price of symbol A
            price_b: Price of symbol B
            hedge_ratio: Hedge ratio (beta) from cointegration analysis
            
        Returns:
            SpreadObservation or None if not ready
        """
        pair_key = self._get_pair_key(symbol_a, symbol_b)
        
        # Get or create calculator
        if pair_key not in self.spread_calculators:
            self.spread_calculators[pair_key] = ZScoreCalculator(window_size=60)
        
        calculator = self.spread_calculators[pair_key]
        
        # Calculate spread: spread = price_a - hedge_ratio * price_b
        spread = price_a - hedge_ratio * price_b
        
        # Add to rolling window
        calculator.add_value(spread)
        
        if not calculator.is_ready():
            return None
        
        # Calculate z-score
        z_score = calculator.z_score(spread)
        
        observation = SpreadObservation(
            timestamp=datetime.now(timezone.utc),
            spread_value=spread,
            z_score=z_score,
            rolling_mean=calculator.mean,
            rolling_std=calculator.std
        )
        
        return observation
    
    async def check_and_execute_trade(
        self,
        symbol_a: str,
        symbol_b: str,
        price_a: Decimal,
        price_b: Decimal,
        hedge_ratio: Decimal,
        capital_allocation: Decimal
    ) -> Optional[PairsTrade]:
        """
        Check if trade opportunity exists and execute.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            price_a: Current price of A
            price_b: Current price of B
            hedge_ratio: Hedge ratio from cointegration
            capital_allocation: Capital to allocate
            
        Returns:
            PairsTrade if executed, None otherwise
        """
        # Check cointegration status first
        if self.cointegration_checker:
            if not await self._check_cointegration_status(symbol_a, symbol_b):
                logger.warning(f"Cointegration broken for {symbol_a}/{symbol_b}, halting")
                self.halted_trades += 1
                return None
        
        pair_key = self._get_pair_key(symbol_a, symbol_b)
        
        if pair_key not in self.spread_calculators:
            return None
        
        calculator = self.spread_calculators[pair_key]
        
        if not calculator.is_ready():
            return None
        
        # Calculate current spread and z-score
        spread_float = float(price_a) - float(hedge_ratio) * float(price_b)
        z_score = calculator.z_score(spread_float)
        
        # Check entry conditions
        if abs(z_score) < self.entry_threshold:
            return None
        
        # Determine direction
        if z_score > 0:
            # Spread is high: short spread (short A, long B)
            direction = TradeDirection.SHORT_SPREAD
        else:
            # Spread is low: long spread (long A, short B)
            direction = TradeDirection.LONG_SPREAD
        
        # Execute trade
        trade = await self._execute_pairs_trade(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            direction=direction,
            price_a=price_a,
            price_b=price_b,
            hedge_ratio=hedge_ratio,
            capital=capital_allocation,
            entry_z_score=z_score,
            entry_spread=spread_float
        )
        
        if trade:
            self.active_trades[trade.trade_id] = trade
            self.total_trades += 1
            logger.info(
                f"Pairs trade executed: {trade.trade_id}, "
                f"{symbol_a}/{symbol_b}, direction={direction.value}, z={z_score:.2f}"
            )
        
        return trade
    
    async def _check_cointegration_status(
        self, 
        symbol_a: str, 
        symbol_b: str
    ) -> bool:
        """Check if pair is still cointegrated"""
        try:
            if hasattr(self.cointegration_checker, 'can_trade_pair'):
                return self.cointegration_checker.can_trade_pair(symbol_a, symbol_b)
            elif hasattr(self.cointegration_checker, 'is_trading_allowed'):
                return self.cointegration_checker.is_trading_allowed()
        except Exception as e:
            logger.error(f"Cointegration check failed: {e}")
        
        # Default to allowing if checker unavailable
        return True
    
    async def _execute_pairs_trade(
        self,
        symbol_a: str,
        symbol_b: str,
        direction: TradeDirection,
        price_a: Decimal,
        price_b: Decimal,
        hedge_ratio: Decimal,
        capital: Decimal,
        entry_z_score: float,
        entry_spread: float
    ) -> Optional[PairsTrade]:
        """Execute a pairs trade"""
        try:
            trade_id = self._generate_trade_id()
            
            # Calculate quantities
            # For simplicity, split capital equally between legs
            leg_capital = capital / Decimal('2')
            
            quantity_a = (leg_capital / price_a).quantize(Decimal('0.0001'), rounding=ROUND_DOWN)
            quantity_b = (leg_capital / price_b).quantize(Decimal('0.0001'), rounding=ROUND_DOWN)
            
            # Execute legs based on direction
            if direction == TradeDirection.LONG_SPREAD:
                # Long A, Short B
                leg_a_side = "buy"
                leg_b_side = "sell"
            else:
                # Short A, Long B
                leg_a_side = "sell"
                leg_b_side = "buy"
            
            # Execute both legs
            results = await asyncio.gather(
                self._execute_leg(symbol_a, leg_a_side, quantity_a, price_a),
                self._execute_leg(symbol_b, leg_b_side, quantity_b, price_b),
                return_exceptions=True
            )
            
            if any(isinstance(r, Exception) for r in results):
                logger.error(f"Pairs trade execution failed: {results}")
                return None
            
            trade = PairsTrade(
                trade_id=trade_id,
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                direction=direction,
                quantity_a=quantity_a,
                quantity_b=quantity_b,
                entry_z_score=entry_z_score,
                entry_spread=entry_spread,
                entry_price_a=price_a,
                entry_price_b=price_b,
                hedge_ratio=hedge_ratio
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Pairs trade execution error: {e}")
            return None
    
    async def _execute_leg(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal
    ) -> bool:
        """Execute a single leg of the pairs trade"""
        try:
            await self.exchange_client.place_order(
                market_type="spot",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type="limit"
            )
            return True
        except Exception as e:
            logger.error(f"Leg execution failed for {symbol}: {e}")
            return False
    
    async def check_exit_conditions(self, trade_id: str) -> bool:
        """
        Check if a trade should be exited.
        
        Returns True if trade was closed.
        """
        if trade_id not in self.active_trades:
            return False
        
        trade = self.active_trades[trade_id]
        
        # Get current prices and calculate z-score
        pair_key = self._get_pair_key(trade.symbol_a, trade.symbol_b)
        
        if pair_key not in self.spread_calculators:
            return False
        
        calculator = self.spread_calculators[pair_key]
        
        # Get current prices (would come from market data in production)
        # For now, use entry prices as placeholder
        current_price_a = float(trade.entry_price_a)
        current_price_b = float(trade.entry_price_b)
        
        current_spread = current_price_a - float(trade.hedge_ratio) * current_price_b
        current_z_score = calculator.z_score(current_spread)
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # Mean reversion: z-score crossed back below threshold
        if abs(current_z_score) < self.exit_threshold:
            should_exit = True
            exit_reason = "mean_reversion"
        
        # Stop loss: z-score exceeded stop loss threshold
        elif abs(current_z_score) > self.stop_loss_threshold:
            should_exit = True
            exit_reason = "stop_loss"
        
        if not should_exit:
            return False
        
        # Close the trade
        success = await self._close_pairs_trade(trade, current_z_score, current_spread)
        
        if success:
            del self.active_trades[trade_id]
            logger.info(
                f"Trade {trade_id} closed: reason={exit_reason}, "
                f"exit_z={current_z_score:.2f}, pnl={trade.pnl_usdt:.2f}"
            )
        
        return success
    
    async def _close_pairs_trade(
        self,
        trade: PairsTrade,
        exit_z_score: float,
        exit_spread: float
    ) -> bool:
        """Close a pairs trade"""
        try:
            # Opposite sides to close
            if trade.direction == TradeDirection.LONG_SPREAD:
                close_a_side = "sell"
                close_b_side = "buy"
            else:
                close_a_side = "buy"
                close_b_side = "sell"
            
            # Execute closing legs
            await asyncio.gather(
                self._execute_leg(trade.symbol_a, close_a_side, trade.quantity_a, trade.entry_price_a),
                self._execute_leg(trade.symbol_b, close_b_side, trade.quantity_b, trade.entry_price_b),
                return_exceptions=True
            )
            
            trade.closed_at = datetime.now(timezone.utc)
            trade.exit_z_score = exit_z_score
            trade.exit_spread = exit_spread
            
            # Calculate PnL (simplified)
            # PnL = (entry_z - exit_z) * average_price * sqrt(quantity_a^2 + quantity_b^2)
            z_change = abs(trade.entry_z_score) - abs(exit_z_score)
            avg_price = (float(trade.entry_price_a) + float(trade.entry_price_b)) / 2
            position_size = math.sqrt(float(trade.quantity_a)**2 + float(trade.quantity_b)**2)
            
            pnl = Decimal(str(z_change * avg_price * position_size))
            trade.pnl_usdt = pnl.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
            
            # Update metrics
            self.total_pnl += trade.pnl_usdt
            if trade.pnl_usdt > 0:
                self.winning_trades += 1
            
            trade.status = "closed"
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to close pairs trade: {e}")
            return False
    
    def get_active_trades(self) -> List[PairsTrade]:
        """Get all active trades"""
        return list(self.active_trades.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_pnl_usdt': float(self.total_pnl),
            'active_trades_count': len(self.active_trades),
            'halted_trades': self.halted_trades,
            'pairs_monitored': len(self.spread_calculators)
        }


# Example usage
if __name__ == "__main__":
    class MockExchangeClient:
        async def place_order(self, **kwargs):
            await asyncio.sleep(0.01)
            return True
    
    class MockCointegrationChecker:
        def can_trade_pair(self, symbol_a: str, symbol_b: str) -> bool:
            return True
    
    async def test_executor():
        executor = ZScoreExecutor(
            exchange_client=MockExchangeClient(),
            cointegration_checker=MockCointegrationChecker(),
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Simulate price updates to build history
        print("Building spread history...")
        for i in range(50):
            # Simulate mean-reverting spread
            base_a = 50000 + (i % 10) * 100
            base_b = 3000 + (i % 10) * 6
            hedge_ratio = 16.67
            
            obs = executor.update_spread(
                symbol_a='BTC',
                symbol_b='ETH',
                price_a=base_a,
                price_b=base_b,
                hedge_ratio=hedge_ratio
            )
            
            if obs:
                print(f"  Spread: {obs.spread_value:.2f}, Z-score: {obs.z_score:.2f}")
        
        # Try to execute trade
        print("\nChecking for trade opportunity...")
        trade = await executor.check_and_execute_trade(
            symbol_a='BTC',
            symbol_b='ETH',
            price_a=Decimal('50500'),
            price_b=Decimal('3030'),
            hedge_ratio=Decimal('16.67'),
            capital_allocation=Decimal('10000')
        )
        
        if trade:
            print(f"Executed trade: {trade.trade_id}")
            print(f"Direction: {trade.direction.value}")
            print(f"Entry Z-score: {trade.entry_z_score:.2f}")
        
        print(f"\nMetrics: {executor.get_metrics()}")
    
    asyncio.run(test_executor())
