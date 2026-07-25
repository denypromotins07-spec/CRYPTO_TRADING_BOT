#!/usr/bin/env python3
"""
Funding Rate Harvester - Predict and Capture 8-Hour Funding Payouts

This module predicts funding rates and executes positions to capture
funding payments in perpetual swap markets. Accounts for UTC cutoffs
and timezone differences for precise timing.

Chapter 1: Spot-Futures Basis Trading and Cash-and-Carry Arbitrage Logic
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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


class FundingPosition(Enum):
    """Type of funding position"""
    LONG = "long"  # Receive funding if rate > 0
    SHORT = "short"  # Pay funding if rate > 0
    NONE = "none"


@dataclass
class FundingRateTick:
    """Represents a funding rate observation"""
    symbol: str
    funding_rate: Decimal  # Per 8-hour period (e.g., 0.0001 = 0.01%)
    predicted_rate: Decimal
    timestamp: datetime
    next_funding_time: datetime
    time_to_funding_seconds: float
    annualized_rate: Decimal


@dataclass
class FundingTrade:
    """Represents a funding rate capture trade"""
    trade_id: str
    symbol: str
    position_side: FundingPosition
    entry_time: datetime
    entry_price: Decimal
    quantity: Decimal
    expected_funding_payment: Decimal
    actual_funding_payment: Decimal = Decimal('0')
    exit_time: Optional[datetime] = None
    exit_price: Optional[Decimal] = None
    pnl_usdt: Decimal = Decimal('0')
    status: str = "open"


class FundingRatePredictor:
    """
    Predicts funding rates using historical data and market indicators.
    
    Uses weighted moving average of recent funding rates with decay factor
    for more accurate short-term predictions.
    """
    
    def __init__(self, lookback_periods: int = 10, decay_factor: float = 0.9):
        """
        Initialize predictor.
        
        Args:
            lookback_periods: Number of historical periods to consider
            decay_factor: Exponential decay factor (0-1) for weighting
        """
        self.lookback_periods = lookback_periods
        self.decay_factor = decay_factor
        self._rate_history: Dict[str, List[Decimal]] = {}
    
    def add_rate_observation(self, symbol: str, rate: Decimal) -> None:
        """Add a new rate observation to history"""
        if symbol not in self._rate_history:
            self._rate_history[symbol] = []
        
        self._rate_history[symbol].append(rate)
        
        # Trim history to lookback period
        if len(self._rate_history[symbol]) > self.lookback_periods * 2:
            self._rate_history[symbol] = self._rate_history[symbol][-self.lookback_periods * 2:]
    
    def predict_rate(self, symbol: str) -> Optional[Decimal]:
        """
        Predict next funding rate using exponential weighted moving average.
        
        Returns:
            Predicted funding rate or None if insufficient data
        """
        if symbol not in self._rate_history or len(self._rate_history[symbol]) < 3:
            return None
        
        rates = self._rate_history[symbol][-self.lookback_periods:]
        
        # Calculate EWMA
        weights = []
        total_weight = Decimal('0')
        
        for i in range(len(rates)):
            weight = Decimal(str(self.decay_factor ** (len(rates) - i - 1)))
            weights.append(weight)
            total_weight += weight
        
        weighted_sum = sum(r * w for r, w in zip(rates, weights))
        prediction = weighted_sum / total_weight
        
        return prediction
    
    def get_annualized_rate(self, funding_rate: Decimal) -> Decimal:
        """Convert 8-hour funding rate to annualized rate"""
        # 365 days / (8/24 days per period) = 1095 periods per year
        periods_per_year = Decimal('1095')
        annualized = ((Decimal('1') + funding_rate) ** periods_per_year) - Decimal('1')
        return annualized


class FundingRateHarvester:
    """
    Executes funding rate arbitrage strategies.
    
    Opens positions before funding snapshots and closes after capturing payment.
    Optimized for minimal capital usage and maximum funding capture efficiency.
    """
    
    # Supported symbols
    SUPPORTED_SYMBOLS = frozenset(['BTC', 'ETH', 'SOL'])
    
    # Minimum predicted funding rate to trigger trade (in bps)
    MIN_FUNDING_BPS = Decimal('5.0')
    
    # Maximum position size per symbol (USDT)
    MAX_POSITION_SIZE_USDT = Decimal('100000')
    
    # Time buffer before funding snapshot to enter (seconds)
    ENTRY_BUFFER_SECONDS = 120  # 2 minutes
    
    # Time after funding snapshot to exit (seconds)
    EXIT_BUFFER_SECONDS = 60  # 1 minute
    
    def __init__(self, exchange_client: Any, risk_manager: Any):
        """
        Initialize funding rate harvester.
        
        Args:
            exchange_client: Async exchange client
            risk_manager: Risk management module
        """
        self.exchange_client = exchange_client
        self.risk_manager = risk_manager
        self.predictor = FundingRatePredictor()
        
        # Active funding trades
        self.active_trades: Dict[str, FundingTrade] = {}
        self.trade_counter = 0
        
        # Historical funding rates by symbol
        self.funding_history: Dict[str, List[FundingRateTick]] = {}
        
        # Performance metrics
        self.total_trades = 0
        self.successful_captures = 0
        self.total_funding_captured = Decimal('0')
        self.total_pnl = Decimal('0')
        
        logger.info("FundingRateHarvester initialized")
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_counter += 1
        timestamp_ns = time.time_ns()
        return f"FR-{timestamp_ns}-{self.trade_counter:06d}"
    
    def update_funding_rate(
        self,
        symbol: str,
        current_rate: Decimal,
        next_funding_time: datetime
    ) -> FundingRateTick:
        """
        Update funding rate for a symbol.
        
        Args:
            symbol: Trading symbol
            current_rate: Current funding rate (8-hour period)
            next_funding_time: Next funding snapshot time
            
        Returns:
            FundingRateTick with calculated metrics
        """
        # Add to predictor history
        self.predictor.add_rate_observation(symbol, current_rate)
        
        # Get predicted rate
        predicted_rate = self.predictor.predict_rate(symbol) or current_rate
        
        # Calculate time to funding
        now = datetime.now(timezone.utc)
        time_to_funding = (next_funding_time - now).total_seconds()
        
        # Annualized rate
        annualized = self.predictor.get_annualized_rate(current_rate)
        
        tick = FundingRateTick(
            symbol=symbol,
            funding_rate=current_rate,
            predicted_rate=predicted_rate,
            timestamp=now,
            next_funding_time=next_funding_time,
            time_to_funding_seconds=time_to_funding,
            annualized_rate=annualized
        )
        
        # Store in history
        if symbol not in self.funding_history:
            self.funding_history[symbol] = []
        self.funding_history[symbol].append(tick)
        
        # Trim history
        if len(self.funding_history[symbol]) > 100:
            self.funding_history[symbol] = self.funding_history[symbol][-100:]
        
        return tick
    
    async def check_and_execute_funding_arb(
        self,
        symbol: str,
        current_price: Decimal,
        available_capital: Decimal
    ) -> Optional[FundingTrade]:
        """
        Check if funding arbitrage opportunity exists and execute.
        
        Args:
            symbol: Trading symbol
            current_price: Current perpetual price
            available_capital: Available capital for this trade
            
        Returns:
            FundingTrade if executed, None otherwise
        """
        if symbol not in self.SUPPORTED_SYMBOLS:
            return None
        
        if symbol not in self.funding_history:
            return None
        
        latest_tick = self.funding_history[symbol][-1]
        
        # Check if predicted rate exceeds threshold
        predicted_bps = abs(latest_tick.predicted_rate) * Decimal('10000')
        
        if predicted_bps < self.MIN_FUNDING_BPS:
            logger.debug(f"Funding rate {predicted_bps} bps below threshold")
            return None
        
        # Check time to funding - only enter if within entry window
        if latest_tick.time_to_funding_seconds > self.ENTRY_BUFFER_SECONDS:
            logger.debug(f"Too early to enter: {latest_tick.time_to_funding_seconds}s to funding")
            return None
        
        # Determine position direction
        # If predicted rate > 0, go short to receive funding
        # If predicted rate < 0, go long to receive funding
        if latest_tick.predicted_rate > 0:
            position_side = FundingPosition.SHORT
        else:
            position_side = FundingPosition.LONG
        
        # Check risk limits
        position_size = min(available_capital, self.MAX_POSITION_SIZE_USDT)
        
        if not await self.risk_manager.check_position_limit(symbol, position_size):
            logger.warning(f"Position limit exceeded for {symbol}")
            return None
        
        # Calculate quantity
        quantity = position_size / current_price
        
        # Expected funding payment
        expected_payment = quantity * current_price * abs(latest_tick.predicted_rate)
        
        # Execute trade
        trade = await self._execute_funding_trade(
            symbol=symbol,
            position_side=position_side,
            quantity=quantity,
            price=current_price,
            expected_payment=expected_payment
        )
        
        if trade:
            self.active_trades[trade.trade_id] = trade
            self.total_trades += 1
            logger.info(
                f"Funding arb executed: {trade.trade_id}, symbol={symbol}, "
                f"side={position_side.value}, expected={expected_payment:.2f} USDT"
            )
        
        return trade
    
    async def _execute_funding_trade(
        self,
        symbol: str,
        position_side: FundingPosition,
        quantity: Decimal,
        price: Decimal,
        expected_payment: Decimal
    ) -> Optional[FundingTrade]:
        """Execute a funding rate trade"""
        try:
            trade_id = self._generate_trade_id()
            
            # Place order
            side = "sell" if position_side == FundingPosition.SHORT else "buy"
            
            order_response = await self.exchange_client.place_order(
                market_type="perp",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type="limit"
            )
            
            if not order_response:
                return None
            
            # Wait for fill
            fill_response = await asyncio.wait_for(
                self.exchange_client.wait_for_fill(
                    order_response['order_id'],
                    timeout_ms=1000
                ),
                timeout=1.0
            )
            
            if fill_response.get('status') != 'filled':
                return None
            
            fill_price = Decimal(str(fill_response.get('fill_price')))
            fill_quantity = Decimal(str(fill_response.get('fill_quantity')))
            
            trade = FundingTrade(
                trade_id=trade_id,
                symbol=symbol,
                position_side=position_side,
                entry_time=datetime.now(timezone.utc),
                entry_price=fill_price,
                quantity=fill_quantity,
                expected_funding_payment=expected_payment
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Funding trade execution failed: {e}")
            return None
    
    async def process_funding_snapshot(self, symbol: str) -> bool:
        """
        Process funding snapshot and close positions to capture payment.
        
        Should be called immediately after funding timestamp.
        
        Args:
            symbol: Symbol that just had funding snapshot
            
        Returns:
            True if positions were successfully closed
        """
        closed_count = 0
        
        for trade_id, trade in list(self.active_trades.items()):
            if trade.symbol != symbol:
                continue
            
            if trade.status != "open":
                continue
            
            # Wait for funding payment to be credited
            await asyncio.sleep(5)  # Brief wait for exchange processing
            
            # Get actual funding payment from exchange
            funding_payment = await self.exchange_client.get_funding_payment(
                symbol=trade.symbol,
                trade_id=trade.trade_id
            )
            
            if funding_payment:
                trade.actual_funding_payment = Decimal(str(funding_payment))
                self.total_funding_captured += trade.actual_funding_payment
                self.successful_captures += 1
            
            # Close position
            success = await self._close_funding_position(trade)
            
            if success:
                closed_count += 1
                trade.status = "closed"
                del self.active_trades[trade_id]
        
        if closed_count > 0:
            logger.info(f"Processed {closed_count} funding captures for {symbol}")
        
        return closed_count > 0
    
    async def _close_funding_position(self, trade: FundingTrade) -> bool:
        """Close a funding position"""
        try:
            # Opposite side to close
            close_side = "buy" if trade.position_side == FundingPosition.SHORT else "sell"
            
            await self.exchange_client.place_order(
                market_type="perp",
                symbol=trade.symbol,
                side=close_side,
                quantity=trade.quantity,
                order_type="market"  # Market order for quick exit
            )
            
            trade.exit_time = datetime.now(timezone.utc)
            
            # Calculate PnL (price change + funding)
            # Simplified: just use funding payment as primary profit source
            trade.pnl_usdt = trade.actual_funding_payment
            
            self.total_pnl += trade.pnl_usdt
            
            logger.info(
                f"Funding position closed: {trade.trade_id}, "
                f"funding_captured={trade.actual_funding_payment:.2f} USDT"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to close funding position: {e}")
            return False
    
    def get_next_funding_times(self) -> Dict[str, datetime]:
        """
        Get next funding times for all supported symbols.
        
        Funding typically occurs at 00:00, 08:00, 16:00 UTC.
        
        Returns:
            Dict mapping symbol to next funding time
        """
        now = datetime.now(timezone.utc)
        
        # Standard funding intervals (every 8 hours)
        funding_hours = [0, 8, 16]
        
        next_times = {}
        
        for symbol in self.SUPPORTED_SYMBOLS:
            # Find next funding hour
            current_hour = now.hour
            
            next_funding_hour = None
            for h in funding_hours:
                if h > current_hour:
                    next_funding_hour = h
                    break
            
            if next_funding_hour is None:
                # Wrap to next day
                next_funding_hour = funding_hours[0]
                next_date = now.date() + timedelta(days=1)
            else:
                next_date = now.date()
            
            next_time = datetime(
                year=next_date.year,
                month=next_date.month,
                day=next_date.day,
                hour=next_funding_hour,
                minute=0,
                second=0,
                tzinfo=timezone.utc
            )
            
            next_times[symbol] = next_time
        
        return next_times
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_trades': self.total_trades,
            'successful_captures': self.successful_captures,
            'capture_rate': self.successful_captures / max(1, self.total_trades),
            'total_funding_captured_usdt': float(self.total_funding_captured),
            'total_pnl_usdt': float(self.total_pnl),
            'active_trades_count': len(self.active_trades)
        }


# Example usage
if __name__ == "__main__":
    class MockExchangeClient:
        async def place_order(self, **kwargs):
            await asyncio.sleep(0.01)
            return {'order_id': 'mock_123'}
        
        async def wait_for_fill(self, order_id: str, timeout_ms: int):
            await asyncio.sleep(0.05)
            return {'status': 'filled', 'fill_price': '50000', 'fill_quantity': '0.002'}
        
        async def get_funding_payment(self, symbol: str, trade_id: str) -> Decimal:
            return Decimal('0.50')  # Mock funding payment
    
    class MockRiskManager:
        async def check_position_limit(self, symbol: str, quantity: Decimal) -> bool:
            return True
    
    async def test_harvester():
        harvester = FundingRateHarvester(MockExchangeClient(), MockRiskManager())
        
        # Simulate funding rate update
        now = datetime.now(timezone.utc)
        next_funding = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        tick = harvester.update_funding_rate(
            symbol='BTC',
            current_rate=Decimal('0.0001'),  # 0.01% per 8h
            next_funding_time=next_funding
        )
        
        print(f"Funding tick: {tick}")
        print(f"Annualized rate: {tick.annualized_rate:.2%}")
        
        # Check for arb opportunity
        trade = await harvester.check_and_execute_funding_arb(
            symbol='BTC',
            current_price=Decimal('50000'),
            available_capital=Decimal('10000')
        )
        
        if trade:
            print(f"Executed funding trade: {trade.trade_id}")
        
        print(f"Metrics: {harvester.get_metrics()}")
    
    asyncio.run(test_harvester())
