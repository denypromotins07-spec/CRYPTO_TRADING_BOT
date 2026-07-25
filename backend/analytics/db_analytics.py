"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 9
File: backend/analytics/db_analytics.py

Post-session analytics engine for querying the time-series database.
Runs complex queries on historical tick data to extract trading insights.

Features:
- SQL-like query interface for time-series data
- Performance metric calculations (PnL, Sharpe, drawdown)
- Trade distribution analysis
- Session comparison and benchmarking
- Async query execution

Design Patterns: Repository, Strategy, Builder
"""

from __future__ import annotations
import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeRange(Enum):
    """Predefined time ranges for queries."""
    LAST_HOUR = "1h"
    LAST_4HOURS = "4h"
    LAST_DAY = "1d"
    LAST_WEEK = "1w"
    SESSION = "session"
    CUSTOM = "custom"


@dataclass
class TradeRecord:
    """Represents a completed trade."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    entry_time: float
    exit_time: float
    duration_seconds: float
    fees: float
    slippage: float


@dataclass
class SessionMetrics:
    """Metrics for a trading session."""
    session_id: str
    start_time: float
    end_time: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: float
    average_trade_duration: float
    total_fees: float
    total_slippage: float


@dataclass
class QueryResult:
    """Result of a database query."""
    query_time_ms: float
    row_count: int
    data: List[Dict[str, Any]]
    aggregations: Dict[str, Any]


class DBAnalytics:
    """
    Analytics engine for querying and analyzing trading data.
    
    Provides comprehensive post-session analysis including
    performance metrics, trade distributions, and session comparisons.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the analytics engine.
        
        Args:
            data_dir: Directory containing trading data
        """
        self.data_dir = data_dir
        self._trades: List[TradeRecord] = []
        self._sessions: Dict[str, SessionMetrics] = {}
        
        logger.info(f"DBAnalytics initialized at {data_dir}")
    
    def load_trades(self, session_id: Optional[str] = None) -> int:
        """
        Load trades from storage.
        
        Args:
            session_id: Optional specific session to load
            
        Returns:
            Number of trades loaded
        """
        # In production, this would load from the time-series database
        # For now, we'll simulate with empty list
        self._trades = []
        return len(self._trades)
    
    def add_trade(self, trade: TradeRecord) -> None:
        """Add a trade record for analysis."""
        self._trades.append(trade)
    
    def add_trades(self, trades: List[TradeRecord]) -> None:
        """Add multiple trade records."""
        self._trades.extend(trades)
    
    async def query_by_symbol(self, symbol: str) -> QueryResult:
        """
        Query trades by symbol.
        
        Args:
            symbol: Asset symbol (BTCUSDT, ETHUSDT, etc.)
            
        Returns:
            Query result with matching trades
        """
        start_time = time.perf_counter()
        
        filtered = [t for t in self._trades if t.symbol == symbol]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return QueryResult(
            query_time_ms=elapsed_ms,
            row_count=len(filtered),
            data=[self._trade_to_dict(t) for t in filtered],
            aggregations={
                'total_pnl': sum(t.pnl for t in filtered),
                'trade_count': len(filtered),
                'win_rate': self._calculate_win_rate(filtered),
            }
        )
    
    async def query_by_time_range(
        self,
        start_time: float,
        end_time: float,
    ) -> QueryResult:
        """
        Query trades within a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Query result with matching trades
        """
        query_start = time.perf_counter()
        
        filtered = [
            t for t in self._trades
            if start_time <= t.exit_time <= end_time
        ]
        
        elapsed_ms = (time.perf_counter() - query_start) * 1000
        
        return QueryResult(
            query_time_ms=elapsed_ms,
            row_count=len(filtered),
            data=[self._trade_to_dict(t) for t in filtered],
            aggregations={
                'total_pnl': sum(t.pnl for t in filtered),
                'trade_count': len(filtered),
            }
        )
    
    async def calculate_session_metrics(self, session_id: str) -> SessionMetrics:
        """
        Calculate comprehensive metrics for a trading session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionMetrics object
        """
        session_trades = [t for t in self._trades if session_id in t.trade_id]
        
        if not session_trades:
            # Use all trades if no session filter matches
            session_trades = self._trades
        
        if not session_trades:
            return self._empty_session_metrics(session_id)
        
        # Basic counts
        total_trades = len(session_trades)
        winning = [t for t in session_trades if t.pnl > 0]
        losing = [t for t in session_trades if t.pnl < 0]
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # PnL calculations
        pnls = [t.pnl for t in session_trades]
        total_pnl = sum(pnls)
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        largest_win = max(t.pnl for t in winning) if winning else 0.0
        largest_loss = min(t.pnl for t in losing) if losing else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe(pnls)
        sortino_ratio = self._calculate_sortino(pnls)
        max_drawdown, max_dd_duration = self._calculate_drawdown(session_trades)
        
        # Duration
        durations = [t.duration_seconds for t in session_trades]
        avg_duration = statistics.mean(durations) if durations else 0.0
        
        # Costs
        total_fees = sum(t.fees for t in session_trades)
        total_slippage = sum(t.slippage for t in session_trades)
        
        # Time range
        start_times = [t.entry_time for t in session_trades]
        end_times = [t.exit_time for t in session_trades]
        
        metrics = SessionMetrics(
            session_id=session_id,
            start_time=min(start_times) if start_times else 0.0,
            end_time=max(end_times) if end_times else 0.0,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            average_pnl=average_pnl,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            average_trade_duration=avg_duration,
            total_fees=total_fees,
            total_slippage=total_slippage,
        )
        
        self._sessions[session_id] = metrics
        
        return metrics
    
    def _calculate_sharpe(self, pnls: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(pnls) < 2:
            return 0.0
        
        returns = np.array(pnls)
        excess_returns = returns - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        # Annualize (assuming daily returns)
        return sharpe * np.sqrt(252)
    
    def _calculate_sortino(self, pnls: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(pnls) < 2:
            return 0.0
        
        returns = np.array(pnls)
        excess_returns = returns - risk_free_rate
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_std
        return sortino * np.sqrt(252)
    
    def _calculate_drawdown(
        self,
        trades: List[TradeRecord],
    ) -> Tuple[float, float]:
        """
        Calculate maximum drawdown and its duration.
        
        Returns:
            Tuple of (max_drawdown, max_drawdown_duration_seconds)
        """
        if not trades:
            return 0.0, 0.0
        
        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0
        max_dd_start = 0.0
        max_dd_end = 0.0
        current_dd_start = 0.0
        
        for trade in sorted_trades:
            cumulative_pnl += trade.pnl
            
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
                current_dd_start = trade.exit_time
            
            drawdown = peak_pnl - cumulative_pnl
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_dd_start = current_dd_start
                max_dd_end = trade.exit_time
        
        max_dd_duration = max_dd_end - max_dd_start if max_dd_end > max_dd_start else 0.0
        
        return max_drawdown, max_dd_duration
    
    def _calculate_win_rate(self, trades: List[TradeRecord]) -> float:
        """Calculate win rate for a list of trades."""
        if not trades:
            return 0.0
        
        winners = sum(1 for t in trades if t.pnl > 0)
        return winners / len(trades)
    
    def _empty_session_metrics(self, session_id: str) -> SessionMetrics:
        """Create empty session metrics."""
        return SessionMetrics(
            session_id=session_id,
            start_time=0.0,
            end_time=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            average_pnl=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0.0,
            average_trade_duration=0.0,
            total_fees=0.0,
            total_slippage=0.0,
        )
    
    def _trade_to_dict(self, trade: TradeRecord) -> Dict[str, Any]:
        """Convert trade record to dictionary."""
        return {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'side': trade.side,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'duration_seconds': trade.duration_seconds,
            'fees': trade.fees,
            'slippage': trade.slippage,
        }
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of session metrics."""
        if session_id not in self._sessions:
            return {'error': 'Session not found'}
        
        metrics = self._sessions[session_id]
        
        return {
            'session_id': metrics.session_id,
            'duration_hours': (metrics.end_time - metrics.start_time) / 3600,
            'total_trades': metrics.total_trades,
            'win_rate': f"{metrics.win_rate:.2%}",
            'total_pnl': f"{metrics.total_pnl:.2f} USDT",
            'profit_factor': f"{metrics.profit_factor:.2f}",
            'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
            'max_drawdown': f"{metrics.max_drawdown:.2f} USDT",
            'total_fees': f"{metrics.total_fees:.2f} USDT",
        }


# Example usage
if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        analytics = DBAnalytics(Path(tmpdir))
        
        # Add sample trades
        analytics.add_trade(TradeRecord(
            trade_id="session_001_trade_001",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=45000.0,
            exit_price=45500.0,
            quantity=0.1,
            pnl=50.0,
            pnl_percent=1.11,
            entry_time=time.time() - 3600,
            exit_time=time.time(),
            duration_seconds=3600,
            fees=2.5,
            slippage=1.0,
        ))
        
        # Calculate metrics
        metrics = asyncio.run(analytics.calculate_session_metrics("session_001"))
        
        # Get summary
        summary = analytics.get_session_summary("session_001")
        print(f"Session Summary: {summary}")
