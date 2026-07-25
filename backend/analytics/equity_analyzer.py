"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
Chapter 4: Backtest Analytics, Equity Curve Analysis, and SOUL.md Integration

File: backend/analytics/equity_analyzer.py
Purpose: Calculate Sortino, Calmar, maximum drawdown, and advanced equity metrics.
Features:
    - Comprehensive risk-adjusted return metrics
    - Drawdown analysis with recovery tracking
    - Tail risk measures (VaR, CVaR)
    - Equity curve quality scoring
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EquityMetrics:
    """Comprehensive equity curve metrics."""
    # Returns metrics
    total_return: float
    annualized_return: float
    cagr: float
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    
    # Tail risk
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    
    # Trade statistics
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Consistency metrics
    consecutive_wins_max: int
    consecutive_losses_max: int
    recovery_factor: float
    
    # Quality score (0-100)
    equity_quality_score: float


class EquityAnalyzer:
    """
    Advanced equity curve analyzer for backtest evaluation.
    Implements institutional-grade performance metrics.
    """
    
    __slots__ = [
        'equity_curve',
        'returns',
        'trades',
        'risk_free_rate',
        'periods_per_year',
        'target_return'
    ]
    
    def __init__(
        self,
        equity_curve: np.ndarray = None,
        trades: List[Dict] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        target_return: float = 0.0
    ):
        self.equity_curve = equity_curve
        self.trades = trades or []
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.target_return = target_return
        
        if equity_curve is not None:
            self.returns = self._calculate_returns(equity_curve)
        else:
            self.returns = None
    
    def _calculate_returns(self, equity: np.ndarray) -> np.ndarray:
        """Calculate log returns from equity curve."""
        equity = np.asarray(equity, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(np.log(equity))
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return returns
    
    def set_equity_curve(self, equity: np.ndarray) -> None:
        """Set the equity curve for analysis."""
        self.equity_curve = np.asarray(equity, dtype=np.float64)
        self.returns = self._calculate_returns(self.equity_curve)
    
    def set_trades(self, trades: List[Dict]) -> None:
        """Set trade list for trade-level analysis."""
        self.trades = trades
    
    def calculate_total_return(self) -> float:
        """Calculate total return percentage."""
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve[-1] / self.equity_curve[0]) - 1
    
    def calculate_cagr(self, years: float = None) -> float:
        """Calculate Compound Annual Growth Rate."""
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0
        
        total_return = self.calculate_total_return()
        
        if years is None:
            years = len(self.equity_curve) / self.periods_per_year
        
        if years <= 0:
            return 0.0
        
        cagr = (1 + total_return) ** (1 / years) - 1
        return cagr
    
    def calculate_volatility(self, annualize: bool = True) -> float:
        """Calculate annualized volatility."""
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        vol = np.std(self.returns, ddof=1)
        
        if annualize:
            vol *= np.sqrt(self.periods_per_year)
        
        return vol
    
    def calculate_downside_deviation(
        self,
        threshold: float = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate downside deviation (semi-deviation).
        Only considers returns below threshold (default: 0 or risk-free rate).
        """
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        if threshold is None:
            threshold = self.risk_free_rate / self.periods_per_year
        
        downside_returns = self.returns[self.returns < threshold]
        
        if len(downside_returns) < 2:
            return 0.0
        
        downside_std = np.sqrt(np.mean((downside_returns - threshold) ** 2))
        
        if annualize:
            downside_std *= np.sqrt(self.periods_per_year)
        
        return downside_std
    
    def calculate_drawdown_series(self) -> np.ndarray:
        """Calculate drawdown series from equity curve."""
        if self.equity_curve is None:
            return np.array([])
        
        equity = np.asarray(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        
        return drawdown
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        drawdown = self.calculate_drawdown_series()
        
        if len(drawdown) == 0:
            return 0.0
        
        return np.min(drawdown)
    
    def calculate_avg_drawdown(self) -> float:
        """Calculate average drawdown during drawdown periods."""
        drawdown = self.calculate_drawdown_series()
        
        if len(drawdown) == 0:
            return 0.0
        
        # Only consider periods when in drawdown
        in_drawdown = drawdown < 0
        
        if not np.any(in_drawdown):
            return 0.0
        
        return np.mean(drawdown[in_drawdown])
    
    def calculate_max_drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in periods."""
        drawdown = self.calculate_drawdown_series()
        
        if len(drawdown) == 0:
            return 0
        
        # Find drawdown periods (when DD < 0)
        in_drawdown = drawdown < 0
        
        # Find consecutive drawdown periods
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        excess_return = np.mean(self.returns) - (self.risk_free_rate / self.periods_per_year)
        volatility = np.std(self.returns, ddof=1)
        
        if volatility == 0:
            return 0.0
        
        sharpe = excess_return / volatility
        sharpe *= np.sqrt(self.periods_per_year)  # Annualize
        
        return sharpe
    
    def calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)."""
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        excess_return = np.mean(self.returns) - (self.risk_free_rate / self.periods_per_year)
        downside_dev = self.calculate_downside_deviation(annualize=True)
        
        if downside_dev == 0:
            return 0.0
        
        sortino = excess_return * np.sqrt(self.periods_per_year) / downside_dev
        
        return sortino
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        cagr = self.calculate_cagr()
        max_dd = abs(self.calculate_max_drawdown())
        
        if max_dd == 0:
            return 0.0
        
        return cagr / max_dd
    
    def calculate_sterling_ratio(self) -> float:
        """Calculate Sterling ratio (return / (max DD - 10%))."""
        cagr = self.calculate_cagr()
        max_dd = abs(self.calculate_max_drawdown())
        
        denominator = max_dd - 0.10
        
        if denominator <= 0:
            return 0.0
        
        return cagr / denominator
    
    def calculate_burke_ratio(self) -> float:
        """Calculate Burke ratio (uses RMS of drawdowns)."""
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        cagr = self.calculate_cagr()
        drawdown = self.calculate_drawdown_series()
        
        # RMS of drawdowns
        rms_dd = np.sqrt(np.mean(drawdown ** 2))
        
        if rms_dd == 0:
            return 0.0
        
        return cagr / rms_dd
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk at given confidence level."""
        if self.returns is None or len(self.returns) < 10:
            return 0.0
        
        return np.percentile(self.returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if self.returns is None or len(self.returns) < 10:
            return 0.0
        
        var = self.calculate_var(confidence)
        cvar = np.mean(self.returns[self.returns <= var])
        
        return cvar
    
    def analyze_trades(self) -> Dict:
        """Analyze trade-level statistics."""
        if not self.trades:
            return {
                'n_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
            }
        
        pnls = [t.get('pnl', 0.0) for t in self.trades]
        pnls = np.array(pnls)
        
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        n_trades = len(pnls)
        n_wins = len(wins)
        n_losses = len(losses)
        
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        
        largest_win = np.max(wins) if len(wins) > 0 else 0.0
        largest_loss = np.min(losses) if len(losses) > 0 else 0.0
        
        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
        }
    
    def calculate_consecutive_stats(self) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        if not self.trades:
            return 0, 0
        
        pnls = [t.get('pnl', 0.0) for t in self.trades]
        
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return max_consec_wins, max_consec_losses
    
    def calculate_recovery_factor(self) -> float:
        """Calculate recovery factor (net profit / max drawdown)."""
        total_profit = sum(t.get('pnl', 0.0) for t in self.trades) if self.trades else 0.0
        max_dd = abs(self.calculate_max_drawdown())
        
        if max_dd == 0:
            return 0.0
        
        return total_profit / max_dd
    
    def calculate_equity_quality_score(self) -> float:
        """
        Calculate overall equity curve quality score (0-100).
        Higher is better.
        """
        scores = []
        
        # Sharpe contribution (0-25 points)
        sharpe = self.calculate_sharpe_ratio()
        sharpe_score = min(25, max(0, sharpe * 10))
        scores.append(sharpe_score)
        
        # Max DD contribution (0-25 points)
        max_dd = abs(self.calculate_max_drawdown())
        dd_score = max(0, 25 - max_dd * 100)
        scores.append(dd_score)
        
        # Win rate contribution (0-15 points)
        trade_stats = self.analyze_trades()
        win_rate_score = trade_stats['win_rate'] * 15
        scores.append(win_rate_score)
        
        # Profit factor contribution (0-20 points)
        pf = trade_stats['profit_factor']
        pf_score = min(20, max(0, (pf - 1) * 10))
        scores.append(pf_score)
        
        # Consistency contribution (0-15 points)
        if self.returns is not None and len(self.returns) > 0:
            # Penalize high kurtosis (fat tails)
            kurt = stats.kurtosis(self.returns)
            consistency_score = max(0, 15 - abs(kurt) * 2)
            scores.append(consistency_score)
        else:
            scores.append(7.5)
        
        total_score = sum(scores)
        return min(100, max(0, total_score))
    
    def compute_all_metrics(self) -> EquityMetrics:
        """Compute all equity metrics at once."""
        trade_stats = self.analyze_trades()
        consec_wins, consec_losses = self.calculate_consecutive_stats()
        
        return EquityMetrics(
            total_return=self.calculate_total_return(),
            annualized_return=np.mean(self.returns) * self.periods_per_year if self.returns is not None else 0.0,
            cagr=self.calculate_cagr(),
            volatility=self.calculate_volatility(),
            downside_deviation=self.calculate_downside_deviation(),
            max_drawdown=self.calculate_max_drawdown(),
            avg_drawdown=self.calculate_avg_drawdown(),
            max_drawdown_duration_days=self.calculate_max_drawdown_duration(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            sortino_ratio=self.calculate_sortino_ratio(),
            calmar_ratio=self.calculate_calmar_ratio(),
            sterling_ratio=self.calculate_sterling_ratio(),
            burke_ratio=self.calculate_burke_ratio(),
            var_95=self.calculate_var(0.95),
            cvar_95=self.calculate_cvar(0.95),
            skewness=stats.skew(self.returns) if self.returns is not None else 0.0,
            kurtosis=stats.kurtosis(self.returns) if self.returns is not None else 0.0,
            n_trades=trade_stats['n_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            largest_win=trade_stats['largest_win'],
            largest_loss=trade_stats['largest_loss'],
            consecutive_wins_max=consec_wins,
            consecutive_losses_max=consec_losses,
            recovery_factor=self.calculate_recovery_factor(),
            equity_quality_score=self.calculate_equity_quality_score()
        )


def main():
    """Example usage of equity analyzer."""
    print("="*60)
    print("EQUITY CURVE ANALYZER")
    print("="*60)
    
    # Generate sample equity curve
    np.random.seed(42)
    n_periods = 252  # 1 year
    
    # Simulate equity curve with positive drift
    returns = 0.001 + np.random.randn(n_periods) * 0.015
    equity = 100000 * np.exp(np.cumsum(returns))
    
    # Create sample trades
    trades = [
        {'pnl': np.random.exponential(500) * (1 if np.random.rand() > 0.4 else -1)}
        for _ in range(100)
    ]
    
    # Analyze
    analyzer = EquityAnalyzer(equity_curve=equity, trades=trades)
    metrics = analyzer.compute_all_metrics()
    
    print("\n" + "="*60)
    print("EQUITY METRICS SUMMARY")
    print("="*60)
    print(f"Total Return:           {metrics.total_return*100:.2f}%")
    print(f"CAGR:                   {metrics.cagr*100:.2f}%")
    print(f"Volatility:             {metrics.volatility*100:.2f}%")
    print(f"Max Drawdown:           {metrics.max_drawdown*100:.2f}%")
    print(f"Avg Drawdown:           {metrics.avg_drawdown*100:.2f}%")
    print(f"Max DD Duration:        {metrics.max_drawdown_duration_days} days")
    print("-"*60)
    print(f"Sharpe Ratio:           {metrics.sharpe_ratio:.3f}")
    print(f"Sortino Ratio:          {metrics.sortino_ratio:.3f}")
    print(f"Calmar Ratio:           {metrics.calmar_ratio:.3f}")
    print(f"Sterling Ratio:         {metrics.sterling_ratio:.3f}")
    print(f"Burke Ratio:            {metrics.burke_ratio:.3f}")
    print("-"*60)
    print(f"VaR (95%):              {metrics.var_95*100:.3f}%")
    print(f"CVaR (95%):             {metrics.cvar_95*100:.3f}%")
    print(f"Skewness:               {metrics.skewness:.3f}")
    print(f"Kurtosis:               {metrics.kurtosis:.3f}")
    print("-"*60)
    print(f"Total Trades:           {metrics.n_trades}")
    print(f"Win Rate:               {metrics.win_rate*100:.1f}%")
    print(f"Profit Factor:          {metrics.profit_factor:.2f}")
    print(f"Avg Win:                ${metrics.avg_win:.2f}")
    print(f"Avg Loss:               ${metrics.avg_loss:.2f}")
    print(f"Largest Win:            ${metrics.largest_win:.2f}")
    print(f"Largest Loss:           ${metrics.largest_loss:.2f}")
    print(f"Consec Wins (max):      {metrics.consecutive_wins_max}")
    print(f"Consec Losses (max):    {metrics.consecutive_losses_max}")
    print(f"Recovery Factor:        {metrics.recovery_factor:.2f}")
    print("="*60)
    print(f"EQUITY QUALITY SCORE:   {metrics.equity_quality_score:.1f}/100")
    print("="*60)


if __name__ == "__main__":
    main()
