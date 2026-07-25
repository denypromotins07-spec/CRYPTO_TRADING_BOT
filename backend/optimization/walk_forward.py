"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
Chapter 3: Walk-Forward Optimization and Hyperparameter Tuning

File: backend/optimization/walk_forward.py
Purpose: Implement rolling-window out-of-sample testing to prevent overfitting.
Features:
    - Rolling in-sample/out-of-sample window management
    - Multi-period walk-forward validation
    - Strategy parameter optimization per window
    - Robustness scoring across market regimes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""
    # Initial in-sample period (days)
    initial_is_days: int = 90
    
    # Out-of-sample period (days)
    os_days: int = 30
    
    # Step size between windows (days)
    step_days: int = 15
    
    # Minimum number of windows required
    min_windows: int = 3
    
    # Maximum number of windows
    max_windows: int = 20
    
    # Enable parallel processing
    parallel: bool = True
    
    # Number of workers for parallel processing
    n_workers: int = None
    
    # Parameter grid for optimization
    param_grid: Dict[str, List] = field(default_factory=dict)


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    window_id: int
    is_start: datetime
    is_end: datetime
    os_start: datetime
    os_end: datetime
    
    # In-sample optimization results
    best_params: Dict[str, Any]
    is_score: float
    is_sharpe: float
    is_drawdown: float
    is_trades: int
    
    # Out-of-sample validation results
    os_score: float
    os_sharpe: float
    os_drawdown: float
    os_trades: int
    os_pnl: float
    
    # Degradation metrics
    sharpe_degradation: float
    drawdown_increase: float
    
    # Market regime during this window
    regime: str = "unknown"


@dataclass
class WalkForwardSummary:
    """Aggregated summary of walk-forward analysis."""
    total_windows: int
    avg_os_sharpe: float
    avg_os_drawdown: float
    avg_sharpe_degradation: float
    robustness_score: float
    passed_test: bool
    window_results: List[WindowResult]
    
    def to_dict(self) -> Dict:
        return {
            'total_windows': self.total_windows,
            'avg_os_sharpe': self.avg_os_sharpe,
            'avg_os_drawdown': self.avg_os_drawdown,
            'avg_sharpe_degradation': self.avg_sharpe_degradation,
            'robustness_score': self.robustness_score,
            'passed_test': self.passed_test,
        }


class WalkForwardOptimizer:
    """
    Implements walk-forward optimization for strategy validation.
    Tests strategy robustness across multiple market periods.
    """
    
    __slots__ = [
        'config',
        'backtest_func',
        'optimize_func',
        'results',
        'data'
    ]
    
    def __init__(
        self,
        config: WalkForwardConfig,
        backtest_func: Callable = None,
        optimize_func: Callable = None
    ):
        self.config = config
        self.backtest_func = backtest_func
        self.optimize_func = optimize_func
        self.results: List[WindowResult] = []
        self.data: Optional[pd.DataFrame] = None
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the historical data for walk-forward analysis."""
        if 'timestamp' not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have timestamp column or DatetimeIndex")
        
        self.data = data.sort_index() if isinstance(data.index, pd.DatetimeIndex) else \
                    data.sort_values('timestamp')
        logger.info(f"Loaded {len(self.data)} rows for walk-forward analysis")
    
    def generate_windows(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows.
        Returns list of (is_start, is_end, os_start, os_end) tuples.
        """
        if self.data is None:
            raise ValueError("Data must be set before generating windows")
        
        if start_date is None:
            start_date = self.data.index[0] if isinstance(self.data.index, pd.DatetimeIndex) \
                        else pd.to_datetime(self.data['timestamp'].iloc[0])
        
        if end_date is None:
            end_date = self.data.index[-1] if isinstance(self.data.index, pd.DatetimeIndex) \
                      else pd.to_datetime(self.data['timestamp'].iloc[-1])
        
        windows = []
        current_start = start_date
        
        while current_start + timedelta(days=self.config.initial_is_days) <= end_date:
            is_end = current_start + timedelta(days=self.config.initial_is_days)
            os_start = is_end
            os_end = is_end + timedelta(days=self.config.os_days)
            
            if os_end > end_date:
                os_end = end_date
            
            # Ensure minimum OS period
            if (os_end - os_start).days < self.config.os_days // 2:
                break
            
            windows.append((current_start, is_end, os_start, os_end))
            
            # Move to next window
            current_start += timedelta(days=self.config.step_days)
            
            # Check max windows
            if len(windows) >= self.config.max_windows:
                break
        
        if len(windows) < self.config.min_windows:
            logger.warning(f"Only {len(windows)} windows generated, minimum required: {self.config.min_windows}")
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def _optimize_window(
        self,
        window_id: int,
        is_start: datetime,
        is_end: datetime,
        os_start: datetime,
        os_end: datetime
    ) -> WindowResult:
        """Optimize parameters on in-sample and validate on out-of-sample."""
        logger.info(f"Processing window {window_id}: IS [{is_start} - {is_end}], OS [{os_start} - {os_end}]")
        
        # Get in-sample data
        is_data = self._get_data_in_range(is_start, is_end)
        os_data = self._get_data_in_range(os_start, os_end)
        
        if len(is_data) == 0 or len(os_data) == 0:
            logger.warning(f"No data for window {window_id}")
            return None
        
        # Optimize on in-sample
        if self.optimize_func:
            best_params, is_metrics = self.optimize_func(is_data, self.config.param_grid)
        else:
            # Default: use first parameter combination
            best_params = {k: v[0] for k, v in self.config.param_grid.items()} if self.config.param_grid else {}
            is_metrics = {'sharpe': 0.0, 'drawdown': 0.0, 'pnl': 0.0, 'trades': 0}
        
        # Validate on out-of-sample
        if self.backtest_func:
            os_metrics = self.backtest_func(os_data, best_params)
        else:
            # Mock metrics
            os_metrics = {
                'sharpe': np.random.uniform(0.5, 2.0),
                'drawdown': np.random.uniform(0.05, 0.20),
                'pnl': np.random.uniform(-1000, 5000),
                'trades': np.random.randint(10, 100)
            }
        
        # Calculate degradation
        is_sharpe = is_metrics.get('sharpe', 0.0)
        os_sharpe = os_metrics.get('sharpe', 0.0)
        sharpe_degradation = (is_sharpe - os_sharpe) / max(is_sharpe, 0.01)
        
        is_dd = is_metrics.get('drawdown', 0.0)
        os_dd = os_metrics.get('drawdown', 0.0)
        drawdown_increase = os_dd - is_dd
        
        # Determine market regime
        regime = self._detect_regime(os_data)
        
        return WindowResult(
            window_id=window_id,
            is_start=is_start,
            is_end=is_end,
            os_start=os_start,
            os_end=os_end,
            best_params=best_params,
            is_score=is_metrics.get('score', 0.0),
            is_sharpe=is_sharpe,
            is_drawdown=is_dd,
            is_trades=is_metrics.get('trades', 0),
            os_score=os_metrics.get('score', 0.0),
            os_sharpe=os_sharpe,
            os_drawdown=os_dd,
            os_trades=os_metrics.get('trades', 0),
            os_pnl=os_metrics.get('pnl', 0.0),
            sharpe_degradation=sharpe_degradation,
            drawdown_increase=drawdown_increase,
            regime=regime
        )
    
    def _get_data_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Extract data within a date range."""
        if isinstance(self.data.index, pd.DatetimeIndex):
            return self.data.loc[start:end]
        else:
            mask = (self.data['timestamp'] >= start) & (self.data['timestamp'] <= end)
            return self.data[mask]
    
    def _detect_regime(self, data: pd.DataFrame) -> str:
        """Detect market regime based on price action."""
        if len(data) < 10:
            return "unknown"
        
        if 'close' not in data.columns:
            return "unknown"
        
        closes = data['close'].values
        returns = np.diff(closes) / closes[:-1]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Simple regime classification
        if std_return > 0.03:
            return "high_volatility"
        elif mean_return > 0.001:
            return "bull"
        elif mean_return < -0.001:
            return "bear"
        else:
            return "sideways"
    
    def run(self) -> WalkForwardSummary:
        """Run the complete walk-forward optimization."""
        if self.data is None:
            raise ValueError("Data must be set before running walk-forward")
        
        windows = self.generate_windows()
        
        if len(windows) < self.config.min_windows:
            logger.error(f"Insufficient windows: {len(windows)} < {self.config.min_windows}")
            return WalkForwardSummary(
                total_windows=len(windows),
                avg_os_sharpe=0.0,
                avg_os_drawdown=0.0,
                avg_sharpe_degradation=0.0,
                robustness_score=0.0,
                passed_test=False,
                window_results=[]
            )
        
        self.results = []
        
        # Process windows
        if self.config.parallel and len(windows) > 2:
            self.results = self._run_parallel(windows)
        else:
            self.results = self._run_sequential(windows)
        
        # Filter out None results
        self.results = [r for r in self.results if r is not None]
        
        # Calculate summary statistics
        return self._calculate_summary()
    
    def _run_sequential(
        self,
        windows: List[Tuple]
    ) -> List[WindowResult]:
        """Run windows sequentially."""
        results = []
        for i, (is_start, is_end, os_start, os_end) in enumerate(windows):
            result = self._optimize_window(i, is_start, is_end, os_start, os_end)
            results.append(result)
        return results
    
    def _run_parallel(
        self,
        windows: List[Tuple]
    ) -> List[WindowResult]:
        """Run windows in parallel using multiprocessing."""
        n_workers = self.config.n_workers or max(1, mp.cpu_count() - 1)
        
        logger.info(f"Running walk-forward with {n_workers} workers")
        
        # Prepare arguments for parallel execution
        args = [
            (i, is_start, is_end, os_start, os_end)
            for i, (is_start, is_end, os_start, os_end) in enumerate(windows)
        ]
        
        # Use ThreadPoolExecutor for I/O bound or ProcessPoolExecutor for CPU bound
        # Since backtesting can be CPU intensive, we'll use threads with GIL considerations
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                lambda x: self._optimize_window(*x),
                args
            ))
        
        return results
    
    def _calculate_summary(self) -> WalkForwardSummary:
        """Calculate aggregated summary statistics."""
        if not self.results:
            return WalkForwardSummary(
                total_windows=0,
                avg_os_sharpe=0.0,
                avg_os_drawdown=0.0,
                avg_sharpe_degradation=0.0,
                robustness_score=0.0,
                passed_test=False,
                window_results=[]
            )
        
        os_sharpes = [r.os_sharpe for r in self.results]
        os_drawdowns = [r.os_drawdown for r in self.results]
        degradations = [r.sharpe_degradation for r in self.results]
        
        avg_os_sharpe = np.mean(os_sharpes)
        avg_os_drawdown = np.mean(os_drawdowns)
        avg_degradation = np.mean(degradations)
        
        # Calculate robustness score (0-1 scale)
        # Higher is better: high Sharpe, low degradation, consistent results
        sharpe_std = np.std(os_sharpes)
        consistency_score = 1.0 / (1.0 + sharpe_std)  # Lower std = higher consistency
        
        robustness_score = (
            0.4 * min(avg_os_sharpe / 2.0, 1.0) +  # Normalized Sharpe contribution
            0.3 * (1.0 - min(avg_degradation, 1.0)) +  # Low degradation contribution
            0.3 * consistency_score  # Consistency contribution
        )
        
        # Pass criteria
        passed = (
            avg_os_sharpe > 0.5 and
            avg_os_drawdown < 0.25 and
            avg_degradation < 0.5 and
            len(self.results) >= self.config.min_windows
        )
        
        return WalkForwardSummary(
            total_windows=len(self.results),
            avg_os_sharpe=avg_os_sharpe,
            avg_os_drawdown=avg_os_drawdown,
            avg_sharpe_degradation=avg_degradation,
            robustness_score=robustness_score,
            passed_test=passed,
            window_results=self.results
        )
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get the most frequently optimal parameters across windows."""
        if not self.results:
            return {}
        
        param_counts = {}
        for result in self.results:
            for key, value in result.best_params.items():
                if key not in param_counts:
                    param_counts[key] = {}
                if value not in param_counts[key]:
                    param_counts[key][value] = 0
                param_counts[key][value] += 1
        
        optimal = {}
        for key, values in param_counts.items():
            optimal[key] = max(values, key=values.get)
        
        return optimal


def main():
    """Example usage of walk-forward optimizer."""
    print("="*60)
    print("WALK-FORWARD OPTIMIZER DEMO")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_days = 365
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # Simulate price series with trends and volatility clusters
    returns = np.random.randn(n_days) * 0.02
    returns[:100] *= 1.5  # High vol period
    returns[200:300] *= 0.5  # Low vol period
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.exponential(1000000, n_days)
    }).set_index('timestamp')
    
    # Configure walk-forward
    config = WalkForwardConfig(
        initial_is_days=60,
        os_days=20,
        step_days=15,
        min_windows=4,
        param_grid={
            'lookback': [10, 20, 30],
            'threshold': [0.5, 1.0, 1.5],
            'stop_loss': [0.02, 0.05, 0.10]
        }
    )
    
    # Create optimizer
    optimizer = WalkForwardOptimizer(config)
    optimizer.set_data(data)
    
    # Run walk-forward
    summary = optimizer.run()
    
    print("\n" + "="*60)
    print("WALK-FORWARD SUMMARY")
    print("="*60)
    print(f"Total Windows: {summary.total_windows}")
    print(f"Avg OOS Sharpe: {summary.avg_os_sharpe:.3f}")
    print(f"Avg OOS Drawdown: {summary.avg_os_drawdown:.2%}")
    print(f"Avg Sharpe Degradation: {summary.avg_sharpe_degradation:.2%}")
    print(f"Robustness Score: {summary.robustness_score:.3f}")
    print(f"Passed Test: {'YES' if summary.passed_test else 'NO'}")
    print("="*60)
    
    # Show individual window results
    print("\nWINDOW DETAILS:")
    for result in summary.window_results:
        print(f"  Window {result.window_id}: OOS Sharpe={result.os_sharpe:.3f}, "
              f"Degradation={result.sharpe_degradation:.2%}, Regime={result.regime}")
    
    # Get optimal parameters
    optimal = optimizer.get_optimal_parameters()
    print(f"\nOptimal Parameters: {optimal}")


if __name__ == "__main__":
    main()
