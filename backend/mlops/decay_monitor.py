#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
High-Frequency Feature Stores, MLOps, and Concept Drift
File: backend/mlops/decay_monitor.py
Chapter 2: Concept Drift Detection, Online Calibration, and Model Decay

Tracks the half-life of alpha signals to trigger retraining.
Automatically disables models whose Sharpe ratio drops below 1.0.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
Uses exponential decay modeling and rolling performance metrics.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import threading
import time


class ModelStatus(Enum):
    """Status of a model in the decay monitoring system."""
    ACTIVE = "active"
    DECAYING = "decaying"
    RETRAINING = "retraining"
    DISABLED = "disabled"
    ARCHIVED = "archived"


@dataclass
class AlphaSignal:
    """Represents an alpha signal with decay characteristics."""
    signal_id: str
    asset_id: int
    initial_strength: float
    current_strength: float
    half_life_periods: float  # Number of periods until half-life
    creation_time: float
    last_update_time: float
    decay_rate: float  # Exponential decay constant
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Rolling performance metrics for a model."""
    returns: deque
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    last_updated: float


@dataclass
class DecayAlert:
    """Alert generated when model decay is detected."""
    timestamp: float
    model_id: str
    alert_type: str  # 'half_life_reached', 'sharpe_below_threshold', 'alpha_decay'
    current_sharpe: float
    threshold_sharpe: float
    alpha_half_life_remaining: float
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecayMonitor:
    """
    Monitors model decay and alpha signal half-life.
    
    Key responsibilities:
    1. Track exponential decay of alpha signals
    2. Monitor rolling Sharpe ratio
    3. Trigger retraining when performance degrades
    4. Automatically disable models with Sharpe < 1.0
    5. Maintain performance history for analysis
    
    Optimized for high-frequency trading with minimal overhead.
    """
    
    def __init__(
        self,
        sharpe_threshold: float = 1.0,
        half_life_warning_threshold: float = 0.5,
        rolling_window_size: int = 100,
        min_trades_for_evaluation: int = 30,
        memory_budget_mb: int = 128
    ):
        """
        Initialize the decay monitor.
        
        Args:
            sharpe_threshold: Minimum acceptable Sharpe ratio
            half_life_warning_threshold: Fraction of half-life remaining for warning
            rolling_window_size: Size of rolling window for performance metrics
            min_trades_for_evaluation: Minimum trades before evaluating Sharpe
            memory_budget_mb: Memory budget for tracking
        """
        self.sharpe_threshold = sharpe_threshold
        self.half_life_warning_threshold = half_life_warning_threshold
        self.rolling_window_size = rolling_window_size
        self.min_trades_for_evaluation = min_trades_for_evaluation
        self.memory_budget_bytes = memory_budget_mb * 1024 * 1024
        
        # Alpha signals per model
        # Structure: {model_id: {signal_id: AlphaSignal}}
        self.alpha_signals: Dict[str, Dict[str, AlphaSignal]] = {}
        
        # Performance metrics per model
        self.performance: Dict[str, PerformanceMetrics] = {}
        
        # Model status
        self.model_status: Dict[str, ModelStatus] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Alert history
        self.alerts: deque = deque(maxlen=500)
        
        # Statistics
        self.stats = {
            'models_monitored': 0,
            'alerts_generated': 0,
            'models_disabled': 0,
            'retrains_triggered': 0
        }
    
    def register_model(
        self,
        model_id: str,
        initial_alpha_signals: Optional[List[AlphaSignal]] = None
    ) -> None:
        """
        Register a new model for decay monitoring.
        
        Args:
            model_id: Unique model identifier
            initial_alpha_signals: Optional list of initial alpha signals
        """
        with self._lock:
            self.alpha_signals[model_id] = {}
            self.performance[model_id] = self._create_performance_metrics()
            self.model_status[model_id] = ModelStatus.ACTIVE
            
            if initial_alpha_signals:
                for signal in initial_alpha_signals:
                    self.add_alpha_signal(model_id, signal)
            
            self.stats['models_monitored'] += 1
    
    def _create_performance_metrics(self) -> PerformanceMetrics:
        """Create empty performance metrics structure."""
        return PerformanceMetrics(
            returns=deque(maxlen=self.rolling_window_size),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_trades=0,
            last_updated=time.time()
        )
    
    def add_alpha_signal(self, model_id: str, signal: AlphaSignal) -> None:
        """
        Add an alpha signal to track.
        
        Args:
            model_id: Model identifier
            signal: AlphaSignal object
        """
        with self._lock:
            if model_id not in self.alpha_signals:
                self.register_model(model_id)
            
            self.alpha_signals[model_id][signal.signal_id] = signal
    
    def record_trade(
        self,
        model_id: str,
        return_pct: float,
        is_win: bool,
        pnl: float
    ) -> Optional[DecayAlert]:
        """
        Record a trade result and check for decay.
        
        Args:
            model_id: Model identifier
            return_pct: Return percentage from the trade
            is_win: Whether the trade was profitable
            pnl: Profit/Loss amount
            
        Returns:
            DecayAlert if decay detected, None otherwise
        """
        with self._lock:
            if model_id not in self.performance:
                self.register_model(model_id)
            
            if self.model_status.get(model_id) != ModelStatus.ACTIVE:
                return None
            
            # Update performance metrics
            perf = self.performance[model_id]
            perf.returns.append(return_pct)
            perf.total_trades += 1
            perf.last_updated = time.time()
            
            # Update win/loss stats
            if is_win:
                perf.avg_win = (perf.avg_win * (perf.total_trades - 1) + pnl) / perf.total_trades
            else:
                perf.avg_loss = (perf.avg_loss * (perf.total_trades - 1) + abs(pnl)) / perf.total_trades
            
            perf.win_rate = sum(1 for r in perf.returns if r > 0) / len(perf.returns)
            
            # Calculate rolling metrics
            self._update_rolling_metrics(perf)
            
            # Update alpha signal decay
            self._update_alpha_decay(model_id)
            
            # Check for decay conditions
            alert = self._check_decay_conditions(model_id)
            
            return alert
    
    def _update_rolling_metrics(self, perf: PerformanceMetrics) -> None:
        """Update rolling performance metrics."""
        returns = np.array(perf.returns)
        
        if len(returns) < 2:
            return
        
        # Annualized Sharpe ratio (assuming daily returns, 252 trading days)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            perf.sharpe_ratio = float((mean_return / std_return) * np.sqrt(252))
        else:
            perf.sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                perf.sortino_ratio = float((mean_return / downside_std) * np.sqrt(252))
            else:
                perf.sortino_ratio = perf.sharpe_ratio
        else:
            perf.sortino_ratio = perf.sharpe_ratio
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        perf.max_drawdown = float(np.min(drawdowns))
    
    def _update_alpha_decay(self, model_id: str) -> None:
        """Update decay state of alpha signals."""
        current_time = time.time()
        
        for signal in self.alpha_signals.get(model_id, {}).values():
            elapsed_periods = current_time - signal.last_update_time
            
            # Exponential decay: strength = initial * exp(-decay_rate * t)
            decay_factor = np.exp(-signal.decay_rate * elapsed_periods)
            signal.current_strength = signal.initial_strength * decay_factor
            signal.last_update_time = current_time
    
    def _check_decay_conditions(self, model_id: str) -> Optional[DecayAlert]:
        """Check for decay conditions and generate alerts."""
        perf = self.performance.get(model_id)
        if not perf or perf.total_trades < self.min_trades_for_evaluation:
            return None
        
        alert = None
        
        # Check Sharpe ratio threshold
        if perf.sharpe_ratio < self.sharpe_threshold:
            action = "disable_model" if perf.sharpe_ratio < self.sharpe_threshold * 0.7 else "retrain_soon"
            
            alert = DecayAlert(
                timestamp=time.time(),
                model_id=model_id,
                alert_type='sharpe_below_threshold',
                current_sharpe=perf.sharpe_ratio,
                threshold_sharpe=self.sharpe_threshold,
                alpha_half_life_remaining=self._get_avg_half_life_remaining(model_id),
                recommended_action=action,
                metadata={
                    'sortino_ratio': perf.sortino_ratio,
                    'max_drawdown': perf.max_drawdown,
                    'win_rate': perf.win_rate,
                    'total_trades': perf.total_trades
                }
            )
            
            if action == "disable_model":
                self.disable_model(model_id, f"Sharpe ratio {perf.sharpe_ratio:.2f} below threshold {self.sharpe_threshold}")
                self.stats['models_disabled'] += 1
            
            self.stats['alerts_generated'] += 1
        
        # Check alpha half-life
        half_life_remaining = self._get_avg_half_life_remaining(model_id)
        if half_life_remaining < self.half_life_warning_threshold:
            if alert is None:  # Don't overwrite Sharpe alert
                alert = DecayAlert(
                    timestamp=time.time(),
                    model_id=model_id,
                    alert_type='half_life_reached',
                    current_sharpe=perf.sharpe_ratio,
                    threshold_sharpe=self.sharpe_threshold,
                    alpha_half_life_remaining=half_life_remaining,
                    recommended_action="retrain",
                    metadata={
                        'signals_tracked': len(self.alpha_signals.get(model_id, {}))
                    }
                )
                
                self.stats['alerts_generated'] += 1
        
        if alert:
            self.alerts.append(alert)
        
        return alert
    
    def _get_avg_half_life_remaining(self, model_id: str) -> float:
        """Calculate average half-life remaining across all signals."""
        signals = self.alpha_signals.get(model_id, {})
        if not signals:
            return 1.0
        
        remaining = []
        for signal in signals.values():
            # Half-life formula: t_1/2 = ln(2) / decay_rate
            half_life = np.log(2) / signal.decay_rate if signal.decay_rate > 0 else float('inf')
            elapsed = time.time() - signal.creation_time
            remaining_fraction = 1.0 - (elapsed / half_life) if half_life != float('inf') else 1.0
            remaining.append(max(0.0, remaining_fraction))
        
        return float(np.mean(remaining))
    
    def disable_model(self, model_id: str, reason: str) -> None:
        """Disable a model due to decay."""
        with self._lock:
            self.model_status[model_id] = ModelStatus.DISABLED
            # In production, this would send a signal to stop trading
    
    def enable_model(self, model_id: str) -> bool:
        """
        Re-enable a disabled model (after retraining).
        
        Returns:
            True if enabled, False if model doesn't exist
        """
        with self._lock:
            if model_id not in self.model_status:
                return False
            
            self.model_status[model_id] = ModelStatus.ACTIVE
            
            # Reset performance metrics
            self.performance[model_id] = self._create_performance_metrics()
            
            # Reset alpha signals
            for signal in self.alpha_signals.get(model_id, {}).values():
                signal.current_strength = signal.initial_strength
                signal.last_update_time = time.time()
            
            return True
    
    def trigger_retrain(self, model_id: str) -> None:
        """Mark a model for retraining."""
        with self._lock:
            if model_id in self.model_status:
                self.model_status[model_id] = ModelStatus.RETRAINING
                self.stats['retrains_triggered'] += 1
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a model."""
        with self._lock:
            status = self.model_status.get(model_id, ModelStatus.ARCHIVED)
            perf = self.performance.get(model_id)
            
            result = {
                'model_id': model_id,
                'status': status.value,
                'sharpe_ratio': perf.sharpe_ratio if perf else 0.0,
                'sortino_ratio': perf.sortino_ratio if perf else 0.0,
                'max_drawdown': perf.max_drawdown if perf else 0.0,
                'win_rate': perf.win_rate if perf else 0.0,
                'total_trades': perf.total_trades if perf else 0,
                'alpha_signals_count': len(self.alpha_signals.get(model_id, {})),
                'avg_half_life_remaining': self._get_avg_half_life_remaining(model_id)
            }
            
            if perf:
                result['avg_win'] = perf.avg_win
                result['avg_loss'] = perf.avg_loss
            
            return result
    
    def get_all_alerts(
        self,
        since_timestamp: Optional[float] = None,
        model_id_filter: Optional[str] = None
    ) -> List[DecayAlert]:
        """Get decay alerts with optional filtering."""
        alerts = list(self.alerts)
        
        if since_timestamp is not None:
            alerts = [a for a in alerts if a.timestamp >= since_timestamp]
        
        if model_id_filter is not None:
            alerts = [a for a in alerts if a.model_id == model_id_filter]
        
        return alerts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        active_models = sum(
            1 for s in self.model_status.values() 
            if s == ModelStatus.ACTIVE
        )
        disabled_models = sum(
            1 for s in self.model_status.values() 
            if s == ModelStatus.DISABLED
        )
        
        return {
            **self.stats,
            'active_models': active_models,
            'disabled_models': disabled_models,
            'retraining_models': sum(
                1 for s in self.model_status.values() 
                if s == ModelStatus.RETRAINING
            ),
            'sharpe_threshold': self.sharpe_threshold,
            'memory_estimate_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage."""
        total_signals = sum(len(sigs) for sigs in self.alpha_signals.values())
        total_returns = sum(
            len(perf.returns) for perf in self.performance.values()
        )
        
        # Approximate bytes per item
        return (
            total_signals * 200 +  # AlphaSignal size
            total_returns * 8 +  # Float returns
            len(self.model_status) * 100  # Status overhead
        )


# Convenience function for creating a production decay monitor
def create_production_decay_monitor(
    models: List[str] = None,
    sharpe_threshold: float = 1.0
) -> DecayMonitor:
    """
    Create a production-ready decay monitor.
    
    Args:
        models: List of model IDs to monitor
        sharpe_threshold: Minimum acceptable Sharpe ratio
        
    Returns:
        Configured DecayMonitor instance
    """
    if models is None:
        models = ['default_model']
    
    monitor = DecayMonitor(
        sharpe_threshold=sharpe_threshold,
        half_life_warning_threshold=0.5,
        rolling_window_size=100,
        min_trades_for_evaluation=30,
        memory_budget_mb=128
    )
    
    for model_id in models:
        monitor.register_model(model_id)
    
    return monitor


if __name__ == '__main__':
    # Test the decay monitor
    print("Testing Decay Monitor for ZAID Trading Bot...")
    
    # Create monitor
    monitor = DecayMonitor(
        sharpe_threshold=1.0,
        min_trades_for_evaluation=10
    )
    
    # Register a model
    monitor.register_model('test_model')
    
    # Add some alpha signals
    signal = AlphaSignal(
        signal_id='momentum_signal',
        asset_id=0,
        initial_strength=1.0,
        current_strength=1.0,
        half_life_periods=1000.0,
        creation_time=time.time(),
        last_update_time=time.time(),
        decay_rate=np.log(2) / 1000.0
    )
    monitor.add_alpha_signal('test_model', signal)
    
    # Simulate trades with good performance
    print("\nSimulating good trades...")
    np.random.seed(42)
    for i in range(50):
        return_pct = np.random.normal(0.001, 0.02)  # Positive expected return
        is_win = return_pct > 0
        pnl = return_pct * 10000
        
        alert = monitor.record_trade('test_model', return_pct, is_win, pnl)
        if alert:
            print(f"  Alert at trade {i}: {alert.alert_type}")
    
    status = monitor.get_model_status('test_model')
    print(f"\nModel status after good trades:")
    print(f"  Sharpe: {status['sharpe_ratio']:.2f}")
    print(f"  Status: {status['status']}")
    print(f"  Win rate: {status['win_rate']:.2%}")
    
    # Simulate trades with poor performance
    print("\nSimulating poor trades...")
    for i in range(50):
        return_pct = np.random.normal(-0.002, 0.03)  # Negative expected return
        is_win = return_pct > 0
        pnl = return_pct * 10000
        
        alert = monitor.record_trade('test_model', return_pct, is_win, pnl)
        if alert:
            print(f"  Alert at trade {i}: {alert.alert_type}, Sharpe={alert.current_sharpe:.2f}")
    
    status = monitor.get_model_status('test_model')
    print(f"\nModel status after poor trades:")
    print(f"  Sharpe: {status['sharpe_ratio']:.2f}")
    print(f"  Status: {status['status']}")
    print(f"  Max drawdown: {status['max_drawdown']:.2%}")
    
    # Get stats
    stats = monitor.get_stats()
    print(f"\nMonitor stats: {stats}")
    
    print("\nDecay Monitor test completed!")
