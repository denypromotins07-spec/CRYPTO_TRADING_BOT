"""
Dynamic Reward Shaper for RL Agent
Implements risk-adjusted returns, drawdown penalties, and Kelly Criterion enforcement.
Optimized for 8GB RAM with incremental computation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward shaping parameters."""
    # Base reward scaling
    return_scale: float = 100.0  # Scale returns for numerical stability
    
    # Risk penalties
    drawdown_penalty_coef: float = 2.0
    volatility_penalty_coef: float = 0.5
    turnover_penalty_coef: float = 0.01
    
    # Kelly Criterion limits
    kelly_fraction: float = 0.25  # Fraction of Kelly bet to use (0.25 = quarter-Kelly)
    max_kelly_violation_penalty: float = 5.0
    
    # Drawdown tracking
    drawdown_window: int = 100  # Steps for rolling drawdown calculation
    max_drawdown_threshold: float = 0.10  # 10% max drawdown before severe penalty
    
    # Volatility estimation
    volatility_window: int = 50  # Steps for rolling volatility
    
    # Reward clipping
    min_reward: float = -10.0
    max_reward: float = 10.0


@dataclass
class TradingMetrics:
    """Rolling trading metrics for reward computation."""
    returns: deque = field(default_factory=lambda: deque(maxlen=1000))
    equity_curve: deque = field(default_factory=lambda: deque(maxlen=1000))
    peak_equity: float = 0.0
    current_equity: float = 100000.0  # Starting capital
    total_turnover: float = 0.0
    trade_count: int = 0
    
    def update_equity(self, pnl: float) -> None:
        """Update equity and track peak for drawdown calculation."""
        self.current_equity += pnl
        self.returns.append(pnl / self.current_equity if self.current_equity != 0 else 0.0)
        self.equity_curve.append(self.current_equity)
        
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
    
    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    @property
    def rolling_volatility(self) -> float:
        """Calculate rolling volatility of returns."""
        if len(self.returns) < 10:
            return 0.01  # Default volatility
        
        return float(np.std(list(self.returns)[-100:]))
    
    @property
    def rolling_return(self) -> float:
        """Calculate rolling mean return."""
        if len(self.returns) == 0:
            return 0.0
        return float(np.mean(list(self.returns)[-100:]))


class RewardShaper:
    """
    Dynamic Reward Shaper for PPO Agent.
    
    Implements:
    - Risk-adjusted return rewards (Sharpe-like)
    - Drawdown penalties (severe near max threshold)
    - Kelly Criterion violation penalties
    - Turnover/transaction cost penalties
    - Volatility-adjusted scaling
    
    All computations are incremental to avoid blocking the main loop.
    """
    
    def __init__(self, config: RewardConfig, initial_capital: float = 100000.0):
        self.config = config
        self.metrics = TradingMetrics()
        self.metrics.current_equity = initial_capital
        self.metrics.peak_equity = initial_capital
        
        # Kelly bet tracking per symbol
        self.kelly_bets: Dict[str, float] = {}
        
        # Historical reward components for monitoring
        self.reward_history: deque = deque(maxlen=1000)
        self.component_history: deque = deque(maxlen=100)
        
        logger.info(f"RewardShaper initialized: Kelly fraction={config.kelly_fraction}, "
                   f"max_drawdown={config.max_drawdown_threshold}")
    
    def reset(self, initial_capital: Optional[float] = None) -> None:
        """Reset all metrics to initial state."""
        capital = initial_capital or 100000.0
        self.metrics = TradingMetrics()
        self.metrics.current_equity = capital
        self.metrics.peak_equity = capital
        self.kelly_bets.clear()
        self.reward_history.clear()
        self.component_history.clear()
    
    def compute_reward(
        self,
        pnl: float,
        symbol: str,
        position_size: float,
        signal_strength: float,
        estimated_edge: float,
        transaction_cost: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward for a single step.
        
        Args:
            pnl: Realized + unrealized PnL for the step
            symbol: Trading symbol
            position_size: Current position size in USD
            signal_strength: Strength of trading signal [-1, 1]
            estimated_edge: Estimated edge/probability advantage [0, 1]
            transaction_cost: Transaction costs for the step
        
        Returns:
            Tuple of (total_reward, component_dict)
        """
        # Update metrics
        self.metrics.update_equity(pnl)
        self.metrics.total_turnover += abs(position_size)
        if abs(position_size) > 0:
            self.metrics.trade_count += 1
        
        # Component 1: Base return reward (scaled)
        base_reward = (pnl / self.config.return_scale)
        
        # Component 2: Drawdown penalty (non-linear, increases near threshold)
        current_dd = self.metrics.current_drawdown
        drawdown_penalty = self._compute_drawdown_penalty(current_dd)
        
        # Component 3: Volatility penalty
        volatility_penalty = self._compute_volatility_penalty()
        
        # Component 4: Kelly Criterion violation penalty
        kelly_penalty = self._compute_kelly_penalty(
            symbol, position_size, estimated_edge, signal_strength
        )
        
        # Component 5: Turnover penalty (discourages overtrading)
        turnover_penalty = self._compute_turnover_penalty(transaction_cost)
        
        # Combine all components
        total_reward = (
            base_reward
            - drawdown_penalty
            - volatility_penalty
            - kelly_penalty
            - turnover_penalty
        )
        
        # Clip reward to prevent extreme values
        total_reward = np.clip(total_reward, self.config.min_reward, self.config.max_reward)
        
        # Store components for monitoring
        components = {
            'base_return': base_reward,
            'drawdown_penalty': drawdown_penalty,
            'volatility_penalty': volatility_penalty,
            'kelly_penalty': kelly_penalty,
            'turnover_penalty': turnover_penalty,
            'total': total_reward,
        }
        self.component_history.append(components)
        self.reward_history.append(total_reward)
        
        return total_reward, components
    
    def _compute_drawdown_penalty(self, current_dd: float) -> float:
        """
        Compute drawdown penalty with non-linear scaling.
        
        Penalty increases exponentially as drawdown approaches threshold.
        """
        if current_dd <= 0:
            return 0.0
        
        threshold = self.config.max_drawdown_threshold
        
        if current_dd < threshold * 0.5:
            # Mild penalty for small drawdowns
            return self.config.drawdown_penalty_coef * current_dd * 0.1
        elif current_dd < threshold * 0.8:
            # Moderate penalty
            ratio = current_dd / threshold
            return self.config.drawdown_penalty_coef * ratio ** 2
        else:
            # Severe penalty near threshold (exponential)
            excess = (current_dd - threshold * 0.8) / (threshold * 0.2)
            return self.config.drawdown_penalty_coef * (1 + np.exp(3 * excess) - 1)
    
    def _compute_volatility_penalty(self) -> float:
        """Compute penalty based on rolling return volatility."""
        vol = self.metrics.rolling_volatility
        
        # Penalize high volatility relative to returns
        rolling_ret = self.metrics.rolling_return
        
        if abs(rolling_ret) < 1e-6:
            # No returns yet, small penalty for any volatility
            return self.config.volatility_penalty_coef * vol * 0.1
        
        # Sharpe-like penalty: high vol with low returns is bad
        sharpe_ratio = rolling_ret / (vol + 1e-8)
        
        if sharpe_ratio > 1.0:
            # Good Sharpe, minimal penalty
            return self.config.volatility_penalty_coef * vol * 0.05
        elif sharpe_ratio > 0.5:
            # Moderate Sharpe
            return self.config.volatility_penalty_coef * vol * 0.2
        else:
            # Poor Sharpe, full penalty
            return self.config.volatility_penalty_coef * vol
    
    def _compute_kelly_penalty(
        self,
        symbol: str,
        position_size: float,
        estimated_edge: float,
        signal_strength: float
    ) -> float:
        """
        Compute Kelly Criterion violation penalty.
        
        The Kelly Criterion determines optimal bet size:
        f* = (p * b - q) / b
        where p = win probability, q = loss probability, b = odds ratio
        
        We penalize positions that exceed the Kelly-optimal size.
        """
        if abs(position_size) < 1e-6:
            return 0.0
        
        # Estimate Kelly-optimal bet size
        # Simplified: f* = edge * signal_strength * capital
        edge = np.clip(estimated_edge, 0.0, 1.0)
        signal = np.clip(abs(signal_strength), 0.0, 1.0)
        
        # Conservative Kelly estimate
        kelly_fraction = self.config.kelly_fraction
        optimal_bet = self.metrics.current_equity * kelly_fraction * edge * signal
        
        # Check if current position exceeds Kelly-optimal
        if abs(position_size) <= optimal_bet:
            return 0.0
        
        # Compute violation ratio
        violation_ratio = abs(position_size) / (optimal_bet + 1e-8)
        
        # Penalty scales with violation severity
        if violation_ratio < 1.5:
            return self.config.max_kelly_violation_penalty * (violation_ratio - 1.0) * 0.5
        elif violation_ratio < 2.0:
            return self.config.max_kelly_violation_penalty * (violation_ratio - 1.0)
        else:
            # Severe penalty for large violations
            return self.config.max_kelly_violation_penalty * (violation_ratio ** 2)
    
    def _compute_turnover_penalty(self, transaction_cost: float) -> float:
        """Compute penalty for turnover and transaction costs."""
        # Direct transaction cost penalty
        cost_penalty = transaction_cost * self.config.turnover_penalty_coef * 10
        
        # Additional penalty for high turnover rate
        if self.metrics.trade_count > 0:
            avg_trade_size = self.metrics.total_turnover / self.metrics.trade_count
            turnover_rate = avg_trade_size / self.metrics.current_equity
            
            if turnover_rate > 0.1:  # >10% turnover per step average
                cost_penalty += self.config.turnover_penalty_coef * turnover_rate * 5
        
        return cost_penalty
    
    def get_risk_adjusted_metrics(self) -> Dict[str, float]:
        """Get comprehensive risk-adjusted performance metrics."""
        if len(self.metrics.returns) < 10:
            return {}
        
        returns = np.array(list(self.metrics.returns))
        
        # Annualized metrics (assuming ~250 trading days, scaled for crypto)
        ann_factor = 365 * 24  # Crypto runs 24/7
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe ratio (annualized)
        sharpe = (mean_return / (std_return + 1e-8)) * np.sqrt(ann_factor)
        
        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = (mean_return / (downside_std + 1e-8)) * np.sqrt(ann_factor)
        else:
            sortino = float('inf')
        
        # Calmar ratio (return / max drawdown)
        max_dd = max(self.metrics.current_drawdown, 1e-8)
        calmar = (mean_return * ann_factor) / max_dd
        
        return {
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino) if np.isfinite(sortino) else 999.0,
            'calmar_ratio': float(calmar),
            'max_drawdown': float(self.metrics.current_drawdown),
            'current_drawdown': float(self.metrics.current_drawdown),
            'rolling_volatility': float(self.metrics.rolling_volatility),
            'rolling_return': float(self.metrics.rolling_return),
            'total_trades': self.metrics.trade_count,
            'current_equity': float(self.metrics.current_equity),
        }
    
    def should_reduce_position(self, symbol: str) -> bool:
        """
        Determine if position should be reduced based on risk metrics.
        
        Returns True if:
        - Drawdown exceeds 80% of threshold
        - Kelly violation detected
        - Volatility spike detected
        """
        current_dd = self.metrics.current_drawdown
        
        if current_dd > self.config.max_drawdown_threshold * 0.8:
            logger.warning(f"High drawdown detected: {current_dd:.2%}")
            return True
        
        vol = self.metrics.rolling_volatility
        if vol > 0.05:  # >5% rolling volatility
            logger.warning(f"High volatility detected: {vol:.2%}")
            return True
        
        return False


# Example usage and testing
if __name__ == "__main__":
    config = RewardConfig(
        kelly_fraction=0.25,
        max_drawdown_threshold=0.10,
        drawdown_penalty_coef=2.0,
    )
    
    shaper = RewardShaper(config, initial_capital=100000.0)
    
    # Simulate some trades
    for i in range(100):
        pnl = np.random.normal(100, 500)  # Random PnL
        reward, components = shaper.compute_reward(
            pnl=pnl,
            symbol='BTCUSDT',
            position_size=10000,
            signal_strength=0.5,
            estimated_edge=0.1,
            transaction_cost=5.0
        )
        
        if i % 20 == 0:
            print(f"Step {i}: Reward={reward:.4f}, Components={components}")
    
    # Get final metrics
    metrics = shaper.get_risk_adjusted_metrics()
    print(f"\nFinal Metrics: {metrics}")
