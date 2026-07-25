"""
Allocation Engine: Dynamic capital distribution across BTC, ETH, SOL, USDT.
Implements modern portfolio theory with real-time rebalancing.
Uses Strategy pattern for different allocation methodologies.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Available allocation strategies."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY_OPTIMAL = "kelly_optimal"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    DYNAMIC_REGIME = "dynamic_regime"


@dataclass
class AssetAllocation:
    """Represents allocation for a single asset."""
    asset: str
    target_weight: float
    current_weight: float
    actual_position: float
    target_position: float
    rebalance_required: bool
    last_rebalance_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "target_weight": round(self.target_weight, 4),
            "current_weight": round(self.current_weight, 4),
            "rebalance_required": self.rebalance_required
        }


@dataclass
class PortfolioState:
    """Complete portfolio state."""
    total_value: float
    allocations: Dict[str, AssetAllocation]
    cash_balance: float
    exposure: float
    leverage: float
    timestamp: float


class AllocationEngine:
    """
    Dynamic capital allocation engine for multi-asset portfolio.
    Implements multiple allocation strategies with automatic rebalancing.
    Ensures optimal risk-adjusted returns across BTC, ETH, SOL, USDT.
    """
    
    SUPPORTED_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "USDT"]
    
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        strategy: AllocationStrategy = AllocationStrategy.RISK_PARITY,
        max_leverage: float = 2.0,
        rebalance_threshold: float = 0.05,  # 5% drift triggers rebalance
    ):
        self.initial_capital = initial_capital
        self.strategy = strategy
        self.max_leverage = max_leverage
        self.rebalance_threshold = rebalance_threshold
        
        # State tracking
        self._total_value = initial_capital
        self._cash_balance = initial_capital
        self._positions: Dict[str, float] = {asset: 0.0 for asset in self.SUPPORTED_ASSETS}
        self._prices: Dict[str, float] = {asset: 1.0 for asset in self.SUPPORTED_ASSETS}
        
        # Historical data for optimization
        self._returns_history: Dict[str, deque] = {
            asset: deque(maxlen=252) for asset in self.SUPPORTED_ASSETS  # 1 year daily
        }
        self._volatility_history: Dict[str, deque] = {
            asset: deque(maxlen=63) for asset in self.SUPPORTED_ASSETS  # Quarter daily
        }
        
        # Correlation matrix cache
        self._correlation_matrix: Optional[np.ndarray] = None
        self._covariance_matrix: Optional[np.ndarray] = None
        
        # Current allocations
        self._target_weights: Dict[str, float] = {}
        self._last_rebalance_time: float = 0.0
        
        logger.info(f"AllocationEngine initialized with {strategy.value} strategy")
    
    def update_price(self, asset: str, price: float) -> None:
        """Update price for an asset."""
        if asset not in self._prices:
            logger.warning(f"Unknown asset: {asset}")
            return
        
        old_price = self._prices[asset]
        self._prices[asset] = price
        
        # Calculate return
        if old_price > 0:
            ret = (price - old_price) / old_price
            self._returns_history[asset].append(ret)
            
            # Update volatility estimate
            if len(self._volatility_history[asset]) >= 2:
                vol = np.std(list(self._returns_history[asset])[-20:]) * np.sqrt(252)
                self._volatility_history[asset].append(vol)
        
        # Recalculate portfolio value
        self._recalculate_portfolio_value()
        
        # Check if rebalance needed
        if self._needs_rebalance():
            self._trigger_rebalance()
    
    def _recalculate_portfolio_value(self) -> None:
        """Recalculate total portfolio value based on current positions and prices."""
        position_value = sum(
            self._positions[asset] * self._prices[asset]
            for asset in self.SUPPORTED_ASSETS
        )
        self._total_value = position_value + self._cash_balance
    
    def _needs_rebalance(self) -> bool:
        """Check if portfolio needs rebalancing."""
        if not self._target_weights:
            return True
        
        current_weights = self._get_current_weights()
        
        for asset in self.SUPPORTED_ASSETS:
            target = self._target_weights.get(asset, 0.0)
            current = current_weights.get(asset, 0.0)
            
            if abs(target - current) > self.rebalance_threshold:
                return True
        
        return False
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights."""
        if self._total_value <= 0:
            return {asset: 0.0 for asset in self.SUPPORTED_ASSETS}
        
        weights = {}
        for asset in self.SUPPORTED_ASSETS:
            position_value = self._positions[asset] * self._prices[asset]
            weights[asset] = position_value / self._total_value
        
        return weights
    
    def calculate_target_allocations(self) -> Dict[str, float]:
        """Calculate target allocations based on current strategy."""
        if self.strategy == AllocationStrategy.EQUAL_WEIGHT:
            return self._equal_weight_allocation()
        elif self.strategy == AllocationStrategy.RISK_PARITY:
            return self._risk_parity_allocation()
        elif self.strategy == AllocationStrategy.KELLY_OPTIMAL:
            return self._kelly_optimal_allocation()
        elif self.strategy == AllocationStrategy.MIN_VARIANCE:
            return self._min_variance_allocation()
        elif self.strategy == AllocationStrategy.MAX_SHARPE:
            return self._max_sharpe_allocation()
        elif self.strategy == AllocationStrategy.DYNAMIC_REGIME:
            return self._dynamic_regime_allocation()
        else:
            return self._equal_weight_allocation()
    
    def _equal_weight_allocation(self) -> Dict[str, float]:
        """Equal weight allocation (1/N portfolio)."""
        n_assets = len(self.SUPPORTED_ASSETS)
        weight = 1.0 / n_assets
        return {asset: weight for asset in self.SUPPORTED_ASSETS}
    
    def _risk_parity_allocation(self) -> Dict[str, float]:
        """Risk parity allocation - equal risk contribution from each asset."""
        # Get latest volatility estimates
        vols = {}
        for asset in self.SUPPORTED_ASSETS:
            vol_history = self._volatility_history[asset]
            if vol_history:
                vols[asset] = vol_history[-1]
            else:
                vols[asset] = 0.1  # Default 10% vol
        
        # Inverse volatility weighting
        inv_vols = {asset: 1.0 / max(vol, 0.001) for asset, vol in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        
        weights = {asset: inv_vol / total_inv_vol for asset, inv_vol in inv_vols.items()}
        
        # Normalize to sum to 1 (excluding USDT which is cash-like)
        risky_assets = [a for a in self.SUPPORTED_ASSETS if a != "USDT"]
        risky_weight_sum = sum(weights[a] for a in risky_assets)
        
        if risky_weight_sum > 0:
            for asset in risky_assets:
                weights[asset] /= risky_weight_sum
            weights["USDT"] = 0.0  # No allocation to stablecoin in risk parity
        
        return weights
    
    def _kelly_optimal_allocation(self) -> Dict[str, float]:
        """Kelly Criterion optimal allocation."""
        # Simplified Kelly calculation based on historical returns
        weights = {}
        total_kelly = 0.0
        
        for asset in self.SUPPORTED_ASSETS:
            returns = list(self._returns_history[asset])
            if len(returns) < 20:
                weights[asset] = 0.0
                continue
            
            # Estimate win probability and win/loss ratio
            positive_returns = [r for r in returns if r > 0]
            negative_returns = [r for r in returns if r <= 0]
            
            p_win = len(positive_returns) / len(returns)
            avg_win = np.mean(positive_returns) if positive_returns else 0.0
            avg_loss = abs(np.mean(negative_returns)) if negative_returns else 0.01
            
            if avg_loss > 0:
                win_loss_ratio = avg_win / avg_loss
                kelly_fraction = p_win - (1 - p_win) / win_loss_ratio
                
                # Apply fractional Kelly (half-Kelly for safety)
                kelly_fraction = max(0, min(kelly_fraction * 0.5, 0.25))
            else:
                kelly_fraction = 0.0
            
            weights[asset] = kelly_fraction
            total_kelly += kelly_fraction
        
        # Normalize if total > 1
        if total_kelly > 1.0:
            for asset in weights:
                weights[asset] /= total_kelly
        
        # Allocate remainder to USDT
        weights["USDT"] = max(0, 1.0 - sum(w for a, w in weights.items() if a != "USDT"))
        
        return weights
    
    def _min_variance_allocation(self) -> Dict[str, float]:
        """Minimum variance portfolio allocation."""
        # Build covariance matrix from historical returns
        self._update_covariance_matrix()
        
        if self._covariance_matrix is None:
            return self._equal_weight_allocation()
        
        try:
            # Solve for minimum variance weights
            # w = Σ^(-1) * 1 / (1^T * Σ^(-1) * 1)
            cov_inv = np.linalg.inv(self._covariance_matrix)
            ones = np.ones(len(self.SUPPORTED_ASSETS))
            
            numerator = cov_inv @ ones
            denominator = ones @ cov_inv @ ones
            
            weights_array = numerator / denominator
            
            # Ensure non-negative weights
            weights_array = np.maximum(weights_array, 0)
            weights_array = weights_array / weights_array.sum()
            
            return {
                asset: float(weight)
                for asset, weight in zip(self.SUPPORTED_ASSETS, weights_array)
            }
            
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, falling back to equal weight")
            return self._equal_weight_allocation()
    
    def _max_sharpe_allocation(self) -> Dict[str, float]:
        """Maximum Sharpe ratio portfolio allocation."""
        self._update_covariance_matrix()
        
        if self._covariance_matrix is None:
            return self._equal_weight_allocation()
        
        try:
            # Calculate expected returns
            expected_returns = np.array([
                np.mean(list(self._returns_history[asset])) * 252  # Annualized
                for asset in self.SUPPORTED_ASSETS
            ])
            
            # Maximize Sharpe: w^T * mu / sqrt(w^T * Sigma * w)
            # Solution: w ∝ Σ^(-1) * mu
            cov_inv = np.linalg.inv(self._covariance_matrix)
            
            weights_array = cov_inv @ expected_returns
            
            # Ensure non-negative and normalize
            weights_array = np.maximum(weights_array, 0)
            if weights_array.sum() > 0:
                weights_array = weights_array / weights_array.sum()
            else:
                weights_array = np.ones(len(self.SUPPORTED_ASSETS)) / len(self.SUPPORTED_ASSETS)
            
            return {
                asset: float(weight)
                for asset, weight in zip(self.SUPPORTED_ASSETS, weights_array)
            }
            
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, falling back to equal weight")
            return self._equal_weight_allocation()
    
    def _dynamic_regime_allocation(self) -> Dict[str, float]:
        """Dynamic allocation based on market regime."""
        # This would integrate with regime detection from memory/vector_store.py
        # For now, use adaptive risk parity
        
        # Check recent volatility regime
        avg_vol = np.mean([
            list(self._volatility_history[asset])[-1] if self._volatility_history[asset] else 0.1
            for asset in self.SUPPORTED_ASSETS if asset != "USDT"
        ])
        
        if avg_vol > 0.5:  # High volatility regime
            # Reduce risky asset allocation
            base_weights = self._risk_parity_allocation()
            for asset in base_weights:
                if asset != "USDT":
                    base_weights[asset] *= 0.5
            base_weights["USDT"] = 1.0 - sum(w for a, w in base_weights.items() if a != "USDT")
            return base_weights
        else:
            # Normal regime - use risk parity
            return self._risk_parity_allocation()
    
    def _update_covariance_matrix(self) -> None:
        """Update covariance matrix from historical returns."""
        min_history = 30
        
        # Check if we have enough data
        for asset in self.SUPPORTED_ASSETS:
            if len(self._returns_history[asset]) < min_history:
                return
        
        # Build returns matrix
        returns_data = []
        for asset in self.SUPPORTED_ASSETS:
            returns_data.append(list(self._returns_history[asset])[-min_history:])
        
        returns_matrix = np.array(returns_data).T
        
        # Calculate covariance matrix (annualized)
        self._covariance_matrix = np.cov(returns_matrix, rowvar=False) * 252
        
        # Calculate correlation matrix
        corr = np.corrcoef(returns_matrix, rowvar=False)
        self._correlation_matrix = corr
    
    def _trigger_rebalance(self) -> None:
        """Trigger portfolio rebalancing."""
        import time
        self._last_rebalance_time = time.time()
        
        # Recalculate target allocations
        self._target_weights = self.calculate_target_allocations()
        
        logger.info(f"Portfolio rebalanced with strategy: {self.strategy.value}")
        logger.debug(f"Target weights: {self._target_weights}")
    
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        current_weights = self._get_current_weights()
        
        allocations = {}
        for asset in self.SUPPORTED_ASSETS:
            target = self._target_weights.get(asset, 0.0)
            current = current_weights.get(asset, 0.0)
            
            position_value = self._positions[asset] * self._prices[asset]
            target_value = self._total_value * target
            
            allocations[asset] = AssetAllocation(
                asset=asset,
                target_weight=target,
                current_weight=current,
                actual_position=self._positions[asset],
                target_position=target_value / self._prices[asset] if self._prices[asset] > 0 else 0,
                rebalance_required=abs(target - current) > self.rebalance_threshold,
                last_rebalance_time=self._last_rebalance_time
            )
        
        # Calculate exposure and leverage
        risky_exposure = sum(
            self._positions[asset] * self._prices[asset]
            for asset in self.SUPPORTED_ASSETS if asset != "USDT"
        )
        
        leverage = risky_exposure / self._total_value if self._total_value > 0 else 0.0
        
        return PortfolioState(
            total_value=self._total_value,
            allocations=allocations,
            cash_balance=self._cash_balance,
            exposure=risky_exposure,
            leverage=leverage,
            timestamp=time.time()
        )
    
    def set_strategy(self, strategy: AllocationStrategy) -> None:
        """Change allocation strategy."""
        self.strategy = strategy
        logger.info(f"Allocation strategy changed to: {strategy.value}")
        self._trigger_rebalance()
    
    def execute_rebalance(self) -> List[Dict[str, Any]]:
        """Generate rebalance trades."""
        state = self.get_portfolio_state()
        trades = []
        
        for asset, alloc in state.allocations.items():
            if alloc.rebalance_required:
                trade_size = alloc.target_position - alloc.actual_position
                
                if abs(trade_size) > 0:
                    trades.append({
                        "asset": asset,
                        "action": "BUY" if trade_size > 0 else "SELL",
                        "quantity": abs(trade_size),
                        "current_weight": alloc.current_weight,
                        "target_weight": alloc.target_weight
                    })
        
        return trades


# Singleton instance
_allocation_engine_instance: Optional[AllocationEngine] = None


def get_allocation_engine() -> AllocationEngine:
    """Get singleton instance of AllocationEngine."""
    global _allocation_engine_instance
    if _allocation_engine_instance is None:
        _allocation_engine_instance = AllocationEngine()
    return _allocation_engine_instance


if __name__ == "__main__":
    # Example usage
    engine = get_allocation_engine()
    
    # Simulate price updates
    engine.update_price("BTCUSDT", 50000.0)
    engine.update_price("ETHUSDT", 3000.0)
    engine.update_price("SOLUSDT", 100.0)
    engine.update_price("USDT", 1.0)
    
    # Get portfolio state
    state = engine.get_portfolio_state()
    print(f"Total Value: ${state.total_value:,.2f}")
    print(f"Leverage: {state.leverage:.2f}")
    
    for asset, alloc in state.allocations.items():
        print(f"{asset}: Target={alloc.target_weight:.2%}, Current={alloc.current_weight:.2%}")
    
    print("Allocation Engine module initialized successfully.")
