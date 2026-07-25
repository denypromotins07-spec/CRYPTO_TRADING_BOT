#!/usr/bin/env python3
"""
Capital Reserve Manager for Crypto Trading Bot

Maintains stablecoin reserves for sudden market opportunities.
Dynamically adjusts reserve levels based on volatility and opportunity detection.
Critical for capturing flash crashes and arbitrage opportunities.

# Features:
- Dynamic reserve allocation based on market regime
- Opportunity-triggered reserve deployment
- Minimum reserve floor for risk management
- Integration with portfolio allocation systems
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


class MarketRegime(Enum):
    """Market condition classification."""
    CALM = "calm"
    VOLATILE = "volatile"
    CRASH = "crash"
    RALLY = "rally"
    UNCERTAIN = "uncertain"


@dataclass
class ReserveConfig:
    """Configuration for capital reserve management."""
    # Minimum reserve percentage (never go below this)
    min_reserve_pct: float = 0.10
    # Maximum reserve percentage
    max_reserve_pct: float = 0.50
    # Target reserve in calm markets
    target_calm_reserve: float = 0.20
    # Target reserve in volatile markets
    target_volatile_reserve: float = 0.35
    # Emergency reserve during crash conditions
    crash_reserve: float = 0.45
    # Volatility threshold for regime change
    vol_threshold_low: float = 0.02  # Daily vol
    vol_threshold_high: float = 0.05
    # Opportunity trigger threshold (price drop %)
    opportunity_threshold: float = 0.05
    # Max single deployment from reserves
    max_deployment_pct: float = 0.15


@dataclass
class ReserveState:
    """Current state of capital reserves."""
    # Total portfolio value in USDT
    total_portfolio_value: float
    # Current reserve amount in USDT
    reserve_amount: float
    # Reserve as percentage of portfolio
    reserve_pct: float
    # Invested amount (non-reserve)
    invested_amount: float
    # Available for immediate deployment
    available_for_deployment: float
    # Current market regime
    market_regime: MarketRegime
    # Recent volatility estimate
    current_volatility: float
    # Pending deployments (reserved but committed)
    pending_deployments: float


@dataclass
class DeploymentResult:
    """Result of a reserve deployment operation."""
    # Amount deployed
    deployed_amount: float
    # Remaining reserve
    remaining_reserve: float
    # Target asset
    asset: str
    # Whether deployment was partial (capped by limits)
    was_capped: bool
    # Reason if capped
    cap_reason: Optional[str]


class CapitalReserveManager:
    """
    Manages stablecoin reserves for opportunistic trading.
    
    Maintains dynamic reserve levels based on:
    - Market volatility regime
    - Detected opportunities (flash crashes)
    - Portfolio drawdown status
    - Risk parameters
    
    Ensures the bot always has dry powder for high-conviction
    opportunities while maintaining adequate risk buffers.
    """
    
    def __init__(self, config: Optional[ReserveConfig] = None) -> None:
        """
        Initialize reserve manager.
        
        Args:
            config: Reserve configuration (uses defaults if None)
        """
        self.config = config or ReserveConfig()
        
        # State tracking
        self._total_value: float = 100_000.0  # Initial portfolio
        self._reserve_balance: float = self.config.target_calm_reserve * self._total_value
        self._pending_deployments: float = 0.0
        
        # Volatility tracking
        self._recent_returns: list[float] = []
        self._max_returns_history = 252  # ~1 year of daily returns
        
        # Opportunity tracking
        self._detected_opportunities: list[dict] = []
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update total portfolio value and recalculate reserves."""
        if new_value <= 0:
            raise ValueError("Portfolio value must be positive")
        
        old_value = self._total_value
        self._total_value = new_value
        
        # Adjust reserve proportionally if no rebalance needed
        if old_value > 0:
            growth_factor = new_value / old_value
            self._reserve_balance *= growth_factor
    
    def update_volatility(self, returns: list[float]) -> None:
        """
        Update volatility estimate from recent returns.
        
        Args:
            returns: List of recent returns (daily or hourly)
        """
        self._recent_returns.extend(returns)
        
        # Keep only recent history
        if len(self._recent_returns) > self._max_returns_history:
            self._recent_returns = self._recent_returns[-self._max_returns_history:]
    
    def _compute_volatility(self) -> float:
        """Compute annualized volatility from recent returns."""
        if len(self._recent_returns) < 10:
            return 0.02  # Default low vol
        
        std = np.std(self._recent_returns, ddof=1)
        
        # Annualize (assuming daily returns)
        return std * np.sqrt(252)
    
    def _determine_regime(self, volatility: float) -> MarketRegime:
        """Determine market regime based on volatility."""
        if volatility < self.config.vol_threshold_low:
            return MarketRegime.CALM
        elif volatility < self.config.vol_threshold_high:
            return MarketRegime.VOLATILE
        else:
            # High volatility could be crash or rally
            if len(self._recent_returns) >= 5:
                recent_mean = np.mean(self._recent_returns[-5:])
                if recent_mean < -0.03:  # 3% average daily decline
                    return MarketRegime.CRASH
                elif recent_mean > 0.03:
                    return MarketRegime.RALLY
            return MarketRegime.UNCERTAIN
    
    def _get_target_reserve(self, regime: MarketRegime) -> float:
        """Get target reserve percentage for current regime."""
        targets = {
            MarketRegime.CALM: self.config.target_calm_reserve,
            MarketRegime.VOLATILE: self.config.target_volatile_reserve,
            MarketRegime.CRASH: self.config.crash_reserve,
            MarketRegime.RALLY: self.config.target_calm_reserve,  # Deploy in rallies
            MarketRegime.UNCERTAIN: self.config.target_volatile_reserve,
        }
        return targets.get(regime, self.config.target_volatile_reserve)
    
    def get_state(self) -> ReserveState:
        """Get current reserve state."""
        volatility = self._compute_volatility()
        regime = self._determine_regime(volatility)
        
        available = self._reserve_balance - self._pending_deployments
        available = max(0.0, available)
        
        return ReserveState(
            total_portfolio_value=self._total_value,
            reserve_amount=self._reserve_balance,
            reserve_pct=self._reserve_balance / self._total_value if self._total_value > 0 else 0,
            invested_amount=self._total_value - self._reserve_balance,
            available_for_deployment=available,
            market_regime=regime,
            current_volatility=volatility,
            pending_deployments=self._pending_deployments,
        )
    
    def compute_target_allocation(self) -> dict[str, float]:
        """
        Compute target reserve allocation based on current conditions.
        
        Returns:
            Dictionary with reserve and investment targets
        """
        state = self.get_state()
        target_pct = self._get_target_reserve(state.market_regime)
        
        # Clamp to min/max
        target_pct = max(self.config.min_reserve_pct, 
                        min(self.config.max_reserve_pct, target_pct))
        
        target_reserve = target_pct * self._total_value
        target_invested = self._total_value - target_reserve
        
        return {
            'target_reserve': target_reserve,
            'target_invested': target_invested,
            'target_reserve_pct': target_pct,
            'current_reserve_pct': state.reserve_pct,
            'rebalance_needed': abs(state.reserve_pct - target_pct) > 0.02,
            'rebalance_amount': target_reserve - self._reserve_balance,
        }
    
    def deploy_reserves(
        self,
        asset: str,
        amount: float,
        priority: float = 1.0,
    ) -> DeploymentResult:
        """
        Deploy reserves to capture an opportunity.
        
        Args:
            asset: Target asset for deployment
            amount: Amount to deploy in USDT
            priority: Priority level (higher = more likely to exceed limits)
        
        Returns:
            DeploymentResult with execution details
        """
        state = self.get_state()
        
        # Check minimum reserve floor
        min_reserve = self.config.min_reserve_pct * self._total_value
        max_deployable = self._reserve_balance - min_reserve - self._pending_deployments
        max_deployable = max(0.0, max_deployable)
        
        # Apply per-deployment limit
        max_single = self.config.max_deployment_pct * self._total_value
        
        # Determine actual deployment amount
        requested = amount
        actual = min(requested, max_deployable, max_single)
        
        was_capped = False
        cap_reason = None
        
        if actual < requested:
            was_capped = True
            if actual >= max_deployable:
                cap_reason = "Minimum reserve floor reached"
            elif actual >= max_single:
                cap_reason = "Single deployment limit reached"
        
        # Record pending deployment
        self._pending_deployments += actual
        self._reserve_balance -= actual
        
        # Track opportunity
        self._detected_opportunities.append({
            'asset': asset,
            'requested': requested,
            'deployed': actual,
            'priority': priority,
            'regime': state.market_regime.value,
        })
        
        return DeploymentResult(
            deployed_amount=actual,
            remaining_reserve=self._reserve_balance,
            asset=asset,
            was_capped=was_capped,
            cap_reason=cap_reason,
        )
    
    def confirm_deployment(self, deployment_id: int) -> None:
        """Confirm a pending deployment (remove from pending)."""
        # In production, would track individual deployments
        # For now, reduce pending by average amount
        if self._pending_deployments > 0:
            self._pending_deployments = max(0, self._pending_deployments - 1000)
    
    def replenish_reserves(self, amount: float) -> None:
        """
        Add funds to reserves (from profits or redemptions).
        
        Args:
            amount: Amount to add to reserves
        """
        if amount < 0:
            raise ValueError("Replenishment amount must be positive")
        
        self._reserve_balance += amount
        self._total_value += amount
    
    def detect_opportunity(
        self,
        asset: str,
        price_drop: float,
        volume_spike: float = 1.0,
    ) -> Optional[dict]:
        """
        Detect trading opportunity based on price action.
        
        Args:
            asset: Asset ticker
            price_drop: Price drop percentage (positive = drop)
            volume_spike: Volume multiplier vs average
        
        Returns:
            Opportunity dict if detected, None otherwise
        """
        if price_drop < self.config.opportunity_threshold:
            return None
        
        # Score the opportunity
        score = price_drop * volume_spike
        
        # Recommended deployment based on opportunity severity
        recommended = min(
            price_drop * self._total_value * 0.5,  # Scale with drop
            self.config.max_deployment_pct * self._total_value,
        )
        
        opportunity = {
            'asset': asset,
            'price_drop': price_drop,
            'volume_spike': volume_spike,
            'score': score,
            'recommended_deployment': recommended,
            'timestamp': None,  # Would set actual timestamp
        }
        
        self._detected_opportunities.append(opportunity)
        return opportunity
    
    def get_reserve_efficiency_metrics(self) -> dict:
        """Compute metrics on reserve utilization efficiency."""
        state = self.get_state()
        
        # Historical deployment stats
        total_deployed = sum(o.get('deployed', 0) for o in self._detected_opportunities)
        n_opportunities = len([o for o in self._detected_opportunities if 'deployed' in o])
        
        avg_deployment = total_deployed / n_opportunities if n_opportunities > 0 else 0
        
        # Reserve turnover
        reserve_turnover = total_deployed / self._reserve_balance if self._reserve_balance > 0 else 0
        
        return {
            'current_reserve_pct': state.reserve_pct,
            'target_reserve_pct': self._get_target_reserve(state.market_regime),
            'available_for_deployment': state.available_for_deployment,
            'total_opportunities_detected': len(self._detected_opportunities),
            'opportunities_executed': n_opportunities,
            'total_deployed': total_deployed,
            'average_deployment': avg_deployment,
            'reserve_turnover': reserve_turnover,
            'market_regime': state.market_regime.value,
        }


def example_usage() -> None:
    """Demonstrate reserve management."""
    config = ReserveConfig(
        min_reserve_pct=0.10,
        max_reserve_pct=0.50,
        target_calm_reserve=0.20,
        opportunity_threshold=0.05,
    )
    
    manager = CapitalReserveManager(config)
    
    # Simulate some returns
    returns = [0.01, -0.02, 0.015, -0.03, 0.02, -0.01]
    manager.update_volatility(returns)
    manager.update_portfolio_value(100_000)
    
    # Get current state
    state = manager.get_state()
    print(f"Market Regime: {state.market_regime.value}")
    print(f"Current Volatility: {state.current_volatility:.2%}")
    print(f"Reserve: ${state.reserve_amount:,.0f} ({state.reserve_pct:.1%})")
    print(f"Available for Deployment: ${state.available_for_deployment:,.0f}")
    
    # Detect opportunity
    opp = manager.detect_opportunity("BTC", price_drop=0.08, volume_spike=2.5)
    if opp:
        print(f"\nOpportunity detected: {opp['asset']} down {opp['price_drop']:.1%}")
        print(f"Recommended deployment: ${opp['recommended_deployment']:,.0f}")
        
        # Deploy reserves
        result = manager.deploy_reserves("BTC", opp['recommended_deployment'])
        print(f"Deployed: ${result.deployed_amount:,.0f}")
        if result.was_capped:
            print(f"Capped: {result.cap_reason}")
    
    # Get efficiency metrics
    metrics = manager.get_reserve_efficiency_metrics()
    print(f"\nReserve Efficiency:")
    print(f"  Opportunities detected: {metrics['total_opportunities_detected']}")
    print(f"  Reserve turnover: {metrics['reserve_turnover']:.2f}x")


if __name__ == "__main__":
    example_usage()
