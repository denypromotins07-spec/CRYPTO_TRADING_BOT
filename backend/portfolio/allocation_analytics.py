#!/usr/bin/env python3
"""
Portfolio Allocation Analytics Module

Tracks tracking error, active share, and other portfolio quality metrics.
Provides comprehensive analytics for evaluating allocation decisions.
Essential for measuring performance against benchmarks and targets.

# Metrics Tracked:
- Tracking Error: Volatility of active returns vs benchmark
- Active Share: Portfolio deviation from benchmark weights
- Information Ratio: Active return / tracking error
- Concentration Metrics: Herfindahl index, top-N concentration
- Risk Decomposition: Factor exposures, sector bets
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from enum import Enum


class BenchmarkType(Enum):
    """Available benchmark types."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    CUSTOM = "custom"


@dataclass
class AllocationMetrics:
    """Complete set of portfolio allocation metrics."""
    # Tracking error (annualized)
    tracking_error: float
    # Active share (% deviation from benchmark)
    active_share: float
    # Information ratio
    information_ratio: float
    # Portfolio concentration (Herfindahl index)
    herfindahl_index: float
    # Top N asset concentration
    top_3_concentration: float
    # Effective number of assets
    effective_n_assets: float
    # Maximum single asset weight
    max_weight: float
    # Minimum single asset weight
    min_weight: float
    # Weight dispersion (std dev of weights)
    weight_dispersion: float
    # Turnover from previous period
    turnover: float


@dataclass
class RiskDecomposition:
    """Breakdown of portfolio risk sources."""
    # Systematic risk (market beta)
    systematic_risk: float
    # Idiosyncratic risk (residual)
    idiosyncratic_risk: float
    # Factor exposures (if applicable)
    factor_exposures: dict[str, float]
    # Asset contribution to total risk
    risk_contributions: dict[str, float]
    # Diversification ratio
    diversification_ratio: float


class PortfolioAnalytics:
    """
    Comprehensive portfolio allocation analytics.
    
    Computes metrics for evaluating portfolio construction quality,
    monitoring drift, and assessing risk-adjusted performance.
    
    # Attributes:
        asset_labels: List of asset tickers [BTC, SOL, ETH, USDT]
        benchmark_weights: Benchmark portfolio weights
        risk_free_rate: Annual risk-free rate
    """
    
    ASSET_LABELS: list[str] = ["BTC", "SOL", "ETH", "USDT"]
    NUM_ASSETS: int = 4
    
    def __init__(
        self,
        covariance: np.ndarray,
        benchmark_type: BenchmarkType = BenchmarkType.EQUAL_WEIGHT,
        custom_benchmark: Optional[dict[str, float]] = None,
        risk_free_rate: float = 0.03,
    ) -> None:
        """
        Initialize analytics engine.
        
        Args:
            covariance: 4x4 covariance matrix
            benchmark_type: Type of benchmark to use
            custom_benchmark: Custom benchmark weights (if benchmark_type=CUSTOM)
            risk_free_rate: Annual risk-free rate
        """
        if covariance.shape != (self.NUM_ASSETS, self.NUM_ASSETS):
            raise ValueError(f"Expected 4x4 covariance, got {covariance.shape}")
        
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.risk_free_rate = risk_free_rate
        
        # Set benchmark weights
        if benchmark_type == BenchmarkType.EQUAL_WEIGHT:
            self.benchmark_weights = np.ones(self.NUM_ASSETS) / self.NUM_ASSETS
        elif benchmark_type == BenchmarkType.MARKET_CAP:
            # Approximate crypto market cap weights
            self.benchmark_weights = np.array([0.50, 0.15, 0.30, 0.05])
        elif benchmark_type == BenchmarkType.CUSTOM:
            if custom_benchmark is None:
                raise ValueError("Custom benchmark requires custom_benchmark parameter")
            self.benchmark_weights = np.array([
                custom_benchmark.get(a, 0.25) for a in self.ASSET_LABELS
            ])
            total = self.benchmark_weights.sum()
            if total > 0:
                self.benchmark_weights /= total
        else:
            self.benchmark_weights = np.ones(self.NUM_ASSETS) / self.NUM_ASSETS
        
        # History for tracking calculations
        self._weight_history: list[np.ndarray] = []
        self._return_history: list[np.ndarray] = []
    
    def compute_metrics(
        self,
        current_weights: dict[str, float],
        historical_returns: Optional[np.ndarray] = None,
    ) -> AllocationMetrics:
        """
        Compute complete set of allocation metrics.
        
        Args:
            current_weights: Current portfolio weights by asset
            historical_returns: Optional T x N array of historical returns
        
        Returns:
            AllocationMetrics with all computed values
        """
        # Convert to array
        weights = np.array([current_weights.get(a, 0.0) for a in self.ASSET_LABELS])
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Track history
        self._weight_history.append(weights.copy())
        if len(self._weight_history) > 252:  # Keep 1 year
            self._weight_history = self._weight_history[-252:]
        
        # Compute individual metrics
        tracking_error = self._compute_tracking_error(weights, historical_returns)
        active_share = self._compute_active_share(weights)
        info_ratio = self._compute_information_ratio(weights, historical_returns)
        herfindahl = self._compute_herfindahl(weights)
        top_3_conc = self._compute_top_n_concentration(weights, n=3)
        eff_n = 1.0 / herfindahl if herfindahl > 0 else self.NUM_ASSETS
        max_w = weights.max()
        min_w = weights.min()
        dispersion = weights.std()
        turnover = self._compute_turnover(weights)
        
        return AllocationMetrics(
            tracking_error=tracking_error,
            active_share=active_share,
            information_ratio=info_ratio,
            herfindahl_index=herfindahl,
            top_3_concentration=top_3_conc,
            effective_n_assets=eff_n,
            max_weight=max_w,
            min_weight=min_w,
            weight_dispersion=dispersion,
            turnover=turnover,
        )
    
    def _compute_tracking_error(
        self,
        weights: np.ndarray,
        returns: Optional[np.ndarray],
    ) -> float:
        """Compute annualized tracking error vs benchmark."""
        if returns is None or len(returns) < 10:
            # Estimate from weight difference and covariance
            active_weights = weights - self.benchmark_weights
            variance = np.dot(active_weights, np.dot(self.covariance, active_weights))
            return np.sqrt(variance * 252)  # Annualize
        
        # Compute portfolio and benchmark returns
        port_returns = np.dot(returns, weights)
        bench_returns = np.dot(returns, self.benchmark_weights)
        
        # Active returns
        active_returns = port_returns - bench_returns
        
        # Tracking error = std(active returns) * sqrt(252)
        return float(np.std(active_returns, ddof=1) * np.sqrt(252))
    
    def _compute_active_share(self, weights: np.ndarray) -> float:
        """
        Compute active share.
        
        Active Share = 0.5 * sum(|w_portfolio - w_benchmark|)
        Measures percentage of portfolio that differs from benchmark.
        """
        diff = np.abs(weights - self.benchmark_weights)
        return 0.5 * diff.sum()
    
    def _compute_information_ratio(
        self,
        weights: np.ndarray,
        returns: Optional[np.ndarray],
    ) -> float:
        """Compute information ratio (active return / tracking error)."""
        if returns is None or len(returns) < 10:
            # Estimate from expected returns
            # Simplified: assume some alpha
            return 0.0
        
        port_returns = np.dot(returns, weights)
        bench_returns = np.dot(returns, self.benchmark_weights)
        active_returns = port_returns - bench_returns
        
        active_mean = np.mean(active_returns)
        active_std = np.std(active_returns, ddof=1)
        
        if active_std < 1e-10:
            return 0.0
        
        # Annualize
        return (active_mean * 252) / (active_std * np.sqrt(252))
    
    def _compute_herfindahl(self, weights: np.ndarray) -> float:
        """
        Compute Herfindahl-Hirschman Index (HHI).
        
        HHI = sum(w_i^2)
        Ranges from 1/N (perfectly diversified) to 1 (single asset)
        """
        return float(np.sum(weights ** 2))
    
    def _compute_top_n_concentration(self, weights: np.ndarray, n: int = 3) -> float:
        """Compute concentration of top N assets."""
        sorted_weights = np.sort(weights)[::-1]  # Descending
        return float(sorted_weights[:n].sum())
    
    def _compute_turnover(self, current_weights: np.ndarray) -> float:
        """
        Compute portfolio turnover from last period.
        
        Turnover = 0.5 * sum(|w_current - w_previous|)
        """
        if len(self._weight_history) < 2:
            return 0.0
        
        previous = self._weight_history[-2]
        diff = np.abs(current_weights - previous)
        return 0.5 * diff.sum()
    
    def compute_risk_decomposition(
        self,
        weights: dict[str, float],
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk into components.
        
        Args:
            weights: Portfolio weights by asset
        
        Returns:
            RiskDecomposition with detailed breakdown
        """
        w = np.array([weights.get(a, 0.0) for a in self.ASSET_LABELS])
        if w.sum() > 0:
            w = w / w.sum()
        
        # Portfolio variance
        port_var = np.dot(w, np.dot(self.covariance, w))
        port_vol = np.sqrt(port_var) if port_var > 0 else 0.0
        
        # Marginal risk contributions
        marginal_risk = np.dot(self.covariance, w) / port_vol if port_vol > 0 else np.zeros(self.NUM_ASSETS)
        
        # Component risk contributions
        risk_contrib = w * marginal_risk
        risk_contrib_pct = risk_contrib / port_vol if port_vol > 0 else np.zeros(self.NUM_ASSETS)
        
        risk_contrib_dict = dict(zip(self.ASSET_LABELS, risk_contrib_pct.tolist()))
        
        # Diversification ratio
        weighted_avg_vol = np.dot(w, np.sqrt(np.diag(self.covariance)))
        div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1.0
        
        # Simplified factor decomposition
        # In production, would use actual factor model
        systematic = port_vol * 0.7  # Assume 70% systematic
        idiosyncratic = port_vol * 0.3  # 30% idiosyncratic
        
        return RiskDecomposition(
            systematic_risk=systematic,
            idiosyncratic_risk=idiosyncratic,
            factor_exposures={"market": 1.0, "size": 0.0, "momentum": 0.0},
            risk_contributions=risk_contrib_dict,
            diversification_ratio=div_ratio,
        )
    
    def get_allocation_quality_score(self, weights: dict[str, float]) -> float:
        """
        Compute overall allocation quality score (0-100).
        
        Combines multiple metrics into single quality indicator:
        - Diversification (lower HHI = better)
        - Active share (moderate = better, too high = risky)
        - Concentration (lower top-N = better)
        - Turnover (lower = better, less trading cost)
        """
        metrics = self.compute_metrics(weights)
        
        # Diversification score (0-25 points)
        # Perfect diversification = 1/N, worst = 1
        perfect_hhi = 1.0 / self.NUM_ASSETS
        div_score = 25.0 * (1.0 - (metrics.herfindahl_index - perfect_hhi) / (1.0 - perfect_hhi))
        div_score = max(0, min(25, div_score))
        
        # Active share score (0-25 points)
        # Optimal around 20-40%
        optimal_active_share = 0.30
        as_deviation = abs(metrics.active_share - optimal_active_share)
        as_score = 25.0 * (1.0 - as_deviation / 0.5)
        as_score = max(0, min(25, as_score))
        
        # Concentration score (0-25 points)
        # Lower top-3 concentration = better
        conc_score = 25.0 * (1.0 - (metrics.top_3_concentration - 0.25) / 0.75)
        conc_score = max(0, min(25, conc_score))
        
        # Turnover score (0-25 points)
        # Lower turnover = better
        turnover_score = 25.0 * (1.0 - min(1.0, metrics.turnover * 4))
        turnover_score = max(0, min(25, turnover_score))
        
        return div_score + as_score + conc_score + turnover_score
    
    def compare_to_benchmark(
        self,
        weights: dict[str, float],
    ) -> dict:
        """Compare portfolio to benchmark."""
        w = np.array([weights.get(a, 0.0) for a in self.ASSET_LABELS])
        if w.sum() > 0:
            w = w / w.sum()
        
        active_weights = w - self.benchmark_weights
        
        comparison = {
            "portfolio_weights": dict(zip(self.ASSET_LABELS, w.tolist())),
            "benchmark_weights": dict(zip(self.ASSET_LABELS, self.benchmark_weights.tolist())),
            "active_weights": dict(zip(self.ASSET_LABELS, active_weights.tolist())),
            "active_share": self._compute_active_share(w),
            "net_long": sum(max(0, aw) for aw in active_weights),
            "net_short": sum(min(0, aw) for aw in active_weights),
        }
        
        return comparison


def example_usage() -> None:
    """Demonstrate portfolio analytics."""
    # Sample covariance
    covariance = np.array([
        [0.04, 0.02, 0.025, 0.0001],
        [0.02, 0.09, 0.04, 0.0002],
        [0.025, 0.04, 0.05, 0.0001],
        [0.0001, 0.0002, 0.0001, 0.0001],
    ])
    
    analytics = PortfolioAnalytics(covariance, BenchmarkType.EQUAL_WEIGHT)
    
    # Example portfolio
    weights = {"BTC": 0.35, "SOL": 0.20, "ETH": 0.30, "USDT": 0.15}
    
    # Compute metrics
    metrics = analytics.compute_metrics(weights)
    
    print("Allocation Metrics:")
    print(f"  Tracking Error: {metrics.tracking_error:.2%}")
    print(f"  Active Share: {metrics.active_share:.1%}")
    print(f"  Herfindahl Index: {metrics.herfindahl_index:.3f}")
    print(f"  Effective N Assets: {metrics.effective_n_assets:.2f}")
    print(f"  Top 3 Concentration: {metrics.top_3_concentration:.1%}")
    print(f"  Turnover: {metrics.turnover:.1%}")
    
    # Quality score
    quality = analytics.get_allocation_quality_score(weights)
    print(f"\nAllocation Quality Score: {quality:.1f}/100")
    
    # Risk decomposition
    risk_decomp = analytics.compute_risk_decomposition(weights)
    print(f"\nRisk Decomposition:")
    print(f"  Diversification Ratio: {risk_decomp.diversification_ratio:.2f}")
    print(f"  Risk Contributions:")
    for asset, contrib in risk_decomp.risk_contributions.items():
        print(f"    {asset}: {contrib:.1%}")


if __name__ == "__main__":
    example_usage()
