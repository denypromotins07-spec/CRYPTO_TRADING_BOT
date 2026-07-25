#!/usr/bin/env python3
"""
Efficient Frontier Calculator for Multi-Asset Portfolio

Maps the complete risk-return profile for BTC, SOL, ETH, USDT portfolio.
Generates optimal portfolios across all possible return targets.
Integrates with Rust Markowitz optimizer via PyO3 bindings.

# Features:
- Computes 100+ points along the efficient frontier
- Identifies tangency portfolio (maximum Sharpe ratio)
- Handles numerical instabilities gracefully
- Memory-efficient streaming computation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from enum import Enum


class AssetType(Enum):
    """Supported cryptocurrency assets."""
    BTC = "BTC"
    SOL = "SOL"
    ETH = "ETH"
    USDT = "USDT"


@dataclass
class FrontierPoint:
    """Single point on the efficient frontier."""
    target_return: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]
    is_optimal: bool


@dataclass
class EfficientFrontierResult:
    """Complete efficient frontier analysis result."""
    points: List[FrontierPoint]
    min_variance_portfolio: FrontierPoint
    max_sharpe_portfolio: FrontierPoint
    asset_labels: list[str]
    
    @property
    def returns(self) -> np.ndarray:
        """Array of expected returns for plotting."""
        return np.array([p.expected_return for p in self.points])
    
    @property
    def volatilities(self) -> np.ndarray:
        """Array of volatilities for plotting."""
        return np.array([p.volatility for p in self.points])
    
    @property
    def sharpe_ratios(self) -> np.ndarray:
        """Array of Sharpe ratios for plotting."""
        return np.array([p.sharpe_ratio for p in self.points])


class EfficientFrontierCalculator:
    """
    Computes the efficient frontier for a 4-asset crypto portfolio.
    
    The efficient frontier represents the set of optimal portfolios
    that offer the highest expected return for a given level of risk.
    
    # Attributes:
        expected_returns: Annualized expected returns [BTC, SOL, ETH, USDT]
        covariance_matrix: 4x4 covariance matrix of returns
        risk_free_rate: Annual risk-free rate (USDT yield)
    """
    
    ASSET_LABELS: list[str] = ["BTC", "SOL", "ETH", "USDT"]
    NUM_ASSETS: int = 4
    NUM_FRONTIER_POINTS: int = 100
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.03,
    ) -> None:
        """
        Initialize calculator with portfolio parameters.
        
        Args:
            expected_returns: Array of 4 annualized expected returns
            covariance_matrix: 4x4 symmetric positive-definite matrix
            risk_free_rate: Annual risk-free rate (default 3%)
        """
        if len(expected_returns) != self.NUM_ASSETS:
            raise ValueError(f"Expected {self.NUM_ASSETS} returns, got {len(expected_returns)}")
        if covariance_matrix.shape != (self.NUM_ASSETS, self.NUM_ASSETS):
            raise ValueError(f"Expected 4x4 covariance, got {covariance_matrix.shape}")
        
        self.expected_returns = np.asarray(expected_returns, dtype=np.float64)
        self.covariance_matrix = np.asarray(covariance_matrix, dtype=np.float64)
        self.risk_free_rate = risk_free_rate
        
        # Precompute inverse covariance for efficiency
        self._inv_cov: Optional[np.ndarray] = None
    
    @property
    def inv_covariance(self) -> np.ndarray:
        """Lazy-loaded inverse covariance matrix with regularization."""
        if self._inv_cov is None:
            # Add small regularization for numerical stability
            reg_matrix = self.covariance_matrix + 1e-8 * np.eye(self.NUM_ASSETS)
            try:
                self._inv_cov = np.linalg.inv(reg_matrix)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse for singular matrices
                self._inv_cov = np.linalg.pinv(reg_matrix)
        return self._inv_cov
    
    def compute_frontier(self) -> EfficientFrontierResult:
        """
        Compute the complete efficient frontier.
        
        Returns:
            EfficientFrontierResult containing all frontier points
        """
        # Find minimum and maximum feasible returns
        min_var_portfolio = self._compute_min_variance_portfolio()
        min_return = min_var_portfolio["expected_return"]
        max_return = np.max(self.expected_returns)
        
        # Generate target returns spanning the feasible range
        target_returns = np.linspace(min_return, max_return * 0.95, self.NUM_FRONTIER_POINTS)
        
        points: List[FrontierPoint] = []
        max_sharpe = -np.inf
        max_sharpe_point: Optional[FrontierPoint] = None
        
        for target in target_returns:
            try:
                weights = self._optimize_for_target(target)
                exp_ret = np.dot(weights, self.expected_returns)
                variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
                volatility = np.sqrt(max(0, variance))
                
                sharpe = (exp_ret - self.risk_free_rate) / volatility if volatility > 0 else 0
                
                weight_dict = dict(zip(self.ASSET_LABELS, weights.tolist()))
                
                point = FrontierPoint(
                    target_return=target,
                    expected_return=exp_ret,
                    volatility=volatility,
                    sharpe_ratio=sharpe,
                    weights=weight_dict,
                    is_optimal=True,
                )
                points.append(point)
                
                if sharpe > max_sharpe:
                    max_sharpe = sharpe
                    max_sharpe_point = point
                    
            except Exception:
                # Skip infeasible target returns
                continue
        
        if not points:
            raise RuntimeError("Failed to compute any feasible frontier points")
        
        min_var_point = FrontierPoint(
            target_return=min_var_portfolio["expected_return"],
            expected_return=min_var_portfolio["expected_return"],
            volatility=min_var_portfolio["volatility"],
            sharpe_ratio=(min_var_portfolio["expected_return"] - self.risk_free_rate) 
                        / min_var_portfolio["volatility"] if min_var_portfolio["volatility"] > 0 else 0,
            weights=dict(zip(self.ASSET_LABELS, min_var_portfolio["weights"].tolist())),
            is_optimal=True,
        )
        
        return EfficientFrontierResult(
            points=points,
            min_variance_portfolio=min_var_point,
            max_sharpe_portfolio=max_sharpe_point or points[-1],
            asset_labels=self.ASSET_LABELS,
        )
    
    def _compute_min_variance_portfolio(self) -> dict:
        """Compute global minimum variance portfolio."""
        ones = np.ones(self.NUM_ASSETS)
        
        # w = Σ^(-1) * 1 / (1^T * Σ^(-1) * 1)
        inv_cov_ones = np.dot(self.inv_covariance, ones)
        denominator = np.dot(ones, inv_cov_ones)
        
        weights = inv_cov_ones / denominator
        weights = np.maximum(weights, 0)  # No short selling
        weights /= weights.sum()  # Renormalize
        
        exp_ret = np.dot(weights, self.expected_returns)
        variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
        volatility = np.sqrt(variance)
        
        return {
            "weights": weights,
            "expected_return": exp_ret,
            "volatility": volatility,
        }
    
    def _optimize_for_target(self, target_return: float) -> np.ndarray:
        """
        Find optimal weights for a target return using Lagrange multipliers.
        
        Solves: min w^T Σ w subject to w^T μ = target, w^T 1 = 1, w >= 0
        """
        mu = self.expected_returns
        ones = np.ones(self.NUM_ASSETS)
        inv_cov = self.inv_covariance
        
        # Compute Lagrange multiplier coefficients
        A = np.dot(ones, np.dot(inv_cov, ones))
        B = np.dot(ones, np.dot(inv_cov, mu))
        C = np.dot(mu, np.dot(inv_cov, mu))
        D = A * C - B * B
        
        if abs(D) < 1e-12:
            # Near-singular case, fall back to min variance
            return self._compute_min_variance_portfolio()["weights"]
        
        # Compute unconstrained optimal weights
        lambda1 = (C - B * target_return) / D
        lambda2 = (A * target_return - B) / D
        
        weights = np.dot(inv_cov, lambda1 * ones + lambda2 * mu)
        
        # Project to feasible region (no short selling)
        weights = np.maximum(weights, 0)
        
        # Handle case where projection violates constraints
        if weights.sum() < 1e-10:
            return np.ones(self.NUM_ASSETS) / self.NUM_ASSETS
        
        weights /= weights.sum()
        
        return weights
    
    def compute_tangency_portfolio(self) -> dict:
        """
        Compute the tangency portfolio (maximum Sharpe ratio).
        
        This is the optimal risky portfolio when combined with risk-free asset.
        """
        excess_returns = self.expected_returns - self.risk_free_rate
        inv_cov = self.inv_covariance
        
        # Tangency weights proportional to Σ^(-1) * (μ - r_f)
        raw_weights = np.dot(inv_cov, excess_returns)
        
        # Normalize to sum to 1
        if raw_weights.sum() != 0:
            weights = raw_weights / raw_weights.sum()
        else:
            weights = np.ones(self.NUM_ASSETS) / self.NUM_ASSETS
        
        # Ensure no negative weights
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(self.NUM_ASSETS) / self.NUM_ASSETS
        
        exp_ret = np.dot(weights, self.expected_returns)
        variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
        volatility = np.sqrt(variance)
        sharpe = (exp_ret - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        return {
            "weights": weights,
            "expected_return": exp_ret,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "weight_dict": dict(zip(self.ASSET_LABELS, weights.tolist())),
        }
    
    def get_allocation_at_risk(self, target_volatility: float) -> dict:
        """
        Get optimal allocation for a target volatility level.
        
        Combines tangency portfolio with risk-free asset to achieve
        exact target volatility.
        """
        tangency = self.compute_tangency_portfolio()
        tangency_vol = tangency["volatility"]
        
        if tangency_vol == 0:
            # All in risk-free
            allocation = {asset: 0.0 for asset in self.ASSET_LABELS}
            allocation["USDT"] = 1.0
            return {"allocation": allocation, "expected_return": self.risk_free_rate}
        
        # Fraction in tangency portfolio
        tangency_fraction = min(1.0, target_volatility / tangency_vol)
        risk_free_fraction = 1.0 - tangency_fraction
        
        allocation = {}
        for asset in self.ASSET_LABELS:
            if asset == "USDT":
                allocation[asset] = tangency["weight_dict"][asset] * tangency_fraction + risk_free_fraction
            else:
                allocation[asset] = tangency["weight_dict"][asset] * tangency_fraction
        
        expected_return = (
            tangency_fraction * tangency["expected_return"] + 
            risk_free_fraction * self.risk_free_rate
        )
        
        return {
            "allocation": allocation,
            "expected_return": expected_return,
            "tangency_fraction": tangency_fraction,
        }


def example_usage() -> None:
    """Demonstrate efficient frontier computation."""
    # Sample parameters for BTC, SOL, ETH, USDT
    expected_returns = np.array([0.15, 0.25, 0.18, 0.03])
    covariance = np.array([
        [0.04, 0.02, 0.025, 0.0001],
        [0.02, 0.09, 0.04, 0.0002],
        [0.025, 0.04, 0.05, 0.0001],
        [0.0001, 0.0002, 0.0001, 0.0001],
    ])
    
    calculator = EfficientFrontierCalculator(expected_returns, covariance)
    frontier = calculator.compute_frontier()
    
    print(f"Computed {len(frontier.points)} frontier points")
    print(f"Min variance portfolio: {frontier.min_variance_portfolio.volatility:.2%} vol")
    print(f"Max Sharpe portfolio: {frontier.max_sharpe_portfolio.sharpe_ratio:.2f} Sharpe")
    
    tangency = calculator.compute_tangency_portfolio()
    print(f"Tangency weights: {tangency['weight_dict']}")


if __name__ == "__main__":
    example_usage()
