#!/usr/bin/env python3
"""
Black-Litterman Portfolio Allocation Model

Integrates market equilibrium returns with the bot's alpha views.
Combines CAPM-implied returns with subjective forecasts for optimal allocation.
Essential for incorporating trading signals into portfolio construction.

# Features:
- Reverse optimization from market cap weights to implied returns
- Bayesian blending of equilibrium and view-based returns
- Confidence-weighted view incorporation
- Handles partial or uncertain views gracefully
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from enum import Enum


class ViewType(Enum):
    """Type of alpha view."""
    ABSOLUTE = "absolute"  # Direct return forecast
    RELATIVE = "relative"  # Outperformance vs another asset


@dataclass
class AlphaView:
    """Represents a single alpha view on assets."""
    view_type: ViewType
    assets: list[str]  # Asset tickers involved
    expected_return: float  # Expected return or spread
    confidence: float  # Confidence level (0 to 1)
    
    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.view_type == ViewType.RELATIVE and len(self.assets) != 2:
            raise ValueError("Relative views require exactly 2 assets")


@dataclass
class BlackLittermanResult:
    """Result of Black-Litterman optimization."""
    # Posterior expected returns
    posterior_returns: dict[str, float]
    # Posterior covariance matrix
    posterior_covariance: np.ndarray
    # Optimal portfolio weights
    optimal_weights: dict[str, float]
    # Implied equilibrium returns (prior)
    equilibrium_returns: dict[str, float]
    # Blended returns after incorporating views
    blended_returns: dict[str, float]
    # Number of views incorporated
    n_views: int
    # Average confidence across views
    avg_confidence: float


class BlackLittermanModel:
    """
    Black-Litterman asset allocation model.
    
    Combines market equilibrium (reverse-optimized from cap weights)
    with subjective views to produce posterior return estimates.
    
    The model solves:
    E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)Π + P'Ω^(-1)Q]
    
    where:
    - τ is the scaling factor (uncertainty in prior)
    - Σ is the covariance matrix
    - P is the pick matrix (view definitions)
    - Ω is the view uncertainty matrix
    - Π is the implied equilibrium returns
    - Q is the view returns vector
    
    # Attributes:
        asset_labels: List of asset tickers
        covariance: Covariance matrix of returns
        risk_aversion: Risk aversion coefficient (default 2.5)
        tau: Scaling factor for prior uncertainty (default 0.05)
    """
    
    ASSET_LABELS: list[str] = ["BTC", "SOL", "ETH", "USDT"]
    NUM_ASSETS: int = 4
    
    def __init__(
        self,
        covariance: np.ndarray,
        market_cap_weights: Optional[dict[str, float]] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> None:
        """
        Initialize Black-Litterman model.
        
        Args:
            covariance: 4x4 covariance matrix
            market_cap_weights: Market cap weights (uses equal weight if None)
            risk_aversion: Risk aversion coefficient
            tau: Prior uncertainty scaling factor
        """
        if covariance.shape != (self.NUM_ASSETS, self.NUM_ASSETS):
            raise ValueError(f"Expected 4x4 covariance, got {covariance.shape}")
        
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # Default to equal weights if no market cap data
        if market_cap_weights is None:
            self.market_weights = np.ones(self.NUM_ASSETS) / self.NUM_ASSETS
        else:
            self.market_weights = np.array([market_cap_weights.get(a, 0.25) for a in self.ASSET_LABELS])
            # Normalize
            total = self.market_weights.sum()
            if total > 0:
                self.market_weights /= total
            else:
                self.market_weights = np.ones(self.NUM_ASSETS) / self.NUM_ASSETS
        
        # Compute implied equilibrium returns
        self.equilibrium_returns = self._compute_equilibrium_returns()
        
        # Regularized inverse covariance
        self._inv_cov: Optional[np.ndarray] = None
    
    @property
    def inv_covariance(self) -> np.ndarray:
        """Lazy-loaded inverse covariance with regularization."""
        if self._inv_cov is None:
            reg_cov = self.covariance + 1e-8 * np.eye(self.NUM_ASSETS)
            try:
                self._inv_cov = np.linalg.inv(reg_cov)
            except np.linalg.LinAlgError:
                self._inv_cov = np.linalg.pinv(reg_cov)
        return self._inv_cov
    
    def _compute_equilibrium_returns(self) -> np.ndarray:
        """
        Compute implied equilibrium returns using reverse optimization.
        
        Π = δ * Σ * w_mkt
        
        This gives the returns that would make the market portfolio optimal.
        """
        return self.risk_aversion * np.dot(self.covariance, self.market_weights)
    
    def compute_allocation(
        self,
        views: list[AlphaView],
        optimize: bool = True,
    ) -> BlackLittermanResult:
        """
        Compute optimal allocation given alpha views.
        
        Args:
            views: List of alpha views to incorporate
            optimize: Whether to compute optimal weights
            
        Returns:
            BlackLittermanResult with posterior returns and weights
        """
        if not views:
            # No views, return equilibrium portfolio
            eq_dict = dict(zip(self.ASSET_LABELS, self.equilibrium_returns.tolist()))
            return BlackLittermanResult(
                posterior_returns=eq_dict,
                posterior_covariance=self.covariance.copy(),
                optimal_weights=dict(zip(self.ASSET_LABELS, self.market_weights.tolist())),
                equilibrium_returns=eq_dict,
                blended_returns=eq_dict,
                n_views=0,
                avg_confidence=0.0,
            )
        
        # Build pick matrix P and view vector Q
        P, Q, Omega = self._build_view_matrices(views)
        
        # Compute posterior expected returns
        posterior_returns = self._compute_posterior_returns(P, Q, Omega)
        
        # Compute posterior covariance (optional, for uncertainty quantification)
        posterior_cov = self._compute_posterior_covariance(P, Omega)
        
        # Compute optimal weights if requested
        if optimize:
            optimal_weights = self._compute_optimal_weights(posterior_returns, posterior_cov)
        else:
            optimal_weights = dict(zip(self.ASSET_LABELS, self.market_weights.tolist()))
        
        # Convert to dictionaries
        posterior_dict = dict(zip(self.ASSET_LABELS, posterior_returns.tolist()))
        eq_dict = dict(zip(self.ASSET_LABELS, self.equilibrium_returns.tolist()))
        blended_dict = posterior_dict  # Posterior is the blended view
        
        avg_confidence = np.mean([v.confidence for v in views]) if views else 0.0
        
        return BlackLittermanResult(
            posterior_returns=posterior_dict,
            posterior_covariance=posterior_cov,
            optimal_weights=optimal_weights,
            equilibrium_returns=eq_dict,
            blended_returns=blended_dict,
            n_views=len(views),
            avg_confidence=avg_confidence,
        )
    
    def _build_view_matrices(
        self,
        views: list[AlphaView],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build pick matrix P, view vector Q, and uncertainty matrix Ω.
        
        P: K x N matrix where K is number of views, N is number of assets
        Q: K x 1 vector of expected returns from views
        Ω: K x K diagonal matrix of view uncertainties
        """
        k = len(views)
        n = self.NUM_ASSETS
        
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)
        
        for idx, view in enumerate(views):
            if view.view_type == ViewType.ABSOLUTE:
                # Absolute view: P has 1 for each asset in view
                for asset in view.assets:
                    if asset in self.ASSET_LABELS:
                        asset_idx = self.ASSET_LABELS.index(asset)
                        P[idx, asset_idx] = 1.0 / len(view.assets)
                Q[idx] = view.expected_return
                
            elif view.view_type == ViewType.RELATIVE:
                # Relative view: long first asset, short second
                if len(view.assets) >= 2:
                    asset_long = view.assets[0]
                    asset_short = view.assets[1]
                    if asset_long in self.ASSET_LABELS:
                        P[idx, self.ASSET_LABELS.index(asset_long)] = 1.0
                    if asset_short in self.ASSET_LABELS:
                        P[idx, self.ASSET_LABELS.index(asset_short)] = -1.0
                Q[idx] = view.expected_return
            
            # View uncertainty based on confidence
            # Higher confidence = lower uncertainty
            # Use proportional scaling: ω = (1 - confidence) * variance
            view_variance = np.dot(P[idx], np.dot(self.covariance, P[idx]))
            confidence_factor = max(0.01, 1.0 - view.confidence)
            omega_diag[idx] = confidence_factor * view_variance
        
        Omega = np.diag(omega_diag)
        return P, Q, Omega
    
    def _compute_posterior_returns(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
    ) -> np.ndarray:
        """
        Compute posterior expected returns using Black-Litterman formula.
        
        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)Π + P'Ω^(-1)Q]
        """
        tau_sigma = self.tau * self.covariance
        tau_sigma_inv = self.inv_covariance / self.tau
        
        # Handle potentially singular Omega
        try:
            omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            # Add regularization
            omega_inv = np.linalg.inv(Omega + 1e-6 * np.eye(len(Q)))
        
        # Compute posterior precision matrix
        prior_precision = tau_sigma_inv
        view_precision = np.dot(P.T, np.dot(omega_inv, P))
        posterior_precision = prior_precision + view_precision
        
        # Compute posterior mean
        prior_term = np.dot(tau_sigma_inv, self.equilibrium_returns)
        view_term = np.dot(P.T, np.dot(omega_inv, Q))
        
        try:
            posterior_cov = np.linalg.inv(posterior_precision)
            posterior_mean = np.dot(posterior_cov, prior_term + view_term)
        except np.linalg.LinAlgError:
            # Fallback to weighted average
            posterior_mean = 0.5 * self.equilibrium_returns + 0.5 * np.dot(P.T, Q)
        
        return posterior_mean
    
    def _compute_posterior_covariance(
        self,
        P: np.ndarray,
        Omega: np.ndarray,
    ) -> np.ndarray:
        """
        Compute posterior covariance matrix.
        
        Var[R] = Σ + [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
        """
        tau_sigma = self.tau * self.covariance
        tau_sigma_inv = self.inv_covariance / self.tau
        
        try:
            omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.inv(Omega + 1e-6 * np.eye(P.shape[0]))
        
        view_precision = np.dot(P.T, np.dot(omega_inv, P))
        posterior_precision = tau_sigma_inv + view_precision
        
        try:
            estimation_uncertainty = np.linalg.inv(posterior_precision)
        except np.linalg.LinAlgError:
            estimation_uncertainty = self.tau * self.covariance
        
        return self.covariance + estimation_uncertainty
    
    def _compute_optimal_weights(
        self,
        posterior_returns: np.ndarray,
        posterior_cov: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute optimal portfolio weights from posterior returns.
        
        w* = (δ * Σ)^(-1) * E[R]
        """
        # Unconstrained optimal weights
        raw_weights = np.dot(self.inv_covariance, posterior_returns) / self.risk_aversion
        
        # Project to long-only simplex
        weights = np.maximum(raw_weights, 0)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            # Fallback to market weights
            weights = self.market_weights.copy()
        
        return dict(zip(self.ASSET_LABELS, weights.tolist()))
    
    def get_implied_returns(self) -> dict[str, float]:
        """Get the implied equilibrium returns."""
        return dict(zip(self.ASSET_LABELS, self.equilibrium_returns.tolist()))


def example_usage() -> None:
    """Demonstrate Black-Litterman allocation."""
    # Sample covariance
    covariance = np.array([
        [0.04, 0.02, 0.025, 0.0001],
        [0.02, 0.09, 0.04, 0.0002],
        [0.025, 0.04, 0.05, 0.0001],
        [0.0001, 0.0002, 0.0001, 0.0001],
    ])
    
    # Market cap weights (approximate crypto market)
    market_weights = {"BTC": 0.50, "ETH": 0.30, "SOL": 0.15, "USDT": 0.05}
    
    model = BlackLittermanModel(covariance, market_weights)
    
    print("Implied equilibrium returns:")
    for asset, ret in model.get_implied_returns().items():
        print(f"  {asset}: {ret:.2%}")
    
    # Define alpha views
    views = [
        AlphaView(
            view_type=ViewType.ABSOLUTE,
            assets=["BTC"],
            expected_return=0.20,  # Expect 20% annual return
            confidence=0.7,
        ),
        AlphaView(
            view_type=ViewType.RELATIVE,
            assets=["SOL", "ETH"],
            expected_return=0.05,  # SOL outperforms ETH by 5%
            confidence=0.5,
        ),
    ]
    
    result = model.compute_allocation(views)
    
    print("\nPosterior expected returns:")
    for asset, ret in result.posterior_returns.items():
        eq_ret = result.equilibrium_returns[asset]
        print(f"  {asset}: {ret:.2%} (vs equilibrium {eq_ret:.2%})")
    
    print("\nOptimal weights:")
    for asset, weight in result.optimal_weights.items():
        mkt_weight = model.market_weights[model.ASSET_LABELS.index(asset)]
        print(f"  {asset}: {weight:.2%} (vs market {mkt_weight:.2%})")


if __name__ == "__main__":
    example_usage()
