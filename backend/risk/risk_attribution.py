#!/usr/bin/env python3
"""
Risk Attribution Analysis

Decomposes portfolio VaR into marginal and component VaR
for precise risk budgeting and position-level accountability.
Optimized for 8GB RAM with efficient matrix operations.

Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class RiskAttributionResult:
    """Results from risk attribution analysis."""
    
    # Portfolio-level metrics
    portfolio_var: float
    portfolio_volatility: float
    
    # Per-asset attributions (asset -> value)
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    incremental_var: Dict[str, float]
    
    # Risk contributions
    percentage_contribution: Dict[str, float]
    diversification_ratio: float
    
    # Metadata
    confidence_level: float
    method: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'portfolio_var': self.portfolio_var,
            'portfolio_volatility': self.portfolio_volatility,
            'marginal_var': self.marginal_var,
            'component_var': self.component_var,
            'incremental_var': self.incremental_var,
            'percentage_contribution': self.percentage_contribution,
            'diversification_ratio': self.diversification_ratio,
            'confidence_level': self.confidence_level,
            'method': self.method,
        }


class RiskAttributionEngine:
    """
    Engine for decomposing portfolio risk into constituent parts.
    
    Supports multiple VaR methodologies and provides
    granular risk attribution at the asset level.
    """
    
    def __init__(
        self,
        assets: List[str],
        confidence_level: float = 0.99,
        time_horizon: int = 1,
    ) -> None:
        """
        Initialize risk attribution engine.
        
        Args:
            assets: List of asset names
            confidence_level: VaR confidence level
            time_horizon: Time horizon in days
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        
        # Z-score for confidence level
        self.z_score = norm.ppf(confidence_level)
        
        # Covariance matrix (will be set externally)
        self.cov_matrix: Optional[np.ndarray] = None
        
        # Portfolio weights
        self.weights: Optional[np.ndarray] = None
        
        # Asset name to index mapping
        self.asset_to_idx = {asset: idx for idx, asset in enumerate(assets)}
    
    def set_covariance_matrix(self, cov_matrix: np.ndarray) -> None:
        """Set the covariance matrix for risk calculations."""
        if cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Covariance matrix must be {self.n_assets}x{self.n_assets}"
            )
        
        # Validate positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        if np.any(eigenvalues < -1e-10):
            # Make positive semi-definite
            cov_matrix = self._nearest_positive_definite(cov_matrix)
        
        self.cov_matrix = cov_matrix.copy()
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set portfolio weights."""
        if set(weights.keys()) != set(self.assets):
            raise ValueError("Weights must cover all assets")
        
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        self.weights = np.array([weights[asset] for asset in self.assets])
    
    def compute_risk_attribution(
        self,
        portfolio_value: float = 1_000_000.0,
        method: str = 'parametric'
    ) -> RiskAttributionResult:
        """
        Compute comprehensive risk attribution.
        
        Args:
            portfolio_value: Total portfolio value
            method: 'parametric' or 'historical'
            
        Returns:
            Complete risk attribution results
        """
        if self.cov_matrix is None or self.weights is None:
            raise ValueError("Must set covariance matrix and weights first")
        
        # Portfolio volatility
        port_variance = self.weights @ self.cov_matrix @ self.weights
        port_volatility = np.sqrt(port_variance)
        
        # Scale by time horizon
        port_volatility *= np.sqrt(self.time_horizon)
        
        # Portfolio VaR
        portfolio_var = self.z_score * port_volatility * portfolio_value
        
        # Marginal VaR (sensitivity of VaR to weight changes)
        marginal_var = self._compute_marginal_var(port_volatility, portfolio_value)
        
        # Component VaR (contribution to total VaR)
        component_var = self._compute_component_var(marginal_var, portfolio_value)
        
        # Incremental VaR (impact of removing each asset)
        incremental_var = self._compute_incremental_var(portfolio_value)
        
        # Percentage contributions
        total_component = sum(component_var.values())
        percentage_contribution = {
            asset: cv / total_component if total_component > 0 else 0.0
            for asset, cv in component_var.items()
        }
        
        # Diversification ratio
        undiversified_var = sum(
            abs(self.weights[i]) * np.sqrt(self.cov_matrix[i, i])
            for i in range(self.n_assets)
        )
        diversified_var = port_volatility
        diversification_ratio = undiversified_var / diversified_var if diversified_var > 0 else 1.0
        
        return RiskAttributionResult(
            portfolio_var=float(portfolio_var),
            portfolio_volatility=float(port_volatility),
            marginal_var=marginal_var,
            component_var=component_var,
            incremental_var=incremental_var,
            percentage_contribution=percentage_contribution,
            diversification_ratio=float(diversification_ratio),
            confidence_level=self.confidence_level,
            method=method,
        )
    
    def _compute_marginal_var(
        self,
        port_volatility: float,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Compute Marginal VaR for each asset.
        
        Marginal VaR = d(VaR)/d(weight) = z * cov(i, portfolio) / sigma_p * portfolio_value
        """
        marginal_var = {}
        
        # Covariance of each asset with portfolio
        cov_with_portfolio = self.cov_matrix @ self.weights
        
        for i, asset in enumerate(self.assets):
            # Marginal contribution to volatility
            marginal_vol = cov_with_portfolio[i] / port_volatility if port_volatility > 0 else 0.0
            
            # Marginal VaR
            marginal_var[asset] = float(
                self.z_score * marginal_vol * np.sqrt(self.time_horizon) * portfolio_value
            )
        
        return marginal_var
    
    def _compute_component_var(
        self,
        marginal_var: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Compute Component VaR for each asset.
        
        Component VaR = weight * Marginal VaR
        Sum of Component VaRs = Total VaR
        """
        component_var = {}
        
        for i, asset in enumerate(self.assets):
            weight = self.weights[i]
            component_var[asset] = float(weight * marginal_var[asset])
        
        return component_var
    
    def _compute_incremental_var(
        self,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Compute Incremental VaR for each asset.
        
        Incremental VaR = VaR(portfolio) - VaR(portfolio without asset)
        Measures the impact of removing an asset.
        """
        incremental_var = {}
        
        # Full portfolio VaR
        full_var = self.weights @ self.cov_matrix @ self.weights
        full_var = self.z_score * np.sqrt(full_var) * np.sqrt(self.time_horizon) * portfolio_value
        
        for i, asset in enumerate(self.assets):
            # Create reduced portfolio without this asset
            reduced_weights = np.delete(self.weights, i)
            reduced_cov = np.delete(np.delete(self.cov_matrix, i, axis=0), i, axis=1)
            
            # Renormalize weights
            weight_sum = np.sum(reduced_weights)
            if weight_sum > 0:
                reduced_weights = reduced_weights / weight_sum
            
            # Reduced portfolio VaR
            if len(reduced_weights) > 0 and reduced_cov.size > 0:
                reduced_var = reduced_weights @ reduced_cov @ reduced_weights
                reduced_var = self.z_score * np.sqrt(reduced_var) * np.sqrt(self.time_horizon) * portfolio_value
            else:
                reduced_var = 0.0
            
            # Incremental VaR
            incremental_var[asset] = float(full_var - reduced_var)
        
        return incremental_var
    
    def _nearest_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix (Higham's algorithm)."""
        # Symmetrize
        B = (matrix + matrix.T) / 2
        
        # SVD
        U, s, Vt = np.linalg.svd(B)
        
        # Replace negative eigenvalues with small positive
        s = np.maximum(s, 1e-10)
        
        # Reconstruct
        result = U @ np.diag(s) @ Vt
        
        # Ensure symmetry
        result = (result + result.T) / 2
        
        return result
    
    def get_riskiest_assets(self, component_var: Dict[str, float], n: int = 3) -> List[Tuple[str, float]]:
        """Get the top N riskiest assets by component VaR."""
        sorted_assets = sorted(
            component_var.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_assets[:n]
    
    def check_risk_budget(
        self,
        component_var: Dict[str, float],
        budget_limits: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Check if risk contributions are within budget limits.
        
        Args:
            component_var: Component VaR per asset
            budget_limits: Maximum allowed VaR per asset
            
        Returns:
            Dictionary of asset -> within_budget
        """
        return {
            asset: abs(component_var.get(asset, 0)) <= limit
            for asset, limit in budget_limits.items()
        }


def analyze_crypto_portfolio_risk(
    assets: List[str] = None,
    weights: Dict[str, float] = None,
    portfolio_value: float = 1_000_000.0,
    volatilities: Dict[str, float] = None,
    correlations: np.ndarray = None,
) -> RiskAttributionResult:
    """
    Convenience function to analyze crypto portfolio risk.
    
    Args:
        assets: List of crypto assets
        weights: Portfolio weights
        portfolio_value: Total portfolio value
        volatilities: Annualized volatilities per asset
        correlations: Correlation matrix
        
    Returns:
        Complete risk attribution results
    """
    if assets is None:
        assets = ['BTC', 'ETH', 'SOL']
    
    if weights is None:
        weights = {asset: 1.0 / len(assets) for asset in assets}
    
    if volatilities is None:
        # Typical crypto daily volatilities
        volatilities = {
            'BTC': 0.04,
            'ETH': 0.05,
            'SOL': 0.07,
        }
    
    if correlations is None:
        # Typical crypto correlations
        correlations = np.array([
            [1.0, 0.7, 0.6],
            [0.7, 1.0, 0.65],
            [0.6, 0.65, 1.0],
        ])
    
    # Build covariance matrix
    n = len(assets)
    cov_matrix = np.zeros((n, n))
    
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets):
            vol1 = volatilities.get(asset1, 0.05)
            vol2 = volatilities.get(asset2, 0.05)
            corr = correlations[i, j]
            cov_matrix[i, j] = corr * vol1 * vol2
    
    # Create engine and compute attribution
    engine = RiskAttributionEngine(assets, confidence_level=0.99, time_horizon=1)
    engine.set_covariance_matrix(cov_matrix)
    engine.set_weights(weights)
    
    result = engine.compute_risk_attribution(portfolio_value, method='parametric')
    
    print("\nRisk Attribution Analysis")
    print("=" * 50)
    print(f"Portfolio VaR (99%): ${result.portfolio_var:,.2f}")
    print(f"Portfolio Volatility: {result.portfolio_volatility*100:.2f}%")
    print(f"Diversification Ratio: {result.diversification_ratio:.3f}")
    
    print("\nComponent VaR by Asset:")
    for asset, cvar in sorted(result.component_var.items(), key=lambda x: abs(x[1]), reverse=True):
        pct = result.percentage_contribution[asset] * 100
        print(f"  {asset}: ${cvar:,.2f} ({pct:.1f}%)")
    
    print("\nIncremental VaR (impact of removal):")
    for asset, ivar in sorted(result.incremental_var.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {asset}: ${ivar:,.2f}")
    
    return result


if __name__ == '__main__':
    result = analyze_crypto_portfolio_risk(
        assets=['BTC', 'ETH', 'SOL'],
        weights={'BTC': 0.5, 'ETH': 0.3, 'SOL': 0.2},
        portfolio_value=1_000_000
    )
    
    print("\nDetailed breakdown:")
    for key, value in result.to_dict().items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.6f}")
        else:
            print(f"{key}: {value}")
