#!/usr/bin/env python3
"""
Monte Carlo Stress Testing with Fat-Tailed Shocks

Injects correlated, fat-tailed shocks into the portfolio using
copula-based dependency modeling and EVT-derived marginals.
Optimized for 8GB RAM constraints with efficient sampling.

Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.stats import t, norm, genpareto
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ShockDistribution(Enum):
    """Available shock distributions for Monte Carlo."""
    NORMAL = "normal"
    STUDENT_T = "student_t"
    GPD = "gpd"
    EMPIRICAL = "empirical"


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo stress testing."""
    
    n_simulations: int = 10000
    confidence_levels: Tuple[float, ...] = (0.95, 0.99, 0.999)
    time_horizon_days: int = 1
    use_correlation: bool = True
    distribution: ShockDistribution = ShockDistribution.STUDENT_T
    t_degrees_of_freedom: float = 3.0  # For Student-t
    gpd_threshold: Optional[float] = None  # For GPD
    random_seed: Optional[int] = None
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.n_simulations < 100:
            return False
        if self.t_degrees_of_freedom <= 0:
            return False
        if any(not (0 < cl < 1) for cl in self.confidence_levels):
            return False
        return True


@dataclass
class StressTestResult:
    """Results from Monte Carlo stress test."""
    
    var_95: float
    var_99: float
    var_99_9: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    max_loss: float
    mean_loss: float
    loss_std: float
    skewness: float
    kurtosis: float
    breach_count_95: int
    breach_count_99: int
    simulation_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'var_95': self.var_95,
            'var_99': self.var_99,
            'var_99_9': self.var_99_9,
            'es_95': self.expected_shortfall_95,
            'es_99': self.expected_shortfall_99,
            'max_loss': self.max_loss,
            'mean_loss': self.mean_loss,
            'loss_std': self.loss_std,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'breaches_95': self.breach_count_95,
            'breaches_99': self.breach_count_99,
            'simulation_time_ms': self.simulation_time_ms,
        }


class MonteCarloStressTester:
    """
    Monte Carlo stress testing engine with fat-tailed shocks.
    
    Uses copula-based correlation structure and configurable
    marginal distributions for realistic crypto market simulation.
    """
    
    def __init__(
        self,
        assets: List[str],
        config: Optional[MonteCarloConfig] = None,
        memory_limit_mb: int = 512,
    ) -> None:
        """
        Initialize stress tester.
        
        Args:
            assets: List of asset names
            config: Monte Carlo configuration
            memory_limit_mb: Memory limit for simulations
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.config = config or MonteCarloConfig()
        self.memory_limit_mb = memory_limit_mb
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid Monte Carlo configuration")
        
        # Set random seed if provided
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Pre-allocate simulation arrays based on memory limit
        max_simulations = self._calculate_max_simulations()
        self.n_simulations = min(self.config.n_simulations, max_simulations)
        
        # Correlation matrix (identity by default)
        self.correlation_matrix = np.eye(self.n_assets)
        
        # Marginal distribution parameters per asset
        self.marginal_params: Dict[str, Dict[str, float]] = {}
        
        # Simulation cache
        self._latest_results: Optional[StressTestResult] = None
        self._simulated_losses: Optional[np.ndarray] = None
    
    def _calculate_max_simulations(self) -> int:
        """Calculate maximum simulations based on memory constraints."""
        # Each simulation stores n_assets floats (8 bytes each)
        bytes_per_simulation = self.n_assets * 8
        available_bytes = self.memory_limit_mb * 1024 * 1024 * 0.5  # Use 50% of limit
        
        max_sims = int(available_bytes / bytes_per_simulation)
        return max(1000, min(max_sims, 100000))  # Clamp between 1K and 100K
    
    def set_correlation_matrix(self, corr_matrix: np.ndarray) -> None:
        """Set the correlation matrix for dependent sampling."""
        if corr_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Correlation matrix must be {self.n_assets}x{self.n_assets}"
            )
        
        # Validate correlation matrix
        if not np.allclose(corr_matrix, corr_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")
        
        if not np.all(np.linalg.eigvalsh(corr_matrix) > -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite")
        
        self.correlation_matrix = corr_matrix.copy()
    
    def set_marginal_parameters(
        self,
        asset: str,
        params: Dict[str, float]
    ) -> None:
        """
        Set marginal distribution parameters for an asset.
        
        Args:
            asset: Asset name
            params: Distribution parameters
                - For Student-t: {'mu': float, 'sigma': float, 'df': float}
                - For GPD: {'xi': float, 'sigma': float, 'threshold': float}
                - For Normal: {'mu': float, 'sigma': float}
        """
        self.marginal_params[asset] = params.copy()
    
    def set_portfolio_weights(self, weights: Dict[str, float]) -> None:
        """Set portfolio weights for aggregation."""
        if set(weights.keys()) != set(self.assets):
            raise ValueError("Weights must cover all assets")
        
        self.weights = np.array([weights[asset] for asset in self.assets])
    
    def run_stress_test(
        self,
        portfolio_value: float = 1_000_000.0
    ) -> StressTestResult:
        """
        Run Monte Carlo stress test.
        
        Args:
            portfolio_value: Total portfolio value for P&L calculation
            
        Returns:
            Comprehensive stress test results
        """
        import time
        start_time = time.perf_counter()
        
        # Generate correlated random shocks
        shocks = self._generate_shocks()
        
        # Calculate portfolio losses
        portfolio_returns = shocks @ self.weights
        losses = -portfolio_returns * portfolio_value
        
        # Store for later analysis
        self._simulated_losses = losses
        
        # Compute risk metrics
        results = self._compute_metrics(losses, portfolio_value)
        
        # Record timing
        results.simulation_time_ms = (time.perf_counter() - start_time) * 1000
        
        self._latest_results = results
        return results
    
    def _generate_shocks(self) -> np.ndarray:
        """Generate correlated fat-tailed shocks."""
        dist = self.config.distribution
        
        if dist == ShockDistribution.NORMAL:
            return self._generate_normal_shocks()
        elif dist == ShockDistribution.STUDENT_T:
            return self._generate_student_t_shocks()
        elif dist == ShockDistribution.GPD:
            return self._generate_gpd_shocks()
        else:
            return self._generate_empirical_shocks()
    
    def _generate_normal_shocks(self) -> np.ndarray:
        """Generate multivariate normal shocks."""
        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(self.correlation_matrix)
        
        # Generate independent standard normals
        Z = np.random.standard_normal((self.n_simulations, self.n_assets))
        
        # Apply correlation structure
        correlated = Z @ L.T
        
        # Scale by marginal volatilities
        for i, asset in enumerate(self.assets):
            params = self.marginal_params.get(asset, {'mu': 0, 'sigma': 0.02})
            mu = params.get('mu', 0)
            sigma = params.get('sigma', 0.02)
            correlated[:, i] = mu + sigma * correlated[:, i]
        
        return correlated
    
    def _generate_student_t_shocks(self) -> np.ndarray:
        """Generate multivariate Student-t shocks (fat tails)."""
        df = self.config.t_degrees_of_freedom
        
        # Generate chi-squared scaling factor
        chi_sq = np.random.chisquare(df, self.n_simulations)
        scale = np.sqrt(df / chi_sq).reshape(-1, 1)
        
        # Generate correlated normals
        normal_shocks = self._generate_normal_shocks()
        
        # Apply Student-t scaling
        t_shocks = normal_shocks * scale
        
        return t_shocks
    
    def _generate_gpd_shocks(self) -> np.ndarray:
        """Generate shocks using Generalized Pareto margins."""
        # Start with uniform margins via Gaussian copula
        L = np.linalg.cholesky(self.correlation_matrix)
        Z = np.random.standard_normal((self.n_simulations, self.n_assets))
        correlated_normal = Z @ L.T
        
        # Transform to uniform
        uniform = norm.cdf(correlated_normal)
        
        # Transform to GPD margins per asset
        shocks = np.zeros_like(uniform)
        
        for i, asset in enumerate(self.assets):
            params = self.marginal_params.get(asset, {
                'xi': 0.3, 'sigma': 0.02, 'threshold': 0.01
            })
            xi = params.get('xi', 0.3)
            sigma = params.get('sigma', 0.02)
            threshold = params.get('threshold', 0.01)
            
            # Inverse GPD CDF
            try:
                gpd_quantiles = genpareto.ppf(uniform[:, i], xi, loc=threshold, scale=sigma)
                shocks[:, i] = gpd_quantiles
            except Exception:
                # Fallback to normal
                shocks[:, i] = correlated_normal[:, i] * sigma
        
        return shocks
    
    def _generate_empirical_shocks(self) -> np.ndarray:
        """Generate shocks by resampling historical returns."""
        # Placeholder: would use historical return matrix
        # For now, fall back to Student-t
        return self._generate_student_t_shocks()
    
    def _compute_metrics(
        self,
        losses: np.ndarray,
        portfolio_value: float
    ) -> StressTestResult:
        """Compute comprehensive risk metrics from simulated losses."""
        # Sort losses for VaR calculation
        sorted_losses = np.sort(losses)
        
        # VaR at different confidence levels
        var_95_idx = int(self.n_simulations * 0.95)
        var_99_idx = int(self.n_simulations * 0.99)
        var_99_9_idx = int(self.n_simulations * 0.999)
        
        var_95 = sorted_losses[min(var_95_idx, len(sorted_losses) - 1)]
        var_99 = sorted_losses[min(var_99_idx, len(sorted_losses) - 1)]
        var_99_9 = sorted_losses[min(var_99_9_idx, len(sorted_losses) - 1)]
        
        # Expected Shortfall (average of tail losses)
        es_95 = np.mean(sorted_losses[var_95_idx:])
        es_99 = np.mean(sorted_losses[var_99_idx:])
        
        # Basic statistics
        max_loss = np.max(losses)
        mean_loss = np.mean(losses)
        loss_std = np.std(losses)
        
        # Higher moments
        if loss_std > 1e-10:
            skewness = np.mean(((losses - mean_loss) / loss_std) ** 3)
            kurtosis = np.mean(((losses - mean_loss) / loss_std) ** 4) - 3  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Breach counts
        breach_95 = np.sum(losses > var_95)
        breach_99 = np.sum(losses > var_99)
        
        return StressTestResult(
            var_95=float(var_95),
            var_99=float(var_99),
            var_99_9=float(var_99_9),
            expected_shortfall_95=float(es_95),
            expected_shortfall_99=float(es_99),
            max_loss=float(max_loss),
            mean_loss=float(mean_loss),
            loss_std=float(loss_std),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
            breach_count_95=int(breach_95),
            breach_count_99=int(breach_99),
            simulation_time_ms=0.0,  # Will be set by caller
        )
    
    def get_loss_distribution(self) -> Optional[np.ndarray]:
        """Get the simulated loss distribution."""
        return self._simulated_losses
    
    def get_worst_scenarios(self, n: int = 10) -> Optional[np.ndarray]:
        """Get the worst n scenarios from the simulation."""
        if self._simulated_losses is None:
            return None
        
        # Get indices of worst losses
        worst_indices = np.argsort(self._simulated_losses)[-n:][::-1]
        return self._simulated_losses[worst_indices]
    
    def calculate_confidence_intervals(
        self,
        n_bootstrap: int = 100
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for VaR estimates.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary of metric -> (lower, upper) CI
        """
        if self._simulated_losses is None:
            raise ValueError("Must run stress test first")
        
        var_95_samples = []
        var_99_samples = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(
                self._simulated_losses,
                size=len(self._simulated_losses),
                replace=True
            )
            
            sorted_sample = np.sort(sample)
            var_95_samples.append(sorted_sample[int(len(sample) * 0.95)])
            var_99_samples.append(sorted_sample[int(len(sample) * 0.99)])
        
        return {
            'var_95': (np.percentile(var_95_samples, 2.5),
                      np.percentile(var_95_samples, 97.5)),
            'var_99': (np.percentile(var_99_samples, 2.5),
                      np.percentile(var_99_samples, 97.5)),
        }


def run_crypto_stress_test(
    assets: List[str] = None,
    portfolio_value: float = 1_000_000.0,
    n_simulations: int = 50000
) -> StressTestResult:
    """
    Convenience function to run stress test on crypto portfolio.
    
    Args:
        assets: List of crypto assets
        portfolio_value: Total portfolio value
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        Comprehensive stress test results
    """
    if assets is None:
        assets = ['BTC', 'ETH', 'SOL']
    
    # Create tester with Student-t distribution (fat tails)
    config = MonteCarloConfig(
        n_simulations=n_simulations,
        distribution=ShockDistribution.STUDENT_T,
        t_degrees_of_freedom=3.0,  # Heavy tails for crypto
    )
    
    tester = MonteCarloStressTester(assets, config)
    
    # Set typical crypto correlations
    corr_matrix = np.array([
        [1.0, 0.7, 0.6],
        [0.7, 1.0, 0.65],
        [0.6, 0.65, 1.0],
    ])
    tester.set_correlation_matrix(corr_matrix)
    
    # Set marginal parameters (typical crypto daily vol)
    tester.set_marginal_parameters('BTC', {'mu': 0.001, 'sigma': 0.04, 'df': 3})
    tester.set_marginal_parameters('ETH', {'mu': 0.002, 'sigma': 0.05, 'df': 3})
    tester.set_marginal_parameters('SOL', {'mu': 0.003, 'sigma': 0.07, 'df': 3})
    
    # Equal weight portfolio
    weights = {asset: 1.0 / len(assets) for asset in assets}
    tester.set_portfolio_weights(weights)
    
    # Run stress test
    results = tester.run_stress_test(portfolio_value)
    
    print(f"\nMonte Carlo Stress Test Results ({n_simulations:,} simulations)")
    print("=" * 50)
    print(f"VaR(95%): ${results.var_95:,.2f}")
    print(f"VaR(99%): ${results.var_99:,.2f}")
    print(f"VaR(99.9%): ${results.var_99_9:,.2f}")
    print(f"ES(95%): ${results.expected_shortfall_95:,.2f}")
    print(f"ES(99%): ${results.expected_shortfall_99:,.2f}")
    print(f"Max Loss: ${results.max_loss:,.2f}")
    print(f"Skewness: {results.skewness:.3f}")
    print(f"Excess Kurtosis: {results.kurtosis:.3f}")
    print(f"Simulation Time: {results.simulation_time_ms:.2f} ms")
    
    return results


if __name__ == '__main__':
    results = run_crypto_stress_test(
        assets=['BTC', 'ETH', 'SOL'],
        portfolio_value=1_000_000,
        n_simulations=50000
    )
    
    print("\nDetailed breakdown:")
    for key, value in results.to_dict().items():
        print(f"  {key}: {value}")
