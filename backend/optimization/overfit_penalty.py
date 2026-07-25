"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
Chapter 3: Walk-Forward Optimization and Hyperparameter Tuning

File: backend/optimization/overfit_penalty.py
Purpose: Apply Deflated Sharpe Ratio and other penalties to punish curve-fitting.
Features:
    - Deflated Sharpe Ratio (DSR) calculation
    - Probability of Backtest Overfitting (PBO)
    - Multiple testing correction (Bonferroni, Holm)
    - Strategy complexity penalty
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.special import comb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OverfitMetrics:
    """Comprehensive overfitting metrics."""
    # Original Sharpe Ratio
    sharpe_ratio: float
    
    # Deflated Sharpe Ratio
    dsr: float
    dsr_adjustment: float
    
    # Probability of Backtest Overfitting
    pbo: float
    
    # Multiple testing corrections
    bonferroni_sharpe: float
    holm_sharpe: float
    
    # Complexity penalty
    complexity_penalty: float
    penalized_sharpe: float
    
    # Additional diagnostics
    n_trials: int
    n_parameters: int
    sample_size: int
    skewness: float
    kurtosis: float
    
    # Pass/fail indicators
    passed_dsr_test: bool
    passed_pbo_test: bool
    passed_complexity_test: bool


class OverfitPenalty:
    """
    Implements statistical tests to detect and penalize backtest overfitting.
    Based on Bailey & López de Prado research on backtest overfitting.
    """
    
    __slots__ = [
        'returns',
        'n_trials',
        'n_parameters',
        'sample_size',
        'risk_free_rate',
        'target_sharpe',
        'random_state'
    ]
    
    def __init__(
        self,
        returns: np.ndarray = None,
        n_trials: int = 1,
        n_parameters: int = 0,
        risk_free_rate: float = 0.0,
        target_sharpe: float = 0.5,
        seed: int = None
    ):
        self.returns = returns
        self.n_trials = n_trials
        self.n_parameters = n_parameters
        self.sample_size = len(returns) if returns is not None else 0
        self.risk_free_rate = risk_free_rate
        self.target_sharpe = target_sharpe
        self.random_state = np.random.RandomState(seed)
    
    def set_returns(self, returns: np.ndarray) -> None:
        """Set the returns array for analysis."""
        self.returns = np.asarray(returns, dtype=np.float64)
        self.sample_size = len(self.returns)
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray = None,
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        rets = returns if returns is not None else self.returns
        
        if rets is None or len(rets) < 2:
            return 0.0
        
        excess_returns = rets - self.risk_free_rate / periods_per_year
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        
        if annualize:
            sharpe *= np.sqrt(periods_per_year)
        
        return sharpe
    
    def calculate_deflated_sharpe_ratio(
        self,
        expected_max_sharpe: float = None
    ) -> Tuple[float, float]:
        """
        Calculate Deflated Sharpe Ratio (DSR).
        
        DSR adjusts the observed Sharpe ratio for:
        1. Multiple testing (number of trials)
        2. Non-normality (skewness and kurtosis)
        3. Track record length
        
        Returns: (DSR, adjustment_factor)
        """
        if self.returns is None or len(self.returns) < 10:
            return 0.0, 0.0
        
        # Calculate observed Sharpe
        observed_sr = self.calculate_sharpe_ratio()
        
        # Calculate moments
        skew = stats.skew(self.returns)
        kurt = stats.kurtosis(self.returns)  # Excess kurtosis
        
        # Expected maximum Sharpe under null (Bailey's formula)
        if expected_max_sharpe is None:
            expected_max_sharpe = self._expected_max_sharpe()
        
        # Variance of Sharpe ratio estimator (with non-normality adjustment)
        var_sr = (1 + 0.5 * skew**2 - kurt * self.risk_free_rate * observed_sr 
                  + 0.25 * (kurt + 2) * observed_sr**2) / self.sample_size
        
        std_sr = np.sqrt(var_sr)
        
        # DSR = CDF of normal distribution at (SR - E[Max])/Std[SR]
        z_score = (observed_sr - expected_max_sharpe) / std_sr if std_sr > 0 else 0
        dsr = stats.norm.cdf(z_score)
        
        adjustment = expected_max_sharpe
        
        return dsr, adjustment
    
    def _expected_max_sharpe(self) -> float:
        """
        Estimate expected maximum Sharpe ratio under the null hypothesis.
        Uses order statistics approximation.
        """
        if self.n_trials <= 1:
            return 0.0
        
        # Euler-Mascheroni constant
        gamma = 0.5772156649
        
        # Approximation for expected maximum of N standard normals
        # E[max] ≈ sqrt(2 * ln(N)) for large N
        if self.n_trials < 100:
            # Exact calculation for small N using numerical integration
            expected_max = 0.0
            for k in range(1, self.n_trials + 1):
                expected_max += 1.0 / k
            expected_max = np.sqrt(2) * (np.log(self.n_trials) + gamma)
        else:
            expected_max = np.sqrt(2 * np.log(self.n_trials))
        
        # Adjust for track record length
        expected_max /= np.sqrt(self.sample_size)
        
        return expected_max
    
    def calculate_pbo(
        self,
        n_simulations: int = 10000
    ) -> float:
        """
        Calculate Probability of Backtest Overfitting (PBO).
        
        PBO estimates the probability that the optimal strategy found
        through backtesting will underperform out-of-sample.
        
        Lower PBO is better (< 0.5 means strategy likely robust).
        """
        if self.returns is None or len(self.returns) < 20:
            return 1.0  # High PBO for insufficient data
        
        if self.n_trials <= 1:
            return 0.0  # No multiple testing issue
        
        # Monte Carlo simulation approach
        n_underperform = 0
        
        for _ in range(n_simulations):
            # Bootstrap sample
            bootstrap_idx = self.random_state.choice(
                len(self.returns), 
                size=len(self.returns), 
                replace=True
            )
            bootstrap_returns = self.returns[bootstrap_idx]
            
            # Simulate multiple strategies with random noise
            max_sharpe_bootstrap = 0.0
            
            for _ in range(self.n_trials):
                # Add random noise to simulate different strategies
                noise = self.random_state.randn(len(bootstrap_returns)) * 0.01
                noisy_returns = bootstrap_returns + noise
                
                sr = self.calculate_sharpe_ratio(noisy_returns)
                max_sharpe_bootstrap = max(max_sharpe_bootstrap, sr)
            
            # Check if best in-sample underperforms true best
            true_best_sr = self.calculate_sharpe_ratio()
            if max_sharpe_bootstrap > true_best_sr:
                n_underperform += 1
        
        pbo = n_underperform / n_simulations
        
        return pbo
    
    def apply_multiple_testing_correction(
        self,
        sharpe_ratios: List[float],
        method: str = 'bonferroni'
    ) -> Tuple[float, float]:
        """
        Apply multiple testing corrections to Sharpe ratios.
        
        Methods:
        - 'bonferroni': Conservative family-wise error rate control
        - 'holm': Step-down Bonferroni (more powerful)
        - 'fdr': False Discovery Rate control (Benjamini-Hochberg)
        
        Returns: (corrected_best_sharpe, significance_level)
        """
        if not sharpe_ratios:
            return 0.0, 1.0
        
        n_tests = len(sharpe_ratios)
        sorted_sharpes = sorted(sharpe_ratios, reverse=True)
        best_sharpe = sorted_sharpes[0]
        
        # Convert Sharpe to p-values (two-tailed test against zero)
        p_values = [2 * (1 - stats.norm.cdf(abs(sr) * np.sqrt(self.sample_size))) 
                   for sr in sharpe_ratios]
        
        if method == 'bonferroni':
            # Bonferroni correction
            alpha_corrected = 0.05 / n_tests
            corrected_p = min(p_values[0] * n_tests, 1.0)
            
        elif method == 'holm':
            # Holm step-down procedure
            sorted_p = sorted(p_values)
            rejected = False
            corrected_p = sorted_p[0]
            
            for i, p in enumerate(sorted_p):
                threshold = 0.05 / (n_tests - i)
                if p > threshold:
                    break
                corrected_p = p * (n_tests - i)
                
        elif method == 'fdr':
            # Benjamini-Hochberg FDR control
            sorted_p = sorted(p_values)
            ranks = np.arange(1, n_tests + 1)
            thresholds = ranks * 0.05 / n_tests
            
            rejected = sorted_p <= thresholds
            if np.any(rejected):
                max_rejected = np.max(np.where(rejected)[0])
                corrected_p = sorted_p[max_rejected]
            else:
                corrected_p = 1.0
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Convert corrected p-value back to Sharpe-like metric
        corrected_sharpe = stats.norm.ppf(1 - corrected_p / 2) / np.sqrt(self.sample_size)
        
        return corrected_sharpe, corrected_p
    
    def calculate_complexity_penalty(
        self,
        n_rules: int = None,
        n_conditions: int = None,
        lookback_periods: List[int] = None
    ) -> float:
        """
        Calculate penalty for strategy complexity.
        
        More complex strategies are more likely to be overfit.
        Penalty based on:
        - Number of parameters
        - Number of decision rules
        - Lookback period optimization
        """
        n_params = n_rules or self.n_parameters
        
        if n_params == 0:
            return 0.0
        
        # Base penalty: log(number of parameters)
        base_penalty = np.log(n_params + 1)
        
        # Additional penalty for optimized lookbacks
        lookback_penalty = 0.0
        if lookback_periods and len(lookback_periods) > 1:
            # Penalize searching over many lookback periods
            lookback_penalty = np.log(len(lookback_periods))
        
        # Sample size adjustment (penalize more for short samples)
        sample_penalty = max(0, np.log(252 / max(self.sample_size, 1)))
        
        total_penalty = base_penalty + lookback_penalty + sample_penalty
        
        return total_penalty
    
    def compute_all_metrics(
        self,
        sharpe_ratios: List[float] = None,
        n_rules: int = None,
        lookback_periods: List[int] = None
    ) -> OverfitMetrics:
        """Compute comprehensive overfitting metrics."""
        # Base Sharpe
        sharpe = self.calculate_sharpe_ratio()
        
        # DSR
        dsr, dsr_adj = self.calculate_deflated_sharpe_ratio()
        
        # PBO
        pbo = self.calculate_pbo(n_simulations=1000)
        
        # Multiple testing corrections
        if sharpe_ratios and len(sharpe_ratios) > 1:
            bonf_sharpe, bonf_p = self.apply_multiple_testing_correction(
                sharpe_ratios, 'bonferroni'
            )
            holm_sharpe, holm_p = self.apply_multiple_testing_correction(
                sharpe_ratios, 'holm'
            )
        else:
            bonf_sharpe = sharpe
            holm_sharpe = sharpe
        
        # Complexity penalty
        complexity_pen = self.calculate_complexity_penalty(
            n_rules=n_rules,
            lookback_periods=lookback_periods
        )
        
        # Penalized Sharpe
        penalized_sharpe = sharpe * (1 - min(complexity_pen / 10, 0.5))
        
        # Moments
        skew = stats.skew(self.returns) if self.returns is not None else 0.0
        kurt = stats.kurtosis(self.returns) if self.returns is not None else 0.0
        
        # Tests
        passed_dsr = dsr > 0.5  # DSR should be > 0.5
        passed_pbo = pbo < 0.5  # PBO should be < 0.5
        passed_complexity = complexity_pen < 2.0  # Reasonable complexity
        
        return OverfitMetrics(
            sharpe_ratio=sharpe,
            dsr=dsr,
            dsr_adjustment=dsr_adj,
            pbo=pbo,
            bonferroni_sharpe=bonf_sharpe,
            holm_sharpe=holm_sharpe,
            complexity_penalty=complexity_pen,
            penalized_sharpe=penalized_sharpe,
            n_trials=self.n_trials,
            n_parameters=self.n_parameters,
            sample_size=self.sample_size,
            skewness=skew,
            kurtosis=kurt,
            passed_dsr_test=passed_dsr,
            passed_pbo_test=passed_pbo,
            passed_complexity_test=passed_complexity
        )


def main():
    """Example usage of overfit penalty calculator."""
    print("="*60)
    print("OVERFIT PENALTY ANALYZER")
    print("="*60)
    
    # Generate sample returns (simulating a backtest)
    np.random.seed(42)
    n_samples = 252  # 1 year of daily returns
    
    # Simulate returns with some signal
    signal = 0.001  # Small daily edge
    noise = 0.02    # Daily volatility
    returns = signal + np.random.randn(n_samples) * noise
    
    # Create analyzer
    analyzer = OverfitPenalty(
        returns=returns,
        n_trials=100,  # Tested 100 parameter combinations
        n_parameters=5,
        seed=42
    )
    
    # Compute all metrics
    metrics = analyzer.compute_all_metrics(
        sharpe_ratios=[analyzer.calculate_sharpe_ratio() * np.random.uniform(0.8, 1.2) 
                      for _ in range(100)],
        n_rules=5,
        lookback_periods=[10, 20, 30, 50, 100]
    )
    
    print("\n" + "="*60)
    print("OVERFITTING METRICS")
    print("="*60)
    print(f"Observed Sharpe Ratio:     {metrics.sharpe_ratio:.3f}")
    print(f"Deflated Sharpe Ratio:     {metrics.dsr:.3f} (adjustment: {metrics.dsr_adjustment:.3f})")
    print(f"Probability of Overfitting: {metrics.pbo:.3f}")
    print(f"Bonferroni-corrected SR:   {metrics.bonferroni_sharpe:.3f}")
    print(f"Holm-corrected SR:         {metrics.holm_sharpe:.3f}")
    print(f"Complexity Penalty:        {metrics.complexity_penalty:.3f}")
    print(f"Penalized Sharpe:          {metrics.penalized_sharpe:.3f}")
    print("-"*60)
    print(f"Skewness:                  {metrics.skewness:.3f}")
    print(f"Kurtosis:                  {metrics.kurtosis:.3f}")
    print(f"Sample Size:               {metrics.sample_size}")
    print(f"Number of Trials:          {metrics.n_trials}")
    print(f"Number of Parameters:      {metrics.n_parameters}")
    print("="*60)
    print("TEST RESULTS:")
    print(f"  DSR Test (>0.5):         {'PASS' if metrics.passed_dsr_test else 'FAIL'}")
    print(f"  PBO Test (<0.5):         {'PASS' if metrics.passed_pbo_test else 'FAIL'}")
    print(f"  Complexity Test (<2.0):  {'PASS' if metrics.passed_complexity_test else 'FAIL'}")
    print("="*60)
    
    overall_pass = metrics.passed_dsr_test and metrics.passed_pbo_test and metrics.passed_complexity_test
    print(f"\nOVERALL: {'STRATEGY APPEARS ROBUST' if overall_pass else 'WARNING: POTENTIAL OVERFITTING'}")


if __name__ == "__main__":
    main()
