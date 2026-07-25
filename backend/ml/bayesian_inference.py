"""
Bayesian Inference for Order Flow Analysis
Updates prior probabilities with new market data using conjugate priors.
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
class BayesianConfig:
    """Configuration for Bayesian inference parameters."""
    # Prior parameters (Beta distribution for proportions)
    prior_alpha: float = 1.0  # Prior successes
    prior_beta: float = 1.0   # Prior failures
    
    # Prior parameters (Normal-Inverse-Gamma for means)
    prior_mean: float = 0.0
    prior_kappa: float = 1.0   # Confidence in prior mean
    prior_alpha_nig: float = 1.0  # Shape for variance
    prior_beta_nig: float = 1.0   # Scale for variance
    
    # Decay factor for older observations
    decay_factor: float = 0.995
    
    # Credible interval level
    credible_level: float = 0.95
    
    # Maximum history for numerical stability
    max_history: int = 10000


@dataclass
class PosteriorState:
    """Current posterior distribution parameters."""
    # Beta posterior (for probabilities)
    alpha: float = 1.0
    beta: float = 1.0
    
    # Normal-Inverse-Gamma posterior (for means and variances)
    mean: float = 0.0
    kappa: float = 1.0
    alpha_nig: float = 1.0
    beta_nig: float = 1.0
    
    # Sufficient statistics (for incremental updates)
    sum_x: float = 0.0
    sum_x2: float = 0.0
    n_samples: int = 0


class BayesianInference:
    """
    Bayesian Inference Engine for Order Flow Analysis.
    
    Implements:
    - Beta-Binomial updating for directional probability
    - Normal-Inverse-Gamma updating for return distributions
    - Exponential decay for concept drift adaptation
    - Credible interval computation for uncertainty quantification
    
    All updates are O(1) incremental operations.
    """
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.state = PosteriorState()
        
        # Initialize with priors
        self.reset()
        
        # History for monitoring
        self.posterior_history: deque = deque(maxlen=config.max_history)
        self.credible_intervals: deque = deque(maxlen=100)
        
        logger.info(f"BayesianInference initialized: prior=({config.prior_alpha}, {config.prior_beta})")
    
    def reset(self) -> None:
        """Reset to prior state."""
        self.state = PosteriorState(
            alpha=self.config.prior_alpha,
            beta=self.config.prior_beta,
            mean=self.config.prior_mean,
            kappa=self.config.prior_kappa,
            alpha_nig=self.config.prior_alpha_nig,
            beta_nig=self.config.prior_beta_nig,
            sum_x=0.0,
            sum_x2=0.0,
            n_samples=0,
        )
        self.posterior_history.clear()
        self.credible_intervals.clear()
    
    def update_directional(
        self,
        success: bool,
        weight: float = 1.0
    ) -> None:
        """
        Update Beta posterior for directional probability.
        
        Args:
            success: Whether the observation was a "success" (e.g., price went up)
            weight: Weight for this observation (for importance weighting)
        """
        # Apply decay to old sufficient statistics (concept drift)
        decay = self.config.decay_factor
        
        if success:
            self.state.alpha = decay * self.state.alpha + weight
        else:
            self.state.beta = decay * self.state.beta + weight
        
        # Store updated posterior
        self._store_posterior()
    
    def update_continuous(
        self,
        observation: float,
        weight: float = 1.0
    ) -> None:
        """
        Update Normal-Inverse-Gamma posterior for continuous observations.
        
        Args:
            observation: Continuous value (e.g., return, order flow imbalance)
            weight: Weight for this observation
        """
        # Apply decay
        decay = self.config.decay_factor
        
        # Update sufficient statistics with decay
        self.state.sum_x = decay * self.state.sum_x + weight * observation
        self.state.sum_x2 = decay * self.state.sum_x2 + weight * observation ** 2
        self.state.n_samples = int(decay * self.state.n_samples) + 1
        
        # Update NIG parameters
        n_eff = self.state.n_samples
        
        # Posterior mean (precision-weighted average)
        self.state.kappa = decay * self.state.kappa + weight
        self.state.mean = self.state.sum_x / (self.state.kappa + 1e-8)
        
        # Posterior shape and scale for variance
        self.state.alpha_nig = decay * self.state.alpha_nig + weight / 2
        
        ss = self.state.sum_x2 - (self.state.sum_x ** 2) / (self.state.kappa + 1e-8)
        self.state.beta_nig = decay * self.state.beta_nig + ss / 2
        
        # Store updated posterior
        self._store_posterior()
    
    def _store_posterior(self) -> None:
        """Store current posterior state for monitoring."""
        self.posterior_history.append({
            'alpha': self.state.alpha,
            'beta': self.state.beta,
            'mean': self.state.mean,
            'variance': self.get_variance(),
            'n_samples': self.state.n_samples,
        })
    
    def get_directional_probability(self) -> float:
        """Get posterior probability of success (directional bet)."""
        return self.state.alpha / (self.state.alpha + self.state.beta + 1e-8)
    
    def get_directional_uncertainty(self) -> float:
        """Get uncertainty (variance) of directional probability."""
        a, b = self.state.alpha, self.state.beta
        total = a + b
        return (a * b) / (total ** 2 * (total + 1) + 1e-8)
    
    def get_mean_estimate(self) -> float:
        """Get posterior mean estimate for continuous variable."""
        return self.state.mean
    
    def get_variance(self) -> float:
        """Get posterior variance estimate."""
        if self.state.alpha_nig <= 0:
            return float('inf')
        return self.state.beta_nig / (self.state.alpha_nig + 1e-8)
    
    def get_credible_interval(
        self,
        level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute credible interval for directional probability.
        
        Uses Beta distribution quantiles.
        
        Args:
            level: Credible level (default from config)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        level = level or self.config.credible_level
        alpha_tail = (1 - level) / 2
        
        # Use Beta distribution percent point function
        # Approximation using normal approximation for speed
        p = self.get_directional_probability()
        var = self.get_directional_uncertainty()
        std = np.sqrt(var)
        
        # Z-score for credible level
        z = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }.get(level, 1.96)
        
        lower = max(0.0, p - z * std)
        upper = min(1.0, p + z * std)
        
        # Store for monitoring
        self.credible_intervals.append((lower, upper))
        
        return lower, upper
    
    def get_posterior_predictive(self, n_simulations: int = 1000) -> np.ndarray:
        """
        Generate samples from posterior predictive distribution.
        
        Useful for simulating future outcomes under uncertainty.
        
        Args:
            n_simulations: Number of samples to generate
        
        Returns:
            Array of simulated outcomes
        """
        # Sample from Beta posterior
        theta_samples = np.random.beta(
            self.state.alpha,
            self.state.beta,
            size=n_simulations
        )
        
        # Sample from Normal-Inverse-Gamma for continuous predictions
        # First sample variance from Inverse-Gamma
        var_samples = np.random.gamma(
            self.state.alpha_nig,
            1.0 / (self.state.beta_nig + 1e-8),
            size=n_simulations
        )
        var_samples = 1.0 / (var_samples + 1e-8)  # Convert to inverse-gamma
        
        # Then sample mean from Normal
        mean_samples = np.random.normal(
            self.state.mean,
            np.sqrt(var_samples)
        )
        
        return theta_samples, mean_samples
    
    def compute_bayes_factor(
        self,
        alternative_hypothesis: float,
        null_hypothesis: float = 0.5
    ) -> float:
        """
        Compute Bayes factor comparing two hypotheses.
        
        Args:
            alternative_hypothesis: Probability under H1
            null_hypothesis: Probability under H0
        
        Returns:
            Bayes factor (BF > 1 supports H1)
        """
        p = self.get_directional_probability()
        
        # Likelihood of data under each hypothesis
        # Using Beta-binomial likelihood
        n = self.state.alpha + self.state.beta
        k = self.state.alpha
        
        # Log-likelihoods to avoid underflow
        log_likelihood_h1 = (
            k * np.log(alternative_hypothesis + 1e-8) +
            (n - k) * np.log(1 - alternative_hypothesis + 1e-8)
        )
        
        log_likelihood_h0 = (
            k * np.log(null_hypothesis + 1e-8) +
            (n - k) * np.log(1 - null_hypothesis + 1e-8)
        )
        
        # Bayes factor (exponentiated difference)
        bf = np.exp(log_likelihood_h1 - log_likelihood_h0)
        
        return min(bf, 1e6)  # Cap at reasonable value
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """Get comprehensive summary of posterior state."""
        ci_lower, ci_upper = self.get_credible_interval()
        
        return {
            'directional_prob': self.get_directional_probability(),
            'directional_uncertainty': self.get_directional_uncertainty(),
            'credible_interval_lower': ci_lower,
            'credible_interval_upper': ci_upper,
            'credible_interval_width': ci_upper - ci_lower,
            'mean_estimate': self.get_mean_estimate(),
            'variance_estimate': self.get_variance(),
            'std_estimate': np.sqrt(self.get_variance()),
            'effective_samples': self.state.n_samples,
            'alpha': self.state.alpha,
            'beta': self.state.beta,
        }
    
    def should_trade(
        self,
        threshold: float = 0.6,
        min_confidence: float = 0.8
    ) -> Tuple[bool, str]:
        """
        Determine if Bayesian evidence supports trading.
        
        Args:
            threshold: Minimum probability threshold for direction
            min_confidence: Minimum confidence (1 - uncertainty) required
        
        Returns:
            Tuple of (should_trade, reason)
        """
        prob = self.get_directional_probability()
        uncertainty = self.get_directional_uncertainty()
        confidence = 1 - uncertainty
        
        ci_lower, ci_upper = self.get_credible_interval()
        
        # Check if probability exceeds threshold
        if prob < threshold and prob > (1 - threshold):
            return False, "Probability too close to 0.5 (no edge)"
        
        # Check confidence
        if confidence < min_confidence:
            return False, f"Confidence too low: {confidence:.2%}"
        
        # Check credible interval excludes 0.5
        if ci_lower <= 0.5 <= ci_upper:
            return False, "Credible interval includes 0.5 (uncertain)"
        
        direction = "BUY" if prob > 0.5 else "SELL"
        return True, f"{direction} signal: prob={prob:.2%}, confidence={confidence:.2%}"


# Example usage and testing
if __name__ == "__main__":
    config = BayesianConfig(
        prior_alpha=1.0,
        prior_beta=1.0,
        decay_factor=0.995,
    )
    
    inference = BayesianInference(config)
    
    # Simulate order flow with positive bias
    np.random.seed(42)
    for i in range(100):
        # 60% chance of upward movement
        success = np.random.random() < 0.6
        inference.update_directional(success)
        
        # Also update continuous returns
        ret = np.random.normal(0.001, 0.01)
        inference.update_continuous(ret)
        
        if i % 20 == 0:
            stats = inference.get_summary_statistics()
            print(f"Step {i}: Prob={stats['directional_prob']:.3f}, "
                  f"CI=[{stats['credible_interval_lower']:.3f}, {stats['credible_interval_upper']:.3f}]")
    
    # Final analysis
    final_stats = inference.get_summary_statistics()
    print(f"\nFinal Summary: {final_stats}")
    
    should_trade, reason = inference.should_trade(threshold=0.55)
    print(f"Trade decision: {should_trade}, Reason: {reason}")
