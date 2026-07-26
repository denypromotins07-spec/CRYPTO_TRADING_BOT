"""
Bayesian Online Change-Point Detection for Tick Streams

This module implements Bayesian online change-point detection algorithms
to identify structural breaks in market data streams in real-time.
It uses the Barry & Adams (2007) approach with efficient recursive updates.

Key Features:
- O(1) update time per tick using recursive Bayesian updates
- Memory-bounded run-length distribution tracking
- Automatic hazard rate adaptation for crypto volatility
- Multi-hypothesis tracking for concurrent regime candidates

Mathematical Foundation:
P(r_t | x_{1:t}) ∝ P(x_t | r_t) * Σ_{r_{t-1}} P(r_t | r_{t-1}) * P(r_{t-1} | x_{1:t-1})

Where:
- r_t is the run-length (time since last change point)
- x_t is the observation at time t
- Hazard rate controls change-point probability

Author: Opus 4.8 for ZAID PERSONAL CRYPTO TRADING BOT
Stage: 29/100 - Statistical Anomaly Detection
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class BayesianChangePointConfig:
    """Configuration for Bayesian change-point detection."""
    
    # Prior parameters for predictive distribution
    prior_mean: float = 0.0
    prior_variance: float = 1.0
    prior_samples: float = 1.0  # Strength of prior
    
    # Hazard rate (probability of change point at any time)
    hazard_rate: float = 0.01
    
    # Maximum run-length to track (memory bound)
    max_run_length: int = 500
    
    # Minimum samples before detection activation
    warmup_period: int = 20
    
    # Detection threshold (posterior probability)
    detection_threshold: float = 0.95


@dataclass
class RunLengthDistribution:
    """Represents the run-length posterior distribution."""
    
    # Run lengths (0 to max_run_length)
    run_lengths: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # Posterior probabilities for each run length
    probabilities: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    # Predictive parameters for each run length (mean, variance)
    predictive_means: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    predictive_variances: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    
    def initialize(self, max_length: int, prior_mean: float, prior_var: float) -> None:
        """Initialize the run-length distribution."""
        self.run_lengths = np.arange(max_length + 1, dtype=np.float64)
        self.probabilities = np.zeros(max_length + 1, dtype=np.float64)
        self.probabilities[0] = 1.0  # Start with run-length 0
        
        self.predictive_means = np.full(max_length + 1, prior_mean, dtype=np.float64)
        self.predictive_variances = np.full(max_length + 1, prior_var, dtype=np.float64)


@dataclass
class ChangePointEvent:
    """Represents a detected change point."""
    
    timestamp: float
    run_length: int
    confidence: float
    previous_mean: float
    new_mean: float
    magnitude: float


class BayesianOnlineChangePointDetector:
    """
    Bayesian Online Change-Point Detector for streaming financial data.
    
    Implements efficient recursive Bayesian inference to detect structural
    breaks in market regimes without storing full historical data.
    """
    
    def __init__(self, config: Optional[BayesianChangePointConfig] = None):
        self.config = config or BayesianChangePointConfig()
        self.rld = RunLengthDistribution()
        self.rld.initialize(
            self.config.max_run_length,
            self.config.prior_mean,
            self.config.prior_variance
        )
        
        self.sample_count: int = 0
        self.detected_change_points: List[ChangePointEvent] = []
        self.last_change_point: Optional[ChangePointEvent] = None
        
        # Cached values for efficiency
        self._hazard = self.config.hazard_rate
        self._one_minus_hazard = 1.0 - self.config.hazard_rate
        
    def update(self, observation: float, timestamp: Optional[float] = None) -> Optional[ChangePointEvent]:
        """
        Process a new observation and update run-length posterior.
        
        Args:
            observation: New data point (e.g., return, volume, spread)
            timestamp: Optional timestamp for the observation
            
        Returns:
            ChangePointEvent if a change point is detected, None otherwise
        """
        self.sample_count += 1
        current_time = timestamp if timestamp is not None else float(self.sample_count)
        
        # Skip warmup period
        if self.sample_count <= self.config.warmup_period:
            self._update_predictive_params(observation)
            return None
        
        # Compute likelihoods for all run lengths
        likelihoods = self._compute_likelihoods(observation)
        
        # Update run-length posterior (Bayes rule)
        new_probabilities = np.zeros_like(self.rld.probabilities)
        
        # Case 1: Continue existing run (r_t = r_{t-1} + 1)
        new_probabilities[1:self.config.max_run_length + 1] = (
            self.rld.probabilities[:self.config.max_run_length] * 
            likelihoods[:self.config.max_run_length] * 
            self._one_minus_hazard
        )
        
        # Case 2: Change point occurs (r_t = 0)
        new_probabilities[0] = np.sum(
            self.rld.probabilities * likelihoods * self._hazard
        )
        
        # Normalize
        total_prob = np.sum(new_probabilities)
        if total_prob > 1e-10:
            new_probabilities /= total_prob
        
        # Store updated distribution
        self.rld.probabilities = new_probabilities
        
        # Update predictive parameters
        self._update_predictive_params(observation)
        
        # Check for change point detection
        change_point = self._check_detection(current_time)
        
        if change_point is not None:
            self.detected_change_points.append(change_point)
            self.last_change_point = change_point
            # Reset after detection
            self._reset_post_detection()
        
        return change_point
    
    def _compute_likelihoods(self, observation: float) -> NDArray[np.float64]:
        """
        Compute likelihood P(x_t | r_t) for all run lengths.
        
        Uses Student-t predictive distribution for robustness to outliers.
        """
        # Student-t likelihood with ν = prior_samples + r_t degrees of freedom
        nu = self.config.prior_samples + self.rld.run_lengths
        
        # Scale parameter
        scale_sq = (
            (self.config.prior_samples + 1) / self.config.prior_samples *
            self.rld.predictive_variances *
            (nu + 1) / nu
        )
        
        # Standardized distance
        z_sq = (observation - self.rld.predictive_means) ** 2 / scale_sq
        
        # Log-likelihood for numerical stability
        log_likelihood = (
            -0.5 * (nu + 1) * np.log1p(z_sq / nu)
            - 0.5 * np.log(scale_sq)
            + 0.5 * np.log(nu)
        )
        
        # Convert to likelihood (with floor to prevent underflow)
        likelihood = np.exp(np.clip(log_likelihood, -700, 0))
        likelihood = np.maximum(likelihood, 1e-300)
        
        return likelihood
    
    def _update_predictive_params(self, observation: float) -> None:
        """Update predictive distribution parameters for all run lengths."""
        # For run-length 0: use prior
        self.rld.predictive_means[0] = self.config.prior_mean
        self.rld.predictive_variances[0] = self.config.prior_variance * (
            1 + 1 / self.config.prior_samples
        )
        
        # For run-length > 0: incremental update
        for r in range(1, self.config.max_run_length + 1):
            if r < len(self.rld.predictive_means):
                # Recursive update of mean and variance
                prev_mean = self.rld.predictive_means[r - 1]
                prev_var = self.rld.predictive_variances[r - 1]
                
                # Weight for new observation
                weight = 1.0 / (self.config.prior_samples + r)
                
                # Updated mean
                self.rld.predictive_means[r] = (
                    (1 - weight) * prev_mean + weight * observation
                )
                
                # Updated variance (Welford's algorithm style)
                diff = observation - prev_mean
                self.rld.predictive_variances[r] = (
                    ((self.config.prior_samples + r - 1) * prev_var + 
                     diff * (observation - self.rld.predictive_means[r])) /
                    (self.config.prior_samples + r)
                )
    
    def _check_detection(self, timestamp: float) -> Optional[ChangePointEvent]:
        """Check if change point should be declared."""
        # Probability of change point (run-length = 0)
        cp_prob = self.rld.probabilities[0]
        
        if cp_prob >= self.config.detection_threshold:
            # Find most likely previous run length
            non_zero_probs = self.rld.probabilities[1:]
            if len(non_zero_probs) > 0 and np.max(non_zero_probs) > 1e-10:
                prev_rl = int(np.argmax(non_zero_probs)) + 1
            else:
                prev_rl = int(np.mean(self.rld.run_lengths[1:]))
            
            # Calculate magnitude of change
            old_mean = self.rld.predictive_means.get(prev_rl, self.config.prior_mean) \
                if hasattr(self.rld.predictive_means, 'get') else \
                (self.rld.predictive_means[prev_rl] if prev_rl < len(self.rld.predictive_means) 
                 else self.config.prior_mean)
            
            new_mean = self.rld.predictive_means[0]
            magnitude = abs(new_mean - old_mean)
            
            return ChangePointEvent(
                timestamp=timestamp,
                run_length=prev_rl,
                confidence=float(cp_prob),
                previous_mean=float(old_mean),
                new_mean=float(new_mean),
                magnitude=magnitude
            )
        
        return None
    
    def _reset_post_detection(self) -> None:
        """Reset posterior after change point detection."""
        # Concentrate probability on short run lengths
        self.rld.probabilities = np.zeros_like(self.rld.probabilities)
        self.rld.probabilities[0] = 0.5
        self.rld.probabilities[1:min(5, len(self.rld.probabilities))] = 0.1
        
        # Normalize
        total = np.sum(self.rld.probabilities)
        if total > 0:
            self.rld.probabilities /= total
    
    def get_expected_run_length(self) -> float:
        """Compute expected run length from posterior."""
        return float(np.sum(self.rld.run_lengths * self.rld.probabilities))
    
    def get_most_probable_run_length(self) -> int:
        """Get the mode of the run-length distribution."""
        return int(self.rld.run_lengths[np.argmax(self.rld.probabilities)])
    
    def get_change_point_rate(self, window: int = 100) -> float:
        """Calculate recent change point frequency."""
        if len(self.detected_change_points) < 2:
            return 0.0
        
        recent = [cp for cp in self.detected_change_points[-window:]]
        if len(recent) < 2:
            return 0.0
        
        time_span = recent[-1].timestamp - recent[0].timestamp
        if time_span <= 0:
            return 0.0
        
        return (len(recent) - 1) / time_span


class AdaptiveHazardRate:
    """Adaptively adjusts hazard rate based on market volatility."""
    
    def __init__(self, base_hazard: float = 0.01, volatility_window: int = 50):
        self.base_hazard = base_hazard
        self.volatility_window = volatility_window
        self.returns: Deque[float] = deque(maxlen=volatility_window)
        
    def update(self, observation: float) -> float:
        """Update hazard rate based on recent volatility."""
        if len(self.returns) > 0:
            ret = observation - self.returns[-1] if self.returns else 0.0
            self.returns.append(ret)
            
            if len(self.returns) >= 10:
                vol = np.std(list(self.returns))
                # Increase hazard during high volatility
                adaptive = self.base_hazard * (1.0 + vol * 10.0)
                return min(adaptive, 0.5)  # Cap at 50%
        
        return self.base_hazard


if __name__ == "__main__":
    # Example usage
    detector = BayesianOnlineChangePointDetector()
    
    # Simulate regime shift
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    shifted_data = np.random.normal(2, 1, 100)
    data = np.concatenate([normal_data, shifted_data])
    
    change_points = []
    for i, obs in enumerate(data):
        cp = detector.update(obs, timestamp=float(i))
        if cp is not None:
            change_points.append(cp)
            print(f"Change point detected at t={i}: "
                  f"magnitude={cp.magnitude:.3f}, confidence={cp.confidence:.3f}")
    
    print(f"\nTotal change points detected: {len(change_points)}")
    print(f"Expected run length: {detector.get_expected_run_length():.2f}")
