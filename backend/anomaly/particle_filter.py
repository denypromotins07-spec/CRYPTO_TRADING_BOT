#!/usr/bin/env python3
"""
Particle Filter Module: Sequential Monte Carlo for Hidden State Tracking

This module implements particle filtering algorithms for tracking hidden
market states such as liquidity regimes, order flow imbalance, and latent
volatility factors. Essential for filtering noisy crypto market data.

Key Features:
- Bootstrap particle filter with resampling
- Adaptive resampling based on effective sample size
- Regularized particle filter for smooth density estimation
- Memory-efficient implementation for streaming data
- Integration with quantitative finance state-space models

Author: ZAID Personal Crypto Trading Bot - Stage 29
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import multivariate_normal, gaussian_kde
import warnings


@dataclass
class Particle:
    """Represents a single particle in the filter."""
    state: np.ndarray
    weight: float = 1.0
    
    def copy(self) -> 'Particle':
        return Particle(state=self.state.copy(), weight=self.weight)


@dataclass
class ParticleFilterResult:
    """Result of particle filter update."""
    timestamp_ms: int
    estimated_state: np.ndarray
    state_covariance: np.ndarray
    effective_sample_size: float
    resampling_performed: bool
    particles: List[Particle]
    log_likelihood: float


@dataclass
class StateSpaceModel:
    """
    Linear Gaussian state-space model for particle filtering.
    
    x_t = F @ x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t = H @ x_t + v_t,      v_t ~ N(0, R)
    """
    # State transition matrix
    F: np.ndarray
    # Observation matrix
    H: np.ndarray
    # Process noise covariance
    Q: np.ndarray
    # Observation noise covariance
    R: np.ndarray
    # Initial state mean
    x0: np.ndarray
    # Initial state covariance
    P0: np.ndarray


class BootstrapParticleFilter:
    """
    Standard bootstrap particle filter with adaptive resampling.
    
    This implementation uses systematic resampling when the effective
    sample size falls below a threshold, preventing sample degeneracy.
    """
    
    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        ess_threshold: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize the particle filter.
        
        Args:
            model: State-space model definition
            n_particles: Number of particles to use
            ess_threshold: Threshold for resampling (fraction of n_particles)
            seed: Random seed for reproducibility
        """
        self.model = model
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold * n_particles
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize particles from prior distribution
        self.particles = self._initialize_particles()
        self.weights = np.ones(n_particles) / n_particles
        self._iteration = 0
    
    def _initialize_particles(self) -> List[Particle]:
        """Initialize particles from the prior distribution."""
        particles = []
        for _ in range(self.n_particles):
            state = np.random.multivariate_normal(
                self.model.x0, 
                self.model.P0
            )
            particles.append(Particle(state=state))
        return particles
    
    def _calculate_effective_sample_size(self) -> float:
        """Calculate effective sample size (ESS) from weights."""
        # ESS = 1 / sum(w_i^2)
        return 1.0 / np.sum(self.weights ** 2)
    
    def _systematic_resampling(self) -> List[Particle]:
        """
        Perform systematic resampling to reduce variance.
        
        This method is preferred over multinomial resampling as it
        produces lower variance estimates.
        """
        n = self.n_particles
        
        # Compute cumulative sum of weights
        cumsum = np.cumsum(self.weights)
        
        # Generate starting point
        u0 = np.random.uniform(0, 1/n)
        u = u0 + np.arange(n) / n
        
        # Resample
        new_particles = []
        i, j = 0, 0
        while i < n:
            if u[i] < cumsum[j]:
                new_particles.append(self.particles[j].copy())
                new_particles[-1].weight = 1.0 / n
                i += 1
            else:
                j += 1
        
        # Reset weights uniformly
        self.weights = np.ones(n) / n
        
        return new_particles
    
    def _resample_if_needed(self) -> bool:
        """Check if resampling is needed and perform it."""
        ess = self._calculate_effective_sample_size()
        
        if ess < self.ess_threshold:
            self.particles = self._systematic_resampling()
            return True
        return False
    
    def update(
        self, 
        timestamp_ms: int, 
        observation: np.ndarray
    ) -> ParticleFilterResult:
        """
        Update the filter with a new observation.
        
        Args:
            timestamp_ms: Timestamp of the observation
            observation: New observation vector
            
        Returns:
            Filter result with updated state estimate
        """
        self._iteration += 1
        
        # Prediction step: propagate particles through state transition
        for i, particle in enumerate(self.particles):
            # x_t = F @ x_{t-1} + w_t
            predicted_state = self.model.F @ particle.state
            process_noise = np.random.multivariate_normal(
                np.zeros(len(predicted_state)),
                self.model.Q
            )
            particle.state = predicted_state + process_noise
        
        # Update step: reweight particles based on likelihood
        log_weights = np.zeros(self.n_particles)
        
        for i, particle in enumerate(self.particles):
            # y_t = H @ x_t + v_t
            predicted_observation = self.model.H @ particle.state
            
            # Calculate log-likelihood under observation noise
            try:
                log_lik = multivariate_normal.logpdf(
                    observation,
                    mean=predicted_observation,
                    cov=self.model.R
                )
            except Exception:
                log_lik = -1e10
            
            log_weights[i] = log_lik
        
        # Normalize weights using log-sum-exp trick for numerical stability
        max_log_weight = np.max(log_weights)
        log_weights_centered = log_weights - max_log_weight
        weights_unnormalized = np.exp(log_weights_centered)
        self.weights = weights_unnormalized / np.sum(weights_unnormalized)
        
        # Calculate log-likelihood of observation
        log_likelihood = max_log_weight + np.log(np.mean(weights_unnormalized))
        
        # Resample if needed
        resampling_performed = self._resample_if_needed()
        
        # Estimate state and covariance
        estimated_state = np.zeros_like(self.particles[0].state)
        for particle, weight in zip(self.particles, self.weights):
            estimated_state += weight * particle.state
        
        state_covariance = np.zeros((len(estimated_state), len(estimated_state)))
        for particle, weight in zip(self.particles, self.weights):
            diff = particle.state - estimated_state
            state_covariance += weight * np.outer(diff, diff)
        
        ess = self._calculate_effective_sample_size()
        
        return ParticleFilterResult(
            timestamp_ms=timestamp_ms,
            estimated_state=estimated_state,
            state_covariance=state_covariance,
            resampling_performed=resampling_performed,
            particles=self.particles.copy(),
            log_likelihood=log_likelihood,
            effective_sample_size=ess
        )
    
    def get_state_estimate(self) -> np.ndarray:
        """Get current state estimate."""
        estimate = np.zeros_like(self.particles[0].state)
        for particle, weight in zip(self.particles, self.weights):
            estimate += weight * particle.state
        return estimate
    
    def get_state_uncertainty(self) -> np.ndarray:
        """Get current state covariance."""
        estimate = self.get_state_estimate()
        covariance = np.zeros((len(estimate), len(estimate)))
        
        for particle, weight in zip(self.particles, self.weights):
            diff = particle.state - estimate
            covariance += weight * np.outer(diff, diff)
        
        return covariance
    
    def reset(self):
        """Reset the filter to initial state."""
        self.particles = self._initialize_particles()
        self.weights = np.ones(self.n_particles) / self.n_particles
        self._iteration = 0


class RegularizedParticleFilter(BootstrapParticleFilter):
    """
    Regularized particle filter using kernel density estimation.
    
    This variant adds regularization to prevent sample impoverishment
    by smoothing the posterior density estimate.
    """
    
    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        ess_threshold: float = 0.5,
        bandwidth_factor: float = 1.0,
        seed: Optional[int] = None
    ):
        super().__init__(model, n_particles, ess_threshold, seed)
        self.bandwidth_factor = bandwidth_factor
    
    def _regularized_resampling(self) -> List[Particle]:
        """
        Perform regularized resampling using kernel smoothing.
        """
        n = self.n_particles
        dim = len(self.particles[0].state)
        
        # Compute weighted mean and covariance
        mean = self.get_state_estimate()
        cov = self.get_state_uncertainty()
        
        # Optimal bandwidth (Silverman's rule adapted for weighted samples)
        bandwidth = self.bandwidth_factor * n ** (-1 / (dim + 4))
        
        # Resample with kernel perturbation
        new_particles = []
        
        # Systematic resampling first
        cumsum = np.cumsum(self.weights)
        u0 = np.random.uniform(0, 1/n)
        u = u0 + np.arange(n) / n
        
        indices = []
        i, j = 0, 0
        while i < n:
            if u[i] < cumsum[j]:
                indices.append(j)
                i += 1
            else:
                j += 1
        
        # Add kernel perturbation
        for idx in indices:
            base_state = self.particles[idx].state.copy()
            
            # Sample from kernel
            perturbation = np.random.multivariate_normal(
                np.zeros(dim),
                cov * (bandwidth ** 2)
            )
            
            new_state = base_state + perturbation
            new_particles.append(Particle(state=new_state, weight=1.0/n))
        
        self.weights = np.ones(n) / n
        return new_particles
    
    def _resample_if_needed(self) -> bool:
        """Override to use regularized resampling."""
        ess = self._calculate_effective_sample_size()
        
        if ess < self.ess_threshold:
            self.particles = self._regularized_resampling()
            return True
        return False


class AuxiliaryParticleFilter(BootstrapParticleFilter):
    """
    Auxiliary particle filter for improved proposal distribution.
    
    Uses look-ahead information to guide particle placement,
    particularly useful when observations are highly informative.
    """
    
    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        ess_threshold: float = 0.5,
        seed: Optional[int] = None
    ):
        super().__init__(model, n_particles, ess_threshold, seed)
    
    def update(
        self, 
        timestamp_ms: int, 
        observation: np.ndarray
    ) -> ParticleFilterResult:
        """Update using auxiliary particle filter algorithm."""
        self._iteration += 1
        
        # First stage: compute likelihoods at predicted means
        predicted_means = []
        for particle in self.particles:
            pred_mean = self.model.H @ (self.model.F @ particle.state)
            predicted_means.append(pred_mean)
        
        # Compute first-stage weights (likelihood at predicted mean)
        first_stage_weights = np.zeros(self.n_particles)
        for i, pred_mean in enumerate(predicted_means):
            try:
                first_stage_weights[i] = multivariate_normal.pdf(
                    observation,
                    mean=pred_mean,
                    cov=self.model.H @ self.model.Q @ self.model.H.T + self.model.R
                )
            except Exception:
                first_stage_weights[i] = 1e-10
        
        # Normalize first-stage weights
        first_stage_weights /= np.sum(first_stage_weights)
        
        # Resample based on first-stage weights
        cumsum = np.cumsum(first_stage_weights)
        u0 = np.random.uniform(0, 1/self.n_particles)
        u = u0 + np.arange(self.n_particles) / self.n_particles
        
        indices = []
        i, j = 0, 0
        while i < self.n_particles:
            if u[i] < cumsum[j]:
                indices.append(j)
                i += 1
            else:
                j += 1
        
        # Second stage: propagate resampled particles and update weights
        new_particles = []
        log_weights = np.zeros(self.n_particles)
        
        for i, idx in enumerate(indices):
            # Propagate
            parent_state = self.particles[idx].state
            predicted_state = self.model.F @ parent_state
            process_noise = np.random.multivariate_normal(
                np.zeros(len(predicted_state)),
                self.model.Q
            )
            new_state = predicted_state + process_noise
            
            # Compute second-stage weight (likelihood ratio)
            predicted_obs = self.model.H @ new_state
            try:
                log_lik_new = multivariate_normal.logpdf(
                    observation,
                    mean=predicted_obs,
                    cov=self.model.R
                )
                log_lik_pred = multivariate_normal.logpdf(
                    observation,
                    mean=self.model.H @ predicted_state,
                    cov=self.model.H @ self.model.Q @ self.model.H.T + self.model.R
                )
                log_weights[i] = log_lik_new - log_lik_pred
            except Exception:
                log_weights[i] = -1e10
            
            new_particles.append(Particle(state=new_state))
        
        self.particles = new_particles
        
        # Normalize weights
        max_log_weight = np.max(log_weights)
        weights_unnormalized = np.exp(log_weights - max_log_weight)
        self.weights = weights_unnormalized / np.sum(weights_unnormalized)
        
        log_likelihood = max_log_weight + np.log(np.sum(weights_unnormalized)) - np.log(self.n_particles)
        
        # Resample if needed
        resampling_performed = self._resample_if_needed()
        
        # Estimate state and covariance
        estimated_state = np.zeros_like(self.particles[0].state)
        for particle, weight in zip(self.particles, self.weights):
            estimated_state += weight * particle.state
        
        state_covariance = np.zeros((len(estimated_state), len(estimated_state)))
        for particle, weight in zip(self.particles, self.weights):
            diff = particle.state - estimated_state
            state_covariance += weight * np.outer(diff, diff)
        
        ess = self._calculate_effective_sample_size()
        
        return ParticleFilterResult(
            timestamp_ms=timestamp_ms,
            estimated_state=estimated_state,
            state_covariance=state_covariance,
            resampling_performed=resampling_performed,
            particles=self.particles.copy(),
            log_likelihood=log_likelihood,
            effective_sample_size=ess
        )


class MarketStateTracker:
    """
    Specialized particle filter for tracking market microstructure states.
    
    Tracks hidden states such as:
    - Order flow imbalance
    - Liquidity regime
    - Latent volatility
    - Informed trader presence
    """
    
    def __init__(self, n_particles: int = 2000, seed: Optional[int] = None):
        """
        Initialize market state tracker.
        
        State vector: [order_flow_imbalance, liquidity_regime, log_volatility, informed_prob]
        """
        # Define state-space model for market microstructure
        dim_state = 4
        dim_obs = 3  # Observed: volume, spread, realized vol
        
        # State transition (persistent states with mean reversion)
        F = np.array([
            [0.8, 0.0, 0.0, 0.0],   # Order flow AR(1)
            [0.0, 0.9, 0.0, 0.0],   # Liquidity regime persistent
            [0.0, 0.0, 0.95, 0.0],  # Volatility persistent
            [0.0, 0.0, 0.0, 0.7],   # Informed trader prob mean-reverting
        ])
        
        # Observation matrix
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],   # Volume related to order flow
            [0.0, 1.0, 0.0, 0.0],   # Spread related to liquidity
            [0.0, 0.0, 1.0, 0.0],   # Realized vol = latent vol
        ])
        
        # Noise covariances
        Q = np.diag([0.01, 0.005, 0.001, 0.02])
        R = np.diag([0.1, 0.05, 0.02])
        
        # Initial state
        x0 = np.array([0.0, 0.5, -2.0, 0.1])  # Neutral initial beliefs
        P0 = np.eye(dim_state) * 0.5
        
        model = StateSpaceModel(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
        
        self.filter = BootstrapParticleFilter(
            model=model,
            n_particles=n_particles,
            seed=seed
        )
        
        self._history: List[ParticleFilterResult] = []
    
    def update(
        self,
        timestamp_ms: int,
        normalized_volume: float,
        normalized_spread: float,
        realized_volatility: float
    ) -> ParticleFilterResult:
        """
        Update market state estimate with new observations.
        
        Args:
            timestamp_ms: Observation timestamp
            normalized_volume: Volume normalized by recent average
            normalized_spread: Bid-ask spread normalized by price
            realized_volatility: Recent realized volatility
            
        Returns:
            Updated state estimate
        """
        observation = np.array([
            normalized_volume,
            normalized_spread,
            realized_volatility
        ])
        
        result = self.filter.update(timestamp_ms, observation)
        self._history.append(result)
        
        # Keep history bounded
        if len(self._history) > 10000:
            self._history = self._history[-5000:]
        
        return result
    
    def get_market_regime(self) -> str:
        """Infer current market regime from state estimate."""
        state = self.filter.get_state_estimate()
        liquidity = state[1]
        vol = np.exp(state[2])
        
        if liquidity > 0.7 and vol < 0.02:
            return "CALM_LIQUID"
        elif liquidity > 0.7 and vol >= 0.02:
            return "VOLATILE_LIQUID"
        elif liquidity <= 0.7 and vol < 0.05:
            return "CALM_ILLIQUID"
        else:
            return "STRESS"
    
    def get_informed_trading_probability(self) -> float:
        """Get probability of informed trader presence."""
        state = self.filter.get_state_estimate()
        return np.clip(state[3], 0, 1)
    
    def get_order_flow_imbalance(self) -> float:
        """Get estimated order flow imbalance."""
        state = self.filter.get_state_estimate()
        return state[0]


def main():
    """Example usage of particle filter for market state tracking."""
    np.random.seed(42)
    
    # Create tracker
    tracker = MarketStateTracker(n_particles=1000, seed=42)
    
    # Simulate some market data
    print("Simulating market state tracking...")
    
    for t in range(100):
        # Generate synthetic observations
        true_vol = 0.02 + 0.01 * np.sin(t / 10)
        true_liquidity = 0.5 + 0.3 * np.cos(t / 20)
        true_order_flow = 0.1 * np.sin(t / 5)
        
        obs_volume = true_order_flow + np.random.normal(0, 0.1)
        obs_spread = true_liquidity + np.random.normal(0, 0.05)
        obs_vol = np.log(true_vol + 0.01) + np.random.normal(0, 0.1)
        
        # Update tracker
        result = tracker.update(
            timestamp_ms=t * 60000,
            normalized_volume=obs_volume,
            normalized_spread=obs_spread,
            realized_volatility=obs_vol
        )
        
        if t % 20 == 0:
            regime = tracker.get_market_regime()
            informed_prob = tracker.get_informed_trading_probability()
            print(f"t={t}: Regime={regime}, Informed Prob={informed_prob:.2%}, ESS={result.effective_sample_size:.0f}")
    
    print("\nFinal state estimate:")
    final_state = tracker.filter.get_state_estimate()
    print(f"  Order Flow Imbalance: {final_state[0]:.4f}")
    print(f"  Liquidity Regime: {final_state[1]:.4f}")
    print(f"  Log Volatility: {final_state[2]:.4f}")
    print(f"  Informed Trader Prob: {final_state[3]:.4f}")


if __name__ == "__main__":
    main()
