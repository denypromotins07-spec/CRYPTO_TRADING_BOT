#!/usr/bin/env python3
"""
Volatility Clustering Detection using Hidden Markov Models (HMM)

This module detects regime shifts in volatility using hidden Markov models,
identifying transitions between low, medium, and high volatility states.
Critical for the ZAID bot to adapt trading strategies to current market regimes.

Features:
- Multi-state HMM for volatility regime detection
- Online parameter updates for non-stationary crypto markets
- Regime transition probability tracking
- Early warning system for volatility spikes

Designed for 8GB RAM constraint with efficient numpy operations.
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Dict
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
import warnings

# Suppress convergence warnings for production stability
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class VolatilityRegime(Enum):
    """Enum representing different volatility regimes."""
    LOW = 0       # Calm market, trend-following strategies work
    MEDIUM = 1    # Normal conditions, balanced approach
    HIGH = 2      # Turbulent market, mean-reversion preferred
    EXTREME = 3   # Crisis mode, reduce exposure


@dataclass
class HMMState:
    """Represents a single HMM state (volatility regime)."""
    regime: VolatilityRegime
    mean_return: float
    variance: float
    probability: float  # Current probability of being in this state
    
    @property
    def volatility(self) -> float:
        return np.sqrt(self.variance)


@dataclass
class TransitionMatrix:
    """Markov transition matrix between volatility regimes."""
    matrix: np.ndarray  # Shape: (n_states, n_states)
    state_names: List[VolatilityRegime] = field(default_factory=list)
    
    def __post_init__(self):
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Transition matrix must be square")
        # Ensure rows sum to 1
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.matrix = self.matrix / row_sums
    
    def get_transition_probability(self, from_state: VolatilityRegime, 
                                    to_state: VolatilityRegime) -> float:
        """Get probability of transitioning from one state to another."""
        from_idx = self.state_names.index(from_state)
        to_idx = self.state_names.index(to_state)
        return self.matrix[from_idx, to_idx]
    
    def most_likely_next_state(self, current_state: VolatilityRegime) -> VolatilityRegime:
        """Find the most likely next state given current state."""
        from_idx = self.state_names.index(current_state)
        to_idx = np.argmax(self.matrix[from_idx])
        return self.state_names[to_idx]


class VolatilityClusteringDetector:
    """
    Detects volatility clustering and regime shifts using Hidden Markov Models.
    
    This implementation uses a simplified Baum-Welch algorithm for online
    parameter updates, optimized for cryptocurrency market dynamics.
    
    Attributes:
        n_states: Number of hidden volatility states
        states: List of HMMState objects
        transition_matrix: Markov transition probabilities
        emission_means: Mean returns for each state
        emission_vars: Variances for each state
        history_window: Size of rolling window for estimation
    """
    
    def __init__(self, n_states: int = 4, history_window: int = 252):
        """
        Initialize the volatility clustering detector.
        
        Args:
            n_states: Number of volatility regimes (default 4: LOW/MEDIUM/HIGH/EXTREME)
            history_window: Rolling window size for parameter estimation
        """
        self.n_states = n_states
        self.history_window = history_window
        
        # Initialize state enums
        self.state_enums = list(VolatilityRegime)[:n_states]
        
        # State parameters (will be updated online)
        self.emission_means = np.zeros(n_states)
        self.emission_vars = np.ones(n_states) * 0.0004  # Initial variance guess
        
        # Transition matrix (initialized with persistence bias)
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Forward probabilities for current observation
        self.forward_probs = np.ones(n_states) / n_states
        
        # History storage
        self.returns_history: List[float] = []
        self.state_history: List[int] = []
        
        # Regime change detection
        self.previous_regime: Optional[VolatilityRegime] = None
        self.regime_change_points: List[Tuple[int, VolatilityRegime, VolatilityRegime]] = []
        
    def _initialize_transition_matrix(self) -> np.ndarray:
        """
        Initialize transition matrix with high diagonal (persistence).
        
        Crypto volatility tends to persist in regimes, so we initialize
        with high self-transition probabilities.
        """
        # Start with 80% persistence, 20% distributed among other states
        matrix = np.eye(self.n_states) * 0.8
        off_diag = 0.2 / (self.n_states - 1)
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j:
                    matrix[i, j] = off_diag
        
        return matrix
    
    def update(self, return_value: float, timestamp: int) -> VolatilityRegime:
        """
        Update the HMM with a new return observation and detect current regime.
        
        Args:
            return_value: Latest log return
            timestamp: Current timestamp for tracking
            
        Returns:
            Current detected volatility regime
        """
        # Store return in history
        self.returns_history.append(return_value)
        if len(self.returns_history) > self.history_window:
            self.returns_history.pop(0)
        
        # Perform E-step: calculate forward probabilities
        self._forward_step(return_value)
        
        # Determine most likely current state
        current_state_idx = int(np.argmax(self.forward_probs))
        current_regime = self.state_enums[current_state_idx]
        
        # Track regime changes
        if self.previous_regime is not None and current_regime != self.previous_regime:
            self.regime_change_points.append((
                timestamp, 
                self.previous_regime, 
                current_regime
            ))
        
        self.previous_regime = current_regime
        self.state_history.append(current_state_idx)
        
        # Periodically update parameters (every 50 observations)
        if len(self.returns_history) % 50 == 0 and len(self.returns_history) >= 100:
            self._update_parameters_online()
        
        return current_regime
    
    def _forward_step(self, observation: float):
        """
        Perform forward step of forward-backward algorithm.
        
        Calculates P(state_t | observations_1:t) recursively.
        """
        if len(self.returns_history) < 2:
            return
        
        # Calculate emission probabilities (Gaussian)
        emission_probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            mean = self.emission_means[i]
            var = max(self.emission_vars[i], 1e-10)
            # Gaussian PDF
            emission_probs[i] = (1.0 / np.sqrt(2 * np.pi * var)) * \
                               np.exp(-0.5 * (observation - mean) ** 2 / var)
        
        # Normalize emission probabilities
        emission_sum = emission_probs.sum()
        if emission_sum > 0:
            emission_probs /= emission_sum
        
        # Forward recursion: alpha_t = A^T * alpha_{t-1} * emission
        new_forward = self.transition_matrix.T @ self.forward_probs * emission_probs
        
        # Normalize
        forward_sum = new_forward.sum()
        if forward_sum > 0:
            self.forward_probs = new_forward / forward_sum
        else:
            self.forward_probs = np.ones(self.n_states) / self.n_states
    
    def _update_parameters_online(self):
        """
        Update HMM parameters using recent data (online learning).
        
        Uses a simplified approach based on state probabilities and 
        recent observations to avoid full Baum-Welch complexity.
        """
        if len(self.returns_history) < 50:
            return
        
        returns_array = np.array(self.returns_history[-100:])  # Use last 100 returns
        
        # Simple k-means-like update based on volatility levels
        volatilities = np.abs(returns_array)
        
        # Sort and divide into quantiles for state initialization
        sorted_vol = np.sort(volatilities)
        n_per_state = len(sorted_vol) // self.n_states
        
        for i in range(self.n_states):
            start_idx = i * n_per_state
            end_idx = (i + 1) * n_per_state if i < self.n_states - 1 else len(sorted_vol)
            
            state_returns = returns_array[np.argsort(np.abs(returns_array))[start_idx:end_idx]]
            
            if len(state_returns) > 0:
                self.emission_means[i] = np.mean(state_returns)
                self.emission_vars[i] = max(np.var(state_returns), 1e-6)
        
        # Update transition matrix based on observed transitions
        self._update_transition_matrix()
    
    def _update_transition_matrix(self):
        """Update transition matrix based on observed state transitions."""
        if len(self.state_history) < 10:
            return
        
        # Count transitions
        counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, len(self.state_history)):
            from_state = self.state_history[t-1]
            to_state = self.state_history[t]
            counts[from_state, to_state] += 1
        
        # Add smoothing to avoid zero probabilities
        counts += 0.1
        
        # Normalize to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = counts / row_sums
    
    def get_current_regime_probabilities(self) -> Dict[VolatilityRegime, float]:
        """Get the probability distribution over current regimes."""
        return {
            state: float(prob) 
            for state, prob in zip(self.state_enums, self.forward_probs)
        }
    
    def get_expected_regime_duration(self, regime: VolatilityRegime) -> float:
        """
        Calculate expected duration (in periods) for a given regime.
        
        Expected duration = 1 / (1 - p_ii) where p_ii is self-transition probability.
        """
        idx = self.state_enums.index(regime)
        self_transition_prob = self.transition_matrix[idx, idx]
        
        if self_transition_prob >= 1.0:
            return float('inf')
        
        return 1.0 / (1.0 - self_transition_prob)
    
    def is_regime_change_imminent(self, threshold: float = 0.3) -> bool:
        """
        Check if a regime change is likely in the next period.
        
        Args:
            threshold: Probability threshold for considering a change imminent
            
        Returns:
            True if probability of leaving current regime exceeds threshold
        """
        current_idx = int(np.argmax(self.forward_probs))
        stay_prob = self.transition_matrix[current_idx, current_idx]
        return (1.0 - stay_prob) > threshold
    
    def get_volatility_forecast(self, steps_ahead: int = 10) -> float:
        """
        Forecast expected volatility N steps ahead.
        
        Uses the stationary distribution of the Markov chain.
        """
        # Find stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary_dist = np.real(eigenvectors[:, idx])
        stationary_dist = stationary_dist / stationary_dist.sum()
        
        # Weighted average of state volatilities
        expected_vol = np.sum(stationary_dist * np.sqrt(self.emission_vars))
        
        return expected_vol
    
    def get_clustering_statistics(self) -> Dict:
        """
        Get statistics about volatility clustering behavior.
        
        Returns:
            Dictionary with clustering metrics
        """
        if len(self.regime_change_points) < 2:
            return {"clustering_strength": 0.0, "avg_regime_duration": 0.0}
        
        # Calculate durations between regime changes
        durations = []
        for i in range(1, len(self.regime_change_points)):
            duration = self.regime_change_points[i][0] - self.regime_change_points[i-1][0]
            durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else 0.0
        
        # Clustering strength: higher avg duration = stronger clustering
        clustering_strength = min(avg_duration / 20.0, 1.0)  # Normalize to [0, 1]
        
        return {
            "clustering_strength": clustering_strength,
            "avg_regime_duration": avg_duration,
            "total_regime_changes": len(self.regime_change_points),
            "current_regime": self.previous_regime.name if self.previous_regime else "UNKNOWN"
        }


# Example usage and testing
if __name__ == "__main__":
    # Simulate some returns with volatility clustering
    np.random.seed(42)
    n_obs = 500
    
    # Generate returns with changing volatility regimes
    returns = []
    current_vol = 0.01
    for i in range(n_obs):
        # Change volatility regime every 100 periods
        if i % 100 == 0:
            current_vol = [0.005, 0.02, 0.05, 0.10][i // 100 % 4]
        
        returns.append(np.random.normal(0, current_vol))
    
    # Test the detector
    detector = VolatilityClusteringDetector(n_states=4)
    
    print("Volatility Clustering Detection Test")
    print("=" * 50)
    
    for i, ret in enumerate(returns):
        regime = detector.update(ret, timestamp=i)
        
        if i % 100 == 0 or i == len(returns) - 1:
            probs = detector.get_current_regime_probabilities()
            print(f"\nPeriod {i}:")
            print(f"  Current Regime: {regime.name}")
            print(f"  Probabilities: {probs}")
    
    # Get final statistics
    stats = detector.get_clustering_statistics()
    print(f"\nClustering Statistics:")
    print(f"  Strength: {stats['clustering_strength']:.2f}")
    print(f"  Avg Duration: {stats['avg_regime_duration']:.1f} periods")
    print(f"  Total Changes: {stats['total_regime_changes']}")
