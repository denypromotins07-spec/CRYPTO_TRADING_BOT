"""
State Transition Matrix Calculator for Volatility Regimes
Computes and analyzes transition probabilities between market states.
Optimized for 8GB RAM with incremental updates.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransitionConfig:
    """Configuration for transition matrix calculation."""
    # Number of states (regimes)
    n_states: int = 3  # Bull, Bear, Chop
    
    # Decay factor for older transitions
    decay_factor: float = 0.99
    
    # Minimum observations per state pair
    min_observations: int = 10
    
    # Smoothing parameter (Laplace smoothing)
    smoothing_alpha: float = 1.0
    
    # History window for regime detection
    regime_window: int = 50
    
    # Maximum history to store
    max_history: int = 10000


@dataclass
class RegimeTransition:
    """Single transition observation."""
    from_state: int
    to_state: int
    timestamp: float
    duration: int  # Time spent in from_state


class StateTransitionMatrix:
    """
    State Transition Matrix Calculator for Volatility Regimes.
    
    Implements:
    - Incremental transition counting with decay
    - Laplace smoothing for sparse transitions
    - Expected duration calculation per regime
    - Steady-state distribution computation
    - Regime persistence analysis
    
    All updates are O(1) incremental operations.
    """
    
    def __init__(self, config: TransitionConfig):
        self.config = config
        self.n_states = config.n_states
        
        # Transition count matrix (with decay)
        self.transition_counts: np.ndarray = np.full(
            (self.n_states, self.n_states),
            config.smoothing_alpha,  # Laplace smoothing
            dtype=np.float64
        )
        
        # Raw counts (for reference)
        self.raw_counts: np.ndarray = np.zeros(
            (self.n_states, self.n_states),
            dtype=np.int64
        )
        
        # Current state tracking
        self.current_state: Optional[int] = None
        self.state_entry_time: Optional[float] = None
        self.state_duration: int = 0
        
        # State history
        self.state_history: deque = deque(maxlen=config.max_history)
        self.transitions: deque = deque(maxlen=config.max_history)
        
        # Duration tracking per state
        self.state_durations: Dict[int, List[int]] = defaultdict(list)
        
        # Total weighted count per state (for normalization)
        self.state_totals: np.ndarray = np.ones(self.n_states, dtype=np.float64) * config.smoothing_alpha * self.n_states
        
        logger.info(f"StateTransitionMatrix initialized: n_states={self.n_states}")
    
    def reset(self) -> None:
        """Reset all counters to initial state."""
        self.transition_counts.fill(self.config.smoothing_alpha)
        self.raw_counts.fill(0)
        self.current_state = None
        self.state_entry_time = None
        self.state_duration = 0
        self.state_history.clear()
        self.transitions.clear()
        self.state_durations.clear()
        self.state_totals.fill(self.config.smoothing_alpha * self.n_states)
    
    def update(
        self,
        new_state: int,
        timestamp: float
    ) -> Optional[RegimeTransition]:
        """
        Update transition matrix with new state observation.
        
        Args:
            new_state: Current regime state (0 to n_states-1)
            timestamp: Current timestamp
        
        Returns:
            RegimeTransition if state changed, None otherwise
        """
        if not (0 <= new_state < self.n_states):
            logger.warning(f"Invalid state: {new_state}")
            return None
        
        # Store in history
        self.state_history.append(new_state)
        
        # First observation
        if self.current_state is None:
            self.current_state = new_state
            self.state_entry_time = timestamp
            self.state_duration = 1
            return None
        
        # Same state - increment duration
        if new_state == self.current_state:
            self.state_duration += 1
            return None
        
        # State transition detected
        old_state = self.current_state
        duration = self.state_duration
        
        # Create transition record
        transition = RegimeTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=timestamp,
            duration=duration
        )
        
        # Store duration
        self.state_durations[old_state].append(duration)
        if len(self.state_durations[old_state]) > 1000:
            self.state_durations[old_state].pop(0)
        
        # Update transition counts with decay
        self._apply_decay()
        self._increment_transition(old_state, new_state)
        
        # Store transition
        self.transitions.append(transition)
        
        # Update current state
        self.current_state = new_state
        self.state_entry_time = timestamp
        self.state_duration = 1
        
        return transition
    
    def _apply_decay(self) -> None:
        """Apply exponential decay to all transition counts."""
        decay = self.config.decay_factor
        self.transition_counts *= decay
        self.state_totals *= decay
    
    def _increment_transition(self, from_state: int, to_state: int) -> None:
        """Increment transition count."""
        self.transition_counts[from_state, to_state] += 1.0
        self.raw_counts[from_state, to_state] += 1
        self.state_totals[from_state] += 1.0
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get normalized transition probability matrix.
        
        Returns:
            Matrix P where P[i,j] = P(state_j | state_i)
        """
        # Normalize rows
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        
        return self.transition_counts / row_sums
    
    def get_steady_state_distribution(self) -> np.ndarray:
        """
        Compute steady-state (stationary) distribution.
        
        Solves: pi * P = pi where sum(pi) = 1
        
        Returns:
            Stationary distribution vector
        """
        P = self.get_transition_matrix()
        
        # Power iteration method
        pi = np.ones(self.n_states) / self.n_states
        
        for _ in range(100):  # Max iterations
            pi_new = pi @ P
            
            # Check convergence
            if np.max(np.abs(pi_new - pi)) < 1e-8:
                break
            
            pi = pi_new
        
        # Ensure normalization
        return pi / pi.sum()
    
    def get_expected_duration(self, state: int) -> float:
        """
        Get expected duration in a given state.
        
        E[duration] = 1 / (1 - P[state,state])
        
        Args:
            state: State index
        
        Returns:
            Expected number of time steps in state
        """
        P = self.get_transition_matrix()
        p_self = P[state, state]
        
        if p_self >= 1.0:
            return float('inf')
        
        return 1.0 / (1.0 - p_self + 1e-8)
    
    def get_all_expected_durations(self) -> Dict[str, float]:
        """Get expected durations for all states."""
        state_names = ['Bull', 'Bear', 'Chop'][:self.n_states]
        
        return {
            state_names[i]: self.get_expected_duration(i)
            for i in range(self.n_states)
        }
    
    def get_persistence_probability(self, state: int) -> float:
        """Get probability of staying in the same state."""
        P = self.get_transition_matrix()
        return P[state, state]
    
    def get_regime_switching_rate(self) -> float:
        """
        Get overall regime switching rate.
        
        Returns:
            Average probability of transitioning to different state
        """
        P = self.get_transition_matrix()
        
        # Average off-diagonal probability
        switching_probs = []
        for i in range(self.n_states):
            off_diag = 1.0 - P[i, i]
            switching_probs.append(off_diag)
        
        return np.mean(switching_probs)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary of transition dynamics."""
        P = self.get_transition_matrix()
        steady_state = self.get_steady_state_distribution()
        expected_durations = self.get_all_expected_durations()
        
        state_names = ['Bull', 'Bear', 'Chop'][:self.n_states]
        
        # Build summary
        summary = {
            'transition_matrix': P.tolist(),
            'steady_state_distribution': dict(zip(state_names, steady_state)),
            'expected_durations': expected_durations,
            'regime_switching_rate': self.get_regime_switching_rate(),
            'total_transitions': int(self.raw_counts.sum()),
            'current_state': state_names[self.current_state] if self.current_state is not None else None,
            'current_duration': self.state_duration,
        }
        
        # Add per-state statistics
        for i, name in enumerate(state_names):
            summary[f'{name}_persistence'] = self.get_persistence_probability(i)
            
            # Duration statistics
            if self.state_durations[i]:
                durations = self.state_durations[i]
                summary[f'{name}_mean_duration'] = float(np.mean(durations))
                summary[f'{name}_std_duration'] = float(np.std(durations))
                summary[f'{name}_max_duration'] = int(np.max(durations))
            else:
                summary[f'{name}_mean_duration'] = 0.0
                summary[f'{name}_std_duration'] = 0.0
                summary[f'{name}_max_duration'] = 0
        
        return summary
    
    def predict_next_state_probabilities(self, current_state: int) -> np.ndarray:
        """
        Get probabilities for next state given current state.
        
        Args:
            current_state: Current state index
        
        Returns:
            Array of probabilities for each possible next state
        """
        P = self.get_transition_matrix()
        return P[current_state]
    
    def is_regime_change_significant(
        self,
        from_state: int,
        to_state: int,
        threshold: float = 0.1
    ) -> bool:
        """
        Check if a regime change is statistically significant.
        
        Args:
            from_state: Starting state
            to_state: Ending state
            threshold: Minimum probability threshold
        
        Returns:
            True if transition probability exceeds threshold
        """
        P = self.get_transition_matrix()
        return P[from_state, to_state] > threshold
    
    def get_mixing_time(self) -> float:
        """
        Estimate mixing time (time to reach steady state).
        
        Based on second eigenvalue of transition matrix.
        
        Returns:
            Estimated number of steps to converge to steady state
        """
        P = self.get_transition_matrix()
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(P)
        
        # Sort by magnitude (excluding eigenvalue = 1)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        if len(eigenvalues) < 2:
            return float('inf')
        
        # Second largest eigenvalue
        lambda_2 = eigenvalues[1] if eigenvalues[0] > 0.99 else eigenvalues[0]
        
        if lambda_2 >= 1.0:
            return float('inf')
        
        # Mixing time approximation
        return -1.0 / np.log(lambda_2 + 1e-8)


# Example usage and testing
if __name__ == "__main__":
    config = TransitionConfig(
        n_states=3,
        decay_factor=0.99,
        smoothing_alpha=1.0,
    )
    
    calculator = StateTransitionMatrix(config)
    
    # Simulate regime sequence
    np.random.seed(42)
    
    # Generate synthetic regime sequence with persistence
    regimes = [0]  # Start in Bull
    for _ in range(200):
        current = regimes[-1]
        
        # High persistence probability
        if np.random.random() < 0.85:
            next_regime = current
        else:
            # Switch to random other state
            other_states = [s for s in range(3) if s != current]
            next_regime = np.random.choice(other_states)
        
        regimes.append(next_regime)
    
    # Feed to calculator
    for i, regime in enumerate(regimes):
        transition = calculator.update(regime, timestamp=float(i))
        
        if transition and i % 20 == 0:
            print(f"Step {i}: Transition {transition.from_state} -> {transition.to_state}, "
                  f"Duration: {transition.duration}")
    
    # Get summary
    summary = calculator.get_summary_statistics()
    print(f"\nTransition Matrix:")
    for row in summary['transition_matrix']:
        print([f"{p:.3f}" for p in row])
    
    print(f"\nSteady State: {summary['steady_state_distribution']}")
    print(f"Expected Durations: {summary['expected_durations']}")
    print(f"Mixing Time: {calculator.get_mixing_time():.2f} steps")
