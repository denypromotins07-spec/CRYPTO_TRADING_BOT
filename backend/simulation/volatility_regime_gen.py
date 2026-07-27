"""
Volatility Regime Generator for Synthetic Stress Testing

Generates synthetic volatility regimes for comprehensive strategy testing:
- Normal/low volatility periods
- High volatility clusters
- Flash crash scenarios
- Volatility spikes and mean reversion
- Multi-asset vol correlation breakdowns

Memory-efficient implementation for 8GB RAM constraint.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "low"           # Calm markets, low vol
    NORMAL = "normal"     # Typical crypto volatility
    HIGH = "high"         # Elevated volatility
    EXTREME = "extreme"   # Crisis/flash crash levels
    CLUSTERING = "clustering"  # Vol clustering (GARCH effects)


@dataclass
class RegimeParameters:
    """Parameters for each volatility regime."""
    mean_vol: float           # Mean volatility level
    vol_of_vol: float         # Volatility of volatility
    mean_reversion_speed: float  # Speed of mean reversion
    jump_intensity: float     # Probability of volatility jump
    jump_size_mean: float     # Mean jump size
    jump_size_std: float      # Std dev of jump size
    persistence: float        # Autocorrelation of volatility
    min_vol: float            # Floor for volatility
    max_vol: float            # Ceiling for volatility
    
    @classmethod
    def low_vol(cls) -> 'RegimeParameters':
        return cls(
            mean_vol=0.0001,
            vol_of_vol=0.1,
            mean_reversion_speed=0.1,
            jump_intensity=0.001,
            jump_size_mean=0.5,
            jump_size_std=0.2,
            persistence=0.9,
            min_vol=0.00005,
            max_vol=0.001,
        )
    
    @classmethod
    def normal_vol(cls) -> 'RegimeParameters':
        return cls(
            mean_vol=0.0003,
            vol_of_vol=0.3,
            mean_reversion_speed=0.05,
            jump_intensity=0.01,
            jump_size_mean=1.0,
            jump_size_std=0.5,
            persistence=0.85,
            min_vol=0.0001,
            max_vol=0.005,
        )
    
    @classmethod
    def high_vol(cls) -> 'RegimeParameters':
        return cls(
            mean_vol=0.001,
            vol_of_vol=0.5,
            mean_reversion_speed=0.03,
            jump_intensity=0.05,
            jump_size_mean=1.5,
            jump_size_std=0.8,
            persistence=0.7,
            min_vol=0.0003,
            max_vol=0.02,
        )
    
    @classmethod
    def extreme_vol(cls) -> 'RegimeParameters':
        """Flash crash / crisis regime."""
        return cls(
            mean_vol=0.005,
            vol_of_vol=1.0,
            mean_reversion_speed=0.01,
            jump_intensity=0.2,
            jump_size_mean=3.0,
            jump_size_std=1.5,
            persistence=0.5,
            min_vol=0.001,
            max_vol=0.1,
        )


@dataclass
class VolatilityState:
    """Current state of volatility process."""
    current_vol: float
    long_term_vol: float
    recent_jumps: List[float] = field(default_factory=list)
    regime: VolatilityRegime = VolatilityRegime.NORMAL
    time_in_regime: int = 0


class VolatilityRegimeGenerator:
    """
    Generate synthetic volatility regimes for stress testing.
    
    Features:
    - Heston-like stochastic volatility
    - GARCH-style volatility clustering
    - Jump diffusion for sudden spikes
    - Regime switching with Markov transitions
    - Multi-asset vol correlation
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state = VolatilityState(
            current_vol=0.0003,
            long_term_vol=0.0003,
            regime=VolatilityRegime.NORMAL,
        )
        
        # Regime transition matrix (row = current, col = next)
        self.transition_matrix = np.array([
            # Low   Norm  High  Extr  Clust
            [0.7,  0.25, 0.04, 0.01, 0.0],   # From Low
            [0.2,  0.5,  0.2,  0.05, 0.05],  # From Normal
            [0.05, 0.2,  0.5,  0.15, 0.1],   # From High
            [0.01, 0.1,  0.3,  0.4,  0.19],  # From Extreme
            [0.1,  0.3,  0.3,  0.1,  0.2],   # From Clustering
        ])
        
        self.regime_list = list(VolatilityRegime)
        self.vol_history: List[float] = []
        self.regime_history: List[VolatilityRegime] = []
        
        # Multi-asset vol correlation
        self.asset_vol_correlation: Optional[np.ndarray] = None
    
    def get_regime_params(self, regime: VolatilityRegime) -> RegimeParameters:
        """Get parameters for specified regime."""
        mapping = {
            VolatilityRegime.LOW: RegimeParameters.low_vol(),
            VolatilityRegime.NORMAL: RegimeParameters.normal_vol(),
            VolatilityRegime.HIGH: RegimeParameters.high_vol(),
            VolatilityRegime.EXTREME: RegimeParameters.extreme_vol(),
            VolatilityRegime.CLUSTERING: RegimeParameters.normal_vol(),  # Special handling
        }
        return mapping[regime]
    
    def simulate_step(self, dt: float = 1.0) -> float:
        """
        Simulate one step of volatility evolution.
        
        Returns current volatility level.
        """
        params = self.get_regime_params(self.state.regime)
        
        # Heston-like mean-reverting square-root process
        drift = params.mean_reversion_speed * (params.mean_vol - self.state.current_vol) * dt
        
        diffusion = params.vol_of_vol * np.sqrt(self.state.current_vol) * \
                    self.rng.normal() * np.sqrt(dt)
        
        new_vol = self.state.current_vol + drift + diffusion
        
        # Check for jumps
        if self.rng.random() < params.jump_intensity:
            jump_size = params.jump_size_mean + params.jump_size_std * self.rng.normal()
            jump_size = max(jump_size, 0.1)  # Minimum 10% jump
            new_vol *= (1 + jump_size)
            
            self.state.recent_jumps.append(jump_size)
            if len(self.state.recent_jumps) > 10:
                self.state.recent_jumps.pop(0)
        
        # Apply bounds
        new_vol = np.clip(new_vol, params.min_vol, params.max_vol)
        
        # Update state
        self.state.current_vol = new_vol
        self.state.time_in_regime += 1
        
        # Possibly switch regime
        self._maybe_switch_regime()
        
        # Record history
        self.vol_history.append(new_vol)
        self.regime_history.append(self.state.regime)
        
        return new_vol
    
    def _maybe_switch_regime(self) -> None:
        """Probabilistically switch volatility regime."""
        params = self.get_regime_params(self.state.regime)
        
        # Base transition probabilities from matrix
        current_idx = self.regime_list.index(self.state.regime)
        transition_probs = self.transition_matrix[current_idx]
        
        # Adjust for time in regime (longer stay = higher switch probability)
        time_factor = min(self.state.time_in_regime / 100, 2.0)
        
        # For clustering regime, increase persistence
        if self.state.regime == VolatilityRegime.CLUSTERING:
            transition_probs[current_idx] *= 1.5
        
        # Normalize
        transition_probs = transition_probs / transition_probs.sum()
        
        # Sample next regime
        next_idx = self.rng.choice(len(self.regime_list), p=transition_probs)
        next_regime = self.regime_list[next_idx]
        
        if next_regime != self.state.regime:
            logger.debug(f"Regime switch: {self.state.regime} -> {next_regime}")
            self.state.regime = next_regime
            self.state.time_in_regime = 0
    
    def generate_path(self, n_steps: int, initial_regime: Optional[VolatilityRegime] = None) -> np.ndarray:
        """
        Generate full volatility path.
        
        Args:
            n_steps: Number of steps to simulate
            initial_regime: Starting regime
            
        Returns:
            Array of volatility values
        """
        if initial_regime is not None:
            self.state.regime = initial_regime
            self.state.time_in_regime = 0
        
        vol_path = np.zeros(n_steps)
        for i in range(n_steps):
            vol_path[i] = self.simulate_step()
        
        return vol_path
    
    def generate_flash_crash(self, duration: int = 50, severity: float = 1.0) -> np.ndarray:
        """
        Generate flash crash volatility scenario.
        
        Args:
            duration: Number of steps for the crash
            severity: Multiplier for crash intensity (0.5-2.0)
            
        Returns:
            Volatility path including crash
        """
        # Pre-crash calm
        pre_crash = self.generate_path(20, VolatilityRegime.LOW)
        
        # Crash phase - rapid vol increase
        crash_vol = []
        base_vol = self.state.current_vol
        params = RegimeParameters.extreme_vol()
        
        for i in range(duration):
            # Exponential increase then decay
            progress = i / duration
            spike_factor = np.exp(-((progress - 0.2) ** 2) / 0.05) * severity
            vol = base_vol * (1 + spike_factor * 20)
            
            # Add noise
            vol *= (1 + 0.3 * self.rng.normal())
            vol = np.clip(vol, params.min_vol, params.max_vol)
            crash_vol.append(vol)
        
        # Post-crash mean reversion
        post_crash = []
        final_crash_vol = crash_vol[-1] if crash_vol else base_vol
        for i in range(30):
            vol = final_crash_vol * np.exp(-i * 0.1) + base_vol * (1 - np.exp(-i * 0.1))
            vol *= (1 + 0.2 * self.rng.normal())
            post_crash.append(np.clip(vol, params.min_vol, params.max_vol))
        
        return np.concatenate([pre_crash, crash_vol, post_crash])
    
    def generate_vol_cluster(self, n_steps: int, cluster_intensity: float = 1.5) -> np.ndarray:
        """
        Generate volatility clustering scenario (GARCH effects).
        
        Args:
            n_steps: Number of steps
            cluster_intensity: Strength of clustering effect
            
        Returns:
            Volatility path with clustering
        """
        self.state.regime = VolatilityRegime.CLUSTERING
        params = self.get_regime_params(VolatilityRegime.CLUSTERING)
        
        vol_path = np.zeros(n_steps)
        prev_squared_return = 0.0
        
        for i in range(n_steps):
            # GARCH(1,1)-like dynamics
            omega = params.mean_vol ** 2 * (1 - 0.1 - 0.85)
            alpha = 0.1 * cluster_intensity
            beta = 0.85
            
            conditional_var = omega + alpha * prev_squared_return + beta * self.state.current_vol ** 2
            new_vol = np.sqrt(conditional_var)
            
            # Generate return for next iteration
            ret = new_vol * self.rng.normal()
            prev_squared_return = ret ** 2
            
            new_vol = np.clip(new_vol, params.min_vol, params.max_vol)
            self.state.current_vol = new_vol
            vol_path[i] = new_vol
        
        return vol_path
    
    def generate_multi_asset_vols(
        self, 
        n_assets: int, 
        n_steps: int,
        correlation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate correlated volatility paths for multiple assets.
        
        Args:
            n_assets: Number of assets
            n_steps: Simulation length
            correlation: Asset vol correlation matrix
            
        Returns:
            Shape (n_steps, n_assets) volatility array
        """
        if correlation is None:
            # Default: moderate positive correlation
            correlation = np.full((n_assets, n_assets), 0.6)
            np.fill_diagonal(correlation, 1.0)
        
        # Cholesky decomposition for correlated sampling
        chol = np.linalg.cholesky(correlation)
        
        vol_paths = np.zeros((n_steps, n_assets))
        base_params = self.get_regime_params(self.state.regime)
        
        for t in range(n_steps):
            # Generate correlated shocks
            independent = self.rng.normal(size=n_assets)
            correlated_shocks = chol @ independent
            
            # Apply to each asset with some idiosyncratic variation
            for a in range(n_assets):
                # Asset-specific parameters (slight variation)
                asset_params = RegimeParameters(
                    mean_vol=base_params.mean_vol * (1 + 0.2 * self.rng.normal()),
                    vol_of_vol=base_params.vol_of_vol,
                    mean_reversion_speed=base_params.mean_reversion_speed,
                    jump_intensity=base_params.jump_intensity,
                    jump_size_mean=base_params.jump_size_mean,
                    jump_size_std=base_params.jump_size_std,
                    persistence=base_params.persistence,
                    min_vol=base_params.min_vol,
                    max_vol=base_params.max_vol,
                )
                
                # Evolve volatility
                drift = asset_params.mean_reversion_speed * \
                        (asset_params.mean_vol - self.state.current_vol)
                diffusion = asset_params.vol_of_vol * np.sqrt(self.state.current_vol) * \
                           correlated_shocks[a]
                
                new_vol = self.state.current_vol + drift + diffusion
                
                # Jumps (correlated across assets in extreme regime)
                if self.rng.random() < asset_params.jump_intensity:
                    if self.state.regime == VolatilityRegime.EXTREME:
                        # Systemic jump - all assets jump together
                        jump = asset_params.jump_size_mean * (1 + correlated_shocks[a] * 0.5)
                    else:
                        jump = asset_params.jump_size_mean + asset_params.jump_size_std * self.rng.normal()
                    new_vol *= (1 + max(jump, 0.1))
                
                new_vol = np.clip(new_vol, asset_params.min_vol, asset_params.max_vol)
                self.state.current_vol = new_vol
                vol_paths[t, a] = new_vol
        
        return vol_paths
    
    def detect_regime_from_data(self, returns: np.ndarray, window: int = 20) -> VolatilityRegime:
        """
        Detect current volatility regime from return data.
        
        Args:
            returns: Historical returns
            window: Lookback window
            
        Returns:
            Detected regime
        """
        if len(returns) < window:
            return VolatilityRegime.NORMAL
        
        recent_returns = returns[-window:]
        realized_vol = np.std(recent_returns)
        
        # Check for extreme moves
        max_move = np.max(np.abs(recent_returns))
        
        # Classify
        if realized_vol < 0.0002:
            return VolatilityRegime.LOW
        elif realized_vol > 0.01 or max_move > 0.05:
            return VolatilityRegime.EXTREME
        elif realized_vol > 0.005:
            return VolatilityRegime.HIGH
        elif np.abs(np.mean(recent_returns)) < 0.001:
            return VolatilityRegime.CLUSTERING
        else:
            return VolatilityRegime.NORMAL
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics of generated volatility path."""
        if not self.vol_history:
            return {}
        
        vols = np.array(self.vol_history)
        return {
            'mean_vol': float(np.mean(vols)),
            'std_vol': float(np.std(vols)),
            'min_vol': float(np.min(vols)),
            'max_vol': float(np.max(vols)),
            'skewness': float(((vols - np.mean(vols)) ** 3).mean() / np.std(vols) ** 3),
            'kurtosis': float(((vols - np.mean(vols)) ** 4).mean() / np.std(vols) ** 4 - 3),
            'autocorr_1': float(np.corrcoef(vols[:-1], vols[1:])[0, 1]) if len(vols) > 1 else 0,
            'jump_count': len([j for j in self.state.recent_jumps if j > 0.5]),
        }
    
    def reset(self, initial_vol: Optional[float] = None, 
              initial_regime: VolatilityRegime = VolatilityRegime.NORMAL) -> None:
        """Reset generator state."""
        self.state = VolatilityState(
            current_vol=initial_vol or 0.0003,
            long_term_vol=initial_vol or 0.0003,
            regime=initial_regime,
        )
        self.vol_history.clear()
        self.regime_history.clear()


# Example usage
if __name__ == "__main__":
    gen = VolatilityRegimeGenerator(seed=42)
    
    # Generate normal path
    print("Generating normal volatility path...")
    normal_path = gen.generate_path(200)
    print(f"Normal path stats: mean={np.mean(normal_path):.6f}, std={np.std(normal_path):.6f}")
    
    # Generate flash crash
    print("\nGenerating flash crash scenario...")
    gen.reset()
    crash_path = gen.generate_flash_crash(duration=30, severity=1.5)
    print(f"Crash path stats: max={np.max(crash_path):.6f}, min={np.min(crash_path):.6f}")
    
    # Generate clustering
    print("\nGenerating volatility clustering...")
    gen.reset()
    cluster_path = gen.generate_vol_cluster(200, cluster_intensity=2.0)
    print(f"Cluster path autocorr: {np.corrcoef(cluster_path[:-1], cluster_path[1:])[0, 1]:.4f}")
    
    # Multi-asset
    print("\nGenerating multi-asset volatilities...")
    gen.reset()
    multi_path = gen.generate_multi_asset_vols(n_assets=4, n_steps=100)
    print(f"Multi-asset shape: {multi_path.shape}")
    print(f"Cross-asset correlation: {np.corrcoef(multi_path[:, 0], multi_path[:, 1])[0, 1]:.4f}")
    
    # Statistics
    print(f"\nFull statistics: {gen.get_statistics()}")
