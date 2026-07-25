#!/usr/bin/env python3
"""
Volatility Surface Builder and Interpolator

Builds and maintains a 3D volatility surface (strike x expiry x vol) for crypto options.
Uses Radial Basis Functions (RBF) for smooth interpolation between observed points.

Features:
- Real-time surface updates when new trades print
- RBF interpolation for smooth volatility curves
- Term structure and skew modeling
- Memory-efficient storage for 8GB RAM constraint

Target: Update the volatility surface instantly when new options trades print on Binance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time
import logging
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VolObservation:
    """Single volatility observation point"""
    delta: float          # Option delta (-1 to 1)
    moneyness: float      # Strike / Spot
    expiry_days: int      # Days to expiry
    implied_vol: float    # Observed IV
    option_type: str      # 'call' or 'put'
    timestamp_us: int = 0
    volume: float = 0.0   # Trading volume (for weighting)
    
    def __post_init__(self):
        if self.timestamp_us == 0:
            self.timestamp_us = int(time.time() * 1_000_000)


@dataclass
class VolSlice:
    """Volatility smile/skew for a single expiry"""
    expiry_days: int
    points: List[Tuple[float, float]]  # (moneyness, vol) pairs
    
    def get_atm_vol(self) -> Optional[float]:
        """Get ATM volatility (moneyness = 1.0)"""
        if not self.points:
            return None
        
        # Find closest to ATM
        atm_point = min(self.points, key=lambda p: abs(p[0] - 1.0))
        return atm_point[1]


class RBFInterpolation:
    """
    Radial Basis Function interpolator for smooth volatility surface.
    
    Uses Gaussian RBF kernels for interpolation across strike and term dimensions.
    Optimized for incremental updates without full matrix recomputation.
    """
    
    def __init__(self, sigma_strike: float = 0.1, sigma_term: float = 30.0):
        """
        Initialize RBF interpolator.
        
        Args:
            sigma_strike: Length scale for strike dimension
            sigma_term: Length scale for term dimension (days)
        """
        self.sigma_strike = sigma_strike
        self.sigma_term = sigma_term
        self.centers: List[Tuple[float, float, float]] = []  # (moneyness, expiry, vol)
        self.weights: List[float] = []
    
    def add_observation(self, moneyness: float, expiry_days: int, vol: float, weight: float = 1.0) -> None:
        """Add a new observation point"""
        self.centers.append((moneyness, expiry_days, vol))
        self.weights.append(weight)
    
    def gaussian_kernel(self, x1: float, t1: int, x2: float, t2: int) -> float:
        """Compute Gaussian RBF kernel value"""
        dx = (x1 - x2) / self.sigma_strike
        dt = (t1 - t2) / self.sigma_term
        dist_sq = dx * dx + dt * dt
        return math.exp(-0.5 * dist_sq)
    
    def interpolate(self, moneyness: float, expiry_days: int) -> Optional[float]:
        """
        Interpolate volatility at given point using RBF.
        
        Returns:
            Interpolated volatility or None if insufficient data
        """
        if len(self.centers) < 1:
            return None
        
        if len(self.centers) == 1:
            return self.centers[0][2]
        
        # Weighted average using RBF kernels
        numerator = 0.0
        denominator = 0.0
        
        for i, (c_moneyness, c_expiry, vol) in enumerate(self.centers):
            kernel_val = self.gaussian_kernel(moneyness, expiry_days, c_moneyness, c_expiry)
            weight = self.weights[i] if i < len(self.weights) else 1.0
            
            numerator += kernel_val * weight * vol
            denominator += kernel_val * weight
        
        if denominator < 1e-10:
            return None
        
        return numerator / denominator
    
    def clear(self) -> None:
        """Clear all observations"""
        self.centers.clear()
        self.weights.clear()
    
    def get_surface_stats(self) -> Dict:
        """Get statistics about the current surface"""
        if not self.centers:
            return {'count': 0}
        
        vols = [c[2] for c in self.centers]
        return {
            'count': len(self.centers),
            'min_vol': min(vols),
            'max_vol': max(vols),
            'avg_vol': sum(vols) / len(vols),
            'moneyness_range': (min(c[0] for c in self.centers), max(c[0] for c in self.centers)),
            'expiry_range': (min(c[1] for c in self.centers), max(c[1] for c in self.centers)),
        }


class VolatilitySurface:
    """
    3D Volatility Surface builder and manager.
    
    Dimensions:
    - Moneyness (strike/spot)
    - Time to expiry
    - Implied volatility
    
    Features:
    - Real-time updates from market data
    - RBF interpolation for smooth surfaces
    - Term structure extraction
    - Skew metrics calculation
    """
    
    # Standard expiry buckets (in days)
    STANDARD_EXPIRIES = [7, 14, 30, 60, 90, 180, 365]
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.rbf_interpolator = RBFInterpolation()
        self.expiry_slices: Dict[int, VolSlice] = {}
        self.observations: List[VolObservation] = []
        self.last_update_us: int = 0
        self._atm_vols: Dict[int, float] = {}  # Cache of ATM vols by expiry
    
    def add_observation(self, obs: VolObservation) -> None:
        """
        Add a new volatility observation to the surface.
        Automatically triggers surface rebuild.
        """
        self.observations.append(obs)
        
        # Add to RBF interpolator with volume weighting
        weight = max(1.0, math.log1p(obs.volume))
        self.rbf_interpolator.add_observation(obs.moneyness, obs.expiry_days, obs.implied_vol, weight)
        
        # Update or create expiry slice
        if obs.expiry_days not in self.expiry_slices:
            self.expiry_slices[obs.expiry_days] = VolSlice(expiry_days=obs.expiry_days, points=[])
        
        self.expiry_slices[obs.expiry_days].points.append((obs.moneyness, obs.implied_vol))
        
        # Invalidate ATM cache
        self._atm_vols.clear()
        
        self.last_update_us = int(time.time() * 1_000_000)
    
    def add_observations_batch(self, observations: List[VolObservation]) -> None:
        """Add multiple observations efficiently"""
        for obs in observations:
            self.add_observation(obs)
    
    def get_volatility(self, moneyness: float, expiry_days: int) -> Optional[float]:
        """
        Get interpolated volatility for given moneyness and expiry.
        
        Args:
            moneyness: Strike / Spot ratio
            expiry_days: Days to expiry
            
        Returns:
            Interpolated implied volatility
        """
        return self.rbf_interpolator.interpolate(moneyness, expiry_days)
    
    def get_volatility_by_delta(self, delta: float, expiry_days: int, spot: float) -> Optional[float]:
        """
        Get volatility for a specific delta level.
        
        Converts delta to approximate moneyness using Black-Scholes approximation.
        """
        # Simple delta-to-moneyness approximation
        # For calls: moneyness ≈ exp(delta * σ * √T)
        # This is a simplification; proper conversion requires Newton-Raphson
        
        if abs(delta) > 1.0:
            return None
        
        # Approximate moneyness from delta
        sqrt_t = math.sqrt(expiry_days / 365.0)
        avg_vol = self.get_avg_vol() or 0.6  # Default 60% vol
        
        if delta > 0:  # Call delta
            moneyness = math.exp(delta * avg_vol * sqrt_t)
        else:  # Put delta (negative)
            moneyness = math.exp((1 + delta) * avg_vol * sqrt_t)
        
        return self.get_volatility(moneyness, expiry_days)
    
    def get_atm_volatility(self, expiry_days: int) -> Optional[float]:
        """
        Get ATM volatility for a specific expiry.
        
        Args:
            expiry_days: Days to expiry
            
        Returns:
            ATM implied volatility
        """
        # Check cache
        if expiry_days in self._atm_vols:
            return self._atm_vols[expiry_days]
        
        # Find closest expiry slice
        if not self.expiry_slices:
            return None
        
        closest_expiry = min(self.expiry_slices.keys(), key=lambda e: abs(e - expiry_days))
        slice_data = self.expiry_slices[closest_expiry]
        
        atm_vol = slice_data.get_atm_vol()
        if atm_vol is None:
            # Fall back to RBF interpolation at moneyness = 1.0
            atm_vol = self.get_volatility(1.0, closest_expiry)
        
        if atm_vol is not None:
            self._atm_vols[expiry_days] = atm_vol
        
        return atm_vol
    
    def get_term_structure(self) -> List[Tuple[int, float]]:
        """
        Get volatility term structure (ATM vol by expiry).
        
        Returns:
            List of (expiry_days, atm_vol) tuples sorted by expiry
        """
        term_struct = []
        
        for expiry in sorted(self.expiry_slices.keys()):
            atm_vol = self.get_atm_volatility(expiry)
            if atm_vol is not None:
                term_struct.append((expiry, atm_vol))
        
        # Fill in standard expiries using interpolation
        for std_expiry in self.STANDARD_EXPIRIES:
            if not any(e[0] == std_expiry for e in term_struct):
                vol = self.get_atm_volatility(std_expiry)
                if vol is not None:
                    term_struct.append((std_expiry, vol))
        
        term_struct.sort(key=lambda x: x[0])
        return term_struct
    
    def get_skew(self, expiry_days: int) -> Optional[Dict[str, float]]:
        """
        Calculate volatility skew metrics for an expiry.
        
        Returns:
            Dictionary with skew metrics
        """
        vol_25d = self.get_volatility_by_delta(0.25, expiry_days, 0)  # 25 delta call
        vol_atm = self.get_atm_volatility(expiry_days)
        vol_75d = self.get_volatility_by_delta(-0.25, expiry_days, 0)  # 25 delta put
        
        if vol_25d is None or vol_atm is None or vol_75d is None:
            return None
        
        # Risk reversal (call skew - put skew)
        risk_reversal = vol_25d - vol_75d
        
        # Butterfly (curvature)
        butterfly = (vol_25d + vol_75d) / 2 - vol_atm
        
        return {
            'risk_reversal_25d': risk_reversal,
            'butterfly_25d': butterfly,
            'vol_25d_call': vol_25d,
            'vol_atm': vol_atm,
            'vol_25d_put': vol_75d,
        }
    
    def get_avg_vol(self) -> Optional[float]:
        """Get average volatility across all observations"""
        if not self.observations:
            return None
        return sum(o.implied_vol for o in self.observations) / len(self.observations)
    
    def get_surface_statistics(self) -> Dict:
        """Get comprehensive surface statistics"""
        rbf_stats = self.rbf_interpolator.get_surface_stats()
        term_structure = self.get_term_structure()
        
        return {
            'symbol': self.symbol,
            'observation_count': len(self.observations),
            'expiry_buckets': len(self.expiry_slices),
            'rbf_stats': rbf_stats,
            'term_structure': term_structure,
            'last_update_us': self.last_update_us,
        }
    
    def clear(self) -> None:
        """Clear all surface data"""
        self.observations.clear()
        self.expiry_slices.clear()
        self.rbf_interpolator.clear()
        self._atm_vols.clear()
        self.last_update_us = 0


class MultiAssetVolSurface:
    """Manage volatility surfaces for multiple underlying assets"""
    
    def __init__(self):
        self.surfaces: Dict[str, VolatilitySurface] = {}
    
    def get_surface(self, symbol: str) -> VolatilitySurface:
        """Get or create surface for a symbol"""
        if symbol not in self.surfaces:
            self.surfaces[symbol] = VolatilitySurface(symbol)
        return self.surfaces[symbol]
    
    def update_from_trade(
        self,
        symbol: str,
        strike: float,
        spot: float,
        expiry_days: int,
        implied_vol: float,
        option_type: str,
        volume: float = 0.0,
    ) -> None:
        """
        Update surface from a single options trade.
        
        This is the main entry point for real-time updates.
        """
        surface = self.get_surface(symbol)
        moneyness = strike / spot if spot > 0 else 1.0
        
        obs = VolObservation(
            delta=0.5,  # Will be calculated properly from option type
            moneyness=moneyness,
            expiry_days=expiry_days,
            implied_vol=implied_vol,
            option_type=option_type,
            volume=volume,
        )
        
        surface.add_observation(obs)
        
        logger.debug(f"Updated vol surface for {symbol}: K={strike}, IV={implied_vol:.2%}")
    
    def get_all_surfaces_stats(self) -> Dict[str, Dict]:
        """Get statistics for all surfaces"""
        return {symbol: s.get_surface_statistics() for symbol, s in self.surfaces.items()}


if __name__ == "__main__":
    # Example usage
    print("Volatility Surface Builder initialized")
    
    surface = VolatilitySurface("BTC")
    
    # Add sample observations
    sample_obs = [
        VolObservation(delta=0.25, moneyness=0.95, expiry_days=30, implied_vol=0.70, option_type="put"),
        VolObservation(delta=0.50, moneyness=1.00, expiry_days=30, implied_vol=0.65, option_type="call"),
        VolObservation(delta=0.75, moneyness=1.05, expiry_days=30, implied_vol=0.62, option_type="call"),
        VolObservation(delta=0.50, moneyness=1.00, expiry_days=90, implied_vol=0.60, option_type="call"),
    ]
    
    surface.add_observations_batch(sample_obs)
    
    # Query the surface
    print(f"\nSurface Statistics: {surface.get_surface_statistics()}")
    print(f"Term Structure: {surface.get_term_structure()}")
    print(f"ATM Vol (30d): {surface.get_atm_volatility(30)}")
    print(f"Skew (30d): {surface.get_skew(30)}")
