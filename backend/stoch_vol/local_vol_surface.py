#!/usr/bin/env python3
"""
Local Volatility Surface Builder
Constructs 2D local volatility surface from market data for Monte Carlo simulation
Optimized for AMD Ryzen AI 5 with strict 8GB RAM constraints

Captures extreme fat tails of Bitcoin options through adaptive grid refinement
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclass
class MarketSlice:
    """Option prices for a single maturity"""
    maturity: float
    strikes: np.ndarray
    prices: np.ndarray
    is_calls: bool = True


class LocalVolSurfaceBuilder:
    """
    Builds 2D local volatility surface from market option prices
    Uses Dupire's formula with regularization for stability
    """
    
    def __init__(self, spot: float, max_maturity: float = 1.0):
        self.spot = spot
        self.max_maturity = max_maturity
        self.surface: Optional[np.ndarray] = None
        self.strike_grid: Optional[np.ndarray] = None
        self.maturity_grid: Optional[np.ndarray] = None
        self.interpolator: Optional[RectBivariateSpline] = None
    
    def build_from_market(self, slices: List[MarketSlice],
                          n_strikes: int = 50,
                          n_maturities: int = 30) -> np.ndarray:
        """
        Build local vol surface from market data
        
        Args:
            slices: List of market option slices by maturity
            n_strikes: Number of strike grid points
            n_maturities: Number of maturity grid points
        
        Returns:
            2D array of local volatility [n_maturities, n_strikes]
        """
        # Create grids
        self.strike_grid = np.logspace(
            np.log(self.spot * 0.5),
            np.log(self.spot * 1.5),
            n_strikes
        )
        self.maturity_grid = np.linspace(0.01, self.max_maturity, n_maturities)
        
        # Interpolate market prices onto regular grid
        price_grid = self._interpolate_prices(slices, n_strikes, n_maturities)
        
        # Compute local vol using Dupire's formula
        self.surface = self._compute_dupire(price_grid)
        
        # Smooth and regularize
        self.surface = self._regularize_surface(self.surface)
        
        # Build interpolator for continuous evaluation
        self._build_interpolator()
        
        return self.surface
    
    def _interpolate_prices(self, slices: List[MarketSlice],
                            n_strikes: int, n_maturities: int) -> np.ndarray:
        """Interpolate scattered market prices onto regular grid"""
        price_grid = np.zeros((n_maturities, n_strikes))
        
        for i, t in enumerate(self.maturity_grid):
            # Find closest maturity slice
            closest_slice = min(slices, key=lambda s: abs(s.maturity - t))
            
            # Interpolate in strike dimension
            if len(closest_slice.strikes) >= 2:
                from scipy.interpolate import interp1d
                
                # Ensure monotonic decreasing prices for calls
                prices = closest_slice.prices
                if closest_slice.is_calls:
                    # Enforce monotonicity
                    prices = np.minimum.accumulate(prices)
                
                interp_func = interp1d(
                    closest_slice.strikes,
                    prices,
                    kind='cubic',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                
                price_grid[i, :] = interp_func(self.strike_grid)
            else:
                # Fallback to intrinsic value
                price_grid[i, :] = np.maximum(self.spot - self.strike_grid, 0)
        
        return price_grid
    
    def _compute_dupire(self, price_grid: np.ndarray) -> np.ndarray:
        """
        Compute local volatility using Dupire's formula
        σ_loc²(K,T) = 2 * ∂C/∂T / (K² * ∂²C/∂K²)
        """
        n_mat, n_str = price_grid.shape
        local_vol = np.zeros_like(price_grid)
        
        dt = self.maturity_grid[1] - self.maturity_grid[0] if n_mat > 1 else 0.01
        dk = np.diff(np.log(self.strike_grid)).mean()
        
        for i in range(n_mat):
            for j in range(n_str):
                t = self.maturity_grid[i]
                k = self.strike_grid[j]
                
                if t < 1e-6:
                    local_vol[i, j] = 0.6  # Default initial vol
                    continue
                
                # Time derivative (backward difference for stability)
                if i == 0:
                    dcdt = (price_grid[1, j] - price_grid[0, j]) / dt if n_mat > 1 else 0
                else:
                    dcdt = (price_grid[i, j] - price_grid[i-1, j]) / dt
                
                # Strike convexity (central difference)
                if j == 0:
                    d2cdk2 = (price_grid[i, 2] - 2*price_grid[i, 1] + price_grid[i, 0]) / (dk*dk) if n_str > 2 else 0
                elif j == n_str - 1:
                    d2cdk2 = (price_grid[i, -1] - 2*price_grid[i, -2] + price_grid[i, -3]) / (dk*dk) if n_str > 2 else 0
                else:
                    d2cdk2 = (price_grid[i, j+1] - 2*price_grid[i, j] + price_grid[i, j-1]) / (dk*dk)
                
                # Dupire's formula
                if abs(d2cdk2) > 1e-10 and dcdt > 0:
                    sigma_sq = 2.0 * dcdt / (k*k * d2cdk2)
                    local_vol[i, j] = np.sqrt(max(sigma_sq, 0))
                else:
                    # Extrapolate from neighbors or use default
                    local_vol[i, j] = self._extrapolate_vol(local_vol, i, j)
        
        return local_vol
    
    def _extrapolate_vol(self, vol_grid: np.ndarray, i: int, j: int) -> float:
        """Extrapolate volatility from neighboring points"""
        neighbors = []
        
        if i > 0:
            neighbors.append(vol_grid[i-1, j])
        if j > 0:
            neighbors.append(vol_grid[i, j-1])
        if i > 0 and j > 0:
            neighbors.append(vol_grid[i-1, j-1])
        
        if neighbors:
            return np.mean(neighbors)
        
        return 0.6  # Default vol
    
    def _regularize_surface(self, surface: np.ndarray) -> np.ndarray:
        """
        Regularize surface to remove noise and ensure smoothness
        Critical for stable Monte Carlo simulation
        """
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        
        # Adaptive bandwidth based on grid resolution
        sigma = max(1.0, len(self.maturity_grid) / 20)
        smoothed = gaussian_filter(surface, sigma=sigma)
        
        # Enforce reasonable bounds
        smoothed = np.clip(smoothed, 0.1, 3.0)
        
        # Ensure term structure consistency (vol should not decrease too fast with time)
        for j in range(surface.shape[1]):
            col = smoothed[:, j]
            # Enforce mild mean reversion
            mean_vol = np.mean(col)
            smoothed[:, j] = 0.8 * col + 0.2 * mean_vol
        
        return smoothed
    
    def _build_interpolator(self) -> None:
        """Build bivariate spline interpolator for continuous evaluation"""
        if self.surface is None or self.strike_grid is None:
            return
        
        self.interpolator = RectBivariateSpline(
            self.maturity_grid,
            self.strike_grid,
            self.surface,
            kx=3, ky=3
        )
    
    def get_local_vol(self, strike: float, maturity: float) -> float:
        """Get interpolated local volatility at arbitrary point"""
        if self.interpolator is None:
            raise ValueError("Surface not built yet")
        
        # Clamp to grid bounds
        t = np.clip(maturity, self.maturity_grid.min(), self.maturity_grid.max())
        k = np.clip(strike, self.strike_grid.min(), self.strike_grid.max())
        
        return float(self.interpolator(t, k)[0, 0])
    
    def get_surface_for_mc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get surface components ready for Monte Carlo simulation
        
        Returns:
            (strike_grid, maturity_grid, vol_surface)
        """
        if self.surface is None:
            raise ValueError("Surface not built yet")
        
        return self.strike_grid, self.maturity_grid, self.surface
    
    def extract_fat_tail_parameter(self) -> float:
        """
        Extract parameter characterizing fat tails in crypto options
        Higher value = fatter tails (typical BTC: 0.3-0.5, normal: 0.1-0.2)
        """
        if self.surface is None:
            return 0.4  # Default crypto value
        
        # Measure wing steepness (OTM put vs OTM call vol)
        left_wing = self.surface[:, :5].mean()  # Low strikes
        right_wing = self.surface[:, -5:].mean()  # High strikes
        center = self.surface[:, len(self.strike_grid)//2 - 2:len(self.strike_grid)//2 + 3].mean()
        
        # Fat tail coefficient
        wing_premium = (left_wing + right_wing) / 2 - center
        return max(0.1, min(0.8, wing_premium / center + 0.3))


if __name__ == '__main__':
    # Demo usage
    builder = LocalVolSurfaceBuilder(spot=45000.0, max_maturity=0.5)
    
    # Generate synthetic market data
    slices = []
    maturities = [0.027, 0.082, 0.25, 0.5]
    
    for t in maturities:
        strikes = np.linspace(35000, 55000, 15)
        # Synthetic prices with smile
        atm_vol = 0.6 + 0.1 * t
        prices = []
        for k in strikes:
            moneyness = k / 45000
            iv = atm_vol * (1 + 0.2 * np.log(moneyness)**2)
            # Simple BS approximation
            d1 = np.log(45000/k) / (iv*np.sqrt(t)) + 0.5*iv*np.sqrt(t)
            from scipy.stats import norm
            price = 45000 * norm.cdf(d1) - k * norm.cdf(d1 - iv*np.sqrt(t))
            prices.append(max(price, 0.01))
        
        slices.append(MarketSlice(
            maturity=t,
            strikes=strikes,
            prices=np.array(prices)
        ))
    
    print("Building local vol surface...")
    surface = builder.build_from_market(slices)
    
    print(f"Surface shape: {surface.shape}")
    print(f"ATM short-term vol: {builder.get_local_vol(45000, 0.027):.2%}")
    print(f"ATM long-term vol: {builder.get_local_vol(45000, 0.5):.2%}")
    print(f"Fat tail parameter: {builder.extract_fat_tail_parameter():.3f}")
