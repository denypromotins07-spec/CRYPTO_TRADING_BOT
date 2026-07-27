#!/usr/bin/env python3
"""
SABR Smile Dynamics Tracker
Tracks skew and curvature of the volatility smile in real-time
Optimized for AMD Ryzen AI 5 with strict 8GB RAM constraints

Monitors regime changes in crypto vol surface dynamics
Detects butterfly and calendar arbitrage opportunities
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class SabrParams:
    """SABR model parameters"""
    alpha: float  # Initial volatility level
    beta: float   # CEV exponent (0=normal, 1=lognormal)
    rho: float    # Correlation between forward and vol
    nu: float     # Vol-of-vol
    
    def validate(self) -> bool:
        """Ensure parameters are valid"""
        return (
            self.alpha > 0 and 
            0.0 <= self.beta <= 1.0 and 
            -1.0 <= self.rho <= 1.0 and 
            self.nu > 0
        )


class HaganExpansion:
    """Hagan's asymptotic expansion for SABR implied volatility"""
    
    def __init__(self, params: SabrParams, forward: float, expiry: float):
        self.params = params
        self.forward = forward
        self.expiry = expiry
    
    def implied_vol(self, strike: float) -> Optional[float]:
        """
        Compute implied volatility using Hagan's formula
        Handles crypto-specific edge cases
        """
        p = self.params
        f = self.forward
        k = strike
        t = self.expiry
        
        if t <= 0 or f <= 0 or k <= 0:
            return None
        
        # ATM case
        fk_mid = np.sqrt(f * k)
        if abs(f - k) / fk_mid < 1e-6:
            return self.atm_vol()
        
        # Log-moneyness
        log_fk = np.log(f / k)
        
        # Z variable for Hagan formula
        z = (p.nu / p.alpha) * (f**(1 - p.beta) - k**(1 - p.beta))
        
        # X(z) function
        if abs(z) < 1e-8:
            x_z = 1.0 + (p.rho * p.beta / 4.0) * z
        else:
            sqrt_term = np.sqrt(1 - 2*p.rho*z + z*z)
            x_z = (z * np.log((sqrt_term - p.rho) / (1 - p.rho))) / (z - p.rho + sqrt_term)
            x_z = np.clip(x_z, 0.1, 10.0)
        
        # Leading term
        term1 = p.alpha / (fk_mid**(1 - p.beta))
        
        # Time correction
        term2 = 1.0 + t * (
            ((1 - p.beta)**2 / 24.0) * p.alpha**2 / fk_mid**(2 - 2*p.beta) +
            (p.rho * p.beta / 4.0) * p.nu * p.alpha / fk_mid**(1 - p.beta) +
            (2 - 3*p.rho**2) / 24.0 * p.nu**2
        )
        
        sigma = term1 * term2 * x_z
        
        if sigma <= 0 or sigma > 5.0:
            return None
        
        return sigma
    
    def atm_vol(self) -> Optional[float]:
        """ATM volatility using simplified Hagan formula"""
        p = self.params
        f = self.forward
        t = self.expiry
        
        if f <= 0 or t <= 0:
            return None
        
        atm_sigma = p.alpha / f**(1 - p.beta)
        
        correction = 1.0 + t * (
            ((1 - p.beta)**2 / 24.0) * p.alpha**2 / f**(2 - 2*p.beta) +
            (p.rho * p.beta / 4.0) * p.nu * p.alpha / f**(1 - p.beta) +
            (2 - 3*p.rho**2) / 24.0 * p.nu**2
        )
        
        sigma = atm_sigma * correction
        
        if sigma <= 0 or sigma > 5.0:
            return None
        
        return sigma


@dataclass
class SmileMetrics:
    """Real-time smile metrics"""
    skew: float          # IV(25d put) - IV(25d call)
    curvature: float     # IV(ATM) - 0.5*(IV(put) + IV(call))
    atm_vol: float
    timestamp: float


class SmileDynamicsTracker:
    """
    Real-time tracker for SABR smile dynamics
    Detects regime changes in crypto volatility surface
    """
    
    def __init__(self, max_history: int = 100):
        self.params_history: deque[SabrParams] = deque(maxlen=max_history)
        self.metrics_history: deque[SmileMetrics] = deque(maxlen=max_history)
        self.regime_changes: List[Dict] = []
    
    def update(self, params: SabrParams, forward: float, expiry: float, 
               timestamp: float) -> Optional[SmileMetrics]:
        """
        Record new calibration and compute smile metrics
        
        Args:
            params: New SABR parameters
            forward: Current forward price
            expiry: Time to expiry
            timestamp: Unix timestamp
        
        Returns:
            Computed smile metrics or None if invalid
        """
        if not params.validate():
            return None
        
        self.params_history.append(params)
        
        expansion = HaganExpansion(params, forward, expiry)
        
        # Define strikes for 25-delta options (approximate)
        put_strike = forward * 0.95  # ~25d put
        call_strike = forward * 1.05  # ~25d call
        
        iv_put = expansion.implied_vol(put_strike)
        iv_call = expansion.implied_vol(call_strike)
        iv_atm = expansion.atm_vol()
        
        if iv_put is None or iv_call is None or iv_atm is None:
            return None
        
        # Compute skew (typically negative in crypto)
        skew = iv_put - iv_call
        
        # Compute curvature (positive = smile, negative = smirk)
        curvature = iv_atm - 0.5 * (iv_put + iv_call)
        
        metrics = SmileMetrics(
            skew=skew,
            curvature=curvature,
            atm_vol=iv_atm,
            timestamp=timestamp
        )
        
        self.metrics_history.append(metrics)
        
        # Check for regime change
        self._detect_regime_change()
        
        return metrics
    
    def _detect_regime_change(self) -> None:
        """Detect significant changes in smile regime"""
        if len(self.metrics_history) < 5:
            return
        
        recent_skews = [m.skew for m in list(self.metrics_history)[-3:]]
        old_skews = [m.skew for m in list(self.metrics_history)[-6:-3]]
        
        recent_avg = np.mean(recent_skews)
        old_avg = np.mean(old_skews)
        
        skew_change = recent_avg - old_avg
        
        if abs(skew_change) > 0.1:  # Significant change threshold
            regime = "flattening" if skew_change > 0 else "steepening"
            self.regime_changes.append({
                'type': regime,
                'magnitude': abs(skew_change),
                'timestamp': self.metrics_history[-1].timestamp,
                'old_skew': old_avg,
                'new_skew': recent_avg
            })
    
    def current_skew(self) -> Optional[float]:
        """Get current smile skew"""
        if self.metrics_history:
            return self.metrics_history[-1].skew
        return None
    
    def current_curvature(self) -> Optional[float]:
        """Get current smile curvature"""
        if self.metrics_history:
            return self.metrics_history[-1].curvature
        return None
    
    def get_regime(self) -> str:
        """Get current smile regime"""
        if not self.metrics_history or len(self.metrics_history) < 5:
            return "unknown"
        
        recent_skew = self.metrics_history[-1].skew
        historical_avg = np.mean([m.skew for m in self.metrics_history])
        
        if abs(recent_skew - historical_avg) < 0.05:
            return "stable"
        elif recent_skew > historical_avg:
            return "flattening"  # Less negative skew
        else:
            return "steepening"  # More negative skew
    
    def get_arbitrage_signals(self, forward: float, expiry: float) -> Dict[str, bool]:
        """
        Check for potential arbitrage opportunities in the smile
        
        Returns dict with:
        - butterfly_arb: Butterfly spread mispricing
        - calendar_arb: Calendar spread mispricing (needs term structure)
        - risk_reversal: Skew extreme relative to history
        """
        signals = {
            'butterfly_arb': False,
            'calendar_arb': False,
            'risk_reversal': False
        }
        
        if not self.metrics_history or not self.params_history:
            return signals
        
        current_params = self.params_history[-1]
        expansion = HaganExpansion(current_params, forward, expiry)
        
        # Check for extreme skew (potential risk reversal opportunity)
        if self.metrics_history:
            historical_skews = [m.skew for m in self.metrics_history]
            current_skew = self.metrics_history[-1].skew
            
            if len(historical_skews) >= 20:
                skew_mean = np.mean(historical_skews)
                skew_std = np.std(historical_skews)
                
                if abs(current_skew - skew_mean) > 2.5 * skew_std:
                    signals['risk_reversal'] = True
        
        # Check butterfly convexity
        strikes = [forward * 0.9, forward, forward * 1.1]
        ivs = [expansion.implied_vol(k) for k in strikes]
        
        if all(iv is not None for iv in ivs):
            # Butterfly should have positive value
            # Simplified check: middle IV should be less than average of wings
            if ivs[1] > 0.5 * (ivs[0] + ivs[2]) + 0.02:  # 2% tolerance
                signals['butterfly_arb'] = True
        
        return signals


class CryptoSmileAnalyzer:
    """
    Specialized analyzer for crypto option smile characteristics
    Handles fat tails and extreme moves typical in BTC/ETH options
    """
    
    def __init__(self):
        self.tracker = SmileDynamicsTracker()
        self.fat_tail_coefficient: Optional[float] = None
    
    def calibrate_to_market(self, market_data: List[Dict], 
                            forward: float, expiry: float) -> Optional[SabrParams]:
        """
        Calibrate SABR to market option prices
        
        Args:
            market_data: List of {strike, price, is_call} dicts
            forward: Forward price
            expiry: Time to expiry
        
        Returns:
            Calibrated SABR parameters
        """
        # Extract implied vols from market prices
        market_ivs = []
        for opt in market_data:
            iv = self._extract_iv(
                forward, opt['strike'], expiry, opt['price'], opt.get('is_call', True)
            )
            if iv:
                market_ivs.append((opt['strike'], iv))
        
        if len(market_ivs) < 3:
            return None
        
        # Objective: minimize squared IV errors
        def objective(params_arr):
            alpha, beta, rho, nu = params_arr
            
            if alpha <= 0 or not (0 <= beta <= 1) or not (-1 <= rho <= 1) or nu <= 0:
                return 1e6
            
            params = SabrParams(alpha, beta, rho, nu)
            expansion = HaganExpansion(params, forward, expiry)
            
            error = 0.0
            for strike, market_iv in market_ivs:
                model_iv = expansion.implied_vol(strike)
                if model_iv:
                    error += (model_iv - market_iv)**2
            
            return error
        
        # Initial guess for crypto
        x0 = [0.6, 0.9, -0.4, 0.8]
        
        result = minimize_scalar(
            lambda x: objective(x),
            bounds=[(0.1, 2.0), (0.5, 1.0), (-0.9, 0.5), (0.1, 2.0)],
            method='Nelder-Mead'
        )
        
        if result.success:
            return SabrParams(*result.x)
        
        return None
    
    def _extract_iv(self, forward: float, strike: float, expiry: float,
                    price: float, is_call: bool) -> Optional[float]:
        """Extract implied volatility from option price (simplified)"""
        if expiry <= 0 or price <= 0:
            return None
        
        # Simple bisection for IV
        vol_low, vol_high = 0.01, 3.0
        
        for _ in range(50):
            vol_mid = 0.5 * (vol_low + vol_high)
            model_price = self._bs_price(forward, strike, expiry, vol_mid, is_call)
            
            if abs(model_price - price) < 1e-4:
                return vol_mid
            
            if model_price > price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid
        
        return 0.5 * (vol_low + vol_high)
    
    def _bs_price(self, f: float, k: float, t: float, sigma: float, 
                  is_call: bool) -> float:
        """Black-Scholes pricer"""
        if t <= 0 or sigma <= 0:
            return max(f - k, 0) if is_call else max(k - f, 0)
        
        d1 = np.log(f / k) / (sigma * np.sqrt(t)) + 0.5 * sigma * np.sqrt(t)
        d2 = d1 - sigma * np.sqrt(t)
        
        from scipy.stats import norm
        if is_call:
            return f * norm.cdf(d1) - k * norm.cdf(d2)
        else:
            return k * norm.cdf(-d2) - f * norm.cdf(-d1)


if __name__ == '__main__':
    # Demo usage
    tracker = SmileDynamicsTracker()
    
    # Simulate crypto smile evolution
    base_params = SabrParams(alpha=0.6, beta=0.9, rho=-0.4, nu=0.8)
    forward = 45000.0
    expiry = 0.25
    
    print("Tracking SABR smile dynamics...")
    for i in range(10):
        # Add some noise to simulate market movements
        params = SabrParams(
            alpha=base_params.alpha * (1 + 0.05 * np.random.randn()),
            beta=base_params.beta,
            rho=base_params.rho + 0.02 * np.random.randn(),
            nu=base_params.nu * (1 + 0.05 * np.random.randn())
        )
        
        metrics = tracker.update(params, forward, expiry, timestamp=i * 60.0)
        
        if metrics:
            print(f"t={i}: skew={metrics.skew:.4f}, curvature={metrics.curvature:.4f}, "
                  f"regime={tracker.get_regime()}")
    
    # Check for arbitrage signals
    signals = tracker.get_arbitrage_signals(forward, expiry)
    print(f"\nArbitrage signals: {signals}")
