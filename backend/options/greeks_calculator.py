#!/usr/bin/env python3
"""
Options Greeks Calculator

Real-time calculation of Delta, Gamma, Theta, Vega, and Rho for crypto options.
Designed for high-frequency updates with minimal latency on the 8GB RAM constraint.

Features:
- Real-time Greeks tracking for multi-leg strategies
- Portfolio-level risk aggregation
- Dynamic hedging signals based on delta thresholds
- Memory-efficient batch calculations

Target: Update Greeks instantly when new options trades print on Binance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Container for all option Greeks"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0  # Per day
    vega: float = 0.0   # Per 1% vol change
    rho: float = 0.0    # Per 1% rate change
    
    def __add__(self, other: 'Greeks') -> 'Greeks':
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
        )
    
    def __mul__(self, scalar: float) -> 'Greeks':
        return Greeks(
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            theta=self.theta * scalar,
            vega=self.vega * scalar,
            rho=self.rho * scalar,
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
        }


@dataclass
class OptionPosition:
    """Represents an option position with its Greeks"""
    symbol: str
    option_type: OptionType
    strike: float
    expiry_date: str
    quantity: int  # Number of contracts
    entry_price: float
    current_price: float
    underlying_price: float
    implied_vol: float
    risk_free_rate: float = 0.05
    timestamp_us: int = 0
    
    def __post_init__(self):
        if self.timestamp_us == 0:
            self.timestamp_us = int(time.time() * 1_000_000)
    
    @property
    def days_to_expiry(self) -> float:
        """Calculate days until expiry"""
        from datetime import datetime
        expiry = datetime.strptime(self.expiry_date, '%Y-%m-%d')
        now = datetime.utcnow()
        return (expiry - now).total_seconds() / 86400.0
    
    @property
    def years_to_expiry(self) -> float:
        """Calculate years until expiry"""
        return max(0, self.days_to_expiry / 365.0)


class BlackScholesCalculator:
    """
    Pure Python implementation of Black-Scholes for Greeks calculation.
    Optimized for speed with precomputed constants.
    """
    
    SQRT_2PI = math.sqrt(2 * math.pi)
    
    @staticmethod
    def norm_cdf(x: float) -> float:
        """Standard normal CDF using Abramowitz and Stegun approximation"""
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        sign = 1.0 if x >= 0 else -1.0
        x = abs(x)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return 0.5 * (1.0 + sign * y)
    
    @staticmethod
    def norm_pdf(x: float) -> float:
        """Standard normal PDF"""
        return math.exp(-0.5 * x * x) / BlackScholesCalculator.SQRT_2PI
    
    @classmethod
    def calculate_d1_d2(cls, spot: float, strike: float, time: float, vol: float, rate: float) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters"""
        if time <= 0 or vol <= 0:
            return 0.0, 0.0
        
        sqrt_t = math.sqrt(time)
        vol_sqrt_t = vol * sqrt_t
        
        d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        
        return d1, d2
    
    @classmethod
    def calculate_greeks(
        cls,
        option_type: OptionType,
        spot: float,
        strike: float,
        time: float,
        vol: float,
        rate: float,
    ) -> Greeks:
        """
        Calculate all Greeks in a single pass for efficiency.
        
        Returns:
            Greeks object with delta, gamma, theta, vega, rho
        """
        if time <= 0 or vol <= 0 or spot <= 0:
            return Greeks()
        
        d1, d2 = cls.calculate_d1_d2(spot, strike, time, vol, rate)
        sqrt_t = math.sqrt(time)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = cls.norm_cdf(d1)
        else:
            delta = cls.norm_cdf(d1) - 1.0
        
        # Gamma (same for call and put)
        gamma = cls.norm_pdf(d1) / (spot * vol * sqrt_t)
        
        # Theta (per day)
        term1 = -spot * cls.norm_pdf(d1) * vol / (2.0 * sqrt_t)
        term2 = rate * strike * math.exp(-rate * time)
        
        if option_type == OptionType.CALL:
            theta = (term1 - term2 * cls.norm_cdf(d2)) / 365.0
        else:
            theta = (term1 + term2 * cls.norm_cdf(-d2)) / 365.0
        
        # Vega (per 1% change)
        vega = spot * sqrt_t * cls.norm_pdf(d1) * 0.01
        
        # Rho (per 1% change)
        if option_type == OptionType.CALL:
            rho = strike * time * math.exp(-rate * time) * cls.norm_cdf(d2) * 0.01
        else:
            rho = -strike * time * math.exp(-rate * time) * cls.norm_cdf(-d2) * 0.01
        
        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


class GreeksCalculator:
    """
    Main Greeks calculator for tracking real-time option risk.
    
    Features:
    - Batch Greek calculations for portfolio efficiency
    - Aggregated portfolio-level risk metrics
    - Delta-hedging signal generation
    """
    
    def __init__(self):
        self.positions: Dict[str, OptionPosition] = {}
        self.calculator = BlackScholesCalculator()
        self._portfolio_greeks_cache: Optional[Greeks] = None
        self._cache_valid = False
    
    def add_position(self, position: OptionPosition) -> None:
        """Add or update an option position"""
        key = f"{position.symbol}_{position.option_type.value}_{position.strike}_{position.expiry_date}"
        self.positions[key] = position
        self._cache_valid = False
    
    def remove_position(self, key: str) -> bool:
        """Remove a position by key"""
        if key in self.positions:
            del self.positions[key]
            self._cache_valid = False
            return True
        return False
    
    def calculate_position_greeks(self, position: OptionPosition) -> Greeks:
        """Calculate Greeks for a single position"""
        greeks = self.calculator.calculate_greeks(
            option_type=position.option_type,
            spot=position.underlying_price,
            strike=position.strike,
            time=position.years_to_expiry,
            vol=position.implied_vol,
            rate=position.risk_free_rate,
        )
        
        # Scale by quantity and contract multiplier (typically 1 for crypto)
        contract_multiplier = 1.0
        return greeks * (position.quantity * contract_multiplier)
    
    def get_portfolio_greeks(self) -> Greeks:
        """
        Calculate aggregated portfolio Greeks.
        Uses caching to avoid redundant calculations.
        """
        if self._cache_valid and self._portfolio_greeks_cache is not None:
            return self._portfolio_greeks_cache
        
        total_greeks = Greeks()
        
        for position in self.positions.values():
            position_greeks = self.calculate_position_greeks(position)
            total_greeks = total_greeks + position_greeks
        
        self._portfolio_greeks_cache = total_greeks
        self._cache_valid = True
        
        return total_greeks
    
    def invalidate_cache(self) -> None:
        """Invalidate the Greeks cache (call when underlying price changes)"""
        self._cache_valid = False
    
    def update_underlying_price(self, symbol: str, new_price: float) -> None:
        """Update underlying price for all positions on that symbol"""
        for position in self.positions.values():
            if position.symbol == symbol:
                position.underlying_price = new_price
        self.invalidate_cache()
    
    def update_implied_vol(self, symbol: str, new_vol: float) -> None:
        """Update implied volatility for all positions on that symbol"""
        for position in self.positions.values():
            if position.symbol == symbol:
                position.implied_vol = new_vol
        self.invalidate_cache()
    
    def get_delta_hedge_signal(self, symbol: str, threshold: float = 0.1) -> Optional[Dict]:
        """
        Generate delta-hedging signal if portfolio delta exceeds threshold.
        
        Args:
            symbol: Underlying asset symbol
            threshold: Delta threshold to trigger hedge
            
        Returns:
            Hedge signal dict or None if no action needed
        """
        portfolio_greeks = self.get_portfolio_greeks()
        
        if abs(portfolio_greeks.delta) < threshold:
            return None
        
        # Calculate required hedge
        hedge_quantity = -portfolio_greeks.delta  # Opposite position to neutralize
        
        return {
            'symbol': symbol,
            'action': 'buy' if hedge_quantity > 0 else 'sell',
            'quantity': abs(hedge_quantity),
            'current_delta': portfolio_greeks.delta,
            'target_delta': 0.0,
            'urgency': 'high' if abs(portfolio_greeks.delta) > threshold * 2 else 'medium',
        }
    
    def get_gamma_exposure(self) -> float:
        """Get total portfolio gamma exposure"""
        return self.get_portfolio_greeks().gamma
    
    def get_theta_decay(self) -> float:
        """Get daily theta decay (positive = earning from time decay)"""
        return self.get_portfolio_greeks().theta
    
    def get_vega_exposure(self) -> float:
        """Get portfolio vega exposure (sensitivity to 1% vol change)"""
        return self.get_portfolio_greeks().vega
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        greeks = self.get_portfolio_greeks()
        
        return {
            'total_delta': greeks.delta,
            'total_gamma': greeks.gamma,
            'daily_theta': greeks.theta,
            'vega_exposure': greeks.vega,
            'total_rho': greeks.rho,
            'position_count': len(self.positions),
            'timestamp_us': int(time.time() * 1_000_000),
        }


class MultiAssetGreeksTracker:
    """Track Greeks across multiple underlying assets"""
    
    def __init__(self):
        self.trackers: Dict[str, GreeksCalculator] = {}
    
    def get_tracker(self, symbol: str) -> GreeksCalculator:
        """Get or create tracker for a symbol"""
        if symbol not in self.trackers:
            self.trackers[symbol] = GreeksCalculator()
        return self.trackers[symbol]
    
    def add_position(self, position: OptionPosition) -> None:
        """Add position to appropriate tracker"""
        tracker = self.get_tracker(position.symbol)
        tracker.add_position(position)
    
    def get_all_greeks(self) -> Dict[str, Greeks]:
        """Get Greeks for all tracked symbols"""
        return {symbol: tracker.get_portfolio_greeks() for symbol, tracker in self.trackers.items()}
    
    def get_total_portfolio_delta(self) -> float:
        """Get aggregate delta across all symbols"""
        return sum(t.get_portfolio_greeks().delta for t in self.trackers.values())
    
    def rebalance_needed(self, threshold: float = 0.5) -> List[Dict]:
        """Check which symbols need rebalancing"""
        signals = []
        for symbol, tracker in self.trackers.items():
            signal = tracker.get_delta_hedge_signal(symbol, threshold)
            if signal:
                signals.append(signal)
        return signals


if __name__ == "__main__":
    # Example usage
    print("Options Greeks Calculator initialized")
    
    # Create sample position
    position = OptionPosition(
        symbol="BTC",
        option_type=OptionType.CALL,
        strike=50000,
        expiry_date="2025-12-31",
        quantity=10,
        entry_price=2000,
        current_price=2100,
        underlying_price=49500,
        implied_vol=0.65,
    )
    
    calculator = GreeksCalculator()
    calculator.add_position(position)
    
    greeks = calculator.get_portfolio_greeks()
    print(f"Portfolio Delta: {greeks.delta:.4f}")
    print(f"Portfolio Gamma: {greeks.gamma:.6f}")
    print(f"Daily Theta: ${greeks.theta:.2f}")
    print(f"Vega Exposure: ${greeks.vega:.2f}")
