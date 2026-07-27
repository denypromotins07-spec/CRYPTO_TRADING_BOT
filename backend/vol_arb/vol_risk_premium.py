#!/usr/bin/env python3
"""
Volatility Risk Premium Calculator
Calculates variance risk premium to harvest vol decay
Optimized for AMD Ryzen AI 5 with strict 8GB RAM constraints

VRP = E[realized_var] - implied_var (typically negative in crypto)
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class VolatilityRiskPremium:
    """VRP measurement result"""
    vrp: float              # Raw VRP value
    vrp_zscore: float       # Standardized VRP
    signal: str             # Trading signal
    confidence: float       # Signal confidence [0, 1]
    expected_return: float  # Expected P&L from harvesting


class VRPCalculator:
    """
    Calculates volatility risk premium for crypto markets
    Exploits the persistent negative VRP in crypto options
    """
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.realized_var_history: deque = deque(maxlen=lookback_days)
        self.implied_var_history: deque = deque(maxlen=lookback_days)
        self.vrp_history: deque = deque(maxlen=lookback_days)
    
    def add_observation(self, realized_vol: float, implied_vol: float,
                        annualization_factor: float = 252.0) -> Optional[VolatilityRiskPremium]:
        """
        Add daily observation and compute VRP
        
        Args:
            realized_vol: Daily realized volatility
            implied_vol: Implied volatility from options
            annualization_factor: Days per year (252 for trading days)
        
        Returns:
            VRP measurement or None if insufficient history
        """
        # Convert to variance
        realized_var = (realized_vol ** 2) / annualization_factor
        implied_var = (implied_vol ** 2) / annualization_factor
        
        self.realized_var_history.append(realized_var)
        self.implied_var_history.append(implied_var)
        
        # VRP = Expected realized var - Implied var
        # In practice: RV(t, t+T) - IV_t^2
        vrp = realized_var - implied_var
        self.vrp_history.append(vrp)
        
        if len(self.vrp_history) < 20:
            return None
        
        # Compute z-score
        vrp_mean = np.mean(self.vrp_history)
        vrp_std = np.std(self.vrp_history)
        
        if vrp_std > 1e-8:
            vrp_zscore = (vrp - vrp_mean) / vrp_std
        else:
            vrp_zscore = 0.0
        
        # Generate signal
        signal, confidence, exp_return = self._generate_signal(vrp, vrp_zscore)
        
        return VolatilityRiskPremium(
            vrp=vrp,
            vrp_zscore=vrp_zscore,
            signal=signal,
            confidence=confidence,
            expected_return=exp_return
        )
    
    def _generate_signal(self, vrp: float, zscore: float) -> Tuple[str, float, float]:
        """
        Generate trading signal based on VRP
        
        Negative VRP = implied > realized = short vol is profitable
        Positive VRP = implied < realized = long vol is profitable
        """
        # Crypto typically has negative VRP (vol sellers get paid)
        threshold = 0.5  # Z-score threshold
        
        if zscore < -threshold:
            # Very negative VRP = strong short vol signal
            confidence = min(1.0, abs(zscore) / 3.0)
            expected_return = abs(vrp) * confidence * 0.5  # Conservative estimate
            return "SHORT_VOL", confidence, expected_return
        elif zscore > threshold:
            # Positive VRP = rare, but indicates long vol opportunity
            confidence = min(1.0, abs(zscore) / 3.0)
            expected_return = abs(vrp) * confidence * 0.3  # Lower confidence
            return "LONG_VOL", confidence, expected_return
        else:
            return "NEUTRAL", 0.3, 0.0
    
    def get_average_vrp(self) -> float:
        """Get historical average VRP (typically negative in crypto)"""
        if not self.vrp_history:
            return 0.0
        return float(np.mean(self.vrp_history))
    
    def get_vrp_percentile(self, current_vrp: float) -> float:
        """Get current VRP percentile in historical distribution"""
        if len(self.vrp_history) < 10:
            return 0.5
        
        below = sum(1 for v in self.vrp_history if v <= current_vrp)
        return below / len(self.vrp_history)
    
    def detect_regime_change(self) -> Optional[str]:
        """Detect significant change in VRP regime"""
        if len(self.vrp_history) < 40:
            return None
        
        recent = list(self.vrp_history)[-10:]
        old = list(self.vrp_history)[-30:-10]
        
        recent_mean = np.mean(recent)
        old_mean = np.mean(old)
        
        # Significant shift detection
        if abs(recent_mean - old_mean) > 2 * np.std(self.vrp_history):
            if recent_mean > old_mean:
                return "VRP_INCREASING"  # Becoming less negative/more positive
            else:
                return "VRP_DECREASING"  # Becoming more negative
        
        return None


class VolHarvestingStrategy:
    """
    Strategy to harvest volatility risk premium
    Systematically sells overpriced volatility in crypto
    """
    
    def __init__(self, max_notional: float = 100000.0):
        self.calculator = VRPCalculator()
        self.max_notional = max_notional
        self.current_position: float = 0.0
        self.cumulative_pnl: float = 0.0
    
    def process_market_data(self, realized_vol: float, implied_vol: float,
                            spot_price: float) -> Optional[Dict]:
        """
        Process market data and generate trading instruction
        
        Args:
            realized_vol: Realized volatility (daily)
            implied_vol: Implied volatility (annualized)
            spot_price: Current spot price
        
        Returns:
            Trading instruction dict or None
        """
        vrp_result = self.calculator.add_observation(realized_vol, implied_vol)
        
        if vrp_result is None:
            return None
        
        # Determine position size based on signal strength
        target_notional = self.max_notional * vrp_result.confidence
        
        if vrp_result.signal == "SHORT_VOL":
            # Sell volatility (short straddle/strangle or variance swap)
            trade_qty = target_notional - self.current_position
            
            if trade_qty < 0:
                action = "REDUCE_SHORT"
            else:
                action = "INCREASE_SHORT"
            
            self.current_position = target_notional
            
        elif vrp_result.signal == "LONG_VOL":
            # Rare: buy volatility
            trade_qty = -target_notional - self.current_position
            
            if trade_qty > 0:
                action = "REDUCE_LONG"
            else:
                action = "INCREASE_LONG"
            
            self.current_position = -target_notional
        else:
            action = "HOLD"
            trade_qty = 0
        
        return {
            'action': action,
            'quantity': abs(trade_qty),
            'notional': abs(self.current_position),
            'vrp': vrp_result.vrp,
            'signal_confidence': vrp_result.confidence,
            'expected_daily_pnl': vrp_result.expected_return,
            'spot_price': spot_price
        }
    
    def update_pnl(self, daily_pnl: float) -> None:
        """Update cumulative P&L tracking"""
        self.cumulative_pnl += daily_pnl
    
    def get_strategy_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return {
            'cumulative_pnl': self.cumulative_pnl,
            'current_position': self.current_position,
            'avg_vrp': self.calculator.get_average_vrp(),
            'position_utilization': abs(self.current_position) / self.max_notional
        }


if __name__ == '__main__':
    # Demo usage
    calculator = VRPCalculator(lookback_days=60)
    
    print("Simulating VRP calculation...")
    print("-" * 50)
    
    # Simulate crypto market: implied typically > realized
    np.random.seed(42)
    for day in range(90):
        # Crypto: realized vol ~50-80%, implied vol ~60-90%
        realized = 0.02 + 0.01 * np.random.randn()  # Daily realized
        implied = 0.025 + 0.008 * np.random.randn()  # Slightly higher implied
        
        result = calculator.add_observation(realized, implied)
        
        if result and day % 10 == 0:
            print(f"Day {day}: VRP={result.vrp:.6f}, z={result.vrp_zscore:.2f}, "
                  f"signal={result.signal}, conf={result.confidence:.2f}")
    
    print("-" * 50)
    print(f"Average VRP: {calculator.get_average_vrp():.6f}")
    print(f"VRP regime: {calculator.detect_regime_change() or 'stable'}")
