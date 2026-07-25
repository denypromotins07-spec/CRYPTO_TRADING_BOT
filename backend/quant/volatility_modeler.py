"""
Volatility Modeler for ZAID Personal Crypto Trading Bot
Tracks implied volatility and market regime detection
Uses statistical methods for regime classification
Memory-efficient implementations

Part of the 152 domains of quantitative finance implementation.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    LOW_VOL_BULL = "LOW_VOL_BULL"
    LOW_VOL_BEAR = "LOW_VOL_BEAR"
    HIGH_VOL_BULL = "HIGH_VOL_BULL"
    HIGH_VOL_BEAR = "HIGH_VOL_BEAR"
    TRANSITION = "TRANSITION"
    CRASH = "CRASH"


@dataclass
class VolatilityState:
    """Current volatility state for an asset."""
    symbol: str
    realized_vol: float
    implied_vol: Optional[float]
    vol_of_vol: float
    regime: MarketRegime
    regime_probability: float
    timestamp: float


class RealizedVolatilityCalculator:
    """
    Calculate realized volatility using various estimators.
    Optimized for high-frequency data.
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.log_returns: deque = deque(maxlen=window_size * 2)
        self.high_low_data: deque = deque(maxlen=window_size)
        
    def update_return(self, price: float) -> Optional[float]:
        """Update with new price and calculate return."""
        if len(self.log_returns) > 0:
            last_price = self.log_returns[-1][1] if self.log_returns else None
            if last_price and last_price > 0:
                log_ret = np.log(price / last_price)
                self.log_returns.append((price, log_ret))
                return log_ret
        self.log_returns.append((price, 0.0))
        return None
    
    def update_ohlcv(self, high: float, low: float, close: float, volume: float):
        """Update with OHLCV data for better estimators."""
        self.high_low_data.append({
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    def get_realized_vol(self, annualize: bool = True) -> Optional[float]:
        """Calculate standard realized volatility."""
        if len(self.log_returns) < self.window_size:
            return None
        
        returns = np.array([r[1] for r in list(self.log_returns)[-self.window_size:]])
        vol = np.std(returns)
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol
    
    def get_parkinson_vol(self, annualize: bool = True) -> Optional[float]:
        """
        Parkinson volatility estimator using high-low range.
        More efficient than close-close estimator.
        """
        if len(self.high_low_data) < self.window_size:
            return None
        
        recent = list(self.high_low_data)[-self.window_size:]
        
        # Parkinson estimator: sqrt(1/(4n*ln2) * sum(ln(H/L)^2))
        ln_hl_sq = [np.log(d['high'] / d['low']) ** 2 for d in recent]
        vol = np.sqrt(np.sum(ln_hl_sq) / (4 * len(recent) * np.log(2)))
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol
    
    def get_garman_klass_vol(self, annualize: bool = True) -> Optional[float]:
        """
        Garman-Klass volatility estimator.
        Uses open, high, low, close for better efficiency.
        """
        if len(self.high_low_data) < self.window_size:
            return None
        
        recent = list(self.high_low_data)[-self.window_size:]
        
        vol_sum = 0
        for d in recent:
            h, l, c, o = d['high'], d['low'], d['close'], d.get('open', d['close'])
            if o > 0 and l > 0:
                term = 0.5 * (np.log(h/l))**2 - (2*np.log(2)-1) * (np.log(c/o))**2
                vol_sum += max(0, term)
        
        vol = np.sqrt(vol_sum / len(recent))
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol


class ImpliedVolatilityEstimator:
    """
    Estimate implied volatility from option prices or use proxy.
    For crypto, uses ATM straddle approximation.
    """
    
    def __init__(self):
        self.atm_straddle_prices: deque = deque(maxlen=50)
        self.spot_prices: deque = deque(maxlen=50)
        
    def update_straddle(self, straddle_price: float, spot: float, 
                       days_to_expiry: int = 30):
        """Update with ATM straddle price."""
        self.atm_straddle_prices.append(straddle_price)
        self.spot_prices.append(spot)
    
    def estimate_iv(self) -> Optional[float]:
        """
        Estimate implied volatility from straddle price.
        Brenner-Subrahmanyam approximation: Straddle ≈ 0.8 * S * σ * √T
        """
        if not self.atm_straddle_prices or not self.spot_prices:
            return None
        
        straddle = self.atm_straddle_prices[-1]
        spot = self.spot_prices[-1]
        T = 30 / 365  # Assuming 30 days
        
        if spot > 0 and T > 0:
            iv = straddle / (0.8 * spot * np.sqrt(T))
            return iv
        
        return None
    
    def get_iv_history(self) -> List[float]:
        """Get historical IV estimates."""
        ivs = []
        for i in range(len(self.atm_straddle_prices)):
            if i < len(self.spot_prices):
                straddle = self.atm_straddle_prices[i]
                spot = self.spot_prices[i]
                T = 30 / 365
                if spot > 0:
                    ivs.append(straddle / (0.8 * spot * np.sqrt(T)))
        return ivs


class RegimeDetector:
    """
    Detect market regimes using volatility and trend indicators.
    Uses Hidden Markov Model-inspired approach.
    """
    
    def __init__(self, vol_threshold: float = 0.3):
        self.vol_threshold = vol_threshold  # Annualized vol threshold
        self.vol_history: deque = deque(maxlen=60)
        self.return_history: deque = deque(maxlen=60)
        
        # Regime transition probabilities (simplified)
        self.transition_matrix = np.array([
            [0.9, 0.1],  # Low vol -> Low vol, High vol
            [0.2, 0.8]   # High vol -> Low vol, High vol
        ])
        
        self.current_regime_prob = np.array([0.7, 0.3])  # [low_vol, high_vol]
        
    def update(self, realized_vol: float, returns: np.ndarray) -> MarketRegime:
        """Update regime detection with new data."""
        self.vol_history.append(realized_vol)
        self.return_history.extend(returns[-10:])
        
        if len(self.vol_history) < 20:
            return MarketRegime.TRANSITION
        
        # Update regime probabilities
        is_high_vol = realized_vol > self.vol_threshold
        
        # Simple Bayesian update
        if is_high_vol:
            self.current_regime_prob[1] *= 1.2
            self.current_regime_prob[0] *= 0.8
        else:
            self.current_regime_prob[0] *= 1.1
            self.current_regime_prob[1] *= 0.9
        
        # Normalize
        self.current_regime_prob /= np.sum(self.current_regime_prob)
        
        # Determine trend direction
        avg_return = np.mean(list(self.return_history)[-20:]) if self.return_history else 0
        
        # Classify regime
        high_vol_prob = self.current_regime_prob[1]
        
        if high_vol_prob > 0.7:
            if avg_return > 0.001:
                regime = MarketRegime.HIGH_VOL_BULL
            elif avg_return < -0.001:
                regime = MarketRegime.HIGH_VOL_BEAR
            else:
                regime = MarketRegime.TRANSITION
        else:
            if avg_return > 0.001:
                regime = MarketRegime.LOW_VOL_BULL
            elif avg_return < -0.001:
                regime = MarketRegime.LOW_VOL_BEAR
            else:
                regime = MarketRegime.TRANSITION
        
        # Check for crash conditions
        if len(self.return_history) >= 5:
            recent_returns = list(self.return_history)[-5:]
            if np.sum(recent_returns) < -0.1:  # 10% drop in 5 periods
                regime = MarketRegime.CRASH
        
        return regime


class VolatilityModeler:
    """
    Main engine for volatility modeling and regime detection.
    Singleton pattern for global access.
    """
    
    _instance: Optional['VolatilityModeler'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.rv_calculators: Dict[str, RealizedVolatilityCalculator] = {}
        self.iv_estimators: Dict[str, ImpliedVolatilityEstimator] = {}
        self.regime_detectors: Dict[str, RegimeDetector] = {}
        
        self.vol_history: Dict[str, deque] = {}
        self.max_history = 200
        
        self._initialized = True
        logger.info("VolatilityModeler initialized")
    
    def get_or_create_models(self, symbol: str):
        """Get or create volatility models for a symbol."""
        if symbol not in self.rv_calculators:
            self.rv_calculators[symbol] = RealizedVolatilityCalculator()
            self.iv_estimators[symbol] = ImpliedVolatilityEstimator()
            self.regime_detectors[symbol] = RegimeDetector()
            self.vol_history[symbol] = deque(maxlen=self.max_history)
    
    def update(self, symbol: str, high: float, low: float, 
               close: float, open_: float, volume: float,
               timestamp: float) -> VolatilityState:
        """Update all volatility models with new candle data."""
        self.get_or_create_models(symbol)
        
        # Update realized vol calculator
        self.rv_calculators[symbol].update_return(close)
        self.rv_calculators[symbol].update_ohlcv(high, low, close, volume)
        
        # Get volatility measures
        realized_vol = self.rv_calculators[symbol].get_realized_vol()
        parkinson_vol = self.rv_calculators[symbol].get_parkinson_vol()
        gk_vol = self.rv_calculators[symbol].get_garman_klass_vol()
        
        # Use best available estimate
        if gk_vol:
            current_vol = gk_vol
        elif parkinson_vol:
            current_vol = parkinson_vol
        elif realized_vol:
            current_vol = realized_vol
        else:
            current_vol = 0.0
        
        self.vol_history[symbol].append(current_vol)
        
        # Get implied vol estimate
        implied_vol = self.iv_estimators[symbol].estimate_iv()
        
        # Calculate vol-of-vol
        vol_of_vol = 0.0
        if len(self.vol_history[symbol]) >= 10:
            vol_array = np.array(list(self.vol_history[symbol])[-20:])
            vol_of_vol = np.std(vol_array) * np.sqrt(252)
        
        # Detect regime
        returns = np.array([r[1] for r in list(self.rv_calculators[symbol].log_returns)[-50:]])
        regime = self.regime_detectors[symbol].update(current_vol, returns)
        
        # Calculate regime probability
        regime_prob = self.regime_detectors[symbol].current_regime_prob[1]
        
        return VolatilityState(
            symbol=symbol,
            realized_vol=current_vol,
            implied_vol=implied_vol,
            vol_of_vol=vol_of_vol,
            regime=regime,
            regime_probability=regime_prob,
            timestamp=timestamp
        )
    
    def get_volatility_term_structure(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get volatility estimates across different time horizons."""
        if symbol not in self.rv_calculators:
            return None
        
        calc = self.rv_calculators[symbol]
        
        # Short-term (5-day)
        short_vol = calc.get_realized_vol()
        
        # Medium-term (20-day default)
        medium_vol = calc.get_realized_vol()
        
        # Long-term (60-day)
        original_window = calc.window_size
        calc.window_size = 60
        long_vol = calc.get_realized_vol()
        calc.window_size = original_window
        
        return {
            'short_term': short_vol,
            'medium_term': medium_vol,
            'long_term': long_vol,
            'term_spread': (long_vol - short_vol) if short_vol and long_vol else None
        }
    
    def get_regime_summary(self, symbol: str) -> Optional[Dict[str, any]]:
        """Get current regime summary for a symbol."""
        if symbol not in self.regime_detectors:
            return None
        
        detector = self.regime_detectors[symbol]
        
        return {
            'symbol': symbol,
            'current_regime': detector.current_regime_prob.argmax(),
            'low_vol_probability': detector.current_regime_prob[0],
            'high_vol_probability': detector.current_regime_prob[1],
            'vol_threshold': detector.vol_threshold
        }


if __name__ == "__main__":
    import time
    
    modeler = VolatilityModeler()
    
    # Simulate price updates
    base_price = 45000
    for i in range(100):
        vol_factor = 0.002 if i < 50 else 0.01  # Increase vol halfway
        
        ret = np.random.randn() * vol_factor
        close = base_price * (1 + ret)
        high = close * (1 + abs(np.random.randn()) * vol_factor)
        low = close * (1 - abs(np.random.randn()) * vol_factor)
        open_ = base_price
        
        state = modeler.update(
            "BTCUSDT", high, low, close, open_, 1000, time.time()
        )
        
        if i % 20 == 0:
            print(f"Step {i}:")
            print(f"  Realized Vol: {state.realized_vol*100:.2f}%")
            print(f"  Regime: {state.regime.value}")
            print(f"  High Vol Prob: {state.regime_probability:.2f}")
        
        base_price = close
