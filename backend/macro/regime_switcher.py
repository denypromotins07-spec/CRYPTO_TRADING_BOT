"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 7
Chapter 2: Macro-Economic Indicators - Regime Switcher

File: backend/macro/regime_switcher.py
Purpose: Map macro states to crypto volatility regimes.
         Dynamically adjust strategy parameters based on regime detection.

Features:
- Hidden Markov Model (HMM) inspired regime detection
- Multi-factor regime classification (volatility, correlation, trend)
- Automatic parameter adjustment per regime
- Regime transition probability tracking
- Memory-efficient streaming implementation

Design Patterns:
- State: Different regime behaviors
- Strategy: Regime-specific trading strategies
- Observer: Notify on regime transitions

Author: Opus 4.8
Domain: Regime Detection, Market States, Adaptive Trading
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Deque
from collections import deque
from enum import Enum
import logging
import math
import statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility-based market regimes."""
    LOW_VOL_BULL = "low_vol_bull"       # Calm uptrend
    LOW_VOL_BEAR = "low_vol_bear"       # Calm downtrend
    HIGH_VOL_BULL = "high_vol_bull"     # Volatile uptrend
    HIGH_VOL_BEAR = "high_vol_bear"     # Volatile downtrend (crash)
    TRANSITION = "transition"           # Regime change in progress
    CRISIS = "crisis"                   # Extreme stress


class MacroState(Enum):
    """Macroeconomic state classification."""
    RISK_ON = "risk_on"           # Growth optimistic
    RISK_OFF = "risk_off"         # Safety seeking
    STAGFLATION = "stagflation"   # High inflation + low growth
    REFLATION = "reflation"       # Recovery phase
    RECESSION = "recession"       # Economic contraction
    OVERHEATING = "overheating"   # High growth + high inflation


@dataclass(slots=True)
class RegimeSignal:
    """Individual signal contributing to regime detection."""
    signal_name: str
    value: float
    normalized_value: float  # 0-1 scale
    weight: float
    timestamp: datetime


@dataclass(slots=True)
class RegimeResult:
    """Complete regime detection result."""
    timestamp: datetime
    volatility_regime: VolatilityRegime
    macro_state: MacroState
    confidence: float
    regime_score: float  # Composite score -1 to +1
    recommended_leverage: float
    recommended_stop_distance: float
    transition_probability: float
    
    def is_safe_to_trade(self) -> bool:
        """Determine if current regime is suitable for trading."""
        unsafe_regimes = {VolatilityRegime.CRISIS, VolatilityRegime.TRANSITION}
        return self.volatility_regime not in unsafe_regimes and self.confidence > 0.5


@dataclass(slots=True)
class RegimeTransition:
    """Records a regime transition event."""
    from_regime: VolatilityRegime
    to_regime: VolatilityRegime
    timestamp: datetime
    trigger_factor: str
    duration_in_previous: timedelta


class RegimeDetector:
    """
    Detects market regime using multiple factors.
    
    Factors considered:
    1. Volatility (ATR, realized vol)
    2. Trend strength (ADX, moving average slope)
    3. Correlation regime (crypto-equity correlation)
    4. Volume profile
    5. Macro indicators (VIX, yield curve)
    """
    
    def __init__(self, history_size: int = 500):
        self.history_size = history_size
        
        # Factor history (bounded deques for memory safety)
        self.volatility_history: Deque[float] = deque(maxlen=history_size)
        self.trend_history: Deque[float] = deque(maxlen=history_size)
        self.correlation_history: Deque[float] = deque(maxlen=history_size)
        self.volume_ratio_history: Deque[float] = deque(maxlen=history_size)
        
        # Transition tracking
        self.transitions: List[RegimeTransition] = []
        self._current_regime: Optional[VolatilityRegime] = None
        self._regime_start_time: Optional[datetime] = None
        
        # Transition probability matrix (simplified)
        self.transition_counts: Dict[Tuple[VolatilityRegime, VolatilityRegime], int] = {}
    
    def add_volatility_reading(self, vol: float):
        """Add volatility reading (e.g., ATR or realized vol)."""
        self.volatility_history.append(vol)
    
    def add_trend_reading(self, trend: float):
        """Add trend strength reading (e.g., ADX or MA slope)."""
        self.trend_history.append(trend)
    
    def add_correlation_reading(self, corr: float):
        """Add correlation reading (e.g., BTC-SPX correlation)."""
        self.correlation_history.append(corr)
    
    def add_volume_ratio(self, ratio: float):
        """Add volume ratio (current vs average)."""
        self.volume_ratio_history.append(ratio)
    
    def detect_regime(self) -> Optional[RegimeResult]:
        """
        Detect current market regime from accumulated signals.
        
        Returns:
            RegimeResult with detected regime and recommendations
        """
        if len(self.volatility_history) < 20:
            return None
        
        now = datetime.now(timezone.utc)
        
        # Calculate factor statistics
        recent_vol = list(self.volatility_history)[-20:]
        recent_trend = list(self.trend_history)[-20:] if self.trend_history else [0.5]
        recent_corr = list(self.correlation_history)[-20:] if self.correlation_history else [0.0]
        
        vol_mean = statistics.mean(recent_vol)
        vol_std = statistics.stdev(recent_vol) if len(recent_vol) > 1 else 0
        
        trend_mean = statistics.mean(recent_trend)
        corr_mean = statistics.mean(recent_corr)
        
        # Classify volatility regime
        vol_percentile = self._calculate_percentile(vol_mean, list(self.volatility_history))
        
        if vol_percentile > 90:
            # High volatility
            if trend_mean > 0.3:
                regime = VolatilityRegime.HIGH_VOL_BULL
            elif trend_mean < -0.3:
                regime = VolatilityRegime.HIGH_VOL_BEAR
            else:
                regime = VolatilityRegime.CRISIS
        elif vol_percentile > 50:
            # Medium volatility
            if trend_mean > 0.2:
                regime = VolatilityRegime.LOW_VOL_BULL
            elif trend_mean < -0.2:
                regime = VolatilityRegime.LOW_VOL_BEAR
            else:
                regime = VolatilityRegime.TRANSITION
        else:
            # Low volatility
            if trend_mean > 0.1:
                regime = VolatilityRegime.LOW_VOL_BULL
            elif trend_mean < -0.1:
                regime = VolatilityRegime.LOW_VOL_BEAR
            else:
                regime = VolatilityRegime.TRANSITION
        
        # Check for regime transition
        if self._current_regime and regime != self._current_regime:
            self._record_transition(regime)
        
        self._current_regime = regime
        self._regime_start_time = now
        
        # Calculate macro state based on correlation and other factors
        macro_state = self._classify_macro_state(corr_mean, vol_mean, trend_mean)
        
        # Calculate confidence based on signal clarity
        confidence = self._calculate_confidence(vol_std, trend_mean, corr_mean)
        
        # Generate recommendations
        leverage = self._recommend_leverage(regime, confidence)
        stop_distance = self._recommend_stop_distance(regime, vol_mean)
        
        # Calculate transition probability
        transition_prob = self._estimate_transition_probability(regime)
        
        # Composite regime score (-1 bearish to +1 bullish)
        regime_score = trend_mean * (1 - vol_percentile / 100)
        
        return RegimeResult(
            timestamp=now,
            volatility_regime=regime,
            macro_state=macro_state,
            confidence=confidence,
            regime_score=regime_score,
            recommended_leverage=leverage,
            recommended_stop_distance=stop_distance,
            transition_probability=transition_prob
        )
    
    def _calculate_percentile(self, value: float, data: List[float]) -> float:
        """Calculate percentile rank of value in data."""
        if not data:
            return 50.0
        
        sorted_data = sorted(data)
        count_below = sum(1 for x in data if x < value)
        return (count_below / len(data)) * 100
    
    def _classify_macro_state(
        self,
        correlation: float,
        volatility: float,
        trend: float
    ) -> MacroState:
        """Classify macroeconomic state from market signals."""
        if correlation > 0.5 and volatility > 0.7:
            return MacroState.RISK_OFF
        elif correlation < 0.2 and trend > 0.3:
            return MacroState.RISK_ON
        elif volatility > 0.8 and trend < 0:
            return MacroState.RECESSION
        elif volatility > 0.6 and trend > 0.5:
            return MacroState.OVERHEATING
        elif 0.2 < correlation < 0.5:
            return MacroState.STAGFLATION
        else:
            return MacroState.REFLATION
    
    def _calculate_confidence(
        self,
        vol_std: float,
        trend: float,
        correlation: float
    ) -> float:
        """Calculate confidence in regime detection."""
        # Higher confidence when signals are clear and consistent
        base_confidence = 0.5
        
        # Reduce confidence if volatility is erratic
        if vol_std > 0.3:
            base_confidence -= 0.2
        
        # Increase confidence with strong trend
        if abs(trend) > 0.4:
            base_confidence += 0.2
        
        # Adjust for correlation clarity
        if abs(correlation) > 0.5:
            base_confidence += 0.1
        
        return max(0.1, min(0.99, base_confidence))
    
    def _recommend_leverage(self, regime: VolatilityRegime, confidence: float) -> float:
        """Recommend leverage based on regime."""
        base_leverage = {
            VolatilityRegime.LOW_VOL_BULL: 3.0,
            VolatilityRegime.LOW_VOL_BEAR: 1.5,
            VolatilityRegime.HIGH_VOL_BULL: 1.5,
            VolatilityRegime.HIGH_VOL_BEAR: 0.5,
            VolatilityRegime.TRANSITION: 0.5,
            VolatilityRegime.CRISIS: 0.25,
        }
        
        leverage = base_leverage.get(regime, 1.0)
        
        # Scale by confidence
        return leverage * confidence
    
    def _recommend_stop_distance(self, regime: VolatilityRegime, vol: float) -> float:
        """Recommend stop-loss distance based on regime and volatility."""
        base_distances = {
            VolatilityRegime.LOW_VOL_BULL: 2.0,
            VolatilityRegime.LOW_VOL_BEAR: 3.0,
            VolatilityRegime.HIGH_VOL_BULL: 5.0,
            VolatilityRegime.HIGH_VOL_BEAR: 8.0,
            VolatilityRegime.TRANSITION: 10.0,
            VolatilityRegime.CRISIS: 15.0,
        }
        
        base = base_distances.get(regime, 5.0)
        
        # Adjust by volatility level
        return base * (1 + vol)
    
    def _estimate_transition_probability(self, current_regime: VolatilityRegime) -> float:
        """Estimate probability of regime transition."""
        # Simplified: would use HMM transition matrix in production
        if current_regime == VolatilityRegime.TRANSITION:
            return 0.7
        elif current_regime == VolatilityRegime.CRISIS:
            return 0.4
        else:
            return 0.2
    
    def _record_transition(self, new_regime: VolatilityRegime):
        """Record a regime transition."""
        if self._current_regime and self._regime_start_time:
            duration = datetime.now(timezone.utc) - self._regime_start_time
            
            transition = RegimeTransition(
                from_regime=self._current_regime,
                to_regime=new_regime,
                timestamp=datetime.now(timezone.utc),
                trigger_factor="auto_detected",
                duration_in_previous=duration
            )
            
            self.transitions.append(transition)
            
            # Update transition counts
            key = (self._current_regime, new_regime)
            self.transition_counts[key] = self.transition_counts.get(key, 0) + 1
            
            logger.info(
                f"Regime transition: {self._current_regime.value} -> {new_regime.value} "
                f"(duration: {duration})"
            )
    
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get normalized transition probability matrix."""
        matrix = {}
        
        for (from_r, to_r), count in self.transition_counts.items():
            if from_r.value not in matrix:
                matrix[from_r.value] = {}
            
            # Normalize by total transitions from this regime
            total_from = sum(
                c for (f, _), c in self.transition_counts.items() if f == from_r
            )
            
            matrix[from_r.value][to_r.value] = count / total_from if total_from > 0 else 0
        
        return matrix


class RegimeSwitcher:
    """
    Main regime switcher combining detection with strategy adaptation.
    
    Features:
    - Real-time regime detection
    - Automatic parameter adjustment
    - Transition alerting
    - Historical regime analysis
    """
    
    def __init__(self, update_interval_seconds: float = 60.0):
        self.update_interval = update_interval_seconds
        self.detector = RegimeDetector()
        
        # Current state
        self._current_result: Optional[RegimeResult] = None
        self._subscribers: List[Callable[[RegimeResult], None]] = []
        
        # Running state
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info("RegimeSwitcher initialized")
    
    def subscribe(self, callback: Callable[[RegimeResult], None]):
        """Subscribe to regime updates."""
        self._subscribers.append(callback)
    
    def update_factors(
        self,
        volatility: float,
        trend: float,
        correlation: float,
        volume_ratio: float
    ):
        """Update all regime detection factors."""
        self.detector.add_volatility_reading(volatility)
        self.detector.add_trend_reading(trend)
        self.detector.add_correlation_reading(correlation)
        self.detector.add_volume_ratio(volume_ratio)
    
    def get_current_regime(self) -> Optional[RegimeResult]:
        """Get current regime result."""
        return self._current_result
    
    def detect_and_notify(self) -> Optional[RegimeResult]:
        """Run detection and notify subscribers."""
        result = self.detector.detect_regime()
        
        if result:
            self._current_result = result
            
            # Notify subscribers
            for subscriber in self._subscribers:
                try:
                    res = subscriber(result)
                    # Note: If async callback needed, use separate async method
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
            
            # Log significant changes
            if result.volatility_regime in [VolatilityRegime.CRISIS, VolatilityRegime.TRANSITION]:
                logger.warning(f"⚠️ ALERT: Market entered {result.volatility_regime.value} regime")
        
        return result
    
    async def detect_and_notify_async(self) -> Optional[RegimeResult]:
        """Async version of detect_and_notify for async subscribers."""
        result = self.detector.detect_regime()
        
        if result:
            self._current_result = result
            
            # Notify subscribers (supporting async callbacks)
            for subscriber in self._subscribers:
                try:
                    res = subscriber(result)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
            
            # Log significant changes
            if result.volatility_regime in [VolatilityRegime.CRISIS, VolatilityRegime.TRANSITION]:
                logger.warning(f"⚠️ ALERT: Market entered {result.volatility_regime.value} regime")
        
        return result
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get recommended strategy parameters for current regime."""
        if not self._current_result:
            return {"error": "No regime detected yet"}
        
        result = self._current_result
        
        return {
            "max_leverage": result.recommended_leverage,
            "stop_loss_pct": result.recommended_stop_distance,
            "take_profit_multiplier": 2.0 if result.volatility_regime in [
                VolatilityRegime.LOW_VOL_BULL, VolatilityRegime.HIGH_VOL_BULL
            ] else 1.5,
            "position_size_reduction": 0.5 if not result.is_safe_to_trade() else 1.0,
            "max_open_positions": 3 if result.is_safe_to_trade() else 1,
            "hedge_required": result.volatility_regime == VolatilityRegime.HIGH_VOL_BEAR,
        }
    
    def get_regime_history(self) -> List[Dict[str, Any]]:
        """Get recent regime history."""
        return [
            {
                "timestamp": t.timestamp.isoformat(),
                "from": t.from_regime.value,
                "to": t.to_regime.value,
                "duration_hours": t.duration_in_previous.total_seconds() / 3600
            }
            for t in self.detector.transitions[-10:]
        ]
    
    async def _monitoring_loop(self):
        """Continuous regime monitoring."""
        while self._running:
            try:
                await self.detect_and_notify_async()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in regime monitoring: {e}")
                await asyncio.sleep(60)
    
    async def start(self):
        """Start regime monitoring."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting RegimeSwitcher")
        
        self._tasks = [
            asyncio.create_task(self._monitoring_loop())
        ]
    
    async def stop(self):
        """Stop monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


# Example usage
async def main():
    """Demonstration of RegimeSwitcher functionality."""
    switcher = RegimeSwitcher(update_interval_seconds=5.0)
    
    def regime_handler(result: RegimeResult):
        print(
            f"📊 Regime: {result.volatility_regime.value} | "
            f"Macro: {result.macro_state.value} | "
            f"Confidence: {result.confidence:.2f}"
        )
        print(f"   Recommended Leverage: {result.recommended_leverage:.2f}x")
        print(f"   Stop Distance: {result.recommended_stop_distance:.2f}%")
    
    switcher.subscribe(regime_handler)
    
    # Simulate factor updates
    import random
    for i in range(10):
        switcher.update_factors(
            volatility=0.3 + random.uniform(-0.1, 0.2),
            trend=0.2 + random.uniform(-0.3, 0.3),
            correlation=random.uniform(-0.3, 0.7),
            volume_ratio=1.0 + random.uniform(-0.3, 0.5)
        )
        switcher.detect_and_notify()
        await asyncio.sleep(1)
    
    # Show strategy parameters
    print("\n📋 Current Strategy Parameters:")
    params = switcher.get_strategy_parameters()
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    await asyncio.sleep(2)
    await switcher.stop()


if __name__ == "__main__":
    asyncio.run(main())
