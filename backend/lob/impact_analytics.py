#!/usr/bin/env python3
"""
Impact Analytics - Compare Theoretical vs Actual Execution Impact

This module compares theoretical impact models against actual execution slippage,
isolating permanent vs temporary components of market impact.

Key Features:
- Theoretical vs actual impact comparison
- Permanent/temporary impact decomposition
- Model calibration from execution data
- Slippage attribution analysis
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
from threading import Lock


@dataclass
class ExecutionRecord:
    """Record of an actual execution."""
    timestamp_us: int
    symbol: str
    side: int  # 1=buy, -1=sell
    quantity: float
    avg_price: float
    arrival_price: float  # Price at order arrival
    vwap: float  # Market VWAP during execution
    participation_rate: float
    volatility: float


@dataclass
class ImpactDecomposition:
    """Decomposition of market impact into components."""
    # Total realized impact (bps)
    total_impact_bps: float
    # Permanent impact component (bps)
    permanent_impact_bps: float
    # Temporary impact component (bps)
    temporary_impact_bps: float
    # Timing luck component (bps)
    timing_luck_bps: float
    # Spread cost (bps)
    spread_cost_bps: float
    # Model prediction error (bps)
    prediction_error_bps: float


@dataclass
class ModelCalibration:
    """Calibrated impact model parameters."""
    # Linear coefficient
    alpha: float
    # Square-root coefficient
    beta: float
    # Permanent impact fraction
    permanent_fraction: float
    # R-squared of fit
    r_squared: float
    # Number of observations
    n_observations: int


class ImpactAnalytics:
    """
    Analyze execution impact and compare against theoretical models.
    
    This class decomposes realized slippage into permanent impact,
    temporary impact, and timing components for model validation.
    """
    
    def __init__(self):
        """Initialize the impact analytics engine."""
        self.executions: List[ExecutionRecord] = []
        self.impact_observations: List[Dict] = []
        
        # Model parameters (to be calibrated)
        self.alpha = 0.1  # Linear coefficient
        self.beta = 0.8   # Square-root coefficient
        self.permanent_frac = 0.3
        
        # Thread safety
        self._lock = Lock()
    
    def add_execution(self, record: ExecutionRecord) -> None:
        """Record an execution for analysis."""
        with self._lock:
            self.executions.append(record)
            
            # Calculate impact metrics
            impact_data = self._analyze_execution(record)
            if impact_data:
                self.impact_observations.append(impact_data)
                
                # Periodically recalibrate
                if len(self.impact_observations) % 50 == 0:
                    self._recalibrate_model()
    
    def _analyze_execution(
        self, 
        record: ExecutionRecord
    ) -> Optional[Dict]:
        """Analyze a single execution."""
        # Realized impact vs arrival price
        if record.side == 1:  # Buy
            realized_impact = (record.avg_price - record.arrival_price) / record.arrival_price
        else:  # Sell
            realized_impact = (record.arrival_price - record.avg_price) / record.arrival_price
        
        realized_impact_bps = realized_impact * 10000
        
        # Impact vs VWAP (market movement during execution)
        if record.side == 1:
            market_impact = (record.vwap - record.arrival_price) / record.arrival_price
        else:
            market_impact = (record.arrival_price - record.vwap) / record.arrival_price
        
        market_impact_bps = market_impact * 10000
        
        # Estimate permanent vs temporary
        # Permanent ≈ market impact, Temporary ≈ realized - market
        permanent_bps = market_impact_bps
        temporary_bps = realized_impact_bps - permanent_bps
        
        # Theoretical prediction (square-root law)
        predicted_impact = self._predict_impact(
            record.quantity,
            record.participation_rate,
            record.volatility,
        )
        
        prediction_error = realized_impact_bps - predicted_impact
        
        return {
            'symbol': record.symbol,
            'quantity': record.quantity,
            'participation': record.participation_rate,
            'volatility': record.volatility,
            'realized_bps': realized_impact_bps,
            'permanent_bps': permanent_bps,
            'temporary_bps': temporary_bps,
            'predicted_bps': predicted_impact,
            'error_bps': prediction_error,
        }
    
    def _predict_impact(
        self,
        quantity: float,
        participation: float,
        volatility: float,
    ) -> float:
        """Predict impact using square-root law."""
        # Impact = α * participation + β * σ * sqrt(participation)
        linear_part = self.alpha * participation * 10000  # Convert to bps
        sqrt_part = self.beta * volatility * math.sqrt(participation) * 10000
        
        return linear_part + sqrt_part
    
    def _recalibrate_model(self) -> None:
        """Recalibrate model parameters from observations."""
        if len(self.impact_observations) < 20:
            return
        
        # Simple moment-based calibration
        # This is simplified; production would use proper regression
        
        sum_realized = 0.0
        sum_sqrt_part = 0.0
        sum_linear_part = 0.0
        n = 0
        
        for obs in self.impact_observations[-100:]:  # Use last 100
            part = obs['participation']
            vol = obs['volatility']
            realized = obs['realized_bps']
            
            if part > 0 and vol > 0:
                sum_realized += realized
                sum_sqrt_part += vol * math.sqrt(part)
                sum_linear_part += part
                n += 1
        
        if n < 10:
            return
        
        # Simplified calibration (OLS would be better)
        avg_realized = sum_realized / n
        avg_sqrt = sum_sqrt_part / n
        avg_linear = sum_linear_part / n
        
        # Estimate beta from sqrt component
        if avg_sqrt > 0:
            self.beta = avg_realized * 0.7 / avg_sqrt
        
        # Estimate alpha from linear component
        if avg_linear > 0:
            self.alpha = avg_realized * 0.3 / avg_linear
        
        # Update permanent fraction estimate
        perm_sum = sum(obs['permanent_bps'] for obs in self.impact_observations[-100:])
        total_sum = sum(obs['realized_bps'] for obs in self.impact_observations[-100:])
        
        if total_sum > 0:
            self.permanent_frac = perm_sum / total_sum
    
    def decompose_impact(
        self,
        quantity: float,
        avg_price: float,
        arrival_price: float,
        vwap: float,
        side: int,
    ) -> ImpactDecomposition:
        """
        Decompose impact for a specific execution.
        
        Args:
            quantity: Executed quantity
            avg_price: Average execution price
            arrival_price: Price at order arrival
            vwap: Market VWAP during execution
            side: Order side (1=buy, -1=sell)
            
        Returns:
            ImpactDecomposition with all components
        """
        # Total realized impact
        if side == 1:
            total_impact = (avg_price - arrival_price) / arrival_price
            market_move = (vwap - arrival_price) / arrival_price
        else:
            total_impact = (arrival_price - avg_price) / arrival_price
            market_move = (arrival_price - vwap) / arrival_price
        
        total_bps = total_impact * 10000
        permanent_bps = market_move * 10000
        temporary_bps = total_bps - permanent_bps
        
        # Timing luck (difference from VWAP)
        if side == 1:
            timing_bps = (avg_price - vwap) / vwap * 10000
        else:
            timing_bps = (vwap - avg_price) / vwap * 10000
        
        # Spread cost (assume half-spread)
        spread_bps = 2.5  # Typical BTC spread
        
        # Model prediction
        participation = quantity / 100000  # Simplified ADV assumption
        predicted = self._predict_impact(quantity, participation, 0.02)
        prediction_error = total_bps - predicted
        
        return ImpactDecomposition(
            total_impact_bps=total_bps,
            permanent_impact_bps=permanent_bps,
            temporary_impact_bps=temporary_bps,
            timing_luck_bps=timing_bps,
            spread_cost_bps=spread_bps,
            prediction_error_bps=prediction_error,
        )
    
    def get_calibration(self) -> ModelCalibration:
        """Get current model calibration statistics."""
        with self._lock:
            # Calculate R-squared
            if len(self.impact_observations) < 10:
                return ModelCalibration(
                    alpha=self.alpha,
                    beta=self.beta,
                    permanent_fraction=self.permanent_frac,
                    r_squared=0.0,
                    n_observations=len(self.impact_observations),
                )
            
            # Compute prediction errors
            predictions = [obs['predicted_bps'] for obs in self.impact_observations[-100:]]
            actuals = [obs['realized_bps'] for obs in self.impact_observations[-100:]]
            
            mean_actual = sum(actuals) / len(actuals)
            ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
            ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return ModelCalibration(
                alpha=self.alpha,
                beta=self.beta,
                permanent_fraction=self.permanent_frac,
                r_squared=max(0.0, r_squared),
                n_observations=len(self.impact_observations),
            )
    
    def get_average_impact_by_symbol(self) -> Dict[str, float]:
        """Get average realized impact by symbol."""
        with self._lock:
            symbol_impacts: Dict[str, List[float]] = {}
            
            for obs in self.impact_observations:
                sym = obs['symbol']
                if sym not in symbol_impacts:
                    symbol_impacts[sym] = []
                symbol_impacts[sym].append(obs['realized_bps'])
            
            return {
                sym: sum(impacts) / len(impacts)
                for sym, impacts in symbol_impacts.items()
            }
    
    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self.executions.clear()
            self.impact_observations.clear()


if __name__ == "__main__":
    # Example usage
    analytics = ImpactAnalytics()
    
    # Add some sample executions
    import time
    base_time = int(time.time() * 1_000_000)
    
    for i in range(20):
        record = ExecutionRecord(
            timestamp_us=base_time + i * 1000000,
            symbol='BTC',
            side=1,
            quantity=1.0 + i * 0.1,
            avg_price=50000.0 + i * 5,
            arrival_price=50000.0,
            vwap=50000.0 + i * 3,
            participation_rate=0.001 + i * 0.0001,
            volatility=0.02,
        )
        analytics.add_execution(record)
    
    # Get decomposition for a sample trade
    decomp = analytics.decompose_impact(
        quantity=5.0,
        avg_price=50025.0,
        arrival_price=50000.0,
        vwap=50015.0,
        side=1,
    )
    
    print(f"Total Impact: {decomp.total_impact_bps:.2f} bps")
    print(f"Permanent: {decomp.permanent_impact_bps:.2f} bps")
    print(f"Temporary: {decomp.temporary_impact_bps:.2f} bps")
    print(f"Timing Luck: {decomp.timing_luck_bps:.2f} bps")
    print(f"Prediction Error: {decomp.prediction_error_bps:.2f} bps")
    
    # Get calibration
    calib = analytics.get_calibration()
    print(f"\nModel Calibration:")
    print(f"  Alpha: {calib.alpha:.4f}")
    print(f"  Beta: {calib.beta:.4f}")
    print(f"  Permanent Fraction: {calib.permanent_fraction:.2%}")
    print(f"  R²: {calib.r_squared:.4f}")
