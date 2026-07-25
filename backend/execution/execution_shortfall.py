#!/usr/bin/env python3
"""
Execution Shortfall: Deviation Measurement from Theoretical VWAP Curve
Measures and analyzes execution deviation from the theoretical VWAP curve.
Provides real-time feedback for execution quality assessment.

Stage 13: Advanced Execution Algorithms
Target: Minimize market impact to secure 8k-20k INR/hour
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import threading


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class ExecutionSlice:
    """Individual execution slice record"""
    slice_id: int
    scheduled_time_ns: int
    actual_time_ns: int
    target_volume: float
    filled_volume: float
    target_price: float  # Theoretical VWAP at scheduled time
    actual_price: float  # Actual fill price
    timing_deviation_ms: float
    price_deviation_bps: float
    volume_deviation_pct: float


@dataclass
class ExecutionShortfallMetrics:
    """Comprehensive execution shortfall metrics"""
    total_shortfall_bps: float
    timing_cost_bps: float
    price_impact_bps: float
    opportunity_cost_bps: float
    vwap_deviation_bps: float
    twap_deviation_bps: float
    fill_rate: float
    slippage_bps: float
    alpha_decay_point_us: Optional[int]  # Microsecond where alpha decay occurred


class ExecutionShortfallAnalyzer:
    """
    Analyzes execution shortfall against theoretical VWAP curve.
    Identifies exact microsecond where alpha decay occurred.
    
    Features:
    - Real-time deviation tracking
    - Component decomposition (timing, price impact, opportunity)
    - Alpha decay detection
    - Fill rate analysis
    - Thread-safe state management
    """
    
    def __init__(
        self,
        benchmark_type: str = "vwap",  # "vwap", "twap", "arrival"
        max_acceptable_deviation_bps: float = 50.0,
        alpha_decay_threshold_bps: float = 20.0,
    ):
        self.benchmark_type = benchmark_type
        self.max_acceptable_deviation_bps = max_acceptable_deviation_bps
        self.alpha_decay_threshold_bps = alpha_decay_threshold_bps
        
        # Execution tracking
        self._slices: List[ExecutionSlice] = []
        self._theoretical_vwap_curve: Optional[np.ndarray] = None
        self._actual_execution_prices: List[float] = []
        self._execution_times_ns: List[int] = []
        
        # Benchmark tracking
        self._benchmark_prices: List[float] = []
        self._benchmark_times_ns: List[int] = []
        
        # State
        self._total_target_volume: float = 0.0
        self._total_filled_volume: float = 0.0
        self._total_target_value: float = 0.0
        self._total_actual_value: float = 0.0
        self._start_time_ns: int = 0
        
        # Alpha decay tracking
        self._alpha_decay_detected: bool = False
        self._alpha_decay_time_us: Optional[int] = None
        self._cumulative_shortfall_history: List[Tuple[int, float]] = []
        
        # Thread safety
        self._lock = threading.RLock()
    
    def set_theoretical_vwap_curve(self, vwap_curve: np.ndarray, times_ns: np.ndarray) -> None:
        """Set the theoretical VWAP curve for comparison"""
        with self._lock:
            if len(vwap_curve) != len(times_ns):
                raise ValueError("VWAP curve and times must have same length")
            self._theoretical_vwap_curve = vwap_curve.copy()
            self._benchmark_times_ns = times_ns.tolist()
            self._benchmark_prices = vwap_curve.tolist()
    
    def start_execution(
        self, 
        total_volume: float, 
        side: OrderSide,
        start_time_ns: Optional[int] = None
    ) -> None:
        """Start tracking a new execution"""
        with self._lock:
            self._total_target_volume = total_volume
            self._total_filled_volume = 0.0
            self._total_target_value = 0.0
            self._total_actual_value = 0.0
            self._start_time_ns = start_time_ns or time.time_ns()
            self._slices.clear()
            self._actual_execution_prices.clear()
            self._execution_times_ns.clear()
            self._alpha_decay_detected = False
            self._alpha_decay_time_us = None
            self._cumulative_shortfall_history.clear()
    
    def record_slice(
        self,
        slice_id: int,
        scheduled_time_ns: int,
        actual_time_ns: int,
        target_volume: float,
        filled_volume: float,
        target_price: float,
        actual_price: float,
    ) -> ExecutionSlice:
        """Record an executed slice"""
        with self._lock:
            # Calculate deviations
            timing_deviation_ms = (actual_time_ns - scheduled_time_ns) / 1_000_000
            
            if target_price > 0:
                price_deviation_bps = ((actual_price - target_price) / target_price) * 10000
            else:
                price_deviation_bps = 0.0
            
            if target_volume > 0:
                volume_deviation_pct = ((filled_volume - target_volume) / target_volume) * 100
            else:
                volume_deviation_pct = 0.0
            
            slice_record = ExecutionSlice(
                slice_id=slice_id,
                scheduled_time_ns=scheduled_time_ns,
                actual_time_ns=actual_time_ns,
                target_volume=target_volume,
                filled_volume=filled_volume,
                target_price=target_price,
                actual_price=actual_price,
                timing_deviation_ms=timing_deviation_ms,
                price_deviation_bps=price_deviation_bps,
                volume_deviation_pct=volume_deviation_pct,
            )
            
            self._slices.append(slice_record)
            self._actual_execution_prices.append(actual_price)
            self._execution_times_ns.append(actual_time_ns)
            
            # Update totals
            self._total_filled_volume += filled_volume
            self._total_actual_value += actual_price * filled_volume
            self._total_target_value += target_price * target_volume
            
            # Check for alpha decay
            self._check_alpha_decay(actual_time_ns, price_deviation_bps)
            
            return slice_record
    
    def _check_alpha_decay(self, current_time_ns: int, current_deviation_bps: float) -> None:
        """Detect the exact microsecond where alpha decay occurred"""
        if self._alpha_decay_detected:
            return
        
        # Track cumulative shortfall
        elapsed_us = (current_time_ns - self._start_time_ns) // 1000
        self._cumulative_shortfall_history.append((elapsed_us, current_deviation_bps))
        
        # Detect significant deviation indicating alpha decay
        if abs(current_deviation_bps) > self.alpha_decay_threshold_bps:
            self._alpha_decay_detected = True
            self._alpha_decay_time_us = elapsed_us
    
    def get_shortfall_metrics(self) -> Optional[ExecutionShortfallMetrics]:
        """Calculate comprehensive execution shortfall metrics"""
        with self._lock:
            if not self._slices or self._total_target_volume == 0:
                return None
            
            # Calculate average prices
            avg_actual_price = self._total_actual_value / self._total_filled_volume if self._total_filled_volume > 0 else 0
            avg_target_price = self._total_target_value / self._total_target_volume
            
            if avg_target_price == 0:
                return None
            
            # Total shortfall in bps
            total_shortfall_bps = ((avg_actual_price - avg_target_price) / avg_target_price) * 10000
            
            # Decompose into components
            timing_costs = [abs(s.timing_deviation_ms) * 0.1 for s in self._slices]  # Simplified model
            price_impacts = [s.price_deviation_bps * 0.7 for s in self._slices]  # 70% attributed to impact
            opportunity_costs = [abs(s.volume_deviation_pct) * 0.2 for s in self._slices]  # 20% for opportunity
            
            timing_cost_bps = np.mean(timing_costs) if timing_costs else 0
            price_impact_bps = np.mean(price_impacts) if price_impacts else 0
            opportunity_cost_bps = np.mean(opportunity_costs) if opportunity_costs else 0
            
            # VWAP deviation
            vwap_deviation_bps = total_shortfall_bps
            
            # TWAP deviation (simplified)
            twap_prices = [s.target_price for s in self._slices]
            twap_avg = np.mean(twap_prices) if twap_prices else avg_target_price
            twap_deviation_bps = ((avg_actual_price - twap_avg) / twap_avg) * 10000 if twap_avg > 0 else 0
            
            # Fill rate
            fill_rate = self._total_filled_volume / self._total_target_volume
            
            # Slippage
            slippage_bps = total_shortfall_bps
            
            return ExecutionShortfallMetrics(
                total_shortfall_bps=total_shortfall_bps,
                timing_cost_bps=timing_cost_bps,
                price_impact_bps=price_impact_bps,
                opportunity_cost_bps=opportunity_cost_bps,
                vwap_deviation_bps=vwap_deviation_bps,
                twap_deviation_bps=twap_deviation_bps,
                fill_rate=fill_rate,
                slippage_bps=slippage_bps,
                alpha_decay_point_us=self._alpha_decay_time_us,
            )
    
    def get_alpha_decay_analysis(self) -> Dict:
        """Detailed analysis of alpha decay point"""
        with self._lock:
            if not self._alpha_decay_detected:
                return {"detected": False}
            
            # Find the inflection point
            inflection_point = None
            max_acceleration = 0
            
            for i in range(1, len(self._cumulative_shortfall_history)):
                prev_time, prev_dev = self._cumulative_shortfall_history[i - 1]
                curr_time, curr_dev = self._cumulative_shortfall_history[i]
                
                time_delta = curr_time - prev_time
                if time_delta > 0:
                    acceleration = (curr_dev - prev_dev) / time_delta
                    if abs(acceleration) > abs(max_acceleration):
                        max_acceleration = acceleration
                        inflection_point = curr_time
            
            return {
                "detected": True,
                "decay_time_us": self._alpha_decay_time_us,
                "inflection_point_us": inflection_point,
                "max_acceleration_bps_per_us": max_acceleration,
                "total_slices_at_decay": sum(
                    1 for s in self._slices 
                    if (s.actual_time_ns - self._start_time_ns) // 1000 <= (self._alpha_decay_time_us or 0)
                ),
            }
    
    def get_slice_analysis(self, slice_id: int) -> Optional[Dict]:
        """Get detailed analysis for a specific slice"""
        with self._lock:
            for slice_record in self._slices:
                if slice_record.slice_id == slice_id:
                    return {
                        "slice_id": slice_record.slice_id,
                        "timing_deviation_ms": slice_record.timing_deviation_ms,
                        "price_deviation_bps": slice_record.price_deviation_bps,
                        "volume_deviation_pct": slice_record.volume_deviation_pct,
                        "target_vs_actual_price": f"{slice_record.target_price:.2f} vs {slice_record.actual_price:.2f}",
                        "fill_quality": "good" if abs(slice_record.price_deviation_bps) < 10 else "poor",
                    }
            return None
    
    def reset(self) -> None:
        """Reset analyzer state"""
        with self._lock:
            self._slices.clear()
            self._actual_execution_prices.clear()
            self._execution_times_ns.clear()
            self._total_target_volume = 0.0
            self._total_filled_volume = 0.0
            self._total_target_value = 0.0
            self._total_actual_value = 0.0
            self._alpha_decay_detected = False
            self._alpha_decay_time_us = None
            self._cumulative_shortfall_history.clear()


if __name__ == "__main__":
    # Test execution shortfall analyzer
    analyzer = ExecutionShortfallAnalyzer(
        benchmark_type="vwap",
        max_acceptable_deviation_bps=50.0,
        alpha_decay_threshold_bps=20.0,
    )
    
    # Set theoretical VWAP curve (simplified)
    vwap_curve = np.array([50000.0 + i * 0.5 for i in range(100)])
    times_ns = np.array([time.time_ns() + i * 60_000_000_000 for i in range(100)])
    analyzer.set_theoretical_vwap_curve(vwap_curve, times_ns)
    
    # Start execution
    analyzer.start_execution(total_volume=10000, side=OrderSide.BUY)
    
    # Simulate slices
    base_time = time.time_ns()
    for i in range(10):
        scheduled_time = base_time + i * 60_000_000_000
        actual_time = scheduled_time + int(np.random.randn() * 100_000_000)  # ±100ms jitter
        target_vol = 1000
        filled_vol = target_vol + np.random.randn() * 50
        target_price = 50000 + i * 0.5
        actual_price = target_price + np.random.randn() * 5  # Some slippage
        
        analyzer.record_slice(
            slice_id=i,
            scheduled_time_ns=scheduled_time,
            actual_time_ns=actual_time,
            target_volume=target_vol,
            filled_volume=filled_vol,
            target_price=target_price,
            actual_price=actual_price,
        )
    
    # Get metrics
    metrics = analyzer.get_shortfall_metrics()
    if metrics:
        print(f"Total Shortfall: {metrics.total_shortfall_bps:.2f} bps")
        print(f"Timing Cost: {metrics.timing_cost_bps:.2f} bps")
        print(f"Price Impact: {metrics.price_impact_bps:.2f} bps")
        print(f"Fill Rate: {metrics.fill_rate:.2%}")
        print(f"Alpha Decay Point: {metrics.alpha_decay_point_us} μs" if metrics.alpha_decay_point_us else "No alpha decay detected")
    
    # Alpha decay analysis
    decay_analysis = analyzer.get_alpha_decay_analysis()
    print(f"\nAlpha Decay Analysis: {decay_analysis}")
