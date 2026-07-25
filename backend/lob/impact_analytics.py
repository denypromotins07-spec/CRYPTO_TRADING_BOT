#!/usr/bin/env python3
"""
Impact Analytics - Theoretical vs Actual Execution Comparison

This module compares theoretical impact models against actual execution slippage,
providing feedback for model calibration and execution quality assessment.

Key Features:
- TCA (Transaction Cost Analysis) metrics
- Model calibration feedback loop
- Slippage attribution (permanent vs temporary)
- Memory-efficient streaming calculations

Author: ZAID Personal Crypto Trading Bot
Stage: 22 - LOB Physics & Hawkes Processes
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time
from threading import Lock


class ImpactModelType(Enum):
    """Supported impact models for comparison."""
    ALMGREN_CHRISS = "almgren_chriss"
    SQUARE_ROOT = "square_root"
    PROPAGATOR = "propagator"
    LINEAR = "linear"


@dataclass
class ExecutionRecord:
    """Record of an executed order for analysis."""
    order_id: str
    timestamp: float
    asset: str
    side: str  # 'buy' or 'sell'
    requested_size: float
    executed_size: float
    arrival_price: float  # Price at order arrival
    vwap_execution: float  # Volume-weighted avg execution price
    final_price: float  # Price after execution completed
    twap_price: float  # Time-weighted average during execution
    
    # Theoretical predictions
    predicted_impact_bps: float
    predicted_temporary_bps: float
    predicted_permanent_bps: float
    
    # Model used
    model_type: ImpactModelType


@dataclass
class ImpactAnalyticsMetrics:
    """Aggregate analytics metrics."""
    # Slippage metrics
    avg_slippage_bps: float
    median_slippage_bps: float
    std_slippage_bps: float
    max_slippage_bps: float
    
    # Model accuracy
    model_bias: float  # Avg(predicted - actual)
    model_rmse: float  # Root mean squared error
    model_r_squared: float
    model_mae: float  # Mean absolute error
    
    # Impact decomposition
    avg_temporary_impact_pct: float
    avg_permanent_impact_pct: float
    
    # Execution quality
    implementation_shortfall_bps: float
    market_timing_cost_bps: float
    
    # Sample statistics
    sample_count: int
    last_update: float


class ImpactAnalytics:
    """
    Analyzes execution quality and calibrates impact models.
    
    Compares theoretical impact predictions against actual execution
    results to improve model accuracy over time.
    """
    
    def __init__(
        self,
        window_size: int = 10000,
        min_samples_for_calibration: int = 100,
    ):
        """
        Initialize impact analytics.
        
        Args:
            window_size: Maximum executions to track
            min_samples_for_calibration: Minimum samples before recalibration
        """
        self.window_size = window_size
        self.min_samples = min_samples_for_calibration
        
        # Execution tracking
        self._executions: Deque[ExecutionRecord] = deque(maxlen=window_size)
        
        # Running statistics for O(1) updates
        self._sum_slippage: float = 0.0
        self._sum_slippage_sq: float = 0.0
        self._sum_prediction_error: float = 0.0
        self._sum_prediction_error_sq: float = 0.0
        self._sum_predicted: float = 0.0
        self._sum_actual: float = 0.0
        self._sum_predicted_actual: float = 0.0
        
        # Impact decomposition tracking
        self._sum_temporary: float = 0.0
        self._sum_permanent: float = 0.0
        self._sum_total_impact: float = 0.0
        
        # Thread safety
        self._lock = Lock()
        
        # Calibration state
        self._calibration_factor: float = 1.0
        self._is_calibrated: bool = False
    
    def record_execution(self, record: ExecutionRecord) -> None:
        """
        Record a new execution for analysis.
        
        Args:
            record: ExecutionRecord with all relevant details
        """
        with self._lock:
            # Calculate actual slippage in bps
            if record.side.lower() == 'buy':
                slippage_bps = (record.vwap_execution - record.arrival_price) / record.arrival_price * 10000
            else:
                slippage_bps = (record.arrival_price - record.vwap_execution) / record.arrival_price * 10000
            
            # Store execution
            self._executions.append(record)
            
            # Update running statistics
            self._sum_slippage += slippage_bps
            self._sum_slippage_sq += slippage_bps ** 2
            
            prediction_error = record.predicted_impact_bps - slippage_bps
            self._sum_prediction_error += prediction_error
            self._sum_prediction_error_sq += prediction_error ** 2
            
            self._sum_predicted += record.predicted_impact_bps
            self._sum_actual += slippage_bps
            self._sum_predicted_actual += record.predicted_impact_bps * slippage_bps
            
            # Impact decomposition
            self._sum_temporary += record.predicted_temporary_bps
            self._sum_permanent += record.predicted_permanent_bps
            self._sum_total_impact += record.predicted_impact_bps
            
            # Check if recalibration needed
            if len(self._executions) >= self.min_samples and not self._is_calibrated:
                self._recalibrate()
    
    def _recalibrate(self) -> None:
        """Recalibrate model based on historical errors."""
        if len(self._executions) < self.min_samples:
            return
        
        n = len(self._executions)
        
        # Calculate average bias
        avg_bias = self._sum_prediction_error / n
        
        # Update calibration factor
        if abs(self._sum_predicted) > 1e-10:
            self._calibration_factor = self._sum_actual / self._sum_predicted
            self._calibration_factor = max(0.5, min(2.0, self._calibration_factor))  # Bound adjustments
            self._is_calibrated = True
    
    def get_metrics(self) -> ImpactAnalyticsMetrics:
        """
        Get current analytics metrics.
        
        Returns:
            ImpactAnalyticsMetrics with comprehensive statistics
        """
        with self._lock:
            n = len(self._executions)
            
            if n == 0:
                return ImpactAnalyticsMetrics(
                    avg_slippage_bps=0.0,
                    median_slippage_bps=0.0,
                    std_slippage_bps=0.0,
                    max_slippage_bps=0.0,
                    model_bias=0.0,
                    model_rmse=0.0,
                    model_r_squared=0.0,
                    model_mae=0.0,
                    avg_temporary_impact_pct=0.0,
                    avg_permanent_impact_pct=0.0,
                    implementation_shortfall_bps=0.0,
                    market_timing_cost_bps=0.0,
                    sample_count=0,
                    last_update=time.time(),
                )
            
            # Basic slippage statistics
            avg_slippage = self._sum_slippage / n
            variance = (self._sum_slippage_sq / n) - (avg_slippage ** 2)
            std_slippage = np.sqrt(max(0, variance))
            
            # Median and max from recent executions
            slippages = []
            for exec in self._executions:
                if exec.side.lower() == 'buy':
                    slip = (exec.vwap_execution - exec.arrival_price) / exec.arrival_price * 10000
                else:
                    slip = (exec.arrival_price - exec.vwap_execution) / exec.arrival_price * 10000
                slippages.append(slip)
            
            median_slippage = np.median(slippages) if slippages else 0.0
            max_slippage = max(slippages) if slippages else 0.0
            
            # Model accuracy metrics
            model_bias = self._sum_prediction_error / n
            model_rmse = np.sqrt(max(0, self._sum_prediction_error_sq / n))
            model_mae = abs(model_bias)  # Simplified
            
            # R-squared calculation
            ss_tot = self._sum_slippage_sq - n * avg_slippage ** 2
            ss_res = self._sum_prediction_error_sq
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            r_squared = max(0, min(1, r_squared))
            
            # Impact decomposition
            total_impact = self._sum_total_impact
            temp_pct = (self._sum_temporary / total_impact * 100) if total_impact > 0 else 0
            perm_pct = (self._sum_permanent / total_impact * 100) if total_impact > 0 else 0
            
            # Implementation shortfall
            impl_shortfall = avg_slippage
            
            # Market timing cost (simplified)
            timing_cost = std_slippage * 0.5  # Approximation
            
            return ImpactAnalyticsMetrics(
                avg_slippage_bps=avg_slippage,
                median_slippage_bps=median_slippage,
                std_slippage_bps=std_slippage,
                max_slippage_bps=max_slippage,
                model_bias=model_bias,
                model_rmse=model_rmse,
                model_r_squared=r_squared,
                model_mae=model_mae,
                avg_temporary_impact_pct=temp_pct,
                avg_permanent_impact_pct=perm_pct,
                implementation_shortfall_bps=impl_shortfall,
                market_timing_cost_bps=timing_cost,
                sample_count=n,
                last_update=time.time(),
            )
    
    def get_calibration_factor(self) -> float:
        """Get current model calibration factor."""
        with self._lock:
            return self._calibration_factor
    
    def adjust_prediction(self, predicted_bps: float) -> float:
        """
        Adjust a prediction using the calibration factor.
        
        Args:
            predicted_bps: Raw model prediction
            
        Returns:
            Calibrated prediction
        """
        with self._lock:
            return predicted_bps * self._calibration_factor
    
    def get_model_accuracy_by_type(self) -> Dict[str, Dict[str, float]]:
        """Get model accuracy broken down by model type."""
        with self._lock:
            by_type: Dict[str, List[Tuple[float, float]]] = {}
            
            for exec in self._executions:
                model = exec.model_type.value
                
                if exec.side.lower() == 'buy':
                    actual = (exec.vwap_execution - exec.arrival_price) / exec.arrival_price * 10000
                else:
                    actual = (exec.arrival_price - exec.vwap_execution) / exec.arrival_price * 10000
                
                error = exec.predicted_impact_bps - actual
                
                if model not in by_type:
                    by_type[model] = []
                by_type[model].append((exec.predicted_impact_bps, error))
            
            results = {}
            for model, pairs in by_type.items():
                if len(pairs) >= 5:
                    errors = [e for _, e in pairs]
                    predictions = [p for p, _ in pairs]
                    
                    results[model] = {
                        'sample_count': len(pairs),
                        'mean_error': np.mean(errors),
                        'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
                        'mean_prediction': np.mean(predictions),
                    }
            
            return results
    
    def reset(self) -> None:
        """Reset all analytics."""
        with self._lock:
            self._executions.clear()
            self._sum_slippage = 0.0
            self._sum_slippage_sq = 0.0
            self._sum_prediction_error = 0.0
            self._sum_prediction_error_sq = 0.0
            self._sum_predicted = 0.0
            self._sum_actual = 0.0
            self._sum_predicted_actual = 0.0
            self._sum_temporary = 0.0
            self._sum_permanent = 0.0
            self._sum_total_impact = 0.0
            self._calibration_factor = 1.0
            self._is_calibrated = False


if __name__ == "__main__":
    # Demo usage
    import random
    
    analytics = ImpactAnalytics()
    
    # Simulate some executions
    for i in range(200):
        side = random.choice(['buy', 'sell'])
        arrival = 50000.0 + random.uniform(-500, 500)
        
        # Simulate slippage
        true_impact = random.uniform(5, 20)
        noise = random.gauss(0, 3)
        actual_impact = true_impact + noise
        
        if side == 'buy':
            vwap = arrival * (1 + actual_impact / 10000)
        else:
            vwap = arrival * (1 - actual_impact / 10000)
        
        # Model prediction (slightly biased)
        predicted = true_impact * 1.1
        
        record = ExecutionRecord(
            order_id=f"order_{i}",
            timestamp=time.time(),
            asset="BTC",
            side=side,
            requested_size=random.uniform(1, 10),
            executed_size=random.uniform(0.9, 1.0) * random.uniform(1, 10),
            arrival_price=arrival,
            vwap_execution=vwap,
            final_price=arrival,
            twap_price=(arrival + vwap) / 2,
            predicted_impact_bps=predicted,
            predicted_temporary_bps=predicted * 0.7,
            predicted_permanent_bps=predicted * 0.3,
            model_type=ImpactModelType.SQUARE_ROOT,
        )
        
        analytics.record_execution(record)
    
    metrics = analytics.get_metrics()
    print(f"Sample Count: {metrics.sample_count}")
    print(f"Avg Slippage: {metrics.avg_slippage_bps:.2f} bps")
    print(f"Model RMSE: {metrics.model_rmse:.2f} bps")
    print(f"Model R²: {metrics.model_r_squared:.3f}")
    print(f"Calibration Factor: {analytics.get_calibration_factor():.3f}")
    print(f"Temporary Impact: {metrics.avg_temporary_impact_pct:.1f}%")
    print(f"Permanent Impact: {metrics.avg_permanent_impact_pct:.1f}%")
