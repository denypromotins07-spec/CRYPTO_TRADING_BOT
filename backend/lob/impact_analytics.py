#!/usr/bin/env python3
"""Impact Analytics - Compares theoretical impact models against actual execution slippage."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from threading import Lock


@dataclass(slots=True)
class ExecutionRecord:
    timestamp: float
    asset: str
    side: str
    size: float
    expected_impact_bps: float
    actual_slippage_bps: float
    model_type: str


@dataclass(slots=True)
class AnalyticsSummary:
    mean_error_bps: float
    rmse_bps: float
    bias: float  # Positive = model underestimates
    r_squared: float
    sample_count: int


class ImpactAnalytics:
    """Compare theoretical vs actual impact."""
    
    def __init__(self):
        self.records: List[ExecutionRecord] = []
        self._lock = Lock()
        
    def record_execution(self, asset: str, side: str, size: float,
                        expected_bps: float, actual_bps: float,
                        model: str = "square_root") -> None:
        import time
        with self._lock:
            self.records.append(ExecutionRecord(
                timestamp=time.time(),
                asset=asset,
                side=side,
                size=size,
                expected_impact_bps=expected_bps,
                actual_slippage_bps=actual_bps,
                model_type=model,
            ))
            if len(self.records) > 1000:
                self.records = self.records[-500:]
    
    def get_summary(self, asset: str = None) -> AnalyticsSummary:
        with self._lock:
            filtered = self.records if not asset else [r for r in self.records if r.asset == asset]
            
            if len(filtered) < 2:
                return AnalyticsSummary(0.0, 0.0, 0.0, 0.0, len(filtered))
            
            errors = [r.actual_slippage_bps - r.expected_impact_bps for r in filtered]
            mean_error = sum(errors) / len(errors)
            
            mse = sum(e**2 for e in errors) / len(errors)
            rmse = mse ** 0.5
            
            # Bias: positive means model underestimates impact
            bias = mean_error
            
            # R-squared calculation
            expected = [r.expected_impact_bps for r in filtered]
            actual = [r.actual_slippage_bps for r in filtered]
            mean_expected = sum(expected) / len(expected)
            mean_actual = sum(actual) / len(actual)
            
            ss_tot = sum((a - mean_actual)**2 for a in actual)
            ss_res = sum((a - e)**2 for a, e in zip(actual, expected))
            
            r_squared = 1 - (ss_res / max(ss_tot, 0.001))
            
            return AnalyticsSummary(mean_error, rmse, bias, r_squared, len(filtered))


if __name__ == "__main__":
    analytics = ImpactAnalytics()
    print("Impact Analytics initialized")
