#!/usr/bin/env python3
"""
Time Series Soul Logger - Model Drift and Forecast Error Logging to SOUL.md

This module logs forecasting model performance, drift metrics, and errors
to the central SOUL.md file for persistence and analysis.

Key Features:
- Automatic logging of SARIMAX seasonal prediction failures
- Model drift detection using rolling error statistics
- Diebold-Mariano test result logging
- Memory-efficient circular buffer for recent errors

Updates SOUL.md when models fail to predict seasonal volatility spikes.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np
import os


@dataclass
class ForecastError:
    """Record of a single forecast error."""
    timestamp: datetime
    model_id: str
    asset: str
    actual: float
    predicted: float
    error: float
    squared_error: float
    horizon: int  # Forecast horizon in periods
    
    @property
    def mape(self) -> float:
        """Mean Absolute Percentage Error (single observation)."""
        if self.actual != 0:
            return abs(self.error / self.actual) * 100
        return float('inf')


@dataclass
class ModelDriftMetrics:
    """Drift metrics for a forecasting model."""
    model_id: str
    asset: str
    window_size: int
    mean_error: float
    rmse: float
    mape: float
    error_std: float
    bias: float  # Mean signed error
    max_error: float
    drift_detected: bool
    drift_severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class SeasonalFailureEvent:
    """Event logged when SARIMAX fails to predict seasonal volatility."""
    timestamp: datetime
    asset: str
    expected_volatility: float
    predicted_volatility: float
    actual_volatility: float
    error_magnitude: float
    season_type: str  # 'intraday', 'daily', 'weekly'
    severity: int  # 1-5 scale


class TimeSeriesSoulLogger:
    """
    Logger for time series forecasting analytics.
    
    Writes model performance, drift alerts, and seasonal failures
    to SOUL.md for persistent tracking.
    """
    
    def __init__(self, 
                 soul_path: str = "SOUL.md",
                 max_errors_buffer: int = 1000,
                 drift_window: int = 100,
                 drift_threshold: float = 0.1):
        """
        Initialize logger.
        
        Args:
            soul_path: Path to SOUL.md file
            max_errors_buffer: Maximum errors to keep in memory
            drift_window: Window size for drift calculation
            drift_threshold: Threshold for drift detection (normalized RMSE increase)
        """
        self.soul_path = soul_path
        self.max_errors_buffer = max_errors_buffer
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        
        # Per-model error tracking
        self._model_errors: Dict[str, deque] = {}
        self._baseline_rmse: Dict[str, float] = {}
        
        # Event logs
        self._seasonal_failures: List[SeasonalFailureEvent] = []
        self._drift_alerts: List[Tuple[datetime, str, str]] = []
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self):
        """Initialize SOUL.md with header if it doesn't exist."""
        if not os.path.exists(self.soul_path):
            header = """# ZAID PERSONAL CRYPTO TRADING BOT - SOUL.md

## Time Series Forecasting Analytics Log

This file contains persistent logs of:
- Forecasting model performance metrics
- Model drift detection alerts
- Seasonal volatility prediction failures
- Diebold-Mariano test results

---

"""
            with open(self.soul_path, 'w') as f:
                f.write(header)
    
    def log_forecast(self, 
                     model_id: str,
                     asset: str,
                     actual: float,
                     predicted: float,
                     horizon: int = 1,
                     timestamp: Optional[datetime] = None):
        """
        Log a single forecast observation.
        
        Args:
            model_id: Unique identifier for the model
            asset: Asset symbol (BTC, ETH, SOL, etc.)
            actual: Actual observed value
            predicted: Model's predicted value
            horizon: Forecast horizon
            timestamp: Observation timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        error = actual - predicted
        
        # Create error record
        err_record = ForecastError(
            timestamp=timestamp,
            model_id=model_id,
            asset=asset,
            actual=actual,
            predicted=predicted,
            error=error,
            squared_error=error ** 2,
            horizon=horizon
        )
        
        # Store in per-model buffer
        key = f"{model_id}:{asset}"
        if key not in self._model_errors:
            self._model_errors[key] = deque(maxlen=self.max_errors_buffer)
            self._baseline_rmse[key] = None
        
        self._model_errors[key].append(err_record)
        
        # Update baseline RMSE on first observations
        if len(self._model_errors[key]) == 50:
            errors = list(self._model_errors[key])
            rmse = np.sqrt(np.mean([e.squared_error for e in errors]))
            self._baseline_rmse[key] = rmse
    
    def log_seasonal_failure(self,
                              asset: str,
                              expected_vol: float,
                              predicted_vol: float,
                              actual_vol: float,
                              season_type: str = 'daily',
                              timestamp: Optional[datetime] = None):
        """
        Log failure to predict seasonal volatility spike.
        
        Args:
            asset: Asset symbol
            expected_vol: Expected volatility level
            predicted_vol: Model's predicted volatility
            actual_vol: Actual realized volatility
            season_type: Type of seasonality
            timestamp: Event timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        error_mag = abs(actual_vol - predicted_vol) / max(expected_vol, 1e-8)
        severity = min(5, max(1, int(error_mag * 10)))
        
        event = SeasonalFailureEvent(
            timestamp=timestamp,
            asset=asset,
            expected_volatility=expected_vol,
            predicted_volatility=predicted_vol,
            actual_volatility=actual_vol,
            error_magnitude=error_mag,
            season_type=season_type,
            severity=severity
        )
        
        self._seasonal_failures.append(event)
        
        # Write to SOUL.md immediately for critical failures
        if severity >= 4:
            self._write_seasonal_failure_to_soul(event)
    
    def check_drift(self, model_id: str, asset: str) -> Optional[ModelDriftMetrics]:
        """
        Check for model drift and return metrics.
        
        Args:
            model_id: Model identifier
            asset: Asset symbol
            
        Returns:
            ModelDriftMetrics if enough data, None otherwise
        """
        key = f"{model_id}:{asset}"
        errors = self._model_errors.get(key, deque())
        
        if len(errors) < self.drift_window:
            return None
        
        recent = list(errors)[-self.drift_window:]
        
        # Compute metrics
        error_values = [e.error for e in recent]
        squared_errors = [e.squared_error for e in recent]
        
        mean_error = np.mean(error_values)
        rmse = np.sqrt(np.mean(squared_errors))
        error_std = np.std(error_values)
        bias = mean_error  # Mean signed error
        max_error = max(abs(e) for e in error_values)
        
        # MAPE (excluding zero actuals)
        non_zero = [(e.actual, e.error) for e in recent if e.actual != 0]
        if non_zero:
            mape = np.mean([abs(err / act) for act, err in non_zero]) * 100
        else:
            mape = float('inf')
        
        # Detect drift
        baseline = self._baseline_rmse.get(key)
        drift_detected = False
        drift_severity = 'none'
        
        if baseline is not None and baseline > 0:
            rmse_increase = (rmse - baseline) / baseline
            if rmse_increase > self.drift_threshold * 3:
                drift_detected = True
                drift_severity = 'critical'
            elif rmse_increase > self.drift_threshold * 2:
                drift_detected = True
                drift_severity = 'high'
            elif rmse_increase > self.drift_threshold:
                drift_detected = True
                drift_severity = 'medium'
            elif rmse_increase > self.drift_threshold / 2:
                drift_severity = 'low'
        
        metrics = ModelDriftMetrics(
            model_id=model_id,
            asset=asset,
            window_size=len(recent),
            mean_error=mean_error,
            rmse=rmse,
            mape=mape,
            error_std=error_std,
            bias=bias,
            max_error=max_error,
            drift_detected=drift_detected,
            drift_severity=drift_severity
        )
        
        # Log drift alert if detected
        if drift_detected:
            self._log_drift_alert(metrics)
        
        return metrics
    
    def _write_seasonal_failure_to_soul(self, event: SeasonalFailureEvent):
        """Write seasonal failure event to SOUL.md."""
        entry = f"""
### 🚨 Seasonal Volatility Prediction Failure

**Timestamp:** {event.timestamp.isoformat()}
**Asset:** {event.asset}
**Season Type:** {event.season_type}
**Severity:** {'🔴 Critical' if event.severity >= 4 else '🟠 High' if event.severity >= 3 else '🟡 Medium'}

| Metric | Value |
|--------|-------|
| Expected Volatility | {event.expected_volatility:.4f} |
| Predicted Volatility | {event.predicted_volatility:.4f} |
| Actual Volatility | {event.actual_volatility:.4f} |
| Error Magnitude | {event.error_magnitude:.2%} |

**Action Required:** Review SARIMAX seasonal parameters and consider recalibration.

---

"""
        with open(self.soul_path, 'a') as f:
            f.write(entry)
    
    def _log_drift_alert(self, metrics: ModelDriftMetrics):
        """Log drift alert to SOUL.md."""
        timestamp = datetime.now()
        
        severity_icon = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }.get(metrics.drift_severity, '⚪')
        
        entry = f"""
### {severity_icon} Model Drift Detected

**Timestamp:** {timestamp.isoformat()}
**Model:** {metrics.model_id}
**Asset:** {metrics.asset}
**Severity:** {metrics.drift_severity.upper()}

| Metric | Value |
|--------|-------|
| RMSE | {metrics.rmse:.6f} |
| MAPE | {metrics.mape:.2f}% |
| Bias | {metrics.bias:.6f} |
| Max Error | {metrics.max_error:.6f} |
| Window Size | {metrics.window_size} |

**Recommendation:** {'Immediate recalibration required' if metrics.drift_severity in ['critical', 'high'] else 'Monitor closely'}

---

"""
        with open(self.soul_path, 'a') as f:
            f.write(entry)
        
        self._drift_alerts.append((timestamp, metrics.model_id, metrics.drift_severity))
    
    def log_dm_test_result(self,
                           model_1: str,
                           model_2: str,
                           dm_statistic: float,
                           p_value: float,
                           winner: str,
                           timestamp: Optional[datetime] = None):
        """
        Log Diebold-Mariano test result.
        
        Args:
            model_1: First model identifier
            model_2: Second model identifier
            dm_statistic: DM test statistic
            p_value: Test p-value
            winner: Which model won ('model_1', 'model_2', or 'tie')
            timestamp: Test timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*"
        
        entry = f"""
### 📊 Diebold-Mariano Test Result {significance}

**Timestamp:** {timestamp.isoformat()}

**Comparison:** {model_1} vs {model_2}

| Statistic | Value |
|-----------|-------|
| DM Statistic | {dm_statistic:.4f} |
| P-value | {p_value:.4f} |
| Winner | {winner} |
| Significance | {'Yes' if p_value < 0.05 else 'No'} |

---

"""
        with open(self.soul_path, 'a') as f:
            f.write(entry)
    
    def get_recent_errors(self, 
                          model_id: str, 
                          asset: str,
                          n: int = 100) -> List[ForecastError]:
        """Get recent errors for a model/asset pair."""
        key = f"{model_id}:{asset}"
        errors = self._model_errors.get(key, deque())
        return list(errors)[-n:]
    
    def summary(self) -> str:
        """Generate summary of all logged activity."""
        lines = ["Time Series Soul Logger Summary", "=" * 50]
        
        for key, errors in self._model_errors.items():
            if len(errors) >= 10:
                recent = list(errors)[-100:]
                rmse = np.sqrt(np.mean([e.squared_error for e in recent]))
                mape = np.mean([abs(e.error/e.actual)*100 for e in recent if e.actual != 0])
                lines.append(f"{key}: RMSE={rmse:.6f}, MAPE={mape:.2f}%")
        
        if self._seasonal_failures:
            lines.append(f"\nSeasonal Failures: {len(self._seasonal_failures)}")
            critical = sum(1 for f in self._seasonal_failures if f.severity >= 4)
            lines.append(f"  Critical: {critical}")
        
        if self._drift_alerts:
            lines.append(f"\nDrift Alerts: {len(self._drift_alerts)}")
        
        return "\n".join(lines)


def create_ts_logger(soul_path: str = "SOUL.md") -> TimeSeriesSoulLogger:
    """Convenience function to create configured logger."""
    return TimeSeriesSoulLogger(
        soul_path=soul_path,
        max_errors_buffer=1000,
        drift_window=100,
        drift_threshold=0.1
    )


if __name__ == "__main__":
    # Example usage
    import time
    
    logger = create_ts_logger()
    
    # Simulate forecast logging
    np.random.seed(42)
    
    for i in range(150):
        t = datetime.now()
        actual = 100 + np.random.randn() * 2
        predicted = actual + np.random.randn() * 1.5
        
        logger.log_forecast("SARIMAX_BTC", "BTC", actual, predicted, horizon=1, timestamp=t)
        
        # Simulate a seasonal failure
        if i == 100:
            logger.log_seasonal_failure(
                asset="BTC",
                expected_vol=0.02,
                predicted_vol=0.015,
                actual_vol=0.05,
                season_type="intraday"
            )
    
    # Check drift
    metrics = logger.check_drift("SARIMAX_BTC", "BTC")
    if metrics:
        print(f"Drift detected: {metrics.drift_detected}")
        print(f"Severity: {metrics.drift_severity}")
    
    # Log DM test
    logger.log_dm_test_result(
        model_1="SARIMAX_BTC",
        model_2="VAR_BTC",
        dm_statistic=2.5,
        p_value=0.012,
        winner="SARIMAX_BTC"
    )
    
    print("\n" + logger.summary())
