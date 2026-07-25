#!/usr/bin/env python3
"""
Volatility Soul Logger - IPC Bottleneck and Event Logger for SOUL.md

Logs all volatility-related events, forecast errors, jump detections,
and arbitrage alerts to the central SOUL.md file. Provides real-time
visibility into the vol engine's health and performance.

Features:
- Microsecond-latency spike detection
- Volatility forecast error tracking
- Jump event logging with severity classification
- Arbitrage opportunity capture
- Trading halt event recording
- Performance metrics (throughput, latency percentiles)

Integration: Updates SOUL.md atomically to prevent corruption.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
import hashlib


class EventType(Enum):
    """Types of volatility events to log."""
    VOL_FORECAST = "vol_forecast"
    VOL_SPIKE = "vol_spike"
    JUMP_DETECTED = "jump_detected"
    REGIME_CHANGE = "regime_change"
    ARBITRAGE_ALERT = "arbitrage_alert"
    TRADING_HALT = "trading_halt"
    LATENCY_SPIKE = "latency_spike"
    MODEL_ERROR = "model_error"
    SURFACE_ANOMALY = "surface_anomaly"
    GARCH_UPDATE = "garch_update"
    FFT_ANALYSIS = "fft_analysis"


class SeverityLevel(Enum):
    """Event severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class VolEvent:
    """Single volatility event record."""
    event_type: EventType
    severity: SeverityLevel
    timestamp_us: int
    asset: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    latency_us: int = 0
    forecast_error: Optional[float] = None
    predicted_vol: Optional[float] = None
    actual_vol: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'severity': self.severity.name,
            'timestamp_us': self.timestamp_us,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp_us / 1e6).isoformat(),
            'asset': self.asset,
            'message': self.message,
            'details': self.details,
            'latency_us': self.latency_us,
            'forecast_error': self.forecast_error,
            'predicted_vol': self.predicted_vol,
            'actual_vol': self.actual_vol,
        }


@dataclass
class LatencyStats:
    """Latency statistics for performance monitoring."""
    count: int = 0
    total_us: int = 0
    min_us: int = 0
    max_us: int = 0
    p50_us: int = 0
    p95_us: int = 0
    p99_us: int = 0
    spikes_count: int = 0  # Events exceeding threshold
    
    def update(self, latencies: List[int], spike_threshold_us: int = 1000) -> None:
        if not latencies:
            return
        
        self.count = len(latencies)
        self.total_us = sum(latencies)
        self.min_us = min(latencies)
        self.max_us = max(latencies)
        
        sorted_lat = sorted(latencies)
        self.p50_us = sorted_lat[len(sorted_lat) // 2]
        self.p95_us = sorted_lat[int(len(sorted_lat) * 0.95)]
        self.p99_us = sorted_lat[int(len(sorted_lat) * 0.99)]
        
        self.spikes_count = sum(1 for l in latencies if l > spike_threshold_us)
    
    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count > 0 else 0.0


class VolatilitySoulLogger:
    """
    Central logger for volatility engine events.
    
    Writes to SOUL.md with atomic operations to prevent corruption.
    Tracks forecast accuracy, latency spikes, and critical events.
    """
    
    # Default paths
    DEFAULT_SOUL_PATH = Path("/workspace/SOUL.md")
    
    # Latency thresholds (microseconds)
    LATENCY_WARNING_US = 500
    LATENCY_CRITICAL_US = 1000
    
    # Forecast error thresholds
    FORECAST_ERROR_WARNING = 0.10  # 10%
    FORECAST_ERROR_CRITICAL = 0.25  # 25%
    
    def __init__(self, soul_path: Optional[Path] = None):
        """
        Initialize the volatility soul logger.
        
        Parameters
        ----------
        soul_path : Path - Path to SOUL.md file
        """
        self.soul_path = soul_path or self.DEFAULT_SOUL_PATH
        self._lock = threading.RLock()
        
        # Event buffers for batching
        self._event_buffer: List[VolEvent] = []
        self._latency_samples: Dict[str, List[int]] = {
            asset: [] for asset in ['BTC', 'SOL', 'ETH', 'USDT']
        }
        
        # Statistics tracking
        self.stats: Dict[str, LatencyStats] = {
            asset: LatencyStats() for asset in ['BTC', 'SOL', 'ETH', 'USDT']
        }
        self.forecast_errors: Dict[str, List[float]] = {
            asset: [] for asset in ['BTC', 'SOL', 'ETH', 'USDT']
        }
        
        # Counters
        self.event_counts: Dict[EventType, int] = {et: 0 for et in EventType}
        self.total_events = 0
        self.critical_events = 0
        
        # Ensure SOUL.md exists with proper header
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Create SOUL.md with header if it doesn't exist."""
        if not self.soul_path.exists():
            self.soul_path.parent.mkdir(parents=True, exist_ok=True)
            
            header = f"""# ZAID Personal Crypto Trading Bot - SOUL.md

## System Overview
**Bot Version**: Stage 18 - Advanced Volatility Modeling  
**Last Updated**: {datetime.now().isoformat()}  
**Target**: 8k-20k INR/hour within 4-hour window  
**Assets**: BTC, SOL, ETH, USDT  

---

## Volatility Engine Logs

### Event Summary
| Metric | Value |
|--------|-------|
| Total Events | 0 |
| Critical Events | 0 |
| Avg Latency (μs) | 0 |
| Forecast Accuracy | N/A |

### Recent Events

---

## Performance Metrics

### Latency Statistics (Microseconds)
| Asset | Count | Avg | P50 | P95 | P99 | Max | Spikes |
|-------|-------|-----|-----|-----|-----|-----|--------|
| BTC | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SOL | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ETH | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| USDT | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Forecast Accuracy
| Asset | MAE | RMSE | Max Error | Samples |
|-------|-----|------|-----------|---------|
| BTC | 0.00 | 0.00 | 0.00 | 0 |
| SOL | 0.00 | 0.00 | 0.00 | 0 |
| ETH | 0.00 | 0.00 | 0.00 | 0 |

---

## Critical Alerts

*No critical alerts yet.*

---

*Generated by Volatility Soul Logger - Stage 18*
"""
            self.soul_path.write_text(header, encoding='utf-8')
    
    def log_event(
        self,
        event_type: EventType,
        asset: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: SeverityLevel = SeverityLevel.INFO,
        latency_us: int = 0,
        forecast_error: Optional[float] = None,
    ) -> None:
        """
        Log a volatility event.
        
        Parameters
        ----------
        event_type : EventType - Type of event
        asset : str - Asset identifier
        message : str - Human-readable message
        details : Dict - Additional structured data
        severity : SeverityLevel - Event severity
        latency_us : int - Processing latency in microseconds
        forecast_error : float - Prediction error if applicable
        """
        current_time_us = int(time.time() * 1e6)
        
        event = VolEvent(
            event_type=event_type,
            severity=severity,
            timestamp_us=current_time_us,
            asset=asset,
            message=message,
            details=details or {},
            latency_us=latency_us,
            forecast_error=forecast_error,
        )
        
        with self._lock:
            # Update counters
            self.event_counts[event_type] += 1
            self.total_events += 1
            
            if severity == SeverityLevel.CRITICAL:
                self.critical_events += 1
            
            # Buffer event
            self._event_buffer.append(event)
            
            # Track latency
            if asset in self._latency_samples and latency_us > 0:
                self._latency_samples[asset].append(latency_us)
                # Keep last 1000 samples per asset
                if len(self._latency_samples[asset]) > 1000:
                    self._latency_samples[asset] = self._latency_samples[asset][-1000:]
            
            # Track forecast error
            if forecast_error is not None and asset in self.forecast_errors:
                self.forecast_errors[asset].append(abs(forecast_error))
                if len(self.forecast_errors[asset]) > 10000:
                    self.forecast_errors[asset] = self.forecast_errors[asset][-10000:]
            
            # Flush buffer periodically or on critical events
            if severity >= SeverityLevel.ERROR or len(self._event_buffer) >= 10:
                self._flush_to_soul()
    
    def log_vol_forecast(
        self,
        asset: str,
        predicted_vol: float,
        actual_vol: Optional[float] = None,
        model: str = "GARCH",
        horizon_hours: int = 1,
        latency_us: int = 0,
    ) -> None:
        """Log a volatility forecast event."""
        forecast_error = None
        if actual_vol is not None:
            forecast_error = (predicted_vol - actual_vol) / actual_vol
        
        severity = SeverityLevel.INFO
        if forecast_error is not None:
            if abs(forecast_error) > self.FORECAST_ERROR_CRITICAL:
                severity = SeverityLevel.CRITICAL
            elif abs(forecast_error) > self.FORECAST_ERROR_WARNING:
                severity = SeverityLevel.WARNING
        
        self.log_event(
            event_type=EventType.VOL_FORECAST,
            asset=asset,
            message=f"{model} forecast: {predicted_vol:.2%} vs actual {actual_vol:.2%}" if actual_vol else f"{model} forecast: {predicted_vol:.2%}",
            details={
                'predicted_vol': predicted_vol,
                'actual_vol': actual_vol,
                'model': model,
                'horizon_hours': horizon_hours,
            },
            severity=severity,
            latency_us=latency_us,
            forecast_error=forecast_error,
        )
    
    def log_vol_spike(
        self,
        asset: str,
        current_vol: float,
        historical_avg: float,
        z_score: float,
        latency_us: int = 0,
    ) -> None:
        """Log a sudden volatility spike."""
        severity = SeverityLevel.WARNING if z_score > 2.0 else SeverityLevel.INFO
        if z_score > 3.0:
            severity = SeverityLevel.CRITICAL
        
        self.log_event(
            event_type=EventType.VOL_SPIKE,
            asset=asset,
            message=f"VOL SPIKE: {current_vol:.2%} (z-score: {z_score:.2f})",
            details={
                'current_vol': current_vol,
                'historical_avg': historical_avg,
                'z_score': z_score,
                'spike_magnitude': (current_vol - historical_avg) / historical_avg,
            },
            severity=severity,
            latency_us=latency_us,
        )
    
    def log_jump_detected(
        self,
        asset: str,
        jump_size: float,
        jump_intensity: float,
        direction: str,
        latency_us: int = 0,
    ) -> None:
        """Log a detected price jump."""
        severity = SeverityLevel.WARNING
        if abs(jump_size) > 0.05:  # >5% jump
            severity = SeverityLevel.CRITICAL
        
        self.log_event(
            event_type=EventType.JUMP_DETECTED,
            asset=asset,
            message=f"JUMP: {direction} {jump_size:.2%} (intensity: {jump_intensity:.4f})",
            details={
                'jump_size': jump_size,
                'jump_intensity': jump_intensity,
                'direction': direction,
            },
            severity=severity,
            latency_us=latency_us,
        )
    
    def log_arbitrage_alert(
        self,
        asset: str,
        arb_type: str,
        expected_profit: float,
        confidence: float,
        latency_us: int = 0,
    ) -> None:
        """Log an arbitrage opportunity detection."""
        severity = SeverityLevel.INFO
        if expected_profit > 0.02 and confidence > 0.8:
            severity = SeverityLevel.WARNING
        if expected_profit > 0.05:
            severity = SeverityLevel.CRITICAL
        
        self.log_event(
            event_type=EventType.ARBITRAGE_ALERT,
            asset=asset,
            message=f"ARB OPPORTUNITY: {arb_type} ({expected_profit:.2%} expected)",
            details={
                'arb_type': arb_type,
                'expected_profit': expected_profit,
                'confidence': confidence,
            },
            severity=severity,
            latency_us=latency_us,
        )
    
    def log_trading_halt(
        self,
        asset: str,
        reason: str,
        expected_duration_sec: int = 0,
        latency_us: int = 0,
    ) -> None:
        """Log a trading halt event."""
        self.log_event(
            event_type=EventType.TRADING_HALT,
            asset=asset,
            message=f"TRADING HALT: {reason}",
            details={
                'reason': reason,
                'expected_duration_sec': expected_duration_sec,
            },
            severity=SeverityLevel.CRITICAL,
            latency_us=latency_us,
        )
    
    def log_latency_spike(
        self,
        asset: str,
        operation: str,
        latency_us: int,
        threshold_us: int,
    ) -> None:
        """Log a latency spike that exceeded thresholds."""
        self.log_event(
            event_type=EventType.LATENCY_SPIKE,
            asset=asset,
            message=f"LATENCY SPIKE: {operation} took {latency_us}μs (threshold: {threshold_us}μs)",
            details={
                'operation': operation,
                'latency_us': latency_us,
                'threshold_us': threshold_us,
                'excess_ratio': latency_us / threshold_us,
            },
            severity=SeverityLevel.WARNING if latency_us < self.LATENCY_CRITICAL_US else SeverityLevel.ERROR,
            latency_us=latency_us,
        )
    
    def _flush_to_soul(self) -> None:
        """Flush buffered events to SOUL.md atomically."""
        if not self._event_buffer:
            return
        
        try:
            # Read current content
            content = self.soul_path.read_text(encoding='utf-8') if self.soul_path.exists() else ""
            
            # Find the "Recent Events" section
            events_header = "### Recent Events\n"
            events_idx = content.find(events_header)
            
            if events_idx == -1:
                # Section doesn't exist, append at end
                new_content = content + "\n" + self._format_events_section()
            else:
                # Find next section header
                next_section = content.find("\n---\n", events_idx + len(events_header))
                if next_section == -1:
                    next_section = len(content)
                
                # Replace events section
                before = content[:events_idx + len(events_header)]
                after = content[next_section:]
                new_content = before + "\n" + self._format_events_section() + after
            
            # Atomic write using temp file
            temp_path = self.soul_path.with_suffix('.tmp')
            temp_path.write_text(new_content, encoding='utf-8')
            temp_path.replace(self.soul_path)
            
            # Clear buffer
            self._event_buffer.clear()
            
        except Exception as e:
            print(f"Error writing to SOUL.md: {e}")
    
    def _format_events_section(self) -> str:
        """Format recent events as markdown table."""
        if not self._event_buffer:
            return "*No recent events.*\n"
        
        lines = [
            "| Timestamp | Asset | Type | Severity | Message |",
            "|-----------|-------|------|----------|---------|"
        ]
        
        # Show last 20 events
        for event in self._event_buffer[-20:]:
            ts = datetime.fromtimestamp(event.timestamp_us / 1e6).strftime('%H:%M:%S.%f')[:-3]
            lines.append(
                f"| {ts} | {event.asset} | {event.event_type.value} | "
                f"{event.severity.name} | {event.message[:50]}..."
            )
        
        return "\n".join(lines) + "\n"
    
    def update_performance_metrics(self) -> None:
        """Update latency statistics in SOUL.md."""
        with self._lock:
            # Calculate stats for each asset
            for asset in self.stats.keys():
                if asset in self._latency_samples and self._latency_samples[asset]:
                    self.stats[asset].update(
                        self._latency_samples[asset],
                        spike_threshold_us=self.LATENCY_WARNING_US
                    )
            
            # Write updated stats
            self._update_stats_section()
    
    def _update_stats_section(self) -> None:
        """Update the performance metrics section in SOUL.md."""
        try:
            content = self.soul_path.read_text(encoding='utf-8')
            
            # Build latency table
            latency_lines = [
                "| Asset | Count | Avg | P50 | P95 | P99 | Max | Spikes |",
                "|-------|-------|-----|-----|-----|-----|-----|--------|"
            ]
            
            for asset, stats in self.stats.items():
                if stats.count > 0:
                    latency_lines.append(
                        f"| {asset} | {stats.count} | {stats.avg_us:.0f} | "
                        f"{stats.p50_us} | {stats.p95_us} | {stats.p99_us} | "
                        f"{stats.max_us} | {stats.spikes_count} |"
                    )
            
            latency_table = "\n".join(latency_lines)
            
            # Find and replace latency section
            start_marker = "### Latency Statistics (Microseconds)"
            end_marker = "### Forecast Accuracy"
            
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                before = content[:start_idx + len(start_marker) + 1]
                after = content[end_idx:]
                
                # Rebuild section
                new_section = f"\n{latency_table}\n"
                content = before + new_section + after
                
                # Atomic write
                temp_path = self.soul_path.with_suffix('.tmp')
                temp_path.write_text(content, encoding='utf-8')
                temp_path.replace(self.soul_path)
                
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current logger summary."""
        with self._lock:
            return {
                'total_events': self.total_events,
                'critical_events': self.critical_events,
                'event_counts': {et.value: c for et, c in self.event_counts.items()},
                'latency_stats': {
                    asset: {
                        'count': s.count,
                        'avg_us': s.avg_us,
                        'p99_us': s.p99_us,
                        'spikes': s.spikes_count,
                    }
                    for asset, s in self.stats.items()
                },
                'forecast_accuracy': {
                    asset: {
                        'mae': sum(errors) / len(errors) if errors else 0,
                        'samples': len(errors),
                    }
                    for asset, errors in self.forecast_errors.items()
                },
            }


def demo_vol_logger():
    """Demonstrate the volatility soul logger."""
    print("=" * 60)
    print("Volatility Soul Logger Demo")
    print("=" * 60)
    
    logger = VolatilitySoulLogger()
    
    # Simulate various events
    logger.log_vol_forecast('BTC', 0.72, 0.68, model="GARCH", latency_us=150)
    logger.log_vol_spike('ETH', 0.95, 0.65, z_score=3.2, latency_us=200)
    logger.log_jump_detected('SOL', 0.08, 0.002, 'UP', latency_us=50)
    logger.log_arbitrage_alert('BTC', 'calendar_spread', 0.03, 0.85, latency_us=300)
    logger.log_latency_spike('ETH', 'FFT_computation', 1200, 1000)
    
    # Simulate forecast with error
    logger.log_vol_forecast('BTC', 0.80, 0.60, model="EGARCH", latency_us=180)
    
    # Update performance metrics
    logger.update_performance_metrics()
    
    # Get summary
    summary = logger.get_summary()
    print(f"\nTotal Events: {summary['total_events']}")
    print(f"Critical Events: {summary['critical_events']}")
    print(f"\nLatency Stats (BTC): {summary['latency_stats']['BTC']}")
    print(f"Forecast Accuracy (BTC): {summary['forecast_accuracy']['BTC']}")
    
    print(f"\nSOUL.md updated at: {logger.soul_path}")
    print("Demo complete.")


if __name__ == "__main__":
    demo_vol_logger()
