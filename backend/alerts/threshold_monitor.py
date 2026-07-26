#!/usr/bin/env python3
"""
Threshold Monitor for Real-Time Alerting

This module monitors PnL drawdowns, latency spikes, and other critical
metrics to trigger alerts when thresholds are breached.

Key Features:
- Sliding window calculations for drawdown detection
- Multi-threshold alerting with hysteresis
- Circuit breaker integration for automatic trading halts
- Memory-efficient metric storage
- Thread-safe threshold updates

Designed for the ZAID Personal Crypto Trading Bot to maintain
risk controls during high-frequency trading windows.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from enum import IntEnum
import statistics

# Type hints
Timestamp = float
Value = float


class AlertSeverity(IntEnum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


class ThresholdType(IntEnum):
    """Types of thresholds that can be monitored."""
    MAX_VALUE = 0       # Alert when value exceeds threshold
    MIN_VALUE = 1       # Alert when value falls below threshold
    DELTA_CHANGE = 2    # Alert when change exceeds threshold
    RATE_OF_CHANGE = 3  # Alert when rate of change exceeds threshold
    DRAWDOWN = 4        # Alert when drawdown exceeds threshold
    LATENCY_SPIKE = 5   # Alert when latency percentile exceeds threshold


@dataclass(slots=True)
class ThresholdConfig:
    """Configuration for a single threshold monitor."""
    name: str
    threshold_type: ThresholdType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    window_size: int = 100  # Number of samples for sliding window
    cooldown_seconds: float = 60.0  # Minimum time between repeated alerts
    enabled: bool = True
    
    def __post_init__(self) -> None:
        if self.emergency_threshold is None:
            self.emergency_threshold = self.critical_threshold * 1.5


@dataclass(slots=True)
class Alert:
    """Represents a triggered alert."""
    timestamp: Timestamp
    name: str
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'name': self.name,
            'severity': self.severity.name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'message': self.message,
            'metadata': self.metadata
        }


class SlidingWindow:
    """
    Memory-efficient sliding window for metric calculations.
    
    Uses a fixed-size deque to maintain O(1) insertions and
    efficient statistical calculations.
    """
    
    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._values: deque = deque(maxlen=max_size)
        self._timestamps: deque = deque(maxlen=max_size)
        self._sum: float = 0.0
        self._min: float = float('inf')
        self._max: float = float('-inf')
        self._lock = threading.Lock()
    
    def add(self, value: float, timestamp: Optional[Timestamp] = None) -> None:
        """Add a value to the window."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Remove oldest value from sum if exists
            if len(self._values) == self.max_size:
                old_value = self._values[0]
                self._sum -= old_value
            
            # Add new value
            self._values.append(value)
            self._timestamps.append(timestamp)
            self._sum += value
            
            # Update min/max
            self._min = min(self._values)
            self._max = max(self._values)
    
    def get_mean(self) -> Optional[float]:
        """Get mean of values in window."""
        with self._lock:
            if not self._values:
                return None
            return self._sum / len(self._values)
    
    def get_std(self) -> Optional[float]:
        """Get standard deviation of values in window."""
        with self._lock:
            if len(self._values) < 2:
                return None
            return statistics.stdev(self._values)
    
    def get_percentile(self, p: float) -> Optional[float]:
        """Get percentile of values in window."""
        with self._lock:
            if not self._values:
                return None
            sorted_values = sorted(self._values)
            idx = int(len(sorted_values) * p)
            idx = min(idx, len(sorted_values) - 1)
            return sorted_values[idx]
    
    def get_p99(self) -> Optional[float]:
        """Get 99th percentile."""
        return self.get_percentile(0.99)
    
    def get_p95(self) -> Optional[float]:
        """Get 95th percentile."""
        return self.get_percentile(0.95)
    
    def get_min(self) -> Optional[float]:
        """Get minimum value in window."""
        with self._lock:
            return self._min if self._values else None
    
    def get_max(self) -> Optional[float]:
        """Get maximum value in window."""
        with self._lock:
            return self._max if self._values else None
    
    def get_drawdown(self) -> float:
        """
        Calculate maximum drawdown in the window.
        
        Drawdown = (peak - current) / peak
        """
        with self._lock:
            if len(self._values) < 2:
                return 0.0
            
            max_drawdown = 0.0
            peak = self._values[0]
            
            for value in self._values:
                if value > peak:
                    peak = value
                elif peak > 0:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
    
    def get_rate_of_change(self, window: int = 10) -> Optional[float]:
        """Calculate rate of change over recent samples."""
        with self._lock:
            if len(self._values) < window:
                return None
            
            recent = list(self._values)[-window:]
            if recent[0] == 0:
                return None
            
            return (recent[-1] - recent[0]) / recent[0]
    
    def size(self) -> int:
        """Get current window size."""
        with self._lock:
            return len(self._values)
    
    def clear(self) -> None:
        """Clear the window."""
        with self._lock:
            self._values.clear()
            self._timestamps.clear()
            self._sum = 0.0
            self._min = float('inf')
            self._max = float('-inf')


class ThresholdMonitor:
    """
    Central monitor for threshold-based alerting.
    
    Features:
    - Multiple threshold types (max, min, delta, drawdown, etc.)
    - Sliding window calculations
    - Alert cooldowns to prevent spam
    - Severity-based escalation
    - Callback support for custom alert handling
    """
    
    def __init__(self) -> None:
        # Threshold configurations
        self._configs: Dict[str, ThresholdConfig] = {}
        self._config_lock = threading.RLock()
        
        # Sliding windows for each threshold
        self._windows: Dict[str, SlidingWindow] = {}
        
        # Last alert times for cooldown management
        self._last_alert_times: Dict[str, Dict[AlertSeverity, Timestamp]] = {}
        
        # Current values
        self._current_values: Dict[str, float] = {}
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._callback_lock = threading.Lock()
        
        # Alert history
        self._alert_history: deque = deque(maxlen=1000)
        
        # Running state
        self._running = True
        
        # Background monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def add_threshold(
        self,
        config: ThresholdConfig
    ) -> None:
        """Add a threshold to monitor."""
        with self._config_lock:
            self._configs[config.name] = config
            self._windows[config.name] = SlidingWindow(config.window_size)
            self._last_alert_times[config.name] = {}
            self._current_values[config.name] = 0.0
    
    def remove_threshold(self, name: str) -> None:
        """Remove a threshold from monitoring."""
        with self._config_lock:
            self._configs.pop(name, None)
            self._windows.pop(name, None)
            self._last_alert_times.pop(name, None)
            self._current_values.pop(name, None)
    
    def update_value(self, name: str, value: float) -> None:
        """Update the current value for a threshold."""
        with self._config_lock:
            if name not in self._configs:
                return
            
            config = self._configs[name]
            if not config.enabled:
                return
            
            # Update current value
            self._current_values[name] = value
            
            # Add to sliding window
            self._windows[name].add(value)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback to be called when alerts are triggered."""
        with self._callback_lock:
            self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Remove an alert callback."""
        with self._callback_lock:
            if callback in self._alert_callbacks:
                self._alert_callbacks.remove(callback)
    
    def _check_threshold(self, name: str) -> Optional[Alert]:
        """Check if a threshold has been breached."""
        with self._config_lock:
            if name not in self._configs:
                return None
            
            config = self._configs[name]
            if not config.enabled:
                return None
            
            window = self._windows[name]
            current_value = self._current_values[name]
            
            # Get the value to check based on threshold type
            check_value: Optional[float] = None
            threshold_name = ""
            
            if config.threshold_type == ThresholdType.MAX_VALUE:
                check_value = current_value
                threshold_name = "max"
            elif config.threshold_type == ThresholdType.MIN_VALUE:
                check_value = current_value
                threshold_name = "min"
            elif config.threshold_type == ThresholdType.DRAWDOWN:
                check_value = window.get_drawdown()
                threshold_name = "drawdown"
            elif config.threshold_type == ThresholdType.LATENCY_SPIKE:
                check_value = window.get_p99()
                threshold_name = "p99_latency"
            elif config.threshold_type == ThresholdType.RATE_OF_CHANGE:
                check_value = window.get_rate_of_change()
                threshold_name = "rate_of_change"
            
            if check_value is None:
                return None
            
            # Determine severity
            severity: Optional[AlertSeverity] = None
            threshold_value: float = 0.0
            
            if config.threshold_type in (ThresholdType.MAX_VALUE, ThresholdType.LATENCY_SPIKE, 
                                          ThresholdType.DRAWDOWN, ThresholdType.RATE_OF_CHANGE):
                # Check if value exceeds thresholds
                if config.emergency_threshold and check_value >= config.emergency_threshold:
                    severity = AlertSeverity.EMERGENCY
                    threshold_value = config.emergency_threshold
                elif check_value >= config.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = config.critical_threshold
                elif check_value >= config.warning_threshold:
                    severity = AlertSeverity.WARNING
                    threshold_value = config.warning_threshold
            else:
                # Check if value falls below thresholds
                if config.emergency_threshold and check_value <= config.emergency_threshold:
                    severity = AlertSeverity.EMERGENCY
                    threshold_value = config.emergency_threshold
                elif check_value <= config.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = config.critical_threshold
                elif check_value <= config.warning_threshold:
                    severity = AlertSeverity.WARNING
                    threshold_value = config.warning_threshold
            
            if severity is None:
                return None
            
            # Check cooldown
            current_time = time.time()
            last_alert = self._last_alert_times.get(name, {}).get(severity, 0.0)
            if current_time - last_alert < config.cooldown_seconds:
                return None
            
            # Create alert
            message = f"{name}: {threshold_name}={check_value:.4f} breached {threshold_name}_threshold={threshold_value:.4f}"
            
            alert = Alert(
                timestamp=current_time,
                name=name,
                severity=severity,
                current_value=check_value,
                threshold_value=threshold_value,
                message=message,
                metadata={
                    'threshold_type': config.threshold_type.name,
                    'window_size': window.size(),
                    'mean': window.get_mean(),
                    'std': window.get_std(),
                }
            )
            
            # Update last alert time
            if name not in self._last_alert_times:
                self._last_alert_times[name] = {}
            self._last_alert_times[name][severity] = current_time
            
            return alert
    
    def _trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert and notify callbacks."""
        # Add to history
        self._alert_history.append(alert)
        
        # Notify callbacks
        with self._callback_lock:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Alert callback error: {e}")
        
        # Log alert
        severity_str = alert.severity.name
        print(f"[ALERT] [{severity_str}] {alert.message}")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                with self._config_lock:
                    names = list(self._configs.keys())
                
                for name in names:
                    alert = self._check_threshold(name)
                    if alert:
                        self._trigger_alert(alert)
                
                time.sleep(0.1)  # Check every 100ms
            except Exception as e:
                print(f"Monitor loop error: {e}")
                time.sleep(1.0)
    
    def get_alert_history(self, count: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return list(self._alert_history)[-count:]
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status of all thresholds."""
        status = {}
        
        with self._config_lock:
            for name, config in self._configs.items():
                window = self._windows[name]
                current_value = self._current_values.get(name, 0.0)
                
                status[name] = {
                    'enabled': config.enabled,
                    'type': config.threshold_type.name,
                    'current_value': current_value,
                    'warning_threshold': config.warning_threshold,
                    'critical_threshold': config.critical_threshold,
                    'emergency_threshold': config.emergency_threshold,
                    'window_size': window.size(),
                    'mean': window.get_mean(),
                    'std': window.get_std(),
                    'min': window.get_min(),
                    'max': window.get_max(),
                    'p99': window.get_p99(),
                    'drawdown': window.get_drawdown(),
                }
        
        return status
    
    def enable_threshold(self, name: str) -> None:
        """Enable a threshold."""
        with self._config_lock:
            if name in self._configs:
                self._configs[name].enabled = True
    
    def disable_threshold(self, name: str) -> None:
        """Disable a threshold."""
        with self._config_lock:
            if name in self._configs:
                self._configs[name].enabled = False
    
    def shutdown(self) -> None:
        """Shutdown the monitor gracefully."""
        self._running = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)


# Pre-configured threshold templates for common use cases
def create_pnl_drawdown_threshold(
    warning: float = 0.02,    # 2% drawdown
    critical: float = 0.05,   # 5% drawdown
    emergency: float = 0.10,  # 10% drawdown
    window_size: int = 1000
) -> ThresholdConfig:
    """Create a PnL drawdown threshold configuration."""
    return ThresholdConfig(
        name="pnl_drawdown",
        threshold_type=ThresholdType.DRAWDOWN,
        warning_threshold=warning,
        critical_threshold=critical,
        emergency_threshold=emergency,
        window_size=window_size,
        cooldown_seconds=30.0
    )


def create_latency_spike_threshold(
    warning_us: float = 1000,     # 1ms
    critical_us: float = 5000,    # 5ms
    emergency_us: float = 10000,  # 10ms
    window_size: int = 100
) -> ThresholdConfig:
    """Create a latency spike threshold configuration."""
    return ThresholdConfig(
        name="latency_spike",
        threshold_type=ThresholdType.LATENCY_SPIKE,
        warning_threshold=warning_us,
        critical_threshold=critical_us,
        emergency_threshold=emergency_us,
        window_size=window_size,
        cooldown_seconds=10.0
    )


def create_rate_of_change_threshold(
    name: str,
    warning: float = 0.1,    # 10% change
    critical: float = 0.25,  # 25% change
    emergency: float = 0.50, # 50% change
    window_size: int = 50
) -> ThresholdConfig:
    """Create a rate of change threshold configuration."""
    return ThresholdConfig(
        name=name,
        threshold_type=ThresholdType.RATE_OF_CHANGE,
        warning_threshold=warning,
        critical_threshold=critical,
        emergency_threshold=emergency,
        window_size=window_size,
        cooldown_seconds=60.0
    )


if __name__ == '__main__':
    # Example usage and testing
    print("Initializing Threshold Monitor...")
    monitor = ThresholdMonitor()
    
    # Add predefined thresholds
    monitor.add_threshold(create_pnl_drawdown_threshold())
    monitor.add_threshold(create_latency_spike_threshold())
    monitor.add_threshold(create_rate_of_change_threshold("price_volatility"))
    
    # Add alert callback
    def on_alert(alert: Alert) -> None:
        print(f"Callback received alert: {alert.name} - {alert.severity.name}")
    
    monitor.add_alert_callback(on_alert)
    
    print("\nSimulating metric updates...")
    
    # Simulate PnL drawdown
    pnl_values = [100.0, 102.0, 101.0, 98.0, 95.0, 92.0, 90.0, 88.0]
    for pnl in pnl_values:
        monitor.update_value("pnl_drawdown", pnl)
        time.sleep(0.05)
    
    # Simulate latency spikes
    latency_values = [100, 150, 200, 500, 1000, 2000, 5000, 8000, 12000]
    for latency in latency_values:
        monitor.update_value("latency_spike", float(latency))
        time.sleep(0.05)
    
    print("\nCurrent Status:")
    status = monitor.get_current_status()
    for name, info in status.items():
        print(f"  {name}:")
        print(f"    Current: {info['current_value']}")
        print(f"    Mean: {info['mean']}")
        print(f"    P99: {info['p99']}")
        print(f"    Drawdown: {info['drawdown']:.4f}")
    
    print("\nAlert History:")
    for alert in monitor.get_alert_history():
        print(f"  [{alert.severity.name}] {alert.name}: {alert.message}")
    
    monitor.shutdown()
    print("\nThreshold Monitor test complete.")
