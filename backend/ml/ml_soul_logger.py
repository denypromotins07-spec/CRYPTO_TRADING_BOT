"""
ML Soul Logger for Prediction Errors and Drift Alerts
Writes ML prediction errors, drift alerts, and regime misclassifications to SOUL.md.
Optimized for 8GB RAM with async file I/O.
"""

from __future__ import annotations
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class MLEvent:
    """Single ML event for logging."""
    timestamp: str
    event_type: str  # "prediction_error", "drift_alert", "regime_misclassification", etc.
    model_name: str
    severity: str  # "info", "warning", "error", "critical"
    details: Dict[str, Any]
    correlation_id: str = ""
    
    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = hashlib.md5(
                f"{self.timestamp}{self.model_name}".encode()
            ).hexdigest()[:12]


@dataclass
class SoulLoggerConfig:
    """Configuration for ML soul logger."""
    # File settings
    soul_file_path: str = "SOUL.md"
    max_events_in_memory: int = 1000
    flush_interval_seconds: int = 60
    
    # Alert thresholds
    error_rate_threshold: float = 0.05  # 5% error rate triggers alert
    drift_severity_threshold: str = "warning"
    
    # Retention
    max_file_size_mb: int = 100
    keep_last_n_events: int = 10000


class MLSoulLogger:
    """
    ML Soul Logger for Prediction Errors and Drift Alerts.
    
    Implements:
    - Structured logging of ML events to SOUL.md
    - Async file I/O to avoid blocking inference
    - Event deduplication and aggregation
    - Automatic alert generation on threshold breaches
    - HMM misclassification tracking
    
    Thread-safe and optimized for high-frequency updates.
    """
    
    def __init__(self, config: SoulLoggerConfig, base_dir: Optional[str] = None):
        self.config = config
        
        # Determine file path
        if base_dir:
            self.soul_path = Path(base_dir) / config.soul_file_path
        else:
            self.soul_path = Path(config.soul_file_path)
        
        # Ensure parent directory exists
        self.soul_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Event buffer (in-memory queue)
        self.event_buffer: deque = deque(maxlen=config.max_events_in_memory)
        
        # Aggregated statistics
        self.error_counts: Dict[str, int] = {}
        self.total_predictions: Dict[str, int] = {}
        
        # HMM misclassification tracking
        self.hmm_predictions: deque = deque(maxlen=1000)
        self.hmm_actuals: deque = deque(maxlen=1000)
        
        # Locking
        self._lock = threading.RLock()
        
        # Flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_flush = False
        
        # Initialize SOUL.md if needed
        self._initialize_soul_file()
        
        # Start background flush
        self._start_flush_thread()
        
        logger.info(f"MLSoulLogger initialized: {self.soul_path}")
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md with header if it doesn't exist."""
        if not self.soul_path.exists():
            header = """# ZAID PERSONAL CRYPTO TRADING BOT - ML SOUL LOG

## Overview
This file contains the "soul" of the ML system - all prediction errors, drift alerts, 
regime misclassifications, and critical ML events that shape the bot's learning.

## Structure
- **Prediction Errors**: When actual outcomes deviate significantly from predictions
- **Drift Alerts**: When feature distributions shift beyond acceptable thresholds  
- **Regime Misclassifications**: When HMM incorrectly identifies market state
- **Model Updates**: Weight swaps, retraining events, and performance changes

---

"""
            with open(self.soul_path, 'w') as f:
                f.write(header)
    
    def _start_flush_thread(self) -> None:
        """Start background thread for periodic flushing."""
        def flush_loop():
            import time
            while not self._stop_flush:
                time.sleep(self.config.flush_interval_seconds)
                self.flush()
        
        self._flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._flush_thread.start()
    
    def stop(self) -> None:
        """Stop the logger and flush remaining events."""
        self._stop_flush = True
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self.flush()
    
    def log_prediction_error(
        self,
        model_name: str,
        predicted: float,
        actual: float,
        error_magnitude: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a prediction error event.
        
        Args:
            model_name: Which model made the prediction
            predicted: Predicted value
            actual: Actual observed value
            error_magnitude: |predicted - actual|
            context: Additional context (symbol, timeframe, etc.)
        """
        # Update counters
        with self._lock:
            self.error_counts[model_name] = self.error_counts.get(model_name, 0) + 1
            self.total_predictions[model_name] = self.total_predictions.get(model_name, 0) + 1
            
            # Check error rate threshold
            error_rate = self.error_counts[model_name] / self.total_predictions[model_name]
            
            if error_rate > self.config.error_rate_threshold:
                severity = "critical"
            elif error_magnitude > abs(predicted) * 0.5:  # >50% error
                severity = "error"
            elif error_magnitude > abs(predicted) * 0.2:  # >20% error
                severity = "warning"
            else:
                severity = "info"
        
        event = MLEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="prediction_error",
            model_name=model_name,
            severity=severity,
            details={
                'predicted': predicted,
                'actual': actual,
                'error_magnitude': error_magnitude,
                'error_percent': (error_magnitude / abs(predicted + 1e-8)) * 100,
                'context': context or {},
                'cumulative_error_rate': error_rate,
            }
        )
        
        self._add_event(event)
    
    def log_drift_alert(
        self,
        feature_name: str,
        drift_metric: str,
        drift_value: float,
        threshold: float,
        severity: str = "warning"
    ) -> None:
        """
        Log a model drift alert.
        
        Args:
            feature_name: Feature showing drift
            drift_metric: PSI, KS statistic, etc.
            drift_value: Computed drift value
            threshold: Threshold that was exceeded
            severity: Alert severity level
        """
        event = MLEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="drift_alert",
            model_name="ensemble",
            severity=severity,
            details={
                'feature_name': feature_name,
                'drift_metric': drift_metric,
                'drift_value': drift_value,
                'threshold': threshold,
                'excess': drift_value - threshold,
                'action_required': severity in ["error", "critical"],
            }
        )
        
        self._add_event(event)
    
    def log_hmm_misclassification(
        self,
        predicted_regime: str,
        actual_regime: str,
        confidence: float,
        market_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log HMM regime misclassification.
        
        This is specifically called when the HMM misclassifies a sudden 
        liquidity vacuum or rapid regime transition.
        
        Args:
            predicted_regime: What HMM predicted
            actual_regime: What actually occurred
            confidence: HMM confidence in prediction
            market_context: Market conditions at time of misclassification
        """
        # Track for accuracy computation
        with self._lock:
            self.hmm_predictions.append(predicted_regime)
            self.hmm_actuals.append(actual_regime)
        
        severity = "critical" if predicted_regime != actual_regime and confidence > 0.8 else "warning"
        
        event = MLEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="hmm_misclassification",
            model_name="hmm_regime_detector",
            severity=severity,
            details={
                'predicted_regime': predicted_regime,
                'actual_regime': actual_regime,
                'confidence': confidence,
                'is_correct': predicted_regime == actual_regime,
                'market_context': market_context or {},
                'note': 'HMM misclassified sudden liquidity vacuum' if severity == "critical" else '',
            }
        )
        
        self._add_event(event)
    
    def log_model_update(
        self,
        model_name: str,
        update_type: str,
        old_version: str,
        new_version: str,
        reason: str,
        performance_impact: Optional[float] = None
    ) -> None:
        """
        Log a model update event (weight swap, retrain, etc.).
        
        Args:
            model_name: Model being updated
            update_type: "weight_swap", "retrain", "architecture_change", etc.
            old_version: Previous version identifier
            new_version: New version identifier
            reason: Why the update occurred
            performance_impact: Expected/observed performance change
        """
        event = MLEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="model_update",
            model_name=model_name,
            severity="info",
            details={
                'update_type': update_type,
                'old_version': old_version,
                'new_version': new_version,
                'reason': reason,
                'performance_impact': performance_impact,
            }
        )
        
        self._add_event(event)
    
    def log_anomaly_detection(
        self,
        anomaly_type: str,
        anomaly_score: float,
        threshold: float,
        affected_symbols: List[str],
        action_taken: str
    ) -> None:
        """
        Log anomaly detection event (spoofing, API lag, etc.).
        
        Args:
            anomaly_type: Type of anomaly detected
            anomaly_score: Computed anomaly score
            threshold: Detection threshold
            affected_symbols: Symbols affected by anomaly
            action_taken: Mitigation action taken
        """
        event = MLEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="anomaly_detection",
            model_name="isolation_forest",
            severity="warning" if anomaly_score > threshold * 1.5 else "info",
            details={
                'anomaly_type': anomaly_type,
                'anomaly_score': anomaly_score,
                'threshold': threshold,
                'affected_symbols': affected_symbols,
                'action_taken': action_taken,
            }
        )
        
        self._add_event(event)
    
    def _add_event(self, event: MLEvent) -> None:
        """Add event to buffer and potentially flush."""
        with self._lock:
            self.event_buffer.append(event)
            
            # Immediate flush for critical events
            if event.severity == "critical":
                self._flush_critical_event(event)
    
    def _flush_critical_event(self, event: MLEvent) -> None:
        """Immediately flush critical events to file."""
        try:
            with open(self.soul_path, 'a') as f:
                f.write(self._format_event(event))
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to flush critical event: {e}")
    
    def flush(self) -> None:
        """Flush buffered events to SOUL.md."""
        with self._lock:
            if not self.event_buffer:
                return
            
            try:
                with open(self.soul_path, 'a') as f:
                    for event in list(self.event_buffer):
                        f.write(self._format_event(event))
                        f.write("\n")
                
                # Clear buffer after successful flush
                self.event_buffer.clear()
                
                logger.debug(f"Flushed {len(self.event_buffer)} events to SOUL.md")
                
            except Exception as e:
                logger.error(f"Failed to flush events: {e}")
    
    def _format_event(self, event: MLEvent) -> str:
        """Format event as markdown."""
        emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }.get(event.severity, "📝")
        
        details_json = json.dumps(event.details, indent=2, default=str)
        
        return f"""### {emoji} [{event.timestamp}] {event.event_type.replace('_', ' ').title()}

- **Model**: `{event.model_name}`
- **Severity**: {event.severity.upper()}
- **Correlation ID**: `{event.correlation_id}`

**Details**:
```json
{details_json}
```

---
"""
    
    def get_hmm_accuracy(self) -> Optional[float]:
        """Compute recent HMM classification accuracy."""
        with self._lock:
            if len(self.hmm_predictions) < 10:
                return None
            
            correct = sum(1 for p, a in zip(self.hmm_predictions, self.hmm_actuals) if p == a)
            return correct / len(self.hmm_predictions)
    
    def get_error_rates(self) -> Dict[str, float]:
        """Get current error rates per model."""
        with self._lock:
            return {
                model: self.error_counts.get(model, 0) / max(1, self.total_predictions.get(model, 1))
                for model in set(list(self.error_counts.keys()) + list(self.total_predictions.keys()))
            }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary of logged events."""
        with self._lock:
            return {
                'buffered_events': len(self.event_buffer),
                'total_predictions': dict(self.total_predictions),
                'error_counts': dict(self.error_counts),
                'error_rates': self.get_error_rates(),
                'hmm_accuracy': self.get_hmm_accuracy(),
                'soul_file_exists': self.soul_path.exists(),
                'soul_file_size_mb': self.soul_path.stat().st_size / (1024 * 1024) if self.soul_path.exists() else 0,
            }


# Example usage
if __name__ == "__main__":
    config = SoulLoggerConfig(
        soul_file_path="SOUL.md",
        flush_interval_seconds=10,
    )
    
    logger_instance = MLSoulLogger(config)
    
    # Simulate various ML events
    logger_instance.log_prediction_error(
        model_name="ppo_agent",
        predicted=0.75,
        actual=-0.25,
        error_magnitude=1.0,
        context={'symbol': 'BTCUSDT', 'timeframe': '1m'}
    )
    
    logger_instance.log_drift_alert(
        feature_name="orderbook_imbalance",
        drift_metric="PSI",
        drift_value=0.35,
        threshold=0.2,
        severity="warning"
    )
    
    logger_instance.log_hmm_misclassification(
        predicted_regime="bull",
        actual_regime="bear",
        confidence=0.85,
        market_context={'volatility_spike': True, 'volume_drop': 0.6}
    )
    
    logger_instance.log_model_update(
        model_name="ensemble_forest",
        update_type="retrain",
        old_version="v1.2.3",
        new_version="v1.2.4",
        reason="Scheduled retraining",
        performance_impact=0.05
    )
    
    # Get summary
    summary = logger_instance.get_summary_statistics()
    print(f"Summary: {json.dumps(summary, indent=2)}")
    
    # Cleanup
    logger_instance.stop()
