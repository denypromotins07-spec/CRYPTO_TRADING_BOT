#!/usr/bin/env python3
"""
LOB Soul Logger - Impact Model Deviation and Anomaly Logging

This module logs impact model deviations and LOB anomalies to SOUL.md,
providing a persistent audit trail for model performance and market conditions.

Key Features:
- Automatic anomaly detection
- Persistent logging to SOUL.md
- Alert escalation for critical conditions
- Memory-efficient circular buffer for recent events

Author: ZAID Personal Crypto Trading Bot
Stage: 22 - LOB Physics & Hawkes Processes
"""

from __future__ import annotations
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock
import hashlib


class AnomalyType(Enum):
    """Types of LOB anomalies."""
    IMPACT_DEVIATION = "impact_deviation"
    LIQUIDITY_VACUUM = "liquidity_vacuum"
    FLASH_CRASH_WARNING = "flash_crash_warning"
    SPOOFING_DETECTED = "spoofing_detected"
    BRANCHING_CRITICAL = "branching_critical"
    RESILIENCE_FAILURE = "resilience_failure"
    PROPAGATOR_FAILURE = "propagator_failure"
    ALPHA_EXHAUSTION = "alpha_exhaustion"
    CANCELLATION_SPIKE = "cancellation_spike"
    DEPTH_ANOMALY = "depth_anomaly"


class SeverityLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """Record of a detected anomaly."""
    timestamp: float
    anomaly_type: str
    severity: str
    asset: str
    description: str
    metrics: Dict[str, float]
    model_predictions: Dict[str, float]
    actual_values: Dict[str, float]
    deviation_bps: float
    recommended_action: str
    event_id: str


class LOBSoulLogger:
    """
    Logs LOB anomalies and impact model deviations to SOUL.md.
    
    Provides persistent tracking of model performance and market
    anomalies for post-trade analysis and model improvement.
    """
    
    def __init__(
        self,
        log_dir: str = "./",
        max_recent_events: int = 1000,
        auto_flush_interval_s: float = 60.0,
    ):
        """
        Initialize the LOB soul logger.
        
        Args:
            log_dir: Directory for log files
            max_recent_events: Maximum events in memory buffer
            auto_flush_interval_s: Interval for auto-flushing to disk
        """
        self.log_dir = os.path.expanduser(log_dir)
        self.max_recent_events = max_recent_events
        self.auto_flush_interval = auto_flush_interval_s
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Event buffer
        self._recent_events: Deque[AnomalyEvent] = deque(maxlen=max_recent_events)
        
        # Statistics tracking
        self._event_counts: Dict[str, int] = {}
        self._severity_counts: Dict[str, int] = {}
        self._last_flush: float = time.time()
        
        # Thread safety
        self._lock = Lock()
        
        # SOUL.md file path
        self.soul_file = os.path.join(self.log_dir, "SOUL.md")
        
        # Initialize SOUL.md if it doesn't exist
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize or append to SOUL.md file."""
        header = """# ZAID Personal Crypto Trading Bot - SOUL Log

## Stage 22: LOB Physics & Market Impact Models

This log captures Limit Order Book anomalies, impact model deviations,
and liquidity physics events for continuous model improvement.

---

"""
        if not os.path.exists(self.soul_file):
            with open(self.soul_file, 'w') as f:
                f.write(header)
        else:
            # Check if we need to add header (empty file)
            with open(self.soul_file, 'r') as f:
                content = f.read(100)
                if not content.strip():
                    with open(self.soul_file, 'w') as fw:
                        fw.write(header)
    
    def _generate_event_id(self, event: AnomalyEvent) -> str:
        """Generate unique event ID."""
        data = f"{event.timestamp}{event.anomaly_type}{event.asset}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def log_anomaly(
        self,
        anomaly_type: AnomalyType,
        severity: SeverityLevel,
        asset: str,
        description: str,
        metrics: Optional[Dict[str, float]] = None,
        model_predictions: Optional[Dict[str, float]] = None,
        actual_values: Optional[Dict[str, float]] = None,
        recommended_action: str = "Monitor",
    ) -> AnomalyEvent:
        """
        Log an anomaly event.
        
        Args:
            anomaly_type: Type of anomaly detected
            severity: Severity level
            asset: Affected asset
            description: Human-readable description
            metrics: Relevant metrics at time of anomaly
            model_predictions: Model predictions that were made
            actual_values: Actual observed values
            recommended_action: Suggested action
            
        Returns:
            The created AnomalyEvent
        """
        with self._lock:
            now = time.time()
            
            # Calculate deviation if both predictions and actuals provided
            deviation_bps = 0.0
            if model_predictions and actual_values:
                for key in model_predictions:
                    if key in actual_values:
                        pred = model_predictions[key]
                        actual = actual_values[key]
                        if abs(pred) > 1e-10:
                            deviation_bps = max(deviation_bps, abs(pred - actual) / abs(pred) * 10000)
            
            event = AnomalyEvent(
                timestamp=now,
                anomaly_type=anomaly_type.value,
                severity=severity.value,
                asset=asset.upper(),
                description=description,
                metrics=metrics or {},
                model_predictions=model_predictions or {},
                actual_values=actual_values or {},
                deviation_bps=deviation_bps,
                recommended_action=recommended_action,
                event_id="",  # Will be set below
            )
            
            event.event_id = self._generate_event_id(event)
            
            # Add to buffer
            self._recent_events.append(event)
            
            # Update counts
            self._event_counts[anomaly_type.value] = \
                self._event_counts.get(anomaly_type.value, 0) + 1
            self._severity_counts[severity.value] = \
                self._severity_counts.get(severity.value, 0) + 1
            
            # Auto-flush if interval exceeded
            if now - self._last_flush > self.auto_flush_interval:
                self.flush_to_disk()
            
            # Immediate flush for critical events
            if severity == SeverityLevel.CRITICAL:
                self._flush_critical_event(event)
            
            return event
    
    def _format_event_markdown(self, event: AnomalyEvent) -> str:
        """Format event as Markdown entry."""
        dt = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        md = f"""
### [{dt}] {event.anomaly_type.upper()} - {event.severity.upper()}

**Asset:** {event.asset}  
**Event ID:** `{event.event_id}`  
**Description:** {event.description}

#### Metrics
"""
        if event.metrics:
            for key, value in event.metrics.items():
                md += f"- **{key}:** {value:.6f}\n"
        else:
            md += "*No specific metrics*\n"
        
        if event.model_predictions or event.actual_values:
            md += "\n#### Model Comparison\n"
            if event.model_predictions:
                md += "**Predictions:**\n"
                for key, value in event.model_predictions.items():
                    md += f"- {key}: {value:.6f}\n"
            if event.actual_values:
                md += "**Actual:**\n"
                for key, value in event.actual_values.items():
                    md += f"- {key}: {value:.6f}\n"
            
            if event.deviation_bps > 0:
                md += f"\n**Deviation:** {event.deviation_bps:.2f} bps\n"
        
        md += f"\n**Recommended Action:** {event.recommended_action}\n"
        md += "\n---\n"
        
        return md
    
    def _flush_critical_event(self, event: AnomalyEvent) -> None:
        """Immediately flush critical event to disk."""
        md_entry = self._format_event_markdown(event)
        
        with open(self.soul_file, 'a') as f:
            f.write(f"\n## CRITICAL ALERT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(md_entry)
    
    def flush_to_disk(self) -> None:
        """Flush all recent events to SOUL.md."""
        with self._lock:
            if not self._recent_events:
                return
            
            entries = []
            for event in self._recent_events:
                entries.append(self._format_event_markdown(event))
            
            with open(self.soul_file, 'a') as f:
                f.write(f"\n## Batch Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"*{len(entries)} events*\n\n")
                for entry in entries:
                    f.write(entry)
            
            self._last_flush = time.time()
    
    def get_recent_events(
        self,
        limit: int = 100,
        anomaly_type: Optional[AnomalyType] = None,
        severity: Optional[SeverityLevel] = None,
        asset: Optional[str] = None,
    ) -> List[AnomalyEvent]:
        """
        Get recent events with optional filtering.
        
        Args:
            limit: Maximum events to return
            anomaly_type: Filter by anomaly type
            severity: Filter by severity
            asset: Filter by asset
            
        Returns:
            List of matching AnomalyEvents
        """
        with self._lock:
            events = list(self._recent_events)
            
            if anomaly_type:
                events = [e for e in events if e.anomaly_type == anomaly_type.value]
            if severity:
                events = [e for e in events if e.severity == severity.value]
            if asset:
                events = [e for e in events if e.asset == asset.upper()]
            
            return events[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics about logged events."""
        with self._lock:
            return {
                'total_events': len(self._recent_events),
                'event_counts': dict(self._event_counts),
                'severity_counts': dict(self._severity_counts),
                'last_flush': self._last_flush,
                'soul_file': self.soul_file,
            }
    
    def check_propagator_failure(
        self,
        predicted_impact: float,
        actual_impact: float,
        threshold_bps: float = 50.0,
        asset: str = "BTC",
    ) -> Optional[AnomalyEvent]:
        """
        Check for propagator model failure (massive liquidity vacuum).
        
        Args:
            predicted_impact: Model's predicted impact
            actual_impact: Observed actual impact
            threshold_bps: Deviation threshold in bps
            asset: Asset symbol
            
        Returns:
            AnomalyEvent if failure detected, None otherwise
        """
        deviation = abs(actual_impact - predicted_impact)
        
        if deviation > threshold_bps / 10000:
            return self.log_anomaly(
                anomaly_type=AnomalyType.PROPAGATOR_FAILURE,
                severity=SeverityLevel.HIGH,
                asset=asset,
                description=f"Propagator model failed to predict massive liquidity vacuum. "
                           f"Predicted: {predicted_impact*10000:.1f}bps, "
                           f"Actual: {actual_impact*10000:.1f}bps",
                metrics={'deviation_bps': deviation * 10000},
                model_predictions={'impact': predicted_impact},
                actual_values={'impact': actual_impact},
                recommended_action="Recalibrate propagator model parameters immediately",
            )
        return None
    
    def alert_on_flash_crash_risk(
        self,
        resilience_score: float,
        branching_ratio: float,
        asset: str = "BTC",
    ) -> Optional[AnomalyEvent]:
        """
        Alert on potential flash crash conditions.
        
        Args:
            resilience_score: Current order book resilience (0-1)
            branching_ratio: Hawkes process branching ratio
            asset: Asset symbol
            
        Returns:
            AnomalyEvent if risk detected
        """
        if resilience_score < 0.2 or branching_ratio > 0.95:
            severity = SeverityLevel.CRITICAL if resilience_score < 0.1 else SeverityLevel.HIGH
            
            return self.log_anomaly(
                anomaly_type=AnomalyType.FLASH_CRASH_WARNING,
                severity=severity,
                asset=asset,
                description=f"Flash crash risk elevated. Resilience: {resilience_score:.3f}, "
                           f"Branching Ratio: {branching_ratio:.3f}",
                metrics={
                    'resilience_score': resilience_score,
                    'branching_ratio': branching_ratio,
                },
                recommended_action="Reduce position size, widen stops, prepare for volatility",
            )
        return None
    
    def reset_statistics(self) -> None:
        """Reset event statistics (not the event log)."""
        with self._lock:
            self._event_counts.clear()
            self._severity_counts.clear()


# Global instance for easy access
_soul_logger: Optional[LOBSoulLogger] = None


def get_soul_logger(log_dir: str = "./") -> LOBSoulLogger:
    """Get or create the global soul logger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = LOBSoulLogger(log_dir=log_dir)
    return _soul_logger


if __name__ == "__main__":
    # Demo usage
    logger = LOBSoulLogger(log_dir="./")
    
    # Log some sample anomalies
    logger.log_anomaly(
        anomaly_type=AnomalyType.IMPACT_DEVIATION,
        severity=SeverityLevel.WARNING,
        asset="BTC",
        description="Square-root model underestimating impact during high volatility",
        metrics={'volatility': 0.08, 'volume': 100.0},
        model_predictions={'impact_bps': 15.0},
        actual_values={'impact_bps': 25.0},
        recommended_action="Increase impact coefficient for current regime",
    )
    
    logger.check_propagator_failure(
        predicted_impact=0.001,  # 10 bps
        actual_impact=0.008,     # 80 bps
        threshold_bps=50,
        asset="ETH",
    )
    
    logger.alert_on_flash_crash_risk(
        resilience_score=0.15,
        branching_ratio=0.97,
        asset="SOL",
    )
    
    # Flush to disk
    logger.flush_to_disk()
    
    # Print statistics
    stats = logger.get_statistics()
    print(f"Total Events: {stats['total_events']}")
    print(f"Event Counts: {stats['event_counts']}")
    print(f"Severity Counts: {stats['severity_counts']}")
    print(f"SOUL File: {stats['soul_file']}")
