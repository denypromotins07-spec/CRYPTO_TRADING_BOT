#!/usr/bin/env python3
"""
LOB Soul Logger - Log Impact Model Deviations and LOB Anomalies to SOUL.md

This module logs impact model deviations, order book anomalies, and liquidity
physics events to the SOUL.md file for audit and analysis.

Key Features:
- Structured logging of LOB events
- Impact model deviation tracking
- Flash crash detection alerts
- Liquidity vacuum warnings
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import threading


@dataclass
class LobEvent:
    """Base class for LOB events."""
    timestamp: str
    event_type: str
    symbol: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    details: Dict[str, Any]


@dataclass
class ImpactDeviation(LobEvent):
    """Impact model deviation event."""
    predicted_impact_bps: float
    actual_impact_bps: float
    deviation_bps: float
    model_name: str


@dataclass
class LiquidityAnomaly(LobEvent):
    """Liquidity anomaly event."""
    anomaly_type: str  # VACUUM, STALL, SPIKE
    depth_change_pct: float
    resilience_score: float
    flash_crash_risk: float


@dataclass 
class HawkesAlert(LobEvent):
    """Hawkes process criticality alert."""
    branching_ratio: float
    regime: str  # SUB_CRITICAL, CRITICAL, SUPER_CRITICAL
    intensity: float
    expected_events: float


class LobsSoulLogger:
    """
    Logger for LOB physics events and impact model deviations.
    
    Writes structured events to SOUL.md for audit trail and
    post-trade analysis.
    """
    
    def __init__(self, soul_path: str = "SOUL.md"):
        """
        Initialize the logger.
        
        Args:
            soul_path: Path to SOUL.md log file
        """
        self.soul_path = Path(soul_path)
        self._lock = threading.Lock()
        
        # Event counters
        self.event_counts: Dict[str, int] = {
            'INFO': 0,
            'WARNING': 0,
            'ERROR': 0,
            'CRITICAL': 0,
        }
        
        # Recent events (for summary)
        self.recent_events: List[LobEvent] = []
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md with header if it doesn't exist."""
        if not self.soul_path.exists():
            self.soul_path.parent.mkdir(parents=True, exist_ok=True)
            
            header = """# ZAID Personal Crypto Trading Bot - SOUL.md

## Stage 22: LOB Physics, Market Impact Models, and Hawkes Processes

This file contains the logged events from the Limit Order Book (LOB) analytics
engine, including impact model deviations, liquidity anomalies, and Hawkes
process alerts.

---

"""
            with open(self.soul_path, 'w') as f:
                f.write(header)
    
    def _timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat() + 'Z'
    
    def log_event(self, event: LobEvent) -> None:
        """Log an LOB event to SOUL.md."""
        with self._lock:
            # Update counters
            self.event_counts[event.severity] += 1
            
            # Add to recent events
            self.recent_events.append(event)
            if len(self.recent_events) > 1000:
                self.recent_events = self.recent_events[-500:]
            
            # Format event entry
            entry = self._format_event(event)
            
            # Append to SOUL.md
            with open(self.soul_path, 'a') as f:
                f.write(entry + "\n")
    
    def _format_event(self, event: LobEvent) -> str:
        """Format event as markdown entry."""
        emoji = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨',
        }.get(event.severity, '📝')
        
        header = f"### {emoji} [{event.timestamp}] {event.event_type}"
        
        lines = [
            header,
            "",
            f"- **Symbol**: {event.symbol}",
            f"- **Severity**: {event.severity}",
            "",
            "**Details**:",
            "```json",
            json.dumps(event.details, indent=2),
            "```",
            "",
            "---",
            "",
        ]
        
        # Add type-specific fields
        if isinstance(event, ImpactDeviation):
            lines.insert(-4, f"- **Predicted Impact**: {event.predicted_impact_bps:.2f} bps")
            lines.insert(-4, f"- **Actual Impact**: {event.actual_impact_bps:.2f} bps")
            lines.insert(-4, f"- **Deviation**: {event.deviation_bps:.2f} bps")
            lines.insert(-4, f"- **Model**: {event.model_name}")
            lines.insert(-4, "")
        
        elif isinstance(event, LiquidityAnomaly):
            lines.insert(-4, f"- **Anomaly Type**: {event.anomaly_type}")
            lines.insert(-4, f"- **Depth Change**: {event.depth_change_pct:.2f}%")
            lines.insert(-4, f"- **Resilience Score**: {event.resilience_score:.3f}")
            lines.insert(-4, f"- **Flash Crash Risk**: {event.flash_crash_risk:.2%}")
            lines.insert(-4, "")
        
        elif isinstance(event, HawkesAlert):
            lines.insert(-4, f"- **Branching Ratio**: {event.branching_ratio:.4f}")
            lines.insert(-4, f"- **Regime**: {event.regime}")
            lines.insert(-4, f"- **Current Intensity**: {event.intensity:.2f} events/s")
            lines.insert(-4, f"- **Expected Events (10s)**: {event.expected_events:.2f}")
            lines.insert(-4, "")
        
        return "\n".join(lines)
    
    def log_impact_deviation(
        self,
        symbol: str,
        predicted: float,
        actual: float,
        model_name: str,
        severity: str = "WARNING",
        **kwargs,
    ) -> None:
        """Log an impact model deviation."""
        deviation = actual - predicted
        
        # Adjust severity based on deviation magnitude
        if abs(deviation) > 50:  # >50 bps error
            severity = "CRITICAL"
        elif abs(deviation) > 20:  # >20 bps error
            severity = "ERROR"
        elif abs(deviation) > 10:  # >10 bps error
            severity = "WARNING"
        
        event = ImpactDeviation(
            timestamp=self._timestamp(),
            event_type="IMPACT_DEVIATION",
            symbol=symbol,
            severity=severity,
            details={
                'deviation_pct': (deviation / predicted * 100) if predicted != 0 else 0,
                **kwargs,
            },
            predicted_impact_bps=predicted,
            actual_impact_bps=actual,
            deviation_bps=deviation,
            model_name=model_name,
        )
        
        self.log_event(event)
    
    def log_liquidity_anomaly(
        self,
        symbol: str,
        anomaly_type: str,
        depth_change_pct: float,
        resilience_score: float,
        flash_crash_risk: float,
        **kwargs,
    ) -> None:
        """Log a liquidity anomaly."""
        # Determine severity
        if flash_crash_risk > 0.7 or anomaly_type == "VACUUM":
            severity = "CRITICAL"
        elif flash_crash_risk > 0.5 or depth_change_pct < -50:
            severity = "ERROR"
        elif flash_crash_risk > 0.3 or depth_change_pct < -30:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        event = LiquidityAnomaly(
            timestamp=self._timestamp(),
            event_type="LIQUIDITY_ANOMALY",
            symbol=symbol,
            severity=severity,
            details={
                **kwargs,
            },
            anomaly_type=anomaly_type,
            depth_change_pct=depth_change_pct,
            resilience_score=resilience_score,
            flash_crash_risk=flash_crash_risk,
        )
        
        self.log_event(event)
    
    def log_hawkes_alert(
        self,
        symbol: str,
        branching_ratio: float,
        regime: str,
        intensity: float,
        expected_events: float,
        **kwargs,
    ) -> None:
        """Log a Hawkes process criticality alert."""
        # Determine severity based on regime
        if regime == "SUPER_CRITICAL":
            severity = "CRITICAL"
        elif regime == "CRITICAL":
            severity = "ERROR"
        elif branching_ratio > 0.9:
            severity = "WARNING"
        else:
            severity = "INFO"
        
        event = HawkesAlert(
            timestamp=self._timestamp(),
            event_type="HAWKES_ALERT",
            symbol=symbol,
            severity=severity,
            details={
                **kwargs,
            },
            branching_ratio=branching_ratio,
            regime=regime,
            intensity=intensity,
            expected_events=expected_events,
        )
        
        self.log_event(event)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged events."""
        with self._lock:
            return {
                'total_events': sum(self.event_counts.values()),
                'by_severity': dict(self.event_counts),
                'recent_count': len(self.recent_events),
                'last_event': self.recent_events[-1].timestamp if self.recent_events else None,
            }
    
    def flush(self) -> None:
        """Force flush any buffered writes."""
        # File is written immediately, but this ensures sync
        pass


# Global logger instance
_logger: Optional[LobsSoulLogger] = None


def get_logger(soul_path: str = "SOUL.md") -> LobsSoulLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = LobsSoulLogger(soul_path)
    return _logger


if __name__ == "__main__":
    # Example usage
    logger = get_logger("backend/microstructure/SOUL.md")
    
    # Log some sample events
    logger.log_impact_deviation(
        symbol='BTC',
        predicted=15.0,
        actual=22.5,
        model_name='SquareRootLaw',
        trade_size=5.0,
    )
    
    logger.log_liquidity_anomaly(
        symbol='ETH',
        anomaly_type='VACUUM',
        depth_change_pct=-65.0,
        resilience_score=0.15,
        flash_crash_risk=0.82,
    )
    
    logger.log_hawkes_alert(
        symbol='SOL',
        branching_ratio=1.15,
        regime='SUPER_CRITICAL',
        intensity=50.0,
        expected_events=125.0,
    )
    
    # Print summary
    summary = logger.get_summary()
    print(f"Events Logged: {summary['total_events']}")
    print(f"By Severity: {summary['by_severity']}")
