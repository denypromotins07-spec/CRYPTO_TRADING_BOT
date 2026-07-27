#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
High-Frequency Feature Stores, MLOps, and Concept Drift
File: backend/mlops/mlops_soul_logger.py
Chapter 4: MLOps Pipeline, Shadow Mode Testing, and SOUL.md Logging

Logs drift alerts and shadow performance to SOUL.md.
Updates SOUL.md if a shadow model discovers a new, profitable market regime.
Implements the Observer pattern for real-time event propagation.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
"""

from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading
import hashlib


class EventType(Enum):
    """Types of MLOps events."""
    DRIFT_ALERT = "drift_alert"
    SHADOW_DISCOVERY = "shadow_discovery"
    MODEL_PROMOTION = "model_promotion"
    MODEL_DEMOTION = "model_demotion"
    RETRAIN_TRIGGERED = "retrain_triggered"
    TRADING_HALTED = "trading_halted"
    TRADING_RESUMED = "trading_resumed"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    NEW_REGIME_DETECTED = "new_regime_detected"


class SeverityLevel(Enum):
    """Severity levels for events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MLOpsEvent:
    """An MLOps event to be logged."""
    event_id: str
    event_type: EventType
    severity: SeverityLevel
    timestamp: float
    message: str
    source: str  # Component that generated the event
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'message': self.message,
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass
class SoulEntry:
    """An entry in the SOUL.md log."""
    event: MLOpsEvent
    analysis: str = ""
    actions_taken: List[str] = field(default_factory=list)
    follow_up_required: bool = False


class SOULLogObserver:
    """Observer interface for SOUL.md logging."""
    
    def on_event(self, event: MLOpsEvent) -> None:
        """Handle an MLOps event."""
        raise NotImplementedError


class SOULmdLogger:
    """
    MLOps SOUL.md logger implementing the Observer pattern.
    
    Key features:
    1. Real-time event logging to SOUL.md
    2. Automatic regime detection from shadow models
    3. Event aggregation and summarization
    4. Thread-safe concurrent access
    5. Configurable retention and rotation
    
    The SOUL.md file serves as the "soul" of the trading bot,
    capturing all significant ML/ops events in a human-readable format.
    """
    
    def __init__(
        self,
        soul_md_path: str = "SOUL.md",
        max_entries: int = 10000,
        auto_rotate: bool = True,
        rotation_size_mb: int = 10,
        include_summary: bool = True
    ):
        """
        Initialize the SOUL.md logger.
        
        Args:
            soul_md_path: Path to the SOUL.md file
            max_entries: Maximum entries before rotation
            auto_rotate: Whether to auto-rotate logs
            rotation_size_mb: Size threshold for rotation
            include_summary: Include summary statistics
        """
        self.soul_md_path = Path(soul_md_path)
        self.max_entries = max_entries
        self.auto_rotate = auto_rotate
        self.rotation_size_mb = rotation_size_mb
        self.include_summary = include_summary
        
        # In-memory event buffer
        self.events: List[MLOpsEvent] = []
        self.entries: List[SoulEntry] = []
        
        # Observers (subscribers)
        self.observers: List[SOULLogObserver] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'regime_discoveries': 0,
            'model_promotions': 0
        }
        
        # Current regime tracking
        self.current_regime: Optional[Dict[str, Any]] = None
        self.regime_history: List[Dict[str, Any]] = []
        
        # Ensure directory exists
        self.soul_md_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SOUL.md if it doesn't exist
        self._initialize_soul_md()
    
    def _initialize_soul_md(self) -> None:
        """Initialize or load the SOUL.md file."""
        if not self.soul_md_path.exists():
            self._write_header()
        else:
            self._load_existing_entries()
    
    def _write_header(self) -> None:
        """Write the SOUL.md header."""
        header = f"""# ZAID Personal Crypto Trading Bot - SOUL.md

## System Overview
**Initialized**: {datetime.now().isoformat()}
**Target**: 8k-20k INR/hour in 4-hour trading window
**Assets**: BTC, ETH, SOL, USDT
**Hardware**: AMD Ryzen AI 5 (8GB RAM)

## MLOps Event Log

This file contains the "soul" of the trading bot - all significant machine learning
and operations events that shape the bot's behavior and performance.

---

## Event Summary

| Metric | Value |
|--------|-------|
| Total Events | 0 |
| Regime Discoveries | 0 |
| Model Promotions | 0 |
| Trading Halts | 0 |

---

## Recent Events

"""
        with open(self.soul_md_path, 'w') as f:
            f.write(header)
    
    def _load_existing_entries(self) -> None:
        """Load existing entries from SOUL.md (simplified - parses recent events)."""
        # In production, would parse the full file
        # For now, just count existing entries
        try:
            with open(self.soul_md_path, 'r') as f:
                content = f.read()
                # Count event markers
                event_count = content.count('### Event:')
                self.stats['total_events'] = event_count
        except Exception:
            pass
    
    def register_observer(self, observer: SOULLogObserver) -> None:
        """Register an observer for events."""
        with self._lock:
            self.observers.append(observer)
    
    def unregister_observer(self, observer: SOULLogObserver) -> None:
        """Unregister an observer."""
        with self._lock:
            if observer in self.observers:
                self.observers.remove(observer)
    
    def log_event(
        self,
        event_type: EventType,
        message: str,
        source: str,
        severity: SeverityLevel = SeverityLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None,
        analysis: Optional[str] = None,
        actions: Optional[List[str]] = None
    ) -> str:
        """
        Log an MLOps event to SOUL.md.
        
        Args:
            event_type: Type of event
            message: Human-readable message
            source: Component that generated the event
            severity: Severity level
            metadata: Additional structured data
            analysis: Optional analysis text
            actions: Optional list of actions taken
            
        Returns:
            Event ID
        """
        with self._lock:
            # Generate event ID
            event_id = self._generate_event_id(event_type, time.time())
            
            # Create event
            event = MLOpsEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                timestamp=time.time(),
                message=message,
                source=source,
                metadata=metadata or {}
            )
            
            # Update statistics
            self.stats['total_events'] += 1
            type_key = event_type.value
            self.stats['events_by_type'][type_key] = \
                self.stats['events_by_type'].get(type_key, 0) + 1
            
            sev_key = severity.value
            self.stats['events_by_severity'][sev_key] = \
                self.stats['events_by_severity'].get(sev_key, 0) + 1
            
            # Special handling for specific event types
            if event_type == EventType.NEW_REGIME_DETECTED:
                self.stats['regime_discoveries'] += 1
                self._handle_regime_discovery(event)
            elif event_type == EventType.MODEL_PROMOTION:
                self.stats['model_promotions'] += 1
            
            # Create entry
            entry = SoulEntry(
                event=event,
                analysis=analysis or "",
                actions_taken=actions or []
            )
            
            # Store in memory
            self.events.append(event)
            self.entries.append(entry)
            
            # Write to file
            self._append_to_soul_md(entry)
            
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_event(event)
                except Exception as e:
                    print(f"Observer error: {e}")
            
            # Check rotation
            if self.auto_rotate and len(self.entries) >= self.max_entries:
                self._rotate_log()
            
            return event_id
    
    def _generate_event_id(self, event_type: EventType, timestamp: float) -> str:
        """Generate a unique event ID."""
        content = f"{event_type.value}:{timestamp}:{self.stats['total_events']}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _append_to_soul_md(self, entry: SoulEntry) -> None:
        """Append an entry to the SOUL.md file."""
        event = entry.event
        
        # Format the entry
        entry_text = f"""
### Event: `{event.event_id}`

**Type**: {event.event_type.value}  
**Severity**: {event.severity.value}  
**Time**: {event.datetime}  
**Source**: {event.source}

**Message**: {event.message}

"""
        if entry.analysis:
            entry_text += f"**Analysis**: {entry.analysis}\n\n"
        
        if entry.actions_taken:
            entry_text += "**Actions Taken**:\n"
            for action in entry.actions_taken:
                entry_text += f"- {action}\n"
            entry_text += "\n"
        
        if event.metadata:
            entry_text += "**Metadata**:\n```json\n"
            entry_text += json.dumps(event.metadata, indent=2, default=str)
            entry_text += "\n```\n"
        
        entry_text += "\n---\n"
        
        # Append to file
        with open(self.soul_md_path, 'a') as f:
            f.write(entry_text)
    
    def _handle_regime_discovery(self, event: MLOpsEvent) -> None:
        """Handle a new regime discovery event."""
        regime_data = event.metadata.get('regime', {})
        
        # Store regime
        regime_entry = {
            'discovered_at': event.timestamp,
            'event_id': event.event_id,
            **regime_data
        }
        
        self.regime_history.append(regime_entry)
        self.current_regime = regime_entry
        
        # Check if this regime is profitable
        if regime_data.get('profitability_score', 0) > 0.7:
            # Log a shadow discovery
            self.log_event(
                event_type=EventType.SHADOW_DISCOVERY,
                message=f"Shadow model discovered profitable regime: {regime_data.get('name', 'unknown')}",
                source="mlops_soul_logger",
                severity=SeverityLevel.WARNING,
                metadata={
                    'regime': regime_data,
                    'parent_event': event.event_id
                },
                analysis="This regime shows strong profitability signals. Consider model retraining.",
                actions=[
                    "Flagged for model retraining",
                    "Added to regime watchlist",
                    "Shadow model performance boosted"
                ]
            )
    
    def _rotate_log(self) -> None:
        """Rotate the SOUL.md log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = self.soul_md_path.parent / f"SOUL_{timestamp}.md"
        
        # Move current file
        self.soul_md_path.rename(rotated_path)
        
        # Write new header
        self._write_header()
        
        # Add rotation note
        self.log_event(
            event_type=EventType.INFO,
            message=f"Log rotated to {rotated_path.name}",
            source="mlops_soul_logger"
        )
    
    def update_summary(self) -> None:
        """Update the summary section in SOUL.md."""
        with self._lock:
            # Read current content
            with open(self.soul_md_path, 'r') as f:
                content = f.read()
            
            # Generate new summary
            summary = f"""## Event Summary

| Metric | Value |
|--------|-------|
| Total Events | {self.stats['total_events']} |
| Regime Discoveries | {self.stats['regime_discoveries']} |
| Model Promotions | {self.stats['model_promotions']} |
| Critical Events | {self.stats['events_by_severity'].get('critical', 0)} |

### Events by Type

| Type | Count |
|------|-------|
"""
            for event_type, count in sorted(self.stats['events_by_type'].items()):
                summary += f"| {event_type} | {count} |\n"
            
            summary += "\n### Events by Severity\n\n"
            summary += "| Severity | Count |\n|----------|-------|\n"
            for severity, count in sorted(self.stats['events_by_severity'].items()):
                summary += f"| {severity} | {count} |\n"
            
            summary += "\n---\n\n"
            
            # Find and replace summary section
            start_marker = "## Event Summary"
            end_marker = "---\n\n## Recent Events"
            
            if start_marker in content and end_marker in content:
                start_idx = content.find(start_marker)
                end_idx = content.find(end_marker) + len(end_marker)
                content = content[:start_idx] + summary + content[end_idx:]
            else:
                # Insert after header
                header_end = content.find("---\n\n## Recent Events")
                if header_end != -1:
                    content = content[:header_end] + "\n" + summary + content[header_end:]
            
            # Write back
            with open(self.soul_md_path, 'w') as f:
                f.write(content)
    
    def get_recent_events(
        self,
        limit: int = 10,
        event_type_filter: Optional[EventType] = None,
        severity_filter: Optional[SeverityLevel] = None
    ) -> List[MLOpsEvent]:
        """Get recent events with optional filtering."""
        with self._lock:
            events = self.events[-limit:] if limit else self.events[:]
            
            if event_type_filter:
                events = [e for e in events if e.event_type == event_type_filter]
            
            if severity_filter:
                severity_order = list(SeverityLevel)
                min_idx = severity_order.index(severity_filter)
                events = [
                    e for e in events 
                    if severity_order.index(e.severity) >= min_idx
                ]
            
            return events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        with self._lock:
            return {
                **self.stats,
                'current_regime': self.current_regime,
                'regime_history_count': len(self.regime_history),
                'observers_count': len(self.observers),
                'file_size_mb': self._get_file_size_mb()
            }
    
    def _get_file_size_mb(self) -> float:
        """Get current SOUL.md file size in MB."""
        try:
            return self.soul_md_path.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def log_drift_alert(
        self,
        feature_name: str,
        ks_statistic: float,
        p_value: float,
        severity: str = "high"
    ) -> str:
        """Convenience method for logging drift alerts."""
        sev_map = {
            'low': SeverityLevel.INFO,
            'medium': SeverityLevel.WARNING,
            'high': SeverityLevel.ERROR,
            'critical': SeverityLevel.CRITICAL
        }
        
        return self.log_event(
            event_type=EventType.DRIFT_ALERT,
            message=f"Concept drift detected in feature '{feature_name}'",
            source="drift_detector",
            severity=sev_map.get(severity, SeverityLevel.WARNING),
            metadata={
                'feature_name': feature_name,
                'ks_statistic': ks_statistic,
                'p_value': p_value
            },
            analysis=f"KS statistic of {ks_statistic:.4f} with p-value {p_value:.6f} indicates distribution shift.",
            actions=[
                "Evaluated drift severity",
                "Checked trading halt conditions",
                "Alerted MLOps team"
            ]
        )
    
    def log_shadow_discovery(
        self,
        model_id: str,
        regime_name: str,
        profitability_score: float,
        evidence: Dict[str, Any]
    ) -> str:
        """Convenience method for logging shadow model discoveries."""
        return self.log_event(
            event_type=EventType.NEW_REGIME_DETECTED,
            message=f"Shadow model '{model_id}' discovered new regime: {regime_name}",
            source="shadow_engine",
            severity=SeverityLevel.WARNING,
            metadata={
                'model_id': model_id,
                'regime': {
                    'name': regime_name,
                    'profitability_score': profitability_score,
                    **evidence
                }
            },
            analysis=f"Regime shows {profitability_score:.2%} profitability score based on shadow model analysis.",
            actions=[
                "Recorded regime characteristics",
                "Scheduled model retraining evaluation",
                "Updated regime watchlist"
            ]
        )


# Global logger instance
_soul_logger: Optional[SOULmdLogger] = None


def get_soul_logger() -> SOULmdLogger:
    """Get or create the global SOUL.md logger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = SOULmdLogger()
    return _soul_logger


if __name__ == '__main__':
    # Test the SOUL.md logger
    print("Testing SOUL.md Logger for ZAID Trading Bot...")
    
    # Create logger
    logger = SOULmdLogger(soul_md_path="./SOUL_test.md")
    
    # Log various events
    print("\nLogging test events...")
    
    # Drift alert
    logger.log_drift_alert(
        feature_name="volatility_20",
        ks_statistic=0.35,
        p_value=0.001,
        severity="high"
    )
    print("  Logged drift alert")
    
    # Shadow discovery
    logger.log_shadow_discovery(
        model_id="shadow_v2",
        regime_name="high_momentum_bull",
        profitability_score=0.82,
        evidence={
            'avg_return': 0.025,
            'sharpe_ratio': 2.1,
            'win_rate': 0.68
        }
    )
    print("  Logged shadow discovery")
    
    # Trading halt
    logger.log_event(
        event_type=EventType.TRADING_HALTED,
        message="Trading halted due to critical drift detection",
        source="decay_monitor",
        severity=SeverityLevel.CRITICAL,
        metadata={
            'trigger': 'sharpe_below_threshold',
            'current_sharpe': 0.5
        },
        analysis="Model Sharpe ratio dropped below minimum threshold of 1.0",
        actions=[
            "All positions closed",
            "New orders blocked",
            "Retrain pipeline triggered"
        ]
    )
    print("  Logged trading halt")
    
    # Update summary
    logger.update_summary()
    print("  Updated summary")
    
    # Get stats
    stats = logger.get_stats()
    print(f"\nLogger stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get recent events
    recent = logger.get_recent_events(limit=3)
    print(f"\nRecent events ({len(recent)}):")
    for event in recent:
        print(f"  [{event.severity.value}] {event.event_type.value}: {event.message}")
    
    print(f"\nSOUL.md written to: {logger.soul_md_path.absolute()}")
    print("\nSOUL.md Logger test completed!")
