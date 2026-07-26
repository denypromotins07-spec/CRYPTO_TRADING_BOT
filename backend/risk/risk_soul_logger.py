#!/usr/bin/env python3
"""
Risk Soul Logger - SOUL.md Integration

Logs tail risk breaches, EVT predictions, and scenario failures
to the SOUL.md file for comprehensive risk audit trail.
Integrates with all Stage 28 risk modules.

Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import threading
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class RiskEventType(Enum):
    """Types of risk events to log."""
    
    # EVT-related events
    EVT_FIT_COMPLETED = "evt_fit_completed"
    TAIL_INDEX_UPDATED = "tail_index_updated"
    EXTREME_EVENT_PREDICTED = "extreme_event_predicted"
    FAT_TAIL_DETECTED = "fat_tail_detected"
    
    # Copula/Dependency events
    COPULA_FIT_COMPLETED = "copula_fit_completed"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    TAIL_DEPENDENCE_SPIKE = "tail_dependence_spike"
    
    # Stress test events
    STRESS_TEST_EXECUTED = "stress_test_executed"
    HISTORICAL_SCENARIO_RUN = "historical_scenario_run"
    REVERSE_STRESS_COMPLETED = "reverse_stress_completed"
    CAPITAL_BUFFER_BREACH = "capital_buffer_breach"
    
    # VaR/Risk limit events
    VAR_BREACH = "var_breach"
    ES_BREACH = "es_breach"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    
    # Trading control events
    TRADING_HALTED = "trading_halted"
    TRADING_RESUMED = "trading_resumed"
    POSITION_REDUCTION_TRIGGERED = "position_reduction_triggered"


class SeverityLevel(Enum):
    """Severity levels for risk events."""
    
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class RiskEvent:
    """A single risk event for logging."""
    
    timestamp: str
    event_type: str
    severity: str
    message: str
    details: Dict[str, Any]
    source_module: str
    event_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'severity': self.severity,
            'message': self.message,
            'details': self.details,
            'source_module': self.source_module,
            'event_id': self.event_id,
        }


@dataclass
class DailyRiskSummary:
    """Daily summary of risk metrics."""
    
    date: str
    total_events: int
    critical_events: int
    var_99: float
    es_99: float
    max_drawdown: float
    tail_index: Optional[float]
    capital_ratio: float
    trading_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RiskSoulLogger:
    """
    Central logger for all risk-related events.
    
    Writes to SOUL.md with structured formatting for
    easy parsing and audit trail maintenance.
    """
    
    def __init__(
        self,
        soul_path: str = "SOUL.md",
        max_events_in_memory: int = 10000,
    ) -> None:
        """
        Initialize the risk soul logger.
        
        Args:
            soul_path: Path to SOUL.md file
            max_events_in_memory: Maximum events to keep in memory
        """
        self.soul_path = Path(soul_path)
        self.max_events = max_events_in_memory
        
        # In-memory event buffer
        self.events: List[RiskEvent] = []
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Event counter for unique IDs
        self._event_counter = 0
        
        # Daily summaries
        self.daily_summaries: Dict[str, DailyRiskSummary] = {}
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md if it doesn't exist."""
        if not self.soul_path.exists():
            initial_content = """# ZAID Trading Bot - Risk SOUL Log

## Overview
This file contains the comprehensive risk audit trail for the ZAID Personal Crypto Trading Bot.
All tail risk events, EVT model updates, stress test results, and capital breaches are logged here.

---

## Event Log

"""
            self.soul_path.write_text(initial_content)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        with self._lock:
            self._event_counter += 1
            return f"RISK-{datetime.now().strftime('%Y%m%d')}-{self._event_counter:06d}"
    
    def log_event(
        self,
        event_type: RiskEventType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: SeverityLevel = SeverityLevel.INFO,
        source_module: str = "unknown",
    ) -> RiskEvent:
        """
        Log a risk event.
        
        Args:
            event_type: Type of risk event
            message: Human-readable message
            details: Additional structured data
            severity: Event severity level
            source_module: Module that generated the event
            
        Returns:
            The created RiskEvent
        """
        event = RiskEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type.value,
            severity=severity.value,
            message=message,
            details=details or {},
            source_module=source_module,
            event_id=self._generate_event_id(),
        )
        
        with self._lock:
            self.events.append(event)
            
            # Trim old events if necessary
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
        
        # Write to SOUL.md
        self._write_to_soul(event)
        
        # Update daily summary
        self._update_daily_summary(event)
        
        return event
    
    def _write_to_soul(self, event: RiskEvent) -> None:
        """Append event to SOUL.md."""
        entry = self._format_event_entry(event)
        
        with self._lock:
            try:
                content = self.soul_path.read_text()
                
                # Find the Event Log section
                if "## Event Log\n" in content:
                    # Insert after the header
                    parts = content.split("## Event Log\n", 1)
                    new_content = parts[0] + "## Event Log\n\n" + entry + parts[1]
                else:
                    # Append at end
                    new_content = content + "\n" + entry
                
                self.soul_path.write_text(new_content)
            except Exception as e:
                # Log error but don't raise
                print(f"Error writing to SOUL.md: {e}")
    
    def _format_event_entry(self, event: RiskEvent) -> str:
        """Format event for SOUL.md."""
        severity_icon = {
            SeverityLevel.INFO: "ℹ️",
            SeverityLevel.WARNING: "⚠️",
            SeverityLevel.HIGH: "🔴",
            SeverityLevel.CRITICAL: "🚨",
            SeverityLevel.EMERGENCY: "💀",
        }.get(SeverityLevel(event.severity), "•")
        
        entry = f"""### {severity_icon} [{event.event_id}] {event.event_type}

- **Timestamp**: `{event.timestamp}`
- **Severity**: {event.severity}
- **Source**: {event.source_module}
- **Message**: {event.message}
"""
        
        if event.details:
            entry += "\n**Details**:\n```json\n"
            entry += json.dumps(event.details, indent=2, default=str)
            entry += "\n```\n"
        
        entry += "\n---\n\n"
        
        return entry
    
    def _update_daily_summary(self, event: RiskEvent) -> None:
        """Update daily risk summary."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today not in self.daily_summaries:
            self.daily_summaries[today] = DailyRiskSummary(
                date=today,
                total_events=0,
                critical_events=0,
                var_99=0.0,
                es_99=0.0,
                max_drawdown=0.0,
                tail_index=None,
                capital_ratio=1.0,
                trading_status="ACTIVE",
            )
        
        summary = self.daily_summaries[today]
        summary.total_events += 1
        
        if event.severity in [SeverityLevel.CRITICAL, SeverityLevel.EMERGENCY]:
            summary.critical_events += 1
        
        # Update from event details if available
        details = event.details
        if 'var_99' in details:
            summary.var_99 = details['var_99']
        if 'es_99' in details:
            summary.es_99 = details['es_99']
        if 'tail_index' in details:
            summary.tail_index = details['tail_index']
        if 'capital_ratio' in details:
            summary.capital_ratio = details['capital_ratio']
        if 'trading_status' in details:
            summary.trading_status = details['trading_status']
    
    # Convenience methods for common event types
    
    def log_evt_fit(
        self,
        xi: float,
        sigma: float,
        threshold: float,
        n_exceedances: int,
        gof_stats: Optional[Dict[str, float]] = None,
    ) -> RiskEvent:
        """Log EVT model fitting completion."""
        severity = SeverityLevel.WARNING if xi > 0.5 else SeverityLevel.INFO
        
        return self.log_event(
            RiskEventType.EVT_FIT_COMPLETED,
            f"EVT model fitted: ξ={xi:.4f}, σ={sigma:.4f}",
            details={
                'xi': xi,
                'sigma': sigma,
                'threshold': threshold,
                'n_exceedances': n_exceedances,
                'has_heavy_tail': xi > 0,
                'tail_index': 1/xi if xi > 0 else None,
                'gof_stats': gof_stats or {},
            },
            severity=severity,
            source_module="evt_peaks_over_threshold",
        )
    
    def log_fat_tail_detection(
        self,
        tail_index: float,
        probability: float,
        asset: str,
    ) -> RiskEvent:
        """Log fat-tail crash prediction."""
        return self.log_event(
            RiskEventType.FAT_TAIL_DETECTED,
            f"Fat-tail crash detected for {asset}: P={probability:.4f}",
            details={
                'tail_index': tail_index,
                'crash_probability': probability,
                'asset': asset,
            },
            severity=SeverityLevel.HIGH,
            source_module="evt_peaks_over_threshold",
        )
    
    def log_tail_dependence_spike(
        self,
        lambda_lower: float,
        asset_pair: tuple,
        historical_avg: float,
    ) -> RiskEvent:
        """Log tail dependence spike during flash crash conditions."""
        increase_pct = ((lambda_lower - historical_avg) / historical_avg * 100) if historical_avg > 0 else 100
        
        return self.log_event(
            RiskEventType.TAIL_DEPENDENCE_SPIKE,
            f"Tail dependence spike: {asset_pair[0]}-{asset_pair[1]} λ_L={lambda_lower:.4f}",
            details={
                'lambda_lower': lambda_lower,
                'asset_pair': list(asset_pair),
                'historical_average': historical_avg,
                'increase_percent': increase_pct,
            },
            severity=SeverityLevel.CRITICAL if lambda_lower > 0.7 else SeverityLevel.HIGH,
            source_module="tail_dependence",
        )
    
    def log_stress_test_result(
        self,
        scenario_name: str,
        portfolio_loss: float,
        var_99: float,
        breach: bool,
    ) -> RiskEvent:
        """Log stress test execution result."""
        severity = SeverityLevel.CRITICAL if breach else SeverityLevel.INFO
        
        return self.log_event(
            RiskEventType.STRESS_TEST_EXECUTED,
            f"Stress test '{scenario_name}': Loss={portfolio_loss:.4f}",
            details={
                'scenario': scenario_name,
                'portfolio_loss': portfolio_loss,
                'var_99': var_99,
                'breach': breach,
            },
            severity=severity,
            source_module="monte_carlo_stress",
        )
    
    def log_capital_breach(
        self,
        current_capital: float,
        required_capital: float,
        shortfall: float,
        trading_halted: bool,
    ) -> RiskEvent:
        """Log economic capital buffer breach."""
        return self.log_event(
            RiskEventType.CAPITAL_BUFFER_BREACH,
            f"Capital breach: ${shortfall:,.2f} shortfall",
            details={
                'current_capital': current_capital,
                'required_capital': required_capital,
                'shortfall': shortfall,
                'capital_ratio': current_capital / required_capital if required_capital > 0 else 0,
                'trading_halted': trading_halted,
                'trading_status': 'HALTED' if trading_halted else 'ACTIVE',
            },
            severity=SeverityLevel.EMERGENCY if trading_halted else SeverityLevel.CRITICAL,
            source_module="economic_capital",
        )
    
    def log_var_breach(
        self,
        actual_loss: float,
        var_limit: float,
        breach_amount: float,
        confidence: float,
    ) -> RiskEvent:
        """Log VaR limit breach."""
        return self.log_event(
            RiskEventType.VAR_BREACH,
            f"VaR breach at {confidence*100}%: ${breach_amount:,.2f} over limit",
            details={
                'actual_loss': actual_loss,
                'var_limit': var_limit,
                'breach_amount': breach_amount,
                'confidence_level': confidence,
            },
            severity=SeverityLevel.CRITICAL,
            source_module="var_calculator",
        )
    
    def log_trading_halt(self, reason: str, details: Optional[Dict] = None) -> RiskEvent:
        """Log trading halt event."""
        return self.log_event(
            RiskEventType.TRADING_HALTED,
            f"Trading halted: {reason}",
            details=details or {'reason': reason},
            severity=SeverityLevel.EMERGENCY,
            source_module="circuit_breaker",
        )
    
    def get_recent_events(
        self,
        n: int = 10,
        severity_filter: Optional[SeverityLevel] = None,
    ) -> List[RiskEvent]:
        """Get recent events, optionally filtered by severity."""
        with self._lock:
            events = self.events[-n:] if n < len(self.events) else self.events.copy()
        
        if severity_filter:
            events = [
                e for e in events 
                if SeverityLevel(e.severity).value >= severity_filter.value
            ]
        
        return events
    
    def get_daily_summary(self, date: Optional[str] = None) -> Optional[DailyRiskSummary]:
        """Get daily risk summary."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        return self.daily_summaries.get(date)
    
    def export_events(self, output_path: str, format: str = "json") -> None:
        """Export events to external file."""
        with self._lock:
            events_data = [e.to_dict() for e in self.events]
        
        output_file = Path(output_path)
        
        if format == "json":
            output_file.write_text(json.dumps(events_data, indent=2))
        elif format == "csv":
            import csv
            if events_data:
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=events_data[0].keys())
                    writer.writeheader()
                    writer.writerows(events_data)


# Global logger instance
_soul_logger: Optional[RiskSoulLogger] = None


def get_soul_logger() -> RiskSoulLogger:
    """Get the global RiskSoulLogger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = RiskSoulLogger()
    return _soul_logger


if __name__ == '__main__':
    # Example usage
    logger = get_soul_logger()
    
    # Log various events
    logger.log_evt_fit(xi=0.35, sigma=0.02, threshold=0.01, n_exceedances=150)
    
    logger.log_fat_tail_detection(
        tail_index=2.86,
        probability=0.15,
        asset='BTC'
    )
    
    logger.log_tail_dependence_spike(
        lambda_lower=0.75,
        asset_pair=('BTC', 'ETH'),
        historical_avg=0.45
    )
    
    logger.log_stress_test_result(
        scenario_name="Black Thursday Replay",
        portfolio_loss=-0.42,
        var_99=-0.25,
        breach=True
    )
    
    logger.log_capital_breach(
        current_capital=250000,
        required_capital=300000,
        shortfall=50000,
        trading_halted=True
    )
    
    print("Events logged to SOUL.md")
    print("\nRecent events:")
    for event in logger.get_recent_events(5):
        print(f"  [{event.severity}] {event.event_type}: {event.message}")
