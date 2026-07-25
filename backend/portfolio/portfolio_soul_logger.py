#!/usr/bin/env python3
"""
Portfolio Soul Logger for Crypto Trading Bot

Logs allocation drift, rebalance events, and slippage to SOUL.md.
Provides comprehensive audit trail for portfolio management decisions.
Critical for post-trade analysis and regulatory compliance.

# Logged Events:
- Allocation drift alerts
- Rebalance triggers and executions
- Transaction slippage tracking
- Reserve deployment events
- Performance attribution
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
from pathlib import Path
from enum import Enum
import json
import threading


class EventType(Enum):
    """Types of portfolio events."""
    ALLOCATION_DRIFT = "allocation_drift"
    REBALANCE_TRIGGERED = "rebalance_triggered"
    REBALANCE_EXECUTED = "rebalance_executed"
    REBALANCE_FAILED = "rebalance_failed"
    RESERVE_DEPLOYED = "reserve_deployed"
    POSITION_SCALED = "position_scaled"
    DRAWDOWN_ALERT = "drawdown_alert"
    KELLY_ADJUSTMENT = "kelly_adjustment"
    PERFORMANCE_UPDATE = "performance_update"
    LIQUIDITY_WARNING = "liquidity_warning"


class SeverityLevel(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PortfolioEvent:
    """Single portfolio event for logging."""
    timestamp: str
    event_type: EventType
    severity: SeverityLevel
    message: str
    details: dict[str, Any]
    portfolio_value: float
    weights_before: dict[str, float]
    weights_after: Optional[dict[str, float]]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "portfolio_value": self.portfolio_value,
            "weights_before": self.weights_before,
            "weights_after": self.weights_after,
        }


@dataclass
class RebalanceRecord:
    """Detailed record of a rebalance operation."""
    timestamp: str
    trigger_type: str
    drift_before: dict[str, float]
    target_weights: dict[str, float]
    trades_executed: list[dict]
    total_turnover: float
    estimated_cost_bps: float
    actual_slippage_bps: float
    success: bool
    failure_reason: Optional[str] = None


@dataclass 
class DriftAlert:
    """Alert for allocation drift exceeding thresholds."""
    timestamp: str
    asset: str
    current_weight: float
    target_weight: float
    drift_pct: float
    threshold_pct: float
    action_recommended: str


class PortfolioSoulLogger:
    """
    Comprehensive portfolio event logger.
    
    Writes all portfolio management events to SOUL.md with
    structured formatting for easy analysis and auditing.
    
    # Features:
    - Thread-safe logging
    - Structured JSON entries in markdown
    - Automatic drift detection
    - Rebalance slippage tracking
    - Performance attribution logging
    """
    
    DEFAULT_LOG_PATH: str = "SOUL.md"
    
    def __init__(
        self,
        log_path: Optional[str] = None,
        portfolio_id: str = "ZAID_CRYPTO_BOT",
    ) -> None:
        """
        Initialize soul logger.
        
        Args:
            log_path: Path to SOUL.md file
            portfolio_id: Unique portfolio identifier
        """
        self.log_path = Path(log_path or self.DEFAULT_LOG_PATH)
        self.portfolio_id = portfolio_id
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Event history (in-memory cache)
        self._events: list[PortfolioEvent] = []
        self._rebalance_history: list[RebalanceRecord] = []
        self._drift_alerts: list[DriftAlert] = []
        
        # State tracking
        self._last_weights: dict[str, float] = {}
        self._last_rebalance_time: Optional[str] = None
        self._cumulative_slippage_bps: float = 0.0
        
        # Initialize log file
        self._initialize_log()
    
    def _initialize_log(self) -> None:
        """Create or initialize the SOUL.md log file."""
        if not self.log_path.exists():
            self._write_header()
    
    def _write_header(self) -> None:
        """Write markdown header to new log file."""
        header = f"""# Portfolio Soul Log - {self.portfolio_id}

## Overview
This log contains the complete audit trail of portfolio management decisions,
allocation drift events, rebalancing operations, and performance metrics.

---

## Configuration
- **Portfolio ID**: {self.portfolio_id}
- **Assets**: BTC, SOL, ETH, USDT
- **Created**: {datetime.utcnow().isoformat()}Z

---

## Event Log

"""
        with open(self.log_path, 'w') as f:
            f.write(header)
    
    def _append_entry(self, entry_type: str, content: str, metadata: Optional[dict] = None) -> None:
        """Append an entry to the log file."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        entry = f"""### [{entry_type}] {timestamp}

{content}

"""
        if metadata:
            entry += f"```json\n{json.dumps(metadata, indent=2)}\n```\n\n"
        
        entry += "---\n\n"
        
        with self._lock:
            with open(self.log_path, 'a') as f:
                f.write(entry)
    
    def log_allocation_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        max_drift_asset: str,
        max_drift_pct: float,
        threshold_pct: float,
    ) -> None:
        """
        Log allocation drift event.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target allocation weights
            portfolio_value: Current portfolio value
            max_drift_asset: Asset with maximum drift
            max_drift_pct: Maximum drift percentage
            threshold_pct: Drift threshold that was exceeded
        """
        drift_details = {asset: current_weights.get(asset, 0) - target_weights.get(asset, 0) 
                        for asset in target_weights}
        
        alert = DriftAlert(
            timestamp=datetime.utcnow().isoformat() + "Z",
            asset=max_drift_asset,
            current_weight=current_weights.get(max_drift_asset, 0),
            target_weight=target_weights.get(max_drift_asset, 0),
            drift_pct=max_drift_pct,
            threshold_pct=threshold_pct,
            action_recommended="REBALANCE" if max_drift_pct > threshold_pct else "MONITOR",
        )
        
        with self._lock:
            self._drift_alerts.append(alert)
        
        content = f"""**Allocation drift detected exceeding threshold.**

| Metric | Value |
|--------|-------|
| Maximum Drift Asset | {max_drift_asset} |
| Drift Percentage | {max_drift_pct:.2%} |
| Threshold | {threshold_pct:.2%} |
| Portfolio Value | ${portfolio_value:,.2f} |

**Drift by Asset:**
"""
        for asset in target_weights:
            drift = drift_details.get(asset, 0)
            direction = "↑" if drift > 0 else "↓" if drift < 0 else "→"
            content += f"- {asset}: {direction} {abs(drift):.2%}\n"
        
        event = PortfolioEvent(
            timestamp=alert.timestamp,
            event_type=EventType.ALLOCATION_DRIFT,
            severity=SeverityLevel.WARNING if max_drift_pct > threshold_pct else SeverityLevel.INFO,
            message=f"Allocation drift of {max_drift_pct:.2%} detected in {max_drift_asset}",
            details={"drift_by_asset": drift_details},
            portfolio_value=portfolio_value,
            weights_before=current_weights,
            weights_after=None,
        )
        
        self._log_event(event)
    
    def log_rebalance_triggered(
        self,
        trigger_type: str,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        estimated_cost_bps: float,
    ) -> None:
        """Log that a rebalance has been triggered."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        content = f"""**Rebalance triggered by {trigger_type}.**

| Parameter | Value |
|-----------|-------|
| Trigger Type | {trigger_type} |
| Estimated Cost | {estimated_cost_bps:.1f} bps |
| Portfolio Value | ${portfolio_value:,.2f} |

**Current vs Target Weights:**
"""
        for asset in target_weights:
            curr = current_weights.get(asset, 0)
            tgt = target_weights.get(asset, 0)
            diff = curr - tgt
            content += f"- {asset}: {curr:.2%} → {tgt:.2%} (diff: {diff:+.2%})\n"
        
        event = PortfolioEvent(
            timestamp=timestamp,
            event_type=EventType.REBALANCE_TRIGGERED,
            severity=SeverityLevel.INFO,
            message=f"Rebalance triggered by {trigger_type}",
            details={
                "trigger_type": trigger_type,
                "estimated_cost_bps": estimated_cost_bps,
            },
            portfolio_value=portfolio_value,
            weights_before=current_weights,
            weights_after=target_weights,
        )
        
        self._log_event(event)
    
    def log_rebalance_executed(
        self,
        trades: list[dict],
        total_turnover: float,
        actual_slippage_bps: float,
        portfolio_value: float,
        final_weights: dict[str, float],
    ) -> None:
        """Log successful rebalance execution."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Update cumulative slippage
        self._cumulative_slippage_bps += actual_slippage_bps
        
        record = RebalanceRecord(
            timestamp=timestamp,
            trigger_type="threshold_breach",
            drift_before={},
            target_weights=final_weights,
            trades_executed=trades,
            total_turnover=total_turnover,
            estimated_cost_bps=actual_slippage_bps,
            actual_slippage_bps=actual_slippage_bps,
            success=True,
        )
        
        with self._lock:
            self._rebalance_history.append(record)
            self._last_rebalance_time = timestamp
            self._last_weights = final_weights.copy()
        
        content = f"""**Rebalance executed successfully.**

| Metric | Value |
|--------|-------|
| Total Turnover | ${total_turnover:,.2f} |
| Slippage | {actual_slippage_bps:.1f} bps |
| Cumulative Slippage | {self._cumulative_slippage_bps:.1f} bps |
| Trades Executed | {len(trades)} |

**Trades:**
"""
        for trade in trades:
            action = "BUY" if trade.get('quantity', 0) > 0 else "SELL"
            content += f"- {action} ${abs(trade.get('quantity', 0)):,.2f} {trade.get('asset', 'UNKNOWN')}\n"
        
        event = PortfolioEvent(
            timestamp=timestamp,
            event_type=EventType.REBALANCE_EXECUTED,
            severity=SeverityLevel.INFO,
            message=f"Rebalance executed with {len(trades)} trades",
            details={
                "trades": trades,
                "turnover": total_turnover,
                "slippage_bps": actual_slippage_bps,
            },
            portfolio_value=portfolio_value,
            weights_before=self._last_weights,
            weights_after=final_weights,
        )
        
        self._log_event(event)
    
    def log_rebalance_failed(
        self,
        reason: str,
        portfolio_value: float,
        attempted_trades: list[dict],
        liquidity_issue: bool = False,
    ) -> None:
        """Log failed rebalance attempt."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        severity = SeverityLevel.CRITICAL if liquidity_issue else SeverityLevel.ERROR
        
        content = f"""**Rebalance FAILED.**

| Field | Value |
|-------|-------|
| Failure Reason | {reason} |
| Liquidity Issue | {liquidity_issue} |
| Attempted Trades | {len(attempted_trades)} |

**Action Required:** Manual intervention may be needed.
"""
        
        if liquidity_issue:
            content += "\n⚠️ **LIQUIDITY EVAPORATION DETECTED** ⚠️\n\n"
            content += "The rebalance could not be completed due to sudden liquidity evaporation. "
            content += "This event has been logged for post-mortem analysis.\n"
        
        event = PortfolioEvent(
            timestamp=timestamp,
            event_type=EventType.REBALANCE_FAILED,
            severity=severity,
            message=f"Rebalance failed: {reason}",
            details={
                "reason": reason,
                "liquidity_issue": liquidity_issue,
                "attempted_trades": attempted_trades,
            },
            portfolio_value=portfolio_value,
            weights_before=self._last_weights,
            weights_after=None,
        )
        
        self._log_event(event)
    
    def log_reserve_deployment(
        self,
        asset: str,
        amount: float,
        price_drop_pct: float,
        remaining_reserve: float,
        portfolio_value: float,
    ) -> None:
        """Log reserve deployment for opportunity capture."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        content = f"""**Reserve deployed to capture market opportunity.**

| Parameter | Value |
|-----------|-------|
| Asset | {asset} |
| Deployment Amount | ${amount:,.2f} |
| Price Drop Trigger | {price_drop_pct:.2%} |
| Remaining Reserve | ${remaining_reserve:,.2f} |

Opportunity captured during market dislocation.
"""
        
        event = PortfolioEvent(
            timestamp=timestamp,
            event_type=EventType.RESERVE_DEPLOYED,
            severity=SeverityLevel.INFO,
            message=f"Deployed ${amount:,.2f} to {asset} during {price_drop_pct:.2%} drop",
            details={
                "asset": asset,
                "amount": amount,
                "price_drop_pct": price_drop_pct,
                "remaining_reserve": remaining_reserve,
            },
            portfolio_value=portfolio_value,
            weights_before=self._last_weights,
            weights_after=None,
        )
        
        self._log_event(event)
    
    def _log_event(self, event: PortfolioEvent) -> None:
        """Internal method to log an event."""
        with self._lock:
            self._events.append(event)
        
        # Format event for markdown
        emoji = {
            SeverityLevel.INFO: "ℹ️",
            SeverityLevel.WARNING: "⚠️",
            SeverityLevel.ERROR: "❌",
            SeverityLevel.CRITICAL: "🚨",
        }.get(event.severity, "📝")
        
        content = f"""{emoji} **{event.message}**

**Event Type:** {event.event_type.value}  
**Severity:** {event.severity.value}  
**Portfolio Value:** ${event.portfolio_value:,.2f}
"""
        
        self._append_entry(event.event_type.value.upper(), content, event.to_dict())
    
    def get_summary_statistics(self) -> dict:
        """Get summary statistics from logged events."""
        with self._lock:
            n_events = len(self._events)
            n_rebalances = len(self._rebalance_history)
            n_drift_alerts = len(self._drift_alerts)
            
            successful_rebalances = sum(1 for r in self._rebalance_history if r.success)
            avg_slippage = (
                sum(r.actual_slippage_bps for r in self._rebalance_history) / n_rebalances
                if n_rebalances > 0 else 0
            )
        
        return {
            "total_events": n_events,
            "total_rebalances": n_rebalances,
            "successful_rebalances": successful_rebalances,
            "drift_alerts": n_drift_alerts,
            "average_slippage_bps": avg_slippage,
            "cumulative_slippage_bps": self._cumulative_slippage_bps,
            "last_rebalance": self._last_rebalance_time,
        }
    
    def export_events(self, output_path: str) -> None:
        """Export all events to JSON file."""
        with self._lock:
            events_data = [e.to_dict() for e in self._events]
        
        with open(output_path, 'w') as f:
            json.dump(events_data, f, indent=2)


def example_usage() -> None:
    """Demonstrate soul logging."""
    logger = PortfolioSoulLogger(portfolio_id="ZAID_TEST_001")
    
    # Simulate drift alert
    logger.log_allocation_drift(
        current_weights={"BTC": 0.35, "SOL": 0.20, "ETH": 0.25, "USDT": 0.20},
        target_weights={"BTC": 0.25, "SOL": 0.25, "ETH": 0.25, "USDT": 0.25},
        portfolio_value=100_000,
        max_drift_asset="BTC",
        max_drift_pct=0.10,
        threshold_pct=0.05,
    )
    
    # Simulate rebalance
    logger.log_rebalance_triggered(
        trigger_type="threshold_breach",
        current_weights={"BTC": 0.35, "SOL": 0.20, "ETH": 0.25, "USDT": 0.20},
        target_weights={"BTC": 0.25, "SOL": 0.25, "ETH": 0.25, "USDT": 0.25},
        portfolio_value=100_000,
        estimated_cost_bps=10.0,
    )
    
    # Simulate successful execution
    trades = [
        {"asset": "BTC", "quantity": -10_000, "action": "SELL"},
        {"asset": "SOL", "quantity": 5_000, "action": "BUY"},
        {"asset": "USDT", "quantity": 5_000, "action": "BUY"},
    ]
    logger.log_rebalance_executed(
        trades=trades,
        total_turnover=20_000,
        actual_slippage_bps=8.5,
        portfolio_value=100_000,
        final_weights={"BTC": 0.25, "SOL": 0.25, "ETH": 0.25, "USDT": 0.25},
    )
    
    # Get summary
    summary = logger.get_summary_statistics()
    print("Summary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_usage()
