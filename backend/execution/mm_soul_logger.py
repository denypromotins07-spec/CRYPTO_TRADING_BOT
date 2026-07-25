"""
MM Soul Logger
Logs adverse selection events and spread capture rates to SOUL.md.
Tracks the "soul" of the market maker - its survival instincts and profitability.

This module implements:
- Adverse selection event logging
- Spread capture rate tracking
- Toxic flow avoidance documentation
- Performance metrics for market making quality

Target: Document every close call and successful spread capture in SOUL.md.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from datetime import datetime, timezone
from pathlib import Path


class EventType(Enum):
    """Types of events to log."""
    SPREAD_CAPTURE = "spread_capture"
    ADVERSE_SELECTION_AVOIDED = "adverse_selection_avoided"
    TOXIC_FLOW_DETECTED = "toxic_flow_detected"
    WHALE_ENCOUNTER = "whale_encounter"
    INVENTORY_RISK_HIGH = "inventory_risk_high"
    QUOTE_REFRESH = "quote_refresh"
    PNL_SPIKE = "pnl_spike"
    MARKET_STRESS = "market_stress"


@dataclass
class SoulEvent:
    """Represents a logged soul event."""
    timestamp: str
    event_type: EventType
    symbol: str
    description: str
    details: Dict
    pnl_impact: float = 0.0
    severity: str = "normal"  # low, normal, high, critical
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "symbol": self.symbol,
            "description": self.description,
            "details": self.details,
            "pnl_impact": self.pnl_impact,
            "severity": self.severity,
        }


@dataclass
class SpreadCaptureMetrics:
    """Metrics for spread capture performance."""
    total_captures: int = 0
    total_volume: float = 0.0
    avg_spread_bps: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    best_capture_bps: float = 0.0
    worst_capture_bps: float = 0.0


@dataclass
class AdverseSelectionMetrics:
    """Metrics for adverse selection avoidance."""
    toxic_events_detected: int = 0
    whales_avoided: int = 0
    losses_prevented: float = 0.0
    quote_widening_count: int = 0
    inventory_liquidations: int = 0


class MMSoulLogger:
    """
    The Soul Logger - documents the market maker's journey.
    
    Records every significant event that affects the bot's
    survival and profitability, creating a living document
    of trading wisdom.
    """
    
    def __init__(self, soul_file_path: str = "SOUL.md"):
        self.soul_file_path = Path(soul_file_path)
        
        # Event storage
        self.events: List[SoulEvent] = []
        
        # Metrics tracking
        self.spread_metrics: Dict[str, SpreadCaptureMetrics] = {}
        self.adverse_metrics: Dict[str, AdverseSelectionMetrics] = {}
        
        # Session statistics
        self.session_start = datetime.now(timezone.utc)
        self.total_pnl = 0.0
        self.trades_executed = 0
        
        # Initialize file if needed
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize or append to the SOUL.md file."""
        if not self.soul_file_path.exists():
            self._write_header()
    
    def _write_header(self) -> None:
        """Write the header for SOUL.md."""
        header = f"""# ZAID Personal Crypto Trading Bot - SOUL.md

## The Soul of the Market Maker

*Created: {datetime.now(timezone.utc).isoformat()}*

This document chronicles the journey of our market making bot - 
its victories against toxic flow, its narrow escapes from adverse selection,
and its relentless pursuit of the bid-ask spread.

---

## Philosophy

> "The market maker's soul is forged in the fire of a thousand spreads,
> tempered by the hammer of adverse selection, and polished by the grindstone
> of a million microseconds."

### Core Principles

1. **Survival First**: Avoid toxic flow at all costs
2. **Spread Capture**: Every basis point earned is a victory
3. **Inventory Discipline**: Never hold what you can't hedge
4. **Speed**: Latency is life

---

## Session Summary

| Metric | Value |
|--------|-------|
| Session Start | {self.session_start.isoformat()} |
| Total Events | 0 |
| Total P&L | $0.00 |
| Trades Executed | 0 |

---

## Event Log

"""
        with open(self.soul_file_path, 'w') as f:
            f.write(header)
    
    def log_event(
        self,
        event_type: EventType,
        symbol: str,
        description: str,
        details: Optional[Dict] = None,
        pnl_impact: float = 0.0,
        severity: str = "normal"
    ) -> None:
        """Log a new soul event."""
        event = SoulEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            symbol=symbol,
            description=description,
            details=details or {},
            pnl_impact=pnl_impact,
            severity=severity,
        )
        
        self.events.append(event)
        self.total_pnl += pnl_impact
        
        # Update metrics based on event type
        self._update_metrics(event)
        
        # Append to file
        self._append_to_soul_file(event)
        
        # Update session summary
        self._update_session_summary()
    
    def log_spread_capture(
        self,
        symbol: str,
        spread_bps: float,
        volume: float,
        pnl: float,
        is_round_trip: bool = False
    ) -> None:
        """Log a successful spread capture."""
        self.log_event(
            event_type=EventType.SPREAD_CAPTURE,
            symbol=symbol,
            description=f"Captured {spread_bps:.1f} bps spread on {volume:.4f} {symbol}",
            details={
                "spread_bps": spread_bps,
                "volume": volume,
                "is_round_trip": is_round_trip,
            },
            pnl_impact=pnl,
            severity="low" if pnl > 0 else "normal"
        )
    
    def log_adverse_selection_avoided(
        self,
        symbol: str,
        estimated_loss_prevented: float,
        detection_reason: str
    ) -> None:
        """Log when we successfully avoided adverse selection."""
        self.log_event(
            event_type=EventType.ADVERSE_SELECTION_AVOIDED,
            symbol=symbol,
            description=f"Avoided toxic flow: {detection_reason}",
            details={
                "estimated_loss_prevented": estimated_loss_prevented,
                "detection_reason": detection_reason,
            },
            pnl_impact=estimated_loss_prevented,  # Saved loss = virtual gain
            severity="high"
        )
    
    def log_toxic_flow_detected(
        self,
        symbol: str,
        vpin_level: float,
        action_taken: str
    ) -> None:
        """Log detection of toxic order flow."""
        self.log_event(
            event_type=EventType.TOXIC_FLOW_DETECTED,
            symbol=symbol,
            description=f"Toxic flow detected (VPIN: {vpin_level:.2f}), {action_taken}",
            details={
                "vpin_level": vpin_level,
                "action_taken": action_taken,
            },
            severity="high" if vpin_level > 0.7 else "normal"
        )
    
    def log_whale_encounter(
        self,
        symbol: str,
        whale_volume: float,
        price_impact_bps: float,
        outcome: str
    ) -> None:
        """Log an encounter with a whale trader."""
        self.log_event(
            event_type=EventType.WHALE_ENCOUNTER,
            symbol=symbol,
            description=f"Whale spotted: {whale_volume:.2f} {symbol}, impact: {price_impact_bps:.1f} bps",
            details={
                "whale_volume": whale_volume,
                "price_impact_bps": price_impact_bps,
                "outcome": outcome,
            },
            severity="critical" if abs(price_impact_bps) > 50 else "high"
        )
    
    def log_inventory_risk(
        self,
        symbol: str,
        inventory_value: float,
        risk_level: str,
        action_taken: str
    ) -> None:
        """Log high inventory risk situation."""
        self.log_event(
            event_type=EventType.INVENTORY_RISK_HIGH,
            symbol=symbol,
            description=f"High inventory risk ({risk_level}): ${inventory_value:,.2f}, {action_taken}",
            details={
                "inventory_value": inventory_value,
                "risk_level": risk_level,
                "action_taken": action_taken,
            },
            severity=risk_level.lower()
        )
    
    def _update_metrics(self, event: SoulEvent) -> None:
        """Update running metrics based on event."""
        symbol = event.symbol
        
        # Initialize symbol metrics if needed
        if symbol not in self.spread_metrics:
            self.spread_metrics[symbol] = SpreadCaptureMetrics()
        if symbol not in self.adverse_metrics:
            self.adverse_metrics[symbol] = AdverseSelectionMetrics()
        
        # Update spread metrics
        if event.event_type == EventType.SPREAD_CAPTURE:
            metrics = self.spread_metrics[symbol]
            metrics.total_captures += 1
            metrics.total_volume += event.details.get('volume', 0.0)
            
            spread = event.details.get('spread_bps', 0.0)
            if metrics.total_captures == 1:
                metrics.avg_spread_bps = spread
                metrics.best_capture_bps = spread
                metrics.worst_capture_bps = spread
            else:
                metrics.avg_spread_bps = (
                    (metrics.avg_spread_bps * (metrics.total_captures - 1) + spread) 
                    / metrics.total_captures
                )
                metrics.best_capture_bps = max(metrics.best_capture_bps, spread)
                metrics.worst_capture_bps = min(metrics.worst_capture_bps, spread)
            
            metrics.total_pnl += event.pnl_impact
            if event.pnl_impact > 0:
                metrics.win_rate = (
                    (metrics.win_rate * (metrics.total_captures - 1) + 1.0) 
                    / metrics.total_captures
                )
            else:
                metrics.win_rate = (
                    metrics.win_rate * (metrics.total_captures - 1) 
                    / metrics.total_captures
                )
        
        # Update adverse selection metrics
        elif event.event_type == EventType.ADVERSE_SELECTION_AVOIDED:
            self.adverse_metrics[symbol].toxic_events_detected += 1
            self.adverse_metrics[symbol].losses_prevented += event.pnl_impact
        
        elif event.event_type == EventType.WHALE_ENCOUNTER:
            self.adverse_metrics[symbol].whales_avoided += 1
        
        elif event.event_type == EventType.TOXIC_FLOW_DETECTED:
            self.adverse_metrics[symbol].quote_widening_count += 1
    
    def _append_to_soul_file(self, event: SoulEvent) -> None:
        """Append a new event to the SOUL.md file."""
        severity_icon = {
            "low": "🟢",
            "normal": "🔵",
            "high": "🟠",
            "critical": "🔴",
        }.get(event.severity, "⚪")
        
        entry = f"""
### {severity_icon} [{event.timestamp}] {event.event_type.value.upper()}

**Symbol:** `{event.symbol}`  
**Description:** {event.description}  
**P&L Impact:** ${event.pnl_impact:+,.2f}  
**Severity:** {event.severity}

**Details:**
```json
{json.dumps(event.details, indent=2)}
```

---

"""
        with open(self.soul_file_path, 'a') as f:
            f.write(entry)
    
    def _update_session_summary(self) -> None:
        """Update the session summary section in SOUL.md."""
        # Read current content
        with open(self.soul_file_path, 'r') as f:
            content = f.read()
        
        # Build new summary
        summary = f"""## Session Summary

| Metric | Value |
|--------|-------|
| Session Start | {self.session_start.isoformat()} |
| Total Events | {len(self.events)} |
| Total P&L | ${self.total_pnl:+,.2f} |
| Trades Executed | {self.trades_executed} |

"""
        # Find and replace the summary section
        start_marker = "## Session Summary"
        end_marker = "---\n\n## Event Log"
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            new_content = content[:start_idx] + summary + content[end_idx:]
            with open(self.soul_file_path, 'w') as f:
                f.write(new_content)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive soul statistics."""
        return {
            "session_duration_hours": (
                datetime.now(timezone.utc) - self.session_start
            ).total_seconds() / 3600,
            "total_events": len(self.events),
            "total_pnl": self.total_pnl,
            "trades_executed": self.trades_executed,
            "events_by_type": {
                et.value: sum(1 for e in self.events if e.event_type == et)
                for et in EventType
            },
            "spread_metrics": {
                s: {
                    "total_captures": m.total_captures,
                    "avg_spread_bps": m.avg_spread_bps,
                    "total_pnl": m.total_pnl,
                    "win_rate": m.win_rate,
                }
                for s, m in self.spread_metrics.items()
            },
            "adverse_metrics": {
                s: {
                    "toxic_events": m.toxic_events_detected,
                    "whales_avoided": m.whales_avoided,
                    "losses_prevented": m.losses_prevented,
                }
                for s, m in self.adverse_metrics.items()
            },
        }
    
    def finalize_session(self) -> str:
        """Finalize the current session and return summary."""
        stats = self.get_statistics()
        
        footer = f"""
---

## Session Finale

*Session ended: {datetime.now(timezone.utc).isoformat()}*

### Final Statistics

| Category | Metric | Value |
|----------|--------|-------|
| Duration | Hours | {stats['session_duration_hours']:.2f} |
| Events | Total | {stats['total_events']} |
| P&L | USD | ${stats['total_pnl']:,.2f} |
| Efficiency | P&L/Event | ${stats['total_pnl']/max(stats['total_events'], 1):,.2f} |

### Wisdom Gained

> Every event logged here represents a lesson learned,
> a danger avoided, or a profit earned. This is the soul
> of our market maker - may it grow wiser with each trade.

---

*End of Session Log*
"""
        with open(self.soul_file_path, 'a') as f:
            f.write(footer)
        
        return f"Session finalized. Total P&L: ${stats['total_pnl']:,.2f}"


# Example usage and testing
if __name__ == "__main__":
    logger = MMSoulLogger("SOUL.md")
    
    # Simulate some events
    logger.log_spread_capture(
        symbol="BTC",
        spread_bps=5.2,
        volume=0.1,
        pnl=26.0,
        is_round_trip=True
    )
    
    logger.log_adverse_selection_avoided(
        symbol="BTC",
        estimated_loss_prevented=150.0,
        detection_reason="VPIN spike to 0.85 detected informed selling"
    )
    
    logger.log_toxic_flow_detected(
        symbol="ETH",
        vpin_level=0.72,
        action_taken="Widened spreads by 3x, reduced quote sizes by 50%"
    )
    
    logger.log_whale_encounter(
        symbol="BTC",
        whale_volume=50.0,
        price_impact_bps=75.0,
        outcome="Successfully widened quotes before whale sold, avoided being run over"
    )
    
    logger.log_inventory_risk(
        symbol="SOL",
        inventory_value=25000.0,
        risk_level="HIGH",
        action_taken="Initiated aggressive liquidation via market orders"
    )
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"Soul Statistics:")
    print(f"  Total Events: {stats['total_events']}")
    print(f"  Total P&L: ${stats['total_pnl']:,.2f}")
    print(f"  Events by Type: {stats['events_by_type']}")
    
    # Finalize
    result = logger.finalize_session()
    print(f"\n{result}")
    print(f"\nCheck SOUL.md for the complete log!")
