#!/usr/bin/env python3
"""
ZAID Personal Crypto Trading Bot - SMC Soul Logger Module
Chapter 4: Institutional Order Flow, Premium/Discount Pricing, and SOUL.md SMC Logging

This module logs successful stop hunt fades and FVG fills to SOUL.md.
Tracks breaker block mitigations and other high-probability SMC events.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.

Design Patterns: Observer Pattern for event logging
Time Complexity: O(1) for log writes
Space Complexity: O(k) for in-memory event cache

Strict type hinting enforced for production reliability.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime
import json
import os


class SMCEventType(Enum):
    """Classification of SMC events to log."""
    STOP_HUNT_FADE = auto()      # Successful stop hunt reversal
    FVG_FILL = auto()            # Fair Value Gap filled
    BREAKER_MITIGATION = auto()  # Breaker block successfully mitigated
    ORDER_BLOCK_BOUNCE = auto()  # Price bounced from order block
    LIQUIDITY_SWEEP = auto()     # Liquidity pool swept
    BOS_CONFIRMED = auto()       # Break of Structure confirmed
    CHoCH_DETECTED = auto()      # Change of Character detected
    PREMIUM_REJECTION = auto()   # Price rejected from Premium zone
    DISCOUNT_SUPPORT = auto()    # Price supported at Discount zone


@dataclass(slots=True)
class SMCEvent:
    """
    Represents a logged SMC event.
    Uses __slots__ for memory efficiency on 8GB RAM systems.
    """
    event_id: int
    event_type: SMCEventType
    timestamp: int
    price: float
    asset: str
    timeframe: str
    confidence: float
    outcome: str  # 'success', 'partial', 'failure'
    profit_loss_pips: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat() if self.timestamp > 0 else '',
            'price': self.price,
            'asset': self.asset,
            'timeframe': self.timeframe,
            'confidence': self.confidence,
            'outcome': self.outcome,
            'profit_loss_pips': self.profit_loss_pips,
            'notes': self.notes
        }


@dataclass
class SMCStatistics:
    """Aggregate statistics for SMC events."""
    total_events: int = 0
    successful_events: int = 0
    partial_events: int = 0
    failed_events: int = 0
    total_profit_loss_pips: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_events == 0:
            return 0.0
        return (self.successful_events / self.total_events) * 100.0
    
    @property
    def average_profit_loss(self) -> float:
        """Calculate average profit/loss per event."""
        if self.total_events == 0:
            return 0.0
        return self.total_profit_loss_pips / self.total_events
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'total_events': self.total_events,
            'successful_events': self.successful_events,
            'partial_events': self.partial_events,
            'failed_events': self.failed_events,
            'success_rate_pct': round(self.success_rate, 2),
            'total_profit_loss_pips': round(self.total_profit_loss_pips, 2),
            'average_profit_loss_pips': round(self.average_profit_loss, 2)
        }


class SMCSoulLogger:
    """
    Advanced SMC event logger for tracking institutional patterns.
    
    Logs to SOUL.md file with structured format including:
    - Stop hunt fade successes
    - FVG fill confirmations
    - Breaker block mitigations
    - Performance statistics
    
    Memory-safe implementation with bounded in-memory cache.
    """
    
    SOUL_MD_FILE = "SOUL.md"
    
    def __init__(self, soul_md_path: Optional[str] = None, max_cache_size: int = 1000):
        """
        Initialize SMC soul logger.
        
        Args:
            soul_md_path: Path to SOUL.md file (default: current directory)
            max_cache_size: Maximum events to keep in memory
        """
        self._soul_md_path = soul_md_path or self.SOUL_MD_FILE
        self._max_cache_size = max_cache_size
        
        # In-memory event cache (bounded)
        self._event_cache: List[SMCEvent] = []
        self._next_event_id: int = 1
        
        # Statistics by event type
        self._stats_by_type: Dict[SMCEventType, SMCStatistics] = {
            event_type: SMCStatistics() for event_type in SMCEventType
        }
        
        # Overall statistics
        self._overall_stats = SMCStatistics()
        
        # Ensure SOUL.md exists with header
        self._initialize_soul_md()
    
    def _initialize_soul_md(self) -> None:
        """Initialize SOUL.md file with header if it doesn't exist."""
        if not os.path.exists(self._soul_md_path):
            header = """# ZAID Bot - SMC Soul Log

## Institutional Pattern Tracking & Performance

This file logs successful Smart Money Concepts (SMC) events including:
- Stop Hunt Fades
- Fair Value Gap (FVG) Fills
- Breaker Block Mitigations
- Order Block Bounces
- Liquidity Sweeps
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Premium/Discount Zone Rejections

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Events | 0 |
| Success Rate | 0.00% |
| Total P&L (pips) | 0.00 |

---

## Event Log

"""
            with open(self._soul_md_path, 'w', encoding='utf-8') as f:
                f.write(header)
    
    def log_event(
        self,
        event_type: SMCEventType,
        price: float,
        asset: str = "BTC/USDT",
        timeframe: str = "4h",
        confidence: float = 0.7,
        outcome: str = "success",
        profit_loss_pips: float = 0.0,
        notes: str = ""
    ) -> int:
        """
        Log a new SMC event.
        
        Args:
            event_type: Type of SMC event
            price: Price level where event occurred
            asset: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "4h")
            confidence: Confidence score 0.0 to 1.0
            outcome: 'success', 'partial', or 'failure'
            profit_loss_pips: Profit/loss in pips
            notes: Additional notes about the event
        
        Returns:
            Event ID
        """
        import time
        timestamp = int(time.time())
        
        event = SMCEvent(
            event_id=self._next_event_id,
            event_type=event_type,
            timestamp=timestamp,
            price=price,
            asset=asset,
            timeframe=timeframe,
            confidence=min(1.0, max(0.0, confidence)),
            outcome=outcome,
            profit_loss_pips=profit_loss_pips,
            notes=notes
        )
        
        # Add to cache
        self._event_cache.append(event)
        if len(self._event_cache) > self._max_cache_size:
            self._event_cache.pop(0)
        
        # Update statistics
        self._update_statistics(event)
        
        # Write to SOUL.md
        self._write_to_soul_md(event)
        
        event_id = event.event_id
        self._next_event_id += 1
        
        return event_id
    
    def _update_statistics(self, event: SMCEvent) -> None:
        """Update statistics based on new event."""
        # Update overall stats
        self._overall_stats.total_events += 1
        
        if event.outcome == 'success':
            self._overall_stats.successful_events += 1
        elif event.outcome == 'partial':
            self._overall_stats.partial_events += 1
        else:
            self._overall_stats.failed_events += 1
        
        self._overall_stats.total_profit_loss_pips += event.profit_loss_pips
        
        # Update type-specific stats
        type_stats = self._stats_by_type.get(event.event_type)
        if type_stats:
            type_stats.total_events += 1
            
            if event.outcome == 'success':
                type_stats.successful_events += 1
            elif event.outcome == 'partial':
                type_stats.partial_events += 1
            else:
                type_stats.failed_events += 1
            
            type_stats.total_profit_loss_pips += event.profit_loss_pips
    
    def _write_to_soul_md(self, event: SMCEvent) -> None:
        """Append event to SOUL.md file."""
        event_line = self._format_event_line(event)
        
        with open(self._soul_md_path, 'a', encoding='utf-8') as f:
            f.write(event_line + "\n")
        
        # Periodically update summary statistics
        if self._overall_stats.total_events % 10 == 0:
            self._update_summary_in_file()
    
    def _format_event_line(self, event: SMCEvent) -> str:
        """Format event as markdown table row."""
        outcome_emoji = {
            'success': '✅',
            'partial': '⚠️',
            'failure': '❌'
        }.get(event.outcome, '❓')
        
        dt_str = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        pl_str = f"{event.profit_loss_pips:+.2f}" if event.profit_loss_pips != 0 else "-"
        
        return (
            f"| {event.event_id} | {dt_str} | {event.event_type.name} | "
            f"{event.asset} | {event.timeframe} | {event.price:.2f} | "
            f"{outcome_emoji} | {event.confidence:.2f} | {pl_str} | "
            f"{event.notes[:50]}... |" if len(event.notes) > 50 else f"{event.notes} |"
        )
    
    def _update_summary_in_file(self) -> None:
        """Update summary statistics section in SOUL.md."""
        try:
            # Read entire file
            with open(self._soul_md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find and replace summary section
            summary_table = f"""| Metric | Value |
|--------|-------|
| Total Events | {self._overall_stats.total_events} |
| Success Rate | {self._overall_stats.success_rate:.2f}% |
| Total P&L (pips) | {self._overall_stats.total_profit_loss_pips:.2f} |"""
            
            # Simple approach: rebuild file with updated summary
            lines = content.split('\n')
            in_summary = False
            new_lines = []
            
            for line in lines:
                if '| Metric | Value |' in line:
                    in_summary = True
                    new_lines.append(summary_table)
                elif in_summary and line.startswith('|---'):
                    in_summary = False
                elif not in_summary:
                    new_lines.append(line)
            
            with open(self._soul_md_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
                
        except Exception:
            # Silently fail on summary update - don't lose event data
            pass
    
    def log_stop_huntFade(
        self,
        price: float,
        sweep_high: float,
        sweep_low: float,
        asset: str = "BTC/USDT",
        timeframe: str = "4h",
        confidence: float = 0.8,
        outcome: str = "success",
        profit_loss_pips: float = 0.0
    ) -> int:
        """Convenience method for logging stop hunt fade."""
        notes = f"Sweep: {sweep_low:.2f}-{sweep_high:.2f}"
        return self.log_event(
            event_type=SMCEventType.STOP_HUNT_FADE,
            price=price,
            asset=asset,
            timeframe=timeframe,
            confidence=confidence,
            outcome=outcome,
            profit_loss_pips=profit_loss_pips,
            notes=notes
        )
    
    def log_fvg_fill(
        self,
        price: float,
        fvg_start: float,
        fvg_end: float,
        asset: str = "BTC/USDT",
        timeframe: str = "4h",
        confidence: float = 0.7,
        outcome: str = "success",
        profit_loss_pips: float = 0.0
    ) -> int:
        """Convenience method for logging FVG fill."""
        notes = f"FVG: {fvg_start:.2f}-{fvg_end:.2f}"
        return self.log_event(
            event_type=SMCEventType.FVG_FILL,
            price=price,
            asset=asset,
            timeframe=timeframe,
            confidence=confidence,
            outcome=outcome,
            profit_loss_pips=profit_loss_pips,
            notes=notes
        )
    
    def log_breaker_mitigation(
        self,
        price: float,
        breaker_level: float,
        original_ob_level: float,
        asset: str = "BTC/USDT",
        timeframe: str = "4h",
        confidence: float = 0.75,
        outcome: str = "success",
        profit_loss_pips: float = 0.0
    ) -> int:
        """Convenience method for logging breaker block mitigation."""
        notes = f"Breaker: {breaker_level:.2f}, Original OB: {original_ob_level:.2f}"
        return self.log_event(
            event_type=SMCEventType.BREAKER_MITIGATION,
            price=price,
            asset=asset,
            timeframe=timeframe,
            confidence=confidence,
            outcome=outcome,
            profit_loss_pips=profit_loss_pips,
            notes=notes
        )
    
    def get_statistics(self, event_type: Optional[SMCEventType] = None) -> SMCStatistics:
        """Get statistics overall or for specific event type."""
        if event_type is None:
            return self._overall_stats
        return self._stats_by_type.get(event_type, SMCStatistics())
    
    def get_recent_events(self, count: int = 10) -> List[SMCEvent]:
        """Get most recent events from cache."""
        return self._event_cache[-count:]
    
    def get_events_by_type(self, event_type: SMCEventType) -> List[SMCEvent]:
        """Get all cached events of specific type."""
        return [e for e in self._event_cache if e.event_type == event_type]
    
    def export_events_json(self, filepath: str) -> None:
        """Export all cached events to JSON file."""
        events_data = [e.to_dict() for e in self._event_cache]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'total_events': len(events_data),
                'statistics': self._overall_stats.to_dict(),
                'events': events_data
            }, f, indent=2)


if __name__ == "__main__":
    # Example usage and basic validation
    print("ZAID Bot - SMC Soul Logger Module")
    print("=" * 50)
    
    logger = SMCSoulLogger(soul_md_path="SOUL.md")
    
    # Log some example events
    print("\nLogging example SMC events...")
    
    id1 = logger.log_stop_huntFade(
        price=98500.0,
        sweep_high=99000.0,
        sweep_low=97500.0,
        asset="BTC/USDT",
        timeframe="4h",
        confidence=0.85,
        outcome="success",
        profit_loss_pips=150.0
    )
    print(f"Logged stop hunt fade: Event ID {id1}")
    
    id2 = logger.log_fvg_fill(
        price=3450.0,
        fvg_start=3420.0,
        fvg_end=3440.0,
        asset="ETH/USDT",
        timeframe="4h",
        confidence=0.75,
        outcome="success",
        profit_loss_pips=80.0
    )
    print(f"Logged FVG fill: Event ID {id2}")
    
    id3 = logger.log_breaker_mitigation(
        price=195.0,
        breaker_level=194.5,
        original_ob_level=192.0,
        asset="SOL/USDT",
        timeframe="4h",
        confidence=0.80,
        outcome="success",
        profit_loss_pips=45.0
    )
    print(f"Logged breaker mitigation: Event ID {id3}")
    
    # Display statistics
    stats = logger.get_statistics()
    print(f"\n=== Overall Statistics ===")
    print(f"Total Events: {stats.total_events}")
    print(f"Success Rate: {stats.success_rate:.2f}%")
    print(f"Total P&L: {stats.total_profit_loss_pips:.2f} pips")
    
    print(f"\nSOUL.md file created/updated at: {logger._soul_md_path}")
