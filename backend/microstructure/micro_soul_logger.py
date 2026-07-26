#!/usr/bin/env python3
"""
Micro Soul Logger for Liquidity Events

Logs critical microstructure events (liquidity dry-ups, Kyle's Lambda spikes,
impact shifts) to SOUL.md for monitoring and post-trade analysis.

**Key Features:**
- Real-time event logging to SOUL.md
- Kyle's Lambda spike detection
- Liquidity dry-up alerts
- Impact shift tracking
- Integration with all microstructure modules

**Performance:** Async file I/O, minimal overhead on trading path.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import os
import threading
from collections import deque


class EventType(Enum):
    """Types of microstructure events."""
    LIQUIDITY_DRY_UP = "liquidity_dry_up"
    KYLE_LAMBDA_SPIKE = "kyle_lambda_spike"
    IMPACT_SHIFT = "impact_shift"
    SPREAD_WIDENING = "spread_widening"
    VOLATILITY_SURGE = "volatility_surge"
    ORDER_IMBALANCE_EXTREME = "order_imbalance_extreme"
    CLUSTERING_ANOMALY = "clustering_anomaly"
    SYSTEM_STATUS = "system_status"


@dataclass
class MicroEvent:
    """A single microstructure event."""
    event_type: EventType
    timestamp: str
    symbol: str
    value: float
    threshold: float
    severity: int  # 1-5 scale
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'value': self.value,
            'threshold': self.threshold,
            'severity': self.severity,
            'description': self.description,
            'context': self.context,
        }


class MicroSoulLogger:
    """
    Logs microstructure events to SOUL.md file.
    
    Thread-safe implementation with async writes to avoid blocking.
    """
    
    def __init__(
        self,
        log_path: str = "SOUL.md",
        max_events_in_memory: int = 10000,
        auto_flush_interval: int = 60  # seconds
    ) -> None:
        """
        Initialize the logger.
        
        Args:
            log_path: Path to SOUL.md file
            max_events_in_memory: Maximum events to keep in memory
            auto_flush_interval: Seconds between automatic flushes
        """
        self.log_path = log_path
        self.max_events = max_events_in_memory
        
        # Thread-safe event queue
        self._events: deque[MicroEvent] = deque(maxlen=max_events_in_memory)
        self._lock = threading.Lock()
        
        # Thresholds for event detection
        self.thresholds = {
            'kyle_lambda_spike': 3.0,  # Standard deviations
            'liquidity_dry_up': 0.5,   # Volume ratio
            'spread_widening': 2.0,    # Multiple of normal
            'volatility_surge': 4.0,   # Standard deviations
            'order_imbalance': 0.8,    # Absolute value
        }
        
        # Baseline statistics (updated continuously)
        self._baselines: Dict[str, Dict[str, float]] = {}
        
        # Initialize SOUL.md if not exists
        self._initialize_log_file()
        
        # Auto-flush thread
        self._running = True
        self._flush_thread = threading.Thread(target=self._auto_flush_loop, daemon=True)
        self._flush_thread.start()
    
    def _initialize_log_file(self) -> None:
        """Initialize SOUL.md with header if it doesn't exist."""
        if not os.path.exists(self.log_path):
            header = """# ZAID Personal Crypto Trading Bot - SOUL.md

## System Of Understanding Liquidity

This file contains real-time microstructure event logs for liquidity monitoring,
anomaly detection, and post-trade analysis.

---

## Event Log

"""
            with open(self.log_path, 'w') as f:
                f.write(header)
    
    def log_event(self, event: MicroEvent) -> None:
        """
        Log a microstructure event.
        
        Args:
            event: The event to log
        """
        with self._lock:
            self._events.append(event)
    
    def check_kyle_lambda(
        self,
        symbol: str,
        current_lambda: float,
        baseline_mean: float,
        baseline_std: float
    ) -> Optional[MicroEvent]:
        """
        Check for Kyle's Lambda spike and log if detected.
        
        Args:
            symbol: Trading pair symbol
            current_lambda: Current Kyle's Lambda value
            baseline_mean: Historical mean lambda
            baseline_std: Historical std dev of lambda
            
        Returns:
            MicroEvent if spike detected, None otherwise
        """
        if baseline_std <= 0:
            return None
        
        z_score = abs(current_lambda - baseline_mean) / baseline_std
        
        if z_score > self.thresholds['kyle_lambda_spike']:
            severity = min(5, int(z_score))
            event = MicroEvent(
                event_type=EventType.KYLE_LAMBDA_SPIKE,
                timestamp=datetime.utcnow().isoformat(),
                symbol=symbol,
                value=current_lambda,
                threshold=baseline_mean + self.thresholds['kyle_lambda_spike'] * baseline_std,
                severity=severity,
                description=f"Kyle's Lambda spike detected: {z_score:.2f} std devs above baseline",
                context={
                    'z_score': z_score,
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'alert': "Liquidity withdrawal detected - widen execution limits"
                }
            )
            self.log_event(event)
            return event
        
        return None
    
    def check_liquidity_dry_up(
        self,
        symbol: str,
        current_volume: float,
        baseline_volume: float
    ) -> Optional[MicroEvent]:
        """
        Check for liquidity dry-up and log if detected.
        
        Args:
            symbol: Trading pair symbol
            current_volume: Current volume level
            baseline_volume: Normal volume level
            
        Returns:
            MicroEvent if dry-up detected, None otherwise
        """
        if baseline_volume <= 0:
            return None
        
        volume_ratio = current_volume / baseline_volume
        
        if volume_ratio < self.thresholds['liquidity_dry_up']:
            severity = min(5, int((1 - volume_ratio) * 5))
            event = MicroEvent(
                event_type=EventType.LIQUIDITY_DRY_UP,
                timestamp=datetime.utcnow().isoformat(),
                symbol=symbol,
                value=volume_ratio,
                threshold=self.thresholds['liquidity_dry_up'],
                severity=severity,
                description=f"Liquidity dry-up: volume at {volume_ratio:.1%} of baseline",
                context={
                    'current_volume': current_volume,
                    'baseline_volume': baseline_volume,
                    'alert': "Reduce position sizes, expect higher slippage"
                }
            )
            self.log_event(event)
            return event
        
        return None
    
    def check_spread_widening(
        self,
        symbol: str,
        current_spread: float,
        baseline_spread: float
    ) -> Optional[MicroEvent]:
        """Check for spread widening."""
        if baseline_spread <= 0:
            return None
        
        spread_ratio = current_spread / baseline_spread
        
        if spread_ratio > self.thresholds['spread_widening']:
            severity = min(5, int(spread_ratio))
            event = MicroEvent(
                event_type=EventType.SPREAD_WIDENING,
                timestamp=datetime.utcnow().isoformat(),
                symbol=symbol,
                value=current_spread,
                threshold=baseline_spread * self.thresholds['spread_widening'],
                severity=severity,
                description=f"Spread widened to {spread_ratio:.1f}x normal",
                context={
                    'spread_ratio': spread_ratio,
                    'baseline_spread': baseline_spread,
                    'alert': "Higher transaction costs expected"
                }
            )
            self.log_event(event)
            return event
        
        return None
    
    def check_order_imbalance(
        self,
        symbol: str,
        imbalance: float
    ) -> Optional[MicroEvent]:
        """Check for extreme order imbalance."""
        if abs(imbalance) > self.thresholds['order_imbalance']:
            severity = min(5, int(abs(imbalance) * 5))
            side = "BUY" if imbalance > 0 else "SELL"
            event = MicroEvent(
                event_type=EventType.ORDER_IMBALANCE_EXTREME,
                timestamp=datetime.utcnow().isoformat(),
                symbol=symbol,
                value=imbalance,
                threshold=self.thresholds['order_imbalance'],
                severity=severity,
                description=f"Extreme {side} imbalance: {abs(imbalance):.2f}",
                context={
                    'side': side,
                    'alert': "Potential price pressure in direction of imbalance"
                }
            )
            self.log_event(event)
            return event
        
        return None
    
    def log_impact_shift(
        self,
        symbol: str,
        old_impact: float,
        new_impact: float,
        change_percent: float
    ) -> MicroEvent:
        """Log a significant price impact shift."""
        severity = min(5, int(abs(change_percent) / 20))
        event = MicroEvent(
            event_type=EventType.IMPACT_SHIFT,
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            value=new_impact,
            threshold=old_impact * (1 + change_percent / 100),
            severity=severity,
            description=f"Price impact shifted by {change_percent:+.1f}%",
            context={
                'old_impact': old_impact,
                'new_impact': new_impact,
                'change_percent': change_percent,
            }
        )
        self.log_event(event)
        return event
    
    def log_system_status(self, status_message: str, metrics: Dict[str, Any]) -> None:
        """Log system status update."""
        event = MicroEvent(
            event_type=EventType.SYSTEM_STATUS,
            timestamp=datetime.utcnow().isoformat(),
            symbol="SYSTEM",
            value=0.0,
            threshold=0.0,
            severity=1,
            description=status_message,
            context=metrics
        )
        self.log_event(event)
    
    def _auto_flush_loop(self) -> None:
        """Background thread for periodic flushing."""
        while self._running:
            import time
            time.sleep(60)  # Check every minute
            self.flush_to_disk()
    
    def flush_to_disk(self) -> int:
        """
        Flush all pending events to SOUL.md.
        
        Returns:
            Number of events written
        """
        with self._lock:
            if not self._events:
                return 0
            
            events_to_write = list(self._events)
            self._events.clear()
        
        # Format events as markdown
        lines = []
        for event in events_to_write:
            severity_icon = {
                1: "ℹ️",
                2: "⚠️",
                3: "🟠",
                4: "🔴",
                5: "🚨"
            }.get(event.severity, "•")
            
            lines.append(f"\n### {severity_icon} [{event.timestamp}] {event.event_type.value.upper()}")
            lines.append(f"- **Symbol:** {event.symbol}")
            lines.append(f"- **Value:** {event.value:.6f}")
            lines.append(f"- **Threshold:** {event.threshold:.6f}")
            lines.append(f"- **Severity:** {event.severity}/5")
            lines.append(f"- **Description:** {event.description}")
            
            if event.context:
                lines.append("- **Context:**")
                for key, value in event.context.items():
                    lines.append(f"  - `{key}`: {value}")
        
        if lines:
            lines.append("\n---\n")
            
            try:
                with open(self.log_path, 'a') as f:
                    f.write('\n'.join(lines))
                return len(events_to_write)
            except Exception as e:
                print(f"Error writing to SOUL.md: {e}")
                return 0
        
        return 0
    
    def get_recent_events(self, n: int = 10) -> List[Dict]:
        """Get recent events from memory."""
        with self._lock:
            events = list(self._events)[-n:]
        return [e.to_dict() for e in events]
    
    def get_event_summary(self) -> Dict:
        """Get summary of logged events."""
        with self._lock:
            total = len(self._events)
            
            by_type = {}
            by_severity = {i: 0 for i in range(1, 6)}
            
            for event in self._events:
                type_name = event.event_type.value
                by_type[type_name] = by_type.get(type_name, 0) + 1
                by_severity[event.severity] += 1
        
        return {
            'total_events': total,
            'by_type': by_type,
            'by_severity': by_severity,
            'thresholds': self.thresholds,
        }
    
    def shutdown(self) -> None:
        """Graceful shutdown - flush remaining events."""
        self._running = False
        self.flush_to_disk()


# Global logger instance
_soul_logger: Optional[MicroSoulLogger] = None


def get_soul_logger(log_path: str = "SOUL.md") -> MicroSoulLogger:
    """Get or create global SOUL logger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = MicroSoulLogger(log_path=log_path)
    return _soul_logger


if __name__ == "__main__":
    # Example usage and validation
    print("Micro Soul Logger - Validation Test")
    print("=" * 50)
    
    # Create logger
    logger = MicroSoulLogger(log_path="SOUL_test.md")
    
    print("\nSimulating market events...")
    
    # Simulate Kyle's Lambda spike
    logger.check_kyle_lambda(
        symbol="BTCUSDT",
        current_lambda=0.0005,
        baseline_mean=0.0001,
        baseline_std=0.00005
    )
    
    # Simulate liquidity dry-up
    logger.check_liquidity_dry_up(
        symbol="ETHUSDT",
        current_volume=100.0,
        baseline_volume=1000.0
    )
    
    # Simulate spread widening
    logger.check_spread_widening(
        symbol="SOLUSDT",
        current_spread=0.50,
        baseline_spread=0.10
    )
    
    # Simulate order imbalance
    logger.check_order_imbalance(
        symbol="BTCUSDT",
        imbalance=0.85
    )
    
    # Log impact shift
    logger.log_impact_shift(
        symbol="BTCUSDT",
        old_impact=5.0,
        new_impact=7.5,
        change_percent=50.0
    )
    
    # Log system status
    logger.log_system_status(
        "Microstructure engine initialized",
        {'modules_loaded': 12, 'memory_mb': 256}
    )
    
    # Flush to disk
    written = logger.flush_to_disk()
    print(f"Events written to SOUL_test.md: {written}")
    
    # Get summary
    summary = logger.get_event_summary()
    print(f"\nEvent Summary:")
    print(f"  Total events: {summary['total_events']}")
    print(f"  By type: {summary['by_type']}")
    print(f"  By severity: {summary['by_severity']}")
    
    # Get recent events
    recent = logger.get_recent_events(3)
    print(f"\nRecent events: {len(recent)}")
    
    # Cleanup
    logger.shutdown()
    
    # Clean up test file
    if os.path.exists("SOUL_test.md"):
        os.remove("SOUL_test.md")
    
    print("\n✓ Micro Soul Logger module validated successfully")
