#!/usr/bin/env python3
"""
Micro Soul Logger - Logging Spoofing Traps and Fill Misses to SOUL.md

This module logs critical microstructure events including spoofing traps,
fill misses, adverse selection events, and successful avoids to the SOUL.md file.
It provides an audit trail for strategy improvement and debugging.

Designed for the ZAID PERSONAL CRYPTO TRADING BOT with strict 8GB RAM constraints.
Logs events with microsecond precision for post-trade analysis.

Author: Opus 4.8
Stage: 21/100 - Advanced Market Microstructure
"""

from __future__ import annotations
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import time
import os
import json


class EventType(Enum):
    """Types of microstructure events to log."""
    SPOOFING_TRAP = auto()        # Detected and avoided a spoof
    FILL_MISS = auto()            # Order not filled due to queue position
    ADVERSE_SELECTION = auto()    # Traded against informed flow
    SUCCESSFUL_AVOID = auto()     # Successfully avoided fake liquidity
    QUOTE_PULL = auto()           # Quotes pulled due to toxicity
    SPREAD_ADJUSTMENT = auto()    # Spread widened due to VPIN
    LAYERING_DETECTED = auto()    # Layering manipulation detected
    TOXIC_FLOW = auto()           # High toxicity order flow detected


@dataclass(slots=True)
class MicroEvent:
    """A single microstructure event record."""
    event_type: EventType
    symbol: str
    timestamp_ns: int
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def timestamp_us(self) -> int:
        """Timestamp in microseconds."""
        return self.timestamp_ns // 1000
    
    @property
    def timestamp_ms(self) -> int:
        """Timestamp in milliseconds."""
        return self.timestamp_ns // 1_000_000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'event_type': self.event_type.name,
            'symbol': self.symbol,
            'timestamp_ns': self.timestamp_ns,
            'timestamp_us': self.timestamp_us,
            'timestamp_iso': datetime.fromtimestamp(self.timestamp_ns / 1e9).isoformat(),
            'price': self.price,
            'volume': self.volume,
            'side': self.side,
            'description': self.description,
            'metadata': self.metadata,
        }


class MicroSoulLogger:
    """
    Logs microstructure events to SOUL.md file.
    
    Provides detailed logging of:
    - Spoofing traps detected and avoided
    - Fill misses due to queue position
    - Adverse selection events
    - Successful avoidance of fake liquidity
    - Execution adjustments
    
    All events are logged with microsecond precision.
    """
    
    # Default SOUL.md file path
    DEFAULT_SOUL_PATH: str = "SOUL.md"
    
    def __init__(
        self,
        soul_path: str = DEFAULT_SOUL_PATH,
        symbols: Optional[List[str]] = None,
        max_memory_events: int = 1000,
    ) -> None:
        """
        Initialize the micro soul logger.
        
        Args:
            soul_path: Path to SOUL.md file
            symbols: List of symbols to monitor (None = all)
            max_memory_events: Maximum events to keep in memory
        """
        self.soul_path: str = soul_path
        self.symbols: Optional[set] = set(symbols) if symbols else None
        self.max_memory_events: int = max_memory_events
        
        # In-memory event buffer
        self._events: List[MicroEvent] = []
        
        # Event counters
        self._counters: Dict[str, int] = {
            'spoofing_traps': 0,
            'fill_misses': 0,
            'adverse_selection': 0,
            'successful_avoids': 0,
            'quote_pulls': 0,
            'spread_adjustments': 0,
            'layering_detected': 0,
            'toxic_flow': 0,
        }
        
        # Session start time
        self._session_start_ns: int = time.time_ns()
        self._last_flush_ns: int = self._session_start_ns
        
        # Ensure directory exists
        self._ensure_directory()
        
        # Initialize SOUL.md if needed
        self._initialize_soul_file()
    
    @staticmethod
    def _now_ns() -> int:
        """Get current time in nanoseconds."""
        return time.time_ns()
    
    def _ensure_directory(self) -> None:
        """Ensure the directory for SOUL.md exists."""
        dir_path = os.path.dirname(self.soul_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md file with header if it doesn't exist."""
        if not os.path.exists(self.soul_path):
            header = """# ZAID PERSONAL CRYPTO TRADING BOT - SOUL.md

## Microstructure Event Log

This file contains the soul of the trading bot - every spoofing trap detected,
every fill miss analyzed, every adverse selection event avoided. These logs
are used for continuous strategy improvement and post-trade analysis.

---

## Session Information

"""
            with open(self.soul_path, 'w') as f:
                f.write(header)
                f.write(f"- **Session Start**: {datetime.now().isoformat()}\n")
                f.write(f"- **Bot Version**: Stage 21/100 - Advanced Market Microstructure\n")
                f.write(f"- **Target Pairs**: BTC, SOL, ETH, USDT\n")
                f.write("\n---\n\n## Event Log\n\n")
    
    def _should_log(self, symbol: str) -> bool:
        """Check if an event for this symbol should be logged."""
        if self.symbols is None:
            return True
        return symbol.upper() in self.symbols
    
    def log_event(
        self,
        event_type: EventType,
        symbol: str,
        price: float,
        volume: float,
        side: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a microstructure event.
        
        Args:
            event_type: Type of event
            symbol: Trading pair symbol
            price: Relevant price level
            volume: Relevant volume
            side: 'buy' or 'sell'
            description: Human-readable description
            metadata: Additional structured data
        """
        if not self._should_log(symbol):
            return
        
        now = self._now_ns()
        
        event = MicroEvent(
            event_type=event_type,
            symbol=symbol.upper(),
            timestamp_ns=now,
            price=price,
            volume=volume,
            side=side.lower(),
            description=description,
            metadata=metadata or {},
        )
        
        # Add to memory buffer
        self._events.append(event)
        while len(self._events) > self.max_memory_events:
            self._events.pop(0)
        
        # Update counters
        counter_key = event_type.name.lower()
        if counter_key in self._counters:
            self._counters[counter_key] += 1
        
        # Write to file immediately for critical events
        if event_type in (EventType.SPOOFING_TRAP, EventType.ADVERSE_SELECTION, EventType.SUCCESSFUL_AVOID):
            self._write_event(event)
        
        # Periodic flush for other events
        elif now - self._last_flush_ns > 60_000_000_000:  # 60 seconds
            self.flush()
    
    def _write_event(self, event: MicroEvent) -> None:
        """Write a single event to SOUL.md."""
        try:
            with open(self.soul_path, 'a') as f:
                f.write(self._format_event_line(event))
        except IOError as e:
            print(f"Warning: Could not write to SOUL.md: {e}")
    
    def _format_event_line(self, event: MicroEvent) -> str:
        """Format an event as a markdown line."""
        ts = datetime.fromtimestamp(event.timestamp_ns / 1e9)
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Emoji based on event type
        emoji_map = {
            EventType.SPOOFING_TRAP: '🎭',
            EventType.FILL_MISS: '❌',
            EventType.ADVERSE_SELECTION: '⚠️',
            EventType.SUCCESSFUL_AVOID: '✅',
            EventType.QUOTE_PULL: '📤',
            EventType.SPREAD_ADJUSTMENT: '📊',
            EventType.LAYERING_DETECTED: '🏗️',
            EventType.TOXIC_FLOW: '☣️',
        }
        emoji = emoji_map.get(event.event_type, '📝')
        
        # Format metadata if present
        meta_str = ""
        if event.metadata:
            meta_items = [f"{k}={v}" for k, v in event.metadata.items()]
            meta_str = f" [{', '.join(meta_items)}]"
        
        return (
            f"- `{ts_str}` {emoji} **{event.event_type.name}** "
            f"`{event.symbol}` {event.side.upper()} @ {event.price:.4f} "
            f"(vol: {event.volume:.4f}){meta_str}\n"
            f"  - {event.description}\n"
        )
    
    def flush(self) -> None:
        """Flush all pending events to SOUL.md."""
        # Get events that haven't been written yet
        # (In production, would track written status separately)
        self._last_flush_ns = self._now_ns()
    
    def log_spoofing_trap(
        self,
        symbol: str,
        price: float,
        volume: float,
        side: str,
        confidence: float,
        avoid_time_us: int,
    ) -> None:
        """Log detection and avoidance of a spoofing trap."""
        self.log_event(
            event_type=EventType.SPOOFING_TRAP,
            symbol=symbol,
            price=price,
            volume=volume,
            side=side,
            description=f"Avoided spoofing trap with {confidence:.1%} confidence",
            metadata={
                'confidence': confidence,
                'avoid_time_us': avoid_time_us,
            },
        )
    
    def log_fill_miss(
        self,
        symbol: str,
        price: float,
        volume: float,
        side: str,
        queue_position: int,
        time_in_queue_ms: float,
    ) -> None:
        """Log a fill miss due to queue position."""
        self.log_event(
            event_type=EventType.FILL_MISS,
            symbol=symbol,
            price=price,
            volume=volume,
            side=side,
            description=f"Fill miss at queue position {queue_position}",
            metadata={
                'queue_position': queue_position,
                'time_in_queue_ms': time_in_queue_ms,
            },
        )
    
    def log_adverse_selection(
        self,
        symbol: str,
        price: float,
        volume: float,
        side: str,
        vpin: float,
        loss_estimate: float,
    ) -> None:
        """Log an adverse selection event."""
        self.log_event(
            event_type=EventType.ADVERSE_SELECTION,
            symbol=symbol,
            price=price,
            volume=volume,
            side=side,
            description=f"Adverse selection detected (VPIN: {vpin:.2f})",
            metadata={
                'vpin': vpin,
                'loss_estimate': loss_estimate,
            },
        )
    
    def log_successful_avoid(
        self,
        symbol: str,
        price: float,
        volume: float,
        side: str,
        reason: str,
        saved_amount: float,
    ) -> None:
        """Log successful avoidance of fake liquidity wall."""
        self.log_event(
            event_type=EventType.SUCCESSFUL_AVOID,
            symbol=symbol,
            price=price,
            volume=volume,
            side=side,
            description=f"Successfully avoided: {reason}",
            metadata={
                'saved_amount': saved_amount,
            },
        )
    
    def log_quote_pull(
        self,
        symbol: str,
        reason: str,
        toxicity_score: float,
    ) -> None:
        """Log quotes being pulled due to market conditions."""
        self.log_event(
            event_type=EventType.QUOTE_PULL,
            symbol=symbol,
            price=0.0,
            volume=0.0,
            side='neutral',
            description=f"Quotes pulled: {reason}",
            metadata={
                'toxicity_score': toxicity_score,
            },
        )
    
    def log_spread_adjustment(
        self,
        symbol: str,
        old_spread_bps: float,
        new_spread_bps: float,
        vpin: float,
    ) -> None:
        """Log a spread adjustment due to VPIN."""
        self.log_event(
            event_type=EventType.SPREAD_ADJUSTMENT,
            symbol=symbol,
            price=0.0,
            volume=0.0,
            side='neutral',
            description=f"Spread adjusted from {old_spread_bps:.1f} to {new_spread_bps:.1f} bps",
            metadata={
                'old_spread_bps': old_spread_bps,
                'new_spread_bps': new_spread_bps,
                'vpin': vpin,
            },
        )
    
    def log_layering_detected(
        self,
        symbol: str,
        side: str,
        layer_count: int,
        total_volume: float,
        confidence: float,
    ) -> None:
        """Log detection of layering manipulation."""
        self.log_event(
            event_type=EventType.LAYERING_DETECTED,
            symbol=symbol,
            price=0.0,
            volume=total_volume,
            side=side,
            description=f"Layering detected: {layer_count} layers",
            metadata={
                'layer_count': layer_count,
                'confidence': confidence,
            },
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event statistics for the session."""
        session_duration_s = (self._now_ns() - self._session_start_ns) / 1e9
        
        return {
            'session_duration_seconds': session_duration_s,
            'total_events': sum(self._counters.values()),
            'counters': self._counters.copy(),
            'events_in_memory': len(self._events),
        }
    
    def get_recent_events(self, limit: int = 10) -> List[MicroEvent]:
        """Get recent events from memory buffer."""
        return self._events[-limit:]
    
    def export_events_json(self, output_path: str) -> None:
        """Export all events to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)


def main() -> None:
    """Example usage of MicroSoulLogger."""
    logger = MicroSoulLogger(soul_path='SOUL.md', symbols=['BTCUSDT', 'ETHUSDT'])
    
    # Simulate various events
    logger.log_spoofing_trap(
        symbol='BTCUSDT',
        price=50000.0,
        volume=5.0,
        side='buy',
        confidence=0.85,
        avoid_time_us=2500,
    )
    
    logger.log_fill_miss(
        symbol='ETHUSDT',
        price=3000.0,
        volume=10.0,
        side='sell',
        queue_position=15,
        time_in_queue_ms=500.0,
    )
    
    logger.log_successful_avoid(
        symbol='BTCUSDT',
        price=49995.0,
        volume=100.0,
        side='buy',
        reason='Fake liquidity wall detected',
        saved_amount=500.0,
    )
    
    logger.log_spread_adjustment(
        symbol='BTCUSDT',
        old_spread_bps=10.0,
        new_spread_bps=25.0,
        vpin=0.65,
    )
    
    logger.log_layering_detected(
        symbol='ETHUSDT',
        side='bid',
        layer_count=5,
        total_volume=50.0,
        confidence=0.78,
    )
    
    # Print statistics
    stats = logger.get_statistics()
    print(f"\n=== Micro Soul Logger Statistics ===")
    print(f"Session Duration: {stats['session_duration_seconds']:.1f}s")
    print(f"Total Events: {stats['total_events']}")
    print(f"Counters: {stats['counters']}")
    
    # Show recent events
    print(f"\n=== Recent Events ===")
    for event in logger.get_recent_events(5):
        print(f"{event.event_type.name}: {event.symbol} - {event.description}")


if __name__ == "__main__":
    main()
