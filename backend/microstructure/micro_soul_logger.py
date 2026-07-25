#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
File: backend/microstructure/micro_soul_logger.py
Chapter 4: Microstructure Analytics, Execution Adjustments, and SOUL.md Logging

Purpose: Log spoofing traps, fill misses, and microstructure events to SOUL.md
Constraints: Must log with microsecond precision; append-only for durability
Target: AMD Ryzen AI 5 laptop with 8GB RAM limit

The SOUL.md file serves as the immutable audit trail for all microstructure
events that affected trading decisions, especially:
- Spoofing attacks detected and avoided
- Fill misses due to queue position
- Adverse selection events prevented
- Successful avoidance of fake liquidity walls

Design Patterns: Singleton for logger, Observer for event streaming
Type Hinting: Strict typing for production reliability
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json
import time
import os
import threading


class SoulEventType(Enum):
    """Types of events logged to SOUL.md"""
    SPOOFING_DETECTED = "spoofing_detected"
    SPOOFING_AVOIDED = "spoofing_avoided"
    FILL_MISS = "fill_miss"
    ADVERSE_SELECTION_PREVENTED = "adverse_selection_prevented"
    FAKE_LIQUIDITY_WALL_AVOIDED = "fake_liquidity_wall_avoided"
    QUEUE_POSITION_LOST = "queue_position_lost"
    EXECUTION_ADJUSTMENT = "execution_adjustment"
    TOXICITY_ALERT = "toxicity_alert"
    LAYERING_DETECTED = "layering_detected"
    MICROSTRUCTURE_ANOMALY = "microstructure_anomaly"


@dataclass(slots=True)
class SoulEvent:
    """An event to be logged to SOUL.md"""
    event_id: str
    timestamp_ns: int
    event_type: SoulEventType
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    symbol: str
    price: float
    quantity: float
    details: Dict[str, Any]
    pnl_impact_estimate: float  # Estimated PnL impact (positive = saved money)
    
    def to_log_line(self) -> str:
        """Convert to JSON log line"""
        return json.dumps({
            'event_id': self.event_id,
            'timestamp_ns': self.timestamp_ns,
            'timestamp_iso': datetime.utcfromtimestamp(self.timestamp_ns / 1e9).isoformat() + 'Z',
            'event_type': self.event_type.value,
            'severity': self.severity,
            'symbol': self.symbol,
            'price': self.price,
            'quantity': self.quantity,
            'details': self.details,
            'pnl_impact_estimate': self.pnl_impact_estimate,
        }, sort_keys=True)


class MicroSoulLogger:
    """
    Immutable logger for microstructure events.
    
    Appends events to SOUL.md in JSON Lines format for easy parsing
    and analysis. Each event includes microsecond-precise timestamps
    and estimated PnL impact.
    """
    
    # Default file path
    DEFAULT_SOUL_PATH: str = "SOUL.md"
    
    def __init__(self, soul_path: str = DEFAULT_SOUL_PATH):
        self.soul_path = soul_path
        self.event_counter: int = 0
        self.total_pnl_saved: float = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Event history (in-memory for quick access)
        self.recent_events: List[SoulEvent] = []
        self.max_in_memory: int = 1000
        
        # Statistics
        self.stats: Dict[SoulEventType, int] = {}
        
        # Ensure file exists
        self._initialize_file()
    
    def _initialize_file(self) -> None:
        """Initialize SOUL.md file with header if needed"""
        if not os.path.exists(self.soul_path):
            with open(self.soul_path, 'w') as f:
                f.write("# ZAID PERSONAL CRYPTO TRADING BOT - SOUL LOG\n")
                f.write("## Microstructure Event Audit Trail\n\n")
                f.write("This file contains an immutable record of all microstructure\n")
                f.write("events that affected trading decisions.\n\n")
                f.write("---\n\n")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        self.event_counter += 1
        timestamp = time.time_ns()
        return f"SOUL-{timestamp}-{self.event_counter:06d}"
    
    def log_event(self, event: SoulEvent) -> None:
        """
        Log an event to SOUL.md.
        
        This is the main method called when significant microstructure
        events occur.
        """
        with self._lock:
            # Write to file
            log_line = event.to_log_line()
            
            with open(self.soul_path, 'a') as f:
                f.write(log_line + '\n')
            
            # Update in-memory history
            if len(self.recent_events) >= self.max_in_memory:
                self.recent_events.pop(0)
            self.recent_events.append(event)
            
            # Update statistics
            event_key = event.event_type
            self.stats[event_key] = self.stats.get(event_key, 0) + 1
            
            # Track PnL saved
            if event.pnl_impact_estimate > 0:
                self.total_pnl_saved += event.pnl_impact_estimate
    
    def log_spoofing_detected(self, symbol: str, price: float, quantity: float,
                               confidence: float, order_ids: List[str]) -> None:
        """Log detection of a spoofing attempt"""
        event = SoulEvent(
            event_id=self._generate_event_id(),
            timestamp_ns=time.time_ns(),
            event_type=SoulEventType.SPOOFING_DETECTED,
            severity='WARNING' if confidence < 0.8 else 'CRITICAL',
            symbol=symbol,
            price=price,
            quantity=quantity,
            details={
                'confidence': confidence,
                'order_ids': order_ids,
                'action_taken': 'quotes_pulled',
            },
            pnl_impact_estimate=0.0  # Will be updated when we know the outcome
        )
        self.log_event(event)
    
    def log_spoofing_avoided(self, symbol: str, price: float, quantity: float,
                              estimated_loss_prevented: float) -> None:
        """Log successful avoidance of a spoofing trap"""
        event = SoulEvent(
            event_id=self._generate_event_id(),
            timestamp_ns=time.time_ns(),
            event_type=SoulEventType.SPOOFING_AVOIDED,
            severity='INFO',
            symbol=symbol,
            price=price,
            quantity=quantity,
            details={
                'outcome': 'successfully_avoided',
                'mechanism': 'spoofing_detector',
            },
            pnl_impact_estimate=estimated_loss_prevented
        )
        self.log_event(event)
    
    def log_fill_miss(self, symbol: str, price: float, quantity: float,
                      queue_position: int, quantity_ahead: int) -> None:
        """Log a missed fill due to queue position"""
        event = SoulEvent(
            event_id=self._generate_event_id(),
            timestamp_ns=time.time_ns(),
            event_type=SoulEventType.FILL_MISS,
            severity='INFO',
            symbol=symbol,
            price=price,
            quantity=quantity,
            details={
                'queue_position': queue_position,
                'quantity_ahead': quantity_ahead,
                'reason': 'queue_position_too_far_back',
            },
            pnl_impact_estimate=0.0  # Opportunity cost, not actual loss
        )
        self.log_event(event)
    
    def log_fake_liquidity_avoided(self, symbol: str, price: float, quantity: float,
                                    wall_size: float, estimated_impact_bps: float) -> None:
        """Log successful avoidance of a fake liquidity wall"""
        # Estimate PnL saved based on wall size and potential slippage
        pnl_saved = (wall_size * price) * (estimated_impact_bps / 10000)
        
        event = SoulEvent(
            event_id=self._generate_event_id(),
            timestamp_ns=time.time_ns(),
            event_type=SoulEventType.FAKE_LIQUIDITY_WALL_AVOIDED,
            severity='WARNING',
            symbol=symbol,
            price=price,
            quantity=quantity,
            details={
                'wall_size': wall_size,
                'estimated_impact_bps': estimated_impact_bps,
                'detection_method': 'fake_liquidity_filter',
            },
            pnl_impact_estimate=pnl_saved
        )
        self.log_event(event)
    
    def log_adverse_selection_prevented(self, symbol: str, price: float,
                                         toxicity_score: float, vpin: float) -> None:
        """Log prevention of adverse selection through toxicity detection"""
        event = SoulEvent(
            event_id=self._generate_event_id(),
            timestamp_ns=time.time_ns(),
            event_type=SoulEventType.ADVERSE_SELECTION_PREVENTED,
            severity='WARNING',
            symbol=symbol,
            price=price,
            quantity=0.0,
            details={
                'toxicity_score': toxicity_score,
                'vpin': vpin,
                'action_taken': 'spread_widened',
            },
            pnl_impact_estimate=0.0  # Prevented loss
        )
        self.log_event(event)
    
    def log_execution_adjustment(self, symbol: str, price: float,
                                  adjustment_type: str, reason: str) -> None:
        """Log an execution adjustment triggered by microstructure signals"""
        event = SoulEvent(
            event_id=self._generate_event_id(),
            timestamp_ns=time.time_ns(),
            event_type=SoulEventType.EXECUTION_ADJUSTMENT,
            severity='INFO',
            symbol=symbol,
            price=price,
            quantity=0.0,
            details={
                'adjustment_type': adjustment_type,
                'reason': reason,
            },
            pnl_impact_estimate=0.0
        )
        self.log_event(event)
    
    def get_statistics(self) -> dict:
        """Get logging statistics"""
        return {
            'total_events_logged': self.event_counter,
            'events_by_type': {k.value: v for k, v in self.stats.items()},
            'total_pnl_saved': self.total_pnl_saved,
            'recent_events_count': len(self.recent_events),
            'soul_file_path': self.soul_path,
        }
    
    def get_recent_events(self, count: int = 10) -> List[dict]:
        """Get recent events as dictionaries"""
        return [
            {
                'event_id': e.event_id,
                'event_type': e.event_type.value,
                'timestamp_ns': e.timestamp_ns,
                'symbol': e.symbol,
                'pnl_impact': e.pnl_impact_estimate,
            }
            for e in self.recent_events[-count:]
        ]
    
    def flush(self) -> None:
        """Ensure all pending writes are flushed to disk"""
        # File writes are immediate in our implementation, but this ensures
        # any OS-level buffering is flushed
        with open(self.soul_path, 'a') as f:
            f.flush()
            os.fsync(f.fileno())


# Global singleton instance
_soul_logger: Optional[MicroSoulLogger] = None
_logger_lock = threading.Lock()


def get_soul_logger(soul_path: str = MicroSoulLogger.DEFAULT_SOUL_PATH) -> MicroSoulLogger:
    """Get or create the global SOUL logger instance"""
    global _soul_logger
    
    with _logger_lock:
        if _soul_logger is None:
            _soul_logger = MicroSoulLogger(soul_path)
        return _soul_logger


def main():
    """Example usage of SOUL logger"""
    logger = MicroSoulLogger(soul_path="SOUL.md")
    
    print("=== ZAID CRYPTO TRADING BOT - SOUL LOGGER ===\n")
    
    # Simulate logging various events
    
    # 1. Spoofing detected
    logger.log_spoofing_detected(
        symbol="BTC/USDT",
        price=50000.0,
        quantity=100.0,
        confidence=0.85,
        order_ids=["SPOOF-001", "SPOOF-002"]
    )
    print("✓ Logged spoofing detection")
    
    # 2. Fake liquidity wall avoided
    logger.log_fake_liquidity_avoided(
        symbol="BTC/USDT",
        price=50000.0,
        quantity=50.0,
        wall_size=5000.0,
        estimated_impact_bps=25.0
    )
    print("✓ Logged fake liquidity avoidance")
    
    # 3. Fill miss
    logger.log_fill_miss(
        symbol="ETH/USDT",
        price=3000.0,
        quantity=10.0,
        queue_position=150,
        quantity_ahead=50000
    )
    print("✓ Logged fill miss")
    
    # 4. Adverse selection prevented
    logger.log_adverse_selection_prevented(
        symbol="SOL/USDT",
        price=100.0,
        toxicity_score=0.75,
        vpin=0.65
    )
    print("✓ Logged adverse selection prevention")
    
    # Show statistics
    stats = logger.get_statistics()
    print(f"\n=== STATISTICS ===")
    print(f"Total Events: {stats['total_events_logged']}")
    print(f"Total PnL Saved: ${stats['total_pnl_saved']:.2f}")
    print(f"Events by Type: {stats['events_by_type']}")
    
    # Show recent events
    recent = logger.get_recent_events(5)
    print(f"\n=== RECENT EVENTS ===")
    for evt in recent:
        print(f"  [{evt['event_type']}] {evt['symbol']} - PnL Impact: ${evt['pnl_impact']:.2f}")
    
    print(f"\n✓ SOUL.md file created/updated at: {stats['soul_file_path']}")


if __name__ == "__main__":
    main()
