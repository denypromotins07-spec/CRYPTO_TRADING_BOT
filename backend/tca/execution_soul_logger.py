#!/usr/bin/env python3
"""
Execution SOUL Logger: Algorithmic Performance and IS Deviation Logging
Logs execution quality metrics and updates SOUL.md on TWAP scheduler failures.
Provides comprehensive audit trail for all execution events.

Stage 13: Advanced Execution Algorithms
Target: Minimize market impact to secure 8k-20k INR/hour
"""

from __future__ import annotations
import os
import time
import json
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import threading


class ExecutionEventType(Enum):
    """Types of execution events"""
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    IS_DEVIATION_ALERT = "is_deviation_alert"
    VWAP_DEVIATION = "vwap_deviation"
    TWAP_FAILURE = "twap_failure"
    ALPHA_DECAY_DETECTED = "alpha_decay_detected"
    MARKET_IMPACT_WARNING = "market_impact_warning"
    LIQUIDITY_SWEEP_DETECTED = "liquidity_sweep_detected"
    ICEBERG_PULL = "iceberg_pull"
    SOR_ROUTING_DECISION = "sor_routing_decision"


class SeverityLevel(Enum):
    """Event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ExecutionEvent:
    """Single execution event record"""
    event_id: str
    timestamp_ns: int
    timestamp_iso: str
    event_type: str
    severity: str
    symbol: str
    side: str
    quantity: float
    price: float
    details: Dict[str, Any]
    algorithm: str
    session_id: str
    is_deviation_event: bool
    deviation_bps: Optional[float]


@dataclass
class ExecutionSessionSummary:
    """Summary statistics for an execution session"""
    session_id: str
    symbol: str
    start_time_iso: str
    end_time_iso: str
    total_quantity: float
    filled_quantity: float
    avg_fill_price: float
    arrival_price: float
    total_is_bps: float
    vwap_deviation_bps: float
    twap_deviation_bps: float
    total_fees_bps: float
    num_orders: int
    num_slices: int
    alpha_decay_detected: bool
    alpha_decay_time_us: Optional[int]
    overall_rating: str  # "excellent", "good", "fair", "poor"


class ExecutionSoulLogger:
    """
    Comprehensive execution logger that tracks algorithmic performance.
    Logs IS deviations and updates SOUL.md on critical events.
    
    Features:
    - Real-time event logging with microsecond timestamps
    - Session-based aggregation
    - Automatic SOUL.md updates for critical events
    - Thread-safe concurrent logging
    - JSON-formatted structured logs
    """
    
    def __init__(
        self,
        log_directory: str = "./logs",
        soul_file_path: str = "./SOUL.md",
        enable_console_output: bool = True,
    ):
        self.log_directory = log_directory
        self.soul_file_path = soul_file_path
        self.enable_console_output = enable_console_output
        
        # Ensure directories exist
        os.makedirs(log_directory, exist_ok=True)
        
        # Event storage
        self._events: List[ExecutionEvent] = []
        self._sessions: Dict[str, Dict] = {}
        
        # Current session tracking
        self._current_session_id: Optional[str] = None
        self._current_session_events: List[ExecutionEvent] = []
        
        # Statistics
        self._total_events_logged = 0
        self._critical_events_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize SOUL.md if it doesn't exist
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md file with header if it doesn't exist"""
        if not os.path.exists(self.soul_file_path):
            with open(self.soul_file_path, 'w') as f:
                f.write("# ZAID Bot - Execution SOUL Log\n\n")
                f.write("## Overview\n")
                f.write("This document contains the Soul of the execution engine - ")
                f.write("every critical decision, every close call, every alpha decay moment.\n\n")
                f.write("---\n\n")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = time.time_ns()
        random_suffix = os.urandom(4).hex()
        return hashlib.sha256(f"{timestamp}{random_suffix}".encode()).hexdigest()[:16]
    
    def _get_current_session_id(self) -> str:
        """Get or create current session ID"""
        if self._current_session_id is None:
            self._current_session_id = f"session_{int(time.time())}"
            self._sessions[self._current_session_id] = {
                "start_time_ns": time.time_ns(),
                "start_time_iso": datetime.utcnow().isoformat() + "Z",
                "events": [],
                "symbols": set(),
            }
        return self._current_session_id
    
    def start_new_session(self, session_id: Optional[str] = None) -> str:
        """Start a new execution session"""
        with self._lock:
            if session_id is None:
                session_id = f"session_{int(time.time())}_{os.urandom(2).hex()}"
            
            self._current_session_id = session_id
            self._current_session_events = []
            self._sessions[session_id] = {
                "start_time_ns": time.time_ns(),
                "start_time_iso": datetime.utcnow().isoformat() + "Z",
                "events": [],
                "symbols": set(),
                "end_time_ns": None,
                "end_time_iso": None,
            }
            
            return session_id
    
    def log_event(
        self,
        event_type: ExecutionEventType,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        details: Optional[Dict[str, Any]] = None,
        severity: SeverityLevel = SeverityLevel.INFO,
        algorithm: str = "unknown",
        deviation_bps: Optional[float] = None,
    ) -> ExecutionEvent:
        """Log an execution event"""
        with self._lock:
            timestamp_ns = time.time_ns()
            timestamp_iso = datetime.utcnow().isoformat() + "Z"
            
            event = ExecutionEvent(
                event_id=self._generate_event_id(),
                timestamp_ns=timestamp_ns,
                timestamp_iso=timestamp_iso,
                event_type=event_type.value,
                severity=severity.value,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                details=details or {},
                algorithm=algorithm,
                session_id=self._get_current_session_id(),
                is_deviation_event=deviation_bps is not None and abs(deviation_bps) > 10,
                deviation_bps=deviation_bps,
            )
            
            # Store event
            self._events.append(event)
            self._current_session_events.append(event)
            self._total_events_logged += 1
            
            # Update session
            session_id = self._current_session_id
            if session_id in self._sessions:
                self._sessions[session_id]["events"].append(event)
                self._sessions[session_id]["symbols"].add(symbol)
            
            # Handle critical events
            if severity == SeverityLevel.CRITICAL:
                self._critical_events_count += 1
                self._handle_critical_event(event)
            
            # Console output if enabled
            if self.enable_console_output:
                self._print_event(event)
            
            return event
    
    def _print_event(self, event: ExecutionEvent) -> None:
        """Print event to console"""
        emoji = {
            SeverityLevel.INFO: "ℹ️",
            SeverityLevel.WARNING: "⚠️",
            SeverityLevel.ERROR: "❌",
            SeverityLevel.CRITICAL: "🚨",
        }.get(SeverityLevel(event.severity), "•")
        
        print(f"{emoji} [{event.timestamp_iso}] {event.event_type.upper()}")
        print(f"   Symbol: {event.symbol} | Side: {event.side} | Qty: {event.quantity}")
        if event.deviation_bps is not None:
            print(f"   Deviation: {event.deviation_bps:.2f} bps")
        if event.details:
            print(f"   Details: {json.dumps(event.details, default=str)[:100]}...")
        print()
    
    def _handle_critical_event(self, event: ExecutionEvent) -> None:
        """Handle critical events by updating SOUL.md"""
        # Append to SOUL.md
        with open(self.soul_file_path, 'a') as f:
            f.write(f"\n### 🚨 Critical Event: {event.event_type}\n\n")
            f.write(f"**Timestamp:** {event.timestamp_iso}\n\n")
            f.write(f"**Symbol:** {event.symbol}\n\n")
            f.write(f"**Side:** {event.side}\n\n")
            f.write(f"**Quantity:** {event.quantity}\n\n")
            f.write(f"**Price:** {event.price}\n\n")
            
            if event.deviation_bps is not None:
                f.write(f"**Deviation:** {event.deviation_bps:.2f} bps\n\n")
            
            if event.details:
                f.write("**Details:**\n```json\n")
                f.write(json.dumps(event.details, indent=2, default=str))
                f.write("\n```\n\n")
            
            f.write("---\n\n")
    
    def log_twap_failure(
        self,
        symbol: str,
        side: str,
        target_quantity: float,
        filled_quantity: float,
        failure_reason: str,
        dead_zone_detected: bool = False,
    ) -> ExecutionEvent:
        """Log TWAP scheduler failure - triggers SOUL.md update"""
        fill_rate = filled_quantity / target_quantity if target_quantity > 0 else 0
        
        event = self.log_event(
            event_type=ExecutionEventType.TWAP_FAILURE,
            symbol=symbol,
            side=side,
            quantity=target_quantity,
            price=0,
            details={
                "filled_quantity": filled_quantity,
                "fill_rate": fill_rate,
                "failure_reason": failure_reason,
                "dead_zone_detected": dead_zone_detected,
                "unfilled_quantity": target_quantity - filled_quantity,
            },
            severity=SeverityLevel.ERROR,
            algorithm="TWAP",
        )
        
        # Additional SOUL.md entry for TWAP failure
        with open(self.soul_file_path, 'a') as f:
            f.write(f"\n## TWAP Scheduler Failure\n\n")
            f.write(f"The TWAP scheduler failed to fill during a dead zone.\n\n")
            f.write(f"- **Symbol:** {symbol}\n")
            f.write(f"- **Target:** {target_quantity}\n")
            f.write(f"- **Filled:** {filled_quantity} ({fill_rate:.1%})\n")
            f.write(f"- **Reason:** {failure_reason}\n")
            f.write(f"- **Dead Zone:** {'Yes' if dead_zone_detected else 'No'}\n\n")
            f.write(f"*This event has been recorded in the Soul of the bot.*\n\n")
            f.write("---\n\n")
        
        return event
    
    def log_alpha_decay(
        self,
        symbol: str,
        decay_time_us: int,
        is_bps_at_decay: float,
        algorithm: str,
    ) -> ExecutionEvent:
        """Log alpha decay detection"""
        event = self.log_event(
            event_type=ExecutionEventType.ALPHA_DECAY_DETECTED,
            symbol=symbol,
            side="N/A",
            quantity=0,
            price=0,
            details={
                "decay_time_us": decay_time_us,
                "is_bps_at_decay": is_bps_at_decay,
            },
            severity=SeverityLevel.WARNING,
            algorithm=algorithm,
            deviation_bps=is_bps_at_decay,
        )
        
        return event
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Optional[ExecutionSessionSummary]:
        """Get summary statistics for a session"""
        with self._lock:
            session_id = session_id or self._current_session_id
            if session_id not in self._sessions:
                return None
            
            session = self._sessions[session_id]
            events = session["events"]
            
            # Calculate statistics
            fills = [e for e in events if e.event_type == ExecutionEventType.ORDER_FILLED.value]
            
            total_qty = sum(e.quantity for e in fills)
            avg_price = sum(e.price * e.quantity for e in fills) / total_qty if total_qty > 0 else 0
            
            # Get IS and deviation metrics
            is_events = [e for e in events if e.is_deviation_event]
            avg_is_bps = sum(e.deviation_bps for e in is_events) / len(is_events) if is_events else 0
            
            # Determine rating
            if abs(avg_is_bps) < 5:
                rating = "excellent"
            elif abs(avg_is_bps) < 15:
                rating = "good"
            elif abs(avg_is_bps) < 30:
                rating = "fair"
            else:
                rating = "poor"
            
            return ExecutionSessionSummary(
                session_id=session_id,
                symbol=list(session["symbols"])[0] if session["symbols"] else "N/A",
                start_time_iso=session["start_time_iso"],
                end_time_iso=session.get("end_time_iso") or datetime.utcnow().isoformat() + "Z",
                total_quantity=total_qty,
                filled_quantity=total_qty,
                avg_fill_price=avg_price,
                arrival_price=fills[0].price if fills else 0,
                total_is_bps=avg_is_bps,
                vwap_deviation_bps=avg_is_bps * 0.8,  # Simplified
                twap_deviation_bps=avg_is_bps * 0.9,  # Simplified
                total_fees_bps=0,  # Would track separately
                num_orders=len(events),
                num_slices=len(fills),
                alpha_decay_detected=any(
                    e.event_type == ExecutionEventType.ALPHA_DECAY_DETECTED.value 
                    for e in events
                ),
                alpha_decay_time_us=None,  # Would track from specific events
                overall_rating=rating,
            )
    
    def export_session_logs(self, session_id: Optional[str] = None) -> str:
        """Export session logs to JSON file"""
        with self._lock:
            session_id = session_id or self._current_session_id
            if session_id not in self._sessions:
                return ""
            
            session = self._sessions[session_id]
            events = session["events"]
            
            filename = os.path.join(
                self.log_directory,
                f"execution_log_{session_id}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump({
                    "session_id": session_id,
                    "summary": asdict(self.get_session_summary(session_id)) if self.get_session_summary(session_id) else None,
                    "events": [asdict(e) for e in events],
                }, f, indent=2, default=str)
            
            return filename
    
    def reset(self) -> None:
        """Reset logger state"""
        with self._lock:
            self._current_session_id = None
            self._current_session_events = []


if __name__ == "__main__":
    # Test execution SOUL logger
    logger = ExecutionSoulLogger(enable_console_output=True)
    
    # Start session
    session_id = logger.start_new_session()
    print(f"Started session: {session_id}\n")
    
    # Log various events
    logger.log_event(
        event_type=ExecutionEventType.ORDER_SUBMITTED,
        symbol="BTCUSDT",
        side="BUY",
        quantity=1.0,
        price=50000.0,
        details={"order_type": "limit", "time_in_force": "GTC"},
        severity=SeverityLevel.INFO,
        algorithm="TWAP",
    )
    
    logger.log_event(
        event_type=ExecutionEventType.ORDER_FILLED,
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.5,
        price=50010.0,
        details={"fill_id": "12345"},
        severity=SeverityLevel.INFO,
        algorithm="TWAP",
        deviation_bps=2.0,
    )
    
    # Simulate TWAP failure
    logger.log_twap_failure(
        symbol="ETHUSDT",
        side="SELL",
        target_quantity=100.0,
        filled_quantity=45.0,
        failure_reason="Dead zone detected - no liquidity during scheduled slice",
        dead_zone_detected=True,
    )
    
    # Log alpha decay
    logger.log_alpha_decay(
        symbol="SOLUSDT",
        decay_time_us=15000,
        is_bps_at_decay=25.5,
        algorithm="POV",
    )
    
    # Get session summary
    summary = logger.get_session_summary()
    if summary:
        print(f"\nSession Summary:")
        print(f"  Rating: {summary.overall_rating}")
        print(f"  Total IS: {summary.total_is_bps:.2f} bps")
        print(f"  Alpha Decay Detected: {summary.alpha_decay_detected}")
    
    print(f"\nCheck SOUL.md for critical event logs.")
