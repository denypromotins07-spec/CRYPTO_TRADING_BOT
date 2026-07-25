#!/usr/bin/env python3
"""
Arb Soul Logger - Recording Missed Arbs and Slippage Costs to SOUL.md

This module logs all arbitrage activity including:
- Successful captures
- Missed opportunities  
- Slippage costs
- Funding rate spikes captured

All data is written to SOUL.md for post-trade analysis and continuous improvement.

Chapter 4: Arbitrage Risk Management, Legging Risk, and SOUL.md Arb Logging
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Tuple
import time
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArbType(Enum):
    """Types of arbitrage"""
    BASIS = "basis"
    TRIANGULAR = "triangular"
    FUNDING = "funding"
    STATISTICAL = "statistical"
    CROSS_MARGIN = "cross_margin"


class ArbOutcome(Enum):
    """Outcome of arbitrage attempt"""
    SUCCESS = "success"
    MISSED = "missed"
    PARTIAL = "partial"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ArbEvent:
    """Represents an arbitrage event"""
    event_id: str
    arb_type: ArbType
    outcome: ArbOutcome
    timestamp: datetime
    symbol: str
    expected_profit_bps: Decimal
    actual_profit_bps: Optional[Decimal]
    slippage_bps: Optional[Decimal]
    capital_deployed: Decimal
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'arb_type': self.arb_type.value,
            'outcome': self.outcome.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'expected_profit_bps': float(self.expected_profit_bps),
            'actual_profit_bps': float(self.actual_profit_bps) if self.actual_profit_bps else None,
            'slippage_bps': float(self.slippage_bps) if self.slippage_bps else None,
            'capital_deployed': float(self.capital_deployed),
            'reason': self.reason,
            'metadata': self.metadata
        }


@dataclass
class FundingSpikeCapture:
    """Records a funding rate spike capture"""
    capture_id: str
    symbol: str
    funding_rate: Decimal
    predicted_rate: Decimal
    payment_received: Decimal
    annualized_rate: Decimal
    timestamp: datetime
    position_side: str  # 'long' or 'short'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'capture_id': self.capture_id,
            'symbol': self.symbol,
            'funding_rate': float(self.funding_rate),
            'predicted_rate': float(self.predicted_rate),
            'payment_received': float(self.payment_received),
            'annualized_rate': float(self.annualized_rate),
            'timestamp': self.timestamp.isoformat(),
            'position_side': self.position_side
        }


class ArbSoulLogger:
    """
    Comprehensive logger for arbitrage activity.
    
    Writes detailed logs to SOUL.md including:
    - All arb attempts (successful and missed)
    - Slippage analysis
    - Funding spike captures
    - Performance metrics
    
    The SOUL.md file serves as the bot's "memory" for learning and optimization.
    """
    
    # Default log file path
    DEFAULT_SOUL_PATH = "SOUL.md"
    
    # Maximum events to keep in memory
    MAX_MEMORY_EVENTS = 1000
    
    # Flush interval (seconds)
    FLUSH_INTERVAL_SECONDS = 60
    
    def __init__(self, soul_path: str = DEFAULT_SOUL_PATH):
        """
        Initialize arb soul logger.
        
        Args:
            soul_path: Path to SOUL.md file
        """
        self.soul_path = soul_path
        
        # Event storage
        self.events: List[ArbEvent] = []
        self.funding_captures: List[FundingSpikeCapture] = []
        
        # Counters
        self._event_counter = 0
        self._capture_counter = 0
        
        # Aggregated statistics
        self.stats = {
            'total_attempts': 0,
            'successful_captures': 0,
            'missed_opportunities': 0,
            'failed_attempts': 0,
            'total_slippage_bps': Decimal('0'),
            'total_profit_bps': Decimal('0'),
            'total_capital_deployed': Decimal('0'),
        }
        
        # Last flush time
        self._last_flush = datetime.now(timezone.utc)
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
        
        logger.info(f"ArbSoulLogger initialized (path={soul_path})")
    
    def _initialize_soul_file(self) -> None:
        """Initialize or verify SOUL.md file"""
        if not os.path.exists(self.soul_path):
            with open(self.soul_path, 'w') as f:
                f.write("# ZAID Personal Crypto Trading Bot - SOUL.md\n\n")
                f.write("## Arbitrage Activity Log\n\n")
                f.write("*This file contains the complete history of arbitrage operations,\n")
                f.write("including successful captures, missed opportunities, and lessons learned.*\n\n")
                f.write("---\n\n")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        self._event_counter += 1
        timestamp_ns = time.time_ns()
        return f"ARB-{timestamp_ns}-{self._event_counter:06d}"
    
    def _generate_capture_id(self) -> str:
        """Generate unique capture ID"""
        self._capture_counter += 1
        timestamp_ns = time.time_ns()
        return f"FUND-{timestamp_ns}-{self._capture_counter:06d}"
    
    def log_arb_attempt(
        self,
        arb_type: ArbType,
        outcome: ArbOutcome,
        symbol: str,
        expected_profit_bps: Decimal,
        capital_deployed: Decimal,
        actual_profit_bps: Optional[Decimal] = None,
        slippage_bps: Optional[Decimal] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an arbitrage attempt.
        
        Args:
            arb_type: Type of arbitrage
            outcome: Outcome of the attempt
            symbol: Trading symbol
            expected_profit_bps: Expected profit in basis points
            capital_deployed: Capital deployed
            actual_profit_bps: Actual profit (if completed)
            slippage_bps: Slippage experienced
            reason: Reason for outcome (especially for misses/failures)
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        event_id = self._generate_event_id()
        
        event = ArbEvent(
            event_id=event_id,
            arb_type=arb_type,
            outcome=outcome,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            expected_profit_bps=expected_profit_bps,
            actual_profit_bps=actual_profit_bps,
            slippage_bps=slippage_bps,
            capital_deployed=capital_deployed,
            reason=reason,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Update statistics
        self.stats['total_attempts'] += 1
        
        if outcome == ArbOutcome.SUCCESS:
            self.stats['successful_captures'] += 1
            if actual_profit_bps:
                self.stats['total_profit_bps'] += actual_profit_bps
        elif outcome == ArbOutcome.MISSED:
            self.stats['missed_opportunities'] += 1
        elif outcome in (ArbOutcome.FAILED, ArbOutcome.ABORTED):
            self.stats['failed_attempts'] += 1
        
        if slippage_bps:
            self.stats['total_slippage_bps'] += abs(slippage_bps)
        
        self.stats['total_capital_deployed'] += capital_deployed
        
        # Trim memory if needed
        if len(self.events) > self.MAX_MEMORY_EVENTS:
            self.events = self.events[-self.MAX_MEMORY_EVENTS:]
        
        # Check if flush needed
        self._check_flush()
        
        logger.info(
            f"Arb logged: {event_id}, type={arb_type.value}, "
            f"outcome={outcome.value}, symbol={symbol}, profit={expected_profit_bps} bps"
        )
        
        return event_id
    
    def log_missed_arb(
        self,
        arb_type: ArbType,
        symbol: str,
        expected_profit_bps: Decimal,
        reason: str,
        capital_required: Decimal,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a missed arbitrage opportunity.
        
        This is critical for identifying systemic issues and improving execution.
        
        Args:
            arb_type: Type of arbitrage missed
            symbol: Trading symbol
            expected_profit_bps: Expected profit that was missed
            reason: Why the arb was missed
            capital_required: Capital that would have been deployed
            metadata: Additional context
            
        Returns:
            Event ID
        """
        return self.log_arb_attempt(
            arb_type=arb_type,
            outcome=ArbOutcome.MISSED,
            symbol=symbol,
            expected_profit_bps=expected_profit_bps,
            capital_deployed=Decimal('0'),  # Nothing deployed
            reason=reason,
            metadata=metadata
        )
    
    def log_funding_spike_capture(
        self,
        symbol: str,
        funding_rate: Decimal,
        predicted_rate: Decimal,
        payment_received: Decimal,
        position_side: str,
        annualized_rate: Optional[Decimal] = None
    ) -> str:
        """
        Log a funding rate spike capture.
        
        Args:
            symbol: Trading symbol
            funding_rate: Actual funding rate
            predicted_rate: Predicted rate
            payment_received: Payment received in USDT
            position_side: Position side ('long' or 'short')
            annualized_rate: Annualized rate
            
        Returns:
            Capture ID
        """
        capture_id = self._generate_capture_id()
        
        if annualized_rate is None:
            # Calculate annualized: (1 + rate)^1095 - 1
            annualized_rate = ((Decimal('1') + funding_rate) ** Decimal('1095')) - Decimal('1')
        
        capture = FundingSpikeCapture(
            capture_id=capture_id,
            symbol=symbol,
            funding_rate=funding_rate,
            predicted_rate=predicted_rate,
            payment_received=payment_received,
            annualized_rate=annualized_rate,
            timestamp=datetime.now(timezone.utc),
            position_side=position_side
        )
        
        self.funding_captures.append(capture)
        
        # Trim if needed
        if len(self.funding_captures) > self.MAX_MEMORY_EVENTS:
            self.funding_captures = self.funding_captures[-self.MAX_MEMORY_EVENTS:]
        
        self._check_flush()
        
        logger.info(
            f"Funding spike captured: {capture_id}, symbol={symbol}, "
            f"rate={funding_rate}, payment={payment_received:.2f} USDT"
        )
        
        return capture_id
    
    def _check_flush(self) -> None:
        """Check if SOUL.md needs to be flushed"""
        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_flush).total_seconds()
        
        if elapsed >= self.FLUSH_INTERVAL_SECONDS:
            asyncio.create_task(self.flush_to_soul())
    
    async def flush_to_soul(self) -> bool:
        """
        Flush all pending events to SOUL.md.
        
        Returns:
            True if successful
        """
        try:
            self._last_flush = datetime.now(timezone.utc)
            
            with open(self.soul_path, 'a') as f:
                # Write summary section
                f.write(f"\n## Update: {self._last_flush.isoformat()}\n\n")
                
                # Write recent events
                recent_events = self.events[-10:]  # Last 10 events
                if recent_events:
                    f.write("### Recent Arbitrage Events\n\n")
                    f.write("| Time | Type | Symbol | Outcome | Expected (bps) | Actual (bps) | Reason |\n")
                    f.write("|------|------|--------|---------|----------------|--------------|--------|\n")
                    
                    for event in recent_events:
                        actual_str = f"{event.actual_profit_bps:.2f}" if event.actual_profit_bps else "-"
                        slippage_str = f"{event.slippage_bps:.2f}" if event.slippage_bps else "-"
                        
                        f.write(
                            f"| {event.timestamp.strftime('%H:%M:%S')} | "
                            f"{event.arb_type.value} | {event.symbol} | "
                            f"{event.outcome.value} | {event.expected_profit_bps:.2f} | "
                            f"{actual_str} | {event.reason or '-'} |\n"
                        )
                    
                    f.write("\n")
                
                # Write funding captures
                recent_captures = self.funding_captures[-5:]  # Last 5 captures
                if recent_captures:
                    f.write("### Recent Funding Spike Captures\n\n")
                    f.write("| Time | Symbol | Rate | Payment (USDT) | Side |\n")
                    f.write("|------|--------|------|----------------|------|\n")
                    
                    for capture in recent_captures:
                        f.write(
                            f"| {capture.timestamp.strftime('%H:%M:%S')} | "
                            f"{capture.symbol} | {capture.funding_rate:.6f} | "
                            f"{capture.payment_received:.2f} | {capture.position_side} |\n"
                        )
                    
                    f.write("\n")
                
                # Write statistics
                f.write("### Cumulative Statistics\n\n")
                f.write(f"- **Total Attempts**: {self.stats['total_attempts']}\n")
                f.write(f"- **Successful Captures**: {self.stats['successful_captures']}\n")
                f.write(f"- **Missed Opportunities**: {self.stats['missed_opportunities']}\n")
                f.write(f"- **Failed Attempts**: {self.stats['failed_attempts']}\n")
                
                success_rate = (
                    self.stats['successful_captures'] / max(1, self.stats['total_attempts']) * 100
                )
                f.write(f"- **Success Rate**: {success_rate:.1f}%\n")
                
                avg_profit = (
                    self.stats['total_profit_bps'] / max(1, self.stats['successful_captures'])
                )
                f.write(f"- **Average Profit (bps)**: {avg_profit:.2f}\n")
                
                avg_slippage = (
                    self.stats['total_slippage_bps'] / max(1, self.stats['total_attempts'])
                )
                f.write(f"- **Average Slippage (bps)**: {avg_slippage:.2f}\n")
                
                f.write(f"- **Total Capital Deployed**: {self.stats['total_capital_deployed']:.2f} USDT\n")
                
                # Write missed opportunity analysis
                missed_by_reason = {}
                for event in self.events:
                    if event.outcome == ArbOutcome.MISSED and event.reason:
                        if event.reason not in missed_by_reason:
                            missed_by_reason[event.reason] = 0
                        missed_by_reason[event.reason] += 1
                
                if missed_by_reason:
                    f.write("\n### Missed Opportunity Analysis\n\n")
                    f.write("| Reason | Count | Total Expected Profit (bps) |\n")
                    f.write("|--------|-------|----------------------------|\n")
                    
                    for reason, count in sorted(missed_by_reason.items(), key=lambda x: -x[1]):
                        total_expected = sum(
                            e.expected_profit_bps for e in self.events
                            if e.outcome == ArbOutcome.MISSED and e.reason == reason
                        )
                        f.write(f"| {reason} | {count} | {total_expected:.2f} |\n")
                
                f.write("\n---\n")
            
            logger.debug(f"Flushed {len(self.events)} events to {self.soul_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to flush to SOUL.md: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        success_rate = (
            self.stats['successful_captures'] / max(1, self.stats['total_attempts'])
        )
        
        avg_profit = (
            self.stats['total_profit_bps'] / max(1, self.stats['successful_captures'])
        )
        
        return {
            **{k: float(v) if isinstance(v, Decimal) else v for k, v in self.stats.items()},
            'success_rate': success_rate,
            'average_profit_bps': float(avg_profit),
            'events_in_memory': len(self.events),
            'funding_captures_count': len(self.funding_captures),
            'unique_symbols': len(set(e.symbol for e in self.events)),
            'arb_types_used': list(set(e.arb_type.value for e in self.events))
        }
    
    def export_summary(self) -> str:
        """Export a text summary of all activity"""
        lines = [
            "=" * 60,
            "ZAID CRYPTO TRADING BOT - ARBITRAGE SUMMARY",
            "=" * 60,
            "",
            f"Total Attempts: {self.stats['total_attempts']}",
            f"Successful: {self.stats['successful_captures']}",
            f"Missed: {self.stats['missed_opportunities']}",
            f"Failed: {self.stats['failed_attempts']}",
            "",
            f"Success Rate: {self.get_metrics()['success_rate']:.1%}",
            f"Avg Profit: {self.get_metrics()['average_profit_bps']:.2f} bps",
            f"Avg Slippage: {self.stats['total_slippage_bps'] / max(1, self.stats['total_attempts']):.2f} bps",
            "",
            f"Total Capital Deployed: {self.stats['total_capital_deployed']:.2f} USDT",
            f"Total Profit: {self.stats['total_profit_bps']:.2f} bps",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    async def test_logger():
        logger_instance = ArbSoulLogger(soul_path="SOUL.md")
        
        # Log some events
        logger_instance.log_arb_attempt(
            arb_type=ArbType.BASIS,
            outcome=ArbOutcome.SUCCESS,
            symbol="BTC",
            expected_profit_bps=Decimal('15.0'),
            actual_profit_bps=Decimal('14.5'),
            slippage_bps=Decimal('0.5'),
            capital_deployed=Decimal('10000')
        )
        
        logger_instance.log_missed_arb(
            arb_type=ArbType.TRIANGULAR,
            symbol="ETH",
            expected_profit_bps=Decimal('8.0'),
            reason="leg_timeout",
            capital_required=Decimal('5000')
        )
        
        logger_instance.log_funding_spike_capture(
            symbol="SOL",
            funding_rate=Decimal('0.0002'),
            predicted_rate=Decimal('0.00018'),
            payment_received=Decimal('1.20'),
            position_side="short"
        )
        
        # Flush to file
        await logger_instance.flush_to_soul()
        
        # Print summary
        print(logger_instance.export_summary())
        print(f"\nMetrics: {logger_instance.get_metrics()}")
    
    asyncio.run(test_logger())
