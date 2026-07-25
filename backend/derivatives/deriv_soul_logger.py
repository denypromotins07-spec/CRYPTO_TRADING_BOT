#!/usr/bin/env python3
"""
Derivatives Soul Logger

Logs funding captures, liquidation dodges, and derivatives trading events
to SOUL.md for the ZAID PERSONAL CRYPTO TRADING BOT.

Features:
- Funding rate trap avoidance logging
- Basis arbitrage capture tracking
- Liquidation prevention records
- Comprehensive derivatives performance metrics

Target: Update SOUL.md when the bot successfully dodges a funding rate trap.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of derivatives events to log"""
    FUNDING_TRAP_DODGED = "funding_trap_dodged"
    FUNDING_CAPTURED = "funding_captured"
    BASIS_ARB_EXECUTED = "basis_arb_executed"
    BASIS_ARB_CLOSED = "basis_arb_closed"
    LIQUIDATION_AVOIDED = "liquidation_avoided"
    DELEVERAGE_TRIGGERED = "deleverage_triggered"
    OPTIONS_HEDGE_PLACED = "options_hedge_placed"
    VOLATILITY_SPIKE_DETECTED = "volatility_spike_detected"
    SMART_MONEY_SIGNAL = "smart_money_signal"


@dataclass
class DerivativesEvent:
    """Represents a derivatives trading event"""
    event_type: EventType
    symbol: str
    timestamp_us: int
    details: Dict
    pnl_impact: float = 0.0
    risk_reduced: bool = False
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.timestamp_us == 0:
            self.timestamp_us = int(time.time() * 1_000_000)
    
    @property
    def timestamp_iso(self) -> str:
        """Get ISO format timestamp"""
        dt = datetime.fromtimestamp(self.timestamp_us / 1_000_000, tz=timezone.utc)
        return dt.isoformat()


@dataclass
class FundingTrapRecord:
    """Record of a funding rate trap that was detected/avoided"""
    symbol: str
    predicted_funding: float
    actual_funding: float
    trap_type: str  # 'manipulation', 'extreme_spread', 'reversal'
    action_taken: str  # 'closed_position', 'reduced_exposure', 'hedged'
    pnl_saved: float
    timestamp_us: int


class DerivativesSoulLogger:
    """
    Main logger for derivatives trading events.
    
    Writes comprehensive logs to SOUL.md including:
    - Funding rate captures and avoidances
    - Basis arbitrage PnL
    - Liquidation prevention events
    - Risk management actions
    """
    
    def __init__(self, soul_path: str = "SOUL.md"):
        self.soul_path = soul_path
        self.events: List[DerivativesEvent] = []
        self.funding_traps: List[FundingTrapRecord] = []
        self.total_funding_captured = 0.0
        self.total_basis_pnl = 0.0
        self.liquidations_avoided = 0
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(soul_path) if os.path.dirname(soul_path) else '.', exist_ok=True)
    
    def log_event(self, event: DerivativesEvent) -> None:
        """Log a derivatives event"""
        self.events.append(event)
        
        # Update counters
        if event.event_type == EventType.FUNDING_CAPTURED:
            self.total_funding_captured += event.pnl_impact
        elif event.event_type in (EventType.BASIS_ARB_EXECUTED, EventType.BASIS_ARB_CLOSED):
            self.total_basis_pnl += event.pnl_impact
        elif event.event_type == EventType.LIQUIDATION_AVOIDED:
            self.liquidations_avoided += 1
        
        # Write to SOUL.md
        self._append_to_soul(event)
        
        logger.info(f"Logged event: {event.event_type.value} for {event.symbol}")
    
    def log_funding_trap_dodged(
        self,
        symbol: str,
        predicted_funding: float,
        actual_funding: float,
        action_taken: str,
        pnl_saved: float,
        trap_type: str = "manipulation",
    ) -> None:
        """
        Log when the bot successfully dodges a funding rate trap.
        
        This is the primary trigger for SOUL.md updates as specified.
        """
        event = DerivativesEvent(
            event_type=EventType.FUNDING_TRAP_DODGED,
            symbol=symbol,
            timestamp_us=int(time.time() * 1_000_000),
            details={
                'predicted_funding': predicted_funding,
                'actual_funding': actual_funding,
                'trap_type': trap_type,
                'action_taken': action_taken,
            },
            pnl_impact=pnl_saved,
            risk_reduced=True,
            confidence=0.9,
        )
        
        record = FundingTrapRecord(
            symbol=symbol,
            predicted_funding=predicted_funding,
            actual_funding=actual_funding,
            trap_type=trap_type,
            action_taken=action_taken,
            pnl_saved=pnl_saved,
            timestamp_us=event.timestamp_us,
        )
        
        self.funding_traps.append(record)
        self.log_event(event)
        
        logger.info(f"Funding trap dodged on {symbol}: saved ${pnl_saved:.2f}")
    
    def log_funding_captured(
        self,
        symbol: str,
        funding_rate: float,
        position_size: float,
        funding_pnl: float,
    ) -> None:
        """Log successful funding rate capture"""
        event = DerivativesEvent(
            event_type=EventType.FUNDING_CAPTURED,
            symbol=symbol,
            timestamp_us=int(time.time() * 1_000_000),
            details={
                'funding_rate': funding_rate,
                'position_size': position_size,
            },
            pnl_impact=funding_pnl,
            risk_reduced=False,
        )
        
        self.log_event(event)
    
    def log_basis_arbitrage(
        self,
        arb_id: str,
        symbol: str,
        basis_bps: float,
        entry_details: Dict,
        pnl: float = 0.0,
        is_close: bool = False,
    ) -> None:
        """Log basis arbitrage execution or close"""
        event_type = EventType.BASIS_ARB_CLOSED if is_close else EventType.BASIS_ARB_EXECUTED
        
        event = DerivativesEvent(
            event_type=event_type,
            symbol=symbol,
            timestamp_us=int(time.time() * 1_000_000),
            details={
                'arb_id': arb_id,
                'basis_bps': basis_bps,
                'entry_details': entry_details,
            },
            pnl_impact=pnl,
            risk_reduced=False,
        )
        
        self.log_event(event)
    
    def log_liquidation_avoided(
        self,
        symbol: str,
        liquidation_price: float,
        current_price: float,
        action_taken: str,
        position_value: float,
    ) -> None:
        """Log when liquidation was successfully avoided"""
        event = DerivativesEvent(
            event_type=EventType.LIQUIDATION_AVOIDED,
            symbol=symbol,
            timestamp_us=int(time.time() * 1_000_000),
            details={
                'liquidation_price': liquidation_price,
                'current_price': current_price,
                'action_taken': action_taken,
                'position_value': position_value,
            },
            pnl_impact=0.0,
            risk_reduced=True,
            confidence=1.0,
        )
        
        self.log_event(event)
        
        logger.warning(f"Liquidation avoided on {symbol} at ${liquidation_price:,.2f}")
    
    def log_deleverage_triggered(
        self,
        symbol: str,
        old_leverage: float,
        new_leverage: float,
        reason: str,
    ) -> None:
        """Log automatic deleveraging event"""
        event = DerivativesEvent(
            event_type=EventType.DELEVERAGE_TRIGGERED,
            symbol=symbol,
            timestamp_us=int(time.time() * 1_000_000),
            details={
                'old_leverage': old_leverage,
                'new_leverage': new_leverage,
                'reason': reason,
            },
            pnl_impact=0.0,
            risk_reduced=True,
        )
        
        self.log_event(event)
    
    def _append_to_soul(self, event: DerivativesEvent) -> None:
        """Append event to SOUL.md file"""
        timestamp = event.timestamp_iso
        
        # Create event entry
        entry = f"\n## [{timestamp}] {event.event_type.value.upper()}\n\n"
        entry += f"**Symbol:** {event.symbol}\n\n"
        entry += "**Details:**\n"
        
        for key, value in event.details.items():
            if isinstance(value, float):
                entry += f"- {key}: {value:.6f}\n"
            else:
                entry += f"- {key}: {value}\n"
        
        if event.pnl_impact != 0.0:
            entry += f"\n**PnL Impact:** ${event.pnl_impact:,.2f}\n"
        
        if event.risk_reduced:
            entry += "\n⚠️ **Risk Reduced** ✅\n"
        
        entry += "\n---\n"
        
        # Append to file
        try:
            with open(self.soul_path, 'a') as f:
                f.write(entry)
        except Exception as e:
            logger.error(f"Failed to write to SOUL.md: {e}")
    
    def write_summary_to_soul(self) -> None:
        """Write comprehensive summary to SOUL.md"""
        summary = "\n# DERIVATIVES SOUL SUMMARY\n\n"
        summary += f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n"
        
        summary += "## Performance Metrics\n\n"
        summary += f"- Total Funding Captured: ${self.total_funding_captured:,.2f}\n"
        summary += f"- Total Basis Arbitrage PnL: ${self.total_basis_pnl:,.2f}\n"
        summary += f"- Liquidations Avoided: {self.liquidations_avoided}\n"
        summary += f"- Funding Traps Dodged: {len(self.funding_traps)}\n"
        summary += f"- Total Events Logged: {len(self.events)}\n\n"
        
        if self.funding_traps:
            summary += "## Recent Funding Trap Dodges\n\n"
            for trap in self.funding_traps[-5:]:
                summary += f"- {trap.symbol}: Saved ${trap.pnl_saved:,.2f} ({trap.trap_type})\n"
            summary += "\n"
        
        summary += "---\n"
        
        try:
            with open(self.soul_path, 'a') as f:
                f.write(summary)
            logger.info(f"Summary written to {self.soul_path}")
        except Exception as e:
            logger.error(f"Failed to write summary to SOUL.md: {e}")
    
    def get_statistics(self) -> Dict:
        """Get derivatives trading statistics"""
        return {
            'total_events': len(self.events),
            'total_funding_captured': self.total_funding_captured,
            'total_basis_pnl': self.total_basis_pnl,
            'liquidations_avoided': self.liquidations_avoided,
            'funding_traps_dodged': len(self.funding_traps),
            'events_by_type': self._count_events_by_type(),
        }
    
    def _count_events_by_type(self) -> Dict[str, int]:
        """Count events by type"""
        counts: Dict[str, int] = {}
        for event in self.events:
            key = event.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts


# Global logger instance
_derivatives_logger: Optional[DerivativesSoulLogger] = None


def get_derivatives_logger(soul_path: str = "SOUL.md") -> DerivativesSoulLogger:
    """Get or create the global derivatives logger"""
    global _derivatives_logger
    if _derivatives_logger is None:
        _derivatives_logger = DerivativesSoulLogger(soul_path)
    return _derivatives_logger


if __name__ == "__main__":
    # Example usage
    print("Derivatives Soul Logger initialized")
    
    logger_instance = get_derivatives_logger()
    
    # Log sample events
    logger_instance.log_funding_trap_dodged(
        symbol="BTCUSDT",
        predicted_funding=0.0015,
        actual_funding=-0.0005,
        action_taken="closed_position",
        pnl_saved=1250.00,
        trap_type="manipulation",
    )
    
    logger_instance.log_funding_captured(
        symbol="ETHUSDT",
        funding_rate=0.0001,
        position_size=50000,
        funding_pnl=5.00,
    )
    
    logger_instance.log_liquidation_avoided(
        symbol="SOLUSDT",
        liquidation_price=95.50,
        current_price=98.00,
        action_taken="auto_deleverage",
        position_value=10000,
    )
    
    # Write summary
    logger_instance.write_summary_to_soul()
    
    print(f"\nStatistics: {logger_instance.get_statistics()}")
