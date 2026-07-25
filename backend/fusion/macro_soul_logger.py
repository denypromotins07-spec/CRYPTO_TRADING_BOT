"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 7
Chapter 4: Data Fusion - Macro Soul Logger

File: backend/fusion/macro_soul_logger.py
Purpose: Log macro-event trade impacts to SOUL.md.
         Track how the bot successfully navigates macro events like CPI.

Features:
- Automatic logging of macro event outcomes
- Trade impact analysis relative to economic events
- SOUL.md integration for contextual awareness
- Performance attribution by macro factor
- Memory-efficient event storage

Design Patterns:
- Observer: Log events on trade completion
- Adapter: Format logs for SOUL.md
- Strategy: Different attribution methods

Author: Opus 4.8
Domain: Performance Attribution, Event Analysis, Logging
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Deque
from collections import deque
from enum import Enum
import logging
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeOutcome(Enum):
    """Trade outcome classification."""
    PROFIT = "profit"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    DODGED_CRASH = "dodged_crash"      # Avoided loss due to macro signal
    CAUGHT_SPIKE = "caught_spike"       # Captured gain from macro event
    HALTED_PRE_EVENT = "halted_pre_event"  # Trading halted before event


class MacroEventType(Enum):
    """Macro event types that affect trades."""
    CPI = "CPI"
    PPI = "PPI"
    FED_RATE = "FED_RATE"
    NFP = "NFP"
    GDP = "GDP"
    RETAIL_SALES = "RETAIL_SALES"
    PMI = "PMI"
    OTHER = "OTHER"


@dataclass(slots=True)
class MacroEventReference:
    """Reference to a macro event."""
    event_id: str
    event_type: MacroEventType
    event_time: datetime
    actual_value: Optional[float]
    forecast_value: Optional[float]
    surprise_pct: Optional[float]
    market_impact_score: float  # 0-10


@dataclass(slots=True)
class TradeRecord:
    """Trade record for attribution."""
    trade_id: str
    asset: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl_usd: float
    pnl_pct: float
    outcome: TradeOutcome
    related_macro_event: Optional[str] = None


@dataclass(slots=True)
class MacroImpactLog:
    """Logged macro impact entry."""
    timestamp: datetime
    event_type: MacroEventType
    event_id: str
    trades_affected: int
    total_pnl_usd: float
    avoided_loss_usd: float  # Losses avoided due to macro signals
    captured_gain_usd: float  # Gains from macro positioning
    net_alpha_bps: float  # Alpha generated from macro awareness
    summary: str
    soul_entry: str  # Formatted entry for SOUL.md


class MacroSoulLogger:
    """
    Logs macro event impacts to SOUL.md file.
    
    Features:
    - Automatic trade attribution to macro events
    - SOUL.md file updates
    - Performance analytics by event type
    - Crash dodge documentation
    """
    
    def __init__(self, soul_md_path: str = "SOUL.md", max_history: int = 1000):
        self.soul_md_path = soul_md_path
        self.max_history = max_history
        
        # Event and trade storage (bounded)
        self.macro_events: Dict[str, MacroEventReference] = {}
        self.trades: Deque[TradeRecord] = deque(maxlen=max_history)
        self.impact_logs: Deque[MacroImpactLog] = deque(maxlen=500)
        
        # Performance tracking
        self.performance_by_event: Dict[MacroEventType, Dict[str, float]] = {}
        
        logger.info(f"MacroSoulLogger initialized, logging to {soul_md_path}")
    
    def register_macro_event(
        self,
        event_id: str,
        event_type: MacroEventType,
        event_time: datetime,
        actual: Optional[float],
        forecast: Optional[float],
        market_impact: float
    ):
        """Register a macro event for later attribution."""
        surprise_pct = None
        if actual is not None and forecast is not None and forecast != 0:
            surprise_pct = ((actual - forecast) / abs(forecast)) * 100
        
        event = MacroEventReference(
            event_id=event_id,
            event_type=event_type,
            event_time=event_time,
            actual_value=actual,
            forecast_value=forecast,
            surprise_pct=surprise_pct,
            market_impact_score=market_impact
        )
        
        self.macro_events[event_id] = event
        logger.debug(f"Registered macro event: {event_id} ({event_type.value})")
    
    def log_trade(
        self,
        trade_id: str,
        asset: str,
        entry_time: datetime,
        entry_price: float,
        position_size: float,
        exit_time: Optional[datetime] = None,
        exit_price: Optional[float] = None,
        related_macro_event: Optional[str] = None
    ):
        """Log a trade for later macro attribution."""
        # Calculate PnL
        if exit_price is not None:
            pnl_usd = (exit_price - entry_price) * position_size
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            
            if pnl_usd > 1:  # Threshold for significance
                outcome = TradeOutcome.PROFIT
            elif pnl_usd < -1:
                outcome = TradeOutcome.LOSS
            else:
                outcome = TradeOutcome.BREAKEVEN
        else:
            pnl_usd = 0
            pnl_pct = 0
            outcome = TradeOutcome.BREAKEVEN  # Open trade
        
        trade = TradeRecord(
            trade_id=trade_id,
            asset=asset,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            outcome=outcome,
            related_macro_event=related_macro_event
        )
        
        self.trades.append(trade)
        
        # Check if we should attribute to a macro event
        if related_macro_event and related_macro_event in self.macro_events:
            self._update_performance_attribution(trade, related_macro_event)
    
    def _update_performance_attribution(
        self,
        trade: TradeRecord,
        event_id: str
    ):
        """Update performance attribution for a macro event."""
        event = self.macro_events.get(event_id)
        if not event:
            return
        
        event_type = event.event_type
        
        if event_type not in self.performance_by_event:
            self.performance_by_event[event_type] = {
                "total_trades": 0,
                "total_pnl": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avoided_losses": 0.0,
                "captured_gains": 0.0
            }
        
        stats = self.performance_by_event[event_type]
        stats["total_trades"] += 1
        stats["total_pnl"] += trade.pnl_usd
        
        if trade.pnl_usd > 0:
            stats["winning_trades"] += 1
            stats["captured_gains"] += trade.pnl_usd
        elif trade.pnl_usd < 0:
            stats["losing_trades"] += 1
    
    def log_dodged_crash(
        self,
        event_id: str,
        estimated_avoided_loss: float,
        reason: str
    ):
        """Log when the bot successfully dodged a crash due to macro signals."""
        event = self.macro_events.get(event_id)
        if not event:
            return
        
        now = datetime.now(timezone.utc)
        
        log = MacroImpactLog(
            timestamp=now,
            event_type=event.event_type,
            event_id=event_id,
            trades_affected=0,
            total_pnl_usd=0,
            avoided_loss_usd=estimated_avoided_loss,
            captured_gain_usd=0,
            net_alpha_bps=(estimated_avoided_loss / 10000) * 100,  # Approximate bps
            summary=f"Successfully dodged {event.event_type.value}-induced crash. Avoided loss: ${estimated_avoided_loss:,.2f}",
            soul_entry=self._format_soul_entry(
                event,
                "DODGED_CRASH",
                f"Avoided ${estimated_avoided_loss:,.2f} loss",
                reason
            )
        )
        
        self.impact_logs.append(log)
        self._write_to_soul_md(log)
        
        logger.warning(
            f"🛡️ CRASH DODGED: {event.event_type.value} | "
            f"Avoided: ${estimated_avoided_loss:,.2f}"
        )
    
    def _format_soul_entry(
        self,
        event: MacroEventReference,
        outcome: str,
        result: str,
        details: str
    ) -> str:
        """Format an entry for SOUL.md."""
        surprise_str = f"{event.surprise_pct:.2f}%" if event.surprise_pct else "N/A"
        
        return f"""
## {event.event_type.value} Event - {outcome}

**Event ID:** `{event.event_id}`  
**Time:** {event.event_time.isoformat()}  
**Result:** {result}

### Event Details
- **Actual:** {event.actual_value}
- **Forecast:** {event.forecast_value}
- **Surprise:** {surprise_str}
- **Market Impact:** {event.market_impact_score}/10

### Analysis
{details}

---
"""
    
    def _write_to_soul_md(self, log: MacroImpactLog):
        """Append log entry to SOUL.md file."""
        try:
            # Create file if doesn't exist
            if not os.path.exists(self.soul_md_path):
                with open(self.soul_md_path, 'w') as f:
                    f.write("# SOUL.md - Macro Event Trade Impact Log\n\n")
                    f.write("This file documents how the ZAID bot navigates macroeconomic events.\n\n")
            
            # Append new entry
            with open(self.soul_md_path, 'a') as f:
                f.write(log.soul_entry)
            
            logger.info(f"Updated SOUL.md with {log.event_type.value} impact")
            
        except Exception as e:
            logger.error(f"Error writing to SOUL.md: {e}")
    
    def generate_impact_report(
        self,
        event_type: Optional[MacroEventType] = None
    ) -> Dict[str, Any]:
        """Generate macro impact report."""
        if event_type:
            stats = self.performance_by_event.get(event_type, {})
            return {
                "event_type": event_type.value,
                "statistics": stats,
                "win_rate": stats.get("winning_trades", 0) / max(stats.get("total_trades", 1), 1)
            }
        
        # Overall report
        total_pnl = sum(s.get("total_pnl", 0) for s in self.performance_by_event.values())
        total_trades = sum(s.get("total_trades", 0) for s in self.performance_by_event.values())
        total_wins = sum(s.get("winning_trades", 0) for s in self.performance_by_event.values())
        total_avoided = sum(s.get("avoided_losses", 0) for s in self.performance_by_event.values())
        
        return {
            "total_macro_trades": total_trades,
            "total_pnl_usd": total_pnl,
            "overall_win_rate": total_wins / max(total_trades, 1),
            "total_avoided_loss_usd": total_avoided,
            "by_event_type": {
                et.value: stats for et, stats in self.performance_by_event.items()
            }
        }
    
    def get_recent_logs(self, limit: int = 10) -> List[MacroImpactLog]:
        """Get recent impact logs."""
        return list(self.impact_logs)[-limit:]


# Example usage
async def main():
    """Demonstration of MacroSoulLogger functionality."""
    logger_instance = MacroSoulLogger(soul_md_path="/workspace/SOUL.md")
    
    # Register a CPI event
    cpi_time = datetime.now(timezone.utc) - timedelta(hours=2)
    logger_instance.register_macro_event(
        event_id="cpi_2024_03",
        event_type=MacroEventType.CPI,
        event_time=cpi_time,
        actual=3.2,
        forecast=3.1,
        market_impact=8.5
    )
    
    # Log some trades around the event
    logger_instance.log_trade(
        trade_id="trade_001",
        asset="BTC",
        entry_time=cpi_time - timedelta(minutes=30),
        entry_price=68000,
        position_size=0.5,
        exit_time=cpi_time + timedelta(minutes=60),
        exit_price=69500,
        related_macro_event="cpi_2024_03"
    )
    
    # Log a dodged crash
    logger_instance.log_dodged_crash(
        event_id="cpi_2024_03",
        estimated_avoided_loss=15000,
        reason="Bot detected high-impact CPI event and reduced exposure 30min before release. Market dropped 3% immediately after."
    )
    
    # Generate report
    print("\n📊 Macro Impact Report:")
    report = logger_instance.generate_impact_report()
    print(json.dumps(report, indent=2, default=str))
    
    print("\n📝 Recent Logs:")
    for log in logger_instance.get_recent_logs(3):
        print(f"  {log.event_type.value}: {log.summary}")


if __name__ == "__main__":
    asyncio.run(main())
