#!/usr/bin/env python3
"""
backend/orderflow/flow_soul_logger.py

Logs trapped traders, liquidity sweeps, and critical order flow events to SOUL.md.
Implements comprehensive logging for post-trade analysis and system monitoring.

Features:
- Logs trapped trader events at key levels
- Records liquidity sweep confirmations
- Tracks CVD divergence-based false breakout fades
- Updates SOUL.md with critical events
- Strict type hinting for production reliability
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum, auto
import time
import json
import os
from datetime import datetime


class EventType(Enum):
    """Types of order flow events to log."""
    TRAPPED_TRADERS = auto()
    LIQUIDITY_SWEEP = auto()
    FALSE_BREAKOUT_FADE = auto()
    ABSorption_CONFIRMED = auto()
    CVD_DIVERGENCE = auto()
    VOLUME_CLUSTER = auto()
    STOP_HUNT = auto()
    IMBALANCE_SPIKE = auto()


@dataclass
class TrappedTradersEvent:
    """Event representing trapped traders at a price level."""
    price: float
    direction: str  # 'long' or 'short'
    trapped_volume: float
    level_type: str  # 'support', 'resistance', 'vwap', 'poc'
    estimated_stop_loss_cluster: float
    timestamp_ns: int
    confidence: float


@dataclass
class LiquiditySweepEvent:
    """Event representing a liquidity sweep."""
    swept_price: float
    sweep_direction: str  # 'up' or 'down'
    volume_swept: float
    follow_through: bool  # Did price continue or reverse?
    reversal_price: Optional[float]
    timestamp_ns: int
    stop_hunt_confirmed: bool


@dataclass
class FalseBreakoutFade:
    """Event representing a successful fade of a false breakout."""
    breakout_price: float
    breakout_direction: str  # 'up' or 'down'
    fakeout_distance: float  # How far did it break before reversing
    cvd_divergence: bool  # Was there CVD divergence?
    entry_price: float
    exit_price: Optional[float]
    pnl_points: Optional[float]
    volume_cluster: float
    timestamp_ns: int


@dataclass
class VolumeClusterEvent:
    """Represents a significant volume cluster where stops were hunted."""
    price_level: float
    total_volume: float
    aggressive_volume_pct: float
    stops_triggered_estimate: float
    price_reaction: str  # 'reversal', 'continuation', 'consolidation'
    timestamp_ns: int


class FlowSoulLogger:
    """
    Main logger for order flow events.
    Writes critical events to SOUL.md for post-analysis.
    """
    
    def __init__(self, soul_md_path: str = "SOUL.md"):
        self.soul_md_path = soul_md_path
        self.events: Dict[EventType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.session_stats: Dict[str, Any] = {
            'total_trapped_events': 0,
            'total_liquidity_sweeps': 0,
            'successful_fades': 0,
            'failed_fades': 0,
            'total_stop_hunts': 0,
            'session_start_ns': time.time_ns(),
        }
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md with header if it doesn't exist."""
        if not os.path.exists(self.soul_md_path):
            with open(self.soul_md_path, 'w') as f:
                f.write("# ZAID BOT - Order Flow Soul Log\n\n")
                f.write("*Tracking the soul of the market through order flow analysis*\n\n")
                f.write("---\n\n")
                f.write(f"## Session Started: {datetime.now().isoformat()}\n\n")
                f.write("## Event Summary\n\n")
    
    def log_trapped_traders(self, event: TrappedTradersEvent) -> None:
        """Log trapped traders event."""
        self.events[EventType.TRAPPED_TRADERS].append(event)
        self.session_stats['total_trapped_events'] += 1
        
        self._write_to_soul(
            "### 🪤 Trapped Traders Detected\n",
            {
                'price': f"${event.price:,.2f}",
                'direction': event.direction.upper(),
                'trapped_volume': f"{event.trapped_volume:.2f}",
                'level_type': event.level_type,
                'stop_loss_cluster': f"${event.estimated_stop_loss_cluster:,.2f}",
                'confidence': f"{event.confidence:.2%}",
                'timestamp': self._ns_to_datetime(event.timestamp_ns),
            }
        )
    
    def log_liquidity_sweep(self, event: LiquiditySweepEvent) -> None:
        """Log liquidity sweep event."""
        self.events[EventType.LIQUIDITY_SWEEP].append(event)
        self.session_stats['total_liquidity_sweeps'] += 1
        
        if event.stop_hunt_confirmed:
            self.session_stats['total_stop_hunts'] += 1
        
        self._write_to_soul(
            "### 🧹 Liquidity Sweep Detected\n",
            {
                'swept_price': f"${event.swept_price:,.2f}",
                'direction': event.sweep_direction.upper(),
                'volume_swept': f"{event.volume_swept:.2f}",
                'follow_through': 'Yes' if event.follow_through else 'No (Reversal)',
                'reversal_price': f"${event.reversal_price:,.2f}" if event.reversal_price else 'N/A',
                'stop_hunt_confirmed': '✅ Yes' if event.stop_hunt_confirmed else '❌ No',
                'timestamp': self._ns_to_datetime(event.timestamp_ns),
            }
        )
    
    def log_false_breakout_fade(self, event: FalseBreakoutFade) -> None:
        """Log successful fade of false breakout using CVD divergence."""
        self.events[EventType.FALSE_BREAKOUT_FADE].append(event)
        
        if event.pnl_points is not None:
            if event.pnl_points > 0:
                self.session_stats['successful_fades'] += 1
            else:
                self.session_stats['failed_fades'] += 1
        
        self._write_to_soul(
            "### 🎭 False Breakout Fade (CVD Divergence)\n",
            {
                'breakout_price': f"${event.breakout_price:,.2f}",
                'breakout_direction': event.breakout_direction.upper(),
                'fakeout_distance': f"${event.fakeout_distance:,.2f}",
                'cvd_divergence': '✅ Yes' if event.cvd_divergence else '❌ No',
                'entry_price': f"${event.entry_price:,.2f}",
                'exit_price': f"${event.exit_price:,.2f}" if event.exit_price else 'Open',
                'pnl_points': f"{event.pnl_points:+.2f}" if event.pnl_points else 'Open',
                'volume_cluster': f"{event.volume_cluster:.2f}",
                'timestamp': self._ns_to_datetime(event.timestamp_ns),
            }
        )
    
    def log_volume_cluster_stop_hunt(
        self,
        event: VolumeClusterEvent
    ) -> None:
        """Log the exact volume cluster where a stop hunt was executed."""
        self.events[EventType.VOLUME_CLUSTER].append(event)
        self.session_stats['total_stop_hunts'] += 1
        
        self._write_to_soul(
            "### 📊 Stop Hunt at Volume Cluster\n",
            {
                'price_level': f"${event.price_level:,.2f}",
                'total_volume': f"{event.total_volume:.2f}",
                'aggressive_volume_pct': f"{event.aggressive_volume_pct:.2%}",
                'stops_triggered_estimate': f"{event.stops_triggered_estimate:.2f}",
                'price_reaction': event.price_reaction.upper(),
                'timestamp': self._ns_to_datetime(event.timestamp_ns),
            },
            highlight=True
        )
    
    def log_cvd_divergence(
        self,
        symbol: str,
        price: float,
        divergence_type: str,
        cvd_value: float,
        confidence: float
    ) -> None:
        """Log CVD divergence event."""
        self.events[EventType.CVD_DIVERGENCE].append({
            'symbol': symbol,
            'price': price,
            'divergence_type': divergence_type,
            'cvd_value': cvd_value,
            'confidence': confidence,
            'timestamp_ns': time.time_ns(),
        })
        
        self._write_to_soul(
            f"### 📉 CVD Divergence on {symbol}\n",
            {
                'price': f"${price:,.2f}",
                'type': divergence_type.upper(),
                'cvd_value': f"{cvd_value:+.2f}",
                'confidence': f"{confidence:.2%}",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            }
        )
    
    def _write_to_soul(
        self,
        header: str,
        details: Dict[str, str],
        highlight: bool = False
    ) -> None:
        """Write an event to SOUL.md file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.soul_md_path, 'a') as f:
            if highlight:
                f.write(f"\n**[{timestamp}]** ")
            else:
                f.write(f"\n[{timestamp}] ")
            
            f.write(f"{header}\n\n")
            
            for key, value in details.items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
            
            f.write("\n---\n")
    
    def _ns_to_datetime(self, timestamp_ns: int) -> str:
        """Convert nanoseconds to formatted datetime string."""
        dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def get_recent_events(
        self,
        event_type: EventType,
        count: int = 10
    ) -> List[Any]:
        """Get recent events of a specific type."""
        return list(self.events[event_type])[-count:]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for current session."""
        return self.session_stats.copy()
    
    def write_session_summary(self) -> None:
        """Write session summary to SOUL.md."""
        with open(self.soul_md_path, 'a') as f:
            f.write("\n## Session Summary\n\n")
            f.write(f"- **Total Trapped Trader Events:** {self.session_stats['total_trapped_events']}\n")
            f.write(f"- **Total Liquidity Sweeps:** {self.session_stats['total_liquidity_sweeps']}\n")
            f.write(f"- **Total Stop Hunts:** {self.session_stats['total_stop_hunts']}\n")
            f.write(f"- **Successful Fades:** {self.session_stats['successful_fades']}\n")
            f.write(f"- **Failed Fades:** {self.session_stats['failed_fades']}\n")
            
            if self.session_stats['successful_fades'] + self.session_stats['failed_fades'] > 0:
                fade_rate = (
                    self.session_stats['successful_fades'] / 
                    (self.session_stats['successful_fades'] + self.session_stats['failed_fades'])
                )
                f.write(f"- **Fade Success Rate:** {fade_rate:.2%}\n")
            
            f.write(f"\n*Last Updated: {datetime.now().isoformat()}*\n")


def create_test_logger() -> FlowSoulLogger:
    """Create a logger with test data."""
    logger = FlowSoulLogger()
    return logger


if __name__ == "__main__":
    # Example usage
    logger = create_test_logger()
    
    base_time = time.time_ns()
    
    # Log trapped traders
    logger.log_trapped_traders(TrappedTradersEvent(
        price=60000.0,
        direction='long',
        trapped_volume=150.0,
        level_type='resistance',
        estimated_stop_loss_cluster=59800.0,
        confidence=0.85,
        timestamp_ns=base_time
    ))
    
    # Log liquidity sweep
    logger.log_liquidity_sweep(LiquiditySweepEvent(
        swept_price=60500.0,
        sweep_direction='up',
        volume_swept=200.0,
        follow_through=False,
        reversal_price=60200.0,
        stop_hunt_confirmed=True,
        timestamp_ns=base_time + 1_000_000_000
    ))
    
    # Log false breakout fade with CVD divergence
    logger.log_false_breakout_fade(FalseBreakoutFade(
        breakout_price=60500.0,
        breakout_direction='up',
        fakeout_distance=300.0,
        cvd_divergence=True,
        entry_price=60200.0,
        exit_price=60400.0,
        pnl_points=200.0,
        volume_cluster=180.0,
        timestamp_ns=base_time + 2_000_000_000
    ))
    
    # Log volume cluster stop hunt
    logger.log_volume_cluster_stop_hunt(VolumeClusterEvent(
        price_level=59800.0,
        total_volume=500.0,
        aggressive_volume_pct=0.75,
        stops_triggered_estimate=350.0,
        price_reaction='reversal',
        timestamp_ns=base_time + 3_000_000_000
    ))
    
    # Write session summary
    logger.write_session_summary()
    
    print(f"Events logged to {logger.soul_md_path}")
    print(f"Session Stats: {logger.get_session_summary()}")
