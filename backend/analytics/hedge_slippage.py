#!/usr/bin/env python3
"""
Hedge Slippage Analytics - Tracking Hedging Costs vs Theoretical

This module tracks the exact cost of continuous delta hedging vs theoretical
expectations. It identifies the precise microsecond where hedging slippage
destroyed theoretical alpha.

Key Features:
- Real-time slippage tracking per hedge execution
- Theoretical vs actual cost comparison
- Microsecond-level timing analysis
- Slippage attribution (spread, impact, latency)
- Strict type hinting for memory safety

Target: Isolate exact microsecond where slippage destroyed alpha
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SlippageComponent(Enum):
    """Components contributing to slippage."""
    BID_ASK_SPREAD = auto()
    MARKET_IMPACT = auto()
    LATENCY = auto()
    EXCHANGE_FEE = auto()
    FUNDING_COST = auto()
    OTHER = auto()


@dataclass
class TheoreticalExpectation:
    """Theoretical hedge execution expectation."""
    expected_price: Decimal
    expected_quantity: Decimal
    expected_cost: Decimal
    expected_slippage_bps: Decimal
    model: str  # e.g., "Black-Scholes", "Binomial"
    confidence_interval: Tuple[Decimal, Decimal]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ActualExecution:
    """Actual hedge execution result."""
    executed_price: Decimal
    executed_quantity: Decimal
    actual_cost: Decimal
    execution_time_us: int
    exchange_timestamp: datetime
    fill_rate: Decimal  # Percentage of order filled
    fees_paid: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SlippageAnalysis:
    """Detailed slippage analysis for a hedge."""
    hedge_id: str
    asset: str
    side: str
    theoretical_cost: Decimal
    actual_cost: Decimal
    total_slippage: Decimal
    slippage_bps: Decimal
    slippage_breakdown: Dict[SlippageComponent, Decimal]
    timing_analysis: TimingAnalysis
    alpha_destroyed: Decimal
    severity: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TimingAnalysis:
    """Microsecond-level timing breakdown."""
    signal_generation_us: int
    order_submission_us: int
    exchange_receipt_us: int
    first_fill_us: int
    completion_us: int
    total_latency_us: int
    critical_microsecond: Optional[int]  # Where alpha was lost
    
    @property
    def latency_breakdown(self) -> Dict[str, int]:
        return {
            'signal_to_submission': self.order_submission_us - self.signal_generation_us,
            'submission_to_receipt': self.exchange_receipt_us - self.order_submission_us,
            'receipt_to_first_fill': self.first_fill_us - self.exchange_receipt_us,
            'first_fill_to_completion': self.completion_us - self.first_fill_us,
        }


@dataclass
class HedgeRecord:
    """Complete hedge record with analysis."""
    hedge_id: str
    asset: str
    side: str
    quantity: Decimal
    theoretical: TheoreticalExpectation
    actual: ActualExecution
    analysis: SlippageAnalysis
    tags: List[str] = field(default_factory=list)


class SlippageTracker:
    """Tracks slippage for individual hedge executions."""
    
    def __init__(self, alpha_threshold_bps: Decimal = Decimal('1')):
        self.alpha_threshold_bps = alpha_threshold_bps
        self._records: Dict[str, HedgeRecord] = {}
        self._lock = threading.RLock()
        
        # Cumulative metrics
        self.total_slippage: Decimal = Decimal('0')
        self.total_alpha_destroyed: Decimal = Decimal('0')
        self.hedges_tracked: int = 0
    
    def record_hedge(
        self,
        hedge_id: str,
        asset: str,
        side: str,
        theoretical: TheoreticalExpectation,
        actual: ActualExecution,
        timing: TimingAnalysis
    ) -> SlippageAnalysis:
        """Record and analyze a hedge execution."""
        with self._lock:
            # Calculate slippage
            total_slippage = actual.actual_cost - theoretical.expected_cost
            
            slippage_bps = Decimal('0')
            if theoretical.expected_cost > 0:
                slippage_bps = (total_slippage / theoretical.expected_cost * 10000).quantize(Decimal('0.01'))
            
            # Break down slippage by component
            breakdown = self._attribute_slippage(theoretical, actual, timing)
            
            # Calculate alpha destroyed
            alpha_destroyed = max(Decimal('0'), total_slippage)
            
            # Determine severity
            if slippage_bps > self.alpha_threshold_bps * 5:
                severity = "CRITICAL"
            elif slippage_bps > self.alpha_threshold_bps * 2:
                severity = "HIGH"
            elif slippage_bps > self.alpha_threshold_bps:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            # Find critical microsecond
            critical_us = self._find_critical_microsecond(timing, breakdown)
            timing.critical_microsecond = critical_us
            
            analysis = SlippageAnalysis(
                hedge_id=hedge_id,
                asset=asset,
                side=side,
                theoretical_cost=theoretical.expected_cost,
                actual_cost=actual.actual_cost,
                total_slippage=total_slippage,
                slippage_bps=slippage_bps,
                slippage_breakdown=breakdown,
                timing_analysis=timing,
                alpha_destroyed=alpha_destroyed,
                severity=severity,
            )
            
            # Store record
            record = HedgeRecord(
                hedge_id=hedge_id,
                asset=asset,
                side=side,
                quantity=theoretical.expected_quantity,
                theoretical=theoretical,
                actual=actual,
                analysis=analysis,
            )
            self._records[hedge_id] = record
            
            # Update cumulative metrics
            self.total_slippage += total_slippage
            self.total_alpha_destroyed += alpha_destroyed
            self.hedges_tracked += 1
            
            return analysis
    
    def _attribute_slippage(
        self,
        theoretical: TheoreticalExpectation,
        actual: ActualExecution,
        timing: TimingAnalysis
    ) -> Dict[SlippageComponent, Decimal]:
        """Attribute slippage to specific components."""
        breakdown = {}
        
        # Bid-ask spread component
        spread_component = abs(actual.executed_price - theoretical.expected_price) * actual.executed_quantity
        breakdown[SlippageComponent.BID_ASK_SPREAD] = spread_component
        
        # Market impact (simplified estimation)
        impact_estimate = actual.executed_quantity * Decimal('0.0001')  # 1 bps per unit
        breakdown[SlippageComponent.MARKET_IMPACT] = impact_estimate
        
        # Latency cost (opportunity cost during delay)
        latency_seconds = Decimal(timing.total_latency_us) / Decimal('1000000')
        latency_cost = theoretical.expected_cost * Decimal('0.0001') * latency_seconds
        breakdown[SlippageComponent.LATENCY] = latency_cost
        
        # Exchange fees
        breakdown[SlippageComponent.EXCHANGE_FEE] = actual.fees_paid
        
        # Funding cost (for perps)
        breakdown[SlippageComponent.FUNDING_COST] = Decimal('0')  # Would be calculated separately
        
        # Remainder to "other"
        attributed = sum(breakdown.values())
        remainder = abs(actual.actual_cost - theoretical.expected_cost) - attributed
        breakdown[SlippageComponent.OTHER] = max(Decimal('0'), remainder)
        
        return breakdown
    
    def _find_critical_microsecond(
        self,
        timing: TimingAnalysis,
        breakdown: Dict[SlippageComponent, Decimal]
    ) -> Optional[int]:
        """Identify the microsecond where alpha was destroyed."""
        # Find largest slippage component
        if not breakdown:
            return None
        
        max_component = max(breakdown.items(), key=lambda x: x[1])
        
        if max_component[0] == SlippageComponent.LATENCY:
            # Alpha lost during latency period
            return timing.exchange_receipt_us
        elif max_component[0] == SlippageComponent.MARKET_IMPACT:
            # Alpha lost at first fill
            return timing.first_fill_us
        elif max_component[0] == SlippageComponent.BID_ASK_SPREAD:
            # Alpha lost at order submission
            return timing.order_submission_us
        
        return None
    
    def get_record(self, hedge_id: str) -> Optional[HedgeRecord]:
        """Get hedge record by ID."""
        with self._lock:
            return self._records.get(hedge_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate slippage statistics."""
        with self._lock:
            if self.hedges_tracked == 0:
                return {'error': 'No hedges tracked'}
            
            avg_slippage = self.total_slippage / Decimal(self.hedges_tracked)
            avg_alpha = self.total_alpha_destroyed / Decimal(self.hedges_tracked)
            
            # Count by severity
            severity_counts = {}
            for record in self._records.values():
                sev = record.analysis.severity
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            return {
                'hedges_tracked': self.hedges_tracked,
                'total_slippage': float(self.total_slippage),
                'total_alpha_destroyed': float(self.total_alpha_destroyed),
                'average_slippage': float(avg_slippage),
                'average_alpha_destroyed': float(avg_alpha),
                'severity_distribution': severity_counts,
            }


class HedgeSlippageAnalyzer:
    """Main analyzer for hedge slippage patterns."""
    
    def __init__(self):
        self.tracker = SlippageTracker()
        self._asset_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def analyze_hedge(
        self,
        hedge_id: str,
        asset: str,
        side: str,
        theoretical: TheoreticalExpectation,
        actual: ActualExecution,
        timing: TimingAnalysis
    ) -> SlippageAnalysis:
        """Analyze a single hedge execution."""
        analysis = self.tracker.record_hedge(
            hedge_id, asset, side, theoretical, actual, timing
        )
        
        # Update asset-level stats
        with self._lock:
            if asset not in self._asset_stats:
                self._asset_stats[asset] = {
                    'count': 0,
                    'total_slippage': Decimal('0'),
                    'critical_microseconds': [],
                }
            
            stats = self._asset_stats[asset]
            stats['count'] += 1
            stats['total_slippage'] += analysis.total_slippage
            
            if timing.critical_microsecond is not None:
                stats['critical_microseconds'].append(timing.critical_microsecond)
        
        return analysis
    
    def find_worst_slippage_hedges(self, limit: int = 10) -> List[HedgeRecord]:
        """Find hedges with worst slippage."""
        with self._lock:
            sorted_records = sorted(
                self.tracker._records.values(),
                key=lambda r: r.analysis.slippage_bps,
                reverse=True
            )
            return sorted_records[:limit]
    
    def get_asset_summary(self, asset: str) -> Dict[str, Any]:
        """Get slippage summary for specific asset."""
        with self._lock:
            if asset not in self._asset_stats:
                return {'error': f'No data for asset {asset}'}
            
            stats = self._asset_stats[asset]
            count = stats['count']
            
            return {
                'asset': asset,
                'hedge_count': count,
                'total_slippage': float(stats['total_slippage']),
                'average_slippage': float(stats['total_slippage'] / Decimal(count)) if count > 0 else 0,
                'critical_microseconds_found': len(stats['critical_microseconds']),
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive slippage report."""
        with self._lock:
            tracker_stats = self.tracker.get_statistics()
            
            worst_hedges = self.find_worst_slippage_hedges(5)
            worst_list = [
                {
                    'hedge_id': h.hedge_id,
                    'asset': h.asset,
                    'slippage_bps': float(h.analysis.slippage_bps),
                    'critical_us': h.analysis.timing_analysis.critical_microsecond,
                }
                for h in worst_hedges
            ]
            
            return {
                'summary': tracker_stats,
                'by_asset': {
                    asset: {
                        'count': stats['count'],
                        'avg_slippage': float(stats['total_slippage'] / Decimal(stats['count'])) if stats['count'] > 0 else 0,
                    }
                    for asset, stats in self._asset_stats.items()
                },
                'worst_hedges': worst_list,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }


def main() -> None:
    """Example usage."""
    from decimal import Decimal
    
    analyzer = HedgeSlippageAnalyzer()
    
    # Create sample theoretical expectation
    theoretical = TheoreticalExpectation(
        expected_price=Decimal('50000'),
        expected_quantity=Decimal('1'),
        expected_cost=Decimal('50000'),
        expected_slippage_bps=Decimal('2'),
        model='Black-Scholes',
        confidence_interval=(Decimal('49900'), Decimal('50100')),
    )
    
    # Create sample actual execution
    actual = ActualExecution(
        executed_price=Decimal('50025'),
        executed_quantity=Decimal('1'),
        actual_cost=Decimal('50025'),
        execution_time_us=1500,
        exchange_timestamp=datetime.now(timezone.utc),
        fill_rate=Decimal('100'),
        fees_paid=Decimal('5'),
    )
    
    # Create timing analysis
    now_us = int(time.time() * 1000000)
    timing = TimingAnalysis(
        signal_generation_us=now_us,
        order_submission_us=now_us + 100,
        exchange_receipt_us=now_us + 150,
        first_fill_us=now_us + 200,
        completion_us=now_us + 1500,
        total_latency_us=1500,
        critical_microsecond=None,
    )
    
    # Analyze hedge
    analysis = analyzer.analyze_hedge(
        hedge_id='HEDGE-001',
        asset='BTC',
        side='SELL',
        theoretical=theoretical,
        actual=actual,
        timing=timing,
    )
    
    print(f"Slippage Analysis:")
    print(f"  Total Slippage: ${analysis.total_slippage}")
    print(f"  Slippage (bps): {analysis.slippage_bps}")
    print(f"  Alpha Destroyed: ${analysis.alpha_destroyed}")
    print(f"  Severity: {analysis.severity}")
    print(f"  Critical Microsecond: {timing.critical_microsecond}")
    
    # Generate report
    report = analyzer.generate_report()
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
