#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Transaction Cost Analysis (TCA)
Chapter 4: Slippage Tracker

This module compares signal price vs actual fill price to measure
execution quality and slippage. It tracks per-trade, per-asset, and
portfolio-level slippage metrics for continuous improvement.

Memory Budget: <40MB for trade history
Target Latency: <50μs per trade analysis
Assets: BTC, SOL, ETH, USDT parallel tracking
Integration: Feeds results to SOUL.md for learning

Author: Opus 4.8
Stage: 5/100 - Advanced Risk Management and Order Book Microstructure
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import statistics


class OrderSide(Enum):
    """Order direction."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order execution type."""
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_MAKER = "limit_maker"  # Passive liquidity


@dataclass
class SlippageConfig:
    """Configuration for slippage tracking."""
    assets: List[str] = field(default_factory=lambda: ["BTC", "SOL", "ETH", "USDT"])
    history_size: int = 10000  # Trades to retain
    acceptable_slippage_bps: float = 10.0  # 10 bps baseline
    warning_slippage_bps: float = 25.0  # Warning threshold
    critical_slippage_bps: float = 50.0  # Critical threshold


@dataclass
class TradeRecord(TypedDict):
    """Complete trade record for TCA."""
    trade_id: str
    asset: str
    side: str
    order_type: str
    signal_price: float
    fill_price: float
    quantity: float
    notional_value: float
    slippage_bps: float
    slippage_usd: float
    timestamp: float
    exchange: str
    fee_bps: float
    total_cost_bps: float


@dataclass
class SlippageMetrics(TypedDict):
    """Aggregated slippage metrics."""
    asset: str
    total_trades: int
    avg_slippage_bps: float
    median_slippage_bps: float
    std_slippage_bps: float
    max_slippage_bps: float
    min_slippage_bps: float
    total_slippage_usd: float
    pct_positive_slippage: float  # Favorable slippage
    pct_acceptable: float  # Within threshold
    time_period_hours: float


class SlippageTracker:
    """
    Comprehensive slippage tracking and analysis system.
    
    Features:
    - Per-trade slippage calculation
    - Multi-timescale aggregation
    - Favorable vs adverse slippage breakdown
    - Order type comparison
    - Exchange comparison
    - SOUL.md integration for lessons learned
    """
    
    __slots__ = (
        '_config', '_trade_history', '_asset_metrics',
        '_rolling_stats', '_alert_callbacks'
    )
    
    def __init__(self, config: SlippageConfig = None) -> None:
        """
        Initialize slippage tracker.
        
        Args:
            config: Slippage tracking configuration
        """
        self._config = config or SlippageConfig()
        
        # Trade history per asset
        self._trade_history: Dict[str, Deque[TradeRecord]] = {
            asset: deque(maxlen=self._config.history_size)
            for asset in self._config.assets
        }
        
        # Aggregated metrics per asset
        self._asset_metrics: Dict[str, Dict[str, float]] = {
            asset: {
                'total_trades': 0,
                'cumulative_slippage_bps': 0.0,
                'cumulative_slippage_usd': 0.0,
                'favorable_count': 0,
                'acceptable_count': 0,
                'sum_slippage': 0.0,
                'sum_slippage_squared': 0.0,
                'max_slippage': 0.0,
                'min_slippage': float('inf')
            }
            for asset in self._config.assets
        }
        
        # Rolling statistics window
        self._rolling_stats: Dict[str, Deque[float]] = {
            asset: deque(maxlen=100) for asset in self._config.assets
        }
        
        # Alert callbacks
        self._alert_callbacks: List[callable] = []
    
    def register_alert_callback(self, callback: callable) -> None:
        """Register callback for slippage alerts."""
        self._alert_callbacks.append(callback)
    
    def record_trade(
        self,
        trade_id: str,
        asset: str,
        side: OrderSide,
        order_type: OrderType,
        signal_price: float,
        fill_price: float,
        quantity: float,
        exchange: str = "default",
        fee_bps: float = 4.0
    ) -> Optional[TradeRecord]:
        """
        Record a completed trade and calculate slippage.
        
        Args:
            trade_id: Unique trade identifier
            asset: Asset traded
            side: BUY or SELL
            order_type: Market, Limit, etc.
            signal_price: Price when signal generated
            fill_price: Actual execution price
            quantity: Quantity traded
            exchange: Exchange name
            fee_bps: Trading fee in basis points
            
        Returns:
            TradeRecord or None if invalid
        """
        if asset not in self._trade_history:
            return None
        
        if signal_price <= 0 or fill_price <= 0 or quantity <= 0:
            return None
        
        # Calculate slippage based on side
        if side == OrderSide.BUY:
            # For buys, slippage is positive if fill > signal (adverse)
            price_diff = fill_price - signal_price
        else:  # SELL
            # For sells, slippage is positive if fill < signal (adverse)
            price_diff = signal_price - fill_price
        
        # Slippage in basis points
        slippage_bps = (price_diff / signal_price) * 10000
        
        # Slippage in USD
        notional = fill_price * quantity
        slippage_usd = (abs(price_diff) * quantity)
        
        # Total cost including fees
        total_cost_bps = abs(slippage_bps) + fee_bps
        
        # Determine if slippage was favorable (negative means we got better price)
        is_favorable = slippage_bps < 0
        is_acceptable = abs(slippage_bps) <= self._config.acceptable_slippage_bps
        
        record = TradeRecord(
            trade_id=trade_id,
            asset=asset,
            side=side.value,
            order_type=order_type.value,
            signal_price=signal_price,
            fill_price=fill_price,
            quantity=quantity,
            notional_value=notional,
            slippage_bps=slippage_bps,
            slippage_usd=slippage_usd,
            timestamp=time.time(),
            exchange=exchange,
            fee_bps=fee_bps,
            total_cost_bps=total_cost_bps
        )
        
        # Store trade
        self._trade_history[asset].append(record)
        
        # Update metrics
        self._update_metrics(asset, record, is_favorable, is_acceptable)
        
        # Add to rolling stats
        self._rolling_stats[asset].append(slippage_bps)
        
        # Check for alerts
        self._check_alerts(asset, record)
        
        return record
    
    def _update_metrics(self, asset: str, record: TradeRecord, 
                        is_favorable: bool, is_acceptable: bool) -> None:
        """Update aggregated metrics for an asset."""
        metrics = self._asset_metrics[asset]
        slippage = record['slippage_bps']
        
        metrics['total_trades'] += 1
        metrics['cumulative_slippage_bps'] += slippage
        metrics['cumulative_slippage_usd'] += record['slippage_usd']
        
        if is_favorable:
            metrics['favorable_count'] += 1
        
        if is_acceptable:
            metrics['acceptable_count'] += 1
        
        # Running sum for mean/variance calculation
        metrics['sum_slippage'] += slippage
        metrics['sum_slippage_squared'] += slippage ** 2
        
        # Track extremes
        metrics['max_slippage'] = max(metrics['max_slippage'], abs(slippage))
        metrics['min_slippage'] = min(metrics['min_slippage'], abs(slippage))
    
    def _check_alerts(self, asset: str, record: TradeRecord) -> None:
        """Generate alerts for excessive slippage."""
        slippage_bps = abs(record['slippage_bps'])
        
        alert = None
        
        if slippage_bps >= self._config.critical_slippage_bps:
            alert = {
                'level': 'CRITICAL',
                'asset': asset,
                'slippage_bps': slippage_bps,
                'message': f"Critical slippage: {slippage_bps:.1f} bps on {asset}"
            }
        elif slippage_bps >= self._config.warning_slippage_bps:
            alert = {
                'level': 'WARNING',
                'asset': asset,
                'slippage_bps': slippage_bps,
                'message': f"High slippage: {slippage_bps:.1f} bps on {asset}"
            }
        
        if alert:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass
    
    def get_metrics(self, asset: str) -> Optional[SlippageMetrics]:
        """Get aggregated slippage metrics for an asset."""
        if asset not in self._asset_metrics:
            return None
        
        metrics = self._asset_metrics[asset]
        trades = list(self._trade_history[asset])
        
        if metrics['total_trades'] == 0:
            return None
        
        # Calculate statistics
        n = metrics['total_trades']
        mean = metrics['sum_slippage'] / n
        
        variance = (metrics['sum_slippage_squared'] / n) - (mean ** 2)
        std = np.sqrt(variance) if variance > 0 else 0.0
        
        # Get median from recent trades
        recent_slippages = [t['slippage_bps'] for t in trades[-100:]]
        median = statistics.median(recent_slippages) if recent_slippages else 0.0
        
        # Time period
        if len(trades) >= 2:
            time_period = (trades[-1]['timestamp'] - trades[0]['timestamp']) / 3600
        else:
            time_period = 0.0
        
        return SlippageMetrics(
            asset=asset,
            total_trades=n,
            avg_slippage_bps=mean,
            median_slippage_bps=median,
            std_slippage_bps=std,
            max_slippage_bps=metrics['max_slippage'],
            min_slippage_bps=metrics['min_slippage'] if metrics['min_slippage'] != float('inf') else 0.0,
            total_slippage_usd=metrics['cumulative_slippage_usd'],
            pct_positive_slippage=(metrics['favorable_count'] / n) * 100 if n > 0 else 0,
            pct_acceptable=(metrics['acceptable_count'] / n) * 100 if n > 0 else 0,
            time_period_hours=max(time_period, 0.0)
        )
    
    def get_all_assets_metrics(self) -> Dict[str, SlippageMetrics]:
        """Get metrics for all tracked assets."""
        return {
            asset: metrics
            for asset in self._config.assets
            if (metrics := self.get_metrics(asset)) is not None
        }
    
    def compare_order_types(self, asset: str) -> Dict[str, Dict[str, float]]:
        """Compare slippage across order types for an asset."""
        trades = list(self._trade_history.get(asset, []))
        
        if not trades:
            return {}
        
        # Group by order type
        by_type: Dict[str, List[float]] = {}
        
        for trade in trades:
            ot = trade['order_type']
            if ot not in by_type:
                by_type[ot] = []
            by_type[ot].append(trade['slippage_bps'])
        
        # Calculate stats per type
        result = {}
        for order_type, slippages in by_type.items():
            if not slippages:
                continue
            
            result[order_type] = {
                'count': len(slippages),
                'avg_slippage_bps': statistics.mean(slippages),
                'std_slippage_bps': statistics.stdev(slippages) if len(slippages) > 1 else 0,
                'median_slippage_bps': statistics.median(slippages)
            }
        
        return result
    
    def get_execution_quality_score(self, asset: str) -> Optional[float]:
        """
        Calculate execution quality score (0-100).
        
        Higher is better. Based on:
        - Average slippage relative to benchmark
        - Percentage of trades within acceptable range
        - Consistency (low variance)
        """
        metrics = self.get_metrics(asset)
        if not metrics:
            return None
        
        # Component scores
        # 1. Slippage score (lower is better)
        benchmark = self._config.acceptable_slippage_bps
        slippage_score = max(0, 100 * (1 - abs(metrics['avg_slippage_bps']) / benchmark))
        
        # 2. Acceptability score
        acceptability_score = metrics['pct_acceptable']
        
        # 3. Consistency score (inverse of std)
        if metrics['std_slippage_bps'] > 0:
            consistency_score = max(0, 100 * (1 - metrics['std_slippage_bps'] / benchmark))
        else:
            consistency_score = 100
        
        # Weighted average
        quality_score = (
            slippage_score * 0.4 +
            acceptability_score * 0.4 +
            consistency_score * 0.2
        )
        
        return min(100, max(0, quality_score))
    
    def export_to_soul_md(self, output_path: str = "SOUL.md") -> None:
        """Export slippage analysis to SOUL.md."""
        import os
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            f"\n## Slippage Analysis Report - {timestamp}",
            "",
            "| Asset | Trades | Avg Slippage (bps) | Quality Score | Acceptable % |",
            "|-------|--------|-------------------|---------------|--------------|"
        ]
        
        for asset in self._config.assets:
            metrics = self.get_metrics(asset)
            if metrics:
                quality = self.get_execution_quality_score(asset)
                quality_str = f"{quality:.1f}" if quality else "N/A"
                lines.append(
                    f"| {asset} | {metrics['total_trades']} | "
                    f"{metrics['avg_slippage_bps']:.2f} | "
                    f"{quality_str} | {metrics['pct_acceptable']:.1f}% |"
                )
        
        # Add lessons section
        lines.extend([
            "",
            "### Execution Lessons",
            "- Monitor slippage during high volatility periods",
            "- Consider limit orders when spread is wide",
            "- Split large orders to reduce market impact",
            "- Track exchange-specific execution quality",
            ""
        ])
        
        try:
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = "# ZAID BOT SOUL.md - Trading Intelligence\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content + '\n'.join(lines))
        except Exception:
            pass


if __name__ == "__main__":
    # Example usage and validation
    config = SlippageConfig()
    tracker = SlippageTracker(config)
    
    import random
    
    print("=== Simulating Trade Executions ===")
    
    base_price = 50000.0
    
    for i in range(100):
        asset = random.choice(["BTC", "ETH", "SOL"])
        side = random.choice([OrderSide.BUY, OrderSide.SELL])
        order_type = random.choice([OrderType.MARKET, OrderType.LIMIT])
        
        # Simulate realistic slippage
        signal_price = base_price * (1 + random.gauss(0, 0.001))
        
        # Market orders have more slippage
        if order_type == OrderType.MARKET:
            slippage_factor = random.gauss(0.0005, 0.001)  # ~5 bps avg
        else:
            slippage_factor = random.gauss(-0.0002, 0.0005)  # Often favorable
        
        fill_price = signal_price * (1 + slippage_factor)
        
        quantity = random.uniform(0.01, 1.0)
        
        record = tracker.record_trade(
            trade_id=f"trade_{i}",
            asset=asset,
            side=side,
            order_type=order_type,
            signal_price=signal_price,
            fill_price=fill_price,
            quantity=quantity,
            exchange="binance",
            fee_bps=4.0
        )
        
        if i % 20 == 0 and record:
            print(f"Trade {i}: {asset} {side.value} @ {fill_price:.2f}, "
                  f"Slippage: {record['slippage_bps']:.2f} bps")
    
    # Get final metrics
    print("\n=== Final Metrics ===")
    for asset in ["BTC", "ETH", "SOL"]:
        metrics = tracker.get_metrics(asset)
        if metrics:
            quality = tracker.get_execution_quality_score(asset)
            print(f"\n{asset}:")
            print(f"  Trades: {metrics['total_trades']}")
            print(f"  Avg Slippage: {metrics['avg_slippage_bps']:.2f} bps")
            print(f"  Quality Score: {quality:.1f}/100")
            print(f"  Acceptable: {metrics['pct_acceptable']:.1f}%")
    
    # Compare order types
    print("\n=== Order Type Comparison (BTC) ===")
    comparison = tracker.compare_order_types("BTC")
    for order_type, stats in comparison.items():
        print(f"{order_type}: avg={stats['avg_slippage_bps']:.2f} bps, "
              f"count={stats['count']}")
