#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Order Book Microstructure
Chapter 2: Spread Analyzer

This module tracks tick-by-tick spread widening patterns to detect
liquidity stress, market maker withdrawal, and optimal entry/exit windows.
It provides early warning signals for deteriorating market conditions.

Memory Budget: <50MB for spread history
Target Latency: <100μs for spread analysis
Assets: BTC, SOL, ETH, USDT parallel processing
Integration: Alerts circuit breaker during extreme spread events

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


class SpreadState(Enum):
    """Market spread condition states."""
    NORMAL = "normal"
    WIDENING = "widening"
    STRESSED = "stressed"
    CRITICAL = "critical"
    ILLIQUID = "illiquid"


@dataclass
class SpreadConfig:
    """Configuration for spread analysis."""
    assets: List[str] = field(default_factory=lambda: ["BTC", "SOL", "ETH", "USDT"])
    history_size: int = 1000  # Number of ticks to retain
    normal_spread_threshold: float = 0.0005  # 0.05% baseline
    warning_spread_threshold: float = 0.002  # 0.2% warning
    critical_spread_threshold: float = 0.01  # 1% critical
    widening_rate_threshold: float = 0.0001  # Rate of change threshold per tick
    sampling_interval_ms: int = 100  # Minimum ms between samples


@dataclass
class SpreadMetrics(TypedDict):
    """Type-safe spread metrics structure."""
    asset: str
    bid: float
    ask: float
    mid_price: float
    absolute_spread: float
    relative_spread: float
    spread_state: str
    widening_rate: float
    z_score: float
    percentile_rank: float
    timestamp: float


@dataclass
class SpreadAlert(TypedDict):
    """Alert triggered by spread anomaly."""
    asset: str
    alert_type: str
    severity: int  # 1-5 scale
    message: str
    spread_value: float
    historical_avg: float
    timestamp: float


class SpreadAnalyzer:
    """
    Real-time spread analyzer with multi-timescale monitoring.
    
    Features:
    - Tick-by-tick spread tracking
    - Statistical deviation detection (z-score)
    - Rate-of-change monitoring for rapid widening
    - Historical percentile ranking
    - Multi-threshold alerting system
    """
    
    __slots__ = (
        '_config', '_spread_history', '_current_quotes',
        '_baseline_stats', '_alert_callbacks'
    )
    
    def __init__(self, config: SpreadConfig = None) -> None:
        """
        Initialize spread analyzer.
        
        Args:
            config: Spread analysis configuration
        """
        self._config = config or SpreadConfig()
        
        # Per-asset spread history
        self._spread_history: Dict[str, Deque[float]] = {
            asset: deque(maxlen=self._config.history_size)
            for asset in self._config.assets
        }
        
        # Current best bid/ask per asset
        self._current_quotes: Dict[str, Tuple[float, float]] = {}
        
        # Baseline statistics (computed from history)
        self._baseline_stats: Dict[str, Dict[str, float]] = {
            asset: {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            for asset in self._config.assets
        }
        
        # Alert callback functions
        self._alert_callbacks: List[callable] = []
    
    def register_alert_callback(self, callback: callable) -> None:
        """Register a function to be called when alerts are generated."""
        self._alert_callbacks.append(callback)
    
    def update_quote(self, asset: str, bid: float, ask: float) -> Optional[SpreadMetrics]:
        """
        Update quote and analyze spread.
        
        Args:
            asset: Asset identifier
            bid: Best bid price
            ask: Best ask price
            
        Returns:
            SpreadMetrics or None if invalid data
        """
        if bid <= 0 or ask <= 0 or bid >= ask:
            return None
        
        # Store current quote
        self._current_quotes[asset] = (bid, ask)
        
        # Calculate spread metrics
        absolute_spread = ask - bid
        mid_price = (bid + ask) / 2.0
        relative_spread = absolute_spread / mid_price if mid_price > 0 else 0.0
        
        # Add to history
        self._spread_history[asset].append(relative_spread)
        
        # Update baseline statistics
        self._update_baseline(asset)
        
        # Calculate statistical measures
        z_score = self._calculate_z_score(asset, relative_spread)
        percentile = self._calculate_percentile(asset, relative_spread)
        widening_rate = self._calculate_widening_rate(asset)
        
        # Determine spread state
        spread_state = self._determine_spread_state(
            relative_spread, widening_rate, z_score
        )
        
        metrics = SpreadMetrics(
            asset=asset,
            bid=bid,
            ask=ask,
            mid_price=mid_price,
            absolute_spread=absolute_spread,
            relative_spread=relative_spread,
            spread_state=spread_state.value,
            widening_rate=widening_rate,
            z_score=z_score,
            percentile_rank=percentile,
            timestamp=time.time()
        )
        
        # Check for alert conditions
        self._check_alerts(asset, metrics)
        
        return metrics
    
    def _update_baseline(self, asset: str) -> None:
        """Update baseline statistics from recent history."""
        history = list(self._spread_history[asset])
        
        if len(history) < 10:
            return
        
        self._baseline_stats[asset] = {
            'mean': statistics.mean(history),
            'std': statistics.stdev(history) if len(history) > 1 else 0.0,
            'median': statistics.median(history)
        }
    
    def _calculate_z_score(self, asset: str, current_spread: float) -> float:
        """Calculate z-score of current spread vs historical."""
        stats = self._baseline_stats.get(asset, {})
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 0.0)
        
        if std == 0 or std < 1e-10:
            return 0.0
        
        return (current_spread - mean) / std
    
    def _calculate_percentile(self, asset: str, current_spread: float) -> float:
        """Calculate percentile rank of current spread in history."""
        history = list(self._spread_history[asset])
        
        if len(history) < 10:
            return 0.5
        
        count_below = sum(1 for s in history if s < current_spread)
        return count_below / len(history)
    
    def _calculate_widening_rate(self, asset: str) -> float:
        """Calculate rate of spread widening (change per tick)."""
        history = list(self._spread_history[asset])
        
        if len(history) < 2:
            return 0.0
        
        # Rate of change over last 5 ticks
        window = min(5, len(history))
        recent = history[-window:]
        
        if len(recent) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _determine_spread_state(self, relative_spread: float, 
                                  widening_rate: float, 
                                  z_score: float) -> SpreadState:
        """Determine current spread condition state."""
        # Check absolute thresholds first
        if relative_spread >= self._config.critical_spread_threshold:
            return SpreadState.CRITICAL
        
        if relative_spread >= self._config.warning_spread_threshold:
            return SpreadState.STRESSED
        
        # Check rate of widening
        if widening_rate > self._config.widening_rate_threshold * 10:
            return SpreadState.WIDENING
        
        # Check statistical deviation
        if z_score > 3.0:
            return SpreadState.STRESSED
        elif z_score > 2.0:
            return SpreadState.WIDENING
        
        return SpreadState.NORMAL
    
    def _check_alerts(self, asset: str, metrics: SpreadMetrics) -> None:
        """Generate alerts for anomalous spread conditions."""
        alerts = []
        
        # Critical spread alert
        if metrics['spread_state'] == SpreadState.CRITICAL.value:
            alerts.append(SpreadAlert(
                asset=asset,
                alert_type="CRITICAL_SPREAD",
                severity=5,
                message=f"Critical spread detected: {metrics['relative_spread']:.4%}",
                spread_value=metrics['relative_spread'],
                historical_avg=self._baseline_stats[asset]['mean'],
                timestamp=time.time()
            ))
        
        # Rapid widening alert
        if metrics['widening_rate'] > self._config.widening_rate_threshold * 5:
            alerts.append(SpreadAlert(
                asset=asset,
                alert_type="RAPID_WIDENING",
                severity=4,
                message=f"Spread widening rapidly: {metrics['widening_rate']:.6f}/tick",
                spread_value=metrics['relative_spread'],
                historical_avg=self._baseline_stats[asset]['mean'],
                timestamp=time.time()
            ))
        
        # Statistical outlier alert
        if metrics['z_score'] > 4.0:
            alerts.append(SpreadAlert(
                asset=asset,
                alert_type="STATISTICAL_OUTLIER",
                severity=3,
                message=f"Spread z-score: {metrics['z_score']:.2f}",
                spread_value=metrics['relative_spread'],
                historical_avg=self._baseline_stats[asset]['mean'],
                timestamp=time.time()
            ))
        
        # Notify callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass  # Don't let callback errors disrupt analysis
    
    def get_spread_metrics(self, asset: str) -> Optional[SpreadMetrics]:
        """Get latest spread metrics for an asset."""
        if asset not in self._current_quotes:
            return None
        
        bid, ask = self._current_quotes[asset]
        return self.update_quote(asset, bid, ask)
    
    def get_all_assets_metrics(self) -> Dict[str, SpreadMetrics]:
        """Get spread metrics for all tracked assets."""
        metrics = {}
        for asset, (bid, ask) in self._current_quotes.items():
            result = self.get_spread_metrics(asset)
            if result:
                metrics[asset] = result
        return metrics
    
    def is_liquid(self, asset: str) -> bool:
        """Check if asset has acceptable liquidity (spread within normal range)."""
        metrics = self.get_spread_metrics(asset)
        if not metrics:
            return False
        
        return metrics['spread_state'] in [
            SpreadState.NORMAL.value,
            SpreadState.WIDENING.value
        ]
    
    def get_optimal_entry_window(self, asset: str) -> bool:
        """
        Determine if current conditions favor order entry.
        
        Returns True when:
        - Spread is at or below historical median
        - Spread is not widening
        - Market is in normal state
        """
        metrics = self.get_spread_metrics(asset)
        if not metrics:
            return False
        
        median = self._baseline_stats[asset].get('median', float('inf'))
        
        return (
            metrics['relative_spread'] <= median and
            metrics['widening_rate'] <= 0 and
            metrics['spread_state'] == SpreadState.NORMAL.value
        )
    
    def get_spread_statistics(self, asset: str, window: int = 100) -> Dict[str, float]:
        """Get detailed spread statistics for an asset."""
        history = list(self._spread_history[asset])
        
        if len(history) < 2:
            return {}
        
        # Use specified window
        recent = history[-window:] if len(history) >= window else history
        
        return {
            'mean': statistics.mean(recent),
            'median': statistics.median(recent),
            'std': statistics.stdev(recent),
            'min': min(recent),
            'max': max(recent),
            'current': recent[-1] if recent else 0.0,
            'percentile_95': np.percentile(recent, 95),
            'percentile_99': np.percentile(recent, 99)
        }
    
    def export_to_soul_md(self, output_path: str = "SOUL.md") -> None:
        """Export spread analysis summary to SOUL.md."""
        import os
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            f"\n## Spread Analysis Report - {timestamp}",
            "",
            "| Asset | State | Rel Spread | Z-Score | Widening Rate |",
            "|-------|-------|------------|---------|---------------|"
        ]
        
        for asset in self._config.assets:
            metrics = self.get_spread_metrics(asset)
            if metrics:
                lines.append(
                    f"| {asset} | {metrics['spread_state']} | "
                    f"{metrics['relative_spread']:.4%} | "
                    f"{metrics['z_score']:.2f} | "
                    f"{metrics['widening_rate']:.6f} |"
                )
        
        lines.extend([
            "",
            "### Spread Lessons",
            "- Wide spreads indicate low liquidity or high volatility",
            "- Avoid market orders during STRESSED or CRITICAL states",
            "- Use limit orders and patience during spread widening",
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


class CrossAssetSpreadMonitor(SpreadAnalyzer):
    """
    Monitor spread relationships across multiple assets.
    
    Detects correlated spread widening that may indicate
    systemic liquidity stress or exchange-wide issues.
    """
    
    __slots__ = ('_correlation_matrix', '_systemic_alert_threshold')
    
    def __init__(self, config: SpreadConfig = None, systemic_threshold: float = 0.7) -> None:
        """
        Initialize cross-asset monitor.
        
        Args:
            config: Base spread configuration
            systemic_threshold: Correlation threshold for systemic alert
        """
        super().__init__(config)
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}
        self._systemic_alert_threshold = systemic_threshold
    
    def update_and_check_systemic(self, quotes: Dict[str, Tuple[float, float]]) -> bool:
        """
        Update all quotes and check for systemic spread stress.
        
        Args:
            quotes: Dictionary of asset -> (bid, ask)
            
        Returns:
            True if systemic stress detected
        """
        # Update all quotes
        for asset, (bid, ask) in quotes.items():
            self.update_quote(asset, bid, ask)
        
        # Count assets in stressed state
        stressed_count = 0
        total_count = 0
        
        for asset in quotes.keys():
            metrics = self.get_spread_metrics(asset)
            if metrics:
                total_count += 1
                if metrics['spread_state'] in [
                    SpreadState.STRESSED.value,
                    SpreadState.CRITICAL.value
                ]:
                    stressed_count += 1
        
        # Systemic stress if >70% of assets are stressed
        return total_count > 0 and (stressed_count / total_count) > 0.7
    
    def calculate_spread_correlations(self) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise spread correlations between assets."""
        correlations = {}
        assets = list(self._spread_history.keys())
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                hist1 = list(self._spread_history[asset1])
                hist2 = list(self._spread_history[asset2])
                
                # Align lengths
                min_len = min(len(hist1), len(hist2))
                if min_len < 20:
                    continue
                
                arr1 = np.array(hist1[-min_len:])
                arr2 = np.array(hist2[-min_len:])
                
                # Calculate correlation
                if np.std(arr1) > 0 and np.std(arr2) > 0:
                    corr = np.corrcoef(arr1, arr2)[0, 1]
                    correlations[(asset1, asset2)] = corr if not np.isnan(corr) else 0.0
        
        self._correlation_matrix = correlations
        return correlations


if __name__ == "__main__":
    # Example usage and validation
    config = SpreadConfig(
        normal_spread_threshold=0.0005,
        warning_spread_threshold=0.002,
        critical_spread_threshold=0.01
    )
    
    analyzer = SpreadAnalyzer(config)
    
    # Simulate normal market conditions
    print("=== Normal Market Conditions ===")
    base_price = 50000.0
    for i in range(100):
        spread_pct = 0.0003 + (np.random.random() * 0.0002)  # 0.03-0.05%
        spread = base_price * spread_pct
        bid = base_price - spread / 2
        ask = base_price + spread / 2
        
        metrics = analyzer.update_quote("BTC", bid, ask)
        
        if i % 20 == 0 and metrics:
            print(f"Tick {i}: Spread={metrics['relative_spread']:.4%}, "
                  f"State={metrics['spread_state']}, Z={metrics['z_score']:.2f}")
    
    # Simulate stress event
    print("\n=== Stress Event Simulation ===")
    for i in range(20):
        # Spread widens progressively
        spread_pct = 0.001 + (i * 0.002)  # Widening from 0.1% to 4%
        spread = base_price * spread_pct
        bid = base_price - spread / 2
        ask = base_price + spread / 2
        
        metrics = analyzer.update_quote("BTC", bid, ask)
        
        if metrics:
            print(f"Stress {i}: Spread={metrics['relative_spread']:.4%}, "
                  f"State={metrics['spread_state']}, Z={metrics['z_score']:.2f}")
    
    # Get final statistics
    print("\n=== Final Statistics ===")
    stats = analyzer.get_spread_statistics("BTC")
    if stats:
        print(f"Mean: {stats['mean']:.4%}")
        print(f"Median: {stats['median']:.4%}")
        print(f"Std: {stats['std']:.4%}")
        print(f"Max: {stats['max']:.4%}")
