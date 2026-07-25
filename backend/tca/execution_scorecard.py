#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Transaction Cost Analysis (TCA)
Chapter 4: Execution Scorecard

This module grades trades and execution quality, feeding results directly
to SOUL.md for continuous learning and improvement. It provides comprehensive
performance attribution and identifies areas for execution optimization.

Memory Budget: <50MB for scorecard data
Target Latency: <100μs per trade grading
Assets: BTC, SOL, ETH, USDT parallel evaluation
Integration: Direct updates to SOUL.md with slippage lessons

Author: Opus 4.8
Stage: 5/100 - Advanced Risk Management and Order Book Microstructure
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import statistics
import json


class Grade(Enum):
    """Execution quality grades."""
    A_PLUS = "A+"  # Exceptional execution
    A = "A"        # Excellent
    A_MINUS = "A-" # Very good
    B_PLUS = "B+"  # Good
    B = "B"        # Average
    B_MINUS = "B-" # Below average
    C = "C"        # Poor
    D = "D"        # Very poor
    F = "F"        # Failed execution


@dataclass
class ScorecardConfig:
    """Configuration for execution scorecard."""
    assets: List[str] = field(default_factory=lambda: ["BTC", "SOL", "ETH", "USDT"])
    grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'A+': 95.0,
        'A': 90.0,
        'A-': 85.0,
        'B+': 80.0,
        'B': 70.0,
        'B-': 60.0,
        'C': 50.0,
        'D': 40.0,
    })
    benchmark_type: str = "arrival_price"  # arrival_price, vwap, twap
    soul_md_path: str = "SOUL.md"


@dataclass
class TradeEvaluation(TypedDict):
    """Complete trade evaluation record."""
    trade_id: str
    asset: str
    side: str
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    total_costs_usd: float
    slippage_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    execution_score: float
    grade: str
    lessons: List[str]


@dataclass
class AssetScorecard(TypedDict):
    """Scorecard summary for an asset."""
    asset: str
    total_trades: int
    win_rate: float
    avg_execution_score: float
    avg_slippage_bps: float
    avg_market_impact_bps: float
    grade_distribution: Dict[str, int]
    best_trade_pnl: float
    worst_trade_pnl: float
    consistency_score: float
    recommendations: List[str]


class ExecutionScorecard:
    """
    Comprehensive execution quality grader and learning system.
    
    Features:
    - Multi-factor execution scoring
    - Grade assignment based on performance
    - Lesson extraction from each trade
    - SOUL.md integration for knowledge persistence
    - Performance attribution analysis
    - Continuous improvement recommendations
    """
    
    __slots__ = (
        '_config', '_trade_evaluations', '_asset_scorecards',
        '_grade_history', '_learning_points'
    )
    
    def __init__(self, config: ScorecardConfig = None) -> None:
        """
        Initialize execution scorecard.
        
        Args:
            config: Scorecard configuration
        """
        self._config = config or ScorecardConfig()
        
        # Trade evaluations per asset
        self._trade_evaluations: Dict[str, List[TradeEvaluation]] = {
            asset: [] for asset in self._config.assets
        }
        
        # Asset-level scorecards
        self._asset_scorecards: Dict[str, AssetScorecard] = {}
        
        # Grade history for trend analysis
        self._grade_history: Dict[str, List[str]] = {
            asset: [] for asset in self._config.assets
        }
        
        # Learning points extracted from trades
        self._learning_points: List[Dict[str, Any]] = []
    
    def evaluate_trade(
        self,
        trade_id: str,
        asset: str,
        side: str,
        entry_time: float,
        exit_time: float,
        entry_price: float,
        exit_price: float,
        quantity: float,
        signal_price: float,
        vwap_price: Optional[float] = None,
        fees_usd: float = 0.0,
        exchange: str = "default"
    ) -> Optional[TradeEvaluation]:
        """
        Evaluate a completed trade and assign grade.
        
        Args:
            trade_id: Unique trade identifier
            asset: Asset traded
            side: BUY or SELL
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry fill price
            exit_price: Exit fill price
            quantity: Trade quantity
            signal_price: Price when signal generated
            vwap_price: VWAP during execution window
            fees_usd: Total fees paid
            exchange: Exchange name
            
        Returns:
            TradeEvaluation or None if invalid
        """
        if asset not in self._trade_evaluations:
            return None
        
        # Calculate P&L
        if side.upper() == "BUY":
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity
        
        # Calculate costs
        notional = entry_price * quantity
        
        # Slippage from signal
        if side.upper() == "BUY":
            slippage_bps = ((entry_price - signal_price) / signal_price) * 10000
        else:
            slippage_bps = ((signal_price - entry_price) / signal_price) * 10000
        
        # Market impact estimate (simplified)
        market_impact_bps = abs(slippage_bps) * 0.5  # Assume 50% is impact
        
        # Timing cost (vs VWAP)
        if vwap_price:
            if side.upper() == "BUY":
                timing_cost_bps = ((entry_price - vwap_price) / vwap_price) * 10000
            else:
                timing_cost_bps = ((vwap_price - entry_price) / vwap_price) * 10000
        else:
            timing_cost_bps = 0.0
        
        # Total costs
        total_costs_usd = fees_usd + (abs(slippage_bps) / 10000) * notional
        
        # Calculate execution score (0-100)
        execution_score = self._calculate_execution_score(
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            trade_duration=exit_time - entry_time,
            pnl=gross_pnl,
            notional=notional
        )
        
        # Assign grade
        grade = self._assign_grade(execution_score)
        
        # Extract lessons
        lessons = self._extract_lessons(
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            grade=grade,
            side=side
        )
        
        evaluation = TradeEvaluation(
            trade_id=trade_id,
            asset=asset,
            side=side.upper(),
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=gross_pnl,
            total_costs_usd=total_costs_usd,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            execution_score=execution_score,
            grade=grade,
            lessons=lessons
        )
        
        # Store evaluation
        self._trade_evaluations[asset].append(evaluation)
        self._grade_history[asset].append(grade)
        
        # Update asset scorecard
        self._update_asset_scorecard(asset)
        
        # Record learning point
        if lessons:
            self._learning_points.append({
                'trade_id': trade_id,
                'asset': asset,
                'timestamp': time.time(),
                'lessons': lessons,
                'grade': grade
            })
        
        return evaluation
    
    def _calculate_execution_score(
        self,
        slippage_bps: float,
        market_impact_bps: float,
        timing_cost_bps: float,
        trade_duration: float,
        pnl: float,
        notional: float
    ) -> float:
        """
        Calculate composite execution score (0-100).
        
        Components:
        - Slippage score (40% weight)
        - Market impact score (25% weight)
        - Timing score (20% weight)
        - Efficiency score (15% weight)
        """
        # Slippage score (lower is better, target <5 bps)
        slippage_score = max(0, 100 - abs(slippage_bps) * 10)
        
        # Market impact score (lower is better)
        impact_score = max(0, 100 - market_impact_bps * 15)
        
        # Timing score (lower is better)
        timing_score = max(0, 100 - abs(timing_cost_bps) * 8)
        
        # Efficiency score (P&L relative to costs)
        if notional > 0:
            cost_ratio = abs(pnl) / notional if pnl != 0 else 0
            efficiency_score = min(100, cost_ratio * 1000)
        else:
            efficiency_score = 50
        
        # Weighted average
        total_score = (
            slippage_score * 0.40 +
            impact_score * 0.25 +
            timing_score * 0.20 +
            efficiency_score * 0.15
        )
        
        return min(100, max(0, total_score))
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score."""
        thresholds = self._config.grade_thresholds
        
        if score >= thresholds['A+']:
            return 'A+'
        elif score >= thresholds['A']:
            return 'A'
        elif score >= thresholds['A-']:
            return 'A-'
        elif score >= thresholds['B+']:
            return 'B+'
        elif score >= thresholds['B']:
            return 'B'
        elif score >= thresholds['B-']:
            return 'B-'
        elif score >= thresholds['C']:
            return 'C'
        elif score >= thresholds['D']:
            return 'D'
        else:
            return 'F'
    
    def _extract_lessons(
        self,
        slippage_bps: float,
        market_impact_bps: float,
        timing_cost_bps: float,
        grade: str,
        side: str
    ) -> List[str]:
        """Extract actionable lessons from trade execution."""
        lessons = []
        
        # Slippage lessons
        if abs(slippage_bps) > 20:
            lessons.append(
                f"High slippage ({slippage_bps:.1f} bps): Consider using limit orders "
                f"or splitting order into smaller chunks"
            )
        elif slippage_bps < -5:
            lessons.append(
                f"Favorable slippage ({slippage_bps:.1f} bps): Good timing, "
                f"continue monitoring liquidity before entry"
            )
        
        # Market impact lessons
        if market_impact_bps > 10:
            lessons.append(
                f"Significant market impact ({market_impact_bps:.1f} bps): "
                f"Reduce order size or use TWAP/VWAP execution strategy"
            )
        
        # Timing lessons
        if abs(timing_cost_bps) > 15:
            side_str = "buying" if side.upper() == "BUY" else "selling"
            if timing_cost_bps > 0:
                lessons.append(
                    f"Poor timing on {side_str}: Price moved against you after entry. "
                    f"Consider waiting for confirmation signals"
                )
            else:
                lessons.append(
                    f"Good timing on {side_str}: Entered before favorable move"
                )
        
        # Grade-based lessons
        if grade in ['F', 'D', 'C']:
            lessons.append(
                "Poor execution grade: Review entry timing, order type selection, "
                "and market conditions. Consider paper trading this setup."
            )
        elif grade in ['A+', 'A']:
            lessons.append(
                "Excellent execution: Document this setup and conditions for "
                "future replication. Consider increasing position size gradually."
            )
        
        return lessons
    
    def _update_asset_scorecard(self, asset: str) -> None:
        """Update aggregate scorecard for an asset."""
        evaluations = self._trade_evaluations.get(asset, [])
        
        if not evaluations:
            return
        
        # Calculate metrics
        total_trades = len(evaluations)
        winning_trades = sum(1 for e in evaluations if e['gross_pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_execution_score = statistics.mean(e['execution_score'] for e in evaluations)
        avg_slippage = statistics.mean(abs(e['slippage_bps']) for e in evaluations)
        avg_impact = statistics.mean(e['market_impact_bps'] for e in evaluations)
        
        # Grade distribution
        grade_dist = {}
        for e in evaluations:
            grade = e['grade']
            grade_dist[grade] = grade_dist.get(grade, 0) + 1
        
        # P&L extremes
        pnls = [e['gross_pnl'] for e in evaluations]
        best_trade = max(pnls)
        worst_trade = min(pnls)
        
        # Consistency score (inverse of score variance)
        scores = [e['execution_score'] for e in evaluations]
        if len(scores) > 1:
            score_std = statistics.stdev(scores)
            consistency_score = max(0, 100 - score_std * 2)
        else:
            consistency_score = 100
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_slippage=avg_slippage,
            avg_impact=avg_impact,
            win_rate=win_rate,
            consistency_score=consistency_score,
            grade_dist=grade_dist
        )
        
        self._asset_scorecards[asset] = AssetScorecard(
            asset=asset,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_execution_score=avg_execution_score,
            avg_slippage_bps=avg_slippage,
            avg_market_impact_bps=avg_impact,
            grade_distribution=grade_dist,
            best_trade_pnl=best_trade,
            worst_trade_pnl=worst_trade,
            consistency_score=consistency_score,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        avg_slippage: float,
        avg_impact: float,
        win_rate: float,
        consistency_score: float,
        grade_dist: Dict[str, int]
    ) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        if avg_slippage > 15:
            recommendations.append(
                "High average slippage: Switch to limit orders or implement "
                "more sophisticated execution algorithms (TWAP/VWAP)"
            )
        
        if avg_impact > 8:
            recommendations.append(
                "Significant market impact: Reduce position sizes or split "
                "orders across multiple venues/time periods"
            )
        
        if win_rate < 0.4:
            recommendations.append(
                "Low win rate: Review entry signal quality and consider "
                "adding additional confirmation filters"
            )
        
        if consistency_score < 60:
            recommendations.append(
                "Inconsistent execution: Standardize entry procedures and "
                "avoid emotional trading decisions"
            )
        
        # Check for too many poor grades
        poor_grades = sum(grade_dist.get(g, 0) for g in ['F', 'D', 'C'])
        total = sum(grade_dist.values())
        if total > 0 and poor_grades / total > 0.3:
            recommendations.append(
                "Too many poor executions: Consider reducing trading frequency "
                "and focusing only on highest-conviction setups"
            )
        
        return recommendations
    
    def get_asset_scorecard(self, asset: str) -> Optional[AssetScorecard]:
        """Get current scorecard for an asset."""
        return self._asset_scorecards.get(asset)
    
    def export_to_soul_md(self, output_path: str = None) -> None:
        """Export comprehensive scorecard to SOUL.md."""
        import os
        
        path = output_path or self._config.soul_md_path
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            f"\n## Execution Scorecard Report - {timestamp}",
            "",
            "### Asset Performance Summary",
            "",
            "| Asset | Trades | Win Rate | Avg Score | Avg Slippage | Grade |",
            "|-------|--------|----------|-----------|--------------|-------|"
        ]
        
        for asset in self._config.assets:
            scorecard = self.get_asset_scorecard(asset)
            if scorecard:
                # Get most common grade
                grade_dist = scorecard['grade_distribution']
                if grade_dist:
                    most_common_grade = max(grade_dist.keys(), key=lambda k: grade_dist[k])
                else:
                    most_common_grade = 'N/A'
                
                lines.append(
                    f"| {asset} | {scorecard['total_trades']} | "
                    f"{scorecard['win_rate']:.1%} | "
                    f"{scorecard['avg_execution_score']:.1f} | "
                    f"{scorecard['avg_slippage_bps']:.1f} bps | "
                    f"{most_common_grade} |"
                )
        
        # Add recent lessons
        lines.extend([
            "",
            "### Recent Trading Lessons",
            ""
        ])
        
        recent_lessons = self._learning_points[-10:]  # Last 10 lessons
        for lesson in recent_lessons:
            if lesson['lessons']:
                lines.append(f"- **{lesson['asset']}** ({lesson['grade']}): "
                           f"{lesson['lessons'][0]}")
        
        # Add recommendations
        lines.extend([
            "",
            "### Improvement Recommendations",
            ""
        ])
        
        all_recommendations = set()
        for asset in self._config.assets:
            scorecard = self.get_asset_scorecard(asset)
            if scorecard:
                all_recommendations.update(scorecard['recommendations'])
        
        for rec in all_recommendations:
            lines.append(f"- {rec}")
        
        # Write to file
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = "# ZAID BOT SOUL.md - Trading Intelligence\n\n"
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content + '\n'.join(lines))
        except Exception as e:
            print(f"Warning: Could not write to SOUL.md: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get aggregated portfolio-level execution summary."""
        all_evaluations = []
        for evals in self._trade_evaluations.values():
            all_evaluations.extend(evals)
        
        if not all_evaluations:
            return {}
        
        total_trades = len(all_evaluations)
        total_pnl = sum(e['gross_pnl'] for e in all_evaluations)
        total_costs = sum(e['total_costs_usd'] for e in all_evaluations)
        avg_score = statistics.mean(e['execution_score'] for e in all_evaluations)
        
        return {
            'total_trades': total_trades,
            'total_pnl_usd': total_pnl,
            'total_costs_usd': total_costs,
            'net_pnl_usd': total_pnl - total_costs,
            'avg_execution_score': avg_score,
            'cost_drag_pct': (total_costs / abs(total_pnl) * 100) if total_pnl != 0 else 0,
            'assets_traded': len([a for a in self._config.assets 
                                  if self._trade_evaluations[a]])
        }


if __name__ == "__main__":
    # Example usage and validation
    config = ScorecardConfig()
    scorecard = ExecutionScorecard(config)
    
    import random
    
    print("=== Evaluating Sample Trades ===")
    
    base_price = 50000.0
    
    for i in range(20):
        asset = random.choice(["BTC", "ETH", "SOL"])
        side = random.choice(["BUY", "SELL"])
        
        signal_price = base_price * (1 + random.gauss(0, 0.001))
        entry_price = signal_price * (1 + random.gauss(0, 0.0005))
        exit_price = entry_price * (1 + random.gauss(0, 0.02))
        
        vwap = signal_price * (1 + random.gauss(0, 0.0003))
        
        evaluation = scorecard.evaluate_trade(
            trade_id=f"trade_{i}",
            asset=asset,
            side=side,
            entry_time=time.time() - random.uniform(0, 3600),
            exit_time=time.time(),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=random.uniform(0.1, 1.0),
            signal_price=signal_price,
            vwap_price=vwap,
            fees_usd=random.uniform(1, 10)
        )
        
        if evaluation and i % 5 == 0:
            print(f"\nTrade {i}: {evaluation['asset']} {evaluation['side']}")
            print(f"  Grade: {evaluation['grade']}")
            print(f"  Score: {evaluation['execution_score']:.1f}")
            print(f"  P&L: ${evaluation['gross_pnl']:.2f}")
            if evaluation['lessons']:
                print(f"  Lesson: {evaluation['lessons'][0]}")
    
    # Get portfolio summary
    print("\n=== Portfolio Summary ===")
    summary = scorecard.get_portfolio_summary()
    if summary:
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Net P&L: ${summary['net_pnl_usd']:.2f}")
        print(f"Avg Execution Score: {summary['avg_execution_score']:.1f}")
        print(f"Cost Drag: {summary['cost_drag_pct']:.2f}%")
    
    # Export to SOUL.md
    scorecard.export_to_soul_md()
    print("\nScorecard exported to SOUL.md")
