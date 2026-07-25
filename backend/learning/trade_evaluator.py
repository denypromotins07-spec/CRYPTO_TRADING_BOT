"""
Trade Evaluator: Deep post-trade analysis on every execution.
Performs comprehensive trade analytics and updates SOUL.md with lessons learned.
Integrates with the self-learning feedback loop for continuous improvement.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeOutcome(Enum):
    """Classification of trade outcomes."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    STOPPED_OUT = "stopped_out"
    LIQUIDATED = "liquidated"
    PARTIAL_WIN = "partial_win"
    PARTIAL_LOSS = "partial_loss"


class TradeQuality(Enum):
    """Quality rating for trade execution."""
    EXCELLENT = 5
    VERY_GOOD = 4
    GOOD = 3
    FAIR = 2
    POOR = 1


@dataclass
class TradeRecord:
    """Complete record of a executed trade."""
    trade_id: str
    asset: str
    side: str  # BUY or SELL
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: float
    exit_time: Optional[float]
    pnl: float
    pnl_pct: float
    fees: float
    slippage: float
    outcome: TradeOutcome
    quality: TradeQuality
    strategy_tags: List[str]
    market_regime: str
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "asset": self.asset,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "fees": self.fees,
            "slippage": self.slippage,
            "outcome": self.outcome.value,
            "quality": self.quality.value,
            "strategy_tags": self.strategy_tags,
            "market_regime": self.market_regime,
            "notes": self.notes,
            "duration_minutes": (self.exit_time - self.entry_time) / 60 if self.exit_time else None
        }


@dataclass
class TradeAnalysis:
    """Analysis results for a trade."""
    trade_id: str
    entry_timing_score: float  # 0-1, how good was entry timing
    exit_timing_score: float   # 0-1, how good was exit timing
    position_size_score: float  # 0-1, was size appropriate
    risk_reward_actual: float
    risk_reward_expected: float
    max_adverse_excursion: float  # Max drawdown during trade
    max_favorable_excursion: float  # Max profit during trade
    hold_time_percentile: float  # How does hold time compare to similar trades
    mistakes_identified: List[str]
    lessons_learned: List[str]
    improvement_suggestions: List[str]


class TradeEvaluator:
    """
    Comprehensive post-trade analysis engine.
    Evaluates every trade and extracts actionable insights for SOUL.md.
    Implements statistical analysis across trade populations.
    """
    
    def __init__(
        self,
        soul_md_path: str = "SOUL.md",
        min_trades_for_analysis: int = 10,
    ):
        self.soul_md_path = Path(soul_md_path)
        self.min_trades_for_analysis = min_trades_for_analysis
        
        # Trade storage
        self._trades: Dict[str, TradeRecord] = {}
        self._trades_by_asset: Dict[str, List[str]] = {}
        self._trades_by_strategy: Dict[str, List[str]] = {}
        
        # Performance metrics
        self._total_pnl = 0.0
        self._win_count = 0
        self._loss_count = 0
        self._total_trades = 0
        
        # Analysis cache
        self._analyses: Dict[str, TradeAnalysis] = {}
        
        logger.info("TradeEvaluator initialized")
    
    def record_trade(
        self,
        trade_id: str,
        asset: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_time: float,
        strategy_tags: Optional[List[str]] = None,
        market_regime: str = "unknown",
    ) -> TradeRecord:
        """Record a new trade (entry only)."""
        trade = TradeRecord(
            trade_id=trade_id,
            asset=asset,
            side=side.upper(),
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            entry_time=entry_time,
            exit_time=None,
            pnl=0.0,
            pnl_pct=0.0,
            fees=0.0,
            slippage=0.0,
            outcome=TradeOutcome.LOSS,  # Will be updated on exit
            quality=TradeQuality.GOOD,
            strategy_tags=strategy_tags or [],
            market_regime=market_regime,
        )
        
        self._trades[trade_id] = trade
        
        # Index by asset
        if asset not in self._trades_by_asset:
            self._trades_by_asset[asset] = []
        self._trades_by_asset[asset].append(trade_id)
        
        # Index by strategy
        for tag in trade.strategy_tags:
            if tag not in self._trades_by_strategy:
                self._trades_by_strategy[tag] = []
            self._trades_by_strategy[tag].append(trade_id)
        
        logger.debug(f"Trade recorded: {trade_id} - {side} {quantity} {asset}")
        
        return trade
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: float,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> TradeRecord:
        """Close a trade and perform initial analysis."""
        if trade_id not in self._trades:
            logger.warning(f"Trade not found: {trade_id}")
            raise ValueError(f"Trade not found: {trade_id}")
        
        trade = self._trades[trade_id]
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.fees = fees
        trade.slippage = slippage
        
        # Calculate PnL
        if trade.side == "BUY":
            gross_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.quantity
        
        trade.pnl = gross_pnl - fees - slippage
        trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.quantity)) * 100 if trade.entry_price > 0 else 0
        
        # Determine outcome
        if abs(trade.pnl_pct) < 0.1:  # Less than 0.1%
            trade.outcome = TradeOutcome.BREAKEVEN
        elif trade.pnl > 0:
            trade.outcome = TradeOutcome.WIN if trade.pnl_pct > 1.0 else TradeOutcome.PARTIAL_WIN
        else:
            trade.outcome = TradeOutcome.LOSS if trade.pnl_pct < -1.0 else TradeOutcome.PARTIAL_LOSS
        
        # Update counters
        self._total_trades += 1
        self._total_pnl += trade.pnl
        
        if trade.pnl > 0:
            self._win_count += 1
        elif trade.pnl < 0:
            self._loss_count += 1
        
        # Perform deep analysis
        analysis = self._analyze_trade(trade)
        self._analyses[trade_id] = analysis
        
        # Extract lessons for SOUL.md
        self._extract_lessons(trade, analysis)
        
        logger.info(f"Trade closed: {trade_id} - PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
        
        return trade
    
    def _analyze_trade(self, trade: TradeRecord) -> TradeAnalysis:
        """Perform deep analysis on a completed trade."""
        mistakes = []
        lessons = []
        suggestions = []
        
        # Entry timing analysis (simplified - would use more sophisticated methods)
        entry_timing_score = 0.7  # Placeholder
        
        # Exit timing analysis
        exit_timing_score = 0.7  # Placeholder
        
        # Position size appropriateness
        position_size_score = 0.8  # Placeholder
        
        # Risk/reward analysis
        risk_reward_expected = 2.0  # Expected R:R from strategy
        actual_risk = abs(trade.entry_price * 0.02)  # Assumed 2% stop
        actual_reward = abs(trade.pnl / trade.quantity) if trade.pnl != 0 else 0
        risk_reward_actual = actual_reward / actual_risk if actual_risk > 0 else 0
        
        # Check for common mistakes
        if trade.pnl < 0 and trade.slippage > trade.entry_price * 0.001:
            mistakes.append("High slippage on entry/exit")
            lessons.append("Consider using limit orders in low liquidity")
            suggestions.append("Implement better order timing logic")
        
        if trade.exit_time and trade.entry_time:
            duration = (trade.exit_time - trade.entry_time) / 60  # minutes
            if duration < 1 and trade.pnl < 0:
                mistakes.append("Premature exit (< 1 minute)")
                lessons.append("Allow trades more time to develop")
                suggestions.append("Review minimum hold time parameters")
        
        if abs(trade.pnl_pct) > 5.0 and trade.pnl < 0:
            mistakes.append("Large loss (> 5%)")
            lessons.append("Review stop-loss placement for this setup")
            suggestions.append("Consider tighter stops or smaller position size")
        
        # Quality assessment
        avg_score = (entry_timing_score + exit_timing_score + position_size_score) / 3
        if avg_score >= 0.8:
            quality = TradeQuality.EXCELLENT
        elif avg_score >= 0.6:
            quality = TradeQuality.VERY_GOOD
        elif avg_score >= 0.4:
            quality = TradeQuality.GOOD
        elif avg_score >= 0.2:
            quality = TradeQuality.FAIR
        else:
            quality = TradeQuality.POOR
        
        trade.quality = quality
        
        return TradeAnalysis(
            trade_id=trade.trade_id,
            entry_timing_score=entry_timing_score,
            exit_timing_score=exit_timing_score,
            position_size_score=position_size_score,
            risk_reward_actual=risk_reward_actual,
            risk_reward_expected=risk_reward_expected,
            max_adverse_excursion=0.0,  # Would calculate from tick data
            max_favorable_excursion=0.0,  # Would calculate from tick data
            hold_time_percentile=0.5,  # Would compare to historical
            mistakes_identified=mistakes,
            lessons_learned=lessons,
            improvement_suggestions=suggestions,
        )
    
    def _extract_lessons(self, trade: TradeRecord, analysis: TradeAnalysis) -> None:
        """Extract lessons and update SOUL.md."""
        if not analysis.lessons_learned and not analysis.mistakes_identified:
            return
        
        # Prepare lesson entry
        lesson_entry = {
            "timestamp": datetime.now().isoformat(),
            "trade_id": trade.trade_id,
            "asset": trade.asset,
            "outcome": trade.outcome.value,
            "pnl_pct": round(trade.pnl_pct, 2),
            "quality": trade.quality.value,
            "mistakes": analysis.mistakes_identified,
            "lessons": analysis.lessons_learned,
            "suggestions": analysis.improvement_suggestions,
        }
        
        # Append to SOUL.md
        self._append_to_soul_md(lesson_entry)
    
    def _append_to_soul_md(self, lesson_entry: Dict[str, Any]) -> None:
        """Append a lesson entry to SOUL.md file."""
        try:
            # Ensure directory exists
            self.soul_md_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read existing content or create new
            if self.soul_md_path.exists():
                content = self.soul_md_path.read_text()
            else:
                content = "# ZAID Bot - SOUL Memory\n\n## Trade Lessons\n\n"
            
            # Format new entry
            entry_text = f"\n### Trade: {lesson_entry['trade_id']} ({lesson_entry['timestamp']})\n"
            entry_text += f"- **Asset**: {lesson_entry['asset']}\n"
            entry_text += f"- **Outcome**: {lesson_entry['outcome']} ({lesson_entry['pnl_pct']}%)\n"
            entry_text += f"- **Quality**: {lesson_entry['quality']}/5\n"
            
            if lesson_entry['mistakes']:
                entry_text += f"- **Mistakes**: {', '.join(lesson_entry['mistakes'])}\n"
            
            if lesson_entry['lessons']:
                entry_text += f"- **Lessons**: {', '.join(lesson_entry['lessons'])}\n"
            
            if lesson_entry['suggestions']:
                entry_text += f"- **Action Items**: {', '.join(lesson_entry['suggestions'])}\n"
            
            # Append and write
            content += entry_text
            self.soul_md_path.write_text(content)
            
            logger.debug(f"Lesson appended to SOUL.md: {lesson_entry['trade_id']}")
            
        except Exception as e:
            logger.error(f"Failed to update SOUL.md: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        win_rate = self._win_count / self._total_trades if self._total_trades > 0 else 0
        
        return {
            "total_trades": self._total_trades,
            "wins": self._win_count,
            "losses": self._loss_count,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(self._total_pnl, 2),
            "avg_pnl_per_trade": round(self._total_pnl / self._total_trades, 2) if self._total_trades > 0 else 0,
        }
    
    def get_asset_performance(self, asset: str) -> Dict[str, Any]:
        """Get performance metrics for a specific asset."""
        if asset not in self._trades_by_asset:
            return {"error": f"No trades for {asset}"}
        
        trade_ids = self._trades_by_asset[asset]
        trades = [self._trades[tid] for tid in trade_ids if tid in self._trades]
        
        if not trades:
            return {"error": f"No completed trades for {asset}"}
        
        total_pnl = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        
        return {
            "asset": asset,
            "trade_count": len(trades),
            "total_pnl": round(total_pnl, 2),
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": round(wins / len(trades), 4) if trades else 0,
        }
    
    def get_strategy_performance(self, strategy_tag: str) -> Dict[str, Any]:
        """Get performance metrics for a specific strategy."""
        if strategy_tag not in self._trades_by_strategy:
            return {"error": f"No trades for strategy {strategy_tag}"}
        
        trade_ids = self._trades_by_strategy[strategy_tag]
        trades = [self._trades[tid] for tid in trade_ids if tid in self._trades]
        
        if not trades:
            return {"error": f"No completed trades for {strategy_tag}"}
        
        total_pnl = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        
        return {
            "strategy": strategy_tag,
            "trade_count": len(trades),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(wins / len(trades), 4) if trades else 0,
        }
    
    def export_analyses(self, output_path: str) -> None:
        """Export all trade analyses to JSON file."""
        analyses_data = {
            tid: {
                "trade": self._trades[tid].to_dict() if tid in self._trades else None,
                "analysis": {
                    "entry_timing_score": a.entry_timing_score,
                    "exit_timing_score": a.exit_timing_score,
                    "position_size_score": a.position_size_score,
                    "risk_reward_actual": a.risk_reward_actual,
                    "mistakes": a.mistakes_identified,
                    "lessons": a.lessons_learned,
                    "suggestions": a.improvement_suggestions,
                }
            }
            for tid, a in self._analyses.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(analyses_data, f, indent=2)
        
        logger.info(f"Exported {len(analyses_data)} trade analyses to {output_path}")


# Singleton instance
_evaluator_instance: Optional[TradeEvaluator] = None


def get_trade_evaluator() -> TradeEvaluator:
    """Get singleton instance of TradeEvaluator."""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = TradeEvaluator()
    return _evaluator_instance


if __name__ == "__main__":
    # Example usage
    evaluator = get_trade_evaluator()
    
    # Record a sample trade
    trade = evaluator.record_trade(
        trade_id="TEST-001",
        asset="BTCUSDT",
        side="BUY",
        entry_price=50000.0,
        quantity=0.1,
        entry_time=time.time(),
        strategy_tags=["momentum", "breakout"],
        market_regime="trending"
    )
    
    # Close the trade
    time.sleep(0.1)
    evaluator.close_trade(
        trade_id="TEST-001",
        exit_price=51000.0,
        exit_time=time.time(),
        fees=5.0,
        slippage=2.0
    )
    
    # Get performance summary
    summary = evaluator.get_performance_summary()
    print(f"Performance Summary: {summary}")
    
    # Get asset performance
    asset_perf = evaluator.get_asset_performance("BTCUSDT")
    print(f"BTCUSDT Performance: {asset_perf}")
    
    print("\nTrade Evaluator module initialized successfully.")
