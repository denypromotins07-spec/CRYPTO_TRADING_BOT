"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
Chapter 4: Backtest Analytics, Equity Curve Analysis, and SOUL.md Integration

File: backend/analytics/soul_backtest_writer.py
Purpose: Store optimization lessons into SOUL.md for continuous learning.
Features:
    - Automatic lesson extraction from backtest failures
    - Mathematical adjustment recommendations
    - Pattern recognition across failed strategies
    - Persistent knowledge base updates
"""

import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestLesson:
    """A single lesson learned from backtest analysis."""
    timestamp: str
    strategy_name: str
    lesson_type: str  # "overfitting", "risk_management", "execution", "market_regime"
    severity: str  # "critical", "warning", "info"
    
    # Problem description
    problem_description: str
    
    # Mathematical details
    mathematical_analysis: Dict[str, Any]
    
    # Recommended adjustments
    recommended_adjustments: List[str]
    
    # Confidence in the lesson
    confidence_score: float
    
    # Related patterns
    related_patterns: List[str]
    
    # Whether this lesson has been applied
    applied_to_live: bool = False


@dataclass
class StrategyProfile:
    """Profile of a strategy's characteristics."""
    strategy_name: str
    avg_sharpe: float
    avg_drawdown: float
    win_rate: float
    profit_factor: float
    sensitivity_to_params: float
    robustness_score: float
    market_regimes_tested: List[str]
    failure_modes: List[str]


class SoulBacktestWriter:
    """
    Writes backtest lessons to SOUL.md knowledge base.
    Implements pattern recognition to extract actionable insights.
    """
    
    __slots__ = [
        'soul_file',
        'lessons',
        'strategy_profiles',
        'pattern_database',
        'last_update_hash'
    ]
    
    def __init__(self, soul_path: str = "./SOUL.md"):
        self.soul_file = Path(soul_path)
        self.lessons: List[BacktestLesson] = []
        self.strategy_profiles: Dict[str, StrategyProfile] = {}
        self.pattern_database: Dict[str, int] = {}
        self.last_update_hash: Optional[str] = None
        
        # Load existing SOUL.md if present
        self._load_existing_soul()
    
    def _load_existing_soul(self) -> None:
        """Load existing lessons from SOUL.md file."""
        if not self.soul_file.exists():
            logger.info("Creating new SOUL.md file")
            self._initialize_soul_file()
            return
        
        try:
            content = self.soul_file.read_text()
            
            # Parse existing lessons (simple JSON block extraction)
            import re
            lesson_blocks = re.findall(
                r'<!-- LESSON_START -->(.*?)<!-- LESSON_END -->',
                content,
                re.DOTALL
            )
            
            for block in lesson_blocks:
                try:
                    lesson_data = json.loads(block.strip())
                    lesson = BacktestLesson(**lesson_data)
                    self.lessons.append(lesson)
                    
                    # Update pattern database
                    for pattern in lesson.related_patterns:
                        self.pattern_database[pattern] = \
                            self.pattern_database.get(pattern, 0) + 1
                    
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse lesson block: {e}")
            
            logger.info(f"Loaded {len(self.lessons)} existing lessons from SOUL.md")
            
        except Exception as e:
            logger.error(f"Error loading SOUL.md: {e}")
            self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Create initial SOUL.md structure."""
        initial_content = """# ZAID PERSONAL CRYPTO TRADING BOT - SOUL.md

## Strategic Optimization & Understanding Ledger

This file contains the accumulated wisdom from backtesting, optimization, and live trading.
Each lesson is extracted from strategy failures and successes to continuously improve the bot.

---

## Summary Statistics

- Total Lessons Learned: 0
- Critical Issues Identified: 0
- Strategies Analyzed: 0
- Last Updated: {timestamp}

---

## Lessons Learned

<!-- Lessons will be added below -->

---

## Strategy Profiles

<!-- Strategy profiles will be added below -->

---

## Pattern Frequency Analysis

<!-- Common failure patterns ranked by frequency -->

---

## Mathematical Principles Discovered

<!-- Key mathematical insights from backtesting -->

""".format(timestamp=datetime.now().isoformat())
        
        self.soul_file.write_text(initial_content)
        logger.info("Initialized SOUL.md with template structure")
    
    def add_lesson(
        self,
        strategy_name: str,
        lesson_type: str,
        problem_description: str,
        mathematical_analysis: Dict[str, Any],
        recommended_adjustments: List[str],
        confidence_score: float,
        related_patterns: List[str] = None,
        severity: str = "warning"
    ) -> BacktestLesson:
        """Add a new lesson to the knowledge base."""
        lesson = BacktestLesson(
            timestamp=datetime.now().isoformat(),
            strategy_name=strategy_name,
            lesson_type=lesson_type,
            severity=severity,
            problem_description=problem_description,
            mathematical_analysis=mathematical_analysis,
            recommended_adjustments=recommended_adjustments,
            confidence_score=confidence_score,
            related_patterns=related_patterns or [],
            applied_to_live=False
        )
        
        self.lessons.append(lesson)
        
        # Update pattern database
        for pattern in lesson.related_patterns:
            self.pattern_database[pattern] = \
                self.pattern_database.get(pattern, 0) + 1
        
        logger.info(f"Added lesson: {lesson_type} for {strategy_name}")
        
        return lesson
    
    def analyze_backtest_results(
        self,
        strategy_name: str,
        metrics: Dict[str, Any],
        equity_curve: List[float] = None,
        trades: List[Dict] = None
    ) -> List[BacktestLesson]:
        """
        Automatically analyze backtest results and extract lessons.
        """
        lessons = []
        
        # Check for overfitting indicators
        if self._detect_overfitting(metrics):
            lesson = self.add_lesson(
                strategy_name=strategy_name,
                lesson_type="overfitting",
                problem_description=self._describe_overfitting(metrics),
                mathematical_analysis=self._analyze_overfitting_math(metrics),
                recommended_adjustments=self._get_overfitting_fixes(),
                confidence_score=0.85,
                related_patterns=["curve_fitting", "parameter_sensitivity"],
                severity="critical"
            )
            lessons.append(lesson)
        
        # Check for risk management issues
        if self._detect_risk_issues(metrics):
            lesson = self.add_lesson(
                strategy_name=strategy_name,
                lesson_type="risk_management",
                problem_description=self._describe_risk_issues(metrics),
                mathematical_analysis=self._analyze_risk_math(metrics),
                recommended_adjustments=self._get_risk_fixes(),
                confidence_score=0.9,
                related_patterns=["excessive_drawdown", "tail_risk"],
                severity="critical"
            )
            lessons.append(lesson)
        
        # Check for execution issues
        if self._detect_execution_issues(metrics, trades):
            lesson = self.add_lesson(
                strategy_name=strategy_name,
                lesson_type="execution",
                problem_description="Strategy performance degrades significantly with realistic slippage and fees",
                mathematical_analysis={
                    "slippage_sensitivity": metrics.get('slippage_impact', 0.0),
                    "fee_impact": metrics.get('fee_drag', 0.0)
                },
                recommended_adjustments=[
                    "Reduce trade frequency to minimize fee impact",
                    "Focus on higher timeframes where slippage is less significant",
                    "Implement smarter order routing and limit orders"
                ],
                confidence_score=0.75,
                related_patterns=["slippage_sensitivity", "fee_drag"],
                severity="warning"
            )
            lessons.append(lesson)
        
        # Check for market regime dependency
        if self._detect_regime_dependency(metrics):
            lesson = self.add_lesson(
                strategy_name=strategy_name,
                lesson_type="market_regime",
                problem_description="Strategy shows strong performance in specific market regimes but fails in others",
                mathematical_analysis={
                    "regime_performance_variance": metrics.get('regime_variance', 0.0),
                    "best_regime": metrics.get('best_regime', 'unknown'),
                    "worst_regime": metrics.get('worst_regime', 'unknown')
                },
                recommended_adjustments=[
                    "Add regime detection filter before entering trades",
                    "Develop regime-specific parameter sets",
                    "Consider ensemble approach combining multiple strategies"
                ],
                confidence_score=0.8,
                related_patterns=["regime_dependency", "conditional_performance"],
                severity="warning"
            )
            lessons.append(lesson)
        
        return lessons
    
    def _detect_overfitting(self, metrics: Dict) -> bool:
        """Detect signs of overfitting in metrics."""
        # Multiple red flags for overfitting
        red_flags = 0
        
        # In-sample vs out-of-sample degradation
        is_sharpe = metrics.get('is_sharpe', 0)
        os_sharpe = metrics.get('os_sharpe', 0)
        if is_sharpe > 0 and (is_sharpe - os_sharpe) / is_sharpe > 0.5:
            red_flags += 1
        
        # Too many parameters relative to data points
        n_params = metrics.get('n_parameters', 0)
        n_trades = metrics.get('n_trades', 1)
        if n_params > 0 and n_trades / n_params < 20:
            red_flags += 1
        
        # Extremely high Sharpe (>3 is suspicious)
        if metrics.get('sharpe_ratio', 0) > 3.0:
            red_flags += 1
        
        # Very high win rate with low profit factor
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        if win_rate > 0.8 and profit_factor < 1.5:
            red_flags += 1
        
        return red_flags >= 2
    
    def _detect_risk_issues(self, metrics: Dict) -> bool:
        """Detect risk management problems."""
        max_dd = abs(metrics.get('max_drawdown', 0))
        var_95 = abs(metrics.get('var_95', 0))
        kurtosis = metrics.get('kurtosis', 0)
        
        # Excessive drawdown
        if max_dd > 0.25:
            return True
        
        # Fat tails (high kurtosis)
        if kurtosis > 5:
            return True
        
        # Extreme VaR
        if var_95 > 0.1:
            return True
        
        return False
    
    def _detect_execution_issues(self, metrics: Dict, trades: List = None) -> bool:
        """Detect execution-related problems."""
        # Compare gross vs net returns
        gross_return = metrics.get('gross_return', 0)
        net_return = metrics.get('net_return', 0)
        
        if gross_return > 0 and (gross_return - net_return) / gross_return > 0.3:
            return True
        
        # Check trade-level metrics if available
        if trades:
            total_fees = sum(t.get('fees', 0) for t in trades)
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            
            if total_pnl > 0 and total_fees / total_pnl > 0.2:
                return True
        
        return False
    
    def _detect_regime_dependency(self, metrics: Dict) -> bool:
        """Detect if strategy is too dependent on specific market regimes."""
        regime_returns = metrics.get('regime_returns', {})
        
        if len(regime_returns) < 2:
            return False
        
        returns = list(regime_returns.values())
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        # High variance across regimes indicates dependency
        if variance > 0.01:  # Threshold depends on scale
            return True
        
        return False
    
    def _describe_overfitting(self, metrics: Dict) -> str:
        """Generate description of overfitting problem."""
        descriptions = []
        
        is_sharpe = metrics.get('is_sharpe', 0)
        os_sharpe = metrics.get('os_sharpe', 0)
        if is_sharpe > 0 and (is_sharpe - os_sharpe) / is_sharpe > 0.5:
            descriptions.append(
                f"Significant degradation from in-sample Sharpe ({is_sharpe:.2f}) "
                f"to out-of-sample ({os_sharpe:.2f}), indicating curve-fitting."
            )
        
        n_params = metrics.get('n_parameters', 0)
        n_trades = metrics.get('n_trades', 1)
        if n_params > 0 and n_trades / n_params < 20:
            descriptions.append(
                f"Too many parameters ({n_params}) relative to trades ({n_trades}). "
                f"Ratio of {n_trades/n_params:.1f} is below recommended minimum of 20."
            )
        
        return " ".join(descriptions) if descriptions else "Multiple overfitting indicators detected."
    
    def _analyze_overfitting_math(self, metrics: Dict) -> Dict[str, Any]:
        """Perform mathematical analysis of overfitting."""
        is_sharpe = metrics.get('is_sharpe', 0)
        os_sharpe = metrics.get('os_sharpe', 0)
        
        # Calculate Deflated Sharpe Ratio approximation
        n_trials = metrics.get('n_trials', 1)
        sample_size = metrics.get('sample_size', 252)
        
        expected_max = np.sqrt(2 * np.log(n_trials)) / np.sqrt(sample_size) if n_trials > 1 else 0
        
        return {
            'is_sharpe': is_sharpe,
            'os_sharpe': os_sharpe,
            'degradation_pct': (is_sharpe - os_sharpe) / max(is_sharpe, 0.01) * 100,
            'expected_max_under_null': expected_max,
            'deflated_sharpe_estimate': os_sharpe - expected_max,
            'probability_of_overfitting': min(1.0, (is_sharpe - os_sharpe) / max(is_sharpe, 0.01))
        }
    
    def _get_overfitting_fixes(self) -> List[str]:
        """Get recommended fixes for overfitting."""
        return [
            "Reduce number of optimized parameters",
            "Apply walk-forward optimization with stricter out-of-sample testing",
            "Use regularization techniques (L1/L2 penalties on parameters)",
            "Increase minimum trades per parameter requirement",
            "Apply Deflated Sharpe Ratio threshold (DSR > 0.5)",
            "Use simpler strategy logic with fewer decision points"
        ]
    
    def _describe_risk_issues(self, metrics: Dict) -> str:
        """Describe risk management problems."""
        issues = []
        
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd > 0.25:
            issues.append(f"Maximum drawdown of {max_dd*100:.1f}% exceeds acceptable threshold of 25%.")
        
        kurtosis = metrics.get('kurtosis', 0)
        if kurtosis > 5:
            issues.append(f"Excess kurtosis of {kurtosis:.1f} indicates fat-tailed return distribution.")
        
        return " ".join(issues) if issues else "Risk metrics indicate potential issues."
    
    def _analyze_risk_math(self, metrics: Dict) -> Dict[str, Any]:
        """Mathematical analysis of risk issues."""
        return {
            'max_drawdown': metrics.get('max_drawdown', 0),
            'var_95': metrics.get('var_95', 0),
            'cvar_95': metrics.get('cvar_95', 0),
            'kurtosis': metrics.get('kurtosis', 0),
            'skewness': metrics.get('skewness', 0),
            'ulcer_index': metrics.get('ulcer_index', 0),
            'tail_ratio': metrics.get('tail_ratio', 1.0)
        }
    
    def _get_risk_fixes(self) -> List[str]:
        """Get recommended fixes for risk issues."""
        return [
            "Implement dynamic position sizing based on volatility",
            "Add hard stop-loss at strategy level (max 2% daily loss)",
            "Reduce leverage during high volatility periods",
            "Diversify across uncorrelated strategies",
            "Implement Kelly criterion or half-Kelly for position sizing",
            "Add tail risk hedging during extreme market conditions"
        ]
    
    def write_to_soul(self) -> None:
        """Write all lessons to SOUL.md file."""
        # Sort lessons by severity and timestamp
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        sorted_lessons = sorted(
            self.lessons,
            key=lambda x: (severity_order.get(x.severity, 3), x.timestamp),
            reverse=False
        )
        
        # Build content
        content = self._build_soul_content(sorted_lessons)
        
        # Write atomically
        temp_file = self.soul_file.with_suffix('.tmp')
        temp_file.write_text(content)
        temp_file.replace(self.soul_file)
        
        # Update hash
        self.last_update_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        logger.info(f"Written {len(sorted_lessons)} lessons to SOUL.md")
    
    def _build_soul_content(self, lessons: List[BacktestLesson]) -> str:
        """Build complete SOUL.md content."""
        # Count statistics
        critical_count = sum(1 for l in lessons if l.severity == 'critical')
        strategies = set(l.strategy_name for l in lessons)
        
        content = f"""# ZAID PERSONAL CRYPTO TRADING BOT - SOUL.md

## Strategic Optimization & Understanding Ledger

*Last Updated: {datetime.now().isoformat()}*
*Content Hash: {self.last_update_hash or "new"}*

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Lessons Learned | {len(lessons)} |
| Critical Issues | {critical_count} |
| Warnings | {sum(1 for l in lessons if l.severity == 'warning')} |
| Informational | {sum(1 for l in lessons if l.severity == 'info')} |
| Strategies Analyzed | {len(strategies)} |
| Unique Patterns | {len(self.pattern_database)} |

---

## Lessons Learned

"""
        
        # Add each lesson
        for i, lesson in enumerate(lessons, 1):
            content += self._format_lesson(lesson, i)
        
        # Add pattern frequency
        content += """
---

## Pattern Frequency Analysis

| Pattern | Occurrences |
|---------|-------------|
"""
        for pattern, count in sorted(
            self.pattern_database.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]:
            content += f"| {pattern} | {count} |\n"
        
        content += """
---

## Action Items

### Immediate Actions Required
"""
        
        critical_lessons = [l for l in lessons if l.severity == 'critical' and not l.applied_to_live]
        for lesson in critical_lessons[:5]:
            content += f"- [ ] **{lesson.strategy_name}**: {lesson.recommended_adjustments[0] if lesson.recommended_adjustments else 'Review immediately'}\n"
        
        content += """
---

## Notes

This document is automatically updated after each backtest run.
Lessons marked as 'applied_to_live' have been incorporated into the production trading system.
"""
        
        return content
    
    def _format_lesson(self, lesson: BacktestLesson, index: int) -> str:
        """Format a single lesson for markdown output."""
        return f"""
<!-- LESSON_START -->
### Lesson {index}: {lesson.lesson_type.replace('_', ' ').title()}

**Strategy:** {lesson.strategy_name}  
**Severity:** {lesson.severity.upper()}  
**Date:** {lesson.timestamp[:10]}  
**Confidence:** {lesson.confidence_score*100:.0f}%

#### Problem Description
{lesson.problem_description}

#### Mathematical Analysis
```json
{json.dumps(lesson.mathematical_analysis, indent=2)}
```

#### Recommended Adjustments
""" + "\n".join(f"- {adj}" for adj in lesson.recommended_adjustments) + f"""

#### Related Patterns
{', '.join(lesson.related_patterns)}

**Applied to Live:** {'Yes' if lesson.applied_to_live else 'No'}
<!-- LESSON_END -->

---

"""


# Import numpy for calculations
try:
    import numpy as np
except ImportError:
    np = None


def main():
    """Example usage of SoulBacktestWriter."""
    print("="*60)
    print("SOUL BACKTEST WRITER DEMO")
    print("="*60)
    
    # Create writer
    writer = SoulBacktestWriter(soul_path="./SOUL.md")
    
    # Simulate backtest metrics with issues
    metrics = {
        'is_sharpe': 2.5,
        'os_sharpe': 0.8,
        'sharpe_ratio': 2.5,
        'max_drawdown': -0.35,
        'n_parameters': 8,
        'n_trades': 100,
        'win_rate': 0.85,
        'profit_factor': 1.3,
        'kurtosis': 8.5,
        'var_95': -0.12,
        'n_trials': 500,
        'sample_size': 252
    }
    
    # Analyze and extract lessons
    lessons = writer.analyze_backtest_results(
        strategy_name="MomentumBreakout_v3",
        metrics=metrics
    )
    
    print(f"\nExtracted {len(lessons)} lessons:")
    for lesson in lessons:
        print(f"  - [{lesson.severity.upper()}] {lesson.lesson_type}: {lesson.problem_description[:80]}...")
    
    # Write to SOUL.md
    writer.write_to_soul()
    print(f"\nWritten to {writer.soul_file.absolute()}")
    
    # Show summary
    print("\n" + "="*60)
    print(f"Total lessons in SOUL.md: {len(writer.lessons)}")
    print(f"Pattern database size: {len(writer.pattern_database)}")
    print("="*60)


if __name__ == "__main__":
    main()
