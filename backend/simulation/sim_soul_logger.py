"""
Simulation SOUL Logger

Logs strategy failures during synthetic black swan simulations to SOUL.md.
Tracks:
- Strategy blow-up scenarios
- Maximum drawdown events
- Risk limit breaches
- Parameter sensitivity failures

Creates comprehensive failure documentation for strategy improvement.
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FailureSeverity(Enum):
    """Severity levels for strategy failures."""
    WARNING = "warning"           # Minor issue, strategy survived
    CRITICAL = "critical"         # Significant drawdown, recovery needed
    CATASTROPHIC = "catastrophic" # Strategy blew up (>50% loss)
    BLACK_SWAN = "black_swan"     # Extreme event failure


@dataclass
class FailureEvent:
    """Record of a strategy failure event."""
    timestamp: str
    severity: str
    scenario_name: str
    strategy_name: str
    
    # Market conditions
    market_regime: str
    volatility_level: float
    price_change_pct: float
    
    # Strategy performance
    drawdown_pct: float
    pnl_pct: float
    max_position_exposure: float
    leverage_used: float
    
    # Risk metrics breached
    var_breach: bool
    drawdown_limit_breach: bool
    position_limit_breach: bool
    
    # Root cause analysis
    primary_cause: str
    contributing_factors: List[str]
    
    # Recovery info
    recovered: bool
    recovery_time_steps: Optional[int]
    
    # Recommendations
    recommendations: List[str]


@dataclass
class BlackSwanScenario:
    """Black swan event scenario definition."""
    name: str
    description: str
    trigger_conditions: Dict[str, float]
    expected_impact: str
    historical_precedent: Optional[str]


class SimSoulLogger:
    """
    Logs strategy failures during synthetic simulations.
    
    Creates detailed SOUL.md documentation of:
    - Failure modes and triggers
    - Market conditions at failure
    - Strategy vulnerabilities
    - Improvement recommendations
    """
    
    def __init__(self, soul_path: str = "SOUL.md"):
        self.soul_path = soul_path
        self.failure_events: List[FailureEvent] = []
        self.black_swan_scenarios: List[BlackSwanScenario] = []
        self.strategy_stats: Dict[str, Dict[str, Any]] = {}
        
        # Initialize SOUL.md if not exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Create or initialize SOUL.md file."""
        if not os.path.exists(self.soul_path):
            with open(self.soul_path, 'w') as f:
                f.write("# ZAID PERSONAL CRYPTO TRADING BOT - SOUL.md\n\n")
                f.write("## Strategy Failure Log & Black Swan Documentation\n\n")
                f.write("*This document tracks all strategy failures during synthetic stress testing.*\n\n")
                f.write("---\n\n")
    
    def register_black_swan_scenario(
        self,
        name: str,
        description: str,
        trigger_conditions: Dict[str, float],
        expected_impact: str,
        historical_precedent: Optional[str] = None
    ) -> None:
        """Register a black swan scenario for tracking."""
        scenario = BlackSwanScenario(
            name=name,
            description=description,
            trigger_conditions=trigger_conditions,
            expected_impact=expected_impact,
            historical_precedent=historical_precedent,
        )
        self.black_swan_scenarios.append(scenario)
        logger.info(f"Registered black swan scenario: {name}")
    
    def log_failure(
        self,
        strategy_name: str,
        scenario_name: str,
        drawdown_pct: float,
        pnl_pct: float,
        market_regime: str,
        volatility_level: float,
        price_change_pct: float,
        max_position_exposure: float,
        leverage_used: float,
        primary_cause: str,
        contributing_factors: Optional[List[str]] = None,
        var_breach: bool = False,
        drawdown_limit_breach: bool = False,
        position_limit_breach: bool = False,
        recovered: bool = False,
        recovery_time_steps: Optional[int] = None,
        recommendations: Optional[List[str]] = None,
    ) -> FailureEvent:
        """Log a strategy failure event."""
        
        # Determine severity
        if drawdown_pct >= 0.5 or pnl_pct <= -0.5:
            severity = FailureSeverity.CATASTROPHIC
        elif drawdown_pct >= 0.2 or pnl_pct <= -0.2:
            severity = FailureSeverity.CRITICAL
        elif "black swan" in scenario_name.lower():
            severity = FailureSeverity.BLACK_SWAN
        else:
            severity = FailureSeverity.WARNING
        
        event = FailureEvent(
            timestamp=datetime.now().isoformat(),
            severity=severity.value,
            scenario_name=scenario_name,
            strategy_name=strategy_name,
            market_regime=market_regime,
            volatility_level=volatility_level,
            price_change_pct=price_change_pct,
            drawdown_pct=drawdown_pct,
            pnl_pct=pnl_pct,
            max_position_exposure=max_position_exposure,
            leverage_used=leverage_used,
            var_breach=var_breach,
            drawdown_limit_breach=drawdown_limit_breach,
            position_limit_breach=position_limit_breach,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors or [],
            recovered=recovered,
            recovery_time_steps=recovery_time_steps,
            recommendations=recommendations or [],
        )
        
        self.failure_events.append(event)
        self._update_strategy_stats(strategy_name, event)
        
        # Write to SOUL.md
        self._append_to_soul(event)
        
        logger.warning(
            f"Failure logged: {strategy_name} in {scenario_name} "
            f"(drawdown={drawdown_pct:.1%}, severity={severity.value})"
        )
        
        return event
    
    def _update_strategy_stats(self, strategy_name: str, event: FailureEvent) -> None:
        """Update aggregate statistics for a strategy."""
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = {
                'total_failures': 0,
                'catastrophic_count': 0,
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'failure_scenarios': [],
                'common_causes': {},
            }
        
        stats = self.strategy_stats[strategy_name]
        stats['total_failures'] += 1
        
        if event.severity == FailureSeverity.CATASTROPHIC.value:
            stats['catastrophic_count'] += 1
        
        stats['max_drawdown'] = max(stats['max_drawdown'], event.drawdown_pct)
        
        # Update average
        n = stats['total_failures']
        stats['avg_drawdown'] = (stats['avg_drawdown'] * (n - 1) + event.drawdown_pct) / n
        
        if scenario_name not in stats['failure_scenarios']:
            stats['failure_scenarios'].append(event.scenario_name)
        
        # Track common causes
        cause = event.primary_cause
        stats['common_causes'][cause] = stats['common_causes'].get(cause, 0) + 1
    
    def _append_to_soul(self, event: FailureEvent) -> None:
        """Append failure event to SOUL.md file."""
        with open(self.soul_path, 'a') as f:
            f.write(f"\n### [{event.timestamp}] {event.severity.upper()}\n\n")
            f.write(f"**Strategy:** `{event.strategy_name}`\n\n")
            f.write(f"**Scenario:** {event.scenario_name}\n\n")
            
            f.write("#### Market Conditions\n")
            f.write(f"- Regime: `{event.market_regime}`\n")
            f.write(f"- Volatility: `{event.volatility_level:.4f}`\n")
            f.write(f"- Price Change: `{event.price_change_pct:.2%}`\n\n")
            
            f.write("#### Performance Impact\n")
            f.write(f"- Drawdown: `{event.drawdown_pct:.2%}`\n")
            f.write(f"- PnL: `{event.pnl_pct:.2%}`\n")
            f.write(f"- Max Exposure: `{event.max_position_exposure:.2f}`\n")
            f.write(f"- Leverage Used: `{event.leverage_used:.2f}x`\n\n")
            
            f.write("#### Risk Breaches\n")
            breaches = []
            if event.var_breach:
                breaches.append("VaR Limit")
            if event.drawdown_limit_breach:
                breaches.append("Drawdown Limit")
            if event.position_limit_breach:
                breaches.append("Position Limit")
            
            if breaches:
                f.write(f"- **BREACHED:** {', '.join(breaches)}\n\n")
            else:
                f.write("- No risk limits breached\n\n")
            
            f.write("#### Root Cause Analysis\n")
            f.write(f"**Primary Cause:** {event.primary_cause}\n\n")
            
            if event.contributing_factors:
                f.write("**Contributing Factors:**\n")
                for factor in event.contributing_factors:
                    f.write(f"- {factor}\n")
                f.write("\n")
            
            f.write("#### Recovery Status\n")
            status = "Recovered" if event.recovered else "Not Recovered"
            f.write(f"- Status: **{status}**\n")
            if event.recovery_time_steps is not None:
                f.write(f"- Recovery Time: {event.recovery_time_steps} steps\n")
            f.write("\n")
            
            if event.recommendations:
                f.write("#### Recommendations\n")
                for rec in event.recommendations:
                    f.write(f"- [ ] {rec}\n")
                f.write("\n")
            
            f.write("---\n")
    
    def generate_summary_report(self) -> str:
        """Generate summary report of all failures."""
        if not self.failure_events:
            return "No failure events recorded."
        
        lines = [
            "# STRATEGY FAILURE SUMMARY REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            f"Total Failure Events: {len(self.failure_events)}",
            "",
        ]
        
        # Count by severity
        severity_counts = {}
        for event in self.failure_events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        lines.append("## Failures by Severity")
        for severity, count in sorted(severity_counts.items()):
            lines.append(f"- {severity}: {count}")
        lines.append("")
        
        # Strategy-specific summaries
        lines.append("## Strategy-Specific Analysis")
        for strategy_name, stats in self.strategy_stats.items():
            lines.append(f"\n### {strategy_name}")
            lines.append(f"- Total Failures: {stats['total_failures']}")
            lines.append(f"- Catastrophic Events: {stats['catastrophic_count']}")
            lines.append(f"- Maximum Drawdown: {stats['max_drawdown']:.2%}")
            lines.append(f"- Average Drawdown: {stats['avg_drawdown']:.2%}")
            
            if stats['common_causes']:
                lines.append("- Common Failure Causes:")
                for cause, count in sorted(stats['common_causes'].items(), 
                                          key=lambda x: -x[1])[:3]:
                    lines.append(f"  - {cause}: {count} occurrences")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_vulnerability_analysis(self, strategy_name: str) -> Dict[str, Any]:
        """Analyze specific strategy vulnerabilities."""
        if strategy_name not in self.strategy_stats:
            return {'error': 'Strategy not found'}
        
        stats = self.strategy_stats[strategy_name]
        strategy_events = [e for e in self.failure_events 
                         if e.strategy_name == strategy_name]
        
        # Analyze market conditions at failure
        regimes_at_failure = {}
        for event in strategy_events:
            regimes_at_failure[event.market_regime] = \
                regimes_at_failure.get(event.market_regime, 0) + 1
        
        # Calculate vulnerability score (0-1, higher = more vulnerable)
        vuln_score = min(1.0, (
            stats['catastrophic_count'] * 0.4 +
            stats['max_drawdown'] * 0.3 +
            len(stats['failure_scenarios']) * 0.05
        ))
        
        return {
            'vulnerability_score': vuln_score,
            'total_failures': stats['total_failures'],
            'catastrophic_events': stats['catastrophic_count'],
            'max_drawdown': stats['max_drawdown'],
            'most_vulnerable_regime': max(regimes_at_failure.items(), 
                                         key=lambda x: x[1])[0] if regimes_at_failure else None,
            'primary_failure_causes': list(stats['common_causes'].keys())[:3],
            'recommendations': self._generate_recommendations(strategy_events),
        }
    
    def _generate_recommendations(self, events: List[FailureEvent]) -> List[str]:
        """Generate improvement recommendations based on failure patterns."""
        recommendations = []
        
        # Check for leverage-related failures
        high_leverage_events = [e for e in events if e.leverage_used > 3.0]
        if high_leverage_events:
            recommendations.append(
                "Reduce maximum leverage - multiple failures occurred with high leverage"
            )
        
        # Check for volatility-related failures
        high_vol_events = [e for e in events if e.volatility_level > 0.05]
        if high_vol_events:
            recommendations.append(
                "Implement volatility-based position sizing reduction"
            )
        
        # Check for regime-specific failures
        regime_counts = {}
        for e in events:
            regime_counts[e.market_regime] = regime_counts.get(e.market_regime, 0) + 1
        
        for regime, count in regime_counts.items():
            if count >= 3:
                recommendations.append(
                    f"Add specific safeguards for {regime} market regime"
                )
        
        # Check for non-recovery events
        non_recovered = [e for e in events if not e.recovered]
        if non_recovered:
            recommendations.append(
                "Implement stop-loss mechanisms to prevent unrecovered drawdowns"
            )
        
        return recommendations
    
    def export_json(self, output_path: str) -> None:
        """Export all failure data to JSON."""
        data = {
            'failure_events': [asdict(e) for e in self.failure_events],
            'black_swan_scenarios': [asdict(s) for s in self.black_swan_scenarios],
            'strategy_statistics': self.strategy_stats,
            'generated_at': datetime.now().isoformat(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported failure data to {output_path}")


# Example usage
if __name__ == "__main__":
    logger_instance = SimSoulLogger()
    
    # Register black swan scenarios
    logger_instance.register_black_swan_scenario(
        name="Flash Crash 2024",
        description="Sudden 30% drop in BTC within 5 minutes",
        trigger_conditions={"price_drop_pct": -0.30, "timeframe_minutes": 5},
        expected_impact="Catastrophic for leveraged long strategies",
        historical_precedent="March 2020 COVID crash",
    )
    
    # Log a failure
    logger_instance.log_failure(
        strategy_name="MomentumBreakout_v3",
        scenario_name="Flash Crash 2024",
        drawdown_pct=0.45,
        pnl_pct=-0.38,
        market_regime="crash",
        volatility_level=0.15,
        price_change_pct=-0.30,
        max_position_exposure=2.5,
        leverage_used=3.0,
        primary_cause="Excessive leverage during sudden reversal",
        contributing_factors=[
            "No stop-loss mechanism",
            "Position sizing not volatility-adjusted",
            "Failed to detect regime change",
        ],
        var_breach=True,
        drawdown_limit_breach=True,
        recovered=False,
        recommendations=[
            "Implement dynamic stop-loss based on ATR",
            "Reduce max leverage to 2x",
            "Add volatility regime detection",
        ],
    )
    
    print(logger_instance.generate_summary_report())
