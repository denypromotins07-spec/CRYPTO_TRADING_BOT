#!/usr/bin/env python3
"""
MARL Soul Logger - Game-Theoretic Exploit and Agent Disagreement Logger

This module logs game-theoretic exploits, agent disagreements, and strategic
decisions to SOUL.md for audit trail and performance analysis.

Features:
- Structured logging of MARL events
- Game-theoretic exploit documentation
- Agent disagreement tracking
- Nash equilibrium convergence logging
- Adversarial detection event recording
- Memory-efficient log rotation

Integrates quantitative finance domains:
- Performance attribution
- Risk management audit trails
- Regulatory compliance logging
- Strategy analytics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from datetime import datetime
import json
import os
from pathlib import Path


class EventType(Enum):
    """Types of events to log."""
    GAME_THEORETIC_EXPLOIT = auto()
    AGENT_DISAGREEMENT = auto()
    NASH_EQUILIBRIUM_FOUND = auto()
    CONSENSUS_OVERRIDE = auto()
    ADVERSARIAL_DETECTED = auto()
    STRATEGY_BLEND_DECISION = auto()
    COUNTERFACTUAL_REWARD = auto()
    REGRET_UPDATE = auto()
    OPPONENT_PATTERN_DETECTED = auto()
    RISK_TIER_CHANGE = auto()


class SeverityLevel(Enum):
    """Severity levels for logged events."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SoulEvent:
    """A single event to be logged."""
    event_type: EventType
    severity: SeverityLevel
    timestamp: str
    episode_id: int
    step: int
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    agents_involved: List[int] = field(default_factory=list)
    outcome: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.name,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "step": self.step,
            "description": self.description,
            "data": self.data,
            "agents_involved": self.agents_involved,
            "outcome": self.outcome
        }


class MARLSoulLogger:
    """
    Logger for Multi-Agent Reinforcement Learning events.
    
    This class maintains a structured log of all significant MARL events,
    including game-theoretic exploits, agent disagreements, and strategic
    decisions. Logs are written to SOUL.md in a human-readable format.
    
    Memory footprint: ~500KB for active buffer (logs flushed periodically)
    """
    
    def __init__(
        self,
        log_path: str = "SOUL.md",
        max_buffer_size: int = 1000,
        flush_interval: int = 100
    ):
        self.log_path = Path(log_path)
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval
        
        # Event buffer
        self.buffer: List[SoulEvent] = []
        
        # Statistics
        self.event_counts: Dict[EventType, int] = {t: 0 for t in EventType}
        self.total_events = 0
        self.last_flush_count = 0
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
        # Initialize log file
        self._initialize_log_file()
    
    def _initialize_log_file(self) -> None:
        """Initialize or append to the log file."""
        if not self.log_path.exists():
            self._write_header()
    
    def _write_header(self) -> None:
        """Write header to new log file."""
        header = f"""# ZAID Personal Crypto Trading Bot - MARL Soul Log

## Session Information
- **Session ID**: {self.session_id}
- **Start Time**: {self.session_start.isoformat()}
- **Bot Version**: Stage 27/100 - Multi-Agent Reinforcement Learning
- **Target**: 8k-20k INR/hour in 4hr window

## Event Types Logged
- Game-Theoretic Exploits
- Agent Disagreements
- Nash Equilibrium Solutions
- Consensus Overrides
- Adversarial Detections
- Strategy Blend Decisions
- Counterfactual Rewards
- Regret Updates
- Opponent Pattern Detection
- Risk Tier Changes

---

## Event Log

"""
        with open(self.log_path, 'w') as f:
            f.write(header)
    
    def log_event(self, event: SoulEvent) -> None:
        """
        Log a single event.
        
        Args:
            event: SoulEvent to log
        """
        self.buffer.append(event)
        self.event_counts[event.event_type] += 1
        self.total_events += 1
        
        # Flush if buffer is full or interval reached
        if len(self.buffer) >= self.max_buffer_size:
            self.flush()
        elif self.total_events % self.flush_interval == 0:
            self.flush()
    
    def log_game_theoretic_exploit(
        self,
        exploit_type: str,
        opponent_id: str,
        expected_profit: float,
        confidence: float,
        episode_id: int,
        step: int,
        details: Dict[str, Any] = None
    ) -> None:
        """Log a successful game-theoretic exploit."""
        event = SoulEvent(
            event_type=EventType.GAME_THEORETIC_EXPLOIT,
            severity=SeverityLevel.INFO,
            timestamp=datetime.now().isoformat(),
            episode_id=episode_id,
            step=step,
            description=f"Exploited predictable pattern in {opponent_id}",
            data={
                "exploit_type": exploit_type,
                "opponent_id": opponent_id,
                "expected_profit": expected_profit,
                "confidence": confidence,
                "details": details or {}
            },
            outcome="Pattern successfully exploited"
        )
        self.log_event(event)
    
    def log_agent_disagreement(
        self,
        proposal_id: str,
        agreeing_agents: List[int],
        disagreeing_agents: List[int],
        veto_agents: List[int],
        final_decision: str,
        episode_id: int,
        step: int
    ) -> None:
        """Log significant agent disagreement."""
        severity = SeverityLevel.WARNING if veto_agents else SeverityLevel.INFO
        
        event = SoulEvent(
            event_type=EventType.AGENT_DISAGREEMENT,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            episode_id=episode_id,
            step=step,
            description=f"Agent disagreement on proposal {proposal_id}",
            data={
                "proposal_id": proposal_id,
                "agreeing_agents": agreeing_agents,
                "disagreeing_agents": disagreeing_agents,
                "veto_agents": veto_agents,
                "final_decision": final_decision,
                "disagreement_ratio": len(disagreeing_agents) / max(len(agreeing_agents) + len(disagreeing_agents), 1)
            },
            agents_involved=agreeing_agents + disagreeing_agents + veto_agents,
            outcome=final_decision
        )
        self.log_event(event)
    
    def log_nash_equilibrium(
        self,
        converged: bool,
        iterations: int,
        strategies: Dict[int, Any],
        utilities: Dict[int, float],
        gap: float,
        episode_id: int,
        step: int
    ) -> None:
        """Log Nash equilibrium computation result."""
        severity = SeverityLevel.INFO if converged else SeverityLevel.WARNING
        
        event = SoulEvent(
            event_type=EventType.NASH_EQUILIBRIUM_FOUND,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            episode_id=episode_id,
            step=step,
            description=f"Nash equilibrium {'converged' if converged else 'did not converge'}",
            data={
                "converged": converged,
                "iterations": iterations,
                "strategies": {str(k): str(v) for k, v in strategies.items()},
                "utilities": utilities,
                "gap": gap
            },
            outcome="Equilibrium found" if converged else "No equilibrium found"
        )
        self.log_event(event)
    
    def log_consensus_override(
        self,
        proposal_id: str,
        override_reason: str,
        original_action: str,
        overridden_action: str,
        risk_tier: str,
        episode_id: int,
        step: int
    ) -> None:
        """Log consensus protocol override."""
        event = SoulEvent(
            event_type=EventType.CONSENSUS_OVERRIDE,
            severity=SeverityLevel.WARNING,
            timestamp=datetime.now().isoformat(),
            episode_id=episode_id,
            step=step,
            description=f"Consensus override: {override_reason}",
            data={
                "proposal_id": proposal_id,
                "override_reason": override_reason,
                "original_action": original_action,
                "overridden_action": overridden_action,
                "risk_tier": risk_tier
            },
            outcome=f"Action changed from {original_action} to {overridden_action}"
        )
        self.log_event(event)
    
    def log_adversarial_detected(
        self,
        attack_type: str,
        threat_level: str,
        confidence: float,
        recommended_action: str,
        episode_id: int,
        step: int
    ) -> None:
        """Log adversarial algorithm detection."""
        severity_map = {
            "CRITICAL": SeverityLevel.CRITICAL,
            "HIGH": SeverityLevel.ERROR,
            "MEDIUM": SeverityLevel.WARNING,
            "LOW": SeverityLevel.INFO,
            "NONE": SeverityLevel.DEBUG
        }
        
        event = SoulEvent(
            event_type=EventType.ADVERSARIAL_DETECTED,
            severity=severity_map.get(threat_level, SeverityLevel.INFO),
            timestamp=datetime.now().isoformat(),
            episode_id=episode_id,
            step=step,
            description=f"Adversarial activity detected: {attack_type}",
            data={
                "attack_type": attack_type,
                "threat_level": threat_level,
                "confidence": confidence,
                "recommended_action": recommended_action
            },
            outcome=f"Defensive measures activated: {recommended_action}"
        )
        self.log_event(event)
    
    def log_strategy_blend_decision(
        self,
        blended_action: str,
        confidence: float,
        position_size: float,
        contributing_agents: int,
        agent_weights: Dict[int, float],
        override_active: bool,
        episode_id: int,
        step: int
    ) -> None:
        """Log strategy blending decision."""
        event = SoulEvent(
            event_type=EventType.STRATEGY_BLEND_DECISION,
            severity=SeverityLevel.DEBUG,
            timestamp=datetime.now().isoformat(),
            episode_id=episode_id,
            step=step,
            description=f"Blended decision: {blended_action}",
            data={
                "blended_action": blended_action,
                "confidence": confidence,
                "position_size": position_size,
                "contributing_agents": contributing_agents,
                "agent_weights": agent_weights,
                "override_active": override_active
            },
            agents_involved=list(agent_weights.keys()),
            outcome=f"Executed {blended_action} with size {position_size}"
        )
        self.log_event(event)
    
    def flush(self) -> None:
        """Flush buffered events to log file."""
        if not self.buffer:
            return
        
        with open(self.log_path, 'a') as f:
            for event in self.buffer[self.last_flush_count:]:
                self._write_event(f, event)
        
        self.last_flush_count = len(self.buffer)
        self.buffer = []
    
    def _write_event(self, f, event: SoulEvent) -> None:
        """Write a single event to file."""
        f.write(f"\n### [{event.severity.value}] {event.event_type.name}\n\n")
        f.write(f"**Timestamp**: {event.timestamp}\n")
        f.write(f"**Episode**: {event.episode_id} | **Step**: {event.step}\n")
        f.write(f"**Description**: {event.description}\n")
        
        if event.agents_involved:
            f.write(f"**Agents Involved**: {event.agents_involved}\n")
        
        if event.data:
            f.write("**Data**:\n")
            f.write("```json\n")
            f.write(json.dumps(event.data, indent=2, default=str))
            f.write("\n```\n")
        
        if event.outcome:
            f.write(f"\n**Outcome**: {event.outcome}\n")
        
        f.write("\n---\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        session_duration = datetime.now() - self.session_start
        
        return {
            "session_id": self.session_id,
            "session_duration_seconds": session_duration.total_seconds(),
            "total_events": self.total_events,
            "events_by_type": {t.name: c for t, c in self.event_counts.items()},
            "buffer_size": len(self.buffer),
            "log_file_path": str(self.log_path)
        }
    
    def generate_summary(self) -> str:
        """Generate a summary report of logged events."""
        stats = self.get_statistics()
        
        summary = f"""## Session Summary

- **Session Duration**: {stats['session_duration_seconds']:.1f} seconds
- **Total Events Logged**: {stats['total_events']}
- **Events by Type**:
"""
        
        for event_type, count in stats['events_by_type'].items():
            if count > 0:
                summary += f"  - {event_type}: {count}\n"
        
        return summary
    
    def close(self) -> None:
        """Finalize logging and write summary."""
        self.flush()
        
        # Write summary at end of file
        with open(self.log_path, 'a') as f:
            f.write(f"\n\n## End of Session\n\n")
            f.write(self.generate_summary())
            f.write(f"\n**End Time**: {datetime.now().isoformat()}\n")


# Global logger instance
_soul_logger: Optional[MARLSoulLogger] = None


def get_soul_logger(log_path: str = "SOUL.md") -> MARLSoulLogger:
    """Get or create the global soul logger instance."""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = MARLSoulLogger(log_path=log_path)
    return _soul_logger


def log_exploit(
    exploit_type: str,
    opponent_id: str,
    expected_profit: float,
    confidence: float,
    episode_id: int = 0,
    step: int = 0
) -> None:
    """Convenience function to log a game-theoretic exploit."""
    logger = get_soul_logger()
    logger.log_game_theoretic_exploit(
        exploit_type=exploit_type,
        opponent_id=opponent_id,
        expected_profit=expected_profit,
        confidence=confidence,
        episode_id=episode_id,
        step=step
    )


def log_disagreement(
    proposal_id: str,
    agreeing: List[int],
    disagreeing: List[int],
    vetoes: List[int],
    decision: str,
    episode_id: int = 0,
    step: int = 0
) -> None:
    """Convenience function to log agent disagreement."""
    logger = get_soul_logger()
    logger.log_agent_disagreement(
        proposal_id=proposal_id,
        agreeing_agents=agreeing,
        disagreeing_agents=disagreeing,
        veto_agents=vetoes,
        final_decision=decision,
        episode_id=episode_id,
        step=step
    )


if __name__ == "__main__":
    # Example usage
    logger = MARLSoulLogger(log_path="SOUL.md")
    
    # Log various events
    logger.log_game_theoretic_exploit(
        exploit_type="momentum_prediction",
        opponent_id="MM_BTC_01",
        expected_profit=150.0,
        confidence=0.85,
        episode_id=1,
        step=42,
        details={"pattern": "sequential_order_cancellation"}
    )
    
    logger.log_agent_disagreement(
        proposal_id="TRADE_042",
        agreeing_agents=[0, 1, 4],
        disagreeing_agents=[2, 3],
        veto_agents=[],
        final_decision="APPROVED",
        episode_id=1,
        step=43
    )
    
    logger.log_nash_equilibrium(
        converged=True,
        iterations=15,
        strategies={0: "Aggressive", 1: "Passive"},
        utilities={0: 0.025, 1: 0.018},
        gap=0.001,
        episode_id=1,
        step=44
    )
    
    logger.log_adversarial_detected(
        attack_type="SPOOFING",
        threat_level="HIGH",
        confidence=0.78,
        recommended_action="Reduce order visibility",
        episode_id=1,
        step=45
    )
    
    # Close and finalize
    logger.close()
    
    print(f"Logged {logger.total_events} events to {logger.log_path}")
    print(logger.generate_summary())
