#!/usr/bin/env python3
"""
Consensus Protocol for Multi-Agent Trade Validation

This module implements a lightweight voting mechanism for high-risk trades,
ensuring that individual agent signals can be overridden if the consensus
protocol flags catastrophic risk.

Features:
- Weighted voting based on agent confidence and historical accuracy
- Quorum-based decision making
- Risk-tier classification for trade validation
- Veto mechanism for extreme risk scenarios
- Memory-efficient vote tracking

Integrates quantitative finance domains:
- Ensemble methods
- Risk management frameworks
- Decision theory
- Multi-agent coordination
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray


class VoteType(Enum):
    """Types of votes an agent can cast."""
    STRONG_APPROVE = auto()
    APPROVE = auto()
    NEUTRAL = auto()
    DISAPPROVE = auto()
    STRONG_DISAPPROVE = auto()
    VETO = auto()  # Absolute veto for catastrophic risk


class RiskTier(Enum):
    """Risk classification for trades."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    EXTREME = auto()
    CATASTROPHIC = auto()


@dataclass
class TradeProposal:
    """A trade proposal requiring consensus approval."""
    proposal_id: str
    agent_id: int
    action: str  # BUY, SELL, HOLD
    pair: str
    size: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def risk_estimate(self) -> float:
        """Estimate risk from trade parameters."""
        if self.stop_loss is None:
            return 1.0
        
        risk_per_unit = abs(self.entry_price - self.stop_loss) / self.entry_price
        total_risk = risk_per_unit * self.size
        return min(1.0, total_risk / 10000)  # Normalized


@dataclass
class AgentVote:
    """Vote cast by an agent."""
    agent_id: int
    vote_type: VoteType
    confidence: float
    reasoning: str = ""
    timestamp: int = 0
    
    @property
    def weight(self) -> float:
        """Compute vote weight based on type and confidence."""
        base_weights = {
            VoteType.STRONG_APPROVE: 1.0,
            VoteType.APPROVE: 0.5,
            VoteType.NEUTRAL: 0.0,
            VoteType.DISAPPROVE: -0.5,
            VoteType.STRONG_DISAPPROVE: -1.0,
            VoteType.VETO: -10.0,  # Veto has overwhelming weight
        }
        return base_weights.get(self.vote_type, 0.0) * self.confidence


@dataclass
class ConsensusResult:
    """Result of consensus voting."""
    approved: bool
    total_score: float
    approve_count: int
    disapprove_count: int
    veto_count: int
    quorum_met: bool
    risk_tier: RiskTier
    reasoning: List[str] = field(default_factory=list)


class ConsensusProtocol:
    """
    Lightweight voting protocol for multi-agent trade validation.
    
    This class implements a weighted voting system where agents vote on
    trade proposals. High-risk trades require stronger consensus.
    
    Memory footprint: ~1MB for active proposals
    """
    
    def __init__(
        self,
        quorum_threshold: float = 0.6,
        veto_threshold: float = 0.8,
        min_agents: int = 3
    ):
        self.quorum_threshold = quorum_threshold
        self.veto_threshold = veto_threshold
        self.min_agents = min_agents
        
        # Active proposals
        self.proposals: Dict[str, TradeProposal] = {}
        
        # Votes per proposal
        self.votes: Dict[str, Dict[int, AgentVote]] = {}
        
        # Agent weights (based on historical accuracy)
        self.agent_weights: Dict[int, float] = {}
        
        # Agent risk tolerance
        self.agent_risk_tolerance: Dict[int, float] = {}
        
        # Historical performance tracking
        self.agent_history: Dict[int, List[bool]] = {}
    
    def register_agent(
        self,
        agent_id: int,
        initial_weight: float = 1.0,
        risk_tolerance: float = 0.5
    ) -> None:
        """Register an agent for voting."""
        self.agent_weights[agent_id] = initial_weight
        self.agent_risk_tolerance[agent_id] = risk_tolerance
        self.agent_history[agent_id] = []
    
    def submit_proposal(self, proposal: TradeProposal) -> str:
        """Submit a trade proposal for consensus voting."""
        self.proposals[proposal.proposal_id] = proposal
        self.votes[proposal.proposal_id] = {}
        return proposal.proposal_id
    
    def cast_vote(
        self,
        proposal_id: str,
        agent_id: int,
        vote_type: VoteType,
        confidence: float = 0.5,
        reasoning: str = ""
    ) -> bool:
        """Cast a vote on a proposal."""
        if proposal_id not in self.proposals:
            return False
        
        if agent_id not in self.agent_weights:
            return False
        
        vote = AgentVote(
            agent_id=agent_id,
            vote_type=vote_type,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=self.proposals[proposal_id].timestamp
        )
        
        self.votes[proposal_id][agent_id] = vote
        return True
    
    def compute_consensus(self, proposal_id: str) -> Optional[ConsensusResult]:
        """
        Compute consensus result for a proposal.
        
        Returns:
            ConsensusResult or None if proposal not found
        """
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        proposal_votes = self.votes.get(proposal_id, {})
        
        # Check minimum participation
        if len(proposal_votes) < self.min_agents:
            return ConsensusResult(
                approved=False,
                total_score=0.0,
                approve_count=0,
                disapprove_count=0,
                veto_count=0,
                quorum_met=False,
                risk_tier=self._classify_risk(proposal),
                reasoning=["Insufficient participation"]
            )
        
        # Compute weighted score
        total_weight = 0.0
        weighted_score = 0.0
        approve_count = 0
        disapprove_count = 0
        veto_count = 0
        reasons = []
        
        for agent_id, vote in proposal_votes.items():
            agent_weight = self.agent_weights.get(agent_id, 1.0)
            vote_weight = vote.weight * agent_weight
            
            weighted_score += vote_weight
            total_weight += abs(vote_weight)
            
            if vote.vote_type in (VoteType.STRONG_APPROVE, VoteType.APPROVE):
                approve_count += 1
            elif vote.vote_type in (VoteType.DISAPPROVE, VoteType.STRONG_DISAPPROVE):
                disapprove_count += 1
            
            if vote.vote_type == VoteType.VETO:
                veto_count += 1
                reasons.append(f"Agent {agent_id} vetoed: {vote.reasoning}")
        
        # Normalize score
        normalized_score = weighted_score / max(total_weight, 1e-10)
        
        # Classify risk
        risk_tier = self._classify_risk(proposal)
        
        # Determine approval
        approved = self._determine_approval(
            normalized_score=normalized_score,
            veto_count=veto_count,
            risk_tier=risk_tier,
            total_votes=len(proposal_votes)
        )
        
        # Check quorum
        quorum_met = len(proposal_votes) / self.min_agents >= self.quorum_threshold
        
        # Add reasoning
        if veto_count > 0:
            reasons.append(f"Veto count ({veto_count}) exceeded threshold")
        if risk_tier == RiskTier.EXTREME:
            reasons.append("Trade classified as extreme risk")
            if normalized_score < 0.5:
                reasons.append("Insufficient support for extreme risk trade")
        
        return ConsensusResult(
            approved=approved,
            total_score=normalized_score,
            approve_count=approve_count,
            disapprove_count=disapprove_count,
            veto_count=veto_count,
            quorum_met=quorum_met,
            risk_tier=risk_tier,
            reasoning=reasons
        )
    
    def _classify_risk(self, proposal: TradeProposal) -> RiskTier:
        """Classify trade risk tier."""
        risk = proposal.risk_estimate
        
        if risk < 0.01:
            return RiskTier.LOW
        elif risk < 0.05:
            return RiskTier.MEDIUM
        elif risk < 0.15:
            return RiskTier.HIGH
        elif risk < 0.30:
            return RiskTier.EXTREME
        else:
            return RiskTier.CATASTROPHIC
    
    def _determine_approval(
        self,
        normalized_score: float,
        veto_count: int,
        risk_tier: RiskTier,
        total_votes: int
    ) -> bool:
        """Determine if proposal is approved based on votes and risk."""
        # Any veto automatically rejects unless overridden by supermajority
        if veto_count > 0:
            override_threshold = 0.9 if risk_tier == RiskTier.EXTREME else 0.8
            if normalized_score < override_threshold:
                return False
        
        # Risk-tier specific thresholds
        thresholds = {
            RiskTier.LOW: 0.0,       # Any positive score approves
            RiskTier.MEDIUM: 0.2,
            RiskTier.HIGH: 0.4,
            RiskTier.EXTREME: 0.7,   # Strong consensus required
            RiskTier.CATASTROPHIC: 1.0,  # Never approve
        }
        
        threshold = thresholds.get(risk_tier, 0.5)
        return normalized_score >= threshold
    
    def update_agent_weight(self, agent_id: int, outcome_successful: bool) -> None:
        """Update agent weight based on trade outcome."""
        if agent_id not in self.agent_history:
            return
        
        self.agent_history[agent_id].append(outcome_successful)
        
        # Recalculate weight based on recent accuracy
        recent = self.agent_history[agent_id][-20:]  # Last 20 trades
        if len(recent) >= 5:
            accuracy = sum(recent) / len(recent)
            # Map accuracy to weight (0.5 to 2.0 range)
            new_weight = 0.5 + 1.5 * accuracy
            self.agent_weights[agent_id] = new_weight
    
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a proposal."""
        if proposal_id not in self.proposals:
            return None
        
        result = self.compute_consensus(proposal_id)
        if result is None:
            return None
        
        return {
            "proposal_id": proposal_id,
            "approved": result.approved,
            "score": result.total_score,
            "risk_tier": result.risk_tier.name,
            "votes": len(self.votes.get(proposal_id, {})),
            "quorum_met": result.quorum_met,
            "reasoning": result.reasoning
        }
    
    def clear_resolved(self, proposal_id: str) -> bool:
        """Clear a resolved proposal from memory."""
        if proposal_id in self.proposals:
            del self.proposals[proposal_id]
            if proposal_id in self.votes:
                del self.votes[proposal_id]
            return True
        return False
    
    def reset(self) -> None:
        """Reset all state."""
        self.proposals.clear()
        self.votes.clear()


def validate_trade_with_consensus(
    proposal: TradeProposal,
    agent_votes: List[Tuple[int, VoteType, float]],
    agent_weights: Optional[Dict[int, float]] = None
) -> ConsensusResult:
    """
    Convenience function to validate a trade with consensus voting.
    
    Args:
        proposal: Trade proposal to validate
        agent_votes: List of (agent_id, vote_type, confidence) tuples
        agent_weights: Optional custom agent weights
        
    Returns:
        ConsensusResult with approval decision
    """
    protocol = ConsensusProtocol(min_agents=2)
    
    # Register voting agents
    all_agent_ids = set(v[0] for v in agent_votes)
    for agent_id in all_agent_ids:
        weight = 1.0
        if agent_weights and agent_id in agent_weights:
            weight = agent_weights[agent_id]
        protocol.register_agent(agent_id, initial_weight=weight)
    
    # Submit proposal
    protocol.submit_proposal(proposal)
    
    # Cast votes
    for agent_id, vote_type, confidence in agent_votes:
        protocol.cast_vote(proposal.proposal_id, agent_id, vote_type, confidence)
    
    # Compute consensus
    result = protocol.compute_consensus(proposal.proposal_id)
    return result or ConsensusResult(
        approved=False,
        total_score=0.0,
        approve_count=0,
        disapprove_count=0,
        veto_count=0,
        quorum_met=False,
        risk_tier=RiskTier.CATASTROPHIC,
        reasoning=["Failed to compute consensus"]
    )


if __name__ == "__main__":
    # Example usage
    protocol = ConsensusProtocol(min_agents=3)
    
    # Register agents
    for i in range(5):
        protocol.register_agent(i, initial_weight=1.0 + 0.1 * i)
    
    # Create a high-risk trade proposal
    proposal = TradeProposal(
        proposal_id="TRADE_001",
        agent_id=0,
        action="BUY",
        pair="BTC/USDT",
        size=10000,
        entry_price=50000.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        confidence=0.8,
        timestamp=1234567890
    )
    
    protocol.submit_proposal(proposal)
    
    # Cast votes
    protocol.cast_vote("TRADE_001", 0, VoteType.STRONG_APPROVE, 0.9, "Strong momentum signal")
    protocol.cast_vote("TRADE_001", 1, VoteType.APPROVE, 0.7, "Technical breakout confirmed")
    protocol.cast_vote("TRADE_001", 2, VoteType.NEUTRAL, 0.5, "Waiting for volume confirmation")
    protocol.cast_vote("TRADE_001", 3, VoteType.DISAPPROVE, 0.6, "Resistance level nearby")
    protocol.cast_vote("TRADE_001", 4, VoteType.APPROVE, 0.8, "Risk/reward favorable")
    
    # Get consensus result
    result = protocol.compute_consensus("TRADE_001")
    
    print(f"Proposal: {proposal.proposal_id}")
    print(f"Approved: {result.approved}")
    print(f"Score: {result.total_score:.3f}")
    print(f"Risk Tier: {result.risk_tier.name}")
    print(f"Votes: {result.approve_count} approve, {result.disapprove_count} disapprove, {result.veto_count} veto")
    print(f"Quorum Met: {result.quorum_met}")
    if result.reasoning:
        print("Reasoning:")
        for r in result.reasoning:
            print(f"  - {r}")
