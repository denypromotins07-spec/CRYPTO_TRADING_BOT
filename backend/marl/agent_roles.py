#!/usr/bin/env python3
"""
Agent Roles Module for Multi-Agent Reinforcement Learning System

This module defines specialized agent roles (Scout, Executor, Risk Manager) 
with distinct responsibilities and decision-making patterns. Each role is 
optimized for specific tasks within the ZAID trading bot ecosystem.

Features:
- Role-based agent specialization using Strategy pattern
- Memory-efficient state management (<8GB RAM constraint)
- Real-time role switching based on market conditions
- Strict type hinting for safety and clarity

Integrates quantitative finance domains:
- Market microstructure analysis
- Risk management frameworks
- Execution algorithms
- Signal processing
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from numpy.typing import NDArray


class AgentRoleType(Enum):
    """Enumeration of available agent roles."""
    SCOUT = auto()
    EXECUTOR = auto()
    RISK_MANAGER = auto()
    ARBITRAGEUR = auto()


class MarketCondition(Enum):
    """Market condition states for role adaptation."""
    TRENDING = auto()
    MEAN_REVERTING = auto()
    HIGH_VOLATILITY = auto()
    LOW_LIQUIDITY = auto()
    NORMAL = auto()


@dataclass
class AgentState:
    """Compact state representation for an individual agent."""
    agent_id: int
    role: AgentRoleType
    active: bool = True
    confidence: float = 0.5
    last_action_timestamp: int = 0
    cumulative_reward: float = 0.0
    recent_sharpe: float = 0.0
    position_size: float = 0.0
    max_position: float = 10000.0  # INR equivalent
    risk_exposure: float = 0.0
    
    def reset(self) -> None:
        """Reset agent state to initial values."""
        self.active = True
        self.confidence = 0.5
        self.cumulative_reward = 0.0
        self.recent_sharpe = 0.0
        self.position_size = 0.0
        self.risk_exposure = 0.0


@dataclass
class ActionRecommendation:
    """Action recommendation from an agent."""
    agent_id: int
    role: AgentRoleType
    action: str  # BUY, SELL, HOLD, HEDGE, CLOSE
    pair: str
    confidence: float
    expected_return: float
    risk_estimate: float
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agent roles."""
    
    def __init__(self, agent_id: int, max_memory: int = 1000):
        self.agent_id = agent_id
        self.state = AgentState(agent_id=agent_id, role=self.get_role_type())
        self.observation_history: List[NDArray[np.float32]] = []
        self.max_memory = max_memory  # Enforce memory limits
        self._initialized = False
        
    @abstractmethod
    def get_role_type(self) -> AgentRoleType:
        """Return the specific role type for this agent."""
        pass
    
    @abstractmethod
    def process_observation(self, observation: NDArray[np.float32]) -> Optional[ActionRecommendation]:
        """Process observation and return action recommendation."""
        pass
    
    def update_state(self, reward: float, sharpe: float) -> None:
        """Update agent state with new reward and Sharpe ratio."""
        self.state.cumulative_reward += reward
        self.state.recent_sharpe = sharpe
        if sharpe > 0:
            self.state.confidence = min(1.0, self.state.confidence + 0.05)
        else:
            self.state.confidence = max(0.1, self.state.confidence - 0.05)
    
    def add_observation(self, obs: NDArray[np.float32]) -> None:
        """Add observation to history with memory bound enforcement."""
        if len(self.observation_history) >= self.max_memory:
            self.observation_history.pop(0)
        self.observation_history.append(obs.astype(np.float32))
    
    def get_recent_observations(self, n: int = 10) -> NDArray[np.float32]:
        """Get last n observations as numpy array."""
        if not self.observation_history:
            return np.zeros((0,), dtype=np.float32)
        n = min(n, len(self.observation_history))
        return np.array(self.observation_history[-n:], dtype=np.float32)
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        self.state.reset()
        self.observation_history.clear()


class ScoutAgent(BaseAgent):
    """
    Scout Agent: Specialized in market reconnaissance and opportunity detection.
    
    Responsibilities:
    - Scan multiple trading pairs for anomalies
    - Detect early trend signals
    - Identify liquidity imbalances
    - Flag potential entry/exit points
    
    Memory footprint: ~2MB per instance
    """
    
    def __init__(self, agent_id: int, sensitivity: float = 0.7):
        super().__init__(agent_id)
        self.sensitivity = sensitivity
        self.anomaly_threshold = 2.5  # Standard deviations
        self.trend_detection_window = 20
        self._initialized = True
    
    def get_role_type(self) -> AgentRoleType:
        return AgentRoleType.SCOUT
    
    def process_observation(self, observation: NDArray[np.float32]) -> Optional[ActionRecommendation]:
        """Detect market opportunities and anomalies."""
        self.add_observation(observation)
        
        if len(self.observation_history) < self.trend_detection_window:
            return None
        
        recent = self.get_recent_observations(self.trend_detection_window)
        
        # Compute momentum and volatility features
        momentum_signal = self._compute_momentum(recent)
        volatility_signal = self._compute_volatility(recent)
        anomaly_score = self._detect_anomaly(recent)
        
        # Generate signal if threshold exceeded
        if abs(momentum_signal) > self.sensitivity or anomaly_score > self.anomaly_threshold:
            action = "BUY" if momentum_signal > 0 else "SELL"
            confidence = min(1.0, abs(momentum_signal) / 2.0)
            
            return ActionRecommendation(
                agent_id=self.agent_id,
                role=self.get_role_type(),
                action=action,
                pair=self._infer_pair(observation),
                confidence=confidence,
                expected_return=momentum_signal * 0.02,
                risk_estimate=volatility_signal,
                timestamp=int(observation[-1]) if len(observation) > 0 else 0,
                metadata={
                    "anomaly_score": float(anomaly_score),
                    "momentum": float(momentum_signal),
                    "signal_type": "scout_detection"
                }
            )
        
        return None
    
    def _compute_momentum(self, observations: NDArray[np.float32]) -> float:
        """Compute normalized momentum signal."""
        if len(observations) < 2:
            return 0.0
        # Assume first feature is price-related
        prices = observations[:, 0] if observations.ndim > 1 else observations
        if len(prices) < 2:
            return 0.0
        diff = np.diff(prices)
        return float(np.mean(diff) / (np.std(diff) + 1e-8))
    
    def _compute_volatility(self, observations: NDArray[np.float32]) -> float:
        """Compute normalized volatility estimate."""
        if len(observations) < 2:
            return 0.0
        prices = observations[:, 0] if observations.ndim > 1 else observations
        return float(np.std(prices) / (np.mean(prices) + 1e-8))
    
    def _detect_anomaly(self, observations: NDArray[np.float32]) -> float:
        """Detect statistical anomalies in recent observations."""
        if len(observations) < 5:
            return 0.0
        latest = observations[-1]
        historical_mean = np.mean(observations[:-1], axis=0)
        historical_std = np.std(observations[:-1], axis=0) + 1e-8
        z_scores = np.abs((latest - historical_mean) / historical_std)
        return float(np.max(z_scores))
    
    def _infer_pair(self, observation: NDArray[np.float32]) -> str:
        """Infer trading pair from observation features."""
        # Simplified: in production, this would use explicit pair encoding
        pairs = ["BTC", "ETH", "SOL", "USDT"]
        idx = int(observation[12]) % 4 if len(observation) > 12 else 0
        return pairs[idx]


class ExecutorAgent(BaseAgent):
    """
    Executor Agent: Specialized in order execution and timing optimization.
    
    Responsibilities:
    - Optimize entry/exit timing
    - Minimize slippage
    - Execute large orders via TWAP/VWAP
    - Monitor fill rates
    
    Memory footprint: ~3MB per instance
    """
    
    def __init__(self, agent_id: int, execution_style: str = "aggressive"):
        super().__init__(agent_id)
        self.execution_style = execution_style
        self.slippage_tolerance = 0.001  # 0.1%
        self.vwap_window = 50
        self.pending_orders: List[Dict[str, Any]] = []
        self._initialized = True
    
    def get_role_type(self) -> AgentRoleType:
        return AgentRoleType.EXECUTOR
    
    def process_observation(self, observation: NDArray[np.float32]) -> Optional[ActionRecommendation]:
        """Determine optimal execution timing."""
        self.add_observation(observation)
        
        if len(self.observation_history) < 10:
            return None
        
        recent = self.get_recent_observations(10)
        
        # Compute execution quality metrics
        vwap_estimate = self._estimate_vwap(recent)
        current_price = observation[0] if len(observation) > 0 else 0.0
        price_vs_vwap = (current_price - vwap_estimate) / (vwap_estimate + 1e-8)
        
        # Determine execution signal
        if self.pending_orders:
            order = self.pending_orders[0]
            if order["side"] == "BUY" and price_vs_vwap < -self.slippage_tolerance:
                return self._create_recommendation("BUY", order["pair"], 0.9)
            elif order["side"] == "SELL" and price_vs_vwap > self.slippage_tolerance:
                return self._create_recommendation("SELL", order["pair"], 0.9)
        
        # Normal execution logic
        if abs(price_vs_vwap) > 0.002:
            action = "SELL" if price_vs_vwap > 0 else "BUY"
            return self._create_recommendation(action, self._infer_pair(observation), 0.7)
        
        return None
    
    def _estimate_vwap(self, observations: NDArray[np.float32]) -> float:
        """Estimate VWAP from recent observations."""
        if len(observations) < 2:
            return 0.0
        # Assume features: [price, volume, ...]
        prices = observations[:, 0] if observations.ndim > 1 else observations
        volumes = observations[:, 1] if observations.ndim > 1 and observations.shape[1] > 1 else np.ones(len(prices))
        return float(np.sum(prices * volumes) / (np.sum(volumes) + 1e-8))
    
    def _create_recommendation(self, action: str, pair: str, confidence: float) -> ActionRecommendation:
        """Create standardized action recommendation."""
        return ActionRecommendation(
            agent_id=self.agent_id,
            role=self.get_role_type(),
            action=action,
            pair=pair,
            confidence=confidence,
            expected_return=0.001 * confidence,
            risk_estimate=0.0005,
            timestamp=0,
            metadata={"execution_style": self.execution_style}
        )
    
    def queue_order(self, side: str, pair: str, size: float) -> None:
        """Queue order for execution."""
        self.pending_orders.append({"side": side, "pair": pair, "size": size})
        if len(self.pending_orders) > 10:
            self.pending_orders.pop(0)
    
    def _infer_pair(self, observation: NDArray[np.float32]) -> str:
        pairs = ["BTC", "ETH", "SOL", "USDT"]
        idx = int(observation[12]) % 4 if len(observation) > 12 else 0
        return pairs[idx]


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent: Specialized in portfolio risk assessment and mitigation.
    
    Responsibilities:
    - Monitor portfolio VaR and CVaR
    - Enforce position limits
    - Detect correlation breakdowns
    - Trigger hedging when necessary
    
    Memory footprint: ~2.5MB per instance
    """
    
    def __init__(self, agent_id: int, var_confidence: float = 0.95):
        super().__init__(agent_id)
        self.var_confidence = var_confidence
        self.max_portfolio_risk = 0.05  # 5% of capital
        self.correlation_window = 100
        self.position_limits: Dict[str, float] = {"BTC": 50000, "ETH": 30000, "SOL": 20000}
        self._initialized = True
    
    def get_role_type(self) -> AgentRoleType:
        return AgentRoleType.RISK_MANAGER
    
    def process_observation(self, observation: NDArray[np.float32]) -> Optional[ActionRecommendation]:
        """Assess portfolio risk and recommend hedging if needed."""
        self.add_observation(observation)
        
        # Compute risk metrics
        portfolio_var = self._compute_var()
        portfolio_cvar = self._compute_cvar()
        
        # Check position limits
        limit_breach = self._check_position_limits()
        
        # Generate risk mitigation signal
        if portfolio_var > self.max_portfolio_risk or limit_breach:
            return ActionRecommendation(
                agent_id=self.agent_id,
                role=self.get_role_type(),
                action="HEDGE",
                pair="USDT",
                confidence=0.95,
                expected_return=0.0,
                risk_estimate=portfolio_var,
                timestamp=0,
                metadata={
                    "var": float(portfolio_var),
                    "cvar": float(portfolio_cvar),
                    "limit_breach": limit_breach,
                    "signal_type": "risk_mitigation"
                }
            )
        
        return None
    
    def _compute_var(self) -> float:
        """Compute Value at Risk using historical simulation."""
        if len(self.observation_history) < 20:
            return 0.02  # Default 2%
        
        returns = self._compute_returns()
        if len(returns) < 10:
            return 0.02
        
        var_threshold = (1 - self.var_confidence) * 100
        return float(np.percentile(np.abs(returns), var_threshold)) / 100
    
    def _compute_cvar(self) -> float:
        """Compute Conditional VaR (Expected Shortfall)."""
        if len(self.observation_history) < 20:
            return 0.03
        
        returns = self._compute_returns()
        var_threshold = (1 - self.var_confidence) * 100
        var = np.percentile(returns, var_threshold)
        cvar = np.mean(returns[returns <= var])
        return float(abs(cvar)) if not np.isnan(cvar) else 0.03
    
    def _compute_returns(self) -> NDArray[np.float32]:
        """Compute returns from observation history."""
        if len(self.observation_history) < 2:
            return np.array([], dtype=np.float32)
        
        prices = np.array([obs[0] for obs in self.observation_history], dtype=np.float32)
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        return returns
    
    def _check_position_limits(self) -> bool:
        """Check if any position exceeds limits."""
        # Simplified: in production, would check actual positions
        return self.state.risk_exposure > self.max_portfolio_risk


class ArbitrageurAgent(BaseAgent):
    """
    Arbitrageur Agent: Specialized in cross-pair and cross-exchange arbitrage.
    
    Responsibilities:
    - Detect price discrepancies between pairs
    - Execute triangular arbitrage
    - Monitor funding rate differentials
    - Exploit temporary mispricings
    
    Memory footprint: ~3MB per instance
    """
    
    def __init__(self, agent_id: int, min_spread: float = 0.002):
        super().__init__(agent_id)
        self.min_spread = min_spread
        self.pair_correlations: Dict[Tuple[str, str], float] = {}
        self._initialized = True
    
    def get_role_type(self) -> AgentRoleType:
        return AgentRoleType.ARBITRAGEUR
    
    def process_observation(self, observation: NDArray[np.float32]) -> Optional[ActionRecommendation]:
        """Detect arbitrage opportunities."""
        self.add_observation(observation)
        
        if len(self.observation_history) < 5:
            return None
        
        # Look for spread opportunities
        spread_opportunity = self._detect_spread()
        
        if spread_opportunity and abs(spread_opportunity[0]) > self.min_spread:
            spread, pair1, pair2 = spread_opportunity
            action = "BUY" if spread > 0 else "SELL"
            
            return ActionRecommendation(
                agent_id=self.agent_id,
                role=self.get_role_type(),
                action=action,
                pair=f"{pair1}-{pair2}",
                confidence=min(1.0, abs(spread) / 0.01),
                expected_return=abs(spread) * 0.5,
                risk_estimate=0.001,
                timestamp=0,
                metadata={
                    "spread": float(spread),
                    "pair1": pair1,
                    "pair2": pair2,
                    "arb_type": "cross_pair"
                }
            )
        
        return None
    
    def _detect_spread(self) -> Optional[Tuple[float, str, str]]:
        """Detect price spread between correlated pairs."""
        if len(self.observation_history) < 2:
            return None
        
        # Simplified spread detection
        recent = self.observation_history[-1]
        prev = self.observation_history[-2]
        
        if len(recent) < 4 or len(prev) < 4:
            return None
        
        # Assume features contain multi-pair prices
        spread = float(recent[0] - recent[1])
        avg_price = (recent[0] + recent[1]) / 2 + 1e-8
        normalized_spread = spread / avg_price
        
        return (normalized_spread, "BTC", "ETH") if abs(normalized_spread) > 0.001 else None


class AgentRoleFactory:
    """Factory for creating specialized agent instances."""
    
    _role_classes = {
        AgentRoleType.SCOUT: ScoutAgent,
        AgentRoleType.EXECUTOR: ExecutorAgent,
        AgentRoleType.RISK_MANAGER: RiskManagerAgent,
        AgentRoleType.ARBITRAGEUR: ArbitrageurAgent,
    }
    
    @classmethod
    def create_agent(
        cls,
        role: AgentRoleType,
        agent_id: int,
        **kwargs: Any
    ) -> BaseAgent:
        """Create an agent instance of the specified role."""
        agent_class = cls._role_classes.get(role)
        if agent_class is None:
            raise ValueError(f"Unknown agent role: {role}")
        return agent_class(agent_id, **kwargs)
    
    @classmethod
    def create_team(
        cls,
        team_config: Dict[AgentRoleType, int],
        base_id: int = 0
    ) -> List[BaseAgent]:
        """Create a team of agents with specified role distribution."""
        agents = []
        current_id = base_id
        
        for role, count in team_config.items():
            for _ in range(count):
                agent = cls.create_agent(role, current_id)
                agents.append(agent)
                current_id += 1
        
        return agents


def assign_roles_to_agents(
    market_condition: MarketCondition,
    available_agents: List[BaseAgent]
) -> Dict[AgentRoleType, List[int]]:
    """
    Dynamically assign roles based on market conditions.
    
    This function implements adaptive role assignment to optimize
    performance under different market regimes.
    """
    role_assignments: Dict[AgentRoleType, List[int]] = {
        role: [] for role in AgentRoleType
    }
    
    # Adjust role priorities based on market condition
    priority_map = {
        MarketCondition.TRENDING: [AgentRoleType.SCOUT, AgentRoleType.EXECUTOR],
        MarketCondition.MEAN_REVERTING: [AgentRoleType.ARBITRAGEUR, AgentRoleType.SCOUT],
        MarketCondition.HIGH_VOLATILITY: [AgentRoleType.RISK_MANAGER, AgentRoleType.EXECUTOR],
        MarketCondition.LOW_LIQUIDITY: [AgentRoleType.RISK_MANAGER, AgentRoleType.SCOUT],
        MarketCondition.NORMAL: list(AgentRoleType),
    }
    
    priorities = priority_map.get(market_condition, list(AgentRoleType))
    
    # Assign agents to roles based on their capabilities
    for agent in available_agents:
        if not agent.state.active:
            continue
        
        # Prefer higher priority roles
        for priority_role in priorities:
            if agent.get_role_type() == priority_role:
                role_assignments[priority_role].append(agent.agent_id)
                break
    
    return role_assignments


if __name__ == "__main__":
    # Example usage
    factory = AgentRoleFactory()
    
    # Create a balanced team
    team_config = {
        AgentRoleType.SCOUT: 2,
        AgentRoleType.EXECUTOR: 2,
        AgentRoleType.RISK_MANAGER: 1,
        AgentRoleType.ARBITRAGEUR: 1,
    }
    
    team = factory.create_team(team_config)
    print(f"Created team with {len(team)} agents")
    
    # Test observation processing
    dummy_obs = np.random.randn(16).astype(np.float32)
    for agent in team:
        rec = agent.process_observation(dummy_obs)
        if rec:
            print(f"Agent {agent.agent_id} ({agent.get_role_type().name}): {rec.action}")
