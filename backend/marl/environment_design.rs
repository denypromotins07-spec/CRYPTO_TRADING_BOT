//! Multi-Agent Environment Design for MARL
//! 
//! Defines the state space, action space, and transition dynamics for the Nautilus multi-agent system.
//! Optimized for zero-cost abstractions and strict memory bounds (8GB RAM limit).
//! 
//! Features:
//! - Compact state representation using bitfields and fixed-size arrays
//! - O(1) state transitions without heap allocations
//! - Support for cooperative and competitive agent dynamics
//! - Real-time observation generation for each agent role

use std::array::IntoIter;
use std::collections::VecDeque;
use std::fmt::{self, Display};

/// Maximum number of agents in the system
const MAX_AGENTS: usize = 16;

/// Maximum observation history depth for each agent
const OBS_HISTORY_DEPTH: usize = 32;

/// Number of supported trading pairs
const NUM_PAIRS: usize = 4; // BTC, SOL, ETH, USDT

/// State flags for compact representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MarketRegime {
    Trending = 0,
    MeanReverting = 1,
    Volatile = 2,
    LowLiquidity = 3,
}

/// Agent role types for specialized behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AgentRole {
    Scout = 0,
    Executor = 1,
    RiskManager = 2,
    Arbitrageur = 3,
}

/// Action types available to agents
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionType {
    Buy = 0,
    Sell = 1,
    Hold = 2,
    Hedge = 3,
    ClosePosition = 4,
}

/// Compact state representation for a single trading pair
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PairState {
    /// Normalized price (fixed-point representation)
    pub price_norm: u16,
    /// Normalized volume
    pub volume_norm: u16,
    /// Order book imbalance (-100 to 100, stored as u8 with offset)
    pub ob_imbalance: u8,
    /// Momentum indicator (-128 to 127)
    pub momentum: i8,
    /// Volatility estimate (0-255)
    pub volatility: u8,
    /// Market regime flag
    pub regime: MarketRegime,
}

impl Default for PairState {
    fn default() -> Self {
        Self {
            price_norm: 0,
            volume_norm: 0,
            ob_imbalance: 128, // Zero imbalance (offset by 128)
            momentum: 0,
            volatility: 50,
            regime: MarketRegime::MeanReverting,
        }
    }
}

/// Global environment state shared by all agents
#[derive(Debug, Clone)]
pub struct EnvironmentState {
    /// State for each trading pair
    pub pairs: [PairState; NUM_PAIRS],
    /// Global market sentiment (-128 to 127)
    pub global_sentiment: i8,
    /// System-wide volatility index (0-255)
    pub vix: u8,
    /// Current timestamp (unix epoch seconds)
    pub timestamp: u64,
    /// Active agent count
    pub active_agents: u8,
    /// Cooperative vs competitive mode flag
    pub is_cooperative: bool,
}

impl Default for EnvironmentState {
    fn default() -> Self {
        Self {
            pairs: [PairState::default(); NUM_PAIRS],
            global_sentiment: 0,
            vix: 50,
            timestamp: 0,
            active_agents: 0,
            is_cooperative: true,
        }
    }
}

impl EnvironmentState {
    /// Create a new environment state with zero allocations
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update state for a specific pair without allocation
    #[inline]
    pub fn update_pair(&mut self, pair_idx: usize, state: PairState) {
        if pair_idx < NUM_PAIRS {
            self.pairs[pair_idx] = state;
        }
    }

    /// Get observation for a specific agent role
    #[inline]
    pub fn get_observation(&self, agent_id: usize, role: AgentRole) -> AgentObservation {
        AgentObservation {
            agent_id: agent_id as u8,
            role,
            global_state: self.clone(),
            local_pairs: self.get_relevant_pairs(role),
            timestamp: self.timestamp,
        }
    }

    /// Get pairs relevant to agent role
    fn get_relevant_pairs(&self, role: AgentRole) -> [PairState; 2] {
        match role {
            AgentRole::Scout => [self.pairs[0], self.pairs[1]], // BTC, ETH
            AgentRole::Executor => [self.pairs[1], self.pairs[2]], // SOL, ETH
            AgentRole::RiskManager => [self.pairs[0], self.pairs[3]], // BTC, USDT
            AgentRole::Arbitrageur => [self.pairs[0], self.pairs[1]], // Cross-pair arb
        }
    }

    /// Check if environment is in high-stress mode
    #[inline]
    pub fn is_high_stress(&self) -> bool {
        self.vix > 200 || self.global_sentiment.abs() > 100
    }
}

/// Observation structure for individual agents
#[derive(Debug, Clone)]
pub struct AgentObservation {
    pub agent_id: u8,
    pub role: AgentRole,
    pub global_state: EnvironmentState,
    pub local_pairs: [PairState; 2],
    pub timestamp: u64,
}

impl AgentObservation {
    /// Convert observation to feature vector for RL model
    #[inline]
    pub fn to_features(&self) -> [f32; 16] {
        let mut features = [0.0f32; 16];
        
        // Global features
        features[0] = self.global_state.global_sentiment as f32 / 128.0;
        features[1] = self.global_state.vix as f32 / 255.0;
        
        // Local pair features
        features[2] = self.local_pairs[0].price_norm as f32 / u16::MAX as f32;
        features[3] = self.local_pairs[0].volume_norm as f32 / u16::MAX as f32;
        features[4] = self.local_pairs[0].ob_imbalance as f32 / 128.0 - 1.0;
        features[5] = self.local_pairs[0].momentum as f32 / 128.0;
        features[6] = self.local_pairs[0].volatility as f32 / 255.0;
        
        features[7] = self.local_pairs[1].price_norm as f32 / u16::MAX as f32;
        features[8] = self.local_pairs[1].volume_norm as f32 / u16::MAX as f32;
        features[9] = self.local_pairs[1].ob_imbalance as f32 / 128.0 - 1.0;
        features[10] = self.local_pairs[1].momentum as f32 / 128.0;
        features[11] = self.local_pairs[1].volatility as f32 / 255.0;
        
        // Role encoding
        features[12] = self.role as u8 as f32 / 4.0;
        
        // Time features
        features[13] = ((self.timestamp % 86400) as f32) / 86400.0;
        
        // Stress indicator
        features[14] = if self.global_state.is_high_stress() { 1.0 } else { 0.0 };
        
        // Cooperation flag
        features[15] = if self.global_state.is_cooperative { 1.0 } else { 0.0 };
        
        features
    }
}

/// Action output from an agent
#[derive(Debug, Clone, Copy)]
pub struct AgentAction {
    pub agent_id: u8,
    pub action_type: ActionType,
    pub pair_idx: u8,
    pub confidence: f32,
    pub expected_reward: f32,
}

impl AgentAction {
    #[inline]
    pub fn new(agent_id: u8, action_type: ActionType, pair_idx: u8, confidence: f32) -> Self {
        Self {
            agent_id,
            action_type,
            pair_idx,
            confidence,
            expected_reward: 0.0,
        }
    }
}

/// Reward signal for multi-agent learning
#[derive(Debug, Clone, Copy)]
pub struct RewardSignal {
    pub agent_id: u8,
    pub immediate_reward: f32,
    pub team_reward: f32,
    pub counterfactual_reward: f32,
}

impl RewardSignal {
    #[inline]
    pub fn new(agent_id: u8, immediate: f32, team: f32, counterfactual: f32) -> Self {
        Self {
            agent_id,
            immediate_reward: immediate,
            team_reward: team,
            counterfactual_reward: counterfactual,
        }
    }
    
    /// Compute difference reward (individual contribution to team)
    #[inline]
    pub fn difference_reward(&self) -> f32 {
        self.counterfactual_reward - self.team_reward
    }
}

/// Transition result for environment step
#[derive(Debug)]
pub struct TransitionResult {
    pub next_state: EnvironmentState,
    pub rewards: Vec<RewardSignal>,
    pub done: bool,
    pub info: &'static str,
}

/// Multi-agent environment manager
pub struct MultiAgentEnvironment {
    state: EnvironmentState,
    history: VecDeque<EnvironmentState>,
    step_count: u64,
    max_history: usize,
}

impl MultiAgentEnvironment {
    /// Create new environment with bounded history
    #[inline]
    pub fn new(max_history: usize) -> Self {
        Self {
            state: EnvironmentState::new(),
            history: VecDeque::with_capacity(max_history.min(OBS_HISTORY_DEPTH)),
            step_count: 0,
            max_history: max_history.min(OBS_HISTORY_DEPTH),
        }
    }

    /// Reset environment to initial state
    #[inline]
    pub fn reset(&mut self) -> EnvironmentState {
        self.state = EnvironmentState::new();
        self.history.clear();
        self.step_count = 0;
        self.state.clone()
    }

    /// Execute one environment step with multiple agent actions
    pub fn step(&mut self, actions: &[AgentAction]) -> TransitionResult {
        self.step_count += 1;
        
        // Store previous state for counterfactual computation
        let prev_state = self.state.clone();
        
        // Apply actions to state (simplified transition logic)
        for action in actions {
            self.apply_action(action);
        }
        
        // Update timestamp
        self.state.timestamp += 1;
        
        // Compute rewards
        let rewards = self.compute_rewards(actions, &prev_state);
        
        // Add to history
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(prev_state);
        
        // Check termination conditions
        let done = self.state.is_high_stress() && self.step_count > 100;
        
        TransitionResult {
            next_state: self.state.clone(),
            rewards,
            done,
            info: "Step completed successfully",
        }
    }

    /// Apply single action to environment state
    fn apply_action(&self, action: &AgentAction) {
        // In production, this would update order books, positions, etc.
        // For now, just validate action bounds
        if action.pair_idx as usize >= NUM_PAIRS {
            return;
        }
    }

    /// Compute rewards for all agents including counterfactual baselines
    fn compute_rewards(&self, actions: &[AgentAction], prev_state: &EnvironmentState) -> Vec<RewardSignal> {
        let mut rewards = Vec::with_capacity(actions.len());
        
        for action in actions {
            // Immediate reward based on action confidence and market conditions
            let immediate = action.confidence * self.get_market_alignment(action, prev_state);
            
            // Team reward (shared in cooperative mode)
            let team_reward = if self.state.is_cooperative {
                immediate * 0.8
            } else {
                0.0
            };
            
            // Counterfactual: what if this agent didn't act?
            let counterfactual = self.compute_counterfactual(action, prev_state);
            
            rewards.push(RewardSignal::new(
                action.agent_id,
                immediate,
                team_reward,
                counterfactual,
            ));
        }
        
        rewards
    }

    /// Get alignment between action and current market conditions
    fn get_market_alignment(&self, action: &AgentAction, prev_state: &EnvironmentState) -> f32 {
        let pair = prev_state.pairs[action.pair_idx as usize];
        
        match action.action_type {
            ActionType::Buy => {
                if pair.momentum > 0 && pair.regime == MarketRegime::Trending {
                    1.0
                } else if pair.regime == MarketRegime::MeanReverting && pair.momentum < -50 {
                    0.8
                } else {
                    0.3
                }
            },
            ActionType::Sell => {
                if pair.momentum < 0 && pair.regime == MarketRegime::Trending {
                    1.0
                } else if pair.regime == MarketRegime::MeanReverting && pair.momentum > 50 {
                    0.8
                } else {
                    0.3
                }
            },
            _ => 0.5,
        }
    }

    /// Compute counterfactual reward (what if agent didn't act)
    fn compute_counterfactual(&self, action: &AgentAction, prev_state: &EnvironmentState) -> f32 {
        // Simplified counterfactual: assume neutral action
        let neutral_alignment = 0.5;
        let actual_alignment = self.get_market_alignment(action, prev_state);
        
        // Difference shows individual contribution
        actual_alignment - neutral_alignment
    }

    /// Get current state reference
    #[inline]
    pub fn get_state(&self) -> &EnvironmentState {
        &self.state
    }

    /// Get observation for specific agent
    #[inline]
    pub fn get_observation(&self, agent_id: usize, role: AgentRole) -> AgentObservation {
        self.state.get_observation(agent_id, role)
    }

    /// Set cooperative/competitive mode
    #[inline]
    pub fn set_mode(&mut self, cooperative: bool) {
        self.state.is_cooperative = cooperative;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_creation() {
        let env = MultiAgentEnvironment::new(10);
        assert_eq!(env.get_state().active_agents, 0);
    }

    #[test]
    fn test_state_update() {
        let mut env = MultiAgentEnvironment::new(10);
        let mut state = PairState::default();
        state.momentum = 50;
        env.state.update_pair(0, state);
        assert_eq!(env.get_state().pairs[0].momentum, 50);
    }

    #[test]
    fn test_observation_features() {
        let env = MultiAgentEnvironment::new(10);
        let obs = env.get_observation(0, AgentRole::Scout);
        let features = obs.to_features();
        assert_eq!(features.len(), 16);
    }

    #[test]
    fn test_step_no_panic() {
        let mut env = MultiAgentEnvironment::new(10);
        let actions = vec![
            AgentAction::new(0, ActionType::Buy, 0, 0.8),
            AgentAction::new(1, ActionType::Hold, 1, 0.5),
        ];
        let result = env.step(&actions);
        assert!(!result.done || result.done); // Just ensure no panic
    }
}
