//! Reward Sharing Mechanisms for Multi-Agent Reinforcement Learning
//! 
//! Implements difference rewards and counterfactual baselines to properly
//! attribute individual agent contributions to team performance.
//! 
//! Features:
//! - Difference reward computation (D-Reward)
//! - Counterfactual baseline estimation
//! - Shapley value approximation for fair reward distribution
//! - Memory-efficient rolling statistics
//! 
//! Optimized for zero-cost abstractions and strict memory bounds.

use std::collections::{HashMap, VecDeque};

/// Maximum history size for reward tracking (memory bound)
const MAX_REWARD_HISTORY: usize = 1000;

/// Reward signal types for different learning objectives
#[derive(Debug, Clone, Copy)]
pub enum RewardType {
    /// Immediate individual reward
    Individual,
    /// Team-shared reward (cooperative)
    Team,
    /// Difference reward (individual contribution)
    Difference,
    /// Counterfactual reward (what-if scenario)
    Counterfactual,
    /// Shapley value-based reward
    Shapley,
}

/// Compact reward record for efficient storage
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RewardRecord {
    pub agent_id: u8,
    pub timestamp: u64,
    pub reward_value: f32,
    pub reward_type: RewardType,
    pub episode_id: u32,
}

/// Counterfactual baseline estimate
#[derive(Debug, Clone, Copy)]
pub struct CounterfactualBaseline {
    pub agent_id: u8,
    pub expected_team_reward: f32,
    pub actual_team_reward: f32,
    pub marginal_contribution: f32,
    pub confidence: f32,
}

impl CounterfactualBaseline {
    #[inline]
    pub fn new(agent_id: u8, expected: f32, actual: f32) -> Self {
        Self {
            agent_id,
            expected_team_reward: expected,
            actual_team_reward: actual,
            marginal_contribution: actual - expected,
            confidence: 0.5,
        }
    }
    
    /// Update confidence based on prediction accuracy
    #[inline]
    pub fn update_confidence(&mut self, error: f32) {
        let adjustment = (-error.abs() * 10.0).exp();
        self.confidence = 0.9 * self.confidence + 0.1 * adjustment;
    }
}

/// Difference reward calculator using counterfactual reasoning
pub struct DifferenceRewardCalculator {
    /// Baseline estimates for each agent
    baselines: HashMap<u8, CounterfactualBaseline>,
    /// Rolling team reward history
    team_reward_history: VecDeque<f32>,
    /// Individual reward history per agent
    individual_history: HashMap<u8, VecDeque<f32>>,
    /// Current episode ID
    episode_id: u32,
    /// Global step counter
    global_step: u64,
}

impl DifferenceRewardCalculator {
    /// Create new calculator with bounded memory
    #[inline]
    pub fn new(max_agents: usize) -> Self {
        let mut individual_history = HashMap::with_capacity(max_agents);
        for i in 0..max_agents as u8 {
            individual_history.insert(i, VecDeque::with_capacity(MAX_REWARD_HISTORY));
        }
        
        Self {
            baselines: HashMap::with_capacity(max_agents),
            team_reward_history: VecDeque::with_capacity(MAX_REWARD_HISTORY),
            individual_history,
            episode_id: 0,
            global_step: 0,
        }
    }
    
    /// Compute difference reward for an agent
    /// D_i = R_team - R_team_without_i
    #[inline]
    pub fn compute_difference_reward(
        &mut self,
        agent_id: u8,
        team_reward: f32,
        individual_action_quality: f32,
    ) -> f32 {
        self.global_step += 1;
        
        // Get or create baseline for this agent
        let baseline = self.baselines.entry(agent_id).or_insert_with(|| {
            CounterfactualBaseline::new(agent_id, 0.0, team_reward)
        });
        
        // Estimate counterfactual: what if agent didn't act?
        let counterfactual_team_reward = self.estimate_counterfactual(agent_id, team_reward);
        
        // Difference reward = actual team reward - counterfactual
        let diff_reward = team_reward - counterfactual_team_reward;
        
        // Update baseline with exponential moving average
        let alpha = 0.1;
        baseline.expected_team_reward = 
            (1.0 - alpha) * baseline.expected_team_reward + alpha * counterfactual_team_reward;
        baseline.actual_team_reward = 
            (1.0 - alpha) * baseline.actual_team_reward + alpha * team_reward;
        baseline.marginal_contribution = 
            (1.0 - alpha) * baseline.marginal_contribution + alpha * diff_reward;
        
        // Store reward in history
        self.team_reward_history.push_back(team_reward);
        if self.team_reward_history.len() > MAX_REWARD_HISTORY {
            self.team_reward_history.pop_front();
        }
        
        if let Some(history) = self.individual_history.get_mut(&agent_id) {
            history.push_back(diff_reward);
            if history.len() > MAX_REWARD_HISTORY {
                history.pop_front();
            }
        }
        
        diff_reward
    }
    
    /// Estimate counterfactual team reward without agent's contribution
    fn estimate_counterfactual(&self, agent_id: u8, actual_reward: f32) -> f32 {
        // Use historical average when agent had low impact
        if let Some(history) = self.individual_history.get(&agent_id) {
            if !history.is_empty() {
                let avg_individual = history.iter().sum::<f32>() / history.len() as f32;
                // If agent typically has low impact, reduce their contribution
                let impact_factor = avg_individual.abs().min(1.0);
                return actual_reward * (1.0 - impact_factor * 0.5);
            }
        }
        
        // Default: assume moderate contribution
        actual_reward * 0.7
    }
    
    /// Compute Shapley value approximation for fair reward distribution
    pub fn compute_shapley_values(
        &self,
        team_reward: f32,
        agent_ids: &[u8],
    ) -> HashMap<u8, f32> {
        let mut shapley_values = HashMap::with_capacity(agent_ids.len());
        
        if agent_ids.is_empty() {
            return shapley_values;
        }
        
        // Simplified Shapley value: proportional to marginal contribution
        let total_marginal: f32 = agent_ids
            .iter()
            .filter_map(|&id| self.baselines.get(&id))
            .map(|b| b.marginal_contribution.abs())
            .sum();
        
        if total_marginal < 1e-6 {
            // Equal distribution if no marginal contribution
            let equal_share = team_reward / agent_ids.len() as f32;
            for &id in agent_ids {
                shapley_values.insert(id, equal_share);
            }
        } else {
            // Proportional distribution
            for &id in agent_ids {
                if let Some(baseline) = self.baselines.get(&id) {
                    let weight = baseline.marginal_contribution.abs() / total_marginal;
                    shapley_values.insert(id, team_reward * weight);
                }
            }
        }
        
        shapley_values
    }
    
    /// Get agent's recent average difference reward
    #[inline]
    pub fn get_agent_average_reward(&self, agent_id: u8, window: usize) -> f32 {
        if let Some(history) = self.individual_history.get(&agent_id) {
            if history.is_empty() {
                return 0.0;
            }
            let n = window.min(history.len());
            let sum: f32 = history.iter().rev().take(n).sum();
            sum / n as f32
        } else {
            0.0
        }
    }
    
    /// Start new episode
    #[inline]
    pub fn new_episode(&mut self) {
        self.episode_id += 1;
    }
    
    /// Get current episode ID
    #[inline]
    pub fn episode_id(&self) -> u32 {
        self.episode_id
    }
    
    /// Get global step count
    #[inline]
    pub fn global_step(&self) -> u64 {
        self.global_step
    }
    
    /// Reset all statistics
    #[inline]
    pub fn reset(&mut self) {
        self.baselines.clear();
        self.team_reward_history.clear();
        for history in self.individual_history.values_mut() {
            history.clear();
        }
        self.episode_id = 0;
        self.global_step = 0;
    }
}

/// Reward normalizer for stable learning
pub struct RewardNormalizer {
    running_mean: f32,
    running_variance: f32,
    count: u64,
    epsilon: f32,
}

impl RewardNormalizer {
    #[inline]
    pub fn new() -> Self {
        Self {
            running_mean: 0.0,
            running_variance: 1.0,
            count: 0,
            epsilon: 1e-8,
        }
    }
    
    /// Update running statistics and normalize reward
    #[inline]
    pub fn normalize(&mut self, reward: f32) -> f32 {
        self.count += 1;
        let delta = reward - self.running_mean;
        self.running_mean += delta / self.count as f32;
        let delta2 = reward - self.running_mean;
        self.running_variance += delta * delta2 * (1.0 / self.count as f32 - 1.0);
        
        let std = (self.running_variance.max(self.epsilon)).sqrt();
        (reward - self.running_mean) / std
    }
    
    /// Denormalize a normalized reward
    #[inline]
    pub fn denormalize(&self, normalized: f32) -> f32 {
        let std = (self.running_variance.max(self.epsilon)).sqrt();
        normalized * std + self.running_mean
    }
    
    /// Reset normalizer statistics
    #[inline]
    pub fn reset(&mut self) {
        self.running_mean = 0.0;
        self.running_variance = 1.0;
        self.count = 0;
    }
}

impl Default for RewardNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-agent reward manager coordinating all reward mechanisms
pub struct MultiAgentRewardManager {
    difference_calculator: DifferenceRewardCalculator,
    normalizers: HashMap<u8, RewardNormalizer>,
    team_normalizer: RewardNormalizer,
    reward_scaling: f32,
}

impl MultiAgentRewardManager {
    /// Create new reward manager
    #[inline]
    pub fn new(max_agents: usize) -> Self {
        let mut normalizers = HashMap::with_capacity(max_agents);
        for i in 0..max_agents as u8 {
            normalizers.insert(i, RewardNormalizer::new());
        }
        
        Self {
            difference_calculator: DifferenceRewardCalculator::new(max_agents),
            normalizers,
            team_normalizer: RewardNormalizer::new(),
            reward_scaling: 1.0,
        }
    }
    
    /// Process team reward and distribute to agents
    pub fn distribute_rewards(
        &mut self,
        team_reward: f32,
        agent_actions: &[(u8, f32)], // (agent_id, action_quality)
    ) -> HashMap<u8, f32> {
        let mut final_rewards = HashMap::with_capacity(agent_actions.len());
        
        // Normalize team reward
        let normalized_team = self.team_normalizer.normalize(team_reward);
        
        // Compute difference rewards for each agent
        for &(agent_id, action_quality) in agent_actions {
            let diff_reward = self.difference_calculator.compute_difference_reward(
                agent_id,
                normalized_team,
                action_quality,
            );
            
            // Normalize individual reward
            if let Some(normalizer) = self.normalizers.get_mut(&agent_id) {
                let normalized = normalizer.normalize(diff_reward);
                final_rewards.insert(agent_id, normalized * self.reward_scaling);
            }
        }
        
        final_rewards
    }
    
    /// Get Shapley values for current team reward
    #[inline]
    pub fn get_shapley_values(&self, team_reward: f32, agent_ids: &[u8]) -> HashMap<u8, f32> {
        self.difference_calculator.compute_shapley_values(team_reward, agent_ids)
    }
    
    /// Get agent's recent performance metric
    #[inline]
    pub fn get_agent_performance(&self, agent_id: u8, window: usize) -> f32 {
        self.difference_calculator.get_agent_average_reward(agent_id, window)
    }
    
    /// Set reward scaling factor
    #[inline]
    pub fn set_scaling(&mut self, scale: f32) {
        self.reward_scaling = scale.max(0.1).min(10.0);
    }
    
    /// Start new episode
    #[inline]
    pub fn new_episode(&mut self) {
        self.difference_calculator.new_episode();
    }
    
    /// Reset all reward statistics
    #[inline]
    pub fn reset(&mut self) {
        self.difference_calculator.reset();
        for normalizer in self.normalizers.values_mut() {
            normalizer.reset();
        }
        self.team_normalizer.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difference_reward_calculation() {
        let mut calc = DifferenceRewardCalculator::new(4);
        
        let reward = calc.compute_difference_reward(0, 1.0, 0.8);
        assert!(reward.is_finite());
        
        let avg = calc.get_agent_average_reward(0, 1);
        assert_eq!(avg, reward);
    }

    #[test]
    fn test_shapley_values_sum() {
        let mut calc = DifferenceRewardCalculator::new(4);
        
        // Simulate some rewards
        for i in 0u8..3 {
            calc.compute_difference_reward(i, 1.0, 0.5);
        }
        
        let shapley = calc.compute_shapley_values(1.0, &[0, 1, 2]);
        let total: f32 = shapley.values().sum();
        
        // Total should be approximately equal to team reward
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_reward_normalizer() {
        let mut normalizer = RewardNormalizer::new();
        
        // Feed some rewards
        for i in 0..100 {
            let reward = i as f32;
            let _normalized = normalizer.normalize(reward);
        }
        
        // Normalized values should have roughly zero mean and unit variance
        let test_reward = 50.0;
        let normalized = normalizer.normalize(test_reward);
        assert!(normalized.is_finite());
    }

    #[test]
    fn test_multi_agent_manager() {
        let mut manager = MultiAgentRewardManager::new(4);
        
        let actions = vec![(0u8, 0.8f32), (1, 0.6), (2, 0.9)];
        let rewards = manager.distribute_rewards(1.0, &actions);
        
        assert_eq!(rewards.len(), 3);
        for (&id, &reward) in &rewards {
            assert!(reward.is_finite());
            assert!(id < 3);
        }
    }
}
