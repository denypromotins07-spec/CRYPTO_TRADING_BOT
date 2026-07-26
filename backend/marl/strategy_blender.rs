//! Strategy Blender for Multi-Agent Signal Fusion
//! 
//! Dynamically weights agent outputs based on recent Sharpe ratios
//! and market regime compatibility. Implements override logic for
//! catastrophic risk scenarios flagged by consensus.
//! 
//! Features:
//! - Sharpe-based dynamic weighting
//! - Regime-aware strategy selection
//! - Risk override mechanism
//! - Memory-efficient rolling statistics
//! 
//! Optimized for sub-millisecond blending decisions.

use std::collections::{HashMap, VecDeque};

/// Maximum agents supported
const MAX_AGENTS: usize = 16;

/// Maximum history for performance tracking
const MAX_HISTORY: usize = 100;

/// Agent signal with confidence
#[derive(Debug, Clone, Copy)]
pub struct AgentSignal {
    pub agent_id: u8,
    pub action: Action,
    pub confidence: f32,
    pub expected_return: f32,
    pub estimated_risk: f32,
}

/// Trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Buy,
    Sell,
    Hold,
    Hedge,
    Close,
}

/// Blended output from strategy fusion
#[derive(Debug, Clone, Copy)]
pub struct BlendedDecision {
    pub action: Action,
    pub confidence: f32,
    pub position_size: f32,
    pub contributing_agents: u8,
    pub override_active: bool,
}

impl Default for BlendedDecision {
    fn default() -> Self {
        Self {
            action: Action::Hold,
            confidence: 0.0,
            position_size: 0.0,
            contributing_agents: 0,
            override_active: false,
        }
    }
}

/// Rolling statistics tracker for an agent
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    /// Recent returns for Sharpe calculation
    returns: VecDeque<f32>,
    /// Cumulative return
    cumulative_return: f32,
    /// Number of trades
    trade_count: u32,
    /// Current Sharpe ratio (annualized)
    current_sharpe: f32,
    /// Win rate
    win_rate: f32,
}

impl PerformanceTracker {
    #[inline]
    pub fn new(window_size: usize) -> Self {
        Self {
            returns: VecDeque::with_capacity(window_size.min(MAX_HISTORY)),
            cumulative_return: 0.0,
            trade_count: 0,
            current_sharpe: 0.0,
            win_rate: 0.5,
        }
    }
    
    /// Add a new return observation
    #[inline]
    pub fn add_return(&mut self, return_pct: f32) {
        self.returns.push_back(return_pct);
        
        // Maintain window size
        if self.returns.len() > MAX_HISTORY {
            self.returns.pop_front();
        }
        
        self.cumulative_return += return_pct;
        self.trade_count += 1;
        
        // Update Sharpe
        self.current_sharpe = self.compute_sharpe();
        
        // Update win rate
        let wins = self.returns.iter().filter(|&&r| r > 0.0).count() as f32;
        self.win_rate = wins / self.returns.len() as f32;
    }
    
    /// Compute annualized Sharpe ratio
    fn compute_sharpe(&self) -> f32 {
        if self.returns.len() < 2 {
            return 0.0;
        }
        
        let n = self.returns.len() as f32;
        let mean: f32 = self.returns.iter().sum::<f32>() / n;
        
        let variance: f32 = self.returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f32>() / (n - 1.0);
        
        let std = variance.sqrt();
        
        if std < 1e-6 {
            return 0.0;
        }
        
        // Annualize (assuming daily returns)
        (mean / std) * 252.0_f32.sqrt()
    }
    
    #[inline]
    pub fn sharpe(&self) -> f32 {
        self.current_sharpe
    }
    
    #[inline]
    pub fn win_rate(&self) -> f32 {
        self.win_rate
    }
    
    #[inline]
    pub fn reset(&mut self) {
        self.returns.clear();
        self.cumulative_return = 0.0;
        self.trade_count = 0;
        self.current_sharpe = 0.0;
        self.win_rate = 0.5;
    }
}

/// Market regime for regime-aware blending
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    TrendingBull,
    TrendingBear,
    MeanReverting,
    HighVolatility,
    LowLiquidity,
}

/// Strategy Blender main struct
pub struct StrategyBlender {
    /// Performance trackers per agent
    trackers: HashMap<u8, PerformanceTracker>,
    /// Agent weights (derived from Sharpe)
    weights: HashMap<u8, f32>,
    /// Current market regime
    regime: MarketRegime,
    /// Risk override threshold
    risk_override_threshold: f32,
    /// Position sizing multiplier
    position_multiplier: f32,
}

impl StrategyBlender {
    #[inline]
    pub fn new(risk_threshold: f32) -> Self {
        let mut trackers = HashMap::with_capacity(MAX_AGENTS);
        let mut weights = HashMap::with_capacity(MAX_AGENTS);
        
        for i in 0..MAX_AGENTS as u8 {
            trackers.insert(i, PerformanceTracker::new(50));
            weights.insert(i, 1.0);
        }
        
        Self {
            trackers,
            weights,
            regime: MarketRegime::MeanReverting,
            risk_override_threshold: risk_threshold,
            position_multiplier: 1.0,
        }
    }
    
    /// Update agent performance with trade outcome
    #[inline]
    pub fn update_performance(&mut self, agent_id: u8, return_pct: f32) {
        if let Some(tracker) = self.trackers.get_mut(&agent_id) {
            tracker.add_return(return_pct);
            self.recalculate_weights();
        }
    }
    
    /// Recalculate all agent weights based on Sharpe ratios
    fn recalculate_weights(&mut self) {
        let mut total_sharpe: f32 = 0.0;
        let mut sharpe_values: HashMap<u8, f32> = HashMap::new();
        
        // Collect positive Sharpe values
        for (&agent_id, tracker) in &self.trackers {
            let sharpe = tracker.sharpe().max(0.0); // Only positive Sharpe gets weight
            if sharpe > 0.0 {
                sharpe_values.insert(agent_id, sharpe);
                total_sharpe += sharpe;
            }
        }
        
        // Normalize weights
        for (&agent_id, weight) in &mut self.weights {
            if let Some(&sharpe) = sharpe_values.get(&agent_id) {
                *weight = sharpe / total_sharpe.max(1e-6);
            } else {
                *weight = 0.0;
            }
        }
    }
    
    /// Blend multiple agent signals into unified decision
    pub fn blend_signals(
        &self,
        signals: &[AgentSignal],
        consensus_approved: bool,
        catastrophic_risk: bool,
    ) -> BlendedDecision {
        if signals.is_empty() {
            return BlendedDecision::default();
        }
        
        // Check for catastrophic risk override
        if catastrophic_risk {
            return BlendedDecision {
                action: Action::Hold,
                confidence: 1.0,
                position_size: 0.0,
                contributing_agents: 0,
                override_active: true,
            };
        }
        
        // Check consensus veto
        if !consensus_approved {
            return BlendedDecision {
                action: Action::Hold,
                confidence: 0.8,
                position_size: 0.0,
                contributing_agents: 0,
                override_active: true,
            };
        }
        
        // Weighted vote aggregation
        let mut buy_weight: f32 = 0.0;
        let mut sell_weight: f32 = 0.0;
        let mut total_confidence: f32 = 0.0;
        let mut weighted_return: f32 = 0.0;
        let mut weighted_risk: f32 = 0.0;
        let mut contributing: u8 = 0;
        
        for signal in signals {
            let weight = self.weights.get(&signal.agent_id).copied().unwrap_or(0.1);
            
            if weight > 0.01 {
                contributing += 1;
                
                let signal_strength = signal.confidence * weight;
                total_confidence += signal_strength;
                
                match signal.action {
                    Action::Buy => buy_weight += signal_strength,
                    Action::Sell => sell_weight += signal_strength,
                    Action::Hedge => {
                        // Hedge counts as partial sell for long bias
                        sell_weight += signal_strength * 0.5;
                    },
                    _ => {},
                }
                
                weighted_return += signal.expected_return * weight;
                weighted_risk += signal.estimated_risk * weight;
            }
        }
        
        // Determine dominant action
        let (action, net_confidence) = if buy_weight > sell_weight * 1.2 {
            (Action::Buy, buy_weight - sell_weight)
        } else if sell_weight > buy_weight * 1.2 {
            (Action::Sell, sell_weight - buy_weight)
        } else {
            (Action::Hold, 0.0)
        };
        
        // Calculate position size based on risk-adjusted return
        let base_size = if weighted_risk > 0.0 {
            (weighted_return / weighted_risk).clamp(0.0, 2.0)
        } else {
            0.5
        };
        
        let position_size = base_size * self.position_multiplier * total_confidence;
        
        BlendedDecision {
            action,
            confidence: net_confidence.clamp(0.0, 1.0),
            position_size,
            contributing_agents: contributing,
            override_active: false,
        }
    }
    
    /// Set current market regime for regime-aware blending
    #[inline]
    pub fn set_regime(&mut self, regime: MarketRegime) {
        self.regime = regime;
        self.adjust_weights_for_regime();
    }
    
    /// Adjust weights based on regime compatibility
    fn adjust_weights_for_regime(&mut self) {
        // In production, this would use historical regime-specific performance
        // For now, apply simple adjustments
        
        let adjustment_factor = match self.regime {
            MarketRegime::TrendingBull => 1.1,  // Favor momentum agents
            MarketRegime::TrendingBear => 1.1,
            MarketRegime::MeanReverting => 0.9, // Reduce aggressive weights
            MarketRegime::HighVolatility => 0.8, // Reduce all weights
            MarketRegime::LowLiquidity => 0.7,
        };
        
        for weight in self.weights.values_mut() {
            *weight *= adjustment_factor;
        }
        
        // Renormalize
        let total: f32 = self.weights.values().sum();
        if total > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= total;
            }
        }
    }
    
    /// Get agent's current Sharpe ratio
    #[inline]
    pub fn get_agent_sharpe(&self, agent_id: u8) -> f32 {
        self.trackers.get(&agent_id).map(|t| t.sharpe()).unwrap_or(0.0)
    }
    
    /// Get agent's win rate
    #[inline]
    pub fn get_agent_win_rate(&self, agent_id: u8) -> f32 {
        self.trackers.get(&agent_id).map(|t| t.win_rate()).unwrap_or(0.5)
    }
    
    /// Set position sizing multiplier
    #[inline]
    pub fn set_position_multiplier(&mut self, multiplier: f32) {
        self.position_multiplier = multiplier.clamp(0.1, 3.0);
    }
    
    /// Reset all performance tracking
    #[inline]
    pub fn reset(&mut self) {
        for tracker in self.trackers.values_mut() {
            tracker.reset();
        }
        self.recalculate_weights();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(50);
        
        // Add some positive returns
        for _ in 0..10 {
            tracker.add_return(0.02);
        }
        
        assert!(tracker.sharpe() > 0.0);
        assert!(tracker.win_rate() > 0.9);
    }

    #[test]
    fn test_strategy_blender_basic() {
        let mut blender = StrategyBlender::new(0.5);
        
        // Simulate some performance
        blender.update_performance(0, 0.02);
        blender.update_performance(0, 0.015);
        blender.update_performance(1, -0.01);
        
        let signals = vec![
            AgentSignal {
                agent_id: 0,
                action: Action::Buy,
                confidence: 0.8,
                expected_return: 0.02,
                estimated_risk: 0.01,
            },
            AgentSignal {
                agent_id: 1,
                action: Action::Sell,
                confidence: 0.6,
                expected_return: -0.01,
                estimated_risk: 0.015,
            },
        ];
        
        let decision = blender.blend_signals(&signals, true, false);
        
        assert!(!decision.override_active);
        assert!(decision.contributing_agents >= 1);
    }

    #[test]
    fn test_catastrophic_override() {
        let blender = StrategyBlender::new(0.5);
        
        let signals = vec![
            AgentSignal {
                agent_id: 0,
                action: Action::Buy,
                confidence: 0.9,
                expected_return: 0.05,
                estimated_risk: 0.01,
            },
        ];
        
        let decision = blender.blend_signals(&signals, true, true);
        
        assert!(decision.override_active);
        assert_eq!(decision.action, Action::Hold);
        assert_eq!(decision.position_size, 0.0);
    }

    #[test]
    fn test_consensus_veto() {
        let blender = StrategyBlender::new(0.5);
        
        let signals = vec![
            AgentSignal {
                agent_id: 0,
                action: Action::Buy,
                confidence: 0.9,
                expected_return: 0.02,
                estimated_risk: 0.01,
            },
        ];
        
        let decision = blender.blend_signals(&signals, false, false);
        
        assert!(decision.override_active);
        assert_eq!(decision.action, Action::Hold);
    }

    #[test]
    fn test_weight_recalculation() {
        let mut blender = StrategyBlender::new(0.5);
        
        // Agent 0 performs well
        for _ in 0..20 {
            blender.update_performance(0, 0.02);
        }
        
        // Agent 1 performs poorly
        for _ in 0..20 {
            blender.update_performance(1, -0.01);
        }
        
        let sharpe_0 = blender.get_agent_sharpe(0);
        let sharpe_1 = blender.get_agent_sharpe(1);
        
        assert!(sharpe_0 > sharpe_1);
    }
}
