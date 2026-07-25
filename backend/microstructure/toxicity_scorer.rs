// ZAID PERSONAL CRYPTO TRADING BOT - Stage 21
// File: backend/microstructure/toxicity_scorer.rs
// Chapter 3: Order Book Imbalance, VPIN, and Toxicity Metrics
//
// Purpose: Grade order flow toxicity to prevent adverse selection
// Constraints: Real-time scoring, memory-efficient
// Target: AMD Ryzen AI 5 laptop with 8GB RAM limit

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Maximum history size for toxicity calculation
const MAX_TOXICITY_HISTORY: usize = 1000;

/// Toxicity score ranges from 0 (safe) to 1 (extremely toxic)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToxicityScore(pub f64);

impl ToxicityScore {
    pub fn safe() -> Self {
        ToxicityScore(0.0)
    }
    
    pub fn extreme() -> Self {
        ToxicityScore(1.0)
    }
    
    pub fn value(&self) -> f64 {
        self.0.clamp(0.0, 1.0)
    }
    
    pub fn level(&self) -> ToxicityLevel {
        match self.0 {
            x if x >= 0.8 => ToxicityLevel::Extreme,
            x if x >= 0.6 => ToxicityLevel::High,
            x if x >= 0.4 => ToxicityLevel::Moderate,
            x if x >= 0.2 => ToxicityLevel::Low,
            _ => ToxicityLevel::Minimal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToxicityLevel {
    Minimal,   // 0.0 - 0.2: Very safe to trade
    Low,       // 0.2 - 0.4: Normal conditions
    Moderate,  // 0.4 - 0.6: Exercise caution
    High,      // 0.6 - 0.8: Dangerous, widen spreads
    Extreme,   // 0.8 - 1.0: Highly toxic, consider stopping
}

/// Factors contributing to toxicity
#[derive(Debug, Clone)]
pub struct ToxicityFactors {
    /// VPIN-based informed trading probability
    pub vpin_component: f64,
    /// Order book imbalance contribution
    pub imbalance_component: f64,
    /// Spread widening indicator
    pub spread_component: f64,
    /// Volatility contribution
    pub volatility_component: f64,
    /// Cancellation rate (spoofing indicator)
    pub cancellation_component: f64,
}

/// Alert when toxicity exceeds thresholds
#[derive(Debug, Clone)]
pub struct ToxicityAlert {
    pub alert_id: u64,
    pub timestamp_us: u64,
    pub score: ToxicityScore,
    pub level: ToxicityLevel,
    pub primary_factor: ToxicityFactorType,
    pub recommended_action: ToxicityAction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToxicityFactorType {
    VPIN,
    Imbalance,
    Spread,
    Volatility,
    Cancellation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToxicityAction {
    ContinueNormal,
    WidenSpreads,
    ReduceSize,
    PauseTrading,
    EmergencyStop,
}

/// Main toxicity scorer combining multiple signals
pub struct ToxicityScorer {
    /// Sequence number for alerts
    sequence_number: AtomicU64,
    
    /// Start time for timestamps
    start_time: Instant,
    
    /// Rolling toxicity scores
    score_history: VecDeque<ToxicityScore>,
    
    /// Component scores
    current_factors: ToxicityFactors,
    
    /// Configuration weights
    vpin_weight: f64,
    imbalance_weight: f64,
    spread_weight: f64,
    volatility_weight: f64,
    cancellation_weight: f64,
    
    /// Thresholds
    warning_threshold: f64,
    critical_threshold: f64,
    
    /// State
    enabled: AtomicBool,
    recent_alerts: VecDeque<ToxicityAlert>,
}

impl ToxicityScorer {
    pub fn new() -> Self {
        Self {
            sequence_number: AtomicU64::new(0),
            start_time: Instant::now(),
            score_history: VecDeque::with_capacity(MAX_TOXICITY_HISTORY),
            current_factors: ToxicityFactors {
                vpin_component: 0.0,
                imbalance_component: 0.0,
                spread_component: 0.0,
                volatility_component: 0.0,
                cancellation_component: 0.0,
            },
            vpin_weight: 0.3,
            imbalance_weight: 0.2,
            spread_weight: 0.2,
            volatility_weight: 0.15,
            cancellation_weight: 0.15,
            warning_threshold: 0.5,
            critical_threshold: 0.75,
            enabled: AtomicBool::new(true),
            recent_alerts: VecDeque::with_capacity(100),
        }
    }
    
    fn now_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }
    
    /// Update VPIN component
    pub fn update_vpin(&mut self, vpin_value: f64) {
        self.current_factors.vpin_component = vpin_value.clamp(0.0, 1.0);
        self.recalculate_score();
    }
    
    /// Update order book imbalance component
    pub fn update_imbalance(&mut self, normalized_imbalance: f64) {
        // Use absolute value - both extreme buying and selling can be toxic
        self.current_factors.imbalance_component = normalized_imbalance.abs().clamp(0.0, 1.0);
        self.recalculate_score();
    }
    
    /// Update spread component (wide spreads indicate toxicity)
    pub fn update_spread(&mut self, spread_bps: f64, normal_spread_bps: f64) {
        let ratio = if normal_spread_bps > 0.0 {
            spread_bps / normal_spread_bps
        } else {
            1.0
        };
        self.current_factors.spread_component = (ratio - 1.0).clamp(0.0, 1.0);
        self.recalculate_score();
    }
    
    /// Update volatility component
    pub fn update_volatility(&mut self, realized_vol: f64, normal_vol: f64) {
        let ratio = if normal_vol > 0.0 {
            realized_vol / normal_vol
        } else {
            1.0
        };
        self.current_factors.volatility_component = (ratio - 1.0).clamp(0.0, 1.0);
        self.recalculate_score();
    }
    
    /// Update cancellation rate component (high cancellation = potential spoofing)
    pub fn update_cancellation_rate(&mut self, cancel_rate: f64) {
        // cancel_rate is 0.0 to 1.0
        self.current_factors.cancellation_component = cancel_rate.clamp(0.0, 1.0);
        self.recalculate_score();
    }
    
    /// Recalculate overall toxicity score from components
    fn recalculate_score(&mut self) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        
        let weighted_sum = 
            self.vpin_weight * self.current_factors.vpin_component +
            self.imbalance_weight * self.current_factors.imbalance_component +
            self.spread_weight * self.current_factors.spread_component +
            self.volatility_weight * self.current_factors.volatility_component +
            self.cancellation_weight * self.current_factors.cancellation_component;
        
        let score = ToxicityScore(weighted_sum.clamp(0.0, 1.0));
        
        // Store in history
        if self.score_history.len() >= MAX_TOXICITY_HISTORY {
            self.score_history.pop_front();
        }
        self.score_history.push_back(score);
        
        // Check thresholds
        self.check_thresholds(score);
    }
    
    /// Check if score exceeds thresholds and generate alerts
    fn check_thresholds(&mut self, score: ToxicityScore) {
        let level = score.level();
        
        if score.value() >= self.critical_threshold {
            let alert = self.generate_alert(score, level, ToxicityAction::PauseTrading);
            self.recent_alerts.push_back(alert);
        } else if score.value() >= self.warning_threshold {
            let alert = self.generate_alert(score, level, ToxicityAction::WidenSpreads);
            self.recent_alerts.push_back(alert);
        }
    }
    
    fn generate_alert(&mut self, score: ToxicityScore, level: ToxicityLevel, 
                      action: ToxicityAction) -> ToxicityAlert {
        // Determine primary factor
        let factors = &self.current_factors;
        let primary_factor = [
            (ToxicityFactorType::VPIN, factors.vpin_component),
            (ToxicityFactorType::Imbalance, factors.imbalance_component),
            (ToxicityFactorType::Spread, factors.spread_component),
            (ToxicityFactorType::Volatility, factors.volatility_component),
            (ToxicityFactorType::Cancellation, factors.cancellation_component),
        ].iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(f, _)| *f)
            .unwrap_or(ToxicityFactorType::VPIN);
        
        ToxicityAlert {
            alert_id: self.sequence_number.fetch_add(1, Ordering::SeqCst),
            timestamp_us: self.now_us(),
            score,
            level,
            primary_factor,
            recommended_action: action,
        }
    }
    
    /// Get current toxicity score
    pub fn get_current_score(&self) -> ToxicityScore {
        self.score_history.back().copied().unwrap_or(ToxicityScore::safe())
    }
    
    /// Get smoothed toxicity score (average of recent scores)
    pub fn get_smoothed_score(&self, window: usize) -> ToxicityScore {
        if self.score_history.is_empty() {
            return ToxicityScore::safe();
        }
        
        let take = window.min(self.score_history.len());
        let recent: Vec<_> = self.score_history.iter().rev().take(take).collect();
        
        let avg = recent.iter().map(|s| s.value()).sum::<f64>() / take as f64;
        ToxicityScore(avg)
    }
    
    /// Get recommended action based on current toxicity
    pub fn get_recommended_action(&self) -> ToxicityAction {
        let score = self.get_current_score();
        match score.level() {
            ToxicityLevel::Minimal | ToxicityLevel::Low => ToxicityAction::ContinueNormal,
            ToxicityLevel::Moderate => ToxicityAction::WidenSpreads,
            ToxicityLevel::High => ToxicityAction::ReduceSize,
            ToxicityLevel::Extreme => ToxicityAction::PauseTrading,
        }
    }
    
    /// Calculate toxicity-adjusted position size multiplier
    pub fn get_size_multiplier(&self) -> f64 {
        let score = self.get_current_score();
        // Reduce size as toxicity increases
        match score.level() {
            ToxicityLevel::Minimal => 1.0,
            ToxicityLevel::Low => 0.9,
            ToxicityLevel::Moderate => 0.6,
            ToxicityLevel::High => 0.3,
            ToxicityLevel::Extreme => 0.0,
        }
    }
    
    /// Calculate toxicity-adjusted spread multiplier
    pub fn get_spread_multiplier(&self) -> f64 {
        let score = self.get_current_score();
        // Widen spreads as toxicity increases
        1.0 + score.value() * 2.0  // Up to 3x normal spread
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> ToxicityStatistics {
        if self.score_history.is_empty() {
            return ToxicityStatistics::default();
        }
        
        let scores: Vec<f64> = self.score_history.iter().map(|s| s.value()).collect();
        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        let max = scores.iter().cloned_by(f64::max).fold(scores[0], f64::max);
        
        ToxicityStatistics {
            current_score: self.get_current_score().value(),
            average_score: avg,
            max_score: max,
            alerts_generated: self.recent_alerts.len(),
            level: self.get_current_score().level(),
        }
    }
    
    /// Set custom weights for toxicity components
    pub fn set_weights(&mut self, vpin: f64, imbalance: f64, spread: f64, 
                       volatility: f64, cancellation: f64) {
        let total = vpin + imbalance + spread + volatility + cancellation;
        if total > 0.0 {
            self.vpin_weight = vpin / total;
            self.imbalance_weight = imbalance / total;
            self.spread_weight = spread / total;
            self.volatility_weight = volatility / total;
            self.cancellation_weight = cancellation / total;
        }
    }
}

impl Default for ToxicityScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToxicityStatistics {
    pub current_score: f64,
    pub average_score: f64,
    pub max_score: f64,
    pub alerts_generated: usize,
    pub level: ToxicityLevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toxicity_score_calculation() {
        let mut scorer = ToxicityScorer::new();
        
        // Set high VPIN
        scorer.update_vpin(0.8);
        
        let score = scorer.get_current_score();
        assert!(score.value() > 0.2); // Should have some toxicity from VPIN
        
        // Add more toxic signals
        scorer.update_imbalance(0.9);
        scorer.update_cancellation_rate(0.7);
        
        let score = scorer.get_current_score();
        assert!(score.value() > 0.5); // Should be moderate+ toxicity
    }

    #[test]
    fn test_size_multiplier() {
        let mut scorer = ToxicityScorer::new();
        
        // Low toxicity = full size
        assert_eq!(scorer.get_size_multiplier(), 1.0);
        
        // High toxicity = reduced size
        scorer.update_vpin(0.9);
        scorer.update_imbalance(0.9);
        
        let multiplier = scorer.get_size_multiplier();
        assert!(multiplier < 0.5);
    }
}
