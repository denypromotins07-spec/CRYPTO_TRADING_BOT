//! Theta Decay Monitor - Time Decay Tracking for Gamma Scalping
//! 
//! This module tracks time decay (theta) to ensure gamma profits exceed theta bleed.
//! It monitors the profitability of long options positions and triggers alerts when
//! theta decay threatens to erode gains.
//! 
//! Key Features:
//! - Real-time theta calculation and monitoring
//! - Gamma profit vs theta cost comparison
//! - Breakeven analysis for straddle positions
//! - Stack-allocated calculations for performance
//! - Automatic position adjustment recommendations
//! 
//! Target: Ensure gamma scalping profits exceed theta decay costs

use std::fmt::{Debug, Display};
use std::time::Instant;

/// Asset types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Asset {
    BTC,
    ETH,
    SOL,
}

impl Display for Asset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Asset::BTC => write!(f, "BTC"),
            Asset::ETH => write!(f, "ETH"),
            Asset::SOL => write!(f, "SOL"),
        }
    }
}

/// Option type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

/// Theta decay metrics for a position
#[derive(Debug, Clone)]
pub struct ThetaMetrics {
    pub daily_theta: f64,      // Dollar decay per day
    pub hourly_theta: f64,     // Dollar decay per hour
    pub total_position_theta: f64,
    pub days_to_expiry: f64,
    pub theta_acceleration: f64, // Rate of theta increase
}

impl ThetaMetrics {
    pub fn new(
        daily_theta: f64,
        days_to_expiry: f64,
    ) -> Self {
        let hourly_theta = daily_theta / 24.0;
        
        // Theta accelerates as expiry approaches (simplified model)
        let theta_acceleration = if days_to_expiry > 0.0 {
            daily_theta / (days_to_expiry * days_to_expiry)
        } else {
            0.0
        };
        
        Self {
            daily_theta,
            hourly_theta,
            total_position_theta: daily_theta,
            days_to_expiry,
            theta_acceleration,
        }
    }
    
    /// Project theta decay over n days
    pub fn projected_decay(&self, days: f64) -> f64 {
        // Simplified: linear projection
        // In reality, theta accelerates near expiry
        self.daily_theta * days
    }
    
    /// Check if position is in theta danger zone (< 7 days to expiry)
    pub fn is_danger_zone(&self) -> bool {
        self.days_to_expiry < 7.0
    }
}

/// Gamma profit tracking
#[derive(Debug, Clone)]
pub struct GammaProfit {
    pub realized_profit: f64,
    pub unrealized_profit: f64,
    pub total_gamma_profit: f64,
    pub scalp_count: u64,
    pub avg_profit_per_scalp: f64,
}

impl GammaProfit {
    pub fn new() -> Self {
        Self {
            realized_profit: 0.0,
            unrealized_profit: 0.0,
            total_gamma_profit: 0.0,
            scalp_count: 0,
            avg_profit_per_scalp: 0.0,
        }
    }
    
    /// Add realized scalp profit
    pub fn add_scalp(&mut self, profit: f64) {
        self.realized_profit += profit;
        self.total_gamma_profit += profit;
        self.scalp_count += 1;
        self.avg_profit_per_scalp = self.total_gamma_profit / self.scalp_count as f64;
    }
    
    /// Update unrealized profit
    pub fn update_unrealized(&mut self, profit: f64) {
        self.unrealized_profit = profit;
        self.total_gamma_profit = self.realized_profit + profit;
    }
}

impl Default for GammaProfit {
    fn default() -> Self {
        Self::new()
    }
}

/// Position profitability status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfitabilityStatus {
    Profitable,      // Gamma profits > Theta costs
    Breakeven,       // Gamma profits ≈ Theta costs
    Warning,         // Theta costs approaching gamma profits
    Critical,        // Theta costs > Gamma profits
    Expiring,        // Near expiry, urgent action needed
}

impl Display for ProfitabilityStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfitabilityStatus::Profitable => write!(f, "Profitable"),
            ProfitabilityStatus::Breakeven => write!(f, "Breakeven"),
            ProfitabilityStatus::Warning => write!(f, "Warning"),
            ProfitabilityStatus::Critical => write!(f, "Critical"),
            ProfitabilityStatus::Expiring => write!(f, "Expiring"),
        }
    }
}

/// Alert for theta/gamma imbalance
#[derive(Debug)]
pub struct ThetaAlert {
    pub alert_type: AlertType,
    pub severity: Severity,
    pub current_theta: f64,
    pub current_gamma_profit: f64,
    pub net_pnl: f64,
    pub recommended_action: RecommendedAction,
    pub message: String,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertType {
    ThetaAcceleration,
    GammaThetaImbalance,
    ExpiryApproaching,
    BreakevenBreach,
    ProfitTargetReached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Low => write!(f, "Low"),
            Severity::Medium => write!(f, "Medium"),
            Severity::High => write!(f, "High"),
            Severity::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    Hold,
    ClosePartial,
    CloseFull,
    RollForward,
    IncreaseScalping,
    HedgeTheta,
}

impl Display for RecommendedAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecommendedAction::Hold => write!(f, "Hold"),
            RecommendedAction::ClosePartial => write!(f, "ClosePartial"),
            RecommendedAction::CloseFull => write!(f, "CloseFull"),
            RecommendedAction::RollForward => write!(f, "RollForward"),
            RecommendedAction::IncreaseScalping => write!(f, "IncreaseScalping"),
            RecommendedAction::HedgeTheta => write!(f, "HedgeTheta"),
        }
    }
}

/// Options position being monitored
#[derive(Debug)]
pub struct MonitoredPosition {
    pub asset: Asset,
    pub option_type: OptionType,
    pub strike: f64,
    pub quantity: f64,
    pub entry_premium: f64,
    pub current_premium: f64,
    pub days_to_expiry: f64,
    pub theta_metrics: ThetaMetrics,
    pub gamma_profit: GammaProfit,
}

impl MonitoredPosition {
    pub fn calculate_net_pnl(&self) -> f64 {
        let premium_pnl = (self.current_premium - self.entry_premium) * self.quantity;
        premium_pnl + self.gamma_profit.total_gamma_profit
    }
    
    pub fn get_status(&self) -> ProfitabilityStatus {
        let net_pnl = self.calculate_net_pnl();
        let total_theta_cost = self.theta_metrics.daily_theta * 
            (self.theta_metrics.days_to_expiry.min(30.0)); // Cap at 30 days
        
        let gamma_profit = self.gamma_profit.total_gamma_profit.max(0.0);
        
        if self.days_to_expiry < 3.0 {
            ProfitabilityStatus::Expiring
        } else if net_pnl > total_theta_cost * 1.5 {
            ProfitabilityStatus::Profitable
        } else if net_pnl > total_theta_cost * 0.9 {
            ProfitabilityStatus::Breakeven
        } else if gamma_profit > total_theta_cost * 0.5 {
            ProfitabilityStatus::Warning
        } else {
            ProfitabilityStatus::Critical
        }
    }
}

/// Main theta decay monitor
pub struct ThetaDecayMonitor {
    /// Positions being monitored
    positions: Vec<MonitoredPosition>,
    
    /// Alert history
    alerts: Vec<ThetaAlert>,
    
    /// Configuration
    warning_threshold: f64,    // Ratio of theta/gamma for warning
    critical_threshold: f64,   // Ratio for critical alert
    
    /// Cumulative metrics
    total_theta_cost: f64,
    total_gamma_profit: f64,
}

impl ThetaDecayMonitor {
    /// Create new monitor with default thresholds
    pub fn new(warning_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            positions: Vec::new(),
            alerts: Vec::new(),
            warning_threshold,
            critical_threshold,
            total_theta_cost: 0.0,
            total_gamma_profit: 0.0,
        }
    }
    
    /// Add position to monitor
    pub fn add_position(&mut self, position: MonitoredPosition) {
        self.positions.push(position);
    }
    
    /// Remove position by index
    pub fn remove_position(&mut self, index: usize) -> Option<MonitoredPosition> {
        if index < self.positions.len() {
            Some(self.positions.remove(index))
        } else {
            None
        }
    }
    
    /// Update position market data
    pub fn update_position(
        &mut self,
        index: usize,
        current_premium: f64,
        days_to_expiry: f64,
        daily_theta: f64,
    ) {
        if let Some(position) = self.positions.get_mut(index) {
            position.current_premium = current_premium;
            position.days_to_expiry = days_to_expiry;
            position.theta_metrics = ThetaMetrics::new(daily_theta, days_to_expiry);
        }
    }
    
    /// Record gamma scalp profit
    pub fn record_scalp(&mut self, index: usize, profit: f64) {
        if let Some(position) = self.positions.get_mut(index) {
            position.gamma_profit.add_scalp(profit);
            self.total_gamma_profit += profit;
        }
    }
    
    /// Check all positions and generate alerts
    pub fn check_positions(&mut self) -> Vec<ThetaAlert> {
        let mut new_alerts = Vec::new();
        let now = Instant::now();
        
        for (idx, position) in self.positions.iter().enumerate() {
            let status = position.get_status();
            let net_pnl = position.calculate_net_pnl();
            
            // Calculate theta/gamma ratio
            let theta_cost = position.theta_metrics.daily_theta;
            let gamma_profit = position.gamma_profit.total_gamma_profit;
            let ratio = if gamma_profit > 0.0 {
                theta_cost / gamma_profit
            } else {
                f64::INFINITY
            };
            
            // Generate alerts based on status
            let alert = match status {
                ProfitabilityStatus::Expiring => Some(ThetaAlert {
                    alert_type: AlertType::ExpiryApproaching,
                    severity: Severity::Critical,
                    current_theta: theta_cost,
                    current_gamma_profit: gamma_profit,
                    net_pnl,
                    recommended_action: RecommendedAction::CloseFull,
                    message: format!(
                        "Position {} expires in {:.1} days - close immediately",
                        idx, position.days_to_expiry
                    ),
                    timestamp: now,
                }),
                ProfitabilityStatus::Critical if ratio > self.critical_threshold => {
                    Some(ThetaAlert {
                        alert_type: AlertType::GammaThetaImbalance,
                        severity: Severity::Critical,
                        current_theta: theta_cost,
                        current_gamma_profit: gamma_profit,
                        net_pnl,
                        recommended_action: RecommendedAction::RollForward,
                        message: format!(
                            "Theta costs exceed gamma profits by {:.1}x - roll or close",
                            ratio
                        ),
                        timestamp: now,
                    })
                }
                ProfitabilityStatus::Warning if ratio > self.warning_threshold => {
                    Some(ThetaAlert {
                        alert_type: AlertType::ThetaAcceleration,
                        severity: Severity::High,
                        current_theta: theta_cost,
                        current_gamma_profit: gamma_profit,
                        net_pnl,
                        recommended_action: RecommendedAction::IncreaseScalping,
                        message: format!(
                            "Theta acceleration detected - ratio {:.1}x",
                            ratio
                        ),
                        timestamp: now,
                    })
                }
                _ => None,
            };
            
            if let Some(a) = alert {
                new_alerts.push(a.clone());
                self.alerts.push(a);
            }
        }
        
        // Update cumulative theta
        self.total_theta_cost = self.positions.iter()
            .map(|p| p.theta_metrics.daily_theta)
            .sum();
        
        new_alerts
    }
    
    /// Get overall portfolio status
    pub fn portfolio_status(&self) -> ProfitabilityStatus {
        if self.positions.is_empty() {
            return ProfitabilityStatus::Breakeven;
        }
        
        let worst_status = self.positions.iter()
            .map(|p| p.get_status())
            .min_by_key(|s| match s {
                ProfitabilityStatus::Expiring => 0,
                ProfitabilityStatus::Critical => 1,
                ProfitabilityStatus::Warning => 2,
                ProfitabilityStatus::Breakeven => 3,
                ProfitabilityStatus::Profitable => 4,
            })
            .unwrap_or(ProfitabilityStatus::Breakeven);
        
        worst_status
    }
    
    /// Get net P&L across all positions
    pub fn total_net_pnl(&self) -> f64 {
        self.positions.iter().map(|p| p.calculate_net_pnl()).sum()
    }
    
    /// Get recent alerts
    pub fn get_recent_alerts(&self, limit: usize) -> &[ThetaAlert] {
        let start = self.alerts.len().saturating_sub(limit);
        &self.alerts[start..]
    }
    
    /// Prune old alerts
    pub fn prune_alerts(&mut self, keep_last: usize) {
        if self.alerts.len() > keep_last {
            self.alerts.drain(..self.alerts.len() - keep_last);
        }
    }
}

impl Default for ThetaDecayMonitor {
    fn default() -> Self {
        Self::new(0.5, 1.0) // Warn at 50% ratio, critical at 100%
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_theta_metrics() {
        let metrics = ThetaMetrics::new(100.0, 30.0);
        
        assert!((metrics.daily_theta - 100.0).abs() < 0.01);
        assert!((metrics.hourly_theta - 4.166).abs() < 0.1);
        assert!(!metrics.is_danger_zone());
        
        let near_expiry = ThetaMetrics::new(500.0, 5.0);
        assert!(near_expiry.is_danger_zone());
    }
    
    #[test]
    fn test_gamma_profit_tracking() {
        let mut profit = GammaProfit::new();
        
        profit.add_scalp(100.0);
        profit.add_scalp(150.0);
        
        assert!((profit.realized_profit - 250.0).abs() < 0.01);
        assert_eq!(profit.scalp_count, 2);
        assert!((profit.avg_profit_per_scalp - 125.0).abs() < 0.01);
    }
    
    #[test]
    fn test_position_status() {
        let position = MonitoredPosition {
            asset: Asset::BTC,
            option_type: OptionType::Call,
            strike: 50000.0,
            quantity: 1.0,
            entry_premium: 2000.0,
            current_premium: 2500.0,
            days_to_expiry: 30.0,
            theta_metrics: ThetaMetrics::new(50.0, 30.0),
            gamma_profit: {
                let mut g = GammaProfit::new();
                g.add_scalp(200.0);
                g
            },
        };
        
        let status = position.get_status();
        assert_eq!(status, ProfitabilityStatus::Profitable);
    }
    
    #[test]
    fn test_theta_monitor_alerts() {
        let mut monitor = ThetaDecayMonitor::new(0.5, 1.0);
        
        // Add a position in critical state
        let position = MonitoredPosition {
            asset: Asset::BTC,
            option_type: OptionType::Call,
            strike: 50000.0,
            quantity: 1.0,
            entry_premium: 2000.0,
            current_premium: 1500.0, // Loss on premium
            days_to_expiry: 2.0,     // Near expiry
            theta_metrics: ThetaMetrics::new(500.0, 2.0),
            gamma_profit: GammaProfit::new(), // No gamma profits
        };
        
        monitor.add_position(position);
        let alerts = monitor.check_positions();
        
        assert!(!alerts.is_empty());
        assert_eq!(monitor.portfolio_status(), ProfitabilityStatus::Expiring);
    }
}
