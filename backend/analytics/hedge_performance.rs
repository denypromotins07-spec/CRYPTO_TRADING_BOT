//! Hedge Performance Analytics - Sharpe Ratio of Isolated Hedging Strategy
//! 
//! This module measures the Sharpe ratio and other performance metrics
//! of the isolated hedging strategy. It tracks risk-adjusted returns
//! to ensure hedging adds value rather than just costs.
//! 
//! Key Features:
//! - Real-time Sharpe ratio calculation
//! - Drawdown tracking and analysis
//! - Win rate and profit factor metrics
//! - Rolling performance windows
//! - Zero-cost abstractions for memory efficiency
//! 
//! Target: Measure hedging strategy alpha generation capability

use std::collections::VecDeque;
use std::fmt::{Debug, Display};
use std::time::Instant;

/// Hedge execution record for performance tracking
#[derive(Debug, Clone)]
pub struct HedgeExecution {
    pub hedge_id: String,
    pub timestamp: Instant,
    pub asset: Asset,
    pub side: PositionSide,
    pub quantity: f64,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub pnl: f64,
    pub cost: f64,
    pub net_pnl: f64,
}

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

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

impl Display for PositionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionSide::Long => write!(f, "Long"),
            PositionSide::Short => write!(f, "Short"),
        }
    }
}

/// Performance metrics snapshot
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_hedges: u64,
    pub total_pnl: f64,
    pub total_cost: f64,
    pub net_pnl: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub consecutive_wins: u64,
    pub consecutive_losses: u64,
    pub avg_holding_time_ms: f64,
    pub calmar_ratio: f64,
}

/// Rolling window statistics calculator
pub struct RollingStats {
    window_size: usize,
    values: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl RollingStats {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::with_capacity(window_size),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }
    
    pub fn add(&mut self, value: f64) {
        if self.values.len() >= self.window_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        
        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }
    
    pub fn mean(&self) -> f64 {
        let n = self.values.len() as f64;
        if n == 0.0 { return 0.0; }
        self.sum / n
    }
    
    pub fn std(&self) -> f64 {
        let n = self.values.len() as f64;
        if n < 2.0 { return 0.0; }
        
        let mean = self.mean();
        let variance = (self.sum_sq / n) - (mean * mean);
        
        if variance > 0.0 {
            variance.sqrt()
        } else {
            0.0
        }
    }
    
    pub fn count(&self) -> usize {
        self.values.len()
    }
}

/// Main hedge performance analyzer
pub struct HedgePerformanceAnalyzer {
    /// Execution history
    executions: Vec<HedgeExecution>,
    
    /// Rolling returns for Sharpe calculation
    rolling_returns: RollingStats,
    
    /// Rolling downside returns for Sortino
    rolling_downside: RollingStats,
    
    /// Cumulative P&L tracking
    cumulative_pnl: f64,
    peak_pnl: f64,
    max_drawdown: f64,
    
    /// Win/loss tracking
    wins: u64,
    losses: u64,
    total_won: f64,
    total_lost: f64,
    largest_win: f64,
    largest_loss: f64,
    current_streak: i64,  // Positive = wins, Negative = losses
    best_streak: u64,
    worst_streak: u64,
    
    /// Configuration
    risk_free_rate: f64,  // Annualized
    periods_per_year: f64, // For annualization
    
    /// Metrics cache
    last_metrics: Option<PerformanceMetrics>,
}

impl HedgePerformanceAnalyzer {
    /// Create new analyzer with default settings
    pub fn new(risk_free_rate: f64, periods_per_year: f64, window_size: usize) -> Self {
        Self {
            executions: Vec::new(),
            rolling_returns: RollingStats::new(window_size),
            rolling_downside: RollingStats::new(window_size),
            cumulative_pnl: 0.0,
            peak_pnl: 0.0,
            max_drawdown: 0.0,
            wins: 0,
            losses: 0,
            total_won: 0.0,
            total_lost: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            current_streak: 0,
            best_streak: 0,
            worst_streak: 0,
            risk_free_rate,
            periods_per_year,
            last_metrics: None,
        }
    }
    
    /// Record a completed hedge execution
    pub fn record_execution(&mut self, execution: HedgeExecution) {
        self.executions.push(execution.clone());
        
        let net_pnl = execution.net_pnl;
        
        // Update cumulative P&L
        self.cumulative_pnl += net_pnl;
        
        // Update peak and drawdown
        if self.cumulative_pnl > self.peak_pnl {
            self.peak_pnl = self.cumulative_pnl;
        }
        
        let drawdown = self.peak_pnl - self.cumulative_pnl;
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }
        
        // Update win/loss stats
        if net_pnl > 0.0 {
            self.wins += 1;
            self.total_won += net_pnl;
            if net_pnl > self.largest_win {
                self.largest_win = net_pnl;
            }
            
            // Update streak
            if self.current_streak >= 0 {
                self.current_streak += 1;
            } else {
                self.current_streak = 1;
            }
            if self.current_streak as u64 > self.best_streak {
                self.best_streak = self.current_streak as u64;
            }
        } else if net_pnl < 0.0 {
            self.losses += 1;
            self.total_lost += net_pnl.abs();
            if net_pnl < self.largest_loss {
                self.largest_loss = net_pnl;
            }
            
            // Update streak
            if self.current_streak <= 0 {
                self.current_streak -= 1;
            } else {
                self.current_streak = -1;
            }
            if self.current_streak.abs() as u64 > self.worst_streak {
                self.worst_streak = self.current_streak.abs() as u64;
            }
        }
        
        // Calculate return for rolling stats (simplified)
        let return_pct = if execution.entry_price > 0.0 {
            net_pnl / (execution.quantity * execution.entry_price)
        } else {
            0.0
        };
        
        self.rolling_returns.add(return_pct);
        
        if return_pct < 0.0 {
            self.rolling_downside.add(return_pct);
        }
        
        // Invalidate cached metrics
        self.last_metrics = None;
    }
    
    /// Calculate current performance metrics
    pub fn calculate_metrics(&mut self) -> PerformanceMetrics {
        let total_hedges = self.executions.len() as u64;
        let total_cost: f64 = self.executions.iter().map(|e| e.cost).sum();
        let net_pnl = self.cumulative_pnl;
        
        // Win rate
        let win_rate = if total_hedges > 0 {
            self.wins as f64 / total_hedges as f64
        } else {
            0.0
        };
        
        // Profit factor
        let profit_factor = if self.total_lost > 0.0 {
            self.total_won / self.total_lost
        } else if self.total_won > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        
        // Sharpe ratio (annualized)
        let sharpe_ratio = self.calculate_sharpe();
        
        // Sortino ratio (annualized)
        let sortino_ratio = self.calculate_sortino();
        
        // Average win/loss
        let avg_win = if self.wins > 0 {
            self.total_won / self.wins as f64
        } else {
            0.0
        };
        
        let avg_loss = if self.losses > 0 {
            self.total_lost / self.losses as f64
        } else {
            0.0
        };
        
        // Calmar ratio (return / max drawdown)
        let calmar_ratio = if self.max_drawdown > 0.0 {
            net_pnl / self.max_drawdown
        } else {
            0.0
        };
        
        // Average holding time (simplified)
        let avg_holding_time_ms = if total_hedges > 0 {
            // Would need actual timestamps for accurate calculation
            0.0
        } else {
            0.0
        };
        
        let metrics = PerformanceMetrics {
            total_hedges,
            total_pnl: self.executions.iter().map(|e| e.pnl).sum(),
            total_cost,
            net_pnl,
            win_rate,
            profit_factor,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: self.max_drawdown,
            avg_win,
            avg_loss,
            largest_win: self.largest_win,
            largest_loss: self.largest_loss,
            consecutive_wins: self.best_streak,
            consecutive_losses: self.worst_streak,
            avg_holding_time_ms,
            calmar_ratio,
        };
        
        self.last_metrics = Some(metrics.clone());
        metrics
    }
    
    /// Calculate Sharpe ratio
    fn calculate_sharpe(&self) -> f64 {
        if self.rolling_returns.count() < 2 {
            return 0.0;
        }
        
        let mean_return = self.rolling_returns.mean();
        let std_return = self.rolling_returns.std();
        
        if std_return == 0.0 {
            return 0.0;
        }
        
        // Annualize
        let excess_return = mean_return - (self.risk_free_rate / self.periods_per_year);
        let sharpe = excess_return / std_return;
        
        sharpe * self.periods_per_year.sqrt()
    }
    
    /// Calculate Sortino ratio (uses downside deviation)
    fn calculate_sortino(&self) -> f64 {
        if self.rolling_downside.count() < 2 {
            return 0.0;
        }
        
        let mean_return = self.rolling_returns.mean();
        let downside_std = self.rolling_downside.std();
        
        if downside_std == 0.0 {
            return 0.0;
        }
        
        let excess_return = mean_return - (self.risk_free_rate / self.periods_per_year);
        let sortino = excess_return / downside_std;
        
        sortino * self.periods_per_year.sqrt()
    }
    
    /// Get recent executions
    pub fn get_recent_executions(&self, limit: usize) -> &[HedgeExecution] {
        let start = self.executions.len().saturating_sub(limit);
        &self.executions[start..]
    }
    
    /// Get performance by asset
    pub fn get_performance_by_asset(&self, asset: Asset) -> AssetPerformance {
        let asset_executions: Vec<_> = self.executions
            .iter()
            .filter(|e| e.asset == asset)
            .collect();
        
        let total_pnl: f64 = asset_executions.iter().map(|e| e.net_pnl).sum();
        let total_cost: f64 = asset_executions.iter().map(|e| e.cost).sum();
        let count = asset_executions.len();
        
        let wins = asset_executions.iter().filter(|e| e.net_pnl > 0.0).count();
        let losses = asset_executions.iter().filter(|e| e.net_pnl < 0.0).count();
        
        AssetPerformance {
            asset,
            count,
            total_pnl,
            total_cost,
            net_pnl: total_pnl,
            win_count: wins as u64,
            loss_count: losses as u64,
            win_rate: if count > 0 { wins as f64 / count as f64 } else { 0.0 },
        }
    }
    
    /// Reset all metrics
    pub fn reset(&mut self) {
        self.executions.clear();
        self.rolling_returns = RollingStats::new(self.rolling_returns.window_size);
        self.rolling_downside = RollingStats::new(self.rolling_downside.window_size);
        self.cumulative_pnl = 0.0;
        self.peak_pnl = 0.0;
        self.max_drawdown = 0.0;
        self.wins = 0;
        self.losses = 0;
        self.total_won = 0.0;
        self.total_lost = 0.0;
        self.largest_win = 0.0;
        self.largest_loss = 0.0;
        self.current_streak = 0;
        self.best_streak = 0;
        self.worst_streak = 0;
        self.last_metrics = None;
    }
}

/// Performance breakdown by asset
#[derive(Debug)]
pub struct AssetPerformance {
    pub asset: Asset,
    pub count: usize,
    pub total_pnl: f64,
    pub total_cost: f64,
    pub net_pnl: f64,
    pub win_count: u64,
    pub loss_count: u64,
    pub win_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_metrics() {
        let mut analyzer = HedgePerformanceAnalyzer::new(0.02, 252.0, 100);
        
        // Add some winning trades
        analyzer.record_execution(HedgeExecution {
            hedge_id: "H1".to_string(),
            timestamp: Instant::now(),
            asset: Asset::BTC,
            side: PositionSide::Short,
            quantity: 1.0,
            entry_price: 50000.0,
            exit_price: Some(49900.0),
            pnl: 100.0,
            cost: 5.0,
            net_pnl: 95.0,
        });
        
        analyzer.record_execution(HedgeExecution {
            hedge_id: "H2".to_string(),
            timestamp: Instant::now(),
            asset: Asset::BTC,
            side: PositionSide::Short,
            quantity: 1.0,
            entry_price: 50000.0,
            exit_price: Some(49950.0),
            pnl: 50.0,
            cost: 5.0,
            net_pnl: 45.0,
        });
        
        // Add a losing trade
        analyzer.record_execution(HedgeExecution {
            hedge_id: "H3".to_string(),
            timestamp: Instant::now(),
            asset: Asset::BTC,
            side: PositionSide::Short,
            quantity: 1.0,
            entry_price: 50000.0,
            exit_price: Some(50100.0),
            pnl: -100.0,
            cost: 5.0,
            net_pnl: -105.0,
        });
        
        let metrics = analyzer.calculate_metrics();
        
        assert_eq!(metrics.total_hedges, 3);
        assert_eq!(metrics.wins, 2);
        assert_eq!(metrics.losses, 1);
        assert!((metrics.win_rate - 0.667).abs() < 0.01);
        assert!(metrics.profit_factor > 1.0);
    }
    
    #[test]
    fn test_drawdown_tracking() {
        let mut analyzer = HedgePerformanceAnalyzer::new(0.0, 252.0, 100);
        
        // Add winning trades
        analyzer.record_execution(HedgeExecution {
            hedge_id: "H1".to_string(),
            timestamp: Instant::now(),
            asset: Asset::BTC,
            side: PositionSide::Short,
            quantity: 1.0,
            entry_price: 50000.0,
            exit_price: Some(49000.0),
            pnl: 1000.0,
            cost: 10.0,
            net_pnl: 990.0,
        });
        
        // Add losing trade that creates drawdown
        analyzer.record_execution(HedgeExecution {
            hedge_id: "H2".to_string(),
            timestamp: Instant::now(),
            asset: Asset::BTC,
            side: PositionSide::Short,
            quantity: 1.0,
            entry_price: 50000.0,
            exit_price: Some(51000.0),
            pnl: -1000.0,
            cost: 10.0,
            net_pnl: -1010.0,
        });
        
        let metrics = analyzer.calculate_metrics();
        
        assert!(metrics.max_drawdown > 0.0);
        assert!((metrics.max_drawdown - 20.0).abs() < 0.1); // ~20 from peak
    }
    
    #[test]
    fn test_streak_tracking() {
        let mut analyzer = HedgePerformanceAnalyzer::new(0.0, 252.0, 100);
        
        // Add 3 wins
        for i in 0..3 {
            analyzer.record_execution(HedgeExecution {
                hedge_id: format!("H{}", i),
                timestamp: Instant::now(),
                asset: Asset::BTC,
                side: PositionSide::Short,
                quantity: 1.0,
                entry_price: 50000.0,
                exit_price: Some(49900.0),
                pnl: 100.0,
                cost: 5.0,
                net_pnl: 95.0,
            });
        }
        
        // Add 2 losses
        for i in 3..5 {
            analyzer.record_execution(HedgeExecution {
                hedge_id: format!("H{}", i),
                timestamp: Instant::now(),
                asset: Asset::BTC,
                side: PositionSide::Short,
                quantity: 1.0,
                entry_price: 50000.0,
                exit_price: Some(50100.0),
                pnl: -100.0,
                cost: 5.0,
                net_pnl: -105.0,
            });
        }
        
        let metrics = analyzer.calculate_metrics();
        
        assert_eq!(metrics.consecutive_wins, 3);
        assert_eq!(metrics.consecutive_losses, 2);
    }
}
