/*
 * ZAID PERSONAL CRYPTO TRADING BOT - Stage 6
 * Chapter 4: Backtest Analytics, Equity Curve Analysis, and SOUL.md Integration
 *
 * File: backend/analytics/trade_distribution.rs
 * Purpose: Analyze win rates, profit factors, fat tails, and trade distributions.
 * Features:
 *     - Trade-level statistical analysis
 *     - Fat tail detection and modeling
 *     - Win/loss distribution fitting
 *     - Monte Carlo simulation for robustness
 */

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use log::{info, debug, warn};

/// Individual trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub trade_id: u64,
    pub symbol: String,
    pub side: String,  // "buy" or "sell"
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub entry_time: u128,
    pub exit_time: u128,
    pub duration_ms: u128,
    pub fees: f64,
    pub slippage: f64,
}

/// Distribution fitting results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionFit {
    pub distribution_name: String,
    pub parameters: HashMap<String, f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub ks_statistic: f64,
    pub ks_pvalue: f64,
}

/// Trade analysis summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeAnalysis {
    // Basic statistics
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub breakeven_trades: usize,
    
    // Win rate metrics
    pub win_rate: f64,
    pub win_rate_long: f64,
    pub win_rate_short: f64,
    
    // Profit metrics
    pub total_pnl: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub net_profit: f64,
    pub profit_factor: f64,
    
    // Average metrics
    pub avg_pnl: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
    pub win_loss_ratio: f64,
    
    // Extreme values
    pub largest_win: f64,
    pub largest_loss: f64,
    pub largest_win_pct: f64,
    pub largest_loss_pct: f64,
    
    // Consecutive stats
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    
    // Time metrics
    pub avg_trade_duration_ms: f64,
    pub median_trade_duration_ms: f64,
    
    // Distribution metrics
    pub skewness: f64,
    pub kurtosis: f64,
    pub tail_index: f64,  // Pareto tail index
    
    // Risk metrics
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub ulcer_index: f64,
    
    // Fat tail indicators
    pub is_fat_tailed: bool,
    pub tail_ratio: f64,
    pub extreme_value_index: f64,
}

/// Trade distribution analyzer
pub struct TradeDistributionAnalyzer {
    trades: Vec<Trade>,
    pnls: Vec<f64>,
    pnl_pcts: Vec<f64>,
}

impl TradeDistributionAnalyzer {
    /// Create a new analyzer with trade data
    pub fn new(trades: Vec<Trade>) -> Self {
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let pnl_pcts: Vec<f64> = trades.iter().map(|t| t.pnl_pct).collect();
        
        Self {
            trades,
            pnls,
            pnl_pcts,
        }
    }

    /// Get number of trades
    pub fn n_trades(&self) -> usize {
        self.trades.len()
    }

    /// Calculate basic win/loss statistics
    fn calculate_win_loss_stats(&self) -> (usize, usize, usize, f64) {
        let mut wins = 0;
        let mut losses = 0;
        let mut breakeven = 0;
        
        for pnl in &self.pnls {
            if *pnl > 0.0 {
                wins += 1;
            } else if *pnl < 0.0 {
                losses += 1;
            } else {
                breakeven += 1;
            }
        }
        
        let total = wins + losses + breakeven;
        let win_rate = if total > 0 {
            wins as f64 / total as f64
        } else {
            0.0
        };
        
        (wins, losses, breakeven, win_rate)
    }

    /// Calculate profit factor
    fn calculate_profit_factor(&self) -> f64 {
        let gross_profit: f64 = self.pnls.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = self.pnls.iter().filter(|&&p| p < 0.0).map(|&p| p.abs()).sum();
        
        if gross_loss == 0.0 {
            if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            gross_profit / gross_loss
        }
    }

    /// Calculate consecutive wins/losses
    fn calculate_consecutive_stats(&self) -> (usize, usize) {
        let mut max_wins = 0;
        let mut max_losses = 0;
        let mut current_wins = 0;
        let mut current_losses = 0;
        
        for pnl in &self.pnls {
            if *pnl > 0.0 {
                current_wins += 1;
                current_losses = 0;
                max_wins = max_wins.max(current_wins);
            } else if *pnl < 0.0 {
                current_losses += 1;
                current_wins = 0;
                max_losses = max_losses.max(current_losses);
            } else {
                current_wins = 0;
                current_losses = 0;
            }
        }
        
        (max_wins, max_losses)
    }

    /// Calculate skewness of PnL distribution
    fn calculate_skewness(&self) -> f64 {
        if self.pnls.len() < 3 {
            return 0.0;
        }
        
        let n = self.pnls.len() as f64;
        let mean = self.mean();
        let std = self.std();
        
        if std == 0.0 {
            return 0.0;
        }
        
        let m3: f64 = self.pnls.iter()
            .map(|&x| ((x - mean) / std).powi(3))
            .sum::<f64>() / n;
        
        // Adjust for sample bias
        m3 * (n * (n - 1.0)).sqrt() / (n - 2.0)
    }

    /// Calculate kurtosis of PnL distribution
    fn calculate_kurtosis(&self) -> f64 {
        if self.pnls.len() < 4 {
            return 0.0;
        }
        
        let n = self.pnls.len() as f64;
        let mean = self.mean();
        let std = self.std();
        
        if std == 0.0 {
            return 0.0;
        }
        
        let m4: f64 = self.pnls.iter()
            .map(|&x| ((x - mean) / std).powi(4))
            .sum::<f64>() / n;
        
        // Excess kurtosis (subtract 3 for normal distribution)
        let excess = m4 - 3.0;
        
        // Adjust for sample bias
        excess * ((n + 1.0) * (n - 1.0)) / ((n - 2.0) * (n - 3.0))
    }

    /// Estimate tail index using Hill estimator
    fn estimate_tail_index(&self, k: usize) -> f64 {
        if self.pnls.len() < k + 1 || k < 10 {
            return 2.0;  // Default to finite variance
        }
        
        // Sort absolute losses in descending order
        let mut abs_losses: Vec<f64> = self.pnls.iter()
            .filter(|&&x| x < 0.0)
            .map(|&x| x.abs())
            .collect();
        
        abs_losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        if abs_losses.len() < k + 1 {
            return 2.0;
        }
        
        // Hill estimator
        let threshold = abs_losses[k];
        let hill_sum: f64 = abs_losses[..k].iter()
            .map(|&x| (x / threshold).ln())
            .sum();
        
        let gamma = hill_sum / k as f64;
        
        // Tail index is inverse of gamma
        if gamma > 0.0 {
            1.0 / gamma
        } else {
            2.0
        }
    }

    /// Mean of PnL
    fn mean(&self) -> f64 {
        if self.pnls.is_empty() {
            return 0.0;
        }
        self.pnls.iter().sum::<f64>() / self.pnls.len() as f64
    }

    /// Standard deviation of PnL
    fn std(&self) -> f64 {
        if self.pnls.len() < 2 {
            return 0.0;
        }
        
        let mean = self.mean();
        let variance: f64 = self.pnls.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (self.pnls.len() - 1) as f64;
        
        variance.sqrt()
    }

    /// Detect fat tails in the distribution
    fn detect_fat_tails(&self) -> (bool, f64) {
        let kurtosis = self.calculate_kurtosis();
        let tail_index = self.estimate_tail_index(self.pnls.len() / 10);
        
        // Fat tails if:
        // 1. Excess kurtosis > 0 (leptokurtic)
        // 2. Tail index < 3 (infinite variance possible)
        let is_fat = kurtosis > 0.5 || tail_index < 3.0;
        
        // Tail ratio: ratio of actual tail probability to normal
        let tail_ratio = if kurtosis > 0.0 {
            1.0 + kurtosis / 6.0  // Approximate adjustment
        } else {
            1.0
        };
        
        (is_fat, tail_ratio)
    }

    /// Perform complete trade analysis
    pub fn analyze(&self) -> TradeAnalysis {
        let (wins, losses, breakeven, win_rate) = self.calculate_win_loss_stats();
        let profit_factor = self.calculate_profit_factor();
        let (max_consec_wins, max_consec_losses) = self.calculate_consecutive_stats();
        let skewness = self.calculate_skewness();
        let kurtosis = self.calculate_kurtosis();
        let (is_fat_tailed, tail_ratio) = self.detect_fat_tails();
        let tail_index = self.estimate_tail_index(self.pnls.len() / 10);
        
        // Calculate profit/loss breakdowns
        let gross_profit: f64 = self.pnls.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = self.pnls.iter().filter(|&&p| p < 0.0).map(|&p| p.abs()).sum();
        let total_pnl: f64 = self.pnls.iter().sum();
        
        let avg_pnl = if !self.pnls.is_empty() {
            total_pnl / self.pnls.len() as f64
        } else {
            0.0
        };
        
        let avg_win = if wins > 0 {
            gross_profit / wins as f64
        } else {
            0.0
        };
        
        let avg_loss = if losses > 0 {
            gross_loss / losses as f64
        } else {
            0.0
        };
        
        let win_loss_ratio = if avg_loss > 0.0 {
            avg_win / avg_loss
        } else {
            0.0
        };
        
        // Find extremes
        let (largest_win, largest_loss) = self.pnls.iter()
            .fold((f64::NEG_INFINITY, f64::INFINITY), |(max_w, min_l), &pnl| {
                (max_w.max(pnl), min_l.min(pnl))
            });
        
        // Duration stats
        let durations: Vec<f64> = self.trades.iter()
            .map(|t| t.duration_ms as f64)
            .collect();
        
        let avg_duration = if !durations.is_empty() {
            durations.iter().sum::<f64>() / durations.len() as f64
        } else {
            0.0
        };
        
        let mut sorted_durations = durations.clone();
        sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_duration = if !sorted_durations.is_empty() {
            sorted_durations[sorted_durations.len() / 2]
        } else {
            0.0
        };
        
        // Sharpe-like metric for trades
        let sharpe = if self.std() > 0.0 {
            self.mean() / self.std() * (252.0).sqrt()  // Annualize assuming daily
        } else {
            0.0
        };
        
        // Ulcer index (measure of downside volatility)
        let ulcer_index = self.calculate_ulcer_index();
        
        TradeAnalysis {
            total_trades: self.trades.len(),
            winning_trades: wins,
            losing_trades: losses,
            breakeven_trades: breakeven,
            win_rate,
            win_rate_long: 0.0,  // Would need to filter by side
            win_rate_short: 0.0,
            total_pnl,
            gross_profit,
            gross_loss,
            net_profit: total_pnl,
            profit_factor,
            avg_pnl,
            avg_win,
            avg_loss,
            avg_win_pct: 0.0,
            avg_loss_pct: 0.0,
            win_loss_ratio,
            largest_win,
            largest_loss,
            largest_win_pct: 0.0,
            largest_loss_pct: 0.0,
            max_consecutive_wins,
            max_consecutive_losses,
            avg_trade_duration_ms: avg_duration,
            median_trade_duration_ms: median_duration,
            skewness,
            kurtosis,
            tail_index,
            sharpe_ratio: sharpe,
            sortino_ratio: 0.0,  // Would need downside calculation
            ulcer_index,
            is_fat_tailed,
            tail_ratio,
            extreme_value_index: 1.0 / tail_index,
        }
    }

    /// Calculate Ulcer Index
    fn calculate_ulcer_index(&self) -> f64 {
        if self.pnls.is_empty() {
            return 0.0;
        }
        
        // Calculate running drawdown from peak
        let mut peak = 0.0;
        let mut sum_sq_dd = 0.0;
        
        for pnl in &self.pnls {
            let cumulative = peak + pnl;
            if cumulative > peak {
                peak = cumulative;
            }
            
            let dd = (peak - cumulative) / peak.max(1.0);
            sum_sq_dd += dd * dd;
        }
        
        (sum_sq_dd / self.pnls.len() as f64).sqrt()
    }

    /// Get trades by symbol
    pub fn get_trades_by_symbol(&self, symbol: &str) -> Vec<&Trade> {
        self.trades.iter()
            .filter(|t| t.symbol == symbol)
            .collect()
    }

    /// Filter trades by profitability
    pub fn get_profitable_trades(&self) -> Vec<&Trade> {
        self.trades.iter()
            .filter(|t| t.pnl > 0.0)
            .collect()
    }

    /// Get worst trades
    pub fn get_worst_trades(&self, n: usize) -> Vec<&Trade> {
        let mut sorted: Vec<&Trade> = self.trades.iter().collect();
        sorted.sort_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_analyzer_creation() {
        let trades = vec![
            Trade {
                trade_id: 1,
                symbol: "BTCUSDT".to_string(),
                side: "buy".to_string(),
                entry_price: 50000.0,
                exit_price: 51000.0,
                quantity: 0.1,
                pnl: 100.0,
                pnl_pct: 0.02,
                entry_time: 1000000,
                exit_time: 1001000,
                duration_ms: 1000,
                fees: 1.0,
                slippage: 0.5,
            },
            Trade {
                trade_id: 2,
                symbol: "BTCUSDT".to_string(),
                side: "sell".to_string(),
                entry_price: 51000.0,
                exit_price: 50500.0,
                quantity: 0.1,
                pnl: 50.0,
                pnl_pct: 0.01,
                entry_time: 1002000,
                exit_time: 1003000,
                duration_ms: 1000,
                fees: 1.0,
                slippage: 0.5,
            },
        ];

        let analyzer = TradeDistributionAnalyzer::new(trades);
        assert_eq!(analyzer.n_trades(), 2);
    }

    #[test]
    fn test_win_loss_calculation() {
        let trades = vec![
            Trade {
                trade_id: 1,
                symbol: "BTCUSDT".to_string(),
                side: "buy".to_string(),
                entry_price: 50000.0,
                exit_price: 51000.0,
                quantity: 0.1,
                pnl: 100.0,
                pnl_pct: 0.02,
                entry_time: 1000000,
                exit_time: 1001000,
                duration_ms: 1000,
                fees: 1.0,
                slippage: 0.5,
            },
            Trade {
                trade_id: 2,
                symbol: "BTCUSDT".to_string(),
                side: "buy".to_string(),
                entry_price: 51000.0,
                exit_price: 50000.0,
                quantity: 0.1,
                pnl: -100.0,
                pnl_pct: -0.02,
                entry_time: 1002000,
                exit_time: 1003000,
                duration_ms: 1000,
                fees: 1.0,
                slippage: 0.5,
            },
        ];

        let analyzer = TradeDistributionAnalyzer::new(trades);
        let analysis = analyzer.analyze();

        assert_eq!(analysis.winning_trades, 1);
        assert_eq!(analysis.losing_trades, 1);
        assert!((analysis.win_rate - 0.5).abs() < 0.01);
    }
}
