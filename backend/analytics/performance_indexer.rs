/**
 * ZAID PERSONAL CRYPTO TRADING BOT - Stage 9
 * File: backend/analytics/performance_indexer.rs
 * 
 * Indexes trade metrics for fast frontend queries and dashboard updates.
 * Provides real-time performance tracking with minimal latency.
 * 
 * Features:
 * - Real-time metric aggregation
 * - Time-bucketed indexing (1m, 5m, 15m, 1h, 4h)
 * - PnL curve construction
 * - Symbol-wise performance breakdown
 * - Memory-efficient storage
 * 
 * Design Patterns: Observer, Builder, Repository
 */

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Trade bucket timeframes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeBucket {
    Minute1,
    Minute5,
    Minute15,
    Hour1,
    Hour4,
    Day1,
}

impl TimeBucket {
    /// Get bucket size in seconds
    pub fn as_seconds(&self) -> u64 {
        match self {
            Self::Minute1 => 60,
            Self::Minute5 => 300,
            Self::Minute15 => 900,
            Self::Hour1 => 3600,
            Self::Hour4 => 14400,
            Self::Day1 => 86400,
        }
    }
    
    /// Get bucket key for a timestamp
    pub fn get_bucket_key(&self, timestamp_ns: u64) -> u64 {
        let timestamp_s = timestamp_ns / 1_000_000_000;
        let bucket_size = self.as_seconds();
        (timestamp_s / bucket_size) * bucket_size
    }
}

/// Aggregated metrics for a time bucket
#[derive(Debug, Clone, Default)]
pub struct BucketMetrics {
    pub bucket_start_ns: u64,
    pub trade_count: u32,
    pub total_volume: f64,
    pub total_pnl: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub average_trade_duration_ms: f64,
    pub max_drawdown: f64,
    pub cumulative_pnl: f64,
}

/// Single trade record for indexing
#[derive(Debug, Clone)]
pub struct IndexedTrade {
    pub trade_id: String,
    pub symbol: String,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub pnl_percent: f64,
    pub entry_time_ns: u64,
    pub exit_time_ns: u64,
    pub duration_ms: f64,
    pub fees: f64,
}

/// Performance summary for a symbol
#[derive(Debug, Clone, Default)]
pub struct SymbolPerformance {
    pub symbol: String,
    pub total_trades: u32,
    pub total_pnl: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_win: f64,
    pub average_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub total_fees: f64,
    pub total_volume: f64,
}

/// Main performance indexer
pub struct PerformanceIndexer {
    /// Trades indexed by time bucket
    buckets: Arc<RwLock<HashMap<TimeBucket, BTreeMap<u64, BucketMetrics>>>>,
    
    /// Trades indexed by symbol
    symbol_metrics: Arc<RwLock<HashMap<String, SymbolPerformance>>>,
    
    /// All indexed trades
    trades: Arc<RwLock<Vec<IndexedTrade>>>,
    
    /// Cumulative PnL curve (timestamp, cumulative_pnl)
    pnl_curve: Arc<RwLock<Vec<(u64, f64)>>>,
    
    /// Session start time
    session_start_ns: u64,
    
    /// Current cumulative PnL
    current_cumulative_pnl: f64,
}

impl PerformanceIndexer {
    /// Create a new performance indexer
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Self {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            symbol_metrics: Arc::new(RwLock::new(HashMap::new())),
            trades: Arc::new(RwLock::new(Vec::new())),
            pnl_curve: Arc::new(RwLock::new(Vec::new())),
            session_start_ns: now,
            current_cumulative_pnl: 0.0,
        }
    }
    
    /// Index a new trade
    pub fn index_trade(&mut self, trade: IndexedTrade) {
        // Add to trades list
        {
            let mut trades = self.trades.write().unwrap();
            trades.push(trade.clone());
        }
        
        // Update cumulative PnL
        self.current_cumulative_pnl += trade.pnl;
        
        // Update PnL curve
        {
            let mut pnl_curve = self.pnl_curve.write().unwrap();
            pnl_curve.push((trade.exit_time_ns, self.current_cumulative_pnl));
        }
        
        // Update time buckets
        self.update_buckets(&trade);
        
        // Update symbol metrics
        self.update_symbol_metrics(&trade);
    }
    
    /// Update time bucket aggregations
    fn update_buckets(&mut self, trade: &IndexedTrade) {
        let buckets_to_update = [
            TimeBucket::Minute1,
            TimeBucket::Minute5,
            TimeBucket::Minute15,
            TimeBucket::Hour1,
            TimeBucket::Hour4,
        ];
        
        let mut buckets = self.buckets.write().unwrap();
        
        for bucket_type in buckets_to_update {
            let bucket_key = bucket_type.get_bucket_key(trade.exit_time_ns);
            
            let entry = buckets
                .entry(bucket_type)
                .or_insert_with(BTreeMap::new)
                .entry(bucket_key)
                .or_insert_with(|| BucketMetrics {
                    bucket_start_ns: bucket_key * 1_000_000_000,
                    ..Default::default()
                });
            
            entry.trade_count += 1;
            entry.total_volume += trade.quantity;
            entry.total_pnl += trade.pnl;
            entry.cumulative_pnl = self.current_cumulative_pnl;
            
            if trade.pnl > 0.0 {
                entry.winning_trades += 1;
                entry.gross_profit += trade.pnl;
            } else if trade.pnl < 0.0 {
                entry.losing_trades += 1;
                entry.gross_loss += trade.pnl.abs();
            }
            
            // Update average duration
            let n = entry.trade_count as f64;
            entry.average_trade_duration_ms += 
                (trade.duration_ms - entry.average_trade_duration_ms) / n;
        }
    }
    
    /// Update symbol-specific metrics
    fn update_symbol_metrics(&mut self, trade: &IndexedTrade) {
        let mut metrics = self.symbol_metrics.write().unwrap();
        
        let entry = metrics
            .entry(trade.symbol.clone())
            .or_insert_with(|| SymbolPerformance {
                symbol: trade.symbol.clone(),
                ..Default::default()
            });
        
        entry.total_trades += 1;
        entry.total_pnl += trade.pnl;
        entry.total_fees += trade.fees;
        entry.total_volume += trade.quantity * trade.entry_price;
        
        // Track wins/losses
        if trade.pnl > 0.0 {
            entry.largest_win = entry.largest_win.max(trade.pnl);
        } else if trade.pnl < 0.0 {
            entry.largest_loss = entry.largest_loss.min(trade.pnl);
        }
        
        // Calculate win rate
        let total = entry.total_trades as f64;
        if entry.total_pnl > 0.0 {
            entry.win_rate = (entry.winning_trades as f64 / total) * 100.0;
        }
        
        // Profit factor
        if entry.gross_loss > 0.0 {
            entry.profit_factor = entry.gross_profit / entry.gross_loss;
        }
    }
    
    /// Get PnL curve data
    pub fn get_pnl_curve(&self) -> Vec<(u64, f64)> {
        self.pnl_curve.read().unwrap().clone()
    }
    
    /// Get metrics for a specific time bucket
    pub fn get_bucket_metrics(
        &self,
        bucket_type: TimeBucket,
        start_ns: u64,
        end_ns: u64,
    ) -> Vec<BucketMetrics> {
        let buckets = self.buckets.read().unwrap();
        
        if let Some(bucket_map) = buckets.get(&bucket_type) {
            bucket_map
                .range((start_ns / 1_000_000_000)..=(end_ns / 1_000_000_000))
                .map(|(_, m)| m.clone())
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get performance summary for all symbols
    pub fn get_symbol_summaries(&self) -> Vec<SymbolPerformance> {
        self.symbol_metrics.read().unwrap().values().cloned().collect()
    }
    
    /// Get performance summary for a specific symbol
    pub fn get_symbol_summary(&self, symbol: &str) -> Option<SymbolPerformance> {
        self.symbol_metrics.read().unwrap().get(symbol).cloned()
    }
    
    /// Get overall session statistics
    pub fn get_session_stats(&self) -> SessionStats {
        let trades = self.trades.read().unwrap();
        
        if trades.is_empty() {
            return SessionStats::default();
        }
        
        let total_trades = trades.len();
        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let winning = trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing = trades.iter().filter(|t| t.pnl < 0.0).count();
        
        let gross_profit: f64 = trades.iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .sum();
        let gross_loss: f64 = trades.iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        };
        
        SessionStats {
            session_start_ns: self.session_start_ns,
            total_trades: total_trades as u32,
            winning_trades: winning as u32,
            losing_trades: losing as u32,
            win_rate: if total_trades > 0 {
                (winning as f64 / total_trades as f64) * 100.0
            } else {
                0.0
            },
            total_pnl,
            gross_profit,
            gross_loss,
            profit_factor,
            total_fees: trades.iter().map(|t| t.fees).sum(),
            average_pnl: total_pnl / total_trades as f64,
            current_cumulative_pnl: self.current_cumulative_pnl,
        }
    }
    
    /// Get recent trades (for live dashboard)
    pub fn get_recent_trades(&self, limit: usize) -> Vec<IndexedTrade> {
        let trades = self.trades.read().unwrap();
        trades.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

impl Default for PerformanceIndexer {
    fn default() -> Self {
        Self::new()
    }
}

/// Session-level statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub session_start_ns: u64,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub profit_factor: f64,
    pub total_fees: f64,
    pub average_pnl: f64,
    pub current_cumulative_pnl: f64,
}

/// Builder for PerformanceIndexer
pub struct PerformanceIndexerBuilder {
    indexer: PerformanceIndexer,
}

impl PerformanceIndexerBuilder {
    pub fn new() -> Self {
        Self {
            indexer: PerformanceIndexer::new(),
        }
    }
    
    pub fn with_initial_capital(mut self, capital: f64) -> Self {
        // Could be used to normalize PnL percentages
        self
    }
    
    pub fn build(self) -> PerformanceIndexer {
        self.indexer
    }
}

impl Default for PerformanceIndexerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_key_calculation() {
        let ts = 1_700_000_000_000_000_000; // Some timestamp in ns
        
        assert_eq!(TimeBucket::Minute1.get_bucket_key(ts) % 60, 0);
        assert_eq!(TimeBucket::Minute5.get_bucket_key(ts) % 300, 0);
        assert_eq!(TimeBucket::Hour1.get_bucket_key(ts) % 3600, 0);
    }

    #[test]
    fn test_trade_indexing() {
        let mut indexer = PerformanceIndexer::new();
        
        let trade = IndexedTrade {
            trade_id: "test_001".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: "BUY".to_string(),
            entry_price: 45000.0,
            exit_price: 45500.0,
            quantity: 0.1,
            pnl: 50.0,
            pnl_percent: 1.11,
            entry_time_ns: 1_700_000_000_000_000_000,
            exit_time_ns: 1_700_000_005_000_000_000,
            duration_ms: 5000.0,
            fees: 2.5,
        };
        
        indexer.index_trade(trade);
        
        let stats = indexer.get_session_stats();
        assert_eq!(stats.total_trades, 1);
        assert_eq!(stats.total_pnl, 50.0);
        
        let summaries = indexer.get_symbol_summaries();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].symbol, "BTCUSDT");
    }

    #[test]
    fn test_pnl_curve() {
        let mut indexer = PerformanceIndexer::new();
        
        for i in 0..5 {
            let trade = IndexedTrade {
                trade_id: format!("trade_{}", i),
                symbol: "BTCUSDT".to_string(),
                side: "BUY".to_string(),
                entry_price: 45000.0,
                exit_price: 45000.0 + (i as f64 * 10.0),
                quantity: 0.1,
                pnl: (i as f64) * 1.0,
                pnl_percent: 0.0,
                entry_time_ns: 1_700_000_000_000_000_000 + (i * 1_000_000_000),
                exit_time_ns: 1_700_000_005_000_000_000 + (i * 1_000_000_000),
                duration_ms: 5000.0,
                fees: 0.0,
            };
            indexer.index_trade(trade);
        }
        
        let curve = indexer.get_pnl_curve();
        assert_eq!(curve.len(), 5);
        
        // Last point should have cumulative PnL of 0+1+2+3+4 = 10
        assert!((curve.last().unwrap().1 - 10.0).abs() < 0.01);
    }
}
