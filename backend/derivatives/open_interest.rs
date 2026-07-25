//! Open Interest Analytics Tracker
//! 
//! Real-time tracking of open interest (OI) deltas and long/short ratios
//! for derivatives markets. Maps OI spikes to smart money accumulation
//! or distribution patterns.
//!
//! Features:
//! - OI delta calculation with configurable windows
//! - Long/short ratio analysis
//! - Smart money flow detection
//! - Zero-cost abstractions for high-frequency updates

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// Open interest snapshot at a point in time
#[derive(Debug, Clone)]
pub struct OISnapshot {
    pub timestamp_us: u64,
    pub open_interest: f64,  // In base asset units
    pub long_open_interest: f64,
    pub short_open_interest: f64,
    pub funding_rate: f64,
    pub volume_24h: f64,
}

impl OISnapshot {
    /// Calculate long/short ratio
    pub fn long_short_ratio(&self) -> f64 {
        if self.short_open_interest <= 0.0 {
            return 1.0;
        }
        self.long_open_interest / self.short_open_interest
    }
    
    /// Get net positioning (positive = net long)
    pub fn net_positioning(&self) -> f64 {
        self.long_open_interest - self.short_open_interest
    }
}

/// OI delta over a time window
#[derive(Debug, Clone)]
pub struct OIDelta {
    pub symbol: String,
    pub window_seconds: u64,
    pub oi_change: f64,
    pub oi_change_pct: f64,
    pub long_change: f64,
    pub short_change: f64,
    pub price_change_pct: f64,
    pub signal: OISignal,
}

/// Signal derived from OI analysis
#[derive(Debug, Clone, PartialEq)]
pub enum OISignal {
    /// OI up, Price up = Strong bullish conviction
    BullishConviction,
    /// OI down, Price up = Short covering (weak bullish)
    ShortCovering,
    /// OI up, Price down = Bearish conviction (smart money shorting)
    BearishConviction,
    /// OI down, Price down = Long liquidation (weak bearish)
    LongLiquidation,
    /// No clear signal
    Neutral,
}

/// Open interest tracker for a single asset
pub struct OITracker {
    symbol: String,
    snapshots: VecDeque<OISnapshot>,
    max_snapshots: usize,
}

impl OITracker {
    /// Create a new OI tracker for an asset
    pub fn new(symbol: &str, max_snapshots: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            snapshots: VecDeque::with_capacity(max_snapshots),
            max_snapshots,
        }
    }
    
    /// Get current timestamp in microseconds
    fn get_timestamp_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
    }
    
    /// Add a new OI snapshot
    pub fn add_snapshot(&mut self, snapshot: OISnapshot) {
        self.snapshots.push_back(snapshot);
        
        // Prune old snapshots
        while self.snapshots.len() > self.max_snapshots {
            self.snapshots.pop_front();
        }
    }
    
    /// Get the latest snapshot
    pub fn latest(&self) -> Option<&OISnapshot> {
        self.snapshots.back()
    }
    
    /// Get snapshot from N seconds ago
    pub fn get_historical(&self, seconds_ago: u64) -> Option<&OISnapshot> {
        let target_us = self.get_timestamp_us().saturating_sub(seconds_ago * 1_000_000);
        
        // Find closest snapshot before target time
        self.snapshots
            .iter()
            .rev()
            .find(|s| s.timestamp_us <= target_us)
    }
    
    /// Calculate OI delta over a time window
    pub fn calculate_delta(&self, window_seconds: u64) -> Option<OIDelta> {
        let current = self.latest()?;
        let historical = self.get_historical(window_seconds)?;
        
        let oi_change = current.open_interest - historical.open_interest;
        let oi_change_pct = if historical.open_interest > 0.0 {
            oi_change / historical.open_interest
        } else {
            0.0
        };
        
        let long_change = current.long_open_interest - historical.long_open_interest;
        let short_change = current.short_open_interest - historical.short_open_interest;
        
        // Approximate price change from OI composition
        let price_change_pct = self.estimate_price_change(historical, current);
        
        // Determine signal
        let signal = self.determine_signal(oi_change, price_change_pct);
        
        Some(OIDelta {
            symbol: self.symbol.clone(),
            window_seconds,
            oi_change,
            oi_change_pct,
            long_change,
            short_change,
            price_change_pct,
            signal,
        })
    }
    
    /// Estimate price change direction from OI shifts
    fn estimate_price_change(&self, old: &OISnapshot, new: &OISnapshot) -> f64 {
        // Simple heuristic: if longs increased more than shorts, price likely up
        let long_pressure = new.long_open_interest - old.long_open_interest;
        let short_pressure = new.short_open_interest - old.short_open_interest;
        
        let net_pressure = long_pressure - short_pressure;
        let avg_oi = (old.open_interest + new.open_interest) / 2.0;
        
        if avg_oi > 0.0 {
            net_pressure / avg_oi
        } else {
            0.0
        }
    }
    
    /// Determine OI signal based on OI and price changes
    fn determine_signal(&self, oi_change: f64, price_change_pct: f64) -> OISignal {
        let oi_positive = oi_change > 0.0;
        let price_positive = price_change_pct > 0.0;
        
        match (oi_positive, price_positive) {
            (true, true) => OISignal::BullishConviction,
            (false, true) => OISignal::ShortCovering,
            (true, false) => OISignal::BearishConviction,
            (false, false) => OISignal::LongLiquidation,
        }
    }
    
    /// Detect smart money accumulation/distribution
    pub fn detect_smart_money_flow(&self) -> Option<SmartMoneySignal> {
        // Check multiple timeframes
        let delta_1h = self.calculate_delta(3600)?;
        let delta_4h = self.calculate_delta(14400)?;
        let delta_24h = self.calculate_delta(86400)?;
        
        // Smart money accumulates when OI increases during price dips
        let is_accumulation = 
            delta_24h.signal == OISignal::BullishConviction &&
            delta_4h.oi_change > 0.0 &&
            delta_1h.long_change > delta_1h.short_change;
        
        // Smart money distributes when OI increases during price rallies
        let is_distribution =
            delta_24h.signal == OISignal::BearishConviction &&
            delta_4h.oi_change > 0.0 &&
            delta_1h.short_change > delta_1h.long_change;
        
        let signal = if is_accumulation {
            SmartMoneySignal::Accumulating
        } else if is_distribution {
            SmartMoneySignal::Distributing
        } else {
            SmartMoneySignal::Neutral
        };
        
        Some(SmartMoneySignal {
            symbol: self.symbol.clone(),
            signal,
            confidence: self.calculate_confidence(&delta_1h, &delta_4h, &delta_24h),
            oi_trend: delta_24h.oi_change_pct,
            positioning_trend: delta_24h.long_change - delta_24h.short_change,
        })
    }
    
    /// Calculate confidence score for smart money signal
    fn calculate_confidence(&self, d1h: &OIDelta, d4h: &OIDelta, d24h: &OIDelta) -> f64 {
        let mut confidence = 0.5;  // Base confidence
        
        // Increase confidence if signals align across timeframes
        if d1h.signal == d4h.signal && d4h.signal == d24h.signal {
            confidence += 0.3;
        } else if d1h.signal == d4h.signal || d4h.signal == d24h.signal {
            confidence += 0.15;
        }
        
        // Increase confidence with magnitude of OI change
        let avg_oi_change = (d1h.oi_change_pct.abs() + d4h.oi_change_pct.abs() + d24h.oi_change_pct.abs()) / 3.0;
        if avg_oi_change > 0.1 {
            confidence += 0.2;
        } else if avg_oi_change > 0.05 {
            confidence += 0.1;
        }
        
        confidence.min(1.0)
    }
    
    /// Get long/short ratio history
    pub fn get_ls_ratio_history(&self) -> Vec<(u64, f64)> {
        self.snapshots
            .iter()
            .map(|s| (s.timestamp_us, s.long_short_ratio()))
            .collect()
    }
    
    /// Get average long/short ratio over recent period
    pub fn average_ls_ratio(&self, count: usize) -> f64 {
        let recent: Vec<_> = self.snapshots.iter().rev().take(count).collect();
        if recent.is_empty() {
            return 1.0;
        }
        
        let sum: f64 = recent.iter().map(|s| s.long_short_ratio()).sum();
        sum / recent.len() as f64
    }
}

/// Smart money flow signal
#[derive(Debug, Clone)]
pub struct SmartMoneySignal {
    pub symbol: String,
    pub signal: SmartMoneyAction,
    pub confidence: f64,
    pub oi_trend: f64,
    pub positioning_trend: f64,
}

/// Smart money action type
#[derive(Debug, Clone, PartialEq)]
pub enum SmartMoneyAction {
    Accumulating,  // Building long positions
    Distributing,  // Building short positions
    Neutral,
}

/// Multi-asset OI tracker
pub struct MultiAssetOITracker {
    trackers: std::collections::HashMap<String, OITracker>,
}

impl MultiAssetOITracker {
    /// Create a new multi-asset tracker
    pub fn new() -> Self {
        Self {
            trackers: std::collections::HashMap::new(),
        }
    }
    
    /// Get or create tracker for an asset
    pub fn get_or_create_tracker(&mut self, symbol: &str) -> &mut OITracker {
        use std::collections::hash_map::Entry;
        
        match self.trackers.entry(symbol.to_string()) {
            Entry::Vacant(entry) => entry.insert(OITracker::new(symbol, 1000)),
            Entry::Occupied(entry) => entry.into_mut(),
        }
    }
    
    /// Update OI data for an asset
    pub fn update_oi(
        &mut self,
        symbol: &str,
        open_interest: f64,
        long_oi: f64,
        short_oi: f64,
        funding_rate: f64,
        volume_24h: f64,
    ) {
        let tracker = self.get_or_create_tracker(symbol);
        
        let snapshot = OISnapshot {
            timestamp_us: OITracker::get_timestamp_us(),
            open_interest,
            long_open_interest: long_oi,
            short_open_interest: short_oi,
            funding_rate,
            volume_24h,
        };
        
        tracker.add_snapshot(snapshot);
    }
    
    /// Get OI delta for an asset
    pub fn get_delta(&self, symbol: &str, window_seconds: u64) -> Option<OIDelta> {
        self.trackers.get(symbol)?.calculate_delta(window_seconds)
    }
    
    /// Get smart money signals for all assets
    pub fn get_all_smart_money_signals(&self) -> Vec<SmartMoneySignal> {
        self.trackers
            .values()
            .filter_map(|t| t.detect_smart_money_flow())
            .collect()
    }
    
    /// Find assets with strongest accumulation signals
    pub fn get_top_accumulation(&self, limit: usize) -> Vec<SmartMoneySignal> {
        let mut signals: Vec<_> = self.get_all_smart_money_signals()
            .into_iter()
            .filter(|s| s.signal == SmartMoneyAction::Accumulating)
            .collect();
        
        signals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        signals.into_iter().take(limit).collect()
    }
    
    /// Find assets with strongest distribution signals
    pub fn get_top_distribution(&self, limit: usize) -> Vec<SmartMoneySignal> {
        let mut signals: Vec<_> = self.get_all_smart_money_signals()
            .into_iter()
            .filter(|s| s.signal == SmartMoneyAction::Distributing)
            .collect();
        
        signals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        signals.into_iter().take(limit).collect()
    }
}

impl Default for MultiAssetOITracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_oi_delta_calculation() {
        let mut tracker = OITracker::new("BTCUSDT", 100);
        
        let base_time = 1_000_000_000_000_000;
        
        // Add initial snapshot
        tracker.add_snapshot(OISnapshot {
            timestamp_us: base_time,
            open_interest: 1000.0,
            long_open_interest: 550.0,
            short_open_interest: 450.0,
            funding_rate: 0.0001,
            volume_24h: 50000.0,
        });
        
        // Add later snapshot with increased OI
        tracker.add_snapshot(OISnapshot {
            timestamp_us: base_time + 3600 * 1_000_000,
            open_interest: 1100.0,
            long_open_interest: 650.0,
            short_open_interest: 450.0,
            funding_rate: 0.0002,
            volume_24h: 55000.0,
        });
        
        let delta = tracker.calculate_delta(3600);
        assert!(delta.is_some());
        
        let delta = delta.unwrap();
        assert_eq!(delta.symbol, "BTCUSDT");
        assert!(delta.oi_change > 0.0);
        assert!(delta.long_change > 0.0);
    }
    
    #[test]
    fn test_long_short_ratio() {
        let snapshot = OISnapshot {
            timestamp_us: 1_000_000_000_000_000,
            open_interest: 1000.0,
            long_open_interest: 600.0,
            short_open_interest: 400.0,
            funding_rate: 0.0001,
            volume_24h: 50000.0,
        };
        
        let ratio = snapshot.long_short_ratio();
        assert!((ratio - 1.5).abs() < 0.001);
    }
}
