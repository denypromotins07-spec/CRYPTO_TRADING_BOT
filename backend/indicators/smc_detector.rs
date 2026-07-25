"""
Smart Money Concepts (SMC) Detector in Rust for ZAID Crypto Trading Bot
Identifies Order Blocks, Fair Value Gaps (FVG), and Inducement levels
Zero-cost abstractions for microsecond execution
Thread-safe implementation using tokio async runtime

Part of the 152 domains of quantitative finance implementation.
"""

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use serde::{Serialize, Deserialize};
use chrono::Utc;

/// Candlestick data for SMC analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
    
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
    
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }
    
    pub fn range_size(&self) -> f64 {
        self.high - self.low
    }
    
    pub fn upper_wick(&self) -> f64 {
        self.high - self.open.max(self.close)
    }
    
    pub fn lower_wick(&self) -> f64 {
        self.low.min(self.open).min(self.close) - self.low
    }
}

/// Order Block types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderBlockType {
    Bullish,
    Bearish,
}

/// Detected Order Block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBlock {
    pub symbol: String,
    pub ob_type: OrderBlockType,
    pub high: f64,
    pub low: f64,
    pub open: f64,
    pub close: f64,
    pub timestamp: f64,
    pub tested: bool,
    pub mitigation_level: Option<f64>,
    pub strength: f64,
}

/// Fair Value Gap (FVG) / Imbalance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairValueGap {
    pub symbol: String,
    pub gap_type: GapType,
    pub high: f64,
    pub low: f64,
    pub start_timestamp: f64,
    pub end_timestamp: f64,
    pub filled: bool,
    pub fill_percentage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapType {
    Bullish, // Price gapped up, expect pullback
    Bearish, // Price gapped down, expect rally
}

/// Inducement level (liquidity pool that attracts price)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inducement {
    pub symbol: String,
    pub price: f64,
    pub inducement_type: InducementType,
    pub liquidity_estimate: f64,
    pub timestamp: f64,
    pub triggered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InducementType {
    EqualHighs,
    EqualLows,
    TrendlineLiquidity,
    PreviousDayHigh,
    PreviousDayLow,
}

/// Premium/Discount array levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDArray {
    pub symbol: String,
    pub level_type: PDLevelType,
    pub price: f64,
    pub percentage: f64, // Position in the range (0-100%)
    pub timestamp: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PDLevelType {
    Premium,      // Above 50%
    Discount,     // Below 50%
    Equilibrium,  // At 50%
    OTE,          // Optimal Trade Entry (62-79%)
}

/// SMC Event types for signaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SMCEvent {
    OrderBlockFormed(OrderBlock),
    FVGDetect(FairValueGap),
    InducementCreated(Inducement),
    OrderBlockTested(OrderBlock),
    FVGFilled(FairValueGap),
    InducementTriggered(Inducement),
    BreakerBlock(OrderBlock),
    MitigationBlock(OrderBlock),
}

/// State for SMC analysis on a single symbol
pub struct SMCState {
    symbol: String,
    candles: VecDeque<Candle>,
    order_blocks: Vec<OrderBlock>,
    fvgs: Vec<FairValueGap>,
    inducements: Vec<Inducement>,
    swing_highs: Vec<(f64, f64)>, // (price, timestamp)
    swing_lows: Vec<(f64, f64)>,
    lookback: usize,
}

impl SMCState {
    pub fn new(symbol: &str, lookback: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            candles: VecDeque::with_capacity(lookback * 2),
            order_blocks: Vec::with_capacity(50),
            fvgs: Vec::with_capacity(100),
            inducements: Vec::with_capacity(20),
            swing_highs: Vec::with_capacity(20),
            swing_lows: Vec::with_capacity(20),
            lookback,
        }
    }

    /// Add candle and detect SMC patterns
    pub fn update(&mut self, candle: Candle) -> Vec<SMCEvent> {
        let mut events = Vec::new();

        self.candles.push_back(candle.clone());
        if self.candles.len() > self.lookback * 2 {
            self.candles.pop_front();
        }

        // Need at least 3 candles for most SMC patterns
        if self.candles.len() < 3 {
            return events;
        }

        // Detect Order Blocks
        if let Some(ob) = self.detect_order_block() {
            events.push(SMCEvent::OrderBlockFormed(ob.clone()));
            self.order_blocks.push(ob);
        }

        // Detect Fair Value Gaps
        let fvg_events = self.detect_fvg();
        for fvg in fvg_events {
            events.push(SMCEvent::FVGDetect(fvg.clone()));
            self.fvgs.push(fvg);
        }

        // Detect Inducement levels
        if let Some(inducement) = self.detect_inducement() {
            events.push(SMCEvent::InducementCreated(inducement.clone()));
            self.inducements.push(inducement);
        }

        // Check for order block tests
        self.check_ob_tests(&mut events);

        // Check for FVG fills
        self.check_fvg_fills(&mut events);

        // Prune old data to maintain memory constraints
        self.prune();

        events
    }

    /// Detect bullish or bearish order blocks
    fn detect_order_block(&self) -> Option<OrderBlock> {
        let candles: Vec<&Candle> = self.candles.iter().collect();
        if candles.len() < 3 {
            return None;
        }

        let last = candles[candles.len() - 1];
        let prev = candles[candles.len() - 2];
        let prev2 = candles[candles.len() - 3];

        // Bullish Order Block: Strong bullish candle after consolidation/downturn
        if last.is_bullish() && last.body_size() > prev.body_size() * 1.5 {
            if last.close > prev.high && last.close > prev2.high {
                // The OB is the candle before the strong move
                let ob = OrderBlock {
                    symbol: self.symbol.clone(),
                    ob_type: OrderBlockType::Bullish,
                    high: prev.high,
                    low: prev.low,
                    open: prev.open,
                    close: prev.close,
                    timestamp: prev.timestamp,
                    tested: false,
                    mitigation_level: Some(prev.low),
                    strength: self.calculate_ob_strength(prev, OrderBlockType::Bullish),
                };
                return Some(ob);
            }
        }

        // Bearish Order Block: Strong bearish candle after consolidation/uptrend
        if last.is_bearish() && last.body_size() > prev.body_size() * 1.5 {
            if last.close < prev.low && last.close < prev2.low {
                let ob = OrderBlock {
                    symbol: self.symbol.clone(),
                    ob_type: OrderBlockType::Bearish,
                    high: prev.high,
                    low: prev.low,
                    open: prev.open,
                    close: prev.close,
                    timestamp: prev.timestamp,
                    tested: false,
                    mitigation_level: Some(prev.high),
                    strength: self.calculate_ob_strength(prev, OrderBlockType::Bearish),
                };
                return Some(ob);
            }
        }

        None
    }

    /// Calculate order block strength based on multiple factors
    fn calculate_ob_strength(&self, candle: &Candle, ob_type: OrderBlockType) -> f64 {
        let mut strength = 0.5;

        // Factor 1: Body size relative to range
        let body_ratio = candle.body_size() / candle.range_size().max(0.0001);
        strength += body_ratio * 0.2;

        // Factor 2: Volume (if available)
        // Higher volume = stronger OB

        // Factor 3: Position relative to recent swings
        // OB at swing points are stronger

        match ob_type {
            OrderBlockType::Bullish => {
                // Check if near recent lows
                if let Some((low_price, _)) = self.swing_lows.last() {
                    if (candle.low - *low_price).abs() / *low_price < 0.01 {
                        strength += 0.15;
                    }
                }
            }
            OrderBlockType::Bearish => {
                if let Some((high_price, _)) = self.swing_highs.last() {
                    if (candle.high - *high_price).abs() / *high_price < 0.01 {
                        strength += 0.15;
                    }
                }
            }
        }

        strength.min(1.0)
    }

    /// Detect Fair Value Gaps (imbalances)
    fn detect_fvg(&self) -> Vec<FairValueGap> {
        let mut fvgs = Vec::new();
        let candles: Vec<&Candle> = self.candles.iter().collect();

        if candles.len() < 3 {
            return fvgs;
        }

        let i = candles.len() - 1;
        let current = candles[i];
        let prev = candles[i - 1];
        let prev2 = candles[i - 2];

        // Bullish FVG: Gap between prev2 high and current low
        if current.is_bullish() && prev.is_bullish() {
            if prev2.high < current.low {
                let gap = FairValueGap {
                    symbol: self.symbol.clone(),
                    gap_type: GapType::Bullish,
                    high: current.low,
                    low: prev2.high,
                    start_timestamp: prev2.timestamp,
                    end_timestamp: current.timestamp,
                    filled: false,
                    fill_percentage: 0.0,
                };
                fvgs.push(gap);
            }
        }

        // Bearish FVG: Gap between prev2 low and current high
        if current.is_bearish() && prev.is_bearish() {
            if prev2.low > current.high {
                let gap = FairValueGap {
                    symbol: self.symbol.clone(),
                    gap_type: GapType::Bearish,
                    high: current.high,
                    low: prev2.low,
                    start_timestamp: prev2.timestamp,
                    end_timestamp: current.timestamp,
                    filled: false,
                    fill_percentage: 0.0,
                };
                fvgs.push(gap);
            }
        }

        fvgs
    }

    /// Detect inducement levels (equal highs/lows, trendline liquidity)
    fn detect_inducement(&self) -> Option<Inducement> {
        // Check for equal highs
        if self.swing_highs.len() >= 2 {
            let last_high = self.swing_highs.last().unwrap();
            let prev_high = self.swing_highs.get(self.swing_highs.len() - 2)?;

            if (last_high.0 - prev_high.0).abs() / last_high.0 < 0.005 {
                // Within 0.5% - considered equal
                return Some(Inducement {
                    symbol: self.symbol.clone(),
                    price: last_high.0,
                    inducement_type: InducementType::EqualHighs,
                    liquidity_estimate: 1.0, // Would need order book for real estimate
                    timestamp: last_high.1,
                    triggered: false,
                });
            }
        }

        // Check for equal lows
        if self.swing_lows.len() >= 2 {
            let last_low = self.swing_lows.last().unwrap();
            let prev_low = self.swing_lows.get(self.swing_lows.len() - 2)?;

            if (last_low.0 - prev_low.0).abs() / last_low.0 < 0.005 {
                return Some(Inducement {
                    symbol: self.symbol.clone(),
                    price: last_low.0,
                    inducement_type: InducementType::EqualLows,
                    liquidity_estimate: 1.0,
                    timestamp: last_low.1,
                    triggered: false,
                });
            }
        }

        None
    }

    /// Check if any order blocks have been tested
    fn check_ob_tests(&mut self, events: &mut Vec<SMCEvent>) {
        let current_price = self.candles.back()?.close;

        for ob in &mut self.order_blocks {
            if !ob.tested {
                match ob.ob_type {
                    OrderBlockType::Bullish => {
                        if current_price <= ob.high && current_price >= ob.low {
                            ob.tested = true;
                            ob.mitigation_level = Some(current_price);
                            let mut tested_ob = ob.clone();
                            tested_ob.tested = true;
                            events.push(SMCEvent::OrderBlockTested(tested_ob));
                            
                            // Log to SOUL.md for successful stop hunt detection
                            log_stop_hunt_to_soul(&self.symbol, "OrderBlock", ob.low, ob.high);
                        }
                    }
                    OrderBlockType::Bearish => {
                        if current_price >= ob.low && current_price <= ob.high {
                            ob.tested = true;
                            ob.mitigation_level = Some(current_price);
                            let mut tested_ob = ob.clone();
                            tested_ob.tested = true;
                            events.push(SMCEvent::OrderBlockTested(tested_ob));
                            
                            log_stop_hunt_to_soul(&self.symbol, "OrderBlock", ob.low, ob.high);
                        }
                    }
                }
            }
        }
    }

    /// Check if any FVGs have been filled
    fn check_fvg_fills(&mut self, events: &mut Vec<SMCEvent>) {
        let current_price = self.candles.back()?.close;

        for fvg in &mut self.fvgs {
            if !fvg.filled {
                match fvg.gap_type {
                    GapType::Bullish => {
                        if current_price <= fvg.high {
                            let fill_pct = (fvg.high - current_price) / (fvg.high - fvg.low).max(0.0001);
                            fvg.fill_percentage = fill_pct.min(1.0);
                            
                            if fill_pct >= 1.0 {
                                fvg.filled = true;
                                let mut filled_fvg = fvg.clone();
                                filled_fvg.filled = true;
                                events.push(SMCEvent::FVGFilled(filled_fvg));
                            }
                        }
                    }
                    GapType::Bearish => {
                        if current_price >= fvg.low {
                            let fill_pct = (current_price - fvg.low) / (fvg.high - fvg.low).max(0.0001);
                            fvg.fill_percentage = fill_pct.min(1.0);
                            
                            if fill_pct >= 1.0 {
                                fvg.filled = true;
                                let mut filled_fvg = fvg.clone();
                                filled_fvg.filled = true;
                                events.push(SMCEvent::FVGFilled(filled_fvg));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Prune old data to maintain memory constraints
    fn prune(&mut self) {
        // Keep only recent order blocks
        if self.order_blocks.len() > 50 {
            self.order_blocks.drain(0..self.order_blocks.len() - 50);
        }

        // Keep only recent FVGs
        if self.fvgs.len() > 100 {
            self.fvgs.retain(|f| !f.filled);
            while self.fvgs.len() > 50 {
                self.fvgs.remove(0);
            }
        }

        // Keep only recent inducements
        if self.inducements.len() > 20 {
            self.inducements.retain(|i| !i.triggered);
            while self.inducements.len() > 10 {
                self.inducements.remove(0);
            }
        }
    }

    /// Get premium/discount array for current range
    pub fn get_pd_array(&self) -> Option<PDArray> {
        if self.candles.is_empty() {
            return None;
        }

        let prices: Vec<f64> = self.candles.iter().flat_map(|c| vec![c.high, c.low]).collect();
        let max = *prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap())?;
        let min = *prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap())?;
        let range = max - min;

        if range == 0.0 {
            return None;
        }

        let current_price = self.candles.back()?.close;
        let percentage = ((current_price - min) / range) * 100.0;

        let level_type = if percentage > 50.0 {
            if percentage > 62.0 && percentage < 79.0 {
                PDLevelType::OTE
            } else {
                PDLevelType::Premium
            }
        } else if percentage < 50.0 {
            PDLevelType::Discount
        } else {
            PDLevelType::Equilibrium
        };

        Some(PDArray {
            symbol: self.symbol.clone(),
            level_type,
            price: current_price,
            percentage,
            timestamp: current_price as f64,
        })
    }
}

/// Main SMC Engine managing multiple symbols
pub struct SMCEngine {
    states: Arc<RwLock<HashMap<String, SMCState>>>,
    events_buffer: Arc<Mutex<VecDeque<(String, SMCEvent)>>>,
}

impl SMCEngine {
    pub fn new(symbols: &[&str], lookback: usize) -> Self {
        let mut states = HashMap::new();

        for symbol in symbols {
            states.insert(symbol.to_string(), SMCState::new(symbol, lookback));
        }

        Self {
            states: Arc::new(RwLock::new(states)),
            events_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(500))),
        }
    }

    /// Update with new candle
    pub fn update(&self, symbol: &str, candle: Candle) -> Vec<SMCEvent> {
        if let Ok(mut states) = self.states.write() {
            if let Some(state) = states.get_mut(symbol) {
                let events = state.update(candle);

                // Store significant events
                if !events.is_empty() {
                    tokio::spawn({
                        let buffer = Arc::clone(&self.events_buffer);
                        let symbol = symbol.to_string();
                        let events_clone = events.clone();
                        async move {
                            let mut buf = buffer.lock().await;
                            for event in events_clone {
                                buf.push_back((symbol.clone(), event));
                                if buf.len() > 500 {
                                    buf.pop_front();
                                }
                            }
                        }
                    });
                }

                return events;
            }
        }

        Vec::new()
    }

    /// Get recent SMC events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<(String, SMCEvent)> {
        let buffer = self.events_buffer.lock().await;
        buffer.iter().rev().take(limit).cloned().collect()
    }

    /// Get order blocks for a symbol
    pub fn get_order_blocks(&self, symbol: &str) -> Option<Vec<OrderBlock>> {
        if let Ok(states) = self.states.read() {
            states.get(symbol).map(|s| s.order_blocks.clone())
        } else {
            None
        }
    }

    /// Get untested order blocks (potential entry zones)
    pub fn get_untested_order_blocks(&self, symbol: &str) -> Option<Vec<OrderBlock>> {
        if let Ok(states) = self.states.read() {
            states.get(symbol).map(|s| {
                s.order_blocks.iter().filter(|ob| !ob.tested).cloned().collect()
            })
        } else {
            None
        }
    }

    /// Get unfilled FVGs
    pub fn get_unfilled_fvgs(&self, symbol: &str) -> Option<Vec<FairValueGap>> {
        if let Ok(states) = self.states.read() {
            states.get(symbol).map(|s| {
                s.fvgs.iter().filter(|f| !f.filled).cloned().collect()
            })
        } else {
            None
        }
    }
}

/// Helper function to log stop hunts to SOUL.md
pub fn log_stop_hunt_to_soul(symbol: &str, pattern_type: &str, low: f64, high: f64) {
    use std::fs::OpenOptions;
    use std::io::Write;

    let log_entry = format!(
        "[{}] SMC Stop Hunt detected on {}: {} pattern between {:.2} - {:.2}\n",
        Utc::now().format("%Y-%m-%d %H:%M:%S"),
        symbol,
        pattern_type,
        low,
        high
    );

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("SOUL.md")
    {
        let _ = file.write_all(log_entry.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smc_engine_initialization() {
        let engine = SMCEngine::new(&["BTCUSDT", "ETHUSDT"], 100);
        let candles = engine.get_order_blocks("BTCUSDT");
        assert!(candles.is_some());
        assert!(candles.unwrap().is_empty());
    }

    #[test]
    fn test_candle_properties() {
        let candle = Candle {
            timestamp: 1234567890.0,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 104.0,
            volume: 1000.0,
        };

        assert!(candle.is_bullish());
        assert_eq!(candle.body_size(), 4.0);
        assert_eq!(candle.range_size(), 6.0);
    }
}
