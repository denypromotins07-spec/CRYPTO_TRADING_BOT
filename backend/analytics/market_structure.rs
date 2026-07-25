"""
Market Structure Detection Engine in Rust for ZAID Crypto Trading Bot
Detects BOS (Break of Structure), CHoCH (Change of Character), and Liquidity Sweeps
Zero-cost abstractions for microsecond execution
Thread-safe implementation using tokio async runtime

Part of the 152 domains of quantitative finance implementation.
"""

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Price point with timestamp for structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub timestamp: f64,
    pub price: f64,
    pub high: f64,
    pub low: f64,
    pub volume: f64,
}

/// Market structure event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StructureEvent {
    BOS(BreakOfStructure),
    CHoCH(ChangeOfCharacter),
    LiquiditySweep(LiquiditySweep),
    HigherHigh,
    LowerLow,
    EqualHighs,
    EqualLows,
}

/// Break of Structure - confirms trend continuation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakOfStructure {
    pub symbol: String,
    pub direction: TrendDirection,
    pub breakout_price: f64,
    pub previous_structure: f64,
    pub strength: f64,
    pub timestamp: f64,
    pub confirmed: bool,
}

/// Change of Character - potential trend reversal signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeOfCharacter {
    pub symbol: String,
    pub from_trend: TrendDirection,
    pub to_trend: TrendDirection,
    pub reversal_price: f64,
    pub key_level_broken: f64,
    pub timestamp: f64,
    pub confidence: f64,
}

/// Liquidity Sweep - stop hunt detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquiditySweep {
    pub symbol: String,
    pub sweep_type: SweepType,
    pub swept_price: f64,
    pub liquidity_level: f64,
    pub rejection_speed: f64,
    pub timestamp: f64,
    pub followed_by_reversal: bool,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Bullish,
    Bearish,
    Neutral,
}

/// Sweep type for liquidity events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SweepType {
    HighSweep,
    LowSweep,
    BothSides,
}

/// Swing point identification
#[derive(Debug, Clone)]
pub struct SwingPoint {
    pub price: f64,
    pub timestamp: f64,
    pub swing_type: SwingType,
    pub strength: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwingType {
    High,
    Low,
}

/// Market structure state for a single symbol
pub struct MarketStructureState {
    symbol: String,
    swing_highs: VecDeque<SwingPoint>,
    swing_lows: VecDeque<SwingPoint>,
    recent_prices: VecDeque<PricePoint>,
    current_trend: TrendDirection,
    last_bos_price: Option<f64>,
    last_choch_price: Option<f64>,
    key_levels: Vec<f64>,
    swing_lookback: usize,
}

impl MarketStructureState {
    pub fn new(symbol: &str, lookback: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            swing_highs: VecDeque::with_capacity(50),
            swing_lows: VecDeque::with_capacity(50),
            recent_prices: VecDeque::with_capacity(lookback * 3),
            current_trend: TrendDirection::Neutral,
            last_bos_price: None,
            last_choch_price: None,
            key_levels: Vec::with_capacity(20),
            swing_lookback: lookback,
        }
    }

    /// Add new price data and detect structure changes
    pub fn update(&mut self, point: PricePoint) -> Vec<StructureEvent> {
        let mut events = Vec::new();

        // Store recent price
        self.recent_prices.push_back(point.clone());
        if self.recent_prices.len() > self.swing_lookback * 3 {
            self.recent_prices.pop_front();
        }

        // Detect swing points
        if let Some(swing) = self.detect_swing_point() {
            match swing.swing_type {
                SwingType::High => {
                    self.swing_highs.push_back(swing.clone());
                    if self.swing_highs.len() > 50 {
                        self.swing_highs.pop_front();
                    }
                }
                SwingType::Low => {
                    self.swing_lows.push_back(swing.clone());
                    if self.swing_lows.len() > 50 {
                        self.swing_lows.pop_front();
                    }
                }
            }

            // Check for structure breaks
            if let Some(event) = self.check_bos(&swing) {
                events.push(event);
            }

            if let Some(event) = self.check_choch(&swing) {
                events.push(event);
            }
        }

        // Check for liquidity sweeps
        if let Some(sweep_event) = self.check_liquidity_sweep(&point) {
            events.push(StructureEvent::LiquiditySweep(sweep_event));
        }

        events
    }

    /// Detect swing point based on lookback period
    fn detect_swing_point(&self) -> Option<SwingPoint> {
        if self.recent_prices.len() < self.swing_lookback * 2 + 1 {
            return None;
        }

        let current_idx = self.recent_prices.len() - 1;
        let lookback = self.swing_lookback;

        // Check for swing high
        let mut is_swing_high = true;
        let current_high = self.recent_prices[current_idx].high;
        
        for i in (current_idx - lookback)..=(current_idx - 1) {
            if self.recent_prices[i].high >= current_high {
                is_swing_high = false;
                break;
            }
        }

        if is_swing_high && current_idx >= lookback {
            for i in (current_idx + 1 - lookback)..=current_idx {
                if i < self.recent_prices.len() && self.recent_prices[i].high >= current_high {
                    is_swing_high = false;
                    break;
                }
            }
        }

        if is_swing_high {
            return Some(SwingPoint {
                price: current_high,
                timestamp: self.recent_prices[current_idx].timestamp,
                swing_type: SwingType::High,
                strength: self.calculate_swing_strength(current_idx, SwingType::High),
            });
        }

        // Check for swing low
        let mut is_swing_low = true;
        let current_low = self.recent_prices[current_idx].low;

        for i in (current_idx - lookback)..=(current_idx - 1) {
            if self.recent_prices[i].low <= current_low {
                is_swing_low = false;
                break;
            }
        }

        if is_swing_low && current_idx >= lookback {
            for i in (current_idx + 1 - lookback)..=current_idx {
                if i < self.recent_prices.len() && self.recent_prices[i].low <= current_low {
                    is_swing_low = false;
                    break;
                }
            }
        }

        if is_swing_low {
            return Some(SwingPoint {
                price: current_low,
                timestamp: self.recent_prices[current_idx].timestamp,
                swing_type: SwingType::Low,
                strength: self.calculate_swing_strength(current_idx, SwingType::Low),
            });
        }

        None
    }

    /// Calculate swing point strength based on surrounding price action
    fn calculate_swing_strength(&self, idx: usize, swing_type: SwingType) -> u32 {
        let mut strength = 1u32;
        let lookback = self.swing_lookback;

        match swing_type {
            SwingType::High => {
                let current_high = self.recent_prices[idx].high;
                for i in idx.saturating_sub(lookback)..idx {
                    if self.recent_prices[i].high < current_high * 0.999 {
                        strength += 1;
                    }
                }
            }
            SwingType::Low => {
                let current_low = self.recent_prices[idx].low;
                for i in idx.saturating_sub(lookback)..idx {
                    if self.recent_prices[i].low > current_low * 1.001 {
                        strength += 1;
                    }
                }
            }
        }

        strength.min(10)
    }

    /// Check for Break of Structure
    fn check_bos(&mut self, swing: &SwingPoint) -> Option<StructureEvent> {
        match swing.swing_type {
            SwingType::High => {
                // In bullish trend, check if we broke previous high
                if self.current_trend == TrendDirection::Bullish {
                    if let Some(prev_high) = self.swing_highs.iter().rev().skip(1).next() {
                        if swing.price > prev_high.price {
                            let strength = (swing.price - prev_high.price) / prev_high.price * 100.0;
                            
                            self.last_bos_price = Some(swing.price);
                            
                            return Some(StructureEvent::BOS(BreakOfStructure {
                                symbol: self.symbol.clone(),
                                direction: TrendDirection::Bullish,
                                breakout_price: swing.price,
                                previous_structure: prev_high.price,
                                strength,
                                timestamp: swing.timestamp,
                                confirmed: strength > 0.001, // 0.1% move
                            }));
                        }
                    }
                } else if self.current_trend == TrendDirection::Bearish {
                    // Potential CHoCH if breaking bearish structure
                    if let Some(prev_high) = self.swing_highs.front() {
                        if swing.price > prev_high.price {
                            self.current_trend = TrendDirection::Bullish;
                            self.last_choch_price = Some(swing.price);
                            
                            return Some(StructureEvent::CHoCH(ChangeOfCharacter {
                                symbol: self.symbol.clone(),
                                from_trend: TrendDirection::Bearish,
                                to_trend: TrendDirection::Bullish,
                                reversal_price: swing.price,
                                key_level_broken: prev_high.price,
                                timestamp: swing.timestamp,
                                confidence: 0.7,
                            }));
                        }
                    }
                }
            }
            SwingType::Low => {
                // In bearish trend, check if we broke previous low
                if self.current_trend == TrendDirection::Bearish {
                    if let Some(prev_low) = self.swing_lows.iter().rev().skip(1).next() {
                        if swing.price < prev_low.price {
                            let strength = (prev_low.price - swing.price) / prev_low.price * 100.0;
                            
                            self.last_bos_price = Some(swing.price);
                            
                            return Some(StructureEvent::BOS(BreakOfStructure {
                                symbol: self.symbol.clone(),
                                direction: TrendDirection::Bearish,
                                breakout_price: swing.price,
                                previous_structure: prev_low.price,
                                strength,
                                timestamp: swing.timestamp,
                                confirmed: strength > 0.001,
                            }));
                        }
                    }
                } else if self.current_trend == TrendDirection::Bullish {
                    // Potential CHoCH if breaking bullish structure
                    if let Some(prev_low) = self.swing_lows.front() {
                        if swing.price < prev_low.price {
                            self.current_trend = TrendDirection::Bearish;
                            self.last_choch_price = Some(swing.price);
                            
                            return Some(StructureEvent::CHoCH(ChangeOfCharacter {
                                symbol: self.symbol.clone(),
                                from_trend: TrendDirection::Bullish,
                                to_trend: TrendDirection::Bearish,
                                reversal_price: swing.price,
                                key_level_broken: prev_low.price,
                                timestamp: swing.timestamp,
                                confidence: 0.7,
                            }));
                        }
                    }
                }
            }
        }

        None
    }

    /// Check for Change of Character
    fn check_choch(&mut self, swing: &SwingPoint) -> Option<StructureEvent> {
        // Additional CHoCH logic for early reversal detection
        match (self.current_trend, swing.swing_type) {
            (TrendDirection::Bullish, SwingType::Low) => {
                if let Some(prev_low) = self.swing_lows.iter().rev().skip(1).next() {
                    if swing.price < prev_low.price {
                        self.current_trend = TrendDirection::Bearish;
                        self.last_choch_price = Some(swing.price);
                        
                        return Some(StructureEvent::CHoCH(ChangeOfCharacter {
                            symbol: self.symbol.clone(),
                            from_trend: TrendDirection::Bullish,
                            to_trend: TrendDirection::Bearish,
                            reversal_price: swing.price,
                            key_level_broken: prev_low.price,
                            timestamp: swing.timestamp,
                            confidence: 0.6,
                        }));
                    }
                }
            }
            (TrendDirection::Bearish, SwingType::High) => {
                if let Some(prev_high) = self.swing_highs.iter().rev().skip(1).next() {
                    if swing.price > prev_high.price {
                        self.current_trend = TrendDirection::Bullish;
                        self.last_choch_price = Some(swing.price);
                        
                        return Some(StructureEvent::CHoCH(ChangeOfCharacter {
                            symbol: self.symbol.clone(),
                            from_trend: TrendDirection::Bearish,
                            to_trend: TrendDirection::Bullish,
                            reversal_price: swing.price,
                            key_level_broken: prev_high.price,
                            timestamp: swing.timestamp,
                            confidence: 0.6,
                        }));
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Check for liquidity sweeps (stop hunts)
    fn check_liquidity_sweep(&self, point: &PricePoint) -> Option<LiquiditySweep> {
        // Check if price swept above recent highs but closed below
        if let Some(recent_high) = self.swing_highs.back() {
            if point.high > recent_high.price && point.price < recent_high.price {
                let rejection_speed = (recent_high.price - point.price) / recent_high.price * 100.0;
                
                return Some(LiquiditySweep {
                    symbol: self.symbol.clone(),
                    sweep_type: SweepType::HighSweep,
                    swept_price: point.high,
                    liquidity_level: recent_high.price,
                    rejection_speed,
                    timestamp: point.timestamp,
                    followed_by_reversal: rejection_speed > 0.002,
                });
            }
        }

        // Check if price swept below recent lows but closed above
        if let Some(recent_low) = self.swing_lows.back() {
            if point.low < recent_low.price && point.price > recent_low.price {
                let rejection_speed = (point.price - recent_low.price) / recent_low.price * 100.0;
                
                return Some(LiquiditySweep {
                    symbol: self.symbol.clone(),
                    sweep_type: SweepType::LowSweep,
                    swept_price: point.low,
                    liquidity_level: recent_low.price,
                    rejection_speed,
                    timestamp: point.timestamp,
                    followed_by_reversal: rejection_speed > 0.002,
                });
            }
        }

        None
    }

    /// Get current trend
    pub fn get_trend(&self) -> TrendDirection {
        self.current_trend
    }

    /// Get key support/resistance levels
    pub fn get_key_levels(&self) -> Vec<f64> {
        let mut levels = Vec::new();
        
        for swing in &self.swing_highs {
            levels.push(swing.price);
        }
        for swing in &self.swing_lows {
            levels.push(swing.price);
        }
        
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        levels.dedup();
        
        levels
    }
}

/// Main market structure engine managing multiple symbols
pub struct MarketStructureEngine {
    states: Arc<RwLock<HashMap<String, MarketStructureState>>>,
    events_buffer: Arc<Mutex<VecDeque<(String, StructureEvent)>>>,
    max_events: usize,
}

impl MarketStructureEngine {
    pub fn new(symbols: &[&str], swing_lookback: usize) -> Self {
        let mut states = HashMap::new();
        
        for symbol in symbols {
            states.insert(symbol.to_string(), MarketStructureState::new(symbol, swing_lookback));
        }
        
        Self {
            states: Arc::new(RwLock::new(states)),
            events_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            max_events: 1000,
        }
    }

    /// Update price data for a symbol
    pub fn update(&self, symbol: &str, point: PricePoint) -> Vec<StructureEvent> {
        if let Ok(mut states) = self.states.write() {
            if let Some(state) = states.get_mut(symbol) {
                let events = state.update(point);
                
                // Store events in buffer
                if !events.is_empty() {
                    tokio::spawn({
                        let buffer = Arc::clone(&self.events_buffer);
                        let symbol = symbol.to_string();
                        let events_clone = events.clone();
                        async move {
                            let mut buf = buffer.lock().await;
                            for event in events_clone {
                                buf.push_back((symbol.clone(), event));
                                if buf.len() > 1000 {
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

    /// Get recent structure events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<(String, StructureEvent)> {
        let buffer = self.events_buffer.lock().await;
        buffer.iter().rev().take(limit).cloned().collect()
    }

    /// Get current trend for a symbol
    pub fn get_trend(&self, symbol: &str) -> Option<TrendDirection> {
        if let Ok(states) = self.states.read() {
            states.get(symbol).map(|s| s.get_trend())
        } else {
            None
        }
    }

    /// Get key levels for a symbol
    pub fn get_key_levels(&self, symbol: &str) -> Option<Vec<f64>> {
        if let Ok(states) = self.states.read() {
            states.get(symbol).map(|s| s.get_key_levels())
        } else {
            None
        }
    }
}

/// Helper function to write significant events to SOUL.md
pub fn log_to_soul(symbol: &str, event: &StructureEvent) {
    use std::fs::OpenOptions;
    use std::io::Write;
    
    let event_type = match event {
        StructureEvent::BOS(bos) => format!(
            "BOS detected on {}: {} breakout at {:.2} (strength: {:.4}%)",
            symbol,
            match bos.direction {
                TrendDirection::Bullish => "BULLISH",
                TrendDirection::Bearish => "BEARISH",
                _ => "NEUTRAL",
            },
            bos.breakout_price,
            bos.strength
        ),
        StructureEvent::CHoCH(choch) => format!(
            "CHoCH on {}: {} -> {} at {:.2}",
            symbol,
            match choch.from_trend {
                TrendDirection::Bullish => "BULLISH",
                TrendDirection::Bearish => "BEARISH",
                _ => "NEUTRAL",
            },
            match choch.to_trend {
                TrendDirection::Bullish => "BULLISH",
                TrendDirection::Bearish => "BEARISH",
                _ => "NEUTRAL",
            },
            choch.reversal_price
        ),
        StructureEvent::LiquiditySweep(sweep) => format!(
            "LIQUIDITY SWEEP on {}: {} at {:.2}, rejection speed: {:.4}%",
            symbol,
            match sweep.sweep_type {
                SweepType::HighSweep => "HIGH",
                SweepType::LowSweep => "LOW",
                SweepType::BothSides => "BOTH",
            },
            sweep.swept_price,
            sweep.rejection_speed
        ),
        _ => format!("Structure event on {}: {:?}", symbol, event),
    };

    let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let log_entry = format!("[{}] {}\n", timestamp, event_type);

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
    fn test_market_structure_initialization() {
        let engine = MarketStructureEngine::new(&["BTCUSDT", "ETHUSDT"], 5);
        assert_eq!(engine.get_trend("BTCUSDT"), Some(TrendDirection::Neutral));
    }

    #[test]
    fn test_price_point_creation() {
        let point = PricePoint {
            timestamp: 1234567890.0,
            price: 45000.0,
            high: 45100.0,
            low: 44900.0,
            volume: 100.0,
        };
        assert_eq!(point.price, 45000.0);
    }
}
