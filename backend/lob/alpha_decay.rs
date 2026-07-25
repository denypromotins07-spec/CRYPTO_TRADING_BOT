//! Alpha Decay - Measures how fast predictive signals degrade due to market impact.

use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct AlphaDecayConfig {
    pub half_life_secs: f64,
    pub decay_type: DecayType,
}

impl Default for AlphaDecayConfig {
    fn default() -> Self {
        Self {
            half_life_secs: 5.0,
            decay_type: DecayType::Exponential,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DecayType {
    Exponential,
    Linear,
    PowerLaw,
}

#[derive(Debug)]
pub struct AlphaSignal {
    pub timestamp: Instant,
    pub initial_alpha: f64,
    pub current_alpha: f64,
    pub signal_type: String,
}

pub struct AlphaDecayTracker {
    config: AlphaDecayConfig,
    signals: Vec<AlphaSignal>,
}

impl AlphaDecayTracker {
    pub fn new(config: AlphaDecayConfig) -> Self {
        Self {
            config,
            signals: Vec::with_capacity(100),
        }
    }
    
    pub fn add_signal(&mut self, alpha: f64, signal_type: &str) {
        let signal = AlphaSignal {
            timestamp: Instant::now(),
            initial_alpha: alpha,
            current_alpha: alpha,
            signal_type: signal_type.to_string(),
        };
        self.signals.push(signal);
    }
    
    pub fn get_decayed_alpha(&mut self) -> f64 {
        let now = Instant::now();
        let mut total_alpha = 0.0;
        
        for signal in &mut self.signals {
            let elapsed = now.duration_since(signal.timestamp).as_secs_f64();
            signal.current_alpha = self.calculate_decay(signal.initial_alpha, elapsed);
            total_alpha += signal.current_alpha;
        }
        
        // Remove fully decayed signals
        self.signals.retain(|s| s.current_alpha > 0.001);
        
        total_alpha
    }
    
    fn calculate_decay(&self, initial: f64, elapsed: f64) -> f64 {
        match self.config.decay_type {
            DecayType::Exponential => {
                initial * (-elapsed * std::f64::consts::LN_2 / self.config.half_life_secs).exp()
            }
            DecayType::Linear => {
                (initial * (1.0 - elapsed / (self.config.half_life_secs * 2.0))).max(0.0)
            }
            DecayType::PowerLaw => {
                initial / (1.0 + elapsed / self.config.half_life_secs).powf(2.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exponential_decay() {
        let mut tracker = AlphaDecayTracker::new(AlphaDecayConfig::default());
        tracker.add_signal(1.0, "test");
        
        std::thread::sleep(Duration::from_millis(100));
        let alpha = tracker.get_decayed_alpha();
        assert!(alpha < 1.0 && alpha > 0.0);
    }
}
