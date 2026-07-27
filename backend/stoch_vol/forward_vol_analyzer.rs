//! Forward Volatility Analyzer
//! Calculates forward implied volatility for term structure trades
//! Essential for calendar spreads and vol curve positioning
//! 
//! Forward vol: σ_fwd²(T1,T2) = (σ₂²T₂ - σ₁²T₁) / (T₂ - T₁)

use std::collections::HashMap;

/// Forward volatility calculation result
#[derive(Debug, Clone)]
pub struct ForwardVolResult {
    pub forward_vol: f64,
    pub start_time: f64,
    pub end_time: f64,
    pub strike: f64,
    pub confidence: f64,  // Based on liquidity and bid-ask spread
}

/// Term structure point
#[derive(Debug, Clone)]
pub struct TermStructurePoint {
    pub expiry: f64,
    pub implied_vol: f64,
    pub total_variance: f64,
}

/// Forward volatility analyzer for term structure trading
pub struct ForwardVolAnalyzer {
    term_structure: Vec<TermStructurePoint>,
    strike: f64,
}

impl ForwardVolAnalyzer {
    pub fn new(strike: f64) -> Self {
        Self {
            term_structure: Vec::new(),
            strike,
        }
    }

    /// Add a point to the term structure
    pub fn add_point(&mut self, expiry: f64, implied_vol: f64) {
        let total_var = implied_vol * implied_vol * expiry;
        self.term_structure.push(TermStructurePoint {
            expiry,
            implied_vol,
            total_variance: total_var,
        });
        
        // Keep sorted by expiry
        self.term_structure.sort_by(|a, b| a.expiry.partial_cmp(&b.expiry).unwrap());
    }

    /// Calculate forward volatility between two expiries
    pub fn calculate_forward_vol(&self, t1: f64, t2: f64) -> Option<ForwardVolResult> {
        if t1 >= t2 || t1 < 0.0 {
            return None;
        }

        // Find closest points in term structure
        let var1 = self.get_total_variance(t1)?;
        let var2 = self.get_total_variance(t2)?;

        // Forward variance formula
        let fwd_var = (var2 - var1) / (t2 - t1);

        if fwd_var <= 0.0 {
            // Negative forward variance indicates arbitrage or data issue
            return None;
        }

        let fwd_vol = fwd_var.sqrt();
        
        // Confidence based on distance to actual quoted points
        let confidence = self.calculate_confidence(t1, t2);

        Some(ForwardVolResult {
            forward_vol: fwd_vol,
            start_time: t1,
            end_time: t2,
            strike: self.strike,
            confidence,
        })
    }

    /// Get total variance at a given time via interpolation
    fn get_total_variance(&self, t: f64) -> Option<f64> {
        if self.term_structure.is_empty() {
            return None;
        }

        // Find bracketing points
        let mut lower: Option<&TermStructurePoint> = None;
        let mut upper: Option<&TermStructurePoint> = None;

        for point in &self.term_structure {
            if point.expiry <= t {
                lower = Some(point);
            } else if upper.is_none() {
                upper = Some(point);
                break;
            }
        }

        match (lower, upper) {
            (None, Some(u)) => Some(u.total_variance),  // Extrapolate flat
            (Some(l), None) => Some(l.total_variance),  // Extrapolate flat
            (Some(l), Some(u)) => {
                // Linear interpolation of total variance
                if u.expiry - l.expiry < 1e-10 {
                    Some(l.total_variance)
                } else {
                    let weight = (t - l.expiry) / (u.expiry - l.expiry);
                    Some(l.total_variance + weight * (u.total_variance - l.total_variance))
                }
            }
            (None, None) => None,
        }
    }

    /// Calculate confidence score based on interpolation distance
    fn calculate_confidence(&self, t1: f64, t2: f64) -> f64 {
        let mut max_distance = 0.0;

        for &t in &[t1, t2] {
            let mut min_dist = f64::MAX;
            for point in &self.term_structure {
                let dist = (point.expiry - t).abs();
                min_dist = min_dist.min(dist);
            }
            max_distance = max_distance.max(min_dist);
        }

        // Confidence decays with distance from quoted points
        // 1.0 at exact points, ~0.5 at 1 week away, ~0.1 at 1 month away
        (1.0 / (1.0 + max_distance * 30.0)).max(0.1)
    }

    /// Get the full forward volatility curve
    pub fn get_forward_curve(&self, start: f64) -> Vec<ForwardVolResult> {
        let mut curve = Vec::new();

        if self.term_structure.is_empty() {
            return curve;
        }

        // Calculate forwards from start to each future expiry
        for point in &self.term_structure {
            if point.expiry > start {
                if let Some(result) = self.calculate_forward_vol(start, point.expiry) {
                    curve.push(result);
                }
            }
        }

        curve
    }

    /// Detect contango/backwardation in forward vol term structure
    pub fn detect_term_structure_regime(&self) -> TermStructureRegime {
        if self.term_structure.len() < 2 {
            return TermStructureRegime::Unknown;
        }

        // Compare short-end vs long-end forward vols
        let short_t = self.term_structure[0].expiry;
        let long_t = self.term_structure.last().unwrap().expiry;
        let mid_t = (short_t + long_t) / 2.0;

        let fwd_short = self.calculate_forward_vol(short_t, mid_t);
        let fwd_long = self.calculate_forward_vol(mid_t, long_t);

        match (fwd_short, fwd_long) {
            (Some(s), Some(l)) => {
                let diff = l.forward_vol - s.forward_vol;
                if diff > 0.05 {
                    TermStructureRegime::Contango  // Upward sloping
                } else if diff < -0.05 {
                    TermStructureRegime::Backwardation  // Downward sloping
                } else {
                    TermStructureRegime::Flat
                }
            }
            _ => TermStructureRegime::Unknown,
        }
    }

    /// Calculate fair value for a variance swap
    pub fn variance_swap_fair_value(&self, t_start: f64, t_end: f64) -> Option<f64> {
        let fwd_result = self.calculate_forward_vol(t_start, t_end)?;
        
        // Variance swap strikes are typically quoted as vol^2
        Some(fwd_result.forward_vol * fwd_result.forward_vol)
    }

    /// Get calendar spread recommendation
    pub fn calendar_spread_signal(&self) -> Option<CalendarSignal> {
        let regime = self.detect_term_structure_regime();
        
        if self.term_structure.len() < 2 {
            return None;
        }

        let short_iv = self.term_structure[0].implied_vol;
        let long_iv = self.term_structure.last().unwrap().implied_vol;

        // Simple signal based on steepness
        let steepness = (long_iv - short_iv) / short_iv;

        Some(CalendarSignal {
            regime,
            steepness,
            recommendation: if steepness > 0.2 {
                CalendarRecommendation::LongCalendar  // Buy spread
            } else if steepness < -0.1 {
                CalendarRecommendation::ShortCalendar  // Sell spread
            } else {
                CalendarRecommendation::Neutral
            },
        })
    }
}

/// Term structure regime classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TermStructureRegime {
    Contango,       // Upward sloping (normal)
    Backwardation,  // Downward sloping (stress)
    Flat,
    Unknown,
}

/// Calendar spread signal
#[derive(Debug, Clone)]
pub struct CalendarSignal {
    pub regime: TermStructureRegime,
    pub steepness: f64,
    pub recommendation: CalendarRecommendation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalendarRecommendation {
    LongCalendar,   // Long far, short near
    ShortCalendar,  // Short far, long near
    Neutral,
}

/// Builder for constructing forward vol surfaces across strikes
pub struct ForwardVolSurfaceBuilder {
    data: HashMap<f64, ForwardVolAnalyzer>,  // strike -> analyzer
}

impl ForwardVolSurfaceBuilder {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Add implied vol data point
    pub fn add_point(&mut self, strike: f64, expiry: f64, iv: f64) {
        let analyzer = self.data.entry(strike).or_insert_with(|| ForwardVolAnalyzer::new(strike));
        analyzer.add_point(expiry, iv);
    }

    /// Get forward vol for specific strike and date range
    pub fn get_forward(&self, strike: f64, t1: f64, t2: f64) -> Option<f64> {
        self.data.get(&strike)?.calculate_forward_vol(t1, t2).map(|r| r.forward_vol)
    }

    /// Build complete forward surface
    pub fn build_surface(&self, t1: f64, t2: f64) -> Vec<(f64, f64)> {
        let mut surface = Vec::new();
        
        for (&strike, analyzer) in &self.data {
            if let Some(result) = analyzer.calculate_forward_vol(t1, t2) {
                surface.push((strike, result.forward_vol));
            }
        }
        
        surface.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        surface
    }
}

impl Default for ForwardVolSurfaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_vol_calculation() {
        let mut analyzer = ForwardVolAnalyzer::new(100.0);
        
        // Add term structure: increasing vol with time (contango)
        analyzer.add_point(0.25, 0.5);  // 3m: 50%
        analyzer.add_point(0.5, 0.55);  // 6m: 55%
        analyzer.add_point(1.0, 0.6);   // 1y: 60%

        let fwd = analyzer.calculate_forward_vol(0.25, 0.5).unwrap();
        
        // Forward vol should be higher than spot vols in contango
        assert!(fwd.forward_vol > 0.55);
        assert!(fwd.confidence > 0.0);
    }

    #[test]
    fn test_regime_detection() {
        let mut analyzer = ForwardVolAnalyzer::new(100.0);
        
        // Contango structure
        analyzer.add_point(0.25, 0.5);
        analyzer.add_point(1.0, 0.7);
        
        assert_eq!(analyzer.detect_term_structure_regime(), TermStructureRegime::Contango);
    }

    #[test]
    fn test_variance_swap_pv() {
        let mut analyzer = ForwardVolAnalyzer::new(100.0);
        analyzer.add_point(0.25, 0.5);
        analyzer.add_point(0.5, 0.55);
        
        let fair_var = analyzer.variance_swap_fair_value(0.25, 0.5);
        assert!(fair_var.is_some());
        assert!(fair_var.unwrap() > 0.0);
    }
}
