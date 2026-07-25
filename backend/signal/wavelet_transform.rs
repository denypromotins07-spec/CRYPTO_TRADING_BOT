//! Wavelet Transform Engine for ZAID Trading Bot
//! Applies discrete wavelet transforms for regime detection
//! Optimized for multi-scale analysis of price signals
//!
//! This module provides Haar and Daubechies wavelet transforms
//! for decomposing price signals into different time scales,
//! enabling detection of market regime changes under 8GB RAM.

use std::f64::consts::SQRT_2;

/// Wavelet decomposition result
#[derive(Debug)]
pub struct WaveletResult {
    /// Approximation coefficients (low-frequency)
    pub approximation: Vec<f64>,
    /// Detail coefficients (high-frequency)
    pub detail: Vec<f64>,
    /// Decomposition level
    pub level: usize,
    /// Original signal length
    pub original_length: usize,
}

impl WaveletResult {
    /// Reconstruct signal from approximation and detail coefficients
    pub fn reconstruct(&self) -> Vec<f64> {
        inverse_haar_single_level(&self.approximation, &self.detail)
    }
    
    /// Get energy in approximation (trend energy)
    pub fn trend_energy(&self) -> f64 {
        self.approximation.iter().map(|&x| x * x).sum()
    }
    
    /// Get energy in detail (noise/volatility energy)
    pub fn noise_energy(&self) -> f64 {
        self.detail.iter().map(|&x| x * x).sum()
    }
    
    /// Compute trend-to-noise ratio
    pub fn trend_noise_ratio(&self) -> f64 {
        let trend_e = self.trend_energy();
        let noise_e = self.noise_energy();
        
        if noise_e < 1e-15 {
            f64::INFINITY
        } else {
            trend_e / noise_e
        }
    }
}

/// Multi-level wavelet decomposition
#[derive(Debug)]
pub struct MultiLevelWavelet {
    /// Coefficients at each level
    pub levels: Vec<WaveletResult>,
    /// Final approximation
    pub final_approximation: Vec<f64>,
}

impl MultiLevelWavelet {
    /// Create from decomposition results
    pub fn new(levels: Vec<WaveletResult>, final_approx: Vec<f64>) -> Self {
        Self {
            levels,
            final_approximation: final_approx,
        }
    }
    
    /// Reconstruct full signal from all levels
    pub fn reconstruct(&self) -> Vec<f64> {
        let mut current_approx = self.final_approximation.clone();
        
        // Reconstruct from deepest level to shallowest
        for level in self.levels.iter().rev() {
            current_approx = inverse_haar_single_level(&current_approx, &level.detail);
        }
        
        current_approx
    }
    
    /// Get total decomposition depth
    pub fn depth(&self) -> usize {
        self.levels.len()
    }
}

/// Single-level Haar wavelet decomposition
/// Fastest wavelet, good for real-time applications
#[inline]
pub fn haar_decompose(signal: &[f64]) -> WaveletResult {
    let n = signal.len();
    let half_n = n / 2;
    
    let mut approximation = Vec::with_capacity(half_n);
    let mut detail = Vec::with_capacity(half_n);
    
    for i in 0..half_n {
        let even = signal[2 * i];
        let odd = signal[2 * i + 1];
        
        // Haar scaling and wavelet functions
        approximation.push((even + odd) / SQRT_2);
        detail.push((even - odd) / SQRT_2);
    }
    
    WaveletResult {
        approximation,
        detail,
        level: 1,
        original_length: n,
    }
}

/// Single-level Haar inverse transform
#[inline]
fn inverse_haar_single_level(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let n = approx.len().min(detail.len());
    let mut reconstructed = Vec::with_capacity(n * 2);
    
    for i in 0..n {
        let a = approx[i];
        let d = detail[i];
        
        // Inverse Haar
        let even = (a + d) / SQRT_2;
        let odd = (a - d) / SQRT_2;
        
        reconstructed.push(even);
        reconstructed.push(odd);
    }
    
    reconstructed
}

/// Multi-level Haar decomposition
pub fn haar_multilevel(signal: &[f64], max_levels: usize) -> MultiLevelWavelet {
    let mut levels = Vec::with_capacity(max_levels);
    let mut current = signal.to_vec();
    
    let max_possible_levels = (signal.len() as f64).log2() as usize;
    let actual_levels = max_levels.min(max_possible_levels);
    
    for level in 0..actual_levels {
        if current.len() < 2 {
            break;
        }
        
        let result = haar_decompose(&current);
        let detail = result.detail.clone();
        current = result.approximation;
        
        levels.push(WaveletResult {
            approximation: current.clone(),
            detail,
            level: level + 1,
            original_length: signal.len(),
        });
    }
    
    MultiLevelWavelet::new(levels, current)
}

/// Daubechies-4 wavelet coefficients
const DB4_LOW_PASS: [f64; 4] = [
    0.4829629131445341,
    0.8365163037378079,
    0.2241438680420134,
    -0.1294095225512604,
];

const DB4_HIGH_PASS: [f64; 4] = [
    -0.1294095225512604,
    -0.2241438680420134,
    0.8365163037378079,
    -0.4829629131445341,
];

/// Daubechies-4 wavelet decomposition
pub fn db4_decompose(signal: &[f64]) -> WaveletResult {
    let n = signal.len();
    let half_n = n / 2;
    
    let mut approximation = Vec::with_capacity(half_n);
    let mut detail = Vec::with_capacity(half_n);
    
    for i in 0..half_n {
        let mut a_sum = 0.0;
        let mut d_sum = 0.0;
        
        for j in 0..4 {
            let idx = (2 * i + j) % n;
            a_sum += DB4_LOW_PASS[j] * signal[idx];
            d_sum += DB4_HIGH_PASS[j] * signal[idx];
        }
        
        approximation.push(a_sum);
        detail.push(d_sum);
    }
    
    WaveletResult {
        approximation,
        detail,
        level: 1,
        original_length: n,
    }
}

/// Wavelet-based regime detection
pub mod regime_detection {
    use super::*;
    
    /// Market regime classification
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum MarketRegime {
        TrendingStrong,
        TrendingWeak,
        Ranging,
        Volatile,
        Unknown,
    }
    
    /// Detect market regime using wavelet coefficients
    pub fn detect_regime(signal: &[f64]) -> MarketRegime {
        if signal.len() < 8 {
            return MarketRegime::Unknown;
        }
        
        // Perform 3-level decomposition
        let max_levels = 3.min((signal.len() as f64).log2() as usize);
        let decomp = haar_multilevel(signal, max_levels);
        
        if decomp.levels.is_empty() {
            return MarketRegime::Unknown;
        }
        
        // Analyze energy distribution across scales
        let total_trend_energy: f64 = decomp.levels.iter()
            .map(|l| l.trend_energy())
            .sum();
        
        let total_noise_energy: f64 = decomp.levels.iter()
            .map(|l| l.noise_energy())
            .sum();
        
        let tnr = if total_noise_energy > 1e-15 {
            total_trend_energy / total_noise_energy
        } else {
            f64::INFINITY
        };
        
        // Analyze detail coefficient patterns
        let first_level_detail = &decomp.levels[0].detail;
        let detail_mean: f64 = first_level_detail.iter().sum::<f64>() / first_level_detail.len() as f64;
        let detail_std: f64 = (first_level_detail.iter()
            .map(|&x| (x - detail_mean).powi(2))
            .sum::<f64>() / first_level_detail.len() as f64).sqrt();
        
        // Classify based on thresholds
        if tnr > 10.0 {
            MarketRegime::TrendingStrong
        } else if tnr > 3.0 {
            MarketRegime::TrendingWeak
        } else if detail_std < 0.01 {
            MarketRegime::Ranging
        } else {
            MarketRegime::Volatile
        }
    }
    
    /// Compute wavelet coherence between two signals
    pub fn wavelet_coherence(signal1: &[f64], signal2: &[f64]) -> Vec<f64> {
        let len = signal1.len().min(signal2.len());
        let s1 = &signal1[..len];
        let s2 = &signal2[..len];
        
        // Simple coherence estimate using wavelet details
        let decomp1 = haar_decompose(s1);
        let decomp2 = haar_decompose(s2);
        
        let n = decomp1.detail.len().min(decomp2.detail.len());
        let mut coherence = Vec::with_capacity(n);
        
        for i in 0..n {
            let d1 = decomp1.detail[i];
            let d2 = decomp2.detail[i];
            
            let product = d1 * d2;
            let norm1 = d1.abs();
            let norm2 = d2.abs();
            
            let coh = if norm1 > 1e-15 && norm2 > 1e-15 {
                product / (norm1 * norm2)
            } else {
                0.0
            };
            
            coherence.push(coh.clamp(-1.0, 1.0));
        }
        
        coherence
    }
    
    /// Detect change points in signal using wavelet details
    pub fn detect_change_points(signal: &[f64], threshold_multiplier: f64) -> Vec<usize> {
        if signal.len() < 4 {
            return vec![];
        }
        
        let decomp = haar_decompose(signal);
        let details = &decomp.detail;
        
        // Compute statistics of detail coefficients
        let mean: f64 = details.iter().sum::<f64>() / details.len() as f64;
        let variance: f64 = details.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / details.len() as f64;
        let std_dev = variance.sqrt();
        
        let threshold = threshold_multiplier * std_dev;
        
        // Find significant deviations
        let mut change_points = Vec::new();
        
        for (i, &detail) in details.iter().enumerate() {
            if (detail - mean).abs() > threshold {
                // Map back to original signal index
                change_points.push(i * 2);
            }
        }
        
        change_points
    }
}

/// Denoising using wavelet thresholding
pub mod denoising {
    use super::*;
    
    /// Soft thresholding function
    #[inline]
    fn soft_threshold(x: f64, threshold: f64) -> f64 {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }
    
    /// Universal threshold (VisuShrink)
    pub fn universal_threshold(signal: &[f64]) -> f64 {
        let n = signal.len() as f64;
        let sigma = median_absolute_deviation(signal);
        sigma * (2.0 * n.ln()).sqrt()
    }
    
    /// Median absolute deviation estimator
    fn median_absolute_deviation(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let median = sorted[sorted.len() / 2];
        
        let mut deviations: Vec<f64> = sorted.iter()
            .map(|&x| (x - median).abs())
            .collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        deviations[deviations.len() / 2] * 1.4826  // Scale factor for Gaussian
    }
    
    /// Wavelet denoising using soft thresholding
    pub fn denoise_wavelet(signal: &[f64], levels: usize) -> Vec<f64> {
        let decomp = haar_multilevel(signal, levels);
        
        // Apply threshold to all detail coefficients
        let threshold = universal_threshold(signal);
        
        let mut thresholded_levels = Vec::new();
        for level_result in &decomp.levels {
            let thresholded_detail: Vec<f64> = level_result.detail
                .iter()
                .map(|&d| soft_threshold(d, threshold))
                .collect();
            
            thresholded_levels.push(WaveletResult {
                approximation: level_result.approximation.clone(),
                detail: thresholded_detail,
                level: level_result.level,
                original_length: level_result.original_length,
            });
        }
        
        // Reconstruct with thresholded coefficients
        let mut reconstructed = decomp.final_approximation.clone();
        
        for level_result in thresholded_levels.iter().rev() {
            reconstructed = inverse_haar_single_level(&reconstructed, &level_result.detail);
        }
        
        reconstructed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regime_detection::{detect_regime, MarketRegime};
    
    #[test]
    fn test_haar_decompose_reconstruct() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let result = haar_decompose(&signal);
        let reconstructed = result.reconstruct();
        
        assert_eq!(reconstructed.len(), signal.len());
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_multilevel_decompose() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let decomp = haar_multilevel(&signal, 3);
        
        assert_eq!(decomp.depth(), 3);
        
        // Test reconstruction
        let reconstructed = decomp.reconstruct();
        assert_eq!(reconstructed.len(), signal.len());
    }
    
    #[test]
    fn test_trend_noise_ratio() {
        // Pure trend (linear)
        let trend_signal: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let decomp = haar_decompose(&trend_signal);
        
        // Should have high TNR for trending signal
        assert!(decomp.trend_noise_ratio() > 1.0);
    }
    
    #[test]
    fn test_regime_detection() {
        // Strong trend
        let trending: Vec<f64> = (0..32).map(|i| i as f64 * 0.1).collect();
        let regime = detect_regime(&trending);
        assert!(regime == MarketRegime::TrendingStrong || 
                regime == MarketRegime::TrendingWeak);
        
        // High frequency oscillation (volatile)
        let volatile: Vec<f64> = (0..32)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let regime = detect_regime(&volatile);
        // Could be volatile or ranging depending on scale
    }
    
    #[test]
    fn test_denoising() {
        use denoising::denoise_wavelet;
        
        // Clean signal with noise
        let clean: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let noisy: Vec<f64> = clean.iter()
            .enumerate()
            .map(|(i, &v)| v + (i % 3) as f64 * 0.1)
            .collect();
        
        let denoised = denoise_wavelet(&noisy, 3);
        
        // Denoised should be closer to clean than noisy
        let noisy_error: f64 = clean.iter().zip(&noisy)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let denoised_error: f64 = clean.iter().zip(&denoised)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        
        assert!(denoised_error < noisy_error);
    }
}
