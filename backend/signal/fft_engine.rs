//! FFT Engine for ZAID Trading Bot
//! Computes real-time Fast Fourier Transforms on tick data
//! Optimized for frequency analysis and regime detection
//!
//! This module provides high-performance FFT operations for
//! isolating high-frequency noise from low-frequency trends
//! in cryptocurrency price data under the 8GB RAM constraint.

use std::f64::consts::PI;

/// Complex number representation for FFT operations
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    
    #[inline]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
    
    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
    
    #[inline]
    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    
    #[inline]
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }
    
    #[inline]
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    
    #[inline]
    fn mul(self, scalar: f64) -> Self {
        Self {
            re: self.re * scalar,
            im: self.im * scalar,
        }
    }
}

/// FFT result container
#[derive(Debug)]
pub struct FftResult {
    /// Frequency domain coefficients
    pub spectrum: Vec<Complex>,
    /// Corresponding frequencies in Hz
    pub frequencies: Vec<f64>,
    /// Sample rate used
    pub sample_rate: f64,
    /// Number of samples processed
    pub n_samples: usize,
}

impl FftResult {
    /// Get power spectrum (magnitude squared)
    pub fn power_spectrum(&self) -> Vec<f64> {
        self.spectrum.iter().map(|c| c.magnitude_squared()).collect()
    }
    
    /// Get amplitude spectrum
    pub fn amplitude_spectrum(&self) -> Vec<f64> {
        let n = self.n_samples as f64;
        self.spectrum.iter().map(|c| c.magnitude() / n).collect()
    }
    
    /// Find dominant frequency component
    pub fn dominant_frequency(&self) -> Option<(f64, f64)> {
        if self.spectrum.is_empty() {
            return None;
        }
        
        // Skip DC component (index 0)
        let mut max_power = 0.0;
        let mut max_freq = 0.0;
        
        for i in 1..(self.spectrum.len() / 2) {
            let power = self.spectrum[i].magnitude_squared();
            if power > max_power {
                max_power = power;
                max_freq = self.frequencies[i];
            }
        }
        
        Some((max_freq, max_power))
    }
}

/// Cooley-Tukey FFT implementation
pub struct FftEngine {
    /// Precomputed twiddle factors
    twiddle_factors: Vec<Complex>,
    /// Bit-reversal lookup table
    bit_reverse_table: Vec<usize>,
    /// Current size
    size: usize,
}

impl FftEngine {
    /// Create new FFT engine for specified size (must be power of 2)
    pub fn new(size: usize) -> Option<Self> {
        if !size.is_power_of_two() || size < 2 {
            return None;
        }
        
        let log_n = size.trailing_zeros() as usize;
        let mut engine = Self {
            twiddle_factors: Vec::with_capacity(size),
            bit_reverse_table: Vec::with_capacity(size),
            size,
        };
        
        engine.precompute_twiddles();
        engine.compute_bit_reverse_table(log_n);
        
        Some(engine)
    }
    
    fn precompute_twiddles(&mut self) {
        let n = self.size;
        self.twiddle_factors.reserve(n);
        
        for k in 0..n {
            let angle = -2.0 * PI * (k as f64) / (n as f64);
            self.twiddle_factors.push(Complex::new(angle.cos(), angle.sin()));
        }
    }
    
    fn compute_bit_reverse_table(&mut self, log_n: usize) {
        let n = self.size;
        self.bit_reverse_table.resize(n, 0);
        
        for i in 0..n {
            self.bit_reverse_table[i] = reverse_bits(i, log_n);
        }
    }
    
    /// Compute forward FFT on real-valued input
    pub fn fft_real(&self, input: &[f64]) -> FftResult {
        assert_eq!(input.len(), self.size);
        
        // Convert to complex with zero imaginary part
        let mut complex_input: Vec<Complex> = input
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // Perform FFT in-place
        self.fft_inplace(&mut complex_input);
        
        // Compute frequency bins
        let dt = 1.0; // Assume unit time step, can be scaled later
        let df = 1.0 / (self.size as f64 * dt);
        let frequencies: Vec<f64> = (0..self.size)
            .map(|i| {
                if i <= self.size / 2 {
                    i as f64 * df
                } else {
                    (i - self.size) as f64 * df
                }
            })
            .collect();
        
        FftResult {
            spectrum: complex_input,
            frequencies,
            sample_rate: 1.0 / dt,
            n_samples: self.size,
        }
    }
    
    /// In-place Cooley-Tukey FFT
    fn fft_inplace(&self, data: &mut [Complex]) {
        let n = self.size;
        
        // Bit-reversal permutation
        for i in 0..n {
            let j = self.bit_reverse_table[i];
            if i < j {
                data.swap(i, j);
            }
        }
        
        // Butterfly operations
        let mut size = 2;
        while size <= n {
            let half_size = size / 2;
            let step = n / size;
            
            for start in (0..n).step_by(size) {
                for k in 0..half_size {
                    let twiddle = self.twiddle_factors[k * step];
                    let even_idx = start + k;
                    let odd_idx = start + k + half_size;
                    
                    let temp = data[odd_idx] * twiddle;
                    let even = data[even_idx];
                    
                    data[even_idx] = even + temp;
                    data[odd_idx] = even - temp;
                }
            }
            
            size *= 2;
        }
    }
    
    /// Compute inverse FFT
    pub fn ifft(&self, spectrum: &[Complex]) -> Vec<f64> {
        assert_eq!(spectrum.len(), self.size);
        
        // Conjugate input
        let mut data: Vec<Complex> = spectrum.iter().map(|c| c.conj()).collect();
        
        // Forward FFT
        self.fft_inplace(&mut data);
        
        // Conjugate and scale
        let scale = 1.0 / self.size as f64;
        data.iter()
            .map(|c| c.re * scale)
            .collect()
    }
}

/// Helper function to reverse bits
fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Spectral analysis utilities for trading signals
pub mod spectral_analysis {
    use super::*;
    
    /// Apply Hann window to reduce spectral leakage
    pub fn hann_window(size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos())
            })
            .collect()
    }
    
    /// Apply Hamming window
    pub fn hamming_window(size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                0.54 - 0.46 * (2.0 * PI * i as f64 / (size - 1) as f64).cos()
            })
            .collect()
    }
    
    /// Compute spectrogram using sliding window FFT
    pub fn spectrogram(data: &[f64], 
                       window_size: usize,
                       hop_size: usize) -> Vec<FftResult> {
        assert!(window_size.is_power_of_two());
        
        let fft_engine = FftEngine::new(window_size).unwrap();
        let window = hann_window(window_size);
        
        let mut results = Vec::new();
        let mut start = 0;
        
        while start + window_size <= data.len() {
            // Apply window
            let mut windowed: Vec<f64> = Vec::with_capacity(window_size);
            for i in 0..window_size {
                windowed.push(data[start + i] * window[i]);
            }
            
            // Compute FFT
            let result = fft_engine.fft_real(&windowed);
            results.push(result);
            
            start += hop_size;
        }
        
        results
    }
    
    /// Filter signal to keep only frequencies in specified range
    pub fn bandpass_filter(signal: &[f64],
                           low_freq: f64,
                           high_freq: f64,
                           sample_rate: f64) -> Vec<f64> {
        let n = signal.len().next_power_of_two();
        let fft_engine = FftEngine::new(n).unwrap();
        
        // Pad signal if necessary
        let mut padded = signal.to_vec();
        while padded.len() < n {
            padded.push(0.0);
        }
        
        // FFT
        let mut spectrum = fft_engine.fft_real(&padded);
        
        // Zero out frequencies outside band
        let df = sample_rate / n as f64;
        for i in 0..n {
            let freq = spectrum.frequencies[i].abs();
            if freq < low_freq || freq > high_freq {
                spectrum.spectrum[i] = Complex::zero();
            }
        }
        
        // IFFT
        fft_engine.ifft(&spectrum.spectrum)
    }
    
    /// Detect market regime using spectral centroid
    /// High centroid = trending, Low centroid = mean-reverting
    pub fn spectral_centroid(result: &FftResult) -> f64 {
        let powers = result.power_spectrum();
        let n = powers.len() / 2; // Only use positive frequencies
        
        let mut weighted_sum = 0.0;
        let mut total_power = 0.0;
        
        for i in 1..n {
            let power = powers[i];
            weighted_sum += result.frequencies[i] * power;
            total_power += power;
        }
        
        if total_power < 1e-15 {
            0.0
        } else {
            weighted_sum / total_power
        }
    }
    
    /// Compute spectral entropy as a measure of signal complexity
    pub fn spectral_entropy(result: &FftResult) -> f64 {
        let powers = result.power_spectrum();
        let n = powers.len() / 2;
        
        // Normalize to probability distribution
        let total: f64 = powers[1..n].iter().sum();
        
        if total < 1e-15 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &power in &powers[1..n] {
            let p = power / total;
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        
        entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fft_sine_wave() {
        let n = 64;
        let freq = 4.0; // 4 cycles in n samples
        
        // Generate sine wave
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())
            .collect();
        
        let fft = FftEngine::new(n).unwrap();
        let result = fft.fft_real(&signal);
        
        // Should have peak at frequency bin 4
        let (dom_freq, dom_power) = result.dominant_frequency().unwrap();
        
        let expected_bin = freq;
        assert!((dom_freq - expected_bin).abs() < 0.1);
        assert!(dom_power > 100.0); // Strong signal
    }
    
    #[test]
    fn test_ifft_roundtrip() {
        let n = 32;
        let signal: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        
        let fft = FftEngine::new(n).unwrap();
        let spectrum = fft.fft_real(&signal);
        let reconstructed = fft.ifft(&spectrum.spectrum);
        
        // Check reconstruction matches original
        for i in 0..n {
            assert!((reconstructed[i] - signal[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        
        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 2.0).abs() < 1e-10);
        
        let product = a * b;
        assert!((product.re - 11.0).abs() < 1e-10);
        assert!((product.im - (-2.0)).abs() < 1e-10);
        
        assert!((a.magnitude() - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_spectral_centroid() {
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 8.0 * i as f64 / n as f64).sin())
            .collect();
        
        let fft = FftEngine::new(n).unwrap();
        let result = fft.fft_real(&signal);
        
        let centroid = spectral_analysis::spectral_centroid(&result);
        
        // Centroid should be near the dominant frequency
        assert!(centroid > 5.0 && centroid < 15.0);
    }
}
