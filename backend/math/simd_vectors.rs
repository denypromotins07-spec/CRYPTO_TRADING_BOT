//! SIMD Vectorization Module for ZAID Trading Bot
//! Implements AVX2 intrinsics for parallel price mathematics
//! Optimized for AMD Ryzen AI 5 with 8GB RAM constraint
//! 
//! This module provides zero-cost abstractions for vectorized operations
//! on price ticks, order book levels, and statistical computations.

use std::arch::x86_64::*;
use std::mem;

/// Number of f64 values that fit in a 256-bit AVX2 register
const LANE_COUNT_F64: usize = 4;
/// Number of f32 values that fit in a 256-bit AVX2 register
const LANE_COUNT_F32: usize = 8;

/// Padded vector structure ensuring 32-byte alignment for AVX2 operations
#[repr(align(32))]
#[derive(Debug, Clone)]
pub struct AlignedVecF64 {
    pub data: Vec<f64>,
    pub len: usize,
    pub padded_len: usize,
}

impl AlignedVecF64 {
    /// Creates a new aligned vector with proper padding for AVX2
    pub fn new(data: &[f64]) -> Self {
        let len = data.len();
        let remainder = len % LANE_COUNT_F64;
        let padded_len = if remainder == 0 { len } else { len + LANE_COUNT_F64 - remainder };
        
        let mut padded_data = Vec::with_capacity(padded_len);
        padded_data.extend_from_slice(data);
        
        // Pad with zeros to ensure full lanes for SIMD operations
        for _ in remainder..LANE_COUNT_F64 {
            if padded_data.len() < padded_len {
                padded_data.push(0.0);
            }
        }
        
        Self {
            data: padded_data,
            len,
            padded_len,
        }
    }
    
    /// Returns the underlying slice without padding
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data[..self.len]
    }
}

/// SIMD-enabled price scaling operation
/// Multiplies all prices by a scalar factor using AVX2 registers
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_scale_f64_avx2(data: &mut [f64], scalar: f64) {
    let broadcast = _mm256_set1_pd(scalar);
    
    let chunks = data.chunks_exact_mut(LANE_COUNT_F64);
    for chunk in chunks {
        let ptr = chunk.as_mut_ptr() as *const __m256d;
        let vec = _mm256_loadu_pd(ptr);
        let result = _mm256_mul_pd(vec, broadcast);
        _mm256_storeu_pd(chunk.as_mut_ptr(), result);
    }
    
    // Handle remaining elements
    let remainder = data.len() % LANE_COUNT_F64;
    let start = data.len() - remainder;
    for i in start..data.len() {
        data[i] *= scalar;
    }
}

/// Public interface for SIMD price scaling
/// Automatically detects CPU features and falls back to scalar ops
pub fn scale_prices(prices: &mut [f64], scalar: f64) {
    if is_x86_feature_detected!("avx2") {
        unsafe {
            simd_scale_f64_avx2(prices, scalar);
        }
    } else {
        // Fallback to scalar multiplication
        for price in prices.iter_mut() {
            *price *= scalar;
        }
    }
}

/// SIMD-enabled element-wise addition of two price arrays
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_add_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    
    let chunks_a = a.chunks_exact(LANE_COUNT_F64);
    let chunks_b = b.chunks_exact(LANE_COUNT_F64);
    let chunks_r = result.chunks_exact_mut(LANE_COUNT_F64);
    
    for (chunk_a, (chunk_b, chunk_r)) in chunks_a.zip(chunks_b.zip(chunks_r)) {
        let vec_a = _mm256_loadu_pd(chunk_a.as_ptr());
        let vec_b = _mm256_loadu_pd(chunk_b.as_ptr());
        let sum = _mm256_add_pd(vec_a, vec_b);
        _mm256_storeu_pd(chunk_r.as_mut_ptr(), sum);
    }
    
    // Handle remainder
    let start = result.len() - (result.len() % LANE_COUNT_F64);
    for i in start..result.len() {
        result[i] = a[i] + b[i];
    }
}

/// Public interface for SIMD array addition
pub fn add_price_arrays(a: &[f64], b: &[f64], result: &mut [f64]) {
    if a.len() != b.len() || a.len() != result.len() {
        panic!("Array length mismatch in SIMD addition");
    }
    
    if is_x86_feature_detected!("avx2") && a.len() >= LANE_COUNT_F64 {
        unsafe {
            simd_add_f64_avx2(a, b, result);
        }
    } else {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
}

/// SIMD-enabled dot product for correlation calculations
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_dot_product_avx2(a: &[f64], b: &[f64]) -> f64 {
    let mut sum_vec = _mm256_setzero_pd();
    
    let chunks_a = a.chunks_exact(LANE_COUNT_F64);
    let chunks_b = b.chunks_exact(LANE_COUNT_F64);
    
    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
        let vec_a = _mm256_loadu_pd(chunk_a.as_ptr());
        let vec_b = _mm256_loadu_pd(chunk_b.as_ptr());
        let prod = _mm256_mul_pd(vec_a, vec_b);
        sum_vec = _mm256_add_pd(sum_vec, prod);
    }
    
    // Horizontal sum of the vector
    let mut result = [0.0; LANE_COUNT_F64];
    _mm256_storeu_pd(result.as_mut_ptr(), sum_vec);
    let mut sum = result[0] + result[1] + result[2] + result[3];
    
    // Add remainder
    let start = a.len() - (a.len() % LANE_COUNT_F64);
    for i in start..a.len() {
        sum += a[i] * b[i];
    }
    
    sum
}

/// Public interface for SIMD dot product
pub fn compute_dot_product(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        panic!("Array length mismatch in dot product");
    }
    
    if is_x86_feature_detected!("avx2") && a.len() >= LANE_COUNT_F64 {
        unsafe {
            simd_dot_product_avx2(a, b)
        }
    } else {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// SIMD-enabled min/max finding for order book analysis
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_min_max_avx2(data: &[f64]) -> (f64, f64) {
    let mut min_vec = _mm256_set1_pd(f64::INFINITY);
    let mut max_vec = _mm256_set1_pd(f64::NEG_INFINITY);
    
    let chunks = data.chunks_exact(LANE_COUNT_F64);
    for chunk in chunks {
        let vec = _mm256_loadu_pd(chunk.as_ptr());
        min_vec = _mm256_min_pd(min_vec, vec);
        max_vec = _mm256_max_pd(max_vec, vec);
    }
    
    let mut min_vals = [0.0; LANE_COUNT_F64];
    let mut max_vals = [0.0; LANE_COUNT_F64];
    _mm256_storeu_pd(min_vals.as_mut_ptr(), min_vec);
    _mm256_storeu_pd(max_vals.as_mut_ptr(), max_vec);
    
    let mut min_val = min_vals[0].min(min_vals[1]).min(min_vals[2]).min(min_vals[3]);
    let mut max_val = max_vals[0].max(max_vals[1]).max(max_vals[2]).max(max_vals[3]);
    
    // Handle remainder
    let start = data.len() - (data.len() % LANE_COUNT_F64);
    for i in start..data.len() {
        min_val = min_val.min(data[i]);
        max_val = max_val.max(data[i]);
    }
    
    (min_val, max_val)
}

/// Public interface for SIMD min/max
pub fn find_min_max(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    
    if is_x86_feature_detected!("avx2") && data.len() >= LANE_COUNT_F64 {
        unsafe {
            simd_min_max_avx2(data)
        }
    } else {
        let mut min_val = data[0];
        let mut max_val = data[0];
        for &val in data.iter().skip(1) {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
        (min_val, max_val)
    }
}

/// Batch normalization for feature vectors using SIMD
pub fn normalize_batch(data: &mut [f64]) {
    if data.is_empty() {
        return;
    }
    
    let (min, max) = find_min_max(data);
    let range = max - min;
    
    if range > 1e-10 {
        let inv_range = 1.0 / range;
        scale_prices(data, inv_range);
        
        // Subtract normalized min
        let offset = min * inv_range;
        for val in data.iter_mut() {
            *val -= offset;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_scaling() {
        let mut prices = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        scale_prices(&mut prices, 2.0);
        
        assert!((prices[0] - 200.0).abs() < 1e-10);
        assert!((prices[4] - 1000.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = compute_dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let (min, max) = find_min_max(&data);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 9.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_aligned_vec() {
        let raw = vec![1.0, 2.0, 3.0];
        let aligned = AlignedVecF64::new(&raw);
        assert_eq!(aligned.len, 3);
        assert_eq!(aligned.padded_len, 4);
        assert_eq!(aligned.as_slice(), &[1.0, 2.0, 3.0]);
    }
}
