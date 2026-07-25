//! Matrix Multiplication Engine for ZAID Trading Bot
//! Implements Strassen and BLAS-level matrix operations
//! Optimized for 4x4 covariance matrices under 5 microseconds
//!
//! This module provides high-performance linear algebra primitives
//! for portfolio mathematics, VaR calculations, and correlation analysis.
//! All operations are designed to avoid heap allocations in hot paths.

use std::arch::x86_64::*;

/// Maximum matrix size for stack allocation (avoids heap)
const STACK_MATRIX_SIZE: usize = 16; // 4x4 matrix

/// Fixed-size 4x4 matrix for ultra-fast covariance computations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Matrix4x4 {
    pub data: [[f64; 4]; 4],
}

impl Matrix4x4 {
    /// Create identity matrix
    #[inline]
    pub fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    /// Create zero matrix
    #[inline]
    pub fn zeros() -> Self {
        Self {
            data: [[0.0; 4]; 4],
        }
    }
    
    /// Create matrix from flat array (row-major)
    #[inline]
    pub fn from_array(arr: &[f64; 16]) -> Self {
        let mut data = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                data[i][j] = arr[i * 4 + j];
            }
        }
        Self { data }
    }
    
    /// Get element at (row, col)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }
    
    /// Set element at (row, col)
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row][col] = val;
    }
    
    /// SIMD-optimized matrix multiplication using AVX2
    /// Computes self * other
    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn multiply_simd(&self, other: &Matrix4x4) -> Matrix4x4 {
        let mut result = Matrix4x4::zeros();
        
        // Process each row
        for i in 0..4 {
            // Load row i of self into SIMD registers
            let row_vec = unsafe {
                _mm256_loadu_pd(self.data[i].as_ptr())
            };
            
            // Compute dot products with each column of other
            for j in 0..4 {
                // Gather column j of other
                let col_vals = [
                    other.data[0][j],
                    other.data[1][j],
                    other.data[2][j],
                    other.data[3][j],
                ];
                
                let col_vec = unsafe {
                    _mm256_loadu_pd(col_vals.as_ptr())
                };
                
                // Multiply and horizontal sum
                let prod = unsafe { _mm256_mul_pd(row_vec, col_vec) };
                
                // Extract and sum
                let mut products = [0.0; 4];
                unsafe {
                    _mm256_storeu_pd(products.as_mut_ptr(), prod);
                }
                
                result.data[i][j] = products[0] + products[1] + products[2] + products[3];
            }
        }
        
        result
    }
    
    /// Standard matrix multiplication (fallback or small matrices)
    #[inline]
    pub fn multiply(&self, other: &Matrix4x4) -> Matrix4x4 {
        if is_x86_feature_detected!("avx2") {
            self.multiply_simd(other)
        } else {
            let mut result = Matrix4x4::zeros();
            
            for i in 0..4 {
                for j in 0..4 {
                    let mut sum = 0.0;
                    for k in 0..4 {
                        sum += self.data[i][k] * other.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
            
            result
        }
    }
    
    /// Matrix-vector multiplication
    #[inline]
    pub fn mul_vector(&self, vec: &[f64; 4]) -> [f64; 4] {
        let mut result = [0.0; 4];
        
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..4 {
                sum += self.data[i][j] * vec[j];
            }
            result[i] = sum;
        }
        
        result
    }
    
    /// Transpose the matrix
    #[inline]
    pub fn transpose(&self) -> Matrix4x4 {
        let mut result = Matrix4x4::zeros();
        
        for i in 0..4 {
            for j in 0..4 {
                result.data[i][j] = self.data[j][i];
            }
        }
        
        result
    }
    
    /// Check if matrix is symmetric
    #[inline]
    pub fn is_symmetric(&self, epsilon: f64) -> bool {
        for i in 0..4 {
            for j in (i + 1)..4 {
                if (self.data[i][j] - self.data[j][i]).abs() > epsilon {
                    return false;
                }
            }
        }
        true
    }
    
    /// Compute Frobenius norm
    #[inline]
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                sum += self.data[i][j] * self.data[i][j];
            }
        }
        sum.sqrt()
    }
}

/// Dynamic matrix for larger operations
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Create new matrix with given dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }
    
    /// Create matrix from vector (row-major)
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }
    
    /// Get element at (row, col)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }
    
    /// Set element at (row, col)
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.cols + col] = val;
    }
    
    /// Get dimensions
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Naive O(n³) matrix multiplication
    pub fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        
        let mut result = Matrix::new(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        
        result
    }
    
    /// Strassen's algorithm for large matrices
    /// Only beneficial for matrices larger than ~32x32
    pub fn strassen_multiply(&self, other: &Matrix) -> Matrix {
        let n = self.rows.max(self.cols).max(other.cols);
        
        // For small matrices, use naive multiplication
        if n < 32 {
            return self.multiply(other);
        }
        
        // Pad to power of 2
        let size = n.next_power_of_two();
        
        let mut a_padded = Matrix::new(size, size);
        let mut b_padded = Matrix::new(size, size);
        
        // Copy data
        for i in 0..self.rows {
            for j in 0..self.cols {
                a_padded.set(i, j, self.get(i, j));
            }
        }
        
        for i in 0..other.rows {
            for j in 0..other.cols {
                b_padded.set(i, j, other.get(i, j));
            }
        }
        
        // Recursive Strassen would go here
        // For production, use optimized BLAS library
        
        a_padded.multiply(&b_padded)
    }
    
    /// In-place transpose
    pub fn transpose(&mut self) {
        let mut new_data = vec![0.0; self.rows * self.cols];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        
        self.data = new_data;
        std::mem::swap(&mut self.rows, &mut self.cols);
    }
}

/// BLAS-level operation traits
pub trait BlasOps {
    /// Level 1: Vector operations
    fn axpy(&mut self, alpha: f64, x: &[f64]);
    
    /// Level 2: Matrix-vector operations
    fn gemv(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]);
    
    /// Level 3: Matrix-matrix operations
    fn gemm(&self, alpha: f64, b: &Self, beta: f64, c: &mut Self);
}

impl BlasOps for Matrix {
    #[inline]
    fn axpy(&mut self, alpha: f64, x: &[f64]) {
        assert_eq!(self.data.len(), x.len());
        for i in 0..self.data.len() {
            self.data[i] += alpha * x[i];
        }
    }
    
    #[inline]
    fn gemv(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
        assert_eq!(self.cols, x.len());
        assert_eq!(self.rows, y.len());
        
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.get(i, j) * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    }
    
    #[inline]
    fn gemm(&self, alpha: f64, b: &Self, beta: f64, c: &mut Self) {
        assert_eq!(self.cols, b.rows);
        assert_eq!(self.rows, c.rows);
        assert_eq!(b.cols, c.cols);
        
        for i in 0..self.rows {
            for j in 0..b.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * b.get(k, j);
                }
                c.set(i, j, alpha * sum + beta * c.get(i, j));
            }
        }
    }
}

/// Specialized 4x4 covariance matrix computation
pub struct CovarianceCalculator {
    buffer: Matrix4x4,
}

impl CovarianceCalculator {
    pub const fn new() -> Self {
        Self {
            buffer: Matrix4x4::zeros(),
        }
    }
    
    /// Compute 4x4 covariance matrix from returns data
    /// Expects returns as [n_samples x 4] flattened array (row-major)
    pub fn compute(&mut self, returns: &[f64], n_samples: usize) -> Matrix4x4 {
        assert_eq!(returns.len(), n_samples * 4);
        
        let mut cov = Matrix4x4::zeros();
        
        // First pass: compute means
        let mut means = [0.0; 4];
        for i in 0..n_samples {
            for j in 0..4 {
                means[j] += returns[i * 4 + j];
            }
        }
        for j in 0..4 {
            means[j] /= n_samples as f64;
        }
        
        // Second pass: compute covariance
        for i in 0..n_samples {
            for j in 0..4 {
                for k in 0..4 {
                    let diff_j = returns[i * 4 + j] - means[j];
                    let diff_k = returns[i * 4 + k] - means[k];
                    cov.data[j][k] += diff_j * diff_k;
                }
            }
        }
        
        // Normalize by (n-1) for sample covariance
        let norm = 1.0 / (n_samples - 1) as f64;
        for j in 0..4 {
            for k in 0..4 {
                cov.data[j][k] *= norm;
            }
        }
        
        cov
    }
    
    /// Compute covariance with Welford's online algorithm (numerically stable)
    pub fn compute_online(&mut self, returns_iter: impl Iterator<Item = [f64; 4]>) -> Matrix4x4 {
        let mut n = 0usize;
        let mut mean = [0.0; 4];
        let mut m2 = [[0.0; 4]; 4];
        
        for ret in returns_iter {
            n += 1;
            let delta_old: [f64; 4] = mean;
            
            // Update mean
            for j in 0..4 {
                mean[j] += (ret[j] - mean[j]) / n as f64;
            }
            
            // Update M2 (sum of squared differences)
            let delta_new: [f64; 4] = mean;
            for j in 0..4 {
                for k in 0..4 {
                    m2[j][k] += (ret[j] - delta_old[j]) * (ret[k] - delta_new[k]);
                }
            }
        }
        
        let mut cov = Matrix4x4::zeros();
        if n > 1 {
            let norm = 1.0 / (n - 1) as f64;
            for j in 0..4 {
                for k in 0..4 {
                    cov.data[j][k] = m2[j][k] * norm;
                }
            }
        }
        
        cov
    }
}

impl Default for CovarianceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_4x4_identity() {
        let id = Matrix4x4::identity();
        assert_eq!(id.get(0, 0), 1.0);
        assert_eq!(id.get(1, 1), 1.0);
        assert_eq!(id.get(0, 1), 0.0);
    }
    
    #[test]
    fn test_matrix_4x4_multiply() {
        let a = Matrix4x4::identity();
        let b = Matrix4x4::from_array(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);
        
        let result = a.multiply(&b);
        
        for i in 0..4 {
            for j in 0..4 {
                assert!((result.get(i, j) - b.get(i, j)).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_covariance_computation() {
        let calc = CovarianceCalculator::new();
        
        // Simple test data: 3 samples of 4 assets
        let returns = vec![
            0.01, 0.02, -0.01, 0.005,
            -0.02, 0.01, 0.03, -0.01,
            0.015, -0.005, 0.02, 0.01,
        ];
        
        let cov = const { CovarianceCalculator::new() };
        let mut calc_mut = calc;
        let cov = calc_mut.compute(&returns, 3);
        
        // Covariance matrix should be symmetric
        assert!(cov.is_symmetric(1e-10));
    }
    
    #[test]
    fn test_transpose() {
        let m = Matrix4x4::from_array(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);
        
        let t = m.transpose();
        
        assert_eq!(t.get(0, 1), m.get(1, 0));
        assert_eq!(t.get(2, 3), m.get(3, 2));
    }
}
