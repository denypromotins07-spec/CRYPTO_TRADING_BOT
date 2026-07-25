//! Cholesky Decomposition Engine for ZAID Trading Bot
//! Performs rapid Cholesky decomposition for VaR models
//! Optimized for 4x4 positive-definite covariance matrices
//!
//! This module provides numerically stable Cholesky factorization
//! for portfolio risk calculations, Monte Carlo simulation, and
//! multivariate normal sampling under the 8GB RAM constraint.

use std::arch::x86_64::*;

/// Fixed-size 4x4 matrix for Cholesky decomposition
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Cholesky4x4 {
    /// Lower triangular factor L where A = L * L^T
    pub lower: [[f64; 4]; 4],
    /// Whether decomposition was successful
    pub success: bool,
    /// Error message if decomposition failed
    pub error: Option<&'static str>,
}

impl Cholesky4x4 {
    /// Create empty decomposition result
    pub const fn new() -> Self {
        Self {
            lower: [[0.0; 4]; 4],
            success: false,
            error: None,
        }
    }
    
    /// Compute Cholesky decomposition of a 4x4 symmetric positive-definite matrix
    /// 
    /// Uses the standard algorithm with early termination on failure
    #[inline]
    pub fn decompose(matrix: &[[f64; 4]; 4]) -> Self {
        let mut result = Self::new();
        
        // Verify symmetry first (with tolerance)
        let epsilon = 1e-12;
        for i in 0..4 {
            for j in (i + 1)..4 {
                if (matrix[i][j] - matrix[j][i]).abs() > epsilon {
                    result.error = Some("Matrix is not symmetric");
                    return result;
                }
            }
        }
        
        // Cholesky-Banachiewicz algorithm
        for i in 0..4 {
            for j in 0..=i {
                let mut sum = 0.0;
                
                if j == i {
                    // Diagonal element
                    for k in 0..j {
                        sum += result.lower[j][k] * result.lower[j][k];
                    }
                    
                    let val = matrix[j][j] - sum;
                    
                    if val <= 0.0 {
                        result.error = Some("Matrix is not positive definite");
                        return result;
                    }
                    
                    result.lower[j][j] = val.sqrt();
                } else {
                    // Off-diagonal element
                    for k in 0..j {
                        sum += result.lower[i][k] * result.lower[j][k];
                    }
                    
                    if result.lower[j][j].abs() < 1e-15 {
                        result.error = Some("Zero diagonal encountered");
                        return result;
                    }
                    
                    result.lower[i][j] = (matrix[i][j] - sum) / result.lower[j][j];
                }
            }
            
            // Zero out upper triangle
            for j in (i + 1)..4 {
                result.lower[i][j] = 0.0;
            }
        }
        
        result.success = true;
        result
    }
    
    /// SIMD-accelerated Cholesky for the inner loop
    #[target_feature(enable = "avx2")]
    unsafe fn decompose_simd_inner(matrix: &[[f64; 4]; 4], 
                                    lower: &mut [[f64; 4]; 4]) -> Result<(), &'static str> {
        for i in 0..4 {
            for j in 0..=i {
                let mut sum_vec = _mm256_setzero_pd();
                
                if j == i {
                    // Load previously computed row
                    let row_ptr = lower[j].as_ptr();
                    
                    // Process 4 elements at once where possible
                    for k in (0..j).step_by(4) {
                        if k + 4 <= j {
                            let vals = _mm256_loadu_pd(row_ptr.add(k));
                            let prod = _mm256_mul_pd(vals, vals);
                            sum_vec = _mm256_add_pd(sum_vec, prod);
                        }
                    }
                    
                    // Horizontal sum
                    let mut sums = [0.0; 4];
                    _mm256_storeu_pd(sums.as_mut_ptr(), sum_vec);
                    let mut sum = sums[0] + sums[1] + sums[2] + sums[3];
                    
                    // Handle remainder
                    let rem_start = (j / 4) * 4;
                    for k in rem_start..j {
                        sum += lower[j][k] * lower[j][k];
                    }
                    
                    let val = matrix[j][j] - sum;
                    
                    if val <= 0.0 {
                        return Err("Matrix is not positive definite");
                    }
                    
                    lower[j][j] = val.sqrt();
                } else {
                    // Off-diagonal computation
                    for k in 0..j {
                        sum += lower[i][k] * lower[j][k];
                    }
                    
                    lower[i][j] = (matrix[i][j] - sum) / lower[j][j];
                }
            }
        }
        
        Ok(())
    }
    
    /// Solve L * y = b using forward substitution
    #[inline]
    pub fn forward_substitute(&self, b: &[f64; 4]) -> Option<[f64; 4]> {
        if !self.success {
            return None;
        }
        
        let mut y = [0.0; 4];
        
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..i {
                sum += self.lower[i][j] * y[j];
            }
            
            if self.lower[i][i].abs() < 1e-15 {
                return None;
            }
            
            y[i] = (b[i] - sum) / self.lower[i][i];
        }
        
        Some(y)
    }
    
    /// Solve L^T * x = y using backward substitution
    #[inline]
    pub fn backward_substitute(&self, y: &[f64; 4]) -> Option<[f64; 4]> {
        if !self.success {
            return None;
        }
        
        let mut x = [0.0; 4];
        
        for i in (0..4).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..4 {
                sum += self.lower[j][i] * x[j]; // Note: L^T[j][i] = L[i][j]
            }
            
            if self.lower[i][i].abs() < 1e-15 {
                return None;
            }
            
            x[i] = (y[i] - sum) / self.lower[i][i];
        }
        
        Some(x)
    }
    
    /// Solve A * x = b using Cholesky decomposition
    /// A = L * L^T, so we solve L * y = b, then L^T * x = y
    #[inline]
    pub fn solve(&self, b: &[f64; 4]) -> Option<[f64; 4]> {
        let y = self.forward_substitute(b)?;
        self.backward_substitute(&y)
    }
    
    /// Compute determinant from Cholesky factor
    /// det(A) = det(L)^2 = (product of diagonal elements)^2
    #[inline]
    pub fn determinant(&self) -> Option<f64> {
        if !self.success {
            return None;
        }
        
        let mut det = 1.0;
        for i in 0..4 {
            det *= self.lower[i][i];
        }
        
        Some(det * det)
    }
    
    /// Get the lower triangular factor
    #[inline]
    pub fn lower_factor(&self) -> Option<&[[f64; 4]; 4]> {
        if self.success {
            Some(&self.lower)
        } else {
            None
        }
    }
    
    /// Reconstruct original matrix from L * L^T (for verification)
    #[inline]
    pub fn reconstruct(&self) -> Option<[[f64; 4]; 4]> {
        if !self.success {
            return None;
        }
        
        let mut result = [[0.0; 4]; 4];
        
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.lower[i][k] * self.lower[j][k];
                }
                result[i][j] = sum;
            }
        }
        
        Some(result)
    }
}

impl Default for Cholesky4x4 {
    fn default() -> Self {
        Self::new()
    }
}

/// Dynamic-size Cholesky decomposition for larger matrices
pub struct CholeskyDecomposition {
    /// Lower triangular factor
    lower: Vec<f64>,
    size: usize,
    success: bool,
    error: Option<String>,
}

impl CholeskyDecomposition {
    /// Create decomposition for n x n matrix
    pub fn new(size: usize) -> Self {
        Self {
            lower: vec![0.0; size * size],
            size,
            success: false,
            error: None,
        }
    }
    
    /// Compute Cholesky decomposition
    pub fn decompose(&mut self, matrix: &[Vec<f64>]) -> Result<(), String> {
        let n = matrix.len();
        
        if n != matrix[0].len() {
            self.error = Some("Matrix must be square".to_string());
            return Err(self.error.clone().unwrap());
        }
        
        // Reset
        self.lower.fill(0.0);
        self.size = n;
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                
                if j == i {
                    for k in 0..j {
                        let l_ik = self.lower[i * n + k];
                        sum += l_ik * l_ik;
                    }
                    
                    let val = matrix[i][i] - sum;
                    
                    if val <= 0.0 {
                        self.error = Some("Matrix is not positive definite".to_string());
                        self.success = false;
                        return Err(self.error.clone().unwrap());
                    }
                    
                    self.lower[i * n + j] = val.sqrt();
                } else {
                    for k in 0..j {
                        sum += self.lower[i * n + k] * self.lower[j * n + k];
                    }
                    
                    let l_jj = self.lower[j * n + j];
                    if l_jj.abs() < 1e-15 {
                        self.error = Some("Zero diagonal encountered".to_string());
                        self.success = false;
                        return Err(self.error.clone().unwrap());
                    }
                    
                    self.lower[i * n + j] = (matrix[i][j] - sum) / l_jj;
                }
            }
        }
        
        self.success = true;
        Ok(())
    }
    
    /// Get element at (i, j) in lower factor
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if j > i {
            0.0
        } else {
            self.lower[i * self.size + j]
        }
    }
    
    /// Check if decomposition succeeded
    #[inline]
    pub fn is_success(&self) -> bool {
        self.success
    }
}

/// Utility functions for VaR calculations using Cholesky
pub mod var_utils {
    use super::*;
    
    /// Generate correlated random samples using Cholesky factor
    /// Given uncorrelated standard normals z, compute L * z for correlated samples
    pub fn generate_correlated_samples(cholesky: &Cholesky4x4, 
                                       z: &[f64; 4]) -> Option<[f64; 4]> {
        if !cholesky.success {
            return None;
        }
        
        let lower = cholesky.lower;
        let mut result = [0.0; 4];
        
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += lower[i][j] * z[j];
            }
            result[i] = sum;
        }
        
        Some(result)
    }
    
    /// Compute portfolio VaR using delta-normal method
    /// VaR = z_alpha * sqrt(w^T * Sigma * w)
    pub fn compute_var(weights: &[f64; 4], 
                       cov_matrix: &[[f64; 4]; 4],
                       z_alpha: f64) -> Option<f64> {
        let chol = Cholesky4x4::decompose(cov_matrix);
        
        if !chol.success {
            return None;
        }
        
        // Compute w^T * Sigma * w = ||L^T * w||^2
        let mut variance = 0.0;
        
        for j in 0..4 {
            let mut sum = 0.0;
            for i in j..4 {
                sum += chol.lower[i][j] * weights[i];
            }
            variance += sum * sum;
        }
        
        Some(z_alpha * variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cholesky_simple() {
        // Simple positive definite matrix
        let matrix = [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let chol = Cholesky4x4::decompose(&matrix);
        
        assert!(chol.success);
        assert!(chol.error.is_none());
        
        // Verify reconstruction
        let reconstructed = chol.reconstruct().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert!((reconstructed[i][j] - matrix[i][j]).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_cholesky_not_positive_definite() {
        // Matrix that is not positive definite
        let matrix = [
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let chol = Cholesky4x4::decompose(&matrix);
        
        assert!(!chol.success);
        assert_eq!(chol.error, Some("Matrix is not positive definite"));
    }
    
    #[test]
    fn test_solve_system() {
        let matrix = [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let chol = Cholesky4x4::decompose(&matrix);
        let b = [1.0, 2.0, 3.0, 4.0];
        
        let x = chol.solve(&b).unwrap();
        
        // Verify A * x = b
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..4 {
                sum += matrix[i][j] * x[j];
            }
            assert!((sum - b[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_determinant() {
        let matrix = [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        
        let chol = Cholesky4x4::decompose(&matrix);
        let det = chol.determinant().unwrap();
        
        // det = 4 * 3 * 2 * 1 = 24
        assert!((det - 24.0).abs() < 1e-10);
    }
}
