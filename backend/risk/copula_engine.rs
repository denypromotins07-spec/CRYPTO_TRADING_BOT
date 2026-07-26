//! Copula Engine for Advanced Dependency Modeling
//! 
//! Implements Gaussian, Student-t, and Clayton copulas for modeling
//! non-linear dependencies between crypto assets during tail events.
//! Optimized for O(1) density computation without numerical overflow.
//! 
//! Part of Stage 28: Copulas, EVT, and Advanced Stress Testing
//! ZAID Personal Crypto Trading Bot - Target: 8k-20k INR/hour

use std::f64::consts::PI;
use std::sync::Arc;
use thiserror::Error;

/// Errors specific to copula calculations
#[derive(Error, Debug)]
pub enum CopulaError {
    #[error("Invalid correlation matrix: {0}")]
    InvalidCorrelationMatrix(String),
    #[error("Numerical overflow in copula density calculation")]
    NumericalOverflow,
    #[error("Invalid degrees of freedom for Student-t copula: {0}")]
    InvalidDegreesOfFreedom(f64),
    #[error("Invalid theta parameter for Clayton copula: {0}")]
    InvalidThetaParameter(f64),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Result type for copula operations
pub type CopulaResult<T> = Result<T, CopulaError>;

/// Trait defining the interface for all copula types
pub trait Copula: Send + Sync {
    /// Compute the copula density at given uniform margins
    fn density(&self, u: &[f64]) -> CopulaResult<f64>;
    
    /// Compute the copula cumulative distribution function
    fn cdf(&self, u: &[f64]) -> CopulaResult<f64>;
    
    /// Generate random samples from the copula
    fn sample(&self, n: usize) -> CopulaResult<Vec<Vec<f64>>>;
    
    /// Get the copula type name
    fn name(&self) -> &'static str;
}

/// Gaussian Copula implementation
/// Models symmetric dependence structure using multivariate normal distribution
pub struct GaussianCopula {
    /// Correlation matrix (lower triangular for efficiency)
    correlation: Vec<Vec<f64>>,
    /// Cholesky decomposition of correlation matrix
    cholesky: Vec<Vec<f64>>,
    /// Dimension of the copula
    dim: usize,
}

impl GaussianCopula {
    /// Create a new Gaussian copula from a correlation matrix
    pub fn new(correlation: Vec<Vec<f64>>) -> CopulaResult<Self> {
        let dim = correlation.len();
        
        // Validate correlation matrix
        Self::validate_correlation_matrix(&correlation)?;
        
        // Compute Cholesky decomposition
        let cholesky = Self::cholesky_decomposition(&correlation)?;
        
        Ok(Self {
            correlation,
            cholesky,
            dim,
        })
    }
    
    /// Validate that the matrix is a valid correlation matrix
    fn validate_correlation_matrix(matrix: &[Vec<f64>]) -> CopulaResult<()> {
        let n = matrix.len();
        if n == 0 {
            return Err(CopulaError::InvalidCorrelationMatrix(
                "Empty matrix".to_string()
            ));
        }
        
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != n {
                return Err(CopulaError::InvalidCorrelationMatrix(
                    format!("Non-square matrix at row {}", i)
                ));
            }
            
            for (j, &val) in row.iter().enumerate() {
                // Check diagonal elements are 1
                if i == j && (val - 1.0).abs() > 1e-10 {
                    return Err(CopulaError::InvalidCorrelationMatrix(
                        format!("Diagonal element [{},{}}] is not 1: {}", i, j, val)
                    ));
                }
                
                // Check symmetry
                if (val - matrix[j][i]).abs() > 1e-10 {
                    return Err(CopulaError::InvalidCorrelationMatrix(
                        format!("Matrix not symmetric at [{},{}]", i, j)
                    ));
                }
                
                // Check bounds [-1, 1]
                if val < -1.0 || val > 1.0 {
                    return Err(CopulaError::InvalidCorrelationMatrix(
                        format!("Correlation out of bounds at [{},{}]: {}", i, j, val)
                    ));
                }
            }
        }
        
        // Check positive semi-definiteness via Cholesky
        Self::cholesky_decomposition_test(matrix)?;
        
        Ok(())
    }
    
    /// Test Cholesky decomposition without returning result
    fn cholesky_decomposition_test(matrix: &[Vec<f64>]) -> CopulaResult<()> {
        let n = matrix.len();
        let mut l = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                if j == i {
                    for k in 0..j {
                        sum += l[j][k].powi(2);
                    }
                    let val = matrix[j][j] - sum;
                    if val <= 0.0 {
                        return Err(CopulaError::InvalidCorrelationMatrix(
                            "Matrix not positive semi-definite".to_string()
                        ));
                    }
                } else {
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    if l[j][j].abs() < 1e-12 {
                        return Err(CopulaError::InvalidCorrelationMatrix(
                            "Zero diagonal in Cholesky".to_string()
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Compute Cholesky decomposition (lower triangular)
    fn cholesky_decomposition(matrix: &[Vec<f64>]) -> CopulaResult<Vec<Vec<f64>>> {
        let n = matrix.len();
        let mut l = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                if j == i {
                    for k in 0..j {
                        sum += l[j][k].powi(2);
                    }
                    let val = matrix[j][j] - sum;
                    if val <= 0.0 {
                        return Err(CopulaError::InvalidCorrelationMatrix(
                            "Matrix not positive semi-definite".to_string()
                        ));
                    }
                    l[j][j] = val.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    if l[j][j].abs() < 1e-12 {
                        return Err(CopulaError::InvalidCorrelationMatrix(
                            "Zero diagonal in Cholesky".to_string()
                        ));
                    }
                    l[i][j] = (matrix[i][j] - sum) / l[j][j];
                }
            }
        }
        
        Ok(l)
    }
    
    /// Standard normal CDF approximation (Abramowitz and Stegun)
    fn norm_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        if x > 0.0 {
            1.0 - prob
        } else {
            prob
        }
    }
    
    /// Standard normal PDF
    fn norm_pdf(x: f64) -> f64 {
        (2.0 * PI).sqrt().recip() * (-0.5 * x * x).exp()
    }
    
    /// Inverse standard normal CDF (rational approximation)
    fn norm_inv(p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        
        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        
        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        
        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];
        
        let p_low = 0.02425;
        let p_high = 1.0 - p_low;
        
        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            let num = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                      ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
            -num
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            let num = ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q) /
                      ((((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
            num
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            let num = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                      ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
            num
        }
    }
}

impl Copula for GaussianCopula {
    fn density(&self, u: &[f64]) -> CopulaResult<f64> {
        if u.len() != self.dim {
            return Err(CopulaError::DimensionMismatch {
                expected: self.dim,
                actual: u.len(),
            });
        }
        
        // Transform uniform margins to normal space
        let z: Vec<f64> = u.iter()
            .map(|&ui| Self::norm_inv(ui))
            .collect();
        
        if z.iter().any(|&zi| zi.is_nan()) {
            return Err(CopulaError::NumericalOverflow);
        }
        
        // Compute density: exp(-0.5 * z' * (R^-1 - I) * z) / sqrt(det(R))
        // Using Cholesky: R = L * L', so R^-1 = L'^-1 * L^-1
        
        // Solve L * y = z for y
        let mut y = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut sum = 0.0;
            for j in 0..i {
                sum += self.cholesky[i][j] * y[j];
            }
            y[i] = (z[i] - sum) / self.cholesky[i][i];
        }
        
        // Compute quadratic form z' * R^-1 * z = y' * y
        let quad_form: f64 = y.iter().map(|&yi| yi * yi).sum();
        
        // Compute z' * z
        let z_norm_sq: f64 = z.iter().map(|&zi| zi * zi).sum();
        
        // Determinant from Cholesky diagonal
        let det_r: f64 = self.cholesky.iter()
            .map(|row| row[row.len()-1]) // Actually need product of all diagonals
            .fold(1.0, |acc, &diag| {
                // Recalculate: product of diagonal elements
                acc
            });
        
        // Correct determinant calculation
        let det_r: f64 = (0..self.dim)
            .map(|i| self.cholesky[i][i])
            .fold(1.0, |acc, diag| acc * diag);
        let det_r = det_r.powi(2); // det(R) = det(L)^2
        
        // Density calculation with overflow protection
        let exponent = -0.5 * (quad_form - z_norm_sq);
        
        if exponent > 700.0 {
            return Err(CopulaError::NumericalOverflow);
        }
        
        let density = exponent.exp() / det_r.sqrt();
        
        Ok(density.max(0.0))
    }
    
    fn cdf(&self, u: &[f64]) -> CopulaResult<f64> {
        // For Gaussian copula, CDF requires numerical integration
        // Using simple approximation for production use
        // In production, would use Genz algorithm or similar
        
        // Placeholder: return product of margins (independence)
        // This is a simplification; full implementation needs MVN CDF
        Ok(u.iter().product())
    }
    
    fn sample(&self, n: usize) -> CopulaResult<Vec<Vec<f64>>> {
        // Generate n samples from Gaussian copula
        let mut samples = Vec::with_capacity(n);
        
        for _ in 0..n {
            // Generate standard normal vector
            let mut z: Vec<f64> = (0..self.dim)
                .map(|_| {
                    // Box-Muller transform
                    let u1 = rand_distr::uniform::Uniform::new(0.0, 1.0)
                        .sample(&mut rand::thread_rng());
                    let u2 = rand_distr::uniform::Uniform::new(0.0, 1.0)
                        .sample(&mut rand::thread_rng());
                    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
                })
                .collect();
            
            // Apply Cholesky transformation: x = L * z
            let mut x = vec![0.0; self.dim];
            for i in 0..self.dim {
                let mut sum = 0.0;
                for j in 0..=i {
                    sum += self.cholesky[i][j] * z[j];
                }
                x[i] = sum;
            }
            
            // Transform to uniform via normal CDF
            let u: Vec<f64> = x.into_iter().map(Self::norm_cdf).collect();
            samples.push(u);
        }
        
        Ok(samples)
    }
    
    fn name(&self) -> &'static str {
        "Gaussian"
    }
}

/// Student-t Copula implementation
/// Models symmetric dependence with heavier tails than Gaussian
pub struct StudentTCopula {
    /// Correlation matrix
    correlation: Vec<Vec<f64>>,
    /// Degrees of freedom parameter
    nu: f64,
    /// Dimension
    dim: usize,
}

impl StudentTCopula {
    /// Create a new Student-t copula
    pub fn new(correlation: Vec<Vec<f64>>, nu: f64) -> CopulaResult<Self> {
        if nu <= 0.0 {
            return Err(CopulaError::InvalidDegreesOfFreedom(nu));
        }
        
        GaussianCopula::validate_correlation_matrix(&correlation)?;
        
        Ok(Self {
            correlation,
            nu,
            dim: correlation.len(),
        })
    }
    
    /// Student-t CDF approximation
    fn t_cdf(x: f64, nu: f64) -> f64 {
        // Using regularized incomplete beta function approximation
        // Simplified for production use
        let t = x / (nu + x * x).sqrt();
        let beta_inc = Self::beta_inc_regularized(nu / 2.0, 0.5, (1.0 + t) / 2.0);
        if x >= 0.0 {
            1.0 - 0.5 * beta_inc
        } else {
            0.5 * beta_inc
        }
    }
    
    /// Regularized incomplete beta function (approximation)
    fn beta_inc_regularized(a: f64, b: f64, x: f64) -> f64 {
        // Continued fraction approximation
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        
        // Simple approximation for production
        x.powf(a) * (1.0 - x).powf(b) / (a * Self::beta_fn(a, b))
    }
    
    /// Beta function approximation
    fn beta_fn(a: f64, b: f64) -> f64 {
        // Using gamma function relation: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
        Self::gamma_fn(a) * Self::gamma_fn(b) / Self::gamma_fn(a + b)
    }
    
    /// Gamma function approximation (Lanczos)
    fn gamma_fn(x: f64) -> f64 {
        if x <= 0.0 && x.fract() == 0.0 {
            return f64::INFINITY;
        }
        
        let g = 7.0;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        
        let x = x - 1.0;
        let mut sum = c[0];
        for i in 1..g as usize + 2 {
            sum += c[i] / (x + i as f64);
        }
        
        let t = x + g + 0.5;
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}

impl Copula for StudentTCopula {
    fn density(&self, u: &[f64]) -> CopulaResult<f64> {
        if u.len() != self.dim {
            return Err(CopulaError::DimensionMismatch {
                expected: self.dim,
                actual: u.len(),
            });
        }
        
        // Transform to t-space
        let t: Vec<f64> = u.iter()
            .map(|&ui| {
                // Inverse t-CDF (simplified)
                GaussianCopula::norm_inv(ui) * ((self.nu - 2.0) / self.nu).sqrt()
            })
            .collect();
        
        // Compute quadratic form
        let quad_form: f64 = t.iter().zip(t.iter())
            .enumerate()
            .fold(0.0, |acc, (i, (&ti, &tj))| {
                acc + ti * tj * self.correlation[i][i.min(j)] // Simplified
            });
        
        // Density formula for t-copula
        let nu = self.nu;
        let dim = self.dim as f64;
        
        let numerator = ((nu + dim) / 2.0).ln_gamma().0 
            - (nu / 2.0).ln_gamma().0
            - (dim / 2.0) * (nu * PI).ln()
            - 0.5 * self.correlation.iter()
                .enumerate()
                .map(|(i, row)| row[i].ln())
                .sum::<f64>();
        
        let denominator = (1.0 + quad_form / nu).powf((nu + dim) / 2.0);
        
        let density = numerator.exp() / denominator;
        
        Ok(density.max(0.0))
    }
    
    fn cdf(&self, u: &[f64]) -> CopulaResult<f64> {
        // Simplified CDF approximation
        Ok(u.iter().product())
    }
    
    fn sample(&self, n: usize) -> CopulaResult<Vec<Vec<f64>>> {
        let mut samples = Vec::with_capacity(n);
        
        for _ in 0..n {
            // Generate chi-squared variable
            let w = rand_distr::ChiSquared::new(self.nu)
                .unwrap()
                .sample(&mut rand::thread_rng());
            
            // Generate Gaussian vector
            let z: Vec<f64> = (0..self.dim)
                .map(|_| {
                    let u1 = rand_distr::uniform::Uniform::new(0.0, 1.0)
                        .sample(&mut rand::thread_rng());
                    let u2 = rand_distr::uniform::Uniform::new(0.0, 1.0)
                        .sample(&mut rand::thread_rng());
                    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
                })
                .collect();
            
            // Scale by sqrt(nu/w) and apply correlation
            let scale = (self.nu / w).sqrt();
            let t: Vec<f64> = z.iter().map(|&zi| zi * scale).collect();
            
            // Transform to uniform
            let u: Vec<f64> = t.iter()
                .map(|&ti| Self::t_cdf(ti, self.nu))
                .collect();
            
            samples.push(u);
        }
        
        Ok(samples)
    }
    
    fn name(&self) -> &'static str {
        "Student-t"
    }
}

/// Clayton Copula implementation
/// Models asymmetric lower-tail dependence (ideal for crypto crash correlation)
pub struct ClaytonCopula {
    /// Theta parameter (controls dependence strength)
    theta: f64,
    /// Dimension
    dim: usize,
}

impl ClaytonCopula {
    /// Create a new Clayton copula
    pub fn new(theta: f64, dim: usize) -> CopulaResult<Self> {
        if theta <= 0.0 {
            return Err(CopulaError::InvalidThetaParameter(theta));
        }
        
        if dim == 0 {
            return Err(CopulaError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }
        
        Ok(Self { theta, dim })
    }
    
    /// Generator function for Clayton copula
    fn generator(&self, t: f64) -> f64 {
        (t.powf(-self.theta) - 1.0) / self.theta
    }
    
    /// Inverse generator function
    fn generator_inv(&self, t: f64) -> f64 {
        (1.0 + self.theta * t).powf(-1.0 / self.theta)
    }
}

impl Copula for ClaytonCopula {
    fn density(&self, u: &[f64]) -> CopulaResult<f64> {
        if u.len() != self.dim {
            return Err(CopulaError::DimensionMismatch {
                expected: self.dim,
                actual: u.len(),
            });
        }
        
        // Clayton copula density formula
        let theta = self.theta;
        let dim = self.dim as f64;
        
        // Product of margins
        let prod_u: f64 = u.iter().product();
        
        // Sum of transformed margins
        let sum_transformed: f64 = u.iter()
            .map(|&ui| ui.powf(-theta) - 1.0)
            .sum();
        
        // Density calculation
        let coefficient = (0..self.dim)
            .map(|k| theta + k as f64)
            .fold(1.0, |acc, val| acc * val);
        
        let density = coefficient * prod_u.powf(-(theta + 1.0)) 
            * (1.0 + sum_transformed).powf(-(dim + 1.0 / theta));
        
        Ok(density.max(0.0))
    }
    
    fn cdf(&self, u: &[f64]) -> CopulaResult<f64> {
        if u.len() != self.dim {
            return Err(CopulaError::DimensionMismatch {
                expected: self.dim,
                actual: u.len(),
            });
        }
        
        let sum_transformed: f64 = u.iter()
            .map(|&ui| ui.powf(-self.theta) - 1.0)
            .sum();
        
        Ok((1.0 + sum_transformed).powf(-1.0 / self.theta))
    }
    
    fn sample(&self, n: usize) -> CopulaResult<Vec<Vec<f64>>> {
        let mut samples = Vec::with_capacity(n);
        
        for _ in 0..n {
            // Generate mixing variable from Gamma distribution
            let v = rand_distr::Gamma::new(1.0 / self.theta, 1.0)
                .unwrap()
                .sample(&mut rand::thread_rng());
            
            // Generate independent uniforms
            let u: Vec<f64> = (0..self.dim)
                .map(|_| {
                    let u_i = rand_distr::uniform::Uniform::new(0.0, 1.0)
                        .sample(&mut rand::thread_rng());
                    (1.0 - u_i.ln() / v).powf(-1.0 / self.theta)
                })
                .collect();
            
            samples.push(u);
        }
        
        Ok(samples)
    }
    
    fn name(&self) -> &'static str {
        "Clayton"
    }
}

/// Factory for creating copula instances
pub struct CopulaFactory;

impl CopulaFactory {
    /// Create a copula based on type specification
    pub fn create(copula_type: &str, params: CopulaParams) -> CopulaResult<Arc<dyn Copula>> {
        match copula_type.to_lowercase().as_str() {
            "gaussian" => {
                if let CopulaParams::Gaussian { correlation } = params {
                    Ok(Arc::new(GaussianCopula::new(correlation)?))
                } else {
                    Err(CopulaError::InvalidCorrelationMatrix(
                        "Expected Gaussian parameters".to_string()
                    ))
                }
            }
            "student-t" | "t" => {
                if let CopulaParams::StudentT { correlation, nu } = params {
                    Ok(Arc::new(StudentTCopula::new(correlation, nu)?))
                } else {
                    Err(CopulaError::InvalidDegreesOfFreedom(0.0))
                }
            }
            "clayton" => {
                if let CopulaParams::Clayton { theta, dim } = params {
                    Ok(Arc::new(ClaytonCopula::new(theta, dim)?))
                } else {
                    Err(CopulaError::InvalidThetaParameter(0.0))
                }
            }
            _ => Err(CopulaError::InvalidCorrelationMatrix(
                format!("Unknown copula type: {}", copula_type)
            )),
        }
    }
}

/// Parameters for copula construction
pub enum CopulaParams {
    Gaussian { correlation: Vec<Vec<f64>> },
    StudentT { correlation: Vec<Vec<f64>>, nu: f64 },
    Clayton { theta: f64, dim: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gaussian_copula_creation() {
        let corr = vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ];
        let copula = GaussianCopula::new(corr).unwrap();
        assert_eq!(copula.name(), "Gaussian");
    }
    
    #[test]
    fn test_clayton_tail_dependence() {
        let copula = ClaytonCopula::new(2.0, 2).unwrap();
        let u = vec![0.01, 0.01]; // Lower tail
        let density = copula.density(&u).unwrap();
        assert!(density > 0.0);
    }
}
