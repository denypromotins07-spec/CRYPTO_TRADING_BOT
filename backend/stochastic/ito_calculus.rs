//! Ito's Lemma Implementation for Stochastic Calculus
//!
//! This module implements Ito's Lemma for derivative pricing adjustments
//! and stochastic differential equation (SDE) transformations.
//! Essential for accurate option pricing and hedging in the ZAID bot.
//!
//! Features:
//! - Ito's Lemma for functions of diffusion processes
//! - Multi-dimensional Ito formula
//! - SDE transformation utilities
//! - Stratonovich correction terms
//!
//! Optimized for zero-cost abstractions and numerical precision.

/// Result of applying Ito's Lemma
#[derive(Debug, Clone)]
pub struct ItoResult {
    /// Drift term of the transformed process
    pub drift: f64,
    /// Diffusion term of the transformed process
    pub diffusion: f64,
    /// Second-order correction term (Ito correction)
    pub ito_correction: f64,
}

/// Ito's Lemma calculator for one-dimensional processes
pub struct ItoCalculus;

impl ItoCalculus {
    /// Apply Ito's Lemma to a function f(S_t) where dS_t = mu*S_t*dt + sigma*S_t*dW_t
    ///
    /// For geometric Brownian motion: df = (∂f/∂t + mu*S*∂f/∂S + 0.5*sigma²*S²*∂²f/∂S²)*dt 
    ///                                   + sigma*S*∂f/∂S*dW_t
    ///
    /// # Arguments
    /// * `s` - Current value of the underlying process
    /// * `mu` - Drift of underlying
    /// * `sigma` - Volatility of underlying
    /// * `df_ds` - First derivative ∂f/∂S
    /// * `d2f_ds2` - Second derivative ∂²f/∂S²
    /// * `df_dt` - Time derivative ∂f/∂t (optional, default 0)
    ///
    /// # Returns
    /// ItoResult with drift and diffusion of df
    pub fn apply_ito_1d(
        s: f64,
        mu: f64,
        sigma: f64,
        df_ds: f64,
        d2f_ds2: f64,
        df_dt: f64,
    ) -> ItoResult {
        // Ito drift: ∂f/∂t + mu*S*∂f/∂S + 0.5*sigma²*S²*∂²f/∂S²
        let drift = df_dt + mu * s * df_ds + 0.5 * sigma.powi(2) * s.powi(2) * d2f_ds2;
        
        // Ito diffusion: sigma*S*∂f/∂S
        let diffusion = sigma * s * df_ds;
        
        // Ito correction term: 0.5*sigma²*S²*∂²f/∂S²
        let ito_correction = 0.5 * sigma.powi(2) * s.powi(2) * d2f_ds2;
        
        ItoResult {
            drift,
            diffusion,
            ito_correction,
        }
    }
    
    /// Apply Ito's Lemma to log(S_t) - commonly used for returns
    ///
    /// If dS_t = mu*S_t*dt + sigma*S_t*dW_t, then
    /// d(log S_t) = (mu - 0.5*sigma²)*dt + sigma*dW_t
    ///
    /// # Arguments
    /// * `s` - Current price
    /// * `mu` - Drift of price process
    /// * `sigma` - Volatility
    ///
    /// # Returns
    /// Drift and volatility of the log process
    pub fn log_transform(s: f64, mu: f64, sigma: f64) -> (f64, f64) {
        // For f(S) = log(S):
        // df/dS = 1/S
        // d2f/dS2 = -1/S²
        
        let df_ds = 1.0 / s;
        let d2f_ds2 = -1.0 / s.powi(2);
        
        let result = Self::apply_ito_1d(s, mu, sigma, df_ds, d2f_ds2, 0.0);
        
        // Drift should be (mu - 0.5*sigma²), diffusion should be sigma
        (result.drift, result.diffusion / s)
    }
    
    /// Apply Ito's Lemma to power function S_t^n
    ///
    /// For f(S) = S^n:
    /// df = n*S^(n-1)*dS + 0.5*n*(n-1)*S^(n-2)*(dS)²
    ///
    /// # Arguments
    /// * `s` - Current value
    /// * `mu` - Drift
    /// * `sigma` - Volatility
    /// * `n` - Power
    pub fn power_transform(s: f64, mu: f64, sigma: f64, n: f64) -> ItoResult {
        if s <= 0.0 {
            return ItoResult {
                drift: 0.0,
                diffusion: 0.0,
                ito_correction: 0.0,
            };
        }
        
        let s_pow_n = s.powf(n);
        let df_ds = n * s_pow_n / s;
        let d2f_ds2 = n * (n - 1.0) * s_pow_n / s.powi(2);
        
        Self::apply_ito_1d(s, mu, sigma, df_ds, d2f_ds2, 0.0)
    }
    
    /// Calculate the Ito-Stratonovich correction
    ///
    /// When converting from Ito to Stratonovich interpretation:
    /// drift_strat = drift_ito - 0.5 * sigma * dsigma/dS
    ///
    /// # Arguments
    /// * `sigma` - Diffusion coefficient
    /// * `dsigma_ds` - Derivative of sigma with respect to S
    pub fn ito_to_stratonovich_correction(sigma: f64, dsigma_ds: f64) -> f64 {
        0.5 * sigma * dsigma_ds
    }
    
    /// Multi-dimensional Ito's Lemma for f(X_t, Y_t)
    ///
    /// For two correlated processes:
    /// dX = mu_x*dt + sigma_x*dW_x
    /// dY = mu_y*dt + sigma_y*dW_y
    /// d<X,Y>_t = rho*sigma_x*sigma_y*dt
    ///
    /// df = (∂f/∂t + mu_x*∂f/∂x + mu_y*∂f/∂y 
    ///       + 0.5*sigma_x²*∂²f/∂x² + 0.5*sigma_y²*∂²f/∂y² 
    ///       + rho*sigma_x*sigma_y*∂²f/∂x∂y)*dt
    ///      + sigma_x*∂f/∂x*dW_x + sigma_y*∂f/∂y*dW_y
    ///
    /// # Arguments
    /// * `x`, `y` - Current values
    /// * `mu_x`, `mu_y` - Drifts
    /// * `sigma_x`, `sigma_y` - Volatilities
    /// * `rho` - Correlation
    /// * `df_dx`, `df_dy` - First derivatives
    /// * `d2f_dx2`, `d2f_dy2`, `d2f_dxdy` - Second derivatives
    /// * `df_dt` - Time derivative
    pub fn apply_ito_2d(
        x: f64, y: f64,
        mu_x: f64, mu_y: f64,
        sigma_x: f64, sigma_y: f64,
        rho: f64,
        df_dx: f64, df_dy: f64,
        d2f_dx2: f64, d2f_dy2: f64, d2f_dxdy: f64,
        df_dt: f64,
    ) -> ItoResult {
        // Drift term
        let drift = df_dt
            + mu_x * df_dx
            + mu_y * df_dy
            + 0.5 * sigma_x.powi(2) * d2f_dx2
            + 0.5 * sigma_y.powi(2) * d2f_dy2
            + rho * sigma_x * sigma_y * d2f_dxdy;
        
        // Combined diffusion (magnitude)
        let diffusion = (
            sigma_x.powi(2) * df_dx.powi(2) 
            + sigma_y.powi(2) * df_dy.powi(2)
            + 2.0 * rho * sigma_x * sigma_y * df_dx * df_dy
        ).sqrt();
        
        // Ito correction (second-order terms)
        let ito_correction = 0.5 * sigma_x.powi(2) * d2f_dx2
            + 0.5 * sigma_y.powi(2) * d2f_dy2
            + rho * sigma_x * sigma_y * d2f_dxdy;
        
        ItoResult {
            drift,
            diffusion,
            ito_correction,
        }
    }
    
    /// Apply Ito's Lemma to product of two processes: f(X,Y) = X*Y
    ///
    /// d(XY) = X*dY + Y*dX + dX*dY
    ///       = X*(mu_y*dt + sigma_y*dW_y) + Y*(mu_x*dt + sigma_x*dW_x) 
    ///         + rho*sigma_x*sigma_y*dt
    pub fn product_rule(
        x: f64, y: f64,
        mu_x: f64, mu_y: f64,
        sigma_x: f64, sigma_y: f64,
        rho: f64,
    ) -> ItoResult {
        // For f(x,y) = x*y:
        // df/dx = y, df/dy = x
        // d2f/dx2 = 0, d2f/dy2 = 0, d2f/dxdy = 1
        
        Self::apply_ito_2d(
            x, y,
            mu_x, mu_y,
            sigma_x, sigma_y,
            rho,
            y, x,          // First derivatives
            0.0, 0.0, 1.0, // Second derivatives
            0.0,           // No explicit time dependence
        )
    }
    
    /// Apply Ito's Lemma to ratio of two processes: f(X,Y) = X/Y
    ///
    /// Useful for calculating dynamics of price ratios or returns
    pub fn quotient_rule(
        x: f64, y: f64,
        mu_x: f64, mu_y: f64,
        sigma_x: f64, sigma_y: f64,
        rho: f64,
    ) -> Option<ItoResult> {
        if y <= 0.0 {
            return None;
        }
        
        let ratio = x / y;
        
        // For f(x,y) = x/y:
        // df/dx = 1/y
        // df/dy = -x/y²
        // d2f/dx2 = 0
        // d2f/dy2 = 2x/y³
        // d2f/dxdy = -1/y²
        
        let df_dx = 1.0 / y;
        let df_dy = -x / y.powi(2);
        let d2f_dx2 = 0.0;
        let d2f_dy2 = 2.0 * x / y.powi(3);
        let d2f_dxdy = -1.0 / y.powi(2);
        
        Some(Self::apply_ito_2d(
            x, y,
            mu_x, mu_y,
            sigma_x, sigma_y,
            rho,
            df_dx, df_dy,
            d2f_dx2, d2f_dy2, d2f_dxdy,
            0.0,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_log_transform() {
        // Test that log transform gives correct drift adjustment
        let s = 100.0;
        let mu = 0.10;     // 10% drift
        let sigma = 0.20;  // 20% vol
        
        let (log_drift, log_vol) = ItoCalculus::log_transform(s, mu, sigma);
        
        // Expected: drift = mu - 0.5*sigma² = 0.10 - 0.5*0.04 = 0.08
        let expected_drift = mu - 0.5 * sigma.powi(2);
        
        assert!((log_drift - expected_drift).abs() < 1e-10);
        assert!((log_vol - sigma).abs() < 1e-10);
    }
    
    #[test]
    fn test_power_transform() {
        let s = 100.0;
        let mu = 0.05;
        let sigma = 0.15;
        let n = 2.0;
        
        let result = ItoCalculus::power_transform(s, mu, sigma, n);
        
        // For S², we can verify manually:
        // d(S²) = 2S*dS + (dS)² = 2S*(mu*S*dt + sigma*S*dW) + sigma²*S²*dt
        //       = (2*mu + sigma²)*S²*dt + 2*sigma*S²*dW
        
        let expected_drift = (2.0 * mu + sigma.powi(2)) * s.powi(2);
        let expected_diffusion = 2.0 * sigma * s.powi(2);
        
        assert!((result.drift - expected_drift).abs() < 1e-6);
        assert!((result.diffusion - expected_diffusion).abs() < 1e-6);
    }
    
    #[test]
    fn test_product_rule() {
        let x = 100.0;
        let y = 50.0;
        let mu_x = 0.05;
        let mu_y = 0.03;
        let sigma_x = 0.20;
        let sigma_y = 0.15;
        let rho = 0.5;
        
        let result = ItoCalculus::product_rule(x, y, mu_x, mu_y, sigma_x, sigma_y, rho);
        
        // Manual verification for product:
        // Drift = x*mu_y + y*mu_x + rho*sigma_x*sigma_y
        let expected_drift = x * mu_y + y * mu_x + rho * sigma_x * sigma_y;
        
        assert!((result.drift - expected_drift).abs() < 1e-6);
    }
    
    #[test]
    fn test_ito_correction_significance() {
        // Demonstrate that Ito correction matters for convex functions
        let s = 100.0;
        let mu = 0.05;
        let sigma = 0.30; // High vol
        
        // For S² (convex function)
        let result = ItoCalculus::power_transform(s, mu, sigma, 2.0);
        
        // Ito correction should be positive and significant
        assert!(result.ito_correction > 0.0);
        assert!(result.ito_correction > sigma.powi(2) * s.powi(2) * 0.1);
    }
}
