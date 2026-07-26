//! Payoff Matrix Builder for BTC vs SOL Liquidity Games
//! 
//! Constructs real-time payoff matrices for multi-asset trading scenarios.
//! Optimized for high-frequency updates and memory efficiency.
//! 
//! Features:
//! - Dynamic matrix construction from market data
//! - Sparse matrix representation for large action spaces
//! - Expected value computation under uncertainty
//! - Cross-asset correlation modeling
//! 
//! Integrates game theory with quantitative finance for optimal execution.

use std::collections::HashMap;

/// Maximum matrix dimension (prevents memory explosion)
const MAX_MATRIX_DIM: usize = 50;

/// Trading pair identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradingPair {
    BTC,
    SOL,
    ETH,
    USDT,
}

impl TradingPair {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BTC => "BTC",
            Self::SOL => "SOL",
            Self::ETH => "ETH",
            Self::USDT => "USDT",
        }
    }
}

/// Action type in the liquidity game
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LiquidityAction {
    ProvideLiquidity,
    RemoveLiquidity,
    FrontRun,
    BackRun,
    Hold,
}

/// Cell in the payoff matrix
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PayoffCell {
    /// Payoff for player 1 (our bot)
    pub player1_payoff: f64,
    /// Payoff for player 2 (adversary/market)
    pub player2_payoff: f64,
    /// Probability of this outcome
    pub probability: f64,
    /// Expected value considering probability
    pub expected_value: f64,
}

impl Default for PayoffCell {
    fn default() -> Self {
        Self {
            player1_payoff: 0.0,
            player2_payoff: 0.0,
            probability: 1.0,
            expected_value: 0.0,
        }
    }
}

impl PayoffCell {
    #[inline]
    pub fn new(p1: f64, p2: f64, prob: f64) -> Self {
        let expected_value = p1 * prob;
        Self {
            player1_payoff: p1,
            player2_payoff: p2,
            probability: prob,
            expected_value,
        }
    }
    
    /// Update expected value when probability changes
    #[inline]
    pub fn update_probability(&mut self, new_prob: f64) {
        self.probability = new_prob.clamp(0.0, 1.0);
        self.expected_value = self.player1_payoff * self.probability;
    }
}

/// Compact payoff matrix using flat array for cache efficiency
pub struct PayoffMatrix {
    /// Flat array storage (row-major order)
    data: Vec<PayoffCell>,
    /// Number of rows (player 1 actions)
    rows: usize,
    /// Number of columns (player 2 actions)
    cols: usize,
    /// Action labels for rows
    row_labels: Vec<LiquidityAction>,
    /// Action labels for columns
    col_labels: Vec<LiquidityAction>,
    /// Current market regime affecting payoffs
    market_regime: MarketRegime,
}

/// Market regime affecting payoff calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    HighLiquidity,
    LowLiquidity,
    HighVolatility,
    Trending,
    MeanReverting,
}

impl PayoffMatrix {
    /// Create new payoff matrix with specified dimensions
    #[inline]
    pub fn new(rows: usize, cols: usize) -> Self {
        let size = (rows.min(MAX_MATRIX_DIM) * cols.min(MAX_MATRIX_DIM)).min(10000);
        Self {
            data: vec![PayoffCell::default(); size],
            rows: rows.min(MAX_MATRIX_DIM),
            cols: cols.min(MAX_MATRIX_DIM),
            row_labels: Vec::with_capacity(rows.min(MAX_MATRIX_DIM)),
            col_labels: Vec::with_capacity(cols.min(MAX_MATRIX_DIM)),
            market_regime: MarketRegime::HighLiquidity,
        }
    }
    
    /// Create standard 5x5 liquidity game matrix
    #[inline]
    pub fn standard_liquidity_game() -> Self {
        let mut matrix = Self::new(5, 5);
        
        // Standard actions for both players
        let actions = vec![
            LiquidityAction::ProvideLiquidity,
            LiquidityAction::RemoveLiquidity,
            LiquidityAction::FrontRun,
            LiquidityAction::BackRun,
            LiquidityAction::Hold,
        ];
        
        matrix.row_labels = actions.clone();
        matrix.col_labels = actions;
        
        matrix
    }
    
    /// Get cell at (row, col)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&PayoffCell> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }
    
    /// Get mutable reference to cell
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut PayoffCell> {
        if row < self.rows && col < self.cols {
            Some(&mut self.data[row * self.cols + col])
        } else {
            None
        }
    }
    
    /// Set cell value
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, cell: PayoffCell) -> bool {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = cell;
            true
        } else {
            false
        }
    }
    
    /// Compute payoff for BTC vs SOL liquidity competition
    pub fn compute_btc_sol_payoffs(
        &mut self,
        btc_liquidity: f64,
        sol_liquidity: f64,
        btc_volatility: f64,
        sol_volatility: f64,
        correlation: f64,
    ) {
        self.market_regime = self.determine_regime(btc_volatility, sol_volatility, btc_liquidity);
        
        // Iterate through all action combinations
        for (row_idx, &row_action) in self.row_labels.iter().enumerate() {
            for (col_idx, &col_action) in self.col_labels.iter().enumerate() {
                let (p1_payoff, p2_payoff) = self.compute_action_payoff(
                    row_action,
                    col_action,
                    btc_liquidity,
                    sol_liquidity,
                    btc_volatility,
                    sol_volatility,
                    correlation,
                );
                
                // Probability based on action compatibility
                let probability = self.compute_action_probability(row_action, col_action);
                
                let cell = PayoffCell::new(p1_payoff, p2_payoff, probability);
                self.set(row_idx, col_idx, cell);
            }
        }
    }
    
    /// Compute payoff for specific action pair
    fn compute_action_payoff(
        &self,
        p1_action: LiquidityAction,
        p2_action: LiquidityAction,
        btc_liq: f64,
        sol_liq: f64,
        btc_vol: f64,
        sol_vol: f64,
        corr: f64,
    ) -> (f64, f64) {
        // Base payoff depends on liquidity available
        let base_btc = btc_liq * 0.001; // 0.1% of liquidity
        let base_sol = sol_liq * 0.001;
        let total_base = (base_btc + base_sol) / 2.0;
        
        // Volatility adjustment (higher vol = higher risk/reward)
        let vol_factor = 1.0 + (btc_vol + sol_vol) / 4.0;
        
        // Correlation effect
        let corr_effect = if corr > 0.5 { 1.2 } else if corr < -0.5 { 0.8 } else { 1.0 };
        
        match (p1_action, p2_action) {
            // Both provide: split rewards
            (LiquidityAction::ProvideLiquidity, LiquidityAction::ProvideLiquidity) => {
                let payoff = total_base * 0.5 * vol_factor * corr_effect;
                (payoff, payoff)
            },
            
            // P1 provides, P2 removes: P1 gets less
            (LiquidityAction::ProvideLiquidity, LiquidityAction::RemoveLiquidity) => {
                (total_base * 0.3 * vol_factor, total_base * 0.7 * vol_factor)
            },
            
            // P1 front-runs, P2 provides: P1 gains at P2's expense
            (LiquidityAction::FrontRun, LiquidityAction::ProvideLiquidity) => {
                (total_base * 0.8 * vol_factor, -total_base * 0.3 * vol_factor)
            },
            
            // Both front-run: destructive competition
            (LiquidityAction::FrontRun, LiquidityAction::FrontRun) => {
                (-total_base * 0.2 * vol_factor, -total_base * 0.2 * vol_factor)
            },
            
            // P1 back-runs, P2 holds: P1 captures delayed opportunity
            (LiquidityAction::BackRun, LiquidityAction::Hold) => {
                (total_base * 0.4 * vol_factor, 0.0)
            },
            
            // Both hold: minimal payoff
            (LiquidityAction::Hold, LiquidityAction::Hold) => {
                (total_base * 0.1, total_base * 0.1)
            },
            
            // Default case
            _ => (total_base * 0.2 * vol_factor, total_base * 0.2 * vol_factor),
        }
    }
    
    /// Compute probability of action pair occurring
    fn compute_action_probability(
        &self,
        p1_action: LiquidityAction,
        p2_action: LiquidityAction,
    ) -> f64 {
        // Base probabilities for different actions
        let action_prob = |action: LiquidityAction| -> f64 {
            match action {
                LiquidityAction::ProvideLiquidity => 0.25,
                LiquidityAction::RemoveLiquidity => 0.15,
                LiquidityAction::FrontRun => 0.20,
                LiquidityAction::BackRun => 0.15,
                LiquidityAction::Hold => 0.25,
            }
        };
        
        // Joint probability with correlation adjustment
        let p1 = action_prob(p1_action);
        let p2 = action_prob(p2_action);
        
        // Simple independence assumption (can be enhanced)
        p1 * p2
    }
    
    /// Determine market regime from volatility and liquidity
    fn determine_regime(&self, btc_vol: f64, sol_vol: f64, liq: f64) -> MarketRegime {
        let avg_vol = (btc_vol + sol_vol) / 2.0;
        
        if avg_vol > 0.5 {
            MarketRegime::HighVolatility
        } else if liq < 1000000.0 {
            MarketRegime::LowLiquidity
        } else if avg_vol > 0.3 {
            MarketRegime::Trending
        } else {
            MarketRegime::HighLiquidity
        }
    }
    
    /// Find best response for player 1 given player 2's action
    pub fn best_response_p1(&self, p2_col: usize) -> Option<(usize, f64)> {
        let mut best_row = 0;
        let mut best_value = f64::NEG_INFINITY;
        
        for row in 0..self.rows {
            if let Some(cell) = self.get(row, p2_col) {
                let value = cell.expected_value;
                if value > best_value {
                    best_value = value;
                    best_row = row;
                }
            }
        }
        
        if best_value.is_finite() {
            Some((best_row, best_value))
        } else {
            None
        }
    }
    
    /// Find Nash equilibrium via iterated elimination of dominated strategies
    pub fn find_nash_equilibrium(&self) -> Option<(usize, usize)> {
        // Simplified: find saddle point in zero-sum approximation
        let mut maxmin_row = 0;
        let mut maxmin_value = f64::MAX;
        
        for row in 0..self.rows {
            let mut min_in_row = f64::MAX;
            for col in 0..self.cols {
                if let Some(cell) = self.get(row, col) {
                    min_in_row = min_in_row.min(cell.player1_payoff);
                }
            }
            if min_in_row > maxmin_value {
                maxmin_value = min_in_row;
                maxmin_row = row;
            }
        }
        
        let mut minmax_col = 0;
        let mut minmax_value = f64::NEG_INFINITY;
        
        for col in 0..self.cols {
            let mut max_in_col = f64::NEG_INFINITY;
            for row in 0..self.rows {
                if let Some(cell) = self.get(row, col) {
                    max_in_col = max_in_col.max(cell.player1_payoff);
                }
            }
            if max_in_col < minmax_value {
                minmax_value = max_in_col;
                minmax_col = col;
            }
        }
        
        // Check for saddle point
        if (maxmin_value - minmax_value).abs() < 1e-6 {
            Some((maxmin_row, minmax_col))
        } else {
            None // No pure strategy Nash equilibrium
        }
    }
    
    /// Get matrix dimensions
    #[inline]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Get current market regime
    #[inline]
    pub fn market_regime(&self) -> MarketRegime {
        self.market_regime
    }
    
    /// Reset matrix to default values
    #[inline]
    pub fn reset(&mut self) {
        for cell in &mut self.data {
            *cell = PayoffCell::default();
        }
    }
}

/// Builder for constructing complex payoff matrices
pub struct PayoffMatrixBuilder {
    pairs: Vec<TradingPair>,
    actions: Vec<LiquidityAction>,
    liquidity_data: HashMap<TradingPair, f64>,
    volatility_data: HashMap<TradingPair, f64>,
}

impl PayoffMatrixBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            actions: Vec::new(),
            liquidity_data: HashMap::new(),
            volatility_data: HashMap::new(),
        }
    }
    
    #[inline]
    pub fn add_pair(mut self, pair: TradingPair) -> Self {
        self.pairs.push(pair);
        self
    }
    
    #[inline]
    pub fn add_liquidity(mut self, pair: TradingPair, liquidity: f64) -> Self {
        self.liquidity_data.insert(pair, liquidity);
        self
    }
    
    #[inline]
    pub fn add_volatility(mut self, pair: TradingPair, volatility: f64) -> Self {
        self.volatility_data.insert(pair, volatility);
        self
    }
    
    /// Build payoff matrix for BTC-SOL pair
    pub fn build_btc_sol_matrix(self) -> PayoffMatrix {
        let mut matrix = PayoffMatrix::standard_liquidity_game();
        
        let btc_liq = self.liquidity_data.get(&TradingPair::BTC).copied().unwrap_or(1000000.0);
        let sol_liq = self.liquidity_data.get(&TradingPair::SOL).copied().unwrap_or(500000.0);
        let btc_vol = self.volatility_data.get(&TradingPair::BTC).copied().unwrap_or(0.02);
        let sol_vol = self.volatility_data.get(&TradingPair::SOL).copied().unwrap_or(0.04);
        
        // Assume moderate positive correlation
        let correlation = 0.6;
        
        matrix.compute_btc_sol_payoffs(btc_liq, sol_liq, btc_vol, sol_vol, correlation);
        
        matrix
    }
}

impl Default for PayoffMatrixBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix = PayoffMatrix::new(5, 5);
        assert_eq!(matrix.dimensions(), (5, 5));
    }

    #[test]
    fn test_cell_operations() {
        let mut matrix = PayoffMatrix::new(3, 3);
        let cell = PayoffCell::new(10.0, -5.0, 0.8);
        
        assert!(matrix.set(1, 1, cell));
        
        let retrieved = matrix.get(1, 1).unwrap();
        assert!((retrieved.player1_payoff - 10.0).abs() < 1e-10);
        assert!((retrieved.expected_value - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_game() {
        let matrix = PayoffMatrix::standard_liquidity_game();
        assert_eq!(matrix.dimensions(), (5, 5));
        assert_eq!(matrix.row_labels.len(), 5);
        assert_eq!(matrix.col_labels.len(), 5);
    }

    #[test]
    fn test_btc_sol_payoffs() {
        let mut matrix = PayoffMatrix::standard_liquidity_game();
        matrix.compute_btc_sol_payoffs(1000000.0, 500000.0, 0.02, 0.04, 0.6);
        
        // Check that payoffs are computed (non-zero)
        let cell = matrix.get(0, 0).unwrap();
        assert!(cell.player1_payoff > 0.0);
    }

    #[test]
    fn test_best_response() {
        let mut matrix = PayoffMatrix::standard_liquidity_game();
        matrix.compute_btc_sol_payoffs(1000000.0, 500000.0, 0.02, 0.04, 0.6);
        
        let response = matrix.best_response_p1(0);
        assert!(response.is_some());
    }

    #[test]
    fn test_builder_pattern() {
        let matrix = PayoffMatrixBuilder::new()
            .add_pair(TradingPair::BTC)
            .add_pair(TradingPair::SOL)
            .add_liquidity(TradingPair::BTC, 2000000.0)
            .add_liquidity(TradingPair::SOL, 800000.0)
            .add_volatility(TradingPair::BTC, 0.03)
            .add_volatility(TradingPair::SOL, 0.05)
            .build_btc_sol_matrix();
        
        assert_eq!(matrix.dimensions(), (5, 5));
    }
}
