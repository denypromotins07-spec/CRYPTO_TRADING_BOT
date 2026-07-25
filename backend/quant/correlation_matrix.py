"""
Correlation Matrix Engine for ZAID Personal Crypto Trading Bot
Calculates cointegration and cross-asset correlation
Optimized with NumPy for efficient matrix operations
Memory-efficient rolling window implementation

Part of the 152 domains of quantitative finance implementation.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Correlation analysis result."""
    asset_pair: Tuple[str, str]
    correlation: float
    cointegration_pvalue: Optional[float]
    rolling_correlation: float
    divergence_score: float
    timestamp: float


class RollingCorrelation:
    """
    Efficient rolling correlation calculator.
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.values_x: deque = deque(maxlen=window_size)
        self.values_y: deque = deque(maxlen=window_size)
        
        # Running statistics
        self.n = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.M2_x = 0.0
        self.M2_y = 0.0
        self.co_moment = 0.0
        
    def update(self, x: float, y: float) -> Optional[float]:
        """Update correlation with new pair of values."""
        self.values_x.append(x)
        self.values_y.append(y)
        
        # Update running statistics
        self.n += 1
        
        # Update means
        delta_x = x - self.mean_x
        delta_y = y - self.mean_y
        self.mean_x += delta_x / self.n
        self.mean_y += delta_y / self.n
        
        # Update co-moment
        delta2_x = x - self.mean_x
        delta2_y = y - self.mean_y
        self.co_moment += delta_x * delta2_y
        
        # Update M2 (sum of squared deviations)
        self.M2_x += delta_x * delta2_x
        self.M2_y += delta_y * delta2_y
        
        if self.n < self.window_size:
            return None
        
        # Calculate correlation from running statistics
        var_x = self.M2_x / self.n
        var_y = self.M2_y / self.n
        
        if var_x <= 0 or var_y <= 0:
            return 0.0
        
        covariance = self.co_moment / self.n
        correlation = covariance / np.sqrt(var_x * var_y)
        
        return np.clip(correlation, -1.0, 1.0)
    
    def get_correlation(self) -> Optional[float]:
        """Get current correlation value."""
        if self.n < 2:
            return None
        
        var_x = self.M2_x / self.n
        var_y = self.M2_y / self.n
        
        if var_x <= 0 or var_y <= 0:
            return 0.0
        
        covariance = self.co_moment / self.n
        return np.clip(covariance / np.sqrt(var_x * var_y), -1.0, 1.0)


class CointegrationTest:
    """
    Simplified cointegration test using Engle-Granger two-step method.
    Optimized for pairs trading opportunities.
    """
    
    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self.prices_x: deque = deque(maxlen=200)
        self.prices_y: deque = deque(maxlen=200)
        
    def update(self, price_x: float, price_y: float) -> Optional[Dict[str, float]]:
        """Update with new prices and test for cointegration."""
        self.prices_x.append(np.log(price_x))
        self.prices_y.append(np.log(price_y))
        
        if len(self.prices_x) < self.min_samples:
            return None
        
        x_arr = np.array(list(self.prices_x))
        y_arr = np.array(list(self.prices_y))
        
        # Step 1: OLS regression
        X = np.vstack([np.ones(len(x_arr)), x_arr]).T
        try:
            beta, alpha = np.linalg.lstsq(X, y_arr, rcond=None)[0]
        except:
            return None
        
        # Calculate residuals
        residuals = y_arr - (alpha + beta * x_arr)
        
        # Step 2: ADF test on residuals (simplified)
        # Check if residuals are stationary using variance ratio
        if len(residuals) > 50:
            first_half_var = np.var(residuals[:len(residuals)//2])
            second_half_var = np.var(residuals[len(residuals)//2:])
            
            # If variance decreases, suggests mean reversion
            variance_ratio = second_half_var / max(first_half_var, 0.0001)
            
            # Augmented Dickey-Fuller approximation
            diff_residuals = np.diff(residuals)
            lagged = residuals[:-1]
            
            if len(lagged) > 10 and np.var(lagged) > 0:
                try:
                    # Simple regression of diff on lagged
                    coef = np.corrcoef(diff_residuals, lagged)[0, 1]
                    adf_stat = coef * np.sqrt(len(lagged))
                    
                    # Approximate p-value (simplified)
                    p_value = 1 / (1 + abs(adf_stat) * 0.5)
                    
                    return {
                        'hedge_ratio': beta,
                        'intercept': alpha,
                        'adf_statistic': adf_stat,
                        'p_value': p_value,
                        'is_cointegrated': p_value < 0.05,
                        'variance_ratio': variance_ratio
                    }
                except:
                    pass
        
        return None


class CorrelationMatrixEngine:
    """
    Main engine for cross-asset correlation analysis.
    Singleton pattern for global access.
    """
    
    _instance: Optional['CorrelationMatrixEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.assets: Set[str] = set()
        self.correlations: Dict[Tuple[str, str], RollingCorrelation] = {}
        self.cointegration_tests: Dict[Tuple[str, str], CointegrationTest] = {}
        
        self.price_history: Dict[str, deque] = {}
        self.window_size = 60
        
        self._initialized = True
        logger.info("CorrelationMatrixEngine initialized")
    
    def add_asset(self, symbol: str):
        """Add a new asset to track."""
        if symbol not in self.assets:
            self.assets.add(symbol)
            self.price_history[symbol] = deque(maxlen=self.window_size * 2)
            
            # Create correlation trackers with existing assets
            for existing in self.assets:
                if existing != symbol:
                    pair = tuple(sorted([symbol, existing]))
                    if pair not in self.correlations:
                        self.correlations[pair] = RollingCorrelation(self.window_size)
                        self.cointegration_tests[pair] = CointegrationTest()
    
    def update_prices(self, prices: Dict[str, float], timestamp: float) -> Dict[str, any]:
        """Update all correlations with new prices."""
        results = {
            'timestamp': timestamp,
            'correlations': {},
            'cointegration': {},
            'divergences': []
        }
        
        # Add any new assets
        for symbol in prices:
            self.add_asset(symbol)
            self.price_history[symbol].append(prices[symbol])
        
        # Calculate returns
        returns = {}
        for symbol, price in prices.items():
            if len(self.price_history[symbol]) > 1:
                prev_price = list(self.price_history[symbol])[-2]
                returns[symbol] = (price - prev_price) / prev_price if prev_price > 0 else 0
            else:
                returns[symbol] = 0
        
        # Update pairwise correlations
        symbols = list(self.assets)
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1, s2 = symbols[i], symbols[j]
                pair = tuple(sorted([s1, s2]))
                
                if s1 in returns and s2 in returns:
                    corr = self.correlations[pair].update(returns[s1], returns[s2])
                    
                    if corr is not None:
                        results['correlations'][f"{s1}_{s2}"] = corr
                        
                        # Check for significant divergence
                        if abs(corr) > 0.7:  # High correlation
                            # Calculate spread
                            spread = returns[s1] - returns[s2]
                            if abs(spread) > 0.02:  # 2% divergence
                                results['divergences'].append({
                                    'pair': f"{s1}_{s2}",
                                    'correlation': corr,
                                    'spread': spread,
                                    'signal': 'LONG_SPREAD' if spread < 0 else 'SHORT_SPREAD'
                                })
                    
                    # Update cointegration test periodically
                    if s1 in prices and s2 in prices:
                        coint_result = self.cointegration_tests[pair].update(
                            prices[s1], prices[s2]
                        )
                        if coint_result and coint_result.get('is_cointegrated'):
                            results['cointegration'][f"{s1}_{s2}"] = coint_result
        
        return results
    
    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get full correlation matrix for all tracked assets."""
        symbols = sorted(list(self.assets))
        n = len(symbols)
        
        if n < 2:
            return None
        
        matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                pair = tuple(sorted([symbols[i], symbols[j]]))
                if pair in self.correlations:
                    corr = self.correlations[pair].get_correlation()
                    if corr is not None:
                        matrix[i, j] = corr
                        matrix[j, i] = corr
        
        return matrix
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Dict]:
        """Find pairs with high correlation."""
        highly_correlated = []
        
        for pair, tracker in self.correlations.items():
            corr = tracker.get_correlation()
            if corr is not None and abs(corr) >= threshold:
                highly_correlated.append({
                    'pair': f"{pair[0]}_{pair[1]}",
                    'correlation': corr,
                    'direction': 'positive' if corr > 0 else 'negative'
                })
        
        return sorted(highly_correlated, key=lambda x: abs(x['correlation']), reverse=True)
    
    def find_hedge_pairs(self, target_symbol: str, min_corr: float = 0.5) -> List[Dict]:
        """Find potential hedge pairs for a given symbol."""
        hedges = []
        
        for pair, tracker in self.correlations.items():
            if target_symbol in pair:
                other = pair[0] if pair[1] == target_symbol else pair[1]
                corr = tracker.get_correlation()
                
                if corr is not None and abs(corr) >= min_corr:
                    hedges.append({
                        'hedge_asset': other,
                        'correlation': corr,
                        'hedge_ratio': -corr if corr < 0 else -1/corr if corr > 0.5 else -1
                    })
        
        return sorted(hedges, key=lambda x: abs(x['correlation']), reverse=True)


if __name__ == "__main__":
    import time
    
    engine = CorrelationMatrixEngine()
    
    # Simulate correlated price updates
    base_btc = 45000
    base_eth = 3200
    
    for i in range(100):
        # Generate somewhat correlated returns
        common_factor = np.random.randn() * 0.01
        btc_return = common_factor + np.random.randn() * 0.005
        eth_return = common_factor * 0.8 + np.random.randn() * 0.008
        
        btc_price = base_btc * (1 + btc_return)
        eth_price = base_eth * (1 + eth_return)
        
        results = engine.update_prices({
            'BTCUSDT': btc_price,
            'ETHUSDT': eth_price
        }, time.time())
        
        if i % 20 == 0 and results['correlations']:
            print(f"Step {i}:")
            for pair, corr in results['correlations'].items():
                print(f"  {pair}: {corr:.4f}")
        
        base_btc = btc_price
        base_eth = eth_price
