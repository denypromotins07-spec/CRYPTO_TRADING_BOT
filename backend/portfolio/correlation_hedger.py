"""
Correlation Hedger: Cross-asset hedge execution during high correlation regimes.
Distinguishes spurious correlations from true cointegration relationships.
Implements dynamic hedging to reduce portfolio risk.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CorrelationPair:
    """Represents correlation between two assets."""
    asset1: str
    asset2: str
    correlation: float
    p_value: float
    is_significant: bool
    sample_size: int
    last_updated: float


@dataclass
class CointegrationTest:
    """Results of cointegration test between two assets."""
    asset1: str
    asset2: str
    is_cointegrated: bool
    test_statistic: float
    critical_value: float
    p_value: float
    hedge_ratio: float
    half_life: float  # Mean reversion half-life in periods


@dataclass
class HedgePosition:
    """Represents an active hedge position."""
    primary_asset: str
    hedge_asset: str
    hedge_ratio: float
    primary_quantity: float
    hedge_quantity: float
    entry_correlation: float
    is_active: bool = True
    pnl: float = 0.0


class CorrelationHedger:
    """
    Cross-asset correlation hedger for portfolio risk reduction.
    Identifies genuine cointegration relationships vs spurious correlations.
    Executes dynamic hedges during high correlation regimes.
    """
    
    SUPPORTED_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    def __init__(
        self,
        correlation_threshold: float = 0.7,
        significance_level: float = 0.05,
        min_sample_size: int = 30,
        max_hedge_ratio: float = 2.0,
    ):
        self.correlation_threshold = correlation_threshold
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.max_hedge_ratio = max_hedge_ratio
        
        # Price history for correlation calculation
        self._price_history: Dict[str, deque] = {
            asset: deque(maxlen=252) for asset in self.SUPPORTED_ASSETS
        }
        
        # Correlation cache
        self._correlations: Dict[Tuple[str, str], CorrelationPair] = {}
        
        # Cointegration cache
        self._cointegration_results: Dict[Tuple[str, str], CointegrationTest] = {}
        
        # Active hedges
        self._active_hedges: List[HedgePosition] = []
        
        # Portfolio state
        self._portfolio_positions: Dict[str, float] = {}
        
        logger.info("CorrelationHedger initialized")
    
    def update_price(self, asset: str, price: float) -> None:
        """Update price and recalculate correlations if needed."""
        if asset not in self._price_history:
            return
        
        self._price_history[asset].append(price)
        
        # Update correlations if we have enough data
        if len(self._price_history[asset]) >= self.min_sample_size:
            self._update_correlations(asset)
    
    def _update_correlations(self, updated_asset: str) -> None:
        """Update correlation calculations for an asset."""
        for other_asset in self.SUPPORTED_ASSETS:
            if other_asset == updated_asset:
                continue
            
            pair_key = tuple(sorted([updated_asset, other_asset]))
            
            # Get aligned price series
            prices1 = list(self._price_history[updated_asset])
            prices2 = list(self._price_history[other_asset])
            
            # Align lengths
            min_len = min(len(prices1), len(prices2))
            if min_len < self.min_sample_size:
                continue
            
            prices1 = prices1[-min_len:]
            prices2 = prices2[-min_len:]
            
            # Calculate returns
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]
            
            # Calculate correlation
            if len(returns1) > 2 and len(returns2) > 2:
                corr, p_value = stats.pearsonr(returns1, returns2)
                
                is_significant = p_value < self.significance_level
                
                self._correlations[pair_key] = CorrelationPair(
                    asset1=pair_key[0],
                    asset2=pair_key[1],
                    correlation=float(corr),
                    p_value=float(p_value),
                    is_significant=is_significant,
                    sample_size=len(returns1),
                    last_updated=self._get_timestamp()
                )
    
    def test_cointegration(self, asset1: str, asset2: str) -> Optional[CointegrationTest]:
        """
        Test for cointegration between two assets using Engle-Granger method.
        Distinguishes true cointegration from spurious correlation.
        """
        if asset1 not in self._price_history or asset2 not in self._price_history:
            return None
        
        prices1 = np.array(list(self._price_history[asset1]))
        prices2 = np.array(list(self._price_history[asset2]))
        
        # Need sufficient data
        if len(prices1) < 60 or len(prices2) < 60:
            return None
        
        # Align lengths
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]
        
        try:
            # Step 1: Estimate long-run relationship (OLS)
            # y = alpha + beta * x + epsilon
            X = sm.add_constant(prices1)  # type: ignore
            model = sm.OLS(prices2, X).fit()  # type: ignore
            
            hedge_ratio = model.params[1]  # Beta coefficient
            residuals = model.resid
            
            # Step 2: ADF test on residuals
            adf_result = stats.adfuller(residuals, maxlag=10)
            test_statistic = adf_result[0]
            p_value = adf_result[1]
            critical_values = adf_result[4]
            
            # Check if cointegrated (test stat < critical value at 5%)
            critical_5pct = critical_values.get('5%', -2.86)
            is_cointegrated = test_statistic < critical_5pct
            
            # Estimate half-life of mean reversion
            half_life = self._estimate_half_life(residuals)
            
            result = CointegrationTest(
                asset1=asset1,
                asset2=asset2,
                is_cointegrated=is_cointegrated,
                test_statistic=float(test_statistic),
                critical_value=float(critical_5pct),
                p_value=float(p_value),
                hedge_ratio=float(hedge_ratio),
                half_life=half_life
            )
            
            pair_key = tuple(sorted([asset1, asset2]))
            self._cointegration_results[pair_key] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"Cointegration test failed for {asset1}-{asset2}: {e}")
            return None
    
    def _estimate_half_life(self, residuals: np.ndarray) -> float:
        """Estimate mean reversion half-life from residuals."""
        if len(residuals) < 10:
            return float('inf')
        
        try:
            # Fit AR(1) model to residuals
            y = residuals[1:]
            x = residuals[:-1]
            
            slope, intercept, _, _, _ = stats.linregress(x, y)
            
            if abs(slope) >= 1:
                return float('inf')  # Not mean reverting
            
            # Half-life = -ln(2) / ln(slope)
            half_life = -np.log(2) / np.log(abs(slope))
            
            return max(0.1, half_life)
            
        except Exception:
            return float('inf')
    
    def identify_hedge_opportunities(self) -> List[Dict[str, Any]]:
        """Identify potential hedging opportunities based on correlations."""
        opportunities = []
        
        for pair_key, corr_pair in self._correlations.items():
            if not corr_pair.is_significant:
                continue
            
            asset1, asset2 = corr_pair.asset1, corr_pair.asset2
            
            # High positive correlation: can hedge with opposite positions
            if corr_pair.correlation > self.correlation_threshold:
                opportunities.append({
                    "type": "correlation_hedge",
                    "asset1": asset1,
                    "asset2": asset2,
                    "correlation": corr_pair.correlation,
                    "hedge_direction": "opposite",
                    "confidence": 1 - corr_pair.p_value
                })
            
            # High negative correlation: natural hedge exists
            elif corr_pair.correlation < -self.correlation_threshold:
                opportunities.append({
                    "type": "natural_hedge",
                    "asset1": asset1,
                    "asset2": asset2,
                    "correlation": corr_pair.correlation,
                    "hedge_direction": "same",
                    "confidence": 1 - corr_pair.p_value
                })
        
        # Check cointegration-based opportunities
        for pair_key, coint_test in self._cointegration_results.items():
            if coint_test.is_cointegrated and coint_test.half_life < 50:
                opportunities.append({
                    "type": "cointegration_hedge",
                    "asset1": coint_test.asset1,
                    "asset2": coint_test.asset2,
                    "hedge_ratio": coint_test.hedge_ratio,
                    "half_life": coint_test.half_life,
                    "mean_reversion_strength": 1 / coint_test.half_life if coint_test.half_life > 0 else 0
                })
        
        return opportunities
    
    def execute_hedge(
        self,
        primary_asset: str,
        hedge_asset: str,
        primary_quantity: float,
        hedge_ratio: Optional[float] = None
    ) -> Optional[HedgePosition]:
        """Execute a hedge position."""
        # Determine hedge ratio
        if hedge_ratio is None:
            pair_key = tuple(sorted([primary_asset, hedge_asset]))
            if pair_key in self._cointegration_results:
                hedge_ratio = self._cointegration_results[pair_key].hedge_ratio
            elif pair_key in self._correlations:
                # Use correlation as proxy
                corr = self._correlations[pair_key].correlation
                hedge_ratio = abs(corr)
            else:
                hedge_ratio = 1.0
        
        # Enforce maximum hedge ratio
        hedge_ratio = min(abs(hedge_ratio), self.max_hedge_ratio)
        
        # Calculate hedge quantity
        hedge_quantity = primary_quantity * hedge_ratio
        
        # Create hedge position
        hedge = HedgePosition(
            primary_asset=primary_asset,
            hedge_asset=hedge_asset,
            hedge_ratio=hedge_ratio,
            primary_quantity=primary_quantity,
            hedge_quantity=hedge_quantity,
            entry_correlation=self._correlations.get(
                tuple(sorted([primary_asset, hedge_asset])), 
                CorrelationPair("", "", 0, 1, False, 0, 0)
            ).correlation
        )
        
        self._active_hedges.append(hedge)
        
        logger.info(f"Hedge executed: {primary_quantity} {primary_asset} vs "
                   f"{hedge_quantity} {hedge_asset} (ratio: {hedge_ratio:.2f})")
        
        return hedge
    
    def adjust_hedges(self, new_correlations: Dict[Tuple[str, str], float]) -> List[Dict[str, Any]]:
        """Adjust existing hedges based on changing correlations."""
        adjustments = []
        
        for hedge in self._active_hedges:
            if not hedge.is_active:
                continue
            
            pair_key = tuple(sorted([hedge.primary_asset, hedge.hedge_asset]))
            current_corr = new_correlations.get(pair_key, hedge.entry_correlation)
            
            # Check if hedge is still effective
            corr_change = abs(current_corr - hedge.entry_correlation)
            
            if corr_change > 0.3:  # Significant correlation change
                # May need to adjust hedge ratio
                adjustments.append({
                    "hedge": {
                        "primary": hedge.primary_asset,
                        "hedge": hedge.hedge_asset
                    },
                    "reason": "correlation_drift",
                    "old_correlation": hedge.entry_correlation,
                    "new_correlation": current_corr,
                    "action": "recalculate_ratio"
                })
        
        return adjustments
    
    def close_hedge(self, hedge_index: int) -> Dict[str, Any]:
        """Close an active hedge position."""
        if hedge_index < 0 or hedge_index >= len(self._active_hedges):
            return {"status": "error", "message": "Invalid hedge index"}
        
        hedge = self._active_hedges[hedge_index]
        hedge.is_active = False
        
        # Calculate PnL (simplified)
        hedge.pnl = self._calculate_hedge_pnl(hedge)
        
        logger.info(f"Hedge closed: {hedge.primary_asset}/{hedge.hedge_asset}, PnL: {hedge.pnl:.2f}")
        
        return {
            "status": "closed",
            "primary_asset": hedge.primary_asset,
            "hedge_asset": hedge.hedge_asset,
            "pnl": hedge.pnl
        }
    
    def _calculate_hedge_pnl(self, hedge: HedgePosition) -> float:
        """Calculate PnL for a hedge position."""
        # Simplified PnL calculation
        # In production, would use actual fill prices and current market prices
        
        if not hedge.primary_asset in self._price_history or \
           not hedge.hedge_asset in self._price_history:
            return 0.0
        
        # Get current prices
        primary_price = self._price_history[hedge.primary_asset][-1]
        hedge_price = self._price_history[hedge.hedge_asset][-1]
        
        # Calculate PnL (assuming long primary, short hedge)
        primary_pnl = hedge.primary_quantity * (primary_price - primary_price * 0.98)  # Example
        hedge_pnl = -hedge.hedge_quantity * (hedge_price - hedge_price * 0.99)  # Example
        
        return primary_pnl + hedge_pnl
    
    def get_portfolio_hedge_ratio(self) -> float:
        """Calculate overall portfolio hedge ratio."""
        if not self._active_hedges:
            return 0.0
        
        total_primary = sum(h.primary_quantity for h in self._active_hedges if h.is_active)
        total_hedge = sum(h.hedge_quantity for h in self._active_hedges if h.is_active)
        
        if total_primary == 0:
            return 0.0
        
        return total_hedge / total_primary
    
    def get_active_hedges(self) -> List[Dict[str, Any]]:
        """Get all active hedge positions."""
        return [
            {
                "primary_asset": h.primary_asset,
                "hedge_asset": h.hedge_asset,
                "hedge_ratio": h.hedge_ratio,
                "primary_quantity": h.primary_quantity,
                "hedge_quantity": h.hedge_quantity,
                "entry_correlation": h.entry_correlation,
                "is_active": h.is_active,
                "pnl": h.pnl
            }
            for h in self._active_hedges
            if h.is_active
        ]
    
    @staticmethod
    def _get_timestamp() -> float:
        """Get current timestamp."""
        import time
        return time.time()


# Import statsmodels for cointegration test
try:
    import statsmodels.api as sm
except ImportError:
    logger.warning("statsmodels not available, cointegration tests will be limited")
    sm = None  # type: ignore


# Singleton instance
_hedger_instance: Optional[CorrelationHedger] = None


def get_correlation_hedger() -> CorrelationHedger:
    """Get singleton instance of CorrelationHedger."""
    global _hedger_instance
    if _hedger_instance is None:
        _hedger_instance = CorrelationHedger()
    return _hedger_instance


if __name__ == "__main__":
    # Example usage
    hedger = get_correlation_hedger()
    
    # Simulate price updates
    import random
    base_price = 50000
    for i in range(100):
        hedger.update_price("BTCUSDT", base_price * (1 + random.gauss(0, 0.02)))
        hedger.update_price("ETHUSDT", base_price * 0.06 * (1 + random.gauss(0, 0.03)))
        hedger.update_price("SOLUSDT", base_price * 0.002 * (1 + random.gauss(0, 0.05)))
    
    # Get correlations
    print("Correlations:")
    for pair, corr in hedger._correlations.items():
        print(f"  {pair}: {corr.correlation:.3f} (p={corr.p_value:.4f})")
    
    # Test cointegration
    coint_result = hedger.test_cointegration("BTCUSDT", "ETHUSDT")
    if coint_result:
        print(f"\nCointegration BTC-ETH: {coint_result.is_cointegrated}")
        print(f"  Hedge ratio: {coint_result.hedge_ratio:.4f}")
        print(f"  Half-life: {coint_result.half_life:.1f} periods")
    
    # Identify opportunities
    opportunities = hedger.identify_hedge_opportunities()
    print(f"\nHedge opportunities: {len(opportunities)}")
    for opp in opportunities[:3]:
        print(f"  {opp['type']}: {opp.get('asset1', '')}-{opp.get('asset2', '')}")
    
    print("\nCorrelation Hedger module initialized successfully.")
