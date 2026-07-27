"""
Simulation Analytics Module

Compares synthetic market statistics with real stylized facts:
- Fat-tailed return distributions
- Volatility clustering
- Volume-price correlation
- Order flow autocorrelation
- Spread dynamics

Validates synthetic data quality for strategy testing.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StylizedFacts:
    """Expected stylized facts for crypto markets."""
    # Return distribution
    return_skewness: float = -0.2       # Slight negative skew
    return_excess_kurtosis: float = 5.0  # Fat tails
    jarque_bera_pvalue_threshold: float = 0.01  # Should reject normality
    
    # Volatility
    volatility_clustering_autocorr: float = 0.7  # High vol persistence
    garch_alpha: float = 0.1            # ARCH term
    garch_beta: float = 0.85            # GARCH term
    
    # Volume
    volume_return_correlation: float = 0.3  # Positive correlation
    volume_autocorr: float = 0.5        # Volume clustering
    
    # Order flow
    order_flow_autocorr: float = 0.6    # Order flow persistence
    trade_sign_autocorr: float = 0.4    # Buy/sell clustering
    
    # Spread
    mean_spread_bps: float = 8.0        # Average spread
    spread_vol_of_vol: float = 0.3      # Spread variability


@dataclass
class SimulationQualityReport:
    """Quality assessment of synthetic data."""
    overall_score: float              # 0-1 quality score
    return_distribution_score: float
    volatility_score: float
    volume_score: float
    order_flow_score: float
    spread_score: float
    issues: List[str]
    warnings: List[str]


class SimulationAnalytics:
    """
    Analyze and validate synthetic market simulation quality.
    
    Compares synthetic data against known stylized facts
    of cryptocurrency markets.
    """
    
    def __init__(self, expected_facts: Optional[StylizedFacts] = None):
        self.expected = expected_facts or StylizedFacts()
        self.real_data_stats: Dict[str, Any] = {}
        self.synthetic_data_stats: Dict[str, Any] = {}
    
    def set_real_data_statistics(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        spreads: np.ndarray,
        order_signs: Optional[np.ndarray] = None
    ) -> None:
        """Calculate statistics from real market data."""
        self.real_data_stats = {
            'returns': returns,
            'volumes': volumes,
            'spreads': spreads,
            'order_signs': order_signs,
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_skew': stats.skew(returns),
            'return_kurtosis': stats.kurtosis(returns),
            'vol_clustering': self._calculate_vol_clustering(returns),
            'volume_return_corr': np.corrcoef(returns[:-1], volumes[1:])[0, 1] if len(returns) > 1 else 0,
            'volume_autocorr': self._autocorrelation(volumes, lag=1),
            'mean_spread': np.mean(spreads),
            'spread_std': np.std(spreads),
        }
        
        if order_signs is not None:
            self.real_data_stats['order_flow_autocorr'] = self._autocorrelation(order_signs, lag=1)
    
    def analyze_synthetic_data(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        spreads: np.ndarray,
        order_signs: Optional[np.ndarray] = None
    ) -> SimulationQualityReport:
        """Analyze synthetic data and compare to stylized facts."""
        issues = []
        warnings = []
        scores = {}
        
        # 1. Return distribution analysis
        ret_skew = stats.skew(returns)
        ret_kurt = stats.kurtosis(returns)
        
        skew_diff = abs(ret_skew - self.expected.return_skewness)
        kurt_diff = abs(ret_kurt - self.expected.return_excess_kurtosis)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        if jb_pvalue > self.expected.jarque_bera_pvalue_threshold:
            warnings.append("Returns may be too close to normal (JB p-value high)")
        
        return_score = max(0, 1.0 - skew_diff * 0.5 - kurt_diff * 0.1)
        scores['return_distribution'] = return_score
        
        # 2. Volatility clustering
        vol_clustering = self._calculate_vol_clustering(returns)
        vol_expected = self.expected.volatility_clustering_autocorr
        vol_diff = abs(vol_clustering - vol_expected)
        
        vol_score = max(0, 1.0 - vol_diff * 2.0)
        if vol_clustering < 0.3:
            issues.append("Volatility clustering too weak")
        scores['volatility'] = vol_score
        
        # 3. Volume analysis
        if len(volumes) > 1:
            vol_ret_corr = np.corrcoef(returns[:-1], volumes[1:])[0, 1]
            vol_auto = self._autocorrelation(volumes, lag=1)
            
            corr_diff = abs(vol_ret_corr - self.expected.volume_return_correlation)
            auto_diff = abs(vol_auto - self.expected.volume_autocorr)
            
            volume_score = max(0, 1.0 - corr_diff - auto_diff * 0.5)
            
            if vol_ret_corr < 0:
                warnings.append("Volume-return correlation is negative (unusual)")
        else:
            volume_score = 0.5
            warnings.append("Insufficient volume data")
        
        scores['volume'] = volume_score
        
        # 4. Order flow analysis
        if order_signs is not None and len(order_signs) > 10:
            order_auto = self._autocorrelation(order_signs, lag=1)
            order_diff = abs(order_auto - self.expected.order_flow_autocorr)
            order_score = max(0, 1.0 - order_diff * 2.0)
            
            if order_auto < 0.2:
                warnings.append("Order flow autocorrelation lower than expected")
        else:
            order_score = 0.5
            warnings.append("Insufficient order flow data")
        
        scores['order_flow'] = order_score
        
        # 5. Spread analysis
        spread_mean = np.mean(spreads) * 10000  # Convert to bps
        spread_diff = abs(spread_mean - self.expected.mean_spread_bps) / self.expected.mean_spread_bps
        spread_score = max(0, 1.0 - spread_diff)
        
        if spread_mean > self.expected.mean_spread_bps * 2:
            warnings.append("Spread significantly wider than typical")
        
        scores['spread'] = spread_score
        
        # Calculate overall score
        overall_score = np.mean(list(scores.values()))
        
        # Add specific issues
        if ret_kurt < 2:
            issues.append("Return distribution lacks fat tails (kurtosis too low)")
        if vol_clustering < 0.4:
            issues.append("Missing volatility clustering effect")
        
        return SimulationQualityReport(
            overall_score=overall_score,
            return_distribution_score=scores.get('return_distribution', 0),
            volatility_score=scores.get('volatility', 0),
            volume_score=scores.get('volume', 0),
            order_flow_score=scores.get('order_flow', 0),
            spread_score=scores.get('spread', 0),
            issues=issues,
            warnings=warnings,
        )
    
    def _calculate_vol_clustering(self, returns: np.ndarray, window: int = 20) -> float:
        """Calculate volatility clustering via absolute return autocorrelation."""
        abs_returns = np.abs(returns)
        return self._autocorrelation(abs_returns, lag=1)
    
    def _autocorrelation(self, x: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(x) <= lag:
            return 0.0
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)
        if var == 0:
            return 0.0
        
        autocov = np.mean((x[:n-lag] - mean) * (x[lag:] - mean))
        return autocov / var
    
    def compare_distributions(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        test_type: str = 'ks'
    ) -> Dict[str, float]:
        """
        Statistical comparison between real and synthetic distributions.
        
        Args:
            real_data: Real market data
            synthetic_data: Synthetic data to validate
            test_type: 'ks' (Kolmogorov-Smirnov) or 'ad' (Anderson-Darling)
            
        Returns:
            Dictionary with test statistics
        """
        results = {}
        
        if test_type == 'ks':
            stat, pvalue = stats.ks_2samp(real_data, synthetic_data)
            results['ks_statistic'] = stat
            results['ks_pvalue'] = pvalue
            results['distributions_similar'] = pvalue > 0.05
        elif test_type == 'ad':
            # Anderson-Darling requires combining samples
            result = stats.anderson_ksamp([real_data, synthetic_data])
            results['ad_statistic'] = result.statistic
            results['ad_pvalue'] = result.pvalue
            results['distributions_similar'] = result.pvalue > 0.05
        
        # Additional moment comparisons
        results['mean_diff'] = abs(np.mean(real_data) - np.mean(synthetic_data))
        results['std_ratio'] = np.std(synthetic_data) / (np.std(real_data) + 1e-10)
        results['skew_diff'] = abs(stats.skew(real_data) - stats.skew(synthetic_data))
        results['kurt_diff'] = abs(stats.kurtosis(real_data) - stats.kurtosis(synthetic_data))
        
        return results
    
    def validate_tail_behavior(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Validate tail behavior matches crypto characteristics."""
        n = len(returns)
        
        # Calculate empirical quantiles
        lower_quantile = np.percentile(returns, (1 - confidence) / 2 * 100)
        upper_quantile = np.percentile(returns, (1 + confidence) / 2 * 100)
        
        # Expected tail indices for crypto (typically 2-4)
        left_tail = returns[returns < lower_quantile]
        right_tail = returns[returns > upper_quantile]
        
        # Hill estimator for tail index (simplified)
        left_tail_index = self._estimate_tail_index(-left_tail) if len(left_tail) > 0 else None
        right_tail_index = self._estimate_tail_index(right_tail) if len(right_tail) > 0 else None
        
        # Check for asymmetry
        tail_asymmetry = abs(left_tail_index - right_tail_index) if (
            left_tail_index and right_tail_index
        ) else None
        
        return {
            'lower_quantile': lower_quantile,
            'upper_quantile': upper_quantile,
            'left_tail_index': left_tail_index,
            'right_tail_index': right_tail_index,
            'tail_asymmetry': tail_asymmetry,
            'expected_tail_range': (2.0, 4.0),
            'tails_too_thin': (
                (left_tail_index and left_tail_index > 5) or
                (right_tail_index and right_tail_index > 5)
            ),
        }
    
    def _estimate_tail_index(self, tail_returns: np.ndarray, k: int = 50) -> Optional[float]:
        """Simple Hill estimator for tail index."""
        if len(tail_returns) < k:
            return None
        
        sorted_returns = np.sort(tail_returns)[::-1][:k]
        log_returns = np.log(sorted_returns)
        
        # Hill estimator
        tail_index = 1.0 / np.mean(log_returns - log_returns[-1])
        
        return tail_index
    
    def generate_quality_report(
        self,
        synthetic_returns: np.ndarray,
        synthetic_volumes: np.ndarray,
        synthetic_spreads: np.ndarray
    ) -> str:
        """Generate human-readable quality report."""
        report = self.analyze_synthetic_data(
            synthetic_returns, synthetic_volumes, synthetic_spreads
        )
        
        lines = [
            "=" * 60,
            "SYNTHETIC DATA QUALITY REPORT",
            "=" * 60,
            f"Overall Score: {report.overall_score:.2f}/1.00",
            "",
            "Component Scores:",
            f"  - Return Distribution: {report.return_distribution_score:.2f}",
            f"  - Volatility Clustering: {report.volatility_score:.2f}",
            f"  - Volume Dynamics: {report.volume_score:.2f}",
            f"  - Order Flow: {report.order_flow_score:.2f}",
            f"  - Spread Dynamics: {report.spread_score:.2f}",
            "",
        ]
        
        if report.issues:
            lines.append("CRITICAL ISSUES:")
            for issue in report.issues:
                lines.append(f"  ! {issue}")
            lines.append("")
        
        if report.warnings:
            lines.append("WARNINGS:")
            for warning in report.warnings:
                lines.append(f"  * {warning}")
            lines.append("")
        
        if report.overall_score >= 0.8 and not report.issues:
            lines.append("STATUS: PASSED - Synthetic data suitable for strategy testing")
        elif report.overall_score >= 0.6:
            lines.append("STATUS: CONDITIONAL - Review warnings before use")
        else:
            lines.append("STATUS: FAILED - Synthetic data does not match market characteristics")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    analytics = SimulationAnalytics()
    
    # Generate sample synthetic data
    np.random.seed(42)
    n = 10000
    
    # GARCH-like returns with fat tails
    vol = np.zeros(n)
    returns = np.zeros(n)
    vol[0] = 0.01
    
    for i in range(1, n):
        vol[i] = np.sqrt(0.000001 + 0.1 * returns[i-1]**2 + 0.85 * vol[i-1]**2)
        returns[i] = vol[i] * np.random.standard_t(4)  # Fat tails
    
    volumes = np.random.exponential(100, n)
    spreads = 0.001 + 0.0005 * np.random.exponential(1, n)
    
    # Analyze
    result = analytics.analyze_synthetic_data(returns, volumes, spreads)
    print(f"Overall Quality Score: {result.overall_score:.2f}")
    print(f"Issues: {result.issues}")
    print(f"Warnings: {result.warnings}")
    
    # Validate tails
    tail_analysis = analytics.validate_tail_behavior(returns)
    print(f"\nTail Analysis:")
    print(f"  Left tail index: {tail_analysis['left_tail_index']}")
    print(f"  Right tail index: {tail_analysis['right_tail_index']}")
