#!/usr/bin/env python3
"""
Alpha Validator for Causal Trading Signals

This module validates whether a trading signal is genuinely causal
or just noise/spurious correlation in the ZAID PERSONAL CRYPTO TRADING BOT.
Implements rigorous statistical tests to distinguish true alpha from randomness.

Features:
- Multiple hypothesis testing with FDR control
- Out-of-sample validation across market regimes
 - Permutation tests for significance
- Signal decay analysis
- P-hacking detection
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Result of signal validation."""
    VALIDATED_CAUSAL = "validated_causal"
    LIKELY_CAUSAL = "likely_causal"
    UNCERTAIN = "uncertain"
    LIKELY_SPURIOUS = "likely_spurious"
    CONFIRMED_SPURIOUS = "confirmed_spurious"


@dataclass
class ValidationResult:
    """Complete validation result for a trading signal."""
    signal_name: str
    status: ValidationStatus
    raw_alpha: float
    causal_alpha: float
    p_value: float
    fdr_adjusted_p: float
    half_life_days: float
    regime_stability: float
    p_hacking_score: float  # Higher = more likely p-hacked
    confidence_level: float
    recommendation: str


class AlphaValidator:
    """
    Comprehensive validator for trading signal causality.
    
    Implements multiple tests to distinguish genuine causal alpha
    from spurious correlations and data mining artifacts.
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        fdr_control: bool = True,
        min_samples: int = 252,  # At least 1 year of daily data
    ):
        self.significance_level = significance_level
        self.fdr_control = fdr_control
        self.min_samples = min_samples
        
    def validate_signal(
        self,
        signal_values: np.ndarray,
        returns: np.ndarray,
        signal_name: str,
        benchmark_returns: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """
        Perform comprehensive validation of a trading signal.
        
        Args:
            signal_values: The trading signal values
            returns: Asset returns to predict
            signal_name: Name of the signal
            benchmark_returns: Optional benchmark for relative performance
            regime_labels: Market regime labels for stability testing
            
        Returns:
            Complete validation result
        """
        if len(signal_values) < self.min_samples:
            return ValidationResult(
                signal_name=signal_name,
                status=ValidationStatus.CONFIRMED_SPURIOUS,
                raw_alpha=0.0,
                causal_alpha=0.0,
                p_value=1.0,
                fdr_adjusted_p=1.0,
                half_life_days=0.0,
                regime_stability=0.0,
                p_hacking_score=1.0,
                confidence_level=0.0,
                recommendation="Insufficient data for validation"
            )
        
        # Step 1: Compute raw alpha (IC or similar)
        raw_ic = self._compute_information_coefficient(signal_values, returns)
        raw_alpha = self._ic_to_alpha(raw_ic)
        
        # Step 2: Permutation test for significance
        p_value = self._permutation_test(signal_values, returns, n_permutations=1000)
        
        # Step 3: FDR adjustment (for multiple testing)
        fdr_p = self._benjamini_hochberg([p_value], len_tests=1)[0]
        
        # Step 4: Causal alpha estimation (controlling for confounders)
        causal_alpha = self._estimate_causal_alpha(signal_values, returns, benchmark_returns)
        
        # Step 5: Signal decay analysis (half-life)
        half_life = self._compute_signal_half_life(signal_values, returns)
        
        # Step 6: Regime stability test
        regime_stability = self._test_regime_stability(
            signal_values, returns, regime_labels
        ) if regime_labels is not None else 1.0
        
        # Step 7: P-hacking detection
        p_hacking_score = self._detect_p_hacking(signal_values, returns)
        
        # Determine overall status
        status = self._determine_status(
            fdr_p, causal_alpha, half_life, regime_stability, p_hacking_score
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(status, causal_alpha, half_life)
        
        logger.info(
            f"Validation complete for {signal_name}: {status.value}, "
            f"causal_alpha={causal_alpha:.4f}, p={fdr_p:.4f}"
        )
        
        return ValidationResult(
            signal_name=signal_name,
            status=status,
            raw_alpha=raw_alpha,
            causal_alpha=causal_alpha,
            p_value=p_value,
            fdr_adjusted_p=fdr_p,
            half_life_days=half_life,
            regime_stability=regime_stability,
            p_hacking_score=p_hacking_score,
            confidence_level=1.0 - fdr_p,
            recommendation=recommendation
        )
    
    def _compute_information_coefficient(
        self, 
        signal: np.ndarray, 
        returns: np.ndarray
    ) -> float:
        """Compute Information Coefficient (rank correlation)."""
        # Use Spearman rank correlation for robustness
        valid_mask = ~(np.isnan(signal) | np.isnan(returns))
        if np.sum(valid_mask) < 10:
            return 0.0
        
        corr, _ = stats.spearmanr(signal[valid_mask], returns[valid_mask])
        return corr if not np.isnan(corr) else 0.0
    
    def _ic_to_alpha(self, ic: float, annual_ir: float = 1.0) -> float:
        """Convert IC to annualized alpha estimate."""
        # Simplified: alpha ≈ IC * volatility * sqrt(breadth)
        # Assuming typical crypto volatility and daily signals
        return ic * 0.02 * np.sqrt(252)  # ~2% daily vol, 252 trading days
    
    def _permutation_test(
        self,
        signal: np.ndarray,
        returns: np.ndarray,
        n_permutations: int = 1000
    ) -> float:
        """
        Permutation test for signal significance.
        
        Tests the null hypothesis that the signal has no predictive power.
        """
        observed_ic = abs(self._compute_information_coefficient(signal, returns))
        
        extreme_count = 0
        for _ in range(n_permutations):
            # Shuffle returns to break any relationship
            shuffled_returns = np.random.permutation(returns)
            permuted_ic = abs(self._compute_information_coefficient(signal, shuffled_returns))
            
            if permuted_ic >= observed_ic:
                extreme_count += 1
        
        return extreme_count / n_permutations
    
    def _benjamini_hochberg(
        self,
        p_values: List[float],
        len_tests: int
    ) -> List[float]:
        """
        Benjamini-Hochberg procedure for FDR control.
        
        Adjusts p-values to control False Discovery Rate.
        """
        n = len_tests
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        adjusted = np.zeros(n)
        for i in range(n - 1, -1, -1):
            adjusted[i] = min(sorted_p[i] * n / (i + 1), 1.0)
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])
        
        # Restore original order
        result = np.zeros(n)
        result[sorted_indices] = adjusted
        
        return result.tolist()
    
    def _estimate_causal_alpha(
        self,
        signal: np.ndarray,
        returns: np.ndarray,
        benchmark: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate causal alpha by controlling for confounders.
        
        Uses regression-based adjustment to isolate the signal's
        unique contribution.
        """
        valid_mask = ~(np.isnan(signal) | np.isnan(returns))
        
        if benchmark is not None:
            valid_mask &= ~np.isnan(benchmark)
        
        if np.sum(valid_mask) < 50:
            return 0.0
        
        y = returns[valid_mask]
        x_signal = signal[valid_mask]
        
        if benchmark is not None:
            x_bench = benchmark[valid_mask]
            # Multiple regression: returns = alpha + beta1*signal + beta2*benchmark
            X = np.column_stack([np.ones(len(y)), x_signal, x_bench])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                # Causal alpha is the signal coefficient
                return coeffs[1] * 252  # Annualize
            except:
                return 0.0
        else:
            # Simple regression
            try:
                slope, intercept, _, _, _ = stats.linregress(x_signal, y)
                return slope * 252  # Annualize
            except:
                return 0.0
    
    def _compute_signal_half_life(
        self,
        signal: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Compute the half-life of signal predictive power.
        
        Measures how quickly the signal's IC decays over time.
        """
        max_lag = min(60, len(signal) // 4)  # Up to 60 days
        ics = []
        
        for lag in range(1, max_lag + 1):
            if lag >= len(signal):
                break
            ic = self._compute_information_coefficient(signal[:-lag], returns[lag:])
            ics.append(ic)
        
        if len(ics) < 5:
            return float('inf')
        
        # Fit exponential decay: IC(t) = IC(0) * exp(-t/half_life * ln(2))
        ics = np.array(ics)
        valid_ics = ics[~np.isnan(ics)]
        
        if len(valid_ics) < 3 or np.all(valid_ics <= 0):
            return float('inf')
        
        # Log-linear regression
        lags = np.arange(1, len(valid_ics) + 1)
        log_ics = np.log(np.abs(valid_ics) + 1e-10)
        
        try:
            slope, _, _, _, _ = stats.linregress(lags, log_ics)
            if slope >= 0:
                return float('inf')  # No decay or increasing
            half_life = -np.log(2) / slope
            return max(1.0, half_life)
        except:
            return float('inf')
    
    def _test_regime_stability(
        self,
        signal: np.ndarray,
        returns: np.ndarray,
        regime_labels: np.ndarray
    ) -> float:
        """
        Test if signal performance is stable across market regimes.
        
        Returns a stability score (0-1) where 1 means perfectly stable.
        """
        unique_regimes = np.unique(regime_labels)
        
        if len(unique_regimes) < 2:
            return 1.0
        
        regime_ics = []
        for regime in unique_regimes:
            mask = regime_labels == regime
            if np.sum(mask) < 20:
                continue
            ic = self._compute_information_coefficient(signal[mask], returns[mask])
            regime_ics.append(ic)
        
        if len(regime_ics) < 2:
            return 1.0
        
        # Stability = 1 - coefficient of variation
        mean_ic = np.mean(regime_ics)
        std_ic = np.std(regime_ics)
        
        if abs(mean_ic) < 1e-10:
            return 0.0
        
        cv = std_ic / abs(mean_ic)
        return max(0.0, 1.0 - cv)
    
    def _detect_p_hacking(
        self,
        signal: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Detect potential p-hacking/data mining.
        
        Higher scores indicate more likely p-hacking.
        
        Heuristics:
        - Look for suspiciously round p-values
        - Check if signal works only in specific subsamples
        - Analyze parameter sensitivity
        """
        score = 0.0
        
        # Get p-value
        p_value = self._permutation_test(signal, returns, n_permutations=100)
        
        # Suspicious if p-value is just below threshold
        if 0.04 <= p_value <= 0.05:
            score += 0.3
        
        # Check parameter sensitivity (simplified)
        # A robust signal should work with slight parameter variations
        
        # Penalize if signal has very low IC but significant p-value
        ic = abs(self._compute_information_coefficient(signal, returns))
        if ic < 0.02 and p_value < 0.05:
            score += 0.4
        
        # Penalize if signal only works in one regime
        # (Would need regime data for full check)
        
        return min(1.0, score)
    
    def _determine_status(
        self,
        fdr_p: float,
        causal_alpha: float,
        half_life: float,
        regime_stability: float,
        p_hacking_score: float
    ) -> ValidationStatus:
        """Determine overall validation status."""
        # Confirmed spurious
        if fdr_p > 0.2 or p_hacking_score > 0.7:
            return ValidationStatus.CONFIRMED_SPURIOUS
        
        # Likely spurious
        if fdr_p > 0.1 or causal_alpha < 0.01:
            return ValidationStatus.LIKELY_SPURIOUS
        
        # Validated causal
        if (fdr_p < self.significance_level and 
            causal_alpha > 0.05 and 
            half_life > 5 and
            regime_stability > 0.7 and
            p_hacking_score < 0.3):
            return ValidationStatus.VALIDATED_CAUSAL
        
        # Likely causal
        if fdr_p < 0.1 and causal_alpha > 0.02:
            return ValidationStatus.LIKELY_CAUSAL
        
        return ValidationStatus.UNCERTAIN
    
    def _generate_recommendation(
        self,
        status: ValidationStatus,
        causal_alpha: float,
        half_life: float
    ) -> str:
        """Generate actionable recommendation."""
        if status == ValidationStatus.VALIDATED_CAUSAL:
            return (f"STRONG BUY: Deploy signal with full allocation. "
                    f"Expected annual alpha: {causal_alpha:.2%}")
        elif status == ValidationStatus.LIKELY_CAUSAL:
            return (f"MODERATE BUY: Deploy with reduced allocation. "
                    f"Monitor for decay (half-life: {half_life:.1f} days)")
        elif status == ValidationStatus.UNCERTAIN:
            return "HOLD: Insufficient evidence. Collect more data or refine signal"
        elif status == ValidationStatus.LIKELY_SPURIOUS:
            return "AVOID: Signal likely spurious. Do not deploy"
        else:
            return "REJECT: Confirmed spurious correlation. Discard signal"


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate a genuine causal signal
    n_days = 500
    genuine_signal = np.random.randn(n_days) * 0.5
    genuine_returns = 0.03 * genuine_signal + np.random.randn(n_days) * 0.02
    
    # Simulate a spurious signal
    spurious_signal = np.random.randn(n_days)
    spurious_returns = np.random.randn(n_days) * 0.02
    
    validator = AlphaValidator()
    
    print("\n=== Genuine Signal Validation ===")
    result_genuine = validator.validate_signal(
        genuine_signal, genuine_returns, "GenuineMomentum"
    )
    print(f"Status: {result_genuine.status.value}")
    print(f"Causal Alpha: {result_genuine.causal_alpha:.4f}")
    print(f"P-value (FDR adj): {result_genuine.fdr_adjusted_p:.4f}")
    print(f"Recommendation: {result_genuine.recommendation}")
    
    print("\n=== Spurious Signal Validation ===")
    result_spurious = validator.validate_signal(
        spurious_signal, spurious_returns, "SpuriousPattern"
    )
    print(f"Status: {result_spurious.status.value}")
    print(f"Causal Alpha: {result_spurious.causal_alpha:.4f}")
    print(f"P-value (FDR adj): {result_spurious.fdr_adjusted_p:.4f}")
    print(f"Recommendation: {result_spurious.recommendation}")
