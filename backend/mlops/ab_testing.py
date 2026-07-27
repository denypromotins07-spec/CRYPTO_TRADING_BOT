#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
High-Frequency Feature Stores, MLOps, and Concept Drift
File: backend/mlops/ab_testing.py
Chapter 4: MLOps Pipeline, Shadow Mode Testing, and SOUL.md Logging

Statistically validates shadow models against live production PnL.
Uses sequential hypothesis testing for early detection of superior models.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
Implements rigorous statistical tests for model comparison.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import threading
import time
from scipy import stats


class TestResult(Enum):
    """Result of an A/B test."""
    INCONCLUSIVE = "inconclusive"
    A_WINS = "a_wins"  # Live model wins
    B_WINS = "b_wins"  # Shadow model wins
    NO_DIFFERENCE = "no_difference"


@dataclass
class TradeObservation:
    """A single trade observation for A/B testing."""
    timestamp: float
    model_id: str  # 'live' or 'shadow'
    prediction: float
    actual_return: float
    pnl: float
    position_size: float
    asset_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_id: str
    live_model_id: str
    shadow_model_id: str
    significance_level: float = 0.05  # Alpha
    power: float = 0.80  # 1 - Beta
    min_sample_size: int = 100
    max_sample_size: int = 10000
    effect_size_threshold: float = 0.1  # Minimum detectable effect
    early_stopping: bool = True


@dataclass
class ABTestResult:
    """Result of a completed A/B test."""
    test_id: str
    result: TestResult
    p_value: float
    confidence_interval: Tuple[float, float]
    live_metrics: Dict[str, float]
    shadow_metrics: Dict[str, float]
    sample_size: int
    test_duration_seconds: float
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingEngine:
    """
    Statistical A/B testing engine for shadow vs live model comparison.
    
    Key features:
    1. Sequential hypothesis testing for early stopping
    2. Multiple metric comparison (PnL, Sharpe, win rate)
    3. Confidence interval estimation
    4. Power analysis for sample size determination
    5. Automatic promotion/demotion recommendations
    
    Optimized for high-frequency trading evaluation.
    """
    
    def __init__(
        self,
        default_significance: float = 0.05,
        default_power: float = 0.80,
        memory_budget_mb: int = 64
    ):
        """
        Initialize the A/B testing engine.
        
        Args:
            default_significance: Default significance level (alpha)
            default_power: Default statistical power (1 - beta)
            memory_budget_mb: Memory budget for storing observations
        """
        self.default_significance = default_significance
        self.default_power = default_power
        self.memory_budget_bytes = memory_budget_mb * 1024 * 1024
        
        # Active tests
        self.active_tests: Dict[str, ABTestConfig] = {}
        
        # Observations per test
        # Structure: {test_id: {'live': deque, 'shadow': deque}}
        self.observations: Dict[str, Dict[str, deque]] = {}
        
        # Completed tests
        self.completed_tests: Dict[str, ABTestResult] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'tests_started': 0,
            'tests_completed': 0,
            'total_observations': 0,
            'early_stops': 0
        }
    
    def start_test(
        self,
        config: ABTestConfig
    ) -> bool:
        """
        Start a new A/B test.
        
        Args:
            config: Test configuration
            
        Returns:
            True if test started successfully
        """
        with self._lock:
            if config.test_id in self.active_tests:
                return False
            
            self.active_tests[config.test_id] = config
            self.observations[config.test_id] = {
                'live': deque(maxlen=config.max_sample_size),
                'shadow': deque(maxlen=config.max_sample_size)
            }
            
            self.stats['tests_started'] += 1
            return True
    
    def add_observation(
        self,
        test_id: str,
        observation: TradeObservation
    ) -> Optional[ABTestResult]:
        """
        Add a trade observation to an active test.
        
        Args:
            test_id: Test identifier
            observation: Trade observation
            
        Returns:
            ABTestResult if test completed, None otherwise
        """
        with self._lock:
            if test_id not in self.active_tests:
                return None
            
            config = self.active_tests[test_id]
            obs_data = self.observations[test_id]
            
            # Add to appropriate queue
            if observation.model_id == 'live':
                obs_data['live'].append(observation)
            elif observation.model_id == 'shadow':
                obs_data['shadow'].append(observation)
            else:
                return None
            
            self.stats['total_observations'] += 1
            
            # Check if we have enough data for analysis
            n_live = len(obs_data['live'])
            n_shadow = len(obs_data['shadow'])
            min_n = min(n_live, n_shadow)
            
            if min_n >= config.min_sample_size:
                result = self._analyze_test(test_id, config)
                
                if result is not None:
                    # Test completed
                    self.completed_tests[test_id] = result
                    del self.active_tests[test_id]
                    del self.observations[test_id]
                    
                    self.stats['tests_completed'] += 1
                    if min_n < config.max_sample_size:
                        self.stats['early_stops'] += 1
                    
                    return result
            
            return None
    
    def _analyze_test(
        self,
        test_id: str,
        config: ABTestConfig
    ) -> Optional[ABTestResult]:
        """
        Analyze an A/B test for completion.
        
        Returns:
            ABTestResult if test should complete, None to continue
        """
        obs_data = self.observations[test_id]
        live_obs = list(obs_data['live'])
        shadow_obs = list(obs_data['shadow'])
        
        if not live_obs or not shadow_obs:
            return None
        
        # Extract PnL arrays
        live_pnl = np.array([o.pnl for o in live_obs])
        shadow_pnl = np.array([o.pnl for o in shadow_obs])
        
        # Calculate metrics
        live_metrics = self._calculate_metrics(live_obs)
        shadow_metrics = self._calculate_metrics(shadow_obs)
        
        # Perform statistical tests
        p_value, ci, result = self._compare_means(live_pnl, shadow_pnl, config)
        
        # Check for early stopping
        should_stop = False
        if config.early_stopping:
            should_stop = self._check_early_stopping(
                live_pnl, shadow_pnl, config, p_value
            )
        
        # Check if max sample size reached
        if len(live_pnl) >= config.max_sample_size:
            should_stop = True
        
        if not should_stop:
            return None
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            result, live_metrics, shadow_metrics, config
        )
        
        # Calculate test duration
        all_times = [o.timestamp for o in live_obs + shadow_obs]
        duration = max(all_times) - min(all_times) if all_times else 0
        
        return ABTestResult(
            test_id=test_id,
            result=result,
            p_value=p_value,
            confidence_interval=ci,
            live_metrics=live_metrics,
            shadow_metrics=shadow_metrics,
            sample_size=len(live_pnl) + len(shadow_pnl),
            test_duration_seconds=duration,
            recommendation=recommendation
        )
    
    def _calculate_metrics(self, observations: List[TradeObservation]) -> Dict[str, float]:
        """Calculate performance metrics from observations."""
        if not observations:
            return {}
        
        pnls = np.array([o.pnl for o in observations])
        returns = np.array([o.actual_return for o in observations])
        
        # Win rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)
        
        # Sharpe ratio (annualized, assuming these are per-trade returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        
        return {
            'total_pnl': float(np.sum(pnls)),
            'mean_pnl': float(np.mean(pnls)),
            'std_pnl': float(np.std(pnls)),
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'sample_size': len(observations)
        }
    
    def _compare_means(
        self,
        live_pnl: np.ndarray,
        shadow_pnl: np.ndarray,
        config: ABTestConfig
    ) -> Tuple[float, Tuple[float, float], TestResult]:
        """
        Compare means between live and shadow models.
        
        Returns:
            Tuple of (p_value, confidence_interval, test_result)
        """
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(live_pnl, shadow_pnl, equal_var=False)
        
        # Calculate confidence interval for difference in means
        diff = np.mean(shadow_pnl) - np.mean(live_pnl)
        se = np.sqrt(np.var(live_pnl)/len(live_pnl) + np.var(shadow_pnl)/len(shadow_pnl))
        
        ci_low = diff - stats.t.ppf(1 - config.significance_level/2, 
                                     df=len(live_pnl)+len(shadow_pnl)-2) * se
        ci_high = diff + stats.t.ppf(1 - config.significance_level/2, 
                                      df=len(live_pnl)+len(shadow_pnl)-2) * se
        
        # Determine result
        if p_value < config.significance_level:
            if diff > 0:
                result = TestResult.B_WINS  # Shadow wins
            else:
                result = TestResult.A_WINS  # Live wins
        else:
            result = TestResult.NO_DIFFERENCE
        
        return p_value, (ci_low, ci_high), result
    
    def _check_early_stopping(
        self,
        live_pnl: np.ndarray,
        shadow_pnl: np.ndarray,
        config: ABTestConfig,
        current_p_value: float
    ) -> bool:
        """Check if early stopping criteria are met."""
        # Sequential probability ratio test (SPRT) approximation
        n = min(len(live_pnl), len(shadow_pnl))
        
        # Calculate effect size
        pooled_std = np.sqrt((np.var(live_pnl) + np.var(shadow_pnl)) / 2)
        if pooled_std == 0:
            return False
        
        effect_size = abs(np.mean(shadow_pnl) - np.mean(live_pnl)) / pooled_std
        
        # Early stop if:
        # 1. Significant result with sufficient power
        # 2. Effect size exceeds threshold
        # 3. Very unlikely to change with more data
        
        if current_p_value < config.significance_level and effect_size > config.effect_size_threshold:
            return True
        
        # Futility stopping: if effect is tiny and sample is large
        if n > config.min_sample_size * 2 and effect_size < config.effect_size_threshold / 2:
            return True
        
        return False
    
    def _generate_recommendation(
        self,
        result: TestResult,
        live_metrics: Dict[str, float],
        shadow_metrics: Dict[str, float],
        config: ABTestConfig
    ) -> str:
        """Generate a recommendation based on test results."""
        if result == TestResult.B_WINS:
            # Shadow model wins
            improvement = (shadow_metrics.get('sharpe_ratio', 0) - live_metrics.get('sharpe_ratio', 0))
            if improvement > 0.1:
                return f"PROMOTE: Shadow model shows {improvement:.2f} Sharpe improvement"
            else:
                return "CONSIDER_PROMOTION: Shadow model statistically better but marginal improvement"
        
        elif result == TestResult.A_WINS:
            # Live model wins
            degradation = (live_metrics.get('sharpe_ratio', 0) - shadow_metrics.get('sharpe_ratio', 0))
            if degradation > 0.1:
                return "REJECT: Shadow model significantly worse than live"
            else:
                return "REJECT_WITH_MONITORING: Shadow model slightly worse, monitor for drift"
        
        else:
            return "NO_ACTION: No statistically significant difference detected"
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status of a test."""
        with self._lock:
            if test_id in self.active_tests:
                config = self.active_tests[test_id]
                obs_data = self.observations[test_id]
                
                return {
                    'test_id': test_id,
                    'status': 'active',
                    'live_observations': len(obs_data['live']),
                    'shadow_observations': len(obs_data['shadow']),
                    'min_required': config.min_sample_size,
                    'max_allowed': config.max_sample_size,
                    'progress': min(1.0, (len(obs_data['live']) + len(obs_data['shadow'])) / 
                                   (2 * config.min_sample_size))
                }
            elif test_id in self.completed_tests:
                result = self.completed_tests[test_id]
                return {
                    'test_id': test_id,
                    'status': 'completed',
                    'result': result.result.value,
                    'p_value': result.p_value,
                    'recommendation': result.recommendation
                }
            else:
                return {'test_id': test_id, 'status': 'not_found'}
    
    def get_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tests."""
        with self._lock:
            return {
                **{tid: self.get_test_status(tid) for tid in self.active_tests},
                **{tid: self.get_test_status(tid) for tid in self.completed_tests}
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return {
                **self.stats,
                'active_tests': len(self.active_tests),
                'completed_tests': len(self.completed_tests),
                'memory_estimate_mb': self._estimate_memory_usage() / (1024 * 1024)
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage."""
        total_obs = sum(
            len(obs['live']) + len(obs['shadow'])
            for obs in self.observations.values()
        )
        # Approximate bytes per observation
        return total_obs * 200


# Convenience function for creating production A/B test
def create_production_ab_test(
    live_model_id: str,
    shadow_model_id: str,
    test_name: Optional[str] = None
) -> Tuple[ABTestingEngine, ABTestConfig]:
    """
    Create a production-ready A/B test.
    
    Args:
        live_model_id: ID of the live production model
        shadow_model_id: ID of the shadow model
        test_name: Optional test name (auto-generated if None)
        
    Returns:
        Tuple of (ABTestingEngine, ABTestConfig)
    """
    import uuid
    
    test_id = test_name or f"ab_test_{uuid.uuid4().hex[:8]}"
    
    engine = ABTestingEngine(
        default_significance=0.01,  # More stringent for production
        default_power=0.90,
        memory_budget_mb=64
    )
    
    config = ABTestConfig(
        test_id=test_id,
        live_model_id=live_model_id,
        shadow_model_id=shadow_model_id,
        significance_level=0.01,
        power=0.90,
        min_sample_size=200,
        max_sample_size=5000,
        effect_size_threshold=0.05,
        early_stopping=True
    )
    
    engine.start_test(config)
    
    return engine, config


if __name__ == '__main__':
    # Test the A/B testing engine
    print("Testing A/B Testing Engine for ZAID Trading Bot...")
    
    # Create engine and start test
    engine = ABTestingEngine()
    
    config = ABTestConfig(
        test_id="test_btc_v1_vs_v2",
        live_model_id="btc_model_v1",
        shadow_model_id="btc_model_v2",
        min_sample_size=50,
        max_sample_size=500
    )
    
    engine.start_test(config)
    print(f"\nStarted test: {config.test_id}")
    
    # Simulate observations
    np.random.seed(42)
    
    print("\nSimulating trades...")
    for i in range(100):
        # Live model: baseline performance
        live_pnl = np.random.normal(10, 50)  # Mean $10, std $50
        live_obs = TradeObservation(
            timestamp=time.time(),
            model_id='live',
            prediction=np.random.random(),
            actual_return=live_pnl / 10000,
            pnl=live_pnl,
            position_size=1.0,
            asset_id=0
        )
        
        # Shadow model: slightly better performance
        shadow_pnl = np.random.normal(15, 45)  # Mean $15, std $45
        shadow_obs = TradeObservation(
            timestamp=time.time(),
            model_id='shadow',
            prediction=np.random.random(),
            actual_return=shadow_pnl / 10000,
            pnl=shadow_pnl,
            position_size=1.0,
            asset_id=0
        )
        
        result_live = engine.add_observation(config.test_id, live_obs)
        result_shadow = engine.add_observation(config.test_id, shadow_obs)
        
        if result_live or result_shadow:
            result = result_live or result_shadow
            print(f"\nTest completed at observation {i+1}!")
            print(f"  Result: {result.result.value}")
            print(f"  P-value: {result.p_value:.4f}")
            print(f"  Recommendation: {result.recommendation}")
            break
    
    # Get final status
    status = engine.get_test_status(config.test_id)
    print(f"\nFinal test status: {status}")
    
    # Get stats
    stats = engine.get_stats()
    print(f"\nEngine stats: {stats}")
    
    print("\nA/B Testing Engine test completed!")
