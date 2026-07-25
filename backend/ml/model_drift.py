"""
Model Drift Detection for ML Ensemble
Monitors feature distributions and triggers retraining when drift exceeds thresholds.
Optimized for 8GB RAM with incremental computation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from collections import deque
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift detection parameters."""
    # Statistical test parameters
    ks_test_alpha: float = 0.01  # Significance level for Kolmogorov-Smirnov
    psi_threshold: float = 0.2   # Population Stability Index threshold
    js_threshold: float = 0.1    # Jensen-Shannon divergence threshold
    
    # Window sizes
    reference_window: int = 1000  # Historical reference window
    monitoring_window: int = 100  # Current monitoring window
    
    # Feature-specific thresholds (override global)
    feature_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Cooldown period after retrain trigger
    retrain_cooldown: int = 500
    
    # Maximum history to store
    max_history: int = 10000


@dataclass
class DriftMetrics:
    """Drift metrics for a single feature."""
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    psi_value: float
    js_divergence: float
    is_drifting: bool
    drift_severity: str  # "none", "warning", "critical"


class ModelDriftDetector:
    """
    Model Drift Detection System.
    
    Implements:
    - Kolmogorov-Smirnov test for distribution shift
    - Population Stability Index (PSI) for bin-based comparison
    - Jensen-Shannon divergence for symmetric KL-like measure
    - Ensemble-level drift aggregation
    - Automatic disable on critical drift
    
    All computations are incremental with bounded memory.
    """
    
    def __init__(self, config: DriftConfig, feature_names: List[str]):
        self.config = config
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        # Reference distributions (historical baseline)
        self.reference_data: Dict[str, deque] = {
            name: deque(maxlen=config.reference_window)
            for name in feature_names
        }
        
        # Monitoring window (recent data)
        self.monitoring_data: Dict[str, deque] = {
            name: deque(maxlen=config.monitoring_window)
            for name in feature_names
        }
        
        # Reference statistics (pre-computed)
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        
        # Drift history
        self.drift_history: deque = deque(maxlen=config.max_history)
        
        # Ensemble state
        self.ensemble_enabled: bool = True
        self.disable_reason: Optional[str] = None
        self.last_retrain_step: int = 0
        self.current_step: int = 0
        
        # Critical features that should never drift
        self.critical_features: set = set()
        
        logger.info(f"ModelDriftDetector initialized: {self.n_features} features")
    
    def update_reference(self, x: Dict[str, float]) -> None:
        """
        Update reference distribution with new observation.
        
        Call this during initial training or after successful retrain.
        
        Args:
            x: Dictionary mapping feature names to values
        """
        for name in self.feature_names:
            if name in x:
                self.reference_data[name].append(x[name])
        
        # Recompute reference statistics periodically
        if len(list(self.reference_data.values())[0]) % 100 == 0:
            self._update_reference_stats()
    
    def update_monitoring(self, x: Dict[str, float]) -> None:
        """
        Update monitoring window with new observation.
        
        Call this for every incoming observation during inference.
        
        Args:
            x: Dictionary mapping feature names to values
        """
        self.current_step += 1
        
        for name in self.feature_names:
            if name in x:
                self.monitoring_data[name].append(x[name])
    
    def _update_reference_stats(self) -> None:
        """Pre-compute reference statistics for efficiency."""
        self.reference_stats = {}
        
        for name in self.feature_names:
            data = list(self.reference_data[name])
            if len(data) < 10:
                continue
            
            arr = np.array(data)
            self.reference_stats[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
    
    def compute_drift_metrics(self, feature_name: str) -> Optional[DriftMetrics]:
        """
        Compute all drift metrics for a single feature.
        
        Args:
            feature_name: Name of feature to analyze
        
        Returns:
            DriftMetrics object or None if insufficient data
        """
        ref_data = list(self.reference_data.get(feature_name, []))
        mon_data = list(self.monitoring_data.get(feature_name, []))
        
        if len(ref_data) < 50 or len(mon_data) < 20:
            return None
        
        ref_arr = np.array(ref_data)
        mon_arr = np.array(mon_data)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(ref_arr, mon_arr)
        
        # Population Stability Index
        psi_val = self._compute_psi(ref_arr, mon_arr)
        
        # Jensen-Shannon divergence
        js_div = self._compute_js_divergence(ref_arr, mon_arr)
        
        # Determine if drifting
        threshold = self.config.feature_thresholds.get(
            feature_name,
            self.config.psi_threshold
        )
        
        is_drifting = psi_val > threshold or ks_pval < self.config.ks_test_alpha
        
        # Determine severity
        if psi_val > threshold * 2 or ks_pval < self.config.ks_test_alpha / 10:
            severity = "critical"
        elif psi_val > threshold or ks_pval < self.config.ks_test_alpha:
            severity = "warning"
        else:
            severity = "none"
        
        # Check critical features
        if feature_name in self.critical_features and is_drifting:
            severity = "critical"
        
        return DriftMetrics(
            feature_name=feature_name,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            psi_value=float(psi_val),
            js_divergence=float(js_div),
            is_drifting=is_drifting,
            drift_severity=severity,
        )
    
    def _compute_psi(self, reference: np.ndarray, monitoring: np.ndarray) -> float:
        """
        Compute Population Stability Index.
        
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        
        Values:
        - < 0.1: No significant change
        - 0.1-0.2: Moderate change
        - > 0.2: Significant change
        """
        # Create bins based on reference distribution
        percentiles = np.linspace(0, 100, 11)  # 10 bins
        bins = np.percentile(reference, percentiles)
        bins = np.unique(bins)  # Remove duplicates
        
        if len(bins) < 3:
            return 0.0
        
        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bins)
        mon_hist, _ = np.histogram(monitoring, bins=bins)
        
        # Convert to proportions with smoothing
        ref_prop = (ref_hist + 1e-6) / (len(reference) + 1e-6 * len(bins))
        mon_prop = (mon_hist + 1e-6) / (len(monitoring) + 1e-6 * len(bins))
        
        # Ensure no zeros
        ref_prop = np.clip(ref_prop, 1e-6, 1.0)
        mon_prop = np.clip(mon_prop, 1e-6, 1.0)
        
        # Compute PSI
        psi = np.sum((mon_prop - ref_prop) * np.log(mon_prop / ref_prop))
        
        return float(psi)
    
    def _compute_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.
        
        Symmetric version of KL divergence, bounded [0, 1].
        """
        # Create common histogram
        all_values = np.concatenate([p, q])
        bins = np.linspace(np.min(all_values), np.max(all_values), 50)
        
        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)
        
        # Normalize
        p_norm = p_hist / (p_hist.sum() + 1e-10)
        q_norm = q_hist / (q_hist.sum() + 1e-10)
        
        # Average distribution
        m = 0.5 * (p_norm + q_norm)
        
        # JS divergence
        js = 0.5 * (
            np.sum(p_norm * np.log((p_norm + 1e-10) / (m + 1e-10))) +
            np.sum(q_norm * np.log((q_norm + 1e-10) / (m + 1e-10)))
        )
        
        return float(js)
    
    def check_all_features(self) -> Dict[str, DriftMetrics]:
        """Check drift for all features."""
        metrics = {}
        
        for name in self.feature_names:
            metric = self.compute_drift_metrics(name)
            if metric:
                metrics[name] = metric
        
        return metrics
    
    def get_aggregate_drift_score(self) -> float:
        """Get aggregate drift score across all features."""
        metrics = self.check_all_features()
        
        if not metrics:
            return 0.0
        
        # Weighted average of PSI values
        total_psi = sum(m.psi_value for m in metrics.values())
        
        return total_psi / len(metrics)
    
    def should_disable_ensemble(self) -> Tuple[bool, Optional[str]]:
        """
        Determine if ensemble should be disabled due to drift.
        
        Returns:
            Tuple of (should_disable, reason)
        """
        if not self.ensemble_enabled:
            return False, None
        
        # Check cooldown
        if self.current_step - self.last_retrain_step < self.config.retrain_cooldown:
            return False, None
        
        metrics = self.check_all_features()
        
        # Count critical and warning drifts
        n_critical = sum(1 for m in metrics.values() if m.drift_severity == "critical")
        n_warning = sum(1 for m in metrics.values() if m.drift_severity == "warning")
        
        # Disable conditions
        if n_critical >= 1:
            critical_features = [m.feature_name for m in metrics.values() 
                               if m.drift_severity == "critical"]
            return True, f"Critical drift detected in: {', '.join(critical_features)}"
        
        if n_warning >= len(metrics) * 0.5:
            return True, f">50% features showing warning-level drift"
        
        aggregate_score = self.get_aggregate_drift_score()
        if aggregate_score > self.config.psi_threshold * 1.5:
            return True, f"Aggregate drift score too high: {aggregate_score:.4f}"
        
        return False, None
    
    def disable_ensemble(self, reason: str) -> None:
        """Disable the ensemble due to drift."""
        self.ensemble_enabled = False
        self.disable_reason = reason
        logger.critical(f"Ensemble DISABLED due to drift: {reason}")
        
        # Log to drift history
        self.drift_history.append({
            'step': self.current_step,
            'action': 'disabled',
            'reason': reason,
            'metrics': {k: v.psi_value for k, v in self.check_all_features().items()},
        })
    
    def enable_ensemble(self) -> None:
        """Re-enable ensemble after retraining."""
        self.ensemble_enabled = True
        self.disable_reason = None
        self.last_retrain_step = self.current_step
        
        # Update reference distribution with recent data
        self._refresh_reference()
        
        logger.info("Ensemble re-enabled after retraining")
        
        self.drift_history.append({
            'step': self.current_step,
            'action': 'enabled',
            'reason': 'Retraining completed',
        })
    
    def _refresh_reference(self) -> None:
        """Refresh reference distribution with monitoring data."""
        for name in self.feature_names:
            mon_data = list(self.monitoring_data[name])
            if mon_data:
                # Clear and refill reference with recent data
                self.reference_data[name].clear()
                for val in mon_data:
                    self.reference_data[name].append(val)
        
        self._update_reference_stats()
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drift summary."""
        metrics = self.check_all_features()
        
        return {
            'ensemble_enabled': self.ensemble_enabled,
            'disable_reason': self.disable_reason,
            'current_step': self.current_step,
            'last_retrain_step': self.last_retrain_step,
            'n_features_drifting': sum(1 for m in metrics.values() if m.is_drifting),
            'n_features_total': len(metrics),
            'aggregate_drift_score': self.get_aggregate_drift_score(),
            'feature_metrics': {
                name: {
                    'psi': m.psi_value,
                    'js': m.js_divergence,
                    'ks_pvalue': m.ks_pvalue,
                    'severity': m.drift_severity,
                }
                for name, m in metrics.items()
            },
        }
    
    def mark_feature_critical(self, feature_name: str) -> None:
        """Mark a feature as critical (any drift triggers immediate action)."""
        self.critical_features.add(feature_name)
        logger.info(f"Feature '{feature_name}' marked as critical")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    config = DriftConfig(
        reference_window=1000,
        monitoring_window=100,
        psi_threshold=0.2,
    )
    
    feature_names = ['feature_1', 'feature_2', 'feature_3']
    detector = ModelDriftDetector(config, feature_names)
    
    # Simulate stable period
    print("=== Stable Period ===")
    for i in range(500):
        x = {
            'feature_1': np.random.normal(0, 1),
            'feature_2': np.random.normal(5, 2),
            'feature_3': np.random.exponential(1),
        }
        detector.update_reference(x)
        detector.update_monitoring(x)
    
    summary = detector.get_summary_statistics()
    print(f"After stable period: {summary['n_features_drifting']} drifting")
    
    # Simulate drift
    print("\n=== Drift Period ===")
    for i in range(200):
        x = {
            'feature_1': np.random.normal(3, 1),  # Shifted mean
            'feature_2': np.random.normal(5, 2),
            'feature_3': np.random.exponential(3),  # Changed scale
        }
        detector.update_monitoring(x)
        
        if i % 50 == 0:
            should_disable, reason = detector.should_disable_ensemble()
            print(f"Step {i}: Drift score={detector.get_aggregate_drift_score():.4f}, "
                  f"Disable={should_disable}")
            if reason:
                print(f"  Reason: {reason}")
    
    # Final summary
    print("\n=== Final Summary ===")
    summary = detector.get_summary_statistics()
    for feat, metrics in summary['feature_metrics'].items():
        print(f"{feat}: PSI={metrics['psi']:.4f}, Severity={metrics['severity']}")
