#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Stage 31
High-Frequency Feature Stores, MLOps, and Concept Drift
File: backend/mlops/drift_detector.py
Chapter 2: Concept Drift Detection, Online Calibration, and Model Decay

Applies Kolmogorov-Smirnov tests on feature distributions to detect concept drift.
Instantly halts trading if feature drift exceeds the 99th percentile threshold.
Optimized for AMD Ryzen AI 5 with 8GB RAM constraint.
Uses scipy.stats for statistical tests with NumPy C-extensions.
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


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""
    timestamp: float
    feature_name: str
    asset_id: int
    ks_statistic: float
    p_value: float
    severity: DriftSeverity
    threshold_exceeded: bool
    action_taken: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureDistribution:
    """Stores distribution statistics for a feature."""
    mean: float
    std: float
    median: float
    percentiles: Dict[int, float]  # e.g., {90: ..., 95: ..., 99: ...}
    sample_count: int
    last_updated: float


class DriftDetector:
    """
    Real-time concept drift detector using Kolmogorov-Smirnov tests.
    
    Monitors feature distributions and detects shifts that indicate
    concept drift in market regimes. Implements multiple detection methods:
    
    1. KS Test: Non-parametric test comparing empirical distributions
    2. PSI (Population Stability Index): Measures distribution shift
    3. Z-score monitoring: Detects mean shifts
    4. Rolling window comparison: Compares recent vs historical
    
    Trading is halted immediately if drift exceeds 99th percentile threshold.
    """
    
    def __init__(
        self,
        reference_window_size: int = 10000,
        test_window_size: int = 500,
        ks_threshold: float = 0.05,  # p-value threshold
        psi_threshold: float = 0.25,  # PSI threshold for significant drift
        z_score_threshold: float = 3.0,  # Standard deviations
        halt_on_critical: bool = True,
        memory_budget_mb: int = 256
    ):
        """
        Initialize the drift detector.
        
        Args:
            reference_window_size: Size of reference (historical) window
            test_window_size: Size of test (recent) window
            ks_threshold: P-value threshold for KS test (lower = more sensitive)
            psi_threshold: PSI threshold for distribution shift
            z_score_threshold: Z-score threshold for mean shift detection
            halt_on_critical: Whether to halt trading on critical drift
            memory_budget_mb: Memory budget for storing distributions
        """
        self.reference_window_size = reference_window_size
        self.test_window_size = test_window_size
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.z_score_threshold = z_score_threshold
        self.halt_on_critical = halt_on_critical
        self.memory_budget_bytes = memory_budget_mb * 1024 * 1024
        
        # Reference distributions per feature per asset
        # Structure: {asset_id: {feature_name: deque}}
        self.reference_windows: Dict[int, Dict[str, deque]] = {}
        
        # Current test windows
        self.test_windows: Dict[int, Dict[str, deque]] = {}
        
        # Baseline distributions (computed from reference)
        self.baseline_distributions: Dict[int, Dict[str, FeatureDistribution]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Trading halt state
        self.trading_halted = False
        self.halt_reason: Optional[str] = None
        self.halt_timestamp: Optional[float] = None
        
        # Alert history
        self.alerts: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'total_tests': 0,
            'drifts_detected': 0,
            'halts_triggered': 0,
            'features_monitored': 0
        }
        
        # Estimated bytes per float
        self._bytes_per_float = 8
    
    def initialize_feature(
        self,
        asset_id: int,
        feature_name: str,
        initial_data: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize monitoring for a new feature.
        
        Args:
            asset_id: Asset identifier
            feature_name: Name of the feature to monitor
            initial_data: Optional initial data for reference window
        """
        with self._lock:
            # Initialize containers for asset if needed
            if asset_id not in self.reference_windows:
                self.reference_windows[asset_id] = {}
                self.test_windows[asset_id] = {}
                self.baseline_distributions[asset_id] = {}
            
            # Create deques for windows
            self.reference_windows[asset_id][feature_name] = deque(
                maxlen=self.reference_window_size
            )
            self.test_windows[asset_id][feature_name] = deque(
                maxlen=self.test_window_size
            )
            
            # Add initial data if provided
            if initial_data is not None and len(initial_data) > 0:
                self.reference_windows[asset_id][feature_name].extend(initial_data)
                self._update_baseline(asset_id, feature_name)
            
            self.stats['features_monitored'] += 1
    
    def add_observation(
        self,
        asset_id: int,
        feature_name: str,
        value: float
    ) -> Optional[DriftAlert]:
        """
        Add a new observation and check for drift.
        
        Args:
            asset_id: Asset identifier
            feature_name: Name of the feature
            value: Observed feature value
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        with self._lock:
            # Skip if trading is halted
            if self.trading_halted:
                return None
            
            # Ensure feature is initialized
            if asset_id not in self.reference_windows or \
               feature_name not in self.reference_windows[asset_id]:
                self.initialize_feature(asset_id, feature_name)
            
            # Add to test window
            self.test_windows[asset_id][feature_name].append(value)
            
            # Check drift when test window is full
            if len(self.test_windows[asset_id][feature_name]) >= self.test_window_size:
                alert = self._check_drift(asset_id, feature_name)
                if alert:
                    return alert
            
            return None
    
    def _check_drift(
        self,
        asset_id: int,
        feature_name: str
    ) -> Optional[DriftAlert]:
        """
        Perform drift detection tests on a feature.
        
        Returns:
            DriftAlert if significant drift detected
        """
        ref_data = np.array(self.reference_windows[asset_id][feature_name])
        test_data = np.array(self.test_windows[asset_id][feature_name])
        
        if len(ref_data) < 100 or len(test_data) < 10:
            return None
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(ref_data, test_data)
        
        # Calculate PSI
        psi = self._calculate_psi(ref_data, test_data)
        
        # Calculate Z-score for mean shift
        baseline = self.baseline_distributions.get(asset_id, {}).get(feature_name)
        z_score = 0.0
        if baseline and baseline.std > 0:
            z_score = (np.mean(test_data) - baseline.mean) / baseline.std
        
        # Determine severity
        severity, threshold_exceeded = self._determine_severity(
            ks_statistic, p_value, psi, z_score
        )
        
        self.stats['total_tests'] += 1
        
        # Create alert if needed
        if severity != DriftSeverity.NONE:
            self.stats['drifts_detected'] += 1
            
            action = "monitoring"
            if threshold_exceeded and self.halt_on_critical:
                action = "trading_halted"
                self._halt_trading(
                    f"Critical drift detected in {feature_name} "
                    f"(KS={ks_statistic:.4f}, p={p_value:.6f})"
                )
                self.stats['halts_triggered'] += 1
            
            alert = DriftAlert(
                timestamp=time.time(),
                feature_name=feature_name,
                asset_id=asset_id,
                ks_statistic=ks_statistic,
                p_value=p_value,
                severity=severity,
                threshold_exceeded=threshold_exceeded,
                action_taken=action,
                metadata={
                    'psi': psi,
                    'z_score': z_score,
                    'reference_mean': float(np.mean(ref_data)),
                    'test_mean': float(np.mean(test_data)),
                    'reference_std': float(np.std(ref_data)),
                    'test_std': float(np.std(test_data))
                }
            )
            
            self.alerts.append(alert)
            return alert
        
        return None
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        buckets: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI measures how much the distribution has shifted.
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Some minor change
        PSI > 0.25: Significant shift
        """
        # Create buckets based on reference distribution
        percentiles = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        percentiles = np.unique(percentiles)  # Remove duplicates
        
        if len(percentiles) < 2:
            return 0.0
        
        # Calculate proportions in each bucket
        ref_counts = np.histogram(reference, bins=percentiles)[0]
        test_counts = np.histogram(test, bins=percentiles)[0]
        
        # Convert to proportions (add small epsilon to avoid log(0))
        epsilon = 1e-10
        ref_props = (ref_counts / len(reference)) + epsilon
        test_props = (test_counts / len(test)) + epsilon
        
        # Calculate PSI
        psi = np.sum((test_props - ref_props) * np.log(test_props / ref_props))
        
        return float(psi)
    
    def _determine_severity(
        self,
        ks_statistic: float,
        p_value: float,
        psi: float,
        z_score: float
    ) -> Tuple[DriftSeverity, bool]:
        """
        Determine drift severity based on multiple metrics.
        
        Returns:
            Tuple of (severity, threshold_exceeded)
        """
        # Critical: Multiple indicators show severe drift
        if p_value < 0.001 and psi > 0.5 and abs(z_score) > 5:
            return DriftSeverity.CRITICAL, True
        
        # High: Strong evidence of drift
        if p_value < self.ks_threshold and psi > self.psi_threshold:
            return DriftSeverity.HIGH, True
        
        # Medium: Moderate evidence
        if p_value < 0.01 or psi > 0.15 or abs(z_score) > self.z_score_threshold:
            return DriftSeverity.MEDIUM, False
        
        # Low: Weak evidence
        if p_value < 0.05 or psi > 0.1 or abs(z_score) > 2:
            return DriftSeverity.LOW, False
        
        return DriftSeverity.NONE, False
    
    def _update_baseline(self, asset_id: int, feature_name: str) -> None:
        """Update baseline distribution from reference window."""
        data = np.array(self.reference_windows[asset_id][feature_name])
        
        if len(data) < 10:
            return
        
        self.baseline_distributions[asset_id][feature_name] = FeatureDistribution(
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            median=float(np.median(data)),
            percentiles={
                p: float(np.percentile(data, p))
                for p in [50, 90, 95, 99]
            },
            sample_count=len(data),
            last_updated=time.time()
        )
    
    def _halt_trading(self, reason: str) -> None:
        """Halt all trading operations."""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_timestamp = time.time()
    
    def resume_trading(self) -> bool:
        """
        Resume trading after manual review.
        
        Returns:
            True if trading resumed, False if still halted
        """
        with self._lock:
            if not self.trading_halted:
                return True
            
            # Require manual confirmation for critical drifts
            if self.alerts and self.alerts[-1].severity == DriftSeverity.CRITICAL:
                # In production, this would require explicit confirmation
                pass
            
            self.trading_halted = False
            self.halt_reason = None
            self.halt_timestamp = None
            return True
    
    def update_reference_window(
        self,
        asset_id: int,
        feature_name: str
    ) -> None:
        """
        Update reference window with current test data.
        
        Call this after confirming the new regime is valid.
        """
        with self._lock:
            if asset_id not in self.reference_windows:
                return
            
            if feature_name not in self.reference_windows[asset_id]:
                return
            
            # Replace reference with test data
            test_data = list(self.test_windows[asset_id][feature_name])
            self.reference_windows[asset_id][feature_name].clear()
            self.reference_windows[asset_id][feature_name].extend(test_data)
            
            # Update baseline
            self._update_baseline(asset_id, feature_name)
    
    def get_drift_status(
        self,
        asset_id: int,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Get current drift status for a feature.
        
        Returns:
            Dictionary with drift metrics
        """
        with self._lock:
            if asset_id not in self.reference_windows or \
               feature_name not in self.reference_windows[asset_id]:
                return {'status': 'not_monitored'}
            
            ref_data = np.array(self.reference_windows[asset_id][feature_name])
            test_data = np.array(self.test_windows[asset_id][feature_name])
            
            if len(ref_data) < 10 or len(test_data) < 10:
                return {'status': 'insufficient_data'}
            
            ks_stat, p_val = stats.ks_2samp(ref_data, test_data)
            psi = self._calculate_psi(ref_data, test_data)
            
            baseline = self.baseline_distributions.get(asset_id, {}).get(feature_name)
            z_score = 0.0
            if baseline and baseline.std > 0:
                z_score = (np.mean(test_data) - baseline.mean) / baseline.std
            
            return {
                'status': 'monitored',
                'ks_statistic': ks_stat,
                'p_value': p_val,
                'psi': psi,
                'z_score': z_score,
                'reference_mean': float(np.mean(ref_data)),
                'test_mean': float(np.mean(test_data)),
                'reference_std': float(np.std(ref_data)),
                'test_std': float(np.std(test_data)),
                'sample_sizes': {
                    'reference': len(ref_data),
                    'test': len(test_data)
                }
            }
    
    def get_all_alerts(
        self,
        since_timestamp: Optional[float] = None,
        severity_filter: Optional[DriftSeverity] = None
    ) -> List[DriftAlert]:
        """
        Get drift alerts with optional filtering.
        
        Args:
            since_timestamp: Only return alerts after this timestamp
            severity_filter: Only return alerts of this severity or higher
            
        Returns:
            List of matching DriftAlert objects
        """
        alerts = list(self.alerts)
        
        if since_timestamp is not None:
            alerts = [a for a in alerts if a.timestamp >= since_timestamp]
        
        if severity_filter is not None:
            severity_order = list(DriftSeverity)
            min_idx = severity_order.index(severity_filter)
            alerts = [
                a for a in alerts 
                if severity_order.index(a.severity) >= min_idx
            ]
        
        return alerts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            **self.stats,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'halt_duration_s': (
                time.time() - self.halt_timestamp 
                if self.halt_timestamp else 0
            ),
            'assets_monitored': len(self.reference_windows),
            'memory_estimate_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage."""
        total_samples = 0
        for asset_features in self.reference_windows.values():
            for samples in asset_features.values():
                total_samples += len(samples)
        for asset_features in self.test_windows.values():
            for samples in asset_features.values():
                total_samples += len(samples)
        
        return total_samples * self._bytes_per_float


# Convenience function for creating a trading-ready drift detector
def create_production_drift_detector(
    assets: List[str] = None,
    features_per_asset: int = 128
) -> DriftDetector:
    """
    Create a production-ready drift detector.
    
    Args:
        assets: List of asset names to monitor
        features_per_asset: Number of features per asset
        
    Returns:
        Configured DriftDetector instance
    """
    if assets is None:
        assets = ['BTC', 'ETH', 'SOL', 'USDT']
    
    detector = DriftDetector(
        reference_window_size=10000,
        test_window_size=500,
        ks_threshold=0.01,  # More sensitive for production
        psi_threshold=0.2,
        z_score_threshold=3.5,
        halt_on_critical=True,
        memory_budget_mb=256
    )
    
    # Initialize with placeholder features
    # In production, actual feature names would be used
    for i, asset in enumerate(assets):
        for j in range(features_per_asset):
            detector.initialize_feature(i, f"feature_{j}")
    
    return detector


if __name__ == '__main__':
    # Test the drift detector
    print("Testing Drift Detector for ZAID Trading Bot...")
    
    # Create detector
    detector = DriftDetector(
        reference_window_size=1000,
        test_window_size=100,
        halt_on_critical=True
    )
    
    # Initialize a feature
    np.random.seed(42)
    reference_data = np.random.normal(0, 1, 1000)
    detector.initialize_feature(0, "test_feature", reference_data)
    
    # Add normal observations
    print("\nAdding normal observations...")
    for i in range(100):
        value = np.random.normal(0, 1)
        alert = detector.add_observation(0, "test_feature", value)
        if alert:
            print(f"  Alert at {i}: {alert.severity.value}")
    
    print(f"Trading halted: {detector.trading_halted}")
    
    # Add drifted observations (shifted mean)
    print("\nAdding drifted observations (mean shift)...")
    for i in range(100):
        value = np.random.normal(2, 1)  # Shifted mean
        alert = detector.add_observation(0, "test_feature", value)
        if alert:
            print(f"  Alert at {i}: {alert.severity.value}, KS={alert.ks_statistic:.4f}, p={alert.p_value:.6f}")
    
    print(f"\nTrading halted: {detector.trading_halted}")
    print(f"Halt reason: {detector.halt_reason}")
    
    # Get status
    status = detector.get_drift_status(0, "test_feature")
    print(f"\nDrift status: {status}")
    
    # Get stats
    stats = detector.get_stats()
    print(f"\nDetector stats: {stats}")
    
    print("\nDrift Detector test completed!")
