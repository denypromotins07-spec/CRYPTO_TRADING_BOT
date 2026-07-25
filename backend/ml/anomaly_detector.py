"""
Anomaly Detection Engine for ZAID Personal Crypto Trading Bot
Detects model drift and statistical outliers
Uses multiple detection methods for robustness
Memory-efficient streaming algorithms

Part of the 152 domains of quantitative finance implementation.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of detected anomalies."""
    OUTLIER = "OUTLIER"
    DRIFT = "DRIFT"
    REGIME_CHANGE = "REGIME_CHANGE"
    DATA_QUALITY = "DATA_QUALITY"
    CORRELATION_BREAK = "CORRELATION_BREAK"


@dataclass
class AnomalyEvent:
    """Detected anomaly with metadata."""
    anomaly_type: AnomalyType
    symbol: str
    severity: float  # 0.0 to 1.0
    description: str
    timestamp: float
    affected_features: List[str]
    recommended_action: str


class StatisticalOutlierDetector:
    """
    Detects outliers using multiple statistical methods.
    Combines Z-score, IQR, and MAD for robustness.
    """
    
    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        
        # Data buffer
        self.values: Dict[str, deque] = {}
        
        # Running statistics (Welford's algorithm)
        self.n: Dict[str, int] = {}
        self.mean: Dict[str, float] = {}
        self.M2: Dict[str, float] = {}
        
    def update(self, feature_name: str, value: float) -> Optional[AnomalyEvent]:
        """Update with new value and check for outliers."""
        if feature_name not in self.values:
            self.values[feature_name] = deque(maxlen=self.window_size)
            self.n[feature_name] = 0
            self.mean[feature_name] = 0.0
            self.M2[feature_name] = 0.0
        
        # Update running statistics
        self.n[feature_name] += 1
        delta = value - self.mean[feature_name]
        self.mean[feature_name] += delta / self.n[feature_name]
        delta2 = value - self.mean[feature_name]
        self.M2[feature_name] += delta * delta2
        
        self.values[feature_name].append(value)
        
        # Need enough data
        if len(self.values[feature_name]) < 30:
            return None
        
        # Check for outlier using multiple methods
        z_score_anomaly = self._check_zscore(feature_name, value)
        iqr_anomaly = self._check_iqr(feature_name, value)
        mad_anomaly = self._check_mad(feature_name, value)
        
        # Combine results
        anomaly_count = sum([z_score_anomaly, iqr_anomaly, mad_anomaly])
        
        if anomaly_count >= 2:  # At least 2 methods agree
            variance = self.M2[feature_name] / self.n[feature_name]
            std = np.sqrt(variance) if variance > 0 else 1.0
            z_score = abs(value - self.mean[feature_name]) / std
            
            return AnomalyEvent(
                anomaly_type=AnomalyType.OUTLIER,
                symbol="unknown",
                severity=min(1.0, z_score / 5.0),
                description=f"Statistical outlier in {feature_name}: z={z_score:.2f}",
                timestamp=0.0,
                affected_features=[feature_name],
                recommended_action="REVIEW" if z_score < 4 else "IGNORE"
            )
        
        return None
    
    def _check_zscore(self, feature_name: str, value: float) -> bool:
        """Check using Z-score method."""
        if self.n[feature_name] < 2:
            return False
        
        variance = self.M2[feature_name] / self.n[feature_name]
        std = np.sqrt(variance) if variance > 0 else 1.0
        
        z_score = abs(value - self.mean[feature_name]) / std
        return z_score > self.z_threshold
    
    def _check_iqr(self, feature_name: str, value: float) -> bool:
        """Check using Interquartile Range method."""
        if len(self.values[feature_name]) < 10:
            return False
        
        sorted_vals = sorted(self.values[feature_name])
        n = len(sorted_vals)
        
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return value < lower_bound or value > upper_bound
    
    def _check_mad(self, feature_name: str, value: float) -> bool:
        """Check using Median Absolute Deviation method."""
        if len(self.values[feature_name]) < 10:
            return False
        
        median = np.median(list(self.values[feature_name]))
        mad = np.median(np.abs(np.array(list(self.values[feature_name])) - median))
        
        if mad == 0:
            return False
        
        # Modified Z-score
        modified_z = 0.6745 * abs(value - median) / mad
        return modified_z > 3.5


class DriftDetector:
    """
    Detects concept drift using ADWIN-like approach.
    Monitors distribution changes over time.
    """
    
    def __init__(self, min_window: int = 30, delta: float = 0.002):
        self.min_window = min_window
        self.delta = delta  # Confidence parameter
        
        self.values: deque = deque(maxlen=1000)
        self.timestamps: deque = deque(maxlen=1000)
        
        self.baseline_mean: Optional[float] = None
        self.baseline_var: Optional[float] = None
        self.baseline_n: int = 0
        
    def update(self, value: float, timestamp: float) -> Optional[AnomalyEvent]:
        """Update detector and check for drift."""
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        # Initialize baseline
        if self.baseline_mean is None and len(self.values) >= self.min_window:
            vals = list(self.values)[:self.min_window]
            self.baseline_mean = np.mean(vals)
            self.baseline_var = np.var(vals)
            self.baseline_n = self.min_window
            return None
        
        if self.baseline_mean is None:
            return None
        
        # Check recent window vs baseline
        if len(self.values) >= self.min_window * 2:
            recent = list(self.values)[-self.min_window:]
            recent_mean = np.mean(recent)
            recent_var = np.var(recent)
            
            # CUSUM-like test
            diff = abs(recent_mean - self.baseline_mean)
            threshold = np.sqrt(2 * self.baseline_var * np.log(1/self.delta) / self.min_window)
            
            if diff > threshold:
                # Calculate drift magnitude
                drift_magnitude = diff / (np.sqrt(self.baseline_var) + 1e-6)
                
                return AnomalyEvent(
                    anomaly_type=AnomalyType.DRIFT,
                    symbol="unknown",
                    severity=min(1.0, drift_magnitude / 2.0),
                    description=f"Concept drift detected: mean shifted by {drift_magnitude:.2f} std devs",
                    timestamp=timestamp,
                    affected_features=["distribution"],
                    recommended_action="RECALIBRATE" if drift_magnitude > 1.0 else "MONITOR"
                )
        
        return None
    
    def reset_baseline(self):
        """Reset baseline to current distribution."""
        if len(self.values) >= self.min_window:
            vals = list(self.values)[-self.min_window:]
            self.baseline_mean = np.mean(vals)
            self.baseline_var = np.var(vals)
            self.baseline_n = self.min_window


class CorrelationBreakDetector:
    """
    Detects breaks in historical correlation patterns.
    Important for multi-asset strategies.
    """
    
    def __init__(self, window_size: int = 60, threshold: float = 0.3):
        self.window_size = window_size
        self.threshold = threshold
        
        self.asset_values: Dict[str, deque] = {}
        self.historical_corr: Dict[Tuple[str, str], float] = {}
        
    def update(self, asset_prices: Dict[str, float], 
               timestamp: float) -> List[AnomalyEvent]:
        """Update with prices for all assets and check correlation breaks."""
        events = []
        
        # Store values
        for asset, price in asset_prices.items():
            if asset not in self.asset_values:
                self.asset_values[asset] = deque(maxlen=self.window_size)
            self.asset_values[asset].append(np.log(price))
        
        # Check pairs
        assets = list(self.asset_values.keys())
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                if len(self.asset_values[asset1]) < self.window_size:
                    continue
                
                # Calculate returns
                vals1 = list(self.asset_values[asset1])
                vals2 = list(self.asset_values[asset2])
                
                ret1 = np.diff(vals1[-self.window_size:])
                ret2 = np.diff(vals2[-self.window_size:])
                
                if len(ret1) < 10:
                    continue
                
                # Current correlation
                current_corr = np.corrcoef(ret1, ret2)[0, 1]
                
                pair = (asset1, asset2)
                if pair in self.historical_corr:
                    hist_corr = self.historical_corr[pair]
                    
                    # Check for significant change
                    corr_change = abs(current_corr - hist_corr)
                    
                    if corr_change > self.threshold:
                        events.append(AnomalyEvent(
                            anomaly_type=AnomalyType.CORRELATION_BREAK,
                            symbol=f"{asset1}_{asset2}",
                            severity=min(1.0, corr_change),
                            description=f"Correlation break: {hist_corr:.2f} -> {current_corr:.2f}",
                            timestamp=timestamp,
                            affected_features=[asset1, asset2],
                            recommended_action="REDUCE_EXPOSURE"
                        ))
                
                # Update historical correlation (EMA)
                if pair not in self.historical_corr:
                    self.historical_corr[pair] = current_corr
                else:
                    alpha = 0.1
                    self.historical_corr[pair] = (1 - alpha) * self.historical_corr[pair] + alpha * current_corr
        
        return events


class AnomalyDetectionEngine:
    """
    Main engine coordinating all anomaly detection methods.
    Singleton pattern for global access.
    """
    
    _instance: Optional['AnomalyDetectionEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.outlier_detectors: Dict[str, StatisticalOutlierDetector] = {}
        self.drift_detectors: Dict[str, DriftDetector] = {}
        self.corr_break_detector = CorrelationBreakDetector()
        
        self.recent_anomalies: deque = deque(maxlen=100)
        self.anomaly_counts: Dict[str, int] = {}
        
        self._initialized = True
        logger.info("AnomalyDetectionEngine initialized")
    
    def get_or_create_detectors(self, symbol: str, feature_names: List[str]):
        """Get or create detectors for a symbol's features."""
        if symbol not in self.outlier_detectors:
            self.outlier_detectors[symbol] = {}
            self.drift_detectors[symbol] = {}
            self.anomaly_counts[symbol] = 0
        
        for feature in feature_names:
            if feature not in self.outlier_detectors[symbol]:
                self.outlier_detectors[symbol][feature] = StatisticalOutlierDetector()
                self.drift_detectors[symbol][feature] = DriftDetector()
    
    def process_features(self, symbol: str, features: Dict[str, float],
                         timestamp: float) -> List[AnomalyEvent]:
        """Process feature values through all detectors."""
        events = []
        
        feature_names = list(features.keys())
        self.get_or_create_detectors(symbol, feature_names)
        
        # Check each feature
        for feature_name, value in features.items():
            # Outlier detection
            outlier_event = self.outlier_detectors[symbol][feature_name].update(
                feature_name, value
            )
            if outlier_event:
                outlier_event.symbol = symbol
                outlier_event.timestamp = timestamp
                events.append(outlier_event)
            
            # Drift detection
            drift_event = self.drift_detectors[symbol][feature_name].update(
                value, timestamp
            )
            if drift_event:
                drift_event.symbol = symbol
                events.append(drift_event)
        
        # Update counts
        self.anomaly_counts[symbol] += len(events)
        
        # Store recent anomalies
        for event in events:
            self.recent_anomalies.append(event)
        
        return events
    
    def process_multi_asset(self, asset_prices: Dict[str, float],
                           timestamp: float) -> List[AnomalyEvent]:
        """Check for correlation breaks across assets."""
        events = self.corr_break_detector.update(asset_prices, timestamp)
        
        for event in events:
            self.recent_anomalies.append(event)
        
        return events
    
    def get_anomaly_summary(self, symbol: str) -> Dict[str, any]:
        """Get summary of anomaly activity for a symbol."""
        recent = [e for e in list(self.recent_anomalies)[-50:] 
                 if e.symbol == symbol or symbol == "all"]
        
        by_type = {}
        for event in recent:
            type_name = event.anomaly_type.value
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1
        
        return {
            'symbol': symbol,
            'total_anomalies': self.anomaly_counts.get(symbol, 0),
            'recent_count': len(recent),
            'by_type': by_type,
            'health_score': max(0, 1.0 - len(recent) / 50)
        }
    
    def should_alert(self, symbol: str, severity_threshold: float = 0.7) -> bool:
        """Check if alert should be triggered."""
        recent = [e for e in list(self.recent_anomalies)[-20:] 
                 if e.symbol == symbol and e.severity >= severity_threshold]
        return len(recent) >= 2
    
    def get_recommendations(self, symbol: str) -> List[str]:
        """Get actionable recommendations based on recent anomalies."""
        recent = list(self.recent_anomalies)[-20:]
        symbol_events = [e for e in recent if e.symbol == symbol]
        
        recommendations = set()
        
        for event in symbol_events:
            if event.recommended_action:
                recommendations.add(event.recommended_action)
        
        # Add default recommendation based on anomaly types
        types = set(e.anomaly_type for e in symbol_events)
        
        if AnomalyType.DRIFT in types:
            recommendations.add("RECALIBRATE_MODELS")
        if AnomalyType.CORRELATION_BREAK in types:
            recommendations.add("REDUCE_POSITION_SIZE")
        if AnomalyType.OUTLIER in types:
            recommendations.add("VERIFY_DATA_QUALITY")
        
        return list(recommendations)


if __name__ == "__main__":
    import time
    
    engine = AnomalyDetectionEngine()
    
    # Simulate normal data then introduce anomaly
    base_value = 100
    for i in range(200):
        # Normal period
        if i < 150:
            value = base_value + np.random.randn() * 2
        # Introduce outlier
        elif i == 150:
            value = base_value + 15  # Large spike
        # Drift period
        else:
            value = base_value + 8 + np.random.randn() * 2
        
        features = {"price_return": value / 100 - 1}
        
        events = engine.process_features("BTCUSDT", features, time.time())
        
        if events:
            for event in events:
                print(f"Step {i}: {event.anomaly_type.value} - {event.description}")
                print(f"  Severity: {event.severity:.2f}, Action: {event.recommended_action}")
    
    # Get summary
    summary = engine.get_anomaly_summary("BTCUSDT")
    print(f"\nSummary: {summary}")
    
    recommendations = engine.get_recommendations("BTCUSDT")
    print(f"Recommendations: {recommendations}")
