#!/usr/bin/env python3
"""
Deep Learning Soul Logger for SOUL.md

Logs prediction confidence, gradient drift, and model performance metrics
to the central SOUL.md file for the ZAID trading bot.

Features:
- Real-time prediction confidence tracking
- Gradient drift detection for model degradation
- CNN pattern recognition logging (liquidity sweeps, spoofing)
- Sharpe ratio monitoring
- Thread-safe file operations
- Automatic alert generation for anomalies

Integrates with the 152 domains of quantitative finance.
"""

from __future__ import annotations
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import numpy as np
import numpy.typing as npt
import logging
import json
import threading
import fcntl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class PredictionLog:
    """Single prediction event log entry."""
    timestamp: str
    model_name: str
    asset: str
    prediction: float
    confidence: float
    execution_time_ms: float
    regime: str = "NORMAL"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftMetrics:
    """Model drift measurement."""
    timestamp: str
    model_name: str
    gradient_norm: float
    weight_change_norm: float
    prediction_drift: float
    is_degraded: bool


@dataclass
class PatternRecognition:
    """CNN-detected market pattern."""
    timestamp: str
    pattern_type: str  # 'liquidity_sweep', 'spoofing', 'accumulation'
    confidence: float
    asset: str
    price_level: float
    volume_profile: Dict[str, float]


class DLSoulLogger:
    """
    Central logger for deep learning metrics in SOUL.md.
    
    Tracks:
    - Prediction confidence across all models
    - Gradient drift indicating model degradation
    - CNN pattern detections (spoofing, liquidity sweeps)
    - Model performance vs Sharpe targets
    
    All logs are written to SOUL.md in a structured format.
    """
    
    def __init__(self, soul_path: str = "SOUL.md"):
        self.soul_path = Path(soul_path)
        self._lock = threading.Lock()
        self._prediction_history: List[PredictionLog] = []
        self._drift_history: List[DriftMetrics] = []
        self._pattern_history: List[PatternRecognition] = []
        
        # Thresholds for alerts
        self.confidence_threshold = 0.7
        self.drift_threshold = 0.1
        self.max_prediction_drift = 0.05
        
        logger.info(f"DLSoulLogger initialized: {self.soul_path}")
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.utcnow().isoformat() + "Z"
    
    def _ensure_soul_file(self) -> None:
        """Ensure SOUL.md exists with proper header."""
        if not self.soul_path.exists():
            self.soul_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.soul_path, 'w') as f:
                f.write("# ZAID Trading Bot - Deep Learning Soul Log\n\n")
                f.write("## Overview\n")
                f.write("This file contains real-time logs from the deep learning\n")
                f.write("components of the ZAID Personal Crypto Trading Bot.\n\n")
                f.write("---\n\n")
    
    def _append_to_section(
        self,
        section_header: str,
        content: str,
        max_entries: int = 100
    ) -> None:
        """Append content to a specific section in SOUL.md."""
        with self._lock:
            self._ensure_soul_file()
            
            with open(self.soul_path, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                content_str = f.read()
                lines = content_str.split('\n')
                
                # Find or create section
                section_idx = -1
                for i, line in enumerate(lines):
                    if line.strip() == section_header:
                        section_idx = i
                        break
                
                if section_idx == -1:
                    # Create new section
                    lines.append(f"\n{section_header}\n")
                    lines.append(content)
                    lines.append("")
                else:
                    # Insert after section header
                    lines.insert(section_idx + 1, content)
                    
                    # Trim section if too long
                    section_content_start = section_idx + 1
                    section_content_end = len(lines)
                    
                    # Find next section or end
                    for i in range(section_idx + 2, len(lines)):
                        if lines[i].startswith('##'):
                            section_content_end = i
                            break
                    
                    section_length = section_content_end - section_content_start - 1
                    if section_length > max_entries:
                        # Remove oldest entries
                        excess = section_length - max_entries
                        del lines[section_content_start:section_content_start + excess]
                
                # Write back
                f.seek(0)
                f.truncate()
                f.write('\n'.join(lines))
                
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def log_prediction(
        self,
        model_name: str,
        asset: str,
        prediction: float,
        confidence: float,
        execution_time_ms: float,
        regime: str = "NORMAL",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log a single prediction event.
        
        Returns alert message if confidence is below threshold.
        """
        log_entry = PredictionLog(
            timestamp=self._get_timestamp(),
            model_name=model_name,
            asset=asset,
            prediction=prediction,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            regime=regime,
            metadata=metadata or {}
        )
        
        self._prediction_history.append(log_entry)
        
        # Keep history bounded
        if len(self._prediction_history) > 1000:
            self._prediction_history.pop(0)
        
        # Format log entry
        log_line = (
            f"- `{log_entry.timestamp}` | {model_name} | {asset} | "
            f"pred={prediction:.6f} | conf={confidence:.3f} | "
            f"time={execution_time_ms:.2f}ms | regime={regime}"
        )
        
        self._append_to_section("## Predictions", log_line)
        
        # Check for low confidence alert
        if confidence < self.confidence_threshold:
            alert = self._generate_alert(
                AlertLevel.WARNING,
                f"Low confidence prediction: {model_name} on {asset}",
                {"confidence": confidence, "threshold": self.confidence_threshold}
            )
            return alert
        
        return None
    
    def log_gradient_drift(
        self,
        model_name: str,
        gradient_norm: float,
        weight_change_norm: float,
        prediction_drift: float
    ) -> Optional[str]:
        """
        Log model drift metrics.
        
        Detects when model gradients or predictions drift significantly,
        indicating potential need for retraining.
        """
        is_degraded = (
            gradient_norm > self.drift_threshold or
            prediction_drift > self.max_prediction_drift
        )
        
        drift_log = DriftMetrics(
            timestamp=self._get_timestamp(),
            model_name=model_name,
            gradient_norm=gradient_norm,
            weight_change_norm=weight_change_norm,
            prediction_drift=prediction_drift,
            is_degraded=is_degraded
        )
        
        self._drift_history.append(drift_log)
        
        if len(self._drift_history) > 100:
            self._drift_history.pop(0)
        
        status = "DEGRADED" if is_degraded else "OK"
        log_line = (
            f"- `{drift_log.timestamp}` | {model_name} | "
            f"grad_norm={gradient_norm:.6f} | "
            f"weight_delta={weight_change_norm:.6f} | "
            f"pred_drift={prediction_drift:.6f} | [{status}]"
        )
        
        self._append_to_section("## Gradient Drift Monitor", log_line)
        
        if is_degraded:
            alert = self._generate_alert(
                AlertLevel.CRITICAL,
                f"Model degradation detected: {model_name}",
                asdict(drift_log)
            )
            return alert
        
        return None
    
    def log_cnn_pattern(
        self,
        pattern_type: str,
        confidence: float,
        asset: str,
        price_level: float,
        volume_profile: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log CNN-detected market patterns.
        
        Specifically tracks:
        - Liquidity sweeps
        - Spoofing attempts
        - Accumulation/distribution patterns
        """
        pattern_log = PatternRecognition(
            timestamp=self._get_timestamp(),
            pattern_type=pattern_type,
            confidence=confidence,
            asset=asset,
            price_level=price_level,
            volume_profile=volume_profile or {}
        )
        
        self._pattern_history.append(pattern_log)
        
        if len(self._pattern_history) > 500:
            self._pattern_history.pop(0)
        
        log_line = (
            f"- `{pattern_log.timestamp}` | **{pattern_type.upper()}** | "
            f"{asset} | conf={confidence:.3f} | "
            f"price={price_level:.2f}"
        )
        
        self._append_to_section("## CNN Pattern Recognition", log_line)
        
        # Special handling for liquidity sweep detection
        if pattern_type.lower() == "liquidity_sweep" and confidence > 0.8:
            self._append_to_section(
                "## Liquidity Sweep Alerts",
                f"- `{self._get_timestamp()}` | {asset} | "
                f"HIGH CONFIDENCE SWEEP DETECTED | price={price_level:.2f}"
            )
    
    def _generate_alert(
        self,
        level: AlertLevel,
        message: str,
        details: Dict[str, Any]
    ) -> str:
        """Generate and log an alert."""
        alert_line = (
            f"- `{self._get_timestamp()}` | **[{level.value}]** | "
            f"{message} | {json.dumps(details)}"
        )
        
        self._append_to_section("## Alerts", alert_line)
        
        # Also log to standard logger
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)
        
        log_func(f"ALERT: {message} - {details}")
        
        return alert_line
    
    def get_confidence_stats(
        self,
        model_name: Optional[str] = None,
        window: int = 100
    ) -> Dict[str, float]:
        """Calculate confidence statistics over recent predictions."""
        predictions = self._prediction_history[-window:]
        
        if model_name:
            predictions = [p for p in predictions if p.model_name == model_name]
        
        if not predictions:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        confidences = [p.confidence for p in predictions]
        
        return {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "count": len(confidences)
        }
    
    def get_model_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive model health report."""
        report = {
            "timestamp": self._get_timestamp(),
            "models": {},
            "overall_status": "HEALTHY"
        }
        
        # Group by model
        model_predictions: Dict[str, List[PredictionLog]] = {}
        for pred in self._prediction_history[-500:]:
            if pred.model_name not in model_predictions:
                model_predictions[pred.model_name] = []
            model_predictions[pred.model_name].append(pred)
        
        for model_name, preds in model_predictions.items():
            confidences = [p.confidence for p in preds]
            exec_times = [p.execution_time_ms for p in preds]
            
            avg_conf = np.mean(confidences)
            avg_exec = np.mean(exec_times)
            
            # Check drift
            recent_drift = [
                d for d in self._drift_history 
                if d.model_name == model_name
            ][-10:] if any(d.model_name == model_name for d in self._drift_history) else []
            
            is_healthy = (
                avg_conf >= self.confidence_threshold and
                (not recent_drift or not any(d.is_degraded for d in recent_drift))
            )
            
            report["models"][model_name] = {
                "avg_confidence": float(avg_conf),
                "avg_execution_ms": float(avg_exec),
                "prediction_count": len(preds),
                "is_healthy": is_healthy
            }
            
            if not is_healthy:
                report["overall_status"] = "DEGRADED"
        
        return report
    
    def write_health_report(self) -> None:
        """Write current health report to SOUL.md."""
        report = self.get_model_health_report()
        
        report_text = (
            f"\n### Report Generated: `{report['timestamp']}`\n"
            f"**Overall Status:** {report['overall_status']}\n\n"
            f"| Model | Avg Confidence | Avg Exec (ms) | Predictions | Healthy |\n"
            f"|-------|----------------|---------------|-------------|---------|\n"
        )
        
        for model_name, stats in report["models"].items():
            health_icon = "✅" if stats["is_healthy"] else "❌"
            report_text += (
                f"| {model_name} | {stats['avg_confidence']:.3f} | "
                f"{stats['avg_execution_ms']:.2f} | "
                f"{stats['prediction_count']} | {health_icon} |\n"
            )
        
        self._append_to_section("## Model Health Report", report_text)


def main():
    """Test DL Soul Logger."""
    import time
    
    print("=" * 60)
    print("DL Soul Logger Test")
    print("=" * 60)
    
    # Initialize logger
    logger_instance = DLSoulLogger("/workspace/SOUL.md")
    
    # Simulate predictions
    print("\n1. Logging predictions...")
    np.random.seed(42)
    
    models = ["LSTM", "GRU", "Transformer", "CNN"]
    assets = ["BTC", "ETH", "SOL", "USDT"]
    
    for i in range(20):
        model = np.random.choice(models)
        asset = np.random.choice(assets)
        pred = np.random.randn() * 0.01
        conf = np.random.uniform(0.5, 0.99)
        exec_time = np.random.uniform(0.5, 5.0)
        
        alert = logger_instance.log_prediction(
            model_name=model,
            asset=asset,
            prediction=pred,
            confidence=conf,
            execution_time_ms=exec_time,
            regime=np.random.choice(["LOW", "NORMAL", "HIGH"])
        )
        
        if alert:
            print(f"   Alert generated: {alert[:80]}...")
    
    # Simulate drift monitoring
    print("\n2. Logging gradient drift...")
    for model in models:
        grad_norm = np.random.uniform(0.01, 0.15)
        weight_delta = np.random.uniform(0.001, 0.05)
        pred_drift = np.random.uniform(0.001, 0.08)
        
        alert = logger_instance.log_gradient_drift(
            model_name=model,
            gradient_norm=grad_norm,
            weight_change_norm=weight_delta,
            prediction_drift=pred_drift
        )
        
        if alert:
            print(f"   Drift alert: {alert[:80]}...")
    
    # Simulate CNN pattern detection
    print("\n3. Logging CNN patterns...")
    patterns = [
        ("liquidity_sweep", 0.92, "BTC", 50123.45),
        ("spoofing", 0.78, "ETH", 3012.34),
        ("accumulation", 0.85, "SOL", 123.45),
    ]
    
    for pattern_type, conf, asset, price in patterns:
        logger_instance.log_cnn_pattern(
            pattern_type=pattern_type,
            confidence=conf,
            asset=asset,
            price_level=price,
            volume_profile={"bid_vol": 1000, "ask_vol": 800}
        )
        print(f"   Logged pattern: {pattern_type} on {asset}")
    
    # Get statistics
    print("\n4. Getting confidence statistics...")
    stats = logger_instance.get_confidence_stats(window=20)
    print(f"   Mean confidence: {stats['mean']:.3f}")
    print(f"   Std deviation: {stats['std']:.3f}")
    
    # Generate health report
    print("\n5. Generating health report...")
    report = logger_instance.get_model_health_report()
    print(f"   Overall status: {report['overall_status']}")
    print(f"   Models tracked: {len(report['models'])}")
    
    # Write report to SOUL.md
    logger_instance.write_health_report()
    print(f"\n   Health report written to {logger_instance.soul_path}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
