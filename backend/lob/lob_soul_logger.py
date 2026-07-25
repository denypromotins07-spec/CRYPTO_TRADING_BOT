#!/usr/bin/env python3
"""LOB Soul Logger - Logs impact model deviations and LOB anomalies to SOUL.md."""

from __future__ import annotations
from datetime import datetime
from typing import Optional, Dict
from threading import Lock
import os


class LOBSoulLogger:
    """Log LOB anomalies and impact deviations to SOUL.md."""
    
    def __init__(self, soul_path: str = "SOUL.md"):
        self.soul_path = soul_path
        self._lock = Lock()
        self._ensure_file_exists()
        
    def _ensure_file_exists(self) -> None:
        os.makedirs(os.path.dirname(self.soul_path) or ".", exist_ok=True)
        if not os.path.exists(self.soul_path):
            with open(self.soul_path, 'w') as f:
                f.write("# SOUL.md - Market Microstructure Event Log\n\n")
                f.write("## Stage 22: LOB Physics & Impact Models\n\n")
    
    def log_propagator_failure(self, predicted_impact: float, actual_impact: float,
                               asset: str, timestamp: Optional[float] = None) -> None:
        """Log when propagator model fails to predict liquidity vacuum."""
        with self._lock:
            ts = datetime.fromtimestamp(timestamp or datetime.now().timestamp())
            deviation = abs(actual_impact - predicted_impact)
            
            entry = f"""
### Propagator Model Failure
- **Timestamp**: {ts.isoformat()}
- **Asset**: {asset}
- **Predicted Impact**: {predicted_impact:.2f} bps
- **Actual Impact**: {actual_impact:.2f} bps
- **Deviation**: {deviation:.2f} bps
- **Severity**: {'CRITICAL' if deviation > 50 else 'WARNING'}
"""
            with open(self.soul_path, 'a') as f:
                f.write(entry)
    
    def log_impact_deviation(self, model: str, expected: float, actual: float,
                            asset: str, size: float) -> None:
        """Log significant impact model deviation."""
        with self._lock:
            ts = datetime.now()
            error_pct = abs(actual - expected) / max(expected, 0.001) * 100
            
            if error_pct < 20:  # Only log significant deviations
                return
            
            entry = f"""
### Impact Model Deviation
- **Timestamp**: {ts.isoformat()}
- **Model**: {model}
- **Asset**: {asset}
- **Size**: {size}
- **Expected**: {expected:.2f} bps
- **Actual**: {actual:.2f} bps
- **Error**: {error_pct:.1f}%
"""
            with open(self.soul_path, 'a') as f:
                f.write(entry)
    
    def log_liquidity_vacuum(self, asset: str, depth_drop_pct: float,
                             recovery_time_ms: float) -> None:
        """Log liquidity vacuum event."""
        with self._lock:
            ts = datetime.now()
            
            entry = f"""
### Liquidity Vacuum Detected
- **Timestamp**: {ts.isoformat()}
- **Asset**: {asset}
- **Depth Drop**: {depth_drop_pct:.1f}%
- **Recovery Time**: {recovery_time_ms:.0f} ms
- **Risk Level**: {'HIGH' if depth_drop_pct > 80 else 'MEDIUM'}
"""
            with open(self.soul_path, 'a') as f:
                f.write(entry)
    
    def log_successful_avoidance(self, event_type: str, avoided_loss_bps: float,
                                 details: Dict = None) -> None:
        """Log when bot successfully avoids fake liquidity wall."""
        with self._lock:
            ts = datetime.now()
            
            entry = f"""
### Successful Avoidance
- **Timestamp**: {ts.isoformat()}
- **Event Type**: {event_type}
- **Avoided Loss**: {avoided_loss_bps:.2f} bps
- **Details**: {details or {}}
"""
            with open(self.soul_path, 'a') as f:
                f.write(entry)


if __name__ == "__main__":
    logger = LOBSoulLogger()
    print("LOB Soul Logger initialized")
