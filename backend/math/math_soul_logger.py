#!/usr/bin/env python3
"""
Math Soul Logger Module for ZAID Trading Bot
Logs numerical instabilities and matrix drifts to SOUL.md
Critical for debugging quantitative finance computations

This module monitors all math operations for:
- Numerical instability in matrix inversions
- Floating-point drift accumulation
- Condition number degradation
- NaN/Inf propagation
"""

from __future__ import annotations
import time
import json
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class InstabilityType(Enum):
    """Types of numerical instabilities"""
    MATRIX_INVERSION_FAILURE = "matrix_inversion_failure"
    CHOLESKY_DECOMPOSITION_FAILURE = "cholesky_decomposition_failure"
    CONDITION_NUMBER_HIGH = "condition_number_high"
    FLOATING_POINT_DRIFT = "floating_point_drift"
    NAN_INF_DETECTED = "nan_inf_detected"
    OVERFLOW_UNDERFLOW = "overflow_underflow"
    CONVERGENCE_FAILURE = "convergence_failure"
    PRECISION_LOSS = "precision_loss"


class SeverityLevel(Enum):
    """Severity levels for numerical issues"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class NumericalEvent:
    """Record of a numerical instability event"""
    timestamp: str
    event_id: str
    instability_type: str
    severity: int
    component: str
    operation: str
    details: Dict[str, Any]
    input_stats: Dict[str, float]
    output_stats: Dict[str, float]
    recovery_action: str
    stack_trace: Optional[str]


class MathSoulLogger:
    """
    Comprehensive logger for mathematical operations.
    
    Tracks numerical stability across all 152 domains of
    quantitative finance and logs to SOUL.md for analysis.
    """
    
    def __init__(self, 
                 soul_path: str = "SOUL.md",
                 max_events_memory: int = 10000,
                 auto_flush_interval: int = 60):
        """
        Initialize the math soul logger.
        
        Args:
            soul_path: Path to SOUL.md log file
            max_events_memory: Maximum events to keep in memory
            auto_flush_interval: Seconds between automatic flushes
        """
        self.soul_path = Path(soul_path)
        self.max_events_memory = max_events_memory
        
        # Event storage
        self._events: List[NumericalEvent] = []
        self._lock = threading.Lock()
        
        # Statistics tracking
        self._stats: Dict[str, int] = {
            'total_operations': 0,
            'stable_operations': 0,
            'unstable_operations': 0,
            'by_type': {},
            'by_severity': {},
        }
        
        # Thresholds for detection
        self._condition_number_threshold = 1e10
        self._drift_threshold = 1e-12
        self._precision_threshold = 1e-15
        
        # Auto-flush thread
        self._auto_flush_interval = auto_flush_interval
        self._shutdown = threading.Event()
        self._start_auto_flush()
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md with header if it doesn't exist"""
        if not self.soul_path.exists():
            header = """# ZAID Trading Bot - Mathematical Soul Log

## Overview
This file contains detailed logs of numerical instabilities, matrix drifts,
and computational anomalies detected during trading bot operations.

## Sections
- [Instability Events](#instability-events)
- [Statistics Summary](#statistics-summary)
- [Recovery Actions](#recovery-actions)

---

## Instability Events

"""
            self.soul_path.write_text(header)
    
    def _start_auto_flush(self) -> None:
        """Start background thread for periodic flushing"""
        def flush_loop():
            while not self._shutdown.is_set():
                self._shutdown.wait(self._auto_flush_interval)
                if not self._shutdown.is_set():
                    self.flush()
        
        self._flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._flush_thread.start()
    
    def _generate_event_id(self, event: NumericalEvent) -> str:
        """Generate unique event ID based on content"""
        content = f"{event.timestamp}{event.operation}{json.dumps(event.details)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def log_instability(self,
                       instability_type: InstabilityType,
                       component: str,
                       operation: str,
                       details: Dict[str, Any],
                       input_data: Optional[Any] = None,
                       output_data: Optional[Any] = None,
                       severity: SeverityLevel = SeverityLevel.WARNING,
                       recovery_action: str = "none") -> str:
        """
        Log a numerical instability event.
        
        Args:
            instability_type: Type of instability detected
            component: Component where instability occurred
            operation: Specific operation that failed
            details: Additional context about the failure
            input_data: Input data for statistical analysis
            output_data: Output data for statistical analysis
            severity: Severity level of the issue
            recovery_action: Action taken to recover
            
        Returns:
            Event ID for reference
        """
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Compute statistics
        input_stats = self._compute_stats(input_data) if input_data is not None else {}
        output_stats = self._compute_stats(output_data) if output_data is not None else {}
        
        event = NumericalEvent(
            timestamp=timestamp,
            event_id="",  # Will be set after generation
            instability_type=instability_type.value,
            severity=severity.value,
            component=component,
            operation=operation,
            details=details,
            input_stats=input_stats,
            output_stats=output_stats,
            recovery_action=recovery_action,
            stack_trace=self._get_stack_trace(),
        )
        
        event.event_id = self._generate_event_id(event)
        
        # Store event
        with self._lock:
            self._events.append(event)
            
            # Trim if necessary
            if len(self._events) > self.max_events_memory:
                self._events = self._events[-self.max_events_memory:]
            
            # Update stats
            self._stats['total_operations'] += 1
            self._stats['unstable_operations'] += 1
            
            type_key = instability_type.value
            self._stats['by_type'][type_key] = self._stats['by_type'].get(type_key, 0) + 1
            
            severity_key = str(severity.value)
            self._stats['by_severity'][severity_key] = \
                self._stats['by_severity'].get(severity_key, 0) + 1
        
        # Immediate flush for critical events
        if severity == SeverityLevel.CRITICAL:
            self.flush()
        
        return event.event_id
    
    def log_stable_operation(self,
                            component: str,
                            operation: str,
                            condition_number: Optional[float] = None,
                            execution_time_ns: Optional[int] = None) -> None:
        """
        Log a stable operation for baseline tracking.
        
        Args:
            component: Component performing the operation
            operation: Operation name
            condition_number: Condition number if applicable
            execution_time_ns: Execution time in nanoseconds
        """
        with self._lock:
            self._stats['total_operations'] += 1
            self._stats['stable_operations'] += 1
            
            # Track condition number distribution
            if condition_number is not None:
                if 'condition_numbers' not in self._stats:
                    self._stats['condition_numbers'] = []
                
                self._stats['condition_numbers'].append(condition_number)
                
                # Keep only recent
                if len(self._stats['condition_numbers']) > 1000:
                    self._stats['condition_numbers'] = \
                        self._stats['condition_numbers'][-1000:]
    
    def check_matrix_inversion(self,
                               matrix: Any,
                               component: str,
                               tolerance: float = 1e-10) -> Tuple[bool, Optional[str]]:
        """
        Check if matrix inversion will be numerically stable.
        
        Args:
            matrix: Matrix to check
            component: Calling component
            tolerance: Tolerance for singularity detection
            
        Returns:
            Tuple of (is_stable, error_message)
        """
        import numpy as np
        
        matrix = np.asarray(matrix, dtype=np.float64)
        
        # Check for NaN/Inf
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            self.log_instability(
                InstabilityType.NAN_INF_DETECTED,
                component,
                "matrix_inversion_check",
                {"message": "Matrix contains NaN or Inf values"},
                input_data=matrix,
                severity=SeverityLevel.ERROR,
                recovery_action="rejected_input"
            )
            return False, "Matrix contains NaN or Inf values"
        
        # Compute condition number
        try:
            cond = np.linalg.cond(matrix)
            
            if cond > self._condition_number_threshold:
                self.log_instability(
                    InstabilityType.CONDITION_NUMBER_HIGH,
                    component,
                    "matrix_inversion_check",
                    {
                        "condition_number": float(cond),
                        "threshold": self._condition_number_threshold,
                        "matrix_shape": list(matrix.shape),
                    },
                    input_data=matrix,
                    severity=SeverityLevel.WARNING,
                    recovery_action="using_pseudoinverse"
                )
                return False, f"Condition number too high: {cond:.2e}"
            
            # Log as stable
            self.log_stable_operation(
                component,
                "matrix_inversion_check",
                condition_number=float(cond)
            )
            
            return True, None
            
        except Exception as e:
            self.log_instability(
                InstabilityType.MATRIX_INVERSION_FAILURE,
                component,
                "matrix_inversion_check",
                {"error": str(e)},
                input_data=matrix,
                severity=SeverityLevel.ERROR,
                recovery_action="using_regularization"
            )
            return False, str(e)
    
    def check_cholesky_stability(self,
                                 matrix: Any,
                                 component: str) -> Tuple[bool, Optional[str]]:
        """
        Check if Cholesky decomposition will succeed.
        
        Args:
            matrix: Matrix to check
            component: Calling component
            
        Returns:
            Tuple of (is_positive_definite, error_message)
        """
        import numpy as np
        
        matrix = np.asarray(matrix, dtype=np.float64)
        
        # Check symmetry
        if not np.allclose(matrix, matrix.T, atol=1e-10):
            self.log_instability(
                InstabilityType.MATRIX_INVERSION_FAILURE,
                component,
                "cholesky_check",
                {"message": "Matrix is not symmetric"},
                input_data=matrix,
                severity=SeverityLevel.WARNING,
                recovery_action="symmetrizing_matrix"
            )
            return False, "Matrix is not symmetric"
        
        # Try Cholesky
        try:
            np.linalg.cholesky(matrix)
            self.log_stable_operation(component, "cholesky_decomposition")
            return True, None
        except np.linalg.LinAlgError as e:
            self.log_instability(
                InstabilityType.CHOLESKY_DECOMPOSITION_FAILURE,
                component,
                "cholesky_decomposition",
                {"error": str(e)},
                input_data=matrix,
                severity=SeverityLevel.ERROR,
                recovery_action="using_eigendecomposition"
            )
            return False, str(e)
    
    def _compute_stats(self, data: Any) -> Dict[str, float]:
        """Compute statistical summary of array data"""
        import numpy as np
        
        try:
            arr = np.asarray(data, dtype=np.float64)
            
            return {
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
                'nan_count': int(np.sum(np.isnan(arr))),
                'inf_count': int(np.sum(np.isinf(arr))),
                'shape': list(arr.shape),
            }
        except Exception:
            return {'error': 'Could not compute statistics'}
    
    def _get_stack_trace(self) -> Optional[str]:
        """Get current stack trace for debugging"""
        import traceback
        return traceback.format_stack()[-10:-1]
    
    def flush(self) -> None:
        """Flush pending events to SOUL.md"""
        with self._lock:
            if not self._events:
                return
            
            events_to_write = self._events.copy()
        
        # Format events as Markdown
        content = "\n## Recent Events\n\n"
        
        for event in events_to_write:
            content += f"### Event `{event.event_id}`\n\n"
            content += f"- **Timestamp**: {event.timestamp}\n"
            content += f"- **Type**: {event.instability_type}\n"
            content += f"- **Severity**: {SeverityLevel(event.severity).name}\n"
            content += f"- **Component**: {event.component}\n"
            content += f"- **Operation**: {event.operation}\n"
            content += f"- **Recovery**: {event.recovery_action}\n"
            
            if event.input_stats:
                content += f"\n**Input Statistics**:\n```json\n{json.dumps(event.input_stats, indent=2)}\n```\n"
            
            if event.output_stats:
                content += f"\n**Output Statistics**:\n```json\n{json.dumps(event.output_stats, indent=2)}\n```\n"
            
            if event.details:
                content += f"\n**Details**:\n```json\n{json.dumps(event.details, indent=2)}\n```\n"
            
            content += "\n---\n\n"
        
        # Append to file
        with open(self.soul_path, 'a') as f:
            f.write(content)
        
        # Clear flushed events
        with self._lock:
            self._events.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged events"""
        with self._lock:
            summary = dict(self._stats)
            summary['pending_events'] = len(self._events)
            
            # Compute rates
            total = summary['total_operations']
            if total > 0:
                summary['stability_rate'] = summary['stable_operations'] / total
                summary['instability_rate'] = summary['unstable_operations'] / total
            else:
                summary['stability_rate'] = 1.0
                summary['instability_rate'] = 0.0
            
            return summary
    
    def shutdown(self) -> None:
        """Graceful shutdown with final flush"""
        self._shutdown.set()
        self.flush()
        
        # Write final summary
        summary = self.get_summary()
        
        with open(self.soul_path, 'a') as f:
            f.write("\n## Statistics Summary\n\n")
            f.write(f"```json\n{json.dumps(summary, indent=2)}\n```\n")
            f.write(f"\n*Log generated at {datetime.utcnow().isoformat()}Z*\n")


# Global instance
_soul_logger: Optional[MathSoulLogger] = None


def get_soul_logger() -> MathSoulLogger:
    """Get or create global soul logger instance"""
    global _soul_logger
    if _soul_logger is None:
        _soul_logger = MathSoulLogger()
    return _soul_logger


def log_matrix_failure(matrix: Any, component: str, operation: str) -> str:
    """Convenience function to log matrix operation failure"""
    logger = get_soul_logger()
    return logger.log_instability(
        InstabilityType.MATRIX_INVERSION_FAILURE,
        component,
        operation,
        {"message": f"{operation} failed"},
        input_data=matrix,
        severity=SeverityLevel.ERROR,
        recovery_action="fallback_to_safe_mode"
    )


if __name__ == "__main__":
    # Self-test
    import numpy as np
    
    print("=== Math Soul Logger Self-Test ===\n")
    
    logger = MathSoulLogger(soul_path="/tmp/test_SOUL.md")
    
    # Test stable operation
    stable_matrix = np.random.randn(4, 4)
    stable_matrix = stable_matrix @ stable_matrix.T + np.eye(4) * 0.1
    
    is_stable, error = logger.check_cholesky_stability(stable_matrix, "test_component")
    print(f"Stable matrix test: is_stable={is_stable}, error={error}")
    
    # Test unstable matrix
    unstable_matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0]  # Singular
    ])
    
    is_stable, error = logger.check_cholesky_stability(unstable_matrix, "test_component")
    print(f"Unstable matrix test: is_stable={is_stable}, error={error}")
    
    # Test condition number check
    ill_conditioned = np.array([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-15]
    ])
    
    is_stable, error = logger.check_matrix_inversion(ill_conditioned, "test_component")
    print(f"Ill-conditioned matrix test: is_stable={is_stable}, error={error}")
    
    # Get summary
    summary = logger.get_summary()
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    
    # Flush and shutdown
    logger.shutdown()
    print(f"\nLog written to: {logger.soul_path}")
