#!/usr/bin/env python3
"""
SSM Soul Logger: Sequence Prediction and ODE Stability Logging

This module logs sequence prediction failures, ODE instabilities, and
micro-reversal predictions to SOUL.md for audit and analysis.

Key Features:
- Microsecond-level timestamp precision
- Structured logging of prediction errors
- ODE stability monitoring with divergence detection
- Integration with hedging soul logger for unified audit trail
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class SequencePredictionLog:
    """Log entry for sequence prediction events."""
    timestamp_us: int
    event_type: str  # 'prediction', 'failure', 'micro_reversal'
    model_name: str
    sequence_length: int
    predicted_value: float
    actual_value: Optional[float]
    prediction_error: float
    confidence: float
    hidden_state_norm: float
    metadata: Dict[str, Any]


@dataclass
class ODEStabilityLog:
    """Log entry for ODE solver stability events."""
    timestamp_us: int
    event_type: str  # 'stable', 'unstable', 'divergence', 'flash_crash'
    solver_name: str
    step_size: float
    state_norm_before: float
    state_norm_after: float
    derivative_norm: float
    is_divergent: bool
    recovery_action: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class GammaProfitLog:
    """Log entry for gamma scalping profit events."""
    timestamp_us: int
    event_type: str  # 'gamma_profit', 'theta_bleed', 'hedge_execution'
    position_id: str
    underlying_asset: str
    gamma_pnl: float
    theta_cost: float
    net_pnl: float
    delta_hedged: float
    implied_vol_change: float
    metadata: Dict[str, Any]


class SsmSoulLogger:
    """
    Centralized logger for SSM inference and ODE stability events.
    
    Writes structured logs to SOUL.md for audit, debugging, and
    performance analysis.
    """
    
    def __init__(self, soul_md_path: str = "SOUL.md"):
        self.soul_md_path = Path(soul_md_path)
        self.log_buffer: List[Dict[str, Any]] = []
        self.buffer_size_limit = 100  # Flush after this many entries
        
        # Statistics tracking
        self.stats = {
            'total_predictions': 0,
            'prediction_failures': 0,
            'micro_reversals_detected': 0,
            'ode_instabilities': 0,
            'gamma_profits': 0,
            'theta_losses': 0,
        }
        
        # Ensure SOUL.md exists
        self._initialize_soul_file()
    
    def _initialize_soul_file(self) -> None:
        """Initialize SOUL.md with header if it doesn't exist."""
        if not self.soul_md_path.exists():
            header = """# ZAID Personal Crypto Trading Bot - SOUL.md

## System Operational Understanding Log

This file contains the complete audit trail for:
- Sequence prediction failures and successes
- ODE solver instabilities and recoveries  
- Gamma scalping profits and theta decay
- Hedging correlation breakdowns
- Micro-reversal predictions

---

"""
            with open(self.soul_md_path, 'w') as f:
                f.write(header)
    
    def _get_timestamp_us(self) -> int:
        """Get current timestamp in microseconds."""
        now = datetime.utcnow()
        return int(now.timestamp() * 1_000_000)
    
    def _format_timestamp(self, timestamp_us: int) -> str:
        """Format microsecond timestamp as ISO string."""
        dt = datetime.utcfromtimestamp(timestamp_us / 1_000_000)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    def log_sequence_prediction(
        self,
        model_name: str,
        sequence: np.ndarray,
        prediction: float,
        actual: Optional[float] = None,
        confidence: float = 0.0,
        hidden_state: Optional[np.ndarray] = None,
        is_micro_reversal: bool = False,
        **metadata
    ) -> None:
        """
        Log a sequence prediction event.
        
        Args:
            model_name: Name of the SSM model
            sequence: Input sequence
            prediction: Model prediction
            actual: Actual observed value (if available)
            confidence: Prediction confidence [0, 1]
            hidden_state: Model hidden state at prediction time
            is_micro_reversal: Whether this predicts a micro-reversal
            metadata: Additional context
        """
        timestamp_us = self._get_timestamp_us()
        
        # Compute prediction error
        prediction_error = abs(prediction - actual) if actual is not None else 0.0
        
        # Compute hidden state norm
        hidden_state_norm = float(np.linalg.norm(hidden_state)) if hidden_state is not None else 0.0
        
        # Determine event type
        if is_micro_reversal:
            event_type = 'micro_reversal'
            self.stats['micro_reversals_detected'] += 1
        elif actual is not None and prediction_error > 0.1:  # Threshold for failure
            event_type = 'failure'
            self.stats['prediction_failures'] += 1
        else:
            event_type = 'prediction'
        
        self.stats['total_predictions'] += 1
        
        log_entry = SequencePredictionLog(
            timestamp_us=timestamp_us,
            event_type=event_type,
            model_name=model_name,
            sequence_length=len(sequence),
            predicted_value=float(prediction),
            actual_value=float(actual) if actual is not None else None,
            prediction_error=prediction_error,
            confidence=confidence,
            hidden_state_norm=hidden_state_norm,
            metadata=metadata,
        )
        
        self._write_log('sequence_prediction', asdict(log_entry))
    
    def log_ode_stability(
        self,
        solver_name: str,
        step_size: float,
        state_before: np.ndarray,
        state_after: np.ndarray,
        derivative: np.ndarray,
        is_flash_crash: bool = False,
        recovery_action: Optional[str] = None,
        **metadata
    ) -> None:
        """
        Log ODE solver stability event.
        
        Args:
            solver_name: Name of the ODE solver
            step_size: Step size used
            state_before: State before integration step
            state_after: State after integration step
            derivative: Computed derivative
            is_flash_crash: Whether triggered by flash crash
            recovery_action: Action taken for instability
            metadata: Additional context
        """
        timestamp_us = self._get_timestamp_us()
        
        norm_before = float(np.linalg.norm(state_before))
        norm_after = float(np.linalg.norm(state_after))
        deriv_norm = float(np.linalg.norm(derivative))
        
        # Detect divergence (state norm explosion)
        is_divergent = norm_after > norm_before * 10 or not np.isfinite(norm_after)
        
        if is_divergent or is_flash_crash:
            event_type = 'divergence' if is_divergent else 'flash_crash'
            self.stats['ode_instabilities'] += 1
        elif deriv_norm > 100:  # High derivative indicates potential instability
            event_type = 'unstable'
            self.stats['ode_instabilities'] += 1
        else:
            event_type = 'stable'
        
        log_entry = ODEStabilityLog(
            timestamp_us=timestamp_us,
            event_type=event_type,
            solver_name=solver_name,
            step_size=step_size,
            state_norm_before=norm_before,
            state_norm_after=norm_after,
            derivative_norm=deriv_norm,
            is_divergent=is_divergent,
            recovery_action=recovery_action,
            metadata=metadata,
        )
        
        self._write_log('ode_stability', asdict(log_entry))
    
    def log_gamma_profit(
        self,
        position_id: str,
        underlying: str,
        gamma_pnl: float,
        theta_cost: float,
        delta_hedged: float,
        iv_change: float,
        **metadata
    ) -> None:
        """
        Log gamma scalping profit/loss event.
        
        Args:
            position_id: Unique position identifier
            underlying: Underlying asset symbol
            gamma_pnl: PnL from gamma scalping
            theta_cost: Time decay cost
            delta_hedged: Delta hedged amount
            iv_change: Implied volatility change
            metadata: Additional context
        """
        timestamp_us = self._get_timestamp_us()
        net_pnl = gamma_pnl - theta_cost
        
        if net_pnl > 0:
            event_type = 'gamma_profit'
            self.stats['gamma_profits'] += 1
        else:
            event_type = 'theta_bleed'
            self.stats['theta_losses'] += 1
        
        log_entry = GammaProfitLog(
            timestamp_us=timestamp_us,
            event_type=event_type,
            position_id=position_id,
            underlying_asset=underlying,
            gamma_pnl=gamma_pnl,
            theta_cost=theta_cost,
            net_pnl=net_pnl,
            delta_hedged=delta_hedged,
            implied_vol_change=iv_change,
            metadata=metadata,
        )
        
        self._write_log('gamma_profit', asdict(log_entry))
    
    def _write_log(self, category: str, entry: Dict[str, Any]) -> None:
        """Write a log entry to buffer and potentially flush."""
        log_record = {
            'category': category,
            'timestamp_formatted': self._format_timestamp(entry['timestamp_us']),
            **entry
        }
        
        self.log_buffer.append(log_record)
        
        # Flush if buffer is full
        if len(self.log_buffer) >= self.buffer_size_limit:
            self.flush()
    
    def flush(self) -> None:
        """Flush buffered logs to SOUL.md."""
        if not self.log_buffer:
            return
        
        with open(self.soul_md_path, 'a') as f:
            for entry in self.log_buffer:
                # Write as JSON line for easy parsing
                f.write(f"```json\n{json.dumps(entry)}\n```\n\n")
        
        self.log_buffer.clear()
    
    def write_summary(self) -> None:
        """Write summary statistics to SOUL.md."""
        self.flush()  # Flush any pending logs first
        
        summary = f"""
## Summary Statistics

Generated at: {self._format_timestamp(self._get_timestamp_us())}

| Metric | Count |
|--------|-------|
| Total Predictions | {self.stats['total_predictions']} |
| Prediction Failures | {self.stats['prediction_failures']} |
| Micro-Reversals Detected | {self.stats['micro_reversals_detected']} |
| ODE Instabilities | {self.stats['ode_instabilities']} |
| Gamma Profits | {self.stats['gamma_profits']} |
| Theta Losses | {self.stats['theta_losses']} |

### Rates

- Prediction Failure Rate: {self.stats['prediction_failures'] / max(self.stats['total_predictions'], 1) * 100:.2f}%
- Gamma Win Rate: {self.stats['gamma_profits'] / max(self.stats['gamma_profits'] + self.stats['theta_losses'], 1) * 100:.2f}%

---

"""
        with open(self.soul_md_path, 'a') as f:
            f.write(summary)
    
    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        for key in self.stats:
            self.stats[key] = 0


def create_correlation_breakdown_log(
    logger: SsmSoulLogger,
    correlation_matrix: np.ndarray,
    expected_correlation: np.ndarray,
    breakdown_threshold: float = 0.5,
) -> None:
    """
    Log a correlation breakdown event that causes temporary directional exposure.
    
    This is called when hedges fail due to correlation breakdown between
    the hedging instrument and the underlying exposure.
    """
    timestamp_us = logger._get_timestamp_us()
    
    # Compute correlation deviation
    deviation = np.abs(correlation_matrix - expected_correlation)
    max_deviation = float(np.max(deviation))
    
    if max_deviation > breakdown_threshold:
        log_entry = {
            'timestamp_us': timestamp_us,
            'event_type': 'correlation_breakdown',
            'max_deviation': max_deviation,
            'threshold': breakdown_threshold,
            'directional_exposure': True,
            'metadata': {
                'correlation_matrix_shape': list(correlation_matrix.shape),
                'breakdown_indices': np.where(deviation > breakdown_threshold),
            }
        }
        
        logger._write_log('correlation_breakdown', log_entry)
        
        # Update SOUL.md with alert
        with open(logger.soul_md_path, 'a') as f:
            f.write(f"""
### ⚠️ CORRELATION BREAKDOWN ALERT

**Time**: {logger._format_timestamp(timestamp_us)}
**Max Deviation**: {max_deviation:.4f} (threshold: {breakdown_threshold})
**Status**: Temporary directional exposure active

The hedging correlation has broken down. The bot is temporarily exposed 
to directional risk until correlation normalizes or manual intervention occurs.

""")


def benchmark_soul_logger() -> None:
    """Benchmark the SSM soul logger."""
    import time
    
    logger = SsmSoulLogger("SOUL.md")
    
    print("Logging sequence predictions...")
    np.random.seed(42)
    
    for i in range(50):
        seq = np.random.randn(100, 16)
        pred = np.random.randn()
        actual = np.random.randn()
        
        logger.log_sequence_prediction(
            model_name='mamba_v1',
            sequence=seq,
            prediction=pred,
            actual=actual,
            confidence=np.random.rand(),
            hidden_state=np.random.randn(64),
            is_micro_reversal=(i % 10 == 0),
        )
    
    print("Logging ODE stability events...")
    for i in range(20):
        state_before = np.random.randn(32)
        
        # Simulate occasional instability
        if i % 7 == 0:
            state_after = state_before * 15  # Divergence
        else:
            state_after = state_before + np.random.randn(32) * 0.1
        
        logger.log_ode_stability(
            solver_name='rk4_ltc',
            step_size=0.01,
            state_before=state_before,
            state_after=state_after,
            derivative=np.random.randn(32),
            is_flash_crash=(i % 15 == 0),
        )
    
    print("Logging gamma profits...")
    for i in range(30):
        gamma_pnl = np.random.randn() * 100
        theta_cost = abs(np.random.randn()) * 10
        
        logger.log_gamma_profit(
            position_id=f'pos_{i:03d}',
            underlying='BTC-PERP',
            gamma_pnl=gamma_pnl,
            theta_cost=theta_cost,
            delta_hedged=np.random.randn() * 0.5,
            iv_change=np.random.randn() * 0.02,
        )
    
    # Flush and write summary
    logger.write_summary()
    
    stats = logger.get_stats()
    print(f"\nLogging complete. Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print(f"\nLogs written to: {logger.soul_md_path.absolute()}")


if __name__ == "__main__":
    benchmark_soul_logger()
