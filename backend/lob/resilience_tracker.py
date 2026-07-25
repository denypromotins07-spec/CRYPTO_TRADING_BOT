#!/usr/bin/env python3
"""
Order Book Resilience Tracker

Measures the speed of liquidity replenishment after large sweeps.
Instantly flags if liquidity replenishment stalls, indicating potential flash crash.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque
from enum import Enum
import time
import math
from threading import Lock


class ResilienceState(Enum):
    NORMAL = "normal"
    STRESSED = "stressed"
    CRITICAL = "critical"
    FLASH_CRASH_RISK = "flash_crash_risk"


@dataclass(slots=True)
class LiquidityEvent:
    timestamp: float
    side: str  # "bid" or "ask"
    volume_removed: float
    volume_replenished: float
    recovery_time_ms: float
    depth_before: float
    depth_after: float


@dataclass(slots=True)
class ResilienceMetrics:
    current_state: ResilienceState
    replenishment_rate: float  # Volume per second
    recovery_half_life_ms: float
    stress_level: float  # 0-1
    flash_crash_probability: float
    avg_recovery_time_ms: float


class ResilienceTracker:
    """Track order book resilience and flag potential flash crashes."""
    
    def __init__(self, max_events: int = 500):
        self.events: Deque[LiquidityEvent] = deque(maxlen=max_events)
        self._lock = Lock()
        self.baseline_depth: float = 1000.0
        self.stress_threshold: float = 0.5
        self.critical_threshold: float = 0.8
        
    def record_sweep(self, side: str, volume_removed: float, 
                     depth_before: float, depth_after: float) -> None:
        """Record a liquidity sweep event."""
        with self._lock:
            event = LiquidityEvent(
                timestamp=time.time(),
                side=side,
                volume_removed=volume_removed,
                volume_replenished=0.0,
                recovery_time_ms=0.0,
                depth_before=depth_before,
                depth_after=depth_after,
            )
            self.events.append(event)
    
    def update_replenishment(self, volume_added: float, side: str) -> None:
        """Update replenishment for most recent sweep on given side."""
        with self._lock:
            for event in reversed(self.events):
                if event.side == side and event.volume_replenished == 0.0:
                    elapsed_ms = (time.time() - event.timestamp) * 1000
                    event.volume_replenished = volume_added
                    event.recovery_time_ms = elapsed_ms
                    break
    
    def get_metrics(self) -> ResilienceMetrics:
        """Get current resilience metrics."""
        with self._lock:
            if not self.events:
                return ResilienceMetrics(
                    current_state=ResilienceState.NORMAL,
                    replenishment_rate=0.0,
                    recovery_half_life_ms=0.0,
                    stress_level=0.0,
                    flash_crash_probability=0.0,
                    avg_recovery_time_ms=0.0,
                )
            
            # Calculate average recovery time
            recovery_times = [e.recovery_time_ms for e in self.events if e.recovery_time_ms > 0]
            avg_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else 1000.0
            
            # Calculate replenishment rate
            total_replenished = sum(e.volume_replenished for e in self.events)
            total_removed = sum(e.volume_removed for e in self.events)
            replenishment_rate = total_replenished / max(total_removed, 0.001)
            
            # Determine stress level
            stress_level = 1.0 - min(replenishment_rate, 1.0)
            
            # Flash crash probability based on stalled replenishment
            flash_prob = 0.0
            if avg_recovery > 5000:  # > 5 seconds
                flash_prob = min((avg_recovery - 5000) / 10000, 0.9)
            
            # Determine state
            if flash_prob > 0.5:
                state = ResilienceState.FLASH_CRASH_RISK
            elif stress_level > self.critical_threshold:
                state = ResilienceState.CRITICAL
            elif stress_level > self.stress_threshold:
                state = ResilienceState.STRESSED
            else:
                state = ResilienceState.NORMAL
            
            return ResilienceMetrics(
                current_state=state,
                replenishment_rate=replenishment_rate,
                recovery_half_life_ms=avg_recovery * 0.5,
                stress_level=stress_level,
                flash_crash_probability=flash_prob,
                avg_recovery_time_ms=avg_recovery,
            )
    
    def is_flash_crash_risk(self) -> bool:
        """Check if current state indicates flash crash risk."""
        return self.get_metrics().current_state == ResilienceState.FLASH_CRASH_RISK


if __name__ == "__main__":
    tracker = ResilienceTracker()
    print("Resilience Tracker initialized")
