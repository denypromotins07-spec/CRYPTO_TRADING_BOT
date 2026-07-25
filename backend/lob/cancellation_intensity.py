#!/usr/bin/env python3
"""Cancellation Intensity Analyzer - Analyzes ratio of cancellations to executions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Deque
from collections import deque
import time
from threading import Lock


@dataclass(slots=True)
class CancellationEvent:
    timestamp: float
    side: str
    volume_cancelled: float
    volume_executed: float


@dataclass(slots=True)
class CancellationMetrics:
    cancellation_ratio: float
    intensity: float  # Cancellations per second
    trend: str  # "increasing", "decreasing", "stable"
    stress_signal: bool


class CancellationIntensityAnalyzer:
    """Analyze cancellation to execution ratios."""
    
    def __init__(self, window_seconds: float = 60.0):
        self.events: Deque[CancellationEvent] = deque()
        self.window_seconds = window_seconds
        self._lock = Lock()
        
    def record_cancellation(self, side: str, volume: float) -> None:
        with self._lock:
            event = CancellationEvent(
                timestamp=time.time(),
                side=side,
                volume_cancelled=volume,
                volume_executed=0.0,
            )
            self.events.append(event)
            self._cleanup()
    
    def record_execution(self, side: str, volume: float) -> None:
        with self._lock:
            for event in reversed(self.events):
                if event.side == side and event.volume_executed == 0.0:
                    event.volume_executed = volume
                    break
            self._cleanup()
    
    def _cleanup(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()
    
    def get_metrics(self) -> CancellationMetrics:
        with self._lock:
            if not self.events:
                return CancellationMetrics(0.0, 0.0, "stable", False)
            
            total_cancelled = sum(e.volume_cancelled for e in self.events)
            total_executed = sum(e.volume_executed for e in self.events)
            
            ratio = total_cancelled / max(total_executed, 0.001)
            intensity = len(self.events) / self.window_seconds
            
            # Determine trend
            half = len(self.events) // 2
            first_half_ratio = sum(e.volume_cancelled for e in list(self.events)[:half]) / max(
                sum(e.volume_executed for e in list(self.events)[:half]), 0.001)
            second_half_ratio = sum(e.volume_cancelled for e in list(self.events)[half:]) / max(
                sum(e.volume_executed for e in list(self.events)[half:]), 0.001)
            
            if second_half_ratio > first_half_ratio * 1.2:
                trend = "increasing"
            elif second_half_ratio < first_half_ratio * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Stress signal if ratio > 3 (more cancels than executes)
            stress = ratio > 3.0
            
            return CancellationMetrics(ratio, intensity, trend, stress)


if __name__ == "__main__":
    analyzer = CancellationIntensityAnalyzer()
    print("Cancellation Intensity Analyzer initialized")
