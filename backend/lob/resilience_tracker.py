#!/usr/bin/env python3
"""
Resilience Tracker for Order Book Liquidity Replenishment

This module measures the speed of liquidity replenishment after large sweeps,
detecting potential flash crash conditions when replenishment stalls.

Key Features:
- Real-time resilience measurement
- Flash crash early warning system
- Volume-weighted replenishment tracking
- Strict type hinting for memory safety

Mathematical Foundation:
Resilience = ΔVolume / ΔTime after sweep event
High resilience = fast replenishment (healthy market)
Low resilience = slow/stalled replenishment (flash crash risk)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum, auto
import time
import math
from threading import Lock


class ResilienceState(Enum):
    """Order book resilience state classification."""
    HIGH = auto()       # Fast replenishment (>80% in 1s)
    NORMAL = auto()     # Normal replenishment
    LOW = auto()        # Slow replenishment
    STALLED = auto()    # No/minimal replenishment (flash crash risk)
    CRITICAL = auto()   # Liquidity vacuum detected


@dataclass
class SweepEvent:
    """Represents a liquidity sweep event."""
    timestamp_us: int
    side: str  # 'bid' or 'ask'
    volume_removed: float
    price_impact: float
    levels_swept: int
    pre_sweep_depth: float
    post_sweep_depth: float


@dataclass
class ReplenishmentMetrics:
    """Metrics for liquidity replenishment."""
    # Time to 50% replenishment (microseconds)
    t_50: Optional[float]
    # Time to 80% replenishment (microseconds)
    t_80: Optional[float]
    # Time to 100% replenishment (microseconds)
    t_100: Optional[float]
    # Current replenishment fraction (0-1)
    current_fraction: float
    # Replenishment rate (volume per second)
    rate: float
    # Resilience score (0-1, higher is better)
    resilience_score: float


@dataclass
class ResilienceSnapshot:
    """Point-in-time resilience assessment."""
    timestamp_us: int
    state: ResilienceState
    bid_resilience: float
    ask_resilience: float
    combined_score: float
    flash_crash_risk: float  # 0-1 probability


class ResilienceTracker:
    """
    Track order book resilience by measuring liquidity replenishment
    after large sweeps.
    
    This class monitors how quickly the order book recovers after
    significant liquidity removal events, providing early warning
    of potential flash crash conditions.
    """
    
    def __init__(
        self,
        sweep_threshold_bps: float = 50.0,  # 0.5% price move
        max_tracking_window_s: float = 60.0,
        stalled_threshold_s: float = 5.0,
    ):
        """
        Initialize the resilience tracker.
        
        Args:
            sweep_threshold_bps: Price move threshold to consider as sweep (bps)
            max_tracking_window_s: Maximum time to track replenishment
            stalled_threshold_s: Time after which replenishment is considered stalled
        """
        self.sweep_threshold_bps = sweep_threshold_bps
        self.max_tracking_window_s = max_tracking_window_s
        self.stalled_threshold_s = stalled_threshold_s
        
        # Active sweep events being tracked
        self.active_sweeps: Dict[str, Deque[SweepEvent]] = {
            'bid': deque(),
            'ask': deque(),
        }
        
        # Replenishment progress for each sweep
        self.replenishment_progress: Dict[int, ReplenishmentMetrics] = {}
        
        # Historical resilience scores
        self.resilience_history: Deque[Tuple[int, float]] = deque(maxlen=1000)
        
        # Baseline depth estimates
        self.baseline_depth: Dict[str, float] = {'bid': 0.0, 'ask': 0.0}
        
        # Thread safety
        self._lock = Lock()
        
        # Alert cooldown
        self.last_alert_time: Optional[float] = None
        self.alert_cooldown_s: float = 10.0
    
    def set_baseline_depth(
        self, 
        bid_depth: float, 
        ask_depth: float
    ) -> None:
        """Set baseline depth estimates for resilience calculation."""
        with self._lock:
            self.baseline_depth['bid'] = bid_depth
            self.baseline_depth['ask'] = ask_depth
    
    def record_sweep(self, event: SweepEvent) -> None:
        """
        Record a liquidity sweep event and start tracking replenishment.
        
        Args:
            event: The sweep event to record
        """
        with self._lock:
            if event.side not in self.active_sweeps:
                return
            
            self.active_sweeps[event.side].append(event)
            
            # Initialize replenishment tracking
            sweep_id = id(event)
            self.replenishment_progress[sweep_id] = ReplenishmentMetrics(
                t_50=None,
                t_80=None,
                t_100=None,
                current_fraction=0.0,
                rate=0.0,
                resilience_score=1.0,
            )
            
            # Prune old sweeps
            self._prune_old_sweeps(event.timestamp_us)
    
    def update_replenishment(
        self,
        side: str,
        current_depth: float,
        timestamp_us: int,
    ) -> Optional[ResilienceSnapshot]:
        """
        Update replenishment progress for active sweeps.
        
        Args:
            side: 'bid' or 'ask'
            current_depth: Current depth on that side
            timestamp_us: Current timestamp
            
        Returns:
            ResilienceSnapshot if any sweep completed or stalled
        """
        with self._lock:
            if side not in self.active_sweeps:
                return None
            
            baseline = self.baseline_depth.get(side, current_depth)
            snapshot = None
            
            # Update all active sweeps on this side
            completed_sweeps = []
            
            for sweep in list(self.active_sweeps[side]):
                sweep_id = id(sweep)
                if sweep_id not in self.replenishment_progress:
                    continue
                
                metrics = self.replenishment_progress[sweep_id]
                
                # Calculate target depth (post-sweep + removed volume)
                target_depth = sweep.post_sweep_depth + sweep.volume_removed
                
                # Calculate current replenishment fraction
                if sweep.volume_removed > 0:
                    replenished = current_depth - sweep.post_sweep_depth
                    fraction = min(1.0, replenished / sweep.volume_removed)
                    metrics.current_fraction = fraction
                    
                    # Calculate time elapsed
                    elapsed_s = (timestamp_us - sweep.timestamp_us) / 1_000_000.0
                    
                    # Record milestone times
                    if metrics.t_50 is None and fraction >= 0.5:
                        metrics.t_50 = elapsed_s * 1_000_000
                    if metrics.t_80 is None and fraction >= 0.8:
                        metrics.t_80 = elapsed_s * 1_000_000
                    if metrics.t_100 is None and fraction >= 0.99:
                        metrics.t_100 = elapsed_s * 1_000_000
                        completed_sweeps.append(sweep_id)
                    
                    # Calculate replenishment rate
                    if elapsed_s > 0:
                        metrics.rate = replenished / elapsed_s
                    
                    # Update resilience score
                    metrics.resilience_score = self._calculate_resilience_score(
                        fraction, elapsed_s
                    )
                
                # Check for stalled replenishment
                elapsed_s = (timestamp_us - sweep.timestamp_us) / 1_000_000.0
                if elapsed_s > self.stalled_threshold_s and fraction < 0.3:
                    # Stalled - high flash crash risk
                    snapshot = self._create_snapshot(timestamp_us, side, True)
            
            # Remove completed sweeps
            for sweep_id in completed_sweeps:
                self.replenishment_progress.pop(sweep_id, None)
            
            # Store resilience score
            if snapshot is None and self.active_sweeps[side]:
                avg_resilience = sum(
                    self.replenishment_progress.get(id(s), ReplenishmentMetrics(
                        None, None, None, 0, 0, 1.0
                    )).resilience_score
                    for s in self.active_sweeps[side]
                ) / max(1, len(self.active_sweeps[side]))
                
                self.resilience_history.append((timestamp_us, avg_resilience))
            
            return snapshot
    
    def _calculate_resilience_score(
        self, 
        fraction: float, 
        elapsed_s: float
    ) -> float:
        """Calculate resilience score from replenishment progress."""
        if elapsed_s <= 0:
            return 1.0
        
        # Ideal: 80% replenishment within 1 second
        ideal_t_80 = 1.0
        stalled_t = self.stalled_threshold_s
        
        # Score based on speed of 80% replenishment
        if fraction >= 0.8:
            speed_ratio = ideal_t_80 / elapsed_s
            score = min(1.0, speed_ratio)
        else:
            # Penalize slow partial replenishment
            time_penalty = elapsed_s / stalled_t
            score = max(0.0, fraction - time_penalty * 0.5)
        
        return score
    
    def _create_snapshot(
        self,
        timestamp_us: int,
        side: str,
        is_stalled: bool,
    ) -> ResilienceSnapshot:
        """Create resilience snapshot for current state."""
        # Get recent resilience scores
        recent_scores = [
            score for ts, score in self.resilience_history 
            if (timestamp_us - ts) / 1_000_000 < 60.0
        ]
        
        avg_resilience = sum(recent_scores) / max(1, len(recent_scores))
        
        # Determine state
        if is_stalled:
            state = ResilienceState.STALLED
        elif avg_resilience > 0.8:
            state = ResilienceState.HIGH
        elif avg_resilience > 0.5:
            state = ResilienceState.NORMAL
        elif avg_resilience > 0.2:
            state = ResilienceState.LOW
        else:
            state = ResilienceState.CRITICAL
        
        # Calculate flash crash risk
        flash_crash_risk = 1.0 - avg_resilience
        if is_stalled:
            flash_crash_risk = min(1.0, flash_crash_risk + 0.3)
        
        return ResilienceSnapshot(
            timestamp_us=timestamp_us,
            state=state,
            bid_resilience=avg_resilience if side == 'bid' else 1.0,
            ask_resilience=avg_resilience if side == 'ask' else 1.0,
            combined_score=avg_resilience,
            flash_crash_risk=flash_crash_risk,
        )
    
    def _prune_old_sweeps(self, current_us: int) -> None:
        """Remove sweeps outside tracking window."""
        cutoff_us = int((current_us / 1_000_000 - self.max_tracking_window_s) * 1_000_000)
        
        for side in ['bid', 'ask']:
            while (
                self.active_sweeps[side] 
                and self.active_sweeps[side][0].timestamp_us < cutoff_us
            ):
                old_sweep = self.active_sweeps[side].popleft()
                self.replenishment_progress.pop(id(old_sweep), None)
    
    def get_current_resilience(self) -> ResilienceSnapshot:
        """Get current overall resilience assessment."""
        now_us = int(time.time() * 1_000_000)
        
        with self._lock:
            recent_scores = [
                score for ts, score in self.resilience_history
                if (now_us - ts) / 1_000_000 < 30.0
            ]
            
            if not recent_scores:
                return ResilienceSnapshot(
                    timestamp_us=now_us,
                    state=ResilienceState.NORMAL,
                    bid_resilience=1.0,
                    ask_resilience=1.0,
                    combined_score=1.0,
                    flash_crash_risk=0.0,
                )
            
            avg = sum(recent_scores) / len(recent_scores)
            
            if avg > 0.8:
                state = ResilienceState.HIGH
            elif avg > 0.5:
                state = ResilienceState.NORMAL
            elif avg > 0.3:
                state = ResilienceState.LOW
            else:
                state = ResilienceState.CRITICAL
            
            return ResilienceSnapshot(
                timestamp_us=now_us,
                state=state,
                bid_resilience=avg,
                ask_resilience=avg,
                combined_score=avg,
                flash_crash_risk=1.0 - avg,
            )
    
    def check_flash_crash_risk(self) -> bool:
        """
        Check if flash crash risk is elevated.
        
        Returns:
            True if risk is above threshold
        """
        snapshot = self.get_current_resilience()
        
        if snapshot.flash_crash_risk > 0.7:
            now = time.time()
            if (
                self.last_alert_time is None 
                or now - self.last_alert_time > self.alert_cooldown_s
            ):
                self.last_alert_time = now
                return True
        
        return False
    
    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            for side in self.active_sweeps:
                self.active_sweeps[side].clear()
            self.replenishment_progress.clear()
            self.resilience_history.clear()


if __name__ == "__main__":
    # Example usage
    tracker = ResilienceTracker()
    tracker.set_baseline_depth(bid_depth=1000.0, ask_depth=1000.0)
    
    # Simulate a sweep
    sweep = SweepEvent(
        timestamp_us=int(time.time() * 1_000_000),
        side='bid',
        volume_removed=500.0,
        price_impact=0.005,
        levels_swept=3,
        pre_sweep_depth=1000.0,
        post_sweep_depth=500.0,
    )
    
    tracker.record_sweep(sweep)
    
    # Simulate replenishment over time
    base_time = sweep.timestamp_us
    for i in range(10):
        current_depth = 500.0 + 50.0 * i  # Gradual replenishment
        snapshot = tracker.update_replenishment(
            side='bid',
            current_depth=current_depth,
            timestamp_us=base_time + i * 100_000,
        )
        
        if snapshot:
            print(f"State: {snapshot.state.name}, Risk: {snapshot.flash_crash_risk:.2f}")
    
    # Final assessment
    final = tracker.get_current_resilience()
    print(f"\nFinal Resilience: {final.combined_score:.2f}")
    print(f"Flash Crash Risk: {final.flash_crash_risk:.2f}")
