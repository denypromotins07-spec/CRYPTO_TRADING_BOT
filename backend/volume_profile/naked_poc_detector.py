#!/usr/bin/env python3
"""
backend/volume_profile/naked_poc_detector.py

Identifies untested VPOCs (Point of Control) from previous sessions as magnetic price targets.
These "naked" POCs often act as magnets for price action as the market seeks to rebalance.

Features:
- Tracks historical VPOCs across multiple sessions
- Identifies which POCs remain untested (naked)
- Calculates magnetic strength based on volume and time
- Strict type hinting for production reliability
- Cross-platform compatibility optimized for Windows PowerShell
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, auto
import time


class PocStatus(Enum):
    """Status of a Point of Control."""
    NAKED = auto()      # Never tested since formation
    TESTED_ONCE = auto()  # Tested once, may still be relevant
    TESTED_MULTIPLE = auto()  # Tested multiple times, less significant
    BREACHED = auto()   # Price moved through and closed beyond


@dataclass
class HistoricalPoc:
    """Represents a historical Point of Control."""
    price: float
    session_date: str  # YYYY-MM-DD format
    session_type: str  # 'regular', 'overnight', 'asian', etc.
    volume_at_poc: float
    total_session_volume: float
    timestamp_ns: int
    status: PocStatus = PocStatus.NAKED
    test_count: int = 0
    last_test_timestamp_ns: Optional[int] = None
    
    @property
    def age_hours(self) -> float:
        """Age of POC in hours."""
        return (time.time_ns() - self.timestamp_ns) / 3_600_000_000_000
    
    @property
    def volume_significance(self) -> float:
        """Ratio of POC volume to total session volume."""
        if self.total_session_volume == 0:
            return 0.0
        return self.volume_at_poc / self.total_session_volume
    
    @property
    def magnetic_strength(self) -> float:
        """
        Calculate magnetic strength based on:
        - Volume significance (higher = stronger)
        - Age (newer = stronger, but some decay)
        - Test count (fewer tests = stronger)
        """
        vol_factor = min(self.volume_significance * 2, 1.0)
        
        # Age factor: peaks around 4-24 hours, then decays
        age_h = self.age_hours
        if age_h < 1:
            age_factor = 0.5  # Too fresh
        elif age_h < 4:
            age_factor = 0.8
        elif age_h < 24:
            age_factor = 1.0  # Prime magnet zone
        elif age_h < 72:
            age_factor = 0.7
        else:
            age_factor = 0.4
        
        # Test factor: naked is strongest
        if self.test_count == 0:
            test_factor = 1.0
        elif self.test_count == 1:
            test_factor = 0.6
        else:
            test_factor = 0.3
        
        return vol_factor * age_factor * test_factor


@dataclass
class MagneticTarget:
    """A naked POC identified as a potential price target."""
    poc: HistoricalPoc
    current_price: float
    distance_points: float
    distance_percent: float
    direction: str  # 'above' or 'below'
    estimated_attraction_time_minutes: float
    confidence_score: float  # 0.0 to 1.0


class NakedPOCDetector:
    """
    Detects and tracks naked (untested) POCs as magnetic price targets.
    """
    
    def __init__(self, max_history_sessions: int = 50):
        # Historical POCs by symbol
        self.poc_history: Dict[str, deque[HistoricalPoc]] = defaultdict(
            lambda: deque(maxlen=max_history_sessions)
        )
        
        # Track price touches for each POC level
        self.touch_tracking: Dict[str, Dict[float, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Configuration
        self.touch_threshold_points = 0.5  # Price within this range counts as touch
        self.max_age_hours = 168  # 1 week max relevance
    
    def add_poc(
        self,
        symbol: str,
        price: float,
        session_date: str,
        session_type: str,
        volume_at_poc: float,
        total_session_volume: float,
        timestamp_ns: Optional[int] = None
    ) -> None:
        """Record a new POC for tracking."""
        ts = timestamp_ns or time.time_ns()
        
        poc = HistoricalPoc(
            price=price,
            session_date=session_date,
            session_type=session_type,
            volume_at_poc=volume_at_poc,
            total_session_volume=total_session_volume,
            timestamp_ns=ts
        )
        
        self.poc_history[symbol].append(poc)
        
        # Initialize touch tracking
        if price not in self.touch_tracking[symbol]:
            self.touch_tracking[symbol][price] = []
    
    def record_price_tick(
        self,
        symbol: str,
        price: float,
        timestamp_ns: Optional[int] = None
    ) -> List[HistoricalPoc]:
        """
        Record a price tick and check if it touches any naked POCs.
        Returns list of POCs that were touched.
        """
        ts = timestamp_ns or time.time_ns()
        touched_pocs = []
        
        if symbol not in self.poc_history:
            return touched_pocs
        
        for poc in self.poc_history[symbol]:
            if abs(poc.price - price) <= self.touch_threshold_points:
                # This is a touch
                if poc.status == PocStatus.NAKED:
                    poc.status = PocStatus.TESTED_ONCE
                    poc.test_count = 1
                    poc.last_test_timestamp_ns = ts
                    touched_pocs.append(poc)
                elif poc.status == PocStatus.TESTED_ONCE:
                    poc.status = PocStatus.TESTED_MULTIPLE
                    poc.test_count += 1
                    poc.last_test_timestamp_ns = ts
                    touched_pocs.append(poc)
                
                # Record touch timestamp
                self.touch_tracking[symbol][poc.price].append(ts)
        
        return touched_pocs
    
    def get_naked_pocs(
        self,
        symbol: str,
        current_price: float,
        search_range_points: float = 500.0
    ) -> List[HistoricalPoc]:
        """Get all naked POCs within search range."""
        if symbol not in self.poc_history:
            return []
        
        cutoff_ts = time.time_ns() - int(self.max_age_hours * 3_600_000_000_000)
        
        naked = []
        for poc in self.poc_history[symbol]:
            if poc.timestamp_ns < cutoff_ts:
                continue  # Too old
            
            if poc.status != PocStatus.NAKED:
                continue
            
            if abs(poc.price - current_price) > search_range_points:
                continue
            
            naked.append(poc)
        
        # Sort by magnetic strength
        naked.sort(key=lambda p: p.magnetic_strength, reverse=True)
        return naked
    
    def identify_magnetic_targets(
        self,
        symbol: str,
        current_price: float,
        search_range_points: float = 500.0
    ) -> List[MagneticTarget]:
        """
        Identify naked POCs as magnetic price targets.
        Returns sorted list of targets with confidence scores.
        """
        naked_pocs = self.get_naked_pocs(symbol, current_price, search_range_points)
        targets = []
        
        for poc in naked_pocs:
            distance = poc.price - current_price
            distance_abs = abs(distance)
            
            if distance_abs == 0:
                continue  # Already at POC
            
            direction = 'above' if distance > 0 else 'below'
            distance_percent = (distance_abs / current_price) * 100 if current_price > 0 else 0
            
            # Estimate attraction time based on recent volatility
            # Simplified: assume average movement of 1% per hour for crypto
            avg_hourly_move_pct = 1.0
            estimated_hours = distance_percent / avg_hourly_move_pct if avg_hourly_move_pct > 0 else float('inf')
            estimated_minutes = estimated_hours * 60
            
            # Confidence based on magnetic strength and distance
            strength_factor = poc.magnetic_strength
            distance_factor = max(0.3, 1.0 - (distance_abs / search_range_points))
            confidence = min(strength_factor * distance_factor + 0.2, 1.0)
            
            target = MagneticTarget(
                poc=poc,
                current_price=current_price,
                distance_points=distance_abs,
                distance_percent=distance_percent,
                direction=direction,
                estimated_attraction_time_minutes=estimated_minutes,
                confidence_score=confidence
            )
            targets.append(target)
        
        # Sort by confidence
        targets.sort(key=lambda t: t.confidence_score, reverse=True)
        return targets
    
    def get_nearest_naked_poc(
        self,
        symbol: str,
        current_price: float,
        search_range_points: float = 500.0
    ) -> Optional[MagneticTarget]:
        """Get the nearest naked POC target."""
        targets = self.identify_magnetic_targets(symbol, current_price, search_range_points)
        return targets[0] if targets else None
    
    def get_all_pocs_summary(self, symbol: str) -> Dict[str, int]:
        """Get summary count of POCs by status."""
        if symbol not in self.poc_history:
            return {}
        
        summary: Dict[str, int] = defaultdict(int)
        for poc in self.poc_history[symbol]:
            summary[poc.status.name] += 1
        
        return dict(summary)
    
    def cleanup_old_data(self, symbol: str, max_age_hours: int = 168) -> int:
        """Remove POCs older than specified age. Returns count removed."""
        if symbol not in self.poc_history:
            return 0
        
        cutoff_ts = time.time_ns() - int(max_age_hours * 3_600_000_000_000)
        original_len = len(self.poc_history[symbol])
        
        # Filter out old POCs
        self.poc_history[symbol] = deque(
            [p for p in self.poc_history[symbol] if p.timestamp_ns > cutoff_ts],
            maxlen=self.poc_history[symbol].maxlen
        )
        
        removed = original_len - len(self.poc_history[symbol])
        return removed
    
    def detect_unfinished_auction_near_poc(
        self,
        symbol: str,
        current_price: float,
        range_points: float = 10.0
    ) -> Optional[Tuple[HistoricalPoc, float]]:
        """
        Detect if current price is near a naked POC with unfinished auction characteristics.
        Returns (POC, distance) if found.
        """
        naked = self.get_naked_pocs(symbol, current_price, range_points)
        
        if not naked:
            return None
        
        # Find closest
        closest = min(naked, key=lambda p: abs(p.price - current_price))
        distance = abs(closest.price - current_price)
        
        return (closest, distance)


def create_sample_detector() -> NakedPOCDetector:
    """Create detector with sample data for testing."""
    detector = NakedPOCDetector()
    
    # Add some historical POCs
    base_time = time.time_ns() - int(24 * 3_600_000_000_000)  # 24 hours ago
    
    detector.add_poc(
        symbol="BTCUSDT",
        price=95000.0,
        session_date="2024-01-14",
        session_type="regular",
        volume_at_poc=1500.0,
        total_session_volume=50000.0,
        timestamp_ns=base_time
    )
    
    detector.add_poc(
        symbol="BTCUSDT",
        price=98000.0,
        session_date="2024-01-15",
        session_type="regular",
        volume_at_poc=2000.0,
        total_session_volume=60000.0,
        timestamp_ns=base_time + int(24 * 3_600_000_000_000)
    )
    
    return detector


if __name__ == "__main__":
    # Example usage
    detector = create_sample_detector()
    
    current_price = 96500.0
    
    print("Naked POC Analysis for BTCUSDT")
    print("=" * 50)
    
    targets = detector.identify_magnetic_targets("BTCUSDT", current_price)
    
    if targets:
        print(f"\nCurrent Price: ${current_price:,.2f}")
        print(f"\nMagnetic Targets ({len(targets)} found):")
        
        for i, target in enumerate(targets[:5], 1):
            print(f"\n{i}. Target: ${target.poc.price:,.2f}")
            print(f"   Direction: {target.direction}")
            print(f"   Distance: ${target.distance_points:,.2f} ({target.distance_percent:.2f}%)")
            print(f"   Est. Time: {target.estimated_attraction_time_minutes:.1f} minutes")
            print(f"   Confidence: {target.confidence_score:.2f}")
            print(f"   Session: {target.poc.session_date} ({target.poc.session_type})")
            print(f"   Volume Significance: {target.poc.volume_significance:.2%}")
    else:
        print("No naked POCs found within search range.")
    
    # Summary
    summary = detector.get_all_pocs_summary("BTCUSDT")
    print(f"\nPOC Status Summary: {summary}")
