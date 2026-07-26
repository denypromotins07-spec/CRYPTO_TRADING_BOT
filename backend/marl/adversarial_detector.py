#!/usr/bin/env python3
"""
Adversarial Detector for Identifying Predatory HFT Algorithms

This module identifies if the bot is being targeted by predatory HFT algos,
distinguishing between natural volatility and targeted spoofing/layering attacks.

Features:
- Spoofing pattern detection
- Layering attack identification
- Order book manipulation signals
- Front-running detection
- Natural vs artificial volatility classification
- Memory-efficient sliding window analysis

Integrates quantitative finance domains:
- Market microstructure analysis
- Behavioral finance (predatory behavior)
- Statistical anomaly detection
- Time series analysis
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from collections import deque
import numpy as np
from numpy.typing import NDArray


class ThreatLevel(Enum):
    """Severity levels for detected threats."""
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class AttackType(Enum):
    """Types of predatory attacks."""
    SPOOFING = auto()
    LAYERING = auto()
    FRONT_RUNNING = auto()
    MOMENTUM_IGNITION = auto()
    LIQUIDITY_DRAIN = auto()
    STOP_HUNTING = auto()


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state."""
    timestamp: int
    best_bid: float
    best_ask: float
    bid_sizes: List[float]
    ask_sizes: List[float]
    mid_price: float
    spread: float
    
    @classmethod
    def from_data(cls, timestamp: int, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> 'OrderBookSnapshot':
        """Create snapshot from bid/ask data."""
        if not bids or not asks:
            return cls(
                timestamp=timestamp,
                best_bid=0.0,
                best_ask=0.0,
                bid_sizes=[],
                ask_sizes=[],
                mid_price=0.0,
                spread=0.0
            )
        
        best_bid = max(b[0] for b in bids)
        best_ask = min(a[0] for a in asks)
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Sort and extract sizes
        sorted_bids = sorted(bids, key=lambda x: -x[0])[:10]
        sorted_asks = sorted(asks, key=lambda x: x[0])[:10]
        
        return cls(
            timestamp=timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_sizes=[b[1] for b in sorted_bids],
            ask_sizes=[a[1] for a in sorted_asks],
            mid_price=mid_price,
            spread=spread
        )


@dataclass
class DetectedThreat:
    """A detected predatory threat."""
    attack_type: AttackType
    threat_level: ThreatLevel
    confidence: float
    timestamp: int
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""


class AdversarialDetector:
    """
    Detector for predatory HFT algorithms and market manipulation.
    
    This class analyzes order book dynamics and trade patterns to identify
    when the bot is being targeted by sophisticated adversarial algorithms.
    
    Memory footprint: ~3MB for default configuration
    """
    
    def __init__(
        self,
        window_size: int = 100,
        spoofing_threshold: float = 0.7,
        layering_threshold: float = 0.6
    ):
        self.window_size = window_size
        self.spoofing_threshold = spoofing_threshold
        self.layering_threshold = layering_threshold
        
        # Order book history
        self.order_book_history: deque[OrderBookSnapshot] = deque(maxlen=window_size)
        
        # Cancellation tracking
        self.cancellation_history: deque[Tuple[int, str, float]] = deque(maxlen=window_size)
        
        # Trade execution history
        self.trade_history: deque[Dict[str, Any]] = deque(maxlen=window_size)
        
        # Price movement history
        self.price_history: deque[float] = deque(maxlen=window_size)
        
        # Detection state
        self.threat_counts: Dict[AttackType, int] = {t: 0 for t in AttackType}
        self.last_threat_time: Dict[AttackType, int] = {}
        
        # Baseline statistics
        self.baseline_spread: float = 0.0
        self.baseline_volatility: float = 0.0
        self.baseline_cancellation_rate: float = 0.0
        
        # Adaptive thresholds
        self.adaptive_thresholds: Dict[str, float] = {}
    
    def add_order_book_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add order book snapshot for analysis."""
        self.order_book_history.append(snapshot)
        self.price_history.append(snapshot.mid_price)
        
        # Update baseline after warm-up
        if len(self.order_book_history) >= self.window_size // 2:
            self._update_baselines()
    
    def add_cancellation(self, timestamp: int, side: str, size: float) -> None:
        """Record a cancellation event."""
        self.cancellation_history.append((timestamp, side, size))
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record a trade execution."""
        self.trade_history.append(trade_data)
    
    def detect_threats(self) -> List[DetectedThreat]:
        """
        Run all detection algorithms and return detected threats.
        
        Returns:
            List of DetectedThreat objects
        """
        threats = []
        
        if len(self.order_book_history) < 10:
            return threats
        
        # Run individual detectors
        spoof_threat = self._detect_spoofing()
        if spoof_threat:
            threats.append(spoof_threat)
        
        layer_threat = self._detect_layering()
        if layer_threat:
            threats.append(layer_threat)
        
        front_run_threat = self._detect_front_running()
        if front_run_threat:
            threats.append(front_run_threat)
        
        momentum_threat = self._detect_momentum_ignition()
        if momentum_threat:
            threats.append(momentum_threat)
        
        # Update threat counts
        for threat in threats:
            self.threat_counts[threat.attack_type] += 1
            self.last_threat_time[threat.attack_type] = threat.timestamp
        
        return threats
    
    def _detect_spoofing(self) -> Optional[DetectedThreat]:
        """Detect spoofing patterns (large orders quickly cancelled)."""
        if len(self.cancellation_history) < 5:
            return None
        
        recent_cancels = list(self.cancellation_history)[-20:]
        
        # Look for large size cancellations
        large_cancels = [c for c in recent_cancels if c[2] > np.mean([x[2] for x in recent_cancels]) * 2]
        
        if len(large_cancels) < 3:
            return None
        
        # Check timing (spoofing involves rapid cancel)
        timestamps = [c[0] for c in large_cancels]
        time_diffs = np.diff(timestamps)
        
        if len(time_diffs) > 0 and np.mean(time_diffs) < 100:  # < 100ms apart
            # Calculate imbalance created
            total_cancelled_size = sum(c[2] for c in large_cancels)
            
            confidence = min(1.0, len(large_cancels) / 10.0)
            
            if confidence >= self.spoofing_threshold:
                return DetectedThreat(
                    attack_type=AttackType.SPOOFING,
                    threat_level=self._classify_threat_level(confidence),
                    confidence=confidence,
                    timestamp=recent_cancels[-1][0],
                    evidence={
                        "large_cancels": len(large_cancels),
                        "total_size": total_cancelled_size,
                        "avg_time_between": float(np.mean(time_diffs))
                    },
                    recommended_action="Reduce order visibility, use iceberg orders"
                )
        
        return None
    
    def _detect_layering(self) -> Optional[DetectedThreat]:
        """Detect layering patterns (multiple orders at different price levels)."""
        if len(self.order_book_history) < 5:
            return None
        
        recent_books = list(self.order_book_history)[-10:]
        
        # Analyze order book depth patterns
        layering_scores = []
        
        for book in recent_books:
            # Check for unusual depth distribution
            if len(book.bid_sizes) >= 5 and len(book.ask_sizes) >= 5:
                # Layering often shows monotonic size patterns
                bid_monotonic = self._check_monotonic_pattern(book.bid_sizes)
                ask_monotonic = self._check_monotonic_pattern(book.ask_sizes)
                
                score = (bid_monotonic + ask_monotonic) / 2
                layering_scores.append(score)
        
        if not layering_scores:
            return None
        
        avg_score = np.mean(layering_scores)
        
        if avg_score >= self.layering_threshold:
            return DetectedThreat(
                attack_type=AttackType.LAYERING,
                threat_level=self._classify_threat_level(avg_score),
                confidence=avg_score,
                timestamp=recent_books[-1].timestamp,
                evidence={
                    "layering_score": avg_score,
                    "books_analyzed": len(layering_scores)
                },
                recommended_action="Widen spreads, reduce position size"
            )
        
        return None
    
    def _check_monotonic_pattern(self, sizes: List[float]) -> float:
        """Check if sizes follow monotonic pattern (indicator of layering)."""
        if len(sizes) < 3:
            return 0.0
        
        # Count monotonically increasing/decreasing sequences
        increasing = sum(1 for i in range(len(sizes)-1) if sizes[i] <= sizes[i+1])
        decreasing = sum(1 for i in range(len(sizes)-1) if sizes[i] >= sizes[i+1])
        
        max_monotonic = max(increasing, decreasing)
        return max_monotonic / (len(sizes) - 1)
    
    def _detect_front_running(self) -> Optional[DetectedThreat]:
        """Detect potential front-running of our orders."""
        if len(self.trade_history) < 5:
            return None
        
        recent_trades = list(self.trade_history)[-20:]
        
        # Look for trades consistently before our executions
        front_run_indicators = []
        
        for i, trade in enumerate(recent_trades):
            if trade.get('is_our_trade', False):
                # Check if there were trades just before ours at worse prices
                if i > 0:
                    prev_trade = recent_trades[i-1]
                    if not prev_trade.get('is_our_trade', False):
                        # Check price impact
                        price_diff = abs(trade.get('price', 0) - prev_trade.get('price', 0))
                        if price_diff > 0 and trade.get('side') == prev_trade.get('side'):
                            front_run_indicators.append(price_diff)
        
        if len(front_run_indicators) >= 3:
            avg_impact = np.mean(front_run_indicators)
            confidence = min(1.0, len(front_run_indicators) / 10.0)
            
            if confidence > 0.5:
                return DetectedThreat(
                    attack_type=AttackType.FRONT_RUNNING,
                    threat_level=self._classify_threat_level(confidence),
                    confidence=confidence,
                    timestamp=recent_trades[-1].get('timestamp', 0),
                    evidence={
                        "front_run_count": len(front_run_indicators),
                        "avg_price_impact": avg_impact
                    },
                    recommended_action="Use randomization in order timing, split orders"
                )
        
        return None
    
    def _detect_momentum_ignition(self) -> Optional[DetectedThreat]:
        """Detect momentum ignition (artificial price moves to trigger stops)."""
        if len(self.price_history) < 20:
            return None
        
        prices = np.array(list(self.price_history)[-30:])
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Look for sharp reversals (signature of momentum ignition)
        if len(returns) < 5:
            return None
        
        # Detect spikes followed by reversal
        spike_threshold = 3.0 * np.std(returns)
        spikes = np.abs(returns) > spike_threshold
        
        if np.sum(spikes) < 2:
            return None
        
        # Check for reversals after spikes
        spike_indices = np.where(spikes)[0]
        reversal_count = 0
        
        for idx in spike_indices:
            if idx + 2 < len(returns):
                # Check if next return reverses direction
                if returns[idx] * returns[idx + 1] < 0:
                    reversal_count += 1
        
        if reversal_count >= 2:
            confidence = min(1.0, reversal_count / 5.0)
            
            return DetectedThreat(
                attack_type=AttackType.MOMENTUM_IGNITION,
                threat_level=self._classify_threat_level(confidence),
                confidence=confidence,
                timestamp=len(self.price_history) - 1,
                evidence={
                    "spike_count": int(np.sum(spikes)),
                    "reversal_count": reversal_count,
                    "volatility_ratio": float(np.std(returns[-10:]) / (np.std(returns) + 1e-10))
                },
                recommended_action="Tighten stops, reduce leverage"
            )
        
        return None
    
    def _classify_threat_level(self, confidence: float) -> ThreatLevel:
        """Classify threat level based on confidence."""
        if confidence >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.7:
            return ThreatLevel.HIGH
        elif confidence >= 0.5:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.3:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE
    
    def _update_baselines(self) -> None:
        """Update baseline statistics from recent data."""
        if len(self.order_book_history) < 10:
            return
        
        books = list(self.order_book_history)
        
        self.baseline_spread = np.mean([b.spread for b in books])
        self.baseline_volatility = np.std([b.mid_price for b in books]) / np.mean([b.mid_price for b in books])
        
        if self.cancellation_history:
            cancels_per_window = len(self.cancellation_history) / self.window_size
            self.baseline_cancellation_rate = cancels_per_window
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of current threat landscape."""
        active_threats = self.detect_threats()
        
        highest_threat = ThreatLevel.NONE
        if active_threats:
            highest_threat = max(t.threat_level for t in active_threats)
        
        return {
            "active_threats": len(active_threats),
            "highest_threat_level": highest_threat.name,
            "threat_types": [t.attack_type.name for t in active_threats],
            "historical_counts": {k.name: v for k, v in self.threat_counts.items()},
            "baseline_spread": self.baseline_spread,
            "baseline_volatility": self.baseline_volatility
        }
    
    def is_natural_volatility(self) -> bool:
        """
        Determine if current volatility is natural or artificial.
        
        Returns:
            True if volatility appears natural, False if potentially manipulated
        """
        if len(self.price_history) < 20:
            return True
        
        prices = np.array(list(self.price_history))
        returns = np.diff(prices) / prices[:-1]
        
        # Natural volatility tends to be more clustered
        # Artificial volatility often shows sudden spikes
        
        recent_vol = np.std(returns[-10:])
        historical_vol = np.std(returns)
        
        vol_ratio = recent_vol / (historical_vol + 1e-10)
        
        # If volatility suddenly spiked without corresponding volume, likely artificial
        if vol_ratio > 3.0:
            # Check for corresponding order book changes
            if len(self.order_book_history) >= 10:
                recent_spreads = [b.spread for b in list(self.order_book_history)[-10:]]
                spread_change = np.std(recent_spreads) / (np.mean(recent_spreads) + 1e-10)
                
                if spread_change < 0.1:  # Spreads didn't widen proportionally
                    return False
        
        return True
    
    def reset(self) -> None:
        """Reset all detection state."""
        self.order_book_history.clear()
        self.cancellation_history.clear()
        self.trade_history.clear()
        self.price_history.clear()
        self.threat_counts = {t: 0 for t in AttackType}
        self.last_threat_time.clear()


def analyze_market_integrity(
    order_books: List[OrderBookSnapshot],
    cancellations: List[Tuple[int, str, float]]
) -> Dict[str, Any]:
    """
    Convenience function to analyze market integrity.
    
    Args:
        order_books: List of order book snapshots
        cancellations: List of (timestamp, side, size) tuples
        
    Returns:
        Dictionary with threat analysis results
    """
    detector = AdversarialDetector(window_size=50)
    
    for book in order_books:
        detector.add_order_book_snapshot(book)
    
    for cancel in cancellations:
        detector.add_cancellation(*cancel)
    
    threats = detector.detect_threats()
    summary = detector.get_threat_summary()
    summary['detected_threats'] = [
        {
            "type": t.attack_type.name,
            "level": t.threat_level.name,
            "confidence": t.confidence
        }
        for t in threats
    ]
    
    return summary


if __name__ == "__main__":
    # Example usage
    import random
    
    detector = AdversarialDetector()
    
    # Simulate normal market conditions
    base_price = 50000.0
    for i in range(50):
        price = base_price + random.gauss(0, 50)
        snapshot = OrderBookSnapshot(
            timestamp=i * 1000,
            best_bid=price - 1,
            best_ask=price + 1,
            bid_sizes=[random.uniform(1, 10) for _ in range(10)],
            ask_sizes=[random.uniform(1, 10) for _ in range(10)],
            mid_price=price,
            spread=2
        )
        detector.add_order_book_snapshot(snapshot)
    
    # Simulate suspicious cancellations (potential spoofing)
    for i in range(50, 60):
        detector.add_cancellation(i * 1000, "BID", random.uniform(50, 100))
    
    # Detect threats
    threats = detector.detect_threats()
    
    print("Threat Analysis Results:")
    print(f"Active threats: {len(threats)}")
    
    for threat in threats:
        print(f"\n  Type: {threat.attack_type.name}")
        print(f"  Level: {threat.threat_level.name}")
        print(f"  Confidence: {threat.confidence:.2f}")
        print(f"  Recommendation: {threat.recommended_action}")
    
    # Get summary
    summary = detector.get_threat_summary()
    print(f"\nOverall Assessment: {summary['highest_threat_level']}")
    print(f"Natural Volatility: {detector.is_natural_volatility()}")
