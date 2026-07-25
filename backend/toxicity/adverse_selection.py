"""
Adverse Selection Detector
Detects informed traders and dynamically adjusts quote sizes.
Optimized for real-time detection of whale activity and toxic flow.

This module implements:
- Trade size anomaly detection
- Price impact analysis
- Informed trader identification
- Dynamic quote adjustment based on toxicity

Target: Protect market maker from adverse selection during news/events.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum
import numpy as np
from datetime import datetime, timezone


class TraderType(Enum):
    """Classification of trader type based on behavior."""
    RETAIL = "retail"  # Small, random trades
    INSTITUTIONAL = "institutional"  # Large, systematic trades
    INFORMED = "informed"  # Toxic, predictive trades
    HFT = "hft"  # High-frequency, low latency
    UNKNOWN = "unknown"


@dataclass
class TradeSignature:
    """Signature characteristics of a trade."""
    timestamp_ms: int
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    aggressor_side: str  # Who initiated the trade
    price_impact_bps: float  # Price movement after trade (basis points)
    subsequent_direction: str  # Direction price moved after trade


@dataclass
class AdverseSelectionConfig:
    """Configuration for adverse selection detection."""
    # Volume threshold for large trades (base currency units)
    large_trade_threshold: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 1.0,
        "ETH": 10.0,
        "SOL": 100.0,
    })
    
    # Price impact threshold for informed detection (basis points)
    price_impact_threshold_bps: float = 5.0
    
    # Lookback window for pattern analysis (milliseconds)
    lookback_window_ms: int = 60000  # 1 minute
    
    # Minimum consecutive informed trades for detection
    min_consecutive_informed: int = 3
    
    # Quote reduction factor when toxicity detected (0.0 to 1.0)
    quote_reduction_factor: float = 0.5
    
    # Spread widening multiplier for toxic flow
    toxic_spread_multiplier: float = 2.0


@dataclass
class TraderProfile:
    """Profile of a detected trader based on their activity."""
    trader_id: str
    trader_type: TraderType
    first_seen_ms: int
    last_seen_ms: int
    trade_count: int
    total_volume: float
    avg_price_impact_bps: float
    win_rate: float  # Percentage of trades followed by favorable price move
    confidence: float  # Confidence in classification (0.0 to 1.0)


class AdverseSelectionDetector:
    """
    Detects adverse selection and informed trading patterns.
    
    Monitors trade flow to identify when the market maker is being
    picked off by traders with superior information.
    """
    
    def __init__(self, config: Optional[AdverseSelectionConfig] = None):
        self.config = config or AdverseSelectionConfig()
        
        # Recent trade history per symbol
        self.trade_history: Dict[str, Deque[TradeSignature]] = {}
        
        # Detected trader profiles
        self.trader_profiles: Dict[str, TraderProfile] = {}
        
        # Current toxicity level per symbol (0.0 to 1.0)
        self.toxicity_levels: Dict[str, float] = {}
        
        # Consecutive informed trade counter
        self.consecutive_informed: Dict[str, int] = {}
        
        # Quote adjustment factors
        self.quote_adjustments: Dict[str, float] = {}
        
        # Alert history
        self.alerts: List[Dict] = []
    
    def record_trade(
        self,
        symbol: str,
        trade: TradeSignature
    ) -> Optional[TraderType]:
        """
        Record a trade and check for adverse selection signals.
        
        Returns detected trader type if anomalous behavior detected.
        """
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=1000)
            self.toxicity_levels[symbol] = 0.0
            self.consecutive_informed[symbol] = 0
            self.quote_adjustments[symbol] = 1.0
        
        self.trade_history[symbol].append(trade)
        
        # Analyze trade for adverse selection
        trader_type = self._analyze_trade(symbol, trade)
        
        # Update toxicity level
        self._update_toxicity(symbol, trader_type)
        
        # Adjust quotes if needed
        self._adjust_quotes(symbol)
        
        return trader_type
    
    def _analyze_trade(self, symbol: str, trade: TradeSignature) -> TraderType:
        """Analyze individual trade for signs of informed trading."""
        threshold = self.config.large_trade_threshold.get(symbol, 1.0)
        
        # Check for large trade
        is_large = trade.volume >= threshold
        
        # Check for significant price impact
        is_impactful = abs(trade.price_impact_bps) >= self.config.price_impact_threshold_bps
        
        # Check if price moved against us (adverse selection signal)
        is_adverse = False
        if trade.aggressor_side == 'buy' and trade.subsequent_direction == 'up':
            # Buyer was correct, we sold too cheap
            is_adverse = True
        elif trade.aggressor_side == 'sell' and trade.subsequent_direction == 'down':
            # Seller was correct, we bought too high
            is_adverse = True
        
        # Classify trader type
        if is_large and is_impactful and is_adverse:
            return TraderType.INFORMED
        elif is_large and is_impactful:
            return TraderType.INSTITUTIONAL
        elif trade.volume < threshold * 0.1:
            return TraderType.RETAIL
        else:
            return TraderType.UNKNOWN
    
    def _update_toxicity(self, symbol: str, trader_type: TraderType) -> None:
        """Update toxicity level based on detected trader type."""
        current_toxicity = self.toxicity_levels.get(symbol, 0.0)
        
        if trader_type == TraderType.INFORMED:
            # Increase toxicity
            self.consecutive_informed[symbol] = self.consecutive_informed.get(symbol, 0) + 1
            
            # Exponential increase for consecutive informed trades
            increment = 0.2 * (1 + self.consecutive_informed[symbol] * 0.5)
            new_toxicity = min(1.0, current_toxicity + increment)
            
            # Generate alert if threshold crossed
            if new_toxicity > 0.7 and current_toxicity <= 0.7:
                self._generate_alert(symbol, "HIGH_TOXICITY", new_toxicity)
                
        else:
            # Decay toxicity over time
            self.consecutive_informed[symbol] = max(0, self.consecutive_informed.get(symbol, 0) - 1)
            new_toxicity = max(0.0, current_toxicity * 0.9)
        
        self.toxicity_levels[symbol] = new_toxicity
    
    def _adjust_quotes(self, symbol: str) -> None:
        """Adjust quote sizes and spreads based on toxicity."""
        toxicity = self.toxicity_levels.get(symbol, 0.0)
        
        if toxicity > 0.7:
            # High toxicity: significantly reduce exposure
            self.quote_adjustments[symbol] = self.config.quote_reduction_factor
        elif toxicity > 0.4:
            # Medium toxicity: moderate reduction
            self.quote_adjustments[symbol] = 1.0 - (toxicity * 0.5)
        else:
            # Low toxicity: normal quoting
            self.quote_adjustments[symbol] = 1.0
    
    def get_quote_adjustment(self, symbol: str) -> float:
        """Get current quote size adjustment factor for symbol."""
        return self.quote_adjustments.get(symbol, 1.0)
    
    def get_spread_multiplier(self, symbol: str) -> float:
        """Get spread widening multiplier based on toxicity."""
        toxicity = self.toxicity_levels.get(symbol, 0.0)
        
        # Linear scaling from 1.0x to toxic_multiplier
        return 1.0 + toxicity * (self.config.toxic_spread_multiplier - 1.0)
    
    def detect_informed_flow(self, symbol: str) -> bool:
        """Check if there's currently informed flow in the symbol."""
        return self.consecutive_informed.get(symbol, 0) >= self.config.min_consecutive_informed
    
    def _generate_alert(self, symbol: str, alert_type: str, severity: float) -> None:
        """Generate an adverse selection alert."""
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "alert_type": alert_type,
            "severity": severity,
            "message": f"Adverse selection detected: {alert_type} in {symbol}",
        }
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts.pop(0)
    
    def get_trader_profile(self, trader_id: str) -> Optional[TraderProfile]:
        """Get profile for a specific trader."""
        return self.trader_profiles.get(trader_id)
    
    def analyze_pattern(self, symbol: str, window_ms: int = None) -> Dict:
        """
        Analyze recent trading patterns for adverse selection signals.
        
        Returns comprehensive analysis of recent flow.
        """
        if window_ms is None:
            window_ms = self.config.lookback_window_ms
        
        if symbol not in self.trade_history:
            return {"error": "No data available"}
        
        current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        cutoff = current_time - window_ms
        
        recent_trades = [t for t in self.trade_history[symbol] if t.timestamp_ms >= cutoff]
        
        if not recent_trades:
            return {"status": "no_recent_trades"}
        
        # Calculate statistics
        buy_volume = sum(t.volume for t in recent_trades if t.side == 'buy')
        sell_volume = sum(t.volume for t in recent_trades if t.side == 'sell')
        
        adverse_trades = [t for t in recent_trades if 
                         (t.aggressor_side == 'buy' and t.subsequent_direction == 'up') or
                         (t.aggressor_side == 'sell' and t.subsequent_direction == 'down')]
        
        adverse_ratio = len(adverse_trades) / len(recent_trades) if recent_trades else 0
        
        large_trades = [t for t in recent_trades if 
                       t.volume >= self.config.large_trade_threshold.get(symbol, 1.0)]
        
        avg_impact = np.mean([abs(t.price_impact_bps) for t in recent_trades]) if recent_trades else 0
        
        return {
            "symbol": symbol,
            "window_ms": window_ms,
            "trade_count": len(recent_trades),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "volume_imbalance": (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0,
            "adverse_selection_ratio": adverse_ratio,
            "large_trade_count": len(large_trades),
            "avg_price_impact_bps": avg_impact,
            "current_toxicity": self.toxicity_levels.get(symbol, 0.0),
            "quote_adjustment": self.quote_adjustments.get(symbol, 1.0),
            "is_informed_flow": self.detect_informed_flow(symbol),
        }
    
    def get_statistics(self) -> Dict:
        """Get overall adverse selection statistics."""
        return {
            "symbols_monitored": list(self.toxicity_levels.keys()),
            "toxicity_levels": dict(self.toxicity_levels),
            "quote_adjustments": dict(self.quote_adjustments),
            "active_alerts": len([a for a in self.alerts]),
            "total_alerts": len(self.alerts),
            "detected_informed_flow": {
                s: self.detect_informed_flow(s) for s in self.toxicity_levels.keys()
            },
        }


# Example usage and testing
if __name__ == "__main__":
    import time
    
    detector = AdverseSelectionDetector()
    
    base_ts = int(time.time() * 1000)
    
    # Simulate normal trades
    for i in range(5):
        trade = TradeSignature(
            timestamp_ms=base_ts + i * 1000,
            price=50000.0 + i * 10,
            volume=0.1,
            side='buy',
            aggressor_side='buy',
            price_impact_bps=1.0,
            subsequent_direction='down'  # Price went against aggressor (good for MM)
        )
        result = detector.record_trade("BTC", trade)
        print(f"Trade {i}: Type = {result}")
    
    print(f"\nAfter normal trades:")
    print(f"  Toxicity: {detector.toxicity_levels.get('BTC', 0.0):.2f}")
    print(f"  Quote Adjustment: {detector.get_quote_adjustment('BTC'):.2f}")
    
    # Simulate informed trades
    print("\n--- Simulating informed flow ---")
    for i in range(5):
        trade = TradeSignature(
            timestamp_ms=base_ts + 5000 + i * 500,
            price=50000.0 + 50 + i * 20,
            volume=2.0,  # Large trade
            side='buy',
            aggressor_side='buy',
            price_impact_bps=8.0,  # Significant impact
            subsequent_direction='up'  # Price went with aggressor (bad for MM)
        )
        result = detector.record_trade("BTC", trade)
        print(f"Informed Trade {i}: Type = {result.value}")
    
    print(f"\nAfter informed trades:")
    print(f"  Toxicity: {detector.toxicity_levels.get('BTC', 0.0):.2f}")
    print(f"  Quote Adjustment: {detector.get_quote_adjustment('BTC'):.2f}")
    print(f"  Spread Multiplier: {detector.get_spread_multiplier('BTC'):.2f}")
    print(f"  Informed Flow Detected: {detector.detect_informed_flow('BTC')}")
    
    # Pattern analysis
    analysis = detector.analyze_pattern("BTC")
    print(f"\nPattern Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
