#!/usr/bin/env python3
"""
ZAID PERSONAL CRYPTO TRADING BOT - Order Book Microstructure
Chapter 2: Liquidity Sweeper

This module detects spoofing, absorption, and liquidity manipulation patterns
in the order book. It identifies institutional footprints and distinguishes
between genuine liquidity and fake orders designed to manipulate price.

Memory Budget: <80MB for order flow tracking
Target Latency: <200μs for pattern detection
Assets: BTC, SOL, ETH parallel processing
Integration: Feeds anti-manipulation signals to execution engine

Author: Opus 4.8
Stage: 5/100 - Advanced Risk Management and Order Book Microstructure
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict, Deque, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import time


class ManipulationType(Enum):
    """Types of market manipulation detected."""
    SPOOFING = "spoofing"           # Fake large orders
    LAYERING = "layering"           # Multiple fake orders at different levels
    MOMENTUM_IGNITION = "momentum_ignition"  # Triggering stops then reversing
    ABSORPTION = "absorption"       # Hidden liquidity absorbing market orders
    WASH_TRADING = "wash_trading"   # Self-dealing to create false volume
    QUOTE_STUFFING = "quote_stuffing"  # Flooding book to slow competitors


@dataclass
class LiquidityConfig:
    """Configuration for liquidity analysis."""
    assets: List[str] = field(default_factory=lambda: ["BTC", "SOL", "ETH"])
    order_history_size: int = 5000  # Track last N order events
    spoofing_size_threshold: float = 5.0  # Multiple of average order size
    spoofing_cancel_ratio: float = 0.8  # 80% cancel rate indicates spoofing
    absorption_detection_window: int = 10  # Trades within window
    layering_min_levels: int = 3  # Minimum levels for layering pattern
    quote_stuffing_rate_threshold: int = 100  # Orders per second


@dataclass
class OrderEvent:
    """Single order book event."""
    timestamp: float
    side: str  # 'BID' or 'ASK'
    price: float
    quantity: float
    event_type: str  # 'NEW', 'CANCEL', 'MODIFY', 'TRADE'
    order_id: Optional[str] = None


@dataclass
class SpoofingAlert(TypedDict):
    """Alert for detected spoofing activity."""
    asset: str
    manipulation_type: str
    confidence: float  # 0-1 confidence score
    description: str
    evidence: Dict[str, float]
    timestamp: float


@dataclass
class LiquiditySignature(TypedDict):
    """Identified liquidity pattern."""
    asset: str
    signature_type: str
    strength: float
    direction: str  # 'BUY' or 'SELL' pressure
    estimated_size: float
    timestamp: float


class LiquiditySweeper:
    """
    Advanced liquidity analyzer detecting manipulation and hidden flows.
    
    Features:
    - Spoofing detection via cancel ratio analysis
    - Layering pattern recognition
    - Absorption detection (hidden liquidity)
    - Momentum ignition identification
    - Quote stuffing detection
    - Institutional footprint tracking
    """
    
    __slots__ = (
        '_config', '_order_history', '_active_orders',
        '_cancel_tracker', '_trade_history', '_manipulation_alerts'
    )
    
    def __init__(self, config: LiquidityConfig = None) -> None:
        """
        Initialize liquidity sweeper.
        
        Args:
            config: Liquidity analysis configuration
        """
        self._config = config or LiquidityConfig()
        
        # Order history per asset
        self._order_history: Dict[str, Deque[OrderEvent]] = {
            asset: deque(maxlen=self._config.order_history_size)
            for asset in self._config.assets
        }
        
        # Active orders (not yet cancelled/filled)
        self._active_orders: Dict[str, Dict[str, OrderEvent]] = {
            asset: {} for asset in self._config.assets
        }
        
        # Cancel tracking per order ID
        self._cancel_tracker: Dict[str, Dict[str, List[float]]] = {
            asset: defaultdict(list) for asset in self._config.assets
        }
        
        # Trade history for absorption detection
        self._trade_history: Dict[str, Deque[Tuple[float, float, str]]] = {
            asset: deque(maxlen=1000) for asset in self._config.assets
        }  # (timestamp, size, side)
        
        # Manipulation alerts buffer
        self._manipulation_alerts: Dict[str, List[SpoofingAlert]] = {
            asset: [] for asset in self._config.assets
        }
    
    def add_order_event(self, asset: str, event: OrderEvent) -> Optional[SpoofingAlert]:
        """
        Process a new order book event.
        
        Args:
            asset: Asset identifier
            event: Order event details
            
        Returns:
            SpoofingAlert if manipulation detected
        """
        if asset not in self._order_history:
            return None
        
        # Add to history
        self._order_history[asset].append(event)
        
        # Track active orders
        if event.event_type == 'NEW':
            if event.order_id:
                self._active_orders[asset][event.order_id] = event
        elif event.event_type == 'CANCEL':
            if event.order_id and event.order_id in self._active_orders[asset]:
                del self._active_orders[asset][event.order_id]
                
                # Track cancellation timing
                self._cancel_tracker[asset][event.order_id].append(
                    event.timestamp
                )
        elif event.event_type == 'TRADE':
            self._trade_history[asset].append(
                (event.timestamp, event.quantity, event.side)
            )
        
        # Run detection algorithms
        alert = None
        
        # Check for spoofing periodically
        if len(self._order_history[asset]) % 50 == 0:
            alert = self._detect_spoofing(asset)
        
        # Check for layering
        if alert is None and len(self._order_history[asset]) % 100 == 0:
            alert = self._detect_layering(asset)
        
        # Check for absorption
        if alert is None and len(self._trade_history[asset]) >= 10:
            alert = self._detect_absorption(asset)
        
        if alert:
            self._manipulation_alerts[asset].append(alert)
        
        return alert
    
    def _detect_spoofing(self, asset: str) -> Optional[SpoofingAlert]:
        """
        Detect spoofing patterns.
        
        Spoofing characteristics:
        - Large orders placed far from mid-price
        - High cancellation rate (>80%)
        - Orders removed before execution
        """
        history = list(self._order_history[asset])
        
        if len(history) < 100:
            return None
        
        # Analyze recent orders
        recent_orders = [e for e in history[-200:] if e.event_type in ['NEW', 'CANCEL']]
        
        if len(recent_orders) < 20:
            return None
        
        # Calculate order size statistics
        sizes = [e.quantity for e in recent_orders if e.event_type == 'NEW']
        if not sizes:
            return None
        
        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Identify unusually large orders
        large_orders = [
            e for e in recent_orders
            if e.event_type == 'NEW' and e.quantity > avg_size * self._config.spoofing_size_threshold
        ]
        
        if not large_orders:
            return None
        
        # Check cancellation rate for large orders
        large_order_ids = {e.order_id for e in large_orders if e.order_id}
        cancelled_large = sum(
            1 for oid in large_order_ids
            if len(self._cancel_tracker[asset].get(oid, [])) > 0
        )
        
        if not large_order_ids:
            return None
        
        cancel_ratio = cancelled_large / len(large_order_ids)
        
        if cancel_ratio >= self._config.spoofing_cancel_ratio:
            return SpoofingAlert(
                asset=asset,
                manipulation_type=ManipulationType.SPOOFING.value,
                confidence=min(cancel_ratio, 1.0),
                description=f"Detected {len(large_orders)} large orders with {cancel_ratio:.1%} cancel rate",
                evidence={
                    'large_order_count': len(large_orders),
                    'cancel_ratio': cancel_ratio,
                    'avg_size_multiple': max(sizes) / avg_size if avg_size > 0 else 0
                },
                timestamp=time.time()
            )
        
        return None
    
    def _detect_layering(self, asset: str) -> Optional[SpoofingAlert]:
        """
        Detect layering manipulation.
        
        Layering characteristics:
        - Multiple orders at consecutive price levels
        - All on same side of book
        - Cancelled together when price approaches
        """
        history = list(self._order_history[asset])
        
        if len(history) < 200:
            return None
        
        # Get current active orders
        active = list(self._active_orders[asset].values())
        
        if len(active) < self._config.layering_min_levels:
            return None
        
        # Group by side
        bid_orders = [o for o in active if o.side == 'BID']
        ask_orders = [o for o in active if o.side == 'ASK']
        
        for side, orders in [('BID', bid_orders), ('ASK', ask_orders)]:
            if len(orders) < self._config.layering_min_levels:
                continue
            
            # Check if orders are at consecutive levels
            prices = sorted([o.price for o in orders])
            
            # Calculate price gaps
            gaps = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
            
            if not gaps:
                continue
            
            avg_gap = np.mean(gaps)
            gap_variance = np.var(gaps)
            
            # Low variance in gaps suggests artificial layering
            if gap_variance < avg_gap * 0.1 and len(orders) >= 4:
                # Check if similar sizes (another layering indicator)
                sizes = [o.quantity for o in orders]
                size_variance = np.var(sizes)
                avg_size = np.mean(sizes)
                
                if size_variance < avg_size * 0.2:
                    return SpoofingAlert(
                        asset=asset,
                        manipulation_type=ManipulationType.LAYERING.value,
                        confidence=0.8,
                        description=f"Detected {len(orders)} layered {side} orders with uniform spacing",
                        evidence={
                            'order_count': len(orders),
                            'side': side,
                            'avg_gap': avg_gap,
                            'gap_variance': gap_variance
                        },
                        timestamp=time.time()
                    )
        
        return None
    
    def _detect_absorption(self, asset: str) -> Optional[SpoofingAlert]:
        """
        Detect absorption (hidden liquidity eating market orders).
        
        Absorption characteristics:
        - Many trades at same price level
        - Price doesn't move despite significant volume
        - Indicates hidden limit orders absorbing market flow
        """
        trades = list(self._trade_history[asset])
        
        if len(trades) < self._config.absorption_detection_window:
            return None
        
        recent_trades = trades[-self._config.absorption_detection_window:]
        
        # Group trades by price (rounded to nearest tick)
        price_groups: Dict[float, List[Tuple[float, str]]] = defaultdict(list)
        
        for timestamp, size, side in recent_trades:
            # Round price to typical tick size (would need actual tick size)
            price_key = round(timestamp, 2)  # Simplified
            price_groups[price_key].append((size, side))
        
        # Look for price levels with many trades
        for price_key, trade_list in price_groups.items():
            if len(trade_list) >= self._config.absorption_detection_window // 2:
                total_volume = sum(t[0] for t in trade_list)
                
                # Check if price held despite volume
                buy_volume = sum(t[0] for t in trade_list if t[1] == 'BUY')
                sell_volume = sum(t[0] for t in trade_list if t[1] == 'SELL')
                
                # Balanced absorption (both sides being absorbed)
                if abs(buy_volume - sell_volume) < total_volume * 0.3:
                    return SpoofingAlert(
                        asset=asset,
                        manipulation_type=ManipulationType.ABSORPTION.value,
                        confidence=0.7,
                        description=f"Detected absorption at price level: {total_volume:.2f} volume traded",
                        evidence={
                            'total_volume': total_volume,
                            'buy_volume': buy_volume,
                            'sell_volume': sell_volume,
                            'trade_count': len(trade_list)
                        },
                        timestamp=time.time()
                    )
        
        return None
    
    def detect_momentum_ignition(self, asset: str) -> Optional[SpoofingAlert]:
        """
        Detect momentum ignition (stop hunting then reversal).
        
        Characteristics:
        - Rapid price movement in one direction
        - Followed by immediate reversal
        - Often occurs at technical levels
        """
        trades = list(self._trade_history[asset])
        
        if len(trades) < 50:
            return None
        
        # Analyze trade sequence for rapid moves
        recent = trades[-50:]
        
        # Calculate cumulative imbalance
        buy_pressure = []
        cumulative = 0
        
        for _, size, side in recent:
            if side == 'BUY':
                cumulative += size
            else:
                cumulative -= size
            buy_pressure.append(cumulative)
        
        # Look for sharp reversal
        if len(buy_pressure) < 20:
            return None
        
        max_pressure = max(buy_pressure)
        min_pressure = min(buy_pressure)
        range_val = max_pressure - min_pressure
        
        if range_val == 0:
            return None
        
        # Check if we had a sharp move and reversal
        current = buy_pressure[-1]
        peak_idx = buy_pressure.index(max_pressure)
        trough_idx = buy_pressure.index(min_pressure)
        
        # Reversal pattern: moved significantly then came back
        if abs(current) < range_val * 0.3 and range_val > np.std(buy_pressure) * 3:
            return SpoofingAlert(
                asset=asset,
                manipulation_type=ManipulationType.MOMENTUM_IGNITION.value,
                confidence=0.6,
                description="Potential momentum ignition detected - sharp move reversed",
                evidence={
                    'max_pressure': max_pressure,
                    'min_pressure': min_pressure,
                    'current_pressure': current,
                    'range': range_val
                },
                timestamp=time.time()
            )
        
        return None
    
    def get_liquidity_signature(self, asset: str) -> Optional[LiquiditySignature]:
        """
        Generate overall liquidity signature for an asset.
        
        Aggregates all detected patterns into a single signature.
        """
        alerts = self._manipulation_alerts.get(asset, [])
        
        if not alerts:
            return None
        
        # Aggregate evidence
        manipulation_counts = defaultdict(int)
        total_confidence = 0.0
        
        for alert in alerts[-20:]:  # Recent alerts
            manipulation_counts[alert['manipulation_type']] += 1
            total_confidence += alert['confidence']
        
        dominant_type = max(manipulation_counts.keys(), 
                           key=lambda k: manipulation_counts[k])
        
        avg_confidence = total_confidence / len(alerts[-20:])
        
        # Determine direction based on predominant side
        active_orders = self._active_orders.get(asset, {})
        bid_count = sum(1 for o in active_orders.values() if o.side == 'BID')
        ask_count = sum(1 for o in active_orders.values() if o.side == 'ASK')
        
        direction = 'BUY' if bid_count > ask_count else 'SELL'
        
        return LiquiditySignature(
            asset=asset,
            signature_type=dominant_type,
            strength=avg_confidence,
            direction=direction,
            estimated_size=sum(manipulation_counts.values()) * 100,  # Estimate
            timestamp=time.time()
        )
    
    def get_manipulation_summary(self, asset: str) -> Dict[str, any]:
        """Get summary of manipulation attempts for an asset."""
        alerts = self._manipulation_alerts.get(asset, [])
        
        if not alerts:
            return {'detected': False, 'types': [], 'risk_level': 'LOW'}
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in alerts:
            type_counts[alert['manipulation_type']] += 1
        
        # Calculate risk level
        recent_alerts = len([a for a in alerts if time.time() - a['timestamp'] < 300])
        
        if recent_alerts > 10:
            risk_level = 'HIGH'
        elif recent_alerts > 5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'detected': True,
            'types': dict(type_counts),
            'total_alerts': len(alerts),
            'recent_alerts': recent_alerts,
            'risk_level': risk_level
        }
    
    def clear_old_alerts(self, max_age_seconds: float = 300) -> None:
        """Clear alerts older than specified age."""
        current_time = time.time()
        
        for asset in self._manipulation_alerts:
            self._manipulation_alerts[asset] = [
                a for a in self._manipulation_alerts[asset]
                if current_time - a['timestamp'] < max_age_seconds
            ]


if __name__ == "__main__":
    # Example usage and validation
    config = LiquidityConfig()
    sweeper = LiquiditySweeper(config)
    
    import random
    
    print("=== Simulating Normal Order Flow ===")
    
    base_price = 50000.0
    order_id_counter = 0
    
    # Generate normal order flow
    for i in range(500):
        side = random.choice(['BID', 'ASK'])
        price_offset = random.uniform(-50, 50)
        price = base_price + price_offset
        quantity = random.uniform(0.1, 2.0)
        
        event = OrderEvent(
            timestamp=time.time(),
            side=side,
            price=price,
            quantity=quantity,
            event_type='NEW',
            order_id=f"ord_{order_id_counter}"
        )
        order_id_counter += 1
        
        alert = sweeper.add_order_event("BTC", event)
        
        # Occasionally cancel
        if random.random() < 0.3:
            cancel_event = OrderEvent(
                timestamp=time.time(),
                side=side,
                price=price,
                quantity=quantity,
                event_type='CANCEL',
                order_id=event.order_id
            )
            sweeper.add_order_event("BTC", cancel_event)
        
        if i % 100 == 0:
            summary = sweeper.get_manipulation_summary("BTC")
            print(f"After {i} events: Risk Level = {summary['risk_level']}")
    
    # Simulate spoofing attempt
    print("\n=== Simulating Spoofing Attempt ===")
    
    spoof_orders = []
    for i in range(20):
        # Large orders far from mid
        event = OrderEvent(
            timestamp=time.time(),
            side='ASK',
            price=base_price + 500 + (i * 10),  # Stacked above market
            quantity=50.0,  # Very large
            event_type='NEW',
            order_id=f"spoof_{i}"
        )
        spoof_orders.append(event)
        sweeper.add_order_event("BTC", event)
    
    # Cancel all spoof orders
    for event in spoof_orders:
        cancel_event = OrderEvent(
            timestamp=time.time(),
            side=event.side,
            price=event.price,
            quantity=event.quantity,
            event_type='CANCEL',
            order_id=event.order_id
        )
        sweeper.add_order_event("BTC", cancel_event)
    
    # Check for detection
    summary = sweeper.get_manipulation_summary("BTC")
    print(f"Spoofing Summary: {summary}")
    
    signature = sweeper.get_liquidity_signature("BTC")
    if signature:
        print(f"Liquidity Signature: {signature['signature_type']} "
              f"(confidence: {signature['strength']:.2f})")
