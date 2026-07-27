#!/usr/bin/env python3
"""
backend/sor/fee_optimizer.py

ZAID PERSONAL CRYPTO TRADING BOT - Stage 33: Multi-Venue SOR
Chapter 2: Fee Optimization and Order Routing

Routes orders to minimize maker/taker fees and withdrawal costs.
Accounts for Binance VIP tiers and potential BNB fee deductions.
Strictly prevents routing to venues where maker fees destroy theoretical alpha.
Strictly respects 8GB RAM limit on AMD Ryzen AI 5 laptop.

Features:
- Exchange-specific fee tier calculations (VIP levels)
- Maker vs taker fee comparison
- BNB fee discount optimization (Binance)
- Withdrawal cost estimation for cross-exchange arb
- Net profitability calculation after all fees
- Real-time fee-adjusted routing decisions

Type hints enforced for memory safety and IDE support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class Venue(Enum):
    """Supported trading venues."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    OKX = "okx"


@dataclass
class FeeTier:
    """Fee structure for a specific venue and VIP tier."""
    venue: Venue
    vip_level: int
    maker_fee_bps: float  # Basis points (1 bp = 0.01%)
    taker_fee_bps: float
    min_volume_30d: float  # Minimum 30-day volume in USD for this tier
    
    def maker_fee_pct(self) -> float:
        """Convert maker fee to percentage."""
        return self.maker_fee_bps / 100.0
    
    def taker_fee_pct(self) -> float:
        """Convert taker fee to percentage."""
        return self.taker_fee_bps / 100.0


@dataclass
class RoutingDecision:
    """Result of fee-optimized routing calculation."""
    venue: Venue
    order_type: str  # 'maker' or 'taker'
    gross_profit_bps: float
    total_fees_bps: float
    net_profit_bps: float
    is_profitable: bool
    recommended_size: float
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    
    @property
    def fee_ratio(self) -> float:
        """Ratio of fees to gross profit."""
        if self.gross_profit_bps <= 0:
            return float('inf')
        return self.total_fees_bps / self.gross_profit_bps


class FeeOptimizer:
    """
    Optimizes order routing by minimizing fees across venues.
    
    Accounts for:
    - VIP tier discounts
    - BNB fee deductions (Binance)
    - Maker vs taker fee differentials
    - Cross-exchange withdrawal costs
    """
    
    def __init__(self):
        """Initialize fee optimizer with default fee tiers."""
        self.fee_tiers: Dict[Venue, List[FeeTier]] = self._init_fee_tiers()
        self.current_vip_levels: Dict[Venue, int] = {
            Venue.BINANCE: 0,
            Venue.COINBASE: 0,
            Venue.KRAKEN: 0,
            Venue.BYBIT: 0,
            Venue.OKX: 0,
        }
        self.bnb_fee_discount: float = 0.25  # 25% discount with BNB on Binance
        self.use_bnb_discount: bool = True
        self.withdrawal_costs: Dict[str, float] = {}  # symbol -> cost in quote currency
        
    def _init_fee_tiers(self) -> Dict[Venue, List[FeeTier]]:
        """Initialize fee tier structures for all venues."""
        return {
            Venue.BINANCE: [
                FeeTier(Venue.BINANCE, 0, 10.0, 10.0, 0),      # Regular
                FeeTier(Venue.BINANCE, 1, 9.0, 10.0, 50_000),   # VIP 1
                FeeTier(Venue.BINANCE, 2, 8.0, 9.0, 500_000),   # VIP 2
                FeeTier(Venue.BINANCE, 3, 6.0, 8.0, 1_000_000), # VIP 3
                FeeTier(Venue.BINANCE, 4, 4.0, 6.0, 5_000_000), # VIP 4
                FeeTier(Venue.BINANCE, 5, 2.0, 4.0, 10_000_000),# VIP 5
            ],
            Venue.COINBASE: [
                FeeTier(Venue.COINBASE, 0, 40.0, 60.0, 0),
                FeeTier(Venue.COINBASE, 1, 35.0, 50.0, 100_000),
                FeeTier(Venue.COINBASE, 2, 25.0, 40.0, 1_000_000),
                FeeTier(Venue.COINBASE, 3, 15.0, 25.0, 10_000_000),
            ],
            Venue.KRAKEN: [
                FeeTier(Venue.KRAKEN, 0, 16.0, 26.0, 0),
                FeeTier(Venue.KRAKEN, 1, 14.0, 24.0, 50_000),
                FeeTier(Venue.KRAKEN, 2, 12.0, 22.0, 100_000),
                FeeTier(Venue.KRAKEN, 3, 10.0, 20.0, 500_000),
            ],
            Venue.BYBIT: [
                FeeTier(Venue.BYBIT, 0, 10.0, 10.0, 0),
                FeeTier(Venue.BYBIT, 1, 8.0, 9.0, 100_000),
                FeeTier(Venue.BYBIT, 2, 6.0, 8.0, 500_000),
            ],
            Venue.OKX: [
                FeeTier(Venue.OKX, 0, 8.0, 10.0, 0),
                FeeTier(Venue.OKX, 1, 7.0, 9.0, 100_000),
                FeeTier(Venue.OKX, 2, 6.0, 8.0, 500_000),
            ],
        }
    
    def set_vip_level(self, venue: Venue, level: int) -> None:
        """Set current VIP level for a venue."""
        if venue in self.fee_tiers:
            max_level = max(t.vip_level for t in self.fee_tiers[venue])
            self.current_vip_levels[venue] = min(level, max_level)
    
    def get_current_fee_tier(self, venue: Venue) -> FeeTier:
        """Get current fee tier for a venue based on VIP level."""
        if venue not in self.fee_tiers:
            raise ValueError(f"Unknown venue: {venue}")
        
        current_level = self.current_vip_levels.get(venue, 0)
        tiers = self.fee_tiers[venue]
        
        # Find matching tier
        for tier in sorted(tiers, key=lambda t: t.vip_level, reverse=True):
            if current_level >= tier.vip_level:
                return tier
        
        return tiers[0]  # Default to lowest tier
    
    def apply_bnb_discount(self, maker_fee_bps: float, taker_fee_bps: float) -> Tuple[float, float]:
        """Apply BNB fee discount if enabled (Binance only)."""
        if not self.use_bnb_discount:
            return maker_fee_bps, taker_fee_bps
        
        discount_factor = 1.0 - self.bnb_fee_discount
        return maker_fee_bps * discount_factor, taker_fee_bps * discount_factor
    
    def calculate_net_profit(
        self,
        venue: Venue,
        gross_profit_bps: float,
        is_maker: bool = True,
        size_usd: float = 1000.0,
    ) -> RoutingDecision:
        """
        Calculate net profit after fees for a potential trade.
        
        Args:
            venue: Target venue for execution
            gross_profit_bps: Expected profit before fees (in basis points)
            is_maker: Whether order will be maker (limit) or taker (market)
            size_usd: Order size in USD
            
        Returns:
            RoutingDecision with profitability analysis
        """
        tier = self.get_current_fee_tier(venue)
        
        # Apply BNB discount for Binance
        if venue == Venue.BINANCE:
            maker_fee, taker_fee = self.apply_bnb_discount(tier.maker_fee_bps, tier.taker_fee_bps)
        else:
            maker_fee, taker_fee = tier.maker_fee_bps, tier.taker_fee_bps
        
        # Select appropriate fee
        fee_bps = maker_fee if is_maker else taker_fee
        
        # Calculate net profit
        net_profit_bps = gross_profit_bps - fee_bps
        is_profitable = net_profit_bps > 0
        
        # Recommend size based on profitability
        if is_profitable:
            recommended_size = size_usd
        else:
            # Reduce size or skip
            recommended_size = 0.0
        
        return RoutingDecision(
            venue=venue,
            order_type='maker' if is_maker else 'taker',
            gross_profit_bps=gross_profit_bps,
            total_fees_bps=fee_bps,
            net_profit_bps=net_profit_bps,
            is_profitable=is_profitable,
            recommended_size=recommended_size,
        )
    
    def should_route_to_venue(
        self,
        venue: Venue,
        gross_profit_bps: float,
        min_net_profit_bps: float = 1.0,
    ) -> bool:
        """
        Determine if order should be routed to a venue.
        
        Strictly prevents routing if maker fees destroy theoretical alpha.
        
        Args:
            venue: Target venue
            gross_profit_bps: Expected gross profit in basis points
            min_net_profit_bps: Minimum acceptable net profit
            
        Returns:
            True if routing is recommended
        """
        decision = self.calculate_net_profit(venue, gross_profit_bps, is_maker=True)
        return decision.net_profit_bps >= min_net_profit_bps
    
    def find_best_venue(
        self,
        gross_profit_bps: float,
        preferred_maker: bool = True,
    ) -> Optional[RoutingDecision]:
        """
        Find the best venue for execution based on fee optimization.
        
        Args:
            gross_profit_bps: Expected gross profit in basis points
            preferred_maker: Prefer maker orders over taker
            
        Returns:
            Best RoutingDecision or None if no profitable venue exists
        """
        decisions: List[RoutingDecision] = []
        
        for venue in Venue:
            # Try maker first if preferred
            if preferred_maker:
                decision = self.calculate_net_profit(venue, gross_profit_bps, is_maker=True)
                if decision.is_profitable:
                    decisions.append(decision)
                    continue
            
            # Fall back to taker
            decision = self.calculate_net_profit(venue, gross_profit_bps, is_maker=False)
            if decision.is_profitable:
                decisions.append(decision)
        
        if not decisions:
            return None
        
        # Return best net profit
        return max(decisions, key=lambda d: d.net_profit_bps)
    
    def set_withdrawal_cost(self, symbol: str, cost_usd: float) -> None:
        """Set withdrawal cost for a symbol (for cross-exchange arb)."""
        self.withdrawal_costs[symbol.upper()] = cost_usd
    
    def calculate_cross_exchange_arb_profit(
        self,
        buy_venue: Venue,
        sell_venue: Venue,
        gross_spread_bps: float,
        symbol: str,
        size_usd: float,
    ) -> float:
        """
        Calculate net profit for cross-exchange arbitrage.
        
        Includes:
        - Trading fees on both venues
        - Withdrawal costs
        
        Args:
            buy_venue: Venue to buy on
            sell_venue: Venue to sell on
            gross_spread_bps: Price spread in basis points
            symbol: Trading pair symbol
            size_usd: Trade size in USD
            
        Returns:
            Net profit in basis points (negative if unprofitable)
        """
        # Get fees for both legs
        buy_decision = self.calculate_net_profit(buy_venue, 0, is_maker=False)
        sell_decision = self.calculate_net_profit(sell_venue, 0, is_maker=False)
        
        total_fees_bps = buy_decision.total_fees_bps + sell_decision.total_fees_bps
        
        # Subtract withdrawal cost
        withdrawal_cost_bps = 0.0
        if symbol.upper() in self.withdrawal_costs:
            withdrawal_cost_usd = self.withdrawal_costs[symbol.upper()]
            withdrawal_cost_bps = (withdrawal_cost_usd / size_usd) * 10_000
        
        net_profit_bps = gross_spread_bps - total_fees_bps - withdrawal_cost_bps
        
        return net_profit_bps
    
    def update_fee_tier(
        self,
        venue: Venue,
        vip_level: int,
        maker_fee_bps: float,
        taker_fee_bps: float,
    ) -> None:
        """Update or add a fee tier for a venue."""
        if venue not in self.fee_tiers:
            self.fee_tiers[venue] = []
        
        # Check if tier exists
        tiers = self.fee_tiers[venue]
        for i, tier in enumerate(tiers):
            if tier.vip_level == vip_level:
                tiers[i] = FeeTier(venue, vip_level, maker_fee_bps, taker_fee_bps, tier.min_volume_30d)
                return
        
        # Add new tier
        tiers.append(FeeTier(venue, vip_level, maker_fee_bps, taker_fee_bps, 0))


# Example usage and testing
if __name__ == "__main__":
    optimizer = FeeOptimizer()
    
    # Set VIP levels
    optimizer.set_vip_level(Venue.BINANCE, 1)  # VIP 1 with BNB discount
    optimizer.set_vip_level(Venue.COINBASE, 0)  # Regular
    
    # Test routing decision
    gross_profit = 15.0  # 15 bps gross profit
    
    binance_decision = optimizer.calculate_net_profit(
        Venue.BINANCE, gross_profit, is_maker=True, size_usd=10000
    )
    coinbase_decision = optimizer.calculate_net_profit(
        Venue.COINBASE, gross_profit, is_maker=True, size_usd=10000
    )
    
    print(f"Binance: net profit = {binance_decision.net_profit_bps:.2f} bps")
    print(f"Coinbase: net profit = {coinbase_decision.net_profit_bps:.2f} bps")
    
    # Find best venue
    best = optimizer.find_best_venue(gross_profit)
    if best:
        print(f"\nBest venue: {best.venue.value}")
        print(f"  Net profit: {best.net_profit_bps:.2f} bps")
        print(f"  Fee ratio: {best.fee_ratio:.2%}")
    
    # Test cross-exchange arb
    optimizer.set_withdrawal_cost("BTC", 5.0)  # $5 withdrawal fee
    arb_profit = optimizer.calculate_cross_exchange_arb_profit(
        Venue.BINANCE, Venue.COINBASE, 25.0, "BTC", 50000
    )
    print(f"\nCross-exchange arb net profit: {arb_profit:.2f} bps")
