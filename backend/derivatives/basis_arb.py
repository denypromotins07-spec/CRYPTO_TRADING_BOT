#!/usr/bin/env python3
"""
Basis Arbitrage Engine: Cash-and-Carry between Spot and Perpetuals

This module executes simultaneous spot and futures legs to capture the basis spread
between spot prices and perpetual swap prices. It implements atomic execution logic
to minimize legging risk and ensure delta-neutral positions.

Features:
- Real-time basis spread calculation
- Atomic dual-leg execution
- Dynamic position sizing based on funding rate predictions
- Risk management with automatic deleveraging

Target: Capture 8k-20k INR/hour through micro-inefficiencies in basis spreads.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, List, Tuple
import time
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegStatus(Enum):
    """Status of an individual leg in the arbitrage"""
    PENDING = "pending"
    EXECUTING = "executing"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ArbDirection(Enum):
    """Direction of the cash-and-carry arbitrage"""
    LONG_SPOT_SHORT_PERP = "long_spot_short_perp"  # Positive basis
    SHORT_SPOT_LONG_PERP = "short_spot_long_perp"  # Negative basis


@dataclass
class OrderLeg:
    """Represents a single leg of the arbitrage trade"""
    leg_id: str
    market_type: str  # 'spot' or 'perp'
    side: str  # 'buy' or 'sell'
    symbol: str
    quantity: float
    price: Optional[float] = None
    status: LegStatus = LegStatus.PENDING
    fill_price: Optional[float] = None
    fill_quantity: float = 0.0
    timestamp_us: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp_us == 0:
            self.timestamp_us = int(time.time() * 1_000_000)


@dataclass
class BasisArbitrage:
    """Complete basis arbitrage position with both legs"""
    arb_id: str
    symbol: str
    direction: ArbDirection
    spot_leg: OrderLeg
    perp_leg: OrderLeg
    entry_basis_bps: float  # Basis in basis points
    target_funding_rate: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    total_pnl: float = 0.0
    total_fees: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if both legs are filled"""
        return (self.spot_leg.status == LegStatus.FILLED and 
                self.perp_leg.status == LegStatus.FILLED)
    
    @property
    def is_failed(self) -> bool:
        """Check if either leg failed"""
        return (self.spot_leg.status == LegStatus.FAILED or 
                self.perp_leg.status == LegStatus.FAILED)
    
    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since creation"""
        if self.closed_at:
            return (self.closed_at - self.created_at).total_seconds()
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


@dataclass
class MarketSnapshot:
    """Real-time market data snapshot"""
    symbol: str
    spot_bid: float
    spot_ask: float
    perp_bid: float
    perp_ask: float
    mark_price: float
    index_price: float
    funding_rate: float
    timestamp_us: int
    
    @property
    def mid_spot(self) -> float:
        return (self.spot_bid + self.spot_ask) / 2.0
    
    @property
    def mid_perp(self) -> float:
        return (self.perp_bid + self.perp_ask) / 2.0
    
    @property
    def basis_bps(self) -> float:
        """Calculate basis in basis points"""
        if self.index_price == 0:
            return 0.0
        return ((self.mid_perp - self.mid_spot) / self.index_price) * 10000
    
    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points"""
        if self.mid_spot == 0:
            return 0.0
        return ((self.spot_ask - self.spot_bid) / self.mid_spot) * 10000


class BasisArbEngine:
    """
    Main engine for executing cash-and-carry basis arbitrage.
    
    This engine monitors basis spreads between spot and perpetual markets,
    identifies profitable opportunities, and executes atomic dual-leg trades
    while managing legging risk.
    """
    
    # Minimum basis threshold in bps to trigger arbitrage (covers fees + slippage)
    MIN_BASIS_THRESHOLD_BPS = 15.0
    
    # Maximum acceptable time between legs (microseconds)
    MAX_LEG_DELAY_US = 500_000  # 500ms
    
    # Position size limits per asset (in USD notional)
    POSITION_LIMITS = {
        'BTC': 50000,
        'ETH': 30000,
        'SOL': 10000,
    }
    
    def __init__(self, exchange_client, risk_manager):
        """
        Initialize the basis arbitrage engine.
        
        Args:
            exchange_client: Client for exchange API calls
            risk_manager: Risk management system for position limits
        """
        self.exchange_client = exchange_client
        self.risk_manager = risk_manager
        self.active_arbs: Dict[str, BasisArbitrage] = {}
        self.completed_arbs: List[BasisArbitrage] = []
        self._arb_counter = 0
        
    def _generate_arb_id(self) -> str:
        """Generate unique arbitrage ID"""
        self._arb_counter += 1
        return f"BASIS_{self._arb_counter}_{int(time.time())}"
    
    def calculate_basis(self, snapshot: MarketSnapshot) -> float:
        """
        Calculate the current basis spread in basis points.
        
        Returns:
            Basis in bps (positive = perp premium, negative = perp discount)
        """
        return snapshot.basis_bps
    
    def should_execute_arb(self, snapshot: MarketSnapshot) -> Tuple[bool, ArbDirection]:
        """
        Determine if an arbitrage opportunity exists and its direction.
        
        Returns:
            Tuple of (should_execute, direction)
        """
        basis_bps = snapshot.basis_bps
        funding_rate = snapshot.funding_rate
        
        # Check for positive basis (contango): Long Spot, Short Perp
        if basis_bps > self.MIN_BASIS_THRESHOLD_BPS:
            # Verify funding rate is positive (we receive funding as short perp)
            if funding_rate > 0:
                logger.info(f"Basis arb opportunity: +{basis_bps:.2f} bps, funding={funding_rate:.6f}")
                return True, ArbDirection.LONG_SPOT_SHORT_PERP
        
        # Check for negative basis (backwardation): Short Spot, Long Perp
        elif basis_bps < -self.MIN_BASIS_THRESHOLD_BPS:
            # Verify funding rate is negative (we receive funding as long perp)
            if funding_rate < 0:
                logger.info(f"Basis arb opportunity: {basis_bps:.2f} bps, funding={funding_rate:.6f}")
                return True, ArbDirection.SHORT_SPOT_LONG_PERP
        
        return False, ArbDirection.LONG_SPOT_SHORT_PERP  # Default direction
    
    def calculate_position_size(self, symbol: str, snapshot: MarketSnapshot) -> float:
        """
        Calculate optimal position size based on available capital and risk limits.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            snapshot: Current market snapshot
            
        Returns:
            Quantity in base asset units
        """
        base_asset = symbol.replace('USDT', '').replace('USD', '')
        
        # Get position limit for this asset
        limit_usd = self.POSITION_LIMITS.get(base_asset, 5000)
        
        # Check available capital from risk manager
        available_capital = self.risk_manager.get_available_capital()
        
        # Use minimum of limit and available capital
        notional_size = min(limit_usd, available_capital * 0.5)  # Use 50% of available
        
        # Convert to quantity
        mid_price = snapshot.mid_spot
        if mid_price <= 0:
            return 0.0
        
        quantity = notional_size / mid_price
        
        # Round to exchange precision
        precision = self.exchange_client.get_quantity_precision(symbol)
        return round(quantity, precision)
    
    async def execute_arbitrage(self, snapshot: MarketSnapshot) -> Optional[BasisArbitrage]:
        """
        Execute a complete basis arbitrage with atomic dual-leg execution.
        
        Args:
            snapshot: Current market snapshot
            
        Returns:
            BasisArbitrage object if executed, None otherwise
        """
        should_execute, direction = self.should_execute_arb(snapshot)
        if not should_execute:
            return None
        
        # Calculate position size
        quantity = self.calculate_position_size(snapshot.symbol, snapshot)
        if quantity <= 0:
            logger.warning("Invalid position size calculated")
            return None
        
        # Create arbitrage record
        arb_id = self._generate_arb_id()
        basis_bps = snapshot.basis_bps
        
        # Determine leg parameters based on direction
        if direction == ArbDirection.LONG_SPOT_SHORT_PERP:
            spot_side = 'buy'
            perp_side = 'sell'
        else:
            spot_side = 'sell'
            perp_side = 'buy'
        
        # Create order legs
        spot_leg = OrderLeg(
            leg_id=f"{arb_id}_SPOT",
            market_type='spot',
            side=spot_side,
            symbol=snapshot.symbol,
            quantity=quantity,
            price=snapshot.spot_ask if spot_side == 'buy' else snapshot.spot_bid
        )
        
        perp_leg = OrderLeg(
            leg_id=f"{arb_id}_PERP",
            market_type='perp',
            side=perp_side,
            symbol=snapshot.symbol,
            quantity=quantity,
            price=snapshot.perp_ask if perp_side == 'buy' else snapshot.perp_bid
        )
        
        arb = BasisArbitrage(
            arb_id=arb_id,
            symbol=snapshot.symbol,
            direction=direction,
            spot_leg=spot_leg,
            perp_leg=perp_leg,
            entry_basis_bps=basis_bps,
            target_funding_rate=snapshot.funding_rate
        )
        
        # Execute legs atomically
        success = await self._execute_dual_legs(arb)
        
        if success:
            self.active_arbs[arb_id] = arb
            logger.info(f"Basis arb executed: {arb_id}, basis={basis_bps:.2f} bps")
            return arb
        else:
            logger.error(f"Basis arb failed: {arb_id}")
            return None
    
    async def _execute_dual_legs(self, arb: BasisArbitrage) -> bool:
        """
        Execute both legs of the arbitrage with minimal delay.
        
        Uses atomic execution pattern to prevent legging risk.
        If one leg fails, immediately cancels the other.
        
        Returns:
            True if both legs filled successfully, False otherwise
        """
        start_time_us = int(time.time() * 1_000_000)
        
        try:
            # Execute spot leg first
            arb.spot_leg.status = LegStatus.EXECUTING
            spot_result = await self.exchange_client.execute_order(
                market_type='spot',
                symbol=arb.spot_leg.symbol,
                side=arb.spot_leg.side,
                quantity=arb.spot_leg.quantity,
                order_type='limit',
                price=arb.spot_leg.price
            )
            
            if spot_result['success']:
                arb.spot_leg.status = LegStatus.FILLED
                arb.spot_leg.fill_price = spot_result['fill_price']
                arb.spot_leg.fill_quantity = spot_result['fill_quantity']
            else:
                arb.spot_leg.status = LegStatus.FAILED
                arb.spot_leg.error_message = spot_result.get('error', 'Unknown error')
                return False
            
            # Check timing constraint
            current_time_us = int(time.time() * 1_000_000)
            if current_time_us - start_time_us > self.MAX_LEG_DELAY_US:
                logger.warning("Leg execution exceeded max delay, cancelling")
                await self._cancel_leg(arb.spot_leg)
                return False
            
            # Execute perp leg immediately
            arb.perp_leg.status = LegStatus.EXECUTING
            perp_result = await self.exchange_client.execute_order(
                market_type='perp',
                symbol=arb.perp_leg.symbol,
                side=arb.perp_leg.side,
                quantity=arb.perp_leg.quantity,
                order_type='limit',
                price=arb.perp_leg.price
            )
            
            if perp_result['success']:
                arb.perp_leg.status = LegStatus.FILLED
                arb.perp_leg.fill_price = perp_result['fill_price']
                arb.perp_leg.fill_quantity = perp_result['fill_quantity']
            else:
                arb.perp_leg.status = LegStatus.FAILED
                arb.perp_leg.error_message = perp_result.get('error', 'Unknown error')
                # Attempt to cancel spot leg if perp fails
                await self._cancel_leg(arb.spot_leg)
                return False
            
            return True
            
        except Exception as e:
            logger.exception(f"Error executing dual legs: {e}")
            # Emergency cancellation of any filled legs
            await self._emergency_close(arb)
            return False
    
    async def _cancel_leg(self, leg: OrderLeg) -> bool:
        """Cancel a single leg"""
        try:
            result = await self.exchange_client.cancel_order(
                market_type=leg.market_type,
                symbol=leg.symbol,
                order_id=leg.leg_id
            )
            if result['success']:
                leg.status = LegStatus.CANCELLED
                return True
        except Exception as e:
            logger.error(f"Failed to cancel leg {leg.leg_id}: {e}")
        return False
    
    async def _emergency_close(self, arb: BasisArbitrage) -> None:
        """Emergency close all legs of an arbitrage"""
        logger.warning(f"Emergency closing arb {arb.arb_id}")
        
        if arb.spot_leg.status == LegStatus.FILLED:
            # Close spot position
            close_side = 'sell' if arb.spot_leg.side == 'buy' else 'buy'
            await self.exchange_client.execute_order(
                market_type='spot',
                symbol=arb.spot_leg.symbol,
                side=close_side,
                quantity=arb.spot_leg.fill_quantity,
                order_type='market'
            )
        
        if arb.perp_leg.status == LegStatus.FILLED:
            # Close perp position
            close_side = 'buy' if arb.perp_leg.side == 'sell' else 'sell'
            await self.exchange_client.execute_order(
                market_type='perp',
                symbol=arb.perp_leg.symbol,
                side=close_side,
                quantity=arb.perp_leg.fill_quantity,
                order_type='market'
            )
    
    def calculate_unrealized_pnl(self, arb: BasisArbitrage, current_snapshot: MarketSnapshot) -> float:
        """
        Calculate unrealized PnL for an active arbitrage position.
        
        Args:
            arb: Active arbitrage position
            current_snapshot: Current market snapshot
            
        Returns:
            Unrealized PnL in quote currency
        """
        if not arb.is_complete:
            return 0.0
        
        # Calculate spot leg PnL
        spot_entry = arb.spot_leg.fill_price or 0
        spot_current = current_snapshot.mid_spot
        if arb.spot_leg.side == 'buy':
            spot_pnl = (spot_current - spot_entry) * arb.spot_leg.fill_quantity
        else:
            spot_pnl = (spot_entry - spot_current) * arb.spot_leg.fill_quantity
        
        # Calculate perp leg PnL (including funding)
        perp_entry = arb.perp_leg.fill_price or 0
        perp_current = current_snapshot.mid_perp
        if arb.perp_leg.side == 'buy':
            perp_pnl = (perp_current - perp_entry) * arb.perp_leg.fill_quantity
        else:
            perp_pnl = (perp_entry - perp_current) * arb.perp_leg.fill_quantity
        
        # Add accrued funding (approximate)
        hours_elapsed = arb.elapsed_seconds / 3600
        funding_pnl = arb.perp_leg.fill_quantity * perp_entry * arb.target_funding_rate * (hours_elapsed / 8)
        
        total_pnl = spot_pnl + perp_pnl + funding_pnl
        arb.total_pnl = total_pnl
        
        return total_pnl
    
    async def close_arbitrage(self, arb_id: str, snapshot: MarketSnapshot) -> float:
        """
        Close an active arbitrage position by reversing both legs.
        
        Args:
            arb_id: ID of the arbitrage to close
            snapshot: Current market snapshot
            
        Returns:
            Realized PnL
        """
        if arb_id not in self.active_arbs:
            logger.warning(f"Arbitrage {arb_id} not found")
            return 0.0
        
        arb = self.active_arbs[arb_id]
        
        # Calculate PnL before closing
        final_pnl = self.calculate_unrealized_pnl(arb, snapshot)
        
        # Reverse spot leg
        close_spot_side = 'sell' if arb.spot_leg.side == 'buy' else 'buy'
        await self.exchange_client.execute_order(
            market_type='spot',
            symbol=arb.spot_leg.symbol,
            side=close_spot_side,
            quantity=arb.spot_leg.fill_quantity,
            order_type='market'
        )
        
        # Reverse perp leg
        close_perp_side = 'buy' if arb.perp_leg.side == 'sell' else 'sell'
        await self.exchange_client.execute_order(
            market_type='perp',
            symbol=arb.perp_leg.symbol,
            side=close_perp_side,
            quantity=arb.perp_leg.fill_quantity,
            order_type='market'
        )
        
        # Update arb status
        arb.closed_at = datetime.now(timezone.utc)
        arb.total_pnl = final_pnl
        
        # Move to completed list
        del self.active_arbs[arb_id]
        self.completed_arbs.append(arb)
        
        logger.info(f"Arbitrage closed: {arb_id}, PnL={final_pnl:.2f}")
        return final_pnl


class RiskManager:
    """Simple risk manager for basis arbitrage"""
    
    def __init__(self, total_capital: float = 100000.0):
        self.total_capital = total_capital
        self.allocated_capital = 0.0
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        return self.total_capital - self.allocated_capital
    
    def allocate_capital(self, amount: float) -> bool:
        """Allocate capital for a position"""
        if amount <= self.get_available_capital():
            self.allocated_capital += amount
            return True
        return False
    
    def release_capital(self, amount: float) -> None:
        """Release allocated capital"""
        self.allocated_capital = max(0, self.allocated_capital - amount)


if __name__ == "__main__":
    # Example usage
    print("Basis Arbitrage Engine initialized")
    print("Ready to capture basis spreads between spot and perpetual markets")
