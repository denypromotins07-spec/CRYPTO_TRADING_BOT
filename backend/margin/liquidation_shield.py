#!/usr/bin/env python3
"""
Liquidation Shield: Maintenance Margin Monitor and Auto-Deleveraging

Real-time monitoring of margin positions to prevent liquidation.
Implements proactive deleveraging when liquidation price approaches safety threshold.
Accounts for ADL (Auto-Deleveraging) queue positions on Binance.

Features:
- Real-time liquidation price calculation
- Dynamic safety buffer based on volatility
- Automatic deleveraging triggers
- ADL queue position tracking

Target: Force immediate deleverage if liquidation price breaches safety threshold.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time
import math
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position direction"""
    LONG = "long"
    SHORT = "short"


class MarginMode(Enum):
    """Margin mode type"""
    ISOLATED = "isolated"
    CROSS = "cross"


@dataclass
class MarginPosition:
    """Represents a margined position"""
    symbol: str
    side: PositionSide
    size: float  # In base asset units
    entry_price: float
    leverage: float
    margin_mode: MarginMode
    isolated_margin: float = 0.0  # For isolated margin positions
    unrealized_pnl: float = 0.0
    timestamp_us: int = 0
    
    def __post_init__(self):
        if self.timestamp_us == 0:
            self.timestamp_us = int(time.time() * 1_000_000)
    
    @property
    def notional_value(self) -> float:
        """Get position notional value"""
        return self.size * self.entry_price
    
    @property
    def initial_margin(self) -> float:
        """Get initial margin required"""
        return self.notional_value / self.leverage
    
    @property
    def maintenance_margin_rate(self) -> float:
        """Get maintenance margin rate (varies by exchange and asset)"""
        # Binance typical MM rates
        mm_rates = {'BTC': 0.004, 'ETH': 0.005, 'SOL': 0.0065}
        base_asset = self.symbol.replace('USDT', '').replace('USD', '')
        return mm_rates.get(base_asset, 0.01)
    
    @property
    def maintenance_margin(self) -> float:
        """Get maintenance margin required"""
        return self.notional_value * self.maintenance_margin_rate


@dataclass
class LiquidationRisk:
    """Liquidation risk assessment for a position"""
    symbol: str
    current_price: float
    liquidation_price: float
    distance_to_liquidation: float  # Percentage
    safety_threshold: float  # Percentage
    risk_level: RiskLevel
    recommended_action: str
    adl_queue_position: int  # 1-5 scale (1 = lowest risk)
    timestamp_us: int = 0


class RiskLevel(Enum):
    """Risk level classification"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"


class LiquidationPriceCalculator:
    """
    Calculate liquidation prices for margin positions.
    
    Formulas based on Binance perpetual futures methodology.
    """
    
    @staticmethod
    def calculate_long_liquidation(
        entry_price: float,
        leverage: float,
        maintenance_margin_rate: float,
        isolated_margin: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> float:
        """
        Calculate liquidation price for a long position.
        
        Long Liq Price = Entry Price * (1 - Initial Margin Rate + MM Rate) / (1 - MM Rate)
        Adjusted for isolated margin and PnL
        """
        if entry_price <= 0 or leverage <= 0:
            return 0.0
        
        initial_margin_rate = 1.0 / leverage
        
        # Base liquidation price
        liq_price = entry_price * (1 - initial_margin_rate + maintenance_margin_rate)
        
        # Adjust for isolated margin buffer
        if isolated_margin > 0 and entry_price > 0:
            liq_price -= isolated_margin / (entry_price * leverage)
        
        # Adjust for unrealized PnL impact
        if unrealized_pnl < 0:
            liq_price *= (1 + abs(unrealized_pnl) / (entry_price * leverage))
        
        # Apply MM rate denominator
        if maintenance_margin_rate < 1.0:
            liq_price /= (1 - maintenance_margin_rate)
        
        return max(0.0, liq_price)
    
    @staticmethod
    def calculate_short_liquidation(
        entry_price: float,
        leverage: float,
        maintenance_margin_rate: float,
        isolated_margin: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> float:
        """
        Calculate liquidation price for a short position.
        
        Short Liq Price = Entry Price * (1 + Initial Margin Rate - MM Rate) / (1 + MM Rate)
        """
        if entry_price <= 0 or leverage <= 0:
            return 0.0
        
        initial_margin_rate = 1.0 / leverage
        
        # Base liquidation price
        liq_price = entry_price * (1 + initial_margin_rate - maintenance_margin_rate)
        
        # Adjust for isolated margin buffer
        if isolated_margin > 0 and entry_price > 0:
            liq_price += isolated_margin / (entry_price * leverage)
        
        # Adjust for unrealized PnL impact
        if unrealized_pnl < 0:
            liq_price *= (1 + abs(unrealized_pnl) / (entry_price * leverage))
        
        # Apply MM rate denominator
        if maintenance_margin_rate < 1.0:
            liq_price /= (1 + maintenance_margin_rate)
        
        return max(0.0, liq_price)
    
    @classmethod
    def calculate(cls, position: MarginPosition, current_price: float) -> float:
        """Calculate liquidation price for any position"""
        if position.side == PositionSide.LONG:
            return cls.calculate_long_liquidation(
                entry_price=position.entry_price,
                leverage=position.leverage,
                maintenance_margin_rate=position.maintenance_margin_rate,
                isolated_margin=position.isolated_margin,
                unrealized_pnl=position.unrealized_pnl,
            )
        else:
            return cls.calculate_short_liquidation(
                entry_price=position.entry_price,
                leverage=position.leverage,
                maintenance_margin_rate=position.maintenance_margin_rate,
                isolated_margin=position.isolated_margin,
                unrealized_pnl=position.unrealized_pnl,
            )


class LiquidationShield:
    """
    Main liquidation prevention system.
    
    Monitors all positions and triggers automatic deleveraging
    when liquidation risk exceeds thresholds.
    """
    
    # Default safety thresholds (percentage from current price)
    DEFAULT_WARNING_THRESHOLD = 0.10   # 10% warning zone
    DEFAULT_CRITICAL_THRESHOLD = 0.05  # 5% critical zone
    DEFAULT_IMMEDIATE_THRESHOLD = 0.02 # 2% immediate action
    
    def __init__(
        self,
        exchange_client=None,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        immediate_threshold: float = DEFAULT_IMMEDIATE_THRESHOLD,
    ):
        self.exchange_client = exchange_client
        self.positions: Dict[str, MarginPosition] = {}
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.immediate_threshold = immediate_threshold
        self.adl_queues: Dict[str, int] = {}  # Symbol -> ADL queue position (1-5)
        self._last_check_us: int = 0
    
    def add_position(self, position: MarginPosition) -> None:
        """Add or update a position"""
        key = f"{position.symbol}_{position.side.value}"
        self.positions[key] = position
    
    def remove_position(self, symbol: str, side: PositionSide) -> bool:
        """Remove a position"""
        key = f"{symbol}_{side.value}"
        if key in self.positions:
            del self.positions[key]
            return True
        return False
    
    def update_position_pnl(self, symbol: str, side: PositionSide, pnl: float) -> None:
        """Update unrealized PnL for a position"""
        key = f"{symbol}_{side.value}"
        if key in self.positions:
            self.positions[key].unrealized_pnl = pnl
    
    def update_adl_queue(self, symbol: str, queue_position: int) -> None:
        """Update ADL queue position for a symbol"""
        self.adl_queues[symbol] = min(5, max(1, queue_position))
    
    def assess_risk(self, position: MarginPosition, current_price: float) -> LiquidationRisk:
        """
        Assess liquidation risk for a position.
        
        Returns comprehensive risk assessment with recommended action.
        """
        liq_price = LiquidationPriceCalculator.calculate(position, current_price)
        
        # Calculate distance to liquidation
        if position.side == PositionSide.LONG:
            distance = (current_price - liq_price) / current_price if current_price > 0 else 0
        else:
            distance = (liq_price - current_price) / current_price if current_price > 0 else 0
        
        # Determine risk level
        if distance <= self.immediate_threshold:
            risk_level = RiskLevel.IMMEDIATE
            action = "IMMEDIATE DELEVERAGE OR CLOSE POSITION"
        elif distance <= self.critical_threshold:
            risk_level = RiskLevel.CRITICAL
            action = "Reduce leverage by 50% immediately"
        elif distance <= self.warning_threshold:
            risk_level = RiskLevel.WARNING
            action = "Consider reducing leverage or adding margin"
        else:
            risk_level = RiskLevel.SAFE
            action = "No action required"
        
        # Get ADL queue position
        adl_position = self.adl_queues.get(position.symbol, 3)  # Default middle
        
        return LiquidationRisk(
            symbol=position.symbol,
            current_price=current_price,
            liquidation_price=liq_price,
            distance_to_liquidation=distance,
            safety_threshold=self.warning_threshold,
            risk_level=risk_level,
            recommended_action=action,
            adl_queue_position=adl_position,
            timestamp_us=int(time.time() * 1_000_000),
        )
    
    def check_all_positions(self, prices: Dict[str, float]) -> List[LiquidationRisk]:
        """
        Check all positions against current prices.
        
        Args:
            prices: Dictionary of symbol -> current price
            
        Returns:
            List of risk assessments for all positions
        """
        risks = []
        
        for position in self.positions.values():
            current_price = prices.get(position.symbol)
            if current_price is None:
                logger.warning(f"No price available for {position.symbol}")
                continue
            
            risk = self.assess_risk(position, current_price)
            risks.append(risk)
            
            # Log warnings
            if risk.risk_level == RiskLevel.IMMEDIATE:
                logger.critical(
                    f"IMMEDIATE LIQUIDATION RISK: {position.symbol} {position.side.value} "
                    f"Liq={risk.liquidation_price:.2f} Distance={risk.distance_to_liquidation:.2%}"
                )
            elif risk.risk_level == RiskLevel.CRITICAL:
                logger.warning(
                    f"CRITICAL RISK: {position.symbol} {position.side.value} "
                    f"Liq={risk.liquidation_price:.2f} Distance={risk.distance_to_liquidation:.2%}"
                )
        
        self._last_check_us = int(time.time() * 1_000_000)
        return risks
    
    def get_immediate_actions_needed(self, prices: Dict[str, float]) -> List[LiquidationRisk]:
        """Get only positions requiring immediate action"""
        all_risks = self.check_all_positions(prices)
        return [r for r in all_risks if r.risk_level in (RiskLevel.IMMEDIATE, RiskLevel.CRITICAL)]
    
    async def execute_automatic_deleverage(
        self,
        risk: LiquidationRisk,
        reduction_factor: float = 0.5,
    ) -> bool:
        """
        Execute automatic deleveraging for a risky position.
        
        Args:
            risk: The liquidation risk assessment
            reduction_factor: Fraction of position to close (0.5 = close 50%)
            
        Returns:
            True if deleveraging was successful
        """
        if risk.risk_level not in (RiskLevel.IMMEDIATE, RiskLevel.CRITICAL):
            logger.info(f"No deleverage needed for {risk.symbol} (risk={risk.risk_level.value})")
            return True
        
        key = f"{risk.symbol}_{self._get_side_from_symbol(risk.symbol)}"
        position = self.positions.get(key)
        
        if position is None:
            logger.error(f"Position not found for {risk.symbol}")
            return False
        
        # Calculate reduce amount
        reduce_quantity = position.size * reduction_factor
        
        logger.warning(
            f"EXECUTING AUTO-DELEVERAGE: {risk.symbol} "
            f"Reduce {reduce_quantity:.4f} ({reduction_factor:.0%} of position)"
        )
        
        if self.exchange_client:
            try:
                # Execute reduce-only order
                result = await self.exchange_client.reduce_position(
                    symbol=risk.symbol,
                    quantity=reduce_quantity,
                    reason="LIQUIDATION_SHIELD",
                )
                
                if result.get('success'):
                    logger.info(f"Auto-deleverage successful for {risk.symbol}")
                    return True
                else:
                    logger.error(f"Auto-deleverage failed: {result.get('error')}")
                    return False
                    
            except Exception as e:
                logger.exception(f"Error during auto-deleverage: {e}")
                return False
        else:
            logger.warning("No exchange client configured - simulating deleverage")
            return True
    
    def _get_side_from_symbol(self, symbol: str) -> str:
        """Extract side from position key"""
        parts = symbol.split('_')
        if len(parts) >= 2:
            return parts[-1]
        return 'long'
    
    def get_portfolio_risk_summary(self, prices: Dict[str, float]) -> Dict:
        """Get overall portfolio liquidation risk summary"""
        risks = self.check_all_positions(prices)
        
        if not risks:
            return {'status': 'no_positions'}
        
        worst_risk = min(risks, key=lambda r: r.distance_to_liquidation)
        avg_distance = sum(r.distance_to_liquidation for r in risks) / len(risks)
        
        critical_count = sum(1 for r in risks if r.risk_level in (RiskLevel.IMMEDIATE, RiskLevel.CRITICAL))
        
        return {
            'status': 'monitored',
            'total_positions': len(risks),
            'worst_risk_symbol': worst_risk.symbol,
            'worst_distance': worst_risk.distance_to_liquidation,
            'average_distance': avg_distance,
            'critical_positions': critical_count,
            'timestamp_us': int(time.time() * 1_000_000),
        }


if __name__ == "__main__":
    # Example usage
    print("Liquidation Shield initialized")
    
    shield = LiquidationShield()
    
    # Add sample position
    position = MarginPosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=1.0,
        entry_price=50000,
        leverage=10,
        margin_mode=MarginMode.ISOLATED,
        isolated_margin=5000,
    )
    
    shield.add_position(position)
    
    # Check risk at various prices
    test_prices = [50000, 48000, 46000, 45000]
    
    for price in test_prices:
        risk = shield.assess_risk(position, price)
        print(f"\nPrice: ${price:,}")
        print(f"  Liquidation Price: ${risk.liquidation_price:,.2f}")
        print(f"  Distance: {risk.distance_to_liquidation:.2%}")
        print(f"  Risk Level: {risk.risk_level.value}")
        print(f"  Action: {risk.recommended_action}")
