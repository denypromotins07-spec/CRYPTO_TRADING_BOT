#!/usr/bin/env python3
"""
Volatility Soul Logger for SOUL.md
Logs volatility surface arbitrage opportunities and calibration errors
Updates SOUL.md when the bot successfully arbitrages mispriced volatility skew
Optimized for AMD Ryzen AI 5 with strict 8GB RAM constraints
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os


@dataclass
class VolEvent:
    """Base class for volatility events"""
    timestamp: str
    event_type: str
    asset: str  # BTC, ETH, SOL
    details: Dict[str, Any]


@dataclass
class ArbitrageEvent(VolEvent):
    """Volatility surface arbitrage opportunity detected"""
    arbitrage_type: str  # 'butterfly', 'calendar', 'skew'
    expected_profit: float
    confidence: float
    strikes_involved: List[float]
    expiry_involved: List[float]


@dataclass
class CalibrationEvent(VolEvent):
    """Model calibration result or error"""
    model_name: str  # 'heston', 'sabr', 'local_vol'
    calibration_error: float
    parameters: Dict[str, float]
    success: bool
    feller_satisfied: Optional[bool] = None


@dataclass
class GammaScalpEvent(VolEvent):
    """Gamma scalping profit/loss event"""
    pnl: float
    gamma_exposure: float
    underlying_move: float
    hedge_count: int


class VolSoulLogger:
    """
    Logs volatility-related events to SOUL.md
    Tracks arbitrage successes, calibration quality, and gamma profits
    """
    
    def __init__(self, soul_md_path: str = "SOUL.md"):
        self.soul_md_path = soul_md_path
        self.event_buffer: List[VolEvent] = []
        self.buffer_size = 100  # Flush after this many events
        self._ensure_soul_file_exists()
    
    def _ensure_soul_file_exists(self) -> None:
        """Create SOUL.md if it doesn't exist"""
        if not os.path.exists(self.soul_md_path):
            with open(self.soul_md_path, 'w') as f:
                f.write("# ZAID BOT - SOUL.md\n\n")
                f.write("## Volatility Trading Log\n\n")
                f.write("*This file tracks volatility arbitrage, calibration events, and gamma scalping.*\n\n")
                f.write("---\n\n")
    
    def log_arbitrage(self, asset: str, arb_type: str, expected_profit: float,
                      confidence: float, strikes: List[float], 
                      expiries: List[float]) -> None:
        """
        Log a volatility arbitrage opportunity
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL)
            arb_type: Type of arbitrage ('butterfly', 'calendar', 'skew')
            expected_profit: Expected P&L from the arb
            confidence: Confidence score [0, 1]
            strikes: Strikes involved in the arb
            expiries: Expiries involved in the arb
        """
        event = ArbitrageEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="ARBITRAGE",
            asset=asset,
            details={
                'arbitrage_type': arb_type,
                'expected_profit': expected_profit,
                'confidence': confidence,
                'strikes_involved': strikes,
                'expiry_involved': expiries,
                'status': 'detected'
            }
        )
        
        self.event_buffer.append(event)
        self._check_flush()
        
        # Immediate write for high-confidence arbs
        if confidence > 0.8 and expected_profit > 100:
            self._write_to_soul(event, immediate=True)
    
    def log_calibration(self, asset: str, model_name: str, 
                        calibration_error: float, parameters: Dict[str, float],
                        success: bool, feller_satisfied: Optional[bool] = None) -> None:
        """
        Log model calibration result
        
        Args:
            asset: Asset symbol
            model_name: Model used ('heston', 'sabr', 'local_vol')
            calibration_error: RMSE of calibration
            parameters: Calibrated parameters
            success: Whether calibration converged
            feller_satisfied: Whether Feller condition is satisfied (Heston only)
        """
        event = CalibrationEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="CALIBRATION",
            asset=asset,
            details={
                'model_name': model_name,
                'calibration_error': calibration_error,
                'parameters': parameters,
                'success': success,
                'feller_satisfied': feller_satisfied,
                'status': 'success' if success else 'failed'
            }
        )
        
        self.event_buffer.append(event)
        self._check_flush()
    
    def log_gamma_scalp(self, asset: str, pnl: float, gamma_exposure: float,
                        underlying_move: float, hedge_count: int) -> None:
        """
        Log gamma scalping result
        
        Args:
            asset: Asset symbol
            pnl: P&L from the scalp
            gamma_exposure: Gamma exposure during the trade
            underlying_move: Size of underlying move
            hedge_count: Number of delta hedges executed
        """
        event = GammaScalpEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="GAMMA_SCALP",
            asset=asset,
            details={
                'pnl': pnl,
                'gamma_exposure': gamma_exposure,
                'underlying_move': underlying_move,
                'hedge_count': hedge_count,
                'efficiency': pnl / abs(gamma_exposure * underlying_move) if gamma_exposure != 0 else 0
            }
        )
        
        self.event_buffer.append(event)
        self._check_flush()
        
        # Log profitable gamma scalps immediately
        if pnl > 50:
            self._write_to_soul(event, immediate=True)
    
    def log_correlation_breakdown(self, asset: str, breakdown_type: str,
                                   impact_on_hedge: float) -> None:
        """
        Log correlation breakdown affecting hedging
        
        This updates SOUL.md when correlation breakdown causes temporary
        directional exposure despite delta-neutral intent.
        """
        event = VolEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="CORRELATION_BREAKDOWN",
            asset=asset,
            details={
                'breakdown_type': breakdown_type,
                'impact_on_hedge': impact_on_hedge,
                'directional_exposure_created': True,
                'mitigation': 'rebalancing_triggered'
            }
        )
        
        self.event_buffer.append(event)
        self._write_to_soul(event, immediate=True)
    
    def _check_flush(self) -> None:
        """Flush buffer if it exceeds threshold"""
        if len(self.event_buffer) >= self.buffer_size:
            self._flush_to_soul()
    
    def _flush_to_soul(self) -> None:
        """Write all buffered events to SOUL.md"""
        with open(self.soul_md_path, 'a') as f:
            for event in self.event_buffer:
                self._write_event(f, event)
        self.event_buffer.clear()
    
    def _write_to_soul(self, event: VolEvent, immediate: bool = False) -> None:
        """Write single event to SOUL.md"""
        with open(self.soul_md_path, 'a') as f:
            self._write_event(f, event)
    
    def _write_event(self, f, event: VolEvent) -> None:
        """Format and write event to file"""
        f.write(f"\n### [{event.timestamp}] {event.event_type} - {event.asset}\n\n")
        
        if isinstance(event, ArbitrageEvent):
            f.write(f"**Arbitrage Type:** {event.arbitrage_type}\n\n")
            f.write(f"**Expected Profit:** ${event.expected_profit:.2f}\n\n")
            f.write(f"**Confidence:** {event.confidence:.1%}\n\n")
            f.write(f"**Strikes:** {event.strikes_involved}\n\n")
            f.write(f"**Expiries:** {event.expiry_involved}\n\n")
            
            # Highlight successful arbs
            if event.confidence > 0.8:
                f.write("> ✅ **HIGH CONFIDENCE ARBITRAGE DETECTED**\n\n")
        
        elif isinstance(event, CalibrationEvent):
            f.write(f"**Model:** {event.model_name}\n\n")
            f.write(f"**Status:** {'✅ Success' if event.success else '❌ Failed'}\n\n")
            f.write(f"**Calibration Error:** {event.calibration_error:.6f}\n\n")
            
            if event.parameters:
                f.write("**Parameters:**\n")
                for k, v in event.parameters.items():
                    f.write(f"- {k}: {v:.4f}\n")
                f.write("\n")
            
            if event.feller_satisfied is not None:
                f.write(f"**Feller Condition:** {'✅ Satisfied' if event.feller_satisfied else '⚠️ Violated'}\n\n")
        
        elif isinstance(event, GammaScalpEvent):
            f.write(f"**P&L:** ${event.pnl:.2f}\n\n")
            f.write(f"**Gamma Exposure:** {event.gamma_exposure:.4f}\n\n")
            f.write(f"**Underlying Move:** {event.underlying_move:.2%}\n\n")
            f.write(f"**Hedges Executed:** {event.hedge_count}\n\n")
            f.write(f"**Efficiency:** {event.efficiency:.2%}\n\n")
            
            if event.pnl > 100:
                f.write("> 🎯 **SUCCESSFUL GAMMA SCALP - Micro-reversal captured!**\n\n")
        
        elif event.event_type == "CORRELATION_BREAKDOWN":
            f.write(f"**Breakdown Type:** {event.breakdown_type}\n\n")
            f.write(f"**Hedge Impact:** {event.impact_on_hedge:.2%}\n\n")
            f.write("> ⚠️ **CORRELATION BREAKDOWN - Temporary directional exposure created**\n\n")
            f.write(f"**Mitigation:** {event.details.get('mitigation', 'manual_review')}\n\n")
        
        f.write("---\n")


def create_vol_logger(soul_md_path: str = "SOUL.md") -> VolSoulLogger:
    """Factory function to create a configured logger"""
    return VolSoulLogger(soul_md_path)


if __name__ == '__main__':
    # Demo usage
    logger = create_vol_logger()
    
    print("Logging volatility events to SOUL.md...")
    
    # Log an arbitrage opportunity
    logger.log_arbitrage(
        asset='BTC',
        arb_type='butterfly',
        expected_profit=250.0,
        confidence=0.85,
        strikes=[44000, 45000, 46000],
        expiries=[0.082]
    )
    
    # Log calibration
    logger.log_calibration(
        asset='ETH',
        model_name='heston',
        calibration_error=0.0023,
        parameters={'v0': 0.04, 'theta': 0.04, 'kappa': 2.0, 'xi': 0.3, 'rho': -0.7},
        success=True,
        feller_satisfied=True
    )
    
    # Log gamma scalp
    logger.log_gamma_scalp(
        asset='SOL',
        pnl=125.50,
        gamma_exposure=0.002,
        underlying_move=0.03,
        hedge_count=5
    )
    
    # Log correlation breakdown
    logger.log_correlation_breakdown(
        asset='BTC',
        breakdown_type='spot_perp_decoupling',
        impact_on_hedge=0.15
    )
    
    print(f"Events logged to {logger.soul_md_path}")
